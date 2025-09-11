from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import json
import re
from datetime import datetime, timezone
from .config import cfg
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .types import RetrievalQuery, Candidate
# Attention mechanisms removed - using fixed weight comprehensive scoring only

def exp_recency(ts_iso: str, now: datetime, half_life_hours: float = 72.0) -> float:
    try:
        ts = datetime.fromisoformat(ts_iso.replace('Z','+00:00'))
        # Ensure both datetimes are timezone-aware or both are naive
        if ts.tzinfo is not None and now.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        elif ts.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
    except Exception:
        return 0.5
    dt = (now - ts).total_seconds() / 3600.0
    # Exponential decay, 0..1
    return max(0.0, min(1.0, 0.5 ** (dt / half_life_hours)))

class HybridRetriever:
    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        
        # No attention mechanisms - using comprehensive scoring with fixed weights

    def _semantic(self, qvec: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        results = self.index.search(qvec, topk)
        # FAISS with METRIC_INNER_PRODUCT returns cosine similarity scores in [-1, 1]
        # where 1 is perfect match, 0 is orthogonal, -1 is opposite
        # We'll map this to [0, 1] range preserving the absolute similarity
        if not results:
            return []
        
        normalized = []
        for mid, score in results:
            # Map from [-1, 1] to [0, 1]
            # This preserves the actual similarity: 
            # - Perfect match (1.0) stays 1.0
            # - No similarity (0.0) becomes 0.5
            # - Opposite (-1.0) becomes 0.0
            norm_score = (score + 1.0) / 2.0
            # Clip to ensure we're in valid range (in case of numerical errors)
            norm_score = max(0.0, min(1.0, norm_score))
            normalized.append((mid, float(norm_score)))
        
        return normalized

    def _lexical(self, query: str, topk: int) -> List[Tuple[str, float]]:
        rows = self.store.lexical_search(query, k=topk)
        # FTS5 bm25 returns lower is better; invert + normalize
        if not rows:
            return []
        scores = [r['score'] for r in rows]
        max_s = max(scores); min_s = min(scores)
        out = []
        for r in rows:
            s = r['score']
            if max_s == min_s:
                norm = 1.0
            else:
                norm = 1.0 - (s - min_s) / (max_s - min_s)
            out.append((r['memory_id'], float(norm)))
        return out
    
    def _actor_based(self, actor_id: str, topk: int) -> List[Tuple[str, float]]:
        """Retrieve memories from specific actor with recency-based scoring."""
        rows = self.store.get_by_actor(actor_id, limit=topk)
        if not rows:
            return []
        
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        results = []
        for row in rows:
            # Score based on recency - more recent memories get higher scores
            recency_score = exp_recency(row['when_ts'], now, half_life_hours=168.0)  # 1 week half-life
            results.append((row['memory_id'], recency_score))
        return results
    
    def _where_based(self, where_value: str, topk: int) -> List[Tuple[str, float]]:
        """Retrieve memories from specific WHERE location with recency-based scoring.
        
        This searches the where_value field in the 5W1H model.
        """
        rows = self.store.get_by_location(where_value, limit=topk)
        if not rows:
            return []
        
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        results = []
        for row in rows:
            recency_score = exp_recency(row['when_ts'], now, half_life_hours=168.0)
            results.append((row['memory_id'], recency_score))
        return results
    
    def _temporal_based(self, temporal_hint: Union[str, Tuple[str, str], Dict], topk: int) -> List[Tuple[str, float]]:
        """Retrieve memories based on temporal hint.
        
        Supports:
        - Single date: "2024-01-15"
        - Date range: ("2024-01-10", "2024-01-20")
        - Relative time: {"relative": "yesterday"}
        """
        from typing import Union, Tuple, Dict
        
        # Parse temporal hint and retrieve memories
        if isinstance(temporal_hint, str):
            # Single date or timestamp - extract date part if needed
            if 'T' in temporal_hint:
                # Full timestamp like "2025-09-07T16:08:33.917577" - extract date
                date_part = temporal_hint.split('T')[0]
            else:
                date_part = temporal_hint
            rows = self.store.get_by_date(date_part, limit=topk)
        elif isinstance(temporal_hint, tuple) and len(temporal_hint) == 2:
            # Date range
            start, end = temporal_hint
            rows = self.store.get_by_date_range(start, end, limit=topk)
        elif isinstance(temporal_hint, dict):
            if "relative" in temporal_hint:
                # Relative time like "yesterday", "last_week"
                rows = self.store.get_by_relative_time(temporal_hint["relative"])
            elif "start" in temporal_hint and "end" in temporal_hint:
                # Timestamp range - extract dates
                start = temporal_hint["start"].split("T")[0] if "T" in temporal_hint["start"] else temporal_hint["start"]
                end = temporal_hint["end"].split("T")[0] if "T" in temporal_hint["end"] else temporal_hint["end"]
                rows = self.store.get_by_date_range(start, end, limit=topk)
            else:
                return []
        else:
            return []
        
        if not rows:
            return []
        
        # Score based on being in temporal window
        # All memories in the window get high base score (0.8)
        # with slight variation based on exact time for ranking
        results = []
        for i, row in enumerate(rows):
            # Higher score for earlier results (they're already sorted by time)
            score = 0.8 - (i * 0.001)  # Small decay for ranking
            results.append((row['memory_id'], score))
        
        return results

    def compute_all_scores(self, 
                           memory_ids: List[str],
                           query_vec: np.ndarray,
                           query: RetrievalQuery,
                           sem_scores: Dict[str, float],
                           lex_scores: Dict[str, float],
                           actor_matches: Dict[str, float],
                           temporal_matches: Dict[str, float]) -> List[Candidate]:
        """Compute comprehensive scores for all candidates across all dimensions."""
        
        # Fetch metadata for all candidates
        if not memory_ids:
            return []
        
        memories = self.store.fetch_memories(memory_ids)
        memory_dict = {m['memory_id']: m for m in memories}
        
        # Get usage stats for all candidates
        usage_stats = self.store.get_usage_stats(memory_ids)
        
        # Current time for recency calculation
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Group memories by topic/entity for smart recency application
        memory_groups = self._group_related_memories(memories, sem_scores, lex_scores)
        
        # Calculate candidates with all dimension scores
        candidates = []
        
        for memory_id in memory_ids:
            memory = memory_dict.get(memory_id)
            if not memory:
                continue
            
            # 1. Semantic similarity (already computed)
            semantic_score = sem_scores.get(memory_id, 0.0)
            
            # 2. Lexical match (already computed)
            lexical_score = lex_scores.get(memory_id, 0.0)
            
            # 3. Smart recency score (only applies as tiebreaker for related memories)
            base_recency = exp_recency(memory['when_ts'], now, half_life_hours=168.0)
            recency_score = self._apply_smart_recency(memory_id, base_recency, memory_groups)
            
            # 4. Actor match (binary or from hint)
            actor_score = actor_matches.get(memory_id, 0.0)
            if not actor_score and query.actor_hint:
                # Check if memory actor matches query hint
                actor_score = 1.0 if memory['who_id'] == query.actor_hint else 0.0
            
            # 5. Temporal match (binary or from hint)
            temporal_score = temporal_matches.get(memory_id, 0.0)
            
            # 6. Spatial/location match (check if location mentioned in query)
            spatial_score = 0.0
            if 'where_value' in memory.keys() and memory['where_value']:
                # Simple check if location is mentioned in query
                if memory['where_value'].lower() in query.text.lower():
                    spatial_score = 1.0
            
            # Get usage data
            usage_data = usage_stats.get(memory_id, {})
            usage_score = min(1.0, usage_data.get('accesses', 0) / 50.0)  # Normalize by 50 accesses
            
            # Fixed weight combination 
            # Default weights (can be configured)
            # Note: recency is reduced since it's now a smart tiebreaker
            w_semantic = 0.45
            w_lexical = 0.25
            w_recency = 0.02  # Small weight - mainly acts as tiebreaker
            w_actor = 0.1
            w_temporal = 0.1
            w_spatial = 0.04
            w_usage = 0.04
            
            final_score = (
                w_semantic * semantic_score +
                w_lexical * lexical_score +
                w_recency * recency_score +
                w_actor * actor_score +
                w_temporal * temporal_score +
                w_spatial * spatial_score +
                w_usage * usage_score
            )
            
            candidate = Candidate(
                memory_id=memory_id,
                score=final_score,
                token_count=int(memory['token_count']),
                base_score=final_score,
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                recency_score=recency_score,
                actor_score=actor_score,
                temporal_score=temporal_score,
                spatial_score=spatial_score,
                usage_score=usage_score
            )
            
            candidates.append(candidate)
        
        return candidates

    
    def _group_related_memories(self, memories: List[Dict], sem_scores: Dict[str, float], lex_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Group memories that are about the same topic/entity for smart recency application.
        
        Memories are considered related if they have:
        1. High semantic similarity to each other (>0.8)
        2. Significant lexical overlap
        3. Same actor
        4. Overlapping key entities/topics
        """
        groups = {}  # group_id -> list of memory_ids
        memory_to_group = {}  # memory_id -> group_id
        
        # Sort memories by score to process highest scoring first
        sorted_memories = sorted(memories, key=lambda m: sem_scores.get(m['memory_id'], 0.0) + lex_scores.get(m['memory_id'], 0.0), reverse=True)
        
        for memory in sorted_memories:
            mid = memory['memory_id']
            assigned = False
            
            # Check if this memory is highly similar to any existing group
            for group_id, group_members in groups.items():
                # Check similarity with group representatives (first few members)
                for other_mid in group_members[:3]:  # Check against first 3 members
                    other_memory = next((m for m in memories if m['memory_id'] == other_mid), None)
                    if not other_memory:
                        continue
                    
                    # Check if memories are related
                    is_related = False
                    
                    # Get entities from the 'what' field (now stored as JSON array)
                    memory_entities = self._extract_memory_entities(memory)
                    other_entities = self._extract_memory_entities(other_memory)
                    
                    # Calculate overlap
                    entity_overlap = len(memory_entities & other_entities)
                    total_entities = len(memory_entities | other_entities)
                    
                    # Check different types of relatedness:
                    
                    # 1. Same actor discussing similar topic (updates/corrections)
                    if memory['who_id'] == other_memory['who_id'] and entity_overlap > 3:
                        is_related = True
                    
                    # 2. High overlap ratio (>40% of words in common) - likely same topic
                    elif total_entities > 0 and entity_overlap / total_entities > 0.4:
                        is_related = True
                    
                    # 3. Both memories highly relevant to query (top scorers discussing same thing)
                    elif (sem_scores.get(mid, 0) > 0.7 and sem_scores.get(other_mid, 0) > 0.7 and 
                          entity_overlap > 2):
                        is_related = True
                    
                    if is_related:
                        groups[group_id].append(mid)
                        memory_to_group[mid] = group_id
                        assigned = True
                        break
                
                if assigned:
                    break
            
            # If not assigned to any group, create new group
            if not assigned:
                group_id = f"group_{len(groups)}"
                groups[group_id] = [mid]
                memory_to_group[mid] = group_id
        
        return memory_to_group
    
    def _extract_memory_entities(self, memory: Dict) -> set:
        """Extract entities from a memory's 'what' field.
        
        The 'what' field can be:
        1. JSON array of entities (new format)
        2. JSON string containing array (stored format)
        3. Plain text (legacy format)
        """
        entities = set()
        
        if 'what' not in memory.keys() or not memory['what']:
            return entities
        
        what_field = memory['what']
        
        # Try to parse as JSON array
        try:
            # If it's already a list
            if isinstance(what_field, list):
                entities.update(str(e).lower() for e in what_field)
            # If it's a JSON string
            elif isinstance(what_field, str) and what_field.startswith('['):
                parsed = json.loads(what_field)
                if isinstance(parsed, list):
                    entities.update(str(e).lower() for e in parsed)
            # Legacy plain text format
            else:
                # Extract key words from text
                text = str(what_field).lower()
                # Split on common delimiters and filter short words
                words = re.split(r'[\s,;:.!?\'"()\[\]{}]+', text)
                entities.update(w for w in words if len(w) > 2)
        except (json.JSONDecodeError, ValueError):
            # Fallback to text extraction
            text = str(what_field).lower()
            words = re.split(r'[\s,;:.!?\'"()\[\]{}]+', text)
            entities.update(w for w in words if len(w) > 2)
        
        # Also add entities from 'why' field if it exists
        if 'why' in memory.keys() and memory['why']:
            why_text = str(memory['why']).lower()
            # Extract key terms from why
            words = re.split(r'[\s,;:.!?\'"()\[\]{}]+', why_text)
            entities.update(w for w in words if len(w) > 3)  # Slightly longer threshold for 'why'
        
        return entities
    
    def _apply_smart_recency(self, memory_id: str, base_recency: float, memory_groups: Dict[str, str]) -> float:
        """Apply recency as a tiebreaker only for memories in the same group.
        
        If a memory is in a group with other memories (same topic/entity),
        boost its recency score. Otherwise, use minimal recency impact.
        """
        group_id = memory_groups.get(memory_id)
        
        if not group_id:
            # Not in any group, minimal recency impact
            return base_recency * 0.1  # Heavily dampened
        
        # Count how many memories are in this group
        group_size = sum(1 for gid in memory_groups.values() if gid == group_id)
        
        if group_size <= 1:
            # Only memory in its group, minimal recency impact
            return base_recency * 0.1
        
        # Multiple memories about same topic - recency matters more
        # The more memories in the group, the more recency matters (competing information)
        recency_importance = min(1.0, group_size / 5.0)  # Max out at 5 related memories
        
        # Apply recency with importance factor
        return base_recency * (0.1 + 0.9 * recency_importance)
    
    def _fetch_meta(self, ids: List[str]):
        rows = self.store.fetch_memories(ids)
        by_id = {r['memory_id']: r for r in rows}
        return by_id

    # Old rerank method removed - using compute_all_scores instead
    def rerank_old(self, merged: Dict[str, float], rq: RetrievalQuery, 
               memory_embeddings: Optional[Dict[str, np.ndarray]] = None,
               temporal_candidate_ids: Optional[List[str]] = None,
               sem_scores: Optional[Dict[str, float]] = None,
               lex_scores: Optional[Dict[str, float]] = None,
               attention_scores: Optional[Dict[str, float]] = None) -> List[Candidate]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        metas = self._fetch_meta(list(merged.keys()))
        cands = []
        
        # Get usage stats for adaptive scoring
        usage_stats = self.store.get_usage_stats(list(merged.keys())) if hasattr(self.store, 'get_usage_stats') else {}
        
        # Determine if we have hints to adjust weights
        has_actor_hint = bool(rq.actor_hint)
        has_temporal_hint = bool(rq.temporal_hint)
        
        # Check if this is a memory recall query
        is_recall_query = False
        recall_boost = 1.0
        query_lower = rq.text.lower()
        recall_indicators = ['remember', 'recall', 'memory', 'what do you know', 
                           'what did we discuss', 'find memories', 'is there any memory']
        for indicator in recall_indicators:
            if indicator in query_lower:
                is_recall_query = True
                recall_boost = 1.5  # Boost all matching memories for recall queries
                break
        
        for mid, base in merged.items():
            m = metas.get(mid)
            if not m:
                continue
                
            # Importance is now only used in attention reranking, not base scoring
            importance = 0.0  # Will be computed during attention phase if needed
            
            # Extra signals
            rec = exp_recency(m['when_ts'], now)
            actor_match = 1.0 if (rq.actor_hint and m['who_id'] == rq.actor_hint) else 0.0
            
            # Check temporal match
            temporal_match = 0.0
            if has_temporal_hint:
                mem_date = m['when_ts'].split('T')[0] if 'T' in m['when_ts'] else m['when_ts'][:10]
                
                if isinstance(rq.temporal_hint, str):
                    # Single date match
                    temporal_match = 1.0 if mem_date == rq.temporal_hint else 0.0
                elif isinstance(rq.temporal_hint, tuple) and len(rq.temporal_hint) == 2:
                    # Date range match
                    start, end = rq.temporal_hint
                    temporal_match = 1.0 if start <= mem_date <= end else 0.0
                elif isinstance(rq.temporal_hint, dict):
                    # For relative times, we'd need to compute the actual date range
                    # This is handled by the retrieval method, so memories retrieved
                    # via temporal_candidates already match
                    if temporal_candidate_ids and mid in temporal_candidate_ids:
                        temporal_match = 1.0
            
            # Get actual usage count if available
            usage_data = usage_stats.get(mid, {})
            usage_score = min(1.0, usage_data.get('accesses', 0) / 100.0) if usage_data else 0.0
            
            # SIMPLIFIED: Use minimal scoring - let attention handle the complexity
            # Just use base score (semantic/lexical) plus hint matches
            score = base
            
            # Add bonus for exact hint matches (but don't dominate the score)
            if actor_match > 0:
                score += 0.1  # Small boost for actor match
            if temporal_match > 0:
                score += 0.1  # Small boost for temporal match
            
            # Apply recall boost if this is a memory recall query
            if is_recall_query:
                score = score * recall_boost
                
            # Create candidate with component scores for debugging
            candidate = Candidate(
                memory_id=mid, 
                score=score, 
                token_count=int(m['token_count']),
                base_score=base,
                semantic_score=sem_scores.get(mid, 0.0) if sem_scores else None,
                lexical_score=lex_scores.get(mid, 0.0) if lex_scores else None,
                recency_score=rec,
                importance_score=importance if self.use_attention else None,
                actor_score=actor_match,
                temporal_score=temporal_match,
                usage_score=usage_score,
                attention_score=attention_scores.get(mid, 0.0) if attention_scores else None
            )
            cands.append(candidate)
            
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def search(self, rq: RetrievalQuery, qvec: np.ndarray, topk_sem: int = 50, topk_lex: int = 50) -> List[Candidate]:
        # REDESIGNED: Retrieve large candidate set and score comprehensively
        # The topk parameters now only control the FINAL output size, not initial retrieval
        
        # Step 1: Get large candidate sets from each source (cast wide net)
        initial_retrieval_size = 1000  # Retrieve top 1000 from each source
        
        # Get semantic candidates (vector similarity)
        sem = self._semantic(qvec, initial_retrieval_size)
        sem_dict = {mid: score for mid, score in sem}
        
        # Get lexical candidates (text matching)
        lex = self._lexical(rq.text, initial_retrieval_size)
        lex_dict = {mid: score for mid, score in lex}
        
        # Get actor-specific candidates if hint provided
        actor_candidates = []
        if rq.actor_hint:
            actor_candidates = self._actor_based(rq.actor_hint, 500)
        actor_dict = {mid: 1.0 for mid, _ in actor_candidates}  # Binary match
        
        # Get temporal candidates if hint provided  
        temporal_candidates = []
        if rq.temporal_hint:
            temporal_candidates = self._temporal_based(rq.temporal_hint, 500)
        temporal_dict = {mid: 1.0 for mid, _ in temporal_candidates}  # Binary match
        
        # Step 2: Combine all unique memory IDs
        all_memory_ids = set()
        all_memory_ids.update(sem_dict.keys())
        all_memory_ids.update(lex_dict.keys())
        all_memory_ids.update(actor_dict.keys())
        all_memory_ids.update(temporal_dict.keys())
        
        # Step 3: Compute comprehensive scores for all candidates
        candidates = self.compute_all_scores(
            memory_ids=list(all_memory_ids),
            query_vec=qvec,
            query=rq,
            sem_scores=sem_dict,
            lex_scores=lex_dict,
            actor_matches=actor_dict,
            temporal_matches=temporal_dict
        )
        
        # Step 4: Apply final top-K selection
        candidates.sort(key=lambda x: x.score, reverse=True)
        final_k = min(topk_sem, len(candidates))
        top_k_candidates = candidates[:final_k]
        
        return top_k_candidates
