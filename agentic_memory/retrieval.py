from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from datetime import datetime, timezone
from .config import cfg
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .types import RetrievalQuery, Candidate
from .attention import MemoryAttentionHead, AdaptiveEmbeddingSpace, MemoryConsolidation

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
        
        # Initialize attention-based components
        self.attention_head = MemoryAttentionHead(embed_dim=cfg.embed_dim)
        self.adaptive_embeddings = AdaptiveEmbeddingSpace(base_dim=cfg.embed_dim)
        self.consolidator = MemoryConsolidation()
        
        # Load saved states if available
        try:
            self.consolidator.load_state(cfg.consolidation_state_path)
        except:
            pass  # Start fresh if no saved state
        
        # Track whether to use attention (can be toggled)
        self.use_attention = cfg.use_attention_retrieval

    def _semantic(self, qvec: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        return self.index.search(qvec, topk)

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

    def merge_scores(self, sem: List[Tuple[str, float]], lex: List[Tuple[str, float]], 
                     attention_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        sdict = {m:s for m,s in sem}
        ldict = {m:s for m,s in lex}
        keys = set(sdict.keys()) | set(ldict.keys())
        merged = {}
        
        if attention_scores and self.use_attention:
            # Use attention-based dynamic weighting
            for k in keys:
                base_score = sdict.get(k, 0.0) * 0.4 + ldict.get(k, 0.0) * 0.2
                attn_score = attention_scores.get(k, 0.0) * 0.4
                merged[k] = base_score + attn_score
        else:
            # Fall back to static weights
            for k in keys:
                merged[k] = cfg.w_semantic * sdict.get(k,0.0) + cfg.w_lexical * ldict.get(k,0.0)
        return merged

    def _fetch_meta(self, ids: List[str]):
        rows = self.store.fetch_memories(ids)
        by_id = {r['memory_id']: r for r in rows}
        return by_id

    def rerank(self, merged: Dict[str, float], rq: RetrievalQuery, 
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
                
            # Compute importance from consolidation network
            importance = self.consolidator.compute_memory_importance(mid) if self.use_attention else 0.0
            
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
            
            if self.use_attention:
                # Dynamic weights based on whether hints are provided
                if has_temporal_hint:
                    # Boost temporal weight when hint is provided
                    score = (base * 0.35 +
                            importance * 0.15 +
                            rec * 0.0 +            # No recency bias when temporal hint active
                            usage_score * 0.05 +
                            actor_match * 0.05 +
                            temporal_match * 0.40)   # Strong temporal weight
                elif has_actor_hint:
                    # Boost actor weight when hint is provided
                    score = (base * 0.45 +          
                            importance * 0.20 +      
                            rec * 0.0 +              # No recency bias - treat all memories equally
                            usage_score * 0.1 +
                            actor_match * 0.25 +     
                            temporal_match * 0.0)
                else:
                    # Default weights when no hints - no recency bias
                    score = (base * 0.65 +           # Increased base weight
                            importance * 0.25 +       # Increased importance weight
                            rec * 0.0 +               # Removed recency bias completely
                            usage_score * 0.1 +
                            actor_match * 0.0 +
                            temporal_match * 0.0)
            else:
                # Original static scoring - also adjust for hints
                if has_actor_hint:
                    actor_weight = 0.25  # Boost from cfg.w_actor (0.07)
                else:
                    actor_weight = cfg.w_actor
                    
                if has_temporal_hint:
                    temporal_weight = 0.35  # Boost for temporal hint
                    recency_weight = 0.0   # No recency bias
                else:
                    temporal_weight = 0.0
                    recency_weight = 0.0  # Remove recency bias - treat all memories equally
                    
                usage_boost = 0.2 if actor_match or temporal_match else 0.0
                score = base + recency_weight*rec + actor_weight*actor_match + temporal_weight*temporal_match + cfg.w_usage*usage_boost
            
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
        # Increase topk for recall queries to cast wider net
        query_lower = rq.text.lower()
        recall_indicators = ['remember', 'recall', 'memory', 'what do you know', 
                           'what did we discuss', 'find memories', 'is there any memory']
        is_recall = any(indicator in query_lower for indicator in recall_indicators)
        
        if is_recall:
            topk_sem = min(200, topk_sem * 3)  # Triple the search space for recalls
            topk_lex = min(200, topk_lex * 3)
        
        sem = self._semantic(qvec, topk_sem)
        lex = self._lexical(rq.text, topk_lex)
        
        # NEW: Add actor-specific retrieval if hint provided
        actor_candidates = []
        if rq.actor_hint:
            actor_candidates = self._actor_based(rq.actor_hint, topk_sem)
        
        # WHERE-based retrieval is now handled through semantic search of the 'where_value' field
        
        # NEW: Add temporal-specific retrieval if hint provided
        temporal_candidates = []
        if rq.temporal_hint:
            temporal_candidates = self._temporal_based(rq.temporal_hint, topk_sem)
        
        # Combine all candidate sources
        all_candidates = sem + lex + actor_candidates + temporal_candidates
        
        # Compute attention scores if enabled
        attention_scores = None
        if self.use_attention:
            # Get embeddings for candidate memories (including actor/spatial candidates)
            candidate_ids = list(set([m for m, _ in all_candidates]))
            if candidate_ids:
                embeddings = self.index.get_embeddings(candidate_ids) if hasattr(self.index, 'get_embeddings') else None
                if embeddings is not None:
                    # Compute attention-based scores
                    attn_weights = self.attention_head.compute_attention_weights(qvec, embeddings)
                    attention_scores = {cid: float(attn_weights[i]) for i, cid in enumerate(candidate_ids)}
        
        # Merge scores with optional attention (now includes actor/spatial/temporal candidates)
        merged = self.merge_scores(sem, lex, attention_scores)
        
        # Add actor/temporal candidates to merged scores if not already present
        for mid, score in actor_candidates + temporal_candidates:
            if mid not in merged:
                merged[mid] = score * 0.8  # Slightly lower base score for hint-only matches
        
        # Create score dictionaries for component tracking
        sem_dict = {mid: score for mid, score in sem}
        lex_dict = {mid: score for mid, score in lex}
        
        # Rerank with all signals
        temporal_candidate_ids = [mid for mid, _ in temporal_candidates] if temporal_candidates else None
        ranked = self.rerank(merged, rq, temporal_candidate_ids=temporal_candidate_ids,
                           sem_scores=sem_dict, lex_scores=lex_dict, attention_scores=attention_scores)
        
        # Update consolidation based on retrieval
        if ranked and self.use_attention:
            retrieved_ids = [c.memory_id for c in ranked[:10]]  # Top 10 results
            self.consolidator.hebbian_update(retrieved_ids)
            self.adaptive_embeddings.update_co_occurrence(retrieved_ids)
            
            # Periodic decay
            if np.random.random() < 0.1:  # 10% chance per search
                self.consolidator.synaptic_decay()
        
        return ranked
