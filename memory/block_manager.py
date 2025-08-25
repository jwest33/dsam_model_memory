"""
Memory Block Manager for content-addressable memory linking

Manages the creation, linking, and retrieval of memory blocks
based on content-addressable properties (5W1H).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from models.event import Event, EventType, FiveW1H
from models.memory_block import MemoryBlock, MemoryLink, LinkType
from embedding.embedder import FiveW1HEmbedder

logger = logging.getLogger(__name__)

class MemoryBlockManager:
    """
    Manages memory blocks and their content-addressable linking
    """
    
    def __init__(self, embedder: Optional[FiveW1HEmbedder] = None):
        """Initialize the block manager"""
        self.blocks: Dict[str, MemoryBlock] = {}
        self.event_to_blocks: Dict[str, Set[str]] = defaultdict(set)  # event_id -> block_ids
        self.embedder = embedder
        
        # Configuration (principled values based on cognitive science)
        self.temporal_window = 300  # 5 minutes - typical short-term memory span
        self.semantic_threshold = 0.5  # 50% similarity - balanced threshold
        self.merge_threshold = 0.8  # 80% similarity - high confidence for merging
        self.max_block_size = 15  # Miller's law range (7±2) * 2 for flexibility

    def process_event(self, event: Event, context_events: Optional[List[Event]] = None) -> MemoryBlock:
        """
        Process a new event and assign it to appropriate memory block(s)
        
        Args:
            event: New event to process
            context_events: Recent events for context
            
        Returns:
            Primary memory block containing the event
        """
        # Find candidate blocks for this event
        candidate_blocks = self._find_candidate_blocks(event, context_events)
        
        if candidate_blocks:
            # Add to best matching block(s)
            primary_block = candidate_blocks[0][0]
            for block, score, link_type in candidate_blocks[:3]:  # Add to top 3 matches
                if score >= self.semantic_threshold:
                    self._add_event_to_block(block, event, link_type)
                    if block != primary_block and score >= self.merge_threshold:
                        # Consider merging blocks if very similar
                        self._maybe_merge_blocks(primary_block, block)
        else:
            # Create new block
            primary_block = self._create_new_block(event)
        
        # Check for special patterns and create additional links
        self._detect_and_link_patterns(event, primary_block, context_events)
        
        return primary_block
    
    def _find_candidate_blocks(
        self, 
        event: Event, 
        context_events: Optional[List[Event]] = None
    ) -> List[Tuple[MemoryBlock, float, LinkType]]:
        """
        Find candidate blocks for an event based on content-addressable properties
        
        Returns:
            List of (block, score, suggested_link_type) tuples, sorted by score
        """
        candidates = []
        
        for block_id, block in self.blocks.items():
            # Skip full blocks
            if len(block.events) >= self.max_block_size:
                continue
            
            # Compute multiple similarity scores
            scores = {}
            
            # 1. Temporal proximity
            if block.events:
                latest_event = max(block.events, key=lambda e: e.created_at)
                time_diff = (event.created_at - latest_event.created_at).total_seconds()
                if abs(time_diff) <= self.temporal_window:
                    scores['temporal'] = np.exp(-abs(time_diff) / self.temporal_window)
                else:
                    scores['temporal'] = 0.0
            else:
                scores['temporal'] = 0.0
            
            # 2. Episode match
            if event.episode_id in {e.episode_id for e in block.events}:
                scores['episodic'] = 1.0
            else:
                scores['episodic'] = 0.0
            
            # 3. Semantic similarity with aggregate signature
            if block.aggregate_signature:
                scores['semantic'] = self._compute_5w1h_similarity(
                    event.five_w1h, 
                    block.aggregate_signature
                )
            else:
                scores['semantic'] = 0.0
            
            # 4. Causal relationship (action -> observation)
            scores['causal'] = 0.0
            if block.events:
                last_event = block.events[-1]
                if (last_event.event_type == EventType.ACTION and 
                    event.event_type == EventType.OBSERVATION):
                    scores['causal'] = 0.9
                elif (last_event.event_type == EventType.USER_INPUT and 
                      event.event_type == EventType.ACTION):
                    scores['causal'] = 0.8
            
            # 5. Conversational continuity
            scores['conversational'] = 0.0
            if self._is_conversation_continuation(event, block):
                scores['conversational'] = 0.85
            
            # 6. Content-addressable match on specific fields
            scores['who_match'] = 0.0
            scores['where_match'] = 0.0
            if block.aggregate_signature:
                if event.five_w1h.who and event.five_w1h.who in block.aggregate_signature.who:
                    scores['who_match'] = 0.7
                if event.five_w1h.where and event.five_w1h.where == block.aggregate_signature.where:
                    scores['where_match'] = 0.6
            
            # Compute weighted overall score (weights sum to 1.0)
            weights = {
                'temporal': 0.15,       # Time proximity
                'episodic': 0.20,       # Same episode/session
                'semantic': 0.25,       # Content similarity (most important)
                'causal': 0.15,         # Cause-effect relationships
                'conversational': 0.15, # Dialog continuity
                'who_match': 0.05,      # Actor consistency
                'where_match': 0.05     # Location consistency
            }
            
            overall_score = sum(scores[k] * weights[k] for k in weights)
            
            # Determine best link type based on highest component score
            if overall_score > 0:
                best_component = max(scores.items(), key=lambda x: x[1])
                link_type_map = {
                    'temporal': LinkType.TEMPORAL,
                    'episodic': LinkType.EPISODIC,
                    'semantic': LinkType.SEMANTIC,
                    'causal': LinkType.CAUSAL,
                    'conversational': LinkType.CONVERSATIONAL,
                    'who_match': LinkType.SEMANTIC,
                    'where_match': LinkType.SEMANTIC
                }
                link_type = link_type_map.get(best_component[0], LinkType.SEMANTIC)
                
                candidates.append((block, overall_score, link_type))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def _compute_5w1h_similarity(self, a: FiveW1H, b: FiveW1H) -> float:
        """Compute similarity between two 5W1H structures"""
        similarities = []
        
        # Compare each field with appropriate weights
        field_weights = {
            'who': 0.15,
            'what': 0.35,  # Most important for content
            'where': 0.15,
            'why': 0.2,
            'how': 0.15
        }
        
        for field, weight in field_weights.items():
            val_a = getattr(a, field, "")
            val_b = getattr(b, field, "")
            
            if not val_a or not val_b:
                similarity = 0.0
            elif val_a == val_b:
                similarity = 1.0
            else:
                # Check for substring matches
                similarity = self._text_similarity(val_a, val_b)

            similarities.append(similarity * weight)
        
        return sum(similarities)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using embeddings with cosine similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Primary: Use embeddings if available
        if self.embedder and hasattr(self.embedder, 'text_embedder'):
            try:
                emb1 = self.embedder.text_embedder.embed_text(text1)
                emb2 = self.embedder.text_embedder.embed_text(text2)
                
                # Compute cosine similarity
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                # Avoid division by zero
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                # Ensure similarity is in [0, 1] range (cosine sim is [-1, 1])
                return float(max(0.0, similarity))
            
            except Exception as e:
                logger.debug(f"Embedding computation failed, using fallback: {e}")
                # Fall through to alternative method
        
        # Fallback: Jaccard similarity on word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_conversation_continuation(self, event: Event, block: MemoryBlock) -> bool:
        """Check if event continues a conversation in the block"""
        if not block.events:
            return False
        
        # Check for user-assistant pattern
        last_event = block.events[-1]
        
        # User -> Assistant
        if (last_event.five_w1h.who == "User" and 
            event.five_w1h.who == "Assistant"):
            return True
        
        # Assistant -> User (follow-up)
        if (last_event.five_w1h.who == "Assistant" and 
            event.five_w1h.who == "User"):
            # Check temporal proximity for follow-up
            time_diff = (event.created_at - last_event.created_at).total_seconds()
            return time_diff <= self.temporal_window  # Within temporal window

        return False
    
    def _add_event_to_block(self, block: MemoryBlock, event: Event, link_type: LinkType):
        """Add an event to a block with appropriate linking"""
        # Find best event to link to
        link_to = None
        if block.events:
            if link_type == LinkType.TEMPORAL:
                # Link to most recent
                link_to = block.events[-1].id
            elif link_type == LinkType.CAUSAL:
                # Link to last action if this is observation
                if event.event_type == EventType.OBSERVATION:
                    for e in reversed(block.events):
                        if e.event_type == EventType.ACTION:
                            link_to = e.id
                            break
                if not link_to:
                    link_to = block.events[-1].id
            elif link_type == LinkType.CONVERSATIONAL:
                # Link to last message from other party
                for e in reversed(block.events):
                    if e.five_w1h.who != event.five_w1h.who:
                        link_to = e.id
                        break
                if not link_to:
                    link_to = block.events[-1].id
            else:
                # For semantic and others, link to most similar
                best_similarity = 0.0
                for e in block.events:
                    sim = self._compute_5w1h_similarity(event.five_w1h, e.five_w1h)
                    if sim > best_similarity:
                        best_similarity = sim
                        link_to = e.id
        
        # Add to block
        link_strength = 0.8  # Default strength
        block.add_event(event, link_to, link_type, link_strength)
        
        # Update tracking
        self.event_to_blocks[event.id].add(block.id)
        
        # Update block salience
        block.compute_block_salience()
    
    def _create_new_block(self, event: Event) -> MemoryBlock:
        """Create a new memory block for an event"""
        block = MemoryBlock()
        
        # Determine block type based on event
        if event.five_w1h.who in ["User", "Assistant"]:
            block.block_type = "conversation"
        elif event.event_type == EventType.SYSTEM_EVENT:
            block.block_type = "system"
        else:
            block.block_type = "general"
        
        # Add the event
        block.add_event(event)
        
        # Store block
        self.blocks[block.id] = block
        self.event_to_blocks[event.id].add(block.id)
        
        # Update salience matrix for the new block
        block._update_salience_matrix()
        
        logger.info(f"Created new memory block: {block.id}")
        
        return block
    
    def _maybe_merge_blocks(self, block1: MemoryBlock, block2: MemoryBlock):
        """Merge two blocks if they're sufficiently similar"""
        if block1.id == block2.id:
            return
        
        # Don't merge if combined size would be too large
        if len(block1.events) + len(block2.events) > self.max_block_size:
            return
        
        # Compute similarity between blocks
        if block1.aggregate_signature and block2.aggregate_signature:
            similarity = self._compute_5w1h_similarity(
                block1.aggregate_signature,
                block2.aggregate_signature
            )
            
            if similarity >= self.merge_threshold:
                logger.info(f"Merging blocks {block1.id} and {block2.id} (similarity: {similarity:.2f})")
                
                # Merge block2 into block1
                block1.merge_with(block2)
                
                # Update tracking
                for event_id in block2.event_ids:
                    self.event_to_blocks[event_id].discard(block2.id)
                    self.event_to_blocks[event_id].add(block1.id)
                
                # Remove block2
                del self.blocks[block2.id]
    
    def _detect_and_link_patterns(
        self, 
        event: Event, 
        block: MemoryBlock,
        context_events: Optional[List[Event]] = None
    ):
        """Detect special patterns and create additional links"""
        
        # Pattern 1: Reference detection (event mentions another event)
        if context_events:
            for context_event in context_events:
                if self._event_references_other(event, context_event):
                    # Create reference link
                    link = MemoryLink(
                        source_id=event.id,
                        target_id=context_event.id,
                        link_type=LinkType.REFERENCE,
                        strength=0.9,
                        metadata={"type": "explicit_reference"}
                    )
                    
                    # Add to appropriate block
                    for block_id in self.event_to_blocks.get(context_event.id, set()):
                        if block_id in self.blocks:
                            self.blocks[block_id].links.append(link)
        
        # Pattern 2: Goal-oriented sequences
        if event.five_w1h.why:
            for other_event in block.events:
                if other_event.id != event.id and other_event.five_w1h.why:
                    if self._share_goal(event, other_event):
                        # Create semantic link for shared goal
                        link = MemoryLink(
                            source_id=other_event.id,
                            target_id=event.id,
                            link_type=LinkType.SEMANTIC,
                            strength=0.7,
                            metadata={"type": "shared_goal"}
                        )
                        block.links.append(link)
    
    def _event_references_other(self, event: Event, other: Event) -> bool:
        """Check if one event references another"""
        # Simple heuristic: check if key words from other event appear in this one
        other_keywords = set(other.five_w1h.what.lower().split()[:5])  # First 5 words
        event_text = event.five_w1h.what.lower()
        
        matches = sum(1 for keyword in other_keywords if keyword in event_text)
        return matches >= 3  # At least 3 keyword matches
    
    def _share_goal(self, event1: Event, event2: Event) -> bool:
        """Check if two events share a goal"""
        why1 = event1.five_w1h.why.lower()
        why2 = event2.five_w1h.why.lower()
        
        # Check for significant overlap
        words1 = set(why1.split())
        words2 = set(why2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return overlap >= 0.5
    
    def get_blocks_for_event(self, event_id: str) -> List[MemoryBlock]:
        """Get all blocks containing an event"""
        blocks = []
        for block_id in self.event_to_blocks.get(event_id, set()):
            if block_id in self.blocks:
                blocks.append(self.blocks[block_id])
        return blocks
    
    def retrieve_relevant_blocks(
        self, 
        query: Dict[str, str], 
        k: int = 5,
        use_embeddings: bool = True
    ) -> List[Tuple[MemoryBlock, float]]:
        """
        Retrieve memory blocks relevant to a query using embeddings or 5W1H matching
        
        Args:
            query: Partial 5W1H query
            k: Number of blocks to retrieve
            use_embeddings: Use block embeddings for retrieval if available
            
        Returns:
            List of (block, relevance_score) tuples
        """
        results = []
        
        # Try embedding-based retrieval first if requested
        if use_embeddings and self.embedder:
            try:
                # Create query embedding
                query_text = ' '.join([v for v in query.values() if v])
                if query_text:
                    query_embedding = self.embedder.text_embedder.embed_text(query_text)
                    query_embedding = query_embedding / np.linalg.norm(query_embedding)
                    
                    for block_id, block in self.blocks.items():
                        if block.block_embedding is not None:
                            # Compute cosine similarity
                            similarity = np.dot(query_embedding, block.block_embedding)
                            
                            # Apply boosts
                            score = similarity
                            
                            # Boost for recency
                            if block.updated_at:
                                age_hours = (datetime.utcnow() - block.updated_at).total_seconds() / 3600
                                recency_boost = np.exp(-age_hours / 168)  # Decay over a week
                                score = score * 0.7 + recency_boost * 0.15
                            
                            # Boost for block salience
                            score = score * 0.7 + block.salience * 0.15
                            
                            if score > 0:
                                results.append((block, score))
                    
                    if results:
                        results.sort(key=lambda x: x[1], reverse=True)
                        return results[:k]
            except Exception as e:
                logger.debug(f"Embedding retrieval failed, falling back to 5W1H: {e}")
        
        # Fallback to 5W1H similarity
        query_5w1h = FiveW1H(
            who=query.get('who', ''),
            what=query.get('what', ''),
            when=query.get('when', ''),
            where=query.get('where', ''),
            why=query.get('why', ''),
            how=query.get('how', '')
        )
        
        for block_id, block in self.blocks.items():
            if not block.aggregate_signature:
                continue
            
            # Compute relevance score
            relevance = self._compute_5w1h_similarity(query_5w1h, block.aggregate_signature)
            
            # Boost for recency
            if block.updated_at:
                age_hours = (datetime.utcnow() - block.updated_at).total_seconds() / 3600
                recency_boost = np.exp(-age_hours / 168)  # Decay over a week
                relevance = relevance * 0.8 + recency_boost * 0.1
            
            # Boost for block salience
            relevance = relevance * 0.8 + block.salience * 0.1
            
            if relevance > 0:
                results.append((block, relevance))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def save_state(self, path: str):
        """Save block manager state to disk"""
        import json
        
        state = {
            "blocks": {
                block_id: block.to_dict() 
                for block_id, block in self.blocks.items()
            },
            "event_to_blocks": {
                event_id: list(block_ids)
                for event_id, block_ids in self.event_to_blocks.items()
            },
            "config": {
                "temporal_window": self.temporal_window,
                "semantic_threshold": self.semantic_threshold,
                "merge_threshold": self.merge_threshold,
                "max_block_size": self.max_block_size
            }
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, path: str, events: List[Event]):
        """Load block manager state from disk"""
        import json
        from pathlib import Path
        
        if not Path(path).exists():
            logger.warning(f"Block manager state file not found: {path}")
            return
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Create event lookup
            event_lookup = {e.id: e for e in events}
            
            # Restore blocks
            self.blocks = {}
            for block_id, block_data in state.get("blocks", {}).items():
                # Get events for this block
                block_events = [
                    event_lookup[eid] 
                    for eid in block_data.get("event_ids", [])
                    if eid in event_lookup
                ]
                
                if block_events:  # Only restore block if it has events
                    block = MemoryBlock.from_dict(block_data, block_events)
                    self.blocks[block_id] = block
            
            # Restore event-to-blocks mapping
            self.event_to_blocks = defaultdict(set)
            for event_id, block_ids in state.get("event_to_blocks", {}).items():
                self.event_to_blocks[event_id] = set(block_ids)
            
            # Restore config
            config = state.get("config", {})
            self.temporal_window = config.get("temporal_window", self.temporal_window)
            self.semantic_threshold = config.get("semantic_threshold", self.semantic_threshold)
            self.merge_threshold = config.get("merge_threshold", self.merge_threshold)
            self.max_block_size = config.get("max_block_size", self.max_block_size)
            
            logger.info(f"Loaded {len(self.blocks)} memory blocks from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load block manager state: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory blocks"""
        if not self.blocks:
            return {
                "total_blocks": 0,
                "total_events": 0,
                "avg_block_size": 0,
                "avg_coherence": 0,
                "avg_salience": 0,
                "block_types": {}
            }
        
        block_sizes = [len(b.events) for b in self.blocks.values()]
        coherences = [b.coherence_score for b in self.blocks.values()]
        saliences = [b.salience for b in self.blocks.values()]
        
        # Count block types
        type_counts = defaultdict(int)
        for block in self.blocks.values():
            type_counts[block.block_type] += 1
        
        return {
            "total_blocks": len(self.blocks),
            "total_events": sum(block_sizes),
            "avg_block_size": np.mean(block_sizes) if block_sizes else 0,
            "max_block_size": max(block_sizes) if block_sizes else 0,
            "min_block_size": min(block_sizes) if block_sizes else 0,
            "avg_coherence": np.mean(coherences) if coherences else 0,
            "avg_salience": np.mean(saliences) if saliences else 0,
            "block_types": dict(type_counts),
            "total_links": sum(len(b.links) for b in self.blocks.values())
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return f"MemoryBlockManager(blocks={len(self.blocks)}, events={sum(len(b.events) for b in self.blocks.values())})"
