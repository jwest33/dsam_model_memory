"""
Unified Temporal Manager for the Memory System.

This module consolidates all temporal logic from across the codebase into a single,
coherent interface. It manages temporal chains, temporal queries, temporal merging,
and temporal dimension attention in a unified way.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dateutil import parser as date_parser
from enum import Enum
import logging
import uuid

from models.event import Event
from models.merged_event import EventRelationship
from models.merge_types import MergeType
from memory.temporal_chain import TemporalChain, ChainType
from memory.temporal_query import TemporalQueryHandler

logger = logging.getLogger(__name__)


class TemporalStrength(Enum):
    """Temporal signal strength levels."""
    STRONG = "strong"       # Explicit temporal indicators like "last thing", "just now"
    MODERATE = "moderate"   # General temporal references like "recent", "earlier"
    WEAK = "weak"          # Implicit temporal context
    NONE = "none"          # No temporal signal


class TemporalManager:
    """
    Unified manager for all temporal operations in the memory system.
    
    Responsibilities:
    1. Temporal chain management (conversation threads, episodes)
    2. Temporal query detection and weighting
    3. Temporal merge group creation and management
    4. Temporal dimension attention computation
    5. Temporal event retrieval and ranking
    """
    
    # Consolidated temporal indicators with strength levels
    TEMPORAL_INDICATORS = {
        TemporalStrength.STRONG: [
            "last thing", "just discussed", "just talked", "most recent",
            "latest", "what did we just", "last time", "previous message",
            "just now", "right before this", "the last", "our last exchange"
        ],
        TemporalStrength.MODERATE: [
            "recent", "recently", "earlier today", "a moment ago",
            "not long ago", "earlier", "before", "previously",
            "a while ago", "sometime today", "this morning", "this afternoon"
        ],
        TemporalStrength.WEAK: [
            "when", "time", "ago", "conversation", "discussion",
            "thread", "session", "chat", "history", "past"
        ]
    }
    
    def get_temporal_indicators_dict(self):
        """Get temporal indicators as a simple dictionary for API responses."""
        return {
            'strong': list(self.TEMPORAL_INDICATORS[TemporalStrength.STRONG]),
            'moderate': list(self.TEMPORAL_INDICATORS[TemporalStrength.MODERATE]),
            'weak': list(self.TEMPORAL_INDICATORS[TemporalStrength.WEAK])
        }
    
    # Temporal keywords for attention scoring
    TEMPORAL_KEYWORDS = [
        'when', 'time', 'last', 'recent', 'latest', 'earlier',
        'before', 'after', 'previous', 'next', 'first', 'then',
        'yesterday', 'today', 'tomorrow', 'ago', 'now', 'just',
        'moment', 'recently', 'previously', 'subsequently'
    ]
    
    def __init__(self, encoder=None, chromadb_store=None, similarity_cache=None):
        """
        Initialize the temporal manager.
        
        Args:
            encoder: Dual-space encoder for embeddings
            chromadb_store: ChromaDB store for persistence
            similarity_cache: Similarity cache for performance
        """
        self.encoder = encoder
        self.chromadb = chromadb_store
        self.similarity_cache = similarity_cache
        
        # Initialize components
        self.temporal_chain = TemporalChain(
            chromadb_store=chromadb_store,
            encoder=encoder,
            similarity_cache=similarity_cache
        )
        self.temporal_query_handler = TemporalQueryHandler(
            encoder=encoder,
            chromadb_store=chromadb_store,
            similarity_cache=similarity_cache
        )
        
        # Temporal merge groups (for multi-dimensional merging)
        self.temporal_merge_groups = {}  # merge_id -> group_data
        self.event_to_temporal_group = {}  # event_id -> merge_id
        
        # Load existing temporal data
        self._load_temporal_data()
    
    def _load_temporal_data(self):
        """Load existing temporal chains from storage."""
        # Load temporal chains only
        self.temporal_chain._load_chains_from_db()
        
        # NOTE: We don't load temporal merge groups here anymore
        # The MultiDimensionalMerger handles all temporal merge group creation and loading
        # This prevents duplicate temporal groups from being created
        # 
        # The temporal_merge_groups dictionary stays empty and is not used
        # All temporal merge groups are managed by MultiDimensionalMerger
        logger.info("Temporal chains loaded (merge groups managed by MultiDimensionalMerger)")
    
    def detect_temporal_intent(self, query: Dict[str, str]) -> Tuple[TemporalStrength, float, Dict]:
        """
        Detect temporal intent in a query and return strength and context.
        
        Args:
            query: Query fields dictionary
            
        Returns:
            Tuple of (temporal_strength, confidence, temporal_context)
        """
        query_text = ' '.join([str(v) for v in query.values() if v]).lower()
        
        # Check for strong temporal indicators
        for indicator in self.TEMPORAL_INDICATORS[TemporalStrength.STRONG]:
            if indicator in query_text:
                return TemporalStrength.STRONG, 0.9, {
                    'type': 'strong_recency',
                    'window_hours': 2,
                    'decay_rate': 0.5
                }
        
        # Check for moderate temporal indicators
        for indicator in self.TEMPORAL_INDICATORS[TemporalStrength.MODERATE]:
            if indicator in query_text:
                return TemporalStrength.MODERATE, 0.6, {
                    'type': 'moderate_recency',
                    'window_hours': 24,
                    'decay_rate': 0.7
                }
        
        # Check for weak temporal indicators
        temporal_keyword_count = sum(1 for keyword in self.TEMPORAL_KEYWORDS if keyword in query_text)
        if temporal_keyword_count >= 2:
            return TemporalStrength.WEAK, 0.3, {
                'type': 'session_based',
                'window_hours': 8,
                'decay_rate': 0.85
            }
        
        return TemporalStrength.NONE, 0.0, {}
    
    def compute_temporal_attention_weight(self, query: Dict[str, str]) -> float:
        """
        Compute the attention weight for the temporal dimension.
        
        Args:
            query: Query fields dictionary
            
        Returns:
            Temporal attention weight (0-1)
        """
        query_text = ' '.join([str(v) for v in query.values() if v]).lower()
        score = 0.0
        
        # Check if 'when' field is specified
        if query.get('when'):
            score += 0.5
        
        # Detect temporal strength
        strength, confidence, _ = self.detect_temporal_intent(query)
        
        if strength == TemporalStrength.STRONG:
            score += 0.7  # Strong boost for explicit temporal queries
        elif strength == TemporalStrength.MODERATE:
            score += 0.5
        elif strength == TemporalStrength.WEAK:
            score += 0.3
        
        # Check for temporal keywords
        temporal_matches = sum(1 for keyword in self.TEMPORAL_KEYWORDS if keyword in query_text)
        score += min(temporal_matches * 0.1, 0.3)
        
        # Check for date/time patterns
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|ago|minutes?|hours?|days?|weeks?|months?'
        if re.search(date_pattern, query_text):
            score += 0.2
        
        # Check for conversation/thread context
        if any(word in query_text for word in ['conversation', 'thread', 'discussion', 'chat', 'session']):
            score += 0.2
        
        return min(score, 1.0)
    
    def add_event_to_temporal_chain(self, event: Event, context: Optional[Dict] = None) -> str:
        """
        Add an event to the appropriate temporal chain.
        
        Args:
            event: The event to add
            context: Optional context for chain selection
            
        Returns:
            Chain ID the event was added to
        """
        chain_id = self.temporal_chain.add_event(event, context)
        
        # NOTE: We don't update temporal merge groups here anymore
        # The MultiDimensionalMerger handles all temporal merge group management
        # This prevents duplicate temporal groups from being created
        
        return chain_id
    
    def _update_temporal_merge_group(self, event: Event, chain_id: str):
        """
        Update or create temporal merge group for an event.
        
        Args:
            event: The event to process
            chain_id: The temporal chain ID
        """
        # Check if this chain already has a merge group
        merge_id = None
        for mid, group in self.temporal_merge_groups.items():
            if group.get('chain_id') == chain_id:
                merge_id = mid
                break
        
        if not merge_id:
            # Create new temporal merge group
            merge_id = f"temporal_{uuid.uuid4().hex[:8]}"
            self.temporal_merge_groups[merge_id] = {
                'key': f"Chain {chain_id}",
                'chain_id': chain_id,
                'merge_count': 0,
                'created_at': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat(),
                'raw_event_ids': []
            }
        
        # Add event to group
        raw_event_id = f"raw_{event.id}" if not event.id.startswith("raw_") else event.id
        if raw_event_id not in self.temporal_merge_groups[merge_id]['raw_event_ids']:
            self.temporal_merge_groups[merge_id]['raw_event_ids'].append(raw_event_id)
            self.temporal_merge_groups[merge_id]['merge_count'] += 1
            self.temporal_merge_groups[merge_id]['last_updated'] = datetime.utcnow().isoformat()
            self.event_to_temporal_group[raw_event_id] = merge_id
            
            # Persist to ChromaDB
            self._save_temporal_merge_group(merge_id)
    
    def _save_temporal_merge_group(self, merge_id: str):
        """Save temporal merge group to ChromaDB."""
        if not self.chromadb or not hasattr(self.chromadb, 'temporal_merges_collection'):
            return
            
        group = self.temporal_merge_groups[merge_id]
        
        # Prepare metadata
        metadata = {
            'merge_type': 'temporal',
            'merge_key': group['key'],
            'merge_count': group['merge_count'],
            'created_at': group['created_at'],
            'last_updated': group['last_updated'],
            'chain_id': group.get('chain_id', ''),
            'raw_event_ids': ','.join(group['raw_event_ids'])
        }
        
        # Use a simple embedding for now (can be enhanced)
        embedding = np.random.randn(768).tolist()  # Placeholder
        
        try:
            self.chromadb.temporal_merges_collection.upsert(
                ids=[merge_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Failed to save temporal merge group {merge_id}: {e}")
    
    def get_temporal_context_for_event(self, event_id: str) -> Dict:
        """
        Get comprehensive temporal context for an event.
        
        Args:
            event_id: The event ID
            
        Returns:
            Dictionary containing temporal context information
        """
        context = {
            'chain_id': None,
            'chain_type': None,
            'chain_position': -1,
            'chain_length': 0,
            'temporal_group_id': None,
            'temporal_group_size': 0,
            'relationships': [],
            'related_events': []
        }
        
        # Get chain information
        chain_id = self.temporal_chain.event_to_chain.get(event_id)
        if chain_id:
            chain_events = self.temporal_chain.chains.get(chain_id, [])
            context['chain_id'] = chain_id
            context['chain_type'] = self.temporal_chain.chain_types.get(chain_id, ChainType.UPDATE_CHAIN).value
            context['chain_length'] = len(chain_events)
            if event_id in chain_events:
                context['chain_position'] = chain_events.index(event_id)
            context['related_events'] = chain_events[:5]  # First 5 related events
            
            # Get relationships
            if event_id in self.temporal_chain.supersedes_map:
                context['relationships'].append(('supersedes', self.temporal_chain.supersedes_map[event_id]))
            if event_id in self.temporal_chain.corrections_map:
                context['relationships'].append(('corrects', self.temporal_chain.corrections_map[event_id]))
            if event_id in self.temporal_chain.continues_map:
                context['relationships'].append(('continues', self.temporal_chain.continues_map[event_id]))
        
        # Get temporal merge group information
        temporal_group_id = self.event_to_temporal_group.get(event_id)
        if temporal_group_id:
            context['temporal_group_id'] = temporal_group_id
            context['temporal_group_size'] = self.temporal_merge_groups[temporal_group_id]['merge_count']
        
        return context
    
    def apply_temporal_weighting(self, candidates: List[Tuple], query: Dict[str, str],
                                query_time: Optional[datetime] = None) -> List[Tuple]:
        """
        Apply temporal weighting to retrieval candidates.
        
        Args:
            candidates: List of (event, score) tuples
            query: Query fields dictionary
            query_time: Time of query (defaults to now)
            
        Returns:
            Reranked list of (event, adjusted_score) tuples
        """
        # Detect temporal intent
        strength, confidence, temporal_context = self.detect_temporal_intent(query)
        
        if strength == TemporalStrength.NONE:
            return candidates  # No temporal weighting needed
        
        if query_time is None:
            query_time = datetime.utcnow()
        
        weighted_results = []
        
        for event, score in candidates:
            # Calculate temporal weight based on event timestamp
            temporal_weight = 1.0
            
            if hasattr(event, 'five_w1h') and event.five_w1h.when:
                try:
                    event_time = date_parser.parse(event.five_w1h.when)
                    if event_time.tzinfo is None and query_time.tzinfo is not None:
                        query_time = query_time.replace(tzinfo=None)
                    elif event_time.tzinfo is not None and query_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=None)
                    
                    time_diff = query_time - event_time
                    hours_diff = max(0, time_diff.total_seconds() / 3600)
                    
                    # Apply decay based on temporal strength
                    if strength == TemporalStrength.STRONG:
                        # Strong exponential decay for very recent items
                        decay_rate = temporal_context.get('decay_rate', 0.5)
                        window_hours = temporal_context.get('window_hours', 2)
                        temporal_weight = np.exp(-hours_diff / (window_hours * decay_rate))
                    elif strength == TemporalStrength.MODERATE:
                        # Moderate Gaussian-like decay
                        window_hours = temporal_context.get('window_hours', 24)
                        temporal_weight = np.exp(-(hours_diff ** 2) / (2 * window_hours ** 2))
                    else:  # WEAK
                        # Mild exponential decay
                        decay_rate = temporal_context.get('decay_rate', 0.85)
                        window_hours = temporal_context.get('window_hours', 8)
                        temporal_weight = decay_rate ** (hours_diff / window_hours)
                    
                    temporal_weight = np.clip(temporal_weight, 0.1, 1.0)
                    
                except Exception as e:
                    logger.debug(f"Could not parse timestamp: {e}")
            
            # Combine semantic and temporal scores
            if strength == TemporalStrength.STRONG:
                # Heavily weight temporal factor for strong signals
                adjusted_score = score * (temporal_weight ** (confidence * 0.9))
            elif strength == TemporalStrength.MODERATE:
                # Moderate blending
                adjusted_score = score * (temporal_weight ** (confidence * 0.6))
            else:  # WEAK
                # Light temporal influence
                adjusted_score = (1 - confidence * 0.3) * score + (confidence * 0.3) * temporal_weight * score
            
            weighted_results.append((event, adjusted_score))
        
        # Re-sort by adjusted scores
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_results
    
    def get_temporal_chains_summary(self) -> Dict:
        """
        Get a summary of all temporal chains.
        
        Returns:
            Dictionary containing chain statistics and information
        """
        summary = {
            'total_chains': len(self.temporal_chain.chains),
            'total_events': sum(len(events) for events in self.temporal_chain.chains.values()),
            'chain_types': {},
            'largest_chains': [],
            'recent_chains': []
        }
        
        # Count chain types
        for chain_type in self.temporal_chain.chain_types.values():
            type_name = chain_type.value
            summary['chain_types'][type_name] = summary['chain_types'].get(type_name, 0) + 1
        
        # Find largest chains
        chains_by_size = sorted(
            [(chain_id, len(events)) for chain_id, events in self.temporal_chain.chains.items()],
            key=lambda x: x[1],
            reverse=True
        )
        summary['largest_chains'] = [
            {'chain_id': cid, 'size': size} for cid, size in chains_by_size[:5]
        ]
        
        # Find most recent chains
        chains_by_time = []
        for chain_id, metadata in self.temporal_chain.chain_metadata.items():
            if 'last_updated' in metadata:
                chains_by_time.append((chain_id, metadata['last_updated']))
        
        chains_by_time.sort(key=lambda x: x[1], reverse=True)
        summary['recent_chains'] = [
            {'chain_id': cid, 'last_updated': timestamp} for cid, timestamp in chains_by_time[:5]
        ]
        
        return summary
    
    def get_temporal_merge_groups(self) -> List[Dict]:
        """
        Get all temporal merge groups from MultiDimensionalMerger via ChromaDB.
        
        Returns:
            List of temporal merge group dictionaries
        """
        # NOTE: This method now returns an empty list since temporal merge groups
        # are managed by MultiDimensionalMerger, not the TemporalManager
        # The web app should query MultiDimensionalMerger directly for temporal groups
        return []
    
    def consolidate_temporal_chains(self, similarity_threshold: float = 0.7):
        """
        Consolidate similar temporal chains to reduce fragmentation.
        
        Args:
            similarity_threshold: Threshold for merging similar chains
        """
        if not self.encoder:
            logger.warning("Cannot consolidate chains without encoder")
            return
        
        # Find and merge similar chains
        merged_count = 0
        chain_ids = list(self.temporal_chain.chains.keys())
        
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                chain1_id = chain_ids[i]
                chain2_id = chain_ids[j]
                
                # Skip if either chain no longer exists (already merged)
                if chain1_id not in self.temporal_chain.chains or chain2_id not in self.temporal_chain.chains:
                    continue
                
                # Check similarity
                if self._chains_are_similar(chain1_id, chain2_id, similarity_threshold):
                    # Merge chains
                    self.temporal_chain.merge_chains(chain1_id, chain2_id)
                    merged_count += 1
                    
                    # Update temporal merge groups
                    self._merge_temporal_groups(chain1_id, chain2_id)
        
        logger.info(f"Consolidated {merged_count} temporal chains")
    
    def _chains_are_similar(self, chain1_id: str, chain2_id: str, threshold: float) -> bool:
        """
        Check if two chains are similar enough to merge.
        
        Args:
            chain1_id: First chain ID
            chain2_id: Second chain ID
            threshold: Similarity threshold
            
        Returns:
            True if chains should be merged
        """
        # Get chain embeddings
        emb1 = self.temporal_chain.chain_embeddings.get(chain1_id)
        emb2 = self.temporal_chain.chain_embeddings.get(chain2_id)
        
        if not emb1 or not emb2:
            return False
        
        # Compute similarity
        euc_sim = self._cosine_similarity(emb1['euclidean'], emb2['euclidean'])
        
        # Check hyperbolic similarity
        from memory.dual_space_encoder import HyperbolicOperations
        hyp_dist = HyperbolicOperations.geodesic_distance(emb1['hyperbolic'], emb2['hyperbolic'])
        hyp_sim = np.exp(-hyp_dist)
        
        # Combine similarities
        combined_sim = 0.5 * euc_sim + 0.5 * hyp_sim
        
        return combined_sim > threshold
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _merge_temporal_groups(self, chain1_id: str, chain2_id: str):
        """
        Merge temporal groups when their chains are merged.
        
        Args:
            chain1_id: First chain ID (kept)
            chain2_id: Second chain ID (merged into first)
        """
        # Find groups for each chain
        group1_id = None
        group2_id = None
        
        for merge_id, group in self.temporal_merge_groups.items():
            if group.get('chain_id') == chain1_id:
                group1_id = merge_id
            elif group.get('chain_id') == chain2_id:
                group2_id = merge_id
        
        if group1_id and group2_id:
            # Merge group2 into group1
            group1 = self.temporal_merge_groups[group1_id]
            group2 = self.temporal_merge_groups[group2_id]
            
            # Combine raw event IDs
            group1['raw_event_ids'].extend(group2['raw_event_ids'])
            group1['merge_count'] = len(group1['raw_event_ids'])
            group1['last_updated'] = datetime.utcnow().isoformat()
            
            # Update event mappings
            for event_id in group2['raw_event_ids']:
                self.event_to_temporal_group[event_id] = group1_id
            
            # Remove group2
            del self.temporal_merge_groups[group2_id]
            
            # Save updated group
            self._save_temporal_merge_group(group1_id)
            
            logger.info(f"Merged temporal group {group2_id} into {group1_id}")