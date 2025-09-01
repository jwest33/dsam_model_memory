"""
Temporal Chain Manager for Event Relationships

This module manages temporal relationships between events, tracking how events
update, correct, or supersede each other over time.

Enhanced for integration with:
- ChromaDB persistence (raw and merged events)
- Multi-dimensional merging system (temporal dimension)
- Dual-space distance metrics for chain-based retrieval
- Similarity cache for efficient chain analysis
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import uuid

from models.event import Event
from models.merged_event import EventRelationship

logger = logging.getLogger(__name__)


class ChainType(Enum):
    """Types of temporal chains"""
    UPDATE_CHAIN = "update_chain"      # Progressive updates to same concept
    CORRECTION_CHAIN = "correction"    # Error corrections
    ITERATION_CHAIN = "iteration"      # Iterative improvements
    CONVERSATION_CHAIN = "conversation"  # Back-and-forth dialogue
    WORKFLOW_CHAIN = "workflow"        # Step-by-step process


class TemporalChain:
    """
    Manages temporal relationships between events, tracking how information
    evolves, gets corrected, or superseded over time.
    
    Integrated with ChromaDB and multi-dimensional merging system.
    """
    
    def __init__(self, chromadb_store=None, encoder=None, similarity_cache=None, config=None):
        """Initialize the temporal chain manager
        
        Args:
            chromadb_store: Optional ChromaDB store for persistence
            encoder: Optional dual-space encoder for chain similarity
            similarity_cache: Optional similarity cache for efficient analysis
            config: Optional Config object with temporal settings
        """
        self.chains: Dict[str, List[str]] = {}  # chain_id -> ordered list of event IDs
        self.chain_types: Dict[str, ChainType] = {}  # chain_id -> chain type
        self.event_to_chain: Dict[str, str] = {}  # event_id -> chain_id
        self.latest_state: Dict[str, Dict] = {}  # chain_id -> current state
        self.chain_metadata: Dict[str, Dict] = {}  # chain_id -> metadata
        
        # Relationship tracking
        self.supersedes_map: Dict[str, str] = {}  # event_id -> superseded_by
        self.corrections_map: Dict[str, str] = {}  # event_id -> corrected_by
        self.continues_map: Dict[str, str] = {}  # event_id -> continues_from
        
        # Integration with new architecture
        self.chromadb = chromadb_store
        self.encoder = encoder
        self.similarity_cache = similarity_cache
        
        # Load config for temporal settings
        if config:
            self.config = config
        else:
            # Use default config if not provided
            from config import Config
            self.config = Config.from_env()
        
        self.temporal_config = self.config.temporal
        
        # Track raw vs merged events in chains
        self.raw_event_chains: Dict[str, List[str]] = {}  # chain_id -> raw event IDs
        self.merged_event_chains: Dict[str, List[str]] = {}  # chain_id -> merged event IDs
        
        # Chain embeddings for similarity-based chain detection
        self.chain_embeddings: Dict[str, Dict] = {}  # chain_id -> embeddings
        
        # Load existing chains from ChromaDB if available
        if self.chromadb:
            self._load_chains_from_db()
        
    def add_event(self, event: Event, chain_context: Optional[Dict] = None,
                  is_raw: bool = False, merged_id: Optional[str] = None) -> str:
        """
        Add an event to the appropriate temporal chain.
        
        Args:
            event: The event to add
            chain_context: Optional context for chain identification
            is_raw: Whether this is a raw event
            merged_id: Optional merged event ID if this is a raw event
            
        Returns:
            The chain ID the event was added to
        """
        # Identify or create chain
        chain_id = self._identify_chain(event, chain_context)
        
        if chain_id not in self.chains:
            self.chains[chain_id] = []
            self.chain_types[chain_id] = self._determine_chain_type(event, chain_context)
            self.chain_metadata[chain_id] = {
                'created_at': datetime.utcnow().isoformat(),
                'initial_event': event.id,
                'participants': set(),
                'topics': set()
            }
        
        # Determine relationship to previous events
        relationship = self._determine_relationship(event, self.chains[chain_id])
        
        # Update relationship maps
        self._update_relationship_maps(event, self.chains[chain_id], relationship)
        
        # Add event to chain
        self.chains[chain_id].append(event.id)
        self.event_to_chain[event.id] = chain_id
        
        # Track raw vs merged
        if is_raw:
            if chain_id not in self.raw_event_chains:
                self.raw_event_chains[chain_id] = []
            self.raw_event_chains[chain_id].append(event.id)
            
            # Also track merged event if provided
            if merged_id:
                if chain_id not in self.merged_event_chains:
                    self.merged_event_chains[chain_id] = []
                if merged_id not in self.merged_event_chains[chain_id]:
                    self.merged_event_chains[chain_id].append(merged_id)
        else:
            if chain_id not in self.merged_event_chains:
                self.merged_event_chains[chain_id] = []
            self.merged_event_chains[chain_id].append(event.id)
        
        # Update chain metadata
        self._update_chain_metadata(chain_id, event)
        
        # Update latest state
        self._update_latest_state(chain_id, event)
        
        # Update chain embedding if encoder available
        if self.encoder:
            self._update_chain_embedding(chain_id, event)
        
        # Persist to ChromaDB if available
        if self.chromadb:
            self._persist_chain_to_db(chain_id)
        
        logger.info(f"Added event {event.id} to chain {chain_id} with relationship {relationship.value}")
        
        return chain_id
    
    def _identify_chain(self, event: Event, context: Optional[Dict]) -> str:
        """
        Identify which chain this event belongs to based on TEMPORAL PROXIMITY.
        
        For temporal groups, time proximity is the PRIMARY factor, not episode ID.
        This ensures temporal groups actually represent events that happened close in time.
        """
        # Priority 1: Explicit chain ID in context (for specific use cases)
        if context and context.get('chain_id'):
            return context['chain_id']
        
        # Priority 2: TIME-BASED GROUPING (This is the key change!)
        # Find chains within the temporal window
        best_temporal_chain = None
        min_time_diff = timedelta(hours=999)  # Large initial value
        
        for chain_id, events in self.chains.items():
            if events:
                # Get the latest event time from this chain
                last_event_time = self.latest_state.get(chain_id, {}).get('timestamp')
                if last_event_time:
                    try:
                        last_time = datetime.fromisoformat(last_event_time)
                        time_diff = abs(event.created_at - last_time)
                        
                        # Check if within temporal group window
                        if time_diff < timedelta(minutes=self.temporal_config.temporal_group_window):
                            # This event is within the temporal window
                            if time_diff < min_time_diff:
                                min_time_diff = time_diff
                                best_temporal_chain = chain_id
                        
                        # Check for max temporal gap - force new chain if gap is too large
                        elif time_diff > timedelta(minutes=self.temporal_config.max_temporal_gap):
                            # Gap is too large, this chain is not a candidate
                            continue
                            
                    except (ValueError, TypeError):
                        # Invalid timestamp, skip this chain
                        continue
        
        # If we found a chain within the temporal window, use it
        if best_temporal_chain:
            return best_temporal_chain
        
        # Priority 3: Episode-based ONLY if configured to use it for temporal
        if self.temporal_config.use_episode_for_temporal and event.episode_id:
            # Check if this episode already has a RECENT chain
            for existing_chain_id, metadata in self.chain_metadata.items():
                if metadata.get('episode_id') == event.episode_id:
                    # But still check time gap!
                    last_event_time = self.latest_state.get(existing_chain_id, {}).get('timestamp')
                    if last_event_time:
                        try:
                            last_time = datetime.fromisoformat(last_event_time)
                            time_diff = abs(event.created_at - last_time)
                            # Only reuse episode chain if within max gap
                            if time_diff < timedelta(minutes=self.temporal_config.max_temporal_gap):
                                return existing_chain_id
                        except:
                            pass
        
        # Priority 4: Conversation continuation (for chat interfaces)
        if event.five_w1h.where in ["web_chat", "chat", "slack", "teams"]:
            # Look for recent chat chains within conversation window
            for chain_id, events in self.chains.items():
                if self.chain_types.get(chain_id) == ChainType.CONVERSATION_CHAIN:
                    if events:
                        last_event_time = self.latest_state.get(chain_id, {}).get('timestamp')
                        if last_event_time:
                            try:
                                time_diff = event.created_at - datetime.fromisoformat(last_event_time)
                                # Use configured conversation window
                                if time_diff < timedelta(minutes=self.temporal_config.conversation_window):
                                    return chain_id
                            except:
                                pass
        
        # Priority 5: Similarity-based ONLY within time window
        if self.encoder and self.chain_embeddings:
            best_chain = self._find_similar_chain_within_window(event)
            if best_chain:
                return best_chain
            
        
        # Create new temporal chain with timestamp-based ID
        # This ensures each distinct time period gets its own temporal group
        return f"temporal_{event.created_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    
    def _determine_chain_type(self, event: Event, context: Optional[Dict]) -> ChainType:
        """Determine the type of chain based on the event and context"""
        # Explicit type in context
        if context and context.get('chain_type'):
            return ChainType(context['chain_type'])
        
        # Chat interactions
        if event.five_w1h.where == "web_chat":
            return ChainType.CONVERSATION_CHAIN
        
        # Look for workflow indicators
        if event.five_w1h.how and any(word in event.five_w1h.how.lower() 
                                      for word in ['step', 'phase', 'stage']):
            return ChainType.WORKFLOW_CHAIN
        
        # Look for correction indicators
        if event.five_w1h.what and any(word in event.five_w1h.what.lower() 
                                       for word in ['fix', 'correct', 'repair']):
            return ChainType.CORRECTION_CHAIN
        
        # Look for iteration indicators
        if event.five_w1h.what and any(word in event.five_w1h.what.lower() 
                                       for word in ['improve', 'optimize', 'refine']):
            return ChainType.ITERATION_CHAIN
        
        # Default to update chain
        return ChainType.UPDATE_CHAIN
    
    def _determine_relationship(self, event: Event, chain_events: List[str]) -> EventRelationship:
        """Determine how this event relates to previous events in the chain"""
        if not chain_events:
            return EventRelationship.INITIAL
        
        # Analyze event content for relationship indicators
        what_lower = event.five_w1h.what.lower() if event.five_w1h.what else ""
        why_lower = event.five_w1h.why.lower() if event.five_w1h.why else ""
        
        # Check for correction
        if any(word in what_lower for word in ['fix', 'correct', 'repair', 'amend']):
            return EventRelationship.CORRECTION
        
        # Check for continuation
        if any(word in what_lower for word in ['continue', 'resume', 'proceed']):
            return EventRelationship.CONTINUATION
        
        # Check for superseding
        if any(phrase in what_lower for phrase in ['instead of', 'replace', 'new approach']):
            return EventRelationship.SUPERSEDES
        
        # Check for update
        if any(word in what_lower for word in ['update', 'modify', 'change']):
            return EventRelationship.UPDATE
        
        # Default to variation
        return EventRelationship.VARIATION
    
    def _update_relationship_maps(self, event: Event, chain_events: List[str], 
                                 relationship: EventRelationship):
        """Update the relationship tracking maps"""
        if not chain_events:
            return
        
        last_event_id = chain_events[-1] if chain_events else None
        
        if relationship == EventRelationship.SUPERSEDES and last_event_id:
            self.supersedes_map[last_event_id] = event.id
            
        elif relationship == EventRelationship.CORRECTION and last_event_id:
            self.corrections_map[last_event_id] = event.id
            
        elif relationship == EventRelationship.CONTINUATION and last_event_id:
            self.continues_map[event.id] = last_event_id
    
    def _update_chain_metadata(self, chain_id: str, event: Event):
        """Update metadata for a chain"""
        metadata = self.chain_metadata[chain_id]
        
        # Update timestamp
        metadata['last_updated'] = event.created_at.isoformat()
        
        # Add participant
        if event.five_w1h.who:
            if 'participants' not in metadata:
                metadata['participants'] = set()
            metadata['participants'].add(event.five_w1h.who)
        
        # Add topics
        if event.five_w1h.what:
            topic_key = self._extract_topic_key(event.five_w1h.what)
            if 'topics' not in metadata:
                metadata['topics'] = set()
            metadata['topics'].add(topic_key)
        
        # Track episode if present
        if event.episode_id and 'episode_id' not in metadata:
            metadata['episode_id'] = event.episode_id
        
        # Update event count
        metadata['event_count'] = len(self.chains[chain_id])
    
    def _update_latest_state(self, chain_id: str, event: Event):
        """Update the latest state for a chain"""
        self.latest_state[chain_id] = {
            'event_id': event.id,
            'timestamp': event.created_at.isoformat(),
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'when': event.five_w1h.when,
            'where': event.five_w1h.where,
            'why': event.five_w1h.why,
            'how': event.five_w1h.how
        }
    
    def _extract_topic_key(self, text: str) -> str:
        """Extract a topic key from text for chain identification"""
        # Simple approach: use first few significant words
        # This can be enhanced with NLP
        words = text.lower().split()
        significant_words = [w for w in words if len(w) > 3][:3]
        return "_".join(significant_words)
    
    def get_chain_for_event(self, event_id: str) -> Optional[List[str]]:
        """Get the complete chain containing this event"""
        chain_id = self.event_to_chain.get(event_id)
        if chain_id:
            return self.chains.get(chain_id, [])
        return None
    
    def get_chain_context(self, chain_id: str, max_events: int = 10) -> List[Dict]:
        """
        Get the context of a chain for LLM consumption.
        
        Args:
            chain_id: The chain to get context for
            max_events: Maximum number of recent events to include
            
        Returns:
            List of event summaries in chronological order
        """
        if chain_id not in self.chains:
            return []
        
        events = self.chains[chain_id]
        recent_events = events[-max_events:] if len(events) > max_events else events
        
        context = []
        for i, event_id in enumerate(recent_events):
            event_context = {
                'event_id': event_id,
                'position': i + 1,
                'total': len(recent_events)
            }
            
            # Add relationship information
            if event_id in self.supersedes_map:
                event_context['superseded_by'] = self.supersedes_map[event_id]
            if event_id in self.corrections_map:
                event_context['corrected_by'] = self.corrections_map[event_id]
            if event_id in self.continues_map:
                event_context['continues_from'] = self.continues_map[event_id]
            
            # Check if this event supersedes or corrects others
            for other_id, superseded_by in self.supersedes_map.items():
                if superseded_by == event_id:
                    event_context['supersedes'] = other_id
                    break
            
            for other_id, corrected_by in self.corrections_map.items():
                if corrected_by == event_id:
                    event_context['corrects'] = other_id
                    break
            
            context.append(event_context)
        
        return context
    
    def get_latest_valid_state(self, chain_id: str) -> Optional[Dict]:
        """
        Get the latest valid state of a chain, accounting for corrections and supersessions.
        
        Args:
            chain_id: The chain to get the latest state for
            
        Returns:
            The latest valid state, or None if chain doesn't exist
        """
        if chain_id not in self.chains:
            return None
        
        events = self.chains[chain_id]
        
        # Work backwards to find the latest non-superseded, non-corrected event
        for event_id in reversed(events):
            if event_id not in self.supersedes_map and event_id not in self.corrections_map:
                # This event hasn't been superseded or corrected
                return self.latest_state.get(chain_id)
        
        # All events have been superseded or corrected, return the very latest
        return self.latest_state.get(chain_id)
    
    def merge_chains(self, chain_id1: str, chain_id2: str) -> str:
        """
        Merge two chains into one.
        
        Args:
            chain_id1: First chain ID
            chain_id2: Second chain ID
            
        Returns:
            The ID of the merged chain
        """
        if chain_id1 not in self.chains or chain_id2 not in self.chains:
            logger.warning(f"Cannot merge chains {chain_id1} and {chain_id2}: one or both don't exist")
            return chain_id1 if chain_id1 in self.chains else chain_id2
        
        # Merge events (chronologically)
        events1 = self.chains[chain_id1]
        events2 = self.chains[chain_id2]
        
        # Create merged list maintaining chronological order
        # This is simplified - in practice, you'd want to sort by actual timestamps
        merged_events = events1 + events2
        
        # Update chain
        self.chains[chain_id1] = merged_events
        
        # Update event-to-chain mapping
        for event_id in events2:
            self.event_to_chain[event_id] = chain_id1
        
        # Merge metadata
        metadata1 = self.chain_metadata.get(chain_id1, {})
        metadata2 = self.chain_metadata.get(chain_id2, {})
        
        if 'participants' in metadata1 and 'participants' in metadata2:
            metadata1['participants'].update(metadata2['participants'])
        if 'topics' in metadata1 and 'topics' in metadata2:
            metadata1['topics'].update(metadata2['topics'])
        
        metadata1['event_count'] = len(merged_events)
        metadata1['merged_from'] = chain_id2
        metadata1['merged_at'] = datetime.utcnow().isoformat()
        
        # Remove chain2
        del self.chains[chain_id2]
        if chain_id2 in self.chain_types:
            del self.chain_types[chain_id2]
        if chain_id2 in self.chain_metadata:
            del self.chain_metadata[chain_id2]
        if chain_id2 in self.latest_state:
            del self.latest_state[chain_id2]
        
        logger.info(f"Merged chain {chain_id2} into {chain_id1}")
        
        return chain_id1
    
    def _find_similar_chain(self, event: Event, threshold: float = 0.7) -> Optional[str]:
        """
        Find the most similar existing chain using dual-space embeddings.
        
        Args:
            event: The event to match
            threshold: Similarity threshold for chain matching
            
        Returns:
            Chain ID of best match or None
        """
        if not self.encoder or not self.chain_embeddings:
            return None
            
        # Encode the event
        event_embedding = self.encoder.encode({
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'why': event.five_w1h.why
        })
        
        best_chain = None
        best_similarity = 0.0
        
        for chain_id, chain_emb in self.chain_embeddings.items():
            # Compute dual-space similarity
            euc_sim = self._cosine_similarity(
                event_embedding['euclidean_anchor'],
                chain_emb['euclidean']
            )
            
            # Hyperbolic similarity using geodesic distance
            from memory.dual_space_encoder import HyperbolicOperations
            hyp_dist = HyperbolicOperations.geodesic_distance(
                event_embedding['hyperbolic_anchor'],
                chain_emb['hyperbolic']
            )
            hyp_sim = np.exp(-hyp_dist)  # Convert to similarity
            
            # Combine similarities (equal weight for chain matching)
            combined_sim = 0.5 * euc_sim + 0.5 * hyp_sim
            
            if combined_sim > best_similarity and combined_sim > threshold:
                best_similarity = combined_sim
                best_chain = chain_id
        
        return best_chain
    
    def _find_similar_chain_within_window(self, event: Event, threshold: float = 0.7) -> Optional[str]:
        """
        Find the most similar existing chain using dual-space embeddings,
        but ONLY consider chains within the temporal window.
        
        Args:
            event: The event to match
            threshold: Similarity threshold for chain matching
            
        Returns:
            Chain ID of best match or None
        """
        if not self.encoder or not self.chain_embeddings:
            return None
            
        # First filter chains by temporal window
        candidate_chains = []
        for chain_id, events in self.chains.items():
            if events:
                last_event_time = self.latest_state.get(chain_id, {}).get('timestamp')
                if last_event_time:
                    try:
                        last_time = datetime.fromisoformat(last_event_time)
                        time_diff = abs(event.created_at - last_time)
                        # Only consider chains within the max temporal gap
                        if time_diff < timedelta(minutes=self.temporal_config.max_temporal_gap):
                            candidate_chains.append(chain_id)
                    except:
                        pass
        
        if not candidate_chains:
            return None
            
        # Encode the event
        event_embedding = self.encoder.encode({
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'why': event.five_w1h.why
        })
        
        best_chain = None
        best_similarity = 0.0
        
        # Only check candidate chains within temporal window
        for chain_id in candidate_chains:
            if chain_id in self.chain_embeddings:
                chain_emb = self.chain_embeddings[chain_id]
                
                # Compute dual-space similarity
                euc_sim = self._cosine_similarity(
                    event_embedding['euclidean_anchor'],
                    chain_emb['euclidean']
                )
                
                hyp_sim = self._hyperbolic_distance_similarity(
                    event_embedding['hyperbolic_anchor'],
                    chain_emb['hyperbolic']
                )
                
                # Combine similarities (equal weight for chain matching)
                combined_sim = 0.5 * euc_sim + 0.5 * hyp_sim
                
                if combined_sim > best_similarity and combined_sim > threshold:
                    best_similarity = combined_sim
                    best_chain = chain_id
        
        return best_chain
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _update_chain_embedding(self, chain_id: str, event: Event):
        """
        Update the chain's aggregate embedding with new event.
        
        Uses exponential moving average for smooth updates.
        """
        if not self.encoder:
            return
            
        # Encode the new event
        event_emb = self.encoder.encode({
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'why': event.five_w1h.why
        })
        
        if chain_id not in self.chain_embeddings:
            # First event in chain
            self.chain_embeddings[chain_id] = {
                'euclidean': event_emb['euclidean_anchor'].copy(),
                'hyperbolic': event_emb['hyperbolic_anchor'].copy()
            }
        else:
            # Update with exponential moving average
            alpha = 0.3  # Weight for new event
            
            # Euclidean update
            self.chain_embeddings[chain_id]['euclidean'] = (
                (1 - alpha) * self.chain_embeddings[chain_id]['euclidean'] +
                alpha * event_emb['euclidean_anchor']
            )
            
            # Hyperbolic update (using Möbius addition)
            from memory.dual_space_encoder import mobius_add
            weighted_old = (1 - alpha) * self.chain_embeddings[chain_id]['hyperbolic']
            weighted_new = alpha * event_emb['hyperbolic_anchor']
            self.chain_embeddings[chain_id]['hyperbolic'] = mobius_add(
                weighted_old, weighted_new
            )
    
    def _persist_chain_to_db(self, chain_id: str):
        """
        Persist chain information to ChromaDB metadata collection.
        """
        if not self.chromadb:
            return
            
        try:
            # Prepare chain data for storage
            chain_data = {
                'chain_id': chain_id,
                'type': self.chain_types.get(chain_id, ChainType.UPDATE_CHAIN).value,
                'events': self.chains.get(chain_id, []),
                'raw_events': self.raw_event_chains.get(chain_id, []),
                'merged_events': self.merged_event_chains.get(chain_id, []),
                'latest_state': self.latest_state.get(chain_id, {}),
                'metadata': self._serialize_metadata(self.chain_metadata.get(chain_id, {})),
                'supersedes_map': {k: v for k, v in self.supersedes_map.items() 
                                  if k in self.chains.get(chain_id, [])},
                'corrections_map': {k: v for k, v in self.corrections_map.items() 
                                   if k in self.chains.get(chain_id, [])},
                'continues_map': {k: v for k, v in self.continues_map.items() 
                                 if k in self.chains.get(chain_id, [])}
            }
            
            # Store in metadata collection if method exists
            if hasattr(self.chromadb, 'store_metadata'):
                self.chromadb.store_metadata(
                    key=f"temporal_chain_{chain_id}",
                    value=chain_data
                )
                
                # Also store chain embedding if available
                if chain_id in self.chain_embeddings:
                    emb_data = {
                        'euclidean': self.chain_embeddings[chain_id]['euclidean'].tolist(),
                        'hyperbolic': self.chain_embeddings[chain_id]['hyperbolic'].tolist()
                    }
                    self.chromadb.store_metadata(
                        key=f"temporal_chain_embedding_{chain_id}",
                        value=emb_data
                    )
            else:
                logger.debug("ChromaDB store_metadata not available, skipping chain persistence")
                
        except Exception as e:
            logger.warning(f"Failed to persist chain {chain_id} to ChromaDB: {e}")
    
    def _load_chains_from_db(self):
        """
        Load existing temporal chains from ChromaDB.
        """
        if not self.chromadb:
            return
            
        try:
            # Check if the ChromaDB store has the get_all_metadata method
            if not hasattr(self.chromadb, 'get_all_metadata'):
                logger.debug("ChromaDB store does not support get_all_metadata, skipping chain loading")
                return
                
            # Get all temporal chain metadata
            metadata = self.chromadb.get_all_metadata()
            
            for key, value in metadata.items():
                if key.startswith('temporal_chain_') and not key.endswith('_embedding'):
                    chain_id = key.replace('temporal_chain_', '')
                    
                    # Restore chain data
                    self.chains[chain_id] = value.get('events', [])
                    self.chain_types[chain_id] = ChainType(value.get('type', 'update_chain'))
                    self.raw_event_chains[chain_id] = value.get('raw_events', [])
                    self.merged_event_chains[chain_id] = value.get('merged_events', [])
                    self.latest_state[chain_id] = value.get('latest_state', {})
                    
                    # Restore metadata with sets
                    self.chain_metadata[chain_id] = self._deserialize_metadata(
                        value.get('metadata', {})
                    )
                    
                    # Restore relationship maps
                    for event_id, superseded_by in value.get('supersedes_map', {}).items():
                        self.supersedes_map[event_id] = superseded_by
                    for event_id, corrected_by in value.get('corrections_map', {}).items():
                        self.corrections_map[event_id] = corrected_by
                    for event_id, continues_from in value.get('continues_map', {}).items():
                        self.continues_map[event_id] = continues_from
                    
                    # Restore event-to-chain mapping
                    for event_id in self.chains[chain_id]:
                        self.event_to_chain[event_id] = chain_id
                    
                    # Load chain embedding if available
                    emb_key = f"temporal_chain_embedding_{chain_id}"
                    if emb_key in metadata:
                        emb_data = metadata[emb_key]
                        self.chain_embeddings[chain_id] = {
                            'euclidean': np.array(emb_data['euclidean']),
                            'hyperbolic': np.array(emb_data['hyperbolic'])
                        }
                        
            logger.info(f"Loaded {len(self.chains)} temporal chains from ChromaDB")
            
        except AttributeError as e:
            # This is expected if ChromaDB doesn't have metadata methods
            logger.debug(f"ChromaDB metadata methods not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load temporal chains from ChromaDB: {e}")
    
    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Convert sets to lists for JSON serialization."""
        serialized = metadata.copy()
        if 'participants' in serialized and isinstance(serialized['participants'], set):
            serialized['participants'] = list(serialized['participants'])
        if 'topics' in serialized and isinstance(serialized['topics'], set):
            serialized['topics'] = list(serialized['topics'])
        return serialized
    
    def _deserialize_metadata(self, metadata: Dict) -> Dict:
        """Convert lists back to sets after deserialization."""
        deserialized = metadata.copy()
        if 'participants' in deserialized and isinstance(deserialized['participants'], list):
            deserialized['participants'] = set(deserialized['participants'])
        if 'topics' in deserialized and isinstance(deserialized['topics'], list):
            deserialized['topics'] = set(deserialized['topics'])
        return deserialized
    
    def get_chains_for_query(self, query: Dict[str, str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the most relevant chains for a query using dual-space similarity.
        
        Args:
            query: Query fields dictionary
            k: Number of chains to return
            
        Returns:
            List of (chain_id, similarity_score) tuples
        """
        if not self.encoder or not self.chain_embeddings:
            return []
            
        # Encode the query
        query_emb = self.encoder.encode(query)
        
        # Compute similarities to all chains
        chain_scores = []
        
        for chain_id, chain_emb in self.chain_embeddings.items():
            # Dual-space similarity
            euc_sim = self._cosine_similarity(
                query_emb['euclidean_anchor'],
                chain_emb['euclidean']
            )
            
            from memory.dual_space_encoder import HyperbolicOperations
            hyp_dist = HyperbolicOperations.geodesic_distance(
                query_emb['hyperbolic_anchor'],
                chain_emb['hyperbolic']
            )
            hyp_sim = np.exp(-hyp_dist)
            
            # Weight based on query type (can be adjusted)
            lambda_e = 0.6  # Slightly favor Euclidean for chain matching
            lambda_h = 0.4
            
            combined_score = lambda_e * euc_sim + lambda_h * hyp_sim
            chain_scores.append((chain_id, combined_score))
        
        # Sort by score and return top k
        chain_scores.sort(key=lambda x: x[1], reverse=True)
        return chain_scores[:k]
    
    def to_dict(self) -> Dict:
        """Serialize the temporal chain manager to a dictionary"""
        return {
            'chains': self.chains,
            'chain_types': {k: v.value for k, v in self.chain_types.items()},
            'event_to_chain': self.event_to_chain,
            'latest_state': self.latest_state,
            'chain_metadata': {
                k: self._serialize_metadata(v)
                for k, v in self.chain_metadata.items()
            },
            'raw_event_chains': self.raw_event_chains,
            'merged_event_chains': self.merged_event_chains,
            'supersedes_map': self.supersedes_map,
            'corrections_map': self.corrections_map,
            'continues_map': self.continues_map,
            'chain_embeddings': {
                k: {
                    'euclidean': v['euclidean'].tolist(),
                    'hyperbolic': v['hyperbolic'].tolist()
                }
                for k, v in self.chain_embeddings.items()
            }
        }
    
    def from_dict(self, data: Dict):
        """Load the temporal chain manager from a dictionary"""
        self.chains = data.get('chains', {})
        self.chain_types = {
            k: ChainType(v) for k, v in data.get('chain_types', {}).items()
        }
        self.event_to_chain = data.get('event_to_chain', {})
        self.latest_state = data.get('latest_state', {})
        
        # Restore metadata with sets
        self.chain_metadata = {}
        for k, v in data.get('chain_metadata', {}).items():
            self.chain_metadata[k] = self._deserialize_metadata(v)
        
        self.raw_event_chains = data.get('raw_event_chains', {})
        self.merged_event_chains = data.get('merged_event_chains', {})
        self.supersedes_map = data.get('supersedes_map', {})
        self.corrections_map = data.get('corrections_map', {})
        self.continues_map = data.get('continues_map', {})
        
        # Restore chain embeddings
        self.chain_embeddings = {}
        for k, v in data.get('chain_embeddings', {}).items():
            self.chain_embeddings[k] = {
                'euclidean': np.array(v['euclidean']),
                'hyperbolic': np.array(v['hyperbolic'])
            }
