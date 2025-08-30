"""
Multi-dimensional merger that creates different merge groups based on various aspects.

A single raw event can belong to multiple merge groups:
- Actor merge: Groups all events from the same WHO
- Temporal merge: Groups conversation threads and sequential actions
- Conceptual merge: Groups events with similar WHY/HOW (goals, methods)
- Spatial merge: Groups events from the same WHERE
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np
import uuid

from models.event import Event
from models.merged_event import MergedEvent, EventRelationship
from models.merge_types import (
    MergeType, MergeStrategy, MERGE_STRATEGIES, MultiMergeTracker
)
from memory.smart_merger import SmartMerger

logger = logging.getLogger(__name__)


class MultiDimensionalMerger:
    """
    Manages multiple merge dimensions for comprehensive memory organization.
    """
    
    def __init__(self, chromadb_store=None):
        """
        Initialize the multi-dimensional merger.
        
        Args:
            chromadb_store: ChromaDB store instance for persistence
        """
        self.chromadb = chromadb_store
        self.strategies = MERGE_STRATEGIES
        self.tracker = MultiMergeTracker()
        
        # Original smart merger for similarity calculations
        self.smart_merger = SmartMerger()
        
        # In-memory cache of merge groups
        self.merge_groups = {
            MergeType.ACTOR: {},
            MergeType.TEMPORAL: {},
            MergeType.CONCEPTUAL: {},
            MergeType.SPATIAL: {}
        }
        
        # Load existing merges from ChromaDB if available
        if self.chromadb:
            self._load_existing_merges()
    
    def _load_existing_merges(self):
        """Load existing merge groups from ChromaDB"""
        # TODO: Implement loading from ChromaDB collections
        pass
    
    def process_new_event(self, event: Event, embeddings: Dict[str, np.ndarray]) -> Dict[MergeType, str]:
        """
        Process a new event and assign it to appropriate merge groups.
        
        Args:
            event: The new event to process
            embeddings: Event embeddings (euclidean_anchor, hyperbolic_anchor)
            
        Returns:
            Dictionary mapping merge types to merge IDs
        """
        merge_assignments = {}
        event_data = self._event_to_dict(event)
        
        # Process each merge type
        for merge_type, strategy in self.strategies.items():
            merge_id = self._find_or_create_merge_group(
                event, event_data, embeddings, merge_type, strategy
            )
            
            if merge_id:
                merge_assignments[merge_type] = merge_id
                self.tracker.add_merge_membership(event.id, merge_type, merge_id)
                
                # Update the merge group in ChromaDB
                if self.chromadb:
                    self._update_merge_in_db(merge_type, merge_id, event, event_data)
        
        logger.info(f"Event {event.id} assigned to merge groups: {merge_assignments}")
        return merge_assignments
    
    def _find_or_create_merge_group(self, event: Event, event_data: Dict, 
                                   embeddings: Dict, merge_type: MergeType, 
                                   strategy: MergeStrategy) -> Optional[str]:
        """
        Find an existing merge group or create a new one for the event.
        
        Returns:
            Merge ID if a group was found or created, None otherwise
        """
        # Get the merge key for this event
        merge_key = strategy.get_merge_key(event_data)
        if not merge_key:
            return None  # Can't merge without a key
        
        # Check existing groups of this type
        existing_groups = self.merge_groups.get(merge_type, {})
        
        # Find best matching group
        best_match = None
        best_distance = float('inf')
        
        for group_id, group_data in existing_groups.items():
            # Check if this group matches the merge criteria
            group_key = group_data.get('key')
            
            # For actor merges, must match the actor key exactly
            if merge_type == MergeType.ACTOR and group_key != merge_key:
                continue
            
            # Calculate distance to group centroid or representative
            if 'centroid_embedding' in group_data:
                distance = self._calculate_distance(
                    embeddings['euclidean_anchor'],
                    group_data['centroid_embedding']
                )
                
                # Check if should merge with this group
                group_events = group_data.get('events', [])
                if strategy.should_merge_with_group(event_data, group_key, 
                                                   group_events, distance):
                    if distance < best_distance:
                        best_match = group_id
                        best_distance = distance
        
        if best_match:
            # Add to existing group
            self._add_to_merge_group(merge_type, best_match, event, event_data, embeddings)
            return best_match
        else:
            # Create new group
            return self._create_new_merge_group(merge_type, merge_key, event, 
                                              event_data, embeddings)
    
    def _create_new_merge_group(self, merge_type: MergeType, merge_key: str,
                               event: Event, event_data: Dict, 
                               embeddings: Dict) -> str:
        """Create a new merge group"""
        merge_id = f"{merge_type.value}_{uuid.uuid4().hex[:8]}"
        
        # Create merged event
        merged_event = MergedEvent(
            id=merge_id,
            base_event_id=event.id
        )
        
        # Add the event to the merged event
        merged_event.add_raw_event(event.id, event_data, EventRelationship.INITIAL)
        
        # Store in memory
        if merge_type not in self.merge_groups:
            self.merge_groups[merge_type] = {}
            
        self.merge_groups[merge_type][merge_id] = {
            'key': merge_key,
            'merged_event': merged_event,
            'events': [event_data],
            'centroid_embedding': embeddings['euclidean_anchor'].copy(),
            'created_at': datetime.utcnow(),
            'last_updated': datetime.utcnow()
        }
        
        logger.info(f"Created new {merge_type.value} merge group: {merge_id}")
        return merge_id
    
    def _add_to_merge_group(self, merge_type: MergeType, merge_id: str,
                           event: Event, event_data: Dict, embeddings: Dict):
        """Add an event to an existing merge group"""
        group = self.merge_groups[merge_type][merge_id]
        merged_event = group['merged_event']
        
        # Determine relationship
        relationship = self.smart_merger._determine_relationship(merged_event, event)
        
        # Add to merged event
        merged_event.add_raw_event(event.id, event_data, relationship)
        
        # Update group data
        group['events'].append(event_data)
        group['last_updated'] = datetime.utcnow()
        
        # Update centroid (simple average for now)
        n = len(group['events'])
        group['centroid_embedding'] = (
            (group['centroid_embedding'] * (n - 1) + embeddings['euclidean_anchor']) / n
        )
        
        logger.info(f"Added event {event.id} to {merge_type.value} merge group {merge_id}")
    
    def _update_merge_in_db(self, merge_type: MergeType, merge_id: str, 
                           event: Event, event_data: Dict):
        """Update merge group in ChromaDB"""
        if not self.chromadb:
            return
            
        collection = self._get_collection_for_type(merge_type)
        if not collection:
            return
            
        group = self.merge_groups[merge_type][merge_id]
        merged_event = group['merged_event']
        
        # Prepare metadata
        metadata = {
            'merge_type': merge_type.value,
            'merge_key': group['key'],
            'merge_count': merged_event.merge_count,
            'created_at': group['created_at'].isoformat(),
            'last_updated': group['last_updated'].isoformat(),
            'raw_event_ids': ','.join(merged_event.raw_event_ids),
            
            # Latest state of components
            **merged_event.get_latest_state()
        }
        
        # Store or update in ChromaDB
        try:
            collection.upsert(
                ids=[merge_id],
                embeddings=[group['centroid_embedding'].tolist()],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Failed to update merge in DB: {e}")
    
    def _get_collection_for_type(self, merge_type: MergeType):
        """Get the ChromaDB collection for a merge type"""
        if not self.chromadb:
            return None
            
        mapping = {
            MergeType.ACTOR: self.chromadb.actor_merges_collection,
            MergeType.TEMPORAL: self.chromadb.temporal_merges_collection,
            MergeType.CONCEPTUAL: self.chromadb.conceptual_merges_collection,
            MergeType.SPATIAL: self.chromadb.spatial_merges_collection
        }
        return mapping.get(merge_type)
    
    def _calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between embeddings"""
        return float(np.linalg.norm(embedding1 - embedding2))
    
    def _event_to_dict(self, event: Event) -> Dict:
        """Convert event to dictionary for processing"""
        return {
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'when': event.five_w1h.when,
            'where': event.five_w1h.where,
            'why': event.five_w1h.why,
            'how': event.five_w1h.how,
            'timestamp': event.created_at
        }
    
    def get_merges_for_event(self, event_id: str) -> Dict[MergeType, MergedEvent]:
        """
        Get all merge groups that contain a specific event.
        
        Args:
            event_id: ID of the raw event
            
        Returns:
            Dictionary mapping merge types to merged events
        """
        merges = {}
        merge_groups = self.tracker.get_merge_groups_for_raw(event_id)
        
        for merge_type, merge_id in merge_groups:
            if merge_type in self.merge_groups and merge_id in self.merge_groups[merge_type]:
                group = self.merge_groups[merge_type][merge_id]
                merges[merge_type] = group['merged_event']
        
        return merges
    
    def query_merges(self, query: Dict[str, str], merge_types: Optional[List[MergeType]] = None,
                    k: int = 10) -> List[Tuple[MergedEvent, float, MergeType]]:
        """
        Query across multiple merge dimensions.
        
        Args:
            query: Query fields (5W1H)
            merge_types: Types to search (None = all)
            k: Number of results per type
            
        Returns:
            List of (merged_event, score, merge_type) tuples
        """
        results = []
        
        if merge_types is None:
            merge_types = list(MergeType)
        
        for merge_type in merge_types:
            # TODO: Implement proper querying with embeddings
            # For now, return all merges of the requested types
            type_groups = self.merge_groups.get(merge_type, {})
            for merge_id, group in type_groups.items():
                merged_event = group['merged_event']
                # Simple relevance scoring based on component matches
                score = self._calculate_relevance_score(merged_event, query)
                results.append((merged_event, score, merge_type))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _calculate_relevance_score(self, merged_event: MergedEvent, query: Dict) -> float:
        """Calculate simple relevance score for a merged event"""
        score = 0.0
        latest_state = merged_event.get_latest_state()
        
        for field, query_value in query.items():
            if query_value and field in latest_state:
                event_value = latest_state[field]
                if event_value:
                    # Simple containment check
                    if query_value.lower() in str(event_value).lower():
                        score += 1.0
                    # Partial match
                    elif any(word in str(event_value).lower() 
                            for word in query_value.lower().split()):
                        score += 0.5
        
        return score