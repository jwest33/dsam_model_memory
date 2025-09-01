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
from datetime import datetime, timezone
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
        """Load existing merge groups from ChromaDB collections"""
        if not self.chromadb:
            return
        
        logger.info("Loading existing multi-dimensional merge groups from ChromaDB...")
        
        try:
            # Map of merge types to their collections
            collection_map = {
                MergeType.ACTOR: self.chromadb.actor_merges_collection,
                MergeType.TEMPORAL: self.chromadb.temporal_merges_collection,
                MergeType.CONCEPTUAL: self.chromadb.conceptual_merges_collection,
                MergeType.SPATIAL: self.chromadb.spatial_merges_collection
            }
            
            for merge_type, collection in collection_map.items():
                if collection:
                    try:
                        # Get all merge groups from this collection
                        logger.debug(f"Querying {merge_type.value} collection...")
                        results = collection.get(include=['metadatas', 'embeddings'])
                        logger.debug(f"Got {len(results.get('ids', [])) if results else 0} results from {merge_type.value} collection")
                        
                        if results and 'ids' in results and results['ids']:
                            for i, merge_id in enumerate(results['ids']):
                                metadata = results['metadatas'][i] if results['metadatas'] is not None and i < len(results['metadatas']) else {}
                                embedding = results['embeddings'][i] if results['embeddings'] is not None and i < len(results['embeddings']) else None
                                
                                # Reconstruct the merge group
                                merged_event = MergedEvent(
                                    id=merge_id,
                                    base_event_id=metadata.get('base_event_id', '')
                                )
                                
                                # Parse raw event IDs
                                raw_ids_str = metadata.get('raw_event_ids', '')
                                if raw_ids_str:
                                    raw_ids = raw_ids_str.split(',')
                                    for raw_id in raw_ids:
                                        merged_event.raw_event_ids.add(raw_id.strip())
                                
                                # Load the component data from metadata
                                # Always add a synthetic event with latest values for display
                                # Even if some fields are empty, this ensures we have something to show
                                merge_key = metadata.get('merge_key', '')
                                
                                # Parse the merge key to extract meaningful data
                                # Use all_who if available for cross-actor groups
                                all_who = metadata.get('all_who', '')
                                who_val = metadata.get('latest_who', '')
                                what_val = metadata.get('latest_what', '')
                                where_val = metadata.get('latest_where', '')
                                why_val = metadata.get('latest_why', '')
                                how_val = metadata.get('latest_how', '')
                                
                                # Use merge key to fill in missing values based on merge type
                                if merge_type == MergeType.ACTOR and not who_val and 'actor' in merge_key:
                                    who_val = merge_key.replace('_actor', '').replace('actor_', '')
                                elif merge_type == MergeType.SPATIAL and not where_val and 'location' in merge_key:
                                    where_val = merge_key.replace('location_', '')
                                elif merge_type == MergeType.CONCEPTUAL and not why_val and 'concept' in merge_key:
                                    why_val = merge_key.replace('concept_', '')
                                elif merge_type == MergeType.TEMPORAL and not what_val:
                                    what_val = merge_key.replace('temporal_', '')
                                
                                # Parse the latest_when timestamp
                                latest_when_str = metadata.get('latest_when', datetime.now(timezone.utc).isoformat())
                                try:
                                    event_timestamp = datetime.fromisoformat(latest_when_str.replace('Z', '+00:00'))
                                except:
                                    event_timestamp = datetime.now(timezone.utc)
                                
                                # For cross-actor groups, add synthetic events for each actor
                                if all_who and ',' in all_who:
                                    # Cross-actor group - add events for each actor
                                    who_list = [w.strip() for w in all_who.split(',') if w.strip()]
                                    for idx, who_actor in enumerate(who_list):
                                        synthetic_event_data = {
                                            'who': who_actor,
                                            'what': what_val or merge_key,
                                            'when': latest_when_str,
                                            'where': where_val or 'unknown',
                                            'why': why_val or merge_key,
                                            'how': how_val or 'unknown',
                                            'timestamp': event_timestamp
                                        }
                                        merged_event.add_raw_event(
                                            f"{merge_id}_{idx}",  # Simplified ID without synthetic_ prefix
                                            synthetic_event_data,
                                            EventRelationship.INITIAL
                                        )
                                else:
                                    # Single actor or no actor info
                                    synthetic_event_data = {
                                        'who': who_val or 'unknown',
                                        'what': what_val or merge_key,
                                        'when': latest_when_str,
                                        'where': where_val or 'unknown',
                                        'why': why_val or merge_key,
                                        'how': how_val or 'unknown',
                                        'timestamp': event_timestamp
                                    }
                                    # Add this event to populate the variants
                                    merged_event.add_raw_event(
                                        merge_id,  # Use merge_id directly without prefix
                                        synthetic_event_data,
                                        EventRelationship.INITIAL
                                    )
                                
                                # Now try to load the actual raw events to build the full timeline
                                if self.chromadb and merged_event.raw_event_ids:
                                    try:
                                        # Try both collections since events might be in either
                                        raw_collection = None
                                        events_collection = None
                                        try:
                                            raw_collection = self.chromadb.client.get_collection('raw_events')
                                        except:
                                            pass
                                        try:
                                            events_collection = self.chromadb.client.get_collection('events')
                                        except:
                                            pass
                                        
                                        # Get all event IDs except synthetic ones
                                        raw_ids_to_load = [rid for rid in merged_event.raw_event_ids if not rid.startswith('synthetic_')]
                                        
                                        if raw_ids_to_load and (raw_collection or events_collection):
                                            # Try raw_events collection first
                                            raw_event_results = {'ids': [], 'metadatas': []}
                                            if raw_collection:
                                                try:
                                                    raw_results = raw_collection.get(
                                                        ids=raw_ids_to_load,
                                                        include=['metadatas']
                                                    )
                                                    if raw_results['ids']:
                                                        raw_event_results['ids'].extend(raw_results['ids'])
                                                        raw_event_results['metadatas'].extend(raw_results['metadatas'] or [])
                                                except:
                                                    pass
                                            
                                            # Try events collection for any missing IDs
                                            if events_collection:
                                                missing_ids = [rid for rid in raw_ids_to_load if rid not in raw_event_results['ids']]
                                                if missing_ids:
                                                    try:
                                                        event_results = events_collection.get(
                                                            ids=missing_ids,
                                                            include=['metadatas']
                                                        )
                                                        if event_results['ids']:
                                                            raw_event_results['ids'].extend(event_results['ids'])
                                                            raw_event_results['metadatas'].extend(event_results['metadatas'] or [])
                                                    except:
                                                        pass
                                            
                                            # Add each raw event to build the component timeline
                                            for i, raw_id in enumerate(raw_event_results['ids']):
                                                if raw_event_results['metadatas'] is not None and i < len(raw_event_results['metadatas']):
                                                    raw_metadata = raw_event_results['metadatas'][i]
                                                    # Parse timestamp - prefer 'when' field over 'timestamp' field
                                                    when_str = raw_metadata.get('when', '')
                                                    timestamp_str = raw_metadata.get('timestamp', '')
                                                    
                                                    # Try to parse 'when' field first as it contains the actual event time
                                                    timestamp = None
                                                    if when_str:
                                                        try:
                                                            timestamp = datetime.fromisoformat(when_str.replace('Z', '+00:00'))
                                                        except:
                                                            pass
                                                    
                                                    # Fall back to timestamp field if when didn't parse
                                                    if timestamp is None:
                                                        if isinstance(timestamp_str, str):
                                                            try:
                                                                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                                            except:
                                                                timestamp = datetime.now(timezone.utc)
                                                        else:
                                                            timestamp = timestamp_str if isinstance(timestamp_str, datetime) else datetime.now(timezone.utc)
                                                    
                                                    raw_event_data = {
                                                        'who': raw_metadata.get('who', ''),
                                                        'what': raw_metadata.get('what', ''),
                                                        'when': raw_metadata.get('when', ''),
                                                        'where': raw_metadata.get('where', ''),
                                                        'why': raw_metadata.get('why', ''),
                                                        'how': raw_metadata.get('how', ''),
                                                        'timestamp': timestamp
                                                    }
                                                    # Add to merged event to build variations
                                                    merged_event.add_raw_event(
                                                        raw_id,  # Use ID as-is
                                                        raw_event_data,
                                                        EventRelationship.VARIATION
                                                    )
                                    except Exception as e:
                                        logger.debug(f"Could not load raw events for timeline: {e}")
                                
                                # Store in memory
                                if merge_type not in self.merge_groups:
                                    self.merge_groups[merge_type] = {}
                                
                                # Debug: Check if we're overwriting
                                #if merge_id in self.merge_groups[merge_type]:
                                #    logger.warning(f"Overwriting existing {merge_type.value} group: {merge_id}")
                                
                                # Parse timestamps from metadata - use latest_when for timestamps
                                latest_when = metadata.get('latest_when', '')
                                if latest_when:
                                    try:
                                        event_timestamp = datetime.fromisoformat(latest_when.replace('Z', '+00:00'))
                                    except:
                                        event_timestamp = datetime.now(timezone.utc)
                                else:
                                    event_timestamp = datetime.now(timezone.utc)
                                
                                self.merge_groups[merge_type][merge_id] = {
                                    'key': metadata.get('merge_key', ''),
                                    'merged_event': merged_event,
                                    'events': [],  # Will be populated as needed
                                    'centroid_embedding': np.array(embedding) if embedding is not None else None,
                                    'created_at': event_timestamp,  # Use event timestamp
                                    'last_updated': event_timestamp,  # Use event timestamp
                                    'system_created_at': datetime.fromisoformat(metadata['created_at']) if 'created_at' in metadata else datetime.now(timezone.utc),
                                    'system_last_updated': datetime.fromisoformat(metadata['last_updated']) if 'last_updated' in metadata else datetime.now(timezone.utc)
                                }
                                
                                # Update tracker
                                for raw_id in merged_event.raw_event_ids:
                                    self.tracker.add_merge_membership(raw_id, merge_type, merge_id)
                            
                            logger.info(f"Loaded {len(results['ids'])} {merge_type.value} merge groups from ChromaDB")
                            logger.info(f"Actually stored in memory: {len(self.merge_groups.get(merge_type, {}))} {merge_type.value} groups")
                            # Debug: show the IDs
                            if self.merge_groups.get(merge_type):
                                logger.debug(f"{merge_type.value} group IDs: {list(self.merge_groups[merge_type].keys())}")
                    except Exception as e:
                        logger.warning(f"Could not load {merge_type.value} merges: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading existing merge groups: {e}")
    
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
                # Use event ID directly without prefix
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
        Uses semantic similarity instead of exact key matching.
        
        Returns:
            Merge ID if a group was found or created, None otherwise
        """
        # Get the merge key for this event (used for creating new groups)
        merge_key = strategy.get_merge_key(event_data)
        if not merge_key:
            return None  # Can't merge without a key
        
        # Check existing groups of this type
        existing_groups = self.merge_groups.get(merge_type, {})
        
        # Find best matching group based on semantic similarity
        best_match = None
        best_distance = float('inf')
        
        # For all groups, use semantic similarity
        for group_id, group_data in existing_groups.items():
            # Calculate distance to group centroid
            if 'centroid_embedding' in group_data and 'euclidean_anchor' in embeddings:
                distance = self._calculate_distance(
                    embeddings['euclidean_anchor'],
                    group_data['centroid_embedding']
                )
                
                # For ACTOR type ONLY, check actor compatibility
                if merge_type == MergeType.ACTOR:
                    # Get the normalized actor types for comparison
                    event_actor = event_data.get('who', '').lower().strip()
                    
                    # Normalize to standard categories
                    user_types = {'user', 'human', 'person', 'customer', 'client'}
                    assistant_types = {'assistant', 'ai', 'agent', 'bot', 'system'}
                    
                    # Determine event actor type
                    if event_actor in user_types:
                        event_actor_type = 'user'
                    elif event_actor in assistant_types:
                        event_actor_type = 'assistant'
                    else:
                        event_actor_type = event_actor  # Keep raw value for unknown actors
                    
                    # Get the primary actor type of the group (first event's actor type)
                    group_actor_type = None
                    if group_data.get('events'):
                        first_actor = group_data['events'][0].get('who', '').lower().strip()
                        if first_actor in user_types:
                            group_actor_type = 'user'
                        elif first_actor in assistant_types:
                            group_actor_type = 'assistant'
                        else:
                            group_actor_type = first_actor
                    
                    # Skip if incompatible actor types
                    if group_actor_type and event_actor_type != group_actor_type:
                        continue  # Don't merge different actor types
                    
                    # If same actor type, use similarity distance for merging
                
                # For SPATIAL type, check normalized location match
                if merge_type == MergeType.SPATIAL:
                    event_where = event_data.get('where', '').lower().strip()
                    if event_where:
                        # Check if normalized locations match
                        event_normalized = self._normalize_location(event_where)
                        for group_event in group_data.get('events', []):
                            group_where = group_event.get('where', '').lower().strip()
                            if group_where:
                                group_normalized = self._normalize_location(group_where)
                                if event_normalized == group_normalized:
                                    distance = 0.0  # Perfect match for same normalized location
                                    break
                
                # For CONCEPTUAL type, check if same concept
                if merge_type == MergeType.CONCEPTUAL:
                    event_why = event_data.get('why', '').lower().strip()
                    event_how = event_data.get('how', '').lower().strip()
                    for group_event in group_data.get('events', []):
                        group_why = group_event.get('why', '').lower().strip()
                        group_how = group_event.get('how', '').lower().strip()
                        # If either why or how matches exactly, consider it very similar
                        if (event_why and event_why == group_why) or (event_how and event_how == group_how):
                            distance = min(distance, 0.1)  # Very close match
                            break
                
                # Check if should merge with this group based on distance
                group_events = group_data.get('events', [])
                if strategy.should_merge_with_group(event_data, group_data.get('key'), 
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
        
        # Add the event to the merged event using its original ID
        merged_event.add_raw_event(event.id, event_data, EventRelationship.INITIAL)
        
        # Store in memory
        if merge_type not in self.merge_groups:
            self.merge_groups[merge_type] = {}
            
        # Parse event's when timestamp for created_at and last_updated
        event_when = event_data.get('when', '')
        if event_when:
            try:
                event_timestamp = datetime.fromisoformat(event_when.replace('Z', '+00:00'))
            except:
                event_timestamp = datetime.now(timezone.utc)
        else:
            event_timestamp = datetime.now(timezone.utc)
            
        self.merge_groups[merge_type][merge_id] = {
            'key': merge_key,
            'merged_event': merged_event,
            'events': [event_data],
            'centroid_embedding': embeddings['euclidean_anchor'].copy(),
            'created_at': event_timestamp,  # Use event's when timestamp
            'last_updated': event_timestamp,  # Use event's when timestamp
            'system_created_at': datetime.now(timezone.utc),  # Keep system time for merge stats
            'system_last_updated': datetime.now(timezone.utc)  # Keep system time for merge stats
        }
        
        logger.info(f"Created new {merge_type.value} merge group: {merge_id}")
        
        # Save to ChromaDB
        self._update_merge_in_db(merge_type, merge_id, event, event_data)
        
        return merge_id
    
    def _add_to_merge_group(self, merge_type: MergeType, merge_id: str,
                           event: Event, event_data: Dict, embeddings: Dict):
        """Add an event to an existing merge group"""
        group = self.merge_groups[merge_type][merge_id]
        merged_event = group['merged_event']
        
        # Determine relationship
        relationship = self.smart_merger._determine_relationship(merged_event, event)
        
        # Add to merged event using original ID
        merged_event.add_raw_event(event.id, event_data, relationship)
        
        # Update group data
        group['events'].append(event_data)
        
        # Update last_updated to the latest event's when timestamp
        event_when = event_data.get('when', '')
        if event_when:
            try:
                event_timestamp = datetime.fromisoformat(event_when.replace('Z', '+00:00'))
                # Only update if this event is newer
                # Use a very old date with timezone awareness as default
                default_old = datetime(1970, 1, 1, tzinfo=timezone.utc)
                last_updated = group.get('last_updated', default_old)
                # Ensure timezone consistency
                if event_timestamp.tzinfo is None:
                    event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)
                if last_updated.tzinfo is None:
                    last_updated = last_updated.replace(tzinfo=timezone.utc)
                if event_timestamp > last_updated:
                    group['last_updated'] = event_timestamp
            except:
                pass
        
        # Also update created_at if this event is older
        if event_when:
            try:
                event_timestamp = datetime.fromisoformat(event_when.replace('Z', '+00:00'))
                # Use a very future date with timezone awareness as default
                default_future = datetime(2999, 12, 31, tzinfo=timezone.utc)
                created_at = group.get('created_at', default_future)
                # Ensure timezone consistency
                if event_timestamp.tzinfo is None:
                    event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if event_timestamp < created_at:
                    group['created_at'] = event_timestamp
            except:
                pass
                
        # Always update system timestamps
        group['system_last_updated'] = datetime.now(timezone.utc)
        
        # Update centroid (simple average for now)
        n = len(group['events'])
        group['centroid_embedding'] = (
            (group['centroid_embedding'] * (n - 1) + embeddings['euclidean_anchor']) / n
        )
        
        logger.info(f"Added event {event.id} to {merge_type.value} merge group {merge_id}")
        
        # Save updates to ChromaDB
        self._update_merge_in_db(merge_type, merge_id, event, event_data)
    
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
        
        # Get latest state
        latest_state = merged_event.get_latest_state()
        
        # Get all unique WHO values for cross-actor support
        all_who_values = []
        for who_key, variants in merged_event.who_variants.items():
            for variant in variants:
                if variant.value and variant.value not in all_who_values:
                    all_who_values.append(variant.value)
        
        # Prepare metadata
        metadata = {
            'merge_type': merge_type.value,
            'merge_key': group['key'],
            'merge_count': merged_event.merge_count,
            'created_at': group['created_at'].isoformat(),
            'last_updated': group['last_updated'].isoformat(),
            'raw_event_ids': ','.join(merged_event.raw_event_ids),
            'base_event_id': merged_event.base_event_id,
            
            # Store ALL who values for cross-actor groups
            'all_who': ','.join(all_who_values) if all_who_values else '',
            
            # Latest state of components with explicit keys
            'latest_who': latest_state.get('who', ''),
            'latest_what': latest_state.get('what', ''),
            'latest_when': latest_state.get('when', ''),
            'latest_where': latest_state.get('where', ''),
            'latest_why': latest_state.get('why', ''),
            'latest_how': latest_state.get('how', '')
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
    
    def _normalize_location(self, location: str) -> str:
        """Clean location text without normalization"""
        # Just clean the text, let semantic similarity handle grouping
        return location.lower().strip()
    
    def _event_to_dict(self, event: Event) -> Dict:
        """Convert event to dictionary for processing"""
        # Parse the 'when' field to get the actual event timestamp
        timestamp = event.created_at  # Default to created_at
        if event.five_w1h.when:
            try:
                timestamp = datetime.fromisoformat(event.five_w1h.when.replace('Z', '+00:00'))
            except:
                pass  # Keep using created_at if when doesn't parse
        
        return {
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'when': event.five_w1h.when,
            'where': event.five_w1h.where,
            'why': event.five_w1h.why,
            'how': event.five_w1h.how,
            'timestamp': timestamp
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
    
    def get_merge_groups_details_for_event(self, event_id: str) -> Dict[str, Dict]:
        """
        Get detailed information about all merge groups a raw event belongs to.
        
        Args:
            event_id: ID of the raw event (can be with or without 'raw_' prefix)
            
        Returns:
            Dictionary mapping merge type strings to group details
        """
        # Use event ID directly
        details = {}
        
        # Look up event ID
        for test_id in [event_id]:
            merge_groups = self.tracker.get_merge_groups_for_raw(test_id)
            
            for merge_type, merge_id in merge_groups:
                if merge_type in self.merge_groups and merge_id in self.merge_groups[merge_type]:
                    group = self.merge_groups[merge_type][merge_id]
                    merged_event = group.get('merged_event')
                    
                    if merged_event:
                        # Get all raw events in this merge group
                        raw_events_in_group = list(merged_event.raw_event_ids)
                        
                        details[merge_type.value] = {
                            'merge_id': merge_id,
                            'merge_type': merge_type.value,
                            'merge_count': merged_event.merge_count,
                            'raw_event_ids': raw_events_in_group,
                            'latest_state': merged_event.get_latest_state() if hasattr(merged_event, 'get_latest_state') else {},
                            'who_variants': {k: len(v) for k, v in merged_event.who_variants.items()},
                            'what_variants': {k: len(v) for k, v in merged_event.what_variants.items()},
                            'when_timeline': [
                                {'timestamp': tp.timestamp.isoformat(), 'description': tp.description}
                                for tp in merged_event.when_timeline
                            ],
                            'where_locations': merged_event.where_locations if hasattr(merged_event, 'where_locations') else {},
                            'why_variants': {k: len(v) for k, v in merged_event.why_variants.items()},
                            'how_variants': merged_event.how_methods if hasattr(merged_event, 'how_methods') else {},
                            'created_at': group.get('created_at', datetime.now(timezone.utc)).isoformat() if 'created_at' in group else '',
                            'last_updated': group.get('last_updated', datetime.now(timezone.utc)).isoformat() if 'last_updated' in group else ''
                        }
            
            # If we found any groups, don't continue checking other IDs
            if details:
                break
        
        return details
    
    def get_all_merge_groups(self) -> Dict[str, Dict]:
        """
        Get all merge groups across all dimensions.
        
        Returns:
            Dictionary with merge_id as key and merge info as value
        """
        all_groups = {}
        
        for merge_type in self.merge_groups:
            for merge_id, group_data in self.merge_groups[merge_type].items():
                if 'merged_event' in group_data:
                    merged_event = group_data['merged_event']
                    all_groups[merge_id] = {
                        'type': merge_type.value,
                        'raw_event_ids': list(merged_event.raw_event_ids),
                        'merge_count': merged_event.merge_count,
                        'key': group_data.get('key', ''),
                        'created_at': group_data.get('created_at', datetime.now(timezone.utc)).isoformat() if 'created_at' in group_data else '',
                        'last_updated': group_data.get('last_updated', datetime.now(timezone.utc)).isoformat() if 'last_updated' in group_data else ''
                    }
        
        return all_groups
    
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
