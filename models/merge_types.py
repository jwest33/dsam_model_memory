"""
Multi-dimensional merge types for different memory aspects.

Each raw memory can contribute to multiple merge types:
- Actor merges: Group by WHO (person-centric view)
- Temporal merges: Group by WHAT/WHEN (conversation threads, sequential actions)
- Conceptual merges: Group by WHY/HOW (goal-oriented, method-based)
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np


class MergeType(Enum):
    """Types of memory merges based on different dimensions"""
    ACTOR = "actor"  # WHO-based merging (all memories from same actor)
    TEMPORAL = "temporal"  # WHAT/WHEN-based (conversation threads, sequences)
    CONCEPTUAL = "conceptual"  # WHY/HOW-based (goals, methods, abstract concepts)
    SPATIAL = "spatial"  # WHERE-based (location-centric grouping)
    HYBRID = "hybrid"  # Multi-dimensional (combines multiple aspects)


@dataclass
class MergeStrategy:
    """Strategy for how to merge events of a specific type"""
    merge_type: MergeType
    primary_field: str  # Primary 5W1H field for this merge type
    secondary_fields: List[str] = field(default_factory=list)
    distance_threshold: float = 0.15  # Default threshold
    time_window_hours: Optional[int] = None  # For temporal merges (deprecated)
    require_same_actor: bool = False  # For actor-specific merges
    temporal_window_min: int = 30  # Min/default temporal window in minutes
    temporal_window_max: int = 60  # Max temporal gap in minutes
    
    def get_merge_key(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Generate a merge key for initial grouping (refined by semantic similarity)"""
        if self.merge_type == MergeType.ACTOR:
            # Group by normalized actor category
            who = event_data.get('who', '').lower().strip()
            if not who:
                return None
            # Normalize to two main categories
            if who in ['user', 'human', 'person', 'customer', 'client']:
                return 'user_actor'
            elif who in ['assistant', 'ai', 'agent', 'bot', 'system']:
                return 'assistant_actor'
            # For unknown actors, use the actual name
            return who
            
        elif self.merge_type == MergeType.TEMPORAL:
            # Use dynamic grouping key - actual grouping happens in merger logic
            # This allows for gap-based dynamic windows rather than fixed buckets
            return 'temporal_dynamic'  # Single key forces similarity-based grouping
            
        elif self.merge_type == MergeType.CONCEPTUAL:
            # For conceptual grouping, return a generic key to allow
            # semantic similarity to be the primary grouping factor
            # The actual grouping will be based on embedding distance
            return 'concept_semantic'
            
        elif self.merge_type == MergeType.SPATIAL:
            # Group by location without normalization (let semantic similarity handle it)
            # DO NOT include actor information - locations should span actors
            where = event_data.get('where', '').lower().strip()
            if not where:
                return 'unspecified_location'
            # Just use the cleaned location as the key
            return f"location_{where[:30]}"
            
        else:  # HYBRID
            # Combine multiple fields for complex grouping
            parts = []
            for field in [self.primary_field] + self.secondary_fields:
                value = event_data.get(field, '')
                if value:
                    parts.append(value.lower().strip()[:20])
            return "-".join(parts) if parts else None
    
    def should_merge_with_group(self, event_data: Dict, group_key: str, 
                                group_events: List[Dict], distance: float) -> bool:
        """
        Determine if an event should merge with an existing group.
        
        Args:
            event_data: New event's data
            group_key: Key of the existing group
            group_events: Events already in the group
            distance: Embedding distance to closest event in group
            
        Returns:
            True if should merge with this group
        """
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"should_merge_with_group for {self.merge_type.value}: distance={distance:.3f}, threshold={self.distance_threshold}")
        
        # Check embedding distance first
        if distance > self.distance_threshold:
            logger.debug(f"Rejecting {self.merge_type.value} merge: distance {distance:.3f} > threshold {self.distance_threshold}")
            return False
            
        # Type-specific checks
        if self.merge_type == MergeType.ACTOR:
            # Must be same actor (only for ACTOR type)
            event_key = self.get_merge_key(event_data)
            if event_key != group_key:
                return False
        
        # For non-ACTOR types, explicitly allow cross-actor merging
        if self.require_same_actor and self.merge_type != MergeType.ACTOR:
            # This should never be true based on our strategies, but make it explicit
            pass  # Do not enforce actor restriction for non-ACTOR merge types
                
        elif self.merge_type == MergeType.TEMPORAL:
            # Use dynamic time windows based on activity patterns WITH A CAP
            if group_events:
                # Get event timestamp from 'when' field (was using 'timestamp')
                event_when = event_data.get('when')
                if not event_when:
                    # Try 'timestamp' as fallback for compatibility
                    event_time = event_data.get('timestamp')
                    if not event_time:
                        return False  # Don't merge if no timestamp
                else:
                    try:
                        # Parse the 'when' field
                        from dateutil import parser as date_parser
                        event_time = date_parser.parse(event_when)
                    except:
                        return False  # Don't merge if can't parse timestamp
                
                # Ensure timezone awareness
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                
                # Sort group events by time, handling both 'when' and 'timestamp'
                sorted_events = []
                for e in group_events:
                    event_ts = None
                    if e.get('when'):
                        try:
                            event_ts = date_parser.parse(e.get('when'))
                        except:
                            pass
                    if not event_ts and e.get('timestamp'):
                        event_ts = e.get('timestamp')
                    if event_ts:
                        if event_ts.tzinfo is None:
                            event_ts = event_ts.replace(tzinfo=timezone.utc)
                        sorted_events.append((e, event_ts))
                
                sorted_events.sort(key=lambda x: x[1])
                
                if not sorted_events:
                    return False  # No timestamps to compare
                
                # Check gap from most recent event
                last_event_time = sorted_events[-1][1]
                gap_minutes = abs((event_time - last_event_time).total_seconds() / 60)
                
                # Debug logging for temporal gap checking
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Temporal merge check: event_time={event_time}, last_event_time={last_event_time}, gap_minutes={gap_minutes:.2f}")
                
                # Calculate dynamic window based on group pattern
                gaps = []
                for i in range(1, len(sorted_events)):
                    prev_time = sorted_events[i-1][1]
                    curr_time = sorted_events[i][1]
                    gap = abs((curr_time - prev_time).total_seconds() / 60)
                    gaps.append(gap)
                
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    # Dynamic window is 2.5x average gap, but CAPPED
                    # Use strategy parameters (which should come from config)
                    MIN_WINDOW = 10  # Minimum window in minutes
                    MAX_TEMPORAL_GAP = self.temporal_window_max  # Use strategy max (default 60)
                    dynamic_window = min(max(avg_gap * 2.5, MIN_WINDOW), MAX_TEMPORAL_GAP)
                else:
                    # Default for first gap - use strategy default
                    dynamic_window = self.temporal_window_min  # Use strategy default (30)
                
                # Reject if gap exceeds dynamic window
                logger.debug(f"Temporal window check: gap={gap_minutes:.2f} min, dynamic_window={dynamic_window:.2f} min, MAX={MAX_TEMPORAL_GAP} min")
                if gap_minutes > dynamic_window:
                    logger.info(f"Rejecting temporal merge: gap ({gap_minutes:.2f} min) exceeds window ({dynamic_window:.2f} min)")
                    return False
                else:
                    logger.debug(f"Accepting temporal merge: gap ({gap_minutes:.2f} min) within window ({dynamic_window:.2f} min)")
                    
        elif self.merge_type == MergeType.CONCEPTUAL:
            # For conceptual merges, rely mainly on embedding distance
            # which captures semantic similarity
            pass
        
        logger.debug(f"Accepting {self.merge_type.value} merge: passed all checks")
        return True


# Pre-defined merge strategies  
# Note: Higher thresholds = more lenient (accepts larger distances)
MERGE_STRATEGIES = {
    MergeType.ACTOR: MergeStrategy(
        merge_type=MergeType.ACTOR,
        primary_field='who',
        secondary_fields=['what'],
        distance_threshold=2.0,  # Keep lenient for actors (working well at 2 groups)
        require_same_actor=True,  # ONLY Actor type requires same actor
        temporal_window_min=30,  # Not used for actor merges
        temporal_window_max=60  # Not used for actor merges
    ),
    
    MergeType.TEMPORAL: MergeStrategy(
        merge_type=MergeType.TEMPORAL,
        primary_field='when',  # Time-based grouping only
        secondary_fields=[],  # No secondary fields for pure temporal grouping
        distance_threshold=0.5,  # STRICT - time gaps are the primary factor
        time_window_hours=None,  # Deprecated - using minute-based windows
        require_same_actor=False,  # EXPLICITLY allow cross-actor for conversations
        temporal_window_min=30,  # Default/minimum temporal window (30 minutes)
        temporal_window_max=60  # Maximum temporal gap (60 minutes, not 240!)
    ),
    
    MergeType.CONCEPTUAL: MergeStrategy(
        merge_type=MergeType.CONCEPTUAL,
        primary_field='what',  # The topic/action is primary for conceptual grouping
        secondary_fields=['why'],  # The purpose/motivation provides additional context
        distance_threshold=0.3,  # Cosine distance threshold (0=identical, 1=orthogonal)
        require_same_actor=False,  # EXPLICITLY allow cross-actor for concepts
        temporal_window_min=30,  # Not used for conceptual merges
        temporal_window_max=60  # Not used for conceptual merges
    ),
    
    MergeType.SPATIAL: MergeStrategy(
        merge_type=MergeType.SPATIAL,
        primary_field='where',
        secondary_fields=['what'],
        distance_threshold=0.3,  # Tighter threshold to create distinct location groups
        require_same_actor=False,  # EXPLICITLY allow cross-actor for locations
        temporal_window_min=30,  # Not used for spatial merges
        temporal_window_max=60  # Not used for spatial merges
    )
}


@dataclass
class MultiMergeTracker:
    """Tracks which raw events belong to which merge groups"""
    
    # Map from raw event ID to set of (merge_type, merge_id) tuples
    raw_to_merges: Dict[str, Set[tuple]] = field(default_factory=dict)
    
    # Map from (merge_type, merge_id) to set of raw event IDs
    merge_to_raws: Dict[tuple, Set[str]] = field(default_factory=dict)
    
    def add_merge_membership(self, raw_event_id: str, merge_type: MergeType, merge_id: str):
        """Record that a raw event belongs to a merge group"""
        merge_key = (merge_type, merge_id)
        
        # Update raw to merges mapping
        if raw_event_id not in self.raw_to_merges:
            self.raw_to_merges[raw_event_id] = set()
        self.raw_to_merges[raw_event_id].add(merge_key)
        
        # Update merge to raws mapping
        if merge_key not in self.merge_to_raws:
            self.merge_to_raws[merge_key] = set()
        self.merge_to_raws[merge_key].add(raw_event_id)
    
    def get_merge_groups_for_raw(self, raw_event_id: str) -> List[tuple]:
        """Get all merge groups this raw event belongs to"""
        return list(self.raw_to_merges.get(raw_event_id, set()))
    
    def get_raw_events_in_merge(self, merge_type: MergeType, merge_id: str) -> List[str]:
        """Get all raw events in a specific merge group"""
        merge_key = (merge_type, merge_id)
        return list(self.merge_to_raws.get(merge_key, set()))
    
    def remove_raw_from_merge(self, raw_event_id: str, merge_type: MergeType, merge_id: str):
        """Remove a raw event from a merge group"""
        merge_key = (merge_type, merge_id)
        
        # Remove from raw to merges
        if raw_event_id in self.raw_to_merges:
            self.raw_to_merges[raw_event_id].discard(merge_key)
            if not self.raw_to_merges[raw_event_id]:
                del self.raw_to_merges[raw_event_id]
        
        # Remove from merge to raws
        if merge_key in self.merge_to_raws:
            self.merge_to_raws[merge_key].discard(raw_event_id)
            if not self.merge_to_raws[merge_key]:
                del self.merge_to_raws[merge_key]
