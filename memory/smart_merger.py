"""
Smart Merger for Intelligent Event Merging

This module provides intelligent merging strategies based on component analysis,
temporal relationships, and semantic similarity.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from difflib import SequenceMatcher

from models.event import Event, FiveW1H
from models.merged_event import MergedEvent, EventRelationship, ComponentVariant

logger = logging.getLogger(__name__)


class SmartMerger:
    """
    Intelligently merges events based on component analysis and temporal relationships.
    
    This class analyzes each component of the 5W1H framework to determine the best
    merging strategy and maintains relationships between events.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the SmartMerger.
        
        Args:
            similarity_threshold: Threshold for considering text as similar
        """
        self.similarity_threshold = similarity_threshold
        
        # Temporal thresholds for relationship detection
        self.update_window_hours = 24  # Events within 24 hours might be updates
        self.correction_keywords = ['fix', 'correct', 'revise', 'amend', 'repair']
        self.continuation_keywords = ['continue', 'resume', 'proceed', 'further']
        
    def merge_events(self, existing: MergedEvent, new_event: Event, 
                    embeddings: Optional[Dict] = None) -> MergedEvent:
        """
        Merge a new event into an existing merged event composite.
        
        Args:
            existing: The existing merged event
            new_event: The new event to merge
            embeddings: Optional embeddings for the new event
            
        Returns:
            Updated merged event with the new event incorporated
        """
        # Determine the relationship of the new event to existing ones
        relationship = self._determine_relationship(existing, new_event)
        
        # Create event data dictionary
        event_data = {
            'who': new_event.five_w1h.who,
            'what': new_event.five_w1h.what,
            'when': new_event.five_w1h.when,
            'where': new_event.five_w1h.where,
            'why': new_event.five_w1h.why,
            'how': new_event.five_w1h.how,
            'timestamp': new_event.created_at
        }
        
        # Add the raw event with its relationship
        existing.add_raw_event(new_event.id, event_data, relationship)
        
        # Update temporal relationships
        self._update_temporal_relationships(existing, new_event, relationship)
        
        # Detect and update dependencies
        self._detect_dependencies(existing, new_event)
        
        # Store embedding history if provided
        if embeddings:
            existing.embedding_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'event_id': new_event.id,
                'euclidean_weight': embeddings.get('euclidean_weight', 0.5),
                'hyperbolic_weight': embeddings.get('hyperbolic_weight', 0.5)
            })
        
        logger.info(f"Merged event {new_event.id} into {existing.id} with relationship: {relationship.value}")
        
        return existing
    
    def _determine_relationship(self, existing: MergedEvent, new_event: Event) -> EventRelationship:
        """
        Determine how the new event relates to existing ones.
        
        Analyzes temporal proximity, content similarity, and keywords to determine
        if this is an update, correction, continuation, or variation.
        """
        if not existing.temporal_chain:
            return EventRelationship.INITIAL
        
        # Get the most recent event data
        latest_state = existing.get_latest_state()
        
        # Check temporal proximity to last event
        if existing.when_timeline:
            last_timestamp = existing.when_timeline[-1].timestamp
            time_diff = new_event.created_at - last_timestamp
            
            # Within update window?
            if time_diff < timedelta(hours=self.update_window_hours):
                # Check for correction keywords
                if self._contains_correction_keywords(new_event.five_w1h.what):
                    return EventRelationship.CORRECTION
                
                # Check for continuation keywords
                if self._contains_continuation_keywords(new_event.five_w1h.what):
                    return EventRelationship.CONTINUATION
                
                # Check if it's an update (similar content with modifications)
                if self._is_update(latest_state, new_event):
                    return EventRelationship.UPDATE
        
        # Check if it supersedes (completely different approach to same goal)
        if self._is_superseding(existing, new_event):
            return EventRelationship.SUPERSEDES
        
        # Default to variation (similar but independent)
        return EventRelationship.VARIATION
    
    def _contains_correction_keywords(self, text: Optional[str]) -> bool:
        """Check if text contains correction-related keywords"""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.correction_keywords)
    
    def _contains_continuation_keywords(self, text: Optional[str]) -> bool:
        """Check if text contains continuation-related keywords"""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.continuation_keywords)
    
    def _is_update(self, latest_state: Dict, new_event: Event) -> bool:
        """
        Determine if the new event is an update to the latest state.
        
        An update maintains the same core action but with modifications.
        """
        if not latest_state.get('what') or not new_event.five_w1h.what:
            return False
        
        # Calculate similarity between actions
        similarity = self._calculate_text_similarity(
            latest_state['what'], 
            new_event.five_w1h.what
        )
        
        # High similarity but not identical = likely an update
        return 0.5 < similarity < 0.95
    
    def _is_superseding(self, existing: MergedEvent, new_event: Event) -> bool:
        """
        Determine if the new event supersedes existing ones.
        
        A superseding event achieves the same goal but with a completely different approach.
        """
        if not new_event.five_w1h.why:
            return False
        
        # Check if the WHY (goal) is similar but the HOW (method) is different
        for why, variants in existing.why_variants.items():
            why_similarity = self._calculate_text_similarity(why, new_event.five_w1h.why)
            if why_similarity > self.similarity_threshold:
                # Similar goal, check if method is different
                if new_event.five_w1h.how and new_event.five_w1h.how not in existing.how_methods:
                    return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _update_temporal_relationships(self, existing: MergedEvent, new_event: Event, 
                                      relationship: EventRelationship):
        """Update temporal relationships based on the determined relationship type"""
        if relationship == EventRelationship.SUPERSEDES:
            # This event supersedes the entire chain
            existing.superseded_by = new_event.id
            
        elif relationship == EventRelationship.CORRECTION:
            # Mark the last event as corrected
            if existing.temporal_chain:
                last_event_id = existing.temporal_chain[-1]
                if last_event_id not in existing.causal_links:
                    existing.causal_links[last_event_id] = {'causes': [], 'effects': []}
                existing.causal_links[last_event_id]['effects'].append(f"corrected_by:{new_event.id}")
    
    def _detect_dependencies(self, existing: MergedEvent, new_event: Event):
        """
        Detect causal dependencies between events.
        
        This analyzes the WHY and HOW components to identify cause-effect relationships.
        """
        # Look for explicit dependencies in the WHY field
        if new_event.five_w1h.why:
            why_lower = new_event.five_w1h.why.lower()
            
            # Check for references to previous events
            for prev_event_id in existing.temporal_chain[-5:]:  # Check last 5 events
                # Simple heuristic: check if previous event's what is mentioned in why
                for what_variants in existing.what_variants.values():
                    for variant in what_variants:
                        if variant.event_id == prev_event_id:
                            if self._text_references(why_lower, variant.value):
                                existing.add_dependency(
                                    new_event.id, 
                                    depends_on=[prev_event_id]
                                )
                                break
    
    def _text_references(self, text: str, reference: str) -> bool:
        """Check if text references another piece of text"""
        if not text or not reference:
            return False
        
        # Simple check - can be enhanced with NLP
        reference_words = reference.lower().split()[:3]  # First 3 words
        return all(word in text for word in reference_words if len(word) > 3)
    
    def analyze_merge_quality(self, merged_event: MergedEvent) -> Dict[str, float]:
        """
        Analyze the quality and coherence of a merged event.
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Temporal coherence - how well events follow in time
        if len(merged_event.when_timeline) > 1:
            time_gaps = []
            for i in range(1, len(merged_event.when_timeline)):
                gap = (merged_event.when_timeline[i].timestamp - 
                      merged_event.when_timeline[i-1].timestamp).total_seconds()
                time_gaps.append(gap)
            
            # Calculate coefficient of variation for time gaps
            if time_gaps:
                mean_gap = np.mean(time_gaps)
                std_gap = np.std(time_gaps)
                metrics['temporal_coherence'] = 1.0 - min(std_gap / (mean_gap + 1), 1.0)
            else:
                metrics['temporal_coherence'] = 1.0
        else:
            metrics['temporal_coherence'] = 1.0
        
        # Actor consistency - how consistent are the actors
        if merged_event.who_variants:
            total_actors = sum(len(variants) for variants in merged_event.who_variants.values())
            unique_actors = len(merged_event.who_variants)
            metrics['actor_consistency'] = 1.0 / (1 + np.log1p(unique_actors))
        else:
            metrics['actor_consistency'] = 1.0
        
        # Action coherence - how similar are the actions
        if merged_event.what_variants:
            action_similarities = []
            what_list = list(merged_event.what_variants.keys())
            for i in range(len(what_list)):
                for j in range(i + 1, len(what_list)):
                    sim = self._calculate_text_similarity(what_list[i], what_list[j])
                    action_similarities.append(sim)
            
            metrics['action_coherence'] = np.mean(action_similarities) if action_similarities else 1.0
        else:
            metrics['action_coherence'] = 1.0
        
        # Location consistency
        if merged_event.where_locations:
            total_occurrences = sum(merged_event.where_locations.values())
            most_common_count = max(merged_event.where_locations.values())
            metrics['location_consistency'] = most_common_count / total_occurrences
        else:
            metrics['location_consistency'] = 1.0
        
        # Overall coherence score
        metrics['overall_coherence'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def should_merge(self, event1_embedding: np.ndarray, event2_embedding: np.ndarray,
                    event1_data: Dict, event2_data: Dict, distance: float) -> bool:
        """
        Determine if two events should be merged based on embeddings and content.
        
        Args:
            event1_embedding: Embedding of first event
            event2_embedding: Embedding of second event
            event1_data: 5W1H data of first event
            event2_data: 5W1H data of second event
            distance: Pre-computed distance between embeddings
            
        Returns:
            True if events should be merged
        """
        # Check if this is a conversation first
        who1 = event1_data.get('who', '').lower()
        who2 = event2_data.get('who', '').lower()
        conversation_actors = {'user', 'assistant', 'ai', 'bot', 'system', 'human'}
        is_conversation = who1 in conversation_actors or who2 in conversation_actors
        
        # Primary check: embedding distance (more lenient for conversations)
        distance_threshold = 0.25 if is_conversation else 0.15
        if distance > distance_threshold:  # Too far apart
            return False
        
        # Secondary checks on components
        
        # If actors are different and NOT a conversation, require higher similarity
        if who1 and who2 and who1 != who2:
            if not is_conversation:
                # Different actors, not a conversation - check if actions are very similar
                if event1_data.get('what') and event2_data.get('what'):
                    action_sim = self._calculate_text_similarity(
                        event1_data['what'], 
                        event2_data['what']
                    )
                    if action_sim < 0.8:  # Not similar enough to merge different actors
                        return False
            # For conversations, we're more lenient - the embedding distance check is enough
        
        # Check temporal distance
        if 'timestamp' in event1_data and 'timestamp' in event2_data:
            time_diff = abs((event1_data['timestamp'] - event2_data['timestamp']).total_seconds())
            # If events are far apart in time and not very similar, don't merge
            if time_diff > 7 * 24 * 3600:  # More than a week apart
                if distance > 0.1:  # And not extremely similar
                    return False
        
        return True
    
    def split_merged_event(self, merged_event: MergedEvent, 
                          split_criteria: Dict[str, Any]) -> List[MergedEvent]:
        """
        Split a merged event based on specified criteria.
        
        This can be used to undo overly aggressive merging.
        
        Args:
            merged_event: The merged event to split
            split_criteria: Criteria for splitting (e.g., by actor, time range, etc.)
            
        Returns:
            List of new merged events after splitting
        """
        split_events = []
        
        # Example: Split by actor
        if split_criteria.get('by_actor'):
            for who, variants in merged_event.who_variants.items():
                new_merged = MergedEvent(
                    id=f"{merged_event.id}_split_{who}",
                    base_event_id=variants[0].event_id if variants else merged_event.base_event_id
                )
                
                # Copy relevant events for this actor
                for variant in variants:
                    # Find corresponding components
                    event_data = self._reconstruct_event_data(merged_event, variant.event_id)
                    new_merged.add_raw_event(variant.event_id, event_data)
                
                split_events.append(new_merged)
        
        # Example: Split by time period
        elif split_criteria.get('by_time_period'):
            period_hours = split_criteria['by_time_period']
            current_group = None
            
            for point in merged_event.when_timeline:
                if current_group is None or \
                   (point.timestamp - current_group.when_timeline[-1].timestamp).total_seconds() > period_hours * 3600:
                    # Start new group
                    current_group = MergedEvent(
                        id=f"{merged_event.id}_split_{len(split_events)}",
                        base_event_id=point.event_id
                    )
                    split_events.append(current_group)
                
                # Add event to current group
                event_data = self._reconstruct_event_data(merged_event, point.event_id)
                current_group.add_raw_event(point.event_id, event_data)
        
        return split_events
    
    def _reconstruct_event_data(self, merged_event: MergedEvent, event_id: str) -> Dict:
        """Reconstruct event data from a merged event for a specific event ID"""
        data = {'timestamp': datetime.utcnow()}
        
        # Find components for this event ID
        for who, variants in merged_event.who_variants.items():
            for v in variants:
                if v.event_id == event_id:
                    data['who'] = v.value
                    data['timestamp'] = v.timestamp
                    break
        
        for what, variants in merged_event.what_variants.items():
            for v in variants:
                if v.event_id == event_id:
                    data['what'] = v.value
                    break
        
        for point in merged_event.when_timeline:
            if point.event_id == event_id:
                data['when'] = point.semantic_time
                break
        
        # WHERE and HOW don't track event IDs, so use dominant
        data['where'] = merged_event._get_dominant_where()
        data['how'] = merged_event._get_dominant_how()
        
        for why, variants in merged_event.why_variants.items():
            for v in variants:
                if v.event_id == event_id:
                    data['why'] = v.value
                    break
        
        return data