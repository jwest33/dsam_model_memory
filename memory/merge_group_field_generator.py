"""
Merge group level field generator for intelligent why/how characterization.
Analyzes entire merge groups to generate meaningful group-level fields.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from models.merged_event import MergedEvent
from models.merge_types import MergeType

logger = logging.getLogger(__name__)


class MergeGroupFieldGenerator:
    """Generates group-level why/how fields for merge groups"""
    
    def __init__(self, llm_client=None):
        """
        Initialize the merge group field generator.
        
        Args:
            llm_client: LLM client for generating fields
        """
        self.llm_client = llm_client
    
    def generate_group_fields(self, 
                            merged_event: MergedEvent,
                            merge_type: MergeType,
                            llm_context: str) -> Dict[str, str]:
        """
        Generate group-level why and how fields based on the entire merge group.
        
        Args:
            merged_event: The merged event containing all group members
            merge_type: Type of merge (ACTOR, TEMPORAL, CONCEPTUAL, SPATIAL)
            llm_context: The formatted LLM context for this group
            
        Returns:
            Dictionary with 'group_why' and 'group_how' fields
        """
        if not self.llm_client:
            # Fallback to heuristic generation
            return self._generate_heuristic_fields(merged_event, merge_type)
        
        try:
            # Generate WHY field for the group
            group_why = self._generate_group_why(merged_event, merge_type, llm_context)
            
            # Generate HOW field for the group
            group_how = self._generate_group_how(merged_event, merge_type, llm_context)
            
            return {
                'group_why': group_why,
                'group_how': group_how,
                'merge_type': merge_type.value,
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"LLM generation failed, using heuristics: {e}")
            return self._generate_heuristic_fields(merged_event, merge_type)
    
    def _generate_group_why(self, 
                           merged_event: MergedEvent,
                           merge_type: MergeType,
                           llm_context: str) -> str:
        """
        Generate a group-level 'why' field that characterizes the purpose of the entire group.
        """
        # Extract key information from the merged event
        latest_state = merged_event.get_latest_state()
        event_count = merged_event.merge_count
        
        # Build prompt based on merge type
        if merge_type == MergeType.CONCEPTUAL:
            prompt = f"""Analyze this conceptual memory group and generate a SHORT (3-7 words) 'why' field that captures the core purpose/concept of the entire group.

Memory Group Context:
{llm_context[:1500]}

Number of events: {event_count}
Primary topic: {latest_state.get('what', 'unknown')}

Generate a concise 'why' field that describes the overarching purpose or concept that unites all these memories.
Examples:
- "implementing authentication system"
- "troubleshooting database errors"
- "learning machine learning concepts"
- "planning product features"

Return ONLY the why field text:"""

        elif merge_type == MergeType.TEMPORAL:
            prompt = f"""Analyze this temporal memory group (conversation/sequence) and generate a SHORT (3-7 words) 'why' field that captures the purpose of this interaction sequence.

Memory Group Context:
{llm_context[:1500]}

Number of events: {event_count}
Time span: {self._calculate_time_span(merged_event)}

Generate a concise 'why' field that describes what this conversation/sequence was trying to achieve.
Examples:
- "resolving technical issue"
- "discussing project requirements"
- "exploring design options"
- "debugging application error"

Return ONLY the why field text:"""

        elif merge_type == MergeType.ACTOR:
            prompt = f"""Analyze this actor-based memory group and generate a SHORT (3-7 words) 'why' field that captures this actor's primary intent or role.

Memory Group Context:
{llm_context[:1500]}

Actor: {latest_state.get('who', 'unknown')}
Number of events: {event_count}

Generate a concise 'why' field that describes this actor's overall purpose or goal.
Examples:
- "seeking technical assistance"
- "providing solutions"
- "monitoring system health"
- "facilitating discussions"

Return ONLY the why field text:"""

        else:  # SPATIAL
            prompt = f"""Analyze this location-based memory group and generate a SHORT (3-7 words) 'why' field that captures the purpose of activities at this location.

Memory Group Context:
{llm_context[:1500]}

Location: {latest_state.get('where', 'unknown')}
Number of events: {event_count}

Generate a concise 'why' field that describes the purpose of activities at this location.
Examples:
- "web interface interactions"
- "api endpoint operations"
- "database transactions"
- "file system operations"

Return ONLY the why field text:"""
        
        # Call LLM
        response = self.llm_client.generate(
            prompt,
            max_tokens=20,
            temperature=0.3,
            stop=["\n", "."]
        )
        
        # Clean and validate
        why_field = response.strip().strip('"').strip("'").strip()
        if not why_field:
            raise ValueError("Empty response from LLM")
            
        return why_field
    
    def _generate_group_how(self,
                           merged_event: MergedEvent,
                           merge_type: MergeType,
                           llm_context: str) -> str:
        """
        Generate a group-level 'how' field that characterizes the mechanisms used by the group.
        """
        # Analyze the how methods used in the group
        how_methods = merged_event.how_methods if hasattr(merged_event, 'how_methods') else {}
        
        prompt = f"""Analyze this memory group and identify the PRIMARY mechanism or method used across all events.

Memory Group Context:
{llm_context[:1500]}

Merge Type: {merge_type.value}
Number of events: {merged_event.merge_count}

Common mechanisms might include:
- "conversation dialogue" (for back-and-forth discussions)
- "iterative development" (for incremental progress)
- "problem solving sequence" (for troubleshooting)
- "information gathering" (for research/exploration)
- "step-by-step guidance" (for tutorials)
- "collaborative interaction" (for joint work)
- "system monitoring" (for observations)
- "api interactions" (for service calls)

Generate a SHORT (2-4 words) mechanism description that best characterizes HOW this group of activities was conducted.
Return ONLY the mechanism text:"""
        
        # Call LLM
        response = self.llm_client.generate(
            prompt,
            max_tokens=15,
            temperature=0.3,
            stop=["\n", "."]
        )
        
        # Clean and validate
        how_field = response.strip().strip('"').strip("'").strip()
        if not how_field:
            # Fallback to most common method
            if how_methods:
                how_field = max(how_methods, key=lambda k: len(how_methods[k]))
            else:
                how_field = "mixed methods"
                
        return how_field
    
    def _generate_heuristic_fields(self,
                                  merged_event: MergedEvent,
                                  merge_type: MergeType) -> Dict[str, str]:
        """
        Generate fields using heuristics when LLM is unavailable.
        """
        latest_state = merged_event.get_latest_state()
        
        # Generate based on merge type
        if merge_type == MergeType.CONCEPTUAL:
            # Analyze the what/why fields
            what = latest_state.get('what', '')
            why = latest_state.get('why', '')
            
            if 'error' in what.lower() or 'error' in why.lower():
                group_why = "troubleshooting errors"
            elif 'implement' in what.lower() or 'create' in what.lower():
                group_why = "building features"
            elif 'learn' in what.lower() or 'understand' in what.lower():
                group_why = "learning concepts"
            else:
                group_why = "exploring topics"
                
            group_how = "conceptual discussion"
            
        elif merge_type == MergeType.TEMPORAL:
            # Analyze the conversation flow
            event_count = merged_event.merge_count
            if event_count > 10:
                group_why = "extended conversation"
                group_how = "multi-turn dialogue"
            elif event_count > 5:
                group_why = "discussing topic"
                group_how = "conversation sequence"
            else:
                group_why = "quick exchange"
                group_how = "brief interaction"
                
        elif merge_type == MergeType.ACTOR:
            who = latest_state.get('who', '').lower()
            if 'user' in who:
                group_why = "user interactions"
                group_how = "user activities"
            elif 'assistant' in who or 'ai' in who:
                group_why = "providing assistance"
                group_how = "ai responses"
            else:
                group_why = f"{who} activities"
                group_how = f"{who} actions"
                
        else:  # SPATIAL
            where = latest_state.get('where', '')
            if 'web' in where.lower():
                group_why = "web interactions"
                group_how = "web interface"
            elif 'api' in where.lower():
                group_why = "api operations"
                group_how = "api calls"
            else:
                group_why = f"{where} activities"
                group_how = f"at {where}"
        
        return {
            'group_why': group_why,
            'group_how': group_how,
            'merge_type': merge_type.value,
            'generated_at': datetime.utcnow().isoformat(),
            'method': 'heuristic'
        }
    
    def _calculate_time_span(self, merged_event: MergedEvent) -> str:
        """Calculate the time span of events in the group"""
        if not hasattr(merged_event, 'when_timeline') or not merged_event.when_timeline:
            return "unknown duration"
        
        timeline = sorted(merged_event.when_timeline, key=lambda x: x.timestamp)
        if len(timeline) < 2:
            return "single event"
            
        first = timeline[0].timestamp
        last = timeline[-1].timestamp
        duration = last - first
        
        if duration.days > 0:
            return f"{duration.days} days"
        elif duration.seconds > 3600:
            return f"{duration.seconds // 3600} hours"
        elif duration.seconds > 60:
            return f"{duration.seconds // 60} minutes"
        else:
            return f"{duration.seconds} seconds"
    
    def update_merge_group_fields(self,
                                 merge_groups: Dict[MergeType, Dict],
                                 llm_client=None) -> Dict[MergeType, int]:
        """
        Batch update all merge groups with group-level fields.
        
        Args:
            merge_groups: Dictionary of merge groups by type
            llm_client: Optional LLM client for generation
            
        Returns:
            Count of updated groups by type
        """
        if llm_client:
            self.llm_client = llm_client
            
        update_counts = {}
        
        for merge_type, groups in merge_groups.items():
            updated = 0
            for group_id, group_data in groups.items():
                try:
                    merged_event = group_data.get('merged_event')
                    if not merged_event:
                        continue
                    
                    # Generate LLM context for this group
                    # This would use the existing context generation logic
                    llm_context = self._generate_simple_context(merged_event)
                    
                    # Generate group fields
                    fields = self.generate_group_fields(
                        merged_event, merge_type, llm_context
                    )
                    
                    # Store the fields in the group data
                    group_data['group_why'] = fields['group_why']
                    group_data['group_how'] = fields['group_how']
                    group_data['fields_generated_at'] = fields['generated_at']
                    
                    updated += 1
                    
                except Exception as e:
                    logger.error(f"Failed to update group {group_id}: {e}")
            
            update_counts[merge_type] = updated
            logger.info(f"Updated {updated} {merge_type.value} groups with group-level fields")
        
        return update_counts
    
    def _generate_simple_context(self, merged_event: MergedEvent) -> str:
        """Generate a simple text context from the merged event"""
        lines = []
        latest = merged_event.get_latest_state()
        
        lines.append(f"Group with {merged_event.merge_count} events")
        lines.append(f"Latest state:")
        for key, value in latest.items():
            if value:
                lines.append(f"  {key}: {value}")
        
        # Add timeline if available
        if hasattr(merged_event, 'when_timeline') and merged_event.when_timeline:
            lines.append("\nTimeline:")
            for tp in merged_event.when_timeline[:5]:  # First 5 events
                lines.append(f"  - {tp.description or 'Event'} at {tp.timestamp}")
        
        return "\n".join(lines)