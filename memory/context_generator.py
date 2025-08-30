"""
Merged Event Context Generator for LLM Consumption

This module generates rich, contextual representations of merged events
for LLM processing, preserving all information while presenting it in
an easily digestible format.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from models.merged_event import MergedEvent, EventRelationship, ComponentVariant
from memory.temporal_chain import TemporalChain

logger = logging.getLogger(__name__)


class MergedEventContextGenerator:
    """
    Generates LLM-friendly context from merged events, presenting all component
    variations, temporal progressions, and relationships in a structured format.
    """
    
    def __init__(self, temporal_chain: Optional[TemporalChain] = None):
        """
        Initialize the context generator.
        
        Args:
            temporal_chain: Optional temporal chain manager for additional context
        """
        self.temporal_chain = temporal_chain
        
        # Relevance thresholds for determining what to include
        self.relevance_thresholds = {
            'who': 0.3,
            'what': 0.5,
            'when': 0.4,
            'where': 0.3,
            'why': 0.6,
            'how': 0.4,
            'temporal': 0.7,
            'causal': 0.5
        }
    
    def generate_context(self, merged_event: MergedEvent, 
                        query_context: Optional[Dict] = None,
                        format_type: str = 'detailed') -> str:
        """
        Generate comprehensive context for LLM from a merged event.
        
        Args:
            merged_event: The merged event to generate context for
            query_context: Optional query context for relevance scoring
            format_type: 'summary', 'detailed', or 'structured'
            
        Returns:
            Formatted context string for LLM consumption
        """
        if format_type == 'summary':
            return self._generate_summary_context(merged_event)
        elif format_type == 'structured':
            return self._generate_structured_context(merged_event, query_context)
        else:
            return self._generate_detailed_context(merged_event, query_context)
    
    def _generate_summary_context(self, merged_event: MergedEvent) -> str:
        """Generate a brief summary of the merged event"""
        parts = []
        
        # Get dominant pattern
        if merged_event.dominant_pattern:
            who = merged_event.dominant_pattern.get('who', 'Unknown')
            what = merged_event.dominant_pattern.get('what', 'performed action')
            when = merged_event.dominant_pattern.get('when', '')
            where = merged_event.dominant_pattern.get('where', '')
            
            summary = f"{who}: {what}"
            if when:
                summary += f" ({when})"
            if where:
                summary += f" at {where}"
            
            parts.append(summary)
        
        # Add merge count if multiple events
        if merged_event.merge_count > 1:
            parts.append(f"[{merged_event.merge_count} merged events]")
        
        return " ".join(parts)
    
    def _generate_detailed_context(self, merged_event: MergedEvent, 
                                  query_context: Optional[Dict]) -> str:
        """Generate detailed context grouped by actor"""
        context_parts = []
        
        # Compute relevance scores if query provided
        relevance_scores = self._compute_component_relevance(merged_event, query_context) \
                          if query_context else self._default_relevance_scores()
        
        # Header with merge information
        if merged_event.merge_count > 1:
            context_parts.append(f"[Conversation with {merged_event.merge_count} exchanges]")
            context_parts.append("")
        
        # WHO - Actors involved
        who_context = self._format_who_component(merged_event, relevance_scores['who'])
        if who_context:
            context_parts.append(f"Participants: {who_context}")
        
        # WHAT - Actions grouped by actor
        what_context = self._format_what_component(merged_event, relevance_scores['what'])
        if what_context:
            context_parts.append("Dialogue:")
            context_parts.append(what_context)
        
        # WHEN - Temporal context
        when_context = self._format_when_component(merged_event, relevance_scores['temporal'])
        if when_context:
            context_parts.append(f"Time: {when_context}")
        
        # WHERE - Locations
        where_context = self._format_where_component(merged_event)
        if where_context:
            context_parts.append(f"Locations: {where_context}")
        
        # WHY - Reasons and their evolution
        why_context = self._format_why_component(merged_event, relevance_scores['why'])
        if why_context:
            context_parts.append("Reasons:")
            context_parts.append(why_context)
        
        # HOW - Methods used
        how_context = self._format_how_component(merged_event)
        if how_context:
            context_parts.append(f"Methods: {how_context}")
        
        # Dependencies and relationships
        if merged_event.causal_links and relevance_scores['causal'] > self.relevance_thresholds['causal']:
            dependencies = self._format_dependencies(merged_event)
            if dependencies:
                context_parts.append("")
                context_parts.append("Dependencies:")
                context_parts.append(dependencies)
        
        # Latest state if there are supersessions
        if merged_event.superseded_by:
            context_parts.append("")
            context_parts.append(f"Note: This has been superseded by event {merged_event.superseded_by}")
        elif merged_event.supersedes:
            context_parts.append("")
            context_parts.append(f"Note: This supersedes event {merged_event.supersedes}")
        
        # Add temporal chain context if available
        if self.temporal_chain and merged_event.temporal_chain:
            chain_context = self._format_temporal_chain_context(merged_event)
            if chain_context:
                context_parts.append("")
                context_parts.append("Event Chain:")
                context_parts.append(chain_context)
        
        return "\n".join(context_parts)
    
    def _generate_structured_context(self, merged_event: MergedEvent, 
                                    query_context: Optional[Dict]) -> str:
        """Generate structured context in a consistent format"""
        lines = []
        
        # Fixed format for easy parsing
        lines.append(f"EVENT_ID: {merged_event.id}")
        lines.append(f"MERGE_COUNT: {merged_event.merge_count}")
        lines.append(f"BASE_EVENT: {merged_event.base_event_id}")
        
        # Latest state
        latest = merged_event.get_latest_state()
        lines.append(f"WHO: {latest.get('who', 'N/A')}")
        lines.append(f"WHAT: {latest.get('what', 'N/A')}")
        lines.append(f"WHEN: {latest.get('when', 'N/A')}")
        lines.append(f"WHERE: {latest.get('where', 'N/A')}")
        lines.append(f"WHY: {latest.get('why', 'N/A')}")
        lines.append(f"HOW: {latest.get('how', 'N/A')}")
        
        # Variations count
        lines.append(f"WHO_VARIANTS: {len(merged_event.who_variants)}")
        lines.append(f"WHAT_VARIANTS: {len(merged_event.what_variants)}")
        lines.append(f"WHY_VARIANTS: {len(merged_event.why_variants)}")
        
        # Temporal range
        if merged_event.when_timeline:
            start = merged_event.when_timeline[0].timestamp
            end = merged_event.when_timeline[-1].timestamp
            lines.append(f"TIME_RANGE: {start.isoformat()} to {end.isoformat()}")
        
        # Dependencies
        if merged_event.depends_on:
            lines.append(f"DEPENDS_ON: {', '.join(merged_event.depends_on)}")
        if merged_event.enables:
            lines.append(f"ENABLES: {', '.join(merged_event.enables)}")
        
        return "\n".join(lines)
    
    def _compute_component_relevance(self, merged_event: MergedEvent, 
                                    query_context: Dict) -> Dict[str, float]:
        """Compute relevance scores for each component based on query context"""
        scores = self._default_relevance_scores()
        
        if not query_context:
            return scores
        
        query = query_context.get('query', {})
        
        # Increase relevance for components mentioned in query
        if isinstance(query, dict):
            for field in ['who', 'what', 'when', 'where', 'why', 'how']:
                if query.get(field):
                    scores[field] = min(scores[field] + 0.5, 1.0)
        
        # Check for temporal queries
        if self._is_temporal_query(query):
            scores['temporal'] = 0.9
            scores['when'] = 0.9
        
        # Check for causal queries
        if self._is_causal_query(query):
            scores['causal'] = 0.9
            scores['why'] = 0.9
        
        return scores
    
    def _default_relevance_scores(self) -> Dict[str, float]:
        """Return default relevance scores"""
        return {
            'who': 0.5,
            'what': 0.7,
            'when': 0.5,
            'where': 0.4,
            'why': 0.6,
            'how': 0.5,
            'temporal': 0.5,
            'causal': 0.5
        }
    
    def _is_temporal_query(self, query: Any) -> bool:
        """Check if query is asking about temporal aspects"""
        temporal_keywords = ['when', 'time', 'date', 'recent', 'latest', 'history', 
                           'timeline', 'progression', 'evolution']
        
        if isinstance(query, dict):
            query_text = ' '.join(str(v) for v in query.values() if v)
        else:
            query_text = str(query)
        
        query_lower = query_text.lower()
        return any(keyword in query_lower for keyword in temporal_keywords)
    
    def _is_causal_query(self, query: Any) -> bool:
        """Check if query is asking about causal relationships"""
        causal_keywords = ['why', 'because', 'cause', 'reason', 'depend', 'enable', 
                          'lead to', 'result', 'consequence']
        
        if isinstance(query, dict):
            query_text = ' '.join(str(v) for v in query.values() if v)
        else:
            query_text = str(query)
        
        query_lower = query_text.lower()
        return any(keyword in query_lower for keyword in causal_keywords)
    
    def _format_who_component(self, merged_event: MergedEvent, relevance: float) -> str:
        """Format the WHO component based on relevance"""
        if not merged_event.who_variants:
            return ""
        
        if relevance < self.relevance_thresholds['who']:
            # Just show dominant
            return merged_event.dominant_pattern.get('who', '')
        
        # Show all actors with frequency
        actors = []
        for who, variants in merged_event.who_variants.items():
            count = len(variants)
            if count > 1:
                actors.append(f"{who} ({count}x)")
            else:
                actors.append(who)
        
        return ", ".join(actors)
    
    def _format_what_component(self, merged_event: MergedEvent, relevance: float) -> str:
        """Format the WHAT component grouped by actor"""
        if not merged_event.what_variants:
            return ""
        
        if relevance < self.relevance_thresholds['what']:
            # Just show latest
            return merged_event.dominant_pattern.get('what', '')
        
        # Group actions by actor
        actor_actions = {}
        for action_key, variants in merged_event.what_variants.items():
            for variant in variants:
                # Get actor from the raw event data if available
                actor = variant.source_event_id if hasattr(variant, 'source_event_id') else 'Unknown'
                # Try to extract actor from merged event raw data
                if hasattr(merged_event, 'raw_events') and variant.source_event_id in merged_event.raw_events:
                    actor = merged_event.raw_events[variant.source_event_id].get('who', 'Unknown')
                
                if actor not in actor_actions:
                    actor_actions[actor] = []
                
                timestamp_str = variant.timestamp.strftime('%Y-%m-%d %H:%M')
                prefix = self._get_relationship_prefix(variant.relationship)
                actor_actions[actor].append(f"    {prefix} {variant.value} [{timestamp_str}]")
        
        # Format by actor
        lines = []
        for actor, actions in sorted(actor_actions.items()):
            lines.append(f"  [{actor}]:")
            lines.extend(actions)
        
        return "\n".join(lines)
    
    def _get_relationship_prefix(self, relationship: EventRelationship) -> str:
        """Get a text prefix for a relationship type"""
        prefixes = {
            EventRelationship.INITIAL: "•",
            EventRelationship.UPDATE: "→ Updated:",
            EventRelationship.CORRECTION: "→ Corrected to:",
            EventRelationship.CONTINUATION: "→ Continued:",
            EventRelationship.VARIATION: "≈ Variation:",
            EventRelationship.SUPERSEDES: "⇒ Superseded by:"
        }
        return prefixes.get(relationship, "•")
    
    def _format_when_component(self, merged_event: MergedEvent, relevance: float) -> str:
        """Format the WHEN component based on relevance"""
        if not merged_event.when_timeline:
            return ""
        
        if len(merged_event.when_timeline) == 1:
            point = merged_event.when_timeline[0]
            return point.semantic_time or point.timestamp.strftime('%Y-%m-%d %H:%M')
        
        # Show time range
        start = merged_event.when_timeline[0].timestamp
        end = merged_event.when_timeline[-1].timestamp
        duration = end - start
        
        if relevance < self.relevance_thresholds['temporal']:
            # Just show range
            return self._format_time_range(start, end, duration)
        
        # Show progression
        lines = [self._format_time_range(start, end, duration)]
        lines.append("  Timeline:")
        
        for point in merged_event.when_timeline[:10]:  # Limit to 10 points
            time_str = point.timestamp.strftime('%H:%M' if duration.days == 0 else '%Y-%m-%d')
            desc = point.description[:50] if point.description else "Event"
            lines.append(f"    - {time_str}: {desc}")
        
        if len(merged_event.when_timeline) > 10:
            lines.append(f"    ... and {len(merged_event.when_timeline) - 10} more events")
        
        return "\n".join(lines)
    
    def _format_time_range(self, start: datetime, end: datetime, duration: timedelta) -> str:
        """Format a time range string"""
        if duration.days > 0:
            return f"{start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')} ({duration.days} days)"
        elif duration.seconds > 3600:
            hours = duration.seconds // 3600
            return f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')} ({hours} hours)"
        else:
            minutes = duration.seconds // 60
            return f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')} ({minutes} minutes)"
    
    def _format_where_component(self, merged_event: MergedEvent) -> str:
        """Format the WHERE component"""
        if not merged_event.where_locations:
            return ""
        
        # Sort by frequency
        sorted_locations = sorted(merged_event.where_locations.items(), 
                                key=lambda x: x[1], reverse=True)
        
        locations = []
        for location, count in sorted_locations[:5]:  # Top 5 locations
            if count > 1:
                locations.append(f"{location} ({count}x)")
            else:
                locations.append(location)
        
        if len(sorted_locations) > 5:
            locations.append(f"... and {len(sorted_locations) - 5} more")
        
        return ", ".join(locations)
    
    def _format_why_component(self, merged_event: MergedEvent, relevance: float) -> str:
        """Format the WHY component showing reason evolution"""
        if not merged_event.why_variants:
            return ""
        
        if relevance < self.relevance_thresholds['why']:
            # Just show latest
            return merged_event.dominant_pattern.get('why', '')
        
        # Show reason evolution
        lines = []
        for why, variants in merged_event.why_variants.items():
            if len(variants) == 1:
                lines.append(f"  • {variants[0].value}")
            else:
                # Show progression
                for i, variant in enumerate(variants):
                    if i == 0:
                        lines.append(f"  Initial: {variant.value}")
                    else:
                        lines.append(f"  → Then: {variant.value}")
        
        return "\n".join(lines)
    
    def _format_how_component(self, merged_event: MergedEvent) -> str:
        """Format the HOW component"""
        if not merged_event.how_methods:
            return ""
        
        # Sort by frequency
        sorted_methods = sorted(merged_event.how_methods.items(), 
                              key=lambda x: x[1], reverse=True)
        
        methods = []
        for method, count in sorted_methods[:3]:  # Top 3 methods
            if count > 1:
                methods.append(f"{method} ({count}x)")
            else:
                methods.append(method)
        
        return ", ".join(methods)
    
    def _format_dependencies(self, merged_event: MergedEvent) -> str:
        """Format dependency relationships"""
        lines = []
        
        if merged_event.depends_on:
            deps = list(merged_event.depends_on)[:5]
            lines.append(f"  Depends on: {', '.join(deps)}")
            if len(merged_event.depends_on) > 5:
                lines.append(f"    ... and {len(merged_event.depends_on) - 5} more")
        
        if merged_event.enables:
            enables = list(merged_event.enables)[:5]
            lines.append(f"  Enables: {', '.join(enables)}")
            if len(merged_event.enables) > 5:
                lines.append(f"    ... and {len(merged_event.enables) - 5} more")
        
        # Format causal links
        for event_id, links in list(merged_event.causal_links.items())[:3]:
            if links.get('causes'):
                lines.append(f"  {event_id} caused by: {', '.join(links['causes'][:3])}")
            if links.get('effects'):
                lines.append(f"  {event_id} leads to: {', '.join(links['effects'][:3])}")
        
        return "\n".join(lines)
    
    def _format_temporal_chain_context(self, merged_event: MergedEvent) -> str:
        """Format temporal chain context if available"""
        if not self.temporal_chain or not merged_event.temporal_chain:
            return ""
        
        lines = []
        
        # Get chain context for the first event in the merged set
        if merged_event.temporal_chain:
            first_event = merged_event.temporal_chain[0]
            chain_context = self.temporal_chain.get_chain_for_event(first_event)
            
            if chain_context:
                lines.append(f"  Part of chain with {len(chain_context)} total events")
                
                # Show position in chain
                try:
                    position = chain_context.index(first_event) + 1
                    lines.append(f"  Position: {position}/{len(chain_context)}")
                except ValueError:
                    pass
        
        return "\n".join(lines)
    
    def generate_batch_context(self, merged_events: List[MergedEvent], 
                              max_length: int = 2000) -> str:
        """
        Generate context for multiple merged events with length limit.
        
        Args:
            merged_events: List of merged events to include
            max_length: Maximum character length for context
            
        Returns:
            Formatted context within length limit
        """
        if not merged_events:
            return ""
        
        contexts = []
        current_length = 0
        
        for event in merged_events:
            # Generate summary for each
            summary = self._generate_summary_context(event)
            summary_with_newline = summary + "\n"
            
            if current_length + len(summary_with_newline) > max_length:
                break
            
            contexts.append(summary)
            current_length += len(summary_with_newline)
        
        if len(contexts) < len(merged_events):
            contexts.append(f"... and {len(merged_events) - len(contexts)} more events")
        
        return "\n".join(contexts)