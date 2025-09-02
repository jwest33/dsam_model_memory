"""
Merged Event Model with 5W1H Decomposition and Dependency Tracking

This module provides a comprehensive representation of merged events that preserves
all component variations, temporal relationships, and causal dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import json
from enum import Enum


class EventRelationship(Enum):
    """Types of relationships between events in a temporal chain"""
    UPDATE = "update"          # Newer information updates older
    CORRECTION = "correction"   # Fixes incorrect information
    CONTINUATION = "continuation"  # Continues from previous
    VARIATION = "variation"     # Alternative version
    SUPERSEDES = "supersedes"   # Completely replaces
    INITIAL = "initial"         # First in chain


@dataclass
class ComponentVariant:
    """Represents a variant of a 5W1H component"""
    value: str
    timestamp: datetime
    event_id: str
    relationship: EventRelationship = EventRelationship.INITIAL
    context: Optional[Dict[str, Any]] = None
    version: int = 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'relationship': self.relationship.value,
            'context': self.context or {},
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComponentVariant':
        """Create from dictionary"""
        return cls(
            value=data['value'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_id=data['event_id'],
            relationship=EventRelationship(data.get('relationship', 'initial')),
            context=data.get('context'),
            version=data.get('version', 1)
        )


@dataclass
class TemporalPoint:
    """Represents a point in the temporal timeline"""
    timestamp: datetime
    semantic_time: Optional[str]  # e.g., "yesterday", "last week"
    event_id: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'semantic_time': self.semantic_time,
            'event_id': self.event_id,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalPoint':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            semantic_time=data.get('semantic_time'),
            event_id=data['event_id'],
            description=data.get('description')
        )


@dataclass
class MergedEvent:
    """
    Enhanced merged event with full component tracking and dependency management.
    
    This class represents a collection of similar events that have been merged,
    preserving all variations in the 5W1H components while tracking temporal
    and causal relationships.
    """
    
    id: str
    base_event_id: str  # First event that started this merge group
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Raw event tracking
    raw_event_ids: Set[str] = field(default_factory=set)
    merge_count: int = 1
    
    # Component decomposition - each maintains full history
    who_variants: Dict[str, List[ComponentVariant]] = field(default_factory=dict)
    what_variants: Dict[str, List[ComponentVariant]] = field(default_factory=dict)
    when_timeline: List[TemporalPoint] = field(default_factory=list)
    where_locations: Dict[str, int] = field(default_factory=dict)  # location -> frequency
    why_variants: Dict[str, List[ComponentVariant]] = field(default_factory=dict)
    how_methods: Dict[str, int] = field(default_factory=dict)  # method -> frequency
    
    # Temporal chain tracking
    temporal_chain: List[str] = field(default_factory=list)  # Ordered event IDs
    supersedes: Optional[str] = None  # ID of event this supersedes
    superseded_by: Optional[str] = None  # ID of event that supersedes this
    
    # Dependency tracking
    depends_on: Set[str] = field(default_factory=set)  # Event IDs this depends on
    enables: Set[str] = field(default_factory=set)  # Event IDs this enables
    causal_links: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # causal_links structure: {event_id: {'causes': [...], 'effects': [...]}}
    
    # Semantic evolution tracking
    embedding_history: List[Dict] = field(default_factory=list)  # Track centroid movement
    dominant_pattern: Optional[Dict[str, Any]] = None  # Most representative pattern
    
    # Group-level characterization fields
    group_why: Optional[str] = None  # LLM-generated purpose of the entire group
    group_how: Optional[str] = None  # LLM-generated mechanism characterizing the group
    group_fields_generated_at: Optional[datetime] = None
    group_fields_method: Optional[str] = None  # 'llm' or 'heuristic'
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    confidence_score: float = 1.0
    coherence_score: float = 1.0
    
    def add_raw_event(self, event_id: str, event_data: Dict[str, Any], 
                      relationship: EventRelationship = EventRelationship.VARIATION):
        """
        Add a raw event to this merged event, decomposing its components.
        
        Args:
            event_id: ID of the raw event
            event_data: Dictionary containing the event's 5W1H data
            relationship: How this event relates to existing ones
        """
        self.raw_event_ids.add(event_id)
        timestamp = event_data.get('timestamp', datetime.utcnow())
        
        # Process WHO
        if event_data.get('who'):
            who = event_data['who']
            who_key = self._normalize_text(who)  # Normalize for deduplication
            
            # Check if this exact value already exists in recent variants
            should_add = True
            if who_key in self.who_variants and self.who_variants[who_key]:
                # Check if the last variant has the same value
                last_variant = self.who_variants[who_key][-1]
                if last_variant.value == who:
                    should_add = False  # Don't add duplicate
            
            if should_add:
                if who_key not in self.who_variants:
                    self.who_variants[who_key] = []
                self.who_variants[who_key].append(ComponentVariant(
                    value=who,
                    timestamp=timestamp,
                    event_id=event_id,
                    relationship=relationship,
                    version=len(self.who_variants[who_key]) + 1
                ))
        
        # Process WHAT
        if event_data.get('what'):
            what = event_data['what']
            what_key = self._normalize_text(what)
            
            # Check if this exact value already exists in recent variants
            should_add = True
            if what_key in self.what_variants and self.what_variants[what_key]:
                # Check if the last variant has the same value
                last_variant = self.what_variants[what_key][-1]
                if last_variant.value == what:
                    should_add = False  # Don't add duplicate
            
            if should_add:
                if what_key not in self.what_variants:
                    self.what_variants[what_key] = []
                self.what_variants[what_key].append(ComponentVariant(
                    value=what,
                    timestamp=timestamp,
                    event_id=event_id,
                    relationship=relationship,
                    version=len(self.what_variants[what_key]) + 1
                ))
        
        # Process WHEN
        self.when_timeline.append(TemporalPoint(
            timestamp=timestamp,
            semantic_time=event_data.get('when'),
            event_id=event_id,
            description=event_data.get('what', '')[:50]
        ))
        self.when_timeline.sort(key=lambda x: x.timestamp)
        
        # Process WHERE
        if event_data.get('where'):
            where = event_data['where']
            self.where_locations[where] = self.where_locations.get(where, 0) + 1
        
        # Process WHY
        if event_data.get('why'):
            why = event_data['why']
            why_key = self._normalize_text(why)  # Normalize for deduplication
            
            # Check if this exact value already exists in recent variants
            should_add = True
            if why_key in self.why_variants and self.why_variants[why_key]:
                # Check if the last variant has the same value
                last_variant = self.why_variants[why_key][-1]
                if last_variant.value == why:
                    should_add = False  # Don't add duplicate
            
            if should_add:
                if why_key not in self.why_variants:
                    self.why_variants[why_key] = []
                self.why_variants[why_key].append(ComponentVariant(
                    value=why,
                    timestamp=timestamp,
                    event_id=event_id,
                    relationship=relationship,
                    version=len(self.why_variants[why_key]) + 1
                ))
        
        # Process HOW
        if event_data.get('how'):
            how = event_data['how']
            self.how_methods[how] = self.how_methods.get(how, 0) + 1
        
        # Update temporal chain
        self.temporal_chain.append(event_id)
        self.merge_count += 1
        self.last_updated = datetime.utcnow()
        
        # Recompute dominant pattern
        self._update_dominant_pattern()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for grouping similar variations"""
        # Simple normalization - can be enhanced with NLP
        return text.lower().strip()[:100]  # Use first 100 chars as key
    
    def _update_dominant_pattern(self):
        """Update the dominant pattern based on current components"""
        self.dominant_pattern = {
            'who': self._get_dominant_who(),
            'what': self._get_dominant_what(),
            'when': self._get_temporal_summary(),
            'where': self._get_dominant_where(),
            'why': self._get_dominant_why(),
            'how': self._get_dominant_how()
        }
    
    def _get_dominant_who(self) -> Optional[str]:
        """Get the most frequent actor"""
        if not self.who_variants:
            return None
        return max(self.who_variants.items(), key=lambda x: len(x[1]))[0]
    
    def _get_dominant_what(self) -> Optional[str]:
        """Get the most representative action"""
        if not self.what_variants:
            return None
        # Return the latest version of the most frequent action
        most_frequent = max(self.what_variants.items(), key=lambda x: len(x[1]))
        return most_frequent[1][-1].value if most_frequent[1] else None
    
    def _get_temporal_summary(self) -> str:
        """Get a summary of the temporal range"""
        if not self.when_timeline:
            return "No temporal data"
        if len(self.when_timeline) == 1:
            return self.when_timeline[0].semantic_time or self.when_timeline[0].timestamp.isoformat()
        
        start = self.when_timeline[0].timestamp
        end = self.when_timeline[-1].timestamp
        duration = end - start
        
        if duration.days > 0:
            return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({duration.days} days)"
        elif duration.seconds > 3600:
            hours = duration.seconds // 3600
            return f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')} ({hours} hours)"
        else:
            return f"{start.strftime('%H:%M')} to {end.strftime('%H:%M')}"
    
    def _get_dominant_where(self) -> Optional[str]:
        """Get the most frequent location"""
        if not self.where_locations:
            return None
        return max(self.where_locations.items(), key=lambda x: x[1])[0]
    
    def _get_dominant_why(self) -> Optional[str]:
        """Get the most representative reason"""
        if not self.why_variants:
            return None
        # Return the latest reason
        latest_reasons = []
        for reason_list in self.why_variants.values():
            if reason_list:
                latest_reasons.append(reason_list[-1])
        if latest_reasons:
            return max(latest_reasons, key=lambda x: x.timestamp).value
        return None
    
    def _get_dominant_how(self) -> Optional[str]:
        """Get the most frequent method"""
        if not self.how_methods:
            return None
        return max(self.how_methods.items(), key=lambda x: x[1])[0]
    
    def add_dependency(self, event_id: str, depends_on: List[str] = None, 
                      enables: List[str] = None):
        """Add dependency relationships"""
        if depends_on:
            self.depends_on.update(depends_on)
            for dep_id in depends_on:
                if event_id not in self.causal_links:
                    self.causal_links[event_id] = {'causes': [], 'effects': []}
                self.causal_links[event_id]['causes'].append(dep_id)
        
        if enables:
            self.enables.update(enables)
            for enabled_id in enables:
                if event_id not in self.causal_links:
                    self.causal_links[event_id] = {'causes': [], 'effects': []}
                self.causal_links[event_id]['effects'].append(enabled_id)
    
    def get_latest_state(self) -> Dict[str, Any]:
        """Get the most recent state of all components"""
        latest = {}
        
        # Get latest WHO
        if self.who_variants:
            latest_who = []
            for who, variants in self.who_variants.items():
                if variants:
                    latest_who.append(variants[-1])
            if latest_who:
                latest['who'] = max(latest_who, key=lambda x: x.timestamp).value
        
        # Get latest WHAT
        if self.what_variants:
            all_variants = []
            for variants_list in self.what_variants.values():
                all_variants.extend(variants_list)
            if all_variants:
                latest['what'] = max(all_variants, key=lambda x: x.timestamp).value
        
        # Get latest WHEN
        if self.when_timeline:
            latest['when'] = self.when_timeline[-1].semantic_time or \
                           self.when_timeline[-1].timestamp.isoformat()
        else:
            latest['when'] = datetime.utcnow().isoformat()
        
        # Most frequent WHERE
        where = self._get_dominant_where()
        if where:
            latest['where'] = where
        
        # Latest WHY  
        why = self._get_dominant_why()
        if why:
            latest['why'] = why
        
        # Most frequent HOW
        how = self._get_dominant_how()
        if how:
            latest['how'] = how
        
        # Ensure we always have at least empty strings for display
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            if field not in latest or latest[field] is None:
                latest[field] = ''
        
        return latest
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'base_event_id': self.base_event_id,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'raw_event_ids': list(self.raw_event_ids),
            'merge_count': self.merge_count,
            'who_variants': {
                who: [v.to_dict() for v in variants]
                for who, variants in self.who_variants.items()
            },
            'what_variants': {
                what: [v.to_dict() for v in variants]
                for what, variants in self.what_variants.items()
            },
            'when_timeline': [tp.to_dict() for tp in self.when_timeline],
            'where_locations': self.where_locations,
            'why_variants': {
                why: [v.to_dict() for v in variants]
                for why, variants in self.why_variants.items()
            },
            'how_methods': self.how_methods,
            'temporal_chain': self.temporal_chain,
            'supersedes': self.supersedes,
            'superseded_by': self.superseded_by,
            'depends_on': list(self.depends_on),
            'enables': list(self.enables),
            'causal_links': self.causal_links,
            'embedding_history': self.embedding_history,
            'dominant_pattern': self.dominant_pattern,
            'tags': list(self.tags),
            'confidence_score': self.confidence_score,
            'coherence_score': self.coherence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MergedEvent':
        """Create from dictionary"""
        merged = cls(
            id=data['id'],
            base_event_id=data['base_event_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )
        
        merged.raw_event_ids = set(data.get('raw_event_ids', []))
        merged.merge_count = data.get('merge_count', 1)
        
        # Restore component variants
        for who, variants in data.get('who_variants', {}).items():
            merged.who_variants[who] = [ComponentVariant.from_dict(v) for v in variants]
        
        for what, variants in data.get('what_variants', {}).items():
            merged.what_variants[what] = [ComponentVariant.from_dict(v) for v in variants]
        
        merged.when_timeline = [TemporalPoint.from_dict(tp) 
                               for tp in data.get('when_timeline', [])]
        merged.where_locations = data.get('where_locations', {})
        
        for why, variants in data.get('why_variants', {}).items():
            merged.why_variants[why] = [ComponentVariant.from_dict(v) for v in variants]
        
        merged.how_methods = data.get('how_methods', {})
        
        # Restore relationships
        merged.temporal_chain = data.get('temporal_chain', [])
        merged.supersedes = data.get('supersedes')
        merged.superseded_by = data.get('superseded_by')
        merged.depends_on = set(data.get('depends_on', []))
        merged.enables = set(data.get('enables', []))
        merged.causal_links = data.get('causal_links', {})
        
        # Restore metadata
        merged.embedding_history = data.get('embedding_history', [])
        merged.dominant_pattern = data.get('dominant_pattern')
        merged.tags = set(data.get('tags', []))
        merged.confidence_score = data.get('confidence_score', 1.0)
        merged.coherence_score = data.get('coherence_score', 1.0)
        
        return merged