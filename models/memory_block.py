"""
Memory Block model for linked memory chains

Groups related memories based on content-addressable properties (5W1H)
and provides contextual linkage for better retrieval and understanding.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import uuid
import numpy as np

from models.event import Event, FiveW1H, EventType

class LinkType(Enum):
    """Types of links between memories"""
    TEMPORAL = "temporal"          # Sequential in time
    CAUSAL = "causal"             # Action->Observation pairs
    SEMANTIC = "semantic"          # Similar content/meaning
    EPISODIC = "episodic"         # Same episode/session
    CONVERSATIONAL = "conversational"  # Dialog chain
    REFERENCE = "reference"        # Explicit references

@dataclass
class MemoryLink:
    """Link between two memories with metadata"""
    source_id: str          # Source event ID
    target_id: str          # Target event ID
    link_type: LinkType     # Type of relationship
    strength: float         # Link strength (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.link_type))

@dataclass
class MemoryBlock:
    """
    A block of linked memories forming a coherent unit
    
    Memory blocks are content-addressable through their aggregate 5W1H signature
    and can span temporally separated but semantically related memories.
    """
    
    # Block identification
    id: str = field(default_factory=lambda: f"mb_{uuid.uuid4().hex[:12]}")
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Core memories in this block
    events: List[Event] = field(default_factory=list)
    event_ids: Set[str] = field(default_factory=set)
    
    # Links between memories
    links: List[MemoryLink] = field(default_factory=list)
    
    # Aggregate 5W1H signature (content-addressable index)
    aggregate_signature: Optional[FiveW1H] = None
    
    # Block metadata
    block_type: str = "general"  # conversation, task, knowledge, etc.
    salience: float = 0.5        # Block-level salience
    coherence_score: float = 0.0 # How well memories fit together
    tags: List[str] = field(default_factory=list)
    
    # Statistics
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def add_event(self, event: Event, link_to: Optional[str] = None, 
                  link_type: LinkType = LinkType.TEMPORAL, link_strength: float = 0.8):
        """Add an event to the block with optional linking"""
        if event.id not in self.event_ids:
            self.events.append(event)
            self.event_ids.add(event.id)
            
            # Create link if specified
            if link_to and link_to in self.event_ids:
                link = MemoryLink(
                    source_id=link_to,
                    target_id=event.id,
                    link_type=link_type,
                    strength=link_strength
                )
                self.links.append(link)
            
            # Update aggregate signature
            self._update_aggregate_signature()
            self.updated_at = datetime.utcnow()
    
    def merge_with(self, other: 'MemoryBlock'):
        """Merge another block into this one"""
        for event in other.events:
            if event.id not in self.event_ids:
                self.events.append(event)
                self.event_ids.add(event.id)
        
        # Merge links, avoiding duplicates
        existing_links = {(l.source_id, l.target_id, l.link_type) for l in self.links}
        for link in other.links:
            link_key = (link.source_id, link.target_id, link.link_type)
            if link_key not in existing_links:
                self.links.append(link)
        
        # Update metadata
        self._update_aggregate_signature()
        self.coherence_score = self.compute_coherence()
        self.updated_at = datetime.utcnow()
    
    def _update_aggregate_signature(self):
        """Update the aggregate 5W1H signature based on all events"""
        if not self.events:
            self.aggregate_signature = None
            return
        
        # Collect all 5W1H fields
        whos = []
        whats = []
        whens = []
        wheres = []
        whys = []
        hows = []
        
        for event in self.events:
            if event.five_w1h.who:
                whos.append(event.five_w1h.who)
            if event.five_w1h.what:
                whats.append(event.five_w1h.what)
            if event.five_w1h.when:
                whens.append(event.five_w1h.when)
            if event.five_w1h.where:
                wheres.append(event.five_w1h.where)
            if event.five_w1h.why:
                whys.append(event.five_w1h.why)
            if event.five_w1h.how:
                hows.append(event.five_w1h.how)
        
        # Create aggregate signature
        self.aggregate_signature = FiveW1H(
            who=self._aggregate_field(whos, "who"),
            what=self._aggregate_field(whats, "what"),
            when=self._aggregate_temporal(whens),
            where=self._aggregate_field(wheres, "where"),
            why=self._aggregate_field(whys, "why"),
            how=self._aggregate_field(hows, "how")
        )
    
    def _aggregate_field(self, values: List[str], field_name: str) -> str:
        """Aggregate multiple field values into a representative string"""
        if not values:
            return ""
        
        # For 'who' field, list unique actors
        if field_name == "who":
            unique_actors = list(set(values))
            if len(unique_actors) == 1:
                return unique_actors[0]
            return f"[{', '.join(unique_actors[:3])}]"
        
        # For 'what' field, create a summary
        if field_name == "what":
            if len(values) == 1:
                return values[0]
            # Take first and last for context
            return f"{values[0][:50]}... -> {values[-1][:50]}..."
        
        # For other fields, find most common or representative
        if len(values) == 1:
            return values[0]
        
        # Find most common value
        from collections import Counter
        counter = Counter(values)
        most_common = counter.most_common(1)[0][0]
        
        if counter[most_common] > len(values) / 2:
            return most_common
        else:
            # Multiple values, indicate variety
            unique_vals = list(set(values))[:3]
            return f"[{', '.join(unique_vals)}]"
    
    def _aggregate_temporal(self, whens: List[str]) -> str:
        """Aggregate temporal information"""
        if not whens:
            return ""
        
        if len(whens) == 1:
            return whens[0]
        
        # Try to parse as timestamps and get range
        try:
            timestamps = []
            for when in whens:
                if when and when[0].isdigit():
                    # Likely a timestamp
                    timestamps.append(when)
            
            if timestamps:
                return f"{timestamps[0]} to {timestamps[-1]}"
        except:
            pass
        
        return f"[{len(whens)} events]"
    
    def compute_coherence(self) -> float:
        """
        Compute coherence score for the block
        Higher score means memories are well-connected and related
        """
        if len(self.events) <= 1:
            return 1.0
        
        factors = []
        
        # Factor 1: Link density (how interconnected are the memories)
        max_links = len(self.events) * (len(self.events) - 1) / 2
        link_density = len(self.links) / max_links if max_links > 0 else 0
        factors.append(link_density * 0.3)
        
        # Factor 2: Temporal coherence (are events close in time)
        if len(self.events) > 1:
            time_spans = []
            for i in range(len(self.events) - 1):
                span = (self.events[i+1].created_at - self.events[i].created_at).total_seconds()
                time_spans.append(span)
            
            # Normalize by 5 minutes (300 seconds)
            avg_span = np.mean(time_spans) if time_spans else 0
            temporal_coherence = np.exp(-avg_span / 300)  # Exponential decay
            factors.append(temporal_coherence * 0.3)
        
        # Factor 3: Semantic similarity (shared 5W1H fields)
        semantic_overlaps = []
        for i in range(len(self.events)):
            for j in range(i + 1, len(self.events)):
                overlap = self._compute_5w1h_overlap(
                    self.events[i].five_w1h,
                    self.events[j].five_w1h
                )
                semantic_overlaps.append(overlap)
        
        avg_overlap = np.mean(semantic_overlaps) if semantic_overlaps else 0
        factors.append(avg_overlap * 0.4)
        
        return sum(factors)
    
    def _compute_5w1h_overlap(self, a: FiveW1H, b: FiveW1H) -> float:
        """Compute overlap between two 5W1H structures"""
        overlaps = []
        
        # Check each field
        fields = ['who', 'where', 'why', 'how']
        for field in fields:
            val_a = getattr(a, field, "")
            val_b = getattr(b, field, "")
            if val_a and val_b:
                if val_a == val_b:
                    overlaps.append(1.0)
                elif val_a.lower() in val_b.lower() or val_b.lower() in val_a.lower():
                    overlaps.append(0.5)
                else:
                    overlaps.append(0.0)
        
        # Special handling for 'what' - use word overlap
        what_overlap = self._text_similarity(a.what, b.what)
        overlaps.append(what_overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_block_salience(self) -> float:
        """
        Compute aggregate salience for the entire block
        Considers individual event saliences and their relationships
        """
        if not self.events:
            return 0.0
        
        # Weighted average of event saliences
        event_saliences = [e.salience for e in self.events]
        avg_salience = np.mean(event_saliences)
        
        # Boost for coherence
        coherence_boost = self.coherence_score * 0.2
        
        # Boost for conversation patterns (Q&A pairs)
        conversation_boost = 0.0
        user_events = [e for e in self.events if e.five_w1h.who == "User"]
        assistant_events = [e for e in self.events if e.five_w1h.who == "Assistant"]
        if user_events and assistant_events:
            conversation_boost = 0.15
        
        # Boost for causal chains
        causal_boost = 0.0
        causal_links = [l for l in self.links if l.link_type == LinkType.CAUSAL]
        if causal_links:
            causal_boost = min(0.2, len(causal_links) * 0.05)
        
        # Combine factors
        block_salience = min(1.0, avg_salience + coherence_boost + conversation_boost + causal_boost)
        
        self.salience = block_salience
        return block_salience
    
    def get_event_graph(self) -> Dict[str, List[Tuple[str, LinkType, float]]]:
        """
        Get graph representation of memory links
        Returns: dict mapping event_id to list of (target_id, link_type, strength)
        """
        graph = {event.id: [] for event in self.events}
        
        for link in self.links:
            if link.source_id in graph:
                graph[link.source_id].append(
                    (link.target_id, link.link_type, link.strength)
                )
        
        return graph
    
    def get_conversation_threads(self) -> List[List[Event]]:
        """Extract conversation threads from the block"""
        threads = []
        current_thread = []
        
        for event in sorted(self.events, key=lambda e: e.created_at):
            if event.five_w1h.who in ["User", "Assistant"]:
                current_thread.append(event)
            elif current_thread:
                if len(current_thread) > 1:
                    threads.append(current_thread)
                current_thread = []
        
        if current_thread and len(current_thread) > 1:
            threads.append(current_thread)
        
        return threads
    
    def update_access(self):
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        
        # Also update individual events
        for event in self.events:
            event.update_access()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "event_ids": list(self.event_ids),
            "links": [
                {
                    "source_id": l.source_id,
                    "target_id": l.target_id,
                    "link_type": l.link_type.value,
                    "strength": l.strength,
                    "metadata": l.metadata
                }
                for l in self.links
            ],
            "aggregate_signature": self.aggregate_signature.to_dict() if self.aggregate_signature else None,
            "block_type": self.block_type,
            "salience": self.salience,
            "coherence_score": self.coherence_score,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], events: List[Event]) -> 'MemoryBlock':
        """Create from dictionary with event list"""
        block = cls(
            id=data.get("id"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow(),
            block_type=data.get("block_type", "general"),
            salience=data.get("salience", 0.5),
            coherence_score=data.get("coherence_score", 0.0),
            tags=data.get("tags", []),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
        )
        
        # Add events
        event_ids = set(data.get("event_ids", []))
        for event in events:
            if event.id in event_ids:
                block.events.append(event)
                block.event_ids.add(event.id)
        
        # Reconstruct links
        for link_data in data.get("links", []):
            link = MemoryLink(
                source_id=link_data["source_id"],
                target_id=link_data["target_id"],
                link_type=LinkType(link_data["link_type"]),
                strength=link_data["strength"],
                metadata=link_data.get("metadata", {})
            )
            block.links.append(link)
        
        # Restore aggregate signature
        if data.get("aggregate_signature"):
            block.aggregate_signature = FiveW1H.from_dict(data["aggregate_signature"])
        
        return block
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"MemoryBlock(id={self.id[:8]}, events={len(self.events)}, "
            f"links={len(self.links)}, salience={self.salience:.2f}, "
            f"coherence={self.coherence_score:.2f})"
        )