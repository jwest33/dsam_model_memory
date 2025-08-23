"""
Event models for 5W1H + MHN Memory Framework

Every interaction is stored as a 5W1H tuple with episode linkage.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import uuid

class EventType(Enum):
    """Types of memory events"""
    ACTION = "action"  # Agent performs an action
    OBSERVATION = "observation"  # Result or feedback from action
    USER_INPUT = "user_input"  # Input from user
    SYSTEM_EVENT = "system_event"  # System-level events

@dataclass
class FiveW1H:
    """The 5W1H structure for memory events"""
    
    who: str  # Agent/actor (e.g., "LLM", "user:jake", "system:retriever")
    what: str  # Action or content (e.g., "generated SQL query", "retrieved document")
    when: str  # Timestamp or step (e.g., ISO timestamp, "step:42")
    where: str  # Context or location (e.g., "conversation:chat42", "file:data.csv")
    why: str  # Intent or trigger (e.g., "user requested data", "follow-up to error")
    how: str  # Mechanism used (e.g., "via Python tool", "API:search()")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            "who": self.who,
            "what": self.what,
            "when": self.when,
            "where": self.where,
            "why": self.why,
            "how": self.how
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'FiveW1H':
        """Create from dictionary"""
        return cls(
            who=data.get("who", ""),
            what=data.get("what", ""),
            when=data.get("when", ""),
            where=data.get("where", ""),
            why=data.get("why", ""),
            how=data.get("how", "")
        )
    
    def get_partial_query(self, slots: List[str]) -> Dict[str, str]:
        """Get partial 5W1H with only specified slots"""
        result = {}
        for slot in slots:
            if hasattr(self, slot):
                result[slot] = getattr(self, slot)
        return result
    
    def fill_missing(self, other: 'FiveW1H') -> 'FiveW1H':
        """Fill missing slots from another 5W1H"""
        return FiveW1H(
            who=self.who or other.who,
            what=self.what or other.what,
            when=self.when or other.when,
            where=self.where or other.where,
            why=self.why or other.why,
            how=self.how or other.how
        )

@dataclass
class Event:
    """Complete event with 5W1H, metadata, and linkage"""
    
    # Core 5W1H structure
    five_w1h: FiveW1H
    
    # Event metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.ACTION
    episode_id: str = field(default_factory=lambda: f"e{uuid.uuid4().hex[:8]}")
    
    # Memory metadata
    salience: float = 0.5  # Importance score (0-1)
    confidence: float = 1.0  # Certainty level (0-1)
    tags: List[str] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    
    # Optional content
    full_content: Optional[str] = None  # Full text if truncated in 'what'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def timestamp(self) -> str:
        """Get ISO timestamp from when field or created_at"""
        if self.five_w1h.when and self.five_w1h.when.startswith("20"):
            return self.five_w1h.when
        return self.created_at.isoformat() + "Z"
    
    @property
    def age_seconds(self) -> float:
        """Get age of event in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score for eviction (lower = keep, higher = evict)"""
        # Normalize factors
        salience_norm = self.salience  # Already 0-1
        usage_norm = min(1.0, self.accessed_count / 10.0)  # Cap at 10 accesses
        age_norm = min(1.0, self.age_seconds / (7 * 24 * 3600))  # Normalize to 1 week
        
        # Higher score = more likely to evict
        # We want: low salience, low usage, old age = high score
        score = (
            1.2 * (1 - salience_norm) +  # Low salience increases score
            0.6 * (1 - usage_norm) +      # Low usage increases score
            0.6 * age_norm                # Old age increases score
        )
        return score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "episode_id": self.episode_id,
            "five_w1h": self.five_w1h.to_dict(),
            "salience": self.salience,
            "confidence": self.confidence,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "accessed_count": self.accessed_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "full_content": self.full_content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            event_type=EventType(data.get("event_type", "action")),
            episode_id=data.get("episode_id", f"e{uuid.uuid4().hex[:8]}"),
            five_w1h=FiveW1H.from_dict(data.get("five_w1h", {})),
            salience=data.get("salience", 0.5),
            confidence=data.get("confidence", 1.0),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            accessed_count=data.get("accessed_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            full_content=data.get("full_content"),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def create_action(
        cls,
        who: str,
        what: str,
        where: str,
        why: str,
        how: str,
        episode_id: Optional[str] = None,
        **kwargs
    ) -> 'Event':
        """Convenience method to create an action event"""
        return cls(
            five_w1h=FiveW1H(
                who=who,
                what=what,
                when=datetime.utcnow().isoformat() + "Z",
                where=where,
                why=why,
                how=how
            ),
            event_type=EventType.ACTION,
            episode_id=episode_id or f"e{uuid.uuid4().hex[:8]}",
            **kwargs
        )
    
    @classmethod
    def create_observation(
        cls,
        who: str,
        what: str,
        where: str,
        episode_id: str,  # Required to link to action
        why: Optional[str] = None,
        how: Optional[str] = None,
        **kwargs
    ) -> 'Event':
        """Convenience method to create an observation event"""
        return cls(
            five_w1h=FiveW1H(
                who=who,
                what=what,
                when=datetime.utcnow().isoformat() + "Z",
                where=where,
                why=why or f"in response to episode {episode_id}",
                how=how or "observation"
            ),
            event_type=EventType.OBSERVATION,
            episode_id=episode_id,
            **kwargs
        )
    
    def update_access(self):
        """Update access tracking"""
        self.accessed_count += 1
        self.last_accessed = datetime.utcnow()
    
    def matches_query(self, query: Dict[str, str], threshold: float = 0.8) -> bool:
        """Check if event matches a partial 5W1H query"""
        matches = 0
        total = 0
        
        for key, value in query.items():
            if hasattr(self.five_w1h, key):
                total += 1
                event_value = getattr(self.five_w1h, key)
                if value.lower() in event_value.lower():
                    matches += 1
        
        if total == 0:
            return False
        
        return (matches / total) >= threshold
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"Event(id={self.id[:8]}, type={self.event_type.value}, "
            f"who={self.five_w1h.who}, what={self.five_w1h.what[:50]}...)"
        )