from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from datetime import datetime
import uuid

EventType = Literal['user_message','llm_message','tool_call','tool_result','system']

def gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

class RawEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: gen_id("evt"))
    session_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor: str  # e.g. 'user:Jake', 'llm:Qwen3-4b-instruct-2507', 'tool:web_search'
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Who(BaseModel):
    type: str  # e.g., 'user', 'llm', 'tool', 'system', 'team', 'user:family', etc.
    id: str
    label: Optional[str] = None

class Where(BaseModel):
    type: str = 'digital'  # e.g., 'physical', 'digital', 'financial', 'academic', 'conceptual', etc.
    value: str  # e.g., 'flask_ui/session:abc'; could also be domain or file path
    lat: Optional[float] = None
    lon: Optional[float] = None

class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: gen_id("mem"))
    session_id: str
    source_event_id: str
    who: Who
    what: str
    when: datetime
    where: Where
    why: str
    how: str
    raw_text: str
    token_count: int
    embed_text: str
    embed_model: str
    extra: Dict[str, Any] = Field(default_factory=dict)

class MemoryPointer(BaseModel):
    block_id: str
    relation: Literal['prev','next','continuation']
    note: Optional[str] = None

class MemoryBlock(BaseModel):
    block_id: str = Field(default_factory=lambda: gen_id("blk"))
    query_fingerprint: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    budget_tokens: int
    used_tokens: int
    has_more: bool = False
    prev_block_id: Optional[str] = None
    next_block_id: Optional[str] = None
    summary_text: Optional[str] = None
    member_ids: List[str] = Field(default_factory=list)  # ordered list of memory_ids

class RetrievalQuery(BaseModel):
    session_id: str
    actor_hint: Optional[str] = None  # Maps to WHO
    temporal_hint: Optional[Union[str, Tuple[str, str], Dict]] = None  # Maps to WHEN - Date, date range, or relative time
    text: str  # Used for WHAT/WHY/HOW semantic search

class Candidate(BaseModel):
    memory_id: str
    score: float
    token_count: int
    # Optional component scores for debugging
    base_score: Optional[float] = None
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    recency_score: Optional[float] = None
    importance_score: Optional[float] = None
    actor_score: Optional[float] = None
    temporal_score: Optional[float] = None
    spatial_score: Optional[float] = None
    usage_score: Optional[float] = None
    attention_score: Optional[float] = None
    # Detailed attention components (when using attention as reranker)
    attention_components: Optional[Dict[str, float]] = None
