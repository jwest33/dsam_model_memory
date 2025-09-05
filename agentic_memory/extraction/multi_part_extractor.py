"""
Multi-part memory extractor that breaks complex information into separate 5W1H memories.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from ..types import RawEvent, MemoryRecord, Who, Where
from ..tokenization import TokenizerAdapter
from ..config import cfg
from sentence_transformers import SentenceTransformer

# Initialize embedder once at module level to avoid reloading
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(cfg.embed_model_name)
    return _embedder

MULTI_PART_PROMPT = """You are a structured-information extractor that identifies DISTINCT pieces of information and converts EACH into separate 5W1H fields.

IMPORTANT: Break down complex or multi-part information into SEPARATE, atomic memories.

For example, if the input contains:
- Multiple facts about different topics
- A list of items or events
- Multiple actions or outcomes
- Different pieces of information about the same topic

Create a SEPARATE memory for each distinct piece of information.

Return a JSON array where each element follows this schema:
[
  {
    "who": { "type": "user|llm|tool|system", "id": "<string>", "label": "<optional string>" },
    "what": "<ONE specific fact, action, or piece of information>",
    "when": "<ISO 8601 timestamp>",
    "where": { "type": "physical|digital", "value": "<context>" },
    "why": "<specific intent for this piece>",
    "how": "<method for this specific piece>"
  },
  // ... more memories if there are more distinct pieces
]

Guidelines:
1. Each memory should be self-contained and independently searchable
2. Don't combine unrelated facts into one memory
3. Break lists into individual items
4. Separate different attributes of the same subject into different memories
5. Keep each "what" field focused on a single fact or action

Return ONLY the JSON array, no other text.
"""

def _call_llm_multi(content: str) -> Optional[List[Dict[str, Any]]]:
    """Call LLM to extract multiple 5W1H structures from content."""
    import requests
    url = f"{cfg.llm_base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": MULTI_PART_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": 2048  # Allow longer responses for multiple memories
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]
        
        # Try to parse JSON array
        start = out.find('[')
        end = out.rfind(']')
        if start >= 0 and end > start:
            obj = json.loads(out[start:end+1])
            if isinstance(obj, list):
                return obj
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return None
    return None

def extract_multi_part_5w1h(raw: RawEvent, context_hint: str = "") -> List[MemoryRecord]:
    """
    Extract multiple 5W1H memory records from a single raw event.
    Returns a list of MemoryRecord objects.
    """
    content = f"EventType: {raw.event_type}\nActor: {raw.actor}\nTimestamp: {raw.timestamp.isoformat()}\nContent: {raw.content}\nMetadata: {raw.metadata}\nContext: {context_hint}"
    
    # Try multi-part extraction
    parsed_list = _call_llm_multi(content)
    
    if not parsed_list or len(parsed_list) == 0:
        # Fallback to single memory using simple extraction
        who_type = 'tool' if raw.event_type in ('tool_call','tool_result') else ('user' if raw.event_type=='user_message' else ('llm' if raw.event_type=='llm_message' else 'system'))
        parsed_list = [{
            'who': {'type': who_type, 'id': raw.actor, 'label': None},
            'what': raw.metadata.get('operation') or raw.content[:160],
            'where': {'type': 'digital', 'value': raw.metadata.get('location') or 'local_ui'},
            'why': raw.metadata.get('intent') or 'unspecified',
            'how': raw.metadata.get('method') or 'message'
        }]
    
    # Create embeddings for each memory
    embedder = _get_embedder()
    token_counter = TokenizerAdapter()
    
    memories = []
    for idx, parsed in enumerate(parsed_list):
        try:
            # Extract fields
            who = Who(**parsed['who'])
            what = parsed.get('what', '').strip() or f"Part {idx+1} of {raw.content[:100]}"
            w = parsed.get('where', {'type': 'digital', 'value': 'local_ui'})
            where = Where(type=w.get('type', 'digital'), value=w.get('value', 'local_ui'))
            why = parsed.get('why', 'unspecified')
            how = parsed.get('how', 'message')
            
            # Create embedding for this specific memory
            embed_text = f"WHAT: {what}\nWHY: {why}\nHOW: {how}"
            vec = embedder.encode([embed_text], normalize_embeddings=True)[0]
            token_count = token_counter.count_tokens(embed_text)
            
            # Create memory record
            rec = MemoryRecord(
                session_id=raw.session_id,
                source_event_id=f"{raw.event_id}_part{idx}",  # Unique ID for each part
                who=who,
                what=what,
                when=raw.timestamp,
                where=where,
                why=why,
                how=how,
                raw_text=raw.content if idx == 0 else f"[Part {idx+1}] {what}",  # Include full content only in first
                token_count=token_count,
                embed_text=embed_text,
                embed_model=cfg.embed_model_name
            )
            rec.extra['embed_vector_np'] = vec.astype('float32').tolist()
            rec.extra['part_index'] = idx
            rec.extra['total_parts'] = len(parsed_list)
            
            memories.append(rec)
            
        except Exception as e:
            print(f"Failed to create memory {idx}: {e}")
            continue
    
    return memories if memories else []
