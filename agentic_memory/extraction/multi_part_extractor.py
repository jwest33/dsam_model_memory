"""
Multi-part memory extractor that breaks complex information into separate 5W1H memories.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ..types import RawEvent, MemoryRecord, Who, Where
from ..tokenization import TokenizerAdapter
from ..config import cfg

# Use llama.cpp embeddings
from ..embedding import get_llama_embedder

# Initialize embedder once at module level to avoid reloading
_embedder = None
_embedder_lock = None

def _get_embedder():
    global _embedder, _embedder_lock
    if _embedder is None:
        if _embedder_lock is None:
            import threading
            _embedder_lock = threading.Lock()
        with _embedder_lock:
            if _embedder is None:
                _embedder = get_llama_embedder()
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
    
    # Batch process embeddings for better performance
    embed_texts = []
    for idx, parsed in enumerate(parsed_list):
        what = parsed.get('what', '').strip() or f"Part {idx+1} of {raw.content[:100]}"
        why = parsed.get('why', 'unspecified')
        how = parsed.get('how', 'message')
        embed_texts.append(f"WHAT: {what}\nWHY: {why}\nHOW: {how}")
    
    # Batch encode all embeddings at once
    if embed_texts:
        embeddings = embedder.encode(embed_texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    else:
        embeddings = []
    
    memories = []
    for idx, (parsed, embed_text, vec) in enumerate(zip(parsed_list, embed_texts, embeddings)):
        try:
            # Extract fields
            who = Who(**parsed['who'])
            what = parsed.get('what', '').strip() or f"Part {idx+1} of {raw.content[:100]}"
            w = parsed.get('where', {'type': 'digital', 'value': 'local_ui'})
            where = Where(type=w.get('type', 'digital'), value=w.get('value', 'local_ui'))
            why = parsed.get('why', 'unspecified')
            how = parsed.get('how', 'message')
            
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


def extract_batch_5w1h(raw_events: List[RawEvent], context_hints: Optional[List[str]] = None) -> List[List[MemoryRecord]]:
    """
    Batch extract memories from multiple events for better performance.
    
    Args:
        raw_events: List of raw events to process
        context_hints: Optional list of context hints (one per event)
        
    Returns:
        List of memory lists (one list per raw event)
    """
    if not context_hints:
        context_hints = [''] * len(raw_events)
    
    results = []
    embedder = _get_embedder()
    token_counter = TokenizerAdapter()
    
    # Collect all parsed data first
    all_parsed_data = []
    for raw, hint in zip(raw_events, context_hints):
        content = f"EventType: {raw.event_type}\nActor: {raw.actor}\nTimestamp: {raw.timestamp.isoformat()}\nContent: {raw.content}\nMetadata: {raw.metadata}\nContext: {hint}"
        parsed_list = _call_llm_multi(content)
        
        if not parsed_list:
            # Fallback
            who_type = 'tool' if raw.event_type in ('tool_call','tool_result') else ('user' if raw.event_type=='user_message' else ('llm' if raw.event_type=='llm_message' else 'system'))
            parsed_list = [{
                'who': {'type': who_type, 'id': raw.actor, 'label': None},
                'what': raw.metadata.get('operation') or raw.content[:160],
                'where': {'type': 'digital', 'value': raw.metadata.get('location') or 'local_ui'},
                'why': raw.metadata.get('intent') or 'unspecified',
                'how': raw.metadata.get('method') or 'message'
            }]
        
        all_parsed_data.append((raw, parsed_list))
    
    # Collect all embed texts for batch processing
    all_embed_texts = []
    embed_text_mapping = []  # Track which texts belong to which event/memory
    
    for event_idx, (raw, parsed_list) in enumerate(all_parsed_data):
        for memory_idx, parsed in enumerate(parsed_list):
            what = parsed.get('what', '').strip() or f"Part {memory_idx+1} of {raw.content[:100]}"
            why = parsed.get('why', 'unspecified')
            how = parsed.get('how', 'message')
            embed_text = f"WHAT: {what}\nWHY: {why}\nHOW: {how}"
            all_embed_texts.append(embed_text)
            embed_text_mapping.append((event_idx, memory_idx))
    
    # Batch encode all embeddings at once
    if all_embed_texts:
        all_embeddings = embedder.encode(all_embed_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    else:
        all_embeddings = []
    
    # Build memory records using batch embeddings
    embedding_idx = 0
    for event_idx, (raw, parsed_list) in enumerate(all_parsed_data):
        memories = []
        for memory_idx, parsed in enumerate(parsed_list):
            try:
                # Get the corresponding embedding
                embed_text = all_embed_texts[embedding_idx]
                vec = all_embeddings[embedding_idx]
                embedding_idx += 1
                
                # Extract fields
                who = Who(**parsed['who'])
                what = parsed.get('what', '').strip() or f"Part {memory_idx+1} of {raw.content[:100]}"
                w = parsed.get('where', {'type': 'digital', 'value': 'local_ui'})
                where = Where(type=w.get('type', 'digital'), value=w.get('value', 'local_ui'))
                why = parsed.get('why', 'unspecified')
                how = parsed.get('how', 'message')
                
                token_count = token_counter.count_tokens(embed_text)
                
                # Create memory record
                rec = MemoryRecord(
                    session_id=raw.session_id,
                    source_event_id=f"{raw.event_id}_part{memory_idx}",
                    who=who,
                    what=what,
                    when=raw.timestamp,
                    where=where,
                    why=why,
                    how=how,
                    raw_text=raw.content if memory_idx == 0 else f"[Part {memory_idx+1}] {what}",
                    token_count=token_count,
                    embed_text=embed_text,
                    embed_model=cfg.embed_model_name
                )
                rec.extra['embed_vector_np'] = vec.astype('float32').tolist()
                rec.extra['part_index'] = memory_idx
                rec.extra['total_parts'] = len(parsed_list)
                rec.extra['batch_processed'] = True
                
                memories.append(rec)
                
            except Exception as e:
                print(f"Failed to create memory for event {event_idx}, memory {memory_idx}: {e}")
                continue
        
        results.append(memories)
    
    return results
