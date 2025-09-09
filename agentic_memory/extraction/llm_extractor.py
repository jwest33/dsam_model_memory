from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime
import json
from ..types import RawEvent, MemoryRecord, Who, Where
from ..tokenization import TokenizerAdapter
from ..config import cfg
import numpy as np

# Use llama.cpp embeddings
from ..embedding import get_llama_embedder

# Initialize embedder once at module level to avoid reloading
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_llama_embedder()
    return _embedder

PROMPT = """You are a structured-information extractor that converts an interaction into 5W1H fields.
Return ONLY valid JSON in the following schema:

{{
  "who": {{ "type": "user|llm|tool|system", "id": "<string>", "label": "<optional string>" }},
  "what": "<concise description of the key action or content>",
  "when": "<ISO 8601 timestamp>",
  "where": {{ "type": "<context type e.g. physical, digital, financial, academic, conceptual, social>", "value": "<specific context like UI path, URL, file, location, or domain>" }},
  "why": "<best-effort intent or reason>",
  "how": "<method used, tool/procedure/parameters>"
}}

Consider the content and metadata; be concise but unambiguous.
"""

def _call_llm(prompt: str, content: str) -> Optional[Dict[str, Any]]:
    # Local llama.cpp in OpenAI-compatible mode
    import requests
    url = f"{cfg.llm_base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"]
        # Try to parse JSON
        start = out.find('{')
        end = out.rfind('}')
        if start >= 0 and end > start:
            obj = json.loads(out[start:end+1])
            return obj
    except Exception:
        return None
    return None

def extract_5w1h(raw: RawEvent, context_hint: str = "") -> MemoryRecord:
    # Try LLM-based extraction first; fall back to rules
    content = f"EventType: {raw.event_type}\nActor: {raw.actor}\nTimestamp: {raw.timestamp.isoformat()}\nContent: {raw.content}\nMetadata: {raw.metadata}\nContext: {context_hint}"
    parsed = _call_llm(PROMPT, content)
    if not parsed:
        # simple fallback
        who_type = 'tool' if raw.event_type in ('tool_call','tool_result') else ('user' if raw.event_type=='user_message' else ('llm' if raw.event_type=='llm_message' else 'system'))
        who = Who(type=who_type, id=raw.actor, label=None)
        what = raw.metadata.get('operation') or raw.content[:160]
        where = Where(type='digital', value=raw.metadata.get('location') or raw.metadata.get('tool_name') or 'local_ui')
        why = raw.metadata.get('intent') or 'unspecified'
        how = raw.metadata.get('method') or raw.metadata.get('tool_name') or 'message'
    else:
        who = Who(**parsed['who'])
        what = parsed.get('what','').strip() or raw.content[:160]
        w = parsed.get('where', {'type':'digital','value':'local_ui'})
        where = Where(type=w.get('type','digital'), value=w.get('value','local_ui'))
        why = parsed.get('why','unspecified')
        how = parsed.get('how','message')

    embedder = _get_embedder()
    embed_text = f"WHAT: {what}\nWHY: {why}\nHOW: {how}\nRAW: {raw.content}"
    vec = embedder.encode([embed_text], normalize_embeddings=True)[0]
    token_counter = TokenizerAdapter().count_tokens(embed_text)

    rec = MemoryRecord(
        session_id=raw.session_id,
        source_event_id=raw.event_id,
        who=who,
        what=what,
        when=raw.timestamp,
        where=where,
        why=why,
        how=how,
        raw_text=raw.content,
        token_count=token_counter,
        embed_text=embed_text,
        embed_model=cfg.embed_model_name
    )
    # Return both record and vector as bytes from caller; to avoid tight coupling we return rec only
    rec.extra['embed_vector_np'] = vec.astype('float32').tolist()
    return rec
