from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import re
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

def _extract_entities_fallback(text: str) -> List[str]:
    """Extract entities from text as a fallback when LLM fails.
    
    This is a simple rule-based extraction for common patterns.
    """
    entities = []
    
    # Clean the text
    text = text.strip()
    if not text:
        return []
    
    # Extract capitalized words (likely proper nouns)
    # But skip common words that are often capitalized
    common_words = {'The', 'This', 'That', 'What', 'When', 'Where', 'Who', 'How', 'Why',
                   'If', 'Then', 'And', 'But', 'Or', 'In', 'On', 'At', 'To', 'From',
                   'Is', 'Are', 'Was', 'Were', 'Can', 'Could', 'Would', 'Should'}
    
    # Find sequences of capitalized words (e.g., "Growth Hormone")
    cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    for match in re.finditer(cap_pattern, text):
        entity = match.group()
        if entity not in common_words and len(entity) > 2:
            entities.append(entity)
    
    # Extract acronyms (e.g., GH, IGF-1, API)
    acronym_pattern = r'\b[A-Z]{2,}(?:-\d+)?\b'
    for match in re.finditer(acronym_pattern, text):
        entities.append(match.group())
    
    # Extract technical terms with numbers (e.g., Python3, IPv4)
    tech_pattern = r'\b[A-Za-z]+\d+[A-Za-z0-9]*\b'
    for match in re.finditer(tech_pattern, text):
        entities.append(match.group())
    
    # Extract quoted strings (often important terms)
    quote_pattern = r'["\']([^"\']+)["\']'
    for match in re.finditer(quote_pattern, text):
        quoted = match.group(1)
        if len(quoted) < 50:  # Don't include long quotes
            entities.append(quoted)
    
    # Extract common technical/scientific terms
    tech_keywords = ['gene', 'protein', 'enzyme', 'hormone', 'receptor', 'molecule',
                     'Python', 'JavaScript', 'Docker', 'Redis', 'API', 'database',
                     'script', 'function', 'class', 'method', 'variable',
                     'player', 'team', 'trade', 'contract', 'game']
    
    text_lower = text.lower()
    for keyword in tech_keywords:
        if keyword.lower() in text_lower:
            entities.append(keyword)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity.lower() not in seen:
            seen.add(entity.lower())
            unique_entities.append(entity)
    
    return unique_entities[:20]  # Limit to 20 entities

PROMPT = """You are a structured-information extractor that converts an interaction into 5W1H fields.
Return ONLY valid JSON in the following schema:

{{
  "who": {{ "type": "<actor type e.g. user, llm, tool, system, team, group, organization>", "id": "<string identifier>", "label": "<optional descriptive label>" }},
  "who_list": ["<person1>", "<person2>", "<organization>", ...],
  "what": ["<entity1>", "<entity2>", ...],
  "when": "<ISO 8601 timestamp>",
  "when_list": ["<time_expression1>", "<date_reference>", "<temporal_phrase>", ...],
  "where": {{ "type": "<context type e.g. physical, digital, financial, academic, conceptual, social>", "value": "<specific context like UI path, URL, file, location, or domain>" }},
  "where_list": ["<location1>", "<place2>", "<context>", ...],
  "why": "<best-effort intent or reason - IMPORTANT: if the user is asking to recall memories, searching for information, or asking 'what do you remember about X' or 'is there any memory about Y', set this to 'memory_recall: <topic>' where <topic> is what they're trying to recall>",
  "how": "<method used, tool/procedure/parameters>"
}}

CRITICAL: All list fields (who_list, what, when_list, where_list) must be JSON arrays containing relevant items.

Extract for WHO_LIST: People, organizations, teams, roles, departments mentioned
- "The CEO told the marketing team" → ["CEO", "marketing team"]
- "Alice and Bob from engineering" → ["Alice", "Bob", "engineering"]
- "OpenAI's GPT-4" → ["OpenAI", "GPT-4"]

Extract for WHAT: Key entities, concepts, and topics
- Names of people, organizations, teams, products
- Technical terms, genes, proteins, chemicals
- Programming languages, frameworks, tools
- Concepts, theories, methodologies
- Specific objects, places, or things
- Numbers, dates, measurements when significant

Extract for WHEN_LIST: Time expressions and temporal references
- "yesterday at 3pm during the meeting" → ["yesterday", "3pm", "during the meeting"]
- "last week's sprint" → ["last week", "sprint"]
- "Q3 2024 planning" → ["Q3 2024", "planning period"]

Extract for WHERE_LIST: Locations, places, and contexts
- "in the conference room at headquarters" → ["conference room", "headquarters"]
- "on GitHub in the main repository" → ["GitHub", "main repository"]
- "Seattle office's lab" → ["Seattle office", "lab"]

Examples:
- "asked which genes encode Growth hormone (GH) and insulin-like growth factor 1" → what: ["genes", "Growth hormone", "GH", "insulin-like growth factor 1", "IGF-1", "encoding"]
- "Python script for data analysis" → what: ["Python", "script", "data analysis"]
- "Player X was traded from Team A to Team B" → who_list: ["Player X"], what: ["trade", "sports transaction"], where_list: ["Team A", "Team B"]

Consider the content and metadata; be concise but unambiguous.
Special instructions for the 'why' field:
- If user asks "do you remember...", "what do you know about...", "recall memories about...", "find memories of...", "is there any memory about...", set why to "memory_recall: <topic>"
- If user asks "what did we discuss about...", "what was said about...", set why to "memory_recall: <topic>"
- If user asks for past information, history, or previous discussions, set why to "memory_recall: <topic>"
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
    # Initialize list fields
    who_list = None
    when_list = None
    where_list = None
    
    if not parsed:
        # simple fallback with recall detection
        who_type = 'tool' if raw.event_type in ('tool_call','tool_result') else ('user' if raw.event_type=='user_message' else ('llm' if raw.event_type=='llm_message' else 'system'))
        who = Who(type=who_type, id=raw.actor, label=None)
        
        # Extract entities from content as fallback
        what_entities = _extract_entities_fallback(raw.content)
        what = json.dumps(what_entities) if what_entities else raw.content[:160]
        
        where = Where(type='digital', value=raw.metadata.get('location') or raw.metadata.get('tool_name') or 'local_ui')
        
        # Detect recall queries in fallback
        content_lower = raw.content.lower()
        recall_keywords = ['remember', 'recall', 'memory', 'memories', 'what do you know', 
                          'what did we discuss', 'what was said', 'find information',
                          'is there any memory', 'do you have any memory']
        
        why = raw.metadata.get('intent') or 'unspecified'
        for keyword in recall_keywords:
            if keyword in content_lower:
                # Extract topic from the content
                topic = raw.content[content_lower.index(keyword)+len(keyword):].strip()[:50]
                why = f"memory_recall: {topic}"
                break
        
        how = raw.metadata.get('method') or raw.metadata.get('tool_name') or 'message'
    else:
        who = Who(**parsed['who'])
        
        # Handle list fields
        who_list_raw = parsed.get('who_list', [])
        if isinstance(who_list_raw, list) and who_list_raw:
            who_list = json.dumps(who_list_raw)
        
        # Handle 'what' as array of entities
        what_raw = parsed.get('what', [])
        if isinstance(what_raw, list):
            # Convert list to JSON string for storage
            what = json.dumps(what_raw) if what_raw else '[]'
        else:
            # Fallback if LLM returns string instead of array
            what = str(what_raw).strip() or raw.content[:160]
        
        when_list_raw = parsed.get('when_list', [])
        if isinstance(when_list_raw, list) and when_list_raw:
            when_list = json.dumps(when_list_raw)
        
        w = parsed.get('where', {'type':'digital','value':'local_ui'})
        where = Where(type=w.get('type','digital'), value=w.get('value','local_ui'))
        
        where_list_raw = parsed.get('where_list', [])
        if isinstance(where_list_raw, list) and where_list_raw:
            where_list = json.dumps(where_list_raw)
        
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
        who_list=who_list,
        what=what,
        when=raw.timestamp,
        when_list=when_list,
        where=where,
        where_list=where_list,
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
