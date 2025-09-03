from __future__ import annotations
import os
from flask import Flask, render_template, request, jsonify
from ..config import cfg
from ..storage.sql_store import MemoryStore
from ..storage.faiss_index import FaissIndex
from ..router import MemoryRouter
from ..types import RawEvent
from ..tools.tool_handler import ToolHandler

import requests
from datetime import datetime
import json

app = Flask(__name__, template_folder='templates', static_folder='static')
store = MemoryStore(cfg.db_path)
# Assume dim from embed model default 384 for all-MiniLM-L6-v2; can be inferred dynamically
index = FaissIndex(dim=384, index_path=cfg.index_path)
router = MemoryRouter(store, index)
tool_handler = ToolHandler()

def llama_chat(messages, tools_enabled=False):
    url = f"{cfg.llm_base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": 0.2,
        "stream": False
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], r.json()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    session_id = data.get('session_id', 'default')
    user_text = data.get('text','')
    messages = data.get('messages', [])
    # Ingest user event
    raw = RawEvent(session_id=session_id, event_type='user_message', actor='user:local', content=user_text, metadata={'location':'flask_ui'})
    router.ingest(raw)

    # Retrieve memory block for context packing
    block = router.retrieve_block(session_id=session_id, context_messages=messages + [{"role":"user","content":user_text}], actor_hint=None, spatial_hint='flask_ui')

    mem_texts = []
    if block and block.get('members'):
        rows = store.fetch_memories(block['members'])
        for r in rows:
            mem_texts.append(f"[MEM:{r['memory_id']}] WHO={r['who_type']}:{r['who_id']} WHEN={r['when_ts']} WHERE={r['where_value']}\nWHAT={r['what']}\nWHY={r['why']}\nHOW={r['how']}\nRAW={r['raw_text']}\n")
        # record usage
        store.record_access(block['members'])

    # Construct LLM messages with tool support
    base_prompt = "You are a helpful assistant. You may reference the [MEM:<id>] annotations to ground your reply. Ask for 'memory.fetch_next' if a pointer indicates more."
    sys_prompt = tool_handler.build_system_prompt_with_tools(base_prompt)
    
    llm_messages = [{"role":"system","content":sys_prompt}]
    if mem_texts:
        llm_messages.append({"role":"system","content":"\n\n".join(mem_texts)})

    llm_messages += messages
    llm_messages.append({"role":"user","content":user_text})

    # Initial LLM response
    reply, raw_json = llama_chat(llm_messages, tools_enabled=True)
    
    # Check for tool calls
    cleaned_reply, tool_calls = tool_handler.parse_tool_calls(reply)
    
    if tool_calls:
        # Execute tools and store as memories
        tool_results = []
        for tool_call in tool_calls:
            # Store tool call as memory
            tool_call_event = RawEvent(
                session_id=session_id,
                event_type='tool_call',
                actor=f'tool:{tool_call.name}',
                content=json.dumps({"name": tool_call.name, "arguments": tool_call.arguments}),
                metadata={'location': 'flask_ui', 'tool_type': 'request'}
            )
            router.ingest(tool_call_event)
            
            # Execute tool
            result = tool_handler.execute_tool(tool_call)
            tool_results.append(result)
            
            # Store tool result as memory
            tool_result_event = RawEvent(
                session_id=session_id,
                event_type='tool_result',
                actor=f'tool:{tool_call.name}',
                content=result.content if result.success else f"Error: {result.error}",
                metadata={'location': 'flask_ui', 'tool_type': 'response', 'success': result.success}
            )
            router.ingest(tool_result_event)
        
        # Add tool results to context and get final response
        tool_message = tool_handler.format_tool_message(tool_results)
        llm_messages.append({"role": "system", "content": tool_message})
        
        # Get final LLM response after tool execution
        final_reply, _ = llama_chat(llm_messages, tools_enabled=False)
        
        # Combine cleaned initial response with final response
        full_reply = cleaned_reply
        if cleaned_reply and final_reply:
            full_reply += "\n\n" + final_reply
        elif final_reply:
            full_reply = final_reply
    else:
        full_reply = reply

    # Ingest LLM reply
    raw_llm = RawEvent(
        session_id=session_id,
        event_type='llm_message',
        actor=f"llm:{cfg.llm_model}",
        content=full_reply,
        metadata={'location':'flask_ui', 'had_tool_calls': len(tool_calls) > 0}
    )
    router.ingest(raw_llm)

    return jsonify({"reply": full_reply, "block": block, "tool_calls": len(tool_calls)})

@app.route('/memories', methods=['GET'])
def list_memories():
    # naive preview
    import sqlite3
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT memory_id, when_ts, who_type, who_id, what FROM memories ORDER BY when_ts DESC LIMIT 200").fetchall()
    con.close()
    return render_template('memories.html', rows=rows)


@app.route('/api/memory/next', methods=['GET'])
def api_memory_next():
    block_id = request.args.get('block_id')
    if not block_id:
        return jsonify({'error':'missing block_id'}), 400
    cur = store.get_block(block_id)
    if not cur:
        return jsonify({'error':'block not found'}), 404
    nxt_id = cur['block'].get('next_block_id') if isinstance(cur['block'], dict) else None
    if not nxt_id:
        return jsonify({'block':cur, 'next':None})
    nxt = store.get_block(nxt_id)
    return jsonify({'block':cur, 'next':nxt})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
