from __future__ import annotations
import os
from flask import Flask, render_template, request, jsonify, session
from ..config import cfg
from ..config_manager import ConfigManager, ConfigType
from ..storage.sql_store import MemoryStore
from ..storage.faiss_index import FaissIndex
from ..router import MemoryRouter
from ..types import RawEvent
from ..tools.tool_handler import ToolHandler
from ..tools.memory_evaluator import MemoryEvaluator

import requests
from datetime import datetime
import json

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)  # For session management

# Initialize config manager
config_manager = ConfigManager(cfg.db_path)

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
    response_json = r.json()
    content = response_json["choices"][0]["message"]["content"]
    
    return content, response_json

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

    # Check if user message likely needs a tool
    suggested_tool = tool_handler.should_use_tool(user_text)
    
    # Construct LLM messages with tool support
    if suggested_tool:
        # Add a more forceful example when weather is mentioned
        if "weather" in user_text.lower():
            example_query = user_text.replace("'", "").replace('"', '')
            base_prompt = f"""You MUST search for current weather information.
Output this EXACT format:
<tool_call>{{"name": "web_search", "arguments": {{"query": "{example_query}"}}}}</tool_call>"""
        else:
            base_prompt = f"""You are a helpful assistant. The user is asking about current/real-time information.
YOU MUST use the {suggested_tool} tool. Do not claim you lack access to current information.
Output the tool call immediately without explanation."""
    else:
        base_prompt = """You are a helpful assistant with memory capabilities.
You may reference the [MEM:<id>] annotations to ground your reply."""
    
    sys_prompt = tool_handler.build_system_prompt_with_tools(base_prompt)
    
    llm_messages = [{"role":"system","content":sys_prompt}]
    if mem_texts:
        llm_messages.append({"role":"system","content":"\n\n".join(mem_texts)})

    llm_messages += messages
    
    # Simple evaluation based on query type - let the LLM decide based on actual retrieved memories
    query_type = MemoryEvaluator.classify_query_type(user_text)
    
    # For time-sensitive queries, suggest using tools
    time_sensitive_types = [
        'weather_current', 'weather_today', 'weather_forecast',
        'stock_price', 'cryptocurrency', 'news_breaking', 
        'news_daily', 'sports_score', 'traffic'
    ]
    
    should_suggest_tool = query_type in time_sensitive_types
    
    # Create a simple evaluation result
    evaluation = {
        'needs_tool': should_suggest_tool,
        'query_type': query_type,
        'reason': f"Query type '{query_type}' typically benefits from fresh data" if should_suggest_tool else f"Query type '{query_type}' can use existing memories",
        'valid_memories': [],
        'invalid_memories': []
    }
    
    print(f"\n=== Memory Evaluation ===")
    print(f"Query Type: {evaluation['query_type']}")
    print(f"Valid Memories: {len(evaluation['valid_memories'])}")
    print(f"Invalid/Outdated: {len(evaluation['invalid_memories'])}")
    print(f"Needs Tool: {evaluation['needs_tool']}")
    print(f"Reason: {evaluation['reason']}")
    print(f"========================\n")
    
    # Add evaluation context to help LLM decide
    if evaluation['needs_tool'] or (evaluation['invalid_memories'] and not evaluation['valid_memories']):
        # Build a prompt that helps the LLM understand it should use tools
        evaluation_prompt = MemoryEvaluator.build_evaluation_prompt(user_text, evaluation)
        
        # Add the evaluation as a system message
        llm_messages.append({"role": "system", "content": evaluation_prompt})
        
        # Let the LLM construct its own search query naturally
        enhanced_prompt = f"""{user_text}

[CONTEXT: The memory evaluation indicates that {evaluation['reason']}]

Please use the web_search tool to find current information. Construct an appropriate search query based on what the user is asking for.

Format your tool call as:
<tool_call>{{"name": "web_search", "arguments": {{"query": "your search query"}}}}</tool_call>"""
        llm_messages.append({"role":"user","content":enhanced_prompt})
    else:
        # Memories are sufficient, proceed normally
        llm_messages.append({"role":"user","content":user_text})

    # Initial LLM response
    reply, raw_json = llama_chat(llm_messages, tools_enabled=True)
    
    # Debug logging
    print(f"\n=== LLM Response Debug ===")
    print(f"Raw LLM reply (first 500 chars): {reply[:500]}")
    print(f"===========================\n")
    
    # Check for tool calls
    cleaned_reply, tool_calls = tool_handler.parse_tool_calls(reply)
    
    # Log for debugging
    if tool_calls:
        print(f"Detected {len(tool_calls)} tool calls from LLM response")
        for tc in tool_calls:
            print(f"  - {tc.name}: {tc.arguments}")
    elif tool_handler.should_use_tool(user_text):
        # User likely wants a tool but LLM didn't use it
        print(f"User message suggests tool use but LLM didn't call any tools")
    
    if tool_calls:
        # Execute tools and store as memories
        tool_results = []
        print(f"Executing {len(tool_calls)} tool calls...")
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
            print(f"Tool {tool_call.name} executed: success={result.success}")
            if result.success:
                print(f"Tool result preview (first 500 chars): {result.content[:500]}")
            else:
                print(f"Tool error: {result.error}")
            
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
        
        # Add tool results as assistant message to maintain conversation flow
        llm_messages.append({"role": "assistant", "content": tool_message})
        
        # Add explicit instruction to use the tool results
        # Provide clear guidance on using the search results
        follow_up = f"""The web search has been completed. Here are the search results above.

Based on these search results, please provide the best answer you can to the user's original question: "{user_text}"

Important instructions:
1. If the search results contain relevant information, use it to answer the question
2. If the results are not directly about the topic but mention related information, extract what's useful
3. If the results don't contain the specific information needed, acknowledge this and suggest what the user should do next
4. Be specific about what information you found or didn't find

Please provide your response now:"""
        
        llm_messages.append({"role": "user", "content": follow_up})
        
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
    rows = con.execute("""SELECT 
        memory_id, session_id, source_event_id,
        who_type, who_id, who_label, 
        what, when_ts, 
        where_type, where_value, where_lat, where_lon,
        why, how, 
        raw_text, token_count, embed_model, 
        extra_json, created_at 
        FROM memories 
        ORDER BY when_ts DESC 
        LIMIT 200""").fetchall()
    con.close()
    return render_template('memories.html', rows=rows)


@app.route('/settings')
def settings_page():
    """Settings page"""
    # Show all categories since advanced is on by default
    categories = config_manager.get_categories()
    return render_template('settings.html', categories=categories)

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    """Get all settings or settings by category"""
    category = request.args.get('category')
    include_advanced = request.args.get('advanced', 'false').lower() == 'true'
    
    if category and category != 'all':
        settings = config_manager.get_settings_by_category(category, include_advanced)
    else:
        settings = list(config_manager.settings.values())
        if not include_advanced:
            settings = [s for s in settings if not s.advanced]
    
    return jsonify({
        'settings': [s.to_dict() for s in settings]
    })

@app.route('/api/settings/<key>', methods=['GET', 'PUT'])
def api_setting(key):
    """Get or update a specific setting"""
    if request.method == 'GET':
        setting = config_manager.get_setting(key)
        if not setting:
            return jsonify({'error': f'Unknown setting: {key}'}), 404
        return jsonify(setting.to_dict())
    
    else:  # PUT
        data = request.get_json()
        value = data.get('value')
        user = session.get('user', 'web_ui')
        reason = data.get('reason')
        
        success, error = config_manager.update_setting(key, value, user, reason)
        
        if success:
            setting = config_manager.get_setting(key)
            return jsonify({
                'success': True,
                'setting': setting.to_dict(),
                'requires_restart': setting.requires_restart
            })
        else:
            return jsonify({'success': False, 'error': error}), 400

@app.route('/api/settings/<key>/reset', methods=['POST'])
def api_reset_setting(key):
    """Reset a setting to its default value"""
    user = session.get('user', 'web_ui')
    success, error = config_manager.reset_setting(key, user)
    
    if success:
        setting = config_manager.get_setting(key)
        return jsonify({
            'success': True,
            'setting': setting.to_dict()
        })
    else:
        return jsonify({'success': False, 'error': error}), 400

@app.route('/api/settings/reset-all', methods=['POST'])
def api_reset_all_settings():
    """Reset all settings to defaults"""
    user = session.get('user', 'web_ui')
    results = config_manager.reset_all(user)
    
    successful = sum(1 for r in results.values() if r[0])
    failed = sum(1 for r in results.values() if not r[0])
    
    return jsonify({
        'success': failed == 0,
        'successful': successful,
        'failed': failed,
        'results': results
    })

@app.route('/api/settings/export', methods=['GET'])
def api_export_settings():
    """Export current configuration"""
    return jsonify(config_manager.export_config())

@app.route('/api/settings/import', methods=['POST'])
def api_import_settings():
    """Import configuration"""
    config_data = request.get_json()
    user = session.get('user', 'web_ui')
    
    results = config_manager.import_config(config_data, user)
    
    if 'error' in results:
        return jsonify({'success': False, 'error': results['error'][1]}), 400
    
    successful = sum(1 for r in results.values() if r[0])
    
    return jsonify({
        'success': True,
        'updated': successful,
        'results': results
    })

@app.route('/api/settings/history', methods=['GET'])
def api_settings_history():
    """Get configuration change history"""
    key = request.args.get('key')
    limit = int(request.args.get('limit', 50))
    
    history = config_manager.get_history(key, limit)
    
    return jsonify({'history': history})

@app.route('/api/settings/validate-weights', methods=['GET'])
def api_validate_weights():
    """Validate that retrieval weights sum to 1.0"""
    valid, error = config_manager.validate_weights()
    
    return jsonify({
        'valid': valid,
        'error': error
    })

@app.route('/api/memories/<memory_id>', methods=['DELETE'])
def api_delete_memory(memory_id):
    """Delete a single memory"""
    try:
        import sqlite3
        con = sqlite3.connect(cfg.db_path)
        cursor = con.cursor()
        
        # Delete from all related tables
        cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        deleted_count = cursor.rowcount
        
        # Also delete from FTS and other tables (they should cascade but let's be explicit)
        cursor.execute("DELETE FROM mem_fts WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM usage_stats WHERE memory_id = ?", (memory_id,))
        
        con.commit()
        con.close()
        
        # Also remove from FAISS index if it exists
        try:
            index.remove(memory_id)
            index.save()
        except:
            pass  # Memory might not be in index
        
        return jsonify({
            'success': True,
            'deleted': deleted_count > 0,
            'memory_id': memory_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/memories/delete-all', methods=['DELETE'])
def api_delete_all_memories():
    """Delete all memories from the database"""
    try:
        import sqlite3
        con = sqlite3.connect(cfg.db_path)
        cursor = con.cursor()
        
        # Count memories before deletion
        count_result = cursor.execute("SELECT COUNT(*) FROM memories").fetchone()
        count = count_result[0] if count_result else 0
        
        # Delete from all tables (skip if table doesn't exist)
        tables_to_clear = [
            'memories', 'mem_fts', 'embeddings', 'usage_stats',
            'memory_synapses', 'memory_importance', 'embedding_drift',
            'blocks', 'block_members', 'cluster_membership'
        ]
        
        for table in tables_to_clear:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                # Table doesn't exist, skip it
                pass
        
        con.commit()
        con.close()
        
        # Clear FAISS index
        try:
            index.reset()
            index.save()
        except:
            pass
        
        return jsonify({
            'success': True,
            'deleted_count': count,
            'message': f'Successfully deleted {count} memories'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
