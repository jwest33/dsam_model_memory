from __future__ import annotations
import os
from flask import Flask, render_template, request, jsonify, session, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from ..config import cfg, get_config_manager, get_upload_config, get_generation_defaults
from ..config_manager import ConfigType
from ..storage.sql_store import MemoryStore
from ..storage.faiss_index import FaissIndex
from ..router import MemoryRouter
from ..types import RawEvent
from ..tools.tool_handler import ToolHandler
from ..tools.memory_evaluator import MemoryEvaluator
from ..import_export import MemoryExporter, MemoryImporter
from ..cluster.concept_cluster import LiquidMemoryClusters
from ..document_parser import DocumentParser, SemanticChunker, ParagraphChunker, SentenceChunker

import requests
from datetime import datetime
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sqlite3

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)  # For session management

# Make range available in templates
app.jinja_env.globals.update(range=range)

# Initialize config manager
config_manager = get_config_manager()

# Configure file upload from config
upload_cfg = get_upload_config()
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = upload_cfg['allowed_extensions']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = upload_cfg['max_size']

store = MemoryStore(cfg.db_path)
# Get embedding dimension from config (1024 for Qwen3-Embedding)
embed_dim = int(os.getenv('AM_EMBEDDING_DIM', '1024'))
faiss_index = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
router = MemoryRouter(store, faiss_index)
tool_handler = ToolHandler()
liquid_clusters = LiquidMemoryClusters(n_clusters=16, dim=embed_dim)

def llama_chat(messages, tools_enabled=False):
    url = f"{cfg.llm_base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    gen_defaults = get_generation_defaults()
    body = {
        "model": cfg.llm_model,
        "messages": messages,
        "temperature": gen_defaults['temperature'],
        "stream": False
    }
    
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        print(f"LLM API Error: Status {r.status_code}")
        print(f"Request body: {body}")
        print(f"Response: {r.text}")
    r.raise_for_status()
    response_json = r.json()
    
    # Debug response structure
    print(f"Response keys: {response_json.keys()}")
    if "choices" in response_json and response_json["choices"]:
        print(f"First choice keys: {response_json['choices'][0].keys()}")
        
    content = response_json["choices"][0]["message"]["content"]
    print(f"Extracted content length: {len(content)}")
    
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
        # Debug: show block info
        print(f"Block info: budget={block.get('budget_tokens', 0)}, used={block.get('used_tokens', 0)}, members={len(block['members'])}")
        
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
        # The BlockBuilder already handled token budgeting, just use what it selected
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
    print(f"Raw LLM reply (first 500 chars): {reply[:500] if reply else 'No reply'}")
    print(f"Full reply length: {len(reply) if reply else 0}")
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
    # Get query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    sort_by = request.args.get('sort_by', 'when_ts')
    sort_order = request.args.get('sort_order', 'desc')
    
    # Filter parameters
    session_filter = request.args.get('session_id', '')
    who_filter = request.args.get('who', '')
    what_filter = request.args.get('what', '')
    where_filter = request.args.get('where', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    # Validate sort column
    valid_sort_columns = ['when_ts', 'created_at', 'memory_id', 'session_id', 
                          'who_id', 'what', 'where_value', 'token_count']
    if sort_by not in valid_sort_columns:
        sort_by = 'when_ts'
    
    # Validate sort order
    if sort_order.lower() not in ['asc', 'desc']:
        sort_order = 'desc'
    
    import sqlite3
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row
    
    # Build query with filters
    where_clauses = []
    params = []
    
    if session_filter:
        where_clauses.append("session_id LIKE ?")
        params.append(f"%{session_filter}%")
    
    if who_filter:
        where_clauses.append("(who_id LIKE ? OR who_label LIKE ? OR who_type LIKE ?)")
        params.extend([f"%{who_filter}%", f"%{who_filter}%", f"%{who_filter}%"])
    
    if what_filter:
        where_clauses.append("what LIKE ?")
        params.append(f"%{what_filter}%")
    
    if where_filter:
        where_clauses.append("(where_value LIKE ? OR where_type LIKE ?)")
        params.extend([f"%{where_filter}%", f"%{where_filter}%"])
    
    if date_from:
        where_clauses.append("when_ts >= ?")
        params.append(date_from)
    
    if date_to:
        where_clauses.append("when_ts <= ?")
        params.append(date_to)
    
    where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Get total count for pagination
    count_query = f"SELECT COUNT(*) as total FROM memories{where_clause}"
    total_count = con.execute(count_query, params).fetchone()['total']
    
    # Calculate pagination
    total_pages = (total_count + per_page - 1) // per_page
    offset = (page - 1) * per_page
    
    # Build main query
    query = f"""SELECT 
        memory_id, session_id, source_event_id,
        who_type, who_id, who_label, 
        what, when_ts, 
        where_type, where_value, where_lat, where_lon,
        why, how, 
        raw_text, token_count, embed_model, 
        extra_json, created_at 
        FROM memories{where_clause}
        ORDER BY {sort_by} {sort_order}
        LIMIT ? OFFSET ?"""
    
    params.extend([per_page, offset])
    rows = con.execute(query, params).fetchall()
    con.close()
    
    # Convert Row objects to dictionaries for JSON serialization
    rows_as_dicts = []
    for row in rows:
        rows_as_dicts.append(dict(row))
    
    # Prepare pagination info
    pagination = {
        'page': page,
        'per_page': per_page,
        'total': total_count,
        'total_pages': total_pages,
        'has_prev': page > 1,
        'has_next': page < total_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < total_pages else None
    }
    
    # Current filters for template
    filters = {
        'session_id': session_filter,
        'who': who_filter,
        'what': what_filter,
        'where': where_filter,
        'date_from': date_from,
        'date_to': date_to,
        'sort_by': sort_by,
        'sort_order': sort_order
    }
    
    return render_template('memories.html', rows=rows_as_dicts, pagination=pagination, filters=filters)


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
            faiss_index.remove(memory_id)
            faiss_index.save()
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
            faiss_index.reset()
            faiss_index.save()
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

@app.route('/api/memories/export', methods=['POST'])
def api_export_memories():
    """Export memories to JSON file"""
    try:
        # Get filter parameters
        data = request.get_json() or {}
        session_id = data.get('session_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Parse dates if provided
        if start_date:
            start_date = datetime.fromisoformat(start_date)
        if end_date:
            end_date = datetime.fromisoformat(end_date)
        
        # Create exporter and export data
        exporter = MemoryExporter(cfg.db_path)
        export_data = exporter.export_memories(session_id, start_date, end_date)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        # Generate filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"memories_export_{timestamp}.json"
        
        # Send file and delete after sending
        def remove_file(response):
            try:
                os.unlink(temp_path)
            except:
                pass
            return response
        
        return send_file(
            temp_path,
            mimetype='application/json',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/memories/import', methods=['POST'])
def api_import_memories():
    """Import memories from JSON file"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Get import options
        merge_strategy = request.form.get('merge_strategy', 'skip')
        regenerate_embeddings = request.form.get('regenerate_embeddings', 'false').lower() == 'true'
        
        # Read and parse file
        file_content = file.read().decode('utf-8')
        import_data = json.loads(file_content)
        
        # Create importer with proper stores
        sql_store = MemoryStore(cfg.db_path)
        vector_store = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
        importer = MemoryImporter(sql_store, vector_store, cfg.embed_model)
        
        # Import memories
        result = importer.import_memories(
            import_data,
            merge_strategy=merge_strategy,
            regenerate_embeddings=regenerate_embeddings
        )
        
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        return jsonify({'success': False, 'error': f'Invalid JSON file: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clusters')
def clusters_page():
    """Clusters visualization page"""
    return render_template('clusters.html')

@app.route('/api/clusters/visualization', methods=['GET'])
def api_clusters_visualization():
    """Get cluster visualization data with dimensionality reduction"""
    try:
        method = request.args.get('method', 'tsne')
        perplexity = int(request.args.get('perplexity', 30))
        max_points = int(request.args.get('max_points', 500))
        
        # Get embeddings directly from FAISS index
        # Increase limit to ensure we get representation from all clusters
        # We'll sample later if needed, but first need to see all clusters
        fetch_limit = min(max_points * 10, 5000)  # Fetch more to ensure cluster diversity
        memory_ids, embeddings_array = faiss_index.get_all_vectors(limit=fetch_limit)
        
        if len(memory_ids) == 0:
            return jsonify({
                'success': False,
                'error': 'No embeddings found in FAISS index'
            })
        
        # Get memory metadata from database
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        # Build query for memory metadata
        placeholders = ','.join(['?' for _ in memory_ids])
        query = f"""
        SELECT m.memory_id, m.what, m.when_ts, m.who_id,
               COALESCE(u.accesses, 0) as usage_count,
               COALESCE(u.last_access, m.created_at) as last_accessed
        FROM memories m
        LEFT JOIN usage_stats u ON m.memory_id = u.memory_id
        WHERE m.memory_id IN ({placeholders})
        """
        
        rows = con.execute(query, memory_ids).fetchall()
        con.close()
        
        if not rows:
            return jsonify({
                'success': False,
                'error': 'No memories found in database for FAISS vectors'
            })
        
        # Create metadata lookup from database rows
        memory_metadata = {}
        embeddings_dict = {}
        
        # Create a mapping of memory_id to row data
        row_lookup = {row['memory_id']: row for row in rows}
        
        # Process memory_ids and embeddings from FAISS
        valid_memory_ids = []
        valid_embeddings = []
        
        for i, memory_id in enumerate(memory_ids):
            if memory_id in row_lookup:
                row = row_lookup[memory_id]
                valid_memory_ids.append(memory_id)
                valid_embeddings.append(embeddings_array[i])
                
                # Store embedding in dict
                embeddings_dict[memory_id] = embeddings_array[i]
                
                # Store metadata
                memory_metadata[memory_id] = {
                    'what': row['what'][:50] if row['what'] else 'N/A',
                    'when': row['when_ts'],
                    'who': row['who_id'],
                    'usage_count': row['usage_count'],
                    'last_accessed': row['last_accessed']
                }
        
        # Update arrays with valid data only
        memory_ids = valid_memory_ids
        embeddings_array = np.array(valid_embeddings) if valid_embeddings else np.array([])
        
        if len(embeddings_array) == 0:
            return jsonify({
                'success': False,
                'error': 'No valid embeddings found after filtering'
            })
        
        # Adjust number of clusters based on sample size
        n_samples = len(embeddings_array)
        optimal_n_clusters = min(max(8, n_samples // 10), 64)  # Between 8 and 64 clusters
        
        # Reinitialize clusterer if needed with appropriate cluster count
        if n_samples < liquid_clusters.base_clusterer.model.n_clusters:
            liquid_clusters.base_clusterer.model.n_clusters = optimal_n_clusters
        
        # Initialize clusters properly if not already done
        if not liquid_clusters.base_clusterer._fitted:
            liquid_clusters.base_clusterer.partial_fit(embeddings_array)
        
        # Assign memories to clusters
        cluster_assignments = liquid_clusters.base_clusterer.assign(embeddings_array)
        for i, memory_id in enumerate(memory_ids):
            cluster_id = int(cluster_assignments[i])
            liquid_clusters.memory_clusters[memory_id] = cluster_id
            
            if cluster_id not in liquid_clusters.cluster_members:
                liquid_clusters.cluster_members[cluster_id] = set()
            liquid_clusters.cluster_members[cluster_id].add(memory_id)
            
            # Initialize cluster energy with some variation
            if cluster_id not in liquid_clusters.cluster_energy:
                liquid_clusters.cluster_energy[cluster_id] = 0.3 + np.random.random() * 0.4
        
        # Update liquid clusters with current data
        liquid_clusters.flow_step(memory_ids, embeddings_dict, memory_metadata)
        liquid_clusters.merge_similar_clusters(embeddings_dict)
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(embeddings_array)
        elif method == 'tsne':
            reducer = TSNE(n_components=3, perplexity=min(perplexity, len(memory_ids)-1), 
                          random_state=42, max_iter=1000)
            coords_3d = reducer.fit_transform(embeddings_array)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=3, n_neighbors=min(15, len(memory_ids)-1),
                                   random_state=42)
                coords_3d = reducer.fit_transform(embeddings_array)
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'UMAP not installed. Use pip install umap-learn'
                })
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown reduction method: {method}'
            })
        
        # Calculate recency scores
        now = datetime.now()
        recency_scores = []
        for memory_id in memory_ids:
            last_accessed = memory_metadata[memory_id]['last_accessed']
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            time_diff = (now - last_accessed).total_seconds()
            recency_score = np.exp(-time_diff / (7 * 86400))  # 7-day decay
            recency_scores.append(recency_score)
        
        # Prepare points for visualization
        points = []
        cluster_representation = {}  # Track how many points per cluster
        
        for i, memory_id in enumerate(memory_ids):
            cluster_id = liquid_clusters.memory_clusters.get(memory_id, 0)
            energy = liquid_clusters.cluster_energy.get(cluster_id, 0.5)
            
            point_data = {
                'memory_id': memory_id,
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'cluster_id': cluster_id,
                'energy': energy,
                'what': memory_metadata[memory_id]['what'],
                'usage_count': memory_metadata[memory_id]['usage_count'],
                'recency_score': recency_scores[i]
            }
            
            points.append(point_data)
            
            # Track cluster representation
            if cluster_id not in cluster_representation:
                cluster_representation[cluster_id] = 0
            cluster_representation[cluster_id] += 1
        
        # If we have too many points, sample intelligently to keep all clusters
        if len(points) > max_points:
            # Calculate points per cluster
            num_clusters = len(cluster_representation)
            points_per_cluster = max(3, max_points // num_clusters)  # At least 3 points per cluster
            
            sampled_points = []
            for cluster_id in cluster_representation:
                cluster_points = [p for p in points if p['cluster_id'] == cluster_id]
                # Sort by energy within cluster
                cluster_points.sort(key=lambda x: x.get('energy', 0), reverse=True)
                # Take top N from each cluster
                sampled_points.extend(cluster_points[:points_per_cluster])
            
            # If we have room, add more high-energy points
            if len(sampled_points) < max_points:
                remaining = [p for p in points if p not in sampled_points]
                remaining.sort(key=lambda x: x.get('energy', 0), reverse=True)
                sampled_points.extend(remaining[:max_points - len(sampled_points)])
            
            points = sampled_points[:max_points]
        
        # Get cluster statistics
        cluster_summary = liquid_clusters.get_cluster_summary()
        stats = {
            'total_memories': len(memory_ids),
            'n_clusters': cluster_summary['n_clusters'],
            'avg_energy': np.mean(list(cluster_summary['cluster_energies'].values())) 
                         if cluster_summary['cluster_energies'] else 0,
            'largest_cluster_size': max(cluster_summary['cluster_sizes'].values()) 
                                  if cluster_summary['cluster_sizes'] else 0
        }
        
        return jsonify({
            'success': True,
            'points': points,
            'stats': stats,
            'method': method
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/clusters/update-access', methods=['POST'])
def api_update_cluster_access():
    """Update cluster co-access patterns when memories are accessed"""
    try:
        data = request.get_json()
        memory_ids = data.get('memory_ids', [])
        
        if memory_ids:
            liquid_clusters.update_co_access(memory_ids)
            liquid_clusters.update_cluster_energy(memory_ids)
        
        return jsonify({
            'success': True,
            'updated': len(memory_ids)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clusters/liquid-state', methods=['GET'])
def api_clusters_liquid_state():
    """Get current liquid cluster state with flow information"""
    try:
        # Initialize clusters if empty
        if not liquid_clusters.memory_clusters:
            # Get limit from query params with a sensible default
            init_limit = int(request.args.get('init_limit', 1000))
            init_limit = min(init_limit, 5000)  # Cap at 5000 for safety
            
            # Load memories from database
            con = sqlite3.connect(cfg.db_path)
            con.row_factory = sqlite3.Row
            
            query = """
            SELECT m.memory_id, m.what
            FROM memories m
            ORDER BY m.created_at DESC
            LIMIT ?
            """
            rows = con.execute(query, (init_limit,)).fetchall()
            con.close()
            
            if rows:
                # Get embeddings and initialize clusters
                memory_ids = []
                embeddings_list = []
                
                # Process in batches for better performance
                batch_size = 100
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    for row in batch:
                        memory_id = row['memory_id']
                        embedding = faiss_index.get_vector(memory_id)
                        if embedding is not None:
                            memory_ids.append(memory_id)
                            embeddings_list.append(embedding)
                
                if embeddings_list:
                    # Initialize clusterer
                    embeddings_array = np.array(embeddings_list)
                    # More clusters for larger datasets
                    n_clusters = min(64, max(8, len(embeddings_list) // 20))
                    liquid_clusters.base_clusterer.model.n_clusters = n_clusters
                    liquid_clusters.base_clusterer.partial_fit(embeddings_array)
                    
                    # Assign memories to clusters
                    cluster_assignments = liquid_clusters.base_clusterer.assign(embeddings_array)
                    for i, memory_id in enumerate(memory_ids):
                        cluster_id = int(cluster_assignments[i])
                        liquid_clusters.memory_clusters[memory_id] = cluster_id
                        
                        if cluster_id not in liquid_clusters.cluster_members:
                            liquid_clusters.cluster_members[cluster_id] = set()
                        liquid_clusters.cluster_members[cluster_id].add(memory_id)
                        
                        # Initialize cluster energy
                        if cluster_id not in liquid_clusters.cluster_energy:
                            liquid_clusters.cluster_energy[cluster_id] = 0.3 + np.random.random() * 0.4
        
        # Get cluster summary
        summary = liquid_clusters.get_cluster_summary()
        
        # Get active flows (memories with high pull to other clusters)
        flows = []
        for memory_id in list(liquid_clusters.memory_clusters.keys())[:50]:  # Sample for performance
            current_cluster = liquid_clusters.memory_clusters[memory_id]
            
            # Check pull to other clusters
            for target_cluster in liquid_clusters.cluster_members.keys():
                if target_cluster != current_cluster:
                    # Simple pull calculation based on cluster energy difference
                    pull = abs(liquid_clusters.cluster_energy.get(target_cluster, 0.5) - 
                             liquid_clusters.cluster_energy.get(current_cluster, 0.5))
                    if pull > liquid_clusters.flow_rate * 0.8:  # Near threshold
                        flows.append({
                            'source': current_cluster,
                            'target': target_cluster,
                            'strength': pull
                        })
                        break  # One flow per memory
        
        # Get cluster information
        clusters = []
        for cluster_id, members in liquid_clusters.cluster_members.items():
            clusters.append({
                'id': cluster_id,
                'name': f'Cluster {cluster_id}',
                'size': len(members),
                'energy': liquid_clusters.cluster_energy.get(cluster_id, 0.5),
                'likely_to_merge': False  # Will be calculated based on similarity
            })
        
        return jsonify({
            'success': True,
            'clusters': clusters,
            'flows': flows[:50],  # Limit flows for visualization
            'total_memories': summary['total_memories'],
            'total_clusters': summary['n_clusters'],
            'active_flows': len(flows),
            'avg_affinity': 0.5  # Placeholder
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/clusters/affinity-matrix', methods=['GET'])
def api_clusters_affinity_matrix():
    """Get affinity matrix for a sample of memories"""
    try:
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        # Initialize if needed
        if not liquid_clusters.memory_clusters:
            # Quick initialization - call liquid-state first
            con = sqlite3.connect(cfg.db_path)
            con.row_factory = sqlite3.Row
            query = "SELECT memory_id FROM memories ORDER BY created_at DESC LIMIT 100"
            rows = con.execute(query).fetchall()
            con.close()
            
            if rows:
                memory_ids = []
                embeddings_list = []
                for row in rows[:50]:
                    embedding = faiss_index.get_vector(row['memory_id'])
                    if embedding is not None:
                        memory_ids.append(row['memory_id'])
                        embeddings_list.append(embedding)
                
                if embeddings_list:
                    embeddings_array = np.array(embeddings_list)
                    n_clusters = min(8, max(2, len(embeddings_list) // 10))
                    liquid_clusters.base_clusterer.model.n_clusters = n_clusters
                    liquid_clusters.base_clusterer.partial_fit(embeddings_array)
                    
                    cluster_assignments = liquid_clusters.base_clusterer.assign(embeddings_array)
                    for i, memory_id in enumerate(memory_ids):
                        cluster_id = int(cluster_assignments[i])
                        liquid_clusters.memory_clusters[memory_id] = cluster_id
                        
                        if cluster_id not in liquid_clusters.cluster_members:
                            liquid_clusters.cluster_members[cluster_id] = set()
                        liquid_clusters.cluster_members[cluster_id].add(memory_id)
                        
                        if cluster_id not in liquid_clusters.cluster_energy:
                            liquid_clusters.cluster_energy[cluster_id] = 0.5
        
        # Get sample memories with offset
        all_memory_ids = list(liquid_clusters.memory_clusters.keys())
        memory_ids = all_memory_ids[offset:offset + limit]
        if not memory_ids:
            return jsonify({'success': False, 'error': 'No memories with clusters at this offset'})
        
        # Get embeddings
        embeddings_dict = {}
        for mid in memory_ids:
            vec = faiss_index.get_vector(mid)
            if vec is not None:
                embeddings_dict[mid] = vec
        
        # Compute affinity matrix
        affinity = liquid_clusters.compute_affinity_matrix(memory_ids, embeddings_dict)
        
        # Get memory previews
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        query = "SELECT memory_id, what FROM memories WHERE memory_id IN ({})".format(
            ','.join(['?'] * len(memory_ids))
        )
        rows = con.execute(query, memory_ids).fetchall()
        con.close()
        
        memories = []
        for row in rows:
            memories.append({
                'id': row['memory_id'],
                'preview': row['what'][:50] if row['what'] else ''
            })
        
        # Compute details for each pair
        details = {}
        for i in range(len(memory_ids)):
            for j in range(i+1, len(memory_ids)):
                key = f"{i}-{j}"
                details[key] = {
                    'co_access': liquid_clusters.co_access_counts.get(
                        tuple(sorted([memory_ids[i], memory_ids[j]])), 0
                    ),
                    'temporal': 0.5,  # Placeholder
                    'semantic': affinity[i, j] if i < len(affinity) and j < len(affinity[0]) else 0
                }
        
        return jsonify({
            'success': True,
            'memories': memories,
            'matrix': affinity.tolist(),
            'details': details
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/memory-blocks', methods=['GET'])
def api_clusters_memory_blocks():
    """Get memory block visualization data"""
    try:
        # Get last query's blocks from database
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        # Get recent blocks
        blocks_query = """
        SELECT block_id, budget_tokens, used_tokens, has_more
        FROM memory_blocks
        ORDER BY created_at DESC
        LIMIT 5
        """
        block_rows = con.execute(blocks_query).fetchall()
        
        blocks = []
        for block_row in block_rows:
            # Get block members
            members_query = """
            SELECT m.memory_id, m.what, bm.token_count
            FROM block_members bm
            JOIN memories m ON bm.memory_id = m.memory_id
            WHERE bm.block_id = ?
            """
            member_rows = con.execute(members_query, (block_row['block_id'],)).fetchall()
            
            members = []
            for member in member_rows:
                members.append({
                    'id': member['memory_id'],
                    'preview': member['what'][:50] if member['what'] else '',
                    'tokens': member['token_count'] if member['token_count'] else 100
                })
            
            blocks.append({
                'block_id': block_row['block_id'],
                'budget_tokens': block_row['budget_tokens'],
                'used_tokens': block_row['used_tokens'],
                'has_more': block_row['has_more'],
                'members': members
            })
        
        con.close()
        
        return jsonify({
            'success': True,
            'blocks': blocks,
            'context_window': cfg.context_window
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/dynamics', methods=['GET'])
def api_clusters_dynamics():
    """Get cluster dynamics dashboard data"""
    try:
        summary = liquid_clusters.get_cluster_summary()
        
        # Calculate flow activity (memories that moved recently)
        flow_activity = 0  # Would track actual migrations
        
        # Find merge candidates
        merge_candidates = 0
        next_merge = None
        
        if len(liquid_clusters.cluster_members) > 1:
            # Check for similar clusters
            cluster_ids = list(liquid_clusters.cluster_members.keys())
            for i, c1 in enumerate(cluster_ids):
                for c2 in cluster_ids[i+1:]:
                    # Simple similarity check based on size
                    size1 = len(liquid_clusters.cluster_members[c1])
                    size2 = len(liquid_clusters.cluster_members[c2])
                    if abs(size1 - size2) < 5:  # Similar size
                        merge_candidates += 1
                        if not next_merge:
                            next_merge = {
                                'cluster1': f'Cluster {c1}',
                                'cluster2': f'Cluster {c2}'
                            }
        
        # Energy distribution
        energies = list(liquid_clusters.cluster_energy.values())
        energy_dist = {
            'high': len([e for e in energies if e > 0.7]),
            'medium': len([e for e in energies if 0.3 <= e <= 0.7]),
            'low': len([e for e in energies if e < 0.3])
        }
        
        # Token efficiency (mock data)
        token_efficiency = 85  # Percentage
        
        return jsonify({
            'success': True,
            'flow_activity': flow_activity,
            'recent_flows': 0,
            'merge_candidates': merge_candidates,
            'next_merge': next_merge,
            'energy_distribution': energy_dist,
            'token_efficiency': token_efficiency,
            'co_access_patterns': []  # Would include actual patterns
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/memory-journey/<memory_id>', methods=['GET'])
def api_memory_journey(memory_id):
    """Get journey history for a specific memory"""
    try:
        # For now, return mock journey data
        # In production, would track cluster history over time
        current_cluster = liquid_clusters.memory_clusters.get(memory_id)
        
        if current_cluster is None:
            return jsonify({'success': False, 'error': 'Memory not found'})
        
        journey = [
            {
                'timestamp': '2024-01-01T00:00:00',
                'cluster_id': 0,
                'cluster_name': 'Initial Cluster',
                'reason': 'Initial assignment'
            },
            {
                'timestamp': '2024-01-15T00:00:00',
                'cluster_id': current_cluster,
                'cluster_name': f'Cluster {current_cluster}',
                'reason': 'High affinity with cluster members'
            }
        ]
        
        return jsonify({
            'success': True,
            'memory_id': memory_id,
            'journey': journey
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/predictions', methods=['GET'])
def api_clusters_predictions():
    """Get predictions for memory migrations and cluster merges"""
    try:
        # Predict likely migrations
        likely_migrations = []
        
        # Sample some memories and check their pull to other clusters
        for memory_id in list(liquid_clusters.memory_clusters.keys())[:20]:
            current = liquid_clusters.memory_clusters[memory_id]
            
            # Check if this memory has high energy difference with its cluster
            current_energy = liquid_clusters.cluster_energy.get(current, 0.5)
            
            for target in liquid_clusters.cluster_members.keys():
                if target != current:
                    target_energy = liquid_clusters.cluster_energy.get(target, 0.5)
                    pull = abs(target_energy - current_energy)
                    
                    if pull > liquid_clusters.flow_rate * 0.9:  # Very close to threshold
                        likely_migrations.append({
                            'memory_id': memory_id,
                            'from_cluster': f'Cluster {current}',
                            'to_cluster': f'Cluster {target}',
                            'pull_strength': pull,
                            'reason': 'High energy differential'
                        })
                        break
        
        # Predict merges
        merge_predictions = []
        cluster_ids = list(liquid_clusters.cluster_members.keys())
        
        for i, c1 in enumerate(cluster_ids[:10]):
            for c2 in cluster_ids[i+1:i+5]:  # Check next few clusters
                # Simple similarity based on size
                size1 = len(liquid_clusters.cluster_members[c1])
                size2 = len(liquid_clusters.cluster_members[c2])
                similarity = 1 - abs(size1 - size2) / max(size1, size2)
                
                if similarity > liquid_clusters.merge_threshold * 0.9:
                    merge_predictions.append({
                        'cluster1': f'Cluster {c1}',
                        'cluster2': f'Cluster {c2}',
                        'similarity': similarity,
                        'combined_size': size1 + size2
                    })
        
        return jsonify({
            'success': True,
            'likely_migrations': likely_migrations[:10],
            'merge_predictions': merge_predictions[:5]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/flow-step', methods=['POST'])
def api_clusters_flow_step():
    """Perform one flow step for animation"""
    try:
        # Get a sample of memories
        memory_ids = list(liquid_clusters.memory_clusters.keys())[:50]
        
        # Get embeddings
        embeddings_dict = {}
        for mid in memory_ids:
            vec = faiss_index.get_vector(mid)
            if vec is not None:
                embeddings_dict[mid] = vec
        
        # Perform flow step
        if memory_ids and embeddings_dict:
            liquid_clusters.flow_step(memory_ids, embeddings_dict)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/settings', methods=['POST'])
def api_clusters_settings():
    """Update cluster settings"""
    try:
        data = request.get_json()
        
        if 'flow_rate' in data:
            liquid_clusters.flow_rate = float(data['flow_rate'])
        
        if 'merge_threshold' in data:
            liquid_clusters.merge_threshold = float(data['merge_threshold'])
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/clear', methods=['POST'])
def api_clusters_clear():
    """Clear cluster data to force re-initialization"""
    try:
        liquid_clusters.memory_clusters.clear()
        liquid_clusters.cluster_members.clear()
        liquid_clusters.cluster_energy.clear()
        liquid_clusters.co_access_counts.clear()
        liquid_clusters.last_access_times.clear()
        
        return jsonify({'success': True, 'message': 'Clusters cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/metrics', methods=['GET'])
def api_clusters_metrics():
    """Get live cluster metrics"""
    try:
        summary = liquid_clusters.get_cluster_summary()
        
        # Calculate metrics
        energies = list(liquid_clusters.cluster_energy.values())
        avg_energy = np.mean(energies) if energies else 0.5
        
        # Count potential merges
        merge_candidates = 0
        if len(liquid_clusters.cluster_members) > 1:
            cluster_ids = list(liquid_clusters.cluster_members.keys())
            for i, c1 in enumerate(cluster_ids):
                for c2 in cluster_ids[i+1:]:
                    size1 = len(liquid_clusters.cluster_members[c1])
                    size2 = len(liquid_clusters.cluster_members[c2])
                    if abs(size1 - size2) < 3:
                        merge_candidates += 1
        
        return jsonify({
            'success': True,
            'flow_activity': 0,  # Would track actual flows
            'flow_change': 0,
            'avg_energy': avg_energy,
            'merge_candidates': merge_candidates
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clusters/hierarchy', methods=['GET'])
def api_clusters_hierarchy():
    """Get hierarchical cluster data with statistics for treemap/sunburst visualization"""
    try:
        # Get all memories with cluster assignments
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        query = """
        SELECT m.memory_id, m.what, m.who_type, m.who_id, m.when_ts, 
               m.where_type, m.where_value, m.why, m.how,
               COALESCE(u.accesses, 0) as usage_count,
               COALESCE(u.last_access, m.created_at) as last_accessed
        FROM memories m
        LEFT JOIN usage_stats u ON m.memory_id = u.memory_id
        ORDER BY u.accesses DESC
        LIMIT 5000
        """
        
        rows = con.execute(query).fetchall()
        con.close()
        
        if not rows:
            return jsonify({'success': False, 'error': 'No memories found'})
        
        # Get embeddings for clustering
        memory_ids = [row['memory_id'] for row in rows]
        embeddings_dict = {}
        
        # Fetch embeddings from FAISS and ensure clustering
        embeddings_list = []
        valid_memory_ids = []
        for memory_id in memory_ids[:1000]:  # Limit for performance
            embedding = faiss_index.get_vector(memory_id)
            if embedding is not None:
                embeddings_dict[memory_id] = embedding
                embeddings_list.append(embedding)
                valid_memory_ids.append(memory_id)
        
        # Initialize clustering if needed
        if embeddings_list and not liquid_clusters.base_clusterer._fitted:
            embeddings_array = np.array(embeddings_list)
            liquid_clusters.base_clusterer.partial_fit(embeddings_array)
            
            # Assign clusters to all memories
            cluster_assignments = liquid_clusters.base_clusterer.assign(embeddings_array)
            for i, memory_id in enumerate(valid_memory_ids):
                cluster_id = int(cluster_assignments[i])
                liquid_clusters.memory_clusters[memory_id] = cluster_id
                
                if cluster_id not in liquid_clusters.cluster_members:
                    liquid_clusters.cluster_members[cluster_id] = set()
                liquid_clusters.cluster_members[cluster_id].add(memory_id)
                
                # Initialize cluster energy
                if cluster_id not in liquid_clusters.cluster_energy:
                    liquid_clusters.cluster_energy[cluster_id] = 0.3 + np.random.random() * 0.4
        
        # Build cluster hierarchy
        cluster_data = {}
        for row in rows:
            memory_id = row['memory_id']
            # Try to assign to cluster if not already assigned
            if memory_id not in liquid_clusters.memory_clusters and memory_id in embeddings_dict:
                # Assign to nearest cluster
                embedding = embeddings_dict[memory_id]
                cluster_id = liquid_clusters.base_clusterer.assign(np.array([embedding]))[0]
                liquid_clusters.memory_clusters[memory_id] = int(cluster_id)
                
                if cluster_id not in liquid_clusters.cluster_members:
                    liquid_clusters.cluster_members[cluster_id] = set()
                liquid_clusters.cluster_members[cluster_id].add(memory_id)
            
            cluster_id = liquid_clusters.memory_clusters.get(memory_id, -1)
            
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = {
                    'id': cluster_id,
                    'name': f'Cluster {cluster_id}',
                    'size': 0,
                    'energy': liquid_clusters.cluster_energy.get(cluster_id, 0.5),
                    'memories': [],
                    'who_types': {},
                    'what_keywords': {},
                    'where_types': {},
                    'temporal_range': {'min': None, 'max': None}
                }
            
            cluster = cluster_data[cluster_id]
            cluster['size'] += 1
            
            # Aggregate statistics
            if row['who_type']:
                cluster['who_types'][row['who_type']] = cluster['who_types'].get(row['who_type'], 0) + 1
            if row['where_type']:
                cluster['where_types'][row['where_type']] = cluster['where_types'].get(row['where_type'], 0) + 1
            
            # Track temporal range
            when_ts = row['when_ts']
            if when_ts:
                if not cluster['temporal_range']['min'] or when_ts < cluster['temporal_range']['min']:
                    cluster['temporal_range']['min'] = when_ts
                if not cluster['temporal_range']['max'] or when_ts > cluster['temporal_range']['max']:
                    cluster['temporal_range']['max'] = when_ts
            
            # Add memory summary (limit per cluster for performance)
            if len(cluster['memories']) < 10:
                cluster['memories'].append({
                    'id': memory_id,
                    'what': row['what'][:100] if row['what'] else '',
                    'usage': row['usage_count']
                })
        
        # Create hierarchical structure
        hierarchy = {
            'name': 'All Memories',
            'children': [],
            'value': len(rows)
        }
        
        for cluster_id, cluster in sorted(cluster_data.items(), key=lambda x: x[1]['size'], reverse=True)[:50]:
            cluster_node = {
                'name': cluster['name'],
                'value': cluster['size'],
                'energy': cluster['energy'],
                'cluster_id': cluster_id,
                'who_types': cluster['who_types'],
                'where_types': cluster['where_types'],
                'temporal_range': cluster['temporal_range'],
                'sample_memories': cluster['memories'][:5]
            }
            hierarchy['children'].append(cluster_node)
        
        return jsonify({
            'success': True,
            'hierarchy': hierarchy,
            'total_memories': len(rows),
            'total_clusters': len(cluster_data)
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/clusters/<int:cluster_id>/members', methods=['GET'])
def api_cluster_members(cluster_id):
    """Get paginated members of a specific cluster"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        sort_by = request.args.get('sort_by', 'usage')  # usage, recency, energy
        
        # Handle unclustered memories (cluster_id = -1)
        if cluster_id == -1:
            # Get all memories that are not in any cluster
            con = sqlite3.connect(cfg.db_path)
            con.row_factory = sqlite3.Row
            
            # Get memories without cluster assignment
            all_clustered = set()
            for members in liquid_clusters.cluster_members.values():
                all_clustered.update(members)
            
            query = """
            SELECT m.*, 
                   COALESCE(u.accesses, 0) as usage_count,
                   COALESCE(u.last_access, m.created_at) as last_accessed
            FROM memories m
            LEFT JOIN usage_stats u ON m.memory_id = u.memory_id
            ORDER BY m.created_at DESC
            LIMIT 500
            """
            
            rows = con.execute(query).fetchall()
            con.close()
            
            # Filter to only unclustered
            unclustered_rows = [row for row in rows if row['memory_id'] not in all_clustered]
            member_ids = [row['memory_id'] for row in unclustered_rows]
            
            # Paginate
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_rows = unclustered_rows[start_idx:end_idx]
            
            # Format response
            members = []
            for row in paginated_rows:
                members.append({
                    'memory_id': row['memory_id'],
                    'what': row['what'],
                    'who': f"{row['who_type']}:{row['who_id']}",
                    'when': row['when_ts'],
                    'where': f"{row['where_type']}:{row['where_value']}",
                    'why': row['why'],
                    'how': row['how'],
                    'usage_count': row['usage_count'],
                    'last_accessed': row['last_accessed'],
                    'energy': 0.5
                })
            
            return jsonify({
                'success': True,
                'members': members,
                'total': len(unclustered_rows),
                'page': page,
                'per_page': per_page,
                'cluster_info': {
                    'id': -1,
                    'size': len(unclustered_rows),
                    'energy': 0.5
                }
            })
        
        # Get cluster members for regular clusters
        member_ids = list(liquid_clusters.cluster_members.get(cluster_id, set()))
        
        if not member_ids:
            return jsonify({
                'success': True,
                'members': [],
                'total': 0,
                'page': page,
                'per_page': per_page
            })
        
        # Get memory details from database
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        placeholders = ','.join(['?' for _ in member_ids])
        query = f"""
        SELECT m.*, 
               COALESCE(u.accesses, 0) as usage_count,
               COALESCE(u.last_access, m.created_at) as last_accessed
        FROM memories m
        LEFT JOIN usage_stats u ON m.memory_id = u.memory_id
        WHERE m.memory_id IN ({placeholders})
        """
        
        # Apply sorting
        if sort_by == 'usage':
            query += " ORDER BY u.accesses DESC"
        elif sort_by == 'recency':
            query += " ORDER BY u.last_access DESC"
        else:
            query += " ORDER BY m.created_at DESC"
        
        rows = con.execute(query, member_ids).fetchall()
        con.close()
        
        # Paginate
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_rows = rows[start_idx:end_idx]
        
        # Format response
        members = []
        for row in paginated_rows:
            members.append({
                'memory_id': row['memory_id'],
                'what': row['what'],
                'who': f"{row['who_type']}:{row['who_id']}",
                'when': row['when_ts'],
                'where': f"{row['where_type']}:{row['where_value']}",
                'why': row['why'],
                'how': row['how'],
                'usage_count': row['usage_count'],
                'last_accessed': row['last_accessed'],
                'energy': liquid_clusters.cluster_energy.get(cluster_id, 0.5)
            })
        
        return jsonify({
            'success': True,
            'members': members,
            'total': len(rows),
            'page': page,
            'per_page': per_page,
            'cluster_info': {
                'id': cluster_id,
                'size': len(member_ids),
                'energy': liquid_clusters.cluster_energy.get(cluster_id, 0.5)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clusters/relations/<relation_type>', methods=['GET'])
def api_cluster_relations(relation_type):
    """Analyze memories by 5W1H relation type"""
    try:
        limit = int(request.args.get('limit', 100))
        
        # Validate relation type
        valid_types = ['who', 'what', 'when', 'where', 'why', 'how']
        if relation_type not in valid_types:
            return jsonify({
                'success': False,
                'error': f'Invalid relation type. Must be one of: {valid_types}'
            })
        
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        # Build query based on relation type
        if relation_type == 'who':
            query = """
            SELECT who_type, who_id, COUNT(*) as count,
                   GROUP_CONCAT(memory_id) as memory_ids
            FROM memories
            WHERE who_type IS NOT NULL
            GROUP BY who_type, who_id
            ORDER BY count DESC
            LIMIT ?
            """
            params = [limit]
            
        elif relation_type == 'where':
            query = """
            SELECT where_type, where_value, COUNT(*) as count,
                   GROUP_CONCAT(memory_id) as memory_ids
            FROM memories
            WHERE where_type IS NOT NULL
            GROUP BY where_type, where_value
            ORDER BY count DESC
            LIMIT ?
            """
            params = [limit]
            
        elif relation_type == 'when':
            query = """
            SELECT DATE(when_ts) as date, COUNT(*) as count,
                   GROUP_CONCAT(memory_id) as memory_ids
            FROM memories
            WHERE when_ts IS NOT NULL
            GROUP BY DATE(when_ts)
            ORDER BY date DESC
            LIMIT ?
            """
            params = [limit]
            
        elif relation_type == 'what':
            # For 'what', we'll extract common keywords
            query = """
            SELECT memory_id, what
            FROM memories
            WHERE what IS NOT NULL
            ORDER BY LENGTH(what) DESC
            LIMIT ?
            """
            params = [limit * 10]  # Get more for keyword extraction
            
        elif relation_type == 'why':
            query = """
            SELECT SUBSTR(why, 1, 50) as why_summary, COUNT(*) as count,
                   GROUP_CONCAT(memory_id) as memory_ids
            FROM memories
            WHERE why IS NOT NULL AND why != ''
            GROUP BY why_summary
            ORDER BY count DESC
            LIMIT ?
            """
            params = [limit]
            
        else:  # how
            query = """
            SELECT SUBSTR(how, 1, 50) as how_summary, COUNT(*) as count,
                   GROUP_CONCAT(memory_id) as memory_ids
            FROM memories
            WHERE how IS NOT NULL AND how != ''
            GROUP BY how_summary
            ORDER BY count DESC
            LIMIT ?
            """
            params = [limit]
        
        rows = con.execute(query, params).fetchall()
        
        # Process results based on type
        if relation_type == 'what':
            # Extract keywords from 'what' field
            from collections import Counter
            import re
            
            keywords = Counter()
            memory_keyword_map = {}
            
            for row in rows:
                text = row['what'].lower()
                # Extract words (simple tokenization)
                words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ chars
                for word in words:
                    keywords[word] += 1
                    if word not in memory_keyword_map:
                        memory_keyword_map[word] = []
                    memory_keyword_map[word].append(row['memory_id'])
            
            # Get top keywords
            top_keywords = keywords.most_common(limit)
            
            relations = []
            for keyword, count in top_keywords:
                relations.append({
                    'value': keyword,
                    'count': count,
                    'memory_ids': memory_keyword_map[keyword][:10]  # Limit memory IDs
                })
        
        else:
            # Process other relation types
            relations = []
            for row in rows:
                if relation_type == 'who':
                    value = f"{row['who_type']}:{row['who_id']}"
                elif relation_type == 'where':
                    value = f"{row['where_type']}:{row['where_value']}"
                elif relation_type == 'when':
                    value = row['date']
                elif relation_type == 'why':
                    value = row['why_summary']
                else:  # how
                    value = row['how_summary']
                
                memory_ids = row['memory_ids'].split(',') if row['memory_ids'] else []
                
                relations.append({
                    'value': value,
                    'count': row['count'],
                    'memory_ids': memory_ids[:10]  # Limit to prevent huge responses
                })
        
        # Build co-occurrence network for visualization
        network_nodes = []
        network_links = []
        
        for i, rel in enumerate(relations[:20]):  # Top 20 for network
            network_nodes.append({
                'id': i,
                'label': str(rel['value'])[:30],
                'value': rel['count'],
                'group': relation_type
            })
        
        # Find co-occurrences (simplified - memories that share entities)
        for i in range(len(network_nodes)):
            for j in range(i + 1, len(network_nodes)):
                shared = set(relations[i]['memory_ids']) & set(relations[j]['memory_ids'])
                if shared:
                    network_links.append({
                        'source': i,
                        'target': j,
                        'value': len(shared)
                    })
        
        con.close()
        
        return jsonify({
            'success': True,
            'relation_type': relation_type,
            'relations': relations,
            'network': {
                'nodes': network_nodes,
                'links': network_links
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/clusters/timeline', methods=['GET'])
def api_clusters_timeline():
    """Get temporal evolution of clusters"""
    try:
        days = int(request.args.get('days', 30))
        
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        query = """
        SELECT DATE(when_ts) as date, 
               COUNT(*) as memory_count,
               COUNT(DISTINCT session_id) as session_count
        FROM memories
        WHERE when_ts >= datetime('now', '-' || ? || ' days')
        GROUP BY DATE(when_ts)
        ORDER BY date
        """
        
        rows = con.execute(query, [days]).fetchall()
        
        # Get cluster evolution data
        timeline_data = []
        for row in rows:
            timeline_data.append({
                'date': row['date'],
                'memory_count': row['memory_count'],
                'session_count': row['session_count']
            })
        
        con.close()
        
        return jsonify({
            'success': True,
            'timeline': timeline_data,
            'days': days
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clusters/network', methods=['GET'])
def api_clusters_network():
    """Get cluster interconnection network"""
    try:
        # Initialize if needed
        if not liquid_clusters.memory_clusters:
            # Quick initialization
            response = api_clusters_liquid_state()
            if isinstance(response, tuple):
                return response
        
        # Build network graph
        nodes = []
        links = []
        
        # Create nodes for each cluster
        for cluster_id in liquid_clusters.cluster_members.keys():
            nodes.append({
                'id': str(cluster_id),  # Ensure string for D3.js
                'label': f'Cluster {cluster_id}',
                'size': len(liquid_clusters.cluster_members.get(cluster_id, [])),
                'energy': liquid_clusters.cluster_energy.get(cluster_id, 0.5)
            })
        
        # Calculate connections between clusters based on affinity
        processed_pairs = set()
        cluster_ids = list(liquid_clusters.cluster_members.keys())
        
        for i, source_id in enumerate(cluster_ids):
            for target_id in cluster_ids[i+1:]:
                # Calculate connection strength based on cluster properties
                # Simple approach: connect clusters with similar energy levels
                energy_diff = abs(
                    liquid_clusters.cluster_energy.get(source_id, 0.5) - 
                    liquid_clusters.cluster_energy.get(target_id, 0.5)
                )
                
                # Create connection if energy difference is small
                if energy_diff < 0.3:  # Threshold for connection
                    weight = 1.0 - energy_diff  # Stronger connection for more similar energy
                    
                    links.append({
                        'source': str(source_id),
                        'target': str(target_id),
                        'value': weight * 10  # Scale up for visualization
                    })
        
        # Also add connections based on co-access if available
        for (mem1, mem2), count in liquid_clusters.co_access_counts.items():
            if count > 0 and mem1 in liquid_clusters.memory_clusters and mem2 in liquid_clusters.memory_clusters:
                cluster1 = liquid_clusters.memory_clusters[mem1]
                cluster2 = liquid_clusters.memory_clusters[mem2]
                
                if cluster1 != cluster2:
                    pair = tuple(sorted([str(cluster1), str(cluster2)]))
                    if pair not in processed_pairs:
                        # Find existing link or create new one
                        existing_link = None
                        for link in links:
                            if (link['source'], link['target']) == pair or (link['target'], link['source']) == pair:
                                existing_link = link
                                break
                        
                        if existing_link:
                            existing_link['value'] = min(existing_link['value'] + count * 0.1, 20)
                        else:
                            links.append({
                                'source': pair[0],
                                'target': pair[1],
                                'value': count * 0.5
                            })
                        processed_pairs.add(pair)
        
        return jsonify({
            'success': True,
            'network': {
                'nodes': nodes,
                'links': links
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/memories/export-preview', methods=['POST'])
def api_export_preview():
    """Preview what will be exported"""
    try:
        # Get filter parameters
        data = request.get_json() or {}
        session_id = data.get('session_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Parse dates if provided
        if start_date:
            start_date = datetime.fromisoformat(start_date)
        if end_date:
            end_date = datetime.fromisoformat(end_date)
        
        # Count memories that match filters
        import sqlite3
        con = sqlite3.connect(cfg.db_path)
        cursor = con.cursor()
        
        query = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if start_date:
            query += " AND when_ts >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND when_ts <= ?"
            params.append(end_date.isoformat())
        
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        
        # Get sample of memories (first 5)
        query = query.replace("COUNT(*)", "memory_id, what, when_ts, who_id") + " LIMIT 5"
        cursor.execute(query, params)
        samples = cursor.fetchall()
        
        con.close()
        
        return jsonify({
            'success': True,
            'count': count,
            'samples': [
                {
                    'memory_id': s[0],
                    'what': s[1],
                    'when': s[2],
                    'who': s[3]
                } for s in samples
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/documents')
def documents():
    """Document upload page"""
    return render_template('documents.html')


@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """Handle document upload and ingestion"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Get chunking parameters
        chunking_strategy = request.form.get('chunking_strategy', 'semantic')
        max_chunk_size = int(request.form.get('max_chunk_size', 2000))
        chunk_overlap = int(request.form.get('chunk_overlap', 200))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Initialize document parser with selected strategy
            if chunking_strategy == 'sentence':
                chunker = SentenceChunker(max_chunk_size, chunk_overlap)
            elif chunking_strategy == 'paragraph':
                chunker = ParagraphChunker(max_chunk_size, chunk_overlap)
            else:
                chunker = SemanticChunker(max_chunk_size, chunk_overlap)
            
            parser = DocumentParser(chunking_strategy=chunker)
            
            # Parse document
            parsed_doc = parser.parse(filepath)
            
            if not parsed_doc.success:
                return jsonify({
                    'success': False,
                    'error': 'Failed to parse document',
                    'details': parsed_doc.extraction_errors
                }), 500
            
            # Ingest chunks into memory
            memories_created = 0
            failed_chunks = []
            
            for chunk in parsed_doc.chunks:
                try:
                    # Create memory text
                    memory_text = chunk.to_memory_text()
                    
                    # Add metadata
                    metadata = {
                        'source': 'document',
                        'file_name': filename,
                        'file_type': parsed_doc.file_type,
                        'chunk_index': chunk.chunk_index,
                        'total_chunks': chunk.total_chunks,
                        'extraction_method': chunk.extraction_method
                    }
                    
                    # Ingest into memory
                    event = RawEvent(
                        raw_text=memory_text,
                        who='document_parser',
                        actor='document',
                        metadata=metadata
                    )
                    
                    memory = router.ingest(event)
                    if memory:
                        memories_created += 1
                except Exception as e:
                    failed_chunks.append({
                        'chunk': chunk.chunk_index,
                        'error': str(e)
                    })
            
            # Return results
            return jsonify({
                'success': True,
                'file_name': filename,
                'file_type': parsed_doc.file_type,
                'chunks_created': len(parsed_doc.chunks),
                'memories_created': memories_created,
                'total_words': parsed_doc.total_words,
                'total_chars': parsed_doc.total_chars,
                'failed_chunks': failed_chunks,
                'extraction_errors': parsed_doc.extraction_errors
            })
            
        finally:
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/documents/parse', methods=['POST'])
def parse_document_path():
    """Parse a document from a file path"""
    try:
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'success': False, 'error': 'No file path provided'}), 400
        
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Get chunking parameters
        chunking_strategy = data.get('chunking_strategy', 'semantic')
        max_chunk_size = data.get('max_chunk_size', 2000)
        chunk_overlap = data.get('chunk_overlap', 200)
        
        # Initialize parser
        if chunking_strategy == 'sentence':
            chunker = SentenceChunker(max_chunk_size, chunk_overlap)
        elif chunking_strategy == 'paragraph':
            chunker = ParagraphChunker(max_chunk_size, chunk_overlap)
        else:
            chunker = SemanticChunker(max_chunk_size, chunk_overlap)
        
        parser = DocumentParser(chunking_strategy=chunker)
        
        # Parse document
        parsed_doc = parser.parse(file_path)
        
        # Return parsing summary
        return jsonify({
            'success': parsed_doc.success,
            'summary': parsed_doc.get_summary(),
            'chunks': len(parsed_doc.chunks),
            'errors': parsed_doc.extraction_errors
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
