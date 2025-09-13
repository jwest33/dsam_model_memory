from __future__ import annotations
import os
import json
import sqlite3
import csv
import io
from flask import Flask, render_template, request, jsonify, session, make_response
from datetime import datetime
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.types import RetrievalQuery, Candidate
from agentic_memory.settings_manager import SettingsManager
from agentic_memory.block_builder import greedy_knapsack
from agentic_memory.import_processor import ImportProcessor
from agentic_memory.router import MemoryRouter
from agentic_memory.types import RawEvent
import httpx
import asyncio

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.urandom(24)  # For session management

# Make range available in templates
app.jinja_env.globals.update(range=range)

# Initialize settings manager
settings_manager = SettingsManager()

# Initialize core components
store = MemoryStore(cfg.db_path)
embed_dim = int(os.getenv('AM_EMBEDDING_DIM', '1024'))
faiss_index = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
retriever = HybridRetriever(store, faiss_index)
embedder = get_llama_embedder()

# Initialize memory router for chat integration
memory_router = MemoryRouter(store, faiss_index)

# Load persisted weights or use defaults
current_weights = settings_manager.get_weights()

@app.route('/analyzer')
def analyzer():
    """Main analyzer interface"""
    # Get context window configuration
    context_window = cfg.context_window
    reserve_output = cfg.reserve_output_tokens
    reserve_system = cfg.reserve_system_tokens
    default_budget = context_window - reserve_output - reserve_system
    
    return render_template('analyzer.html', 
                         context_window=context_window,
                         default_budget=default_budget)

@app.route('/api/search', methods=['POST'])
def search():
    """Perform retrieval with custom weights and knapsack token packing"""
    try:
        data = request.json
        query = data.get('query', '')
        weights = data.get('weights', current_weights)
        token_budget = data.get('token_budget')
        
        # If no token budget specified, use context window minus reserves
        if token_budget is None:
            token_budget = cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens - 512
        token_budget = max(512, token_budget)  # Ensure minimum budget
        
        # For initial candidate retrieval, get ALL memories for comprehensive search
        # The knapsack algorithm will handle selection based on token budget
        # Use a very large number to effectively get everything
        initial_top_k = data.get('initial_candidates', 999999)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Decompose query
        decomposition = retriever.decompose_query(query)

        # Create retrieval query
        rq = RetrievalQuery(
            session_id='analyzer',
            text=decomposition.get('what', query),
            actor_hint=decomposition['who'].get('id') if decomposition.get('who') else None,
            temporal_hint=decomposition.get('when')
        )

        # Generate query embedding in the same format as stored memories
        # Memories are embedded as: "WHAT: {what}\nWHY: {why}\nHOW: {how}\nRAW: {raw_text}"
        # For search, we'll use the decomposed components if available
        what_text = decomposition.get('what', query)
        why_text = decomposition.get('why', 'search')
        how_text = decomposition.get('how', 'query')

        # Format the embedding text to match storage format
        embed_text = f"WHAT: {what_text}\nWHY: {why_text}\nHOW: {how_text}\nRAW: {query}"
        qvec = embedder.encode([embed_text], normalize_embeddings=True)[0]
        
        # Search with custom weights - get many candidates for knapsack
        candidates = retriever.search_with_weights(rq, qvec, weights, topk_sem=initial_top_k, topk_lex=initial_top_k)
        
        # Apply knapsack algorithm to select memories within token budget
        selected_ids, tokens_used = greedy_knapsack(candidates, token_budget)
        
        # Filter candidates to only selected ones
        selected_candidates = [c for c in candidates if c.memory_id in selected_ids]
        
        # Get detailed scores for selected memories
        detailed = retriever.get_detailed_scores(selected_candidates)
        
        # Calculate statistics
        total_candidates = len(candidates)
        selected_count = len(selected_candidates)
        
        return jsonify({
            'success': True,
            'results': detailed,
            'decomposition': decomposition,
            'weights': weights,
            'token_budget': token_budget,
            'tokens_used': tokens_used,
            'selected_count': selected_count,
            'total_candidates': total_candidates,
            'selection_ratio': f"{selected_count}/{total_candidates}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/context', methods=['GET'])
def get_context_info():
    """Get context window configuration"""
    return jsonify({
        'context_window': cfg.context_window,
        'reserve_output': cfg.reserve_output_tokens,
        'reserve_system': cfg.reserve_system_tokens,
        'default_budget': cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens,
        'model': cfg.llm_model
    })

@app.route('/api/model_config', methods=['GET'])
def get_model_config():
    """Get model configuration (alias for context endpoint)"""
    return jsonify({
        'context_window': cfg.context_window,
        'reserve_output': cfg.reserve_output_tokens,
        'reserve_system': cfg.reserve_system_tokens,
        'default_budget': cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens,
        'model': cfg.llm_model
    })

@app.route('/api/weights', methods=['GET'])
def get_weights():
    """Get current weight configuration"""
    return jsonify(current_weights)

@app.route('/api/weights', methods=['POST'])
def update_weights():
    """Update and normalize weights"""
    try:
        weights = request.json
        normalized = retriever.update_weights(weights)
        
        # Save to settings manager
        normalized = settings_manager.set_weights(normalized)
        
        # Update global weights
        global current_weights
        current_weights = normalized
        
        return jsonify({
            'success': True,
            'weights': normalized,
            'sum': sum(normalized.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/weights/reset', methods=['POST'])
def reset_weights():
    """Reset weights to defaults"""
    try:
        # Reset in settings manager
        default_weights = settings_manager.reset_weights()
        
        # Update global weights
        global current_weights
        current_weights = default_weights
        
        return jsonify({
            'success': True,
            'weights': default_weights,
            'sum': sum(default_weights.values())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/decompose', methods=['POST'])
def decompose():
    """Decompose query into 5W1H components"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        components = retriever.decompose_query(query)
        return jsonify(components)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/memories')
def browse_memories():
    """Enhanced memory browser page"""
    return render_template('browser.html')

@app.route('/api/memories', methods=['GET'])
def api_memories():
    """API endpoint for memory browsing with filters"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        sort_by = request.args.get('sort_by', 'when_ts')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Filter parameters
        search_text = request.args.get('search', '')
        session_filter = request.args.get('session_id', '')
        who_filter = request.args.get('who', '')
        where_filter = request.args.get('where', '')
        entity_filter = request.args.get('entity', '')
        date_from = request.args.get('date_from', '')
        date_to = request.args.get('date_to', '')
        
        # Build SQL query with filters
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        where_clauses = []
        params = []
        
        # Text search across multiple fields
        if search_text:
            where_clauses.append("""
                (raw_text LIKE ? OR what LIKE ? OR why LIKE ? OR how LIKE ?)
            """)
            search_pattern = f"%{search_text}%"
            params.extend([search_pattern] * 4)
        
        if session_filter:
            where_clauses.append("session_id LIKE ?")
            params.append(f"%{session_filter}%")
        
        if who_filter:
            where_clauses.append("(who_id LIKE ? OR who_label LIKE ? OR who_type LIKE ?)")
            params.extend([f"%{who_filter}%"] * 3)
        
        if where_filter:
            where_clauses.append("(where_value LIKE ? OR where_type LIKE ?)")
            params.extend([f"%{where_filter}%"] * 2)
        
        if entity_filter:
            # Search for entity in the 'what' field (which contains JSON array)
            where_clauses.append("what LIKE ?")
            params.append(f"%{entity_filter}%")
        
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
        
        # Validate sort column
        valid_sort_columns = ['when_ts', 'created_at', 'memory_id', 'session_id', 
                            'who_id', 'what', 'where_value', 'token_count']
        if sort_by not in valid_sort_columns:
            sort_by = 'when_ts'
        
        # Build main query
        query = f"""
            SELECT 
                memory_id, session_id, source_event_id,
                who_type, who_id, who_label, who_list,
                what, when_ts, when_list,
                where_type, where_value, where_list,
                why, how, 
                raw_text, token_count, extra_json,
                created_at
            FROM memories{where_clause}
            ORDER BY {sort_by} {sort_order}
            LIMIT ? OFFSET ?
        """
        
        params.extend([per_page, offset])
        rows = con.execute(query, params).fetchall()
        
        # Process rows to extract entities
        memories = []
        for row in rows:
            memory = dict(row)
            # Extract entities from 'what' field
            memory['entities'] = retriever.extract_entities_from_what(memory.get('what', ''))
            memories.append(memory)
        
        con.close()
        
        return jsonify({
            'success': True,
            'memories': memories,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'total_pages': total_pages,
                'has_prev': page > 1,
                'has_next': page < total_pages
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/<memory_id>', methods=['GET'])
def get_memory(memory_id):
    """Get detailed information about a specific memory"""
    try:
        # Use direct SQL query to ensure we get all fields
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        query = """
            SELECT 
                memory_id, session_id, source_event_id,
                who_type, who_id, who_label, who_list,
                what, when_ts, when_list,
                where_type, where_value, where_list,
                why, how, 
                raw_text, token_count, extra_json,
                created_at
            FROM memories
            WHERE memory_id = ?
        """
        
        cursor = con.cursor()
        cursor.execute(query, (memory_id,))
        row = cursor.fetchone()
        con.close()
        
        if not row:
            return jsonify({'error': 'Memory not found'}), 404
        
        # Convert row to dict
        memory = dict(row)
        
        # Extract entities from 'what' field
        memory['entities'] = retriever.extract_entities_from_what(memory.get('what', ''))
        
        return jsonify(memory)
        
    except Exception as e:
        print(f"Error fetching memory {memory_id}: {str(e)}")  # Log to console
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/<memory_id>', methods=['DELETE'])
def delete_memory(memory_id):
    """Delete a specific memory"""
    try:
        con = sqlite3.connect(cfg.db_path)
        cursor = con.cursor()
        
        # Delete from all related tables
        cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        deleted_count = cursor.rowcount
        
        # Also delete from FTS and other tables
        cursor.execute("DELETE FROM mem_fts WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
        cursor.execute("DELETE FROM usage_stats WHERE memory_id = ?", (memory_id,))
        
        con.commit()
        con.close()
        
        # Remove from FAISS index
        try:
            faiss_index.remove(memory_id)
            faiss_index.save()
        except:
            pass  # Memory might not be in index
        
        return jsonify({
            'success': True,
            'deleted': deleted_count > 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/export', methods=['POST'])
def export_memories():
    """Export memories based on filters or selection"""
    try:
        data = request.json or {}
        memory_ids = data.get('memory_ids', [])
        format_type = data.get('format', 'json')  # json or csv
        filters = data.get('filters', {})
        
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        # Build query based on whether specific IDs or filters
        if memory_ids:
            # Export specific memories
            placeholders = ','.join(['?' for _ in memory_ids])
            query = f"""
                SELECT * FROM memories 
                WHERE memory_id IN ({placeholders})
                ORDER BY when_ts DESC
            """
            rows = con.execute(query, memory_ids).fetchall()
        else:
            # Export based on filters (similar to browse query)
            query = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            # Apply filters
            if filters.get('search'):
                query += " AND (raw_text LIKE ? OR what LIKE ? OR why LIKE ? OR how LIKE ?)"
                search_term = f"%{filters['search']}%"
                params.extend([search_term] * 4)
            
            if filters.get('session_id'):
                query += " AND session_id LIKE ?"
                params.append(f"%{filters['session_id']}%")
            
            if filters.get('who'):
                query += " AND (who_type LIKE ? OR who_id LIKE ?)"
                who_term = f"%{filters['who']}%"
                params.extend([who_term, who_term])
            
            if filters.get('where'):
                query += " AND (where_type LIKE ? OR where_value LIKE ?)"
                where_term = f"%{filters['where']}%"
                params.extend([where_term, where_term])
            
            if filters.get('date_from'):
                query += " AND when_ts >= ?"
                params.append(filters['date_from'])
            
            if filters.get('date_to'):
                query += " AND when_ts <= ?"
                params.append(filters['date_to'])
            
            query += " ORDER BY when_ts DESC"
            rows = con.execute(query, params).fetchall()
        
        con.close()
        
        # Convert to desired format
        memories = [dict(row) for row in rows]
        
        if format_type == 'csv':
            # Create CSV
            import csv
            import io
            
            output = io.StringIO()
            if memories:
                fieldnames = memories[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(memories)
            
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = 'attachment; filename=memories_export.csv'
            return response
        else:
            # JSON format
            response = jsonify({
                'memories': memories,
                'count': len(memories),
                'export_time': datetime.now().isoformat()
            })
            response.headers['Content-Disposition'] = 'attachment; filename=memories_export.json'
            return response
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/delete', methods=['POST'])
def delete_memories_bulk():
    """Delete multiple memories at once"""
    try:
        data = request.json or {}
        memory_ids = data.get('memory_ids', [])
        
        if not memory_ids:
            return jsonify({'error': 'No memory IDs provided'}), 400
        
        con = sqlite3.connect(cfg.db_path)
        cursor = con.cursor()
        
        deleted_count = 0
        failed_ids = []
        
        for memory_id in memory_ids:
            try:
                # Delete from memories table
                cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                if cursor.rowcount > 0:
                    deleted_count += 1
                    
                    # Delete from FTS
                    cursor.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
                    
                    # Remove from FAISS index
                    try:
                        faiss_index.remove(memory_id)
                    except:
                        pass  # Memory might not be in index
                else:
                    failed_ids.append(memory_id)
            except Exception as e:
                failed_ids.append(memory_id)
                print(f"Failed to delete {memory_id}: {e}")
        
        con.commit()
        con.close()
        
        # Save FAISS index if any deletions
        if deleted_count > 0:
            try:
                faiss_index.save()
            except:
                pass
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'failed_ids': failed_ids
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize import processor singleton
import_processor = ImportProcessor(cfg)

@app.route('/api/memories/import', methods=['POST'])
def import_memories():
    """Import memories from uploaded file or data"""
    try:
        content = None
        source_type = None
        session_id = None
        metadata = None
        
        # Check if file upload (multipart/form-data)
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read file content
            content = file.read().decode('utf-8')
            
            # Determine file type from extension
            if file.filename.endswith('.json'):
                source_type = 'json'
            elif file.filename.endswith('.csv'):
                source_type = 'csv'
            else:
                source_type = 'text'
                
            # Get optional parameters from form data
            session_id = request.form.get('session_id')
            metadata = request.form.get('metadata')
            
        # Check if JSON data submission
        elif request.is_json:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            content = data.get('content')
            source_type = data.get('source_type', 'auto')
            session_id = data.get('session_id')
            metadata = data.get('metadata')
        else:
            # No valid data format
            return jsonify({'error': 'No file or JSON data provided'}), 400
        
        if metadata and isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = None
        
        # Validate content
        if not content:
            return jsonify({'error': 'No content to import'}), 400
        
        # Create import job
        job_id = import_processor.create_import_job(
            data=content,
            source_type=source_type,
            session_id=session_id,
            metadata=metadata
        )
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Import job started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/import/status/<job_id>', methods=['GET'])
def get_import_status(job_id):
    """Get status of an import job"""
    try:
        status = import_processor.get_job_status(job_id)
        if not status:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/import/cancel/<job_id>', methods=['POST'])
def cancel_import(job_id):
    """Cancel an import job"""
    try:
        success = import_processor.cancel_job(job_id)
        if not success:
            return jsonify({'error': 'Job not found or already completed'}), 400
        
        return jsonify({
            'success': True,
            'message': 'Import job cancelled'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/import/jobs', methods=['GET'])
def list_import_jobs():
    """List all import jobs"""
    try:
        jobs = []
        for job_id in import_processor.jobs:
            status = import_processor.get_job_status(job_id)
            if status:
                jobs.append(status)
        
        # Sort by start time descending
        jobs.sort(key=lambda x: x['start_time'], reverse=True)
        
        return jsonify({
            'jobs': jobs[:20],  # Return last 20 jobs
            'total': len(jobs)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def analytics():
    """Analytics dashboard page"""
    return render_template('analytics.html')

@app.route('/api/analytics/entities', methods=['GET'])
def entity_analytics():
    """Get entity frequency and co-occurrence data"""
    try:
        limit = int(request.args.get('limit', 1000))
        
        # Get memories
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        query = "SELECT memory_id, what FROM memories ORDER BY created_at DESC LIMIT ?"
        rows = con.execute(query, (limit,)).fetchall()
        con.close()
        
        # Extract and count entities
        entity_counts = {}
        entity_memories = {}  # Track which memories contain each entity
        
        for row in rows:
            # Convert SQLite Row to dict
            row_dict = dict(row)
            entities = retriever.extract_entities_from_what(row_dict['what'])
            for entity in entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
                if entity not in entity_memories:
                    entity_memories[entity] = []
                entity_memories[entity].append(row_dict['memory_id'])
        
        # Calculate co-occurrence
        co_occurrence = []
        entities_list = list(entity_counts.keys())
        for i, entity1 in enumerate(entities_list[:50]):  # Limit for performance
            for entity2 in entities_list[i+1:i+20]:  # Check next 20 entities
                # Count memories that contain both entities
                shared = set(entity_memories[entity1]) & set(entity_memories[entity2])
                if len(shared) > 0:
                    co_occurrence.append({
                        'source': entity1,
                        'target': entity2,
                        'weight': len(shared)
                    })
        
        # Sort entities by frequency
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'success': True,
            'top_entities': sorted_entities[:50],
            'co_occurrence': co_occurrence[:100],  # Limit connections
            'total_entities': len(entity_counts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/temporal', methods=['GET'])
def temporal_analytics():
    """Get temporal distribution of memories"""
    try:
        days = int(request.args.get('days', 30))
        
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        # First get the actual date range from the database
        date_range_query = "SELECT MIN(when_ts) as min_date, MAX(when_ts) as max_date FROM memories"
        date_range = con.execute(date_range_query).fetchone()
        
        if not date_range or not date_range['min_date']:
            con.close()
            return jsonify({
                'success': True,
                'timeline': [],
                'days': days
            })
        
        # Determine the actual query range
        from datetime import datetime, timedelta
        end_date = datetime.fromisoformat(date_range['max_date'].replace(' ', 'T') if date_range['max_date'] else datetime.now().isoformat())
        
        # For specific day requests, show last N days from most recent data
        # For "all time" (days > 365), show full range
        if days > 365:
            # Show all data from beginning
            start_date = datetime.fromisoformat(date_range['min_date'].replace(' ', 'T'))
        else:
            # Show last N days from the most recent date
            start_date = end_date - timedelta(days=days)
        
        # Get aggregated data by date
        query = """
            SELECT 
                DATE(when_ts) as date,
                COUNT(*) as count,
                COUNT(DISTINCT session_id) as sessions,
                COUNT(DISTINCT who_list) as actors
            FROM memories
            WHERE DATE(when_ts) >= DATE(?) AND DATE(when_ts) <= DATE(?)
            GROUP BY DATE(when_ts)
            ORDER BY date
        """
        
        rows = con.execute(query, (start_date.date().isoformat(), end_date.date().isoformat())).fetchall()
        con.close()
        
        # Create a dict for easy lookup
        data_by_date = {}
        for row in rows:
            row_dict = dict(row)
            data_by_date[row_dict['date']] = {
                'date': row_dict['date'],
                'memories': row_dict['count'],
                'sessions': row_dict['sessions'],
                'actors': row_dict['actors']
            }
        
        # Fill in timeline with all dates in range (including gaps with 0)
        timeline = []
        current_date = start_date.date()
        end_date_obj = end_date.date()
        
        # Limit to reasonable number of data points for display
        total_days = (end_date_obj - current_date).days + 1
        
        # If more than 365 days, aggregate by week or month
        if total_days > 365:
            # Aggregate by month for very long ranges
            month_data = {}
            for date_str, data in data_by_date.items():
                month_key = date_str[:7]  # YYYY-MM format
                if month_key not in month_data:
                    month_data[month_key] = {
                        'date': f"{month_key}-01",
                        'memories': 0,
                        'sessions': set(),
                        'actors': set()
                    }
                month_data[month_key]['memories'] += data['memories']
                month_data[month_key]['sessions'].add(data['sessions'])
                month_data[month_key]['actors'].add(data['actors'])
            
            for month_key in sorted(month_data.keys()):
                timeline.append({
                    'date': month_data[month_key]['date'],
                    'memories': month_data[month_key]['memories'],
                    'sessions': len(month_data[month_key]['sessions']),
                    'actors': len(month_data[month_key]['actors'])
                })
        elif total_days > 90:
            # Aggregate by week for medium ranges
            week_data = {}
            for date_str, data in data_by_date.items():
                date_obj = datetime.fromisoformat(date_str).date()
                week_start = date_obj - timedelta(days=date_obj.weekday())
                week_key = week_start.isoformat()
                
                if week_key not in week_data:
                    week_data[week_key] = {
                        'date': week_key,
                        'memories': 0,
                        'sessions': set(),
                        'actors': set()
                    }
                week_data[week_key]['memories'] += data['memories']
                week_data[week_key]['sessions'].add(data['sessions'])
                week_data[week_key]['actors'].add(data['actors'])
            
            for week_key in sorted(week_data.keys()):
                timeline.append({
                    'date': week_data[week_key]['date'],
                    'memories': week_data[week_key]['memories'],
                    'sessions': len(week_data[week_key]['sessions']),
                    'actors': len(week_data[week_key]['actors'])
                })
        else:
            # Daily data for short ranges
            while current_date <= end_date_obj:
                date_str = current_date.isoformat()
                if date_str in data_by_date:
                    timeline.append(data_by_date[date_str])
                else:
                    timeline.append({
                        'date': date_str,
                        'memories': 0,
                        'sessions': 0,
                        'actors': 0
                    })
                current_date += timedelta(days=1)
        
        return jsonify({
            'success': True,
            'timeline': timeline,
            'days': days,
            'actual_range': {
                'from': start_date.date().isoformat(),
                'to': end_date.date().isoformat(),
                'total_days': total_days
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/network/<component>', methods=['GET'])
def network_analytics(component):
    """Get network data for any 5W1H component"""
    try:
        if component not in ['what', 'who', 'when', 'where', 'why', 'how']:
            return jsonify({'error': 'Invalid component'}), 400
            
        limit = int(request.args.get('limit', 1000))
        
        # Get memories
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        # Select the appropriate column based on component
        column_map = {
            'what': 'what',
            'who': 'who_list',
            'when': 'when_list',
            'where': 'where_list',
            'why': 'why',
            'how': 'how'
        }
        
        column = column_map[component]
        query = f"SELECT memory_id, {column} FROM memories WHERE {column} IS NOT NULL ORDER BY created_at DESC LIMIT ?"
        rows = con.execute(query, (limit,)).fetchall()
        con.close()
        
        # Extract entities/items based on component type
        item_counts = {}
        item_memories = {}
        
        for row in rows:
            # Convert SQLite Row to dict
            row_dict = dict(row)
            items = []
            data = row_dict[column]
            
            if component in ['who', 'when', 'where'] and data:
                # Parse JSON list for list columns
                try:
                    items = json.loads(data) if isinstance(data, str) else []
                except:
                    items = []
            elif component == 'what' and data:
                # Use existing entity extraction for what
                items = retriever.extract_entities_from_what(data)
            elif data:
                # For why and how, extract key phrases
                # Simple extraction - split on punctuation and take significant phrases
                import re
                phrases = re.split(r'[;,.]', str(data))
                items = [p.strip()[:30] for p in phrases if len(p.strip()) > 10][:3]
            
            for item in items:
                if item:
                    item_counts[item] = item_counts.get(item, 0) + 1
                    if item not in item_memories:
                        item_memories[item] = []
                    item_memories[item].append(row_dict['memory_id'])
        
        # Calculate co-occurrence
        co_occurrence = []
        items_list = list(item_counts.keys())
        for i, item1 in enumerate(items_list[:50]):
            for item2 in items_list[i+1:i+20]:
                shared = set(item_memories[item1]) & set(item_memories[item2])
                if len(shared) > 0:
                    co_occurrence.append({
                        'source': item1,
                        'target': item2,
                        'weight': len(shared)
                    })
        
        # Sort by frequency
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'success': True,
            'component': component,
            'top_items': sorted_items[:50],
            'co_occurrence': co_occurrence[:100],
            'total_items': len(item_counts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/scores', methods=['POST'])
def score_distribution():
    """Analyze score distribution for a query"""
    try:
        data = request.json
        query = data.get('query', '')
        weights = data.get('weights', current_weights)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get query embedding
        qvec = embedder.encode([query], normalize_embeddings=True)[0]
        
        # Create retrieval query
        rq = RetrievalQuery(
            session_id='analyzer',
            text=query,
            actor_hint=None,
            temporal_hint=None
        )
        
        # Get more candidates for analysis
        candidates = retriever.search_with_weights(rq, qvec, weights, topk_sem=200, topk_lex=200)
        
        # Analyze score distribution
        score_ranges = {
            '0.9-1.0': 0,
            '0.8-0.9': 0,
            '0.7-0.8': 0,
            '0.6-0.7': 0,
            '0.5-0.6': 0,
            '0.4-0.5': 0,
            '0.3-0.4': 0,
            '0.2-0.3': 0,
            '0.1-0.2': 0,
            '0.0-0.1': 0
        }
        
        component_averages = {
            'semantic': [],
            'lexical': [],
            'recency': [],
            'actor': [],
            'temporal': [],
            'spatial': [],
            'usage': []
        }
        
        for c in candidates:
            # Categorize by total score
            score = c.score
            if score >= 0.9:
                score_ranges['0.9-1.0'] += 1
            elif score >= 0.8:
                score_ranges['0.8-0.9'] += 1
            elif score >= 0.7:
                score_ranges['0.7-0.8'] += 1
            elif score >= 0.6:
                score_ranges['0.6-0.7'] += 1
            elif score >= 0.5:
                score_ranges['0.5-0.6'] += 1
            elif score >= 0.4:
                score_ranges['0.4-0.5'] += 1
            elif score >= 0.3:
                score_ranges['0.3-0.4'] += 1
            elif score >= 0.2:
                score_ranges['0.2-0.3'] += 1
            elif score >= 0.1:
                score_ranges['0.1-0.2'] += 1
            else:
                score_ranges['0.0-0.1'] += 1
            
            # Collect component scores
            component_averages['semantic'].append(c.semantic_score or 0)
            component_averages['lexical'].append(c.lexical_score or 0)
            component_averages['recency'].append(c.recency_score or 0)
            component_averages['actor'].append(c.actor_score or 0)
            component_averages['temporal'].append(c.temporal_score or 0)
            component_averages['spatial'].append(getattr(c, 'spatial_score', 0) or 0)
            component_averages['usage'].append(c.usage_score or 0)
        
        # Calculate averages
        for key in component_averages:
            values = component_averages[key]
            if values:
                component_averages[key] = sum(values) / len(values)
            else:
                component_averages[key] = 0
        
        return jsonify({
            'success': True,
            'score_distribution': score_ranges,
            'component_averages': component_averages,
            'total_candidates': len(candidates)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        stats = {}
        
        # Total memories
        stats['total_memories'] = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        
        # Unique sessions
        stats['unique_sessions'] = con.execute("SELECT COUNT(DISTINCT session_id) FROM memories").fetchone()[0]
        
        # Unique actors
        stats['unique_actors'] = con.execute("SELECT COUNT(DISTINCT who_id) FROM memories").fetchone()[0]
        
        # Date range
        date_range = con.execute("SELECT MIN(when_ts) as min_date, MAX(when_ts) as max_date FROM memories").fetchone()
        stats['date_range'] = {
            'from': date_range['min_date'],
            'to': date_range['max_date']
        }
        
        # Average token count
        stats['avg_tokens'] = con.execute("SELECT AVG(token_count) FROM memories").fetchone()[0] or 0
        
        con.close()
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get all application settings"""
    try:
        return jsonify(settings_manager.get_all_settings())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    try:
        updates = request.json
        success = settings_manager.update_settings(updates)
        
        # If weights were updated, sync with global variable
        if 'weights' in updates:
            global current_weights
            current_weights = settings_manager.get_weights()
        
        return jsonify({
            'success': success,
            'settings': settings_manager.get_all_settings()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/settings/section/<section>', methods=['POST'])
def reset_section(section):
    """Reset a specific settings section to defaults"""
    try:
        reset_settings = settings_manager.reset_section(section)
        
        # If weights section was reset, sync with global variable
        if section == 'weights':
            global current_weights
            current_weights = settings_manager.get_weights()
        
        return jsonify({
            'success': True,
            'section': section,
            'settings': reset_settings
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/settings/reset', methods=['POST'])
def reset_all_settings():
    """Reset all settings to defaults"""
    try:
        default_settings = settings_manager.reset_all()
        
        # Sync weights with global variable
        global current_weights
        current_weights = settings_manager.get_weights()
        
        return jsonify({
            'success': True,
            'settings': default_settings
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/settings/export', methods=['GET'])
def export_settings():
    """Export settings as JSON download"""
    try:
        settings = settings_manager.get_all_settings()
        return jsonify(settings), 200, {
            'Content-Disposition': 'attachment; filename=analyzer_settings.json',
            'Content-Type': 'application/json'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/import', methods=['POST'])
def import_settings():
    """Import settings from uploaded JSON"""
    try:
        if 'file' in request.files:
            file = request.files['file']
            settings = json.load(file)
        else:
            settings = request.json
        
        # Import settings
        success = settings_manager.update_settings(settings)
        
        # Sync weights with global variable
        global current_weights
        current_weights = settings_manager.get_weights()
        
        return jsonify({
            'success': success,
            'settings': settings_manager.get_all_settings()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Chat Interface Routes
@app.route('/')
@app.route('/chat')
def chat():
    """Chat interface page with memory search (default page)"""
    return render_template('chat_enhanced.html')

@app.route('/chat_basic')
def chat_basic():
    """Basic chat interface without memory search"""
    return render_template('chat.html')

@app.route('/api/memory/search', methods=['POST'])
def memory_search():
    """Search memories using the EXACT same process as the analyzer tab"""
    try:
        data = request.json
        query = data.get('query', '')
        weights = data.get('weights', current_weights)  # Use current_weights as default
        token_budget = data.get('token_budget')
        removed_ids = data.get('removed_ids', [])

        # If no token budget specified, use context window minus reserves
        if token_budget is None:
            token_budget = cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens - 512
        token_budget = max(512, token_budget)  # Ensure minimum budget

        # Get initial candidates - same as analyzer
        initial_candidates = data.get('initial_candidates', 999999)  # Default to effectively unlimited

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Decompose query - exactly like analyzer
        decomposition = retriever.decompose_query(query)

        # Create retrieval query - exactly like analyzer
        rq = RetrievalQuery(
            session_id='memory_search',
            text=decomposition.get('what', query),
            actor_hint=decomposition['who'].get('id') if decomposition.get('who') else None,
            temporal_hint=decomposition.get('when')
        )

        # Generate query embedding in the same format as analyzer
        what_text = decomposition.get('what', query)
        why_text = decomposition.get('why', 'search')
        how_text = decomposition.get('how', 'query')

        # Format the embedding text to match storage format - exactly like analyzer
        embed_text = f"WHAT: {what_text}\nWHY: {why_text}\nHOW: {how_text}\nRAW: {query}"
        qvec = embedder.encode([embed_text], normalize_embeddings=True)[0]

        # Search with custom weights - get many candidates for knapsack
        candidates = retriever.search_with_weights(rq, qvec, weights, topk_sem=initial_candidates, topk_lex=initial_candidates)

        # Apply knapsack algorithm to select memories within token budget
        selected_ids, tokens_used = greedy_knapsack(candidates, token_budget)

        # Filter candidates to only selected ones
        selected_candidates = [c for c in candidates if c.memory_id in selected_ids]

        # Apply exclusion list if provided (but NOT score_threshold or top_k - let client handle those)
        if removed_ids:
            excluded_set = set(removed_ids)
            selected_candidates = [c for c in selected_candidates if c.memory_id not in excluded_set]

        # Get detailed scores - exactly like analyzer
        detailed_scores = retriever.get_detailed_scores(selected_candidates)

        # Build response - matching analyzer format
        memories = []
        # Fetch all selected memories at once for efficiency
        if selected_candidates:
            memory_ids = [c.memory_id for c in selected_candidates]
            fetched_memories = store.fetch_memories(memory_ids)
            memory_lookup = {m['memory_id']: m for m in fetched_memories}

            for i, candidate in enumerate(selected_candidates):
                mem = memory_lookup.get(candidate.memory_id)
                if mem:
                    # Find detailed score for this memory
                    detail = next((d for d in detailed_scores if d['memory_id'] == candidate.memory_id), {})
                    memories.append({
                        'memory_id': candidate.memory_id,
                        'rank': i + 1,
                        'total_score': round(candidate.score, 3),
                        'token_count': candidate.token_count,
                        'scores': detail.get('scores', {}),
                        'when': mem.get('when_ts', mem.get('created_at', '')),
                        'who': f"{mem.get('who_type', '')}: {mem.get('who_id', '')}",
                        'what': mem.get('what', ''),
                        'where': f"{mem.get('where_type', '')}: {mem.get('where_value', '')}",
                        'why': mem.get('why', ''),
                        'how': mem.get('how', ''),
                        'raw_text': mem.get('raw_text', ''),
                        'entities': mem.get('entities', [])
                    })

        return jsonify({
            'memories': memories,
            'selected_count': len(memories),
            'total_candidates': len(candidates),
            'tokens_used': tokens_used,
            'token_budget': token_budget,
            'decomposition': decomposition
        })

    except Exception as e:
        import traceback
        print(f"Memory search error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/chat/completions', methods=['POST'])
def chat_completions():
    """Proxy chat completions to llama.cpp with memory logging"""
    try:
        data = request.json
        session_id = data.get('session_id', f'chat_{datetime.now().isoformat()}')
        log_memory = data.get('log_memory', True)
        use_multi_part = data.get('use_multi_part', True)
        
        # Extract messages
        messages = data.get('messages', [])

        # Check if pre-loaded memory context is provided
        memory_context = data.get('memory_context', None)

        # If memory context is provided, inject it as a system message
        if memory_context and len(memory_context) > 0:
            # Build memory context message from provided memories
            memory_content = "Based on your memory database, here are relevant past experiences:\n\n"

            for mem in memory_context[:10]:  # Limit to top 10 to avoid overwhelming context
                # Format each memory nicely
                memory_content += f"• [{mem.get('when', 'Unknown time')}] {mem.get('what', 'No description')}\n"
                if mem.get('why'):
                    memory_content += f"  Context: {mem['why'][:100]}...\n"
                if mem.get('who') and mem['who'] != 'Unknown':
                    memory_content += f"  Actor: {mem['who']}\n"

            # Create memory system message
            memory_msg = {
                'role': 'system',
                'content': memory_content
            }

            # Insert after initial system message or at beginning
            if messages and messages[0]['role'] == 'system':
                messages = [messages[0], memory_msg] + messages[1:]
            else:
                messages = [memory_msg] + messages
        
        # Prepare request for llama.cpp with all parameters
        llama_request = {
            'messages': messages,
            'temperature': data.get('temperature', 0.7),
            'max_tokens': data.get('max_tokens', 2048),
            'top_p': data.get('top_p', 0.9),
            'top_k': data.get('top_k', 40),
            'min_p': data.get('min_p', 0.05),
            'repetition_penalty': data.get('repetition_penalty', 1.1),
            'frequency_penalty': data.get('frequency_penalty', 0.0),
            'presence_penalty': data.get('presence_penalty', 0.0),
            'typical_p': data.get('typical_p', 1.0),
            'tfs_z': data.get('tfs_z', 1.0),
            'mirostat': data.get('mirostat', 0),
            'mirostat_tau': data.get('mirostat_tau', 5.0),
            'mirostat_eta': data.get('mirostat_eta', 0.1),
            'seed': data.get('seed', -1),
            'stop': data.get('stop', []),
            'grammar': data.get('grammar', None),
            'n_predict': data.get('n_predict', -1),
            'penalize_nl': data.get('penalize_nl', False),
            'stream': False
        }
        
        # Forward to llama.cpp server
        llama_url = f"{cfg.llm_base_url}/chat/completions"
        
        import requests
        response = requests.post(llama_url, json=llama_request, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Log messages as memories if enabled
        memory_ids = []
        if log_memory and messages:
            # Log only the last user message (the one just sent)
            if messages and messages[-1]['role'] == 'user':
                user_msg = messages[-1]
                user_event = RawEvent(
                    session_id=session_id,
                    event_type='user_message',
                    actor='user:chat',
                    content=user_msg['content'],
                    metadata={
                        'role': 'user',
                        'timestamp': datetime.now().isoformat()
                    }
                )
                user_memory_id = memory_router.ingest(
                    user_event, 
                    context_hint=f"User message in chat conversation session {session_id}. This is a direct conversation between a human user and an AI assistant.",
                    use_multi_part=use_multi_part
                )
                if user_memory_id:
                    memory_ids.extend(user_memory_id.split(','))
            
            # Log the assistant response
            if result.get('choices') and result['choices'][0].get('message'):
                assistant_msg = result['choices'][0]['message']['content']
                assistant_event = RawEvent(
                    session_id=session_id,
                    event_type='llm_message',
                    actor=f'llm:{cfg.llm_model}',
                    content=assistant_msg,
                    metadata={
                        'role': 'assistant',
                        'timestamp': datetime.now().isoformat()
                    }
                )
                assistant_memory_id = memory_router.ingest(
                    assistant_event,
                    context_hint=f"AI assistant response in chat conversation session {session_id}. This is a direct conversation between a human user and an AI assistant.",
                    use_multi_part=use_multi_part
                )
                if assistant_memory_id:
                    memory_ids.extend(assistant_memory_id.split(','))
        
        # Add memory IDs to response
        result['memory_ids'] = memory_ids
        
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error communicating with LLM server: {str(e)}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get list of chat sessions"""
    try:
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        query = """
            SELECT 
                session_id,
                MIN(created_at) as start_time,
                MAX(created_at) as last_message,
                COUNT(*) as message_count
            FROM memories
            WHERE session_id LIKE 'chat_%'
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            LIMIT 50
        """
        
        rows = con.execute(query).fetchall()
        con.close()
        
        sessions = []
        for row in rows:
            sessions.append({
                'session_id': row['session_id'],
                'start_time': row['start_time'],
                'last_message': row['last_message'],
                'message_count': row['message_count']
            })
        
        return jsonify(sessions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/session/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get chat history for a specific session"""
    try:
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        
        query = """
            SELECT 
                memory_id,
                raw_text,
                extra_json,
                created_at
            FROM memories
            WHERE session_id = ?
            ORDER BY created_at ASC
        """
        
        rows = con.execute(query, (session_id,)).fetchall()
        con.close()
        
        messages = []
        for row in rows:
            # Parse metadata from extra_json
            extra = json.loads(row['extra_json']) if row['extra_json'] else {}
            metadata = extra.get('metadata', {})
            
            role = metadata.get('role', 'user')
            content = row['raw_text']
            
            messages.append({
                'role': role,
                'content': content,
                'timestamp': row['created_at'],
                'memory_id': row['memory_id']
            })
        
        return jsonify({
            'session_id': session_id,
            'messages': messages,
            'message_count': len(messages)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session and its memories"""
    try:
        con = sqlite3.connect(cfg.db_path)
        cursor = con.cursor()
        
        # Get memory IDs for this session
        memory_ids = cursor.execute(
            "SELECT memory_id FROM memories WHERE session_id = ?",
            (session_id,)
        ).fetchall()
        
        # Delete from all tables
        cursor.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM mem_fts WHERE memory_id IN (SELECT memory_id FROM memories WHERE session_id = ?)", (session_id,))
        cursor.execute("DELETE FROM embeddings WHERE memory_id IN (SELECT memory_id FROM memories WHERE session_id = ?)", (session_id,))
        cursor.execute("DELETE FROM usage_stats WHERE memory_id IN (SELECT memory_id FROM memories WHERE session_id = ?)", (session_id,))
        
        deleted_count = cursor.rowcount
        con.commit()
        con.close()
        
        # Remove from FAISS index
        for (memory_id,) in memory_ids:
            try:
                faiss_index.remove(memory_id)
            except:
                pass
        
        if deleted_count > 0:
            faiss_index.save()
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/process-chain', methods=['POST'])
def process_conversation_chain():
    """Process an entire conversation chain into interconnected memories"""
    try:
        data = request.json
        session_id = data.get('session_id')
        messages = data.get('messages', [])
        process_type = data.get('process_type', 'deep_chain')
        extract_relationships = data.get('extract_relationships', True)
        use_multi_part = data.get('multi_part', True)
        
        if not messages:
            return jsonify({'error': 'No messages to process'}), 400
        
        memories_created = 0
        relationships_found = 0
        entities_extracted = set()
        
        # Build conversation context for better extraction
        conversation_context = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in messages
        ])
        
        # Process each message with enhanced context
        for i, msg in enumerate(messages):
            # Create context hints that include surrounding messages
            prev_msg = messages[i-1]['content'] if i > 0 else ""
            next_msg = messages[i+1]['content'] if i < len(messages)-1 else ""
            
            context_hint = f"""This is message {i+1} of {len(messages)} in a chat conversation.
Previous message: {prev_msg[:100]}...
Current role: {msg['role']}
Next message: {next_msg[:100]}...
Full conversation context for relationship extraction."""
            
            # Create RawEvent for the message
            event = RawEvent(
                session_id=f"{session_id}_chain",
                event_type='user_message' if msg['role'] == 'user' else 'llm_message',
                actor=f"{msg['role']}:chain_process",
                content=msg['content'],
                metadata={
                    'role': msg['role'],
                    'position': i,
                    'total_messages': len(messages),
                    'process_type': process_type,
                    'original_session': session_id,
                    'timestamp': msg.get('timestamp', datetime.now().isoformat())
                }
            )
            
            # Ingest with enhanced context
            memory_id = memory_router.ingest(
                event,
                context_hint=context_hint,
                use_multi_part=use_multi_part
            )
            
            if memory_id:
                memories_created += len(memory_id.split(','))
            
            # Extract entities from the message for tracking
            if extract_relationships:
                # Get the memory to extract entities
                con = sqlite3.connect(cfg.db_path)
                con.row_factory = sqlite3.Row
                row = con.execute(
                    "SELECT what FROM memories WHERE memory_id = ?",
                    (memory_id.split(',')[0] if ',' in memory_id else memory_id,)
                ).fetchone()
                con.close()
                
                if row and row['what']:
                    extracted = retriever.extract_entities_from_what(row['what'])
                    entities_extracted.update(extracted)
        
        # Identify relationships between messages (simplified)
        if extract_relationships and len(messages) > 1:
            # Count Q&A pairs, follow-ups, etc.
            for i in range(len(messages)-1):
                if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                    relationships_found += 1  # Q&A relationship
                elif messages[i]['role'] == messages[i+1]['role']:
                    relationships_found += 1  # Follow-up relationship
        
        return jsonify({
            'success': True,
            'memories_created': memories_created,
            'relationships_found': relationships_found,
            'entities_extracted': len(entities_extracted),
            'entities': list(entities_extracted)[:50],  # Return first 50 entities
            'session_id': f"{session_id}_chain"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Main entry point for the Flask server"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='JAM Memory Web Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()
