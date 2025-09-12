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

# Load persisted weights or use defaults
current_weights = settings_manager.get_weights()

@app.route('/')
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
        
        # For initial candidate retrieval, get more than we need
        initial_top_k = data.get('initial_candidates', 500)
        
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
        
        # Get query embedding
        qvec = embedder.encode([query], normalize_embeddings=True)[0]
        
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
                who_type, who_id, who_label, 
                what, when_ts, 
                where_type, where_value,
                why, how, 
                raw_text, token_count, 
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
                who_type, who_id, who_label, 
                what, when_ts, 
                where_type, where_value,
                why, how, 
                raw_text, token_count, 
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
            entities = retriever.extract_entities_from_what(row['what'])
            for entity in entities:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
                if entity not in entity_memories:
                    entity_memories[entity] = []
                entity_memories[entity].append(row['memory_id'])
        
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
        
        query = """
            SELECT 
                DATE(when_ts) as date,
                COUNT(*) as count,
                COUNT(DISTINCT session_id) as sessions,
                COUNT(DISTINCT who_id) as actors
            FROM memories
            WHERE when_ts >= datetime('now', '-' || ? || ' days')
            GROUP BY DATE(when_ts)
            ORDER BY date
        """
        
        rows = con.execute(query, (days,)).fetchall()
        con.close()
        
        timeline = []
        for row in rows:
            timeline.append({
                'date': row['date'],
                'memories': row['count'],
                'sessions': row['sessions'],
                'actors': row['actors']
            })
        
        return jsonify({
            'success': True,
            'timeline': timeline,
            'days': days
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
            items = []
            data = row[column]
            
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
                    item_memories[item].append(row['memory_id'])
        
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)