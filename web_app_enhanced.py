"""
Enhanced Flask web application for Dual-Space Memory System
Provides improved API endpoints for the new frontend
"""

from flask import Flask, render_template, request, jsonify, session
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import logging
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from agent.memory_agent import MemoryAgent
from models.event import EventType, Event, FiveW1H
from config import get_config
from llm.llm_interface import LLMInterface
from memory.dual_space_encoder import DualSpaceEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'dual-space-memory-system-2024'

# Global instances
memory_agent = None
llm_interface = None
config = None
encoder = None

# Track analytics data
analytics_data = {
    'residual_history': defaultdict(list),
    'space_usage_history': []
}

def initialize_system():
    """Initialize the memory system and LLM"""
    global memory_agent, llm_interface, config, encoder
    
    config = get_config()
    memory_agent = MemoryAgent(config)
    llm_interface = LLMInterface(config.llm)
    encoder = DualSpaceEncoder()
    
    logger.info("Enhanced system initialized successfully")

@app.route('/')
def index():
    """Main page with enhanced UI"""
    return render_template('index.html')

@app.route('/test')
def test():
    """Test page"""
    with open('test_frontend.html', 'r') as f:
        return f.read()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with space weight calculation"""
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Store user input as memory
        success, msg, user_event = memory_agent.remember(
            who="User",
            what=user_message,
            where="web_chat",
            why="User query",
            how="Chat interface",
            event_type="user_input"
        )

        # Calculate query space weights
        query_fields = {'what': user_message}
        lambda_e, lambda_h = encoder.compute_query_weights(query_fields)

        # Retrieve relevant memories
        relevant_memories = memory_agent.recall(
            what=user_message,
            k=5
        )

        # Build context
        context = ""
        if relevant_memories:
            context_lines = ["Relevant memories:"]
            for mem, score in relevant_memories:
                snippet = mem.five_w1h.what[:100]
                ellipsis = "..." if len(mem.five_w1h.what) > 100 else ""
                context_lines.append(f"- [{score:.2f}] {mem.five_w1h.who}: {snippet}{ellipsis}")
            context = "\n".join(context_lines) + "\n\n"

        # Generate LLM response
        prompt = (
            "You are a helpful AI assistant with access to conversation history.\n"
            "Use the relevant memories below to inform your answer.\n"
            "Answer directly and concisely.\n\n"
            f"{context}"
            "Current question:\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

        llm_response = llm_interface.generate(prompt)

        if not llm_response:
            llm_response = "I'm processing your request. Could you provide more details?"

        # Store assistant response
        memory_agent.remember(
            who="Assistant",
            what=llm_response,
            where="web_chat",
            why=f"Response to: {user_message[:50]}",
            how="LLM generation",
            event_type="action"
        )

        return jsonify({
            'response': llm_response,
            'memories_used': len(relevant_memories),
            'space_weights': {
                'euclidean': float(lambda_e),
                'hyperbolic': float(lambda_h)
            }
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories', methods=['GET'])
def get_memories():
    """Get all memories with residual information"""
    try:
        stats = memory_agent.get_statistics()
        
        memories = []
        
        # Get all events from ChromaDB
        try:
            collection = memory_agent.memory_store.chromadb.client.get_collection("events")
            results = collection.get(include=["metadatas"])
            
            for i, metadata in enumerate(results['metadatas']):
                event_id = results['ids'][i] if 'ids' in results else str(uuid.uuid4())
                
                # Check if this memory has residuals
                has_residual = event_id in memory_agent.memory_store.residuals
                residual_norm = 0.0
                
                if has_residual:
                    eu_norm = np.linalg.norm(memory_agent.memory_store.residuals[event_id]['euclidean'])
                    hy_norm = np.linalg.norm(memory_agent.memory_store.residuals[event_id]['hyperbolic'])
                    residual_norm = (eu_norm + hy_norm) / 2
                
                memories.append({
                    'id': event_id,
                    'who': metadata.get('who', ''),
                    'what': metadata.get('what', ''),
                    'when': metadata.get('when', ''),
                    'where': metadata.get('where', ''),
                    'why': metadata.get('why', ''),
                    'how': metadata.get('how', ''),
                    'type': metadata.get('event_type', 'observation'),
                    'episode_id': metadata.get('episode_id', ''),
                    'has_residual': has_residual,
                    'residual_norm': float(residual_norm)
                })
        except Exception as e:
            logger.warning(f"Could not retrieve events: {e}")
        
        return jsonify({
            'memories': memories,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Get memories error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories', methods=['POST'])
def create_memory():
    """Create a new memory"""
    data = request.json
    
    try:
        success, msg, event = memory_agent.remember(
            who=data.get('who', 'User'),
            what=data.get('what', ''),
            where=data.get('where', 'web_interface'),
            why=data.get('why', 'Manual entry'),
            how=data.get('how', 'Direct input'),
            event_type=data.get('type', 'observation'),
            tags=data.get('tags', [])
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': msg,
                'memory': {
                    'id': event.id
                }
            })
        else:
            return jsonify({'success': False, 'message': msg}), 400
            
    except Exception as e:
        logger.error(f"Create memory error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories/<memory_id>', methods=['DELETE'])
def delete_memory(memory_id):
    """Delete a memory"""
    try:
        # Delete from ChromaDB
        collection = memory_agent.memory_store.chromadb.client.get_collection("events")
        collection.delete(ids=[memory_id])
        
        # Remove from residuals and momentum
        if memory_id in memory_agent.memory_store.residuals:
            del memory_agent.memory_store.residuals[memory_id]
        if memory_id in memory_agent.memory_store.momentum:
            del memory_agent.memory_store.momentum[memory_id]
        if memory_id in memory_agent.memory_store.embedding_cache:
            del memory_agent.memory_store.embedding_cache[memory_id]
        
        # Update total count
        memory_agent.memory_store.total_events = max(0, memory_agent.memory_store.total_events - 1)
        
        return jsonify({'success': True, 'message': 'Memory deleted successfully'})
        
    except Exception as e:
        logger.error(f"Delete memory error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph', methods=['POST'])
def get_graph():
    """Get memory graph data with clustering"""
    data = request.json
    components = data.get('components', ['who', 'what', 'when', 'where', 'why', 'how'])
    use_clustering = data.get('use_clustering', True)
    viz_mode = data.get('visualization_mode', 'dual')
    
    try:
        # Build query from selected components
        query = {}
        for comp in components:
            query[comp] = ""  # Empty values to get all memories
        
        # Retrieve memories with clustering
        results = memory_agent.memory_store.retrieve_memories(
            query=query,
            k=50,  # Get more for graph
            use_clustering=use_clustering,
            update_residuals=False
        )
        
        # Build graph nodes
        nodes = []
        node_map = {}
        
        for i, (event, score) in enumerate(results):
            # Determine dominant space
            concrete_fields = sum([1 for f in ['who', 'what', 'where'] if getattr(event.five_w1h, f)])
            abstract_fields = sum([1 for f in ['why', 'how'] if getattr(event.five_w1h, f)])
            
            if concrete_fields > abstract_fields:
                space = 'euclidean'
            elif abstract_fields > concrete_fields:
                space = 'hyperbolic'
            else:
                space = 'balanced'
            
            # Get residual norm if available
            residual_norm = 0.0
            if event.id in memory_agent.memory_store.residuals:
                eu_norm = np.linalg.norm(memory_agent.memory_store.residuals[event.id]['euclidean'])
                hy_norm = np.linalg.norm(memory_agent.memory_store.residuals[event.id]['hyperbolic'])
                residual_norm = (eu_norm + hy_norm) / 2
            
            node = {
                'id': event.id,
                'label': f"{event.five_w1h.who}: {event.five_w1h.what[:30]}...",
                'who': event.five_w1h.who,
                'what': event.five_w1h.what,
                'when': event.five_w1h.when,
                'where': event.five_w1h.where,
                'why': event.five_w1h.why,
                'how': event.five_w1h.how,
                'space': space,
                'cluster_id': -1,  # Will be set if clustering finds groups
                'centrality': score,
                'residual_norm': float(residual_norm)
            }
            nodes.append(node)
            node_map[event.id] = i
        
        # Build edges based on similarity
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Compute similarity between nodes
                similarity = compute_node_similarity(nodes[i], nodes[j], components)
                if similarity > 0.3:  # Threshold for edge creation
                    edges.append({
                        'from': nodes[i]['id'],
                        'to': nodes[j]['id'],
                        'weight': float(similarity)
                    })
        
        # Cluster assignment (simplified)
        if use_clustering and len(nodes) > 3:
            # Simple clustering based on similarity groups
            clusters = assign_clusters(nodes, edges)
            for node_id, cluster_id in clusters.items():
                for node in nodes:
                    if node['id'] == node_id:
                        node['cluster_id'] = cluster_id
                        break
        
        # Count clusters
        cluster_ids = set(node['cluster_id'] for node in nodes)
        cluster_count = len([c for c in cluster_ids if c >= 0])
        
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'cluster_count': cluster_count
        })
        
    except Exception as e:
        logger.error(f"Graph generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_memories():
    """Search memories with query"""
    data = request.json
    query_text = data.get('query', '')
    
    try:
        # Parse query to determine which fields to search
        query = {'what': query_text}  # Default to searching 'what' field
        
        results = memory_agent.recall(**query, k=20)
        
        memories = []
        for event, score in results:
            memories.append({
                'id': event.id,
                'who': event.five_w1h.who,
                'what': event.five_w1h.what,
                'when': event.five_w1h.when,
                'where': event.five_w1h.where,
                'why': event.five_w1h.why,
                'how': event.five_w1h.how,
                'type': event.event_type.value,
                'score': float(score)
            })
        
        return jsonify({'results': memories})
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get enhanced system statistics"""
    try:
        stats = memory_agent.get_statistics()
        
        # Add residual norm information
        if 'average_residual_norm' not in stats:
            stats['average_residual_norm'] = {
                'euclidean': 0.0,
                'hyperbolic': 0.0
            }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for charts"""
    try:
        # Generate sample data for charts
        # In production, this would track actual history
        
        # Residual evolution (mock data showing convergence)
        time_points = 10
        residual_history = {
            'labels': [f"T{i}" for i in range(time_points)],
            'euclidean': [0.01 + 0.02 * np.exp(-i/3) + np.random.random() * 0.005 for i in range(time_points)],
            'hyperbolic': [0.05 + 0.1 * np.exp(-i/3) + np.random.random() * 0.01 for i in range(time_points)]
        }
        
        # Space distribution
        total_memories = memory_agent.memory_store.total_events
        if total_memories > 0:
            # Estimate based on residual distribution
            euclidean_count = sum(1 for r in memory_agent.memory_store.residuals.values() 
                                 if np.linalg.norm(r['euclidean']) > np.linalg.norm(r['hyperbolic']))
            hyperbolic_count = sum(1 for r in memory_agent.memory_store.residuals.values() 
                                  if np.linalg.norm(r['hyperbolic']) > np.linalg.norm(r['euclidean']))
            balanced_count = total_memories - euclidean_count - hyperbolic_count
        else:
            euclidean_count = hyperbolic_count = balanced_count = 0
        
        space_distribution = {
            'euclidean': euclidean_count,
            'hyperbolic': hyperbolic_count,
            'balanced': balanced_count
        }
        
        return jsonify({
            'residual_history': residual_history,
            'space_distribution': space_distribution
        })
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

# Helper functions
def compute_node_similarity(node1, node2, components):
    """Compute similarity between two nodes based on selected components"""
    similarity = 0.0
    count = 0
    
    for comp in components:
        val1 = node1.get(comp, '')
        val2 = node2.get(comp, '')
        
        if val1 and val2:
            # Simple text similarity (in production, use embeddings)
            common_words = set(val1.lower().split()) & set(val2.lower().split())
            all_words = set(val1.lower().split()) | set(val2.lower().split())
            if all_words:
                similarity += len(common_words) / len(all_words)
                count += 1
    
    return similarity / max(1, count)

def assign_clusters(nodes, edges):
    """Simple clustering based on connected components"""
    clusters = {}
    cluster_id = 0
    
    # Build adjacency list
    adj = defaultdict(list)
    for edge in edges:
        if edge['weight'] > 0.5:  # Strong connections only
            adj[edge['from']].append(edge['to'])
            adj[edge['to']].append(edge['from'])
    
    # Find connected components
    visited = set()
    for node in nodes:
        if node['id'] not in visited:
            # BFS to find component
            component = []
            queue = [node['id']]
            
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    queue.extend(adj[current])
            
            # Assign cluster ID
            if len(component) > 1:
                for node_id in component:
                    clusters[node_id] = cluster_id
                cluster_id += 1
            else:
                clusters[component[0]] = -1  # Singleton
    
    return clusters

if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=5000)