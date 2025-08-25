"""
Flask web application for 5W1H Memory Framework
Provides chat interface and memory management GUI
"""

from flask import Flask, render_template, request, jsonify, session
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from agent.memory_agent import MemoryAgent
from models.event import EventType
from config import get_config
from llm.llm_interface import LLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Global instances
memory_agent = None
llm_interface = None
config = None

def initialize_system():
    """Initialize the memory system and LLM"""
    global memory_agent, llm_interface, config
    
    config = get_config()
    memory_agent = MemoryAgent(config)
    llm_interface = LLMInterface(config.llm)
    
    logger.info("System initialized successfully")

@app.route('/')
def index():
    """Main page with tabs"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # 1) Store user input as memory
        success, msg, user_event = memory_agent.remember(
            who="User",
            what=user_message,
            where="web_chat",
            why="User query",
            how="Chat interface",
            event_type="user_input"
        )

        # 2) Retrieve relevant memories for context
        relevant_memories = memory_agent.recall(
            what=user_message,
            k=5
        )

        # 3) Build an optional context section
        context = ""
        if relevant_memories:
            context_lines = ["Relevant memories:"]
            for mem, score in relevant_memories:
                snippet = mem.five_w1h.what[:100]
                ellipsis = "..." if len(mem.five_w1h.what) > 100 else ""
                context_lines.append(f"- [{score:.2f}] {mem.five_w1h.who}: {snippet}{ellipsis}")
            context = "\n".join(context_lines) + "\n\n"

        # 4) Compose the prompt and generate an LLM response
        prompt = (
            "You are a helpful AI assistant with access to conversation memory. "
            "Use any relevant memories (if provided) to make your response more context-aware and specific.\n\n"
            f"{context}"
            "Current conversation:\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

        llm_response = llm_interface.generate(prompt)

        # Fallback to something graceful if LLM returns empty
        if not llm_response:
            llm_response = "I'm having trouble generating a response right now, but I've saved your message. Could you rephrase or provide a bit more detail?"

        # 5) Store the assistant response as memory
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
            'memories_used': len(relevant_memories) if relevant_memories else 0
        })

    except Exception as e:
        # Log and surface the error cleanly
        app.logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/memories', methods=['GET'])
def get_memories():
    """Get all memories with block information"""
    try:
        stats = memory_agent.get_statistics()
        
        # Get all memories from ChromaDB
        raw_memories = []
        try:
            # Retrieve all events from ChromaDB
            all_events = memory_agent.memory_store.chromadb.retrieve_all_events()
            for event in all_events:
                raw_memories.append({
                    'id': event.id,
                    'who': event.five_w1h.who,
                    'what': event.five_w1h.what,
                    'when': event.five_w1h.when,
                    'where': event.five_w1h.where,
                    'why': event.five_w1h.why,
                    'how': event.five_w1h.how,
                    'type': event.event_type.value,
                    'salience': None,  # No individual salience in dynamic system
                    'episode_id': event.episode_id,
                    'confidence': event.confidence
                })
        except Exception as e:
            logger.warning(f"Could not retrieve events: {e}")
        
        # In the new dynamic system, show different useful views
        # Recent memories - last 10 events
        all_memories = raw_memories.copy()
        recent_memories = sorted(all_memories, key=lambda x: x.get('when', ''), reverse=True)[:10]
        
        # Get a sample of dynamic clusters by doing a test query
        memory_blocks = []
        try:
            # Perform a sample clustering to show how memories group
            sample_query = {"what": ""}  # Empty query to get general clusters
            sample_results = memory_agent.memory_store.retrieve_memories(
                query=sample_query, 
                k=10, 
                use_clustering=True
            )
            
            # Get cluster information from the last clustering operation
            if hasattr(memory_agent.memory_store, 'current_clusters'):
                for cluster_id, cluster in memory_agent.memory_store.current_clusters.items():
                    memory_blocks.append({
                        'id': cluster_id,
                        'type': 'dynamic',
                        'event_count': len(cluster.events) if hasattr(cluster, 'events') else 0,
                        'coherence': cluster.coherence if hasattr(cluster, 'coherence') else 0,
                        'relevance': cluster.relevance if hasattr(cluster, 'relevance') else 0,
                        'created_at': datetime.utcnow().isoformat()
                    })
        except Exception as e:
            logger.debug(f"Could not generate sample clusters: {e}")
        
        return jsonify({
            'raw': all_memories,  # All memories
            'processed': recent_memories,  # Recent memories (last 10)
            'blocks': memory_blocks,
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
                    'id': event.id,
                    'salience': 0.5  # Now handled by block salience matrix
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
        # Find and remove from raw memories
        if hasattr(memory_agent.memory_store, 'raw_memories'):
            memory_agent.memory_store.raw_memories = [
                m for m in memory_agent.memory_store.raw_memories 
                if m.id != memory_id
            ]
        
        # Find and remove from processed memories
        if hasattr(memory_agent.memory_store, 'processed_memories'):
            memory_agent.memory_store.processed_memories = [
                m for m in memory_agent.memory_store.processed_memories 
                if m.id != memory_id
            ]
        
        # Remove from Hopfield network if present
        if hasattr(memory_agent.memory_store.hopfield, 'memories'):
            memory_agent.memory_store.hopfield.memories = {
                k: v for k, v in memory_agent.memory_store.hopfield.memories.items()
                if k != memory_id
            }
        
        # Save changes
        memory_agent.save()
        
        return jsonify({'success': True, 'message': 'Memory deleted'})
        
    except Exception as e:
        logger.error(f"Delete memory error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_memories():
    """Query memories with dynamic clustering"""
    data = request.json
    
    try:
        # Extract query parameters
        query = {}
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            if field in data and data[field]:
                query[field] = data[field]
        
        k = data.get('k', 5)
        
        # Perform query with dynamic clustering
        results = memory_agent.recall(**query, k=k)
        
        # Format results
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
                'relevance': score,
                'episode_id': event.episode_id
            })
        
        # Get cluster info if available
        clusters = []
        if hasattr(memory_agent.memory_store, 'current_clusters'):
            for cluster_id, cluster in memory_agent.memory_store.current_clusters.items():
                clusters.append({
                    'id': cluster_id,
                    'coherence': getattr(cluster, 'coherence', 0),
                    'relevance': getattr(cluster, 'relevance', 0),
                    'size': len(getattr(cluster, 'events', []))
                })
        
        return jsonify({
            'success': True,
            'query': query,
            'results': memories,
            'clusters': clusters,
            'total': len(memories)
        })
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/memory/<memory_id>/cluster', methods=['GET'])
def get_memory_cluster(memory_id):
    """Get the dynamic cluster graph for a specific memory"""
    try:
        # Find the memory
        all_events = memory_agent.memory_store.chromadb.retrieve_all_events()
        target_event = None
        for event in all_events:
            if event.id == memory_id:
                target_event = event
                break
        
        if not target_event:
            return jsonify({'error': 'Memory not found'}), 404
        
        # Use the memory's content as a query to find related memories
        query = {
            'who': target_event.five_w1h.who,
            'what': target_event.five_w1h.what[:50],  # Use partial to avoid being too specific
        }
        
        # Get related memories with clustering
        results = memory_agent.recall(**query, k=15)
        
        # Build graph data
        nodes = []
        edges = []
        
        # Add the target memory as the central node
        nodes.append({
            'id': target_event.id,
            'label': f"{target_event.five_w1h.who}: {target_event.five_w1h.what[:30]}...",
            'group': 'target',
            'size': 30
        })
        
        # Add related memories as nodes
        for event, score in results:
            if event.id != target_event.id:
                nodes.append({
                    'id': event.id,
                    'label': f"{event.five_w1h.who}: {event.five_w1h.what[:30]}...",
                    'group': event.episode_id,  # Group by episode
                    'size': 20 * score  # Size based on relevance
                })
                
                # Add edge from target to related memory
                edges.append({
                    'from': target_event.id,
                    'to': event.id,
                    'value': score,
                    'label': f"{score:.2f}"
                })
        
        # Find connections between related memories
        for i, (event1, score1) in enumerate(results):
            for j, (event2, score2) in enumerate(results):
                if i < j and event1.episode_id == event2.episode_id:
                    # Same episode - add a connection
                    edges.append({
                        'from': event1.id,
                        'to': event2.id,
                        'value': 0.3,
                        'dashes': True  # Dashed line for episode connection
                    })
        
        return jsonify({
            'success': True,
            'nodes': nodes,
            'edges': edges,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Failed to get memory cluster: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blocks/<block_id>', methods=['DELETE'])
def delete_block(block_id):
    """Delete a memory block"""
    try:
        # Check if block exists
        if not hasattr(memory_agent.memory_store, 'memory_blocks'):
            return jsonify({'error': 'Memory blocks not initialized'}), 404
        
        blocks = memory_agent.memory_store.memory_blocks
        if block_id not in blocks:
            return jsonify({'error': 'Block not found'}), 404
        
        # Get the block to find its event IDs
        block = blocks[block_id]
        event_ids = block.event_ids if hasattr(block, 'event_ids') else []
        
        # Remove block from memory_blocks
        del blocks[block_id]
        
        # Remove block references from events in processed memories
        if hasattr(memory_agent.memory_store, 'processed_memories'):
            for event_id, event in memory_agent.memory_store.processed_memories.items():
                if hasattr(event, 'block_ids') and event.block_ids:
                    event.block_ids = [bid for bid in event.block_ids if bid != block_id]
                    if hasattr(event, 'block_saliences') and event.block_saliences:
                        # Remove corresponding salience if it exists
                        if len(event.block_saliences) > len(event.block_ids):
                            event.block_saliences = event.block_saliences[:len(event.block_ids)]
        
        # Remove block references from raw memories
        if hasattr(memory_agent.memory_store, 'raw_memories'):
            for event_id, event in memory_agent.memory_store.raw_memories.items():
                if hasattr(event, 'block_ids') and event.block_ids:
                    event.block_ids = [bid for bid in event.block_ids if bid != block_id]
                    if hasattr(event, 'block_saliences') and event.block_saliences:
                        if len(event.block_saliences) > len(event.block_ids):
                            event.block_saliences = event.block_saliences[:len(event.block_ids)]
        
        # Save changes
        memory_agent.save()
        
        return jsonify({'success': True, 'message': f'Block {block_id} deleted successfully'})
        
    except Exception as e:
        logger.error(f"Delete block error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recall', methods=['POST'])
def recall():
    """Recall memories based on query"""
    data = request.json
    
    try:
        results = memory_agent.recall(
            query=data.get('query', {}),
            k=data.get('k', 10)
        )
        
        memories = []
        for event, score in results:
            memories.append({
                'id': event.id,
                'score': score,
                'who': event.five_w1h.who,
                'what': event.five_w1h.what,
                'when': event.five_w1h.when,
                'where': event.five_w1h.where,
                'why': event.five_w1h.why,
                'how': event.five_w1h.how,
                'type': event.event_type.value,
                'salience': 0.5  # Now handled by block salience matrix
            })
        
        return jsonify({'memories': memories})
        
    except Exception as e:
        logger.error(f"Recall error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = memory_agent.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_system()
    app.run(debug=True, port=5000)
