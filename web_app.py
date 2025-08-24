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
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Store user input as memory
        success, msg, event = memory_agent.remember(
            who="User",
            what=user_message,
            where="web_chat",
            why="User query",
            how="Chat interface",
            event_type="user_input"
        )
        
        # Get relevant memories for context
        relevant_memories = memory_agent.recall(
            what=user_message,
            k=5
        )
        
        # Build context from memories
        context = "Relevant memories:\n"
        for mem, score in relevant_memories:
            context += f"- [{score:.2f}] {mem.five_w1h.who}: {mem.five_w1h.what[:100]}\n"
        
        # Generate LLM response
        prompt = f"""

You are a helpful AI assistant with access to a memory system.

Context from memory:
{context}

User: {user_message}
            what=llm_response,
            where="web_chat",
            why=f"Response to: {user_message[:50]}",
            how="LLM generation",
            event_type="action"
        )
        
        return jsonify({
            'response': llm_response,
            'memories_used': len(relevant_memories)
        })
    """
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memories', methods=['GET'])
def get_memories():
    """Get all memories with block information"""
    try:
        stats = memory_agent.get_statistics()
        
        # Get raw memories
        raw_memories = []
        if hasattr(memory_agent.memory_store, 'raw_memories'):
            for event in memory_agent.memory_store.raw_memories:
                raw_memories.append({
                    'id': event.id,
                    'who': event.five_w1h.who,
                    'what': event.five_w1h.what,
                    'when': event.five_w1h.when,
                    'where': event.five_w1h.where,
                    'why': event.five_w1h.why,
                    'how': event.five_w1h.how,
                    'type': event.event_type.value,
                    'salience': event.salience,
                    'episode_id': event.episode_id
                })
        
        # Get processed memories with block info
        processed_memories = []
        if hasattr(memory_agent.memory_store, 'processed_memories'):
            for event in memory_agent.memory_store.processed_memories:
                # Find which blocks contain this event
                blocks = memory_agent.memory_store.block_manager.get_blocks_for_event(event.id)
                block_ids = [b.id for b in blocks]
                block_saliences = [b.salience for b in blocks]
                
                processed_memories.append({
                    'id': event.id,
                    'who': event.five_w1h.who,
                    'what': event.five_w1h.what,
                    'when': event.five_w1h.when,
                    'where': event.five_w1h.where,
                    'why': event.five_w1h.why,
                    'how': event.five_w1h.how,
                    'type': event.event_type.value,
                    'salience': event.salience,
                    'episode_id': event.episode_id,
                    'block_ids': block_ids,
                    'block_saliences': block_saliences
                })
        
        # Get memory blocks
        memory_blocks = []
        if hasattr(memory_agent.memory_store, 'block_manager'):
            for block_id, block in memory_agent.memory_store.block_manager.blocks.items():
                memory_blocks.append({
                    'id': block.id,
                    'type': block.block_type,
                    'event_count': len(block.events),
                    'event_ids': list(block.event_ids),
                    'salience': block.salience,
                    'coherence': block.coherence_score,
                    'link_count': len(block.links),
                    'aggregate_signature': block.aggregate_signature.to_dict() if block.aggregate_signature else None,
                    'created_at': block.created_at.isoformat() if block.created_at else None,
                    'updated_at': block.updated_at.isoformat() if block.updated_at else None
                })
        
        return jsonify({
            'raw': raw_memories,
            'processed': processed_memories,
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
                    'salience': event.salience
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
                'salience': event.salience
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
