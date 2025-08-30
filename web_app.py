"""
Enhanced Flask web application for [DSAM] Dual-Space Agentic Memory
Provides improved API endpoints for the new frontend
"""

from flask import Flask, render_template, request, jsonify, session, make_response
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
from models.merge_types import MergeType
from config import get_config
from llm.llm_interface import LLMInterface
from memory.dual_space_encoder import DualSpaceEncoder
from memory.multi_dimensional_merger import MultiDimensionalMerger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'dual-space-memory-system-2024'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Global instances
memory_agent = None
llm_interface = None
config = None
encoder = None
multi_merger = None

# Track analytics data
analytics_data = {
    'residual_history': defaultdict(list),
    'space_usage_history': []
}

def initialize_system():
    """Initialize the memory system and LLM"""
    global memory_agent, llm_interface, config, encoder, multi_merger
    
    config = get_config()
    memory_agent = MemoryAgent(config)
    llm_interface = LLMInterface(config.llm)
    encoder = DualSpaceEncoder()
    
    # Initialize multi-dimensional merger with ChromaDB
    multi_merger = MultiDimensionalMerger(
        chromadb_store=memory_agent.memory_store.chromadb if hasattr(memory_agent.memory_store, 'chromadb') else None
    )
    
    # Attach multi-merger to memory store for access in API endpoints
    memory_agent.memory_store.multi_merger = multi_merger
    
    # Load existing merge groups from ChromaDB
    try:
        multi_merger._load_existing_merges()
        logger.info(f"Loaded multi-dimensional merge groups: "
                   f"Actor: {len(multi_merger.merge_groups.get(MergeType.ACTOR, {}))}, "
                   f"Temporal: {len(multi_merger.merge_groups.get(MergeType.TEMPORAL, {}))}, "
                   f"Conceptual: {len(multi_merger.merge_groups.get(MergeType.CONCEPTUAL, {}))}, "
                   f"Spatial: {len(multi_merger.merge_groups.get(MergeType.SPATIAL, {}))}")
    except Exception as e:
        logger.warning(f"Could not load existing merge groups: {e}")
    
    logger.info("Enhanced system initialized successfully")

@app.route('/')
def index():
    """Main page with enhanced UI"""
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

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

        # Retrieve relevant memories with enhanced context
        relevant_memories = memory_agent.memory_store.retrieve_memories_with_context(
            {'what': user_message},
            k=5,
            context_format='detailed'
        )

        # Build context and track memory details
        context = ""
        memory_details = []
        if relevant_memories:
            context_lines = ["Relevant memories:"]
            for memory_obj, score, context_str in relevant_memories:
                # Add the generated context
                context_lines.append(f"\n[Score: {score:.2f}]")
                context_lines.append(context_str)
                
                # Collect memory details for frontend
                # Check if it's a merged event or regular event
                from models.merged_event import MergedEvent
                if isinstance(memory_obj, MergedEvent):
                    latest_state = memory_obj.get_latest_state()
                    memory_details.append({
                        'id': memory_obj.id,
                        'who': latest_state.get('who', ''),
                        'what': latest_state.get('what', ''),
                        'when': latest_state.get('when', ''),
                        'where': latest_state.get('where', ''),
                        'why': latest_state.get('why', ''),
                        'how': latest_state.get('how', ''),
                        'score': float(score),
                        'merge_count': memory_obj.merge_count,
                        'is_merged': True
                    })
                else:
                    memory_details.append({
                        'id': memory_obj.id,
                        'who': memory_obj.five_w1h.who or '',
                        'what': memory_obj.five_w1h.what or '',
                        'when': memory_obj.five_w1h.when or '',
                        'where': memory_obj.five_w1h.where or '',
                        'why': memory_obj.five_w1h.why or '',
                        'how': memory_obj.five_w1h.how or '',
                        'score': float(score),
                        'is_merged': False
                    })
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
            'memory_details': memory_details,
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
    """Get all memories with residual information and raw/merged views"""
    view_mode = request.args.get('view', 'merged')  # 'merged' or 'raw'
    
    try:
        stats = memory_agent.get_statistics()
        
        memories = []
        
        # Check for raw view mode
        if view_mode == 'raw':
            # ONLY get raw events from the raw_events collection
            memories = []
            
            try:
                # Get ONLY from raw_events collection
                raw_collection = memory_agent.memory_store.chromadb.client.get_collection('raw_events')
                raw_results = raw_collection.get(include=['metadatas', 'documents'])
                
                for i in range(len(raw_results['ids'])):
                    raw_id = raw_results['ids'][i]
                    metadata = raw_results['metadatas'][i]
                    
                    # Skip if this is not actually a raw event
                    if not raw_id.startswith('raw_'):
                        continue
                    
                    memories.append({
                        'id': raw_id,
                        'merged_id': metadata.get('merged_id', ''),
                        'type': 'raw',
                        'five_w1h': {
                            'who': metadata.get('who', ''),
                            'what': metadata.get('what', ''),
                            'when': metadata.get('when', ''),
                            'where': metadata.get('where', ''),
                            'why': metadata.get('why', ''),
                            'how': metadata.get('how', '')
                        },
                        'event_type': metadata.get('event_type', 'observation'),
                        'timestamp': metadata.get('timestamp', ''),
                        'episode_id': metadata.get('episode_id', ''),
                        'residual_norm': 0,  # Raw events don't have residuals
                        'euclidean_weight': float(metadata.get('euclidean_weight', 0.5)),
                        'hyperbolic_weight': float(metadata.get('hyperbolic_weight', 0.5))
                    })
            except Exception as e:
                logger.warning(f"Error loading raw events: {e}")
                memories = []
            
            # Get merge groups for information
            merge_groups = memory_agent.memory_store.get_merge_groups()
            
            return jsonify({
                'memories': memories,
                'merge_groups': merge_groups,
                'total_raw': len(memories),
                'total_merged': len(merge_groups),
                'total_events': stats.get('total_events', 0),
                'view_mode': view_mode
            })
        
        # Default merged view - get all events from ChromaDB
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
                
                # Check if this is a merged event
                is_merged = False
                merge_count = 1
                if event_id in memory_agent.memory_store.merged_events_cache:
                    merged = memory_agent.memory_store.merged_events_cache[event_id]
                    is_merged = True
                    merge_count = merged.merge_count
                elif f"merged_{event_id}" in memory_agent.memory_store.merged_events_cache:
                    merged = memory_agent.memory_store.merged_events_cache[f"merged_{event_id}"]
                    is_merged = True
                    merge_count = merged.merge_count
                
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
                    'residual_norm': float(residual_norm),
                    'euclidean_weight': float(metadata.get('euclidean_weight', 0.5)),
                    'hyperbolic_weight': float(metadata.get('hyperbolic_weight', 0.5)),
                    'is_merged': is_merged,
                    'merge_count': merge_count
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

@app.route('/api/memory/<memory_id>/merge-groups', methods=['GET'])
def get_memory_merge_groups(memory_id):
    """Get all merge groups that contain this memory (raw or merged)"""
    try:
        merge_groups = []
        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
        
        # Check standard deduplication merges
        standard_merges = memory_agent.memory_store.get_merge_groups()
        
        # Check if this is a merged event ID
        if memory_id in standard_merges:
            merge_groups.append({
                'type': 'standard',
                'id': memory_id,
                'name': 'Standard Deduplication',
                'merge_count': len(standard_merges[memory_id]),
                'raw_event_ids': list(standard_merges[memory_id])
            })
        
        # Check if this is a raw event in standard merges
        for merged_id, raw_ids in standard_merges.items():
            clean_id = memory_id.replace('raw_', '')
            if clean_id in raw_ids or f"raw_{clean_id}" in raw_ids:
                merge_groups.append({
                    'type': 'standard',
                    'id': merged_id,
                    'name': 'Standard Deduplication',
                    'merge_count': len(raw_ids),
                    'raw_event_ids': list(raw_ids)
                })
                break
        
        # Check multi-dimensional merges if available
        if multi_merger:
            from models.merge_types import MergeType
            
            # Clean the ID for comparison
            clean_id = memory_id.replace('raw_', '').replace('merged_', '')
            
            # Check each merge type
            for merge_type in [MergeType.ACTOR, MergeType.TEMPORAL, MergeType.CONCEPTUAL, MergeType.SPATIAL]:
                groups = multi_merger.merge_groups.get(merge_type, {})
                
                for group_id, group_data in groups.items():
                    merged_event = group_data.get('merged_event')
                    if merged_event:
                        # Check if our event is in this group
                        if clean_id in merged_event.raw_event_ids or memory_id in merged_event.raw_event_ids:
                            merge_groups.append({
                                'type': merge_type.value,
                                'id': group_id,
                                'name': merge_type.name.title(),
                                'key': group_data.get('key', 'Unknown'),
                                'merge_count': merged_event.merge_count,
                                'raw_event_ids': list(merged_event.raw_event_ids)
                            })
        
        return jsonify({
            'memory_id': memory_id,
            'merge_groups': merge_groups,
            'total_groups': len(merge_groups)
        })
    except Exception as e:
        logger.error(f"Error getting merge groups for memory: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<memory_id>/raw', methods=['GET'])
def get_memory_raw_events(memory_id):
    """Get all raw events for a merged memory"""
    try:
        raw_events = memory_agent.memory_store.get_raw_events_for_merged(memory_id)
        
        # If no raw events found in memory, try loading from ChromaDB directly
        if not raw_events:
            logger.info(f"No raw events in memory for {memory_id}, checking ChromaDB directly")
            try:
                raw_collection = memory_agent.memory_store.chromadb.client.get_collection('raw_events')
                # Get all raw events and filter by merged_id
                results = raw_collection.get(
                    where={"merged_id": memory_id},
                    include=['metadatas', 'documents']
                )
                
                raw_events = []
                for i, metadata in enumerate(results['metadatas']):
                    from models.event import Event, FiveW1H, EventType
                    event = Event(
                        id=metadata.get('original_id', results['ids'][i].replace('raw_', '')),
                        five_w1h=FiveW1H(
                            who=metadata.get('who', ''),
                            what=metadata.get('what', ''),
                            when=metadata.get('when', ''),
                            where=metadata.get('where', ''),
                            why=metadata.get('why', ''),
                            how=metadata.get('how', '')
                        ),
                        event_type=EventType(metadata.get('event_type', 'observation')),
                        episode_id=metadata.get('episode_id', '')
                    )
                    raw_events.append(event)
                logger.info(f"Loaded {len(raw_events)} raw events from ChromaDB for {memory_id}")
            except Exception as e:
                logger.error(f"Failed to load raw events from ChromaDB: {e}")
        
        formatted_raw = []
        for event in raw_events:
            five_w1h_dict = event.five_w1h.to_dict()
            # Log first event to debug
            if len(formatted_raw) == 0:
                logger.info(f"First raw event 5W1H: {five_w1h_dict}")
            
            formatted_raw.append({
                'id': f"raw_{event.id}",
                'five_w1h': five_w1h_dict,
                'event_type': event.event_type.value,
                'timestamp': event.created_at.isoformat() if hasattr(event, 'created_at') else event.timestamp,
                'episode_id': event.episode_id
            })
        
        return jsonify({
            'merged_id': memory_id,
            'raw_events': formatted_raw,
            'total_raw': len(formatted_raw)
        })
        
    except Exception as e:
        logger.error(f"Failed to get raw events: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/<memory_id>/merged-details', methods=['GET'])
def get_merged_event_details(memory_id):
    """Get detailed information about a merged event or create one for single memories"""
    try:
        from models.merged_event import MergedEvent
        
        merged_event = None
        
        # 1. Check in-memory cache first (for performance)
        if memory_id in memory_agent.memory_store.merged_events_cache:
            merged_event = memory_agent.memory_store.merged_events_cache[memory_id]
            logger.info(f"Found merged event {memory_id} in cache")
        
        # 2. If not in cache, load from ChromaDB (primary storage)
        if not merged_event:
            merged_event = memory_agent.memory_store.chromadb.get_merged_event(memory_id)
            if merged_event:
                # Update cache for future quick access
                memory_agent.memory_store.merged_events_cache[memory_id] = merged_event
                logger.info(f"Loaded merged event {memory_id} from ChromaDB")
        
        # 3. Check if this is a raw event that's part of a merged group
        if not merged_event and memory_id in memory_agent.memory_store.raw_to_merged:
            merged_id = memory_agent.memory_store.raw_to_merged[memory_id]
            # Check cache first
            if merged_id in memory_agent.memory_store.merged_events_cache:
                merged_event = memory_agent.memory_store.merged_events_cache[merged_id]
                logger.info(f"Found cached merged event {merged_id} for raw event {memory_id}")
            else:
                # Load from ChromaDB
                merged_event = memory_agent.memory_store.chromadb.get_merged_event(merged_id)
                if merged_event:
                    memory_agent.memory_store.merged_events_cache[merged_id] = merged_event
                    logger.info(f"Loaded merged event {merged_id} for raw event {memory_id}")
        
        # 4. If still no merged event, try to get the regular event and create a single-event merged view
        if not merged_event:
            # Try to get the event from ChromaDB
            event = memory_agent.memory_store.chromadb.get_event(memory_id)
            if event:
                # Create a merged event view for single event
                merged_event = MergedEvent(
                    id=f"single_{memory_id}",
                    base_event_id=memory_id
                )
                event_data = {
                    'who': event.five_w1h.who or '',
                    'what': event.five_w1h.what or '',
                    'when': event.five_w1h.when or '',
                    'where': event.five_w1h.where or '',
                    'why': event.five_w1h.why or '',
                    'how': event.five_w1h.how or '',
                    'timestamp': event.created_at
                }
                merged_event.add_raw_event(memory_id, event_data)
                logger.info(f"Created single-event merged view for {memory_id}")
            else:
                logger.info(f"No event found for {memory_id}")
                return jsonify({'error': 'Memory not found', 'memory_id': memory_id}), 404
        
        # Format the response with all component details
        response = {
            'id': merged_event.id,
            'base_event_id': merged_event.base_event_id,
            'merge_count': merged_event.merge_count,
            'created_at': merged_event.created_at.isoformat(),
            'last_updated': merged_event.last_updated.isoformat(),
            
            # Component variations
            'who_variants': {},
            'what_variants': {},
            'when_timeline': [],
            'where_locations': merged_event.where_locations,
            'why_variants': {},
            'how_methods': merged_event.how_methods,
            
            # Latest state
            'latest_state': merged_event.get_latest_state(),
            'dominant_pattern': merged_event.dominant_pattern,
            
            # Relationships
            'temporal_chain': merged_event.temporal_chain,
            'supersedes': merged_event.supersedes,
            'superseded_by': merged_event.superseded_by,
            'depends_on': list(merged_event.depends_on),
            'enables': list(merged_event.enables),
            
            # Raw events
            'raw_event_ids': list(merged_event.raw_event_ids)
        }
        
        # Format WHO variants
        for who, variants in merged_event.who_variants.items():
            response['who_variants'][who] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'event_id': v.event_id,
                    'relationship': v.relationship.value,
                    'version': v.version
                }
                for v in variants
            ]
        
        # Format WHAT variants
        for what, variants in merged_event.what_variants.items():
            response['what_variants'][what] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'event_id': v.event_id,
                    'relationship': v.relationship.value,
                    'version': v.version
                }
                for v in variants
            ]
        
        # Format WHEN timeline
        response['when_timeline'] = [
            {
                'timestamp': tp.timestamp.isoformat(),
                'semantic_time': tp.semantic_time,
                'event_id': tp.event_id,
                'description': tp.description
            }
            for tp in merged_event.when_timeline
        ]
        
        # Format WHY variants
        for why, variants in merged_event.why_variants.items():
            response['why_variants'][why] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'event_id': v.event_id,
                    'relationship': v.relationship.value,
                    'version': v.version
                }
                for v in variants
            ]
        
        # Generate context preview
        context_generator = memory_agent.memory_store.context_generator
        context_summary = context_generator.generate_context(merged_event, None, 'summary')
        context_detailed = context_generator.generate_context(merged_event, None, 'detailed')
        
        response['context_preview'] = {
            'summary': context_summary,
            'detailed': context_detailed
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Failed to get merged event details: {e}")
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

@app.route('/api/graph', methods=['POST', 'GET'])
def get_graph():
    """Get memory graph data with clustering"""
    # Check if system is initialized
    if memory_agent is None:
        logger.error("Memory agent not initialized when /api/graph was called")
        return jsonify({'error': 'System not initialized'}), 503
    
    if request.method == 'GET':
        # Return empty graph for GET requests (testing)
        return jsonify({
            'nodes': [],
            'edges': [],
            'cluster_count': 0,
            'space_weights': {'euclidean': 0.5, 'hyperbolic': 0.5}
        })
    
    try:
        data = request.json
        if data is None:
            logger.warning("No JSON data in request body")
            data = {}
    except Exception as e:
        logger.error(f"Error parsing JSON from request: {e}")
        return jsonify({'error': 'Invalid JSON in request body'}), 400
    
    components = data.get('components', ['who', 'what', 'when', 'where', 'why', 'how'])
    use_clustering = data.get('use_clustering', True)
    viz_mode = data.get('visualization_mode', 'dual')
    center_node_id = data.get('center_node', None)  # For individual node view
    similarity_threshold = data.get('similarity_threshold', 0.3)
    
    logger.info(f"Graph request - components: {components}, center_node: {center_node_id}")
    
    # Get HDBSCAN parameters from request
    min_cluster_size = data.get('min_cluster_size', config.dual_space.hdbscan_min_cluster_size)
    min_samples = data.get('min_samples', config.dual_space.hdbscan_min_samples)
    
    # Temporarily update the memory store's HDBSCAN parameters
    if use_clustering:
        memory_agent.memory_store.hdbscan_min_cluster_size = min_cluster_size
        memory_agent.memory_store.hdbscan_min_samples = min_samples
    
    try:
        center_metadata = None
        display_center_id = None
        raw_event_ids_to_include = []  # Initialize here so it's in the outer scope
        
        # If center_node specified, get related memories
        if center_node_id:
            # Get the center node's data
            try:
                center_metadata = None
                
                # Check if it's a merged event ID
                if center_node_id.startswith('merged_'):
                    # Look in merged events cache
                    if center_node_id in memory_agent.memory_store.merged_events_cache:
                        merged_event = memory_agent.memory_store.merged_events_cache[center_node_id]
                        latest_state = merged_event.get_latest_state()
                        center_metadata = {
                            'who': latest_state.get('who', ''),
                            'what': latest_state.get('what', ''),
                            'when': latest_state.get('when', ''),
                            'where': latest_state.get('where', ''),
                            'why': latest_state.get('why', ''),
                            'how': latest_state.get('how', '')
                        }
                        # Get all raw event IDs from this merged event
                        raw_event_ids_to_include = list(merged_event.raw_event_ids)
                        logger.info(f"Including {len(raw_event_ids_to_include)} raw events from merged group")
                        logger.info(f"First few raw IDs: {raw_event_ids_to_include[:3]}")
                        # Use the actual merged event ID for the graph
                        actual_center_id = center_node_id
                    else:
                        # Try to get from events collection (might be stored there)
                        actual_center_id = center_node_id.replace('merged_', '')
                else:
                    # Regular event ID - check both with and without raw_ prefix
                    actual_center_id = center_node_id
                    if not center_node_id.startswith('raw_'):
                        # Try with raw_ prefix first
                        collection = memory_agent.memory_store.chromadb.client.get_collection("raw_events")
                        test_result = collection.get(
                            ids=[f"raw_{center_node_id}"],
                            include=["metadatas"]
                        )
                        if test_result['ids']:
                            actual_center_id = f"raw_{center_node_id}"
                
                # If we haven't found metadata yet, try the collections
                if center_metadata is None:
                    # Try events collection first
                    try:
                        collection = memory_agent.memory_store.chromadb.client.get_collection("events")
                        center_result = collection.get(
                            ids=[actual_center_id],
                            include=["metadatas", "embeddings"]
                        )
                        if center_result['ids']:
                            center_metadata = center_result['metadatas'][0]
                    except:
                        pass
                    
                    # If not found, try raw_events collection
                    if center_metadata is None:
                        try:
                            collection = memory_agent.memory_store.chromadb.client.get_collection("raw_events")
                            center_result = collection.get(
                                ids=[actual_center_id],
                                include=["metadatas", "embeddings"]
                            )
                            if center_result['ids']:
                                center_metadata = center_result['metadatas'][0]
                        except:
                            pass
                
                if center_metadata is None:
                    logger.warning(f"Center node not found: {center_node_id}")
                    return jsonify({'error': 'Center node not found'}), 404
                
                # Build query from center node's fields
                query = {
                    'what': center_metadata.get('what', ''),
                    'who': center_metadata.get('who', ''),
                    'why': center_metadata.get('why', ''),
                    'how': center_metadata.get('how', '')
                }
                
                # Store the actual center ID for use in graph (but keep display_center_id for UI)
                # center_node_id = actual_center_id  # Don't overwrite, we need both
                
                # Get more memories for similarity comparison
                # If we have raw events from a merge group, we don't need many more
                k = 10 if raw_event_ids_to_include else 30  
            except Exception as e:
                logger.error(f"Error getting center node: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            # Build query from selected components for full graph
            query = {}
            for comp in components:
                query[comp] = ""  # Empty values to get all memories
            k = 50
        
        # Build query from selected components
        query_filtered = {k: v for k, v in query.items() if k in components}
        
        # Try to retrieve memories
        try:
            results = memory_agent.memory_store.retrieve_memories(
                query=query_filtered,
                k=k,  
                use_clustering=use_clustering,
                update_residuals=False
            )
        except Exception as e:
            logger.error(f"Error retrieving memories for graph: {e}")
            # Fallback: get raw memories directly
            results = []
            try:
                # Get some memories from the merged events cache
                for event_id, merged_event in list(memory_agent.memory_store.merged_events_cache.items())[:k]:
                    # Create a simple event object from merged event
                    latest_state = merged_event.get_latest_state()
                    class SimpleEvent:
                        def __init__(self, id, state):
                            self.id = id
                            self.five_w1h = type('FiveW1H', (), {
                                'who': state.get('who', ''),
                                'what': state.get('what', ''),
                                'when': state.get('when', ''),
                                'where': state.get('where', ''),
                                'why': state.get('why', ''),
                                'how': state.get('how', '')
                            })()
                    
                    event = SimpleEvent(event_id, latest_state)
                    results.append((event, 0.5))  # Default score
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                results = []
        
        # Build graph nodes
        nodes = []
        node_map = {}
        included_ids = set()
        
        # If center node specified, add it first
        if center_node_id and center_metadata:
            # Create a simple object to hold center node data
            class CenterEvent:
                def __init__(self, id, metadata):
                    self.id = id
                    self.five_w1h = type('FiveW1H', (), {})()
                    self.five_w1h.who = metadata.get('who', '')
                    self.five_w1h.what = metadata.get('what', '')
                    self.five_w1h.when = metadata.get('when', '')
                    self.five_w1h.where = metadata.get('where', '')
                    self.five_w1h.why = metadata.get('why', '')
                    self.five_w1h.how = metadata.get('how', '')
            
            # Use the original center_node_id for the graph display
            # (might be merged_ prefix which is what frontend expects)
            display_center_id = data.get('center_node', center_node_id)
            center_event = CenterEvent(display_center_id, center_metadata)
            
            # Add center node with highest score
            results_with_center = [(center_event, 1.0)]
            
            # If this is a merged event, add all its raw events
            if raw_event_ids_to_include:
                logger.info(f"Fetching {len(raw_event_ids_to_include)} raw events for merged node")
                logger.info(f"Raw IDs to fetch: {raw_event_ids_to_include[:5]}...")
                try:
                    # Get raw events from ChromaDB
                    raw_collection = memory_agent.memory_store.chromadb.client.get_collection("raw_events")
                    raw_results = raw_collection.get(
                        ids=raw_event_ids_to_include,
                        include=["metadatas"]
                    )
                    
                    logger.info(f"Retrieved {len(raw_results['ids'])} raw events from ChromaDB")
                    
                    for idx, raw_id in enumerate(raw_results['ids']):
                        if raw_id in raw_results['ids']:
                            metadata_idx = raw_results['ids'].index(raw_id)
                            metadata = raw_results['metadatas'][metadata_idx]
                            
                            # Create event object for raw event
                            class RawEvent:
                                def __init__(self, id, metadata):
                                    self.id = id
                                    self.five_w1h = type('FiveW1H', (), {})()
                                    self.five_w1h.who = metadata.get('who', '')
                                    self.five_w1h.what = metadata.get('what', '')
                                    self.five_w1h.when = metadata.get('when', '')
                                    self.five_w1h.where = metadata.get('where', '')
                                    self.five_w1h.why = metadata.get('why', '')
                                    self.five_w1h.how = metadata.get('how', '')
                            
                            raw_event = RawEvent(raw_id, metadata)
                            # Add with high score since they're part of the merge group
                            results_with_center.append((raw_event, 0.8))
                            included_ids.add(raw_id)
                    
                    logger.info(f"Added {len(raw_results['ids'])} raw events to graph")
                except Exception as e:
                    logger.error(f"Error fetching raw events: {e}")
            
            # Add other results, avoiding duplicates
            for event, score in results:
                if event.id not in included_ids and event.id != display_center_id and event.id != center_node_id:
                    results_with_center.append((event, score))
            
            results = results_with_center
            included_ids.add(display_center_id)
            
            logger.info(f"Added center node {display_center_id[:8]}... to results, total nodes: {len(results)}")
        
        for i, (event, score) in enumerate(results):
            # Don't filter when center node is specified - retrieve_memories already returns relevant nodes
            # The first 30 results are already the most relevant
            included_ids.add(event.id)
            
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
            
            # Create label safely
            what_text = event.five_w1h.what or ""
            if len(what_text) > 30:
                label_text = what_text[:30] + "..."
            else:
                label_text = what_text
            
            node = {
                'id': event.id,
                'label': f"{event.five_w1h.who or 'Unknown'}: {label_text}",
                'who': event.five_w1h.who or '',
                'what': event.five_w1h.what or '',
                'when': event.five_w1h.when or '',
                'where': event.five_w1h.where or '',
                'why': event.five_w1h.why or '',
                'how': event.five_w1h.how or '',
                'space': space,
                'cluster_id': -1,  # Will be set if clustering finds groups
                'centrality': score,
                'residual_norm': float(residual_norm),
                'is_center': event.id == display_center_id if center_node_id else False  # Mark center node
            }
            nodes.append(node)
            node_map[event.id] = i
        
        # Build edges based on cached similarity scores
        edges = []
        edge_threshold = 0.3 if not center_node_id else 0.25  # Lower threshold for focused view
        
        # Get space weights from query for similarity adjustment
        query = {comp: '' for comp in components}
        lambda_e, lambda_h = memory_agent.memory_store.encoder.compute_query_weights(query)
        
        # Use cached similarities for efficiency
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Skip if nodes not in included set
                if nodes[i]['id'] not in included_ids or nodes[j]['id'] not in included_ids:
                    continue
                
                # Try to get cached similarity first
                similarity = memory_agent.memory_store.chromadb.get_cached_similarity(
                    nodes[i]['id'], nodes[j]['id'], lambda_e, lambda_h
                )
                
                if similarity is None:
                    # Fallback: compute on-demand if not cached
                    if (nodes[i]['id'] in memory_agent.memory_store.embedding_cache and 
                        nodes[j]['id'] in memory_agent.memory_store.embedding_cache):
                        
                        emb1 = memory_agent.memory_store.embedding_cache[nodes[i]['id']]
                        emb2 = memory_agent.memory_store.embedding_cache[nodes[j]['id']]
                        
                        # Compute product distance (convert to similarity)
                        distance = memory_agent.memory_store.encoder.compute_product_distance(
                            emb1, emb2, lambda_e, lambda_h
                        )
                        similarity = 1.0 / (1.0 + distance)
                        
                        # Cache this similarity for future use
                        memory_agent.memory_store.chromadb.update_similarity_cache(
                            nodes[i]['id'], emb1
                        )
                        memory_agent.memory_store.chromadb.update_similarity_cache(
                            nodes[j]['id'], emb2
                        )
                    else:
                        # Last resort: simple text similarity
                        similarity = compute_node_similarity(nodes[i], nodes[j], components)
                
                if similarity and similarity > edge_threshold:  # Threshold for edge creation
                    edges.append({
                        'from': nodes[i]['id'],
                        'to': nodes[j]['id'],
                        'weight': float(similarity)
                    })
        
        # Add temporal edges for conversation flow (User -> Assistant pairs)
        # Sort nodes by episode and timestamp to find conversation pairs
        sorted_nodes = sorted(nodes, key=lambda n: (n.get('episode_id', ''), n.get('when', '')))
        
        for i in range(len(sorted_nodes) - 1):
            curr = sorted_nodes[i]
            next = sorted_nodes[i + 1]
            
            # Connect User -> Assistant or Assistant -> User in same episode
            if curr.get('episode_id') == next.get('episode_id'):
                if (curr.get('who') == 'User' and next.get('who') == 'Assistant') or \
                   (curr.get('who') == 'Assistant' and next.get('who') == 'User'):
                    # Add conversation flow edge with higher weight
                    edges.append({
                        'from': curr['id'],
                        'to': next['id'],
                        'weight': 0.8,  # Strong connection for conversation flow
                        'type': 'conversation'
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
        
        # Calculate overall space weights based on the actual graph content
        total_concrete_score = 0
        total_abstract_score = 0
        node_count = 0
        
        # Debug: log components being checked
        logger.debug(f"Components to check: {components}")
        
        for node in nodes:
            # Count nodes that have at least one field
            has_content = False
            node_concrete = 0
            node_abstract = 0
            
            # Weight contributions based on which components are selected and non-empty
            if node.get('who') and node['who'] and 'who' in components:
                node_concrete += 1.0
                has_content = True
            if node.get('what') and node['what'] and 'what' in components:
                node_concrete += 2.0  # What is most concrete
                has_content = True
            if node.get('when') and node['when'] and 'when' in components:
                node_concrete += 0.5
                has_content = True
            if node.get('where') and node['where'] and 'where' in components:
                node_concrete += 0.5
                has_content = True
            if node.get('why') and node['why'] and 'why' in components:
                node_abstract += 1.5  # Why is most abstract
                has_content = True
            if node.get('how') and node['how'] and 'how' in components:
                node_abstract += 1.0
                has_content = True
            
            total_concrete_score += node_concrete
            total_abstract_score += node_abstract
            
            if has_content:
                node_count += 1
        
        # Calculate average space weights
        if len(nodes) > 0:
            # Average across all nodes, not just those with content
            avg_concrete = total_concrete_score / len(nodes)
            avg_abstract = total_abstract_score / len(nodes)
            total = avg_concrete + avg_abstract
            
            logger.info(f"Space weight calculation: nodes={len(nodes)}, concrete_total={total_concrete_score:.2f}, abstract_total={total_abstract_score:.2f}")
            logger.info(f"Averages: concrete={avg_concrete:.3f}, abstract={avg_abstract:.3f}, total={total:.3f}")
            
            if total > 0:
                euclidean_weight = avg_concrete / total
                hyperbolic_weight = avg_abstract / total
                space_weights = {
                    'euclidean': euclidean_weight,
                    'hyperbolic': hyperbolic_weight
                }
                logger.info(f"Final weights: euclidean={euclidean_weight:.3f}, hyperbolic={hyperbolic_weight:.3f}")
            else:
                logger.warning("Total score is 0, using default 50/50 weights")
                space_weights = {'euclidean': 0.5, 'hyperbolic': 0.5}
        else:
            logger.warning("No nodes found, using default 50/50 weights")
            space_weights = {'euclidean': 0.5, 'hyperbolic': 0.5}
        
        return jsonify({
            'nodes': nodes,
            'edges': edges,
            'cluster_count': cluster_count,
            'space_weights': space_weights
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

@app.route('/api/merge-stats', methods=['GET'])
def get_merge_statistics():
    """Get merge statistics for the UI including multi-dimensional merges"""
    try:
        stats = memory_agent.memory_store.get_statistics()
        
        # Get standard merge groups
        standard_merge_groups = memory_agent.memory_store.get_merge_groups()
        
        # Collect all merge groups including multi-dimensional
        all_merge_groups = {}
        
        # Add standard deduplication merges
        all_merge_groups['standard'] = standard_merge_groups
        
        # Add multi-dimensional merges if available
        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
        if multi_merger:
            # Add each merge type
            for merge_type in [MergeType.ACTOR, MergeType.TEMPORAL, MergeType.CONCEPTUAL, MergeType.SPATIAL]:
                groups = multi_merger.merge_groups.get(merge_type, {})
                type_merges = {}
                
                for group_id, group_data in groups.items():
                    merged_event = group_data.get('merged_event')
                    if merged_event:
                        type_merges[group_id] = list(merged_event.raw_event_ids)
                
                if type_merges:
                    all_merge_groups[merge_type.value] = type_merges
        
        # Calculate distribution of merge sizes for standard merges
        size_distribution = {}
        for merged_id, raw_ids in standard_merge_groups.items():
            size = len(raw_ids)
            size_key = str(size) if size <= 5 else "6+"
            size_distribution[size_key] = size_distribution.get(size_key, 0) + 1
        
        # Calculate multi-dimensional statistics
        multi_stats = {}
        if multi_merger:
            for merge_type in [MergeType.ACTOR, MergeType.TEMPORAL, MergeType.CONCEPTUAL, MergeType.SPATIAL]:
                groups = multi_merger.merge_groups.get(merge_type, {})
                multi_stats[merge_type.value] = {
                    'group_count': len(groups),
                    'total_events': sum(len(g.get('events', [])) for g in groups.values())
                }
        
        return jsonify({
            'total_raw': stats.get('total_raw_events', 0),
            'total_merged': stats.get('total_merged_groups', 0),
            'average_merge_size': stats.get('average_merge_size', 0),
            'size_distribution': size_distribution,
            'merge_groups': standard_merge_groups,  # Keep for backward compatibility
            'all_merge_groups': all_merge_groups,  # New field with all merge types
            'multi_dimensional_stats': multi_stats  # Statistics for each merge type
        })
    except Exception as e:
        logger.error(f"Failed to get merge stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/similarity-cache-stats', methods=['GET'])
def get_similarity_cache_stats():
    """Get similarity cache statistics"""
    try:
        stats = memory_agent.memory_store.chromadb.get_similarity_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting similarity cache stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for charts"""
    try:
        # Get actual residual history if available
        residual_history = {
            'labels': [],
            'euclidean': [],
            'hyperbolic': []
        }
        
        # Collect real residual norms for existing memories
        if memory_agent.memory_store.residuals:
            for i, (event_id, residuals) in enumerate(list(memory_agent.memory_store.residuals.items())[:10]):
                residual_history['labels'].append(f"M{i}")
                residual_history['euclidean'].append(float(np.linalg.norm(residuals['euclidean'])))
                residual_history['hyperbolic'].append(float(np.linalg.norm(residuals['hyperbolic'])))
        else:
            # Generate sample data if no residuals yet
            time_points = 10
            residual_history = {
                'labels': [f"T{i}" for i in range(time_points)],
                'euclidean': [0.01 + 0.02 * np.exp(-i/3) + np.random.random() * 0.005 for i in range(time_points)],
                'hyperbolic': [0.05 + 0.1 * np.exp(-i/3) + np.random.random() * 0.01 for i in range(time_points)]
            }
        
        # Calculate collective space usage ratio across all memories
        total_memories = memory_agent.memory_store.total_events
        euclidean_weight_sum = 0.0
        hyperbolic_weight_sum = 0.0
        
        if total_memories > 0:
            try:
                # Get all memories from ChromaDB
                collection = memory_agent.memory_store.chromadb.client.get_collection("events")
                all_memories = collection.get(include=["metadatas"])
                
                logger.info(f"Analyzing {len(all_memories['metadatas'])} memories for space distribution")
                
                # Check if we have stored weight values
                has_weight_values = any('euclidean_weight' in m for m in all_memories['metadatas'])
                
                if has_weight_values:
                    # Use actual weight values for accurate collective ratio
                    for metadata in all_memories['metadatas']:
                        euclidean_weight = float(metadata.get('euclidean_weight', 0.5))
                        hyperbolic_weight = float(metadata.get('hyperbolic_weight', 0.5))
                        euclidean_weight_sum += euclidean_weight
                        hyperbolic_weight_sum += hyperbolic_weight
                else:
                    # Calculate based on field presence
                    for metadata in all_memories['metadatas']:
                        # Use the encoder's logic to compute weights for this memory
                        query_fields = {}
                        for field in ['who', 'what', 'where', 'why', 'how']:
                            value = metadata.get(field, '')
                            if value and value.strip():
                                query_fields[field] = value
                        
                        # Compute weights using the same logic as DualSpaceEncoder
                        concrete_score = 0
                        abstract_score = 0
                        
                        for field, text in query_fields.items():
                            if field in ['what', 'where', 'who']:
                                concrete_score += len(text.split())
                            elif field in ['why', 'how']:
                                abstract_score += len(text.split())
                        
                        # Calculate lambda values
                        total = concrete_score + abstract_score
                        if total > 0:
                            lambda_e = concrete_score / total
                            lambda_h = abstract_score / total
                        else:
                            lambda_e = lambda_h = 0.5
                        
                        # Apply smoothing (same as encoder)
                        lambda_e = 0.3 + 0.4 * lambda_e
                        lambda_h = 0.3 + 0.4 * lambda_h
                        
                        # Normalize to sum to 1
                        total = lambda_e + lambda_h
                        lambda_e = lambda_e / total
                        lambda_h = lambda_h / total
                        
                        euclidean_weight_sum += lambda_e
                        hyperbolic_weight_sum += lambda_h
                
                # Calculate percentages for the pie chart
                total_weight = euclidean_weight_sum + hyperbolic_weight_sum
                if total_weight > 0:
                    euclidean_percentage = (euclidean_weight_sum / total_weight) * 100
                    hyperbolic_percentage = (hyperbolic_weight_sum / total_weight) * 100
                else:
                    euclidean_percentage = 50
                    hyperbolic_percentage = 50
                
                logger.info(f"Collective space ratio: Euclidean={euclidean_percentage:.1f}%, Hyperbolic={hyperbolic_percentage:.1f}%")
                        
            except Exception as e:
                logger.warning(f"Could not analyze memories for space distribution: {e}")
                # Fall back to equal distribution if error
                euclidean_percentage = 50
                hyperbolic_percentage = 50
        else:
            euclidean_percentage = 50
            hyperbolic_percentage = 50
        
        # Return percentages for pie chart
        space_distribution = {
            'euclidean': euclidean_percentage,
            'hyperbolic': hyperbolic_percentage
        }
        
        # Add the same percentages as average lambda values for consistency
        space_distribution['avg_lambda_e'] = euclidean_percentage / 100.0
        space_distribution['avg_lambda_h'] = hyperbolic_percentage / 100.0
        
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

@app.route('/api/merge-dimensions', methods=['GET'])
def get_merge_dimensions():
    """Get available merge dimensions and their statistics"""
    try:
        dimensions = []
        
        for merge_type in MergeType:
            # Count events for this dimension from existing merged events
            count = 0
            total_events = 0
            merged_events = memory_agent.memory_store.merged_events_cache
            
            for merged_event in merged_events.values():
                should_include = False
                
                if merge_type == MergeType.TEMPORAL:
                    should_include = True
                elif merge_type == MergeType.ACTOR:
                    who_variants = set()
                    for variants in merged_event.who_variants.values():
                        for v in variants:
                            who_variants.add(v.value.lower())
                    should_include = len(who_variants) <= 2
                elif merge_type == MergeType.CONCEPTUAL:
                    should_include = merged_event.merge_count > 1
                elif merge_type == MergeType.SPATIAL:
                    where_locations = merged_event.where_locations
                    should_include = len(where_locations) > 0 and any(loc for loc in where_locations if loc)
                elif merge_type == MergeType.HYBRID:
                    should_include = False  # Skip hybrid for now
                
                if should_include:
                    count += 1
                    total_events += merged_event.merge_count
            
            dimension_info = {
                'type': merge_type.value,
                'name': merge_type.name.replace('_', ' ').title(),
                'description': get_merge_type_description(merge_type),
                'group_count': count,
                'total_events': total_events,
                'primary_field': get_primary_field(merge_type)
            }
            dimensions.append(dimension_info)
        
        return jsonify({
            'dimensions': dimensions,
            'active_dimension': request.args.get('active', 'temporal')
        })
    except Exception as e:
        logger.error(f"Error getting merge dimensions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/merge-types', methods=['GET'])
def get_merge_types():
    """Get available merge types and their statistics"""
    try:
        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
        
        merge_types = []
        
        # Add standard deduplication merges
        standard_groups = memory_agent.memory_store.get_merge_groups()
        merge_types.append({
            'type': 'standard',
            'name': 'Standard Deduplication',
            'description': 'Events merged based on similarity threshold',
            'group_count': len(standard_groups),
            'total_events': sum(len(ids) for ids in standard_groups.values())
        })
        
        if multi_merger:
            # Add multi-dimensional merge types
            from models.merge_types import MergeType
            
            for merge_type in [MergeType.ACTOR, MergeType.TEMPORAL, MergeType.CONCEPTUAL, MergeType.SPATIAL]:
                groups = multi_merger.merge_groups.get(merge_type, {})
                total_events = sum(len(g.get('events', [])) for g in groups.values())
                
                descriptions = {
                    MergeType.ACTOR: 'Events grouped by actor (who)',
                    MergeType.TEMPORAL: 'Events grouped by conversation threads',
                    MergeType.CONCEPTUAL: 'Events grouped by concepts/goals',
                    MergeType.SPATIAL: 'Events grouped by location'
                }
                
                merge_types.append({
                    'type': merge_type.value,
                    'name': merge_type.name.title(),
                    'description': descriptions.get(merge_type, ''),
                    'group_count': len(groups),
                    'total_events': total_events
                })
        
        return jsonify({
            'merge_types': merge_types,
            'total_types': len(merge_types)
        })
    except Exception as e:
        logger.error(f"Error getting merge types: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/merge-groups/<merge_type>', methods=['GET'])
def get_merge_groups(merge_type):
    """Get all merge groups for a specific dimension"""
    try:
        # Check if we have multi-dimensional merger
        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
        
        if not multi_merger:
            # Fallback to standard merge groups
            if merge_type == 'standard':
                merge_groups = memory_agent.memory_store.get_merge_groups()
                formatted_groups = []
                for merged_id, raw_ids in merge_groups.items():
                    formatted_groups.append({
                        'id': merged_id,
                        'type': 'standard',
                        'key': f"Merged ({len(raw_ids)} events)",
                        'merge_count': len(raw_ids),
                        'raw_event_ids': list(raw_ids)
                    })
                return jsonify({
                    'merge_type': merge_type,
                    'groups': formatted_groups,
                    'total': len(formatted_groups)
                })
            else:
                return jsonify({
                    'merge_type': merge_type,
                    'groups': [],
                    'total': 0,
                    'message': 'Multi-dimensional merger not initialized'
                })
        
        # Get the appropriate merge type enum
        from models.merge_types import MergeType
        merge_type_map = {
            'actor': MergeType.ACTOR,
            'temporal': MergeType.TEMPORAL,
            'conceptual': MergeType.CONCEPTUAL,
            'spatial': MergeType.SPATIAL,
            'standard': None  # For standard deduplication merges
        }
        
        merge_type_enum = merge_type_map.get(merge_type.lower())
        
        # Handle standard deduplication merges
        if merge_type == 'standard':
            merge_groups = memory_agent.memory_store.get_merge_groups()
            formatted_groups = []
            for merged_id, raw_ids in merge_groups.items():
                formatted_groups.append({
                    'id': merged_id,
                    'type': 'standard',
                    'key': f"Merged ({len(raw_ids)} events)",
                    'merge_count': len(raw_ids),
                    'raw_event_ids': list(raw_ids)
                })
        else:
            # Get multi-dimensional merge groups
            if merge_type_enum is None:
                return jsonify({'error': f'Invalid merge type: {merge_type}'}), 400
            
            groups = multi_merger.merge_groups.get(merge_type_enum, {})
            formatted_groups = []
            
            for group_id, group_data in groups.items():
                merged_event = group_data.get('merged_event')
                if merged_event:
                    latest_state = merged_event.get_latest_state()
                    formatted_groups.append({
                        'id': group_id,
                        'type': merge_type,
                        'key': group_data.get('key', 'Unknown'),
                        'merge_count': merged_event.merge_count,
                        'created_at': group_data.get('created_at', datetime.utcnow()).isoformat() if 'created_at' in group_data else '',
                        'last_updated': group_data.get('last_updated', datetime.utcnow()).isoformat() if 'last_updated' in group_data else '',
                        'latest_state': latest_state,
                        'raw_event_ids': list(merged_event.raw_event_ids),
                        'events_count': len(group_data.get('events', []))
                    })
        
        # Sort by merge count (largest groups first)
        formatted_groups.sort(key=lambda x: x.get('merge_count', 0), reverse=True)
        
        return jsonify({
            'merge_type': merge_type,
            'groups': formatted_groups,
            'total': len(formatted_groups)
        })
    except Exception as e:
        logger.error(f"Error getting merge groups: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/merge-group/<merge_type>/<group_id>', methods=['GET'])
def get_merge_group_detail(merge_type, group_id):
    """Get detailed information about a specific merge group"""
    try:
        # Parse merge type
        merge_type_enum = MergeType(merge_type)
        
        # Get the specific group
        groups = multi_merger.merge_groups.get(merge_type_enum, {})
        if group_id not in groups:
            return jsonify({'error': 'Merge group not found'}), 404
        
        group_data = groups[group_id]
        merged_event = group_data['merged_event']
        
        # Format the response similar to the original merged event endpoint
        response = {
            'id': group_id,
            'type': merge_type,
            'base_event_id': merged_event.base_event_id,
            'merge_count': merged_event.merge_count,
            'created_at': merged_event.created_at.isoformat(),
            'last_updated': merged_event.last_updated.isoformat(),
            
            # Component variations
            'who_variants': {},
            'what_variants': {},
            'when_timeline': [],
            'where_locations': merged_event.where_locations,
            'why_variants': {},
            'how_methods': merged_event.how_methods,
            
            # Latest state
            'latest_state': merged_event.get_latest_state(),
            'dominant_pattern': merged_event.dominant_pattern,
            
            # Relationships
            'temporal_chain': merged_event.temporal_chain,
            'supersedes': merged_event.supersedes,
            'superseded_by': merged_event.superseded_by,
            'depends_on': list(merged_event.depends_on),
            'enables': list(merged_event.enables),
            
            # Raw events
            'raw_event_ids': list(merged_event.raw_event_ids),
            'events': group_data['events']  # Include the actual event data
        }
        
        # Format WHO variants
        for who, variants in merged_event.who_variants.items():
            response['who_variants'][who] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'event_id': v.event_id,
                    'relationship': v.relationship.value,
                    'version': v.version
                }
                for v in variants
            ]
        
        # Format WHAT variants
        for what, variants in merged_event.what_variants.items():
            response['what_variants'][what] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'event_id': v.event_id,
                    'relationship': v.relationship.value,
                    'version': v.version
                }
                for v in variants
            ]
        
        # Format WHEN timeline
        response['when_timeline'] = [
            {
                'timestamp': tp.timestamp.isoformat(),
                'semantic_time': tp.semantic_time,
                'event_id': tp.event_id,
                'description': tp.description
            }
            for tp in merged_event.when_timeline
        ]
        
        # Format WHY variants
        for why, variants in merged_event.why_variants.items():
            response['why_variants'][why] = [
                {
                    'value': v.value,
                    'timestamp': v.timestamp.isoformat(),
                    'event_id': v.event_id,
                    'relationship': v.relationship.value,
                    'version': v.version
                }
                for v in variants
            ]
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting merge group detail: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/event-merges/<event_id>', methods=['GET'])
def get_event_merges(event_id):
    """Get all merge groups that contain a specific raw event"""
    try:
        merges = multi_merger.get_merges_for_event(event_id)
        
        response = {
            'event_id': event_id,
            'merge_dimensions': []
        }
        
        for merge_type, merged_event in merges.items():
            response['merge_dimensions'].append({
                'type': merge_type.value,
                'name': merge_type.name.replace('_', ' ').title(),
                'merge_id': merged_event.id,
                'merge_count': merged_event.merge_count,
                'latest_state': merged_event.get_latest_state()
            })
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting event merges: {e}")
        return jsonify({'error': str(e)}), 500

def get_merge_type_description(merge_type: MergeType) -> str:
    """Get a description for a merge type"""
    descriptions = {
        MergeType.ACTOR: "Groups all memories from the same person or actor",
        MergeType.TEMPORAL: "Groups conversation threads and sequential actions",
        MergeType.CONCEPTUAL: "Groups memories by goals, purposes, and abstract concepts",
        MergeType.SPATIAL: "Groups memories by location or spatial context"
    }
    return descriptions.get(merge_type, "")

def get_primary_field(merge_type: MergeType) -> str:
    """Get the primary 5W1H field for a merge type"""
    fields = {
        MergeType.ACTOR: "who",
        MergeType.TEMPORAL: "what",
        MergeType.CONCEPTUAL: "why",
        MergeType.SPATIAL: "where"
    }
    return fields.get(merge_type, "")

def preload_similarity_cache():
    """Pre-populate similarity cache on startup"""
    try:
        logger.info("Pre-loading similarity cache...")
        # This will be done automatically when memory_store initializes
        # through the _load_embeddings_to_cache() method
        stats = memory_agent.memory_store.chromadb.get_similarity_stats()
        logger.info(f"Similarity cache loaded: {stats['cached_pairs']} pairs, "
                   f"{stats['num_embeddings']} embeddings")
    except Exception as e:
        logger.warning(f"Could not preload similarity cache: {e}")

if __name__ == '__main__':
    initialize_system()
    # Pre-load similarity cache after system initialization
    preload_similarity_cache()
    app.run(debug=True, port=5000)
