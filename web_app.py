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
from memory.dimension_attention_retriever import DimensionAttentionRetriever

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
dimension_retriever = None

# Track analytics data
analytics_data = {
    'residual_history': defaultdict(list),
    'space_usage_history': []
}

<<<<<<< HEAD
def generate_llm_text_block(context, merge_type, merge_id):
    """
    Generate the exact text block that will be sent to the LLM.
    This is used for both display in the UI and actual LLM queries.
    """
    llm_text_lines = []
    llm_text_lines.append("=" * 60)
    llm_text_lines.append(f"{merge_type.upper()} MEMORY GROUP - {merge_id}")
    llm_text_lines.append("=" * 60)
    llm_text_lines.append("")
    
    # Summary section
    if 'narrative_summary' in context:
        llm_text_lines.append("SUMMARY:")
        llm_text_lines.append(context['narrative_summary'])
        llm_text_lines.append("")
    
    # Key Information section
    if context.get('key_information'):
        llm_text_lines.append("KEY INFORMATION:")
        for key, value in context['key_information'].items():
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value[:10]) if value else 'None'
            else:
                value_str = str(value)
            llm_text_lines.append(f"  - {key.replace('_', ' ').title()}: {value_str}")
        llm_text_lines.append("")
    
    # Patterns section
    if context.get('patterns'):
        llm_text_lines.append("PATTERNS IDENTIFIED:")
        for pattern in context['patterns']:
            llm_text_lines.append(f"  • {pattern}")
        llm_text_lines.append("")
    
    # Relationships section
    if context.get('relationships'):
        llm_text_lines.append("RELATIONSHIPS:")
        for relationship in context['relationships']:
            llm_text_lines.append(f"  • {relationship}")
        llm_text_lines.append("")
    
    # Timeline/Events section - show ALL events in chronological order
    if context.get('timeline'):
        # Sort events chronologically (oldest first)
        sorted_timeline = sorted(context['timeline'], key=lambda x: x.get('components', {}).get('when', '') or '')
        
        llm_text_lines.append(f"TIMELINE ({len(sorted_timeline)} events):")
        for i, event in enumerate(sorted_timeline, 1):  # Show ALL events
            components = event.get('components', {})
            llm_text_lines.append(f"\n  Event {i}:")
            llm_text_lines.append(f"    Who: {components.get('who', 'Unknown')}")
            llm_text_lines.append(f"    What: {components.get('what', '')}")
            when_str = components.get('when', '')
            # Add UTC label if timestamp looks like ISO format
            if when_str and ('T' in when_str or 'Z' in when_str):
                llm_text_lines.append(f"    When: {when_str} (UTC)")
            else:
                llm_text_lines.append(f"    When: {when_str}")
            llm_text_lines.append(f"    Where: {components.get('where', 'unspecified')}")
            llm_text_lines.append(f"    Why: {components.get('why', 'unspecified')}")
            llm_text_lines.append(f"    How: {components.get('how', 'unspecified')}")
    
    llm_text_lines.append("")
    llm_text_lines.append("=" * 60)
    
    return '\n'.join(llm_text_lines)

=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
def generate_enhanced_llm_context(merged_event, merge_type, merge_id, response_data):
    """Generate enhanced context for LLM consumption"""
    logger.info(f"Generating LLM context - event_count: {response_data.get('event_count')}, merge_count: {response_data.get('merge_count')}")
    logger.info(f"What variants keys: {list(response_data.get('what_variants', {}).keys())}")
<<<<<<< HEAD
    logger.info(f"Raw events count: {len(response_data.get('raw_events', []))}")
=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
    
    context = {
        'key_information': {},
        'timeline': [],
        'patterns': [],
        'relationships': []
    }
    
    # Extract key actors, concepts, and locations from variants
    actors = set()
    concepts = set()
    locations = set()
    methods = set()
    
    # Extract from WHO variants
    for who_key, variants in response_data.get('who_variants', {}).items():
        if isinstance(variants, list):
            for v in variants:
                actors.add(v.get('value', who_key))
        else:
            actors.add(who_key)
    
    # Extract from WHERE locations
    for where_key, variants in response_data.get('where_locations', {}).items():
        if isinstance(variants, list):
            for v in variants:
                locations.add(v.get('value', where_key))
        else:
            locations.add(where_key)
    
    # Extract from WHY variants
    for why_key, variants in response_data.get('why_variants', {}).items():
        if isinstance(variants, list):
            for v in variants:
                concepts.add(v.get('value', why_key))
        else:
            concepts.add(why_key)
    
    # Extract from HOW variants
    for how_key, variants in response_data.get('how_variants', {}).items():
        if isinstance(variants, list):
            for v in variants:
                methods.add(v.get('value', how_key))
        else:
            methods.add(how_key)
    
    # Also add from latest state
    latest = response_data.get('latest_state', {})
    if latest.get('who'):
        actors.add(latest['who'])
    if latest.get('where'):
        locations.add(latest['where'])
    if latest.get('why'):
        concepts.add(latest['why'])
    if latest.get('how'):
        methods.add(latest['how'])
    
    # Get event count from response data or what_variants
    event_count = response_data.get('event_count', 0)
    if event_count == 0:
        # Count from what_variants if event_count is 0
        for variants in response_data.get('what_variants', {}).values():
            if isinstance(variants, list):
                event_count += len(variants)
            else:
                event_count += 1
    
    # Build key information based on merge type
    if merge_type == 'actor':
        context['key_information'] = {
            'primary_focus': f"Actor-based memory group tracking interactions",
            'main_actors': list(actors),
            'interaction_count': event_count,
            'locations': list(locations),
            'key_concepts': list(concepts)[:5],  # Top 5 concepts
            'methods_used': list(methods)[:5]
        }
    elif merge_type == 'temporal':
        context['key_information'] = {
            'primary_focus': f"Temporal conversation thread",
            'thread_length': event_count,
            'participants': list(actors),
            'main_topics': list(concepts)[:5],
            'locations': list(locations)
        }
    elif merge_type == 'conceptual':
        context['key_information'] = {
            'primary_focus': f"Conceptual memory group",
            'core_concepts': list(concepts),
            'related_events': event_count,
            'involved_actors': list(actors),
            'implementation_methods': list(methods),
            'relevant_locations': list(locations)
        }
    elif merge_type == 'spatial':
        context['key_information'] = {
            'primary_focus': f"Location-based memory group",
            'primary_location': list(locations)[0] if locations else "Unknown",
            'events_at_location': event_count,
            'actors_present': list(actors),
            'activities': list(concepts)[:5],
            'methods': list(methods)[:5]
        }
    
<<<<<<< HEAD
    # Build timeline from all available sources
    what_events = []
    
    # The Timeline tab successfully uses variants, so we should too!
    # Don't bother with raw_events since they're not loading from ChromaDB
    # Just use the variants directly like the Timeline does
    
    raw_events = []  # We'll build from variants instead
    
    # Check if we have raw_events from the response (contains all timeline events with complete 5W1H)
    if 'raw_events' in response_data and response_data['raw_events']:
        raw_events = response_data['raw_events']
        logger.info(f"Using {len(raw_events)} raw events for timeline")
        # Log the actors found in raw events
        actors_in_raw = set()
        for event in raw_events:
            # Handle both new nested structure and old flat structure
            five_w1h = event.get('five_w1h', {})
            who = five_w1h.get('who') or event.get('who', 'Unknown')
            what = five_w1h.get('what') or event.get('what', '')
            actors_in_raw.add(who)
            logger.info(f"Raw event: who={who}, what={what[:50]}...")
        logger.info(f"Actors found in raw_events: {actors_in_raw}")
        
        # Use raw_events directly as they have complete 5W1H for each event
        for event in raw_events:
            # Handle both new nested structure and old flat structure
            five_w1h = event.get('five_w1h', {})
            what_events.append({
                'what': five_w1h.get('what') or event.get('what', ''),
                'timestamp': five_w1h.get('when') or event.get('when', ''),  # Use WHEN field for timestamp
                'event_id': event.get('id', ''),
                'who': five_w1h.get('who') or event.get('who', 'Unknown'),
                'when': five_w1h.get('when') or event.get('when', ''),
                'where': five_w1h.get('where') or event.get('where', 'unspecified'),
                'why': five_w1h.get('why') or event.get('why', 'unspecified'),
                'how': five_w1h.get('how') or event.get('how', 'unspecified')
            })
    else:
        logger.info("No raw_events found, falling back to variants")
        # Fall back to building from variants if raw_events not available
        # First collect all component information by event_id
        event_who_map = {}  # Map event_id to who value
        event_when_map = {}  # Map event_id to when value (the actual WHEN field)
        event_where_map = {}  # Map event_id to where value
        event_why_map = {}  # Map event_id to why value
        event_how_map = {}  # Map event_id to how value
        
        # Collect WHO
        for who_key, variants in response_data.get('who_variants', {}).items():
            if isinstance(variants, list):
                for v in variants:
                    event_id = v.get('event_id', '')
                    if event_id:
                        event_who_map[event_id] = v.get('value', who_key)
        
        # Collect WHEN from when_timeline or when_variants
        when_data = response_data.get('when_timeline', [])
        if when_data:
            for tp in when_data:
                event_id = tp.get('event_id', '')
                if event_id:
                    # Use semantic_time if available, otherwise timestamp
                    when_value = tp.get('semantic_time') or tp.get('timestamp', '')
                    event_when_map[event_id] = when_value
        
        # Also check when_variants if available
        when_variants = response_data.get('when_variants', {})
        if isinstance(when_variants, dict):
            for when_key, variants in when_variants.items():
                if isinstance(variants, list):
                    for v in variants:
                        event_id = v.get('event_id', '')
                        if event_id and event_id not in event_when_map:
                            event_when_map[event_id] = v.get('value', when_key)
        
        # Collect WHERE
        for where_key, variants in response_data.get('where_locations', {}).items():
            if isinstance(variants, list):
                for v in variants:
                    event_id = v.get('event_id', '')
                    if event_id:
                        event_where_map[event_id] = v.get('value', where_key)
        
        # Collect WHY
        for why_key, variants in response_data.get('why_variants', {}).items():
            if isinstance(variants, list):
                for v in variants:
                    event_id = v.get('event_id', '')
                    if event_id:
                        event_why_map[event_id] = v.get('value', why_key)
        
        # Collect HOW
        for how_key, variants in response_data.get('how_variants', {}).items():
            if isinstance(variants, list):
                for v in variants:
                    event_id = v.get('event_id', '')
                    if event_id:
                        event_how_map[event_id] = v.get('value', how_key)
        
        # Now build timeline with complete 5W1H values
        for what_key, variants in response_data.get('what_variants', {}).items():
            if isinstance(variants, list):
                for v in variants:
                    event_id = v.get('event_id', '')
                    # Use WHEN value for timestamp, fallback to internal timestamp if not available
                    when_value = event_when_map.get(event_id, v.get('timestamp', ''))
                    what_events.append({
                        'what': v.get('value', what_key),
                        'timestamp': when_value,  # Use WHEN field value
                        'event_id': event_id,
                        'who': event_who_map.get(event_id, latest.get('who', 'Unknown')),
                        'when': when_value,  # Store WHEN separately too
                        'where': event_where_map.get(event_id, latest.get('where', 'unspecified')),
                        'why': event_why_map.get(event_id, latest.get('why', 'unspecified')),
                        'how': event_how_map.get(event_id, latest.get('how', 'unspecified'))
                    })
=======
    # Build timeline from what_variants
    what_events = []
    for what_key, variants in response_data.get('what_variants', {}).items():
        if isinstance(variants, list):
            for v in variants:
                what_events.append({
                    'what': v.get('value', what_key),
                    'timestamp': v.get('timestamp', ''),
                    'event_id': v.get('event_id', '')
                })
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
    
    # Sort by timestamp (newest first)
    what_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
<<<<<<< HEAD
    # Log what we're about to add to timeline
    logger.info(f"Building timeline from {len(what_events)} what_events")
    actors_in_timeline = set()
    for event in what_events[:5]:  # Log first 5
        who = event.get('who', 'Unknown')
        actors_in_timeline.add(who)
        logger.info(f"Timeline event: who={who}, what={event.get('what', '')[:50]}...")
    logger.info(f"Actors in timeline: {actors_in_timeline}")
    
    # Format timeline with properly ordered 5W1H using actual event data - include ALL events
    for idx, event in enumerate(what_events):  # Include ALL events, no limit
        timeline_entry = []
        
        # Build entry using actual 5W1H values from event
        timeline_entry.append(f"Who: {event.get('who', 'Unknown')}")
        timeline_entry.append(f"What: {event['what'][:100]}...")  # Truncate long entries
        when_str = event.get('when', event.get('timestamp', ''))
        # Add UTC label if timestamp looks like ISO format
        if when_str and ('T' in when_str or 'Z' in when_str):
            timeline_entry.append(f"When: {when_str} (UTC)")
        else:
            timeline_entry.append(f"When: {when_str}")
        timeline_entry.append(f"Where: {event.get('where', 'unspecified')}")
        timeline_entry.append(f"Why: {event.get('why', 'unspecified')}")
        timeline_entry.append(f"How: {event.get('how', 'unspecified')}")
=======
    # Format timeline with properly ordered 5W1H using latest state and what_variants
    for idx, event in enumerate(what_events[:10]):  # Limit to 10 most recent
        timeline_entry = []
        
        # Build entry using latest state as template and specific event data
        timeline_entry.append(f"Who: {latest.get('who', 'User')}")
        timeline_entry.append(f"What: {event['what']}...")  # Truncate long entries
        timeline_entry.append(f"When: {event['timestamp']}")
        timeline_entry.append(f"Where: {latest.get('where', 'unspecified')}")
        timeline_entry.append(f"Why: {latest.get('why', 'unspecified')}")
        timeline_entry.append(f"How: {latest.get('how', 'unspecified')}")
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        
        context['timeline'].append({
            'event_id': event.get('event_id', ''),
            'formatted': ' | '.join(timeline_entry),
            'components': {
<<<<<<< HEAD
                'who': event.get('who', 'Unknown'),
                'what': event['what'],
                'when': event.get('when', event.get('timestamp', '')),  # Use WHEN field
                'where': event.get('where', 'unspecified'),
                'why': event.get('why', 'unspecified'),
                'how': event.get('how', 'unspecified')
=======
                'who': latest.get('who', 'User'),
                'what': event['what'],
                'when': event['timestamp'],
                'where': latest.get('where', ''),
                'why': latest.get('why', ''),
                'how': latest.get('how', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            }
        })
    
    # Identify patterns and relationships
    if len(what_events) > 1:
        # Check for recurring patterns
        what_counts = {}
        for event in what_events:
            what = event.get('what', '')
            if what:
                # Simplify by taking first 50 chars for comparison
                what_key = what[:50]
                what_counts[what_key] = what_counts.get(what_key, 0) + 1
        
        # Find repeated activities
        repeated = [what for what, count in what_counts.items() if count > 1]
        if repeated:
            context['patterns'].append(f"Repeated activities: {', '.join(repeated[:3])}")
        
        # Actor relationships
        if len(actors) > 1:
            context['relationships'].append(f"Multi-actor interaction between: {', '.join(list(actors)[:3])}")
        
        # Temporal patterns
        if merge_type == 'temporal' and what_events:
            context['patterns'].append(f"Conversation thread with {len(what_events)} exchanges")
    
    # Generate narrative summary
    summary_parts = []
    summary_parts.append(f"This {merge_type} memory group contains {event_count} related events.")
    
    if actors:
        summary_parts.append(f"Primary actors involved: {', '.join(list(actors)[:3])}.")
    
    if concepts:
        summary_parts.append(f"Key concepts: {', '.join(list(concepts)[:3])}.")
    
    if locations and list(locations)[0] != 'unspecified':
        summary_parts.append(f"Locations: {', '.join(list(locations)[:2])}.")
    
    context['narrative_summary'] = ' '.join(summary_parts)
    
<<<<<<< HEAD
    # Generate the exact LLM text block (this will be used for both display AND actual LLM queries)
    context['llm_text_block'] = generate_llm_text_block(context, merge_type, merge_id)
    
=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
    return context

def initialize_system():
    """Initialize the memory system and LLM"""
    global memory_agent, llm_interface, config, encoder, multi_merger, dimension_retriever
    
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
    
    # Initialize dimension attention retriever
    dimension_retriever = DimensionAttentionRetriever(
        encoder=encoder,
        multi_merger=multi_merger,
        similarity_cache=memory_agent.memory_store.similarity_cache if hasattr(memory_agent.memory_store, 'similarity_cache') else None,
<<<<<<< HEAD
        chromadb_store=memory_agent.memory_store.chromadb if hasattr(memory_agent.memory_store, 'chromadb') else None,
        temporal_manager=memory_agent.memory_store.temporal_manager if hasattr(memory_agent.memory_store, 'temporal_manager') else None
=======
        chromadb_store=memory_agent.memory_store.chromadb if hasattr(memory_agent.memory_store, 'chromadb') else None
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
    )
    
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
    """Handle chat messages with multi-dimensional attention-based retrieval"""
    data = request.json
    user_message = data.get('message', '').strip()
    use_dimension_attention = data.get('use_dimension_attention', True)  # Allow toggling

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
<<<<<<< HEAD
        # Calculate query space weights BEFORE storing the query
=======
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
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        query_fields = {'what': user_message, 'who': 'User'}
        lambda_e, lambda_h = encoder.compute_query_weights(query_fields)

        # Choose retrieval method based on flag
        dimension_weights = None
        dominant_dimension = None
        
        if use_dimension_attention and dimension_retriever:
            # NEW: Use dimension-aware retrieval
            # Get top 2 memory groups based on attention scores
<<<<<<< HEAD
            # NOTE: This happens BEFORE storing the current query to avoid retrieving it
            dimension_results, dimension_weights = dimension_retriever.retrieve_with_dimension_attention(
                query_fields,
                k=2  # Get top 2 memory groups
=======
            dimension_results, dimension_weights = dimension_retriever.retrieve_with_dimension_attention(
                query_fields,
                k=2  # Only get top 2 memory groups
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            )
            
            # Convert dimension results to memory context
            relevant_memories = dimension_retriever.get_context_from_dimension_results(
                dimension_results,
                memory_agent.memory_store.chromadb
            )
            
            # Find dominant dimension
            dominant_dimension = max(dimension_weights, key=dimension_weights.get) if dimension_weights else None
        else:
            # Fallback to standard retrieval (also limit to 2 for consistency)
            relevant_memories = memory_agent.memory_store.retrieve_memories_with_context(
                {'what': user_message},
                k=2,  # Limit to 2 memory groups
                context_format='detailed'
            )

        # Build context and track memory details
<<<<<<< HEAD
        # NOTE: We do NOT store the user query yet - wait until after response generation
        context = ""
        memory_details = []
        if relevant_memories:
            # Build context using our exact LLM text block format
            context_lines = []
            
            for i, (memory_obj, score, context_str) in enumerate(relevant_memories, 1):
                # For tracing: Use the EXACT format from generate_llm_text_block if available
                # The context_str should already be in our exact format from the retrieval
                # But we'll ensure consistency by regenerating if needed
                
                # Try to get merge type and ID from memory_obj
                merge_type = 'unknown'
                merge_id = memory_obj.id if hasattr(memory_obj, 'id') else f'group_{i}'
                
                # If context_str starts with our delimiter, it's already in the right format
                if context_str and context_str.startswith("=" * 60):
                    # Use the exact formatted text block
                    context_lines.append(context_str)
                else:
                    # Fallback: add basic formatting
                    context_lines.append("=" * 60)
                    context_lines.append(f"MEMORY GROUP {i} (Relevance: {score:.2f})")
                    context_lines.append("=" * 60)
                    context_lines.append(context_str)
                    context_lines.append("=" * 60)
                
                # Add spacing between groups
                if i < len(relevant_memories):
                    context_lines.append("\n")
                
                # Collect memory details for frontend
=======
        context = ""
        memory_details = []
        if relevant_memories:
            # For memory groups, present them clearly to the LLM
            context_lines = ["Retrieved Memory Groups (choose the most relevant):"]
            
            for i, (memory_obj, score, context_str) in enumerate(relevant_memories, 1):
                # Add the memory group context with clear labeling
                context_lines.append(f"\n--- Memory Group {i} (Relevance: {score:.2f}) ---")
                context_lines.append(context_str)
                
                # Collect memory details for frontend
                # The memory_obj is a synthetic Event from merge group metadata
                # context_str contains the actual context provided to LLM
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                memory_details.append({
                    'id': memory_obj.id if hasattr(memory_obj, 'id') else '',
                    'who': memory_obj.five_w1h.who if hasattr(memory_obj, 'five_w1h') and memory_obj.five_w1h.who else '',
                    'what': memory_obj.five_w1h.what if hasattr(memory_obj, 'five_w1h') and memory_obj.five_w1h.what else '',
                    'when': memory_obj.five_w1h.when if hasattr(memory_obj, 'five_w1h') and memory_obj.five_w1h.when else '',
                    'where': memory_obj.five_w1h.where if hasattr(memory_obj, 'five_w1h') and memory_obj.five_w1h.where else '',
                    'why': memory_obj.five_w1h.why if hasattr(memory_obj, 'five_w1h') and memory_obj.five_w1h.why else '',
                    'how': memory_obj.five_w1h.how if hasattr(memory_obj, 'five_w1h') and memory_obj.five_w1h.how else '',
                    'score': float(score),
                    'context': context_str,  # Include the actual context shown to LLM
                    'is_merged': True  # These are all from merge groups now
                })
            context = "\n".join(context_lines) + "\n\n"

        # Generate LLM response with current timestamp
        from datetime import datetime
        current_time = datetime.utcnow().isoformat() + 'Z'
        
        prompt = (
            "You are a helpful AI assistant with access to conversation history stored in memory groups.\n"
            f"Current time: {current_time}\n"
            "Use the memory groups below to understand context and answer the user's question.\n"
            "The memory groups show consolidated information from multiple related events.\n"
            "Each group's timeframe shows when those events occurred.\n"
            "Answer based on the information provided in the memory groups.\n"
            "Be direct and concise.\n\n"
            f"{context}"
            "\nCurrent question:\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

        llm_response = llm_interface.generate(prompt)

        if not llm_response:
            llm_response = "I'm processing your request. Could you provide more details?"

<<<<<<< HEAD
        # NOW store BOTH user query and assistant response (after generation to avoid self-retrieval)
        # Store user input first
        memory_agent.remember(
            who="User",
            what=user_message,
            where="web_chat",
            why="User query",
            how="Chat interface",
            event_type="user_input"
        )
        
        # Then store assistant response
=======
        # Store assistant response
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        memory_agent.remember(
            who="Assistant",
            what=llm_response,
            where="web_chat",
            why=f"Response to: {user_message[:50]}",
            how="LLM generation",
            event_type="action"
        )

<<<<<<< HEAD
        # Build response with dimension information and full LLM prompt for tracing
=======
        # Build response with dimension information
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        response_data = {
            'response': llm_response,
            'memories_used': len(relevant_memories),
            'memory_details': memory_details,
<<<<<<< HEAD
            'memory_context': context,  # The actual memory group context sent to LLM
            'llm_prompt': prompt,  # The EXACT prompt sent to the LLM for tracing
=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            'space_weights': {
                'euclidean': float(lambda_e),
                'hyperbolic': float(lambda_h)
            }
        }
        
        # Add dimension weights if using attention-based retrieval
        if dimension_weights:
            response_data['dimension_weights'] = {
                dim.value: float(weight) for dim, weight in dimension_weights.items()
            }
            response_data['dominant_dimension'] = dominant_dimension.value if dominant_dimension else None
        
        return jsonify(response_data)

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
                    
<<<<<<< HEAD
                    # Include all events from raw_events collection
=======
                    # Skip if this is not actually a raw event
                    if not raw_id.startswith('raw_'):
                        continue
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                    
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
<<<<<<< HEAD
                merged_event = None
                if event_id in memory_agent.memory_store.merged_events_cache:
                    merged_event = memory_agent.memory_store.merged_events_cache[event_id]
                    is_merged = True
                    merge_count = merged_event.merge_count
                elif f"merged_{event_id}" in memory_agent.memory_store.merged_events_cache:
                    merged_event = memory_agent.memory_store.merged_events_cache[f"merged_{event_id}"]
                    is_merged = True
                    merge_count = merged_event.merge_count
                
                # Get the latest state for merged events to get the latest WHEN
                if is_merged and merged_event:
                    latest_state = merged_event.get_latest_state()
                    when_value = latest_state.get('when', metadata.get('when', ''))
                    who_value = latest_state.get('who', metadata.get('who', ''))
                    what_value = latest_state.get('what', metadata.get('what', ''))
                    where_value = latest_state.get('where', metadata.get('where', ''))
                    why_value = latest_state.get('why', metadata.get('why', ''))
                    how_value = latest_state.get('how', metadata.get('how', ''))
                else:
                    # For non-merged events, use metadata directly
                    when_value = metadata.get('when', '')
                    who_value = metadata.get('who', '')
                    what_value = metadata.get('what', '')
                    where_value = metadata.get('where', '')
                    why_value = metadata.get('why', '')
                    how_value = metadata.get('how', '')
                
                memories.append({
                    'id': event_id,
                    'who': who_value,
                    'what': what_value,
                    'when': when_value,  # Now uses latest state for merged events
                    'where': where_value,
                    'why': why_value,
                    'how': how_value,
=======
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
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
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
<<<<<<< HEAD
            # Check if this memory is in the raw IDs
            if memory_id in raw_ids:
=======
            clean_id = memory_id.replace('raw_', '')
            if clean_id in raw_ids or f"raw_{clean_id}" in raw_ids:
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
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
            
<<<<<<< HEAD
=======
            # Clean the ID for comparison
            clean_id = memory_id.replace('raw_', '').replace('merged_', '')
            
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            # Check each merge type
            for merge_type in [MergeType.ACTOR, MergeType.TEMPORAL, MergeType.CONCEPTUAL, MergeType.SPATIAL]:
                groups = multi_merger.merge_groups.get(merge_type, {})
                
                for group_id, group_data in groups.items():
                    merged_event = group_data.get('merged_event')
                    if merged_event:
                        # Check if our event is in this group
<<<<<<< HEAD
                        if memory_id in merged_event.raw_event_ids:
=======
                        if clean_id in merged_event.raw_event_ids or memory_id in merged_event.raw_event_ids:
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
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

@app.route('/api/raw-event/<event_id>/merge-groups', methods=['GET'])
def get_raw_event_merge_groups(event_id):
    """Get all multi-dimensional merge groups a raw event belongs to"""
    try:
        # Get merge groups from the multi-dimensional merger
        if hasattr(memory_agent.memory_store, 'multi_merger') and memory_agent.memory_store.multi_merger:
            merge_groups = memory_agent.memory_store.multi_merger.get_merge_groups_details_for_event(event_id)
            
<<<<<<< HEAD
            # Add standard merge group if exists
            merged_id = None
            if event_id in memory_agent.memory_store.raw_to_merged:
                merged_id = memory_agent.memory_store.raw_to_merged[event_id]
=======
            # Add standard merge group if exists (for backwards compatibility)
            merged_id = None
            if event_id.startswith('raw_'):
                base_id = event_id.replace('raw_', '')
                if base_id in memory_agent.memory_store.raw_to_merged:
                    merged_id = memory_agent.memory_store.raw_to_merged[base_id]
                elif event_id in memory_agent.memory_store.raw_to_merged:
                    merged_id = memory_agent.memory_store.raw_to_merged[event_id]
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            
            return jsonify({
                'event_id': event_id,
                'multi_dimensional_groups': merge_groups,
                'standard_merged_id': merged_id,
                'total_groups': len(merge_groups)
            })
        else:
            return jsonify({
                'event_id': event_id,
                'multi_dimensional_groups': {},
                'standard_merged_id': None,
                'total_groups': 0
            })
            
    except Exception as e:
        logger.error(f"Error getting merge groups for raw event {event_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
<<<<<<< HEAD
                'id': event.id,  # Use original ID without prefix
=======
                'id': f"raw_{event.id}",
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
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
<<<<<<< HEAD
            'system_created_at': merged_event.created_at.isoformat(),  # For now, use same until we have system timestamps in MergedEvent
            'system_last_updated': merged_event.last_updated.isoformat(),  # For now, use same until we have system timestamps
=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            
            # Component variations
            'who_variants': {},
            'what_variants': {},
            'when_timeline': [],
            'where_locations': merged_event.where_locations,
            'why_variants': {},
            'how_methods': merged_event.how_methods,
            
            # Latest state
            'latest_state': merged_event.get_latest_state() if hasattr(merged_event, 'get_latest_state') else {},
            'dominant_pattern': getattr(merged_event, 'dominant_pattern', None),
            
            # Relationships (with safe attribute access)
            'temporal_chain': getattr(merged_event, 'temporal_chain', None),
            'supersedes': getattr(merged_event, 'supersedes', None),
            'superseded_by': getattr(merged_event, 'superseded_by', None),
            'depends_on': list(getattr(merged_event, 'depends_on', [])),
            'enables': list(getattr(merged_event, 'enables', [])),
            
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
        
        # Build timeline events from raw event IDs if available
        timeline_events = []
        if merged_event and hasattr(merged_event, 'raw_event_ids') and merged_event.raw_event_ids:
            # Try to get raw events for timeline
            chromadb = memory_agent.memory_store.chromadb if hasattr(memory_agent.memory_store, 'chromadb') else None
            if chromadb:
                for event_id in list(merged_event.raw_event_ids)[:10]:  # Limit to 10 for performance
                    try:
                        # Try to get from raw events collection
<<<<<<< HEAD
                        # Just query with the event ID directly
                        raw_results = chromadb.raw_events_collection.get(
                            ids=[event_id],
=======
                        clean_id = event_id.replace('raw_', '').replace('merged_', '')
                        raw_results = chromadb.raw_events_collection.get(
                            ids=[f"raw_{clean_id}", clean_id, event_id],
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                            include=['metadatas']
                        )
                        if raw_results and raw_results['metadatas']:
                            for i, metadata in enumerate(raw_results['metadatas']):
                                if metadata:
                                    timeline_events.append({
                                        'id': raw_results['ids'][i],
<<<<<<< HEAD
                                        'event_type': metadata.get('event_type', 'observation'),
                                        'timestamp': metadata.get('when', metadata.get('created_at', '')),
                                        'five_w1h': {
                                            'who': metadata.get('who', ''),
                                            'what': metadata.get('what', ''),
                                            'when': metadata.get('when', ''),
                                            'where': metadata.get('where', ''),
                                            'why': metadata.get('why', ''),
                                            'how': metadata.get('how', '')
                                        }
=======
                                        'who': metadata.get('who', ''),
                                        'what': metadata.get('what', ''),
                                        'when': metadata.get('when', ''),
                                        'where': metadata.get('where', ''),
                                        'why': metadata.get('why', ''),
                                        'how': metadata.get('how', ''),
                                        'event_type': metadata.get('event_type', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                                    })
                                    break
                    except Exception as e:
                        logger.debug(f"Could not retrieve event {event_id}: {e}")
        
        # Generate enhanced LLM context
        response['llm_context'] = generate_enhanced_llm_context(
            merged_event, 
            merge_type='standard',  # Default to standard merge type
            merge_id=merged_event.id if merged_event else memory_id,
            timeline_events=timeline_events
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Failed to get merged event details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multi-merge/<merge_type>/<merge_id>/details', methods=['GET'])
def get_multi_merge_details(merge_type, merge_id):
    """Get details about a multi-dimensional merge group"""
    try:
        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
        if not multi_merger:
            return jsonify({'error': 'Multi-dimensional merger not available'}), 404
        
        # Get the merge type enum
        from models.merge_types import MergeType
        try:
            merge_type_enum = MergeType(merge_type)
        except ValueError:
            return jsonify({'error': f'Invalid merge type: {merge_type}'}), 400
        
        # Get the merge group
        groups = multi_merger.merge_groups.get(merge_type_enum, {})
        if merge_id not in groups:
            logger.info(f"Multi-merge group not found: {merge_type}/{merge_id}")
            return jsonify({'error': f'Merge group not found: {merge_id}'}), 404
        
        group_data = groups[merge_id]
        merged_event = group_data.get('merged_event')
        
        if not merged_event:
            return jsonify({'error': 'Merge group has no merged event'}), 404
        
        # Get latest state from merged event
        latest_state = merged_event.get_latest_state()
        
        # Get actual events from ChromaDB for timeline
        timeline_events = []
        
        # First check if we have raw_event_ids
        logger.info(f"Checking for raw_event_ids in merged_event...")
<<<<<<< HEAD
        logger.info(f"merged_event type: {type(merged_event)}")
        logger.info(f"merged_event attributes: {dir(merged_event) if merged_event else 'None'}")
        
        if hasattr(merged_event, 'raw_event_ids'):
            logger.info(f"Found raw_event_ids attribute: {merged_event.raw_event_ids}")
=======
        if hasattr(merged_event, 'raw_event_ids'):
            logger.info(f"Found raw_event_ids: {merged_event.raw_event_ids}")
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        else:
            # Try to get from group_data
            raw_ids = group_data.get('raw_event_ids', [])
            logger.info(f"No raw_event_ids attribute, checking group_data: {raw_ids}")
<<<<<<< HEAD
            logger.info(f"group_data keys: {list(group_data.keys())}")
=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            if raw_ids:
                merged_event.raw_event_ids = raw_ids
        
        if hasattr(merged_event, 'raw_event_ids') and merged_event.raw_event_ids:
            # Get raw events from ChromaDB
            chromadb = memory_agent.memory_store.chromadb if hasattr(memory_agent.memory_store, 'chromadb') else None
            if chromadb:
                for event_id in merged_event.raw_event_ids:
                    # Try to get from raw events collection
                    try:
<<<<<<< HEAD
                        # Skip events that are placeholders or generated
                        if 'temporal_' in event_id or 'actor_' in event_id or 'conceptual_' in event_id or 'spatial_' in event_id:
                            logger.debug(f"Skipping synthetic event: {event_id}")
                            continue
                        
                        # Try raw events first
                        # Simply query with the event ID
                        logger.debug(f"Querying raw_events_collection with ID: {event_id}")
                        
                        raw_results = chromadb.raw_events_collection.get(
                            ids=[event_id],
                            include=['metadatas']
                        )
                        logger.debug(f"Raw results found: {len(raw_results.get('ids', []))} events")
                        
                        found_event = False
=======
                        clean_id = event_id.replace('raw_', '').replace('merged_', '').replace('synthetic_', '')
                        
                        # Try raw events first
                        raw_results = chromadb.raw_events_collection.get(
                            ids=[f"raw_{clean_id}", clean_id, event_id],
                            include=['metadatas']
                        )
                        
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                        if raw_results and raw_results['metadatas']:
                            for i, metadata in enumerate(raw_results['metadatas']):
                                if metadata:
                                    timeline_events.append({
                                        'id': raw_results['ids'][i],
<<<<<<< HEAD
                                        'event_type': metadata.get('event_type', 'observation'),
                                        'timestamp': metadata.get('when', metadata.get('created_at', '')),
                                        'five_w1h': {
                                            'who': metadata.get('who', ''),
                                            'what': metadata.get('what', ''),
                                            'when': metadata.get('when', ''),
                                            'where': metadata.get('where', ''),
                                            'why': metadata.get('why', ''),
                                            'how': metadata.get('how', '')
                                        }
                                    })
                                    found_event = True
                                    logger.debug(f"Found event in raw_events: {raw_results['ids'][i]}")
                                    break  # Only add once per event_id
                        
                        if not found_event:
                            logger.debug(f"Event not found in raw_events: {event_id}")
                        
                        # If not found in raw, try regular events collection
                        if not any(e['id'] == event_id for e in timeline_events):
                            results = chromadb.events_collection.get(
                                ids=[event_id],
=======
                                        'who': metadata.get('who', ''),
                                        'what': metadata.get('what', ''),
                                        'when': metadata.get('when', ''),
                                        'where': metadata.get('where', ''),
                                        'why': metadata.get('why', ''),
                                        'how': metadata.get('how', ''),
                                        'event_type': metadata.get('event_type', '')
                                    })
                                    break  # Only add once per event_id
                        
                        # If not found in raw, try regular events collection
                        if not any(e['id'] == event_id for e in timeline_events):
                            results = chromadb.events_collection.get(
                                ids=[clean_id, event_id],
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                                include=['metadatas']
                            )
                            if results and results['metadatas']:
                                for i, metadata in enumerate(results['metadatas']):
                                    if metadata:
                                        timeline_events.append({
                                            'id': results['ids'][i],
<<<<<<< HEAD
                                            'event_type': metadata.get('event_type', 'observation'),
                                            'timestamp': metadata.get('when', metadata.get('created_at', '')),
                                            'five_w1h': {
                                                'who': metadata.get('who', ''),
                                                'what': metadata.get('what', ''),
                                                'when': metadata.get('when', ''),
                                                'where': metadata.get('where', ''),
                                                'why': metadata.get('why', ''),
                                                'how': metadata.get('how', '')
                                            }
=======
                                            'who': metadata.get('who', ''),
                                            'what': metadata.get('what', ''),
                                            'when': metadata.get('when', ''),
                                            'where': metadata.get('where', ''),
                                            'why': metadata.get('why', ''),
                                            'how': metadata.get('how', ''),
                                            'event_type': metadata.get('event_type', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                                        })
                                        break
                    except Exception as e:
                        logger.debug(f"Could not retrieve event {event_id}: {e}")
        
        # For temporal groups, also check temporal chain for additional events
        if merge_type == 'temporal' and hasattr(memory_agent.memory_store, 'temporal_manager'):
            temporal_chain = memory_agent.memory_store.temporal_manager.temporal_chain
            # Try to find the corresponding chain
            for chain_id, chain_events in temporal_chain.chains.items():
                # Check if this temporal group corresponds to this chain
                if chain_id.startswith('episode_') or chain_id.startswith('topic_'):
                    # Get chain metadata
                    chain_metadata = temporal_chain.chain_metadata.get(chain_id, {})
                    # Check if any events in the chain match our raw_event_ids
                    if hasattr(merged_event, 'raw_event_ids'):
                        matching_events = set(chain_events) & set(merged_event.raw_event_ids)
                        if matching_events:
                            # This chain corresponds to our temporal group
                            # Add any missing events from the chain
                            for event_id in chain_events:
                                if not any(e['id'] == event_id for e in timeline_events):
                                    # Create placeholder event with minimal info
                                    timeline_events.append({
                                        'id': event_id,
                                        'who': '',
                                        'what': f'[Event {event_id[:8]}... from temporal chain]',
                                        'when': chain_metadata.get('last_updated', ''),
                                        'where': '',
                                        'why': '',
                                        'how': '',
                                        'event_type': 'temporal_chain_event'
                                    })
                            break
        
<<<<<<< HEAD
        # Sort timeline chronologically (oldest first)
        timeline_events.sort(key=lambda x: x.get('timestamp') or x.get('five_w1h', {}).get('when', '') or '')
        
        # Debug log the timeline events
        logger.info(f"Timeline events loaded: {len(timeline_events)} total")
        timeline_actors = set()
        for evt in timeline_events:
            five_w1h = evt.get('five_w1h', {})
            timeline_actors.add(five_w1h.get('who', 'Unknown'))
        logger.info(f"Actors in timeline_events from DB: {timeline_actors}")
        for evt in timeline_events[:5]:  # Log first 5 events
            five_w1h = evt.get('five_w1h', {})
            logger.info(f"  Event from DB: who={five_w1h.get('who')}, what={five_w1h.get('what', '')[:50]}...")
=======
        # Sort timeline by when field (newest first for latest state at top)
        timeline_events.sort(key=lambda x: x.get('when', ''), reverse=True)
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        
        # Format response similar to standard merged event
        response = {
            'id': merge_id,
            'merge_type': merge_type,
            'merge_key': group_data.get('key', ''),
            'base_event_id': merged_event.base_event_id if hasattr(merged_event, 'base_event_id') else '',
            # Count actual timeline events for accurate count
            'merge_count': len(timeline_events) if timeline_events else len(merged_event.raw_event_ids) if hasattr(merged_event, 'raw_event_ids') else 0,
            'event_count': len(timeline_events) if timeline_events else len(merged_event.raw_event_ids) if hasattr(merged_event, 'raw_event_ids') else 0,
            
            # Latest state
            'latest_state': latest_state,
            
            # Component variants (from merged event)
            'who_variants': {},
            'what_variants': {},
            'when_variants': {},
            'where_locations': {},
            'why_variants': {},
            'how_variants': {},
            
            # Add formatted timeline events for frontend
            'raw_events': timeline_events,
            
            # Metadata
            'created_at': group_data.get('created_at', datetime.utcnow()).isoformat(),
            'last_updated': group_data.get('last_updated', datetime.utcnow()).isoformat()
        }
        
        # Format WHO variants - collect from timeline events if available
        if timeline_events and merge_type != 'actor':
            # For non-actor merge types, collect WHO from all timeline events
            who_counts = {}
            for event in timeline_events:
<<<<<<< HEAD
                five_w1h = event.get('five_w1h', {})
                who_value = five_w1h.get('who', '')
=======
                who_value = event.get('who', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                if who_value:
                    if who_value not in who_counts:
                        who_counts[who_value] = []
                    who_counts[who_value].append({
                        'value': who_value,
<<<<<<< HEAD
                        'timestamp': five_w1h.get('when', ''),
=======
                        'timestamp': event.get('when', ''),
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                        'event_id': event.get('id', ''),
                        'relationship': 'participant'
                    })
            
            # Add all collected WHO variants
            for who, events in who_counts.items():
                response['who_variants'][who] = events
        elif hasattr(merged_event, 'who_variants') and merged_event.who_variants:
            # Fall back to merged event's who_variants for actor merge type
            for who, variants in merged_event.who_variants.items():
                if isinstance(variants, list):
                    response['who_variants'][who] = [
                        {
                            'value': v.value if hasattr(v, 'value') else str(v),
                            'timestamp': v.timestamp.isoformat() if hasattr(v, 'timestamp') and hasattr(v.timestamp, 'isoformat') else '',
                            'event_id': v.event_id if hasattr(v, 'event_id') else '',
                            'relationship': v.relationship.value if hasattr(v, 'relationship') and hasattr(v.relationship, 'value') else 'unknown',
                        }
                        for v in variants
                    ]
                else:
                    # Handle non-list variants
                    response['who_variants'][who] = [{'value': str(variants), 'timestamp': '', 'event_id': '', 'relationship': 'unknown'}]
        
        # Format WHAT variants - collect from timeline events if available for non-actor types
        if timeline_events and merge_type != 'actor':
            # For non-actor merge types, collect WHAT from all timeline events
            what_counts = {}
            for event in timeline_events:
<<<<<<< HEAD
                five_w1h = event.get('five_w1h', {})
                what_value = five_w1h.get('what', '')
=======
                what_value = event.get('what', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                if what_value:
                    # Use a truncated version as key to group similar whats
                    what_key = what_value[:50] + '...' if len(what_value) > 50 else what_value
                    if what_key not in what_counts:
                        what_counts[what_key] = []
                    what_counts[what_key].append({
                        'value': what_value,
<<<<<<< HEAD
                        'timestamp': five_w1h.get('when', ''),
=======
                        'timestamp': event.get('when', ''),
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                        'event_id': event.get('id', ''),
                        'relationship': 'action'
                    })
            
            # Add all collected WHAT variants
            for what, events in what_counts.items():
                response['what_variants'][what] = events
        elif hasattr(merged_event, 'what_variants') and merged_event.what_variants:
            # Fall back to merged event's what_variants
            for what, variants in merged_event.what_variants.items():
                response['what_variants'][what] = [
                    {
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat() if hasattr(v.timestamp, 'isoformat') else str(v.timestamp),
                        'event_id': v.event_id,
                        'relationship': v.relationship.value if hasattr(v.relationship, 'value') else str(v.relationship),
                    }
                    for v in variants
                ]
        
        # Format WHEN timeline - check for timeline or raw events
        timeline_items = []
        if hasattr(merged_event, 'when_timeline') and merged_event.when_timeline:
            for tp in merged_event.when_timeline:
                timeline_items.append({
                    'value': tp.semantic_time if hasattr(tp, 'semantic_time') else (tp.timestamp.isoformat() if hasattr(tp, 'timestamp') else str(tp)),
                    'timestamp': tp.timestamp.isoformat() if hasattr(tp, 'timestamp') and hasattr(tp.timestamp, 'isoformat') else str(tp),
                    'event_id': tp.event_id if hasattr(tp, 'event_id') else '',
                    'relationship': 'temporal'
                })
        elif latest_state.get('when'):
            # Fallback to latest state if no timeline
            timeline_items.append({
                'value': latest_state['when'],
                'timestamp': latest_state['when'],
                'event_id': merge_id,
                'relationship': 'temporal'
            })
        response['when_variants']['timeline'] = timeline_items
        
        # Format WHERE locations - collect from timeline events if available for non-actor types
        if timeline_events and merge_type != 'actor':
            # For non-actor merge types, collect WHERE from all timeline events
            where_counts = {}
            for event in timeline_events:
<<<<<<< HEAD
                five_w1h = event.get('five_w1h', {})
                where_value = five_w1h.get('where', '')
=======
                where_value = event.get('where', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                if where_value:
                    if where_value not in where_counts:
                        where_counts[where_value] = 0
                    where_counts[where_value] += 1
            
            # Format as locations list
            response['where_locations']['locations'] = [
                {
                    'value': location,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_id': '',
                    'relationship': 'location',
                    'count': count
                }
                for location, count in where_counts.items()
            ]
        elif hasattr(merged_event, 'where_locations') and merged_event.where_locations:
            # Fall back to merged event's where_locations
            response['where_locations']['locations'] = [
                {
                    'value': location,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_id': '',
                    'relationship': 'location',
                    'count': count
                }
                for location, count in merged_event.where_locations.items()
            ]
        
        # Format WHY variants - collect from timeline events if available for non-actor types
        if timeline_events and merge_type != 'actor':
            # For non-actor merge types, collect WHY from all timeline events
            why_counts = {}
            for event in timeline_events:
<<<<<<< HEAD
                five_w1h = event.get('five_w1h', {})
                why_value = five_w1h.get('why', '')
=======
                why_value = event.get('why', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                if why_value:
                    if why_value not in why_counts:
                        why_counts[why_value] = []
                    why_counts[why_value].append({
                        'value': why_value,
<<<<<<< HEAD
                        'timestamp': five_w1h.get('when', ''),
=======
                        'timestamp': event.get('when', ''),
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                        'event_id': event.get('id', ''),
                        'relationship': 'reason'
                    })
            
            # Add all collected WHY variants
            for why, events in why_counts.items():
                response['why_variants'][why] = events
        elif hasattr(merged_event, 'why_variants') and merged_event.why_variants:
            # Fall back to merged event's why_variants
            for why, variants in merged_event.why_variants.items():
                response['why_variants'][why] = [
                    {
                        'value': v.value,
                        'timestamp': v.timestamp.isoformat() if hasattr(v.timestamp, 'isoformat') else str(v.timestamp),
                        'event_id': v.event_id,
                        'relationship': v.relationship.value if hasattr(v.relationship, 'value') else str(v.relationship),
                    }
                    for v in variants
                ]
        
        # Format HOW methods - collect from timeline events if available for non-actor types
        if timeline_events and merge_type != 'actor':
            # For non-actor merge types, collect HOW from all timeline events
            how_counts = {}
            for event in timeline_events:
<<<<<<< HEAD
                five_w1h = event.get('five_w1h', {})
                how_value = five_w1h.get('how', '')
=======
                how_value = event.get('how', '')
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                if how_value:
                    if how_value not in how_counts:
                        how_counts[how_value] = 0
                    how_counts[how_value] += 1
            
            # Format as methods list
            response['how_variants']['methods'] = [
                {
                    'value': method,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_id': '',
                    'relationship': 'method',
                    'count': count
                }
                for method, count in how_counts.items()
            ]
        elif hasattr(merged_event, 'how_methods') and merged_event.how_methods:
            # Fall back to merged event's how_methods
            response['how_variants']['methods'] = [
                {
                    'value': method,
                    'timestamp': datetime.utcnow().isoformat(),
                    'event_id': '',
                    'relationship': 'method',
                    'count': count
                }
                for method, count in merged_event.how_methods.items()
            ]
        
        # We'll add LLM context after response is fully built
        
        # Add timeline events if we retrieved them
        if 'timeline_events' not in response:
            response['timeline_events'] = timeline_events if 'timeline_events' in locals() else []
        else:
            response['timeline_events'] = timeline_events if 'timeline_events' in locals() else response['timeline_events']
        
<<<<<<< HEAD
        # Keep original timeline_events in raw_events for LLM context
        # The raw_events was already set to timeline_events at line 1159
        # We should NOT overwrite it here
        
        # Format raw_events for the frontend display in a separate field
        if timeline_events:
            response['formatted_raw_events'] = []
=======
        # Format raw_events for the frontend (similar to standard merged event)
        if timeline_events:
            response['raw_events'] = []
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
            response['total_raw'] = len(timeline_events)
            
            for event_data in timeline_events:
                formatted_event = {
                    'id': event_data['id'],
                    'event_type': event_data.get('event_type', 'observation'),
                    'timestamp': event_data.get('when', ''),
                    'episode_id': f"episode_{merge_type}_{merge_id[:8]}",
                    'five_w1h': {
                        'who': event_data.get('who', ''),
                        'what': event_data.get('what', ''),
                        'when': event_data.get('when', ''),
                        'where': event_data.get('where', ''),
                        'why': event_data.get('why', ''),
                        'how': event_data.get('how', '')
                    }
                }
<<<<<<< HEAD
                response['formatted_raw_events'].append(formatted_event)
            
            # Ensure raw_events still has the original timeline_events
            if not response.get('raw_events'):
                response['raw_events'] = timeline_events
        
        # Generate enhanced LLM context after response is fully built
        logger.info(f"Before LLM context - timeline_events count: {len(timeline_events) if timeline_events else 0}")
        logger.info(f"Before LLM context - response raw_events count: {len(response.get('raw_events', [])) if response.get('raw_events') else 0}")
=======
                response['raw_events'].append(formatted_event)
        
        # Generate enhanced LLM context after response is fully built
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
        llm_context = generate_enhanced_llm_context(merged_event, merge_type, merge_id, response)
        response['llm_context'] = llm_context
        response['functional_context_model'] = llm_context.get('narrative_summary', '')  # For backwards compatibility
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Failed to get multi-merge details: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
                raw_event_ids_to_include = []
                
                # Check if it's a multi-dimensional merge group ID (e.g., temporal_xxx, actor_xxx, etc.)
<<<<<<< HEAD
                if '_' in center_node_id and (center_node_id.startswith('temporal_') or center_node_id.startswith('actor_') or center_node_id.startswith('conceptual_') or center_node_id.startswith('spatial_')):
=======
                if '_' in center_node_id and not center_node_id.startswith('merged_') and not center_node_id.startswith('raw_'):
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                    # Try to parse as multi-dimensional merge group
                    parts = center_node_id.split('_', 1)
                    logger.info(f"Checking multi-dimensional merge: parts={parts}")
                    if len(parts) == 2:
                        merge_type_str, group_id = parts
                        # Try to get the merge group
                        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
                        logger.info(f"Multi-merger exists: {multi_merger is not None}")
                        if multi_merger:
                            from models.merge_types import MergeType
                            try:
                                merge_type_enum = MergeType(merge_type_str)
                                groups = multi_merger.merge_groups.get(merge_type_enum, {})
                                logger.info(f"Groups for {merge_type_str}: {len(groups)} groups, looking for {center_node_id}")
                                logger.info(f"Available group IDs: {list(groups.keys())[:5]}")  # Show first 5 IDs
                                # The group IDs are stored with the full prefix, so use the full center_node_id
                                if center_node_id in groups:
                                    group_data = groups[center_node_id]
                                    logger.info(f"Found group {center_node_id}, group_data keys: {group_data.keys() if isinstance(group_data, dict) else 'not a dict'}")
                                    # Get the merged event from the group data
                                    merged_event = group_data.get('merged_event')
                                    if merged_event:
                                        logger.info(f"Found merged_event for {center_node_id}")
                                        latest_state = merged_event.get_latest_state()
                                        logger.info(f"Latest state type: {type(latest_state)}, keys: {latest_state.keys() if isinstance(latest_state, dict) else 'not a dict'}")
                                        center_metadata = {
                                            'who': latest_state.get('who', ''),
                                            'what': latest_state.get('what', ''),
                                            'when': latest_state.get('when', ''),
                                            'where': latest_state.get('where', ''),
                                            'why': latest_state.get('why', ''),
                                            'how': latest_state.get('how', '')
                                        }
                                        raw_event_ids_to_include = list(merged_event.raw_event_ids)
                                        logger.info(f"Successfully extracted metadata for {center_node_id}")
                                    else:
                                        logger.warning(f"No merged_event in group_data for {center_node_id}")
                            except ValueError:
                                pass  # Not a valid merge type
                
                # Check if it's a merged event ID
                if center_metadata is None and center_node_id.startswith('merged_'):
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
                elif center_metadata is None:
                    # Regular event ID - check both with and without raw_ prefix
                    actual_center_id = center_node_id
<<<<<<< HEAD
                    # Check raw_events collection
                    if True:  # Always check
                        # Try with raw_ prefix first
                        collection = memory_agent.memory_store.chromadb.client.get_collection("raw_events")
                        test_result = collection.get(
                            ids=[center_node_id],
                            include=["metadatas"]
                        )
                        if test_result['ids']:
                            actual_center_id = center_node_id
=======
                    if not center_node_id.startswith('raw_'):
                        # Try with raw_ prefix first
                        collection = memory_agent.memory_store.chromadb.client.get_collection("raw_events")
                        test_result = collection.get(
                            ids=[f"raw_{center_node_id}"],
                            include=["metadatas"]
                        )
                        if test_result['ids']:
                            actual_center_id = f"raw_{center_node_id}"
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                
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

@app.route('/api/temporal-summary', methods=['GET'])
def get_temporal_summary():
    """Get comprehensive temporal system summary"""
    try:
        if not hasattr(memory_agent.memory_store, 'temporal_manager'):
            return jsonify({'error': 'Temporal manager not available'}), 404
        
        temporal_manager = memory_agent.memory_store.temporal_manager
        
        # Get temporal chains summary
        chains_summary = temporal_manager.get_temporal_chains_summary()
        
        # Get temporal merge groups
        temporal_groups = temporal_manager.get_temporal_merge_groups()
        
        # Create summary response
        response = {
            'chains': chains_summary,
            'temporal_groups': {
                'total': len(temporal_groups),
                'groups': temporal_groups[:10]  # Top 10 groups
            },
            'temporal_indicators': temporal_manager.get_temporal_indicators_dict()
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting temporal summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/temporal-context/<event_id>', methods=['GET'])
def get_temporal_context(event_id):
    """Get temporal context for a specific event"""
    try:
        if not hasattr(memory_agent.memory_store, 'temporal_manager'):
            return jsonify({'error': 'Temporal manager not available'}), 404
        
        temporal_manager = memory_agent.memory_store.temporal_manager
        context = temporal_manager.get_temporal_context_for_event(event_id)
        
        return jsonify(context)
    except Exception as e:
        logger.error(f"Error getting temporal context: {e}")
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
                total_events = 0
                for group_data in groups.values():
                    merged_event = group_data.get('merged_event')
                    if merged_event:
                        total_events += len(merged_event.raw_event_ids)
                multi_stats[merge_type.value] = {
                    'group_count': len(groups),
                    'total_events': total_events
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
        
        # Check if we have multi-dimensional merger
        multi_merger = getattr(memory_agent.memory_store, 'multi_merger', None)
        
        for merge_type in MergeType:
            count = 0
            total_events = 0
            
            if multi_merger and merge_type != MergeType.HYBRID:
                # Get counts from multi-dimensional merger
                groups = multi_merger.merge_groups.get(merge_type, {})
                count = len(groups)
                for group_data in groups.values():
                    merged_event = group_data.get('merged_event')
                    if merged_event:
                        total_events += len(merged_event.raw_event_ids)
            
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
<<<<<<< HEAD
                        'system_created_at': group_data.get('system_created_at', datetime.utcnow()).isoformat() if 'system_created_at' in group_data else '',
                        'system_last_updated': group_data.get('system_last_updated', datetime.utcnow()).isoformat() if 'system_last_updated' in group_data else '',
=======
>>>>>>> 592e91b67bc5d7e9207c23ad123f482137168f02
                        'latest_state': latest_state,
                        'raw_event_ids': list(merged_event.raw_event_ids),
                        'events_count': merged_event.merge_count  # Use merge_count for consistency
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
            'latest_state': merged_event.get_latest_state() if hasattr(merged_event, 'get_latest_state') else {},
            'dominant_pattern': getattr(merged_event, 'dominant_pattern', None),
            
            # Relationships (with safe attribute access)
            'temporal_chain': getattr(merged_event, 'temporal_chain', None),
            'supersedes': getattr(merged_event, 'supersedes', None),
            'superseded_by': getattr(merged_event, 'superseded_by', None),
            'depends_on': list(getattr(merged_event, 'depends_on', [])),
            'enables': list(getattr(merged_event, 'enables', [])),
            
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
        
        # Generate context preview for LLM tab
        context_summary = f"## Merged Event Summary\n\n"
        context_summary += f"This merged event contains {merged_event.merge_count} similar events.\n"
        latest = merged_event.get_latest_state()
        if latest.get('who'):
            context_summary += f"Primary actor: {latest['who']}\n"
        if latest.get('what'):
            context_summary += f"Latest action: {latest['what'][:100]}...\n" if len(latest.get('what', '')) > 100 else f"Latest action: {latest['what']}\n"
        if merged_event.dominant_pattern:
            context_summary += f"Dominant pattern: {merged_event.dominant_pattern}\n"
        
        context_detailed = "### Detailed Context\n\n"
        context_detailed += "**Latest State:**\n"
        for key, value in latest.items():
            if value:
                if key == 'what' and len(str(value)) > 200:
                    context_detailed += f"- **{key.title()}:** {str(value)[:200]}...\n"
                else:
                    context_detailed += f"- **{key.title()}:** {value}\n"
        
        context_detailed += f"\n**Merge Statistics:**\n"
        context_detailed += f"- Total merged events: {merged_event.merge_count}\n"
        context_detailed += f"- Created: {merged_event.created_at.isoformat()}\n"
        context_detailed += f"- Last updated: {merged_event.last_updated.isoformat()}\n"
        
        if merged_event.who_variants:
            context_detailed += f"\n**Actors involved:** {', '.join(merged_event.who_variants.keys())}\n"
        
        if merged_event.where_locations:
            context_detailed += f"\n**Locations:** {', '.join(merged_event.where_locations.keys())}\n"
        
        response['context_preview'] = {
            'summary': context_summary,
            'detailed': context_detailed
        }
        
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
