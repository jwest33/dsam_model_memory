#!/usr/bin/env python3
"""
Enhanced dataset loader that uses intelligent field generation.
Loads the LLM-generated benchmark dataset with proper field generation.
"""

import json
import sys
from pathlib import Path
import warnings
import os
import logging
from datetime import datetime

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.memory_agent import MemoryAgent
from memory.field_generator import FieldGenerator, MechanismType
from llm.llm_interface import LLMInterface
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Dataset file
    dataset_file = "benchmark_datasets/dataset_small_llm_20250901_035752.json"
    
    if not Path(dataset_file).exists():
        print(f"Dataset file not found: {dataset_file}")
        print("Please ensure you have generated a dataset first.")
        return
    
    print(f"Loading dataset: {dataset_file}")
    
    # Load the dataset
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Show dataset info
    metadata = data.get('metadata', {})
    stats = metadata.get('stats', {})
    events = data.get('events', [])
    
    print(f"\nDataset Info:")
    print(f"  Name: {metadata.get('name', 'Unknown')}")
    print(f"  Created: {metadata.get('created_at', 'Unknown')}")
    print(f"  Generator: {metadata.get('generator', 'Unknown')}")
    print(f"  Total Events: {len(events)}")
    print(f"  Conversations: {stats.get('num_conversations', 0)}")
    print(f"  Technical: {stats.get('technical_conversations', 0)}")
    print(f"  Casual: {stats.get('casual_conversations', 0)}")
    
    # Initialize components
    print("\nInitializing System Components...")
    config = get_config()
    memory_agent = MemoryAgent(config)
    
    # Initialize field generator with LLM if available
    try:
        llm_interface = LLMInterface(config.llm)
        field_generator = FieldGenerator(llm_client=llm_interface)
        print("  ✓ Field generator initialized with LLM support")
        
        # Also set LLM client for merge group field generation
        if hasattr(memory_agent.memory_store, 'multi_merger'):
            memory_agent.memory_store.multi_merger.llm_client = llm_interface
            memory_agent.memory_store.multi_merger.group_field_generator.llm_client = llm_interface
            print("  ✓ Merge group field generator initialized with LLM support")
    except Exception as e:
        field_generator = FieldGenerator(llm_client=None)
        print(f"  ⚠ Field generator using heuristics (LLM unavailable): {e}")
    
    # Check if enhanced features are available
    print("\nEnhanced Features Status:")
    if hasattr(memory_agent.memory_store, 'encoder'):
        print("  ✓ Dual-space encoding (Euclidean + Hyperbolic) enabled")
        print(f"    - Euclidean dimension: {memory_agent.memory_store.encoder.euclidean_dim}")
        print(f"    - Hyperbolic dimension: {memory_agent.memory_store.encoder.hyperbolic_dim}")
    
    if hasattr(memory_agent.memory_store, 'multi_merger'):
        print("  ✓ Multi-dimensional merging enabled")
        print("  ✓ Group-level field generation (group_why, group_how) enabled")
        print("  ✓ Enhanced conceptual search with text matching enabled")
    
    print("\nNote: The system will intelligently generate 'why' and 'how' fields")
    print("      based on content analysis. This provides better semantic grouping")
    print("      but may take longer than simple loading.")
    
    # Process events
    print(f"\nLoading {len(events)} events into memory system...")
    successful = 0
    failed = 0
    
    # Track conversation context for better field generation
    conversation_context = []
    
    for i, event in enumerate(events, 1):
        try:
            # Map event types
            event_type_map = {
                'user_message': 'user_input',
                'assistant_response': 'action',
                'user_input': 'user_input',
                'action': 'action',
                'observation': 'observation',
                'system_event': 'system_event'
            }
            
            original_event_type = event.get('event_type', 'observation')
            mapped_event_type = event_type_map.get(original_event_type, 'observation')
            
            # Extract basic fields
            who = event.get('who', 'Unknown')
            what = event.get('what', '')
            when = event.get('when')
            where = event.get('where', 'chat')
            
            # Add to conversation context
            field_generator.add_to_context(what, who)
            
            # Generate intelligent 'why' field
            if who.lower() in ['user', 'human']:
                why = field_generator.generate_why_field(
                    current_message=what,
                    who=who,
                    message_type='query'
                )
                mechanism = MechanismType.CHAT_INTERFACE
            else:
                why = field_generator.generate_why_field(
                    current_message=what,
                    who=who,
                    message_type='response'
                )
                mechanism = MechanismType.LLM_GENERATION
            
            # Generate intelligent 'how' field
            how = field_generator.generate_how_field(
                mechanism=mechanism,
                details={'source': 'dataset', 'conversation_id': event.get('conversation_id')}
            )
            
            # Store the event with generated fields
            success, message, stored_event = memory_agent.remember(
                who=who,
                what=what,
                when=when,
                where=where,
                why=why,  # Intelligently generated
                how=how,  # Intelligently generated
                event_type=mapped_event_type
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                print(f"  Failed to store event {i}: {message}")
            
            # Progress update
            if i % 10 == 0 or i == len(events):
                print(f"  Progress: {i}/{len(events)} events ({successful} successful, {failed} failed)")
                
                # Show sample of generated fields
                if successful > 0 and stored_event and i % 30 == 0:
                    print(f"    Last event fields:")
                    print(f"      who: {who}")
                    print(f"      what: {what[:60]}...")
                    print(f"      why: {why}")
                    print(f"      how: {how}")
                    
        except Exception as e:
            failed += 1
            logger.error(f"Error processing event {i}: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"LOADING COMPLETE")
    print(f"{'='*70}")
    print(f"Total events processed: {len(events)}")
    print(f"Successfully stored: {successful}")
    print(f"Failed: {failed}")
    
    # Get memory statistics
    stats = memory_agent.get_statistics()
    print(f"\nMemory System Statistics:")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  Total episodes: {stats.get('total_episodes', 0)}")
    print(f"  Average memories per episode: {stats.get('avg_memories_per_episode', 0):.1f}")
    
    # Show merge statistics
    if hasattr(memory_agent.memory_store, 'chromadb'):
        try:
            merge_stats = memory_agent.memory_store.chromadb.get_merge_statistics()
            
            print(f"\nMerge Statistics:")
            print(f"  Merged events: {merge_stats.get('merged_events', 0)}")
            print(f"  Raw events: {merge_stats.get('raw_events', 0)}")
            print(f"  Merge ratio: {merge_stats.get('merge_ratio', 0):.2%}")
            print(f"  Average merge size: {merge_stats.get('avg_merge_size', 0):.1f}")
            
            # Dimensional merge stats
            dimensional_stats = merge_stats.get('dimensional_merges', {})
            if dimensional_stats:
                print(f"\nDimensional Merge Groups:")
                for dim, count in dimensional_stats.items():
                    print(f"  {dim.capitalize()}: {count} groups")
        except Exception as e:
            logger.debug(f"Could not retrieve merge statistics: {e}")
    
    # Show sample merge group characterizations
    if hasattr(memory_agent.memory_store, 'multi_merger'):
        try:
            print(f"\n{'='*70}")
            print("Sample Merge Group Characterizations:")
            print('-'*70)
            
            merger = memory_agent.memory_store.multi_merger
            from models.merge_types import MergeType
            
            samples_shown = 0
            max_samples = 3
            
            for merge_type in [MergeType.CONCEPTUAL, MergeType.TEMPORAL, MergeType.ACTOR]:
                if merge_type in merger.merge_groups and samples_shown < max_samples:
                    groups = merger.merge_groups[merge_type]
                    if groups:
                        # Get a few sample groups
                        for group_id, group_data in list(groups.items())[:2]:
                            merged_event = group_data.get('merged_event')
                            if merged_event and merged_event.group_why:
                                print(f"\n{merge_type.value.capitalize()} Group '{group_id}':")
                                print(f"  Purpose (why): {merged_event.group_why}")
                                print(f"  Mechanism (how): {merged_event.group_how}")
                                print(f"  Size: {merged_event.merge_count} events")
                                
                                # Check for dual-space embeddings
                                has_euclidean = 'centroid_embedding' in group_data
                                has_hyperbolic = 'hyperbolic_embedding' in group_data
                                print(f"  Embeddings: Euclidean={'✓' if has_euclidean else '✗'}, Hyperbolic={'✓' if has_hyperbolic else '✗'}")
                                
                                # Show sample events in group
                                latest = merged_event.get_latest_state()
                                print(f"  Latest state: {latest.get('what', '')[:60]}...")
                                
                                samples_shown += 1
                                if samples_shown >= max_samples:
                                    break
                                    
        except Exception as e:
            logger.debug(f"Could not show merge group samples: {e}")
    
    print(f"\n{'='*70}")
    print("Dataset loaded successfully with intelligent field generation!")
    print("\nKey Features Applied:")
    print("  • Context-aware 'why' fields based on message content")
    print("  • Mechanism-specific 'how' fields")
    print("  • Semantic grouping in 4 dimensions (Actor, Temporal, Conceptual, Spatial)")
    print("  • Group-level characterization with purpose and mechanism")
    print("\nEnhanced Retrieval Features:")
    print("  • Dual-space embeddings (Euclidean + Hyperbolic) for all merge groups")
    print("  • LLM-generated group_why and group_how fields stored in metadata")
    print("  • Enhanced conceptual search combining:")
    print("    - Embedding similarity (dual-space ready)")
    print("    - Text matching with group characterization fields")
    print("    - Keyword boosting for conceptual terms")
    print("  • Faster retrieval through indexed text fields")
    print("\nYou can now use the web interface to explore the conversations:")
    print("  python run_web.py")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()