#!/usr/bin/env python3
"""
Load the LLM-generated benchmark dataset into the memory system.
"""

import json
import sys
from pathlib import Path
import warnings
import os

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.memory_agent import MemoryAgent
from datetime import datetime

def main():
    # Dataset file
    dataset_file = "benchmark_datasets/dataset_small_llm_20250901_035752.json"
    
    if not Path(dataset_file).exists():
        print(f"Dataset file not found: {dataset_file}")
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
    
    # Initialize memory agent
    print("\nInitializing Memory Agent...")
    memory_agent = MemoryAgent()  # Initialize with default settings
    
    # Process events
    print(f"\nLoading {len(events)} events into memory system...")
    successful = 0
    failed = 0
    
    for i, event in enumerate(events, 1):
        try:
            # Map event types from dataset to valid EventType values
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
            
            # Store the event using the 5W1H structure with original timestamp
            success, message, stored_event = memory_agent.remember(
                who=event.get('who', 'Unknown'),
                what=event.get('what', ''),
                when=event.get('when'),  # Pass the original timestamp
                where=event.get('where', 'unknown'),
                why=event.get('why', 'conversation'),
                how=event.get('how', 'typed'),
                event_type=mapped_event_type
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                print(f"  Failed to store event {i}: {message}")
            
            # Progress update
            if i % 50 == 0 or i == len(events):
                print(f"  Progress: {i}/{len(events)} events processed ({successful} successful, {failed} failed)")
                
        except Exception as e:
            failed += 1
            print(f"  Error processing event {i}: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"LOADING COMPLETE")
    print(f"{'='*60}")
    print(f"Total events processed: {len(events)}")
    print(f"Successfully stored: {successful}")
    print(f"Failed: {failed}")
    
    # Get memory statistics
    stats = memory_agent.get_statistics()
    print(f"\nMemory System Statistics:")
    print(f"  Total memories: {stats.get('total_memories', 0)}")
    print(f"  Total episodes: {stats.get('total_episodes', 0)}")
    print(f"  Average memories per episode: {stats.get('avg_memories_per_episode', 0):.1f}")
    
    # Show merge statistics if available
    if hasattr(memory_agent.memory_store, 'chromadb'):
        try:
            # Get merge stats from ChromaDB
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
            print(f"Could not retrieve merge statistics: {e}")
    
    print("\nDataset loaded successfully!")
    print("You can now use the web interface to explore the loaded conversations.")
    print("Run: python run_web.py")

if __name__ == "__main__":
    main()
