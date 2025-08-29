"""
Script to reload benchmark dataset with raw event preservation
Stores all 596 events as raw while maintaining merged view
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
warnings.filterwarnings('ignore', category=FutureWarning)

from memory.memory_store import MemoryStore
from models.event import Event, FiveW1H, EventType
from config import get_config
from agent.memory_agent import MemoryAgent
import uuid
from datetime import datetime

def load_benchmark_with_raw_preservation(dataset_path: str, memory_agent: MemoryAgent):
    """
    Load benchmark dataset preserving all raw events
    
    Args:
        dataset_path: Path to the benchmark JSON file
        memory_agent: MemoryAgent instance
    """
    print(f"\nLoading benchmark dataset from: {dataset_path}")
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    total_events = len(data['events'])
    print(f"Found {total_events} events in dataset")
    print("Each event will be stored as raw and potentially merged for display")
    
    # Track statistics
    successful = 0
    failed = 0
    merged_count = 0
    
    # Process each event
    for i, event_data in enumerate(data['events']):
        try:
            # Create Event object with original ID
            event = Event(
                id=event_data.get('event_id', str(uuid.uuid4())),
                five_w1h=FiveW1H(
                    who=event_data['five_w1h'].get('who', ''),
                    what=event_data['five_w1h'].get('what', ''),
                    when=event_data['five_w1h'].get('when', ''),
                    where=event_data['five_w1h'].get('where', ''),
                    why=event_data['five_w1h'].get('why', ''),
                    how=event_data['five_w1h'].get('how', '')
                ),
                event_type=EventType(event_data.get('event_type', 'observation')),
                episode_id=event_data.get('episode_id', f"benchmark_{i}")
            )
            
            # Store event with raw preservation
            success, message = memory_agent.memory_store.store_event(event, preserve_raw=True)
            
            if success:
                successful += 1
                if "Merged" in message:
                    merged_count += 1
                
                # Progress update
                if (i + 1) % 50 == 0:
                    stats = memory_agent.memory_store.get_statistics()
                    print(f"Progress: {i+1}/{total_events} events")
                    print(f"  Stored: {successful} (Merged: {merged_count})")
                    print(f"  Raw events: {stats.get('total_raw_events', 0)}")
                    print(f"  Merged groups: {stats.get('total_merged_groups', 0)}")
            else:
                failed += 1
                print(f"Failed to store event {i}: {message}")
                
        except Exception as e:
            failed += 1
            print(f"Error processing event {i}: {e}")
    
    # Final statistics
    stats = memory_agent.memory_store.get_statistics()
    merge_groups = memory_agent.memory_store.get_merge_groups()
    
    print("\n" + "="*60)
    print("LOADING COMPLETE!")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Total events in file: {total_events}")
    print(f"Successfully processed: {successful}")
    print(f"Failed to store: {failed}")
    print(f"\nMemory Store Statistics:")
    print(f"  Total merged events (for display): {stats.get('total_events', 0)}")
    print(f"  Total raw events (preserved): {stats.get('total_raw_events', 0)}")
    print(f"  Total merge groups: {stats.get('total_merged_groups', 0)}")
    print(f"  Average events per group: {stats.get('average_merge_size', 0):.2f}")
    
    # Show some example merge groups
    if merge_groups:
        print(f"\nExample merge groups (showing first 3):")
        for i, (merged_id, raw_ids) in enumerate(list(merge_groups.items())[:3]):
            print(f"  Group {merged_id[:8]}... has {len(raw_ids)} raw events")
    
    print("="*60)
    
    return successful, failed

def main():
    """Main function to reload benchmark dataset"""
    # Initialize configuration
    config = get_config()
    
    # Create memory agent
    print("Initializing Memory Agent...")
    memory_agent = MemoryAgent(config)
    
    # Check for existing data
    initial_stats = memory_agent.memory_store.get_statistics()
    if initial_stats.get('total_raw_events', 0) > 0:
        print(f"\nWARNING: Memory store already contains {initial_stats['total_raw_events']} raw events")
        print(f"         and {initial_stats['total_events']} merged events")
        response = input("Do you want to continue and add more events? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Find the most recent benchmark dataset
    benchmark_dir = Path("benchmark_datasets")
    if not benchmark_dir.exists():
        print(f"Error: {benchmark_dir} directory not found")
        return
    
    # Get all benchmark files (exclude query files)
    benchmark_files = sorted([f for f in benchmark_dir.glob("benchmark_*.json") 
                             if not f.name.endswith("_queries.json")])
    if not benchmark_files:
        print("No benchmark datasets found")
        return
    
    # Use the most recent one
    latest_file = benchmark_files[-1]
    
    print(f"\nUsing benchmark file: {latest_file.name}")
    
    # Load the dataset
    successful, failed = load_benchmark_with_raw_preservation(str(latest_file), memory_agent)
    
    print(f"\nDataset successfully loaded!")
    print(f"The web interface will show:")
    print(f"  - Merged view: Deduplicated events (default)")
    print(f"  - Raw view: All {successful} original events grouped by merge")
    print(f"\nYou can now run 'python run_web.py' to see the results")

if __name__ == "__main__":
    main()