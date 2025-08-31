#!/usr/bin/env python3
"""
Simplified benchmark dataset loader - just loads events, no manual dimensional grouping.
All dimensional grouping is handled internally by the memory system.
"""

import json
import sys
from pathlib import Path
import warnings
import os
import time
from typing import Dict, List, Tuple

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_store import MemoryStore
from memory.similarity_cache import SimilarityCache
from models.event import Event, FiveW1H, EventType
from config import get_config
from agent.memory_agent import MemoryAgent
import uuid
from datetime import datetime

def load_benchmark_dataset(
    dataset_path: str, 
    memory_agent: MemoryAgent,
    batch_size: int = 50,
    compute_cache: bool = True
) -> Tuple[int, int, Dict]:
    """
    Load benchmark dataset - simplified version.
    
    The memory system handles all:
    - Raw event storage
    - Event merging/deduplication
    - Multi-dimensional grouping (actor, temporal, conceptual, spatial)
    - Similarity caching
    
    Args:
        dataset_path: Path to the benchmark JSON file
        memory_agent: MemoryAgent instance
        batch_size: Number of events to process before updating cache
        compute_cache: Whether to compute similarity cache
        
    Returns:
        Tuple of (successful_count, failed_count, statistics)
    """
    print(f"\nLoading benchmark dataset from: {dataset_path}")
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: events in 'events' key or nested in conversations
    if 'events' in data:
        # New format: events are in a separate 'events' key
        events = data['events']
        conversations = data.get('conversations', [])
        total_events = len(events)
        print(f"Found {len(conversations)} conversations with {total_events} total events")
    else:
        # Old format: events are nested in conversations as 'messages'
        conversations = data.get('conversations', [])
        total_events = sum(len(conv.get('messages', [])) for conv in conversations)
        events = []
        for conv in conversations:
            for msg in conv.get('messages', []):
                events.append(msg)
        print(f"Found {len(conversations)} conversations with {total_events} total events")
    
    successful = 0
    failed = 0
    merged_count = 0
    batch_embeddings = {}
    
    # Process events in batches
    for batch_start in range(0, total_events, batch_size):
        batch_end = min(batch_start + batch_size, total_events)
        print(f"\nProcessing batch {batch_start}-{batch_end}...")
        
        # Get events for this batch
        batch_events = events[batch_start:batch_end]
        
        # Process batch
        for i, event_data in enumerate(batch_events, start=batch_start):
            try:
                # Create Event from event data - handle both formats
                if 'five_w1h' in event_data:
                    # New format with nested five_w1h
                    event = Event(
                        id=event_data.get('event_id', event_data.get('id', str(uuid.uuid4()))),
                        five_w1h=FiveW1H(
                            who=event_data['five_w1h'].get('who', 'Unknown'),
                            what=event_data['five_w1h'].get('what', ''),
                            when=event_data['five_w1h'].get('when', datetime.utcnow().isoformat()),
                            where=event_data['five_w1h'].get('where', 'unspecified'),
                            why=event_data['five_w1h'].get('why', ''),
                            how=event_data['five_w1h'].get('how', '')
                        ),
                        event_type=EventType(event_data.get('event_type', 'observation')),
                        episode_id=event_data.get('episode_id', event_data.get('conversation_id', 'unknown'))
                    )
                else:
                    # Old format with flat structure
                    event = Event(
                        id=event_data.get('id', str(uuid.uuid4())),
                        five_w1h=FiveW1H(
                            who=event_data.get('who', 'Unknown'),
                            what=event_data.get('what', ''),
                            when=event_data.get('when', datetime.utcnow().isoformat()),
                            where=event_data.get('where', 'unspecified'),
                            why=event_data.get('why', ''),
                            how=event_data.get('how', '')
                        ),
                        event_type=EventType(event_data.get('event_type', 'observation')),
                        episode_id=event_data.get('conversation_id', event_data.get('episode_id', 'unknown'))
                    )
                
                # Store the event - memory_store handles everything internally
                success, message = memory_agent.memory_store.store_event(event, preserve_raw=True)
                
                if success:
                    successful += 1
                    if 'Merged' in message:
                        merged_count += 1
                    
                    # Collect embeddings for batch cache update if available
                    if hasattr(memory_agent.memory_store, 'embedding_cache') and event.id in memory_agent.memory_store.embedding_cache:
                        batch_embeddings[event.id] = memory_agent.memory_store.embedding_cache[event.id]
                else:
                    failed += 1
                    print(f"  Failed to store event {i}: {message}")
                    
            except Exception as e:
                failed += 1
                print(f"  Error processing event {i}: {e}")
        
        # Update similarity cache after each batch
        if compute_cache and batch_embeddings and hasattr(memory_agent.memory_store, 'similarity_cache'):
            print(f"  Updating similarity cache for {len(batch_embeddings)} events...")
            cache_start = time.time()
            memory_agent.memory_store.similarity_cache.batch_add_embeddings(batch_embeddings)
            cache_time = time.time() - cache_start
            print(f"  Cache updated in {cache_time:.2f}s")
            batch_embeddings.clear()
        
        # Progress update
        if batch_end < total_events:
            stats = memory_agent.memory_store.get_statistics()
            print(f"  Progress: {batch_end}/{total_events} events processed")
            print(f"  Stored: {successful} (Merged: {merged_count})")
            if 'total_raw_events' in stats:
                print(f"  Raw events: {stats['total_raw_events']}")
            if 'total_events' in stats:
                print(f"  Merged events: {stats['total_events']}")
    
    # Build complete similarity cache if needed
    if compute_cache and hasattr(memory_agent.memory_store, 'embedding_cache'):
        print("\nBuilding complete similarity cache...")
        cache_start = time.time()
        
        try:
            all_embeddings = {}
            for event_id, embeddings in memory_agent.memory_store.embedding_cache.items():
                if 'euclidean_anchor' in embeddings and 'hyperbolic_anchor' in embeddings:
                    all_embeddings[event_id] = {
                        'euclidean': embeddings['euclidean_anchor'],
                        'hyperbolic': embeddings['hyperbolic_anchor']
                    }
            
            if all_embeddings:
                if not hasattr(memory_agent.memory_store, 'similarity_cache'):
                    memory_agent.memory_store.similarity_cache = SimilarityCache(similarity_threshold=0.2)
                
                memory_agent.memory_store.similarity_cache.batch_add_embeddings(all_embeddings)
                
                if hasattr(memory_agent.memory_store, 'chromadb') and hasattr(memory_agent.memory_store.chromadb, '_save_similarity_cache'):
                    memory_agent.memory_store.chromadb._save_similarity_cache()
                
                cache_time = time.time() - cache_start
                cache_stats = memory_agent.memory_store.similarity_cache.stats
                print(f"  Similarity cache built in {cache_time:.2f}s")
                print(f"  Cached pairs: {cache_stats.get('cached_pairs', 0)}")
                print(f"  Total embeddings cached: {len(all_embeddings)}")
            else:
                print("  No embeddings found to cache")
        except Exception as e:
            print(f"  Warning: Could not build similarity cache: {e}")
    
    # Get final statistics
    stats = memory_agent.memory_store.get_statistics()
    
    # Get dimensional merge statistics from multi_merger
    merge_stats = {}
    if hasattr(memory_agent.memory_store, 'multi_merger'):
        multi_merger = memory_agent.memory_store.multi_merger
        print("\nMulti-Dimensional Merge Statistics:")
        for merge_type in multi_merger.merge_groups:
            groups = multi_merger.merge_groups[merge_type]
            if groups:
                merge_stats[merge_type.value] = {
                    'groups': len(groups),
                    'groups_details': []
                }
                print(f"\n  {merge_type.value.upper():12} - {len(groups)} groups")
                # Show first few groups
                for gid, group in list(groups.items())[:3]:
                    event_count = len(group.get('events', []))
                    merge_stats[merge_type.value]['groups_details'].append({
                        'id': gid,
                        'key': group.get('key', ''),
                        'event_count': event_count
                    })
                    print(f"    - {gid}: {event_count} events, key='{group.get('key', '')}'")
                if len(groups) > 3:
                    print(f"    ... and {len(groups) - 3} more groups")
    
    # Compile return statistics
    final_stats = {
        'successful': successful,
        'failed': failed,
        'merged_count': merged_count,
        'memory_store_stats': stats,
        'merge_groups': merge_stats
    }
    
    print("\n" + "="*60)
    print("LOADING COMPLETE!")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Total events in file: {total_events}")
    print(f"Successfully processed: {successful}")
    print(f"Failed to store: {failed}")
    print(f"Events merged with existing: {merged_count}")
    print(f"\nMemory Store Statistics:")
    print(f"  Total merged events (deduplicated): {stats.get('total_events', 0)}")
    print(f"  Total raw events (all preserved): {stats.get('total_raw_events', 0)}")
    
    if hasattr(memory_agent.memory_store, 'similarity_cache'):
        cache_stats = memory_agent.memory_store.similarity_cache.stats
        print(f"\nSimilarity Cache Statistics:")
        print(f"  Cached pairs: {cache_stats.get('cached_pairs', 0)}")
        print(f"  Cache threshold: {memory_agent.memory_store.similarity_cache.similarity_threshold}")
        print(f"  Total embeddings: {len(all_embeddings) if 'all_embeddings' in locals() else 0}")
    
    print("="*60)
    
    return successful, failed, final_stats

def main():
    """Main function to load benchmark dataset"""
    import argparse
    parser = argparse.ArgumentParser(description='Load benchmark dataset into memory system')
    parser.add_argument('--dataset', type=str, help='Path to specific dataset file')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--no-cache', action='store_true', help='Skip similarity cache computation')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before loading')
    args = parser.parse_args()
    
    # Initialize configuration
    config = get_config()
    
    # Handle database cleanup if needed
    if args.clear:
        print("Clearing existing ChromaDB database...")
        import shutil
        chromadb_path = Path(config.storage.chromadb_path)
        if chromadb_path.exists():
            try:
                shutil.rmtree(chromadb_path)
                print(f"Removed existing database at {chromadb_path}")
            except Exception as e:
                print(f"Warning: Could not remove database: {e}")
                print("Please manually delete the directory and try again.")
                return
    
    # Create memory agent
    print("Initializing Memory Agent...")
    try:
        memory_agent = MemoryAgent(config)
    except Exception as e:
        if "no such column: collections.topic" in str(e):
            print("\nERROR: Incompatible ChromaDB database detected!")
            print("Please run with --clear flag to remove the old database:")
            print("  python benchmark/load_benchmark.py --clear")
            return
        else:
            raise
    
    # Check for existing data
    initial_stats = memory_agent.memory_store.get_statistics()
    existing_events = initial_stats.get('total_events', 0)
    existing_raw = initial_stats.get('total_raw_events', 0)
    
    if (existing_events > 0 or existing_raw > 0) and not args.clear:
        print(f"\nWARNING: Memory store already contains data:")
        print(f"  - {existing_raw} raw events")
        print(f"  - {existing_events} merged events")
        response = input("Do you want to continue and add more events? (y/n): ").lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Find the dataset to load
    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset file not found: {dataset_path}")
            return
    else:
        # Find the most recent benchmark dataset
        benchmark_dir = Path("benchmark_datasets")
        if not benchmark_dir.exists():
            print(f"Error: {benchmark_dir} directory not found")
            print("Please run the benchmark generator first to create a dataset.")
            return
        
        benchmark_files = sorted([f for f in benchmark_dir.glob("benchmark_*.json") 
                                 if not f.name.endswith("_queries.json")])
        if not benchmark_files:
            print("No benchmark datasets found in benchmark_datasets/")
            print("Please run the benchmark generator first.")
            return
        
        dataset_path = benchmark_files[-1]
        print(f"\nUsing most recent benchmark: {dataset_path.name}")
    
    # Load the dataset
    successful, failed, stats = load_benchmark_dataset(
        str(dataset_path),
        memory_agent,
        batch_size=args.batch_size,
        compute_cache=not args.no_cache
    )
    
    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dataset successfully loaded: {dataset_path.name}")
    print(f"Events processed: {successful}/{successful + failed}")
    
    if not args.no_cache and hasattr(memory_agent.memory_store, 'similarity_cache'):
        cache_stats = memory_agent.memory_store.similarity_cache.stats
        if cache_stats.get('cached_pairs', 0) > 0:
            print(f"Similarity cache ready: {cache_stats['cached_pairs']} pairs cached")
    
    print(f"\nThe system is now ready with:")
    print(f"  - Raw events: All original events preserved")
    print(f"  - Merged events: Deduplicated for efficient display")
    print(f"  - Multi-dimensional merges: Handled automatically by memory system")
    if not args.no_cache:
        print(f"  - Similarity cache: Pre-computed for fast retrieval")
    
    print(f"\nYou can now:")
    print(f"  1. Run 'python run_web.py' to explore the data visually")
    print(f"  2. Run 'python benchmark/evaluator.py' to test performance")
    print(f"  3. Use the CLI to query: python cli.py recall --what 'your query'")

if __name__ == "__main__":
    main()
