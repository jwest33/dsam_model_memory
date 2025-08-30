"""
Enhanced benchmark dataset loader with raw event preservation and similarity cache
Properly loads events into raw/merged collections and builds similarity cache
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
from memory.multi_dimensional_merger import MultiDimensionalMerger
from models.event import Event, FiveW1H, EventType
from models.merge_types import MergeType
from config import get_config
from agent.memory_agent import MemoryAgent
import uuid
from datetime import datetime

def load_benchmark_with_full_processing(
    dataset_path: str, 
    memory_agent: MemoryAgent,
    multi_merger: MultiDimensionalMerger = None,
    batch_size: int = 50,
    compute_cache: bool = True
) -> Tuple[int, int, Dict]:
    """
    Load benchmark dataset with full processing:
    - Store all events as raw events
    - Create merged events for deduplication
    - Create multi-dimensional merges (actor, temporal, conceptual, spatial)
    - Build similarity cache for efficient retrieval
    
    Args:
        dataset_path: Path to the benchmark JSON file
        memory_agent: MemoryAgent instance
        multi_merger: MultiDimensionalMerger instance (optional)
        batch_size: Number of events to process before updating cache
        compute_cache: Whether to compute similarity cache
        
    Returns:
        Tuple of (successful_count, failed_count, statistics)
    """
    print(f"\nLoading benchmark dataset from: {dataset_path}")
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    total_events = len(data['events'])
    print(f"Found {total_events} events in dataset")
    print(f"Processing in batches of {batch_size} for efficiency")
    
    # Track statistics
    successful = 0
    failed = 0
    merged_count = 0
    raw_events_stored = 0
    batch_embeddings = {}
    multi_merge_stats = {
        MergeType.ACTOR: 0,
        MergeType.TEMPORAL: 0,
        MergeType.CONCEPTUAL: 0,
        MergeType.SPATIAL: 0,
        MergeType.HYBRID: 0
    }
    
    # Process events in batches
    for batch_start in range(0, total_events, batch_size):
        batch_end = min(batch_start + batch_size, total_events)
        batch_events = data['events'][batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start+1}-{batch_end} of {total_events}")
        
        # Process each event in the batch
        for i, event_data in enumerate(batch_events, start=batch_start):
            try:
                # Create Event object with original ID or generate new one
                event_id = event_data.get('event_id', str(uuid.uuid4()))
                event = Event(
                    id=event_id,
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
                
                # Store event - this handles both raw and merged storage
                success, message = memory_agent.memory_store.store_event(event)
                
                if success:
                    successful += 1
                    raw_events_stored += 1
                    
                    # Check if it was merged
                    if "merged" in message.lower() or "updated" in message.lower():
                        merged_count += 1
                    
                    # Process multi-dimensional merges if merger available
                    if multi_merger:
                        # Get embeddings - try multiple sources
                        embeddings = None
                        
                        # First check memory store's embedding cache
                        if hasattr(memory_agent.memory_store, 'embedding_cache'):
                            # Check for embeddings with both the original ID and raw_ prefix
                            embedding_id = None
                            if event.id in memory_agent.memory_store.embedding_cache:
                                embedding_id = event.id
                            elif f"raw_{event.id}" in memory_agent.memory_store.embedding_cache:
                                embedding_id = f"raw_{event.id}"
                            
                            if embedding_id:
                                embeddings = memory_agent.memory_store.embedding_cache[embedding_id]
                        
                        # If no embeddings found, generate them
                        if embeddings is None:
                            # Generate embeddings using the dual-space encoder
                            raw_embeddings = memory_agent.memory_store.encoder.encode(event.five_w1h.to_dict())
                            
                            # Ensure proper structure for multi-dimensional merger
                            embeddings = {
                                'euclidean_anchor': raw_embeddings.get('euclidean_anchor', raw_embeddings.get('euclidean', None)),
                                'hyperbolic_anchor': raw_embeddings.get('hyperbolic_anchor', raw_embeddings.get('hyperbolic', None))
                            }
                            
                            # Store in cache for future use
                            if hasattr(memory_agent.memory_store, 'embedding_cache'):
                                memory_agent.memory_store.embedding_cache[event.id] = embeddings
                        
                        # Process with multi-dimensional merger
                        merge_assignments = multi_merger.process_new_event(event, embeddings)
                        
                        # Track which dimensions got merges
                        for merge_type in merge_assignments:
                            multi_merge_stats[merge_type] += 1
                    
                    # Collect embeddings for batch cache update if available
                    if hasattr(memory_agent.memory_store, 'embedding_cache') and event.id in memory_agent.memory_store.embedding_cache:
                        # Get the embeddings from memory store's cache
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
            # Get all embeddings from memory store's embedding cache
            all_embeddings = {}
            
            # The memory_store.embedding_cache contains properly structured embeddings
            # with euclidean_anchor and hyperbolic_anchor keys
            for event_id, embeddings in memory_agent.memory_store.embedding_cache.items():
                if 'euclidean_anchor' in embeddings and 'hyperbolic_anchor' in embeddings:
                    # For similarity cache, we need euclidean and hyperbolic embeddings
                    # The anchors are the base embeddings used for similarity computation
                    all_embeddings[event_id] = {
                        'euclidean': embeddings['euclidean_anchor'],
                        'hyperbolic': embeddings['hyperbolic_anchor']
                    }
            
            if all_embeddings:
                # Initialize similarity cache if not exists
                if not hasattr(memory_agent.memory_store, 'similarity_cache'):
                    memory_agent.memory_store.similarity_cache = SimilarityCache(similarity_threshold=0.2)
                
                # Batch add all embeddings to similarity cache
                memory_agent.memory_store.similarity_cache.batch_add_embeddings(all_embeddings)
                
                # Persist cache to ChromaDB if method available
                if hasattr(memory_agent.memory_store, 'chromadb') and hasattr(memory_agent.memory_store.chromadb, '_save_similarity_cache'):
                    memory_agent.memory_store.chromadb._save_similarity_cache()
                
                cache_time = time.time() - cache_start
                cache_stats = memory_agent.memory_store.similarity_cache.stats
                print(f"  Similarity cache built in {cache_time:.2f}s")
                print(f"  Cached pairs: {cache_stats.get('cached_pairs', 0)}")
                print(f"  Total embeddings cached: {len(all_embeddings)}")
                print(f"  Computation time: {cache_stats.get('computation_time_ms', 0):.1f}ms")
            else:
                print("  No embeddings found to cache")
        except Exception as e:
            print(f"  Warning: Could not build similarity cache: {e}")
    
    # Get final statistics
    stats = memory_agent.memory_store.get_statistics()
    
    # Compile return statistics
    final_stats = {
        'total_events_in_file': total_events,
        'successful': successful,
        'failed': failed,
        'merged_count': merged_count,
        'raw_events_stored': raw_events_stored,
        'memory_store_stats': stats,
        'multi_merge_stats': multi_merge_stats if multi_merger else None
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
    if 'total_events' in stats:
        print(f"  Total merged events (deduplicated): {stats['total_events']}")
    if 'total_raw_events' in stats:
        print(f"  Total raw events (all preserved): {stats['total_raw_events']}")
    if 'total_merged_groups' in stats:
        print(f"  Total merge groups: {stats['total_merged_groups']}")
    if 'average_merge_size' in stats:
        print(f"  Average events per merge group: {stats['average_merge_size']:.2f}")
    
    # Show multi-dimensional merge statistics if available
    if multi_merger:
        print(f"\nMulti-Dimensional Merge Statistics:")
        for merge_type in MergeType:
            if merge_type == MergeType.HYBRID:
                continue  # Skip HYBRID type as it's not used directly
            type_groups = multi_merger.merge_groups.get(merge_type, {})
            group_count = len(type_groups)
            events_merged = multi_merge_stats.get(merge_type, 0)
            
            # Show details for each merge group
            print(f"\n  {merge_type.name:12} - Groups: {group_count:4}, Events merged: {events_merged:4}")
            
            # List the first few groups for verification
            if type_groups:
                for i, (merge_id, group_data) in enumerate(list(type_groups.items())[:3]):
                    merged_event = group_data.get('merged_event')
                    if merged_event:
                        print(f"    - {merge_id}: {merged_event.merge_count} events, key='{group_data.get('key', 'N/A')}'")
                if len(type_groups) > 3:
                    print(f"    ... and {len(type_groups) - 3} more groups")
    
    # Show similarity cache stats if available
    if hasattr(memory_agent.memory_store, 'similarity_cache'):
        cache_stats = memory_agent.memory_store.similarity_cache.stats
        print(f"\nSimilarity Cache Statistics:")
        print(f"  Cached pairs: {cache_stats.get('cached_pairs', 0)}")
        print(f"  Cache threshold: {memory_agent.memory_store.similarity_cache.similarity_threshold}")
        print(f"  Total embeddings: {len(memory_agent.memory_store.similarity_cache.embeddings)}")
    
    print("="*60)
    
    return successful, failed, final_stats

def main():
    """Main function to reload benchmark dataset with enhanced processing"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Load benchmark dataset into memory system')
    parser.add_argument('--dataset', type=str, help='Path to specific dataset file')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--no-cache', action='store_true', help='Skip similarity cache computation')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before loading')
    args = parser.parse_args()
    
    # Initialize configuration
    config = get_config()
    
    # Create memory agent
    print("Initializing Memory Agent...")
    memory_agent = MemoryAgent(config)
    
    # Initialize multi-dimensional merger (create once and reuse across all batches)
    print("Initializing Multi-Dimensional Merger...")
    multi_merger = MultiDimensionalMerger(
        chromadb_store=memory_agent.memory_store.chromadb if hasattr(memory_agent.memory_store, 'chromadb') else None
    )
    
    # Attach to memory store so it persists across batches
    if hasattr(memory_agent.memory_store, 'chromadb'):
        memory_agent.memory_store.multi_merger = multi_merger
    
    # Check for existing data
    initial_stats = memory_agent.memory_store.get_statistics()
    existing_events = initial_stats.get('total_events', 0)
    existing_raw = initial_stats.get('total_raw_events', 0)
    
    if (existing_events > 0 or existing_raw > 0) and not args.clear:
        print(f"\nWARNING: Memory store already contains data:")
        print(f"  - {existing_raw} raw events")
        print(f"  - {existing_events} merged events")
        response = input("Do you want to continue and add more events? (y/n/clear): ").lower()
        if response == 'clear':
            args.clear = True
        elif response != 'y':
            print("Aborted.")
            return
    
    # Clear if requested
    if args.clear:
        print("\nClearing existing memory store...")
        if hasattr(memory_agent.memory_store, 'clear'):
            memory_agent.memory_store.clear()
            print("Memory store cleared.")
        else:
            print("Warning: Could not clear memory store (method not available)")
    
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
        
        # Get all benchmark files (exclude query files)
        benchmark_files = sorted([f for f in benchmark_dir.glob("benchmark_*.json") 
                                 if not f.name.endswith("_queries.json")])
        if not benchmark_files:
            print("No benchmark datasets found in benchmark_datasets/")
            print("Please run the benchmark generator first.")
            return
        
        # Use the most recent one
        dataset_path = benchmark_files[-1]
        print(f"\nUsing most recent benchmark: {dataset_path.name}")
    
    # Load the dataset with full processing
    successful, failed, stats = load_benchmark_with_full_processing(
        str(dataset_path),
        memory_agent,
        multi_merger=multi_merger,
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
    print(f"  - Multi-dimensional merges: Actor, Temporal, Conceptual, Spatial views")
    if not args.no_cache:
        print(f"  - Similarity cache: Pre-computed for fast retrieval")
    
    print(f"\nYou can now:")
    print(f"  1. Run 'python run_web.py' to explore the data visually")
    print(f"  2. Run 'python benchmark/evaluator.py' to test performance")
    print(f"  3. Use the CLI to query: python cli.py recall --what 'your query'")

if __name__ == "__main__":
    main()