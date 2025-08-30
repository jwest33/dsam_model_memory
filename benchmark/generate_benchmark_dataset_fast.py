"""
Fast benchmark dataset generator with batching and parallel processing
Optimized version that generates datasets much faster
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import the original classes
try:
    # Try relative import first (when run as module)
    from .generate_benchmark_dataset import (
        PersonaType, LocationType, ActivityType, ConversationScenario,
        BenchmarkDatasetGenerator
    )
except ImportError:
    # Fall back to adding parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmark.generate_benchmark_dataset import (
        PersonaType, LocationType, ActivityType, ConversationScenario,
        BenchmarkDatasetGenerator
    )

class FastBenchmarkDatasetGenerator(BenchmarkDatasetGenerator):
    """Enhanced generator with batching and parallel processing"""
    
    def __init__(self, output_dir: str = "./benchmark_datasets", batch_size: int = 100):
        super().__init__(output_dir=output_dir, use_dual_llm=False)
        self.batch_size = batch_size
        self.lock = threading.Lock()
        
    def generate_dataset_fast(
        self,
        num_conversations: int = 100,
        time_span_days: int = 30,
        dataset_name: str = None,
        num_threads: int = 1
    ) -> Dict:
        """Generate dataset with parallel processing and batching"""
        
        if dataset_name is None:
            dataset_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"FAST DATASET GENERATION")
        print(f"{'='*60}")
        print(f"Dataset name: {dataset_name}")
        print(f"Target conversations: {num_conversations}")
        print(f"Time span: {time_span_days} days")
        print(f"Batch size: {self.batch_size}")
        print(f"Parallel threads: {num_threads}")
        print(f"{'='*60}\n")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_span_days)
        
        # Thread-safe collections
        all_events = []
        conversation_metadata = []
        all_events_lock = threading.Lock()
        metadata_lock = threading.Lock()
        
        start_generation = time.time()
        
        # Generate conversation parameters in advance
        conversation_params = []
        for conv_idx in range(num_conversations):
            scenario = random.choice(self.scenarios)
            persona = random.choice(scenario.personas)
            base_time = start_time + timedelta(
                seconds=random.randint(0, int((end_time - start_time).total_seconds()))
            )
            timestamp = self.generate_synthetic_timestamp(base_time, variance_hours=24)
            
            conversation_params.append({
                'conv_idx': conv_idx,
                'scenario': scenario,
                'persona': persona,
                'timestamp': timestamp
            })
        
        # Process in batches
        num_batches = (num_conversations + self.batch_size - 1) // self.batch_size
        print(f"Processing {num_conversations} conversations in {num_batches} batches...\n")
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, num_conversations)
            batch_params = conversation_params[batch_start:batch_end]
            
            print(f"[Batch {batch_idx + 1}/{num_batches}] Processing conversations {batch_start + 1}-{batch_end}")
            batch_start_time = time.time()
            
            # Generate conversations in parallel within batch
            batch_events = []
            batch_metadata = []
            
            def process_conversation(params):
                """Process a single conversation"""
                conv_idx = params['conv_idx']
                scenario = params['scenario']
                persona = params['persona']
                timestamp = params['timestamp']
                
                # Generate conversation events
                events = self.generate_conversation_exchange(scenario, persona, timestamp)
                
                # Create metadata
                metadata = {
                    'conversation_id': conv_idx,
                    'scenario': scenario.topic,
                    'category': scenario.category,
                    'complexity': scenario.complexity,
                    'space_preference': scenario.space_preference,
                    'persona': persona.value,
                    'num_events': len(events),
                    'timestamp': timestamp
                }
                
                # Convert events to serializable format
                serialized_events = []
                for event in events:
                    serialized_events.append({
                        'event_id': event.id,
                        'event_type': event.event_type.value,
                        'five_w1h': {
                            'who': event.five_w1h.who,
                            'what': event.five_w1h.what,
                            'when': event.five_w1h.when,
                            'where': event.five_w1h.where,
                            'why': event.five_w1h.why,
                            'how': event.five_w1h.how
                        },
                        'episode_id': event.episode_id,
                        'scenario': scenario.topic,
                        'category': scenario.category,
                        'complexity': scenario.complexity,
                        'space_preference': scenario.space_preference
                    })
                
                # Update stats (inline like parent class does)
                for event in events:
                    with self.lock:  # Thread-safe stats update
                        self.stats['total_memories'] += 1
                        self.stats['by_persona'][persona.value] = self.stats['by_persona'].get(persona.value, 0) + 1
                        self.stats['by_category'][scenario.category] = self.stats['by_category'].get(scenario.category, 0) + 1
                        self.stats['space_distribution'][scenario.space_preference] += 1
                
                return serialized_events, metadata, events
            
            # Process conversations in parallel
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for params in batch_params:
                    future = executor.submit(process_conversation, params)
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(batch_params)} conversations", end='\r')
                    
                    serialized_events, metadata, raw_events = future.result()
                    batch_events.extend(serialized_events)
                    batch_metadata.append(metadata)
            
            print(f"  Generated {len(batch_events)} events in batch")
            
            # Batch store to memory system (sequential to avoid conflicts)
            print(f"  Storing batch to memory system...", end="", flush=True)
            store_start = time.time()
            stored_count = 0
            
            # Store raw events to memory system
            for params in batch_params:
                events = self.generate_conversation_exchange(
                    params['scenario'], 
                    params['persona'], 
                    params['timestamp']
                )
                for event in events:
                    success, _ = self.memory_agent.memory_store.store_event(event)
                    if success:
                        stored_count += 1
            
            store_time = time.time() - store_start
            print(f" ({stored_count} stored in {store_time:.1f}s)")
            
            # Add to global collections
            with all_events_lock:
                all_events.extend(batch_events)
            with metadata_lock:
                conversation_metadata.extend(batch_metadata)
            
            batch_time = time.time() - batch_start_time
            print(f"  Batch completed in {batch_time:.1f}s")
            
            # Estimate remaining time
            if batch_idx < num_batches - 1:
                avg_time_per_batch = batch_time
                remaining_batches = num_batches - batch_idx - 1
                estimated_remaining = avg_time_per_batch * remaining_batches
                print(f"  Estimated time remaining: {estimated_remaining:.0f}s ({estimated_remaining/60:.1f} minutes)")
            print()
        
        # Compile dataset
        dataset = {
            'metadata': {
                'name': dataset_name,
                'created_at': datetime.now().isoformat(),
                'num_conversations': num_conversations,
                'num_events': len(all_events),
                'time_span_days': time_span_days,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'generation_method': 'fast_batched'
            },
            'statistics': self.stats,
            'conversations': conversation_metadata,
            'events': all_events
        }
        
        # Save dataset
        print(f"\n{'='*60}")
        print(f"SAVING DATASET")
        print(f"{'='*60}")
        dataset_path = self.output_dir / f"{dataset_name}.json"
        print(f"  Writing to: {dataset_path}")
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # Final statistics
        total_time = time.time() - start_generation
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Summary:")
        print(f"  Total conversations: {num_conversations}")
        print(f"  Total events: {len(all_events)}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Average per conversation: {total_time/num_conversations:.3f}s")
        print(f"  Average per batch: {total_time/num_batches:.1f}s")
        
        # Calculate speedup vs sequential
        sequential_estimate = total_time * num_threads  # Rough estimate
        speedup = sequential_estimate / total_time
        print(f"\nPerformance:")
        print(f"  Estimated speedup: ~{speedup:.1f}x faster than sequential")
        print(f"  Processing rate: {len(all_events)/total_time:.1f} events/second")
        
        print(f"\nBreakdown by category:")
        for cat, count in sorted(self.stats.get('by_category', {}).items()):
            print(f"  {cat:15s}: {count:5d} events")
        
        print(f"\nDataset saved to: {dataset_path}")
        print(f"{'='*60}")
        
        return dataset

def main():
    """Main function for fast dataset generation"""
    print("="*60)
    print("FAST BENCHMARK DATASET GENERATOR")
    print("="*60)
    
    print("\nSelect dataset size:")
    print("1. Small (100 conversations)")
    print("2. Medium (500 conversations)")
    print("3. Large (1000 conversations)")
    print("4. Extra Large (2000 conversations)")
    print("5. Massive (5000 conversations)")
    print("6. Custom")
    
    choice = input("\nEnter choice (1-6) [default: 2]: ").strip() or "2"
    
    size_configs = {
        "1": (100, 7, 50, 2),
        "2": (500, 30, 100, 4),
        "3": (1000, 60, 100, 4),
        "4": (2000, 90, 200, 6),
        "5": (5000, 180, 250, 8),
    }
    
    if choice in size_configs:
        num_conversations, time_span_days, batch_size, num_threads = size_configs[choice]
    elif choice == "6":
        num_conversations = int(input("Number of conversations: "))
        time_span_days = int(input("Time span (days): "))
        batch_size = int(input("Batch size [default: 100]: ") or "100")
        num_threads = int(input("Number of threads [default: 1]: ") or "1")
    else:
        print("Invalid choice")
        return
    
    # Create generator with batch size
    generator = FastBenchmarkDatasetGenerator(batch_size=batch_size)
    
    # Generate dataset
    dataset = generator.generate_dataset_fast(
        num_conversations=num_conversations,
        time_span_days=time_span_days,
        num_threads=num_threads
    )
    
    # Generate query set
    num_queries = min(100, num_conversations // 5)
    generator.generate_query_set(dataset, num_queries=num_queries)
    
    print("\nFast generation complete!")

if __name__ == "__main__":
    main()
