#!/usr/bin/env python
"""Quick test of LLM dataset generation with minimal settings"""

from benchmark.generate_benchmark_dataset_fast import FastBenchmarkDatasetGenerator
import time

# Create generator with local LLM
generator = FastBenchmarkDatasetGenerator(
    output_dir="./test_datasets",
    use_local_llm=True
)

print("Generating tiny test dataset with LLM...")
print("This will create just 3 conversations with 2 exchanges each")
print("-" * 60)

start = time.time()

# Generate tiny dataset for testing
dataset = generator.generate_dataset(
    num_conversations=3,        # Just 3 conversations
    time_span_days=7,           # Last week
    dataset_name="test_llm_tiny",
    num_threads=1,              # Single thread for now
    technical_ratio=0.6,        # 60% technical
    exchanges_per_conversation=(2, 2)  # Exactly 2 exchanges (4 LLM calls per conversation)
)

elapsed = time.time() - start

print(f"\nGeneration took {elapsed:.1f} seconds")
print(f"Average: {elapsed/3:.1f} seconds per conversation")
print(f"Average: {elapsed/12:.1f} seconds per LLM call")

# Show the generated content
if dataset['events']:
    print("\n" + "="*60)
    print("GENERATED CONVERSATIONS")
    print("="*60)
    
    current_conv = None
    for event in dataset['events']:
        # Check if new conversation based on time gap
        who = event['who']
        what = event['what']
        
        print(f"\n[{event['when'][:19]}] {who}:")
        # Show full text for this small dataset
        print(f"  {what}")
        
print("\nTest complete!")
