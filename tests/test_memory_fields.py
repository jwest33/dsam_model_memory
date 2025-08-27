#!/usr/bin/env python3
"""Check what fields are actually populated in the memory database"""

from agent.memory_agent import MemoryAgent
from config import get_config
import json

def check_memory_fields():
    config = get_config()
    agent = MemoryAgent(config)
    
    # Get all memories
    memories = agent.recall(what='', k=200)  # Get up to 200 memories
    
    print(f"Total memories found: {len(memories)}")
    print("-" * 80)
    
    # Statistics
    field_counts = {
        'who': 0,
        'what': 0,
        'when': 0,
        'where': 0,
        'why': 0,
        'how': 0
    }
    
    # Sample some memories
    print("First 5 memories with their fields:")
    print("-" * 80)
    
    for i, (memory, score) in enumerate(memories[:5]):
        print(f"\nMemory {i+1} (Score: {score:.3f}):")
        print(f"  ID: {memory.id[:8]}...")
        
        # Check each field
        fields = {
            'who': memory.five_w1h.who,
            'what': memory.five_w1h.what,
            'when': memory.five_w1h.when,
            'where': memory.five_w1h.where,
            'why': memory.five_w1h.why,
            'how': memory.five_w1h.how
        }
        
        for field_name, field_value in fields.items():
            if field_value:
                print(f"  {field_name}: {field_value[:50]}..." if len(str(field_value)) > 50 else f"  {field_name}: {field_value}")
        
        print()
    
    # Count fields across all memories
    concrete_total = 0
    abstract_total = 0
    
    for memory, _ in memories:
        if memory.five_w1h.who:
            field_counts['who'] += 1
            concrete_total += 1.0
        if memory.five_w1h.what:
            field_counts['what'] += 1
            concrete_total += 2.0  # What is weighted more
        if memory.five_w1h.when:
            field_counts['when'] += 1
            concrete_total += 0.5
        if memory.five_w1h.where:
            field_counts['where'] += 1
            concrete_total += 0.5
        if memory.five_w1h.why:
            field_counts['why'] += 1
            abstract_total += 1.5
        if memory.five_w1h.how:
            field_counts['how'] += 1
            abstract_total += 1.0
    
    print("-" * 80)
    print("Field population statistics:")
    print("-" * 80)
    
    for field, count in field_counts.items():
        percentage = (count / len(memories)) * 100 if memories else 0
        print(f"  {field:6s}: {count:3d} / {len(memories)} ({percentage:.1f}%)")
    
    print("-" * 80)
    print("Space weight calculation:")
    print("-" * 80)
    
    if len(memories) > 0:
        avg_concrete = concrete_total / len(memories)
        avg_abstract = abstract_total / len(memories)
        total = avg_concrete + avg_abstract
        
        print(f"  Total concrete score: {concrete_total:.2f}")
        print(f"  Total abstract score: {abstract_total:.2f}")
        print(f"  Avg concrete per memory: {avg_concrete:.3f}")
        print(f"  Avg abstract per memory: {avg_abstract:.3f}")
        print(f"  Total avg: {total:.3f}")
        
        if total > 0:
            euclidean_pct = (avg_concrete / total) * 100
            hyperbolic_pct = (avg_abstract / total) * 100
            print(f"\n  Expected Euclidean weight: {euclidean_pct:.1f}%")
            print(f"  Expected Hyperbolic weight: {hyperbolic_pct:.1f}%")
        else:
            print("\n  Total is 0 - would default to 50/50")
    else:
        print("  No memories found!")

if __name__ == "__main__":
    print("Checking memory field population in database...")
    print("=" * 80)
    check_memory_fields()