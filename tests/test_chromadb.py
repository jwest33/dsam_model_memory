#!/usr/bin/env python3
"""Check ChromaDB directly to see what's stored"""

import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path="./state/chromadb",
    settings=Settings(anonymized_telemetry=False)
)

# Get the events collection
try:
    collection = client.get_collection("events")
    
    # Get all events (up to 100 for analysis)
    results = collection.get(
        limit=100,
        include=["metadatas"]
    )
    
    print(f"Total events in collection: {collection.count()}")
    print(f"Retrieved {len(results['ids'])} events for analysis")
    print("=" * 80)
    
    # Analyze field population
    field_counts = {
        'who': 0,
        'what': 0, 
        'when': 0,
        'where': 0,
        'why': 0,
        'how': 0
    }
    
    concrete_total = 0
    abstract_total = 0
    
    # Sample first few events
    print("\nFirst 3 events:")
    print("-" * 80)
    
    for i in range(min(3, len(results['metadatas']))):
        metadata = results['metadatas'][i]
        print(f"\nEvent {i+1}:")
        for key in ['who', 'what', 'when', 'where', 'why', 'how']:
            value = metadata.get(key, '')
            if value:
                display_value = value[:60] + "..." if len(value) > 60 else value
                print(f"  {key:6s}: {display_value}")
    
    # Count all fields
    for metadata in results['metadatas']:
        # Count and score concrete fields
        if metadata.get('who', '').strip():
            field_counts['who'] += 1
            concrete_total += 1.0
        if metadata.get('what', '').strip():
            field_counts['what'] += 1
            concrete_total += 2.0
        if metadata.get('when', '').strip():
            field_counts['when'] += 1
            concrete_total += 0.5
        if metadata.get('where', '').strip():
            field_counts['where'] += 1
            concrete_total += 0.5
        
        # Count and score abstract fields
        if metadata.get('why', '').strip():
            field_counts['why'] += 1
            abstract_total += 1.5
        if metadata.get('how', '').strip():
            field_counts['how'] += 1
            abstract_total += 1.0
    
    print("\n" + "=" * 80)
    print("Field Population Statistics:")
    print("-" * 80)
    
    total_events = len(results['metadatas'])
    for field, count in field_counts.items():
        percentage = (count / total_events * 100) if total_events > 0 else 0
        print(f"  {field:6s}: {count:3d} / {total_events} ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Space Weight Calculation:")
    print("-" * 80)
    
    if total_events > 0:
        avg_concrete = concrete_total / total_events
        avg_abstract = abstract_total / total_events
        total = avg_concrete + avg_abstract
        
        print(f"  Concrete total score: {concrete_total:.2f}")
        print(f"  Abstract total score: {abstract_total:.2f}")
        print(f"  Avg concrete per event: {avg_concrete:.3f}")
        print(f"  Avg abstract per event: {avg_abstract:.3f}")
        print(f"  Combined average: {total:.3f}")
        
        if total > 0:
            euclidean_weight = avg_concrete / total
            hyperbolic_weight = avg_abstract / total
            
            print(f"\n  EXPECTED WEIGHTS:")
            print(f"  Euclidean:  {euclidean_weight:.3f} ({euclidean_weight*100:.1f}%)")
            print(f"  Hyperbolic: {hyperbolic_weight:.3f} ({hyperbolic_weight*100:.1f}%)")
        else:
            print("\n  Total is 0 - would use default 50/50")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to list available collections...")
    collections = client.list_collections()
    for col in collections:
        print(f"  - {col.name}")
