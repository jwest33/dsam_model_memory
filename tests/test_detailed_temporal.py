#!/usr/bin/env python3
"""Test temporal query retrieval in detail."""

import requests
import json

# Make the request
response = requests.post('http://localhost:5000/api/chat', 
    json={'message': 'what is the last thing we discussed', 'merge_dimension': 'temporal'})

print('Status:', response.status_code)
result = response.json()
print('Memories used:', result['memories_used'])
print('Dimension weights:', result['dimension_weights'])
print('Dominant dimension:', result.get('dominant_dimension', 'N/A'))

# Check if recency-based retrieval is working
if result['memory_details']:
    print("\n=== Retrieved Memory Groups ===")
    for i, memory in enumerate(result['memory_details'][:5]):
        print(f"\nMemory {i+1}:")
        print(f"  ID: {memory.get('id', 'N/A')}")
        print(f"  Score: {memory.get('score', 'N/A'):.3f}")
        print(f"  When: {memory.get('when', 'N/A')}")
        print(f"  Who: {memory.get('who', 'N/A')}")
        print(f"  What: {memory.get('what', 'N/A')[:100]}...")
        # Check if this is marked as recency-based
        context = memory.get('context', '')
        if 'recency_based' in str(memory) or 'Recency' in context:
            print(f"  >>> RECENCY-BASED RETRIEVAL <<<")
else:
    print("\nNo memories returned!")

# Get all temporal groups sorted by recency
print("\n=== All Temporal Groups (sorted by recency) ===")
groups_response = requests.get('http://localhost:5000/api/merge-groups/temporal')
if groups_response.status_code == 200:
    groups = groups_response.json().get('groups', [])
    # Sort by last_updated to get the most recent
    sorted_groups = sorted(groups, key=lambda x: x.get('last_updated', ''), reverse=True)
    
    print(f"Total temporal groups: {len(groups)}")
    print("\nTop 5 most recent groups:")
    for i, group in enumerate(sorted_groups[:5]):
        print(f"\n{i+1}. ID: {group['id']}")
        print(f"   Last updated: {group.get('last_updated', 'N/A')}")
        print(f"   Events: {group.get('events_count', 'N/A')}")
        latest = group.get('latest_state', {})
        if latest:
            print(f"   Latest what: {latest.get('what', 'N/A')[:80]}...")
            
# Check if memory context is properly formatted
print("\n=== Memory Context Preview ===")
if result.get('memory_context'):
    print(result['memory_context'][:500] + "..." if len(result['memory_context']) > 500 else result['memory_context'])
