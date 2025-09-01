#!/usr/bin/env python3
"""Test temporal query retrieval."""

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

if result['memory_details']:
    for i, memory in enumerate(result['memory_details'][:3]):
        print(f"\nMemory {i+1}:")
        print(f"  ID: {memory.get('id', 'N/A')}")
        print(f"  Score: {memory.get('score', 'N/A')}")
        print(f"  When: {memory.get('when', 'N/A')}")
        print(f"  What: {memory.get('what', 'N/A')[:100]}")
        if 'recency_based' in str(memory):
            print(f"  Recency-based: True")
else:
    print("\nNo memories returned!")
    
# Try a direct temporal groups query
groups_response = requests.get('http://localhost:5000/api/merge-groups/temporal')
if groups_response.status_code == 200:
    groups = groups_response.json().get('groups', [])
    print(f"\nFound {len(groups)} temporal groups in total")
    if groups:
        # Sort by last_updated to get the most recent
        sorted_groups = sorted(groups, key=lambda x: x.get('last_updated', ''), reverse=True)
        print(f"Most recent temporal group:")
        print(f"  ID: {sorted_groups[0]['id']}")
        print(f"  Last updated: {sorted_groups[0].get('last_updated', 'N/A')}")
        print(f"  Events count: {sorted_groups[0].get('events_count', 'N/A')}")
