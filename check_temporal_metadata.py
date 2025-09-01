#!/usr/bin/env python3
"""Check temporal group metadata fields."""

import os
import sys
sys.path.insert(0, '.')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from memory.memory_store import MemoryStore

store = MemoryStore()
temporal_collection = store.chromadb.client.get_collection('temporal_merges')
groups = temporal_collection.get(limit=2, include=['metadatas'])

print("Checking temporal group metadata fields:\n")
for i, metadata in enumerate(groups.get('metadatas', [])):
    print(f'Temporal group {i+1} ({groups["ids"][i]}) metadata:')
    for key in sorted(metadata.keys()):
        value = metadata[key]
        if len(str(value)) > 100:
            value = str(value)[:100] + '...'
        print(f'  {key}: {value}')
    print()

print("\nKey observations:")
print("- 'created_at' field:", "present" if any('created_at' in m for m in groups.get('metadatas', [])) else "missing")
print("- 'last_updated' field:", "present" if any('last_updated' in m for m in groups.get('metadatas', [])) else "missing") 
print("- 'latest_when' field:", "present" if any('latest_when' in m for m in groups.get('metadatas', [])) else "missing")