"""Check what's actually stored in the multi-dimensional merge collections"""

import os
import sys
import warnings

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore')

from memory.chromadb_store import ChromaDBStore

# Initialize ChromaDB
chromadb = ChromaDBStore()

print("Checking ChromaDB collections for multi-dimensional merges...")
print("="*60)

# Check each collection
collections = {
    'actor_merges': chromadb.actor_merges_collection,
    'temporal_merges': chromadb.temporal_merges_collection,
    'conceptual_merges': chromadb.conceptual_merges_collection,
    'spatial_merges': chromadb.spatial_merges_collection
}

for name, collection in collections.items():
    if collection:
        try:
            # Get count
            count = collection.count()
            print(f"\n{name}: {count} items")
            
            # Get first few items
            if count > 0:
                results = collection.get(limit=3, include=['metadatas'])
                for i, id in enumerate(results['ids']):
                    metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                    print(f"  - {id}: key='{metadata.get('merge_key', 'N/A')}', count={metadata.get('merge_count', 0)}")
                if count > 3:
                    print(f"  ... and {count - 3} more")
        except Exception as e:
            print(f"{name}: Error - {e}")
    else:
        print(f"{name}: Collection not initialized")

print("\n" + "="*60)
print("Checking standard collections:")

# Check standard collections
try:
    events_count = chromadb.events_collection.count()
    print(f"events: {events_count} items")
except:
    print("events: Error getting count")

try:
    raw_count = chromadb.client.get_collection('raw_events').count()
    print(f"raw_events: {raw_count} items")
except:
    print("raw_events: Error getting count")