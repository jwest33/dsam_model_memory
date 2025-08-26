#!/usr/bin/env python
"""
Clear memories from ChromaDB while server is running
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from memory.chromadb_store import ChromaDBStore
from config import get_config

def clear_memories():
    """Clear all memories from ChromaDB collections"""
    
    print("Clearing memories from ChromaDB...")
    
    # Initialize ChromaDB
    config = get_config()
    store = ChromaDBStore(config)
    
    try:
        # Get all event IDs
        results = store.events_collection.get()
        if results['ids']:
            print(f"Found {len(results['ids'])} events to delete")
            # Delete all events
            store.events_collection.delete(ids=results['ids'])
            print("✅ Events cleared")
        else:
            print("No events to clear")
        
        # Clear blocks collection
        results = store.blocks_collection.get()
        if results['ids']:
            print(f"Found {len(results['ids'])} blocks to delete")
            store.blocks_collection.delete(ids=results['ids'])
            print("✅ Blocks cleared")
        else:
            print("No blocks to clear")
            
        # Clear metadata collection
        results = store.metadata_collection.get()
        if results['ids']:
            print(f"Found {len(results['ids'])} metadata entries to delete")
            store.metadata_collection.delete(ids=results['ids'])
            print("✅ Metadata cleared")
        else:
            print("No metadata to clear")
            
        print("\n✅ All collections cleared successfully!")
        
    except Exception as e:
        print(f"❌ Error clearing collections: {e}")

if __name__ == "__main__":
    response = input("Are you sure you want to clear all memories? (yes/no): ")
    if response.lower() == "yes":
        clear_memories()
    else:
        print("Cancelled")