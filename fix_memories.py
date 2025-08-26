#!/usr/bin/env python
"""
Fix existing memories by migrating 5W1H data from documents to metadata
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from memory.chromadb_store import ChromaDBStore
from config import get_config

def fix_memories():
    """Migrate 5W1H data from document field to metadata"""
    
    print("Fixing memory metadata...")
    
    # Initialize ChromaDB
    config = get_config()
    store = ChromaDBStore(config)
    
    # Get all events
    try:
        results = store.events_collection.get(include=["documents", "metadatas"])
        
        if not results['ids']:
            print("No memories found in database")
            return
        
        print(f"Found {len(results['ids'])} memories to fix")
        
        fixed_count = 0
        for i, doc_str in enumerate(results['documents']):
            event_id = results['ids'][i]
            metadata = results['metadatas'][i]
            
            # Parse the document JSON
            try:
                doc_data = json.loads(doc_str)
                
                # Check if we need to update
                needs_update = False
                updated_metadata = metadata.copy()
                
                # Add missing 5W1H fields to metadata
                for field in ['what', 'when', 'why', 'how']:
                    if field not in metadata or not metadata[field]:
                        if field in doc_data and doc_data[field]:
                            updated_metadata[field] = doc_data[field]
                            needs_update = True
                
                if needs_update:
                    # Update the metadata in ChromaDB
                    store.events_collection.update(
                        ids=[event_id],
                        metadatas=[updated_metadata]
                    )
                    fixed_count += 1
                    print(f"Fixed memory {event_id[:8]}...")
                    
            except json.JSONDecodeError as e:
                print(f"Could not parse document for {event_id}: {e}")
                continue
        
        print(f"\nFixed {fixed_count} memories")
        
        # Verify the fix
        print("\nVerifying...")
        results = store.events_collection.get(include=["metadatas"], limit=5)
        for i, metadata in enumerate(results['metadatas']):
            print(f"\nMemory {i+1}:")
            print(f"  Who: {metadata.get('who', 'N/A')}")
            print(f"  What: {metadata.get('what', 'N/A')[:50]}..." if metadata.get('what') else "  What: N/A")
            print(f"  When: {metadata.get('when', 'N/A')}")
            print(f"  Where: {metadata.get('where', 'N/A')}")
            print(f"  Why: {metadata.get('why', 'N/A')[:50]}..." if metadata.get('why') else "  Why: N/A")
            print(f"  How: {metadata.get('how', 'N/A')[:50]}..." if metadata.get('how') else "  How: N/A")
        
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return

if __name__ == "__main__":
    fix_memories()