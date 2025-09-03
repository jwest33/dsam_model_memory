#!/usr/bin/env python
"""
Clear memories from Qdrant storage while server is running
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from memory.qdrant_store import QdrantStore
from config import get_config

def clear_memories():
    """Clear all memories from Qdrant collections"""
    
    print("Clearing memories from Qdrant...")
    
    # Initialize Qdrant
    config = get_config()
    store = QdrantStore(config)
    
    try:
        # Clear all collections
        store.clear_all()
        print("\nAll collections cleared successfully!")
        
    except Exception as e:
        print(f"Error clearing collections: {e}")

if __name__ == "__main__":
    response = input("Are you sure you want to clear all memories? (yes/no): ")
    if response.lower() == "yes":
        clear_memories()
    else:
        print("Cancelled")
