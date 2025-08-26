#!/usr/bin/env python
"""
Clear the ChromaDB database
"""

import sys
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config

def clear_database():
    """Clear the ChromaDB database"""
    
    config = get_config()
    db_path = Path(config.storage.chromadb_path)
    
    print(f"Clearing ChromaDB at: {db_path}")
    
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
            print("✅ ChromaDB cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing ChromaDB: {e}")
    else:
        print("📁 ChromaDB directory doesn't exist")
    
    # Also clear any state files
    state_dir = Path("state")
    if state_dir.exists():
        for file in state_dir.glob("*.json"):
            try:
                file.unlink()
                print(f"✅ Removed state file: {file.name}")
            except Exception as e:
                print(f"❌ Error removing {file.name}: {e}")

if __name__ == "__main__":
    response = input("Are you sure you want to clear the database? (yes/no): ")
    if response.lower() == "yes":
        clear_database()
    else:
        print("Cancelled")