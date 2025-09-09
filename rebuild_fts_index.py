#!/usr/bin/env python3
"""
Rebuild the FTS (Full-Text Search) index for the JAM memory system.
The FTS index is out of sync with the main memories table.
"""

import sqlite3
import sys
from pathlib import Path

def rebuild_fts_index(db_path: str = "./amemory.sqlite3"):
    """Rebuild the FTS index from the memories table."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current state
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mem_fts")
        current_fts = cursor.fetchone()[0]
        
        print(f"Current state:")
        print(f"  Total memories: {total_memories:,}")
        print(f"  FTS indexed: {current_fts:,}")
        print(f"  Missing: {total_memories - current_fts:,}")
        
        if total_memories == current_fts:
            print("\nFTS index is already up to date!")
            return
        
        print("\nRebuilding FTS index...")
        
        # For contentless FTS5 tables, we need to drop and recreate
        print("  Dropping old FTS table...")
        cursor.execute("DROP TABLE IF EXISTS mem_fts")
        
        # Recreate the FTS table
        print("  Recreating FTS table...")
        cursor.execute("""
            CREATE VIRTUAL TABLE mem_fts USING fts5(
                memory_id UNINDEXED,
                what,
                why,
                how,
                raw_text,
                content='',
                tokenize='porter'
            )
        """)
        
        # Rebuild from memories table
        print("  Inserting all memories into FTS index...")
        cursor.execute("""
            INSERT INTO mem_fts (memory_id, what, why, how, raw_text)
            SELECT 
                memory_id,
                COALESCE(what, ''),
                COALESCE(why, ''),
                COALESCE(how, ''),
                COALESCE(raw_text, '')
            FROM memories
        """)
        
        inserted = cursor.rowcount
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM mem_fts")
        new_fts = cursor.fetchone()[0]
        
        print(f"\nRebuild complete!")
        print(f"  Inserted: {inserted:,} entries")
        print(f"  FTS now contains: {new_fts:,} entries")
        
        # Test search
        print("\nTesting FTS search...")
        cursor.execute("""
            SELECT COUNT(*) FROM mem_fts 
            WHERE mem_fts MATCH '"user"' 
        """)
        test_count = cursor.fetchone()[0]
        print(f"  Test search for 'user' found {test_count:,} results")
        
    except Exception as e:
        print(f"\nError rebuilding FTS index: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "./amemory.sqlite3"
    
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    
    rebuild_fts_index(db_path)