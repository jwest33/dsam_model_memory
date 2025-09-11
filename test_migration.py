#!/usr/bin/env python3
"""
Test the migration to verify data integrity and functionality
"""

import sqlite3
import json
from datetime import datetime

def test_database_structure():
    """Test that the database has the expected structure"""
    print("\n=== Testing Database Structure ===")
    
    conn = sqlite3.connect('amemory.sqlite3')
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['memories', 'mem_fts']
    missing_tables = []
    unexpected_tables = []
    
    for table in expected_tables:
        if table not in tables and not any(table in t for t in tables):
            missing_tables.append(table)
    
    # Check for tables that should have been dropped
    dropped_tables = ['memory_blocks', 'embeddings_drift', 'co_access_patterns', 
                     'blocks', 'block_members', 'clusters', 'embeddings']
    
    for table in dropped_tables:
        if table in tables:
            unexpected_tables.append(table)
    
    if missing_tables:
        print(f"  [ERROR] Missing tables: {missing_tables}")
    else:
        print(f"  [OK] All expected tables present")
    
    if unexpected_tables:
        print(f"  [ERROR] Unexpected tables still exist: {unexpected_tables}")
    else:
        print(f"  [OK] All unnecessary tables removed")
    
    # Check columns in memories table
    cursor.execute("PRAGMA table_info(memories)")
    columns = {col[1] for col in cursor.fetchall()}
    
    expected_list_columns = ['who_list', 'when_list', 'where_list', 'what']
    missing_columns = [col for col in expected_list_columns if col not in columns]
    
    if missing_columns:
        print(f"  [ERROR] Missing list columns: {missing_columns}")
    else:
        print(f"  [OK] All list columns present")
    
    conn.close()
    return len(missing_tables) == 0 and len(unexpected_tables) == 0 and len(missing_columns) == 0

def test_data_migration():
    """Test that data was properly migrated to list columns"""
    print("\n=== Testing Data Migration ===")
    
    conn = sqlite3.connect('amemory.sqlite3')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Sample some memories to check migration
    cursor.execute("""
        SELECT memory_id, who_list, when_list, where_list, what
        FROM memories
        LIMIT 100
    """)
    
    rows = cursor.fetchall()
    
    valid_who = 0
    valid_when = 0
    valid_where = 0
    valid_what = 0
    
    for row in rows:
        # Check who_list
        try:
            who_list = json.loads(row['who_list']) if row['who_list'] else []
            if isinstance(who_list, list):
                valid_who += 1
        except:
            pass
        
        # Check when_list
        try:
            when_list = json.loads(row['when_list']) if row['when_list'] else []
            if isinstance(when_list, list):
                valid_when += 1
        except:
            pass
        
        # Check where_list
        try:
            where_list = json.loads(row['where_list']) if row['where_list'] else []
            if isinstance(where_list, list):
                valid_where += 1
        except:
            pass
        
        # Check what
        try:
            what_list = json.loads(row['what']) if row['what'] else []
            if isinstance(what_list, list):
                valid_what += 1
        except:
            pass
    
    total = len(rows)
    print(f"  WHO lists valid: {valid_who}/{total} ({valid_who*100//total}%)")
    print(f"  WHEN lists valid: {valid_when}/{total} ({valid_when*100//total}%)")
    print(f"  WHERE lists valid: {valid_where}/{total} ({valid_where*100//total}%)")
    print(f"  WHAT lists valid: {valid_what}/{total} ({valid_what*100//total}%)")
    
    conn.close()
    return valid_who == total and valid_when == total and valid_where == total

def test_retrieval_functionality():
    """Test that retrieval still works with new schema"""
    print("\n=== Testing Retrieval Functionality ===")
    
    try:
        from agentic_memory.retrieval import HybridRetriever
        from agentic_memory.storage.sql_store import MemoryStore
        from agentic_memory.storage.faiss_index import FaissIndex
        from agentic_memory.types import RetrievalQuery
        
        # Initialize components
        from agentic_memory.config import cfg
        store = MemoryStore(cfg.db_path)
        index = FaissIndex(cfg.index_path)
        retriever = HybridRetriever(store, index)
        
        # Test lexical search
        query = RetrievalQuery(
            session_id="test",
            text="test query"
        )
        
        # This should not crash
        import numpy as np
        dummy_vec = np.random.randn(2048).astype('float32')
        dummy_vec = dummy_vec / np.linalg.norm(dummy_vec)
        
        results = retriever.search(query, dummy_vec, topk_sem=5, topk_lex=5)
        
        print(f"  [OK] Retrieval returned {len(results)} results")
        
        # Check that results have expected format
        if results:
            sample = results[0]
            if hasattr(sample, 'memory_id') and hasattr(sample, 'score'):
                print(f"  [OK] Results have correct format")
            else:
                print(f"  [ERROR] Results missing expected fields")
                return False
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Retrieval test failed: {e}")
        return False

def test_web_interface():
    """Test that web interface endpoints still work"""
    print("\n=== Testing Web Interface ===")
    
    try:
        import requests
        
        # Check if web server is running
        response = requests.get('http://localhost:5001/health', timeout=2)
        if response.status_code == 200:
            print("  [OK] Web server is responsive")
            
            # Test analytics endpoint
            response = requests.get('http://localhost:5001/api/analytics/entities', timeout=2)
            if response.status_code == 200:
                print("  [OK] Analytics endpoint works")
            else:
                print(f"  [WARNING] Analytics endpoint returned {response.status_code}")
        else:
            print(f"  [WARNING] Web server returned {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("  [INFO] Web server not running (start it to test)")
    except Exception as e:
        print(f"  [ERROR] Web interface test failed: {e}")
    
    return True

def main():
    print("=" * 60)
    print("Migration Verification Tests")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_database_structure()
    all_passed &= test_data_migration()
    all_passed &= test_retrieval_functionality()
    test_web_interface()  # Non-critical
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] All critical tests passed!")
        print("[OK] Migration completed successfully!")
    else:
        print("[ERROR] Some tests failed - review output above")
    print("=" * 60)

if __name__ == "__main__":
    main()