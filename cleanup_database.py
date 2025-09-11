#!/usr/bin/env python3
"""
Database Cleanup Script - Drop unnecessary tables
"""

import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

def backup_database(db_path):
    """Create a timestamped backup of the database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.cleanup_backup_{timestamp}"
    
    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"[OK] Backup created successfully")
    return backup_path

def cleanup_tables(conn):
    """Drop unnecessary tables"""
    cursor = conn.cursor()
    
    # Get current tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    
    # Tables to drop (based on our analysis)
    tables_to_drop = [
        ('block_members', 'Dynamic block building replaces this'),
        ('blocks', 'Dynamic block building replaces this'),
        ('cluster_membership', 'Clustering done dynamically'),
        ('clusters', 'Clustering done dynamically'),
        ('embedding_drift', 'Drift tracking now in-memory'),
        ('embeddings', 'Embeddings stored in memories table'),
        ('memory_importance', 'Importance in memories table'),
        ('memory_synapses', 'Co-access computed dynamically'),
        ('usage_stats', 'Usage tracked in memories table')
    ]
    
    print("\n=== Dropping Unnecessary Tables ===")
    
    for table_name, reason in tables_to_drop:
        if table_name in existing_tables:
            try:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                print(f"\n{table_name} ({row_count} rows)")
                print(f"  Reason: {reason}")
                
                # Drop the table
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"  [OK] Dropped successfully")
            except Exception as e:
                print(f"  [ERROR] Failed to drop: {e}")
        else:
            print(f"\n{table_name}: Already gone")
    
    conn.commit()

def verify_cleanup(conn):
    """Verify remaining tables"""
    cursor = conn.cursor()
    
    print("\n=== Remaining Tables ===")
    cursor.execute("""
        SELECT name, type 
        FROM sqlite_master 
        WHERE type IN ('table', 'index')
        ORDER BY type, name
    """)
    
    current_type = None
    for name, obj_type in cursor.fetchall():
        if obj_type != current_type:
            current_type = obj_type
            print(f"\n{obj_type.upper()}S:")
        
        if obj_type == 'table':
            cursor.execute(f"SELECT COUNT(*) FROM {name}")
            count = cursor.fetchone()[0]
            print(f"  {name}: {count} rows")
        else:
            print(f"  {name}")

def main():
    db_path = "amemory.sqlite3"
    
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found!")
        return
    
    print("=" * 60)
    print("Database Cleanup - Drop Unnecessary Tables")
    print("=" * 60)
    
    backup_path = backup_database(db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Perform cleanup
        cleanup_tables(conn)
        verify_cleanup(conn)
        
        # Optimize
        print("\n=== Optimizing Database ===")
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        print("[OK] VACUUM complete")
        cursor.execute("ANALYZE")
        print("[OK] ANALYZE complete")
        
        conn.commit()
        conn.close()
        
        print("\n" + "=" * 60)
        print("[OK] Cleanup completed successfully!")
        print(f"[OK] Backup saved at: {backup_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Cleanup failed: {e}")
        print(f"[ERROR] Restore from: {backup_path}")
        raise

if __name__ == "__main__":
    main()