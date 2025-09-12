#!/usr/bin/env python3
"""
Database Migration Script for JAM Memory System
Consolidates WHO, WHEN, WHERE columns to JSON arrays
Drops unnecessary tables
"""

import sqlite3
import json
import shutil
from datetime import datetime
from pathlib import Path

def backup_database(db_path):
    """Create a timestamped backup of the database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"[OK] Backup created successfully")
    return backup_path

def check_columns_exist(cursor):
    """Check which columns and tables exist"""
    cursor.execute("PRAGMA table_info(memories)")
    columns = {col[1] for col in cursor.fetchall()}
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    return columns, tables

def migrate_to_list_columns(conn):
    """Phase 1: Migrate data to list columns"""
    cursor = conn.cursor()
    columns, tables = check_columns_exist(cursor)
    
    print("\n=== Phase 1: Data Migration ===")
    
    # Check if list columns exist, add them if not
    if 'who_list' not in columns:
        print("Adding who_list column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN who_list TEXT")
    
    if 'when_list' not in columns:
        print("Adding when_list column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN when_list TEXT")
    
    if 'where_list' not in columns:
        print("Adding where_list column...")
        cursor.execute("ALTER TABLE memories ADD COLUMN where_list TEXT")
    
    # Migrate WHO data
    print("\nMigrating WHO data...")
    cursor.execute("""
        UPDATE memories 
        SET who_list = CASE 
            WHEN who_list IS NOT NULL AND who_list != '' THEN who_list
            WHEN who_id IS NOT NULL AND who_id != '' THEN json_array(who_id)
            ELSE '[]'
        END
        WHERE who_list IS NULL OR who_list = ''
    """)
    who_updated = cursor.rowcount
    print(f"[OK] Updated {who_updated} WHO records")
    
    # Migrate WHEN data
    print("\nMigrating WHEN data...")
    cursor.execute("""
        UPDATE memories
        SET when_list = CASE
            WHEN when_list IS NOT NULL AND when_list != '' THEN when_list
            WHEN when_ts IS NOT NULL AND when_ts != '' THEN json_array(when_ts)
            ELSE '[]'
        END
        WHERE when_list IS NULL OR when_list = ''
    """)
    when_updated = cursor.rowcount
    print(f"[OK] Updated {when_updated} WHEN records")
    
    # Migrate WHERE data
    print("\nMigrating WHERE data...")
    cursor.execute("""
        UPDATE memories
        SET where_list = CASE
            WHEN where_list IS NOT NULL AND where_list != '' THEN where_list
            WHEN where_value IS NOT NULL AND where_value != '' THEN json_array(where_value)
            ELSE '[]'
        END
        WHERE where_list IS NULL OR where_list = ''
    """)
    where_updated = cursor.rowcount
    print(f"[OK] Updated {where_updated} WHERE records")
    
    conn.commit()
    return who_updated, when_updated, where_updated

def drop_unnecessary_tables(conn):
    """Phase 2: Drop tables that are no longer needed"""
    cursor = conn.cursor()
    columns, tables = check_columns_exist(cursor)
    
    print("\n=== Phase 2: Table Cleanup ===")
    
    tables_to_drop = [
        ('memory_blocks', 'Block building is now dynamic'),
        ('embeddings_drift', 'Drift tracking is now in-memory'),
        ('co_access_patterns', 'Patterns computed dynamically')
    ]
    
    for table_name, reason in tables_to_drop:
        if table_name in tables:
            # Get row count before dropping
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                print(f"\nDropping {table_name} ({row_count} rows)")
                print(f"  Reason: {reason}")
                
                # Drop the table
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                print(f"  [OK] Table dropped successfully")
            except Exception as e:
                print(f"  [ERROR] Error dropping {table_name}: {e}")
        else:
            print(f"\n{table_name} doesn't exist (already dropped)")
    
    conn.commit()

def verify_migration(conn):
    """Verify the migration was successful"""
    cursor = conn.cursor()
    
    print("\n=== Verification ===")
    
    # Check that list columns have data
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN who_list IS NOT NULL AND who_list != '[]' THEN 1 ELSE 0 END) as who_populated,
            SUM(CASE WHEN when_list IS NOT NULL AND when_list != '[]' THEN 1 ELSE 0 END) as when_populated,
            SUM(CASE WHEN where_list IS NOT NULL AND where_list != '[]' THEN 1 ELSE 0 END) as where_populated
        FROM memories
    """)
    
    stats = cursor.fetchone()
    print(f"Total memories: {stats[0]}")
    print(f"WHO lists populated: {stats[1]} ({stats[1]*100//stats[0] if stats[0] else 0}%)")
    print(f"WHEN lists populated: {stats[2]} ({stats[2]*100//stats[0] if stats[0] else 0}%)")
    print(f"WHERE lists populated: {stats[3]} ({stats[3]*100//stats[0] if stats[0] else 0}%)")
    
    # Sample some migrated data
    print("\n=== Sample Migrated Data ===")
    cursor.execute("""
        SELECT memory_id, who_list, when_list, where_list 
        FROM memories 
        WHERE (who_list != '[]' OR when_list != '[]' OR where_list != '[]')
        LIMIT 3
    """)
    
    for row in cursor.fetchall():
        print(f"\nMemory ID: {row[0][:8]}...")
        print(f"  WHO: {row[1][:50]}..." if row[1] else "  WHO: []")
        print(f"  WHEN: {row[2][:50]}..." if row[2] else "  WHEN: []")
        print(f"  WHERE: {row[3][:50]}..." if row[3] else "  WHERE: []")
    
    # Check remaining tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    print("\n=== Remaining Tables ===")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cursor.fetchone()[0]
        print(f"  {table[0]}: {count} rows")

def optimize_database(conn):
    """Optimize the database after migration"""
    cursor = conn.cursor()
    
    print("\n=== Optimization ===")
    print("Running VACUUM...")
    cursor.execute("VACUUM")
    print("[OK] VACUUM complete")
    
    print("Running ANALYZE...")
    cursor.execute("ANALYZE")
    print("[OK] ANALYZE complete")
    
    conn.commit()

def main():
    """Main migration function"""
    db_path = "amemory.sqlite3"
    
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found!")
        return
    
    print("=" * 60)
    print("JAM Memory Database Migration")
    print("=" * 60)
    
    # Create backup
    backup_path = backup_database(db_path)
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Show current state
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]
        print(f"\nDatabase contains {total_memories} memories")
        
        # Perform migration
        migrate_to_list_columns(conn)
        drop_unnecessary_tables(conn)
        verify_migration(conn)
        optimize_database(conn)
        
        print("\n" + "=" * 60)
        print("[OK] Migration completed successfully!")
        print(f"[OK] Backup saved at: {backup_path}")
        print("=" * 60)
        
        conn.close()
        
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        print(f"[ERROR] Database backup available at: {backup_path}")
        print("[ERROR] Restore with: shutil.copy2('{backup_path}', '{db_path}')")
        raise

if __name__ == "__main__":
    main()
