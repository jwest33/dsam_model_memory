#!/usr/bin/env python3
"""
Update code to use new schema with list columns
"""

import os
import re
from pathlib import Path

def update_retrieval_py():
    """Update retrieval.py to use list columns"""
    file_path = "agentic_memory/retrieval.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update references to old columns
    replacements = [
        # Use first element of when_list for recency calculations
        (r"row\['when_ts'\]", "json.loads(row.get('when_list', '[]'))[0] if json.loads(row.get('when_list', '[]')) else row.get('when_ts', '')"),
        (r"memory\['when_ts'\]", "json.loads(memory.get('when_list', '[]'))[0] if json.loads(memory.get('when_list', '[]')) else memory.get('when_ts', '')"),
        (r"m\['when_ts'\]", "json.loads(m.get('when_list', '[]'))[0] if json.loads(m.get('when_list', '[]')) else m.get('when_ts', '')"),
        
        # Use who_list instead of who_id
        (r"memory\['who_id'\]", "json.loads(memory.get('who_list', '[]'))[0] if json.loads(memory.get('who_list', '[]')) else ''"),
        (r"m\['who_id'\]", "json.loads(m.get('who_list', '[]'))[0] if json.loads(m.get('who_list', '[]')) else ''"),
        (r"other_memory\['who_id'\]", "json.loads(other_memory.get('who_list', '[]'))[0] if json.loads(other_memory.get('who_list', '[]')) else ''"),
        
        # Update get references
        (r"\.get\('who_id', ''\)", ".get('who_list', '[]')"),
        (r"\.get\('when_ts', ''\)", ".get('when_list', '[]')"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Add json import if not present
    if 'import json' not in content:
        content = 'import json\n' + content
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Updated {file_path}")

def update_sql_store_py():
    """Update sql_store.py to use list columns"""
    file_path = "agentic_memory/storage/sql_store.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update SQL queries
    replacements = [
        # Update check_actor_exists to use who_list
        (r"SELECT COUNT\(\*\) FROM memories WHERE who_id = \?", 
         "SELECT COUNT(*) FROM memories WHERE who_list LIKE '%\"' || ? || '\"%'"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Updated {file_path}")

def update_flask_app():
    """Update Flask app to use list columns"""
    file_path = "agentic_memory/server/flask_app.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update queries that reference old columns
    replacements = [
        # Update temporal analytics query
        (r"COUNT\(DISTINCT who_id\) as actors", 
         "COUNT(DISTINCT who_list) as actors"),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Updated {file_path}")

def verify_changes():
    """Verify no old column references remain in key files"""
    print("\n=== Verifying Changes ===")
    
    key_files = [
        "agentic_memory/retrieval.py",
        "agentic_memory/storage/sql_store.py",
        "agentic_memory/server/flask_app.py",
    ]
    
    old_columns = ['who_id', 'when_ts', 'when_expr', 'where_loc']
    
    issues_found = False
    for file_path in key_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for old_col in old_columns:
            # Skip if it's in a comment or part of who_list/when_list
            pattern = rf'\b{old_col}\b(?!_list)'
            matches = re.findall(pattern, content)
            if matches and old_col not in ['who_id']:  # who_id might still appear in comments
                print(f"  WARNING: Found {len(matches)} references to '{old_col}' in {file_path}")
                issues_found = True
    
    if not issues_found:
        print("  [OK] No old column references found in key files")

def main():
    print("=" * 60)
    print("Updating Code for New Schema")
    print("=" * 60)
    
    try:
        # Update key files
        update_retrieval_py()
        update_sql_store_py()
        update_flask_app()
        
        # Verify changes
        verify_changes()
        
        print("\n" + "=" * 60)
        print("[OK] Code update completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Update failed: {e}")
        raise

if __name__ == "__main__":
    main()