import sqlite3
import json

con = sqlite3.connect('amemory.sqlite3')
con.row_factory = sqlite3.Row
cursor = con.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('All tables in database:')
print('-' * 50)
for table in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table['name']}")
        count = cursor.fetchone()[0]
        print(f"{table['name']:25} ({count} records)")
    except:
        print(f"{table['name']:25} (virtual table)")

print('\nSample memory record:')
print('-' * 50)

# Get a more recent memory
cursor.execute("""
    SELECT * FROM memories 
    ORDER BY created_at DESC 
    LIMIT 1
""")
row = cursor.fetchone()
if row:
    for key in row.keys():
        value = row[key]
        if key == 'extra_json' and value:
            try:
                value = json.loads(value)
                value = json.dumps(value, indent=2)[:200] + '...' if len(json.dumps(value)) > 200 else json.dumps(value, indent=2)
            except:
                pass
        elif isinstance(value, str) and len(str(value)) > 100:
            value = str(value)[:100] + '...'
        print(f'{key:20}: {value}')

# Check for related data
print('\n\nRelated data for this memory:')
print('-' * 50)
if row:
    memory_id = row['memory_id']
    
    # Check embeddings
    cursor.execute("SELECT dim FROM embeddings WHERE memory_id = ?", (memory_id,))
    embedding = cursor.fetchone()
    if embedding:
        print(f"Has embedding: Yes (dimension: {embedding['dim']})")
    else:
        print("Has embedding: No")
    
    # Check usage stats
    cursor.execute("SELECT accesses, last_access FROM usage_stats WHERE memory_id = ?", (memory_id,))
    usage = cursor.fetchone()
    if usage:
        print(f"Usage stats: {usage['accesses']} accesses, last: {usage['last_access']}")
    else:
        print("Usage stats: None")
    
    # Check cluster membership
    cursor.execute("""
        SELECT c.cluster_id, c.label, cm.weight 
        FROM cluster_membership cm
        JOIN clusters c ON cm.cluster_id = c.cluster_id
        WHERE cm.memory_id = ?
    """, (memory_id,))
    clusters = cursor.fetchall()
    if clusters:
        print(f"Cluster memberships: {len(clusters)}")
        for cluster in clusters[:3]:  # Show first 3
            print(f"  - Cluster {cluster['cluster_id']}: {cluster['label']} (weight: {cluster['weight']:.3f})")
    else:
        print("Cluster memberships: None")

con.close()