import chromadb

client = chromadb.PersistentClient(path='./state/chromadb')
collections = client.list_collections()
print('Collections:', [c.name for c in collections])

for collection in collections:
    try:
        count = collection.count()
        print(f'{collection.name}: {count} items')
        
        # Get a sample of IDs
        results = collection.get(limit=5)
        if results['ids']:
            print(f'  Sample IDs: {results["ids"]}')
    except Exception as e:
        print(f'  Error accessing {collection.name}: {e}')