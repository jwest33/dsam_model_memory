from memory.chromadb_store import ChromaDBStore
import logging
logging.basicConfig(level=logging.INFO)

store = ChromaDBStore()
print('\nEuclidean Collections:')
print('  temporal_merges:', store.temporal_merges_collection.count())
print('  conceptual_merges:', store.conceptual_merges_collection.count())
print('  actor_merges:', store.actor_merges_collection.count())
print('  spatial_merges:', store.spatial_merges_collection.count())

print('\nHyperbolic Collections:')
if hasattr(store, 'temporal_merges_hyperbolic'):
    print('  temporal_merges_hyperbolic:', store.temporal_merges_hyperbolic.count())
else:
    print('  temporal_merges_hyperbolic: NOT FOUND')
    
if hasattr(store, 'conceptual_merges_hyperbolic'):
    print('  conceptual_merges_hyperbolic:', store.conceptual_merges_hyperbolic.count())
else:
    print('  conceptual_merges_hyperbolic: NOT FOUND')

if hasattr(store, 'actor_merges_hyperbolic'):
    print('  actor_merges_hyperbolic:', store.actor_merges_hyperbolic.count())
else:
    print('  actor_merges_hyperbolic: NOT FOUND')

if hasattr(store, 'spatial_merges_hyperbolic'):
    print('  spatial_merges_hyperbolic:', store.spatial_merges_hyperbolic.count())
else:
    print('  spatial_merges_hyperbolic: NOT FOUND')

print('\nCollections successfully created!' if hasattr(store, 'conceptual_merges_hyperbolic') else '\nHyperbolic collections not created!')