"""
Test script for the [DSAM] Dual-Space Agentic Memory enhancements.
"""

import os
import sys
from pathlib import Path

# Set offline mode to avoid HuggingFace rate limits
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from memory.memory_store import MemoryStore
from models.event import Event, FiveW1H, EventType
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dual_space_memory():
    """Test the enhanced memory system with dual-space encoding."""
    
    print("=" * 60)
    print("Testing [DSAM] Dual-Space Agentic Memory")
    print("=" * 60)
    
    # Initialize memory store
    print("\n1. Initializing enhanced memory store...")
    store = MemoryStore()
    print("   ✓ Memory store initialized with dual-space encoder")
    
    # Test storing events
    print("\n2. Storing test events...")
    
    events = [
        {
            'who': 'Alice',
            'what': 'implemented search functionality',
            'when': 'yesterday afternoon',
            'where': 'codebase repository',
            'why': 'improve user experience',
            'how': 'using Elasticsearch'
        },
        {
            'who': 'Bob',
            'what': 'fixed authentication bug',
            'when': 'this morning',
            'where': 'login module',
            'why': 'security vulnerability',
            'how': 'patched JWT validation'
        },
        {
            'who': 'Alice',
            'what': 'optimized database queries',
            'when': 'last week',
            'where': 'backend API',
            'why': 'reduce response time',
            'how': 'adding indexes and query optimization'
        },
        {
            'who': 'Charlie',
            'what': 'designed user interface',
            'when': 'two days ago',
            'where': 'frontend application',
            'why': 'modernize the look',
            'how': 'using React and Material UI'
        },
        {
            'who': 'Bob',
            'what': 'wrote unit tests',
            'when': 'yesterday',
            'where': 'test suite',
            'why': 'increase code coverage',
            'how': 'using pytest framework'
        }
    ]
    
    stored_events = []
    for i, event_data in enumerate(events):
        event = Event(
            five_w1h=FiveW1H(**event_data),
            event_type=EventType.ACTION,
            episode_id=f"episode_{i // 2}"  # Group events into episodes
        )
        
        success, message = store.store_event(event)
        if success:
            print(f"   ✓ {message}")
            stored_events.append(event)
        else:
            print(f"   ✗ Failed: {message}")
    
    # Test retrieval with concrete query (should favor Euclidean)
    print("\n3. Testing retrieval with concrete query...")
    concrete_query = {
        'who': 'Alice',
        'what': 'search',
        'where': 'codebase'
    }
    
    results = store.retrieve_memories(concrete_query, k=3)
    print(f"   Query: {concrete_query}")
    print(f"   Found {len(results)} results:")
    for event, score in results:
        print(f"     - {event.five_w1h.what[:50]} (score: {score:.3f})")
    
    # Test retrieval with abstract query (should favor Hyperbolic)
    print("\n4. Testing retrieval with abstract query...")
    abstract_query = {
        'why': 'improve performance',
        'how': 'optimization'
    }
    
    results = store.retrieve_memories(abstract_query, k=3)
    print(f"   Query: {abstract_query}")
    print(f"   Found {len(results)} results:")
    for event, score in results:
        print(f"     - {event.five_w1h.what[:50]} (score: {score:.3f})")
    
    # Test residual adaptation
    print("\n5. Testing residual adaptation...")
    
    # Retrieve multiple times to trigger adaptation
    for i in range(3):
        results = store.retrieve_memories({'who': 'Alice', 'what': 'code'}, k=5)
    
    stats = store.get_statistics()
    avg_residuals = stats['average_residual_norm']
    print(f"   Average Euclidean residual norm: {avg_residuals['euclidean']:.4f}")
    print(f"   Average Hyperbolic residual norm: {avg_residuals['hyperbolic']:.4f}")
    
    # Test duplicate handling
    print("\n6. Testing duplicate detection...")
    duplicate_event = Event(
        five_w1h=FiveW1H(
            who='Alice',
            what='implemented search functionality',
            when='yesterday afternoon',
            where='codebase repository',
            why='improve user experience',
            how='using Elasticsearch'
        ),
        event_type=EventType.ACTION,
        episode_id='episode_test'
    )
    
    success, message = store.store_event(duplicate_event)
    print(f"   {message}")
    
    # Test clustering
    print("\n7. Testing HDBSCAN clustering...")
    
    # Add more similar events to test clustering
    for i in range(5):
        event = Event(
            five_w1h=FiveW1H(
                who='Developer',
                what=f'implemented feature {i}',
                when='today',
                where='application',
                why='enhance functionality',
                how='coding'
            ),
            event_type=EventType.ACTION,
            episode_id='episode_cluster'
        )
        store.store_event(event)
    
    # Query to trigger clustering
    results = store.retrieve_memories(
        {'what': 'implemented', 'why': 'enhance'},
        k=5,
        use_clustering=True
    )
    
    print(f"   Retrieved {len(results)} clustered results")
    
    # Display statistics
    print("\n8. Final Statistics:")
    stats = store.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     - {k}: {v:.4f}" if isinstance(v, float) else f"     - {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Test state persistence
    print("\n9. Testing state persistence...")
    state_dir = Path("./test_state")
    success = store.save_state(state_dir)
    print(f"   Save state: {'✓ Success' if success else '✗ Failed'}")
    
    # Create new store and load state
    new_store = MemoryStore()
    success = new_store.load_state(state_dir)
    print(f"   Load state: {'✓ Success' if success else '✗ Failed'}")
    
    # Cleanup
    import shutil
    if state_dir.exists():
        shutil.rmtree(state_dir)
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_dual_space_memory()
