"""
Test script for similarity cache functionality
"""

import numpy as np
from memory.chromadb_store import ChromaDBStore
from memory.memory_store import MemoryStore
from models.event import Event, FiveW1H, EventType
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_similarity_cache():
    """Test the similarity cache system"""
    
    print("Initializing memory store...")
    memory_store = MemoryStore()
    
    # Create some test events
    events = [
        Event(
            five_w1h=FiveW1H(
                who="User",
                what="Asking about performance optimization",
                when=datetime.now().isoformat(),
                where="chat",
                why="improve speed",
                how="questioning"
            ),
            event_type=EventType.OBSERVATION
        ),
        Event(
            five_w1h=FiveW1H(
                who="Assistant",
                what="Explaining caching strategies",
                when=datetime.now().isoformat(),
                where="chat",
                why="answer question",
                how="explaining"
            ),
            event_type=EventType.OBSERVATION
        ),
        Event(
            five_w1h=FiveW1H(
                who="User",
                what="Implementing similarity cache",
                when=datetime.now().isoformat(),
                where="codebase",
                why="optimize performance",
                how="coding"
            ),
            event_type=EventType.ACTION
        )
    ]
    
    print(f"\nStoring {len(events)} test events...")
    for event in events:
        success, msg = memory_store.store_event(event)
        print(f"  - Stored: {event.five_w1h.what[:30]}... - {msg}")
    
    # Check similarity cache stats
    print("\nSimilarity Cache Statistics:")
    stats = memory_store.chromadb.get_similarity_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test cached similarity retrieval
    if len(events) >= 2:
        print("\nTesting cached similarity between first two events:")
        sim = memory_store.chromadb.get_cached_similarity(
            events[0].id, events[1].id, 
            lambda_e=0.5, lambda_h=0.5
        )
        if sim is not None:
            print(f"  Similarity: {sim:.4f}")
        else:
            print("  No cached similarity found (computing now...)")
            # Force computation by updating cache
            if events[0].id in memory_store.embedding_cache:
                memory_store.chromadb.update_similarity_cache(
                    events[0].id, 
                    memory_store.embedding_cache[events[0].id]
                )
            sim = memory_store.chromadb.get_cached_similarity(
                events[0].id, events[1].id,
                lambda_e=0.5, lambda_h=0.5
            )
            if sim is not None:
                print(f"  Computed similarity: {sim:.4f}")
    
    # Test retrieval with cached similarities
    print("\nTesting retrieval (should use cached similarities):")
    query = {"what": "performance optimization"}
    results = memory_store.retrieve_memories(query, k=3)
    
    print(f"  Found {len(results)} results")
    for event, score in results[:3]:
        print(f"    - {event.five_w1h.what[:40]}... (score: {score:.4f})")
    
    # Final cache stats
    print("\nFinal Cache Statistics:")
    stats = memory_store.chromadb.get_similarity_stats()
    print(f"  Cache hits: {stats.get('cache_hits', 0)}")
    print(f"  Cache misses: {stats.get('cache_misses', 0)}")
    print(f"  Hit rate: {stats.get('hit_rate', 0):.2%}")
    print(f"  Cached pairs: {stats.get('cached_pairs', 0)}")
    print(f"  Total possible pairs: {stats.get('total_pairs', 0)}")
    
    print("\nSimilarity cache test completed successfully!")

if __name__ == "__main__":
    test_similarity_cache()
