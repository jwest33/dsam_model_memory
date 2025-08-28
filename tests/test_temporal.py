"""
Test script for temporal query handling.

Demonstrates how the system handles queries like "what is the last thing we talked about"
using probabilistic temporal detection and smooth time-decay weighting.
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from datetime import datetime, timedelta
from memory.memory_store import MemoryStore
from memory.temporal_query import TemporalQueryHandler
from models.event import Event, FiveW1H, EventType
import time


def create_test_memories(store: MemoryStore):
    """Create a series of test memories with different timestamps."""
    
    base_time = datetime.utcnow()
    
    # Create memories at different time intervals
    test_events = [
        # Very recent (1 hour ago)
        {
            "who": "user",
            "what": "asked about implementing a caching system for the API",
            "when": (base_time - timedelta(hours=1)).isoformat() + "Z",
            "where": "development_discussion",
            "why": "improve performance",
            "how": "technical inquiry"
        },
        # Recent (4 hours ago)
        {
            "who": "assistant",
            "what": "explained Redis caching patterns and best practices",
            "when": (base_time - timedelta(hours=4)).isoformat() + "Z",
            "where": "development_discussion",
            "why": "answer technical question",
            "how": "detailed explanation"
        },
        # Today (8 hours ago)
        {
            "who": "user",
            "what": "discussed database schema design for user authentication",
            "when": (base_time - timedelta(hours=8)).isoformat() + "Z",
            "where": "architecture_planning",
            "why": "system design",
            "how": "collaborative discussion"
        },
        # Yesterday (30 hours ago)
        {
            "who": "assistant",
            "what": "helped debug a memory leak in the JavaScript application",
            "when": (base_time - timedelta(hours=30)).isoformat() + "Z",
            "where": "debugging_session",
            "why": "fix performance issue",
            "how": "step-by-step debugging"
        },
        # Few days ago (72 hours ago)
        {
            "who": "user",
            "what": "requested help with machine learning model deployment",
            "when": (base_time - timedelta(hours=72)).isoformat() + "Z",
            "where": "ml_discussion",
            "why": "production deployment",
            "how": "deployment planning"
        },
        # A week ago
        {
            "who": "assistant",
            "what": "provided initial project setup instructions and guidelines",
            "when": (base_time - timedelta(days=7)).isoformat() + "Z",
            "where": "project_start",
            "why": "begin new project",
            "how": "comprehensive guide"
        }
    ]
    
    # Store all events
    print("Storing test memories...")
    for event_data in test_events:
        event = Event(
            five_w1h=FiveW1H(**event_data),
            event_type=EventType.ACTION,
            episode_id="test_session"
        )
        success, msg = store.store_event(event)
        if success:
            print(f"  ✓ Stored: {event_data['what'][:50]}...")
        else:
            print(f"  ✗ Failed: {msg}")
    
    print()


def test_temporal_queries(store: MemoryStore):
    """Test various temporal queries."""
    
    # Test queries with expected temporal behavior
    test_queries = [
        {
            "query": {"what": "what is the last thing we talked about"},
            "description": "Strong recency query - should heavily favor recent memories"
        },
        {
            "query": {"what": "recent discussions about performance"},
            "description": "Moderate recency with topic - balances recency and relevance"
        },
        {
            "query": {"what": "today's conversation topics"},
            "description": "Session-based query - should focus on last 8-12 hours"
        },
        {
            "query": {"what": "how did our project discussion begin"},
            "description": "Ordered query - should favor older memories"
        },
        {
            "query": {"what": "caching system implementation details"},
            "description": "Non-temporal query - pure semantic similarity"
        }
    ]
    
    print("=" * 80)
    print("TEMPORAL QUERY TESTS")
    print("=" * 80)
    
    for test in test_queries:
        print(f"\n📝 Query: {list(test['query'].values())[0]}")
        print(f"   Type: {test['description']}")
        print("-" * 60)
        
        # Test with temporal weighting enabled
        print("\n  With Temporal Weighting:")
        results_temporal = store.retrieve_memories(
            test['query'],
            k=3,
            use_clustering=False,
            use_temporal=True
        )
        
        for i, (event, score) in enumerate(results_temporal, 1):
            # Parse time for display
            event_time = datetime.fromisoformat(event.five_w1h.when.replace('Z', '+00:00'))
            time_ago = datetime.utcnow() - event_time.replace(tzinfo=None)
            hours_ago = time_ago.total_seconds() / 3600
            
            if hours_ago < 24:
                time_str = f"{hours_ago:.1f} hours ago"
            else:
                time_str = f"{hours_ago/24:.1f} days ago"
            
            print(f"    {i}. [{score:.3f}] {event.five_w1h.what[:60]}...")
            print(f"       Time: {time_str}")
        
        # Test without temporal weighting for comparison
        print("\n  Without Temporal Weighting:")
        results_baseline = store.retrieve_memories(
            test['query'],
            k=3,
            use_clustering=False,
            use_temporal=False
        )
        
        for i, (event, score) in enumerate(results_baseline, 1):
            event_time = datetime.fromisoformat(event.five_w1h.when.replace('Z', '+00:00'))
            time_ago = datetime.utcnow() - event_time.replace(tzinfo=None)
            hours_ago = time_ago.total_seconds() / 3600
            
            if hours_ago < 24:
                time_str = f"{hours_ago:.1f} hours ago"
            else:
                time_str = f"{hours_ago/24:.1f} days ago"
            
            print(f"    {i}. [{score:.3f}] {event.five_w1h.what[:60]}...")
            print(f"       Time: {time_str}")


def test_temporal_detection():
    """Test the temporal detection on various queries."""
    
    print("\n" + "=" * 80)
    print("TEMPORAL DETECTION TESTS")
    print("=" * 80)
    
    # Create a simple encoder for testing
    from memory.dual_space_encoder import DualSpaceEncoder
    encoder = DualSpaceEncoder()
    handler = TemporalQueryHandler(encoder=encoder)
    
    test_phrases = [
        "what is the last thing we discussed",
        "tell me about our recent conversation",
        "what did we talk about today",
        "how did this conversation start",
        "explain the caching system",  # Non-temporal
        "what were the main topics yesterday",
        "most recent error we debugged",
        "database schema design considerations"  # Non-temporal
    ]
    
    print("\nTemporal Intent Detection (Probabilistic):")
    print("-" * 60)
    
    for phrase in test_phrases:
        context = handler.compute_temporal_context({"what": phrase})
        
        if context['temporal_strength'] > 0.2:
            print(f"\n✓ '{phrase}'")
            print(f"  Type: {context['temporal_type']}")
            print(f"  Similarity: {context['similarity']:.3f}")
            print(f"  Strength: {context['temporal_strength']:.3f}")
            print(f"  Window: {context['suggested_window']:.1f} hours")
        else:
            print(f"\n✗ '{phrase}' - No significant temporal intent detected")
            print(f"  (Strength: {context['temporal_strength']:.3f})")


def main():
    """Run all temporal tests."""
    
    print("Initializing Memory Store with Temporal Support...")
    store = MemoryStore()
    
    # Clear any existing memories for clean test
    print("Clearing existing memories...")
    store.chromadb.clear_all()
    
    # Create test memories
    create_test_memories(store)
    
    # Test temporal detection
    test_temporal_detection()
    
    # Test temporal queries
    test_temporal_queries(store)
    
    print("\n" + "=" * 80)
    print("TEMPORAL QUERY TESTING COMPLETE")
    print("=" * 80)
    print("\nKey observations:")
    print("• Temporal queries are detected probabilistically using embeddings")
    print("• Time decay is applied smoothly based on temporal strength")
    print("• System balances semantic relevance with temporal recency")
    print("• No brittle regex patterns - fully probabilistic approach")


if __name__ == "__main__":
    main()
