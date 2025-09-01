"""Test script to verify temporal dimension prioritization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.dimension_attention_retriever import DimensionAttentionRetriever
from memory.dual_space_encoder import DualSpaceEncoder
from memory.chromadb_store import ChromaDBStore
from memory.temporal_query import TemporalQueryHandler
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_temporal_queries():
    """Test various temporal queries to ensure proper prioritization."""
    
    # Initialize components
    encoder = DualSpaceEncoder()
    chromadb = ChromaDBStore()
    retriever = DimensionAttentionRetriever(encoder, chromadb)
    temporal_handler = TemporalQueryHandler(encoder=encoder)
    
    # Test queries with explicit temporal intent
    test_queries = [
        {"what": "What is the last thing we discussed?"},
        {"what": "What did we just talk about?"},
        {"what": "Show me the most recent conversation"},
        {"what": "What was our latest discussion about?"},
        {"what": "What did we discuss earlier today?"},
        {"what": "What happened a moment ago?"},
        {"what": "Show me yesterdays conversations"},
        {"what": "What did Alice say about testing?"},  # Non-temporal for comparison
        {"what": "Explain the architecture"},  # Non-temporal for comparison
    ]
    
    print("\n" + "="*80)
    print("TEMPORAL QUERY PRIORITIZATION TEST")
    print("="*80 + "\n")
    
    for query in test_queries:
        query_text_display = query["what"]
        print(f"\nQuery: {query_text_display}")
        print("-" * 60)
        
        # Get query text for analysis
        query_text = " ".join([str(v) for v in query.values() if v])
        
        # Test dimension attention computation
        euclidean_emb, hyperbolic_emb = encoder.encode_dual(query_text)
        query_embedding = (euclidean_emb, hyperbolic_emb)
        
        dim_weights = retriever.compute_dimension_attention(query, query_embedding)
        
        # Sort dimensions by weight for display
        sorted_dims = sorted(dim_weights.items(), key=lambda x: x[1], reverse=True)
        
        print("Dimension Weights:")
        for dim, weight in sorted_dims:
            bar = "█" * int(weight * 50)
            print(f"  {dim.value:12s}: {weight:.3f} {bar}")
        
        # Test temporal intent detection
        lambda_e, lambda_h = encoder.compute_query_weights(query)
        temporal_type, similarity, strength = temporal_handler.detect_temporal_intent(
            query, lambda_e, lambda_h
        )
        
        if temporal_type:
            print(f"\nTemporal Detection:")
            print(f"  Type: {temporal_type}")
            print(f"  Similarity: {similarity:.3f}")
            print(f"  Strength: {strength:.3f}")
        
        # Check if temporal is ranked appropriately
        temporal_weight = dim_weights.get(retriever.MergeType.TEMPORAL, 0)
        temporal_rank = sorted_dims.index((retriever.MergeType.TEMPORAL, temporal_weight)) + 1
        
        is_temporal_query = any(keyword in query_text.lower() for keyword in [
            "last", "just", "recent", "latest", "earlier", "ago", "yesterday", "today"
        ])
        
        if is_temporal_query:
            if temporal_rank == 1:
                print(f"\n✓ PASS: Temporal dimension correctly ranked #1")
            else:
                print(f"\n✗ ISSUE: Temporal dimension ranked #{temporal_rank} (should be #1)")
        else:
            print(f"\nℹ Non-temporal query - Temporal ranked #{temporal_rank}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_temporal_queries()
