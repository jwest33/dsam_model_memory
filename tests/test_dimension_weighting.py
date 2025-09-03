#!/usr/bin/env python3
"""Test script for dimension attention weighting fixes."""

import sys
import logging
from models.merge_types import MergeType
from memory.dimension_attention_retriever import DimensionAttentionRetriever
from memory.dual_space_encoder import DualSpaceEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dimension_weights():
    """Test dimension attention weights for various query types."""
    
    # Initialize encoder (needed for the retriever)
    encoder = DualSpaceEncoder()
    
    # Initialize retriever (minimal setup for testing)
    retriever = DimensionAttentionRetriever(
        encoder=encoder,
        multi_merger=None,
        similarity_cache=None,
        storage_backend=None,
        temporal_manager=None  # Explicitly set to None to avoid attribute errors
    )
    
    # Test queries with expected dominant dimensions
    test_cases = [
        # Original test cases
        {
            'query': {'what': 'what have we discussed about diagnosing firewall packet drops'},
            'expected_dominant': 'conceptual',
            'description': 'Discussion about specific technical topic'
        },
        {
            'query': {'what': 'what have we discussed'},
            'expected_dominant': 'conceptual',
            'description': 'General discussion history (no specific topic)'
        },
        {
            'query': {'what': 'what is the last thing we talked about'},
            'expected_dominant': 'temporal',
            'description': 'Explicit temporal query'
        },
        {
            'query': {'who': 'assistant', 'what': 'explained the firewall configuration'},
            'expected_dominant': 'actor',
            'description': 'Actor-focused query with who field'
        },
        {
            'query': {'what': 'how do you configure packet filtering rules'},
            'expected_dominant': 'conceptual',
            'description': 'Conceptual how-to query'
        },
        {
            'query': {'what': 'what did I ask about network security'},
            'expected_dominant': 'actor',
            'description': 'Actor-specific query (I asked)'
        },
        {
            'query': {'what': 'where is the firewall configuration stored'},
            'expected_dominant': 'spatial',
            'description': 'Spatial/location query'
        },
        
        # Additional temporal test cases with "when" scenarios
        {
            'query': {'when': '2024-09-03', 'what': 'firewall configuration'},
            'expected_dominant': 'temporal',
            'description': 'Query with explicit when field (date)'
        },
        {
            'query': {'what': 'when did we discuss firewall rules'},
            'expected_dominant': 'temporal',
            'description': 'When did we discuss X'
        },
        {
            'query': {'what': 'what happened yesterday'},
            'expected_dominant': 'temporal',
            'description': 'Yesterday temporal reference'
        },
        {
            'query': {'what': 'events from 5 minutes ago'},
            'expected_dominant': 'temporal',
            'description': 'Specific time ago reference'
        },
        {
            'query': {'what': 'what did we talk about this morning'},
            'expected_dominant': 'temporal',
            'description': 'This morning temporal reference'
        },
        {
            'query': {'what': 'show me our conversation from last week'},
            'expected_dominant': 'temporal',
            'description': 'Last week temporal reference'
        },
        {
            'query': {'when': 'recently', 'what': 'network issues'},
            'expected_dominant': 'temporal',
            'description': 'When field with "recently"'
        },
        {
            'query': {'what': 'the first thing we discussed today'},
            'expected_dominant': 'temporal',
            'description': 'Ordered temporal - first thing today'
        },
        {
            'query': {'what': 'what came after the firewall discussion'},
            'expected_dominant': 'temporal',
            'description': 'Relative temporal position'
        },
        {
            'query': {'what': 'memories from before lunch'},
            'expected_dominant': 'temporal',
            'description': 'Before lunch temporal marker'
        },
        {
            'query': {'when': 'during the security audit', 'what': 'issues found'},
            'expected_dominant': 'temporal',
            'description': 'During a specific event'
        },
        
        # Edge cases mixing temporal with other dimensions
        {
            'query': {'who': 'John', 'when': 'yesterday', 'what': 'reported'},
            'expected_dominant': 'temporal',
            'description': 'Actor + explicit when (temporal should win)'
        },
        {
            'query': {'what': 'when was the last system update'},
            'expected_dominant': 'temporal',
            'description': 'When was X (temporal question)'
        },
        {
            'query': {'what': 'timeline of security incidents'},
            'expected_dominant': 'temporal',
            'description': 'Timeline keyword'
        },
        {
            'query': {'what': 'history of configuration changes'},
            'expected_dominant': 'temporal',
            'description': 'History keyword'
        },
        {
            'query': {'what': 'sequence of events leading to the crash'},
            'expected_dominant': 'temporal',
            'description': 'Sequence of events'
        },
        {
            'query': {'what': 'chronological order of deployments'},
            'expected_dominant': 'temporal',
            'description': 'Chronological ordering'
        },
        {
            'query': {'what': 'earlier today we mentioned something about ports'},
            'expected_dominant': 'temporal',
            'description': 'Earlier today reference'
        },
        {
            'query': {'what': 'just now you said something'},
            'expected_dominant': 'temporal',
            'description': 'Just now temporal marker'
        },
        {
            'query': {'what': 'recent memories about firewall'},
            'expected_dominant': 'temporal',
            'description': 'Recent memories'
        },
        
        # Non-temporal "when" usage (conceptual)
        {
            'query': {'what': 'explain when to use firewall rules'},
            'expected_dominant': 'conceptual',
            'description': 'When to use (conceptual, not temporal)'
        },
        {
            'query': {'what': 'when should I configure packet filtering'},
            'expected_dominant': 'conceptual',
            'description': 'When should (advice, not temporal)'
        }
    ]
    
    print("=" * 80)
    print("DIMENSION ATTENTION WEIGHT TESTING")
    print("=" * 80)
    
    passed = 0
    failed = 0
    failures = []
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        expected = test_case['expected_dominant']
        description = test_case['description']
        
        # Compute dimension attention weights
        query_text = ' '.join([str(v) for v in query.values() if v])
        euclidean_emb, hyperbolic_emb = encoder.encode_dual(query_text)
        query_embedding = (euclidean_emb, hyperbolic_emb)
        
        weights = retriever.compute_dimension_attention(query, query_embedding)
        
        # Find dominant dimension
        dominant_dim = max(weights, key=weights.get)
        dominant_name = dominant_dim.value
        
        # Check if it matches expected
        if dominant_name == expected:
            status = "✓"
            passed += 1
        else:
            status = "✗"
            failed += 1
            failures.append((i, description, expected, dominant_name))
        
        print(f"\nTest Case {i}: {description}")
        print(f"Query: {query_text}")
        print(f"Expected Dominant: {expected}")
        print(f"Actual Dominant: {dominant_name} {status}")
        print(f"Weights: ", end="")
        for dim_type, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"{dim_type.value}: {weight:.2f}  ", end="")
        print()
        
        if dominant_name != expected:
            print(f"  WARNING: Expected {expected} to dominate, but got {dominant_name}")
    
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")
    
    if failures:
        print("\nFailed Test Cases:")
        for test_num, desc, exp, actual in failures:
            print(f"  #{test_num}: {desc}")
            print(f"    Expected: {exp}, Got: {actual}")
    
    print("=" * 80)

if __name__ == "__main__":
    test_dimension_weights()
