#!/usr/bin/env python3
"""Debug why benchmark shows 0% for semantic despite retrieval working."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.recall_benchmark import RecallBenchmark

# Load and run benchmark on just semantic cases
benchmark = RecallBenchmark()
benchmark.load_testset("benchmarks/test_data/benchmark_testset_20250909_132419.json")

# Filter to just semantic cases
semantic_cases = [tc for tc in benchmark.test_cases if tc.test_type == 'semantic']
print(f"Found {len(semantic_cases)} semantic test cases")

# Run evaluation on first semantic case
if semantic_cases:
    test_case = semantic_cases[0]
    print(f"\nEvaluating: {test_case.test_id}")
    print(f"Query: {test_case.query_text[:100]}...")
    print(f"Expected relevant: {test_case.expected_relevant[:5]}")
    
    # Run the exact same evaluation as benchmark
    print(f"\nRunning benchmark.evaluate_retrieval()...")
    result = benchmark.evaluate_retrieval(test_case)
    
    print(f"\nBenchmark result:")
    print(f"  Total retrieved: {result['total_retrieved']}")
    print(f"  Total expected: {result['total_expected']}")
    print(f"  Metrics at k=5: {result['metrics'].get(5, {})}")
    
    # Check what the benchmark actually retrieved
    if hasattr(benchmark, '_last_retrieved_ids'):
        print(f"  Benchmark retrieved (top 5): {benchmark._last_retrieved_ids[:5]}")
    
    # Get the actual retrieved IDs
    from agentic_memory.types import RetrievalQuery
    retrieval_query = RetrievalQuery(
        session_id=test_case.query_metadata.get('session_id', 'benchmark_session'),
        text=test_case.query_text,
        actor_hint=test_case.query_metadata.get('actor_hint'),
        spatial_hint=test_case.query_metadata.get('spatial_hint'),
        temporal_hint=test_case.query_metadata.get('temporal_hint')
    )
    
    query_embedding = benchmark.embedder.encode([retrieval_query.text], normalize_embeddings=True)[0]
    candidates = benchmark.retriever.search(
        rq=retrieval_query,
        qvec=query_embedding,
        topk_sem=50,
        topk_lex=50
    )
    
    retrieved_ids = [c.memory_id for c in candidates]
    print(f"\nActual retrieved (top 5): {retrieved_ids[:5]}")
    
    # Manual calculation
    expected_set = set(test_case.expected_relevant)
    retrieved_set = set(retrieved_ids[:5])
    overlap = expected_set & retrieved_set
    
    print(f"\nManual calculation:")
    print(f"  Overlap: {overlap}")
    print(f"  Precision@5: {len(overlap)/5:.3f}")
    print(f"  Recall@5: {len(overlap)/len(expected_set):.3f}")