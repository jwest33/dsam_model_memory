#!/usr/bin/env python3
"""Debug why benchmark shows 0% for semantic search despite retrieval working."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Load the latest test set
test_file = Path("benchmarks/test_data/benchmark_testset_20250909_125618.json")
with open(test_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find semantic test cases
semantic_cases = [tc for tc in data['test_cases'] if tc['test_type'] == 'semantic']
print(f"Found {len(semantic_cases)} semantic test cases")

# Check first semantic case
if semantic_cases:
    tc = semantic_cases[0]
    print(f"\nFirst semantic test case:")
    print(f"  ID: {tc['test_id']}")
    print(f"  Subtype: {tc['query_metadata'].get('query_subtype')}")
    print(f"  Query: {tc['query_text'][:100]}...")
    print(f"  Expected relevant: {tc['expected_relevant'][:5]}")
    
    # Now run the actual retrieval
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex
    from agentic_memory.retrieval import HybridRetriever
    from agentic_memory.embedding import get_llama_embedder
    from agentic_memory.config import cfg
    from agentic_memory.types import RetrievalQuery
    
    store = MemoryStore(cfg.db_path)
    index = FaissIndex(dim=int(cfg.get('embedding_dim', 1024)), index_path=cfg.index_path)
    retriever = HybridRetriever(store, index)
    embedder = get_llama_embedder()
    
    query = RetrievalQuery(
        session_id=tc['query_metadata'].get('session_id', 'benchmark_session'),
        text=tc['query_text']
    )
    
    embedding = embedder.encode([tc['query_text']], normalize_embeddings=True)[0]
    candidates = retriever.search(rq=query, qvec=embedding, topk_sem=50, topk_lex=50)
    
    retrieved_ids = [c.memory_id for c in candidates]
    print(f"\n  Retrieved {len(retrieved_ids)} candidates")
    print(f"  Top 5 retrieved: {retrieved_ids[:5]}")
    
    # Check overlap
    expected_set = set(tc['expected_relevant'])
    retrieved_set = set(retrieved_ids[:5])
    overlap = expected_set & retrieved_set
    
    print(f"\n  Overlap analysis:")
    print(f"    Expected: {expected_set}")
    print(f"    Retrieved (top 5): {retrieved_set}")
    print(f"    Intersection: {overlap}")
    print(f"    Precision@5: {len(overlap)/5 if retrieved_set else 0:.2f}")
    print(f"    Recall@5: {len(overlap)/len(expected_set) if expected_set else 0:.2f}")
    
    # Check if it's a ground truth mismatch
    if len(overlap) == 0:
        print("\n  ❌ NO OVERLAP - This is why benchmark shows 0%!")
        print("  The expected_relevant list doesn't match what's actually retrieved.")
        print("  This is a test generation problem, not a retrieval problem.")