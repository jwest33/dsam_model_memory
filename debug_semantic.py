#!/usr/bin/env python3
"""Debug why semantic search shows 0% precision/recall."""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.config import cfg
from agentic_memory.types import RetrievalQuery

# Load the test case
test_file = Path("benchmarks/test_data/benchmark_testset_20250909_132419.json")
with open(test_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find first semantic test case
semantic_case = next(tc for tc in data['test_cases'] if tc['test_type'] == 'semantic')

print(f"Testing semantic case: {semantic_case['test_id']}")
print(f"Query: {semantic_case['query_text'][:100]}...")
print(f"Expected relevant (first 5): {semantic_case['expected_relevant'][:5]}")

# Initialize components
store = MemoryStore(cfg.db_path)
index = FaissIndex(dim=int(cfg.get('embedding_dim', 1024)), index_path=cfg.index_path)
retriever = HybridRetriever(store, index)
embedder = get_llama_embedder()

# Create retrieval query
rq = RetrievalQuery(
    session_id=semantic_case['query_metadata'].get('session_id', 'benchmark_session'),
    text=semantic_case['query_text']
)

# Generate embedding and retrieve
print("\nGenerating embedding...")
qvec = embedder.encode([semantic_case['query_text']], normalize_embeddings=True)[0]

print("Retrieving...")
candidates = retriever.search(rq=rq, qvec=qvec, topk_sem=50, topk_lex=50)

retrieved_ids = [c.memory_id for c in candidates[:20]]
print(f"\nRetrieved {len(candidates)} total candidates")
print(f"Top 5 retrieved: {retrieved_ids[:5]}")

# Check overlap
expected_list = semantic_case['expected_relevant']
expected_set = set(expected_list)
retrieved_set = set(retrieved_ids[:5])
overlap = expected_set & retrieved_set

print(f"\nOverlap analysis:")
print(f"  Expected (first 5): {expected_list[:5]}")
print(f"  Retrieved (top 5): {list(retrieved_ids[:5])}")
print(f"  Intersection: {overlap}")
print(f"  Precision@5: {len(overlap)/5:.2f}")
print(f"  Recall@5: {len(overlap)/min(5, len(expected_list)):.2f}")

# Debug: Check if expected memories exist and have embeddings
print("\nChecking expected memories:")
for mem_id in semantic_case['expected_relevant'][:3]:
    # Check if memory exists
    mem = store.get_memory(mem_id)
    if mem:
        print(f"  {mem_id}: EXISTS, text length={len(mem.raw_text)}")
        # Check if it has an embedding in FAISS
        try:
            # Search for this specific memory
            test_results = index.search(qvec, k=100)
            found_in_faiss = any(mid == mem_id for mid, _ in test_results)
            print(f"    Found in FAISS: {found_in_faiss}")
            if found_in_faiss:
                position = next(i for i, (mid, _) in enumerate(test_results) if mid == mem_id)
                score = test_results[position][1]
                print(f"    Position: {position+1}, Score: {score:.4f}")
        except Exception as e:
            print(f"    Error checking FAISS: {e}")
    else:
        print(f"  {mem_id}: NOT FOUND in database!")