#!/usr/bin/env python3
"""Debug the full retrieval pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.config import cfg
from agentic_memory.types import RetrievalQuery
import numpy as np

# Test case from benchmark
test_text = "I'm sorry, but I cannot answer that question. As an AI language model, I don't have access to your personal information, including your current location. However, if you would like to know more about"
expected_id = "mem_5feb3e5f8cc0"

print(f"Test text: {test_text[:100]}...")
print(f"Expected memory ID: {expected_id}")

# Initialize components as benchmark does
store = MemoryStore(cfg.db_path)
index = FaissIndex(dim=int(cfg.get('embedding_dim', 1024)), index_path=cfg.index_path)
retriever = HybridRetriever(store, index)
embedder = get_llama_embedder()

# Create retrieval query
retrieval_query = RetrievalQuery(
    session_id='benchmark_session',
    text=test_text,
    actor_hint=None,
    spatial_hint=None,
    temporal_hint=None
)

# Generate embedding
print("\nGenerating embedding...")
query_embedding = embedder.encode([test_text], normalize_embeddings=True)[0]

# Call retriever.search as benchmark does
print("Calling retriever.search()...")
candidates = retriever.search(
    rq=retrieval_query,
    qvec=query_embedding,
    topk_sem=50,
    topk_lex=50
)

print(f"\nReturned {len(candidates)} candidates")
print("\nTop 10 candidates:")
for i, cand in enumerate(candidates[:10]):
    print(f"{i+1}. {cand.memory_id}: score={cand.score:.4f}")
    
# Check if expected memory is in results
found = any(c.memory_id == expected_id for c in candidates)
print(f"\nExpected memory {expected_id} found: {found}")

if found:
    position = next(i for i, c in enumerate(candidates) if c.memory_id == expected_id)
    print(f"Found at position: {position + 1}")
else:
    print("NOT FOUND - This is why semantic search fails in benchmark!")
    
    # Debug: Check what happened in each step
    print("\n--- Debugging retrieval steps ---")
    
    # Test semantic search directly
    sem_results = index.search(query_embedding, k=10)
    print(f"\nDirect FAISS search found {len(sem_results)} results")
    for i, (mid, score) in enumerate(sem_results[:5]):
        print(f"  {i+1}. {mid}: {score:.4f}")
    
    # Test lexical search directly  
    lex_results = store.lexical_search(test_text, k=10)
    print(f"\nDirect FTS search found {len(lex_results)} results")
    for i, r in enumerate(lex_results[:5]):
        print(f"  {i+1}. {r['memory_id']}: {r['score']:.4f}")