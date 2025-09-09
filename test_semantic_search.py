#!/usr/bin/env python3
"""Test semantic search to debug why it's failing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.embedding import get_llama_embedder
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.config import cfg
import numpy as np

# Test text from the benchmark
test_text = "I'm sorry, but I cannot answer that question. As an AI language model, I don't have access to your personal information, including your current location. However, if you would like to know more about"

print(f"Test text: {test_text[:100]}...")

# Initialize embedder
print("\nInitializing embedder...")
embedder = get_llama_embedder()

# Generate embedding for test text
print("Generating embedding for test text...")
test_embedding = embedder.encode([test_text], normalize_embeddings=True)[0]
print(f"Embedding shape: {test_embedding.shape}")
print(f"Embedding norm: {np.linalg.norm(test_embedding):.4f}")

# Load FAISS index
print("\nLoading FAISS index...")
index = FaissIndex(dim=int(cfg.get('embedding_dim', 1024)), index_path=cfg.index_path)
print(f"FAISS index has {len(index.id_map)} memories")

# Search for similar memories
print("\nSearching FAISS for similar memories...")
results = index.search(test_embedding, k=10)

print(f"\nTop 10 results:")
for i, (memory_id, score) in enumerate(results):
    print(f"{i+1}. {memory_id}: score={score:.4f}")

# Check if our expected memory is in the results
expected_id = "mem_5feb3e5f8cc0"
found = any(memory_id == expected_id for memory_id, _ in results)
print(f"\nExpected memory {expected_id} found: {found}")

if not found:
    # Get the position of this memory in the index
    if expected_id in index.id_map:
        idx = index.id_map.index(expected_id)
        print(f"Memory {expected_id} is at index position {idx}")
        
        # Try to get its embedding and compare
        import sqlite3
        conn = sqlite3.connect('./amemory.sqlite3')
        cursor = conn.cursor()
        cursor.execute("SELECT raw_text FROM memories WHERE memory_id = ?", (expected_id,))
        row = cursor.fetchone()
        if row:
            original_text = row[0]
            print(f"\nOriginal text: {original_text[:100]}...")
            
            # Check if texts match
            if original_text.strip() == test_text.strip():
                print("✓ Texts are IDENTICAL")
            else:
                print("✗ Texts are DIFFERENT")
                
            # Generate embedding for original and compare
            original_embedding = embedder.encode([original_text], normalize_embeddings=True)[0]
            similarity = np.dot(test_embedding, original_embedding)
            print(f"Cosine similarity between embeddings: {similarity:.4f}")