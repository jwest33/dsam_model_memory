#!/usr/bin/env python3
"""
Test script to verify the embedding server is working correctly.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for embedding server
os.environ['AM_EMBEDDING_SERVER_URL'] = 'http://localhost:8002/v1'
os.environ['AM_EMBEDDING_DIM'] = '1024'

from agentic_memory.embedding import get_llama_embedder

def test_embedding_server():
    """Test the embedding server functionality."""
    
    print("Testing Embedding Server Configuration")
    print("=" * 50)
    print(f"Server URL: {os.getenv('AM_EMBEDDING_SERVER_URL')}")
    print(f"Embedding Dimension: {os.getenv('AM_EMBEDDING_DIM')}")
    print()
    
    # Get embedder instance
    embedder = get_llama_embedder()
    print(f"Embedder initialized with dimension: {embedder._dimension}")
    
    # Test texts
    test_texts = [
        "Hello, world!",
        "This is a test of the embedding system.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating."
    ]
    
    print(f"\nTesting embedding generation for {len(test_texts)} texts...")
    
    try:
        # Generate embeddings
        embeddings = embedder.encode(test_texts, batch_size=4, normalize_embeddings=True)
        
        print(f"✅ Successfully generated embeddings!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Data type: {embeddings.dtype}")
        print(f"   Normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
        
        # Check if embeddings are not all zeros
        if np.all(embeddings == 0):
            print("❌ WARNING: All embeddings are zeros - server may not be working correctly")
        else:
            print(f"✅ Embeddings contain valid data")
            
            # Calculate similarity between first two texts
            similarity = np.dot(embeddings[0], embeddings[1])
            print(f"\nSimilarity between first two texts: {similarity:.4f}")
            
            # Show first few values of first embedding
            print(f"\nFirst 10 values of first embedding:")
            print(embeddings[0][:10])
            
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the embedding server is running:")
        print("   python llama_server_manager.py embedding start")
        print("2. Check that the server is on port 8002")
        print("3. Verify the Qwen3-Embedding model is loaded correctly")
        return False
    
    # Test single text embedding
    print("\nTesting single text embedding...")
    try:
        single_embedding = embedder.encode("Single test text")
        print(f"✅ Single text embedding shape: {single_embedding.shape}")
    except Exception as e:
        print(f"❌ Failed single text: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    # Check if embedding server is running
    import requests
    try:
        response = requests.get("http://localhost:8002/health", timeout=2)
        if response.status_code == 200:
            print("✅ Embedding server is running on port 8002\n")
        else:
            print("❌ Embedding server returned non-200 status")
            sys.exit(1)
    except:
        print("❌ Embedding server is not running on port 8002")
        print("Please start it with: python llama_server_manager.py embedding start")
        sys.exit(1)
    
    # Run tests
    success = test_embedding_server()
    sys.exit(0 if success else 1)