"""
LLama.cpp embedding client for fast GPU-accelerated embeddings.
Uses a dedicated embedding server for optimal performance.
"""
from typing import List, Union, Optional
import numpy as np
import requests
import json
import os
from ..config import cfg
import logging

logger = logging.getLogger(__name__)

class LlamaEmbedder:
    """Fast embeddings using dedicated llama.cpp embedding server."""
    
    def __init__(self, base_url: Optional[str] = None, normalize: bool = True):
        # Use separate embedding server on port 8002 by default
        self.base_url = base_url or os.getenv('AM_EMBEDDING_SERVER_URL', 'http://localhost:8002/v1')
        self.normalize = normalize
        self._dimension = int(os.getenv('AM_EMBEDDING_DIM', '1024'))  # Qwen3-Embedding dimension
        
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension from the model."""
        return self._dimension
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        normalize_embeddings: bool = None,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings using llama.cpp server.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            normalize_embeddings: Whether to normalize (defaults to self.normalize)
            show_progress_bar: Ignored for compatibility
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if normalize_embeddings is None:
            normalize_embeddings = self.normalize
            
        embeddings = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._get_embeddings_batch(batch)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings, dtype='float32')
        
        # Normalize if requested
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts with retry logic."""
        url = f"{self.base_url}/embeddings"
        
        # Prepare request
        payload = {
            "input": texts
        }
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract embeddings from response
                embeddings = []
                for item in data.get('data', []):
                    embedding = item.get('embedding', [])
                    embeddings.append(np.array(embedding, dtype='float32'))
                
                if len(embeddings) == len(texts):
                    return embeddings
                else:
                    logger.warning(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
                    # Try again
                    continue
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Embedding request timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to get embeddings from llama.cpp (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
            except Exception as e:
                logger.error(f"Unexpected error getting embeddings: {e}")
                break
        
        # After all retries failed, return zero embeddings
        logger.error(f"All embedding attempts failed for {len(texts)} texts. Using zero embeddings.")
        return [np.zeros(self._dimension, dtype='float32') for _ in texts]

# Singleton instance
_llama_embedder = None

def get_llama_embedder() -> LlamaEmbedder:
    """Get or create the global LlamaEmbedder instance."""
    global _llama_embedder
    if _llama_embedder is None:
        _llama_embedder = LlamaEmbedder()
    return _llama_embedder

