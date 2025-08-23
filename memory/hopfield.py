"""
Modern Hopfield Network implementation for associative memory

Uses attention-based retrieval mechanism similar to transformers.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from config import get_config

logger = logging.getLogger(__name__)

class ModernHopfieldNetwork:
    """
    Modern Hopfield Network with attention-based retrieval
    
    Memory Read: output = softmax(β · q·K^T) @ V
    where:
        q: query vector
        K: key matrix (stored queries)
        V: value matrix (stored observations)
        β: inverse temperature (sharpness)
    """
    
    def __init__(self, config=None):
        """Initialize the network"""
        self.config = config or get_config().memory
        
        # Memory matrices
        self.keys = None  # K matrix: stored query embeddings
        self.values = None  # V matrix: stored value embeddings
        self.metadata = []  # Associated metadata for each memory
        
        # Memory management
        self.memory_count = 0
        self.max_capacity = self.config.max_memory_slots
        self.embedding_dim = self.config.embedding_dim
        
        # Parameters
        self.beta = self.config.temperature  # Inverse temperature
        self.learning_rate = self.config.base_learning_rate
        
        # Initialize empty memory
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize empty memory matrices"""
        self.keys = np.zeros((self.max_capacity, self.embedding_dim), dtype=np.float32)
        self.values = np.zeros((self.max_capacity, self.embedding_dim), dtype=np.float32)
        self.metadata = [None] * self.max_capacity
        self.memory_count = 0
    
    def store(
        self,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        salience: float = 1.0
    ) -> int:
        """
        Store a key-value pair in memory
        
        Args:
            key: Query embedding (d-dimensional)
            value: Value embedding (d-dimensional)
            metadata: Optional metadata to associate
            salience: Importance weight for learning rate
        
        Returns:
            Index of stored memory
        """
        # Normalize inputs
        key = self._normalize(key)
        value = self._normalize(value)
        
        # Check for similar existing memory
        if self.memory_count > 0:
            similarities = self._compute_similarities(key, value)
            best_match_idx = np.argmax(similarities)
            best_match_sim = similarities[best_match_idx]
            
            # Update existing memory if similar enough
            if best_match_sim >= self.config.similarity_threshold:
                self._update_memory(best_match_idx, key, value, metadata, salience)
                return best_match_idx
        
        # Add new memory
        if self.memory_count >= self.max_capacity:
            # Evict least important memory
            evict_idx = self._select_eviction_candidate()
            self._add_memory(evict_idx, key, value, metadata)
            return evict_idx
        else:
            # Add to next available slot
            idx = self.memory_count
            self._add_memory(idx, key, value, metadata)
            self.memory_count += 1
            return idx
    
    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Tuple[np.ndarray, Optional[Dict], float]]:
        """
        Retrieve top-k memories using attention mechanism
        
        Args:
            query: Query vector (d-dimensional)
            k: Number of memories to retrieve
            return_scores: Whether to return attention scores
        
        Returns:
            List of (value, metadata, score) tuples
        """
        if self.memory_count == 0:
            return []
        
        # Normalize query
        query = self._normalize(query)
        
        # Compute attention scores: softmax(β · q·K^T)
        active_keys = self.keys[:self.memory_count]
        scores = self.beta * np.dot(query, active_keys.T)
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Get top-k indices
        k = min(k, self.memory_count)
        top_k_indices = np.argsort(attention_weights)[-k:][::-1]
        
        # Retrieve values and metadata
        results = []
        for idx in top_k_indices:
            value = self.values[idx].copy()
            meta = self.metadata[idx].copy() if self.metadata[idx] else None
            score = float(attention_weights[idx]) if return_scores else 0.0
            results.append((value, meta, score))
        
        return results
    
    def retrieve_weighted(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve weighted combination of all memories
        
        Args:
            query: Query vector
        
        Returns:
            Weighted sum of values
        """
        if self.memory_count == 0:
            return np.zeros(self.embedding_dim)
        
        # Normalize query
        query = self._normalize(query)
        
        # Compute attention: softmax(β · q·K^T)
        active_keys = self.keys[:self.memory_count]
        scores = self.beta * np.dot(query, active_keys.T)
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Weighted sum: attention @ V
        active_values = self.values[:self.memory_count]
        output = np.dot(attention_weights, active_values)
        
        return output
    
    def _compute_similarities(self, key: np.ndarray, value: np.ndarray) -> np.ndarray:
        """Compute combined similarity with existing memories"""
        if self.memory_count == 0:
            return np.array([])
        
        # Compute key similarities
        active_keys = self.keys[:self.memory_count]
        key_sims = np.dot(active_keys, key)
        
        # Compute value similarities
        active_values = self.values[:self.memory_count]
        value_sims = np.dot(active_values, value)
        
        # Weighted combination
        combined_sims = (
            self.config.query_weight * key_sims +
            self.config.content_weight * value_sims
        )
        
        return combined_sims
    
    def _update_memory(
        self,
        idx: int,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict],
        salience: float
    ):
        """Update existing memory with exponential moving average"""
        # Adaptive learning rate based on salience
        alpha = self.learning_rate * salience
        
        # EMA update
        self.keys[idx] = self._normalize((1 - alpha) * self.keys[idx] + alpha * key)
        self.values[idx] = self._normalize((1 - alpha) * self.values[idx] + alpha * value)
        
        # Update metadata if provided
        if metadata:
            if self.metadata[idx]:
                # Merge metadata
                self.metadata[idx].update(metadata)
            else:
                self.metadata[idx] = metadata.copy()
    
    def _add_memory(
        self,
        idx: int,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict]
    ):
        """Add new memory at specified index"""
        self.keys[idx] = key
        self.values[idx] = value
        self.metadata[idx] = metadata.copy() if metadata else None
    
    def _select_eviction_candidate(self) -> int:
        """Select memory to evict based on priority scoring"""
        if self.memory_count == 0:
            return 0
        
        # Simple strategy: evict oldest (FIFO)
        # In a full implementation, would use priority scores from metadata
        scores = []
        
        for i in range(self.memory_count):
            meta = self.metadata[i]
            if meta and 'priority_score' in meta:
                scores.append(meta['priority_score'])
            else:
                # Default high score (likely to evict)
                scores.append(float('inf'))
        
        # Return index of highest score (most likely to evict)
        return int(np.argmax(scores))
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def clear(self):
        """Clear all memories"""
        self._initialize_memory()
        logger.info("Cleared all memories from Hopfield network")
    
    def save_state(self, path: Path):
        """Save network state to file"""
        state = {
            'keys': self.keys[:self.memory_count],
            'values': self.values[:self.memory_count],
            'metadata': self.metadata[:self.memory_count],
            'memory_count': self.memory_count,
            'config': {
                'max_capacity': self.max_capacity,
                'embedding_dim': self.embedding_dim,
                'beta': self.beta,
                'learning_rate': self.learning_rate
            }
        }
        
        np.savez_compressed(path, **state)
        logger.info(f"Saved Hopfield network state to {path} ({self.memory_count} memories)")
    
    def load_state(self, path: Path):
        """Load network state from file"""
        if not path.exists():
            logger.warning(f"State file not found: {path}")
            return
        
        try:
            state = np.load(path, allow_pickle=True)
            
            # Load memories
            loaded_keys = state['keys']
            loaded_values = state['values']
            loaded_metadata = state['metadata'].tolist()
            loaded_count = int(state['memory_count'])
            
            # Validate dimensions
            if loaded_keys.shape[1] != self.embedding_dim:
                raise ValueError(f"Dimension mismatch: expected {self.embedding_dim}, got {loaded_keys.shape[1]}")
            
            # Load into current network
            self._initialize_memory()
            self.memory_count = min(loaded_count, self.max_capacity)
            
            self.keys[:self.memory_count] = loaded_keys[:self.memory_count]
            self.values[:self.memory_count] = loaded_values[:self.memory_count]
            self.metadata[:self.memory_count] = loaded_metadata[:self.memory_count]
            
            logger.info(f"Loaded {self.memory_count} memories from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load state from {path}: {e}")
            self._initialize_memory()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics"""
        stats = {
            'memory_count': self.memory_count,
            'max_capacity': self.max_capacity,
            'utilization': self.memory_count / self.max_capacity if self.max_capacity > 0 else 0,
            'embedding_dim': self.embedding_dim,
            'temperature': self.beta
        }
        
        if self.memory_count > 0:
            # Compute average key/value norms
            active_keys = self.keys[:self.memory_count]
            active_values = self.values[:self.memory_count]
            
            stats['avg_key_norm'] = float(np.mean(np.linalg.norm(active_keys, axis=1)))
            stats['avg_value_norm'] = float(np.mean(np.linalg.norm(active_values, axis=1)))
            
            # Compute diversity (average pairwise distance)
            if self.memory_count > 1:
                key_similarity_matrix = np.dot(active_keys, active_keys.T)
                # Exclude diagonal
                mask = ~np.eye(self.memory_count, dtype=bool)
                avg_similarity = float(np.mean(key_similarity_matrix[mask]))
                stats['avg_key_similarity'] = avg_similarity
                stats['diversity'] = 1.0 - avg_similarity
            else:
                stats['avg_key_similarity'] = 0.0
                stats['diversity'] = 1.0
        
        return stats
    
    def __len__(self) -> int:
        """Get number of stored memories"""
        return self.memory_count
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ModernHopfieldNetwork(memories={self.memory_count}/{self.max_capacity}, "
            f"dim={self.embedding_dim}, β={self.beta})"
        )