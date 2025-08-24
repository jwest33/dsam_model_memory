"""
Dynamic Modern Hopfield Network for associative memory

This implementation removes the arbitrary memory limit and uses dynamic arrays
that grow as needed, suitable for use with ChromaDB as primary storage.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from config import get_config

logger = logging.getLogger(__name__)

class DynamicHopfieldNetwork:
    """
    Dynamic Modern Hopfield Network without arbitrary capacity limits.
    
    Uses growing arrays and efficient operations for scalability.
    The network can grow indefinitely as memories are added.
    """
    
    def __init__(self, config=None):
        """Initialize the dynamic Hopfield network"""
        self.config = config or get_config().memory
        
        # Core memory storage - starts small and grows as needed
        self.keys = []  # List of key embeddings
        self.values = []  # List of value embeddings
        self.metadata = []  # Associated metadata for each memory
        
        # Memory management
        self.memory_count = 0
        self.embedding_dim = self.config.embedding_dim
        
        # Parameters
        self.beta = self.config.temperature  # Inverse temperature
        self.learning_rate = self.config.base_learning_rate
        
        # Index for fast lookup
        self.memory_index = {}  # memory_id -> array index
        
        # Statistics
        self.access_counts = {}  # Index -> access count
        self.last_access = {}  # Index -> timestamp
        
        logger.info(f"Initialized Dynamic Hopfield Network (no capacity limit)")
    
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
        
        # Check for similar existing memory if we have any
        if self.memory_count > 0:
            similarities = self._compute_similarities(key, value)
            best_match_idx = np.argmax(similarities)
            best_match_sim = similarities[best_match_idx]
            
            # Update existing memory if similar enough
            if best_match_sim >= self.config.similarity_threshold:
                self._update_memory(best_match_idx, key, value, metadata, salience)
                return best_match_idx
        
        # Add new memory - no capacity limit!
        idx = self._add_memory(key, value, metadata)
        return idx
    
    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[np.ndarray, Dict[str, Any], float]]:
        """
        Retrieve memories using attention mechanism
        
        Args:
            query: Query embedding
            k: Number of results to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (value, metadata, score) tuples
        """
        if self.memory_count == 0:
            return []
        
        query = self._normalize(query)
        
        # Convert lists to arrays for efficient computation
        K = np.array(self.keys[:self.memory_count])
        V = np.array(self.values[:self.memory_count])
        
        # Compute attention scores: softmax(β * q·K^T)
        scores = np.dot(K, query) * self.beta
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Get top-k indices
        top_k_indices = np.argsort(attention_weights)[-k:][::-1]
        
        # Filter by threshold and prepare results
        results = []
        for idx in top_k_indices:
            score = float(attention_weights[idx])
            if score >= threshold:
                value = V[idx]
                metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                
                # Update access statistics
                self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
                
                results.append((value, metadata, score))
        
        return results
    
    def retrieve_by_metadata(
        self,
        filter_fn,
        k: int = 5
    ) -> List[Tuple[np.ndarray, Dict[str, Any], int]]:
        """
        Retrieve memories by metadata filter
        
        Args:
            filter_fn: Function that takes metadata dict and returns bool
            k: Maximum number of results
        
        Returns:
            List of (value, metadata, index) tuples
        """
        results = []
        for i in range(self.memory_count):
            if i < len(self.metadata) and self.metadata[i]:
                if filter_fn(self.metadata[i]):
                    value = self.values[i] if i < len(self.values) else None
                    if value is not None:
                        results.append((value, self.metadata[i], i))
                        if len(results) >= k:
                            break
        return results
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Get memory by its ID (stored in metadata)"""
        for i in range(self.memory_count):
            if i < len(self.metadata) and self.metadata[i]:
                if self.metadata[i].get('event_id') == memory_id:
                    return (
                        self.keys[i] if i < len(self.keys) else None,
                        self.values[i] if i < len(self.values) else None,
                        self.metadata[i]
                    )
        return None
    
    def remove_memory(self, idx: int) -> bool:
        """
        Remove a memory at given index
        
        Args:
            idx: Index to remove
        
        Returns:
            Success status
        """
        if idx >= self.memory_count:
            return False
        
        # Remove from lists (this is O(n) but happens rarely)
        if idx < len(self.keys):
            del self.keys[idx]
        if idx < len(self.values):
            del self.values[idx]
        if idx < len(self.metadata):
            del self.metadata[idx]
        
        # Update count
        self.memory_count -= 1
        
        # Update indices for all subsequent items
        for meta_idx in range(idx, len(self.metadata)):
            if self.metadata[meta_idx] and 'event_id' in self.metadata[meta_idx]:
                event_id = self.metadata[meta_idx]['event_id']
                if event_id in self.memory_index:
                    self.memory_index[event_id] = meta_idx
        
        return True
    
    def _add_memory(
        self,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict[str, Any]]
    ) -> int:
        """Add a new memory to the network"""
        # Append to lists
        self.keys.append(key)
        self.values.append(value)
        self.metadata.append(metadata or {})
        
        # Update index
        idx = self.memory_count
        if metadata and 'event_id' in metadata:
            self.memory_index[metadata['event_id']] = idx
        
        self.memory_count += 1
        
        # Log growth periodically
        if self.memory_count % 1000 == 0:
            logger.info(f"Hopfield network grown to {self.memory_count} memories")
        
        return idx
    
    def _update_memory(
        self,
        idx: int,
        key: np.ndarray,
        value: np.ndarray,
        metadata: Optional[Dict[str, Any]],
        salience: float
    ):
        """Update existing memory using EMA"""
        if idx >= len(self.keys) or idx >= len(self.values):
            return
        
        # Compute adaptive learning rate
        alpha = self.learning_rate * salience * self.config.salience_ema_alpha
        
        # Update using exponential moving average
        self.keys[idx] = (1 - alpha) * self.keys[idx] + alpha * key
        self.values[idx] = (1 - alpha) * self.values[idx] + alpha * value
        
        # Renormalize
        self.keys[idx] = self._normalize(self.keys[idx])
        self.values[idx] = self._normalize(self.values[idx])
        
        # Update metadata
        if metadata and idx < len(self.metadata):
            self.metadata[idx].update(metadata)
    
    def _compute_similarities(
        self,
        key: np.ndarray,
        value: np.ndarray
    ) -> np.ndarray:
        """Compute combined similarity with existing memories"""
        if self.memory_count == 0:
            return np.array([])
        
        # Convert to arrays for computation
        K = np.array(self.keys[:self.memory_count])
        V = np.array(self.values[:self.memory_count])
        
        # Compute similarities
        key_similarities = np.dot(K, key)
        value_similarities = np.dot(V, value)
        
        # Weighted combination
        combined = (self.config.query_weight * key_similarities + 
                   self.config.content_weight * value_similarities)
        
        return combined
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            "memory_count": self.memory_count,
            "embedding_dim": self.embedding_dim,
            "beta": self.beta,
            "total_accesses": sum(self.access_counts.values()),
            "unique_accessed": len(self.access_counts),
            "growth_rate": self.memory_count / max(1, len(self.access_counts))  # Memories per access
        }
    
    def save_state(self, filepath: str):
        """Save network state to file"""
        import pickle
        state = {
            'keys': self.keys,
            'values': self.values,
            'metadata': self.metadata,
            'memory_count': self.memory_count,
            'memory_index': self.memory_index,
            'access_counts': self.access_counts
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved Hopfield state with {self.memory_count} memories")
    
    def load_state(self, filepath: str):
        """Load network state from file"""
        import pickle
        from pathlib import Path
        
        if not Path(filepath).exists():
            logger.warning(f"State file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.keys = state.get('keys', [])
            self.values = state.get('values', [])
            self.metadata = state.get('metadata', [])
            self.memory_count = state.get('memory_count', 0)
            self.memory_index = state.get('memory_index', {})
            self.access_counts = state.get('access_counts', {})
            
            logger.info(f"Loaded Hopfield state with {self.memory_count} memories")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def clear(self):
        """Clear all memories"""
        self.keys = []
        self.values = []
        self.metadata = []
        self.memory_count = 0
        self.memory_index = {}
        self.access_counts = {}
        logger.info("Cleared all memories from Hopfield network")
    
    def __len__(self) -> int:
        """Number of memories stored"""
        return self.memory_count
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DynamicHopfieldNetwork(memories={self.memory_count}, dim={self.embedding_dim})"