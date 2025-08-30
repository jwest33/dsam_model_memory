"""
Similarity Cache System for Pre-computed Memory Comparisons

This module provides efficient caching of pairwise similarity scores between memories,
computing them when memories are stored/loaded rather than during retrieval.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Set
from collections import defaultdict
import json
import pickle
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class SimilarityCache:
    """
    Manages pre-computed similarity scores between memory embeddings.
    
    Features:
    - Incremental updates when new memories are added
    - Batch computation for efficiency
    - Persistent storage in ChromaDB
    - Memory-efficient sparse storage (only stores similarities above threshold)
    """
    
    def __init__(self, similarity_threshold: float = 0.2):
        """
        Initialize similarity cache.
        
        Args:
            similarity_threshold: Minimum similarity to store (saves memory)
        """
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[Tuple[str, str], float] = {}  # (id1, id2) -> similarity
        self.embeddings: Dict[str, Dict] = {}  # id -> embeddings dict
        self.dirty_ids: Set[str] = set()  # IDs that need similarity updates
        self.stats = {
            'total_pairs': 0,
            'cached_pairs': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time_ms': 0
        }
        
    def add_embedding(self, event_id: str, embeddings: Dict, update_similarities: bool = True):
        """
        Add a new embedding to the cache and optionally compute similarities.
        
        Args:
            event_id: Event ID
            embeddings: Dictionary with euclidean/hyperbolic embeddings
            update_similarities: Whether to compute similarities immediately
        """
        self.embeddings[event_id] = embeddings
        self.dirty_ids.add(event_id)
        
        if update_similarities:
            self._update_similarities_for_id(event_id)
    
    def batch_add_embeddings(self, embeddings_dict: Dict[str, Dict]):
        """
        Add multiple embeddings at once and batch compute similarities.
        
        Args:
            embeddings_dict: Dictionary mapping event_id -> embeddings
        """
        import time
        start = time.time()
        
        # Add all embeddings first
        for event_id, embeddings in embeddings_dict.items():
            self.embeddings[event_id] = embeddings
            self.dirty_ids.add(event_id)
        
        # Batch compute similarities
        self._batch_update_similarities()
        
        elapsed_ms = (time.time() - start) * 1000
        self.stats['computation_time_ms'] += elapsed_ms
        logger.info(f"Batch computed similarities for {len(embeddings_dict)} events in {elapsed_ms:.1f}ms")
    
    def get_similarity(self, id1: str, id2: str, lambda_e: float = 0.5, lambda_h: float = 0.5) -> Optional[float]:
        """
        Get cached similarity between two events.
        
        Args:
            id1: First event ID
            id2: Second event ID
            lambda_e: Euclidean space weight
            lambda_h: Hyperbolic space weight
            
        Returns:
            Similarity score or None if not cached
        """
        # Ensure consistent ordering
        key = (min(id1, id2), max(id1, id2))
        
        if key in self.cache:
            self.stats['cache_hits'] += 1
            # Adjust for space weights if needed
            base_sim = self.cache[key]
            # Note: base_sim was computed with equal weights, may need adjustment
            return base_sim
        else:
            self.stats['cache_misses'] += 1
            # Compute on-demand if embeddings available
            if id1 in self.embeddings and id2 in self.embeddings:
                sim = self._compute_similarity(id1, id2, lambda_e, lambda_h)
                if sim > self.similarity_threshold:
                    self.cache[key] = sim
                return sim
            return None
    
    def get_all_similarities_for_id(self, event_id: str) -> Dict[str, float]:
        """
        Get all cached similarities for a specific event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Dictionary mapping other event IDs to similarity scores
        """
        similarities = {}
        
        for (id1, id2), sim in self.cache.items():
            if id1 == event_id:
                similarities[id2] = sim
            elif id2 == event_id:
                similarities[id1] = sim
        
        return similarities
    
    def get_top_k_similar(self, event_id: str, k: int = 10, exclude_ids: Set[str] = None) -> List[Tuple[str, float]]:
        """
        Get top-k most similar events to a given event.
        
        Args:
            event_id: Event ID
            k: Number of similar events to return
            exclude_ids: Set of IDs to exclude from results
            
        Returns:
            List of (event_id, similarity) tuples
        """
        similarities = self.get_all_similarities_for_id(event_id)
        
        if exclude_ids:
            similarities = {id: sim for id, sim in similarities.items() if id not in exclude_ids}
        
        # Sort by similarity descending
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_sims[:k]
    
    def _update_similarities_for_id(self, event_id: str):
        """Update similarities between one event and all others."""
        if event_id not in self.embeddings:
            return
        
        for other_id in self.embeddings:
            if other_id != event_id:
                key = (min(event_id, other_id), max(event_id, other_id))
                if key not in self.cache:
                    sim = self._compute_similarity(event_id, other_id)
                    if sim > self.similarity_threshold:
                        self.cache[key] = sim
        
        self.dirty_ids.discard(event_id)
    
    def _batch_update_similarities(self):
        """Batch compute similarities for all dirty IDs."""
        dirty_list = list(self.dirty_ids)
        n = len(dirty_list)
        
        if n == 0:
            return
        
        # Compute pairwise similarities for dirty IDs
        for i in range(n):
            id1 = dirty_list[i]
            if id1 not in self.embeddings:
                continue
                
            # Compute with all other embeddings
            for id2 in self.embeddings:
                if id1 != id2:
                    key = (min(id1, id2), max(id1, id2))
                    if key not in self.cache:
                        sim = self._compute_similarity(id1, id2)
                        if sim > self.similarity_threshold:
                            self.cache[key] = sim
        
        self.dirty_ids.clear()
        self.stats['total_pairs'] = len(self.embeddings) * (len(self.embeddings) - 1) // 2
        self.stats['cached_pairs'] = len(self.cache)
    
    def _compute_similarity(self, id1: str, id2: str, lambda_e: float = 0.5, lambda_h: float = 0.5) -> float:
        """
        Compute similarity between two events using dual-space embeddings.
        
        Args:
            id1: First event ID
            id2: Second event ID
            lambda_e: Euclidean space weight
            lambda_h: Hyperbolic space weight
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embeddings.get(id1)
        emb2 = self.embeddings.get(id2)
        
        if not emb1 or not emb2:
            return 0.0
        
        # Euclidean similarity (cosine)
        eu1 = emb1.get('euclidean_anchor', np.zeros(768)) + emb1.get('euclidean_residual', np.zeros(768))
        eu2 = emb2.get('euclidean_anchor', np.zeros(768)) + emb2.get('euclidean_residual', np.zeros(768))
        
        eu_sim = np.dot(eu1, eu2) / (np.linalg.norm(eu1) * np.linalg.norm(eu2) + 1e-8)
        eu_sim = max(0, eu_sim)  # Clamp to [0, 1]
        
        # Hyperbolic similarity (using Poincaré distance)
        hy1 = emb1.get('hyperbolic_anchor', np.zeros(64)) + emb1.get('hyperbolic_residual', np.zeros(64))
        hy2 = emb2.get('hyperbolic_anchor', np.zeros(64)) + emb2.get('hyperbolic_residual', np.zeros(64))
        
        # Compute Poincaré distance and convert to similarity
        norm1 = np.linalg.norm(hy1)
        norm2 = np.linalg.norm(hy2)
        
        # Ensure points are in Poincaré ball (norm < 1)
        if norm1 >= 1:
            hy1 = hy1 * 0.95 / norm1
            norm1 = 0.95
        if norm2 >= 1:
            hy2 = hy2 * 0.95 / norm2
            norm2 = 0.95
        
        # Poincaré distance
        diff_norm = np.linalg.norm(hy1 - hy2)
        poincare_dist = np.arccosh(1 + 2 * diff_norm**2 / ((1 - norm1**2) * (1 - norm2**2) + 1e-8))
        
        # Convert distance to similarity (exponential decay)
        hy_sim = np.exp(-poincare_dist / 2.0)
        
        # Weighted combination
        similarity = lambda_e * eu_sim + lambda_h * hy_sim
        
        return float(similarity)
    
    def remove_embedding(self, event_id: str):
        """
        Remove an embedding and its associated similarities from cache.
        
        Args:
            event_id: Event ID to remove
        """
        if event_id in self.embeddings:
            del self.embeddings[event_id]
        
        # Remove all similarities involving this ID
        keys_to_remove = []
        for key in self.cache:
            if event_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        self.dirty_ids.discard(event_id)
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.embeddings.clear()
        self.dirty_ids.clear()
        self.stats = {
            'total_pairs': 0,
            'cached_pairs': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'computation_time_ms': 0
        }
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            **self.stats,
            'num_embeddings': len(self.embeddings),
            'num_dirty': len(self.dirty_ids),
            'cache_size': len(self.cache),
            'hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        }
    
    def to_dict(self) -> Dict:
        """Serialize cache to dictionary for storage."""
        # Convert numpy arrays to lists for JSON serialization
        embeddings_serialized = {}
        for event_id, emb_dict in self.embeddings.items():
            embeddings_serialized[event_id] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in emb_dict.items()
            }
        
        return {
            'cache': {f"{k[0]}_{k[1]}": v for k, v in self.cache.items()},
            'embeddings': embeddings_serialized,
            'dirty_ids': list(self.dirty_ids),
            'stats': self.stats,
            'similarity_threshold': self.similarity_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimilarityCache':
        """Deserialize cache from dictionary."""
        cache = cls(similarity_threshold=data.get('similarity_threshold', 0.2))
        
        # Restore cache
        for key_str, sim in data.get('cache', {}).items():
            id1, id2 = key_str.split('_', 1)
            cache.cache[(id1, id2)] = sim
        
        # Restore embeddings
        for event_id, emb_dict in data.get('embeddings', {}).items():
            cache.embeddings[event_id] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in emb_dict.items()
            }
        
        cache.dirty_ids = set(data.get('dirty_ids', []))
        cache.stats = data.get('stats', cache.stats)
        
        return cache