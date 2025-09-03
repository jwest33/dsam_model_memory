from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ConceptClusterer:
    """Lightweight incremental conceptual clustering using MiniBatchKMeans.
    For production-scale volumes, you might shard and retrain periodically.
    """
    def __init__(self, n_clusters: int = 64, dim: int = 384):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, n_init='auto')
        self.dim = dim
        self._fitted = False

    def partial_fit(self, vectors: np.ndarray):
        self.model.partial_fit(vectors)
        self._fitted = True

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        if not self._fitted:
            # cold start: uniform cluster 0
            return np.zeros((vectors.shape[0],), dtype=int)
        return self.model.predict(vectors)

    def centroids(self) -> np.ndarray:
        if not self._fitted:
            return np.zeros((1, self.dim), dtype='float32')
        return self.model.cluster_centers_
