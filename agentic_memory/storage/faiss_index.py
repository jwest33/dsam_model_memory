from __future__ import annotations
import faiss
import numpy as np
import os
from typing import List, Tuple

class FaissIndex:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.id_map = []  # integer position -> memory_id
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # id_map loaded separately
            map_path = index_path + '.map'
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    self.id_map = [line.strip() for line in f]
        else:
            # cosine similarity -> normalize and use Inner Product
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
            self.index.hnsw.efConstruction = 80

    def _persist_map(self):
        with open(self.index_path + '.map', 'w', encoding='utf-8') as f:
            for mid in self.id_map:
                f.write(mid + '\n')

    def save(self):
        faiss.write_index(self.index, self.index_path)
        self._persist_map()

    def add(self, memory_id: str, vec: np.ndarray):
        # Expect vec already L2-normalized
        vec = vec.astype('float32')[None, :]
        self.index.add(vec)
        self.id_map.append(memory_id)

    def search(self, qvec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        qvec = qvec.astype('float32')[None, :]
        scores, idxs = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            memory_id = self.id_map[idx]
            # Skip deleted entries (marked as empty string)
            if memory_id and memory_id != "":
                results.append((memory_id, float(score)))
        return results
    
    def remove(self, memory_id: str):
        """Remove a memory from the index by ID"""
        # Note: FAISS doesn't support direct removal, so we just mark it in id_map
        # A more complete implementation would rebuild the index periodically
        try:
            idx = self.id_map.index(memory_id)
            # Mark as deleted by setting to None or empty string
            self.id_map[idx] = ""
        except ValueError:
            pass  # Memory not in index
    
    def reset(self):
        """Clear the entire index"""
        self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = 64
        self.index.hnsw.efConstruction = 80
        self.id_map = []
        # Save the empty index
        self.save()
