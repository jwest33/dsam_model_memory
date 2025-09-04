from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import networkx as nx

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


class LiquidMemoryClusters:
    """Self-organizing clusters that flow and merge based on access patterns"""
    
    def __init__(self, n_clusters: int = 64, dim: int = 384):
        self.base_clusterer = ConceptClusterer(n_clusters, dim)
        self.cluster_graph = nx.DiGraph()  # Directed graph of cluster relationships
        self.cluster_energy = {}  # Activation energy per cluster
        self.flow_rate = 0.1
        self.merge_threshold = 0.85  # Similarity threshold for merging
        
        # Track cluster memberships
        self.memory_clusters = {}  # memory_id -> cluster_id
        self.cluster_members = {}  # cluster_id -> set of memory_ids
        
        # Co-access history for affinity computation
        self.co_access_counts = {}  # (mem1, mem2) -> count
        self.access_timestamps = {}  # memory_id -> list of timestamps
        
    def compute_affinity_matrix(self, memory_ids: List[str], 
                                embeddings: Dict[str, np.ndarray],
                                metadata: Optional[Dict] = None) -> np.ndarray:
        """Build affinity matrix based on multiple signals"""
        n = len(memory_ids)
        affinity = np.zeros((n, n))
        
        for i, m1 in enumerate(memory_ids):
            for j, m2 in enumerate(memory_ids):
                if i >= j:
                    continue
                    
                # Co-access frequency
                key = tuple(sorted([m1, m2]))
                co_access = self.co_access_counts.get(key, 0)
                co_access_score = min(1.0, co_access / 10.0)  # Normalize to 0-1
                
                # Temporal proximity (if timestamps available)
                temporal_score = 0.5  # Default
                if m1 in self.access_timestamps and m2 in self.access_timestamps:
                    t1_latest = max(self.access_timestamps[m1]) if self.access_timestamps[m1] else datetime.now()
                    t2_latest = max(self.access_timestamps[m2]) if self.access_timestamps[m2] else datetime.now()
                    time_diff = abs((t1_latest - t2_latest).total_seconds())
                    temporal_score = np.exp(-time_diff / 86400.0)  # Daily decay
                
                # Semantic similarity from embeddings
                semantic_sim = 0.5  # Default
                if m1 in embeddings and m2 in embeddings:
                    e1 = embeddings[m1].reshape(1, -1)
                    e2 = embeddings[m2].reshape(1, -1)
                    semantic_sim = cosine_similarity(e1, e2)[0, 0]
                    semantic_sim = (semantic_sim + 1) / 2  # Normalize to 0-1
                
                # Weighted combination
                affinity[i, j] = (0.4 * co_access_score + 
                                 0.3 * temporal_score + 
                                 0.3 * semantic_sim)
                affinity[j, i] = affinity[i, j]
                
        return affinity
    
    def _compute_pull(self, memory_id: str, current_cluster: int, 
                     target_cluster: int, affinity_matrix: np.ndarray,
                     memory_ids: List[str]) -> float:
        """Compute pull strength from target cluster"""
        if current_cluster == target_cluster:
            return 0.0
            
        # Get members of target cluster
        target_members = self.cluster_members.get(target_cluster, set())
        if not target_members:
            return 0.0
            
        # Calculate average affinity to target cluster members
        mem_idx = memory_ids.index(memory_id)
        pull = 0.0
        count = 0
        
        for target_mem in target_members:
            if target_mem in memory_ids:
                target_idx = memory_ids.index(target_mem)
                pull += affinity_matrix[mem_idx, target_idx]
                count += 1
                
        if count > 0:
            pull = pull / count
            
        # Modulate by cluster energy
        energy = self.cluster_energy.get(target_cluster, 0.5)
        return pull * energy
    
    def flow_step(self, memory_ids: List[str], embeddings: Dict[str, np.ndarray],
                  metadata: Optional[Dict] = None) -> Dict[str, int]:
        """One step of gradient flow clustering"""
        # Compute affinity matrix
        affinity = self.compute_affinity_matrix(memory_ids, embeddings, metadata)
        
        # Initialize clusters if needed
        for mid in memory_ids:
            if mid not in self.memory_clusters:
                # Assign to base cluster
                if mid in embeddings:
                    vec = embeddings[mid].reshape(1, -1)
                    cluster_id = int(self.base_clusterer.assign(vec)[0])
                else:
                    cluster_id = 0
                self.memory_clusters[mid] = cluster_id
                
                if cluster_id not in self.cluster_members:
                    self.cluster_members[cluster_id] = set()
                self.cluster_members[cluster_id].add(mid)
        
        # Flow memories between clusters based on affinity gradient
        new_clusters = self.memory_clusters.copy()
        moves = {}
        
        for i, mem_id in enumerate(memory_ids):
            current_cluster = self.memory_clusters[mem_id]
            
            # Compute pull from each cluster
            pulls = {}
            for cluster_id in self.cluster_members.keys():
                pull_strength = self._compute_pull(
                    mem_id, current_cluster, cluster_id, affinity, memory_ids
                )
                pulls[cluster_id] = pull_strength
            
            # Flow to strongest pull if above threshold
            if pulls:
                max_cluster, max_pull = max(pulls.items(), key=lambda x: x[1])
                if max_pull > self.flow_rate and max_cluster != current_cluster:
                    new_clusters[mem_id] = max_cluster
                    moves[mem_id] = (current_cluster, max_cluster)
        
        # Apply moves
        for mem_id, (old_c, new_c) in moves.items():
            self.cluster_members[old_c].discard(mem_id)
            if not self.cluster_members[old_c]:
                del self.cluster_members[old_c]
            
            if new_c not in self.cluster_members:
                self.cluster_members[new_c] = set()
            self.cluster_members[new_c].add(mem_id)
            
            self.memory_clusters[mem_id] = new_c
            
        return new_clusters
    
    def merge_similar_clusters(self, embeddings: Dict[str, np.ndarray]):
        """Merge clusters that have become too similar"""
        if len(self.cluster_members) < 2:
            return
            
        # Compute cluster centroids
        centroids = {}
        for cluster_id, members in self.cluster_members.items():
            member_embeds = []
            for mem_id in members:
                if mem_id in embeddings:
                    member_embeds.append(embeddings[mem_id])
            
            if member_embeds:
                centroids[cluster_id] = np.mean(member_embeds, axis=0)
        
        # Find pairs of similar clusters
        merges = []
        cluster_ids = list(centroids.keys())
        for i, c1 in enumerate(cluster_ids):
            for c2 in cluster_ids[i+1:]:
                sim = cosine_similarity(
                    centroids[c1].reshape(1, -1),
                    centroids[c2].reshape(1, -1)
                )[0, 0]
                
                if sim > self.merge_threshold:
                    merges.append((c1, c2, sim))
        
        # Execute merges (keep cluster with more energy)
        for c1, c2, sim in merges:
            if c1 not in self.cluster_members or c2 not in self.cluster_members:
                continue  # Already merged
                
            e1 = self.cluster_energy.get(c1, 0.5)
            e2 = self.cluster_energy.get(c2, 0.5)
            
            keep, merge = (c1, c2) if e1 >= e2 else (c2, c1)
            
            # Move all members
            for mem_id in self.cluster_members[merge]:
                self.memory_clusters[mem_id] = keep
                self.cluster_members[keep].add(mem_id)
            
            # Clean up
            del self.cluster_members[merge]
            if merge in self.cluster_energy:
                del self.cluster_energy[merge]
    
    def update_cluster_energy(self, accessed_memories: List[str]):
        """Update cluster activation energy based on access"""
        for mem_id in accessed_memories:
            if mem_id in self.memory_clusters:
                cluster_id = self.memory_clusters[mem_id]
                current = self.cluster_energy.get(cluster_id, 0.5)
                # Increase energy with saturation
                self.cluster_energy[cluster_id] = min(1.0, current + 0.05)
        
        # Decay all cluster energies
        for cluster_id in list(self.cluster_energy.keys()):
            self.cluster_energy[cluster_id] *= 0.99
            if self.cluster_energy[cluster_id] < 0.1:
                del self.cluster_energy[cluster_id]
    
    def update_co_access(self, accessed_memories: List[str]):
        """Update co-access patterns"""
        now = datetime.now()
        
        # Update timestamps
        for mem_id in accessed_memories:
            if mem_id not in self.access_timestamps:
                self.access_timestamps[mem_id] = []
            self.access_timestamps[mem_id].append(now)
            # Keep only recent timestamps (last 7 days)
            cutoff = now - timedelta(days=7)
            self.access_timestamps[mem_id] = [
                t for t in self.access_timestamps[mem_id] if t > cutoff
            ]
        
        # Update co-access counts
        for i, m1 in enumerate(accessed_memories):
            for m2 in accessed_memories[i+1:]:
                key = tuple(sorted([m1, m2]))
                self.co_access_counts[key] = self.co_access_counts.get(key, 0) + 1
    
    def get_cluster_summary(self) -> Dict:
        """Get summary of current cluster state"""
        return {
            'n_clusters': len(self.cluster_members),
            'cluster_sizes': {cid: len(members) for cid, members in self.cluster_members.items()},
            'cluster_energies': dict(self.cluster_energy),
            'total_memories': len(self.memory_clusters)
        }
