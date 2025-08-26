"""
Dynamic Memory Clustering System

This module implements query-driven, context-aware memory clustering that
creates memory groups on-the-fly based on the specific 5W1H fields in queries.
No pre-computed blocks or arbitrary thresholds - everything is dynamic.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from sklearn.cluster import DBSCAN
from collections import defaultdict

from models.event import Event, FiveW1H
from embedding.embedder import FiveW1HEmbedder
from embedding.singleton_embedder import get_five_w1h_embedder, get_text_embedder

logger = logging.getLogger(__name__)

@dataclass
class DynamicCluster:
    """
    A dynamically created cluster of related memories.
    Created on-the-fly based on query context.
    """
    id: str
    events: List[Event]
    centroid: np.ndarray  # Cluster center in embedding space
    coherence: float  # How tightly grouped the cluster is
    relevance: float  # Relevance to the current query
    query_context: Dict[str, Any]  # What query created this cluster
    
    def get_salience_matrix(self) -> np.ndarray:
        """Compute salience matrix for events in this cluster"""
        n = len(self.events)
        if n == 0:
            return np.array([])
        
        matrix = np.zeros((n, n))
        
        # This is computed on-demand, not stored
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    # Compute similarity based on the query context
                    matrix[i, j] = self._contextual_similarity(
                        self.events[i], 
                        self.events[j],
                        self.query_context
                    )
        
        return matrix
    
    def _contextual_similarity(self, e1: Event, e2: Event, context: Dict) -> float:
        """
        Compute similarity between events based on query context.
        Different 5W1H fields are weighted differently based on what was queried.
        """
        weights = context.get('field_weights', {})
        similarity = 0.0
        total_weight = 0.0
        
        # Compare each 5W1H field with context-specific weighting
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            weight = weights.get(field, 0.1)
            
            val1 = getattr(e1.five_w1h, field, '')
            val2 = getattr(e2.five_w1h, field, '')
            
            if val1 and val2:
                if val1 == val2:
                    similarity += weight * 1.0
                else:
                    # Use embedding similarity for semantic comparison
                    sim = self._text_similarity(val1, val2)
                    similarity += weight * sim
            
            total_weight += weight
        
        return similarity / total_weight if total_weight > 0 else 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using embeddings"""
        try:
            embedder = get_text_embedder()
            
            emb1 = embedder.embed_text(text1)
            emb2 = embedder.embed_text(text2)
            
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 > 0 and norm2 > 0:
                return float(dot_product / (norm1 * norm2))
            return 0.0
        except:
            # Fallback to simple comparison
            return 0.5 if text1.lower() in text2.lower() or text2.lower() in text1.lower() else 0.0
    
    def get_importance_scores(self) -> np.ndarray:
        """
        Get importance scores for each event in the cluster.
        Uses eigenvector centrality on the salience matrix.
        """
        matrix = self.get_salience_matrix()
        if matrix.size == 0:
            return np.array([])
        
        # Compute eigenvector centrality
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            idx = np.argmax(np.abs(eigenvalues))
            centrality = np.abs(eigenvectors[:, idx])
            
            # Normalize
            if centrality.max() > 0:
                centrality = centrality / centrality.max()
            
            return centrality
        except:
            # Fallback to uniform importance
            return np.ones(len(self.events)) / len(self.events)


class DynamicMemoryClustering:
    """
    Dynamic clustering system that creates memory groups based on queries.
    No pre-computed blocks, no arbitrary thresholds.
    """
    
    def __init__(self, embedder: Optional[FiveW1HEmbedder] = None):
        """Initialize the dynamic clustering system"""
        self.embedder = embedder or get_five_w1h_embedder()
        self.clustering_cache = {}  # Cache recent clustering results
        self.cache_ttl = 60  # Cache for 60 seconds
        
    def cluster_by_query(
        self,
        events: List[Event],
        query: Dict[str, str],
        max_clusters: int = 10,
        min_cluster_size: int = 2,
        component_mode: Optional[str] = None
    ) -> List[DynamicCluster]:
        """
        Dynamically cluster events based on the specific query.
        
        Args:
            events: All available events
            query: 5W1H query dict
            max_clusters: Maximum number of clusters to return
            min_cluster_size: Minimum events per cluster
            component_mode: Optional mode for component-specific clustering:
                          'single' - cluster by individual 5W1H components
                          'combination' - cluster by component combinations
                          None - default weighted clustering
            
        Returns:
            List of dynamically created clusters
        """
        if not events:
            return []
        
        # Determine field weights based on query and mode
        if component_mode == 'single':
            field_weights = self._compute_single_component_weights(query)
        elif component_mode == 'combination':
            field_weights = self._compute_combination_weights(query)
        else:
            field_weights = self._compute_field_weights(query)
        
        # Get embeddings weighted by query context
        embeddings = self._get_contextual_embeddings(events, field_weights)
        
        # Dynamic clustering using DBSCAN with adaptive parameters
        eps = self._compute_adaptive_eps(embeddings, len(events))
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Create clusters
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_indices = np.where(labels == label)[0]
            cluster_events = [events[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]
            
            # Compute cluster properties
            centroid = np.mean(cluster_embeddings, axis=0)
            coherence = self._compute_coherence(cluster_embeddings, centroid)
            relevance = self._compute_relevance(centroid, query, field_weights)
            
            cluster = DynamicCluster(
                id=f"dyn_{label}_{hash(tuple(query.items()))}",
                events=cluster_events,
                centroid=centroid,
                coherence=coherence,
                relevance=relevance,
                query_context={'query': query, 'field_weights': field_weights}
            )
            
            clusters.append(cluster)
        
        # Sort by relevance and limit
        clusters.sort(key=lambda c: c.relevance, reverse=True)
        return clusters[:max_clusters]
    
    def find_contextual_neighbors(
        self,
        event: Event,
        all_events: List[Event],
        context: Dict[str, str],
        k: int = 10,
        temporal_window: Optional[timedelta] = None
    ) -> List[Tuple[Event, float]]:
        """
        Find events similar to a given event, weighted by context.
        
        Args:
            event: Target event
            all_events: Pool of events to search
            context: 5W1H context for weighting
            k: Number of neighbors
            temporal_window: Optional time constraint
            
        Returns:
            List of (event, similarity) tuples
        """
        field_weights = self._compute_field_weights(context)
        
        # Filter by temporal window if specified
        candidates = all_events
        if temporal_window:
            cutoff_time = event.created_at - temporal_window
            candidates = [e for e in all_events 
                         if e.created_at >= cutoff_time]
        
        # Compute contextual similarities
        similarities = []
        target_embedding = self._get_contextual_embedding(event, field_weights)
        
        for candidate in candidates:
            if candidate.id == event.id:
                continue
            
            cand_embedding = self._get_contextual_embedding(candidate, field_weights)
            
            # Cosine similarity
            sim = np.dot(target_embedding, cand_embedding)
            norm1 = np.linalg.norm(target_embedding)
            norm2 = np.linalg.norm(cand_embedding)
            
            if norm1 > 0 and norm2 > 0:
                sim = sim / (norm1 * norm2)
                similarities.append((candidate, float(sim)))
        
        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _compute_field_weights(self, query: Dict[str, str]) -> Dict[str, float]:
        """
        Compute importance weights for each 5W1H field based on the query.
        Fields present in the query get higher weights.
        """
        weights = {}
        query_fields = [k for k, v in query.items() if v and k in ['who', 'what', 'when', 'where', 'why', 'how']]
        
        # Base weight for all fields
        base_weight = 0.1
        
        # Higher weight for queried fields
        query_weight = 0.7 / len(query_fields) if query_fields else 0.0
        
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            if field in query_fields:
                weights[field] = base_weight + query_weight
            else:
                weights[field] = base_weight
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _compute_single_component_weights(self, query: Dict[str, str]) -> Dict[str, float]:
        """
        Compute weights focusing on individual 5W1H components.
        Each queried field gets maximum weight independently.
        """
        weights = {}
        query_fields = [k for k, v in query.items() if v and k in ['who', 'what', 'when', 'where', 'why', 'how']]
        
        # Give dominant weight to each queried field
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            if field in query_fields:
                weights[field] = 0.8  # Dominant weight
            else:
                weights[field] = 0.04  # Minimal weight for non-queried
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _compute_combination_weights(self, query: Dict[str, str]) -> Dict[str, float]:
        """
        Compute weights for component combinations.
        Creates balanced weights for specified field combinations.
        """
        weights = {}
        query_fields = [k for k, v in query.items() if v and k in ['who', 'what', 'when', 'where', 'why', 'how']]
        
        if not query_fields:
            # Default equal weights
            for field in ['who', 'what', 'when', 'where', 'why', 'how']:
                weights[field] = 1.0 / 6
        else:
            # Equal weight distribution among queried fields
            weight_per_field = 1.0 / len(query_fields)
            for field in ['who', 'what', 'when', 'where', 'why', 'how']:
                if field in query_fields:
                    weights[field] = weight_per_field
                else:
                    weights[field] = 0.0
        
        return weights
    
    def _get_contextual_embeddings(
        self,
        events: List[Event],
        field_weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Get embeddings for events weighted by field importance.
        """
        embeddings = []
        
        for event in events:
            embedding = self._get_contextual_embedding(event, field_weights)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_contextual_embedding(
        self,
        event: Event,
        field_weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Get a single event's embedding weighted by field importance.
        """
        # Get embeddings for each field
        field_embeddings = {}
        
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            value = getattr(event.five_w1h, field, '')
            if value:
                try:
                    embedder = get_text_embedder()
                    field_embeddings[field] = embedder.embed_text(value)
                except:
                    # Fallback to hash embedding
                    field_embeddings[field] = self._hash_embedding(value)
        
        # Weighted average of field embeddings
        if not field_embeddings:
            return np.zeros(768)  # Default embedding dimension
        
        weighted_sum = None
        total_weight = 0.0
        
        for field, embedding in field_embeddings.items():
            weight = field_weights.get(field, 0.1)
            
            if weighted_sum is None:
                weighted_sum = weight * embedding
            else:
                weighted_sum += weight * embedding
            
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum = weighted_sum / total_weight
        
        # Normalize
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum = weighted_sum / norm
        
        return weighted_sum
    
    def _compute_adaptive_eps(self, embeddings: np.ndarray, n_events: int) -> float:
        """
        Compute adaptive DBSCAN epsilon based on data distribution.
        """
        if n_events < 10:
            return 0.3  # Small dataset, tighter clusters
        elif n_events < 100:
            return 0.4  # Medium dataset
        else:
            # Compute based on nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            
            k = min(10, n_events // 10)
            nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
            nbrs.fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            
            # Use median of k-th nearest neighbor distances
            kth_distances = distances[:, -1]
            eps = np.median(kth_distances)
            
            # Bound between reasonable values
            return np.clip(eps, 0.2, 0.6)
    
    def _compute_coherence(self, embeddings: np.ndarray, centroid: np.ndarray) -> float:
        """
        Compute cluster coherence (how tightly grouped it is).
        """
        if len(embeddings) == 0:
            return 0.0
        
        distances = []
        for embedding in embeddings:
            dist = 1 - np.dot(embedding, centroid)  # Cosine distance
            distances.append(dist)
        
        # Lower average distance = higher coherence
        avg_distance = np.mean(distances)
        coherence = 1.0 - avg_distance  # Convert to coherence score
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def _compute_relevance(
        self,
        centroid: np.ndarray,
        query: Dict[str, str],
        field_weights: Dict[str, float]
    ) -> float:
        """
        Compute cluster relevance to the query.
        """
        # Create query embedding
        query_text = ' '.join([v for v in query.values() if v])
        if not query_text:
            return 0.5
        
        try:
            embedder = get_text_embedder()
            query_embedding = embedder.embed_text(query_text)
            
            # Normalize
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Cosine similarity with centroid
            relevance = np.dot(centroid, query_embedding)
            
            # Boost based on field match importance
            boost = sum(field_weights.get(k, 0) for k, v in query.items() if v)
            relevance = relevance * (1 + boost * 0.2)
            
            return float(np.clip(relevance, 0.0, 1.0))
        except:
            return 0.5
    
    def _hash_embedding(self, text: str, dim: int = 768) -> np.ndarray:
        """Fallback hash-based embedding"""
        import hashlib
        
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert to numpy array
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8)
        
        # Extend or truncate to desired dimension
        if len(embedding) < dim:
            embedding = np.pad(embedding, (0, dim - len(embedding)))
        else:
            embedding = embedding[:dim]
        
        # Normalize to [-1, 1]
        embedding = (embedding.astype(np.float32) - 128) / 128
        
        return embedding
    
    def merge_clusters(
        self,
        clusters: List[DynamicCluster],
        threshold: float = 0.7
    ) -> List[DynamicCluster]:
        """
        Merge similar clusters to reduce redundancy.
        """
        if len(clusters) <= 1:
            return clusters
        
        merged = []
        used = set()
        
        for i, cluster1 in enumerate(clusters):
            if i in used:
                continue
            
            # Start new merged cluster
            merged_events = list(cluster1.events)
            merged_centroid = cluster1.centroid * len(cluster1.events)
            total_events = len(cluster1.events)
            
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                
                # Check similarity
                sim = np.dot(cluster1.centroid, cluster2.centroid)
                if sim >= threshold:
                    # Merge
                    merged_events.extend(cluster2.events)
                    merged_centroid += cluster2.centroid * len(cluster2.events)
                    total_events += len(cluster2.events)
                    used.add(j)
            
            # Create merged cluster
            merged_centroid = merged_centroid / total_events
            
            merged_cluster = DynamicCluster(
                id=f"merged_{i}",
                events=merged_events,
                centroid=merged_centroid,
                coherence=cluster1.coherence,  # Recompute if needed
                relevance=cluster1.relevance,
                query_context=cluster1.query_context
            )
            
            merged.append(merged_cluster)
            used.add(i)
        
        return merged
    
    def get_temporal_clusters(
        self,
        events: List[Event],
        window_size: timedelta = timedelta(minutes=5)
    ) -> List[DynamicCluster]:
        """
        Create clusters based on temporal proximity.
        """
        if not events:
            return []
        
        # Sort by time
        sorted_events = sorted(events, key=lambda e: e.created_at)
        
        clusters = []
        current_cluster = [sorted_events[0]]
        
        for event in sorted_events[1:]:
            time_diff = event.created_at - current_cluster[-1].created_at
            
            if time_diff <= window_size:
                current_cluster.append(event)
            else:
                # Create cluster from current group
                if len(current_cluster) >= 2:
                    cluster = self._create_temporal_cluster(current_cluster)
                    clusters.append(cluster)
                
                current_cluster = [event]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            cluster = self._create_temporal_cluster(current_cluster)
            clusters.append(cluster)
        
        return clusters
    
    def cluster_by_components(
        self,
        events: List[Event],
        components: List[str],
        values: Optional[Dict[str, str]] = None,
        max_clusters: int = 10
    ) -> Dict[str, List[DynamicCluster]]:
        """
        Create clusters based on specific 5W1H components or their combinations.
        
        Args:
            events: Events to cluster
            components: List of 5W1H components to cluster by (e.g., ['who'], ['what', 'where'])
            values: Optional specific values to filter by
            max_clusters: Maximum clusters per component/combination
            
        Returns:
            Dictionary mapping component combinations to their clusters
        """
        result = {}
        
        # Filter events if specific values provided
        filtered_events = events
        if values:
            filtered_events = [
                e for e in events
                if all(
                    getattr(e.five_w1h, comp, None) == val 
                    for comp, val in values.items()
                    if comp in components
                )
            ]
        
        # Single component clustering
        if len(components) == 1:
            component = components[0]
            
            # Group events by component value
            component_groups = defaultdict(list)
            for event in filtered_events:
                value = getattr(event.five_w1h, component, None)
                if value:
                    component_groups[value].append(event)
            
            # Create clusters for each unique value
            clusters = []
            for value, group_events in component_groups.items():
                if len(group_events) >= 2:  # Minimum cluster size
                    query = {component: value}
                    sub_clusters = self.cluster_by_query(
                        group_events, query, max_clusters=1, component_mode='single'
                    )
                    if sub_clusters:
                        clusters.extend(sub_clusters)
            
            result[component] = clusters[:max_clusters]
        
        # Multiple component clustering (combinations)
        else:
            # Create composite key from components
            composite_key = '+'.join(sorted(components))
            
            # Build query from specified components
            query = {}
            for comp in components:
                if values and comp in values:
                    query[comp] = values[comp]
                else:
                    query[comp] = ''  # Will match any value
            
            clusters = self.cluster_by_query(
                filtered_events, query, max_clusters=max_clusters, component_mode='combination'
            )
            
            result[composite_key] = clusters
        
        return result
    
    def _create_temporal_cluster(self, events: List[Event]) -> DynamicCluster:
        """Create a cluster from temporally related events"""
        # Compute centroid
        embeddings = []
        for event in events:
            key, _ = self.embedder.embed_event(event)
            embeddings.append(key)
        
        centroid = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return DynamicCluster(
            id=f"temporal_{events[0].created_at.timestamp()}",
            events=events,
            centroid=centroid,
            coherence=0.8,  # Temporal clusters are usually coherent
            relevance=0.5,  # Neutral relevance until queried
            query_context={'type': 'temporal'}
        )
