"""
Dynamic Memory Store with Dual-Space Encoding and Product Distance Metrics

This module implements the enhanced memory system with:
- Dual-space encoder (Euclidean + Hyperbolic)
- Immutable anchors with bounded adaptive residuals
- Product distance metrics for retrieval
- HDBSCAN clustering with product similarity
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import hdbscan
from sklearn.preprocessing import StandardScaler

from models.event import Event, FiveW1H, EventType
from memory.chromadb_store import ChromaDBStore
from memory.dual_space_encoder import DualSpaceEncoder, HyperbolicOperations, mobius_add
from memory.hopfield import ModernHopfieldNetwork
from config import get_config

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Unified memory store with dual-space encoding and product distance metrics.
    Features immutable anchors with bounded adaptive residuals.
    """
    
    def __init__(self):
        """Initialize the enhanced memory store"""
        self.config = get_config()
        
        # Core storage (ChromaDB is primary)
        self.chromadb = ChromaDBStore()
        
        # Dual-space encoder
        self.encoder = DualSpaceEncoder(
            euclidean_dim=768,
            hyperbolic_dim=64,
            field_weights={
                'who': 1.0,
                'what': 2.0,
                'when': 0.5,
                'where': 0.5,
                'why': 1.5,
                'how': 1.0
            }
        )
        
        # Hopfield network for associative memory
        self.hopfield = ModernHopfieldNetwork()
        
        # Residual storage and momentum tracking
        self.residuals = {}  # event_id -> {'euclidean': np.array, 'hyperbolic': np.array}
        self.momentum = {}   # event_id -> {'euclidean': np.array, 'hyperbolic': np.array}
        
        # Residual bounds
        self.max_euclidean_norm = 0.35
        self.max_hyperbolic_geodesic = 0.75
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum_factor = 0.9
        self.residual_decay = 0.995  # Decay factor for aging residuals
        
        # Episode tracking
        self.episode_map = {}  # episode_id -> List[event_id]
        
        # Cache for recent embeddings
        self.embedding_cache = {}  # event_id -> embeddings dict
        
        # Statistics - initialize from existing data
        self._initialize_statistics()
        
        logger.info("Enhanced memory store initialized with dual-space encoding")
    
    def _initialize_statistics(self):
        """Initialize statistics from existing ChromaDB data."""
        try:
            # Count existing events in ChromaDB
            from chromadb import Client
            client = Client()
            
            # Try to get the events collection if it exists
            try:
                collection = self.chromadb.client.get_collection("events")
                self.total_events = collection.count()
            except:
                self.total_events = 0
            
            self.total_queries = 0
            
            # Load any saved state
            state_file = Path("./state/residuals.pkl")
            if state_file.exists():
                self.load_state()
        except:
            self.total_events = 0
            self.total_queries = 0
    
    def store_event(self, event: Event) -> Tuple[bool, str]:
        """
        Store an event with dual-space encoding.
        Creates immutable anchors and initializes residuals to zero.
        
        Args:
            event: Event to store
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Generate dual-space embeddings
            embeddings = self.encoder.encode(event.five_w1h.to_dict())
            
            # Check for duplicates based on product distance
            similar_events = self._find_similar_events(embeddings, k=5)
            
            if similar_events:
                for similar_event, distance in similar_events:
                    if distance < 0.15:  # Very close in product space
                        # Update residuals of existing event instead of creating duplicate
                        self._merge_into_existing(similar_event.id, embeddings)
                        return True, f"Merged into existing event {similar_event.id[:8]}"
            
            # Store immutable anchors in ChromaDB
            # For ChromaDB, concatenate euclidean anchor for vector search
            chromadb_embedding = embeddings['euclidean_anchor']
            self.chromadb.store_event(event, chromadb_embedding)
            
            # Store full embeddings in cache
            self.embedding_cache[event.id] = embeddings
            
            # Initialize residuals to zero
            self.residuals[event.id] = {
                'euclidean': np.zeros_like(embeddings['euclidean_anchor']),
                'hyperbolic': np.zeros_like(embeddings['hyperbolic_anchor'])
            }
            
            # Initialize momentum to zero
            self.momentum[event.id] = {
                'euclidean': np.zeros_like(embeddings['euclidean_anchor']),
                'hyperbolic': np.zeros_like(embeddings['hyperbolic_anchor'])
            }
            
            # Add to Hopfield network (use euclidean for now)
            self.hopfield.store(
                embeddings['euclidean_anchor'],
                embeddings['euclidean_anchor'],
                metadata=event.to_dict()
            )
            
            # Track episode
            if event.episode_id not in self.episode_map:
                self.episode_map[event.episode_id] = []
            self.episode_map[event.episode_id].append(event.id)
            
            # Update statistics
            self.total_events += 1
            
            # Log growth periodically
            if self.total_events % 100 == 0:
                logger.info(f"Memory store size: {self.total_events} events")
            
            return True, f"Stored event {event.id[:8]} with dual-space encoding"
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False, str(e)
    
    def retrieve_memories(
        self,
        query: Dict[str, str],
        k: int = 10,
        use_clustering: bool = True,
        update_residuals: bool = True
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve memories using product distance and HDBSCAN clustering.
        
        Args:
            query: 5W1H query fields
            k: Number of memories to retrieve
            use_clustering: Whether to use HDBSCAN clustering
            update_residuals: Whether to update residuals based on retrieval
            
        Returns:
            List of (event, relevance_score) tuples
        """
        try:
            self.total_queries += 1
            
            # Generate query embeddings
            query_embeddings = self.encoder.encode(query)
            
            # Compute query-dependent weights
            lambda_e, lambda_h = self.encoder.compute_query_weights(query)
            
            # Initial retrieval from ChromaDB (using Euclidean prefilter)
            candidates = self.chromadb.retrieve_events_by_query(
                query_embeddings['euclidean_anchor'],
                k=min(k * 10, 200)  # Get more candidates for reranking
            )
            
            if not candidates:
                return []
            
            # Rerank using product distance
            reranked = []
            for event, _ in candidates:
                # Get full embeddings for this event
                if event.id in self.embedding_cache:
                    event_embeddings = self.embedding_cache[event.id]
                else:
                    # Reconstruct if not cached
                    event_embeddings = self.encoder.encode(event.five_w1h.to_dict())
                    self.embedding_cache[event.id] = event_embeddings
                
                # Add residuals to get effective embeddings
                event_embeddings = self._get_effective_embeddings(event.id, event_embeddings)
                
                # Compute product distance
                distance = self.encoder.compute_product_distance(
                    query_embeddings, event_embeddings, lambda_e, lambda_h
                )
                
                reranked.append((event, distance))
            
            # Sort by distance (lower is better)
            reranked.sort(key=lambda x: x[1])
            
            if use_clustering and len(reranked) > 10:
                # Apply HDBSCAN clustering
                clustered_results = self._cluster_and_rank(
                    reranked[:min(len(reranked), 100)],
                    query_embeddings,
                    lambda_e,
                    lambda_h
                )
                results = clustered_results[:k]
            else:
                # Convert distance to similarity score
                results = [(event, 1.0 / (1.0 + dist)) for event, dist in reranked[:k]]
            
            # Update residuals based on co-retrieval if enabled
            if update_residuals and len(results) > 1:
                self._update_residuals_from_retrieval(results)
            
            # Decay old residuals periodically
            if self.total_queries % 100 == 0:
                self._decay_residuals()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def _get_effective_embeddings(self, event_id: str, base_embeddings: Dict) -> Dict:
        """Get embeddings with residuals applied."""
        effective = base_embeddings.copy()
        
        if event_id in self.residuals:
            effective['euclidean_residual'] = self.residuals[event_id]['euclidean']
            effective['hyperbolic_residual'] = self.residuals[event_id]['hyperbolic']
        else:
            effective['euclidean_residual'] = np.zeros_like(base_embeddings['euclidean_anchor'])
            effective['hyperbolic_residual'] = np.zeros_like(base_embeddings['hyperbolic_anchor'])
        
        return effective
    
    def _cluster_and_rank(
        self,
        candidates: List[Tuple[Event, float]],
        query_embeddings: Dict,
        lambda_e: float,
        lambda_h: float
    ) -> List[Tuple[Event, float]]:
        """
        Apply HDBSCAN clustering and rank by centrality.
        """
        if len(candidates) < 3:
            return [(event, 1.0 / (1.0 + dist)) for event, dist in candidates]
        
        # Build feature matrix for clustering (product space)
        features = []
        events = []
        
        for event, _ in candidates:
            event_embeddings = self._get_effective_embeddings(
                event.id,
                self.embedding_cache.get(event.id, self.encoder.encode(event.five_w1h.to_dict()))
            )
            
            # Concatenate weighted Euclidean and Hyperbolic features
            eu_features = event_embeddings['euclidean_anchor'] + event_embeddings['euclidean_residual']
            hy_features = event_embeddings['hyperbolic_anchor'] + event_embeddings['hyperbolic_residual']
            
            # Weight and concatenate
            combined = np.concatenate([
                eu_features * np.sqrt(lambda_e),
                hy_features * np.sqrt(lambda_h)
            ])
            features.append(combined)
            events.append(event)
        
        features = np.array(features)
        
        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(features)
        
        # Compute centrality scores within clusters
        results = []
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                cluster_events = [events[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                # Give noise points lower scores
                for event in cluster_events:
                    idx = events.index(event)
                    dist = candidates[idx][1]
                    results.append((event, 0.5 / (1.0 + dist)))
            else:
                # Get cluster members
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_events = [events[i] for i in cluster_indices]
                
                if len(cluster_events) == 1:
                    # Single member cluster
                    event = cluster_events[0]
                    idx = events.index(event)
                    dist = candidates[idx][1]
                    results.append((event, 1.0 / (1.0 + dist)))
                else:
                    # Compute centrality within cluster
                    cluster_features = features[cluster_indices]
                    
                    # Compute pairwise similarities
                    similarities = np.zeros((len(cluster_events), len(cluster_events)))
                    for i in range(len(cluster_events)):
                        for j in range(len(cluster_events)):
                            if i != j:
                                # Cosine similarity in combined space
                                sim = np.dot(cluster_features[i], cluster_features[j]) / (
                                    np.linalg.norm(cluster_features[i]) * np.linalg.norm(cluster_features[j]) + 1e-8
                                )
                                similarities[i, j] = max(0, sim)
                    
                    # Compute eigenvector centrality
                    if similarities.sum() > 0:
                        eigenvalues, eigenvectors = np.linalg.eig(similarities)
                        idx = eigenvalues.argsort()[-1]
                        centrality = np.abs(eigenvectors[:, idx])
                        centrality = centrality / centrality.sum()
                    else:
                        centrality = np.ones(len(cluster_events)) / len(cluster_events)
                    
                    # Combine centrality with distance scores
                    for i, event in enumerate(cluster_events):
                        idx = events.index(event)
                        dist = candidates[idx][1]
                        base_score = 1.0 / (1.0 + dist)
                        final_score = base_score * (0.7 + 0.3 * centrality[i])
                        results.append((event, final_score))
        
        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _update_residuals_from_retrieval(self, results: List[Tuple[Event, float]]):
        """
        Update residuals based on co-retrieval patterns.
        Events retrieved together should gravitate towards each other.
        """
        # Only update top results
        top_results = results[:5]
        
        for i, (event_i, score_i) in enumerate(top_results):
            if event_i.id not in self.embedding_cache:
                continue
            
            for j, (event_j, score_j) in enumerate(top_results):
                if i >= j or event_j.id not in self.embedding_cache:
                    continue
                
                # Compute relevance based on scores
                relevance = min(score_i, score_j) * 0.1
                
                # Get embeddings
                embeddings_i = self._get_effective_embeddings(
                    event_i.id, self.embedding_cache[event_i.id]
                )
                embeddings_j = self._get_effective_embeddings(
                    event_j.id, self.embedding_cache[event_j.id]
                )
                
                # Update residuals for both events
                if event_i.id not in self.momentum:
                    self.momentum[event_i.id] = {
                        'euclidean': np.zeros_like(embeddings_i['euclidean_anchor']),
                        'hyperbolic': np.zeros_like(embeddings_i['hyperbolic_anchor'])
                    }
                
                updated_i = self.encoder.adapt_residuals(
                    embeddings_i, embeddings_j, relevance,
                    self.momentum[event_i.id],
                    self.learning_rate, self.momentum_factor,
                    self.max_euclidean_norm, self.max_hyperbolic_geodesic
                )
                
                # Store updated residuals
                self.residuals[event_i.id] = {
                    'euclidean': updated_i['euclidean_residual'],
                    'hyperbolic': updated_i['hyperbolic_residual']
                }
    
    def _decay_residuals(self):
        """Apply decay to all residuals to prevent unbounded drift."""
        for event_id in self.residuals:
            self.residuals[event_id]['euclidean'] *= self.residual_decay
            self.residuals[event_id]['hyperbolic'] *= self.residual_decay
            
            # Also decay momentum
            if event_id in self.momentum:
                self.momentum[event_id]['euclidean'] *= self.residual_decay
                self.momentum[event_id]['hyperbolic'] *= self.residual_decay
    
    def _find_similar_events(self, embeddings: Dict, k: int = 5) -> List[Tuple[Event, float]]:
        """Find similar events using product distance."""
        # Quick retrieval using Euclidean anchor
        candidates = self.chromadb.retrieve_events_by_query(
            embeddings['euclidean_anchor'], k=k * 2
        )
        
        if not candidates:
            return []
        
        # Rerank using product distance
        results = []
        for event, _ in candidates:
            if event.id in self.embedding_cache:
                event_embeddings = self._get_effective_embeddings(
                    event.id, self.embedding_cache[event.id]
                )
                distance = self.encoder.compute_product_distance(
                    embeddings, event_embeddings, lambda_e=0.5, lambda_h=0.5
                )
                results.append((event, distance))
        
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def _merge_into_existing(self, existing_id: str, new_embeddings: Dict):
        """Merge a new event into an existing one by updating residuals."""
        if existing_id not in self.embedding_cache:
            return
        
        # Get current embeddings
        existing_embeddings = self._get_effective_embeddings(
            existing_id, self.embedding_cache[existing_id]
        )
        
        # Initialize momentum if needed
        if existing_id not in self.momentum:
            self.momentum[existing_id] = {
                'euclidean': np.zeros_like(existing_embeddings['euclidean_anchor']),
                'hyperbolic': np.zeros_like(existing_embeddings['hyperbolic_anchor'])
            }
        
        # Adapt residuals towards new event
        updated = self.encoder.adapt_residuals(
            existing_embeddings, new_embeddings,
            relevance=0.3,  # Higher relevance for merging
            momentum=self.momentum[existing_id],
            learning_rate=self.learning_rate * 2,  # Faster adaptation for merging
            momentum_factor=self.momentum_factor,
            max_euclidean_norm=self.max_euclidean_norm,
            max_hyperbolic_geodesic=self.max_hyperbolic_geodesic
        )
        
        # Store updated residuals
        self.residuals[existing_id] = {
            'euclidean': updated['euclidean_residual'],
            'hyperbolic': updated['hyperbolic_residual']
        }
    
    def save_state(self, path: Optional[Path] = None) -> bool:
        """Save the current state of the memory store."""
        try:
            import pickle
            
            state_dir = path or Path("./state")
            state_dir.mkdir(exist_ok=True)
            
            # Save residuals and momentum
            with open(state_dir / "residuals.pkl", "wb") as f:
                pickle.dump({
                    'residuals': self.residuals,
                    'momentum': self.momentum,
                    'embedding_cache': self.embedding_cache
                }, f)
            
            # ChromaDB persists automatically
            
            logger.info(f"Saved memory state to {state_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, path: Optional[Path] = None) -> bool:
        """Load a previously saved state."""
        try:
            import pickle
            
            state_dir = path or Path("./state")
            
            # Load residuals and momentum
            residuals_file = state_dir / "residuals.pkl"
            if residuals_file.exists():
                with open(residuals_file, "rb") as f:
                    state = pickle.load(f)
                    self.residuals = state['residuals']
                    self.momentum = state['momentum']
                    self.embedding_cache = state.get('embedding_cache', {})
            
            # ChromaDB loads automatically from persistent storage
            
            logger.info(f"Loaded memory state from {state_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        return {
            'total_events': self.total_events,
            'total_queries': self.total_queries,
            'total_episodes': len(self.episode_map),
            'events_with_residuals': len(self.residuals),
            'cached_embeddings': len(self.embedding_cache),
            'average_residual_norm': self._compute_average_residual_norm()
        }
    
    def _compute_average_residual_norm(self) -> Dict[str, float]:
        """Compute average residual norms."""
        if not self.residuals:
            return {'euclidean': 0.0, 'hyperbolic': 0.0}
        
        eu_norms = [np.linalg.norm(r['euclidean']) for r in self.residuals.values()]
        hy_norms = [HyperbolicOperations.geodesic_distance(
            np.zeros_like(r['hyperbolic']), r['hyperbolic'], self.encoder.c
        ) for r in self.residuals.values()]
        
        return {
            'euclidean': np.mean(eu_norms) if eu_norms else 0.0,
            'hyperbolic': np.mean(hy_norms) if hy_norms else 0.0
        }
    
    def clear(self):
        """Clear all memories and reset state."""
        # Clear ChromaDB
        try:
            # Delete and recreate collections
            self.chromadb.client.delete_collection("events")
        except:
            pass
        
        # Reinitialize ChromaDB
        self.chromadb = ChromaDBStore()
        
        # Clear all tracking
        self.residuals.clear()
        self.momentum.clear()
        self.episode_map.clear()
        self.embedding_cache.clear()
        self.total_events = 0
        self.total_queries = 0
        
        # Clear Hopfield network
        self.hopfield = ModernHopfieldNetwork()
        
        logger.info("Cleared all memories")
    
    def _find_event_by_id(self, event_id: str) -> Optional[Event]:
        """Find an event by its ID."""
        try:
            # Try to get from ChromaDB
            result = self.chromadb.client.get_collection("events").get(
                ids=[event_id],
                include=["metadatas"]
            )
            
            if result and result['metadatas']:
                metadata = result['metadatas'][0]
                # Reconstruct Event from metadata
                event = Event(
                    id=event_id,
                    five_w1h=FiveW1H(
                        who=metadata.get('who', ''),
                        what=metadata.get('what', ''),
                        when=metadata.get('when', ''),
                        where=metadata.get('where', ''),
                        why=metadata.get('why', ''),
                        how=metadata.get('how', '')
                    ),
                    event_type=EventType(metadata.get('event_type', 'action')),
                    episode_id=metadata.get('episode_id', '')
                )
                return event
        except Exception as e:
            logger.error(f"Error finding event {event_id}: {e}")
        
        return None