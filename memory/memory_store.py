"""
Dynamic Memory Store with Dual-Space Encoding and Product Distance Metrics

This module implements the enhanced memory system with:
- Dual-space encoder (Euclidean + Hyperbolic)
- Immutable anchors with bounded adaptive residuals
- Product distance metrics for retrieval
- HDBSCAN clustering with product similarity
- Raw event preservation with merged presentation layer
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import logging
from pathlib import Path
import hdbscan
from sklearn.preprocessing import StandardScaler
import uuid

from models.event import Event, FiveW1H, EventType
from models.merged_event import MergedEvent, EventRelationship
from memory.qdrant_store import QdrantStore
from memory.dual_space_encoder import DualSpaceEncoder, HyperbolicOperations, mobius_add
from memory.temporal_query import integrate_temporal_with_dual_space
from memory.smart_merger import SmartMerger
from memory.temporal_manager import TemporalManager
from memory.temporal_chain import TemporalChain
from memory.context_generator import MergedEventContextGenerator
from config import get_config

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Unified memory store with dual-space encoding and product distance metrics.
    Features immutable anchors with bounded adaptive residuals.
    Now includes raw event tracking alongside merged events.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the enhanced memory store
        
        Args:
            llm_client: Optional LLM client for enhanced features
        """
        self.config = get_config()
        self.llm_client = llm_client
        
        # Core storage - Qdrant is the primary storage backend
        self.db_store = QdrantStore(self.config)
        
        # Raw events are now handled directly in memory
        
        # Raw event tracking (in memory for fast access)
        self.raw_events = {}  # raw_event_id -> Event
        self.merged_to_raw = {}  # merged_event_id -> Set[raw_event_ids]
        self.raw_to_merged = {}  # raw_event_id -> merged_event_id
        
        # New merging components
        self.smart_merger = SmartMerger(similarity_threshold=0.85)
        
        # Multi-dimensional merger for handling all dimensional grouping
        from memory.multi_dimensional_merger import MultiDimensionalMerger
        self.multi_merger = MultiDimensionalMerger(
            storage_backend=self.db_store,
            llm_client=llm_client
        )
        
        # Unified temporal manager with LLM support if available
        self.temporal_manager = TemporalManager(
            encoder=None,  # Will be set after encoder is initialized
            storage_backend=self.db_store,
            similarity_cache=None,  # Will be set after similarity cache is initialized
            config=self.config,
            llm_client=llm_client
        )
        
        # Legacy temporal chain reference for backward compatibility
        self.temporal_chain = self.temporal_manager.temporal_chain
        
        self.context_generator = MergedEventContextGenerator(self.temporal_chain)
        self.merged_events_cache = {}  # merged_event_id -> MergedEvent
        
        # Load existing merged events from storage
        self._load_merged_events_from_db()
        
        # Dual-space encoder with config values
        self.encoder = DualSpaceEncoder(
            euclidean_dim=self.config.dual_space.euclidean_dim,
            hyperbolic_dim=self.config.dual_space.hyperbolic_dim,
            field_weights={
                'who': 1.0,
                'what': 2.0,
                'when': 0.5,
                'where': 0.5,
                'why': 1.5,
                'how': 1.0
            },
            max_norm=self.config.dual_space.max_norm,
            epsilon=self.config.dual_space.epsilon
        )
        
        # Residual storage and momentum tracking
        self.residuals = {}  # event_id -> {'euclidean': np.array, 'hyperbolic': np.array}
        self.momentum = {}   # event_id -> {'euclidean': np.array, 'hyperbolic': np.array}
        
        # Residual bounds from config
        self.max_euclidean_norm = self.config.dual_space.euclidean_bound
        self.max_hyperbolic_geodesic = self.config.dual_space.hyperbolic_bound
        self.use_relative_bounds = self.config.dual_space.use_relative_bounds
        
        # Learning parameters from config
        self.learning_rate = self.config.dual_space.learning_rate
        self.momentum_factor = self.config.dual_space.momentum
        self.residual_decay = self.config.dual_space.decay_factor
        self.min_residual_norm = self.config.dual_space.min_residual_norm
        
        # Field adaptation limits from config
        self.field_adaptation_limits = self.config.dual_space.field_adaptation_limits
        self.enable_forgetting = self.config.dual_space.enable_forgetting
        
        # HDBSCAN parameters from config
        self.hdbscan_min_cluster_size = self.config.dual_space.hdbscan_min_cluster_size
        self.hdbscan_min_samples = self.config.dual_space.hdbscan_min_samples
        
        # Episode tracking
        self.episode_map = {}  # episode_id -> List[event_id]
        
        # Cache for recent embeddings
        self.embedding_cache = {}  # event_id -> embeddings dict
        
        # Statistics - initialize from existing data
        self.total_events = 0
        self.total_queries = 0  # Track number of queries made
        self._initialize_statistics()
        
        # Update temporal manager with encoder and similarity cache
        self.temporal_manager.encoder = self.encoder
        if hasattr(self, 'similarity_cache'):
            self.temporal_manager.similarity_cache = self.similarity_cache
        
        logger.info("Enhanced memory store initialized with dual-space encoding")
    
    def _initialize_statistics(self):
        """Initialize statistics from existing storage data."""
        try:
            # Get statistics from Qdrant
            stats = self.db_store.get_statistics()
            self.total_events = stats.get('total_events', 0)
            
            # Load existing embeddings into similarity cache if needed
            if self.total_events > 0:
                self._load_embeddings_to_cache()
            
            self.total_queries = 0
            
            # Load any saved state
            state_file = Path("./state/residuals.pkl")
            if state_file.exists():
                self.load_state()
        except Exception as e:
            logger.debug(f"Could not initialize statistics: {e}")
            self.total_events = 0
            self.total_queries = 0
    
    def _load_merged_events_from_db(self):
        """Load all merged events from storage on initialization."""
        try:
            # The merged_events collection now stores multi-dimensional merge groups
            # created by MultiDimensionalMerger, not standard MergedEvent objects.
            # Standard merged events are kept in memory only.
            # Skip loading for now since we don't have standard merged events persisted.
            logger.info("Skipping merged events load - using multi-dimensional merging only")
                
        except Exception as e:
            logger.warning(f"Could not load merged events from storage: {e}")
    
    def _load_embeddings_to_cache(self):
        """Load all existing embeddings into similarity cache."""
        try:
            # For now, skip loading embeddings to cache on initialization
            # This will be populated as events are accessed
            logger.info("Skipping embedding cache load - will populate on demand")
        except Exception as e:
            logger.warning(f"Could not load embeddings to cache: {e}")
    
    def _init_raw_events_collection(self):
        """Initialize the raw events collection (deprecated)"""
        # Raw events are now handled in memory, not in the database
        pass
    
    def _load_raw_events_from_db(self):
        """Load existing raw events from storage (deprecated)"""
        # Raw events are now handled in memory, not in the database
        pass
        return
        try:
            # This code is deprecated and no longer used
            results = None
            
            for i, metadata in enumerate(results['metadatas']):
                raw_id = results['ids'][i]
                merged_id = metadata.get('merged_id')
                
                # Reconstruct event
                event = Event(
                    id=metadata.get('original_id', raw_id.replace('raw_', '')),
                    five_w1h=FiveW1H(
                        who=metadata.get('who', ''),
                        what=metadata.get('what', ''),
                        when=metadata.get('when', ''),
                        where=metadata.get('where', ''),
                        why=metadata.get('why', ''),
                        how=metadata.get('how', '')
                    ),
                    event_type=EventType(metadata.get('event_type', 'observation')),
                    episode_id=metadata.get('episode_id', '')
                )
                
                # Store in memory
                self.raw_events[raw_id] = event
                
                # Track merge mappings
                if merged_id:
                    self.raw_to_merged[raw_id] = merged_id
                    if merged_id not in self.merged_to_raw:
                        self.merged_to_raw[merged_id] = set()
                    # Store the raw_id as is (with raw_ prefix) to match store_event behavior
                    self.merged_to_raw[merged_id].add(raw_id)
            
            logger.info(f"Loaded {len(self.raw_events)} raw events from storage")
            
        except Exception as e:
            logger.warning(f"Could not load raw events: {e}")
    
    def store_event(self, event: Event, preserve_raw: bool = True) -> Tuple[bool, str]:
        """
        Store an event with dual-space encoding.
        Creates immutable anchors and initializes residuals to zero.
        Optionally preserves raw event while merging for display.
        
        Args:
            event: Event to store
            preserve_raw: If True, stores raw event even when merging
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Use the event's original ID without any prefix
            raw_event_id = event.id
            
            # Generate dual-space embeddings first
            embeddings = self.encoder.encode(event.five_w1h.to_dict())
            
            # Calculate actual space weights from embeddings
            euclidean_weight, hyperbolic_weight = self.encoder.calculate_space_activation(embeddings)
            
            # Store raw event if preserving
            if preserve_raw:
                self.raw_events[raw_event_id] = event
                
                # Raw events are stored in memory only
            
            # NO MERGING/DEDUPLICATION HERE!
            # The MultiDimensionalMerger handles all grouping
            # We just store events individually and let the merger organize them
            
            # Add to temporal chain (for conversation tracking)
            self.temporal_manager.add_event_to_temporal_chain(event, {})
            
            # Store immutable anchors in storage
            # Calculate actual space weights from embeddings
            euclidean_weight, hyperbolic_weight = self.encoder.calculate_space_activation(embeddings)
            
            # Store with both Euclidean and Hyperbolic embeddings
            euclidean_embedding = embeddings['euclidean_anchor']
            hyperbolic_embedding = embeddings['hyperbolic_anchor']
            
            self.db_store.store_event(event, euclidean_embedding,
                                     euclidean_weight=euclidean_weight,
                                     hyperbolic_weight=hyperbolic_weight,
                                     hyperbolic_embedding=hyperbolic_embedding)
            
            # Track raw-to-merged mapping for new merged event
            if preserve_raw:
                self.raw_to_merged[raw_event_id] = event.id
                if event.id not in self.merged_to_raw:
                    self.merged_to_raw[event.id] = set()
                self.merged_to_raw[event.id].add(raw_event_id)
                
                # Raw event to merged mapping is tracked in memory only
            
            # Store full embeddings in cache
            self.embedding_cache[event.id] = embeddings
            
            # Update similarity cache in storage
            self.db_store.update_similarity_cache(event.id, embeddings)
            
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
            
            # Track episode
            if event.episode_id not in self.episode_map:
                self.episode_map[event.episode_id] = []
            self.episode_map[event.episode_id].append(event.id)
            
            # Process with multi-dimensional merger for dimensional grouping
            # This handles actor, temporal, conceptual, and spatial grouping
            try:
                # Prepare embeddings for multi-merger (needs euclidean_anchor and hyperbolic_anchor)
                merger_embeddings = {
                    'euclidean_anchor': embeddings['euclidean_anchor'],
                    'hyperbolic_anchor': embeddings['hyperbolic_anchor']
                }
                
                # Process the event through all dimensional mergers
                merge_assignments = self.multi_merger.process_new_event(event, merger_embeddings)
                
                # Log the dimensional assignments
                if merge_assignments:
                    logger.debug(f"Event {event.id[:8]} assigned to dimensions: {list(merge_assignments.keys())}")
            except Exception as e:
                # Don't fail the whole storage if dimensional merging fails
                logger.warning(f"Multi-dimensional merging failed for event {event.id[:8]}: {e}")
            
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
        update_residuals: bool = True,
        use_temporal: bool = True
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve memories using product distance and HDBSCAN clustering.
        
        Args:
            query: 5W1H query fields
            k: Number of memories to retrieve
            use_clustering: Whether to use HDBSCAN clustering
            update_residuals: Whether to update residuals based on retrieval
            use_temporal: Whether to apply temporal weighting for time-aware queries
            
        Returns:
            List of (event, relevance_score) tuples
        """
        try:
            self.total_queries += 1
            
            # Generate query embeddings
            query_embeddings = self.encoder.encode(query)
            
            # Compute query-dependent weights
            lambda_e, lambda_h = self.encoder.compute_query_weights(query)
            
            # Initial retrieval using dual-space if available
            # Retrieve using dual-space search
            candidates = self.db_store.retrieve_events_by_query(
                query_embeddings['euclidean_anchor'],
                k=min(k * 10, 200),  # Get more candidates for reranking
                hyperbolic_query=query_embeddings['hyperbolic_anchor'],
                lambda_e=lambda_e,
                lambda_h=lambda_h
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
            
            # Apply temporal weighting if enabled
            if use_temporal:
                results, temporal_context = integrate_temporal_with_dual_space(
                    query, results, self.encoder, lambda_e, lambda_h
                )
                if temporal_context.get('applied', False):
                    logger.info(f"Temporal weighting applied: {temporal_context['temporal_type']} "
                              f"(strength: {temporal_context['temporal_strength']:.2f})")
            
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
    
    def retrieve_memories_with_context(
        self,
        query: Dict[str, str],
        k: int = 10,
        use_clustering: bool = True,
        update_residuals: bool = True,
        use_temporal: bool = True,
        context_format: str = 'detailed'
    ) -> List[Tuple[Any, float, str]]:
        """
        Retrieve memories with full context generation for LLM consumption.
        
        Args:
            query: 5W1H query fields
            k: Number of memories to retrieve
            use_clustering: Whether to use HDBSCAN clustering
            update_residuals: Whether to update residuals based on retrieval
            use_temporal: Whether to apply temporal weighting
            context_format: 'summary', 'detailed', or 'structured'
            
        Returns:
            List of (event, relevance_score, context_string) tuples
        """
        # Get basic retrieval results
        results = self.retrieve_memories(
            query, k, use_clustering, update_residuals, use_temporal
        )
        
        enhanced_results = []
        for event, score in results:
            # Check if this is part of a merged event
            merged_event = None
            
            # Check if event ID maps to a merged event
            if event.id in self.merged_events_cache:
                merged_event = self.merged_events_cache[event.id]
            elif f"merged_{event.id}" in self.merged_events_cache:
                merged_event = self.merged_events_cache[f"merged_{event.id}"]
            else:
                # Try to load from storage
                merged_event = self.db_store.get_merged_event(f"merged_{event.id}")
                if merged_event:
                    self.merged_events_cache[merged_event.id] = merged_event
            
            if merged_event:
                # Generate rich context using the context generator
                query_context = {'query': query, 'score': score}
                context = self.context_generator.generate_context(
                    merged_event, query_context, context_format
                )
                enhanced_results.append((merged_event, score, context))
            else:
                # Single event - generate simple context
                context = self._generate_simple_event_context(event, context_format)
                enhanced_results.append((event, score, context))
        
        return enhanced_results
    
    def _generate_simple_event_context(self, event: Event, format_type: str) -> str:
        """Generate context for a single non-merged event"""
        if format_type == 'summary':
            who = event.five_w1h.who or "Unknown"
            what = event.five_w1h.what or "performed action"
            when = event.five_w1h.when or ""
            return f"{who}: {what}" + (f" ({when})" if when else "")
        
        elif format_type == 'structured':
            lines = [
                f"EVENT_ID: {event.id}",
                f"WHO: {event.five_w1h.who or 'N/A'}",
                f"WHAT: {event.five_w1h.what or 'N/A'}",
                f"WHEN: {event.five_w1h.when or 'N/A'}",
                f"WHERE: {event.five_w1h.where or 'N/A'}",
                f"WHY: {event.five_w1h.why or 'N/A'}",
                f"HOW: {event.five_w1h.how or 'N/A'}"
            ]
            return "\n".join(lines)
        
        else:  # detailed
            parts = []
            if event.five_w1h.who:
                parts.append(f"Actor: {event.five_w1h.who}")
            if event.five_w1h.what:
                parts.append(f"Action: {event.five_w1h.what}")
            if event.five_w1h.when:
                parts.append(f"Time: {event.five_w1h.when}")
            if event.five_w1h.where:
                parts.append(f"Location: {event.five_w1h.where}")
            if event.five_w1h.why:
                parts.append(f"Reason: {event.five_w1h.why}")
            if event.five_w1h.how:
                parts.append(f"Method: {event.five_w1h.how}")
            return "\n".join(parts) if parts else "No details available"
    
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
        
        # Apply HDBSCAN with config parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
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
                
                # Apply field-level adaptation limits
                field_relevance = self._apply_field_limits(
                    event_i, event_j, relevance
                )
                
                updated_i = self.encoder.adapt_residuals(
                    embeddings_i, embeddings_j, field_relevance,
                    self.momentum[event_i.id],
                    self.learning_rate, self.momentum_factor,
                    self._get_scale_aware_bound(embeddings_i, 'euclidean'),
                    self._get_scale_aware_bound(embeddings_i, 'hyperbolic')
                )
                
                # Store updated residuals
                self.residuals[event_i.id] = {
                    'euclidean': updated_i['euclidean_residual'],
                    'hyperbolic': updated_i['hyperbolic_residual']
                }
                
                # Update provenance tracking
                self._update_provenance(event_i.id, event_j.id, score_i)
    
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
        # Quick retrieval using dual-space
        candidates = self.db_store.retrieve_events_by_query(
            embeddings['euclidean_anchor'], 
            k=k * 2,
            hyperbolic_query=embeddings.get('hyperbolic_anchor')
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
    
    def _compute_centroid_embedding(self, merged_event: MergedEvent) -> np.ndarray:
        """
        Compute the centroid embedding for a merged event.
        
        Args:
            merged_event: The merged event
            
        Returns:
            Centroid embedding vector
        """
        embeddings_list = []
        
        # Collect embeddings for all raw events
        for raw_id in merged_event.raw_event_ids:
            if raw_id in self.embedding_cache:
                emb = self.embedding_cache[raw_id]
                # Get effective embedding with residuals
                effective = self._get_effective_embeddings(raw_id, emb)
                combined = effective['euclidean_anchor'] + effective['euclidean_residual']
                embeddings_list.append(combined)
        
        if not embeddings_list:
            # Fallback: encode dominant pattern
            if merged_event.dominant_pattern:
                embeddings = self.encoder.encode(merged_event.dominant_pattern)
                return embeddings['euclidean_anchor']
            else:
                return np.zeros(self.config.dual_space.euclidean_dim)
        
        # Compute centroid
        centroid = np.mean(embeddings_list, axis=0)
        return centroid
    
    def _update_merged_embeddings(self, merged_event_id: str, new_embeddings: Dict):
        """
        Update the embeddings for a merged event by incorporating new event.
        
        Args:
            merged_event_id: ID of the merged event
            new_embeddings: Embeddings of the new event being merged
        """
        if merged_event_id not in self.embedding_cache:
            # Initialize with new embeddings
            self.embedding_cache[merged_event_id] = new_embeddings
            self.residuals[merged_event_id] = {
                'euclidean': np.zeros_like(new_embeddings['euclidean_anchor']),
                'hyperbolic': np.zeros_like(new_embeddings['hyperbolic_anchor'])
            }
        else:
            # Update existing embeddings using momentum
            existing = self.embedding_cache[merged_event_id]
            
            if merged_event_id not in self.momentum:
                self.momentum[merged_event_id] = {
                    'euclidean': np.zeros_like(existing['euclidean_anchor']),
                    'hyperbolic': np.zeros_like(existing['hyperbolic_anchor'])
                }
            
            # Adapt towards new embeddings
            updated = self.encoder.adapt_residuals(
                existing, new_embeddings,
                relevance=0.3,
                momentum=self.momentum[merged_event_id],
                learning_rate=self.learning_rate,
                momentum_factor=self.momentum_factor,
                max_euclidean_norm=self.max_euclidean_norm,
                max_hyperbolic_geodesic=self.max_hyperbolic_geodesic
            )
            
            # Update residuals
            self.residuals[merged_event_id] = {
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
            
            # Storage persists automatically
            
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
            
            # Storage loads automatically
            
            logger.info(f"Loaded memory state from {state_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def get_raw_events_for_merged(self, merged_event_id: str) -> List[Event]:
        """Get all raw events associated with a merged event"""
        # First try the merged_to_raw mapping (which may have raw_ prefixed IDs)
        raw_event_ids = self.merged_to_raw.get(merged_event_id, set())
        
        # If empty, try getting from the merged event itself
        if not raw_event_ids and merged_event_id in self.merged_events_cache:
            merged_event = self.merged_events_cache[merged_event_id]
            raw_event_ids = merged_event.raw_event_ids
        
        logger.info(f"Looking for raw events for merged {merged_event_id}: {list(raw_event_ids)[:5]}...")
        
        events = []
        for rid in raw_event_ids:
            # Check both with and without raw_ prefix
            # Some IDs in merged_to_raw have the prefix, some in raw_event_ids don't
            found = False
            
            # Just look up the ID directly without prefix manipulation
            if rid in self.raw_events:
                event = self.raw_events[rid]
                events.append(event)
                found = True
            
            if found and len(events) == 1:  # Log first event
                logger.info(f"First raw event 5W1H - who: {event.five_w1h.who}, what: {event.five_w1h.what[:50] if event.five_w1h.what else 'None'}")
            elif not found:
                logger.warning(f"Raw event not found: {rid}")
        
        logger.info(f"Found {len(events)} raw events for merged {merged_event_id}")
        return events
    
    def get_merged_event_for_raw(self, raw_event_id: str) -> Optional[str]:
        """Get the merged event ID that a raw event belongs to"""
        return self.raw_to_merged.get(raw_event_id)
    
    def get_all_raw_events(self) -> Dict[str, Event]:
        """Get all raw events"""
        return self.raw_events.copy()
    
    def get_merge_groups(self) -> Dict[str, List[str]]:
        """Get all merge groups (merged_id -> list of raw_ids)"""
        return {mid: list(rids) for mid, rids in self.merged_to_raw.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory store."""
        # Raw events are memory-only now, no database collection
        total_raw = len(self.raw_events)
        
        total_merged = len(self.merged_to_raw)
        avg_merge_size = sum(len(rids) for rids in self.merged_to_raw.values()) / max(1, total_merged) if total_merged > 0 else 0
        
        return {
            'total_events': self.total_events,
            'total_raw_events': total_raw,
            'total_merged_groups': total_merged,
            'average_merge_size': avg_merge_size,
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
        # Clear storage
        self.db_store.clear_all()
        
        # Reinitialize storage
        self.db_store = QdrantStore(self.config)
        
        # Clear all tracking
        self.residuals.clear()
        self.momentum.clear()
        self.episode_map.clear()
        self.embedding_cache.clear()
        self.total_events = 0
        self.total_queries = 0
        
        logger.info("Cleared all memories")
    
    def _find_event_by_id(self, event_id: str) -> Optional[Event]:
        """Find an event by its ID."""
        try:
            # Try to get from storage using Qdrant's get_event method
            event = self.db_store.get_event(event_id)
            if event:
                return event
            
            # If not found, return None
            return None
        except Exception as e:
            logger.error(f"Error finding event {event_id}: {e}")
        
        return None
    
    def _get_scale_aware_bound(self, embeddings: Dict, space: str) -> float:
        """
        Calculate scale-aware bound based on anchor norm.
        Uses relative bounds when enabled, otherwise fixed bounds.
        """
        if not self.use_relative_bounds:
            if space == 'euclidean':
                return self.max_euclidean_norm
            else:
                return self.max_hyperbolic_geodesic
        
        # Use relative bounds based on anchor norm
        if space == 'euclidean':
            anchor_norm = np.linalg.norm(embeddings['euclidean_anchor'])
            return min(self.max_euclidean_norm, anchor_norm * self.max_euclidean_norm)
        else:
            # For hyperbolic space, use geodesic distance from origin
            origin = np.zeros_like(embeddings['hyperbolic_anchor'])
            anchor_dist = HyperbolicOperations.geodesic_distance(
                origin, embeddings['hyperbolic_anchor'], self.encoder.c, self.encoder.max_norm
            )
            return min(self.max_hyperbolic_geodesic, anchor_dist * self.max_hyperbolic_geodesic)
    
    def _apply_field_limits(self, event_i: Event, event_j: Event, relevance: float) -> float:
        """
        Apply field-level adaptation limits based on which fields are dominant.
        Limits adaptation for sensitive fields like 'who' and 'when'.
        """
        if not self.field_adaptation_limits:
            return relevance
        
        # Determine dominant fields based on content
        dominant_fields = []
        
        # Check which fields have significant content  
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            field_i = getattr(event_i.five_w1h, field, '')
            field_j = getattr(event_j.five_w1h, field, '')
            
            # Both fields must have significant content (more than 3 chars)
            if field_i and field_j and len(str(field_i)) > 3 and len(str(field_j)) > 3:
                dominant_fields.append(field)
        
        if not dominant_fields:
            return relevance
        
        # Apply minimum limit from dominant fields
        min_limit = 1.0
        for field in dominant_fields:
            if field in self.field_adaptation_limits:
                min_limit = min(min_limit, self.field_adaptation_limits[field])
        
        return relevance * min_limit
    
    def forget_residuals(self, event_ids: Optional[List[str]] = None, field: Optional[str] = None):
        """
        Zero out residuals for specific events or fields.
        Implements forgetting for drift hygiene.
        
        Args:
            event_ids: List of event IDs to forget (None = all)
            field: Specific field to forget (applies to all events)
        """
        if not self.enable_forgetting:
            logger.warning("Forgetting is disabled in configuration")
            return
        
        if event_ids is None:
            event_ids = list(self.residuals.keys())
        
        for event_id in event_ids:
            if event_id not in self.residuals:
                continue
            
            if field:
                # Selective field forgetting would require field-specific residuals
                # For now, reduce residuals proportionally
                reduction_factor = self.field_adaptation_limits.get(field, 0.5)
                self.residuals[event_id]['euclidean'] *= reduction_factor
                self.residuals[event_id]['hyperbolic'] *= reduction_factor
            else:
                # Complete forgetting
                self.residuals[event_id]['euclidean'] = np.zeros_like(
                    self.residuals[event_id]['euclidean']
                )
                self.residuals[event_id]['hyperbolic'] = np.zeros_like(
                    self.residuals[event_id]['hyperbolic']
                )
            
            # Also reset momentum
            if event_id in self.momentum:
                if field:
                    reduction_factor = self.field_adaptation_limits.get(field, 0.5)
                    self.momentum[event_id]['euclidean'] *= reduction_factor
                    self.momentum[event_id]['hyperbolic'] *= reduction_factor
                else:
                    self.momentum[event_id]['euclidean'] = np.zeros_like(
                        self.momentum[event_id]['euclidean']
                    )
                    self.momentum[event_id]['hyperbolic'] = np.zeros_like(
                        self.momentum[event_id]['hyperbolic']
                    )
        
        logger.info(f"Forgot residuals for {len(event_ids)} events")
    
    def clean_small_residuals(self):
        """Remove residuals below minimum norm threshold."""
        cleaned = 0
        for event_id in list(self.residuals.keys()):
            eu_norm = np.linalg.norm(self.residuals[event_id]['euclidean'])
            hy_norm = np.linalg.norm(self.residuals[event_id]['hyperbolic'])
            
            if eu_norm < self.min_residual_norm:
                self.residuals[event_id]['euclidean'] = np.zeros_like(
                    self.residuals[event_id]['euclidean']
                )
                cleaned += 1
            
            if hy_norm < self.min_residual_norm:
                self.residuals[event_id]['hyperbolic'] = np.zeros_like(
                    self.residuals[event_id]['hyperbolic']
                )
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} small residuals")
    
    def _update_provenance(self, event_id: str, partner_id: str, score: float):
        """Update provenance tracking for an event."""
        try:
            # Get current residual norms
            eu_norm = np.linalg.norm(self.residuals.get(event_id, {}).get('euclidean', 0))
            hy_norm = np.linalg.norm(self.residuals.get(event_id, {}).get('hyperbolic', 0))
            
            # Get current provenance
            current_provenance = self.db_store.get_provenance(event_id) or {}
            
            # Update co-retrieval partners
            partners = current_provenance.get('co_retrieval_partners', [])
            if partner_id not in partners:
                partners.append(partner_id)
            
            # Prepare provenance update
            provenance_data = {
                'residual_norm_euclidean': eu_norm,
                'residual_norm_hyperbolic': hy_norm,
                'last_accessed': datetime.now().isoformat(),
                'access_count': current_provenance.get('access_count', 0) + 1,
                'co_retrieval_partners': partners[-20:],  # Keep last 20 partners
            }
            
            # Update in storage
            self.db_store.update_provenance(event_id, provenance_data)
            
        except Exception as e:
            logger.debug(f"Could not update provenance for {event_id}: {e}")
