"""
Dynamic Memory Store with Query-Driven Clustering and Adaptive Embeddings

This module replaces the static block-based storage with dynamic clustering
and adaptive embeddings that evolve based on usage patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path

from models.event import Event, FiveW1H
from memory.chromadb_store import ChromaDBStore
from memory.dynamic_clustering import DynamicMemoryClustering
from memory.adaptive_embeddings import AdaptiveEmbeddingSystem
from memory.hopfield import ModernHopfieldNetwork
from embedding.singleton_embedder import get_five_w1h_embedder
from config import get_config

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Unified memory store with dynamic clustering and adaptive embeddings.
    No static blocks, no arbitrary thresholds.
    """
    
    def __init__(self):
        """Initialize the dynamic memory store"""
        self.config = get_config()
        
        # Core storage (ChromaDB is primary)
        self.chromadb = ChromaDBStore()
        
        # Dynamic systems
        self.clustering = DynamicMemoryClustering()
        self.adaptive_embeddings = AdaptiveEmbeddingSystem(
            learning_rate=0.01,
            momentum=0.9
        )
        
        # Hopfield network for associative memory
        self.hopfield = ModernHopfieldNetwork()
        
        # Embedder for generating embeddings (singleton to avoid repeated model loading)
        self.embedder = get_five_w1h_embedder()
        
        # Episode tracking
        self.episode_map = {}  # episode_id -> List[event_id]
        self.current_clusters = {}  # Cache of recent clusters
        
        # Statistics
        self.total_events = 0
        self.total_queries = 0
        
        logger.info("Dynamic memory store initialized")
    
    def store_event(self, event: Event) -> Tuple[bool, str]:
        """
        Store an event without any salience threshold.
        All events are stored and dynamically clustered on retrieval.
        
        Args:
            event: Event to store
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Generate embedding (use value embedding for storage)
            key_embedding, value_embedding = self.embedder.embed_event(event)
            embedding = value_embedding  # Use value embedding for main storage
            
            # Check for duplicates based on similarity
            similar_events = self.chromadb.retrieve_events_by_query(
                embedding, k=5
            )
            
            if similar_events:
                # Check if this is a duplicate
                for similar_event, similarity in similar_events:
                    if similarity > self.config.memory.similarity_threshold:
                        # Update existing event with EMA instead of creating duplicate
                        self._update_existing_event(similar_event, event, embedding)
                        return True, f"Updated existing event {similar_event.id[:8]} via EMA"
            
            # Store in ChromaDB (primary storage)
            self.chromadb.store_event(event, embedding)
            
            # Add to Hopfield network (use key_embedding for queries, value_embedding for content)
            self.hopfield.store(key_embedding, value_embedding, metadata=event.to_dict())
            
            # Track episode
            if event.episode_id not in self.episode_map:
                self.episode_map[event.episode_id] = []
            self.episode_map[event.episode_id].append(event.id)
            
            # Update statistics
            self.total_events += 1
            
            # Log growth periodically
            if self.total_events % 100 == 0:
                logger.info(f"Memory store size: {self.total_events} events")
            
            return True, f"Stored event {event.id[:8]} (total: {self.total_events})"
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False, str(e)
    
    def retrieve_memories(
        self,
        query: Dict[str, str],
        k: int = 10,
        use_clustering: bool = True,
        update_embeddings: bool = True
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve memories using dynamic clustering and adaptive embeddings.
        
        Args:
            query: 5W1H query fields
            k: Number of memories to retrieve
            use_clustering: Whether to use dynamic clustering
            
        Returns:
            List of (event, relevance_score) tuples
        """
        try:
            self.total_queries += 1
            
            # Generate query embedding
            # Fill in missing fields for partial queries
            full_query = {
                'who': query.get('who', ''),
                'what': query.get('what', ''),
                'when': query.get('when', ''),
                'where': query.get('where', ''),
                'why': query.get('why', ''),
                'how': query.get('how', '')
            }
            query_embedding, _ = self.embedder.embed_five_w1h(FiveW1H(**full_query))
            
            if use_clustering:
                # Retrieve more candidates for clustering
                candidates = self.chromadb.retrieve_events_by_query(
                    query_embedding, k=min(k * 5, 100)
                )
                
                if not candidates:
                    return []
                
                # Extract events
                events = []
                for event, _ in candidates:
                    events.append(event)
                
                # Perform dynamic clustering (it generates embeddings internally)
                clusters = self.clustering.cluster_by_query(
                    events=events,
                    query=query
                )
                
                # Process clusters to extract top memories
                results = []
                for cluster in clusters:
                    # Generate embeddings for events in this cluster
                    cluster_embeddings = []
                    for event in cluster.events:
                        _, embedding = self.embedder.embed_event(event)
                        evolved = self.adaptive_embeddings.get_evolved_embedding(
                            event.id, embedding
                        )
                        cluster_embeddings.append((event.id, evolved))
                    
                    # Create relevance scores based on cluster coherence and relevance
                    relevance_scores = np.ones(len(cluster.events)) * cluster.relevance
                    
                    # Update embeddings based on cluster formation (only if enabled)
                    if update_embeddings:
                        self.adaptive_embeddings.update_embeddings_from_cluster(
                            cluster_events=cluster_embeddings,
                            cluster_centroid=cluster.centroid,
                            relevance_scores=relevance_scores,
                            query_context=query
                        )
                    
                    # Add events from cluster with their relevance
                    for event in cluster.events:
                        if len(results) < k:
                            results.append((event, cluster.relevance))
                
                # Sort by relevance
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:k]
                
            else:
                # Simple retrieval without clustering
                candidates = self.chromadb.retrieve_events_by_query(
                    query_embedding, k=k
                )
                
                results = []
                for event, similarity in candidates:
                    # event is already an Event object
                    results.append((event, similarity))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def get_insights(
        self,
        query: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Get insights about memories using dynamic clustering.
        
        Args:
            query: Natural language query
            k: Number of relevant memories to analyze
            
        Returns:
            Dictionary with insights
        """
        try:
            # Convert to 5W1H query
            query_5w1h = self._parse_natural_query(query)
            
            # Retrieve with clustering
            memories = self.retrieve_memories(query_5w1h, k=k, use_clustering=True)
            
            if not memories:
                return {
                    'query': query,
                    'insights': 'No relevant memories found',
                    'clusters': [],
                    'patterns': []
                }
            
            # Analyze patterns
            patterns = self._analyze_patterns(memories)
            
            # Get cluster information
            cluster_info = []
            if hasattr(self, 'current_clusters'):
                for cluster_id, cluster in self.current_clusters.items():
                    cluster_info.append({
                        'id': cluster_id,
                        'size': len(cluster.get('event_indices', [])),
                        'coherence': cluster.get('coherence_score', 0),
                        'theme': self._extract_theme(cluster)
                    })
            
            return {
                'query': query,
                'total_memories': len(memories),
                'clusters': cluster_info,
                'patterns': patterns,
                'top_memories': [
                    {
                        'what': m[0].five_w1h.what,
                        'when': m[0].five_w1h.when,
                        'relevance': float(m[1])
                    }
                    for m in memories[:5]
                ],
                'embedding_drift': self._calculate_drift_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return {'error': str(e)}
    
    def adapt_from_feedback(
        self,
        query: Dict[str, str],
        positive_events: List[str],
        negative_events: List[str]
    ):
        """
        Adapt embeddings based on user feedback.
        
        Args:
            query: The query that was used
            positive_events: Event IDs marked as relevant
            negative_events: Event IDs marked as irrelevant
        """
        try:
            # Generate query embedding
            # Fill in missing fields for partial queries
            full_query = {
                'who': query.get('who', ''),
                'what': query.get('what', ''),
                'when': query.get('when', ''),
                'where': query.get('where', ''),
                'why': query.get('why', ''),
                'how': query.get('how', '')
            }
            query_embedding, _ = self.embedder.embed_five_w1h(FiveW1H(**full_query))
            
            # Collect positive examples
            positive_examples = []
            for event_id in positive_events:
                event = self.chromadb.get_event(event_id)
                if event:
                    _, embedding = self.embedder.embed_event(event)
                    positive_examples.append((event_id, embedding))
            
            # Collect negative examples
            negative_examples = []
            for event_id in negative_events:
                event = self.chromadb.get_event(event_id)
                if event:
                    _, embedding = self.embedder.embed_event(event)
                    negative_examples.append((event_id, embedding))
            
            # Update embeddings based on feedback
            self.adaptive_embeddings.adapt_from_feedback(
                query_embedding=query_embedding,
                positive_examples=positive_examples,
                negative_examples=negative_examples
            )
            
            logger.info(f"Adapted embeddings from feedback: "
                       f"+{len(positive_examples)}, -{len(negative_examples)}")
            
        except Exception as e:
            logger.error(f"Failed to adapt from feedback: {e}")
    
    def _update_existing_event(
        self,
        existing: Event,
        new_event: Event,
        new_embedding: np.ndarray
    ):
        """Update existing event with EMA instead of creating duplicate"""
        try:
            # Get existing embedding
            existing_embedding = self.chromadb.get_embedding(existing.id)
            
            if existing_embedding is not None:
                # Apply gravitational pull between embeddings
                self.adaptive_embeddings.gravitational_pull(
                    source_id=new_event.id,
                    source_embedding=new_embedding,
                    target_id=existing.id,
                    target_embedding=existing_embedding,
                    interaction_type='semantic',
                    strength=0.2
                )
                
                # Get updated embedding
                updated_embedding = self.adaptive_embeddings.get_evolved_embedding(
                    existing.id, existing_embedding
                )
                
                # Update in ChromaDB
                self.chromadb.update_embedding(existing.id, updated_embedding)
                
                # Update access tracking
                existing.update_access()
                self.chromadb.update_event(existing)
                
        except Exception as e:
            logger.error(f"Failed to update existing event: {e}")
    
    def _parse_natural_query(self, query: str) -> Dict[str, str]:
        """Parse natural language query into 5W1H structure"""
        # Simple keyword-based parsing (can be enhanced with NLP)
        result = {}
        
        query_lower = query.lower()
        
        # Look for who
        if 'who' in query_lower or 'user' in query_lower or 'agent' in query_lower:
            if 'user' in query_lower:
                result['who'] = 'user'
            elif 'agent' in query_lower or 'llm' in query_lower:
                result['who'] = 'agent'
        
        # Look for what (most queries have this)
        result['what'] = query
        
        # Look for when
        if 'when' in query_lower or 'today' in query_lower or 'yesterday' in query_lower:
            if 'today' in query_lower:
                result['when'] = datetime.utcnow().isoformat()
            # Add more temporal parsing as needed
        
        # Look for where
        if 'where' in query_lower or 'location' in query_lower:
            # Extract location context
            pass
        
        # Look for why
        if 'why' in query_lower or 'because' in query_lower or 'reason' in query_lower:
            # Extract causal context
            pass
        
        # Look for how
        if 'how' in query_lower or 'method' in query_lower or 'using' in query_lower:
            # Extract method context
            pass
        
        return result
    
    def _analyze_patterns(self, memories: List[Tuple[Event, float]]) -> List[Dict]:
        """Analyze patterns in retrieved memories"""
        patterns = []
        
        if not memories:
            return patterns
        
        # Temporal patterns
        timestamps = [m[0].created_at for m in memories]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            avg_interval = np.mean(time_diffs) if time_diffs else 0
            patterns.append({
                'type': 'temporal',
                'description': f'Average interval: {avg_interval:.1f} seconds'
            })
        
        # Actor patterns
        actors = {}
        for event, _ in memories:
            actor = event.five_w1h.who
            actors[actor] = actors.get(actor, 0) + 1
        
        if actors:
            dominant_actor = max(actors, key=actors.get)
            patterns.append({
                'type': 'actor',
                'description': f'Dominant actor: {dominant_actor} ({actors[dominant_actor]} events)'
            })
        
        # Context patterns
        contexts = {}
        for event, _ in memories:
            context = event.five_w1h.where
            contexts[context] = contexts.get(context, 0) + 1
        
        if contexts:
            dominant_context = max(contexts, key=contexts.get)
            patterns.append({
                'type': 'context',
                'description': f'Dominant context: {dominant_context}'
            })
        
        return patterns
    
    def _extract_theme(self, cluster: Dict) -> str:
        """Extract theme from a cluster"""
        # Simple theme extraction based on common words
        if 'events' not in cluster:
            return "Unknown theme"
        
        words = []
        for event in cluster['events']:
            words.extend(event.five_w1h.what.lower().split())
        
        # Find most common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            theme_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            return ' '.join([w[0] for w in theme_words])
        
        return "General"
    
    def _calculate_drift_stats(self) -> Dict:
        """Calculate embedding drift statistics"""
        drift_stats = {
            'total_evolved': len(self.adaptive_embeddings.embedding_store),
            'avg_drift': 0.0,
            'max_drift': 0.0
        }
        
        if self.adaptive_embeddings.embedding_store:
            drifts = []
            for event_id in self.adaptive_embeddings.embedding_store:
                # Get original embedding
                event = self.chromadb.get_event(event_id)
                if event:
                    _, original = self.embedder.embed_event(event)
                    drift = self.adaptive_embeddings.compute_drift(event_id, original)
                    drifts.append(drift)
            
            if drifts:
                drift_stats['avg_drift'] = float(np.mean(drifts))
                drift_stats['max_drift'] = float(np.max(drifts))
        
        return drift_stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        return {
            'total_events': self.total_events,
            'total_queries': self.total_queries,
            'episodes': len(self.episode_map),
            'evolved_embeddings': len(self.adaptive_embeddings.embedding_store),
            'co_occurrence_pairs': len(self.adaptive_embeddings.co_occurrence_matrix),
            'hopfield_size': self.hopfield.memory_count,
            'chromadb_stats': self.chromadb.get_stats(),
            'drift_stats': self._calculate_drift_stats()
        }
    
    def save(self):
        """Alias for save_state for compatibility"""
        self.save_state()
    
    def save_state(self):
        """Save the current state"""
        try:
            # Save adaptive embeddings
            state_dir = self.config.storage.state_dir
            self.adaptive_embeddings.save_state(
                str(state_dir / "adaptive_embeddings.json")
            )
            
            # Save episode map
            import json
            with open(state_dir / "episode_map.json", 'w') as f:
                json.dump(self.episode_map, f)
            
            # ChromaDB persists automatically
            
            logger.info("Saved dynamic memory store state")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load saved state"""
        try:
            state_dir = self.config.storage.state_dir
            
            # Load adaptive embeddings
            embeddings_file = state_dir / "adaptive_embeddings.json"
            if embeddings_file.exists():
                self.adaptive_embeddings.load_state(str(embeddings_file))
            
            # Load episode map
            episode_file = state_dir / "episode_map.json"
            if episode_file.exists():
                import json
                with open(episode_file, 'r') as f:
                    self.episode_map = json.load(f)
            
            logger.info("Loaded dynamic memory store state")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
