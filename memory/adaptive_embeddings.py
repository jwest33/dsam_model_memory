"""
Adaptive Embedding System

This module implements gravitational embedding updates where memories naturally
drift towards each other based on real-world usage patterns and co-occurrence.
Embeddings evolve to reflect discovered relationships through queries.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingUpdate:
    """Record of an embedding update for tracking evolution"""
    event_id: str
    timestamp: datetime
    attraction_source: str  # What caused the attraction
    strength: float  # How strong the attraction was
    dimension_shifts: np.ndarray  # Which dimensions changed most


class AdaptiveEmbeddingSystem:
    """
    System for evolving embeddings based on real-world usage patterns.
    Memories that are frequently accessed together gravitate towards each other.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize adaptive embedding system.
        
        Args:
            learning_rate: How quickly embeddings adapt (0.01 = slow, 0.1 = fast)
            momentum: Smoothing factor for updates (prevents oscillation)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Track embedding evolution
        self.embedding_store = {}  # event_id -> current embedding
        self.velocity_store = {}  # event_id -> momentum vector
        self.update_history = defaultdict(list)  # event_id -> list of updates
        
        # Co-occurrence tracking
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(float))
        self.access_patterns = defaultdict(list)  # query -> list of accessed events
        
        # Dimension importance (learned over time)
        self.dimension_importance = None  # Will be learned
        
    def update_embeddings_from_cluster(
        self,
        cluster_events: List[Tuple[str, np.ndarray]],
        cluster_centroid: np.ndarray,
        relevance_scores: np.ndarray,
        query_context: Dict[str, Any]
    ):
        """
        Update embeddings based on cluster formation.
        Events in the same cluster attract each other.
        
        Args:
            cluster_events: List of (event_id, embedding) tuples
            cluster_centroid: Center of the cluster
            relevance_scores: How relevant each event is (from eigenvector centrality)
            query_context: What query created this cluster
        """
        if len(cluster_events) < 2:
            return
        
        # Determine which dimensions are most important for this query
        query_dimensions = self._identify_query_dimensions(query_context)
        
        for i, (event_id, embedding) in enumerate(cluster_events):
            # Initialize velocity if needed
            if event_id not in self.velocity_store:
                self.velocity_store[event_id] = np.zeros_like(embedding)
            
            # Compute attraction forces
            forces = np.zeros_like(embedding)
            
            # 1. Attraction to cluster centroid (weighted by relevance)
            centroid_force = (cluster_centroid - embedding) * relevance_scores[i]
            forces += centroid_force * 0.4
            
            # 2. Attraction to other highly relevant events in cluster
            for j, (other_id, other_embedding) in enumerate(cluster_events):
                if i == j:
                    continue
                
                # Stronger attraction between highly relevant events
                mutual_relevance = relevance_scores[i] * relevance_scores[j]
                if mutual_relevance > 0.5:  # Only attract if both are relevant
                    pairwise_force = (other_embedding - embedding) * mutual_relevance
                    forces += pairwise_force * 0.2
                    
                    # Update co-occurrence
                    self.co_occurrence_matrix[event_id][other_id] += mutual_relevance
            
            # 3. Apply dimension-specific weighting
            forces = self._apply_dimension_weighting(forces, query_dimensions)
            
            # 4. Apply momentum and learning rate
            self.velocity_store[event_id] = (
                self.momentum * self.velocity_store[event_id] +
                self.learning_rate * forces
            )
            
            # 5. Update embedding
            new_embedding = embedding + self.velocity_store[event_id]
            
            # 6. Normalize to maintain unit sphere
            norm = np.linalg.norm(new_embedding)
            if norm > 0:
                new_embedding = new_embedding / norm
            
            # 7. Store updated embedding
            self.embedding_store[event_id] = new_embedding
            
            # 8. Record update
            update = EmbeddingUpdate(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                attraction_source=f"cluster_{query_context.get('query', 'unknown')}",
                strength=float(np.linalg.norm(forces)),
                dimension_shifts=np.abs(new_embedding - embedding)
            )
            self.update_history[event_id].append(update)
    
    def gravitational_pull(
        self,
        source_id: str,
        source_embedding: np.ndarray,
        target_id: str,
        target_embedding: np.ndarray,
        interaction_type: str,
        strength: float = 0.1
    ):
        """
        Apply gravitational pull between two embeddings based on interaction.
        
        Args:
            source_id: ID of source event
            source_embedding: Current source embedding
            target_id: ID of target event
            target_embedding: Current target embedding
            interaction_type: Type of interaction (e.g., 'causal', 'temporal', 'semantic')
            strength: How strong the pull should be
        """
        # Different interaction types affect different dimensions
        dimension_mask = self._get_dimension_mask(interaction_type)
        
        # Compute bidirectional attraction
        # They pull each other, but stronger pulls weaker
        source_to_target = target_embedding - source_embedding
        target_to_source = source_embedding - target_embedding
        
        # Apply dimension mask
        source_to_target *= dimension_mask
        target_to_source *= dimension_mask
        
        # Update velocities
        if source_id not in self.velocity_store:
            self.velocity_store[source_id] = np.zeros_like(source_embedding)
        if target_id not in self.velocity_store:
            self.velocity_store[target_id] = np.zeros_like(target_embedding)
        
        # Apply forces with learning rate
        self.velocity_store[source_id] += self.learning_rate * strength * source_to_target
        self.velocity_store[target_id] += self.learning_rate * strength * 0.5 * target_to_source
        
        # Update embeddings
        new_source = source_embedding + self.velocity_store[source_id]
        new_target = target_embedding + self.velocity_store[target_id]
        
        # Normalize
        new_source = new_source / np.linalg.norm(new_source)
        new_target = new_target / np.linalg.norm(new_target)
        
        # Store
        self.embedding_store[source_id] = new_source
        self.embedding_store[target_id] = new_target
        
        # Update co-occurrence
        self.co_occurrence_matrix[source_id][target_id] += strength
        self.co_occurrence_matrix[target_id][source_id] += strength * 0.5
    
    def learn_dimension_importance(
        self,
        successful_retrievals: List[Dict[str, Any]]
    ):
        """
        Learn which embedding dimensions are most important based on successful retrievals.
        
        Args:
            successful_retrievals: List of retrieval results with feedback
        """
        if not successful_retrievals:
            return
        
        dimension_scores = np.zeros(768)  # Assuming 768-dim embeddings
        
        for retrieval in successful_retrievals:
            query_embedding = retrieval.get('query_embedding')
            result_embeddings = retrieval.get('result_embeddings', [])
            relevance_scores = retrieval.get('relevance_scores', [])
            
            if not query_embedding or not result_embeddings:
                continue
            
            for result_emb, relevance in zip(result_embeddings, relevance_scores):
                # Dimensions where query and result align contribute to importance
                alignment = query_embedding * result_emb  # Element-wise
                dimension_scores += alignment * relevance
        
        # Normalize and store
        dimension_scores = dimension_scores / len(successful_retrievals)
        
        # Smooth with previous importance (if exists)
        if self.dimension_importance is not None:
            self.dimension_importance = 0.9 * self.dimension_importance + 0.1 * dimension_scores
        else:
            self.dimension_importance = dimension_scores
    
    def adapt_from_feedback(
        self,
        query_embedding: np.ndarray,
        positive_examples: List[Tuple[str, np.ndarray]],
        negative_examples: List[Tuple[str, np.ndarray]]
    ):
        """
        Adapt embeddings based on user feedback.
        Positive examples move closer, negative examples move apart.
        
        Args:
            query_embedding: The query that was used
            positive_examples: Events marked as relevant
            negative_examples: Events marked as irrelevant
        """
        # Positive examples attract each other and the query
        for event_id, embedding in positive_examples:
            if event_id not in self.velocity_store:
                self.velocity_store[event_id] = np.zeros_like(embedding)
            
            # Attract to query
            force = (query_embedding - embedding) * 0.1
            
            # Attract to other positive examples
            for other_id, other_emb in positive_examples:
                if event_id != other_id:
                    force += (other_emb - embedding) * 0.05
            
            # Apply force
            self.velocity_store[event_id] += self.learning_rate * force
            new_embedding = embedding + self.velocity_store[event_id]
            new_embedding = new_embedding / np.linalg.norm(new_embedding)
            self.embedding_store[event_id] = new_embedding
        
        # Negative examples repel from query and positive examples
        for event_id, embedding in negative_examples:
            if event_id not in self.velocity_store:
                self.velocity_store[event_id] = np.zeros_like(embedding)
            
            # Repel from query
            force = (embedding - query_embedding) * 0.05  # Gentler repulsion
            
            # Repel from positive examples
            for pos_id, pos_emb in positive_examples:
                direction = embedding - pos_emb
                distance = np.linalg.norm(direction)
                if distance > 0:
                    force += (direction / distance) * 0.02
            
            # Apply force
            self.velocity_store[event_id] += self.learning_rate * force
            new_embedding = embedding + self.velocity_store[event_id]
            new_embedding = new_embedding / np.linalg.norm(new_embedding)
            self.embedding_store[event_id] = new_embedding
    
    def _identify_query_dimensions(self, query_context: Dict) -> np.ndarray:
        """
        Identify which dimensions are important for a given query.
        """
        # Start with uniform importance
        importance = np.ones(768) * 0.5
        
        # Boost dimensions based on query fields
        field_weights = query_context.get('field_weights', {})
        
        # Different fields affect different dimension ranges
        # This is a simplified model - in practice, learn this
        dimension_ranges = {
            'who': (0, 64),      # First 64 dims for actor
            'what': (64, 192),   # Next 128 dims for content
            'when': (192, 224),  # 32 dims for temporal
            'where': (224, 256), # 32 dims for location
            'why': (256, 320),   # 64 dims for intent
            'how': (320, 768)    # 64 dims for method
        }
        
        for field, weight in field_weights.items():
            if field in dimension_ranges:
                start, end = dimension_ranges[field]
                importance[start:end] *= (1 + weight)
        
        # Apply learned dimension importance if available
        if self.dimension_importance is not None:
            importance *= self.dimension_importance
        
        # Normalize
        importance = importance / importance.max()
        
        return importance
    
    def _apply_dimension_weighting(
        self,
        forces: np.ndarray,
        dimension_weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply dimension-specific weighting to forces.
        """
        return forces * dimension_weights
    
    def _get_dimension_mask(self, interaction_type: str) -> np.ndarray:
        """
        Get dimension mask for different interaction types.
        """
        mask = np.ones(768)
        
        if interaction_type == 'causal':
            # Causal relationships affect why/how dimensions more
            mask[256:768] *= 2.0
        elif interaction_type == 'temporal':
            # Temporal relationships affect when dimensions
            mask[192:224] *= 2.0
        elif interaction_type == 'semantic':
            # Semantic relationships affect what dimensions
            mask[64:192] *= 2.0
        elif interaction_type == 'actor':
            # Actor relationships affect who dimensions
            mask[0:64] *= 2.0
        
        # Normalize
        mask = mask / mask.max()
        
        return mask
    
    def get_evolved_embedding(self, event_id: str, original: np.ndarray) -> np.ndarray:
        """
        Get the evolved embedding for an event, or return original if not evolved.
        """
        if event_id in self.embedding_store:
            return self.embedding_store[event_id]
        return original
    
    def compute_drift(self, event_id: str, original: np.ndarray) -> float:
        """
        Compute how much an embedding has drifted from its original.
        """
        if event_id not in self.embedding_store:
            return 0.0
        
        evolved = self.embedding_store[event_id]
        
        # Cosine distance
        similarity = np.dot(original, evolved)
        drift = 1.0 - similarity
        
        return float(drift)
    
    def get_evolution_trajectory(self, event_id: str) -> List[Dict]:
        """
        Get the evolution trajectory of an embedding.
        """
        if event_id not in self.update_history:
            return []
        
        trajectory = []
        for update in self.update_history[event_id]:
            trajectory.append({
                'timestamp': update.timestamp.isoformat(),
                'source': update.attraction_source,
                'strength': update.strength,
                'top_dimensions': np.argsort(update.dimension_shifts)[-5:].tolist()
            })
        
        return trajectory
    
    def stabilize_embeddings(self, decay_factor: float = 0.95):
        """
        Stabilize embeddings by reducing velocities (cooling).
        """
        for event_id in self.velocity_store:
            self.velocity_store[event_id] *= decay_factor
    
    def save_state(self, filepath: str):
        """Save the adaptive embedding state"""
        state = {
            'embeddings': {k: v.tolist() for k, v in self.embedding_store.items()},
            'velocities': {k: v.tolist() for k, v in self.velocity_store.items()},
            'co_occurrence': dict(self.co_occurrence_matrix),
            'dimension_importance': self.dimension_importance.tolist() if self.dimension_importance is not None else None,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load adaptive embedding state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.embedding_store = {k: np.array(v) for k, v in state['embeddings'].items()}
        self.velocity_store = {k: np.array(v) for k, v in state['velocities'].items()}
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(float), state['co_occurrence'])
        
        if state['dimension_importance']:
            self.dimension_importance = np.array(state['dimension_importance'])
        
        self.learning_rate = state['learning_rate']
        self.momentum = state['momentum']
    
    def get_co_occurrence_strength(self, event1_id: str, event2_id: str) -> float:
        """Get the co-occurrence strength between two events"""
        return self.co_occurrence_matrix[event1_id][event2_id]
    
    def get_embedding_neighborhood(
        self,
        event_id: str,
        radius: float = 0.2,
        max_neighbors: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get events whose embeddings are within a certain radius.
        """
        if event_id not in self.embedding_store:
            return []
        
        center = self.embedding_store[event_id]
        neighbors = []
        
        for other_id, other_embedding in self.embedding_store.items():
            if other_id == event_id:
                continue
            
            # Cosine distance
            similarity = np.dot(center, other_embedding)
            distance = 1.0 - similarity
            
            if distance <= radius:
                neighbors.append((other_id, distance))
        
        # Sort by distance and limit
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:max_neighbors]
