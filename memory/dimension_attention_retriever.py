"""
Dimension-aware attention-based retrieval system for multi-dimensional memory merges.

This module provides intelligent retrieval by computing attention weights for different
merge dimensions (Actor, Temporal, Conceptual, Spatial) based on query characteristics.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re

from models.merge_types import MergeType
from models.event import Event
from models.merged_event import MergedEvent

logger = logging.getLogger(__name__)


class DimensionAttentionRetriever:
    """
    Retrieves memories using dimension-aware attention mechanism.
    
    Computes attention weights for each merge dimension based on query
    characteristics and retrieves the most relevant memories.
    """
    
    def __init__(self, encoder, multi_merger, similarity_cache=None, chromadb_store=None):
        """
        Initialize the dimension attention retriever.
        
        Args:
            encoder: Dual-space encoder for embeddings
            multi_merger: Multi-dimensional merger instance
            similarity_cache: Optional similarity cache for fast lookups
            chromadb_store: ChromaDB store for retrieving merge groups
        """
        self.encoder = encoder
        self.multi_merger = multi_merger
        self.similarity_cache = similarity_cache
        self.chromadb = chromadb_store
        
        # Keywords for dimension detection
        self.actor_keywords = {
            'who', 'person', 'user', 'assistant', 'they', 'them', 'someone',
            'everyone', 'anybody', 'somebody', 'alice', 'bob', 'agent'
        }
        
        self.temporal_keywords = {
            'when', 'time', 'yesterday', 'today', 'tomorrow', 'last', 'next',
            'before', 'after', 'during', 'recently', 'earlier', 'later',
            'conversation', 'thread', 'session', 'sequence', 'history'
        }
        
        self.conceptual_keywords = {
            'why', 'how', 'concept', 'idea', 'theory', 'principle', 'method',
            'approach', 'strategy', 'goal', 'purpose', 'reason', 'understand',
            'implement', 'design', 'architecture', 'pattern', 'abstract'
        }
        
        self.spatial_keywords = {
            'where', 'location', 'place', 'here', 'there', 'file', 'directory',
            'module', 'component', 'backend', 'frontend', 'database', 'server',
            'local', 'remote', 'system', 'environment'
        }
    
    def compute_dimension_attention(self, query_fields: Dict[str, str], 
                                   query_embedding: Tuple[np.ndarray, np.ndarray]) -> Dict[MergeType, float]:
        """
        Compute attention weights for each dimension based on query.
        
        Args:
            query_fields: Query fields (who, what, where, etc.)
            query_embedding: Dual-space embeddings (euclidean, hyperbolic)
        
        Returns:
            Dictionary mapping MergeType to attention weight
        """
        weights = {}
        
        # Extract query text for analysis
        query_text = ' '.join([str(v) for v in query_fields.values() if v]).lower()
        
        # Compute individual dimension scores
        weights[MergeType.ACTOR] = self._compute_actor_attention(query_fields, query_text)
        weights[MergeType.TEMPORAL] = self._compute_temporal_attention(query_fields, query_text)
        weights[MergeType.CONCEPTUAL] = self._compute_conceptual_attention(
            query_fields, query_text, query_embedding
        )
        weights[MergeType.SPATIAL] = self._compute_spatial_attention(query_fields, query_text)
        
        # Normalize weights to sum to 1
        return self._normalize_weights(weights)
    
    def _compute_actor_attention(self, query_fields: Dict[str, str], query_text: str) -> float:
        """Compute attention weight for actor dimension."""
        score = 0.0
        
        # Check if 'who' field is specified
        if query_fields.get('who'):
            score += 0.5
        
        # Check for actor-related keywords
        actor_matches = sum(1 for keyword in self.actor_keywords if keyword in query_text)
        score += min(actor_matches * 0.2, 0.5)
        
        # Check for proper nouns (capitalized words)
        words = query_text.split()
        proper_nouns = sum(1 for word in words if word and word[0].isupper())
        score += min(proper_nouns * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _compute_temporal_attention(self, query_fields: Dict[str, str], query_text: str) -> float:
        """Compute attention weight for temporal dimension."""
        score = 0.0
        
        # Check if 'when' field is specified
        if query_fields.get('when'):
            score += 0.5
        
        # Check for temporal keywords
        temporal_matches = sum(1 for keyword in self.temporal_keywords if keyword in query_text)
        score += min(temporal_matches * 0.25, 0.5)
        
        # Check for date/time patterns
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|ago|minutes?|hours?|days?|weeks?|months?'
        if re.search(date_pattern, query_text):
            score += 0.3
        
        # Check for conversation/thread context
        if any(word in query_text for word in ['conversation', 'thread', 'discussion', 'chat']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _compute_conceptual_attention(self, query_fields: Dict[str, str], query_text: str,
                                     query_embedding: Tuple[np.ndarray, np.ndarray]) -> float:
        """Compute attention weight for conceptual dimension."""
        score = 0.0
        
        # Check if 'why' or 'how' fields are specified
        if query_fields.get('why'):
            score += 0.4
        if query_fields.get('how'):
            score += 0.3
        
        # Check for conceptual keywords
        conceptual_matches = sum(1 for keyword in self.conceptual_keywords if keyword in query_text)
        score += min(conceptual_matches * 0.2, 0.5)
        
        # Use hyperbolic weight as indicator of abstract/conceptual query
        # Higher hyperbolic weight suggests more abstract/hierarchical thinking
        if hasattr(self.encoder, 'compute_query_weights'):
            lambda_e, lambda_h = self.encoder.compute_query_weights(query_fields)
            score += float(lambda_h) * 0.3  # Add up to 0.3 based on hyperbolic weight
        
        # Check for question patterns about concepts
        if any(pattern in query_text for pattern in ['how does', 'why does', 'what is', 'explain']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _compute_spatial_attention(self, query_fields: Dict[str, str], query_text: str) -> float:
        """Compute attention weight for spatial dimension."""
        score = 0.0
        
        # Check if 'where' field is specified
        if query_fields.get('where'):
            score += 0.5
        
        # Check for spatial keywords
        spatial_matches = sum(1 for keyword in self.spatial_keywords if keyword in query_text)
        score += min(spatial_matches * 0.2, 0.5)
        
        # Check for file paths or module references
        if any(char in query_text for char in ['/', '\\', '.py', '.js', '.ts']):
            score += 0.3
        
        # Check for system component references
        if any(word in query_text for word in ['backend', 'frontend', 'database', 'api', 'server']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _normalize_weights(self, weights: Dict[MergeType, float]) -> Dict[MergeType, float]:
        """Normalize weights to sum to 1, with minimum threshold."""
        total = sum(weights.values())
        
        if total == 0:
            # Equal weights if no dimension is preferred
            return {k: 0.25 for k in weights.keys()}
        
        # Normalize
        normalized = {k: v / total for k, v in weights.items()}
        
        # Apply minimum threshold (0.1) to avoid completely ignoring dimensions
        min_threshold = 0.1
        for k in normalized:
            if normalized[k] < min_threshold:
                normalized[k] = min_threshold
        
        # Re-normalize after applying threshold
        total = sum(normalized.values())
        return {k: v / total for k, v in normalized.items()}
    
    def retrieve_with_dimension_attention(self, query_fields: Dict[str, str], 
                                         k: int = 5) -> Tuple[List[Tuple], Dict[MergeType, float]]:
        """
        Retrieve memories using dimension-aware attention.
        
        Args:
            query_fields: Query fields (who, what, where, etc.)
            k: Number of results to retrieve
        
        Returns:
            Tuple of (retrieved memories with scores, dimension weights)
        """
        # Encode query
        query_text = ' '.join([str(v) for v in query_fields.values() if v])
        euclidean_emb, hyperbolic_emb = self.encoder.encode_dual(query_text)
        query_embedding = (euclidean_emb, hyperbolic_emb)
        
        # Compute dimension attention weights
        dim_weights = self.compute_dimension_attention(query_fields, query_embedding)
        
        logger.info(f"Dimension weights for query: {dim_weights}")
        
        # Search each relevant dimension
        dimension_results = {}
        for dim_type, weight in dim_weights.items():
            if weight > 0.15:  # Only search dimensions with meaningful weight
                results = self._search_dimension(dim_type, query_embedding, query_fields, k * 2)
                if results:
                    dimension_results[dim_type] = (results, weight)
                    logger.debug(f"Found {len(results)} results in {dim_type.value} dimension")
        
        # Combine results with attention weighting
        combined_results = self._combine_dimension_results(dimension_results, k)
        
        return combined_results, dim_weights
    
    def _search_dimension(self, dimension: MergeType, query_embedding: Tuple[np.ndarray, np.ndarray],
                         query_fields: Dict[str, str], k: int) -> List[Tuple[Any, float]]:
        """
        Search within a specific dimension.
        
        Args:
            dimension: The merge dimension to search
            query_embedding: Dual-space query embeddings
            query_fields: Query fields
            k: Number of results
        
        Returns:
            List of (memory, score) tuples
        """
        results = []
        
        if not self.chromadb:
            return results
        
        # Get the appropriate collection for this dimension
        collection_map = {
            MergeType.ACTOR: self.chromadb.actor_merges_collection,
            MergeType.TEMPORAL: self.chromadb.temporal_merges_collection,
            MergeType.CONCEPTUAL: self.chromadb.conceptual_merges_collection,
            MergeType.SPATIAL: self.chromadb.spatial_merges_collection
        }
        
        collection = collection_map.get(dimension)
        if not collection:
            return results
        
        try:
            # Compute space weights for this query
            lambda_e, lambda_h = self.encoder.compute_query_weights(query_fields)
            
            # Combine embeddings with space weights
            euclidean_emb, hyperbolic_emb = query_embedding
            
            # Weight and concatenate embeddings for search
            weighted_embedding = np.concatenate([
                euclidean_emb * float(lambda_e),
                hyperbolic_emb * float(lambda_h)
            ])
            
            # Query the collection
            query_results = collection.query(
                query_embeddings=[weighted_embedding.tolist()],
                n_results=k,
                include=['metadatas', 'distances', 'embeddings']
            )
            
            if query_results and query_results['ids']:
                for i, merge_id in enumerate(query_results['ids'][0]):
                    metadata = query_results['metadatas'][0][i] if query_results['metadatas'] else {}
                    distance = query_results['distances'][0][i] if query_results['distances'] else 1.0
                    
                    # Convert distance to similarity score (inverse)
                    score = 1.0 / (1.0 + distance)
                    
                    # Create a lightweight result object with metadata
                    result_obj = {
                        'id': merge_id,
                        'dimension': dimension.value,
                        'metadata': metadata,
                        'merge_key': metadata.get('merge_key', ''),
                        'event_count': int(metadata.get('event_count', 1)),
                        'raw_event_ids': metadata.get('raw_event_ids', '').split(',') if metadata.get('raw_event_ids') else []
                    }
                    
                    results.append((result_obj, score))
            
        except Exception as e:
            logger.error(f"Error searching {dimension.value} dimension: {e}")
        
        return results
    
    def _combine_dimension_results(self, dimension_results: Dict[MergeType, Tuple[List, float]], 
                                  k: int) -> List[Tuple]:
        """
        Combine results from multiple dimensions with attention weighting.
        
        Args:
            dimension_results: Dict mapping dimension to (results, weight)
            k: Number of final results
        
        Returns:
            Combined and ranked results
        """
        combined_scores = {}
        result_objects = {}
        
        # Aggregate scores across dimensions
        for dim_type, (results, dim_weight) in dimension_results.items():
            for result_obj, score in results:
                result_id = result_obj['id']
                
                # Weight the score by dimension attention
                weighted_score = score * dim_weight
                
                if result_id not in combined_scores:
                    combined_scores[result_id] = 0
                    result_objects[result_id] = result_obj
                
                # Take maximum score if same result appears in multiple dimensions
                combined_scores[result_id] = max(combined_scores[result_id], weighted_score)
                
                # Add dimension info to result
                if 'dimensions' not in result_objects[result_id]:
                    result_objects[result_id]['dimensions'] = []
                result_objects[result_id]['dimensions'].append({
                    'type': dim_type.value,
                    'weight': dim_weight,
                    'score': score
                })
        
        # Sort by combined score
        sorted_results = sorted(
            [(result_objects[rid], score) for rid, score in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def get_context_from_dimension_results(self, results: List[Tuple], 
                                          chromadb_store) -> List[Tuple[Any, float, str]]:
        """
        Convert dimension search results to full memory objects with context.
        
        Args:
            results: List of (result_obj, score) from dimension search
            chromadb_store: ChromaDB store to retrieve full events
        
        Returns:
            List of (event, score, context_string) tuples
        """
        contextualized_results = []
        
        for result_obj, score in results:
            try:
                # Get the raw events for this merge group
                raw_event_ids = result_obj.get('raw_event_ids', [])
                
                if raw_event_ids and chromadb_store:
                    # Get the first/most representative raw event for context
                    raw_events = chromadb_store.get_raw_events(raw_event_ids[:3])  # Get up to 3 for context
                    
                    if raw_events:
                        # Build context from the merge group
                        context_lines = []
                        context_lines.append(f"[{result_obj['dimension'].title()} Dimension - {result_obj.get('merge_key', 'Group')}]")
                        context_lines.append(f"This group contains {result_obj['event_count']} related events.")
                        
                        # Add sample events from the group
                        for event in raw_events[:2]:  # Show first 2 events as examples
                            context_lines.append(f"• {event.five_w1h.who}: {event.five_w1h.what[:100]}")
                        
                        context_str = '\n'.join(context_lines)
                        
                        # Use first event as representative
                        contextualized_results.append((raw_events[0], score, context_str))
                
            except Exception as e:
                logger.error(f"Error getting context for result {result_obj.get('id')}: {e}")
        
        return contextualized_results