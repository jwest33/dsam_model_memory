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
from memory.temporal_query import integrate_enhanced_temporal

logger = logging.getLogger(__name__)


class DimensionAttentionRetriever:
    """
    Retrieves memories using dimension-aware attention mechanism.
    
    Computes attention weights for each merge dimension based on query
    characteristics and retrieves the most relevant memories.
    """
    
    def __init__(self, encoder, multi_merger, similarity_cache=None, storage_backend=None, temporal_manager=None, llm_client=None):
        """
        Initialize the dimension attention retriever.
        
        Args:
            encoder: Dual-space encoder for embeddings
            multi_merger: Multi-dimensional merger instance
            similarity_cache: Optional similarity cache for fast lookups
            storage_backend: storage backend for retrieving merge groups
            temporal_manager: Temporal manager for handling temporal queries
            llm_client: Optional LLM client for enhanced temporal detection
        """
        self.encoder = encoder
        self.multi_merger = multi_merger
        self.similarity_cache = similarity_cache
        self.storage = storage_backend
        self.temporal_manager = temporal_manager
        self.llm_client = llm_client
        
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
        # Use the temporal manager if available
        if hasattr(self, 'temporal_manager'):
            logger.debug(f"Using temporal_manager for attention, has LLM: {self.temporal_manager.llm_client is not None}")
            return self.temporal_manager.compute_temporal_attention_weight(query_fields)
        
        # Fallback to original implementation
        score = 0.0
        
        # Check if 'when' field is specified
        if query_fields.get('when'):
            score += 0.5
        
        # Strong temporal indicators - highest priority
        strong_temporal_indicators = [
            'last thing', 'just discussed', 'just talked', 'most recent',
            'latest', 'what did we just', 'last time', 'previous message',
            'before this', 'earlier today', 'a moment ago'
        ]
        for indicator in strong_temporal_indicators:
            if indicator in query_text:
                score += 0.7  # Strong boost for explicit temporal phrases
                break
        
        # Check for temporal keywords with enhanced weighting
        temporal_matches = sum(1 for keyword in self.temporal_keywords if keyword in query_text)
        score += min(temporal_matches * 0.35, 0.6)  # Increased from 0.25 to 0.35
        
        # Check for date/time patterns
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|ago|minutes?|hours?|days?|weeks?|months?'
        if re.search(date_pattern, query_text):
            score += 0.4  # Increased from 0.3
        
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
        combined_results = self._combine_dimension_results(dimension_results, k, query_fields)
        
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
        
        if not self.storage:
            return results
        
        # For conceptual dimension, use enhanced search with dual embeddings and text
        if dimension == MergeType.CONCEPTUAL:
            return self._search_conceptual_dimension(query_embedding, query_fields, k)
        
        # Special handling for temporal dimension with strong temporal intent
        if dimension == MergeType.TEMPORAL and self.temporal_manager:
            # Detect temporal intent strength
            from memory.temporal_manager import TemporalStrength
            strength, confidence, params = self.temporal_manager.detect_temporal_intent(query_fields)
            
            logger.debug(f"Temporal intent detection for query: {query_fields}")
            logger.debug(f"Temporal strength: {strength}, confidence: {confidence}")
            
            # For strong temporal queries (e.g., "last thing we discussed"), use recency-based retrieval
            if strength == TemporalStrength.STRONG:
                logger.info(f"Using recency-based retrieval for strong temporal query (confidence: {confidence})")
                recent_results = self._get_recent_temporal_groups(k, params)
                logger.info(f"Recency-based retrieval returned {len(recent_results)} results")
                return recent_results
        
        # Get the appropriate collection for this dimension
        collection_map = {
            MergeType.ACTOR: self.storage.actor_merges_collection,
            MergeType.TEMPORAL: self.storage.temporal_merges_collection,
            MergeType.CONCEPTUAL: self.storage.conceptual_merges_collection,
            MergeType.SPATIAL: self.storage.spatial_merges_collection
        }
        
        collection = collection_map.get(dimension)
        if not collection:
            return results
        
        try:
            # Compute space weights for this query
            lambda_e, lambda_h = self.encoder.compute_query_weights(query_fields)
            
            # Combine embeddings with space weights
            euclidean_emb, hyperbolic_emb = query_embedding
            
            # Use only Euclidean embedding for storage search
            # (storage collections are created with 768-dim Euclidean embeddings only)
            weighted_embedding = euclidean_emb
            
            # Query the collection (handle empty collections gracefully)
            try:
                query_results = collection.query(
                    query_embeddings=[weighted_embedding.tolist()],
                    n_results=k,
                    include=['metadatas', 'distances', 'embeddings']
                )
            except Exception as query_error:
                # Handle empty collections or other query errors
                if "Nothing found on disk" in str(query_error):
                    # Collection exists but is empty - not an error, just no results
                    logger.debug(f"{dimension.value} collection is empty")
                    return results
                else:
                    # Re-raise other errors
                    raise query_error
            
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
                        'event_count': int(metadata.get('merge_count', metadata.get('event_count', 1))),
                        'raw_event_ids': metadata.get('raw_event_ids', '').split(',') if metadata.get('raw_event_ids') else [],
                        # Include the latest state fields
                        'latest_who': metadata.get('latest_who', ''),
                        'latest_what': metadata.get('latest_what', ''),
                        'latest_when': metadata.get('latest_when', ''),
                        'latest_where': metadata.get('latest_where', ''),
                        'latest_why': metadata.get('latest_why', ''),
                        'latest_how': metadata.get('latest_how', '')
                    }
                    
                    results.append((result_obj, score))
            
        except Exception as e:
            logger.error(f"Error searching {dimension.value} dimension: {e}")
        
        return results
    
    def _search_conceptual_dimension(self, query_embedding: Tuple[np.ndarray, np.ndarray],
                                    query_fields: Dict[str, str], k: int) -> List[Tuple[Any, float]]:
        """
        Enhanced search for conceptual dimension using parallel dual-space search.
        
        Searches both Euclidean and Hyperbolic spaces independently, then merges results
        based on actual similarity scores rather than predicted space weights.
        
        Args:
            query_embedding: Dual-space query embeddings (euclidean, hyperbolic)
            query_fields: Query fields
            k: Number of results
        
        Returns:
            List of (memory, score) tuples
        """
        results = []
        euclidean_collection = self.storage.conceptual_merges_collection
        hyperbolic_collection = self.storage.conceptual_merges_hyperbolic if hasattr(self.storage, 'conceptual_merges_hyperbolic') else None
        
        if not euclidean_collection:
            return results
        
        try:
            euclidean_emb, hyperbolic_emb = query_embedding
            query_text = ' '.join([str(v) for v in query_fields.values() if v]).lower()
            
            # Step 1: Parallel search in BOTH spaces
            initial_k = min(k * 2, 30)  # Get candidates from each space
            
            # Search Euclidean space
            euclidean_results = {}
            try:
                euclidean_query = euclidean_collection.query(
                    query_embeddings=[euclidean_emb.tolist()],
                    n_results=initial_k,
                    include=['metadatas', 'distances']
                )
                if euclidean_query and euclidean_query['ids']:
                    for i, merge_id in enumerate(euclidean_query['ids'][0]):
                        euclidean_results[merge_id] = {
                            'distance': euclidean_query['distances'][0][i],
                            'metadata': euclidean_query['metadatas'][0][i],
                            'similarity': 1.0 / (1.0 + euclidean_query['distances'][0][i])
                        }
            except Exception as e:
                if "Nothing found" not in str(e):
                    logger.debug(f"Euclidean search error: {e}")
            
            # Search Hyperbolic space (if available)
            hyperbolic_results = {}
            if hyperbolic_collection:
                try:
                    hyperbolic_query = hyperbolic_collection.query(
                        query_embeddings=[hyperbolic_emb.tolist()],
                        n_results=initial_k,
                        include=['metadatas', 'distances']
                    )
                    if hyperbolic_query and hyperbolic_query['ids']:
                        for i, merge_id in enumerate(hyperbolic_query['ids'][0]):
                            # Hyperbolic distance is already in hyperbolic metric
                            # Convert to similarity (smaller distance = higher similarity)
                            hyperbolic_results[merge_id] = {
                                'distance': hyperbolic_query['distances'][0][i],
                                'metadata': hyperbolic_query['metadatas'][0][i],
                                'similarity': np.exp(-hyperbolic_query['distances'][0][i])  # Exponential decay
                            }
                except Exception as e:
                    if "Nothing found" not in str(e):
                        logger.debug(f"Hyperbolic search error: {e}")
            
            # Step 2: Merge results from both spaces
            # Create unified scoring that lets the data determine which space is more relevant
            all_candidates = {}
            
            # Collect all unique candidates from both spaces
            all_merge_ids = set(euclidean_results.keys()) | set(hyperbolic_results.keys())
            
            for merge_id in all_merge_ids:
                # Get similarities from each space (0 if not in that space's results)
                euclidean_sim = euclidean_results[merge_id]['similarity'] if merge_id in euclidean_results else 0.0
                hyperbolic_sim = hyperbolic_results[merge_id]['similarity'] if merge_id in hyperbolic_results else 0.0
                
                # Get metadata (prefer from euclidean if available in both)
                if merge_id in euclidean_results:
                    metadata = euclidean_results[merge_id]['metadata']
                elif merge_id in hyperbolic_results:
                    metadata = hyperbolic_results[merge_id]['metadata']
                else:
                    continue
                
                # Text similarity score based on group fields
                text_score = 0.0
                group_why = metadata.get('group_why', '').lower()
                group_how = metadata.get('group_how', '').lower()
                
                if group_why or group_how:
                    query_words = set(query_text.split())
                    why_words = set(group_why.split()) if group_why else set()
                    how_words = set(group_how.split()) if group_how else set()
                    
                    # Calculate Jaccard similarity
                    why_similarity = len(query_words & why_words) / max(len(query_words | why_words), 1)
                    how_similarity = len(query_words & how_words) / max(len(query_words | how_words), 1)
                    
                    # Check for conceptual keywords in group fields
                    conceptual_boost = 0.0
                    for keyword in self.conceptual_keywords:
                        if keyword in group_why:
                            conceptual_boost += 0.1
                        if keyword in group_how:
                            conceptual_boost += 0.1
                    
                    text_score = (why_similarity * 0.5 + how_similarity * 0.3 + min(conceptual_boost, 0.3))
                
                # NEW APPROACH: Let the actual similarities determine the weighting
                # Instead of predicting λ_E and λ_H, we use the relative strength of similarities
                
                # Normalize similarities to determine which space is more confident
                total_embedding_sim = euclidean_sim + hyperbolic_sim
                if total_embedding_sim > 0:
                    # Dynamic weights based on which space has stronger signal
                    euclidean_weight = euclidean_sim / total_embedding_sim
                    hyperbolic_weight = hyperbolic_sim / total_embedding_sim
                else:
                    euclidean_weight = 0.5
                    hyperbolic_weight = 0.5
                
                # Combined score with dynamic weights
                combined_score = (
                    euclidean_sim * euclidean_weight * 0.4 +  # Euclidean contribution
                    hyperbolic_sim * hyperbolic_weight * 0.4 +  # Hyperbolic contribution  
                    text_score * 0.2  # Text matching contribution
                )
                
                # Store candidate info
                all_candidates[merge_id] = {
                    'metadata': metadata,
                    'euclidean_sim': euclidean_sim,
                    'hyperbolic_sim': hyperbolic_sim,
                    'text_score': text_score,
                    'combined_score': combined_score,
                    'dominant_space': 'hyperbolic' if hyperbolic_weight > euclidean_weight else 'euclidean'
                }
                
            
            # Step 3: Sort candidates by combined score and take top k
            sorted_candidates = sorted(
                all_candidates.items(), 
                key=lambda x: x[1]['combined_score'], 
                reverse=True
            )[:k]
            
            # Step 4: Retrieve actual merge events for top candidates
            for merge_id, candidate_info in sorted_candidates:
                metadata = candidate_info['metadata']
                score = candidate_info['combined_score']
                
                # Retrieve the actual merge event
                merge_event = self._get_merge_event_from_metadata(merge_id, metadata)
                if merge_event:
                    results.append((merge_event, score))
                    
                    # Log which space was dominant for this result
                    logger.debug(
                        f"Conceptual result: {merge_id} "
                        f"(score: {score:.3f}, "
                        f"dominant: {candidate_info['dominant_space']}, "
                        f"E: {candidate_info['euclidean_sim']:.3f}, "
                        f"H: {candidate_info['hyperbolic_sim']:.3f}, "
                        f"T: {candidate_info['text_score']:.3f}, "
                        f"why: '{metadata.get('group_why', '')[:30]}...')"
                    )
            
        except Exception as e:
            logger.error(f"Error in conceptual dimension search: {e}")
        
        return results
    
    def _get_merge_event_from_metadata(self, merge_id: str, metadata: Dict) -> Optional[Dict]:
        """
        Create a lightweight merge event object from metadata.
        
        Args:
            merge_id: The merge event ID
            metadata: The metadata dictionary from storage
            
        Returns:
            A dictionary representing the merge event, or None if creation fails
        """
        try:
            # Create a lightweight result object similar to _search_dimension
            merge_event = {
                'id': merge_id,
                'dimension': 'conceptual',
                'metadata': metadata,
                'merge_key': metadata.get('merge_key', ''),
                'event_count': int(metadata.get('merge_count', metadata.get('event_count', 1))),
                'raw_event_ids': metadata.get('raw_event_ids', '').split(',') if metadata.get('raw_event_ids') else [],
                # Include the latest state fields
                'latest_who': metadata.get('latest_who', ''),
                'latest_what': metadata.get('latest_what', ''),
                'latest_when': metadata.get('latest_when', ''),
                'latest_where': metadata.get('latest_where', ''),
                'latest_why': metadata.get('latest_why', ''),
                'latest_how': metadata.get('latest_how', ''),
                # Include group-level fields if available
                'group_why': metadata.get('group_why', ''),
                'group_how': metadata.get('group_how', '')
            }
            return merge_event
        except Exception as e:
            logger.debug(f"Error creating merge event from metadata: {e}")
            return None
    
    def _compute_hyperbolic_similarity(self, query_emb: np.ndarray, stored_emb: np.ndarray) -> float:
        """
        Compute similarity in hyperbolic space using Poincaré ball model.
        
        Args:
            query_emb: Query hyperbolic embedding
            stored_emb: Stored hyperbolic embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Ensure embeddings are numpy arrays
            query_emb = np.array(query_emb)
            stored_emb = np.array(stored_emb)
            
            # Compute hyperbolic distance in Poincaré ball
            # d_H(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
            
            # Compute norms
            query_norm_sq = np.sum(query_emb ** 2)
            stored_norm_sq = np.sum(stored_emb ** 2)
            
            # Ensure we're within the Poincaré ball (norm < 1)
            # Clip to avoid numerical issues
            query_norm_sq = min(query_norm_sq, 0.99)
            stored_norm_sq = min(stored_norm_sq, 0.99)
            
            # Compute difference
            diff = query_emb - stored_emb
            diff_norm_sq = np.sum(diff ** 2)
            
            # Compute hyperbolic distance
            denominator = (1 - query_norm_sq) * (1 - stored_norm_sq)
            if denominator > 0:
                argument = 1 + 2 * diff_norm_sq / denominator
                # Ensure argument is valid for arcosh (>= 1)
                argument = max(argument, 1.0)
                distance = np.arccosh(argument)
            else:
                # If denominator is 0 or negative, points are at boundary
                distance = float('inf')
            
            # Convert distance to similarity (inverse relationship)
            # Use exponential decay for smooth similarity scores
            similarity = np.exp(-distance)
            
            return float(similarity)
            
        except Exception as e:
            logger.debug(f"Error computing hyperbolic similarity: {e}")
            return 0.0
    
    def _get_recent_temporal_groups(self, k: int, params: Dict) -> List[Tuple[Any, float]]:
        """
        Get the most recent temporal groups based on timestamps.
        
        Args:
            k: Number of results to retrieve
            params: Parameters from temporal intent detection
        
        Returns:
            List of (memory, score) tuples ordered by recency
        """
        logger.info(f"_get_recent_temporal_groups called with k={k}, params={params}")
        results = []
        
        if not self.storage or not self.storage.temporal_merges_collection:
            logger.warning("storage or temporal_merges_collection not available")
            return results
        
        try:
            # Get all temporal groups
            all_groups = self.storage.temporal_merges_collection.get(
                include=['metadatas']
            )
            
            if not all_groups or not all_groups['ids']:
                return results
            
            # Sort by recency using last_updated (which contains latest event's 'when' value)
            groups_with_time = []
            for i, group_id in enumerate(all_groups['ids']):
                metadata = all_groups['metadatas'][i] if all_groups['metadatas'] else {}
                last_updated = metadata.get('last_updated', metadata.get('latest_when', ''))
                
                if last_updated:
                    try:
                        # Parse the timestamp
                        from dateutil import parser as date_parser
                        timestamp = date_parser.parse(last_updated)
                        groups_with_time.append((group_id, metadata, timestamp))
                    except:
                        # If parsing fails, treat as very old
                        groups_with_time.append((group_id, metadata, datetime.min))
                else:
                    # No timestamp, treat as very old
                    groups_with_time.append((group_id, metadata, datetime.min))
            
            # Sort by timestamp (most recent first)
            groups_with_time.sort(key=lambda x: x[2], reverse=True)
            
            # Take top k results and assign scores based on recency
            for i, (group_id, metadata, timestamp) in enumerate(groups_with_time[:k]):
                # Score based on recency rank (most recent = 1.0, declining)
                score = 1.0 - (i * 0.1)  # Gradual decline in score
                score = max(score, 0.1)  # Minimum score of 0.1
                
                # Apply decay based on params if provided
                if params.get('decay_rate'):
                    decay = params['decay_rate'] ** i
                    score *= decay
                
                # Create result object
                result_obj = {
                    'id': group_id,
                    'dimension': 'temporal',
                    'metadata': metadata,
                    'merge_key': metadata.get('merge_key', ''),
                    'event_count': int(metadata.get('merge_count', metadata.get('event_count', 1))),
                    'raw_event_ids': metadata.get('raw_event_ids', '').split(',') if metadata.get('raw_event_ids') else [],
                    # Include the latest state fields
                    'latest_who': metadata.get('latest_who', ''),
                    'latest_what': metadata.get('latest_what', ''),
                    'latest_when': metadata.get('latest_when', ''),
                    'latest_where': metadata.get('latest_where', ''),
                    'latest_why': metadata.get('latest_why', ''),
                    'latest_how': metadata.get('latest_how', ''),
                    'recency_based': True  # Flag to indicate this was retrieved by recency
                }
                
                results.append((result_obj, score))
                
                logger.debug(f"Recent temporal group {i+1}: {group_id} (score: {score:.2f}, last_updated: {metadata.get('last_updated', 'unknown')})")
            
        except Exception as e:
            logger.error(f"Error retrieving recent temporal groups: {e}")
        
        return results
    
    def _combine_dimension_results(self, dimension_results: Dict[MergeType, Tuple[List, float]], 
                                  k: int, query_fields: Optional[Dict] = None) -> List[Tuple]:
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
        
        # Check if temporal dimension has high weight (for recency-based queries)
        temporal_weight = dimension_results.get(MergeType.TEMPORAL, ([], 0))[1] if MergeType.TEMPORAL in dimension_results else 0
        
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
        
        # Apply enhanced temporal weighting if temporal dimension is significant
        if temporal_weight > 0.3 and query_fields:  # Lower threshold for better temporal detection
            # Convert results to format expected by temporal integration
            temporal_candidates = []
            for result_id, result_obj in result_objects.items():
                # Create a mock event object with the necessary attributes
                mock_event = type('obj', (object,), {
                    'five_w1h': type('obj', (object,), {
                        'when': result_obj.get('latest_when', result_obj.get('metadata', {}).get('latest_when', ''))
                    }),
                    'id': result_id,
                    'metadata': result_obj.get('metadata', {})
                })
                temporal_candidates.append((mock_event, combined_scores[result_id]))
            
            # Apply enhanced temporal weighting
            if temporal_candidates:
                weighted_results, temporal_context = integrate_enhanced_temporal(
                    query=query_fields,
                    candidates=temporal_candidates,
                    encoder=self.encoder,
                    lambda_e=0.5,  # Use balanced weights
                    lambda_h=0.5,
                    llm_client=self.llm_client,
                    view_mode='merged'
                )
                
                # Update scores based on temporal weighting
                if temporal_context.get('applied'):
                    logger.info(f"Applied enhanced temporal weighting: {temporal_context}")
                    for weighted_event, weighted_score in weighted_results:
                        result_id = weighted_event.id
                        if result_id in combined_scores:
                            combined_scores[result_id] = weighted_score
        
        # Sort by combined score
        sorted_results = sorted(
            [(result_objects[rid], score) for rid, score in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def get_context_from_dimension_results(self, results: List[Tuple], 
                                          storage_store) -> List[Tuple[Any, float, str]]:
        """
        Convert dimension search results to full memory objects with context.
        
        Args:
            results: List of (result_obj, score) from dimension search
            storage_store: storage store to retrieve full events
        
        Returns:
            List of (event, score, context_string) tuples
        """
        contextualized_results = []
        
        for result_obj, score in results:
            try:
                # Get the full temporal group data to build proper context
                dimension = result_obj.get('dimension', 'unknown')
                merge_id = result_obj.get('id', '')
                merge_key = result_obj.get('merge_key', 'Group')
                event_count = result_obj.get('event_count', 1)
                
                # Initialize 5W1H variables from result metadata (might be overridden later)
                metadata = result_obj.get('metadata', {})
                who = metadata.get('latest_who', result_obj.get('latest_who', ''))
                what = metadata.get('latest_what', result_obj.get('latest_what', ''))
                when = metadata.get('latest_when', result_obj.get('latest_when', ''))
                where = metadata.get('latest_where', result_obj.get('latest_where', ''))
                why = metadata.get('latest_why', result_obj.get('latest_why', ''))
                how = metadata.get('latest_how', result_obj.get('latest_how', ''))
                
                # For temporal groups, get the full event timeline from storage
                if dimension == 'temporal' and storage_store and merge_id:
                    # Get the actual temporal merge group with all events
                    try:
                        # Query the temporal_merges collection for full group data
                        temporal_result = storage_store.client.get_collection('temporal_merges').get(
                            ids=[merge_id],
                            include=['metadatas']
                        )
                        
                        if temporal_result and temporal_result['metadatas']:
                            group_metadata = temporal_result['metadatas'][0]
                            raw_event_ids = group_metadata.get('raw_event_ids', '')
                            
                            # Parse the raw_event_ids (they might be comma-separated)
                            if isinstance(raw_event_ids, str) and raw_event_ids:
                                event_ids = [eid.strip() for eid in raw_event_ids.split(',')]
                            elif isinstance(raw_event_ids, list):
                                event_ids = raw_event_ids
                            else:
                                event_ids = []
                            
                            # Get all events from the raw_events collection
                            timeline_events = []
                            if event_ids:
                                raw_events = storage_store.raw_events_collection.get(
                                    ids=event_ids,
                                    include=['metadatas']
                                )
                                
                                for i, metadata in enumerate(raw_events.get('metadatas', [])):
                                    if metadata:
                                        timeline_events.append({
                                            'id': raw_events['ids'][i],
                                            'who': metadata.get('who', ''),
                                            'what': metadata.get('what', ''),
                                            'when': metadata.get('when', ''),
                                            'where': metadata.get('where', ''),
                                            'why': metadata.get('why', ''),
                                            'how': metadata.get('how', ''),
                                            'event_type': metadata.get('event_type', '')
                                        })
                            
                            # Build proper LLM context using our format
                            context_data = {
                                'timeline': [
                                    {
                                        'components': {
                                            'who': evt['who'],
                                            'what': evt['what'],
                                            'when': evt['when'],
                                            'where': evt['where'],
                                            'why': evt['why'],
                                            'how': evt['how']
                                        }
                                    }
                                    for evt in timeline_events
                                ],
                                'key_information': {},
                                'patterns': [],
                                'relationships': [],
                                'narrative_summary': f"This temporal memory group contains {len(timeline_events)} related events."
                            }
                            
                            # Import the function from web_app
                            import sys
                            if 'web_app' in sys.modules:
                                from web_app import generate_llm_text_block
                                context_str = generate_llm_text_block(context_data, dimension, merge_id)
                            else:
                                # Fallback to simple format
                                context_str = f"[Temporal Memory Group - {merge_id}]\nContains {len(timeline_events)} events\n"
                                for evt in timeline_events:
                                    context_str += f"• {evt['who']}: {evt['what'][:100]}...\n"
                                    
                    except Exception as e:
                        logger.error(f"Error getting full temporal group data: {e}")
                        # Fall back to original simple format
                        context_str = f"[{dimension.title()} Memory Group - {merge_key}]\nContains {event_count} events"
                else:
                    # For non-temporal or if no storage, use simple format
                    context_lines = []
                    context_lines.append(f"[{dimension.title()} Memory Group - {merge_key}]")
                    context_lines.append(f"Contains {event_count} events")
                    
                    if who and what:
                        context_lines.append(f"\nEvents in this group:")
                        context_lines.append(f"• {who}: {what}")
                        if when:
                            # Add UTC label if timestamp looks like ISO format
                            if 'T' in when or 'Z' in when:
                                context_lines.append(f"  When: {when} (UTC)")
                            else:
                                context_lines.append(f"  When: {when}")
                    
                    context_str = '\n'.join(context_lines)
                
                # Create a synthetic Event from the merge group metadata for the LLM
                # This represents the merged/summarized state of the group
                from models.event import Event, FiveW1H, EventType
                from datetime import datetime
                
                # Parse created_at if available
                created_at_str = result_obj.get('created_at', datetime.utcnow().isoformat())
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    except:
                        created_at = datetime.utcnow()
                else:
                    created_at = datetime.utcnow()
                
                synthetic_event = Event(
                    id=result_obj.get('id', ''),
                    event_type=EventType.OBSERVATION,  # Default type
                    five_w1h=FiveW1H(
                        who=who,
                        what=what,
                        when=when,
                        where=where,
                        why=why,
                        how=how
                    ),
                    episode_id=result_obj.get('episode_id', ''),
                    created_at=created_at
                )
                
                contextualized_results.append((synthetic_event, score, context_str))
                
            except Exception as e:
                logger.error(f"Error getting context for result {result_obj.get('id')}: {e}")
        
        return contextualized_results
