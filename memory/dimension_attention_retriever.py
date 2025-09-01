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
    
    def __init__(self, encoder, multi_merger, similarity_cache=None, chromadb_store=None, temporal_manager=None):
        """
        Initialize the dimension attention retriever.
        
        Args:
            encoder: Dual-space encoder for embeddings
            multi_merger: Multi-dimensional merger instance
            similarity_cache: Optional similarity cache for fast lookups
            chromadb_store: ChromaDB store for retrieving merge groups
            temporal_manager: Temporal manager for handling temporal queries
        """
        self.encoder = encoder
        self.multi_merger = multi_merger
        self.similarity_cache = similarity_cache
        self.chromadb = chromadb_store
        self.temporal_manager = temporal_manager
        
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
            
            # Use only Euclidean embedding for ChromaDB search
            # (ChromaDB collections are created with 768-dim Euclidean embeddings only)
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
        
        if not self.chromadb or not self.chromadb.temporal_merges_collection:
            logger.warning("ChromaDB or temporal_merges_collection not available")
            return results
        
        try:
            # Get all temporal groups
            all_groups = self.chromadb.temporal_merges_collection.get(
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
        
        # Apply temporal recency boost for high temporal weight queries
        if temporal_weight > 0.4:  # If temporal dimension is dominant
            from datetime import datetime
            from dateutil import parser as date_parser
            current_time = datetime.utcnow()
            
            for result_id, result_obj in result_objects.items():
                # Get the latest timestamp from metadata
                latest_when = result_obj.get('latest_when', '')
                if latest_when:
                    try:
                        event_time = date_parser.parse(latest_when)
                        # Normalize timezones
                        if event_time.tzinfo is not None:
                            event_time = event_time.replace(tzinfo=None)
                        
                        # Calculate recency boost (exponential decay)
                        time_diff = current_time - event_time
                        hours_diff = max(0, time_diff.total_seconds() / 3600)
                        
                        # Strong exponential decay for recency queries
                        decay_rate = 0.5
                        window_hours = 24.0
                        recency_factor = np.exp(-hours_diff / (window_hours * decay_rate))
                        
                        # Apply recency boost proportional to temporal weight
                        combined_scores[result_id] *= (1.0 + temporal_weight * recency_factor)
                        
                    except Exception as e:
                        logger.debug(f"Could not parse timestamp for recency boost: {e}")
        
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
                
                # For temporal groups, get the full event timeline from ChromaDB
                if dimension == 'temporal' and chromadb_store and merge_id:
                    # Get the actual temporal merge group with all events
                    try:
                        # Query the temporal_merges collection for full group data
                        temporal_result = chromadb_store.client.get_collection('temporal_merges').get(
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
                                raw_events = chromadb_store.raw_events_collection.get(
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
                    # For non-temporal or if no ChromaDB, use simple format
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
