"""
Temporal query detection and handling for the memory system.

Uses probabilistic matching via embeddings to detect temporal intent and
applies smooth time-based weighting that integrates with the dual-space system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dateutil import parser as date_parser
import logging

logger = logging.getLogger(__name__)


class TemporalQueryHandler:
    """
    Detects temporal intent using semantic similarity and applies time-aware weighting.
    
    This integrates seamlessly with the dual-space system by:
    1. Using embeddings to detect temporal intent probabilistically
    2. Computing smooth temporal weight factors
    3. Preserving semantic relevance while adding temporal awareness
    """
    
    # Temporal reference examples for semantic matching
    TEMPORAL_REFERENCES = {
        'strong_recency': [
            "last thing we discussed",
            "what did we just talk about",
            "most recent conversation",
            "latest discussion",
            "what was the last topic"
        ],
        'moderate_recency': [
            "recent memories",
            "recently mentioned",
            "earlier today",
            "a moment ago",
            "not long ago"
        ],
        'session_based': [
            "today's conversation",
            "this session",
            "current discussion",
            "our talk today",
            "this morning's chat"
        ],
        'historical': [
            "previous discussions",
            "earlier conversations", 
            "before this",
            "past topics",
            "history of our talks"
        ],
        'ordered': [
            "first thing we discussed",
            "how this started",
            "beginning of conversation",
            "initial topic",
            "where we began"
        ]
    }
    
    def __init__(self, encoder=None, decay_factor: float = 0.9, window_hours: float = 24.0):
        """
        Initialize temporal handler.
        
        Args:
            encoder: The dual-space encoder for computing embeddings
            decay_factor: Base decay rate for time-based scoring (0-1)
            window_hours: Time window for "recent" queries (in hours)
        """
        self.encoder = encoder
        self.decay_factor = decay_factor
        self.window_hours = window_hours
        self._temporal_embeddings = None
        
    def _ensure_temporal_embeddings(self):
        """Lazily compute and cache temporal reference embeddings."""
        if self._temporal_embeddings is not None or self.encoder is None:
            return
            
        self._temporal_embeddings = {}
        
        for temporal_type, examples in self.TEMPORAL_REFERENCES.items():
            # Compute embeddings for each example
            embeddings = []
            for example in examples:
                # Use the encoder's field encoding for 'what' field
                emb = self.encoder.encode({'what': example})
                embeddings.append(emb['euclidean_anchor'])
            
            # Store mean embedding as prototype for this temporal type
            if embeddings:
                self._temporal_embeddings[temporal_type] = np.mean(embeddings, axis=0)
    
    def detect_temporal_intent(self, query: Dict[str, str]) -> Tuple[str, float, float]:
        """
        Detect temporal intent probabilistically using embeddings.
        
        Returns:
            (temporal_type, similarity_score, temporal_strength)
        """
        if self.encoder is None:
            return '', 0.0, 0.0
            
        self._ensure_temporal_embeddings()
        
        # Encode the query
        query_embedding = self.encoder.encode(query)
        query_vec = query_embedding['euclidean_anchor']
        
        # Normalize for cosine similarity
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Compute similarities to temporal prototypes
        best_type = ''
        best_similarity = 0.0
        
        for temporal_type, prototype in self._temporal_embeddings.items():
            # Normalize prototype
            proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(query_vec, proto_norm)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_type = temporal_type
        
        # Convert similarity to temporal strength
        # Use sigmoid-like transformation for smooth probability
        temporal_strength = self._similarity_to_strength(best_similarity, best_type)
        
        return best_type, best_similarity, temporal_strength
    
    def _similarity_to_strength(self, similarity: float, temporal_type: str) -> float:
        """
        Convert similarity score to temporal strength using smooth probability function.
        
        Different temporal types have different thresholds and curves.
        """
        # Type-specific parameters for sigmoid curves
        params = {
            'strong_recency': {'threshold': 0.4, 'steepness': 8},
            'moderate_recency': {'threshold': 0.35, 'steepness': 6},
            'session_based': {'threshold': 0.3, 'steepness': 5},
            'historical': {'threshold': 0.3, 'steepness': 5},
            'ordered': {'threshold': 0.35, 'steepness': 6}
        }
        
        p = params.get(temporal_type, {'threshold': 0.3, 'steepness': 5})
        
        # Sigmoid function centered at threshold
        # strength = 1 / (1 + exp(-steepness * (similarity - threshold)))
        strength = 1.0 / (1.0 + np.exp(-p['steepness'] * (similarity - p['threshold'])))
        
        # Apply subtle minimum to avoid complete zeroing
        return max(0.1, strength)
    
    def compute_temporal_weight(self, event_timestamp: str, 
                               query_time: Optional[datetime] = None,
                               temporal_type: str = '',
                               temporal_strength: float = 0.0) -> float:
        """
        Compute smooth temporal weight for an event based on its timestamp.
        
        Uses probabilistic decay functions rather than hard thresholds.
        
        Args:
            event_timestamp: ISO format timestamp from event
            query_time: Time of query (defaults to now)
            temporal_type: Type of temporal query detected
            temporal_strength: Strength of temporal signal (0-1)
            
        Returns:
            Temporal weight factor (0-1) to blend with semantic similarity
        """
        if not event_timestamp or temporal_strength < 0.1:
            return 1.0  # No temporal weighting
            
        try:
            # Parse event time
            event_time = date_parser.parse(event_timestamp)
            if query_time is None:
                query_time = datetime.utcnow()
            
            # Ensure both are timezone-aware or both naive
            if event_time.tzinfo is None and query_time.tzinfo is not None:
                query_time = query_time.replace(tzinfo=None)
            elif event_time.tzinfo is not None and query_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=None)
                
            # Calculate time difference in hours
            time_diff = query_time - event_time
            hours_diff = max(0, time_diff.total_seconds() / 3600)
            
            # Compute base temporal weight using smooth decay
            if temporal_type == 'strong_recency':
                # Strong exponential decay for very recent items
                decay_rate = 0.5  # Faster decay
                weight = np.exp(-hours_diff / (self.window_hours * decay_rate))
                
            elif temporal_type == 'moderate_recency':
                # Moderate Gaussian-like decay
                sigma = self.window_hours
                weight = np.exp(-(hours_diff ** 2) / (2 * sigma ** 2))
                
            elif temporal_type == 'session_based':
                # Smooth step function for session boundaries
                session_hours = 8  # Typical session length
                weight = 1.0 / (1.0 + np.exp((hours_diff - session_hours) / 2))
                
            elif temporal_type == 'historical':
                # Inverse recency - older is slightly better
                # But with bounds to avoid extreme values
                weight = 1.0 - np.exp(-hours_diff / (self.window_hours * 4))
                weight = 0.3 + 0.7 * weight  # Bound between 0.3 and 1.0
                
            elif temporal_type == 'ordered':
                # Prefer older events (for "first", "beginning")
                weight = 1.0 - np.exp(-hours_diff / (self.window_hours * 10))
                
            else:
                # Default: mild exponential decay
                weight = self.decay_factor ** (hours_diff / self.window_hours)
            
            # Blend temporal weight with neutral based on strength
            # This creates smooth interpolation between temporal and non-temporal
            final_weight = temporal_strength * weight + (1 - temporal_strength) * 0.7
            
            # Ensure weight stays in reasonable bounds
            return np.clip(final_weight, 0.1, 1.0)
            
        except Exception as e:
            logger.warning(f"Error parsing timestamp {event_timestamp}: {e}")
            return 1.0  # No penalty if we can't parse
    
    def apply_temporal_weighting(self, candidates: List[Tuple],
                                temporal_type: str,
                                temporal_strength: float,
                                query_time: Optional[datetime] = None) -> List[Tuple]:
        """
        Apply smooth temporal weighting to retrieval candidates.
        
        Uses probabilistic blending rather than hard reranking.
        
        Args:
            candidates: List of (event, score) tuples
            temporal_type: Type of temporal query
            temporal_strength: Strength of temporal signal
            query_time: Time of query
            
        Returns:
            Reranked list of (event, adjusted_score) tuples
        """
        if temporal_strength < 0.1:
            return candidates
            
        weighted_results = []
        
        for event, score in candidates:
            # Get temporal weight based on event's timestamp
            temporal_weight = self.compute_temporal_weight(
                event.five_w1h.when,
                query_time,
                temporal_type,
                temporal_strength
            )
            
            # Probabilistic combination of semantic and temporal scores
            # Use harmonic mean for smoother blending
            if temporal_strength > 0.5:
                # Strong temporal signal: geometric mean
                adjusted_score = score * (temporal_weight ** (temporal_strength * 0.7))
            else:
                # Weak temporal signal: arithmetic blend
                adjusted_score = (1 - temporal_strength * 0.3) * score + (temporal_strength * 0.3) * temporal_weight * score
            
            weighted_results.append((event, adjusted_score))
        
        # Re-sort by adjusted scores
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_results
    
    def compute_temporal_context(self, query: Dict[str, str]) -> Dict[str, any]:
        """
        Extract probabilistic temporal context from query.
        
        Returns dict with:
            - temporal_type: str (detected type)
            - similarity: float (how similar to temporal prototype)
            - temporal_strength: float (probability of temporal intent)
            - suggested_window: Time window in hours
        """
        temporal_type, similarity, strength = self.detect_temporal_intent(query)
        
        # Compute suggested time window based on type and strength
        window_multiplier = {
            'strong_recency': 0.5,
            'moderate_recency': 1.0,
            'session_based': 1.5,
            'historical': 10.0,
            'ordered': 20.0
        }.get(temporal_type, 1.0)
        
        # Adjust window based on strength
        suggested_window = self.window_hours * window_multiplier * (0.5 + strength)
        
        return {
            'temporal_type': temporal_type,
            'similarity': float(similarity),
            'temporal_strength': float(strength),
            'suggested_window': float(suggested_window)
        }


def integrate_temporal_with_dual_space(query: Dict[str, str],
                                      candidates: List[Tuple],
                                      encoder,
                                      lambda_e: float,
                                      lambda_h: float) -> Tuple[List[Tuple], Dict]:
    """
    Helper to integrate probabilistic temporal weighting with dual-space retrieval.
    
    This preserves the dual-space balance while adding temporal awareness through
    smooth probability functions rather than hard rules.
    
    Args:
        query: Query fields dict
        candidates: List of (event, dual_space_score) tuples
        encoder: The dual-space encoder (needed for temporal detection)
        lambda_e: Euclidean space weight
        lambda_h: Hyperbolic space weight
        
    Returns:
        (reranked_results, temporal_context)
    """
    # Create temporal handler with the encoder
    temporal_handler = TemporalQueryHandler(encoder=encoder)
    
    # Extract temporal context probabilistically
    temporal_context = temporal_handler.compute_temporal_context(query)
    
    # Only apply if we have sufficient temporal signal
    if temporal_context['temporal_strength'] < 0.2:
        # Too weak - don't apply temporal weighting
        temporal_context['applied'] = False
        return candidates, temporal_context
    
    # Apply smooth temporal weighting
    weighted_results = temporal_handler.apply_temporal_weighting(
        candidates,
        temporal_context['temporal_type'],
        temporal_context['temporal_strength']
    )
    
    temporal_context['applied'] = True
    
    # Log temporal influence
    if temporal_context['temporal_strength'] > 0.3:
        logger.info(f"Applied temporal weighting: type={temporal_context['temporal_type']}, "
                    f"strength={temporal_context['temporal_strength']:.2f}, "
                    f"similarity={temporal_context['similarity']:.3f}")
    
    return weighted_results, temporal_context