"""
Temporal query detection and handling for the memory system.

Enhanced with LLM integration, expanded pattern matching, and regex-based time extraction.
Uses probabilistic matching via embeddings to detect temporal intent and
applies smooth time-based weighting that integrates with the dual-space system.
"""

import re
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dateutil import parser as date_parser
import logging

logger = logging.getLogger(__name__)


class TemporalQueryHandler:
    """
    Enhanced temporal query handler with LLM integration and expanded patterns.
    
    Provides intelligent temporal intent detection through:
    - LLM-based contextual understanding when available (primary)
    - Expanded semantic matching with comprehensive reference patterns (fallback)
    - Regex-based time expression extraction
    - Smooth probabilistic weighting functions
    """
    
    # Comprehensive temporal reference examples for semantic matching
    TEMPORAL_REFERENCES = {
        'strong_recency': [
            # Original patterns
            "last thing we discussed",
            "what did we just talk about",
            "most recent conversation",
            "latest discussion",
            "what was the last topic",
            "just now",
            "the previous message",
            "our last exchange",
            "the last point you made",
            "right before this",
            # Enhanced patterns
            "what were we just discussing",
            "remind me what we just covered",
            "the thing we just mentioned",
            "what did you just say",
            "what was that last bit",
            "going back to what we just said",
            "that last part",
            "the most recent thing",
            "what we were just talking about",
            "latest thing we covered",
            "very last topic",
            "immediately before this",
            "right before now",
            "just a second ago",
            "literally just now"
        ],
        'moderate_recency': [
            # Original patterns
            "recent memories",
            "recently mentioned",
            "earlier today",
            "a moment ago",
            "not long ago",
            "a short while back",
            "something you said earlier",
            "from a little bit ago",
            "earlier in the day",
            "what we touched on recently",
            # Enhanced patterns
            "a few minutes ago",
            "several minutes back",
            "within the last hour",
            "in the past hour",
            "sometime recently",
            "not too long ago",
            "a little while ago",
            "fairly recently",
            "somewhat recently",
            "in recent memory",
            "from earlier",
            "a bit ago",
            "some time ago today",
            "earlier this session",
            "previously today"
        ],
        'session_based': [
            # Original patterns
            "today's conversation",
            "this session",
            "current discussion",
            "our talk today",
            "this morning's chat",
            "this afternoon's talk",
            "our ongoing conversation",
            "what we've covered so far",
            "discussion from earlier in this session",
            "the thread of today",
            # Enhanced patterns
            "during this chat",
            "in our current talk",
            "throughout this discussion",
            "over the course of this conversation",
            "since we started talking",
            "from when we began",
            "in this exchange",
            "during our dialogue",
            "this particular conversation",
            "our discussion today",
            "everything we've discussed",
            "all we've talked about",
            "the full conversation",
            "complete discussion so far"
        ],
        'historical': [
            # Original patterns
            "previous discussions",
            "earlier conversations", 
            "before this",
            "past topics",
            "history of our talks",
            "last week's chat",
            "what we discussed previously",
            "prior discussions",
            "old conversations",
            "in the past",
            # Enhanced patterns
            "from before",
            "historical context",
            "way back when",
            "long ago",
            "ages ago",
            "from way earlier",
            "ancient history",
            "from the archives",
            "old topics",
            "bygone discussions",
            "former conversations",
            "past exchanges",
            "previous sessions",
            "older chats"
        ],
        'ordered': [
            # Original patterns
            "first thing we discussed",
            "how this started",
            "beginning of conversation",
            "initial topic",
            "where we began",
            "what came next",
            "the second thing we talked about",
            "middle of our chat",
            "toward the end",
            "the final point we discussed",
            # Enhanced patterns
            "at the start",
            "when we first began",
            "the opening topic",
            "how we kicked off",
            "the starting point",
            "chronologically",
            "in order",
            "sequentially",
            "the progression",
            "step by step",
            "the third thing",
            "after that",
            "following that",
            "subsequently",
            "penultimate topic",
            "second to last",
            "near the end"
        ],
        'long_term': [
            # Original patterns
            "conversations from last week",
            "chats from last month",
            "a while back",
            "discussions long ago",
            "previous sessions",
            "our past interactions",
            "from months ago",
            "old session memories",
            "earlier history",
            "long-term memory"
        ],
        'specific_time': [
            # Patterns for specific time references
            "5 minutes ago",
            "10 minutes back",
            "half an hour ago",
            "an hour ago",
            "2 hours ago",
            "3 hours back",
            "yesterday",
            "last night",
            "this morning",
            "yesterday morning",
            "yesterday afternoon",
            "last Monday",
            "last week",
            "two weeks ago",
            "last month",
            "couple days ago",
            "few days back",
            "other day"
        ],
        'relative_position': [
            # Relative positioning in conversation
            "before we talked about",
            "after we discussed",
            "around the time we mentioned",
            "when we were talking about",
            "during the discussion of",
            "while covering",
            "in the context of",
            "related to when",
            "connected to our talk about",
            "tied to the discussion of"
        ]
    }
    
    # Regex patterns for time extraction
    TIME_PATTERNS = [
        # Relative time with units
        (r'(\d+)\s*(second|minute|hour|day|week|month)s?\s*ago', 'relative_past'),
        (r'(\d+)\s*(second|minute|hour|day|week|month)s?\s*back', 'relative_past'),
        (r'last\s+(\d+)\s*(second|minute|hour|day|week|month)s?', 'relative_past'),
        (r'past\s+(\d+)\s*(second|minute|hour|day|week|month)s?', 'relative_past'),
        (r'previous\s+(\d+)\s*(second|minute|hour|day|week|month)s?', 'relative_past'),
        
        # Specific relative times
        (r'yesterday\s*(morning|afternoon|evening|night)?', 'yesterday'),
        (r'today\s*(morning|afternoon|evening|night)?', 'today'),
        (r'this\s+(morning|afternoon|evening)', 'today_part'),
        (r'last\s+(night|evening)', 'last_night'),
        (r'last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'last_weekday'),
        
        # Approximate times
        (r'(a|an)\s+(moment|minute|second)\s+ago', 'very_recent'),
        (r'just\s+now', 'immediate'),
        (r'right\s+now', 'immediate'),
        (r'(a\s+)?few\s+(seconds|minutes|hours)\s*(ago|back)', 'approximate_recent'),
        (r'(a\s+)?couple\s+(of\s+)?(minutes|hours|days)\s*(ago|back)', 'approximate_recent'),
        
        # ISO timestamps
        (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', 'iso_timestamp'),
        (r'\d{4}-\d{2}-\d{2}', 'iso_date'),
        
        # Time ranges
        (r'between\s+(\d+)\s+and\s+(\d+)\s+(minute|hour|day)s?\s*ago', 'time_range'),
        (r'within\s+(the\s+)?(last|past)\s+(\d+)\s+(minute|hour|day)s?', 'within_time'),
    ]

    def __init__(self, encoder=None, decay_factor: float = 0.9, window_hours: float = 24.0,
                 similarity_cache=None, chromadb_store=None, llm_client=None):
        """
        Initialize enhanced temporal handler.
        
        Args:
            encoder: The dual-space encoder for computing embeddings
            decay_factor: Base decay rate for time-based scoring (0-1)
            window_hours: Time window for "recent" queries (in hours)
            similarity_cache: Optional similarity cache for faster prototype matching
            chromadb_store: Optional ChromaDB store for metadata-based filtering
            llm_client: Optional LLM client for intelligent temporal analysis
        """
        self.encoder = encoder
        self.decay_factor = decay_factor
        self.window_hours = window_hours
        self.similarity_cache = similarity_cache
        self.chromadb_store = chromadb_store
        self.llm_client = llm_client
        self._temporal_embeddings = None
        self._temporal_embedding_ids = {}  # Maps temporal_type to virtual IDs for cache
        self._compiled_patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in self.TIME_PATTERNS]
        
    def _ensure_temporal_embeddings(self):
        """Lazily compute and cache temporal reference embeddings using dual-space encoder."""
        if self._temporal_embeddings is not None or self.encoder is None:
            return
            
        self._temporal_embeddings = {}
        
        for temporal_type, examples in self.TEMPORAL_REFERENCES.items():
            # Compute embeddings for each example using dual-space encoder
            euclidean_embeddings = []
            hyperbolic_embeddings = []
            
            for example in examples:
                # Use the encoder's field-aware encoding
                emb = self.encoder.encode({'what': example})
                euclidean_embeddings.append(emb['euclidean_anchor'])
                hyperbolic_embeddings.append(emb['hyperbolic_anchor'])
            
            # Store mean embeddings as prototypes for this temporal type
            if euclidean_embeddings:
                self._temporal_embeddings[temporal_type] = {
                    'euclidean': np.mean(euclidean_embeddings, axis=0),
                    'hyperbolic': np.mean(hyperbolic_embeddings, axis=0)
                }
                # Generate virtual ID for cache compatibility
                self._temporal_embedding_ids[temporal_type] = f"temporal_proto_{temporal_type}"
    
    def detect_temporal_intent(self, query: Dict[str, str], 
                               lambda_e: float = 0.5, lambda_h: float = 0.5) -> Tuple[str, float, float]:
        """
        Detect temporal intent probabilistically using dual-space embeddings.
        
        Args:
            query: Query fields dictionary
            lambda_e: Weight for Euclidean space (from query analysis)
            lambda_h: Weight for Hyperbolic space (from query analysis)
        
        Returns:
            (temporal_type, similarity_score, temporal_strength)
        """
        if self.encoder is None:
            return '', 0.0, 0.0
            
        self._ensure_temporal_embeddings()
        
        # Encode the query using dual-space encoder
        query_embedding = self.encoder.encode(query)
        query_euc = query_embedding['euclidean_anchor']
        query_hyp = query_embedding['hyperbolic_anchor']
        
        # Normalize for similarity computation
        query_euc_norm = query_euc / (np.linalg.norm(query_euc) + 1e-8)
        
        # Compute similarities to temporal prototypes
        best_type = ''
        best_similarity = 0.0
        
        for temporal_type, prototypes in self._temporal_embeddings.items():
            # Euclidean similarity (cosine)
            proto_euc_norm = prototypes['euclidean'] / (np.linalg.norm(prototypes['euclidean']) + 1e-8)
            euc_similarity = np.dot(query_euc_norm, proto_euc_norm)
            
            # Hyperbolic similarity (using geodesic distance)
            from memory.dual_space_encoder import HyperbolicOperations
            hyp_distance = HyperbolicOperations.geodesic_distance(query_hyp, prototypes['hyperbolic'])
            # Convert distance to similarity (bounded [0,1])
            hyp_similarity = np.exp(-hyp_distance)
            
            # Combine similarities using query weights
            combined_similarity = lambda_e * euc_similarity + lambda_h * hyp_similarity
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_type = temporal_type
        
        # Convert similarity to temporal strength
        temporal_strength = self._similarity_to_strength(best_similarity, best_type)
        
        return best_type, best_similarity, temporal_strength
    
    def detect_temporal_intent_with_llm(self, query: Dict[str, str], 
                                       lambda_e: float = 0.5, 
                                       lambda_h: float = 0.5) -> Tuple[str, float, float, Optional[Dict]]:
        """
        Detect temporal intent using LLM when available, with fallback to embeddings.
        
        Args:
            query: Query fields dictionary
            lambda_e: Weight for Euclidean space
            lambda_h: Weight for Hyperbolic space
        
        Returns:
            (temporal_type, similarity_score, temporal_strength, extracted_context)
            extracted_context includes any parsed time references
        """
        # Try LLM-based detection first if available
        if self.llm_client and hasattr(self.llm_client, 'is_available') and self.llm_client.is_available():
            try:
                llm_result = self._analyze_with_llm(query)
                if llm_result:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM temporal analysis failed, falling back to embeddings: {e}")
        
        # Fallback to embedding-based detection
        temporal_type, similarity, strength = self.detect_temporal_intent(query, lambda_e, lambda_h)
        
        # Try to extract time references with regex
        extracted_context = self._extract_time_references(query)
        
        return temporal_type, similarity, strength, extracted_context
    
    def _analyze_with_llm(self, query: Dict[str, str]) -> Optional[Tuple[str, float, float, Dict]]:
        """
        Use LLM to analyze temporal intent in the query.
        
        Returns:
            Tuple of (temporal_type, similarity, strength, extracted_context) or None
        """
        if not self.llm_client:
            return None
        
        # Build prompt for LLM
        query_text = ' '.join([str(v) for v in query.values() if v])
        
        prompt = f"""Analyze the temporal intent in this query and extract any time references.

Query: "{query_text}"

Identify:
1. Temporal type (one of: strong_recency, moderate_recency, session_based, historical, ordered, specific_time, relative_position, none)
2. Confidence score (0.0 to 1.0)
3. Any specific time references (e.g., "5 minutes ago", "yesterday", "last week")
4. Relative time in hours from now (if applicable)

Respond in JSON format:
{{
    "temporal_type": "type_here",
    "confidence": 0.0,
    "time_reference": "extracted reference or null",
    "hours_ago": null or number,
    "explanation": "brief explanation"
}}

Examples:
- "what did we just discuss" -> strong_recency, high confidence
- "remind me what we talked about 5 minutes ago" -> specific_time, high confidence, 5 minutes = 0.083 hours
- "our conversation from yesterday" -> specific_time, high confidence, ~24 hours ago
- "how did this discussion start" -> ordered, high confidence
"""
        
        try:
            # Get LLM response
            response = self.llm_client.generate(
                prompt,
                max_tokens=200,
                temperature=0.1  # Low temperature for consistency
            )
            
            # Parse JSON response
            result = self._parse_llm_response(response)
            if result:
                temporal_type = result.get('temporal_type', 'none')
                confidence = float(result.get('confidence', 0.0))
                
                # Map confidence to strength using our sigmoid function
                strength = self._similarity_to_strength(confidence, temporal_type)
                
                # Build extracted context
                extracted_context = {
                    'time_reference': result.get('time_reference'),
                    'hours_ago': result.get('hours_ago'),
                    'explanation': result.get('explanation'),
                    'llm_analyzed': True
                }
                
                # Calculate approximate datetime if hours_ago provided
                if extracted_context['hours_ago'] is not None:
                    try:
                        reference_time = datetime.utcnow() - timedelta(hours=float(extracted_context['hours_ago']))
                        extracted_context['reference_time'] = reference_time.isoformat()
                    except:
                        pass
                
                return temporal_type, confidence, strength, extracted_context
                
        except Exception as e:
            logger.debug(f"LLM temporal analysis error: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response to extract temporal analysis."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Try parsing entire response as JSON
            return json.loads(response)
        except:
            logger.debug(f"Could not parse LLM response as JSON: {response[:200]}")
            return None
    
    def _extract_time_references(self, query: Dict[str, str]) -> Dict:
        """
        Extract time references from query using regex patterns.
        
        Args:
            query: Query fields dictionary
            
        Returns:
            Dictionary with extracted time information
        """
        query_text = ' '.join([str(v) for v in query.values() if v]).lower()
        extracted = {
            'time_references': [],
            'has_time_reference': False
        }
        
        for pattern, pattern_type in self._compiled_patterns:
            matches = pattern.finditer(query_text)
            for match in matches:
                extracted['has_time_reference'] = True
                reference = {
                    'text': match.group(0),
                    'type': pattern_type,
                    'groups': match.groups()
                }
                
                # Try to convert to approximate hours ago
                hours_ago = self._convert_to_hours(reference)
                if hours_ago is not None:
                    reference['hours_ago'] = hours_ago
                    reference['reference_time'] = (datetime.utcnow() - timedelta(hours=hours_ago)).isoformat()
                
                extracted['time_references'].append(reference)
        
        return extracted
    
    def _convert_to_hours(self, reference: Dict) -> Optional[float]:
        """
        Convert extracted time reference to hours ago.
        
        Args:
            reference: Extracted time reference dict
            
        Returns:
            Number of hours ago, or None if cannot convert
        """
        pattern_type = reference['type']
        groups = reference['groups']
        
        if pattern_type == 'relative_past' and len(groups) >= 2:
            try:
                value = float(groups[0])
                unit = groups[1].lower()
                
                multipliers = {
                    'second': 1/3600,
                    'minute': 1/60,
                    'hour': 1,
                    'day': 24,
                    'week': 168,
                    'month': 720  # Approximate
                }
                
                return value * multipliers.get(unit, 1)
            except:
                pass
        
        elif pattern_type == 'yesterday':
            return 24.0  # Approximate
        
        elif pattern_type == 'very_recent':
            return 0.017  # About 1 minute
        
        elif pattern_type == 'immediate':
            return 0.0
        
        elif pattern_type == 'approximate_recent':
            # "a few minutes" -> ~5 minutes
            text = reference['text'].lower()
            if 'second' in text:
                return 0.001
            elif 'minute' in text:
                return 0.083  # ~5 minutes
            elif 'hour' in text:
                return 2.0  # ~2 hours
            elif 'day' in text:
                return 48.0  # ~2 days
        
        return None
    
    def _similarity_to_strength(self, similarity: float, temporal_type: str) -> float:
        """
        Convert similarity score to temporal strength using smooth probability function.
        
        Different temporal types have different thresholds and curves.
        """
        # Type-specific parameters for sigmoid curves - lowered thresholds for better detection
        params = {
            'strong_recency': {'threshold': 0.25, 'steepness': 10},  # Lower threshold, steeper curve
            'moderate_recency': {'threshold': 0.25, 'steepness': 8},  # Lower threshold
            'session_based': {'threshold': 0.2, 'steepness': 6},      # Lower threshold
            'historical': {'threshold': 0.2, 'steepness': 6},
            'ordered': {'threshold': 0.25, 'steepness': 7},
            'specific_time': {'threshold': 0.2, 'steepness': 8},
            'relative_position': {'threshold': 0.25, 'steepness': 7}
        }
        
        p = params.get(temporal_type, {'threshold': 0.2, 'steepness': 6})
        
        # Sigmoid function centered at threshold
        # strength = 1 / (1 + exp(-steepness * (similarity - threshold)))
        strength = 1.0 / (1.0 + np.exp(-p['steepness'] * (similarity - p['threshold'])))
        
        # Apply subtle minimum to avoid complete zeroing
        return max(0.15, strength)  # Increased minimum from 0.1 to 0.15
    
    def compute_temporal_weight(self, event_timestamp: str, 
                               query_time: Optional[datetime] = None,
                               temporal_type: str = '',
                               temporal_strength: float = 0.0,
                               is_raw_event: bool = False) -> float:
        """
        Compute smooth temporal weight for an event based on its timestamp.
        
        Uses probabilistic decay functions rather than hard thresholds.
        
        Args:
            event_timestamp: ISO format timestamp from event
            query_time: Time of query (defaults to now)
            temporal_type: Type of temporal query detected
            temporal_strength: Strength of temporal signal (0-1)
            is_raw_event: Whether this is a raw event (may need different weighting)
            
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
    
    def compute_temporal_weight_enhanced(self, event_timestamp: str,
                                        query_time: Optional[datetime] = None,
                                        temporal_type: str = '',
                                        temporal_strength: float = 0.0,
                                        extracted_context: Optional[Dict] = None,
                                        is_raw_event: bool = False) -> float:
        """
        Enhanced temporal weight computation using extracted context.
        
        Args:
            event_timestamp: ISO format timestamp from event
            query_time: Time of query (defaults to now)
            temporal_type: Type of temporal query detected
            temporal_strength: Strength of temporal signal (0-1)
            extracted_context: Additional context from LLM/regex extraction
            is_raw_event: Whether this is a raw event
            
        Returns:
            Temporal weight factor (0-1)
        """
        # If we have a specific reference time from extraction, use it
        if extracted_context and extracted_context.get('reference_time'):
            try:
                reference_time = date_parser.parse(extracted_context['reference_time'])
                event_time = date_parser.parse(event_timestamp)
                
                # Ensure timezone compatibility
                if event_time.tzinfo is None and reference_time.tzinfo is not None:
                    reference_time = reference_time.replace(tzinfo=None)
                elif event_time.tzinfo is not None and reference_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=None)
                
                # Calculate distance from reference time
                time_diff = abs((event_time - reference_time).total_seconds() / 3600)
                
                # Use Gaussian-like scoring centered on reference time
                sigma = 2.0  # 2-hour window around reference
                weight = np.exp(-(time_diff ** 2) / (2 * sigma ** 2))
                
                # Apply strength scaling
                return temporal_strength * weight + (1 - temporal_strength) * 0.5
                
            except Exception as e:
                logger.debug(f"Error using reference time: {e}")
        
        # Otherwise use base implementation
        return self.compute_temporal_weight(
            event_timestamp, query_time, temporal_type, 
            temporal_strength, is_raw_event
        )
    
    def apply_temporal_weighting(self, candidates: List[Tuple],
                                temporal_type: str,
                                temporal_strength: float,
                                query_time: Optional[datetime] = None,
                                view_mode: str = 'merged') -> List[Tuple]:
        """
        Apply smooth temporal weighting to retrieval candidates.
        
        Uses probabilistic blending rather than hard reranking.
        
        Args:
            candidates: List of (event, score) tuples
            temporal_type: Type of temporal query
            temporal_strength: Strength of temporal signal
            query_time: Time of query
            view_mode: 'merged' or 'raw' - affects temporal weighting strategy
            
        Returns:
            Reranked list of (event, adjusted_score) tuples
        """
        if temporal_strength < 0.1:
            return candidates
            
        weighted_results = []
        
        for event, score in candidates:
            # Get temporal weight based on event's timestamp
            # Determine if this is a raw event
            is_raw = view_mode == 'raw' or not hasattr(event, 'merged_count')
            
            temporal_weight = self.compute_temporal_weight(
                event.five_w1h.when,
                query_time,
                temporal_type,
                temporal_strength,
                is_raw_event=is_raw
            )
            
            # Probabilistic combination of semantic and temporal scores
            # Use harmonic mean for smoother blending
            if temporal_strength > 0.5:
                # Strong temporal signal: heavily weight temporal factor
                adjusted_score = score * (temporal_weight ** (temporal_strength * 0.9))  # Increased from 0.7
            else:
                # Weak temporal signal: still apply meaningful temporal weighting
                adjusted_score = (1 - temporal_strength * 0.4) * score + (temporal_strength * 0.4) * temporal_weight * score  # Increased from 0.3
            
            weighted_results.append((event, adjusted_score))
        
        # Re-sort by adjusted scores
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_results
    
    def apply_temporal_weighting_enhanced(self, candidates: List[Tuple],
                                         temporal_type: str,
                                         temporal_strength: float,
                                         extracted_context: Optional[Dict] = None,
                                         query_time: Optional[datetime] = None,
                                         view_mode: str = 'merged') -> List[Tuple]:
        """
        Apply enhanced temporal weighting with extracted context.
        
        Args:
            candidates: List of (event, score) tuples
            temporal_type: Type of temporal query
            temporal_strength: Strength of temporal signal
            extracted_context: Additional context from extraction
            query_time: Time of query
            view_mode: 'merged' or 'raw'
            
        Returns:
            Reranked list of (event, adjusted_score) tuples
        """
        if temporal_strength < 0.1:
            return candidates
        
        weighted_results = []
        
        for event, score in candidates:
            is_raw = view_mode == 'raw' or not hasattr(event, 'merged_count')
            
            # Use enhanced weight computation
            temporal_weight = self.compute_temporal_weight_enhanced(
                event.five_w1h.when,
                query_time,
                temporal_type,
                temporal_strength,
                extracted_context,
                is_raw_event=is_raw
            )
            
            # Apply stronger weighting when we have specific time references
            if extracted_context and extracted_context.get('has_time_reference'):
                # Stronger temporal influence for specific time queries
                adjusted_score = score * (temporal_weight ** (temporal_strength * 1.2))
            else:
                # Standard weighting
                if temporal_strength > 0.5:
                    adjusted_score = score * (temporal_weight ** (temporal_strength * 0.9))
                else:
                    adjusted_score = (1 - temporal_strength * 0.4) * score + (temporal_strength * 0.4) * temporal_weight * score
            
            weighted_results.append((event, adjusted_score))
        
        # Re-sort by adjusted scores
        weighted_results.sort(key=lambda x: x[1], reverse=True)
        
        return weighted_results
    
    def compute_temporal_context(self, query: Dict[str, str], 
                                 lambda_e: float = 0.5, lambda_h: float = 0.5) -> Dict[str, Any]:
        """
        Extract probabilistic temporal context from query.
        
        Returns dict with:
            - temporal_type: str (detected type)
            - similarity: float (how similar to temporal prototype)
            - temporal_strength: float (probability of temporal intent)
            - suggested_window: Time window in hours
        """
        temporal_type, similarity, strength = self.detect_temporal_intent(query, lambda_e, lambda_h)
        
        # Compute suggested time window based on type and strength
        window_multiplier = {
            'strong_recency': 0.5,
            'moderate_recency': 1.0,
            'session_based': 1.5,
            'historical': 10.0,
            'ordered': 20.0,
            'specific_time': 2.0,
            'relative_position': 5.0
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
                                      lambda_h: float,
                                      similarity_cache=None,
                                      chromadb_store=None,
                                      llm_client=None,
                                      view_mode: str = 'merged') -> Tuple[List[Tuple], Dict]:
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
        similarity_cache: Optional similarity cache for faster processing
        chromadb_store: Optional ChromaDB store for metadata filtering
        llm_client: Optional LLM client for intelligent temporal analysis
        view_mode: 'merged' or 'raw' - affects temporal weighting
        
    Returns:
        (reranked_results, temporal_context)
    """
    # Create temporal handler with encoder and optional components
    temporal_handler = TemporalQueryHandler(
        encoder=encoder,
        similarity_cache=similarity_cache,
        chromadb_store=chromadb_store,
        llm_client=llm_client
    )
    
    # Extract temporal context - use LLM if available
    if llm_client:
        temporal_type, similarity, strength, extracted_context = temporal_handler.detect_temporal_intent_with_llm(
            query, lambda_e, lambda_h
        )
        temporal_context = {
            'temporal_type': temporal_type,
            'similarity': float(similarity),
            'temporal_strength': float(strength),
            'extracted_context': extracted_context,
            'applied': False
        }
        
        # Calculate suggested window
        if extracted_context and extracted_context.get('hours_ago') is not None:
            temporal_context['suggested_window'] = float(extracted_context['hours_ago']) * 1.5
        else:
            window_multiplier = {
                'strong_recency': 0.5,
                'moderate_recency': 1.0,
                'session_based': 1.5,
                'historical': 10.0,
                'ordered': 20.0,
                'specific_time': 2.0,
                'relative_position': 5.0
            }.get(temporal_type, 1.0)
            temporal_context['suggested_window'] = 24.0 * window_multiplier * (0.5 + strength)
    else:
        # Fallback to standard detection
        temporal_context = temporal_handler.compute_temporal_context(query, lambda_e, lambda_h)
        extracted_context = None
        strength = temporal_context['temporal_strength']
        temporal_type = temporal_context['temporal_type']
    
    # Only apply if we have sufficient temporal signal
    if temporal_context['temporal_strength'] < 0.2:
        # Too weak - don't apply temporal weighting
        temporal_context['applied'] = False
        return candidates, temporal_context
    
    # Apply temporal weighting - use enhanced version if we have extracted context
    if extracted_context:
        weighted_results = temporal_handler.apply_temporal_weighting_enhanced(
            candidates,
            temporal_type,
            strength,
            extracted_context,
            view_mode=view_mode
        )
    else:
        weighted_results = temporal_handler.apply_temporal_weighting(
            candidates,
            temporal_type,
            strength,
            view_mode=view_mode
        )
    
    temporal_context['applied'] = True
    
    # Log temporal influence
    if temporal_context['temporal_strength'] > 0.3:
        logger.info(f"Applied temporal weighting: type={temporal_context['temporal_type']}, "
                    f"strength={temporal_context['temporal_strength']:.2f}, "
                    f"similarity={temporal_context.get('similarity', 0):.3f}")
        if extracted_context and extracted_context.get('time_reference'):
            logger.info(f"Extracted time reference: {extracted_context['time_reference']}")
    
    return weighted_results, temporal_context


# Alias for backward compatibility
integrate_enhanced_temporal = integrate_temporal_with_dual_space


class TemporalMetadataFilter:
    """
    Helper class for creating ChromaDB metadata filters based on temporal queries.
    """
    
    @staticmethod
    def create_temporal_filter(temporal_type: str, temporal_strength: float,
                               query_time: Optional[datetime] = None) -> Optional[Dict]:
        """
        Create ChromaDB metadata filter for temporal queries.
        
        Args:
            temporal_type: Type of temporal query detected
            temporal_strength: Strength of temporal signal
            query_time: Reference time for query
            
        Returns:
            ChromaDB-compatible filter dict or None
        """
        if temporal_strength < 0.3:  # Too weak to filter
            return None
            
        if query_time is None:
            query_time = datetime.utcnow()
            
        # Create filters based on temporal type
        if temporal_type == 'strong_recency':
            # Last 2 hours
            cutoff = (query_time - timedelta(hours=2)).isoformat()
            return {"when": {"$gte": cutoff}}
            
        elif temporal_type == 'moderate_recency':
            # Last 24 hours
            cutoff = (query_time - timedelta(hours=24)).isoformat()
            return {"when": {"$gte": cutoff}}
            
        elif temporal_type == 'session_based':
            # Last 8 hours (typical session)
            cutoff = (query_time - timedelta(hours=8)).isoformat()
            return {"when": {"$gte": cutoff}}
            
        elif temporal_type == 'historical':
            # Older than 24 hours
            cutoff = (query_time - timedelta(hours=24)).isoformat()
            return {"when": {"$lt": cutoff}}
            
        return None
    
    @staticmethod
    def enhance_query_with_temporal_bounds(query_embedding: Dict,
                                          temporal_context: Dict) -> Dict:
        """
        Enhance query embedding with temporal bounds for more efficient retrieval.
        
        Args:
            query_embedding: The dual-space query embedding
            temporal_context: Temporal context from detection
            
        Returns:
            Enhanced query embedding with temporal hints
        """
        enhanced = query_embedding.copy()
        
        # Add temporal metadata to guide retrieval
        enhanced['temporal_hints'] = {
            'type': temporal_context.get('temporal_type', ''),
            'strength': temporal_context.get('temporal_strength', 0.0),
            'window_hours': temporal_context.get('suggested_window', 24.0)
        }
        
        return enhanced