"""
Dual-space encoder with Euclidean and Hyperbolic heads for field-aware composition.
Replaces adaptive_embeddings.py with a more sophisticated encoding system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class HyperbolicOperations:
    """Operations in the Poincaré ball model of hyperbolic space with numerical stability."""
    
    @staticmethod
    def clip_norm(x: np.ndarray, max_norm: float = 0.999, epsilon: float = 1e-5) -> np.ndarray:
        """Clip vector norm to stay within Poincaré ball."""
        norm = np.linalg.norm(x)
        if norm >= max_norm:
            x = x * (max_norm - epsilon) / (norm + epsilon)
        return x
    
    @staticmethod
    def safe_sqrt(x: float, epsilon: float = 1e-10) -> float:
        """Safe square root with epsilon stabilization."""
        return np.sqrt(np.maximum(x, epsilon))
    
    @staticmethod
    def exp_map(x: np.ndarray, v: np.ndarray, c: float = 1.0, max_norm: float = 0.999) -> np.ndarray:
        """Exponential map from tangent space at x to the Poincaré ball with stability."""
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            return x
        
        # Clip x to ensure we're within the ball
        x = HyperbolicOperations.clip_norm(x, max_norm)
        
        x_dot = c * np.dot(x, x)
        lambda_x = 2 / (1 - x_dot + 1e-10)  # Add epsilon for stability
        
        # Stable tanh computation
        arg = np.sqrt(c) * lambda_x * v_norm / 2
        tanh_term = np.tanh(np.minimum(arg, 15.0))  # Prevent overflow in tanh
        direction = v / v_norm
        
        result = mobius_add(x, tanh_term * direction / (np.sqrt(c) * lambda_x), c)
        return HyperbolicOperations.clip_norm(result, max_norm)
    
    @staticmethod
    def log_map(x: np.ndarray, y: np.ndarray, c: float = 1.0, max_norm: float = 0.999) -> np.ndarray:
        """Logarithmic map from y to tangent space at x with stability."""
        # Ensure both points are within the ball
        x = HyperbolicOperations.clip_norm(x, max_norm)
        y = HyperbolicOperations.clip_norm(y, max_norm)
        
        diff = mobius_add(-x, y, c)
        diff_norm = np.linalg.norm(diff)
        
        if diff_norm < 1e-10:
            return np.zeros_like(x)
        
        x_dot = c * np.dot(x, x)
        lambda_x = 2 / (1 - x_dot + 1e-10)  # Add epsilon for stability
        
        # Stable arctanh computation
        arg = np.sqrt(c) * diff_norm
        arg = np.minimum(arg, 0.999)  # Ensure arctanh argument < 1
        
        return (2 / (np.sqrt(c) * lambda_x)) * np.arctanh(arg) * (diff / diff_norm)
    
    @staticmethod
    def geodesic_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0, max_norm: float = 0.999) -> float:
        """Geodesic distance in the Poincaré ball with stability."""
        # Ensure both points are within the ball
        x = HyperbolicOperations.clip_norm(x, max_norm)
        y = HyperbolicOperations.clip_norm(y, max_norm)
        
        diff = mobius_add(-x, y, c)
        diff_norm = np.linalg.norm(diff)
        
        if diff_norm < 1e-10:
            return 0.0
        
        # Stable arctanh computation
        arg = np.sqrt(c) * diff_norm
        arg = np.minimum(arg, 0.999)  # Ensure arctanh argument < 1
        
        return (2 / np.sqrt(c)) * np.arctanh(arg)


def mobius_add(x: np.ndarray, y: np.ndarray, c: float = 1.0, max_norm: float = 0.999) -> np.ndarray:
    """Möbius addition in the Poincaré ball with numerical stability."""
    # Clip inputs to ensure they're within the ball
    x = HyperbolicOperations.clip_norm(x, max_norm)
    y = HyperbolicOperations.clip_norm(y, max_norm)
    
    xy = np.dot(x, y)
    x_norm_sq = c * np.dot(x, x)
    y_norm_sq = c * np.dot(y, y)
    
    denominator = 1 + 2 * c * xy + x_norm_sq * y_norm_sq + 1e-10  # Add epsilon
    
    numerator = (1 + 2 * c * xy + y_norm_sq) * x + (1 - x_norm_sq) * y
    
    result = numerator / denominator
    return HyperbolicOperations.clip_norm(result, max_norm)


def gyro_midpoint(points: List[Tuple[float, np.ndarray]], c: float = 1.0) -> np.ndarray:
    """Weighted gyrovector space midpoint (Einstein midpoint)."""
    if not points:
        return np.zeros(points[0][1].shape[0])
    
    weights, vectors = zip(*points)
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    result = np.zeros_like(vectors[0])
    for w, v in zip(weights, vectors):
        if w > 1e-10:
            gamma_v = 1.0 / np.sqrt(1 - c * np.dot(v, v))
            result += w * gamma_v * v
    
    gamma_sum = np.sqrt(1 + c * np.dot(result, result))
    return result / gamma_sum


class DualSpaceEncoder:
    """
    Encoder producing both Euclidean and Hyperbolic embeddings with field-aware composition.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2',
                 euclidean_dim: int = 768, hyperbolic_dim: int = 64,
                 field_weights: Optional[Dict[str, float]] = None,
                 max_norm: float = 0.999, epsilon: float = 1e-5):
        self.model = SentenceTransformer(model_name)
        self.base_dim = self.model.get_sentence_embedding_dimension()
        self.euclidean_dim = euclidean_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.max_norm = max_norm  # Maximum norm in Poincaré ball
        self.epsilon = epsilon  # Numerical stability epsilon
        
        # Field tokens for prepending
        self.field_tokens = {
            'who': '[WHO]',
            'what': '[WHAT]',
            'when': '[WHEN]',
            'where': '[WHERE]',
            'why': '[WHY]',
            'how': '[HOW]'
        }
        
        # Initialize projection heads (as numpy arrays for simplicity)
        # For all-mpnet-base-v2, base_dim already equals euclidean_dim (768)
        # So euclidean_head is identity, only project for hyperbolic
        if self.base_dim == euclidean_dim:
            self.euclidean_head = np.eye(self.base_dim)  # Identity matrix, no projection needed
        else:
            self.euclidean_head = np.random.randn(self.base_dim, euclidean_dim) * 0.01
        self.hyperbolic_head = np.random.randn(self.base_dim, hyperbolic_dim) * 0.01
        
        # Field weights (learnable gates)
        if field_weights is None:
            field_weights = {
                'who': 1.0,
                'what': 2.0,
                'when': 0.5,
                'where': 0.5,
                'why': 1.5,
                'how': 1.0
            }
        self.field_weights = field_weights
        
        # Curvature for hyperbolic space
        self.c = 1.0
        
        logger.info(f"Initialized DualSpaceEncoder with Euclidean dim={euclidean_dim}, Hyperbolic dim={hyperbolic_dim}")
    
    def encode_field(self, field: str, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a single field with its field token."""
        if not text or not text.strip():
            return np.zeros(self.euclidean_dim), np.zeros(self.hyperbolic_dim)
        
        # Prepend field token
        field_token = self.field_tokens.get(field, '')
        tagged_text = f"{field_token} {text}" if field_token else text
        
        # Get base embedding
        base_embedding = self.model.encode(tagged_text, convert_to_numpy=True)
        
        # Project to Euclidean space and L2 normalize
        euclidean_vec = np.dot(base_embedding, self.euclidean_head)
        euclidean_vec = euclidean_vec / (np.linalg.norm(euclidean_vec) + 1e-8)
        
        # Project to hyperbolic space (tangent space then exp map)
        tangent_vec = np.dot(base_embedding, self.hyperbolic_head)
        # Map from origin to Poincaré ball with stability
        tangent_norm = np.linalg.norm(tangent_vec)
        if tangent_norm > 0:
            # Ensure we stay within the ball with configured max_norm
            scale = np.tanh(tangent_norm) * min(1.0, self.max_norm / tangent_norm)
            hyperbolic_vec = scale * tangent_vec / tangent_norm
            # Apply retraction to ensure we're within bounds
            hyperbolic_vec = HyperbolicOperations.clip_norm(hyperbolic_vec, self.max_norm, self.epsilon)
        else:
            hyperbolic_vec = np.zeros(self.hyperbolic_dim)
        
        return euclidean_vec, hyperbolic_vec
    
    def compose_fields(self, field_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Compose field embeddings using learned weights."""
        if not field_embeddings:
            return np.zeros(self.euclidean_dim), np.zeros(self.hyperbolic_dim)
        
        # Apply softplus to weights for non-negative importance
        weights = {}
        for field in field_embeddings:
            w = self.field_weights.get(field, 1.0)
            weights[field] = np.log(1 + np.exp(w))  # softplus
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Euclidean composition: weighted sum then L2 normalize
        euclidean_composed = np.zeros(self.euclidean_dim)
        for field, (eu_vec, _) in field_embeddings.items():
            euclidean_composed += weights[field] * eu_vec
        
        euclidean_norm = np.linalg.norm(euclidean_composed)
        if euclidean_norm > 0:
            euclidean_composed = euclidean_composed / euclidean_norm
        
        # Hyperbolic composition: weighted gyro-sum
        hyperbolic_points = []
        for field, (_, hy_vec) in field_embeddings.items():
            if weights[field] > 1e-10:
                hyperbolic_points.append((weights[field], hy_vec))
        
        if hyperbolic_points:
            hyperbolic_composed = gyro_midpoint(hyperbolic_points, self.c)
        else:
            hyperbolic_composed = np.zeros(self.hyperbolic_dim)
        
        return euclidean_composed, hyperbolic_composed
    
    def encode(self, five_w1h: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Encode 5W1H fields into dual-space representation.
        Returns dict with 'euclidean_anchor' and 'hyperbolic_anchor'.
        """
        # Encode each field
        field_embeddings = {}
        for field, text in five_w1h.items():
            if text and text.strip():
                field_embeddings[field] = self.encode_field(field, text)
        
        # Compose into final embeddings
        euclidean_anchor, hyperbolic_anchor = self.compose_fields(field_embeddings)
        
        return {
            'euclidean_anchor': euclidean_anchor,
            'hyperbolic_anchor': hyperbolic_anchor,
            'euclidean_residual': np.zeros_like(euclidean_anchor),
            'hyperbolic_residual': np.zeros_like(hyperbolic_anchor)
        }
    
    def compute_product_distance(self, q_embeddings: Dict[str, np.ndarray],
                                m_embeddings: Dict[str, np.ndarray],
                                lambda_e: float = 0.5, lambda_h: float = 0.5) -> float:
        """
        Compute product distance between query and memory.
        Lower distance = higher similarity.
        """
        # Get effective embeddings (anchor + residual)
        q_eu = q_embeddings['euclidean_anchor'] + q_embeddings.get('euclidean_residual', 0)
        q_hy = q_embeddings['hyperbolic_anchor'] + q_embeddings.get('hyperbolic_residual', 0)
        
        m_eu = m_embeddings['euclidean_anchor'] + m_embeddings.get('euclidean_residual', 0)
        m_hy = m_embeddings['hyperbolic_anchor'] + m_embeddings.get('hyperbolic_residual', 0)
        
        # Normalize Euclidean vectors
        q_eu = q_eu / (np.linalg.norm(q_eu) + 1e-8)
        m_eu = m_eu / (np.linalg.norm(m_eu) + 1e-8)
        
        # Euclidean distance (1 - cosine similarity)
        euclidean_dist = 1.0 - np.dot(q_eu, m_eu)
        
        # Hyperbolic distance (geodesic) with stability
        hyperbolic_dist = HyperbolicOperations.geodesic_distance(q_hy, m_hy, self.c, self.max_norm)
        
        # Product distance with query-dependent weights
        return lambda_e * euclidean_dist + lambda_h * hyperbolic_dist
    
    def adapt_residuals(self, embeddings: Dict[str, np.ndarray],
                       partner_embeddings: Dict[str, np.ndarray],
                       relevance: float, momentum: Dict[str, np.ndarray],
                       learning_rate: float = 0.01, momentum_factor: float = 0.9,
                       max_euclidean_norm: float = 0.35,
                       max_hyperbolic_geodesic: float = 0.75) -> Dict[str, np.ndarray]:
        """
        Update residuals based on co-access patterns (gravitational adaptation).
        Returns updated embeddings with bounded residuals.
        """
        # Get effective embeddings
        self_eu = embeddings['euclidean_anchor'] + embeddings['euclidean_residual']
        partner_eu = partner_embeddings['euclidean_anchor'] + partner_embeddings['euclidean_residual']
        
        self_hy = embeddings['hyperbolic_anchor'] + embeddings['hyperbolic_residual']
        partner_hy = partner_embeddings['hyperbolic_anchor'] + partner_embeddings['hyperbolic_residual']
        
        # Euclidean residual update with momentum
        force_eu = relevance * (partner_eu - self_eu)
        momentum['euclidean'] = momentum_factor * momentum.get('euclidean', np.zeros_like(force_eu)) + learning_rate * force_eu
        new_eu_residual = embeddings['euclidean_residual'] + momentum['euclidean']
        
        # Clip Euclidean residual by norm
        eu_norm = np.linalg.norm(new_eu_residual)
        if eu_norm > max_euclidean_norm:
            new_eu_residual = new_eu_residual * (max_euclidean_norm / eu_norm)
        
        # Hyperbolic residual update (in tangent space) with stability
        tangent_update = HyperbolicOperations.log_map(self_hy, partner_hy, self.c, self.max_norm)
        momentum['hyperbolic'] = momentum_factor * momentum.get('hyperbolic', np.zeros_like(tangent_update)) + learning_rate * relevance * tangent_update
        
        # Apply update via exp map with retraction
        updated_hy = HyperbolicOperations.exp_map(self_hy, momentum['hyperbolic'], self.c, self.max_norm)
        new_hy_residual = mobius_add(embeddings['hyperbolic_residual'], 
                                     mobius_add(-embeddings['hyperbolic_anchor'], updated_hy, self.c), 
                                     self.c)
        
        # Clip hyperbolic residual by geodesic distance with stability
        origin = np.zeros_like(new_hy_residual)
        geo_dist = HyperbolicOperations.geodesic_distance(origin, new_hy_residual, self.c, self.max_norm)
        if geo_dist > max_hyperbolic_geodesic:
            scale = max_hyperbolic_geodesic / geo_dist
            new_hy_residual = new_hy_residual * scale
        
        # Return updated embeddings
        updated = embeddings.copy()
        updated['euclidean_residual'] = new_eu_residual
        updated['hyperbolic_residual'] = new_hy_residual
        
        return updated
    
    def calculate_space_activation(self, embeddings: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Calculate the actual space activation based on embedding magnitudes and properties.
        Returns (euclidean_weight, hyperbolic_weight) representing how much each space is activated.
        
        This is based on:
        1. Magnitude of embeddings in each space
        2. Variance/spread of the embeddings
        3. Distance from origin (especially important for hyperbolic)
        """
        euclidean_embedding = embeddings.get('euclidean_anchor')
        hyperbolic_embedding = embeddings.get('hyperbolic_anchor')
        
        # Calculate Euclidean space activation
        # Higher magnitude and variance = more information content
        euclidean_magnitude = np.linalg.norm(euclidean_embedding)
        euclidean_variance = np.var(euclidean_embedding)
        euclidean_activation = euclidean_magnitude * (1 + euclidean_variance)
        
        # Calculate Hyperbolic space activation
        # In hyperbolic space, distance from origin indicates hierarchical depth
        # We use the Poincaré ball model where ||x|| < 1
        hyperbolic_norm = np.linalg.norm(hyperbolic_embedding)
        
        # Hyperbolic distance from origin in Poincaré ball
        # d(0, x) = arctanh(||x||)
        if hyperbolic_norm < 0.999:  # Avoid numerical issues near boundary
            hyperbolic_distance = np.arctanh(hyperbolic_norm)
        else:
            hyperbolic_distance = 3.0  # Cap at reasonable value
            
        # Higher distance from origin = more abstract/hierarchical
        # Also consider variance for information content
        hyperbolic_variance = np.var(hyperbolic_embedding)
        hyperbolic_activation = hyperbolic_distance * (1 + hyperbolic_variance)
        
        # Normalize to get weights (ensure they sum to 1)
        total_activation = euclidean_activation + hyperbolic_activation
        
        if total_activation > 0:
            euclidean_weight = euclidean_activation / total_activation
            hyperbolic_weight = hyperbolic_activation / total_activation
        else:
            # Default to equal weights if no activation
            euclidean_weight = 0.5
            hyperbolic_weight = 0.5
            
        return float(euclidean_weight), float(hyperbolic_weight)
    
    def compute_query_weights(self, query_fields: Dict[str, str]) -> Tuple[float, float]:
        """
        Compute lambda_E and lambda_H based on query field presence.
        More concrete fields (what, where) -> higher Euclidean weight
        More abstract fields (why, how) -> higher Hyperbolic weight
        """
        concrete_score = 0
        abstract_score = 0
        
        for field, text in query_fields.items():
            if text and text.strip():
                if field in ['what', 'where', 'who']:
                    concrete_score += len(text.split())
                elif field in ['why', 'how']:
                    abstract_score += len(text.split())
                elif field == 'when':
                    concrete_score += 0.5 * len(text.split())
        
        # Normalize to get weights
        total = concrete_score + abstract_score
        if total > 0:
            lambda_e = concrete_score / total
            lambda_h = abstract_score / total
        else:
            lambda_e = lambda_h = 0.5
        
        # Apply smoothing to avoid extreme values
        lambda_e = 0.3 + 0.4 * lambda_e
        lambda_h = 0.3 + 0.4 * lambda_h
        
        # Normalize to sum to 1
        total = lambda_e + lambda_h
        return lambda_e / total, lambda_h / total
