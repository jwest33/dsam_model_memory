from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import json
import pickle

@dataclass
class AttentionWeights:
    """Stores attention weights for different memory aspects"""
    semantic: float
    temporal: float
    usage: float
    actor: float
    spatial: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.semantic, self.temporal, self.usage, self.actor, self.spatial])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AttentionWeights':
        return cls(
            semantic=float(arr[0]),
            temporal=float(arr[1]),
            usage=float(arr[2]),
            actor=float(arr[3]),
            spatial=float(arr[4])
        )

class MemoryAttentionHead(nn.Module):
    """Multi-head attention for memory retrieval with learned relevance"""
    
    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, context_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Learnable query transformations for different aspects
        self.semantic_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_proj = nn.Linear(embed_dim, embed_dim)
        self.usage_proj = nn.Linear(embed_dim, embed_dim)
        self.actor_proj = nn.Linear(embed_dim, embed_dim)
        self.spatial_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multi-head attention for each aspect
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Cross-attention for memory-to-memory relationships
        self.memory_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Learned weighting of different attention aspects
        self.aspect_weights = nn.Parameter(torch.ones(5) / 5)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * 5, embed_dim)
        
    def forward(self, 
                query_embedding: torch.Tensor,
                memory_embeddings: torch.Tensor,
                memory_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embedding: [1, embed_dim] query vector
            memory_embeddings: [n_memories, embed_dim] memory vectors
            memory_metadata: Optional metadata for memories
            
        Returns:
            attention_scores: [n_memories] attention scores for each memory
            mem_relations: [n_memories, n_memories] memory-to-memory relationships
        """
        batch_size = 1
        n_memories = memory_embeddings.shape[0]
        
        # Ensure proper shapes
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if memory_embeddings.dim() == 2:
            memory_embeddings = memory_embeddings.unsqueeze(0)
            
        # Generate specialized query vectors
        q_semantic = self.semantic_proj(query_embedding)
        q_temporal = self.temporal_proj(query_embedding)
        q_usage = self.usage_proj(query_embedding)
        q_actor = self.actor_proj(query_embedding)
        q_spatial = self.spatial_proj(query_embedding)
        
        # Stack queries for parallel attention
        queries = torch.stack([q_semantic, q_temporal, q_usage, q_actor, q_spatial], dim=1)
        queries = queries.view(5, 1, self.embed_dim)
        
        # Compute attention for each aspect
        attention_outputs = []
        for i in range(5):
            q = queries[i:i+1]
            attn_output, attn_weights = self.multihead_attn(q, memory_embeddings, memory_embeddings)
            attention_outputs.append(attn_output)
            
        # Combine attention outputs with learned weights
        attention_stack = torch.cat(attention_outputs, dim=-1)
        weighted_attention = self.output_proj(attention_stack)
        
        # Compute final attention scores
        attention_scores = F.softmax(
            torch.matmul(weighted_attention, memory_embeddings.transpose(-2, -1)).squeeze(0),
            dim=-1
        )
        
        # Memory-to-memory attention for relationship learning
        mem_relations, _ = self.memory_cross_attn(
            memory_embeddings,
            memory_embeddings,
            memory_embeddings
        )
        
        return attention_scores.squeeze(0), mem_relations.squeeze(0)
    
    def compute_attention_weights(self, 
                                   query_embedding: np.ndarray,
                                   memory_embeddings: np.ndarray) -> np.ndarray:
        """Numpy-compatible interface for attention scoring"""
        with torch.no_grad():
            q_tensor = torch.FloatTensor(query_embedding)
            m_tensor = torch.FloatTensor(memory_embeddings)
            
            scores, _ = self.forward(q_tensor, m_tensor)
            return scores.numpy()
    
    def compute_detailed_attention(self,
                                  query_embedding: np.ndarray,
                                  memory_embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute detailed attention scores with per-aspect breakdown.
        
        Returns:
            Dictionary containing:
            - aspect_scores: Dict of aspect name -> array of scores per memory
            - aspect_weights: The learned weights for combining aspects
            - combined_scores: Final attention scores after combining all aspects
            - raw_scores: Scores before softmax normalization
        """
        with torch.no_grad():
            # Convert to tensors
            q_tensor = torch.FloatTensor(query_embedding)
            m_tensor = torch.FloatTensor(memory_embeddings)
            
            # Ensure proper shapes
            if q_tensor.dim() == 1:
                q_tensor = q_tensor.unsqueeze(0)
            if m_tensor.dim() == 2:
                m_tensor = m_tensor.unsqueeze(0)
            
            n_memories = m_tensor.shape[1]
            
            # Generate specialized query vectors for each aspect
            q_semantic = self.semantic_proj(q_tensor)
            q_temporal = self.temporal_proj(q_tensor)
            q_usage = self.usage_proj(q_tensor)
            q_actor = self.actor_proj(q_tensor)
            q_spatial = self.spatial_proj(q_tensor)
            
            # Compute attention for each aspect separately
            aspect_names = ['semantic', 'temporal', 'usage', 'actor', 'spatial']
            aspect_queries = [q_semantic, q_temporal, q_usage, q_actor, q_spatial]
            aspect_scores = {}
            
            for name, q_aspect in zip(aspect_names, aspect_queries):
                # Compute attention scores for this aspect
                # Using simplified dot product attention for interpretability
                scores = torch.matmul(q_aspect, m_tensor.transpose(-2, -1)).squeeze(0)
                scores = scores / np.sqrt(self.embed_dim)  # Scale by sqrt(d)
                # Use sigmoid with temperature scaling for more discriminative scores
                # Temperature of 2.0 spreads the sigmoid response
                temperature = 2.0
                aspect_scores[name] = torch.sigmoid(scores * temperature).squeeze().numpy()
            
            # Compute combined score using weighted average of aspect scores
            # Use learnable aspect weights to combine the scores
            aspect_weights_norm = F.softmax(self.aspect_weights, dim=0).numpy()
            combined_scores = np.zeros(n_memories)
            for i, name in enumerate(aspect_names):
                combined_scores += aspect_weights_norm[i] * aspect_scores[name]
            
            # Get the raw scores before softmax (for debugging)
            attention_outputs = []
            queries = torch.stack(aspect_queries, dim=1).squeeze(0)
            for i in range(5):
                q = queries[i:i+1].unsqueeze(0)
                attn_output, _ = self.multihead_attn(q, m_tensor, m_tensor)
                attention_outputs.append(attn_output)
            
            attention_stack = torch.cat(attention_outputs, dim=-1)
            weighted_attention = self.output_proj(attention_stack)
            raw_scores = torch.matmul(weighted_attention, m_tensor.transpose(-2, -1)).squeeze()
            
            return {
                'aspect_scores': aspect_scores,
                'aspect_weights': self.aspect_weights.numpy(),
                'combined_scores': combined_scores,
                'raw_scores': raw_scores.numpy()
            }

class AdaptiveEmbeddingSpace:
    """Embeddings that evolve based on access patterns"""
    
    def __init__(self, base_dim: int = 1024, meta_dim: int = 64):
        self.base_dim = base_dim
        self.meta_dim = meta_dim
        
        # Learned transformation based on usage patterns
        self.usage_transform = nn.Sequential(
            nn.Linear(base_dim + meta_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )
        
        # Momentum-based embedding updates
        self.embedding_momentum = {}  # memory_id -> running avg
        self.momentum_rate = 0.95
        
        # Co-occurrence matrix for tracking patterns
        self.co_occurrence = {}  # (mem1, mem2) -> count
        
    def encode_with_context(self, 
                            base_embedding: np.ndarray,
                            usage_stats: Dict,
                            co_access_patterns: Optional[Dict] = None) -> np.ndarray:
        """Transform embedding based on usage context"""
        
        # Check if embedding dimension matches expected dimension
        if base_embedding.shape[0] != self.base_dim:
            # Re-initialize the transformation layers with the correct dimensions
            self.base_dim = base_embedding.shape[0]
            self.usage_transform = nn.Sequential(
                nn.Linear(self.base_dim + self.meta_dim, self.base_dim),
                nn.LayerNorm(self.base_dim),
                nn.ReLU(),
                nn.Linear(self.base_dim, self.base_dim)
            )
        
        # Create usage feature vector
        usage_features = np.zeros(self.meta_dim)
        usage_features[0] = usage_stats.get('access_count', 0) / 100.0
        usage_features[1] = usage_stats.get('recency_score', 0.5)
        usage_features[2] = usage_stats.get('diversity_score', 0.0)
        
        if co_access_patterns:
            usage_features[3] = co_access_patterns.get('strength', 0.0)
            
        # Pad or truncate usage features to meta_dim
        if len(usage_features) < self.meta_dim:
            usage_features = np.pad(usage_features, (0, self.meta_dim - len(usage_features)))
            
        # Concatenate and transform
        combined = np.concatenate([base_embedding, usage_features])
        combined_tensor = torch.FloatTensor(combined).unsqueeze(0)
        
        with torch.no_grad():
            contextualized = self.usage_transform(combined_tensor)
            
        return contextualized.squeeze(0).numpy()
    
    def update_embedding_drift(self, memory_id: str, access_context: List[np.ndarray]):
        """Gradually drift embeddings based on access contexts"""
        if not access_context:
            return
            
        context_mean = np.mean(access_context, axis=0)
        
        if memory_id in self.embedding_momentum:
            old_emb = self.embedding_momentum[memory_id]
            new_emb = self.momentum_rate * old_emb + (1 - self.momentum_rate) * context_mean
            self.embedding_momentum[memory_id] = new_emb
        else:
            self.embedding_momentum[memory_id] = context_mean
            
    def update_co_occurrence(self, accessed_memories: List[str]):
        """Update co-occurrence patterns"""
        for i, m1 in enumerate(accessed_memories):
            for m2 in accessed_memories[i+1:]:
                key = tuple(sorted([m1, m2]))
                self.co_occurrence[key] = self.co_occurrence.get(key, 0) + 1
                
    def get_drifted_embedding(self, memory_id: str, base_embedding: np.ndarray) -> np.ndarray:
        """Get embedding with momentum drift applied"""
        if memory_id in self.embedding_momentum:
            drift = self.embedding_momentum[memory_id]
            # Blend base with drift
            return 0.7 * base_embedding + 0.3 * drift
        return base_embedding

class MemoryConsolidation:
    """Hebbian-inspired memory strengthening and pruning"""
    
    def __init__(self):
        self.synapse_weights = {}  # (mem1, mem2) -> weight
        self.memory_importance = {}  # mem_id -> importance score
        self.hebbian_rate = 0.01
        self.decay_rate = 0.001
        self.min_synapse_weight = 0.01
        
    def hebbian_update(self, activated_memories: List[str]):
        """Memories that fire together, wire together"""
        for i, m1 in enumerate(activated_memories):
            for j, m2 in enumerate(activated_memories):
                if i < j:  # Only update once per pair
                    key = tuple(sorted([m1, m2]))
                    current = self.synapse_weights.get(key, 0.0)
                    # Strengthen connection with saturation at 1.0
                    new_weight = min(1.0, current + self.hebbian_rate)
                    self.synapse_weights[key] = new_weight
                    
    def synaptic_decay(self):
        """Gradual weakening of unused connections"""
        keys_to_remove = []
        for key in self.synapse_weights:
            self.synapse_weights[key] *= (1 - self.decay_rate)
            if self.synapse_weights[key] < self.min_synapse_weight:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.synapse_weights[key]
            
    def compute_memory_importance(self, memory_id: str) -> float:
        """Compute importance based on synaptic connections"""
        total_weight = 0.0
        connection_count = 0
        
        for (m1, m2), weight in self.synapse_weights.items():
            if m1 == memory_id or m2 == memory_id:
                total_weight += weight
                connection_count += 1
                
        if connection_count == 0:
            return 0.0
            
        # Average connection strength * sqrt(number of connections)
        return (total_weight / connection_count) * np.sqrt(connection_count)
    
    def get_related_memories(self, memory_id: str, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Get memories strongly connected to the given memory"""
        related = []
        for (m1, m2), weight in self.synapse_weights.items():
            if weight >= threshold:
                if m1 == memory_id:
                    related.append((m2, weight))
                elif m2 == memory_id:
                    related.append((m1, weight))
                    
        return sorted(related, key=lambda x: x[1], reverse=True)
    
    def adaptive_forgetting(self, all_memory_ids: List[str], memory_budget: int) -> Tuple[List[str], List[str]]:
        """Prune least important memories when approaching limits"""
        importances = {}
        for mid in all_memory_ids:
            importances[mid] = self.compute_memory_importance(mid)
            
        # Keep top memories within budget
        sorted_mems = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        keep = [m[0] for m in sorted_mems[:memory_budget]]
        forget = [m[0] for m in sorted_mems[memory_budget:]]
        
        return keep, forget
    
    def save_state(self, path: str):
        """Save consolidation state to disk"""
        state = {
            'synapse_weights': dict(self.synapse_weights),
            'memory_importance': dict(self.memory_importance)
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state(self, path: str):
        """Load consolidation state from disk"""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
                self.synapse_weights = state['synapse_weights']
                self.memory_importance = state['memory_importance']
        except FileNotFoundError:
            pass  # Start with empty state
