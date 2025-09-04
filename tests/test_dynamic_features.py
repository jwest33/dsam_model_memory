"""Test suite for dynamic memory features (attention, liquid clustering, consolidation)"""
import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
import tempfile
import os

from agentic_memory.attention import (
    MemoryAttentionHead, 
    AdaptiveEmbeddingSpace,
    MemoryConsolidation,
    AttentionWeights
)
from agentic_memory.cluster.concept_cluster import LiquidMemoryClusters
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.types import RawEvent, MemoryRecord, Actor, Location

class TestAttentionMechanisms:
    """Test attention-based memory retrieval"""
    
    def test_attention_head_initialization(self):
        """Test that attention head initializes correctly"""
        attention = MemoryAttentionHead(embed_dim=384, num_heads=8)
        assert attention.embed_dim == 384
        assert attention.num_heads == 8
        assert attention.aspect_weights.shape == (5,)
    
    def test_attention_forward_pass(self):
        """Test attention forward pass with dummy data"""
        attention = MemoryAttentionHead(embed_dim=384, num_heads=8)
        
        # Create dummy query and memory embeddings
        query = torch.randn(1, 384)
        memories = torch.randn(10, 384)
        
        # Forward pass
        scores, relations = attention(query, memories)
        
        # Check output shapes
        assert scores.shape == (10,)
        assert relations.shape == (10, 384)
        assert torch.allclose(scores.sum(), torch.tensor(1.0), atol=0.1)
    
    def test_attention_weights_dataclass(self):
        """Test AttentionWeights dataclass"""
        weights = AttentionWeights(
            semantic=0.5,
            temporal=0.2,
            usage=0.15,
            actor=0.1,
            spatial=0.05
        )
        
        arr = weights.to_array()
        assert arr.shape == (5,)
        assert np.isclose(arr.sum(), 1.0)
        
        weights2 = AttentionWeights.from_array(arr)
        assert weights2.semantic == weights.semantic

class TestAdaptiveEmbeddings:
    """Test adaptive embedding space"""
    
    def test_embedding_initialization(self):
        """Test adaptive embedding space initialization"""
        adaptive = AdaptiveEmbeddingSpace(base_dim=384, meta_dim=64)
        assert adaptive.base_dim == 384
        assert adaptive.meta_dim == 64
        assert adaptive.momentum_rate == 0.95
    
    def test_encode_with_context(self):
        """Test context-aware encoding"""
        adaptive = AdaptiveEmbeddingSpace(base_dim=384, meta_dim=64)
        
        base_embedding = np.random.randn(384).astype(np.float32)
        usage_stats = {
            'access_count': 10,
            'recency_score': 0.8,
            'diversity_score': 0.5
        }
        
        contextualized = adaptive.encode_with_context(base_embedding, usage_stats)
        
        assert contextualized.shape == (384,)
        assert not np.array_equal(base_embedding, contextualized)
    
    def test_embedding_drift(self):
        """Test embedding drift updates"""
        adaptive = AdaptiveEmbeddingSpace(base_dim=384)
        
        memory_id = "test_mem_1"
        context_embeddings = [np.random.randn(384) for _ in range(5)]
        
        # Update drift
        adaptive.update_embedding_drift(memory_id, context_embeddings)
        
        assert memory_id in adaptive.embedding_momentum
        
        # Get drifted embedding
        base = np.random.randn(384)
        drifted = adaptive.get_drifted_embedding(memory_id, base)
        
        assert not np.array_equal(base, drifted)
    
    def test_co_occurrence_tracking(self):
        """Test co-occurrence pattern tracking"""
        adaptive = AdaptiveEmbeddingSpace()
        
        memories = ["mem1", "mem2", "mem3"]
        adaptive.update_co_occurrence(memories)
        
        assert ("mem1", "mem2") in adaptive.co_occurrence
        assert ("mem2", "mem3") in adaptive.co_occurrence
        assert adaptive.co_occurrence[("mem1", "mem2")] == 1

class TestMemoryConsolidation:
    """Test Hebbian-inspired memory consolidation"""
    
    def test_consolidation_initialization(self):
        """Test memory consolidation initialization"""
        consolidator = MemoryConsolidation()
        assert consolidator.hebbian_rate == 0.01
        assert consolidator.decay_rate == 0.001
        assert consolidator.min_synapse_weight == 0.01
    
    def test_hebbian_update(self):
        """Test Hebbian learning update"""
        consolidator = MemoryConsolidation()
        
        activated = ["mem1", "mem2", "mem3"]
        consolidator.hebbian_update(activated)
        
        # Check synapses were created
        assert ("mem1", "mem2") in consolidator.synapse_weights
        assert ("mem1", "mem3") in consolidator.synapse_weights
        assert ("mem2", "mem3") in consolidator.synapse_weights
        
        # Check weights increased
        assert consolidator.synapse_weights[("mem1", "mem2")] == 0.01
    
    def test_synaptic_decay(self):
        """Test synaptic weight decay"""
        consolidator = MemoryConsolidation()
        
        # Create some synapses
        consolidator.synapse_weights[("mem1", "mem2")] = 0.5
        consolidator.synapse_weights[("mem3", "mem4")] = 0.005
        
        # Apply decay
        consolidator.synaptic_decay()
        
        # Check decay was applied
        assert consolidator.synapse_weights[("mem1", "mem2")] < 0.5
        # Very weak connection should be removed
        assert ("mem3", "mem4") not in consolidator.synapse_weights
    
    def test_memory_importance(self):
        """Test memory importance computation"""
        consolidator = MemoryConsolidation()
        
        # Create network of connections
        consolidator.synapse_weights[("mem1", "mem2")] = 0.8
        consolidator.synapse_weights[("mem1", "mem3")] = 0.6
        consolidator.synapse_weights[("mem1", "mem4")] = 0.4
        consolidator.synapse_weights[("mem2", "mem3")] = 0.3
        
        importance = consolidator.compute_memory_importance("mem1")
        
        assert importance > 0
        # mem1 should have high importance (many connections)
        assert importance > consolidator.compute_memory_importance("mem4")
    
    def test_adaptive_forgetting(self):
        """Test adaptive forgetting based on importance"""
        consolidator = MemoryConsolidation()
        
        # Create memories with different importance
        consolidator.synapse_weights[("important", "mem1")] = 0.9
        consolidator.synapse_weights[("important", "mem2")] = 0.8
        consolidator.synapse_weights[("unimportant", "mem3")] = 0.1
        
        all_memories = ["important", "mem1", "mem2", "unimportant", "mem3"]
        keep, forget = consolidator.adaptive_forgetting(all_memories, memory_budget=3)
        
        assert len(keep) == 3
        assert len(forget) == 2
        assert "important" in keep

class TestLiquidClustering:
    """Test liquid memory clustering"""
    
    def test_liquid_cluster_initialization(self):
        """Test liquid cluster initialization"""
        clusters = LiquidMemoryClusters(n_clusters=32, dim=384)
        assert clusters.flow_rate == 0.1
        assert clusters.merge_threshold == 0.85
        assert len(clusters.memory_clusters) == 0
    
    def test_affinity_matrix_computation(self):
        """Test affinity matrix computation"""
        clusters = LiquidMemoryClusters()
        
        memory_ids = ["mem1", "mem2", "mem3"]
        embeddings = {
            "mem1": np.random.randn(384),
            "mem2": np.random.randn(384),
            "mem3": np.random.randn(384)
        }
        
        affinity = clusters.compute_affinity_matrix(memory_ids, embeddings)
        
        assert affinity.shape == (3, 3)
        assert np.allclose(affinity, affinity.T)  # Should be symmetric
        assert np.all(affinity >= 0) and np.all(affinity <= 1)
    
    def test_flow_step(self):
        """Test cluster flow dynamics"""
        clusters = LiquidMemoryClusters(n_clusters=4, dim=128)
        
        memory_ids = ["mem1", "mem2", "mem3", "mem4"]
        embeddings = {
            mid: np.random.randn(128) for mid in memory_ids
        }
        
        # Set up initial co-access to create affinity
        clusters.co_access_counts[("mem1", "mem2")] = 10
        clusters.co_access_counts[("mem3", "mem4")] = 10
        
        # Perform flow step
        new_clusters = clusters.flow_step(memory_ids, embeddings)
        
        assert len(new_clusters) == 4
        assert all(mid in clusters.memory_clusters for mid in memory_ids)
    
    def test_cluster_energy_update(self):
        """Test cluster energy dynamics"""
        clusters = LiquidMemoryClusters()
        
        # Set up some clusters
        clusters.memory_clusters = {"mem1": 0, "mem2": 0, "mem3": 1}
        clusters.cluster_members = {0: {"mem1", "mem2"}, 1: {"mem3"}}
        
        # Update energy based on access
        clusters.update_cluster_energy(["mem1", "mem3"])
        
        assert 0 in clusters.cluster_energy
        assert 1 in clusters.cluster_energy
        assert clusters.cluster_energy[0] > 0.5
    
    def test_cluster_merging(self):
        """Test similar cluster merging"""
        clusters = LiquidMemoryClusters()
        clusters.merge_threshold = 0.8
        
        # Create very similar embeddings for two clusters
        emb1 = np.random.randn(384)
        emb2 = emb1 + np.random.randn(384) * 0.01  # Very similar
        
        embeddings = {
            "mem1": emb1,
            "mem2": emb1,
            "mem3": emb2,
            "mem4": emb2
        }
        
        # Set up clusters
        clusters.memory_clusters = {"mem1": 0, "mem2": 0, "mem3": 1, "mem4": 1}
        clusters.cluster_members = {
            0: {"mem1", "mem2"},
            1: {"mem3", "mem4"}
        }
        clusters.cluster_energy = {0: 0.6, 1: 0.4}
        
        # Merge similar clusters
        clusters.merge_similar_clusters(embeddings)
        
        # Should have merged into one cluster
        assert len(clusters.cluster_members) == 1

class TestIntegration:
    """Test integration of dynamic features"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_synapse_storage(self, temp_db):
        """Test synapse weight storage in database"""
        store = MemoryStore(temp_db)
        
        # Update synapse
        store.update_synapse("mem1", "mem2", 0.5)
        
        # Retrieve synapses
        synapses = store.get_synapses("mem1")
        
        assert len(synapses) == 1
        assert synapses[0] == ("mem2", 0.5)
    
    def test_importance_storage(self, temp_db):
        """Test importance score storage"""
        store = MemoryStore(temp_db)
        
        # Update importance
        store.update_importance("mem1", 0.75, 5)
        
        # Retrieve importance
        scores = store.get_importance_scores(["mem1"])
        
        assert "mem1" in scores
        assert scores["mem1"] == 0.75
    
    def test_embedding_drift_storage(self, temp_db):
        """Test embedding drift storage"""
        store = MemoryStore(temp_db)
        
        # Store drift
        drift = np.random.randn(384).astype(np.float32)
        store.store_embedding_drift("mem1", drift)
        
        # Retrieve drift
        retrieved = store.get_embedding_drift("mem1")
        
        assert retrieved is not None
        assert np.allclose(drift, retrieved)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
