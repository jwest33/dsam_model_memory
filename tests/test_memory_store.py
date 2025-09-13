"""Tests for MemoryStore component."""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.types import MemoryRecord, Who, Where


class TestMemoryStore:
    """Test suite for MemoryStore."""
    
    def test_initialization(self, test_db_path):
        """Test memory store initialization."""
        store = MemoryStore(test_db_path)
        assert store is not None
        assert store.db_path == test_db_path
        
    def test_upsert_memory(self, memory_store, sample_memory_record, sample_vector):
        """Test upserting a memory record."""
        # Serialize the embedding
        embedding_bytes = sample_vector.tobytes()
        
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Verify memory was added
        with memory_store.connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE memory_id = ?",
                (sample_memory_record.memory_id,)
            )
            count = cursor.fetchone()[0]
            assert count == 1
    
    def test_upsert_duplicate_memory(self, memory_store, sample_memory_record, sample_vector):
        """Test that duplicate memories are handled properly via upsert."""
        embedding_bytes = sample_vector.tobytes()
        
        # Insert first time
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Update the record
        sample_memory_record.what = "Updated action"
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Check that there's still only one record
        with memory_store.connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE memory_id = ?",
                (sample_memory_record.memory_id,)
            )
            count = cursor.fetchone()[0]
            assert count == 1
            
            # Check that it was updated
            cursor = conn.execute(
                "SELECT what FROM memories WHERE memory_id = ?",
                (sample_memory_record.memory_id,)
            )
            what = cursor.fetchone()[0]
            assert what == "Updated action"
    
    def test_fetch_memories(self, memory_store, sample_memory_record, sample_vector):
        """Test fetching memories by IDs."""
        embedding_bytes = sample_vector.tobytes()
        
        # Add multiple memories
        memory_ids = []
        for i in range(3):
            record = sample_memory_record
            record.memory_id = f"mem_{i:04d}"
            record.what = f"Test action {i}"
            memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
            memory_ids.append(record.memory_id)
        
        # Fetch memories
        memories = memory_store.fetch_memories(memory_ids)
        
        assert len(memories) == 3
        memory_ids_fetched = [m['memory_id'] for m in memories]
        for mem_id in memory_ids:
            assert mem_id in memory_ids_fetched
    
    # Lexical search test removed - FTS5 was broken and removed from application
    
    def test_record_access(self, memory_store, sample_memory_record, sample_vector):
        """Test recording memory access."""
        embedding_bytes = sample_vector.tobytes()
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Record access
        memory_store.record_access([sample_memory_record.memory_id])
        
        # Check usage stats
        stats = memory_store.get_usage_stats([sample_memory_record.memory_id])
        assert sample_memory_record.memory_id in stats
        assert stats[sample_memory_record.memory_id]['accesses'] == 1
    
    def test_get_by_actor(self, memory_store, sample_vector):
        """Test retrieving memories by actor."""
        from agentic_memory.types import MemoryRecord, Who, Where
        from datetime import datetime, timezone
        
        embedding_bytes = sample_vector.tobytes()
        
        # Add memories with specific actor - create fresh records each time
        for i in range(3):
            record = MemoryRecord(
                memory_id=f"mem_{i:04d}",
                session_id="test_session",
                source_event_id=f"evt_{i}",
                who=Who(type="user", id="test_user", label="Test User"),
                what=f"Action {i}",
                when=datetime.now(timezone.utc),
                where=Where(type="digital", value="test"),
                why="Testing",
                how="Test",
                raw_text=f"Content {i}",
                token_count=10,
                embed_text="test",
                embed_model="test-model"
            )
            memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
        
        # Get by actor - who_id is stored without the type prefix
        memories = memory_store.get_by_actor("test_user", limit=5)
        assert len(memories) == 3
    
    def test_get_by_location(self, memory_store, sample_memory_record, sample_vector):
        """Test retrieving memories by location."""
        embedding_bytes = sample_vector.tobytes()
        
        # Add memory with specific location
        sample_memory_record.where = Where(type="digital", value="test_location")
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Get by location
        memories = memory_store.get_by_location("test_location", limit=5)
        assert len(memories) >= 1
        assert any(m['where_value'] == "test_location" for m in memories)
    
    def test_actor_exists(self, memory_store, sample_memory_record, sample_vector):
        """Test checking if actor exists."""
        embedding_bytes = sample_vector.tobytes()
        
        # Initially actor doesn't exist - who_id is stored without type prefix
        assert not memory_store.actor_exists("test")
        
        # Add memory with actor
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Now actor should exist
        assert memory_store.actor_exists("test")
    
    def test_connection_management(self, memory_store):
        """Test database connection."""
        # Should be able to execute queries using context manager
        with memory_store.connect() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_schema_creation(self, test_db_path):
        """Test that all required tables are created."""
        store = MemoryStore(test_db_path)
        
        expected_tables = [
            'memories', 'embeddings', 'usage_stats',
            'clusters', 'cluster_membership', 'blocks', 'block_members',
            'memory_synapses', 'embedding_drift', 'memory_importance'
        ]
        
        with store.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in expected_tables:
                assert table in tables, f"Table {table} not found"


class TestMemoryStoreAdvanced:
    """Advanced tests for MemoryStore."""
    
    def test_batch_upsert_memories(self, memory_store, sample_vector):
        """Test upserting multiple memories in batch."""
        embedding_bytes = sample_vector.tobytes()
        
        for i in range(10):
            record = MemoryRecord(
                session_id=f"session_{i}",
                source_event_id=f"evt_{i}",
                who=Who(type="user", id=f"user_{i}", label=f"User {i}"),
                what=f"Action {i}",
                when=datetime.now(timezone.utc) + timedelta(seconds=i),
                where=Where(type="digital", value=f"location_{i}"),
                why=f"Reason {i}",
                how=f"Method {i}",
                raw_text=f"Content {i}",
                token_count=10 + i,
                embed_text=f"embed {i}",
                embed_model="test-model"
            )
            memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
        
        # Verify all were added
        with memory_store.connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            assert count == 10
    
    def test_memory_with_location_coords(self, memory_store, sample_vector):
        """Test storing memories with geographic coordinates."""
        embedding_bytes = sample_vector.tobytes()
        
        record = MemoryRecord(
            session_id="geo_session",
            source_event_id="evt_geo",
            who=Who(type="user", id="geo_user", label="Geo User"),
            what="Location-based event",
            when=datetime.now(timezone.utc),
            where=Where(
                type="physical",
                value="New York",
                lat=40.7128,
                lon=-74.0060
            ),
            why="Testing geo storage",
            how="GPS",
            raw_text="Geo test",
            token_count=10,
            embed_text="geo",
            embed_model="test-model"
        )
        
        memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
        
        # Verify coordinates were stored
        with memory_store.connect() as conn:
            cursor = conn.execute(
                "SELECT where_lat, where_lon FROM memories WHERE memory_id = ?",
                (record.memory_id,)
            )
            result = cursor.fetchone()
            assert result[0] == pytest.approx(40.7128, rel=1e-4)
            assert result[1] == pytest.approx(-74.0060, rel=1e-4)
    
    def test_temporal_retrieval(self, memory_store, sample_vector):
        """Test retrieving memories by date."""
        embedding_bytes = sample_vector.tobytes()
        
        # Add memories with specific dates
        now = datetime.now(timezone.utc)
        for i in range(5):
            record = MemoryRecord(
                session_id="temporal_test",
                source_event_id=f"evt_{i}",
                who=Who(type="user", id="test", label="Test"),
                what=f"Event {i}",
                when=now + timedelta(days=i),
                where=Where(type="digital", value="test"),
                why="Testing",
                how="Test",
                raw_text=f"Content {i}",
                token_count=10,
                embed_text="test",
                embed_model="test-model"
            )
            memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
        
        # Get memories from today
        today = now.strftime("%Y-%m-%d")
        memories = memory_store.get_by_date(today)
        assert len(memories) >= 1
    
    def test_synaptic_connections(self, memory_store, sample_memory_record, sample_vector):
        """Test synaptic connections between memories."""
        embedding_bytes = sample_vector.tobytes()
        
        # Create two memories
        memory1 = sample_memory_record
        memory1.memory_id = "mem_001"
        memory_store.upsert_memory(memory1, embedding_bytes, len(sample_vector))
        
        memory2 = sample_memory_record
        memory2.memory_id = "mem_002"
        memory_store.upsert_memory(memory2, embedding_bytes, len(sample_vector))
        
        # Update synapse between them
        memory_store.update_synapse("mem_001", "mem_002", 0.5)
        
        # Get synapses
        synapses = memory_store.get_synapses("mem_001")
        assert len(synapses) > 0
        assert any(s[0] == "mem_002" for s in synapses)
    
    def test_importance_scores(self, memory_store, sample_memory_record, sample_vector):
        """Test importance score tracking."""
        embedding_bytes = sample_vector.tobytes()
        
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Update importance
        memory_store.update_importance(sample_memory_record.memory_id, 0.8, 5)
        
        # Get importance scores
        scores = memory_store.get_importance_scores([sample_memory_record.memory_id])
        assert sample_memory_record.memory_id in scores
        assert scores[sample_memory_record.memory_id] == pytest.approx(0.8, rel=1e-2)
    
    def test_embedding_drift(self, memory_store, sample_memory_record, sample_vector):
        """Test embedding drift storage and retrieval."""
        embedding_bytes = sample_vector.tobytes()
        
        memory_store.upsert_memory(sample_memory_record, embedding_bytes, len(sample_vector))
        
        # Store drift vector
        drift_vector = np.random.randn(len(sample_vector)).astype('float32')
        memory_store.store_embedding_drift(sample_memory_record.memory_id, drift_vector)
        
        # Retrieve drift vector
        retrieved_drift = memory_store.get_embedding_drift(sample_memory_record.memory_id)
        assert retrieved_drift is not None
        np.testing.assert_array_almost_equal(retrieved_drift, drift_vector)
    
    def test_block_creation(self, memory_store, sample_memory_record, sample_vector):
        """Test block creation and retrieval."""
        embedding_bytes = sample_vector.tobytes()
        
        # Add memories
        memory_ids = []
        for i in range(3):
            record = sample_memory_record
            record.memory_id = f"mem_{i:04d}"
            memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
            memory_ids.append(record.memory_id)
        
        # Create block
        block = {
            'block_id': 'test_block_001',
            'query_fingerprint': 'test_query',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'budget_tokens': 1000,
            'used_tokens': 300,
            'has_more': False,
            'summary_text': 'Test block summary'
        }
        memory_store.create_block(block, memory_ids)
        
        # Retrieve block
        retrieved_block = memory_store.get_block('test_block_001')
        assert retrieved_block.get('block', {}).get('block_id') == 'test_block_001'
        assert retrieved_block.get('block', {}).get('used_tokens') == 300
        assert len(retrieved_block.get('members', [])) == 3
    
    @pytest.mark.parametrize("num_memories", [10, 50, 100])
    def test_performance_batch_operations(self, memory_store, sample_vector, performance_timer, num_memories):
        """Test performance with batch operations."""
        embedding_bytes = sample_vector.tobytes()
        
        performance_timer.start()
        
        # Batch insert
        memory_ids = []
        for i in range(num_memories):
            record = MemoryRecord(
                memory_id=f"perf_{i:04d}",
                session_id="perf_test",
                source_event_id=f"evt_{i}",
                who=Who(type="user", id="perf", label="Perf"),
                what=f"Performance test {i}",
                when=datetime.now(timezone.utc),
                where=Where(type="digital", value="test"),
                why="Performance testing",
                how="Automated",
                raw_text=f"Content {i} with keywords for search",
                token_count=50,
                embed_text=f"embed {i}",
                embed_model="test-model"
            )
            memory_store.upsert_memory(record, embedding_bytes, len(sample_vector))
            memory_ids.append(record.memory_id)
        
        elapsed = performance_timer.stop()
        
        # Check reasonable performance (adjust threshold as needed)
        assert elapsed < num_memories * 0.02  # Less than 20ms per memory
        
        # Test retrieval performance (no longer testing lexical search)
        performance_timer.start()
        memories = memory_store.fetch_memories(memory_ids[:min(10, num_memories)])
        fetch_time = performance_timer.stop()
        
        assert fetch_time < 0.5  # Fetch should be under 500ms
        assert len(memories) == min(10, num_memories)