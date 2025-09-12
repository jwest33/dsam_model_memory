"""Shared fixtures and configuration for all tests."""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import sqlite3
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, List, Any, Optional
import uuid
import hashlib

# Set test environment
os.environ['AM_TEST_MODE'] = '1'
os.environ['AM_LLM_BASE_URL'] = 'http://localhost:8000/v1'
os.environ['AM_LLM_MODEL'] = 'test-model'
os.environ['AM_CONTEXT_WINDOW'] = '8192'
os.environ['AM_EMBED_DIM'] = '384'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_db_path(temp_dir):
    """Create a temporary database path."""
    return str(temp_dir / "test.db")


@pytest.fixture
def test_index_path(temp_dir):
    """Create a temporary FAISS index path."""
    return str(temp_dir / "test.faiss")


@pytest.fixture
def sample_event():
    """Create a sample raw event."""
    from agentic_memory.types import RawEvent
    return RawEvent(
        session_id="test_session",
        event_type="user_message",
        actor="user:test",
        content="Hello, this is a test message about Python programming.",
        metadata={"location": "test", "timestamp": datetime.utcnow().isoformat()}
    )


@pytest.fixture
def sample_events():
    """Create multiple sample events for batch testing."""
    from agentic_memory.types import RawEvent
    events = []
    event_types = ["user_message", "llm_message", "tool_call", "tool_result", "system_event"]
    
    for i, event_type in enumerate(event_types):
        events.append(RawEvent(
            session_id=f"test_session_{i}",
            event_type=event_type,
            actor=f"actor:{i}",
            content=f"Test content {i}: This is about {event_type} processing.",
            metadata={"index": i, "test": True}
        ))
    
    return events


@pytest.fixture
def sample_memory_record():
    """Create a sample memory record."""
    from agentic_memory.types import MemoryRecord, Who, Where
    return MemoryRecord(
        session_id="test_session",
        source_event_id="evt_test123",
        who=Who(type="user", id="test", label="Test User"),
        who_list=None,  # Optional field
        what="User sent a test message about Python programming",
        when=datetime.now(timezone.utc),  # Use timezone-aware datetime
        when_list=None,  # Optional field
        where=Where(type="digital", value="test_location"),
        where_list=None,  # Optional field
        why="To test the system functionality",
        how="Via direct message input",
        raw_text="Hello, this is a test message about Python programming.",
        token_count=10,
        embed_text="test message Python programming",
        embed_model="test-model"
    )


@pytest.fixture
def sample_vector():
    """Create a sample embedding vector."""
    vec = np.random.randn(384).astype('float32')
    vec = vec / np.linalg.norm(vec)  # Normalize
    return vec


@pytest.fixture
def sample_vectors():
    """Create multiple sample embedding vectors."""
    vectors = []
    for _ in range(10):
        vec = np.random.randn(384).astype('float32')
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [{
            "message": {
                "content": "This is a test response from the LLM."
            }
        }]
    }


@pytest.fixture
def mock_extraction_response():
    """Mock 5W1H extraction response."""
    return {
        "who": {"type": "user", "id": "test", "label": "Test User"},
        "what": "User sent a test message",
        "when": "2024-01-01T12:00:00",
        "where": {"type": "digital", "value": "test"},
        "why": "Testing the system",
        "how": "Direct input"
    }


@pytest.fixture
def memory_store(test_db_path):
    """Create a test memory store."""
    from agentic_memory.storage.sql_store import MemoryStore
    store = MemoryStore(test_db_path)
    yield store
    # No need to close, uses context manager


@pytest.fixture
def faiss_index(test_index_path):
    """Create a test FAISS index."""
    from agentic_memory.storage.faiss_index import FaissIndex
    index = FaissIndex(dim=384, index_path=test_index_path)
    yield index
    # Cleanup happens automatically


@pytest.fixture
def memory_router(memory_store, faiss_index):
    """Create a test memory router."""
    from agentic_memory.router import MemoryRouter
    return MemoryRouter(memory_store, faiss_index)


@pytest.fixture
def tool_handler():
    """Create a test tool handler."""
    from agentic_memory.tools.tool_handler import ToolHandler
    return ToolHandler()


# ============================================================================
# Mock Fixtures for External Dependencies
# ============================================================================

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for API calls."""
    client = MagicMock()
    
    # Mock embedding response
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=np.random.randn(384).tolist())]
    )
    
    # Mock completion response
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "who": {"type": "user", "id": "test", "label": "Test User"},
                    "what": "Test action performed",
                    "when": datetime.utcnow().isoformat(),
                    "where": {"type": "digital", "value": "test_location"},
                    "why": "Testing purposes",
                    "how": "Via unit test"
                })
            )
        )]
    )
    
    return client


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for HTTP requests."""
    with patch('httpx.Client') as mock_client_class:
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance
        
        # Mock embedding endpoint
        mock_instance.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [{
                    "embedding": np.random.randn(384).tolist()
                }]
            }
        )
        
        yield mock_instance


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing."""
    embedder = MagicMock()
    
    def embed_side_effect(text):
        # Generate deterministic embeddings based on text hash
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        vec = np.random.randn(384).astype('float32')
        return vec / np.linalg.norm(vec)
    
    embedder.embed.side_effect = embed_side_effect
    embedder.embed_batch.side_effect = lambda texts: [embed_side_effect(t) for t in texts]
    
    return embedder


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    from agentic_memory.config_manager import ConfigManager
    
    config = ConfigManager(db_path=":memory:")  # In-memory database for tests
    
    # Override critical settings for testing
    config.set_value("context_window", 4096)
    config.set_value("embed_dim", 384)
    config.set_value("use_attention_retrieval", False)  # Disable for simpler testing
    config.set_value("use_liquid_clustering", False)
    config.set_value("use_multi_part_extraction", True)
    config.set_value("multi_part_threshold", 100)
    
    return config


@pytest.fixture
def test_config_with_attention():
    """Create test configuration with attention enabled."""
    from agentic_memory.config_manager import ConfigManager
    
    config = ConfigManager(db_path=":memory:")
    config.set_value("use_attention_retrieval", True)
    config.set_value("use_liquid_clustering", True)
    
    return config


# ============================================================================
# Database and Storage Fixtures
# ============================================================================

@pytest.fixture
def populated_memory_store(memory_store, sample_memory_record, sample_vector):
    """Create a memory store with pre-populated data."""
    # Add multiple memories
    for i in range(10):
        record = sample_memory_record
        record.memory_id = f"mem_{i:04d}"
        record.what = f"Test action {i}"
        record.raw_text = f"Test content {i} with various keywords"
        
        memory_store.add_memory(record)
        memory_store.add_embedding(record.memory_id, sample_vector)
        
        # Add usage stats
        memory_store.increment_access(record.memory_id)
    
    # Add FTS entries
    memory_store.update_fts_index()
    
    return memory_store


@pytest.fixture
def populated_faiss_index(faiss_index, sample_vectors):
    """Create a FAISS index with pre-populated vectors."""
    memory_ids = [f"mem_{i:04d}" for i in range(len(sample_vectors))]
    
    for mem_id, vec in zip(memory_ids, sample_vectors):
        faiss_index.add(mem_id, vec)
    
    return faiss_index


# ============================================================================
# Query and Retrieval Fixtures
# ============================================================================

@pytest.fixture
def sample_retrieval_query():
    """Create a sample retrieval query."""
    from agentic_memory.types import RetrievalQuery
    
    return RetrievalQuery(
        text="Python programming test",
        k=5,
        session_id="test_session",
        actor_hint="user:test",
        temporal_hint=datetime.utcnow(),
        location_hint="test_location"
    )


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    results = []
    for i in range(5):
        results.append({
            "memory_id": f"mem_{i:04d}",
            "score": 0.9 - (i * 0.1),
            "what": f"Test result {i}",
            "raw_text": f"Test content {i}",
            "token_count": 50 + (i * 10)
        })
    return results


# ============================================================================
# Component Integration Fixtures
# ============================================================================

@pytest.fixture
def mock_memory_router(memory_store, faiss_index, mock_embedder):
    """Create a fully mocked memory router."""
    from agentic_memory.router import MemoryRouter
    
    with patch('agentic_memory.router.get_llama_embedder', return_value=mock_embedder):
        router = MemoryRouter(memory_store, faiss_index)
        router.embedder = mock_embedder
        return router


@pytest.fixture
def hybrid_retriever(memory_store, faiss_index):
    """Create a test hybrid retriever."""
    from agentic_memory.retrieval import HybridRetriever
    return HybridRetriever(memory_store, faiss_index)


@pytest.fixture
def block_builder(memory_store):
    """Create a test block builder."""
    from agentic_memory.block_builder import BlockBuilder
    return BlockBuilder(memory_store)


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
async def async_mock_client():
    """Async mock client for testing async operations."""
    client = AsyncMock()
    
    client.post.return_value = AsyncMock(
        status_code=200,
        json=AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Test async response"
                }
            }]
        })
    )
    
    return client


# ============================================================================
# Benchmark and Performance Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time
            
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def large_dataset():
    """Generate a large dataset for stress testing."""
    from agentic_memory.types import RawEvent
    
    events = []
    for i in range(1000):
        events.append(RawEvent(
            session_id=f"session_{i % 10}",
            event_type="user_message" if i % 2 == 0 else "llm_message",
            actor=f"user:{i % 5}",
            content=f"Large dataset content {i}: " + "test " * 50,
            metadata={"batch": i // 100}
        ))
    
    return events


# ============================================================================
# Cleanup and Teardown Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def isolated_config(temp_dir):
    """Create an isolated configuration for testing."""
    config_db = temp_dir / "test_config.db"
    os.environ['AM_CONFIG_DB'] = str(config_db)
    
    from agentic_memory.config_manager import ConfigManager
    config = ConfigManager(db_path=str(config_db))
    
    yield config
    
    # Cleanup
    if config_db.exists():
        config_db.unlink()


# ============================================================================
# Parametrized Fixtures for Multiple Test Scenarios
# ============================================================================

@pytest.fixture(params=[True, False])
def multi_part_enabled(request):
    """Parametrized fixture for testing with/without multi-part extraction."""
    os.environ['AM_USE_MULTI_PART'] = str(request.param)
    return request.param


@pytest.fixture(params=[100, 500, 1000, 2000])
def token_budget(request):
    """Parametrized fixture for testing different token budgets."""
    return request.param


@pytest.fixture(params=["semantic", "lexical", "hybrid"])
def retrieval_mode(request):
    """Parametrized fixture for testing different retrieval modes."""
    return request.param
