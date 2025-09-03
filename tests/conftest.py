"""Shared fixtures and configuration for all tests."""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
from datetime import datetime

# Set test environment
os.environ['AM_TEST_MODE'] = '1'
os.environ['AM_LLM_BASE_URL'] = 'http://localhost:8000/v1'
os.environ['AM_LLM_MODEL'] = 'test-model'


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
        metadata={"location": "test"}
    )


@pytest.fixture
def sample_memory_record():
    """Create a sample memory record."""
    from agentic_memory.types import MemoryRecord, Who, Where
    return MemoryRecord(
        session_id="test_session",
        source_event_id="evt_test123",
        who=Who(type="user", id="test", label="Test User"),
        what="User sent a test message about Python programming",
        when=datetime.utcnow(),
        where=Where(type="digital", value="test_location"),
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
