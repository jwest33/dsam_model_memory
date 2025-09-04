import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np

from agentic_memory.import_export import MemoryExporter, MemoryImporter
from agentic_memory.types import MemoryRecord, Who, Where, gen_id
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.config import cfg


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Initialize database
    store = MemoryStore(db_path)
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_index():
    """Create a temporary FAISS index for testing."""
    import faiss
    with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as f:
        index_path = f.name
    
    # Create empty FAISS index
    index = faiss.IndexFlatL2(384)
    faiss.write_index(index, index_path)
    
    yield index_path
    
    # Cleanup
    Path(index_path).unlink(missing_ok=True)


@pytest.fixture
def sample_memories():
    """Create sample memory records for testing."""
    memories = []
    
    for i in range(5):
        memory = MemoryRecord(
            memory_id=gen_id("mem"),
            session_id="test_session",
            source_event_id=gen_id("evt"),
            who=Who(type="user", id=f"user_{i}", label=f"Test User {i}"),
            what=f"Test action {i} occurred",
            when=datetime.utcnow(),
            where=Where(type="digital", value="test_location"),
            why=f"Testing purpose {i}",
            how=f"Test method {i}",
            raw_text=f"This is test memory {i}",
            token_count=10 + i,
            embed_text=f"Test embedding text {i}",
            embed_model=cfg.embed_model_name,
            extra={"test_field": f"value_{i}"}
        )
        memories.append(memory)
    
    return memories


def test_export_memories(temp_db, sample_memories):
    """Test exporting memories to JSON format."""
    # Store sample memories
    store = MemoryStore(temp_db)
    # Store each memory with embedding
    for memory in sample_memories:
        embedding = np.random.rand(384).astype(np.float32)
        store.upsert_memory(memory, embedding.tobytes(), 384)
    
    # Export memories
    exporter = MemoryExporter(temp_db)
    export_data = exporter.export_memories()
    
    # Verify export structure
    assert "version" in export_data
    assert export_data["version"] == "1.0"
    assert "export_date" in export_data
    assert "total_memories" in export_data
    assert export_data["total_memories"] == len(sample_memories)
    assert "memories" in export_data
    assert len(export_data["memories"]) == len(sample_memories)
    
    # Verify memory structure
    exported_memory = export_data["memories"][0]
    assert "memory_id" in exported_memory
    assert "session_id" in exported_memory
    assert "who" in exported_memory
    assert "what" in exported_memory
    assert "when" in exported_memory
    assert "where" in exported_memory
    assert "why" in exported_memory
    assert "how" in exported_memory
    assert "raw_text" in exported_memory
    assert "token_count" in exported_memory
    assert "extra" in exported_memory


def test_export_with_filters(temp_db, sample_memories):
    """Test exporting with session filter."""
    # Store memories with different sessions
    store = MemoryStore(temp_db)
    
    # Modify some memories to have different session
    sample_memories[0].session_id = "session_a"
    sample_memories[1].session_id = "session_a"
    sample_memories[2].session_id = "session_b"
    sample_memories[3].session_id = "session_b"
    sample_memories[4].session_id = "session_b"
    
    # Store each memory with embedding
    for memory in sample_memories:
        embedding = np.random.rand(384).astype(np.float32)
        store.upsert_memory(memory, embedding.tobytes(), 384)
    
    # Export only session_a memories
    exporter = MemoryExporter(temp_db)
    export_data = exporter.export_memories(session_id="session_a")
    
    assert export_data["total_memories"] == 2
    assert all(m["session_id"] == "session_a" for m in export_data["memories"])


def test_export_to_file(temp_db, sample_memories):
    """Test exporting memories to a file."""
    # Store sample memories
    store = MemoryStore(temp_db)
    # Store each memory with embedding
    for memory in sample_memories:
        embedding = np.random.rand(384).astype(np.float32)
        store.upsert_memory(memory, embedding.tobytes(), 384)
    
    # Export to file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        export_path = f.name
    
    try:
        exporter = MemoryExporter(temp_db)
        exporter.export_to_file(export_path)
        
        # Verify file contents
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        assert export_data["total_memories"] == len(sample_memories)
        assert len(export_data["memories"]) == len(sample_memories)
    
    finally:
        Path(export_path).unlink(missing_ok=True)


def test_import_memories(temp_db, temp_index, sample_memories):
    """Test importing memories from export data."""
    # Create export data
    export_data = {
        "version": "1.0",
        "export_date": datetime.utcnow().isoformat(),
        "total_memories": len(sample_memories),
        "memories": []
    }
    
    for memory in sample_memories:
        memory_dict = {
            "memory_id": memory.memory_id,
            "session_id": memory.session_id,
            "source_event_id": memory.source_event_id,
            "who": {
                "type": memory.who.type,
                "id": memory.who.id,
                "label": memory.who.label
            },
            "what": memory.what,
            "when": memory.when.isoformat(),
            "where": {
                "type": memory.where.type,
                "value": memory.where.value,
                "lat": memory.where.lat,
                "lon": memory.where.lon
            },
            "why": memory.why,
            "how": memory.how,
            "raw_text": memory.raw_text,
            "token_count": memory.token_count,
            "embed_model": memory.embed_model,
            "extra": memory.extra,
            "created_at": datetime.utcnow().isoformat(),
            "embedding": np.random.rand(384).tolist()  # Mock embedding
        }
        export_data["memories"].append(memory_dict)
    
    # Import memories
    sql_store = MemoryStore(temp_db)
    vector_store = FaissIndex(dim=384, index_path=temp_index)
    importer = MemoryImporter(sql_store, vector_store, cfg.embed_model_name)
    
    result = importer.import_memories(export_data, merge_strategy="skip")
    
    # Verify import results
    assert result["success"] is True
    assert result["imported"] == len(sample_memories)
    assert result["skipped"] == 0
    assert result["errors"] == 0
    
    # Verify memories were stored by trying to fetch them
    memory_ids = [m.memory_id for m in sample_memories]
    stored_memories = sql_store.fetch_memories(memory_ids)
    assert len(stored_memories) == len(sample_memories)


def test_import_validation(temp_db, temp_index):
    """Test import data validation."""
    sql_store = MemoryStore(temp_db)
    vector_store = FaissIndex(dim=384, index_path=temp_index)
    importer = MemoryImporter(sql_store, vector_store)
    
    # Test missing version
    invalid_data = {"memories": []}
    is_valid, errors = importer.validate_import_data(invalid_data)
    assert not is_valid
    assert any("version" in e for e in errors)
    
    # Test invalid version
    invalid_data = {"version": "2.0", "memories": []}
    is_valid, errors = importer.validate_import_data(invalid_data)
    assert not is_valid
    assert any("Unsupported version" in e for e in errors)
    
    # Test missing required fields
    invalid_data = {
        "version": "1.0",
        "memories": [
            {
                "memory_id": "test",
                "session_id": "test",
                # Missing required fields
            }
        ]
    }
    is_valid, errors = importer.validate_import_data(invalid_data)
    assert not is_valid
    assert len(errors) > 0


def test_import_merge_strategies(temp_db, temp_index):
    """Test different merge strategies during import."""
    sql_store = MemoryStore(temp_db)
    vector_store = FaissIndex(dim=384, index_path=temp_index)
    importer = MemoryImporter(sql_store, vector_store, cfg.embed_model_name)
    
    # Create initial memory
    memory1 = MemoryRecord(
        memory_id="mem_001",
        session_id="test",
        source_event_id="evt_001",
        who=Who(type="user", id="user1"),
        what="Original memory",
        when=datetime.utcnow(),
        where=Where(type="digital", value="test"),
        why="Original why",
        how="Original how",
        raw_text="Original text",
        token_count=10,
        embed_text="Original",
        embed_model=cfg.embed_model_name
    )
    
    embedding = np.random.rand(384).astype(np.float32)
    sql_store.upsert_memory(memory1, embedding.tobytes(), 384)
    
    # Create import data with same memory ID
    import_data = {
        "version": "1.0",
        "memories": [{
            "memory_id": "mem_001",
            "session_id": "test",
            "source_event_id": "evt_002",
            "who": {"type": "user", "id": "user2"},
            "what": "Updated memory",
            "when": datetime.utcnow().isoformat(),
            "where": {"type": "digital", "value": "test"},
            "why": "Updated why",
            "how": "Updated how",
            "raw_text": "Updated text",
            "token_count": 15,
            "embedding": np.random.rand(384).tolist()
        }]
    }
    
    # Test skip strategy
    result = importer.import_memories(import_data, merge_strategy="skip")
    assert result["imported"] == 0
    assert result["skipped"] == 1
    
    # Verify original memory unchanged
    stored = sql_store.fetch_memories(["mem_001"])
    assert stored[0]["what"] == "Original memory"
    
    # Test overwrite strategy
    result = importer.import_memories(import_data, merge_strategy="overwrite")
    assert result["imported"] == 1
    assert result["skipped"] == 0
    
    # Verify memory was updated
    stored = sql_store.fetch_memories(["mem_001"])
    assert stored[0]["what"] == "Updated memory"
    
    # Test new_ids strategy
    result = importer.import_memories(import_data, merge_strategy="new_ids")
    assert result["imported"] == 1
    assert result["skipped"] == 0
    
    # Verify new memory was created (can't easily check count without fetch_all method)
    # Just verify import succeeded
    assert result["imported"] == 1


def test_import_from_file(temp_db, temp_index):
    """Test importing from a JSON file."""
    # Create export file
    export_data = {
        "version": "1.0",
        "export_date": datetime.utcnow().isoformat(),
        "total_memories": 1,
        "memories": [{
            "memory_id": "mem_file_001",
            "session_id": "test",
            "source_event_id": "evt_file_001",
            "who": {"type": "user", "id": "file_user"},
            "what": "Memory from file",
            "when": datetime.utcnow().isoformat(),
            "where": {"type": "digital", "value": "file_test"},
            "why": "File import test",
            "how": "Via file",
            "raw_text": "Test memory from file",
            "token_count": 10,
            "embedding": np.random.rand(384).tolist()
        }]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(export_data, f)
        import_path = f.name
    
    try:
        sql_store = MemoryStore(temp_db)
        vector_store = FaissIndex(dim=384, index_path=temp_index)
        importer = MemoryImporter(sql_store, vector_store, cfg.embed_model_name)
        
        result = importer.import_from_file(import_path)
        
        assert result["success"] is True
        assert result["imported"] == 1
        
        # Verify memory was imported
        stored = sql_store.fetch_memories(["mem_file_001"])
        assert len(stored) > 0
        assert stored[0]["what"] == "Memory from file"
    
    finally:
        Path(import_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
