"""
Tests for the benchmark system.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from benchmarks import (
    MemoryBenchmark,
    InteractionGenerator,
    InteractionScenario,
    BenchmarkResult
)
from agentic_memory.router import MemoryRouter


@pytest.mark.asyncio
async def test_interaction_generator():
    """Test the interaction generator."""
    generator = InteractionGenerator()
    
    # Test single interaction generation
    scenario = await generator.generate_interaction(
        topic="technology",
        interaction_type="conversation",
        actor="user",
        complexity="simple"
    )
    
    assert scenario.topic == "technology"
    assert scenario.interaction_type == "conversation"
    assert scenario.actor == "user"
    assert scenario.complexity == "simple"
    assert scenario.content  # Should have generated content
    assert scenario.timestamp
    assert scenario.metadata.get("generated") == True


@pytest.mark.asyncio
async def test_batch_generation():
    """Test batch generation of interactions."""
    generator = InteractionGenerator()
    
    # Generate a small batch
    scenarios = await generator.generate_batch(
        count=10,
        time_range_days=7,
        scenario_mix={"simple": 0.5, "medium": 0.3, "complex": 0.2}
    )
    
    assert len(scenarios) == 10
    
    # Check that timestamps are spread over the time range
    timestamps = [s.timestamp for s in scenarios]
    assert min(timestamps) < max(timestamps)
    
    # Check that scenarios are sorted by timestamp
    for i in range(1, len(scenarios)):
        assert scenarios[i-1].timestamp <= scenarios[i].timestamp
    
    # Check variety in topics and actors
    topics = set(s.topic for s in scenarios)
    actors = set(s.actor for s in scenarios)
    assert len(topics) > 1
    assert len(actors) > 1


@pytest.mark.asyncio
async def test_memory_benchmark():
    """Test the memory benchmark system."""
    # Create a temporary database for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        index_path = Path(tmpdir) / "test.index"
        
        # Create router with test paths
        router = MemoryRouter(db_path=str(db_path), index_path=str(index_path))
        
        # Create benchmark with test router
        benchmark = MemoryBenchmark(memory_router=router)
        
        # Run a small benchmark
        result = await benchmark.run_benchmark(
            interaction_count=5,
            time_range_days=3,
            retrieval_queries=2,
            scenario_mix={"simple": 0.8, "medium": 0.2, "complex": 0.0}
        )
        
        # Verify result structure
        assert isinstance(result, BenchmarkResult)
        assert result.total_interactions == 5
        assert result.total_time_seconds > 0
        assert result.memory_count >= 5  # Should have at least 5 memories
        assert result.avg_ingestion_time >= 0
        assert result.avg_retrieval_time >= 0
        
        # Check that memories were actually stored
        memories = router.sql_store.get_all_memories(limit=10)
        assert len(memories) >= 5


@pytest.mark.asyncio
async def test_benchmark_metrics():
    """Test that benchmark collects proper metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        index_path = Path(tmpdir) / "test.index"
        
        router = MemoryRouter(db_path=str(db_path), index_path=str(index_path))
        benchmark = MemoryBenchmark(memory_router=router)
        
        # Run benchmark
        result = await benchmark.run_benchmark(
            interaction_count=3,
            time_range_days=1,
            retrieval_queries=1
        )
        
        # Check metrics were collected
        assert len(benchmark.metrics["ingestion_times"]) == 3
        assert len(benchmark.metrics["retrieval_times"]) >= 1
        assert all(t > 0 for t in benchmark.metrics["ingestion_times"])
        
        # Check result calculations
        assert result.avg_ingestion_time == sum(benchmark.metrics["ingestion_times"]) / len(benchmark.metrics["ingestion_times"])


def test_scenario_mix_parsing():
    """Test parsing of scenario mix from CLI."""
    from benchmarks.cli import parse_scenario_mix
    
    # Test valid mix
    mix = parse_scenario_mix("simple:0.3,medium:0.5,complex:0.2")
    assert mix == {"simple": 0.3, "medium": 0.5, "complex": 0.2}
    
    # Test normalization
    mix = parse_scenario_mix("simple:1,medium:2,complex:1")
    assert abs(sum(mix.values()) - 1.0) < 0.01  # Should sum to 1
    assert mix["medium"] == 0.5  # Should be normalized
    
    # Test empty string
    mix = parse_scenario_mix("")
    assert mix is None


def test_benchmark_result_serialization():
    """Test that benchmark results can be serialized to JSON."""
    result = BenchmarkResult(
        total_interactions=100,
        total_time_seconds=45.5,
        avg_ingestion_time=0.05,
        avg_retrieval_time=0.02,
        memory_count=100,
        index_size=100,
        db_size_bytes=1024000,
        errors=["test error"],
        metadata={"test": "data"}
    )
    
    # Convert to dict (as done in save_results)
    from dataclasses import asdict
    result_dict = asdict(result)
    
    # Should be JSON serializable
    json_str = json.dumps(result_dict)
    assert json_str
    
    # Can be loaded back
    loaded = json.loads(json_str)
    assert loaded["total_interactions"] == 100
    assert loaded["metadata"]["test"] == "data"


@pytest.mark.asyncio
async def test_fallback_generation():
    """Test fallback generation when LLM is unavailable."""
    generator = InteractionGenerator()
    
    # Test fallback directly
    content = generator._fallback_generation("Topic: testing\nType: task")
    
    assert content  # Should generate something
    assert len(content) > 10  # Should be meaningful
    assert "testing" in content or "various subjects" in content


@pytest.mark.asyncio
async def test_benchmark_with_progress_callback():
    """Test benchmark with progress callback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        index_path = Path(tmpdir) / "test.index"
        
        router = MemoryRouter(db_path=str(db_path), index_path=str(index_path))
        benchmark = MemoryBenchmark(memory_router=router)
        
        # Track progress
        progress_scenarios = []
        
        def progress_callback(scenario):
            progress_scenarios.append(scenario)
        
        # Run with callback
        result = await benchmark.run_benchmark(
            interaction_count=3,
            time_range_days=1,
            retrieval_queries=1,
            progress_callback=progress_callback
        )
        
        # Verify callback was called for each scenario
        assert len(progress_scenarios) == 3
        assert all(isinstance(s, InteractionScenario) for s in progress_scenarios)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
