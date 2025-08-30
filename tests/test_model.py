"""
Test script to validate the implemented enhancements
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_config():
    """Test configuration enhancements"""
    print("Testing configuration...")
    from config import get_config
    
    config = get_config()
    
    # Test DualSpaceConfig
    assert hasattr(config, 'dual_space'), "DualSpaceConfig missing"
    assert config.dual_space.euclidean_dim == 768, f"Euclidean dim: {config.dual_space.euclidean_dim}"
    assert config.dual_space.hyperbolic_dim == 64, f"Hyperbolic dim: {config.dual_space.hyperbolic_dim}"
    assert config.dual_space.max_norm == 0.999, f"Max norm: {config.dual_space.max_norm}"
    assert config.dual_space.epsilon == 1e-5, f"Epsilon: {config.dual_space.epsilon}"
    assert config.dual_space.use_relative_bounds == True, "Relative bounds not enabled"
    assert config.dual_space.hdbscan_min_cluster_size == 5, "HDBSCAN min cluster size incorrect"
    assert config.dual_space.field_adaptation_limits is not None, "Field adaptation limits missing"
    
    print("  Configuration tests passed")

def test_hyperbolic_stability():
    """Test hyperbolic numerical stability"""
    print("\nTesting hyperbolic numerical stability...")
    from memory.dual_space_encoder import HyperbolicOperations
    
    # Test norm clipping
    x = np.array([0.5, 0.5, 0.5])
    clipped = HyperbolicOperations.clip_norm(x, max_norm=0.999)
    assert np.linalg.norm(clipped) <= 0.999, "Norm clipping failed"
    
    # Test with extreme values
    x_extreme = np.array([10.0, 10.0, 10.0])
    clipped_extreme = HyperbolicOperations.clip_norm(x_extreme, max_norm=0.999)
    assert np.linalg.norm(clipped_extreme) < 0.999, f"Extreme norm clipping failed: {np.linalg.norm(clipped_extreme)}"
    
    # Test safe sqrt
    result = HyperbolicOperations.safe_sqrt(0.0)
    assert result > 0, "Safe sqrt failed"
    
    # Test geodesic distance with near-boundary points
    x = np.array([0.99, 0, 0])
    y = np.array([0, 0.99, 0])
    dist = HyperbolicOperations.geodesic_distance(x, y, c=1.0, max_norm=0.999)
    assert not np.isnan(dist) and not np.isinf(dist), f"Geodesic distance unstable: {dist}"
    
    print("  Hyperbolic stability tests passed")

def test_scale_aware_bounds():
    """Test scale-aware residual bounds"""
    print("\nTesting scale-aware bounds...")
    from memory.memory_store import MemoryStore
    from models.event import Event, FiveW1H, EventType
    
    store = MemoryStore()
    
    # Test scale-aware bound calculation
    test_embeddings = {
        'euclidean_anchor': np.random.randn(store.encoder.euclidean_dim),
        'hyperbolic_anchor': np.random.randn(store.encoder.hyperbolic_dim) * 0.5
    }
    
    eu_bound = store._get_scale_aware_bound(test_embeddings, 'euclidean')
    hy_bound = store._get_scale_aware_bound(test_embeddings, 'hyperbolic')
    
    assert eu_bound > 0, "Euclidean bound invalid"
    assert hy_bound > 0, "Hyperbolic bound invalid"
    
    print(f"  Euclidean bound: {eu_bound:.4f}")
    print(f"  Hyperbolic bound: {hy_bound:.4f}")
    
    print("  Scale-aware bounds tests passed")

def test_field_adaptation():
    """Test field-level adaptation limits"""
    print("\nTesting field adaptation limits...")
    from memory.memory_store import MemoryStore
    from models.event import Event, FiveW1H, EventType
    
    store = MemoryStore()
    
    # Create test events
    event1 = Event(
        five_w1h=FiveW1H(
            who="Alice",
            what="testing system",
            when="2024-01-01",
            where="office",
            why="validation",
            how="manually"
        ),
        event_type=EventType.ACTION
    )
    
    event2 = Event(
        five_w1h=FiveW1H(
            who="Bobby",  # Changed to > 3 chars
            what="reviewing code",
            when="2024-01-02",
            where="remote",
            why="quality check",
            how="automated"
        ),
        event_type=EventType.ACTION
    )
    
    # Test field limit application
    relevance = 1.0
    limited_relevance = store._apply_field_limits(event1, event2, relevance)
    
    # Should be limited due to dominant fields
    assert limited_relevance <= relevance, f"Field limits not applied: {limited_relevance}"
    
    # All fields have content > 3 chars, so all are dominant
    # The minimum limit is 'who' at 0.2
    assert abs(limited_relevance - 0.2) < 0.01, f"Incorrect limit applied: {limited_relevance} vs expected 0.2"
    
    print(f"  Original relevance: {relevance}")
    print(f"  Limited relevance: {limited_relevance}")
    
    print("Field adaptation tests passed")

def test_provenance_tracking():
    """Test provenance and versioning"""
    print("\nTesting provenance tracking...")
    from memory.chromadb_store import ChromaDBStore
    from models.event import Event, FiveW1H, EventType
    import numpy as np
    
    store = ChromaDBStore()
    
    # Create and store a test event
    event = Event(
        five_w1h=FiveW1H(
            who="Test User",
            what="testing provenance",
            when="2024-01-01",
            where="lab",
            why="validation",
            how="automated"
        ),
        event_type=EventType.ACTION
    )
    
    embedding = np.random.randn(768)
    success = store.store_event(event, embedding)
    assert success, "Failed to store event"
    
    # Update provenance
    provenance_data = {
        'residual_norm_euclidean': 0.25,
        'residual_norm_hyperbolic': 0.45,
        'access_count': 5,
        'co_retrieval_partners': ['event1', 'event2'],
        'lambda_e': 0.6,
        'lambda_h': 0.4
    }
    
    success = store.update_provenance(event.id, provenance_data)
    assert success, "Failed to update provenance"
    
    # Retrieve provenance
    provenance = store.get_provenance(event.id)
    assert provenance is not None, "Failed to retrieve provenance"
    assert provenance['residual_norm_euclidean'] == 0.25, "Euclidean norm mismatch"
    assert provenance['residual_norm_hyperbolic'] == 0.45, "Hyperbolic norm mismatch"
    assert provenance['access_count'] == 5, "Access count mismatch"
    assert len(provenance['co_retrieval_partners']) == 2, "Co-retrieval partners mismatch"
    
    print(f"  Provenance version: {provenance['version']}")
    print(f"  Access count: {provenance['access_count']}")
    print(f"  Residual norms: E={provenance['residual_norm_euclidean']:.3f}, H={provenance['residual_norm_hyperbolic']:.3f}")
    
    print("Provenance tracking tests passed")

def test_forgetting():
    """Test forgetting mechanism"""
    print("\nTesting forgetting mechanism...")
    from memory.memory_store import MemoryStore
    
    store = MemoryStore()
    
    # Add some test residuals
    test_id = "test_event_123"
    store.residuals[test_id] = {
        'euclidean': np.ones(store.encoder.euclidean_dim) * 0.5,
        'hyperbolic': np.ones(store.encoder.hyperbolic_dim) * 0.3
    }
    
    # Test forgetting
    initial_eu_norm = np.linalg.norm(store.residuals[test_id]['euclidean'])
    store.forget_residuals([test_id])
    final_eu_norm = np.linalg.norm(store.residuals[test_id]['euclidean'])
    
    assert final_eu_norm == 0, f"Forgetting failed: {final_eu_norm}"
    
    print(f"  Initial norm: {initial_eu_norm:.3f}")
    print(f"  Final norm: {final_eu_norm:.3f}")
    
    print("  Forgetting tests passed")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Enhanced Memory System")
    print("=" * 50)
    
    try:
        test_config()
        test_hyperbolic_stability()
        test_scale_aware_bounds()
        test_field_adaptation()
        test_provenance_tracking()
        test_forgetting()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\nX  Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\nX Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
