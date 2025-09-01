# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Dual-Space Memory System v2.1 with sophisticated encoding, adaptive learning, raw event preservation, unified temporal management, and interactive visualization for AI agents.

**Core Architecture:**
- **Dual-Space Encoding**: Euclidean (768-dim) for concrete/lexical similarity + Hyperbolic (64-dim) for abstract/hierarchical relationships
- **Multi-Dimensional Merging**: Events organized by Actor, Temporal, Conceptual, and Spatial dimensions
- **Unified Temporal Management**: Consolidated temporal operations with strength-based query detection
- **Immutable Anchors**: Base embeddings preserved with bounded residual adaptation
- **Raw Memory Preservation**: All original events preserved alongside merged/deduplicated views
- **ChromaDB Backend**: Scalable vector storage with full 5W1H metadata + raw events collection
- **Enhanced Web Interface**: Real-time chat with memory groups, visualization, analytics, and graph exploration

**Recent Enhancements (v2.1):**
- Unified temporal manager consolidating all temporal logic
- Enhanced chat interface with memory group routing
- Improved merge group details and navigation
- Temporal strength detection (Strong/Moderate/Weak)
- Better pagination support for large datasets
- Streamlined JS modules with consolidated functionality
- Event-based timestamps for memory groups (created_at/last_updated use event 'when' values)
- System timestamps preserved separately for merge statistics
- LLM-powered benchmark dataset generation with realistic conversations
- Timestamp preservation when loading historical datasets

## Key Commands

### Running the System

```bash
# Set offline mode to avoid HuggingFace rate limits
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Launch enhanced web interface
python run_web.py

# Access at http://localhost:5000

# Run conversation simulations
python simulate_conversations.py

# Run automated experiments
python run_experiments.py

# Clear database (stop server first)
python clear_memories.py

# Test chat API
python test_chat.py "Your message here"

# Generate benchmark datasets (fast version)
python benchmark/generate_benchmark_dataset_fast.py

# Test similarity cache performance
python benchmark_similarity_performance.py
```

### CLI Commands

```bash
# Initialize system
python cli.py init

# Store memories with full 5W1H
python cli.py remember --who "Alice" --what "implemented feature" --where "backend" --why "requirement" --how "coding"

# Recall memories
python cli.py recall --what "feature" --k 10

# View statistics
python cli.py stats

# Save/Load state
python cli.py save
python cli.py load
```

## Architecture Details

### Core Components

1. **DualSpaceEncoder** (`memory/dual_space_encoder.py`)
   - Sentence transformer base: 'sentence-transformers/all-MiniLM-L6-v2'
   - Euclidean operations for local semantics
   - Hyperbolic operations (Poincaré ball model)
   - Field-aware gating with learned weights
   - Query-dependent space weighting

2. **MemoryStore** (`memory/memory_store.py`)
   - Dual-space encoding integration
   - Residual storage with bounds (Euclidean: 0.35, Hyperbolic: 0.75)
   - Momentum tracking (α=0.01, momentum=0.9)
   - Decay factor: 0.995
   - HDBSCAN clustering support
   - Similarity cache integration for performance

3. **ChromaDBStore** (`memory/chromadb_store.py`)
   - Events collection with full 5W1H metadata (merged/deduplicated)
   - Raw events collection for preserving all original events
   - Multi-dimensional merge collections (actor, temporal, conceptual, spatial)
   - Blocks collection for clustering
   - Metadata collection for system state
   - Similarity cache collection for pre-computed scores
   - Bidirectional mapping between raw and merged events
   - Persistent storage at `./state/chromadb`

4. **MultiDimensionalMerger** (`memory/multi_dimensional_merger.py`)
   - Manages multiple merge dimensions simultaneously
   - Actor merging: Groups by participants (who)
   - Temporal merging: Groups by conversation threads
   - Conceptual merging: Groups by concepts and goals
   - Spatial merging: Groups by location context
   - Tracks merge memberships across dimensions
   - Persists merge groups to ChromaDB
   - Event-based timestamps: `created_at` uses earliest event's 'when', `last_updated` uses latest event's 'when'
   - System timestamps (`system_created_at`, `system_last_updated`) track actual merge operations

5. **Temporal Manager** (`memory/temporal_manager.py`)
   - Unified temporal operations management
   - Temporal chain tracking for conversation threads
   - Temporal query detection with strength levels (Strong/Moderate/Weak)
   - Temporal merge group creation and management
   - Temporal dimension attention computation
   - Episode and conversation thread management

6. **Enhanced Web App** (`web_app.py`, `run_web.py`)
   - Flask backend with enhanced endpoints
   - Chat interface with memory group routing
   - Space weight calculation for queries
   - Graph generation with optional center node and interactive controls
   - Analytics data aggregation with merge statistics
   - Raw/Merged view toggle support
   - Auto-browser opening and graceful shutdown

7. **Frontend** (`static/js/app.js`, `templates/index.html`)
   - Bootstrap 5 + custom synthwave theme + Bootstrap Icons
   - vis.js for graph visualization with individual memory focus
   - Chart.js for analytics (residuals, space usage, lambda weights)
   - Real-time updates and indicators
   - Multi-dimensional merge dimension selector
   - Merged/Raw view toggle with counts
   - Simplified graph controls with layout slider
   - Unified component tables with aligned columns
   - Chat interface with memory group selection
   - Memory group details modal
   - Pagination support for large datasets

8. **Similarity Cache** (`memory/similarity_cache.py`)
   - Pre-computed pairwise similarity scores
   - Sparse storage (threshold: 0.2 for graph edges)
   - Batch computation for efficiency
   - Persistent storage in ChromaDB
   - Cache statistics and hit rate tracking

### Memory Flow

```
User Input → 5W1H Event Creation
    ↓
Dual-Space Encoding (Euclidean + Hyperbolic)
    ↓
Store Raw Event in ChromaDB
    ↓
Multi-Dimensional Merging
    ├── Actor Dimension (who)
    ├── Temporal Dimension (conversation threads)
    ├── Conceptual Dimension (goals/concepts)
    └── Spatial Dimension (location)
    ↓
Update Similarity Cache (batch/incremental)
    ↓
Check for Duplicates/Merging (threshold: 0.85)
    ↓
Store/Update Merged Events with Metadata
    ↓
Query → Compute Space Weights (λ_E, λ_H)
    ↓
Retrieve with Cached Similarities → Product Distance
    ↓
Residual Adaptation
    ↓
Return Ranked Results with Multi-Dimensional Views
```

### Key Innovations

1. **Query-Adaptive Retrieval**
   - Concrete queries (code, errors) → Higher Euclidean weight
   - Abstract queries (concepts, philosophy) → Higher Hyperbolic weight
   - Balanced queries → Equal weights

2. **Bounded Residual Adaptation**
   - Residuals bounded to prevent drift
   - Momentum for smooth updates
   - Automatic decay over time
   - Co-retrieval strengthens connections

3. **Interactive Visualization**
   - Graph view with multi-dimensional merge group support
   - Individual memory focus view (center node as star + related)
   - Simplified controls with layout adjustment slider
   - Support for visualizing any merge dimension
   - Color-coded nodes by entity type

4. **Raw Memory Preservation**
   - All original events preserved in `raw_events` collection
   - Bidirectional mapping between raw and merged events
   - Toggle between Merged (deduplicated) and Raw (all events) views
   - Merge statistics and raw event counts
   - Complete event auditability

5. **Performance Optimization**
   - Similarity cache eliminates redundant computations
   - Batch processing for dataset generation
   - Multi-threaded conversation generation
   - Pre-computed similarities loaded on startup
   - O(1) similarity lookups vs O(n²) computation

## Configuration

Key settings in `config.py`:

```python
class MemoryConfig:
    embedding_dim: int = 384
    temperature: float = 15.0
    similarity_threshold: float = 0.85
    
class DualSpaceConfig:
    euclidean_dim: int = 768
    hyperbolic_dim: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    euclidean_bound: float = 0.35
    hyperbolic_bound: float = 0.75
    decay_factor: float = 0.995

class StorageConfig:
    chromadb_path: str = "./state/chromadb"
    chromadb_required: bool = True
    benchmark_datasets_path: str = "./benchmark_datasets"
```

## Implementation Patterns

### Adding New Memory Operations

```python
# In memory_agent.py
def new_operation(self, ...):
    # 1. Manage episode if needed
    self._manage_episode()
    
    # 2. Create Event with full 5W1H
    event = Event(
        five_w1h=FiveW1H(
            who=who,
            what=what,
            when=when,
            where=where,
            why=why,
            how=how
        ),
        episode_id=self.current_episode_id
    )
    
    # 3. Store event (no salience check)
    success, message = self.memory_store.store_event(event)
    
    return success, message, event
```

### Using Dual-Space Retrieval

```python
# Retrieve with dual-space encoding
results = memory_store.retrieve_memories(
    query={"what": "search query", "who": "user"},
    k=10,
    use_clustering=True,
    update_residuals=True
)

# Adapt from feedback
memory_store.adapt_from_feedback(
    query=query,
    positive_events=[...],
    negative_events=[...]
)
```

### Adding API Endpoints

```python
# In web_app_enhanced.py
@app.route('/api/new-endpoint', methods=['POST'])
def new_endpoint():
    data = request.json
    
    # Process with dual-space system
    query_fields = {'what': data.get('query')}
    lambda_e, lambda_h = encoder.compute_query_weights(query_fields)
    
    # Use memory agent
    results = memory_agent.recall(**query_fields, k=10)
    
    return jsonify({
        'results': results,
        'space_weights': {
            'euclidean': float(lambda_e),
            'hyperbolic': float(lambda_h)
        }
    })
```

### Key API Endpoints

```python
# Chat interface with memory groups
POST /api/chat
{
    "message": "user message",
    "merge_dimension": "temporal|actor|conceptual|spatial",
    "merge_group_id": "optional_group_id"
}

# Get memories with view mode and pagination
GET /api/memories?view=raw|merged&page=1&per_page=50

# Get raw events for a merged memory
GET /api/memory/<id>/raw

# Get merge groups for a memory
GET /api/memory/<memory_id>/merge-groups

# Get merge groups for a raw event
GET /api/raw-event/<event_id>/merge-groups

# Get merged details for a memory
GET /api/memory/<memory_id>/merged-details

# Get merge statistics
GET /api/merge-stats

# Get merge dimensions info
GET /api/merge-dimensions

# Get merge types
GET /api/merge-types

# Get merge groups by type
GET /api/merge-groups/<type>  # type: actor|temporal|conceptual|spatial

# Get specific merge group
GET /api/merge-group/<type>/<group_id>

# Get multi-dimensional merge details
GET /api/multi-merge/<type>/<id>/details

# Get event merges
GET /api/event-merges/<event_id>

# Search memories
POST /api/search
{
    "query": "search text",
    "k": 10
}

# Get temporal summary
GET /api/temporal-summary

# Get temporal context for an event
GET /api/temporal-context/<event_id>

# Get analytics data
GET /api/analytics

# Get similarity cache statistics
GET /api/similarity-cache-stats

# Graph generation with center node
POST /api/graph
{
    "center_node": "memory_id",  # Optional, supports multi-dimensional IDs
    "visualization_mode": "dual",
    "use_clustering": false
}

## Important Implementation Details

### ChromaDB Metadata Storage
All 5W1H fields MUST be stored in metadata for retrieval:
```python
metadata = {
    "event_type": event.event_type.value,
    "episode_id": event.episode_id,
    "who": event.five_w1h.who or "",
    "what": event.five_w1h.what or "",
    "when": event.five_w1h.when or "",
    "where": event.five_w1h.where or "",
    "why": event.five_w1h.why or "",
    "how": event.five_w1h.how or "",
}
```

### Graph Visualization
- Individual memory view: Pass `center_node` parameter
- Multi-dimensional support: Handles temporal_xxx, actor_xxx, conceptual_xxx, spatial_xxx IDs
- Node styling: Center node displayed as star with red border
- Physics: Adjustable via layout slider (compact to spread out)
- Simplified controls: Single layout adjustment slider
- Color coding: Different colors for users, assistants, and memory types

### Raw Memory System
- Store raw events: All events preserved in `raw_events` collection
- Merge tracking: `merged_event_id` and `raw_event_ids` bidirectional mapping
- View toggle: Switch between Merged and Raw views in UI
- Statistics: Track merge ratios, average merge sizes, raw counts

### Timestamp Management
- Event timestamps: Memory groups use earliest/latest event 'when' values for created_at/last_updated
- System timestamps: Actual merge operation times stored as system_created_at/system_last_updated
- UI display: Event timestamps shown everywhere except Merge Statistics section
- Memory Store tab: Shows latest event 'when' value for merged events
- Historical data: Original timestamps preserved when loading datasets with `load_llm_dataset.py`

### Frontend Event Handling
All interactive functions must be exposed to window:
```javascript
window.functionName = async function(...) {
    // Implementation
}
```

## Testing Approach

```bash
# Test basic functionality
python test_offline.py

# Test temporal updates and manager
python test_temporal_updates.py

# Test merge groups functionality
python test_merge_groups.py

# Generate test conversations
python generate_conversations.py

# Test API endpoints
python test_api.py

# Test chat with memory groups
python test_chat.py "Your message" --merge-dimension temporal --merge-group-id <group_id>

# Manual testing
python -c "
from memory.memory_store import MemoryStore
from memory.dual_space_encoder import DualSpaceEncoder
from memory.temporal_manager import TemporalManager
store = MemoryStore()
encoder = DualSpaceEncoder()
temporal_mgr = TemporalManager(encoder, store.chromadb, store.similarity_cache)
# Test operations...
"
```

## Benchmarking and Evaluation System

### Unified Evaluator (`benchmark/evaluator.py`)
Comprehensive evaluation module supporting multiple modes:
- **Performance Mode**: Response time, memory usage, CPU metrics
- **Quality Mode**: Precision@K, Recall@K, NDCG@K, semantic coherence
- **Both Mode**: Complete evaluation with all metrics
- **Quick Mode**: Fast test run with subset of queries
- **Similarity Cache Benchmarking**: Tests cache performance improvements

```bash
# Run unified evaluation
python benchmark/evaluator.py --mode both

# Performance evaluation only
python benchmark/evaluator.py --mode performance --dataset benchmark_datasets/dataset.json

# Quality evaluation with existing memories
python benchmark/evaluator.py --mode quality --num-queries 50

# Compare configurations
python benchmark/evaluator.py --compare

# Quick test
python benchmark/evaluator.py --mode quick

# Test similarity cache performance
python benchmark_similarity_performance.py
```

### Dataset Generation

#### Fast Dataset Generation (`benchmark/generate_benchmark_dataset_fast.py`)
- LLM-powered conversation generation using llama.cpp server
- Realistic timestamps with proper delays between messages
- Mixed technical (60%) and casual (40%) conversations
- Proper 5W1H structure for all events
- Multi-threaded conversation generation (2-8 threads)
- Real-time progress tracking with time estimates
- Supports datasets up to 5000+ conversations
- Preserves timestamps when loading with `load_llm_dataset.py`

```bash
# Generate benchmark dataset with fast mode
python benchmark/generate_benchmark_dataset_fast.py
# Options: Small (100), Medium (500), Large (1000), Extra Large (2000), Massive (5000)
```

#### Reload Existing Data (`load_llm_dataset.py`)
- Reload previously generated benchmark datasets
- Useful for re-running tests with existing data

```bash
# Reload existing benchmark data
python load_llm_dataset.py
```

### Performance Metrics
- Response times: Average, median, P95, P99
- Memory usage: Peak and average
- CPU usage: Monitoring during operations
- Space usage distribution: Euclidean vs Hyperbolic
- Configuration comparisons

### Quality Metrics
- Precision@K, Recall@K, NDCG@K (K = 1, 3, 5, 10)
- Semantic coherence scores
- Space utilization analysis
- Statistical significance testing

### Output
- JSON results with comprehensive metrics
- CSV files with detailed performance data
- Configuration comparison reports
- Statistical analysis with confidence intervals
- Results stored in `./evaluation_results/`

## Troubleshooting

### Common Issues

1. **Empty memory fields in UI**
   - Ensure ChromaDB metadata includes all 5W1H fields
   - Check `store_event` in chromadb_store.py

2. **Graph not displaying**
   - Verify vis.js and Chart.js are loaded
   - Check browser console for errors

3. **High residuals**
   - Monitor via Analytics tab
   - System auto-decays with factor 0.995
   - Clear if consistently > bounds

4. **ChromaDB locked**
   - Stop web server before clearing
   - Use `clear_memories.py` instead of `clear_db.py`

## Performance Notes

- Graph visualization: Optimized for ≤200 nodes
- Multi-dimensional merging: Handles 4 dimensions simultaneously
- Real-time updates: < 1s for < 1000 memories
- Residual bounds: Euclidean < 0.35, Hyperbolic < 0.75
- Raw event storage: Minimal overhead (~5-10% additional storage)
- Benchmark performance: ~50ms avg response for 1000 events
- Similarity cache: 100% hit rate after initial population
- Cache threshold: 0.2 for graph edges (vs 0.85 for deduplication)
- LLM dataset generation: ~0.3-0.5 conversations/sec with realistic content
- Template dataset generation: ~100+ conversations/sec (non-LLM)
- Similarity lookups: O(1) with cache vs O(n²) without
- Temporal sorting: Proper chronological ordering with recency boost

## Windows-Specific Notes

- Use offline mode to avoid network issues
- ChromaDB stores in `state/chromadb/`
- Backups use JSON for portability
- May need to stop server to clear database

## Key Principles

1. **Dual-Space Balance**: Let query determine space weights
2. **Bounded Adaptation**: Preserve anchor stability
3. **Full Context**: Always store complete 5W1H
4. **Visual Feedback**: Show space usage throughout UI
5. **Scalable Design**: No arbitrary limits or thresholds
6. **Event Preservation**: Keep all raw events for complete auditability
7. **User Control**: Provide toggles for views, parameters, and visualizations