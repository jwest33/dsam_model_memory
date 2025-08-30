# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Dual-Space Memory System v2.0 with sophisticated encoding, adaptive learning, raw event preservation, and interactive visualization for AI agents.

**Core Architecture:**
- **Dual-Space Encoding**: Euclidean (768-dim) for concrete/lexical similarity + Hyperbolic (64-dim) for abstract/hierarchical relationships
- **Immutable Anchors**: Base embeddings preserved with bounded residual adaptation
- **Raw Memory Preservation**: All original events preserved alongside merged/deduplicated views
- **HDBSCAN Clustering**: Dynamic clustering based on query context with adjustable parameters
- **ChromaDB Backend**: Scalable vector storage with full 5W1H metadata + raw events collection
- **Enhanced Web Interface**: Real-time visualization, analytics, dual view modes, and graph exploration

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

3. **ChromaDBStore** (`memory/chromadb_store.py`)
   - Events collection with full 5W1H metadata (merged/deduplicated)
   - Raw events collection for preserving all original events
   - Blocks collection for clustering
   - Metadata collection for system state
   - Bidirectional mapping between raw and merged events
   - Persistent storage at `./state/chromadb`

4. **Enhanced Web App** (`web_app_enhanced.py`, `run_web.py`)
   - Flask backend with enhanced endpoints
   - Space weight calculation for queries
   - Graph generation with optional center node and interactive controls
   - Analytics data aggregation with merge statistics
   - Raw/Merged view toggle support
   - Auto-browser opening and graceful shutdown

5. **Frontend** (`static/js/app-enhanced.js`, `templates/index.html`)
   - Bootstrap 5 + custom synthwave theme + Bootstrap Icons
   - vis.js for graph visualization with individual memory focus
   - Chart.js for analytics (residuals, space usage, lambda weights)
   - Real-time updates and indicators
   - Merged/Raw view toggle with counts
   - Adjustable HDBSCAN parameters
   - Relation strength filters and gravity controls

### Memory Flow

```
User Input → 5W1H Event Creation
    ↓
Dual-Space Encoding (Euclidean + Hyperbolic)
    ↓
Store Raw Event in ChromaDB
    ↓
Check for Duplicates/Merging
    ↓
Store/Update Merged Event with Metadata
    ↓
Query → Compute Space Weights (λ_E, λ_H)
    ↓
Retrieve with Product Distance (Merged or Raw view)
    ↓
HDBSCAN Clustering (Optional, adjustable params)
    ↓
Residual Adaptation
    ↓
Return Ranked Results with View Toggle
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
   - Graph view with 5W1H component selection
   - Individual memory focus view (center node as star + related)
   - Multiple visualization modes with adjustable physics
   - Real-time clustering toggle with parameter controls
   - Relation strength filters and gravity adjustments

4. **Raw Memory Preservation**
   - All original events preserved in `raw_events` collection
   - Bidirectional mapping between raw and merged events
   - Toggle between Merged (deduplicated) and Raw (all events) views
   - Merge statistics and raw event counts
   - Complete event auditability

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
# Get memories with view mode
GET /api/memories?view=raw|merged

# Get raw events for a merged memory
GET /api/memory/<id>/raw

# Get merge statistics
GET /api/merge-stats

# Get analytics data
GET /api/analytics

# Graph generation with center node
POST /api/graph
{
    "center_node": "memory_id",  # Optional
    "min_cluster_size": 3,
    "min_samples": 2
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
- Similarity threshold: 0.4 for focused view
- Node styling: Center node displayed as star with red border
- Physics: Reduced repulsion (gravitationalConstant: -6000), added central gravity (0.4)
- Interactive controls: Relation strength filter, gravity adjustment
- HDBSCAN parameters: Adjustable via UI (min_cluster_size, min_samples)

### Raw Memory System
- Store raw events: All events preserved in `raw_events` collection
- Merge tracking: `merged_event_id` and `raw_event_ids` bidirectional mapping
- View toggle: Switch between Merged and Raw views in UI
- Statistics: Track merge ratios, average merge sizes, raw counts

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

# Generate test conversations
python generate_conversations.py

# Test API endpoints
python test_api.py

# Manual testing
python -c "
from memory.memory_store import MemoryStore
from memory.dual_space_encoder import DualSpaceEncoder
store = MemoryStore()
encoder = DualSpaceEncoder()
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
```

### Dataset Generation

#### Standard Conversations (`benchmark/generate_benchmark_dataset.py`)
- Generates diverse technical conversations
- Configurable conversation length and complexity
- Multiple scenario types (debugging, learning, architecture)
- Space preference modeling (Euclidean, Hyperbolic, Balanced)

```bash
# Generate standard benchmark dataset
python benchmark/generate_benchmark_dataset.py
# Interactive prompts for size and LLM options
```

#### Extended Conversations (`benchmark/generate_extended_conversations.py`)
- Creates long-form conversations (15-40 exchanges)
- Realistic multi-phase discussions
- Single user-assistant model (no personas)
- Pattern-based conversation flow:
  - Debugging sessions (15-25 exchanges)
  - Learning progressions (20-30 exchanges)
  - Project development (25-40 exchanges)
  - Architecture exploration (18-28 exchanges)

```bash
# Generate extended conversations
python benchmark/generate_extended_conversations.py --conversations 10

# With LLM enhancement
python benchmark/generate_extended_conversations.py --conversations 50 --use-llm
```

#### Reload Existing Data (`benchmark/reload_benchmark.py`)
- Reload previously generated benchmark datasets
- Useful for re-running tests with existing data

```bash
# Reload existing benchmark data
python benchmark/reload_benchmark.py
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
- HDBSCAN: Best with 20+ memories, adjustable parameters
- Real-time updates: < 1s for < 1000 memories
- Residual bounds: Euclidean < 0.35, Hyperbolic < 0.75
- Raw event storage: Minimal overhead (~5-10% additional storage)
- Benchmark performance: ~50ms avg response for 1000 events

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