# Dual-Space Agentic Memory System (DSAM)

> **Pre-release v2.1** - Functionality may change

## Overview

DSAM is a content-addressable memory system for AI agents that uses dual geometric spaces (Euclidean + Hyperbolic) for semantic retrieval. The system features multi-dimensional event merging, adaptive residual learning, and comprehensive provenance tracking.

## Technical Architecture

### Core Components

**Dual-Space Encoder** (`memory/dual_space_encoder.py`)
- Euclidean space (768-dim): Sentence transformer embeddings for concrete/lexical similarity
- Hyperbolic space (64-dim): Poincaré ball model for hierarchical/abstract relationships  
- Query-adaptive weighting between spaces (λ_E + λ_H = 1.0)
- Field-aware gating for 5W1H composition

**Memory Store** (`memory/memory_store.py`)
- Content-addressable retrieval without explicit memory IDs
- Bounded residual adaptation (Euclidean ≤ 0.35, Hyperbolic ≤ 0.75)
- Momentum-based updates (α=0.9, decay=0.995)
- Automatic event deduplication (threshold: 0.85)
- HDBSCAN clustering support

**ChromaDB Backend** (`memory/chromadb_store.py`)
- Collections: events (merged), raw_events (originals), blocks, metadata, similarity_cache
- Full 5W1H metadata preservation
- Bidirectional raw↔merged mapping
- Persistent vector storage at `./state/chromadb`

**Multi-Dimensional Merger** (`memory/multi_dimensional_merger.py`)
- Actor dimension: Groups by participants (who)
- Temporal dimension: Groups by conversation threads
- Conceptual dimension: Groups by concepts/goals
- Spatial dimension: Groups by location context

**Temporal Manager** (`memory/temporal_manager.py`)
- Unified temporal operations and chain tracking
- Query strength detection (Strong/Moderate/Weak)
- Episode and conversation thread management

**Similarity Cache** (`memory/similarity_cache.py`)
- Pre-computed pairwise similarities
- O(1) lookup performance
- Sparse storage (threshold: 0.2)

## Installation

```bash
git clone https://github.com/jwest33/dsam_model_memory
cd agent-wip
python setup_venv.py

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/MacOS
```

### Requirements
- Python 3.11+
- PyTorch
- ChromaDB
- Sentence Transformers
- Flask

## Usage

### Web Interface
```bash
python run_web.py
# Access at http://localhost:5000
```

### CLI Operations
```bash
# Store memory
python cli.py remember --who "Alice" --what "implemented feature" --where "backend"

# Retrieve memories
python cli.py recall --what "feature" --k 10

# View statistics
python cli.py stats
```

### Generate Test Data
```bash
python benchmark/generate_benchmark_dataset_fast.py
# Options: Small (100), Medium (500), Large (1000), Extra Large (2000), Massive (5000)
```

## API Endpoints

- `POST /api/chat` - Chat interface with memory groups
- `GET /api/memories?view=raw|merged` - Retrieve memories
- `POST /api/memories` - Create memory
- `GET /api/memory/<id>/raw` - Get raw events for merged memory
- `GET /api/merge-groups/<type>` - Get merge groups by dimension
- `POST /api/graph` - Generate memory graph
- `GET /api/analytics` - System analytics

## Configuration

Key settings in `config.py`:

```python
DualSpaceConfig:
  euclidean_dim: 768
  hyperbolic_dim: 64
  euclidean_bound: 0.35
  hyperbolic_bound: 0.75
  momentum: 0.9
  decay_factor: 0.995
  similarity_threshold: 0.85

StorageConfig:
  chromadb_path: "./state/chromadb"
```

## Performance

- Graph visualization: Optimized for ≤200 nodes
- Real-time updates: <1s for <1000 memories
- Similarity cache: 100% hit rate after initial population
- Memory retrieval: ~50ms average for 1000 events
- Dataset generation: ~0.3-0.5 conversations/sec (LLM mode)

## License

MIT
