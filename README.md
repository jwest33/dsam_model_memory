# Dual-Space Agentic Memory System (DSAM)

> **Pre-release v2.1** - Advanced semantic memory with multi-dimensional organization

## Overview

DSAM is a content-addressable memory system for AI agents that uses dual geometric spaces (Euclidean + Hyperbolic) for semantic retrieval. The system features multi-dimensional event merging, intelligent field generation, adaptive residual learning, and comprehensive provenance tracking.

## Key Features

- **Dual-Space Embeddings**: Combines Euclidean (768-dim) for local similarity and Hyperbolic (64-dim) for hierarchical relationships
- **4-Dimensional Merge Groups**: Organizes memories by Actor, Temporal, Conceptual, and Spatial dimensions
- **Intelligent Field Generation**: LLM-powered generation of 'why' and 'how' fields for semantic context
- **Cosine Similarity Merging**: Uses cosine distance for all merge types for better high-dimensional similarity
- **Adaptive Memory Retrieval**: Product distance combining both embedding spaces with query-dependent weights

## Technical Architecture

### Core Components

**Dual-Space Encoder** (`memory/dual_space_encoder.py`)
- Euclidean space (768-dim): Sentence transformer embeddings for concrete/lexical similarity
- Hyperbolic space (64-dim): Poincaré ball model for hierarchical/abstract relationships  
- Product distance retrieval combining both spaces (λ_E + λ_H = 1.0)
- Field-aware gating for 5W1H composition
- Geodesic distance calculation in hyperbolic space

**Memory Store** (`memory/memory_store.py`)
- Content-addressable retrieval without explicit memory IDs
- Bounded residual adaptation (Euclidean ≤ 0.35, Hyperbolic ≤ 0.75)
- Momentum-based updates (α=0.9, decay=0.995)
- Automatic event deduplication (threshold: 0.85)
- HDBSCAN clustering support for retrieval

**Multi-Dimensional Merger** (`memory/multi_dimensional_merger.py`)
- Actor dimension: Groups by participants (who)
- Temporal dimension: Groups by conversation threads (dynamic windows)
- Conceptual dimension: Groups by semantic similarity (cosine distance)
- Spatial dimension: Groups by location context
- Size-based regeneration with logarithmic scaling

**Field Generators** 
- `memory/field_generator.py`: Individual memory field generation with LLM fallback
- `memory/merge_group_field_generator.py`: Group-level characterization
- Multiple mechanism types: chat_interface, tool_use, llm_generation, etc.

**Qdrant Backend** (`memory/qdrant_store.py`)
- Collections: events, raw_events, merged_events, similarity_cache
- Dimensional collections: actor_merges, temporal_merges, conceptual_merges, spatial_merges
- Full 5W1H metadata preservation
- Persistent vector storage at `./qdrant_db`

## Installation

```bash
git clone https://github.com/jwest33/agent-wip
cd agent-wip
python setup_venv.py

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/MacOS
```

### Requirements
- Python 3.11+
- PyTorch
- Qdrant
- Sentence Transformers
- Flask
- OpenAI API key (optional, for field generation)

## Usage

### Web Interface
```bash
python run_web.py
# Access at http://localhost:5000
```

Features:
- Memory exploration with raw/merged views
- 4-dimensional merge group visualization
- Interactive memory graph
- Analytics dashboard
- Real-time memory creation

### Load Dataset with Field Generation
```bash
python load_llm_dataset.py
# Generates intelligent 'why' and 'how' fields for memories
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

### Analysis Tools
```bash
# Analyze conceptual similarity between groups
python analyze_conceptual_similarity.py

# Test field generation
python test_field_generation.py

# Test scaling regeneration
python test_scaling_regeneration.py
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

Key parameters:

```python
# Embedding Dimensions
Euclidean: 768 dimensions
Hyperbolic: 64 dimensions

# Merge Thresholds (Cosine Distance)
Conceptual: 0.3
Spatial: 0.3
Temporal: 0.5
Actor: 2.0

# Temporal Windows
Min: 10 minutes
Max: 60 minutes (dynamic)

# Regeneration
Max interval: 50 events
Logarithmic scaling function
```

## Performance

- Graph visualization: Optimized for ≤200 nodes
- Real-time updates: <1s for <1000 memories
- Memory retrieval: ~50ms average using dual-space product distance
- Cosine similarity calculation for merging
- Size-based regeneration reduces computational overhead

## Architecture Highlights

1. **Dual-Space Retrieval**: Combines Euclidean and Hyperbolic distances for nuanced similarity
2. **Cosine Similarity Merging**: Better performance in high-dimensional spaces
3. **Dynamic Temporal Windows**: Adapts to conversation patterns
4. **Group-Level Characterization**: Generates meaningful descriptions for merge groups
5. **Intelligent Field Generation**: Context-aware 'why' and 'how' fields

## License

MIT