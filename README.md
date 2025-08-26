# Dual-Space Memory System v2.0

An advanced memory system for AI agents featuring dual-space encoding (Euclidean + Hyperbolic), adaptive residual learning, and dynamic visualization.

## Overview

This system implements a sophisticated memory architecture that combines Euclidean space for concrete/lexical similarity with Hyperbolic space for abstract/hierarchical relationships. The system features immutable anchor embeddings with bounded residual adaptation, enabling memories to evolve while maintaining stable representations.

## Core Architecture

### Dual-Space Encoding
- **Euclidean Space** (768-dim): Captures local semantic similarity for concrete information
- **Hyperbolic Space** (64-dim): Models hierarchical relationships using Poincaré ball geometry
- **Field-Aware Composition**: Learned gates weight contributions from 5W1H fields
- **Product Distance Metrics**: Query-dependent weighting between spaces (λ_E + λ_H = 1.0)

### Adaptive Memory System
- **Immutable Anchors**: Base embeddings never corrupted
- **Bounded Residuals**: Euclidean ≤ 0.35, Hyperbolic ≤ 0.75
- **Momentum-Based Updates**: Smooth adaptation with decay factor 0.995
- **HDBSCAN Clustering**: Superior density-based clustering for varied data distributions

### 5W1H Framework
Complete context encoding for each memory:
- **Who**: Entity or actor involved
- **What**: Action, observation, or content
- **When**: Temporal information
- **Where**: Location or context
- **Why**: Purpose, reasoning, or intent
- **How**: Method, approach, or process

## Installation

```bash
# Clone repository
git clone <repository-url>
cd agent-wip

# Install dependencies
pip install -r requirements.txt

# For offline mode (recommended)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Quick Start

### Command Line Interface

```bash
# Initialize memory system
python cli.py init

# Store memories
python cli.py remember --who "Alice" --what "implemented search feature" --where "backend" --why "user requirement" --how "elasticsearch integration"

# Recall memories
python cli.py recall --what "search" --k 10

# View statistics
python cli.py stats

# Save/Load state
python cli.py save
python cli.py load
```

### Web Interface

```bash
# Launch enhanced web interface
python run_web.py

# Access at http://localhost:5000
```

### Running Experiments

```bash
# Run conversation simulations
python simulate_conversations.py

# Run automated experiments
python run_experiments.py

# Test chat functionality
python test_chat.py "Your message here"
```

## Web Interface Features

### Chat Interface
- Real-time space weight visualization (Euclidean vs Hyperbolic)
- Query type detection (concrete/abstract/balanced)
- Memory usage indicators
- Context-aware responses

### Memory Management
- Full 5W1H field display
- Space dominance indicators
- Residual norm visualization
- Individual memory graph view
- Batch operations support

### Graph Visualization
- **Interactive Network Graph**: Powered by vis.js
- **5W1H Component Selection**: Choose which fields to visualize
- **Multiple Visualization Modes**:
  - Dual-space view
  - Euclidean-only
  - Hyperbolic-only
  - Residual magnitude
- **HDBSCAN Clustering**: Toggle clustering visualization
- **Individual Memory Focus**: View a memory and its related connections
- **Graph Statistics**: Nodes, edges, clusters, average degree

### Analytics Dashboard
- Total events and queries metrics
- Average residual norms by space
- Residual evolution time series
- Space usage distribution chart
- Real-time updates

## API Endpoints

### Core Endpoints
- `POST /api/chat`: Send chat messages with space weight calculation
- `GET /api/memories`: Retrieve all memories with metadata
- `POST /api/memories`: Create new memory
- `DELETE /api/memories/<id>`: Delete specific memory
- `POST /api/graph`: Get memory graph data with optional center node
- `POST /api/search`: Search memories by query
- `GET /api/stats`: Get system statistics
- `GET /api/analytics`: Get analytics data for charts

## System Architecture

### Memory Store (`memory/memory_store.py`)
- Dual-space encoding integration
- Residual and momentum tracking
- HDBSCAN clustering support
- ChromaDB backend interface

### Dual-Space Encoder (`memory/dual_space_encoder.py`)
- Sentence transformer for base embeddings
- Hyperbolic operations (Möbius addition, exp/log maps)
- Field-aware gating mechanism
- Query weight computation

### ChromaDB Storage (`memory/chromadb_store.py`)
- Persistent vector storage
- Full 5W1H metadata preservation
- Efficient similarity search
- Unlimited capacity scaling

## Key Innovations

### Query-Adaptive Retrieval
- Dynamic space weighting based on query type
- Concrete queries favor Euclidean space
- Abstract queries favor Hyperbolic space
- Balanced queries use both spaces equally

### Residual Adaptation
- Memories adapt based on co-retrieval patterns
- Bounded updates prevent representation drift
- Momentum smoothing for stable learning
- Automatic decay over time

### Visual Analytics
- Space usage indicators throughout UI
- Real-time residual tracking
- Interactive graph exploration
- Component-based filtering

## Configuration

Key settings in `config.py`:

```python
MemoryConfig:
  embedding_dim: 384
  temperature: 15.0
  similarity_threshold: 0.85

DualSpaceConfig:
  euclidean_dim: 768
  hyperbolic_dim: 64
  learning_rate: 0.01
  momentum: 0.9
  euclidean_bound: 0.35
  hyperbolic_bound: 0.75
  decay_factor: 0.995

StorageConfig:
  chromadb_path: "./state/chromadb"
  chromadb_required: true
```

## Performance Characteristics

- **Graph Visualization**: Optimized for up to 200 nodes
- **HDBSCAN Clustering**: Best with 20+ memories
- **Real-time Updates**: Sub-second for < 1000 memories
- **Residual Bounds**: Euclidean < 0.35, Hyperbolic < 0.75

## Troubleshooting

### Common Issues

**HuggingFace Rate Limiting**
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**ChromaDB Locked**
```bash
# Stop web server before clearing
python clear_memories.py
```

**High Residuals**
- System automatically decays residuals
- Monitor via Analytics dashboard
- Consider clearing if consistently > threshold

## Development

### Testing
```bash
# Basic offline test
python test_offline.py

# Generate test conversations
python generate_conversations.py

# Simple functionality test
python test_simple.py
```

### Adding New Features
1. Extend memory operations in `memory_agent.py`
2. Add API endpoints in `web_app_enhanced.py`
3. Update frontend in `static/js/app-enhanced.js`
4. Document in relevant markdown files

## License

[Specify your license here]

## Contributing

[Contribution guidelines if applicable]

## Citation

If you use this system in research, please cite:
```
[Citation format if applicable]
```