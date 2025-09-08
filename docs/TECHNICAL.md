# Technical Architecture

## System Overview

JAM (Journalistic Agent Memory) is a sophisticated memory system that provides LLM agents with persistent, searchable memory using a multi-stage pipeline for ingestion, storage, retrieval, and context building. The system combines vector similarity search, full-text search, and dynamic clustering to create a human-like memory system.

## Core Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         Web Interface                        │
│                    Flask Server (Port 5001)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                      API Wrapper Layer                       │
│              FastAPI Server (Port 8001)                      │
│            OpenAI-Compatible Endpoints                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Memory Router Core                        │
│         Orchestration, Tool Handling, Retrieval              │
└────────┬───────────────────────────────┬────────────────────┘
         │                               │
┌────────▼────────┐             ┌────────▼────────┐
│  Storage Layer  │             │  Retrieval Layer │
│  SQLite + FAISS │             │  Hybrid Search   │
└─────────────────┘             └─────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    LLM Backend                               │
│              llama.cpp Server (Port 8000)                    │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

### 1. Memory Ingestion

The ingestion pipeline transforms raw events into structured, searchable memories:

```
Raw Event → 5W1H Extraction → Embedding Generation → Dual Storage → Index Update
```

#### 1.1 Event Capture
- **Sources**: User messages, assistant responses, tool executions
- **Metadata**: Timestamps, actors, event types, session IDs
- **Format**: Raw text with structured metadata envelope

#### 1.2 5W1H Extraction
The system uses a local LLM to decompose events into journalistic dimensions:

```python
{
    "who": "Actor identification (user, assistant, tool:name)",
    "what": "Core action or content",
    "when": "ISO timestamp with timezone",
    "where": "Context location (URL, file path, session)",
    "why": "Intent or purpose",
    "how": "Method or approach used"
}
```

#### 1.3 Embedding Generation
- **Model**: Configured LLM server (same model for consistency)
- **Dimensions**: Model-dependent (typically 2048-4096)
- **Normalization**: L2 normalization for cosine similarity
- **Input**: Concatenated 5W1H fields for comprehensive representation

#### 1.4 Storage Architecture

**SQLite Database**:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    who TEXT,
    what TEXT,
    when TEXT,
    where TEXT,
    why TEXT,
    how TEXT,
    embedding BLOB,
    metadata JSON,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    created_at TIMESTAMP,
    cluster_id INTEGER
);

-- Full-text search index
CREATE VIRTUAL TABLE memories_fts USING fts5(
    who, what, when, where, why, how,
    content=memories
);
```

**FAISS Vector Index**:
- Flat index for exact nearest neighbor search
- No approximation ensures perfect recall
- Memory-mapped for large datasets
- Supports incremental updates

### 2. Retrieval System

The hybrid retrieval system combines multiple signals for human-like memory recall:

```
Query → Multi-Signal Scoring → Score Fusion → MMR Diversification → Selection
```

#### 2.1 Six Retrieval Signals

**Semantic Similarity** (Weight: 0.55)
- Vector similarity using FAISS
- Cosine distance in embedding space
- Captures meaning and concepts

**Lexical Match** (Weight: 0.20)
- BM25 scoring via SQLite FTS5
- Exact keyword matching
- Porter stemming for variants

**Recency Bias** (Weight: 0.10)
- Exponential decay with 30-day half-life
- `score = e^(-0.693 * age_days / 30)`
- Prioritizes recent memories

**Actor Relevance** (Weight: 0.07)
- Matches memories by participant
- Boosts actor-specific memories
- Tracks conversation threads

**Spatial Context** (Weight: 0.03)
- Location similarity scoring
- Path component matching
- Context preservation

**Usage Patterns** (Weight: 0.05)
- Access frequency tracking
- Co-activation patterns
- Reinforcement through use

#### 2.2 Score Fusion

```python
final_score = Σ(weight_i * signal_i)
where Σ(weights) = 1.0
```

### 3. Context Building

The system optimally packs memories into limited context windows:

```
Scored Memories → Token Budgeting → MMR Selection → Knapsack Packing → Context
```

#### 3.1 Token Budget Management
```python
available_tokens = context_window - system_tokens - output_reserve - current_conversation
# Typical: 8192 - 512 - 1024 - conversation ≈ 6000+ tokens
```

#### 3.2 Maximal Marginal Relevance (MMR)
Balances relevance with diversity:

```python
MMR = λ * relevance - (1-λ) * max_similarity_to_selected
# λ = 0.5 (default): Equal weight to relevance and diversity
```

#### 3.3 0-1 Knapsack Optimization
Treats context building as an optimization problem:
- **Maximize**: Total relevance value
- **Constraint**: Token budget
- **Method**: Dynamic programming solution
- **Result**: Optimal memory selection

## Clustering System

### Liquid Memory Clusters

The system implements self-organizing memory topology:

#### Clustering Algorithm
- **Method**: MiniBatch KMeans
- **Clusters**: Dynamic (8-64 based on data size)
- **Update**: Incremental via partial_fit
- **Features**: Real-time adaptation

#### Cluster Dynamics
```python
# Energy increases with access
energy = min(1.0, energy + 0.05 * access_count)

# Natural decay over time
energy *= 0.99  # Per timestep

# Clusters merge when similar
if similarity(cluster1, cluster2) > 0.85:
    merge_clusters()
```

### 3D Visualization

The visualization system provides intuitive memory exploration:

#### Dimensionality Reduction
Three methods available:

**PCA** (Principal Component Analysis)
- Linear projection
- Fast, deterministic
- Preserves global structure

**t-SNE** (t-Distributed Stochastic Neighbor Embedding)
- Non-linear projection
- Preserves local neighborhoods
- Best for cluster visualization

**UMAP** (Uniform Manifold Approximation)
- Manifold learning
- Balances local/global structure
- Fastest for large datasets

#### Visualization Features
- **Real-time Updates**: 30-second refresh
- **Color Schemes**: Cluster ID, energy, recency, usage
- **Interactive**: 3D rotation and zoom
- **Adaptive**: Cluster count adjusts to data

## Performance Characteristics

### Latency Metrics
- **Memory Ingestion**: 100-200ms per memory
- **Retrieval (1K memories)**: <100ms
- **Retrieval (10K memories)**: 200-300ms
- **Retrieval (100K memories)**: 1-2 seconds
- **Context Building**: 50-100ms

### Scalability
- **Current Testing**: 5000+ production memories
- **Practical Limit**: ~1M memories (with current architecture)
- **Index Size**: ~1MB per 1000 memories (FAISS)
- **Database Size**: ~2MB per 1000 memories (SQLite)

### Resource Usage
- **RAM**: 2-4GB typical usage
- **CPU**: Moderate (spikes during embedding/retrieval)
- **Disk**: Linear growth with memory count
- **GPU**: Optional (CPU inference supported)

## Configuration System

### ConfigManager Architecture
- **Centralized**: Single source of truth
- **Persistent**: Database-backed settings
- **Hot-reload**: Runtime configuration changes
- **UI Integration**: Web-based configuration panel

### Key Parameters

#### Retrieval Weights
Must sum to 1.0:
```python
{
    "semantic": 0.55,   # Vector similarity
    "lexical": 0.20,    # Keyword matching
    "recency": 0.10,    # Time decay
    "actor": 0.07,      # Participant relevance
    "spatial": 0.03,    # Location context
    "usage": 0.05       # Access patterns
}
```

#### System Parameters
```python
{
    "context_window": 8192,        # Model context limit
    "system_reserve": 512,         # System prompt tokens
    "output_reserve": 1024,        # Response space
    "mmr_lambda": 0.5,            # Diversity vs relevance
    "decay_half_life": 30,        # Days for 50% decay
    "max_memories_per_query": 50  # Retrieval limit
}
```

## API Specifications

### OpenAI-Compatible Endpoints

#### Chat Completions
```http
POST /v1/chat/completions
{
    "messages": [...],
    "model": "local",
    "temperature": 0.7,
    "max_tokens": 1000
}
```

#### Completions
```http
POST /v1/completions
{
    "prompt": "...",
    "model": "local",
    "max_tokens": 500
}
```

#### Memory Operations
```http
POST /memory/add
{
    "content": "Memory to add",
    "metadata": {...}
}

POST /memory/search
{
    "query": "search terms",
    "limit": 10
}
```

### WebSocket Updates
The Flask server provides real-time updates via Server-Sent Events:
- Memory addition notifications
- Retrieval results streaming
- Cluster reorganization events

## Security Considerations

### Data Privacy
- **Local-Only**: No external API calls
- **No Telemetry**: Zero data collection
- **Encryption**: Optional SQLite encryption
- **Access Control**: Local filesystem permissions

### Input Validation
- SQL injection prevention via parameterized queries
- XSS protection in web interface
- Rate limiting on API endpoints
- Input sanitization for LLM prompts

## Optimization Strategies

### Index Optimization
- Composite indexes on frequently queried fields
- FTS5 with Porter stemmer for better recall
- Periodic VACUUM for database optimization
- Index rebuilding for fragmentation

### Memory Optimization
- Lazy loading of embeddings
- Batch processing for bulk operations
- Memory-mapped FAISS for large indices
- Connection pooling for SQLite

### Caching Strategy
- LRU cache for frequent queries
- Embedding cache for common inputs
- Pre-computed token counts
- Cached cluster assignments

## Future Architecture Enhancements

### Planned Improvements

#### Graph Memory Network
- Explicit relationship edges between memories
- Graph traversal for associative recall
- Community detection for topic modeling

#### Distributed Architecture
- Sharding across multiple indices
- Federated search across shards
- Consistent hashing for distribution

#### Advanced Retrieval
- Learned retrieval weights from feedback
- Query expansion with synonyms
- Hierarchical retrieval strategies

#### Multi-Modal Support
- Image memory with CLIP embeddings
- Audio transcription and indexing
- Structured data memories (tables, JSON)

## Monitoring and Debugging

### Metrics Collection
- Request latency histograms
- Memory usage over time
- Retrieval accuracy metrics
- Cluster stability measures

### Debug Interface
- Memory block visualization
- Attention weight display
- Score breakdown analysis
- Query execution plans

### Logging Architecture
```python
{
    "ingestion": "INFO",     # Memory creation events
    "retrieval": "DEBUG",    # Search operations
    "clustering": "INFO",    # Cluster updates
    "api": "INFO",          # API requests
    "llm": "DEBUG"          # LLM interactions
}
```

## Deployment Considerations

### Production Setup
1. Use persistent volume for database
2. Configure appropriate context window
3. Set up log rotation
4. Enable monitoring endpoints
5. Configure backup strategy

### Performance Tuning
1. Adjust retrieval weights for use case
2. Optimize cluster count for data size
3. Tune MMR lambda for diversity needs
4. Configure appropriate token budgets
5. Set decay rates for memory retention

### Scaling Guidelines
- **<10K memories**: Default configuration
- **10K-100K memories**: Increase cache sizes
- **100K-1M memories**: Consider sharding
- **>1M memories**: Distributed architecture

---

*This document provides a comprehensive technical overview of JAM's architecture. For implementation details, see the source code and inline documentation.*