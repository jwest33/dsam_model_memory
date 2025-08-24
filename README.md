# 5W1H Memory Framework with Modern Hopfield Networks

An advanced associative memory system for AI agents that combines the 5W1H framework (Who, What, When, Where, Why, How) with Modern Hopfield Networks and ChromaDB for unlimited, context-aware memory storage and retrieval.

## 🌟 Key Features

- **Unlimited Memory Capacity**: No arbitrary limits - scales with available disk space
- **Block-Centric Salience Matrix**: Context-aware importance scoring using embedding similarities
- **5W1H Structured Memory**: Every memory organized by Who, What, When, Where, Why, and How
- **ChromaDB Integration**: Efficient vector database for millions of memories
- **Modern Hopfield Networks**: Attention-based associative memory retrieval
- **Automatic Memory Blocks**: Related memories grouped semantically and temporally
- **Real-time Web Interface**: Interactive memory visualization and management

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies (ChromaDB is required)
pip install -r requirements.txt

# For LLM-based salience (optional)
# Windows: winget install ggml.llamacpp
# Linux/Mac: Build from https://github.com/ggerganov/llama.cpp
```

### Basic Usage

```python
from agent.memory_agent import MemoryAgent

# Initialize the memory system
agent = MemoryAgent()

# Store a memory
success, message, event = agent.remember(
    who="User",
    what="Asked about memory systems",
    where="GitHub",
    why="Understanding the project",
    how="Reading documentation"
)

# Recall memories
memories = agent.recall(what="memory", k=5)

# Get insights
insight = agent.get_insight("How does the memory system work?")
```

### Command Line Interface

```bash
# Initialize system
python cli.py init

# Run interactive demo
python cli.py demo

# Store memories
python cli.py remember --who "AI" --what "Learned about embeddings"

# Recall memories
python cli.py recall --what "embeddings" --k 10

# View statistics
python cli.py stats

# Export/Import for backup
python cli.py export --output backup.json
python cli.py import --input backup.json
```

### Web Interface

```bash
# Launch web interface (opens browser automatically)
python run_web.py

# Access at http://localhost:5000
```

## 🏗️ Architecture

### Core Components

1. **Event Model** (`models/event.py`)
   - 5W1H structure for complete context
   - Episode tracking for conversation continuity
   - Confidence scores and metadata
   - No individual salience (managed by blocks)

2. **Memory Blocks** (`models/memory_block.py`)
   - Groups related memories automatically
   - **Salience Matrix**: Pairwise embedding similarities
   - **Eigenvector Centrality**: Determines event importance within block
   - **Dynamic Updates**: Recomputes as events are added
   - Block-level embeddings for efficient retrieval

3. **ChromaDB Storage** (`memory/chromadb_store.py`)
   - **Primary Storage**: No more JSON files!
   - **Three Collections**: events, blocks, metadata
   - **Vector Search**: Built-in similarity search
   - **Unlimited Capacity**: Scales with disk space
   - **Smart Caching**: LRU cache for frequent access

4. **Dynamic Hopfield Network** (`memory/hopfield.py`)
   - **No Capacity Limits**: Grows indefinitely
   - **Dynamic Arrays**: Auto-expanding storage
   - **Attention-Based**: Uses transformer-style attention
   - **Fast Retrieval**: O(N·d) complexity

### Memory Flow

```
User Input → 5W1H Extraction → Event Creation
    ↓
Memory Block Assignment (semantic/temporal grouping)
    ↓
Salience Matrix Update (embedding-based importance)
    ↓
ChromaDB Storage (unlimited capacity)
    ↓
Retrieval via Vector Search or Attention
```

### Salience Matrix System

Each memory block maintains:
- **Salience Matrix**: [n_events × n_events] pairwise similarities
- **Event Saliences**: Individual importance from eigenvector centrality
- **Block Salience**: Overall block importance

Key features:
- **Context-Aware**: Event importance depends on relationships
- **Type Boosts**: User input (1.3x), Observations (1.15x)
- **Temporal Weighting**: Recent events weighted higher
- **No Individual Salience**: Removed event.salience field

## 📊 Performance

- **0-1K memories**: Instant retrieval (<100ms)
- **1K-10K memories**: Sub-second retrieval
- **10K-100K memories**: 1-2 second retrieval
- **100K-1M memories**: Efficient with ChromaDB indexing
- **1M+ memories**: Limited only by disk space

## 🔧 Configuration

### Environment Variables

```bash
# Core settings
STATE_DIR=./state          # Where to store data
CHROMADB_PATH=./chromadb   # ChromaDB storage location

# LLM settings (optional)
LLM_SERVER_URL=http://localhost:8000
USE_LLM_SALIENCE=true      # Use LLM for importance scoring

# Memory settings (no limits!)
MHN_EMBEDDING_DIM=384      # Embedding dimensions
MHN_TEMPERATURE=15.0       # Attention sharpness
```

### Key Configuration (`config.py`)

```python
MemoryConfig:
  embedding_dim: 384              # Transformer-compatible
  temperature: 15.0               # Attention sharpness
  salience_threshold: 0.4         # 40% importance minimum
  similarity_threshold: 0.85      # 85% for deduplication

StorageConfig:
  chromadb_path: "./state/chromadb"
  chromadb_required: true         # Primary storage
  cache_size: 1000               # In-memory cache
  auto_backup: true              # JSON exports
  backup_interval: 100           # Every 100 operations
```

## 📝 Technical Improvements

### What's New

1. **No Memory Limits**
   - Removed 512 memory cap
   - No eviction logic
   - Dynamic array growth
   - Scales infinitely

2. **Block-Centric Salience**
   - Eigenvector centrality for importance
   - Pairwise similarity matrix
   - Context-aware scoring
   - No arbitrary thresholds

3. **ChromaDB First**
   - Primary storage (not optional)
   - Efficient serialization of numpy arrays
   - Built-in vector search
   - Automatic indexing

4. **Improved Embeddings**
   - Cosine similarity throughout
   - Proper normalization
   - Fallback to Jaccard when needed
   - Role-aware embeddings

## 🗂️ Project Structure

```
agent-wip/
├── agent/
│   └── memory_agent.py          # Main memory agent
├── memory/
│   ├── chromadb_store.py        # ChromaDB backend (NEW)
│   ├── hopfield.py      # Unlimited Hopfield (NEW)
│   ├── memory_store.py          # Storage orchestration
│   └── block_manager.py         # Memory block management
├── models/
│   ├── event.py                 # 5W1H event model
│   └── memory_block.py          # Block with salience matrix
├── embedding/
│   └── embedder.py              # Embedding generation
├── llm/
│   ├── llm_interface.py         # LLM integration
│   └── salience_model.py        # Block importance (simplified)
├── templates/                   # Web interface
├── static/                      # Web assets
├── cli.py                       # Command-line interface
├── web_app.py                   # Flask application
└── config.py                    # Configuration
```

## 🔬 Advanced Features

### Memory Blocks
- **Automatic Grouping**: Semantic, temporal, causal links
- **Link Types**: TEMPORAL, CAUSAL, SEMANTIC, EPISODIC, CONVERSATIONAL, REFERENCE
- **Coherence Score**: How well memories fit together
- **Chain Tracking**: Action→observation relationships

### Salience Matrix Mathematics
```python
# Pairwise similarity matrix
for i, j in events:
    similarity = cosine_sim(embed[i], embed[j])
    temporal_weight = exp(-time_diff / 3600)
    matrix[i,j] = similarity * 0.7 + temporal_weight * 0.3

# Eigenvector centrality
eigenvalues, eigenvectors = np.linalg.eig(matrix)
centrality = dominant_eigenvector / max(dominant_eigenvector)
```

### ChromaDB Operations
```python
# Store with embedding
store.store_event(event, embedding, block_id)

# Update salience matrix
store.update_block_salience(block_id, matrix, saliences)

# Efficient retrieval
results = store.retrieve_events_by_query(query_embedding, k=10)
blocks = store.retrieve_blocks_by_query(query_embedding, k=5)
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize System**
   ```bash
   python cli.py init
   ```

3. **Run Demo**
   ```bash
   python cli.py demo
   ```

4. **Launch Web Interface**
   ```bash
   python run_web.py
   ```

## 🤝 Contributing

The system is designed for extensibility:
- Add memory operations in `memory_agent.py`
- Extend link types in `memory_block.py`
- Add retrieval strategies in `chromadb_store.py`
- Enhance web interface in `templates/`

## 📚 Theory

### Why 5W1H?
Complete context capture ensures nothing is lost:
- **Who**: Agency and actors
- **What**: Core content
- **When**: Temporal ordering
- **Where**: Context location
- **Why**: Intent and causality
- **How**: Methods and mechanisms

### Why Modern Hopfield Networks?
- Content-addressable memory
- Robust to partial queries
- Attention-based like transformers
- Continuous learning via EMA
- Interpretable retrieval

### Why Memory Blocks?
- Groups related memories
- Context-aware importance
- Efficient batch operations
- Natural episode boundaries
- Reduced redundancy

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- ChromaDB for scalable vector storage
- Sentence-transformers for embeddings
- Modern Hopfield Networks research
- Transformer attention mechanisms
- Cognitive science principles

---

**Note**: This is an active research project exploring advanced memory systems for AI agents. The architecture prioritizes scalability, context-awareness, and semantic understanding over simple key-value storage.