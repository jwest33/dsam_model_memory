# [SAM] Self-Organizing Agentic Memory

A memory system for AI agents featuring unlimited storage capacity, and query-driven self-organizing clustering via adaptive embeddings.

## Overview

This system implements a memory architecture for AI agents that eliminates traditional constraints such as fixed capacity limits and salience thresholds. Memory organization occurs dynamically at query time through context-aware clustering, while embeddings evolve based on usage patterns.

## Key Features

### Dynamic Memory Clustering
- Query-driven clustering using DBSCAN algorithm
- Context-aware field weighting based on query parameters
- Eigenvector centrality for determining memory importance
- No pre-computed memory blocks

### Adaptive Embeddings
- Gravitational updates between frequently co-accessed memories
- Momentum-based learning (α=0.01, momentum=0.9)
- Dimension-specific evolution tracking
- Co-occurrence-based relationship strengthening

### Storage Architecture
- ChromaDB backend for scalable vector storage
- No memory eviction or capacity limits
- All memories stored without salience filtering
- Efficient similarity-based retrieval

### 5W1H Framework
Complete context encoding for each memory:
- Who: Entity or actor
- What: Action or observation
- When: Temporal information
- Where: Location or context
- Why: Purpose or intent
- How: Method or approach

## Installation

```bash
# Clone repository
git clone <repository-url>
cd agent-wip

# Install dependencies
pip install -r requirements.txt

# For Windows users with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Basic Usage

```python
from agent.memory_agent import MemoryAgent

# Initialize the memory system
agent = MemoryAgent()

# Store a memory
success, message, event = agent.remember(
    who="Alice",
    what="implemented dynamic clustering algorithm",
    where="memory/dynamic_clustering.py",
    why="remove arbitrary thresholds",
    how="DBSCAN with eigenvector centrality"
)

# Recall memories with dynamic clustering
memories = agent.recall(what="clustering", k=5)
for event, score in memories:
    print(f"[{score:.3f}] {event.five_w1h.what}")

# Provide feedback to adapt embeddings
agent.adapt_from_feedback(
    query={"what": "clustering"},
    positive_events=[memories[0][0].id],  # Most relevant
    negative_events=[]
)

# Get insights about memory relationships
insights = agent.get_insights(topic="clustering")
print(insights)
```

### Command Line Interface

```bash
# Initialize system
python cli.py init

# Store memories
python cli.py remember --who "User" --what "asked about memory systems" \
                      --why "understand architecture" --how "questioning"

# Recall with dynamic clustering
python cli.py recall --what "memory" --k 10

# View system statistics
python cli.py stats

# Generate sample conversations
python generate_conversations.py
```

### Web Interface

```bash
# Launch web UI (auto-opens browser)
python run_web.py

# Access at http://localhost:5000
# Features: Chat interface, memory visualization, block management
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     User Input (5W1H)                    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Memory Agent                          │
│  - Episode management                                    │
│  - Event creation                                        │
│  - Query processing                                      │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                Dynamic Memory Store                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  ChromaDB    │  │   Dynamic    │  │   Adaptive   │ │
│  │   Storage    │◄─┤  Clustering  │◄─┤  Embeddings  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Processing**: 5W1H structure → Event object with metadata
2. **Storage**: Direct to ChromaDB, no filtering or thresholds
3. **Query Time**:
   - Retrieve candidates via vector similarity
   - Dynamic clustering with DBSCAN
   - Eigenvector centrality ranking
   - Adaptive embedding updates
4. **Feedback Loop**: User interactions update embeddings gravitationally

### Key Algorithms

#### Dynamic Clustering (DBSCAN)
```python
# Pseudocode
candidates = chromadb.query(embedding, k=100)
distance_matrix = compute_similarities(candidates)
clusters = DBSCAN(eps=adaptive, min_samples=2).fit(distance_matrix)
```

#### Gravitational Embeddings
```python
# Pseudocode
force = (target_embedding - source_embedding) * relevance
velocity = momentum * velocity + learning_rate * force
new_embedding = normalize(embedding + velocity)
```

#### Eigenvector Centrality
```python
# Pseudocode
adjacency = similarity_matrix > threshold
eigenvalues, eigenvectors = np.linalg.eig(adjacency)
centrality = eigenvectors[:, 0]  # Principal eigenvector
```

## Configuration

### Environment Variables

```bash
# Core settings
STATE_DIR=./state                    # Memory storage location
USE_CHROMADB=true                   # Enable vector database
USE_LLM_SALIENCE=false              # Disable LLM scoring (no thresholds anyway)

# Embedding settings
EMBEDDING_MODEL=all-MiniLM-L6-v2    # Sentence transformer model
EMBEDDING_DIM=384                   # Embedding dimensions

# Clustering parameters
DBSCAN_EPS=0.3                      # Cluster density threshold
MIN_CLUSTER_SIZE=2                  # Minimum cluster size

# Adaptive embedding parameters
LEARNING_RATE=0.01                  # Embedding update rate
MOMENTUM=0.9                        # Update smoothing factor
```

### Advanced Configuration

```python
# config.py
@dataclass
class DynamicMemoryConfig:
    # Clustering
    max_clusters: int = 10
    cluster_merge_threshold: float = 0.7
    
    # Embeddings
    gravity_strength: float = 0.1
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'who': 1.0, 'what': 2.0, 'when': 0.5,
        'where': 0.5, 'why': 1.5, 'how': 1.0
    })
    
    # ChromaDB
    collection_name: str = "events"
    persist_directory: str = "./state/chromadb"
```

## System Statistics

The system tracks comprehensive metrics:

```python
stats = agent.get_statistics()
# Returns:
{
    'total_events': 1543,
    'total_queries': 89,
    'episodes': 12,
    'evolved_embeddings': 743,
    'average_cluster_size': 4.2,
    'embedding_drift': 0.15,  # Average cosine distance from original
    'chromadb_stats': {
        'events': 1543,
        'blocks': 0,  # No static blocks in new system
        'collection_size_mb': 12.4
    }
}
```

## Testing

```bash
# Run demonstration
python generate_conversations.py

# Test without external dependencies
USE_CHROMADB=false python cli.py demo

# Test API endpoints
python test_api.py

# Validate clustering
python -c "from memory.dynamic_clustering import test_clustering; test_clustering()"

# Check embedding evolution
python -c "from memory.adaptive_embeddings import visualize_evolution; visualize_evolution()"
```

## Applications

### Personal AI Assistant
- Persistent memory of all interactions
- Context emergence from usage patterns
- Automatic importance determination

### Research Tool
- Unlimited storage for research data
- Connection discovery through clustering
- Relationship evolution through embedding adaptation

### Game AI
- Persistent NPC memory systems
- Dynamic relationship formation
- Natural memory drift over time

### Knowledge Management
- Self-organizing information structure
- Automatic categorization through clustering
- Topic emergence from usage patterns

## Technical Architecture

### Storage Without Thresholds

The system stores all memories without salience filtering:
- Information that appears unimportant may become relevant in future contexts
- Importance determination occurs at retrieval time based on query context
- No arbitrary threshold values required

### Dynamic Clustering Approach

Clustering occurs at query time rather than using pre-computed blocks:
- Context-dependent grouping based on query parameters
- Emergent structure from data patterns
- Adaptive cluster boundaries based on data density

### Evolving Embeddings

Embeddings adapt based on usage patterns:
- Co-accessed memories gravitate together in embedding space
- Relationship strength increases with repeated associations
- Context-specific meaning evolution

## Development

Potential areas for enhancement:

1. Alternative clustering algorithms (hierarchical, spectral, affinity propagation)
2. Additional embedding evolution strategies
3. Query optimization for faster retrieval
4. Enhanced visualization tools
5. Integration with existing frameworks

## Research Foundation

This system builds on several key concepts:

- **Modern Hopfield Networks** (Ramsauer et al., 2021): Attention-based associative memory
- **DBSCAN** (Ester et al., 1996): Density-based spatial clustering
- **Eigenvector Centrality** (Bonacich, 1972): Node importance in networks
- **Momentum SGD** (Polyak, 1964): Accelerated gradient descent
- **5W1H Framework**: Journalistic approach to complete information

## License

MIT License - See LICENSE file for details

## Dependencies

- ChromaDB for vector storage
- Sentence Transformers for embeddings
- NumPy for numerical operations
- Streamlit for web interface
