# SAM - Self-Organizing Agentic Memory

An agentic memory framework supporting dynamic clustering, adaptive embeddings, and unlimited capacity. No arbitrary thresholds, no fixed limits - memories organize themselves based on real-world usage patterns.

## Core Philosophy

Traditional memory systems force artificial constraints: salience thresholds, capacity limits, static relationships. This system removes all arbitrary boundaries, allowing memories to:

- **Self-organize** through dynamic clustering based on query context
- **Evolve** via gravitational embedding updates reflecting co-occurrence
- **Scale infinitely** with ChromaDB backend and no eviction logic
- **Adapt naturally** to usage patterns without manual tuning

## Key Features

### Dynamic Memory Clustering
- **No pre-computed blocks** - Memories cluster dynamically based on query context
- **DBSCAN algorithm** adapts cluster formation to data density
- **Context-aware weighting** emphasizes different 5W1H dimensions per query
- **Eigenvector centrality** determines memory importance within clusters

### Adaptive Embeddings
- **Gravitational updates** - Frequently accessed memories drift closer
- **Momentum-based learning** prevents oscillation (α=0.01, momentum=0.9)
- **Dimension-specific attraction** based on interaction types
- **Co-occurrence tracking** strengthens relationships over time

### Unlimited Capacity
- **ChromaDB backend** scales to millions of memories
- **No eviction** - All memories preserved indefinitely
- **No salience filtering** - Every experience matters
- **Efficient retrieval** through vector similarity search

### 5W1H Framework
Every memory encodes the complete context:
- **Who**: Actor or entity involved
- **What**: Action or observation
- **When**: Temporal information
- **Where**: Location or context
- **Why**: Intent or purpose
- **How**: Method or approach

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agent-memory
cd agent-memory

# Install dependencies
pip install -r requirements.txt

# For Windows users with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install llama.cpp server (optional, for LLM salience)
# Windows:
winget install ggml.llamacpp
# Linux/Mac:
brew install llama.cpp
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

## Use Cases

### Personal AI Assistant
- Remembers all interactions without forgetting
- Contexts naturally emerge from usage patterns
- No manual configuration of importance

### Research Tool
- Stores unlimited research notes and findings
- Discovers unexpected connections through clustering
- Embeddings evolve to reflect discovered relationships

### Game AI
- NPCs with truly persistent memories
- Dynamic relationship formation
- Natural forgetting through embedding drift

### Knowledge Management
- Self-organizing information architecture
- No manual tagging or categorization
- Emergent topics through clustering

## Technical Deep Dive

### Why No Salience Thresholds?

Traditional systems use salience scores to determine what to remember. This creates:
- **Information loss**: Low-salience events might become important later
- **Arbitrary boundaries**: Who decides 0.3 vs 0.4 salience?
- **Static evaluation**: Importance changes with context

Our approach: Store everything, let retrieval determine relevance.

### Why Dynamic Clustering?

Static memory blocks assume fixed relationships. Reality is fluid:
- **Context-dependent**: "debugging" memories cluster differently when querying about "performance" vs "errors"
- **Emergent structure**: Patterns arise from usage, not pre-definition
- **Adaptive boundaries**: Cluster sizes and densities adjust to data

### Why Adaptive Embeddings?

Static embeddings assume fixed semantic space. Real understanding evolves:
- **Usage patterns matter**: Frequently co-accessed memories should be similar
- **Relationships strengthen**: Repeated associations increase attraction
- **Context shapes meaning**: Same memory means different things in different contexts

## Contributing

We welcome contributions! Key areas for enhancement:

1. **Alternative clustering algorithms**: Hierarchical, spectral, affinity propagation
2. **Embedding evolution strategies**: Different gravitational models
3. **Query optimization**: Faster candidate retrieval
4. **Visualization tools**: Memory landscape mapping
5. **Integration examples**: LangChain, AutoGPT, etc.

## Research Foundation

This system builds on several key concepts:

- **Modern Hopfield Networks** (Ramsauer et al., 2021): Attention-based associative memory
- **DBSCAN** (Ester et al., 1996): Density-based spatial clustering
- **Eigenvector Centrality** (Bonacich, 1972): Node importance in networks
- **Momentum SGD** (Polyak, 1964): Accelerated gradient descent
- **5W1H Framework**: Journalistic approach to complete information

## License

MIT License - See LICENSE file for details

## Special Acknowledgments

- ChromaDB
- Qwen model family
- all-MiniLM-L6 model
