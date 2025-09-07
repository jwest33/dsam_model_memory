# [JAM] Journalistic Agent Memory
> [!NOTE]
> Formerly DSAM (Dual-Space Agentic Memory)

A local-first journalistic memory system that gives LLM agents persistent, searchable memory with human-like organization and recall patterns.

## Core Philosophy

This framework provides agents with a memory system inspired by how humans actually remember: through stories, associations, and patterns rather than perfect recall. Every interaction becomes a memory that can be retrieved, reasoned about, and built upon.

## Key Features

### Intelligent Memory System
- **5W1H Semantic Extraction**: Automatically decomposes events into Who, What, When, Where, Why, and How
- **Hybrid Search**: Combines semantic similarity, lexical matching, recency, and usage patterns
- **Self-Organizing**: Memories cluster naturally based on concepts, time, actors, and context
- **Token-Aware**: Optimally packs memories into LLM context windows using knapsack algorithms

### Web Interface
- **Synthwave Theme**: Professional cyberpunk aesthetics with neon accents (cyan, pink, purple)
- **Chat Interface**: Real-time messaging with loading animations and session management
- **Memory Explorer**: Browse and search through stored memories with 5W1H field display
- **Debug Panel**: View memory blocks used in responses for transparency
- **Responsive Design**: Optimized for desktop and tablet viewing

### Tool Integration
- **Web Search**: Multi-backend search with automatic provider selection (DuckDuckGo, Google, Serper)
- **Tool Memory**: All tool calls and results stored as searchable memories with metadata
- **Import/Export**: Bulk memory management with JSON format support
- **Extensible Framework**: Easy to add new tools via base classes and registration

### Privacy-First
- **100% Local**: Runs entirely on your machine!
- **No Cloud Dependencies**: Your data never leaves your system
- **Open Source**: Full transparency and control

### Liquid Memory Visualization
- **3D Cluster Visualization**: Interactive exploration of memory topology in 3D space
- **Dimensionality Reduction**: PCA, t-SNE, and UMAP for intuitive memory landscape mapping
- **Dynamic Coloring**: Visualize clusters by ID, energy levels, recency, or usage patterns
- **Real-time Updates**: Watch memory clusters flow and reorganize based on access patterns

## The Journal Memory Model

### 5W1H Semantic Normalization

Every piece of information is decomposed into fundamental semantic dimensions:

- **Who**: Actor identification (user, assistant, tool, system)
- **What**: The core content or action
- **When**: Temporal anchoring with decay modeling  
- **Where**: Spatial/contextual location (digital or physical)
- **Why**: Intent and purpose extraction
- **How**: Method, process, or mechanism used

This normalization enables memories to be compared, clustered, and retrieved across different modalities and contexts.

### Liquid Memory Clusters

Memories organize into fluid blocks that reshape based on access patterns:

- **Temporal Clustering**: Recent events naturally group together
- **Conceptual Clustering**: Similar ideas coalesce regardless of time
- **Actor Clustering**: Interactions with the same entities form narrative threads
- **Spatial Clustering**: Events in similar contexts maintain proximity

These clusters emerge from actual usage patterns, creating a self-organizing memory topology.

### Hybrid Retrieval System

Memory retrieval combines multiple cognitive strategies:

1. **Semantic Search** (FAISS vectors): "What's similar in meaning?"
2. **Lexical Search** (SQLite FTS5): "What contains these exact words?"
3. **Recency Bias**: "What happened recently?"
4. **Actor Relevance**: "What involves these participants?"
5. **Spatial Context**: "What happened in similar places?"
6. **Usage Patterns**: "What gets accessed together?"

Each signal contributes to a unified relevance score, mimicking how humans use multiple cues to trigger memories.

## Quick Start

### Prerequisites
- Python 3.11+
- llama.cpp server with a compatible model (e.g., Qwen-4B)
- 8GB+ VRAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/jwest33/agentic_memory.git
cd agentic_memory

# Install dependencies
pip install -r requirements.txt

# Start llama.cpp server (in separate terminal)
# Note: The system expects the server on port 8000
./llama-server -m /path/to/qwen3-4b-instruct.gguf --port 8000 --host 127.0.0.1

# Or use the Python client that can auto-start the server:
python llama_server_client.py

# Launch the web interface
python -m agentic_memory.server.flask_app
```

Open your browser to `http://127.0.0.1:5001` and start chatting!

## Architecture

### Core Components

- **Memory Router**: Central orchestrator handling ingestion, retrieval, and tool execution
- **Storage Layer**: SQLite with FTS5 for full-text search + FAISS for vector similarity
- **Extraction Pipeline**: LLM-powered 5W1H decomposition with structured output
- **Block Builder**: Token-budgeted context packing using knapsack algorithm with MMR diversity
- **Hybrid Retrieval**: Multi-signal scoring combining 6 retrieval strategies
- **Concept Clustering**: MiniBatchKMeans for incremental production-scale clustering
- **Tool System**: Extensible framework with automatic detection and execution
- **Flask Server**: Web interface with WebSocket-style updates and debug transparency

### Tech Stack

- **LLM**: llama.cpp for local inference
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for similarity search
- **Database**: SQLite with FTS5 for full-text search
- **Web Framework**: Flask with Jinja2 templates
- **UI Theme**: Custom synthwave CSS with Michroma font

## Configuration

The system is highly configurable through environment variables:

```bash
# LLM Settings
AM_LLM_BASE_URL=http://localhost:8000/v1  # llama.cpp server URL
AM_LLM_MODEL=Qwen3-4b-instruct-2507       # Model name
AM_CONTEXT_WINDOW=8192                    # Context size
AM_RESERVE_OUTPUT_TOKENS=1024             # Token budget for output
AM_RESERVE_SYSTEM_TOKENS=512              # Token budget for system

# Storage Paths
AM_DB_PATH=./amemory.sqlite3              # SQLite database
AM_INDEX_PATH=./faiss.index               # FAISS index
AM_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Weights (must sum to 1.0)
AM_W_SEMANTIC=0.55   # Semantic similarity weight
AM_W_LEXICAL=0.20    # Lexical match weight  
AM_W_RECENCY=0.10    # Recency bias weight
AM_W_ACTOR=0.07      # Actor relevance weight
AM_W_SPATIAL=0.03    # Spatial proximity weight
AM_W_USAGE=0.05      # Usage pattern weight
AM_MMR_LAMBDA=0.5    # MMR diversity parameter

# Tool Configuration (optional)
SERPER_API_KEY=your_key_here  # For paid web search
SEARCH_API_KEY=your_key_here  # Alternative search API
```

## Advanced Features

### Memory Decay and Reinforcement
- Frequently accessed memories gain priority through usage tracking
- Time-based decay with configurable half-life (default: 30 days)
- Related memories reinforce each other through co-activation patterns
- Attention mechanism tracks which memories contribute to responses

### Token-Aware Context Building
- Treats context as 0-1 knapsack optimization problem
- Maximizes value density (relevance/tokens ratio) with diversity
- MMR (Maximal Marginal Relevance) prevents redundant memories
- Pointer chaining for graceful overflow with continuation markers
- Respects configurable token budgets for system, output, and context

### Tool Memory Integration  
- Tool calls stored with structured JSON metadata
- Results preserved with success/failure states and execution time
- Actor format: `tool:tool_name` for easy filtering
- Event types: `tool_call` and `tool_result` for tracking
- Enables learning from past tool interactions and failures

## License

MIT License - see [LICENSE](LICENSE) for details.
