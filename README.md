# [JAM] Jouranlistic Agent Memory
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
- **Synthwave Theme**: Synthwave aesthetics with neon colors and gradients
- **Chat Interface**: Real-time messaging with loading animations and session management
- **Memory Explorer**: Browse and search through stored memories with advanced filtering
- **Responsive Design**: Optimized for desktop and tablet viewing

### Tool Integration
- **Web Search**: Multi-backend search with automatic provider selection
- **Tool Memory**: All tool calls and results are stored as searchable memories
- **Extensible Framework**: Easy to add new tools and capabilities

### Privacy-First
- **100% Local**: Runs entirely on your machine!
- **No Cloud Dependencies**: Your data never leaves your system
- **Open Source**: Full transparency and control

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

### Fluid Memory Blocks

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
./llama-server -m models/qwen3-4b-instruct.gguf --port 8000 --host 127.0.0.1

# Or use the Python client that can auto-start the server:
python llama_server_client.py

# Launch the web interface
python -m agentic_memory.server.flask_app
```

Open your browser to `http://127.0.0.1:5001` and start chatting!

## Architecture

### Core Components

- **Memory Router**: Central orchestrator for ingestion and retrieval
- **Storage Layer**: SQLite with FTS5 + FAISS vector index
- **Extraction Pipeline**: LLM-powered 5W1H decomposition
- **Block Builder**: Token-aware context optimization
- **Tool System**: Extensible framework for agent capabilities
- **Flask Server**: Web interface with real-time updates

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
AM_LLM_BASE_URL=http://localhost:8080/v1
AM_CONTEXT_WINDOW=8192

# Storage Paths
AM_DB_PATH=./memory.db
AM_INDEX_PATH=./memory.faiss

# Retrieval Weights
AM_W_SEMANTIC=0.3    # Semantic similarity
AM_W_LEXICAL=0.2     # Exact text matches
AM_W_RECENCY=0.2     # Recent memories
AM_W_ACTOR=0.1       # Actor relevance
AM_W_SPATIAL=0.1     # Spatial context
AM_W_USAGE=0.1       # Usage patterns
```

## Advanced Features

### Memory Decay and Reinforcement
- Frequently accessed memories gain priority
- Unused memories gradually decay in retrieval ranking
- Related memories reinforce each other through co-activation

### Token-Aware Context Building
- Treats context building as an optimization problem
- Uses value density and diversity measures
- Implements pointer chaining for overflow handling
- Adaptive chunking based on retrieval patterns

### Tool Memory Integration
- Tool calls stored as structured events
- Results preserved with success/failure states
- Maintains execution context and reasoning
- Enables learning from past tool interactions

## License

MIT License - see [LICENSE](LICENSE) for details.
