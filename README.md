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

# Start all servers with the CLI (recommended)
python -m agentic_memory.cli server start --all --daemon

# Or start services individually:
# 1. Start llama.cpp server
python llama_server_manager.py both start

# 2. Start API wrapper (port 8001)
python llama_api.py start

# 3. Launch web interface (port 5001)
python -m agentic_memory.server.flask_app

# Check server status
python -m agentic_memory.cli server status
```

Open your browser to `http://127.0.0.1:5001` for the web interface, or use the API at `http://127.0.0.1:8001`.

## Architecture

### Core Components

- **Memory Router**: Central orchestrator handling ingestion, retrieval, and tool execution
- **Storage Layer**: SQLite with FTS5 for full-text search + FAISS for vector similarity
- **Extraction Pipeline**: LLM-powered 5W1H decomposition with structured output
- **Block Builder**: Token-budgeted context packing using knapsack algorithm with MMR diversity
- **Hybrid Retrieval**: Multi-signal scoring combining 6 retrieval strategies
- **Concept Clustering**: MiniBatchKMeans for incremental production-scale clustering
- **Tool System**: Extensible framework with automatic detection and execution
- **API Wrapper**: FastAPI server providing OpenAI-compatible endpoints with caching and metrics
- **Flask Server**: Web interface with WebSocket-style updates and debug transparency
- **CLI Tool**: Unified command-line interface for server management and memory operations

### Tech Stack

- **LLM**: llama.cpp for local inference
- **Embeddings**: Llama.cpp server (using same model for LLM and embeddings)
- **Vector Store**: FAISS for similarity search
- **Database**: SQLite with FTS5 for full-text search
- **API Framework**: FastAPI with automatic OpenAPI documentation
- **Web Framework**: Flask with Jinja2 templates
- **CLI Framework**: Click for command-line interface
- **UI Theme**: Custom synthwave CSS with Michroma font

## Configuration

Configuration can be managed through:
1. **Web UI**: Visit `http://localhost:5001/config` for runtime configuration
2. **Environment Variables**: Copy `.env.example` to `.env` and customize
3. **ConfigManager**: Provides all defaults automatically

### Key Settings

```bash
# Model Configuration (required)
AM_MODEL_PATH=/path/to/model.gguf         # Path to GGUF model file
AM_LLM_MODEL=Qwen3-4b-instruct-2507       # Model name/alias

# Server Ports
AM_WEB_PORT=5001                          # Web interface port
AM_API_PORT=8001                          # API wrapper port
AM_LLAMA_PORT=8000                        # llama.cpp server port

# Memory System
AM_DB_PATH=./amemory.sqlite3              # SQLite database
AM_INDEX_PATH=./faiss.index               # FAISS index
AM_CONTEXT_WINDOW=8192                    # Context size

# Advanced Features (all optional)
AM_USE_ATTENTION=true                     # Attention-based retrieval
AM_USE_LIQUID_CLUSTERS=true               # Self-organizing clusters
AM_USE_MULTI_PART=true                    # Multi-part extraction

# Retrieval Weights (when attention disabled)
AM_W_SEMANTIC=0.55   # Semantic similarity
AM_W_LEXICAL=0.20    # Keyword matching
AM_W_RECENCY=0.10    # Recency bias
# ... see .env.example for complete list
```

See `.env.example` for all available settings with documentation.

## Advanced Features

### API Wrapper
- **OpenAI-Compatible**: Drop-in replacement for OpenAI API clients
- **Request Caching**: Reduces redundant LLM calls with TTL-based cache
- **Metrics & Health**: Real-time request counting and health monitoring
- **Background Mode**: Run as daemon process with PID management
- **Admin Endpoints**: Remote server restart and management
- **CORS Support**: Configurable cross-origin resource sharing

### CLI Tools
```bash
# Server management
python -m agentic_memory.cli server start --all --daemon  # Start all servers
python -m agentic_memory.cli server stop --api           # Stop specific server
python -m agentic_memory.cli server status               # Check server status
python -m agentic_memory.cli server restart              # Restart all servers

# Memory operations
python -m agentic_memory.cli memory add "Meeting at 3pm with team"
python -m agentic_memory.cli memory search "meeting" --limit 10
python -m agentic_memory.cli memory stats                # Show statistics

# API testing
python -m agentic_memory.cli api complete "Tell me a joke"
python -m agentic_memory.cli api chat "Hello" --system "You are helpful"
python -m agentic_memory.cli api health                  # Check API health

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
