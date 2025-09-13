# JAM (Journalistic Agent Memory)

A local-first memory system for LLM agents that provides persistent, searchable memory using journalistic 5W1H (Who, What, When, Where, Why, How) semantic extraction.

## Features

- **5W1H Memory Extraction**: Automatically decomposes events into journalistic dimensions
- **Hybrid Retrieval**: Combines 7 retrieval strategies (semantic, lexical, recency, actor, temporal, spatial, usage)
- **Local-First**: Runs entirely on your machine using llama.cpp
- **Token-Optimized**: Smart memory selection using knapsack algorithm for optimal context usage
- **Web Interface**: Synthwave-themed UI for memory browsing and search
- **3D Visualization**: Interactive memory topology using PCA/t-SNE/UMAP
- **Production Ready**: 5000+ memories in production database

## Quick Start

### Prerequisites

- Python 3.11+
- A GGUF model file (e.g., Qwen3-4b-instruct)
- 8GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jam_model_memory.git
cd jam_model_memory

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install document parsing dependencies
pip install -r requirements-optional.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your model path:
```
AM_MODEL_PATH=/path/to/your/model.gguf
```

### Running the Application

Start all servers with the CLI:

```bash
python -m agentic_memory.cli server start --all --daemon
```

The application will be available at:
- Web Interface: http://localhost:5001
- API: http://localhost:8001
- LLM Server: http://localhost:8000

## Usage

### Web Interface

Navigate to http://localhost:5001 to access:

- **Analyzer** (`/`): Advanced search with token-based memory selection
- **Browser** (`/browser`): Browse all memories with filtering
- **Analytics** (`/analytics`): Statistics and visualizations
- **3D Visualization** (`/visualize`): Interactive memory topology
- **Configuration** (`/config`): Runtime settings management

### CLI Commands

```bash
# Memory operations
python -m agentic_memory.cli memory add "Meeting with John about project X"
python -m agentic_memory.cli memory search "project X"
python -m agentic_memory.cli memory stats

# Server management
python -m agentic_memory.cli server status
python -m agentic_memory.cli server restart
python -m agentic_memory.cli server stop --all

# API testing
python -m agentic_memory.cli api chat "What do you remember about project X?"
```

### Python API

```python
from agentic_memory import MemoryRouter

# Initialize the memory system
router = MemoryRouter()

# Add a memory
await router.ingest("Had a productive meeting with the team about the new feature")

# Search memories
results = await router.retrieve("team meetings", k=10)

# Chat with memory context
response = await router.chat("What meetings have we had recently?")
```

## Architecture

### Core Components

- **MemoryRouter**: Central orchestrator for memory operations
- **HybridRetriever**: Multi-signal scoring combining 7 retrieval strategies
- **BlockBuilder**: Token-budgeted context packing using knapsack algorithm
- **LLMExtractor**: Extracts 5W1H dimensions from text
- **MemoryStore**: SQLite with FTS5 for full-text search
- **FaissIndex**: Vector similarity search

### Key Features

- **Attention-Based Retrieval**: Adaptive embeddings based on query context
- **Liquid Memory Clusters**: Self-organizing concept clusters
- **Multi-Part Extraction**: Breaks complex events into multiple memories
- **Token Optimization**: Maximizes retrieval utility within context limits

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_memory

# Run specific test file
pytest tests/test_integration.py -v
```

### Project Structure

```
jam_model_memory/
├── agentic_memory/          # Core memory system
│   ├── extraction/          # 5W1H extraction pipeline
│   ├── storage/             # Database and vector stores
│   ├── server/              # Web interface
│   ├── cli.py               # Command-line interface
│   └── router.py            # Main orchestrator
├── tests/                   # Test suite
└── llama_server_client.py  # LLM server management
```

## Configuration Options

### Essential Settings

- `AM_MODEL_PATH`: Path to GGUF model file (required)
- `AM_CONTEXT_WINDOW`: Context size (default: 8192)
- `AM_DB_PATH`: SQLite database path (default: ./amemory.sqlite3)

### Advanced Settings

- `AM_USE_ATTENTION`: Enable attention-based retrieval (default: true)
- `AM_USE_LIQUID_CLUSTERS`: Enable self-organizing clusters (default: true)
- `AM_USE_MULTI_PART`: Enable multi-part extraction (default: true)

See `.env.example` for all available options.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything passes
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with llama.cpp for local LLM inference
- Uses FAISS for efficient vector search
- Inspired by journalistic principles of information organization
