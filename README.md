# 5W1H + Modern Hopfield Network Memory Framework

A sophisticated associative memory system that stores and retrieves experiences using the 5W1H framework (Who, What, When, Where, Why, How) combined with Modern Hopfield Networks for content-addressable memory with attention-based retrieval.

## Overview

This framework provides an intelligent memory system for AI agents that mimics human episodic memory. Every interaction, observation, and action is decomposed into a structured 5W1H format and stored in a dual-layer memory architecture. The system uses Modern Hopfield Networks (MHN) - an attention-based extension of classical Hopfield networks - to enable associative retrieval similar to how transformer models process information.

### Key Capabilities

- **Structured Memory Storage**: Every event is captured as a complete 5W1H tuple, ensuring no context is lost
- **Associative Retrieval**: Query with partial information and retrieve complete memories
- **Causal Linking**: Automatically chains actions with their observations to track cause-effect relationships
- **Salience-Based Filtering**: Intelligently determines which memories are worth preserving long-term
- **Episode Management**: Groups related events into episodes for temporal coherence
- **LLM Integration**: Leverages language models for salience computation, memory completion, and summarization

## Architecture

### Core Components

#### 1. Event Model (`models/event.py`)
The fundamental unit of memory is an `Event` containing:
- **5W1H Structure**: Complete contextual information about what happened
- **Episode Linking**: Groups related events together temporally
- **Metadata**: Salience scores, confidence levels, access tracking
- **Event Types**: Actions, Observations, User Input, System Events

```python
Event(
    five_w1h=FiveW1H(
        who="LLM",                          # Agent/actor
        what="Generated SQL query",         # Action/content  
        when="2025-01-23T10:15:32Z",       # Timestamp
        where="conversation:chat42",        # Context
        why="User requested data",          # Intent
        how="SQL generation module"          # Mechanism
    ),
    episode_id="e8937",                    # Links related events
    salience=0.85,                         # Importance score
    event_type=EventType.ACTION
)
```

#### 2. Modern Hopfield Network (`memory/hopfield.py`)
Implements attention-based associative memory:

**Mathematical Foundation:**
```
Memory Read: output = softmax(β · q·K^T) @ V
```
Where:
- `q`: Query vector (embedded search query)
- `K`: Key matrix (stored query embeddings) 
- `V`: Value matrix (stored observation embeddings)
- `β`: Inverse temperature parameter (controls attention sharpness)

**Key Features:**
- Capacity: 512 memory slots (configurable)
- Embedding dimension: 384 (matches small transformer models)
- Temperature β: 15.0 (higher = sharper attention focus)
- Update mechanism: Exponential Moving Average (EMA) for similar memories
- Eviction: Priority-based when at capacity

#### 3. Dual Storage System (`memory/memory_store.py`)

**Raw Memory Store:**
- Stores every single event without filtering
- Complete audit trail for debugging and analysis
- Persistent storage via JSON or ChromaDB
- No deduplication or compression

**Processed Memory Store:**
- Salience-filtered (threshold: 0.3 default)
- Deduplicated using similarity matching
- Stored in Modern Hopfield Network for fast associative retrieval
- ChromaDB indexing for semantic search

#### 4. Embedding System (`embedding/embedder.py`)

**Multi-Model Support:**
- Primary: Sentence transformers (all-MiniLM-L6-v2)
- Fallback: Deterministic hash-based embeddings
- Role-aware embeddings: Distinguishes "who" from "what" using role vectors

**5W1H Embedding Process:**
1. Each slot (who, what, when, etc.) is embedded separately
2. Role vectors are added to maintain semantic distinction
3. Slots are fused into a single key vector for retrieval
4. Content (what) becomes the value vector

#### 5. Salience Model (`llm/salience_model.py`)

Determines memory importance through multi-factor analysis:

**LLM-Based Computation:**
- Analyzes relevance to current goal
- Evaluates information novelty
- Considers potential future utility
- Returns score 0-1

**Heuristic Fallback:**
```python
salience = 0.4*novelty + 0.3*(1-overlap) + 0.2*length_factor + 0.1*keyword_boost
```

#### 6. LLM Interface (`llm/llm_interface.py`)

Integrates with llama.cpp server for:
- Salience scoring of new memories
- Completing partial 5W1H structures
- Analyzing causal relationships
- Generating episode summaries
- Suggesting relevant tags

## Memory Operations

### Storing Memories

```python
from agent.memory_agent import MemoryAgent

agent = MemoryAgent()
agent.set_goal("Build a data analysis pipeline")

# Store an action
success, message, event = agent.remember(
    who="LLM",
    what="Generated Python code for data preprocessing",
    where="jupyter:notebook",
    why="User needs to clean dataset",
    how="Code generation with pandas",
    event_type="action",
    tags=["python", "data-cleaning"]
)

# Record the observation (result)
success, message, observation = agent.observe(
    what="Code executed successfully, processed 10,000 rows",
    who="Python kernel",
    where="jupyter:cell_5"
)

# Link action to observation
agent.chain_events(event, observation)
```

### Retrieving Memories

```python
# Query with partial 5W1H
results = agent.recall(
    what="data",      # Search in 'what' field
    who="LLM",        # Filter by actor
    k=5               # Return top 5 matches
)

for event, similarity_score in results:
    print(f"[{similarity_score:.3f}] {event.five_w1h.what}")

# Retrieve full episode
episode_events = agent.get_episode()  # Current episode
for event in episode_events:
    print(f"{event.five_w1h.who}: {event.five_w1h.what}")

# Get causal chains
chains = agent.get_causal_chain(event.id)
for action, observation in chains:
    print(f"Action: {action.five_w1h.what}")
    print(f"Result: {observation.five_w1h.what}")
```

### Memory Completion

```python
# Complete missing information using LLM
partial = {
    "who": "database",
    "what": "returned error",
    "when": "",  # Missing
    "where": "", # Missing
    "why": "",   # Missing
    "how": "SQL execution"
}

completed = agent.complete_memory(partial, context="During data migration")
# LLM fills in missing slots based on context
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Memory Network Settings
MHN_MAX_MEMORY=512              # Maximum memory slots
MHN_EMBEDDING_DIM=384           # Embedding dimension
MHN_TEMPERATURE=15.0            # Attention temperature

# LLM Server (uses existing llama_server_client.py)
LLM_SERVER_URL=http://localhost:8000
LLM_MODEL_PATH=C:\models\your-model.gguf

# Storage
STATE_DIR=./state               # Where to store memories
USE_CHROMADB=true              # Enable vector database
```

### Configuration Structure (`config.py`)

```python
Config:
├── MemoryConfig
│   ├── max_memory_slots: 512
│   ├── embedding_dim: 384
│   ├── temperature: 15.0
│   ├── salience_threshold: 0.3
│   └── similarity_threshold: 0.8
├── StorageConfig
│   ├── use_chromadb: true
│   ├── auto_save: true
│   └── save_interval: 100
├── EmbeddingConfig
│   ├── model_name: "sentence-transformers/all-MiniLM-L6-v2"
│   └── add_role_embeddings: true
├── LLMConfig
│   ├── server_url: "http://localhost:8000"
│   └── use_llm_salience: true
└── AgentConfig
    ├── auto_link_episodes: true
    └── episode_timeout: 300
```

## Command Line Interface

The system includes a comprehensive CLI for testing and interaction:

### Basic Commands

```bash
# Initialize memory system
python cli.py init --path ./my_memories

# Store a memory
python cli.py remember \
    --who "User" \
    --what "Requested data analysis" \
    --where "slack:#data-team" \
    --why "Monthly report needed" \
    --how "Text message" \
    --type user_input \
    --tags "analysis,report"

# Recall memories
python cli.py recall --what "analysis" --k 10
python cli.py recall --who "User" --include-raw

# Record observations
python cli.py observe \
    --what "Analysis completed with 95% accuracy" \
    --who "ML Pipeline"

# View episodes
python cli.py episode                  # Current episode
python cli.py episode --id e8937       # Specific episode

# System management
python cli.py stats                    # View statistics
python cli.py save                     # Manual save
python cli.py clear --confirm          # Clear all memories
python cli.py goal "Learn from user interactions"
```

### Running the Demo

```bash
python cli.py demo
```

This runs a complete demonstration showing:
1. Storing action and observation memories
2. Linking events causally
3. Retrieving memories by query
4. Episode management
5. System statistics

## Installation

### Requirements

```bash
# Install Python dependencies
pip install -r requirements.txt

# Core dependencies:
# - numpy: Matrix operations for Hopfield network
# - requests, psutil: LLM server communication
# - sentence-transformers: Text embeddings (optional but recommended)
# - chromadb: Vector database (optional but recommended)
```

### LLM Server Setup

The system uses your existing `llama_server_client.py` which manages the llama.cpp server:

```bash
# Windows: Install llama.cpp
winget install ggml.llamacpp

# Configure in config/llm_server.env
LLM_MODEL_PATH=C:\models\your-model.gguf
LLM_PORT=8000
LLM_CONTEXT_SIZE=10000

# Server auto-starts when needed
```

## Usage Examples

### Example 1: Building a Knowledge Base

```python
from agent.memory_agent import MemoryAgent

agent = MemoryAgent()
agent.set_goal("Build comprehensive knowledge base")

# Store various types of information
agent.remember(
    who="Documentation",
    what="API endpoint /users returns user list",
    where="docs:api-reference",
    why="API documentation",
    how="Markdown file",
    tags=["api", "users", "reference"]
)

agent.remember(
    who="Developer",
    what="Fixed bug in user authentication",
    where="github:PR-1234",
    why="Security vulnerability",
    how="Code patch",
    tags=["security", "bugfix", "auth"]
)

# Later, query the knowledge
results = agent.recall(what="user", k=10)
```

### Example 2: Tracking Conversations

```python
# Start a new conversation episode
agent.start_episode("conversation_001")

# Track the flow
agent.remember(
    who="User",
    what="How do I process CSV files?",
    event_type="user_input"
)

agent.remember(
    who="Assistant",
    what="You can use pandas.read_csv() function",
    why="Answering user question about CSV",
    event_type="action"
)

agent.observe(
    what="User indicated understanding",
    who="User"
)

# Get conversation summary
summary = agent.summarize_episode("conversation_001")
```

### Example 3: Debugging with Causal Chains

```python
# Track a series of operations
action1 = agent.remember(
    who="System",
    what="Initiated database backup",
    why="Scheduled maintenance"
)

obs1 = agent.observe(
    what="Backup failed: insufficient disk space"
)

agent.chain_events(action1, obs1)

# Later, investigate failures
chains = agent.get_causal_chain(obs1.id)
for action, observation in chains:
    print(f"Tried: {action.five_w1h.what}")
    print(f"Result: {observation.five_w1h.what}")
```

## Advanced Features

### Memory Deduplication

The system automatically detects and merges similar memories:
```python
# Similarity computation:
combined_sim = 0.7 * cosine_sim(keys) + 0.3 * cosine_sim(values)

# If similarity > 0.8, memories are merged using EMA:
merged_key = (1 - α) * old_key + α * new_key
```

### Priority-Based Eviction

When memory is full, least important memories are evicted:
```python
priority_score = 1.2*(1-salience) + 0.6*(1-usage_rate) + 0.6*age_normalized
# Higher score = more likely to evict
```

### Episode Management

Episodes automatically timeout after 5 minutes of inactivity:
- New episode starts automatically
- Related events stay linked
- Enables temporal segmentation

### Attention Mechanism

The Modern Hopfield Network uses transformer-style attention:
1. Query is embedded and normalized
2. Dot product with all stored keys
3. Softmax with temperature scaling
4. Weighted sum of values

This enables:
- Partial match retrieval
- Graceful degradation with noise
- Multiple relevant memories in single query

## Performance Characteristics

- **Memory Complexity**: O(N) space for N memories
- **Retrieval Complexity**: O(N·d) for d-dimensional embeddings
- **Update Complexity**: O(N·d) for similarity matching, O(1) for EMA update
- **ChromaDB Search**: O(log N) with indexing
- **Embedding Generation**: O(L) for sequence length L

### Optimization Tips

1. **Adjust Temperature**: Higher β (20-30) for exact matches, lower (5-10) for fuzzy matching
2. **Tune Salience Threshold**: Lower (0.2) to keep more memories, higher (0.5) for only important ones
3. **Episode Timeout**: Shorter for rapid context switches, longer for extended tasks
4. **Embedding Cache**: Enable to avoid re-computing common phrases
5. **Batch Operations**: Use batch methods when storing multiple related events

## Troubleshooting

### Common Issues

**LLM Server Not Available**
- Check if llama.cpp server is running: `python llama_server_client.py status`
- Verify model path in `config/llm_server.env`
- System falls back to heuristic salience if LLM unavailable

**Memory Not Found**
- Check salience threshold - memory might be below threshold
- Use `--include-raw` flag to search all memories
- Verify episode ID if searching specific episodes

**High Memory Usage**
- Reduce `max_memory_slots` in configuration
- Enable ChromaDB for disk-based storage
- Clear old episodes periodically

**Slow Retrieval**
- Reduce embedding dimension for faster computation
- Enable embedding cache
- Use ChromaDB indexing for large memory stores

## Theory and Design

### Why 5W1H?

The 5W1H framework ensures complete context capture:
- **Who**: Tracks agency and responsibility
- **What**: Core content/action
- **When**: Temporal ordering and recency
- **Where**: Spatial/digital context
- **Why**: Causal reasoning and intent
- **How**: Methods and mechanisms

### Why Modern Hopfield Networks?

MHNs provide several advantages over traditional storage:
1. **Content-Addressable**: Retrieve by meaning, not location
2. **Robust to Noise**: Partial/corrupted queries still work
3. **Continuous Updates**: EMA allows gradual learning
4. **Scalable**: Linear growth with memory size
5. **Interpretable**: Attention weights show retrieval reasoning

### Dual Storage Rationale

Separating raw and processed storage serves different needs:
- **Raw**: Debugging, audit trails, compliance
- **Processed**: Fast retrieval, deduplication, relevance filtering

This mirrors human memory with sensory (raw) and long-term (processed) stores.

## Future Enhancements

Potential additions to the framework:
- Graph-based episode visualization
- Reinforcement learning for salience
- Multi-agent memory sharing
- Temporal decay models
- Hierarchical memory organization
- Cross-episode pattern mining

## License

MIT License - See LICENSE file for details
