# Agentic Memory Framework

A local-first memory system that gives LLM agents persistent, searchable memory with human-like organization and recall patterns.

## Core Philosophy

Traditional chatbots forget everything between sessions. This framework provides agents with a memory system inspired by how humans actually remember: through stories, associations, and patterns rather than perfect recall. Every interaction becomes a memory that can be retrieved, reasoned about, and built upon.

## The Memory Model

### 5W1H Semantic Normalization

Every piece of information - whether from user input, LLM responses, or tool executions - is decomposed into fundamental semantic dimensions:

- **Who**: Actor identification (user, assistant, tool, system)
- **What**: The core content or action
- **When**: Temporal anchoring with decay modeling  
- **Where**: Spatial/contextual location (digital or physical)
- **Why**: Intent and purpose extraction
- **How**: Method, process, or mechanism used

This normalization enables memories to be compared, clustered, and retrieved across different modalities and contexts.

### Fluid Memory Blocks

Unlike rigid database records, memories organize into fluid blocks that reshape based on access patterns:

- **Temporal Clustering**: Recent events naturally group together
- **Conceptual Clustering**: Similar ideas coalesce regardless of time
- **Actor Clustering**: Interactions with the same entities form narrative threads
- **Spatial Clustering**: Events in similar contexts maintain proximity

These clusters aren't predefined - they emerge from actual usage patterns, creating a self-organizing memory topology.

### Hybrid Retrieval System

Memory retrieval combines multiple cognitive strategies:

1. **Semantic Search** (FAISS vectors): "What's similar in meaning?"
2. **Lexical Search** (SQLite FTS5): "What contains these exact words?"
3. **Recency Bias**: "What happened recently?"
4. **Actor Relevance**: "What involves these participants?"
5. **Spatial Context**: "What happened in similar places?"
6. **Usage Patterns**: "What gets accessed together?"

Each signal contributes to a unified relevance score, mimicking how humans use multiple cues to trigger memories.

## Token-Aware Context Building

### The Knapsack Problem

LLMs have finite context windows. The framework treats context building as an optimization problem: given a token budget, which memories provide maximum value?

The system uses:
- **Value Density**: Information value per token
- **MMR Diversity**: Avoiding redundant memories
- **Pointer Chaining**: When context overflows, create navigable memory chains
- **Adaptive Chunking**: Memories split or merge based on retrieval patterns

### Memory Decay and Reinforcement

Memories aren't static - they strengthen with use and fade without it:
- Frequently accessed memories gain priority
- Unused memories gradually decay in retrieval ranking
- Related memories reinforce each other through co-activation

## Tool Memory Integration

The framework treats tool usage as first-class memories:

- **Tool Calls**: Stored as structured events with full arguments
- **Tool Results**: Preserved with success/failure states
- **Execution Context**: Maintains the why and how of tool usage
- **Learning Patterns**: Agents can learn from past tool interactions

This enables agents to remember not just facts, but also *how they learned them*.

## Self-Organizing Knowledge

Over time, the memory system develops structure without explicit programming:

### Emergent Ontologies
Concepts naturally cluster through usage, creating implicit categories and relationships.

### Narrative Threading
Related memories link into coherent stories, maintaining context across sessions.

### Adaptive Indexing
Frequently traversed memory paths strengthen, creating efficient retrieval highways.

## Local-First Architecture

Everything runs on your machine:
- **Local LLM** via llama.cpp
- **Local Embeddings** via SentenceTransformers  
- **Local Vector Store** via FAISS
- **Local Database** via SQLite with FTS5

No cloud dependencies, no data leaving your system.

## Cognitive Inspirations

The framework draws from cognitive science:

- **Spreading Activation**: Related memories activate each other
- **Constructive Memory**: Memories reconstruct from fragments
- **Schema Theory**: Patterns emerge from repeated experiences
- **Episodic vs Semantic**: Both specific events and general knowledge

## Future Directions

The framework enables research into:
- Memory consolidation and abstraction
- Cross-agent memory sharing
- Temporal reasoning and planning
- Emergent knowledge graphs
- Memory-guided learning

## Getting Started

See [CLAUDE.md](CLAUDE.md) for setup instructions and technical details.

## License

MIT