# DSAM Technical Architecture

## System Overview

DSAM implements a content-addressable memory system using dual geometric spaces (Euclidean + Hyperbolic) for semantic retrieval without explicit memory addresses. The system features multi-dimensional merging, intelligent field generation, and adaptive residual learning.

## Core Mathematical Framework

### Dual-Space Representation

**Euclidean Space (ℝ^768)**
- Direct embeddings from sentence-transformers/all-MiniLM-L6-v2
- Cosine distance metric for merging: `d_cos(x, y) = 1 - (x·y)/(||x||·||y||)`
- Optimized for lexical/syntactic similarity
- Example: "Python error" matches "Python exception"

**Hyperbolic Space (𝔹^64)**  
- Poincaré ball model with radius < 1
- Geodesic distance: `d_H(x, y) = arcosh(1 + 2||x - y||²/((1 - ||x||²)(1 - ||y||²)))`
- Captures hierarchical/abstract relationships
- Example: "authentication" relates to "login", "security", "sessions"

### Embedding Pipeline

```python
# Base embedding generation
base_embedding = sentence_transformer.encode(text)  # 768-dim

# Euclidean: Use directly
e_euclidean = base_embedding

# Hyperbolic: Project and map to Poincaré ball
v = W_H @ base_embedding + b_H  # W_H ∈ ℝ^(64×768)
e_hyperbolic = tanh(||v||/2) * (v/||v||)  # Exponential map
```

### Product Distance for Retrieval

The system combines both spaces for memory retrieval:

```python
# Product distance metric
D(q, m) = λ_E * d_cos(q_E, m_E) + λ_H * d_geodesic(q_H, m_H)

# Where λ_E + λ_H = 1.0 (default: 0.5 each)
```

## Multi-Dimensional Merging

### Merge Strategy

All merge types now use **cosine similarity** for better performance in high-dimensional spaces:

```python
# Distance calculation for merging
cosine_distance = 1 - cosine_similarity(emb1, emb2)

# Merge thresholds (cosine distance)
CONCEPTUAL: 0.3   # Tighter for distinct concepts
SPATIAL: 0.3      # Similar locations
TEMPORAL: 0.5     # Conversation threads
ACTOR: 2.0        # Lenient for actor grouping
```

### Merge Dimensions

**Actor Dimension**
- Groups by participants (who field)
- Normalized actor categories (user, assistant, etc.)
- Maintains conversation participant relationships

**Temporal Dimension**
- Groups by conversation threads
- Dynamic temporal windows (10-60 minutes)
- Adapts to conversation patterns

**Conceptual Dimension**
- Groups by semantic similarity using cosine distance
- Primary field: 'what' (topic/action)
- Secondary field: 'why' (purpose/goal)
- LLM-generated group characterizations

**Spatial Dimension**
- Groups by location context
- Semantic similarity for location matching
- Cross-actor spatial relationships

## Intelligent Field Generation

### Individual Memory Fields

The system generates context-aware 'why' and 'how' fields:

```python
# Why field: Intent/purpose analysis
why = analyze_intent(what, who, conversation_context)

# How field: Mechanism detection
mechanisms = [
    'chat_interface',
    'tool_use',
    'llm_generation',
    'user_interaction',
    'analysis',
    'retrieval'
]
how = detect_mechanism(what, context)
```

### Group-Level Characterization

Merge groups receive intelligent characterizations:

```python
# Group-level fields generated based on all members
group_why = generate_group_purpose(all_events)
group_how = generate_group_mechanism(all_events)

# Size-based regeneration with logarithmic scaling
regeneration_interval = 1.2 * log(size + 1.5)^1.6 * (1 + size/200)
```

## Adaptive Learning

### Bounded Residual Adaptation

Memories maintain adaptive residuals that modify base embeddings:

```python
# Update with momentum
v_E = μ * v_E + (1 - μ) * ∇r_E  # μ = 0.9
r_E = clip(r_E + α * v_E, -B_E, B_E)  # B_E = 0.35

# Temporal decay
r_E(t) = r_E(t-1) * γ  # γ = 0.995
```

Bounds prevent catastrophic drift:
- Euclidean: ±0.35 (tighter for lexical stability)
- Hyperbolic: ±0.75 (looser for hierarchical flexibility)

## Storage Architecture

### ChromaDB Collections

```python
# Main collections
collections = {
    'events': merged_events,        # Deduplicated events
    'raw_events': original_events,   # All original events
    'merged_events': merge_mappings, # Merge relationships
    'similarity_cache': similarities # Pre-computed scores
}

# Dimensional collections (Euclidean + Hyperbolic)
merge_collections = {
    'actor_merges': actor_groups,
    'temporal_merges': temporal_groups,
    'conceptual_merges': conceptual_groups,
    'spatial_merges': spatial_groups,
    # Plus hyperbolic versions for each
}
```

### Similarity Cache

Pre-computed pairwise similarities:
- Sparse storage (threshold: 0.2)
- O(1) lookup vs O(n²) computation
- Uses both Euclidean and Hyperbolic distances
- Hit rate typically >99% after warmup

## Performance Characteristics

### Computational Complexity
- Embedding generation: O(1) per query
- Cosine similarity: O(d) for d dimensions
- Product distance: O(768 + 64) for dual spaces
- Cache lookup: O(1) amortized

### Memory Requirements
Per event:
- Euclidean embedding: 3KB (768 × 4 bytes)
- Hyperbolic embedding: 256B (64 × 4 bytes)
- Residuals: 3.3KB (832 × 4 bytes)
- Metadata + fields: ~1KB
- Total: ~7.5KB per memory

### Latency Benchmarks
For 1000 memories:
- Embedding: ~20ms
- Retrieval: ~50ms with dual-space product distance
- Cosine similarity: ~5ms per pair
- Merge computation: ~30ms
- Field generation: ~100ms with LLM

## Implementation Details

### Numerical Stability

**Cosine Similarity**
```python
# Normalize to prevent numerical issues
norm1 = np.linalg.norm(emb1)
norm2 = np.linalg.norm(emb2)
if norm1 == 0 or norm2 == 0:
    return 1.0  # Max distance
cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
```

**Hyperbolic Operations**
```python
# Prevent boundary issues
x = clip_norm(x, max_norm=0.999)

# Stable geodesic distance
epsilon = 1e-5
d = arcosh(max(1 + delta, 1 + epsilon))
```

### HDBSCAN Clustering

```python
clusterer = HDBSCAN(
    min_cluster_size=5,
    min_samples=3,
    metric='precomputed',  # Uses similarity cache
    cluster_selection_epsilon=0.3
)
```

Benefits:
- No fixed cluster count
- Varying density support
- Automatic outlier detection
- Hierarchical structure preservation

## Usage Examples

### Content-Addressable Retrieval

```python
# Store with intelligent field generation
memory_agent.remember(
    who="Alice",
    what="implemented caching",
    # 'why' and 'how' auto-generated
)

# Retrieve using dual-space similarity
results = memory_agent.recall(
    what="cache",  # Partial content
    k=5
)
# Returns semantically similar memories using product distance
```

### Multi-Dimensional Merging

```python
# Events automatically merge into 4 dimensions
event = {
    'who': 'user',
    'what': 'debugging JSON parsing',
    'when': '2024-01-15T10:30:00Z',
    'where': 'API endpoint'
}

# Creates/updates groups in:
# - Actor: user group
# - Temporal: current conversation thread
# - Conceptual: debugging group (cosine similarity)
# - Spatial: API endpoint group
```

## Configuration

Key parameters:

```python
# Embedding Configuration
DualSpaceConfig:
    euclidean_dim: 768
    hyperbolic_dim: 64
    euclidean_bound: 0.35
    hyperbolic_bound: 0.75
    
# Merge Configuration
MergeThresholds:
    conceptual: 0.3  # Cosine distance
    spatial: 0.3     # Cosine distance
    temporal: 0.5    # Cosine distance
    actor: 2.0       # Cosine distance

# Adaptation Parameters
    momentum: 0.9
    decay_factor: 0.995
    learning_rate: 0.01
    similarity_threshold: 0.85
```

## Key Innovations

1. **Dual-Space Architecture**: Combines Euclidean and Hyperbolic geometries for nuanced similarity
2. **Cosine Similarity Merging**: Superior performance in high-dimensional semantic spaces
3. **Intelligent Field Generation**: LLM-powered context-aware field creation
4. **Size-Based Regeneration**: Logarithmic scaling for efficient group updates
5. **Product Distance Retrieval**: Weighted combination of both embedding spaces

## Future Enhancements

- Graph neural networks for relationship modeling
- Attention mechanisms for dynamic field weighting
- Distributed storage for horizontal scaling
- Real-time stream processing for continuous learning
- Hierarchical memory consolidation using hyperbolic structure
- Explainable retrieval with path visualization