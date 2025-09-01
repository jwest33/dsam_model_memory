# DSAM Technical Architecture

## System Overview

DSAM implements a content-addressable memory system using dual geometric spaces (Euclidean + Hyperbolic) for semantic retrieval without explicit memory addresses. The system adaptively weights between spaces based on query characteristics.

## Core Mathematical Framework

### Dual-Space Representation

**Euclidean Space (ℝ^768)**
- Direct embeddings from sentence-transformers/all-MiniLM-L6-v2
- L2 distance metric: `d_E(x, y) = ||x - y||_2`
- Optimized for lexical/syntactic similarity
- Example: "Python error" matches "Python exception"

**Hyperbolic Space (𝔹^64)**  
- Poincaré ball model with radius < 1
- Hyperbolic distance: `d_H(x, y) = arcosh(1 + 2||x - y||²/((1 - ||x||²)(1 - ||y||²)))`
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

### Query-Adaptive Weighting

The system dynamically computes space weights λ_E and λ_H:

```python
# Feature extraction
f_concrete = count(technical_terms) + count(specific_entities)
f_abstract = count(conceptual_terms) + count(relationships)

# Softmax normalization
λ_E = exp(β * f_concrete) / (exp(β * f_concrete) + exp(β * f_abstract))
λ_H = 1 - λ_E
```

Product distance metric:
```
D(q, m) = d_E(q, m)^λ_E × d_H(q, m)^λ_H
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

### Field-Level Adaptation

Per-field limits in 5W1H structure:
- `who`: 0.2 (entity stability)
- `what`: 0.5 (content flexibility)
- `when`: 0.3 (temporal consistency)
- `where`: 0.4 (spatial context)
- `why`: 0.5 (reasoning adaptation)
- `how`: 0.5 (method variation)

## Multi-Dimensional Merging

### Event Deduplication

Automatic merging when similarity > 0.85:
1. Detect similar events using product distance
2. Create merged representation preserving latest values
3. Maintain bidirectional raw↔merged mappings
4. Track merge statistics and provenance

### Merge Dimensions

**Actor Dimension**
- Groups by participants (who field)
- Tracks conversation participants
- Maintains speaker relationships

**Temporal Dimension**
- Groups by conversation threads
- Detects temporal patterns (Strong/Moderate/Weak)
- Links sequential events

**Conceptual Dimension**
- Groups by semantic concepts
- Clusters related ideas
- Preserves goal relationships

**Spatial Dimension**
- Groups by location context
- Maintains spatial relationships
- Tracks environment changes

## Storage Architecture

### ChromaDB Collections

```python
collections = {
    'events': merged_events,        # Deduplicated events
    'raw_events': original_events,   # All original events
    'blocks': memory_blocks,         # HDBSCAN clusters
    'metadata': system_state,        # Configuration/stats
    'similarity_cache': similarities # Pre-computed scores
}
```

### Similarity Cache

Pre-computed pairwise similarities:
- Sparse storage (threshold: 0.2)
- O(1) lookup vs O(n²) computation
- Persistent across restarts
- Hit rate typically >99% after warmup

## Performance Characteristics

### Computational Complexity
- Embedding generation: O(1) per query
- Distance computation: O(n) for n memories
- Product metric: O(d_E + d_H) = O(768 + 64)
- Cache lookup: O(1) amortized

### Memory Requirements
Per event:
- Euclidean embedding: 3KB (768 × 4 bytes)
- Hyperbolic embedding: 256B (64 × 4 bytes)
- Residuals: 3.3KB (832 × 4 bytes)
- Metadata: ~500B
- Total: ~7KB per memory

### Latency Benchmarks
For 1000 memories:
- Embedding: ~20ms
- Retrieval: ~50ms with cache
- Graph generation: ~100ms
- Merge computation: ~30ms

## Implementation Details

### Numerical Stability

**Hyperbolic Operations**
```python
# Prevent boundary issues
x = clip_norm(x, max_norm=0.999)

# Stable distance computation
epsilon = 1e-5
d = arcosh(max(1 + delta, 1 + epsilon))
```

**Möbius Addition**
```python
# Gyrovector addition in Poincaré ball
x ⊕ y = ((1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y) / 
         (1 + 2⟨x,y⟩ + ||x||²||y||²)
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

Benefits over alternatives:
- No fixed cluster count (vs k-means)
- Varying density support (vs DBSCAN)
- Automatic outlier detection
- Hierarchical structure preservation

## Usage Examples

### Content-Addressable Retrieval

```python
# Store without explicit address
memory_agent.remember(
    who="Alice",
    what="implemented caching",
    how="Redis with LRU"
)

# Retrieve by content similarity
results = memory_agent.recall(
    what="cache",  # Partial content
    k=5
)
# Returns semantically similar memories
```

### Adaptive Learning

```python
# System adapts from co-retrieval patterns
query = "authentication flow"
results = retrieve(query)

# Positive feedback strengthens associations
adapt_residuals(query, results, positive=True)

# Future queries benefit from adaptation
```

## Configuration

Key parameters in `config.py`:

```python
DualSpaceConfig:
    euclidean_dim: 768
    hyperbolic_dim: 64
    euclidean_bound: 0.35
    hyperbolic_bound: 0.75
    momentum: 0.9
    decay_factor: 0.995
    learning_rate: 0.01
    similarity_threshold: 0.85
```

## Future Enhancements

- Graph neural networks for relationship modeling
- Attention mechanisms for field weighting
- Distributed storage for scaling
- Real-time stream processing
- Hierarchical memory consolidation