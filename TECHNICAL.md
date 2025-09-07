# Technical Architecture: Data Flow and Mathematical Models

## Overview

JAM (Journalistic Agent Memory) implements a multi-stage pipeline for memory ingestion, storage, retrieval, and context building. This document details the data flow and mathematical formulations used throughout the system.

## Data Flow Architecture

### 1. Memory Ingestion Pipeline

```
Raw Event -> LLM Extraction -> 5W1H Normalization -> Embedding -> Storage -> Indexing
```

#### Stage 1.1: Event Capture
- **Input**: Raw text from user interaction, assistant response, or tool execution
- **Format**: Unstructured natural language with metadata (timestamp, actor, event_type)
- **Output**: Structured event object with raw content and metadata

#### Stage 1.2: LLM Extraction (5W1H)
- **Process**: Local LLM (Qwen-4B) extracts semantic dimensions
- **Prompt Engineering**: Zero-shot extraction with structured output format
- **Fields Extracted**:
  - `who`: Actor identification (user, assistant, tool:name)
  - `what`: Core content/action (main event description)
  - `when`: Temporal anchor (ISO timestamp)
  - `where`: Location context (URL, file path, chat session)
  - `why`: Intent/purpose (goal or reason)
  - `how`: Method/mechanism (approach taken)

#### Stage 1.3: Vector Embedding
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Input**: Concatenated 5W1H fields
- **Output**: 384-dimensional dense vector
- **Formula**: 
  ```
  text = f"{who} {what} {when} {where} {why} {how}"
  embedding = model.encode(text, normalize_embeddings=True)
  ```

#### Stage 1.4: Dual Storage
- **SQLite Storage**:
  - Full 5W1H fields with FTS5 indexing
  - Metadata JSON blob
  - Usage statistics (access_count, last_accessed)
- **FAISS Index**:
  - L2-normalized embeddings
  - Flat index for exact similarity search
  - Memory ID mapping

### 2. Retrieval Pipeline

```
Query -> Multi-Signal Scoring -> Fusion -> Ranking -> Selection
```

#### Stage 2.1: Query Processing
- **Input**: Natural language query or context
- **Embedding**: Same model as ingestion (all-MiniLM-L6-v2)
- **Output**: Query vector + tokenized terms

#### Stage 2.2: Multi-Signal Retrieval

The system computes six independent relevance signals:

##### Semantic Similarity (FAISS)
```python
# Cosine similarity via L2-normalized vectors
semantic_scores = index.search(query_vector, k=top_k)
# Returns distances in [0, 2] range, converted to [0, 1] similarity
similarity = 1 - (distance / 2)
```

##### Lexical Match (FTS5)
```sql
-- BM25 scoring via SQLite FTS5
SELECT id, bm25(memories_fts) as score
FROM memories_fts 
WHERE memories_fts MATCH ?
ORDER BY score DESC
```

##### Recency Scoring
```python
# Exponential decay with 30-day half-life
age_days = (now - memory_timestamp).days
recency_score = exp(-0.693 * age_days / 30)
```

##### Actor Relevance
```python
# Binary relevance with boost for exact match
actor_score = 1.0 if memory.who == query_actor else 0.3
```

##### Spatial Proximity
```python
# Location similarity based on path components
def spatial_similarity(loc1, loc2):
    parts1 = loc1.split('/')
    parts2 = loc2.split('/')
    common = len(set(parts1) & set(parts2))
    return common / max(len(parts1), len(parts2))
```

##### Usage Patterns
```python
# Normalized access frequency with decay
usage_score = (access_count / max_count) * recency_factor
```

#### Stage 2.3: Score Fusion

Final relevance score combines all signals with configurable weights:

```python
final_score = (
    w_semantic * semantic_score +     # Default: 0.55
    w_lexical * lexical_score +       # Default: 0.20
    w_recency * recency_score +       # Default: 0.10
    w_actor * actor_score +           # Default: 0.07
    w_spatial * spatial_score +       # Default: 0.03
    w_usage * usage_score             # Default: 0.05
)
# Weights sum to 1.0 for normalized scores
```

### 3. Context Building Pipeline

```
Ranked Memories -> Token Budget -> MMR Diversity -> Knapsack Packing -> Context
```

#### Stage 3.1: Token Budget Calculation
```python
available_tokens = (
    context_window_size -            # Total context (8192)
    reserve_system_tokens -          # System prompt (512)
    reserve_output_tokens -          # Response space (1024)
    current_conversation_tokens      # Current chat
)
```

#### Stage 3.2: Maximal Marginal Relevance (MMR)

Balances relevance with diversity to avoid redundancy:

```python
def mmr_score(candidate, selected, lambda_param=0.5):
    """
    MMR = λ * Relevance - (1-λ) * max(Similarity to selected)
    """
    relevance = candidate.score
    
    if not selected:
        return relevance
    
    # Compute similarity to all selected memories
    max_similarity = max(
        cosine_similarity(candidate.embedding, s.embedding)
        for s in selected
    )
    
    return lambda_param * relevance - (1 - lambda_param) * max_similarity
```

#### Stage 3.3: 0-1 Knapsack Optimization

Treats context building as a bounded knapsack problem:

```python
def knapsack_memories(memories, token_budget):
    """
    Maximize: Σ(value_i * x_i)
    Subject to: Σ(tokens_i * x_i) ≤ token_budget
    Where x_i ∈ {0, 1}
    
    value_i = mmr_score_i * importance_weight_i
    tokens_i = token_count(memory_i)
    """
    
    # Dynamic programming solution
    dp = [[0] * (token_budget + 1) for _ in range(len(memories) + 1)]
    
    for i in range(1, len(memories) + 1):
        memory = memories[i-1]
        tokens = memory.token_count
        value = memory.mmr_score
        
        for w in range(token_budget + 1):
            if tokens <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # Don't include
                    dp[i-1][w-tokens] + value  # Include
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find selected memories
    return backtrack_solution(dp, memories, token_budget)
```

### 4. Clustering Pipeline

```
Embeddings -> MiniBatch KMeans -> Concept Assignment -> Incremental Update
```

#### Stage 4.1: Incremental Clustering
```python
class ConceptClusterer:
    def __init__(self, n_clusters=50):
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=100,
            n_init=3,
            max_no_improvement=10
        )
    
    def update(self, new_embeddings):
        """Incremental cluster update"""
        if self.kmeans.cluster_centers_ is None:
            # Initial fit
            self.kmeans.fit(new_embeddings)
        else:
            # Partial fit for new data
            self.kmeans.partial_fit(new_embeddings)
```

#### Stage 4.2: Concept Assignment
```python
def assign_concept(embedding):
    """Assign memory to nearest cluster"""
    distances = euclidean_distances(
        embedding.reshape(1, -1),
        self.kmeans.cluster_centers_
    )
    cluster_id = np.argmin(distances)
    confidence = 1 / (1 + distances[0, cluster_id])
    return cluster_id, confidence
```

## Liquid Memory Clustering

### Self-Organizing Memory Topology

The system implements a dynamic "liquid" clustering approach where memories flow between clusters based on affinity gradients:

#### Affinity Matrix Computation
```python
def compute_affinity_matrix(memory_ids, embeddings, metadata):
    """
    Multi-signal affinity between memory pairs
    """
    affinity[i,j] = (
        0.4 * co_access_score +      # How often accessed together
        0.3 * temporal_proximity +    # Time-based closeness
        0.3 * semantic_similarity     # Meaning similarity
    )
```

#### Gradient Flow Dynamics
```python
def flow_step(memories, affinity_matrix):
    """
    Memories migrate following affinity gradients
    """
    for memory in memories:
        # Compute pull from each cluster
        pulls = {}
        for cluster in clusters:
            pull_strength = avg_affinity_to_cluster_members
            pulls[cluster] = pull_strength * cluster_energy
        
        # Flow to strongest pull if above threshold
        if max(pulls.values()) > flow_rate:
            memory.cluster = argmax(pulls)
```

#### Energy Dynamics
```python
# Cluster energy increases with access
energy[cluster] = min(1.0, energy[cluster] + 0.05 * access_count)

# Energy decays over time
energy[cluster] *= 0.99  # Per timestep decay

# Clusters merge when similarity exceeds threshold
if cosine_similarity(centroid1, centroid2) > 0.85:
    merge_clusters(cluster1, cluster2)
```

### 3D Visualization Pipeline

The visualization system projects high-dimensional embeddings into 3D space for intuitive exploration:

#### Dimensionality Reduction Methods

##### PCA (Principal Component Analysis)
```python
# Linear projection preserving maximum variance
reducer = PCA(n_components=3)
coords_3d = reducer.fit_transform(embeddings)
# Fast, deterministic, preserves global structure
```

##### t-SNE (t-Distributed Stochastic Neighbor Embedding)
```python
# Non-linear projection preserving local neighborhoods
reducer = TSNE(n_components=3, perplexity=30, max_iter=1000)
coords_3d = reducer.fit_transform(embeddings)
# Perplexity controls local vs global structure balance
```

##### UMAP (Uniform Manifold Approximation and Projection)
```python
# Manifold learning preserving both local and global structure
reducer = UMAP(n_components=3, n_neighbors=15)
coords_3d = reducer.fit_transform(embeddings)
# Fastest for large datasets, good structure preservation
```

#### Visualization Features

##### Dynamic Coloring Schemes
- **Cluster ID**: Discrete colors per cluster (Viridis colormap)
- **Energy Level**: Continuous heat map (Hot colormap)
- **Recency Score**: Temporal gradient (Plasma colormap)
- **Usage Count**: Frequency visualization (Electric colormap)

##### Adaptive Clustering
```python
# Dynamically adjust cluster count based on data
n_samples = len(embeddings)
optimal_clusters = min(max(8, n_samples // 10), 64)
# Ensures meaningful clusters: 8 ≤ k ≤ 64
```

##### Real-time Updates
- Auto-refresh every 30 seconds
- Co-access patterns tracked across sessions
- Energy propagation through cluster network
- Automatic merger of converged clusters

## Mathematical Models

### Attention Mechanism

Tracks which memories contribute to each response:

```python
class AttentionHead:
    def compute_attention(self, query, memories):
        """
        Scaled dot-product attention
        Attention(Q,K,V) = softmax(QK^T/√d)V
        """
        d_k = len(query)
        
        # Compute attention scores
        scores = []
        for memory in memories:
            score = np.dot(query, memory.embedding) / np.sqrt(d_k)
            scores.append(score)
        
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / exp_scores.sum()
        
        # Store attention pattern
        self.last_attention = {
            memory.id: weight 
            for memory, weight in zip(memories, attention_weights)
        }
        
        return attention_weights
```

### Time Decay Models

#### Exponential Decay (Default)
```python
def exponential_decay(age_days, half_life=30):
    """
    score(t) = e^(-λt)
    where λ = ln(2)/half_life
    """
    decay_rate = 0.693 / half_life
    return np.exp(-decay_rate * age_days)
```

#### Power Law Decay (Alternative)
```python
def power_law_decay(age_days, alpha=1.5):
    """
    score(t) = 1 / (1 + t)^α
    Slower decay for long-term retention
    """
    return 1 / (1 + age_days) ** alpha
```

### Diversity Metrics

#### Intra-List Diversity (ILD)
```python
def intra_list_diversity(selected_memories):
    """
    Average pairwise distance between selected items
    Higher values indicate more diverse selection
    """
    if len(selected_memories) < 2:
        return 0
    
    distances = []
    for i, m1 in enumerate(selected_memories):
        for m2 in selected_memories[i+1:]:
            dist = 1 - cosine_similarity(m1.embedding, m2.embedding)
            distances.append(dist)
    
    return np.mean(distances)
```

### Performance Metrics

#### Retrieval Precision@K
```python
def precision_at_k(retrieved, relevant, k):
    """
    Fraction of retrieved items that are relevant
    """
    retrieved_k = retrieved[:k]
    relevant_in_k = len(set(retrieved_k) & set(relevant))
    return relevant_in_k / k
```

#### Normalized Discounted Cumulative Gain (NDCG)
```python
def ndcg(ranked_items, relevance_scores, k):
    """
    Measures ranking quality with position-based discounting
    """
    def dcg(scores, k):
        return sum(
            (2**score - 1) / np.log2(i + 2)
            for i, score in enumerate(scores[:k])
        )
    
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True), k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
```

## Configuration Parameters

### Critical Tuning Parameters

1. **Retrieval Weights** (must sum to 1.0):
   - Semantic: 0.55 (primary signal)
   - Lexical: 0.20 (exact matches)
   - Recency: 0.10 (time bias)
   - Actor: 0.07 (participant relevance)
   - Spatial: 0.03 (location context)
   - Usage: 0.05 (access patterns)

2. **MMR Lambda** (0.5 default):
   - Higher (->1.0): More relevance-focused
   - Lower (->0.0): More diversity-focused

3. **Token Budgets**:
   - Context Window: 8192 (model limit)
   - System Reserve: 512 (prompts)
   - Output Reserve: 1024 (responses)
   - Effective Budget: ~6656 tokens

4. **Clustering Parameters**:
   - Clusters: 50 (empirically determined)
   - Batch Size: 100 (memory efficiency)
   - Convergence: 10 iterations no improvement

## Optimization Strategies

### Index Optimization
- FAISS flat index for exact search (no approximation)
- FTS5 with Porter stemmer for better recall
- Composite SQLite indexes on (who, when, event_type)

### Memory Optimization
- Lazy loading of embeddings
- Batch processing for embedding generation
- Memory-mapped FAISS index for large datasets

### Latency Optimization
- Parallel signal computation
- Cached embeddings for frequent queries
- Pre-computed token counts

## Scalability Considerations

### Current Limits
- ~100K memories: Sub-second retrieval
- ~1M memories: 2-3 second retrieval
- ~10M memories: Requires index sharding

### Scaling Strategies
1. **Horizontal Sharding**: Partition by time ranges
2. **Hierarchical Indexing**: Two-level FAISS with coarse quantization
3. **Selective Loading**: Load only recent + relevant memories
4. **Compression**: Product quantization for embeddings

## Future Enhancements

### Planned Improvements
1. **Graph-based Memory**: Add explicit memory relationships
2. **Learned Retrieval**: Train retrieval weights from user feedback
3. **Adaptive Clustering**: Dynamic cluster count based on data
4. **Streaming Updates**: Real-time index updates without rebuild
5. **Multi-Modal Memories**: Support for images and structured data
