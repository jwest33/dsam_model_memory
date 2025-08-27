# Mathematical Foundations of the Dual-Space Memory System

## Overview

This document provides a technical explanation of the mathematical concepts underlying the Dual-Space Memory System, which combines Euclidean and Hyperbolic geometries for enhanced memory representation and retrieval in AI systems.

## 1. Dual-Space Architecture

The system employs two complementary geometric spaces:

### 1.1 Euclidean Space (ℝ^768)
A 768-dimensional real vector space using standard L2 (Euclidean) distance metric for capturing lexical and syntactic similarity.

**Why 768 dimensions?** Higher dimensions provide more capacity to encode nuanced differences between similar concepts without interference (the "blessing of dimensionality" for embeddings). The 768 dimensions match common transformer output sizes, allowing rich semantic representation where each dimension can capture different linguistic features (syntax, semantics, style, domain).

**Distance metric:**
```
d_E(x, y) = ||x - y||_2 = √(Σᵢ(xᵢ - yᵢ)²)
```

**Example:** 
- Query: "Python TypeError in login.py line 42"
- High similarity: "Python error in login.py", "TypeError at line 42"
- Low similarity: "JavaScript authentication flow"

### 1.2 Hyperbolic Space (𝔹^64)
A 64-dimensional Poincaré ball model for representing hierarchical and semantic relationships.

**Why Poincaré ball for hierarchies?** Hyperbolic space has exponentially growing volume with radius, naturally matching tree structures where nodes have exponentially more descendants at each level. Near the origin represents abstract/general concepts, while the boundary represents specific instances. This geometry preserves tree distances with minimal distortion - a binary tree of depth k needs only O(k) hyperbolic dimensions but O(2^k) Euclidean dimensions for isometric embedding.

**Distance metric:**
```
d_H(x, y) = arcosh(1 + 2||x - y||²/((1 - ||x||²)(1 - ||y||²)))
```

where points x, y ∈ 𝔹^n = {x ∈ ℝ^n : ||x|| < 1}

**Example:**
- Query: "How does authentication work?"
- High similarity: "User login flow", "Security protocols", "Session management"
- These are hierarchically related even if lexically different

## 2. Embedding Generation

### 2.1 Base Embeddings
We use the pre-trained sentence transformer `all-MiniLM-L6-v2` to generate initial 384-dimensional embeddings, which are then projected:

**Why this architecture?** The sentence transformer provides semantically meaningful base representations trained on millions of text pairs. Projecting to different dimensions for each space allows optimization - more dimensions (768) for Euclidean to capture fine-grained lexical details, fewer (64) for Hyperbolic since the curved geometry provides additional representational power.

**Euclidean projection:**
```python
W_E ∈ ℝ^(768×384)  # Learned projection matrix
e_euclidean = W_E @ base_embedding + b_E
```

**Hyperbolic projection with exponential map:**
```python
W_H ∈ ℝ^(64×384)   # Learned projection matrix
v = W_H @ base_embedding + b_H
e_hyperbolic = exp_0(v) = tanh(||v||/2) * (v/||v||)
```

The exponential map exp_0 projects from tangent space at origin to the Poincaré ball.

### 2.2 Field-Aware Gating
Different 5W1H fields contribute differently to each space:

```python
g_E = σ(W_gate_E @ field_indicators + b_gate_E)  # Euclidean gates
g_H = σ(W_gate_H @ field_indicators + b_gate_H)  # Hyperbolic gates

final_euclidean = g_E ⊙ e_euclidean
final_hyperbolic = g_H ⊙ e_hyperbolic
```

**Example field weights (learned):**
- "what" field: g_E = 0.85, g_H = 0.60 (more Euclidean-focused)
- "why" field: g_E = 0.45, g_H = 0.90 (more Hyperbolic-focused)

## 3. Query-Adaptive Space Weighting

The system dynamically computes space weights λ_E and λ_H based on query characteristics:

### 3.1 Weight Computation
```python
# Feature extraction from query
f_concrete = count(technical_terms) + count(specific_entities)
f_abstract = count(conceptual_terms) + count(relationship_words)

# Softmax normalization
λ_E = exp(β * f_concrete) / (exp(β * f_concrete) + exp(β * f_abstract))
λ_H = 1 - λ_E
```

where β is a temperature parameter (typically β = 2.0)

**Example weight distributions:**
- Query: "Fix null pointer exception in user.py" → λ_E = 0.78, λ_H = 0.22
- Query: "Explain the authentication architecture" → λ_E = 0.31, λ_H = 0.69
- Query: "How does the login system handle errors?" → λ_E = 0.52, λ_H = 0.48

## 4. Product Distance Metric

The final distance combines both spaces using a weighted product:

```
D(q, m) = d_E(q, m)^λ_E × d_H(q, m)^λ_H
```

**Why product over sum?** Product metrics better handle the different scales of the two spaces. Unlike weighted sums (αd_E + βd_H), products naturally accommodate that hyperbolic distances grow exponentially while Euclidean distances grow linearly. The exponential weighting (d^λ) provides smooth interpolation between pure Euclidean (λ_E=1) and pure Hyperbolic (λ_H=1) retrieval.

This formulation ensures:
- When λ_E → 1: Euclidean distance dominates
- When λ_H → 1: Hyperbolic distance dominates  
- When λ_E = λ_H = 0.5: Equal contribution

**Numerical example:**
```
Query: "database connection error"
Memory 1: d_E = 0.3, d_H = 0.8
Memory 2: d_E = 0.7, d_H = 0.4

With λ_E = 0.7, λ_H = 0.3:
D(q, m1) = 0.3^0.7 × 0.8^0.3 = 0.42 × 0.93 = 0.39
D(q, m2) = 0.7^0.7 × 0.4^0.3 = 0.77 × 0.74 = 0.57

Memory 1 is ranked higher despite worse hyperbolic distance.
```

## 5. Bounded Residual Adaptation

### 5.1 Residual Update Mechanism
The system maintains adaptive residuals r_E and r_H that modify base embeddings:

```python
# Gradient computation from user feedback
∇r_E = η_E * Σᵢ (y_i - ŷ_i) * ∂D/∂r_E
∇r_H = η_H * Σᵢ (y_i - ŷ_i) * ∂D/∂r_H

# Momentum update
v_E = μ * v_E + (1 - μ) * ∇r_E
v_H = μ * v_H + (1 - μ) * ∇r_H
```

**Why momentum?** Momentum (μ=0.9) smooths updates by accumulating gradients over time, preventing oscillations from noisy feedback and ensuring stable convergence. This is especially important in dual-space systems where gradients from different geometries might conflict. The high momentum value creates "heavy ball" dynamics that roll through local minima.

# Bounded update
r_E_new = clip(r_E + v_E, -B_E, B_E)
r_H_new = project_to_ball(r_H + v_H, B_H)
```

Parameters:
- η_E = η_H = 0.01 (learning rate)
- μ = 0.9 (momentum coefficient)
- B_E = 0.35 (Euclidean bound)
- B_H = 0.75 (Hyperbolic bound)

### 5.2 Temporal Decay
Residuals decay exponentially over time:

```python
r_E(t) = r_E(t-1) * γ
r_H(t) = r_H(t-1) * γ
```

where γ = 0.995 (decay factor)

**Example evolution:**
- Initial residual: ||r_E|| = 0.30
- After 100 iterations: ||r_E|| = 0.30 × 0.995^100 ≈ 0.18
- After 500 iterations: ||r_E|| = 0.30 × 0.995^500 ≈ 0.02

## 6. HDBSCAN Clustering

Hierarchical Density-Based Spatial Clustering of Applications with Noise groups memories dynamically:

**Why HDBSCAN over k-means or DBSCAN?** HDBSCAN excels at finding clusters of varying densities without requiring the number of clusters a priori. Unlike k-means, it handles non-spherical clusters and identifies outliers. Unlike DBSCAN, it automatically adapts to varying density levels through its hierarchical approach. This is crucial for memories that naturally form groups of different sizes and densities (e.g., many memories about common tasks, few about edge cases).

### 6.1 Algorithm Parameters
```python
clusterer = HDBSCAN(
    min_cluster_size=3,      # Minimum points to form cluster
    min_samples=2,            # Conservative density requirement
    metric='euclidean',       # Uses combined embedding
    cluster_selection_epsilon=0.3
)
```

### 6.2 Clustering Process
1. Compute mutual reachability distance matrix
2. Construct minimum spanning tree
3. Build cluster hierarchy
4. Extract flat clustering using stability

**Example clustering outcome:**
```
Memories about "authentication":
- Cluster 0: [login_endpoint, user_validation, session_creation]
- Cluster 1: [password_hashing, salt_generation, bcrypt_config]
- Noise: [oauth_redirect]  # Too different from others
```

## 7. Hyperbolic Operations

### 7.1 Möbius Addition
For combining hyperbolic vectors:

```
x ⊕ y = ((1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y) / (1 + 2⟨x,y⟩ + ||x||²||y||²)
```

### 7.2 Exponential and Logarithmic Maps
**Exponential map** (tangent space to Poincaré ball):
```
exp_x(v) = x ⊕ (tanh(λ_x||v||/2) * v/||v||)
```

**Why these maps?** The exponential map preserves the Riemannian metric, ensuring that operations in tangent space (where we can use familiar linear algebra) correctly translate to the curved hyperbolic space. The tanh function naturally bounds points within the unit ball while preserving differentiability for gradient-based learning.

**Logarithmic map** (Poincaré ball to tangent space):
```
log_x(y) = (2/λ_x) * arctanh(||-x ⊕ y||) * (-x ⊕ y)/||-x ⊕ y||
```

where λ_x = 2/(1 - ||x||²) is the conformal factor.

### 7.3 Parallel Transport
For moving vectors between tangent spaces:

```
PT_{x→y}(v) = (λ_x/λ_y) * gyration[y, -x](v)
```

## 8. Performance Characteristics

### 8.1 Computational Complexity
- Euclidean distance: O(d) where d = 768
- Hyperbolic distance: O(d) where d = 64, but with higher constant factor
- Product distance: O(d_E + d_H)
- HDBSCAN: O(n² log n) worst case, O(n log n) average

### 8.2 Memory Requirements
Per stored event:
- Euclidean: 768 × 4 bytes = 3,072 bytes
- Hyperbolic: 64 × 4 bytes = 256 bytes
- Residuals: (768 + 64) × 4 bytes = 3,328 bytes
- Metadata: ~500 bytes
- Total: ~7KB per memory

### 8.3 Retrieval Latency
For n = 1000 memories:
- Embedding generation: ~20ms
- Distance computation: ~5ms
- Sorting: ~1ms
- Total: <30ms for top-k retrieval

## 9. Mathematical Justification

### 9.1 Why Hyperbolic Space?
Hyperbolic geometry naturally embeds trees with low distortion. For a tree with n nodes:
- Euclidean space requires Ω(log n) dimensions
- Hyperbolic space requires only 2 dimensions (theoretically)

### 9.2 Why Product Metric?
The product metric satisfies:
1. **Identity**: D(x,x) = 0
2. **Symmetry**: D(x,y) = D(y,x)
3. **Distinguishability**: D(x,y) = 0 ⟺ x = y
4. **Weighted triangle inequality** (approximate)

### 9.3 Why Bounded Residuals?
Unbounded adaptation leads to:
- Catastrophic forgetting
- Embedding drift
- Loss of semantic coherence

Bounds ensure ||e_adapted - e_base|| ≤ B, maintaining semantic stability.

**Why different bounds (0.35 vs 0.75)?** Euclidean space requires tighter bounds (0.35) because changes directly affect lexical matching - too much drift would make "login error" match unrelated errors. Hyperbolic space allows larger bounds (0.75) because its hierarchical nature is more robust to local changes - moving within the hierarchy preserves relationships even with larger adjustments.

## 10. Implementation Examples

### 10.1 Storing a Memory
```python
# Input event
event = Event(
    five_w1h=FiveW1H(
        who="user123",
        what="implemented binary search algorithm",
        where="search_module.py",
        when="2024-01-15T10:30:00",
        why="optimize lookup performance",
        how="divide-and-conquer approach"
    )
)

# Generate embeddings
e_base = sentence_transformer.encode(concatenate_fields(event))
e_euclidean = project_euclidean(e_base)  # → ℝ^768
e_hyperbolic = project_hyperbolic(e_base)  # → 𝔹^64

# Store with residuals initialized to zero
store(event_id, e_euclidean, e_hyperbolic, r_E=0, r_H=0)
```

### 10.2 Retrieving Memories
```python
# Query processing
query = "how to optimize search performance"
λ_E, λ_H = compute_weights(query)  # → (0.35, 0.65)

# Compute distances
for memory in all_memories:
    d_E = euclidean_distance(q_euclidean, m.e_euclidean + m.r_E)
    d_H = hyperbolic_distance(q_hyperbolic, m.e_hyperbolic ⊕ m.r_H)
    score = d_E^λ_E × d_H^λ_H
    
# Return top-k
return sorted(memories, key=lambda m: m.score)[:k]
```

## Summary

The dual-space architecture leverages the complementary strengths of Euclidean geometry (precise lexical matching) and Hyperbolic geometry (hierarchical semantic relationships). Through adaptive residuals with bounds, momentum-based updates, and query-dependent space weighting, the system provides robust and flexible memory retrieval that improves with usage while maintaining stability.
