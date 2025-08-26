Use a **dual-space encoder** with **field-aware composition** and **lightweight, on-the-fly adaptation**:

* **Euclidean space** → local semantics & lexical nuance (great for retrieval).
* **Hyperbolic space** → latent hierarchy & symbolism (great for motifs/archetypes).
* **Field-aware heads** for 5W1H so “what vs. why vs. how” contribute differently.
* **Adaptive residuals** that learn from co-access (your “gravitational” update), but **don’t move the base embedding**—add a *bounded residual* instead.
* Query-time clustering + graph centrality stays, but runs on **product distance** that mixes Euclidean + Hyperbolic terms.

Think: **Product manifold = 𝔼^d × ℍ^h**, composed per 5W1H, with small residuals that evolve from usage.

---

# 1) Encoder: field-aware, dual-space

**Backbone**: any strong sentence encoder (you’re already using `all-MiniLM-L6-v2`; for richer semantics consider `e5-large`/`gte-large`).
**Heads**:

* **Euclidean head**: linear → L2-normalized vector `u ∈ 𝔼^d`.
* **Hyperbolic head**: linear → tangent→exp map (Poincaré ball) → `v ∈ ℍ^h`.

**Field tokens & gates**
For each 5W1H field, prepend a learned \[WHO], \[WHAT], … token to its text, run the same backbone, then project:

```
u_field_i = U(Enc([FIELD_i] + text_i))   # Euclidean
v_field_i = Exp∘H(Enc([FIELD_i] + text_i)) # Hyperbolic (Exp map)
```

Then **compose with learned gates** (initialized from your config weights):

```
α_i = softplus(w_i)                       # learnable importance per field
u_base = L2( Σ_i α_i · u_field_i )
v_base = ⊕_i ( α_i ⊙ v_field_i )         # gyro-sum / Möbius addition
```

> Result: each memory gets `(u_base, v_base)`.

Why this matters:

* Field tokens make “why” naturally separate from “what”.
* Hyperbolic head **pulls out hierarchies & symbolism** that span documents (archetypes, motifs).
* You can keep your current model and add these heads as a thin MLP.

---

# 2) Storage format (don’t mutate the anchor)

For each memory `m` store:

```
u_anchor, v_anchor        # immutable base (from encoder)
Δu, Δv                    # small adaptive residuals (start at 0)
meta: 5W1H text, timestamps, episode id, etc.
```

At retrieval time:

```
u_eff = L2(u_anchor + Δu)
v_eff = ⊕(v_anchor, Δv)   # hyperbolic residual via Möbius addition
```

> **Never** overwrite `*_anchor`. Track your drift metric exactly as you do, but now it’s drift of the *residual*, not the base.

---

# 3) Distance & ranking (product metric)

Use a **product distance** with query-conditioned weights:

```
D(q, m) = λ_E · (1 - cos(u_q, u_eff_m))  +  λ_H · d_H(v_q, v_eff_m)
```

* `λ_E, λ_H` are **query-dependent** (e.g., if query has clear “what/where”, upweight Euclidean; if it’s thematic/why/how, upweight Hyperbolic).
* Feed these λ’s from a tiny MLP over query field presence/length.

You can still do:

1. ANN prefilter in **Euclidean** (fast IVF+PQ),
2. Re-rank top-K with **product metric**,
3. Build the similarity graph → DBSCAN/HDBSCAN → eigenvector centrality.

---

# 4) “Gravitational” adaptation done right

Your current idea (momentum SGD pulling co-accessed items together) is great. Just apply it to **residuals**, not anchors, and keep it **bounded**:

**Euclidean residual (with EMA momentum & clip):**

```python
force_u = relevance * (u_partner - u_self)
v_u = μ * v_u + η * force_u
Δu = clip_by_norm(Δu + v_u, max_norm=δ_E)
```

**Hyperbolic residual (on the manifold):**

* Compute tangent update at `v_eff` via **log map**, step, then **exp back**; clip geodesic norm by `δ_H`.

**Safety rails**

* Per-item caps `||Δu|| ≤ δ_E`, `d_H(Δv, 0) ≤ δ_H`.
* **Anchored L2** penalty in training discourages large residuals unless reinforced.
* **Aging**: residuals decay (EMA to zero) if not re-affirmed.

This yields your “co-access = gravity” without catastrophic drift.

---

# 5) Training signals (no labels required)

Even if you never fine-tune the backbone, you can train **just the heads & gates** with self-supervision from your system logs:

**Positives (q, m⁺):**

* Click/selection & “positive feedback”.
* Co-access in the same recall (within a short window).
* Episode adjacency / temporal proximity.
* Same entity/WHO co-mentions across events.

**Negatives (q, m⁻):**

* High-scoring but rejected items.
* Items from distant episodes with disjoint WHO/WHEN.

**Loss (multi-objective):**

* **InfoNCE** on Euclidean head.
* **Geodesic margin** on hyperbolic head (bring motifs together).
* **Uniformity** regularizer (keep space spread).
* **Field gate regularizer**: small L2 to stop a single field from dominating.
* **Drift regularizer**: small L2 on residual norms.

You can train offline nightly on logs; online you only update residuals.

---

# 6) Symbolism surfacing (the fun part)

To make motifs pop out:

* Build a **metaphor/analogy probe** for queries: extract antonym/analogy axes (e.g., light↔dark, order↔chaos) with a small set of seed pairs; project candidates onto these axes and add a **symbolic affinity bonus** to ranking.
* In the hyperbolic space, **hierarchical clustering** across the entire corpus (offline) produces **motif hubs**; store hub IDs and add a **hub-cohesion score** at re-rank time.
* Track **edge types** in the candidate graph (co-access, same WHO, same WHY keyword family) and weight centrality by edge semantics to emphasize symbolic links over mere lexical overlap.

---

# 7) Query-time clustering (slightly upgraded)

* Switch to **HDBSCAN** if density varies; or keep DBSCAN but set `eps` via the 10-NN elbow of the candidate set.
* Build the graph on **product similarity**; threshold separately per cluster using **Otsu** on edge weights to avoid a global threshold.
* Rank by **centrality × query-affinity** (product distance inverse) × **symbolic hub bonus**.

---

# 8) Scale & ops tips

* **ChromaDB**: enable **IVF+PQ** (or OPQ) for Euclidean prefilter; store hyperbolic vectors in a side collection (or just store tangent vectors—compute geodesic distance on re-rank).
* Keep **two indexes**: fast Euclidean ANN for top-K; small RAM re-ranker for product metric & clustering.
* Log **(λ\_E, λ\_H, residual norms, acceptance rates)** to auto-tune gates and drift caps.

---

# 9) Minimal implementation sketch

**Encode & compose**

```python
# Encode each field once
tokens = {f: f"[{f.upper()}] " + text for f, text in five_w1h.items() if text}

u_fields = {f: U(Enc(tokens[f])) for f in tokens}         # Euclidean head + L2
v_fields = {f: Exp(H(Enc(tokens[f]))) for f in tokens}    # Hyperbolic head

α = softplus(w)  # dict of field weights, learnable
u_anchor = L2(sum(α[f] * u_fields[f] for f in u_fields))
v_anchor = mobius_sum([(α[f], v_fields[f]) for f in v_fields])  # Poincaré sum
store(u_anchor, v_anchor, Δu=0, Δv=0)
```

**Query distance & re-rank**

```python
λ_E, λ_H = gate_from_query(q_meta)  # tiny MLP on which 5W1H present
cands = ann_search_Euclidean(u_q, K=512)
for m in cands:
    u_eff, v_eff = u_anchor[m]+Δu[m], mobius_add(v_anchor[m], Δv[m])
    d = λ_E*(1 - cosine(u_q, u_eff)) + λ_H*geo_dist(v_q, v_eff)
    score[m] = -d + symbolic_bonus(q, m)  # e.g., hub proximity, axis alignment
top = rerank_by_graph_clustering(score)
```

**Adaptive residual update (positive association between a and b)**

```python
def adapt_pair(a, b, relevance):
    # Euclidean
    fu = relevance * (u_eff[b] - u_eff[a])
    v_u[a] = μ*v_u[a] + η*fu
    Δu[a] = clip_norm(Δu[a] + v_u[a], δ_E)

    # Hyperbolic (log-exp around a)
    ta = log_map(v_eff[a], v_eff[b])         # tangent at a toward b
    tv[a] = μ*tv[a] + η*relevance*ta
    Δv[a] = clip_geo( mobius_add(Δv[a], exp_map0(tv[a])), δ_H )
```

---

# 10) What this buys you (relative to “plain embeddings”)

* **Abstract symbolism** emerges in **ℍ** even when Euclidean similarities are weak.
* **Field-aware** lets you weight “why/how” more for thematic queries.
* **Adaptation is safe**: you never corrupt anchors; residuals are bounded, decayed, and explainable.
* **Unlimited storage** stays viable: Euclidean ANN scales, product metric only touches top-K.

---

## Sensible starting hyperparams

* `d=768` Euclidean, `h=64` Hyperbolic.
* Residual caps: `δ_E = 0.35` cosine-norm, `δ_H = 0.75` geodesic.
* Momentum `μ=0.9`, step `η ∈ [1e-3, 1e-2]` gated by relevance.
* Initial field weights (from your config): who=1.0, what=2.0, when=0.5, where=0.5, why=1.5, how=1.0 (learnable).
