# Running Experiments with the Dual-Space Memory System

This document explains how to run experiments to test the enhanced memory system with dual-space encoding, HDBSCAN clustering, and adaptive residuals.

## Overview

The enhanced system features:
- **Dual-space encoding**: Euclidean space for concrete/lexical similarity, Hyperbolic space for abstract/hierarchical relationships
- **Product distance metrics**: Query-dependent weighting between spaces
- **HDBSCAN clustering**: Better handling of varying density clusters
- **Bounded residual adaptation**: Safe evolution of embeddings based on usage patterns

## Quick Start

### 1. Basic CLI Testing

```bash
# Initialize the system
python cli.py init

# Store some memories
python cli.py remember --who "Alice" --what "implemented caching layer" --where "backend" --why "improve performance" --how "using Redis"
python cli.py remember --who "Bob" --what "designed API endpoints" --where "REST service" --why "enable client access" --how "following OpenAPI spec"

# Test retrieval
python cli.py recall --what "performance" --k 5
python cli.py recall --why "architecture" --how "design" --k 5

# View statistics
python cli.py stats
```

### 2. Standalone Experiments (No Web Server Required)

```bash
# Run interactive experiment menu
python test_experiments.py

# Options:
# 1. Run all experiments
# 2. Space separation test - Tests how concrete vs abstract content separates
# 3. Clustering quality test - Tests HDBSCAN clustering
# 4. Residual adaptation test - Tests embedding evolution
# 5. Query weighting test - Tests query-dependent space weighting
```

### 3. Web-Based Conversation Simulations

```bash
# Option 1: Automated runner (starts web server automatically)
python run_experiments.py

# Option 2: Manual setup
# Terminal 1: Start web server
python run_web.py

# Terminal 2: Run simulations
python simulate_conversations.py
```

## Experiment Types

### Space Separation Test
Tests how different content types distribute in dual spaces:
- **Concrete technical content**: Bug fixes, code snippets, specific implementations
- **Abstract conceptual content**: Design patterns, architectural principles, philosophies
- **Expected result**: Abstract content should show higher separation in Hyperbolic space

### Clustering Quality Test  
Tests HDBSCAN's ability to group related memories:
- Creates distinct topic clusters (Frontend, Backend, DevOps)
- Tests intra-cluster queries (should retrieve mostly from one cluster)
- Tests inter-cluster queries (should retrieve from multiple clusters)

### Residual Adaptation Test
Tests how embeddings evolve based on co-retrieval:
- Creates related memories
- Performs repeated queries to trigger adaptation
- Measures residual norm changes over iterations
- **Expected result**: Related memories should gravitate together

### Query Weighting Test
Tests how query field presence affects space weighting:
- **Concrete queries** (what/where/who) → Higher Euclidean weight
- **Abstract queries** (why/how) → Higher Hyperbolic weight
- **Mixed queries** → Balanced weights

## Understanding Results

### Key Metrics

1. **Space Separation Ratio (Hyperbolic/Euclidean)**
   - `> 1.2`: Good separation, abstract content utilizing hyperbolic properties
   - `< 1.0`: Poor separation, may need tuning

2. **Average Residual Norms**
   - **Euclidean**: Should stay below 0.35 (max bound)
   - **Hyperbolic**: Should stay below 0.75 (max geodesic bound)
   - Small positive values indicate healthy adaptation

3. **Clustering Distribution**
   - Intra-cluster queries should retrieve 60-80% from target cluster
   - Inter-cluster queries should show more even distribution

4. **Query Weights (λ_E, λ_H)**
   - Should sum to 1.0
   - Concrete queries: λ_E > 0.6
   - Abstract queries: λ_H > 0.6

## Advanced Testing

### Custom Experiments

Create your own experiment by modifying `test_experiments.py`:

```python
def experiment_custom(self):
    """Your custom experiment"""
    # Store specific memories
    # Run targeted queries
    # Measure specific metrics
    pass
```

### Performance Testing

```bash
# Test with larger dataset
python generate_bulk_memories.py --count 1000
python cli.py recall --what "test" --k 50
```

### Visualization

The web interface (http://localhost:5000) provides:
- Real-time memory visualization
- Cluster graph display
- Query result highlighting

## Troubleshooting

### Common Issues

1. **"Cannot connect to web application"**
   - Ensure web server is running: `python run_web.py`
   - Check port 5000 is not in use

2. **"LLM server not available"**
   - The system works without LLM but with limited conversation generation
   - To enable LLM, ensure llama-server is running

3. **High residual norms**
   - Normal after many adaptations
   - System automatically decays residuals periodically
   - Can clear with: `rm -rf state/chromadb`

4. **Poor clustering results**
   - May need more memories (minimum 10-20 for good clustering)
   - Try adjusting HDBSCAN parameters in memory_store.py

## Experiment Outputs

Results are displayed in console with:
- 📊 Statistical summaries
- ✓ Success indicators  
- ⚠️ Warning indicators
- 🔍 Query indicators
- 📈 Progress indicators

Final statistics show:
- Total events stored
- Total queries performed
- Average residual norms
- Clustering quality metrics

## Next Steps

After running experiments:

1. **Tune parameters** in `memory_store.py`:
   - `max_euclidean_norm`: Residual bounds
   - `learning_rate`: Adaptation speed
   - `residual_decay`: Aging factor

2. **Adjust field weights** in initialization:
   - Increase weights for important fields
   - Decrease weights for less relevant fields

3. **Experiment with different content**:
   - Technical documentation
   - Conversational dialogue
   - Mixed abstract/concrete content

4. **Measure retrieval quality**:
   - Precision/recall metrics
   - User feedback simulation
   - Response time analysis