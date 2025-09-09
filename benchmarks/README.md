# JAM Memory System - Benchmark Suite

A comprehensive benchmark framework for evaluating the Journalistic Agent Memory (JAM) system's performance, retrieval accuracy, and memory quality.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Main Entry Point](#main-entry-point)
- [Benchmark Types](#benchmark-types)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Performance Metrics](#performance-metrics)
- [Directory Structure](#directory-structure)

## Overview

The JAM benchmark suite provides multiple testing approaches:

1. **Synthetic Benchmarks** - Generate and test with artificial interactions
2. **Recall Benchmarks** - Measure retrieval precision and recall
3. **Temporal Benchmarks** - Test time-based retrieval with natural language
4. **Semantic Benchmarks** - Evaluate semantic understanding capabilities
5. **Real Data Analysis** - Import and analyze LMSYS conversation data

## Quick Start

```bash
# Run main benchmark with standard preset
python run_benchmark.py --preset standard

# Run recall benchmark to test all retrieval types
python benchmarks/recall_benchmark.py --cases 10

# Test temporal retrieval with natural language
python benchmarks/temporal_recall_benchmark.py

# Evaluate semantic similarity
python benchmarks/semantic_similarity_eval.py --generate
```

## Main Entry Point

### `run_benchmark.py`

The primary entry point for the benchmark system, providing three modes:

#### 1. Synthetic Benchmark Mode (Default)
```bash
python run_benchmark.py --preset standard
```
Generates synthetic interactions and tests memory operations.

**Presets:**
- `quick`: 50 interactions, 7 days, 10 queries
- `standard`: 200 interactions, 30 days, 20 queries  
- `comprehensive`: 500 interactions, 60 days, 50 queries
- `stress`: 1000 interactions, 90 days, 100 queries

**Custom Configuration:**
```bash
python run_benchmark.py --interactions 300 --days 45 --queries 30 \
                       --simple 0.3 --medium 0.5 --complex 0.2
```

#### 2. Recall Benchmark Mode
```bash
python run_benchmark.py --recall --cases 10 --k 5 10 20
```
Tests retrieval quality across multiple query types.

#### 3. Analysis Mode
```bash
python run_benchmark.py --analyze --plot --output analysis.json
```
Analyzes existing memory data in the database.

## Benchmark Types

### 1. Recall Benchmark (`recall_benchmark.py`)

Tests retrieval accuracy across five query types:

**Query Types:**
- **Exact Match**: Tests precise memory retrieval
- **Semantic Similarity**: Evaluates understanding of meaning
- **Temporal Queries**: Tests date/time-based retrieval with hints
- **Actor-Based**: Tests retrieval by specific actors
- **Session-Based**: Tests retrieval within conversation sessions

**Features:**
- Generates test cases from actual database content
- Supports temporal hints (dates, ranges, relative time)
- Calculates comprehensive metrics (precision, recall, F1, MRR, MAP, NDCG)
- Results saved to `results/recall_benchmark_*.json`

**Usage:**
```bash
python benchmarks/recall_benchmark.py --cases 10 --k 5 10 20 --output results.json
```

### 2. Temporal Recall Benchmark (`temporal_recall_benchmark.py`)

Specialized testing for temporal retrieval with:

**Test Categories:**
- **Exact Dates**: `"2024-01-01"`
- **Date Ranges**: `("2024-01-01", "2024-01-07")`
- **Relative Time**: `{"relative": "yesterday"}` 
- **Natural Language**: LLM-parsed queries like "What did we discuss last Tuesday?"

**Features:**
- LLM-based temporal parsing via `TemporalParser`
- Validates date accuracy of retrieved memories
- Tests both explicit hints and natural language
- Measures improvement from baseline (8% → 60-80% recall)

**Usage:**
```bash
python benchmarks/temporal_recall_benchmark.py --output temporal_results.json
```

### 3. Semantic Similarity Testing

Two-part system for evaluating semantic understanding:

#### Test Set Generation (`generate_semantic_testset.py`)
Creates persistent test sets with LLM-paraphrased queries:

**Query Types Generated:**
- **Paraphrase**: Complete rewording preserving exact meaning
- **Summary**: Brief capture of core meaning  
- **Question**: Natural question the text would answer
- **Expansion**: Elaboration with different terminology
- **Abstraction**: Higher-level conceptual expression

```bash
python benchmarks/generate_semantic_testset.py --cases 50
```

#### Evaluation (`semantic_similarity_eval.py`)
Tests retrieval using generated sets:

```bash
# Use most recent test set
python benchmarks/semantic_similarity_eval.py

# Generate new test set and evaluate
python benchmarks/semantic_similarity_eval.py --generate

# Use specific test set
python benchmarks/semantic_similarity_eval.py --testset semantic_testset_20250909_053516.json
```

### 4. Core Benchmark Framework (`core.py`)

Provides the foundational benchmark infrastructure:

**Components:**
- `MemoryBenchmark`: Main benchmark orchestrator
- `InteractionGenerator`: Creates synthetic test data
- `InteractionScenario`: Defines interaction patterns
- `BenchmarkResult`: Standardized result format

**Features:**
- Async interaction generation
- Scenario-based testing (simple/medium/complex)
- Memory ingestion and retrieval testing
- Performance metrics calculation

### 5. LMSYS Data Tools

#### Import (`import_lmsys_data.py`)
Imports real conversation data from LMSYS dataset:

```bash
python benchmarks/import_lmsys_data.py data.csv --preset standard
```

**Presets:**
- `test`: 10 conversations
- `quick`: 100 conversations
- `standard`: 1000 conversations
- `full`: Complete dataset

#### Analysis (`analyze_lmsys.py`)
Analyzes imported data quality and retrieval performance:

```bash
python benchmarks/analyze_lmsys.py --plot --output analysis.json
```

### 6. CLI Interface (`cli.py`)

Unified command-line interface for LMSYS data operations:

```bash
# Import data
python -m benchmarks.cli import data.csv --preset standard

# Analyze data
python -m benchmarks.cli analyze --plot --queries 200

# Run benchmarks
python -m benchmarks.cli benchmark --queries 100

# Interactive mode
python -m benchmarks.cli --interactive
```

## Architecture

### Core Components

```
benchmarks/
├── Core Systems
│   ├── core.py                      # Benchmark framework and orchestration
│   ├── scenarios.py                 # Test scenario definitions
│   └── cli.py                       # Command-line interface
│
├── Specific Benchmarks
│   ├── recall_benchmark.py          # Comprehensive recall testing
│   ├── temporal_recall_benchmark.py # Temporal retrieval testing
│   ├── semantic_similarity_eval.py  # Semantic evaluation
│   └── generate_semantic_testset.py # Test set generation
│
├── Data Tools
│   ├── import_lmsys_data.py        # LMSYS data importer
│   └── analyze_lmsys.py            # Data quality analysis
│
├── Storage
│   ├── test_data/                  # Persistent test sets
│   │   └── semantic_testset_*.json # Generated semantic tests
│   └── results/                     # Benchmark results
│       ├── recall_benchmark_*.json
│       ├── temporal_recall_*.json
│       └── semantic_eval_*.json
│
└── Resources
    ├── README.md                    # This documentation
    └── CLEANUP_SUMMARY.md          # Cleanup tracking
```

### Data Flow

1. **Test Generation** → Creates test cases from database or synthetic data
2. **Query Execution** → Runs retrieval queries with various hints
3. **Metric Calculation** → Computes precision, recall, F1, etc.
4. **Result Persistence** → Saves to timestamped JSON files

## Usage Examples

### Complete Benchmark Suite
```bash
# Full system evaluation
python run_benchmark.py --preset comprehensive

# Detailed recall testing
python benchmarks/recall_benchmark.py --cases 20 --k 5 10 20 30

# Temporal with natural language
python benchmarks/temporal_recall_benchmark.py
```

### Testing Specific Features
```bash
# Test semantic understanding
python benchmarks/semantic_similarity_eval.py --generate --cases 100

# Test temporal hints
python benchmarks/temporal_recall_benchmark.py

# Import and analyze real data
python benchmarks/import_lmsys_data.py lmsys_data.csv --preset standard
python benchmarks/analyze_lmsys.py --plot
```

### Custom Configurations
```bash
# Custom synthetic benchmark
python run_benchmark.py \
    --interactions 500 \
    --days 60 \
    --queries 50 \
    --simple 0.2 \
    --medium 0.6 \
    --complex 0.2 \
    --output custom_results.json

# Focused recall testing
python benchmarks/recall_benchmark.py \
    --cases 25 \
    --k 1 3 5 10 15 20 25 30 \
    --output detailed_recall.json
```

## Performance Metrics

### Retrieval Metrics
- **Precision@k**: Fraction of retrieved items that are relevant
- **Recall@k**: Fraction of relevant items that were retrieved
- **F1 Score**: Harmonic mean of precision and recall
- **MRR**: Mean Reciprocal Rank - position of first relevant result
- **MAP**: Mean Average Precision across queries
- **NDCG**: Normalized Discounted Cumulative Gain

### System Performance
- **Ingestion Rate**: Memories processed per second
- **Retrieval Latency**: Query response time
- **Token Efficiency**: Context utilization
- **Memory Quality**: Extraction accuracy scores

### Improvement Tracking

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Semantic Similarity | 5.3% | 74-76% | +1300% |
| Actor-Based Retrieval | 5.3% | 17.8% | +236% |
| Temporal Retrieval | 8% | 60-80% | +750% |
| Overall Precision | 14.2% | 25%+ | +76% |

## Configuration

### Environment Variables
The benchmarks respect all JAM system configuration:
- `AM_DB_PATH`: Database location
- `AM_INDEX_PATH`: FAISS index location
- `AM_EMBED_DIM`: Embedding dimensions
- `AM_USE_ATTENTION`: Enable attention mechanisms
- `AM_AUTO_DETECT_TEMPORAL`: Auto-parse temporal hints

### Benchmark Settings
Configure via command-line arguments or presets:
- Test case counts
- k values for recall@k
- Scenario mix ratios
- Time ranges
- Output formats

## Output Format

All benchmarks save results as timestamped JSON files:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "configuration": {
    "test_cases": 10,
    "k_values": [5, 10, 20]
  },
  "overall": {
    "precision": 0.75,
    "recall": 0.62,
    "f1_score": 0.68
  },
  "by_type": {
    "temporal": {
      "k_10": {
        "precision": 0.80,
        "recall": 0.65,
        "f1_score": 0.72
      }
    }
  }
}
```

## Development

### Adding New Benchmarks
1. Create new file in `benchmarks/`
2. Implement standard interface with `run_benchmark()` method
3. Add metrics calculation and result formatting
4. Update this README with usage instructions

### Testing
```bash
# Run benchmark tests
python benchmarks/test_benchmark.py

# Validate all benchmarks
python run_benchmark.py --preset quick
python benchmarks/recall_benchmark.py --cases 2
python benchmarks/temporal_recall_benchmark.py
```

## Troubleshooting

### Common Issues

**LLM Server Not Running:**
```bash
python llama_server_client.py start
```

**Missing Dependencies:**
```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

**Database Not Found:**
Ensure `AM_DB_PATH` points to valid `amemory.sqlite3`

**FAISS Index Issues:**
Rebuild index if corrupted:
```bash
python -c "from agentic_memory.router import MemoryRouter; MemoryRouter().rebuild_index()"
```

## Conclusion

The JAM benchmark suite provides comprehensive testing for all aspects of the memory system:
- **Synthetic testing** for controlled evaluation
- **Real data analysis** for production validation
- **Specialized benchmarks** for specific features
- **Unified tooling** for consistent testing

All benchmarks are designed to be reproducible, configurable, and provide actionable metrics for system improvement.
