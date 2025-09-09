# JAM Memory System Benchmarks

## Semantic Similarity Testing

The semantic similarity benchmark has been consolidated into two main modules:

### 1. Test Set Generation (`generate_semantic_testset.py`)
Generates persistent test sets with LLM-paraphrased queries:
```bash
python benchmarks/generate_semantic_testset.py --cases 50
```

Creates 5 query types:
- **Paraphrase**: Complete rewording preserving exact meaning
- **Summary**: Brief capture of core meaning
- **Question**: Natural question the text would answer
- **Expansion**: Elaboration with different terminology
- **Abstraction**: Higher-level conceptual expression

Test sets are saved to `benchmarks/test_data/` as timestamped JSON files.

### 2. Evaluation (`semantic_similarity_eval.py`)
Evaluates retrieval performance using persisted test sets:
```bash
# Use most recent test set (default)
python benchmarks/semantic_similarity_eval.py

# Generate new test set and evaluate
python benchmarks/semantic_similarity_eval.py --generate

# Use specific test set
python benchmarks/semantic_similarity_eval.py --testset semantic_testset_20250909_053516.json
```

Results are automatically saved to `benchmarks/results/` for analysis.

## Key Improvements

1. **True Semantic Testing**: Uses LLM paraphrasing instead of keyword matching
2. **Reproducible Benchmarks**: Persistent test sets ensure consistent evaluation
3. **Automatic Result Saving**: All benchmark results saved to `results/` directory
4. **Default Test Set Loading**: Automatically uses most recent test set if not specified

## Current Performance

Latest results show 74-76% recall@5-20 for semantic similarity, indicating the system effectively captures semantic meaning through embeddings rather than relying on keyword matching.

## Other Benchmarks

- **Recall Benchmark** (`recall_benchmark.py`): Tests 5 query types (temporal, actor, semantic, keyword, partial)
- **LMSYS Analysis** (`analyze_lmsys.py`): Analyzes imported conversation data
- **Core Benchmark** (`core.py`): Main benchmark framework with scenario-based testing