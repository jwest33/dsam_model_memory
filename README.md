# Agentic Memory Framework

A production-grade, **local-first** agentic memory system for LLM tool-using agents running
on **llama.cpp** (tested with Qwen-family models), with a Flask GUI.

Core features:

- **5W1H normalization** of every interaction (who/what/when/where/why/how).
- **Hybrid search**: FAISS semantic + SQLite FTS5 lexical + recency/actor/spatial/usage signals.
- **Fluid memory blocks**: dynamic, soft-clustered (temporal, actor, conceptual, spatial).
- **Block builder**: token-budgeted context packing with **pointer chaining** for overflow.
- **Adaptive layout**: memory blocks rearrange based on actual retrieval & access patterns.
- **Local embeddings** (SentenceTransformers) and **local vector index** (FAISS).
- **SQLite+FTS5** storage with WAL and integrity constraints.
- **Flask GUI** for chat + memory inspector. Everything runs offline, locally.

## Quickstart

> Requires Python 3.10+ (recommended), and CPU/GPU support for FAISS as desired.

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

1. Start your **llama.cpp** server for Qwen3-4b-instruct-2507 (or similar) in OpenAI-compatible mode, for example:

```bash
./server -m /path/to/qwen3-4b-instruct-2507.gguf --port 8080 --host 127.0.0.1 --api
```

2. Configure environment (optional; defaults work):

```bash
export AM_LLM_BASE_URL=http://127.0.0.1:8080/v1
export AM_LLM_MODEL=Qwen3-4b-instruct-2507
export AM_CONTEXT_WINDOW=8192
export AM_DB_PATH=./amemory.sqlite3
export AM_INDEX_PATH=./faiss.index
export AM_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

3. Run the Flask app:

```bash
python -m agentic_memory.server.flask_app
```

Open http://127.0.0.1:5001/ in your browser.

## Project layout

```
agentic_memory/
  ├── agentic_memory/
  │   ├── cluster/
  │   ├── extraction/
  │   ├── server/ (Flask GUI)
  │   ├── storage/ (SQLite+FTS5, FAISS)
  │   ├── block_builder.py
  │   ├── config.py
  │   ├── retrieval.py
  │   ├── router.py
  │   ├── tokenization.py
  │   ├── types.py
  │   ├── usage.py
  │   └── __init__.py
  ├── tests/
  ├── requirements.txt
  └── README.md
```

## Design Notes

- **5W1H extraction**: implemented with a structured schema and a small LLM prompt (local) + fallbacks.
- **Hybrid retrieval**: FTS5 BM25 + FAISS cosine; tunable weighting (see `retrieval.py`).
- **Knapsack-style block builder**: greedy + MMR diversity, respects token budget (see `block_builder.py`).
- **Pointer chaining**: if a block overflows, it creates `prev/next` block pointers usable via a small tool call.
- **Clustering**: incremental conceptual clusters (`MiniBatchKMeans`) + natural temporal, actor, spatial facets.
- **Observability**: structured logs with `structlog`; basic tests for core components.

## Security & Privacy

Everything is local. For optional encryption-at-rest you may swap SQLite for SQLCipher or encrypt FAISS
files with filesystem-level encryption.

## License

MIT
