from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import cfg
from .types import RawEvent, RetrievalQuery
from .extraction.llm_extractor import extract_5w1h
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .retrieval import HybridRetriever
from .block_builder import BlockBuilder
from .tokenization import TokenizerAdapter

class MemoryRouter:
    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        self.retriever = HybridRetriever(store, index)
        self.builder = BlockBuilder(store)
        self.embedder = SentenceTransformer(cfg.embed_model_name)
        self.tok = TokenizerAdapter()

    def ingest(self, raw_event: RawEvent, context_hint: str = "") -> str:
        rec = extract_5w1h(raw_event, context_hint=context_hint)
        vec = np.array(rec.extra.pop('embed_vector_np'), dtype='float32')
        # Persist
        self.store.upsert_memory(rec, embedding=vec.tobytes(), dim=vec.shape[0])
        # Add to FAISS (normalized already)
        self.index.add(rec.memory_id, vec)
        self.index.save()
        return rec.memory_id

    def retrieve_block(self, session_id: str, context_messages: List[Dict[str, str]],
                       actor_hint: Optional[str] = None, spatial_hint: Optional[str] = None) -> Dict[str, Any]:
        # Construct query text from context (last N messages)
        ctx_text = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in context_messages[-6:]])
        rq = RetrievalQuery(session_id=session_id, actor_hint=actor_hint, spatial_hint=spatial_hint, text=ctx_text)
        qvec = self.embedder.encode([ctx_text], normalize_embeddings=True)[0]
        ranked = self.retriever.search(rq, qvec, topk_sem=60, topk_lex=60)
        overhead = sum([len(m.get('content','')) for m in context_messages]) // 3  # rough token overhead
        blocks = self.builder.build(rq, ranked, context_overhead=overhead)
        if not blocks:
            return {"block": None, "members": []}
        first = blocks[0]
        out = self.store.get_block(first.block_id)
        return out
