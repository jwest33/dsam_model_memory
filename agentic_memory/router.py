from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import cfg
from .types import RawEvent, RetrievalQuery
from .extraction.llm_extractor import extract_5w1h
from .extraction.multi_part_extractor import extract_multi_part_5w1h
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

    def ingest(self, raw_event: RawEvent, context_hint: str = "", use_multi_part: bool = True) -> str:
        """
        Ingest a raw event, optionally breaking it into multiple memories.
        
        Args:
            raw_event: The raw event to process
            context_hint: Additional context for extraction
            use_multi_part: If True, attempt to break complex content into multiple memories
            
        Returns:
            Comma-separated list of memory IDs created
        """
        memory_ids = []
        
        # Decide whether to use multi-part extraction
        should_use_multi = use_multi_part and cfg.use_multi_part_extraction and (
            len(raw_event.content) > cfg.multi_part_threshold or  # Long content
            '\n\n' in raw_event.content or  # Multi-paragraph content
            raw_event.content.count('\n') > 3 or  # Multiple lines
            raw_event.event_type in ['llm_message', 'tool_result']  # Often have complex responses
        )
        
        if should_use_multi:
            # Try multi-part extraction
            memories = extract_multi_part_5w1h(raw_event, context_hint=context_hint)
            
            # Process each memory
            for rec in memories:
                vec = np.array(rec.extra.pop('embed_vector_np'), dtype='float32')
                # Persist
                self.store.upsert_memory(rec, embedding=vec.tobytes(), dim=vec.shape[0])
                # Add to FAISS (normalized already)
                self.index.add(rec.memory_id, vec)
                memory_ids.append(rec.memory_id)
            
            if memory_ids:
                self.index.save()
                return ','.join(memory_ids)
        
        # Fallback to single extraction
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
