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
from .attention import AdaptiveEmbeddingSpace
from .cluster.concept_cluster import LiquidMemoryClusters

class MemoryRouter:
    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        self.retriever = HybridRetriever(store, index)
        self.builder = BlockBuilder(store)
        self.embedder = SentenceTransformer(cfg.embed_model_name)
        self.tok = TokenizerAdapter()
        
        # Initialize dynamic components if enabled
        if cfg.use_attention_retrieval:
            self.adaptive_embeddings = AdaptiveEmbeddingSpace(
                base_dim=cfg.embed_dim,
                meta_dim=64
            )
        else:
            self.adaptive_embeddings = None
            
        if cfg.use_liquid_clustering:
            self.liquid_clusters = LiquidMemoryClusters(
                n_clusters=64,
                dim=cfg.embed_dim
            )
            self.liquid_clusters.flow_rate = cfg.cluster_flow_rate
            self.liquid_clusters.merge_threshold = cfg.cluster_merge_threshold
        else:
            self.liquid_clusters = None

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
                
                # Apply adaptive embedding if enabled
                if self.adaptive_embeddings:
                    usage_stats = self.store.get_usage_stats([rec.memory_id])
                    vec = self.adaptive_embeddings.encode_with_context(
                        vec, 
                        usage_stats.get(rec.memory_id, {}),
                        None
                    )
                
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
        
        # Apply adaptive embedding if enabled
        if self.adaptive_embeddings:
            usage_stats = self.store.get_usage_stats([rec.memory_id])
            vec = self.adaptive_embeddings.encode_with_context(
                vec,
                usage_stats.get(rec.memory_id, {}),
                None
            )
        
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
        
        # Apply adaptive query embedding if enabled
        if self.adaptive_embeddings:
            # Create synthetic usage stats for query
            query_stats = {'access_count': 1, 'recency_score': 1.0, 'diversity_score': 0.5}
            qvec = self.adaptive_embeddings.encode_with_context(qvec, query_stats, None)
        
        ranked = self.retriever.search(rq, qvec, topk_sem=60, topk_lex=60)
        
        # Update liquid clusters if enabled and we have results
        if self.liquid_clusters and ranked:
            retrieved_ids = [c.memory_id for c in ranked[:20]]
            
            # Get embeddings for clustering
            embeddings_dict = {}
            for cand in ranked[:20]:
                # Try to get embedding from index or store
                emb = self.store.get_embedding_drift(cand.memory_id)
                if emb is not None:
                    embeddings_dict[cand.memory_id] = emb
            
            if embeddings_dict:
                # Update co-access patterns
                self.liquid_clusters.update_co_access(retrieved_ids)
                
                # Update cluster energy
                self.liquid_clusters.update_cluster_energy(retrieved_ids)
                
                # Perform flow step
                self.liquid_clusters.flow_step(retrieved_ids, embeddings_dict)
                
                # Periodically merge similar clusters
                if np.random.random() < 0.05:  # 5% chance
                    self.liquid_clusters.merge_similar_clusters(embeddings_dict)
        
        overhead = sum([len(m.get('content','')) for m in context_messages]) // 3  # rough token overhead
        blocks = self.builder.build(rq, ranked, context_overhead=overhead)
        if not blocks:
            return {"block": None, "members": []}
        first = blocks[0]
        out = self.store.get_block(first.block_id)
        
        # Update embedding drift based on retrieval context
        if self.adaptive_embeddings and out.get('members'):
            member_ids = out['members'][:10]  # Top 10 members
            embeddings = []
            for mid in member_ids:
                emb = self.store.get_embedding_drift(mid)
                if emb is not None:
                    embeddings.append(emb)
            
            # Update drift for each retrieved memory
            for mid in member_ids:
                if embeddings:
                    self.adaptive_embeddings.update_embedding_drift(mid, embeddings)
                    
                    # Store updated drift
                    if mid in self.adaptive_embeddings.embedding_momentum:
                        drift = self.adaptive_embeddings.embedding_momentum[mid]
                        self.store.store_embedding_drift(mid, drift)
        
        return out
