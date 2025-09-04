from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timezone
import torch
from .config import cfg
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .types import RetrievalQuery, Candidate
from .attention import MemoryAttentionHead, AdaptiveEmbeddingSpace, MemoryConsolidation

def exp_recency(ts_iso: str, now: datetime, half_life_hours: float = 72.0) -> float:
    try:
        ts = datetime.fromisoformat(ts_iso.replace('Z','+00:00'))
    except Exception:
        return 0.5
    dt = (now - ts).total_seconds() / 3600.0
    # Exponential decay, 0..1
    return max(0.0, min(1.0, 0.5 ** (dt / half_life_hours)))

class HybridRetriever:
    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        
        # Initialize attention-based components
        self.attention_head = MemoryAttentionHead(embed_dim=cfg.embed_dim)
        self.adaptive_embeddings = AdaptiveEmbeddingSpace(base_dim=cfg.embed_dim)
        self.consolidator = MemoryConsolidation()
        
        # Load saved states if available
        try:
            self.consolidator.load_state(cfg.consolidation_state_path)
        except:
            pass  # Start fresh if no saved state
        
        # Track whether to use attention (can be toggled)
        self.use_attention = cfg.use_attention_retrieval

    def _semantic(self, qvec: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        return self.index.search(qvec, topk)

    def _lexical(self, query: str, topk: int) -> List[Tuple[str, float]]:
        rows = self.store.lexical_search(query, k=topk)
        # FTS5 bm25 returns lower is better; invert + normalize
        if not rows:
            return []
        scores = [r['score'] for r in rows]
        max_s = max(scores); min_s = min(scores)
        out = []
        for r in rows:
            s = r['score']
            if max_s == min_s:
                norm = 1.0
            else:
                norm = 1.0 - (s - min_s) / (max_s - min_s)
            out.append((r['memory_id'], float(norm)))
        return out

    def merge_scores(self, sem: List[Tuple[str, float]], lex: List[Tuple[str, float]], 
                     attention_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        sdict = {m:s for m,s in sem}
        ldict = {m:s for m,s in lex}
        keys = set(sdict.keys()) | set(ldict.keys())
        merged = {}
        
        if attention_scores and self.use_attention:
            # Use attention-based dynamic weighting
            for k in keys:
                base_score = sdict.get(k, 0.0) * 0.4 + ldict.get(k, 0.0) * 0.2
                attn_score = attention_scores.get(k, 0.0) * 0.4
                merged[k] = base_score + attn_score
        else:
            # Fall back to static weights
            for k in keys:
                merged[k] = cfg.w_semantic * sdict.get(k,0.0) + cfg.w_lexical * ldict.get(k,0.0)
        return merged

    def _fetch_meta(self, ids: List[str]):
        rows = self.store.fetch_memories(ids)
        by_id = {r['memory_id']: r for r in rows}
        return by_id

    def rerank(self, merged: Dict[str, float], rq: RetrievalQuery, 
               memory_embeddings: Optional[Dict[str, np.ndarray]] = None) -> List[Candidate]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        metas = self._fetch_meta(list(merged.keys()))
        cands = []
        
        # Get usage stats for adaptive scoring
        usage_stats = self.store.get_usage_stats(list(merged.keys())) if hasattr(self.store, 'get_usage_stats') else {}
        
        for mid, base in merged.items():
            m = metas.get(mid)
            if not m:
                continue
                
            # Compute importance from consolidation network
            importance = self.consolidator.compute_memory_importance(mid) if self.use_attention else 0.0
            
            # Extra signals
            rec = exp_recency(m['when_ts'], now)
            actor_match = 1.0 if (rq.actor_hint and m['who_id'] == rq.actor_hint) else 0.0
            spatial_match = 1.0 if (rq.spatial_hint and m['where_value'] == rq.spatial_hint) else 0.0
            
            # Get actual usage count if available
            usage_data = usage_stats.get(mid, {})
            usage_score = min(1.0, usage_data.get('accesses', 0) / 100.0) if usage_data else 0.0
            
            if self.use_attention:
                # Dynamic scoring with learned importance
                score = (base * 0.5 + 
                        importance * 0.2 +
                        rec * 0.15 +
                        usage_score * 0.1 +
                        actor_match * 0.025 +
                        spatial_match * 0.025)
            else:
                # Original static scoring
                usage_boost = 0.2 if actor_match or spatial_match else 0.0
                score = base + cfg.w_recency*rec + cfg.w_actor*actor_match + cfg.w_spatial*spatial_match + cfg.w_usage*usage_boost
                
            cands.append(Candidate(memory_id=mid, score=score, token_count=int(m['token_count'])))
            
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def search(self, rq: RetrievalQuery, qvec: np.ndarray, topk_sem: int = 50, topk_lex: int = 50) -> List[Candidate]:
        sem = self._semantic(qvec, topk_sem)
        lex = self._lexical(rq.text, topk_lex)
        
        # Compute attention scores if enabled
        attention_scores = None
        if self.use_attention:
            # Get embeddings for candidate memories
            candidate_ids = list(set([m for m, _ in sem + lex]))
            if candidate_ids:
                embeddings = self.index.get_embeddings(candidate_ids) if hasattr(self.index, 'get_embeddings') else None
                if embeddings is not None:
                    # Compute attention-based scores
                    attn_weights = self.attention_head.compute_attention_weights(qvec, embeddings)
                    attention_scores = {cid: float(attn_weights[i]) for i, cid in enumerate(candidate_ids)}
        
        # Merge scores with optional attention
        merged = self.merge_scores(sem, lex, attention_scores)
        
        # Rerank with all signals
        ranked = self.rerank(merged, rq)
        
        # Update consolidation based on retrieval
        if ranked and self.use_attention:
            retrieved_ids = [c.memory_id for c in ranked[:10]]  # Top 10 results
            self.consolidator.hebbian_update(retrieved_ids)
            self.adaptive_embeddings.update_co_occurrence(retrieved_ids)
            
            # Periodic decay
            if np.random.random() < 0.1:  # 10% chance per search
                self.consolidator.synaptic_decay()
        
        return ranked
