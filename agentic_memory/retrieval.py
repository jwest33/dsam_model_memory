from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime, timezone
from .config import cfg
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .types import RetrievalQuery, Candidate

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

    def merge_scores(self, sem: List[Tuple[str, float]], lex: List[Tuple[str, float]]) -> Dict[str, float]:
        sdict = {m:s for m,s in sem}
        ldict = {m:s for m,s in lex}
        keys = set(sdict.keys()) | set(ldict.keys())
        merged = {}
        for k in keys:
            merged[k] = cfg.w_semantic * sdict.get(k,0.0) + cfg.w_lexical * ldict.get(k,0.0)
        return merged

    def _fetch_meta(self, ids: List[str]):
        rows = self.store.fetch_memories(ids)
        by_id = {r['memory_id']: r for r in rows}
        return by_id

    def rerank(self, merged: Dict[str, float], rq: RetrievalQuery) -> List[Candidate]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        metas = self._fetch_meta(list(merged.keys()))
        cands = []
        for mid, base in merged.items():
            m = metas.get(mid)
            if not m:
                continue
            # Extra signals
            rec = exp_recency(m['when_ts'], now)
            actor_match = 1.0 if (rq.actor_hint and m['who_id'] == rq.actor_hint) else 0.0
            spatial_match = 1.0 if (rq.spatial_hint and m['where_value'] == rq.spatial_hint) else 0.0
            # usage
            # we'll pull usage inline by join to simplify (not implemented => assume 0)
            usage_boost = 0.2 if actor_match or spatial_match else 0.0
            score = base + cfg.w_recency*rec + cfg.w_actor*actor_match + cfg.w_spatial*spatial_match + cfg.w_usage*usage_boost
            cands.append(Candidate(memory_id=mid, score=score, token_count=int(m['token_count'])))
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def search(self, rq: RetrievalQuery, qvec: np.ndarray, topk_sem: int = 50, topk_lex: int = 50) -> List[Candidate]:
        sem = self._semantic(qvec, topk_sem)
        lex = self._lexical(rq.text, topk_lex)
        merged = self.merge_scores(sem, lex)
        ranked = self.rerank(merged, rq)
        return ranked
