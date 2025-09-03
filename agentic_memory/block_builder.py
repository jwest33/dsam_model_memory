from __future__ import annotations
from typing import List, Tuple, Dict, Any
import hashlib
from datetime import datetime
import numpy as np
from .config import cfg
from .types import RetrievalQuery, MemoryBlock, Candidate
from .tokenization import TokenizerAdapter
from .storage.sql_store import MemoryStore

def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def greedy_knapsack(cands: List[Candidate], budget: int) -> Tuple[List[str], int]:
    # Sort by score/token_count ratio to maximize utility under token budget
    ordered = sorted(cands, key=lambda c: (c.score / max(1, c.token_count)), reverse=True)
    picked = []
    used = 0
    for c in ordered:
        if used + c.token_count <= budget:
            picked.append(c.memory_id)
            used += c.token_count
    return picked, used

class BlockBuilder:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.tok = TokenizerAdapter()

    def build(self, rq: RetrievalQuery, ranked: List[Candidate], context_overhead: int) -> List[MemoryBlock]:
        # Compute budget
        budget = cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens - context_overhead
        budget = max(512, budget)  # ensure sane minimum
        qfp = fingerprint(rq.text)
        blocks: List[MemoryBlock] = []

        remaining = list(ranked)
        prev_block_id = None
        while remaining:
            picked_ids, used = greedy_knapsack(remaining, budget)
            if not picked_ids:
                # force smallest item
                smallest = min(remaining, key=lambda c: c.token_count)
                picked_ids = [smallest.memory_id]
                used = smallest.token_count
            has_more = len(picked_ids) < len(remaining)
            blk = MemoryBlock(
                query_fingerprint=qfp,
                budget_tokens=budget,
                used_tokens=used,
                has_more=has_more,
                prev_block_id=prev_block_id,
                member_ids=picked_ids
            )
            blocks.append(blk)

            # Persist block
            self.store.create_block(
                {
                    'block_id': blk.block_id,
                    'query_fingerprint': blk.query_fingerprint,
                    'created_at': blk.created_at.isoformat(),
                    'budget_tokens': blk.budget_tokens,
                    'used_tokens': blk.used_tokens,
                    'has_more': blk.has_more,
                    'prev_block_id': blk.prev_block_id,
                    'next_block_id': None,
                    'summary_text': blk.summary_text
                },
                member_ids=picked_ids
            )
            if prev_block_id is not None:
                # back-link previous block to this one
                # (for simplicity, we won't update here; UI can resolve through store.get_block if needed)
                pass
            prev_block_id = blk.block_id

            # Remove picked from remaining
            picked_set = set(picked_ids)
            remaining = [c for c in remaining if c.memory_id not in picked_set]

        # update next_block_id pointers
        for i in range(len(blocks) - 1):
            b = blocks[i]
            b.next_block_id = blocks[i+1].block_id
        return blocks
