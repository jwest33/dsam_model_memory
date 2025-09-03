from __future__ import annotations
from typing import List
from .storage.sql_store import MemoryStore

class UsageTracker:
    def __init__(self, store: MemoryStore):
        self.store = store

    def note_access(self, memory_ids: List[str]):
        self.store.record_access(memory_ids)
