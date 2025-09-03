from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional, Iterable, Dict, Any
from contextlib import contextmanager
from datetime import datetime
import json
import os

from ..types import MemoryRecord

SCHEMA = '''
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    who_type TEXT NOT NULL,
    who_id TEXT NOT NULL,
    who_label TEXT,
    what TEXT NOT NULL,
    when_ts TEXT NOT NULL,
    where_type TEXT NOT NULL,
    where_value TEXT NOT NULL,
    where_lat REAL,
    where_lon REAL,
    why TEXT,
    how TEXT,
    raw_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embed_model TEXT NOT NULL,
    extra_json TEXT,
    created_at TEXT NOT NULL
);

-- SQLite FTS5 for lexical retrieval (what/why/how/raw_text)
CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
    memory_id UNINDEXED,
    what,
    why,
    how,
    raw_text,
    content='',
    tokenize='porter'
);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_stats (
    memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    accesses INTEGER NOT NULL DEFAULT 0,
    last_access TEXT
);

-- Conceptual clusters (MiniBatchKMeans centroids)
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL, -- 'conceptual'
    label TEXT,
    dim INTEGER,
    centroid BLOB,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_membership (
    cluster_id INTEGER REFERENCES clusters(cluster_id) ON DELETE CASCADE,
    memory_id TEXT REFERENCES memories(memory_id) ON DELETE CASCADE,
    weight REAL NOT NULL,
    PRIMARY KEY (cluster_id, memory_id)
);

-- Blocks and membership (pointer-chained)
CREATE TABLE IF NOT EXISTS blocks (
    block_id TEXT PRIMARY KEY,
    query_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    budget_tokens INTEGER NOT NULL,
    used_tokens INTEGER NOT NULL,
    has_more INTEGER NOT NULL DEFAULT 0,
    prev_block_id TEXT,
    next_block_id TEXT,
    summary_text TEXT
);

CREATE TABLE IF NOT EXISTS block_members (
    block_id TEXT REFERENCES blocks(block_id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    memory_id TEXT REFERENCES memories(memory_id) ON DELETE CASCADE,
    PRIMARY KEY (block_id, rank)
);
'''

class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self):
        with self.connect() as con:
            con.executescript(SCHEMA)

    def upsert_memory(self, rec: MemoryRecord, embedding: bytes, dim: int):
        with self.connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO memories
                (memory_id, session_id, source_event_id, who_type, who_id, who_label, what, when_ts,
                 where_type, where_value, where_lat, where_lon, why, how, raw_text, token_count,
                 embed_model, extra_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                (
                    rec.memory_id, rec.session_id, rec.source_event_id, rec.who.type, rec.who.id, rec.who.label,
                    rec.what, rec.when.isoformat(), rec.where.type, rec.where.value, rec.where.lat, rec.where.lon,
                    rec.why, rec.how, rec.raw_text, rec.token_count, rec.embed_model, json.dumps(rec.extra),
                    datetime.utcnow().isoformat()
                )
            )
            con.execute(
                """INSERT OR REPLACE INTO mem_fts (memory_id, what, why, how, raw_text)
                VALUES (?, ?, ?, ?, ?)""",
                (rec.memory_id, rec.what, rec.why or '', rec.how or '', rec.raw_text)
            )
            con.execute(
                """INSERT OR REPLACE INTO embeddings (memory_id, dim, vector) VALUES (?, ?, ?)""", 
                (rec.memory_id, dim, embedding)
            )
            con.execute(
                """INSERT OR IGNORE INTO usage_stats (memory_id, accesses, last_access) VALUES (?, 0, ?)""", 
                (rec.memory_id, datetime.utcnow().isoformat())
            )

    def fetch_memories(self, ids: List[str]):
        qmarks = ','.join('?' * len(ids))
        with self.connect() as con:
            rows = con.execute(f"SELECT * FROM memories WHERE memory_id IN ({qmarks})", ids).fetchall()
        return rows

    def record_access(self, memory_ids: List[str]):
        if not memory_ids:
            return
        with self.connect() as con:
            now = datetime.utcnow().isoformat()
            for mid in memory_ids:
                con.execute(
                    """INSERT INTO usage_stats (memory_id, accesses, last_access)
                           VALUES (?, 1, ?)
                           ON CONFLICT(memory_id) DO UPDATE SET 
                             accesses = accesses + 1,
                             last_access = excluded.last_access""",
                    (mid, now)
                )

    def lexical_search(self, query: str, k: int = 50) -> List[sqlite3.Row]:
        # Use FTS5 bm25
        # Escape special FTS5 characters by quoting the entire query
        escaped_query = '"' + query.replace('"', '""') + '"'
        sql = """SELECT memory_id, bm25(mem_fts) AS score FROM mem_fts
                 WHERE mem_fts MATCH ?
                 ORDER BY score LIMIT ?"""
        with self.connect() as con:
            rows = con.execute(sql, (escaped_query, k)).fetchall()
        return rows

    # Blocks
    def create_block(self, block: Dict[str, Any], member_ids: List[str]):
        with self.connect() as con:
            con.execute(
                """INSERT INTO blocks
                (block_id, query_fingerprint, created_at, budget_tokens, used_tokens, has_more, prev_block_id, next_block_id, summary_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    block['block_id'], block['query_fingerprint'], block['created_at'],
                    block['budget_tokens'], block['used_tokens'], 1 if block['has_more'] else 0,
                    block.get('prev_block_id'), block.get('next_block_id'), block.get('summary_text')
                )
            )
            for rank, mid in enumerate(member_ids):
                con.execute(
                    """INSERT INTO block_members (block_id, rank, memory_id) VALUES (?, ?, ?)""", 
                    (block['block_id'], rank, mid)
                )

    def get_block(self, block_id: str) -> Dict[str, Any]:
        with self.connect() as con:
            b = con.execute("SELECT * FROM blocks WHERE block_id=?", (block_id,)).fetchone()
            if not b:
                return {}
            ms = con.execute("SELECT memory_id FROM block_members WHERE block_id=? ORDER BY rank", (block_id,)).fetchall()
        return {'block': dict(b), 'members': [m['memory_id'] for m in ms]}
