from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional, Iterable, Dict, Any
from contextlib import contextmanager
from datetime import datetime
import json
import os
import numpy as np

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

-- Synaptic connections between memories (Hebbian learning)
CREATE TABLE IF NOT EXISTS memory_synapses (
    memory_id1 TEXT NOT NULL,
    memory_id2 TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    last_activation TEXT NOT NULL,
    PRIMARY KEY (memory_id1, memory_id2),
    CHECK (memory_id1 < memory_id2)  -- Ensure unique pairs
);

-- Memory importance scores (PageRank-style)
CREATE TABLE IF NOT EXISTS memory_importance (
    memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    importance_score REAL NOT NULL DEFAULT 0.0,
    connection_count INTEGER NOT NULL DEFAULT 0,
    last_computed TEXT NOT NULL
);

-- Embedding drift tracking
CREATE TABLE IF NOT EXISTS embedding_drift (
    memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    drift_vector BLOB,
    momentum_rate REAL DEFAULT 0.95,
    last_update TEXT NOT NULL
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

    def get_by_actor(self, actor_id: str, limit: int = 100) -> List[sqlite3.Row]:
        """Retrieve memories from a specific actor."""
        sql = """
            SELECT memory_id, who_id, raw_text, when_ts, token_count
            FROM memories
            WHERE who_id = ?
            ORDER BY when_ts DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (actor_id, limit)).fetchall()
        return rows
    
    def get_by_location(self, location: str, limit: int = 100) -> List[sqlite3.Row]:
        """Retrieve memories from a specific location."""
        sql = """
            SELECT memory_id, where_value, raw_text, when_ts, token_count
            FROM memories
            WHERE where_value = ?
            ORDER BY when_ts DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (location, limit)).fetchall()
        return rows
    
    def get_by_actor_and_text(self, actor_id: str, query: str, limit: int = 50) -> List[sqlite3.Row]:
        """Hybrid search within an actor's memories."""
        escaped_query = '"' + query.replace('"', '""') + '"'
        sql = """
            SELECT m.memory_id, bm25(f) AS score, m.when_ts, m.token_count
            FROM memories m
            JOIN mem_fts f ON m.memory_id = f.memory_id
            WHERE m.who_id = ? AND f MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (actor_id, escaped_query, limit)).fetchall()
        return rows
    
    def actor_exists(self, actor_id: str) -> bool:
        """Check if an actor exists in the database."""
        sql = "SELECT COUNT(*) FROM memories WHERE who_id = ? LIMIT 1"
        with self.connect() as con:
            count = con.execute(sql, (actor_id,)).fetchone()[0]
        return count > 0

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
    
    def get_usage_stats(self, memory_ids: List[str]) -> Dict[str, Dict]:
        """Get usage statistics for multiple memories"""
        if not memory_ids:
            return {}
        
        qmarks = ','.join('?' * len(memory_ids))
        with self.connect() as con:
            rows = con.execute(
                f"SELECT memory_id, accesses, last_access FROM usage_stats WHERE memory_id IN ({qmarks})",
                memory_ids
            ).fetchall()
        
        return {row['memory_id']: dict(row) for row in rows}
    
    def update_synapse(self, memory_id1: str, memory_id2: str, weight_delta: float):
        """Update synaptic weight between two memories"""
        # Ensure consistent ordering
        m1, m2 = (memory_id1, memory_id2) if memory_id1 < memory_id2 else (memory_id2, memory_id1)
        now = datetime.utcnow().isoformat()
        
        with self.connect() as con:
            con.execute(
                """INSERT INTO memory_synapses (memory_id1, memory_id2, weight, last_activation)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(memory_id1, memory_id2) DO UPDATE SET
                     weight = MIN(1.0, weight + ?),
                     last_activation = ?""",
                (m1, m2, weight_delta, now, weight_delta, now)
            )
    
    def get_synapses(self, memory_id: str, threshold: float = 0.01) -> List[Tuple[str, float]]:
        """Get all synaptic connections for a memory above threshold"""
        with self.connect() as con:
            rows = con.execute(
                """SELECT memory_id2 as other, weight FROM memory_synapses
                   WHERE memory_id1 = ? AND weight >= ?
                   UNION
                   SELECT memory_id1 as other, weight FROM memory_synapses
                   WHERE memory_id2 = ? AND weight >= ?""",
                (memory_id, threshold, memory_id, threshold)
            ).fetchall()
        
        return [(r['other'], r['weight']) for r in rows]
    
    def update_importance(self, memory_id: str, importance_score: float, connection_count: int):
        """Update importance score for a memory"""
        now = datetime.utcnow().isoformat()
        
        with self.connect() as con:
            con.execute(
                """INSERT INTO memory_importance (memory_id, importance_score, connection_count, last_computed)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(memory_id) DO UPDATE SET
                     importance_score = ?,
                     connection_count = ?,
                     last_computed = ?""",
                (memory_id, importance_score, connection_count, now,
                 importance_score, connection_count, now)
            )
    
    def get_importance_scores(self, memory_ids: List[str]) -> Dict[str, float]:
        """Get importance scores for multiple memories"""
        if not memory_ids:
            return {}
        
        qmarks = ','.join('?' * len(memory_ids))
        with self.connect() as con:
            rows = con.execute(
                f"SELECT memory_id, importance_score FROM memory_importance WHERE memory_id IN ({qmarks})",
                memory_ids
            ).fetchall()
        
        return {row['memory_id']: row['importance_score'] for row in rows}
    
    def decay_synapses(self, decay_rate: float = 0.001):
        """Apply decay to all synaptic weights"""
        with self.connect() as con:
            con.execute(
                """UPDATE memory_synapses 
                   SET weight = weight * ?
                   WHERE weight > 0""",
                (1.0 - decay_rate,)
            )
            # Remove very weak connections
            con.execute("DELETE FROM memory_synapses WHERE weight < 0.01")
    
    def store_embedding_drift(self, memory_id: str, drift_vector: np.ndarray):
        """Store embedding drift vector for a memory"""
        now = datetime.utcnow().isoformat()
        drift_blob = drift_vector.tobytes()
        
        with self.connect() as con:
            con.execute(
                """INSERT INTO embedding_drift (memory_id, drift_vector, last_update)
                   VALUES (?, ?, ?)
                   ON CONFLICT(memory_id) DO UPDATE SET
                     drift_vector = ?,
                     last_update = ?""",
                (memory_id, drift_blob, now, drift_blob, now)
            )
    
    def get_embedding_drift(self, memory_id: str) -> Optional[np.ndarray]:
        """Get embedding drift vector for a memory"""
        with self.connect() as con:
            row = con.execute(
                "SELECT drift_vector FROM embedding_drift WHERE memory_id = ?",
                (memory_id,)
            ).fetchone()
        
        if row and row['drift_vector']:
            return np.frombuffer(row['drift_vector'], dtype=np.float32)
        return None
