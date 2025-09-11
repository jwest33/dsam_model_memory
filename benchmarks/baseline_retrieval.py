#!/usr/bin/env python3
"""
Simple baseline retrieval system for comparison.
Uses only basic lexical search without any 5W1H understanding or hints.
"""

import sys
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timezone

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.config import cfg


class BaselineRetriever:
    """Simple baseline retrieval using only lexical search."""
    
    def __init__(self):
        """Initialize baseline retriever."""
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
    
    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Simple BM25 lexical search without any sophistication.
        Ignores all hints and 5W1H structure.
        
        Returns: List of (memory_id, score) tuples
        """
        # Clean query for FTS5 - just use simple terms, not phrase search
        # Remove special characters that might break FTS5
        cleaned_query = query_text.replace('"', '').replace("'", '').replace('?', '').replace('!', '')
        
        # Simple FTS5 search on raw_text only
        cursor = self.db_conn.cursor()
        try:
            # Try FTS5 search
            cursor.execute("""
                SELECT m.memory_id, bm25(f) as score
                FROM memories m
                JOIN mem_fts f ON m.memory_id = f.memory_id
                WHERE f MATCH ?
                ORDER BY score
                LIMIT ?
            """, (cleaned_query, k))
        except (sqlite3.OperationalError, Exception) as e:
            # If FTS5 query fails, fall back to simple LIKE search
            try:
                search_term = cleaned_query[:50] if cleaned_query else "memory"
                cursor.execute("""
                    SELECT memory_id
                    FROM memories
                    WHERE raw_text LIKE ?
                    ORDER BY LENGTH(raw_text)
                    LIMIT ?
                """, (f'%{search_term}%', k))
            except:
                # Last resort - just get some random memories
                cursor.execute("""
                    SELECT memory_id
                    FROM memories
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (k,))
        
        results = []
        for i, row in enumerate(cursor.fetchall()):
            if 'score' in row.keys():
                # BM25 returns negative scores (more negative = better match)
                # Convert to positive score between 0 and 1
                score = 1.0 / (1.0 + abs(row['score']))
            else:
                # For LIKE search, give decreasing scores
                score = 1.0 / (1.0 + i)
            results.append((row['memory_id'], score))
        
        # If no FTS matches, fall back to random sampling
        if not results:
            cursor.execute("""
                SELECT memory_id
                FROM memories
                ORDER BY RANDOM()
                LIMIT ?
            """, (k,))
            
            for i, row in enumerate(cursor.fetchall()):
                # Give decreasing scores to random results
                score = 0.1 * (1.0 - i/k)
                results.append((row['memory_id'], score))
        
        return results


class SimpleSemanticRetriever:
    """
    Slightly smarter baseline using basic semantic search.
    Still ignores hints and 5W1H structure.
    """
    
    def __init__(self):
        """Initialize semantic baseline."""
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
        
        # Try to use embeddings if available
        try:
            from agentic_memory.storage.faiss_index import FaissIndex
            from agentic_memory.embedding import get_llama_embedder
            self.index = FaissIndex(dim=int(cfg.get('embedding_dim', 1024)), index_path=cfg.index_path)
            self.embedder = get_llama_embedder()
            self.has_embeddings = True
        except:
            self.has_embeddings = False
    
    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Simple semantic search without any hints or reranking.
        
        Returns: List of (memory_id, score) tuples
        """
        if not self.has_embeddings:
            # Fall back to lexical search
            baseline = BaselineRetriever()
            return baseline.search(query_text, k)
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query_text], normalize_embeddings=True)[0]
        
        # Simple FAISS search
        results = self.index.search(query_embedding, k=k)
        
        if not results:
            # Fall back to lexical if no semantic matches
            baseline = BaselineRetriever()
            return baseline.search(query_text, k)
        
        return results


class CombinedBaselineRetriever:
    """
    Best baseline: Combines lexical and semantic with fixed weights.
    Still ignores all hints and 5W1H structure.
    """
    
    def __init__(self):
        """Initialize combined baseline."""
        self.lexical = BaselineRetriever()
        self.semantic = SimpleSemanticRetriever()
    
    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Combine lexical and semantic search with fixed 50/50 weighting.
        No hints, no reranking, no 5W1H understanding.
        
        Returns: List of (memory_id, score) tuples
        """
        # Get results from both
        lex_results = self.lexical.search(query_text, k=k*2)
        sem_results = self.semantic.search(query_text, k=k*2)
        
        # Merge with fixed weights (50% lexical, 50% semantic)
        scores = {}
        
        for memory_id, score in lex_results:
            scores[memory_id] = scores.get(memory_id, 0) + 0.5 * score
        
        for memory_id, score in sem_results:
            scores[memory_id] = scores.get(memory_id, 0) + 0.5 * score
        
        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:k]


def get_baseline_retriever(baseline_type: str = 'combined'):
    """
    Get a baseline retriever.
    
    Args:
        baseline_type: 'lexical', 'semantic', or 'combined'
    """
    if baseline_type == 'lexical':
        return BaselineRetriever()
    elif baseline_type == 'semantic':
        return SimpleSemanticRetriever()
    else:
        return CombinedBaselineRetriever()