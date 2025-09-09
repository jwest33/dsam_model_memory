#!/usr/bin/env python3
"""
Recall benchmark for JAM Memory System.
Tests retrieval precision, recall, and F1 scores across different query types.
"""

import sys
import time
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from collections import defaultdict
import random

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.types import RetrievalQuery
from agentic_memory.embedding import get_llama_embedder


@dataclass
class RecallMetrics:
    """Metrics for recall evaluation."""
    precision: float  # Retrieved & Relevant / Retrieved
    recall: float     # Retrieved & Relevant / Total Relevant
    f1_score: float   # Harmonic mean of precision and recall
    mrr: float        # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg: float       # Normalized Discounted Cumulative Gain
    
    
@dataclass
class QueryTestCase:
    """A test case for recall evaluation."""
    query_text: str
    query_type: str  # 'exact', 'semantic', 'temporal', 'actor', 'session'
    relevant_memory_ids: Set[str]
    session_id: Optional[str] = None
    actor_hint: Optional[str] = None
    temporal_hint: Optional[Union[str, Tuple[str, str], Dict]] = None
    metadata: Dict = None


class RecallBenchmark:
    """Comprehensive recall testing for memory retrieval."""
    
    def __init__(self):
        """Initialize benchmark components."""
        self.store = MemoryStore(cfg.db_path)
        embed_dim = int(cfg.get('embed_dim', 1024))
        self.index = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
        self.router = MemoryRouter(self.store, self.index)
        self.retriever = HybridRetriever(self.store, self.index)
        
        # Try to initialize embedder
        try:
            self.embedder = get_llama_embedder()
            test_embedding = self.embedder.encode(["test"], normalize_embeddings=True)
            self.embedder_available = True
        except:
            print("Warning: Embedding server not available, using fallback")
            self.embedder = None
            self.embedder_available = False
            
        # Database connection
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
        
    def generate_test_cases(self, num_cases_per_type: int = 10) -> List[QueryTestCase]:
        """Generate diverse test cases with known ground truth."""
        test_cases = []
        cursor = self.db_conn.cursor()
        
        # 1. EXACT MATCH TEST CASES - Find memories with specific phrases
        print("Generating exact match test cases...")
        cursor.execute("""
            SELECT memory_id, raw_text, session_id, who_id
            FROM memories
            WHERE LENGTH(raw_text) > 50 AND LENGTH(raw_text) < 500
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases_per_type * 2,))
        
        exact_memories = cursor.fetchall()
        for i in range(min(num_cases_per_type, len(exact_memories))):
            mem = exact_memories[i]
            # Extract a meaningful phrase from the middle
            text = mem['raw_text']
            words = text.split()
            if len(words) > 10:
                start_idx = len(words) // 4
                query_phrase = ' '.join(words[start_idx:start_idx+5])
                
                # Find all memories containing this exact phrase
                cursor.execute("""
                    SELECT memory_id FROM memories
                    WHERE raw_text LIKE ?
                    LIMIT 20
                """, (f'%{query_phrase}%',))
                
                relevant_ids = {row['memory_id'] for row in cursor.fetchall()}
                if relevant_ids:
                    test_cases.append(QueryTestCase(
                        query_text=query_phrase,
                        query_type='exact',
                        relevant_memory_ids=relevant_ids,
                        session_id=mem['session_id']
                    ))
        
        # 2. SEMANTIC SIMILARITY TEST CASES - Find topically related memories
        print("Generating semantic similarity test cases...")
        topics = [
            "programming", "health", "travel", "food", "technology",
            "education", "business", "science", "music", "sports"
        ]
        
        for topic in topics[:num_cases_per_type]:
            # Find memories mentioning the topic
            cursor.execute("""
                SELECT memory_id, raw_text, session_id
                FROM memories
                WHERE raw_text LIKE ?
                LIMIT 30
            """, (f'%{topic}%',))
            
            topic_memories = cursor.fetchall()
            if len(topic_memories) >= 5:
                relevant_ids = {row['memory_id'] for row in topic_memories}
                # Use first memory's text as query
                query_text = topic_memories[0]['raw_text'][:200]
                
                test_cases.append(QueryTestCase(
                    query_text=query_text,
                    query_type='semantic',
                    relevant_memory_ids=relevant_ids,
                    session_id=topic_memories[0]['session_id']
                ))
        
        # 3. SESSION-BASED TEST CASES - Retrieve all memories from same session
        print("Generating session-based test cases...")
        cursor.execute("""
            SELECT session_id, COUNT(*) as count
            FROM memories
            GROUP BY session_id
            HAVING count >= 5 AND count <= 20
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases_per_type,))
        
        sessions = cursor.fetchall()
        for session in sessions:
            # Get all memories from this session
            cursor.execute("""
                SELECT memory_id, raw_text
                FROM memories
                WHERE session_id = ?
            """, (session['session_id'],))
            
            session_memories = cursor.fetchall()
            if session_memories:
                relevant_ids = {row['memory_id'] for row in session_memories}
                # Use middle memory as query
                query_mem = session_memories[len(session_memories)//2]
                
                test_cases.append(QueryTestCase(
                    query_text=query_mem['raw_text'][:200],
                    query_type='session',
                    relevant_memory_ids=relevant_ids,
                    session_id=session['session_id']
                ))
        
        # 4. ACTOR-BASED TEST CASES - Find all memories from same actor
        print("Generating actor-based test cases...")
        cursor.execute("""
            SELECT who_id, COUNT(*) as count
            FROM memories
            GROUP BY who_id
            HAVING count >= 10 AND count <= 50
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases_per_type,))
        
        actors = cursor.fetchall()
        for actor in actors:
            cursor.execute("""
                SELECT memory_id, raw_text, session_id
                FROM memories
                WHERE who_id = ?
                LIMIT 30
            """, (actor['who_id'],))
            
            actor_memories = cursor.fetchall()
            if actor_memories:
                relevant_ids = {row['memory_id'] for row in actor_memories}
                query_mem = random.choice(actor_memories)
                
                test_cases.append(QueryTestCase(
                    query_text=query_mem['raw_text'][:200],
                    query_type='actor',
                    relevant_memory_ids=relevant_ids,
                    session_id=query_mem['session_id'],
                    actor_hint=actor['who_id']
                ))
        
        # 5. TEMPORAL TEST CASES - Find memories from same time period
        print("Generating temporal test cases...")
        cursor.execute("""
            SELECT DATE(when_ts) as date, COUNT(*) as count
            FROM memories
            WHERE when_ts IS NOT NULL
            GROUP BY DATE(when_ts)
            HAVING count >= 5
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases_per_type,))
        
        dates = cursor.fetchall()
        for date_row in dates:
            cursor.execute("""
                SELECT memory_id, raw_text, session_id
                FROM memories
                WHERE DATE(when_ts) = ?
            """, (date_row['date'],))
            
            temporal_memories = cursor.fetchall()
            if temporal_memories:
                relevant_ids = {row['memory_id'] for row in temporal_memories}
                query_mem = random.choice(temporal_memories)
                
                test_cases.append(QueryTestCase(
                    query_text=query_mem['raw_text'][:200],
                    query_type='temporal',
                    relevant_memory_ids=relevant_ids,
                    session_id=query_mem['session_id'],
                    temporal_hint=date_row['date'],  # Use actual date as temporal hint
                    metadata={'date': date_row['date']}
                ))
        
        print(f"Generated {len(test_cases)} test cases across {len(set(tc.query_type for tc in test_cases))} types")
        return test_cases
    
    def evaluate_retrieval(self, test_case: QueryTestCase, k_values: List[int] = [5, 10, 20]) -> Dict[int, RecallMetrics]:
        """Evaluate retrieval for a single test case at different k values."""
        results = {}
        
        # Generate embedding for query
        if self.embedder_available and self.embedder:
            try:
                qvec = self.embedder.encode([test_case.query_text], normalize_embeddings=True)[0]
            except:
                qvec = np.random.randn(int(cfg.get('embed_dim', 1024))).astype('float32')
                qvec = qvec / np.linalg.norm(qvec)
        else:
            qvec = np.random.randn(int(cfg.get('embed_dim', 1024))).astype('float32')
            qvec = qvec / np.linalg.norm(qvec)
        
        # Create retrieval query
        rq = RetrievalQuery(
            session_id=test_case.session_id or "test_session",
            text=test_case.query_text,
            actor_hint=test_case.actor_hint,
            temporal_hint=test_case.temporal_hint
        )
        
        # Retrieve with maximum k
        max_k = max(k_values)
        try:
            candidates = self.retriever.search(rq, qvec, topk_sem=max_k, topk_lex=max_k)
        except Exception as e:
            print(f"Retrieval error: {e}")
            # Return zero metrics
            for k in k_values:
                results[k] = RecallMetrics(0, 0, 0, 0, 0, 0)
            return results
        
        # Calculate metrics for each k
        for k in k_values:
            top_k_candidates = candidates[:k]
            retrieved_ids = {c.memory_id for c in top_k_candidates}
            
            # Calculate intersection
            relevant_retrieved = retrieved_ids.intersection(test_case.relevant_memory_ids)
            
            # Precision: What fraction of retrieved items are relevant?
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            
            # Recall: What fraction of relevant items were retrieved?
            recall = len(relevant_retrieved) / len(test_case.relevant_memory_ids) if test_case.relevant_memory_ids else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Mean Reciprocal Rank (MRR)
            mrr = 0
            for i, candidate in enumerate(top_k_candidates, 1):
                if candidate.memory_id in test_case.relevant_memory_ids:
                    mrr = 1.0 / i
                    break
            
            # Mean Average Precision (MAP)
            avg_precision = 0
            relevant_count = 0
            for i, candidate in enumerate(top_k_candidates, 1):
                if candidate.memory_id in test_case.relevant_memory_ids:
                    relevant_count += 1
                    avg_precision += relevant_count / i
            map_score = avg_precision / len(test_case.relevant_memory_ids) if test_case.relevant_memory_ids else 0
            
            # Normalized Discounted Cumulative Gain (NDCG)
            dcg = 0
            for i, candidate in enumerate(top_k_candidates, 1):
                if candidate.memory_id in test_case.relevant_memory_ids:
                    dcg += 1.0 / np.log2(i + 1)
            
            # Ideal DCG (all relevant items at top)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_case.relevant_memory_ids))))
            ndcg = dcg / idcg if idcg > 0 else 0
            
            results[k] = RecallMetrics(precision, recall, f1, mrr, map_score, ndcg)
        
        return results
    
    def run_benchmark(self, num_cases_per_type: int = 5, k_values: List[int] = [5, 10, 20]) -> Dict:
        """Run complete recall benchmark."""
        print("\n" + "="*60)
        print("RECALL BENCHMARK")
        print("="*60)
        
        # Generate test cases
        test_cases = self.generate_test_cases(num_cases_per_type)
        
        # Results storage
        results_by_type = defaultdict(lambda: defaultdict(list))
        overall_results = defaultdict(list)
        
        # Evaluate each test case
        print(f"\nEvaluating {len(test_cases)} test cases...")
        for i, test_case in enumerate(test_cases, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_cases)}")
            
            metrics = self.evaluate_retrieval(test_case, k_values)
            
            for k, metric in metrics.items():
                # Store by type
                results_by_type[test_case.query_type][k].append(metric)
                # Store overall
                overall_results[k].append(metric)
        
        # Calculate averages
        summary = {
            'overall': {},
            'by_type': {}
        }
        
        # Overall averages
        for k in k_values:
            if overall_results[k]:
                metrics_list = overall_results[k]
                summary['overall'][f'k_{k}'] = {
                    'precision': np.mean([m.precision for m in metrics_list]),
                    'recall': np.mean([m.recall for m in metrics_list]),
                    'f1_score': np.mean([m.f1_score for m in metrics_list]),
                    'mrr': np.mean([m.mrr for m in metrics_list]),
                    'map': np.mean([m.map_score for m in metrics_list]),
                    'ndcg': np.mean([m.ndcg for m in metrics_list])
                }
        
        # By type averages
        for query_type, type_results in results_by_type.items():
            summary['by_type'][query_type] = {}
            for k in k_values:
                if type_results[k]:
                    metrics_list = type_results[k]
                    summary['by_type'][query_type][f'k_{k}'] = {
                        'precision': np.mean([m.precision for m in metrics_list]),
                        'recall': np.mean([m.recall for m in metrics_list]),
                        'f1_score': np.mean([m.f1_score for m in metrics_list]),
                        'count': len(metrics_list)
                    }
        
        # Print results
        self.print_results(summary, test_cases)
        
        return summary
    
    def print_results(self, summary: Dict, test_cases: List[QueryTestCase]):
        """Print formatted benchmark results."""
        print("\n" + "-"*60)
        print("OVERALL RECALL METRICS")
        print("-"*60)
        
        for k_key, metrics in summary['overall'].items():
            k = k_key.replace('k_', '')
            print(f"\n@{k} Results:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1_score']:.3f}")
            print(f"  MRR:       {metrics['mrr']:.3f}")
            print(f"  MAP:       {metrics['map']:.3f}")
            print(f"  NDCG:      {metrics['ndcg']:.3f}")
        
        print("\n" + "-"*60)
        print("RECALL BY QUERY TYPE")
        print("-"*60)
        
        for query_type, type_metrics in summary['by_type'].items():
            print(f"\n{query_type.upper()} Queries:")
            for k_key, metrics in type_metrics.items():
                k = k_key.replace('k_', '')
                print(f"  @{k}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f} (n={metrics['count']})")
        
        # Test case distribution
        print("\n" + "-"*60)
        print("TEST CASE DISTRIBUTION")
        print("-"*60)
        type_counts = defaultdict(int)
        for tc in test_cases:
            type_counts[tc.query_type] += 1
        
        for query_type, count in type_counts.items():
            print(f"  {query_type}: {count} cases")
        print(f"  Total: {len(test_cases)} cases")


def main():
    """Main entry point for recall benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recall benchmark for JAM Memory System')
    parser.add_argument('--cases', type=int, default=5,
                       help='Number of test cases per query type')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20],
                       help='k values for recall@k evaluation')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = RecallBenchmark()
    results = benchmark.run_benchmark(
        num_cases_per_type=args.cases,
        k_values=args.k
    )
    
    # Always save results for analysis
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"recall_benchmark_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {output_file}")


if __name__ == "__main__":
    main()
