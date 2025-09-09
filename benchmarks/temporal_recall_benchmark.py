#!/usr/bin/env python3
"""
Enhanced temporal recall benchmark for JAM Memory System.
Tests temporal retrieval with various hint types and natural language queries.
"""

import sys
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.types import RetrievalQuery
from agentic_memory.extraction.temporal_parser import TemporalParser


@dataclass
class TemporalTestCase:
    """Test case for temporal recall evaluation."""
    query_text: str
    temporal_hint: Union[str, Tuple[str, str], Dict]
    temporal_type: str  # 'exact_date', 'date_range', 'relative', 'natural_language'
    expected_date_range: Tuple[str, str]  # Expected date range for validation
    description: str


class TemporalRecallBenchmark:
    """Comprehensive temporal recall testing."""
    
    def __init__(self):
        """Initialize benchmark components."""
        self.store = MemoryStore(cfg.get('db_path', 'amemory.sqlite3'))
        embed_dim = int(cfg.get('embed_dim', 1024))
        index_path = cfg.get('index_path', 'faiss.index')
        self.index = FaissIndex(embed_dim, index_path)
        self.retriever = HybridRetriever(self.store, self.index)
        self.parser = TemporalParser()
        
        # Get database connection for analysis
        self.db_path = cfg.get('db_path', 'amemory.sqlite3')
    
    def generate_test_cases(self) -> List[TemporalTestCase]:
        """Generate diverse temporal test cases."""
        test_cases = []
        
        # Get reference date (most recent memory date)
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            cursor = con.cursor()
            
            cursor.execute("""
                SELECT MAX(DATE(when_ts)) as latest_date
                FROM memories
                WHERE when_ts IS NOT NULL
            """)
            latest = cursor.fetchone()
            ref_date = datetime.fromisoformat(latest['latest_date']) if latest['latest_date'] else datetime.now()
        
        # 1. Exact date tests
        test_cases.append(TemporalTestCase(
            query_text="Show memories from this date",
            temporal_hint="2024-01-01",
            temporal_type="exact_date",
            expected_date_range=("2024-01-01", "2024-01-01"),
            description="Exact date: 2024-01-01"
        ))
        
        test_cases.append(TemporalTestCase(
            query_text="What happened on this day",
            temporal_hint="2025-09-07",
            temporal_type="exact_date",
            expected_date_range=("2025-09-07", "2025-09-07"),
            description="Exact date: 2025-09-07"
        ))
        
        # 2. Date range tests
        test_cases.append(TemporalTestCase(
            query_text="Show memories from this period",
            temporal_hint=("2025-09-07", "2025-09-08"),
            temporal_type="date_range",
            expected_date_range=("2025-09-07", "2025-09-08"),
            description="Date range: Sep 7-8, 2025"
        ))
        
        test_cases.append(TemporalTestCase(
            query_text="Activities during this time",
            temporal_hint=("2024-01-01", "2024-01-07"),
            temporal_type="date_range",
            expected_date_range=("2024-01-01", "2024-01-07"),
            description="Date range: First week of 2024"
        ))
        
        # 3. Relative time tests
        yesterday = (ref_date - timedelta(days=1)).strftime('%Y-%m-%d')
        test_cases.append(TemporalTestCase(
            query_text="What did we discuss",
            temporal_hint={"relative": "yesterday"},
            temporal_type="relative",
            expected_date_range=(yesterday, yesterday),
            description="Relative: yesterday"
        ))
        
        week_start = (ref_date - timedelta(days=7)).strftime('%Y-%m-%d')
        week_end = ref_date.strftime('%Y-%m-%d')
        test_cases.append(TemporalTestCase(
            query_text="Recent conversations",
            temporal_hint={"relative": "last_week"},
            temporal_type="relative",
            expected_date_range=(week_start, week_end),
            description="Relative: last week"
        ))
        
        # 4. Natural language tests (requires LLM parsing)
        natural_queries = [
            ("What did we talk about yesterday?", "yesterday"),
            ("Show me memories from last Tuesday", "specific_day"),
            ("Find conversations from two weeks ago", "weeks_ago"),
            ("What happened last month?", "last_month"),
        ]
        
        for query, hint_type in natural_queries:
            # Parse with LLM to get temporal hint
            temporal_hint, cleaned = self.parser.extract_and_clean_query(query)
            if temporal_hint:
                # Determine expected range based on parsed hint
                if isinstance(temporal_hint, str):
                    expected_range = (temporal_hint, temporal_hint)
                elif isinstance(temporal_hint, tuple):
                    expected_range = temporal_hint
                else:
                    # For relative hints, compute actual dates
                    if temporal_hint.get("relative") == "yesterday":
                        date = (ref_date - timedelta(days=1)).strftime('%Y-%m-%d')
                        expected_range = (date, date)
                    elif temporal_hint.get("relative") == "last_week":
                        start = (ref_date - timedelta(days=7)).strftime('%Y-%m-%d')
                        end = ref_date.strftime('%Y-%m-%d')
                        expected_range = (start, end)
                    elif temporal_hint.get("relative") == "last_month":
                        start = (ref_date - timedelta(days=30)).strftime('%Y-%m-%d')
                        end = ref_date.strftime('%Y-%m-%d')
                        expected_range = (start, end)
                    else:
                        continue
                
                test_cases.append(TemporalTestCase(
                    query_text=query,
                    temporal_hint=temporal_hint,
                    temporal_type="natural_language",
                    expected_date_range=expected_range,
                    description=f"Natural: {hint_type}"
                ))
        
        return test_cases
    
    def evaluate_test_case(self, test_case: TemporalTestCase, k: int = 20) -> Dict:
        """Evaluate a single temporal test case."""
        # Generate query vector
        qvec = np.random.randn(int(cfg.get('embed_dim', 1024))).astype('float32')
        qvec = qvec / np.linalg.norm(qvec)
        
        # Create retrieval query
        rq = RetrievalQuery(
            session_id="test_session",
            text=test_case.query_text,
            temporal_hint=test_case.temporal_hint
        )
        
        # Retrieve
        candidates = self.retriever.search(rq, qvec, topk_sem=k, topk_lex=k)
        
        # Get actual memories in expected date range
        start_date, end_date = test_case.expected_date_range
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            cursor = con.cursor()
            
            cursor.execute("""
                SELECT memory_id
                FROM memories
                WHERE DATE(when_ts) BETWEEN ? AND ?
            """, (start_date, end_date))
            
            expected_ids = {row['memory_id'] for row in cursor.fetchall()}
        
        # Calculate metrics
        retrieved_ids = {c.memory_id for c in candidates[:k]}
        
        # Check temporal accuracy of retrieved memories
        temporal_matches = 0
        for candidate in candidates[:k]:
            mem = self.store.fetch_memories([candidate.memory_id])[0]
            mem_date = mem['when_ts'].split('T')[0] if 'T' in mem['when_ts'] else mem['when_ts'][:10]
            if start_date <= mem_date <= end_date:
                temporal_matches += 1
        
        # Calculate recall (what fraction of expected memories were retrieved)
        if expected_ids:
            relevant_retrieved = retrieved_ids.intersection(expected_ids)
            recall = len(relevant_retrieved) / min(len(expected_ids), k)
        else:
            recall = 0.0
        
        # Calculate precision (what fraction of retrieved memories are from correct time)
        precision = temporal_matches / k if k > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'test_case': test_case.description,
            'temporal_type': test_case.temporal_type,
            'expected_range': test_case.expected_date_range,
            'expected_count': len(expected_ids),
            'retrieved_count': len(retrieved_ids),
            'temporal_matches': temporal_matches,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top_candidates': [
                {
                    'memory_id': c.memory_id,
                    'score': c.score,
                    'date': self.store.fetch_memories([c.memory_id])[0]['when_ts'].split('T')[0]
                }
                for c in candidates[:5]
            ]
        }
    
    def run_benchmark(self) -> Dict:
        """Run complete temporal benchmark."""
        print("\n" + "="*80)
        print("TEMPORAL RECALL BENCHMARK")
        print("="*80)
        
        test_cases = self.generate_test_cases()
        print(f"\nGenerated {len(test_cases)} test cases")
        
        results = []
        summary_by_type = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
        
        for test_case in test_cases:
            print(f"\nTesting: {test_case.description}")
            result = self.evaluate_test_case(test_case)
            results.append(result)
            
            # Update summary
            temporal_type = result['temporal_type']
            summary_by_type[temporal_type]['precision'].append(result['precision'])
            summary_by_type[temporal_type]['recall'].append(result['recall'])
            summary_by_type[temporal_type]['f1'].append(result['f1_score'])
            
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall:    {result['recall']:.3f}")
            print(f"  F1 Score:  {result['f1_score']:.3f}")
            print(f"  Temporal matches: {result['temporal_matches']}/20")
        
        # Calculate overall summary
        overall_precision = np.mean([r['precision'] for r in results])
        overall_recall = np.mean([r['recall'] for r in results])
        overall_f1 = np.mean([r['f1_score'] for r in results])
        
        print("\n" + "-"*80)
        print("SUMMARY BY TEMPORAL TYPE")
        print("-"*80)
        
        for temp_type, metrics in summary_by_type.items():
            avg_precision = np.mean(metrics['precision']) if metrics['precision'] else 0
            avg_recall = np.mean(metrics['recall']) if metrics['recall'] else 0
            avg_f1 = np.mean(metrics['f1']) if metrics['f1'] else 0
            
            print(f"\n{temp_type.upper()}:")
            print(f"  Avg Precision: {avg_precision:.3f}")
            print(f"  Avg Recall:    {avg_recall:.3f}")
            print(f"  Avg F1 Score:  {avg_f1:.3f}")
            print(f"  Test cases:    {len(metrics['precision'])}")
        
        print("\n" + "-"*80)
        print("OVERALL PERFORMANCE")
        print("-"*80)
        print(f"Overall Precision: {overall_precision:.3f}")
        print(f"Overall Recall:    {overall_recall:.3f}")
        print(f"Overall F1 Score:  {overall_f1:.3f}")
        
        return {
            'test_cases': results,
            'summary_by_type': {
                k: {
                    'avg_precision': np.mean(v['precision']) if v['precision'] else 0,
                    'avg_recall': np.mean(v['recall']) if v['recall'] else 0,
                    'avg_f1': np.mean(v['f1']) if v['f1'] else 0,
                    'count': len(v['precision'])
                }
                for k, v in summary_by_type.items()
            },
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'total_tests': len(results)
            }
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Temporal recall benchmark for JAM Memory System')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = TemporalRecallBenchmark()
    results = benchmark.run_benchmark()
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        results_dir =  "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"temporal_recall_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[OK] Results saved to {output_file}")


if __name__ == "__main__":
    main()
