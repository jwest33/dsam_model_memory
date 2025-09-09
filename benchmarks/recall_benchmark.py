#!/usr/bin/env python3
"""
Recall benchmark for JAM memory system.
Evaluates retrieval performance using pre-generated test sets.
"""

import sys
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.config import cfg
from agentic_memory.types import RetrievalQuery


@dataclass
class QueryTestCase:
    """A test case for recall evaluation."""
    test_id: str
    test_type: str
    query_text: str
    query_metadata: Dict
    expected_relevant: List[str]
    ground_truth_metadata: Dict
    difficulty: str


@dataclass
class RecallMetrics:
    """Metrics for recall evaluation."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    ndcg: float  # Normalized Discounted Cumulative Gain
    
    def to_dict(self):
        return asdict(self)


class RecallBenchmark:
    """Benchmark for evaluating memory recall performance."""
    
    def __init__(self, testset_path: Optional[str] = None):
        """Initialize benchmark with optional test set."""
        # Initialize memory system
        store = MemoryStore(cfg.db_path)
        index = FaissIndex(dim=int(cfg.get('embedding_dim', 1024)), index_path=cfg.index_path)
        self.router = MemoryRouter(store, index)
        
        # Database connection for analysis
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
        
        # Load test set if provided
        self.test_cases = []
        if testset_path:
            self.load_testset(testset_path)
    
    def load_testset(self, filepath: str) -> List[QueryTestCase]:
        """Load test cases from JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            # Check in test_data directory
            test_data_path = Path("benchmarks/test_data") / filepath.name
            if test_data_path.exists():
                filepath = test_data_path
            else:
                raise FileNotFoundError(f"Test set not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.test_cases = []
        for tc_dict in data['test_cases']:
            # Handle both old semantic testset format and new benchmark format
            if 'paraphrased_query' in tc_dict:
                # Old semantic testset format
                query_text = tc_dict['paraphrased_query']
                test_type = tc_dict.get('query_type', 'semantic')
                query_metadata = tc_dict.get('metadata', {})
                # Map old metadata to new format
                if 'session_id' in query_metadata:
                    query_metadata = {'session_id': query_metadata['session_id']}
            else:
                # New benchmark format
                query_text = tc_dict['query_text']
                test_type = tc_dict['test_type']
                query_metadata = tc_dict.get('query_metadata', {})
            
            self.test_cases.append(QueryTestCase(
                test_id=tc_dict['test_id'],
                test_type=test_type,
                query_text=query_text,
                query_metadata=query_metadata,
                expected_relevant=tc_dict['expected_relevant'],
                ground_truth_metadata=tc_dict.get('ground_truth_metadata', {}),
                difficulty=tc_dict.get('difficulty', 'medium')
            ))
        
        print(f"Loaded {len(self.test_cases)} test cases from {filepath}")
        
        # Show distribution
        type_counts = defaultdict(int)
        for tc in self.test_cases:
            type_counts[tc.test_type] += 1
        
        print("Test case distribution:")
        for test_type, count in sorted(type_counts.items()):
            print(f"  {test_type}: {count}")
        
        return self.test_cases
    
    def find_latest_testset(self) -> Optional[Path]:
        """Find the most recent test set file."""
        test_data_dir = Path("benchmarks/test_data")
        if not test_data_dir.exists():
            return None
        
        # Look for benchmark testset files
        testset_files = list(test_data_dir.glob("benchmark_testset_*.json"))
        
        if not testset_files:
            # Fall back to semantic testset files
            testset_files = list(test_data_dir.glob("semantic_testset_*.json"))
        
        if testset_files:
            # Sort by modification time and return the newest
            return max(testset_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def evaluate_retrieval(self, 
                          test_case: QueryTestCase,
                          k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate retrieval for a single test case."""
        
        # Build retrieval query
        retrieval_query = RetrievalQuery(
            session_id=test_case.query_metadata.get('session_id', 'benchmark_session'),
            text=test_case.query_text,
            actor_hint=test_case.query_metadata.get('actor_hint'),
            spatial_hint=test_case.query_metadata.get('spatial_hint'),
            temporal_hint=test_case.query_metadata.get('temporal_hint')
        )
        
        # Retrieve memories
        start_time = time.time()
        candidates = self.router.retrieve(
            query=retrieval_query,
            budget_tokens=8000,  # Large budget to get many candidates
            diversity_weight=0.0  # No diversity for benchmark
        )
        retrieval_time = time.time() - start_time
        
        # Extract retrieved memory IDs
        retrieved_ids = [c.memory_id for c in candidates]
        
        # Convert expected to set for faster lookup
        expected_set = set(test_case.expected_relevant)
        
        # Calculate metrics at different k values
        results = {
            'test_id': test_case.test_id,
            'test_type': test_case.test_type,
            'difficulty': test_case.difficulty,
            'retrieval_time': retrieval_time,
            'total_retrieved': len(retrieved_ids),
            'total_expected': len(expected_set),
            'first_relevant_position': None,  # Position of first relevant result
            'metrics': {}
        }
        
        for k in k_values:
            retrieved_at_k = retrieved_ids[:k]
            retrieved_set_at_k = set(retrieved_at_k)
            
            # Calculate metrics
            true_positives = len(retrieved_set_at_k & expected_set)
            
            precision = true_positives / k if k > 0 else 0
            recall = true_positives / len(expected_set) if expected_set else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['metrics'][f'@{k}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': true_positives
            }
        
        # Calculate MRR (Mean Reciprocal Rank) and first relevant position
        reciprocal_rank = 0
        for i, mem_id in enumerate(retrieved_ids):
            if mem_id in expected_set:
                reciprocal_rank = 1.0 / (i + 1)
                if results['first_relevant_position'] is None:
                    results['first_relevant_position'] = i + 1
                break
        results['mrr'] = reciprocal_rank
        
        # Calculate Average Precision
        precisions = []
        num_relevant_found = 0
        for i, mem_id in enumerate(retrieved_ids):
            if mem_id in expected_set:
                num_relevant_found += 1
                precisions.append(num_relevant_found / (i + 1))
        
        results['average_precision'] = sum(precisions) / len(expected_set) if expected_set else 0
        
        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        import math
        dcg = 0
        for i, mem_id in enumerate(retrieved_ids[:20]):
            if mem_id in expected_set:
                # Use proper log2 for discount factor
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate ideal DCG (all relevant items at top positions)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_set), 20)))
        results['ndcg'] = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        return results
    
    def run_benchmark(self, 
                     test_cases: Optional[List[QueryTestCase]] = None,
                     k_values: List[int] = [5, 10, 20]) -> Dict:
        """Run complete benchmark evaluation."""
        
        if test_cases is None:
            if not self.test_cases:
                raise ValueError("No test cases loaded. Provide test cases or load a test set first.")
            test_cases = self.test_cases
        
        print(f"\nEvaluating {len(test_cases)} test cases...")
        
        all_results = []
        
        # Process test cases
        for i, test_case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(test_cases)}")
            
            result = self.evaluate_retrieval(test_case, k_values)
            all_results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_results, k_values)
        
        # Print results
        self._print_results(aggregated)
        
        return {
            'summary': aggregated,
            'individual_results': all_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_test_cases': len(test_cases),
                'k_values': k_values
            }
        }
    
    def _aggregate_metrics(self, results: List[Dict], k_values: List[int]) -> Dict:
        """Aggregate metrics across all test cases."""
        
        # Overall metrics
        overall = {
            'total_cases': len(results),
            'avg_retrieval_time': sum(r['retrieval_time'] for r in results) / len(results),
            'metrics_at_k': {}
        }
        
        for k in k_values:
            precisions = [r['metrics'][f'@{k}']['precision'] for r in results]
            recalls = [r['metrics'][f'@{k}']['recall'] for r in results]
            f1_scores = [r['metrics'][f'@{k}']['f1'] for r in results]
            
            overall['metrics_at_k'][f'@{k}'] = {
                'precision': sum(precisions) / len(precisions),
                'recall': sum(recalls) / len(recalls),
                'f1': sum(f1_scores) / len(f1_scores)
            }
        
        overall['mrr'] = sum(r['mrr'] for r in results) / len(results)
        overall['map'] = sum(r['average_precision'] for r in results) / len(results)
        overall['ndcg'] = sum(r['ndcg'] for r in results) / len(results)
        
        # Metrics by test type
        by_type = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            test_type = result['test_type']
            by_type[test_type]['results'].append(result)
        
        for test_type, type_data in by_type.items():
            type_results = type_data['results']
            type_data['count'] = len(type_results)
            type_data['metrics_at_k'] = {}
            
            for k in k_values:
                precisions = [r['metrics'][f'@{k}']['precision'] for r in type_results]
                recalls = [r['metrics'][f'@{k}']['recall'] for r in type_results]
                f1_scores = [r['metrics'][f'@{k}']['f1'] for r in type_results]
                
                type_data['metrics_at_k'][f'@{k}'] = {
                    'precision': sum(precisions) / len(precisions) if precisions else 0,
                    'recall': sum(recalls) / len(recalls) if recalls else 0,
                    'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0
                }
            
            type_data['mrr'] = sum(r['mrr'] for r in type_results) / len(type_results)
            type_data['map'] = sum(r['average_precision'] for r in type_results) / len(type_results)
            type_data['ndcg'] = sum(r['ndcg'] for r in type_results) / len(type_results)
            
            # Remove the raw results to keep output clean
            del type_data['results']
        
        # Metrics by difficulty
        by_difficulty = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            difficulty = result['difficulty']
            by_difficulty[difficulty]['results'].append(result)
        
        for difficulty, diff_data in by_difficulty.items():
            diff_results = diff_data['results']
            diff_data['count'] = len(diff_results)
            diff_data['metrics_at_k'] = {}
            
            for k in k_values:
                precisions = [r['metrics'][f'@{k}']['precision'] for r in diff_results]
                recalls = [r['metrics'][f'@{k}']['recall'] for r in diff_results]
                f1_scores = [r['metrics'][f'@{k}']['f1'] for r in diff_results]
                
                diff_data['metrics_at_k'][f'@{k}'] = {
                    'precision': sum(precisions) / len(precisions) if precisions else 0,
                    'recall': sum(recalls) / len(recalls) if recalls else 0,
                    'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0
                }
            
            diff_data['mrr'] = sum(r['mrr'] for r in diff_results) / len(diff_results)
            diff_data['map'] = sum(r['average_precision'] for r in diff_results) / len(diff_results)
            diff_data['ndcg'] = sum(r['ndcg'] for r in diff_results) / len(diff_results)
            
            # Remove raw results
            del diff_data['results']
        
        return {
            'overall': overall,
            'by_type': dict(by_type),
            'by_difficulty': dict(by_difficulty)
        }
    
    def _print_results(self, aggregated: Dict):
        """Print formatted benchmark results."""
        
        print("\n" + "="*60)
        print("RECALL BENCHMARK RESULTS")
        print("="*60)
        
        # Overall metrics
        overall = aggregated['overall']
        print("\nOVERALL METRICS")
        print("-"*60)
        
        for k_label, metrics in overall['metrics_at_k'].items():
            print(f"\n{k_label} Results:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1 Score:  {metrics['f1']:.3f}")
        
        print(f"\nRanking Metrics:")
        print(f"  MRR:  {overall['mrr']:.3f}")
        print(f"  MAP:  {overall['map']:.3f}")
        print(f"  NDCG: {overall['ndcg']:.3f}")
        
        print(f"\nPerformance:")
        print(f"  Avg retrieval time: {overall['avg_retrieval_time']*1000:.2f} ms")
        
        # Metrics by type
        if aggregated['by_type']:
            print("\n" + "-"*60)
            print("METRICS BY TEST TYPE")
            print("-"*60)
            
            for test_type, metrics in sorted(aggregated['by_type'].items()):
                print(f"\n{test_type.upper()} (n={metrics['count']}):")
                
                # Show metrics for first k value as summary
                first_k = list(metrics['metrics_at_k'].keys())[0]
                k_metrics = metrics['metrics_at_k'][first_k]
                print(f"  {first_k}: P={k_metrics['precision']:.3f}, R={k_metrics['recall']:.3f}, F1={k_metrics['f1']:.3f}")
                print(f"  MRR={metrics['mrr']:.3f}, MAP={metrics['map']:.3f}, NDCG={metrics['ndcg']:.3f}")
        
        # Metrics by difficulty
        if aggregated['by_difficulty']:
            print("\n" + "-"*60)
            print("METRICS BY DIFFICULTY")
            print("-"*60)
            
            for difficulty in ['easy', 'medium', 'hard']:
                if difficulty in aggregated['by_difficulty']:
                    metrics = aggregated['by_difficulty'][difficulty]
                    print(f"\n{difficulty.upper()} (n={metrics['count']}):")
                    
                    # Show metrics for first k value
                    first_k = list(metrics['metrics_at_k'].keys())[0]
                    k_metrics = metrics['metrics_at_k'][first_k]
                    print(f"  {first_k}: P={k_metrics['precision']:.3f}, R={k_metrics['recall']:.3f}, F1={k_metrics['f1']:.3f}")
                    print(f"  MRR={metrics['mrr']:.3f}, MAP={metrics['map']:.3f}, NDCG={metrics['ndcg']:.3f}")


def main():
    """Run recall benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run recall benchmark for JAM memory system')
    parser.add_argument('--testset', '-t',
                       help='Path to test set JSON file')
    parser.add_argument('--k', nargs='+', type=int, default=[5, 10, 20],
                       help='k values for precision/recall@k (default: 5 10 20)')
    parser.add_argument('--output', '-o',
                       help='Output file for results')
    parser.add_argument('--latest', action='store_true',
                       help='Use the latest test set file')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = RecallBenchmark()
    
    # Load test set
    if args.testset:
        benchmark.load_testset(args.testset)
    elif args.latest:
        latest = benchmark.find_latest_testset()
        if latest:
            print(f"Using latest test set: {latest}")
            benchmark.load_testset(str(latest))
        else:
            print("No test set files found in benchmarks/test_data/")
            print("Generate one using: python benchmarks/generate_benchmark_testset.py")
            return
    else:
        print("No test set specified. Use --testset <file> or --latest")
        print("Generate a test set using: python benchmarks/generate_benchmark_testset.py")
        return
    
    # Run benchmark
    results = benchmark.run_benchmark(k_values=args.k)
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    else:
        # Save to default location
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"recall_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()