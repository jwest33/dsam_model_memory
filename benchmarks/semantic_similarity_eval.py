#!/usr/bin/env python3
"""
Evaluate semantic similarity using pre-generated test sets.
This ensures reproducible benchmarking with the same test cases.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever
from agentic_memory.types import RetrievalQuery
from agentic_memory.embedding import get_llama_embedder
from benchmarks.generate_semantic_testset import SemanticTestCase


@dataclass
class EvaluationMetrics:
    """Metrics for a single test case evaluation."""
    test_id: str
    query_type: str
    found_original: bool
    rank_of_original: int  # -1 if not found
    precision_at_k: float
    recall_at_k: float
    mrr: float
    retrieval_time_ms: float
    num_retrieved: int
    num_expected: int


class SemanticSimilarityEvaluator:
    """Evaluate retrieval performance on semantic test sets."""
    
    def __init__(self):
        """Initialize evaluator with retrieval components."""
        self.store = MemoryStore(cfg.db_path)
        embed_dim = int(cfg.get('embed_dim', 1024))
        self.index = FaissIndex(dim=embed_dim, index_path=cfg.index_path)
        self.router = MemoryRouter(self.store, self.index)
        self.retriever = HybridRetriever(self.store, self.index)
        
        # Initialize embedder
        try:
            self.embedder = get_llama_embedder()
            test_embedding = self.embedder.encode(["test"], normalize_embeddings=True)
            self.embedder_available = True
            print("✓ Embedding server connected")
        except Exception as e:
            print(f"✗ Embedding server not available: {e}")
            self.embedder_available = False
            
        # Test data directory
        self.test_data_dir =  "test_data"
        
    def load_test_set(self, filename: str = None) -> List[SemanticTestCase]:
        """Load test set from file."""
        
        if filename is None:
            # Find most recent test set
            test_files = list(self.test_data_dir.glob("semantic_testset_*.json"))
            if not test_files:
                raise FileNotFoundError("No test sets found. Run: python benchmarks/semantic_similarity_eval.py --generate")
            filename = max(test_files).name
            print(f"Using most recent test set: {filename}")
        
        filepath = self.test_data_dir / filename
        
        if not filepath.exists():
            # Check if it's just a filename without directory
            if (self.test_data_dir / filename).exists():
                filepath = self.test_data_dir / filename
            else:
                raise FileNotFoundError(f"Test set file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = []
        for tc_dict in data['test_cases']:
            test_cases.append(SemanticTestCase(**tc_dict))
        
        print(f"Loaded {len(test_cases)} test cases from {filename}")
        print(f"Generated at: {data['metadata']['generated_at']}")
        print(f"Query types: {', '.join(data['metadata']['query_types'])}")
        
        return test_cases
    
    def evaluate_single_case(self, test_case: SemanticTestCase, k: int = 10) -> EvaluationMetrics:
        """Evaluate retrieval for a single test case."""
        
        if not self.embedder_available:
            raise RuntimeError("Embedder not available - cannot run evaluation")
        
        # Generate embedding for the query
        start_time = time.time()
        
        try:
            qvec = self.embedder.encode([test_case.paraphrased_query], normalize_embeddings=True)[0]
        except Exception as e:
            print(f"  Error encoding query for {test_case.test_id}: {e}")
            return None
        
        # Create retrieval query
        rq = RetrievalQuery(
            session_id=test_case.metadata.get('session_id', 'test'),
            text=test_case.paraphrased_query
        )
        
        # Perform retrieval
        try:
            candidates = self.retriever.search(rq, qvec, topk_sem=k, topk_lex=k)
        except Exception as e:
            print(f"  Error retrieving for {test_case.test_id}: {e}")
            return None
        
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Extract retrieved memory IDs
        retrieved_ids = [c.memory_id for c in candidates]
        
        # Calculate metrics
        found_original = test_case.original_memory_id in retrieved_ids
        rank_of_original = retrieved_ids.index(test_case.original_memory_id) + 1 if found_original else -1
        
        # Precision and recall against expected relevant set
        expected_set = set(test_case.expected_relevant)
        retrieved_set = set(retrieved_ids)
        
        true_positives = len(expected_set & retrieved_set)
        
        precision_at_k = true_positives / len(retrieved_set) if retrieved_set else 0
        recall_at_k = true_positives / len(expected_set) if expected_set else 0
        
        # MRR (Mean Reciprocal Rank) for the original memory
        mrr = 1 / rank_of_original if rank_of_original > 0 else 0
        
        return EvaluationMetrics(
            test_id=test_case.test_id,
            query_type=test_case.query_type,
            found_original=found_original,
            rank_of_original=rank_of_original,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            retrieval_time_ms=retrieval_time,
            num_retrieved=len(retrieved_ids),
            num_expected=len(expected_set)
        )
    
    def evaluate_test_set(self, test_cases: List[SemanticTestCase], k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate entire test set at different k values."""
        
        print(f"\nEvaluating {len(test_cases)} test cases at k={k_values}...")
        
        results = defaultdict(list)
        
        for i, test_case in enumerate(test_cases):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_cases)}")
            
            # Evaluate at different k values
            for k in k_values:
                metrics = self.evaluate_single_case(test_case, k=k)
                if metrics:
                    results[k].append(metrics)
        
        # Calculate aggregate statistics
        summary = self.calculate_summary(results)
        
        return summary
    
    def calculate_summary(self, results: Dict[int, List[EvaluationMetrics]]) -> Dict:
        """Calculate summary statistics from evaluation results."""
        
        summary = {
            "overall": {},
            "by_query_type": {},
            "detailed_metrics": {}
        }
        
        for k, metrics_list in results.items():
            if not metrics_list:
                continue
            
            # Overall metrics
            summary["overall"][f"@{k}"] = {
                "recall": np.mean([m.found_original for m in metrics_list]),
                "mrr": np.mean([m.mrr for m in metrics_list]),
                "precision": np.mean([m.precision_at_k for m in metrics_list]),
                "recall_at_k": np.mean([m.recall_at_k for m in metrics_list]),
                "avg_retrieval_time_ms": np.mean([m.retrieval_time_ms for m in metrics_list]),
                "total_cases": len(metrics_list)
            }
            
            # Metrics by query type
            by_type = defaultdict(list)
            for m in metrics_list:
                by_type[m.query_type].append(m)
            
            summary["by_query_type"][f"@{k}"] = {}
            for query_type, type_metrics in by_type.items():
                summary["by_query_type"][f"@{k}"][query_type] = {
                    "recall": np.mean([m.found_original for m in type_metrics]),
                    "mrr": np.mean([m.mrr for m in type_metrics]),
                    "precision": np.mean([m.precision_at_k for m in type_metrics]),
                    "avg_rank": np.mean([m.rank_of_original for m in type_metrics if m.rank_of_original > 0]) if any(m.rank_of_original > 0 for m in type_metrics) else -1,
                    "count": len(type_metrics)
                }
            
            # Store detailed metrics for analysis
            summary["detailed_metrics"][f"@{k}"] = [
                {
                    "test_id": m.test_id,
                    "found": m.found_original,
                    "rank": m.rank_of_original,
                    "type": m.query_type
                }
                for m in metrics_list
            ]
        
        return summary
    
    def print_results(self, summary: Dict):
        """Print formatted evaluation results."""
        
        print("\n" + "="*60)
        print("SEMANTIC SIMILARITY EVALUATION RESULTS")
        print("="*60)
        
        # Overall results
        print("\nOVERALL PERFORMANCE:")
        print("-"*40)
        for k_label, metrics in summary["overall"].items():
            print(f"\n{k_label}:")
            print(f"  Recall (found original): {metrics['recall']:.2%}")
            print(f"  MRR: {metrics['mrr']:.3f}")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall@k: {metrics['recall_at_k']:.2%}")
            print(f"  Avg retrieval time: {metrics['avg_retrieval_time_ms']:.1f}ms")
        
        # Results by query type
        print("\n" + "-"*40)
        print("PERFORMANCE BY QUERY TYPE:")
        print("-"*40)
        
        for k_label, type_results in summary["by_query_type"].items():
            print(f"\n{k_label}:")
            for query_type, metrics in type_results.items():
                print(f"  {query_type.upper()}:")
                print(f"    Recall: {metrics['recall']:.2%} | MRR: {metrics['mrr']:.3f} | Count: {metrics['count']}")
                if metrics['avg_rank'] > 0:
                    print(f"    Avg rank when found: {metrics['avg_rank']:.1f}")
        
        # Interpretation
        print("\n" + "="*60)
        print("INTERPRETATION:")
        print("="*60)
        
        # Get @10 metrics for interpretation
        if "@10" in summary["overall"]:
            recall_10 = summary["overall"]["@10"]["recall"]
            mrr_10 = summary["overall"]["@10"]["mrr"]
            
            if recall_10 < 0.3:
                print("⚠️  LOW SEMANTIC RECALL - System is not capturing semantic similarity well")
                print("   Possible causes:")
                print("   - Embeddings not capturing semantic meaning")
                print("   - Retriever over-relying on lexical matching")
                print("   - Embedding model needs fine-tuning")
            elif recall_10 < 0.6:
                print("⚡ MODERATE SEMANTIC RECALL - Room for improvement")
                print("   Consider:")
                print("   - Adjusting retrieval weights to favor semantic search")
                print("   - Testing different embedding models")
            else:
                print("✅ GOOD SEMANTIC RECALL - System captures semantic similarity well")
            
            if mrr_10 < 0.2:
                print("\n⚠️  LOW MRR - Relevant results not ranked highly")
            elif mrr_10 > 0.5:
                print("\n✅ GOOD MRR - Relevant results appear near the top")
    
    def save_results(self, summary: Dict, filename: str = None):
        """Save evaluation results to file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_eval_results_{timestamp}.json"
        
        results_dir =  "results"
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath


def main():
    """Run semantic similarity evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate semantic similarity retrieval')
    parser.add_argument('--testset', '-t', 
                       help='Test set filename (default: most recent)')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20],
                       help='k values for evaluation (default: 5 10 20)')
    parser.add_argument('--output', '-o',
                       help='Output filename for results (default: auto-generated)')
    parser.add_argument('--generate', '-g', action='store_true',
                       help='Generate new test set before evaluation')
    parser.add_argument('--cases', type=int, default=50,
                       help='Number of test cases to generate (with --generate)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SemanticSimilarityEvaluator()
    
    if not evaluator.embedder_available:
        print("\n❌ Cannot run evaluation without embedding server")
        print("Please start the embedding server first:")
        print("  python -m agentic_memory.cli server start --all")
        return
    
    # Generate new test set if requested
    if args.generate:
        print("\n" + "="*60)
        print("GENERATING NEW TEST SET")
        print("="*60)
        from benchmarks.generate_semantic_testset import SemanticTestGenerator
        generator = SemanticTestGenerator()
        test_cases = generator.generate_test_cases(num_cases=args.cases)
        if test_cases:
            generator.save_test_set(test_cases)
            print("\n✓ New test set generated successfully")
        else:
            print("\n❌ Failed to generate test cases")
            return
    
    # Load test set
    try:
        test_cases = evaluator.load_test_set(args.testset)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nOptions:")
        print("1. Generate a new test set: python benchmarks/semantic_similarity_eval.py --generate")
        print("2. Specify a test set file: python benchmarks/semantic_similarity_eval.py --testset <filename>")
        return
    
    # Run evaluation
    summary = evaluator.evaluate_test_set(test_cases, k_values=args.k)
    
    # Print results
    evaluator.print_results(summary)
    
    # Always save results for analysis
    results_file = evaluator.save_results(summary, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Test cases evaluated: {len(test_cases)}")
    print(f"Results saved to: {results_file}")
    
    # Quick summary of key metrics
    if "@10" in summary["overall"]:
        recall = summary["overall"]["@10"]["recall"]
        mrr = summary["overall"]["@10"]["mrr"]
        print(f"\nKey Metrics @10:")
        print(f"  Recall: {recall:.2%}")
        print(f"  MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()
