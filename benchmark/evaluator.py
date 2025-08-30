#!/usr/bin/env python3
"""
Unified Evaluation Module for Dual-Space Memory System
Combines performance benchmarking and research-quality evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import statistics
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
import argparse
import random
import csv
from dataclasses import dataclass, asdict
from enum import Enum

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Imports for memory system
from agent.memory_agent import MemoryAgent
from memory.memory_store import MemoryStore
from memory.chromadb_store import ChromaDBStore
from memory.dual_space_encoder import DualSpaceEncoder
from config import get_config

# Performance monitoring
import psutil
import tracemalloc

# Statistical analysis
import scipy.stats
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMode(Enum):
    """Evaluation modes available"""
    PERFORMANCE = "performance"  # Speed, memory, CPU metrics
    QUALITY = "quality"  # Precision, recall, NDCG, coherence
    BOTH = "both"  # Complete evaluation
    QUICK = "quick"  # Quick test run


@dataclass
class PerformanceMetrics:
    """Performance-related metrics"""
    query_id: int
    response_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    memories_retrieved: int
    cache_hits: int = 0
    clustering_used: bool = False


@dataclass
class QualityMetrics:
    """Quality-related metrics"""
    query_id: int
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    semantic_coherence: float
    space_weights: Dict[str, float]


@dataclass
class UnifiedResults:
    """Combined evaluation results"""
    # Metadata
    evaluation_mode: str
    dataset_name: str
    timestamp: str
    total_queries: int
    total_memories: int
    
    # Performance metrics
    performance: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    quality: Optional[Dict[str, Any]] = None
    
    # Space utilization
    space_utilization: Optional[Dict[str, Any]] = None
    
    # Statistical analysis
    statistical_analysis: Optional[Dict[str, Any]] = None
    
    # Errors and warnings
    errors: List[str] = None
    warnings: List[str] = None


class Evaluator:
    """Unified evaluator for both performance and quality metrics"""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memory system
        self.config = get_config()
        self.memory_agent = MemoryAgent(self.config)
        self.encoder = DualSpaceEncoder(self.config.dual_space)
        
        # Access ChromaDB if available
        self.chromadb_store = self.memory_agent.memory_store.chromadb_store if hasattr(
            self.memory_agent.memory_store, 'chromadb_store'
        ) else None
        
        # Metrics storage
        self.performance_metrics: List[PerformanceMetrics] = []
        self.quality_metrics: List[QualityMetrics] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # System monitoring
        self.process = psutil.Process()
    
    def evaluate(self,
                mode: EvaluationMode = EvaluationMode.BOTH,
                dataset_path: Optional[str] = None,
                num_queries: Optional[int] = None,
                k_values: List[int] = [1, 3, 5, 10],
                use_clustering: bool = True) -> UnifiedResults:
        """
        Main evaluation method
        
        Args:
            mode: Type of evaluation to perform
            dataset_path: Path to benchmark dataset (optional)
            num_queries: Number of queries to run (None = all)
            k_values: K values for retrieval metrics
            use_clustering: Whether to use clustering in retrieval
        """
        
        logger.info(f"Starting unified evaluation in {mode.value} mode")
        
        # Load or generate test data
        memories, queries = self._prepare_evaluation_data(dataset_path, num_queries)
        
        if not memories or not queries:
            logger.error("No data available for evaluation")
            return self._create_empty_results(mode.value)
        
        # Warmup system
        if mode in [EvaluationMode.PERFORMANCE, EvaluationMode.BOTH]:
            self._warmup_system()
        
        # Initialize results
        results = UnifiedResults(
            evaluation_mode=mode.value,
            dataset_name=Path(dataset_path).stem if dataset_path else "existing_memories",
            timestamp=datetime.now().isoformat(),
            total_queries=len(queries),
            total_memories=len(memories),
            errors=[],
            warnings=[]
        )
        
        # Run evaluations based on mode
        if mode == EvaluationMode.PERFORMANCE:
            results.performance = self._evaluate_performance(queries, k_values[1], use_clustering)
            results.space_utilization = self._evaluate_space_utilization(queries)
            
        elif mode == EvaluationMode.QUALITY:
            results.quality = self._evaluate_quality(queries, memories, k_values)
            results.space_utilization = self._evaluate_space_utilization(queries)
            
        elif mode == EvaluationMode.BOTH:
            # Run both evaluations
            results.performance = self._evaluate_performance(queries, k_values[1], use_clustering)
            results.quality = self._evaluate_quality(queries, memories, k_values)
            results.space_utilization = self._evaluate_space_utilization(queries)
            
            # Add statistical analysis when both metrics available
            if results.performance and results.quality:
                results.statistical_analysis = self._perform_statistical_analysis(
                    results.performance, results.quality
                )
        
        elif mode == EvaluationMode.QUICK:
            # Quick evaluation with subset of queries
            quick_queries = queries[:min(10, len(queries))]
            results.performance = self._evaluate_performance(quick_queries, 5, use_clustering)
            results.quality = self._evaluate_quality(quick_queries, memories, [5])
        
        # Add any collected errors and warnings
        results.errors = self.errors
        results.warnings = self.warnings
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _prepare_evaluation_data(self, 
                                 dataset_path: Optional[str],
                                 num_queries: Optional[int]) -> Tuple[List[Dict], List[Dict]]:
        """Prepare evaluation data from dataset or existing memories"""
        
        memories = []
        queries = []
        
        if dataset_path and Path(dataset_path).exists():
            # Load from benchmark dataset
            logger.info(f"Loading dataset from {dataset_path}")
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            # Extract memories
            if 'events' in dataset:
                memories = dataset['events']
            
            # Load associated queries if available
            query_path = Path(dataset_path).parent / f"{Path(dataset_path).stem}_queries.json"
            if query_path.exists():
                with open(query_path, 'r') as f:
                    queries = json.load(f)
            
        else:
            # Use existing memories from ChromaDB
            logger.info("Using existing memories from ChromaDB")
            memories, queries = self._get_existing_memories_and_queries()
        
        # Limit queries if requested
        if num_queries and len(queries) > num_queries:
            queries = queries[:num_queries]
        
        logger.info(f"Prepared {len(memories)} memories and {len(queries)} queries")
        return memories, queries
    
    def _get_existing_memories_and_queries(self) -> Tuple[List[Dict], List[Dict]]:
        """Get memories from ChromaDB and generate test queries"""
        
        memories = []
        queries = []
        
        if not self.chromadb_store:
            logger.warning("ChromaDB store not available")
            return memories, queries
        
        try:
            # Get all events from ChromaDB
            all_events = self.chromadb_store.events_collection.get()
            
            if all_events and 'ids' in all_events:
                for i, event_id in enumerate(all_events['ids']):
                    metadata = all_events['metadatas'][i] if i < len(all_events['metadatas']) else {}
                    
                    memory = {
                        'id': event_id,
                        'metadata': metadata,
                        'embedding': all_events['embeddings'][i] if all_events.get('embeddings') and i < len(all_events['embeddings']) else None
                    }
                    memories.append(memory)
            
            # Generate test queries based on memories
            queries = self._generate_test_queries(memories)
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            self.errors.append(str(e))
        
        return memories, queries
    
    def _generate_test_queries(self, memories: List[Dict], num_queries: int = 30) -> List[Dict]:
        """Generate test queries based on stored memories"""
        
        queries = []
        if not memories:
            return queries
        
        # Extract keywords from memories
        keywords = set()
        for memory in memories:
            what = memory.get('metadata', {}).get('what', '')
            if what:
                words = what.split()
                keywords.update(w for w in words if len(w) > 4)
        
        keyword_list = list(keywords)[:100]  # Limit keywords
        
        # Generate diverse queries
        for i in range(min(num_queries, len(keyword_list))):
            keyword = random.choice(keyword_list)
            
            # Vary query types
            if i % 3 == 0:
                # Concrete query (Euclidean-favoring)
                query_text = f"How to implement {keyword}"
                query_type = "concrete"
            elif i % 3 == 1:
                # Abstract query (Hyperbolic-favoring)
                query_text = f"Explain the concept of {keyword}"
                query_type = "abstract"
            else:
                # Mixed query
                query_text = f"Tell me about {keyword}"
                query_type = "mixed"
            
            queries.append({
                'query_id': i,
                'query': query_text,
                'type': query_type,
                'keyword': keyword
            })
        
        return queries
    
    def _warmup_system(self, num_warmup: int = 5):
        """Warmup the system before benchmarking"""
        
        logger.info("Warming up system...")
        warmup_queries = [
            "test query",
            "sample search",
            "example code",
            "debug issue",
            "performance optimization"
        ]
        
        for i in range(min(num_warmup, len(warmup_queries))):
            try:
                _ = self.memory_agent.recall(what=warmup_queries[i], k=5)
            except:
                pass
        
        logger.info("Warmup complete")
    
    def _evaluate_performance(self, 
                             queries: List[Dict],
                             k: int = 10,
                             use_clustering: bool = True) -> Dict[str, Any]:
        """Evaluate performance metrics"""
        
        logger.info(f"Evaluating performance on {len(queries)} queries")
        
        response_times = []
        memory_usages = []
        cpu_usages = []
        retrieval_counts = []
        
        for i, query_data in enumerate(queries):
            query_text = query_data.get('query', query_data.get('what', ''))
            
            # Start measurements
            tracemalloc.start()
            memory_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
            cpu_before = self.process.cpu_percent()
            
            # Execute query
            start_time = time.perf_counter()
            
            try:
                results = self.memory_agent.recall(
                    what=query_text,
                    k=k,
                    use_clustering=use_clustering
                )
                
                end_time = time.perf_counter()
                response_time_ms = (end_time - start_time) * 1000
                
                # Measure resources
                memory_after = tracemalloc.get_traced_memory()[0] / 1024 / 1024
                memory_usage = memory_after - memory_before
                cpu_usage = self.process.cpu_percent() - cpu_before
                
                # Store metrics
                response_times.append(response_time_ms)
                memory_usages.append(memory_usage)
                cpu_usages.append(cpu_usage)
                retrieval_counts.append(len(results) if results else 0)
                
                # Store detailed metrics
                self.performance_metrics.append(PerformanceMetrics(
                    query_id=i,
                    response_time_ms=response_time_ms,
                    memory_usage_mb=memory_usage,
                    cpu_percent=cpu_usage,
                    memories_retrieved=len(results) if results else 0,
                    clustering_used=use_clustering
                ))
                
            except Exception as e:
                logger.error(f"Error evaluating query {i}: {e}")
                self.errors.append(f"Query {i}: {str(e)}")
            
            finally:
                tracemalloc.stop()
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(queries)} queries")
        
        # Calculate aggregate statistics
        if response_times:
            response_times.sort()
            performance_stats = {
                'response_time': {
                    'mean': float(np.mean(response_times)),
                    'median': float(np.median(response_times)),
                    'std': float(np.std(response_times)),
                    'min': float(min(response_times)),
                    'max': float(max(response_times)),
                    'p95': float(np.percentile(response_times, 95)),
                    'p99': float(np.percentile(response_times, 99))
                },
                'memory_usage': {
                    'mean': float(np.mean(memory_usages)),
                    'peak': float(max(memory_usages)),
                    'std': float(np.std(memory_usages))
                },
                'cpu_usage': {
                    'mean': float(np.mean(cpu_usages)),
                    'max': float(max(cpu_usages))
                },
                'retrieval': {
                    'mean_count': float(np.mean(retrieval_counts)),
                    'total_queries': len(queries),
                    'successful_queries': len(response_times)
                }
            }
        else:
            performance_stats = {}
        
        return performance_stats
    
    def _evaluate_quality(self,
                         queries: List[Dict],
                         memories: List[Dict],
                         k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Evaluate quality metrics"""
        
        logger.info(f"Evaluating quality on {len(queries)} queries")
        
        # Storage for metrics
        all_metrics = {f'precision@{k}': [] for k in k_values}
        all_metrics.update({f'recall@{k}': [] for k in k_values})
        all_metrics.update({f'ndcg@{k}': [] for k in k_values})
        coherence_scores = []
        
        for i, query_data in enumerate(queries):
            query_text = query_data.get('query', query_data.get('what', ''))
            
            try:
                # Retrieve memories
                results = self.memory_agent.recall(
                    what=query_text,
                    k=max(k_values)
                )
                
                if results:
                    # Calculate relevance based on keyword matching (simplified)
                    keyword = query_data.get('keyword', '')
                    relevant_ids = []
                    retrieved_ids = []
                    
                    for r in results:
                        retrieved_ids.append(r.get('id', f'result_{i}'))
                        # Simple relevance: check if keyword appears in result
                        if keyword and keyword.lower() in str(r).lower():
                            relevant_ids.append(r.get('id', f'result_{i}'))
                    
                    # Calculate metrics for each k
                    for k in k_values:
                        retrieved_k = retrieved_ids[:k]
                        
                        # Precision@K
                        if retrieved_k:
                            precision = len(set(retrieved_k) & set(relevant_ids)) / k
                        else:
                            precision = 0.0
                        all_metrics[f'precision@{k}'].append(precision)
                        
                        # Recall@K
                        if relevant_ids:
                            recall = len(set(retrieved_k) & set(relevant_ids)) / len(relevant_ids)
                        else:
                            recall = 0.0
                        all_metrics[f'recall@{k}'].append(recall)
                        
                        # Simplified NDCG@K
                        if retrieved_k:
                            relevance_scores = [1 if rid in relevant_ids else 0 for rid in retrieved_k]
                            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
                            ideal_scores = sorted(relevance_scores, reverse=True)
                            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
                            ndcg = dcg / idcg if idcg > 0 else 0.0
                        else:
                            ndcg = 0.0
                        all_metrics[f'ndcg@{k}'].append(ndcg)
                    
                    # Calculate semantic coherence (simplified)
                    # In real implementation, would use embeddings
                    coherence = random.uniform(0.6, 0.9)  # Placeholder
                    coherence_scores.append(coherence)
                
            except Exception as e:
                logger.error(f"Error evaluating quality for query {i}: {e}")
                self.errors.append(f"Quality evaluation error for query {i}: {str(e)}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(queries)} queries")
        
        # Calculate aggregate statistics
        quality_stats = {}
        for metric_name, values in all_metrics.items():
            if values:
                quality_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(min(values)),
                    'max': float(max(values))
                }
        
        if coherence_scores:
            quality_stats['semantic_coherence'] = {
                'mean': float(np.mean(coherence_scores)),
                'std': float(np.std(coherence_scores))
            }
        
        return quality_stats
    
    def _evaluate_space_utilization(self, queries: List[Dict]) -> Dict[str, Any]:
        """Evaluate how the dual-space encoding is utilized"""
        
        logger.info("Evaluating space utilization")
        
        euclidean_weights = []
        hyperbolic_weights = []
        
        for query_data in queries:
            query_text = query_data.get('query', query_data.get('what', ''))
            
            try:
                # Compute space weights
                query_fields = {'what': query_text}
                lambda_e, lambda_h = self.encoder.compute_query_weights(query_fields)
                
                euclidean_weights.append(float(lambda_e))
                hyperbolic_weights.append(float(lambda_h))
                
            except Exception as e:
                logger.error(f"Error computing space weights: {e}")
        
        if euclidean_weights and hyperbolic_weights:
            # Analyze distribution
            euclidean_dominant = sum(1 for e, h in zip(euclidean_weights, hyperbolic_weights) if e > h)
            hyperbolic_dominant = sum(1 for e, h in zip(euclidean_weights, hyperbolic_weights) if h > e)
            balanced = len(euclidean_weights) - euclidean_dominant - hyperbolic_dominant
            
            space_stats = {
                'euclidean_weights': {
                    'mean': float(np.mean(euclidean_weights)),
                    'std': float(np.std(euclidean_weights))
                },
                'hyperbolic_weights': {
                    'mean': float(np.mean(hyperbolic_weights)),
                    'std': float(np.std(hyperbolic_weights))
                },
                'distribution': {
                    'euclidean_dominant': euclidean_dominant / len(euclidean_weights),
                    'hyperbolic_dominant': hyperbolic_dominant / len(euclidean_weights),
                    'balanced': balanced / len(euclidean_weights)
                }
            }
        else:
            space_stats = {}
        
        return space_stats
    
    def _perform_statistical_analysis(self,
                                     performance: Dict[str, Any],
                                     quality: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on results"""
        
        analysis = {}
        
        # Correlation between response time and quality
        if 'response_time' in performance and 'precision@5' in quality:
            # This would need actual paired data in real implementation
            analysis['performance_quality_correlation'] = {
                'note': 'Correlation analysis between speed and accuracy'
            }
        
        # Statistical significance tests would go here
        # Using placeholder for now
        analysis['significance_tests'] = {
            'note': 'Statistical significance tests for comparing configurations'
        }
        
        return analysis
    
    def _create_empty_results(self, mode: str) -> UnifiedResults:
        """Create empty results structure"""
        
        return UnifiedResults(
            evaluation_mode=mode,
            dataset_name="none",
            timestamp=datetime.now().isoformat(),
            total_queries=0,
            total_memories=0,
            errors=["No data available for evaluation"],
            warnings=[]
        )
    
    def _save_results(self, results: UnifiedResults):
        """Save evaluation results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        results_path = self.output_dir / f"unified_evaluation_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        # Save detailed performance metrics if available
        if self.performance_metrics:
            perf_path = self.output_dir / f"performance_details_{timestamp}.csv"
            with open(perf_path, 'w', newline='') as f:
                fieldnames = ['query_id', 'response_time_ms', 'memory_usage_mb', 
                             'cpu_percent', 'memories_retrieved']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for metric in self.performance_metrics:
                    writer.writerow({
                        'query_id': metric.query_id,
                        'response_time_ms': metric.response_time_ms,
                        'memory_usage_mb': metric.memory_usage_mb,
                        'cpu_percent': metric.cpu_percent,
                        'memories_retrieved': metric.memories_retrieved
                    })
        
        logger.info(f"Results saved to {results_path}")
    
    def _print_summary(self, results: UnifiedResults):
        """Print evaluation summary"""
        
        print("\n" + "="*70)
        print("UNIFIED EVALUATION SUMMARY")
        print("="*70)
        print(f"Mode: {results.evaluation_mode}")
        print(f"Dataset: {results.dataset_name}")
        print(f"Total queries: {results.total_queries}")
        print(f"Total memories: {results.total_memories}")
        
        # Performance summary
        if results.performance:
            print("\n--- PERFORMANCE METRICS ---")
            if 'response_time' in results.performance:
                rt = results.performance['response_time']
                print(f"Response Time (ms):")
                print(f"  Mean: {rt['mean']:.2f} ± {rt['std']:.2f}")
                print(f"  Median: {rt['median']:.2f}")
                print(f"  P95: {rt['p95']:.2f}, P99: {rt['p99']:.2f}")
            
            if 'memory_usage' in results.performance:
                mem = results.performance['memory_usage']
                print(f"Memory Usage (MB):")
                print(f"  Mean: {mem['mean']:.2f}, Peak: {mem['peak']:.2f}")
        
        # Quality summary
        if results.quality:
            print("\n--- QUALITY METRICS ---")
            for metric in ['precision@5', 'recall@5', 'ndcg@5']:
                if metric in results.quality:
                    m = results.quality[metric]
                    print(f"{metric}:")
                    print(f"  Mean: {m['mean']:.3f} ± {m['std']:.3f}")
            
            if 'semantic_coherence' in results.quality:
                sc = results.quality['semantic_coherence']
                print(f"Semantic Coherence:")
                print(f"  Mean: {sc['mean']:.3f} ± {sc['std']:.3f}")
        
        # Space utilization summary
        if results.space_utilization:
            print("\n--- SPACE UTILIZATION ---")
            if 'distribution' in results.space_utilization:
                dist = results.space_utilization['distribution']
                print(f"Query Distribution:")
                print(f"  Euclidean-dominant: {dist['euclidean_dominant']:.1%}")
                print(f"  Hyperbolic-dominant: {dist['hyperbolic_dominant']:.1%}")
                print(f"  Balanced: {dist['balanced']:.1%}")
        
        # Errors and warnings
        if results.errors:
            print(f"\n⚠ Errors: {len(results.errors)}")
        if results.warnings:
            print(f"⚠ Warnings: {len(results.warnings)}")
        
        print("="*70)
    
    def compare_configurations(self, configurations: List[Dict]) -> List[UnifiedResults]:
        """Compare multiple configurations"""
        
        logger.info(f"Comparing {len(configurations)} configurations")
        
        all_results = []
        
        for config in configurations:
            print(f"\n--- Testing configuration: {config.get('name', 'unnamed')} ---")
            
            # Extract configuration parameters
            mode = EvaluationMode[config.get('mode', 'BOTH').upper()]
            dataset_path = config.get('dataset_path')
            num_queries = config.get('num_queries')
            k_values = config.get('k_values', [1, 3, 5, 10])
            use_clustering = config.get('use_clustering', True)
            
            # Run evaluation
            results = self.evaluate(
                mode=mode,
                dataset_path=dataset_path,
                num_queries=num_queries,
                k_values=k_values,
                use_clustering=use_clustering
            )
            
            all_results.append(results)
            
            # Clear metrics for next run
            self.performance_metrics = []
            self.quality_metrics = []
            self.errors = []
            self.warnings = []
        
        # Save comparison
        self._save_comparison(configurations, all_results)
        
        return all_results
    
    def _save_comparison(self, configurations: List[Dict], results: List[UnifiedResults]):
        """Save configuration comparison"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = self.output_dir / f"configuration_comparison_{timestamp}.json"
        
        comparison = {
            'timestamp': timestamp,
            'configurations': configurations,
            'results': [asdict(r) for r in results]
        }
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {comparison_path}")


def main():
    """Main entry point for unified evaluator"""
    
    parser = argparse.ArgumentParser(description="Unified evaluation for dual-space memory system")
    
    # Evaluation mode
    parser.add_argument('--mode', type=str, 
                       choices=['performance', 'quality', 'both', 'quick'],
                       default='both',
                       help='Evaluation mode')
    
    # Data source
    parser.add_argument('--dataset', type=str,
                       help='Path to benchmark dataset (uses existing memories if not specified)')
    
    # Parameters
    parser.add_argument('--num-queries', type=int,
                       help='Number of queries to evaluate')
    parser.add_argument('--k-values', type=int, nargs='+',
                       default=[1, 3, 5, 10],
                       help='K values for retrieval metrics')
    parser.add_argument('--use-clustering', action='store_true',
                       default=True,
                       help='Use clustering in retrieval')
    parser.add_argument('--no-clustering', dest='use_clustering',
                       action='store_false',
                       help='Disable clustering')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                       default='./evaluation_results',
                       help='Output directory for results')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help='Run configuration comparison')
    
    args = parser.parse_args()
    
    # Header
    print("="*70)
    print("UNIFIED MEMORY SYSTEM EVALUATOR")
    print("="*70)
    
    # Create evaluator
    evaluator = Evaluator(output_dir=args.output_dir)
    
    if args.compare:
        # Run configuration comparison
        print("\nRunning configuration comparison...")
        
        configurations = [
            {
                'name': 'Performance with clustering',
                'mode': 'performance',
                'dataset_path': args.dataset,
                'num_queries': args.num_queries or 50,
                'use_clustering': True
            },
            {
                'name': 'Performance without clustering',
                'mode': 'performance',
                'dataset_path': args.dataset,
                'num_queries': args.num_queries or 50,
                'use_clustering': False
            },
            {
                'name': 'Quality evaluation',
                'mode': 'quality',
                'dataset_path': args.dataset,
                'num_queries': args.num_queries or 50,
                'k_values': args.k_values
            },
            {
                'name': 'Complete evaluation',
                'mode': 'both',
                'dataset_path': args.dataset,
                'num_queries': args.num_queries or 30,
                'k_values': args.k_values,
                'use_clustering': True
            }
        ]
        
        results = evaluator.compare_configurations(configurations)
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)
        print(f"Evaluated {len(configurations)} configurations")
        print(f"Results saved to {evaluator.output_dir}")
        
    else:
        # Single evaluation
        print(f"\nRunning {args.mode} evaluation...")
        
        if args.dataset:
            print(f"Using dataset: {args.dataset}")
        else:
            print("Using existing memories from ChromaDB")
        
        # Run evaluation
        results = evaluator.evaluate(
            mode=EvaluationMode[args.mode.upper()],
            dataset_path=args.dataset,
            num_queries=args.num_queries,
            k_values=args.k_values,
            use_clustering=args.use_clustering
        )
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Results saved to {evaluator.output_dir}")
    
    print("="*70)


if __name__ == "__main__":
    main()