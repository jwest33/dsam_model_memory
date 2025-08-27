"""
Benchmark runner for memory system performance testing
Executes comprehensive performance tests using generated datasets
"""

import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import sys
import warnings
from dataclasses import dataclass, asdict
import csv

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore')

from agent.memory_agent import MemoryAgent
from memory.memory_store import MemoryStore
from memory.dual_space_encoder import DualSpaceEncoder
from config import get_config
import psutil
import tracemalloc

@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark execution"""
    query_id: int
    query: str
    query_type: str
    response_time_ms: float
    memories_retrieved: int
    space_weights: Dict[str, float]
    memory_usage_mb: float
    cpu_percent: float
    cache_hits: int
    residual_updates: int
    clustering_used: bool
    
@dataclass
class BenchmarkResults:
    """Overall benchmark results"""
    dataset_name: str
    total_queries: int
    total_memories: int
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    avg_memories_retrieved: float
    avg_memory_usage_mb: float
    peak_memory_usage_mb: float
    avg_cpu_percent: float
    total_cache_hits: int
    cache_hit_rate: float
    space_usage_distribution: Dict[str, float]
    errors: List[str]
    timestamp: str

class BenchmarkRunner:
    def __init__(self, dataset_path: str, output_dir: str = "./benchmark_results"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load dataset
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        # Load queries if available
        query_path = self.dataset_path.parent / f"{self.dataset['metadata']['name']}_queries.json"
        if query_path.exists():
            with open(query_path, 'r') as f:
                self.queries = json.load(f)
        else:
            self.queries = []
        
        # Initialize memory system
        self.config = get_config()
        self.memory_agent = MemoryAgent(self.config)
        self.encoder = DualSpaceEncoder(self.config.dual_space)
        
        # Metrics storage
        self.metrics: List[BenchmarkMetrics] = []
        self.errors: List[str] = []
        
        # System monitoring
        self.process = psutil.Process()
        
    def warmup(self, num_queries: int = 10):
        """Warmup the system with initial queries"""
        print("Warming up system...")
        warmup_queries = [
            "How to implement error handling?",
            "What are design patterns?",
            "Show me Python code examples",
            "Explain software architecture",
            "Database optimization techniques"
        ]
        
        for i in range(min(num_queries, len(warmup_queries))):
            query = warmup_queries[i % len(warmup_queries)]
            try:
                _ = self.memory_agent.recall(what=query, k=5)
            except Exception:
                pass
        
        print("  Warmup complete")
    
    def measure_query_performance(
        self,
        query: str,
        query_type: str = "mixed",
        use_clustering: bool = True,
        k: int = 10
    ) -> BenchmarkMetrics:
        """Measure performance for a single query"""
        # Start memory tracking
        tracemalloc.start()
        memory_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        
        # CPU usage before
        cpu_before = self.process.cpu_percent()
        
        # Execute query with timing
        start_time = time.perf_counter()
        
        try:
            # Compute space weights
            query_fields = {'what': query}
            lambda_e, lambda_h = self.encoder.compute_query_weights(query_fields)
            
            # Retrieve memories
            results = self.memory_agent.recall(
                what=query,
                k=k,
                use_clustering=use_clustering
            )
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            # Memory usage after
            memory_after = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # CPU usage
            cpu_percent = self.process.cpu_percent() - cpu_before
            
            # Cache statistics (if available)
            cache_hits = 0  # Would need to track this in the encoder
            residual_updates = 0  # Would need to track this in memory store
            
            metrics = BenchmarkMetrics(
                query_id=len(self.metrics),
                query=query,
                query_type=query_type,
                response_time_ms=response_time_ms,
                memories_retrieved=len(results) if results else 0,
                space_weights={'euclidean': float(lambda_e), 'hyperbolic': float(lambda_h)},
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_percent,
                cache_hits=cache_hits,
                residual_updates=residual_updates,
                clustering_used=use_clustering
            )
            
        except Exception as e:
            self.errors.append(f"Query failed: {query[:50]}... - {str(e)}")
            metrics = BenchmarkMetrics(
                query_id=len(self.metrics),
                query=query,
                query_type=query_type,
                response_time_ms=0,
                memories_retrieved=0,
                space_weights={'euclidean': 0, 'hyperbolic': 0},
                memory_usage_mb=0,
                cpu_percent=0,
                cache_hits=0,
                residual_updates=0,
                clustering_used=use_clustering
            )
        
        finally:
            tracemalloc.stop()
        
        return metrics
    
    def run_benchmark(
        self,
        num_queries: Optional[int] = None,
        use_clustering: bool = True,
        k: int = 10
    ) -> BenchmarkResults:
        """Run the complete benchmark"""
        print("\n" + "="*60)
        print("RUNNING BENCHMARK")
        print("="*60)
        print(f"Dataset: {self.dataset['metadata']['name']}")
        print(f"Total memories: {self.dataset['metadata']['num_events']}")
        print(f"Queries to run: {num_queries or len(self.queries)}")
        print(f"Clustering: {'enabled' if use_clustering else 'disabled'}")
        print(f"K value: {k}")
        
        # Warmup
        self.warmup()
        
        # Select queries
        if self.queries:
            queries_to_run = self.queries[:num_queries] if num_queries else self.queries
        else:
            # Generate simple queries if none provided
            print("\nNo query set found, generating simple queries...")
            queries_to_run = [
                {'query': f"Tell me about {word}", 'type': 'mixed'}
                for word in ['Python', 'database', 'API', 'testing', 'deployment']
            ][:num_queries or 5]
        
        # Run benchmark queries
        print("\nRunning benchmark queries...")
        for i, query_data in enumerate(queries_to_run):
            if isinstance(query_data, dict):
                query = query_data.get('query', query_data.get('what', ''))
                query_type = query_data.get('type', 'mixed')
            else:
                query = str(query_data)
                query_type = 'mixed'
            
            # Measure performance
            metrics = self.measure_query_performance(
                query=query,
                query_type=query_type,
                use_clustering=use_clustering,
                k=k
            )
            self.metrics.append(metrics)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                avg_time = statistics.mean([m.response_time_ms for m in self.metrics if m.response_time_ms > 0])
                print(f"  Processed {i + 1}/{len(queries_to_run)} queries (avg: {avg_time:.2f}ms)")
        
        # Calculate aggregate results
        valid_metrics = [m for m in self.metrics if m.response_time_ms > 0]
        
        if valid_metrics:
            response_times = [m.response_time_ms for m in valid_metrics]
            response_times.sort()
            
            results = BenchmarkResults(
                dataset_name=self.dataset['metadata']['name'],
                total_queries=len(self.metrics),
                total_memories=self.dataset['metadata']['num_events'],
                avg_response_time_ms=statistics.mean(response_times),
                median_response_time_ms=statistics.median(response_times),
                p95_response_time_ms=response_times[int(len(response_times) * 0.95)] if len(response_times) > 20 else max(response_times),
                p99_response_time_ms=response_times[int(len(response_times) * 0.99)] if len(response_times) > 100 else max(response_times),
                min_response_time_ms=min(response_times),
                max_response_time_ms=max(response_times),
                avg_memories_retrieved=statistics.mean([m.memories_retrieved for m in valid_metrics]),
                avg_memory_usage_mb=statistics.mean([m.memory_usage_mb for m in valid_metrics]),
                peak_memory_usage_mb=max([m.memory_usage_mb for m in valid_metrics]),
                avg_cpu_percent=statistics.mean([m.cpu_percent for m in valid_metrics]),
                total_cache_hits=sum([m.cache_hits for m in valid_metrics]),
                cache_hit_rate=sum([m.cache_hits for m in valid_metrics]) / len(valid_metrics) if valid_metrics else 0,
                space_usage_distribution=self._calculate_space_distribution(valid_metrics),
                errors=self.errors,
                timestamp=datetime.now().isoformat()
            )
        else:
            # No valid metrics
            results = BenchmarkResults(
                dataset_name=self.dataset['metadata']['name'],
                total_queries=len(self.metrics),
                total_memories=self.dataset['metadata']['num_events'],
                avg_response_time_ms=0,
                median_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                avg_memories_retrieved=0,
                avg_memory_usage_mb=0,
                peak_memory_usage_mb=0,
                avg_cpu_percent=0,
                total_cache_hits=0,
                cache_hit_rate=0,
                space_usage_distribution={'euclidean': 0, 'hyperbolic': 0, 'balanced': 0},
                errors=self.errors,
                timestamp=datetime.now().isoformat()
            )
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _calculate_space_distribution(self, metrics: List[BenchmarkMetrics]) -> Dict[str, float]:
        """Calculate distribution of space usage"""
        euclidean_dominant = 0
        hyperbolic_dominant = 0
        balanced = 0
        
        for m in metrics:
            e_weight = m.space_weights.get('euclidean', 0)
            h_weight = m.space_weights.get('hyperbolic', 0)
            
            if abs(e_weight - h_weight) < 0.1:
                balanced += 1
            elif e_weight > h_weight:
                euclidean_dominant += 1
            else:
                hyperbolic_dominant += 1
        
        total = len(metrics)
        return {
            'euclidean': euclidean_dominant / total if total > 0 else 0,
            'hyperbolic': hyperbolic_dominant / total if total > 0 else 0,
            'balanced': balanced / total if total > 0 else 0
        }
    
    def _save_results(self, results: BenchmarkResults):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary JSON
        summary_path = self.output_dir / f"benchmark_{timestamp}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        # Save detailed metrics CSV
        metrics_path = self.output_dir / f"benchmark_{timestamp}_metrics.csv"
        with open(metrics_path, 'w', newline='') as f:
            if self.metrics:
                fieldnames = list(asdict(self.metrics[0]).keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for metric in self.metrics:
                    writer.writerow(asdict(metric))
        
        print(f"\nResults saved:")
        print(f"  Summary: {summary_path}")
        print(f"  Details: {metrics_path}")
    
    def _print_summary(self, results: BenchmarkResults):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Dataset: {results.dataset_name}")
        print(f"Total memories: {results.total_memories}")
        print(f"Total queries: {results.total_queries}")
        
        print(f"\nResponse Time (ms):")
        print(f"  Average: {results.avg_response_time_ms:.2f}")
        print(f"  Median: {results.median_response_time_ms:.2f}")
        print(f"  P95: {results.p95_response_time_ms:.2f}")
        print(f"  P99: {results.p99_response_time_ms:.2f}")
        print(f"  Min: {results.min_response_time_ms:.2f}")
        print(f"  Max: {results.max_response_time_ms:.2f}")
        
        print(f"\nRetrieval Performance:")
        print(f"  Avg memories retrieved: {results.avg_memories_retrieved:.1f}")
        print(f"  Cache hit rate: {results.cache_hit_rate:.1%}")
        
        print(f"\nResource Usage:")
        print(f"  Avg memory: {results.avg_memory_usage_mb:.2f} MB")
        print(f"  Peak memory: {results.peak_memory_usage_mb:.2f} MB")
        print(f"  Avg CPU: {results.avg_cpu_percent:.1f}%")
        
        print(f"\nSpace Usage Distribution:")
        print(f"  Euclidean-dominant: {results.space_usage_distribution['euclidean']:.1%}")
        print(f"  Hyperbolic-dominant: {results.space_usage_distribution['hyperbolic']:.1%}")
        print(f"  Balanced: {results.space_usage_distribution['balanced']:.1%}")
        
        if results.errors:
            print(f"\nErrors encountered: {len(results.errors)}")
    
    def compare_configurations(self, configurations: List[Dict]) -> Dict:
        """Run benchmarks with different configurations and compare"""
        comparison_results = []
        
        for config in configurations:
            print(f"\n{'='*60}")
            print(f"Testing configuration: {config['name']}")
            print(f"{'='*60}")
            
            # Apply configuration
            use_clustering = config.get('use_clustering', True)
            k = config.get('k', 10)
            
            # Run benchmark
            results = self.run_benchmark(
                use_clustering=use_clustering,
                k=k
            )
            
            comparison_results.append({
                'config': config,
                'results': asdict(results)
            })
            
            # Clear metrics for next run
            self.metrics = []
            self.errors = []
        
        # Save comparison
        comparison_path = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nComparison saved to: {comparison_path}")
        
        return comparison_results

def main():
    """Main function for running benchmarks"""
    print("="*60)
    print("MEMORY SYSTEM BENCHMARK RUNNER")
    print("="*60)
    
    # Find available datasets
    dataset_dir = Path("./benchmark_datasets")
    if not dataset_dir.exists():
        print("\nNo benchmark datasets found!")
        print("Please run 'python generate_benchmark_dataset.py' first")
        return
    
    datasets = list(dataset_dir.glob("benchmark_*.json"))
    if not datasets:
        print("\nNo benchmark datasets found!")
        print("Please run 'python generate_benchmark_dataset.py' first")
        return
    
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets):
        # Load metadata
        with open(dataset, 'r') as f:
            meta = json.load(f)['metadata']
        print(f"{i+1}. {dataset.name}")
        print(f"   Events: {meta['num_events']}, Time span: {meta['time_span_days']} days")
    
    choice = input(f"\nSelect dataset (1-{len(datasets)}) [default: 1]: ").strip() or "1"
    
    try:
        dataset_idx = int(choice) - 1
        selected_dataset = datasets[dataset_idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    # Initialize runner
    runner = BenchmarkRunner(str(selected_dataset))
    
    print("\nBenchmark options:")
    print("1. Quick benchmark (50 queries)")
    print("2. Standard benchmark (100 queries)")
    print("3. Comprehensive benchmark (all queries)")
    print("4. Configuration comparison")
    
    option = input("\nSelect option (1-4) [default: 2]: ").strip() or "2"
    
    if option == "1":
        runner.run_benchmark(num_queries=50)
    elif option == "2":
        runner.run_benchmark(num_queries=100)
    elif option == "3":
        runner.run_benchmark()  # All queries
    elif option == "4":
        # Compare different configurations
        configurations = [
            {'name': 'With Clustering', 'use_clustering': True, 'k': 10},
            {'name': 'Without Clustering', 'use_clustering': False, 'k': 10},
            {'name': 'Small K', 'use_clustering': True, 'k': 5},
            {'name': 'Large K', 'use_clustering': True, 'k': 20}
        ]
        runner.compare_configurations(configurations)
    else:
        print("Invalid option")
        return
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)

if __name__ == "__main__":
    main()
