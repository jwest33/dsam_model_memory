#!/usr/bin/env python3
"""
Wrapper script for running JAM Memory System benchmarks.
Provides easy access to benchmark functionality with sensible defaults.
"""
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.core import MemoryBenchmark
from benchmarks.analyze_lmsys import LMSYSAnalyzer
from benchmarks.recall_benchmark import RecallBenchmark


def run_synthetic_benchmark(args):
    """Run synthetic benchmark using generated interactions."""
    print("\n" + "="*60)
    print("JAM MEMORY SYSTEM - SYNTHETIC BENCHMARK")
    print("="*60)
    
    # Select preset or use custom values
    presets = {
        'quick': {
            'interaction_count': 50,
            'time_range_days': 7,
            'retrieval_queries': 10,
            'description': 'Quick test (50 interactions)'
        },
        'standard': {
            'interaction_count': 200,
            'time_range_days': 30,
            'retrieval_queries': 20,
            'description': 'Standard benchmark (200 interactions)'
        },
        'comprehensive': {
            'interaction_count': 500,
            'time_range_days': 60,
            'retrieval_queries': 50,
            'description': 'Comprehensive test (500 interactions)'
        },
        'stress': {
            'interaction_count': 1000,
            'time_range_days': 90,
            'retrieval_queries': 100,
            'description': 'Stress test (1000 interactions)'
        }
    }
    
    if args.preset and args.preset in presets:
        preset = presets[args.preset]
        interaction_count = preset['interaction_count']
        time_range_days = preset['time_range_days']
        retrieval_queries = preset['retrieval_queries']
        print(f"Using preset: {args.preset} - {preset['description']}")
    else:
        interaction_count = args.interactions or 100
        time_range_days = args.days or 30
        retrieval_queries = args.queries or 20
        print(f"Custom configuration: {interaction_count} interactions over {time_range_days} days")
    
    # Scenario mix
    scenario_mix = {
        'simple': args.simple or 0.3,
        'medium': args.medium or 0.5,
        'complex': args.complex or 0.2
    }
    
    # Create benchmark
    benchmark = MemoryBenchmark()
    
    # Run benchmark
    async def run():
        result = await benchmark.run_benchmark(
            interaction_count=interaction_count,
            time_range_days=time_range_days,
            retrieval_queries=retrieval_queries,
            scenario_mix=scenario_mix,
            skip_llm_extraction=args.skip_llm
        )
        benchmark.print_results(result)
        if args.output:
            benchmark.save_results(result, args.output)
        return result
    
    return asyncio.run(run())


def run_recall_benchmark(args):
    """Run recall benchmark for retrieval quality."""
    print("\n" + "="*60)
    print("JAM MEMORY SYSTEM - RECALL BENCHMARK")
    print("="*60)
    
    try:
        benchmark = RecallBenchmark()
        
        # Run benchmark with specified parameters
        results = benchmark.run_benchmark(
            num_cases_per_type=args.cases or 5,
            k_values=args.k or [5, 10, 20]
        )
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {args.output}")
        
        return results
    except Exception as e:
        print(f"\n\nERROR: Recall benchmark failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_existing_data(args):
    """Analyze existing memory data in the database."""
    print("\n" + "="*60)
    print("JAM MEMORY SYSTEM - DATA ANALYSIS")
    print("="*60)
    
    try:
        analyzer = LMSYSAnalyzer()
        
        # Generate report
        result = analyzer.generate_report(output_file=args.output)
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating visualization plots...")
            analyzer.plot_analysis()
        
        return result
    except Exception as e:
        print(f"\n\nERROR: Analysis failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return None


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description='JAM Memory System Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with preset
  python run_benchmark.py --preset standard
  
  # Run with custom settings
  python run_benchmark.py --interactions 300 --days 45 --queries 30
  
  # Analyze existing data
  python run_benchmark.py --analyze --plot
  
  # Run recall benchmark
  python run_benchmark.py --recall --cases 10 --k 5 10 20
        """
    )
    
    # Mode selection
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing data instead of running benchmark')
    parser.add_argument('--recall', action='store_true',
                       help='Run recall benchmark for retrieval quality')
    
    # Benchmark parameters
    parser.add_argument('--preset', choices=['quick', 'standard', 'comprehensive', 'stress'],
                       help='Use a preset configuration')
    parser.add_argument('--interactions', type=int,
                       help='Number of interactions to generate')
    parser.add_argument('--days', type=int,
                       help='Time range in days')
    parser.add_argument('--queries', type=int,
                       help='Number of retrieval queries')
    
    # Scenario mix
    parser.add_argument('--simple', type=float, default=0.3,
                       help='Ratio of simple interactions (0-1)')
    parser.add_argument('--medium', type=float, default=0.5,
                       help='Ratio of medium interactions (0-1)')
    parser.add_argument('--complex', type=float, default=0.2,
                       help='Ratio of complex interactions (0-1)')
    
    # Recall benchmark parameters
    parser.add_argument('--cases', type=int,
                       help='Number of test cases per query type (recall mode)')
    parser.add_argument('--k', type=int, nargs='+',
                       help='k values for recall@k evaluation (recall mode)')
    
    # Options
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM extraction (faster but less accurate)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots (analyze mode)')
    parser.add_argument('--output', '-o',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.analyze:
        result = analyze_existing_data(args)
        if result is None:
            print("\nAnalysis failed. Please check the error messages above.")
            sys.exit(1)
    elif args.recall:
        result = run_recall_benchmark(args)
        if result is None:
            print("\nRecall benchmark failed. Please check the error messages above.")
            sys.exit(1)
    else:
        run_synthetic_benchmark(args)


if __name__ == "__main__":
    main()
