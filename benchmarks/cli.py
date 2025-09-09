#!/usr/bin/env python3
"""
CLI interface for JAM Memory System - LMSYS Data Import and Analysis.

Provides tools for:
1. Importing LMSYS conversation data
2. Analyzing imported memory quality and performance  
3. Running retrieval benchmarks on real data
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.analyze_lmsys import LMSYSAnalyzer
import benchmarks.import_lmsys_data as import_lmsys_data


class BenchmarkCLI:
    """Command-line interface for LMSYS data import and analysis."""
    
    IMPORT_PRESETS = {
        'test': {
            'limit': 10,
            'batch_size': 10,
            'workers': 2,
            'description': 'Test import with 10 conversations'
        },
        'quick': {
            'limit': 100,
            'batch_size': 50,
            'workers': 4,
            'description': 'Quick import (100 conversations)'
        },
        'standard': {
            'limit': 1000,
            'batch_size': 500,
            'workers': 4,
            'description': 'Standard import (1000 conversations)'
        },
        'full': {
            'limit': None,
            'batch_size': 1000,
            'workers': 8,
            'description': 'Full dataset import'
        }
    }
    
    def import_data(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Import LMSYS conversation data."""
        
        if not args.csv_path:
            print("Error: CSV file path is required for import")
            sys.exit(1)
        
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)
        
        # Get import configuration
        if args.preset and args.preset in self.IMPORT_PRESETS:
            preset = self.IMPORT_PRESETS[args.preset]
            limit = preset['limit']
            batch_size = preset['batch_size']
            workers = preset['workers']
            print(f"Using preset: {args.preset} - {preset['description']}")
        else:
            limit = args.limit
            batch_size = args.batch_size or 500
            workers = args.workers or 4
        
        print("\n" + "="*60)
        print("LMSYS DATA IMPORT")
        print("="*60)
        print(f"CSV File: {csv_path}")
        print(f"Batch Size: {batch_size}")
        print(f"Workers: {workers}")
        if limit:
            print(f"Limit: {limit} conversations")
        
        # Create importer and run
        importer = import_lmsys_data.LMSYSImporter(
            str(csv_path),
            batch_size=batch_size,
            num_workers=workers
        )
        
        stats = importer.import_from_csv(limit=limit)
        
        return stats
    
    def analyze_data(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Analyze imported LMSYS data."""
        
        print("\n" + "="*60)
        print("LMSYS DATA ANALYSIS")
        print("="*60)
        
        # Create analyzer
        analyzer = LMSYSAnalyzer()
        
        # Run analysis
        if args.quick:
            print("Running quick analysis (limited queries)...")
            analyzer.test_retrieval_performance(num_queries=10)
            result = analyzer.generate_report(output_file=args.output)
        else:
            print(f"Running full analysis with {args.queries} retrieval queries...")
            result = analyzer.generate_report(output_file=args.output)
        
        # Generate plots if requested
        if args.plot:
            print("\nGenerating visualization plots...")
            analyzer.plot_analysis()
        
        return {
            'total_memories': result.total_memories,
            'unique_sessions': result.unique_sessions,
            'quality_scores': result.quality_scores,
            'retrieval_metrics': result.retrieval_metrics
        }
    
    def list_presets(self):
        """List available import presets."""
        print("\nAvailable Import Presets:")
        print("-" * 40)
        for name, preset in self.IMPORT_PRESETS.items():
            print(f"  {name:12} - {preset['description']}")
            if preset['limit']:
                print(f"               Limit: {preset['limit']} conversations")
            print(f"               Batch: {preset['batch_size']}, Workers: {preset['workers']}")
    
    def run_interactive(self):
        """Run interactive mode for import and analysis."""
        print("\n" + "="*60)
        print("JAM MEMORY SYSTEM - LMSYS DATA TOOLS")
        print("="*60)
        print("\nWhat would you like to do?")
        print("1. Import LMSYS conversation data")
        print("2. Analyze existing memory data")
        print("3. Run retrieval benchmarks")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            csv_path = input("Enter path to LMSYS CSV file: ").strip()
            if not Path(csv_path).exists():
                print(f"Error: File not found: {csv_path}")
                return
            
            print("\nSelect import preset:")
            self.list_presets()
            preset = input("\nEnter preset name (or press Enter for custom): ").strip()
            
            if preset and preset in self.IMPORT_PRESETS:
                config = self.IMPORT_PRESETS[preset]
                importer = import_lmsys_data.LMSYSImporter(
                    csv_path,
                    batch_size=config['batch_size'],
                    num_workers=config['workers']
                )
                importer.import_from_csv(limit=config['limit'])
            else:
                limit = input("Enter limit (or press Enter for all): ").strip()
                limit = int(limit) if limit else None
                batch_size = input("Enter batch size (default 500): ").strip()
                batch_size = int(batch_size) if batch_size else 500
                workers = input("Enter number of workers (default 4): ").strip()
                workers = int(workers) if workers else 4
                
                importer = import_lmsys_data.LMSYSImporter(
                    csv_path,
                    batch_size=batch_size,
                    num_workers=workers
                )
                importer.import_from_csv(limit=limit)
        
        elif choice == '2':
            analyzer = LMSYSAnalyzer()
            plot = input("Generate visualization plots? (y/n): ").strip().lower() == 'y'
            output = input("Save report to file? Enter filename or press Enter to skip: ").strip()
            output = output if output else None
            
            analyzer.generate_report(output_file=output)
            if plot:
                analyzer.plot_analysis()
        
        elif choice == '3':
            analyzer = LMSYSAnalyzer()
            num_queries = input("Number of retrieval queries (default 100): ").strip()
            num_queries = int(num_queries) if num_queries else 100
            
            print(f"\nRunning {num_queries} retrieval queries...")
            metrics = analyzer.test_retrieval_performance(num_queries=num_queries)
            
            print("\nRetrieval Performance:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}")
        
        elif choice == '4':
            print("Exiting...")
            return
        else:
            print("Invalid choice")


def main():
    """Main entry point for LMSYS data tools CLI."""
    
    parser = argparse.ArgumentParser(
        description='JAM Memory System - LMSYS Data Import and Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import LMSYS data with preset
  python -m benchmarks.cli import data.csv --preset standard
  
  # Import with custom settings
  python -m benchmarks.cli import data.csv --limit 5000 --batch-size 1000 --workers 8
  
  # Analyze imported data
  python -m benchmarks.cli analyze --plot --output analysis.json
  
  # Run retrieval benchmarks
  python -m benchmarks.cli benchmark --queries 200
  
  # Interactive mode
  python -m benchmarks.cli --interactive
  
  # List available presets
  python -m benchmarks.cli list
        """
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import LMSYS conversation data')
    import_parser.add_argument('csv_path', help='Path to LMSYS CSV file')
    import_parser.add_argument(
        '--preset',
        choices=list(BenchmarkCLI.IMPORT_PRESETS.keys()),
        help='Use a preset configuration'
    )
    import_parser.add_argument('--limit', type=int, help='Maximum conversations to import')
    import_parser.add_argument('--batch-size', type=int, help='Batch size for processing')
    import_parser.add_argument('--workers', type=int, help='Number of parallel workers')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze imported memory data')
    analyze_parser.add_argument('--output', '-o', help='Save analysis to JSON file')
    analyze_parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    analyze_parser.add_argument('--queries', type=int, default=100, help='Number of retrieval queries')
    analyze_parser.add_argument('--quick', action='store_true', help='Quick analysis (fewer queries)')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run retrieval benchmarks')
    bench_parser.add_argument('--queries', type=int, default=100, help='Number of queries to run')
    bench_parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available presets')
    
    # Interactive mode
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    cli = BenchmarkCLI()
    
    # Handle interactive mode
    if args.interactive:
        cli.run_interactive()
        return
    
    # Handle commands
    if args.command == 'import':
        cli.import_data(args)
    
    elif args.command == 'analyze':
        cli.analyze_data(args)
    
    elif args.command == 'benchmark':
        analyzer = LMSYSAnalyzer()
        print(f"\nRunning {args.queries} retrieval benchmarks...")
        metrics = analyzer.test_retrieval_performance(num_queries=args.queries)
        
        print("\nRetrieval Performance:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
        
        # Always save results for analysis
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = results_dir / f"lmsys_benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    
    elif args.command == 'list':
        cli.list_presets()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()