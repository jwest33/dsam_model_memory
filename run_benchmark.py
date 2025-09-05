#!/usr/bin/env python
"""
CLI interface for running JAM benchmarks.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from agentic_memory.benchmark import MemoryBenchmark, InteractionGenerator
from agentic_memory.router import MemoryRouter


def parse_scenario_mix(mix_str: str) -> dict:
    """Parse scenario mix from string format 'simple:0.3,medium:0.5,complex:0.2'."""
    if not mix_str:
        return None
    
    mix = {}
    for part in mix_str.split(','):
        complexity, weight = part.split(':')
        mix[complexity] = float(weight)
    
    # Normalize weights to sum to 1.0
    total = sum(mix.values())
    if total > 0:
        mix = {k: v/total for k, v in mix.items()}
    
    return mix


async def run_interactive_benchmark():
    """Run an interactive benchmark with user prompts."""
    print("\nJAM Interactive Benchmark Setup")
    print("="*50)
    
    # Get parameters
    interactions = int(input("Number of interactions to generate [100]: ") or "100")
    days = int(input("Time range in days [30]: ") or "30")
    queries = int(input("Number of retrieval queries to test [20]: ") or "20")
    
    print("\nScenario complexity mix (current: simple:0.3, medium:0.5, complex:0.2)")
    custom_mix = input("Enter custom mix or press Enter for default: ")
    scenario_mix = parse_scenario_mix(custom_mix) if custom_mix else None
    
    print("\n" + "="*50)
    confirm = input(f"\nGenerate {interactions} interactions over {days} days? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Benchmark cancelled.")
        return
    
    # Run benchmark
    benchmark = MemoryBenchmark()
    result = await benchmark.run_benchmark(
        interaction_count=interactions,
        time_range_days=days,
        retrieval_queries=queries,
        scenario_mix=scenario_mix
    )
    
    benchmark.print_results(result)
    
    # Ask if user wants to save results
    save = input("\nSave results to file? (y/n): ")
    if save.lower() == 'y':
        benchmark.save_results(result)


async def run_preset_benchmark(preset: str):
    """Run a preset benchmark configuration."""
    presets = {
        "quick": {
            "interaction_count": 25,
            "time_range_days": 3,
            "retrieval_queries": 5,
            "scenario_mix": {"simple": 0.6, "medium": 0.3, "complex": 0.1},
            "skip_llm_extraction": True  # Fast mode for quick preset
        },
        "standard": {
            "interaction_count": 100,
            "time_range_days": 30,
            "retrieval_queries": 20,
            "scenario_mix": {"simple": 0.3, "medium": 0.5, "complex": 0.2},
            "skip_llm_extraction": False
        },
        "comprehensive": {
            "interaction_count": 500,
            "time_range_days": 90,
            "retrieval_queries": 50,
            "scenario_mix": {"simple": 0.25, "medium": 0.5, "complex": 0.25},
            "skip_llm_extraction": False
        },
        "stress": {
            "interaction_count": 1000,
            "time_range_days": 365,
            "retrieval_queries": 100,
            "scenario_mix": {"simple": 0.2, "medium": 0.4, "complex": 0.4},
            "skip_llm_extraction": False
        }
    }
    
    if preset not in presets:
        print(f"Unknown preset: {preset}")
        print(f"Available presets: {', '.join(presets.keys())}")
        return
    
    config = presets[preset]
    print(f"\nRunning '{preset}' benchmark preset")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    benchmark = MemoryBenchmark()
    result = await benchmark.run_benchmark(**config)
    
    benchmark.print_results(result)
    benchmark.save_results(result, f"benchmark_{preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


async def generate_sample_data(count: int, output_file: str = None):
    """Generate sample interaction data without ingesting."""
    print(f"\nGenerating {count} sample interactions...")
    
    generator = InteractionGenerator()
    scenarios = await generator.generate_batch(count=count, time_range_days=30)
    
    # Convert to JSON-serializable format
    data = []
    for scenario in scenarios:
        data.append({
            "timestamp": scenario.timestamp.isoformat(),
            "actor": scenario.actor,
            "content": scenario.content,
            "topic": scenario.topic,
            "interaction_type": scenario.interaction_type,
            "complexity": scenario.complexity,
            "metadata": scenario.metadata
        })
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {count} interactions to {output_file}")
    else:
        # Print sample
        print(f"\nGenerated {count} interactions. Sample:")
        for item in data[:3]:
            print(f"\n[{item['timestamp']}] {item['actor']} ({item['topic']}):")
            print(f"  {item['content'][:100]}...")


async def analyze_existing_memory():
    """Analyze existing memory system statistics."""
    from agentic_memory.storage.sql_store import MemoryStore
    from agentic_memory.storage.faiss_index import FaissIndex
    from agentic_memory.config import cfg
    
    store = MemoryStore(cfg.db_path)
    index = FaissIndex(cfg.embed_dim, cfg.index_path)
    router = MemoryRouter(store, index)
    
    print("\nMemory System Analysis")
    print("="*50)
    
    # Get memories directly from database
    with store.connect() as con:
        memory_count = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print(f"\nStorage Statistics:")
        print(f"  • Total memories: {memory_count}")
        
        if memory_count > 0:
            # Get all memories
            memories = con.execute("SELECT * FROM memories ORDER BY when_ts DESC LIMIT 10000").fetchall()
            
            # Get date range
            dates = []
            for m in memories:
                if m['when_ts']:
                    try:
                        dates.append(datetime.fromisoformat(m['when_ts']))
                    except:
                        pass
            if dates:
                dates.sort()
                print(f"  • Date range: {dates[0].date()} to {dates[-1].date()}")
                
            # Get actor distribution
            actors = {}
            for row in memories:
                # SQLite Row objects need index access for column names
                actor = row[4] or "unknown"  # 'who' is the 5th column
                actors[actor] = actors.get(actor, 0) + 1
            
            print(f"\nTop Actors:")
            for actor, count in sorted(actors.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {actor}: {count} memories")
            
            # Get event type distribution
            events = {}
            for row in memories:
                event_type = row[2] or "unknown"  # 'event_type' is the 3rd column
                events[event_type] = events.get(event_type, 0) + 1
            
            print(f"\nEvent Types:")
            for event_type, count in sorted(events.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  • {event_type}: {count} memories")
    
    # Storage sizes
    import os
    from agentic_memory.config import Config
    config = Config()
    
    db_size = os.path.getsize(config.db_path) if os.path.exists(config.db_path) else 0
    print(f"\nDatabase size: {db_size / (1024*1024):.2f} MB")
    
    if os.path.exists(config.index_path):
        index_size = os.path.getsize(config.index_path)
        print(f"Index size: {index_size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="JAM Memory System Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive benchmark
  python run_benchmark.py --interactive
  
  # Run preset benchmark
  python run_benchmark.py --preset standard
  
  # Custom benchmark
  python run_benchmark.py --interactions 200 --days 60 --queries 30
  
  # Generate sample data
  python run_benchmark.py --generate 50 --output samples.json
  
  # Analyze existing memory
  python run_benchmark.py --analyze
        """
    )
    
    # Modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--interactive', '-i', action='store_true',
                           help='Run interactive benchmark setup')
    mode_group.add_argument('--preset', '-p', choices=['quick', 'standard', 'comprehensive', 'stress'],
                           help='Run a preset benchmark configuration')
    mode_group.add_argument('--generate', '-g', type=int, metavar='COUNT',
                           help='Generate sample interactions without ingesting')
    mode_group.add_argument('--analyze', '-a', action='store_true',
                           help='Analyze existing memory system')
    
    # Custom parameters
    parser.add_argument('--interactions', type=int, default=100,
                       help='Number of interactions to generate (default: 100)')
    parser.add_argument('--days', type=int, default=30,
                       help='Time range in days (default: 30)')
    parser.add_argument('--queries', type=int, default=20,
                       help='Number of retrieval queries (default: 20)')
    parser.add_argument('--mix', type=str,
                       help='Scenario mix as "simple:0.3,medium:0.5,complex:0.2"')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results or generated data')
    parser.add_argument('--fast', '-f', action='store_true',
                       help='Fast mode: skip LLM extraction for quicker benchmarks')
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.interactive:
        asyncio.run(run_interactive_benchmark())
    elif args.preset:
        asyncio.run(run_preset_benchmark(args.preset))
    elif args.generate:
        asyncio.run(generate_sample_data(args.generate, args.output))
    elif args.analyze:
        asyncio.run(analyze_existing_memory())
    else:
        # Run custom benchmark
        scenario_mix = parse_scenario_mix(args.mix) if args.mix else None
        
        async def run_custom():
            benchmark = MemoryBenchmark()
            result = await benchmark.run_benchmark(
                interaction_count=args.interactions,
                time_range_days=args.days,
                retrieval_queries=args.queries,
                scenario_mix=scenario_mix,
                skip_llm_extraction=args.fast
            )
            benchmark.print_results(result)
            if args.output:
                benchmark.save_results(result, args.output)
        
        asyncio.run(run_custom())


if __name__ == "__main__":
    main()
