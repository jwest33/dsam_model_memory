"""
Command-line interface for the 5W1H + MHN Memory Framework

Provides tools for memory operations and testing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from agent.memory_agent import MemoryAgent
from models.event import Event, EventType
from config import get_config, reset_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryCLI:
    """CLI for memory operations"""
    
    def __init__(self):
        """Initialize CLI"""
        self.agent = None
        self.config = get_config()
    
    def initialize(self, path: Optional[str] = None):
        """Initialize memory system"""
        if path:
            self.config.storage.state_dir = Path(path)
            reset_config()
            self.config = get_config()
        
        self.agent = MemoryAgent(self.config)
        print(f"✓ Initialized memory system at {self.config.storage.state_dir}")
        
        # Show statistics
        stats = self.agent.get_statistics()
        print(f"  Raw memories: {stats['raw_count']}")
        print(f"  Processed memories: {stats['processed_count']}")
        print(f"  Episodes: {stats['episode_count']}")
    
    def remember(
        self,
        who: str,
        what: str,
        where: Optional[str] = None,
        why: Optional[str] = None,
        how: Optional[str] = None,
        event_type: str = "action",
        tags: Optional[str] = None
    ):
        """Store a memory"""
        if not self.agent:
            self.initialize()
        
        # Parse tags
        tag_list = tags.split(',') if tags else []
        
        # Store memory
        success, message, event = self.agent.remember(
            who=who,
            what=what,
            where=where,
            why=why,
            how=how,
            event_type=event_type,
            tags=tag_list
        )
        
        if success:
            print(f"✓ Stored memory: {event.id[:8]}")
            print(f"  Salience: {event.salience:.2f}")
            print(f"  Episode: {event.episode_id}")
            print(f"  Message: {message}")
        else:
            print(f"✗ Failed to store memory: {message}")
    
    def recall(
        self,
        query: Optional[str] = None,
        who: Optional[str] = None,
        what: Optional[str] = None,
        where: Optional[str] = None,
        why: Optional[str] = None,
        how: Optional[str] = None,
        k: int = 5,
        include_raw: bool = False
    ):
        """Recall memories"""
        if not self.agent:
            self.initialize()
        
        # Parse query if provided as JSON
        query_dict = {}
        if query:
            try:
                query_dict = json.loads(query)
            except:
                # Treat as 'what' field
                query_dict = {"what": query}
        
        # Add individual fields
        if who: query_dict["who"] = who
        if what: query_dict["what"] = what
        if where: query_dict["where"] = where
        if why: query_dict["why"] = why
        if how: query_dict["how"] = how
        
        # Retrieve memories
        results = self.agent.recall(
            query=query_dict,
            k=k,
            include_raw=include_raw
        )
        
        print(f"Found {len(results)} memories matching query:")
        print(f"Query: {query_dict}\n")
        
        for i, (event, score) in enumerate(results, 1):
            print(f"{i}. [{score:.3f}] {event.five_w1h.who}: {event.five_w1h.what[:100]}")
            print(f"   When: {event.five_w1h.when}")
            print(f"   Where: {event.five_w1h.where}")
            print(f"   Why: {event.five_w1h.why[:100]}")
            print(f"   How: {event.five_w1h.how}")
            print(f"   Salience: {event.salience:.2f}, Type: {event.event_type.value}")
            print()
    
    def observe(
        self,
        what: str,
        who: Optional[str] = None,
        where: Optional[str] = None
    ):
        """Record an observation"""
        if not self.agent:
            self.initialize()
        
        success, message, event = self.agent.observe(
            what=what,
            who=who,
            where=where
        )
        
        if success:
            print(f"✓ Recorded observation: {event.id[:8]}")
            print(f"  Episode: {event.episode_id}")
            print(f"  Message: {message}")
        else:
            print(f"✗ Failed to record observation: {message}")
    
    def episode(self, episode_id: Optional[str] = None):
        """Show episode events"""
        if not self.agent:
            self.initialize()
        
        events = self.agent.get_episode(episode_id)
        
        if not events:
            print(f"No events found for episode: {episode_id or 'current'}")
            return
        
        episode_id = events[0].episode_id if events else "unknown"
        print(f"Episode {episode_id} - {len(events)} events:\n")
        
        for i, event in enumerate(events, 1):
            marker = "→" if event.event_type == EventType.ACTION else "←"
            print(f"{i}. {marker} [{event.event_type.value}] {event.five_w1h.who}: {event.five_w1h.what[:100]}")
            print(f"   {event.five_w1h.when}")
        
        # Show summary if available
        print(f"\nSummary:")
        summary = self.agent.summarize_episode(episode_id)
        print(summary)
    
    def stats(self):
        """Show memory statistics"""
        if not self.agent:
            self.initialize()
        
        stats = self.agent.get_statistics()
        
        print("Memory System Statistics:")
        print("-" * 40)
        print(f"Raw memories: {stats['raw_count']}")
        print(f"Processed memories: {stats['processed_count']}")
        print(f"Total episodes: {stats['episode_count']}")
        print(f"Current episode: {stats.get('current_episode', 'None')}")
        print(f"Current goal: {stats.get('current_goal', 'None')}")
        print(f"Operations: {stats['operation_count']}")
        print(f"LLM available: {stats['llm_available']}")
        
        print(f"\nHopfield Network:")
        hopfield = stats['hopfield']
        print(f"  Memories: {hopfield['memory_count']}/{hopfield['max_capacity']}")
        print(f"  Utilization: {hopfield['utilization']:.1%}")
        print(f"  Diversity: {hopfield.get('diversity', 0):.2f}")
        print(f"  Temperature: {hopfield['temperature']}")
        
        if 'event_types' in stats:
            print(f"\nEvent Types:")
            for event_type, count in stats['event_types'].items():
                print(f"  {event_type}: {count}")
        
        print(f"\nLast save: {stats['last_save']}")
    
    def clear(self, confirm: bool = False):
        """Clear all memories"""
        if not confirm:
            response = input("Are you sure you want to clear all memories? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return
        
        if not self.agent:
            self.initialize()
        
        self.agent.clear()
        print("✓ Cleared all memories")
    
    def save(self):
        """Save memory state"""
        if not self.agent:
            self.initialize()
        
        self.agent.save()
        print(f"✓ Saved memory state to {self.config.storage.state_dir}")
    
    def goal(self, goal: str):
        """Set the current goal"""
        if not self.agent:
            self.initialize()
        
        self.agent.set_goal(goal)
        print(f"✓ Set goal: {goal}")
    
    def demo(self):
        """Run a demonstration"""
        print("Running memory system demonstration...\n")
        
        if not self.agent:
            self.initialize()
        
        # Set a goal
        self.agent.set_goal("Demonstrate the 5W1H memory system")
        
        # Store some memories
        print("1. Storing action memory...")
        success, msg, event1 = self.agent.remember(
            who="LLM",
            what="Generated Python code for data analysis",
            where="conversation:demo",
            why="User requested data summary",
            how="Code generation module",
            event_type="action",
            tags=["code", "python", "data"]
        )
        print(f"   {msg}\n")
        
        print("2. Recording observation...")
        success, msg, event2 = self.agent.observe(
            what="Code executed successfully, returned 10 rows",
            who="Python interpreter",
            where="jupyter:cell_5"
        )
        print(f"   {msg}\n")
        
        # Chain events
        print("3. Chaining action and observation...")
        self.agent.chain_events(event1, event2)
        print("   ✓ Events linked\n")
        
        print("4. Storing another action...")
        success, msg, event3 = self.agent.remember(
            who="User",
            what="Asked for visualization of the data",
            where="conversation:demo",
            why="Need to see data patterns",
            how="Natural language input",
            event_type="user_input"
        )
        print(f"   {msg}\n")
        
        # Recall memories
        print("5. Recalling memories about 'data'...")
        results = self.agent.recall(what="data", k=3)
        for event, score in results:
            print(f"   [{score:.3f}] {event.five_w1h.who}: {event.five_w1h.what[:50]}...")
        print()
        
        # Show episode
        print("6. Current episode summary:")
        summary = self.agent.summarize_episode()
        print(f"   {summary}\n")
        
        # Show statistics
        print("7. System statistics:")
        stats = self.agent.get_statistics()
        print(f"   Total memories: {stats['raw_count'] + stats['processed_count']}")
        print(f"   Hopfield utilization: {stats['hopfield']['utilization']:.1%}")
        print(f"   Current episode: {stats['current_episode']}")
        
        print("\n✓ Demonstration complete!")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="5W1H + MHN Memory Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize memory system')
    init_parser.add_argument('--path', help='State directory path')
    
    # Remember command
    remember_parser = subparsers.add_parser('remember', help='Store a memory')
    remember_parser.add_argument('--who', required=True, help='Agent/actor')
    remember_parser.add_argument('--what', required=True, help='Action/content')
    remember_parser.add_argument('--where', help='Context/location')
    remember_parser.add_argument('--why', help='Intent/trigger')
    remember_parser.add_argument('--how', help='Mechanism')
    remember_parser.add_argument('--type', default='action', 
                                choices=['action', 'observation', 'user_input', 'system_event'])
    remember_parser.add_argument('--tags', help='Comma-separated tags')
    
    # Recall command
    recall_parser = subparsers.add_parser('recall', help='Recall memories')
    recall_parser.add_argument('--query', help='Query (JSON or text)')
    recall_parser.add_argument('--who', help='Filter by who')
    recall_parser.add_argument('--what', help='Filter by what')
    recall_parser.add_argument('--where', help='Filter by where')
    recall_parser.add_argument('--why', help='Filter by why')
    recall_parser.add_argument('--how', help='Filter by how')
    recall_parser.add_argument('--k', type=int, default=5, help='Number of results')
    recall_parser.add_argument('--include-raw', action='store_true', help='Include raw memories')
    
    # Observe command
    observe_parser = subparsers.add_parser('observe', help='Record an observation')
    observe_parser.add_argument('--what', required=True, help='Observation content')
    observe_parser.add_argument('--who', help='Observer')
    observe_parser.add_argument('--where', help='Location')
    
    # Episode command
    episode_parser = subparsers.add_parser('episode', help='Show episode events')
    episode_parser.add_argument('--id', help='Episode ID')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all memories')
    clear_parser.add_argument('--confirm', action='store_true', help='Skip confirmation')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save memory state')
    
    # Goal command
    goal_parser = subparsers.add_parser('goal', help='Set current goal')
    goal_parser.add_argument('goal', help='Goal description')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = MemoryCLI()
    
    # Execute command
    if args.command == 'init':
        cli.initialize(args.path)
    
    elif args.command == 'remember':
        cli.remember(
            who=args.who,
            what=args.what,
            where=args.where,
            why=args.why,
            how=args.how,
            event_type=args.type,
            tags=args.tags
        )
    
    elif args.command == 'recall':
        cli.recall(
            query=args.query,
            who=args.who,
            what=args.what,
            where=args.where,
            why=args.why,
            how=args.how,
            k=args.k,
            include_raw=args.include_raw
        )
    
    elif args.command == 'observe':
        cli.observe(
            what=args.what,
            who=args.who,
            where=args.where
        )
    
    elif args.command == 'episode':
        cli.episode(args.id)
    
    elif args.command == 'stats':
        cli.stats()
    
    elif args.command == 'clear':
        cli.clear(args.confirm)
    
    elif args.command == 'save':
        cli.save()
    
    elif args.command == 'goal':
        cli.goal(args.goal)
    
    elif args.command == 'demo':
        cli.demo()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
