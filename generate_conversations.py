"""
Generate sample conversations to demonstrate the dynamic memory system.

This script creates realistic conversation scenarios that showcase:
- Dynamic clustering based on context
- Adaptive embeddings that evolve
- No arbitrary salience thresholds
- Unlimited memory capacity
"""

import os
# Run in offline mode to avoid HuggingFace rate limits
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import logging
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

from agent.memory_agent import MemoryAgent
from models.event import Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConversationGenerator:
    """Generate realistic conversations for the memory system"""
    
    def __init__(self):
        """Initialize the conversation generator"""
        self.agent = MemoryAgent()
        self.topics = [
            "machine learning", "web development", "data analysis",
            "system design", "debugging", "documentation", "testing",
            "deployment", "optimization", "security"
        ]
        self.actors = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        self.locations = ["office", "meeting room", "slack", "github", "lab", "home"]
        
    def generate_technical_conversation(self) -> List[Dict]:
        """Generate a technical conversation about a project"""
        topic = random.choice(self.topics)
        events = []
        
        # Initial question
        events.append({
            "who": random.choice(self.actors),
            "what": f"asked about implementing {topic} features",
            "where": random.choice(self.locations),
            "why": "understand requirements",
            "how": "verbal discussion"
        })
        
        # Research phase
        events.append({
            "who": random.choice(self.actors),
            "what": f"researched {topic} best practices and patterns",
            "where": "documentation sites",
            "why": "gather information",
            "how": "web search and reading"
        })
        
        # Implementation
        events.append({
            "who": random.choice(self.actors),
            "what": f"implemented initial {topic} prototype",
            "where": "development environment",
            "why": "proof of concept",
            "how": "coding"
        })
        
        # Testing
        events.append({
            "who": "system",
            "what": f"ran tests on {topic} implementation",
            "where": "CI/CD pipeline",
            "why": "verify functionality",
            "how": "automated testing"
        })
        
        # Review
        events.append({
            "who": random.choice(self.actors),
            "what": f"reviewed and provided feedback on {topic} code",
            "where": "pull request",
            "why": "ensure quality",
            "how": "code review"
        })
        
        return events
    
    def generate_debugging_session(self) -> List[Dict]:
        """Generate a debugging session"""
        bug_type = random.choice(["memory leak", "race condition", "null pointer", 
                                 "infinite loop", "type error", "network timeout"])
        events = []
        
        # Bug discovery
        events.append({
            "who": random.choice(self.actors),
            "what": f"discovered {bug_type} in production",
            "where": "monitoring dashboard",
            "why": "system alert triggered",
            "how": "automated monitoring"
        })
        
        # Investigation
        events.append({
            "who": random.choice(self.actors),
            "what": f"investigated {bug_type} using logs and traces",
            "where": "debugging tools",
            "why": "identify root cause",
            "how": "log analysis"
        })
        
        # Hypothesis
        events.append({
            "who": random.choice(self.actors),
            "what": f"formed hypothesis about {bug_type} cause",
            "where": "team discussion",
            "why": "narrow down possibilities",
            "how": "collaborative analysis"
        })
        
        # Fix attempt
        events.append({
            "who": random.choice(self.actors),
            "what": f"implemented fix for {bug_type}",
            "where": "codebase",
            "why": "resolve issue",
            "how": "code modification"
        })
        
        # Verification
        events.append({
            "who": "system",
            "what": f"verified {bug_type} fix in staging",
            "where": "staging environment",
            "why": "ensure fix works",
            "how": "integration testing"
        })
        
        return events
    
    def generate_planning_session(self) -> List[Dict]:
        """Generate a planning session"""
        project = random.choice(["API redesign", "database migration", "UI overhaul",
                               "microservices split", "performance optimization"])
        events = []
        
        # Planning initiation
        events.append({
            "who": random.choice(self.actors),
            "what": f"initiated planning for {project}",
            "where": "planning meeting",
            "why": "quarterly objectives",
            "how": "meeting agenda"
        })
        
        # Requirements gathering
        events.append({
            "who": random.choice(self.actors),
            "what": f"gathered requirements for {project}",
            "where": "stakeholder interviews",
            "why": "understand needs",
            "how": "interviews and surveys"
        })
        
        # Design proposal
        events.append({
            "who": random.choice(self.actors),
            "what": f"created design proposal for {project}",
            "where": "design documents",
            "why": "outline approach",
            "how": "technical writing"
        })
        
        # Estimation
        events.append({
            "who": random.choice(self.actors),
            "what": f"estimated timeline and resources for {project}",
            "where": "project management tool",
            "why": "planning purposes",
            "how": "estimation techniques"
        })
        
        # Approval
        events.append({
            "who": random.choice(self.actors),
            "what": f"approved {project} plan with modifications",
            "where": "decision meeting",
            "why": "move forward",
            "how": "consensus building"
        })
        
        return events
    
    def store_conversation(self, events: List[Dict], episode_name: str):
        """Store a conversation in the memory system"""
        logger.info(f"Storing conversation: {episode_name}")
        
        # Start a new episode for this conversation
        self.agent.start_episode()
        stored_events = []
        
        for i, event_data in enumerate(events):
            # Add some temporal spacing
            time.sleep(0.05)
            
            # Store the event
            success, message, event = self.agent.remember(**event_data)
            stored_events.append(event)
            
            if success:
                logger.debug(f"  Stored: {event.five_w1h.what[:50]}...")
            else:
                logger.warning(f"  Failed to store: {message}")
        
        # End the episode
        episode_id = self.agent.end_episode()
        logger.info(f"  Episode {episode_id} completed with {len(stored_events)} events")
        
        return stored_events
    
    def demonstrate_clustering(self):
        """Demonstrate dynamic clustering based on queries"""
        logger.info("\n" + "="*60)
        logger.info("Demonstrating Dynamic Clustering")
        logger.info("="*60)
        
        # Query for technical topics
        logger.info("\n1. Querying for 'implementation' (technical context):")
        results = self.agent.recall(what="implementation", k=5)
        for event, score in results[:3]:
            logger.info(f"   [{score:.3f}] {event.five_w1h.who}: {event.five_w1h.what[:40]}...")
        
        # Query for debugging
        logger.info("\n2. Querying for 'bug' (debugging context):")
        results = self.agent.recall(what="bug", k=5)
        for event, score in results[:3]:
            logger.info(f"   [{score:.3f}] {event.five_w1h.who}: {event.five_w1h.what[:40]}...")
        
        # Query for planning
        logger.info("\n3. Querying for 'planning' (project context):")
        results = self.agent.recall(what="planning", k=5)
        for event, score in results[:3]:
            logger.info(f"   [{score:.3f}] {event.five_w1h.who}: {event.five_w1h.what[:40]}...")
        
        # Cross-context query
        logger.info("\n4. Querying across contexts - who='Alice':")
        results = self.agent.recall(who="Alice", k=5)
        for event, score in results[:3]:
            logger.info(f"   [{score:.3f}] {event.five_w1h.what[:50]}...")
    
    def demonstrate_adaptation(self):
        """Demonstrate adaptive embeddings"""
        logger.info("\n" + "="*60)
        logger.info("Demonstrating Adaptive Embeddings")
        logger.info("="*60)
        
        # Initial query
        logger.info("\n1. Initial query for 'testing':")
        results = self.agent.recall(what="testing", k=3)
        if results:
            for event, score in results:
                logger.info(f"   [{score:.3f}] {event.five_w1h.what[:40]}...")
            
            # Provide feedback
            logger.info("\n2. Marking first result as highly relevant...")
            self.agent.adapt_from_feedback(
                query={"what": "testing"},
                positive_events=[results[0][0].id],
                negative_events=[]
            )
            
            # Query again
            logger.info("\n3. Querying again after adaptation:")
            new_results = self.agent.recall(what="testing", k=3)
            for event, score in new_results:
                logger.info(f"   [{score:.3f}] {event.five_w1h.what[:40]}...")
            
            logger.info("\n   Note: Embeddings have adapted based on feedback!")
    
    def show_statistics(self):
        """Show system statistics"""
        stats = self.agent.get_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("System Statistics")
        logger.info("="*60)
        logger.info(f"Total events stored: {stats.get('total_events', 0)}")
        logger.info(f"Total queries made: {stats.get('total_queries', 0)}")
        logger.info(f"Episodes created: {stats.get('episodes', 0)}")
        logger.info(f"Embeddings evolved: {stats.get('evolved_embeddings', 0)}")
        logger.info(f"Operations performed: {stats.get('operation_count', 0)}")
        
        if 'chromadb_stats' in stats:
            db_stats = stats['chromadb_stats']
            logger.info(f"\nChromaDB Statistics:")
            logger.info(f"  Events in database: {db_stats.get('events', 0)}")
            logger.info(f"  Memory blocks: {db_stats.get('blocks', 0)}")
    
    def run(self, num_conversations: int = 3):
        """Run the conversation generator"""
        logger.info("Starting Conversation Generation")
        logger.info(f"Generating {num_conversations} conversations per type...")
        
        # Generate different types of conversations
        for i in range(num_conversations):
            # Technical conversations
            tech_events = self.generate_technical_conversation()
            self.store_conversation(tech_events, f"Technical-{i+1}")
            
            # Debugging sessions
            debug_events = self.generate_debugging_session()
            self.store_conversation(debug_events, f"Debugging-{i+1}")
            
            # Planning sessions
            plan_events = self.generate_planning_session()
            self.store_conversation(plan_events, f"Planning-{i+1}")
        
        # Demonstrate features
        self.demonstrate_clustering()
        self.demonstrate_adaptation()
        self.show_statistics()
        
        # Save the state
        logger.info("\nSaving memory state...")
        self.agent.save()
        logger.info("Memory state saved successfully!")
        
        logger.info("\n" + "="*60)
        logger.info("Conversation Generation Complete!")
        logger.info("="*60)
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ No salience thresholds - all memories stored")
        logger.info("✓ Dynamic clustering based on query context")
        logger.info("✓ Adaptive embeddings that evolve with usage")
        logger.info("✓ Unlimited memory capacity")
        logger.info("✓ ChromaDB backend for scalability")


def main():
    """Main entry point"""
    generator = ConversationGenerator()
    generator.run(num_conversations=2)  # Generate 2 of each type


if __name__ == "__main__":
    main()