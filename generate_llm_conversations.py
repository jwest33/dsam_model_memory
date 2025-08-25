"""
Generate realistic conversations using LLM for the dynamic memory system.

This script creates realistic conversation scenarios using an LLM to generate
both sides of technical conversations, demonstrating the memory system features.
"""

import os
# Run in offline mode to avoid HuggingFace rate limits
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import logging
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple

from agent.memory_agent import MemoryAgent
from models.event import Event
from llm.llm_interface import LLMInterface
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMConversationGenerator:
    """Generate realistic conversations using LLM"""
    
    def __init__(self):
        """Initialize the conversation generator"""
        self.agent = MemoryAgent()
        self.config = get_config()
        self.llm = LLMInterface(self.config.llm)
        
        # Conversation topics
        self.topics = [
            "implementing a distributed cache system",
            "debugging memory leaks in production",
            "designing a microservices architecture", 
            "optimizing database query performance",
            "setting up CI/CD pipelines",
            "implementing real-time data streaming",
            "building a recommendation engine",
            "scaling a web application",
            "implementing authentication and authorization",
            "migrating from monolith to microservices"
        ]
        
        # Participant names
        self.participants = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        
    def generate_conversation_with_llm(self, topic: str, num_turns: int = 5) -> List[Dict]:
        """
        Generate a conversation using the LLM
        
        Args:
            topic: The topic of conversation
            num_turns: Number of conversation turns
            
        Returns:
            List of event dictionaries
        """
        events = []
        conversation_history = []
        participants = random.sample(self.participants, 2)
        location = random.choice(["slack", "meeting room", "github", "video call"])
        
        # Initial prompt to set up the conversation
        system_prompt = f"""You are simulating a technical conversation between {participants[0]} and {participants[1]} about {topic}.
Generate realistic, technical dialogue. Be specific and detailed.
Keep responses concise (1-2 sentences).
Focus on technical aspects, problems, and solutions."""
        
        for turn in range(num_turns):
            current_speaker = participants[turn % 2]
            other_speaker = participants[(turn + 1) % 2]
            
            # Generate the conversation turn
            if turn == 0:
                # First turn - initiate the conversation
                prompt = f"As {current_speaker}, start a conversation about {topic}. Ask a specific technical question or raise a concern."
            else:
                # Continue the conversation based on history
                last_message = conversation_history[-1]
                prompt = f"As {current_speaker}, respond to {other_speaker}'s message: '{last_message}'. Provide a technical response, suggestion, or follow-up question."
            
            # Get LLM response (combine system prompt with user prompt)
            full_prompt = f"{system_prompt}\n\n{prompt}\n\nResponse:"
            llm_response = self.llm.generate(
                prompt=full_prompt,
                temperature=0.7,
                max_tokens=100
            )
            
            # Clean up the response
            message = llm_response.strip()
            if message.startswith(f"{current_speaker}:"):
                message = message[len(f"{current_speaker}:")].strip()
            
            # Determine the type of message
            if "?" in message:
                event_type = "question"
                why = "seeking information"
            elif any(word in message.lower() for word in ["implement", "create", "build", "develop"]):
                event_type = "action"
                why = "implementation task"
            elif any(word in message.lower() for word in ["found", "discovered", "noticed", "see"]):
                event_type = "observation"
                why = "sharing findings"
            else:
                event_type = "discussion"
                why = "collaborative planning"
            
            # Create the event
            event = {
                "who": current_speaker,
                "what": message,
                "where": location,
                "why": why,
                "how": "verbal communication"
            }
            
            events.append(event)
            conversation_history.append(message)
            
            # Small delay to simulate natural conversation flow
            time.sleep(0.05)
        
        return events
    
    def generate_debugging_session_with_llm(self) -> List[Dict]:
        """Generate a debugging session using LLM"""
        bug_types = [
            "memory leak in the authentication service",
            "race condition in the payment processor",
            "deadlock in the database connection pool",
            "infinite loop in the data processing pipeline",
            "null pointer exception in the API gateway",
            "performance degradation in the search service"
        ]
        
        bug = random.choice(bug_types)
        events = []
        
        # Generate the debugging conversation
        prompts = [
            f"Describe discovering {bug} in production. Be specific about symptoms.",
            f"Explain how you would investigate {bug}. Mention specific tools and techniques.",
            f"Propose a hypothesis for what's causing {bug}.",
            f"Describe the fix you would implement for {bug}.",
            f"Explain how you would verify the fix for {bug} works."
        ]
        
        participants = random.sample(self.participants, 3)
        
        for i, prompt in enumerate(prompts):
            speaker = participants[i % len(participants)]
            
            full_prompt = f"You are {speaker}, a senior engineer debugging a production issue. Be technical and specific.\n\n{prompt}\n\nResponse:"
            llm_response = self.llm.generate(
                prompt=full_prompt,
                temperature=0.6,
                max_tokens=100
            )
            
            event = {
                "who": speaker if i > 0 else "monitoring system",
                "what": llm_response.strip(),
                "where": "production environment" if i == 0 else "debugging session",
                "why": ["alert triggered", "root cause analysis", "hypothesis formation", "bug fix", "verification"][i],
                "how": ["automated monitoring", "log analysis", "collaborative debugging", "code modification", "testing"][i]
            }
            
            events.append(event)
            time.sleep(0.05)
        
        return events
    
    def store_conversation(self, events: List[Dict], conversation_name: str):
        """Store a conversation in the memory system"""
        logger.info(f"Storing conversation: {conversation_name}")
        
        # Start a new episode
        self.agent.start_episode()
        stored_events = []
        
        for event_data in events:
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
    
    def demonstrate_features(self):
        """Demonstrate dynamic clustering and adaptive embeddings"""
        logger.info("\n" + "="*60)
        logger.info("Demonstrating Dynamic Memory Features")
        logger.info("="*60)
        
        # Test various queries
        test_queries = [
            {"what": "implementation", "context": "technical"},
            {"what": "bug", "context": "debugging"},
            {"what": "performance", "context": "optimization"},
            {"who": "Alice", "context": "person-specific"}
        ]
        
        for query_info in test_queries:
            query = {k: v for k, v in query_info.items() if k != "context"}
            context = query_info.get("context", "general")
            
            logger.info(f"\nQuerying for {query} ({context} context):")
            results = self.agent.recall(**query, k=3)
            
            for event, score in results:
                logger.info(f"  [{score:.3f}] {event.five_w1h.who}: {event.five_w1h.what[:40]}...")
        
        # Demonstrate adaptation
        logger.info("\n" + "="*60)
        logger.info("Demonstrating Adaptive Embeddings")
        logger.info("="*60)
        
        # Query and adapt
        results = self.agent.recall(what="optimization", k=3)
        if results:
            logger.info("Initial results for 'optimization':")
            for event, score in results:
                logger.info(f"  [{score:.3f}] {event.five_w1h.what[:40]}...")
            
            # Provide feedback
            logger.info("\nAdapting based on relevance feedback...")
            self.agent.adapt_from_feedback(
                query={"what": "optimization"},
                positive_events=[results[0][0].id],
                negative_events=[]
            )
            
            # Query again
            new_results = self.agent.recall(what="optimization", k=3)
            logger.info("\nResults after adaptation:")
            for event, score in new_results:
                logger.info(f"  [{score:.3f}] {event.five_w1h.what[:40]}...")
    
    def run(self, num_conversations: int = 2):
        """Run the conversation generator"""
        logger.info("Starting LLM-based Conversation Generation")
        logger.info(f"Generating {num_conversations} conversations per type...")
        
        # Generate different types of conversations
        for i in range(num_conversations):
            # Technical conversation
            topic = random.choice(self.topics)
            logger.info(f"\nGenerating technical conversation about: {topic}")
            tech_events = self.generate_conversation_with_llm(topic, num_turns=5)
            self.store_conversation(tech_events, f"Technical-{i+1}: {topic}")
            
            # Debugging session
            logger.info(f"\nGenerating debugging session...")
            debug_events = self.generate_debugging_session_with_llm()
            self.store_conversation(debug_events, f"Debugging-{i+1}")
        
        # Demonstrate features
        self.demonstrate_features()
        
        # Show statistics
        stats = self.agent.get_statistics()
        logger.info("\n" + "="*60)
        logger.info("System Statistics")
        logger.info("="*60)
        logger.info(f"Total events: {stats.get('total_events', 0)}")
        logger.info(f"Episodes: {stats.get('episodes', 0)}")
        logger.info(f"ChromaDB events: {stats.get('chromadb_stats', {}).get('total_events', 0)}")
        
        # Save state
        logger.info("\nSaving memory state...")
        self.agent.save()
        logger.info("Memory state saved successfully!")
        
        logger.info("\n" + "="*60)
        logger.info("LLM Conversation Generation Complete!")
        logger.info("="*60)


def main():
    """Main entry point"""
    generator = LLMConversationGenerator()
    generator.run(num_conversations=2)


if __name__ == "__main__":
    main()
