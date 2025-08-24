#!/usr/bin/env python
"""
Conversation Generator for Memory System Testing

Generates realistic multi-turn conversations on various topics to populate
the memory system with meaningful data for analysis.
"""

import json
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add imports for memory system
from agent.memory_agent import MemoryAgent
from models.event import EventType
from llm.llm_interface import LLMInterface
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTopic:
    """Represents a conversation topic with related subtopics"""
    name: str
    subtopics: List[str]
    personas: List[Dict[str, str]]  # Different personas that might discuss this
    context_snippets: List[str]  # Context to make conversations richer

# Predefined conversation topics with rich context
CONVERSATION_TOPICS = [
    ConversationTopic(
        name="Software Development",
        subtopics=[
            "debugging a memory leak", "code review best practices", 
            "choosing a database", "API design patterns", "testing strategies",
            "refactoring legacy code", "performance optimization"
        ],
        personas=[
            {"role": "junior developer", "style": "asking questions, seeking clarification"},
            {"role": "senior engineer", "style": "providing detailed explanations, sharing experience"},
            {"role": "tech lead", "style": "focusing on architecture, team coordination"},
        ],
        context_snippets=[
            "We've been seeing increased memory usage in production",
            "The codebase is 5 years old and needs modernization",
            "Our API response times have degraded by 30%",
            "We need to scale to handle 10x more users"
        ]
    ),
    ConversationTopic(
        name="Machine Learning",
        subtopics=[
            "model overfitting issues", "feature engineering", "hyperparameter tuning",
            "data preprocessing", "model deployment", "A/B testing ML models",
            "handling imbalanced datasets"
        ],
        personas=[
            {"role": "data scientist", "style": "analytical, focuses on metrics"},
            {"role": "ML engineer", "style": "practical, deployment-focused"},
            {"role": "researcher", "style": "theoretical, citing papers"},
        ],
        context_snippets=[
            "Our model accuracy is 92% on training but only 71% on validation",
            "We have 1M samples but only 1000 positive cases",
            "The model needs to run in real-time with <100ms latency",
            "We're comparing transformer vs LSTM architectures"
        ]
    ),
    ConversationTopic(
        name="Project Management",
        subtopics=[
            "sprint planning", "deadline negotiations", "resource allocation",
            "stakeholder communication", "risk assessment", "team motivation",
            "scope creep handling"
        ],
        personas=[
            {"role": "project manager", "style": "organized, deadline-focused"},
            {"role": "product owner", "style": "feature-focused, customer-oriented"},
            {"role": "scrum master", "style": "process-focused, team advocate"},
        ],
        context_snippets=[
            "The client wants to add three new features this sprint",
            "We're two weeks behind schedule",
            "The team is showing signs of burnout",
            "Budget has been cut by 20%"
        ]
    ),
    ConversationTopic(
        name="System Architecture",
        subtopics=[
            "microservices vs monolith", "event-driven architecture", "caching strategies",
            "database sharding", "load balancing", "service mesh implementation",
            "disaster recovery planning"
        ],
        personas=[
            {"role": "solutions architect", "style": "big picture thinking, trade-offs"},
            {"role": "DevOps engineer", "style": "operational concerns, monitoring"},
            {"role": "security architect", "style": "risk-focused, compliance-aware"},
        ],
        context_snippets=[
            "We're experiencing 10K requests per second at peak",
            "Data consistency is critical for financial transactions",
            "We need 99.99% uptime SLA",
            "Compliance requires data residency in EU"
        ]
    ),
    ConversationTopic(
        name="Customer Support",
        subtopics=[
            "bug report triage", "feature requests", "user onboarding issues",
            "performance complaints", "billing problems", "documentation gaps",
            "integration difficulties"
        ],
        personas=[
            {"role": "support agent", "style": "helpful, patient, solution-oriented"},
            {"role": "customer", "style": "frustrated or confused, seeking help"},
            {"role": "support manager", "style": "process improvement, metrics-driven"},
        ],
        context_snippets=[
            "Multiple users reporting login issues since yesterday",
            "Customer can't figure out how to export their data",
            "Enterprise client threatening to cancel subscription",
            "New feature causing confusion among users"
        ]
    )
]

class ConversationGenerator:
    """Generates realistic conversations and stores them in memory system"""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the conversation generator
        
        Args:
            use_llm: Whether to use LLM for generation (True) or templates (False)
        """
        self.memory_agent = MemoryAgent()
        self.use_llm = use_llm
        
        if use_llm:
            config = get_config()
            self.llm = LLMInterface(config.llm)
            # Test LLM connection
            try:
                self.llm.test_connection()
                logger.info("LLM connected successfully")
            except:
                logger.warning("LLM not available, falling back to template mode")
                self.use_llm = False
                self.llm = None
        else:
            self.llm = None
        
        self.conversation_count = 0
        self.turn_count = 0
        self.current_topic = None
        self.topic_history = []
        self.stats = {
            "total_conversations": 0,
            "total_turns": 0,
            "topics_covered": set(),
            "recurring_topics": 0,
            "memory_blocks_created": 0
        }
    
    def generate_conversation_turn(
        self, 
        topic: ConversationTopic,
        subtopic: str,
        persona: Dict[str, str],
        context: str,
        previous_turns: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        """
        Generate a single conversation turn (user message + assistant response)
        
        Returns:
            (user_message, assistant_response)
        """
        if self.use_llm and self.llm:
            # Use LLM to generate realistic conversation
            conversation_history = "\n".join([
                f"User: {u}\nAssistant: {a}" 
                for u, a in previous_turns[-3:]  # Last 3 turns for context
            ])
            
            prompt = f"""Generate the next turn in a conversation about {topic.name} - {subtopic}.
Context: {context}
Persona: {persona['role']} with style: {persona['style']}

Previous conversation:
{conversation_history if conversation_history else "This is the start of the conversation."}

Generate a realistic next exchange where the user (acting as {persona['role']}) asks something or makes a statement,
and the assistant provides a helpful response. Make it natural and specific to the context.

Format your response as:
USER: [user message]
ASSISTANT: [assistant response]"""
            
            try:
                response = self.llm.generate(prompt, max_tokens=200)
                # Parse the response
                lines = response.strip().split('\n')
                user_msg = ""
                assistant_msg = ""
                
                for line in lines:
                    if line.startswith("USER:"):
                        user_msg = line.replace("USER:", "").strip()
                    elif line.startswith("ASSISTANT:"):
                        assistant_msg = line.replace("ASSISTANT:", "").strip()
                
                if user_msg and assistant_msg:
                    return user_msg, assistant_msg
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, using template")
        
        # Fallback to template-based generation
        return self._generate_template_turn(topic, subtopic, persona, context, previous_turns)
    
    def _generate_template_turn(
        self,
        topic: ConversationTopic,
        subtopic: str,
        persona: Dict[str, str],
        context: str,
        previous_turns: List[Tuple[str, str]]
    ) -> Tuple[str, str]:
        """Template-based fallback for conversation generation"""
        
        # Template patterns based on conversation position
        if len(previous_turns) == 0:
            # Opening patterns
            user_patterns = [
                f"I'm working on {subtopic} and {context}. Any suggestions?",
                f"We need help with {subtopic}. {context}",
                f"Question about {subtopic}: {context}. What do you think?",
                f"I'm a {persona['role']} dealing with {subtopic}. {context}"
            ]
            assistant_patterns = [
                f"I can help with {subtopic}. Let me analyze the situation...",
                f"Based on your context about {subtopic}, here's what I recommend...",
                f"For {subtopic} issues, especially given that {context}, you should consider...",
                f"As a {persona['role']}, you'll want to focus on these aspects of {subtopic}..."
            ]
        elif len(previous_turns) < 3:
            # Middle patterns
            user_patterns = [
                f"That makes sense. What about the {random.choice(['performance', 'security', 'scalability', 'maintenance'])} implications?",
                f"How would this work with our current setup where {context}?",
                f"I see. Could you elaborate on the {random.choice(['technical', 'practical', 'implementation'])} details?",
                f"What are the trade-offs we should consider?"
            ]
            assistant_patterns = [
                f"Good question about implications. For {subtopic}, you need to consider...",
                f"Given that {context}, the approach would need to be adjusted...",
                f"Let me break down the details for {subtopic}...",
                f"The main trade-offs in this {topic.name} scenario are..."
            ]
        else:
            # Closing patterns
            user_patterns = [
                f"Thanks, this helps clarify our approach to {subtopic}.",
                f"I'll implement this solution. One last question about {subtopic}...",
                f"Great insights. How do we measure success for this?",
                f"This has been helpful for understanding {subtopic} better."
            ]
            assistant_patterns = [
                f"You're welcome! Remember to monitor the results as you implement...",
                f"For {subtopic}, success metrics would include...",
                f"Glad I could help. Don't hesitate to ask if you need more guidance on {topic.name}.",
                f"To summarize the key points about {subtopic}..."
            ]
        
        return random.choice(user_patterns), random.choice(assistant_patterns)
    
    def generate_conversation(
        self,
        topic: Optional[ConversationTopic] = None,
        min_turns: int = 3,
        max_turns: int = 7
    ) -> List[Dict[str, any]]:
        """
        Generate a complete conversation on a topic
        
        Args:
            topic: Topic to use (random if None)
            min_turns: Minimum conversation turns
            max_turns: Maximum conversation turns
            
        Returns:
            List of memory events created
        """
        if topic is None:
            topic = random.choice(CONVERSATION_TOPICS)
        
        subtopic = random.choice(topic.subtopics)
        persona = random.choice(topic.personas)
        context = random.choice(topic.context_snippets)
        
        num_turns = random.randint(min_turns, max_turns)
        conversation_turns = []
        memory_events = []
        
        # Track this topic
        self.current_topic = topic.name
        if topic.name in self.topic_history:
            self.stats["recurring_topics"] += 1
        self.topic_history.append(topic.name)
        self.stats["topics_covered"].add(topic.name)
        
        logger.info(f"Generating conversation about {topic.name} - {subtopic} ({num_turns} turns)")
        
        # Start a new episode for this conversation
        self.memory_agent.start_episode(goal=f"Discussing {topic.name}: {subtopic}")
        
        for turn_idx in range(num_turns):
            # Generate conversation turn
            user_msg, assistant_msg = self.generate_conversation_turn(
                topic, subtopic, persona, context, conversation_turns
            )
            conversation_turns.append((user_msg, assistant_msg))
            
            # Store user message as observation
            success, message, event = self.memory_agent.observe(
                what=user_msg,
                who=f"User ({persona['role']})",
                where=f"Conversation about {topic.name}",
                why=f"Discussing {subtopic}",
                how="Text message"
            )
            if success and event:
                memory_events.append(event.to_dict())
                logger.debug(f"Stored user message: {message}")
            
            # Store assistant response as action
            success, message, event = self.memory_agent.act(
                what=assistant_msg,
                who="Assistant",
                where=f"Conversation about {topic.name}",
                why=f"Responding about {subtopic}",
                how="Generated response"
            )
            if success and event:
                memory_events.append(event.to_dict())
                logger.debug(f"Stored assistant response: {message}")
            
            self.turn_count += 1
            self.stats["total_turns"] += 1
            
            # Small delay to simulate realistic conversation timing
            time.sleep(0.1)
        
        # End episode
        self.memory_agent.end_episode()
        
        self.conversation_count += 1
        self.stats["total_conversations"] += 1
        
        return memory_events
    
    def generate_conversation_dataset(
        self,
        num_conversations: int = 20,
        topic_switch_frequency: int = 4,
        recurring_topic_probability: float = 0.3
    ) -> Dict[str, any]:
        """
        Generate a dataset of conversations with topic management
        
        Args:
            num_conversations: Total number of conversations to generate
            topic_switch_frequency: Switch topics every N conversations
            recurring_topic_probability: Chance to revisit an old topic
            
        Returns:
            Dataset with all conversations and statistics
        """
        dataset = {
            "conversations": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_conversations": num_conversations,
                "use_llm": self.use_llm
            }
        }
        
        current_topic = None
        topic_conversation_count = 0
        
        logger.info(f"Starting generation of {num_conversations} conversations")
        
        for conv_idx in range(num_conversations):
            # Decide whether to switch topics
            should_switch = (topic_conversation_count >= topic_switch_frequency or 
                           current_topic is None)
            
            if should_switch:
                # Potentially revisit an old topic
                if (self.topic_history and 
                    random.random() < recurring_topic_probability):
                    # Revisit an old topic
                    old_topic_name = random.choice(self.topic_history[-10:])  # Recent topics
                    current_topic = next(
                        t for t in CONVERSATION_TOPICS 
                        if t.name == old_topic_name
                    )
                    logger.info(f"Revisiting topic: {current_topic.name}")
                else:
                    # Pick a new topic
                    current_topic = random.choice(CONVERSATION_TOPICS)
                    logger.info(f"Switching to new topic: {current_topic.name}")
                
                topic_conversation_count = 0
            
            # Generate conversation on current topic
            conversation_events = self.generate_conversation(
                topic=current_topic,
                min_turns=3,
                max_turns=7
            )
            
            dataset["conversations"].append({
                "index": conv_idx,
                "topic": current_topic.name,
                "num_turns": len(conversation_events) // 2,  # Each turn has 2 events
                "events": conversation_events
            })
            
            topic_conversation_count += 1
            
            # Progress update
            if (conv_idx + 1) % 5 == 0:
                logger.info(f"Progress: {conv_idx + 1}/{num_conversations} conversations generated")
                # Save memory state periodically
                self.memory_agent.save()
        
        # Final save
        self.memory_agent.save()
        
        # Get memory statistics
        memory_stats = self.memory_agent.memory_store.get_statistics()
        self.stats["memory_blocks_created"] = memory_stats.get("blocks", {}).get("total_blocks", 0)
        
        # Add statistics to dataset
        dataset["statistics"] = {
            **self.stats,
            "topics_covered": list(self.stats["topics_covered"]),
            "memory_stats": memory_stats
        }
        
        # Save dataset to file
        dataset_path = Path("generated_conversations.json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        logger.info(f"Dataset saved to {dataset_path}")
        
        return dataset
    
    def analyze_memory_blocks(self) -> Dict[str, any]:
        """Analyze the memory blocks created from conversations"""
        analysis = {
            "block_analysis": [],
            "topic_distribution": {},
            "link_type_distribution": {},
            "coherence_scores": [],
            "salience_scores": []
        }
        
        # Get all memory blocks
        blocks = self.memory_agent.memory_store.block_manager.blocks
        
        for block_id, block in blocks.items():
            # Analyze block content
            block_info = {
                "block_id": block_id,
                "num_events": len(block.events),
                "coherence_score": block.coherence_score,
                "salience": block.salience,
                "access_count": block.access_count,
                "link_types": {}
            }
            
            # Count link types
            for link in block.links:
                link_type = link.link_type.value
                block_info["link_types"][link_type] = block_info["link_types"].get(link_type, 0) + 1
                analysis["link_type_distribution"][link_type] = \
                    analysis["link_type_distribution"].get(link_type, 0) + 1
            
            # Extract topics from events
            for event in block.events:
                if "Conversation about" in event.five_w1h.where:
                    topic = event.five_w1h.where.split("Conversation about")[1].strip()
                    analysis["topic_distribution"][topic] = \
                        analysis["topic_distribution"].get(topic, 0) + 1
            
            analysis["block_analysis"].append(block_info)
            analysis["coherence_scores"].append(block.coherence_score)
            analysis["salience_scores"].append(block.salience)
        
        # Calculate statistics
        if analysis["coherence_scores"]:
            analysis["avg_coherence"] = sum(analysis["coherence_scores"]) / len(analysis["coherence_scores"])
            analysis["max_coherence"] = max(analysis["coherence_scores"])
            analysis["min_coherence"] = min(analysis["coherence_scores"])
        
        if analysis["salience_scores"]:
            analysis["avg_salience"] = sum(analysis["salience_scores"]) / len(analysis["salience_scores"])
            analysis["max_salience"] = max(analysis["salience_scores"])
            analysis["min_salience"] = min(analysis["salience_scores"])
        
        # Save analysis
        analysis_path = Path("memory_block_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {analysis_path}")
        
        return analysis

def main():
    """Main entry point for conversation generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate conversations for memory system testing")
    parser.add_argument(
        "--num-conversations", 
        type=int, 
        default=20,
        help="Number of conversations to generate (default: 20)"
    )
    parser.add_argument(
        "--topic-switch", 
        type=int, 
        default=4,
        help="Switch topics every N conversations (default: 4)"
    )
    parser.add_argument(
        "--recurring-probability", 
        type=float, 
        default=0.3,
        help="Probability of revisiting old topics (default: 0.3)"
    )
    parser.add_argument(
        "--use-llm", 
        action="store_true",
        help="Use LLM for generation (requires LLM server running)"
    )
    parser.add_argument(
        "--analyze-only", 
        action="store_true",
        help="Only analyze existing memory blocks without generating new conversations"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ConversationGenerator(use_llm=args.use_llm)
    
    if args.analyze_only:
        # Just analyze existing memory blocks
        logger.info("Analyzing existing memory blocks...")
        analysis = generator.analyze_memory_blocks()
        
        # Print summary
        print("\n" + "="*60)
        print("MEMORY BLOCK ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total blocks: {len(analysis['block_analysis'])}")
        print(f"Average coherence: {analysis.get('avg_coherence', 0):.3f}")
        print(f"Average salience: {analysis.get('avg_salience', 0):.3f}")
        print(f"\nTopic distribution:")
        for topic, count in analysis['topic_distribution'].items():
            print(f"  {topic}: {count} events")
        print(f"\nLink type distribution:")
        for link_type, count in analysis['link_type_distribution'].items():
            print(f"  {link_type}: {count} links")
    else:
        # Generate conversations
        logger.info(f"Generating {args.num_conversations} conversations...")
        logger.info(f"Using LLM: {args.use_llm}")
        
        dataset = generator.generate_conversation_dataset(
            num_conversations=args.num_conversations,
            topic_switch_frequency=args.topic_switch,
            recurring_topic_probability=args.recurring_probability
        )
        
        # Print summary
        print("\n" + "="*60)
        print("CONVERSATION GENERATION COMPLETE")
        print("="*60)
        print(f"Total conversations: {dataset['statistics']['total_conversations']}")
        print(f"Total turns: {dataset['statistics']['total_turns']}")
        print(f"Topics covered: {', '.join(dataset['statistics']['topics_covered'])}")
        print(f"Recurring topics: {dataset['statistics']['recurring_topics']}")
        print(f"Memory blocks created: {dataset['statistics']['memory_blocks_created']}")
        
        # Now analyze the created memory blocks
        print("\nAnalyzing created memory blocks...")
        analysis = generator.analyze_memory_blocks()
        
        print(f"\nMemory block statistics:")
        print(f"  Average coherence: {analysis.get('avg_coherence', 0):.3f}")
        print(f"  Average salience: {analysis.get('avg_salience', 0):.3f}")
        print(f"  Total blocks: {len(analysis['block_analysis'])}")

if __name__ == "__main__":
    main()