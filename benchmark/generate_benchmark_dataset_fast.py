"""
Fast benchmark dataset generator with LLM-powered conversation generation.
Generates realistic conversations using an LLM while maintaining proper timestamps and 5W1H structure.
"""

import json
import random
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import LLM interface
from llm.llm_interface import LLMInterface

# Set offline mode to avoid API calls for embeddings
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class ConversationTopic:
    """Defines conversation topics and initial prompts"""
    
    TECHNICAL_TOPICS = [
        {
            "name": "debugging",
            "initial_context": "discussing a technical bug or error",
            "starter_prompts": [
                "I'm encountering an error in my code",
                "My application is crashing when",
                "I need help debugging",
                "There's a strange bug where",
                "The system throws an exception when"
            ]
        },
        {
            "name": "architecture",
            "initial_context": "discussing system design and architecture",
            "starter_prompts": [
                "What's the best architecture for",
                "How should I structure",
                "I'm designing a system that needs to",
                "What design pattern would work for",
                "Should I use microservices or"
            ]
        },
        {
            "name": "performance",
            "initial_context": "discussing performance optimization",
            "starter_prompts": [
                "My application is running slowly",
                "How can I optimize",
                "What's the best way to improve performance",
                "I need to reduce latency in",
                "The system is using too much memory"
            ]
        },
        {
            "name": "implementation",
            "initial_context": "discussing how to implement a feature",
            "starter_prompts": [
                "How do I implement",
                "What's the best way to build",
                "I need to create a feature that",
                "Can you help me code",
                "What approach should I take to implement"
            ]
        },
        {
            "name": "database",
            "initial_context": "discussing database design and queries",
            "starter_prompts": [
                "How should I structure my database for",
                "What's the most efficient query for",
                "Should I use SQL or NoSQL for",
                "I need help with a complex join",
                "My database queries are slow"
            ]
        },
        {
            "name": "security",
            "initial_context": "discussing security best practices",
            "starter_prompts": [
                "How can I secure",
                "What's the best authentication method for",
                "I'm concerned about security vulnerabilities in",
                "How do I protect against",
                "What encryption should I use for"
            ]
        }
    ]
    
    CASUAL_TOPICS = [
        {
            "name": "learning",
            "initial_context": "learning something new",
            "starter_prompts": [
                "Can you explain how",
                "I'm trying to understand",
                "What's the difference between",
                "I'm new to this, can you help me understand",
                "Could you break down the concept of"
            ]
        },
        {
            "name": "planning",
            "initial_context": "planning a project or task",
            "starter_prompts": [
                "I need to plan",
                "What's the best approach for organizing",
                "How should I prioritize",
                "I'm starting a new project about",
                "Can you help me create a roadmap for"
            ]
        },
        {
            "name": "troubleshooting",
            "initial_context": "general problem-solving",
            "starter_prompts": [
                "I'm stuck with",
                "Can't figure out why",
                "Something's not working right with",
                "I need help solving",
                "What could be causing"
            ]
        },
        {
            "name": "best_practices",
            "initial_context": "discussing best practices",
            "starter_prompts": [
                "What are the best practices for",
                "What's the recommended way to",
                "Should I follow any conventions for",
                "What's considered good practice when",
                "Are there any guidelines for"
            ]
        }
    ]
    
    @classmethod
    def get_random_topic(cls, topic_type: str = None) -> Dict:
        """Get a random topic with its context"""
        if topic_type == "technical":
            return random.choice(cls.TECHNICAL_TOPICS)
        elif topic_type == "casual":
            return random.choice(cls.CASUAL_TOPICS)
        else:
            all_topics = cls.TECHNICAL_TOPICS + cls.CASUAL_TOPICS
            return random.choice(all_topics)


class LLMConversationGenerator:
    """Generates realistic conversations using an LLM"""
    
    def __init__(self, use_local_llm: bool = True, web_app_url: str = "http://localhost:5000"):
        """
        Initialize the conversation generator.
        
        Args:
            use_local_llm: If True, uses local LLM. If False, uses web app's chat endpoint.
            web_app_url: URL of the web application for chat processing
        """
        self.use_local_llm = use_local_llm
        self.web_app_url = web_app_url
        
        if use_local_llm:
            self.llm = LLMInterface()
        else:
            # Test connection to web app
            try:
                response = requests.get(f"{web_app_url}/api/memories")
                if response.status_code != 200:
                    print(f"Warning: Web app at {web_app_url} may not be running")
            except:
                print(f"Warning: Cannot connect to web app at {web_app_url}")
    
    def generate_user_query(self, topic: Dict, conversation_history: List[Dict] = None) -> str:
        """Generate a user query based on topic and history"""
        if not conversation_history:
            # Initial query - use a starter prompt
            starter = random.choice(topic["starter_prompts"])
            
            if self.use_local_llm:
                # Expand the starter into a full question
                prompt = f"""Generate a realistic user question about {topic['initial_context']}.
The question should start with: "{starter}"
Complete this into a natural, specific question (1-2 sentences).
Only return the question, nothing else."""
                
                response = self.llm.generate(prompt)
                return response.strip()
            else:
                # Use the starter with some random completion
                completions = [
                    "a large-scale application",
                    "a real-time system",
                    "a distributed service",
                    "handling user data",
                    "processing high volumes",
                    "managing state",
                    "handling concurrent requests",
                    "dealing with async operations"
                ]
                return f"{starter} {random.choice(completions)}?"
        else:
            # Follow-up query based on conversation
            last_response = conversation_history[-1]["what"]
            
            if self.use_local_llm:
                prompt = f"""Based on this assistant response:
"{last_response}"

Generate a natural follow-up question that a user might ask. 
The question should be related but explore a different aspect or ask for clarification.
Keep it concise (1-2 sentences). Only return the question."""
                
                response = self.llm.generate(prompt)
                return response.strip()
            else:
                # Generate a simple follow-up
                follow_ups = [
                    "What about edge cases?",
                    "How does that handle errors?",
                    "What's the performance impact?",
                    "Are there any alternatives?",
                    "Can you give an example?",
                    "What are the trade-offs?",
                    "How would that scale?",
                    "What about security concerns?"
                ]
                return random.choice(follow_ups)
    
    def generate_assistant_response(self, user_query: str, topic: Dict, conversation_history: List[Dict] = None) -> str:
        """Generate an assistant response to the user query"""
        if self.use_local_llm:
            # Build context from conversation history
            context = ""
            if conversation_history:
                context = "Previous conversation:\n"
                for event in conversation_history[-4:]:  # Last 2 exchanges
                    context += f"{event['who']}: {event['what']}\n"
                context += "\n"
            
            prompt = f"""You are a helpful AI assistant discussing {topic['initial_context']}.
{context}
User: {user_query}

Provide a helpful, informative response (2-4 sentences).
Be specific and technical when appropriate, but keep it concise.
Only return your response, nothing else."""
            
            response = self.llm.generate(prompt)
            return response.strip()
        else:
            # Send to web app's chat endpoint
            try:
                response = requests.post(
                    f"{self.web_app_url}/api/chat",
                    json={"message": user_query},
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "I understand your question. Let me help you with that.")
                else:
                    return "I understand your question. Let me help you with that."
            except Exception as e:
                print(f"Error calling web app: {e}")
                return "I understand your question. Let me help you with that."
    
    def generate_conversation(
        self,
        topic: Dict,
        base_timestamp: datetime,
        num_exchanges: int = None,
        location: str = None
    ) -> List[Dict]:
        """
        Generate a complete conversation with realistic timestamps.
        
        Args:
            topic: Topic dictionary with name and context
            base_timestamp: Starting timestamp for the conversation
            num_exchanges: Number of back-and-forth exchanges
            location: Where the conversation takes place
            
        Returns:
            List of events in 5W1H format
        """
        if num_exchanges is None:
            num_exchanges = random.randint(2, 5)
        
        if location is None:
            location = random.choice(["web_chat", "terminal", "IDE", "browser", "slack", "teams"])
        
        events = []
        current_time = base_timestamp
        conversation_history = []
        
        for i in range(num_exchanges):
            # Generate user query
            user_query = self.generate_user_query(topic, conversation_history)
            
            # Add realistic delay for user typing (30 seconds to 2 minutes)
            current_time += timedelta(seconds=random.randint(30, 120))
            
            user_event = {
                "who": "User",
                "what": user_query,
                "when": current_time.isoformat() + "Z",
                "where": location,
                "why": f"Asking about {topic['name']}",
                "how": "typed_message",
                "event_type": "user_message"
            }
            events.append(user_event)
            conversation_history.append(user_event)
            
            # Generate assistant response with thinking time (3-15 seconds)
            current_time += timedelta(seconds=random.randint(3, 15))
            
            assistant_response = self.generate_assistant_response(user_query, topic, conversation_history)
            
            assistant_event = {
                "who": "Assistant",
                "what": assistant_response,
                "when": current_time.isoformat() + "Z",
                "where": "system",
                "why": f"Responding about {topic['name']}",
                "how": "generated_response",
                "event_type": "assistant_response"
            }
            events.append(assistant_event)
            conversation_history.append(assistant_event)
            
            # Add gap between exchanges (30 seconds to 3 minutes)
            if i < num_exchanges - 1:
                current_time += timedelta(seconds=random.randint(30, 180))
        
        return events


class FastBenchmarkDatasetGenerator:
    """Fast dataset generator with LLM-powered conversations"""
    
    def __init__(
        self,
        output_dir: str = "./benchmark_datasets",
        use_local_llm: bool = True,
        web_app_url: str = "http://localhost:5000"
    ):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save datasets
            use_local_llm: If True, uses local LLM. If False, uses web app.
            web_app_url: URL of the web application
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.conversation_generator = LLMConversationGenerator(use_local_llm, web_app_url)
        
    def generate_dataset(
        self,
        num_conversations: int = 100,
        time_span_days: int = 30,
        dataset_name: str = None,
        num_threads: int = 1,
        technical_ratio: float = 0.6,
        exchanges_per_conversation: Tuple[int, int] = (2, 5)
    ) -> Dict:
        """
        Generate a benchmark dataset with LLM-powered conversations.
        
        Args:
            num_conversations: Number of conversations to generate
            time_span_days: Time span for conversations (days before now)
            dataset_name: Name for the dataset file
            num_threads: Number of parallel threads (limited effectiveness with LLM)
            technical_ratio: Ratio of technical vs casual conversations
            exchanges_per_conversation: Min and max exchanges per conversation
            
        Returns:
            Generated dataset dictionary
        """
        if dataset_name is None:
            dataset_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"LLM-POWERED DATASET GENERATION")
        print(f"{'='*60}")
        print(f"Dataset name: {dataset_name}")
        print(f"Conversations: {num_conversations}")
        print(f"Time span: {time_span_days} days")
        print(f"Technical ratio: {technical_ratio:.0%}")
        print(f"Exchanges per conversation: {exchanges_per_conversation[0]}-{exchanges_per_conversation[1]}")
        print(f"LLM Mode: {'Local' if self.conversation_generator.use_local_llm else 'Web App'}")
        print(f"Parallel threads: {num_threads}")
        print(f"{'='*60}\n")
        
        # Calculate time range (ending now, going back time_span_days)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_span_days)
        
        all_events = []
        conversation_metadata = []
        
        # Thread-safe lock for shared data
        lock = threading.Lock()
        
        def generate_single_conversation(conv_idx: int) -> Tuple[List[Dict], Dict]:
            """Generate a single conversation with LLM"""
            try:
                # Decide topic type based on ratio
                if random.random() < technical_ratio:
                    topic_type = "technical"
                else:
                    topic_type = "casual"
                
                # Get random topic
                topic = ConversationTopic.get_random_topic(topic_type)
                
                # Generate random timestamp within the time range
                # Bias towards more recent times
                days_ago = random.expovariate(1 / (time_span_days / 3))
                days_ago = min(days_ago, time_span_days - 0.1)
                
                base_time = end_time - timedelta(days=days_ago)
                base_time += timedelta(hours=random.uniform(-12, 12))
                
                # Generate conversation with LLM
                num_exchanges = random.randint(*exchanges_per_conversation)
                events = self.conversation_generator.generate_conversation(
                    topic=topic,
                    base_timestamp=base_time,
                    num_exchanges=num_exchanges
                )
                
                # Create metadata
                metadata = {
                    "conversation_id": f"conv_{conv_idx:04d}",
                    "topic": topic["name"],
                    "topic_type": topic_type,
                    "num_exchanges": len(events) // 2,
                    "start_time": events[0]["when"] if events else None,
                    "end_time": events[-1]["when"] if events else None,
                }
                
                return events, metadata
                
            except Exception as e:
                print(f"Error generating conversation {conv_idx}: {e}")
                return [], {}
        
        # Generate conversations (with limited parallelism for LLM calls)
        start_generation = time.time()
        
        if num_threads > 1 and self.conversation_generator.use_local_llm:
            # Parallel generation for local LLM
            with ThreadPoolExecutor(max_workers=min(num_threads, 4)) as executor:
                futures = []
                for i in range(num_conversations):
                    future = executor.submit(generate_single_conversation, i)
                    futures.append(future)
                
                # Process results with progress indicator
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        events, metadata = future.result(timeout=60)
                        
                        if events:  # Only add if generation succeeded
                            with lock:
                                all_events.extend(events)
                                conversation_metadata.append(metadata)
                        
                        # Progress update
                        if i % 5 == 0 or i == num_conversations:
                            elapsed = time.time() - start_generation
                            rate = i / elapsed if elapsed > 0 else 0
                            remaining = (num_conversations - i) / rate if rate > 0 else 0
                            print(f"Progress: {i}/{num_conversations} conversations "
                                  f"({i*100/num_conversations:.1f}%) - "
                                  f"Rate: {rate:.2f} conv/s - "
                                  f"ETA: {remaining:.0f}s")
                    
                    except Exception as e:
                        print(f"Error processing conversation {i}: {e}")
        else:
            # Sequential generation (better for web app endpoint)
            for i in range(num_conversations):
                events, metadata = generate_single_conversation(i)
                
                if events:
                    all_events.extend(events)
                    conversation_metadata.append(metadata)
                
                # Progress update
                if (i + 1) % 5 == 0 or (i + 1) == num_conversations:
                    elapsed = time.time() - start_generation
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (num_conversations - i - 1) / rate if rate > 0 else 0
                    print(f"Progress: {i + 1}/{num_conversations} conversations "
                          f"({(i + 1)*100/num_conversations:.1f}%) - "
                          f"Rate: {rate:.2f} conv/s - "
                          f"ETA: {remaining:.0f}s")
        
        # Sort events by timestamp
        all_events.sort(key=lambda x: x["when"])
        
        # Calculate statistics
        generation_time = time.time() - start_generation
        
        stats = {
            "num_conversations": len(conversation_metadata),
            "num_events": len(all_events),
            "time_span_days": time_span_days,
            "actual_start": all_events[0]["when"] if all_events else None,
            "actual_end": all_events[-1]["when"] if all_events else None,
            "generation_time_seconds": generation_time,
            "events_per_second": len(all_events) / generation_time if generation_time > 0 else 0,
            "conversations_per_second": len(conversation_metadata) / generation_time if generation_time > 0 else 0,
            "technical_conversations": sum(1 for m in conversation_metadata if m.get("topic_type") == "technical"),
            "casual_conversations": sum(1 for m in conversation_metadata if m.get("topic_type") == "casual"),
            "unique_topics": len(set(m.get("topic", "") for m in conversation_metadata))
        }
        
        # Create dataset
        dataset = {
            "metadata": {
                "name": dataset_name,
                "created_at": datetime.now().isoformat(),
                "generator": "LLM-powered",
                "llm_mode": "local" if self.conversation_generator.use_local_llm else "web_app",
                "stats": stats
            },
            "events": all_events,
            "conversations": conversation_metadata
        }
        
        # Save dataset
        output_path = self.output_dir / f"{dataset_name}.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total events: {len(all_events)}")
        print(f"Successful conversations: {len(conversation_metadata)}")
        print(f"Technical: {stats['technical_conversations']}")
        print(f"Casual: {stats['casual_conversations']}")
        print(f"Unique topics: {stats['unique_topics']}")
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Rate: {stats['conversations_per_second']:.2f} conv/s")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        return dataset


def main():
    """Main function with interactive menu"""
    print("\n" + "="*60)
    print("LLM-POWERED BENCHMARK DATASET GENERATOR")
    print("="*60)
    
    # Ask for LLM mode
    print("\nSelect LLM mode:")
    print("1. Local LLM")
    print("2. Web App Chat Endpoint")
    
    llm_choice = input("\nEnter choice (1-2): ").strip()
    use_local_llm = llm_choice != "2"
    
    # Get web app URL if using web app mode
    web_app_url = "http://localhost:5000"
    if not use_local_llm:
        custom_url = input("Enter web app URL (press Enter for http://localhost:5000): ").strip()
        if custom_url:
            web_app_url = custom_url
    
    # Get dataset size
    print("\nSelect dataset size:")
    print("1. Tiny (10 conversations) - for testing")
    print("2. Small (25 conversations)")
    print("3. Medium (50 conversations)")
    print("4. Large (100 conversations)")
    print("5. Custom")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    size_map = {
        "1": (10, "tiny"),
        "2": (25, "small"),
        "3": (50, "medium"),
        "4": (100, "large"),
        "5": (None, "custom")
    }
    
    num_conversations, size_name = size_map.get(choice, (10, "tiny"))
    
    if num_conversations is None:
        num_conversations = int(input("Enter number of conversations: "))
        size_name = "custom"
    
    # Get time span
    print("\nSelect time span:")
    print("1. Last week (7 days)")
    print("2. Last month (30 days)")
    print("3. Last quarter (90 days)")
    print("4. Custom")
    
    time_choice = input("\nEnter choice (1-4): ").strip()
    
    time_map = {
        "1": 7,
        "2": 30,
        "3": 90,
        "4": None
    }
    
    time_span = time_map.get(time_choice, 30)
    
    if time_span is None:
        time_span = int(input("Enter time span in days: "))
    
    # Get conversation length
    print("\nSelect conversation length:")
    print("1. Short (2-3 exchanges)")
    print("2. Medium (3-5 exchanges)")
    print("3. Long (4-7 exchanges)")
    print("4. Custom")
    
    length_choice = input("\nEnter choice (1-4): ").strip()
    
    length_map = {
        "1": (2, 3),
        "2": (3, 5),
        "3": (4, 7),
        "4": None
    }
    
    exchanges = length_map.get(length_choice, (2, 5))
    
    if exchanges is None:
        min_ex = int(input("Enter minimum exchanges per conversation: "))
        max_ex = int(input("Enter maximum exchanges per conversation: "))
        exchanges = (min_ex, max_ex)
    
    # Thread count (limited for LLM calls)
    if use_local_llm:
        print("\nSelect parallel threads (limited effectiveness with LLM):")
        print("1. Single thread (safest)")
        print("2. 2 threads")
        print("3. 4 threads (may have issues with some LLMs)")
        
        thread_choice = input("\nEnter choice (1-3): ").strip()
        
        thread_map = {
            "1": 1,
            "2": 2,
            "3": 4
        }
        
        num_threads = thread_map.get(thread_choice, 1)
    else:
        num_threads = 1  # Always use single thread for web app
        print("\nUsing single thread for web app mode")
    
    # Generate dataset
    generator = FastBenchmarkDatasetGenerator(
        use_local_llm=use_local_llm,
        web_app_url=web_app_url
    )
    
    dataset_name = f"dataset_{size_name}_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\nStarting generation... This may take a while with LLM generation.")
    print("Each conversation requires multiple LLM calls.\n")
    
    dataset = generator.generate_dataset(
        num_conversations=num_conversations,
        time_span_days=time_span,
        dataset_name=dataset_name,
        num_threads=num_threads,
        exchanges_per_conversation=exchanges
    )
    
    print("\nDataset generation complete!")
    print(f"Generated {len(dataset['events'])} events from {len(dataset['conversations'])} conversations")
    
    # Show sample of generated content
    if dataset['events']:
        print("\nSample generated conversation:")
        print("-" * 40)
        sample_events = dataset['events'][:4]  # First 2 exchanges
        for event in sample_events:
            print(f"{event['who']}: {event['what'][:100]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()
