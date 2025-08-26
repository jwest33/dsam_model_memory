"""
Generate sample conversations through the web API
This simulates real users having conversations with the assistant
Enhanced to test dual-space encoding and clustering capabilities
"""

import requests
import json
import time
import random
from typing import List, Dict, Optional
from datetime import datetime
import os
import sys
import warnings

# Set offline mode to avoid HuggingFace rate limits
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress sklearn deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from llm.llm_interface import LLMInterface
from config import get_config

class ConversationGenerator:
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.config = get_config()
        self.llm = LLMInterface(self.config.llm)
        self.session_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def generate_user_message(self, context: str, topic: str, message_style: str = "normal") -> str:
        """Generate a realistic user message using LLM"""
        
        style_prompts = {
            "normal": "natural and conversational",
            "technical": "technically detailed and specific",
            "abstract": "abstract, focusing on concepts and theory",
            "concrete": "concrete with specific examples and details",
            "exploratory": "exploratory, asking about relationships and connections"
        }
        
        prompt = f"""Generate a realistic user message for a conversation about {topic}.
Context: {context}

The message should be {style_prompts.get(message_style, 'natural')}.
Keep it concise (1-2 sentences).

User message:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=50
            )
            return response.strip()
        except Exception as e:
            # Fallback messages if LLM fails
            fallback_messages = {
                "normal": f"Can you tell me more about {topic}?",
                "technical": f"What are the technical details of {topic}?",
                "abstract": f"What's the underlying concept behind {topic}?",
                "concrete": f"Can you give me a specific example of {topic}?",
                "exploratory": f"How does {topic} relate to other concepts?"
            }
            return fallback_messages.get(message_style, f"Tell me about {topic}.")
    
    def send_chat_message(self, message: str) -> Dict:
        """Send a message to the chat API and get response"""
        try:
            response = requests.post(
                f"{self.api_url}/api/chat",
                json={"message": message},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"  [Timeout sending message]")
            return {"response": "Request timed out", "memories_used": 0}
        except Exception as e:
            print(f"  [Error: {e}]")
            return {"response": "Error occurred", "memories_used": 0}
    
    def generate_conversation(self, topic: str, num_exchanges: int = 3, style_sequence: Optional[List[str]] = None):
        """Generate a full conversation on a topic"""
        print(f"\n{'='*60}")
        print(f"Conversation: {topic}")
        print(f"{'='*60}")
        
        context = f"Starting a new conversation about {topic}"
        
        # Default style sequence if not provided
        if not style_sequence:
            style_sequence = ["normal"] * num_exchanges
        
        for i in range(num_exchanges):
            style = style_sequence[i % len(style_sequence)]
            
            # Generate and send user message
            user_message = self.generate_user_message(context, topic, style)
            print(f"\n[Exchange {i+1}/{num_exchanges}] Style: {style}")
            print(f"User: {user_message}")
            
            # Send to API and get assistant response
            result = self.send_chat_message(user_message)
            assistant_response = result.get("response", "No response")
            memories_used = result.get("memories_used", 0)
            
            print(f"Assistant: {assistant_response[:200]}{'...' if len(assistant_response) > 200 else ''}")
            if memories_used > 0:
                print(f"Used {memories_used} memories")
            
            # Update context for next exchange
            context = f"Previous exchange:\nUser: {user_message}\nAssistant: {assistant_response[:100]}"
            
            # Small delay to simulate natural conversation pace
            time.sleep(1)
    
    def test_dual_space_retrieval(self):
        """Test conversations that should trigger different space preferences"""
        print("\n" + "="*60)
        print("TESTING DUAL-SPACE RETRIEVAL")
        print("="*60)
        
        # First, seed some memories with concrete technical content
        concrete_topics = [
            ("Python error handling", ["technical", "concrete"]),
            ("Database indexing strategies", ["technical", "concrete"]),
            ("REST API endpoints", ["technical", "concrete"])
        ]
        
        print("\n1. Seeding concrete technical memories...")
        for topic, styles in concrete_topics:
            self.generate_conversation(topic, num_exchanges=2, style_sequence=styles)
            time.sleep(1)
        
        # Now seed abstract conceptual content
        abstract_topics = [
            ("Software design principles", ["abstract", "exploratory"]),
            ("System architecture patterns", ["abstract", "exploratory"]),
            ("Code optimization philosophy", ["abstract", "exploratory"])
        ]
        
        print("\n2. Seeding abstract conceptual memories...")
        for topic, styles in abstract_topics:
            self.generate_conversation(topic, num_exchanges=2, style_sequence=styles)
            time.sleep(1)
        
        # Test retrieval with concrete queries (should favor Euclidean space)
        print("\n3. Testing concrete queries (should use Euclidean space)...")
        concrete_test_messages = [
            "Show me Python code for handling exceptions",
            "What's the SQL syntax for creating an index?",
            "Give me an example of a GET endpoint"
        ]
        
        for msg in concrete_test_messages:
            print(f"\n  Query: {msg}")
            result = self.send_chat_message(msg)
            print(f"Response preview: {result.get('response', '')[:150]}...")
            print(f"Memories used: {result.get('memories_used', 0)}")
        
        # Test retrieval with abstract queries (should favor Hyperbolic space)
        print("\n4. Testing abstract queries (should use Hyperbolic space)...")
        abstract_test_messages = [
            "What's the philosophy behind error handling?",
            "How do design patterns relate to each other?",
            "Explain the concept of optimization in software"
        ]
        
        for msg in abstract_test_messages:
            print(f"\n  Query: {msg}")
            result = self.send_chat_message(msg)
            print(f"Response preview: {result.get('response', '')[:150]}...")
            print(f"Memories used: {result.get('memories_used', 0)}")
    
    def test_clustering_behavior(self):
        """Test HDBSCAN clustering with related memories"""
        print("\n" + "="*60)
        print("TESTING HDBSCAN CLUSTERING")
        print("="*60)
        
        # Create clusters of related topics
        clusters = [
            {
                "theme": "Web Development",
                "topics": [
                    "HTML and CSS basics",
                    "JavaScript frameworks",
                    "Frontend optimization",
                    "Responsive design"
                ]
            },
            {
                "theme": "Data Science",
                "topics": [
                    "Machine learning algorithms",
                    "Data preprocessing",
                    "Statistical analysis",
                    "Model evaluation"
                ]
            },
            {
                "theme": "DevOps",
                "topics": [
                    "Container orchestration",
                    "CI/CD pipelines",
                    "Infrastructure as code",
                    "Monitoring and logging"
                ]
            }
        ]
        
        # Generate conversations for each cluster
        for cluster in clusters:
            print(f"\nCreating cluster: {cluster['theme']}")
            for topic in cluster['topics']:
                self.generate_conversation(topic, num_exchanges=2, style_sequence=["normal", "technical"])
                time.sleep(0.5)
        
        # Test cross-cluster queries
        print("\nTesting cluster-aware retrieval...")
        test_queries = [
            ("How do I deploy a web application?", "Should connect Web Dev + DevOps"),
            ("What's the relationship between frontend and data visualization?", "Should connect Web Dev + Data Science"),
            ("How do I monitor ML model performance in production?", "Should connect Data Science + DevOps")
        ]
        
        for query, expected in test_queries:
            print(f"\n  Query: {query}")
            print(f"   Expected: {expected}")
            result = self.send_chat_message(query)
            print(f"   Response preview: {result.get('response', '')[:150]}...")
            print(f"   Memories used: {result.get('memories_used', 0)}")
    
    def test_residual_adaptation(self):
        """Test how residuals adapt based on co-retrieval"""
        print("\n" + "="*60)
        print("TESTING RESIDUAL ADAPTATION")
        print("="*60)
        
        # Create initial memories
        print("\n1. Creating initial memory set...")
        topics = [
            "Python decorators",
            "Function composition",
            "Higher-order functions",
            "Metaprogramming"
        ]
        
        for topic in topics:
            self.generate_conversation(topic, num_exchanges=1)
        
        # Repeatedly query related concepts to trigger adaptation
        print("\n2. Repeated queries to trigger residual updates...")
        adaptation_queries = [
            "How do decorators relate to higher-order functions?",
            "What's the connection between decorators and metaprogramming?",
            "Explain function composition with decorators"
        ]
        
        for iteration in range(3):
            print(f"\n  Adaptation iteration {iteration + 1}/3")
            for query in adaptation_queries:
                print(f"  Query: {query[:50]}...")
                result = self.send_chat_message(query)
                time.sleep(0.5)
        
        # Test if memories have gravitated together
        print("\n3. Testing adapted retrieval...")
        test_query = "Tell me about advanced Python function techniques"
        print(f"  Final query: {test_query}")
        result = self.send_chat_message(test_query)
        print(f"Response preview: {result.get('response', '')[:200]}...")
        print(f"Memories used: {result.get('memories_used', 0)}")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("\n" + "="*30)
        print("STARTING COMPREHENSIVE TEST SUITE")
        print("="*30)
        
        # Basic conversations
        print("\n[TEST 1/4] Basic Conversations")
        topics = [
            ("Python best practices", 3),
            ("API design patterns", 2),
            ("Database optimization", 2)
        ]
        
        for topic, exchanges in topics:
            self.generate_conversation(topic, num_exchanges=exchanges)
            time.sleep(1)
        
        # Dual-space retrieval test
        print("\n[TEST 2/4] Dual-Space Retrieval")
        self.test_dual_space_retrieval()
        
        # Clustering test
        print("\n[TEST 3/4] HDBSCAN Clustering")
        self.test_clustering_behavior()
        
        # Residual adaptation test
        print("\n[TEST 4/4] Residual Adaptation")
        self.test_residual_adaptation()
        
        # Get final statistics
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        
        try:
            stats_response = requests.get(f"{self.api_url}/api/stats")
            if stats_response.ok:
                stats = stats_response.json()
                print(f"Total Events: {stats.get('total_events', 0)}")
                print(f"Total Queries: {stats.get('total_queries', 0)}")
                print(f"Total Episodes: {stats.get('total_episodes', 0)}")
                print(f"Events with Residuals: {stats.get('events_with_residuals', 0)}")
                print(f"Cached Embeddings: {stats.get('cached_embeddings', 0)}")
                
                if 'average_residual_norm' in stats:
                    norms = stats['average_residual_norm']
                    print(f"\nResidual Norms:")
                    print(f"  Euclidean: {norms.get('euclidean', 0):.4f}")
                    print(f"  Hyperbolic: {norms.get('hyperbolic', 0):.4f}")
        except Exception as e:
            print(f"Could not retrieve statistics: {e}")

def main():
    """Main function with improved error handling"""
    # Check if the web app is running
    print("Checking web application status...")
    try:
        response = requests.get("http://localhost:5000/api/stats", timeout=5)
        if not response.ok:
            print("   Web application is not responding properly")
            print("Please ensure the web app is running: python run_web.py")
            return
    except requests.exceptions.ConnectionError:
        print("  Cannot connect to the web application")
        print("Please start the web app first: python run_web.py")
        return
    except requests.exceptions.Timeout:
        print("   Web application is slow to respond")
        print("Continuing anyway...")
    
    print("  Web application is running")
    
    # Let user choose test mode
    print("\n" + "="*60)
    print("CONVERSATION SIMULATOR")
    print("="*60)
    print("\nSelect test mode:")
    print("1. Quick test (5 conversations)")
    print("2. Comprehensive test (all features)")
    print("3. Dual-space test only")
    print("4. Clustering test only")
    print("5. Adaptation test only")
    
    try:
        choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return
    
    generator = ConversationGenerator()
    
    if choice == "1":
        # Quick test with a few conversations
        print("\nRunning quick test...")
        topics = [
            ("Python programming tips", 3),
            ("Machine learning basics", 2),
            ("Web development tools", 2),
            ("Database design", 2),
            ("Cloud deployment", 2)
        ]
        for topic, exchanges in topics:
            generator.generate_conversation(topic, num_exchanges=exchanges)
            time.sleep(1)
    
    elif choice == "2":
        # Comprehensive test
        generator.run_comprehensive_test()
    
    elif choice == "3":
        # Dual-space test only
        generator.test_dual_space_retrieval()
    
    elif choice == "4":
        # Clustering test only
        generator.test_clustering_behavior()
    
    elif choice == "5":
        # Adaptation test only
        generator.test_residual_adaptation()
    
    else:
        print(f"Invalid choice: {choice}")
        return
    
    print("\n" + "="*60)
    print("  All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()
