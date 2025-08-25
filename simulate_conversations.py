"""
Generate sample conversations through the web API
This simulates real users having conversations with the assistant
"""

import requests
import json
import time
import random
from typing import List, Dict
from llm.llm_interface import LLMInterface
from config import get_config

class ConversationGenerator:
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.config = get_config()
        self.llm = LLMInterface(self.config.llm)
        
    def generate_user_message(self, context: str, topic: str) -> str:
        """Generate a realistic user message using LLM"""
        prompt = f"""Generate a realistic user message for a conversation about {topic}.
Context: {context}

The message should be natural, conversational, and something a real user might ask an AI assistant.
Keep it concise (1-2 sentences).

User message:"""
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=50
        )
        
        return response.strip()
    
    def send_chat_message(self, message: str) -> Dict:
        """Send a message to the chat API and get response"""
        try:
            response = requests.post(
                f"{self.api_url}/api/chat",
                json={"message": message},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error sending message: {e}")
            return {"response": "Error occurred", "memories_used": 0}
    
    def generate_conversation(self, topic: str, num_exchanges: int = 3):
        """Generate a full conversation on a topic"""
        print(f"\n--- Starting conversation about: {topic} ---")
        
        context = f"Starting a new conversation about {topic}"
        
        for i in range(num_exchanges):
            # Generate and send user message
            user_message = self.generate_user_message(context, topic)
            print(f"\nUser: {user_message}")
            
            # Send to API and get assistant response
            result = self.send_chat_message(user_message)
            assistant_response = result.get("response", "No response")
            memories_used = result.get("memories_used", 0)
            
            print(f"Assistant: {assistant_response}")
            if memories_used > 0:
                print(f"  (Used {memories_used} memories)")
            
            # Update context for next exchange
            context = f"Previous exchange:\nUser: {user_message}\nAssistant: {assistant_response}"
            
            # Small delay to simulate natural conversation pace
            time.sleep(1)
    
    def run_sample_conversations(self):
        """Run a set of sample conversations"""
        topics = [
            {
                "topic": "Python programming and debugging",
                "exchanges": 4
            },
            {
                "topic": "machine learning model optimization",
                "exchanges": 3
            },
            {
                "topic": "web development best practices",
                "exchanges": 3
            },
            {
                "topic": "database design and SQL queries",
                "exchanges": 3
            },
            {
                "topic": "API design and RESTful services",
                "exchanges": 4
            }
        ]
        
        print("Starting conversation generation...")
        print(f"Will generate {len(topics)} conversations")
        
        for topic_config in topics:
            self.generate_conversation(
                topic=topic_config["topic"],
                num_exchanges=topic_config["exchanges"]
            )
            
            # Pause between conversations
            print("\n" + "="*50)
            time.sleep(2)
        
        print("\nAll conversations completed!")
        
        # Get final statistics
        try:
            stats_response = requests.get(f"{self.api_url}/api/stats")
            if stats_response.ok:
                stats = stats_response.json()
                print(f"\nFinal Statistics:")
                print(f"  Total Events: {stats.get('total_events', 0)}")
                print(f"  Episodes: {stats.get('episodes', 0)}")
        except Exception as e:
            print(f"Could not retrieve statistics: {e}")

def main():
    # Check if the web app is running
    try:
        response = requests.get("http://localhost:5000/api/stats")
        if not response.ok:
            print("Error: Web application is not responding properly")
            print("Please ensure the web app is running: python run_web.py")
            return
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the web application")
        print("Please start the web app first: python run_web.py")
        return
    
    # Generate conversations
    generator = ConversationGenerator()
    generator.run_sample_conversations()

if __name__ == "__main__":
    main()
