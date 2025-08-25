"""
Quick test script to send a single message to the chat API
"""

import requests
import json
import sys

def test_chat(message: str = None):
    """Send a test message to the chat API"""
    
    if message is None:
        message = "Hello! Can you help me understand how memory clustering works in this system?"
    
    api_url = "http://localhost:5000"
    
    # Check if server is running
    try:
        response = requests.get(f"{api_url}/api/stats")
        if not response.ok:
            print("Error: Web application is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the web application")
        print("Please start the web app first: python run_web.py")
        return
    
    print(f"Sending message: {message}")
    print("-" * 50)
    
    # Send chat message
    try:
        response = requests.post(
            f"{api_url}/api/chat",
            json={"message": message},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        print(f"Assistant: {data['response']}")
        
        if data.get('memories_used', 0) > 0:
            print(f"\n(Used {data['memories_used']} memories in response)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use command line argument as message
        message = " ".join(sys.argv[1:])
        test_chat(message)
    else:
        # Use default message
        test_chat()