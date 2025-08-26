#!/usr/bin/env python
"""Test the /api/memories endpoint directly"""

import requests
import json

def test_memories_endpoint():
    """Test if the memories endpoint is working"""
    
    url = "http://localhost:5000/api/memories"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response keys: {data.keys()}")
            
            if 'memories' in data:
                print(f"Number of memories: {len(data['memories'])}")
                if data['memories']:
                    print(f"First memory: {json.dumps(data['memories'][0], indent=2)}")
            
            if 'stats' in data:
                print(f"Stats: {json.dumps(data['stats'], indent=2)}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error calling endpoint: {e}")

if __name__ == "__main__":
    test_memories_endpoint()
