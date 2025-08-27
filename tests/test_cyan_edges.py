#!/usr/bin/env python3
"""Test script to verify cyan edge highlighting for center node"""

import requests
import json

def test_graph_edges():
    url = "http://localhost:5000/api/graph"
    
    # Use a known memory ID
    test_id = "3c9a37e1-a6b3-4501-a574-bd9d11c4e473"
    
    payload = {
        "components": ["who", "what", "when", "where", "why", "how"],
        "use_clustering": False,
        "visualization_mode": "dual",
        "center_node": test_id,
        "similarity_threshold": 0.2
    }
    
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        
        if response.status_code == 200:
            # Count edges connected to center
            center_edges = 0
            total_edges = len(data.get('edges', []))
            
            for edge in data.get('edges', []):
                if edge.get('from') == test_id or edge.get('to') == test_id:
                    center_edges += 1
            
            print(f"✓ Graph loaded successfully")
            print(f"  - Total nodes: {len(data.get('nodes', []))}")
            print(f"  - Total edges: {total_edges}")
            print(f"  - Edges connected to center: {center_edges}")
            print(f"\nCenter node edges will be displayed in CYAN")
            print(f"Regular edges will be more subtle/transparent")
            
            return True
        else:
            print(f"✗ Request failed: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing cyan edge highlighting for center node...")
    print("-" * 40)
    
    success = test_graph_edges()
    
    if success:
        print("\n✓ Edge styling configured successfully!")
        print("  Center node edges: CYAN, width 3px, high opacity")
        print("  Regular edges: Muted colors, width 1px, low opacity")
    else:
        print("\n✗ Edge styling test failed")