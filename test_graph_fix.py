#!/usr/bin/env python3
"""Test script to verify graph modal fix"""

import requests
import json

# Test the graph endpoint with a center node
def test_graph_with_center():
    url = "http://localhost:5000/api/graph"
    
    # Use a known memory ID from the data
    test_id = "3c9a37e1-a6b3-4501-a574-bd9d11c4e473"
    
    payload = {
        "components": ["who", "what", "when", "where", "why", "how"],
        "use_clustering": False,
        "visualization_mode": "dual",
        "center_node": test_id,
        "similarity_threshold": 0.4
    }
    
    try:
        print(f"Sending request with center_node: {test_id[:8]}...")
        response = requests.post(url, json=payload)
        data = response.json()
        
        if response.status_code == 200:
            # Check if center node is in the nodes
            center_found = False
            for node in data.get('nodes', []):
                if node['id'] == test_id:
                    center_found = True
                    if node.get('is_center'):
                        print(f"✓ Center node {test_id[:8]}... found and marked correctly")
                    else:
                        print(f"⚠ Center node {test_id[:8]}... found but not marked as center")
                    break
            
            if not center_found:
                print(f"✗ Center node {test_id[:8]}... NOT found in graph nodes")
                print(f"  Total nodes returned: {len(data.get('nodes', []))}")
                # Debug: print first few node IDs
                print(f"  First 5 node IDs:")
                for i, node in enumerate(data.get('nodes', [])[:5]):
                    print(f"    {i+1}. {node['id'][:8]}... - {node.get('who', 'Unknown')}")
            
            # Print summary
            print(f"\nGraph summary:")
            print(f"  - Nodes: {len(data.get('nodes', []))}")
            print(f"  - Edges: {len(data.get('edges', []))}")
            print(f"  - Clusters: {data.get('cluster_count', 0)}")
            
            return center_found
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Error: {data.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to web app. Make sure it's running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing graph modal fix...")
    print("-" * 40)
    
    success = test_graph_with_center()
    
    if success:
        print("\n✓ Graph modal fix is working!")
    else:
        print("\n✗ Graph modal fix needs more work")