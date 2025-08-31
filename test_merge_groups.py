#!/usr/bin/env python3
"""Test script to verify merge groups show all participants"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def create_memory(who, what, when="", where="", why="", how=""):
    """Create a memory via the API"""
    response = requests.post(f"{BASE_URL}/api/memories", json={
        "who": who,
        "what": what,
        "when": when,
        "where": where,
        "why": why,
        "how": how
    })
    return response.json()

def get_merge_groups(dimension="temporal"):
    """Get merge groups for a dimension"""
    response = requests.get(f"{BASE_URL}/api/merge-groups/{dimension}")
    return response.json()

def get_merge_group_details(merge_type, merge_id):
    """Get detailed info about a merge group"""
    response = requests.get(f"{BASE_URL}/api/multi-merge/{merge_type}/{merge_id}/details")
    return response.json()

def main():
    print("Testing merge groups with multiple participants...")
    
    # Create a conversation with multiple participants
    print("\n1. Creating memories from different participants...")
    
    # User asks a question
    result1 = create_memory(
        who="User",
        what="How do I implement a binary search tree in Python?",
        when="2025-08-31T10:00:00",
        where="chat_interface",
        why="learning data structures",
        how="asking question"
    )
    print(f"Created memory 1: {result1['message']}")
    time.sleep(0.5)
    
    # Assistant responds
    result2 = create_memory(
        who="Assistant",
        what="To implement a binary search tree in Python, you need to create a Node class with left and right children, and implement insert, search, and delete methods.",
        when="2025-08-31T10:00:05",
        where="chat_interface",
        why="teaching data structures",
        how="explaining implementation"
    )
    print(f"Created memory 2: {result2['message']}")
    time.sleep(0.5)
    
    # User follows up
    result3 = create_memory(
        who="User",
        what="Can you show me the code for the insert method?",
        when="2025-08-31T10:00:10",
        where="chat_interface",
        why="understanding implementation details",
        how="requesting code example"
    )
    print(f"Created memory 3: {result3['message']}")
    time.sleep(0.5)
    
    # Assistant provides code
    result4 = create_memory(
        who="Assistant",
        what="Here's the insert method: def insert(self, value): if value < self.value: if self.left is None: self.left = Node(value) else: self.left.insert(value)...",
        when="2025-08-31T10:00:15",
        where="chat_interface",
        why="providing code example",
        how="sharing code snippet"
    )
    print(f"Created memory 4: {result4['message']}")
    time.sleep(1)
    
    # Get temporal merge groups
    print("\n2. Fetching temporal merge groups...")
    groups = get_merge_groups("temporal")
    print(f"Found {len(groups['groups'])} temporal merge groups")
    
    if groups['groups']:
        # Get details of the first group
        first_group = groups['groups'][0]
        print(f"\n3. Getting details for merge group: {first_group['id']}")
        
        details = get_merge_group_details("temporal", first_group['id'])
        
        # Check WHO variants
        print("\n4. Checking WHO participants in the merge group:")
        who_variants = details.get('who_variants', {})
        participants = list(who_variants.keys())
        print(f"   Participants found: {participants}")
        
        if 'User' in participants and 'Assistant' in participants:
            print("   ✓ SUCCESS: Both User and Assistant are shown in the merge group!")
        else:
            print("   ✗ ISSUE: Not all participants are shown")
            print(f"   Expected: ['User', 'Assistant']")
            print(f"   Got: {participants}")
        
        # Check raw events
        print("\n5. Checking raw events:")
        raw_events = details.get('raw_events', [])
        print(f"   Total raw events: {len(raw_events)}")
        
        event_participants = set()
        for event in raw_events:
            who = event.get('five_w1h', {}).get('who', event.get('who', ''))
            if who:
                event_participants.add(who)
                print(f"   - Event from: {who}")
        
        print(f"\n   Unique participants in raw events: {list(event_participants)}")
        
        if 'User' in event_participants and 'Assistant' in event_participants:
            print("   ✓ SUCCESS: Raw events include all participants!")
        else:
            print("   ✗ ISSUE: Raw events missing some participants")
    
    print("\n6. Testing other merge dimensions...")
    
    # Test conceptual groups
    conceptual_groups = get_merge_groups("conceptual")
    print(f"   Conceptual groups: {len(conceptual_groups['groups'])}")
    
    # Test spatial groups  
    spatial_groups = get_merge_groups("spatial")
    print(f"   Spatial groups: {len(spatial_groups['groups'])}")
    
    # Test actor groups (should group by individual actors)
    actor_groups = get_merge_groups("actor")
    print(f"   Actor groups: {len(actor_groups['groups'])}")
    
    if actor_groups['groups']:
        print("\n7. Checking actor groups (should be grouped by individual actors):")
        for group in actor_groups['groups']:
            details = get_merge_group_details("actor", group['id'])
            who_variants = details.get('who_variants', {})
            participants = list(who_variants.keys())
            print(f"   Actor group {group['id'][:8]}... has participants: {participants}")
            if len(participants) == 1:
                print(f"     ✓ Correctly grouped by single actor: {participants[0]}")
            else:
                print(f"     ✗ Actor group has multiple participants (should be single)")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()