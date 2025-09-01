#!/usr/bin/env python3
"""Check timestamps via API."""

import requests
import json
from datetime import datetime

# Get a chat response to see what's in the LLM context
response = requests.post('http://localhost:5000/api/chat', 
    json={'message': 'what happened yesterday', 'merge_dimension': 'temporal'})

if response.status_code == 200:
    result = response.json()
    
    # Check the memory details
    print("=== Memory Details ===")
    for i, memory in enumerate(result.get('memory_details', [])[:2]):
        print(f"\nMemory {i+1}:")
        print(f"  ID: {memory.get('id', 'N/A')}")
        print(f"  When field: {memory.get('when', 'N/A')}")
        print(f"  What: {memory.get('what', '')[:60]}...")
    
    # Check the actual LLM prompt to see what timestamps are being used
    print("\n=== LLM Prompt (checking timestamps) ===")
    llm_prompt = result.get('llm_prompt', '')
    
    # Find and print lines with "When:" to see what timestamps are shown
    lines = llm_prompt.split('\n')
    for i, line in enumerate(lines):
        if 'When:' in line or 'Event ' in line:
            print(line)
            # Also print the next few lines for context
            if 'Event ' in line:
                for j in range(1, min(8, len(lines)-i)):
                    if i+j < len(lines):
                        print(lines[i+j])
                        if 'When:' in lines[i+j]:
                            break
                            
# Also check a temporal group directly
print("\n\n=== Temporal Group Check ===")
groups_response = requests.get('http://localhost:5000/api/merge-groups/temporal')
if groups_response.status_code == 200:
    groups = groups_response.json().get('groups', [])
    if groups:
        # Get the first group's details
        first_group = groups[0]
        group_id = first_group['id']
        
        print(f"Checking group: {group_id}")
        print(f"Group last_updated: {first_group.get('last_updated', 'N/A')}")
        print(f"Group created_at: {first_group.get('created_at', 'N/A')}")
        
        # Get the group details
        detail_response = requests.get(f'http://localhost:5000/api/merge-group/temporal/{group_id}')
        if detail_response.status_code == 200:
            details = detail_response.json()
            events = details.get('events', [])
            
            print(f"\nFirst 3 events in group:")
            for i, event in enumerate(events[:3]):
                print(f"\nEvent {i+1}:")
                print(f"  When (should be event time): {event.get('when', 'N/A')}")
                print(f"  What: {event.get('what', '')[:60]}...")