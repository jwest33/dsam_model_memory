#!/usr/bin/env python3
"""Find the most recent temporal group."""

import requests
import json
from datetime import datetime

# Get all temporal groups
groups_response = requests.get('http://localhost:5000/api/merge-groups/temporal')
if groups_response.status_code == 200:
    groups = groups_response.json().get('groups', [])
    
    # Sort by last_updated to get the most recent
    sorted_groups = sorted(groups, key=lambda x: x.get('last_updated', ''), reverse=True)
    
    print(f"Total temporal groups: {len(groups)}")
    print("\n" + "="*60)
    print("MOST RECENT TEMPORAL GROUPS (by last_updated):")
    print("="*60)
    
    for i, group in enumerate(sorted_groups[:5]):
        print(f"\n{i+1}. Group ID: {group['id']}")
        print(f"   Last updated: {group.get('last_updated', 'N/A')}")
        print(f"   Created at: {group.get('created_at', 'N/A')}")
        print(f"   Events count: {group.get('events_count', 'N/A')}")
        
        latest = group.get('latest_state', {})
        if latest:
            print(f"   Latest Who: {latest.get('who', 'N/A')}")
            print(f"   Latest What: {latest.get('what', 'N/A')[:100]}...")
            print(f"   Latest When: {latest.get('when', 'N/A')}")
    
    # The answer to "what is the last thing we discussed"
    if sorted_groups:
        print("\n" + "="*60)
        print("ANSWER TO 'What is the last thing we discussed?'")
        print("="*60)
        most_recent = sorted_groups[0]
        print(f"\nThe most recent temporal group is: {most_recent['id']}")
        print(f"Last updated: {most_recent.get('last_updated', 'N/A')}")
        
        latest = most_recent.get('latest_state', {})
        if latest:
            print(f"\nThe last thing discussed was:")
            print(f"  Who: {latest.get('who', 'N/A')}")
            print(f"  What: {latest.get('what', 'N/A')}")
            print(f"  When: {latest.get('when', 'N/A')}")
            print(f"  Where: {latest.get('where', 'N/A')}")
            
        # Get the full details of this group
        detail_response = requests.get(f'http://localhost:5000/api/merge-group/temporal/{most_recent["id"]}')
        if detail_response.status_code == 200:
            details = detail_response.json()
            events = details.get('events', [])
            if events:
                # Sort events by 'when' to get the actual most recent event
                sorted_events = sorted(events, key=lambda x: x.get('when', ''), reverse=True)
                print(f"\nThis group contains {len(events)} events")
                print(f"Most recent event in the group:")
                most_recent_event = sorted_events[0]
                print(f"  Who: {most_recent_event.get('who', 'N/A')}")
                print(f"  What: {most_recent_event.get('what', 'N/A')}")
                print(f"  When: {most_recent_event.get('when', 'N/A')}")
else:
    print(f"Failed to get temporal groups: {groups_response.status_code}")