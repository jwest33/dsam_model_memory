#!/usr/bin/env python3
"""Check what events are in a temporal group and their timestamps."""

import requests
import json
from datetime import datetime

# Get the specific temporal group that seems problematic
group_id = "temporal_35ec133d"  # The group with mixed timestamps

detail_response = requests.get(f'http://localhost:5000/api/merge-group/temporal/{group_id}')
if detail_response.status_code == 200:
    details = detail_response.json()
    events = details.get('events', [])
    
    print(f"Temporal Group: {group_id}")
    print(f"Total events: {len(events)}")
    print(f"Group created_at: {details.get('created_at', 'N/A')}")
    print(f"Group last_updated: {details.get('last_updated', 'N/A')}")
    print("\n" + "="*80)
    print("ALL EVENTS IN THIS GROUP (sorted by 'when'):")
    print("="*80)
    
    # Sort events by their 'when' timestamp
    sorted_events = sorted(events, key=lambda x: x.get('when', ''), reverse=False)
    
    prev_time = None
    for i, event in enumerate(sorted_events):
        when = event.get('when', 'N/A')
        who = event.get('who', 'N/A')
        what = event.get('what', '')[:80] + '...' if len(event.get('what', '')) > 80 else event.get('what', '')
        
        # Parse timestamp to check time differences
        time_diff_str = ""
        if when != 'N/A' and prev_time:
            try:
                current = datetime.fromisoformat(when.replace('Z', '+00:00'))
                previous = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                diff = current - previous
                hours = diff.total_seconds() / 3600
                if hours > 1:
                    time_diff_str = f" [+{hours:.1f} hours from previous]"
                else:
                    minutes = diff.total_seconds() / 60
                    time_diff_str = f" [+{minutes:.1f} minutes from previous]"
            except:
                pass
        
        print(f"\nEvent {i+1}:")
        print(f"  When: {when}{time_diff_str}")
        print(f"  Who: {who}")
        print(f"  What: {what}")
        
        prev_time = when
    
    # Calculate total time span
    if sorted_events:
        first_when = sorted_events[0].get('when', '')
        last_when = sorted_events[-1].get('when', '')
        if first_when and last_when:
            try:
                first = datetime.fromisoformat(first_when.replace('Z', '+00:00'))
                last = datetime.fromisoformat(last_when.replace('Z', '+00:00'))
                span = last - first
                hours = span.total_seconds() / 3600
                print(f"\n" + "="*80)
                print(f"TIME SPAN: {hours:.1f} hours between first and last event")
                print(f"First event: {first_when}")
                print(f"Last event: {last_when}")
            except:
                pass
else:
    print(f"Failed to get group details: {detail_response.status_code}")
    
# Also check what episode_id these events have
print("\n" + "="*80)
print("CHECKING RAW EVENT EPISODE IDs:")
print("="*80)

# We need to get the raw event IDs from the group
groups_response = requests.get('http://localhost:5000/api/merge-groups/temporal')
if groups_response.status_code == 200:
    groups = groups_response.json().get('groups', [])
    target_group = next((g for g in groups if g['id'] == group_id), None)
    
    if target_group:
        print(f"Raw event IDs in group: {target_group.get('raw_event_ids', 'N/A')[:200]}...")
        # Note: We can't directly check episode_ids through the API without modifying it
        print("\nNote: Episode IDs are not exposed through the current API.")
        print("The issue is likely that all events share the same episode_id,")
        print("causing them to be grouped together regardless of time gaps.")