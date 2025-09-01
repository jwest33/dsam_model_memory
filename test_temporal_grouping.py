#!/usr/bin/env python3
"""Test temporal grouping behavior"""

from datetime import datetime, timedelta, timezone
import uuid

# Test temporal grouping behavior
now = datetime.now(timezone.utc)
times = [
    now - timedelta(hours=8),  # 8 hours ago
    now - timedelta(hours=7, minutes=45),  # 7:45 ago
    now - timedelta(minutes=45),  # 45 minutes ago
    now - timedelta(minutes=30),  # 30 minutes ago  
    now - timedelta(minutes=10),  # 10 minutes ago
    now  # Now
]

# Simulate what the merge strategy would do
from models.merge_types import MERGE_STRATEGIES, MergeType

temporal_strategy = MERGE_STRATEGIES[MergeType.TEMPORAL]

print('Temporal window min:', temporal_strategy.temporal_window_min, 'minutes')
print('Temporal window max:', temporal_strategy.temporal_window_max, 'minutes')
print()

# Test grouping
print('Testing temporal grouping with events at:')
for i, t in enumerate(times):
    print(f'  Event {i+1}: {t.strftime("%H:%M")} ({t.strftime("%Y-%m-%d %H:%M:%S %Z")})')

print()
print('Expected grouping:')
print('  Group 1: Events 1-2 (8 hours ago, close together)')
print('  Group 2: Events 3-6 (recent, within 45 minutes)')

print()
print('Testing should_merge_with_group logic:')

# Simulate events with timestamps
events = []
for i, t in enumerate(times):
    events.append({
        'when': t.isoformat(),
        'timestamp': t,
        'what': f'Event {i+1}'
    })

# Test merging
groups = []
for event in events:
    merged = False
    for group in groups:
        # Check if should merge with this group
        distance = 0.3  # Assume moderate distance
        if temporal_strategy.should_merge_with_group(event, None, group['events'], distance):
            group['events'].append(event)
            merged = True
            print(f"  Event {event['what']} merged with existing group")
            break
    
    if not merged:
        # Create new group
        groups.append({'events': [event]})
        print(f"  Event {event['what']} created new group")

print()
print(f'Result: {len(groups)} temporal groups created')
for i, group in enumerate(groups):
    print(f'  Group {i+1}: {len(group["events"])} events')
    for event in group['events']:
        print(f'    - {event["what"]} at {event["timestamp"].strftime("%H:%M")}')