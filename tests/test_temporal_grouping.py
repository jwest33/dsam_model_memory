#!/usr/bin/env python3
"""Test that temporal grouping splits chains based on time gaps."""

import os
import sys
sys.path.insert(0, '.')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from datetime import datetime, timedelta, timezone
from models.event import Event, FiveW1H, EventType
from memory.temporal_chain import TemporalChain
from memory.temporal_manager import TemporalManager
from config import Config
import uuid

# Create config with specific temporal settings
config = Config.from_env()
config.temporal.temporal_group_window = 30  # 30 minutes
config.temporal.max_temporal_gap = 60  # 60 minutes
config.temporal.use_episode_for_temporal = False  # Don't use episode_id

print("Temporal Grouping Configuration:")
print(f"  Temporal group window: {config.temporal.temporal_group_window} minutes")
print(f"  Max temporal gap: {config.temporal.max_temporal_gap} minutes")
print(f"  Use episode for temporal: {config.temporal.use_episode_for_temporal}")
print()

# Create temporal chain with config
temporal_chain = TemporalChain(config=config)

# Create test events with time gaps
base_time = datetime.now(timezone.utc)
episode_id = f"test_episode_{uuid.uuid4().hex[:8]}"

test_events = [
    # Group 1: Events within 30-minute window (should be in same temporal group)
    Event(
        id=f"event_1",
        event_type=EventType.USER_INPUT,
        five_w1h=FiveW1H(
            who="User",
            what="First question about Python",
            when=base_time.isoformat(),
            where="chat",
            why="learning",
            how="typed"
        ),
        episode_id=episode_id,
        created_at=base_time
    ),
    Event(
        id=f"event_2",
        event_type=EventType.USER_INPUT,
        five_w1h=FiveW1H(
            who="User",
            what="Follow-up about Python loops",
            when=(base_time + timedelta(minutes=10)).isoformat(),
            where="chat",
            why="learning",
            how="typed"
        ),
        episode_id=episode_id,
        created_at=base_time + timedelta(minutes=10)
    ),
    Event(
        id=f"event_3",
        event_type=EventType.USER_INPUT,
        five_w1h=FiveW1H(
            who="User",
            what="Another Python question",
            when=(base_time + timedelta(minutes=25)).isoformat(),
            where="chat",
            why="learning",
            how="typed"
        ),
        episode_id=episode_id,
        created_at=base_time + timedelta(minutes=25)
    ),
    
    # Group 2: Event after 2 hours (should be in NEW temporal group)
    Event(
        id=f"event_4",
        event_type=EventType.USER_INPUT,
        five_w1h=FiveW1H(
            who="User",
            what="Question about databases",
            when=(base_time + timedelta(hours=2)).isoformat(),
            where="chat",
            why="learning",
            how="typed"
        ),
        episode_id=episode_id,  # Same episode!
        created_at=base_time + timedelta(hours=2)
    ),
    Event(
        id=f"event_5",
        event_type=EventType.USER_INPUT,
        five_w1h=FiveW1H(
            who="User",
            what="Follow-up about SQL",
            when=(base_time + timedelta(hours=2, minutes=15)).isoformat(),
            where="chat",
            why="learning",
            how="typed"
        ),
        episode_id=episode_id,  # Same episode!
        created_at=base_time + timedelta(hours=2, minutes=15)
    ),
    
    # Group 3: Event after 5 hours (should be in ANOTHER NEW temporal group)
    Event(
        id=f"event_6",
        event_type=EventType.USER_INPUT,
        five_w1h=FiveW1H(
            who="User",
            what="Question about machine learning",
            when=(base_time + timedelta(hours=5)).isoformat(),
            where="chat",
            why="learning",
            how="typed"
        ),
        episode_id=episode_id,  # Same episode!
        created_at=base_time + timedelta(hours=5)
    ),
]

# Add events to temporal chain
print("Adding events to temporal chain...")
chains_created = set()
for event in test_events:
    chain_id = temporal_chain.add_event(event)
    chains_created.add(chain_id)
    print(f"  Event {event.id}: Added to chain {chain_id}")
    print(f"    Time: {event.created_at.strftime('%H:%M:%S')}")
    print(f"    What: {event.five_w1h.what}")

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"Total unique temporal chains created: {len(chains_created)}")
print(f"Chain IDs: {chains_created}")

# Expected: 3 different temporal groups
if len(chains_created) >= 3:
    print("\nSUCCESS: Temporal grouping correctly split events based on time gaps!")
    print("   Events were split into separate temporal groups despite having the same episode_id.")
else:
    print(f"\nFAILURE: Expected at least 3 temporal groups but got {len(chains_created)}")
    print("   Events may still be grouped by episode_id instead of time proximity.")

# Show the chains
print(f"\n{'='*60}")
print("TEMPORAL CHAINS DETAIL:")
print(f"{'='*60}")
for chain_id in chains_created:
    events_in_chain = temporal_chain.chains.get(chain_id, [])
    print(f"\nChain: {chain_id}")
    print(f"  Events: {events_in_chain}")
    
    if events_in_chain:
        # Get time span of this chain
        first_event = next((e for e in test_events if e.id == events_in_chain[0]), None)
        last_event = next((e for e in test_events if e.id == events_in_chain[-1]), None)
        
        if first_event and last_event:
            time_span = last_event.created_at - first_event.created_at
            print(f"  Time span: {time_span.total_seconds() / 60:.1f} minutes")
            print(f"  First: {first_event.five_w1h.what[:50]}")
            print(f"  Last: {last_event.five_w1h.what[:50]}")
