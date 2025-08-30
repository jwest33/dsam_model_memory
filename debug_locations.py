"""Debug where locations are getting lost"""

import json
from models.event import Event, EventType, FiveW1H

# Load the dataset
data = json.load(open('benchmark_datasets/benchmark_20250830_064528.json'))

print("First 5 events from file:")
for i, event_data in enumerate(data['events'][:5]):
    where = event_data['five_w1h'].get('where', 'NOT_FOUND')
    print(f"  Event {i}: where = '{where}'")

print("\nCreating Event objects:")
for i, event_data in enumerate(data['events'][:5]):
    event = Event(
        five_w1h=FiveW1H(
            who=event_data['five_w1h'].get('who', ''),
            what=event_data['five_w1h'].get('what', ''),
            when=event_data['five_w1h'].get('when', ''),
            where=event_data['five_w1h'].get('where', ''),
            why=event_data['five_w1h'].get('why', ''),
            how=event_data['five_w1h'].get('how', '')
        ),
        event_type=EventType(event_data.get('event_type', 'observation')),
        episode_id=event_data.get('episode_id', f"benchmark_{i}")
    )
    print(f"  Event {i}: event.five_w1h.where = '{event.five_w1h.where}'")