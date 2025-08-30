"""Test location generation in benchmark dataset"""

from benchmark.generate_benchmark_dataset import BenchmarkDatasetGenerator, LocationType, PersonaType
import random

# Initialize generator
generator = BenchmarkDatasetGenerator()

# Test each persona's locations
print("Testing location generation for each persona:\n")

for persona in PersonaType:
    persona_info = generator.personas[persona]
    print(f"{persona.value}:")
    print(f"  Typical locations: {[loc.value for loc in persona_info['typical_locations']]}")
    
    # Generate a few samples
    samples = []
    for _ in range(5):
        location = random.choice(persona_info['typical_locations'])
        samples.append(location.value)
    print(f"  5 random samples: {samples}")
    print()

# Test actual conversation generation
print("="*60)
print("Testing actual conversation generation:\n")

scenario = generator.scenarios[0]
persona = PersonaType.DEVELOPER
timestamp = "2025-08-30T12:00:00"

events = generator.generate_conversation_exchange(scenario, persona, timestamp)

print(f"Generated {len(events)} events")
print("\nLocation distribution in generated events:")
locations = {}
for event in events:
    loc = event.five_w1h.where
    locations[loc] = locations.get(loc, 0) + 1

for loc, count in locations.items():
    print(f"  {loc}: {count} events")