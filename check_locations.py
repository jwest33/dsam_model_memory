import json

data = json.load(open('benchmark_datasets/benchmark_20250830_064528.json'))
locations = {}
for e in data['events']:
    loc = e.get('where', 'unknown')
    locations[loc] = locations.get(loc, 0) + 1
    
print('Location distribution:')
for k, v in sorted(locations.items()):
    print(f'  {k}: {v} events')