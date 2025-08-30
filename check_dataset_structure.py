import json
import pprint

data = json.load(open('benchmark_datasets/benchmark_20250830_064528.json'))

print('Dataset structure:')
print(f'  - metadata: {list(data.get("metadata", {}).keys())}')
print(f'  - conversations: {len(data.get("conversations", []))} items')
print(f'  - events: {len(data.get("events", []))} items')
print()

if data.get('conversations'):
    print('First conversation structure:')
    conv = data['conversations'][0]
    for key in conv.keys():
        value = conv[key]
        if isinstance(value, list):
            print(f'    - {key}: list with {len(value)} items')
        elif isinstance(value, dict):
            print(f'    - {key}: dict with keys {list(value.keys())}')
        else:
            print(f'    - {key}: {type(value).__name__} = {value}')
    
    print('\nFirst conversation exchanges (first 2):')
    for i, exchange in enumerate(conv.get('exchanges', [])[:2]):
        print(f'  Exchange {i+1}:')
        for k, v in exchange.items():
            if len(str(v)) > 50:
                print(f'    {k}: {str(v)[:50]}...')
            else:
                print(f'    {k}: {v}')