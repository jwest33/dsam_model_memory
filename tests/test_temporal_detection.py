#!/usr/bin/env python3
"""Test temporal intent detection."""

import os
import sys
sys.path.insert(0, '.')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from memory.temporal_manager import TemporalManager

# Create temporal manager
temporal_mgr = TemporalManager()

# Test queries
test_queries = [
    {'what': 'what is the last thing we discussed'},
    {'what': 'what is the last thing we discussed', 'who': 'User'},
    {'message': 'what is the last thing we discussed'},
    {'query': 'what is the last thing we discussed'},
]

for query in test_queries:
    strength, confidence, params = temporal_mgr.detect_temporal_intent(query)
    print(f"\nQuery: {query}")
    print(f"  Strength: {strength}")
    print(f"  Confidence: {confidence}")
    print(f"  Params: {params}")

# Show what indicators it's looking for
print("\n\nStrong temporal indicators:")
for indicator in temporal_mgr.TEMPORAL_INDICATORS[temporal_mgr.TemporalStrength.STRONG]:
    print(f"  - '{indicator}'")
