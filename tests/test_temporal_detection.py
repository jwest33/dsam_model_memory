"""Quick test of enhanced temporal detection."""

import sys
import numpy as np
from memory.temporal_query import TemporalQueryHandler

def test_regex_extraction():
    """Test regex-based time extraction."""
    print("Testing regex time extraction...")
    
    # Create handler without LLM
    handler = TemporalQueryHandler(encoder=None, llm_client=None)
    
    test_cases = [
        ("what did we discuss 5 minutes ago", "5 minutes ago"),
        ("remind me what we talked about yesterday", "yesterday"),
        ("what was the last thing we discussed", "last thing"),
        ("a few minutes ago we covered", "a few minutes ago"),
        ("2 hours ago you mentioned", "2 hours ago"),
    ]
    
    for query_text, expected_phrase in test_cases:
        query = {'what': query_text}
        context = handler._extract_time_references(query)
        
        print(f"\nQuery: '{query_text}'")
        print(f"  Has time reference: {context.get('has_time_reference')}")
        
        if context.get('time_references'):
            for ref in context['time_references']:
                print(f"  Found: '{ref['text']}' (type: {ref['type']})")
                if 'hours_ago' in ref:
                    print(f"    Hours ago: {ref['hours_ago']}")
        else:
            print("  No time references found")
    
    print("\n" + "="*60)

def test_expanded_patterns():
    """Test expanded temporal reference patterns."""
    print("\nTesting expanded temporal patterns...")
    
    # Create mock encoder
    class MockEncoder:
        def encode(self, fields):
            return {
                'euclidean_anchor': np.random.randn(768),
                'hyperbolic_anchor': np.random.randn(64)
            }
    
    handler = TemporalQueryHandler(encoder=MockEncoder(), llm_client=None)
    
    test_queries = [
        "remind me what we just covered",
        "what did you just say",
        "going back to what we just said",
        "a few minutes ago",
        "within the last hour",
        "during this chat",
        "everything we've discussed",
        "from before",
        "at the start",
        "chronologically speaking",
    ]
    
    print("\nChecking if patterns are recognized in TEMPORAL_REFERENCES:")
    for query_text in test_queries:
        found = False
        for category, patterns in handler.TEMPORAL_REFERENCES.items():
            for pattern in patterns:
                if pattern.lower() in query_text.lower():
                    print(f"  '{query_text}' -> Found in {category}")
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"  '{query_text}' -> Not found directly")

if __name__ == "__main__":
    test_regex_extraction()
    test_expanded_patterns()
    print("\nTests completed!")
