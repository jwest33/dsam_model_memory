"""
Test script to verify size-based group field regeneration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.multi_dimensional_merger import MultiDimensionalMerger
from models.merge_types import MergeType

def test_regeneration_logic():
    """Test the size-based regeneration logic"""
    
    print("=" * 60)
    print("Testing Size-Based Regeneration Logic")
    print("=" * 60)
    
    # Create a merger instance
    merger = MultiDimensionalMerger()
    
    # Test cases for different group sizes
    test_cases = [
        (1, True, "Size 1: Always regenerate"),
        (2, True, "Size 2: Always regenerate"),
        (3, True, "Size 3: Always regenerate"),
        (4, True, "Size 4: Every 2 events (4%2=0)"),
        (5, False, "Size 5: Every 2 events (5%2=1)"),
        (6, True, "Size 6: Every 2 events (6%2=0)"),
        (10, True, "Size 10: Every 2 events (10%2=0)"),
        (11, False, "Size 11: Every 5 events (11%5=1)"),
        (15, True, "Size 15: Every 5 events (15%5=0)"),
        (20, True, "Size 20: Every 5 events (20%5=0)"),
        (25, True, "Size 25: Every 5 events (25%5=0)"),
        (26, False, "Size 26: Every 10 events (26%10=6)"),
        (30, True, "Size 30: Every 10 events (30%10=0)"),
        (40, True, "Size 40: Every 10 events (40%10=0)"),
        (50, True, "Size 50: Every 10 events (50%10=0)"),
        (51, False, "Size 51: Every 20 events (51%20=11)"),
        (60, True, "Size 60: Every 20 events (60%20=0)"),
        (100, True, "Size 100: Every 20 events (100%20=0)"),
        (101, False, "Size 101: Every 20 events (101%20=1)"),
    ]
    
    print("\nRegeneration Schedule:")
    print("-" * 40)
    all_passed = True
    
    for size, expected, description in test_cases:
        result = merger._should_regenerate_based_on_size(size)
        status = "✓" if result == expected else "✗"
        
        if result != expected:
            all_passed = False
            print(f"{status} FAILED: {description}")
            print(f"   Expected: {expected}, Got: {result}")
        else:
            print(f"{status} {description}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ All regeneration tests passed!")
    else:
        print("❌ Some regeneration tests failed!")
    
    # Show regeneration frequency analysis
    print("\n" + "=" * 60)
    print("Regeneration Frequency Analysis:")
    print("-" * 40)
    
    ranges = [
        (1, 3, "Very small groups"),
        (4, 10, "Small groups"),
        (11, 25, "Medium groups"),
        (26, 50, "Large groups"),
        (51, 100, "Very large groups")
    ]
    
    for start, end, label in ranges:
        regenerations = sum(1 for i in range(start, end+1) 
                          if merger._should_regenerate_based_on_size(i))
        total = end - start + 1
        percentage = (regenerations / total) * 100
        print(f"{label} (size {start}-{end}):")
        print(f"  Regenerates {regenerations}/{total} times ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("-" * 40)
    print("• Very small groups (1-3): Regenerate 100% - capture rapid evolution")
    print("• Small groups (4-10): Regenerate ~50% - balance freshness and stability")
    print("• Medium groups (11-25): Regenerate ~20% - more stable characterization")
    print("• Large groups (26-50): Regenerate ~10% - established patterns")
    print("• Very large groups (51+): Regenerate ~5% - highly stable")
    print("=" * 60)

if __name__ == "__main__":
    test_regeneration_logic()
