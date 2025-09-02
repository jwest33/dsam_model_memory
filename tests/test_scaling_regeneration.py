"""
Test and visualize the scaling function for group field regeneration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
from memory.multi_dimensional_merger import MultiDimensionalMerger

def calculate_interval(merge_count: int) -> float:
    """Calculate the regeneration interval for a given group size"""
    if merge_count <= 2:
        return 1.0
    
    base_interval = 1.5
    log_offset = 2.0
    power_factor = 1.8
    
    interval = base_interval * math.pow(math.log(merge_count + log_offset), power_factor)
    size_stability_factor = 1.0 + (merge_count / 100.0) * 0.5
    interval = interval * size_stability_factor
    interval = max(1.0, interval)
    interval = min(50.0, interval)
    
    return interval

def test_scaling_function():
    """Test and visualize the scaling regeneration function"""
    
    print("=" * 70)
    print("Scaling Function for Group Field Regeneration")
    print("=" * 70)
    
    merger = MultiDimensionalMerger()
    
    # Test specific sizes
    test_sizes = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    
    print("\nRegeneration Intervals by Group Size:")
    print("-" * 70)
    print(f"{'Size':<8} {'Interval':<12} {'Frequency %':<15} {'Regenerates?':<15}")
    print("-" * 70)
    
    for size in test_sizes:
        interval = calculate_interval(size)
        frequency = (1.0 / interval) * 100 if interval > 0 else 100
        regenerates = merger._should_regenerate_based_on_size(size)
        
        print(f"{size:<8} {interval:<12.2f} {frequency:<15.1f} {str(regenerates):<15}")
    
    # Analyze regeneration patterns for ranges
    print("\n" + "=" * 70)
    print("Regeneration Frequency Analysis by Size Range:")
    print("-" * 70)
    
    ranges = [
        (1, 5, "Very small"),
        (6, 20, "Small"),
        (21, 50, "Medium"),
        (51, 100, "Large"),
        (101, 200, "Very large"),
        (201, 500, "Massive")
    ]
    
    for start, end, label in ranges:
        regenerations = sum(1 for i in range(start, end + 1) 
                          if merger._should_regenerate_based_on_size(i))
        total = end - start + 1
        percentage = (regenerations / total) * 100
        
        # Calculate average interval for this range
        avg_interval = sum(calculate_interval(i) for i in range(start, end + 1)) / total
        
        print(f"{label:<12} groups (size {start:3d}-{end:3d}):")
        print(f"  Regenerates: {regenerations:3d}/{total:3d} times ({percentage:5.1f}%)")
        print(f"  Avg interval: {avg_interval:.1f} events")
    
    # Show the smoothness of the scaling
    print("\n" + "=" * 70)
    print("Scaling Function Smoothness (size 1-50):")
    print("-" * 70)
    
    # Create a simple ASCII visualization
    max_bar_width = 50
    for size in range(1, 51, 2):  # Every other size for clarity
        interval = calculate_interval(size)
        regenerates = merger._should_regenerate_based_on_size(size)
        
        # Calculate bar width (inverse of interval for visualization)
        bar_width = int((1.0 / interval) * max_bar_width)
        bar = "█" * bar_width
        
        marker = "●" if regenerates else "○"
        print(f"{size:3d} {marker} {bar} {interval:.1f}")
    
    print("\n" + "=" * 70)
    print("Key Properties of the Scaling Function:")
    print("-" * 70)
    print("• Smooth logarithmic scaling - no abrupt transitions")
    print("• Very small groups (1-5): High regeneration rate for rapid evolution")
    print("• Progressive stability: Larger groups regenerate less frequently")
    print("• Size stability factor: Additional scaling for very large groups")
    print("• Maximum cap at 50 events to ensure eventual updates")
    print("• Continuous function allows fine-tuning via parameters")
    print("=" * 70)

if __name__ == "__main__":
    test_scaling_function()
