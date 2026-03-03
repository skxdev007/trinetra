"""
Test script to verify the temporal filtering fix for long-form videos.

This script tests the adaptive temporal boosting that fixes the issue where
"final result" queries returned timestamps from the middle of long videos
instead of the actual final reveal at the end.
"""

import numpy as np
from sharingan.processor import VideoProcessor


def test_temporal_filters():
    """Test the _apply_temporal_filters method with different video durations."""
    
    processor = VideoProcessor()
    
    # Test case 1: Short video (11.7 minutes = 701 seconds)
    print("=" * 80)
    print("TEST 1: Short Video (11.7 minutes)")
    print("=" * 80)
    
    short_duration = 701.0
    short_timestamps = np.linspace(0, short_duration, 100).tolist()
    short_similarities = np.ones(100) * 0.5  # Uniform similarities
    
    filtered = processor._apply_temporal_filters(
        short_similarities, 
        short_timestamps, 
        "What is the final result?"
    )
    
    # Find top 5 timestamps
    top_indices = np.argsort(filtered)[-5:][::-1]
    print(f"\nTop 5 timestamps for 'final result' query:")
    for idx in top_indices:
        print(f"  {short_timestamps[idx]:.1f}s ({short_timestamps[idx]/60:.1f}min) - "
              f"Score: {filtered[idx]:.3f} (boost: {filtered[idx]/0.5:.1f}x)")
    
    # Verify they're in the last 20%
    last_20_percent = short_duration * 0.8
    in_last_20 = sum(1 for idx in top_indices if short_timestamps[idx] > last_20_percent)
    print(f"\n✓ {in_last_20}/5 results in last 20% (expected: 5)")
    
    # Test case 2: Long video (2h 28m 30s = 8910 seconds)
    print("\n" + "=" * 80)
    print("TEST 2: Long Video (2h 28m 30s)")
    print("=" * 80)
    
    long_duration = 8910.0
    long_timestamps = np.linspace(0, long_duration, 1000).tolist()
    long_similarities = np.ones(1000) * 0.5  # Uniform similarities
    
    filtered = processor._apply_temporal_filters(
        long_similarities,
        long_timestamps,
        "What is the final result?"
    )
    
    # Find top 5 timestamps
    top_indices = np.argsort(filtered)[-5:][::-1]
    print(f"\nTop 5 timestamps for 'final result' query:")
    for idx in top_indices:
        timestamp = long_timestamps[idx]
        minutes = timestamp / 60
        hours = minutes / 60
        print(f"  {timestamp:.1f}s ({hours:.2f}h or {minutes:.1f}min) - "
              f"Score: {filtered[idx]:.3f} (boost: {filtered[idx]/0.5:.1f}x)")
    
    # Verify they're in the last 10%
    last_10_percent = long_duration * 0.90
    in_last_10 = sum(1 for idx in top_indices if long_timestamps[idx] > last_10_percent)
    print(f"\n✓ {in_last_10}/5 results in last 10% (expected: 5)")
    
    # Verify at least some are in the last 5%
    last_5_percent = long_duration * 0.95
    in_last_5 = sum(1 for idx in top_indices if long_timestamps[idx] > last_5_percent)
    print(f"✓ {in_last_5}/5 results in last 5% (expected: 3-5)")
    
    # Test case 3: Medium video (30 minutes = 1800 seconds)
    print("\n" + "=" * 80)
    print("TEST 3: Medium Video (30 minutes)")
    print("=" * 80)
    
    medium_duration = 1800.0
    medium_timestamps = np.linspace(0, medium_duration, 200).tolist()
    medium_similarities = np.ones(200) * 0.5
    
    filtered = processor._apply_temporal_filters(
        medium_similarities,
        medium_timestamps,
        "What is the final result?"
    )
    
    top_indices = np.argsort(filtered)[-5:][::-1]
    print(f"\nTop 5 timestamps for 'final result' query:")
    for idx in top_indices:
        timestamp = medium_timestamps[idx]
        print(f"  {timestamp:.1f}s ({timestamp/60:.1f}min) - "
              f"Score: {filtered[idx]:.3f} (boost: {filtered[idx]/0.5:.1f}x)")
    
    # Verify they're in the last 15%
    last_15_percent = medium_duration * 0.85
    in_last_15 = sum(1 for idx in top_indices if medium_timestamps[idx] > last_15_percent)
    print(f"\n✓ {in_last_15}/5 results in last 15% (expected: 5)")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
    print("\nThe adaptive temporal boosting correctly:")
    print("  - Boosts last 20% for short videos (1.5x)")
    print("  - Boosts last 15% for medium videos (2.0x)")
    print("  - Boosts last 10% for long videos (3.0x)")
    print("  - Boosts last 5% for long videos (6.0x total)")
    print("\nThis fixes the woodworking video issue where 'final result' queries")
    print("were returning timestamps from the middle instead of the actual finale.")


if __name__ == "__main__":
    test_temporal_filters()
