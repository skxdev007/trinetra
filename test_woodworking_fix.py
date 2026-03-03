"""
Test the temporal filtering fix on the actual woodworking video.

This script will:
1. Download the "$18,000 Table" video (if not already cached)
2. Process it with SHARINGAN
3. Run the "final result" query
4. Verify timestamps are in the last 5-10% of the video
"""

import sys
from pathlib import Path
from sharingan.processor import VideoProcessor


def test_woodworking_video():
    """Test the woodworking video with temporal filtering fix."""
    
    print("\n" + "="*80)
    print("TESTING TEMPORAL FIX ON WOODWORKING VIDEO")
    print("="*80)
    
    video_url = "https://www.youtube.com/watch?v=1iG1sXaYhwY"
    print(f"\nVideo: $18,000 Table by Blacktail Studio")
    print(f"URL: {video_url}")
    
    # Check if video is already downloaded
    video_path = None
    possible_paths = [
        r"C:\Users\KHAVIN S\AppData\Local\Temp\sharingan_youtube_cache\$18,000 Table-1iG1sXaYhwY.mp4",
        "egoschema_videos/1iG1sXaYhwY.mp4",
        "videos/1iG1sXaYhwY.mp4",
        "1iG1sXaYhwY.mp4"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            video_path = path
            print(f"\n✓ Found cached video: {video_path}")
            break
    
    if not video_path:
        print("\n⚠️  Video not found locally.")
        print("Please download the video first using:")
        print(f"  yt-dlp -f 'bestvideo[height<=720]+bestaudio/best[height<=720]' {video_url}")
        print("\nOr run the full stress test:")
        print("  python test_complex_videos.py")
        return
    
    # Initialize processor
    print("\n" + "-"*80)
    print("Initializing SHARINGAN processor...")
    print("-"*80)
    
    processor = VideoProcessor(
        vlm_model='clip',
        device='auto',
        target_fps=5.0,
        enable_temporal=True,
        batch_size=32
    )
    
    # Process video
    print("\n" + "-"*80)
    print("Processing video (this may take a while for long videos)...")
    print("-"*80)
    
    results = processor.process(video_path)
    
    # Get video duration from timestamps if video_info is None (cached)
    if results['video_info'] is None:
        video_duration = max(processor.timestamps) if processor.timestamps else 0
        print(f"\n✓ Video loaded from cache!")
        print(f"  Duration: {video_duration:.1f}s ({video_duration/60:.1f} min or {video_duration/3600:.2f} hours)")
        print(f"  Frames processed: {len(processor.timestamps)}")
        print(f"  Events detected: {len(results['events'])}")
    else:
        video_duration = results['video_info']['duration']
        print(f"\n✓ Video processed successfully!")
        print(f"  Duration: {video_duration:.1f}s ({video_duration/60:.1f} min or {video_duration/3600:.2f} hours)")
        print(f"  Frames processed: {results['video_info']['processed_frames']}")
        print(f"  Events detected: {len(results['events'])}")
    
    # Test the critical query
    print("\n" + "="*80)
    print("TESTING CRITICAL QUERY: 'What is the final result?'")
    print("="*80)
    
    query = "What is the final result?"
    matches = processor.query(query, top_k=5)
    
    print(f"\nTop 5 timestamps for '{query}':")
    print("-"*80)
    
    for i, match in enumerate(matches, 1):
        timestamp = match['timestamp']
        percentage = (timestamp / video_duration) * 100
        hours = timestamp / 3600
        minutes = timestamp / 60
        
        print(f"{i}. {timestamp:.1f}s ({hours:.2f}h or {minutes:.1f}min) - "
              f"{percentage:.1f}% into video - "
              f"Confidence: {match['confidence']:.3f}")
    
    # Verify results
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    last_10_percent = video_duration * 0.90
    last_5_percent = video_duration * 0.95
    
    in_last_10 = sum(1 for m in matches if m['timestamp'] > last_10_percent)
    in_last_5 = sum(1 for m in matches if m['timestamp'] > last_5_percent)
    
    print(f"\nVideo duration: {video_duration:.1f}s")
    print(f"Last 10% starts at: {last_10_percent:.1f}s ({last_10_percent/60:.1f}min)")
    print(f"Last 5% starts at: {last_5_percent:.1f}s ({last_5_percent/60:.1f}min)")
    
    print(f"\nResults in last 10%: {in_last_10}/5")
    print(f"Results in last 5%: {in_last_5}/5")
    
    if video_duration >= 3600:  # Long video (>1 hour)
        if in_last_10 >= 4:
            print("\n✅ PASS: Temporal filtering working correctly for long video!")
            print("   Most results are in the last 10% where finales typically occur.")
        else:
            print("\n❌ FAIL: Results not concentrated in last 10%")
            print("   Expected at least 4/5 results in last 10% for long videos.")
    else:
        print("\n⚠️  Video is shorter than 1 hour - different thresholds apply")
    
    # Show expected vs actual
    print("\n" + "="*80)
    print("EXPECTED VS ACTUAL")
    print("="*80)
    
    print("\nBased on user verification:")
    print("  Expected final reveal: ~8910s (2h 28m 30s, 98.5% into video)")
    print(f"  Actual top result: {matches[0]['timestamp']:.1f}s "
          f"({matches[0]['timestamp']/3600:.2f}h, "
          f"{(matches[0]['timestamp']/video_duration)*100:.1f}% into video)")
    
    if matches[0]['timestamp'] > video_duration * 0.90:
        print("\n✅ Top result is in the last 10% - Fix is working!")
    else:
        print("\n⚠️  Top result is NOT in the last 10% - May need adjustment")


if __name__ == "__main__":
    try:
        test_woodworking_video()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
