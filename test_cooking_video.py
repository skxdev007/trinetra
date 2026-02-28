"""
Enhanced test script for SHARINGAN with cooking video.
Tests timestamp-based queries and text recognition.
"""

import sys
from pathlib import Path

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.processor import VideoProcessor
from sharingan.ui.gradio_app import download_youtube_video

def main():
    # Using the video that already worked in our previous test
    youtube_url = "https://www.youtube.com/watch?v=xFsAooD_Qy8"  # The Rise of Skanda - already cached
    
    print("=" * 80)
    print("SHARINGAN Video Test - Timestamp-based Queries")
    print("=" * 80)
    print(f"\nYouTube URL: {youtube_url}")
    print("Video Type: Animated story with visual scenes")
    print("Note: Using cached video from previous test\n")
    
    # Step 1: Download video
    print("Step 1: Downloading YouTube video...")
    print("-" * 80)
    success, video_path, error = download_youtube_video(youtube_url)
    
    if not success:
        print(f"Download failed: {error}")
        return
    
    print(f"Video downloaded: {video_path}\n")
    
    # Step 2: Initialize processor with BALANCED settings
    print("Step 2: Initializing SHARINGAN processor...")
    print("-" * 80)
    print("Using BALANCED preset:")
    print("  - FPS: 1.0-5.0 (adaptive)")
    print("  - TAS Kernels: 2/8/32")
    print("  - Window Size: 64")
    print("  - Verifier Threshold: 0.7")
    
    processor = VideoProcessor(
        vlm_model='clip',
        device='auto',
        target_fps=5.0,  # Balanced preset - higher FPS for better text capture
        enable_temporal=True,
        batch_size=32
    )
    print("Processor initialized\n")
    
    # Step 3: Process video
    print("Step 3: Processing video...")
    print("-" * 80)
    try:
        results = processor.process(video_path)
        
        print("Video processed successfully!\n")
        print("Processing Results:")
        print("-" * 80)
        
        video_info = results.get('video_info')
        if not video_info:
            print("Warning: No video_info in results")
            video_info = {}
        
        duration = video_info.get('duration', 0)
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Total Frames: {video_info.get('total_frames', 0):,}")
        print(f"Processed Frames: {video_info.get('processed_frames', 0):,}")
        print(f"FPS: {video_info.get('fps', 0):.1f}")
        print(f"Events Detected: {len(results.get('events', []))}")
        
        # Show first few events with timestamps
        events = results.get('events', [])
        if events:
            print(f"\nFirst 10 Events (with timestamps):")
            print("-" * 80)
            for i, event in enumerate(events[:10], 1):
                timestamp = event.get('timestamp', 0)
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                description = event.get('description', 'No description')
                print(f"{i}. [{minutes:02d}:{seconds:02d}] {description[:70]}")
        
        print()
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test timestamp-based queries
    print("\nStep 4: Testing timestamp-based queries...")
    print("-" * 80)
    
    # Calculate some interesting timestamps
    quarter_time = duration / 4
    half_time = duration / 2
    three_quarter_time = 3 * duration / 4
    
    test_queries = [
        # General queries
        "What ingredients are shown?",
        "What are the main steps in this recipe?",
        "Summarize what happens in this video",
        
        # Timestamp-based queries
        f"What happens at the beginning?",
        f"What happens around {int(quarter_time)} seconds?",
        f"What happens in the middle of the video?",
        f"What happens at the end?",
        
        # Specific content queries
        "When is butter added?",
        "When are the cookies put in the oven?",
        "Show me the mixing steps",
        "What temperature is mentioned?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        try:
            response = processor.chat(query, use_llm=False)
            
            if isinstance(response, dict):
                answer = response.get('answer', str(response))
                # Extract timestamps if present
                if 'moments' in answer.lower() or 's' in answer:
                    print(f"Answer: {answer}")
                else:
                    print(f"Answer: {answer[:150]}")
                    if len(answer) > 150:
                        print("   ...")
            else:
                print(f"Answer: {str(response)[:150]}")
        except Exception as e:
            print(f"Query failed: {e}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"- Video Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"- Events Detected: {len(events)}")
    print(f"- Processed Frames: {video_info.get('processed_frames', 0):,}")
    if duration > 0:
        print(f"- Average FPS: {video_info.get('processed_frames', 0) / duration:.2f}")
    print("\nNote: SHARINGAN processes visual content only (no audio).")
    print("Text overlays and visual cues are detected through CLIP embeddings.")
    print("\nTip: Open http://127.0.0.1:7860 to use the full Gradio UI")
    print()

if __name__ == "__main__":
    main()
