"""
Test script to process YouTube video and query it.
This tests the SHARINGAN system end-to-end.
"""

import sys
from pathlib import Path

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.processor import VideoProcessor
from sharingan.ui.gradio_app import download_youtube_video

def main():
    youtube_url = "https://www.youtube.com/watch?v=xFsAooD_Qy8"
    
    print("=" * 80)
    print("SHARINGAN YouTube Video Test")
    print("=" * 80)
    print(f"\n📹 YouTube URL: {youtube_url}\n")
    
    # Step 1: Download video
    print("Step 1: Downloading YouTube video...")
    print("-" * 80)
    success, video_path, error = download_youtube_video(youtube_url)
    
    if not success:
        print(f"❌ Download failed: {error}")
        return
    
    print(f"✅ Video downloaded: {video_path}\n")
    
    # Step 2: Initialize processor
    print("Step 2: Initializing SHARINGAN processor...")
    print("-" * 80)
    processor = VideoProcessor(
        vlm_model='clip',
        device='auto',
        target_fps=2.0,  # Use Fast Processing preset
        enable_temporal=True,
        batch_size=32
    )
    print("✅ Processor initialized\n")
    
    # Step 3: Process video
    print("Step 3: Processing video (this may take a while)...")
    print("-" * 80)
    try:
        results = processor.process(video_path)
        
        print("✅ Video processed successfully!\n")
        print("📊 Processing Results:")
        print("-" * 80)
        
        video_info = results.get('video_info', {})
        print(f"Duration: {video_info.get('duration', 0):.1f} seconds")
        print(f"Total Frames: {video_info.get('total_frames', 0):,}")
        print(f"Processed Frames: {video_info.get('processed_frames', 0):,}")
        print(f"FPS: {video_info.get('fps', 0):.1f}")
        print(f"Events Detected: {len(results.get('events', []))}")
        
        # Show first few events
        events = results.get('events', [])
        if events:
            print(f"\n📝 First 5 Events:")
            print("-" * 80)
            for i, event in enumerate(events[:5], 1):
                timestamp = event.get('timestamp', 0)
                description = event.get('description', 'No description')
                print(f"{i}. [{timestamp:.1f}s] {description[:80]}")
        
        print()
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test queries
    print("\nStep 4: Testing queries...")
    print("-" * 80)
    
    test_queries = [
        "What is happening in this video?",
        "Summarize the main events",
        "What objects are visible?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        try:
            response = processor.chat(query, use_llm=False)
            
            if isinstance(response, dict):
                answer = response.get('answer', str(response))
                print(f"💬 Answer: {answer[:200]}")
                if len(answer) > 200:
                    print("   ...")
            else:
                print(f"💬 Answer: {str(response)[:200]}")
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)
    print("\n💡 Tip: Open http://127.0.0.1:7860 in your browser to use the full UI")
    print()

if __name__ == "__main__":
    main()
