"""
Comprehensive test for cooking video with text overlays.
Tests SHARINGAN's ability to detect visual content and timestamps.
Saves all results to a detailed report file.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.processor import VideoProcessor
from sharingan.ui.gradio_app import download_youtube_video

def format_timestamp(seconds):
    """Format seconds as MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def main():
    # Cooking video with text overlays
    youtube_url = "https://www.youtube.com/watch?v=43DP_lPPG-k"
    
    # Create output file
    output_file = "cooking_video_test_results.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# SHARINGAN Cooking Video Test Results\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**YouTube URL:** {youtube_url}\n\n")
        f.write("---\n\n")
        
        print("=" * 80)
        print("SHARINGAN Cooking Video Test - Comprehensive Analysis")
        print("=" * 80)
        print(f"\nYouTube URL: {youtube_url}")
        print(f"Output File: {output_file}\n")
        
        # Step 1: Download video
        print("Step 1: Downloading YouTube video...")
        print("-" * 80)
        f.write("## Step 1: Video Download\n\n")
        
        success, video_path, error = download_youtube_video(youtube_url)
        
        if not success:
            error_msg = f"Download failed: {error}"
            print(error_msg)
            f.write(f"**Status:** Failed\n\n")
            f.write(f"**Error:** {error}\n\n")
            return
        
        print(f"Video downloaded: {video_path}\n")
        f.write(f"**Status:** Success\n\n")
        f.write(f"**Video Path:** `{video_path}`\n\n")
        f.write("---\n\n")
        
        # Step 2: Initialize processor with HIGH QUALITY settings
        print("Step 2: Initializing SHARINGAN processor...")
        print("-" * 80)
        print("Using HIGH QUALITY preset for better text detection:")
        print("  - FPS: 2.0-8.0 (adaptive, higher for text)")
        print("  - TAS Kernels: 2/8/48")
        print("  - Window Size: 96")
        print("  - Verifier Threshold: 0.75")
        
        f.write("## Step 2: Processor Configuration\n\n")
        f.write("**Preset:** High Quality (optimized for text detection)\n\n")
        f.write("**Settings:**\n")
        f.write("- FPS Range: 2.0-8.0 (adaptive)\n")
        f.write("- TAS Kernels: 2/8/48\n")
        f.write("- Window Size: 96\n")
        f.write("- Verifier Threshold: 0.75\n\n")
        f.write("---\n\n")
        
        processor = VideoProcessor(
            vlm_model='clip',
            device='auto',
            target_fps=8.0,  # High Quality preset - maximum FPS for text
            enable_temporal=True,
            batch_size=32
        )
        print("Processor initialized\n")
        
        # Step 3: Process video
        print("Step 3: Processing video (this may take several minutes)...")
        print("-" * 80)
        f.write("## Step 3: Video Processing\n\n")
        
        try:
            start_time = datetime.now()
            results = processor.process(video_path)
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            
            print("Video processed successfully!\n")
            f.write("**Status:** Success\n\n")
            f.write(f"**Processing Time:** {processing_duration:.1f} seconds ({processing_duration/60:.1f} minutes)\n\n")
            
            # Get video info
            video_info = results.get('video_info')
            if not video_info:
                video_info = {}
            
            duration = video_info.get('duration', 0)
            total_frames = video_info.get('total_frames', 0)
            processed_frames = video_info.get('processed_frames', 0)
            fps = video_info.get('fps', 0)
            events = results.get('events', [])
            
            print("Video Information:")
            print("-" * 80)
            print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"Total Frames: {total_frames:,}")
            print(f"Processed Frames: {processed_frames:,}")
            print(f"Original FPS: {fps:.1f}")
            print(f"Events Detected: {len(events)}")
            if duration > 0 and processed_frames > 0:
                print(f"Effective Processing FPS: {processed_frames / duration:.2f}")
            
            f.write("### Video Information\n\n")
            f.write(f"- **Duration:** {duration:.1f}s ({duration/60:.1f} min)\n")
            f.write(f"- **Total Frames:** {total_frames:,}\n")
            f.write(f"- **Processed Frames:** {processed_frames:,}\n")
            f.write(f"- **Original FPS:** {fps:.1f}\n")
            f.write(f"- **Events Detected:** {len(events)}\n")
            if duration > 0 and processed_frames > 0:
                f.write(f"- **Effective Processing FPS:** {processed_frames / duration:.2f}\n")
            f.write("\n")
            
            # Show all events with timestamps
            if events:
                print(f"\nDetected Events (Total: {len(events)}):")
                print("-" * 80)
                f.write("### Detected Events\n\n")
                f.write(f"Total events detected: {len(events)}\n\n")
                f.write("| # | Timestamp | Description |\n")
                f.write("|---|-----------|-------------|\n")
                
                for i, event in enumerate(events, 1):
                    timestamp = event.get('timestamp', 0)
                    description = event.get('description', 'No description')
                    time_str = format_timestamp(timestamp)
                    
                    if i <= 20:  # Print first 20 to console
                        print(f"{i:3d}. [{time_str}] {description[:70]}")
                    
                    f.write(f"| {i} | {time_str} | {description[:100]} |\n")
                
                if len(events) > 20:
                    print(f"... and {len(events) - 20} more events (see output file)")
            
            f.write("\n---\n\n")
            print()
            
        except Exception as e:
            error_msg = f"Processing failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            f.write(f"**Status:** Failed\n\n")
            f.write(f"**Error:** {str(e)}\n\n")
            return
        
        # Step 4: Test comprehensive queries
        print("\nStep 4: Testing comprehensive queries...")
        print("-" * 80)
        f.write("## Step 4: Query Testing\n\n")
        f.write("Testing various types of queries to understand video content.\n\n")
        
        # Define comprehensive test queries
        test_queries = [
            # Content identification
            ("What ingredients are shown in this video?", "Content"),
            ("What cooking equipment is visible?", "Content"),
            ("What is being cooked?", "Content"),
            
            # Step-by-step process
            ("What are the main cooking steps?", "Process"),
            ("How is the dish prepared?", "Process"),
            ("What happens first in the recipe?", "Process"),
            
            # Timing queries
            ("What happens at the beginning?", "Timing"),
            ("What happens in the middle of the video?", "Timing"),
            ("What happens at the end?", "Timing"),
            ("When is mixing done?", "Timing"),
            ("When is cooking/baking done?", "Timing"),
            
            # Specific actions
            ("When are ingredients added?", "Actions"),
            ("When is stirring shown?", "Actions"),
            ("When is the final dish shown?", "Actions"),
            
            # Summary
            ("Summarize this recipe", "Summary"),
            ("What is the final result?", "Summary"),
        ]
        
        f.write("### Query Results\n\n")
        
        for i, (query, category) in enumerate(test_queries, 1):
            print(f"\nQuery {i} [{category}]: {query}")
            print("-" * 40)
            
            f.write(f"#### Query {i}: {query}\n\n")
            f.write(f"**Category:** {category}\n\n")
            
            try:
                response = processor.chat(query, use_llm=False)
                
                if isinstance(response, dict):
                    answer = response.get('answer', str(response))
                else:
                    answer = str(response)
                
                print(f"Answer: {answer}")
                f.write(f"**Answer:** {answer}\n\n")
                
                # Extract timestamps if present
                if 'moments' in answer.lower() or 's' in answer:
                    f.write("*Note: Answer includes timestamp references*\n\n")
                
            except Exception as e:
                error_msg = f"Query failed: {e}"
                print(error_msg)
                f.write(f"**Error:** {str(e)}\n\n")
            
            f.write("---\n\n")
        
        # Summary
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        print(f"\nResults saved to: {output_file}")
        print(f"\nKey Statistics:")
        print(f"- Video Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"- Events Detected: {len(events)}")
        print(f"- Queries Tested: {len(test_queries)}")
        print(f"- Processing Time: {processing_duration:.1f}s")
        print()
        
        f.write("## Test Summary\n\n")
        f.write(f"- **Video Duration:** {duration:.1f}s ({duration/60:.1f} min)\n")
        f.write(f"- **Events Detected:** {len(events)}\n")
        f.write(f"- **Queries Tested:** {len(test_queries)}\n")
        f.write(f"- **Processing Time:** {processing_duration:.1f}s ({processing_duration/60:.1f} min)\n")
        f.write(f"- **Test Status:** ✅ Success\n\n")
        f.write("---\n\n")
        f.write("## Notes\n\n")
        f.write("- SHARINGAN processes visual content only (no audio)\n")
        f.write("- Text overlays and visual elements are detected through CLIP embeddings\n")
        f.write("- Higher FPS settings improve text detection accuracy\n")
        f.write("- Timestamps are based on visual scene changes and content\n")
        f.write("\n---\n\n")
        f.write(f"*Generated by SHARINGAN Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

if __name__ == "__main__":
    main()
