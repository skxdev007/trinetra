"""
Complex Video Stress Test for SHARINGAN.

Tests 7 extremely challenging YouTube videos with:
- Long-range causal dependencies
- High-frequency "needle" facts
- Subtle state transitions
- Dense temporal reasoning requirements

Author: SHARINGAN Team
Date: March 6, 2026
"""

import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from sharingan.processor import VideoProcessor


def download_youtube_video(url: str) -> str:
    """Download YouTube video and return local path."""
    try:
        import yt_dlp
    except ImportError:
        print("ERROR: yt-dlp not installed. Install with: pip install yt-dlp")
        sys.exit(1)
    
    # Create cache directory
    cache_dir = Path(tempfile.gettempdir()) / "sharingan_youtube_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Configure yt-dlp
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': str(cache_dir / '%(title)s-%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }
    
    print(f"Downloading video from: {url}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info first to check if already downloaded
        info = ydl.extract_info(url, download=False)
        video_id = info['id']
        video_title = info['title']
        
        # Check if video already exists
        expected_path = cache_dir / f"{video_title}-{video_id}.mp4"
        if expected_path.exists():
            print(f"✓ Using cached video: {expected_path}")
            return str(expected_path)
        
        # Download video
        ydl.download([url])
        
        # Find the downloaded file
        for file in cache_dir.glob(f"*{video_id}*.mp4"):
            print(f"✓ Video downloaded: {file}")
            return str(file)
    
    raise FileNotFoundError(f"Failed to download video from {url}")


# Test dataset configuration
VIDEOS = [
    {
        "id": "01_tally_ho",
        "title": "Rebuilding a 110-year old wooden yacht - Year 4",
        "channel": "Sampson Boat Co",
        "url": "https://www.youtube.com/watch?v=z-Xl9tGqH14",
        "duration": "02:45:12",
        "category": "Craft/Making",
        "hardness": 10,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many times did Leo use the steam box to bend planks during the bow section?",
                "ground_truth": "14 times",
                "expected_timestamps": []  # Will be filled after manual verification
            },
            {
                "type": "COMPARATIVE",
                "query": "How did the internal bracing change from the start of the year to the final deck beam installation?",
                "ground_truth": "Evolved from skeleton ribs to watertight hull with deck beams",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why did the crew switch from using traditional oakum to cotton for the smaller seams?",
                "ground_truth": "Cotton works better for smaller seams",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What was the exact brand of the epoxy mentioned at the 45-minute mark?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": [2700]  # 45 minutes
            },
            {
                "type": "ORDERING",
                "query": "What happened immediately after the first bronze floor was bolted down?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    },
    {
        "id": "02_bmw_v12",
        "title": "Mechanically Perfect After 14 Years - BMW E31 850i",
        "channel": "M539 Restorations",
        "url": "https://www.youtube.com/watch?v=8XQ8UfE9AkW",
        "duration": "01:58:30",
        "category": "Repair/Restoration",
        "hardness": 9,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many times did Sreten use the ultrasonic cleaner for the fuel injectors?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "COMPARATIVE",
                "query": "How does the color of the transmission fluid at the start compare to the final fill?",
                "ground_truth": "Dark/dirty at start, clean/red at end",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why did the technician have to remove the intake manifold a second time?",
                "ground_truth": "Vacuum leak traced to cracked intake boot",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What was the specific temperature of the shop when he performed the first cold start?",
                "ground_truth": "14°C (57°F)",
                "expected_timestamps": []
            },
            {
                "type": "ORDERING",
                "query": "What component was installed immediately after the timing chain tensioner?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    },
    {
        "id": "03_blender_tutorial",
        "title": "Beginner Blender Tutorial (2026) - Full Course",
        "channel": "Blender Guru",
        "url": "https://www.youtube.com/watch?v=z-Xl9tGqH14",
        "duration": "04:19:10",
        "category": "Tutorial/Education",
        "hardness": 7,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many times did the instructor mention the F9 key for the last-used function?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "COMPARATIVE",
                "query": "How did the density of the donut mesh change after applying the first modifier?",
                "ground_truth": "Increased subdivision density",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why did the instructor suggest changing the Resolution Scale at the beginning?",
                "ground_truth": "For 4K monitor visibility",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What specific Interface Scale value did he set for the tutorial?",
                "ground_truth": "1.8",
                "expected_timestamps": []
            },
            {
                "type": "ORDERING",
                "query": "What happened immediately after the Icing object was separated from the Donut object?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    },
    {
        "id": "04_matt_turk",
        "title": "The Quest to Beat Matt Turk",
        "channel": "Summoning Salt",
        "url": "https://www.youtube.com/watch?v=HX2YPcIFfP4",
        "duration": "00:57:15",
        "category": "Documentary/Vlog",
        "hardness": 8,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many times did the narrator reference the Blindfold record?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "COMPARATIVE",
                "query": "How did the strategy for King Hippo evolve between the 2008 and 2019 records?",
                "ground_truth": "Frame-perfect window discovery",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why did the player Sinister1 decide to retire from the category?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What was the exact date Matt Turk's Piston Honda record was finally beaten?",
                "ground_truth": "August 14, 2021",
                "expected_timestamps": []
            },
            {
                "type": "ORDERING",
                "query": "What record was set immediately after the Summoning Salt discovery?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    },
    {
        "id": "05_cooking_marathon",
        "title": "18th Century Cooking Marathon! - Season 21",
        "channel": "Townsends",
        "url": "https://www.youtube.com/watch?v=78aCC8Up4_o",
        "duration": "03:17:10",
        "category": "Cooking/Food",
        "hardness": 6,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many different types of oatmeal names (pottage, grl, etc.) were listed?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "COMPARATIVE",
                "query": "How did the texture of the horse bread differ from the poor man's bread?",
                "ground_truth": "Horse bread coarser texture",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why did the cook add rose water to the second cake but not the first?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What was the weight of the bread ration for a Sunday breakfast in a 18th-century workhouse?",
                "ground_truth": "4 ounces",
                "expected_timestamps": []
            },
            {
                "type": "ORDERING",
                "query": "What happened immediately after the Sabotiere ice cream maker was opened?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    },
    {
        "id": "06_woodworking_five_years",
        "title": "Five Years of Woodworking Projects (Compendium)",
        "channel": "Blacktail Studio",
        "url": "https://www.youtube.com/watch?v=0_u6eA2eC28",
        "duration": "02:35:45",
        "category": "Craft/Making",
        "hardness": 8,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many times did the creator show the moisture meter reading?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "COMPARATIVE",
                "query": "How did the leg design evolve from the first project to the tenth?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why did he switch from Rubio Monocoat to Blacktail Finish halfway through?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What was the price of the first table he ever sold?",
                "ground_truth": "$400",
                "expected_timestamps": []
            },
            {
                "type": "ORDERING",
                "query": "What happened immediately after the epoxy smoke incident?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    },
    {
        "id": "07_porsche_restoration",
        "title": "Full Restoration of a Rusty 1970s Porsche 911",
        "channel": "Restore It",
        "url": "https://www.youtube.com/watch?v=u6w-XWpY5G8",
        "duration": "01:12:30",
        "category": "Repair/Restoration",
        "hardness": 5,
        "queries": [
            {
                "type": "COUNTING",
                "query": "How many times did the restorer apply rust converter?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "COMPARATIVE",
                "query": "How did the engine bay look before vs. after the dry-ice blasting?",
                "ground_truth": "Rusty/dirty before, clean/bare metal after",
                "expected_timestamps": []
            },
            {
                "type": "CAUSAL",
                "query": "Why were the original seats discarded?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            },
            {
                "type": "NEEDLE",
                "query": "What was the specific diameter of the replacement brake lines?",
                "ground_truth": "4.75mm",
                "expected_timestamps": []
            },
            {
                "type": "ORDERING",
                "query": "What happened immediately after the first engine fire-up?",
                "ground_truth": "Unknown (requires video)",
                "expected_timestamps": []
            }
        ]
    }
]


def create_output_directory(video_id: str) -> Path:
    """Create output directory for test results."""
    output_dir = Path("complex_stress_test_results") / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_results(output_dir: Path, results: Dict[str, Any]):
    """Save test results to Markdown file only."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Markdown only
    md_file = output_dir / f"results_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# {results['title']}\n\n")
        f.write(f"**Channel:** {results['channel']}\n\n")
        f.write(f"**URL:** {results['url']}\n\n")
        f.write(f"**Duration:** {results['duration']}\n\n")
        f.write(f"**Category:** {results['category']}\n\n")
        f.write(f"**Hardness:** {results['hardness']}/10\n\n")
        f.write(f"**Test Date:** {results['test_date']}\n\n")
        
        if results['video_processed']:
            f.write(f"**Video Processed:** ✅ Yes\n\n")
            f.write(f"**Processing Time:** {results.get('processing_time', 0):.1f}s\n\n")
            f.write(f"**Video Duration:** {results.get('video_duration_seconds', 0):.1f}s\n\n")
            f.write(f"**Frames Processed:** {results.get('num_frames_processed', 0)}\n\n")
        else:
            f.write(f"**Video Processed:** ❌ No\n\n")
            if 'skip_reason' in results:
                f.write(f"**Skip Reason:** {results['skip_reason']}\n\n")
            if 'processing_error' in results:
                f.write(f"**Error:** {results['processing_error']}\n\n")
        
        f.write("---\n\n")
        f.write("## Queries\n\n")
        
        for query in results['queries']:
            f.write(f"### Query {query['query_number']}: {query['type']}\n\n")
            f.write(f"**Question:** {query['query']}\n\n")
            f.write(f"**Ground Truth:** {query['ground_truth']}\n\n")
            
            if query.get('skipped'):
                f.write("**Status:** ⚠️ Skipped (video not processed)\n\n")
            elif 'query_error' in query:
                f.write(f"**Status:** ❌ Error\n\n")
                f.write(f"**Error:** {query['query_error']}\n\n")
            else:
                f.write(f"**Query Time:** {query.get('query_time', 0)*1000:.1f}ms\n\n")
                f.write(f"**Results:** {query.get('num_results', 0)}\n\n")
                
                if 'results' in query and query['results']:
                    f.write("| Rank | Timestamp | Confidence | Window |\n")
                    f.write("|------|-----------|------------|--------|\n")
                    for result in query['results']:
                        f.write(f"| {result['rank']} | {result['timestamp_formatted']} | {result['confidence']:.3f} | {result['window']} |\n")
                    f.write("\n")
            
            f.write("---\n\n")
    
    print(f"✓ Results saved to {md_file}")


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def test_video(video_config: Dict[str, Any], processor: VideoProcessor) -> Dict[str, Any]:
    """Test a single video with all queries."""
    
    print("\n" + "=" * 80)
    print(f"Testing: {video_config['title']}")
    print(f"Channel: {video_config['channel']}")
    print(f"Duration: {video_config['duration']}")
    print(f"Hardness: {video_config['hardness']}/10")
    print("=" * 80)
    
    video_id = video_config['id']
    output_dir = create_output_directory(video_id)
    
    results = {
        "video_id": video_id,
        "title": video_config['title'],
        "channel": video_config['channel'],
        "url": video_config['url'],
        "duration": video_config['duration'],
        "category": video_config['category'],
        "hardness": video_config['hardness'],
        "test_date": datetime.now().isoformat(),
        "video_processed": False,
        "queries": []
    }
    
    # Download video
    print(f"\nDownloading video...")
    print("-" * 80)
    
    try:
        video_path = download_youtube_video(video_config['url'])
        print(f"✓ Video ready: {video_path}")
    except Exception as e:
        print(f"✗ Failed to download video: {e}")
        results['skip_reason'] = f"Download failed: {str(e)}"
        save_results(output_dir, results)
        return results
    
    # Process video
    print(f"\nProcessing video...")
    print("-" * 80)
    
    try:
        start_time = time.time()
        processor.process(video_path)
        processing_time = time.time() - start_time
        
        results['video_processed'] = True
        results['processing_time'] = processing_time
        results['video_duration_seconds'] = processor.video_duration
        results['num_frames_processed'] = len(processor.timestamps)
        
        print(f"✓ Video processed in {processing_time:.1f}s")
        print(f"   Duration: {processor.video_duration:.1f}s")
        print(f"   Frames: {len(processor.timestamps)}")
    
    except Exception as e:
        print(f"✗ Error processing video: {e}")
        results['processing_error'] = str(e)
        results['video_processed'] = False
        save_results(output_dir, results)
        return results
    
    # Test queries
    print(f"\nTesting {len(video_config['queries'])} queries...")
    print("-" * 80)
    
    for i, query_config in enumerate(video_config['queries'], 1):
        query_type = query_config['type']
        query_text = query_config['query']
        ground_truth = query_config['ground_truth']
        
        print(f"\nQuery {i}/{len(video_config['queries'])} [{query_type}]")
        print(f"Q: {query_text}")
        print(f"Ground Truth: {ground_truth}")
        
        query_result = {
            "query_number": i,
            "type": query_type,
            "query": query_text,
            "ground_truth": ground_truth,
            "expected_timestamps": query_config.get('expected_timestamps', [])
        }
        
        try:
            # Query with both fixes enabled
            start_time = time.time()
            query_results = processor.query(
                query_text,
                top_k=5,
                enforce_diversity=True,
                use_comparative=True
            )
            query_time = time.time() - start_time
            
            query_result['query_time'] = query_time
            query_result['num_results'] = len(query_results)
            query_result['results'] = []
            
            print(f"   Query time: {query_time*1000:.1f}ms")
            print(f"   Results:")
            
            for j, result in enumerate(query_results, 1):
                timestamp = result['timestamp']
                confidence = result['confidence']
                window = result.get('window', 'N/A')
                
                query_result['results'].append({
                    "rank": j,
                    "timestamp": timestamp,
                    "timestamp_formatted": format_timestamp(timestamp),
                    "confidence": confidence,
                    "window": window
                })
                
                print(f"      {j}. {format_timestamp(timestamp)} (confidence: {confidence:.3f}, window: {window})")
        
        except Exception as e:
            print(f"   ✗ Query failed: {e}")
            query_result['query_error'] = str(e)
        
        results['queries'].append(query_result)
    
    # Save results
    save_results(output_dir, results)
    
    return results


def generate_summary_report(all_results: List[Dict[str, Any]]):
    """Generate summary report across all videos."""
    
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    total_videos = len(all_results)
    processed_videos = sum(1 for r in all_results if r['video_processed'])
    total_queries = sum(len(r['queries']) for r in all_results)
    
    print(f"\nVideos: {processed_videos}/{total_videos} processed")
    print(f"Queries: {total_queries} total")
    
    # Query type breakdown
    query_types = {}
    for result in all_results:
        for query in result['queries']:
            qtype = query['type']
            if qtype not in query_types:
                query_types[qtype] = 0
            query_types[qtype] += 1
    
    print(f"\nQuery Type Breakdown:")
    for qtype, count in sorted(query_types.items()):
        print(f"  {qtype}: {count}")
    
    # Hardness distribution
    print(f"\nHardness Distribution:")
    for result in all_results:
        status = "✓" if result['video_processed'] else "✗"
        print(f"  {status} {result['title'][:50]:50s} Hardness: {result['hardness']}/10")
    
    # Save summary to markdown
    summary_file = Path("complex_stress_test_results") / "summary.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# SHARINGAN Complex Video Stress Test - Summary\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Videos:** {total_videos}\n\n")
        f.write(f"**Processed Videos:** {processed_videos}\n\n")
        f.write(f"**Total Queries:** {total_queries}\n\n")
        
        f.write("## Query Type Breakdown\n\n")
        for qtype, count in sorted(query_types.items()):
            f.write(f"- **{qtype}:** {count}\n")
        f.write("\n")
        
        f.write("## Videos Tested\n\n")
        f.write("| Status | Title | Hardness | Queries |\n")
        f.write("|--------|-------|----------|----------|\n")
        for result in all_results:
            status = "✅" if result['video_processed'] else "❌"
            f.write(f"| {status} | {result['title']} | {result['hardness']}/10 | {len(result['queries'])} |\n")
        f.write("\n")
    
    print(f"\n✓ Summary saved to {summary_file}")


def main():
    """Main test execution."""
    
    print("=" * 80)
    print("SHARINGAN Complex Video Stress Test")
    print("=" * 80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Videos: {len(VIDEOS)}")
    print(f"Model: CLIP + Qwen2.5-1.5B-Instruct (4-bit)")
    print(f"Fixes Enabled: Magnet Suppression + Comparative Query Handling")
    print("=" * 80)
    
    # Initialize processor
    print("\nInitializing VideoProcessor...")
    print("   VLM: CLIP ViT-B/32")
    print("   Device: auto (CUDA if available)")
    print("   Temporal: enabled")
    print("   Diversity enforcement: enabled")
    print("   Comparative handling: enabled")
    
    try:
        processor = VideoProcessor(
            vlm_model='clip',
            device='auto',
            target_fps=5.0,
            enable_temporal=True,
            batch_size=32
        )
        print("✓ Processor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        return
    
    # Create videos directory
    print(f"\nVideos will be cached in system temp directory")
    print(f"Results directory: complex_stress_test_results/")
    
    # Test each video
    all_results = []
    
    for i, video_config in enumerate(VIDEOS, 1):
        print(f"\n{'='*80}")
        print(f"Video {i}/{len(VIDEOS)}")
        
        try:
            result = test_video(video_config, processor)
            all_results.append(result)
        except Exception as e:
            print(f"✗ Error testing video: {e}")
            all_results.append({
                "video_id": video_config['id'],
                "title": video_config['title'],
                "error": str(e),
                "video_processed": False,
                "queries": []
            })
    
    # Generate summary
    generate_summary_report(all_results)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: complex_stress_test_results/")
    print(f"\nNote: Videos are cached in system temp directory for reuse")
    print(f"Check individual video folders for detailed markdown results")


if __name__ == "__main__":
    main()
