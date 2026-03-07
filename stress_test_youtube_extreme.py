"""
Extreme Stress Test for SHARINGAN Video Understanding System

Tests 7 challenging YouTube videos with:
- Long-range causal dependencies
- High-frequency "needle" facts
- Subtle state transitions
- Dense temporal reasoning requirements

Results saved to: D:/PROJECTS/webstromprojects/sharingan/complex_stress_test_results/
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json
import hashlib

from sharingan.processor import VideoProcessor


# Test dataset configuration
STRESS_TEST_VIDEOS = [
    {
        "id": "01_tally_ho",
        "title": "Rebuilding Tally Ho (Year 4 Compilation)",
        "url": "https://www.youtube.com/watch?v=z-Xl9tGqH14",
        "duration": "02:45:12",
        "category": "Craft/Making",
        "hardness": 10,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many times did Leo use the steam box to bend planks during the bow section?",
                "ground_truth": "14 times",
                "why_hard": "Steam box activity looks identical to carrying wood unless tracking wood state"
            },
            {
                "type": "COMPARATIVE",
                "question": "How did the internal bracing change from the start of the year to the final deck beam installation?",
                "ground_truth": "Evolved from skeleton ribs to full watertight hull with deck beams",
                "why_hard": "Requires comparing state across 2+ hour video"
            },
            {
                "type": "CAUSAL",
                "question": "Why did the crew switch from using traditional oakum to cotton for the smaller seams?",
                "ground_truth": "Cotton works better for smaller seams due to flexibility",
                "why_hard": "Causal reasoning across long temporal gap"
            },
            {
                "type": "NEEDLE",
                "question": "What was the exact brand of the epoxy mentioned at the 45-minute mark?",
                "ground_truth": "Specific brand name mentioned once",
                "why_hard": "Single mention in 2.5+ hour video"
            },
            {
                "type": "ORDERING",
                "question": "What happened immediately after the first bronze floor was bolted down?",
                "ground_truth": "Specific next action",
                "why_hard": "Requires precise temporal ordering"
            }
        ]
    },
    {
        "id": "02_bmw_v12",
        "title": "The Ultimate BMW V12 Restoration",
        "url": "https://www.youtube.com/watch?v=8XQ8UfE9Akw",
        "duration": "01:58:30",
        "category": "Repair/Restoration",
        "hardness": 9,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many times did Sreten use the ultrasonic cleaner for the fuel injectors?",
                "ground_truth": "Specific count",
                "why_hard": "Repetitive activity across long video"
            },
            {
                "type": "COMPARATIVE",
                "question": "How does the color of the transmission fluid at the start compare to the final fill?",
                "ground_truth": "Dark/contaminated vs clean/red",
                "why_hard": "Visual comparison across 2-hour gap"
            },
            {
                "type": "CAUSAL",
                "question": "Why did the technician have to remove the intake manifold a second time?",
                "ground_truth": "Vacuum leak traced to cracked intake boot",
                "why_hard": "Causal chain spanning 90 minutes"
            },
            {
                "type": "NEEDLE",
                "question": "What was the specific temperature of the shop when he performed the first cold start?",
                "ground_truth": "14°C (57°F)",
                "why_hard": "2-second visual/audio cue in 2-hour video"
            },
            {
                "type": "ORDERING",
                "question": "What component was installed immediately after the timing chain tensioner?",
                "ground_truth": "Specific component",
                "why_hard": "Precise temporal ordering in dense assembly sequence"
            }
        ]
    },
    {
        "id": "03_blender_tutorial",
        "title": "Beginner Blender Tutorial (2026 Edition)",
        "url": "https://www.youtube.com/watch?v=z-Xl9tGqH14",
        "duration": "04:19:10",
        "category": "Tutorial/Education",
        "hardness": 7,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many times did the instructor mention the F9 key for the last-used function?",
                "ground_truth": "Specific count",
                "why_hard": "Repeated mentions across 4+ hour tutorial"
            },
            {
                "type": "COMPARATIVE",
                "question": "How did the density of the donut mesh change after applying the first modifier?",
                "ground_truth": "Increased subdivision",
                "why_hard": "Visual state comparison"
            },
            {
                "type": "CAUSAL",
                "question": "Why did the instructor suggest changing the Resolution Scale at the beginning?",
                "ground_truth": "For 4K monitor visibility",
                "why_hard": "Early setup decision affecting later workflow"
            },
            {
                "type": "NEEDLE",
                "question": "What specific Interface Scale value did he set for the tutorial?",
                "ground_truth": "1.8",
                "why_hard": "Single UI setting in 4-hour video"
            },
            {
                "type": "ORDERING",
                "question": "What happened immediately after the Icing object was separated from the Donut object?",
                "ground_truth": "Specific next action",
                "why_hard": "Precise ordering in long tutorial"
            }
        ]
    },
    {
        "id": "04_matt_turk",
        "title": "The Quest to Beat Matt Turk",
        "url": "https://www.youtube.com/watch?v=HX2YPcIFfP4",
        "duration": "00:57:15",
        "category": "Documentary/Vlog",
        "hardness": 8,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many times did the narrator reference the Blindfold record?",
                "ground_truth": "Specific count",
                "why_hard": "Verbal references across documentary"
            },
            {
                "type": "COMPARATIVE",
                "question": "How did the strategy for King Hippo evolve between the 2008 and 2019 records?",
                "ground_truth": "Frame-perfect window discovery",
                "why_hard": "Comparing strategies across decade"
            },
            {
                "type": "CAUSAL",
                "question": "Why did the player Sinister1 decide to retire from the category?",
                "ground_truth": "Specific reason mentioned",
                "why_hard": "Causal reasoning about player motivation"
            },
            {
                "type": "NEEDLE",
                "question": "What was the exact date Matt Turk's Piston Honda record was finally beaten?",
                "ground_truth": "August 14, 2021",
                "why_hard": "Single date mention in documentary"
            },
            {
                "type": "ORDERING",
                "question": "What record was set immediately after the Summoning Salt discovery?",
                "ground_truth": "Specific record",
                "why_hard": "Temporal ordering of record progression"
            }
        ]
    },
    {
        "id": "05_cooking_marathon",
        "title": "Cooking Marathon - Season 21",
        "url": "https://www.youtube.com/watch?v=78aCC8Up4_o",
        "duration": "03:17:10",
        "category": "Cooking/Food",
        "hardness": 6,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many different types of oatmeal names (pottage, gruel, etc.) were listed?",
                "ground_truth": "Specific count",
                "why_hard": "Verbal list across long video"
            },
            {
                "type": "COMPARATIVE",
                "question": "How did the texture of the horse bread differ from the poor man's bread?",
                "ground_truth": "Coarser vs finer texture",
                "why_hard": "Visual/verbal comparison"
            },
            {
                "type": "CAUSAL",
                "question": "Why did the cook add rose water to the second cake but not the first?",
                "ground_truth": "Period-specific recipe variation",
                "why_hard": "Causal reasoning about recipe choices"
            },
            {
                "type": "NEEDLE",
                "question": "What was the weight of the bread ration for a Sunday breakfast in a 18th-century workhouse?",
                "ground_truth": "4 ounces",
                "why_hard": "Verbal needle in 3+ hour video"
            },
            {
                "type": "ORDERING",
                "question": "What happened immediately after the Sabotiere ice cream maker was opened?",
                "ground_truth": "Specific next action",
                "why_hard": "Precise temporal ordering"
            }
        ]
    },
    {
        "id": "06_woodworking",
        "title": "Five Years of Woodworking Projects",
        "url": "https://www.youtube.com/watch?v=0_u6eA2eC28",
        "duration": "02:35:45",
        "category": "Craft/Making",
        "hardness": 8,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many times did the creator show the moisture meter reading?",
                "ground_truth": "Specific count",
                "why_hard": "Repeated tool usage across compilation"
            },
            {
                "type": "COMPARATIVE",
                "question": "How did the leg design evolve from the first project to the tenth?",
                "ground_truth": "Design evolution",
                "why_hard": "Comparing across multiple projects"
            },
            {
                "type": "CAUSAL",
                "question": "Why did he switch from Rubio Monocoat to Blacktail Finish halfway through?",
                "ground_truth": "Product development/preference",
                "why_hard": "Causal reasoning about product choice"
            },
            {
                "type": "NEEDLE",
                "question": "What was the price of the first table he ever sold?",
                "ground_truth": "$400",
                "why_hard": "Brief mention in compilation"
            },
            {
                "type": "ORDERING",
                "question": "What happened immediately after the epoxy smoke incident?",
                "ground_truth": "Specific next action",
                "why_hard": "Precise temporal ordering"
            }
        ]
    },
    {
        "id": "07_porsche_911",
        "title": "1970s Porsche 911 Restoration (Full)",
        "url": "https://www.youtube.com/watch?v=u6w-XWpY5G8",
        "duration": "01:12:30",
        "category": "Repair/Restoration",
        "hardness": 5,
        "queries": [
            {
                "type": "COUNTING",
                "question": "How many times did the restorer apply rust converter?",
                "ground_truth": "Specific count",
                "why_hard": "Repetitive activity"
            },
            {
                "type": "COMPARATIVE",
                "question": "How did the engine bay look before vs. after the dry-ice blasting?",
                "ground_truth": "Rusty/dirty vs clean metal",
                "why_hard": "Visual comparison"
            },
            {
                "type": "CAUSAL",
                "question": "Why were the original seats discarded?",
                "ground_truth": "Beyond repair/safety",
                "why_hard": "Causal reasoning"
            },
            {
                "type": "NEEDLE",
                "question": "What was the specific diameter of the replacement brake lines?",
                "ground_truth": "4.75mm",
                "why_hard": "Small text/verbal detail"
            },
            {
                "type": "ORDERING",
                "question": "What happened immediately after the first engine fire-up?",
                "ground_truth": "Specific next action",
                "why_hard": "Precise temporal ordering"
            }
        ]
    }
]


def download_video(url: str, video_id: str, output_dir: Path = Path("videos")) -> str:
    """
    Download YouTube video using yt-dlp.
    
    Args:
        url: YouTube URL
        video_id: Video identifier for naming
        output_dir: Directory to save videos
    
    Returns:
        Path to downloaded video file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from video_id
    output_template = str(output_dir / f"{video_id}.%(ext)s")
    
    # Check if video already exists
    for ext in ['mp4', 'webm', 'mkv']:
        existing_file = output_dir / f"{video_id}.{ext}"
        if existing_file.exists():
            print(f"✓ Video already downloaded: {existing_file}")
            return str(existing_file)
    
    print(f"\n📥 Downloading video from YouTube...")
    print(f"   URL: {url}")
    print(f"   This may take several minutes for long videos...")
    
    try:
        # Check if yt-dlp is installed
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ Error: yt-dlp is not installed!")
        print("   Install with: pip install yt-dlp")
        print("   Or: python -m pip install yt-dlp")
        sys.exit(1)
    
    # Download with yt-dlp
    # Use 720p max to save space and download time
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', output_template,
        '--no-playlist',
        url
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Find the downloaded file
        for ext in ['mp4', 'webm', 'mkv']:
            video_file = output_dir / f"{video_id}.{ext}"
            if video_file.exists():
                print(f"✓ Video downloaded: {video_file}")
                return str(video_file)
        
        raise FileNotFoundError("Downloaded video file not found")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error downloading video: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        sys.exit(1)


def run_stress_test(
    video_path: str,
    video_config: Dict[str, Any],
    vlm_model: str = 'clip',
    device: str = 'auto',
    target_fps: float = 5.0
) -> Dict[str, Any]:
    """
    Run stress test on a single video.
    
    Args:
        video_path: Path to video file
        video_config: Video configuration from STRESS_TEST_VIDEOS
        vlm_model: Vision model to use ('clip', 'siglip', 'smolvlm')
        device: Device to use
        target_fps: Target FPS for processing
    
    Returns:
        Test results dictionary
    """
    print(f"\n{'='*80}")
    print(f"STRESS TEST: {video_config['title']}")
    print(f"{'='*80}")
    print(f"Duration: {video_config['duration']}")
    print(f"Category: {video_config['category']}")
    print(f"Hardness: {video_config['hardness']}/10")
    print(f"Model: {vlm_model.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize processor
    print(f"🚀 Initializing processor with {vlm_model.upper()}...")
    processor = VideoProcessor(
        vlm_model=vlm_model,
        device=device,
        target_fps=target_fps,
        enable_temporal=True
    )
    
    # Process video
    print(f"\n📹 Processing video: {video_path}")
    start_time = time.time()
    
    try:
        process_result = processor.process(video_path)
        process_time = time.time() - start_time
        
        print(f"\n✓ Video processed in {process_time/60:.2f} minutes")
        print(f"  - Frames processed: {process_result['num_frames']}")
        print(f"  - Video duration: {process_result['duration']:.1f}s")
        print(f"  - Events detected: {len(process_result.get('events', []))}")
        
    except Exception as e:
        print(f"\n❌ Failed to process video: {str(e)}")
        return {
            "video_id": video_config["id"],
            "status": "failed",
            "error": str(e),
            "process_time": time.time() - start_time
        }
    
    # Run queries
    print(f"\n🔍 Running {len(video_config['queries'])} stress test queries...")
    query_results = []
    
    for i, query in enumerate(video_config['queries'], 1):
        print(f"\n[{i}/{len(video_config['queries'])}] {query['type']}: {query['question']}")
        print(f"   Ground Truth: {query['ground_truth']}")
        print(f"   Why Hard: {query['why_hard']}")
        
        query_start = time.time()
        
        try:
            # Run query
            matches = processor.query(query['question'], top_k=5)
            query_time = time.time() - query_start
            
            # Get LLM answer if available
            try:
                answer = processor.chat(query['question'], use_llm=True)
            except:
                answer = "LLM not available"
            
            print(f"   ⏱️  Query time: {query_time*1000:.0f}ms")
            print(f"   📍 Top match: {matches[0]['timestamp']:.1f}s (confidence: {matches[0]['confidence']:.3f})")
            print(f"   💬 Answer: {answer[:100]}...")
            
            query_results.append({
                "type": query["type"],
                "question": query["question"],
                "ground_truth": query["ground_truth"],
                "why_hard": query["why_hard"],
                "query_time_ms": query_time * 1000,
                "top_matches": [
                    {
                        "timestamp": m["timestamp"],
                        "confidence": m["confidence"]
                    }
                    for m in matches
                ],
                "answer": answer,
                "status": "completed"
            })
            
        except Exception as e:
            print(f"   ❌ Query failed: {str(e)}")
            query_results.append({
                "type": query["type"],
                "question": query["question"],
                "ground_truth": query["ground_truth"],
                "error": str(e),
                "status": "failed"
            })
    
    # Compile results
    results = {
        "video_id": video_config["id"],
        "title": video_config["title"],
        "duration": video_config["duration"],
        "category": video_config["category"],
        "hardness": video_config["hardness"],
        "model": vlm_model,
        "target_fps": target_fps,
        "process_time_minutes": process_time / 60,
        "num_frames_processed": process_result["num_frames"],
        "num_events_detected": len(process_result.get("events", [])),
        "queries": query_results,
        "status": "completed"
    }
    
    return results


def save_results(results: Dict[str, Any], output_dir: Path):
    """Save test results to markdown file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_id = results["video_id"]
    filename = f"results_{timestamp}.md"
    filepath = output_dir / video_id / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown report
    md_content = f"""# Stress Test Results: {results['title']}

**Test Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Video ID:** {results['video_id']}  
**Duration:** {results['duration']}  
**Category:** {results['category']}  
**Hardness:** {results['hardness']}/10  
**Model:** {results['model'].upper()}  
**Target FPS:** {results['target_fps']}  

---

## Processing Summary

- **Process Time:** {results['process_time_minutes']:.2f} minutes
- **Frames Processed:** {results['num_frames_processed']}
- **Events Detected:** {results['num_events_detected']}
- **Status:** {results['status']}

---

## Query Results

"""
    
    # Add query results
    for i, query in enumerate(results['queries'], 1):
        md_content += f"""### Query {i}: {query['type']}

**Question:** {query['question']}

**Ground Truth:** {query['ground_truth']}

**Why Hard:** {query['why_hard']}

"""
        
        if query['status'] == 'completed':
            md_content += f"""**Query Time:** {query['query_time_ms']:.0f}ms

**Top Matches:**
"""
            for j, match in enumerate(query['top_matches'], 1):
                md_content += f"{j}. {match['timestamp']:.1f}s (confidence: {match['confidence']:.3f})\n"
            
            md_content += f"""
**Answer:**
```
{query['answer']}
```

"""
        else:
            md_content += f"""**Status:** Failed
**Error:** {query.get('error', 'Unknown error')}

"""
        
        md_content += "---\n\n"
    
    # Add summary statistics
    completed_queries = [q for q in results['queries'] if q['status'] == 'completed']
    if completed_queries:
        avg_query_time = sum(q['query_time_ms'] for q in completed_queries) / len(completed_queries)
        md_content += f"""## Summary Statistics

- **Total Queries:** {len(results['queries'])}
- **Completed:** {len(completed_queries)}
- **Failed:** {len(results['queries']) - len(completed_queries)}
- **Average Query Time:** {avg_query_time:.0f}ms

"""
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✓ Results saved to: {filepath}")
    
    # Also save JSON version
    json_filepath = filepath.with_suffix('.json')
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ JSON saved to: {json_filepath}")


def main():
    """Main stress test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run extreme stress tests on YouTube videos')
    parser.add_argument('--video-path', help='Path to video file (optional, will download if not provided)')
    parser.add_argument('--video-id', required=True, help='Video ID from stress test dataset')
    parser.add_argument('--model', default='siglip', choices=['clip', 'siglip', 'siglip-base', 'siglip-large', 'smolvlm'],
                       help='Vision model to use (default: siglip)')
    parser.add_argument('--device', default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--fps', type=float, default=5.0, help='Target FPS for processing')
    parser.add_argument('--output-dir', default='D:/PROJECTS/webstromprojects/sharingan/complex_stress_test_results',
                       help='Output directory for results')
    parser.add_argument('--download-dir', default='videos', help='Directory to download videos to')
    parser.add_argument('--no-download', action='store_true', help='Do not download video, use existing file only')
    
    args = parser.parse_args()
    
    # Find video config
    video_config = None
    for config in STRESS_TEST_VIDEOS:
        if config['id'] == args.video_id:
            video_config = config
            break
    
    if not video_config:
        print(f"❌ Error: Video ID '{args.video_id}' not found in stress test dataset")
        print(f"\nAvailable video IDs:")
        for config in STRESS_TEST_VIDEOS:
            print(f"  - {config['id']}: {config['title']}")
        sys.exit(1)
    
    # Determine video path
    if args.video_path:
        video_path = args.video_path
        if not Path(video_path).exists():
            print(f"❌ Error: Video file not found: {video_path}")
            sys.exit(1)
    elif args.no_download:
        print(f"❌ Error: --no-download specified but no --video-path provided")
        sys.exit(1)
    else:
        # Download video
        video_path = download_video(
            url=video_config['url'],
            video_id=args.video_id,
            output_dir=Path(args.download_dir)
        )
    
    # Run stress test
    results = run_stress_test(
        video_path=video_path,
        video_config=video_config,
        vlm_model=args.model,
        device=args.device,
        target_fps=args.fps
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    
    print(f"\n{'='*80}")
    print(f"STRESS TEST COMPLETED")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
