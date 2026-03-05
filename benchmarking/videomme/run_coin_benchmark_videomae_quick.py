"""
Run COIN benchmark with VideoMAE V2 + Qwen2.5-1.5B - QUICK TEST (20 videos)

Features:
- Time tracking for VideoMAE and LLM separately
- Incremental results saving (saves after each question)
- Resume capability (skips already processed videos)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Starting VideoMAE Quick Test (20 videos)...", flush=True)

from sharingan.processor_videomae import VideoProcessorVideoMAE

print("VideoProcessorVideoMAE imported successfully!", flush=True)


def load_existing_results(predictions_file, results_file):
    """Load existing results to resume from where we stopped."""
    processed_videos = set()
    existing_results = []
    
    if predictions_file.exists() and results_file.exists():
        print(f"\n📂 Found existing results, loading...", flush=True)
        
        # Load predictions
        with open(predictions_file, 'r') as f:
            for line in f:
                pred = json.loads(line)
                processed_videos.add(pred['video'])
        
        # Load detailed results
        with open(results_file, 'r') as f:
            data = json.load(f)
            existing_results = data.get('results', [])
        
        print(f"✓ Loaded {len(processed_videos)} already processed videos", flush=True)
    
    return processed_videos, existing_results


def save_results_incremental(predictions_file, results_file, results, metadata):
    """Save results incrementally after each question."""
    # Save predictions (append mode)
    with open(predictions_file, 'w') as f:
        for result in results:
            f.write(json.dumps({
                'video': result['video'],
                'question_id': result['question_id'],
                'pred': result['predicted']
            }) + '\n')
    
    # Save detailed results (overwrite mode)
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': metadata,
            'results': results
        }, f, indent=2)


def main():
    print("\n" + "="*80, flush=True)
    print("TEMPORALBENCH COIN QUICK TEST - VideoMAE V2 + Qwen2.5-1.5B (20 videos)", flush=True)
    print("="*80, flush=True)
    
    # Check device
    print("\nChecking device...", flush=True)
    import torch
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = 'cpu'
        print(f"⚠ Using CPU (slower)", flush=True)
    
    # Load QA data
    qa_file = Path(__file__).parent / "long_video_coin/temporalbench_long_qa.json"
    print(f"\nLoading QA data...", flush=True)
    with open(qa_file) as f:
        qa_data = json.load(f)
    
    print(f"Total QA pairs: {len(qa_data)}", flush=True)
    
    # Group by video
    videos_qa = {}
    for qa in qa_data:
        video_name = Path(qa['video_name']).name
        if video_name not in videos_qa:
            videos_qa[video_name] = []
        videos_qa[video_name].append(qa)
    
    # LIMIT TO FIRST 20 VIDEOS
    videos_qa_list = list(videos_qa.items())[:20]
    videos_qa = dict(videos_qa_list)
    
    print(f"Testing on first {len(videos_qa)} videos", flush=True)
    
    # Initialize processor with VideoMAE and Qwen2.5-1.5B
    print(f"\nInitializing VideoProcessorVideoMAE with Qwen2.5-1.5B...", flush=True)
    processor = VideoProcessorVideoMAE(
        device=device,
        target_fps=2.0,  # Reduced from 5.0 for faster processing
        llm_model='qwen-1.5b'
    )
    print(f"Processor ready!", flush=True)
    
    # Setup output
    output_dir = Path(__file__).parent / "long_video_coin/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"results_videomae_quick_{timestamp}.json"
    predictions_file = output_dir / f"predictions_videomae_quick_{timestamp}.jsonl"
    
    # Load existing results (for resume capability)
    processed_videos, results = load_existing_results(predictions_file, results_file)
    
    # Process videos
    video_dir = Path(__file__).parent / "long_video_coin/dataset/long_video/COIN"
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    
    # Timing stats
    total_videomae_time = 0
    total_llm_time = 0
    videos_processed = 0
    
    print(f"\nProcessing {len(videos_qa)} videos...")
    print("="*80)
    
    overall_start = time.time()
    
    for video_idx, (video_name, questions) in enumerate(videos_qa.items(), 1):
        video_path = video_dir / video_name
        
        if not video_path.exists():
            print(f"\n[{video_idx}/{len(videos_qa)}] SKIP: {video_name} not found")
            continue
        
        print(f"\n[{video_idx}/{len(videos_qa)}] VIDEO: {video_name}")
        print(f"  Questions: {len(questions)}")
        
        # Skip if already processed (resume capability)
        if video_name in processed_videos:
            print(f"  ⏭️  Already processed, skipping...")
            continue
        
        # Process video
        try:
            print(f"  Processing with VideoMAE...")
            proc_start = time.time()
            video_results = processor.process(str(video_path))
            proc_time = time.time() - proc_start
            total_videomae_time += proc_time
            videos_processed += 1
            print(f"  ✓ Processed in {proc_time:.1f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Answer questions
        video_correct = 0
        for q_idx, qa in enumerate(questions, 1):
            question = qa['question']
            gt = qa['GT']
            question_id = qa['idx']
            
            try:
                # Time LLM generation
                llm_start = time.time()
                response = processor.chat(question)
                llm_time = time.time() - llm_start
                total_llm_time += llm_time
                
                # Extract answer
                import re
                response_tail = response.strip()[-50:]
                match = re.search(r'\b([AB])\b', response_tail)
                if match:
                    predicted = match.group(1)
                else:
                    answer_match = re.search(r'(?:answer|choice|option)[:\s]+([AB])', response.lower())
                    predicted = answer_match.group(1).upper() if answer_match else 'A'
                
                is_correct = (predicted == gt)
                if is_correct:
                    correct += 1
                    video_correct += 1
                total += 1
                
                print(f"    Q{q_idx}: Predicted={predicted} GT={gt} {'✓' if is_correct else '✗'} (LLM: {llm_time:.1f}s)")
                
                results.append({
                    'video': video_name,
                    'question_id': question_id,
                    'question': question[:100],
                    'predicted': predicted,
                    'ground_truth': gt,
                    'correct': is_correct,
                    'llm_time': llm_time
                })
                
                # Save incrementally after each question
                metadata = {
                    'model': 'videomae-large + qwen-1.5b',
                    'vlm_model': 'videomae-large',
                    'llm_model': 'qwen-1.5b',
                    'videos_tested': video_idx,
                    'total_questions': total,
                    'correct': correct,
                    'accuracy': (correct/total*100) if total > 0 else 0,
                    'device': device,
                    'total_videomae_time': total_videomae_time,
                    'total_llm_time': total_llm_time,
                    'avg_videomae_time': total_videomae_time / videos_processed if videos_processed > 0 else 0,
                    'avg_llm_time': total_llm_time / total if total > 0 else 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'in_progress'
                }
                save_results_incremental(predictions_file, results_file, results, metadata)
                
            except Exception as e:
                print(f"    Q{q_idx}: Error - {e}")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        video_accuracy = (video_correct / len(questions) * 100) if questions else 0
        
        print(f"  Video accuracy: {video_accuracy:.1f}% ({video_correct}/{len(questions)})")
        print(f"  Running accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    total_time = time.time() - overall_start
    
    # Save final results
    final_metadata = {
        'model': 'videomae-large + qwen-1.5b',
        'vlm_model': 'videomae-large',
        'llm_model': 'qwen-1.5b',
        'videos_tested': len(videos_qa),
        'total_questions': total,
        'correct': correct,
        'accuracy': (correct/total*100) if total > 0 else 0,
        'device': device,
        'total_time': total_time,
        'total_videomae_time': total_videomae_time,
        'total_llm_time': total_llm_time,
        'avg_videomae_time': total_videomae_time / videos_processed if videos_processed > 0 else 0,
        'avg_llm_time': total_llm_time / total if total > 0 else 0,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }
    save_results_incremental(predictions_file, results_file, results, final_metadata)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"QUICK TEST RESULTS - VideoMAE V2 + Qwen2.5-1.5B")
    print(f"{'='*80}")
    print(f"Videos: {len(videos_qa)}")
    print(f"Questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {(correct/total*100):.1f}%")
    print(f"\nTiming Breakdown:")
    print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  VideoMAE Time: {total_videomae_time:.1f}s ({total_videomae_time/total_time*100:.1f}%)")
    print(f"  LLM Time: {total_llm_time:.1f}s ({total_llm_time/total_time*100:.1f}%)")
    print(f"  Avg VideoMAE/video: {total_videomae_time/videos_processed:.1f}s" if videos_processed > 0 else "")
    print(f"  Avg LLM/question: {total_llm_time/total:.1f}s" if total > 0 else "")
    print(f"\nDevice: {device.upper()}")
    print(f"\nResults saved:")
    print(f"  Predictions: {predictions_file}")
    print(f"  Detailed: {results_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
