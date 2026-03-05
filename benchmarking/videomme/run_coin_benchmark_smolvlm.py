"""
Run COIN benchmark with SmolVLM - works with CPU or GPU
Saves results incrementally after each video
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Starting SmolVLM benchmark...", flush=True)

from sharingan.processor import VideoProcessor

print("VideoProcessor imported successfully!", flush=True)

def save_incremental_results(output_file, predictions_file, metadata, results):
    """Save results incrementally to both JSON and JSONL files"""
    # Save detailed JSON
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': metadata,
            'results': results
        }, f, indent=2)
    
    # Save predictions in TemporalBench .jsonl format
    with open(predictions_file, 'w') as f:
        for result in results:
            f.write(json.dumps({
                'video': result['video'],
                'question_id': result['question_id'],
                'pred': result['predicted']
            }) + '\n')

def main():
    print("\n" + "="*80, flush=True)
    print("TEMPORALBENCH COIN BENCHMARK - SmolVLM", flush=True)
    print("="*80, flush=True)
    
    # Check device - import torch only when needed
    print("\nChecking device...", flush=True)
    import torch
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = 'cpu'
        print(f"⚠ Using CPU (slower - consider enabling GPU)", flush=True)
    
    # Load QA data
    qa_file = Path(__file__).parent / "long_video_coin/temporalbench_long_qa.json"
    print(f"\nLoading QA data...", flush=True)
    with open(qa_file) as f:
        qa_data = json.load(f)
    
    print(f"Total QA pairs: {len(qa_data)}", flush=True)
    print(f"Running full dataset benchmark with SmolVLM", flush=True)
    
    # Group by video
    videos_qa = {}
    for qa in qa_data:
        video_name = Path(qa['video_name']).name
        if video_name not in videos_qa:
            videos_qa[video_name] = []
        videos_qa[video_name].append(qa)
    
    print(f"Unique videos: {len(videos_qa)}", flush=True)
    
    # Initialize processor with SmolVLM
    print(f"\nInitializing VideoProcessor with SmolVLM...", flush=True)
    processor = VideoProcessor(
        vlm_model='smolvlm',  # Use SmolVLM instead of CLIP
        device=device,
        target_fps=5.0,
        enable_temporal=True,
        batch_size=32
    )
    print(f"Processor ready!", flush=True)
    
    # Setup output directory and files
    output_dir = Path(__file__).parent / "long_video_coin/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"results_smolvlm_{timestamp}.json"
    predictions_file = output_dir / f"predictions_smolvlm_{timestamp}.jsonl"
    
    # Process videos and answer questions
    video_dir = Path(__file__).parent / "long_video_coin/dataset/long_video/COIN"
    results = []
    correct = 0
    total = 0
    
    print(f"\nProcessing videos...")
    print(f"Results will be saved incrementally to:")
    print(f"  Predictions: {predictions_file}")
    print(f"  Detailed: {output_file}")
    print("="*80)
    
    overall_start = time.time()
    
    for video_idx, (video_name, questions) in enumerate(videos_qa.items(), 1):
        video_path = video_dir / video_name
        
        if not video_path.exists():
            print(f"\n[{video_idx}/{len(videos_qa)}] SKIP: {video_name} not found")
            continue
        
        print(f"\n[{video_idx}/{len(videos_qa)}] VIDEO: {video_name}")
        print(f"  Questions: {len(questions)}")
        
        # Process video
        try:
            print(f"  Processing with SmolVLM...")
            proc_start = time.time()
            video_results = processor.process(str(video_path))
            proc_time = time.time() - proc_start
            print(f"  ✓ Processed in {proc_time:.1f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Answer questions for this video
        video_correct = 0
        for q_idx, qa in enumerate(questions, 1):
            question = qa['question']
            gt = qa['GT']
            question_id = qa['idx']  # Use the idx field as question_id
            
            try:
                response = processor.chat(question, use_llm=False)
                
                # Extract answer - look for final answer in last 50 characters
                import re
                response_tail = response.strip()[-50:]
                match = re.search(r'\b([AB])\b', response_tail)
                if match:
                    predicted = match.group(1)
                else:
                    # Fallback: look for "Answer: X" pattern
                    answer_match = re.search(r'(?:answer|choice|option)[:\s]+([AB])', response.lower())
                    predicted = answer_match.group(1).upper() if answer_match else 'A'
                
                is_correct = (predicted == gt)
                if is_correct:
                    correct += 1
                    video_correct += 1
                total += 1
                
                print(f"    Q{q_idx}: Predicted={predicted} GT={gt} {'✓' if is_correct else '✗'}")
                
                results.append({
                    'video': video_name,
                    'question_id': question_id,
                    'question': question[:100],
                    'predicted': predicted,
                    'ground_truth': gt,
                    'correct': is_correct
                })
                
                # Save incrementally after each question
                elapsed_time = time.time() - overall_start
                accuracy = (correct / total * 100) if total > 0 else 0
                
                metadata = {
                    'model': 'smolvlm',
                    'videos_processed': video_idx,
                    'total_videos': len(videos_qa),
                    'total_questions': total,
                    'correct': correct,
                    'accuracy': accuracy,
                    'device': device,
                    'elapsed_time': elapsed_time,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'in_progress'
                }
                
                save_incremental_results(output_file, predictions_file, metadata, results)
                
            except Exception as e:
                print(f"    Q{q_idx}: Error - {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate current accuracy
        accuracy = (correct / total * 100) if total > 0 else 0
        video_accuracy = (video_correct / len(questions) * 100) if questions else 0
        
        print(f"  Video accuracy: {video_accuracy:.1f}% ({video_correct}/{len(questions)})")
        print(f"  Running accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"  ✓ Results saved incrementally")
    
    total_time = time.time() - overall_start
    
    # Final save with completion status
    final_metadata = {
        'model': 'smolvlm',
        'videos_processed': len(videos_qa),
        'total_videos': len(videos_qa),
        'total_questions': total,
        'correct': correct,
        'accuracy': (correct/total*100) if total > 0 else 0,
        'device': device,
        'total_time': total_time,
        'timestamp': datetime.now().isoformat(),
        'status': 'completed'
    }
    
    save_incremental_results(output_file, predictions_file, final_metadata, results)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - SmolVLM")
    print(f"{'='*80}")
    print(f"Questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {(correct/total*100):.1f}%")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Device: {device.upper()}")
    print(f"\nResults saved:")
    print(f"  Predictions (TemporalBench format): {predictions_file}")
    print(f"  Detailed results: {output_file}")
    print(f"\nTo get official BA/MBA scores, run:")
    print(f"  python get_qa_acc.py --pred_file {predictions_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
