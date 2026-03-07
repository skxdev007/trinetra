"""
TemporalBench Long Video COIN Benchmark for SHARINGAN

Tests fine-grained temporal understanding using the TemporalBench dataset.
Questions test subtle differences in:
- Counting (5 times vs 10 times)
- Direction (left to right vs right to left)
- Order (A then B vs B then A)
- State changes (clean vs dirty water)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Tuple

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sharingan.processor import VideoProcessor


def load_qa_data(qa_file: Path) -> List[Dict]:
    """Load QA pairs from JSON file."""
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {qa_file.name}")
    return data


def extract_answer_from_response(response: str, question: str) -> str:
    """
    Extract A or B from the model's response.
    Uses improved regex-based extraction focusing on the final answer.
    """
    import re
    
    response = response.strip()
    
    # Method 1: Look for isolated A or B in the last 50 characters
    # This catches "Answer: A", "The answer is B", etc.
    response_tail = response[-50:]
    match = re.search(r'\b([AB])\b', response_tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Method 2: Look for "Answer: X" or "Option X" patterns anywhere
    answer_patterns = [
        r'(?:answer|choice|option|select)[:\s]+([AB])',
        r'(?:the\s+)?(?:correct\s+)?(?:answer\s+is\s+)?([AB])',
        r'(?:I\s+choose\s+)?([AB])\b'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Method 3: If response is very short and contains only A or B
    if len(response) < 10:
        response_upper = response.upper()
        if 'A' in response_upper and 'B' not in response_upper:
            return 'A'
        if 'B' in response_upper and 'A' not in response_upper:
            return 'B'
    
    # Method 4: Count occurrences in last 100 chars (avoid question text)
    tail = response[-100:].upper()
    a_count = tail.count('A')
    b_count = tail.count('B')
    
    if a_count > b_count:
        return 'A'
    elif b_count > a_count:
        return 'B'
    
    # Default to A if completely unclear
    print(f"    Could not extract clear answer from: '{response[:100]}...'")
    return 'A'


def process_video_and_answer(
    processor: VideoProcessor,
    video_path: Path,
    question: str,
    processed_videos: Dict[str, dict],
    system_prompt: str = None,
    user_prompt_template: str = None,
    verbose: bool = True,  # Added to control printing
    enable_action_classification: bool = True  # NEW: Enable action classification
) -> Tuple[str, float, str]:  # Fixed return type
    """
    Process video if not already processed, then answer the question.
    Returns (answer, query_time, response).
    """
    video_key = str(video_path)
    
    # Process video if not already done
    if video_key not in processed_videos:
        print(f"\n  >>> Processing NEW video: {video_path.name}")
        print(f"  >>> Video path: {video_path}")
        start_time = time.time()
        results = processor.process(str(video_path))
        process_time = time.time() - start_time
        processed_videos[video_key] = {
            'results': results,
            'process_time': process_time
        }
        print(f"  >>>  Video processed in {process_time:.1f}s")
        print(f"  >>> Events detected: {len(results.get('events', []))}")
        print(f"  >>> Total videos processed so far: {len(processed_videos)}")
    else:
        print(f"  >>> Using cached video: {video_path.name}")
    
    # Answer the question
    start_time = time.time()
    try:
        # Get video context first WITHOUT action classification (too noisy)
        segments = processor.query(
            question, 
            top_k=10,  # Increased from 5 to get more context
            enable_action_classification=False  # DISABLED: CLIP zero-shot too noisy
        )
        
        # Print context if verbose
        if verbose:
            print(f"\n{'='*80}")
            print(f"CONTEXT SENT TO LLM:")
            print(f"{'='*80}")
            print("VIDEO TIMELINE:")
            print("-" * 60)
            for i, seg in enumerate(segments, 1):
                timestamp = seg['timestamp']
                confidence = seg['confidence']
                description = seg.get('description', 'Content detected')
                actions = seg.get('actions', {})
                
                mins = int(timestamp // 60)
                secs = int(timestamp % 60)
                time_str = f"{mins}:{secs:02d}"
                
                print(f"{i}. [{time_str}] {description}")
                if actions:
                    print(f"   Actions: {actions}")
                print(f"   Relevance: {confidence:.1%}")
            print("-" * 60)
            print(f"{'='*80}\n")
        
        # Override prompts if provided
        if system_prompt or user_prompt_template:
            # Get the LLM directly and set custom prompts
            if not processor._llm:
                from sharingan.chat import VideoLLM
                print(f" Initializing Qwen2.5-1.5B-Instruct...")
                processor._llm = VideoLLM(model_name='qwen-1.5b', device=processor.device)
            
            # Use custom prompts
            response = processor._llm.chat(
                query=question,
                video_context=segments,
                max_new_tokens=10,
                temperature=0.3
            )
            
            # Override the chat method's prompts if needed
            if system_prompt:
                processor._llm.chat_history = []  # Clear history for custom prompts
        else:
            response = processor.chat(question, use_llm=True)
            
    except Exception as e:
        # Fallback to non-LLM response if LLM fails
        print(f"    LLM failed, using fallback: {e}")
        response = processor.chat(question, use_llm=False)
    query_time = time.time() - start_time
    
    answer = extract_answer_from_response(response, question)
    
    return answer, query_time, response


def run_benchmark(
    qa_file: Path,
    dataset_dir: Path,
    output_dir: Path,
    max_questions: int = None,
    target_fps: float = 5.0,
    vlm_model: str = 'siglip-so400m',
    enable_descriptions: bool = True,
    delta_captioning: bool = True,
    system_prompt: str = None,  # Added parameter
    user_prompt_template: str = None  # Added parameter
):
    """Run the benchmark on TemporalBench Long Video COIN dataset."""
    
    print("\n" + "="*80)
    print("SHARINGAN - TemporalBench Long Video COIN Benchmark")
    print("="*80)
    
    # Load QA data
    qa_data = load_qa_data(qa_file)
    
    if max_questions:
        qa_data = qa_data[:max_questions]
        print(f"Limiting to first {max_questions} questions for testing")
    
    # Initialize processor
    print(f"\nInitializing VideoProcessor (target_fps={target_fps})...")
    print(f"Vision Model: {vlm_model.upper()}")
    if enable_descriptions:
        if delta_captioning:
            print(f"Frame descriptions: ENABLED with Delta-Captioning (InternVL2.5 - 6x faster)")
        else:
            print(f"Frame descriptions: ENABLED (InternVL2.5 will caption all frames)")
    else:
        print(f"Frame descriptions: DISABLED (using 'Content detected')")
    print(f"Checking GPU availability...")
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU will be used for processing")
    else:
        print(f"No GPU detected - using CPU (this will be slower)")
    
    processor = VideoProcessor(
        vlm_model=vlm_model,  # Use parameter instead of hardcoded 'clip'
        device='auto',
        target_fps=target_fps,
        enable_temporal=True,
        enable_descriptions=enable_descriptions,
        lazy_descriptions=True,  # FAST: Generate descriptions only for retrieved frames
        delta_captioning=delta_captioning,  # Only caption keyframes (6x faster)
        batch_size=32
    )
    
    # Track results
    results = []
    processed_videos = {}
    correct = 0
    total = 0
    
    # Process each question
    print(f"\nProcessing {len(qa_data)} questions...")
    print("-" * 80)
    
    start_time = time.time()
    
    for i, qa in enumerate(qa_data, 1):
        idx = qa['idx']
        video_name = qa['video_name']
        question = qa['question']
        ground_truth = qa['GT']
        
        # Construct video path
        # video_name format: "long_video/ActivityNet/filename.mp4"
        # We need to map to: dataset_dir/long_video/COIN/filename.mp4
        video_filename = Path(video_name).name
        video_path = dataset_dir / "long_video" / "COIN" / video_filename
        
        if not video_path.exists():
            print(f"\n[{i}/{len(qa_data)}] SKIP: Video not found: {video_path.name}")
            continue
        
        print(f"\n[{i}/{len(qa_data)}] Question ID: {idx}")
        print(f"  Video: {video_path.name}")
        
        try:
            # Process and answer
            predicted_answer, query_time, response = process_video_and_answer(
                processor, video_path, question, processed_videos,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                verbose=(i <= 3),  # Show context for first 3 questions
                enable_action_classification=True  # Enable action classification
            )
            
            # Check correctness
            is_correct = (predicted_answer == ground_truth)
            if is_correct:
                correct += 1
            
            total += 1
            accuracy = (correct / total) * 100
            
            print(f"  Question: {question[:100]}...")
            print(f"  Response: {response[:150]}...")
            print(f"  Predicted: {predicted_answer} | Ground Truth: {ground_truth} | {'' if is_correct else ''}")
            print(f"  Query time: {query_time:.3f}s")
            print(f"  Running accuracy: {accuracy:.2f}% ({correct}/{total})")
            
            # Store result
            results.append({
                'idx': idx,
                'video_name': video_name,
                'video_path': str(video_path),
                'question': question,
                'ground_truth': ground_truth,
                'predicted': predicted_answer,
                'response': response,  # Added for debugging
                'correct': is_correct,
                'query_time': query_time
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_query_time = sum(r['query_time'] for r in results) / len(results) if results else 0
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    results_json = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'dataset': 'TemporalBench Long Video COIN',
                'model': vlm_model,  # Added model info
                'total_questions': len(qa_data),
                'answered_questions': total,
                'correct_answers': correct,
                'accuracy': accuracy,
                'total_time': total_time,
                'avg_query_time': avg_query_time,
                'unique_videos_processed': len(processed_videos),
                'target_fps': target_fps,
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n Detailed results saved to: {results_json}")
    
    # Save summary as markdown
    save_summary_markdown(
        output_dir, results, processed_videos, 
        total, correct, accuracy, total_time, avg_query_time, target_fps, vlm_model, enable_descriptions
    )
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(qa_data)}")
    print(f"Answered: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Unique Videos Processed: {len(processed_videos)}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average Query Time: {avg_query_time:.3f}s")
    print("="*80)


def save_summary_markdown(
    output_dir: Path,
    results: List[Dict],
    processed_videos: Dict,
    total: int,
    correct: int,
    accuracy: float,
    total_time: float,
    avg_query_time: float,
    target_fps: float,
    vlm_model: str,
    enable_descriptions: bool
):
    """Save benchmark summary as markdown."""
    
    output_file = output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# SHARINGAN - TemporalBench Long Video COIN Benchmark Results\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Overall metrics
        f.write("## Overall Performance\n\n")
        f.write(f"- **Model:** {vlm_model.upper()}\n")
        f.write(f"- **Frame Descriptions:** {'ENABLED (SmolVLM)' if enable_descriptions else 'DISABLED'}\n")
        f.write(f"- **Accuracy:** {accuracy:.2f}% ({correct}/{total})\n")
        f.write(f"- **Total Questions:** {total}\n")
        f.write(f"- **Unique Videos:** {len(processed_videos)}\n")
        f.write(f"- **Total Time:** {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"- **Average Query Time:** {avg_query_time:.3f}s\n")
        f.write(f"- **Target FPS:** {target_fps}\n\n")
        
        # Video processing stats
        f.write("## Video Processing Statistics\n\n")
        total_process_time = sum(v['process_time'] for v in processed_videos.values())
        avg_process_time = total_process_time / len(processed_videos) if processed_videos else 0
        f.write(f"- **Total Processing Time:** {total_process_time:.1f}s ({total_process_time/60:.1f} min)\n")
        f.write(f"- **Average Processing Time per Video:** {avg_process_time:.1f}s\n")
        f.write(f"- **Videos Processed:** {len(processed_videos)}\n\n")
        
        # Sample results
        f.write("## Sample Results\n\n")
        f.write("| # | Video | Question | GT | Predicted | Correct | Query Time |\n")
        f.write("|---|-------|----------|----|-----------|---------|-----------|\n")
        
        for i, result in enumerate(results[:20], 1):  # Show first 20
            video_name = Path(result['video_path']).name
            question_short = result['question'][:60].replace('\n', ' ') + "..."
            gt = result['ground_truth']
            pred = result['predicted']
            correct_mark = '' if result['correct'] else ''
            query_time = result['query_time']
            
            f.write(f"| {i} | {video_name} | {question_short} | {gt} | {pred} | {correct_mark} | {query_time:.3f}s |\n")
        
        if len(results) > 20:
            f.write(f"\n*Showing first 20 of {len(results)} results. See JSON file for complete results.*\n")
        
        f.write("\n---\n\n")
        f.write(f"*Generated by SHARINGAN TemporalBench Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    print(f" Summary saved to: {output_file}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TemporalBench Long Video COIN benchmark')
    parser.add_argument('--qa-file', type=str, 
                       default='benchmarking/videomme/long_video_coin/temporalbench_long_qa.json',
                       help='Path to QA JSON file')
    parser.add_argument('--dataset-dir', type=str,
                       default='benchmarking/videomme/long_video_coin/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str,
                       default='benchmarking/videomme/long_video_coin/results',
                       help='Output directory for results')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to process (for testing)')
    parser.add_argument('--target-fps', type=float, default=5.0,
                       help='Target FPS for video processing')
    parser.add_argument('--model', type=str, default='siglip-base',
                       choices=['clip', 'siglip', 'siglip-base', 'siglip-large', 'siglip-so400m', 'smolvlm'],
                       help='Vision model to use (default: siglip-base - FAST + GOOD)')
    parser.add_argument('--enable-descriptions', action='store_true', default=True,
                       help='Generate frame descriptions using InternVL2.5 (default: True)')
    parser.add_argument('--no-descriptions', dest='enable_descriptions', action='store_false',
                       help='Disable frame descriptions (faster but less accurate)')
    parser.add_argument('--delta-captioning', action='store_true', default=False,
                       help='Only caption keyframes (6x faster but lower accuracy, default: False)')
    parser.add_argument('--no-delta-captioning', dest='delta_captioning', action='store_false',
                       help='Caption all frames (slower but more detailed)')
    parser.add_argument('--system-prompt', type=str, default=None,
                       help='Custom system prompt for LLM')
    parser.add_argument('--user-prompt-template', type=str, default=None,
                       help='Custom user prompt template')
    parser.add_argument('--enable-action-classification', action='store_true', default=True,
                       help='Enable CLIP zero-shot action classification (default: True)')
    parser.add_argument('--no-action-classification', dest='enable_action_classification', action='store_false',
                       help='Disable action classification')
    
    args = parser.parse_args()
    
    qa_file = Path(args.qa_file)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    if not qa_file.exists():
        print(f"ERROR: QA file not found: {qa_file}")
        sys.exit(1)
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        print("Please download the Long Video COIN dataset and place it in:")
        print(f"  {dataset_dir.absolute()}")
        sys.exit(1)
    
    run_benchmark(
        qa_file=qa_file,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        max_questions=args.max_questions,
        target_fps=args.target_fps,
        vlm_model=args.model,
        enable_descriptions=args.enable_descriptions,
        delta_captioning=args.delta_captioning,
        system_prompt=args.system_prompt,
        user_prompt_template=args.user_prompt_template
    )


if __name__ == "__main__":
    main()
