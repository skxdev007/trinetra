"""Quick test of SigLIP support on a sample video."""

import time
from pathlib import Path
from datetime import datetime
from sharingan.processor import VideoProcessor

# Test video
VIDEO_PATH = "benchmarking/videomme/long_video_coin/dataset/long_video/COIN/WIsYqmhKo_I_start_78.0_end_224.0.mp4"

# Test queries
TEST_QUERIES = [
    "What is the person doing?",
    "What tools or objects are visible?",
    "What happens at the beginning?",
    "What happens at the end?",
    "What is the main activity in this video?"
]

def test_model(model_name: str, device: str = 'auto'):
    """Test a model on the video."""
    print(f"\n{'='*80}")
    print(f"Testing {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Initialize processor
    print(f"🚀 Initializing processor with {model_name.upper()}...")
    processor = VideoProcessor(
        vlm_model=model_name,
        device=device,
        target_fps=5.0,
        enable_temporal=True
    )
    
    # Process video
    print(f"\n📹 Processing video: {VIDEO_PATH}")
    start_time = time.time()
    
    try:
        result = processor.process(VIDEO_PATH)
        process_time = time.time() - start_time
        
        print(f"\n✓ Video processed in {process_time:.2f} seconds")
        print(f"  - Frames processed: {result.get('num_frames', len(result.get('embeddings', [])))}")
        print(f"  - Video duration: {result.get('duration', 0):.1f}s")
        print(f"  - Events detected: {len(result.get('events', []))}")
        
    except Exception as e:
        print(f"\n❌ Failed to process video: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Run test queries
    print(f"\n🔍 Running {len(TEST_QUERIES)} test queries...")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {query}")
        
        query_start = time.time()
        
        try:
            matches = processor.query(query, top_k=3)
            query_time = time.time() - query_start
            
            print(f"   ⏱️  Query time: {query_time*1000:.0f}ms")
            print(f"   📍 Top matches:")
            for j, match in enumerate(matches, 1):
                print(f"      {j}. {match['timestamp']:.1f}s (confidence: {match['confidence']:.3f})")
            
        except Exception as e:
            print(f"   ❌ Query failed: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Test completed for {model_name.upper()}")
    print(f"{'='*80}\n")

def main():
    """Run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick SigLIP test')
    parser.add_argument('--model', default='siglip', choices=['clip', 'siglip', 'siglip-base', 'siglip-large'],
                       help='Model to test')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--compare', action='store_true', help='Compare CLIP vs SigLIP')
    
    args = parser.parse_args()
    
    if args.compare:
        print("\n" + "="*80)
        print("COMPARING CLIP vs SigLIP")
        print("="*80)
        
        test_model('clip', args.device)
        test_model('siglip', args.device)
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
    else:
        test_model(args.model, args.device)

if __name__ == '__main__':
    main()
