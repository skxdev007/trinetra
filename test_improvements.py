"""
Test script to verify Option 1 (LLM bias fix) and Option 2 (action classification) work.
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.processor import VideoProcessor
from sharingan.chat import VideoLLM


def test_action_classification():
    """Test CLIP zero-shot action classification."""
    print("\n" + "="*80)
    print("TEST 1: Action Classification")
    print("="*80)
    
    # Use a test video
    video_path = "benchmarking/videomme/long_video_coin/dataset/long_video/COIN/XY-aOfWBDSs_start_11.0_end_110.0.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return False
    
    print(f"\n📹 Processing video: {Path(video_path).name}")
    
    processor = VideoProcessor(
        vlm_model='clip',
        device='auto',
        target_fps=5.0,
        enable_descriptions=False,  # Disable for speed
        batch_size=32
    )
    
    # Process video
    processor.process(video_path)
    
    # Query with action classification enabled
    print(f"\n🔍 Querying with action classification...")
    query = "Which hand is used to tighten the screw?"
    
    results = processor.query(
        query,
        top_k=3,
        enable_action_classification=True
    )
    
    print(f"\n✓ Retrieved {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Timestamp: {result['timestamp']:.1f}s")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Description: {result['description']}")
        if 'actions' in result:
            print(f"   Actions: {result['actions']}")
    
    # Check if actions were detected
    has_actions = any('actions' in r and r['actions'] for r in results)
    
    if has_actions:
        print(f"\n✅ Action classification WORKING")
        return True
    else:
        print(f"\n❌ Action classification NOT working")
        return False


def test_option_randomization():
    """Test option randomization to eliminate bias."""
    print("\n" + "="*80)
    print("TEST 2: Option Randomization")
    print("="*80)
    
    from sharingan.chat.llm import VideoLLM
    
    llm = VideoLLM(model_name='qwen-1.5b', device='auto')
    
    # Test swapping function
    original_query = """Which caption best describes this video?
A. Person tightens screw then switches on bulb
B. Person tightens screw then switches off bulb
Answer with the option's letter from the given choices directly."""
    
    swapped_query = llm._swap_options(original_query)
    
    print(f"\nOriginal query:")
    print(original_query)
    print(f"\nSwapped query:")
    print(swapped_query)
    
    # Check if swap worked
    if "A. Person tightens screw then switches off bulb" in swapped_query:
        print(f"\n✅ Option randomization WORKING")
        return True
    else:
        print(f"\n❌ Option randomization NOT working")
        return False


def test_improved_prompting():
    """Test improved prompting with chain-of-thought."""
    print("\n" + "="*80)
    print("TEST 3: Improved Prompting")
    print("="*80)
    
    from sharingan.chat.llm import VideoLLM
    
    llm = VideoLLM(model_name='qwen-1.5b', device='auto')
    
    # Mock video context with actions
    video_context = [
        {
            'timestamp': 5.0,
            'confidence': 0.85,
            'description': 'Person holds screwdriver',
            'actions': {
                'hand_used': 'a person using their right hand',
                'screw_action': 'a person tightening a screw'
            }
        },
        {
            'timestamp': 12.0,
            'confidence': 0.78,
            'description': 'Person pulls string',
            'actions': {
                'hand_used': 'a person using their left hand',
                'light_state': 'a light bulb turning on'
            }
        }
    ]
    
    query = """Which caption best describes this video?
A. Person tightens screw then switches on bulb
B. Person tightens screw then switches off bulb
Answer with the option's letter from the given choices directly."""
    
    print(f"\nGenerating response with improved prompting...")
    response = llm.chat(query, video_context, randomize_options=False)
    
    print(f"\nResponse: {response}")
    
    # Check if response is A or B
    if response.strip().upper() in ['A', 'B']:
        print(f"\n✅ Improved prompting WORKING (got clean answer)")
        return True
    else:
        print(f"\n⚠️  Improved prompting may need tuning (got: {response})")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING OPTION 1 (LLM Bias Fix) + OPTION 2 (Action Classification)")
    print("="*80)
    
    results = []
    
    # Test 1: Action Classification
    try:
        results.append(("Action Classification", test_action_classification()))
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Action Classification", False))
    
    # Test 2: Option Randomization
    try:
        results.append(("Option Randomization", test_option_randomization()))
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Option Randomization", False))
    
    # Test 3: Improved Prompting
    try:
        results.append(("Improved Prompting", test_improved_prompting()))
    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Improved Prompting", False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print(f"\n🎉 All tests passed! Ready to run benchmark.")
        print(f"\nRun benchmark with:")
        print(f"python benchmarking/videomme/benchmark_long_video_coin.py --model clip --max-questions 20")
    else:
        print(f"\n⚠️  Some tests failed. Please review errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
