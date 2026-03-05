"""
Quick test to verify VideoMAE integration works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_videomae_encoder():
    """Test VideoMAE encoder can be imported and initialized."""
    print("Test 1: VideoMAE Encoder Import and Initialization")
    print("="*60)
    
    try:
        from sharingan.vlm.videomae_encoder import VideoMAEEncoder
        print("✓ VideoMAEEncoder imported successfully")
        
        # Test initialization (will download model on first run)
        print("\nInitializing VideoMAE encoder...")
        encoder = VideoMAEEncoder('videomae-large', 'cpu')
        print(f"✓ Encoder initialized: {encoder}")
        print(f"✓ Embedding dimension: {encoder.embedding_dim}")
        
        # Test encoding a dummy frame
        import numpy as np
        print("\nTesting frame encoding...")
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embedding = encoder.encode_frame(frame)
        print(f"✓ Frame encoded successfully")
        print(f"✓ Embedding shape: {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor_integration():
    """Test VideoProcessor with VideoMAE."""
    print("\n\nTest 2: VideoProcessor Integration")
    print("="*60)
    
    try:
        from sharingan.processor import VideoProcessor
        print("✓ VideoProcessor imported successfully")
        
        # Test initialization with VideoMAE
        print("\nInitializing VideoProcessor with VideoMAE...")
        processor = VideoProcessor(
            vlm_model='videomae',
            llm_model='qwen-1.5b',
            device='cpu'
        )
        print(f"✓ Processor initialized")
        print(f"  VLM Model: {processor.vlm_model}")
        print(f"  LLM Model: {processor.llm_model}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_model_selection():
    """Test LLM model selection."""
    print("\n\nTest 3: LLM Model Selection")
    print("="*60)
    
    try:
        from sharingan.chat.llm import VideoLLM
        print("✓ VideoLLM imported successfully")
        
        # Test with qwen-0.5b (default)
        print("\nTesting Qwen-0.5B...")
        llm_05b = VideoLLM(model_name='qwen-0.5b', device='cpu')
        print(f"✓ Qwen-0.5B initialized: {llm_05b}")
        
        # Test with qwen-1.5b
        print("\nTesting Qwen-1.5B...")
        llm_15b = VideoLLM(model_name='qwen-1.5b', device='cpu')
        print(f"✓ Qwen-1.5B initialized: {llm_15b}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VIDEOMAE INTEGRATION TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("VideoMAE Encoder", test_videomae_encoder()))
    results.append(("Processor Integration", test_processor_integration()))
    results.append(("LLM Model Selection", test_llm_model_selection()))
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nYou can now run the benchmark:")
        print("  python benchmarking/videomme/run_coin_benchmark_videomae.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the errors above before running the benchmark.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
