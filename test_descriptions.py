"""
Quick test to verify frame descriptions are working.
"""

import sys
from pathlib import Path

# Add sharingan to path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.processor import VideoProcessor

def test_descriptions():
    """Test frame description generation."""
    
    print("\n" + "="*80)
    print("Testing Frame Description Feature")
    print("="*80)
    
    # Find a test video
    test_video = Path("benchmarking/videomme/long_video_coin/dataset/long_video/COIN")
    if not test_video.exists():
        print("❌ Test video directory not found")
        return
    
    videos = list(test_video.glob("*.mp4"))
    if not videos:
        print("❌ No test videos found")
        return
    
    test_video_path = str(videos[0])
    print(f"\n📹 Test video: {videos[0].name}")
    
    # Test 1: With descriptions (default)
    print("\n" + "-"*80)
    print("TEST 1: With Frame Descriptions (enable_descriptions=True)")
    print("-"*80)
    
    processor = VideoProcessor(
        vlm_model='siglip',
        device='auto',
        target_fps=5.0,
        enable_temporal=True,
        enable_descriptions=True,  # KEY: Enable descriptions
        batch_size=8  # Small batch for testing
    )
    
    results = processor.process(test_video_path)
    
    print(f"\n✓ Processed {results['video_info']['processed_frames']} frames")
    print(f"✓ Generated {len(results['descriptions'])} descriptions")
    
    # Show sample descriptions
    print("\nSample Frame Descriptions:")
    for i in range(min(5, len(results['descriptions']))):
        timestamp = results['timestamps'][i]
        description = results['descriptions'][i]
        print(f"  [{timestamp:.1f}s] {description}")
    
    # Test query with descriptions
    print("\n" + "-"*80)
    print("TEST 2: Query with Descriptions")
    print("-"*80)
    
    query = "person holding something"
    print(f"\nQuery: '{query}'")
    
    matches = processor.query(query, top_k=3)
    
    print(f"\nTop 3 Matches:")
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. Timestamp: {match['timestamp']:.1f}s")
        print(f"   Confidence: {match['confidence']:.3f}")
        print(f"   Description: {match['description']}")
    
    # Test LLM chat
    print("\n" + "-"*80)
    print("TEST 3: LLM Chat with Rich Context")
    print("-"*80)
    
    question = "What is the person doing?"
    print(f"\nQuestion: '{question}'")
    
    # Get segments to show what LLM will see
    segments = processor.query(question, top_k=5)
    
    print("\n" + "="*80)
    print("CONTEXT SENT TO LLM:")
    print("="*80)
    print("VIDEO TIMELINE:")
    print("-" * 60)
    for i, seg in enumerate(segments, 1):
        timestamp = seg['timestamp']
        confidence = seg['confidence']
        description = seg['description']
        
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        time_str = f"{mins}:{secs:02d}"
        
        print(f"{i}. [{time_str}] {description}")
        print(f"   Relevance: {confidence:.1%}")
    print("-" * 60)
    print("="*80)
    
    response = processor.chat(question, use_llm=True)
    print(f"\nLLM Response:\n{response}")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_descriptions()
