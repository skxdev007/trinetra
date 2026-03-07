"""Test SigLIP-SO400M encoder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.vlm import FrameEncoder
import numpy as np

print("="*80)
print("Testing SigLIP-SO400M (Best Vision Encoder)")
print("="*80)

# Create encoder
print("\n1. Loading SigLIP-SO400M...")
encoder = FrameEncoder(model_name='siglip-so400m', device='auto')

# Create test frames
print("\n2. Creating test frames...")
frames = [np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8) for _ in range(4)]

# Encode frames
print("\n3. Encoding frames...")
embeddings = encoder.encode_batch(frames)

print(f"\n✓ Embeddings shape: {embeddings.shape}")
print(f"✓ Embedding dim: {embeddings.shape[1]}D (should be 1152D)")
print(f"✓ Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
print(f"✓ Embedding norm: {np.linalg.norm(embeddings[0]):.3f} (should be ~1.0)")

# Test text encoding
print("\n4. Testing text encoding...")
text_emb = encoder.encode_text("a person using their right hand to tighten a screw")
print(f"✓ Text embedding shape: {text_emb.shape}")

# Test similarity
similarity = np.dot(embeddings[0], text_emb)
print(f"✓ Frame-text similarity: {similarity:.3f}")

print("\n" + "="*80)
print("SigLIP-SO400M Test Complete!")
print("="*80)
print("\nSigLIP-SO400M is the BEST vision encoder because:")
print("  • 400M parameters (vs CLIP's 150M)")
print("  • 384x384 resolution (vs CLIP's 224x224)")
print("  • Sigmoid loss training (better calibration)")
print("  • 1152D embeddings (richer features)")
print("  • Better fine-grained understanding")
print("\nReady to run benchmark with:")
print("  python benchmarking/videomme/benchmark_long_video_coin.py --model siglip-so400m --max-questions 20")
