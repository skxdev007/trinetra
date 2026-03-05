"""Quick test of Kinetics-400 action classifier."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sharingan.vlm.action_classifier import ActionClassifier
import numpy as np

print("Testing Kinetics-400 Action Classifier...")
print("="*60)

# Initialize classifier
classifier = ActionClassifier(embedding_dim=1024, device='cuda', use_videomae_classifier=True)

print(f"\nClassifier initialized successfully!")
print(f"Using VideoMAE classifier: {classifier.use_videomae_classifier}")

if classifier.videomae_model:
    print(f"Number of classes: {len(classifier.videomae_model.config.id2label)}")
    print(f"\nSample Kinetics-400 labels:")
    for i in range(10):
        print(f"  {i}: {classifier.videomae_model.config.id2label[i]}")

print("\n" + "="*60)
print("✓ Kinetics-400 classifier ready!")
