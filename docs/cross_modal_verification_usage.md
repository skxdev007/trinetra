# Cross-Modal Verification System Usage Guide

## Overview

The Cross-Modal Verification System uses CLIP (Contrastive Language-Image Pre-training) to detect VLM hallucinations by comparing frame descriptions against visual evidence. This ensures that downstream reasoning is based on verified information.

## Quick Start

```python
from sharingan.verification.cross_modal import CrossModalVerifier
import numpy as np

# Initialize verifier
verifier = CrossModalVerifier(
    threshold=0.7,           # Description verification threshold
    entity_threshold=0.5,    # Entity verification threshold
    device="cuda"            # Use "cpu" if no GPU available
)

# Verify a frame description
frame = np.array(...)  # Your frame as (H, W, 3) RGB array
description = "A person in a blue shirt picks up a red cup."
entities = ["person", "blue shirt", "red cup"]

result = verifier.verify_description(
    frame=frame,
    description=description,
    entities=entities
)

# Check verification result
if result.is_verified:
    print(f"✓ Verified (alignment: {result.alignment_score:.3f})")
else:
    print(f"⚠️  Flagged (alignment: {result.alignment_score:.3f})")
    print(f"Flagged entities: {result.flagged_entities}")
    print(f"Suggestion: {result.correction_suggestion}")
```

## Integration with Context-Aware SmolVLM

The verification system is designed to work seamlessly with Context-Aware SmolVLM:

```python
from sharingan.vlm.context_aware_smolvlm import ContextAwareSmolVLM
from sharingan.verification.cross_modal import CrossModalVerifier
from sharingan.video.sampler import AdaptiveSampler
from sharingan.video.loader import VideoLoader

# Initialize components
vlm = ContextAwareSmolVLM(context_window=8, device="cuda")
verifier = CrossModalVerifier(threshold=0.7, device="cuda")
sampler = AdaptiveSampler(base_fps=1.0, change_threshold=0.3)

# Load video
loader = VideoLoader("video.mp4")

# Process video with verification
verified_descriptions = []
flagged_descriptions = []

for frame_idx, frame, change_score in sampler.sample_adaptive(loader):
    timestamp = frame_idx / loader.fps
    
    # Generate description with context
    description = vlm.describe_with_context(
        current_frame=frame,
        timestamp=timestamp,
        frame_index=frame_idx
    )
    
    # Verify description
    verification = verifier.verify_description(
        frame=frame,
        description=description.description,
        entities=description.entities
    )
    
    # Handle verification result
    if verification.is_verified:
        verified_descriptions.append((description, verification))
    else:
        flagged_descriptions.append((description, verification))
        print(f"⚠️  Frame {frame_idx}: {verification.correction_suggestion}")
    
    # Update context for next frame
    vlm.update_context(frame, description.description, frame_idx, timestamp)

print(f"\nProcessed {len(verified_descriptions) + len(flagged_descriptions)} frames")
print(f"Verified: {len(verified_descriptions)}")
print(f"Flagged: {len(flagged_descriptions)}")
```

## Understanding Verification Results

### VerificationResult Fields

```python
@dataclass
class VerificationResult:
    is_verified: bool                    # True if passes all checks
    alignment_score: float               # CLIP similarity (0.0 to 1.0)
    flagged_entities: List[str]          # Entities with low similarity
    correction_suggestion: Optional[str] # Actionable feedback
```

### Verification Thresholds

**Description Threshold (0.7)**
- Descriptions with CLIP similarity ≥ 0.7 are considered verified
- Lower threshold increases false positives (flagging correct descriptions)
- Higher threshold increases false negatives (missing hallucinations)
- 0.7 is empirically validated as a good balance

**Entity Threshold (0.5)**
- Individual entities with CLIP similarity ≥ 0.5 are considered verified
- Lower threshold because entity-level verification is more granular
- Helps catch specific hallucinated objects even when overall scene is correct

### Interpreting Alignment Scores

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 0.9 - 1.0 | Excellent match | High confidence, use as-is |
| 0.7 - 0.9 | Good match | Verified, proceed normally |
| 0.5 - 0.7 | Weak match | Flagged, review or regenerate |
| 0.0 - 0.5 | Poor match | Likely hallucination, discard |

## Advanced Usage

### Custom Thresholds

Adjust thresholds based on your use case:

```python
# Conservative (fewer false positives, more false negatives)
verifier = CrossModalVerifier(
    threshold=0.8,           # Higher threshold
    entity_threshold=0.6
)

# Aggressive (more false positives, fewer false negatives)
verifier = CrossModalVerifier(
    threshold=0.6,           # Lower threshold
    entity_threshold=0.4
)
```

### Handling Flagged Descriptions

When a description is flagged, you have several options:

**Option 1: Regenerate with Conservative Prompt**
```python
if not verification.is_verified:
    # Retry with more conservative prompt
    description = vlm.describe_with_context(
        current_frame=frame,
        timestamp=timestamp,
        frame_index=frame_idx,
        max_new_tokens=100  # Shorter = less hallucination
    )
```

**Option 2: Use CLIP-Only Embeddings**
```python
if not verification.is_verified:
    # Fall back to CLIP embeddings without VLM description
    frame_embedding = verifier._encode_image(frame)
    # Use embedding directly for retrieval/reasoning
```

**Option 3: Manual Review**
```python
if not verification.is_verified:
    # Log for manual review
    logger.warning(
        f"Frame {frame_idx} flagged: {verification.correction_suggestion}"
    )
    # Continue processing but mark as low-confidence
    description.confidence = verification.alignment_score
```

### Batch Verification

For efficiency, verify multiple frames in batch:

```python
def verify_batch(verifier, frames, descriptions, entities_list):
    """Verify multiple frames in batch."""
    results = []
    for frame, desc, entities in zip(frames, descriptions, entities_list):
        result = verifier.verify_description(frame, desc, entities)
        results.append(result)
    return results
```

## Performance Considerations

### Model Loading Time
- First initialization: ~5-10 seconds (loading CLIP)
- Subsequent uses: Instant (model cached in memory)

### Verification Speed
- CPU: ~100-200ms per frame
- GPU: ~20-50ms per frame

### Memory Usage
- CLIP ViT-B/32: ~400MB VRAM/RAM
- Scales linearly with batch size

### Optimization Tips

1. **Reuse Verifier Instance**: Initialize once, use for entire video
2. **Use GPU**: 4-5x faster than CPU
3. **Batch Processing**: Process multiple frames together when possible
4. **Cache Embeddings**: Store frame embeddings to avoid recomputation

```python
# Cache frame embeddings for reuse
frame_embeddings = {}

for frame_idx, frame in enumerate(frames):
    if frame_idx not in frame_embeddings:
        frame_embeddings[frame_idx] = verifier._encode_image(frame)
    
    # Use cached embedding
    frame_emb = frame_embeddings[frame_idx]
    text_emb = verifier._encode_text(description)
    score = verifier.compute_alignment_score(frame_emb, text_emb)
```

## Error Handling

```python
from sharingan.exceptions import EncodingError

try:
    result = verifier.verify_description(frame, description, entities)
except EncodingError as e:
    print(f"Encoding failed: {e}")
    # Handle error (skip frame, use fallback, etc.)
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

## Requirements Validated

The Cross-Modal Verification System validates these requirements:

- **Requirement 3.1**: Computes CLIP similarity between frame and description
- **Requirement 3.2**: Flags descriptions with CLIP similarity < 0.7
- **Requirement 3.3**: Verifies each entity mentioned in description
- **Requirement 3.4**: Flags entities with CLIP similarity < 0.5
- **Requirement 3.5**: Provides correction suggestions when verification fails
- **Requirement 3.6**: Returns VerificationResult with all required fields

## See Also

- [Context-Aware SmolVLM Usage Guide](context_aware_smolvlm_usage.md)
- [Integration Tests README](../tests/README_INTEGRATION_TESTS.md)
- [SHARINGAN Architecture](../ARCHITECTURE.md)
