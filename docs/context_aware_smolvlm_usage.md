# Context-Aware SmolVLM Usage Guide

## Overview

The `ContextAwareSmolVLM` class provides temporally coherent frame descriptions by maintaining a rolling context buffer of up to 8 previous frames. This reduces hallucinations and improves description consistency across video timelines.

## Key Features

- **Rolling Context Buffer**: Maintains up to 8 previous frames (FIFO)
- **Temporal Coherence**: Descriptions build on previous observations
- **Continuous Time Encoding**: Uses sinusoidal encoding for timestamps
- **Entity & Action Extraction**: Automatically parses entities and actions from descriptions
- **Confidence Scoring**: Provides confidence scores for each description

## Basic Usage

```python
from sharingan.vlm import ContextAwareSmolVLM, FrameDescription
from sharingan.video.sampler import AdaptiveSampler
import numpy as np

# Initialize context-aware SmolVLM
vlm = ContextAwareSmolVLM(
    model_name="HuggingFaceTB/SmolVLM-500M-Instruct",
    context_window=8,  # Keep up to 8 previous frames
    device="cuda"      # or "cpu" or "auto"
)

# Process frames from adaptive sampler
sampler = AdaptiveSampler(base_fps=1.0, change_threshold=0.3)

for frame_idx, frame, change_score in sampler.sample_adaptive(video_loader):
    timestamp = frame_idx / video_loader.fps
    
    # Generate description with context
    description = vlm.describe_with_context(
        current_frame=frame,
        timestamp=timestamp,
        frame_index=frame_idx
    )
    
    # Access description components
    print(f"Frame {frame_idx} at {timestamp:.2f}s:")
    print(f"  Description: {description.description}")
    print(f"  Entities: {description.entities}")
    print(f"  Actions: {description.actions}")
    print(f"  Confidence: {description.confidence:.2f}")
    print(f"  Context used: {description.context_used}")
    
    # Update context buffer for next frame
    vlm.update_context(frame, description.description, frame_idx, timestamp)
```

## FrameDescription Output

Each call to `describe_with_context()` returns a `FrameDescription` object with:

- `timestamp`: Continuous time in seconds (float)
- `frame_index`: Original frame index in video (int)
- `description`: Natural language description (str)
- `entities`: List of detected entities (List[str])
- `actions`: List of detected actions (List[str])
- `confidence`: Model confidence score 0.0-1.0 (float)
- `context_used`: Frame indices used as context (List[int])

## Context Management

### Checking Context Size

```python
# Get current number of frames in context buffer
context_size = vlm.get_context_size()
print(f"Context buffer contains {context_size} frames")
```

### Clearing Context

```python
# Clear context buffer (useful when starting a new video or after scene cuts)
vlm.clear_context()
```

### FIFO Behavior

The context buffer automatically implements FIFO (First-In-First-Out):
- When buffer size < 8: New frames are appended
- When buffer size = 8: Oldest frame is removed before adding new frame

## Integration with Ingest Pipeline

```python
from sharingan.vlm import ContextAwareSmolVLM
from sharingan.video.sampler import AdaptiveSampler
from sharingan.verification.cross_modal import CrossModalVerifier

# Initialize components
vlm = ContextAwareSmolVLM(context_window=8, device="cuda")
sampler = AdaptiveSampler(base_fps=1.0, change_threshold=0.3)
verifier = CrossModalVerifier(threshold=0.7)

# Process video
for frame_idx, frame, change_score in sampler.sample_adaptive(video_loader):
    timestamp = frame_idx / video_loader.fps
    
    # Generate description with context
    description = vlm.describe_with_context(frame, timestamp, frame_idx)
    
    # Verify description against visual evidence
    verification = verifier.verify_description(
        frame=frame,
        description=description.description,
        entities=description.entities
    )
    
    if not verification.is_verified:
        print(f"⚠️  Low confidence description at {timestamp:.2f}s")
        print(f"   Alignment score: {verification.alignment_score:.2f}")
        print(f"   Flagged entities: {verification.flagged_entities}")
    
    # Update context for next frame
    vlm.update_context(frame, description.description, frame_idx, timestamp)
```

## Requirements Validated

This implementation validates the following requirements:

- **Requirement 2.1**: Maintains rolling context buffer of up to 8 previous frames
- **Requirement 2.2**: Includes context from up to 8 previous frames in prompt
- **Requirement 2.3**: Encodes continuous timestamps using sinusoidal positional encoding
- **Requirement 2.4**: Returns frame descriptions with all required fields
- **Requirement 2.5**: Removes oldest frame when buffer exceeds 8 frames (FIFO)
- **Requirement 2.6**: Records which context frames were used for each description

## Performance Considerations

- **Context Window Size**: Default 8 frames balances context richness with prompt length
- **Memory Usage**: Stores 8 frames + descriptions (manageable for long videos)
- **Processing Time**: ~150ms per frame on GPU (includes SmolVLM inference)
- **Prompt Length**: 8 descriptions ≈ 800 tokens, leaving room for current frame analysis

## Advanced Usage

### Custom Context Window

```python
# Use smaller context window for faster processing
vlm = ContextAwareSmolVLM(context_window=4, device="cuda")

# Use larger context window for more context (may increase prompt length)
vlm = ContextAwareSmolVLM(context_window=12, device="cuda")
```

### Custom Max Tokens

```python
# Generate longer descriptions
description = vlm.describe_with_context(
    current_frame=frame,
    timestamp=timestamp,
    frame_index=frame_idx,
    max_new_tokens=200  # Default is 150
)
```

## Troubleshooting

### Issue: Descriptions are inconsistent

**Solution**: Ensure you're calling `update_context()` after each frame to maintain the context buffer.

### Issue: Low confidence scores

**Solution**: Check if the frame quality is good and if entities/actions are being parsed correctly. Consider using CrossModalVerifier to validate descriptions.

### Issue: Out of memory

**Solution**: Reduce context window size or use CPU instead of GPU for inference.

## Next Steps

After generating frame descriptions with ContextAwareSmolVLM:

1. **Cross-Modal Verification**: Use `CrossModalVerifier` to detect hallucinations
2. **Event Detection**: Group frames into semantic events
3. **Causal Edge Scoring**: Score causal relationships between events
4. **Hierarchical Memory**: Store descriptions in frame/event/chapter hierarchy
