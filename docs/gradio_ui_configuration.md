# Gradio UI Configuration Panel

## Overview

The SHARINGAN Gradio UI includes an advanced configuration panel that allows you to customize all aspects of the video processing pipeline. This document explains each configuration option and when to adjust them.

## Accessing the Configuration Panel

The configuration panel is located in the left column under "⚙️ Advanced Configuration". Click the accordion to expand it and see all available settings.

## Configuration Sections

### 1. Multi-Scale TAS Settings

Multi-Scale Temporal Attention Shift (TAS) processes video at three different temporal scales to understand gestures, actions, and scenes.

**Short-scale Kernel Size** (default: 2)
- Range: 1-8 frames
- Purpose: Captures quick gestures and cuts
- When to adjust: Increase for slower gestures, decrease for very rapid movements

**Mid-scale Kernel Size** (default: 8)
- Range: 4-16 frames
- Purpose: Captures actions and interactions
- When to adjust: Increase for longer actions, decrease for quick interactions

**Long-scale Kernel Size** (default: 32)
- Range: 16-64 frames
- Purpose: Captures scenes and narrative context
- When to adjust: Increase for slow-paced videos, decrease for fast-paced content

**TAS Window Size** (default: 64)
- Range: 32-128 frames
- Purpose: Maximum frames for long-scale attention
- When to adjust: Increase for videos requiring very long-range context (may increase memory usage)

### 2. Adaptive Sampler Settings

The adaptive sampler intelligently adjusts frame sampling rate based on visual change detection.

**Base FPS** (default: 1.0)
- Range: 0.5-5.0 FPS
- Purpose: Sampling rate for static scenes
- When to adjust: Increase for videos with subtle changes, decrease to save processing time

**Max FPS** (default: 5.0)
- Range: 1.0-10.0 FPS
- Purpose: Maximum sampling rate during high motion
- When to adjust: Increase for fast-paced videos, decrease to reduce processing time

**Change Threshold** (default: 0.3)
- Range: 0.1-0.9
- Purpose: Threshold for detecting scene changes
- When to adjust: Lower for more sensitive detection, higher for less sensitive

### 3. Cross-Modal Verifier Settings

The cross-modal verifier uses CLIP to detect VLM hallucinations by comparing descriptions to visual evidence.

**Similarity Threshold** (default: 0.7)
- Range: 0.5-0.95
- Purpose: CLIP similarity threshold for verification
- When to adjust: Lower for stricter verification (more descriptions flagged), higher for lenient verification

**Entity Threshold** (default: 0.5)
- Range: 0.3-0.8
- Purpose: Threshold for entity verification
- When to adjust: Lower to catch more entity hallucinations, higher to reduce false positives

### 4. Causal Edge Scorer Settings

The causal edge scorer determines relationships between events in the temporal graph.

**Scorer Mode** (default: heuristic)
- Options: heuristic, learned
- Purpose: Method for scoring causal relationships
- When to use:
  - **Heuristic**: Fast, no training needed, uses cosine similarity
  - **Learned**: More accurate, requires trained model (V2 feature)

**Causal Threshold (Heuristic)** (default: 0.7)
- Range: 0.5-0.95
- Purpose: Similarity threshold for causal edges
- When to adjust: Lower to detect more causal relationships, higher for stricter causality

**Semantic Threshold (Heuristic)** (default: 0.5)
- Range: 0.3-0.8
- Purpose: Similarity threshold for semantic edges
- When to adjust: Lower to detect more semantic relationships, higher for stricter semantics

### 5. Model Selection

Choose which models to use for different components of the pipeline.

**SmolVLM Model** (default: HuggingFaceTB/SmolVLM-500M-Instruct)
- Options:
  - HuggingFaceTB/SmolVLM-500M-Instruct (538 MB, more accurate)
  - HuggingFaceTB/SmolVLM-256M-Instruct (256 MB, faster)
- Purpose: Vision-language model for frame descriptions
- When to adjust: Use 256M for faster processing with less memory

**CLIP Model** (default: openai/clip-vit-base-patch32)
- Options:
  - openai/clip-vit-base-patch32 (400 MB, faster)
  - openai/clip-vit-large-patch14 (890 MB, more accurate)
- Purpose: Model for cross-modal verification
- When to adjust: Use large model for better hallucination detection

**Qwen LLM Model** (default: Qwen/Qwen2.5-0.5B-Instruct)
- Options:
  - Qwen/Qwen2.5-0.5B-Instruct (538 MB, fastest)
  - Qwen/Qwen2.5-1.5B-Instruct (1.5 GB, more capable)
  - Qwen/Qwen2.5-3B-Instruct (3 GB, most capable)
- Purpose: Language model for query responses
- When to adjust: Use larger models for more complex reasoning

**SmolVLM Context Window** (default: 8)
- Range: 1-16 frames
- Purpose: Number of previous frames for context
- When to adjust: Increase for videos requiring more temporal context, decrease to save memory

## Recommended Configurations

### Fast Processing (Low Memory)
- TAS Kernels: 2/8/32 (default)
- TAS Window: 32
- Base FPS: 0.5
- Max FPS: 3.0
- SmolVLM: 256M variant
- CLIP: base-patch32
- Qwen: 0.5B
- Context Window: 4

### Balanced (Default)
- TAS Kernels: 2/8/32
- TAS Window: 64
- Base FPS: 1.0
- Max FPS: 5.0
- SmolVLM: 500M variant
- CLIP: base-patch32
- Qwen: 0.5B
- Context Window: 8

### High Accuracy (High Memory)
- TAS Kernels: 2/8/32
- TAS Window: 128
- Base FPS: 2.0
- Max FPS: 10.0
- SmolVLM: 500M variant
- CLIP: large-patch14
- Qwen: 1.5B
- Context Window: 16

### Long Videos (>1 hour)
- TAS Kernels: 2/8/64 (longer scenes)
- TAS Window: 64
- Base FPS: 0.5 (reduce sampling)
- Max FPS: 3.0
- SmolVLM: 256M (save memory)
- CLIP: base-patch32
- Qwen: 0.5B
- Context Window: 8

### Fast-Paced Videos (Sports, Action)
- TAS Kernels: 2/4/16 (shorter scales)
- TAS Window: 32
- Base FPS: 2.0 (higher base rate)
- Max FPS: 10.0
- Change Threshold: 0.2 (more sensitive)
- SmolVLM: 500M
- CLIP: base-patch32
- Qwen: 0.5B
- Context Window: 4

### Slow-Paced Videos (Lectures, Interviews)
- TAS Kernels: 4/16/64 (longer scales)
- TAS Window: 128
- Base FPS: 0.5
- Max FPS: 2.0
- Change Threshold: 0.4 (less sensitive)
- SmolVLM: 500M
- CLIP: base-patch32
- Qwen: 1.5B (better reasoning)
- Context Window: 16

## Performance Impact

### Memory Usage
- TAS Window Size: +50 MB per 32 frames
- SmolVLM Context Window: +20 MB per frame
- Model sizes: SmolVLM (256-538 MB), CLIP (400-890 MB), Qwen (538 MB-3 GB)

### Processing Time
- Higher FPS: Linear increase in processing time
- Larger TAS kernels: ~10% increase per doubling
- Larger models: 2-3x slower for large variants

### Accuracy
- Larger TAS window: Better long-range reasoning
- Higher FPS: More events detected
- Larger models: Better descriptions and reasoning
- Lower thresholds: More relationships detected (may include false positives)

## Troubleshooting

### Out of Memory Errors
1. Reduce TAS window size to 32
2. Reduce SmolVLM context window to 4
3. Use smaller models (256M SmolVLM, 0.5B Qwen)
4. Reduce max FPS to 3.0

### Processing Too Slow
1. Reduce max FPS to 3.0
2. Reduce base FPS to 0.5
3. Use smaller models
4. Reduce TAS window to 32

### Poor Event Detection
1. Increase max FPS to 10.0
2. Lower change threshold to 0.2
3. Increase base FPS to 2.0

### Poor Causal Reasoning
1. Increase TAS window to 128
2. Increase long-scale kernel to 64
3. Lower causal threshold to 0.6
4. Use larger Qwen model (1.5B or 3B)

### Too Many False Positives
1. Increase verifier threshold to 0.8
2. Increase causal threshold to 0.8
3. Increase semantic threshold to 0.6

## Requirements Mapping

This configuration panel implements the following requirements from the spec:

- **Requirement 20.1**: Multi-Scale TAS kernel sizes and window size configuration
- **Requirement 20.2**: Adaptive Sampler FPS bounds and change threshold configuration
- **Requirement 20.3**: Cross-Modal Verifier similarity threshold configuration
- **Requirement 20.4**: Causal Edge Scorer mode (heuristic vs learned) configuration
- **Requirement 20.5**: SmolVLM context window size configuration
- **Requirement 20.6**: Query Router classification model configuration (via Qwen model selection)

## API Usage

The configuration is passed to the `VideoProcessor` via a config dictionary:

```python
config = {
    'tas_config': {
        'kernel_sizes': [2, 8, 32],
        'window_size': 64,
        'causal': True
    },
    'sampler_config': {
        'base_fps': 1.0,
        'max_fps': 5.0,
        'change_threshold': 0.3
    },
    'verifier_config': {
        'threshold': 0.7,
        'entity_threshold': 0.5,
        'clip_model': 'openai/clip-vit-base-patch32'
    },
    'scorer_config': {
        'mode': 'heuristic',
        'causal_threshold': 0.7,
        'semantic_threshold': 0.5
    },
    'model_config': {
        'smolvlm_model': 'HuggingFaceTB/SmolVLM-500M-Instruct',
        'clip_model': 'openai/clip-vit-base-patch32',
        'qwen_model': 'Qwen/Qwen2.5-0.5B-Instruct',
        'smolvlm_context_window': 8
    }
}

processor = VideoProcessor(config=config)
```

## Future Enhancements

Planned for future versions:

1. **Preset Profiles**: Save and load configuration presets
2. **Auto-tuning**: Automatically adjust settings based on video characteristics
3. **Real-time Preview**: See configuration impact before processing
4. **Batch Configuration**: Apply different configs to different video segments
5. **Performance Profiling**: Show detailed breakdown of processing time per component
