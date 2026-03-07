# SHARINGAN Current Architecture Status

## What's Actually Running (March 2026)

### ✅ ENABLED Components

#### 1. Vision Encoder: SigLIP-Base-Patch16-384
- **Model**: `google/siglip-base-patch16-384`
- **Embedding Dimension**: 768D
- **Why**: Better than CLIP for fine-grained temporal understanding
- **Performance**: Faster than CLIP, better accuracy on TemporalBench

#### 2. Temporal Reasoning: CrossFrameGatingNetwork + TemporalMemoryTokens
```python
engine = TemporalEngine([
    CrossFrameGatingNetwork(feature_dim=768),
    TemporalMemoryTokens(num_tokens=8, token_dim=768)
])
```

**What it does:**
- **CrossFrameGatingNetwork**: Gates information flow between consecutive frames
  - Learns which temporal information to keep/discard
  - Helps with motion understanding and action boundaries
  
- **TemporalMemoryTokens**: 8 learnable tokens that accumulate context
  - Acts like a "working memory" for the video
  - Helps maintain long-range dependencies

**What it DOESN'T do:**
- ❌ NOT using TAS (Temporal Attention Shift) - it's implemented but not enabled
- ❌ NOT using GRU - it's implemented but not enabled
- ❌ NOT using multi-scale temporal processing

#### 3. Event Detection
- **Detector**: EventDetector with sensitivity=0.5
- **Method**: Detects changes in embedding space
- **Output**: Events with timestamps, confidence, descriptions
- **Usage**: Currently detected but NOT used for LLM context (yet)

#### 4. Frame Descriptions: InternVL2.5-1B (Lazy Mode)
- **Model**: `OpenGVLab/InternVL2_5-1B`
- **Mode**: Lazy (generates descriptions only for retrieved frames)
- **Why**: 2.6x faster than full captioning
- **VRAM**: ~3.8GB when loaded, unloads before Qwen

#### 5. LLM Reasoning: Qwen2.5-1.5B-Instruct (4-bit)
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization**: 4-bit (BitsAndBytes)
- **VRAM**: ~900MB
- **Context Format**: Event-grouped (NEW - just implemented)

#### 6. Retrieval: Cosine Similarity + Magnet Suppression
- **Method**: Dense retrieval with cosine similarity
- **Top-K**: 10 frames
- **Magnet Suppression**: Enforces temporal diversity (prevents clustering)
- **Threshold**: 5 seconds minimum gap between retrieved frames

### ❌ DISABLED/NOT USED Components

#### 1. TAS (Temporal Attention Shift)
- **Status**: Implemented but NOT enabled
- **Location**: `sharingan/temporal/tas.py`
- **Why disabled**: CrossFrameGatingNetwork + TemporalMemoryTokens are simpler and working

#### 2. Multi-Scale TAS
- **Status**: Implemented but NOT enabled
- **Location**: `sharingan/temporal/multi_scale_tas.py`
- **What it would do**: Process at 3 scales (short/mid/long)
- **Why disabled**: Not needed for current accuracy level

#### 3. GRU (Gated Recurrent Unit)
- **Status**: Implemented but NOT enabled
- **What it would do**: Sequential temporal modeling
- **Why disabled**: TemporalMemoryTokens serve similar purpose

#### 4. Entity Tracking
- **Status**: Implemented but NOT enabled
- **Flag**: `enable_tracking=False`
- **What it would do**: Track objects/people across frames
- **Why disabled**: Not needed for TemporalBench COIN

#### 5. Delta Captioning
- **Status**: Implemented but DISABLED
- **Flag**: `delta_captioning=False`
- **What it would do**: Only caption keyframes with attention shifts
- **Why disabled**: Lazy descriptions are better (2.6x faster, same accuracy)

#### 6. Temporal Event Graph (TEG)
- **Status**: Implemented but NOT used for LLM context
- **Location**: `sharingan/graph/event_graph.py`
- **What it would do**: Build causal graph of events
- **Why not used**: Event detection runs but graph not built/used

## Current Pipeline Flow

### Video Processing (Once)
```
1. Load video → Extract frames @ 5 FPS
2. Encode frames → SigLIP-Base → 768D embeddings
3. Temporal reasoning → CrossFrameGatingNetwork + TemporalMemoryTokens
4. Event detection → Detect temporal boundaries
5. Cache embeddings → INT8 quantization (4× compression)
```

### Query Processing (Multiple times)
```
1. Encode query → SigLIP text encoder → 768D
2. Retrieve frames → Cosine similarity + Magnet suppression → Top 10
3. Generate descriptions → InternVL2.5-1B (lazy) → Captions for top 10
4. Group into events → NEW: Group similar consecutive frames
5. Build context → Event-based format with timing
6. LLM reasoning → Qwen2.5-1.5B → Answer
```

## What Changed Recently (Event Grouping)

### Before (Frame-Level Context)
```
STEP 1 [0:26]: Person tightening screw
STEP 2 [0:30]: Person tightening screw  
STEP 3 [0:35]: Person tightening screw
STEP 4 [1:38]: Person pulls string
```
**Problem**: Redundant, hard to see action boundaries

### After (Event-Level Context) - NEW
```
EVENT 1 [0:26-0:35]: Person TIGHTENS screw (9s, 4 frames)
         ↓
EVENT 2 [1:38-1:42]: Person PULLS string (4s, 2 frames)
         ↓
EVENT 3 [1:43]: Light turns ON

ACTION SEQUENCE: TIGHTEN → PULL → LIGHT ON
```
**Benefit**: Clearer boundaries, reduced redundancy, better temporal structure

## Performance Metrics

### TemporalBench COIN (30 questions)
- **Accuracy**: 53.33% (16/30)
- **Gemini 1.5 Pro**: Low-to-mid 40s
- **Query Time**: 12.6s average
- **VRAM**: <4GB (RTX 3050 compatible)

### Processing Speed
- **Video Processing**: ~0s (cached)
- **Query Processing**: 12.6s average
  - First query: ~30-40s (loads InternVL + Qwen)
  - Subsequent: ~1.5s (models cached)

## Why This Configuration?

### Design Decisions

1. **SigLIP over CLIP**: Better text-image alignment, higher dimensional embeddings
2. **Lazy descriptions**: 2.6x faster than full captioning, same accuracy
3. **Event grouping**: Reduces redundancy, emphasizes action boundaries
4. **4-bit Qwen**: Fits in 4GB VRAM with InternVL
5. **CrossFrameGating + MemoryTokens**: Simpler than TAS+GRU, works well enough

### What's NOT Needed (Yet)

1. **TAS**: CrossFrameGating is sufficient for current accuracy
2. **GRU**: TemporalMemoryTokens serve similar purpose
3. **Multi-scale**: Single-scale temporal reasoning is working
4. **Entity tracking**: Not needed for TemporalBench COIN
5. **Temporal graph**: Event detection runs but graph not needed yet

## When to Enable Advanced Features

### Enable TAS + Multi-Scale when:
- Need to distinguish fast vs slow actions
- Working with sports/gymnastics videos (FineGym)
- Need motion-specific features

### Enable GRU when:
- Need stronger long-range dependencies
- Working with 2+ hour videos
- Need sequential state tracking

### Enable Temporal Event Graph when:
- Need causal reasoning ("Why did X happen?")
- Need to answer "What caused Y?"
- Working with complex procedural videos

### Enable Entity Tracking when:
- Need to track specific people/objects
- Questions about "Who did X?"
- Need to distinguish between multiple actors

## Summary

**What's running**: SigLIP + CrossFrameGating + MemoryTokens + InternVL + Qwen + Event Grouping

**What's NOT running**: TAS, GRU, Multi-scale, Entity tracking, Temporal graph

**Why**: Current configuration achieves 53.33% on TemporalBench COIN (better than Gemini) with just 4GB VRAM. Advanced features are implemented but not needed yet.

**Next steps to improve accuracy**:
1. ✅ Event grouping (DONE - restored 53.33%)
2. Upgrade to InternVL2.5-4B (better perception)
3. Enable TAS for motion-specific features
4. Use temporal graph for causal reasoning
5. Better prompting for 5 critical attributes (COUNT, DIRECTION, STATE, HAND, ORDER)
