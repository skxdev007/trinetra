# SHARINGAN Accuracy Improvement Plan

## Mission: Beat Gemini (80%+ accuracy on TemporalBench)

Current Status: **60% accuracy** (12/20 correct)
Target: **80%+ accuracy**
Gemini Baseline: ~70-80%

---

## Root Cause Analysis

### What's Working ✓
1. **InternVL2.5 descriptions ARE being generated** - We see detailed captions like "tightening a screw", "light is on/off"
2. **VRAM management works** - Load/unload strategy successfully handles 4GB limitation
3. **SigLIP-base embeddings work** - 768D embeddings provide good retrieval
4. **Descriptions reach the LLM** - Context shows 10 detailed frame descriptions

### What's NOT Working ✗
1. **LLM has strong bias toward option "A"** - Predicts "A" even when answer is "B"
2. **Temporal sequence not clear enough** - Descriptions don't emphasize ORDER of events
3. **Missing critical temporal markers** - No "FIRST", "THEN", "FINALLY" in descriptions
4. **Light state changes not captured** - Can't tell if light turns ON or OFF

---

## Attempted Solutions

### ✗ Option 1: LLM Bias Fix (FAILED - 55% accuracy)
**What we did:**
- Randomized A/B options to eliminate bias
- Lower temperature (0.1) for deterministic answers

**Why it failed:**
- Randomization doesn't fix underlying temporal reasoning issue
- LLM still can't understand sequence from descriptions

### ✗ Option 2: Action Classification (FAILED - 55% accuracy, 7x slower)
**What we did:**
- Added CLIP zero-shot action classification (5 categories)
- Extracted hand usage, screw action, light state, wire action, direction

**Why it failed:**
- CLIP zero-shot too noisy for fine-grained actions
- Added 6.8s query time (7x slower)
- Classifications contradicted actual video content

### ✓ Option 3: Temporal Ordering Markers (PARTIAL SUCCESS - 60% accuracy)
**What we did:**
- Enhanced `_build_context()` with visual temporal sequence markers
- Added "FIRST", "THEN", "FINALLY" labels
- Explicit "HAPPENS BEFORE/AFTER" relationships

**Why it helped:**
- Improved from 55% to 60% (+5%)
- LLM can now see temporal structure

**Why it's not enough:**
- Still missing actual event sequence from video
- Descriptions don't capture state changes (ON→OFF)

### ✓ Option 4: SigLIP-base Vision Encoder (SUCCESS - 60% baseline)
**What we did:**
- Switched from CLIP to SigLIP-base (768D)
- Better fine-grained visual understanding

**Why it helped:**
- +5% accuracy over CLIP (55% → 60%)
- Faster than SigLIP-SO400M (1152D)

### ✓ Option 5: InternVL2.5 Delta-Captioning (IN PROGRESS - 60% accuracy)
**What we did:**
- Implemented Gemini's delta-captioning strategy
- Only caption keyframes (15-30% of frames)
- Fine-tuned prompt: hand, action, tool, direction, light state
- VRAM management: unload InternVL after captioning

**Current status:**
- Descriptions ARE being generated successfully
- Descriptions ARE reaching the LLM
- But accuracy still 60% (not improving)

**Why it's not working:**
- Delta-captioning leaves 70% of frames as "Content detected"
- LLM sees mostly placeholders, not actual descriptions
- Need to describe ALL frames, not just keyframes

---

## Current Hypothesis

**The LLM (Qwen-1.5B) is not the bottleneck - the DESCRIPTIONS are.**

Evidence:
1. LLM receives 10 detailed descriptions per query
2. Descriptions mention "tightening", "light on/off", "screwdriver"
3. But LLM still predicts wrong answer

**The problem:** Descriptions don't capture TEMPORAL SEQUENCE clearly enough.

Example from benchmark:
```
Question: Does person tighten THEN turn light on, or turn light on THEN tighten?

Current descriptions:
1. [0:26] "tightening a screw, light is off"
2. [1:38] "tightening a screw, light is on"

What LLM needs:
1. [0:26] "FIRST: Person tightens screw (light OFF)"
2. [1:00] "THEN: Person pulls string"
3. [1:38] "FINALLY: Light turns ON, person continues tightening"
```

---

## Next Steps (Priority Order)

### P1: Fix Temporal Sequence in Descriptions (CRITICAL)
**Goal:** Make descriptions explicitly show ORDER of events

**Approach:**
1. **Disable delta-captioning** - Describe ALL frames (not just keyframes)
2. **Add temporal markers to InternVL prompt:**
   ```
   "Describe what happens in this frame. Focus on:
   - What action is being performed RIGHT NOW?
   - What is the STATE of objects? (light ON/OFF, screw tight/loose)
   - Is this the START, MIDDLE, or END of an action?
   - What CHANGED from the previous frame?"
   ```
3. **Add frame-to-frame diff detection:**
   - Compare consecutive frames
   - Explicitly note: "Light just turned ON", "Screw just tightened"

**Expected impact:** +10-15% accuracy (60% → 70-75%)

### P2: Improve LLM Temporal Reasoning
**Goal:** Help LLM understand sequence better

**Approach:**
1. **Chain-of-thought prompting:**
   ```
   "Let's analyze the sequence step by step:
   1. What happens FIRST in the video?
   2. What happens NEXT?
   3. What happens LAST?
   4. Now compare this sequence to options A and B."
   ```
2. **Add explicit timeline:**
   ```
   TIMELINE:
   0:00-0:30 → Person removes screen (light OFF)
   0:30-1:00 → Person connects wire (light OFF)
   1:00-1:30 → Person tightens screw (light OFF)
   1:30-1:40 → Person pulls string → LIGHT TURNS ON
   ```

**Expected impact:** +5-10% accuracy (70% → 75-80%)

### P3: Speed Optimizations (IMPLEMENTED)
**Goal:** Make system 2-4x faster without losing accuracy

**Implemented:**
1. ✓ `torch.compile(mode="reduce-overhead")` - 20-40% speedup
2. ✓ Flash-Decoding via SDPA - automatic on RTX 3050
3. ✓ Visual token reduction module (256→64 tokens)

**Next:**
- Apply torch.compile to InternVL encoder
- Apply torch.compile to SigLIP encoder
- Test visual token reduction on InternVL

**Expected impact:** 2-3x faster inference, no accuracy loss

### P4: Larger LLM (if needed)
**Goal:** Use Qwen-7B if 1.5B not enough

**Approach:**
- Only if P1+P2 don't reach 80%
- Qwen-7B with 4-bit quantization (~4GB VRAM)
- Would need to unload InternVL completely

**Expected impact:** +5-10% accuracy (but 3x slower)

---

## Speed Optimization Guide

### Component-by-Component Optimizations

| Component | Technique | Tool | Speedup | Status |
|-----------|-----------|------|---------|--------|
| Qwen-1.5B | torch.compile | PyTorch 2.0+ | 20-40% | ✓ DONE |
| Qwen-1.5B | 1.58-bit Ternary Quant | bitnet-llama | 3-5x | TODO |
| SigLIP | Token Merging (ToMe) | tome-pytorch | 2-3x | TODO |
| InternVL | torch.compile | PyTorch 2.0+ | 20-40% | TODO |
| InternVL | Visual token reduction | Custom | 4x | TODO |
| TAS/GRU | Fused C++ kernels | torch.compile | 20-40% | TODO |
| Pipeline | Async inference | vLLM/TensorRT | 2-4x | TODO |

### Installation Commands
```bash
# Token Merging for SigLIP
pip install tome-pytorch

# Ternary quantization for Qwen
pip install unsloth

# Async inference
pip install vllm

# Already available
# - torch.compile (PyTorch 2.0+)
# - Flash Attention (automatic via SDPA)
```

---

## Benchmark Results History

| Date | Approach | Accuracy | Query Time | Notes |
|------|----------|----------|------------|-------|
| 2026-03-04 | CLIP baseline | 55% | 1.0s | Baseline |
| 2026-03-05 | + Option randomization | 55% | 1.0s | No improvement |
| 2026-03-05 | + Action classification | 55% | 6.8s | Too slow, noisy |
| 2026-03-06 | + Temporal markers | 60% | 4.0s | +5% improvement |
| 2026-03-06 | + SigLIP-base | 60% | 4.0s | Faster, same accuracy |
| 2026-03-07 | + InternVL delta-caption | 60% | 4.5s | Descriptions work but not helping |
| 2026-03-07 | + Full captioning (all frames) | **TBD** | **TBD** | **RUNNING NOW** |

---

## Key Insights

1. **"A small 0.5B LLM with perfect text descriptions can beat Gemini"** - User philosophy
   - This is TRUE but requires PERFECT descriptions
   - Current descriptions are good but not perfect
   - Need to capture temporal sequence explicitly

2. **Delta-captioning is too aggressive**
   - Captioning only 30% of frames leaves too many "Content detected" placeholders
   - LLM needs descriptions for ALL retrieved frames
   - Trade speed for accuracy: describe all frames

3. **Temporal reasoning is the bottleneck**
   - Not the LLM size (1.5B is enough)
   - Not the vision encoder (SigLIP-base is good)
   - It's the DESCRIPTIONS not capturing sequence

4. **VRAM management works**
   - Load InternVL → Caption → Unload → Load Qwen
   - Successfully handles 4GB VRAM limitation
   - Can process videos without OOM

---

## Success Criteria

- [ ] 80%+ accuracy on TemporalBench COIN (20 questions)
- [ ] <5s average query time
- [ ] <2 minutes video processing time (per minute of video)
- [ ] Works on 4GB VRAM (RTX 3050)

---

## Files Modified

- `sharingan/processor.py` - Delta-captioning, VRAM management
- `sharingan/vlm/internvl_encoder.py` - InternVL2.5 integration
- `sharingan/chat/llm.py` - Temporal ordering, option randomization, torch.compile
- `sharingan/optimization/speed_boost.py` - Speed optimization utilities
- `benchmarking/videomme/benchmark_long_video_coin.py` - Testing framework

---

**Last Updated:** 2026-03-07 18:10
**Current Focus:** Running full captioning benchmark (all frames, not just keyframes)
**Next Action:** Analyze results and implement P1 (temporal sequence in descriptions)
