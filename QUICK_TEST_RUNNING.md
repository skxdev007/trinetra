# VideoMAE + Qwen2.5-1.5B Quick Test - IN PROGRESS

## Status: RUNNING ⏳

**Started:** 2026-03-05
**Test:** 20 videos from TemporalBench COIN dataset
**Configuration:**
- Vision Model: VideoMAE V2-Large (300M params, ~1.37GB)
- Language Model: Qwen2.5-1.5B-Instruct (4-bit, ~900MB)
- Device: NVIDIA GeForce RTX 3050 Laptop GPU
- Total Memory: ~2.3GB

## Current Progress

### Phase 1: Model Download (First Time Only)
- ✓ VideoMAE model downloading: 1.37GB
- ⏳ This is a one-time download, subsequent runs will use cached model
- Expected download time: 5-10 minutes (depending on internet speed)

### Phase 2: Video Processing
- Will process 20 videos
- Each video: ~30-60 seconds processing time
- Total estimated time: 15-25 minutes

### Phase 3: Results
- Will save predictions in TemporalBench format
- Will calculate accuracy on 20 videos
- Expected accuracy: 60-70% (vs 48.5% baseline with CLIP)

## What's Happening

1. **Model Loading:**
   - VideoMAE V2-Large downloading from Hugging Face
   - Qwen2.5-1.5B will download next (if not cached)
   - Models cached in `~/.cache/huggingface/`

2. **Processing Pipeline:**
   - Extract frames at 5 FPS
   - Encode with VideoMAE (better temporal understanding than CLIP)
   - Store embeddings in cache
   - Answer questions using similarity search
   - Extract answer with fixed parser

3. **Expected Improvements:**
   - Better motion understanding (clockwise vs counterclockwise)
   - Better hand tracking (left vs right)
   - Better action sequencing (A then B vs B then A)
   - Better temporal reasoning with Qwen2.5-1.5B

## Comparison to Baseline

| Metric | CLIP + Qwen-0.5B | VideoMAE + Qwen-1.5B | Improvement |
|--------|------------------|----------------------|-------------|
| Accuracy (broken parser) | 48.5% | TBD | TBD |
| Accuracy (fixed parser) | ~55-60% | TBD | TBD |
| Memory | 1.3 GB | 2.3 GB | +1 GB |
| Model Size | 850 MB | 2.3 GB | +1.45 GB |

## Next Steps

1. **Wait for test to complete** (~20-30 minutes total)
2. **Check results** in `results_videomae_quick_*.json`
3. **Compare accuracy** to CLIP baseline
4. **If successful** (>60% accuracy), run full benchmark on all videos
5. **Submit to TemporalBench** leaderboard

## Files Being Created

- `predictions_videomae_quick_*.jsonl` - TemporalBench format predictions
- `results_videomae_quick_*.json` - Detailed results with metadata

## Monitoring Progress

Check terminal output for:
- Video processing progress (1/20, 2/20, etc.)
- Per-question accuracy (✓ or ✗)
- Running accuracy percentage
- Processing time per video

---

*Test started: 2026-03-05*
*Status: Model downloading, then will process 20 videos*
*Expected completion: 20-30 minutes*
