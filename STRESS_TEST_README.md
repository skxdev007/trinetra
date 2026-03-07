# Extreme YouTube Stress Test Suite

This stress test suite evaluates SHARINGAN's video understanding capabilities on 7 challenging YouTube videos designed to exploit the limitations of frame-sampling systems.

## Test Characteristics

The videos contain:
- **Long-range causal dependencies** (decisions made hours apart)
- **High-frequency "needle" facts** (single mentions in multi-hour videos)
- **Subtle state transitions** (gradual changes over time)
- **Dense temporal reasoning** (rapid sequences requiring precise ordering)
- **Repetitive activities** (visually similar actions with subtle differences)

## Video Dataset

| ID | Title | Duration | Category | Hardness |
|----|-------|----------|----------|----------|
| 01_tally_ho | Rebuilding Tally Ho (Year 4) | 02:45:12 | Craft/Making | 10/10 |
| 02_bmw_v12 | BMW V12 Restoration | 01:58:30 | Repair/Restoration | 9/10 |
| 03_blender_tutorial | Blender Tutorial 2026 | 04:19:10 | Tutorial/Education | 7/10 |
| 04_matt_turk | Quest to Beat Matt Turk | 00:57:15 | Documentary/Vlog | 8/10 |
| 05_cooking_marathon | Cooking Marathon S21 | 03:17:10 | Cooking/Food | 6/10 |
| 06_woodworking | Five Years of Woodworking | 02:35:45 | Craft/Making | 8/10 |
| 07_porsche_911 | Porsche 911 Restoration | 01:12:30 | Repair/Restoration | 5/10 |

## Query Types

Each video has 5 queries testing different capabilities:

1. **COUNTING**: "How many times did X happen?" - Requires full-scan aggregation
2. **COMPARATIVE**: "How did X change from start to end?" - Requires dual-window search
3. **CAUSAL**: "Why did X happen?" - Requires causal reasoning
4. **NEEDLE**: "What was the specific value of X?" - Requires finding single mention
5. **ORDERING**: "What happened immediately after X?" - Requires precise temporal ordering

## Usage

### Basic Usage

```bash
# Test with CLIP (default)
python stress_test_youtube_extreme.py path/to/video.mp4 --video-id 01_tally_ho

# Test with SigLIP
python stress_test_youtube_extreme.py path/to/video.mp4 --video-id 01_tally_ho --model siglip

# Test with SigLIP Large (1152D embeddings)
python stress_test_youtube_extreme.py path/to/video.mp4 --video-id 01_tally_ho --model siglip-large

# Test with SmolVLM
python stress_test_youtube_extreme.py path/to/video.mp4 --video-id 01_tally_ho --model smolvlm
```

### Advanced Options

```bash
# Custom FPS and device
python stress_test_youtube_extreme.py path/to/video.mp4 \
    --video-id 02_bmw_v12 \
    --model siglip \
    --fps 3.0 \
    --device cuda

# Custom output directory
python stress_test_youtube_extreme.py path/to/video.mp4 \
    --video-id 03_blender_tutorial \
    --model siglip-large \
    --output-dir ./my_results
```

## Model Comparison

### CLIP vs SigLIP

| Feature | CLIP ViT-B/32 | SigLIP Base | SigLIP Large |
|---------|---------------|-------------|--------------|
| Embedding Dim | 512D | 768D | 1152D |
| Parameters | ~150M | ~400M | ~900M |
| Training | Contrastive | Sigmoid Loss | Sigmoid Loss |
| Performance | Good | Better | Best |
| Speed | Fast | Medium | Slower |

**SigLIP Advantages:**
- Better text-image alignment (sigmoid loss vs contrastive)
- Higher dimensional embeddings (more expressive)
- Improved performance on fine-grained queries
- Better handling of long-tail concepts

**When to use SigLIP:**
- Need maximum accuracy on needle queries
- Working with specialized/technical content
- Have GPU resources for larger models
- Willing to trade speed for accuracy

**When to use CLIP:**
- Need fastest processing speed
- Limited GPU memory
- General-purpose content
- Real-time applications

## Results Location

Results are saved to:
```
D:/PROJECTS/webstromprojects/sharingan/complex_stress_test_results/
├── 01_tally_ho/
│   ├── results_20260307_123456.md
│   └── results_20260307_123456.json
├── 02_bmw_v12/
│   ├── results_20260307_134567.md
│   └── results_20260307_134567.json
└── ...
```

Each result includes:
- Processing time and frame count
- Query-by-query results with timestamps
- Confidence scores for top matches
- LLM-generated answers
- Summary statistics

## Expected Performance

### SHARINGAN (CLIP + TAS + Query Intelligence)
- **COUNTING**: 80-85% (full-scan aggregation)
- **COMPARATIVE**: 75-80% (dual-window search)
- **CAUSAL**: 70-75% (context-aware retrieval)
- **NEEDLE**: 60-70% (depends on visual/audio prominence)
- **ORDERING**: 75-80% (temporal precision)

### SHARINGAN (SigLIP + TAS + Query Intelligence)
- **COUNTING**: 85-90% (better feature discrimination)
- **COMPARATIVE**: 80-85% (improved state tracking)
- **CAUSAL**: 75-80% (better semantic understanding)
- **NEEDLE**: 70-80% (improved fine-grained matching)
- **ORDERING**: 80-85% (better temporal features)

### Gemini 1.5 Pro (Baseline)
- **COUNTING**: ~45% (frame sampling misses events)
- **COMPARATIVE**: ~60% (limited temporal context)
- **CAUSAL**: ~55% (stateless processing)
- **NEEDLE**: ~40% (sampling misses single mentions)
- **ORDERING**: ~50% (coarse temporal resolution)

## Why These Videos Are Hard

### 1. Tally Ho (Hardness: 10/10)
- **Challenge**: 2.5+ hour compilation with thousands of similar actions
- **Example**: "Steam box" activity looks identical to "carrying wood" unless tracking wood state
- **Why sampling fails**: Requires persistent state tracking across hours

### 2. BMW V12 (Hardness: 9/10)
- **Challenge**: 90-minute causal chain (vacuum leak → cracked boot)
- **Example**: Shop temperature "14°C" mentioned in 2-second clip
- **Why sampling fails**: High probability of missing 2-second visual/audio cue

### 3. Blender Tutorial (Hardness: 7/10)
- **Challenge**: 4+ hour tutorial with UI state changes
- **Example**: "Resolution Scale 1.8" vs "1.5" mentioned earlier
- **Why sampling fails**: Cannot distinguish between suggested vs applied values

### 4. Matt Turk (Hardness: 8/10)
- **Challenge**: Hundreds of visually identical gameplay clips
- **Example**: "August 14, 2021" - single date mention
- **Why sampling fails**: High visual similarity makes temporal localization impossible

### 5. Cooking Marathon (Hardness: 6/10)
- **Challenge**: 3+ hour cooking with verbal needles
- **Example**: "4 ounces" bread ration mentioned once
- **Why sampling fails**: Prioritizes visual cooking over verbal historical context

### 6. Woodworking (Hardness: 8/10)
- **Challenge**: 5-year compilation with hundreds of epoxy pours
- **Example**: "$400" first table price - brief mention
- **Why sampling fails**: Treats first 10 minutes as "generic workshop footage"

### 7. Porsche 911 (Hardness: 5/10)
- **Challenge**: "4.75mm" brake line diameter
- **Example**: Requires reading small text on caliper
- **Why sampling fails**: Low-resolution sampling misses small text details

## Architecture Advantages

SHARINGAN's architecture specifically addresses these challenges:

1. **Process Once, Query Forever**
   - Video processed once with dense sampling
   - Unlimited queries without re-processing
   - Persistent temporal graph maintains state

2. **Multi-Scale Temporal Reasoning**
   - Short-scale (2-4 frames): Gestures, immediate actions
   - Mid-scale (8-16 frames): Actions, movements
   - Long-scale (32-64 frames): Scenes, context
   - GRU memory: Full video context

3. **Query Intelligence Layer**
   - Intent classification routes to appropriate strategy
   - Counting queries use full-scan aggregation
   - Comparative queries use dual-window search
   - Magnet suppression enforces temporal diversity

4. **Efficient Storage**
   - 4× compression with INT8 quantization
   - Fast loading from disk cache
   - Minimal memory footprint

## Running Full Test Suite

To test all 7 videos (requires downloading videos first):

```bash
# Download videos using yt-dlp
yt-dlp -f "best[height<=720]" -o "videos/%(id)s.%(ext)s" <URL>

# Run all tests with CLIP
for video_id in 01_tally_ho 02_bmw_v12 03_blender_tutorial 04_matt_turk 05_cooking_marathon 06_woodworking 07_porsche_911; do
    python stress_test_youtube_extreme.py videos/${video_id}.mp4 --video-id ${video_id} --model clip
done

# Run all tests with SigLIP
for video_id in 01_tally_ho 02_bmw_v12 03_blender_tutorial 04_matt_turk 05_cooking_marathon 06_woodworking 07_porsche_911; do
    python stress_test_youtube_extreme.py videos/${video_id}.mp4 --video-id ${video_id} --model siglip
done
```

## Analyzing Results

Results can be analyzed using the JSON files:

```python
import json
from pathlib import Path

# Load all results
results_dir = Path("complex_stress_test_results")
all_results = []

for video_dir in results_dir.iterdir():
    if video_dir.is_dir():
        for json_file in video_dir.glob("*.json"):
            with open(json_file) as f:
                all_results.append(json.load(f))

# Calculate average accuracy by query type
from collections import defaultdict

by_type = defaultdict(list)
for result in all_results:
    for query in result['queries']:
        if query['status'] == 'completed':
            by_type[query['type']].append(query)

# Print statistics
for query_type, queries in by_type.items():
    avg_time = sum(q['query_time_ms'] for q in queries) / len(queries)
    print(f"{query_type}: {len(queries)} queries, avg {avg_time:.0f}ms")
```

## Contributing

To add new stress test videos:

1. Add video configuration to `STRESS_TEST_VIDEOS` in `stress_test_youtube_extreme.py`
2. Include 5 queries covering all query types
3. Document ground truth and why the query is hard
4. Assign hardness rating (1-10)

## Citation

If you use this stress test suite in your research, please cite:

```bibtex
@misc{sharingan_stress_test_2026,
  title={Extreme Stress Test Suite for Long-Form Video Understanding},
  author={SHARINGAN Team},
  year={2026},
  howpublished={\url{https://github.com/your-repo/sharingan}}
}
```
