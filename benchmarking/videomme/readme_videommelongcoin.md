# TemporalBench Long Video COIN Benchmark

This directory contains the TemporalBench Long Video COIN benchmark for testing fine-grained temporal understanding.

## Overview

The TemporalBench Long Video COIN dataset tests SHARINGAN's ability to understand subtle temporal differences in videos through multiple-choice questions. Questions test:

- **Counting**: "5 times" vs "10 times"
- **Direction**: "left to right" vs "right to left"  
- **Order**: "A then B" vs "B then A"
- **State changes**: "clean water" vs "dirty water"

## Dataset Structure

```
benchmarking/videomme/long_video_coin/
├── temporalbench_long_qa.json          # 5,485 QA pairs
├── dataset/                             # Video files (gitignored)
│   └── long_video/
│       └── COIN/
│           └── *.mp4                    # 492 video files
└── results/                             # Benchmark results (generated)
```

## Setup

1. Download the Long Video COIN dataset videos and place them in:
   ```
   benchmarking/videomme/long_video_coin/dataset/long_video/COIN/
   ```

2. The dataset directory is already added to `.gitignore` to avoid committing large video files.

## Running the Benchmark

### Full Benchmark (5,485 questions)

```bash
python benchmarking/videomme/benchmark_long_video_coin.py
```

### Test Run (first 10 questions)

```bash
python benchmarking/videomme/benchmark_long_video_coin.py --max-questions 10
```

### Custom Configuration

```bash
python benchmarking/videomme/benchmark_long_video_coin.py \
  --qa-file benchmarking/videomme/long_video_coin/temporalbench_long_qa.json \
  --dataset-dir benchmarking/videomme/long_video_coin/dataset \
  --output-dir benchmarking/videomme/long_video_coin/results \
  --target-fps 5.0 \
  --max-questions 100
```

## Output

The benchmark generates two files in the results directory:

1. **results_YYYYMMDD_HHMMSS.json** - Detailed results with all predictions
2. **summary_YYYYMMDD_HHMMSS.md** - Human-readable summary with metrics

### Example Output

```
BENCHMARK SUMMARY
================================================================================
Total Questions: 5485
Answered: 5485
Correct: 4521
Accuracy: 82.43%
Unique Videos Processed: 492
Total Time: 3245.2s (54.1 min)
Average Query Time: 0.592s
================================================================================
```

## Performance Metrics

The benchmark tracks:
- **Accuracy**: Percentage of correct answers
- **Processing Time**: Time to process each unique video
- **Query Time**: Time to answer each question (after video is processed)
- **Videos Processed**: Number of unique videos (492 total)

## Question Format

Each question follows this format:

```
Which caption best describes this video?
A. [Caption with subtle difference]
B. [Caption with subtle difference]
Answer with the option's letter from the given choices directly.
```

The model must choose between two nearly identical captions that differ in one key temporal detail.

## Implementation Details

- Videos are processed once and cached in memory
- Multiple questions can reference the same video
- Answer extraction handles various response formats (A, B, "Option A", etc.)
- Progress is printed for each question
- Errors are logged but don't stop the benchmark
