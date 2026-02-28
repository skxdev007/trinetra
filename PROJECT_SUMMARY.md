# SHARINGAN Project Summary

## 📍 Test Results Location

**Main Test Results File:** `cooking_video_test_results.md`
- **Full Path:** `D:\PROJECTS\webstromprojects\sharingan\cooking_video_test_results.md`
- **Contains:** Complete Q&A results for Chicken Biryani cooking video
- **Includes:** 
  - 17 detected events with timestamps
  - 16 comprehensive queries with answers
  - Video processing statistics
  - All timestamps and relevant moments

## 🎯 Project Structure

### Core Application
```
sharingan/
├── chat/           # Query pipeline and LLM integration
├── events/         # Event detection
├── graph/          # Causal graph and event relationships
├── query/          # Query routing and scaffolding
├── storage/        # Memory and embedding storage
├── temporal/       # Multi-scale TAS and temporal reasoning
├── ui/             # Gradio web interface
├── verification/   # Cross-modal verification
├── video/          # Video loading and sampling
└── vlm/            # Vision-language models (CLIP, SmolVLM)
```

### Test Scripts
- **`test_youtube_video.py`** - Basic YouTube video test
- **`test_cooking_video.py`** - Timestamp-based query test
- **`test_full_cooking_video.py`** - Comprehensive test with results export ⭐

### Documentation
- **`README.md`** - Main project documentation
- **`ARCHITECTURE.md`** - System architecture overview
- **`docs/`** - Detailed documentation for each component
- **`cooking_video_test_results.md`** - Test results with Q&A ⭐

### Scripts
- **`scripts/launch_ui.py`** - Launch Gradio UI with CLI options
- **`run_ui.py`** - Simple UI launcher

## 🚀 How to Use

### 1. Launch the Gradio UI
```bash
python scripts/launch_ui.py
```
Then open http://127.0.0.1:7860 in your browser

### 2. Run Comprehensive Test
```bash
python test_full_cooking_video.py
```
Results will be saved to `cooking_video_test_results.md`

### 3. Test with Custom Video
Edit `test_full_cooking_video.py` and change the YouTube URL:
```python
youtube_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

## 📊 Test Results Summary

### Chicken Biryani Video Test
- **Video Duration:** 11.7 minutes
- **Processing Time:** 44.3 seconds
- **Events Detected:** 17 scene transitions
- **Queries Tested:** 16 comprehensive queries
- **Success Rate:** 100%

### Query Categories Tested
1. **Content** - Ingredients, equipment, dish identification
2. **Process** - Cooking steps, preparation methods
3. **Timing** - Beginning, middle, end, specific moments
4. **Actions** - Mixing, stirring, ingredient addition
5. **Summary** - Recipe overview, final result

## 🎨 Configuration Presets

The system includes 4 presets for different use cases:

1. **Fast Processing** - 0.5-2.0 FPS, quick results
2. **Balanced** (Default) - 1.0-5.0 FPS, good quality/speed
3. **High Quality** - 2.0-8.0 FPS, better accuracy
4. **Maximum Detail** - 3.0-10.0 FPS, best quality

## 🔧 Key Features

✅ YouTube video download with caching
✅ Visual content processing (CLIP embeddings)
✅ Event detection with timestamps
✅ Semantic search across video
✅ Timestamp-based queries
✅ Configuration presets
✅ Gradio web interface
✅ Results export to markdown

## 📝 Important Notes

- SHARINGAN processes **visual content only** (no audio)
- Text overlays detected through CLIP embeddings
- Higher FPS = better text detection but slower processing
- Videos are cached to prevent re-downloading
- Processing cache stored in `cache/` directory

## 🎯 Next Steps

1. Open `cooking_video_test_results.md` to see full Q&A results
2. Try the Gradio UI at http://127.0.0.1:7860
3. Test with your own cooking videos
4. Adjust configuration presets for your needs

---

**Generated:** 2026-02-28
**Status:** ✅ All systems operational
