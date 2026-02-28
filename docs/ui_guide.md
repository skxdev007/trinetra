# SHARINGAN UI User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [UI Components](#ui-components)
4. [Processing Videos](#processing-videos)
5. [Asking Questions](#asking-questions)
6. [Advanced Features](#advanced-features)
7. [Configuration Options](#configuration-options)
8. [Tips for Best Results](#tips-for-best-results)
9. [Troubleshooting](#troubleshooting)
10. [Keyboard Shortcuts](#keyboard-shortcuts)

---

## Introduction

Welcome to SHARINGAN's Gradio-based web interface! This guide will help you understand and use all features of the UI to get the most out of SHARINGAN's deep video understanding capabilities.

### What is SHARINGAN?

SHARINGAN is a deep video understanding system that:
- **Processes video once** and stores understanding in memory
- **Answers unlimited questions** without re-processing the video
- **Runs completely locally** with zero API costs
- **Understands temporal reasoning** (why, when, how questions)
- **Provides fast responses** (<500ms query time)

### Key Advantages

✅ **Privacy**: Your videos never leave your machine  
✅ **Cost**: Zero API costs, all models run locally  
✅ **Speed**: Fast queries after one-time processing  
✅ **Intelligence**: Multi-scale temporal reasoning  
✅ **Transparency**: See reasoning paths and confidence scores

---

## Getting Started

### Launching the UI

There are two ways to launch the SHARINGAN UI:

#### Method 1: Direct Launch (Simple)
```bash
python -m sharingan.ui.gradio_app
```

This opens the UI at `http://localhost:7860`

#### Method 2: Using Launch Script (Recommended)
```bash
python scripts/launch_ui.py
```

For more launch options, see the [Launch Guide](./gradio_ui_launch.md).

### First Time Setup

When you first launch SHARINGAN, it will automatically download required models:

1. **SmolVLM-500M** (~538 MB) - Vision-language model
2. **CLIP ViT-B/32** (~400 MB) - Cross-modal verification
3. **Qwen2.5-0.5B** (~538 MB) - Language model for responses

**Total download**: ~1.5 GB (one-time only)

**Note**: Models are cached locally and won't be re-downloaded.

### System Requirements

**Minimum** (CPU-only, slower):
- 8 GB RAM
- 4 CPU cores
- 5 GB disk space

**Recommended** (GPU-accelerated):
- 16 GB RAM
- NVIDIA GPU with 8 GB VRAM
- 20 GB disk space

**Optimal** (best performance):
- 32 GB RAM
- NVIDIA GPU with 16 GB VRAM
- 100 GB SSD

---

## UI Components

The SHARINGAN UI is divided into several main sections:

### 1. Video Upload Section (Left Column)

**Components:**
- **Video Upload**: Drag and drop or click to upload video files
- **Vision Model Selector**: Choose between CLIP (faster) or SmolVLM (more detailed)
- **Device Selector**: Auto-detect GPU, or force CPU/CUDA
- **Advanced Configuration**: Expandable panel with detailed settings
- **Process Button**: Start video processing
- **Status Display**: Shows processing progress and results

**Supported Formats:**
- MP4 (recommended)
- AVI
- MOV
- WebM
- MKV

**File Size Limit**: 10 GB (configurable)

### 2. Query Interface Section (Right Column)

**Components:**
- **Chatbot Display**: Shows conversation history with timestamps
- **Question Input**: Text box for entering questions
- **Ask Button**: Submit your question
- **Example Questions**: Pre-filled example queries
- **Clear Button**: Reset conversation history

### 3. Advanced Analysis Section (Bottom)

**Tabs:**
- **Causal Graph**: Visual representation of event relationships
- **Timeline**: Chronological view of detected events
- **Reasoning**: Shows how SHARINGAN arrived at answers
- **Confidence**: Displays confidence scores for answers

---

## Processing Videos

### Step-by-Step Guide

#### Step 1: Upload Your Video

1. Click the **"Upload Video"** area or drag and drop a video file
2. Wait for the video preview to appear
3. Verify the video loaded correctly

**Tips:**
- Use MP4 format for best compatibility
- Keep videos under 10 GB for optimal performance
- Shorter videos (1-10 minutes) process faster

#### Step 2: Choose Processing Settings

**Quick Settings:**
- **Vision Model**: 
  - `clip` - Faster, uses visual embeddings only
  - `smolvlm` - Slower, generates detailed descriptions
- **Device**:
  - `auto` - Automatically detects GPU (recommended)
  - `cuda` - Force GPU usage
  - `cpu` - Force CPU usage (slower)

**Advanced Settings** (optional):
- Expand the "⚙️ Advanced Configuration" panel
- Adjust settings based on your video type
- See [Configuration Options](#configuration-options) for details

#### Step 3: Process the Video

1. Click the **"🚀 Process Video"** button
2. Watch the status display for progress updates
3. Wait for processing to complete

**Processing Time Estimates:**
- 1-minute video: ~5 minutes (GPU) / ~15 minutes (CPU)
- 5-minute video: ~25 minutes (GPU) / ~75 minutes (CPU)
- 10-minute video: ~50 minutes (GPU) / ~150 minutes (CPU)

**What Happens During Processing:**
1. **Frame Sampling**: Intelligently samples frames based on visual change
2. **Description Generation**: Creates descriptions for each frame
3. **Verification**: Checks descriptions against visual evidence
4. **Event Detection**: Identifies meaningful events in the video
5. **Graph Construction**: Builds causal relationships between events
6. **Memory Storage**: Stores everything for fast querying

#### Step 4: Review Processing Results

After processing completes, you'll see:

**Status Message:**
```
✅ Video processed successfully!

Processed 300 frames in 120.5s
Detected 45 events

You can now ask questions about the video!
```

**Video Information:**
- Duration
- Total frames
- Processed frames
- FPS (frames per second)
- Events detected
- Processing time
- Configuration used

**Visualizations:**
- Causal graph showing event relationships
- Timeline showing when events occurred

---

## Asking Questions

### How to Ask Questions

1. Type your question in the **"Your Question"** text box
2. Click **"Ask"** or press **Enter**
3. Wait for SHARINGAN to respond (typically <500ms)
4. View the answer in the chatbot display

### Question Types

SHARINGAN supports four types of questions:

#### 1. Window Queries (Time-based)

Ask about specific time ranges in the video.

**Examples:**
- "What happened between 0:30 and 1:00?"
- "Describe the scene from 2:15 to 2:45"
- "What was the person doing at 1:30?"

**How it works:**
- SHARINGAN detects temporal bounds in your question
- Retrieves events within that time window
- Provides answer with timestamps

#### 2. Semantic Queries (Content-based)

Ask about specific objects, people, or actions.

**Examples:**
- "Find person speaking"
- "Show me all scenes with a red car"
- "When does the dog appear?"
- "What objects are on the table?"

**How it works:**
- SHARINGAN extracts entities from your question
- Searches memory for matching events
- Returns relevant moments with timestamps

#### 3. Causal Queries (Why/How questions)

Ask about causes, reasons, and relationships.

**Examples:**
- "Why did the person pick up the knife?"
- "What caused the person to react?"
- "How did the object fall?"
- "Why did the scene change?"

**How it works:**
- SHARINGAN detects causal keywords (why, caused, because)
- Traverses the causal event graph
- Builds reasoning chain showing cause and effect

#### 4. Summary Queries (Overview)

Ask for high-level summaries of the video.

**Examples:**
- "Summarize this video"
- "What are the main events?"
- "Give me an overview"
- "What happened in this video?"

**How it works:**
- SHARINGAN queries chapter-level memory
- Aggregates key events
- Provides structured summary

### Example Questions

Click on any example question to auto-fill the input box:

1. "What happens in this video?"
2. "What happened between 0:30 and 1:00?"
3. "Find person speaking"
4. "Why did the person pick up the object?"
5. "Summarize the main events"
6. "What objects are visible?"
7. "Describe the scene at 0:45"
8. "What caused the person to react?"
9. "Show me all actions involving the red object"
10. "What is the sequence of events?"

### Understanding Responses

Each response includes:

**Answer Text:**
- Natural language answer to your question
- Includes timestamps for relevant moments
- References specific events and objects

**Query Time:**
- Shows how long it took to answer (typically <500ms)
- Demonstrates SHARINGAN's speed advantage

**Reasoning Path** (in Advanced Analysis):
- Shows the steps SHARINGAN took to answer
- Displays evidence used
- Helps you understand the reasoning process

**Confidence Score** (in Advanced Analysis):
- Overall confidence in the answer
- Component-level confidence breakdown
- Visual confidence bar

---

## Advanced Features

### Causal Graph Visualization

The causal graph shows relationships between detected events.

**How to View:**
1. Process a video
2. Expand "🔍 Advanced Analysis"
3. Click the "Causal Graph" tab

**Understanding the Graph:**

**Nodes (Circles):**
- Each node represents a detected event
- Label shows timestamp and brief description
- Blue color indicates event node

**Edges (Lines):**
- **Red solid lines**: Causal relationships (A caused B)
- **Blue dashed lines**: Semantic relationships (A and B are related)
- **Gray dotted lines**: Temporal relationships (A happened before B)

**Use Cases:**
- Understand event relationships
- Trace causal chains
- Identify key moments
- Debug reasoning issues

### Timeline Visualization

The timeline shows when events occurred in the video.

**How to View:**
1. Process a video
2. Expand "🔍 Advanced Analysis"
3. Click the "Timeline" tab

**Understanding the Timeline:**

**Event Markers:**
- Dots on the timeline represent detected events
- Position shows when the event occurred
- Color indicates confidence (green=high, yellow=medium, red=low)

**Labels:**
- Brief description of each event
- Alternates above/below for readability

**Time Axis:**
- Shows video duration
- Marked with timestamps (MM:SS or HH:MM:SS)

**Use Cases:**
- See event distribution over time
- Identify dense vs sparse periods
- Verify event detection accuracy
- Navigate to specific moments

### Reasoning Scaffold Display

The reasoning scaffold shows how SHARINGAN answered your question.

**How to View:**
1. Ask a question
2. Expand "🔍 Advanced Analysis"
3. Click the "Reasoning" tab

**Components:**

**Scaffold Type:**
- `Causal Chain`: For why/how questions
- `Temporal Order`: For sequence questions
- `State Change`: For transition questions

**Reasoning Steps:**
- Numbered list of reasoning steps
- Shows logical progression
- Explains how answer was derived

**Evidence:**
- Specific events used as evidence
- Includes timestamps and descriptions
- Links to source material

**Use Cases:**
- Understand how SHARINGAN thinks
- Verify reasoning is sound
- Debug incorrect answers
- Learn about temporal reasoning

### Confidence Indicators

Confidence scores show how certain SHARINGAN is about its answer.

**How to View:**
1. Ask a question
2. Expand "🔍 Advanced Analysis"
3. Click the "Confidence" tab

**Components:**

**Overall Confidence:**
- Visual bar showing confidence level
- Percentage score (0-100%)
- Color-coded: 🟢 High (≥80%), 🟡 Medium (50-80%), 🔴 Low (<50%)

**Component Scores:**
- Breakdown by component (retrieval, reasoning, generation)
- Shows which parts are most/least confident
- Helps identify weak points

**Interpretation:**
- **High confidence (≥80%)**: Answer is likely accurate
- **Medium confidence (50-80%)**: Answer is reasonable but verify
- **Low confidence (<50%)**: Answer may be incorrect, ask differently

**Use Cases:**
- Assess answer reliability
- Decide whether to trust the answer
- Identify when to rephrase questions
- Debug low-confidence responses

---

## Configuration Options

### When to Adjust Configuration

**Default settings work well for most videos**, but you may want to adjust for:

- Very long videos (>1 hour)
- Fast-paced videos (sports, action)
- Slow-paced videos (lectures, interviews)
- Memory-constrained systems
- Accuracy vs speed tradeoffs

### Quick Configuration Presets

#### Fast Processing (Low Memory)
**Use when:** Limited RAM/GPU, need quick results

- TAS Kernels: 2/8/32
- TAS Window: 32
- Base FPS: 0.5
- Max FPS: 3.0
- SmolVLM: 256M variant
- Qwen: 0.5B

**Tradeoff:** Faster but less accurate

#### Balanced (Default)
**Use when:** Normal videos, balanced performance

- TAS Kernels: 2/8/32
- TAS Window: 64
- Base FPS: 1.0
- Max FPS: 5.0
- SmolVLM: 500M variant
- Qwen: 0.5B

**Tradeoff:** Good balance of speed and accuracy

#### High Accuracy (High Memory)
**Use when:** Accuracy is critical, have powerful hardware

- TAS Kernels: 2/8/32
- TAS Window: 128
- Base FPS: 2.0
- Max FPS: 10.0
- CLIP: large-patch14
- Qwen: 1.5B

**Tradeoff:** More accurate but slower and memory-intensive

### Configuration Sections

For detailed explanations of each configuration option, see the [Configuration Guide](./gradio_ui_configuration.md).

**Quick Reference:**

1. **Multi-Scale TAS**: Controls temporal reasoning at different scales
2. **Adaptive Sampler**: Controls frame sampling rate
3. **Cross-Modal Verifier**: Controls hallucination detection
4. **Causal Edge Scorer**: Controls relationship detection
5. **Model Selection**: Choose model sizes and variants

---

## Tips for Best Results

### Video Preparation

✅ **DO:**
- Use MP4 format for best compatibility
- Ensure good video quality (720p or higher)
- Use videos with clear audio (if relevant)
- Keep videos under 10 minutes for faster processing
- Trim unnecessary intro/outro sections

❌ **DON'T:**
- Use corrupted or incomplete video files
- Use extremely low resolution (<480p)
- Use videos with heavy compression artifacts
- Upload videos larger than 10 GB

### Asking Better Questions

✅ **DO:**
- Be specific about what you want to know
- Include time ranges when relevant ("between 1:00 and 2:00")
- Use clear, simple language
- Ask one question at a time
- Reference specific objects or people

❌ **DON'T:**
- Ask vague questions ("What's happening?")
- Combine multiple questions in one
- Use overly complex language
- Exceed 512 characters
- Ask about things not in the video

### Question Examples

**Good Questions:**
- "What is the person holding at 0:45?"
- "Why did the person pick up the red cup?"
- "Describe the scene between 1:00 and 1:30"
- "When does the dog first appear?"
- "What caused the glass to fall?"

**Poor Questions:**
- "What?" (too vague)
- "Tell me everything about the video and also what happens at 1:00 and why the person did that thing" (too complex)
- "What is the person thinking?" (not observable)
- "What happens in the next video?" (wrong context)

### Optimizing Performance

**For Faster Processing:**
1. Use CLIP instead of SmolVLM
2. Reduce Max FPS to 3.0
3. Use smaller models (256M SmolVLM, 0.5B Qwen)
4. Reduce TAS window to 32
5. Lower base FPS to 0.5

**For Better Accuracy:**
1. Use SmolVLM instead of CLIP
2. Increase Max FPS to 10.0
3. Use larger models (500M SmolVLM, 1.5B Qwen)
4. Increase TAS window to 128
5. Increase base FPS to 2.0

**For Long Videos (>1 hour):**
1. Reduce sampling rates (base FPS: 0.5, max FPS: 3.0)
2. Use smaller models to save memory
3. Increase TAS long-scale kernel to 64
4. Consider splitting video into segments

**For Fast-Paced Videos:**
1. Increase Max FPS to 10.0
2. Lower change threshold to 0.2 (more sensitive)
3. Use shorter TAS kernels (2/4/16)
4. Increase base FPS to 2.0

**For Slow-Paced Videos:**
1. Decrease Max FPS to 2.0
2. Increase change threshold to 0.4 (less sensitive)
3. Use longer TAS kernels (4/16/64)
4. Use larger Qwen model (1.5B) for better reasoning

### Interpreting Results

**High Confidence Answers:**
- Trust the answer
- Use timestamps to verify
- Check reasoning path if curious

**Medium Confidence Answers:**
- Answer is probably correct
- Verify with video if important
- Consider rephrasing question

**Low Confidence Answers:**
- Answer may be incorrect
- Try rephrasing the question
- Check if question is answerable from video
- Verify event detection in timeline

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Video file too large"

**Cause:** Video exceeds 10 GB limit

**Solutions:**
1. Compress the video using ffmpeg:
   ```bash
   ffmpeg -i input.mp4 -vcodec h264 -acodec aac output.mp4
   ```
2. Trim unnecessary sections
3. Reduce video resolution
4. Split into smaller segments

#### Issue: "Out of memory during processing"

**Cause:** Insufficient RAM or GPU memory

**Solutions:**
1. Close other applications
2. Reduce TAS window size to 32
3. Reduce SmolVLM context window to 4
4. Use smaller models (256M SmolVLM, 0.5B Qwen)
5. Reduce Max FPS to 3.0
6. Force CPU mode (slower but uses less GPU memory)

#### Issue: "Processing is very slow"

**Cause:** CPU-only mode or large video

**Solutions:**
1. Verify GPU is available:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Install CUDA if not available
3. Reduce Max FPS to 3.0
4. Use CLIP instead of SmolVLM
5. Use smaller models
6. Reduce video length

#### Issue: "No events detected"

**Cause:** Video is too static or change threshold too high

**Solutions:**
1. Lower change threshold to 0.2
2. Increase base FPS to 2.0
3. Increase Max FPS to 10.0
4. Verify video has actual content (not blank)
5. Check video preview plays correctly

#### Issue: "Poor answer quality"

**Cause:** Low-quality event detection or reasoning

**Solutions:**
1. Use SmolVLM instead of CLIP for better descriptions
2. Increase TAS window to 128
3. Use larger Qwen model (1.5B or 3B)
4. Lower causal threshold to 0.6 for more relationships
5. Rephrase question to be more specific
6. Check confidence scores to assess reliability

#### Issue: "Cannot access UI from other devices"

**Cause:** Server bound to localhost only

**Solutions:**
1. Launch with `--host 0.0.0.0`:
   ```bash
   python scripts/launch_ui.py --host 0.0.0.0
   ```
2. Check firewall settings
3. Verify devices are on same network
4. Use correct IP address (not localhost)

#### Issue: "Models not downloading"

**Cause:** Network issues or insufficient disk space

**Solutions:**
1. Check internet connection
2. Verify disk space (need ~5 GB)
3. Check firewall/proxy settings
4. Manually download models:
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')"
   ```

#### Issue: "Causal graph not showing"

**Cause:** No causal relationships detected

**Solutions:**
1. Lower causal threshold to 0.6
2. Verify events were detected (check timeline)
3. Use SmolVLM for better event descriptions
4. Check if video has causal relationships
5. Try a different video

#### Issue: "Query returns empty answer"

**Cause:** No relevant events found

**Solutions:**
1. Rephrase question to be more general
2. Check if question is answerable from video
3. Verify video processed correctly
4. Try a summary query first
5. Check timeline to see what events were detected

### Error Messages

#### "Please process a video first"
**Solution:** Upload and process a video before asking questions

#### "Question too long (max 512 characters)"
**Solution:** Shorten your question or split into multiple questions

#### "Unsupported video format"
**Solution:** Convert video to MP4, AVI, MOV, or WebM

#### "Video file not found"
**Solution:** Re-upload the video file

#### "CUDA out of memory"
**Solution:** Follow "Out of memory" solutions above

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for error messages in the console
2. **Enable debug mode**: Launch with `--debug` flag
3. **Search GitHub issues**: https://github.com/yourusername/sharingan/issues
4. **Ask in discussions**: https://github.com/yourusername/sharingan/discussions
5. **Report a bug**: Create a new issue with:
   - Video details (length, format, size)
   - Configuration used
   - Error message
   - Steps to reproduce

---

## Keyboard Shortcuts

### Global Shortcuts

- **Enter**: Submit question (when in question input box)
- **Ctrl+C**: Stop server (in terminal)
- **F5**: Refresh page (resets UI state)

### Text Input Shortcuts

- **Ctrl+A**: Select all text
- **Ctrl+C**: Copy text
- **Ctrl+V**: Paste text
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo

### Browser Shortcuts

- **Ctrl+T**: New tab
- **Ctrl+W**: Close tab
- **Ctrl+Shift+T**: Reopen closed tab
- **Ctrl+R**: Reload page
- **Ctrl++**: Zoom in
- **Ctrl+-**: Zoom out
- **Ctrl+0**: Reset zoom

---

## Additional Resources

### Documentation

- **Main README**: [README.md](../README.md)
- **Architecture Guide**: [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Configuration Guide**: [gradio_ui_configuration.md](./gradio_ui_configuration.md)
- **Launch Guide**: [gradio_ui_launch.md](./gradio_ui_launch.md)
- **API Reference**: [api_reference.md](./api_reference.md)

### Examples

- **Basic Usage**: [examples/basic_usage.py](../examples/basic_usage.py)
- **Advanced Queries**: [examples/advanced_query.py](../examples/advanced_query.py)
- **Training Scorer**: [examples/training_scorer.py](../examples/training_scorer.py)

### Community

- **GitHub Repository**: https://github.com/yourusername/sharingan
- **Issue Tracker**: https://github.com/yourusername/sharingan/issues
- **Discussions**: https://github.com/yourusername/sharingan/discussions
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

### Support

- **Email**: support@sharingan.ai
- **Discord**: [Join our Discord](https://discord.gg/sharingan)
- **Twitter**: [@sharingan_ai](https://twitter.com/sharingan_ai)

---

## Conclusion

You now have a comprehensive understanding of the SHARINGAN UI! Here's a quick recap:

1. **Upload** your video and choose settings
2. **Process** the video once (this takes time)
3. **Ask** unlimited questions (fast responses)
4. **Explore** advanced features (graphs, timelines, reasoning)
5. **Adjust** configuration for your specific needs

Remember:
- ✅ Process once, query forever
- ✅ All processing is local and private
- ✅ Zero API costs
- ✅ Fast query responses (<500ms)
- ✅ Transparent reasoning and confidence scores

**Happy video understanding!** 🎬🧠

---

*Last updated: 2024*  
*SHARINGAN Version: 1.0*  
*Documentation Version: 1.0*
