# 👁️ Sharingan - Semantic Video Understanding

<p align="center">
  <img src="https://media1.tenor.com/m/YeM3fMlamBoAAAAd/naruto.gif" alt="Sharingan GIF" style="width:100%; height:auto;"/>
</p>


**Sharingan** is a lightweight Python library for semantic video understanding with temporal reasoning. It combines vision-language models (CLIP, SmolVLM) with temporal analysis to understand video content at a deep semantic level.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

* 🎬 **Semantic Video Processing** – Understand video content beyond pixels
* 🔍 **Natural Language Queries** – Search videos using text descriptions
* 🤖 **AI Chat** – Conversational interface with Qwen2.5-0.5B
* ⚡ **Temporal Reasoning** – Cross-frame attention and memory tokens
* 🎯 **Event Detection** – Automatically identify key moments
* 💾 **Efficient Storage** – 130x compression with Int8 quantization
* 🚀 **Fast Processing** – Batch processing and GPU acceleration

---
You can read the [Author Note](https://github.com/skhavindev/sharingan/blob/master/author_note.md), check out the [Architecture](https://github.com/skhavindev/sharingan/blob/master/architecture.md), and see the [Contributing Guidelines](https://github.com/skhavindev/sharingan/blob/master/contributing.md) on GitHub.

---

## 🚀 Quick Start

### Installation

```bash
pip install sharingan-core

# Optional: GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: AI chat
pip install transformers bitsandbytes accelerate
```

### Basic Usage

```python
from sharingan import VideoProcessor

processor = VideoProcessor(
    vlm_model='clip',  # or 'smolvlm'
    device='auto'
)

results = processor.process('video.mp4')

matches = processor.query('person speaking')
for match in matches:
    print(f"Found at {match.timestamp}s - {match.confidence:.2%}")

response = processor.chat('What happens in this video?')
print(response)
```

---

## 📖 Documentation

**Vision Models**

* **CLIP** – Fast semantic embeddings; memory ~400MB
* **SmolVLM-500M** – Detailed frame descriptions; memory ~538MB (8-bit quantized)

**Processing Options**

```python
processor = VideoProcessor(
    vlm_model='clip',
    device='auto',
    target_fps=5.0,
    enable_temporal=True,
    enable_tracking=False
)
```

**Query Options**

```python
results = processor.query('person speaking', top_k=5)
response = processor.chat('Describe main events', use_llm=True)
```

---

## 🎯 Use Cases

* Video Search – Find moments using natural language
* Content Moderation – Detect inappropriate content
* Video Summarization – Auto summaries
* Accessibility – Descriptions for visually impaired
* Research – Analyze video datasets at scale

---

## 🌐 Gradio Web Interface

SHARINGAN includes a modern Gradio-based web interface for intuitive video processing and querying. The UI provides a complete visual experience for interacting with the deep video understanding system.

### Quick Start

Launch the UI with one command:

```bash
python scripts/launch_ui.py
```

The interface will open at `http://localhost:7860` in your browser.

### Launch Options

```bash
# Launch on custom host and port
python scripts/launch_ui.py --host 0.0.0.0 --port 8080

# Create public URL for sharing (great for demos!)
python scripts/launch_ui.py --share

# Enable authentication for security
python scripts/launch_ui.py --auth username:password

# Load pre-processed videos from cache
python scripts/launch_ui.py --load-cache ./cache

# Combine options for production deployment
python scripts/launch_ui.py --host 0.0.0.0 --port 8080 --auth admin:pass --share
```

### UI Components

The Gradio interface includes four main sections:

#### 1. 📹 Video Upload & Processing
- **Drag-and-drop** video upload (supports MP4, AVI, MOV, WebM)
- **Real-time progress** tracking with ETA and memory usage
- **Processing status** display showing current pipeline stage
- **Automatic caching** for instant re-querying

![Video Upload Interface](docs/images/ui_upload.png)

#### 2. ⚙️ Advanced Configuration Panel
- **Multi-Scale TAS Settings** – Adjust temporal attention scales (short/mid/long)
- **Adaptive Sampler** – Configure FPS bounds and change detection
- **Cross-Modal Verifier** – Set similarity thresholds for hallucination detection
- **Causal Edge Scorer** – Choose heuristic or learned mode
- **Model Selection** – Pick SmolVLM, CLIP, and Qwen variants
- **Preset Profiles** – Fast, Balanced, High Accuracy configurations

![Configuration Panel](docs/images/ui_config.png)

#### 3. 💬 Query Interface
- **Natural language input** – Ask questions in plain English
- **Example queries** dropdown with common question types:
  - "What happens between 0:30 and 1:00?" (window query)
  - "Find person speaking" (semantic query)
  - "Why did the person leave?" (causal query)
  - "Summarize this video" (summary query)
- **Query history** – Review previous questions and answers
- **Response with timestamps** – Answers include precise time references

![Query Interface](docs/images/ui_query.png)

#### 4. 📊 Visualizations & Results
- **Causal Graph View** – Interactive network showing event relationships
- **Timeline View** – Visual representation of detected events
- **Reasoning Scaffold** – Step-by-step reasoning path display
- **Confidence Scores** – Visual indicators for answer reliability
- **Event Details** – Expandable cards with frame descriptions

![Visualizations](docs/images/ui_viz.png)

### Example Usage Workflow

Here's a typical workflow using the Gradio UI:

**Step 1: Upload Video**
```
1. Click "Upload Video" or drag-and-drop your video file
2. Wait for upload to complete (progress bar shows status)
3. Video preview appears automatically
```

**Step 2: Configure Processing (Optional)**
```
1. Expand "⚙️ Advanced Configuration" panel
2. Choose a preset: "Fast", "Balanced", or "High Accuracy"
   - OR customize individual settings
3. Adjust Multi-Scale TAS kernels for your video type:
   - Fast-paced (sports): [2, 4, 16]
   - Slow-paced (lectures): [4, 16, 64]
4. Set sampling rate based on video length:
   - Short videos (<5 min): Max FPS = 5.0
   - Long videos (>1 hour): Max FPS = 2.0
```

**Step 3: Process Video**
```
1. Click "Process Video" button
2. Watch real-time progress:
   - "Sampling frames..." (adaptive frame selection)
   - "Generating descriptions..." (SmolVLM processing)
   - "Verifying descriptions..." (CLIP verification)
   - "Building event graph..." (causal edge scoring)
   - "Creating memory store..." (hierarchical storage)
3. Processing completes with summary:
   - Total frames processed
   - Events detected
   - Causal relationships found
   - Processing time
```

**Step 4: Query Your Video**
```
1. Type your question in the query box, or select from examples:
   - "What happens in the first minute?"
   - "Find all scenes with people talking"
   - "Why did the person pick up the object?"
   - "Summarize the main events"
2. Click "Submit Query" or press Enter
3. View results:
   - Natural language answer with timestamps
   - Reasoning path showing how answer was derived
   - Confidence score
   - Related events in timeline
4. Click on timeline events to see frame details
5. Explore causal graph to understand relationships
```

**Step 5: Iterate and Refine**
```
1. Ask follow-up questions based on initial results
2. Adjust configuration if needed (e.g., lower thresholds for more events)
3. Re-process video with new settings (uses cache when possible)
4. Export results or save processed video for later
```

### UI Features

* 📹 **Video Upload** – Drag-and-drop interface with format validation
* ⚙️ **Advanced Configuration** – Full control over all pipeline parameters
* 💬 **Chat Interface** – Natural language queries with example templates
* 📊 **Visualizations** – Interactive causal graphs and event timelines
* 🔒 **Privacy** – All processing happens locally, no external API calls
* 💾 **Caching** – Instant re-querying of processed videos
* 🎨 **Responsive Design** – Works on desktop, tablet, and mobile
* 🌐 **Public Sharing** – Optional public URLs for demos and collaboration

### Why Gradio?

SHARINGAN uses Gradio instead of traditional web frameworks because:

- **ML/AI Optimized** – Built specifically for machine learning applications
- **Zero Frontend Code** – No HTML/CSS/JavaScript required
- **Real-time Updates** – Built-in progress tracking and streaming
- **Interactive Visualizations** – Native support for plots, graphs, and media
- **One-Command Deploy** – Launch with a single Python command
- **Public Sharing** – Create shareable URLs instantly with `--share`
- **Authentication** – Built-in user authentication support
- **Modern UI** – Clean, professional interface out of the box

### Documentation

For complete UI documentation, see:
- **Launch Guide**: [docs/gradio_ui_launch.md](docs/gradio_ui_launch.md)
- **Configuration Guide**: [docs/gradio_ui_configuration.md](docs/gradio_ui_configuration.md)

---

## 🔧 Advanced Features

**Temporal Reasoning**

* Cross-Frame Gating – Learns important frames
* Memory Tokens – Maintains context across video
* Temporal Attention – Understand relationships between frames

**Efficient Storage**

* 5-min video: ~2.3MB (vs 300MB raw)
* Fast cache loading
* Minimal quality loss for search

**Event Detection**

* Scene changes
* Motion patterns
* Content transitions

---

## 📊 Performance

| Model   | Speed | Memory | Quality   |
| ------- | ----- | ------ | --------- |
| CLIP    | ⚡⚡⚡   | 400MB  | Good      |
| SmolVLM | ⚡⚡    | 538MB  | Excellent |

*Tested on NVIDIA RTX 3050 (4GB VRAM)*

---

## 🤝 Contributing

Contributions welcome! Please submit a PR.

## 📄 License

MIT License – see LICENSE file.

## 🙏 Acknowledgments

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
* [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## 📧 Contact

Open an issue on GitHub for support.

---

Made with ☕ & ❤️

