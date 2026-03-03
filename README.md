# 👁️ Trinetra - Deep Temporal Video Understanding

<p align="center">
  <img src="https://media1.tenor.com/m/n7LORMHtCRcAAAAC/marah-sedih.gif" alt="Trinetra GIF" style="width:100%; height:auto;"/>
</p>


**Trinetra** is a proactive video understanding system that processes video once and enables unlimited queries at near-zero cost. Unlike reactive models (Gemini, GPT-4o) that re-process video for every query, Trinetra builds a rich temporal event graph during ingestion and answers from structured text.

> **The Core Insight:** A 0.5B model reading perfect text beats a 70B model squinting at compressed frames.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Website](https://img.shields.io/badge/website-trinetra-blue)](https://skxdev007.github.io/trinetra)

---

## 🎯 Why Trinetra?

**Reactive Models (Gemini/GPT-4o):**
- ❌ Process video at query time
- ❌ Limited by context window (~30-60 min)
- ❌ Expensive per query ($50 per 100 queries)
- ❌ Slow (15s+ latency)
- ❌ Lose information in compression

**Trinetra (Proactive):**
- ✅ Process video once at ingest
- ✅ Unlimited temporal span (tested to 155 min)
- ✅ Near-zero cost per query (<$0.01 per 100 queries)
- ✅ Fast (<1s latency)
- ✅ Preserve meaning in structured text

---

## 🚀 Key Results

| Metric | Reactive Models | Trinetra |
|--------|----------------|----------|
| **Max Video Length** | 30-60 min | 155+ min |
| **Query Latency** | 15.2s | 0.8s |
| **Cost per 100 Queries** | $50 | <$0.01 |
| **Frame Processing** | All 279K frames | 9.6K frames (96.6% reduction) |
| **Hardware** | H100 Cluster | Consumer Laptop |

**Real-World Test:** 2.5-hour woodworking video
- ✅ 605 events detected (1 every 15 seconds)
- ✅ 99.3% temporal precision for "final result" queries
- ✅ Cross-horizon reasoning (linked events 2+ hours apart)
- ✅ 15.94x realtime processing speed

---

## ✨ Features

* 🎬 **Temporal Event Graph** – Causal relationships between events
* 🔍 **Natural Language Queries** – Search videos using text descriptions
* 🤖 **Small LLM Backend** – Qwen-0.5B with reasoning scaffolds
* ⚡ **Multi-Scale Temporal Reasoning** – Short/medium/long-term patterns
* 🎯 **Adaptive Frame Sampling** – 96.6% frame reduction with no quality loss
* 💾 **Hierarchical Memory** – Frame/Event/Episode/Video levels
* 🚀 **15.94x Realtime Processing** – Process 2.5 hours in 9.7 minutes
* 🌐 **Web Interface** – Modern Gradio UI for easy interaction

---

## 📊 Architecture

Trinetra uses a proactive architecture that processes video once and enables unlimited queries:

```
Video → Adaptive Sampling → Multi-Scale TAS → Event Detection → 
Temporal Event Graph → Hierarchical Memory → Query Router → Small LLM
```

**Key Components:**

1. **Multi-Scale Temporal Adaptive Sampling (TAS)** – Captures patterns at multiple time scales
2. **Temporal Event Graph** – Causal relationships between events
3. **Cross-Modal Verification** – Reduces hallucinations via audio/text/temporal consistency
4. **Hierarchical Memory** – 4-level compression (frame → event → episode → video)
5. **Context-Aware Query Routing** – Routes queries to appropriate memory levels
6. **Reasoning Scaffolds** – Guides small LLMs through complex temporal reasoning

For detailed architecture, visit: [https://skxdev007.github.io/trinetra](https://skxdev007.github.io/trinetra)

---

## 🚀 Quick Start

### Installation

```bash
pip install trinetra

# Optional: GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: AI chat with small LLM
pip install transformers bitsandbytes accelerate
```

### Basic Usage

```python
from trinetra import VideoProcessor

# Initialize processor
processor = VideoProcessor(
    vlm_model='clip',  # or 'smolvlm'
    device='auto',
    enable_temporal=True
)

# Process video once (builds event graph)
results = processor.process('video.mp4')
print(f"Detected {len(results['events'])} events")

# Query unlimited times at near-zero cost
matches = processor.query('person speaking', top_k=5)
for match in matches:
    print(f"Found at {match['timestamp']:.1f}s - confidence: {match['confidence']:.2%}")

# Chat with AI about the video
response = processor.chat('What happens in this video?', use_llm=True)
print(response)
```

### Advanced Queries

```python
# Temporal queries
processor.query("What happens between 1:30 and 2:00?")

# Causal queries
processor.query("Why did the person leave?")

# Counting queries
processor.query("How many times does X appear?")

# Summary queries
processor.query("Summarize the main events")
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

* **Educational Content** – Process lectures once, students query forever
* **Long-Form Analysis** – 2.5-hour videos with perfect temporal recall
* **Procedural Understanding** – Extract step-by-step workflows with causal relationships
* **Temporal QA** – Answer complex questions: "What happened between X and Y?"
* **Content Moderation** – Detect events across entire video timeline
* **Research** – Analyze video datasets at scale with minimal compute

---

## 🌐 Web Interface

Trinetra includes a modern Gradio-based web interface for intuitive video processing and querying.

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

# Combine options for production deployment
python scripts/launch_ui.py --host 0.0.0.0 --port 8080 --auth admin:pass --share
```

### UI Features

* 📹 **Video Upload** – Drag-and-drop interface with format validation
* ⚙️ **Advanced Configuration** – Full control over all pipeline parameters
* 💬 **Chat Interface** – Natural language queries with example templates
* � **Visualizations** – Interactive causal graphs and event timelines
* 🔒 **Privacy** – All processing happens locally, no external API calls
* 💾 **Caching** – Instant re-querying of processed videos

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

Contributions welcome! Please submit a PR or open an issue.

## 📄 License

Apache 2.0 License – see LICENSE file.

## 🙏 Acknowledgments

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
* [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## � Citation

If you use Trinetra in your research, please cite:

```bibtex
@software{trinetra2026,
  title={Trinetra: Proactive Video Understanding via Temporal Event Graphs},
  author={S Khavin},
  year={2026},
  url={https://github.com/skxdev007/trinetra}
}
```

## 📧 Contact

Open an issue on GitHub for support or visit [https://skxdev007.github.io/trinetra](https://skxdev007.github.io/trinetra)

---

Made with ☕ & ❤️

