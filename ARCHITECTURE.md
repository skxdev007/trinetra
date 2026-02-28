<p align="center">
<pre>
                           ⢀⣀⣠⣤⣤⣤⣤⣄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⢀⣠⣶⣿⣿⣿⣿⡿⠃⠘⢿⣿⣿⣿⣿⣶⣄⡀⠀⠀⠀⠀⠀
                ⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀
                ⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀
                ⠀⣸⣿⡉⠀⡀⠈⠉⠉⢙⡟⠲⠤⣄⣠⠤⠖⢻⡋⠉⠉⠁⠀⠀⠉⣿⣧⠀
                ⢰⣿⣿⣷⡀⠈⢻⣷⣶⣼⣤⣔⠊⠁⠈⠑⣢⣤⣧⣶⣾⡿⠁⢀⣾⣿⣿⡆
                ⣼⣿⣿⣿⣿⣄⠀⢉⡿⣿⣿⣿⡿⠖⠲⢿⣿⣿⣿⠿⡋⠀⣠⣾⣿⣿⣿⣷
                ⣿⣿⣿⣿⣿⣿⡷⣏⠀⡏⠻⢿⡁⠀⠀⢈⡿⠟⢹⠀⣨⢾⣿⣿⣿⣿⣿⣿
                ⢻⣿⣿⣿⡿⠋⠀⠈⠳⣧⡀⠀⣷⣦⣴⣾⠀⢀⣸⠞⠁⠀⠙⢿⣿⣿⣿⡿
                ⠸⣿⣿⡿⠁⠀⠀⠀⠀⢸⠉⠲⢼⣿⣿⡯⠖⠋⡇⠀⠀⠀⠀⠈⢿⣿⣿⠇
                ⠀⠹⣿⣀⣀⣀⣀⣀⣀⣨⣧⠔⠚⣿⣿⠗⠢⢼⣅⣀⣀⣀⣀⣀⣀⣿⡏⠀
                ⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⢻⡿⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀
                ⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣷⡀⠈⠃⢀⣾⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀
                ⠀⠀⠀⠀⠀⠈⠙⠻⢿⣿⣿⣿⣷⣄⢠⣾⣿⣿⣿⡿⠿⠋⠁⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠈⠉⠙⠛⠛⠛⠛⠋⠉⠁⠀⠀⠀⠀⠀⠀
</pre>
</p>

# **Sharingan Architecture**

This document describes the internal architecture of the Sharingan system.
It is intended for researchers, contributors, and advanced users who require a detailed understanding of the framework’s modular design, internal pipelines, and novel temporal reasoning mechanisms.

---

# **1. System Overview**

Sharingan provides a unified pipeline for efficient, scalable, and semantic video understanding. It integrates frame-level vision encoding, lightweight vision-language modeling, novel temporal reasoning modules, and compressed embedding storage to enable fast retrieval, event detection, and natural language querying.

```
Video → Frame Sampling → VLM Encoding → Temporal Reasoning → Storage → Query Engine
```

---

# **2. Core Subsystems**

## **2.1 Video Processing (`sharingan/video/`)**

### **VideoLoader (`loader.py`)**

* Handles decoding via OpenCV or PyAV
* Extracts framerate, duration, resolution, bit-depth
* Supports streaming and offline files

### **FrameSampler (`sampler.py`)**

* Uniform sampling (N frames per second)
* Adaptive sampling using motion estimation
* Keyframe-based sampling (I-frame detection)
* Reduces computational overhead while preserving semantic content

### **VideoProcessor (`api.py`)**

* High-level orchestrator for the entire pipeline
* Performs caching, state management, and concurrency control

---

## **2.2 Vision-Language Models (`sharingan/vlm/`)**

### **FrameEncoder (`encoder.py`)**

* CLIP-based encoding (ViT-B/32, ViT-B/16, ViT-L/14)
* Produces 512-dim embeddings for each frame
* Batch inference optimized for GPU/CPU

### **SmolVLMEncoder (`smolvlm.py`)**

* Leverages SmolVLM-500M for multi-frame reasoning
* Generates natural-language descriptions per sampled frame window
* Outputs text embeddings via CLIP text encoder
* Supports 8-bit quantization for efficient deployment

---

# **3. Temporal Reasoning (`sharingan/temporal/`)**

The temporal engine represents Sharingan’s primary innovation.
It integrates five novel and lightweight modules enabling efficient long-range temporal context modeling without the cost of 3D CNNs or full video transformers.

---

# **3.1 Temporal Engine (`engine.py`)**

* Coordinates all temporal processing modules
* Maintains cross-frame state and accumulated video memory
* Ensures temporal modules operate in streaming or batch modes

---

# **3.2 Cross-Frame Gating Network (`gating.py`)**

A lightweight network designed to estimate the relative importance of adjacent frames.

**Core Concepts:**

* Two-frame input concatenation `[Ft-1 || Ft]`
* Two-layer MLP + sigmoid gating head
* Gate controls how much information is propagated from the previous frame

**Properties:**

* <1M parameters
* Parallelizable (unlike ConvLSTM)
* Enables dynamic frame influence weighting

---

# **3.3 Temporal Memory Tokens (`memory_tokens.py`)**

A persistent, learnable memory mechanism for video-level context.

**Structure:**

* `M` tokens (default: 8), each 512-dim
* Cross-attention between frame embeddings and the memory bank
* Gated residual update:
  `Mt ← gate * Mt + (1 - gate) * update`

**Capabilities:**

* Captures >1000 frames of context
* Enables long-range coherence for event detection and querying
* Constant memory cost → supports streaming inference

---

# **3.4 Temporal Difference Attention (TDA) (`tda.py`)**

A multi-scale attention mechanism focusing on frame-to-frame changes.

**Design:**

* Computes difference embeddings Δ(Ft, Ft−k) for k ∈ {1, 4, 8, 16}
* Applies dilated temporal attention across these offsets
* Produces long-range temporal signals at low computational cost

**Advantages:**

* Captures both short-term motion and long-term transitions
* ~50% faster than full self-attention across all frames
* Scales efficiently to long videos

---

# **3.5 Motion-Aware Adaptive Pooling (`motion_pooling.py`)**

Integrates classical optical-flow-based motion estimation with embedding pooling.

**Workflow:**

1. Estimate motion using Farnebäck or RAFT-lite flow
2. Compute motion magnitude maps
3. Allocate higher weights to dynamic frames
4. Pool embeddings based on motion intensity

**Impact:**

* Reduces redundant computation
* Emphasizes semantically relevant temporal segments
* Ideal for low-FPS surveillance and static scenes

---

# **3.6 Novel Module: Temporal Attention Shift (TAS)**

A lightweight, differentiable substitute for 3D convolutions inspired by channel-shift operations.

**Purpose:**
Efficient temporal feature exchange without dense attention or 3D kernels.

### **Mechanism**

For each frame embedding `Ft`:

1. Split channels into three groups:

   * forward-shifted
   * backward-shifted
   * static
2. Assign learnable, per-channel attention weights
3. Shift feature slices across time:

   * forward slice receives features from `Ft-1`
   * backward slice uses features from `Ft+1`
4. Fuse with sigmoid-gated attention.

**Computational Cost:**
`O(T × C)` — linear in time and channels.

### **Advantages**

* Dramatically cheaper than 3D CNNs
* Better adaptive behavior than fixed TSM shifts
* No temporal convolution required
* Drop-in replacement for temporal modules in lightweight video models

### **Why TAS is Novel**

Traditional Temporal Shift Modules (TSM) use **fixed**, non-learnable channel shifts.
TAS introduces **learnable temporal mixing**, making the shift:

* Content-aware
* Video-adaptive
* Differentiable end-to-end
* Compatible with both CLIP and VLM embeddings
* Suitable for streaming video

---

# **4. Storage Layer (`sharingan/storage/`)**

### **EmbeddingStore (`embedding_store.py`)**

* Serializes embeddings, timestamps, metadata
* Supports three precision types:

  * FP32
  * FP16
  * INT8 (default, 130× compression)
* Average storage for a **5-minute video**: ~2.3MB

The store is optimized for:

* Sequential writes (video streams)
* Zero-copy partial reads
* Memory-mapped loading

---

# **5. Event Detection System (`sharingan/events/`)**

### **EventDetector (`detector.py`)**

Detects semantic changes using embedding deltas and temporal attention signals.

Supported event types:

* Scene changes
* High motion episodes
* Content transitions
* Entity appearance/disappearance

Events are timestamped and can be visualized in the UI.

---

# **6. Query Engine (`sharingan/query/`)**

### **NaturalLanguageQuery (`nl_query.py`)**

Implements CLIP-style text-to-video retrieval.

Features:

* Text → embedding encoding
* Dot-product similarity search
* Efficient top-K search across entire video embedding arrays
* Optional LLM-based query interpretation

---

# **7. Conversational Agent (`sharingan/chat/`)**

### **VideoLLM (`llm.py`)**

* Qwen2.5-0.5B (8-bit) for conversational inference
* RAG pipeline over video embeddings
* Chain-of-thought reasoning with temporal awareness

---

# **8. Processing Pipeline**

## **8.1 Video Processing Flow**

```
Video
   ↓
Frame Sampling (adaptive FPS)
   ↓
Vision Encoding (CLIP / SmolVLM)
   ↓
Temporal Engine
   ↓
Embedding Storage (INT8)
   ↓
Event Detector
   ↓
Queryable Cache
```

---

## **8.2 Query Processing Flow**

```
User Query
   ↓
Text Encoder (CLIP)
   ↓
Similarity Search
   ↓
Top-K Timestamps
   ↓
Optional: LLM Explanation
   ↓
Returned Results
```

---

# **9. Performance Characteristics**

| Component         | Memory | Notes            |
| ----------------- | ------ | ---------------- |
| CLIP ViT-B/32     | ~400MB | FP16             |
| SmolVLM-500M      | ~538MB | 8-bit            |
| Qwen 0.5B         | ~538MB | 8-bit            |
| Embeddings (5min) | ~2.3MB | INT8             |
| Temporary Buffers | ~100MB | Depends on video |

**Total system footprint:** ~1.5GB

---

# **10. Extensibility**

### **Adding a New Temporal Module**

```python
class CustomTemporalModule:
    def process(self, embeddings, timestamps):
        return modified_embeddings
```

### **Adding a Custom Encoder**

```python
class CustomEncoder:
    def encode_frame(self, frame):
        return self.model(frame)
```

---

# **11. Future Work**

* Audio-video fusion
* Real-time stream ingestion
* Distributed multi-node processing
* Learning-based quantization
* Multi-object tracking
* 3D scene graph construction

---