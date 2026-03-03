# SHARINGAN Deep Architecture - Complete Learning Roadmap

**For:** Complete beginners (10th grade level)  
**Goal:** Understand everything in SHARINGAN's architecture from scratch  
**Time:** 4-6 weeks of dedicated study

---

## 📚 Learning Path Overview

```
Level 1: Foundations (Week 1-2)
    ↓
Level 2: Core Concepts (Week 2-3)
    ↓
Level 3: Advanced Topics (Week 3-4)
    ↓
Level 4: SHARINGAN Architecture (Week 4-6)
```

---

## 🎯 SHARINGAN Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHARINGAN DEEP ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      INGEST PIPELINE (Run Once)                  │
└─────────────────────────────────────────────────────────────────┘

Video File
    ↓
┌──────────────────────┐
│ 1. Adaptive Sampler  │  ← Selects important frames (1-8 FPS)
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 2. CLIP Encoder      │  ← Converts frames to 512-dim vectors
│    (ViT-B/32)        │     (Vision Transformer)
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 3. Multi-Scale TAS   │  ← Temporal reasoning (3 scales + GRU)
│    - Short (2 frames)│     Understands gestures → actions → scenes
│    - Mid (8 frames)  │
│    - Long (32 frames)│
│    - GRU Memory      │
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 4. Event Detector    │  ← Finds scene changes & important moments
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 5. Hierarchical      │  ← Stores 3 levels:
│    Memory Store      │     - Frames (dense)
│                      │     - Events (semantic)
│                      │     - Chapters (summary)
└──────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (Run Forever)                  │
└─────────────────────────────────────────────────────────────────┘

User Query: "When is RAM installed?"
    ↓
┌──────────────────────┐
│ 1. Query Router      │  ← Classifies query type:
│                      │     - Window (time-based)
│                      │     - Semantic (similarity)
│                      │     - Causal (why/how)
│                      │     - Summary (overview)
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 2. Memory Retrieval  │  ← Searches hierarchical memory
│    (FAISS/Linear)    │     Returns top-5 relevant moments
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 3. Temporal Filter   │  ← Applies fixes:
│                      │     - Teaser bias filter
│                      │     - Temporal weighting
│                      │     - First-frame penalty
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 4. Reasoning         │  ← Builds structured reasoning:
│    Scaffold Builder  │     - Causal chain
│                      │     - Temporal order
│                      │     - State change
└──────────────────────┘
    ↓
┌──────────────────────┐
│ 5. Small LLM         │  ← Generates natural language answer
│    (Qwen-0.5B)       │     "RAM installed at 15:03"
└──────────────────────┘
```

---

## 📖 Level 1: Foundations (Week 1-2)

### 1.1 Neural Networks Basics

**What you need to know:**
- What is a neural network?
- How do neurons work?
- What is forward propagation?
- What is backpropagation?
- What are weights and biases?

**Resources:**

**📄 Paper:** None (start with tutorials)

**🎥 Video Tutorial:**
- 3Blue1Brown - Neural Networks Series (4 videos, ~1 hour total)
  - https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

**📚 Interactive:**
- Neural Network Playground: https://playground.tensorflow.org/

**Key Concepts:**
- Neuron = takes inputs, multiplies by weights, adds bias, applies activation
- Layer = collection of neurons
- Network = stack of layers
- Training = adjusting weights to minimize error

---

### 1.2 Convolutional Neural Networks (CNNs)

**What you need to know:**
- What is a convolution?
- How do filters/kernels work?
- What is pooling?
- Why CNNs for images?

**📄 Papers to Read:**

1. **LeNet-5 (1998)** - The first CNN
   - Paper: "Gradient-Based Learning Applied to Document Recognition"
   - Authors: Yann LeCun et al.
   - Link: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
   - **Read:** Sections 2-3 (CNN architecture)
   - **Skip:** Mathematical proofs

2. **AlexNet (2012)** - Deep learning revolution
   - Paper: "ImageNet Classification with Deep Convolutional Neural Networks"
   - Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
   - Link: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
   - **Read:** Sections 1-3 (architecture and training)
   - **Key insight:** Deep CNNs + GPUs + ReLU = breakthrough

3. **ResNet (2015)** - Skip connections
   - Paper: "Deep Residual Learning for Image Recognition"
   - Authors: Kaiming He et al.
   - Link: https://arxiv.org/pdf/1512.03385.pdf
   - **Read:** Sections 1-3 (residual learning concept)
   - **Key insight:** Skip connections solve vanishing gradient

**🎥 Video:**
- Stanford CS231n Lecture 5: CNNs
  - https://www.youtube.com/watch?v=bNb2fEVKeEo

**Key Concepts:**
- Convolution = sliding filter over image to detect patterns
- Filter = learns to detect edges, textures, shapes
- Pooling = downsampling to reduce size
- Deep CNN = stack many layers to learn complex features

---

### 1.3 Transformers & Attention

**What you need to know:**
- What is attention mechanism?
- How do transformers work?
- Why transformers replaced RNNs?

**📄 Papers to Read:**

1. **Attention is All You Need (2017)** - THE transformer paper
   - Authors: Vaswani et al. (Google Brain)
   - Link: https://arxiv.org/pdf/1706.03762.pdf
   - **Read:** Sections 1-3 (attention mechanism, architecture)
   - **Skip:** Section 5 (training details)
   - **Key insight:** Self-attention replaces recurrence

**🎥 Video:**
- The Illustrated Transformer (Blog + Video)
  - https://jalammar.github.io/illustrated-transformer/

**Key Concepts:**
- Attention = "which parts of input are important?"
- Self-attention = input attends to itself
- Multi-head attention = multiple attention patterns
- Positional encoding = adds position information

---

## 📖 Level 2: Core Concepts (Week 2-3)

### 2.1 Vision Transformers (ViT)

**What you need to know:**
- How to apply transformers to images?
- What are image patches?
- Why ViT works better than CNNs?

**📄 Paper to Read:**

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)**
- Authors: Dosovitskiy et al. (Google Research)
- Link: https://arxiv.org/pdf/2010.11929.pdf
- **Read:** Sections 1-3 (ViT architecture)
- **Key insight:** Split image into patches, treat as sequence

**Key Concepts:**
- Image → 16x16 patches → flatten → sequence of tokens
- Patch embedding = linear projection of flattened patch
- Position embedding = learnable position for each patch
- Transformer encoder = same as NLP transformers

**Diagram:**
```
Image (224x224)
    ↓
Split into patches (14x14 patches of 16x16 pixels)
    ↓
Flatten each patch (16x16x3 = 768 values)
    ↓
Linear projection (768 → 512 embedding)
    ↓
Add position embedding
    ↓
Transformer encoder (12 layers)
    ↓
Classification head
```

---

### 2.2 CLIP (Vision-Language Model)

**What you need to know:**
- How to connect vision and language?
- What is contrastive learning?
- How CLIP enables zero-shot classification?

**📄 Paper to Read:**

**Learning Transferable Visual Models From Natural Language Supervision (2021)**
- Authors: Radford et al. (OpenAI)
- Link: https://arxiv.org/pdf/2103.00020.pdf
- **Read:** Sections 1-2 (CLIP architecture, training)
- **Key insight:** Train image and text encoders together

**Key Concepts:**
- Dual encoder: Image encoder (ViT) + Text encoder (Transformer)
- Contrastive learning: Match correct image-text pairs
- Zero-shot: Can classify images without training examples
- Embedding space: Images and text in same 512-dim space

**Diagram:**
```
Image                    Text
  ↓                       ↓
ViT Encoder          Transformer
  ↓                       ↓
Image Embedding      Text Embedding
  (512-dim)             (512-dim)
       ↓                 ↓
       Cosine Similarity
            ↓
      Match Score
```

**This is the CORE of SHARINGAN!**
- SHARINGAN uses CLIP to convert frames to embeddings
- Query text is also converted to embedding
- Similarity search finds relevant frames

---

### 2.3 Recurrent Networks (LSTM/GRU)

**What you need to know:**
- How to process sequences?
- What is the vanishing gradient problem?
- How do LSTM/GRU solve it?

**📄 Papers to Read:**

1. **Long Short-Term Memory (1997)**
   - Authors: Hochreiter & Schmidhuber
   - Link: https://www.bioinf.jku.at/publications/older/2604.pdf
   - **Read:** Sections 1-3 (LSTM architecture)
   - **Key insight:** Gates control information flow

2. **Empirical Evaluation of Gated Recurrent Neural Networks (2014)**
   - Authors: Chung et al.
   - Link: https://arxiv.org/pdf/1412.3555.pdf
   - **Read:** Sections 1-2 (GRU architecture)
   - **Key insight:** GRU is simpler than LSTM, similar performance

**Key Concepts:**
- RNN = processes sequence one step at a time
- Vanishing gradient = gradients become too small in long sequences
- LSTM = 3 gates (forget, input, output) + cell state
- GRU = 2 gates (reset, update) - simpler than LSTM

**This is used in SHARINGAN's Multi-Scale TAS!**
- GRU maintains memory across entire video
- Helps understand long-range dependencies

---

## 📖 Level 3: Advanced Topics (Week 3-4)

### 3.1 Temporal Action Segmentation

**What you need to know:**
- How to understand actions in videos?
- What is temporal modeling?
- How to capture short/mid/long-term patterns?

**📄 Papers to Read:**

1. **Temporal Segment Networks (TSN) (2016)**
   - Authors: Wang et al.
   - Link: https://arxiv.org/pdf/1608.00859.pdf
   - **Read:** Sections 1-3 (TSN framework)
   - **Key insight:** Sample sparse frames across video

2. **Temporal Shift Module (TSM) (2019)**
   - Authors: Lin et al.
   - Link: https://arxiv.org/pdf/1811.08383.pdf
   - **Read:** Sections 1-3 (TSM architecture)
   - **Key insight:** Shift channels temporally for free

**Key Concepts:**
- Temporal modeling = understanding how things change over time
- Sparse sampling = don't need every frame
- Temporal shift = shift features between frames
- Multi-scale = capture different time scales

**This is the CORE of SHARINGAN's TAS!**
- Multi-Scale TAS uses 3 temporal scales (2/8/32 frames)
- Short scale = gestures (hand movements)
- Mid scale = actions (installing component)
- Long scale = scenes (entire build phase)

---

### 3.2 Optical Flow & Motion Analysis

**What you need to know:**
- How to detect motion in videos?
- What is optical flow?
- How to track objects?

**📄 Papers to Read:**

1. **Lucas-Kanade Method (1981)**
   - Authors: Lucas & Kanade
   - Link: https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_2/lucas_bruce_d_1981_2.pdf
   - **Read:** Sections 1-2 (optical flow basics)
   - **Key insight:** Assume brightness constancy

2. **Horn-Schunck Method (1981)**
   - Authors: Horn & Schunck
   - Link: http://image.diku.dk/imagecanon/material/HornSchunckOptical_Flow.pdf
   - **Read:** Sections 1-2 (global optical flow)
   - **Key insight:** Smoothness constraint

**Key Concepts:**
- Optical flow = motion of pixels between frames
- Brightness constancy = pixel intensity doesn't change
- Lucas-Kanade = local method (sparse flow)
- Horn-Schunck = global method (dense flow)

**This is PROPOSED for SHARINGAN improvements!**
- Optical flow can detect "installing" vs "showing"
- High motion = action happening
- Low motion = explanation or showcase

---

### 3.3 Video Question Answering (VideoQA)

**What you need to know:**
- How to answer questions about videos?
- What is temporal reasoning?
- What are VideoQA benchmarks?

**📄 Papers to Read:**

1. **NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions (2021)**
   - Authors: Xiao et al.
   - Link: https://arxiv.org/pdf/2105.08276.pdf
   - **Read:** Sections 1-3 (NExT-QA dataset, temporal reasoning)
   - **Key insight:** Causal and temporal reasoning

2. **EgoSchema: Long-Form Video Understanding (2023)**
   - Authors: Mangalam et al.
   - Link: https://arxiv.org/pdf/2308.09126.pdf
   - **Read:** Sections 1-2 (long-form video understanding)
   - **Key insight:** Temporal certificate sets

**Key Concepts:**
- VideoQA = answer questions about video content
- Temporal reasoning = understand time relationships
- Causal reasoning = understand cause-effect
- Benchmarks = datasets to evaluate models

**This is what SHARINGAN is designed for!**
- SHARINGAN answers temporal questions
- "When is X installed?" = temporal query
- "Why did X happen?" = causal query

---

## 📖 Level 4: SHARINGAN Architecture (Week 4-6)

### 4.1 SHARINGAN Overview

**Read SHARINGAN's documentation:**

1. **README.md** - Project overview
2. **ARCHITECTURE.md** - System design
3. **sharingan/processor.py** - Main pipeline (read docstrings)

**Key Components:**

```python
# 1. Adaptive Sampler
# Selects important frames based on visual change
# High motion → 8 FPS, Low motion → 1 FPS

# 2. CLIP Encoder (ViT-B/32)
# Converts frames to 512-dim embeddings
# Same model used for text queries

# 3. Multi-Scale TAS
# 3 parallel temporal scales + GRU memory
# Short (2 frames) → Mid (8 frames) → Long (32 frames)

# 4. Event Detector
# Finds scene changes using embedding similarity
# Threshold: 0.5 (cosine distance)

# 5. Hierarchical Memory
# 3 levels: Frame → Event → Chapter
# Enables multi-granularity retrieval

# 6. Query Router
# Classifies queries: window/semantic/causal/summary
# Routes to appropriate retrieval strategy

# 7. Temporal Filters
# Fixes: teaser bias, temporal weighting, first-frame bias
# Improves accuracy from 40% → 100%
```

---

### 4.2 Deep Dive: Multi-Scale TAS

**File:** `sharingan/temporal/multi_scale_tas.py`

**Architecture:**
```python
class MultiScaleTAS:
    def __init__(self):
        # 3 parallel temporal attention shifts
        self.short_tas = TAS(kernel_size=2)   # Gestures
        self.mid_tas = TAS(kernel_size=8)     # Actions
        self.long_tas = TAS(kernel_size=32)   # Scenes
        
        # GRU for full-video memory
        self.gru = GRU(hidden_size=512)
        
        # Fusion layer
        self.fusion = Linear(512*4, 512)
    
    def forward(self, embeddings):
        # embeddings: [T, 512] where T = number of frames
        
        # Apply 3 scales in parallel
        short = self.short_tas(embeddings)  # [T, 512]
        mid = self.mid_tas(embeddings)      # [T, 512]
        long = self.long_tas(embeddings)    # [T, 512]
        
        # GRU for long-range memory
        memory, _ = self.gru(embeddings)    # [T, 512]
        
        # Concatenate and fuse
        combined = torch.cat([short, mid, long, memory], dim=-1)  # [T, 2048]
        output = self.fusion(combined)      # [T, 512]
        
        return output
```

**Why Multi-Scale?**
- Short scale (2 frames): Detects quick gestures (hand moving)
- Mid scale (8 frames): Detects actions (installing RAM)
- Long scale (32 frames): Detects scenes (entire build phase)
- GRU: Maintains context across entire video

**Example:**
```
Frame 100: Hand reaching for RAM
    ↓
Short TAS: "Hand moving" (2-frame context)
Mid TAS: "Picking up RAM" (8-frame context)
Long TAS: "RAM installation phase" (32-frame context)
GRU: "PC building tutorial, currently at RAM step"
    ↓
Fused understanding: "Installing RAM into motherboard"
```

---

### 4.3 Deep Dive: Temporal Filters

**File:** `sharingan/processor.py` → `_apply_temporal_filters()`

**Problem:** System confuses intro montages with actual actions

**Solution:** Apply 3 filters based on query keywords

```python
def _apply_temporal_filters(self, similarities, query, video_duration):
    # Filter 1: Teaser Bias (for "final" queries)
    if 'final' in query or 'end' in query:
        # Penalize first 60 seconds (intro montage)
        for i, timestamp in enumerate(self.timestamps):
            if timestamp < 60.0:
                similarities[i] *= 0.1  # 90% penalty
            elif timestamp > video_duration * 0.8:
                similarities[i] *= 1.5  # 50% boost
    
    # Filter 2: Temporal Weighting (for timing queries)
    if 'beginning' in query:
        # Boost early timestamps
        weight = 1.0 / (1.0 + timestamp / 60.0)
        similarities[i] *= weight
    
    elif 'end' in query:
        # Boost late timestamps
        weight = timestamp / video_duration
        similarities[i] *= weight
    
    # Filter 3: First-Frame Bias (for all queries)
    if timestamp < 2.0:
        similarities[i] *= 0.3  # 70% penalty
    
    return similarities
```

**Impact:**
- Before: "When is RAM installed?" → 2:16 (explanation) ❌
- After: "When is RAM installed?" → 15:03 (actual action) ✅

---

### 4.4 Deep Dive: Query Pipeline

**File:** `sharingan/chat/pipeline.py`

**Flow:**
```python
def query(self, query_text):
    # Step 1: Route query
    query_plan = self.router.route_query(query_text)
    # Determines: window/semantic/causal/summary
    
    # Step 2: Retrieve from memory
    context = self._retrieve_context(query_plan)
    # Searches hierarchical memory (frame/event/chapter)
    
    # Step 3: Apply temporal filters
    context = self._apply_temporal_filters(context, query_text)
    # Fixes teaser bias, temporal weighting, first-frame bias
    
    # Step 4: Build reasoning scaffold
    scaffold = self.scaffold_builder.build(query_plan, context)
    # Structures reasoning for small LLM
    
    # Step 5: Generate response
    response = self.llm.chat(query_text, scaffold)
    # Uses Qwen-0.5B with structured guidance
    
    return response
```

**Example Query Flow:**

```
User: "When is RAM installed?"
    ↓
Router: "action query" → semantic retrieval
    ↓
Memory: Search event-level for "RAM" + "install"
    ↓
Temporal Filter: Penalize first 60s, boost tutorial section
    ↓
Top Results: [15:03, 15:20, 15:45, 16:10, 16:30]
    ↓
Scaffold: "Temporal order: RAM installation sequence"
    ↓
LLM: "RAM is installed at 15:03. The process involves..."
```

---

## 🎓 Study Schedule

### Week 1: Foundations
- **Day 1-2:** Neural networks basics (3Blue1Brown videos)
- **Day 3-4:** CNNs (LeNet, AlexNet papers)
- **Day 5-7:** Transformers (Attention is All You Need)

### Week 2: Vision & Language
- **Day 1-3:** Vision Transformers (ViT paper)
- **Day 4-7:** CLIP (paper + experiments)

### Week 3: Temporal Understanding
- **Day 1-3:** LSTM/GRU (papers + tutorials)
- **Day 4-7:** Temporal Action Segmentation (TSN, TSM)

### Week 4: Advanced Topics
- **Day 1-3:** Optical Flow (Lucas-Kanade, Horn-Schunck)
- **Day 4-7:** VideoQA (NExT-QA, EgoSchema)

### Week 5-6: SHARINGAN Deep Dive
- **Day 1-3:** Read all SHARINGAN documentation
- **Day 4-7:** Study each component in detail
- **Day 8-10:** Trace code execution
- **Day 11-14:** Experiment with modifications

---

## 📝 Learning Checklist

### Level 1: Foundations ✓
- [ ] Understand neural network basics
- [ ] Understand CNNs (convolution, pooling)
- [ ] Understand transformers (attention mechanism)
- [ ] Can explain forward/backward propagation

### Level 2: Core Concepts ✓
- [ ] Understand Vision Transformers (ViT)
- [ ] Understand CLIP (vision-language models)
- [ ] Understand LSTM/GRU (recurrent networks)
- [ ] Can explain contrastive learning

### Level 3: Advanced Topics ✓
- [ ] Understand temporal action segmentation
- [ ] Understand optical flow
- [ ] Understand VideoQA benchmarks
- [ ] Can explain multi-scale temporal reasoning

### Level 4: SHARINGAN ✓
- [ ] Understand complete architecture
- [ ] Understand each component's role
- [ ] Can trace query execution
- [ ] Can explain design decisions
- [ ] Can propose improvements

---

## 🔗 Quick Reference Links

### Essential Papers (Must Read)
1. Attention is All You Need: https://arxiv.org/pdf/1706.03762.pdf
2. Vision Transformer (ViT): https://arxiv.org/pdf/2010.11929.pdf
3. CLIP: https://arxiv.org/pdf/2103.00020.pdf
4. LSTM: https://www.bioinf.jku.at/publications/older/2604.pdf
5. TSN: https://arxiv.org/pdf/1608.00859.pdf
6. TSM: https://arxiv.org/pdf/1811.08383.pdf

### Tutorials & Visualizations
1. 3Blue1Brown Neural Networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
2. Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
3. CS231n CNNs: https://cs231n.github.io/convolutional-networks/
4. Neural Network Playground: https://playground.tensorflow.org/

### SHARINGAN Documentation
1. README.md - Project overview
2. ARCHITECTURE.md - System design
3. ACCURACY_IMPROVEMENT_PLAN.md - Current issues & solutions
4. FIXES_VERIFICATION.md - Implemented improvements

---

## 💡 Tips for Success

1. **Don't rush** - Take time to understand each concept
2. **Code along** - Implement simple versions yourself
3. **Draw diagrams** - Visualize architectures
4. **Ask questions** - Use ChatGPT/Claude to clarify
5. **Experiment** - Modify SHARINGAN code and see what happens
6. **Connect concepts** - See how everything fits together

---

## 🎯 Final Goal

After completing this roadmap, you should be able to:

1. **Explain** every component of SHARINGAN's architecture
2. **Understand** why each design decision was made
3. **Trace** how a query flows through the system
4. **Identify** strengths and weaknesses
5. **Propose** meaningful improvements
6. **Implement** modifications to the codebase

---

*Created: 2026-02-28*  
*For: Complete beginners wanting deep understanding*  
*Time: 4-6 weeks of dedicated study*
