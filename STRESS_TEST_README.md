# SHARINGAN Complex Video Stress Test Suite

This test suite evaluates SHARINGAN's performance on 4 categories of complex videos that challenge different aspects of the system.

## Test Categories

### 1. Texture & State Change (Chocolate Sculpture)
**Video:** Amaury Guichon - The Dragon Chocolate Sculpture  
**URL:** https://www.youtube.com/watch?v=f2vO_G_pY_E

**Challenges:**
- Everything starts as "brown liquid" or "brown blocks" (chocolate)
- Must distinguish between: melting, carving, spraying, painting, gluing
- Specialized tools that look like hardware tools (lathes, spray guns)
- Fine texture details and state changes

**Key Queries:**
- "When is chocolate being poured?" vs "When is chocolate being painted?"
- "When is the spray gun used?" vs "When is the lathe used?"
- Tool identification in food context

---

### 2. Object Density & Text (PC Building)
**Video:** Linus Tech Tips - How to Build a PC (2024 Guide)  
**URL:** https://www.youtube.com/watch?v=PXaLc9AYIcg

**Challenges:**
- Hundreds of small, similar-looking components (screws, headers, RAM slots)
- High density of text on boxes and motherboards
- Brand identification (Intel, NVIDIA, ASUS)
- Similar actions: "installing CPU" vs "installing RAM"

**Key Queries:**
- "When is Intel shown?" (text detection)
- "When is the CPU installed?" vs "When is the RAM installed?"
- Brand and component identification

---

### 3. State Transformation (Chemistry)
**Video:** NileRed - Turning Plastic Gloves into Grape Soda  
**URL:** https://www.youtube.com/watch?v=zFZ5jQ0yuNA

**Challenges:**
- Long-form video (20+ minutes)
- Materials change color, viscosity, and form multiple times
- Solid → Liquid → Gas transformations
- Various types of glassware (beakers vs flasks)

**Key Queries:**
- "When does the liquid turn purple?" (exact frame among 30 minutes)
- Equipment detection (beakers vs flasks)
- State transformation tracking

---

### 4. Action Segmentation (Woodworking)
**Video:** Blacktail Studio - Making a $20,000 Epoxy Table  
**URL:** https://www.youtube.com/watch?v=1iG1sXaYhwY

**Challenges:**
- Video spans weeks of real-time work
- Many distinct phases: selection, building, pouring, sanding, finishing
- Repetitive actions that look similar (sanding vs buffing)
- Long-form process summarization

**Key Queries:**
- "Summarize the table building process" (20-step process)
- "When is sanding shown?" vs "When is buffing shown?"
- Phase detection and segmentation

---

## Running the Tests

### Prerequisites
```bash
pip install yt-dlp
```

### Run All Tests
```bash
python test_complex_videos.py
# Select option 5 or 'all'
```

### Run Individual Test
```bash
python test_complex_videos.py
# Select option 1-4 for specific test
```

---

## Test Configuration

### Preset Settings

**Test 1 & 2 (High Quality):**
- FPS: 8.0 (adaptive)
- Reason: Better texture and text detection
- Use case: Fine details, text overlays

**Test 3 & 4 (Balanced):**
- FPS: 5.0 (adaptive)
- Reason: Long videos (20+ minutes)
- Use case: Efficient processing for extended content

---

## Output Structure

```
stress_test_results/
├── 01_chocolate_sculpture/
│   └── 01_chocolate_sculpture_results.md
├── 02_pc_building/
│   └── 02_pc_building_results.md
├── 03_chemistry/
│   └── 03_chemistry_results.md
└── 04_woodworking/
    └── 04_woodworking_results.md
```

Each results file contains:
- Video information (duration, frames, FPS)
- Detected events with timestamps
- Query results for all test queries
- Test summary and statistics

---

## Expected Challenges

### Test 1: Texture & State Change
- **Challenge:** Distinguishing similar brown textures
- **Success Metric:** Can differentiate pouring vs painting chocolate
- **Risk:** May confuse similar tools (spray gun vs airbrush)

### Test 2: Object Density & Text
- **Challenge:** Text detection at various angles and sizes
- **Success Metric:** Can identify brands (Intel, NVIDIA, ASUS)
- **Risk:** May miss small text on components

### Test 3: State Transformation
- **Challenge:** Tracking color changes over 20+ minutes
- **Success Metric:** Can find exact frame when liquid turns purple
- **Risk:** May lose temporal context in long videos

### Test 4: Action Segmentation
- **Challenge:** Distinguishing repetitive similar actions
- **Success Metric:** Can differentiate sanding vs buffing
- **Risk:** May merge similar phases together

---

## Evaluation Criteria

### 1. Temporal Accuracy
- Do "beginning" queries return early timestamps?
- Do "end" queries return late timestamps?
- Do "middle" queries return center timestamps?

### 2. Action Differentiation
- Can system distinguish similar actions? (pouring vs painting)
- Can system identify specific tools? (spray gun vs lathe)

### 3. Text Detection
- Can system identify brands from text overlays?
- Can system read component labels?

### 4. State Change Tracking
- Can system track color changes?
- Can system identify phase transitions?

### 5. Process Summarization
- Can system summarize multi-step processes?
- Can system identify key phases?

---

## Known Limitations

### CLIP-Based Limitations
- CLIP treats frames independently (no temporal context)
- Text detection limited to visible, clear text
- Similar textures may be confused (brown chocolate)

### Temporal Reasoning
- TAS window size (96 frames) may not capture full video context
- Long videos (20+ min) may lose global narrative structure

### Action Segmentation
- Repetitive actions may be merged
- Similar tools may be confused

---

## Success Metrics

### Minimum Acceptable Performance
- 70% temporal accuracy (beginning/middle/end queries)
- 60% action differentiation (similar actions)
- 50% text detection (brand identification)
- 60% state change tracking (color/form changes)

### Target Performance
- 90% temporal accuracy
- 80% action differentiation
- 70% text detection
- 80% state change tracking

---

## Troubleshooting

### Video Download Fails
```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Try alternative format
# Edit test script: 'format': 'best'
```

### Out of Memory
```bash
# Reduce batch size
# Edit test script: batch_size=16

# Reduce FPS
# Edit test script: target_fps=3.0
```

### Slow Processing
```bash
# Use CPU if GPU is slow
# Edit test script: device='cpu'

# Reduce FPS for long videos
# Edit test script: target_fps=2.0
```

---

## Future Enhancements

### Planned Improvements
1. Add SmolVLM support for better descriptions
2. Implement phase detection for long videos
3. Add OCR for better text detection
4. Implement action segmentation model
5. Add contrastive learning for similar actions

### Additional Test Categories
- Sports videos (fast motion, multiple objects)
- Tutorial videos (step-by-step instructions)
- Documentary videos (narrative structure)
- Gaming videos (UI elements, rapid changes)

---

*Last Updated: 2026-02-28*
