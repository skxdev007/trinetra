# Action Grouping & Temporal Graph Proposal

## Your Questions

1. **Instead of processing individual frames, what about grouping them by actions (start/stop)?**
2. **Is the tree being sent to LLM actually a tree or just text?**

## Answers

### Q1: Action Grouping - Already Partially Implemented!

You already have the infrastructure for this, it's just not being used effectively:

#### What Exists Now:

1. **EventDetector** (`sharingan/events/detector.py`)
   - Detects events from embedding changes
   - Groups frames into events with start/end timestamps
   - Currently runs but results aren't sent to LLM

2. **TemporalEventGraph** (`sharingan/graph/event_graph.py`)
   - Full graph structure with nodes (events) and edges (causal/temporal)
   - Supports causal chain discovery
   - Can find "A caused B" relationships
   - Currently only used in advanced VideoQueryPipeline, NOT in VideoProcessor

3. **Current Flow:**
   ```
   Video → Individual Frames → Embeddings → Retrieval → LLM
                                    ↓
                            EventDetector (runs but ignored)
   ```

#### What You Should Do:

**Option A: Use Event-Level Context (Recommended)**
```
Video → Frames → Embeddings → EventDetector → Event Groups → LLM
```

Instead of sending 10 individual frames to LLM:
```
STEP 1 [0:26]: Person tightening screw
STEP 2 [0:30]: Person tightening screw  
STEP 3 [0:35]: Person tightening screw
STEP 4 [1:38]: Person pulls string
```

Send event groups:
```
EVENT 1 [0:26-0:35]: Person TIGHTENS screw (3 frames, 9 seconds)
         ↓
EVENT 2 [1:38-1:42]: Person PULLS string (2 frames, 4 seconds)
         ↓
EVENT 3 [1:43-1:45]: Light turns ON (1 frame, 2 seconds)
```

**Benefits:**
- Reduces noise (10 similar frames → 1 event)
- Emphasizes action boundaries (when things START and STOP)
- Better temporal structure (events have duration)
- Easier for LLM to reason about ordering

**Implementation:**
```python
# In processor.py, modify _build_context()

def _build_event_context(self, query: str) -> str:
    """Build context from events instead of individual frames."""
    
    # 1. Retrieve relevant frames (existing)
    segments = self.query(query, top_k=10)
    
    # 2. Group frames into events
    events = self._group_frames_into_events(segments)
    
    # 3. Build event-level context
    context = []
    for i, event in enumerate(events, 1):
        start_time = event['start_timestamp']
        end_time = event['end_timestamp']
        duration = end_time - start_time
        action = event['action']  # Extracted from descriptions
        
        context.append(
            f"EVENT {i} [{start_time:.1f}s-{end_time:.1f}s]: "
            f"{action} (duration: {duration:.1f}s)"
        )
    
    return "\n".join(context)

def _group_frames_into_events(self, segments: List[Dict]) -> List[Dict]:
    """Group consecutive similar frames into events."""
    events = []
    current_event = None
    
    for seg in sorted(segments, key=lambda x: x['timestamp']):
        action = self._extract_key_action(seg['description'])
        
        if current_event is None:
            # Start new event
            current_event = {
                'start_timestamp': seg['timestamp'],
                'end_timestamp': seg['timestamp'],
                'action': action,
                'frames': [seg]
            }
        elif self._is_same_action(current_event['action'], action):
            # Continue current event
            current_event['end_timestamp'] = seg['timestamp']
            current_event['frames'].append(seg)
        else:
            # Action changed, save current event and start new one
            events.append(current_event)
            current_event = {
                'start_timestamp': seg['timestamp'],
                'end_timestamp': seg['timestamp'],
                'action': action,
                'frames': [seg]
            }
    
    if current_event:
        events.append(current_event)
    
    return events
```

**Option B: Use TemporalEventGraph (Advanced)**

Build a full causal graph during processing:
```python
# During video processing
graph = TemporalEventGraph()

for event in detected_events:
    graph.add_event(
        event_id=event.id,
        timestamp=event.timestamp,
        description=event.description,
        embedding=event.embedding,
        entities=event.entities,
        actions=event.actions
    )

# Add causal edges
for i in range(len(detected_events) - 1):
    if self._is_causal(detected_events[i], detected_events[i+1]):
        graph.add_edge(
            source_id=detected_events[i].id,
            target_id=detected_events[i+1].id,
            edge_type="causal",
            confidence=0.8
        )

# Query the graph
causal_chain = graph.find_causal_chain(
    start_event="tighten_screw",
    end_event="light_on"
)
```

Then send the causal chain to LLM:
```
CAUSAL CHAIN:
1. Person TIGHTENS screw [0:26-0:35]
   ↓ (enables)
2. Person PULLS string [1:38-1:42]
   ↓ (causes)
3. Light turns ON [1:43-1:45]
```

### Q2: Tree vs Text - It's Just Text

**Current Reality:**
The "tree" is just formatted text with arrows. No actual tree data structure is sent to the LLM.

```python
# What's actually sent (from llm.py line 227)
context = """
📹 VIDEO SEQUENCE (step-by-step):
======================================================================
STEP 1 [0:26]: Person TIGHTENS screw (light OFF)
         ↓
STEP 2 [0:30]: Person TIGHTENS screw (light OFF)
         ↓
STEP 3 [1:38]: Person PULLS string (light ON) ⚡ LIGHT TURNS ON
======================================================================

🎯 SEQUENCE SUMMARY:
  TIGHTEN → PULL STRING → LIGHT ON
"""
```

This is just a string with Unicode arrows. The LLM sees it as text, not as a structured tree.

**Why This Matters:**

LLMs are good at understanding structured text, but they're even better with:
1. **Explicit relationships** ("A causes B" not just "A ↓ B")
2. **Reduced redundancy** (3 "tightening" frames → 1 "tightening event")
3. **Clear boundaries** ("Event 1 starts at 0:26, ends at 0:35")

## Recommendation: Implement Option A First

**Why:**
1. Quick to implement (2-3 hours)
2. Uses existing EventDetector
3. Reduces context noise significantly
4. Expected impact: +10-15% accuracy

**Steps:**
1. Modify `_build_context()` in `llm.py` to group frames into events
2. Extract action boundaries (when action starts/stops)
3. Send event-level context instead of frame-level
4. Test on 30 questions

**Then:**
If that works, implement Option B (TemporalEventGraph) for causal reasoning.

## Example Comparison

### Current (Frame-Level):
```
STEP 1 [0:26]: Person tightening screw
STEP 2 [0:30]: Person tightening screw
STEP 3 [0:35]: Person tightening screw
STEP 4 [1:38]: Person pulls string
STEP 5 [1:43]: Light is on
```
**Problem:** Redundant, hard to see action boundaries

### Proposed (Event-Level):
```
EVENT 1 [0:26-0:35]: Person TIGHTENS screw (9s duration)
         ↓
EVENT 2 [1:38-1:42]: Person PULLS string (4s duration)
         ↓
EVENT 3 [1:43-1:45]: Light turns ON (2s duration)

SEQUENCE: TIGHTEN (9s) → PULL (4s) → LIGHT ON (2s)
```
**Better:** Clear boundaries, no redundancy, emphasizes transitions

## Code Location

Files to modify:
1. `sharingan/chat/llm.py` - `_build_context()` method (line 227)
2. `sharingan/processor.py` - Add `_group_frames_into_events()` method
3. `benchmarking/videomme/benchmark_long_video_coin.py` - No changes needed

Existing code to leverage:
1. `sharingan/events/detector.py` - EventDetector (already runs)
2. `sharingan/graph/event_graph.py` - TemporalEventGraph (for future)

## Expected Impact

**Conservative estimate:** +10-15% accuracy (53% → 63-68%)

**Why:**
- Reduces redundant frames (10 frames → 3 events)
- Emphasizes action boundaries (when things change)
- Clearer temporal structure for LLM
- Better matches how humans describe videos ("First he tightens, then he pulls")

---

**TL;DR:** You already have event detection, just not using it for LLM context. Group frames into events, send event-level context instead of frame-level. Quick win, big impact.
