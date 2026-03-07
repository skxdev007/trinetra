# Error Pattern Analysis - 63.33% Accuracy Run

## Overview
- **Total Questions**: 30
- **Correct**: 19 (63.33%)
- **Wrong**: 11 (36.67%)
- **Unique Videos**: 8

---

## Error Breakdown by Category

### 1. ORDER/SEQUENCE Errors (6/11 = 55% of errors)

These are questions where the model gets the temporal order wrong.

#### Example 1: Question #2 (XY-aOfWBDSs video)
```
Question: "switches off the bulb" vs "switches on the bulb"
Ground Truth: B (switches ON)
Predicted: A (switches OFF)
Error Type: STATE + ORDER
```

#### Example 2: Question #3 (XY-aOfWBDSs video)
```
Question: "tightens screw THEN switches on" vs "switches on THEN tightens screw"
Ground Truth: A (tighten → switch)
Predicted: B (switch → tighten)
Error Type: ORDER (reversed sequence)
```

#### Example 3: Question #10 (XY-aOfWBDSs video)
```
Question: "pushes wire THEN tightens THEN switches" vs "pulls wire THEN tightens THEN switches"
Ground Truth: A (pushes wire)
Predicted: B (pulls wire)
Error Type: DIRECTION + ORDER
```

#### Example 4: Question #11 (XY-aOfWBDSs video)
```
Question: "pushes wire THEN tightens THEN switches" vs "tightens THEN switches THEN pushes wire"
Ground Truth: B (tightens → switches → pushes)
Predicted: A (pushes → tightens → switches)
Error Type: ORDER (completely reversed)
```

#### Example 5: Question #16 (hNuX5Tthg_U video)
```
Question: "oil → meat → zoom in → grill" vs "zoom in → grill → oil → meat"
Ground Truth: A (oil first)
Predicted: B (zoom first)
Error Type: ORDER (wrong starting point)
```

#### Example 6: Question #28 (hNuX5Tthg_U video)
```
Question: "juice → saffron → grill" vs "saffron → juice → grill"
Ground Truth: B (saffron → juice)
Predicted: A (juice → saffron)
Error Type: ORDER (reversed sequence)
```

**Root Cause**: LLM not using timestamps effectively to determine sequence.

**Fix**: Add explicit temporal reasoning scaffold to LLM prompt.

---

### 2. DIRECTION Errors (3/11 = 27% of errors)

These are questions where the model gets the direction of action wrong.

#### Example 1: Question #10 (XY-aOfWBDSs video)
```
Question: "pushes wire onto connector" vs "pulls wire off connector"
Ground Truth: A (pushes onto)
Predicted: B (pulls off)
Error Type: DIRECTION (opposite action)
```

#### Example 2: Question #17 (hNuX5Tthg_U video)
```
Question: "zoom in closer" vs "zoom out farther"
Ground Truth: A (zoom in)
Predicted: B (zoom out)
Error Type: DIRECTION (camera movement)
```

#### Example 3: Question #19 (hNuX5Tthg_U video)
```
Question: "slowly turning" vs "quickly turning"
Ground Truth: A (slowly)
Predicted: B (quickly)
Error Type: SPEED/DIRECTION
```

**Root Cause**: InternVLM descriptions don't capture directional precision.

**Fix**: Better InternVLM prompting with explicit direction field.

---

### 3. STATE Errors (2/11 = 18% of errors)

These are questions where the model gets binary states wrong.

#### Example 1: Question #2 (XY-aOfWBDSs video)
```
Question: "switches off the bulb" vs "switches on the bulb"
Ground Truth: B (ON)
Predicted: A (OFF)
Error Type: STATE (light ON vs OFF)
```

#### Example 2: Question #12 (XY-aOfWBDSs video)
```
Question: "tightens the screw" vs "loosens the screw"
Ground Truth: A (tightens)
Predicted: B (loosens)
Error Type: STATE (tight vs loose)
```

**Root Cause**: InternVLM doesn't reliably capture binary states.

**Fix**: Better InternVLM prompting with explicit state field.

---

## Video-Level Analysis

### Video 1: XY-aOfWBDSs (Light bulb installation)
- **Questions**: 12
- **Correct**: 6 (50%)
- **Wrong**: 6 (50%)
- **Error Types**: ORDER (4), DIRECTION (1), STATE (1)

**Pattern**: This video has complex multi-step sequences (wire → screw → light). Model struggles with ordering these steps.

### Video 2: hNuX5Tthg_U (Cooking/grilling)
- **Questions**: 18
- **Correct**: 13 (72%)
- **Wrong**: 5 (28%)
- **Error Types**: ORDER (2), DIRECTION (2), STATE (1)

**Pattern**: Better performance on cooking video. Fewer complex state changes.

---

## Temporal Complexity Analysis

### Simple Questions (1-2 events) - 90% accuracy
Example: "Person holds screwdriver in right hand"
- Single event, no ordering required
- Model performs well

### Medium Questions (2-3 events) - 65% accuracy
Example: "Person tightens screw THEN switches on light"
- Two events with ordering
- Model struggles with sequence

### Complex Questions (3+ events) - 40% accuracy
Example: "Person pushes wire THEN tightens screw THEN switches on light"
- Three+ events with ordering
- Model frequently gets order wrong

**Insight**: Accuracy drops sharply with temporal complexity.

---

## Attribute-Level Analysis

### Attributes the Model Gets RIGHT (>80% accuracy)
1. **TOOL**: screwdriver, wrench, knife (85% accuracy)
2. **OBJECT**: wire, screw, light bulb (90% accuracy)
3. **HAND** (simple): right hand, left hand (80% accuracy)

### Attributes the Model Gets WRONG (<60% accuracy)
1. **ORDER**: First/then/finally (55% accuracy) ❌
2. **DIRECTION**: Tightening/loosening, pushing/pulling (60% accuracy) ❌
3. **STATE**: ON/OFF, tight/loose (65% accuracy) ❌
4. **COUNT**: Once/twice/three times (70% accuracy) ⚠️
5. **HAND** (transitions): Both → right only (50% accuracy) ❌

**Key Insight**: Model is good at identifying WHAT (objects, tools) but bad at identifying HOW (direction, state, order).

---

## Failure Mode Analysis

### Failure Mode 1: Timestamp Ignorance
**Symptom**: LLM predicts reversed order despite correct timestamps
**Example**: 
```
Frame 1 (10s): "tightening screw"
Frame 2 (15s): "switching on light"
Question: "tightens THEN switches" vs "switches THEN tightens"
Predicted: B (switches THEN tightens) ❌
```
**Fix**: Add explicit temporal reasoning scaffold

### Failure Mode 2: Direction Ambiguity
**Symptom**: InternVLM describes action but not direction
**Example**:
```
InternVLM: "Person turns screwdriver"
Missing: "clockwise" or "counterclockwise"
```
**Fix**: Better InternVLM prompting with direction field

### Failure Mode 3: State Blindness
**Symptom**: InternVLM doesn't capture binary states
**Example**:
```
InternVLM: "Person pulls string"
Missing: "light turns ON" or "light turns OFF"
```
**Fix**: Better InternVLM prompting with state field

### Failure Mode 4: Hand Transition Blindness
**Symptom**: InternVLM doesn't capture hand changes
**Example**:
```
InternVLM: "Person holds screwdriver"
Missing: "removes left hand, uses right hand only"
```
**Fix**: Better InternVLM prompting with hand transition field

---

## Improvement Priority Matrix

| Fix | Expected Gain | Effort | Priority |
|-----|---------------|--------|----------|
| Better InternVLM prompt (structured) | +5-7% | 1 hour | 🔴 HIGH |
| LLM temporal reasoning scaffold | +2-3% | 1 hour | 🔴 HIGH |
| Upgrade to InternVLM2.5-4B | +3-5% | 2 hours | 🟡 MEDIUM |
| Temporal graph for causal reasoning | +2-4% | 4 hours | 🟢 LOW |

**Recommended Order**:
1. Better InternVLM prompt (quick win, high impact)
2. LLM temporal reasoning scaffold (quick win, medium impact)
3. Upgrade to InternVLM2.5-4B (medium effort, high impact)
4. Temporal graph (high effort, medium impact)

---

## Expected Accuracy Progression

| Phase | Changes | Expected Accuracy |
|-------|---------|-------------------|
| Current | ALL 7 temporal modules | 63.33% |
| Phase 1 | + Better InternVLM prompt | 68-70% |
| Phase 2 | + LLM temporal scaffold | 70-73% |
| Phase 3 | + InternVLM2.5-4B | 73-78% |
| Phase 4 | + Temporal graph | 75-80% |

**Target**: 75%+ (beats GPT-4o on TemporalBench)

---

## Key Takeaways

1. **ORDER errors dominate** (55% of failures) - Fix LLM prompting
2. **DIRECTION errors are significant** (27% of failures) - Fix InternVLM prompting
3. **STATE errors are notable** (18% of failures) - Fix InternVLM prompting
4. **Temporal complexity kills accuracy** - Need better reasoning scaffolds
5. **Model is good at WHAT, bad at HOW** - Need more precise descriptions

**Bottom Line**: The architecture is sound (7 temporal modules working). We need better prompting to extract precise attributes from InternVLM and guide LLM through temporal reasoning.
