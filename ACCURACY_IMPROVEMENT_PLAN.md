# SHARINGAN Accuracy Improvement Plan

**Date:** 2026-02-28  
**Based on:** PC Building Stress Test Analysis

---

## 🔴 Critical Issues Identified

### Issue 1: Explanation vs Action Confusion
**Problem:** System confuses educational explanations with actual installation actions.

**Evidence:**
- Query: "When is RAM installed?"
  - **System returned:** 136s (2:16) - explanation of what RAM is
  - **Actual action:** 15:03 (903s) - physical RAM installation
- Query: "When is GPU installed?"
  - **System returned:** 1850s (30:50) - radiator/AIO installation section
  - **Actual action:** 53:07 (3187s) - GPU installation

**Root Cause:**
- CLIP embeddings match visual similarity (RAM being shown) not action context
- No distinction between "showing component" vs "installing component"
- Intro montages show completed build, confusing "final result" queries

**Impact:** High - Fundamentally misunderstands tutorial structure

---

### Issue 2: Intro Montage Bias
**Problem:** Intro cinematics showing completed build are confused with actual build steps.

**Evidence:**
- Query: "When is PC powered on?"
  - **System returned:** 2.1s - intro cinematic (correct by accident)
  - **Actual first boot:** 55:36 (3336s) - test boot
- Intro shows finished build with all components, causing false positives

**Root Cause:**
- Similar to cooking video "teaser bias"
- Intro shows final result, system thinks it's the actual installation
- No understanding of video structure (intro → tutorial → conclusion)

**Impact:** High - Misidentifies tutorial structure

---

### Issue 3: Visual Similarity Over Action Context
**Problem:** System matches visual appearance rather than action being performed.

**Evidence:**
- CPU installation query returns end of Threadripper section (14:50)
  - Misses main Intel/AMD sections (10:14-13:16)
- System finds "component visible" not "component being installed"

**Root Cause:**
- CLIP focuses on object presence, not action/motion
- No motion analysis to detect "hands installing" vs "component sitting"
- No temporal context about "before/during/after" installation

**Impact:** Medium-High - Reduces precision for action queries

---

## 💡 Proposed Solutions

### Solution 1: Action vs Explanation Classifier

**Approach:** Add action detection layer to distinguish showing vs doing.

**Implementation:**
```python
class ActionClassifier:
    """Classify frames as: explanation, action, transition, result"""
    
    def classify_frame(self, frame, prev_frames):
        features = {
            'hands_visible': detect_hands(frame),
            'motion_intensity': compute_motion(frame, prev_frames),
            'close_up': detect_close_up(frame),
            'text_overlay': detect_text(frame),
            'talking_head': detect_face(frame)
        }
        
        # Action: hands visible + high motion + close-up
        if features['hands_visible'] and features['motion_intensity'] > 0.5:
            return 'action'
        
        # Explanation: talking head + text overlay + low motion
        elif features['talking_head'] and features['text_overlay']:
            return 'explanation'
        
        # Result: static shot + no hands + low motion
        elif features['motion_intensity'] < 0.2:
            return 'result'
        
        return 'transition'
```

**Benefits:**
- Distinguishes "showing RAM" from "installing RAM"
- Filters out explanation sections for action queries
- Improves precision for "when is X installed?" queries

**Complexity:** Medium - requires hand detection and motion analysis

---

### Solution 2: Tutorial Structure Detection

**Approach:** Detect video structure phases (intro → tutorial → conclusion).

**Implementation:**
```python
class TutorialStructureDetector:
    """Detect tutorial video phases"""
    
    def detect_phases(self, video_duration, events):
        phases = {
            'intro': (0, 60),  # First minute - montage/overview
            'tutorial': (60, video_duration - 120),  # Main content
            'conclusion': (video_duration - 120, video_duration)  # Last 2 min
        }
        
        # Refine based on scene changes
        # High scene change rate in intro (montage)
        # Low scene change rate in tutorial (focused work)
        # Medium scene change rate in conclusion (recap)
        
        return phases
    
    def filter_by_phase(self, query, results, phases):
        """Filter results based on query intent and video phase"""
        
        # Action queries: prioritize tutorial phase
        if any(word in query for word in ['install', 'connect', 'mount', 'plug']):
            return [r for r in results if phases['tutorial'][0] < r['timestamp'] < phases['tutorial'][1]]
        
        # Overview queries: allow intro phase
        elif any(word in query for word in ['what', 'show', 'visible']):
            return results  # No filtering
        
        # Result queries: prioritize conclusion phase
        elif any(word in query for word in ['final', 'complete', 'finished']):
            return [r for r in results if r['timestamp'] > phases['tutorial'][1]]
        
        return results
```

**Benefits:**
- Filters out intro montages for action queries
- Prioritizes tutorial section for "when is X installed?"
- Handles conclusion/recap sections appropriately

**Complexity:** Low-Medium - uses existing scene change detection

---

### Solution 3: Motion-Based Action Detection

**Approach:** Use optical flow to detect actual installation actions.

**Implementation:**
```python
class MotionActionDetector:
    """Detect actions using motion analysis"""
    
    def detect_installation_action(self, frames, query):
        """Detect if frames show actual installation vs just showing component"""
        
        # Compute optical flow between consecutive frames
        flow = compute_optical_flow(frames)
        
        # Installation actions have:
        # 1. Localized motion (hands moving)
        # 2. Consistent direction (inserting component)
        # 3. Medium-high magnitude (deliberate movement)
        
        motion_features = {
            'magnitude': np.mean(np.abs(flow)),
            'localization': compute_spatial_variance(flow),
            'consistency': compute_temporal_consistency(flow)
        }
        
        # Score as installation action
        action_score = (
            motion_features['magnitude'] * 0.4 +
            motion_features['localization'] * 0.3 +
            motion_features['consistency'] * 0.3
        )
        
        return action_score > 0.6  # Threshold for "action"
```

**Benefits:**
- Distinguishes static shots from active installation
- Detects hands-on work vs explanations
- Improves "when is X installed?" accuracy

**Complexity:** Medium - requires optical flow computation

---

### Solution 4: Query Intent Refinement

**Approach:** Better understand query intent and adjust retrieval strategy.

**Implementation:**
```python
class QueryIntentClassifier:
    """Classify query intent for better retrieval"""
    
    def classify_intent(self, query):
        intents = {
            'action': ['install', 'connect', 'mount', 'plug', 'insert', 'attach'],
            'explanation': ['what', 'why', 'how', 'explain'],
            'identification': ['show', 'visible', 'appear', 'display'],
            'result': ['final', 'complete', 'finished', 'done'],
            'timing': ['when', 'start', 'begin', 'first', 'last']
        }
        
        query_lower = query.lower()
        detected_intents = []
        
        for intent, keywords in intents.items():
            if any(kw in query_lower for kw in keywords):
                detected_intents.append(intent)
        
        return detected_intents
    
    def adjust_retrieval(self, query, results, intent):
        """Adjust retrieval based on intent"""
        
        if 'action' in intent:
            # For action queries, boost frames with high motion
            results = boost_high_motion_frames(results)
            # Filter out intro/outro
            results = filter_tutorial_section(results)
        
        elif 'identification' in intent:
            # For identification, allow all sections
            pass  # No filtering
        
        elif 'result' in intent:
            # For result queries, boost end sections
            results = boost_end_sections(results)
        
        return results
```

**Benefits:**
- Tailors retrieval to query intent
- Improves precision for action queries
- Reduces false positives from intro/outro

**Complexity:** Low - extends existing query routing

---

## 🎯 Implementation Priority

### Phase 1: Quick Wins (Implement First)
**Estimated Time:** 2-3 hours

1. **Tutorial Structure Detection** (Low complexity, high impact)
   - Detect intro/tutorial/conclusion phases
   - Filter action queries to tutorial section
   - Extend existing temporal filtering

2. **Query Intent Refinement** (Low complexity, medium impact)
   - Classify query intent (action vs identification)
   - Adjust retrieval strategy per intent
   - Extend existing query router

**Expected Improvement:**
- Action query accuracy: 40% → 70%
- Reduces intro montage false positives by 80%

---

### Phase 2: Motion Analysis (Implement Second)
**Estimated Time:** 4-6 hours

3. **Motion-Based Action Detection** (Medium complexity, high impact)
   - Compute optical flow for sampled frames
   - Detect high-motion regions (hands working)
   - Boost action frames in retrieval

**Expected Improvement:**
- Action query accuracy: 70% → 85%
- Distinguishes "showing" from "installing"

---

### Phase 3: Advanced Classification (Future Work)
**Estimated Time:** 8-12 hours

4. **Action vs Explanation Classifier** (Medium complexity, high impact)
   - Hand detection using pose estimation
   - Face detection for talking head sections
   - Text overlay detection for explanations

**Expected Improvement:**
- Action query accuracy: 85% → 95%
- Full tutorial structure understanding

---

## 📊 Expected Results After Improvements

### Before Improvements (Current)
| Query | Current Result | Actual | Accuracy |
|-------|---------------|--------|----------|
| "When is CPU installed?" | 14:50 (end of section) | 10:14 (start) | ❌ Off by 4:36 |
| "When is RAM installed?" | 2:16 (explanation) | 15:03 (action) | ❌ Off by 12:47 |
| "When is GPU installed?" | 30:50 (wrong section) | 53:07 (action) | ❌ Off by 22:17 |

**Average Error:** 13 minutes off target

---

### After Phase 1 (Structure + Intent)
| Query | Expected Result | Actual | Accuracy |
|-------|----------------|--------|----------|
| "When is CPU installed?" | 11:30 (tutorial section) | 10:14 | ✅ Within 1:16 |
| "When is RAM installed?" | 16:20 (tutorial section) | 15:03 | ✅ Within 1:17 |
| "When is GPU installed?" | 52:00 (tutorial section) | 53:07 | ✅ Within 1:07 |

**Average Error:** <2 minutes off target (85% improvement)

---

### After Phase 2 (+ Motion Analysis)
| Query | Expected Result | Actual | Accuracy |
|-------|----------------|--------|----------|
| "When is CPU installed?" | 10:30 (high motion) | 10:14 | ✅ Within 0:16 |
| "When is RAM installed?" | 15:20 (high motion) | 15:03 | ✅ Within 0:17 |
| "When is GPU installed?" | 53:30 (high motion) | 53:07 | ✅ Within 0:23 |

**Average Error:** <30 seconds off target (98% improvement)

---

## 🔧 Implementation Details

### File Changes Required

**Phase 1:**
1. `sharingan/processor.py` - Add structure detection
2. `sharingan/query/router.py` - Add intent classification
3. `sharingan/processor.py` - Extend temporal filtering

**Phase 2:**
4. `sharingan/video/motion.py` - New file for optical flow
5. `sharingan/processor.py` - Integrate motion scores

**Phase 3:**
6. `sharingan/video/action_classifier.py` - New file for action detection
7. `sharingan/processor.py` - Integrate action classification

---

## 🧪 Testing Strategy

### Test Videos
1. **PC Building** (current) - Tutorial with intro montage
2. **Cooking** (current) - Tutorial with teaser
3. **Woodworking** - Long-form tutorial
4. **Chemistry** - State transformation tutorial

### Success Metrics
- **Action Query Accuracy:** >85% within 1 minute of actual action
- **Intro Filtering:** <10% false positives from intro/outro
- **Motion Detection:** >80% precision for "installing" vs "showing"

### Validation Queries
```python
test_queries = [
    # Action queries (should return tutorial section)
    "When is CPU installed?",
    "When is RAM installed?",
    "When is GPU installed?",
    "When are cables connected?",
    
    # Identification queries (can return any section)
    "What components are shown?",
    "What brands are visible?",
    
    # Result queries (should return end section)
    "When is PC powered on?",
    "When is the final build shown?",
]
```

---

## 📝 Implementation Checklist

### Phase 1: Structure + Intent (Quick Wins)
- [ ] Implement `TutorialStructureDetector` class
- [ ] Add phase detection to `VideoProcessor.process()`
- [ ] Implement `QueryIntentClassifier` class
- [ ] Extend `_apply_temporal_filters()` with phase filtering
- [ ] Test on PC Building video
- [ ] Verify action query accuracy improvement
- [ ] Document results

### Phase 2: Motion Analysis
- [ ] Create `sharingan/video/motion.py` module
- [ ] Implement optical flow computation
- [ ] Add motion scoring to frame processing
- [ ] Integrate motion scores in retrieval
- [ ] Test on PC Building video
- [ ] Verify motion detection accuracy
- [ ] Document results

### Phase 3: Action Classification
- [ ] Create `sharingan/video/action_classifier.py` module
- [ ] Implement hand detection
- [ ] Implement face detection
- [ ] Implement text overlay detection
- [ ] Integrate action classification
- [ ] Test on all tutorial videos
- [ ] Verify final accuracy metrics
- [ ] Document results

---

## 🚀 Next Steps

**Immediate Actions:**
1. Review and approve this plan
2. Implement Phase 1 (Structure + Intent)
3. Re-test PC Building video
4. Compare before/after results
5. Proceed to Phase 2 if Phase 1 shows improvement

**Success Criteria for Phase 1:**
- Action queries return timestamps within 2 minutes of actual action
- Intro montage false positives reduced by >80%
- Ready to proceed to Phase 2

---

*Plan created: 2026-02-28*  
*Status: Awaiting approval to proceed*
