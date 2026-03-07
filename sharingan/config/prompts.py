"""
Configurable prompts for SHARINGAN video understanding.

This module provides default prompts optimized for different use cases,
and allows users to customize prompts for their specific needs.
"""

# ============================================================================
# INTERNVLM CAPTION PROMPTS
# ============================================================================

# Default: Optimized for TemporalBench (ORDER, DIRECTION, STATE, HAND, COUNT)
TEMPORALBENCH_CAPTION_PROMPT = """Analyze this frame and answer EXACTLY:

1. HAND: Which hand holds the tool? (left/right/both/neither)
2. TOOL: What tool is visible? (screwdriver/wrench/knife/none)
3. ACTION: What is happening? (tightening/loosening/pushing/pulling/connecting/disconnecting)
4. DIRECTION: Which direction? (clockwise/counterclockwise/left-to-right/right-to-left/toward/away)
5. STATE: What is the current state?
   - Light: ON/OFF/not visible
   - Screw: tight/loose/not visible
   - Wire: connected/disconnected/not visible
6. COUNT: How many times has this action occurred? (first time/second time/third time)
7. EVENT: What JUST changed in this moment? (light turned ON, wire pushed onto connector, screw became tight)

Format: "HAND: right | TOOL: screwdriver | ACTION: tightening | DIRECTION: clockwise | STATE: light=ON, screw=tight | COUNT: second time | EVENT: screw became tight"

Be EXACT. Use structured format. Max 80 words."""

# General: Detailed scene description for general video understanding
GENERAL_CAPTION_PROMPT = """Describe this frame in detail:

1. SCENE: What is the overall setting? (kitchen, workshop, outdoors, etc.)
2. PEOPLE: How many people? What are they doing?
3. OBJECTS: What key objects are visible?
4. ACTIONS: What actions are being performed?
5. SPATIAL: Where are things located? (left/right/center/foreground/background)
6. MOTION: What is moving? How fast?

Be descriptive and specific. Max 100 words."""

# Cooking: Optimized for cooking videos
COOKING_CAPTION_PROMPT = """Describe this cooking step:

1. INGREDIENT: What ingredient is being used?
2. TOOL: What cooking tool/utensil? (knife/pan/spoon/etc)
3. ACTION: What cooking action? (chopping/stirring/pouring/mixing/grilling)
4. TECHNIQUE: What technique? (dicing/sautéing/boiling/etc)
5. STATE: What is the current state? (raw/cooked/golden/crispy)
6. HAND: Which hand? (left/right/both)

Be precise about cooking details. Max 80 words."""

# Sports: Optimized for sports videos
SPORTS_CAPTION_PROMPT = """Describe this sports moment:

1. SPORT: What sport is this?
2. PLAYER: Which player(s) are visible? (jersey number/position)
3. ACTION: What action is happening? (running/jumping/throwing/kicking)
4. BALL: Where is the ball? What's happening to it?
5. SCORE: Any score visible?
6. INTENSITY: How intense is the action? (slow/moderate/fast/explosive)

Be specific about player actions. Max 80 words."""

# Surveillance: Optimized for security/surveillance videos
SURVEILLANCE_CAPTION_PROMPT = """Describe this surveillance frame:

1. PEOPLE: How many people? Describe appearance (clothing color, height, etc)
2. LOCATION: Where in the frame? (entrance/exit/left/right/center)
3. ACTION: What are they doing? (entering/exiting/standing/walking/running)
4. DIRECTION: Which direction are they moving? (left-to-right/toward camera/away)
5. OBJECTS: Any objects being carried or interacted with?
6. TIME: Any visible time indicators?

Be factual and precise. Max 80 words."""

# Tutorial: Optimized for how-to/tutorial videos
TUTORIAL_CAPTION_PROMPT = """Describe this tutorial step:

1. STEP: What step number is this? (first/second/third/etc)
2. ACTION: What is being demonstrated?
3. TOOL: What tools are being used?
4. HAND: Which hand is doing what?
5. RESULT: What is the expected result of this step?
6. TIP: Any visible technique or tip?

Be instructional and clear. Max 80 words."""

# ============================================================================
# LLM SYSTEM PROMPTS
# ============================================================================

# Default: Temporal reasoning for TemporalBench
TEMPORALBENCH_SYSTEM_PROMPT = """You are a precise video temporal reasoning expert. 
Your task is to determine the CORRECT ORDER of events.

TEMPORAL REASONING PROTOCOL:
1. READ the EVENT SEQUENCE - events are listed in CHRONOLOGICAL ORDER (earliest to latest)
2. IDENTIFY the timestamps - earlier timestamp = happened FIRST
3. EXTRACT key attributes from each event:
   - HAND: Which hand? (left/right/both)
   - ACTION: What action? (tightening/loosening/pushing/pulling)
   - DIRECTION: Which way? (clockwise/counterclockwise/onto/off)
   - STATE: What state? (light ON/OFF, screw tight/loose, wire connected/disconnected)
   - COUNT: How many times? (first/second/third)
4. COMPARE both options against the timeline:
   - For 'THEN' questions: Check if Event A timestamp < Event B timestamp
   - For 'BEFORE/AFTER' questions: Compare timestamps directly
   - For 'FIRST/LAST' questions: Use Event 1 (earliest) or Event N (latest)
5. MATCH the sequence:
   - Does option A match the chronological order?
   - Does option B match the chronological order?
6. The ONLY difference is usually ORDER, DIRECTION, STATE, or HAND
7. Pay attention to: FIRST, THEN, FINALLY, BEFORE, AFTER markers

RESPOND WITH ONLY THE LETTER (A or B). NO EXPLANATION."""

# General: Conversational video Q&A
GENERAL_SYSTEM_PROMPT = """You are a helpful video analysis assistant. 
Answer questions about the video based on the provided context. 
Be concise and specific, referencing timestamps when relevant."""

# Cooking: Recipe and cooking Q&A
COOKING_SYSTEM_PROMPT = """You are a culinary expert analyzing cooking videos.
Answer questions about ingredients, techniques, and cooking steps.
Reference timestamps and be specific about cooking methods."""

# Sports: Sports analysis
SPORTS_SYSTEM_PROMPT = """You are a sports analyst reviewing game footage.
Answer questions about player actions, game events, and tactics.
Reference timestamps and player positions when relevant."""

# Surveillance: Security analysis
SURVEILLANCE_SYSTEM_PROMPT = """You are a security analyst reviewing surveillance footage.
Answer questions about people, actions, and events factually.
Reference timestamps and locations precisely."""

# Tutorial: How-to instruction
TUTORIAL_SYSTEM_PROMPT = """You are an instructional expert analyzing tutorial videos.
Answer questions about steps, techniques, and procedures.
Be clear and instructional, referencing step numbers and timestamps."""

# ============================================================================
# PROMPT PRESETS
# ============================================================================

PROMPT_PRESETS = {
    'temporalbench': {
        'caption_prompt': TEMPORALBENCH_CAPTION_PROMPT,
        'system_prompt': TEMPORALBENCH_SYSTEM_PROMPT,
        'description': 'Optimized for temporal reasoning benchmarks (ORDER, DIRECTION, STATE)'
    },
    'general': {
        'caption_prompt': GENERAL_CAPTION_PROMPT,
        'system_prompt': GENERAL_SYSTEM_PROMPT,
        'description': 'General-purpose video understanding'
    },
    'cooking': {
        'caption_prompt': COOKING_CAPTION_PROMPT,
        'system_prompt': COOKING_SYSTEM_PROMPT,
        'description': 'Optimized for cooking and recipe videos'
    },
    'sports': {
        'caption_prompt': SPORTS_CAPTION_PROMPT,
        'system_prompt': SPORTS_SYSTEM_PROMPT,
        'description': 'Optimized for sports and game footage'
    },
    'surveillance': {
        'caption_prompt': SURVEILLANCE_CAPTION_PROMPT,
        'system_prompt': SURVEILLANCE_SYSTEM_PROMPT,
        'description': 'Optimized for security and surveillance footage'
    },
    'tutorial': {
        'caption_prompt': TUTORIAL_CAPTION_PROMPT,
        'system_prompt': TUTORIAL_SYSTEM_PROMPT,
        'description': 'Optimized for how-to and tutorial videos'
    }
}

def get_preset(preset_name: str) -> dict:
    """
    Get a prompt preset by name.
    
    Args:
        preset_name: Name of preset ('temporalbench', 'general', 'cooking', etc.)
        
    Returns:
        Dictionary with 'caption_prompt' and 'system_prompt'
        
    Raises:
        ValueError: If preset not found
    """
    if preset_name not in PROMPT_PRESETS:
        available = ', '.join(PROMPT_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PROMPT_PRESETS[preset_name]

def list_presets() -> None:
    """Print all available prompt presets."""
    print("Available Prompt Presets:")
    print("=" * 70)
    for name, preset in PROMPT_PRESETS.items():
        print(f"\n{name}:")
        print(f"  {preset['description']}")
    print("\n" + "=" * 70)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Use TemporalBench preset (default)
>>> from sharingan.processor import VideoProcessor
>>> processor = VideoProcessor(vlm_model='siglip')
>>> # Uses TemporalBench prompts by default

Example 2: Use cooking preset
>>> from sharingan.config.prompts import get_preset
>>> cooking_preset = get_preset('cooking')
>>> processor = VideoProcessor(
...     vlm_model='siglip',
...     caption_prompt=cooking_preset['caption_prompt']
... )

Example 3: Custom prompt
>>> custom_prompt = "Describe what you see in 20 words or less."
>>> processor = VideoProcessor(
...     vlm_model='siglip',
...     caption_prompt=custom_prompt
... )

Example 4: List all presets
>>> from sharingan.config.prompts import list_presets
>>> list_presets()

Example 5: Change prompt after initialization
>>> processor = VideoProcessor(vlm_model='siglip')
>>> processor._internvl.set_caption_prompt(get_preset('sports')['caption_prompt'])
"""
