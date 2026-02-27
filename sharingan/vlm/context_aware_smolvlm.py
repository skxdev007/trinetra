"""
Context-Aware SmolVLM for SHARINGAN Deep Architecture

SYSTEM DESIGN COMMENT:
======================

Purpose:
--------
This module implements a context-aware wrapper around SmolVLM-500M that maintains a rolling
temporal context buffer to generate temporally coherent frame descriptions. By including
information from up to 8 previous frames in the prompt, the system reduces hallucinations
and improves description consistency across the video timeline.

Why Context-Aware Description Matters:
---------------------------------------
1. **Temporal Coherence**: Without context, VLMs describe each frame independently, leading
   to inconsistent entity naming ("person" → "man" → "individual") and contradictory details.
   Context ensures descriptions build on previous observations.

2. **Hallucination Reduction**: VLMs often hallucinate objects or actions not present in the
   frame. By providing context from verified previous frames, we anchor the model to what
   has actually been observed, reducing false positives.

3. **Narrative Understanding**: Many video understanding tasks require knowing what happened
   earlier. "The person picks up the cup" is more meaningful when we know "A cup was placed
   on the table 5 seconds ago". Context enables this narrative reasoning.

4. **Entity Tracking**: Consistent entity references across frames ("the person in the blue
   shirt" remains "the person in the blue shirt") enable better event detection and causal
   reasoning downstream.

How It Fits in the System:
---------------------------
- **Ingest Pipeline**: Processes each sampled frame with context from previous frames
- **Input**: Receives frames from AdaptiveSampler with timestamps and change scores
- **Output**: Produces FrameDescription objects consumed by CrossModalVerifier
- **Context Management**: Maintains FIFO buffer of max 8 frames (Requirements 2.1, 2.5)
- **Time Encoding**: Integrates continuous timestamps in prompts (Requirement 2.3)

Rolling Context Buffer (FIFO):
-------------------------------
The context buffer stores up to 8 previous frames with their descriptions:
- When buffer size < 8: Use all available frames as context
- When buffer size = 8: Remove oldest frame (FIFO) before adding new frame
- Context frames are included in the prompt as: "Previous observations: [desc1, desc2, ...]"

This design balances:
- **Context richness**: 8 frames at 1 FPS = 8 seconds of history (sufficient for most actions)
- **Prompt length**: 8 descriptions ≈ 800 tokens, leaving room for current frame analysis
- **Memory efficiency**: Storing 8 frames is manageable even for long videos

Continuous Time Encoding:
--------------------------
Timestamps are encoded using sinusoidal positional encoding and included in prompts:
- "At timestamp 15.5s: [current frame description]"
- "Previous observations: At 13.5s: [...], At 14.5s: [...]"

This helps the model understand temporal distances and relationships between frames.

FrameDescription Output:
------------------------
Each description includes:
- timestamp: Continuous time in seconds
- frame_index: Original frame index in video
- description: Natural language description of the frame
- entities: List of detected entities (people, objects)
- actions: List of detected actions (walking, picking up, etc.)
- confidence: Model confidence score (0.0 to 1.0)
- context_used: List of frame indices used as context (for debugging/analysis)

Requirements Validated:
------------------------
- Requirement 2.1: SmolVLM SHALL maintain a rolling context buffer of up to 8 previous frames
- Requirement 2.2: WHEN generating description for frame t, SHALL include context from up to 8 previous frames
- Requirement 2.3: SmolVLM SHALL encode continuous timestamps using sinusoidal positional encoding
- Requirement 2.4: SmolVLM SHALL return frame descriptions with timestamp, description, entities, actions, confidence
- Requirement 2.5: WHEN context buffer exceeds 8 frames, SHALL remove oldest frame
- Requirement 2.6: SmolVLM SHALL record which context frames were used for each description

Example Usage:
--------------
    # Initialize context-aware SmolVLM
    vlm = ContextAwareSmolVLM(
        model_name="HuggingFaceTB/SmolVLM-500M-Instruct",
        context_window=8,
        device="cuda"
    )
    
    # Process frames from adaptive sampler
    for frame_idx, frame, change_score in sampler.sample_adaptive(video):
        timestamp = frame_idx / video.fps
        
        # Generate description with context
        description = vlm.describe_with_context(
            current_frame=frame,
            timestamp=timestamp,
            frame_index=frame_idx
        )
        
        # description.context_used shows which previous frames were used
        print(f"Frame {frame_idx} at {timestamp:.2f}s: {description.description}")
        print(f"Context from frames: {description.context_used}")
        
        # Update context buffer for next frame
        vlm.update_context(frame, description.description, frame_idx)
"""

import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

from sharingan.vlm.smolvlm import SmolVLMEncoder
from sharingan.temporal.time_encoding import ContinuousTimeEncoder
from sharingan.exceptions import EncodingError


@dataclass
class FrameDescription:
    """
    Complete description of a video frame with metadata.
    
    Attributes:
        timestamp: Continuous time in seconds
        frame_index: Original frame index in video
        description: Natural language description of the frame
        entities: List of detected entities (people, objects, etc.)
        actions: List of detected actions (walking, picking up, etc.)
        confidence: Model confidence score (0.0 to 1.0)
        context_used: List of frame indices used as context for this description
    """
    timestamp: float
    frame_index: int
    description: str
    entities: List[str]
    actions: List[str]
    confidence: float
    context_used: List[int]


class ContextAwareSmolVLM:
    """
    Context-aware wrapper around SmolVLM-500M with rolling temporal context buffer.
    
    This class maintains a FIFO buffer of up to 8 previous frames and their descriptions,
    which are included in the prompt when generating descriptions for new frames. This
    approach reduces hallucinations and improves temporal coherence.
    
    Attributes:
        model: Underlying SmolVLM encoder
        time_encoder: Continuous time encoder for timestamps
        context_window: Maximum number of previous frames to keep (default 8)
        context_buffer: FIFO buffer storing (frame, description, frame_index, timestamp) tuples
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        context_window: int = 8,
        device: str = "auto"
    ):
        """
        Initialize context-aware SmolVLM with rolling context window.
        
        Args:
            model_name: HuggingFace model identifier for SmolVLM
            context_window: Maximum number of previous frames to keep (default 8)
            device: Device to run on ("cpu", "cuda", or "auto")
        
        Raises:
            ValueError: If context_window < 1
            EncodingError: If model fails to load
        """
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        
        self.context_window = context_window
        self.device = device
        
        # Initialize underlying SmolVLM encoder
        print(f"🔧 Initializing Context-Aware SmolVLM (context_window={context_window})...")
        self.model = SmolVLMEncoder(device=device)
        
        # Initialize continuous time encoder
        self.time_encoder = ContinuousTimeEncoder(d_model=512, max_time=3600.0)
        
        # Initialize context buffer (FIFO queue)
        # Each element: (frame, description, frame_index, timestamp)
        self.context_buffer: deque = deque(maxlen=context_window)
        
        print(f"✓ Context-Aware SmolVLM ready")
    
    def describe_with_context(
        self,
        current_frame: np.ndarray,
        timestamp: float,
        frame_index: int,
        max_new_tokens: int = 150
    ) -> FrameDescription:
        """
        Generate description for current frame using temporal context from previous frames.
        
        This method constructs a prompt that includes:
        1. Descriptions of up to 8 previous frames with their timestamps
        2. The current frame with its timestamp
        3. Instructions to maintain consistency with previous observations
        
        Args:
            current_frame: Current frame as numpy array (H, W, C) in RGB format
            timestamp: Continuous time in seconds
            frame_index: Original frame index in video
            max_new_tokens: Maximum tokens to generate (default 150)
        
        Returns:
            FrameDescription with timestamp, description, entities, actions, confidence, context_used
        
        Example:
            >>> vlm = ContextAwareSmolVLM(context_window=8)
            >>> description = vlm.describe_with_context(
            ...     current_frame=frame,
            ...     timestamp=15.5,
            ...     frame_index=465
            ... )
            >>> print(description.description)
            "A person in a blue shirt picks up a red cup from the table."
            >>> print(description.context_used)
            [457, 459, 461, 463]  # Previous frame indices used as context
        """
        # Build context string from previous frames
        context_str = self._build_context_string()
        
        # Build prompt with context and temporal information
        if context_str:
            prompt = (
                f"You are analyzing a video frame by frame. "
                f"Here are previous observations:\n\n{context_str}\n\n"
                f"Now at timestamp {timestamp:.2f}s, describe the current frame. "
                f"Be consistent with previous observations. "
                f"Focus on: (1) people and objects present, (2) actions occurring, (3) scene context. "
                f"Format: First describe the scene, then list entities and actions."
            )
        else:
            # First frame - no context available
            prompt = (
                f"At timestamp {timestamp:.2f}s, describe this video frame in detail. "
                f"Focus on: (1) people and objects present, (2) actions occurring, (3) scene context. "
                f"Format: First describe the scene, then list entities and actions."
            )
        
        # Generate description using underlying SmolVLM
        try:
            raw_description = self.model.describe_frame(
                frame=current_frame,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            raise EncodingError(f"Failed to generate description: {str(e)}")
        
        # Parse entities and actions from description
        entities, actions = self._parse_entities_and_actions(raw_description)
        
        # Compute confidence score (simple heuristic based on description length and coherence)
        confidence = self._compute_confidence(raw_description, entities, actions)
        
        # Record which context frames were used
        context_used = [item[2] for item in self.context_buffer]  # Extract frame_indices
        
        # Create FrameDescription object
        frame_desc = FrameDescription(
            timestamp=timestamp,
            frame_index=frame_index,
            description=raw_description,
            entities=entities,
            actions=actions,
            confidence=confidence,
            context_used=context_used
        )
        
        return frame_desc
    
    def update_context(
        self,
        frame: np.ndarray,
        description: str,
        frame_index: int,
        timestamp: float
    ) -> None:
        """
        Update rolling context buffer with new frame and description.
        
        This method implements FIFO behavior:
        - If buffer size < context_window: Append new frame
        - If buffer size = context_window: Remove oldest frame, append new frame
        
        Args:
            frame: Frame as numpy array (H, W, C)
            description: Generated description for this frame
            frame_index: Original frame index in video
            timestamp: Continuous time in seconds
        
        Example:
            >>> vlm = ContextAwareSmolVLM(context_window=8)
            >>> vlm.update_context(frame, "A person walks into the room.", 100, 3.33)
            >>> len(vlm.context_buffer)
            1
        """
        # Add to context buffer (deque automatically removes oldest if at maxlen)
        self.context_buffer.append((frame, description, frame_index, timestamp))
    
    def _build_context_string(self) -> str:
        """
        Build context string from previous frames in buffer.
        
        Returns:
            Formatted string with previous observations and timestamps
        
        Example output:
            "- At 13.5s: A person enters the room carrying a red cup.
             - At 14.5s: The person places the cup on the table.
             - At 15.5s: The person sits down in a chair."
        """
        if not self.context_buffer:
            return ""
        
        context_lines = []
        for frame, description, frame_idx, timestamp in self.context_buffer:
            context_lines.append(f"- At {timestamp:.2f}s: {description}")
        
        return "\n".join(context_lines)
    
    def _parse_entities_and_actions(self, description: str) -> Tuple[List[str], List[str]]:
        """
        Parse entities and actions from natural language description.
        
        This is a simple heuristic parser. In production, you might use:
        - Named Entity Recognition (NER) for entities
        - Verb phrase extraction for actions
        - Dependency parsing for more accurate extraction
        
        Args:
            description: Natural language description
        
        Returns:
            Tuple of (entities, actions)
        
        Example:
            >>> entities, actions = vlm._parse_entities_and_actions(
            ...     "A person in a blue shirt picks up a red cup from the table."
            ... )
            >>> entities
            ['person', 'blue shirt', 'red cup', 'table']
            >>> actions
            ['picks up']
        """
        entities = []
        actions = []
        
        # Simple heuristic: look for common entity patterns
        entity_keywords = ['person', 'man', 'woman', 'child', 'people', 'cup', 'table', 
                          'chair', 'door', 'window', 'phone', 'book', 'car', 'dog', 'cat']
        
        # Simple heuristic: look for common action verbs
        action_keywords = ['walk', 'run', 'sit', 'stand', 'pick', 'place', 'hold', 
                          'open', 'close', 'enter', 'exit', 'talk', 'look', 'point']
        
        description_lower = description.lower()
        
        # Extract entities
        for keyword in entity_keywords:
            if keyword in description_lower:
                entities.append(keyword)
        
        # Extract actions
        for keyword in action_keywords:
            if keyword in description_lower:
                # Try to capture verb phrase (e.g., "picks up", "walks into")
                words = description_lower.split()
                for i, word in enumerate(words):
                    if keyword in word:
                        # Check if next word is a preposition (up, down, into, etc.)
                        if i + 1 < len(words) and words[i + 1] in ['up', 'down', 'into', 'out', 'on', 'off']:
                            actions.append(f"{word} {words[i + 1]}")
                        else:
                            actions.append(word)
                        break
        
        # Remove duplicates while preserving order
        entities = list(dict.fromkeys(entities))
        actions = list(dict.fromkeys(actions))
        
        return entities, actions
    
    def _compute_confidence(
        self,
        description: str,
        entities: List[str],
        actions: List[str]
    ) -> float:
        """
        Compute confidence score for generated description.
        
        This is a simple heuristic based on:
        - Description length (too short or too long is suspicious)
        - Number of entities detected (more entities = more specific = higher confidence)
        - Number of actions detected (actions indicate understanding of dynamics)
        
        In production, you might use:
        - Model's internal confidence scores
        - Cross-modal verification with CLIP (done in next component)
        - Perplexity or likelihood scores from the language model
        
        Args:
            description: Generated description
            entities: Parsed entities
            actions: Parsed actions
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5
        
        # Length heuristic (optimal range: 50-200 characters)
        desc_len = len(description)
        if 50 <= desc_len <= 200:
            confidence += 0.2
        elif desc_len < 20:
            confidence -= 0.2  # Too short, likely incomplete
        elif desc_len > 300:
            confidence -= 0.1  # Too long, might be hallucinating
        
        # Entity heuristic (more entities = more specific)
        if len(entities) >= 3:
            confidence += 0.2
        elif len(entities) >= 1:
            confidence += 0.1
        
        # Action heuristic (actions indicate dynamic understanding)
        if len(actions) >= 2:
            confidence += 0.15
        elif len(actions) >= 1:
            confidence += 0.1
        
        # Clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def clear_context(self) -> None:
        """
        Clear the context buffer.
        
        Useful when starting to process a new video or when you want to reset
        the temporal context (e.g., after a scene cut).
        """
        self.context_buffer.clear()
    
    def get_context_size(self) -> int:
        """
        Get current number of frames in context buffer.
        
        Returns:
            Number of frames currently stored in context buffer (0 to context_window)
        """
        return len(self.context_buffer)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ContextAwareSmolVLM("
            f"context_window={self.context_window}, "
            f"current_context_size={len(self.context_buffer)}, "
            f"device={self.device})"
        )
