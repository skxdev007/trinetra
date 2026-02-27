"""
Cross-Modal Verification System for SHARINGAN Deep Architecture

SYSTEM DESIGN COMMENT:
======================

Purpose:
--------
This module implements a cross-modal verification system that detects VLM hallucinations
by comparing frame descriptions against visual evidence using CLIP (Contrastive Language-Image
Pre-training). By computing similarity between visual and textual representations, we can
flag low-confidence descriptions and provide correction suggestions.

Why Cross-Modal Verification Matters:
--------------------------------------
1. **Hallucination Detection**: Vision-Language Models (VLMs) often hallucinate objects,
   actions, or attributes that are not present in the frame. CLIP provides an independent
   verification mechanism by checking if the description aligns with visual evidence.

2. **Quality Assurance**: Not all frame descriptions are equally reliable. By flagging
   low-confidence descriptions, we prevent downstream reasoning from being based on
   incorrect information.

3. **Entity-Level Verification**: Beyond verifying the overall description, we verify
   each mentioned entity individually. This catches cases where the overall scene is
   correct but specific entities are hallucinated.

4. **Correction Suggestions**: When verification fails, we provide actionable feedback
   that can be used to retry description generation with more conservative prompts or
   to flag the frame for manual review.

How It Fits in the System:
---------------------------
- **Ingest Pipeline**: Receives FrameDescription from ContextAwareSmolVLM
- **Input**: Frame image + description text + entity list
- **Output**: VerificationResult with is_verified flag, alignment score, flagged entities
- **Downstream**: Verified descriptions flow to TemporalEventGraph and HierarchicalMemory
- **Error Handling**: Flagged descriptions trigger warnings and correction suggestions

CLIP-Based Verification:
-------------------------
CLIP (ViT-B/32) is a vision-language model trained on 400M image-text pairs to learn
aligned representations. Key properties:

1. **Cross-Modal Alignment**: CLIP embeddings for semantically similar images and texts
   are close in the shared embedding space (high cosine similarity).

2. **Zero-Shot Capability**: CLIP can verify descriptions without task-specific training,
   making it robust to diverse video content.

3. **Efficient**: ViT-B/32 is lightweight (~400MB) and fast (~50ms per frame on GPU).

Verification Thresholds:
-------------------------
Based on empirical analysis of CLIP similarity distributions:

- **Description Threshold (0.7)**: Descriptions with CLIP similarity < 0.7 are flagged
  as unverified. This threshold balances false positives (flagging correct descriptions)
  and false negatives (missing hallucinations).

- **Entity Threshold (0.5)**: Individual entities with CLIP similarity < 0.5 are flagged.
  Lower threshold because entity-level verification is more granular and prone to noise.

Verification Algorithm:
-----------------------
1. Encode frame with CLIP vision encoder → frame_embedding (512-dim)
2. Encode description with CLIP text encoder → text_embedding (512-dim)
3. Compute cosine similarity: alignment_score = cos(frame_embedding, text_embedding)
4. If alignment_score < 0.7: Flag description as unverified
5. For each entity in entities list:
   a. Encode entity text with CLIP text encoder → entity_embedding
   b. Compute cosine similarity: entity_score = cos(frame_embedding, entity_embedding)
   c. If entity_score < 0.5: Add entity to flagged_entities list
6. Generate correction suggestion based on failure mode

Correction Suggestions:
-----------------------
When verification fails, we provide specific suggestions:

- **Low alignment score**: "Description may not match visual content. Consider regenerating
  with more conservative prompt or using CLIP-only embeddings."

- **Flagged entities**: "Entities may be hallucinated: [entity1, entity2]. Verify these
  entities are actually present in the frame."

- **Both issues**: Combine both suggestions with priority on entity-level issues.

VerificationResult Output:
---------------------------
Each verification produces:
- is_verified: Boolean flag (True if alignment_score >= 0.7 AND no flagged entities)
- alignment_score: Float in [0.0, 1.0] (cosine similarity between frame and description)
- flagged_entities: List of entity strings with similarity < 0.5
- correction_suggestion: Optional string with actionable feedback

Requirements Validated:
------------------------
- Requirement 3.1: WHEN frame description generated, SHALL compute CLIP similarity
- Requirement 3.2: IF CLIP similarity < 0.7, SHALL flag description as unverified
- Requirement 3.3: SHALL verify each entity mentioned in description against frame
- Requirement 3.4: IF entity CLIP similarity < 0.5, SHALL add to flagged entities list
- Requirement 3.5: WHEN description flagged, SHALL provide correction suggestion
- Requirement 3.6: SHALL return verification results with is_verified, alignment_score,
  flagged_entities, correction_suggestion

Example Usage:
--------------
    # Initialize verifier
    verifier = CrossModalVerifier(
        clip_model="openai/clip-vit-base-patch32",
        threshold=0.7,
        entity_threshold=0.5,
        device="cuda"
    )
    
    # Verify frame description
    verification = verifier.verify_description(
        frame=frame_image,
        description="A person in a blue shirt picks up a red cup.",
        entities=["person", "blue shirt", "red cup"]
    )
    
    if not verification.is_verified:
        print(f"⚠️  Low confidence: {verification.alignment_score:.3f}")
        print(f"Flagged entities: {verification.flagged_entities}")
        print(f"Suggestion: {verification.correction_suggestion}")
    else:
        print(f"✓ Verified (score: {verification.alignment_score:.3f})")
"""

import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
from transformers import CLIPProcessor, CLIPModel

from sharingan.exceptions import EncodingError


@dataclass
class VerificationResult:
    """
    Result of cross-modal verification for a frame description.
    
    Attributes:
        is_verified: True if description passes verification (alignment >= threshold AND no flagged entities)
        alignment_score: Cosine similarity between frame and description embeddings (0.0 to 1.0)
        flagged_entities: List of entities with similarity below entity_threshold
        correction_suggestion: Optional suggestion for improving description if verification fails
    """
    is_verified: bool
    alignment_score: float
    flagged_entities: List[str]
    correction_suggestion: Optional[str]


class CrossModalVerifier:
    """
    Cross-modal verification system using CLIP to detect VLM hallucinations.
    
    This class uses CLIP (Contrastive Language-Image Pre-training) to verify that
    frame descriptions align with visual evidence. Descriptions with low CLIP similarity
    are flagged as potentially hallucinated.
    
    Attributes:
        model: CLIP model for encoding images and text
        processor: CLIP processor for preprocessing inputs
        threshold: Minimum alignment score for description verification (default 0.7)
        entity_threshold: Minimum alignment score for entity verification (default 0.5)
        device: Device to run on ("cpu", "cuda", or "auto")
    """
    
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        threshold: float = 0.7,
        entity_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize cross-modal verifier with CLIP model.
        
        Args:
            clip_model: HuggingFace model identifier for CLIP (default: ViT-B/32)
            threshold: Minimum alignment score for description verification (default 0.7)
            entity_threshold: Minimum alignment score for entity verification (default 0.5)
            device: Device to run on ("cpu", "cuda", or "auto")
        
        Raises:
            ValueError: If threshold or entity_threshold not in [0.0, 1.0]
            EncodingError: If CLIP model fails to load
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        if not (0.0 <= entity_threshold <= 1.0):
            raise ValueError(f"entity_threshold must be in [0.0, 1.0], got {entity_threshold}")
        
        self.threshold = threshold
        self.entity_threshold = entity_threshold
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load CLIP model and processor
        print(f"🔧 Initializing Cross-Modal Verifier with CLIP ({clip_model})...")
        try:
            self.processor = CLIPProcessor.from_pretrained(clip_model)
            self.model = CLIPModel.from_pretrained(clip_model).to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise EncodingError(f"Failed to load CLIP model: {str(e)}")
        
        print(f"✓ Cross-Modal Verifier ready (threshold={threshold}, entity_threshold={entity_threshold})")
    
    def verify_description(
        self,
        frame: np.ndarray,
        description: str,
        entities: List[str]
    ) -> VerificationResult:
        """
        Verify frame description against visual evidence using CLIP.
        
        This method:
        1. Encodes frame with CLIP vision encoder
        2. Encodes description with CLIP text encoder
        3. Computes cosine similarity (alignment score)
        4. Flags description if alignment < threshold
        5. Verifies each entity individually
        6. Generates correction suggestion if verification fails
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
            description: Natural language description to verify
            entities: List of entity strings mentioned in description
        
        Returns:
            VerificationResult with is_verified, alignment_score, flagged_entities, correction_suggestion
        
        Example:
            >>> verifier = CrossModalVerifier(threshold=0.7)
            >>> result = verifier.verify_description(
            ...     frame=frame_image,
            ...     description="A person picks up a red cup.",
            ...     entities=["person", "red cup"]
            ... )
            >>> if not result.is_verified:
            ...     print(f"Warning: {result.correction_suggestion}")
        """
        # Encode frame with CLIP vision encoder
        frame_embedding = self._encode_image(frame)
        
        # Encode description with CLIP text encoder
        text_embedding = self._encode_text(description)
        
        # Compute alignment score (cosine similarity)
        alignment_score = self.compute_alignment_score(frame_embedding, text_embedding)
        
        # Verify individual entities
        flagged_entities = []
        for entity in entities:
            entity_embedding = self._encode_text(entity)
            entity_score = self.compute_alignment_score(frame_embedding, entity_embedding)
            
            if entity_score < self.entity_threshold:
                flagged_entities.append(entity)
        
        # Determine verification status
        is_verified = (alignment_score >= self.threshold) and (len(flagged_entities) == 0)
        
        # Generate correction suggestion if verification fails
        correction_suggestion = None
        if not is_verified:
            correction_suggestion = self._generate_correction_suggestion(
                alignment_score, flagged_entities
            )
        
        return VerificationResult(
            is_verified=is_verified,
            alignment_score=alignment_score,
            flagged_entities=flagged_entities,
            correction_suggestion=correction_suggestion
        )
    
    def compute_alignment_score(
        self,
        frame_embedding: np.ndarray,
        text_embedding: np.ndarray
    ) -> float:
        """
        Compute cross-modal alignment score (cosine similarity).
        
        Args:
            frame_embedding: Frame embedding from CLIP vision encoder (512-dim)
            text_embedding: Text embedding from CLIP text encoder (512-dim)
        
        Returns:
            Cosine similarity between embeddings (0.0 to 1.0)
        
        Example:
            >>> frame_emb = verifier._encode_image(frame)
            >>> text_emb = verifier._encode_text("A person walks.")
            >>> score = verifier.compute_alignment_score(frame_emb, text_emb)
            >>> print(f"Alignment: {score:.3f}")
        """
        # Normalize embeddings
        frame_norm = frame_embedding / (np.linalg.norm(frame_embedding) + 1e-8)
        text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(frame_norm, text_norm)
        
        # Clamp to [0.0, 1.0] (cosine similarity can be negative, but CLIP embeddings are typically positive)
        similarity = float(np.clip(similarity, 0.0, 1.0))
        
        return similarity
    
    def _encode_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode frame with CLIP vision encoder.
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB format
        
        Returns:
            Frame embedding as numpy array (512-dim)
        
        Raises:
            EncodingError: If encoding fails
        """
        try:
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            
            # Preprocess and encode
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = image_features.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            raise EncodingError(f"Failed to encode image: {str(e)}")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text with CLIP text encoder.
        
        Args:
            text: Text string to encode
        
        Returns:
            Text embedding as numpy array (512-dim)
        
        Raises:
            EncodingError: If encoding fails
        """
        try:
            # Preprocess and encode
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Convert to numpy and normalize
            embedding = text_features.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            raise EncodingError(f"Failed to encode text: {str(e)}")
    
    def _generate_correction_suggestion(
        self,
        alignment_score: float,
        flagged_entities: List[str]
    ) -> str:
        """
        Generate correction suggestion based on verification failure mode.
        
        Args:
            alignment_score: Alignment score between frame and description
            flagged_entities: List of entities with low similarity scores
        
        Returns:
            Correction suggestion string
        """
        suggestions = []
        
        # Check alignment score
        if alignment_score < self.threshold:
            suggestions.append(
                f"Description may not match visual content (alignment: {alignment_score:.3f} < {self.threshold}). "
                "Consider regenerating with more conservative prompt or using CLIP-only embeddings."
            )
        
        # Check flagged entities
        if flagged_entities:
            entity_list = ", ".join(flagged_entities)
            suggestions.append(
                f"Entities may be hallucinated: [{entity_list}]. "
                "Verify these entities are actually present in the frame."
            )
        
        # Combine suggestions
        if len(suggestions) == 2:
            return suggestions[0] + " " + suggestions[1]
        elif len(suggestions) == 1:
            return suggestions[0]
        else:
            return "Verification failed for unknown reason."
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CrossModalVerifier("
            f"threshold={self.threshold}, "
            f"entity_threshold={self.entity_threshold}, "
            f"device={self.device})"
        )
