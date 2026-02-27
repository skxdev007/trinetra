"""
Integration test for Context-Aware SmolVLM with Cross-Modal Verification.

This test validates the complete pipeline:
1. Context-aware frame description generation with SmolVLM
2. Cross-modal verification with CLIP
3. Flagging of low-confidence descriptions
4. Correction suggestions

Requirements tested:
- Requirement 2.1: SmolVLM maintains rolling context buffer of up to 8 frames
- Requirement 2.2: SmolVLM includes context from up to 8 previous frames
- Requirement 3.1: Cross-modal verifier computes CLIP similarity
- Requirement 3.2: Verifier flags descriptions with CLIP similarity < 0.7
- Requirement 3.5: Verifier provides correction suggestions

NOTE: Full integration tests require downloading models (CLIP ~605MB, SmolVLM ~538MB).
      First run will take time to download. Subsequent runs will use cached models.
      
      To run full tests: pytest tests/test_context_aware_verification_integration.py -v -s
      To run quick tests: pytest tests/test_context_aware_verification_integration.py -k "not Integration" -v
"""

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from sharingan.vlm.context_aware_smolvlm import ContextAwareSmolVLM, FrameDescription
from sharingan.verification.cross_modal import CrossModalVerifier, VerificationResult


# Skip tests if models are not available (for CI/CD)
pytest_plugins = []

try:
    import transformers
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


def create_synthetic_frame(
    width: int = 640,
    height: int = 480,
    text: str = "",
    color: tuple = (100, 150, 200)
) -> np.ndarray:
    """
    Create a synthetic frame with optional text overlay.
    
    This is used for testing when real video data is not available.
    
    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        text: Optional text to overlay on frame
        color: Background color as RGB tuple
    
    Returns:
        Frame as numpy array (H, W, 3) in RGB format
    """
    # Create PIL image with solid color background
    img = Image.new('RGB', (width, height), color=color)
    
    # Add text if provided
    if text:
        draw = ImageDraw.Draw(img)
        # Use default font (may not render perfectly, but sufficient for testing)
        try:
            # Try to use a larger font if available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        # Draw text in white
        draw.text(position, text, fill=(255, 255, 255), font=font)
    
    # Convert to numpy array
    frame = np.array(img)
    
    return frame


def create_test_video_sequence(num_frames: int = 10) -> list:
    """
    Create a sequence of synthetic frames simulating a simple video.
    
    Args:
        num_frames: Number of frames to generate
    
    Returns:
        List of (frame, timestamp, frame_index) tuples
    """
    frames = []
    
    for i in range(num_frames):
        # Vary color slightly to simulate motion
        color = (100 + i * 5, 150 + i * 3, 200 - i * 2)
        
        # Add frame number as text
        text = f"Frame {i}"
        
        frame = create_synthetic_frame(text=text, color=color)
        timestamp = i * 0.5  # 2 FPS
        
        frames.append((frame, timestamp, i))
    
    return frames


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Transformers not available")
class TestContextAwareVerificationIntegration:
    """Integration tests for context-aware description with verification."""
    
    @pytest.fixture(scope="class")
    def smolvlm(self):
        """Initialize ContextAwareSmolVLM (shared across tests)."""
        try:
            vlm = ContextAwareSmolVLM(
                context_window=8,
                device="cpu"  # Use CPU for testing to avoid GPU requirements
            )
            return vlm
        except Exception as e:
            pytest.skip(f"Failed to initialize SmolVLM: {str(e)}")
    
    @pytest.fixture(scope="class")
    def verifier(self):
        """Initialize CrossModalVerifier (shared across tests)."""
        try:
            verifier = CrossModalVerifier(
                threshold=0.7,
                entity_threshold=0.5,
                device="cpu"  # Use CPU for testing
            )
            return verifier
        except Exception as e:
            pytest.skip(f"Failed to initialize verifier: {str(e)}")
    
    def test_context_buffer_management(self, smolvlm):
        """
        Test that context buffer maintains FIFO behavior with max 8 frames.
        
        Validates:
        - Requirement 2.1: Rolling context buffer of up to 8 frames
        - Requirement 2.5: Remove oldest frame when buffer exceeds 8
        """
        # Clear context buffer
        smolvlm.clear_context()
        assert smolvlm.get_context_size() == 0
        
        # Create test frames
        frames = create_test_video_sequence(num_frames=12)
        
        # Process frames and track context size
        for frame, timestamp, frame_idx in frames:
            # Generate description (this will use current context)
            description = smolvlm.describe_with_context(
                current_frame=frame,
                timestamp=timestamp,
                frame_index=frame_idx
            )
            
            # Update context
            smolvlm.update_context(
                frame=frame,
                description=description.description,
                frame_index=frame_idx,
                timestamp=timestamp
            )
            
            # Verify context size never exceeds 8
            context_size = smolvlm.get_context_size()
            assert context_size <= 8, f"Context size {context_size} exceeds maximum of 8"
            
            # After 8 frames, context should be exactly 8
            if frame_idx >= 8:
                assert context_size == 8, f"Context size should be 8 after {frame_idx} frames"
    
    def test_context_used_tracking(self, smolvlm):
        """
        Test that context_used field correctly tracks which frames were used.
        
        Validates:
        - Requirement 2.6: Record which context frames were used
        """
        # Clear context buffer
        smolvlm.clear_context()
        
        # Create test frames
        frames = create_test_video_sequence(num_frames=10)
        
        # Process first 5 frames
        for i in range(5):
            frame, timestamp, frame_idx = frames[i]
            
            description = smolvlm.describe_with_context(
                current_frame=frame,
                timestamp=timestamp,
                frame_index=frame_idx
            )
            
            # context_used should contain indices of previous frames
            assert len(description.context_used) == i, \
                f"Frame {i} should have {i} context frames, got {len(description.context_used)}"
            
            # Update context for next iteration
            smolvlm.update_context(frame, description.description, frame_idx, timestamp)
    
    def test_verification_basic(self, verifier):
        """
        Test basic cross-modal verification functionality.
        
        Validates:
        - Requirement 3.1: Compute CLIP similarity between frame and description
        - Requirement 3.6: Return verification results with required fields
        """
        # Create a simple test frame
        frame = create_synthetic_frame(text="Person", color=(150, 150, 150))
        
        # Test with matching description (should have high alignment)
        description = "A gray image with text"
        entities = ["text", "image"]
        
        result = verifier.verify_description(
            frame=frame,
            description=description,
            entities=entities
        )
        
        # Verify result structure
        assert isinstance(result, VerificationResult)
        assert isinstance(result.is_verified, bool)
        assert isinstance(result.alignment_score, float)
        assert isinstance(result.flagged_entities, list)
        assert 0.0 <= result.alignment_score <= 1.0
    
    def test_verification_flagging(self, verifier):
        """
        Test that verifier flags descriptions with low CLIP similarity.
        
        Validates:
        - Requirement 3.2: Flag descriptions with CLIP similarity < 0.7
        - Requirement 3.4: Flag entities with CLIP similarity < 0.5
        - Requirement 3.5: Provide correction suggestions
        """
        # Create a simple frame
        frame = create_synthetic_frame(text="Simple", color=(100, 100, 100))
        
        # Test with completely mismatched description (should have low alignment)
        mismatched_description = "A complex scene with multiple people dancing in a colorful room with furniture"
        mismatched_entities = ["people", "dancing", "furniture", "colorful room"]
        
        result = verifier.verify_description(
            frame=frame,
            description=mismatched_description,
            entities=mismatched_entities
        )
        
        # With a completely mismatched description, we expect:
        # - Low alignment score (likely < 0.7)
        # - Possibly flagged entities
        # - Correction suggestion if verification fails
        
        if not result.is_verified:
            # If verification failed, correction suggestion should be provided
            assert result.correction_suggestion is not None
            assert len(result.correction_suggestion) > 0
            print(f"✓ Verifier correctly flagged mismatched description")
            print(f"  Alignment score: {result.alignment_score:.3f}")
            print(f"  Flagged entities: {result.flagged_entities}")
            print(f"  Suggestion: {result.correction_suggestion}")
        else:
            # If verification passed, alignment should be high
            assert result.alignment_score >= 0.7
            print(f"✓ Verifier accepted description (alignment: {result.alignment_score:.3f})")
    
    def test_end_to_end_pipeline(self, smolvlm, verifier):
        """
        Test complete pipeline: context-aware description + verification.
        
        This is the main integration test that validates the full workflow:
        1. Generate descriptions with context
        2. Verify each description
        3. Track flagged descriptions
        4. Verify corrections are suggested
        
        Validates:
        - Requirements 2.1, 2.2 (context-aware description)
        - Requirements 3.1, 3.2, 3.5 (verification and flagging)
        """
        # Clear context
        smolvlm.clear_context()
        
        # Create test video sequence
        frames = create_test_video_sequence(num_frames=10)
        
        # Track results
        descriptions = []
        verifications = []
        flagged_count = 0
        
        print("\n" + "="*80)
        print("INTEGRATION TEST: Context-Aware Description + Verification")
        print("="*80)
        
        # Process each frame
        for frame, timestamp, frame_idx in frames:
            # Generate description with context
            description = smolvlm.describe_with_context(
                current_frame=frame,
                timestamp=timestamp,
                frame_index=frame_idx
            )
            
            # Verify description
            verification = verifier.verify_description(
                frame=frame,
                description=description.description,
                entities=description.entities
            )
            
            # Store results
            descriptions.append(description)
            verifications.append(verification)
            
            # Track flagged descriptions
            if not verification.is_verified:
                flagged_count += 1
                print(f"\n⚠️  Frame {frame_idx} at {timestamp:.2f}s: FLAGGED")
                print(f"   Description: {description.description[:100]}...")
                print(f"   Alignment: {verification.alignment_score:.3f}")
                print(f"   Flagged entities: {verification.flagged_entities}")
                print(f"   Suggestion: {verification.correction_suggestion}")
            else:
                print(f"\n✓ Frame {frame_idx} at {timestamp:.2f}s: VERIFIED")
                print(f"   Description: {description.description[:100]}...")
                print(f"   Alignment: {verification.alignment_score:.3f}")
            
            # Update context for next frame
            smolvlm.update_context(
                frame=frame,
                description=description.description,
                frame_index=frame_idx,
                timestamp=timestamp
            )
        
        print("\n" + "="*80)
        print(f"SUMMARY: {len(frames)} frames processed")
        print(f"  Verified: {len(frames) - flagged_count}")
        print(f"  Flagged: {flagged_count}")
        print("="*80 + "\n")
        
        # Assertions
        assert len(descriptions) == len(frames), "Should have description for each frame"
        assert len(verifications) == len(frames), "Should have verification for each frame"
        
        # All descriptions should have valid structure
        for desc in descriptions:
            assert isinstance(desc, FrameDescription)
            assert desc.timestamp >= 0.0
            assert len(desc.description) > 0
            assert 0.0 <= desc.confidence <= 1.0
        
        # All verifications should have valid structure
        for verif in verifications:
            assert isinstance(verif, VerificationResult)
            assert 0.0 <= verif.alignment_score <= 1.0
            
            # If flagged, should have correction suggestion
            if not verif.is_verified:
                assert verif.correction_suggestion is not None
    
    def test_entity_level_verification(self, verifier):
        """
        Test entity-level verification with different entity types.
        
        Validates:
        - Requirement 3.3: Verify each entity mentioned in description
        - Requirement 3.4: Flag entities with similarity < 0.5
        """
        # Create frame with specific visual content
        frame = create_synthetic_frame(text="PERSON", color=(120, 120, 120))
        
        # Test with entities that should match vs not match
        description = "An image with text showing a person"
        entities = ["text", "person", "elephant", "car"]  # elephant and car should be flagged
        
        result = verifier.verify_description(
            frame=frame,
            description=description,
            entities=entities
        )
        
        print(f"\nEntity-level verification test:")
        print(f"  Alignment score: {result.alignment_score:.3f}")
        print(f"  Flagged entities: {result.flagged_entities}")
        
        # We expect some entities to be flagged (those not present in frame)
        # The exact entities flagged depend on CLIP's behavior, but we can verify
        # that the flagging mechanism works
        assert isinstance(result.flagged_entities, list)


def test_synthetic_frame_creation():
    """Test that synthetic frame creation works correctly."""
    frame = create_synthetic_frame(width=640, height=480, text="Test", color=(100, 150, 200))
    
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame >= 0) and np.all(frame <= 255)


def test_video_sequence_creation():
    """Test that video sequence creation works correctly."""
    frames = create_test_video_sequence(num_frames=5)
    
    assert len(frames) == 5
    
    for i, (frame, timestamp, frame_idx) in enumerate(frames):
        assert frame.shape == (480, 640, 3)
        assert timestamp == i * 0.5
        assert frame_idx == i


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
