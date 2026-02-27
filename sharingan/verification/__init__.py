"""
Cross-Modal Verification Module for SHARINGAN Deep Architecture

This module provides cross-modal verification capabilities to detect and flag
VLM hallucinations by comparing visual and textual representations using CLIP.
"""

from sharingan.verification.cross_modal import (
    CrossModalVerifier,
    VerificationResult
)

__all__ = [
    'CrossModalVerifier',
    'VerificationResult'
]
