"""
Continuous Time Encoding for SHARINGAN Deep Architecture

SYSTEM DESIGN COMMENT:
======================

Purpose:
--------
This module provides continuous time encoding for video understanding, enabling the system
to preserve temporal relationships and distances between events regardless of variable frame
rates. Unlike discrete frame indices, continuous timestamps maintain the actual temporal
structure of the video.

Why Continuous Time Encoding Matters:
--------------------------------------
1. **Variable Frame Rates**: Videos may have different FPS, and adaptive sampling changes
   the effective frame rate. Continuous time encoding ensures temporal consistency.

2. **Temporal Distance Preservation**: The time difference between events at t=1.5s and t=3.0s
   (delta=1.5s) should be encoded differently from events at t=10.0s and t=11.5s (same delta
   but different absolute positions). Sinusoidal encoding captures both.

3. **Causal Edge Scoring**: When scoring causal relationships between events, the actual time
   delta (in seconds) is more meaningful than frame index differences. An event 0.5s after
   another is more likely causal than one 30s later.

4. **Context-Aware Descriptions**: SmolVLM uses temporal context from previous frames. Encoding
   real timestamps helps the model understand "this happened 2 seconds after that" rather than
   "this is frame 47 after frame 23".

How It Fits in the System:
---------------------------
- **Ingest Pipeline**: Encodes timestamps for each sampled frame before SmolVLM processing
- **Event Graph**: Encodes time deltas between events for causal edge scoring
- **Multi-Scale TAS**: Provides temporal position information for attention mechanisms
- **Query Pipeline**: Encodes temporal bounds for window queries ("between 1:30 and 2:00")

Sinusoidal Encoding:
--------------------
We use the Transformer-style sinusoidal positional encoding adapted for continuous time:

    PE(t, 2i)   = sin(t / 10000^(2i/d_model))
    PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

Where:
- t is the timestamp in seconds
- i is the dimension index (0 to d_model/2)
- d_model is the embedding dimension (default 512)

This encoding has several desirable properties:
- Smooth and continuous (small time changes → small encoding changes)
- Periodic at different frequencies (captures multiple timescales)
- Deterministic (same timestamp always produces same encoding)
- Allows the model to learn relative positions (PE(t+k) can be expressed as linear function of PE(t))

Requirements Validated:
------------------------
- Requirement 2.3: SmolVLM SHALL encode continuous timestamps using sinusoidal positional encoding
- Requirement 6.1: Causal Edge Scorer SHALL accept time delta as input
- Requirement 8.2: Query Router SHALL extract temporal bounds from queries

Example Usage:
--------------
    # Initialize encoder
    encoder = ContinuousTimeEncoder(d_model=512, max_time=3600.0)
    
    # Encode a timestamp (e.g., frame at 15.5 seconds)
    timestamp_encoding = encoder.encode_timestamp(15.5)
    # Returns: np.ndarray of shape (512,)
    
    # Encode time difference between events
    time_delta_encoding = encoder.encode_time_delta(2.5)
    # Returns: np.ndarray of shape (512,)
    
    # Use in causal edge scoring
    edge_score = causal_scorer.score_edge(
        event1_embedding,
        event2_embedding,
        time_delta=encoder.encode_time_delta(event2.timestamp - event1.timestamp)
    )
"""

import numpy as np
from typing import Optional


class ContinuousTimeEncoder:
    """
    Encodes continuous timestamps and time deltas using sinusoidal positional encoding.
    
    This encoder transforms real-valued timestamps (in seconds) into high-dimensional
    embeddings that preserve temporal relationships and distances. It uses the same
    sinusoidal encoding scheme as Transformers, adapted for continuous time rather
    than discrete positions.
    
    Attributes:
        d_model: Embedding dimension (must be even)
        max_time: Maximum expected timestamp in seconds (for normalization)
        _div_term: Precomputed division term for sinusoidal encoding
    """
    
    def __init__(self, d_model: int = 512, max_time: float = 3600.0):
        """
        Initialize continuous time encoder.
        
        Args:
            d_model: Embedding dimension (must be even). Default 512 matches typical
                     video embedding dimensions.
            max_time: Maximum expected timestamp in seconds. Default 3600.0 (1 hour)
                      covers most video lengths. Used for optional normalization.
        
        Raises:
            ValueError: If d_model is not even (required for sin/cos pairs)
        """
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        
        self.d_model = d_model
        self.max_time = max_time
        
        # Precompute the division term for efficiency
        # div_term[i] = 1 / (10000^(2i/d_model)) for i in [0, d_model/2)
        position = np.arange(0, d_model, 2, dtype=np.float32)
        self._div_term = 1.0 / np.power(10000.0, position / d_model)
    
    def encode_timestamp(self, timestamp: float) -> np.ndarray:
        """
        Encode a continuous timestamp as a sinusoidal embedding.
        
        This method transforms a real-valued timestamp (in seconds) into a high-dimensional
        embedding using sinusoidal functions at different frequencies. The encoding preserves
        temporal relationships: similar timestamps produce similar embeddings, and the model
        can learn to extract relative time differences.
        
        Args:
            timestamp: Time in seconds (can be any non-negative float)
        
        Returns:
            np.ndarray of shape (d_model,) containing the sinusoidal encoding
        
        Example:
            >>> encoder = ContinuousTimeEncoder(d_model=512)
            >>> encoding = encoder.encode_timestamp(15.5)
            >>> encoding.shape
            (512,)
        """
        # Create output array
        encoding = np.zeros(self.d_model, dtype=np.float32)
        
        # Apply sinusoidal encoding
        # Even indices: sin(timestamp * div_term)
        # Odd indices: cos(timestamp * div_term)
        encoding[0::2] = np.sin(timestamp * self._div_term)
        encoding[1::2] = np.cos(timestamp * self._div_term)
        
        return encoding
    
    def encode_time_delta(self, delta: float) -> np.ndarray:
        """
        Encode a time difference between events.
        
        This method encodes the temporal distance between two events (in seconds).
        It uses the same sinusoidal encoding as timestamps, but the semantics are
        different: this represents a duration rather than an absolute position.
        
        The encoding is particularly useful for causal edge scoring, where the time
        gap between events is a strong signal for causal relationships:
        - Small deltas (0.5-2s): High causal likelihood
        - Medium deltas (2-10s): Moderate causal likelihood
        - Large deltas (>30s): Low causal likelihood
        
        Args:
            delta: Time difference in seconds (should be non-negative)
        
        Returns:
            np.ndarray of shape (d_model,) containing the sinusoidal encoding
        
        Example:
            >>> encoder = ContinuousTimeEncoder(d_model=512)
            >>> # Encode 2.5 second gap between events
            >>> delta_encoding = encoder.encode_time_delta(2.5)
            >>> delta_encoding.shape
            (512,)
        """
        # Time deltas use the same sinusoidal encoding as timestamps
        # The interpretation is different (duration vs position), but the
        # mathematical encoding is identical
        return self.encode_timestamp(delta)
    
    def encode_batch_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Encode a batch of timestamps efficiently.
        
        This is a vectorized version of encode_timestamp for processing multiple
        timestamps at once, which is more efficient than calling encode_timestamp
        in a loop.
        
        Args:
            timestamps: np.ndarray of shape (N,) containing timestamps in seconds
        
        Returns:
            np.ndarray of shape (N, d_model) containing sinusoidal encodings
        
        Example:
            >>> encoder = ContinuousTimeEncoder(d_model=512)
            >>> timestamps = np.array([0.0, 1.5, 3.0, 10.5])
            >>> encodings = encoder.encode_batch_timestamps(timestamps)
            >>> encodings.shape
            (4, 512)
        """
        N = len(timestamps)
        encodings = np.zeros((N, self.d_model), dtype=np.float32)
        
        # Vectorized computation: timestamps[:, None] broadcasts to (N, d_model/2)
        # _div_term broadcasts to (N, d_model/2)
        encodings[:, 0::2] = np.sin(timestamps[:, None] * self._div_term[None, :])
        encodings[:, 1::2] = np.cos(timestamps[:, None] * self._div_term[None, :])
        
        return encodings
    
    def encode_batch_deltas(self, deltas: np.ndarray) -> np.ndarray:
        """
        Encode a batch of time deltas efficiently.
        
        Vectorized version of encode_time_delta for processing multiple time
        differences at once.
        
        Args:
            deltas: np.ndarray of shape (N,) containing time deltas in seconds
        
        Returns:
            np.ndarray of shape (N, d_model) containing sinusoidal encodings
        
        Example:
            >>> encoder = ContinuousTimeEncoder(d_model=512)
            >>> deltas = np.array([0.5, 1.0, 2.5, 10.0])
            >>> encodings = encoder.encode_batch_deltas(deltas)
            >>> encodings.shape
            (4, 512)
        """
        return self.encode_batch_timestamps(deltas)
    
    def __repr__(self) -> str:
        return f"ContinuousTimeEncoder(d_model={self.d_model}, max_time={self.max_time})"
