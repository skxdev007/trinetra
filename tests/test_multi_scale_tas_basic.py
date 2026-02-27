"""
Basic tests for Multi-Scale TAS implementation.

This is a simple smoke test to verify the implementation works.
Full unit tests and property tests are in tasks 1.2 and 1.3.
"""

import torch
import pytest
from sharingan.temporal.multi_scale_tas import (
    TemporalAttentionShift,
    MultiScaleTASStream
)


def test_temporal_attention_shift_basic():
    """Test basic TemporalAttentionShift functionality."""
    embed_dim = 512
    kernel_size = 8
    num_frames = 100
    
    model = TemporalAttentionShift(embed_dim, kernel_size)
    embeddings = torch.randn(num_frames, embed_dim)
    
    output = model(embeddings)
    
    assert output.shape == embeddings.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_multi_scale_tas_basic():
    """Test basic MultiScaleTASStream functionality."""
    embed_dim = 512
    window_size = 64
    num_frames = 100
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(num_frames, embed_dim)
    timestamps = torch.linspace(0, 10, num_frames)
    
    output = model(embeddings, timestamps)
    
    assert output.shape == embeddings.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_multi_scale_tas_batch():
    """Test MultiScaleTASStream with batch input."""
    embed_dim = 512
    window_size = 64
    batch_size = 2
    num_frames = 50
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(batch_size, num_frames, embed_dim)
    timestamps = torch.linspace(0, 5, num_frames).unsqueeze(0).expand(batch_size, -1)
    
    output = model(embeddings, timestamps)
    
    assert output.shape == embeddings.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_non_causal_mode_raises_error():
    """Test that non-causal mode raises ValueError."""
    with pytest.raises(ValueError, match="Non-causal mode not supported"):
        MultiScaleTASStream(embed_dim=512, window_size=64, causal=False)


def test_multi_scale_tas_short_sequence():
    """Test MultiScaleTASStream with very short sequence."""
    embed_dim = 512
    window_size = 64
    num_frames = 5  # Shorter than all kernel sizes
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(num_frames, embed_dim)
    
    output = model(embeddings)
    
    assert output.shape == embeddings.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_multi_scale_tas_single_frame():
    """Test MultiScaleTASStream with single frame."""
    embed_dim = 512
    window_size = 64
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(1, embed_dim)
    
    output = model(embeddings)
    
    assert output.shape == embeddings.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
