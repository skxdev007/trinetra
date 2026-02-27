"""
Run Multi-Scale TAS tests directly without pytest.
"""

import sys
import torch
from sharingan.temporal.multi_scale_tas import (
    TemporalAttentionShift,
    MultiScaleTASStream
)


def test_temporal_attention_shift_basic():
    """Test basic TemporalAttentionShift functionality."""
    print("Testing TemporalAttentionShift basic functionality...")
    embed_dim = 512
    kernel_size = 8
    num_frames = 100
    
    model = TemporalAttentionShift(embed_dim, kernel_size)
    embeddings = torch.randn(num_frames, embed_dim)
    
    output = model(embeddings)
    
    assert output.shape == embeddings.shape, f"Shape mismatch: {output.shape} != {embeddings.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("  ✓ PASSED")


def test_multi_scale_tas_basic():
    """Test basic MultiScaleTASStream functionality."""
    print("Testing MultiScaleTASStream basic functionality...")
    embed_dim = 512
    window_size = 64
    num_frames = 100
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(num_frames, embed_dim)
    timestamps = torch.linspace(0, 10, num_frames)
    
    output = model(embeddings, timestamps)
    
    assert output.shape == embeddings.shape, f"Shape mismatch: {output.shape} != {embeddings.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("  ✓ PASSED")


def test_multi_scale_tas_batch():
    """Test MultiScaleTASStream with batch input."""
    print("Testing MultiScaleTASStream with batch input...")
    embed_dim = 512
    window_size = 64
    batch_size = 2
    num_frames = 50
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(batch_size, num_frames, embed_dim)
    timestamps = torch.linspace(0, 5, num_frames).unsqueeze(0).expand(batch_size, -1)
    
    output = model(embeddings, timestamps)
    
    assert output.shape == embeddings.shape, f"Shape mismatch: {output.shape} != {embeddings.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("  ✓ PASSED")


def test_non_causal_mode_raises_error():
    """Test that non-causal mode raises ValueError."""
    print("Testing non-causal mode raises error...")
    try:
        MultiScaleTASStream(embed_dim=512, window_size=64, causal=False)
        print("  ✗ FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        if "Non-causal mode not supported" in str(e):
            print("  ✓ PASSED")
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            sys.exit(1)


def test_multi_scale_tas_short_sequence():
    """Test MultiScaleTASStream with very short sequence."""
    print("Testing MultiScaleTASStream with short sequence...")
    embed_dim = 512
    window_size = 64
    num_frames = 5  # Shorter than all kernel sizes
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(num_frames, embed_dim)
    
    output = model(embeddings)
    
    assert output.shape == embeddings.shape, f"Shape mismatch: {output.shape} != {embeddings.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("  ✓ PASSED")


def test_multi_scale_tas_single_frame():
    """Test MultiScaleTASStream with single frame."""
    print("Testing MultiScaleTASStream with single frame...")
    embed_dim = 512
    window_size = 64
    
    model = MultiScaleTASStream(embed_dim, window_size, causal=True)
    embeddings = torch.randn(1, embed_dim)
    
    output = model(embeddings)
    
    assert output.shape == embeddings.shape, f"Shape mismatch: {output.shape} != {embeddings.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    print("  ✓ PASSED")


def main():
    print("="*80)
    print("MULTI-SCALE TAS BASIC TESTS")
    print("="*80)
    print()
    
    try:
        test_temporal_attention_shift_basic()
        test_multi_scale_tas_basic()
        test_multi_scale_tas_batch()
        test_non_causal_mode_raises_error()
        test_multi_scale_tas_short_sequence()
        test_multi_scale_tas_single_frame()
        
        print()
        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        
    except Exception as e:
        print()
        print("="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()
