"""
Dual-window retrieval for comparative queries.

Handles "first vs last" queries by performing independent retrieval
in two temporal windows and merging results.

Author: SHARINGAN Team
Date: March 6, 2026
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    timestamp: float
    frame_idx: int
    confidence: float
    window_label: str  # "first", "last", "middle", etc.


class ComparativeRetrieval:
    """Dual-window retrieval for comparative queries."""
    
    def __init__(self, video_duration: float):
        """
        Initialize comparative retrieval.
        
        Args:
            video_duration: Total video duration in seconds
        """
        self.video_duration = video_duration
    
    def retrieve_dual_window(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        timestamps: np.ndarray,
        frame_indices: np.ndarray,
        window1: Tuple[float, float],  # (start%, end%)
        window2: Tuple[float, float],  # (start%, end%)
        top_k_per_window: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve from two independent temporal windows.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: All frame embeddings (N, D)
            timestamps: Frame timestamps (N,)
            frame_indices: Frame indices (N,)
            window1: First temporal window (start%, end%)
            window2: Second temporal window (start%, end%)
            top_k_per_window: Results per window
            
        Returns:
            List of retrieval results from both windows
            
        Example:
            >>> retriever = ComparativeRetrieval(video_duration=9300)  # 155 min
            >>> results = retriever.retrieve_dual_window(
            ...     query_embedding, embeddings, timestamps, frame_indices,
            ...     window1=(0.0, 0.2),  # First 20%
            ...     window2=(0.8, 1.0),  # Last 20%
            ...     top_k_per_window=3
            ... )
            >>> # Returns 6 results: 3 from first 20%, 3 from last 20%
        """
        results = []
        
        # Retrieve from window 1
        window1_results = self._retrieve_from_window(
            query_embedding, embeddings, timestamps, frame_indices,
            window1, top_k_per_window, label="first"
        )
        results.extend(window1_results)
        
        # Retrieve from window 2
        window2_results = self._retrieve_from_window(
            query_embedding, embeddings, timestamps, frame_indices,
            window2, top_k_per_window, label="last"
        )
        results.extend(window2_results)
        
        return results
    
    def _retrieve_from_window(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        timestamps: np.ndarray,
        frame_indices: np.ndarray,
        window: Tuple[float, float],
        top_k: int,
        label: str
    ) -> List[RetrievalResult]:
        """Retrieve from single temporal window."""
        # Convert window percentages to absolute timestamps
        window_start = window[0] * self.video_duration
        window_end = window[1] * self.video_duration
        
        # Filter embeddings to window
        mask = (timestamps >= window_start) & (timestamps <= window_end)
        window_embeddings = embeddings[mask]
        window_timestamps = timestamps[mask]
        window_frame_indices = frame_indices[mask]
        
        if len(window_embeddings) == 0:
            return []
        
        # Compute similarities
        similarities = window_embeddings @ query_embedding
        
        # Get top-K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                timestamp=float(window_timestamps[idx]),
                frame_idx=int(window_frame_indices[idx]),
                confidence=float(similarities[idx]),
                window_label=label
            ))
        
        return results


def test_comparative_retrieval():
    """Test comparative retrieval."""
    
    print("Testing Comparative Retrieval...")
    print("=" * 60)
    
    # Create test data
    video_duration = 9300.0  # 155 minutes
    n_frames = 1000
    
    np.random.seed(42)
    embeddings = np.random.randn(n_frames, 512).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    timestamps = np.linspace(0, video_duration, n_frames)
    frame_indices = np.arange(n_frames)
    
    query_embedding = np.random.randn(512).astype(np.float32)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Test 1: Dual-window retrieval
    print("\nTest 1: Dual-Window Retrieval")
    print("-" * 60)
    
    retriever = ComparativeRetrieval(video_duration=video_duration)
    
    results = retriever.retrieve_dual_window(
        query_embedding,
        embeddings,
        timestamps,
        frame_indices,
        window1=(0.0, 0.2),  # First 20%
        window2=(0.8, 1.0),  # Last 20%
        top_k_per_window=3
    )
    
    print(f"Total results: {len(results)}")
    
    # Verify results span both windows
    first_window_results = [r for r in results if r.window_label == "first"]
    last_window_results = [r for r in results if r.window_label == "last"]
    
    print(f"First window results: {len(first_window_results)}")
    print(f"Last window results: {len(last_window_results)}")
    
    assert len(first_window_results) == 3, "Should have 3 results from first window"
    assert len(last_window_results) == 3, "Should have 3 results from last window"
    
    # Verify timestamps are in correct windows
    first_20_percent = video_duration * 0.2
    last_20_percent = video_duration * 0.8
    
    for r in first_window_results:
        assert r.timestamp < first_20_percent, f"First window result at {r.timestamp}s should be < {first_20_percent}s"
        print(f"  First window: {r.timestamp:.1f}s (confidence: {r.confidence:.3f})")
    
    for r in last_window_results:
        assert r.timestamp > last_20_percent, f"Last window result at {r.timestamp}s should be > {last_20_percent}s"
        print(f"  Last window: {r.timestamp:.1f}s (confidence: {r.confidence:.3f})")
    
    print("✓ Dual-window retrieval works correctly")
    
    # Test 2: Empty window handling
    print("\nTest 2: Empty Window Handling")
    print("-" * 60)
    
    # Create data with no frames in first window
    sparse_timestamps = np.linspace(video_duration * 0.5, video_duration, n_frames)
    
    results = retriever.retrieve_dual_window(
        query_embedding,
        embeddings,
        sparse_timestamps,
        frame_indices,
        window1=(0.0, 0.2),  # Empty window
        window2=(0.8, 1.0),  # Has frames
        top_k_per_window=3
    )
    
    first_window_results = [r for r in results if r.window_label == "first"]
    last_window_results = [r for r in results if r.window_label == "last"]
    
    print(f"First window results: {len(first_window_results)} (expected 0)")
    print(f"Last window results: {len(last_window_results)} (expected 3)")
    
    assert len(first_window_results) == 0, "Empty window should return no results"
    assert len(last_window_results) == 3, "Non-empty window should return results"
    
    print("✓ Empty window handling works correctly")
    
    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    test_comparative_retrieval()
