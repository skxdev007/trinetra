"""
Magnet Cluster Suppressor for SHARINGAN.

Detects and suppresses semantically rich segments (summaries, chapter cards, narration)
that dominate retrieval across unrelated queries.

Author: SHARINGAN Team
Date: March 6, 2026
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClusterInfo:
    """Information about a detected temporal cluster."""
    center: float  # Center timestamp of cluster
    timestamps: List[float]  # All timestamps in cluster
    size: int  # Number of timestamps in cluster
    ratio: float  # Ratio of cluster size to total results
    is_magnet: bool  # Whether this is a magnet cluster


class MagnetClusterSuppressor:
    """
    Detect and suppress magnet clusters in retrieval results.
    
    A magnet cluster is a single temporal region that dominates retrieval
    across multiple unrelated queries. This typically happens with:
    - Spoken summaries ("In this project, we'll build...")
    - Chapter cards (text overlays listing all steps)
    - Narration (voiceover describing entire project)
    
    These segments contain keywords for EVERYTHING in the video, causing
    them to score highly against almost any query.
    
    Example:
        Woodworking video at 984s: "In this project, we'll build a table
        using epoxy, sanding, and finishing techniques..."
        
        This segment matches queries about:
        - "What is being built?" → "table"
        - "When is epoxy used?" → "epoxy"
        - "When is sanding shown?" → "sanding"
        - "When is finishing applied?" → "finishing"
        
        Result: 984s appears in 7 out of 16 queries (43% contamination)
    """
    
    def __init__(
        self,
        cluster_threshold: float = 60.0,
        max_cluster_ratio: float = 0.4
    ):
        """
        Initialize magnet cluster suppressor.
        
        Args:
            cluster_threshold: Timestamps within this many seconds are
                considered part of the same cluster (default: 60s)
            max_cluster_ratio: Maximum fraction of results allowed in
                one cluster before it's considered a magnet (default: 0.4 = 40%)
        """
        self.cluster_threshold = cluster_threshold
        self.max_cluster_ratio = max_cluster_ratio
    
    def detect_magnet_cluster(
        self,
        timestamps: List[float],
        top_k: int = 5
    ) -> Optional[ClusterInfo]:
        """
        Detect if results are dominated by a single temporal cluster.
        
        Args:
            timestamps: Retrieved timestamps (top-K results)
            top_k: Number of results
            
        Returns:
            ClusterInfo if magnet detected, None otherwise
            
        Example:
            >>> timestamps = [984, 985, 987, 984, 986, 752, 568]
            >>> magnet = suppressor.detect_magnet_cluster(timestamps, top_k=7)
            >>> print(f"Magnet at {magnet.center}s with {magnet.size} results")
            Magnet at 985.2s with 5 results
        """
        if len(timestamps) < 3:
            return None
        
        # Cluster timestamps by temporal proximity
        clusters = self._cluster_timestamps(timestamps, self.cluster_threshold)
        
        if not clusters:
            return None
        
        # Find largest cluster
        largest_cluster = max(clusters, key=lambda c: c['size'])
        cluster_size = largest_cluster['size']
        cluster_ratio = cluster_size / len(timestamps)
        
        # Check if cluster dominates results
        if cluster_ratio > self.max_cluster_ratio:
            return ClusterInfo(
                center=largest_cluster['center'],
                timestamps=largest_cluster['timestamps'],
                size=cluster_size,
                ratio=cluster_ratio,
                is_magnet=True
            )
        
        return None
    
    def _cluster_timestamps(
        self,
        timestamps: List[float],
        threshold: float
    ) -> List[Dict]:
        """
        Cluster timestamps by temporal proximity.
        
        Args:
            timestamps: List of timestamps
            threshold: Maximum gap between timestamps in same cluster
            
        Returns:
            List of clusters, each with 'center', 'timestamps', 'size'
            
        Example:
            >>> timestamps = [2, 5, 6, 50, 52, 100]
            >>> clusters = self._cluster_timestamps(timestamps, threshold=10)
            >>> # Returns 3 clusters: [2,5,6], [50,52], [100]
        """
        if not timestamps:
            return []
        
        sorted_ts = sorted(timestamps)
        clusters = []
        current_cluster = [sorted_ts[0]]
        
        for ts in sorted_ts[1:]:
            if ts - current_cluster[-1] <= threshold:
                # Add to current cluster
                current_cluster.append(ts)
            else:
                # Finalize current cluster and start new one
                clusters.append({
                    'center': float(np.mean(current_cluster)),
                    'timestamps': current_cluster.copy(),
                    'size': len(current_cluster)
                })
                current_cluster = [ts]
        
        # Add final cluster
        if current_cluster:
            clusters.append({
                'center': float(np.mean(current_cluster)),
                'timestamps': current_cluster.copy(),
                'size': len(current_cluster)
            })
        
        return clusters
    
    def suppress_and_rerank(
        self,
        similarities: np.ndarray,
        timestamps: np.ndarray,
        magnet_cluster: ClusterInfo,
        suppression_factor: float = 0.3
    ) -> np.ndarray:
        """
        Suppress magnet cluster region and allow retrieval from other regions.
        
        Args:
            similarities: Similarity scores for all frames (N,)
            timestamps: Timestamps for all frames (N,)
            magnet_cluster: Detected magnet cluster info
            suppression_factor: Multiply magnet scores by this factor
                (default: 0.3 = 70% penalty)
            
        Returns:
            Adjusted similarity scores with magnet suppressed
            
        Example:
            >>> # Before: Top-5 results all from 984-987s
            >>> # After: Top-5 results distributed across video
            >>> adjusted = suppressor.suppress_and_rerank(
            ...     similarities, timestamps, magnet, suppression_factor=0.3
            ... )
        """
        adjusted = similarities.copy()
        
        # Define magnet region (cluster center ± threshold/2)
        magnet_center = magnet_cluster.center
        magnet_radius = self.cluster_threshold / 2
        
        # Suppress all frames in magnet region
        magnet_mask = np.abs(timestamps - magnet_center) <= magnet_radius
        adjusted[magnet_mask] *= suppression_factor
        
        return adjusted
    
    def enforce_diversity(
        self,
        similarities: np.ndarray,
        timestamps: np.ndarray,
        frame_indices: np.ndarray,
        top_k: int = 5,
        max_iterations: int = 3
    ) -> Tuple[np.ndarray, bool]:
        """
        Iteratively detect and suppress magnets until results are diverse.
        
        Args:
            similarities: Similarity scores for all frames
            timestamps: Timestamps for all frames
            frame_indices: Frame indices for all frames
            top_k: Number of results to return
            max_iterations: Maximum suppression iterations
            
        Returns:
            Tuple of (top_indices, magnet_detected)
            
        Example:
            >>> top_indices, had_magnet = suppressor.enforce_diversity(
            ...     similarities, timestamps, frame_indices, top_k=5
            ... )
            >>> if had_magnet:
            ...     print("Magnet detected and suppressed")
        """
        adjusted_similarities = similarities.copy()
        magnet_detected = False
        
        for iteration in range(max_iterations):
            # Get current top-K
            top_indices = np.argsort(adjusted_similarities)[-top_k:][::-1]
            top_timestamps = timestamps[top_indices]
            
            # Check for magnet cluster
            magnet = self.detect_magnet_cluster(top_timestamps.tolist(), top_k)
            
            if magnet is None:
                # No magnet, results are diverse
                break
            
            magnet_detected = True
            
            # Suppress magnet and continue
            adjusted_similarities = self.suppress_and_rerank(
                adjusted_similarities,
                timestamps,
                magnet,
                suppression_factor=0.3
            )
        
        # Final top-K after all suppressions
        top_indices = np.argsort(adjusted_similarities)[-top_k:][::-1]
        
        return top_indices, magnet_detected
    
    def get_diversity_score(self, timestamps: List[float]) -> float:
        """
        Calculate diversity score for a set of timestamps.
        
        Higher score = more diverse (spread across video).
        Lower score = less diverse (clustered together).
        
        Args:
            timestamps: List of timestamps
            
        Returns:
            Diversity score between 0.0 (all same) and 1.0 (maximally spread)
            
        Example:
            >>> # All timestamps at 984s
            >>> score = suppressor.get_diversity_score([984, 985, 984, 986])
            >>> print(score)  # ~0.1 (very low diversity)
            
            >>> # Timestamps spread across video
            >>> score = suppressor.get_diversity_score([10, 500, 1000, 1500])
            >>> print(score)  # ~0.9 (high diversity)
        """
        if len(timestamps) < 2:
            return 1.0
        
        sorted_ts = sorted(timestamps)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(sorted_ts) - 1):
            distances.append(sorted_ts[i+1] - sorted_ts[i])
        
        # Diversity = coefficient of variation of distances
        # High CV = uneven spacing = low diversity
        # Low CV = even spacing = high diversity
        
        if len(distances) == 0:
            return 1.0
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist == 0:
            return 0.0
        
        cv = std_dist / mean_dist
        
        # Normalize: CV of 0 = perfect diversity (1.0)
        #            CV of 2+ = poor diversity (0.0)
        diversity = max(0.0, min(1.0, 1.0 - cv / 2.0))
        
        return diversity


def test_magnet_suppressor():
    """Test magnet cluster suppressor with example data."""
    
    print("Testing Magnet Cluster Suppressor...")
    print("=" * 60)
    
    # Create suppressor
    suppressor = MagnetClusterSuppressor(
        cluster_threshold=60.0,
        max_cluster_ratio=0.4
    )
    
    # Test 1: Detect magnet cluster
    print("\nTest 1: Detect Magnet Cluster")
    print("-" * 60)
    
    # Woodworking video: 984s magnet appears in 7/16 queries
    timestamps = [984, 985, 987, 984, 986, 984, 752]
    
    magnet = suppressor.detect_magnet_cluster(timestamps, top_k=7)
    
    if magnet:
        print(f"✓ Magnet detected at {magnet.center:.1f}s")
        print(f"  Size: {magnet.size}/{len(timestamps)} results ({magnet.ratio*100:.1f}%)")
        print(f"  Timestamps: {magnet.timestamps}")
    else:
        print("✗ No magnet detected")
    
    # Test 2: Diverse results (no magnet)
    print("\nTest 2: Diverse Results (No Magnet)")
    print("-" * 60)
    
    diverse_timestamps = [10, 500, 1000, 1500, 2000]
    
    magnet = suppressor.detect_magnet_cluster(diverse_timestamps, top_k=5)
    
    if magnet:
        print(f"✗ False positive: magnet detected at {magnet.center:.1f}s")
    else:
        print("✓ No magnet detected (correct)")
    
    # Test 3: Diversity score
    print("\nTest 3: Diversity Score")
    print("-" * 60)
    
    clustered_score = suppressor.get_diversity_score([984, 985, 984, 986])
    diverse_score = suppressor.get_diversity_score([10, 500, 1000, 1500])
    
    print(f"Clustered timestamps: diversity = {clustered_score:.2f}")
    print(f"Diverse timestamps: diversity = {diverse_score:.2f}")
    
    # Test 4: Suppress and rerank
    print("\nTest 4: Suppress and Rerank")
    print("-" * 60)
    
    # Simulate retrieval with magnet
    np.random.seed(42)
    similarities = np.random.rand(1000) * 0.5  # Base scores 0-0.5
    timestamps_all = np.linspace(0, 1740, 1000)  # 29 min video
    
    # Artificially boost magnet region (984s)
    magnet_idx = np.argmin(np.abs(timestamps_all - 984))
    similarities[magnet_idx-5:magnet_idx+5] = 0.95  # Very high scores
    
    # Get top-5 before suppression
    top_indices_before = np.argsort(similarities)[-5:][::-1]
    top_ts_before = timestamps_all[top_indices_before]
    
    print(f"Before suppression: {top_ts_before}")
    
    # Detect magnet
    magnet = suppressor.detect_magnet_cluster(top_ts_before.tolist(), top_k=5)
    
    if magnet:
        print(f"Magnet detected at {magnet.center:.1f}s")
        
        # Suppress
        adjusted = suppressor.suppress_and_rerank(
            similarities, timestamps_all, magnet, suppression_factor=0.3
        )
        
        # Get top-5 after suppression
        top_indices_after = np.argsort(adjusted)[-5:][::-1]
        top_ts_after = timestamps_all[top_indices_after]
        
        print(f"After suppression: {top_ts_after}")
        
        # Check diversity improved
        diversity_before = suppressor.get_diversity_score(top_ts_before.tolist())
        diversity_after = suppressor.get_diversity_score(top_ts_after.tolist())
        
        print(f"Diversity before: {diversity_before:.2f}")
        print(f"Diversity after: {diversity_after:.2f}")
        
        if diversity_after > diversity_before:
            print("✓ Diversity improved")
        else:
            print("✗ Diversity did not improve")
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    test_magnet_suppressor()
