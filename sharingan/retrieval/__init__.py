"""
Retrieval module for SHARINGAN.

Provides intelligent retrieval strategies and result processing.
"""

from sharingan.retrieval.magnet_suppressor import (
    MagnetClusterSuppressor,
    ClusterInfo
)
from sharingan.retrieval.comparative_search import (
    ComparativeRetrieval,
    RetrievalResult
)

__all__ = [
    'MagnetClusterSuppressor',
    'ClusterInfo',
    'ComparativeRetrieval',
    'RetrievalResult'
]
