"""Speed optimization utilities for SHARINGAN."""

from .speed_boost import (
    VisualTokenReducer,
    compile_model_for_speed,
    enable_flash_attention,
    SpeedOptimizer,
    print_optimization_guide,
    OPTIMIZATION_GUIDE
)

__all__ = [
    'VisualTokenReducer',
    'compile_model_for_speed',
    'enable_flash_attention',
    'SpeedOptimizer',
    'print_optimization_guide',
    'OPTIMIZATION_GUIDE'
]
