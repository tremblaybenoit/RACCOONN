"""
MLP building blocks for forward models.
Reuses components from inverse.model.architecture with forward-specific extensions.
"""
from inverse.model.architecture.mlp import (
    MLPBlock,
    MLPBlocks,
    PredictionHead,
    PredictionHeads,
    MLPModular
)

__all__ = [
    'MLPBlock',
    'MLPBlocks',
    'PredictionHead',
    'PredictionHeads',
    'MLPModular',
]

