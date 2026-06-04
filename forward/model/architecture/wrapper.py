"""
Wrapper modules for forward model architectures.
Reuses components from inverse architecture where appropriate.
"""
from inverse.model.architecture.wrapper import Residual, Concatenate

__all__ = [
    'Residual',
    'Concatenate',
]

