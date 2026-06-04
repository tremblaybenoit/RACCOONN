"""
Architecture components for RACCOONN inverse models.
Similar structure to 3DClouds architecture module.
"""
from inverse.model.architecture.activation import *
from inverse.model.architecture.encoding import *
from inverse.model.architecture.mlp import *
from inverse.model.architecture.wrapper import *

__all__ = [
    # Activation functions
    'Scale',
    'Swish',
    'SuperLearnableSwish',
    'NonLearnableSwish',
    'Sine',
    'LearnableSine',
    'SuperLearnableSine',
    'ScaledTanh',
    'Snake',
    # Encoding modules
    'RescaledPositionalEncoding',
    'IdentityPositionalEncoding',
    'MultiScaleGaussianEncoding',
    # MLP modules
    'MLPBlock',
    'MLPBlocks',
    'PredictionHead',
    'PredictionHeads',
    'MLPModular',
    # Wrapper modules
    'Residual',
    'Concatenate',
]

