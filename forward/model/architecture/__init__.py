"""
Forward model architecture components.
Organized similarly to 3DClouds and inverse model architecture.
"""
from forward.model.architecture.activation import *
from forward.model.architecture.mlp import *
from forward.model.architecture.wrapper import *
from forward.model.architecture.ode import *
from forward.model.architecture.crtm import *

# Legacy architectures for backward compatibility
from forward.model.architecture.legacy import (
    CRTMArchitecture,
    CRTMSmoothArchitecture,
    CRTMSirenArchitecture,
)

__all__ = [
    # Activation functions
    'Scale',
    'Swish',
    'SuperLearnableSwish',
    'NonLearnableSwish',
    'Sine',
    'LearnableSine',
    'SuperLearnableSine',
    # MLP modules (from inverse)
    'MLPBlock',
    'MLPBlocks',
    'PredictionHead',
    'PredictionHeads',
    'MLPModular',
    # Wrapper modules (from inverse)
    'Residual',
    'Concatenate',
    # ODE modules
    'ODEFunc',
    'ConditionalODEFunc',
    'PressureConditionalODEFunc',
    'NeuralODEIntegrator',
    'CRTMNeuralODE',
    # CRTM architectures
    'CRTMBackbone',
    'CRTMDualHead',
    'CRTMModular',
    'CRTMSkipConnection',
    # Legacy architectures
    'CRTMArchitecture',
    'CRTMSmoothArchitecture',
    'CRTMSirenArchitecture',
]

