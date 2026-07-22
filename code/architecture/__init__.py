"""
Unified architecture components for RACCOONN models (forward and inverse).
Consolidates shared components from both model types.
"""
# Activation functions
from code.architecture.activation import (
    Scale,
    Swish,
    SuperLearnableSwish,
    NonLearnableSwish,
    Sine,
    LearnableSine,
    SuperLearnableSine,
    ScaledTanh,
    Snake,
    gelu,
)

# Positional Encoding modules
from code.architecture.encoding import (
    IdentityPositionalEncoding,
    RescaledPositionalEncoding,
    GaussianPositionalEncoding,
    MultiScaleGaussianEncoding,
)

# Wrapper modules (residual, concatenate)
from code.architecture.wrapper import (
    Residual,
    Concatenate,
)

# MLP components
from code.architecture.mlp import (
    MLPBlock,
    MLPBlocks,
    PredictionHead,
    PredictionHeads,
    MLPModular,
    init_weights_uniform,
    init_weights_siren_first,
    init_weights_siren_hidden,
    init_weights_he,
    init_weights_xavier,
    init_weights_default,
    get_init_func,
)

# ODE architectures (forward model specific)
from code.architecture.ode import (
    ODEFunc,
    ConditionalODEFunc,
    PressureConditionalODEFunc,
    NeuralODEIntegrator,
    CRTMNeuralODE,
)

# CRTM architectures (forward model specific)
from code.architecture.crtm import (
    CRTMBackbone,
    CRTMDualHead,
    CRTMModular,
    CRTMSkipConnection,
)

# Legacy architectures (forward model specific, for backward compatibility)
from code.architecture.legacy import (
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
    'ScaledTanh',
    'Snake',
    'gelu',
    # Encoding modules
    'IdentityPositionalEncoding',
    'RescaledPositionalEncoding',
    'GaussianPositionalEncoding',
    'MultiScaleGaussianEncoding',
    # Wrapper modules
    'Residual',
    'Concatenate',
    # MLP modules
    'MLPBlock',
    'MLPBlocks',
    'PredictionHead',
    'PredictionHeads',
    'MLPModular',
    'init_weights_uniform',
    'init_weights_siren_first',
    'init_weights_siren_hidden',
    'init_weights_he',
    'init_weights_xavier',
    'init_weights_default',
    'get_init_func',
    # ODE modules
    'ODEFunc',
    'ConditionalODEFunc',
    'PressureConditionalODEFunc',
    'NeuralODEIntegrator',
    'CRTMNeuralODE',
    # CRTM modules
    'CRTMBackbone',
    'CRTMDualHead',
    'CRTMModular',
    'CRTMSkipConnection',
    # Legacy architectures
    'CRTMArchitecture',
    'CRTMSmoothArchitecture',
    'CRTMSirenArchitecture',
]

