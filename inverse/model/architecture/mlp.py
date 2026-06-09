import torch
import torch.nn as nn
from typing import Callable
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from inverse.model.architecture.wrapper import Residual


# ============================================================================
# Weight Initialization Functions
# ============================================================================

def init_weights_uniform(layer: nn.Linear, a: float = -1.0, b: float = 1.0) -> None:
    """
    Uniform weight initialization for SIREN networks.

    Weights and biases are sampled from U(a, b). Common usage:
    - First layer: U(-1, 1)
    - Hidden layers: U(-sqrt(c), sqrt(c)) where c = 1 / input_features

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    a : float, default=-1.0
        Lower bound of uniform distribution.
    b : float, default=1.0
        Upper bound of uniform distribution.
    """
    with torch.no_grad():
        nn.init.uniform_(layer.weight, a, b)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, a, b)


def init_weights_siren_first(layer: nn.Linear) -> None:
    """
    Initialize first layer of SIREN network (Sine activation).

    Weights and biases sampled from U(-1, 1). This is the recommended
    initialization for the input layer of SIREN networks.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    """
    init_weights_uniform(layer, a=-1.0, b=1.0)


def init_weights_siren_hidden(layer: nn.Linear) -> None:
    """
    Initialize hidden layers of SIREN network (Sine activation).

    Weights and biases sampled from U(-sqrt(c), sqrt(c)), where c = 1 / input_features.
    This is the recommended initialization for hidden layers of SIREN networks.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    """
    with torch.no_grad():
        in_features = layer.weight.shape[1]
        bound = (1.0 / in_features) ** 0.5
        nn.init.uniform_(layer.weight, -bound, bound)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -bound, bound)


def init_weights_he(layer: nn.Linear, nonlinearity: str = 'relu') -> None:
    """
    He (Kaiming) initialization for ReLU-like activations.

    Recommended for ReLU, Leaky ReLU, Swish, and similar activations that benefit
    from variance-aware initialization. Biases are initialized to zero.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    nonlinearity : str, default='relu'
        Type of nonlinearity ('relu', 'leaky_relu', etc.).
    """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_weights_xavier(layer: nn.Linear, gain: float = 1.0) -> None:
    """
    Xavier (Glorot) uniform initialization.

    Recommended for GELU, Tanh, Sigmoid, and other S-shaped activations.
    Biases are initialized to zero.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    gain : float, default=1.0
        Scaling factor for the initialization.
    """
    with torch.no_grad():
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def init_weights_default(layer: nn.Linear) -> None:
    """
    Default weight initialization (Xavier uniform).

    This is the fallback initialization used when no specific initialization
    can be determined from the activation function.

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    """
    init_weights_xavier(layer, gain=1.0)


def get_init_func(activation: DictConfig | nn.Module | None) -> Callable:
    """
    Auto-select weight initialization function based on activation type.

    Automatically determines the most appropriate initialization function based on
    the activation function provided. This enables intelligent weight initialization
    that adapts to different activation types without manual configuration.

    Activation-to-Initialization Mapping:
    - Sine / LearnableSine: Uniform initialization (SIREN networks)
    - ReLU / Leaky ReLU: He initialization (Kaiming)
    - Swish / SiLU: He initialization (similar to ReLU)
    - GELU / Tanh / Sigmoid: Xavier initialization (Glorot)
    - Scale / Snake: Xavier initialization (stable activations)
    - None / Unknown: Xavier initialization (default fallback)

    Parameters
    ----------
    activation : DictConfig or nn.Module or None
        Activation function. Can be:
        - DictConfig with '_target_' pointing to an activation class
        - nn.Module instance of an activation
        - None (defaults to Xavier initialization)

    Returns
    -------
    Callable
        Weight initialization function appropriate for the given activation.
    """
    if activation is None:
        # Default to Xavier initialization for None
        return init_weights_xavier

    # Extract target string from DictConfig or nn.Module
    if isinstance(activation, DictConfig):
        if hasattr(activation, '_target_'):
            target = activation._target_
        else:
            return init_weights_default
    else:
        # Get class name from module instance
        target = activation.__class__.__name__

    # Normalize target to lowercase for case-insensitive matching
    target_str = str(target).lower()

    # Map activation name to appropriate initialization function
    if 'sine' in target_str:
        # SIREN networks use uniform initialization
        return init_weights_siren_hidden
    elif 'relu' in target_str or 'leaky' in target_str:
        # ReLU-like activations use He initialization
        return init_weights_he
    elif 'swish' in target_str or 'silu' in target_str:
        # Swish is similar to ReLU, use He initialization
        return init_weights_he
    elif 'gelu' in target_str:
        # GELU uses Xavier initialization
        return init_weights_xavier
    elif 'tanh' in target_str:
        # Tanh uses Xavier initialization
        return init_weights_xavier
    elif 'sigmoid' in target_str:
        # Sigmoid uses Xavier initialization
        return init_weights_xavier
    elif 'scale' in target_str or 'snake' in target_str:
        # Scale and Snake use Xavier initialization
        return init_weights_xavier
    else:
        # Default to Xavier for unknown activations
        return init_weights_xavier


# ============================================================================
# Architecture & components
# ============================================================================


class MLPBlock(nn.Module):
    """
    Single MLP block with normalization, dropout, activation, and optional residual connection.

    A flexible building block combining: Linear -> Normalization -> Activation -> Dropout.
    Supports residual connections with automatic projection for dimension mismatches.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: DictConfig | nn.Module | None = None,
        norm_type: str | None = None,
        dropout_rate: float = 0.0,
        residual: bool = False,
        init_func: DictConfig | nn.Module | None = None,
    ) -> None:
        """
        Initialize a single MLP block.

        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension.
        activation : DictConfig or nn.Module or None, default=None
            Activation function. If None, defaults to GELU.
            Can be a DictConfig with '_target_' for lazy instantiation.
        norm_type : str or None, default=None
            Normalization type applied after the linear layer.
            Options: 'layer' (LayerNorm), 'batch' (BatchNorm1d), 'none' (Identity).
        dropout_rate : float, default=0.0
            Dropout probability applied after activation. Range [0.0, 1.0].
        residual : bool, default=False
            Whether to add a skip connection. A learned projection is added
            automatically when in_features differs from out_features.
        init_func : DictConfig or Callable or None, default=None
            Weight initialization function for the linear layer.
            If None, auto-selected based on activation function.
            If DictConfig, instantiated automatically.
        """
        super().__init__()

        # ============================
        # Resolve Activation Function
        # ============================
        if activation is None:
            act = nn.GELU()
        elif isinstance(activation, DictConfig):
            act = instantiate(activation)
        else:
            act = activation

        # ============================
        # Resolve Normalization Layer
        # ============================
        if norm_type == 'layer':
            norm = nn.LayerNorm(out_features)
        elif norm_type == 'batch':
            norm = nn.BatchNorm1d(out_features)
        elif norm_type == 'none' or norm_type is None:
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type '{norm_type}'. Choose 'layer', 'batch', or 'none'.")

        # ============================
        # Create Basic Layers
        # ============================
        linear = nn.Linear(in_features, out_features)
        drop = nn.Dropout(dropout_rate)

        # ============================
        # Weight Initialization
        # ============================
        if init_func is not None:
            # Custom initialization function provided
            if isinstance(init_func, DictConfig):
                init_func_callable = instantiate(init_func)
                init_func_callable(linear)
            else:
                init_func(linear)
        else:
            # Auto-select initialization based on activation function
            auto_init = get_init_func(activation)
            auto_init(linear)

        # ============================
        # Build Block Sequence
        # ============================
        if residual:
            # Residual path: linear -> norm -> dropout
            # Activation applied after residual addition
            block = nn.Sequential(linear, norm, drop)
            # Projection handles dimension mismatch in skip connection
            projection = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else None
            self.model = nn.Sequential(Residual(block, projection=projection), act)
        else:
            # Standard path: linear -> norm -> activation -> dropout
            self.model = nn.Sequential(linear, norm, act, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        return self.model(x)


class MLPBlocks(nn.Module):
    """
    Sequence of stacked MLP blocks with configurable depth.

    Projects from in_features to out_features through multiple MLPBlock layers.
    Supports different activation and normalization for intermediate and final blocks.
    Enables residual connections across all blocks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        n_blocks: int = 1,
        activation: DictConfig | nn.Module | None = None,
        final_activation: DictConfig | nn.Module | None = None,
        norm_type: str | None = None,
        final_norm_type: str | None = None,
        dropout_rate: float = 0.0,
        final_dropout_rate: float | None = None,
        residual: bool = False,
        init_func: DictConfig | nn.Module | None = None,
    ) -> None:
        """
        Initialize a sequence of MLP blocks.

        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension of the final block.
        hidden_features : int or None, default=None
            Intermediate feature dimension for all non-final blocks.
            Defaults to in_features if None.
        n_blocks : int, default=1
            Total number of stacked linear blocks. At least 1.
        activation : DictConfig or nn.Module or None, default=None
            Intermediate activation function. Defaults to GELU if None.
            Applied to intermediate blocks.
        final_activation : DictConfig or nn.Module or None, default=None
            Activation for the final block. Falls back to activation if None.
            Pass nn.Identity() to suppress activation on the final block.
        norm_type : str or None, default=None
            Intermediate normalization type.
            Options: 'layer', 'batch', 'none'.
        final_norm_type : str or None, default=None
            Final block normalization type. Falls back to norm_type if None.
        dropout_rate : float, default=0.0
            Intermediate dropout probability. Range [0.0, 1.0].
        final_dropout_rate : float or None, default=None
            Final block dropout probability. Falls back to dropout_rate if None.
        residual : bool, default=False
            Whether to use residual connections within each block.
        init_func : DictConfig or Callable or None, default=None
            Weight initialization function for all blocks.
            If None, auto-selected based on activation per block.
        """
        super().__init__()

        # Store output dimension for external access
        self.out_features = out_features

        # Fall back to intermediate settings for final-layer overrides not explicitly provided
        hidden_features = hidden_features if hidden_features is not None else in_features
        final_act = final_activation if final_activation is not None else activation
        final_norm = final_norm_type if final_norm_type is not None else norm_type
        final_drop = final_dropout_rate if final_dropout_rate is not None else dropout_rate

        blocks = []
        current_in = in_features

        for i in range(n_blocks):
            is_final = (i == n_blocks - 1)
            # Final block projects to out_features; intermediate blocks use hidden_features
            current_out = out_features if is_final else hidden_features
            blocks.append(
                MLPBlock(
                    in_features=current_in,
                    out_features=current_out,
                    activation=final_act if is_final else activation,
                    norm_type=final_norm if is_final else norm_type,
                    dropout_rate=final_drop if is_final else dropout_rate,
                    residual=residual,
                    init_func=init_func,
                )
            )
            current_in = current_out

        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequential MLP block chain.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        return self.model(x)


class PredictionHead(nn.Module):
    """
    Single prediction head for outputting predictions.

    Can be a simple linear layer (n_layers=1) or a more complex MLP (n_layers>1).
    Typically used as the output layer of a neural network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        n_layers: int = 1,
        activation: DictConfig | nn.Module | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize a prediction head.

        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension (number of predictions).
        hidden_features : int or None, default=None
            Hidden layer dimension (used if n_layers > 1).
        n_layers : int, default=1
            Number of layers in the head. If 1, a simple linear layer.
            If > 1, uses MLPBlocks for multi-layer processing.
        activation : DictConfig or nn.Module or None, default=None
            Activation function (applied to intermediate layers, not final).
        dropout_rate : float, default=0.0
            Dropout probability for intermediate layers.
        """
        super().__init__()
        self.out_features = out_features

        if n_layers == 1:
            # Simple linear head (no activation, no dropout)
            self.head = nn.Linear(in_features, out_features)
        else:
            # Multi-layer head using MLPBlocks
            # No activation on final layer, no dropout on final layer
            self.head = MLPBlocks(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features or in_features,
                n_blocks=n_layers,
                activation=activation,
                final_activation=nn.Identity(),
                dropout_rate=dropout_rate,
                final_dropout_rate=0.0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the prediction head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        return self.head(x)


class PredictionHeads(nn.Module):
    """
    Multiple prediction heads with identical architectures.

    Creates n_heads identical PredictionHead modules for multi-task learning.
    Outputs from all heads are concatenated along the feature dimension.
    """

    def __init__(
        self,
        n_heads: int,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        n_layers: int = 1,
        activation: DictConfig | nn.Module | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize multiple prediction heads.

        Parameters
        ----------
        n_heads : int
            Number of prediction heads to create.
        in_features : int
            Input feature dimension (shared by all heads).
        out_features : int
            Output feature dimension per head.
        hidden_features : int or None, default=None
            Hidden layer dimension (if n_layers > 1).
        n_layers : int, default=1
            Number of layers in each head.
        activation : DictConfig or nn.Module or None, default=None
            Activation function for intermediate layers in each head.
        dropout_rate : float, default=0.0
            Dropout probability for each head.
        """
        super().__init__()
        self.n_heads = n_heads
        # Total output dimension is sum of all head outputs
        self.out_features = out_features * n_heads

        # Create n_heads identical prediction heads
        self.heads = nn.ModuleList([
            PredictionHead(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                n_layers=n_layers,
                activation=activation,
                dropout_rate=dropout_rate,
            )
            for _ in range(n_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all prediction heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Concatenated output from all heads, shape (..., out_features * n_heads).
        """
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)


class MLPModular(nn.Module):
    """
    Modular MLP with fully configurable components.

    A flexible architecture where all components (positional encoding, input layer,
    hidden layers, output layer) are specified via configuration. Supports skip
    connections, encoding injection, and flexible component combinations.

    This design enables easy experimentation with different architectural choices
    without changing code - all configuration is done via YAML/DictConfig.
    """

    def __init__(
        self,
        positional_encoding: DictConfig | nn.Module | None = None,
        input_layer: DictConfig | nn.Module | None = None,
        hidden_layer: DictConfig | nn.Module | None = None,
        output_layer: DictConfig | nn.Module | None = None,
        hidden_skip: bool = False,
        output_skip: bool = False,
        inject_encoding_hidden: bool = False,
        inject_encoding_output: bool = False,
    ) -> None:
        """
        Initialize MLPModular with configurable components.

        Parameters
        ----------
        positional_encoding : DictConfig or nn.Module or None, default=None
            Optional positional/coordinate encoding to prepend to input.
            If None, input is used as-is (no encoding).
        input_layer : DictConfig or nn.Module or None, default=None
            Input projection layer. Can be:
            - MLPBlock: single block
            - MLPBlocks: multiple blocks for multi-layer input
            - Any nn.Module for custom input processing
        hidden_layer : DictConfig or nn.Module or None, default=None
            Hidden layer(s) for feature extraction. Can be:
            - MLPBlock: single block (use hidden_skip=True for residual)
            - MLPBlocks: multiple blocks (control depth via n_blocks parameter)
            - Any nn.Module for custom hidden processing
        output_layer : DictConfig or nn.Module or None, default=None
            Output layer for predictions. Can be:
            - PredictionHead: single prediction head
            - PredictionHeads: multiple heads for multi-task learning
            - nn.Linear: simple linear layer
            - Any nn.Module for custom output
        hidden_skip : bool, default=False
            Whether to wrap hidden_layer with residual connection.
            Only applicable when hidden_layer is a single MLPBlock.
            MLPBlocks already has internal residuals if residual=True.
        output_skip : bool, default=False
            Whether to concatenate input_layer output to output_layer input.
            Useful for skip connections across the entire network.
        inject_encoding_hidden : bool, default=False
            Whether to concatenate positional encoding to hidden_layer input.
            Allows encoding information to flow at multiple points in the network.
        inject_encoding_output : bool, default=False
            Whether to concatenate positional encoding to output_layer input.
            Allows encoding information to directly influence output predictions.
        """
        super().__init__()

        # ============================
        # Positional Encoding
        # ============================
        if positional_encoding is not None:
            self.positional_encoding = instantiate(positional_encoding)
            self.d_encoding = self.positional_encoding.d_output
        else:
            self.positional_encoding = nn.Identity()
            self.d_encoding = 0

        # Store injection flags for forward pass
        self.inject_encoding_hidden = inject_encoding_hidden
        self.inject_encoding_output = inject_encoding_output

        # ============================
        # Input Layer
        # ============================
        self.input_layer = instantiate(input_layer) if input_layer is not None else nn.Identity()

        # Determine input_layer output dimension for subsequent layers
        if hasattr(self.input_layer, 'out_features'):
            input_out_dim = int(self.input_layer.out_features)
        elif hasattr(self.input_layer, 'model') and hasattr(self.input_layer.model, 'out_features'):
            input_out_dim = int(self.input_layer.model[-1].out_features)
        else:
            # Try to infer from config
            if isinstance(input_layer, DictConfig) and hasattr(input_layer, 'out_features'):
                input_out_dim = int(input_layer.out_features)
            else:
                raise ValueError("Unable to determine output dimension of input_layer.")

        # ============================
        # Hidden Layer
        # ============================
        if hidden_layer is not None:
            self.hidden_layer = instantiate(hidden_layer)
            
            # Update in_features if encoding injection is enabled
            if inject_encoding_hidden and hasattr(self.hidden_layer, 'in_features'):
                self.hidden_layer.in_features = input_out_dim + self.d_encoding

            # Optionally wrap with residual connection
            if hidden_skip:
                # Determine output dimension of hidden layer
                if hasattr(self.hidden_layer, 'out_features'):
                    hidden_out_dim = int(self.hidden_layer.out_features)
                elif isinstance(hidden_layer, DictConfig) and hasattr(hidden_layer, 'out_features'):
                    hidden_out_dim = int(hidden_layer.out_features)
                else:
                    hidden_out_dim = int(self.hidden_layer.in_features)
                
                # Add residual wrapper with optional projection for dimension mismatch
                layer_in_dim = input_out_dim + self.d_encoding if inject_encoding_hidden else input_out_dim
                if layer_in_dim != hidden_out_dim:
                    projection = nn.Linear(layer_in_dim, hidden_out_dim, bias=False)
                else:
                    projection = None
                self.hidden_layer = Residual(self.hidden_layer, projection=projection)
                current_dim = hidden_out_dim
            else:
                # Get output dimension without residual wrapper
                if hasattr(self.hidden_layer, 'out_features'):
                    current_dim = self.hidden_layer.out_features
                elif isinstance(hidden_layer, DictConfig) and hasattr(hidden_layer, 'out_features'):
                    current_dim = hidden_layer.out_features
                else:
                    current_dim = input_out_dim
        else:
            self.hidden_layer = nn.Identity()
            current_dim = input_out_dim

        # ============================
        # Output Layer Input Dimension
        # ============================
        self.output_skip = output_skip
        if output_skip:
            # Concatenate hidden output with input output
            output_in_dim = current_dim + input_out_dim
        else:
            output_in_dim = current_dim

        # Add encoding dimension if injection is enabled
        if inject_encoding_output:
            output_in_dim += self.d_encoding

        # ============================
        # Output Layer
        # ============================
        if output_layer is not None:
            self.output_layer = instantiate(output_layer) if isinstance(output_layer, DictConfig) else output_layer
            # Update input dimension if layer supports it
            if hasattr(self.output_layer, 'in_features'):
                self.output_layer.in_features = output_in_dim
        else:
            self.output_layer = nn.Identity()

        # Store dimensions for skip concatenation in forward pass
        self.input_out_dim = input_out_dim
        self.hidden_out_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the modular MLP.

        Data flow:
        1. Positional encoding (if configured)
        2. Input layer projection
        3. Hidden layer with optional encoding injection
        4. Output layer with optional skip connections and encoding injection

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor from output_layer.
        """
        # ============================
        # Positional Encoding
        # ============================
        x_enc = self.positional_encoding(x)

        # ============================
        # Input Layer
        # ============================
        x_input = self.input_layer(x_enc)

        # ============================
        # Hidden Layer
        # ============================
        if self.inject_encoding_hidden:
            # Concatenate encoding to hidden layer input
            x_hidden = self.hidden_layer(torch.cat([x_input, x_enc], dim=-1))
        else:
            x_hidden = self.hidden_layer(x_input)

        # ============================
        # Prepare Output Layer Input
        # ============================
        if self.output_skip:
            # Concatenate hidden and input features
            x_out = torch.cat([x_hidden, x_input], dim=-1)
        else:
            x_out = x_hidden

        # Optionally inject encoding at output
        if self.inject_encoding_output:
            x_out = torch.cat([x_out, x_enc], dim=-1)

        # ============================
        # Output Layer
        # ============================
        return self.output_layer(x_out)

