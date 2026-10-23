import torch
import torch.nn as nn
from typing import Callable
from omegaconf import DictConfig, ListConfig
from utilities.instantiators import instantiate
from code.architecture.encoding import IdentityPositionalEncoding
from code.architecture.wrapper import Residual


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

    Weights and biases sampled from U(-sqrt(6/n), sqrt(6/n)), where n = input_features,
    as specified in Sitzmann et al. (2020), Supplementary. With ω₀ scaling the bound
    is divided by ω₀; here ω₀ = 1 so the formula reduces to sqrt(6 / n).

    Parameters
    ----------
    layer : nn.Linear
        Linear layer to initialize.
    w0 : float, default=1.0
        Frequency multiplier ω₀ used to scale the bound (bound /= w0).
    """
    with torch.no_grad():
        in_features = layer.weight.shape[1]
        bound = (6.0 / in_features) ** 0.5
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


def is_siren_activation(activation: DictConfig | nn.Module | None) -> bool:
    """Return True if activation is a Sine/LearnableSine (SIREN) type."""
    if activation is None:
        return False
    if isinstance(activation, DictConfig):
        target = activation._target_ if hasattr(activation, '_target_') else ''
    else:
        target = activation.__class__.__name__
    return 'sine' in str(target).lower()


def get_init_func(activation: DictConfig | nn.Module | None, first_layer: bool = False) -> Callable:
    """
    Auto-select weight initialization function based on activation type.

    Automatically determines the most appropriate initialization function based on
    the activation function provided. This enables intelligent weight initialization
    that adapts to different activation types without manual configuration.

    Activation-to-Initialization Mapping:
    - Sine / LearnableSine (first layer): U(-1, 1)  (SIREN input-layer contract)
    - Sine / LearnableSine (hidden):      U(-√(6/n), √(6/n))  (Sitzmann et al., 2020)
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
    first_layer : bool, default=False
        If True and the activation is a Sine type, returns init_weights_siren_first
        (U(-1, 1)) instead of the hidden-layer variant.

    Returns
    -------
    Callable
        Weight initialization function appropriate for the given activation.
    """
    if activation is None:
        return init_weights_xavier

    # Extract target string from DictConfig or nn.Module
    if isinstance(activation, DictConfig):
        if hasattr(activation, '_target_'):
            target = activation._target_
        else:
            return init_weights_default
    else:
        target = activation.__class__.__name__

    target_str = str(target).lower()

    if 'sine' in target_str:
        return init_weights_siren_first if first_layer else init_weights_siren_hidden
    elif 'relu' in target_str or 'leaky' in target_str:
        return init_weights_he
    elif 'swish' in target_str or 'silu' in target_str:
        return init_weights_he
    elif 'gelu' in target_str:
        return init_weights_xavier
    elif 'tanh' in target_str:
        return init_weights_xavier
    elif 'sigmoid' in target_str:
        return init_weights_xavier
    elif 'scale' in target_str or 'snake' in target_str:
        return init_weights_xavier
    else:
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
        in_features: int | DictConfig,
        out_features: int | DictConfig,
        activation: DictConfig | None = None,
        norm_type: str | None = None,
        dropout_rate: float = 0.0,
        residual: bool = False,
        init_func: DictConfig | nn.Module | None = None,
        first_layer: bool = False,
    ) -> None:
        """
        Initialize a single MLP block.

        Parameters
        ----------
        in_features : int
            Input feature dimension.
        out_features : int
            Output feature dimension.
        activation : DictConfig or None, default=None
            Activation function as a DictConfig with '_target_'. If None, defaults to GELU.
            Always pass DictConfig to ensure independent activation instances across blocks.
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
        first_layer : bool, default=False
            If True and using a SIREN activation, applies the first-layer SIREN
            initialization U(-1, 1) instead of the hidden-layer variant. Has no
            effect for non-SIREN activations or when init_func is provided.
        """

        # Class inheritance
        super().__init__()

        # Determine in_features and out_features if provided as DictConfig
        if isinstance(in_features, DictConfig):
            in_features = instantiate(in_features)
        if isinstance(out_features, DictConfig):
            out_features = instantiate(out_features)

        # Resolve Activation Function
        if activation is None:
            act = nn.Identity()  # No activation if None
        else:
            # activation must be DictConfig - instantiate to ensure independence
            act = instantiate(activation)

        # Resolve Normalization Layer
        if norm_type == 'layer':
            norm = nn.LayerNorm(out_features)
        elif norm_type == 'batch':
            norm = nn.BatchNorm1d(out_features)
        elif norm_type == 'none' or norm_type is None:
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type '{norm_type}'. Choose 'layer', 'batch', or 'none'.")

        # Create Basic Layers
        linear = nn.Linear(in_features, out_features)
        drop = nn.Dropout(dropout_rate)

        # Weight Initialization
        if init_func is not None:
            # Custom initialization function provided
            if isinstance(init_func, DictConfig):
                init_func_callable = instantiate(init_func)
                init_func_callable(linear)
            else:
                init_func(linear)
        else:
            # Auto-select initialization based on activation function and layer position
            auto_init = get_init_func(activation, first_layer=first_layer)
            auto_init(linear)

        # Build Block Sequence — same operation order for both paths:
        #   Linear -> Norm -> Activation -> Dropout
        # For the residual path this sits inside F(x), giving out = x + F(x).
        # Activation is therefore inside the transformation, not after the add,
        # which preserves near-identity at initialisation (F(x) ≈ 0 → out ≈ x).
        if residual:
            block = nn.Sequential(linear, norm, act, drop)
            if in_features != out_features:
                projection = nn.Linear(in_features, out_features, bias=False)
                auto_init(projection)
            else:
                projection = None
            self.model = Residual(block, projection=projection)
        else:
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
        in_features: int | DictConfig,
        out_features: int | DictConfig,
        hidden_features: int | DictConfig | None = None,
        n_blocks: int = 1,
        activation: DictConfig | None = None,
        final_activation: DictConfig | None = None,
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
        activation : DictConfig or None, default=None
            Intermediate activation function as DictConfig. Defaults to GELU if None.
            Always pass DictConfig to ensure independent activation instances.
            Applied to intermediate blocks.
        final_activation : DictConfig or None, default=None
            Activation for the final block as DictConfig. Falls back to activation if None.
            Pass nn.Identity as DictConfig to suppress activation on the final block.
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

        # Class inheritance
        super().__init__()

        # Resolve in_features and out_features if provided as DictConfig
        if isinstance(in_features, DictConfig):
            in_features = instantiate(in_features)
        if isinstance(out_features, DictConfig):
            out_features = instantiate(out_features)

        # Store output dimension for external access
        self.out_features = out_features

        # Fall back to intermediate settings for final-layer overrides not explicitly provided
        hidden_features = hidden_features if hidden_features is not None else in_features
        if isinstance(hidden_features, DictConfig):
            hidden_features = instantiate(hidden_features)
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
                    first_layer=(i == 0),
                )
            )
            current_in = current_out
        self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequential MLP chain of blocks.

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


class PredictionHead(MLPBlocks):
    """
    Single prediction head for outputting predictions.

    Inherits from MLPBlocks and adds optional post-processing capability.
    Supports arbitrary depth (n_blocks >= 1) with configurable final activation.

    Post-processing is applied after all MLP layers to transform outputs
    (e.g., denormalization, coordinate transformations, etc.).
    """

    def __init__(
        self,
        in_features: int | DictConfig,
        out_features: int | DictConfig,
        hidden_features: int | DictConfig | None = None,
        n_blocks: int = 1,
        activation: DictConfig | None = None,
        final_activation: DictConfig | None = None,
        norm_type: str | None = None,
        final_norm_type: str | None = None,
        dropout_rate: float = 0.0,
        final_dropout_rate: float | None = None,
        residual: bool = False,
        init_func: DictConfig | nn.Module | None = None,
        post_process: DictConfig | Callable | None = None,
    ) -> None:
        """
        Initialize a prediction head.

        Parameters
        ----------
        in_features : int | DictConfig
            Input feature dimension.
        out_features : int | DictConfig
            Output feature dimension (number of predictions).
        hidden_features : int | DictConfig | None, default=None
            Hidden layer dimension (used if n_blocks > 1).
        n_blocks : int, default=1
            Number of layers in the head. At least 1.
        activation : DictConfig or None, default=None
            Intermediate activation function as DictConfig. Applied to non-final layers.
            Always pass DictConfig to ensure independent activation instances.
        final_activation : DictConfig or None, default=None
            Activation for the final layer as DictConfig. Falls back to activation if None.
            Pass OmegaConf.create({'_target_': 'torch.nn.Identity'}) to suppress activation.
        norm_type : str or None, default=None
            Intermediate normalization type. Options: 'layer', 'batch', 'none'.
        final_norm_type : str or None, default=None
            Final block normalization type. Falls back to norm_type if None.
        dropout_rate : float, default=0.0
            Intermediate dropout probability. Range [0.0, 1.0].
        final_dropout_rate : float or None, default=None
            Final block dropout probability. Falls back to dropout_rate if None.
        residual : bool, default=False
            Whether to use residual connections within each block.
        init_func : DictConfig | Callable or None, default=None
            Weight initialization function for all blocks.
            If None, auto-selected based on activation per block.
        post_process : DictConfig | Callable, optional
            Post-processing function applied after MLP layers.
            If DictConfig, instantiated via instantiate().
        """

        # Initialize MLPBlocks with all parameters
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            n_blocks=n_blocks,
            activation=activation,
            final_activation=final_activation,
            norm_type=norm_type,
            final_norm_type=final_norm_type,
            dropout_rate=dropout_rate,
            final_dropout_rate=final_dropout_rate,
            residual=residual,
            init_func=init_func,
        )

        # Post-processing function (denormalization, transformations, etc.)
        if post_process is not None:
            if isinstance(post_process, DictConfig):
                self.post_process = instantiate(post_process)
            else:
                self.post_process = post_process
        else:
            self.post_process = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the prediction head.

        Applies MLPBlocks forward pass, then optional post-processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
            If post_process is configured, applied after MLP layers.
        """

        # Forward through parent MLPBlocks
        x = super().forward(x)
        # Apply post-processing
        if self.post_process is not None:
            x = self.post_process(x)
        return x


class HeterogeneousPredictionHeads(nn.Module):
    """
    Multiple prediction heads with heterogeneous (per-head) configurations.

    Each head can have different architectures, activations, layers, dropout rates, etc.

    Can be initialized in two ways:
    1. Via `heads` (ListConfig of DictConfigs) - configs are instantiated into heads
    2. Via `heads` (pre-constructed nn.ModuleList) - heads are used directly

    This design allows both declarative config-driven initialization and programmatic
    construction. Outputs from all heads can be either concatenated or returned as a tuple.
    """

    def __init__(
        self,
        heads: ListConfig | nn.ModuleList,
        out_concat: bool = True,
    ) -> None:
        """
        Initialize heterogeneous prediction heads.

        Parameters
        ----------
        heads : ListConfig[DictConfig] | nn.ModuleList
            Either:
            - ListConfig of DictConfigs with `_target_: code.architecture.mlp.PredictionHead`.
              Each head config can specify:
              - in_features : int
                  Input feature dimension for this head.
              - out_features : int
                  Output feature dimension for this head.
              - hidden_features : int, optional
                  Hidden layer dimension (if n_blocks > 1).
              - n_blocks : int, default=1
                  Number of layers in this head.
              - activation : DictConfig or None, optional
                  Activation function as DictConfig.
              - dropout_rate : float, default=0.0
                  Dropout probability for this head.
              - post_process : DictConfig | Callable, optional
                  Post-processing function (e.g., denormalization, transformations).
            - Pre-constructed nn.ModuleList of prediction heads to use directly.
        out_concat : bool, default=True
            If True, concatenates outputs from all heads along the feature dimension.
            If False, returns outputs as a tuple of tensors.
        """

        # Class inheritance
        super().__init__()

        # Store output format behavior
        self.out_concat = out_concat

        # Handle ListConfig (config-driven initialization)
        if isinstance(heads, ListConfig):
            # Instantiate each head config (expects _target_: code.architecture.mlp.PredictionHead)
            self.heads = nn.ModuleList([instantiate(head_cfg) for head_cfg in heads])
            # Calculate total output dimension from instantiated heads
            self.out_features = sum(
                h.out_features for h in self.heads
                if hasattr(h, 'out_features')
            )

        # Handle nn.ModuleList (pre-constructed heads)
        elif isinstance(heads, nn.ModuleList):
            self.heads = heads
            # Calculate total output dimension
            self.out_features = sum(
                h.out_features for h in self.heads
                if hasattr(h, 'out_features')
            )
        else:
            raise TypeError(
                f"heads must be ListConfig or nn.ModuleList, got {type(heads)}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass through all prediction heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            If out_concat=True: Concatenated output from all heads, shape (..., out_features_total).
            If out_concat=False: Tuple of tensors, one per head.
        """
        outputs = [head(x) for head in self.heads]
        if self.out_concat:
            return torch.cat(outputs, dim=-1)
        else:
            return tuple(outputs)


class HomogeneousPredictionHeads(nn.Module):
    """
    Multiple prediction heads with identical architectures.

    Creates n_heads identical prediction head modules from a single head configuration.
    Useful for multitask learning where all heads share the same architecture but have
    independent weights.

    Each head configuration is instantiated n times via `instantiate()` to ensure every
    head is a distinct module with independent weights (no parameter sharing).

    Outputs from all heads can be either concatenated or returned as a tuple.
    """

    def __init__(
        self,
        n_heads: int,
        head: DictConfig,
        out_concat: bool = True,
    ) -> None:
        """
        Initialize multiple identical prediction heads.

        Parameters
        ----------
        n_heads : int
            Number of prediction heads to create. Each head is instantiated independently.
        head : DictConfig
            Head configuration with `_target_`.
            Note: Must be DictConfig to ensure each head gets independent weights.
            If you need to use a pre-constructed module, use HeterogeneousPredictionHeads
            with pre-constructed heads passed as nn.ModuleList.
        out_concat : bool, default=True
            If True, concatenates outputs from all heads along the feature dimension.
            If False, returns outputs as a tuple of tensors.
        """

        # Class inheritance
        super().__init__()

        # Store output format behavior
        self.out_concat = out_concat

        # Create n_heads independent head instances
        if isinstance(n_heads, DictConfig):
            n_heads = instantiate(n_heads)
        if isinstance(head, DictConfig):
            # Each instantiate() call creates independent head with fresh weights
            self.heads = nn.ModuleList([instantiate(head) for _ in range(n_heads)])
        else:
            raise TypeError(
                f"HomogeneousPredictionHeads requires DictConfig (got {type(head)}). "
                f"To use a pre-constructed module, use HeterogeneousPredictionHeads with heads as nn.ModuleList."
            )

        # Calculate total output dimension from instantiated heads
        self.out_features = sum(
            h.out_features for h in self.heads
            if hasattr(h, 'out_features')
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass through all prediction heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            If out_concat=True: Concatenated output from all heads, shape (..., out_features_total).
            If out_concat=False: Tuple of tensors, one per head.
        """
        outputs = [head(x) for head in self.heads]
        if self.out_concat:
            return torch.cat(outputs, dim=-1)
        else:
            return tuple(outputs)


class MLPModular0(nn.Module):
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
        pre_process: DictConfig | nn.Module | None = None,
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
        pre_process : DictConfig or nn.Module or None, default=None
            Optional preprocessing layer applied to raw input before positional encoding.
            Can be used for normalization, transformations, or other input preprocessing.
            If None, input is passed directly to positional encoding.
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
            - PredictionHeads: multiple heads for multitask learning
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

        # Class inheritance
        super().__init__()

        # Preprocessing Layer (applied first to raw input)
        if isinstance(pre_process, DictConfig):
            self.pre_process = instantiate(pre_process)
        elif isinstance(pre_process, nn.Module):
            self.pre_process = pre_process
        else:
            self.pre_process = nn.Identity()

        # Positional Encoding
        if isinstance(positional_encoding, DictConfig):
            self.positional_encoding = instantiate(positional_encoding)
            self.d_encoding = self.positional_encoding.d_output
        elif isinstance(positional_encoding, nn.Module):
            self.positional_encoding = positional_encoding
            self.d_encoding = getattr(positional_encoding, 'd_output', 1)
        else:
            self.positional_encoding = nn.Identity()
            self.d_encoding = 0

        # Store injection flags for forward pass
        self.inject_encoding_hidden = inject_encoding_hidden
        self.inject_encoding_output = inject_encoding_output

        # Input Layer
        if isinstance(input_layer, DictConfig):
            # Adjust in_features to account for positional encoding
            if hasattr(input_layer, 'in_features') and input_layer.in_features is None:
                input_layer.in_features = self.d_encoding
            self.input_layer = instantiate(input_layer)
        elif isinstance(input_layer, nn.Module):
            self.input_layer = input_layer
        else:
            self.input_layer = nn.Identity()

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

        # Hidden Layer
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
        # Nothing; identity layer
        else:
            self.hidden_layer = nn.Identity()
            current_dim = input_out_dim

        # Output Layer Input Dimension
        if output_skip:
            # Concatenate hidden output with input output
            output_in_dim = current_dim + input_out_dim
        else:
            output_in_dim = current_dim

        # Add encoding dimension if injection is enabled
        if inject_encoding_output:
            output_in_dim += self.d_encoding

        # Output Layer
        if output_layer is not None:
            # Patch in_features BEFORE instantiation.
            # For layers with a top-level in_features (e.g. PredictionHead, MLPBlocks):
            #   mutate the DictConfig directly.
            # For HomogeneousPredictionHeads / HeterogeneousPredictionHeads, which nest
            #   in_features under 'head', Hydra may deep-copy the config before resolving
            #   interpolations, making DictConfig mutation unreliable. Instead, pass
            #   in_features as a kwarg override to instantiate() — the class accepts it
            #   and propagates it to each head internally.
            if isinstance(output_layer, DictConfig):
                if hasattr(output_layer, 'in_features'):
                    output_layer.in_features = output_in_dim
                    self.output_layer = instantiate(output_layer)
                elif hasattr(output_layer, 'head'):
                    self.output_layer = instantiate(output_layer, in_features=output_in_dim)
                else:
                    self.output_layer = instantiate(output_layer)
            else:
                self.output_layer = output_layer
        else:
            self.output_layer = nn.Identity()

        # Store dimensions for forward pass
        self.input_out_dim = input_out_dim
        self.hidden_out_dim = current_dim
        self.output_skip = output_skip

    def forward(self, x: torch.Tensor | dict) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass through the modular MLP.

        Data flow:
        1. Preprocessing layer (if configured)
        2. Positional encoding (if configured)
        3. Input layer projection
        4. Hidden layer with optional encoding injection
        5. Output layer with optional skip connections and encoding injection

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            Output from output_layer. Can be either a single tensor or a tuple of tensors
            if the output_layer returns multiple outputs.
        """

        # Preprocessing Layer
        x = self.pre_process(x)

        # Concatenate input
        if isinstance(x, dict):
            x = [
                x.reshape(x.size(0), -1) if x.dim() > 2 else x
                for x in x.values()
            ]
            x = torch.cat(x, dim=-1)

        # Positional Encoding
        x_enc = self.positional_encoding(x)

        # Input Layer
        x_input = self.input_layer(x_enc)

        # Hidden Layer
        if self.inject_encoding_hidden:
            # Concatenate encoding to hidden layer input
            x_hidden = self.hidden_layer(torch.cat([x_input, x_enc], dim=-1))
        else:
            x_hidden = self.hidden_layer(x_input)

        # Prepare Output Layer Input
        if self.output_skip:
            # Concatenate hidden layer output with input layer output
            x_out = torch.cat([x_hidden, x_input], dim=-1)
        else:
            x_out = x_hidden

        # Optionally inject encoding at output
        if self.inject_encoding_output:
            x_out = torch.cat([x_out, x_enc], dim=-1)

        # Output Layer
        return self.output_layer(x_out)


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
        pre_process: DictConfig | nn.Module | None = None,
        positional_encoding: DictConfig | nn.Module | None = None,
        input_layer: DictConfig | nn.Module | None = None,
        hidden_layer: DictConfig | nn.Module | None = None,
        output_layer: DictConfig | nn.Module | None = None,
        input_skip_hidden: bool = False,
        input_skip_output: bool = False,
        enc_skip_hidden: bool = False,
        enc_skip_output: bool = False,
    ) -> None:
        """
        Initialize MLPModular with configurable components.

        Parameters
        ----------
        pre_process : DictConfig or nn.Module or None, default=None
            Optional preprocessing layer applied to raw input before positional encoding.
            If None, input is passed directly to positional encoding.
        positional_encoding : DictConfig or nn.Module or None, default=None
            Optional positional/coordinate encoding. If None, nn.Identity() is used
            and d_encoding is set to None (enc_skip_* flags must both be False, and
            input_layer.in_features must be set explicitly).
        input_layer : DictConfig or nn.Module or None, default=None
            Input projection layer (e.g. MLPBlock or MLPBlocks).
        hidden_layer : DictConfig or nn.Module or None, default=None
            Hidden layer(s) for feature extraction (e.g. MLPBlocks).
        output_layer : DictConfig or nn.Module or None, default=None
            Output layer for predictions (e.g. PredictionHead, HomogeneousPredictionHeads).
        input_skip_hidden : bool, default=False
            Wrap hidden_layer in a macro residual connection:
            ``x_hidden = hidden_layer(x_input) + projection(x_input)``.
            Distinct from internal per-block residuals inside MLPBlocks.
        input_skip_output : bool, default=False
            Concatenate input_layer output to output_layer input (long skip):
            ``x_out = cat([x_hidden, x_input])``.
        enc_skip_hidden : bool, default=False
            Concatenate positional encoding to hidden_layer input (NeRF-style):
            ``x_hidden = hidden_layer(cat([x_input, x_enc]))``.
            Requires positional_encoding to be provided.
        enc_skip_output : bool, default=False
            Concatenate positional encoding to output_layer input:
            ``x_out = cat([x_out, x_enc])``.
            Requires positional_encoding to be provided.
        """

        # Class inheritance
        super().__init__()

        # Preprocessing Layer (applied first to raw input)
        if isinstance(pre_process, DictConfig):
            self.pre_process = instantiate(pre_process)
        elif isinstance(pre_process, nn.Module):
            self.pre_process = pre_process
        else:
            self.pre_process = nn.Identity()

        # Positional Encoding
        self.d_encoding = 0
        if isinstance(positional_encoding, DictConfig):
            self.positional_encoding = instantiate(positional_encoding)
            self.d_encoding = int(self.positional_encoding.d_output)
        elif isinstance(positional_encoding, nn.Module):
            self.positional_encoding = positional_encoding
            self.d_encoding = int(getattr(positional_encoding, 'd_output', 1))
        else:
            # No encoding: identity pass-through. d_encoding is unknown until
            # the first forward call, so encoding-dependent features are disabled.
            self.positional_encoding = nn.Identity()

        # Guard: enc_skip_* require a concrete d_encoding
        if (enc_skip_hidden or enc_skip_output) and self.d_encoding == 0:
            raise ValueError(
                "enc_skip_hidden and enc_skip_output require a positional_encoding "
                "with a d_output attribute. Provide a positional_encoding config or "
                "set both enc_skip flags to False."
            )

        # Store skip flags
        self.enc_skip_hidden = enc_skip_hidden
        self.enc_skip_output = enc_skip_output
        self.input_skip_hidden = input_skip_hidden
        self.input_skip_output = input_skip_output

        # Input Layer
        if isinstance(input_layer, DictConfig):
            if self.d_encoding != 0:
                input_layer.in_features = self.d_encoding
            elif hasattr(input_layer, 'in_features') and input_layer.in_features is None:
                raise ValueError(
                    "input_layer.in_features is None but no positional_encoding was provided "
                    "to infer it from. Set input_layer.in_features explicitly."
                )
            self.input_layer = instantiate(input_layer)
        elif isinstance(input_layer, nn.Module):
            self.input_layer = input_layer
        else:
            self.input_layer = nn.Identity()

        # Determine input_layer output dimension for subsequent layers
        if hasattr(self.input_layer, 'out_features'):
            input_out_dim = int(self.input_layer.out_features)
        elif hasattr(self.input_layer, 'model') and hasattr(self.input_layer.model, 'out_features'):
            input_out_dim = int(self.input_layer.model[-1].out_features)
        elif isinstance(self.input_layer, nn.Identity):
            input_out_dim = self.d_encoding
        else:
            # Try to infer from config
            if isinstance(input_layer, DictConfig) and hasattr(input_layer, 'out_features'):
                input_out_dim = int(input_layer.out_features)
            else:
                raise ValueError("Unable to determine output dimension of input_layer.")

        # Hidden Layer
        if hidden_layer is not None:
            # Patch in_features BEFORE instantiation, same pattern as input/output layers
            if enc_skip_hidden and isinstance(hidden_layer, DictConfig) and hasattr(hidden_layer, 'in_features'):
                hidden_layer.in_features = input_out_dim + self.d_encoding

            self.hidden_layer = instantiate(hidden_layer)

            # Optionally wrap with residual connection
            if input_skip_hidden:
                # Determine output dimension of hidden layer
                if hasattr(self.hidden_layer, 'out_features'):
                    hidden_out_dim = int(self.hidden_layer.out_features)
                elif isinstance(hidden_layer, DictConfig) and hasattr(hidden_layer, 'out_features'):
                    hidden_out_dim = int(hidden_layer.out_features)
                else:
                    hidden_out_dim = int(self.hidden_layer.in_features)

                # Add residual wrapper with optional projection for dimension mismatch
                layer_in_dim = input_out_dim + self.d_encoding if enc_skip_hidden else input_out_dim
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
        # Nothing; identity layer
        else:
            self.hidden_layer = nn.Identity()
            current_dim = input_out_dim

        # Output Layer Input Dimension
        if input_skip_output:
            output_in_dim = current_dim + input_out_dim
        else:
            output_in_dim = current_dim

        if enc_skip_output:
            output_in_dim += self.d_encoding

        # Output Layer
        if output_layer is not None:
            # Set in_features in config BEFORE instantiation (important for nested layers)
            if isinstance(output_layer, DictConfig):
                if hasattr(output_layer, 'in_features'):
                    output_layer.in_features = output_in_dim
                elif hasattr(output_layer, 'head'):
                    output_layer.head.in_features = output_in_dim
                elif hasattr(output_layer, 'heads'):
                    for head_cfg in output_layer.heads:
                        if hasattr(head_cfg, 'in_features'):
                            head_cfg.in_features = output_in_dim

            self.output_layer = instantiate(output_layer) if isinstance(output_layer, DictConfig) else output_layer

        else:
            self.output_layer = nn.Identity()

        # Store dimensions for forward pass
        self.input_out_dim = input_out_dim
        self.hidden_out_dim = current_dim

    def forward(self, x: torch.Tensor | dict) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass through the modular MLP.

        Data flow:
        1. Preprocessing layer (if configured)
        2. Positional encoding (if configured)
        3. Input layer projection
        4. Hidden layer with optional encoding injection
        5. Output layer with optional skip connections and encoding injection

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, ...]
            Output from output_layer. Can be either a single tensor or a tuple of tensors
            if the output_layer returns multiple outputs.
        """

        # Preprocessing Layer
        x = self.pre_process(x)

        # Concatenate input
        if isinstance(x, dict):
            x = [
                x.reshape(x.size(0), -1) if x.dim() > 2 else x
                for x in x.values()
            ]
            x = torch.cat(x, dim=-1)

        # Positional Encoding
        x_enc = self.positional_encoding(x)

        # Input Layer
        x_input = self.input_layer(x_enc)

        # Hidden Layer
        if self.enc_skip_hidden:
            x_hidden = self.hidden_layer(torch.cat([x_input, x_enc], dim=-1))
        else:
            x_hidden = self.hidden_layer(x_input)

        # Prepare Output Layer Input
        if self.input_skip_output:
            x_out = torch.cat([x_hidden, x_input], dim=-1)
        else:
            x_out = x_hidden

        # Optionally inject encoding at output
        if self.enc_skip_output:
            x_out = torch.cat([x_out, x_enc], dim=-1)

        # Output Layer
        return self.output_layer(x_out)


class MLPModulars(nn.Module):
    """
    Parallel MLPs with a shared positional encoding (Cerberus-style).

    A single positional encoding maps raw coordinates to encoded features,
    which are then passed to N independent MLPModular sub-networks in parallel.
    Each sub-network produces its own output; results are concatenated along
    the feature dimension.

    Designed for multi-variable neural fields where variables share the same
    coordinate space but benefit from independent representations (e.g. separate
    networks for temperature, humidity, and ozone), with inter-variable coupling
    enforced through the loss rather than through shared weights.

    Data flow::

        x -> pre_process -> positional_encoding -> x_enc  (shared)
                 |-- MLPModular_0(x_enc) -> y_0
                 |-- MLPModular_1(x_enc) -> y_1
                 `-- MLPModular_{N-1}(x_enc) -> y_{N-1}
                              cat(dim=-1) -> y  (B, N * out_features_per_model)

    Notes
    -----
    Sub-models receive ``x_enc`` directly as a tensor, so their own
    ``positional_encoding`` should be ``null`` in the sub-model config.
    If a sub-model specifies its own positional encoding it will be applied
    on top of the shared one, which is almost certainly unintended.

    ``input_layer.in_features`` in the sub-model config should be ``null``;
    ``MLPModulars`` patches it to ``d_encoding`` before instantiation so that
    ``MLPModular`` does not overwrite it with zero (which it would, because the
    sub-model has no positional encoding and therefore ``d_encoding=0``).
    """

    def __init__(
        self,
        n_models: int | DictConfig,
        model: DictConfig,
        pre_process: DictConfig | nn.Module | None = None,
        positional_encoding: DictConfig | nn.Module | None = None,
        out_concat: bool = True,
    ) -> None:
        """
        Initialize MLPModulars.

        Parameters
        ----------
        n_models : int
            Number of parallel sub-networks to instantiate.
            Typically equals the number of output variables.
        model : DictConfig
            Configuration for each sub-network. Must have ``_target_`` pointing
            to ``MLPModular`` (or a compatible class). Each call to
            ``instantiate(model)`` produces an independent module with fresh
            weights. Set ``positional_encoding: null`` and
            ``input_layer.in_features: null`` in the sub-model config.
        pre_process : DictConfig or nn.Module or None, default=None
            Shared preprocessing applied before positional encoding.
        positional_encoding : DictConfig or nn.Module or None, default=None
            Shared positional encoding. Its ``d_output`` attribute is used to
            patch ``input_layer.in_features`` in each sub-model.
        out_concat : bool, default=True
            If True, concatenates sub-model outputs along the last dimension.
            If False, returns a tuple of tensors.
        """

        super().__init__()

        # Shared preprocessing
        if isinstance(pre_process, DictConfig):
            self.pre_process = instantiate(pre_process)
        elif isinstance(pre_process, nn.Module):
            self.pre_process = pre_process
        else:
            self.pre_process = nn.Identity()

        # Shared positional encoding
        if isinstance(positional_encoding, DictConfig):
            self.positional_encoding = instantiate(positional_encoding)
            self.d_encoding = self.positional_encoding.d_output
        elif isinstance(positional_encoding, nn.Module):
            self.positional_encoding = positional_encoding
            self.d_encoding = getattr(positional_encoding, 'd_output', 1)
        else:
            self.positional_encoding = nn.Identity()
            self.d_encoding = 0

        if isinstance(n_models, DictConfig):
            n_models = instantiate(n_models)

        self.out_concat = out_concat

        if not isinstance(model, DictConfig):
            raise TypeError(f"model must be a DictConfig, got {type(model)}")

        # Patch input_layer.in_features to d_encoding before any instantiation.
        # MLPModular only overwrites in_features when it is None, so setting it
        # here prevents the sub-model from clobbering it with its own d_encoding=0
        # (which it would compute because its positional_encoding is null).
        if (
            hasattr(model, 'input_layer')
            and model.input_layer is not None
            and hasattr(model.input_layer, 'in_features')
            and model.input_layer.in_features is None
        ):
            model.input_layer.in_features = self.d_encoding

        # Each instantiate() call produces an independent module with fresh weights
        self.models = nn.ModuleList([instantiate(model) for _ in range(n_models)])

    def forward(self, x: torch.Tensor | dict) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass: shared encoding then parallel sub-networks.

        Parameters
        ----------
        x : torch.Tensor or dict
            Raw input coordinates. If a dict, tensors are concatenated along
            the feature dimension before encoding.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, ...]
            If ``out_concat=True``: concatenated outputs, shape
            ``(batch, N * out_features_per_model)``.
            If ``out_concat=False``: tuple of per-model tensors.
        """

        # Shared preprocessing
        x = self.pre_process(x)
        if isinstance(x, dict):
            x = torch.cat(
                [v.reshape(v.size(0), -1) if v.dim() > 2 else v for v in x.values()],
                dim=-1,
            )

        # Shared positional encoding
        x_enc = self.positional_encoding(x)

        # Parallel sub-networks — each receives the same x_enc
        outputs = [model(x_enc) for model in self.models]

        if self.out_concat:
            return torch.cat(outputs, dim=-1)
        return tuple(outputs)
