"""
MLP building blocks for RACCOONN.
Adapted from 3DClouds repository with RACCOONN-specific modifications.
Includes modular architectures and multi-head prediction support.
"""
import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from inverse.model.architecture.wrapper import Residual


class MLPBlock(nn.Module):
    """Single linear block with optional normalization, dropout, and residual connection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Union[DictConfig, nn.Module] = None,
        norm_type: str = 'none',
        dropout_rate: float = 0.0,
        residual: bool = False,
        init_func: callable = None,
    ) -> None:
        """
        Initialize a single MLP block.

        Parameters
        ----------
        in_features : int. Input feature dimension.
        out_features : int. Output feature dimension.
        activation : DictConfig or nn.Module or None. Activation function.
                     Defaults to GELU if None.
        norm_type : str. Normalization type applied after the linear layer.
                    Options: 'layer' (LayerNorm), 'batch' (BatchNorm1d), 'none' (Identity).
        dropout_rate : float. Dropout probability applied after activation.
        residual : bool. Whether to add a skip connection. A learned projection
                   is added automatically when in_features differs from out_features.
        init_func : callable. Optional custom initialization function for the linear layer.
        """
        super().__init__()

        # Resolve activation: DictConfig -> instantiate, None -> default GELU, else use as-is
        if activation is None:
            act = nn.GELU()
        elif isinstance(activation, DictConfig):
            act = instantiate(activation)
        else:
            act = activation

        # Normalization layer applied after the linear projection
        if norm_type == 'layer':
            norm = nn.LayerNorm(out_features)
        elif norm_type == 'batch':
            norm = nn.BatchNorm1d(out_features)
        elif norm_type == 'none':
            norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm_type '{norm_type}'. Choose 'layer', 'batch', or 'none'.")

        linear = nn.Linear(in_features, out_features)
        drop = nn.Dropout(dropout_rate)

        # Weight initialization
        if init_func is not None:
            init_func(linear)
        else:
            # Default Xavier initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        # Build the block sequence, with or without skip connection
        if residual:
            # Residual: norm and activation are applied after the skip addition
            block = nn.Sequential(linear, norm, drop)
            projection = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else None
            self.model = nn.Sequential(Residual(block, projection=projection), act)
        else:
            self.model = nn.Sequential(linear, norm, act, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP block.

        Parameters
        ----------
        x : torch.Tensor. Shape (..., in_features).

        Returns
        -------
        x : torch.Tensor. Shape (..., out_features).
        """
        return self.model(x)


class MLPBlocks(nn.Module):
    """Sequence of MLP blocks projecting from in_features to out_features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        n_blocks: int = 1,
        activation: Union[DictConfig, nn.Module] = None,
        final_activation: Union[DictConfig, nn.Module] = None,
        norm_type: str = 'none',
        final_norm_type: str = None,
        dropout_rate: float = 0.0,
        final_dropout_rate: float = None,
        residual: bool = False,
        init_func: callable = None,
    ) -> None:
        """
        Initialize a sequence of MLP blocks.

        Parameters
        ----------
        in_features : int. Input feature dimension.
        out_features : int. Output feature dimension of the final block.
        hidden_features : int or None. Intermediate feature dimension for all
                          non-final blocks. Defaults to in_features.
        n_blocks : int. Total number of stacked linear blocks.
        activation : DictConfig or nn.Module or None. Intermediate activation.
                     Defaults to GELU if None.
        final_activation : DictConfig or nn.Module or None. Activation for the
                           last block. Falls back to activation when None.
                           Pass nn.Identity() to suppress activation on the final block.
        norm_type : str. Intermediate normalization type. Options: 'layer', 'batch', 'none'.
        final_norm_type : str or None. Final block normalization type.
                          Falls back to norm_type when None.
        dropout_rate : float. Intermediate dropout probability.
        final_dropout_rate : float or None. Final block dropout probability.
                             Falls back to dropout_rate when None.
        residual : bool. Whether to use residual connections within each block.
        init_func : callable. Optional custom initialization function for linear layers.
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
        x : torch.Tensor. Shape (..., in_features).

        Returns
        -------
        x : torch.Tensor. Shape (..., out_features).
        """
        return self.model(x)


class PredictionHead(nn.Module):
    """Single prediction head (can be a simple linear layer or a more complex MLP)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        n_layers: int = 1,
        activation: Union[DictConfig, nn.Module] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize a prediction head.

        Parameters
        ----------
        in_features : int. Input feature dimension.
        out_features : int. Output feature dimension.
        hidden_features : int. Hidden layer dimension (if n_layers > 1).
        n_layers : int. Number of layers in the head.
        activation : DictConfig or nn.Module. Activation function.
        dropout_rate : float. Dropout rate.
        """
        super().__init__()
        self.out_features = out_features

        if n_layers == 1:
            # Simple linear head
            self.head = nn.Linear(in_features, out_features)
        else:
            # Multi-layer head
            self.head = MLPBlocks(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features or in_features,
                n_blocks=n_layers,
                activation=activation,
                final_activation=nn.Identity(),  # No activation on final layer
                dropout_rate=dropout_rate,
                final_dropout_rate=0.0,  # No dropout on final layer
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the prediction head."""
        return self.head(x)


class PredictionHeads(nn.Module):
    """Multiple prediction heads with identical architectures."""

    def __init__(
        self,
        n_heads: int,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        n_layers: int = 1,
        activation: Union[DictConfig, nn.Module] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Initialize multiple prediction heads.

        Parameters
        ----------
        n_heads : int. Number of prediction heads.
        in_features : int. Input feature dimension.
        out_features : int. Output feature dimension per head.
        hidden_features : int. Hidden layer dimension (if n_layers > 1).
        n_layers : int. Number of layers in each head.
        activation : DictConfig or nn.Module. Activation function.
        dropout_rate : float. Dropout rate.
        """
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features * n_heads  # Total output dimension

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
        """Forward pass through all prediction heads, concatenating outputs."""
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=-1)


class MLPModular(nn.Module):
    """
    Modular MLP where all components (encoding, input, hidden, output) are
    specified via config. Supports skip connections and flexible output layers.
    """

    def __init__(
        self,
        positional_encoding: Union[DictConfig, nn.Module] = None,
        input_layer: Union[DictConfig, nn.Module] = None,
        hidden_layer: Union[DictConfig, nn.Module] = None,
        output_layer: Union[DictConfig, nn.Module] = None,
        hidden_skip: bool = False,
        output_skip: bool = False,
        inject_encoding_hidden: bool = False,
        inject_encoding_output: bool = False,
    ) -> None:
        """
        Initialize MLPModular.

        Parameters
        ----------
        positional_encoding : DictConfig or nn.Module. Optional positional encoding.
        input_layer : DictConfig or nn.Module. Input projection layer.
        hidden_layer : DictConfig or nn.Module. Hidden layer(s). Can be:
            - MLPBlock: single block (use hidden_skip for residual)
            - MLPBlocks: multiple blocks (control depth via n_blocks parameter)
            - Any custom module
        output_layer : DictConfig or nn.Module. Output layer. Can be:
            - PredictionHead: single prediction head
            - PredictionHeads: multiple prediction heads with identical architecture
            - nn.Linear: simple linear layer
            - Any custom module
        hidden_skip : bool. Wrap hidden_layer with residual connection (only for single blocks).
        output_skip : bool. Concatenate input features to output.
        inject_encoding_hidden : bool. Concatenate encoding to hidden layer input.
        inject_encoding_output : bool. Concatenate encoding to output layer input.
        """
        super().__init__()

        # Positional encoding
        if positional_encoding is not None:
            self.positional_encoding = instantiate(positional_encoding)
            self.d_encoding = self.positional_encoding.d_output
        else:
            self.positional_encoding = nn.Identity()
            self.d_encoding = 0

        # Store injection flags
        self.inject_encoding_hidden = inject_encoding_hidden
        self.inject_encoding_output = inject_encoding_output

        # Input layer
        self.input_layer = instantiate(input_layer) if input_layer is not None else nn.Identity()

        # Get input layer output dimension
        if hasattr(self.input_layer, 'out_features'):
            input_out_dim = self.input_layer.out_features
        elif hasattr(self.input_layer, 'model') and hasattr(self.input_layer.model, 'out_features'):
            input_out_dim = self.input_layer.model[-1].out_features
        else:
            # Try to infer from config
            if isinstance(input_layer, DictConfig) and hasattr(input_layer, 'out_features'):
                input_out_dim = input_layer.out_features
            else:
                input_out_dim = None

        # Hidden layer (single module that can be MLPBlock, MLPBlocks, or custom)
        if hidden_layer is not None:
            self.hidden_layer = instantiate(hidden_layer)
            
            # Update in_features if encoding injection is enabled
            if inject_encoding_hidden and hasattr(self.hidden_layer, 'in_features'):
                self.hidden_layer.in_features = input_out_dim + self.d_encoding

            # Wrap with residual if requested (for single blocks)
            if hidden_skip:
                # Get output dimension
                if hasattr(self.hidden_layer, 'out_features'):
                    hidden_out_dim = self.hidden_layer.out_features
                elif isinstance(hidden_layer, DictConfig) and hasattr(hidden_layer, 'out_features'):
                    hidden_out_dim = hidden_layer.out_features
                else:
                    hidden_out_dim = input_out_dim
                
                # Add residual wrapper
                layer_in_dim = input_out_dim + self.d_encoding if inject_encoding_hidden else input_out_dim
                if layer_in_dim != hidden_out_dim:
                    projection = nn.Linear(layer_in_dim, hidden_out_dim, bias=False)
                else:
                    projection = None
                self.hidden_layer = Residual(self.hidden_layer, projection=projection)
                current_dim = hidden_out_dim
            else:
                # Get output dimension without residual
                if hasattr(self.hidden_layer, 'out_features'):
                    current_dim = self.hidden_layer.out_features
                elif isinstance(hidden_layer, DictConfig) and hasattr(hidden_layer, 'out_features'):
                    current_dim = hidden_layer.out_features
                else:
                    current_dim = input_out_dim
        else:
            self.hidden_layer = nn.Identity()
            current_dim = input_out_dim

        # Output skip connection
        self.output_skip = output_skip
        if output_skip:
            output_in_dim = current_dim + input_out_dim
        else:
            output_in_dim = current_dim

        # Add encoding injection dimension
        if inject_encoding_output:
            output_in_dim += self.d_encoding

        # Output layer
        if output_layer is not None:
            self.output_layer = instantiate(output_layer) if isinstance(output_layer, DictConfig) else output_layer
            # Update in_features if supported
            if hasattr(self.output_layer, 'in_features'):
                self.output_layer.in_features = output_in_dim
        else:
            self.output_layer = nn.Identity()

        # Store dimensions for skip concatenation
        self.input_out_dim = input_out_dim
        self.hidden_out_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the modular MLP.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor.
        """
        # Positional encoding
        x_enc = self.positional_encoding(x)

        # Input layer
        x_input = self.input_layer(x_enc)

        # Hidden layer with optional encoding injection
        if self.inject_encoding_hidden:
            x_hidden = self.hidden_layer(torch.cat([x_input, x_enc], dim=-1))
        else:
            x_hidden = self.hidden_layer(x_input)

        # Prepare output input
        if self.output_skip:
            x_out = torch.cat([x_hidden, x_input], dim=-1)
        else:
            x_out = x_hidden

        if self.inject_encoding_output:
            x_out = torch.cat([x_out, x_enc], dim=-1)

        # Output layer
        return self.output_layer(x_out)

