import torch
import torch.nn as nn
from typing import Union
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from code.architecture.activation import Scale
from code.architecture.mlp import MLPBlocks, PredictionHead


class CRTMBackbone(nn.Module):
    """
    Shared feature extraction backbone for CRTM architectures.
    Can be configured via MLPBlocks for flexibility.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 512,
        n_layers: int = 3,
        activation: Union[DictConfig, nn.Module] = None,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize CRTM Backbone.

        Parameters
        ----------
        in_features : int. Input feature dimension.
        hidden_features : int. Hidden layer dimension.
        n_layers : int. Number of hidden layers.
        activation : DictConfig or nn.Module. Activation function.
        dropout_rate : float. Dropout rate.
        """
        super().__init__()

        # Use MLPBlocks for the backbone
        self.backbone = MLPBlocks(
            in_features=in_features,
            out_features=hidden_features,
            hidden_features=hidden_features,
            n_blocks=n_layers,
            activation=activation,
            dropout_rate=dropout_rate,
        )
        self.out_features = hidden_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.backbone(x)


class CRTMDualHead(nn.Module):
    """
    Dual-head output for CRTM: mean brightness temperature + uncertainty.
    """

    def __init__(
        self,
        in_features: int,
        n_channels: int = 10,
        bt_norm_max: float = 355.0,
        bt_norm_min: float = 180.0,
        std_output_activation_offset: float = 0.001,
        std_scale_trainable: bool = True,
        mean_activation: str = 'sigmoid',
    ):
        """
        Initialize CRTM Dual Head.

        Parameters
        ----------
        in_features : int. Input feature dimension from backbone.
        n_channels : int. Number of output channels.
        bt_norm_max : float. Maximum brightness temperature.
        bt_norm_min : float. Minimum brightness temperature.
        std_output_activation_offset : float. Offset for std output.
        std_scale_trainable : bool. Whether std scale is trainable.
        mean_activation : str. Activation for mean output ('sigmoid' or 'identity').
        """
        super().__init__()

        self.max_T = bt_norm_max
        self.min_T = bt_norm_min
        self.std_offset = std_output_activation_offset

        # Output heads
        self.mean_head = nn.Linear(in_features, n_channels)
        self.std_head = nn.Linear(in_features, n_channels)

        # Activations
        self.mean_activation = nn.Sigmoid() if mean_activation == 'sigmoid' else nn.Identity()
        self.std_activation = nn.Softplus()

        # Optional learnable scaling for std
        self.std_scale = Scale() if std_scale_trainable else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual head.

        Parameters
        ----------
        features : torch.Tensor. Features from backbone.

        Returns
        -------
        torch.Tensor. Concatenated [mean_bt, std] outputs.
        """
        # Mean brightness temperature
        mean_out = self.mean_head(features)
        mean_out = self.mean_activation(mean_out)
        mean_out = mean_out * (self.max_T - self.min_T) + self.min_T

        # Standard deviation (uncertainty)
        std_out = self.std_head(features)
        std_out = self.std_activation(std_out)
        if self.std_scale is not None:
            std_out = self.std_scale(std_out)
        std_out = std_out + self.std_offset

        return torch.cat([mean_out, std_out], dim=1)


class CRTMModular(nn.Module):
    """
    Modular CRTM architecture using composable components.
    Separates input preprocessing, backbone, and output heads.
    """

    def __init__(
        self,
        nprofvars: int,
        nsurfvars: int,
        nmetavars: int,
        nlevels: int,
        backbone: Union[DictConfig, nn.Module] = None,
        output_head: Union[DictConfig, nn.Module] = None,
        # Default backbone parameters
        nnodes_bt: int = 512,
        nhidden_bt: int = 3,
        dropout_rate: float = 0.0,
        activation: Union[DictConfig, nn.Module] = None,
        # Default output parameters
        bt_norm_max: float = 355.0,
        bt_norm_min: float = 180.0,
        std_output_activation_offset: float = 0.001,
        std_scale_trainable: bool = True,
    ):
        """
        Initialize Modular CRTM Architecture.

        Parameters
        ----------
        nprofvars : int. Number of profile variables.
        nsurfvars : int. Number of surface variables.
        nmetavars : int. Number of meta variables.
        nlevels : int. Number of vertical levels.
        backbone : DictConfig or nn.Module. Feature extraction backbone.
        output_head : DictConfig or nn.Module. Output head.
        nnodes_bt : int. Hidden layer size (if using default backbone).
        nhidden_bt : int. Number of hidden layers (if using default backbone).
        dropout_rate : float. Dropout rate.
        activation : DictConfig or nn.Module. Activation function.
        bt_norm_max : float. Max BT for normalization.
        bt_norm_min : float. Min BT for normalization.
        std_output_activation_offset : float. Std offset.
        std_scale_trainable : bool. Trainable std scale.
        """
        super().__init__()

        # Store dimensions
        self.nprofvars = nprofvars
        self.nsurfvars = nsurfvars
        self.nmetavars = nmetavars
        self.nlevels = nlevels
        self.input_dim = nprofvars * nlevels + nsurfvars + nmetavars

        # Input preprocessing
        self.flatten = nn.Flatten()

        # Backbone (feature extraction)
        if backbone is not None:
            self.backbone = instantiate(backbone) if isinstance(backbone, DictConfig) else backbone
        else:
            # Default backbone using MLPBlocks
            self.backbone = CRTMBackbone(
                in_features=self.input_dim,
                hidden_features=nnodes_bt,
                n_layers=nhidden_bt,
                activation=activation,
                dropout_rate=dropout_rate,
            )

        # Get backbone output dimension
        if hasattr(self.backbone, 'out_features'):
            backbone_out_dim = self.backbone.out_features
        else:
            backbone_out_dim = nnodes_bt

        # Output head
        if output_head is not None:
            self.output_head = instantiate(output_head) if isinstance(output_head, DictConfig) else output_head
        else:
            # Default dual head
            self.output_head = CRTMDualHead(
                in_features=backbone_out_dim,
                n_channels=10,
                bt_norm_max=bt_norm_max,
                bt_norm_min=bt_norm_min,
                std_output_activation_offset=std_output_activation_offset,
                std_scale_trainable=std_scale_trainable,
            )

    def forward(self, input: dict) -> torch.Tensor:
        """
        Forward pass through modular CRTM.

        Parameters
        ----------
        input : dict. Dictionary containing:
            - 'prof': (batch, nprofvars, nlevels)
            - 'surf': (batch, nsurfvars)
            - 'meta': (batch, nmetavars)

        Returns
        -------
        torch.Tensor. Output tensor (mean BT + std).
        """
        # Flatten and concatenate inputs
        prof = self.flatten(input['prof'])
        x = torch.cat([prof, input['surf'], input['meta']], dim=1)

        # Extract features
        features = self.backbone(x)

        # Generate outputs
        return self.output_head(features)


class CRTMSkipConnection(nn.Module):
    """
    CRTM architecture with skip connection for smoother gradients.
    Combines linear baseline with nonlinear residual.
    """

    def __init__(
        self,
        nprofvars: int,
        nsurfvars: int,
        nmetavars: int,
        nlevels: int,
        nnodes_bt: int = 512,
        nhidden_bt: int = 3,
        dropout_rate: float = 0.0,
        activation: Union[DictConfig, nn.Module] = None,
        bt_norm_max: float = 355.0,
        bt_norm_min: float = 180.0,
        std_output_activation_offset: float = 0.001,
        std_scale_trainable: bool = True,
    ):
        """Initialize CRTM with Skip Connection."""
        super().__init__()

        # Dimensions
        self.input_dim = nprofvars * nlevels + nsurfvars + nmetavars
        self.max_T = bt_norm_max
        self.min_T = bt_norm_min
        self.std_offset = std_output_activation_offset

        # Input preprocessing
        self.flatten = nn.Flatten()

        # Linear skip path (gradient highway)
        self.skip_connection = nn.Linear(self.input_dim, 10)

        # Nonlinear residual path
        self.backbone = CRTMBackbone(
            in_features=self.input_dim,
            hidden_features=nnodes_bt,
            n_layers=nhidden_bt,
            activation=activation,
            dropout_rate=dropout_rate,
        )

        # Output heads on residual path
        self.residual_mean = nn.Linear(nnodes_bt, 10)
        self.residual_std = nn.Linear(nnodes_bt, 10)

        # Activations
        self.std_activation = nn.Softplus()
        self.std_scale = Scale() if std_scale_trainable else None

    def forward(self, input: dict) -> torch.Tensor:
        """Forward pass with skip connection."""
        # Flatten and concatenate
        prof = self.flatten(input['prof'])
        x = torch.cat([prof, input['surf'], input['meta']], dim=1)

        # Linear baseline path
        linear_bt = self.skip_connection(x)

        # Nonlinear residual path
        features = self.backbone(x)
        residual_bt = self.residual_mean(features)

        # Combine paths for mean
        mean_out = linear_bt + residual_bt
        mean_out = mean_out * (self.max_T - self.min_T) + self.min_T

        # Std output from residual path
        std_out = self.residual_std(features)
        std_out = self.std_activation(std_out)
        if self.std_scale is not None:
            std_out = self.std_scale(std_out)
        std_out = std_out + self.std_offset

        return torch.cat([mean_out, std_out], dim=1)

