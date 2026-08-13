import numpy as np
import torch
import torch.nn as nn
from code.architecture.activation import Swish, Scale, Sine


class CRTMArchitecture(nn.Module):
    """
    CRTM emulator architecture (Community Radiative Transfer Model).
    This is translation from Keras to Pytorch of the CRTM emulator by Howard et al. (2025).
    Link: https://zenodo.org/records/13963758.
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
        bt_norm_max: float = 355.0,
        bt_norm_min: float = 180.0,
        std_output_activation_offset: float = 0.001,
        std_scale_trainable: bool = True,
    ):
        """
        Initialize CRTM Architecture.

        Parameters
        ----------
        nprofvars : int. Number of profile variables.
        nsurfvars : int. Number of surface variables.
        nmetavars : int. Number of meta variables.
        nlevels : int. Number of vertical levels.
        nnodes_bt : int. Number of neurons in hidden layers.
        nhidden_bt : int. Number of hidden layers.
        dropout_rate : float. Dropout rate.
        bt_norm_max : float. Maximum brightness temperature for normalization.
        bt_norm_min : float. Minimum brightness temperature for normalization.
        std_output_activation_offset : float. Offset for standard deviation output.
        std_scale_trainable : bool. Whether the standard deviation scale is trainable.
        """

        # Class inheritance
        super().__init__()

        # Store parameters
        self.nprofvars = nprofvars
        self.nsurfvars = nsurfvars
        self.nmetavars = nmetavars
        self.nlevels = nlevels
        self.max_T = bt_norm_max
        self.min_T = bt_norm_min
        self.std_output_activation_offset = std_output_activation_offset

        # Output activations
        self.bt_output_activation = nn.Sigmoid()
        self.std_output_activation = nn.Softplus()

        # Neural network layer components
        self.flatten = nn.Flatten()
        self.concat = lambda *tensors: torch.cat(tensors, dim=1)
        self.hidden_layers = nn.ModuleList()
        self.swish_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First dense layer
        input_dim = nprofvars * nlevels + nsurfvars + nmetavars
        self.hidden_layers.append(nn.Linear(input_dim, nnodes_bt))
        self.swish_layers.append(Swish())
        self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Additional hidden layers
        for _ in range(nhidden_bt - 1):
            self.hidden_layers.append(nn.Linear(nnodes_bt, nnodes_bt))
            self.swish_layers.append(Swish())
            self.dropout_layers.append(nn.Dropout(dropout_rate))

        # Output layers
        self.out_T = nn.Linear(nnodes_bt, 10)
        self.out_std = nn.Linear(nnodes_bt, 10)
        if std_scale_trainable:
            self.std_scale = Scale()
        else:
            self.std_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CRTM architecture.

        Parameters
        ----------
        x: torch.Tensor. Concatenation of input tensors (profiles, surface, meta).
            profiles: torch.Tensor. Input tensor for profiles.
            surface: torch.Tensor. Input tensor for surface variables.
            meta: torch.Tensor. Input tensor for meta variables.

        Returns
        -------
        torch.Tensor. Output tensor (mean BT + std).
        """

        # Shared feature extraction
        features = x
        for dense, swish, drop in zip(self.hidden_layers, self.swish_layers, self.dropout_layers):
            features = dense(features)
            features = swish(features)
            features = drop(features)

        # Mean brightness temperature output
        out = self.out_T(features)
        out = self.bt_output_activation(out)
        out = out * (self.max_T - self.min_T) + self.min_T

        # Standard deviation output
        out_std = self.out_std(features)
        out_std = self.std_output_activation(out_std)
        if self.std_scale is not None:
            out_std = self.std_scale(out_std)
        out_std = out_std + self.std_output_activation_offset

        # Concatenate outputs
        return torch.cat([out, out_std], dim=1)
