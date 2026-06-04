"""
Wrapper modules for building complex architectures.
Adapted from 3DClouds repository for RACCOONN.
"""
import torch
import torch.nn as nn
from typing import Union


class Residual(nn.Module):
    """Wraps a module with an additive skip connection."""

    def __init__(
        self,
        module: Union[nn.Module, nn.ModuleList],
        projection: nn.Module = None,
    ) -> None:
        """
        Initialize Residual block.

        Parameters
        ----------
        module : nn.Module. The block to apply on the main path.
        projection : nn.Module. Optional projection applied to the skip path
                     (e.g., a linear layer to match dimensions).
                     Defaults to nn.Identity.
        """
        super().__init__()
        self.module = module
        self.projection = projection if projection is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: out = module(x) + projection(x).

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Sum of module output and (projected) skip.
        """
        return self.module(x) + self.projection(x)


class Concatenate(nn.Module):
    """
    Wraps a module and concatenates a skip tensor with its output.

    Supports three usage patterns:
    - Self-concatenation: cat([x, module(x)], dim)
    - Skip connection:    cat([skip, module(x)], dim)
    - Projected skip:     cat([projection(skip), module(x)], dim)
    """

    def __init__(
        self,
        module: Union[nn.Module, nn.ModuleList],
        projection: nn.Module = None,
        dim: int = -1,
    ) -> None:
        """
        Initialize Concatenate block.

        Parameters
        ----------
        module : nn.Module. The block applied to the main input x.
        projection : nn.Module. Optional projection applied to the skip tensor
                     before concatenation. Defaults to nn.Identity.
        dim : int. Dimension along which to concatenate. Default is -1
              (feature dim for MLPs).
        """
        super().__init__()
        self.module = module
        self.projection = projection if projection is not None else nn.Identity()
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass: cat([projection(skip or x), module(x)], dim).

        Parameters
        ----------
        x : torch.Tensor. Main input passed through module.
        skip : torch.Tensor. Optional skip tensor to concatenate. If None,
               x is used as the skip (self-concatenation).

        Returns
        -------
        torch.Tensor. Concatenation of (projected) skip and module output.
        """
        branch = self.projection(skip if skip is not None else x)
        return torch.cat([branch, self.module(x)], dim=self.dim)

