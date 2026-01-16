import torch
import torch.nn as nn


class ScaledTanh(nn.Module):
    """ScaledTanh activation function. """
    def __init__(self, limit=4.0):
        """ Initialize ScaledTanh activation function.

        Parameters
        ----------
        limit : float. Scaling factor for Tanh activation.

        Returns
        -------
        None.
        """
        super().__init__()
        self.limit = limit

    def forward(self, x):
        """ Forward pass for ScaledTanh activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying ScaledTanh activation.
        """
        return self.limit * torch.tanh(x/self.limit)
