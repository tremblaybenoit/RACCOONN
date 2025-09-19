import torch
import torch.nn as nn
import numpy as np


class Scale(nn.Module):
    """Scale activation function. """

    def __init__(self):
        """ Initialize Scale activation function.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        super().__init__()

        # Trainable parameter for the Scale function
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """ Forward pass for Scale activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying Scale activation.
        """

        return x * self.b


def gelu(x):
    """Gaussian Error Linear Unit activation function.

    Parameters
    ----------
    x : torch.Tensor. Input tensor.

    Returns
    -------
    torch.Tensor. Output tensor after applying GELU activation.
    """
    return 0.5 * x * (1 + torch.erf(x / np.sqrt(2.0)))


class Swish(nn.Module):
    """Swish activation function. """

    def __init__(self):
        """ Initialize Swish activation function.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        super().__init__()
        # Trainable parameter for the Swish function
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """ Forward pass for Swish activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying Swish activation.
        """

        return x * torch.sigmoid(self.b * x)


class Sine(nn.Module):
    """Sine activation function. """
    def __init__(self, w0=1.):
        """ Initialize Sine activation function.

        Parameters
        ----------
        w0 : float. Frequency of the sine function.

        Returns
        -------
        None.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """ Forward pass for Sine activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying Sine activation.
        """
        return torch.sin(self.w0 * x)