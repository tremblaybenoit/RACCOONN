import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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


class SuperLearnableSwish(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # Initializing with 1.0 (SiLU)
        self.b = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for SuperLearnableSwish activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying SuperLearnableSwish activation.
        """

        # Use softplus to ensure beta > 0 without hard clipping
        # This keeps the landscape smooth for PINN second derivatives
        b_safe = F.softplus(self.b)
        return x * torch.sigmoid(b_safe * x)


class NonLearnableSwish(nn.Module):

    def __init__(self, b: float = 1.0):
        super().__init__()
        # Initializing with 1.0 (SiLU)
        self.b = b

    """Non-learnable Swish activation function. """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for Non-learnable Swish activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying Non-learnable Swish activation.
        """
        return x * torch.sigmoid(self.b*x)


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


class LearnableSine(nn.Module):
    """Learnable Sine activation function. """
    def __init__(self, w0=1.):
        """ Initialize Learnable Sine activation function.

        Parameters
        ----------
        w0 : float. Initial frequency of the sine function.

        Returns
        -------
        None.
        """
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0), requires_grad=True)

    def forward(self, x):
        """ Forward pass for Learnable Sine activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying Learnable Sine activation.
        """
        return torch.sin(self.w0.abs() * x)


class SuperLearnableSine(nn.Module):
    def __init__(self, in_features, w0=30.0):
        super().__init__()
        # Each input feature gets its own learnable frequency scaling
        self.w0 = nn.Parameter(torch.ones(in_features) * w0, requires_grad=True)

    def forward(self, x):
        # x has shape (Batch, In_Features)
        w0_constrained = torch.clamp(self.w0, min=0.0, max=30.0)
        return torch.sin(w0_constrained * x)


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


class Snake(nn.Module):
    """Snake activation function from 3DClouds. """
    def __init__(self, a: float = 1.0):
        """ Initialize Snake activation function.

        Parameters
        ----------
        a : float. Frequency parameter for the snake function.

        Returns
        -------
        None.
        """
        super().__init__()
        self.a = a

    def forward(self, x):
        """ Forward pass for Snake activation function.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Output tensor after applying Snake activation.
        """
        return x + (1.0 / self.a) * torch.sin(self.a * x) ** 2

