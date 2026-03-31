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

    def __init__(self, in_features, b: float = 1.0):
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
        # Learnable vector: one frequency per feature
        self.freq = nn.Parameter(torch.full((in_features,), float(w0)))

    def forward(self, x):
        # Element-wise multiplication of frequencies and features
        # x: (Batch, In_Features), self.freq: (In_Features)
        return torch.sin(self.freq * x)
