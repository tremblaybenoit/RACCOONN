import torch
import torch.nn as nn


class IdentityPositionalEncoding(nn.Module):
    """ Identity Positional Encoding. This is a simple positional encoding that does not change the input."""
    def __init__(self, d_input):
        """ Initialize Identity Positional Encoding.

        Parameters
        ----------
        d_input : int. Input dimension.

        Returns
        -------
        None.
        """
        super().__init__()
        self.d_output = d_input

    def forward(self, x):
        """ Forward pass through the Identity Positional Encoding.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Encoded tensor.
        """
        return x


class GaussianPositionalEncoding(nn.Module):
    """ Gaussian Positional Encoding. Credit: Robert Jarolim, Momchil Molnar."""
    def __init__(self, num_freqs, d_input):
        """ Initialize Gaussian Positional Encoding.

        Parameters
        ----------
        num_freqs : int. Number of frequencies.
        d_input : int. Input dimension.

        Returns
        -------
        None.
        """
        super().__init__()
        # Initialize frequencies
        frequencies = torch.randn(num_freqs, d_input)
        self.frequencies = nn.Parameter(frequencies[None], requires_grad=False)
        # Output dimension
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        """ Forward pass through the Gaussian Positional Encoding.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Encoded tensor.
        """

        # Apply Gaussian positional encoding
        encoded = x[:, None, :] * self.frequencies
        encoded = encoded.reshape(x.shape[0], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
        return encoded
