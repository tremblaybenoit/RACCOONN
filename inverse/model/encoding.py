import torch
import torch.nn as nn


class IdentityPositionalEncoding(nn.Module):
    """ Identity Positional Encoding. This is a simple positional encoding that does not change the input."""
    def __init__(self, d_input: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, num_freqs: int, d_input: int):
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
        self.register_buffer("frequencies", torch.randn(num_freqs, d_input))
        self.num_freqs= num_freqs
        # Output dimension
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Gaussian Positional Encoding.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Encoded tensor.
        """

        # Number of samples
        n_samples, n_variables = x.shape[0], x.shape[1]

        # Apply Gaussian positional encoding
        encoded_in = x.unsqueeze(1) * self.frequencies.unsqueeze(0)  # (n_samples, num_freqs, d_input)
        encoded_in = encoded_in.reshape(x.shape[0], -1)  # (n_samples, num_freqs * d_input)

        # Compute sin and cos in-place
        encoded_sin = torch.empty_like(encoded_in)
        encoded_cos = torch.empty_like(encoded_in)
        torch.sin(encoded_in, out=encoded_sin)
        torch.cos(encoded_in, out=encoded_cos)

        # Output
        encoded_out = torch.empty((n_samples, self.d_output), device=encoded_in.device, dtype=encoded_in.dtype)
        encoded_out[:, :n_variables] = x * 2.0 - 1.0  # Scale input to [-1, 1]
        encoded_out[:, n_variables:n_variables+encoded_in.shape[1]] = encoded_sin
        encoded_out[:, n_variables+encoded_in.shape[1]:] = encoded_cos

        return encoded_out
