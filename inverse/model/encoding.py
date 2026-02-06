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


class RescaledPositionalEncoding(IdentityPositionalEncoding):
    """ Identity Positional Encoding. This is a simple positional encoding that does not change the input."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Identity Positional Encoding.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Encoded tensor.
        """
        return x.sub_(0.5).mul_(2.0)  # Scale input to [-1, 1]


class GaussianPositionalEncoding(nn.Module):
    """ Gaussian Positional Encoding. Credit: Robert Jarolim, Momchil Molnar."""
    def __init__(self, num_freqs: int, d_input: int, sigma: float = 1.0):
        """ Initialize Gaussian Positional Encoding.

        Parameters
        ----------
        num_freqs : int. Number of frequencies.
        d_input : int. Input dimension.
        sigma : float. Standard deviation for frequency sampling.

        Returns
        -------
        None.
        """
        super().__init__()
        # Initialize frequencies
        self.register_buffer("frequencies", torch.randn(num_freqs, d_input)*sigma)
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
        encoded_in = x.unsqueeze(1) * self.frequencies.unsqueeze(0) * (2.0 * torch.pi)  # (n_samples, num_freqs, d_input)
        encoded_in = encoded_in.reshape(n_samples, -1)  # (n_samples, num_freqs * d_input)
        # Output
        encoded_sin = torch.sin(encoded_in)
        encoded_cos = torch.cos(encoded_in)

        return torch.cat([x * 2.0 - 1.0, encoded_sin, encoded_cos], dim=-1)


class MultiScaleGaussianEncoding(nn.Module):
    """
    Multi-Scale Gaussian Positional Encoding.
    Scales frequency bandwidth per dimension to match physical scales.
    """

    def __init__(self, num_freqs: int, d_input: int, sigma_per_dim: list = None):
        """ Initialize Multi-Scale Gaussian Positional Encoding.

        Parameters
        ----------
        num_freqs : int. Number of frequencies.
        d_input : int. Input dimension.
        sigma_per_dim : list. Standard deviation per dimension. If None, defaults to 1.0 for all dimensions.

        Returns
        -------
        None.
        """

        # Initialize parent class
        super().__init__()

        # Set default sigmas if not provided
        if sigma_per_dim is None:
            sigma_per_dim = [1.0] * d_input

        # Register sigmas as a buffer to ensure they move with the model (cpu/gpu)
        sigmas = torch.tensor(sigma_per_dim).unsqueeze(0)  # Shape: (1, d_input)

        # Initialize frequencies N(0, 1) and scale by sigma
        base_freqs = torch.randn(num_freqs, d_input)
        scaled_freqs = base_freqs * sigmas

        # Register frequencies as a buffer
        self.register_buffer("frequencies", scaled_freqs)
        self.num_freqs = num_freqs
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Multi-Scale Gaussian Positional Encoding.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Encoded tensor.
        """

        # Number of samples and variables
        n_samples, n_variables = x.shape[0], x.shape[1]

        # Projection: (B, 1, D) * (1, F, D) -> (B, F, D)
        proj = x.unsqueeze(1) * self.frequencies.unsqueeze(0) * (2.0 * torch.pi)
        proj = proj.reshape(n_samples, -1)

        # Sin/Cos encoding
        encoded_sin = torch.sin(proj)
        encoded_cos = torch.cos(proj)

        # Output: [Original Scaled, Sin, Cos]
        return torch.cat([x * 2.0 - 1.0, encoded_sin, encoded_cos], dim=-1)
