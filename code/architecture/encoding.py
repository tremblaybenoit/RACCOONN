import torch
import torch.nn as nn


class IdentityPositionalEncoding(nn.Module):
    """ Identity Positional Encoding. This is a simple positional encoding that does not change the input."""
    def __init__(self, d_input: int) -> None:
        """ Initialize Identity Positional Encoding.

        Parameters
        ----------
        d_input : int. Input dimension.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Output dimensions
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
    def __init__(
            self,
            num_freqs: int,
            d_input: int,
            sigma: float = 1.0
    ) -> None:
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

        # Class inheritance
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

    def __init__(
            self,
            num_freqs: int,
            d_input: int,
            sigma_per_dim: list | None = None
    ) -> None:
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

        # Class inheritance
        super().__init__()

        # Set default sigmas if not provided
        if sigma_per_dim is None:
            sigma_per_dim = [1.0] * d_input
        sigmas = torch.tensor(sigma_per_dim, dtype=torch.float32)

        # Identify which dimensions want encoding (sigma > 0)
        self.register_buffer("encoding_mask", sigmas > 0)
        self.d_encoded = int(self.encoding_mask.sum().item())

        # Only generate frequencies for those specific dimensions
        if self.d_encoded > 0:
            # Filter sigmas to only active dimensions: (1, d_encoded)
            active_sigmas = sigmas[self.encoding_mask.bool()].unsqueeze(0)

            # (num_freqs, d_encoded)
            base_freqs = torch.randn(num_freqs, self.d_encoded)
            scaled_freqs = base_freqs * active_sigmas
            self.register_buffer("frequencies", scaled_freqs)
        else:
            self.frequencies = None

        # Output = [All original dims] + [2 * num_freqs * active dims]
        self.d_output = d_input + (num_freqs * 2 * self.d_encoded)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass through the Multi-Scale Gaussian Positional Encoding.

        Parameters
        ----------
        x : torch.Tensor. Input tensor.

        Returns
        -------
        torch.Tensor. Encoded tensor.
        """

        # Start with the scaled original input (Normalized to [-1, 1])
        out_parts = [x * 2.0 - 1.0]

        # If frequencies were generated for some dimensions
        if self.frequencies is not None:
            # Select only the columns of x that need encoding
            x_to_encode = x[:, self.encoding_mask.bool()]  # (B, d_encoded)

            # Project: (B, 1, d_encoded) * (num_freqs, d_encoded) -> (B, num_freqs, d_encoded)
            proj = x_to_encode.unsqueeze(1) * self.frequencies * (2.0 * torch.pi)

            # Flatten frequencies: (B, num_freqs * d_encoded)
            proj_flat = proj.reshape(x.shape[0], -1)
            # Apply sin/cos projection
            out_parts.append(torch.sin(proj_flat))
            out_parts.append(torch.cos(proj_flat))

        return torch.cat(out_parts, dim=-1)

