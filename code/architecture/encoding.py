import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from utilities.instantiators import instantiate


class IdentityPositionalEncoding(nn.Module):
    """ Identity Positional Encoding. This is a simple positional encoding that does not change the input."""
    def __init__(self, d_input: int | DictConfig) -> None:
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
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
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
            d_input: int | DictConfig,
            num_freqs: int | DictConfig = 20,
            sigma: float | DictConfig = 1.0
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
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(num_freqs, DictConfig):
            num_freqs = instantiate(num_freqs)
        if isinstance(sigma, DictConfig):
            sigma = instantiate(sigma)
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
            d_input: int | DictConfig,
            num_freqs: int | DictConfig = 20,
            sigma_per_dim: list | DictConfig | None = None
    ) -> None:
        """ Initialize Multi-Scale Gaussian Positional Encoding.

        Parameters
        ----------
        d_input : int. Input dimension.
        num_freqs : int. Number of frequencies.
        sigma_per_dim : list. Standard deviation per dimension. If None, defaults to 1.0 for all dimensions.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Initialize frequencies
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(num_freqs, DictConfig):
            num_freqs = instantiate(num_freqs)
        if isinstance(sigma_per_dim, DictConfig):
            sigma_per_dim = instantiate(sigma_per_dim)

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


class HarmonicEncoding(nn.Module):
    """
    Harmonic (Fourier) Positional Encoding.

    Encodes coordinates using sine and cosine basis functions at multiple frequencies.
    Similar to Fourier features in neural fields (e.g., NeRF, neural implicit representations).

    For each input dimension, generates:
        [sin(π*k*x), cos(π*k*x) for k=1,2,...,num_freqs]

    This is more expressive than simple coordinate inputs for capturing fine details.
    """

    def __init__(
        self,
        d_input: int | DictConfig,
        num_freqs: int | DictConfig = 10,
        freq_mode: str = "linear",
        include_input: bool = True,
    ) -> None:
        """
        Initialize HarmonicEncoding.

        Parameters
        ----------
        d_input : int
            Input dimension (number of coordinates).
        num_freqs : int
            Number of harmonic frequencies per dimension. Default 10.
        freq_mode : str
            How to generate frequencies:
            - "linear": k = 1, 2, ..., num_freqs (equal spacing)
            - "exponential": k = 2^0, 2^1, ..., 2^(num_freqs-1) (exponential spacing)
            - "log": k = log-spaced from 1 to 2π
        include_input : bool
            If True, include original input coordinates in output. Default True.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Resolve config types
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(num_freqs, DictConfig):
            num_freqs = instantiate(num_freqs)

        self.d_input = d_input
        self.num_freqs = num_freqs
        self.freq_mode = freq_mode
        self.include_input = include_input

        # Generate frequency bases
        if freq_mode == "linear":
            # k = 1, 2, ..., num_freqs
            freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32)
        elif freq_mode == "exponential":
            # k = 2^0, 2^1, ..., 2^(num_freqs-1)
            freqs = 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
        elif freq_mode == "log":
            # k = log-spaced from 1 to 2π
            freqs = torch.logspace(0, torch.log10(torch.tensor(2 * torch.pi)), num_freqs)
        else:
            raise ValueError(f"Unknown freq_mode: {freq_mode}")

        self.register_buffer("freqs", freqs)

        # Output dimension
        # Each input dimension gets: sin + cos for each frequency = 2 * num_freqs
        self.d_output = d_input * (2 * num_freqs)
        if include_input:
            self.d_output += d_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode coordinates harmonically.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d_input).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, d_output).
        """

        # Compute harmonics: (B, d_input, 1) * (1, 1, num_freqs) -> (B, d_input, num_freqs)
        x_expanded = x.unsqueeze(-1)  # (B, d_input, 1)
        freqs_expanded = self.freqs.view(1, 1, -1)  # (1, 1, num_freqs)

        # x * freqs * π: (B, d_input, num_freqs)
        harmonics = x_expanded * freqs_expanded * torch.pi

        # Compute sin and cos: each (B, d_input, num_freqs)
        sin_harmonics = torch.sin(harmonics)
        cos_harmonics = torch.cos(harmonics)

        # Flatten: (B, d_input * num_freqs)
        sin_flat = sin_harmonics.reshape(x.shape[0], -1)
        cos_flat = cos_harmonics.reshape(x.shape[0], -1)

        # Concatenate sin and cos
        out_parts = [sin_flat, cos_flat]

        # Optionally include original input
        if self.include_input:
            out_parts.insert(0, x)

        return torch.cat(out_parts, dim=-1)


class BSplinePressureEncoding(nn.Module):
    """
    B-Spline Basis Pressure Encoding for CONTINUOUS normalized pressure.

    Represents continuous normalized pressure [0, 1] using smooth B-spline basis functions.
    The model learns coefficients for each basis function, allowing smooth
    and parameter-efficient pressure encoding.

    Advantages:
    - Parameter-efficient: n_basis × embedding_dim parameters (e.g., 10 × 16 = 160)
    - Smooth representation across entire normalized pressure range [0, 1]
    - Natural interpolation between pressure values
    - Works with any continuous pressure value in [0, 1]

    Uses cubic B-splines (order=3).

    Example:
    --------
    x = torch.tensor([[0.5, 0.7, 0.2, 0.42],  # pressure_normalized = 0.42
                      [0.6, 0.8, 0.3, 0.78]]) # pressure_normalized = 0.78
    encoding = BSplinePressureEncoding(n_levels=127, n_basis=10, embedding_dim=16, d_input=4)
    output = encoding(x)  # shape: (2, 3 + 16) = (2, 19)
    """

    def __init__(
        self,
        n_levels: int | DictConfig = 127,
        n_basis: int | DictConfig = 10,
        embedding_dim: int | DictConfig = 16,
        order: int = 3,
        d_input: int | DictConfig | None = None,
    ) -> None:
        """
        Initialize BSplinePressureEncoding.

        Parameters
        ----------
        n_levels : int
            Reference number of discrete pressure levels (used for documentation).
            Default 127 (WRF pressure levels).
        n_basis : int
            Number of B-spline basis functions. Default 10.
            Fewer basis functions = smoother representation, fewer parameters.
            Typical range: 8-16 for smooth 127-level representation.
        embedding_dim : int
            Output embedding dimension after B-spline projection. Default 16.
        order : int
            Spline order (3 = cubic). Default 3. Must be ≥ 1.
        d_input : int or None
            Total input dimension. Used to compute output size.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Resolve config types
        if isinstance(n_levels, DictConfig):
            n_levels = instantiate(n_levels)
        if isinstance(n_basis, DictConfig):
            n_basis = instantiate(n_basis)
        if isinstance(embedding_dim, DictConfig):
            embedding_dim = instantiate(embedding_dim)
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)

        self.n_levels = n_levels
        self.n_basis = n_basis
        self.embedding_dim = embedding_dim
        self.order = order

        # Create knot vector for B-spline on [0, 1]
        # For scipy.interpolate.BSpline: len(knots) must equal n_basis + order + 1
        # Clamped knots: (order+1) zeros, interior knots, (order+1) ones
        n_interior = n_basis - order - 1
        if n_interior < 0:
            raise ValueError(
                f"n_basis ({n_basis}) must be > order ({order}) for valid B-spline. "
                f"Typically n_basis ≥ order + 2. Try n_basis ≥ {order + 2}."
            )

        interior_knots = torch.linspace(0, 1, n_interior + 2, dtype=torch.float32)
        knots = torch.cat([
            torch.zeros(order + 1),
            interior_knots[1:-1],
            torch.ones(order + 1)
        ])
        # Verify knot vector length
        expected_len = n_basis + order + 1
        if len(knots) != expected_len:
            raise ValueError(
                f"Knot vector length mismatch: got {len(knots)}, expected {expected_len}. "
                f"(n_basis={n_basis}, order={order})"
            )
        self.register_buffer("knots", knots)

        # Learnable coefficients for each basis function: (n_basis, embedding_dim)
        self.spline_coefs = nn.Parameter(torch.randn(n_basis, embedding_dim) * 0.01)

        # Build basis evaluator
        self._build_basis_evaluator()

        # Output dimension
        if d_input is not None:
            # Replace pressure (1 dim) with embedding_dim
            self.d_output = d_input - 1 + embedding_dim
        else:
            self.d_output = embedding_dim

    def _build_basis_evaluator(self) -> None:
        """Build B-spline basis function evaluators."""
        try:
            from scipy.interpolate import BSpline
        except ImportError:
            raise ImportError("scipy required for B-spline encoding. Install with: pip install scipy")

        self._basis_funcs = []

        for i in range(self.n_basis):
            # Create unit basis vector (1 at position i, 0 elsewhere)
            coefs = np.zeros(self.n_basis, dtype=np.float32)
            coefs[i] = 1.0

            # Create B-spline basis function
            basis_func = BSpline(
                self.knots.cpu().numpy(),
                coefs,
                self.order,
                extrapolate=False
            )
            self._basis_funcs.append(basis_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: replace continuous normalized pressure with B-spline embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d_input).
            Assumes last column is continuous normalized pressure ∈ [0, 1].

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, d_output) where normalized pressure
            is replaced with its B-spline embedding.
        """

        # Separate pressure (last column) from other coordinates
        other_coords = x[:, :-1]  # (B, d_input-1)
        pressure_norm = x[:, -1]  # (B,) continuous values in [0, 1]

        # Clamp to valid range [0, 1]
        pressure_norm = torch.clamp(pressure_norm, 0.0, 1.0)

        # Evaluate B-spline basis functions at continuous pressure values
        # Shape: (B, n_basis)
        batch_size = pressure_norm.shape[0]
        basis_vals_list = []

        for i, basis_func in enumerate(self._basis_funcs):
            # Evaluate basis function at each batch element's pressure value
            basis_vals = basis_func(pressure_norm.cpu().detach().numpy())  # (B,)
            # Replace NaN with 0 for numerical stability
            basis_vals = np.nan_to_num(basis_vals, nan=0.0)
            basis_vals_list.append(torch.tensor(basis_vals, dtype=torch.float32, device=x.device))

        basis_vals = torch.stack(basis_vals_list, dim=1)  # (B, n_basis)

        # Compute B-spline embedding: (B, n_basis) @ (n_basis, embedding_dim) = (B, embedding_dim)
        pressure_embedding = torch.matmul(basis_vals, self.spline_coefs)

        # Concatenate other coordinates with pressure embedding
        return torch.cat([other_coords, pressure_embedding], dim=-1)


class HybridPositionalEncoding(nn.Module):
    """
    Hybrid positional encoding with different strategies per dimension group.

    Allows applying different encodings to different input dimensions. For example:
    - HarmonicEncoding for lat/lon/scans (continuous spatial coordinates)
    - BSplinePressureEncoding for normalized pressure (continuous [0, 1])
    - GaussianEncoding for other continuous variables

    Useful for multi-modal inputs where different dimensions have different properties.

    ⚠️  All inputs to HybridPositionalEncoding must be continuous and normalized [0, 1].

    Example (min-max normalized inputs):
    -----------------------------------
    x = torch.tensor([
        [0.5, 0.7, 0.2, 0.42],  # [lat_norm, lon_norm, scans_norm, pressure_norm]
        [0.6, 0.8, 0.3, 0.78]
    ])

    Config:
    -------
    positional_encoding:
      _target_: code.architecture.encoding.HybridPositionalEncoding
      d_input: 4
      d_split: [3, 1]  # First 3 dims to spatial, last 1 to pressure
      encodings:
        spatial:
          _target_: code.architecture.encoding.HarmonicEncoding
          d_input: 3
          num_freqs: 8
          freq_mode: linear
          include_input: true
        pressure:
          _target_: code.architecture.encoding.BSplinePressureEncoding
          n_levels: 127
          n_basis: 10
          embedding_dim: 16
          d_input: 1
    """

    def __init__(
        self,
        encodings: DictConfig | dict,
        d_input: int | DictConfig,
        d_split: list | DictConfig | None = None,
    ) -> None:
        """
        Initialize HybridPositionalEncoding.

        Parameters
        ----------
        encodings : DictConfig or dict
            Dictionary mapping names to encoding configs. Each encoding should have:
            - _target_: path to encoding class
            - d_input: input dimension for this encoding
            - Other encoding-specific parameters

        d_input : int
            Total input dimension (must match all inputs to forward()).
        d_split : list or None
            How to split input dimensions across encodings. Length = len(encodings).
            Example: [3, 1] means first 3 dims go to first encoding, next 1 to second.
            Sum of d_split must equal d_input.
            If None, will try to infer from encoding d_input parameters.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Resolve config types
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        # NOTE: Do NOT convert encodings DictConfig to dict - keep it as DictConfig
        # so that instantiate() can properly handle nested _target_ configs
        if isinstance(encodings, dict) and not isinstance(encodings, DictConfig):
            # If it's a regular dict, convert values to DictConfig for compatibility
            encodings = DictConfig(encodings)

        self.d_input = d_input
        self.encoding_names = list(encodings.keys())

        # Parse d_split
        if d_split is not None:
            if isinstance(d_split, DictConfig):
                d_split = instantiate(d_split)
            self.d_split = d_split
        else:
            # Try to infer from encoding configs
            self.d_split = []
            for name in self.encoding_names:
                enc_cfg = encodings[name]
                if isinstance(enc_cfg, DictConfig) and "d_input" in enc_cfg:
                    self.d_split.append(int(instantiate(enc_cfg.d_input)))
                else:
                    raise ValueError(
                        f"Cannot infer d_input for encoding '{name}'. "
                        f"Provide d_split explicitly or ensure each encoding has d_input parameter."
                    )

        # Validate split
        if sum(self.d_split) != d_input:
            raise ValueError(
                f"d_split {self.d_split} sums to {sum(self.d_split)}, "
                f"but d_input is {d_input}. They must match."
            )

        # Instantiate encodings
        self.encodings = nn.ModuleDict()
        for name, enc_cfg in encodings.items():
            if isinstance(enc_cfg, DictConfig):
                # DictConfig with _target_ - instantiate it
                self.encodings[name] = instantiate(enc_cfg)
            elif isinstance(enc_cfg, nn.Module):
                # Already instantiated module - use directly
                self.encodings[name] = enc_cfg
            else:
                raise TypeError(
                    f"Encoding '{name}' must be a DictConfig or nn.Module, got {type(enc_cfg)}"
                )

        # Compute cumulative split for indexing
        self.d_cumsum = [0] + list(np.cumsum(self.d_split))

        # Compute total output dimension
        self.d_output = sum(
            enc.d_output if hasattr(enc, "d_output") else self.d_split[i]
            for i, enc in enumerate(self.encodings.values())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply each encoding to its corresponding input dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d_input).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (batch_size, d_output).
        """

        # Split input according to d_split
        out_parts = []
        for i, (name, encoding) in enumerate(self.encodings.items()):
            # Extract the portion of input for this encoding
            start_idx = self.d_cumsum[i]
            end_idx = self.d_cumsum[i + 1]
            x_i = x[:, start_idx:end_idx]

            # Apply encoding
            encoded_i = encoding(x_i)
            out_parts.append(encoded_i)

        # Concatenate all encoded parts
        return torch.cat(out_parts, dim=-1)
