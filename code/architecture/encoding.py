import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
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
    """ Rescaled Positional Encoding. This scales the input to the range [-1, 1]."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the Rescaled Positional Encoding.

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
        num_freqs: int | list[int] | DictConfig | ListConfig = 8,
        freq_mode: str = "linear",
        include_input: bool = True,
    ) -> None:
        """
        Initialize HarmonicEncoding.

        Parameters
        ----------
        d_input : int
            Input dimension (number of coordinates).
        num_freqs : int or list[int]
            Number of harmonic frequencies. If an int, the same number of frequencies
            is used for every input dimension. If a list of length d_input, each entry
            specifies the number of frequencies for the corresponding dimension, allowing
            per-variable frequency resolution (e.g. finer sampling for log-pressure than
            for lat/lon).
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
        elif isinstance(num_freqs, ListConfig):
            num_freqs = list(num_freqs)

        self.d_input = d_input
        self.freq_mode = freq_mode
        self.include_input = include_input

        # Normalise num_freqs to a per-dimension list
        if isinstance(num_freqs, int):
            num_freqs_list = [num_freqs] * d_input
        else:
            if len(num_freqs) != d_input:
                raise ValueError(
                    f"num_freqs list length ({len(num_freqs)}) must match d_input ({d_input})."
                )
            num_freqs_list = [int(nf) for nf in num_freqs]
        self.num_freqs: list[int] = num_freqs_list

        # Register one frequency buffer per input dimension
        for i, nf in enumerate(num_freqs_list):
            self.register_buffer(f"freqs_{i}", self._make_freqs(nf, freq_mode))

        # Output dimension: sum of 2*nf per dim, plus raw coords if requested
        self.d_output = sum(2 * nf for nf in num_freqs_list)
        if include_input:
            self.d_output += d_input

    @staticmethod
    def _make_freqs(num_freqs: int, freq_mode: str) -> torch.Tensor:
        if freq_mode == "linear":
            return torch.arange(1, num_freqs + 1, dtype=torch.float32)
        elif freq_mode == "exponential":
            return 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
        elif freq_mode == "log":
            return torch.logspace(0, torch.log10(torch.tensor(2 * torch.pi)), num_freqs)
        else:
            raise ValueError(f"Unknown freq_mode: {freq_mode}")

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

        out_parts = []
        x_scaled = x * 2.0 - 1.0
        if self.include_input:
            out_parts.append(x_scaled)

        for i, nf in enumerate(self.num_freqs):
            freqs_i = getattr(self, f"freqs_{i}")       # (nf,)
            # (B, 1) * (1, nf) * π → (B, nf)
            harmonics = x_scaled[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
            out_parts.append(torch.sin(harmonics))
            out_parts.append(torch.cos(harmonics))

        return torch.cat(out_parts, dim=-1)


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


class HybridSpatialPressureEncoding(nn.Module):
    """
    Hybrid Positional Encoding for Atmospheric Fields:
    - Harmonic (Fourier) encoding for spatial coordinates (lat, lon -> indices 0 and 1).
    - Chebyshev polynomial basis expansion for vertical coordinate (log_pressure -> index 2).
    """

    def __init__(
            self,
            d_input: int | DictConfig = 3,
            num_freqs: int | list[int] | DictConfig | ListConfig = 8,
            freq_mode: str = "linear",
            include_input: bool = True,
            max_degree: int | DictConfig = 10,
    ) -> None:
        """
        Initialize HybridSpatialPressureEncoding.

        Parameters
        ----------
        d_input : int
            Input dimension (expected to be 3: [lat, lon, log_pressure]).
        num_freqs : int or list[int]
            Number of harmonic frequencies for lat and lon.
        freq_mode : str
            How to generate harmonic frequencies: "linear", "exponential", or "log".
        include_input : bool
            If True, include original input coordinates in output. Default True.
        max_degree : int
            Maximum polynomial degree for the Chebyshev expansion on log_pressure.

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
        elif isinstance(num_freqs, ListConfig):
            num_freqs = list(num_freqs)
        if isinstance(max_degree, DictConfig):
            max_degree = instantiate(max_degree)

        self.d_input = d_input
        self.freq_mode = freq_mode
        self.include_input = include_input
        self.max_degree = max_degree

        if d_input != 3:
            raise ValueError(
                f"HybridSpatialPressureEncoding expects d_input=3 [lat, lon, log_pressure], got {d_input}.")

        # Harmonic frequencies configuration for spatial dims (lat, lon -> 2 dimensions)
        spatial_dims = 2
        if isinstance(num_freqs, int):
            self.num_freqs_list = [num_freqs] * spatial_dims
        else:
            if len(num_freqs) != spatial_dims:
                raise ValueError(
                    f"num_freqs list length ({len(num_freqs)}) must match spatial dimensions ({spatial_dims}).")
            self.num_freqs_list = [int(nf) for nf in num_freqs]

        # Register one frequency buffer per spatial input dimension
        for i, nf in enumerate(self.num_freqs_list):
            self.register_buffer(f"freqs_{i}", self._make_freqs(nf, freq_mode))

        # Calculate total output dimension:
        # 1. Original raw inputs (if included)
        self.d_output = d_input if include_input else 0
        # 2. Harmonic outputs for lat & lon: sum of 2*nf per spatial dimension
        self.d_output += sum(2 * nf for nf in self.num_freqs_list)
        # 3. Chebyshev polynomial outputs for log_pressure: max_degree
        self.d_output += max_degree

    @staticmethod
    def _make_freqs(num_freqs: int, freq_mode: str) -> torch.Tensor:
        if freq_mode == "linear":
            return torch.arange(1, num_freqs + 1, dtype=torch.float32)
        elif freq_mode == "exponential":
            return 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
        elif freq_mode == "log":
            return torch.logspace(0, torch.log10(torch.tensor(2.0 * torch.pi)), num_freqs)
        else:
            raise ValueError(f"Unknown freq_mode: {freq_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode lat/lon harmonically and log_pressure via Chebyshev polynomials,
        appending uniformly rescaled [-1, 1] raw inputs if requested.
        """
        out_parts = []

        # 1. Rescale all raw inputs from [0, 1] to [-1, 1] for uniform network scaling
        if self.include_input:
            x_scaled = x * 2.0 - 1.0
            out_parts.append(x_scaled)
        else:
            # We still need the scaled version for the Chebyshev expansion below
            x_scaled = x * 2.0 - 1.0

        # 2. Harmonic Encoding for Lat (index 0) and Lon (index 1)
        # (Harmonic functions operate smoothly on bounded coordinates)
        for i in range(2):
            nf = self.num_freqs_list[i]
            freqs_i = getattr(self, f"freqs_{i}")  # (nf,)
            # Using the original or scaled coordinate for frequencies (typically original [0, 1] or [-1, 1] scaled)
            harmonics = x[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
            out_parts.append(torch.sin(harmonics))
            out_parts.append(torch.cos(harmonics))

        # 3. Chebyshev Polynomial Expansion for log_pressure (index 2)
        # Using x_scaled[:, 2:3] which is already mapped to [-1, 1]
        p_scaled = x_scaled[:, 2:3]  # (B, 1)

        if self.max_degree > 0:
            cheby_parts = []

            # T_1(x) = x
            t_prev = p_scaled
            cheby_parts.append(t_prev)

            if self.max_degree >= 2:
                # T_2(x) = 2x^2 - 1
                t_curr = 2.0 * p_scaled * p_scaled - 1.0
                cheby_parts.append(t_curr)

                # Higher degrees via stable recurrence: T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
                for _ in range(3, self.max_degree + 1):
                    t_next = 2.0 * p_scaled * t_curr - t_prev
                    cheby_parts.append(t_next)
                    t_prev = t_curr
                    t_curr = t_next

            out_parts.append(torch.cat(cheby_parts, dim=-1))

        return torch.cat(out_parts, dim=-1)
