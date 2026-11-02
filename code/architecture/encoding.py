import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig
from utilities.instantiators import instantiate
from typing import Union, List


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
        include_input: bool | list[bool] | DictConfig | ListConfig = True,
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

        # Resolve config types if using OmegaConf
        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(num_freqs, DictConfig):
            num_freqs = instantiate(num_freqs)
        elif isinstance(num_freqs, ListConfig):
            num_freqs = list(num_freqs)
        if isinstance(include_input, DictConfig):
            include_input = instantiate(include_input)
        elif isinstance(include_input, ListConfig):
            include_input = list(include_input)

        self.d_input = d_input
        self.freq_mode = freq_mode

        # 1. Normalize num_freqs to a per-dimension list
        if isinstance(num_freqs, int):
            num_freqs_list = [num_freqs] * d_input
        else:
            if len(num_freqs) != d_input:
                raise ValueError(
                    f"num_freqs list length ({len(num_freqs)}) must match d_input ({d_input})."
                )
            num_freqs_list = [int(nf) for nf in num_freqs]
        self.num_freqs: list[int] = num_freqs_list

        # 2. Normalize include_input to a per-dimension boolean list
        if isinstance(include_input, bool):
            include_input_list = [include_input] * d_input
        else:
            if len(include_input) != d_input:
                raise ValueError(
                    f"include_input list length ({len(include_input)}) must match d_input ({d_input})."
                )
            include_input_list = [bool(val) for val in include_input]
        self.include_input: list[bool] = include_input_list

        # Register frequency buffers for dimensions where num_freqs > 0
        for i, nf in enumerate(num_freqs_list):
            if nf > 0:
                self.register_buffer(f"freqs_{i}", self._make_freqs(nf, freq_mode))

        # Calculate exact output dimension per coordinate (raw input + 2 * num_freqs for sin/cos)
        self.d_output = sum(
            (1 if self.include_input[i] else 0) + (2 * nf)
            for i, nf in enumerate(num_freqs_list)
        )

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
        # Rescale inputs uniformly from [0, 1] to [-1, 1]
        x_scaled = x * 2.0 - 1.0

        for i, nf in enumerate(self.num_freqs):
            # Include raw scaled input for this coordinate if enabled
            if self.include_input[i]:
                out_parts.append(x_scaled[:, i:i + 1])

            # Include harmonic frequencies if greater than 0
            if nf > 0:
                freqs_i = getattr(self, f"freqs_{i}")  # (nf,)
                harmonics = x_scaled[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
                out_parts.append(torch.sin(harmonics))
                out_parts.append(torch.cos(harmonics))

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
            include_input: bool | list[bool] | DictConfig | ListConfig = True,
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
        if isinstance(include_input, DictConfig):
            include_input = instantiate(include_input)
        elif isinstance(include_input, ListConfig):
            include_input = list(include_input)
        if isinstance(max_degree, DictConfig):
            max_degree = instantiate(max_degree)

        self.d_input = d_input
        self.freq_mode = freq_mode
        self.max_degree = max_degree

        if d_input != 3:
            raise ValueError(
                f"HybridSpatialPressureEncoding expects d_input=3 [lat, lon, log_pressure], got {d_input}."
            )

        # 1. Normalize include_input to a per-dimension list of length 3
        if isinstance(include_input, bool):
            include_input_list = [include_input] * d_input
        else:
            if len(include_input) != d_input:
                raise ValueError(
                    f"include_input list length ({len(include_input)}) must match d_input ({d_input})."
                )
            include_input_list = [bool(val) for val in include_input]
        self.include_input_list: list[bool] = include_input_list

        # 2. Harmonic frequencies configuration for spatial dims (lat, lon -> 2 dimensions)
        spatial_dims = 2
        if isinstance(num_freqs, int):
            self.num_freqs_list = [num_freqs] * spatial_dims
        else:
            if len(num_freqs) != spatial_dims:
                raise ValueError(
                    f"num_freqs list length ({len(num_freqs)}) must match spatial dimensions ({spatial_dims})."
                )
            self.num_freqs_list = [int(nf) for nf in num_freqs]

        # Register one frequency buffer per spatial input dimension (if nf > 0)
        for i, nf in enumerate(self.num_freqs_list):
            if nf > 0:
                self.register_buffer(f"freqs_{i}", self._make_freqs(nf, freq_mode))

        # 3. Calculate total output dimension dynamically:
        self.d_output = 0
        # Raw inputs included per flag
        for i, inc in enumerate(self.include_input_list):
            if inc:
                self.d_output += 1
        # Harmonic outputs for lat & lon
        for nf in self.num_freqs_list:
            if nf > 0:
                self.d_output += 2 * nf
        # Chebyshev polynomial outputs for log_pressure
        if self.max_degree > 0:
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

        # Scale all raw inputs from [0, 1] to [-1, 1] for uniform network scaling
        x_scaled = x * 2.0 - 1.0

        # 1. Append raw inputs according to per-variable inclusion flags
        for i in range(self.d_input):
            if self.include_input_list[i]:
                out_parts.append(x_scaled[:, i:i + 1])

        # 2. Harmonic Encoding for Lat (index 0) and Lon (index 1)
        for i in range(2):
            nf = self.num_freqs_list[i]
            if nf > 0:
                freqs_i = getattr(self, f"freqs_{i}")  # (nf,)
                harmonics = x[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
                out_parts.append(torch.sin(harmonics))
                out_parts.append(torch.cos(harmonics))

        # 3. Chebyshev Polynomial Expansion for log_pressure (index 2)
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


class HarmonicSpatialGaussianPressureEncoding(nn.Module):
    """
    Hybrid Encoding: Harmonic (Fourier) for spatial dims (lat, lon),
    and Gaussian Fourier Features for vertical coordinate (log_pressure).
    """

    def __init__(
        self,
        d_input: int | DictConfig = 3,
        spatial_num_freqs: int | list[int] | DictConfig | ListConfig = 8,
        spatial_freq_mode: str = "linear",
        pressure_num_freqs: int = 16,
        pressure_sigma: float = 1.0,
        include_input: Union[bool, List[bool], DictConfig, ListConfig] = True,
    ) -> None:
        super().__init__()

        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(spatial_num_freqs, DictConfig):
            spatial_num_freqs = instantiate(spatial_num_freqs)
        elif isinstance(spatial_num_freqs, ListConfig):
            spatial_num_freqs = list(spatial_num_freqs)
        if isinstance(include_input, DictConfig):
            include_input = instantiate(include_input)
        elif isinstance(include_input, ListConfig):
            include_input = list(include_input)

        self.d_input = d_input
        if d_input != 3:
            raise ValueError(f"Hybrid encodings expect d_input=3 [lat, lon, log_pressure], got {d_input}.")

        # Normalize include_input to list of length 3
        if isinstance(include_input, bool):
            self.include_input = [include_input] * d_input
        else:
            self.include_input = [bool(v) for v in include_input]

        # 1. Harmonic frequencies for spatial dims (lat, lon -> 2 dimensions)
        spatial_dims = 2
        if isinstance(spatial_num_freqs, int):
            self.spatial_num_freqs_list = [spatial_num_freqs] * spatial_dims
        else:
            if len(spatial_num_freqs) != spatial_dims:
                raise ValueError(f"spatial_num_freqs length must match spatial dimensions ({spatial_dims}).")
            self.spatial_num_freqs_list = [int(nf) for nf in spatial_num_freqs]

        for i, nf in enumerate(self.spatial_num_freqs_list):
            if nf > 0:
                self.register_buffer(f"spatial_freqs_{i}", self._make_harmonic_freqs(nf, spatial_freq_mode))

        # 2. Gaussian random frequencies for log_pressure (index 2)
        pressure_freqs = torch.randn(pressure_num_freqs, 1) * pressure_sigma
        self.register_buffer("pressure_frequencies", pressure_freqs)

        # Output dimension calculation
        raw_count = sum(1 for inc in self.include_input if inc)
        spatial_encoded = sum(2 * nf for nf in self.spatial_num_freqs_list if nf > 0)
        pressure_encoded = 2 * pressure_num_freqs if pressure_num_freqs > 0 else 0
        self.d_output = raw_count + spatial_encoded + pressure_encoded

    @staticmethod
    def _make_harmonic_freqs(num_freqs: int, freq_mode: str) -> torch.Tensor:
        if freq_mode == "linear":
            return torch.arange(1, num_freqs + 1, dtype=torch.float32)
        elif freq_mode == "exponential":
            return 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
        elif freq_mode == "log":
            return torch.logspace(0, torch.log10(torch.tensor(2.0 * torch.pi)), num_freqs)
        else:
            raise ValueError(f"Unknown freq_mode: {freq_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_parts = []
        x_scaled = x * 2.0 - 1.0  # Scale to [-1, 1]

        # Append raw inputs per flag
        for i, inc in enumerate(self.include_input):
            if inc:
                out_parts.append(x_scaled[:, i:i + 1])

        # 1. Harmonic Encoding for Lat (index 0) and Lon (index 1)
        for i in range(2):
            nf = self.spatial_num_freqs_list[i]
            if nf > 0:
                freqs_i = getattr(self, f"spatial_freqs_{i}")
                harmonics = x_scaled[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
                out_parts.append(torch.sin(harmonics))
                out_parts.append(torch.cos(harmonics))

        # 2. Gaussian Fourier Features for log_pressure (index 2)
        if len(self.pressure_frequencies) > 0:
            p_coord = x_scaled[:, 2:3]  # (B, 1)
            p_proj = p_coord.unsqueeze(1) * self.pressure_frequencies.unsqueeze(0) * (2.0 * torch.pi)
            p_proj_flat = p_proj.reshape(x.shape[0], -1)
            out_parts.append(torch.sin(p_proj_flat))
            out_parts.append(torch.cos(p_proj_flat))

        return torch.cat(out_parts, dim=-1)


class HarmonicSpatialWaveletPressureEncoding(nn.Module):
    """
    Hybrid Encoding: Harmonic (Fourier) for spatial dims (lat, lon),
    and Morlet Wavelets for vertical coordinate (log_pressure).
    """

    def __init__(
        self,
        d_input: int | DictConfig = 3,
        spatial_num_freqs: int | list[int] | DictConfig | ListConfig = 8,
        spatial_freq_mode: str = "linear",
        pressure_num_scales: int = 12,
        include_input: Union[bool, List[bool], DictConfig, ListConfig] = True,
    ) -> None:
        super().__init__()

        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(spatial_num_freqs, DictConfig):
            spatial_num_freqs = instantiate(spatial_num_freqs)
        elif isinstance(spatial_num_freqs, ListConfig):
            spatial_num_freqs = list(spatial_num_freqs)
        if isinstance(include_input, DictConfig):
            include_input = instantiate(include_input)
        elif isinstance(include_input, ListConfig):
            include_input = list(include_input)

        self.d_input = d_input
        if d_input != 3:
            raise ValueError(f"Hybrid encodings expect d_input=3 [lat, lon, log_pressure], got {d_input}.")

        if isinstance(include_input, bool):
            self.include_input = [include_input] * d_input
        else:
            self.include_input = [bool(v) for v in include_input]

        # 1. Harmonic spatial frequencies
        spatial_dims = 2
        if isinstance(spatial_num_freqs, int):
            self.spatial_num_freqs_list = [spatial_num_freqs] * spatial_dims
        else:
            self.spatial_num_freqs_list = [int(nf) for nf in spatial_num_freqs]

        for i, nf in enumerate(self.spatial_num_freqs_list):
            if nf > 0:
                self.register_buffer(f"spatial_freqs_{i}", HarmonicSpatialGaussianPressureEncoding._make_harmonic_freqs(nf, spatial_freq_mode))

        # 2. Morlet Wavelet scales for log_pressure
        scales = torch.logspace(-0.5, 0.5, pressure_num_scales)
        self.register_buffer("pressure_scales", scales)

        raw_count = sum(1 for inc in self.include_input if inc)
        spatial_encoded = sum(2 * nf for nf in self.spatial_num_freqs_list if nf > 0)
        pressure_encoded = 2 * pressure_num_scales  # envelope*cos, envelope*sin per scale
        self.d_output = raw_count + spatial_encoded + pressure_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_parts = []
        x_scaled = x * 2.0 - 1.0

        for i, inc in enumerate(self.include_input):
            if inc:
                out_parts.append(x_scaled[:, i:i + 1])

        # 1. Harmonic Encoding for lat, lon (indices 0 and 1)
        for i in range(2):
            nf = self.spatial_num_freqs_list[i]
            if nf > 0:
                freqs_i = getattr(self, f"spatial_freqs_{i}")
                harmonics = x_scaled[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
                out_parts.append(torch.sin(harmonics))
                out_parts.append(torch.cos(harmonics))

        # 2. Morlet Wavelet for log_pressure (index 2)
        p_coord = x_scaled[:, 2:3]  # (B, 1)
        scaled_p = p_coord / self.pressure_scales.unsqueeze(0)  # (B, pressure_num_scales)
        envelope = torch.exp(-0.5 * (scaled_p ** 2))
        out_parts.append(envelope * torch.cos(5.0 * scaled_p))
        out_parts.append(envelope * torch.sin(5.0 * scaled_p))

        return torch.cat(out_parts, dim=-1)


class HarmonicSpatialTentPressureEncoding(nn.Module):
    """
    Hybrid Encoding: Harmonic (Fourier) for spatial dims (lat, lon),
    and Triangular Tent Functions for vertical coordinate (log_pressure).
    """

    def __init__(
        self,
        d_input: int | DictConfig = 3,
        spatial_num_freqs: int | list[int] | DictConfig | ListConfig = 8,
        spatial_freq_mode: str = "linear",
        pressure_num_centers: int = 16,
        include_input: Union[bool, List[bool], DictConfig, ListConfig] = True,
    ) -> None:
        super().__init__()

        if isinstance(d_input, DictConfig):
            d_input = instantiate(d_input)
        if isinstance(spatial_num_freqs, DictConfig):
            spatial_num_freqs = instantiate(spatial_num_freqs)
        elif isinstance(spatial_num_freqs, ListConfig):
            spatial_num_freqs = list(spatial_num_freqs)
        if isinstance(include_input, DictConfig):
            include_input = instantiate(include_input)
        elif isinstance(include_input, ListConfig):
            include_input = list(include_input)

        self.d_input = d_input
        if d_input != 3:
            raise ValueError(f"Hybrid encodings expect d_input=3 [lat, lon, log_pressure], got {d_input}.")

        if isinstance(include_input, bool):
            self.include_input = [include_input] * d_input
        else:
            self.include_input = [bool(v) for v in include_input]

        # 1. Harmonic spatial frequencies
        spatial_dims = 2
        if isinstance(spatial_num_freqs, int):
            self.spatial_num_freqs_list = [spatial_num_freqs] * spatial_dims
        else:
            self.spatial_num_freqs_list = [int(nf) for nf in spatial_num_freqs]

        for i, nf in enumerate(self.spatial_num_freqs_list):
            if nf > 0:
                self.register_buffer(f"spatial_freqs_{i}", HarmonicSpatialGaussianPressureEncoding._make_harmonic_freqs(nf, spatial_freq_mode))

        # 2. Tent function centers for log_pressure
        centers = torch.linspace(-1.0, 1.0, pressure_num_centers)
        self.register_buffer("pressure_centers", centers)
        self.pressure_width = 2.0 / max(1, (pressure_num_centers - 1))

        raw_count = sum(1 for inc in self.include_input if inc)
        spatial_encoded = sum(2 * nf for nf in self.spatial_num_freqs_list if nf > 0)
        pressure_encoded = pressure_num_centers
        self.d_output = raw_count + spatial_encoded + pressure_encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_parts = []
        x_scaled = x * 2.0 - 1.0

        for i, inc in enumerate(self.include_input):
            if inc:
                out_parts.append(x_scaled[:, i:i + 1])

        # 1. Harmonic Encoding for lat, lon (indices 0 and 1)
        for i in range(2):
            nf = self.spatial_num_freqs_list[i]
            if nf > 0:
                freqs_i = getattr(self, f"spatial_freqs_{i}")
                harmonics = x_scaled[:, i:i + 1] * freqs_i.unsqueeze(0) * torch.pi
                out_parts.append(torch.sin(harmonics))
                out_parts.append(torch.cos(harmonics))

        # 2. Triangular Tents for log_pressure (index 2)
        p_coord = x_scaled[:, 2:3]  # (B, 1)
        dist = torch.abs(p_coord - self.pressure_centers.unsqueeze(0))
        tents = torch.clamp(1.0 - (dist / self.pressure_width), min=0.0)
        out_parts.append(tents)

        return torch.cat(out_parts, dim=-1)