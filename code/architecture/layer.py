import torch
import torch.nn as nn
from omegaconf import DictConfig
from code.data.transformations import Compose


class ScaleLayer(nn.Module):
    """
    Learnable element-wise scaling layer.

    Useful for post-processing: multiplies input by a learnable scale factor.
    Common use: adapting uncertainty magnitude (std output) to data noise.
    """

    def __init__(
            self,
            scale: float = 1.0,
            learnable: bool = True,
    ) -> None:
        """  Set scaling.

        Parameters
        ----------
        scale : float, default=1.0
            Initial scale value.
        learnable : bool, default=True
            If True, scale is a learnable nn.Parameter.
            If False, scale is fixed.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Set scaling to be learnable or constant
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Multiply input by learnable/fixed scale.

        Parameters
        ----------
        x : torch.Tensor to rescale.

        Returns
        -------
        Rescaled tensor.
        """
        return x * self.scale


class OffsetLayer(nn.Module):
    """
    Fixed offset layer for post-processing.

    Adds a constant offset to the input. Used after activation functions
    to shift outputs (e.g., adding minimum uncertainty offset after Softplus).
    """

    def __init__(
            self,
            offset: float = 0.0,
    ) -> None:
        """
        Set offset value.

        Parameters
        ----------
        offset : float, default=0.0
            Offset value to add to input. Stored as a buffer (not learnable).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Register offset as buffer (fixed, not learnable)
        self.register_buffer('offset', torch.tensor(offset, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add offset to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to offset.

        Returns
        -------
        torch.Tensor
            Input with offset added.
        """
        return x + self.offset


class AffineLayer(nn.Module):
    """
    Affine transformation (linear scaling): data_transform = factor*data + value.
    """

    def __init__(
            self,
            factor: float = 1.0,
            value: float = 0.0,
            factor_learnable: bool = True,
            value_learnable: bool = True,
            inverse_transform: bool = True,
    ) -> None:
        """
        Affine transformation (linear scaling): data_transform = factor*data + value.

        Parameters
        ----------
        factor : float, default=1.0
            Initial scale value (multiplier).
        value : float, default=0.0
            Offset value to add (can be learnable or fixed).
        factor_learnable : bool, default=True
            If True, factor is a learnable nn.Parameter.
            If False, it is fixed.
        value_learnable: bool, default=True
            If True, value is a learnable nn.Parameter.
            If False, it is fixed.
        inverse_transform: bool, default=False
            If True, applies inverse transformation (denormalization).
            If False, applies forward transformation (normalization).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Set scaling and offset to be learnable or constant
        if factor_learnable:
            self.scale = nn.Parameter(torch.tensor(factor, dtype=torch.float32))
        else:
            self.register_buffer('scale', torch.tensor(factor, dtype=torch.float32))
        if value_learnable:
            self.offset = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        else:
            self.register_buffer('offset', torch.tensor(value, dtype=torch.float32))

        # Inverse or forward
        self.inverse_transform = inverse_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation: scale * x + offset.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Affine-transformed tensor.
        """

        # If inverse transformation
        if self.inverse_transform:
            return x * self.scale + self.offset
        else:
            return (x - self.offset) / self.scale


class TransformationsLayer(nn.Module):
    """Wrapper for applying a single transformation pipeline as a torch.nn.Module.
    """

    def __init__(self, transformations: DictConfig | None = None,
                 inverse_transform: bool = False) -> None:
        """Initialize transformation layer.

        Parameters
        ----------
        transformations : DictConfig | None
            Transformation config: {_target_: min_max, stats: {...}}
            Or None for identity (no-op).
        
        inverse_transform : bool, default=False
            If True, applies inverse transformations (denormalization).
            If False, applies forward transformations (normalization).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Create Compose transformation pipeline
        self.transform = Compose(transformations, inverse_transform=inverse_transform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformation to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Transformed tensor
        """
        return self.transform(x)


class TransformationsLayers(nn.Module):
    """Wrapper for applying transformations (from code.data.transformations) as a torch.nn.Module.

    Supports both single-variable and multi-variable transformations:
    - Single variable: transformation config → applied to input tensor
    - Multi-variable: dict of configs → applied to dict of tensors
    """

    def __init__(self, transformations: DictConfig,
                 inverse_transform: bool = False) -> None:
        """Initialize transformation layer(s).

        Parameters
        ----------
        transformations : DictConfig
            Single transformation config (DictConfig):
                {_target_: min_max, stats: {...}}
                Applied to single input tensor

            Or dict of transformation configs (for multi-variable):
                {'prof': {_target_: min_max, ...},
                 'surf': {_target_: min_max, ...},
                 'meta': {_target_: min_max, ...}}
                Applied to dict of tensors with matching keys

        inverse_transform : bool, default=False
            If True, applies inverse transformations (denormalization).
            If False, applies forward transformations (normalization).

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Pre-build modules and segregate keys to eliminate runtime string hashing
        modules = {}
        for var_name, var_cfg in transformations.items():
            if var_cfg is not None:
                modules[str(var_name)] = TransformationsLayer(var_cfg, inverse_transform=inverse_transform)

        self.layers = nn.ModuleDict(modules)
        # Store keys as a tuple for ultra-fast iteration
        self.keys = tuple(self.layers.keys())

    def forward(self, x: dict) -> dict:
        """Apply transformation(s) to input.

        Parameters
        ----------
        x : dict
            Dict of tensors (for multi-variable) or
            dict with a single tensor (for single-variable)

        Returns
        -------
        dict
            Transformed input, same type as input
        """

        # Fast path execution using pre-registered keys
        transformed = {}
        for key, val in x.items():

            # Route through transformation layer if it exists, otherwise pass through
            if key in self.keys:
                transformed[key] = self.layers[key](val)
            else:
                transformed[key] = val

        return transformed

    def to(self, device, dtype: torch.dtype | None = None, non_blocking: bool = False):
        """Move all transformation parameters to the specified device.

        Propagates device/dtype to all transformation layers and their underlying Compose objects.

        Parameters
        ----------
        device : torch.device or str
            The device to move parameters to.
        dtype : torch.dtype, optional
            The desired data type (optional).
        non_blocking : bool, default=False
            If True, use asynchronous transfers when possible.

        Returns
        -------
        TransformationsLayers
            Self for method chaining.
        """
        super().to(device, dtype=dtype, non_blocking=non_blocking)
        for layer in self.layers.values():
            if hasattr(layer.transform, 'to'):
                layer.transform.to(device)
        return self
