import torch
import torch.nn as nn
from omegaconf import DictConfig
from utilities.instantiators import instantiate


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


class TransformationLayer(nn.Module):
    """Wrapper for applying a transformation (from code.data.transformations) as a torch.nn.Module."""

    def __init__(self, transformation: DictConfig) -> None:
        """Initialize a transformation layer.

        Parameters
        ----------
        transformation : DictConfig
            Configuration dictionary specifying the transformation to apply.
            Must include '_target_' key pointing to the transformation function.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Transform to apply
        self.transform = instantiate(transformation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        return self.transform(x)


class TransformationsLayer(nn.Module):
    """Wrapper for applying transformations (from code.data.transformations) as a torch.nn.Module.

    Supports both single-variable and multi-variable transformations:
    - Single variable: transformation config → applied to input tensor
    - Multi-variable: dict of configs → applied to dict of tensors
    """

    def __init__(self, transformations: DictConfig | dict) -> None:
        """Initialize transformation layer(s).

        Parameters
        ----------
        transformations : DictConfig or dict
            Single transformation config (DictConfig):
                {_target_: min_max, stats: {...}}
                Applied to single input tensor

            Or dict of transformation configs (for multi-variable):
                {'prof': {_target_: min_max, ...},
                 'surf': {_target_: min_max, ...},
                 'meta': {_target_: min_max, ...}}
                Applied to dict of tensors with matching keys

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Detect single vs multi-variable
        if isinstance(transformations, dict) and not hasattr(transformations, '_target_'):
            # Multi-variable: dict of configs
            self.is_multi = True
            self.transforms = {}
            for var_name, var_cfg in transformations.items():
                if var_cfg is not None:
                    self.transforms[var_name] = instantiate(var_cfg)
        else:
            # Single variable: single config (DictConfig)
            self.is_multi = False
            self.transform = instantiate(transformations)

    def forward(self, x: torch.Tensor | dict) -> torch.Tensor | dict:
        """Apply transformation(s) to input.

        Parameters
        ----------
        x : torch.Tensor or dict
            Single tensor (for single-variable) or
            dict of tensors (for multi-variable)

        Returns
        -------
        torch.Tensor or dict
            Transformed input, same type as input
        """

        if self.is_multi:
            # Multi-variable: x is dict
            output = {}
            for key, value in x.items():
                if key in self.transforms and value is not None:
                    output[key] = self.transforms[key](value)
                else:
                    # Pass through unchanged
                    output[key] = value
            return output
        else:
            # Single variable: x is tensor
            return self.transform(x)