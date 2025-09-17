import torch
import numpy as np
from typing import Union


class NormalizeProfiles:
    """ Normalize profiles using min-max scaling. """

    def __init__(self, profmin: np.ndarray, profmax: np.ndarray, inverse_transform: bool=False):
        """ Initialize NormalizeProfiles.

        Parameters
        ----------
        profmin : np.ndarray. Minimum profile values.
        profmax : np.ndarray. Maximum profile values.
        inverse_transform : bool. If True, applies inverse normalization.

        Returns
        -------
        None.
        """
        super().__init__()

        # Load min and max profiles
        self.profmin = profmin.astype(np.float64) if len(profmin.shape) >= 2 \
            else profmin.reshape([9, 1]).astype(np.float64)

        self.profmax = profmax.astype(np.float64)  if len(profmax.shape) >= 2 \
            else profmax.reshape([9, 1]).astype(np.float64)

        # Inverse transform flag
        self.inverse_transform = inverse_transform

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """ Forward pass for NormalizeProfiles.

        Parameters
        ----------
        x : torch.Tensor. Input tensor (profiles).

        Returns
        -------
        torch.Tensor. Normalized tensor.
        """

        # Apply transformation
        if self.inverse_transform:
            # Inverse normalization
            return x * self.profmax + self.profmin
        # Normalize profiles
        return (x - self.profmin) / self.profmax

    __call__ = forward  # Make the instance callable for normalization

    def to(self, device):
        """ Move the normalization parameters to the specified device.
            This should only be called if the instance is used within a PyTorch model.

        Parameters
        ----------
        device : torch.device. The device to move the parameters to.

        Returns
        -------
        NormalizeProfiles. The instance with parameters moved to the specified device.
        """

        # If called, convert profmin and profmax to torch tensors and move to device
        if not isinstance(self.profmin, torch.Tensor):
            self.profmin = torch.tensor(self.profmin, dtype=torch.float64)
        if not isinstance(self.profmax, torch.Tensor):
            self.profmax = torch.tensor(self.profmax, dtype=torch.float64)
        self.profmin = self.profmin.to(device)
        self.profmax = self.profmax.to(device)
        return self


class NormalizeSurface:
    """ Normalize surface using min-max scaling. """
    def __init__(self, surfmin, surfmax, inverse_transform=False):
        """ Initialize NormalizeSurface.

        Parameters
        ----------
        surfmin : np.ndarray. Minimum surface values.
        surfmax : np.ndarray. Maximum surface values.
        inverse_transform : bool. If True, applies inverse normalization.

        Returns
        -------
        None.
        """

        super().__init__()

        # Assignment
        surfmin[[0, 1, 2, 3, 6, 11, 12]] = 0
        surfmax[[0, 1, 5, 13, 14, 15]] = 1
        # Load min and max surfaces
        self.surfmin = surfmin.astype(np.float64)
        self.surfmax = surfmax.astype(np.float64)

        # Inverse transform flag
        self.inverse_transform = inverse_transform


    def forward(self, x):
        """ Forward pass for NormalizeSurface.

        Parameters
        ----------
        x : torch.Tensor. Input tensor (surface).

        Returns
        -------
        torch.Tensor. Normalized tensor.
        """

        # Apply transformation
        if self.inverse_transform:
            # Inverse normalization
            return x * (self.surfmax - self.surfmin) + self.surfmin
        else:
            # Normalize surface
            return (x - self.surfmin) / (self.surfmax - self.surfmin)

    __call__ = forward  # Make the instance callable for normalization

    def to(self, device):
        """ Move the normalization parameters to the specified device.
            This should only be called if the instance is used within a PyTorch model.

        Parameters
        ----------
        device : torch.device. The device to move the parameters to.

        Returns
        -------
        NormalizeSurface. The instance with parameters moved to the specified device.
        """

        # If called, convert surfmin and surfmax to torch tensors and move to device
        if not isinstance(self.surfmin, torch.Tensor):
            self.surfmin = torch.tensor(self.surfmin, dtype=torch.float64)
        if not isinstance(self.surfmax, torch.Tensor):
            self.surfmax = torch.tensor(self.surfmax, dtype=torch.float64)
        self.surfmin = self.surfmin.to(device)
        self.surfmax = self.surfmax.to(device)
        return self


class NormalizeMeta:
    """ Normalize meta variables using sine and cosine transformations. """

    def __init__(self, settings: dict):
        """ Initialize NormalizeMeta.

        Parameters
        ----------
        settings : dict. Dictionary containing meta normalization settings.
        - meta_sin_vars : list. Indices of variables to apply sine transformation.
        - meta_cos_vars : list. Indices of variables to apply cosine transformation.
        - meta_scale_vars : list. Indices of variables to apply scaling.
        - meta_scale_factor : float. Scaling factor for the variables.
        - meta_scale_offset : float. Scaling offset for the variables.

        Returns
        -------
        None.
        """

        super().__init__()

        # Load settings
        self.meta_sin_vars = settings['meta_sin_vars']
        self.meta_cos_vars = settings['meta_cos_vars']
        self.meta_scale_vars = settings['meta_scale_vars']
        self.meta_scale_factor = settings['meta_scale_factor']
        self.meta_scale_offset = settings['meta_scale_offset']

    def forward(self, x):
        """ Forward pass for NormalizeMeta.

        Parameters
        ----------
        x : torch.Tensor. Input tensor (meta variables).

        Returns
        -------
        torch.Tensor. Normalized tensor.
        """

        # Normalize meta variables
        meta = torch.remainder(x, 360) if isinstance(x, torch.Tensor) else np.remainder(x, 360)
        meta_list = []

        # Apply scaling
        if self.meta_scale_vars:
            meta_scale = x[:, self.meta_scale_vars]
            meta_scale = (meta_scale + self.meta_scale_offset) / self.meta_scale_factor
            meta_list.append(meta_scale)
        # Convert to radians
        meta_rad = np.pi * meta / 180
        # Apply sine and cosine transformations
        if len(self.meta_sin_vars) > 0:
            meta_sin = torch.sin(meta_rad[:, self.meta_sin_vars]) if isinstance(meta_rad, torch.Tensor) \
                else np.sin(meta_rad[:, self.meta_sin_vars])
            meta_list.append(meta_sin)
        if len(self.meta_cos_vars) > 0:
            meta_cos = torch.cos(meta_rad[:, self.meta_cos_vars]) if isinstance(meta_rad, torch.Tensor) \
                else np.cos(meta_rad[:, self.meta_cos_vars])
            meta_list.append(meta_cos)

        # Concatenate all normalized meta variables
        return torch.cat(meta_list, dim=1) if isinstance(meta_list[0], torch.Tensor) else np.concatenate(meta_list,
                                                                                                         axis=1)

    __call__ = forward  # Make the instance callable for normalization


def identity(data: Union[np.ndarray, torch.Tensor], **kwargs) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Identity transformation.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    return data
