import torch
import torch.nn as nn
import numpy as np
from typing import Union


class NormalizeProfiles:
    """ Normalize profiles using min-max scaling. """

    def __init__(self, profmin, profmax, inverse_transform=False):
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
        if isinstance(profmin, np.ndarray):
            self.profmin = torch.tensor(profmin if len(profmin.shape) >= 2 else profmin.reshape([9, 1]),
                                        dtype=torch.float32)
        elif isinstance(profmin, torch.Tensor):
            self.profmin = profmin if len(profmin.shape) >= 2 else profmin.view(9, 1)
        else:
            raise TypeError("profmin must be a numpy array or a torch tensor.")
        if isinstance(profmax, np.ndarray):
            self.profmax = torch.tensor(profmax if len(profmax.shape) >= 2 else profmax.reshape([9, 1]),
                                        dtype=torch.float32)
        elif isinstance(profmax, torch.Tensor):
            self.profmax = profmax if len(profmax.shape) >= 2 else profmax.view(9, 1)
        else:
            raise TypeError("profmax must be a numpy array or a torch tensor.")

        # Inverse transform flag
        self.inverse_transform = inverse_transform

    def forward(self, x):
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

        Parameters
        ----------
        device : torch.device. The device to move the parameters to.

        Returns
        -------
        NormalizeProfiles. The instance with parameters moved to the specified device.
        """
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
        self.surfmin = surfmin.astype(np.float32)
        self.surfmax = surfmax.astype(np.float32)

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

        if self.inverse_transform:
            # Inverse normalization
            if isinstance(x, np.ndarray):
                return x * (self.surfmax - self.surfmin) + self.surfmin
            return (x * (torch.tensor(self.surfmax, dtype=torch.float32).to(x.device) -
                         torch.tensor(self.surfmin, dtype=torch.float32).to(x.device)) +
                    torch.tensor(self.surfmin, dtype=torch.float32).to(x.device))
        else:
            # Normalize surface
            if isinstance(x, np.ndarray):
                return (x - self.surfmin) / (self.surfmax - self.surfmin)
            return (x - torch.tensor(self.surfmin, dtype=torch.float32).to(x.device)) / \
                (torch.tensor(self.surfmax, dtype=torch.float32).to(x.device) -
                 torch.tensor(self.surfmin, dtype=torch.float32).to(x.device))

    __call__ = forward  # Make the instance callable for normalization


class NormalizeMeta:
    """ Normalize meta variables using sine and cosine transformations. """

    def __init__(self, settings):
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
        self.meta_scale_factor =settings['meta_scale_factor']
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
