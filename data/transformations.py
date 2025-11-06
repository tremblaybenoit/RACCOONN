from typing import Union, Dict
import numpy as np
import torch


class NormalizeProfiles:
    """ Normalize profiles using min-max scaling. """

    def __init__(self, profmin: np.ndarray, profmax: np.ndarray, inverse_transform: bool=False, dtype: str='float32'):
        """ Initialize NormalizeProfiles.

        Parameters
        ----------
        profmin : np.ndarray. Minimum profile values.
        profmax : np.ndarray. Maximum profile values.
        inverse_transform : bool. If True, applies inverse normalization.
        dtype : str. Data type for the profiles (default is 'float64').

        Returns
        -------
        None.
        """
        super().__init__()

        # Load min and max profiles
        self.profmin = profmin.astype(dtype) if len(profmin.shape) >= 2 \
            else profmin.reshape([9, 1]).astype(dtype)

        self.profmax = profmax.astype(dtype)  if len(profmax.shape) >= 2 \
            else profmax.reshape([9, 1]).astype(dtype)

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
            self.profmin = torch.tensor(self.profmin)
        if not isinstance(self.profmax, torch.Tensor):
            self.profmax = torch.tensor(self.profmax)
        self.profmin = self.profmin.to(device)
        self.profmax = self.profmax.to(device)
        return self


class NormalizeSurface:
    """ Normalize surface using min-max scaling. """
    def __init__(self, surfmin, surfmax, inverse_transform=False, dtype='float32'):
        """ Initialize NormalizeSurface.

        Parameters
        ----------
        surfmin : np.ndarray. Minimum surface values.
        surfmax : np.ndarray. Maximum surface values.
        inverse_transform : bool. If True, applies inverse normalization.
        dtype : str. Data type for the surfaces (default is 'float64').

        Returns
        -------
        None.
        """

        super().__init__()

        # Assignment
        surfmin[[0, 1, 2, 3, 6, 11, 12]] = 0
        surfmax[[0, 1, 5, 13, 14, 15]] = 1
        # Load min and max surfaces
        self.surfmin = surfmin.astype(dtype)
        self.surfmax = surfmax.astype(dtype)

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
            self.surfmin = torch.tensor(self.surfmin)
        if not isinstance(self.surfmax, torch.Tensor):
            self.surfmax = torch.tensor(self.surfmax)
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


def broadcast(var1: Union[np.ndarray, torch.Tensor], var2: Union[np.ndarray, torch.Tensor]) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Broadcast var2 to the same dimensions as var1.
        Works for both numpy arrays and torch tensors.

        Parameters
        ----------
        var1: numpy arr or torch tensor. Reference.
        var2: numpy arr or torch tensor. Variable to change dimensions of.

        Returns
        -------
        var2: Broadcast to the same number of dimensions as var1.
    """

    # Handle Python float/int or NumPy scalar
    if isinstance(var2, (float, int, np.floating, np.integer)):
        if isinstance(var1, torch.Tensor):
            var2 = torch.tensor(var2, dtype=var1.dtype, device=var1.device)
        elif isinstance(var1, np.ndarray):
            var2 = np.array(var2, dtype=var1.dtype)
    # For Numpy arrays
    elif isinstance(var2, np.ndarray):
        var2 = np.reshape(var2, (1,) * (var1.ndim - var2.ndim) + var2.shape)
        # Convert to torch tensor if var1 is a torch tensor
        if isinstance(var1, torch.Tensor):
            var2 = torch.from_numpy(var2).to(var1.device, dtype=var1.dtype)
    # For Torch tensors
    elif isinstance(var2, torch.Tensor):
        var2 = var2.view((1,) * (var1.ndim - var2.ndim) + var2.shape)
        # Convert to numpy if var1 is a numpy array
        if isinstance(var1, np.ndarray):
            var2 = var2.cpu().numpy()
        else:
            var2 = var2.to(var1.device, dtype=var1.dtype)
    return var2


def multiplication(data: Union[np.ndarray, torch.Tensor], factor: Union[np.ndarray, torch.Tensor],
                   inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Multiply dataset by a factor.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        factor: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for dividing, True for multiplying.

        Returns
        -------
        data_transform: arr or tensor. Scaled dataset.
    """

    # Broadcast to data dimensions
    scaling_factor = broadcast(data, factor)

    # Unstandardization or standardization
    if inverse_transform:
        # Divide
        eps = np.finfo(scaling_factor.dtype).eps if isinstance(scaling_factor, np.ndarray) \
            else torch.finfo(scaling_factor.dtype).eps
        data_transform = data / (scaling_factor + eps)  # Avoid division by zero
    else:
        # Multiply
        data_transform = data * scaling_factor

    return data_transform


def translation(data: Union[np.ndarray, torch.Tensor], value: Union[np.ndarray, torch.Tensor],
                inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Shift dataset by adding a value.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        value: arr or tensor. Shift to be added to the data.
        inverse_transform: bool. False for subtracting, True for adding.

        Returns
        -------
        data_transform: arr or tensor. Translated dataset.
    """

    # Broadcast to data dimensions
    shift_value = broadcast(data, value)

    # Unstandardization or standardization
    if inverse_transform:
        # Subtract
        data_transform = data - shift_value
    else:
        # Add
        data_transform = data + shift_value

    return data_transform


def affine(data: Union[np.ndarray, torch.Tensor], factor: Union[np.ndarray, torch.Tensor],
           value: Union[np.ndarray, torch.Tensor], inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Affine transformation (linear scaling): data_transform = factor*data + value.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        factor: arr or tensor. Multiply the data by a factor.
        value: arr or tensor. Shift to be added to the data.
        inverse_transform: bool. False for subtracting, True for adding.

        Returns
        -------
        data_transform: arr or tensor. Translated dataset.
    """

    # Affine transformation
    if inverse_transform:
        # data_transform = (data - value)/factor
        data_transform = multiplication(translation(data, value, inverse_transform=inverse_transform),
                                        factor, inverse_transform=inverse_transform)
    else:
        # data_transform = factor*data + value
        data_transform = translation(multiplication(data, factor), value)

    return data_transform


def stdev(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Divide/multiply dataset by stddev.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for standardization, True for unstandardization.

        Returns
        -------
        data_transform: arr or tensor. Standardized/unstandardized dataset.
    """

    return multiplication(data, stats['stdev'], inverse_transform=not inverse_transform)


def mean_stdev(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Standardize dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for standardization, True for unstandardization.

        Returns
        -------
        data_transform: arr or tensor. Standardized/unstandardized dataset.
    """

    return affine(data, stats['stdev'], stats['mean'], inverse_transform=not inverse_transform)


def min_max(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False, axis=None) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Normalize dataset.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization.
        axis: int or None. Axis along which to normalize. If None, normalize across all dimensions.

        Returns
        -------
        data_transform: arr or tensor. Normalized/unnormalized dataset.
    """
    if axis is None:
        return affine(data, stats['max']-stats['min'], stats['min'], inverse_transform=not inverse_transform)
    else:
        return affine(data, stats['max'].max(axis=axis, keepdims=True) - stats['min'].min(axis=axis, keepdims=True),
                      stats['min'].min(axis=axis, keepdims=True), inverse_transform=not inverse_transform)


def max(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False, axis=None)\
        -> Union[np.ndarray, torch.Tensor]:
    """ Divide dataset by its maximum value.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization.
        axis: int or None. Axis along which to normalize. If None, normalize across all dimensions.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """
    if axis is None:
        return multiplication(data, stats['max'], inverse_transform=not inverse_transform)
    else:
        return multiplication(data, stats['max'].max(axis=axis, keepdims=True), inverse_transform=not inverse_transform)


def median(data: Union[np.ndarray, torch.Tensor], stats: Dict, inverse_transform: bool = False) \
           -> Union[np.ndarray, torch.Tensor]:
    """ Divide dataset by its median.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: arr or tensor. Statistics of the data.
        inverse_transform: bool. False for normalization, True for unnormalization

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    return multiplication(data, stats['median'], inverse_transform=not inverse_transform)


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


def sin_cos(data: Union[np.ndarray, torch.Tensor], **kwargs) \
           -> Union[tuple[np.ndarray, ...], tuple[torch.Tensor, ...]]:
    """ Apply sine and cosine transformation to the data.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.

        Returns
        -------
        data_transform: arr or tensor. Transformed dataset.
    """

    return torch.sin(data), torch.cos(data)


def clip(data: Union[np.ndarray, torch.Tensor], stats: Dict) \
           -> Union[np.ndarray, torch.Tensor]:
    """ Clip the data to the specified range.

        Parameters
        ----------
        data: arr or tensor. Contains data to transform.
        stats: dict. Contains 'min' and 'max' values for clipping.

        Returns
        -------
        data_transform: arr or tensor. Clipped dataset.
    """

    # Extract min and max values from stats
    min_value = mean_stdev(stats['min'], stats)
    max_value = mean_stdev(stats['max'], stats)

    # Apply clipping based on the type of data
    if isinstance(data, np.ndarray):
        return np.clip(data, min_value, max_value)
    elif isinstance(data, torch.Tensor):
        min_value = torch.as_tensor(min_value, device=data.device, dtype=data.dtype)
        max_value = torch.as_tensor(max_value, device=data.device, dtype=data.dtype)
        # Diffuser si besoin
        min_value = min_value.expand_as(data)
        max_value = max_value.expand_as(data)
        return torch.clamp(data, min=min_value, max=max_value)
    else:
        raise TypeError("Input data must be a numpy array or a torch tensor.")
