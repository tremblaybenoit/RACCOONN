from typing import Union, Dict
import numpy as np
import torch


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
        raise TypeError("Unsupported data type. Expected numpy array or torch tensor.")
