from typing import Any, Optional, Union
import numpy as np
import torch


def _to_numpy_dtype(dtype: Optional[Union[str, torch.dtype, np.dtype]]) -> Optional[np.dtype]:
    """
    Normalize a dtype input to a `np.dtype` or None.

    Accepts: str | torch.dtype | np.dtype | None.
    - torch.dtype -> mapped to corresponding np.dtype for common types.
    - str (e.g. 'float32' or 'torch.float32') -> parsed to np.dtype.
    - np.dtype -> returned as-is.
    - None or unrecognized -> None.
    """

    # If already a numpy dtype
    if isinstance(dtype, np.dtype):
        return dtype

    # Direct mapping for common torch dtypes (fast & explicit)
    if isinstance(dtype, torch.dtype):
        mapping = {
            torch.float32: np.dtype('float32'),
            torch.float64: np.dtype('float64'),
            torch.int32: np.dtype('int32'),
            torch.int64: np.dtype('int64'),
            torch.bool: np.dtype('bool'),
        }
        return mapping.get(dtype)

    # Strings like 'float32' or 'torch.float32' -> parse name
    if isinstance(dtype, str):
        name = dtype.split('.')[-1]
        return np.dtype(name)

    # Let numpy try to interpret the input
    return np.dtype(dtype)


def _to_torch_dtype(dtype: Optional[Union[str, torch.dtype, np.dtype]]) -> Optional[torch.dtype]:
    """
    Normalize a dtype input to a torch.dtype or None.

    Parameters
    ----------
    dtype: str | np.dtype | torch.dtype | None.
        Input dtype specification.

    Returns
    -------
    torch.dtype | None
        Corresponding torch.dtype or None if input is None or unrecognized.
    """

    # If already torch.dtype, return as-is
    if isinstance(dtype, torch.dtype):
        return dtype
    # String like 'float32' => torch.float32
    if isinstance(dtype, str):
        name = dtype.split('.')[-1]
        return getattr(torch, name)
    # If the previous check fails, numpy dtype -> map common types
    npd = np.dtype(dtype)
    mapping = {
        np.dtype('float32'): torch.float32,
        np.dtype('float64'): torch.float64,
        np.dtype('int32'): torch.int32,
        np.dtype('int64'): torch.int64,
        np.dtype('bool'): torch.bool,
    }
    return mapping.get(npd)


def to_torch(obj: Any, dtype: Optional[Union[str, torch.dtype, np.dtype]] = None,
             device: Optional[Union[str, torch.device]] = None) -> Any:
    """
    Recursively convert numpy arrays and Python/numpy scalars inside a data structure to torch.Tensors.

    - numpy.ndarray -> torch.from_numpy(...), then optional .to(dtype, device)
    - Python/numpy scalars -> torch.tensor(...)
    - torch.Tensor -> returned as-is or moved/casted if dtype/device provided
    - Supports dict, list, tuple, set (note: sets with unhashable elements may raise)
    - Other types are returned unchanged

    Parameters
    ----------
    obj : Any
        Object to convert (could be nested).
    dtype : str | np.dtype | torch.dtype | None
        Desired torch dtype (e.g. 'float32' or torch.float32). If None, keep original dtype.
    device : str | torch.device | None
        Desired device (e.g. 'cpu' or 'cuda:0'). If None, keep CPU for new tensors or original device for input tensors.

    Returns
    -------
    Any
        Same structure with numpy arrays and scalars replaced by torch.Tensors.
    """

    # Normalize dtype and assign device
    torch_dtype = _to_torch_dtype(dtype)
    torch_device = torch.device(device) if device is not None else None

    # torch.Tensor -> optionally cast/move
    if isinstance(obj, torch.Tensor):
        if torch_dtype is None and torch_device is None:
            return obj
        # use .to with kwargs as needed
        kwargs = {}
        if torch_dtype is not None:
            kwargs['dtype'] = torch_dtype
        if torch_device is not None:
            kwargs['device'] = torch_device
        return obj.to(**kwargs)

    # numpy array -> torch tensor
    if isinstance(obj, np.ndarray):
        # Map desired torch dtype to numpy dtype to cast on numpy side first
        np_dtype = _to_numpy_dtype(torch_dtype)
        arr = obj
        if np_dtype is not None and arr.dtype != np_dtype:
            # astype(copy=False) will avoid copy if already correct view-compatible
            arr = arr.astype(np_dtype, copy=False)
        # Convert to tensor
        t = torch.from_numpy(arr)
        # If torch dtype still doesn't match (e.g. mapping failed), cast in torch
        if torch_dtype is not None and t.dtype != torch_dtype:
            t = t.to(dtype=torch_dtype)
        if torch_device is not None and torch_device.type != 'cpu':
            t = t.to(device=torch_device)
        return t

    # Python / numpy scalars -> torch.tensor
    if isinstance(obj, (int, float, np.number, bool)):
        kwargs = {}
        if torch_dtype is not None:
            kwargs['dtype'] = torch_dtype
        if torch_device is not None:
            kwargs['device'] = torch_device
        return torch.tensor(obj, **kwargs)

    # dict -> recurse
    if isinstance(obj, dict):
        return {k: to_torch(v, dtype=dtype, device=device) for k, v in obj.items()}

    # list -> recurse and keep list
    if isinstance(obj, list):
        return [to_torch(v, dtype=dtype, device=device) for v in obj]

    # tuple -> recurse and keep tuple
    if isinstance(obj, tuple):
        return tuple(to_torch(v, dtype=dtype, device=device) for v in obj)

    # set -> recurse and keep set (may raise if elements are unhashable)
    if isinstance(obj, set):
        return {to_torch(v, dtype=dtype, device=device) for v in obj}

    # Fallback: return as-is
    return obj


def to_numpy(obj: Any, dtype: Optional[Union[str, np.dtype]] = None) -> Any:
    """
    Recursively convert torch.Tensor objects inside a data structure to numpy arrays.
    - Tensors are detached, moved to CPU and converted with .numpy().
    - Existing numpy arrays are optionally cast to the requested dtype.
    - Supports dict, list, tuple, set and scalar values.
    - Other types are returned unchanged.

    Parameters
    ----------
    obj : Any
        Object to convert (could be nested structures).
    dtype : str | np.dtype | None
        Optional numpy dtype (e.g. 'float32' or np.float32). If provided, arrays are cast to this dtype.

    Returns
    -------
    Any
        Same structure with torch.Tensor replaced by numpy.ndarray.
    """

    # Normalize dtype
    np_dtype = np.dtype(dtype) if dtype is not None else None

    # torch.Tensor -> numpy
    if isinstance(obj, torch.Tensor):
        # Detach first
        t = obj.detach()
        # Move to CPU only if necessary
        if t.device.type != 'cpu':
            t = t.cpu()
        # Convert to numpy
        arr = t.numpy()
        if np_dtype is not None and arr.dtype != np_dtype:
            # astype(copy=False) will copy only if required
            return arr.astype(np_dtype, copy=False)
        return arr

    # numpy array -> optionally cast
    if isinstance(obj, np.ndarray):
        return obj.astype(np_dtype, copy=False) if np_dtype is not None else obj

    # dict -> recurse
    if isinstance(obj, dict):
        return {k: to_numpy(v, dtype) for k, v in obj.items()}

    # list -> recurse and keep list
    if isinstance(obj, list):
        return [to_numpy(v, dtype) for v in obj]

    # tuple -> recurse and keep tuple
    if isinstance(obj, tuple):
        return tuple(to_numpy(v, dtype) for v in obj)

    # set -> recurse and keep set
    if isinstance(obj, set):
        return {to_numpy(v, dtype) for v in obj}

    # python / numpy scalars -> optionally cast floats
    if isinstance(obj, (int, float, np.number)):
        if np_dtype is not None:
            try:
                # Only meaningful for numeric types
                return np.array(obj).astype(np_dtype).item()
            except Exception:
                return obj
        return obj

    # Fallback: return as-is
    return obj
