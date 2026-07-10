from typing import Any
import numpy as np
import torch


def array_to_tensor(arr: np.ndarray, shared: bool = True) -> torch.Tensor:
    """ Convert numpy array to torch tensor in shared memory.

        Centralizes the common logic of converting and sharing memory.

        Parameters
        ----------
        arr : np.ndarray. Input array.
        shared : bool. Whether to enable shared memory.

        Returns
        -------
        torch.Tensor. Tensor in shared memory if requested.
    """

    # Convert to contiguous array for efficient memory layout
    arr = np.ascontiguousarray(arr)

    # Create torch tensor from numpy array
    t = torch.from_numpy(arr)

    # Enable sharing across processes if requested
    if shared:
        t.share_memory_()

    return t


def to_array(obj: Any, dtype: str | np.dtype | None = None) -> Any:
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
        return {k: to_array(v, dtype) for k, v in obj.items()}

    # list -> recurse and keep list
    if isinstance(obj, list):
        return [to_array(v, dtype) for v in obj]

    # tuple -> recurse and keep tuple
    if isinstance(obj, tuple):
        return tuple(to_array(v, dtype) for v in obj)

    # set -> recurse and keep set
    if isinstance(obj, set):
        return {to_array(v, dtype) for v in obj}

    # python / numpy scalars -> optionally cast floats
    if isinstance(obj, (int, float, np.number)):
        if np_dtype is not None:
            try:
                # Only meaningful for numeric types
                return np.array(obj).astype(np_dtype).item()
            except (TypeError, ValueError):
                return obj
        return obj

    # Fallback: return as-is
    return obj
