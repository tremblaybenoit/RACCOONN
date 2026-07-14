import numpy as np
import pickle
import torch
from omegaconf import DictConfig, ListConfig
from utilities.instantiators import instantiate
from src.data.transformations import identity
from typing import Literal


def save_pkl(path: str, data: dict) -> None:
    """ Save a dictionary as a pickle file.

        Parameters
        ----------
        path: str. The file path to save the pickle file.
        data: dict. The dictionary to be saved.

        Returns
        -------
        None.
    """

    # Open file
    with open(path, 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(data, file)


def load_pkl(path: str) -> dict:
    """ Load a dictionary from a pickle file.

        Parameters
        ----------
        path: str. The file path to the pickle file.

        Returns
        -------
        data: dict. The loaded dictionary.
    """

    # Open file
    with open(path, 'rb') as file:
        # noinspection PyTypeChecker
        data = pickle.load(file)
    return data


def save_torch(path: str, data: dict | torch.Tensor) -> None:
    """ Save a dictionary as a torch file.

        Parameters
        ----------
        path: str. The file path to save the torch file.
        data: dict. The dictionary to be saved.

        Returns
        -------
        None.
    """

    torch.save(data, path)


def load_torch(path: str) -> dict | torch.Tensor:
    """ Load a dictionary from a torch file.

        Parameters
        ----------
        path: str. The file path to the torch file.

        Returns
        -------
        data: dict or tensor. The loaded data.
    """

    return torch.load(path)


def save_npy(path: str, data: np.ndarray, dtype: str | None = None) -> None:
    """ Save a numpy array to a .npy file.

        Parameters:
        path: str. The file path to save the .npy file.
        data: np.ndarray. The numpy array to be saved.
        dtype: str, optional. The desired data type of the saved array. Defaults to None.

        Returns:
        None.
    """

    if dtype is not None:
        data = data.astype(dtype)
    np.save(path, data)


def load_npy(path: str, split: np.ndarray | int | slice | None = None, dtype: str | None = None,
             mmap_mode: Literal["r+", "r", "w+", "c"] | None =None) -> np.ndarray:
    """ Load a numpy array from a .npy file and optionally split it.

        Parameters:
        path: str. The file path to the .npy file.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.
        dtype: str. The desired data type of the loaded array. Defaults to 'float32'.
        mmap_mode: str, optional. The memory-mapping mode. Defaults to None.

        Returns:
        data: np.ndarray. The loaded (and possibly split) numpy array.
    """

    # Load the numpy array from the specified path
    data = np.load(path, mmap_mode=mmap_mode)
    # Set desired dtype
    if dtype is None:
        dtype = data.dtype

    # No split: prefer returning the memmap directly when dtype matches
    if split is None:
        if data.dtype == np.dtype(dtype):
            return data
        return np.asarray(data, dtype=dtype)
    # Apply split
    else:
        # Single index
        if isinstance(split, int):
            return np.asarray(data[split:split+1].squeeze(0), dtype=dtype)
        # Slice of indices
        elif isinstance(split, slice):
            if data.dtype == np.dtype(dtype):
                return data[split]
            return np.asarray(data[split], dtype=dtype)
        # Array of indices
        else:
            # If boolean array
            if split.dtype == np.bool_:
                split = np.flatnonzero(split)
            # Preallocate output array and use np.take for efficient indexing
            out_shape = (split.shape[0],) + data.shape[1:]
            out = np.empty(out_shape, dtype=dtype)
            np.take(data, split, axis=0, out=out)
            return out


def load_latlon(path: str, scans: np.ndarray | None = None, split: np.ndarray | None = None,
                dtype: str = 'float32') -> np.ndarray:
    """ Load latitude or longitude variable and tile it to match the number of scans.

        Parameters
        ----------
        path: str. The file path to the .npy file.
        scans: np.ndarray. Array of scan indices to determine the number of scans.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.
        dtype: str. The desired data type of the loaded array. Defaults to 'float32'.

        Returns
        -------
        latlon: np.ndarray. The loaded and tiled latitude or longitude variable.
    """

    # Load the latitude or longitude variable
    latlon = np.load(path, mmap_mode='r')

    # If scans array is not provided, return the lat/lon as is
    if scans is not None:
        # Extract number of scans from the shape of the loaded array
        n_scans = scans.shape[0]
        n_coords = latlon.shape[0]

        # If split is provided, adjust the number of scans accordingly
        if split is not None:
            # If split is a boolean array
            if split.dtype == np.bool_:
                pos = np.flatnonzero(split)
                idx = pos % n_coords
            else:
                idx = split % n_coords
            return np.asarray(latlon[idx], dtype=dtype)

        # Tile the latitude or longitude variable to match the number of scans
        return np.tile(np.asarray(latlon, dtype=dtype), n_scans)

    # If a split is provided, return the split data
    if split is not None:
        return np.asarray(latlon[split], dtype=dtype)
    return np.asarray(latlon, dtype=dtype)


def load_scans(path: str, lat: np.ndarray | None = None, split: np.ndarray | None = None,
               dtype: str = 'float32') -> np.ndarray:
    """ Load scan variable and repeat it to match the number of coordinates.

        Parameters
        ----------
        path: str. The file path to the .npy file containing the scan variable.
        lat: np.ndarray. Array of latitude values to determine the number of coordinates.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.
        dtype: str. The desired data type of the loaded array. Defaults to 'float32'.

        Returns
        -------
        scans: np.ndarray. The loaded and repeated scan variable.
    """

    # Load the scan variable
    scans = np.loadtxt(path).astype(dtype)

    # If latitude array is not provided, return the scans as is
    if lat is not None:
        # Repeat the scan variable to match the number of coordinates
        n_coords = lat.shape[0]
        scans = np.repeat(scans, n_coords, axis=0)

    # If a split is provided, return the split data
    if split is not None:
        return scans[split]
    return scans


def load_var(config: DictConfig, split: np.ndarray | int | slice | None = None) -> np.ndarray:
    """ Load variable.

        Parameters
        ----------
        config: DictConfig. Configuration object for the variables.
        split : np.ndarray, slice. Indices for the specified stage.

        Returns
        -------
        dict. Array containing the loaded variable.
    """

    # Load and normalize variable
    data = np.asarray(instantiate(config['load']))

    # If no split and data is already a memmap/ndarray and dtype matches, return directly or asarray
    if split is None:
        # prefer returning memmap unchanged; np.asarray won't copy a memmap
        return np.asarray(data)

    # With split: handle int / slice / fancy indexing convert to array-like only when needed (np.asarray keeps memmap)
    data = np.asarray(data)
    if isinstance(split, int):
        return np.asarray(data[split:split+1].squeeze(0))
    elif isinstance(split, slice):
        return data[split]
    else:
        # Boolean or integer indices: Fancy indexing will produce a copy into a new ndarray
        if split.dtype == np.bool_:
            split = np.flatnonzero(split)
        out_shape = (split.shape[0],) + data.shape[1:]
        out = np.empty(out_shape, dtype=data.dtype)
        np.take(data, split, axis=0, out=out)
        return out


def load_var_and_normalize(config: DictConfig, split: np.ndarray | int | slice | None = None) -> np.ndarray:
    """ Load and normalize variable.

        Parameters
        ----------
        config: DictConfig. Configuration object for the variables.
        split : np.ndarray, slice. Indices for the specified stage.

        Returns
        -------
        dict. Array containing the loaded and normalized variable.
    """

    # Extract normalization function
    if hasattr(config, 'normalization') and config['normalization'] is not None:
        f_norm = instantiate(config['normalization'])
    else:
        f_norm = identity

    # If no split or split is a slice, load and normalize directly
    if split is None or isinstance(split, slice) or isinstance(split, int):
        return f_norm(load_var(config, split=split))

    # If the split is fancy indexing, extract from memory-mapped array
    if isinstance(split, np.ndarray) and split.dtype == np.bool_:
        split = np.flatnonzero(split)
    else:
        split = np.asarray(split)
    # Load variable without split (to keep memmap if possible)
    data = load_var(config, split=None)
    # Extract efficiently using np.take
    out_shape = (split.shape[0],) + data.shape[1:]
    out = np.empty(out_shape, dtype=data.dtype)
    np.take(data, split, axis=0, out=out)
    # Normalize and return
    return f_norm(out)
# TODO: Verify type

def load_stack(stack: ListConfig) -> np.ndarray:
    """ Load and stack multiple variables.

        Parameters
        ----------
        stack: List[DictConfig]. List of configuration objects for the variables.

        Returns
        -------
        np.ndarray. Array containing the loaded and stacked variables.
    """

    # Load and stack variables along the last axis
    return np.concatenate([load_var(c) for c in stack], axis=0)


def load_stack_and_normalize(stack: ListConfig) -> np.ndarray:
    """ Load and stack multiple normalized variables.

        Parameters
        ----------
        stack: List[DictConfig]. List of configuration objects for the variables.

        Returns
        -------
        np.ndarray. Array containing the loaded and stacked normalized variables.
    """

    # Load and stack variables along the last axis
    return np.concatenate([load_var_and_normalize(c) for c in stack], axis=0)
