import numpy as np
import pickle
import torch
from omegaconf import DictConfig, ListConfig
from utilities.tensors import to_tensor
from utilities.instantiators import instantiate
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


def load_variable(config: DictConfig | ListConfig, apply_transform: bool = False,
                  as_tensor: bool = False) -> np.ndarray | torch.Tensor:
    """ Load variable.

        Parameters
        ----------
        config: DictConfig or ListConfig. Configuration object for the variable(s).
        apply_transform: bool. Whether to apply transformations or not.
        as_tensor: bool. Whether to return a torch.Tensor or numpy array.

        Returns
        -------
        np.ndarray or torch.Tensor. Array containing the loaded variable(s).
            - If as_tensor=True: torch.Tensor
            - If as_tensor=False: np.ndarray
    """

    # If ListConfig, iterate on each DictConfig and concatenate
    if isinstance(config, ListConfig):
        results = [load_variable(c, apply_transform=apply_transform, as_tensor=as_tensor) for c in config]
        # If tensor
        if as_tensor:
            # Ensure all results are tensors before concatenating
            results = [r if isinstance(r, torch.Tensor) else to_tensor(r) for r in results]
            return torch.cat(results, dim=0)
        # If numpy array
        else:
            # Concatenate
            return np.concatenate(results, axis=0)

    # If individual DictConfig
    elif isinstance(config, DictConfig):

        # Instantiate dataset (with optional transformation of the data)
        if apply_transform:
            dataset = instantiate(config, as_tensor=as_tensor)
        else:
            dataset = instantiate(config, transformations=None, as_tensor=as_tensor)

        # Determine whether the data needs to be lazy loaded or is eager
        # If a path is provided in the load function, then it's eager
        if hasattr(config.load, 'path') and dataset.data is not None:
            return dataset.data
        # If lazy dataset, materialize the data
        else:
            data = dataset.load()
            return data

    # Otherwise raise error
    else:
        raise TypeError('Config must be of type DictConfig or ListConfig.')
