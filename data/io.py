import numpy as np
from omegaconf import DictConfig, ListConfig
from utilities.instantiators import instantiate
from data.transformations import identity
from typing import Union


def load_npy(path: str, split: Union[np.ndarray, int, slice] = None, dtype: str = 'float32') -> np.ndarray:
    """ Load a numpy array from a .npy file and optionally split it.

        Parameters:
        path: str. The file path to the .npy file.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.
        dtype: str. The desired data type of the loaded array. Defaults to 'float32'.

        Returns:
        data: np.ndarray. The loaded (and possibly split) numpy array.
    """

    # Load the numpy array from the specified path
    data = np.load(path, mmap_mode='r')

    # No split: prefer returning the memmap directly when dtype matches
    if split is None:
        if data.dtype == np.dtype(dtype):
            return data
        return data.astype(dtype, copy=True)
    # Apply split
    else:
        # Single index
        if isinstance(split, int):
            return np.asarray(data[split], dtype=dtype)
        # Slice of indices
        elif isinstance(split, slice):
            if data.dtype == np.dtype(dtype):
                return data[split]
            return data[split].astype(dtype=dtype, copy=True)
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


def load_latlon(path: str, scans: np.ndarray = None, split: np.ndarray = None, dtype: str = 'float32') -> np.ndarray:
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


def load_scans(path: str, lat: np.ndarray = None, split: np.ndarray = None, dtype: str = 'float32') -> np.ndarray:
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


def load_var(config: DictConfig, split: Union[np.ndarray, int, slice] = None) -> np.ndarray:
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
    data = np.array(instantiate(config['load']))
    # Apply split if available
    if split is not None and data.shape[0] == len(split):
        data = data[split]
    return data


def load_var_and_normalize(config: DictConfig, split: Union[np.ndarray, int, slice] = None) -> np.ndarray:
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
    f_norm = instantiate(config['normalization']) if hasattr(config, 'normalization') else identity
    # Load and normalize variable
    data = f_norm(load_var(config))
    # Apply split if available
    if split is not None and data.shape[0] == len(split):
        data = data[split]
    return data


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


def extract_indices(indices: np.ndarray, filter: np.ndarray = None) -> np.ndarray:
    """ Extract split indices for a given stage from the configuration.

        Parameters
        ----------
        indices: np.ndarray. Array of indices for the specified stage.
        filter: np.ndarray, optional. Boolean array to filter the indices. Defaults to None.

        Returns
        -------
        np.ndarray. Array of indices for the specified stage, or None if not found.
    """
    return indices[filter[indices]] if filter is not None else indices
