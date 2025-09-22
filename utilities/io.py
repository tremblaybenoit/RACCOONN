import numpy as np
from omegaconf import DictConfig, ListConfig
from utilities.instantiators import instantiate
from forward.data.transformations import identity


def load_npy(path: str, split: np.ndarray = None) -> np.ndarray:
    """ Load a numpy array from a .npy file and optionally split it.

        Parameters:
        path: str. The file path to the .npy file.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.

        Returns:
        data: np.ndarray. The loaded (and possibly split) numpy array.
    """

    # Load the numpy array from the specified path
    data = np.load(path)

    # If a split is provided, return the split data
    if split is not None:
        return data[split]
    return data


def load_latlon(path: str, scans: np.ndarray = None, split: np.ndarray = None) -> np.ndarray:
    """ Load latitude or longitude variable and tile it to match the number of scans.

        Parameters
        ----------
        path: str. The file path to the .npy file.
        scans: np.ndarray. Array of scan indices to determine the number of scans.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.
        Returns
        -------
        latlon: np.ndarray. The loaded and tiled latitude or longitude variable.
    """

    # Load the latitude or longitude variable
    latlon = np.load(path)

    # If scans array is not provided, return the lat/lon as is
    if scans is not None:
        # Extract number of scans from the shape of the loaded array
        n_scans = scans.shape[0]
        # Tile the latitude or longitude variable to match the number of scans
        latlon = np.tile(latlon, n_scans)

    # If a split is provided, return the split data
    if split is not None:
        return latlon[split]
    return latlon


def load_scans(path: str, lat: np.ndarray = None, split: np.ndarray = None) -> np.ndarray:
    """ Load scan variable and repeat it to match the number of coordinates.

        Parameters
        ----------
        path: str. The file path to the .npy file containing the scan variable.
        lat: np.ndarray. Array of latitude values to determine the number of coordinates.
        split: np.ndarray, optional. An array of indices to split the loaded array. Defaults to None.

        Returns
        -------
        scans: np.ndarray. The loaded and repeated scan variable.
    """

    # Load the scan variable
    scans = np.loadtxt(path)

    # If latitude array is not provided, return the scans as is
    if lat is not None:
        # Repeat the scan variable to match the number of coordinates
        n_coords = lat.shape[0]
        scans = np.repeat(scans, n_coords, axis=0)

    # If a split is provided, return the split data
    if split is not None:
        return scans[split]
    return scans


def load_var(config: DictConfig, split: np.ndarray = None) -> np.ndarray:
    """ Load variable.

        Parameters
        ----------
        config: DictConfig. Configuration object for the variables.
        split : np.ndarray. Array of indices for the specified stage.

        Returns
        -------
        dict. Array containing the loaded variable.
    """

    # Load and normalize variable
    data = np.array(instantiate(config['load']), dtype=np.float32)
    # Apply split if available
    if split is not None and data.shape[0] == len(split):
        data = data[split]
    return data


def load_var_and_normalize(config: DictConfig, split: np.ndarray = None) -> np.ndarray:
    """ Load and normalize variable.

        Parameters
        ----------
        config: DictConfig. Configuration object for the variables.
        split : np.ndarray. Array of indices for the specified stage.

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
