from typing import Union, Dict, List
import numpy as np
import pickle
import torch
import hydra
from omegaconf import DictConfig
from forward.utilities.instantiators import instantiate
from forward.utilities.logic import get_config_path
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def read_statistics(filename: str, tensor: bool = False) -> Dict:
    """ Read statistics from a file.

        Parameters
        ----------
        filename: str. Path to the file containing statistics.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Load statistics from file
    with open(filename, 'rb') as file:
        stats = pickle.load(file)

    # Convert statistics to torch tensors if required
    if tensor:
        # Loop through each variable in stats and convert numpy arrays to torch tensors
        stats = {var: {key: torch.tensor(value, dtype=torch.float32) if isinstance(value, np.ndarray) else value
                       for key, value in var_stats.items()} for var, var_stats in stats.items()}

    return stats


def read_statistics_var(filename: str, var: str, tensor: bool = False) -> Dict:
    """ Read statistics of a specific variable from a file.

        Parameters
        ----------
        filename: str. Path to the file containing statistics.
        var: str. Variable to read statistics for.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.

        Returns
        -------
        Dictionary containing statistics of the specified variable.
    """

    # Load statistics from file
    stats = read_statistics(filename)[var]

    # Convert statistics to torch tensors if required
    if tensor:
        stats = {key: torch.tensor(value, dtype=torch.float32) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}

    # Return statistics for the specified variable
    return stats


def combine_statistics(stats: List[Dict[str, Union[np.ndarray, int]]]) \
        -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Combine statistics of multiple datasets based on the mathematical definition of mean, var, stdev, etc.
        The datasets make come from different sources, e.g. different instruments, different heights, etc.
        Thus, they may have been computed from a different number of samples.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, torch.Tensor]]]. List of statistics of different datasets.

        Returns
        -------
        Dictionary containing combined statistics (as if in a single dataset).
    """
    logger.info("Combining statistics of all datasets...")

    # Initialize combined statisticsL min, max, mean
    n_samples = np.sum([stat["n_samples"] for stat in stats], axis=0)
    combined_stats = {"min": np.min([stat["min"] for stat in stats], axis=0),
                      "max": np.max([stat["max"] for stat in stats], axis=0),
                      "mean": (np.sum([stat["n_samples"] * stat["mean"] for stat in stats], axis=0) / n_samples)}

    # Combine variance (based on its mathematical definition)
    combined_stats["variance"] = \
        (np.sum([stat["n_samples"] * (stat["variance"] + (stat["mean"] - combined_stats["mean"])**2)
                 for stat in stats], axis=0) / n_samples)

    # Combine stdev (based on its mathematical definition)
    combined_stats["stdev"] = np.sqrt(combined_stats["variance"])

    return combined_stats


def statistics_positive(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None) \
        -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a given dataset for non-zero values (unless it's all zeros).

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Convert torch tensor to numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Replace zeros with NaN to avoid computing statistics on them
    data = np.where(data <= 0, np.nan, data)

    # Create dictionary to store statistics
    stats = {"mean": np.nanmean(data, axis=axis), "stdev": np.nanstd(data, axis=axis),
             "min": np.nanmin(data, axis=axis), "max": np.nanmax(data, axis=axis),
             "median": np.nanmedian(data, axis=axis), "variance": np.nanvar(data, axis=axis),
             "n_samples": np.sum(~np.isnan(data), axis=axis)}

    # If all values are zero at a given position in stats, set them to zero
    for key in stats:
        # Identify positions where all values are NaN
        nan_positions = np.isnan(stats[key])
        stats[key][nan_positions] = 0

    # Compute statistics
    return stats


def statistics(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None) \
        -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Convert torch tensor to a numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Create a dictionary to store statistics
    stats = {"mean": np.nanmean(data, axis=axis), "stdev": np.nanstd(data, axis=axis),
             "min": np.nanmin(data, axis=axis), "max": np.nanmax(data, axis=axis),
             "median": np.nanmedian(data, axis=axis), "variance": np.nanvar(data, axis=axis),
             "n_samples": np.sum(~np.isnan(data), axis=axis)}

    # Compute statistics
    return stats


def compute_statistics(config: DictConfig, variables: List[str]) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        variables: List[str]. List of variables to compute statistics for
                   (e.g. ['prof', 'surf', 'meta', 'hofx', 'lat', 'lon', 'scans', 'pressure']).

        Returns
        -------
        None.
    """

    # Filter clearsky or not
    clrsky = None
    if hasattr(config, 'clrsky_filter'):
        if config.clrsky_filter:
            # Load profiles
            prof = np.array(instantiate(config.input.data['prof'].load)).astype(np.float32)
            # Find indices of clear-sky profiles
            clrsky = np.logical_and(prof[:, 5, :].sum(axis=1) == 0, prof[:, 6, :].sum(axis=1) == 0)
            clrsky = np.logical_and(clrsky, prof[:, 7, :].sum(axis=1) == 0)

    # Compute statistics per variable
    stats = {}
    # Loop sequentially for memory efficiency (over speed)
    for v, variable in enumerate(variables):
        # Load data
        data = np.array(instantiate(config.input.data[variable].load)).astype(np.float32)
        # Compute statistics per height
        logger.info(f"Computing statistics of variable {variable} ({v + 1}/{len(variables)})...")
        if variable in ['prof', 'surf', 'meta', 'hofx'] and clrsky is not None:
            stats[variable] = statistics(data[~clrsky, ...], axis=0)
        else:
            stats[variable] = statistics(data, axis=0)

    # Save statistics to file
    with open(config.output.data.path, 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(stats, file)
    # Clear memory
    io = None

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute statistics of a given dataset.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # If statistics is part of the radiance preparation steps:
    if hasattr(config.data.sets.preparation, "statistics"):
        logger.info("Computing statistics of data...")
        # Compute radiance statistics per filter per instrument
        compute_statistics(config.data.sets.preparation.statistics,
                           variables=['prof', 'surf', 'meta', 'hofx', 'lat', 'lon', 'scans', 'pressure'])

    return


if __name__ == '__main__':
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        zarr file containing data statistics.
    """

    main()
