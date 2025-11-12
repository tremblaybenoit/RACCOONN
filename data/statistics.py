from typing import Union
import numpy as np
import pickle
import torch
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from data.io import load_var
from utilities.logic import get_config_path
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def read_statistics(path: str, tensor: bool = False, dtype: str = 'float32') -> dict:
    """ Read statistics from a file.

        Parameters
        ----------
        path: str. Path to the file containing statistics.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Load statistics from file
    with open(path, 'rb') as file:
        stats = pickle.load(file)

    # Convert statistics to torch tensors if required
    if tensor:
        # Loop through each variable in stats and convert numpy arrays to torch tensors
        stats = {var: {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                       for key, value in var_stats.items()} for var, var_stats in stats.items()}

    return stats


def read_statistics_var(path: str, var: str, tensor: bool = False, dtype: str = 'float32') -> dict:
    """ Read statistics of a specific variable from a file.

        Parameters
        ----------
        path: str. Path to the file containing statistics.
        var: str. Variable to read statistics for.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).

        Returns
        -------
        Dictionary containing statistics of the specified variable.
    """

    # Load statistics from file
    stats = read_statistics(path, dtype=dtype)[var]

    # Convert statistics to torch tensors if required
    if tensor:
        stats = {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}
    else:
        # Convert to correct numpy dtype
        stats = {key: value.astype(getattr(np, dtype)) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}

    # Return statistics for the specified variable
    return stats


def accumulate_rmse(stats: list[dict[str, Union[np.ndarray, int]]]) -> dict[str, np.ndarray]:
    """ Combine RMSE of multiple datasets based on the mathematical definition of RMSE.
        The datasets make come from different sources, e.g. different instruments, different heights, etc.
        Thus, they may have been computed from a different number of samples.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of RMSE statistics of different datasets.

        Returns
        -------
        Dictionary containing combined RMSE (as if in a single dataset).
    """
    logger.info("Combining RMSE of all datasets...")

    # Total number of samples
    n_samples = np.sum([stat["n_samples"] for stat in stats], axis=0)

    # Combine RMSE (based on its mathematical definition)
    combined_rmse = np.sqrt(np.sum([stat["n_samples"] * stat["rmse"]**2 for stat in stats], axis=0) / n_samples)

    combined_stats = {"rmse": combined_rmse,
                      "n_samples": n_samples}

    return combined_stats


def compute_rmse(err: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None) \
        -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute RMSE of a given dataset.

        Parameters
        ----------
        err: np.ndarray or torch.Tensor. Dataset to compute RMSE on.
        axis: int or tuple. Axis to compute RMSE along.

        Returns
        -------
        Dictionary containing RMSE of the dataset.
    """

    # Convert torch tensor to a numpy array
    if isinstance(err, torch.Tensor):
        err = err.detach().cpu().numpy()
    # Compute RMSE
    rmse_value = np.sqrt(np.nanmean(err**2, axis=axis))

    stats = {"rmse": rmse_value,
             "n_samples": np.sum(~np.isnan(err), axis=axis)}

    return stats


def accumulate_mean(stats: list[dict[str, Union[np.ndarray, int]]]) -> np.ndarray:
    """ accumulate mean from multiple datasets.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of statistics of different datasets.

        Returns
        -------
        np.ndarray containing accumulate mean.
    """
    logger.info("Aggregating mean from all datasets...")
    n_samples = np.sum([stat["n_samples"] for stat in stats], axis=0)
    return np.sum([stat["n_samples"] * stat["mean"] for stat in stats], axis=0) / n_samples


def accumulate_variance(stats: list[dict[str, Union[np.ndarray, int]]]) -> np.ndarray:
    """ accumulate variance from multiple datasets.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of statistics of different datasets.

        Returns
        -------
        np.ndarray containing accumulate variance.
    """
    logger.info("Aggregating variance from all datasets...")
    n_samples = np.sum([stat["n_samples"] for stat in stats], axis=0)
    var = \
        (np.sum([stat["n_samples"] * (stat["variance"] + (stat["mean"] - accumulate_mean(stats))**2)
                 for stat in stats], axis=0) / n_samples)
    return var


def accumulate_statistics(stats: list[dict[str, Union[np.ndarray, int]]], which: list[str] = None) -> dict[str, np.ndarray]:
    """ Combine statistics of multiple datasets based on the mathematical definition of mean, var, stdev, etc.
        The datasets make come from different sources, e.g. different instruments, different heights, etc.
        Thus, they may have been computed from a different number of samples.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of statistics of different datasets.
        which: List[str]. List of statistics to accumulate.
        Returns
        -------
        Dictionary containing accumulate statistics.
    """
    logger.info("Aggregating specific statistics from all datasets...")
    accumulate_stats = {}

    # Requested statistics
    which = ['min', 'max', 'mean', 'variance', 'stdev'] if which is None else which
        
    # Number of samples
    accumulate_stats['n_samples'] = np.sum([stat["n_samples"] for stat in stats], axis=0)

    # Loop through requested statistics
    if 'min' in which:
        accumulate_stats['min'] = np.min([stat["min"] for stat in stats], axis=0)
    if 'max' in which:
        accumulate_stats['max'] = np.max([stat["max"] for stat in stats], axis=0)
    if 'mean' in which or 'variance' in which or 'stdev' in which:
        accumulate_stats['mean'] = accumulate_mean(stats)
    if 'variance' in which or 'stdev' in which:
        accumulate_stats['variance'] = accumulate_variance(stats)
    if 'stdev' in which:
        accumulate_stats['stdev'] = np.sqrt(accumulate_stats['variance'])
    if 'mae' in which:
        accumulate_stats['mae'] = accumulate_mean([{'mean': stat['mae'], 'n_samples': stat['n_samples']} 
                                                  for stat in stats])
    if 'mape' in which:
        accumulate_stats['mape'] = accumulate_mean([{'mean': stat['mape'], 'n_samples': stat['n_samples']} 
                                                   for stat in stats])
    if 'rmse' in which:
        accumulate_stats['rmse'] = np.sqrt(accumulate_mean([{'mean': stat['rmse']**2, 'n_samples': stat['n_samples']} 
                                                           for stat in stats]))
        
    return accumulate_stats


def statistics(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None, which: list[str] = None,
               target: Union[np.ndarray, torch.Tensor] = None) \
        -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """
    logger.info("Computing statistics of given dataset...")
    stats = {}

    # Requested statistics
    which_allowed = ['min', 'max', 'mean', 'variance', 'stdev', 'rmse', 'mae', 'mape']
    if which is not None:
        which = set(which).intersection(which_allowed)
        which_invalid = set(which).difference(which_allowed)
        if len(which_invalid) > 0:
            logger.warning(
                f"Requested statistics {which_invalid} are not supported for aggregation and will be ignored.")
    else:
        which = ['min', 'max', 'mean', 'variance', 'stdev', 'rmse']

    # Convert torch tensor to a numpy array
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    # Compute basic statistics
    stats['n_samples'] = data.shape[0]
    if 'min' in which:
        stats['min'] = np.nanmin(data, axis=axis)
    if 'max' in which:
        stats['max'] = np.nanmax(data, axis=axis)
    if 'mean' in which:
        stats['mean'] = np.nanmean(data, axis=axis)
    if 'variance' in which:
        stats['variance'] = np.nanvar(data, axis=axis)
    if 'stdev' in which:
        stats['stdev'] = np.nanstd(data, axis=axis)

    # If target is provided, compute error-based statistics
    if target is not None:
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        err = data - target
        if 'rmse' in which:
            stats['rmse'] = np.sqrt(np.nanmean(err**2, axis=axis))
        if 'mae' in which:
            stats['mae'] = np.nanmean(np.abs(err), axis=axis)
        if 'mape' in which:
            stats['mape'] = np.nanmean(np.abs(err / target) * 100, axis=axis)
        err, target = None, None  # Free memory

    # Compute statistics
    return stats


def compute_statistics(input: DictConfig, output: DictConfig = None) -> dict:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Compute statistics per variable
    stats = {}
    variables = list(input.keys())
    # Loop sequentially for memory efficiency (over speed)
    for v, variable in enumerate(variables):
        # Load data
        data = load_var(input[variable])
        # Compute statistics per height
        logger.info(f"Computing statistics of variable {variable} ({v + 1}/{len(variables)})...")
        stats[variable] = statistics(data, axis=0)

    # Save statistics to file
    if output is not None:
        logger.info(f"Saving statistics to file {output.path}.")
        with open(output.path, 'wb') as file:
            # noinspection PyTypeChecker
            pickle.dump(stats, file)

    return stats


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

    # If statistics is part of the preparation steps:
    if hasattr(config.preparation, "statistics"):
        for dataset, config_statistics in config.preparation.statistics.items():
            logger.info(f"Computing statistics of {dataset} set")
            _ = instantiate(config_statistics)

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
