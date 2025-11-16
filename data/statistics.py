from typing import Union
import numpy
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


def torch_nanmin(a: torch.Tensor, axis: Union[int, tuple]=None) -> torch.Tensor:
    """ Compute nanmin along specified axis/axes.

        Parameters
        ----------
        a: torch.Tensor. Input tensor.
        axis: int or tuple. Axis/axes along which to compute nanmin. If None,
              compute over all elements.

        Returns
        -------
        torch.Tensor containing nanmin values.
    """

    # If axis is a tuple, compute nanmin sequentially along each axis
    if isinstance(axis, tuple):
        for d in sorted(axis, reverse=True):
            a = torch.min(a, dim=d).values
        return a
    # If axis is an int or None, compute nanmin along that axis
    return torch.min(a, dim=axis).values


def torch_nanmax(a: torch.Tensor, axis: Union[int, tuple]=None) -> torch.Tensor:
    """ Compute nanmax along specified axis/axes.

        Parameters
        ----------
        a: torch.Tensor. Input tensor.
        axis: int or tuple. Axis/axes along which to compute nanmax. If None,
              compute over all elements.

        Returns
        -------
        torch.Tensor containing nanmax values.
    """

    # If axis is a tuple, compute nanmax sequentially along each axis
    if isinstance(axis, tuple):
        for d in sorted(axis, reverse=True):
            a = torch.max(a, dim=d).values
        return a
    # If axis is an int or None, compute nanmax along that axis
    return torch.max(a, dim=axis).values


def torch_nanmean(a: torch.Tensor, axis: Union[int, tuple]=None) -> torch.Tensor:
    """ Compute nanmean along specified axis/axes.

        Parameters
        ----------
        a: torch.Tensor. Input tensor.
        axis: int or tuple. Axis/axes along which to compute nanmean. If None,
              compute over all elements.

        Returns
        -------
        torch.Tensor containing nanmean values.
    """

    # If axis is a tuple, compute nanmean sequentially along each axis
    if isinstance(axis, tuple):
        for d in sorted(axis, reverse=True):
            a = torch.mean(a, dim=d)
        return a
    # If axis is an int or None, compute nanmean along that axis
    return torch.mean(a, dim=axis)


def torch_nanvar(a: torch.Tensor, axis: Union[int, tuple]=None) -> torch.Tensor:
    """ Compute nanvar along specified axis/axes.

        Parameters
        ----------
        a: torch.Tensor. Input tensor.
        axis: int or tuple. Axis/axes along which to compute nanvar. If None,
              compute over all elements.

        Returns
        -------
        torch.Tensor containing nanvar values.
    """

    # If axis is a tuple, compute nanvar sequentially along each axis
    if isinstance(axis, tuple):
        for d in sorted(axis, reverse=True):
            a = torch.var(a, dim=d)
        return a
    # If axis is an int or None, compute nanvar along that axis
    return torch.var(a, dim=axis)


def torch_nanstd(a: torch.Tensor, axis: Union[int, tuple]=None) -> torch.Tensor:
    """ Compute nanstd along specified axis/axes.

        Parameters
        ----------
        a: torch.Tensor. Input tensor.
        axis: int or tuple. Axis/axes along which to compute nanstd. If None,
              compute over all elements.

        Returns
        -------
        torch.Tensor containing nanstd values.
    """

    # If axis is a tuple, compute nanstd sequentially along each axis
    if isinstance(axis, tuple):
        for d in sorted(axis, reverse=True):
            a = torch.std(a, dim=d)
        return a
    # If axis is an int or None, compute nanstd along that axis
    return torch.std(a, dim=axis)


def torch_nansum_mask(a: torch.Tensor, axis: Union[int, tuple]=None) -> torch.Tensor:
    """ Compute number of non-nan elements along specified axis/axes.

        Parameters
        ----------
        a: torch.Tensor. Input tensor.
        axis: int or tuple. Axis/axes along which to compute number of non-nan elements. If None,
              compute over all elements.

        Returns
        -------
        torch.Tensor containing number of non-nan elements.
    """

    # If axis is a tuple, compute number of non-nan elements sequentially along each axis
    if isinstance(axis, tuple):
        for d in sorted(axis, reverse=True):
            a = (~torch.isnan(a)).sum(dim=d)
        return a
    # If axis is an int or None, compute number of non-nan elements along that axis
    return (~torch.isnan(a)).sum(dim=axis)


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


def accumulate_mean(stats: list[dict[str, Union[np.ndarray, torch.Tensor]]]) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Accumulate mean from multiple datasets.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of statistics of different datasets.

        Returns
        -------
        np.ndarray or torch.Tensor containing accumulate mean.
    """

    # Extract means and number of samples
    means = [stat["mean"] for stat in stats]
    n_samples_list = [stat["n_samples"] for stat in stats]

    # If the means are torch tensors
    if all(isinstance(m, torch.Tensor) for m in means):
        n_samples = torch.sum(torch.stack(n_samples_list, dim=0), dim=0)
        weighted_means = torch.stack([n * m for n, m in zip(n_samples_list, means)], dim=0)
        return torch.sum(weighted_means, dim=0) / n_samples
    # If the means are numpy arrays
    elif all(isinstance(m, numpy.ndarray) for m in means):
        n_samples = np.sum(np.stack(n_samples_list, axis=0), axis=0)
        weighted_means = np.stack([n * m for n, m in zip(n_samples_list, means)], axis=0)
        return np.sum(weighted_means, axis=0) / n_samples
    else:
        raise TypeError("All means must be either numpy arrays or torch tensors.")


def accumulate_variance(stats: list[dict[str, Union[np.ndarray, torch.Tensor]]]) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Accumulate variance from multiple datasets.

        Parameters
        ----------
        stats: List[Dict[str, Union[np.ndarray, int]]]. List of statistics of different datasets.

        Returns
        -------
        np.ndarray or torch.Tensor containing accumulate variance.
    """

    # Extract means, variances, and number of samples
    means = [stat["mean"] for stat in stats]
    variances = [stat["variance"] for stat in stats]
    n_samples_list = [stat["n_samples"] for stat in stats]
    # Accumulate mean for variance calculation
    accumulated_mean = accumulate_mean(stats)

    # If the statistics are torch tensors
    if all(isinstance(m, torch.Tensor) for m in means) and \
       all(isinstance(v, torch.Tensor) for v in variances):
        n_samples = torch.sum(torch.stack(n_samples_list, dim=0), dim=0)
        var = torch.sum(torch.stack([(n * (var + (mean - accumulated_mean)**2)) / n_samples
                                     for n, mean, var in zip(n_samples_list, means, variances)], dim=0), dim=0)
    # If the statistics are numpy arrays
    elif all(isinstance(m, numpy.ndarray) for m in means) and \
         all(isinstance(v, numpy.ndarray) for v in variances):
        n_samples = np.sum(np.stack(n_samples_list, axis=0), axis=0)
        var = np.sum(np.stack([n * (var + (mean - accumulated_mean)**2) / n_samples
                               for n, mean, var in zip(n_samples_list, means, variances)], axis=0), axis=0)
    else:
        raise TypeError("All means and variances must be either numpy arrays or torch tensors.")

    return var


def accumulate_statistics(stats: list[dict[str, Union[np.ndarray, torch.Tensor]]],
                          which: list[str] = None) -> dict[str, np.ndarray]:
    """ Combine statistics of multiple datasets based on the mathematical definition of mean, var, stdev, etc.
        The datasets make come from different sources, e.g. different instruments, different heights, etc.
        Thus, they may have been computed from a different number of samples.

        Parameters
        ----------
        stats: List[dict[str, Union[np.ndarray, torch.Tensor, int, torch.int]]]. List of stats of different datasets.
        which: List[str]. List of statistics to accumulate.

        Returns
        -------
        Dictionary containing accumulate statistics.
    """

    # Requested statistics
    accumulate_stats = {}
    which = list(stats[0].keys()) if which is None else which
        
    # Number of samples
    accumulated_samples = [stat["n_samples"] for stat in stats]
    accumulate_stats['n_samples'] = np.sum(np.stack(accumulated_samples, axis=0), axis=0) if isinstance(accumulated_samples[0], np.ndarray) \
        else torch.sum(torch.stack(accumulated_samples, dim=0), dim=0)

    # Loop through requested statistics
    if 'min' in which:
        accumulated_min = [stat["min"] for stat in stats]
        accumulate_stats['min'] = np.min(accumulated_min, axis=0) if isinstance(accumulated_min[0], np.ndarray) \
            else torch.min(torch.stack(accumulated_min, dim=0), dim=0).values
    if 'max' in which:
        accumulated_max = [stat["max"] for stat in stats]
        accumulate_stats['max'] = np.max(accumulated_max, axis=0) if isinstance(accumulated_max[0], np.ndarray) \
            else torch.max(torch.stack(accumulated_max, dim=0), dim=0).values
    if 'mean' in which or 'variance' in which or 'stdev' in which:
        accumulate_stats['mean'] = accumulate_mean(stats)
    if 'variance' in which:
        accumulate_stats['variance'] = accumulate_variance(stats)
    if 'stdev' in which:
        if 'variance' in which:
            var = accumulate_variance(stats)
        else:
            var = accumulate_variance([{'variance': stat['stdev']**2, 'mean': stat['mean'],
                                        'n_samples': stat['n_samples']} for stat in stats])
        accumulate_stats['stdev'] = np.sqrt(var) if isinstance(var, np.ndarray) else torch.sqrt(var)
    if 'mae' in which:
        accumulate_stats['mae'] = accumulate_mean([{'mean': stat['mae'], 'n_samples': stat['n_samples']} 
                                                  for stat in stats])
    if 'mape' in which:
        accumulate_stats['mape'] = accumulate_mean([{'mean': stat['mape'], 'n_samples': stat['n_samples']} 
                                                   for stat in stats])
    if 'rmse' in which:
        accumulated_mean = accumulate_mean([{'mean': stat['rmse']**2, 'n_samples': stat['n_samples']}
                                            for stat in stats])
        accumulate_stats['rmse'] = np.sqrt(accumulated_mean) if isinstance(accumulated_mean, np.ndarray) \
            else torch.sqrt(accumulated_mean)
        
    return accumulate_stats


def statistics(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = 0, which: list[str] = None,
               target: Union[np.ndarray, torch.Tensor] = None) \
        -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.
        which: List[str]. List of statistics to compute.
        target: np.ndarray or torch.Tensor. Target dataset to compute error-based statistics.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Allowed statistics
    stats = {}
    which_allowed = ['min', 'max', 'mean', 'variance', 'stdev', 'rmse', 'mae', 'mape']
    if which is not None:
        which = set(which).intersection(which_allowed)
        which_invalid = set(which).difference(which_allowed)
        if len(which_invalid) > 0:
            logger.warning(f"Requested statistics {which_invalid} are not supported and will be ignored.")
    else:
        which = ['min', 'max', 'mean', 'variance', 'stdev', 'rmse']

    # If the data is a torch tensor
    if isinstance(data, torch.Tensor):

        # Compute basic statistics (torch)
        stats['n_samples'] = torch_nansum_mask(data, axis=axis)
        if 'min' in which:
            stats['min'] = torch_nanmin(data, axis=axis)
        if 'max' in which:
            stats['max'] = torch_nanmax(data, axis=axis)
        if 'mean' in which:
            stats['mean'] = torch_nanmean(data, axis=axis)
        if 'variance' in which:
            stats['variance'] = torch_nanvar(data, axis=axis)
        if 'stdev' in which:
            stats['stdev'] = torch_nanstd(data, axis=axis)

        # Compute error-based statistics (torch)
        if target is not None:
            # Ensure target is a torch tensor
            if not isinstance(target, torch.Tensor):
                raise TypeError("Target must be a torch.Tensor when data is a torch.Tensor.")
            # Compute error
            err = data - target
            if 'rmse' in which:
                stats['rmse'] = torch.sqrt(torch_nanmean(err**2, axis=axis))
            if 'mae' in which:
                stats['mae'] = torch_nanmean(torch.abs(err), axis=axis)
            if 'mape' in which:
                stats['mape'] = torch_nanmean(torch.abs(err/target)*100, axis=axis)
            # Free memory
            err = None

    # If the data is a numpy array
    elif isinstance(data, numpy.ndarray):
        # Compute basic statistics (numpy)
        stats['n_samples'] = np.sum(~np.isnan(data), axis=axis)
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

        # Compute error-based statistics (numpy)
        if target is not None:
            # Ensure target is a numpy array
            if not isinstance(target, np.ndarray):
                raise TypeError("Target must be a np.ndarray when data is a np.ndarray.")
            # Compute error
            err = data - target
            if 'rmse' in which:
                stats['rmse'] = np.sqrt(np.nanmean(err**2, axis=axis))
            if 'mae' in which:
                stats['mae'] = np.nanmean(np.abs(err), axis=axis)
            if 'mape' in which:
                stats['mape'] = np.nanmean(np.abs(err/target)*100, axis=axis)
            # Free memory
            err = None

    # If the data is neither a numpy array nor a torch tensor, raise an error
    else:
        raise TypeError("Data must be either a numpy.ndarray or a torch.Tensor.")

    return stats


# def compute_statistics_lazy(input: DictConfig, output: DictConfig = None, batch_size: int = None) -> dict:
    """ Compute statistics of a given dataset using lazy loading.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Main hydra configuration file containing all model hyperparameters.
        batch_size: int. Size of the batches (files or chunks) to use for lazy loading.

        Returns
        -------
        None.
    """

    # Compute statistics per variable, per file(s)



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
