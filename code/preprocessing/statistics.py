import numpy as np
import torch
import hydra
import os
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from tqdm import tqdm
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def read_statistics(load: DictConfig, tensor: bool = False, dtype: str = 'float32') -> dict:
    """ Read statistics from a file.

        Parameters
        ----------
        load: DictConfig. Configuration for loading the statistics.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Load statistics from file
    stats = instantiate(load)

    # Convert statistics to torch tensors if required
    if tensor:
        # Loop through each variable in stats and convert numpy arrays to torch tensors
        stats = {var: {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                       for key, value in var_stats.items()} for var, var_stats in stats.items()}

    return stats


def read_statistics_var(load: DictConfig, key: str, tensor: bool = False, dtype: str = 'float32',
                        split: DictConfig | slice | None = None) -> dict:
    """ Read statistics of a specific variable from a file.

        Parameters
        ----------
        load: DictConfig. Configuration for loading the statistics.
        key: str. Variable to read statistics for.
        tensor: bool. If True, returns statistics as torch tensors, otherwise as numpy arrays.
        dtype: str. Data type of the torch tensors (if tensor=True).
        split: slice or None. If provided, slice the statistics along the first axis.

        Returns
        -------
        Dictionary containing statistics of the specified variable.
    """

    # Load statistics from file
    stats = read_statistics(load, dtype=dtype)[key]

    # Convert statistics to torch tensors if required
    if tensor:
        stats = {key: torch.tensor(value, dtype=getattr(torch, dtype)) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}
    else:
        # Convert to correct numpy dtype
        stats = {key: value.astype(getattr(np, dtype)) if isinstance(value, np.ndarray) else value
                 for key, value in stats.items()}

    # If key is hofx, only read the first 10 values
    # if key == 'hofx':
    #    stats = {key: value[0:10] for key, value in stats.items()}

    # Apply slicing if split is provided
    if split is not None:
        if isinstance(split, DictConfig):
            split = instantiate(split)
        stats = {key: value[split] if isinstance(value, np.ndarray) else value for key, value in stats.items()}

    # Return statistics for the specified variable
    return stats


def statistics_dataset(
    dataset,
    which: list[str] | None = None,
    axis: int | tuple | None = 0,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """ Compute statistics for a single variable's dataset using online RunningStats.

        Iterates through a univariate dataset (Eager or Lazy) and accumulates
        statistics batch-by-batch using Chan's parallel algorithm.

        Can use either manual iteration (num_workers=0, simple) or PyTorch DataLoader
        (num_workers>0, enables multiprocessing for faster I/O on Lazy datasets).

        Parameters
        ----------
        dataset : UnivariateDataset (EagerDataset, ConstantDataset, or LazyDataset).
                  Already instantiated and forced to as_tensor=False (numpy output).
        which : list[str] or None. Statistics to compute. Defaults to
                ['min', 'max', 'mean', 'stdev'].
        axis : int, tuple, or None. Reduction axis/axes. Typical: 0 for [B, ...].
        batch_size : int. Mini-batch size for accumulation. Default 32.
        num_workers : int. Number of DataLoader workers for multiprocessing.
                      0 (default) uses simple manual iteration; >0 enables DataLoader.

        Returns
        -------
        dict[str, np.ndarray]. Statistics: {'mean': array, 'stdev': array, ...}
    """

    # Statistics to compute
    which = which or ['min', 'max', 'mean', 'stdev']
    # Initialize runner
    runner = RunningStats(which=which)

    logger.info(f"Accumulating statistics over {len(dataset)} samples in batches of {batch_size}...")

    if num_workers > 0:
        # Use DataLoader with multiprocessing
        logger.info(f"  Using DataLoader with {num_workers} workers for I/O parallelism")
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

        for batch in tqdm(loader, desc="Computing stats"):
            # batch is numpy array(s) from dataset[idx]
            runner.update(batch, axis=axis)
    else:
        # Manual iteration (simple, no DataLoader overhead)
        n_samples = len(dataset)
        for batch_start in tqdm(range(0, n_samples, batch_size), desc="Computing stats"):
            # Adjust batch end
            batch_end = min(batch_start + batch_size, n_samples)

            # Collect samples for this batch
            batch_data = []
            for idx in range(batch_start, batch_end):
                sample = dataset[idx]  # numpy array (forced via as_tensor=False)
                batch_data.append(sample)

            # Stack and update runner
            batch_array = np.stack(batch_data, axis=0)
            runner.update(batch_array, axis=axis)

    return runner.compute()


def compute_statistics(input: DictConfig, output: DictConfig | None = None, exclude: list[str] | None = None,
                       which: list[str] | None = None, batch_size: int = 32, axis: int | tuple | None = 0,
                       num_workers: int = 0) -> dict:
    """ Compute statistics for each variable in the dataset.

        Loops through each variable in the input config, instantiates its dataset
        (as univariate, forced to numpy output), and computes statistics using
        RunningStats with Chan's parallel algorithm for numerical stability.

        Parameters
        ----------
        input : DictConfig. Dataset config mapping variable names to dataset configs.
                Structure: {variable_name: {_target_: ..., load: {...}, ...}}
                Each variable's config should be instantiable as a univariate dataset.
        output : DictConfig or None. Output config with 'path' field for saving stats.
        exclude : list[str] or None. Variable names to skip (e.g., masks, flags).
        which : list[str] or None. Statistics to compute. Defaults to
                ['min', 'max', 'mean', 'stdev'].
        batch_size : int. Mini-batch size for accumulation. Default 32.
        axis: int. Axis along which to compute the statistics. Default 0.
        num_workers : int. Number of DataLoader workers. 0 (default) uses manual iteration;
                      >0 enables multiprocessing for faster I/O.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]. Statistics per variable:
            {variable: {stat_name: np.ndarray}, ...}
    """

    # Statistics to compute and to exclude
    which = which or ['min', 'max', 'mean', 'stdev']
    exclude = exclude or []
    # Filter out excluded variables
    variables = {k: v for k, v in input.items() if k not in exclude}

    logger.info(f"Computing statistics for {len(variables)} variable(s): {list(variables.keys())}")

    # Loop through each variable
    stats = {}
    for v, (variable_name, variable_config) in enumerate(variables.items()):
        logger.info(f"Computing statistics of variable '{variable_name}' ({v + 1}/{len(variables)})...")

        # Instantiate the univariate dataset for this variable, force as_tensor=False
        dataset = instantiate(variable_config, transformations=None, as_tensor=False)

        # Compute statistics for this variable
        stats[variable_name] = statistics_dataset(
            dataset=dataset,
            which=which,
            axis=axis,
            batch_size=batch_size,
            num_workers=num_workers
        )

        logger.info(f"  Computed {len(stats[variable_name])} statistics")

    logger.info(f"Statistics computed for {len(stats)} variable(s).")

    # Save statistics to file if output config provided
    if output is not None:
        # Create directory if needed
        if hasattr(output, 'path'):
            logger.info(f"Saving statistics to file {output.path}.")
            os.makedirs(os.path.dirname(output.path), exist_ok=True)
        # Save function
        if hasattr(output, 'save'):
            save_fn = instantiate(output.save)
            save_fn(stats)

    return stats


class RunningStats:
    """ Online statistics accumulator using Chan's parallel algorithm.

        Maintains internal state in float64 for numerical stability regardless
        of input precision. Final results are cast to specified dtype via compute().

        Supports per-variable and per-height statistics by reducing only along
        the sample axis (axis=0 by default), preserving all other dimensions.

        Error-based metrics (rmse, mae, mape) accumulate raw element-wise sums
        rather than per-batch results, avoiding the sqrt-then-accumulate-then-sqrt
        precision loss.

        Usage
        -----
        runner = RunningStats(which=['mean', 'stdev'])
        for batch in data_iterator:
            runner.update(batch, axis=0)
        stats = runner.compute(dtype='float64')   # dict of float64 numpy arrays
    """

    _ALLOWED = {'min', 'max', 'mean', 'variance', 'stdev', 'rmse', 'mae', 'mape'}

    def __init__(self, which: list[str] | None = None):
        """ Initialize the accumulator.

            Parameters
            ----------
            which: list[str] or None. Statistics to track. Defaults to
                   ['min', 'max', 'mean', 'variance', 'stdev'] when None.
                   'rmse', 'mae', 'mape' require a target in update().
        """

        # Statistics to compute
        which_set = set(which) if which is not None else {'min', 'max', 'mean', 'variance', 'stdev'}
        # Filter out invalid metrics
        invalid = which_set - self._ALLOWED
        if invalid:
            logger.warning(f"Unsupported statistics {invalid} will be ignored.")
            which_set -= invalid
        self._which = which_set
        # Determine which metrics require the mean to be computed
        self._needs_mean = bool(self._which & {'mean', 'variance', 'stdev', 'rmse', 'mae', 'mape'})

        # Declare all instance attributes with type hints
        self._n: float | np.ndarray = 0.0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._sum_sq_err: np.ndarray | None = None
        self._sum_abs_err: np.ndarray | None = None
        self._sum_abs_pct_err: np.ndarray | None = None


    def _initialize_accumulators(self) -> None:
        """ Initialize all accumulators to empty state (float64 scalars/arrays). """

        # Count of non-NaN elements
        self._n = 0.0
        # Mean and variance accumulators (Welford's algorithm)
        self._mean = None
        self._m2 = None
        # Min/max
        self._min = None
        self._max = None
        # Error-based metrics (accumulated sums, not per-batch scalars)
        self._sum_sq_err = None
        self._sum_abs_err = None
        self._sum_abs_pct_err = None

    def reset(self) -> None:
        """ Reset all accumulated state to empty. """
        self._initialize_accumulators()

    @staticmethod
    def _as_f64(x: np.ndarray | torch.Tensor) -> np.ndarray:
        """ Convert input to a float64 numpy array. """

        # Convert tensor
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        # Numpy array with float64 precision
        return np.asarray(x, dtype=np.float64)

    def update(self, data: np.ndarray | torch.Tensor,
               target: np.ndarray | torch.Tensor | None = None,
               axis: int | tuple | None = 0) -> None:
        """ Incorporate a new batch into the running statistics.

            Parameters
            ----------
            data: np.ndarray or torch.Tensor. Batch of samples.
                  The sample (reduction) axis is specified by axis.
            target: np.ndarray or torch.Tensor or None. Reference values
                    for error-based metrics (rmse, mae, mape). Must match
                    the shape of data.
            axis: int, tuple, or None. Axis/axes along which to reduce (sample
                  dimension). All other axes are preserved in the output.
        """

        # Convert to float64 numpy array for stable accumulation
        x = self._as_f64(data)
        # Save dimensions
        ndim = x.ndim

        # Normalize axis to a sorted tuple of positive indices
        if axis is None:
            axes = tuple(range(ndim))
        elif isinstance(axis, int):
            axes = (axis % ndim,)
        else:
            axes = tuple(sorted(a % ndim for a in axis))

        # Count non-NaN elements per output cell
        n_b = np.sum(~np.isnan(x), axis=axes)  # output shape

        # Initialize or update accumulators
        if isinstance(self._n, (int, float)) and self._n == 0.0:
            # First batch — initialization
            # Current number of samples
            self._n = n_b.copy()
            # For metrics requiring the mean value
            if self._needs_mean:
                self._mean = np.nanmean(x, axis=axes)
                self._m2 = np.nansum((x - np.nanmean(x, axis=axes, keepdims=True)) ** 2, axis=axes)
            # Min/max
            if 'min' in self._which:
                self._min = np.nanmin(x, axis=axes)
            if 'max' in self._which:
                self._max = np.nanmax(x, axis=axes)
        else:
            # Update - Chan's parallel merge
            # Update number of samples
            n_a = self._n
            n = n_a + n_b
            safe_n = np.where(n > 0, n, 1.0)
            # For metrics requiring the mean value
            if self._needs_mean and self._mean is not None:
                mean_b = np.nanmean(x, axis=axes)
                m2_b = np.nansum((x - np.nanmean(x, axis=axes, keepdims=True)) ** 2, axis=axes)
                delta = mean_b - self._mean
                self._mean += delta * n_b / safe_n
                self._m2 += m2_b + delta ** 2 * n_a * n_b / safe_n
            # Min/max
            if 'min' in self._which and self._min is not None:
                self._min = np.minimum(self._min, np.nanmin(x, axis=axes))
            if 'max' in self._which and self._max is not None:
                self._max = np.maximum(self._max, np.nanmax(x, axis=axes))

            # Update number of samples
            self._n = n

        # Error-based metrics: accumulate raw element-wise sums (not per-batch scalars)
        if target is not None:
            t = self._as_f64(target)
            err = x - t
            if 'rmse' in self._which:
                sq = np.nansum(err ** 2, axis=axes)
                self._sum_sq_err = sq if self._sum_sq_err is None else self._sum_sq_err + sq
            if 'mae' in self._which:
                ab = np.nansum(np.abs(err), axis=axes)
                self._sum_abs_err = ab if self._sum_abs_err is None else self._sum_abs_err + ab
            if 'mape' in self._which:
                pct = np.nansum(np.abs(err / t) * 100, axis=axes)
                self._sum_abs_pct_err = pct if self._sum_abs_pct_err is None else self._sum_abs_pct_err + pct

    def compute(self, dtype: str = 'float64') -> dict[str, np.ndarray]:
        """ Finalize and return statistics in specified dtype.

            Parameters
            ----------
            dtype : str. NumPy dtype string (e.g., 'float64', 'float32').
                   Default 'float64' for numerical consistency.

            Returns
            -------
            dict[str, np.ndarray]. Keys: 'n_samples', plus whichever statistics
            were requested at construction. All arrays are cast to specified dtype
            except 'n_samples' which is int64.
        """

        # Ensure samples have been seen
        if isinstance(self._n, (int, float)) and self._n == 0.0:
            raise RuntimeError("No data accumulated — call update() at least once before compute().")

        # Compute number of samples
        np_dtype = getattr(np, dtype)
        safe_n = np.where(self._n > 0, self._n, 1.0)
        out: dict[str, np.ndarray] = {'n_samples': self._n.astype(np.int64)}

        # Metrics
        if 'min' in self._which and self._min is not None:
            out['min'] = self._min.astype(np_dtype)
        if 'max' in self._which and self._max is not None:
            out['max'] = self._max.astype(np_dtype)
        if 'mean' in self._which and self._mean is not None:
            out['mean'] = self._mean.astype(np_dtype)
        if ('variance' in self._which or 'stdev' in self._which) and self._m2 is not None:
            var = self._m2 / safe_n
            if 'variance' in self._which:
                out['variance'] = var.astype(np_dtype)
            if 'stdev' in self._which:
                out['stdev'] = np.sqrt(var).astype(np_dtype)
        if 'rmse' in self._which:
            if self._sum_sq_err is None:
                logger.warning("'rmse' requested but no target was passed to update() — skipped.")
            else:
                out['rmse'] = np.sqrt(self._sum_sq_err / safe_n).astype(np_dtype)
        if 'mae' in self._which:
            if self._sum_abs_err is None:
                logger.warning("'mae' requested but no target was passed to update() — skipped.")
            else:
                out['mae'] = (self._sum_abs_err / safe_n).astype(np_dtype)
        if 'mape' in self._which:
            if self._sum_abs_pct_err is None:
                logger.warning("'mape' requested but no target was passed to update() — skipped.")
            else:
                out['mape'] = (self._sum_abs_pct_err / safe_n).astype(np_dtype)

        return out


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute statistics.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # If statistics is part of the preprocessing steps:
    if hasattr(config.preprocessing, "statistics"):
        # If single operation, execute
        if hasattr(config.preprocessing.statistics, "_target_"):
            logger.info(f"Computing statistics...")
            _ = instantiate(config.preprocessing.statistics)
        # Execute individual operations
        else:
            for dataset, config_statistics in config.preprocessing.statistics.items():
                logger.info(f"Computing statistics of {dataset} set")
                _ = instantiate(config_statistics)

    return


if __name__ == '__main__':
    """ Compute statistics.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        File containing data statistics.
    """

    main()
