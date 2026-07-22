import numpy as np
from scipy.linalg import block_diag
import hydra
from omegaconf import DictConfig
from code.data.io import load_variable
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from code.evaluation.plot import plot_map, save_plot, flexible_gridspec
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def obs_error(data: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """ Return innovation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).
        obs:

        Returns
        -------
        np.ndarray. Solution.
    """
    return data[:, :10]-obs[:, :10]


def obs_uncertainty(data: np.ndarray) -> np.ndarray:
    """ Return uncertainty estimation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray. Spatiotemporal standard deviation of the dataset.
    """

    # Return the standard deviation (last 10 values)
    return data[:, 10:]


def increment(config_true: DictConfig, config_prior: DictConfig) -> np.ndarray:
     """ Compute error between ground truth and prior.

         Parameters
         ----------
         config_true: DictConfig. Configuration for the ground truth dataset.
         config_prior: DictConfig. Configuration for the prior dataset.

         Returns
         -------
         np.ndarray. Error between ground truth and prior.
     """

     # Truth - prior
     x_true = load_variable(config_true, apply_transform=True)
     x_prior = load_variable(config_prior, apply_transform=True)

     # Check dimensions and add new axis if necessary
     if x_true.ndim != x_prior.ndim:
         if x_prior.ndim == x_true.ndim - 1:
             x_prior = x_prior[np.newaxis, :]
         else:
             raise ValueError("The shapes of the true and prior data do not match.")

     # Return increment
     return np.subtract(x_true, x_prior, out=x_true)



def prior_from_bounded_perturbations(input: DictConfig, stats: DictConfig, output: DictConfig | None = None,
                                     seed: int | None = None) -> np.ndarray | None:
    """ Compute the prior from the model error covariance matrix and perturbations.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        stats: DictConfig. Variable statistics to compute upper/lower bounds.
        output: DictConfig. Output configuration.
        seed: int. Seed to ensure reproducibility.

        Returns
        -------
        None.
    """
    # Load Cholesky matrix
    cov_cholesky = load_variable(input.cholesky)

    # Load physical base space and original dimensions
    x_base_phys = load_variable(input.prof)
    x_dims = x_base_phys.shape

    # Load and flatten the standardized/normalized initial state
    xp_base_std = load_variable(input.prof, apply_transform=True).reshape(x_dims[0], -1)

    # Extract physical bounds
    x_stats = instantiate(stats)
    x_min, x_max = x_stats['min'], x_stats['max']

    # Initialize independent random number generator
    rng = np.random.default_rng(seed)

    # Track array for our verified physical outputs
    final_x_phys = np.zeros_like(x_base_phys)

    # Track which profile row indices still violate physical boundaries
    invalid_indices = np.arange(x_dims[0])

    # Safety configuration to prevent infinite loops in tough regimes
    max_iterations = 200
    iteration = 0

    logger.info(f"Starting rejection sampling loop for {x_dims[0]} samples...")

    while len(invalid_indices) > 0 and iteration < max_iterations:
        n_to_resample = len(invalid_indices)
        logger.info(f"Iteration {iteration}: Processing/Resampling {n_to_resample} profiles...")

        # 1. Generate perturbations ONLY for the remaining invalid profiles
        p = rng.normal(0, 1, size=(cov_cholesky.shape[1], n_to_resample))
        dx = (cov_cholesky @ p).T

        # 2. Extract the baseline standardized states for these specific invalid profiles
        xp_current = xp_base_std[invalid_indices].copy()

        # 3. Apply the pressure filter if configured
        if hasattr(input, 'pressure_filter') and input.pressure_filter is not None:
            pressure_filter = instantiate(input.pressure_filter.load)
            if x_dims[1] != pressure_filter.shape[0]:
                pressure_filter = np.take(pressure_filter, [0, 4, 8], axis=0)

            xp_current[:, np.flatnonzero(pressure_filter)] += dx
        else:
            xp_current += dx

        # 4. Safely reshape the current batch to 3D for physical transformation
        xp_current_3d = xp_current.reshape(n_to_resample, x_dims[1], x_dims[2])

        # 5. Transform back to physical space
        if hasattr(input.prof, 'normalization') and input.prof.normalization is not None:
            norm_func = instantiate(input.prof.normalization, inverse_transform=True)
            x_phys_candidate = norm_func(xp_current_3d)
        else:
            x_phys_candidate = xp_current_3d

        # 6. Check boundaries across ALL variables and levels for this sub-batch
        # Evaluates to a boolean array of shape: (n_to_resample, n_vars, n_levels)
        out_of_bounds = (x_phys_candidate < x_min) | (x_phys_candidate > x_max)

        # Collapse dimensions to find which specific profiles failed anywhere in their column
        profile_failed = np.any(out_of_bounds, axis=(1, 2))
        profile_passed = ~profile_failed

        # 7. For the profiles that passed, lock them into the final output array
        if np.any(profile_passed):
            passed_global_indices = invalid_indices[profile_passed]
            final_x_phys[passed_global_indices] = x_phys_candidate[profile_passed]

        # 8. Filter down our invalid pointer list to only contain the persistent failures
        invalid_indices = invalid_indices[profile_failed]
        iteration += 1

    # Loop termination checks
    if len(invalid_indices) > 0:
        raise ValueError(
            f"Rejection sampling failed to converge for {len(invalid_indices)} "
            f"profiles within {max_iterations} iterations. The physical boundaries "
            f"might be too narrow for the specified covariance matrix variance."
        )

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving validated prior to {output.path}...")
        save_func = instantiate(output.save)
        save_func(final_x_phys)
        return None
    else:
        return final_x_phys


def climatological_matrix(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
                          regularization_factor: float = 1.0, plot_flag: bool=True, recenter: bool=False,
                          univariate: bool = False, mean_type: str='spatiotemporal') -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        scaling_factor: float. Scaling factor for covariance matrix.
        regularization_factor: float. Regularization factor for covariance matrix.
        plot_flag: bool. If True, plot the covariance matrix.
        recenter: bool. If True, recenter by removing the mean.
        univariate: bool. If True, compute univariate covariance matrix.
        mean_type: str. Type of mean to compute ('spatiotemporal' or 'temporal').

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    # TODO: Does this make sense?
    data = load_variable(input.data, apply_transform=True)
    data_shape = data.shape
    n_samples, n_vars = data_shape[0], data_shape[1]
    # Denominator (computation of the mean)
    denom = n_samples - 1

    # Apply recentering
    if recenter:
        logger.info("Recentering data around the mean...")
        # Compute spatiotemporal mean
        if mean_type == 'spatiotemporal':
            mu = np.mean(data, axis=0)
            # Compute anomalies (truth - mean)
            data -= mu
        # Compute temporal mean (but maintain coordinate dependency)
        elif mean_type == 'temporal':
            # Read coordinates
            if hasattr(input, 'lat') and hasattr(input, 'lon') and hasattr(input, 'scans'):
                # Read coordinates
                lat = load_variable(input.lat)
                lon = load_variable(input.lon)
                scans = load_variable(input.scans)

                # For clearsky-only or cloud-only datasets, the available coordinates points.
                # In other words, two consecutive timesteps may not have the same (lat, lon) pairs.
                # To compute the temporal mean at every available (lat, lon) point, we...

                # Identify unique coordinate pairs and their mapping
                # coords shape: (n_samples, 2)
                coords = np.column_stack((lat, lon))

                # unique_coords: the actual list of physical locations available
                # inverse_indices: an array of shape (n_samples,) containing the location ID (0 to N-1) for every sample
                unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
                n_unique_coords = len(unique_coords)

                # To be completely safe against whether data is currently 2D or 3D,
                # we flatten the feature/level dimensions temporarily
                data = data.reshape(n_samples, -1)
                n_features = data.shape[1]

                # Vectorized Accumulation (Split-Apply-Combine in NumPy)
                # 1. Allocate a destination array for the sums of each unique coordinate
                group_sums = np.zeros((n_unique_coords, n_features), dtype=data.dtype)

                # 2. np.add.at performs unbuffered in-place addition for repeating indices
                np.add.at(group_sums, inverse_indices, data)

                # 3. Count how many times each unique coordinate appears across all timesteps
                group_counts = np.bincount(inverse_indices)[:, None]  # Shape: (n_unique_coords, 1)

                # 4. Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

                # 5. # Compute anomalies (truth - climatological mean)
                data -= group_means[inverse_indices]  # Shape: (n_samples, n_features)
                # Broadcast the means back out to match the original sample layout
                data = data.reshape(data_shape)
                # Denominator (computation of the mean)
                denom = n_samples - n_unique_coords
            else:
                mu = np.mean(data)
                # Compute anomalies (truth - mean)
                data -= mu
        else:
            raise ValueError("mean_type not supported.")

    # Pressure filter (prior only)
    if hasattr(input, 'pressure_mask') and input.pressure_mask is not None:
        # Load filter
        pressure_mask = instantiate(input.pressure_mask.load)
        if n_vars != pressure_mask.shape[0]:
            pressure_mask = np.take(pressure_mask, [0, 4, 8], axis=0)
    else:
        pressure_mask = None

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <=2:
            raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
        # Initialize empty matrix
        m = []
        # Loop over variables
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply pressure mask
            if pressure_mask is not None:
                logger.info(f"Applying pressure mask for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(pressure_mask[i]), axis=1)
            # Store diagonal block
            logger.info(f"Computing univariate covariance matrix for variable {i}...")
            m.append(scaling_factor*(data_i.T @ data_i)/denom)
        # Assemble
        del data_i
        cov = {'matrix': block_diag(*m)}
    # Multivariate matrix computation steps
    else:
        # Flatten data
        data = data.reshape(n_samples, -1)
        # Apply pressure mask
        if pressure_mask is not None:
            logger.info("Applying pressure mask...")
            # Remove constant pressure levels from the data
            data = np.take(data, np.flatnonzero(pressure_mask), axis=1)
        # Compute matrix
        logger.info("Computing covariance matrix...")
        cov = {'matrix': scaling_factor*(data.T @ data)/denom}
    # Release memory
    del data

    # Apply regularization
    if regularization_factor > 0:
        logger.info("Applying regularization factor...")
        cov['matrix'] += regularization_factor * np.mean(np.diag(cov['matrix'])) * np.eye(cov['matrix'].shape[0])
    # Compute correlation matrix
    if hasattr(output, 'correlation'):
        std = np.sqrt(np.diag(cov['matrix']))
        cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
    # Compute Cholesky decomposition
    if hasattr(output, 'cholesky'):
        logger.info("Computing Cholesky decomposition...")
        cov['cholesky'] = np.linalg.cholesky(cov['matrix'])
    # Compute matrix inverse
    if hasattr(output, 'matrix_inverse'):
        logger.info("Computing the inverse of the covariance matrix...")
        cov['matrix_inverse'] = np.linalg.inv(cov['matrix'])
    # Compute matrix pseudo-inverse
    if hasattr(output, 'matrix_pseudo_inverse'):
        logger.info("Computing the pseudo-inverse of the covariance matrix...")
        rcond = output.matrix_pseudo_inverse.get('params.rcond', 1.e-3)
        cov['matrix_pseudo_inverse'] = np.linalg.pinv(cov['matrix'], rcond=rcond)
    # Compute inverse of the correlation matrix
    if hasattr(output, 'correlation_inverse') and 'correlation' in cov:
        logger.info("Computing the inverse of the correlation matrix...")
        cov['correlation_inverse'] = np.linalg.inv(cov['correlation'])
    # Compute pseudo inverse of the correlation matrix
    if hasattr(output, 'correlation_pseudo_inverse') and 'correlation' in cov:
        logger.info("Computing the pseudo-inverse of the correlation matrix...")
        rcond = output.correlation_pseudo_inverse.get('params.rcond', 1.e-3)
        cov['correlation_pseudo_inverse'] = np.linalg.pinv(cov['correlation'], rcond=rcond)

    # Loop over keys
    for key in list(cov.keys()):
        # Save to file
        if hasattr(output, key):
            if hasattr(output[key], 'save'):
               logger.info(f"Saving {key} matrix...")
               save_func = instantiate(output[key].save)
               save_func(cov[key])
            # Plot covariance matrix if requested
            if plot_flag and key in ['matrix', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute covariance matrices of given datasets.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute model and observation covariance matrices
    if hasattr(config.preprocessing, "covariance"):
        for dataset, config_covariance in config.preprocessing.covariance.items():
            if hasattr(config_covariance, '_target_'):
                logger.info(f"Computing error covariance matrix {dataset}")
                instantiate(config_covariance)

    return


if __name__ == '__main__':
    """ Compute covariance matrices of given datasets.

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