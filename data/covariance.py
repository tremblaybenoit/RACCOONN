import numpy as np
from scipy.linalg import block_diag
import hydra
from omegaconf import DictConfig
from data.io import load_var, load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from utilities.plot import plot_map, save_plot, flexible_gridspec
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
    """ Return uncertainy estimation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray. Spatiotemporal standard deviation of the dataset.
    """

    # Return the standard deviation (last 10 values)
    return data[:, 10:]


def increment(config_true: DictConfig, config_background: DictConfig) -> np.ndarray:
    """ Compute error between ground truth and background.

        Parameters
        ----------
        config_true: DictConfig. Configuration for the ground truth dataset.
        config_background: DictConfig. Configuration for the background dataset.

        Returns
        -------
        np.ndarray. Error between ground truth and background.
    """

    # Truth - Background
    x_true = load_var_and_normalize(config_true)
    x_background = load_var_and_normalize(config_background)

    # Check dimensions and add new axis if necessary
    if x_true.ndim != x_background.ndim:
        if x_background.ndim == x_true.ndim - 1:
            x_background = x_background[np.newaxis, :]
        else:
            raise ValueError("The shapes of the true and background data do not match.")

    # Return increment
    return np.subtract(x_true, x_background, out=x_true)


def regularize_b_matrix_spectral(B: np.ndarray, variance_fraction: float = 0.99, floor_value: float = 1e-3):
    """
    Decomposes B, floors tiny eigenvalues or truncates low-variance modes,
    and returns a symmetric square root matrix W for the CVT transformation.

    Parameters
    ----------
    B : np.ndarray. Original covariance matrix. Shape (M, M)
    variance_fraction : float. Amount of total variance to preserve (e.g., 0.99 = 99%)
    floor_value : float. Absolute minimum eigenvalue threshold to prevent gradient dampening.
    """
    # 1. Compute Eigenvalues (S) and Eigenvectors (V)
    # Since B is symmetric, np.linalg.eigh is fast and stable
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 2. Option A: Truncate based on Cumulative Variance
    # Find where the trailing eigenvalues contribute to less than 1% of the system
    cum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_modes_to_keep = np.searchsorted(cum_variance, variance_fraction) + 1

    # 3. Option B: Floor the remaining eigenvalues so they don't crush gradients
    clipped_eigenvalues = np.copy(eigenvalues)
    # Set anything past the truncation point (or below absolute threshold) to the floor
    clipped_eigenvalues[num_modes_to_keep:] = floor_value
    # clipped_eigenvalues[clipped_eigenvalues < floor_value] = floor_value

    # 4. Construct the Symmetric Square Root Matrix: W = V @ diag(sqrt(eigenvalues)) @ V^T
    sqrt_lambda = np.diag(np.sqrt(clipped_eigenvalues))
    W = eigenvectors @ sqrt_lambda @ eigenvectors.T

    return W


def background_from_perturbations(input: DictConfig, output: DictConfig | None = None,
                                  seed: int | None = None) -> np.ndarray | None:
    """ Compute the background from the model error covariance matrix and perturbations.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        seed: int. Seed to ensure reproducibility.

        Returns
        -------
        None.
    """

    # Load Cholesky matrix
    cov_cholesky = load_var(input.cholesky)
    # Load samples
    x_t = load_var_and_normalize(input.prof)
    x_dims = x_t.shape
    x_t = x_t.reshape(x_dims[0], -1)

    # Initialize independent random number generator
    rng = np.random.default_rng(seed)

    # Compute random perturbations from a normal distribution
    logger.info(f"Generating independent random perturbations for {x_dims[0]} samples...")
    p = rng.normal(0, 1, size=(cov_cholesky.shape[1], x_dims[0]))
    dx = (cov_cholesky @ p).T

    # Pressure filter
    if hasattr(input, 'pressure_filter') and input.pressure_filter is not None:
        logger.info("Applying pressure filter...")
        # Load filter
        pressure_filter = instantiate(input.pressure_filter.load)
        if x_dims[1] != pressure_filter.shape[0]:
            pressure_filter = np.take(pressure_filter, [0, 4, 8], axis=0)
        # Compute background by applying perturbations to samples
        logger.info("Apply perturbations...")
        x_t[:, np.flatnonzero(pressure_filter)] += dx
    else:
        # Compute background by applying perturbations to samples
        logger.info("Apply perturbations...")
        x_t += dx

    # Reshape
    x_t = x_t.reshape(x_dims)

    # Unnormalize the data prior to saving
    if hasattr(input.prof, 'normalization'):
        norm_func = instantiate(input.prof.normalization, inverse_transform=True)
        x_t = norm_func(x_t)

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving background to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_t)
        return None
    else:
        return x_t


def background_from_bounded_perturbations(input: DictConfig, stats: DictConfig,
                                          output: DictConfig | None = None,
                                          seed: int | None = None) -> np.ndarray | None:
    """ Compute the background from the model error covariance matrix and perturbations.

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
    cov_cholesky = load_var(input.cholesky)

    # Load physical base space and original dimensions
    x_base_phys = load_var(input.prof)
    x_dims = x_base_phys.shape

    # Load and flatten the standardized/normalized initial state
    xp_base_std = load_var_and_normalize(input.prof).reshape(x_dims[0], -1)

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
        logger.info(f"Saving validated background to {output.path}...")
        save_func = instantiate(output.save)
        save_func(final_x_phys)
        return None
    else:
        return final_x_phys

def climatological_matrix(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0, regularization_factor: float = 1.0,
                          plot_flag: bool=True, recenter: bool=False, univariate: bool = False, mean_type: str='spatiotemporal') -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        plot_flag: bool. If True, plot the covariance matrix.
        recenter: bool. If True, recenter by removing the mean.

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    # TODO: Does this make sense?
    data = load_var_and_normalize(input.data)
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
            # Compute anomalies (thruth - climatological mean)
            data -= mu
        # Compute temporal mean (but maintain coordinate dependency)
        elif mean_type == 'temporal':
            # Read coordinates
            if hasattr(input, 'lat') and hasattr(input, 'lon') and hasattr(input, 'scans'):
                # Read coordinates
                lat = load_var(input.lat)
                lon = load_var(input.lon)
                scans = load_var(input.scans)

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

                # 5. # Compute anomalies (thruth - climatological mean)
                data -= group_means[inverse_indices]  # Shape: (n_samples, n_features)
                # Broadcast the means back out to match the original sample layout
                data = data.reshape(data_shape)
                # Denominator (computation of the mean)
                denom = n_samples - n_unique_coords
            else:
                raise ValueError("Timesteps not found.")
        else:
            raise ValueError("mean_type not supported.")

    # Pressure filter (background only)
    if hasattr(input, 'pressure_filter') and input.pressure_filter is not None:
        # Load filter
        pressure_filter = instantiate(input.pressure_filter.load)
        if n_vars != pressure_filter.shape[0]:
            pressure_filter = np.take(pressure_filter, [0, 4, 8], axis=0)
    else:
        pressure_filter = None

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <=2:
            raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
        # Initialize empty matrix
        m = []
        # Loop over varialbes
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply pressure filter
            if pressure_filter is not None:
                logger.info(f"Applying pressure filter for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(pressure_filter[i]), axis=1)
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
        # Apply pressure filter
        if pressure_filter is not None:
            logger.info("Applying pressure filter...")
            # Remove constant pressure levels from the data
            data = np.take(data, np.flatnonzero(pressure_filter), axis=1)
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
    if hasattr(config.preparation, "covariance"):
        for dataset, config_covariance in config.preparation.covariance.items():
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