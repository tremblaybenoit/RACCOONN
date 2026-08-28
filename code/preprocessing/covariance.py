import numpy as np
from scipy.linalg import block_diag
import hydra
import torch
from omegaconf import DictConfig
from code.data.io import load_variable
from code.data.transformations import Compose, identity
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from code.evaluation.plot import plot_map, save_plot, flexible_gridspec
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def err(input: DictConfig, output: DictConfig | None = None, apply_transform: bool = False) -> np.ndarray | torch.Tensor | None:
    """ Compute the model error covariance matrix.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        apply_transform: bool. If True, apply normalization transform to the data before computing covariance.

        Returns
        -------
        None.
    """

    # Load data and reference (with transformations applied)
    x = load_variable(input.x, as_tensor=False, apply_transform=apply_transform)
    x_true = load_variable(input.x_true, as_tensor=False, apply_transform=apply_transform)
    # Compute model error (data - reference)
    x_err = x - x_true

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving validated prior to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_err)
        return None
    else:
        return x_err


def prior_from_correlation_perturbations(input: DictConfig, output: DictConfig | None = None,
                                         seed: int | None = None, apply_transform: bool = False) -> np.ndarray | None:
    """ Compute the prior from the model error correlation matrix and perturbations.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        seed: int. Seed to ensure reproducibility.
        apply_transform: bool. If True, apply normalization transform to the data before computing covariance.


        Returns
        -------
        None.
    """

    # Load Cholesky decomposition of correlation matrix
    correlation_cholesky = instantiate(input.correlation_cholesky.load)
    # Load covariance matrix standard deviations
    std = np.sqrt(np.diag(instantiate(input.matrix.load)))

    # Load true state (with transformations applied) and original dimensions
    x_true_transformed = load_variable(input.prof, apply_transform=apply_transform)
    if apply_transform and input.prof.get('transformations', None) is not None:
        # Create inverse transformation function (Compose returns identity if transformations is None)
        inverse_transform_fn = Compose(
            transformations=input.prof.get('transformations', None),
            inverse_transform=True
        )
    else:
        inverse_transform_fn = identity
    x_dims = x_true_transformed.shape
    # Initialize prior in physical space
    x_prior_physical = np.zeros_like(x_true_transformed)
    # Flatten truth
    x_true_transformed = x_true_transformed.reshape(x_dims[0], -1)
    # Extract physical bounds
    x_stats = instantiate(input.stats)
    x_min_physical, x_max_physical = x_stats['min'], x_stats['max']
    del x_stats

    # Initialize independent random number generator
    rng = np.random.default_rng(seed)

    # Track which profile row indices still need to be perturbed to fit within physical boundaries
    # At the beginning, assume that all do
    indices_to_perturb = np.arange(x_dims[0])

    # Apply the pressure filter if configured
    pressure_mask = None
    if input.get('pressure_mask', None) is not None:
        pressure_mask = instantiate(input.pressure_mask.load)
        if x_dims[1] != pressure_mask.shape[0]:
            pressure_mask = np.take(pressure_mask, [0, 4, 8], axis=0)

    # Safety configuration to prevent infinite loops in tough regimes
    max_iterations = 250
    iteration = 0
    logger.info(f"Starting perturbation sampling loop for {x_dims[0]} samples...")
    while len(indices_to_perturb) > 0 and iteration < max_iterations:
        # Number of points to perturb
        n_to_resample = len(indices_to_perturb)
        logger.info(f"Iteration {iteration}: Processing/Resampling {n_to_resample} profiles...")

        # Generate perturbations ONLY for the remaining invalid profiles
        p = rng.normal(0, 1, size=(correlation_cholesky.shape[1], n_to_resample))
        dx_transformed = (correlation_cholesky @ p).T * std

        # Compute (transformed) prior at the perturbed locations, accounting for the pressure filter
        x_prior_transformed = x_true_transformed[indices_to_perturb]
        if pressure_mask is not None:
            x_prior_transformed[:, np.flatnonzero(pressure_mask)] += dx_transformed
        else:
            x_prior_transformed += dx_transformed
        # Reshape prior to true dimensions
        x_prior_transformed = x_prior_transformed.reshape(n_to_resample, x_dims[1], x_dims[2])

        # Transform back to physical space
        x_prior_physical_samples = inverse_transform_fn(x_prior_transformed)

        # Check boundaries across ALL variables and levels for this sub-batch
        # Evaluates to a boolean array of shape: (n_to_resample, n_vars, n_levels)
        out_of_min = x_prior_physical_samples < x_min_physical
        out_of_max = x_prior_physical_samples > x_max_physical
        out_of_bounds = out_of_min | out_of_max
        logger.info(f"        {out_of_min.sum()} {out_of_max.sum()}...")

        # Collapse dimensions to find which specific profiles failed anywhere in their column
        profile_failed = np.any(out_of_bounds, axis=(1, 2))
        profile_passed = ~profile_failed

        # For the profiles that passed, lock them into the final output array
        if np.any(profile_passed):
            passed_global_indices = indices_to_perturb[profile_passed]
            x_prior_physical[passed_global_indices] = x_prior_physical_samples[profile_passed]

        # Filter down our invalid pointer list to only contain the persistent failures
        indices_to_perturb = indices_to_perturb[profile_failed]
        iteration += 1

    # Loop termination checks
    if len(indices_to_perturb) > 0:
        raise ValueError(
            f"Rejection sampling failed to converge for {len(indices_to_perturb)} "
            f"profiles within {max_iterations} iterations. The physical boundaries "
            f"might be too narrow for the specified covariance matrix variance."
        )

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving validated prior to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_prior_physical)
        return None
    else:
        return x_prior_physical


def prior_from_bounded_perturbations(input: DictConfig, output: DictConfig | None = None,
                                     seed: int | None = None, apply_transform: bool = False) -> np.ndarray | None:
    """ Compute the prior from the model error covariance matrix and perturbations.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        seed: int. Seed to ensure reproducibility.
        apply_transform: bool. If True, apply normalization transform to the data before computing covariance.


        Returns
        -------
        None.
    """
    # Load Cholesky matrix
    cov_cholesky = instantiate(input.cholesky.load)

    # Load true state (with transformations applied) and original dimensions
    x_true_transformed = load_variable(input.prof, apply_transform=apply_transform)
    if apply_transform and input.prof.get('transformations', None) is not None:
        # Create inverse transformation function (Compose returns identity if transformations is None)
        inverse_transform_fn = Compose(
            transformations=input.prof.get('transformations', None),
            inverse_transform=True
        )
    else:
        inverse_transform_fn = identity
    x_dims = x_true_transformed.shape
    # Initialize prior in physical space
    x_prior_physical = np.zeros_like(x_true_transformed)
    # Flatten truth
    x_true_transformed = x_true_transformed.reshape(x_dims[0], -1)
    # Extract physical bounds
    x_stats = instantiate(input.stats)
    x_min_physical, x_max_physical = x_stats['min'], x_stats['max']
    del x_stats

    # Initialize independent random number generator
    rng = np.random.default_rng(seed)

    # Track which profile row indices still need to be perturbed to fit within physical boundaries
    # At the beginning, assume that all do
    indices_to_perturb = np.arange(x_dims[0])

    # Apply the pressure filter if configured
    pressure_mask = None
    if input.get('pressure_mask', None) is not None:
        pressure_mask = instantiate(input.pressure_mask.load)
        if x_dims[1] != pressure_mask.shape[0]:
            pressure_mask = np.take(pressure_mask, [0, 4, 8], axis=0)

    # Safety configuration to prevent infinite loops in tough regimes
    max_iterations = 250
    iteration = 0
    logger.info(f"Starting perturbation sampling loop for {x_dims[0]} samples...")
    while len(indices_to_perturb) > 0 and iteration < max_iterations:
        # Number of points to perturb
        n_to_resample = len(indices_to_perturb)
        logger.info(f"Iteration {iteration}: Processing/Resampling {n_to_resample} profiles...")

        # Generate perturbations ONLY for the remaining invalid profiles
        p = rng.normal(0, 1, size=(cov_cholesky.shape[1], n_to_resample))
        dx_transformed = (cov_cholesky @ p).T

        # Compute (transformed) prior at the perturbed locations, accounting for the pressure filter
        x_prior_transformed = x_true_transformed[indices_to_perturb]
        if pressure_mask is not None:
            x_prior_transformed[:, np.flatnonzero(pressure_mask)] += dx_transformed
        else:
            x_prior_transformed += dx_transformed
        # Reshape prior to true dimensions
        x_prior_transformed = x_prior_transformed.reshape(n_to_resample, x_dims[1], x_dims[2])

        # Transform back to physical space
        x_prior_physical_samples = inverse_transform_fn(x_prior_transformed)

        # Check boundaries across ALL variables and levels for this sub-batch
        # Evaluates to a boolean array of shape: (n_to_resample, n_vars, n_levels)
        out_of_min = x_prior_physical_samples < x_min_physical
        out_of_max = x_prior_physical_samples > x_max_physical
        out_of_bounds = out_of_min | out_of_max
        logger.info(f"        {out_of_min.sum()} {out_of_max.sum()}...")

        # Collapse dimensions to find which specific profiles failed anywhere in their column
        profile_failed = np.any(out_of_bounds, axis=(1, 2))
        profile_passed = ~profile_failed

        # For the profiles that passed, lock them into the final output array
        if np.any(profile_passed):
            passed_global_indices = indices_to_perturb[profile_passed]
            x_prior_physical[passed_global_indices] = x_prior_physical_samples[profile_passed]

        # Filter down our invalid pointer list to only contain the persistent failures
        indices_to_perturb = indices_to_perturb[profile_failed]
        iteration += 1

    # Loop termination checks
    if len(indices_to_perturb) > 0:
        raise ValueError(
            f"Rejection sampling failed to converge for {len(indices_to_perturb)} "
            f"profiles within {max_iterations} iterations. The physical boundaries "
            f"might be too narrow for the specified covariance matrix variance."
        )

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving validated prior to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_prior_physical)
        return None
    else:
        return x_prior_physical


def climatological_matrix(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
                          regularization_factor: float = 1.0, plot_flag: bool=True, recenter: bool=False,
                          univariate: bool = False, mean_type: str='spatiotemporal', apply_transform: bool=False) -> None:
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
        apply_transform: bool. If True, apply normalization transform to the data before computing covariance.

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    data = load_variable(input.data, apply_transform=apply_transform)

    # Build sample mask
    mask = np.ones(data.shape[0], dtype=bool)
    # Spatial mask: Consider only data within the specified latitude and longitude bounds
    if hasattr(input, 'spatial_mask') and input.spatial_mask is not None:
        mask &= instantiate(input.spatial_mask)
    # Temporal mask: Consider only data within the specified time bounds
    if hasattr(input, 'temporal_mask') and input.temporal_mask is not None:
        mask &= instantiate(input.temporal_mask)
    # Apply mask to data
    data = data[mask]

    # Dimensions
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
            if hasattr(input, 'lat') and hasattr(input, 'lon'):
                # Read coordinates
                lat = load_variable(input.lat)
                lon = load_variable(input.lon)
                # Apply mask to coordinates
                lat = lat[mask]
                lon = lon[mask]

                # For clearsky-only or cloud-only datasets, the available coordinates points.
                # In other words, two consecutive timesteps may not have the same (lat, lon) pairs.
                # To compute the temporal mean at every available (lat, lon) point,
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

                # Allocate a destination array for the sums of each unique coordinate
                group_sums = np.zeros((n_unique_coords, n_features), dtype=data.dtype)

                # np.add.at performs unbuffered in-place addition for repeating indices
                np.add.at(group_sums, inverse_indices, data)

                # Count how many times each unique coordinate appears across all timesteps
                group_counts = np.bincount(inverse_indices)[:, None]  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

                # Compute anomalies (truth - climatological mean)
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

    # Variant filter (prior only)
    if hasattr(input, 'variant_mask') and input.variant_mask is not None:
        # Load filter
        variant_mask = instantiate(input.variant_mask.load)
        if n_vars != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0, 4, 8], axis=0)
    else:
        variant_mask = None

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
            # Apply variant mask
            if variant_mask is not None:
                logger.info(f"Applying variant mask for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(variant_mask[i]), axis=1)
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
        # Apply variant mask
        if variant_mask is not None:
            logger.info("Applying pressure mask...")
            # Remove constant pressure levels from the data
            data = np.take(data, np.flatnonzero(variant_mask), axis=1)
        # Compute matrix
        logger.info("Computing covariance matrix...")
        cov = {'matrix': scaling_factor*(data.T @ data)/denom}
    # Release memory
    del data

    # Apply regularization
    std = np.sqrt(np.diag(cov['matrix']))
    if regularization_factor > 0:
        logger.info("Applying regularization factor...")
        cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(cov['matrix'].shape[0])
    # Compute correlation matrix
    if hasattr(output, 'correlation'):
        cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
    # Compute Cholesky decomposition
    if hasattr(output, 'matrix_cholesky'):
        logger.info("Computing Cholesky decomposition...")
        cov['matrix_cholesky'] = np.linalg.cholesky(cov['matrix'])
    # Compute matrix inverse
    if hasattr(output, 'matrix_inverse'):
        logger.info("Computing the inverse of the covariance matrix...")
        cov['matrix_inverse'] = np.linalg.inv(cov['matrix'])
    # Compute matrix pseudo-inverse
    if hasattr(output, 'matrix_pseudo_inverse'):
        logger.info("Computing the pseudo-inverse of the covariance matrix...")
        rcond = output.matrix_pseudo_inverse.get('params.rcond', 1.e-3)
        cov['matrix_pseudo_inverse'] = np.linalg.pinv(cov['matrix'], rcond=rcond)
    # Compute Cholesky decomposition
    if hasattr(output, 'correlation_cholesky'):
        logger.info("Computing Cholesky decomposition...")
        cov['correlation_cholesky'] = np.linalg.cholesky(cov['correlation'])
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


def climatological_matrix2(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
                          regularization_factor: float = 1.0, plot_flag: bool = True, recenter: bool = False,
                          univariate: bool = False, mean_type: str = 'spatiotemporal',
                          apply_transform: bool = False) -> None:
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
        apply_transform: bool. If True, apply normalization transform to the data before computing covariance.

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    data = load_variable(input.data, apply_transform=apply_transform)

    # Build sample mask
    mask = np.ones(data.shape[0], dtype=bool)
    # Spatial mask: Consider only data within the specified latitude and longitude bounds
    if hasattr(input, 'spatial_mask') and input.spatial_mask is not None:
        mask &= instantiate(input.spatial_mask)
    # Temporal mask: Consider only data within the specified time bounds
    if hasattr(input, 'temporal_mask') and input.temporal_mask is not None:
        mask &= instantiate(input.temporal_mask)
    # Apply mask to data
    data = data[mask]

    # Dimensions
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
            if hasattr(input, 'lat') and hasattr(input, 'lon'):
                # Read coordinates
                lat = load_variable(input.lat)
                lon = load_variable(input.lon)
                # Apply mask to coordinates
                lat = lat[mask]
                lon = lon[mask]

                # For clearsky-only or cloud-only datasets, the available coordinates points.
                # In other words, two consecutive timesteps may not have the same (lat, lon) pairs.
                # To compute the temporal mean at every available (lat, lon) point,
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

                # Allocate a destination array for the sums of each unique coordinate
                group_sums = np.zeros((n_unique_coords, n_features), dtype=data.dtype)

                # np.add.at performs unbuffered in-place addition for repeating indices
                np.add.at(group_sums, inverse_indices, data)

                # Count how many times each unique coordinate appears across all timesteps
                group_counts = np.bincount(inverse_indices)[:, None]  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

                # Compute anomalies (truth - climatological mean)
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

    # Variant filter (prior only)
    if hasattr(input, 'variant_mask') and input.variant_mask is not None:
        # Load filter
        variant_mask = instantiate(input.variant_mask.load)
        if n_vars != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0, 4, 8], axis=0)
    else:
        variant_mask = None

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <= 2:
            raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
        # Initialize empty matrix
        m = []
        # Loop over variables
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply variant mask
            if variant_mask is not None:
                logger.info(f"Applying variant mask for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(variant_mask[i]), axis=1)

            # Compute stable correlation / covariance safely block-by-block if needed,
            # or retain standard physical scaling here. If univariate sub-blocks
            # also suffer from scale issues, compute via correlation and scale back:
            std_i = np.std(data_i, axis=0)
            # Avoid division by zero for constant features
            std_i = np.where(std_i == 0, 1.0, std_i)
            corr_i = (data_i.T @ data_i) / denom / (std_i[:, None] @ std_i[None, :])
            cov_i = scaling_factor * corr_i * (std_i[:, None] @ std_i[None, :])

            # Store diagonal block
            logger.info(f"Computing univariate covariance matrix for variable {i}...")
            m.append(cov_i)
        # Assemble
        del data_i
        cov = {'matrix': block_diag(*m)}
    # Multivariate matrix computation steps
    else:
        # Flatten data
        data = data.reshape(n_samples, -1)
        # Apply variant mask
        if variant_mask is not None:
            logger.info("Applying pressure mask...")
            # Remove constant pressure levels from the data
            data = np.take(data, np.flatnonzero(variant_mask), axis=1)

        # Compute matrix via intermediate correlation to prevent scale ill-conditioning
        logger.info("Computing covariance matrix via stable correlation scaling...")
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1.0, std)  # safeguard against zero-variance columns
        correlation_matrix = (data.T @ data) / denom / (std[:, None] @ std[None, :])

        cov = {'matrix': scaling_factor * correlation_matrix * (std[:, None] @ std[None, :])}

    # Release memory
    del data

    # Apply regularization
    if regularization_factor > 0:
        logger.info("Applying regularization factor...")
        cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(cov['matrix'].shape[0])

    # Compute correlation matrix
    if hasattr(output, 'correlation'):
        # If already computed multivariately, we can extract or re-derive standard deviations safely
        std_full = np.sqrt(np.diag(cov['matrix']))
        std_full = np.where(std_full == 0, 1.0, std_full)
        cov['correlation'] = cov['matrix'] / (std_full[:, None] @ std_full[None, :])

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
        # If single operation, execute
        if hasattr(config.preprocessing.covariance, "_target_"):
            logger.info(f"Computing covariance...")
            instantiate(config.preprocessing.covariance)
        # Execute individual operations
        else:
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