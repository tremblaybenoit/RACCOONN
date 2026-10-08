import numpy as np
from scipy.linalg import block_diag
from scipy.stats import truncnorm
from scipy.optimize import minimize_scalar
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
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

    # Load Cholesky matrix (L)
    cov_cholesky = instantiate(input.cholesky.load)

    # Load true state and dimensions
    x_true_transformed = load_variable(input.prof, apply_transform=apply_transform)
    inverse_transform_fn = (
        Compose(transformations=input.prof.get('transformations', None), inverse_transform=True)
        if apply_transform and input.prof.get('transformations') is not None
        else identity
    )

    x_dims = x_true_transformed.shape
    x_true_flat = x_true_transformed.reshape(x_dims[0], -1)

    # Load physical bounds
    x_stats = instantiate(input.stats)
    x_min_physical, x_max_physical = x_stats['min'], x_stats['max']
    del x_stats

    rng = np.random.default_rng(seed)

    # Optional variant mask setup
    variant_mask = None
    if input.get('variant_mask', None) is not None:
        variant_mask = instantiate(input.variant_mask.load)
        if x_dims[1] != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0], axis=0)

    # Single-pass perturbation generation (avoiding while-loop rejection bottlenecks)
    n_samples = x_dims[0]
    p = rng.normal(0, 1, size=(cov_cholesky.shape[1], n_samples))
    # p = truncnorm.rvs(-1, 1, size=(cov_cholesky.shape[1], n_samples), random_state=rng)
    dx_transformed = (cov_cholesky @ p).T

    x_prior_transformed = x_true_flat.copy()
    if variant_mask is not None:
        x_prior_transformed[:, np.flatnonzero(variant_mask)] += dx_transformed
    else:
        x_prior_transformed += dx_transformed

    # Reshape and map back to physical space
    x_prior_transformed = x_prior_transformed.reshape(n_samples, x_dims[1], x_dims[2])
    x_prior_physical = inverse_transform_fn(x_prior_transformed)

    # Apply safe boundary enforcement (clipping) instead of rejection sampling loops
    # to maintain unbiased bulk statistics and prevent infinite hanging.
    # Count samples with any out-of-bounds values using vectorized operations
    nval = ((x_prior_physical < x_min_physical) | (x_prior_physical > x_max_physical)).any(axis=(1, 2)).sum()
    logger.info(f"Number of samples outside physical bounds: {nval} out of {n_samples}")
    # x_prior_physical = np.clip(x_prior_physical, x_min_physical, x_max_physical)

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving validated prior to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_prior_physical)
        return None
    else:
        return x_prior_physical


def perturbations_from_covariance(input: DictConfig, output: DictConfig | None = None,
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

    # Load Cholesky matrix (L)
    cov_cholesky = instantiate(input.cholesky.load)

    # Load true state and dimensions
    x_true_transformed = load_variable(input.data, apply_transform=apply_transform)
    inverse_transform_fn = (
        Compose(transformations=input.data.get('transformations', None), inverse_transform=True)
        if apply_transform and input.data.get('transformations') is not None
        else identity
    )

    x_dims = x_true_transformed.shape
    x_true_flat = x_true_transformed.reshape(x_dims[0], -1)

    # Load physical bounds
    x_stats = instantiate(input.stats)
    x_min_physical, x_max_physical = x_stats['min'], x_stats['max']
    del x_stats

    rng = np.random.default_rng(seed)

    # Optional variant mask setup
    variant_mask = None
    if input.get('variant_mask', None) is not None:
        variant_mask = instantiate(input.variant_mask.load)
        if x_dims[1] != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0], axis=0)

    # Single-pass perturbation generation (avoiding while-loop rejection bottlenecks)
    n_samples = x_dims[0]
    p = rng.normal(0, 1, size=(cov_cholesky.shape[1], n_samples))
    # p = truncnorm.rvs(-1, 1, size=(cov_cholesky.shape[1], n_samples), random_state=rng)
    dx_transformed = (cov_cholesky @ p).T

    x_prior_transformed = x_true_flat.copy()
    if variant_mask is not None:
        x_prior_transformed[:, np.flatnonzero(variant_mask)] += dx_transformed
    else:
        x_prior_transformed += dx_transformed

    # Reshape and map back to physical space
    x_prior_transformed = x_prior_transformed.reshape(n_samples, x_dims[1], x_dims[2])
    x_prior_physical = inverse_transform_fn(x_prior_transformed)

    # Apply safe boundary enforcement (clipping) instead of rejection sampling loops
    # to maintain unbiased bulk statistics and prevent infinite hanging.
    # Count samples with any out-of-bounds values using vectorized operations
    nval = ((x_prior_physical < x_min_physical) | (x_prior_physical > x_max_physical)).any(axis=(1, 2)).sum()
    logger.info(f"Number of samples outside physical bounds: {nval} out of {n_samples}")
    # x_prior_physical = np.clip(x_prior_physical, x_min_physical, x_max_physical)

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving validated prior to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_prior_physical)
        return None
    else:
        return x_prior_physical


def prior_from_covariance(input: DictConfig, output: DictConfig | None = None,
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

    # Load Cholesky matrix (L)
    cov_cholesky = load_variable(input.cholesky)
    # Load physical bounds
    prof_stats = instantiate(input.stats)
    x_min_physical, x_max_physical = prof_stats['min'], prof_stats['max']
    del prof_stats

    # Loop over stages
    for stage, config_stage in input.stage.keys():
        logger.info(f"Loading stage '{stage}'")

        # Load atmospheric profiles
        if hasattr(config_stage, 'variables') and config_stage.variables is not None:

            if hasattr(config_stage.variables, 'prof') and config_stage.variables.prof is not None:
                logger.info(f"Loading profile variable 'prof'")
                x_true_transformed = load_variable(config_stage.variables.prof.load, apply_transform=apply_transform)
                x_inverse_transform_fn = (
                    Compose(transformations=input.prof.get('transformations', None), inverse_transform=True)
                    if apply_transform and input.prof.get('transformations') is not None
                    else identity
                )
                x_dims = x_true_transformed.shape
                x_true_flat = x_true_transformed.reshape(x_dims[0], -1)
                x_prior_transformed = x_true_flat.copy()

                # Optional variant mask setup
                variant_mask = None
                if config_stage.variables.get('variant_mask', None) is not None:
                    variant_mask = instantiate(config_stage.variables.variant_mask.load)
                    if x_dims[1] != variant_mask.shape[0]:
                        variant_mask = np.take(variant_mask, [0], axis=0)

                # Single-pass perturbation generation (avoiding while-loop rejection bottlenecks)
                n_samples = x_dims[0]
                rng = np.random.default_rng(seed)
                p = rng.normal(0, 1, size=(cov_cholesky.shape[1], n_samples))
                dx_transformed = (cov_cholesky @ p).T
                if variant_mask is not None:
                    x_prior_transformed[:, np.flatnonzero(variant_mask)] += dx_transformed
                else:
                    x_prior_transformed += dx_transformed

                # Reshape and map back to physical space
                x_prior_transformed = x_prior_transformed.reshape(n_samples, x_dims[1], x_dims[2])
                x_prior_physical = x_inverse_transform_fn(x_prior_transformed)

                # Apply safe boundary enforcement (clipping) instead of rejection sampling loops
                # to maintain unbiased bulk statistics and prevent infinite hanging.
                # Count samples with any out-of-bounds values using vectorized operations
                nval = ((x_prior_physical < x_min_physical) | (x_prior_physical > x_max_physical)).any(axis=(1, 2)).sum()
                logger.info(f"Number of samples outside physical bounds: {nval} out of {n_samples}")
                x_prior_physical = np.clip(x_prior_physical, x_min_physical, x_max_physical)

                # Save to file
                if output is not None and hasattr(output, stage):
                    if hasattr(output[stage], 'variables') and hasattr(output[stage].variables, 'prof_prior'):
                        if hasattr(output[stage].variables.prof_prior, 'save'):
                            logger.info(f"Saving validated prior to {output[stage].variables.prof_prior.save.path}...")
                            save_func = instantiate(output[stage].variables.prof_prior.save)
                            save_func(x_prior_physical)
                            return None
                        else:
                            return x_prior_physical
                    else:
                        logger.warning(f"No 'prof_prior' variable found in output for stage '{stage}'")
            else:
                logger.warning(f"No 'prof' variable found in stage '{stage}'")
        else:
            logger.warning(f"No 'variables' found in stage '{stage}'")


def prior_from_mean(input: DictConfig, output: DictConfig | None = None, 
                    seed: int | None = None, apply_transform: bool = False) -> np.ndarray | None:
    """ Compute prior from climatological mean at specified coordinates.
    
    Loads climatological mean matrix (computed by climatological_matrix with mean_type='temporal'),
    then maps input (lat, lon) coordinates to their corresponding mean values.
    
    For each input sample, finds the matching (lat, lon) in the stored unique coordinates
    and returns the associated climatological mean profile.
    
    Parameters
    ----------
    input: DictConfig with keys:
        - mean_data: Load config for climatological mean matrix (shape: [n_unique_coords, n_vars, n_levels])
        - unique_coords: Load config for unique coordinate array (shape: [n_unique_coords, 2])
        - lat: Latitude array/file for input samples (shape: [n_samples])
        - lon: Longitude array/file for input samples (shape: [n_samples])
        - prof: Configuration for profile structure/transformations
        - stats: Physical bounds (min/max)
        - spatial_mask: Optional spatial mask for input coordinates
        - temporal_mask: Optional temporal mask for input coordinates
        - variant_mask: Optional pressure level mask
    output: DictConfig for saving
    seed: Optional random seed for reproducibility
    apply_transform: bool. If True, apply normalization transform to the data before computing covariance.
    
    Returns
    -------
    np.ndarray or None: Prior with shape [n_samples, n_vars, n_levels]
    """
    
    # Load climatological mean and unique coordinates
    logger.info("Loading climatological mean and unique coordinates...")
    mu_all = instantiate(input.mean_data.load)
    unique_coords = instantiate(input.unique_coords.load)
    
    # Load input coordinates
    lat = load_variable(input.lat)
    lon = load_variable(input.lon)
    
    # Build sample mask (spatial + temporal)
    mask = np.ones(lat.shape[0], dtype=bool)
    
    # Spatial mask
    if hasattr(input, 'spatial_mask') and input.spatial_mask is not None:
        logger.info("Applying spatial mask...")
        mask &= instantiate(input.spatial_mask)
    
    # Temporal mask
    if hasattr(input, 'temporal_mask') and input.temporal_mask is not None:
        logger.info("Applying temporal mask...")
        mask &= instantiate(input.temporal_mask)
    
    # Apply mask to coordinates
    lat = lat[mask]
    lon = lon[mask]
    n_samples = len(lat)
    
    # Create input coordinates array
    input_coords = np.column_stack((lat, lon))
    
    # Map input coordinates to unique_coords using structured lookup
    # unique_coords shape: [n_unique_coords, 2]
    # input_coords shape: [n_samples, 2]
    logger.info("Mapping input coordinates to unique coordinate set...")
    
    # Create a dictionary for O(1) lookup: (lat, lon) -> index in mu_all
    coord_to_idx = {tuple(coord): i for i, coord in enumerate(unique_coords)}
    
    # Build prior by looking up each input coordinate
    n_unique, n_vars, n_levels = mu_all.shape
    x_prior = np.zeros((n_samples, n_vars, n_levels), dtype=mu_all.dtype)
    
    n_found = 0
    for i, coord in enumerate(input_coords):
        coord_tuple = tuple(coord)
        if coord_tuple in coord_to_idx:
            idx = coord_to_idx[coord_tuple]
            x_prior[i] = mu_all[idx]
            n_found += 1
        else:
            logger.warning(f"Sample {i} with coordinate {coord} not found in unique coordinate set")
    
    logger.info(f"Found {n_found}/{n_samples} samples in unique coordinate set")
    
    # Apply variant mask if specified
    if hasattr(input, 'variant_mask') and input.variant_mask is not None:
        logger.info("Applying variant mask...")
        variant_mask = instantiate(input.variant_mask.load)
        if n_vars != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0], axis=0)
        
        # Zero out masked levels for each variable
        for i in range(n_vars):
            masked_levels = np.where(~variant_mask[i])[0]
            x_prior[:, i, masked_levels] = 0

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving prior from mean to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_prior)
        return None
    else:
        return x_prior


def climatological_matrix(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
                          regularization_factor: float = 1.0, regularization_type: str = 'per_variable',
                          plot_flag: bool=True, recenter: bool=False,
                          univariate: bool = False, mean_type: str='spatiotemporal', apply_transform: bool=False) -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        scaling_factor: float. Scaling factor for covariance matrix.
        regularization_factor: float. Regularization factor for covariance matrix.
        regularization_type: str. Type of regularization to apply.
            Options:
            - 'per_variable': Apply regularization per-variable block (recommended, current default)
            - 'global': Apply global regularization to entire matrix using global mean diagonal
            - 'correlation_based': Regularize via correlation matrix, then reconstruct covariance
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
        logger.info("Applying spatial mask...")
        mask &= instantiate(input.spatial_mask)
    # Temporal mask: Consider only data within the specified time bounds
    if hasattr(input, 'temporal_mask') and input.temporal_mask is not None:
        logger.info("Applying temporal mask...")
        mask &= instantiate(input.temporal_mask)
    # Apply mask to data
    data = data[mask]

    # Dimensions
    data_shape = data.shape
    n_samples, n_vars = data_shape[0], data_shape[1]
    # Denominator (computation of the mean)
    denom = float(n_samples - 1)

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
                group_counts = np.bincount(inverse_indices)[:, None].astype(data.dtype)  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

                # Save mean and unique coordinates for prior_from_mean function
                if hasattr(output, 'mu') and hasattr(output.mu, 'save'):
                    logger.info("Saving climatological mean to file...")
                    # Reshape means back to original dimensions (excluding batch axis)
                    inverse_transform_fn = (
                        Compose(transformations=input.data.get('transformations', None), inverse_transform=True)
                        if apply_transform and input.data.get('transformations') is not None
                        else identity
                    )
                    mu_to_save = inverse_transform_fn(group_means.reshape(n_unique_coords, *data_shape[1:]))
                    save_func_mean = instantiate(output.mu.save)
                    save_func_mean(mu_to_save)
                
                if hasattr(output, 'unique_coords') and hasattr(output.unique_coords, 'save'):
                    logger.info("Saving unique coordinates to file...")
                    save_func_coords = instantiate(output.unique_coords.save)
                    save_func_coords(unique_coords)

                # Compute anomalies (truth - climatological mean)
                data -= group_means[inverse_indices]  # Shape: (n_samples, n_features)
                # Broadcast the means back out to match the original sample layout
                data = data.reshape(data_shape)
                # Denominator (computation of the mean)
                denom = float(n_samples - n_unique_coords)
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
            variant_mask = np.take(variant_mask, [0], axis=0)
    else:
        variant_mask = None

    # Covariance matrix initialization
    cov = {}

    # Validate regularization_type
    valid_reg_types = ['per_variable', 'global', 'correlation_based']
    if regularization_type not in valid_reg_types:
        raise ValueError(f"regularization_type must be one of {valid_reg_types}, got '{regularization_type}'")

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <=2:
            # raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
           data = data[..., None]  # Add a singleton dimension for levels if not present
        # Initialize empty lists for matrices
        m_cov = []
        m_corr = []
        # Loop over variables
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply variant mask
            if variant_mask is not None:
                logger.info(f"Applying variant mask for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(variant_mask[i]), axis=1)
            # Compute covariance block
            logger.info(f"Computing univariate covariance matrix for variable {i}...")
            sub_cov = scaling_factor*(data_i.T @ data_i)/denom

            # Compute correlation block from unregularized covariance
            if hasattr(output, 'correlation'):
                std_i = np.sqrt(np.diag(sub_cov))
                sub_corr = sub_cov / (std_i[:, None] @ std_i[None, :])
                m_corr.append(sub_corr)

            # Apply regularization based on regularization_type
            if regularization_factor > 0:
                if regularization_type == 'per_variable':
                    # Per-variable regularization: scale by each block's mean diagonal
                    var_mean_diag = float(np.mean(np.diag(sub_cov)))
                    logger.info(f"Applying per-variable regularization for variable {i} (mean_diag={var_mean_diag:.6e})...")
                    sub_cov = sub_cov + regularization_factor * var_mean_diag * np.eye(sub_cov.shape[0], dtype=sub_cov.dtype)
                elif regularization_type == 'global':
                    # Global regularization: will be applied after block assembly
                    pass
                elif regularization_type == 'correlation_based':
                    # Correlation-based: will be applied after correlation computation
                    pass

            m_cov.append(sub_cov)

        # Assemble into block-diagonal matrices
        del data_i, data
        cov['matrix'] = block_diag(*m_cov)
        if hasattr(output, 'correlation'):
            cov['correlation'] = block_diag(*m_corr)

        # Apply global regularization to assembled matrix if requested
        if regularization_factor > 0 and regularization_type == 'global':
            logger.info("Applying global regularization to univariate covariance matrix...")
            cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(cov['matrix'].shape[0], dtype=cov['matrix'].dtype)

        # Apply correlation-based regularization if requested
        if regularization_factor > 0 and regularization_type == 'correlation_based':
            logger.info("Applying correlation-based regularization to univariate covariance matrix...")
            # Recompute from correlation to ensure consistency
            cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)  # Ensure symmetry
            std = np.sqrt(np.diag(cov['matrix']))
            cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
            cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry
            # Regularize correlation
            cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
            # Reconstruct covariance from regularized correlation
            cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']

    # Multivariate matrix computation steps
    else:
        # Keep track of variable boundaries for per-variable regularization
        # Reshape to (n_samples, n_vars, -1) to preserve variable dimension
        data_reshaped = data.reshape(n_samples, n_vars, -1)  # (n_samples, n_vars, total_features_per_var)

        # Apply variant mask BEFORE flattening to track per-variable feature counts
        if variant_mask is not None:
            logger.info("Applying pressure mask...")
            # variant_mask shape: (n_vars, n_features_per_var) or similar
            # Need to apply per-variable masking
            data_masked_list = []
            features_per_var_list = []  # Track features per variable AFTER masking

            for i in range(n_vars):
                data_i = data_reshaped[:, i, :]  # (n_samples, n_features)

                # Apply variable-specific mask if available
                if isinstance(variant_mask, np.ndarray):
                    if variant_mask.ndim == 2:
                        # variant_mask shape: (n_vars, n_features_per_var)
                        mask_i = np.flatnonzero(variant_mask[i])
                    elif variant_mask.ndim == 1:
                        # variant_mask shape: (n_features_total,) - single mask for all
                        mask_i = np.flatnonzero(variant_mask)
                    else:
                        raise ValueError(f"Unexpected variant_mask shape: {variant_mask.shape}")

                    data_i_masked = np.take(data_i, mask_i, axis=1)
                else:
                    data_i_masked = data_i

                data_masked_list.append(data_i_masked)
                features_per_var_list.append(data_i_masked.shape[1])
                logger.info(f"Variable {i} features after masking: {data_i_masked.shape[1]}")

            # Flatten concatenated data
            data = np.concatenate(data_masked_list, axis=1)  # (n_samples, total_features)

            # Free up memory
            del data_i, data_i_masked, data_masked_list
        else:
            # No masking: all variables have same number of features
            data = data_reshaped.reshape(n_samples, -1)
            features_per_var_list = [data_reshaped.shape[2]] * n_vars

        # Compute full covariance matrix (before regularization)
        logger.info("Computing covariance matrix...")
        cov['matrix'] = scaling_factor*(data.T @ data)/denom

        # Free up memory
        del data, data_reshaped

        # Compute correlation matrix from ORIGINAL (unregularized) covariance
        if hasattr(output, 'correlation'):
            logger.info("Computing correlation matrix from covariance matrix...")
            std = np.sqrt(np.diag(cov['matrix']))
            cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])

        # Apply regularization based on regularization_type
        if regularization_factor > 0:
            if regularization_type == 'per_variable':
                logger.info("Applying per-variable regularization to covariance matrix...")

                # Loop over variables with their specific feature counts
                start_idx = 0
                for i, features_per_var in enumerate(features_per_var_list):
                    end_idx = start_idx + features_per_var

                    # Extract diagonal for this variable's block
                    var_diag = np.diag(cov['matrix'])[start_idx:end_idx]
                    var_mean_diag = float(np.mean(var_diag))

                    logger.info(f"Applying regularization for variable {i} (features={features_per_var}, mean_diag={var_mean_diag:.6e})...")

                    # Add regularization to the diagonal block of this variable
                    cov['matrix'][start_idx:end_idx, start_idx:end_idx] += regularization_factor * var_mean_diag * np.eye(features_per_var, dtype=cov['matrix'].dtype)
                    start_idx = end_idx

            elif regularization_type == 'global':
                logger.info("Applying global regularization to covariance matrix...")
                cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(cov['matrix'].shape[0], dtype=cov['matrix'].dtype)

            elif regularization_type == 'correlation_based':
                logger.info("Applying correlation-based regularization to covariance matrix...")
                # Ensure symmetry before reconstruction
                cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)
                std = np.sqrt(np.diag(cov['matrix']))
                cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
                cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry
                # Regularize correlation
                cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
                # Reconstruct covariance from regularized correlation
                cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']

    # Compute diagonal
    if hasattr(output, 'diagonal'):
        logger.info("Computing diagonal of the covariance matrix...")
        cov['diagonal'] = np.diag(np.diag(cov['matrix']))
    if hasattr(output, 'diagonal_inverse'):
        logger.info("Computing diagonal inverse of the covariance matrix...")
        cov['diagonal_inverse'] = np.diag(1.0 / np.diag(cov['matrix']))
    # Compute Cholesky decomposition
    if hasattr(output, 'matrix_cholesky'):
        logger.info("Computing Cholesky decomposition of the covariance matrix...")
        cov['matrix_cholesky'] = np.linalg.cholesky(cov['matrix'])
    # Compute matrix inverse
    if hasattr(output, 'matrix_inverse'):
        logger.info("Computing the inverse of the covariance matrix...")
        cov['matrix_inverse'] = np.linalg.inv(cov['matrix'])
    # Compute matrix pseudo-inverse
    if hasattr(output, 'matrix_pseudo_inverse'):
        logger.info("Computing the pseudo-inverse of the covariance matrix...")
        rcond = OmegaConf.select(output.matrix_pseudo_inverse, 'params.rcond', default=1.e-3)
        cov['matrix_pseudo_inverse'] = np.linalg.pinv(cov['matrix'], rcond=rcond)
    # Compute Cholesky decomposition
    if hasattr(output, 'correlation_cholesky'):
        logger.info("Computing Cholesky decomposition of the correlation matrix...")
        cov['correlation_cholesky'] = np.linalg.cholesky(cov['correlation'])
    # Compute inverse of the correlation matrix
    if hasattr(output, 'correlation_inverse') and 'correlation' in cov:
        logger.info("Computing the inverse of the correlation matrix...")
        cov['correlation_inverse'] = np.linalg.inv(cov['correlation'])
    # Compute pseudo inverse of the correlation matrix
    if hasattr(output, 'correlation_pseudo_inverse') and 'correlation' in cov:
        logger.info("Computing the pseudo-inverse of the correlation matrix...")
        rcond = OmegaConf.select(output.correlation_pseudo_inverse, 'params.rcond', default=1.e-3)
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
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse', 'diagonal', 'diagonal_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


def pressure_only_matrix(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
                         regularization_factor: float = 1.0, regularization_type: str = 'per_variable',
                         plot_flag: bool = True, recenter: bool = False,
                         univariate: bool = False, mean_type: str = 'spatiotemporal',
                         apply_transform: bool = False) -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        scaling_factor: float. Scaling factor for covariance matrix.
        regularization_factor: float. Regularization factor for covariance matrix.
        regularization_type: str. Type of regularization to apply.
            Options:
            - 'per_variable': Apply regularization per-variable block (recommended, current default)
            - 'global': Apply global regularization to entire matrix using global mean diagonal
            - 'correlation_based': Regularize via correlation matrix, then reconstruct covariance
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
        logger.info("Applying spatial mask...")
        mask &= instantiate(input.spatial_mask)
    # Temporal mask: Consider only data within the specified time bounds
    if hasattr(input, 'temporal_mask') and input.temporal_mask is not None:
        logger.info("Applying temporal mask...")
        mask &= instantiate(input.temporal_mask)
    # Apply mask to data
    data = data[mask]

    # Dimensions
    data_shape = data.shape
    n_samples, n_vars = data_shape[0], data_shape[1]
    # Denominator (computation of the mean)
    denom = float(n_samples - 1)

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
                group_counts = np.bincount(inverse_indices)[:, None].astype(data.dtype)  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

                # Save mean and unique coordinates for prior_from_mean function
                if hasattr(output, 'mu') and hasattr(output.mu, 'save'):
                    logger.info("Saving climatological mean to file...")
                    # Reshape means back to original dimensions (excluding batch axis)
                    inverse_transform_fn = (
                        Compose(transformations=input.data.get('transformations', None), inverse_transform=True)
                        if apply_transform and input.data.get('transformations') is not None
                        else identity
                    )
                    mu_to_save = inverse_transform_fn(group_means.reshape(n_unique_coords, *data_shape[1:]))
                    save_func_mean = instantiate(output.mu.save)
                    save_func_mean(mu_to_save)

                if hasattr(output, 'unique_coords') and hasattr(output.unique_coords, 'save'):
                    logger.info("Saving unique coordinates to file...")
                    save_func_coords = instantiate(output.unique_coords.save)
                    save_func_coords(unique_coords)

                # Compute anomalies (truth - climatological mean)
                data -= group_means[inverse_indices]  # Shape: (n_samples, n_features)
                # Broadcast the means back out to match the original sample layout
                data = data.reshape(data_shape)
                # Denominator (computation of the mean)
                denom = float(n_samples - n_unique_coords)
            else:
                mu = np.mean(data)
                # Compute anomalies (truth - mean)
                data -= mu
        else:
            raise ValueError("mean_type not supported.")

    # Guard against invalid degrees of freedom (e.g. n_samples <= n_groups)
    if denom <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom for covariance computation: denominator is {denom:.1f}. "
            f"Ensure total samples ({n_samples}) exceed group count."
        )

    # Variant filter (prior only)
    if hasattr(input, 'variant_mask') and input.variant_mask is not None:
        # Load filter
        variant_mask = instantiate(input.variant_mask.load)
        if n_vars != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0], axis=0)
    else:
        variant_mask = None

    # Covariance matrix initialization
    cov = {}

    # Validate regularization_type
    valid_reg_types = ['per_variable', 'global', 'correlation_based']
    if regularization_type not in valid_reg_types:
        raise ValueError(f"regularization_type must be one of {valid_reg_types}, got '{regularization_type}'")

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <= 2:
            # raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
            data = data[..., None]  # Add a singleton dimension for levels if not present
        # Initialize empty lists for matrices
        m_cov = []
        m_corr = []
        # Loop over variables
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply variant mask
            if variant_mask is not None:
                logger.info(f"Applying variant mask for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(variant_mask[i]), axis=1)
            # Compute covariance block
            logger.info(f"Computing univariate covariance matrix for variable {i}...")
            sub_cov = scaling_factor * (data_i.T @ data_i) / denom

            # Apply per-variable regularization directly to sub_cov BEFORE computing correlation
            if regularization_factor > 0 and regularization_type == 'per_variable':
                var_mean_diag = float(np.mean(np.diag(sub_cov)))
                logger.info(f"Applying per-variable regularization for variable {i} (mean_diag={var_mean_diag:.6e})...")
                sub_cov = sub_cov + regularization_factor * var_mean_diag * np.eye(sub_cov.shape[0],
                                                                                   dtype=sub_cov.dtype)

            # Compute correlation block from regularized sub_cov (with zero-variance safety)
            if hasattr(output, 'correlation'):
                std_i = np.sqrt(np.maximum(np.diag(sub_cov), 1e-12))
                sub_corr = sub_cov / (std_i[:, None] @ std_i[None, :])
                m_corr.append(sub_corr)

            m_cov.append(sub_cov)

        # Assemble into block-diagonal matrices
        del data_i, data
        cov['matrix'] = block_diag(*m_cov)
        if hasattr(output, 'correlation'):
            cov['correlation'] = block_diag(*m_corr)

        # Apply global regularization to assembled matrix if requested
        if regularization_factor > 0 and regularization_type == 'global':
            logger.info("Applying global regularization to univariate covariance matrix...")
            cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(
                cov['matrix'].shape[0], dtype=cov['matrix'].dtype)
            # Recompute block correlation matrix after global regularization
            if hasattr(output, 'correlation'):
                std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
                cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])

        # Apply correlation-based regularization if requested
        elif regularization_factor > 0 and regularization_type == 'correlation_based':
            logger.info("Applying correlation-based regularization to univariate covariance matrix...")
            # Recompute from correlation to ensure consistency
            cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)  # Ensure symmetry
            std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
            cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
            cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry
            # Regularize correlation
            cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(
                cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
            # Reconstruct covariance from regularized correlation
            cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']

    # Multivariate matrix computation steps
    else:
        # Keep track of variable boundaries for per-variable regularization
        # Reshape to (n_samples, n_vars, -1) to preserve variable dimension
        data_reshaped = data.reshape(n_samples, n_vars, -1)  # (n_samples, n_vars, total_features_per_var)

        # Apply variant mask BEFORE flattening to track per-variable feature counts
        if variant_mask is not None:
            logger.info("Applying pressure mask...")
            # variant_mask shape: (n_vars, n_features_per_var) or similar
            # Need to apply per-variable masking
            data_masked_list = []
            features_per_var_list = []  # Track features per variable AFTER masking

            for i in range(n_vars):
                data_i = data_reshaped[:, i, :]  # (n_samples, n_features)

                # Apply variable-specific mask if available
                if isinstance(variant_mask, np.ndarray):
                    if variant_mask.ndim == 2:
                        # variant_mask shape: (n_vars, n_features_per_var)
                        mask_i = np.flatnonzero(variant_mask[i])
                    elif variant_mask.ndim == 1:
                        # variant_mask shape: (n_features_total,) - single mask for all
                        mask_i = np.flatnonzero(variant_mask)
                    else:
                        raise ValueError(f"Unexpected variant_mask shape: {variant_mask.shape}")

                    data_i_masked = np.take(data_i, mask_i, axis=1)
                else:
                    data_i_masked = data_i

                data_masked_list.append(data_i_masked)
                features_per_var_list.append(data_i_masked.shape[1])
                logger.info(f"Variable {i} features after masking: {data_i_masked.shape[1]}")

            # Flatten concatenated data
            data = np.concatenate(data_masked_list, axis=1)  # (n_samples, total_features)

            # Free up memory
            del data_i, data_i_masked, data_masked_list
        else:
            # No masking: all variables have same number of features
            data = data_reshaped.reshape(n_samples, -1)
            features_per_var_list = [data_reshaped.shape[2]] * n_vars

        # Compute full covariance matrix (before regularization)
        logger.info("Computing covariance matrix...")
        cov['matrix'] = scaling_factor * (data.T @ data) / denom

        # Free up memory
        del data, data_reshaped

        # Apply regularization BEFORE computing correlation to guarantee matrix consistency
        if regularization_factor > 0:
            if regularization_type == 'per_variable':
                logger.info("Applying per-variable regularization to covariance matrix...")

                # Loop over variables with their specific feature counts
                start_idx = 0
                for i, features_per_var in enumerate(features_per_var_list):
                    end_idx = start_idx + features_per_var

                    # Extract diagonal for this variable's block
                    var_diag = np.diag(cov['matrix'])[start_idx:end_idx]
                    var_mean_diag = float(np.mean(var_diag))

                    logger.info(
                        f"Applying regularization for variable {i} (features={features_per_var}, mean_diag={var_mean_diag:.6e})...")

                    # Add regularization to the diagonal block of this variable
                    cov['matrix'][
                        start_idx:end_idx, start_idx:end_idx] += regularization_factor * var_mean_diag * np.eye(
                        features_per_var, dtype=cov['matrix'].dtype)
                    start_idx = end_idx

            elif regularization_type == 'global':
                logger.info("Applying global regularization to covariance matrix...")
                cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(
                    cov['matrix'].shape[0], dtype=cov['matrix'].dtype)

            elif regularization_type == 'correlation_based':
                logger.info("Applying correlation-based regularization to covariance matrix...")
                # Ensure symmetry before reconstruction
                cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)
                std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
                cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
                cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry
                # Regularize correlation
                cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(
                    cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
                # Reconstruct covariance from regularized correlation
                cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']

        # Compute correlation matrix AFTER regularization (if requested and not already derived)
        if hasattr(output, 'correlation') and 'correlation' not in cov:
            logger.info("Computing correlation matrix from regularized covariance matrix...")
            std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
            cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])

    # Compute diagonal
    if hasattr(output, 'diagonal'):
        logger.info("Computing diagonal of the covariance matrix...")
        cov['diagonal'] = np.diag(np.diag(cov['matrix']))
    if hasattr(output, 'diagonal_inverse'):
        logger.info("Computing diagonal inverse of the covariance matrix...")
        cov['diagonal_inverse'] = np.diag(1.0 / np.maximum(np.diag(cov['matrix']), 1e-12))
    # Compute Cholesky decomposition
    if hasattr(output, 'matrix_cholesky'):
        logger.info("Computing Cholesky decomposition of the covariance matrix...")
        cov['matrix_cholesky'] = np.linalg.cholesky(cov['matrix'])
    # Compute matrix inverse
    if hasattr(output, 'matrix_inverse'):
        logger.info("Computing the inverse of the covariance matrix...")
        cov['matrix_inverse'] = np.linalg.inv(cov['matrix'])
    # Compute matrix pseudo-inverse
    if hasattr(output, 'matrix_pseudo_inverse'):
        logger.info("Computing the pseudo-inverse of the covariance matrix...")
        rcond = OmegaConf.select(output.matrix_pseudo_inverse, 'params.rcond', default=1.e-3)
        cov['matrix_pseudo_inverse'] = np.linalg.pinv(cov['matrix'], rcond=rcond)
    # Compute Cholesky decomposition
    if hasattr(output, 'correlation_cholesky') and 'correlation' in cov:
        logger.info("Computing Cholesky decomposition of the correlation matrix...")
        cov['correlation_cholesky'] = np.linalg.cholesky(cov['correlation'])
    # Compute inverse of the correlation matrix
    if hasattr(output, 'correlation_inverse') and 'correlation' in cov:
        logger.info("Computing the inverse of the correlation matrix...")
        cov['correlation_inverse'] = np.linalg.inv(cov['correlation'])
    # Compute pseudo inverse of the correlation matrix
    if hasattr(output, 'correlation_pseudo_inverse') and 'correlation' in cov:
        logger.info("Computing the pseudo-inverse of the correlation matrix...")
        rcond = OmegaConf.select(output.correlation_pseudo_inverse, 'params.rcond', default=1.e-3)
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
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse', 'diagonal',
                                     'diagonal_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return



def pressure_only_matrix2(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
                         regularization_factor: float = 1.0, regularization_type: str = 'per_variable',
                         plot_flag: bool = True, recenter: bool = False,
                         univariate: bool = False, mean_type: str = 'spatiotemporal',
                         apply_transform: bool = False) -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        scaling_factor: float. Scaling factor for covariance matrix.
        regularization_factor: float. Regularization factor for covariance matrix.
        regularization_type: str. Type of regularization to apply.
            Options:
            - 'per_variable': Apply regularization per-variable block (recommended, current default)
            - 'global': Apply global regularization to entire matrix using global mean diagonal
            - 'correlation_based': Regularize via correlation matrix, then reconstruct covariance
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
        logger.info("Applying spatial mask...")
        mask &= instantiate(input.spatial_mask)
    # Temporal mask: Consider only data within the specified time bounds
    if hasattr(input, 'temporal_mask') and input.temporal_mask is not None:
        logger.info("Applying temporal mask...")
        mask &= instantiate(input.temporal_mask)
    # Apply mask to data
    data = data[mask]

    # Dimensions
    data_shape = data.shape
    n_samples, n_vars = data_shape[0], data_shape[1]
    # Denominator (computation of the mean)
    denom = float(n_samples - 1)

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
                group_counts = np.bincount(inverse_indices)[:, None].astype(data.dtype)  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

                # Save mean and unique coordinates for prior_from_mean function
                if hasattr(output, 'mu') and hasattr(output.mu, 'save'):
                    logger.info("Saving climatological mean to file...")
                    # Reshape means back to original dimensions (excluding batch axis)
                    inverse_transform_fn = (
                        Compose(transformations=input.data.get('transformations', None), inverse_transform=True)
                        if apply_transform and input.data.get('transformations') is not None
                        else identity
                    )
                    mu_to_save = inverse_transform_fn(group_means.reshape(n_unique_coords, *data_shape[1:]))
                    save_func_mean = instantiate(output.mu.save)
                    save_func_mean(mu_to_save)

                if hasattr(output, 'unique_coords') and hasattr(output.unique_coords, 'save'):
                    logger.info("Saving unique coordinates to file...")
                    save_func_coords = instantiate(output.unique_coords.save)
                    save_func_coords(unique_coords)

                # Compute anomalies (truth - climatological mean)
                data -= group_means[inverse_indices]  # Shape: (n_samples, n_features)
                # Broadcast the means back out to match the original sample layout
                data = data.reshape(data_shape)
                # Denominator (computation of the mean)
                denom = float(n_samples - n_unique_coords)
            else:
                mu = np.mean(data)
                # Compute anomalies (truth - mean)
                data -= mu
        else:
            raise ValueError("mean_type not supported.")

    # Guard against invalid degrees of freedom (e.g. n_samples <= n_groups)
    if denom <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom for covariance computation: denominator is {denom:.1f}. "
            f"Ensure total samples ({n_samples}) exceed group count."
        )

    # Variant filter (prior only)
    if hasattr(input, 'variant_mask') and input.variant_mask is not None:
        # Load filter
        variant_mask = instantiate(input.variant_mask.load)
        if n_vars != variant_mask.shape[0]:
            variant_mask = np.take(variant_mask, [0], axis=0)
    else:
        variant_mask = None

    # Covariance matrix initialization
    cov = {}

    # Validate regularization_type
    valid_reg_types = ['per_variable', 'global', 'correlation_based']
    if regularization_type not in valid_reg_types:
        raise ValueError(f"regularization_type must be one of {valid_reg_types}, got '{regularization_type}'")

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <= 2:
            # raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
            data = data[..., None]  # Add a singleton dimension for levels if not present
        # Initialize empty lists for matrices
        m_cov = []
        m_corr = []
        # Loop over variables
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply variant mask
            if variant_mask is not None:
                logger.info(f"Applying variant mask for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(variant_mask[i]), axis=1)
            # Compute covariance block
            logger.info(f"Computing univariate covariance matrix for variable {i}...")
            sub_cov = scaling_factor * (data_i.T @ data_i) / denom

            # Apply per-variable regularization directly to sub_cov BEFORE computing correlation
            if regularization_factor > 0 and regularization_type == 'per_variable':
                var_mean_diag = float(np.mean(np.diag(sub_cov)))
                logger.info(f"Applying per-variable regularization for variable {i} (mean_diag={var_mean_diag:.6e})...")
                sub_cov = sub_cov + regularization_factor * var_mean_diag * np.eye(sub_cov.shape[0],
                                                                                   dtype=sub_cov.dtype)

            # Compute correlation block from regularized sub_cov (with zero-variance safety)
            if hasattr(output, 'correlation'):
                std_i = np.sqrt(np.maximum(np.diag(sub_cov), 1e-12))
                sub_corr = sub_cov / (std_i[:, None] @ std_i[None, :])
                m_corr.append(sub_corr)

            m_cov.append(sub_cov)

        # Assemble into block-diagonal matrices
        del data_i, data
        cov['matrix'] = block_diag(*m_cov)
        if hasattr(output, 'correlation'):
            cov['correlation'] = block_diag(*m_corr)

        # Apply global regularization to assembled matrix if requested
        if regularization_factor > 0 and regularization_type == 'global':
            logger.info("Applying global regularization to univariate covariance matrix...")
            cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(
                cov['matrix'].shape[0], dtype=cov['matrix'].dtype)
            # Recompute block correlation matrix after global regularization
            if hasattr(output, 'correlation'):
                std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
                cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])

        # Apply correlation-based regularization if requested
        elif regularization_factor > 0 and regularization_type == 'correlation_based':
            logger.info("Applying correlation-based regularization to univariate covariance matrix...")
            # Recompute from correlation to ensure consistency
            cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)  # Ensure symmetry
            std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
            cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
            cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry
            # Regularize correlation
            cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(
                cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
            # Reconstruct covariance from regularized correlation
            cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']

    # Multivariate matrix computation steps
    else:
        # Keep track of variable boundaries for per-variable regularization
        # Reshape to (n_samples, n_vars, -1) to preserve variable dimension
        data_reshaped = data.reshape(n_samples, n_vars, -1)  # (n_samples, n_vars, total_features_per_var)

        # Apply variant mask BEFORE flattening to track per-variable feature counts
        if variant_mask is not None:
            logger.info("Applying pressure mask...")
            # variant_mask shape: (n_vars, n_features_per_var) or similar
            # Need to apply per-variable masking
            data_masked_list = []
            features_per_var_list = []  # Track features per variable AFTER masking

            for i in range(n_vars):
                data_i = data_reshaped[:, i, :]  # (n_samples, n_features)

                # Apply variable-specific mask if available
                if isinstance(variant_mask, np.ndarray):
                    if variant_mask.ndim == 2:
                        # variant_mask shape: (n_vars, n_features_per_var)
                        mask_i = np.flatnonzero(variant_mask[i])
                    elif variant_mask.ndim == 1:
                        # variant_mask shape: (n_features_total,) - single mask for all
                        mask_i = np.flatnonzero(variant_mask)
                    else:
                        raise ValueError(f"Unexpected variant_mask shape: {variant_mask.shape}")

                    data_i_masked = np.take(data_i, mask_i, axis=1)
                else:
                    data_i_masked = data_i

                data_masked_list.append(data_i_masked)
                features_per_var_list.append(data_i_masked.shape[1])
                logger.info(f"Variable {i} features after masking: {data_i_masked.shape[1]}")

            # Flatten concatenated data
            data = np.concatenate(data_masked_list, axis=1)  # (n_samples, total_features)

            # Free up memory
            del data_i, data_i_masked, data_masked_list
        else:
            # No masking: all variables have same number of features
            data = data_reshaped.reshape(n_samples, -1)
            features_per_var_list = [data_reshaped.shape[2]] * n_vars

        # Compute full covariance matrix (before regularization)
        logger.info("Computing covariance matrix...")
        cov['matrix'] = scaling_factor * (data.T @ data) / denom

        # Free up memory
        del data, data_reshaped

        # Apply regularization BEFORE computing correlation to guarantee matrix consistency
        if regularization_factor > 0:
            if regularization_type == 'per_variable':
                logger.info("Applying per-variable regularization to covariance matrix...")

                # Loop over variables with their specific feature counts
                start_idx = 0
                for i, features_per_var in enumerate(features_per_var_list):
                    end_idx = start_idx + features_per_var

                    # Extract diagonal for this variable's block
                    var_diag = np.diag(cov['matrix'])[start_idx:end_idx]
                    var_mean_diag = float(np.mean(var_diag))

                    logger.info(
                        f"Applying regularization for variable {i} (features={features_per_var}, mean_diag={var_mean_diag:.6e})...")

                    # Add regularization to the diagonal block of this variable
                    cov['matrix'][
                        start_idx:end_idx, start_idx:end_idx] += regularization_factor * var_mean_diag * np.eye(
                        features_per_var, dtype=cov['matrix'].dtype)
                    start_idx = end_idx

            elif regularization_type == 'global':
                logger.info("Applying global regularization to covariance matrix...")
                cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(
                    cov['matrix'].shape[0], dtype=cov['matrix'].dtype)

            elif regularization_type == 'correlation_based':
                logger.info("Applying correlation-based regularization to covariance matrix...")
                # Ensure symmetry before reconstruction
                cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)
                std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
                cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
                cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry
                # Regularize correlation
                cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(
                    cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
                # Reconstruct covariance from regularized correlation
                cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']

        # Compute correlation matrix AFTER regularization (if requested and not already derived)
        if hasattr(output, 'correlation') and 'correlation' not in cov:
            logger.info("Computing correlation matrix from regularized covariance matrix...")
            std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
            cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])

    # 1. Retrieve the rcond truncation threshold from output config if specified
    rcond = 1.e-3
    if hasattr(output, 'matrix_pseudo_inverse'):
        rcond = OmegaConf.select(output.matrix_pseudo_inverse, 'params.rcond', default=1.e-3)

    # 2. Spectral decomposition of cov['matrix']
    if any(hasattr(output, key) for key in
           ['matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse', 'diagonal', 'diagonal_inverse']):
        logger.info("Performing spectral decomposition on covariance matrix...")

        # Enforce exact symmetry
        cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)
        eigvals, eigvecs = np.linalg.eigh(cov['matrix'])

        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Determine cutoff threshold and active subspace rank
        max_eig = eigvals[0]
        cutoff = rcond * max_eig
        mask = eigvals > cutoff
        k = int(np.sum(mask))

        logger.info(
            f"Covariance subspace truncation: keeping top {k}/{len(eigvals)} modes (rcond={rcond:.1e}, cutoff={cutoff:.2e})")

        # Active components
        eigvals_k = eigvals[:k]
        eigvecs_k = eigvecs[:, :k]

        # Reconstruct regularized rank-k covariance matrix
        cov['matrix'] = (eigvecs_k * eigvals_k) @ eigvecs_k.T

        # Compute pseudo-inverse / inverse (identical in active range space)
        inv_eigvals = 1.0 / eigvals_k
        B_pinv = (eigvecs_k * inv_eigvals) @ eigvecs_k.T

        if hasattr(output, 'matrix_pseudo_inverse'):
            cov['matrix_pseudo_inverse'] = B_pinv

        if hasattr(output, 'matrix_inverse'):
            # Set matrix_inverse to pinv to avoid amplifying nullspace noise
            cov['matrix_inverse'] = B_pinv

        if hasattr(output, 'matrix_cholesky'):
            # Low-rank square root L of shape (N, k) such that B = L @ L.T
            # Guarantees all generated perturbations lie in the range space of B+
            cov['matrix_cholesky'] = eigvecs_k * np.sqrt(eigvals_k)

        if hasattr(output, 'diagonal'):
            cov['diagonal'] = np.diag(np.diag(cov['matrix']))

        if hasattr(output, 'diagonal_inverse'):
            diag_vals = np.diag(cov['matrix'])
            cov['diagonal_inverse'] = np.diag(np.where(diag_vals > 0, 1.0 / diag_vals, 0.0))

    # 3. Spectral decomposition of cov['correlation'] (if requested)
    if 'correlation' in cov and any(hasattr(output, key) for key in
                                    ['correlation_cholesky', 'correlation_inverse', 'correlation_pseudo_inverse']):
        logger.info("Performing spectral decomposition on correlation matrix...")

        rcond_corr = 1.e-3
        if hasattr(output, 'correlation_pseudo_inverse'):
            rcond_corr = OmegaConf.select(output.correlation_pseudo_inverse, 'params.rcond', default=1.e-3)

        cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)
        c_eigvals, c_eigvecs = np.linalg.eigh(cov['correlation'])

        c_idx = np.argsort(c_eigvals)[::-1]
        c_eigvals = c_eigvals[c_idx]
        c_eigvecs = c_eigvecs[:, c_idx]

        c_max_eig = c_eigvals[0]
        c_cutoff = rcond_corr * c_max_eig
        c_mask = c_eigvals > c_cutoff
        c_k = int(np.sum(c_mask))

        logger.info(
            f"Correlation subspace truncation: keeping top {c_k}/{len(c_eigvals)} modes (rcond={rcond_corr:.1e})")

        c_eigvals_k = c_eigvals[:c_k]
        c_eigvecs_k = c_eigvecs[:, :c_k]

        cov['correlation'] = (c_eigvecs_k * c_eigvals_k) @ c_eigvecs_k.T

        c_inv_eigvals = 1.0 / c_eigvals_k
        C_pinv = (c_eigvecs_k * c_inv_eigvals) @ c_eigvecs_k.T

        if hasattr(output, 'correlation_pseudo_inverse'):
            cov['correlation_pseudo_inverse'] = C_pinv

        if hasattr(output, 'correlation_inverse'):
            cov['correlation_inverse'] = C_pinv

        if hasattr(output, 'correlation_cholesky'):
            cov['correlation_cholesky'] = c_eigvecs_k * np.sqrt(c_eigvals_k)

    # Loop over keys
    for key in list(cov.keys()):
        # Save to file
        if hasattr(output, key):
            if hasattr(output[key], 'save'):
                logger.info(f"Saving {key} matrix...")
                save_func = instantiate(output[key].save)
                save_func(cov[key])
            # Plot covariance matrix if requested
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse', 'diagonal',
                                     'diagonal_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


def persistent_matrix(
    input: DictConfig,
    output: DictConfig,
    method: str = "persistence",
    expected_scan_step: int = 1,
    missing_timestep_policy: str = "reject",
    scaling_factor: float = 1.0,
    regularization_factor: float = 0.0,
    plot_flag: bool = True,
    recenter: bool = True,
    univariate: bool = False,
    apply_transform: bool = False,
) -> None:
    """Compute a background-error covariance matrix from temporal persistence.

    The background profile is constructed independently at every latitude and
    longitude coordinate using atmospheric profiles from neighboring scans.
    All eligible location-scan background-error samples are pooled to estimate
    a climatological background-error covariance matrix. The covariance is
    assumed to be independent of latitude, longitude, and scan.

    Parameters
    ----------
    input : DictConfig
        Main Hydra configuration containing atmospheric-profile data,
        latitude, longitude, scans, and optional masks.
    output : DictConfig
        Output configuration specifying which covariance products should be
        saved and plotted.
    method : str
        Background-generation method. Supported values are ``persistence``
        and ``centered_average``.
    expected_scan_step : int
        Expected difference between neighboring scan indices. If consecutive
        scan indices correspond to six-hour timesteps, the default value of
        one requires an exact six-hour temporal separation.
    missing_timestep_policy : str
        Policy used when an exact neighboring scan is unavailable.

        ``reject``
            Reject the candidate error sample. This is the default and
            recommended policy.

        ``available``
            Use the immediately previous or next available profile, even when
            the scan-index separation differs from ``expected_scan_step``.
            This can mix persistence errors from different temporal lags.

        ``raise``
            Raise an exception when a candidate target profile does not have
            the required exact neighboring scan.
    scaling_factor : float
        Multiplicative scaling applied to the estimated covariance matrix.
        Multiplying covariance by ``scaling_factor`` multiplies corresponding
        standard deviations by ``sqrt(scaling_factor)``.
    regularization_factor : float
        Correlation-shrinkage coefficient in the interval [0, 1].
        Regularization is applied as

        ``C_reg = (1 - alpha) * C + alpha * I``.

        A value of zero applies no regularization. A value of one removes all
        off-diagonal correlations while preserving pressure-dependent
        variances.
    plot_flag : bool
        If True, plot requested covariance products.
    recenter : bool
        If True, subtract the mean background error before estimating the
        covariance. The saved mean error can be interpreted as the systematic
        persistence bias.
    univariate : bool
        If True, estimate one independent pressure-level covariance block for
        every atmospheric variable and assemble a block-diagonal covariance.
        If False, estimate a full multivariate covariance matrix.
    apply_transform : bool
        If True, apply configured profile transformations before constructing
        temporal backgrounds and errors. The resulting covariance is defined
        in the transformed state space.

    Returns
    -------
    None
        Requested covariance products are saved through the output
        configuration.

    Notes
    -----
    The function assumes that ``input.scans`` contains integer scan indices
    with a globally consistent temporal interpretation. If consecutive scan
    indices correspond to six-hour intervals, ``expected_scan_step=1``
    estimates a six-hour persistence-error covariance.

    Under the default ``reject`` policy, a target at scan ``k`` requires an
    available profile at scan ``k - expected_scan_step`` for persistence. For
    centered averaging, profiles must be available at both
    ``k - expected_scan_step`` and ``k + expected_scan_step``.

    The ``available`` policy should be used cautiously because it can mix
    errors from different temporal intervals.
    """

    # Begin by loading the data
    logger.info("Loading atmospheric profiles...")
    data = load_variable(input.data, apply_transform=apply_transform)

    # Read coordinates and scans
    lat = load_variable(input.lat)
    lon = load_variable(input.lon)
    scans = load_variable(input.scans)

    # Convert inputs to NumPy arrays
    data = np.asarray(data)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    scans = np.asarray(scans)

    # Validate leading dimensions
    n_input_samples = data.shape[0]

    # Build sample mask
    mask = np.ones(n_input_samples, dtype=bool)

    # Spatial mask: Consider only data within the specified latitude
    # and longitude bounds
    if (hasattr(input, "spatial_mask") and input.spatial_mask is not None):
        logger.info("Applying spatial mask...")
        mask &= np.asarray(instantiate(input.spatial_mask), dtype=bool)

    # Temporal mask: Consider only data within the specified scan bounds
    if (hasattr(input, "temporal_mask") and input.temporal_mask is not None):
        logger.info("Applying temporal mask...")
        mask &= np.asarray(instantiate(input.temporal_mask), dtype=bool)

    # Apply masks to data, coordinates, and scans
    data = data[mask]
    lat = lat[mask]
    lon = lon[mask]
    scans = scans[mask]

    # Dimensions
    if data.ndim < 3:
        data = data.reshape(data.shape[0], data.shape[1], 1)
    data_shape = data.shape
    n_samples = data_shape[0]
    n_vars = data_shape[1]

    # Identify unique horizontal coordinates and their mapping
    #
    # unique_coords contains the physical locations.
    # inverse_indices contains the location ID for every sample.
    coords = np.column_stack((lat, lon))
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    n_unique_coords = len(unique_coords)
    logger.info("Found %d unique horizontal coordinates.", n_unique_coords)

    # Flatten all profile feature dimensions temporarily
    #
    # Expected original shape:
    #   (n_samples, n_variables, n_levels)
    #
    # Flattened shape:
    #   (n_samples, n_variables * n_levels)
    data_flat = data.reshape(n_samples, -1)

    # Store generated background errors
    error_samples = []

    # Keep track of sample indices for diagnostics
    target_indices = []
    previous_indices = []
    next_indices = []

    # Keep track of scan-index separations used
    previous_scan_steps = []
    next_scan_steps = []

    # Count accepted and rejected candidate samples
    n_candidate_targets = 0
    n_accepted = 0
    n_rejected_missing_previous = 0
    n_rejected_missing_next = 0
    n_nonstandard_lag_accepted = 0

    # Track number of available temporal samples per coordinate
    samples_per_coordinate = np.bincount(
        inverse_indices,
        minlength=n_unique_coords,
    )

    logger.info(
        "Constructing background errors using method '%s', "
        "expected scan step %d, and missing-timestep policy '%s'...",
        method,
        expected_scan_step,
        missing_timestep_policy,
    )

    # Loop over horizontal coordinates
    for location_id in range(n_unique_coords):

        # Find all samples available at this horizontal coordinate
        location_sample_indices = np.flatnonzero(inverse_indices == location_id)

        # Sort samples at this coordinate by scan index
        location_scans = scans[location_sample_indices]
        order = np.argsort(location_scans, kind="stable")
        sorted_indices = location_sample_indices[order]
        sorted_scans = location_scans[order]

        # Duplicate scans at the same coordinate make temporal pairing ambiguous.
        if sorted_scans.shape[0] > 1:
            duplicate_scan_mask = (sorted_scans[1:] == sorted_scans[:-1])

            if np.any(duplicate_scan_mask):
                duplicate_scans = sorted_scans[1:][duplicate_scan_mask]

                raise ValueError(
                    "Duplicate scans were found at coordinate "
                    f"{unique_coords[location_id].tolist()}: "
                    f"{duplicate_scans.tolist()}. Each coordinate-scan "
                    "pair must identify one atmospheric profile."
                )

        # Persistence requires one previous profile
        if method == "persistence":

            if sorted_indices.shape[0] < 2:
                continue

            # Every sample after the first is a candidate target.
            for scan_index in range(1, sorted_indices.shape[0]):
                n_candidate_targets += 1

                previous_index = sorted_indices[scan_index - 1]
                target_index = sorted_indices[scan_index]

                previous_scan = sorted_scans[scan_index - 1]
                target_scan = sorted_scans[scan_index]
                previous_scan_step = int(target_scan - previous_scan)
                exact_previous_available = (previous_scan_step == expected_scan_step)

                # Treat a missing exact previous scan according to
                # the selected policy.
                if not exact_previous_available:

                    if missing_timestep_policy == "reject":
                        n_rejected_missing_previous += 1
                        continue

                    if missing_timestep_policy == "raise":
                        raise ValueError(
                            "Exact previous scan is unavailable for "
                            f"coordinate "
                            f"{unique_coords[location_id].tolist()} "
                            f"at target scan {int(target_scan)}. "
                            f"Observed previous scan step is "
                            f"{previous_scan_step}; expected "
                            f"{expected_scan_step}."
                        )

                    # Under the 'available' policy, retain the previous
                    # available profile even when its scan-index
                    # separation differs.
                    n_nonstandard_lag_accepted += 1

                x_previous = data_flat[previous_index]
                x_true = data_flat[target_index]

                # Background definition:
                # x_b(s_i) = x(s_previous)
                x_background = x_previous

                # Background error:
                # e(s_i) = x_b(s_i) - x_true(s_i)
                x_error = x_background - x_true

                error_samples.append(x_error)
                previous_indices.append(previous_index)
                target_indices.append(target_index)
                previous_scan_steps.append(previous_scan_step)

                n_accepted += 1

        # Centered averaging requires one previous and one next profile
        elif method == "centered_average":

            if sorted_indices.shape[0] < 3:
                continue

            # The first and last samples cannot be candidate targets.
            for scan_index in range(1, sorted_indices.shape[0] - 1):
                n_candidate_targets += 1

                previous_index = sorted_indices[scan_index - 1]
                target_index = sorted_indices[scan_index]
                next_index = sorted_indices[scan_index + 1]

                previous_scan = sorted_scans[scan_index - 1]
                target_scan = sorted_scans[scan_index]
                next_scan = sorted_scans[scan_index + 1]
                previous_scan_step = int(target_scan - previous_scan)
                next_scan_step = int(next_scan - target_scan)
                exact_previous_available = (previous_scan_step == expected_scan_step)
                exact_next_available = (next_scan_step == expected_scan_step)
                exact_neighbors_available = (exact_previous_available and exact_next_available)

                # Treat missing exact neighbors according to
                # the selected policy.
                if not exact_neighbors_available:

                    if missing_timestep_policy == "reject":

                        if not exact_previous_available:
                            n_rejected_missing_previous += 1

                        if not exact_next_available:
                            n_rejected_missing_next += 1

                        continue

                    if missing_timestep_policy == "raise":
                        raise ValueError(
                            "Exact centered-average neighbors are "
                            "unavailable for coordinate "
                            f"{unique_coords[location_id].tolist()} "
                            f"at target scan {int(target_scan)}. "
                            f"Observed scan steps are "
                            f"{previous_scan_step} and "
                            f"{next_scan_step}; expected "
                            f"{expected_scan_step} on both sides."
                        )

                    # Under the 'available' policy, retain adjacent
                    # available profiles even when their scan-index
                    # separations differ.
                    n_nonstandard_lag_accepted += 1

                x_previous = data_flat[previous_index]
                x_true = data_flat[target_index]
                x_next = data_flat[next_index]

                # Background definition:
                # x_b(s_i) =
                # 0.5 * [x(s_previous) + x(s_next)]
                x_background = 0.5 * (x_previous + x_next)

                # Background error:
                # e(s_i) = x_b(s_i) - x_true(s_i)
                x_error = x_background - x_true

                error_samples.append(x_error)
                previous_indices.append(previous_index)
                target_indices.append(target_index)
                next_indices.append(next_index)
                previous_scan_steps.append(previous_scan_step)
                next_scan_steps.append(next_scan_step)

                n_accepted += 1

    if len(error_samples) == 0:
        raise ValueError(
            "No eligible temporal profile combinations were found. "
            f"The selected method is '{method}', the expected scan "
            f"step is {expected_scan_step}, and the missing-timestep "
            f"policy is '{missing_timestep_policy}'."
        )

    # Convert error samples to a single array
    #
    # Shape:
    #   (n_error_samples, n_variables * n_levels)
    error_samples = np.stack(error_samples, axis=0)
    n_error_samples = error_samples.shape[0]

    # Convert diagnostics to arrays
    target_indices = np.asarray(target_indices, dtype=np.int64)
    previous_indices = np.asarray(previous_indices, dtype=np.int64)
    previous_scan_steps = np.asarray(previous_scan_steps, dtype=np.int64)
    if method == "centered_average":
        next_indices = np.asarray(next_indices, dtype=np.int64)
        next_scan_steps = np.asarray(next_scan_steps, dtype=np.int64)

    logger.info(
        "Constructed %d background-error samples from %d "
        "candidate targets.",
        n_error_samples,
        n_candidate_targets,
    )

    logger.info(
        "Accepted fraction: %.4f",
        n_accepted / max(n_candidate_targets, 1),
    )

    logger.info(
        "Rejected because an exact previous scan was unavailable: %d",
        n_rejected_missing_previous,
    )

    if method == "centered_average":
        logger.info(
            "Rejected because an exact next scan was unavailable: %d",
            n_rejected_missing_next,
        )

    if missing_timestep_policy == "available":
        logger.warning(
            "Accepted %d samples with nonstandard scan steps. "
            "The resulting covariance mixes persistence errors "
            "from different temporal intervals.",
            n_nonstandard_lag_accepted,
        )

    logger.info(
        "Previous-profile scan-step statistics: "
        "min=%d, median=%.1f, max=%d.",
        int(np.min(previous_scan_steps)),
        float(np.median(previous_scan_steps)),
        int(np.max(previous_scan_steps)),
    )

    if method == "centered_average":
        logger.info(
            "Next-profile scan-step statistics: "
            "min=%d, median=%.1f, max=%d.",
            int(np.min(next_scan_steps)),
            float(np.median(next_scan_steps)),
            int(np.max(next_scan_steps)),
        )

    logger.info(
        "Temporal samples per coordinate: "
        "min=%d, median=%.1f, max=%d.",
        int(np.min(samples_per_coordinate)),
        float(np.median(samples_per_coordinate)),
        int(np.max(samples_per_coordinate)),
    )

    # Compute and preserve mean background error
    #
    # A nonzero mean represents a systematic bias of the selected
    # background-generation method. B should normally describe
    # random errors around this mean rather than absorbing the bias.
    error_mean = np.mean(error_samples, axis=0, keepdims=True)

    # Apply recentering
    if recenter:
        logger.info("Recentering background errors around their mean...")
        error_samples = error_samples - error_mean

        # One mean background-error profile was estimated.
        denom = float(n_error_samples - 1)

    else:
        logger.info("Computing second moments without error recentering...")
        denom = float(n_error_samples)

    if denom <= 0.0:
        raise ValueError("Insufficient background-error samples to estimate the covariance matrix.")

    # Restore variable and profile-level dimensions
    error_samples = error_samples.reshape(n_error_samples, *data_shape[1:])

    # Variant filter
    if hasattr(input, "variant_mask") and input.variant_mask is not None:
        variant_mask = instantiate(input.variant_mask.load)
        variant_mask = np.asarray(variant_mask)

        if variant_mask.ndim == 1:
            # Apply one shared level mask to every variable
            variant_mask = np.repeat(variant_mask[None, :], n_vars, axis=0)

        elif variant_mask.ndim == 2 and variant_mask.shape[0] == 1 and n_vars > 1:
            variant_mask = np.repeat(variant_mask, n_vars, axis=0)

        elif variant_mask.ndim != 2 or variant_mask.shape[0] != n_vars:
            raise ValueError(
                "variant_mask must have shape (n_variables, n_levels), (1, n_levels), or (n_levels,). "
                f"Received shape {variant_mask.shape}."
            )

    else:
        variant_mask = None

    # Covariance matrix initialization
    cov = {}

    # Store diagnostics that may optionally be saved through output
    cov["error_mean"] = error_mean.reshape(data_shape[1:])
    cov["n_error_samples"] = np.asarray(n_error_samples, dtype=np.int64)
    cov["samples_per_coordinate"] = samples_per_coordinate
    cov["previous_scan_steps"] = previous_scan_steps
    cov["target_indices"] = target_indices
    cov["previous_indices"] = previous_indices

    if method == "centered_average":
        cov["next_scan_steps"] = next_scan_steps
        cov["next_indices"] = next_indices

    # Univariate matrix computation steps
    if univariate:

        # Check dimensions
        if error_samples.ndim <= 2:
            raise ValueError("Univariate covariance matrix computation requires background errors with more than two dimensions.")

        # Initialize empty lists for matrices
        m_cov = []
        m_corr = []

        # Loop over variables
        for i in range(n_vars):

            # Select all pressure-level errors for this variable
            error_i = error_samples[:, i]

            # Flatten any dimensions after the variable dimension
            error_i = error_i.reshape(n_error_samples, -1)

            # Apply variant mask
            if variant_mask is not None:
                logger.info("Applying variant mask for variable %d...",i)

                mask_i = np.flatnonzero(variant_mask[i])
                error_i = np.take(error_i, mask_i, axis=1)

            # Compute univariate covariance block
            logger.info(
                "Computing univariate persistence covariance matrix for variable %d...",
                i,
            )

            sub_cov = scaling_factor * (error_i.T @ error_i) / denom

            # Ensure numerical symmetry
            sub_cov = 0.5 * (sub_cov + sub_cov.T)

            # Compute pressure-dependent variances
            variances_i = np.diag(sub_cov).copy()

            if np.any(~np.isfinite(variances_i)):
                raise ValueError(f"Variable {i} contains non-finite variances.")

            if np.any(variances_i <= 0.0):
                bad_indices = np.flatnonzero(variances_i <= 0.0)

                raise ValueError(f"Variable {i} has non-positive variances at indices {bad_indices.tolist()}.")

            std_i = np.sqrt(variances_i)
            std_outer_i = np.outer(std_i, std_i)

            # Compute correlation block from unregularized covariance
            sub_corr = sub_cov / std_outer_i

            # Ensure numerical symmetry and exact unit diagonal
            sub_corr = 0.5 * (sub_corr + sub_corr.T)

            np.fill_diagonal(sub_corr,1.0)

            # Apply per-variable regularization through the
            # correlation matrix
            if regularization_factor > 0.0:
                logger.info(
                    "Applying correlation regularization for variable %d with factor %.6g...",
                    i, regularization_factor,
                )

                sub_corr = ((1.0 - regularization_factor) * sub_corr +
                            regularization_factor * np.eye(sub_corr.shape[0], dtype=sub_corr.dtype))

                # Defensive numerical cleanup
                sub_corr = 0.5 * (sub_corr + sub_corr.T)
                np.fill_diagonal(sub_corr,1.0)

                # Reconstruct covariance while preserving the original
                # pressure-dependent variances
                sub_cov = (std_outer_i * sub_corr)
                sub_cov = 0.5 * (sub_cov + sub_cov.T)

            m_cov.append(sub_cov)
            m_corr.append(sub_corr)

        # Assemble into block-diagonal matrices
        cov["matrix"] = block_diag(*m_cov)
        cov["correlation"] = block_diag(*m_corr)

        del error_i
        del error_samples

    # Multivariate matrix computation steps
    else:

        # Reshape to preserve the variable dimension
        error_reshaped = error_samples.reshape(n_error_samples, n_vars, -1)

        # Apply variant mask before flattening
        if variant_mask is not None:
            logger.info("Applying pressure mask...")

            error_masked_list = []

            for i in range(n_vars):

                error_i = error_reshaped[:, i, :]
                mask_i = np.flatnonzero(variant_mask[i])
                error_i_masked = np.take(error_i, mask_i, axis=1)
                error_masked_list.append(error_i_masked)

                logger.info("Variable %d features after masking: %d",i, error_i_masked.shape[1],)

            # Flatten concatenated background errors
            error_samples_flat = np.concatenate(error_masked_list, axis=1)

            del error_i
            del error_i_masked
            del error_masked_list

        else:
            # No masking: retain all variables and pressure levels
            error_samples_flat = error_reshaped.reshape(n_error_samples, -1)

        # Compute full covariance matrix
        logger.info("Computing multivariate persistence covariance matrix...")
        cov["matrix"] = scaling_factor * (error_samples_flat.T @ error_samples_flat)/ denom

        # Ensure numerical symmetry
        cov["matrix"] = 0.5 * (cov["matrix"] + cov["matrix"].T)

        # Compute component variances
        variances = np.diag(cov["matrix"]).copy()

        if np.any(~np.isfinite(variances)):
            raise ValueError("Covariance matrix contains non-finite variances.")

        if np.any(variances <= 0.0):
            bad_indices = np.flatnonzero(variances <= 0.0)
            raise ValueError(f"Covariance matrix has non-positive variances at indices {bad_indices.tolist()}.")

        std = np.sqrt(variances)
        std_outer = np.outer(std, std)

        # Compute correlation matrix from covariance matrix
        logger.info("Computing correlation matrix from covariance matrix...")
        cov["correlation"] = cov["matrix"] / std_outer

        # Ensure numerical symmetry and exact unit diagonal
        cov["correlation"] = 0.5 * (cov["correlation"] + cov["correlation"].T)

        np.fill_diagonal(cov["correlation"],1.0)

        # Apply regularization through the correlation matrix
        if regularization_factor > 0.0:
            logger.info("Applying correlation regularization with factor %.6g...",
                regularization_factor,
            )

            cov["correlation"] = ((1.0 - regularization_factor) * cov["correlation"] +
                                  regularization_factor * np.eye(cov["correlation"].shape[0], dtype=cov["correlation"].dtype))

            # Defensive numerical cleanup
            cov["correlation"] = 0.5 * (cov["correlation"] + cov["correlation"].T)
            np.fill_diagonal(cov["correlation"],1.0)

            # Reconstruct covariance while preserving the original
            # component variances
            cov["matrix"] = std_outer * cov["correlation"]
            cov["matrix"] = 0.5 * (cov["matrix"] + cov["matrix"].T)

        del error_samples
        del error_samples_flat
        del error_reshaped

    # Compute correlation matrix
    if hasattr(output, 'correlation'):
        std = np.sqrt(np.diag(cov['matrix']))
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
        rcond = output.matrix_pseudo_inverse.params.rcond if hasattr(output.matrix_pseudo_inverse, 'params') else 1.e-3
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
        rcond = output.correlation_pseudo_inverse.params.rcond if hasattr(output.correlation_pseudo_inverse, 'params') else 1.e-3
        cov['correlation_pseudo_inverse'] = np.linalg.pinv(cov['correlation'], rcond=rcond)
    if hasattr(output, 'diagonal'):
        logger.info("Computing diagonal of the covariance matrix...")
        cov['diagonal'] = np.diag(np.diag(cov['matrix']))
    if hasattr(output, 'diagonal_inverse'):
        logger.info("Computing diagonal inverse of the covariance matrix...")
        cov['diagonal_inverse'] = np.diag(1.0 / np.diag(cov['matrix']))

    # Loop over keys
    for key in list(cov.keys()):
        # Save to file
        if hasattr(output, key):
            if hasattr(output[key], 'save'):
               logger.info(f"Saving {key} matrix...")
               save_func = instantiate(output[key].save)
               save_func(cov[key])
            # Plot covariance matrix if requested
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse', 'diagonal', 'diagonal_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


def diagnose_covariance_matrix(
        B_joint: np.ndarray,
        pressure_hpa: np.ndarray,
        variable_names: list[str] = ['T', 'log_q', 'log_O3'],
        fitted_params: dict = None
) -> dict:
    """
    Performs numerical diagnostics on a joint covariance matrix B to identify
    the exact cause of positive-definiteness failure (LinAlgError).
    """
    n_vars = len(variable_names)
    n_levels = len(pressure_hpa)

    print("\n" + "=" * 70)
    print(" COVARIANCE MATRIX NUMERICAL DIAGNOSTICS")
    print("=" * 70)

    # --- 1. Check Symmetry ---
    sym_err = np.max(np.abs(B_joint - B_joint.T))
    print(f"\n[1] Symmetry Check:")
    print(f"    Max asymmetry |B - B^T|: {sym_err:.2e}")
    if sym_err > 1e-12:
        print("    --> WARNING: Matrix is asymmetric due to floating-point roundoff.")

    # --- 2. Diagonal Variance Statistics per Variable Block ---
    diag_B = np.diag(B_joint)
    print(f"\n[2] Diagonal Variance Range per Variable:")

    for i, var in enumerate(variable_names):
        start_idx = i * n_levels
        end_idx = (i + 1) * n_levels
        var_diag = diag_B[start_idx:end_idx]

        min_v, max_v = np.min(var_diag), np.max(var_diag)
        ratio = max_v / (min_v + 1e-30)
        min_p = pressure_hpa[np.argmin(var_diag)]

        print(f"    - {var:6s}: Min = {min_v:.2e} (at p={min_p:.3f} hPa) | Max = {max_v:.2e} | Ratio = {ratio:.2e}")
        if min_v < 1e-14:
            print(f"      --> CRITICAL: {var} variance is near zero or unfloored (< 1e-14)!")

    # Overall Dynamic Range
    overall_ratio = np.max(diag_B) / (np.min(diag_B) + 1e-30)
    print(f"    - Overall Variance Dynamic Range: {overall_ratio:.2e}")
    if overall_ratio > 1e12:
        print(
            "      --> CRITICAL: Dynamic range exceeds 1e12! Combined with correlation, this causes float64 underflow.")

    # --- 3. Eigendecomposition & Condition Numbers ---
    print(f"\n[3] Spectral Eigendecomposition:")

    # Full Joint Spectrum
    eigvals_full = np.linalg.eigvalsh(0.5 * (B_joint + B_joint.T))
    min_eig_full, max_eig_full = eigvals_full[0], eigvals_full[-1]
    neg_eigs_full = np.sum(eigvals_full <= 0)
    cond_full = max_eig_full / max(min_eig_full, 1e-30)

    print(f"    - Full Joint Matrix B ({B_joint.shape[0]}x{B_joint.shape[1]}):")
    print(f"      * Max Eigenvalue: {max_eig_full:.2e}")
    print(f"      * Min Eigenvalue: {min_eig_full:.2e}")
    print(f"      * Condition Num:  {cond_full:.2e}")
    print(f"      * Non-positive Eigencount: {neg_eigs_full} / {len(eigvals_full)}")

    # Per-Block Correlation Spectrum (to isolate which variable drives ill-conditioning)
    print(f"\n[4] Per-Variable Block Spectra (Isolating the ill-conditioned variable):")
    for i, var in enumerate(variable_names):
        start_idx = i * n_levels
        end_idx = (i + 1) * n_levels
        B_sub = B_joint[start_idx:end_idx, start_idx:end_idx]

        # Convert sub-block to correlation
        std_sub = np.sqrt(np.maximum(np.diag(B_sub), 1e-15))
        C_sub = B_sub / (std_sub[:, None] @ std_sub[None, :])

        sub_eigs = np.linalg.eigvalsh(0.5 * (C_sub + C_sub.T))
        min_e, max_e = sub_eigs[0], sub_eigs[-1]
        cond_sub = max_e / max(min_e, 1e-30)

        print(
            f"    - Block [{var}]: Min Eig = {min_e:.2e} | Condition Num = {cond_sub:.2e} | Neg Eigs = {np.sum(sub_eigs <= 0)}")
        if min_e <= 0:
            print(f"      --> CRITICAL: Variable [{var}] correlation block is rank-deficient!")

    # --- 5. Grid Sampling Ratio vs Length Scale ---
    if fitted_params is not None:
        print(f"\n[5] Grid Resolution vs Fitted Length Scale L_p:")
        log_p = np.log10(pressure_hpa) if fitted_params.get('use_log10', False) else np.log(pressure_hpa)
        min_delta_p = np.min(np.abs(np.diff(log_p)))

        for var in ['T', 'q', 'o3']:
            L_key = f'L_p_{var}'
            if L_key in fitted_params:
                L_val = fitted_params[L_key]
                ratio = min_delta_p / L_val
                print(
                    f"    - {var}: Min grid step = {min_delta_p:.4f} | L_p = {L_val:.4f} | Ratio (step/L_p) = {ratio:.4f}")
                if ratio < 0.01:
                    print(
                        f"      --> WARNING: Grid step is < 1% of L_p! Adjacent correlation approaches 0.99999999, driving rank deficiency.")

    print("=" * 70 + "\n")

    return {
        'min_eig': min_eig_full,
        'max_eig': max_eig_full,
        'condition_number': cond_full,
        'num_negative_eigs': neg_eigs_full,
        'variance_dynamic_range': overall_ratio
    }


def gasparicohn_pressure_matrix_subspace(
        data: np.ndarray,
        pressure_hpa: np.ndarray,
        univariate: bool = True,
        use_log10: bool = False,
        rcond: float = 1e-5,
        min_sigma_floor: float = 1e-4
) -> dict:
    """
    Data-calibrated Gaspari-Cohn covariance matrix generator using Low-Rank
    Subspace Factorization (Solution 2) to eliminate null grid modes.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, 3, n_levels) or (n_samples, 3 * n_levels)
        containing [T (K), log_q (-), log_O3 (-)].
    pressure_hpa : np.ndarray
        Pressure levels in hPa of length n_levels.
    univariate : bool
        If True, builds block-diagonal correlation matrix.
    use_log10 : bool
        If True, scales computations for log10 space.
    rcond : float
        Relative eigenvalue cutoff threshold. Modes with lambda_i < rcond * lambda_max
        are truncated as numerical grid noise.
    min_sigma_floor : float
        Floor for empirical standard deviations.

    Returns
    -------
    dict
        Dictionary containing 'L_k' factor (N x k), rank-k pseudo-inverse,
        eigenvectors V_k, active eigenvalues, and subspace transformation metadata.
    """
    n_levels = len(pressure_hpa)

    if data.ndim == 2:
        data = data.reshape(data.shape[0], 3, n_levels)

    T_data, q_data, O3_data = data[:, 0, :], data[:, 1, :], data[:, 2, :]

    log_scale = (1.0 / np.log(10.0)) if use_log10 else 1.0
    log_p = np.log10(pressure_hpa) if use_log10 else np.log(pressure_hpa)
    log_p_dist = np.abs(log_p[:, None] - log_p[None, :])

    # 1. Empirical Sigmas with Hard Floors
    sigma_T = np.maximum(np.std(T_data, axis=0, ddof=1), 1e-3)
    sigma_q = np.maximum(np.std(q_data, axis=0, ddof=1), min_sigma_floor * log_scale)
    sigma_O3 = np.maximum(np.std(O3_data, axis=0, ddof=1), min_sigma_floor * log_scale)

    sigma_joint = np.concatenate([sigma_T, sigma_q, sigma_O3])

    # 2. Fit Length Scales L_p
    C_TT_emp = np.corrcoef(T_data, rowvar=False)
    C_qq_emp = np.corrcoef(q_data, rowvar=False)
    C_O3O3_emp = np.corrcoef(O3_data, rowvar=False)

    def fit_length_scale(C_emp):
        def loss_fn(L_eff):
            C_fit = gaspari_cohn_kernel(log_p_dist / L_eff)
            return np.mean((C_emp - C_fit) ** 2)

        res = minimize_scalar(loss_fn, bounds=(0.05, 2.0), method='bounded')
        return res.x / log_scale

    L_p_T = fit_length_scale(C_TT_emp)
    L_p_q = fit_length_scale(C_qq_emp)
    L_p_o3 = fit_length_scale(C_O3O3_emp)

    # 3. Correlation Blocks
    C_TT = gaspari_cohn_kernel(log_p_dist / (L_p_T * log_scale))
    C_qq = gaspari_cohn_kernel(log_p_dist / (L_p_q * log_scale))
    C_O3O3 = gaspari_cohn_kernel(log_p_dist / (L_p_o3 * log_scale))

    if univariate:
        C_joint = block_diag(C_TT, C_qq, C_O3O3)
    else:
        C_Tq_emp = np.cov(T_data, q_data, rowvar=False)[:n_levels, n_levels:] / np.outer(sigma_T, sigma_q)
        C_TO3_emp = np.cov(T_data, O3_data, rowvar=False)[:n_levels, n_levels:] / np.outer(sigma_T, sigma_O3)
        C_qO3_emp = np.cov(q_data, O3_data, rowvar=False)[:n_levels, n_levels:] / np.outer(sigma_q, sigma_O3)

        rho_Tq = float(np.mean(np.diag(C_Tq_emp)))
        rho_TO3 = float(np.mean(np.diag(C_TO3_emp)))
        rho_qO3 = float(np.mean(np.diag(C_qO3_emp)))

        L_Tq = np.sqrt(((L_p_T * log_scale) ** 2 + (L_p_q * log_scale) ** 2) / 2.0)
        L_TO3 = np.sqrt(((L_p_T * log_scale) ** 2 + (L_p_o3 * log_scale) ** 2) / 2.0)
        L_qO3 = np.sqrt(((L_p_q * log_scale) ** 2 + (L_p_o3 * log_scale) ** 2) / 2.0)

        C_Tq = rho_Tq * gaspari_cohn_kernel(log_p_dist / L_Tq) if rho_Tq != 0 else np.zeros((n_levels, n_levels))
        C_TO3 = rho_TO3 * gaspari_cohn_kernel(log_p_dist / L_TO3) if rho_TO3 != 0 else np.zeros((n_levels, n_levels))
        C_qO3 = rho_qO3 * gaspari_cohn_kernel(log_p_dist / L_qO3) if rho_qO3 != 0 else np.zeros((n_levels, n_levels))

        C_joint = np.block([
            [C_TT, C_Tq, C_TO3],
            [C_Tq.T, C_qq, C_qO3],
            [C_TO3.T, C_qO3.T, C_O3O3]
        ])

    # 4. Construct Full Covariance Matrix B
    B_full = np.outer(sigma_joint, sigma_joint) * C_joint
    B_full = 0.5 * (B_full + B_full.T)

    # 5. Eigendecomposition and Subspace Truncation
    eigvals, eigvecs = np.linalg.eigh(B_full)

    # Sort eigenvalues/vectors descending
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Mask active modes above threshold
    max_eig = eigvals[0]
    cutoff = rcond * max_eig
    active_mask = eigvals > cutoff
    k = int(np.sum(active_mask))

    logger.info(
        f"Subspace Truncation: Kept {k}/{len(eigvals)} active physical modes (truncated {len(eigvals) - k} null grid modes).")

    eigvals_k = eigvals[:k]
    V_k = eigvecs[:, :k]  # Shape: (N, k)

    # 6. Compute Subspace Square-Root Factor L_k and Pseudo-Inverse B_k^+
    # L_k has shape (N, k) such that B_k = L_k @ L_k.T
    L_k = V_k * np.sqrt(eigvals_k)  # Shape: (N, k)

    # Reconstructed rank-k covariance (N x N)
    B_subspace = (V_k * eigvals_k) @ V_k.T

    # Subspace Precision Matrix B_k^+ (N x N)
    B_inv_subspace = (V_k * (1.0 / eigvals_k)) @ V_k.T

    return {
        'matrix_full': B_full,
        'matrix_subspace': B_subspace,
        'matrix_inverse': B_inv_subspace,
        'L_k': L_k,  # Rectangular square-root factor (N x k)
        'V_k': V_k,  # Subspace Eigenvectors (N x k)
        'eigenvalues_k': eigvals_k,  # Active Eigenvalues (k,)
        'n_active_modes': k,
        'n_total_modes': len(eigvals),
        'fitted_params': {
            'L_p_T': L_p_T, 'L_p_q': L_p_q, 'L_p_o3': L_p_o3,
            'sigma_T': sigma_T, 'sigma_q': sigma_q, 'sigma_O3': sigma_O3
        }
    }


class ControlVariableTransform:
    """
    Helper class implementing the v-control variable transformation (v-transform)
    for 1D-Var optimization and ensemble prior sampling.

        x = x_b + L_k @ v
        v = Lambda_k^(-1/2) @ V_k^T @ (x - x_b)
    """

    def __init__(self, L_k: np.ndarray, V_k: np.ndarray, eigenvalues_k: np.ndarray):
        """
        Parameters
        ----------
        L_k : np.ndarray
            Rectangular square-root factor of shape (N, k).
        V_k : np.ndarray
            Subspace eigenvectors of shape (N, k).
        eigenvalues_k : np.ndarray
            Active eigenvalues of shape (k,).
        """
        self.L_k = L_k  # (N, k)
        self.V_k = V_k  # (N, k)
        self.eigvals_k = eigenvalues_k  # (k,)
        self.inv_sqrt_eigvals = 1.0 / np.sqrt(eigenvalues_k)  # (k,)
        self.N, self.k = L_k.shape

    def control_to_state(self, v: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """
        Maps control vector v (k,) to state vector x (N,).

        x = x_b + L_k @ v
        """
        return x_b + self.L_k @ v

    def state_to_control(self, x: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """
        Maps state vector perturbation (x - x_b) back to control vector v (k,).

        v = diag(1 / sqrt(lambda_k)) @ V_k^T @ (x - x_b)
        """
        dx = x - x_b
        return self.inv_sqrt_eigvals * (self.V_k.T @ dx)

    def compute_J_b(self, v: np.ndarray) -> float:
        """
        Computes background cost penalty J_b in control space.

        J_b(v) = 0.5 * ||v||_2^2
        """
        return 0.5 * float(np.sum(v ** 2))

    def sample_prior_ensemble(self, x_b: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Draws n_samples physical perturbations from the exact subspace prior:

        dx ~ N(0, B_k) === L_k @ z,  z ~ N(0, I_k)
        """
        z = np.random.randn(self.k, n_samples)
        dx = self.L_k @ z  # (N, n_samples)
        return x_b[:, None] + dx


def fit_crtm_B_parameters_from_data(
        data: np.ndarray,
        pressure_hpa: np.ndarray,
        univariate: bool = True,
        use_log10: bool = False,
        rel_ridge_tol: float = 1e-6,
        min_sigma_floor: float = 1e-4
) -> dict:
    """
    Data-calibrated Gaspari-Cohn covariance matrix generator using an Adaptive
    Spectral Ridge to guarantee full-rank positive-definiteness across oversampled grids.
    """
    n_levels = len(pressure_hpa)

    # Reshape flattened input (n_samples, 3 * n_levels) -> (n_samples, 3, n_levels)
    if data.ndim == 2:
        data = data.reshape(data.shape[0], 3, n_levels)

    T_data, q_data, O3_data = data[:, 0, :], data[:, 1, :], data[:, 2, :]

    log_scale = (1.0 / np.log(10.0)) if use_log10 else 1.0
    log_p = np.log10(pressure_hpa) if use_log10 else np.log(pressure_hpa)
    log_p_dist = np.abs(log_p[:, None] - log_p[None, :])

    # 1. Compute Empirical Sigmas and Apply Hard Floors
    sigma_T = np.maximum(np.std(T_data, axis=0, ddof=1), 1e-3)
    sigma_q = np.maximum(np.std(q_data, axis=0, ddof=1), min_sigma_floor * log_scale)
    sigma_O3 = np.maximum(np.std(O3_data, axis=0, ddof=1), min_sigma_floor * log_scale)
    sigma_joint = np.concatenate([sigma_T, sigma_q, sigma_O3])

    # 2. Fit Gaspari-Cohn Length Scales L_p from Empirical Correlations
    C_TT_emp = np.corrcoef(T_data, rowvar=False)
    C_qq_emp = np.corrcoef(q_data, rowvar=False)
    C_O3O3_emp = np.corrcoef(O3_data, rowvar=False)

    def fit_length_scale(C_emp):
        def loss_fn(L_eff):
            C_fit = gaspari_cohn_kernel(log_p_dist / L_eff)
            return np.mean((C_emp - C_fit) ** 2)

        res = minimize_scalar(loss_fn, bounds=(0.05, 2.0), method='bounded')
        return res.x / log_scale

    L_p_T = fit_length_scale(C_TT_emp)
    L_p_q = fit_length_scale(C_qq_emp)
    L_p_o3 = fit_length_scale(C_O3O3_emp)

    logger.info(f"Fitted length scales (L_p): T={L_p_T:.3f}, q={L_p_q:.3f}, O3={L_p_o3:.3f}")

    # 3. Build Correlation Blocks
    C_TT = gaspari_cohn_kernel(log_p_dist / (L_p_T * log_scale))
    C_qq = gaspari_cohn_kernel(log_p_dist / (L_p_q * log_scale))
    C_O3O3 = gaspari_cohn_kernel(log_p_dist / (L_p_o3 * log_scale))

    if univariate:
        C_joint = block_diag(C_TT, C_qq, C_O3O3)
    else:
        C_Tq_emp = np.cov(T_data, q_data, rowvar=False)[:n_levels, n_levels:] / np.outer(sigma_T, sigma_q)
        C_TO3_emp = np.cov(T_data, O3_data, rowvar=False)[:n_levels, n_levels:] / np.outer(sigma_T, sigma_O3)
        C_qO3_emp = np.cov(q_data, O3_data, rowvar=False)[:n_levels, n_levels:] / np.outer(sigma_q, sigma_O3)

        rho_Tq = float(np.mean(np.diag(C_Tq_emp)))
        rho_TO3 = float(np.mean(np.diag(C_TO3_emp)))
        rho_qO3 = float(np.mean(np.diag(C_qO3_emp)))

        L_Tq = np.sqrt(((L_p_T * log_scale) ** 2 + (L_p_q * log_scale) ** 2) / 2.0)
        L_TO3 = np.sqrt(((L_p_T * log_scale) ** 2 + (L_p_o3 * log_scale) ** 2) / 2.0)
        L_qO3 = np.sqrt(((L_p_q * log_scale) ** 2 + (L_p_o3 * log_scale) ** 2) / 2.0)

        C_Tq = rho_Tq * gaspari_cohn_kernel(log_p_dist / L_Tq) if rho_Tq != 0 else np.zeros((n_levels, n_levels))
        C_TO3 = rho_TO3 * gaspari_cohn_kernel(log_p_dist / L_TO3) if rho_TO3 != 0 else np.zeros((n_levels, n_levels))
        C_qO3 = rho_qO3 * gaspari_cohn_kernel(log_p_dist / L_qO3) if rho_qO3 != 0 else np.zeros((n_levels, n_levels))

        C_joint = np.block([
            [C_TT, C_Tq, C_TO3],
            [C_Tq.T, C_qq, C_qO3],
            [C_TO3.T, C_qO3.T, C_O3O3]
        ])

    # 4. Construct Full Unregularized Covariance Matrix B
    C_joint = 0.5 * (C_joint + C_joint.T)

    # 4. Inspect Correlation Spectrum directly (No recomputation needed)
    eigvals_C = np.linalg.eigvalsh(C_joint)
    min_eig_C, max_eig_C = eigvals_C[0], eigvals_C[-1]

    # Calculate scale-free ridge shift alpha matching C_joint precision
    alpha = C_joint.dtype.type(0.0)
    if min_eig_C <= 0 or (max_eig_C / max(min_eig_C, 1e-30)) > (1.0 / rel_ridge_tol):
        alpha_val = np.abs(min_eig_C) + (rel_ridge_tol * max_eig_C)
        alpha = C_joint.dtype.type(alpha_val)
        logger.info(f"Applying correlation-space ridge: alpha = {float(alpha):.4e}")

    # 5. Construct Regularized Covariance Matrix B
    # Apply (1 + alpha) scaling directly to diagonal entries during assembly
    C_joint_regularized = C_joint + alpha * np.eye(C_joint.shape[0], dtype=C_joint.dtype)

    sigma_joint = np.concatenate([sigma_T, sigma_q, sigma_O3])
    B = np.outer(sigma_joint, sigma_joint) * C_joint_regularized
    B = 0.5 * (B + B.T)

    # 6. Compute Cholesky Factor and Inverse
    L = np.linalg.cholesky(B)
    B_inv = np.linalg.inv(B)

    return {
        'matrix': B,
        'matrix_cholesky': L,
        'matrix_inverse': B_inv
    }


def gaspari_cohn_kernel(r: np.ndarray) -> np.ndarray:
    """ Computes the Gaspari-Cohn 5th-order compactly supported correlation function.

        Parameters
        ----------
        r : np.ndarray. Input distances.

        Returns
        -------
        np.ndarray. Gaspari-Cohn correlation values for the input distances.
    """
    r = np.abs(r)
    C = np.zeros_like(r)

    m1 = r <= 1.0
    x = r[m1]
    C[m1] = (
        1
        - (5/3)*x**2
        + (5/8)*x**3
        + 0.5*x**4
        - 0.25*x**5
    )

    m2 = (r > 1.0) & (r <= 2.0)
    x = r[m2]
    C[m2] = (
        4
        - 5*x
        + (5/3)*x**2
        + (5/8)*x**3
        - 0.5*x**4
        + (1/12)*x**5
        - 2/(3*x)
    )

    return C


def generate_crtm_clear_sky_B(
        pressure_hpa: np.ndarray,
        L_p_T: float = 0.35,
        L_p_q: float = 0.25,
        L_p_o3: float = 0.45,
        rho_Tq: float = -0.25,
        rho_TO3: float = 0.15,
        rho_qO3: float = 0.05,
        univariate: bool = False,
        use_log10: bool = False,
        jitter_factor: float = 1e-8
) -> dict:
    """
    Generates an analytical joint B matrix for clear-sky CRTM state vectors:
    x = [T (K), log_q (-), log_O3 (-)] on input pressure levels.

    Parameters
    ----------
    pressure_hpa : np.ndarray
        Array of pressure level boundaries/centers in hPa.
    L_p_T, L_p_q, L_p_o3 : float
        Vertical decorrelation length scales (in natural log-pressure scale heights).
    rho_Tq, rho_TO3, rho_qO3 : float
        Cross-variable correlation parameters between T, log_q, and log_O3.
    univariate : bool
        If True, forces zero cross-correlations (block-diagonal matrix).
        If False, constructs a fully coupled multivariate matrix using rho parameters.
    use_log10 : bool
        If True, converts pressure coordinates, profile sigmas, and length scales
        to base-10 log10 space while preserving physical correlation shapes.
    jitter_factor : float
        Proportional diagonal loading for numerical stability.
    """
    n_levels = len(pressure_hpa)

    # Force zero cross-correlations if univariate mode is requested
    if univariate:
        rho_Tq = 0.0
        rho_TO3 = 0.0
        rho_qO3 = 0.0

    log_scale = (1.0 / np.log(10.0)) if use_log10 else 1.0
    log_p = np.log10(pressure_hpa) if use_log10 else np.log(pressure_hpa)

    # --- 1. Check Cross-Correlation Matrix R_cross ---
    R_cross = np.array([
        [1.0, rho_Tq, rho_TO3],
        [rho_Tq, 1.0, rho_qO3],
        [rho_TO3, rho_qO3, 1.0]
    ])
    min_eig_R = np.min(np.linalg.eigvalsh(R_cross))
    if min_eig_R <= 0:
        raise ValueError(
            f"Specified cross-correlations yield a non-positive definite R_cross (min eig = {min_eig_R:.4f}). "
            "Adjust rho values."
        )

    # --- 2. Standard Deviations sigma(p) with Floors ---
    log_p_1000 = np.log10(1000.0) if use_log10 else np.log(1000.0)
    log_p_300 = np.log10(300.0) if use_log10 else np.log(300.0)
    log_p_100 = np.log10(100.0) if use_log10 else np.log(100.0)

    sigma_T = (1.0
               + 0.8 * np.exp(-((log_p - log_p_1000) / (1.2 * log_scale)) ** 2)
               + 1.0 * np.exp(-((log_p - log_p_100) / (0.8 * log_scale)) ** 2))
    sigma_T = np.maximum(sigma_T, 1e-3)

    sigma_log_q = (0.20 / (1.0 + np.exp(-(log_p - log_p_100) / (0.5 * log_scale)))) * log_scale
    sigma_log_q = np.maximum(sigma_log_q, 1e-4 * log_scale)

    sigma_log_o3 = (0.15 + 0.10 / (1.0 + np.exp((log_p - log_p_300) / (0.5 * log_scale)))) * log_scale
    sigma_log_o3 = np.maximum(sigma_log_o3, 1e-4 * log_scale)

    # --- 3. Compute Composite Cross-Variable Length Scales ---
    L_T_eff = L_p_T * log_scale
    L_q_eff = L_p_q * log_scale
    L_O3_eff = L_p_o3 * log_scale

    L_Tq_eff = np.sqrt((L_T_eff ** 2 + L_q_eff ** 2) / 2.0)
    L_TO3_eff = np.sqrt((L_T_eff ** 2 + L_O3_eff ** 2) / 2.0)
    L_qO3_eff = np.sqrt((L_q_eff ** 2 + L_O3_eff ** 2) / 2.0)

    # --- 4. Build Correlation Blocks ---
    log_p_dist = np.abs(log_p[:, None] - log_p[None, :])

    C_TT = gaspari_cohn_kernel(log_p_dist / L_T_eff)
    C_qq = gaspari_cohn_kernel(log_p_dist / L_q_eff)
    C_O3O3 = gaspari_cohn_kernel(log_p_dist / L_O3_eff)

    C_Tq = rho_Tq * gaspari_cohn_kernel(log_p_dist / L_Tq_eff) if rho_Tq != 0.0 else np.zeros((n_levels, n_levels))
    C_TO3 = rho_TO3 * gaspari_cohn_kernel(log_p_dist / L_TO3_eff) if rho_TO3 != 0.0 else np.zeros((n_levels, n_levels))
    C_qO3 = rho_qO3 * gaspari_cohn_kernel(log_p_dist / L_qO3_eff) if rho_qO3 != 0.0 else np.zeros((n_levels, n_levels))

    # --- 5. Assemble Covariance Blocks ---
    B_TT = np.outer(sigma_T, sigma_T) * C_TT
    B_qq = np.outer(sigma_log_q, sigma_log_q) * C_qq
    B_O3O3 = np.outer(sigma_log_o3, sigma_log_o3) * C_O3O3

    B_Tq = np.outer(sigma_T, sigma_log_q) * C_Tq
    B_TO3 = np.outer(sigma_T, sigma_log_o3) * C_TO3
    B_qO3 = np.outer(sigma_log_q, sigma_log_o3) * C_qO3

    # Assemble full joint matrix
    B_joint = np.block([
        [B_TT, B_Tq, B_TO3],
        [B_Tq.T, B_qq, B_qO3],
        [B_TO3.T, B_qO3.T, B_O3O3]
    ])

    # --- 6. Regularization & Cholesky/Inversion ---
    B_joint = 0.5 * (B_joint + B_joint.T)
    if jitter_factor > 0:
        B_joint += jitter_factor * np.diag(np.diag(B_joint))

    try:
        L_joint = np.linalg.cholesky(B_joint)
    except np.linalg.LinAlgError:
        print("Cholesky decomposition failed; applying eigenvalue regularization...")
        eigvals, eigvecs = np.linalg.eigh(B_joint)
        eigvals = np.maximum(eigvals, 1e-12)
        B_joint = (eigvecs * eigvals) @ eigvecs.T
        B_joint = 0.5 * (B_joint + B_joint.T)
        L_joint = eigvecs * np.sqrt(eigvals)

    B_inv_joint = np.linalg.inv(B_joint)

    return {
        'matrix': B_joint,
        'matrix_cholesky': L_joint,
        'matrix_inverse': B_inv_joint,
        'sigma_profiles': {'T': sigma_T, 'log_q': sigma_log_q, 'log_O3': sigma_log_o3}
    }


def gaspari_pressure_matrix(input: DictConfig, output: DictConfig, plot_flag: bool = True, univariate: bool = True, **kwargs) -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        plot_flag: bool. If True, plot the covariance matrix.
        **kwargs: Additional keyword arguments.

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    pressure = load_variable(input.pressure)
    cov = generate_crtm_clear_sky_B(pressure, use_log10=True, univariate=univariate)
    std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
    cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
    cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry

    # Loop over keys
    for key in list(cov.keys()):
        # Save to file
        if hasattr(output, key):
            if hasattr(output[key], 'save'):
                logger.info(f"Saving {key} matrix...")
                save_func = instantiate(output[key].save)
                save_func(cov[key])
            # Plot covariance matrix if requested
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'correlation']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


def gasparicohn_pressure_matrix(input: DictConfig, output: DictConfig, plot_flag: bool = True, univariate: bool = True, **kwargs) -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        plot_flag: bool. If True, plot the covariance matrix.
        **kwargs: Additional keyword arguments.

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    pressure = load_variable(input.pressure)
    data = load_variable(input.data)
    cov = fit_crtm_B_parameters_from_data(data, pressure, use_log10=True, univariate=univariate)
    std = np.sqrt(np.maximum(np.diag(cov['matrix']), 1e-12))
    cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
    cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry

    # Loop over keys
    for key in list(cov.keys()):
        # Save to file
        if hasattr(output, key):
            if hasattr(output[key], 'save'):
                logger.info(f"Saving {key} matrix...")
                save_func = instantiate(output[key].save)
                save_func(cov[key])
            # Plot covariance matrix if requested
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'correlation']:
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