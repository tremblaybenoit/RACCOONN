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
    x_prior_physical = np.clip(x_prior_physical, x_min_physical, x_max_physical)

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
                group_counts = np.bincount(inverse_indices)[:, None]  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

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
    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <=2:
            raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
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

            # Apply per-variable regularization to covariance block
            if regularization_factor > 0:
                var_mean_diag = float(np.mean(np.diag(sub_cov)))
                logger.info(f"Applying per-variable regularization for variable {i} (mean_diag={var_mean_diag:.6e})...")
                sub_cov = sub_cov + regularization_factor * var_mean_diag * np.eye(sub_cov.shape[0])

            m_cov.append(sub_cov)

        # Assemble into block-diagonal matrices
        del data_i, data
        cov['matrix'] = block_diag(*m_cov)
        if hasattr(output, 'correlation'):
            cov['correlation'] = block_diag(*m_corr)

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

        # Apply per-variable regularization to covariance matrix
        if regularization_factor > 0:
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
                cov['matrix'][start_idx:end_idx, start_idx:end_idx] += regularization_factor * var_mean_diag * np.eye(features_per_var)
                start_idx = end_idx

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
        rcond = output.matrix_pseudo_inverse.get('params.rcond', 1.e-3)
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
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse']:
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

    Two background-generation methods are supported:

    1. ``persistence``:

       .. math::

           x_b(s_i) = x(s_i - \\Delta s)

       and the corresponding background error is

       .. math::

           e(s_i) = x(s_i - \\Delta s) - x(s_i).

    2. ``centered_average``:

       .. math::

           x_b(s_i)
           =
           \\frac{1}{2}
           \\left[
               x(s_i - \\Delta s)
               +
               x(s_i + \\Delta s)
           \\right]

       and the corresponding background error is

       .. math::

           e(s_i)
           =
           \\frac{1}{2}
           \\left[
               x(s_i - \\Delta s)
               +
               x(s_i + \\Delta s)
           \\right]
           -
           x(s_i).

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

    # Validate background-generation method
    supported_methods = (
        "persistence",
        "centered_average",
    )

    if method not in supported_methods:
        raise ValueError(
            f"Unsupported persistence method '{method}'. "
            f"Supported methods are {supported_methods}."
        )

    # Validate missing-timestep policy
    supported_missing_timestep_policies = (
        "reject",
        "available",
        "raise",
    )

    if (
        missing_timestep_policy
        not in supported_missing_timestep_policies
    ):
        raise ValueError(
            f"Unsupported missing_timestep_policy "
            f"'{missing_timestep_policy}'. Supported policies are "
            f"{supported_missing_timestep_policies}."
        )

    # Begin by loading the data
    logger.info("Loading atmospheric profiles...")
    data = load_variable(
        input.data,
        apply_transform=apply_transform,
    )

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
    if (
        hasattr(input, "spatial_mask")
        and input.spatial_mask is not None
    ):
        logger.info("Applying spatial mask...")
        mask &= np.asarray(
            instantiate(input.spatial_mask),
            dtype=bool,
        )

    # Temporal mask: Consider only data within the specified scan bounds
    if (
        hasattr(input, "temporal_mask")
        and input.temporal_mask is not None
    ):
        logger.info("Applying temporal mask...")
        mask &= np.asarray(
            instantiate(input.temporal_mask),
            dtype=bool,
        )

    # Apply masks to data, coordinates, and scans
    data = data[mask]
    lat = lat[mask]
    lon = lon[mask]
    scans = scans[mask]

    # Dimensions
    if data.ndim < 3:
        raise ValueError(
            "persistent_matrix expects profile data with shape "
            "(n_samples, n_variables, n_levels), or an equivalent "
            "shape with at least three dimensions."
        )

    data_shape = data.shape
    n_samples = data_shape[0]
    n_vars = data_shape[1]

    if n_samples < 2:
        raise ValueError(
            "At least two profile samples are required."
        )

    # Optional coordinate rounding
    #
    # np.unique requires coordinates representing the same physical point
    # to have exactly equal values. Rounding can be enabled through
    # input.coordinate_decimals if coordinates contain small floating-point
    # differences.
    coordinate_decimals = input.get(
        "coordinate_decimals",
        None,
    )

    if coordinate_decimals is not None:
        logger.info(
            "Rounding latitude and longitude to %d decimal places...",
            coordinate_decimals,
        )

        lat_group = np.round(
            lat,
            decimals=coordinate_decimals,
        )

        lon_group = np.round(
            lon,
            decimals=coordinate_decimals,
        )

    else:
        lat_group = lat
        lon_group = lon

    # Identify unique horizontal coordinates and their mapping
    #
    # unique_coords contains the physical locations.
    # inverse_indices contains the location ID for every sample.
    coords = np.column_stack(
        (lat_group, lon_group)
    )

    unique_coords, inverse_indices = np.unique(
        coords,
        axis=0,
        return_inverse=True,
    )

    n_unique_coords = len(unique_coords)

    logger.info(
        "Found %d unique horizontal coordinates.",
        n_unique_coords,
    )

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
        location_sample_indices = np.flatnonzero(
            inverse_indices == location_id
        )

        # Sort samples at this coordinate by scan index
        location_scans = scans[location_sample_indices]

        order = np.argsort(
            location_scans,
            kind="stable",
        )

        sorted_indices = location_sample_indices[order]
        sorted_scans = location_scans[order]

        # Duplicate scans at the same coordinate make temporal pairing
        # ambiguous.
        if sorted_scans.shape[0] > 1:
            duplicate_scan_mask = (
                sorted_scans[1:] == sorted_scans[:-1]
            )

            if np.any(duplicate_scan_mask):
                duplicate_scans = sorted_scans[1:][
                    duplicate_scan_mask
                ]

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
            for scan_index in range(
                1,
                sorted_indices.shape[0],
            ):
                n_candidate_targets += 1

                previous_index = sorted_indices[
                    scan_index - 1
                ]

                target_index = sorted_indices[
                    scan_index
                ]

                previous_scan = sorted_scans[
                    scan_index - 1
                ]

                target_scan = sorted_scans[
                    scan_index
                ]

                previous_scan_step = int(
                    target_scan - previous_scan
                )

                exact_previous_available = (
                    previous_scan_step
                    == expected_scan_step
                )

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
                previous_scan_steps.append(
                    previous_scan_step
                )

                n_accepted += 1

        # Centered averaging requires one previous and one next profile
        elif method == "centered_average":

            if sorted_indices.shape[0] < 3:
                continue

            # The first and last samples cannot be candidate targets.
            for scan_index in range(
                1,
                sorted_indices.shape[0] - 1,
            ):
                n_candidate_targets += 1

                previous_index = sorted_indices[
                    scan_index - 1
                ]

                target_index = sorted_indices[
                    scan_index
                ]

                next_index = sorted_indices[
                    scan_index + 1
                ]

                previous_scan = sorted_scans[
                    scan_index - 1
                ]

                target_scan = sorted_scans[
                    scan_index
                ]

                next_scan = sorted_scans[
                    scan_index + 1
                ]

                previous_scan_step = int(
                    target_scan - previous_scan
                )

                next_scan_step = int(
                    next_scan - target_scan
                )

                exact_previous_available = (
                    previous_scan_step
                    == expected_scan_step
                )

                exact_next_available = (
                    next_scan_step
                    == expected_scan_step
                )

                exact_neighbors_available = (
                    exact_previous_available
                    and exact_next_available
                )

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
                x_background = 0.5 * (
                    x_previous + x_next
                )

                # Background error:
                # e(s_i) = x_b(s_i) - x_true(s_i)
                x_error = x_background - x_true

                error_samples.append(x_error)
                previous_indices.append(previous_index)
                target_indices.append(target_index)
                next_indices.append(next_index)
                previous_scan_steps.append(
                    previous_scan_step
                )
                next_scan_steps.append(
                    next_scan_step
                )

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
    error_samples = np.stack(
        error_samples,
        axis=0,
    )

    n_error_samples = error_samples.shape[0]

    # Convert diagnostics to arrays
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    previous_indices = np.asarray(
        previous_indices,
        dtype=np.int64,
    )

    previous_scan_steps = np.asarray(
        previous_scan_steps,
        dtype=np.int64,
    )

    if method == "centered_average":
        next_indices = np.asarray(
            next_indices,
            dtype=np.int64,
        )

        next_scan_steps = np.asarray(
            next_scan_steps,
            dtype=np.int64,
        )

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
    error_mean = np.mean(
        error_samples,
        axis=0,
        keepdims=True,
    )

    # Apply recentering
    if recenter:
        logger.info(
            "Recentering background errors around their mean..."
        )

        error_samples = (
            error_samples - error_mean
        )

        # One mean background-error profile was estimated.
        denom = float(n_error_samples - 1)

    else:
        logger.info(
            "Computing second moments without error recentering..."
        )

        denom = float(n_error_samples)

    if denom <= 0.0:
        raise ValueError(
            "Insufficient background-error samples to estimate "
            "the covariance matrix."
        )

    # Restore variable and profile-level dimensions
    error_samples = error_samples.reshape(
        n_error_samples,
        *data_shape[1:],
    )

    # Variant filter
    if (
        hasattr(input, "variant_mask")
        and input.variant_mask is not None
    ):
        variant_mask = instantiate(
            input.variant_mask.load
        )

        variant_mask = np.asarray(
            variant_mask
        )

        if variant_mask.ndim == 1:
            # Apply one shared level mask to every variable
            variant_mask = np.repeat(
                variant_mask[None, :],
                n_vars,
                axis=0,
            )

        elif (
            variant_mask.ndim == 2
            and variant_mask.shape[0] == 1
            and n_vars > 1
        ):
            variant_mask = np.repeat(
                variant_mask,
                n_vars,
                axis=0,
            )

        elif (
            variant_mask.ndim != 2
            or variant_mask.shape[0] != n_vars
        ):
            raise ValueError(
                "variant_mask must have shape "
                "(n_variables, n_levels), (1, n_levels), "
                "or (n_levels,). "
                f"Received shape {variant_mask.shape}."
            )

    else:
        variant_mask = None

    # Covariance matrix initialization
    cov = {}

    # Store diagnostics that may optionally be saved through output
    cov["error_mean"] = error_mean.reshape(
        data_shape[1:]
    )

    cov["n_error_samples"] = np.asarray(
        n_error_samples,
        dtype=np.int64,
    )

    cov["samples_per_coordinate"] = (
        samples_per_coordinate
    )

    cov["previous_scan_steps"] = (
        previous_scan_steps
    )

    cov["target_indices"] = target_indices
    cov["previous_indices"] = previous_indices

    if method == "centered_average":
        cov["next_scan_steps"] = (
            next_scan_steps
        )

        cov["next_indices"] = next_indices

    # Univariate matrix computation steps
    if univariate:

        # Check dimensions
        if error_samples.ndim <= 2:
            raise ValueError(
                "Univariate covariance matrix computation requires "
                "background errors with more than two dimensions."
            )

        # Initialize empty lists for matrices
        m_cov = []
        m_corr = []

        # Loop over variables
        for i in range(n_vars):

            # Select all pressure-level errors for this variable
            error_i = error_samples[:, i]

            # Flatten any dimensions after the variable dimension
            error_i = error_i.reshape(
                n_error_samples,
                -1,
            )

            # Apply variant mask
            if variant_mask is not None:
                logger.info(
                    "Applying variant mask for variable %d...",
                    i,
                )

                mask_i = np.flatnonzero(
                    variant_mask[i]
                )

                error_i = np.take(
                    error_i,
                    mask_i,
                    axis=1,
                )

            # Compute univariate covariance block
            logger.info(
                "Computing univariate persistence covariance "
                "matrix for variable %d...",
                i,
            )

            sub_cov = (
                scaling_factor
                * (error_i.T @ error_i)
                / denom
            )

            # Ensure numerical symmetry
            sub_cov = 0.5 * (
                sub_cov + sub_cov.T
            )

            # Compute pressure-dependent variances
            variances_i = np.diag(
                sub_cov
            ).copy()

            if np.any(~np.isfinite(variances_i)):
                raise ValueError(
                    f"Variable {i} contains non-finite variances."
                )

            if np.any(variances_i <= 0.0):
                bad_indices = np.flatnonzero(
                    variances_i <= 0.0
                )

                raise ValueError(
                    f"Variable {i} has non-positive variances "
                    f"at indices {bad_indices.tolist()}."
                )

            std_i = np.sqrt(
                variances_i
            )

            std_outer_i = np.outer(
                std_i,
                std_i,
            )

            # Compute correlation block from unregularized covariance
            sub_corr = sub_cov / std_outer_i

            # Ensure numerical symmetry and exact unit diagonal
            sub_corr = 0.5 * (
                sub_corr + sub_corr.T
            )

            np.fill_diagonal(
                sub_corr,
                1.0,
            )

            # Apply per-variable regularization through the
            # correlation matrix
            if regularization_factor > 0.0:
                logger.info(
                    "Applying correlation regularization for "
                    "variable %d with factor %.6g...",
                    i,
                    regularization_factor,
                )

                sub_corr = (
                    (1.0 - regularization_factor)
                    * sub_corr
                    + regularization_factor
                    * np.eye(
                        sub_corr.shape[0],
                        dtype=sub_corr.dtype,
                    )
                )

                # Defensive numerical cleanup
                sub_corr = 0.5 * (
                    sub_corr + sub_corr.T
                )

                np.fill_diagonal(
                    sub_corr,
                    1.0,
                )

                # Reconstruct covariance while preserving the original
                # pressure-dependent variances
                sub_cov = (
                    std_outer_i * sub_corr
                )

                sub_cov = 0.5 * (
                    sub_cov + sub_cov.T
                )

            m_cov.append(
                sub_cov
            )

            m_corr.append(
                sub_corr
            )

        # Assemble into block-diagonal matrices
        cov["matrix"] = block_diag(
            *m_cov
        )

        cov["correlation"] = block_diag(
            *m_corr
        )

        del error_i
        del error_samples

    # Multivariate matrix computation steps
    else:

        # Reshape to preserve the variable dimension
        error_reshaped = error_samples.reshape(
            n_error_samples,
            n_vars,
            -1,
        )

        # Apply variant mask before flattening
        if variant_mask is not None:
            logger.info("Applying pressure mask...")

            error_masked_list = []

            for i in range(n_vars):

                error_i = error_reshaped[
                    :,
                    i,
                    :,
                ]

                mask_i = np.flatnonzero(
                    variant_mask[i]
                )

                error_i_masked = np.take(
                    error_i,
                    mask_i,
                    axis=1,
                )

                error_masked_list.append(
                    error_i_masked
                )

                logger.info(
                    "Variable %d features after masking: %d",
                    i,
                    error_i_masked.shape[1],
                )

            # Flatten concatenated background errors
            error_samples_flat = np.concatenate(
                error_masked_list,
                axis=1,
            )

            del error_i
            del error_i_masked
            del error_masked_list

        else:
            # No masking: retain all variables and pressure levels
            error_samples_flat = error_reshaped.reshape(
                n_error_samples,
                -1,
            )

        # Compute full covariance matrix
        logger.info(
            "Computing multivariate persistence "
            "covariance matrix..."
        )

        cov["matrix"] = (
            scaling_factor
            * (
                error_samples_flat.T
                @ error_samples_flat
            )
            / denom
        )

        # Ensure numerical symmetry
        cov["matrix"] = 0.5 * (
            cov["matrix"]
            + cov["matrix"].T
        )

        # Compute component variances
        variances = np.diag(
            cov["matrix"]
        ).copy()

        if np.any(~np.isfinite(variances)):
            raise ValueError(
                "Covariance matrix contains non-finite variances."
            )

        if np.any(variances <= 0.0):
            bad_indices = np.flatnonzero(
                variances <= 0.0
            )

            raise ValueError(
                "Covariance matrix has non-positive variances "
                f"at indices {bad_indices.tolist()}."
            )

        std = np.sqrt(
            variances
        )

        std_outer = np.outer(
            std,
            std,
        )

        # Compute correlation matrix from covariance matrix
        logger.info(
            "Computing correlation matrix from covariance matrix..."
        )

        cov["correlation"] = (
            cov["matrix"] / std_outer
        )

        # Ensure numerical symmetry and exact unit diagonal
        cov["correlation"] = 0.5 * (
            cov["correlation"]
            + cov["correlation"].T
        )

        np.fill_diagonal(
            cov["correlation"],
            1.0,
        )

        # Apply regularization through the correlation matrix
        if regularization_factor > 0.0:
            logger.info(
                "Applying correlation regularization "
                "with factor %.6g...",
                regularization_factor,
            )

            cov["correlation"] = (
                (1.0 - regularization_factor)
                * cov["correlation"]
                + regularization_factor
                * np.eye(
                    cov["correlation"].shape[0],
                    dtype=cov["correlation"].dtype,
                )
            )

            # Defensive numerical cleanup
            cov["correlation"] = 0.5 * (
                cov["correlation"]
                + cov["correlation"].T
            )

            np.fill_diagonal(
                cov["correlation"],
                1.0,
            )

            # Reconstruct covariance while preserving the original
            # component variances
            cov["matrix"] = (
                std_outer
                * cov["correlation"]
            )

            cov["matrix"] = 0.5 * (
                cov["matrix"]
                + cov["matrix"].T
            )

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
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


def climatological_matrix0(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
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
                group_counts = np.bincount(inverse_indices)[:, None]  # Shape: (n_unique_coords, 1)

                # Compute the local temporal mean for each unique coordinate
                group_means = group_sums / group_counts  # Shape: (n_unique_coords, n_features)

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
            m.append(scaling_factor * (data_i.T @ data_i) / denom)
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
        cov = {'matrix': scaling_factor * (data.T @ data) / denom}
        breakpoint()
        # Release memory
    del data

    # Apply regularization
    if regularization_factor > 0:
        logger.info("Applying regularization factor...")
        cov['matrix'] += regularization_factor * float(np.mean(np.diag(cov['matrix']))) * np.eye(cov['matrix'].shape[0])
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
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
                                     'correlation', 'correlation_inverse', 'correlation_pseudo_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return



def climatological_matrix00(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0,
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
    denom = float(n_samples - 1)

    # Apply recentering
    mu = np.mean(data, axis=0)
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
            m.append(scaling_factor * (data_i.T @ data_i) / denom)
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
        cov = {'matrix': scaling_factor * (data.T @ data) / denom}
        # Release memory
    del data

    # Compute correlation matrix
    cov['matrix'] = 0.5 * (cov['matrix'] + cov['matrix'].T)  # Ensure symmetry (preserve dtype)
    std = np.sqrt(np.diag(cov['matrix']))
    cov['correlation'] = cov['matrix'] / (std[:, None] @ std[None, :])
    cov['correlation'] = 0.5 * (cov['correlation'] + cov['correlation'].T)  # Ensure symmetry (preserve dtype)
    # np.fill_diagonal(cov['correlation'], 1.0)
    # Apply regularization via correlation matrix
    if regularization_factor > 0:
        logger.info("Applying regularization factor...")
        # Regularize correlation matrix (preserve dtype by matching np.eye to correlation dtype)
        cov['correlation'] = (1. - regularization_factor) * cov['correlation'] + regularization_factor * np.eye(cov['correlation'].shape[0], dtype=cov['correlation'].dtype)
        # Reconstruct covariance matrix from regularized correlation matrix
        cov['matrix'] = (std[:, None] @ std[None, :]) * cov['correlation']
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
            if plot_flag and key in ['matrix', 'matrix_cholesky', 'matrix_inverse', 'matrix_pseudo_inverse',
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