import os
import numpy as np
import hydra
from omegaconf import DictConfig
from scipy.spatial import cKDTree
import logging
from utilities.logic import get_config_path
from utilities.instantiators import instantiate


logger = logging.getLogger(__name__)


def reshuffle_synthetic(data: dict,
                        ratio: dict | None = None,
                        seed: int = 42,
                        tolerance: float = 0.75,
                        n_neighbors: int = 9,
                        dist_neighbors: float = 2.,
                        ) -> dict:
    """ Reshuffle already-masked data into train/valid/test splits using spatial boundaries.

        Useful for static scenarios (single timestep) to create meaningful train/valid/test
        splits based on spatial location (convex hull boundaries) rather than just timesteps.

        Parameters
        ----------
        data: dict. Dictionary containing already-masked data to split
        ratio: dict | None. Dictionary specifying the train/valid/test split ratios
        seed: int. Random seed for reproducibility
        tolerance: float. Buffer for latitude/longitude.
        n_neighbors: int. Number of nearest neighbors to use.
        dist_neighbors: float. Maximum distance to consider neighbor points.

        Returns
        -------
        dict. Dictionary containing the indices for each stage (train, valid, test).
    """
    logger.info(f"Reshuffling data splits...")

    # Initialize independent random number generator
    rng = np.random.default_rng(seed)

    # Default split
    if ratio is None:
        ratio = {'train': 0.6, 'valid': 0.2, 'test': 0.2}
    # Validate split ratios
    total = sum(ratio.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Get unique scan values
    scans = data['scans']
    scans_unique = np.unique(scans)
    n_total_samples = len(scans)

    # Stage coordinates: Shuffle and assign to set
    coords_stage = {'train': [], 'valid': [], 'test': []}

    # Loop over each unique scan
    for scan in scans_unique:
        # Create mask for this specific scan
        scan_mask = (scans == scan)

        # Load coordinate data for this scan
        lat = data['lat'][scan_mask]
        lon = data['lon'][scan_mask]

        # Identify neighboring points
        coords = np.column_stack((lat, lon))
        tree = cKDTree(coords)
        dist, neigh = tree.query(coords, k=n_neighbors)  # n_neighbors (including itself)
        neigh = neigh[:, 1:]  # Remove self-neighbor

        # Only keep points with neighbors within minimum range
        has_neighbor = (dist[:, 1] > 0) & (dist[:, 1] <= dist_neighbors)

        # Perform edge_detection for this scan's domain
        has_edge = (
                (lat >= lat.max() - tolerance) |
                (lat <= lat.min() + tolerance) |
                (lon >= lon.max() - tolerance) |
                (lon <= lon.min() + tolerance)
        )

        # Identify hull and remaining points for this scan
        coords_scan = np.flatnonzero(scan_mask)
        # Hull: boundary points or isolated points (kept in training)
        coords_hull = coords_scan[has_edge | ~has_neighbor]
        # Remaining: interior points with neighbors (split by ratios)
        coords_remaining = coords_scan[~has_edge & has_neighbor]

        # Shuffle remaining indices for this scan
        rng.shuffle(coords_remaining)

        # Assign hull coordinates to training set
        coords_stage['train'].extend(coords_hull)

        # Split remaining for this scan according to ratios
        n_remaining = len(coords_remaining)
        split_train = int(ratio['train'] * n_remaining)
        split_valid = int((ratio['train'] + ratio['valid']) * n_remaining)

        coords_stage['train'].extend(coords_remaining[:split_train])
        coords_stage['valid'].extend(coords_remaining[split_train:split_valid])
        coords_stage['test'].extend(coords_remaining[split_valid:])

    # Convert to arrays
    coords_stage = {k: np.array(v, dtype='int64') for k, v in coords_stage.items()}

    # Log split sizes
    for split_name, indices in coords_stage.items():
        logger.info(f"  {split_name}: {len(indices)} samples ({100*len(indices)/n_total_samples:.1f}%)")

    # Return split
    return coords_stage


def recast_synthetic(input: DictConfig, output: DictConfig) -> None:
    """ Perform recast, cropping, redistribution of data.

        Parameters
        ----------
        input: DictConfig. Input configuration.
        output: DictConfig. Output configuration.

        Returns
        -------
        None.
    """

    # Build dictionary from input data
    data_all = {}
    for stage, config_stage in input.stage.items():
        logger.info(f"Loading stage '{stage}'")
        data_stage = {}

        # Build mask
        mask = None
        cloud_keep = True
        clear_keep = True
        if hasattr(input, 'mask'):

            # Spatial extent
            if hasattr(input.mask, 'spatial_domain'):
                # Spatiotemporal mask
                logger.info(f"Spatial domain mask...")
                # Load coordinates
                if 'lat' not in data_stage:
                    data_stage['lat'] = instantiate(config_stage.variables['lat'].load)
                if 'lon' not in data_stage:
                    data_stage['lon'] = instantiate(config_stage.variables['lon'].load)
                # Apply mask
                if mask is None:
                    mask = np.ones_like(data_stage['lat'], dtype=bool)
                # Apply spatial bounds (use defaults if not specified)
                lat_min = input.mask.spatial_domain.get('lat_min', -90)
                mask &= (data_stage['lat'] >= lat_min)
                lat_max = input.mask.spatial_domain.get('lat_max', 90)
                mask &= (data_stage['lat'] <= lat_max)
                lon_min = input.mask.spatial_domain.get('lon_min', -180)
                mask &= (data_stage['lon'] >= lon_min)
                lon_max = input.mask.spatial_domain.get('lon_max', 180)
                mask &= (data_stage['lon'] <= lon_max)
            # Temporal extent
            if hasattr(input.mask, 'temporal_window'):
                logger.info(f"Temporal window mask...")
                # Load coordinates
                if 'scans' not in data_stage:
                    data_stage['scans'] = instantiate(config_stage.variables['scans'].load)
                # Apply mask
                if mask is None:
                    mask = np.ones_like(data_stage['scans'], dtype=bool)
                # Apply temporal bounds (use data min/max if not specified)
                scans_min = input.mask.temporal_window.get('scans_min', data_stage['scans'].min())
                mask &= (data_stage['scans'] >= scans_min)
                scans_max = input.mask.temporal_window.get('scans_max', data_stage['scans'].max())
                mask &= (data_stage['scans'] <= scans_max)

            # Clouds or clear sky
            cloud_keep = input.mask.get('cloud_mask', True)
            clear_keep = input.mask.get('clear_mask', True)
            # If we keep one
            if cloud_keep != clear_keep:
                cloud_mask = instantiate(config_stage.variables['cloud_mask'].load)
                # Clouds only
                if cloud_keep and not clear_keep:
                    logger.info(f"Cloud mask...")
                    if mask is None:
                        mask = cloud_mask
                    else:
                        mask &= cloud_mask
                # Clear sky only
                elif clear_keep and not cloud_keep:
                    logger.info(f"Clear mask...")
                    if mask is None:
                        mask = ~cloud_mask
                    else:
                        mask &= ~cloud_mask
                del cloud_mask

            # Daytime or nighttime
            daytime_keep = input.mask.get('daytime_mask', True)
            nighttime_keep = input.mask.get('nighttime_mask', True)
            # If we keep one
            if daytime_keep != nighttime_keep:
                daytime_mask = instantiate(config_stage.variables['daytime_mask'].load)
                # Daytime only
                if daytime_keep and not nighttime_keep:
                    logger.info(f"Daytime mask...")
                    if mask is None:
                        mask = daytime_mask
                    else:
                        mask &= daytime_mask
                # Nighttime only
                elif nighttime_keep and not daytime_keep:
                    logger.info(f"Nighttime mask...")
                    if mask is None:
                        mask = ~daytime_mask
                    else:
                        mask &= ~daytime_mask
                del daytime_mask

        # Shuffle or maintain distribution
        if hasattr(input, 'split') and input.split is not None:
            # Store data
            logger.info(f"Storing stage: {stage}")
            for variable in output.stage[stage].variables.keys():
                # Check that the variable is available
                if variable not in data_stage:
                    logger.info(f"Loading variable '{variable}'")
                    data_stage[variable] = instantiate(config_stage.variables[variable].load)
                # Apply mask
                if variable not in ('pressure', 'variant_mask', 'invariant_mask') and mask is not None:
                    data_stage[variable] = data_stage[variable][mask]
                # If atmospheric profiles and clear sky, extract non-zero profiles
                if variable == 'prof' and data_stage[variable].shape[1] > 3 and (clear_keep and not cloud_keep):
                    data_stage[variable] = np.take(data_stage[variable], [0, 4, 8], axis=1)
                # Convert dtype if needed
                if hasattr(output, 'dtype'):
                    data_stage[variable] = data_stage[variable].astype(output.dtype)
                # Check if variable is in data_all
                if variable not in data_all or variable in ('pressure', 'variant_mask', 'invariant_mask'):
                    data_all[variable] = data_stage[variable]
                else:
                    # Append along first dimension
                    data_all[variable] = np.concatenate([data_all[variable], data_stage[variable]], axis=0)
                # Free memory
                del data_stage[variable]
        else:
            # Apply mask and convert to right precision and save
            logger.info(f"Saving stage: {stage}")
            for variable, config_variable in output.stage[stage].variables.items():
                # Check that the variable is available
                if variable not in data_stage:
                    logger.info(f"Loading variable '{variable}'")
                    data_stage[variable] = instantiate(config_stage.variables[variable].load)
                # Save function
                save_fn = instantiate(config_variable.save)
                # Create directory if needed
                if hasattr(config_variable, 'path'):
                    os.makedirs(os.path.dirname(config_variable.path), exist_ok=True)
                # Pressure is constant, skip masking, otherwise mask
                if variable not in ('pressure', 'variant_mask', 'invariant_mask') and mask is not None:
                    data_stage[variable] = data_stage[variable][mask]
                # If atmospheric profiles and clear sky, extract non-zero profiles
                if variable == 'prof' and data_stage[variable].shape[1] > 3 and (clear_keep and not cloud_keep):
                    data_stage[variable] = np.take(data_stage[variable], [0, 4, 8], axis=1)
                # Convert dtype if needed
                if hasattr(output, 'dtype'):
                     data_stage[variable] = data_stage[variable].astype(output.dtype)
                # Save
                save_fn(data_stage[variable])
                logger.info(f"  Saved {variable}: shape={data_stage[variable].shape}")
                # Free memory
                del data_stage[variable]

    # Shuffle distribution
    if hasattr(input, 'split') and input.split is not None:
        # Shuffle distribution
        logger.info("Combining all loaded stages and redistributing...")
        # Reshuffle and split into stages
        ratio = input.split.get('ratio', {'train': 0.6, 'valid': 0.2, 'test': 0.2})
        seed = input.split.get('seed', 0)
        tolerance = input.split.get('tolerance', 0.75)
        split_coords = reshuffle_synthetic(data_all, ratio=ratio, seed=seed, tolerance=tolerance)

        # Save each split stage directly from data['all'] using indices
        for stage, coords in split_coords.items():
            if stage in output.stage:
                logger.info(f"Saving stage: {stage}")
                for variable, config_variable in output.stage[stage].variables.items():
                    logger.info(f"  Saving variable: {variable}")
                    if variable in data_all:
                        # Create parent directory
                        if hasattr(config_variable, 'path'):
                            os.makedirs(os.path.dirname(config_variable.path), exist_ok=True)
                        # Save function
                        save_fn = instantiate(config_variable.save)
                        # Pressure is constant, skip masking, otherwise mask
                        if variable not in ('pressure', 'variant_mask', 'invariant_mask'):
                            save_fn(data_all[variable][coords])
                        else:
                            save_fn(data_all[variable])
                        logger.debug(f"  Saved {variable}: shape={data_all[variable].shape}")

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Recast dataset.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Execute recast
    if hasattr(config.preprocessing, 'recast'):
        # If single operation, execute
        if hasattr(config.preprocessing.recast, "_target_"):
            logger.info(f"Performing recast...")
            instantiate(config.preprocessing.recast)
        # Execute individual operations
        else:
            for key, config in config.preprocessing.recast.items():
                logger.info(f"Performing recast: {key}")
                instantiate(config)


if __name__ == '__main__':
    """ Recast dataset.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Dataset in new format, and/or filtered for clouds or clear skies.
    """

    main()

