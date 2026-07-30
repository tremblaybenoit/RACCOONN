import os

import numpy as np
import hydra
from omegaconf import DictConfig
from scipy.spatial import cKDTree
import logging
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from code.preprocessing.filters import cloud_mask, daytime_mask, pressure_mask


logger = logging.getLogger(__name__)


def reshuffle_synthetic(data: dict,
                        mask: np.ndarray,
                        ratio: dict | None = None,
                        seed: int = 42,
                        tolerance: float = 0.75,
                        ) -> dict:
    """ Reshuffle data into train/valid/test splits using spatial boundaries.

        Useful for static scenarios (single timestep) to create meaningful train/valid/test
        splits based on spatial location (convex hull boundaries) rather than just timesteps.

        Parameters
        ----------
        data: dict. Dictionary containing the data to split
        mask: np.ndarray. Boolean mask indicating which samples to consider
        ratio: dict | None. Dictionary specifying the train/valid/test split ratios
        seed: int. Random seed for reproducibility
        tolerance: float. Buffer for latitude/longitude.

        Returns
        -------
        dict. Dictionary containing the coordinates for each stage.
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

    # Get unique scan values that exist in the filtered mask
    scans = data['scans'][mask]
    scans_unique = np.unique(scans)
    n_total_samples = mask.sum()

    # Stage coordinates: Shuffle and assign to set
    coords_stage = {'train': [], 'valid': [], 'test': []}

    # Loop over each unique scan
    for scan in scans_unique:
        # Create mask for this specific scan (using already-filtered data)
        scan_mask = mask & (data['scans'] == scan)

        # Load coordinate data for this scan
        lat = data['lat'][scan_mask]
        lon = data['lon'][scan_mask]

        # Identify neighboring points
        coords = np.column_stack((lat, lon))
        tree = cKDTree(coords)
        dist, neigh = tree.query(coords, k=5)  # 5 neighbors + itself
        neigh = neigh[:, 1:]  # Remove self-neighbor

        # Check if points have neighbors (dist to nearest neighbor > 0)
        has_neighbor = dist[:, 1] > 0  # Distance to nearest neighbor (excluding self)

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

    # Variables to recast (only those with save configuration)
    variables = [key for key, value in output.variables.items() if hasattr(value, 'save')]
    logger.info(f"Variables to save: {variables}")

    # Build dictionary from input data
    data = {}
    for variable in variables:
        # Load data
        if variable in input.variables:
            logger.debug(f"Loading variable '{variable}'")
            data[variable] = instantiate(input.variables[variable].load)
        else:
            logger.error(f"Variable '{variable}' not found in input.variables")

    # Build mask
    mask = np.ones(data[variables[0]].shape[0], dtype='bool')
    if hasattr(input, 'mask'):
        # Initialize masks
        data['spatiotemporal_mask'] = mask.copy()
        # Spatial extent
        if hasattr(input.mask, 'spatial_domain'):
            # Load coordinates
            if 'lat' not in data:
                data['lat'] = instantiate(input.variables['lat'].load)
            if 'lon' not in data:
                data['lon'] = instantiate(input.variables['lon'].load)
            # Apply mask
            if hasattr(input.mask.spatial_domain, 'lat_min'):
                lat_min = input.mask.spatial_domain.get('lat_min', -90)
                data['spatiotemporal_mask'] &= (data['lat'] >= lat_min).astype(bool)
            if hasattr(input.mask.spatial_domain, 'lat_max'):
                lat_max = input.mask.spatial_domain.get('lat_max', 90)
                data['spatiotemporal_mask'] &= (data['lat'] <= lat_max).astype(bool)
            if hasattr(input.mask.spatial_domain, 'lon_min'):
                lon_min = input.mask.spatial_domain.get('lon_min', -180)
                data['spatiotemporal_mask'] &= (data['lon'] >= lon_min).astype(bool)
            if hasattr(input.mask.spatial_domain, 'lon_max'):
                lon_max = input.mask.spatial_domain.get('lon_max', 180)
                data['spatiotemporal_mask'] &= (data['lon'] <= lon_max).astype(bool)
        # Temporal extent
        if hasattr(input.mask, 'temporal_window'):
            # Load coordinates
            if 'scans' not in data:
                data['scans'] = instantiate(input.variables['scans'].load)
            if hasattr(input.mask.temporal_window, 'scans_min'):
                scans_min = input.mask.temporal_window.get('scans_min', data['scans'].min())
                data['spatiotemporal_mask'] &= (data['scans'] >= scans_min).astype(bool)
            if hasattr(input.mask.temporal_window, 'scans_max'):
                scans_max = input.mask.temporal_window.get('scans_max', data['scans'].max())
                data['spatiotemporal_mask'] &= (data['scans'] <= scans_max).astype(bool)
        # Update mask
        mask &= data['spatiotemporal_mask']

        # Clouds or clear sky
        cloud_keep = input.mask.get('cloud_mask', True)
        clear_keep = input.mask.get('clear_mask', True)
        # If there is a mask
        if cloud_keep != clear_keep:
            # Clouds or clear sky masks
            if 'prof' not in data:
                data['prof'] = instantiate(input.variables['prof'].load)
            if 'cloud_mask' not in data:
                data['cloud_mask'] = cloud_mask(data['prof'])
            data['clear_mask'] = ~data['cloud_mask']
            # Clouds only
            if cloud_keep and not clear_keep:
                mask &= data['cloud_mask']
            # Clear sky only
            elif clear_keep and not cloud_keep:
                mask &= data['clear_mask']
                # Remove null profiles
                data['prof'] = np.take(data['prof'], [0, 4, 8], axis=1)

        # Daytime or nighttime
        daytime_keep = input.mask.get('daytime_mask', True)
        nighttime_keep = input.mask.get('nighttime_mask', True)
        # If there is a mask
        if daytime_keep != nighttime_keep:
            # Daytime or nighttime masks
            if 'meta' not in data:
                data['meta'] = instantiate(input.variables['meta'].load)
            if 'daytime_mask' not in data:
                data['daytime_mask'] = daytime_mask(data['meta'])
            data['nighttime_mask'] = ~data['daytime_mask']
            # Daytime only
            if daytime_keep and not nighttime_keep:
                mask &= data['daytime_mask']
            # Nighttime only
            elif nighttime_keep and not daytime_keep:
                mask &= data['nighttime_mask']

    # Shuffle or maintain distribution
    if hasattr(input, 'split') and input.split is not None:
        logger.info("\n  Reshuffling into train/valid/test splits...")

        # Parameters
        ratio = input.split.get('ratio', {'train': 0.6, 'valid': 0.2, 'test': 0.2})
        seed = input.split.get('seed', 0)
        tolerance = input.split.get('tolerance', 0.75)
        # Update distribution
        split = reshuffle_synthetic(data, mask, ratio=ratio, seed=seed, tolerance=tolerance)
        # Save to disk, acoording to new split
        for stage, coords in split.items():
            # Loop over variables per stage
            for variable in variables:
                logger.info(f"Saving variable '{variable}'")
                # Create parent directory if needed
                if hasattr(output.stage[stage].variables[variable], 'path'):
                    os.makedirs(os.path.dirname(output.stage[stage].variables[variable].path), exist_ok=True)
                # Save function
                save_fn = instantiate(output.stage[stage].variables[variable].save)
                # Save data
                if hasattr(output, 'dtype'):
                    save_fn(data[variable][coords].astype(output.dtype))
                else:
                    save_fn(data[variable][coords])
    else:
        # Apply mask and convert to right precision (pressure is constant, skip masking)
        if hasattr(output, 'dtype'):
            data = {key: (value[mask] if key not in ('pressure', 'pressure_mask') else value).astype(output.dtype)
                    for key, value in data.items()}
        else:
            data = {key: (value[mask] if key not in ('pressure', 'pressure_mask') else value) for key, value in data.items()}
        # Save to disk
        for variable in variables:
            logger.info(f"Saving variable '{variable}'")
            # Create parent directory if needed
            if hasattr(output.variables[variable], 'path'):
                os.makedirs(os.path.dirname(output.variables[variable].path), exist_ok=True)
            # Save function
            save_fn = instantiate(output.variables[variable].save)
            if variable in data:
                save_fn(data[variable])
            else:
                logger.warning(f"Variable '{variable}' not found in data dict, skipping save")

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

