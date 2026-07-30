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


def _combine_stages(data: dict, stages_to_combine: list) -> dict:
    """ Combine multiple stages into data['all'] while freeing source stages.

        Parameters
        ----------
        data: dict. Dictionary containing loaded stage data (modified in-place)
        stages_to_combine: list. List of stage names to combine (e.g., ['train', 'valid', 'test'])

        Returns
        -------
        dict. Combined dataset accessible as data['all']
    """
    # Build combined dataset by concatenating variables
    combined = {}
    for var_name in next(iter(data[stages_to_combine[0]].values() if stages_to_combine[0] in data else data.values())).keys():
        var_arrays = []
        for stage_name in stages_to_combine:
            if stage_name in data and var_name in data[stage_name]:
                var_arrays.append(data[stage_name][var_name])

        if var_arrays:
            combined[var_name] = np.concatenate(var_arrays, axis=0)

    # Store combined data
    data['all'] = combined

    # Free original stage data to save memory
    for stage_name in stages_to_combine:
        if stage_name in data:
            del data[stage_name]
            logger.debug(f"Freed stage '{stage_name}' from memory")

    logger.info(f"Combined {len(stages_to_combine)} stages into data['all'] with {len(combined)} variables")
    return combined


def reshuffle_synthetic(data: dict,
                        ratio: dict | None = None,
                        seed: int = 42,
                        tolerance: float = 0.75,
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

    # Build dictionary from input data
    data = {}
    for stage, input_stage in input.stage.items():
        logger.info(f"Loading stage '{stage}'")
        data[stage] = {}
        for variable, input_variable in input_stage.variables.items():
            logger.info(f"Loading variable '{variable}'")
            data[stage][variable] = instantiate(input_variable.load)

        # Build mask
        mask = np.ones(next(iter(data[stage].values())).shape[0], dtype='bool')
        if hasattr(input, 'mask'):

            # Spatiotemporal mask
            data[stage]['spatiotemporal_mask'] = mask.copy()

            # Spatial extent
            if hasattr(input.mask, 'spatial_domain'):
                logger.info(f"Spatial domain mask...")
                # Load coordinates
                if 'lat' not in data[stage]:
                    data[stage]['lat'] = instantiate(input_stage.variables['lat'].load)
                if 'lon' not in data[stage]:
                    data[stage]['lon'] = instantiate(input_stage.variables['lon'].load)
                # Apply mask
                if hasattr(input.mask.spatial_domain, 'lat_min'):
                    lat_min = input.mask.spatial_domain.get('lat_min', -90)
                    data[stage]['spatiotemporal_mask'] &= (data[stage]['lat'] >= lat_min).astype(bool)
                if hasattr(input.mask.spatial_domain, 'lat_max'):
                    lat_max = input.mask.spatial_domain.get('lat_max', 90)
                    data[stage]['spatiotemporal_mask'] &= (data[stage]['lat'] <= lat_max).astype(bool)
                if hasattr(input.mask.spatial_domain, 'lon_min'):
                    lon_min = input.mask.spatial_domain.get('lon_min', -180)
                    data[stage]['spatiotemporal_mask'] &= (data[stage]['lon'] >= lon_min).astype(bool)
                if hasattr(input.mask.spatial_domain, 'lon_max'):
                    lon_max = input.mask.spatial_domain.get('lon_max', 180)
                    data[stage]['spatiotemporal_mask'] &= (data[stage]['lon'] <= lon_max).astype(bool)
            # Temporal extent
            if hasattr(input.mask, 'temporal_window'):
                logger.info(f"Temporal window mask...")
                # Load coordinates
                if 'scans' not in data[stage]:
                    data[stage]['scans'] = instantiate(input_stage.variables['scans'].load)
                if hasattr(input.mask.temporal_window, 'scans_min'):
                    scans_min = input.mask.temporal_window.get('scans_min', data[stage]['scans'].min())
                    data[stage]['spatiotemporal_mask'] &= (data[stage]['scans'] >= scans_min).astype(bool)
                if hasattr(input.mask.temporal_window, 'scans_max'):
                    scans_max = input.mask.temporal_window.get('scans_max', data[stage]['scans'].max())
                    data[stage]['spatiotemporal_mask'] &= (data[stage]['scans'] <= scans_max).astype(bool)
            # Update mask
            logger.info(f"Masking variables...")
            mask &= data[stage]['spatiotemporal_mask']

            # Clouds or clear sky
            cloud_keep = input.mask.get('cloud_mask', True)
            clear_keep = input.mask.get('clear_mask', True)
            # If there is a mask
            if cloud_keep != clear_keep:
                # Clouds or clear sky masks
                if 'prof' not in data[stage]:
                    data[stage]['prof'] = instantiate(input_stage.variables['prof'].load)
                if 'cloud_mask' not in data[stage]:
                    data[stage]['cloud_mask'] = cloud_mask(data[stage]['prof'])
                data[stage]['clear_mask'] = ~data[stage]['cloud_mask']
                # Clouds only
                if cloud_keep and not clear_keep:
                    logger.info(f"Cloud mask...")
                    mask &= data[stage]['cloud_mask']
                # Clear sky only
                elif clear_keep and not cloud_keep:
                    logger.info(f"Clear mask...")
                    mask &= data[stage]['clear_mask']
                    # Remove null profiles
                    data[stage]['prof'] = np.take(data[stage]['prof'], [0, 4, 8], axis=1)

            # Daytime or nighttime
            daytime_keep = input.mask.get('daytime_mask', True)
            nighttime_keep = input.mask.get('nighttime_mask', True)
            # If there is a mask
            if daytime_keep != nighttime_keep:
                # Daytime or nighttime masks
                if 'meta' not in data[stage]:
                    data[stage]['meta'] = instantiate(input_stage.variables['meta'].load)
                if 'daytime_mask' not in data[stage]:
                    data[stage]['daytime_mask'] = daytime_mask(data[stage]['meta'])
                data[stage]['nighttime_mask'] = ~data[stage]['daytime_mask']
                # Daytime only
                if daytime_keep and not nighttime_keep:
                    logger.info(f"Daytime mask...")
                    mask &= data[stage]['daytime_mask']
                # Nighttime only
                elif nighttime_keep and not daytime_keep:
                    logger.info(f"Nighttime mask...")
                    mask &= data[stage]['nighttime_mask']

            # Apply mask and convert to right precision (pressure is constant, skip masking)
            for key in list(data[stage].keys()):
                if key not in ('pressure', 'pressure_mask'):
                    data[stage][key] = data[stage][key][mask]
                if hasattr(output, 'dtype'):
                    data[stage][key] = data[stage][key].astype(output.dtype)

            # TODO: Improve efficiency

    # Shuffle or maintain distribution
    if hasattr(input, 'split') and input.split is not None:
        logger.info("Combining all loaded stages and redistributing...")

        # Combine all loaded stages into data['all'], freeing original stages
        if 'all' in data:
            # If 'all' was explicitly loaded, just use it
            logger.info("Using pre-combined 'all' stage")
        else:
            # Combine all available stages into data['all'] and free originals
            stages_to_combine = list(data.keys())
            _combine_stages(data, stages_to_combine)
            logger.info(f"Combined stages {stages_to_combine} into data['all'] (freed {len(stages_to_combine)} stages)")

        # Reshuffle and split into stages
        ratio = input.split.get('ratio', {'train': 0.6, 'valid': 0.2, 'test': 0.2})
        seed = input.split.get('seed', 0)
        tolerance = input.split.get('tolerance', 0.75)
        split_coords = reshuffle_synthetic(data['all'], ratio=ratio, seed=seed, tolerance=tolerance)

        # Save each split stage directly from data['all'] using indices
        for stage_name, coords in split_coords.items():
            if stage_name in output.stage:
                logger.info(f"Saving stage: {stage_name}")
                for var_name in output.stage[stage_name].variables.keys():
                    if var_name in data['all']:
                        output_config = output.stage[stage_name].variables[var_name]
                        # Create parent directory
                        if hasattr(output_config, 'path'):
                            os.makedirs(os.path.dirname(output_config.path), exist_ok=True)
                        # Save function
                        save_fn = instantiate(output_config.save)
                        save_fn(data['all'][var_name][coords])
                        logger.debug(f"  Saved {var_name}: shape={data['all'][var_name][coords].shape}")

    else:
        # Save each stage independently (no splitting)
        for stage_name in data.keys():
            if stage_name in output.stage:
                logger.info(f"Saving stage: {stage_name}")
                for var_name in output.stage[stage_name].variables.keys():
                    if var_name in data[stage_name]:
                        output_config = output.stage[stage_name].variables[var_name]
                        # Create parent directory
                        if hasattr(output_config, 'path'):
                            os.makedirs(os.path.dirname(output_config.path), exist_ok=True)
                        # Save function
                        save_fn = instantiate(output_config.save)
                        save_fn(data[stage_name][var_name])
                        logger.debug(f"  Saved {var_name}: shape={data[stage_name][var_name].shape}")

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

