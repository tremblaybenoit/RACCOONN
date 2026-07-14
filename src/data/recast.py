import os
import numpy as np
import hydra
from omegaconf import DictConfig
from scipy.spatial import cKDTree
from tqdm import tqdm
import gc
import logging
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from src.data.filters import cloud_mask, daytime_mask


logger = logging.getLogger(__name__)


def reshuffle_spatiotemporal(data: dict, mask: np.ndarray, ratio: dict | None = None,
                             seed: int = 42, static: bool = False) -> None:
    """Reshuffle data into train/valid/test splits using spatial boundaries.

    Useful for static scenarios (single timestep) to create meaningful train/valid/test
    splits based on spatial location (convex hull boundaries) rather than just timesteps.

    Parameters
    ----------

    """
    logger.info(f"Reshuffling data spatiotemporal splits...")

    # Default split
    if ratio is None:
        ratio = {'train': 0.6, 'valid': 0.2, 'test': 0.2}
    # Validate split ratios
    total = sum(ratio.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Load coordinate data
    lat = data['lat'][data['spatiotemporal_mask']]
    lon = data['lon'][data['spatiotemporal_mask']]
    # Identify neighboring points
    coords = np.column_stack((lat, lon))
    tree = cKDTree(coords)
    dist, neigh = tree.query(coords, k=5)  # 5 neighbors + itself
    neigh = neigh[:, 1:]  # Remove self-neighbor
    # Perform edge_detection
    lat_tolerance = 0.75
    lon_tolerance = 0.75
    # TODO: Test
    has_edge = (
            (lat >= lat.max() - lat_tolerance) |
            (lat <= lat.min() + lat_tolerance) |
             (lon >= lon.max() - lon_tolerance) |
             (lon <= lon.min() + lon_tolerance)
    )
    # Load cloud/clear sky mask
    cloud_mask = data['cloud_mask'][data['spatiotemporal_mask']]
    clear_mask = ~cloud_mask
    has_cloud_neighbor = cloud_mask[neigh].any(axis=1)
    # Identify domain/group edges
    # TODO: Update to properly handle both cloud or clear cases
    coords_hull = np.flatnonzero(data['spatiotemporal_mask'])[clear_mask & (has_cloud_neighbor | has_edge)]
    coords_remaining = np.flatnonzero(data['spatiotemporal_mask'])[clear_mask & ~(has_cloud_neighbor | has_edge)]

    # Load scans
    scans = data['scans'][mask]
    n_scans = len(np.unique(scans))
    n_coords = mask.sum()

    # Stage coordinates: Shuffle and assign to set
    coords_stage = {'Train2': [], 'Val2': [], 'Test2': []}
    for scan in range(n_scans):
        # Shuffle remaining indices
        np.random.shuffle(coords_remaining)
        # Assign hull coordinates to training set
        coords_stage['Train2'].extend(coords_hull + scan * n_coords)
        coords_stage['Train2'].extend(coords_remaining[0:int(ratio['train'] * len(coords_remaining))])
        # Assign the rest to the validation and test sets
        coords_stage['Val2'].extend(coords_remaining[int(ratio['train'] * len(coords_remaining)):
                                                int((ratio['train'] + ratio['valid']) * len(coords_remaining))])
        coords_stage['Test2'].extend(coords_remaining[int((ratio['train'] + ratio['valid']) * len(coords_remaining)):])
    # Convert to arrays
    coords_stage = {k: np.array(v, dtype='int64') for k, v in coords_stage.items()}

    # Log split sizes
    for split_name, indices in coords_stage.items():
        logger.info(f"  {split_name}: {len(indices)} samples ({100*len(indices)/n_samples:.1f}%)")

    # Get all filenames from input
    filenames = os.listdir(input_path)

    # Save splits
    for split_name, indices in coords_stage.items():
        split_output_dir = os.path.join(output_path, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        for filename in filenames:
            input_file = os.path.join(input_path, filename)
            if not os.path.isfile(input_file):
                continue

            # Load data
            data = np.load(input_file)

            # Apply split
            split_data = data[indices]

            # Save
            output_file = os.path.join(split_output_dir, filename)
            np.save(output_file, split_data)

        logger.info(f"  Saved {split_name} split to {split_output_dir}")

    logger.info(f"✅ Reshuffling complete!")



def synthetic(input: DictConfig, output: DictConfig) -> None:
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
    for key, value in input.data.items():
        # Load data
        data[key] = instantiate(value.load)

    # Build mask
    mask = np.ones_like(data['lat'], dtype='bool')
    if hasattr(input, 'mask'):
        # Initialize masks
        data['spatiotemporal_mask'] = np.ones_like(data['lat'], dtype='bool')
        # Spatial extent
        if hasattr(input.mask, 'spatial_domain'):
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
            if hasattr(input.mask.temporal_window, 'scans_min'):
                scans_min = input.mask.temporal_window.get('scans_min', data['scans'].min())
                data['spatiotemporal_mask'] &= (data['scans'] >= scans_min).astype(bool)
            if hasattr(input.mask.temporal_window, 'scans_max'):
                scans_max = input.mask.temporal_window.get('scans_max', data['scans'].max())
                data['spatiotemporal_mask'] &= (data['scans'] <= scans_max).astype(bool)
        # Update mask
        mask &= data['spatiotemporal_mask']
        # Clouds or clear sky
        data['cloud_mask'] = cloud_mask(data['prof'])
        data['clear_mask'] = ~data['cloud_mask']
        if hasattr(input.mask, 'cloud_mask'):
            if input.mask.cloud_mask is True:
                mask &= data['cloud_mask']
        if hasattr(input.mask, 'clear_mask'):
            if input.mask.clear_mask is True:
                mask &= data['clear_mask']
        # Daytime or nighttime
        data['daytime_mask'] = daytime_mask(data['meta'])
        data['nighttime_mask'] = ~data['daytime_mask']
        if hasattr(input.mask, 'daytime_mask'):
            if input.mask.daytime_mask is True:
                mask &= data['daytime_mask']
        if hasattr(input.mask, 'nighttime_mask'):
            if input.mask.nighttime_mask is True:
                mask &= data['nighttime_mask']

    # Shuffle or maintain distribution
    if hasattr(input, 'split') and input.split is not None:
        logger.info("\n  Reshuffling into train/valid/test splits...")

        # Parameters
        ratio = input.split.get('ratio', {'train': 0.6, 'valid': 0.2, 'test': 0.2})
        seed = input.split.get('seed', 0)
        static = input.split.get('static', False)
        # Update distribution
        reshuffle_spatiotemporal(
            data, mask, ratio=ratio, seed=seed, static=static
        )
    else:
        # Apply mask and convert to right precision
        data = {key: value[mask].astype(output.dtype) for key, value in data.items()}
        # Save to disk
        for key, value in output.data.items():
            # Save function
            save_fn = instantiate(value.save)
            save_fn(data[key])

    return

def recast_data(config: DictConfig) -> None:
    """Main recasting pipeline."""



    # Unpack configuration
    recast_cfg = config.preparation.recast if hasattr(config, 'preparation') else config.recast
    input_cfg = recast_cfg.input
    output_cfg = recast_cfg.output
    filters_cfg = recast_cfg.filters
    precision = recast_cfg.precision
    spatial_bounds = recast_cfg.get('spatial_bounds', None)
    temporal_bounds = recast_cfg.get('temporal_bounds', None)
    cloud_filter_type = recast_cfg.cloud_filter_type
    reshuffle_cfg = recast_cfg.get('reshuffle', None)

    logger.info(f"\nInput dir: {input_cfg.path}")
    logger.info(f"Output dir: {output_cfg.path}")
    logger.info(f"Output precision: {precision}")

    # Create output directory
    os.makedirs(output_cfg.path, exist_ok=True)

    # ===== LOAD CORE DATA =====
    logger.info("\n1️⃣  Loading core data...")

    # Load profiles
    prof_path = os.path.join(input_cfg.path, input_cfg.files.prof)
    prof = load_file(prof_path, dtype=precision)
    logger.info(f"   Profiles shape: {prof.shape}")
    n_samples = prof.shape[0]

    # Load coordinates
    lat_path = os.path.join(input_cfg.path, input_cfg.files.lat)
    lat = load_file(lat_path, dtype=precision)

    lon_path = os.path.join(input_cfg.path, input_cfg.files.lon)
    lon = load_file(lon_path, dtype=precision)

    logger.info(f"   Latitude shape: {lat.shape}, Longitude shape: {lon.shape}")

    # Load scans (time/scan indices)
    scans_path = os.path.join(input_cfg.path, input_cfg.files.scans)
    scans = load_file(scans_path, dtype='int32')
    logger.info(f"   Scans shape: {scans.shape}, unique scans: {np.unique(scans).size}")

    # Load pressure levels
    pressure_path = os.path.join(input_cfg.path, input_cfg.files.pressure)
    pressure = load_file(pressure_path, dtype=precision)
    logger.info(f"   Pressure shape: {pressure.shape}")

    # Load hofx (optional)
    hofx = None
    if hasattr(input_cfg.files, 'hofx') and input_cfg.files.hofx:
        hofx_path = os.path.join(input_cfg.path, input_cfg.files.hofx)
        if os.path.exists(hofx_path):
            hofx = load_file(hofx_path, dtype=precision)
            logger.info(f"   HOFX shape: {hofx.shape}")

    # ===== CREATE FILTERS =====
    logger.info("\n2️⃣  Creating filters...")

    # Cloud/clear filter
    cloud_mask = None
    if cloud_filter_type == 'auto':
        cloud_mask = create_cloud_filter(prof)
        n_cloud = cloud_mask.sum()
        n_clear = (~cloud_mask).sum()
        logger.info(f"   Cloud/clear: {n_cloud} cloud, {n_clear} clear")

    # Spatial filter
    spatial_mask = np.ones(n_samples, dtype=bool)
    if spatial_bounds is not None:
        lat_min = spatial_bounds.get('lat_min', -90)
        lat_max = spatial_bounds.get('lat_max', 90)
        lon_min = spatial_bounds.get('lon_min', -180)
        lon_max = spatial_bounds.get('lon_max', 180)
        spatial_mask &= (lat >= lat_min) & (lat <= lat_max)
        spatial_mask &= (lon >= lon_min) & (lon <= lon_max)
        n_spatial = spatial_mask.sum()
        logger.info(f"   Spatial bounds: {n_spatial} samples in region")

    # Temporal filter
    temporal_mask = np.ones(n_samples, dtype=bool)
    if temporal_bounds is not None:
        scan_min = temporal_bounds.get('scan_min', scans.min())
        scan_max = temporal_bounds.get('scan_max', scans.max())
        temporal_mask &= (scans >= scan_min) & (scans <= scan_max)
        n_temporal = temporal_mask.sum()
        logger.info(f"   Temporal bounds: {n_temporal} samples in range")

    # ===== APPLY FILTERS =====
    logger.info("\n3️⃣  Applying filters...")

    only_cloud = filters_cfg.get('only_cloud', False)
    only_clear = filters_cfg.get('only_clear', False)

    if only_cloud and only_clear:
        raise ValueError("Cannot filter to both cloud and clear simultaneously")

    # Apply filters to all data
    prof = apply_filters(prof, cloud_mask, spatial_mask, temporal_mask, only_cloud, only_clear)
    lat = apply_filters(lat.reshape(-1, 1), cloud_mask, spatial_mask, temporal_mask,
                       only_cloud, only_clear).flatten()
    lon = apply_filters(lon.reshape(-1, 1), cloud_mask, spatial_mask, temporal_mask,
                       only_cloud, only_clear).flatten()
    scans = apply_filters(scans.reshape(-1, 1), cloud_mask, spatial_mask, temporal_mask,
                         only_cloud, only_clear).flatten()

    if hofx is not None:
        hofx = apply_filters(hofx, cloud_mask, spatial_mask, temporal_mask,
                            only_cloud, only_clear)

    n_final = prof.shape[0]
    logger.info(f"   Final sample count: {n_final} ({100*n_final/n_samples:.1f}% of original)")

    # ===== COMPUTE DERIVED QUANTITIES =====
    logger.info("\n4️⃣  Computing derived quantities...")

    # Cloud filter (for output)
    output_cloud_mask = None
    if filters_cfg.get('save_cloud_mask', False) and cloud_mask is not None:
        # Re-apply filtering
        output_cloud_mask = apply_filters(cloud_mask.reshape(-1, 1), cloud_mask, spatial_mask,
                                         temporal_mask, only_cloud, only_clear).flatten()
        logger.info(f"   Cloud mask saved ({output_cloud_mask.sum()} clouds)")

    # Pressure mask (constant levels)
    pressure_mask = None
    if filters_cfg.get('compute_pressure_mask', False):
        # For now, assume pressure is per-level (not per-sample)
        # This would typically be done from statistics
        logger.info(f"   Pressure mask: skipped (requires statistics)")

    # ===== SAVE DATA =====
    logger.info("\n5️⃣  Saving recasted data...")

    # Output file format
    output_filetype = output_cfg.get('filetype', 'npy')

    # Save profiles
    prof_out = os.path.join(output_cfg.path, output_cfg.files.get('prof', 'prof.npy'))
    save_file(prof.astype(precision), prof_out, filetype=output_filetype)

    # Save coordinates
    lat_out = os.path.join(output_cfg.path, output_cfg.files.get('lat', 'lat.npy'))
    save_file(lat.astype(precision), lat_out, filetype=output_filetype)

    lon_out = os.path.join(output_cfg.path, output_cfg.files.get('lon', 'lon.npy'))
    save_file(lon.astype(precision), lon_out, filetype=output_filetype)

    # Save scans
    scans_out = os.path.join(output_cfg.path, output_cfg.files.get('scans', 'scans.npy'))
    save_file(scans.astype('int32'), scans_out, filetype='npy')  # scans always int

    # Save pressure
    pressure_out = os.path.join(output_cfg.path, output_cfg.files.get('pressure', 'pressure.npy'))
    save_file(pressure.astype(precision), pressure_out, filetype=output_filetype)

    # Save HOFX
    if hofx is not None:
        hofx_out = os.path.join(output_cfg.path, output_cfg.files.get('hofx', 'hofx.npy'))
        save_file(hofx.astype(precision), hofx_out, filetype=output_filetype)

    # Save cloud mask
    if output_cloud_mask is not None:
        mask_out = os.path.join(output_cfg.path, 'cloud_mask.npy')
        save_file(output_cloud_mask.astype('bool'), mask_out, filetype='npy')

    logger.info(f"\n✅ Recasting complete!")
    logger.info(f"   Input samples: {n_samples}")
    logger.info(f"   Output samples: {n_final}")
    logger.info(f"   Output dir: {output_cfg.path}")
    logger.info("=" * 70)

    # ===== OPTIONAL: RESHUFFLE DATA INTO TRAIN/VALID/TEST =====
    if reshuffle_cfg is not None and reshuffle_cfg.get('enabled', False):
        logger.info("\n6️⃣  Reshuffling into train/valid/test splits...")

        split_ratios = reshuffle_cfg.get('split_ratios', {'train': 0.6, 'valid': 0.2, 'test': 0.2})
        single_scan = reshuffle_cfg.get('single_scan', None)
        reshuffle_output = reshuffle_cfg.get('output_path', os.path.join(output_cfg.path, 'splits'))

        reshuffle_spatiotemporal(
            input_path=output_cfg.path,
            output_path=reshuffle_output,
            split_ratios=split_ratios,
            single_scan=single_scan
        )


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:

    # Execute recast
    if hasattr(config.preparation, 'recast'):
        for key, config in config.preparation.recast.items():
            logger.info(f"Performing recast: {key}")
            instantiate(config)


if __name__ == '__main__':
    main()

