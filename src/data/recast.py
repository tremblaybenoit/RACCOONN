import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import ConvexHull
from tqdm import tqdm
import gc
import logging
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from src.data.filters import clear_mask, cloud_mask


logger = logging.getLogger(__name__)


def load_file(path: str, dtype: str = 'float32', filetype: str | None = None):
    """Load file with auto-detection of format (npy/txt) if not specified."""
    if filetype is None:
        # Auto-detect from extension
        if path.endswith('.npy'):
            filetype = 'npy'
        elif path.endswith(('.txt', '.csv')):
            filetype = 'txt'
        else:
            raise ValueError(f"Cannot auto-detect format for {path}")

    if filetype == 'npy':
        data = np.load(path)
    elif filetype == 'txt':
        data = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    # Convert to desired dtype
    if data.dtype != dtype:
        data = data.astype(dtype)

    return data


def save_file(data: np.ndarray, path: str, filetype: str = 'npy', **kwargs):
    """Save file in specified format."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if filetype == 'npy':
        np.save(path, data)
    elif filetype == 'txt':
        np.savetxt(path, data, **kwargs)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    logger.info(f"Saved {path}")


def apply_filters(data: np.ndarray,
                  cloud_mask: np.ndarray | None = None,
                  spatial_mask: np.ndarray | None = None,
                  temporal_mask: np.ndarray | None = None,
                  only_cloud: bool = False,
                  only_clear: bool = False) -> np.ndarray:
    """Apply combination of cloud/clear, spatial, and temporal filters."""

    combined_mask = np.ones(data.shape[0], dtype=bool)

    # Cloud/clear filtering
    if cloud_mask is not None:
        if only_cloud:
            combined_mask &= cloud_mask.astype(bool)
        elif only_clear:
            combined_mask &= ~cloud_mask.astype(bool)

    # Spatial filtering
    if spatial_mask is not None:
        combined_mask &= spatial_mask

    # Temporal filtering
    if temporal_mask is not None:
        combined_mask &= temporal_mask

    return data[combined_mask]


def reshuffle_spatiotemporal(input_path: str, output_path: str,
                             split_ratios: dict | None = None,
                             single_scan: int | None = None) -> None:
    """Reshuffle data into train/valid/test splits using spatial boundaries.

    Useful for static scenarios (single timestep) to create meaningful train/valid/test
    splits based on spatial location (convex hull boundaries) rather than just timesteps.

    Parameters
    ----------
    input_path: str. Path to recasted data containing prof.npy, lat.npy, lon.npy, etc.
    output_path: str. Base path where to save new splits (train/, valid/, test/)
    split_ratios: dict or None. Split proportions {train: 0.6, valid: 0.2, test: 0.2}
                 Default: {train: 0.6, valid: 0.2, test: 0.2}
    single_scan: int or None. If specified, extract only this scan before splitting
    """
    logger.info(f"Reshuffling data spatiotemporal splits...")

    # Default split
    if split_ratios is None:
        split_ratios = {'train': 0.6, 'valid': 0.2, 'test': 0.2}

    # Validate split ratios
    total = sum(split_ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    # Load coordinate data
    lat = np.load(os.path.join(input_path, 'lat.npy'))
    lon = np.load(os.path.join(input_path, 'lon.npy'))

    # Load scans if available
    scans_path = os.path.join(input_path, 'scans.npy')
    scans = np.load(scans_path) if os.path.exists(scans_path) else np.arange(len(lat))

    # Filter to single scan if requested
    if single_scan is not None:
        mask = scans == single_scan
        lat = lat[mask]
        lon = lon[mask]
        scans = scans[mask]
        logger.info(f"  Filtered to scan {single_scan}: {len(lat)} samples")

    # Compute spatial boundaries using convex hull
    n_samples = len(lat)
    coords = np.column_stack((lat, lon))

    logger.info(f"  Computing convex hull for {n_samples} samples...")
    hull = ConvexHull(coords)
    hull_indices = hull.vertices
    remaining_indices = np.setdiff1d(np.arange(n_samples), hull_indices)

    logger.info(f"  Boundary points: {len(hull_indices)}, Interior points: {len(remaining_indices)}")

    # Shuffle remaining indices
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(remaining_indices)

    # Create splits: give boundary points to training
    split_indices = {'train': [], 'valid': [], 'test': []}
    split_indices['train'].extend(hull_indices)

    # Split remaining points according to ratios
    remaining_train = int(split_ratios['train'] * len(remaining_indices))
    remaining_valid = int(split_ratios['valid'] * len(remaining_indices))

    split_indices['train'].extend(remaining_indices[:remaining_train])
    split_indices['valid'].extend(remaining_indices[remaining_train:remaining_train + remaining_valid])
    split_indices['test'].extend(remaining_indices[remaining_train + remaining_valid:])

    # Convert to arrays
    split_indices = {k: np.array(v, dtype='int64') for k, v in split_indices.items()}

    # Log split sizes
    for split_name, indices in split_indices.items():
        logger.info(f"  {split_name}: {len(indices)} samples ({100*len(indices)/n_samples:.1f}%)")

    # Get all filenames from input
    filenames = os.listdir(input_path)

    # Save splits
    for split_name, indices in split_indices.items():
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
        coverage_mask = np.zeros_like(data['lat'], dtype='bool')
        spatiotemporal_mask = np.ones_like(data['lat'], dtype='bool')
        # Spatial extent
        if hasattr(input.mask, 'spatial_domain'):
            if hasattr(input.mask.spatial_domain, 'lat_min'):
                spatiotemporal_mask &= (data['lat'] >= input.mask.spatial_domain.lat_min)
            if hasattr(input.mask.spatial_domain, 'lat_max'):
                spatiotemporal_mask &= (data['lat'] <= input.mask.spatial_domain.lat_max)
            if hasattr(input.mask.spatial_domain, 'lon_min'):
                spatiotemporal_mask &= (data['lon'] >= input.mask.spatial_domain.lon_min)
            if hasattr(input.mask.spatial_domain, 'lon_max'):
                spatiotemporal_mask &= (data['lon'] <= input.mask.spatial_domain.lon_max)
        # Temporal extent
        if hasattr(input.mask, 'temporal_window'):
            if hasattr(input.mask.temporal_window, 'scan_min'):
                spatiotemporal_mask &= (data['scans'] >= input.mask.temporal_window.scan_min)
            if hasattr(input.mask.temporal_window, 'scan_max'):
                spatiotemporal_mask &= (data['scans'] <= input.mask.temporal_window.scan_max)
        # Update mask
        mask &= spatiotemporal_mask
        # Clouds or clear sky
        if hasattr(input.mask, 'cloud_mask'):
            if input.mask.cloud_mask is True:
                coverage_mask &= cloud_mask(data['prof'])
        if hasattr(input.mask, 'clear_mask'):
            if input.mask.clear_mask is True:
                coverage_mask &= clear_mask(data['prof'])
        # Update mask
        mask &= coverage_mask

    # Shuffle or maintain distribution
    if hasattr(input, 'shuffle') and input.shuffle:
        # TODO: Complete
        pass
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

