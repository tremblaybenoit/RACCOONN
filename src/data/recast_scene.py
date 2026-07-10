import numpy as np
from data.filters import clearsky_filter, pressure_filter, daytime_filter
import os
import logging
from data.io import load_npy
from tqdm import tqdm
import gc
import argparse
from scipy.spatial import ConvexHull
from utilities.plot import plot_map, save_plot, flexible_gridspec
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def onion_peel_boundary(coords, layers=5):
    all_indices = np.arange(len(coords))
    boundary_indices = []
    current_coords = coords.copy()
    current_indices = all_indices.copy()

    for _ in range(layers):
        if len(current_coords) < 3: break
        hull = ConvexHull(current_coords)
        # Store the original indices of the hull vertices
        boundary_indices.extend(current_indices[hull.vertices])
        # Remove these points and repeat
        mask = np.ones(len(current_coords), dtype=bool)
        mask[hull.vertices] = False
        current_coords = current_coords[mask]
        current_indices = current_indices[mask]

    return np.array(boundary_indices)


def main(in_dir, in_precision, out_dir, out_precision, out_cloud_filter, out_clearsky_filter,
         timestep=None, lat_min=None, lat_max=None, lon_min=None, lon_max=None) -> None:
    """
    Compute statistics of a given dataset.

    Parameters
    ----------
    in_dir: str. Input data directory.
    in_precision: str. Input data precision (e.g., 'float32', 'float64').
    out_dir: str. Output data directory.
    out_precision: str. Output data precision (e.g., 'float32', 'float64').
    out_cloud_filter: bool. If True, output only cloudy profiles.
    out_clearsky_filter: bool. If True, output only clearsky profiles.
    timestep: int. Timestep/scan to filter on (if None, keep all timesteps).
    lat_min: float. Minimum latitude for spatial filtering (inclusive).
    lat_max: float. Maximum latitude for spatial filtering (inclusive).
    lon_min: float. Minimum longitude for spatial filtering (inclusive).
    lon_max: float. Maximum longitude for spatial filtering (inclusive).

    Returns
    -------
    None.
    """

    # Steps:
    # 1. Resave data in original precision, but computing beforehand some of the quantities for simplicity (e.g., background, filters, etc).
    # 2. Resave data from 1. in new precision (float32).
    # 3. Filter data into clearsky/cloudy cases and resave in both precisions.

    # Output directory
    os.makedirs(out_dir, exist_ok=True)
    # Files to recast
    filenames = ['prof.npy', 'cloud_filter.npy', 'daytime_filter.npy', 'hofx.npy', 'lat.npy', 'lon.npy',
                 'meta.npy', 'obs.npy', 'pressure.npy', 'scans.npy', 'surf.npy']

    # Stages
    stages = ["Train2", "Val2", "Test2"]
    stages_split = {"Train2": 0.5, "Val2": 0.25, "Test2": 0.25}

    # Load scans (timesteps) as one stack
    scans = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "scans.npy"), dtype=in_precision)
                            for stage_name in stages], axis=0)
    lat = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "lat.npy"), dtype=in_precision)
                          for stage_name in stages], axis=0)
    lon = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "lon.npy"), dtype=in_precision)
                          for stage_name in stages], axis=0)
    cloud_filter = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "cloud_filter.npy"), dtype=in_precision)
                                  for stage_name in stages], axis=0).astype(int)

    # Apply temporal and spatial filtering if needed
    mask = np.ones_like(lat, dtype=bool)
    if lat_min is not None:
        mask &= (lat >= lat_min)
    if lat_max is not None:
        mask &= (lat <= lat_max)
    if lon_min is not None:
        mask &= (lon >= lon_min)
    if lon_max is not None:
        mask &= (lon <= lon_max)
    extent_mask = mask.copy()
    if out_cloud_filter:
        mask &= (cloud_filter == 1)
    if out_clearsky_filter:
        mask &= (cloud_filter == 0)
    spatial_mask = mask.copy()  # Store spatial mask for later use
    if timestep is not None:
        mask &= (scans == timestep)
        extent_mask &= (scans == timestep)

    lat_extent, lon_extent, cloud_extent = lat[extent_mask], lon[extent_mask], cloud_filter[extent_mask]
    coords = np.column_stack((lat_extent, lon_extent))
    tree = cKDTree(coords)
    dist, neigh = tree.query(coords, k=5)  # 5 neighbors + itself
    cloud_extent = cloud_extent.astype(bool)
    clear_extent = ~cloud_extent
    neigh = neigh[:, 1:]  # Remove self-neighbor
    has_cloud_neighbor = cloud_extent[neigh].any(axis=1)

    # fraction of the total span used as tolerance (adjust as needed)
    frac_lat, frac_lon = 0.050, 0.050
    lat_span = lat_max - lat_extent.min()
    lon_span = lon_max - lon_extent.min()
    tol_lat = 0.75 # frac_lat * lat_span if lat_span > 0 else frac_lat * (abs(lat_extent.max()) + 1e-6)
    tol_lon = 0.75 # frac_lon * lon_span if lon_span > 0 else frac_lon * (abs(lon_extent.max()) + 1e-6)
    has_edge = (
            (lat_extent >= lat_max - tol_lat) |
            (lat_extent <= lat_min + tol_lat) |
             (lon_extent >= lon_max - tol_lon) |
             (lon_extent <= lon_min + tol_lon)
    )

    hull_indices = np.flatnonzero(extent_mask)[clear_extent & (has_cloud_neighbor | has_edge)]
    remaining_indices = np.flatnonzero(extent_mask)[clear_extent & ~(has_cloud_neighbor | has_edge)]
    breakpoint()

    # Extract indices of hull points and remaining points. We want to ensure that no validation/test points are alone or on boundaries.
    # coords = np.column_stack((lat, lon))
    # tree = cKDTree(coords)
    # dist, neigh = tree.query(coords, k=5)  # 4 neighbors + itself
    # neigh = neigh[:, 1:]  # Remove self-neighbor
    # has_neighbor = (~mask)[neigh].any(axis=1)
    # hull_indices = np.flatnonzero(mask & has_neighbor)
    # remaining_indices = np.flatnonzero(mask & ~has_neighbor)
    # breakpoint()

    lat, lon, scans = lat[mask], lon[mask], scans[mask]
    gc.collect()

    # Number of samples, scans, coordinates
    n_scans = np.unique(scans).shape[0]
    n_coords = mask.sum()
    # Establish boundaries of spatial domain. We want to ensure that no validation/test points are alone or on boundaries.
    # Look for hull points and add to training set.



    # fraction of the total span used as tolerance (adjust as needed)
    frac_lat, frac_lon = 0.050, 0.050
    lat_span = lat_max - lat_extent.min()
    lon_span = lon_max - lon_extent.min()
    tol_lat = frac_lat * lat_span if lat_span > 0 else frac_lat * (abs(lat_extent.max()) + 1e-6)
    tol_lon = frac_lon * lon_span if lon_span > 0 else frac_lon * (abs(lon_extent.max()) + 1e-6)
    edge_indices = np.flatnonzero(
            (lat_extent >= lat_max - tol_lat) |
            (lat_extent <= lat_min + tol_lat) |
             (lon_extent >= lon_max - tol_lon) |
             (lon_extent <= lon_min + tol_lon)
    )
    # remaining_indices = np.setdiff1d(np.arange(n_coords), hull_indices)
    # Update indices
    # hull_indices = np.flatnonzero(mask)[hull_indices]
    # remaining_indices = np.flatnonzero(mask)[remaining_indices]

    # Static scenario: We only extract one timestep (scan). We shuffle lat/lon pairs across the training,
    # validation and test sets. However, the hull_indices must be part of the training set indices.
    stages_indices = {'Train2': [], 'Val2': [], 'Test2': []}
    # for scan in range(n_scans):
    for scan in range(n_scans):
        np.random.shuffle(remaining_indices)
        stages_indices['Train2'].extend(hull_indices + scan * n_coords)
        stages_indices['Train2'].extend(
            remaining_indices[0:int(stages_split['Train2'] * len(remaining_indices))] + scan * n_coords)
        stages_indices['Val2'].extend(remaining_indices[int(stages_split['Train2'] * len(remaining_indices)):
                                               int((stages_split['Train2'] + stages_split['Val2']) * len(
                                                   remaining_indices))] + scan * n_coords)
        stages_indices['Test2'].extend(remaining_indices[int((stages_split['Train2'] + stages_split['Val2']) * len(
            remaining_indices)):] + scan * n_coords)
    stages_indices['Train2'] = np.array(stages_indices['Train2'])
    stages_indices['Val2'] = np.array(stages_indices['Val2'])
    stages_indices['Test2'] = np.array(stages_indices['Test2'])

    # Create output directory
    for stage_name in tqdm(stages):
        # Output directory
        out_dir_stage = os.path.join(out_dir, stage_name)
        os.makedirs(out_dir_stage, exist_ok=True)

    # Load data, concatenante, filter out, and save
    for filename in tqdm(filenames):
        print(filename)

        # Examine shape of data
        data = np.load(os.path.join(in_dir, "Train2", filename))
        # If there is more than one sample, load full data stack
        if data.shape[0] > 127:
            # Load full data stack
            data = np.concatenate([data] + [load_npy(os.path.join(in_dir, stage_name, filename), dtype=in_precision)
                                                for stage_name in stages[1:]], axis=0)
            # For profiles
            if filename == "prof.npy":
                # If clearsky profiles, only keep non-zero profile types
                if out_clearsky_filter:
                    data = np.take(data, [0, 4, 8], axis=1)
                # Compute background
                prof_background = data[spatial_mask].mean(axis=0, keepdims=True)

            # Loop over stages and save filtered data
            for stage_name in stages:
                # Output directory
                out_dir_stage = os.path.join(out_dir, stage_name)
                # Indices
                indices = stages_indices[stage_name]
                # Filter data
                data_out = np.take(data, indices, axis=0)
                # If profiles, also save increments
                if filename == "prof.npy":
                    # Compute profile increments
                    prof_increment = data_out - prof_background
                    # Save profile background and increments
                    prof_background_path = os.path.join(out_dir_stage, "prof_background.npy")
                    np.save(prof_background_path, prof_background.astype(out_precision))
                    logger.info("Saved profile background to `%s`", prof_background_path)
                    prof_increment_path = os.path.join(out_dir_stage, "prof_increment.npy")
                    np.save(prof_increment_path, prof_increment.astype(out_precision))
                    logger.info("Saved profile increments to `%s`", prof_increment_path)
                    del prof_increment
                    gc.collect()
                # Save filtered data
                out_path = os.path.join(out_dir_stage, filename)
                np.save(out_path, data_out.astype(out_precision))
                print(data_out.min(), data_out.max())
                logger.info("Saved `%s` to `%s`", filename, out_path)
                del data_out
                gc.collect()
        else:
            # If there is only one sample, save it in the new precision without loading the full stack
            for stage_name in stages:
                # Output directory
                out_dir_stage = os.path.join(out_dir, stage_name)
                # Save filtered data
                out_path = os.path.join(out_dir_stage, filename)
                np.save(out_path, data.astype(out_precision))
                logger.info("Saved `%s` to `%s`", filename, out_path)

    return

if __name__ == '__main__':
    """ Recast a given dataset.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Recasted dataset saved to disk.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', type=str, default="../data_float32",
                        help='Path to configuration file containing all model hyperparameters.')
    parser.add_argument('-in_precision', type=str, default='float32',
                        help='Name of the configuration file containing all model hyperparameters.')
    parser.add_argument('-out_dir', type=str, default="../sets_float32",
                        help='Name of the experiment that overrides the main hydra configuration.')
    parser.add_argument('-out_precision', type=str, default="float32",
                        help='Name of the experiment that overrides the main hydra configuration.')
    parser.add_argument('-cloud_filter', type=bool, default=False,
                        help='Flag to print the configuration file contents.')
    parser.add_argument('-clearsky_filter', type=bool, default=False,
                        help='Flag to print the configuration file contents.')
    parser.add_argument('-timestep', type=int, default=0,
                        help='Timestep/scan to extract for the static scenario.')
    parser.add_argument('-lat_min', type=float, default=-15.,
                        help='Minimum latitude of the spatial domain to keep.')
    parser.add_argument('-lat_max', type=float, default=0.,
                        help='Maximum latitude of the spatial domain to keep.')
    parser.add_argument('-lon_min', type=float, default=-90.,
                        help='Minimum longitude of the spatial domain to keep.')
    parser.add_argument('-lon_max', type=float, default=-15.,
                        help='Maximum longitude of the spatial domain to keep.')
    args = parser.parse_args()

    main(args.in_dir, args.in_precision, args.out_dir, args.out_precision,
         args.cloud_filter, args.clearsky_filter, timestep=args.timestep, lat_min=args.lat_min,
         lat_max=args.lat_max, lon_min=args.lon_min, lon_max=args.lon_max)
