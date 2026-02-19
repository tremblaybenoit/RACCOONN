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

    # Create output directory
    for stage_name in tqdm(stages):
        # Output directory
        out_dir_stage = os.path.join(out_dir, stage_name)
        os.makedirs(out_dir_stage, exist_ok=True)

        # Load scans (timesteps) as one stack
        scans = load_npy(os.path.join(in_dir, stage_name, "scans.npy"), dtype=in_precision)
        lat = load_npy(os.path.join(in_dir, stage_name, "lat.npy"), dtype=in_precision)
        lon = load_npy(os.path.join(in_dir, stage_name, "lon.npy"), dtype=in_precision)
        cloud_filter = load_npy(os.path.join(in_dir, stage_name, "cloud_filter.npy"), dtype=in_precision)

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
        if out_cloud_filter:
            mask &= (cloud_filter == 1)
        if out_clearsky_filter:
            mask &= (cloud_filter == 0)
        if timestep is not None:
            mask &= (scans == timestep)

        # Load data, concatenante, filter out, and save
        for filename in tqdm(filenames):
            print(filename)

            # Examine shape of data
            data = np.load(os.path.join(in_dir, stage_name, filename))
            # If there is more than one sample, load full data stack
            if data.shape[0] > 127:

                # For profiles
                if filename == "prof.npy":
                    # If clearsky profiles, only keep non-zero profile types
                    if out_clearsky_filter:
                        data = np.take(data, [0, 4, 8], axis=1)

                # Filter data
                data = data[mask]

            # Save filtered data
            out_path = os.path.join(out_dir_stage, filename)
            np.save(out_path, data.astype(out_precision))
            logger.info("Saved `%s` to `%s`", filename, out_path)
            del data
            gc.collect()

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
    parser.add_argument('-timestep', type=int, default=None,
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
