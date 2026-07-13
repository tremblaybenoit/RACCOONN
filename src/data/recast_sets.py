import numpy as np
from data.filters import clearsky_filter, pressure_filter, daytime_filter
import os
import logging
from utilities.tensors import to_numpy
from data.io import load_npy, load_latlon, load_scans, load_stack
from data.covariance import innovation_uncertainty, background_climatology
from tqdm import tqdm
import gc
import argparse
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def onion_peel_boundary(coords, layers=2):
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


def main(in_dir, in_precision, out_dir, out_precision, out_cloud_filter, out_clearsky_filter) -> None:
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
    train_in_dir = os.path.join(in_dir, "Train2")
    filenames = os.listdir(train_in_dir)

    # Stages
    stages = ["Train2", "Val2", "Test2"]
    stages_split = {"Train2": 0.6, "Val2": 0.2, "Test2": 0.2}

    # Load scans (timesteps) as one stack
    scans = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "scans.npy"), dtype=in_precision)
                            for stage_name in stages], axis=0)
    lat = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "lat.npy"), dtype=in_precision)
                          for stage_name in stages], axis=0)
    lon = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "lon.npy"), dtype=in_precision)
                          for stage_name in stages], axis=0)
    # Number of samples, scans, coordinates
    n_samples = scans.shape[0]
    n_scans = np.unique(scans).shape[0]
    n_coords = n_samples//n_scans
    # Establish boundaries of spatial domain using a convex hull
    lat, lon = lat[0:n_coords], lon[0:n_coords]
    coords = np.stack((lat, lon), axis=-1)
    hull_indices = onion_peel_boundary(coords, layers=2)
    remaining_indices = np.setdiff1d(np.arange(n_coords), hull_indices)
    # Split the remaining indices according to the stages proportions
    # np.random.shuffle(remaining_indices)
    # train_indices = np.concatenate([train_indices, remaining_indices[0:int(stages_split['Train2']*len(remaining_indices))]], axis=0)
    # val_indices = remaining_indices[int(stages_split['Train2']*len(remaining_indices)):
    #                                  int((stages_split['Train2']+stages_split['Val2'])*len(remaining_indices))]
    # test_indices = remaining_indices[int((stages_split['Train2']+stages_split['Val2'])*len(remaining_indices)):]

    # Broadcast to all scans
    # remaining_indices = np.concatenate([remaining_indices + i*n_coords for i in range(n_scans)], axis=0)
    # breakpoint()

    # Currently, the Train2, Val2, Test2 datasets do not have overlapping scans, but share the same lat/lon coordinates.
    # Let's generate datasets with different permutations of the samples to avoid any potential data leakage.

    # 1. Permutate the data in ascending scan order (i.e., group by scans first, then by lat/lon coordinates).
    scans_order = np.argsort(scans, kind='stable')

    # Cloud/clearsky filters
    data_filter = None
    # if out_cloud_filter or out_clearsky_filter:
    #     # Load cloud filter
    #     data_filter = np.concatenate([load_npy(os.path.join(in_dir, stage_name, "cloud_filter.npy"), dtype=in_precision)
    #                                   for stage_name in stages], axis=0)
    #     # If clearsky filter, invert mask
    #     if out_clearsky_filter:
    #         data_filter = ~data_filter
    #     # Apply scan ordering
    #     data_filter = data_filter[scans_order]

    # Loop over files to recast
    data = {}
    for filename in tqdm(filenames):

        print(filename)

        # Examine shape of data
        data[filename] = np.load(os.path.join(in_dir, "Train2", filename))
        # If there is more than one sample, load full data stack
        if data[filename].shape[0] > 127:
            # Load full data stack
            data[filename] = np.concatenate([data[filename]] + [load_npy(os.path.join(in_dir, stage_name, filename), dtype=in_precision)
                                            for stage_name in stages[1:]], axis=0)
            # Reorder data by scans
            data[filename] = data[filename][scans_order]
            # Filter data if needed
            # if data_filter is not None:
            #     data[filename]= data[filename][data_filter]
            # For profiles
            if filename == "prof.npy":
                # If clearsky profiles, only keep non-zero profile types
                if out_clearsky_filter:
                    data[filename] = np.take(data[filename], [0, 4, 8], axis=1)
                # Compute background
                # data['prof_background.npy'] = data[filename].mean(axis=0)
                # data['prof_increment.npy'] = data[filename] - data['prof_background.npy']
            # Extract data filter
            if filename == 'cloud_filter.npy' and (out_cloud_filter or out_clearsky_filter):
                data_filter = data[filename].astype(np.bool_, copy=False)
                if out_clearsky_filter:
                    data_filter = ~data_filter

    # Static scenario: We only extract one timestep (scan) but keep all lat/lon coordinates.
    stages_indices = {'Train2': [], 'Val2': [], 'Test2': []}
    # for scan in range(n_scans):
    for scan in range(1):
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

    # Update filter
    if data_filter is not None:
        stages_indices['Train2'] = stages_indices['Train2'][data_filter[stages_indices['Train2']]]
        stages_indices['Val2'] = stages_indices['Val2'][data_filter[stages_indices['Val2']]]
        stages_indices['Test2'] = stages_indices['Test2'][data_filter[stages_indices['Test2']]]

    # Filter data and save in output directory
    for stage_name in tqdm(stages):
        # Output directory
        out_dir_stage = os.path.join(out_dir, stage_name)
        os.makedirs(out_dir_stage, exist_ok=True)
        # Indices
        indices = stages_indices[stage_name]
        # Loop over files to recast
        for filename in list(data.keys()):
            # Check shape of data
            if data[filename].shape[0] <= 127:
                # If small data, no need to filter
                data_out = data[filename]
            else:
                # Else, filter data
                data_out = data[filename][indices]

            # Save filtered data
            out_path = os.path.join(out_dir_stage, filename)
            np.save(out_path, data_out)
            logger.info("Saved `%s` to `%s`", filename, out_path)
            del data_out
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
    args = parser.parse_args()

    main(args.in_dir, args.in_precision, args.out_dir, args.out_precision,
         args.cloud_filter, args.clearsky_filter)
