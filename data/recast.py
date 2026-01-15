import numpy as np
from data.filters import clearsky_filter, pressure_filter, daytime_filter
import os
import logging
from utilities.tensors import to_numpy
from data.io import load_npy, load_latlon, load_scans
from data.covariance import innovation_uncertainty, background_climatology
from tqdm import tqdm
import gc
import argparse


logger = logging.getLogger(__name__)


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

    # Input data directories
    for stage_name in tqdm(["Train2", "Val2", "Test2"]):

        # Input directory
        in_dir_stage = os.path.join(in_dir, stage_name)
        # Output directory
        out_dir_stage = os.path.join(out_dir, stage_name)
        os.makedirs(out_dir_stage, exist_ok=True)

        # Load profiles and convert to desired precision
        prof_path = os.path.join(in_dir_stage, "prof.npy")
        prof = load_npy(prof_path, dtype=in_precision)

        # Convert to desired precision
        if in_precision != out_precision:
            prof = to_numpy(prof, dtype=out_precision)

        # Compute clearsky/cloudy filters
        clearsky_mask = clearsky_filter(prof).astype(np.bool_, copy=False)
        cloud_mask = ~clearsky_mask

        # Apply mask
        if out_cloud_filter:
            if out_clearsky_filter:
                raise ValueError("Cannot output both cloud and clearsky filters.")
            prof_filter = cloud_mask
            prof = prof[prof_filter]
            cloud_mask = cloud_mask[prof_filter]
        elif out_clearsky_filter:
            prof_filter = clearsky_mask
            prof = prof[prof_filter]
            cloud_mask = cloud_mask[prof_filter]
            # Extract non-zero profiles types
            prof = np.take(prof, [0, 4, 8], axis=1)
        else:
            prof_filter = None

        # Save cloud filter in input directory
        cloud_filter_path = os.path.join(out_dir_stage, "cloud_filter.npy")
        np.save(cloud_filter_path, cloud_mask)
        logger.info("Saved cloud filter to `%s`", cloud_filter_path)
        del cloud_mask
        gc.collect()

        # Save profiles in new directory
        prof_out_path = os.path.join(out_dir_stage, "prof.npy")
        np.save(prof_out_path, prof)
        logger.info("Saved profiles to `%s`", prof_out_path)

        # Compute pressure filter
        pressure_mask = pressure_filter(prof).astype(np.bool_, copy=False)
        # Save pressure filter in new directory
        pressure_filter_path = os.path.join(out_dir, "pressure_filter.npy")
        np.save(pressure_filter_path, pressure_mask)
        logger.info("Saved pressure filter to `%s`", pressure_filter_path)
        del pressure_mask
        gc.collect()

        # Compute background profiles (mean profile) in a memory efficient way
        # prof_mean = background_climatology(prof, keepdims=True)
        # prof_background_path = os.path.join(out_dir_stage, "prof_background.npy")
        # np.save(prof_background_path, prof_mean.astype(prof.dtype, copy=False))
        # logger.info("Saved background profile to `%s`", prof_background_path)

        # Compute profile increments and save
        # np.subtract(prof, prof_mean, out=prof)
        # prof_increment_path = os.path.join(out_dir_stage, "prof_increment.npy")
        # np.save(prof_increment_path, prof.astype(prof.dtype, copy=False))
        # logger.info("Saved profile increments to `%s`", prof_increment_path)

        # Clean up
        del prof  #, prof_mean
        # gc.collect()

        # Coordinates
        lat_path = os.path.join(in_dir_stage, "lat.npy")
        lon_path = os.path.join(in_dir_stage, "lon.npy")
        pressure_path = os.path.join(in_dir_stage, "pressure.npy")
        if os.path.exists(os.path.join(in_dir_stage, "scans.npy")):
            scans_path = os.path.join(in_dir_stage, "scans.npy")
            scans = load_npy(scans_path, dtype=in_precision)
            lat = load_npy(lat_path, dtype=in_precision)
            lon = load_npy(lon_path, dtype=in_precision)
        else:
            scans_path = os.path.join(in_dir_stage, "scans.txt")
            scans = load_scans(scans_path, lat=load_npy(lat_path, dtype=in_precision),
                               dtype=in_precision)
            lat = load_latlon(lat_path, scans=load_scans(scans_path, dtype=in_precision), dtype=in_precision)
            lon = load_latlon(lon_path, scans=load_scans(scans_path, dtype=in_precision), dtype=in_precision)
        pressure = load_npy(pressure_path, dtype=in_precision)

        # Filter out profiles if needed
        if prof_filter is not None:
            scans = scans[prof_filter]
            lat = lat[prof_filter]
            lon = lon[prof_filter]

        # Convert to desired precision
        if in_precision != out_precision:
            scans = to_numpy(scans, dtype=out_precision)
            lat = to_numpy(lat, dtype=out_precision)
            lon = to_numpy(lon, dtype=out_precision)
            pressure = to_numpy(pressure, dtype=out_precision)

        # Save coordinates in new directory
        scans_out_path = os.path.join(out_dir_stage, "scans.npy")
        lat_out_path = os.path.join(out_dir_stage, "lat.npy")
        lon_out_path = os.path.join(out_dir_stage, "lon.npy")
        pressure_out_path = os.path.join(out_dir_stage, "pressure.npy")
        np.save(scans_out_path, scans)
        np.save(lat_out_path, lat)
        np.save(lon_out_path, lon)
        np.save(pressure_out_path, pressure)
        logger.info("Saved coordinates to `%s`, `%s`, `%s`, `%s`",
                    scans_out_path, lat_out_path, lon_out_path, pressure_out_path)
        # Clean up
        del scans, lat, lon, pressure
        gc.collect()

        # Surface data
        surf_path = os.path.join(in_dir_stage, "surf.npy")
        surf = load_npy(surf_path, dtype=in_precision)
        # Apply mask
        if prof_filter is not None:
            surf = surf[prof_filter]
        # Convert to desired precision
        if in_precision != out_precision:
            surf = to_numpy(surf, dtype=out_precision)
        # Save surface data in new directory
        surf_out_path = os.path.join(out_dir_stage, "surf.npy")
        np.save(surf_out_path, surf)
        logger.info("Saved surface data to `%s`", surf_out_path)
        # Clean up
        del surf
        gc.collect()

        # Meta data
        meta_path = os.path.join(in_dir_stage, "meta.npy")
        meta = load_npy(meta_path, dtype=in_precision)
        # Apply mask
        if prof_filter is not None:
            meta = meta[prof_filter]
        # Convert to desired precision
        if in_precision != out_precision:
            meta = to_numpy(meta, dtype=out_precision)
        # Save metadata in new directory
        meta_out_path = os.path.join(out_dir_stage, "meta.npy")
        np.save(meta_out_path, meta)
        logger.info("Saved meta data to `%s`", meta_out_path)
        # Compute daytime filter and save
        daytime_mask = daytime_filter(meta).astype(np.bool_, copy=False)
        daytime_filter_path = os.path.join(out_dir_stage, "daytime_filter.npy")
        np.save(daytime_filter_path, daytime_mask)
        logger.info("Saved daytime filter to `%s`", daytime_filter_path)
        # Clean up
        del meta, daytime_mask
        gc.collect()

        # Hofx
        obs_path = os.path.join(in_dir_stage, "obs.npy")
        obs = load_npy(obs_path, dtype=in_precision)
        # Apply mask
        if prof_filter is not None:
            obs = obs[prof_filter]
        # Convert to desired precision
        if in_precision != out_precision:
            obs = to_numpy(obs, dtype=out_precision)
        # Save obs data in new directory
        obs_out_path = os.path.join(out_dir_stage, "obs.npy")
        np.save(obs_out_path, obs)
        logger.info("Saved obs data to `%s`", obs_out_path)
        # Clean up
        del obs
        gc.collect()
        # Hofx (inferences)
        hofx_path = os.path.join(in_dir_stage, "hofx.npy")
        if os.path.exists(hofx_path):
            hofx = load_npy(hofx_path, dtype=in_precision)
            # Apply mask
            if prof_filter is not None:
                hofx = hofx[prof_filter]
            # Convert to desired precision
            if in_precision != out_precision:
                hofx = to_numpy(hofx, dtype=out_precision)
            # Save hofx data in new directory
            hofx_out_path = os.path.join(out_dir_stage, "hofx.npy")
            np.save(hofx_out_path, hofx)
            logger.info("Saved hofx data to `%s`", hofx_out_path)
        hofx_predict_v3_path = os.path.join(f'../data_{out_precision}/{stage_name}', "predict_hofx_v3.npy")
        if os.path.exists(hofx_predict_v3_path):
            hofx = load_npy(hofx_predict_v3_path, dtype=in_precision)
            # Apply mask
            if prof_filter is not None:
                hofx = hofx[prof_filter]
            # Convert to desired precision
            if in_precision != out_precision:
                hofx = to_numpy(hofx, dtype=out_precision)
            # Save hofx data in new directory
            hofx_out_path = os.path.join(out_dir_stage, "predict_hofx_v3.npy")
            np.save(hofx_out_path, hofx)
            logger.info("Saved predict_hofx_v3 data to `%s`", hofx_out_path)
        hofx_predict_v4_path = os.path.join(f'../data_{out_precision}/{stage_name}', "predict_hofx_v4.npy")
        if os.path.exists(hofx_predict_v4_path):
            hofx = load_npy(hofx_predict_v4_path, dtype=in_precision)
            # Apply mask
            if prof_filter is not None:
                hofx = hofx[prof_filter]
            # Convert to desired precision
            if in_precision != out_precision:
                hofx = to_numpy(hofx, dtype=out_precision)
            # Save hofx data in new directory
            hofx_out_path = os.path.join(out_dir_stage, "predict_hofx_v4.npy")
            np.save(hofx_out_path, hofx)
            logger.info("Saved predict_hofx_v4 data to `%s`", hofx_out_path)
        hofx_predict_v5_path = os.path.join(f'../data_{out_precision}/{stage_name}', "predict_hofx_v5.npy")
        if os.path.exists(hofx_predict_v5_path):
            hofx = load_npy(hofx_predict_v5_path, dtype=in_precision)
            # Apply mask
            if prof_filter is not None:
                hofx = hofx[prof_filter]
            # Convert to desired precision
            if in_precision != out_precision:
                hofx = to_numpy(hofx, dtype=out_precision)
            # Save hofx data in new directory
            hofx_out_path = os.path.join(out_dir_stage, "predict_hofx_v5.npy")
            np.save(hofx_out_path, hofx)
            logger.info("Saved predict_hofx_v5 data to `%s`", hofx_out_path)
        hofx_predict_v6_path = os.path.join(f'../data_{out_precision}/{stage_name}', "predict_hofx_v6.npy")
        if os.path.exists(hofx_predict_v6_path):
            hofx = load_npy(hofx_predict_v6_path, dtype=in_precision)
            # Apply mask
            if prof_filter is not None:
                hofx = hofx[prof_filter]
            # Convert to desired precision
            if in_precision != out_precision:
                hofx = to_numpy(hofx, dtype=out_precision)
            # Save hofx data in new directory
            hofx_out_path = os.path.join(out_dir_stage, "predict_hofx_v6.npy")
            np.save(hofx_out_path, hofx)
            logger.info("Saved predict_hofx_v6 data to `%s`", hofx_out_path)
        # End of stage loop
        del prof_filter
        gc.collect()

    breakpoint()
    # Reload all profiles and compute background and increments. Store in each stage directory.
    prof = [load_npy(os.path.join(out_dir, stage_name, "prof.npy"), dtype=out_precision) for stage_name in ["Train2", "Val2", "Test2"]]
    prof_mean = background_climatology(np.concatenate(prof, axis=0), keepdims=True)
    for s, stage in enumerate(["Train", "Val", "Test"]):
        prof_stage = prof[s]
        prof_background_path = os.path.join(out_dir, f"{stage}2", "prof_background.npy")
        np.save(prof_background_path, prof_mean)
        logger.info("Saved background profile to `%s`", prof_background_path)
        # Compute profile increments and save
        np.subtract(prof_stage, prof_mean, out=prof_stage)
        prof_increment_path = os.path.join(out_dir, f"{stage}2", "prof_increment.npy")
        np.save(prof_increment_path, prof_stage)
        logger.info("Saved profile increments to `%s`", prof_increment_path)
        # Clean up
        del prof_stage
        gc.collect()
    del prof, prof_mean
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
    parser.add_argument('-in_dir', type=str, default="../../GOES_ML-main/Data",
                        help='Path to configuration file containing all model hyperparameters.')
    parser.add_argument('-in_precision', type=str, default=None,
                        help='Name of the configuration file containing all model hyperparameters.')
    parser.add_argument('-out_dir', type=str, default="../data_float64",
                        help='Name of the experiment that overrides the main hydra configuration.')
    parser.add_argument('-out_precision', type=str, default="float64",
                        help='Name of the experiment that overrides the main hydra configuration.')
    parser.add_argument('-cloud_filter', type=bool, default=False,
                        help='Flag to print the configuration file contents.')
    parser.add_argument('-clearsky_filter', type=bool, default=False,
                        help='Flag to print the configuration file contents.')
    args = parser.parse_args()

    main(args.in_dir, args.in_precision, args.out_dir, args.out_precision,
         args.cloud_filter, args.clearsky_filter)
