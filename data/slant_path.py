import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def slant_path_geometry(pressure: np.ndarray, temperature: np.ndarray, sza: np.ndarray, az: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Compute slant path heights and latitude and longitude offsets.

        Parameters:
        -----------
        pressure: Pressure levels.
        temperature: Temperature at pressure levels.
        sza: Zenith angle.
        az: Azimuth angle.
        lat: Latitude angle.
        lon: Longitude angle.

        Returns:
        --------
        h: Hypsometric height.
        lat: Latitude angle with offset applied.
        lon: Longitude angle with offset applied.
    """

    # Constants
    Rd = 287.05
    g = 9.81
    km_factor = 1000.0
    deg_to_km = 111.12

    # Convert angles to radians
    sza_rad = np.radians(sza)[:, np.newaxis]
    az_rad = np.radians(az)[:, np.newaxis]
    lat_rad = np.radians(lat)[:, np.newaxis]

    # Compute Hypsometric Heights (Integration)
    # We compute the thickness (dz) between each pressure level
    # P[i] is current, P[i+1] is next (lower pressure, higher altitude)
    p_layer_ratio = np.log(pressure[:-1] / pressure[1:])
    t_layer_avg = 0.5 * (temperature[:, 1:] + temperature[:, -1])

    # Thickness of each layer in km
    dz = (Rd * t_layer_avg / g) * p_layer_ratio / km_factor

    # Cumulative sum to get heights at each level above surface
    # We insert 0 at the start for the surface level height
    h_km = np.zeros_like(temperature)
    h_km[:, 1:] = np.cumsum(dz, axis=1)

    # Compute Horizontal Displacement (km)
    # d is the 'spread' of the ray from the vertical at height h
    d_km = h_km * np.tan(sza_rad)

    # Compute Coordinate Offsets
    # Lat/Lon shifts based on Azimuth
    dlat = (d_km * np.cos(az_rad)) / deg_to_km
    dlon = (d_km * np.sin(az_rad)) / (deg_to_km * np.cos(lat_rad))

    # Return heights, offset latitude, offset longitude
    return h_km, lat[:, np.newaxis]+dlat, lon[:, np.newaxis]+dlon


def compute_slant_path_geometry(input: DictConfig, output: DictConfig) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.

        Returns
        -------
        None.
    """

    # Load latitude, longitude, pressure levels, sensor azimuth angle, sensor zenith angle, temperature profiles
    logger.info("Loading data...")
    lat = load_var(input.lat)
    lon = load_var(input.lon)
    pressure = load_var(input.pressure)
    meta = load_var(input.meta)
    azimuth = meta[:, 3]
    zenith = meta[:, 1]
    temperature = load_var(input.prof)[:, 0]

    # Compute offset coordinates
    logger.info("Estimating slant path geometry...")
    logger.info(f"Latitude stats (min, max, mean, stdev): {lat.min()}, {lat.max()}, {lat.mean()}, {lat.std()}")
    logger.info(f"Longitude stats (min, max, mean, stdev): {lon.min()}, {lon.max()}, {lon.mean()}, {lon.std()}")
    h, lat, lon = slant_path_geometry(pressure, temperature, zenith, azimuth, lat, lon)
    logger.info(f"Height stats (min, max, mean, stdev): {h.min()}, {h.max()}, {h.mean()}, {h.std()}")
    logger.info(f"Latitude stats (min, max, mean, stdev): {lat.min()}, {lat.max()}, {lat.mean()}, {lat.std()}")
    logger.info(f"Longitude stats (min, max, mean, stdev): {lon.min()}, {lon.max()}, {lon.mean()}, {lon.std()}")

    # Save results
    logger.info("Saving data to file...")
    if hasattr(output.lat, 'save'):
        save_func = instantiate(output.lat.save)
        save_func(lat)
    if hasattr(output.lon, 'save'):
        save_func = instantiate(output.lon.save)
        save_func(lon)
    if hasattr(output.h, 'save'):
        save_func = instantiate(output.h.save)
        save_func(h)

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

    # Compute latitude and longitude offsets
    if hasattr(config.preparation, "slant"):
        for dataset, config_slant in config.preparation.slant.items():
            logger.info(f"Computing slant path offsets of {dataset} set...")
            instantiate(config_slant)

    return


if __name__ == '__main__':
    """ Compute slant path offsets.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Corrected latitude & longitude.
    """

    main()