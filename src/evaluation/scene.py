import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from src.evaluation.plot import fig_geostationnary, save_plot
from src.data.io import load_variable
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Assess the spatial distribution of the training, validation and test sets at a given timestep.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load variables for training, validation and test sets
    logger.info("Loading coordinates for training, validation and test sets...")
    lat_train = load_variable(config.data.stage.train.variables.lat)
    lon_train = load_variable(config.data.stage.train.variables.lon)
    scans_train = load_variable(config.data.stage.train.variables.scans)
    mask_train = load_variable(config.data.stage.train.variables.cloud_filter)
    lat_valid = load_variable(config.data.stage.valid.variables.lat)
    lon_valid = load_variable(config.data.stage.valid.variables.lon)
    scans_valid = load_variable(config.data.stage.valid.variables.scans)
    mask_valid = load_variable(config.data.stage.valid.variables.cloud_filter)
    lat_test = load_variable(config.data.stage.test.variables.lat)
    lon_test = load_variable(config.data.stage.test.variables.lon)
    scans_test = load_variable(config.data.stage.test.variables.scans)
    mask_test = load_variable(config.data.stage.test.variables.cloud_filter)
    lat = np.concatenate([lat_train, lat_valid, lat_test], axis=0)
    lon = np.concatenate([lon_train, lon_valid, lon_test], axis=0)
    scans = np.concatenate([scans_train, scans_valid, scans_test], axis=0)
    mask = np.concatenate([mask_train, mask_valid, mask_test], axis=0).astype(int)

    # Number of samples, scans, coordinates
    n_samples = scans.shape[0]
    n_scans = np.unique(scans).shape[0]
    n_coords = n_samples // n_scans
    scans_order = np.argsort(scans, kind='stable')
    # Reorder coordinates according to scans
    lat = lat[scans_order]
    lon = lon[scans_order]
    scans = scans[scans_order]
    mask = mask[scans_order]
    # Reshape data to have one sample per scan
    lat = lat.reshape(-1, n_coords)
    lon = lon.reshape(-1, n_coords)
    scans = scans.reshape(-1, n_coords)
    mask = mask.reshape(-1, n_coords)

    # Loop over scans and plot spatial distribution of clouds for each scan
    if not os.path.exists(os.path.join(config.paths.data_dir, 'scenes')):
        os.mkdir(os.path.join(config.paths.data_dir, 'scenes'))
    for i in tqdm(range(n_scans)):
        logger.info(f"Plotting spatial distribution of clouds for scan {i}...")
        colors = ['#56B4E9','#E69F00']
        cmap = ListedColormap(colors)
        fig = fig_geostationnary(lon[i], lat[i], mask[i], 0, 1, title=f'Spatial distribution of clouds for scan {i}',
                                 cb_cmap=cmap, cb_ticks=2, cb_ticklabels=['Clear', 'Cloud'], markersize=1.)
        save_plot(fig, os.path.join(config.paths.data_dir, f'scenes/scan_{i:02d}.png'))

if __name__ == '__main__':
    """ Evaluate spatial distribution of data.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Plots.
    """

    main()
