import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from code.evaluation.plot import fig_geostationnary, save_plot
from code.data.io import load_variable
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Assess the spatial distribution of the training, validation and test sets at a given timestep.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load variables for training, validation and test sets
    logger.info("Loading coordinates for training, validation and test sets...")
    lat = np.asarray(load_variable(config.data.stage.all.variables.lat))
    lon = np.asarray(load_variable(config.data.stage.all.variables.lon))
    scans = np.asarray(load_variable(config.data.stage.all.variables.scans))
    mask = np.asarray(load_variable(config.data.stage.all.variables.cloud_mask)).astype(int)

    # Get unique scans
    unique_scans = np.unique(scans)
    n_scans = unique_scans.shape[0]

    # Loop over scans and plot spatial distribution of clouds for each scan
    save_dir = os.path.join(config.paths.data_dir, 'figures/scenes')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    colors = ['#56B4E9', '#E69F00']
    cmap = ListedColormap(colors)
    for i in tqdm(range(n_scans)):
        # Get indices for current scan
        scan_idx = unique_scans[i]
        indices = np.where(scans == scan_idx)[0]
        # Get coordinates for current scan
        lat_scan = lat[indices]
        lon_scan = lon[indices]
        mask_scan = mask[indices]
        # logger.info(f"Plotting spatial distribution of clouds for scan {i}...")
        fig = fig_geostationnary(lon_scan, lat_scan, mask_scan, 0, 1, title=f'Spatial distribution of clouds for scan {i}',
                                 cb_cmap=cmap, cb_ticks=2, cb_ticklabels=['Clear', 'Cloud'], markersize=1.)
        save_plot(fig, os.path.join(save_dir, f'scan_{i:02d}.png'))

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
