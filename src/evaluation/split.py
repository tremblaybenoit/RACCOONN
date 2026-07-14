import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from src.evaluation.plot import fig_geostationnary, save_plot
from src.data.io import load_variable
from matplotlib.colors import ListedColormap

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
    lat_train = load_variable(config.data.stage.train.vars.lat)
    lon_train = load_variable(config.data.stage.train.vars.lon)
    mask_train = np.ones_like(lon_train, dtype=int)*0
    lat_valid = load_variable(config.data.stage.valid.vars.lat)
    lon_valid = load_variable(config.data.stage.valid.vars.lon)
    mask_valid = np.ones_like(lon_valid, dtype=int)*1
    lat_test = load_variable(config.data.stage.test.vars.lat)
    lon_test = load_variable(config.data.stage.test.vars.lon)
    mask_test = np.ones_like(lon_test, dtype=int)*2
    lat = np.concatenate([lat_train, lat_valid, lat_test], axis=0)
    lon = np.concatenate([lon_train, lon_valid, lon_test], axis=0)
    mask = np.concatenate([mask_train, mask_valid, mask_test], axis=0)

    # Plot spatial distribution of training, validation and test sets
    logger.info("Plotting spatial distribution of training, validation and test sets...")
    colors = ['#56B4E9','#E69F00', '#009E73']
    cmap = ListedColormap(colors)
    fig = fig_geostationnary(lon, lat, mask, mask.min(), mask.max(), title='(a) Spatial distribution of the data',
                             cb_cmap=cmap, cb_ticks=3, cb_label='', cb_ticklabels=['Train', 'Valid', 'Test'], markersize=20.)
    save_plot(fig, os.path.join(config.paths.data_dir, f'split.png'))

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
