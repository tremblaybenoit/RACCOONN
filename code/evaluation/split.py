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
    """ Assess the spatial distribution of the training, validation and test sets per scan.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load variables for training, validation and test sets
    logger.info("Loading coordinates and scan indices for training, validation and test sets...")
    lat_train = load_variable(config.data.stage.train.variables.lat)
    lon_train = load_variable(config.data.stage.train.variables.lon)
    scans_train = load_variable(config.data.stage.train.variables.scans)
    mask_train = np.ones_like(lon_train, dtype=int) * 0  # 0 = Train

    lat_valid = load_variable(config.data.stage.valid.variables.lat)
    lon_valid = load_variable(config.data.stage.valid.variables.lon)
    scans_valid = load_variable(config.data.stage.valid.variables.scans)
    mask_valid = np.ones_like(lon_valid, dtype=int) * 1  # 1 = Valid

    lat_test = load_variable(config.data.stage.test.variables.lat)
    lon_test = load_variable(config.data.stage.test.variables.lon)
    scans_test = load_variable(config.data.stage.test.variables.scans)
    mask_test = np.ones_like(lon_test, dtype=int) * 2  # 2 = Test

    # Concatenate all data
    lat = np.concatenate([lat_train, lat_valid, lat_test], axis=0)
    lon = np.concatenate([lon_train, lon_valid, lon_test], axis=0)
    scans = np.concatenate([scans_train, scans_valid, scans_test], axis=0)
    mask = np.concatenate([mask_train, mask_valid, mask_test], axis=0).astype(int)

    # Get unique scans
    unique_scans = np.unique(scans)
    n_scans = unique_scans.shape[0]

    # Loop over scans and plot spatial distribution of train/valid/test splits for each scan
    output_dir = os.path.join(config.paths.data_dir, 'figures/splits')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    colors = ['#56B4E9', '#E69F00', '#009E73']
    cmap = ListedColormap(colors)
    for i in tqdm(range(n_scans)):
        # Get indices for current scan
        scan_idx = unique_scans[i]
        indices = np.where(scans == scan_idx)[0]
        # Get coordinates for current scan
        lat_scan = lat[indices]
        lon_scan = lon[indices]
        mask_scan = mask[indices]
        # logger.info(f"Plotting train/valid/test split for scan {i}...")
        fig = fig_geostationnary(lon_scan, lat_scan, mask_scan, 0, 2,
                                 title=f'Train/Valid/Test split for scan {i}',
                                 cb_cmap=cmap, cb_ticks=3, cb_ticklabels=['Train', 'Valid', 'Test'],
                                 markersize=1.)
        save_plot(fig, os.path.join(output_dir, f'scan_{i:02d}.png'))

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
