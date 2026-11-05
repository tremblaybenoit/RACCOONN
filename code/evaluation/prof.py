import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from code.evaluation.plot import fig_vertical_profiles, save_plot
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
    prof_train = load_variable(config.data.stage.train.variables.prof)
    prof_prior_train = load_variable(config.data.stage.train.variables.prof_prior)
    prof_labels = config.data.stage.train.variables.prof.type
    pressure = load_variable(config.data.stage.train.variables.pressure)
    color = ['#56B4E9', '#009E73']
    label = ['Target', 'Prior']

    prof_valid = load_variable(config.data.stage.valid.variables.prof)
    prof_prior_valid = load_variable(config.data.stage.valid.variables.prof_prior)

    prof_test = load_variable(config.data.stage.test.variables.prof)
    prof_prior_test = load_variable(config.data.stage.test.variables.prof_prior)

    # Concatenate all data
    prof = np.concatenate([prof_train, prof_valid, prof_test], axis=0)
    prof_prior = np.concatenate([prof_prior_train, prof_prior_valid, prof_prior_test], axis=0)
    n_prof = prof.shape[0]

    # Loop over scans and plot spatial distribution of train/valid/test splits for each scan
    output_dir = os.path.join(config.paths.data_dir, 'figures/prof')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in tqdm(range(n_prof)):
        # logger.info(f"Plotting train/valid/test split for scan {i}...")
        fig = fig_vertical_profiles(
                        [prof[i, :], prof_prior[i, :]],
                        label=label,
                        color=color,
                        y=pressure,
                        y_label='Pressure (hPa)',
                        x_label='Profile value',
                        title=[f"Profile: {label}" for label in prof_labels]
                    )
        save_plot(fig, os.path.join(output_dir, f'sample_{i:04d}.png'))

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
