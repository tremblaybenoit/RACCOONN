import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from utilities.plot import fig_vertical_profiles, save_plot
from data.io import load_var, load_var_and_normalize

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Validation predictions made by the inverse model.
        Compare against the test set expected values.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load profiles with different normalization methods
    logger.info("Loading profiles with different normalization methods...")
    config0 = config.data.stage.train.vars.prof
    prof_train0 = load_var(config0)
    config1 = config.data.stage.train.vars.prof
    config1.normalization.axis = 1
    prof_train1 = load_var_and_normalize(config1)
    config2 = config.data.stage.train.vars.prof
    config2.normalization._target_ = 'data.transformations.mean_stdev'
    config2.normalization.axis = None
    prof_train2 = load_var_and_normalize(config2)
    # Stack profiles and compute statistics
    logger.info("Computing profile statistics...")
    prof = np.concatenate([prof_train0, prof_train1, prof_train2], axis=1)
    prof_mean = np.mean(prof, axis=0)
    prof_stdev = np.std(prof, axis=0)
    prof_types = config.data.stage.test.vars.prof.type
    prof_labels = ([f'No norm. - {prof_label}' for prof_label in prof_types] +
                   [f'Min-Max norm. - {prof_label}' for prof_label in prof_types] +
                   [f'Standardized. - {prof_label}' for prof_label in prof_types])
    pressure_levels = load_var(config.data.stage.train.vars.pressure)/100.0  # Convert to hPa

    # Plot profiles
    logger.info("Plotting profiles...")
    fig0 = fig_vertical_profiles([prof_mean], ['Data'], stdev=[prof_stdev],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Normalized profile value (no units)',
                                 title=prof_labels)
    save_plot(fig0, os.path.join(config.paths.data_dir, 'normalization/prof.png'))


if __name__ == '__main__':
    """ Predict using the inverse model.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        checkpoint: Training weights & biases.
    """

    main()
