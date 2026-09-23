import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import os
from code.evaluation.plot import fig_vertical_profiles, save_plot
from code.data.io import load_variable
from tqdm import tqdm

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
    config0 = config.data.stage.train.variables.prof
    prof_train0 = load_variable(config0)
    #config1 = config.data.stage.train.variables.prof
    #prof_train1 = load_variable(config1)
    #prof_train1[:, 0] = np.log10(prof_train1[:, 0])  # Convert to log10
    #prof_train1[:, 1] = np.log10(prof_train1[:, 1])  # Convert to log10
    #prof_train1[:, 2] = np.log10(prof_train1[:, 2])  # Convert to log10
    #config2 = config.data.stage.train.variables.prof
    #prof_train2 = load_variable(config2)
    #prof_train2[:, 0] = np.log(prof_train2[:, 0])  # Convert to log10
    #prof_train2[:, 1] = np.log(prof_train2[:, 1])  # Convert to log10
    #prof_train2[:, 2] = np.log(prof_train2[:, 2])  # Convert to log10
    config1 = config.data.stage.train.variables.prof
    config1.transformations.normalization._target_ = 'code.data.transformations.min_max'
    config1.transformations.normalization.axis = 1
    prof_train1 = load_variable(config1, apply_transform=True)
    config2 = config.data.stage.train.variables.prof
    config2.transformations.normalization._target_ = 'code.data.transformations.min_max'
    config2.transformations.normalization.axis = None
    prof_train2 = load_variable(config2, apply_transform=True)
    print(prof_train2.max())
    config3 = config.data.stage.train.variables.prof
    config3.transformations.normalization._target_ = 'code.data.transformations.mean_stdev'
    config3.transformations.normalization.axis = None
    prof_train3 = load_variable(config3, apply_transform=True)
    # Stack profiles and compute statistics
    logger.info("Computing profile statistics...")
    prof = np.concatenate([prof_train0, prof_train1, prof_train2, prof_train3], axis=1)
    prof_mean = np.mean(prof, axis=0)
    prof_stdev = np.std(prof, axis=0)
    prof_types = config.data.stage.test.variables.prof.type
    prof_labels = ([f'No norm. - {prof_label}' for prof_label in prof_types] +
                   [f'Min-Max norm. 1 - {prof_label}' for prof_label in prof_types] +
                   [f'Min-Max norm. N - {prof_label}' for prof_label in prof_types] +
                   [f'Standardized. N - {prof_label}' for prof_label in prof_types])
    pressure_levels = load_variable(config.data.stage.train.variables.pressure)

    # Plot profiles
    logger.info("Plotting profiles...")
    fig0 = fig_vertical_profiles([prof_mean], ['Data'], stdev=[prof_stdev],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Normalized profile value (no units)',
                                 title=prof_labels)
    save_plot(fig0, 'prof.png')


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
