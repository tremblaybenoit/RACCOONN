import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from utilities.plot import fig_scatterplots, fig_coordinate_distributions, save_plot
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

    # Load coordinates for the different stages
    logger.info("Loading profiles with different normalization methods...")
    config_stage = {'train': config.data.stage.train.vars,
                    'valid': config.data.stage.valid.vars,
                    'test': config.data.stage.test.vars}
    coords = {}
    for stage_name, stage_cfg in config_stage.items():
        coords[stage_name] = {}
        for var_name in ('lat', 'lon', 'scans'):
            if var_name not in coords:
                coords[stage_name][var_name] = load_var(stage_cfg[var_name])

    # Create coordinate distribution plots
    fig1 = fig_coordinate_distributions(coords['train']['lat'], coords['train']['lon'],
                                        coords['valid']['lat'], coords['valid']['lon'])
    # Save figure
    save_plot(fig1, os.path.join(config.paths.data_dir, 'normalization/coord_distributions.png'))

    breakpoint()

    # Create scatterplot latitude vs longitude
    fig0 = fig_scatterplots([c['lon'] for c in coords.values()],
                            [c['lat'] for c in coords.values()],
                            x_label='Longitude', y_label='Latitude',
                            labels=list(coords.keys()),
                            title='Coordinates distribution: Latitude vs Longitude')
    # Save figure
    save_plot(fig0, os.path.join(config.paths.data_dir, 'normalization/latlon.png'))

    breakpoint()


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
