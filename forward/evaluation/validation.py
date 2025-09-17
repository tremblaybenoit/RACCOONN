import numpy as np
import logging
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from forward.utilities.logic import get_config_path
from forward.utilities.instantiators import instantiate
from inverse.utilities.plot import fig_rmse_bars

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Validation predictions made by the Pytorch Lightning version of the CRTM emulator.
        Compare against the CRTM model test set expected values.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Evaluate on test set (i.e., the inputs of the prediction are from the test set)
    logger.info("Load test set results...")
    pred = instantiate(config.loader.stage.test.results.hofx.load)

    # Load additional data from the test set
    logger.info("Load test set references...")
    prof = np.array(instantiate(config.data.stage.test.vars.prof.load)).astype(np.float32)
    hofx = np.array(instantiate(config.data.stage.test.vars.hofx.load)).astype(np.float32)
    clear = np.logical_and(prof[:, 5, :].sum(axis=1) == 0, prof[:, 6, :].sum(axis=1) == 0)
    clear = np.logical_and(clear, prof[:, 7, :].sum(axis=1) == 0)

    # Plot
    logger.info("Plot comparison...")
    fig_rmse_bars(hofx, pred, clear, title=["(a) Test Set - Forward model errors",
                                            "(b) Test Set - Normalized forward model errors"])
    plt.savefig(config.paths.run_dir + '/rmse_bars_test.png')


if __name__ == '__main__':
    """ Predict using the CRTM emulator.

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
