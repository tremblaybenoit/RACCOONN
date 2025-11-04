import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from utilities.plot import fig_rmse_bars, fig_rmse_bars2, fig_errs_by_channel, save_plot

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

    # Load test set results (predictions)
    logger.info("Load test set results...")
    pred = instantiate(config.loader.stage.test.results.hofx.load)

    # Load test set references
    logger.info("Load test set references...")
    hofx = np.array(instantiate(config.data.stage.test.vars.hofx.load)).astype(np.float32)
    clear = ~instantiate(config.data.stage.test.vars.cloud_filter.load)
    meta = instantiate(config.data.stage.test.vars.meta.load).astype(np.float32)
    daytime = meta[:, 6] < 90

    # Plots
    logger.info("Plot comparison...")
    fig1 = fig_rmse_bars(hofx, pred, clear, x_range=[[0, 1.0], [0, 2.0]],
                         title=["(a) Forward model - Forward model errors",
                                "(b) Forward model - Normalized forward model errors"])
    save_plot(fig1, config.paths.run_dir + '/Figure3_rmse_bars_test.png')
    fig2 = fig_rmse_bars2(hofx, pred, clear, daytime, x_range=[[0, 1.5], [0, 2.0]],
                          title=["(a) Forward model - Daytime forward model errors",
                                 "(b) Forward model - Nighttime forward model errors"])
    save_plot(fig2, config.paths.run_dir + '/Figure2_rmse_bars2_test.png')
    fig3 = fig_errs_by_channel(hofx, pred, title=[f'Channel {i+7}' for i in range(hofx.shape[1])],
                               orientation='horizontal', x_range=[[0, 7.0], [0, 0.9], [0, 1.1], [0, 1.3], [0, 1.9],
                                                                  [0, 1.0], [0, 1.6], [0, 1.8], [0, 2.1], [0, 1.6]])
    save_plot(fig3, config.paths.run_dir + '/Figure4_errs_by_channel_vert_test.png')


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
