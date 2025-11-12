import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from data.statistics import statistics
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from utilities.plot import fig_rmse_bars, fig_errs_by_channel, save_plot

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
    cloud_filter = instantiate(config.data.stage.test.vars.cloud_filter.load)
    meta = instantiate(config.data.stage.test.vars.meta.load).astype(np.float32)

    # Create masks and compute rmse by condition
    mask = {
        'Clear sky': ~cloud_filter,
        'Cloudy': cloud_filter,
        'Day': meta[:, 6] < 90,
        'Night': meta[:, 6] >= 90
    }
    stats = {'hofx': {}, 'hofx_norm': {}}
    for key, m in mask.items():
        # Compute rmse
        stats['hofx'][key] = statistics(pred[m, :10], axis=0, which=['rmse'], target=hofx[m, :10])
        stats['hofx_norm'][key] = statistics(pred[m, :10] / pred[m, 10:], axis=0, which=['rmse'],
                                             target=hofx[m, :10] / pred[m, 10:])

    # Plots
    logger.info("Plot comparison...")
    fig2 = fig_rmse_bars([stats['hofx'][key]['rmse'] for key in stats['hofx'].keys()],
                         [stats['hofx_norm'][key]['rmse'] for key in stats['hofx_norm'].keys()],
                         x_range=[[0, 1.5], [0, 2.0]], labels=list(stats['hofx'].keys()),
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
