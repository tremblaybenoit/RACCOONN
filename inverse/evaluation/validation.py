import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from data.statistics import statistics
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from utilities.plot import fig_rmse_bars, fig_vertical_profiles, save_plot

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

    # Load test set results (predictions)
    logger.info("Load test set results...")
    prof_pred = instantiate(config.loader.stage.test.results.prof.load)
    prof_prior = instantiate(config.loader.stage.test.results.prof_background.load)
    hofx_pred = instantiate(config.loader.stage.test.results.hofx.load)
    hofx_config = config.loader.stage.test.results.hofx.load
    hofx_config.path = ''
    hofx_prior = instantiate(hofx_config)

    # Load test set references
    logger.info("Load test set references...")
    prof = np.array(instantiate(config.data.stage.test.vars.prof.load)).astype(np.float32)
    hofx = np.array(instantiate(config.data.stage.test.vars.hofx.load)).astype(np.float32)
    pressure = 0.01*10**np.array(instantiate(config.data.stage.test.results.pressure.load)).astype(np.float32)

    # Create masks and compute rmse by condition
    stats_prof = statistics(prof_pred, axis=0, which=['mean', 'stdev', 'rmse'], target=prof)
    stats_prior = statistics(prof_prior, axis=0, which=['mean', 'stdev', 'rmse'], target=prof)
    stats_hofx = statistics(hofx_pred[:, :10], axis=0, which=['rmse'], target=hofx[:, :10])
    stats_hofx_prior = statistics(hofx_prior[:, :10], axis=0, which=['rmse'], target=hofx[:, :10])

    # Plots
    logger.info("Plot comparison...")
    fig2 = fig_rmse_bars([stats_hofx['rmse'], stats_hofx_prior['rmse']],
                         x_range=[[0, 1.5]], labels=list(['Prediction', 'Prior']),
                         title=["(b) Test set - Forward model RMSE"])
    save_plot(fig2, config.paths.run_dir + '/Figure2_rmse_bars2_test.png')

    prof_mean_labels, prof_mean_colors = ['Target', 'Prediction'], ['#1f77b4', '#ff7f0e']
    prof_rmse_labels, prof_rmse_colors = ['Target-Prediction'], ['#ff7f0e']
    prof_mean_labels.insert(0, 'Prior')
    prof_mean_colors.insert(0, '#2ca02c')
    prof_rmse_labels.insert(0, 'Target-Prior')
    prof_rmse_colors.insert(0, '#2ca02c')


    # Profile Mean
    prof_labels = ['(c) Test set - Air temperature profile',
                   '(d) Test set - Humidity mixing ratio profile',
                   '(e) Test set - Ozone mixing ratio profile']
    fig0 = fig_vertical_profiles([stats_prior['mean'], stats_prof['mean']], prof_mean_labels,
                                 stdev=[stats_prior['stdev'], stats_prof['stdev']],
                                 y=pressure, y_label='Pressure (hPa)',
                                 x_label='Normalized profile value (no units)', color=prof_mean_colors,
                                 title=[f"Test set - {prof_label}" for prof_label in prof_labels])
    save_plot(fig0, config.paths.run_dir + '/Figure0_profile_test.png')
    prof_labels = ['(c) Test set - Air temperature RMSE',
                   '(d) Test set - Humidity mixing ratio RMSE',
                   '(e) Test set - Ozone mixing ratio RMSE']
    # Profile RMSE
    fig1 = fig_vertical_profiles([stats_prior['rmse'], stats_prof['rmse']], prof_rmse_labels, y=pressure, y_label='Pressure (hPa)',
                                 x_label='Normalized profile RMSE (no units)', color=prof_rmse_colors,
                                 title=[f"Test set - {prof_label}" for prof_label in prof_labels])
    save_plot(fig1, config.paths.run_dir + '/Figure0_profile_rmse_test.png')

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
