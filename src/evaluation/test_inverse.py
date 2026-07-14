import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from data.statistics import statistics
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from src.evaluation.plot import fig_rmse_bars3, fig_vertical_profiles3, save_plot

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
    prof_prior = instantiate(config.data.stage.test.variables.prof_background.load)
    hofx_pred = instantiate(config.loader.stage.test.results.hofx.load)
    hofx_prior = instantiate(config.data.stage.test.variables.hofx_background.load)

    # Load test set references
    logger.info("Load test set references...")
    prof = np.array(instantiate(config.data.stage.test.variables.prof.load)).astype(np.float32)
    hofx = np.array(instantiate(config.data.stage.test.variables.hofx.load)).astype(np.float32)
    pressure = np.array(instantiate(config.data.stage.test.variables.pressure.load)).astype(np.float32)

    # Create masks and compute rmse by condition
    stats_prof = statistics(prof, axis=0, which=['mean', 'stdev', 'rmse'], target=prof)
    stats_pred = statistics(prof_pred, axis=0, which=['mean', 'stdev', 'rmse'], target=prof)
    stats_prior = statistics(prof_prior, axis=0, which=['mean', 'stdev', 'rmse'], target=prof)
    stats_hofx = statistics(hofx_pred[:, :10], axis=0, which=['rmse'], target=hofx[:, :10])
    stats_hofx_prior = statistics(hofx_prior[:, :10], axis=0, which=['rmse'], target=hofx[:, :10])

    # Plots
    logger.info("Plot comparison...")
    fig2 = fig_rmse_bars3([stats_hofx['rmse'], stats_hofx_prior['rmse']],
                          x_range=[[0, 2.65]], labels=list(['Prediction', 'Prior']),
                          title=["(b) Test set - Forward model RMSE"],
                          # colors=['#ff7f0e', '#2ca02c'])
                          colors=['#E69F00', '#009E73'])
    save_plot(fig2, config.paths.run_dir + '/Figure2_rmse_bars2_test.png')

    prof_mean_labels, prof_mean_colors = ['Target', 'Prediction'], ['#56B4E9', '#E69F00']  # ['#1f77b4', '#ff7f0e']
    prof_rmse_labels, prof_rmse_colors = ['Target-Prediction'], ['#E69F00']  # ['#ff7f0e']
    prof_mean_labels.insert(0, 'Prior')
    prof_mean_colors.insert(0, '#009E73')  # '#2ca02c')
    prof_rmse_labels.insert(0, 'Target-Prior')
    prof_rmse_colors.insert(0, '#009E73')  # '#2ca02c')

    # Profile Mean
    prof_labels = ['(c) Air temperature profile',
                   '(d) Humidity mixing ratio profile',
                   '(e) Ozone mixing ratio profile']
    x_labels = ['Profile (K)', 'Profile (g/kg)', 'Profile (ppmv)']
    fig0 = fig_vertical_profiles3([stats_prior['mean'], stats_prof['mean'], stats_pred['mean']], prof_mean_labels,
                                 stdev=[stats_prior['stdev'], stats_prof['stdev'], stats_pred['stdev']],
                                 y=pressure, y_label='Pressure (hPa)',
                                 x_label=x_labels, color=prof_mean_colors,
                                 title=[f"{prof_label}" for prof_label in prof_labels])
    save_plot(fig0, config.paths.run_dir + '/Figure0_profile_test.png')
    prof_labels = ['(f) Air temperature RMSE',
                   '(g) Humidity mixing ratio RMSE',
                   '(h) Ozone mixing ratio RMSE']
    # Profile RMSE
    x_labels = ['RMSE (K)', 'RMSE (g/kg)', 'RMSE (ppmv)']
    fig1 = fig_vertical_profiles3([stats_prior['rmse'], stats_pred['rmse']], prof_rmse_labels, y=pressure, y_label='Pressure (hPa)',
                                 x_label=x_labels, color=prof_rmse_colors,
                                 title=[f"{prof_label}" for prof_label in prof_labels])
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
