import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from utilities.logic import get_config_path
import os
from utilities.plot import fig_vertical_profiles, save_plot
from data.io import load_var, load_var_and_normalize
from utilities.instantiators import instantiate

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

    # Load pca buffers
    pca_buffers = instantiate(config.model.pca_buffers)

    # Load profiles with different normalization methods
    logger.info("Loading profiles with different normalization methods...")
    prof = np.concatenate([load_var(prof_config) for prof_config in [config.data.stage.train.vars.prof,
                                                                     config.data.stage.valid.vars.prof,
                                                                     config.data.stage.test.vars.prof]], axis=0)
    prof_background = np.concatenate([load_var(prof_config) for prof_config in [config.data.stage.train.vars.prof_background,
                                                                                config.data.stage.valid.vars.prof_background,
                                                                                config.data.stage.test.vars.prof_background]], axis=0)
    prof_norm = np.concatenate([load_var_and_normalize(prof_config) for prof_config in [config.data.stage.train.vars.prof,
                                                                                        config.data.stage.valid.vars.prof,
                                                                                        config.data.stage.test.vars.prof]], axis=0)
    prof_norm_mean = np.mean(prof_norm, axis=0)
    prof_norm_stdev = np.std(prof_norm, axis=0) + 1.e-12
    prof_norm_standardized = (prof_norm - prof_norm_mean) / prof_norm_stdev
    prof_norm_standardized_mean = np.mean(prof_norm_standardized, axis=0)
    prof_norm_standardized_stdev = np.std(prof_norm_standardized, axis=0)
    prof_norm_background = np.concatenate([load_var_and_normalize(prof_config) for prof_config in [config.data.stage.train.vars.prof_background,
                                                                                                   config.data.stage.valid.vars.prof_background,
                                                                                                   config.data.stage.test.vars.prof_background]], axis=0)
    prof_norm_background_mean = np.mean(prof_norm_background, axis=0)
    prof_norm_background_stdev = np.std(prof_norm_background, axis=0) + 1.e-12
    prof_norm_background_standardized = (prof_norm_background - prof_norm_background_mean) / prof_norm_background_stdev
    prof_norm_background_standardized_mean = np.mean(prof_norm_background_standardized, axis=0)
    prof_norm_background_standardized_stdev = np.std(prof_norm_background_standardized, axis=0)

    # Load whitemed coefficients
    w_white = np.concatenate([load_var(config.data.stage.train.vars.prof_white),
                              load_var(config.data.stage.valid.vars.prof_white),
                              load_var(config.data.stage.test.vars.prof_white)], axis=0)
    w_white_background = np.concatenate([load_var(config.data.stage.train.vars.prof_white_background),
                                       load_var(config.data.stage.valid.vars.prof_white_background),
                                       load_var(config.data.stage.test.vars.prof_white_background)], axis=0)

    # 3. Profile Reconstruction
    w_standardized = w_white * pca_buffers['scales']
    wpca_norm_standardized = w_standardized @ pca_buffers['basis']   # torch.matmul(w_standardized, pca_buffers['basis'])
    wpca_norm_standardized_mean = np.mean(wpca_norm_standardized, axis=0).reshape(prof_norm_mean.shape)
    wpca_norm_standardized_stdev = np.std(wpca_norm_standardized, axis=0).reshape(prof_norm_mean.shape) + 1.e-12
    wpca_norm = (pca_buffers['mu'] + (wpca_norm_standardized *pca_buffers['std']))
    wpca_norm_mean = np.mean(wpca_norm, axis=0).reshape(prof_norm_background_mean.shape)
    wpca_norm_stdev = np.std(wpca_norm, axis=0).reshape(prof_norm_background_mean.shape) + 1.e-12
    # TODO: Handle true background. Currently using the mean.
    w_standardized_background = np.zeros_like(w_white_background) * pca_buffers['scales']
    wpca_norm_background_standardized = w_standardized_background @ pca_buffers['basis']   # torch.matmul(w_standardized_background, pca_buffers['basis'])
    wpca_norm_background_standardized_mean = np.mean(wpca_norm_background_standardized, axis=0).reshape(prof_norm_background_mean.shape)
    wpca_norm_background_standardized_stdev = np.std(wpca_norm_background_standardized, axis=0).reshape(prof_norm_background_mean.shape)  + 1.e-12
    wpca_norm_background = (pca_buffers['mu'] + (wpca_norm_background_standardized *pca_buffers['std']))
    wpca_norm_background_mean = np.mean(wpca_norm_background, axis=0).reshape(prof_norm_background_mean.shape)
    wpca_norm_background_stdev = np.std(wpca_norm_background, axis=0).reshape(prof_norm_background_mean.shape) + 1.e-12
    breakpoint()

    # Stack profiles and compute statistics
    logger.info("Computing profile statistics...")
    prof_types = config.data.stage.test.vars.prof.type
    prof_labels = ([f'{prof_label}' for prof_label in prof_types])
    pressure_levels = load_var(config.data.stage.train.vars.pressure)/100.0  # Convert to hPa

    # Plot profiles
    logger.info("Plotting profiles...")
    fig0 = fig_vertical_profiles([prof_norm_mean, wpca_norm_mean, prof_norm_background_mean, wpca_norm_background_mean],
                                 ['Data', 'PCA', 'Background', 'PCA Background'],
                                 stdev=[prof_norm_stdev, wpca_norm_stdev, prof_norm_background_stdev, wpca_norm_background_stdev],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Normalized profile value (no units)',
                                 title=prof_labels)
    save_plot(fig0, os.path.join(config.paths.data_dir, 'normalization/prof_wpca0.png'))
    fig1 = fig_vertical_profiles([prof_norm_standardized_mean, wpca_norm_standardized_mean, prof_norm_background_standardized_mean, wpca_norm_background_standardized_mean],
                                 ['Data', 'PCA', 'Background', 'PCA Background'],
                                 stdev=[prof_norm_standardized_stdev, wpca_norm_standardized_stdev, prof_norm_background_standardized_stdev, wpca_norm_background_standardized_stdev],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Normalized profile value (no units)',
                                 title=prof_labels)
    save_plot(fig1, os.path.join(config.paths.data_dir, 'normalization/prof_wpca1.png'))
    # Plot RMSE between data and PCA reconstructions
    fig2 = fig_vertical_profiles([np.sqrt(np.mean((prof_norm - prof_norm)**2, axis=0)),
                                 np.sqrt(np.mean((prof_norm - wpca_norm.reshape(prof_norm.shape))**2, axis=0)),
                                 np.sqrt(np.mean((prof_norm - prof_norm_background_mean)**2, axis=0)),
                                 np.sqrt(np.mean((prof_norm - wpca_norm_background_mean)**2, axis=0))],
                                 ['Data-Data', 'Data-PCA', 'Data-Background', 'Data-Background-PCA'],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Profile RMSE (no units)',
                                 title=prof_labels)
    save_plot(fig2, os.path.join(config.paths.data_dir, 'normalization/prof_wpca2.png'))
    fig3 = fig_vertical_profiles([np.sqrt(np.mean((prof_norm_standardized - prof_norm_standardized)**2, axis=0)),
                                 np.sqrt(np.mean((prof_norm_standardized - wpca_norm_standardized.reshape(prof_norm.shape))**2, axis=0)),
                                 np.sqrt(np.mean((prof_norm_standardized - prof_norm_background_standardized_mean)**2, axis=0)),
                                 np.sqrt(np.mean((prof_norm_standardized - wpca_norm_background_standardized_mean)**2, axis=0))],
                                 ['Data-Data', 'Data-PCA', 'Data-Background', 'Data-Background-PCA'],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Profile RMSE (no units)',
                                 title=prof_labels)
    save_plot(fig3, os.path.join(config.paths.data_dir, 'normalization/prof_wpca3.png'))
    fig4 = fig_vertical_profiles([np.sqrt(np.mean((prof_norm - prof_norm) ** 2, axis=0)),
                                  np.sqrt(np.mean((prof_norm - wpca_norm.reshape(prof_norm.shape)) ** 2, axis=0))],
                                 ['Data-Data', 'Data-PCA'],
                                 y=pressure_levels, y_label='Pressure (hPa)',
                                 x_label='Profile RMSE (no units)',
                                 title=prof_labels)
    save_plot(fig4, os.path.join(config.paths.data_dir, 'normalization/prof_wpca4.png'))
    fig5 = fig_vertical_profiles([np.sqrt(np.mean((prof_norm_standardized - prof_norm_standardized) ** 2, axis=0)),
                                  np.sqrt(np.mean((prof_norm_standardized - wpca_norm_standardized.reshape(prof_norm.shape)) ** 2, axis=0))],
                                 ['Data-Data', 'Data-PCA'],
                                  y=pressure_levels, y_label='Pressure (hPa)',
                                  x_label='Profile RMSE (no units)',
                                  title=prof_labels)
    save_plot(fig5, os.path.join(config.paths.data_dir, 'normalization/prof_wpca5.png'))


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
