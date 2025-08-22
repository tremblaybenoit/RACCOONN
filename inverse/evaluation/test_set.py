import hydra
from omegaconf import DictConfig
import numpy as np
from forward.utilities.logic import get_config_path
from forward.utilities.instantiators import instantiate
from inverse.utilities.plot import fig_vertical_profiles, fig_vertical_profile, fig_rmse_bars, save_plot
from inverse.data.transformations import min_max
import pickle


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Test neural network based on set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load reference data from the test set
    prof_target = np.array(instantiate(config.data.sets.test.prof.load)).astype(np.float32)[342140:, ...]
    bt_target = np.array(instantiate(config.data.sets.test.hofx.load)).astype(np.float32)[342140:, ...]
    clrsky = np.logical_and(prof_target[:, 5, :].sum(axis=1) == 0, prof_target[:, 6, :].sum(axis=1) == 0)
    clrsky = np.logical_and(clrsky, prof_target[:, 7, :].sum(axis=1) == 0)
    with open(config.model.parameters.data.statistics.path, 'rb') as file:
        stats = pickle.load(file)
    stats['prof']['max'] = stats['prof']['max'] + np.finfo(float).eps
    normalize_profiles = instantiate(min_max, stats=stats['prof'], _partial_=True)

    # Load test set results
    if hasattr(config.data.sets.test.results, 'hofx'):
        bt_pred = instantiate(config.data.sets.test.results.hofx.load)
    else:
        bt_pred = None
    if hasattr(config.data.sets.test.results, 'prof'):
        prof_pred = instantiate(config.data.sets.test.results.prof.load)
    else:
        prof_pred = None

    # Compute errors
    prof_target_tmp = prof_target.copy()
    prof_target_tmp[np.where(prof_target_tmp <= 0)] = np.nan
    std = np.nanmax(np.stack([np.std(prof_target, axis=0) + np.finfo(float).eps, np.nanstd(prof_target_tmp, axis=0) + np.finfo(float).eps], axis=0), axis=0)
    prof_err = (prof_pred - prof_target) / (np.std(prof_target, axis=0) + np.finfo(float).eps)

    # Plot results
    fig0 = fig_vertical_profiles(prof_target, prof_pred, title=f"Test set - Vertical profiles")
    fig1 = fig_vertical_profile(prof_err, title=f"Test set - Vertical profile errors")
    fig2 = fig_rmse_bars(bt_target, bt_pred, clrsky,
                         title=[f"Test set - Forward model errors", f"Test set - Normalized forward model errors"])
    # Plot some invidiual vertical profiles
    for i in range(0, 10):
        fig4 = fig_vertical_profiles(prof_target[i:i+1, ...], prof_pred[i:i+1, ...],
                                     title=f"Test set - Vertical profile {i+1}")
        save_plot(fig4, config.paths.checkpoint_dir + f'/{config.task_name}_vertical_profiles_{i+1}.png')

    # Save figures to output directory
    save_plot(fig0, config.paths.checkpoint_dir + f'/{config.task_name}_vertical_profiles.png')
    save_plot(fig1, config.paths.checkpoint_dir + f'/{config.task_name}_vertical_profile.png')
    save_plot(fig2, config.paths.checkpoint_dir + f'/{config.task_name}_rmse_bars.png')

    # Apply normalization to the vertical profiles
    fig5 = fig_vertical_profiles(normalize_profiles(prof_target), normalize_profiles(prof_pred),
                                 title=f"Test set - Vertical profiles")
    save_plot(fig5, config.paths.checkpoint_dir + f'/{config.task_name}_vertical_profiles_normalized.png')


if __name__ == '__main__':
    """ Evaluate results from applying the model to the test set.

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


