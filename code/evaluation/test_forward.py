import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from code.preprocessing.statistics import RunningStats
from utilities.logic import get_config_path
from utilities.instantiators import instantiate
from code.evaluation.plot import fig_rmse_bars, save_plot

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

    # Load test set output (predictions)
    logger.info("Load test set outputs...")
    bt_forward = instantiate(config.data.stage.test.variables.bt_forward.load)
    bt_forward_stdev = instantiate(config.data.stage.test.variables.bt_forward_stdev.load)

    # Load test set references
    logger.info("Load test set references...")
    bt_crtm = instantiate(config.data.stage.test.variables.bt_crtm.load)
    n_channels = bt_crtm.shape[1]

    # Create masks and compute rmse by condition
    mask_dict = {
        'clear': instantiate(config.data.stage.test.variables.clear_mask.load),
        'cloud': instantiate(config.data.stage.test.variables.cloud_mask.load),
        'day': instantiate(config.data.stage.test.variables.daytime_mask.load),
        'night': instantiate(config.data.stage.test.variables.nighttime_mask.load)
    }
    colors_dict = {
        'clear': '#D81B60',
        'cloud': '#1E88E5',
        'day': '#FFC107',
        'night': 'r'
    }
    # Create separate runners for each mask condition
    runners = {
        'clear': RunningStats(which=['rmse']),
        'cloud': RunningStats(which=['rmse']),
        'day': RunningStats(which=['rmse']),
        'night': RunningStats(which=['rmse']),
    }

    # Metrics
    metrics = {}
    for key, m in mask_dict.items():
        # Compute rmse
        runners[key].update(data=bt_forward[m], target=bt_crtm[m], axis=0)
    for runner_name, runner  in runners.items():
        # Skip runners with no accumulated data (e.g., cloud_mask in clear-sky datasets)
        if isinstance(runner._n, (int, float)) and runner._n == 0.0:
            continue
        metrics[runner_name] = runner.compute(dtype='float32')

    # Extract metrics of interest and metadata
    rmse = []
    labels = []
    colors = []
    for runner_name in runners.keys():
        if runner_name in metrics:
            rmse.append(metrics[runner_name]['rmse'])
        else:
            rmse.append(np.zeros((n_channels,)))
        labels.append(runner_name)
        colors.append(colors_dict.get(runner_name, '#000000'))

    # Plots
    logger.info("Plot comparison...")
    fig2 = fig_rmse_bars(rmse, channels=np.arange(n_channels), x_range=[[0, 0.8]],
                         colors=colors, labels=labels,
                         title=["Forward model RMSE per channel"])
    save_plot(fig2, config.paths.run_dir + '/bt_crtm_test_rmse_bars.png')


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
