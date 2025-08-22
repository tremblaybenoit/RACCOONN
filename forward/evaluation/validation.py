import numpy as np
import logging
import hydra
from omegaconf import DictConfig
from forward.train import CRTMEmulator
from forward.evaluation.metrics import rmse
from forward.utilities.logic import get_config_path
import matplotlib.pyplot as plt
from forward.utilities.instantiators import instantiate


# Initialize logger
logger = logging.getLogger(__name__)


def make_rmse_plot(ypred, bt, clrsky, cat, figname=None):
    """
    Plots raw and normalized RMSE by channel.

    Parameters
    ----------
    ypred : np.ndarray. Predicted values.
    bt : np.ndarray. True values.
    clrsky : np.ndarray. Clear sky mask.
    cat : str. Category of the data.
    figname : str. Filename to save the plot (optional).

    Returns
    -------
    None.
    """

    # Plot settings
    height = 0.3
    x = np.arange(7, 17)
    colors = ['#D81B60', '#1E88E5']

    # Plot raw RMSE
    plt.figure(figsize=(7, 5))
    plt.subplot(1, 2, 1)
    plt.title('a) Error (NN-CRTM)')
    for i, (mask, label, color) in enumerate([(~clrsky, 'Cloudy', colors[0]), (clrsky, 'Clear Sky', colors[1])]):
        err = ypred[mask, :10] - bt[mask]
        bars = plt.barh(x + i * height, rmse(err, axis=0), color=color, height=height, align='edge', label=label)
        plt.gca().bar_label(bars, fmt="%.2f", fontsize=10)
    plt.xlim([0, 1.6])
    plt.ylabel('Channel', fontsize=12)
    plt.xlabel(f'{cat} RMSE (K)', fontsize=12)
    plt.legend(fontsize=12)

    # Plot normalized RMSE
    plt.subplot(1, 2, 2)
    plt.title('b) Normalized Error')
    for i, (mask, label, color) in enumerate([(~clrsky, 'Cloudy', colors[0]), (clrsky, 'Clear Sky', colors[1])]):
        err = (ypred[mask, :10] - bt[mask]) / ypred[mask, 10:]
        bars = plt.barh(x + i * height, rmse(err, axis=0), color=color, height=height, align='edge', label=label)
        plt.gca().bar_label(bars, fmt="%.2f", fontsize=10)
    plt.xlim([0, 2])
    plt.xlabel(f'{cat} RMSE (Standard Deviations)', fontsize=12)

    # Adjust layout and save figure
    plt.tight_layout()
    if figname:
        plt.savefig(figname)


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

    # Initialize trainer object
    logger.info("Initializing CRTM emulator...")
    crtm = CRTMEmulator(config)

    # Evaluate on test set (i.e., the inputs of the prediction are from the test set)
    logger.info("Testing CRTM emulator...")
    pred = crtm.predict(config.data)

    # Load additional data from the test set
    prof = np.array(instantiate(config.data.sets.test.prof.load)).astype(np.float32)
    lat = np.array(instantiate(config.data.sets.test.lat.load)).astype(np.float32)
    lon = np.array(instantiate(config.data.sets.test.lon.load)).astype(np.float32)
    meta = np.array(instantiate(config.data.sets.test.meta.load)).astype(np.float32)
    surf = np.array(instantiate(config.data.sets.test.surf.load)).astype(np.float32)
    pressure = np.array(instantiate(config.data.sets.test.pressure.load)).astype(np.float32)
    hofx = np.array(instantiate(config.data.sets.test.hofx.load)).astype(np.float32)
    clear = np.logical_and(prof[:, 5, :].sum(axis=1) == 0, prof[:, 6, :].sum(axis=1) == 0)
    clear = np.logical_and(clear, prof[:, 7, :].sum(axis=1) == 0)
    cloudy = np.logical_not(clear)

    # Plot
    make_rmse_plot(pred, hofx, clear, 'Test', figname=config.paths.output_dir + '/rmse_test.png')


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
