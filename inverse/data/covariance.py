import numpy as np
import hydra
from omegaconf import DictConfig
from utilities.io import load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from utilities.plot import plot_map, save_plot, flexible_gridspec
import os
import logging


# Initialize logger
logger = logging.getLogger(__name__)


def innovation_uncertainty(data: np.ndarray) -> np.ndarray:
    """ Return uncertainy estimation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray. Spatiotemporal standard deviation of the dataset.
    """

    # Return the standard deviation (last 10 values)
    return data[:, 10:]


def background_climatology(data: np.ndarray, axis: int=0, keepdims: bool=False) \
        -> np.ndarray:
    """ Compute spatiotemporal mean of a given dataset.

        Parameters
        ----------
        data: np.ndarray. Input profiles of shape (n_samples, n_profiles, n_levels).
        axis: int or tuple of int. Axis or axes along which the means are computed.
        keepdims: bool. If True, the reduced axes are left in the result as dimensions with size one.

        Returns
        -------
        np.ndarray. Spatiotemporal mean of the dataset.
    """

    # Compute mean
    return np.mean(data, axis=axis, keepdims=keepdims).astype(np.float32)


def background_increment(config_true: DictConfig, config_background: DictConfig) -> np.ndarray:
    """ Compute error between ground truth and background.

        Parameters
        ----------
        config_true: DictConfig. Configuration for the ground truth dataset.
        config_background: DictConfig. Configuration for the background dataset.

        Returns
        -------
        np.ndarray. Error between ground truth and background.
    """

    # Truth - Background
    x_true = load_var_and_normalize(config_true)
    x_background = load_var_and_normalize(config_background)

    # Check dimensions and add new axis if necessary
    if x_true.shape != x_background.shape:
        if x_background.ndim == x_true.ndim - 1:
            x_background = x_background[np.newaxis, :]
        else:
            raise ValueError("The shapes of the true and background data do not match.")

    return x_true-x_background


def covariance_matrix(input: DictConfig, output: DictConfig, plot_flag: bool=True, recenter: bool=True) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        plot_flag: bool. If True, plot the covariance matrix.
        recenter: bool. If True, recenter the error by removing the mean.

        Returns
        -------
        None.
    """

    # Error
    logger.info("Loading data...")
    err = load_var_and_normalize(input.err)

    # Pressure filter (background only)
    if hasattr(input, 'pressure_filter'):
        logger.info("Applying pressure filter...")
        pressure_filter = instantiate(input.pressure_filter.load)
        err = err[:, pressure_filter]

    # Compute covariance matrix
    if recenter:
        logger.info("Computing covariance matrix with recentered data...")
        cov = np.cov(err.reshape(err.shape[0], -1) - np.mean(err, axis=0).reshape(1, -1), rowvar=False)
    else:
        logger.info("Computing covariance matrix...")
        cov = np.cov(err.reshape(err.shape[0], -1), rowvar=False)
    # Compute inverse covariance matrix
    logger.info("Computing inverse covariance matrix...")
    cov_inv = np.linalg.inv(cov).astype(np.float32)

    # Save statistics to file
    logger.info("Saving inverse covariance matrix to file...")
    save_func = instantiate(output.save)
    save_func(cov_inv)

    # Plot covariance matrix if required
    if plot_flag:
        logger.info("Plotting inverse covariance matrix...")
        # Create a flexible gridspec
        fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                          lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, cov_inv/10000, title=f"Inverse covariance matrix", img_range=(-1, 1), plt_origin='upper',
                 cb_label=r'Values (divided by 10$^4$)')
        save_plot(fig, filename=os.path.splitext(output.path)[0] + '.png')

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute statistics of a given dataset.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute model and observation covariance matrices
    if hasattr(config.preparation, "covariance"):
        for dataset, config_covariance in config.preparation.covariance.items():
            logger.info(f"Computing error covariance matrix of {dataset} set")
            instantiate(config_covariance)

    return


if __name__ == '__main__':
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        zarr file containing data statistics.
    """

    main()