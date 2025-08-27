import numpy as np
import hydra
from omegaconf import DictConfig
from forward.utilities.instantiators import instantiate
from forward.utilities.logic import get_config_path
from inverse.utilities.plot import plot_map, save_plot, flexible_gridspec
from inverse.data.transformations import identity


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


def background_climatology(data: np.ndarray, cloud_filter: np.ndarray=None, axis: int=0, keepdims: bool=True) \
        -> np.ndarray:
    """ Compute spatiotemporal mean of a given dataset.

        Parameters
        ----------
        data: np.ndarray. Input profiles of shape (n_samples, n_profiles, n_levels).
        cloud_filter: np.ndarray or None. Boolean mask to filter out clear-sky profiles.
        axis: int or tuple of int. Axis or axes along which the means are computed.
        keepdims: bool. If True, the reduced axes are left in the result as dimensions with size one.

        Returns
        -------
        np.ndarray. Spatiotemporal mean of the dataset.
    """

    # Apply cloud filter if provided
    if cloud_filter is not None:
        data = data[cloud_filter]

    # Compute mean
    return np.mean(data, axis=axis, keepdims=keepdims).astype(np.float64)


def background_increment(config_true: DictConfig, config_background: DictConfig,
                         cloud_filter: np.ndarray=None) -> np.ndarray:
    """ Compute error between ground truth and background.

        Parameters
        ----------
        config_true: DictConfig. Configuration for the ground truth dataset.
        config_background: DictConfig. Configuration for the background dataset.
        cloud_filter: np.ndarray or None. Boolean mask to filter out clear-sky profiles.

        Returns
        -------
        np.ndarray. Error between ground truth and background.
    """

    # Truth
    f_norm = config_true['normalization'] if hasattr(config_true, 'normalization') else identity
    xt = f_norm(np.array(config_true['load'], dtype=np.float64))

    # Background
    f_norm = config_background['normalization'] if hasattr(config_background, 'normalization') else identity
    xb = f_norm(np.array(config_background['load'], dtype=np.float64))

    # Apply cloud filter if provided
    if cloud_filter is not None:
        xt = xt[cloud_filter] if xt.shape[0] == xb.shape[0] else xt
        xb = xb[cloud_filter]

    return xt - xb


def covariance_matrix(config: DictConfig, plot_flag: bool=True, recenter: bool=True) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        plot_flag: bool. If True, plot the covariance matrix.
        recenter: bool. If True, recenter the error by removing the mean.

        Returns
        -------
        None.
    """

    # Error
    f_norm = instantiate(config['input']['err']['normalization']) \
        if hasattr(config['input']['err'], 'normalization') else identity
    err = f_norm(np.array(instantiate(config['input']['err']['load']), dtype=np.float64))

    # Cloud filter
    if hasattr(config['input'], 'cloud_filter'):
        cloud_filter = instantiate(config['input']['cloud_filter']['load'])
        err = err[cloud_filter]

    # Pressure filter (background only)
    if hasattr(config['input'], 'pressure_filter'):
        pressure_filter = instantiate(config['input']['pressure_filter']['load'])
        err = err[:, pressure_filter]

    # Compute covariance matrix
    if recenter:
        cov = np.cov(err.reshape(err.shape[0], -1) - np.mean(err, axis=0).reshape(1, -1), rowvar=False)
    else:
        cov = np.cov(err.reshape(err.shape[0], -1), rowvar=False)
    # Compute inverse covariance matrix
    cov_inv = np.linalg.inv(cov).astype(np.float64)

    # Save statistics to file
    filename = config['output']['path']
    np.save(config['output']['path'], cov_inv)

    # Plot covariance matrix if required
    if plot_flag:
        # From n_profiles, determine optimal layout for flexible_gridspec
        n_rows, n_cols = 1, 1
        # Create a flexible gridspec
        list_cols = [n_cols for _ in range(n_rows)]
        fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, cov_inv, title=f"Covariance matrix of profiles", img_range=(-10000, 10000), plt_origin='upper')
        save_plot(fig, filename=filename.replace('.npy', '.png'))

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
    if hasattr(config.data.preparation.covariance, 'model'):
        covariance_matrix(config.data.preparation.covariance.model)
    if hasattr(config.data.preparation.covariance, 'observation'):
        covariance_matrix(config.data.preparation.covariance.observation)

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