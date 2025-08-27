import numpy as np
import hydra
from omegaconf import DictConfig
from forward.utilities.instantiators import instantiate
from forward.utilities.logic import get_config_path
from inverse.utilities.plot import plot_map, save_plot, flexible_gridspec
import os
from inverse.data.transformations import identity


def spatiotemporal_mean(prof: np.ndarray, cloud_filter: np.ndarray=None, axis: int=0, keepdims: bool=True) \
        -> np.ndarray:
    """ Compute spatiotemporal mean of a given dataset.

        Parameters
        ----------
        prof: np.ndarray. Input profiles of shape (n_samples, n_profiles, n_levels).
        cloud_filter: np.ndarray or None. Boolean mask to filter out clear-sky profiles.
        axis: int or tuple of int. Axis or axes along which the means are computed.
        keepdims: bool. If True, the reduced axes are left in the result as dimensions with size one.

        Returns
        -------
        np.ndarray. Spatiotemporal mean of the dataset.
    """

    # Apply cloud filter if provided
    if cloud_filter is not None:
        prof = prof[cloud_filter]

    # Compute mean
    return np.mean(prof, axis=axis, keepdims=keepdims).astype(np.float64)


def compute_err(xt: np.ndarray, xb: np.ndarray, cloud_filter: np.ndarray=None) -> np.ndarray:
    """ Compute error between ground truth and background.

        Parameters
        ----------
        xt: np.ndarray. Ground truth profiles of shape (n_samples, n_profiles, n_levels).
        xb: np.ndarray. Background profiles of shape (n_samples, n_profiles, n_levels).

        Returns
        -------
        np.ndarray. Error between ground truth and background.
    """

    # Apply cloud filter if provided
    if cloud_filter is not None:
        xb = xb[cloud_filter] if xb.shape[0] == xt.shape[0] else xb
        xt = xt[cloud_filter]

    return (xb - xt).astype(np.float64)

# TODO: Make generic covariance matrix function with ground truth, background, and error inputs
# TODO: Decide on computation of error in the context of observations
def r_covariance_matrix(config: DictConfig, plot_flag: bool = True) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        plot_flag: bool. If True, plot the covariance matrix.

        Returns
        -------
        None.
    """

    # Ground truth
    f_norm = instantiate(config['input']['hofx']['normalization']) \
        if hasattr(config['input']['hofx'], 'normalization') else identity
    yt = f_norm(np.array(instantiate(config['input']['hofx']['load']), dtype=np.float64))

    # Simulated observations
    f_norm = instantiate(config['input']['hofx']['normalization']) \
        if hasattr(config['input']['hofx'], 'normalization') else identity
    yb = f_norm(np.array(instantiate(config['input']['hofx']['load']), dtype=np.float64))

    # Cloud filter
    if hasattr(config['input'], 'cloud_filter'):
        cloud_filter = instantiate(config['input']['cloud_filter']['load'])
        yb = yb[cloud_filter] if yb.shape[0] == yt.shape[0] else yb
        yt = yt[cloud_filter]

    # Compute error
    y_err = yt - yb
    rr = np.cov(y_err - np.mean(y_err, axis=0), rowvar=False)
    rr_inv = np.linalg.inv(rr).astype(np.float32)

    # Save statistics to file
    filename = config['output']['path']
    np.save(config['output']['path'], rr_inv)

    # Plot covariance matrix if required
    if plot_flag:
        # From n_profiles, determine optimal layout for flexible_gridspec
        n_rows, n_cols = 1, 1
        # Create a flexible gridspec
        list_cols = [n_cols for _ in range(n_rows)]
        fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, rr_inv, title=f"Covariance matrix of profiles", img_range=(-1000, 1000), plt_origin='upper')
        save_plot(fig, filename=filename.replace('.npy', '.png'))

    return


def b_covariance_matrix(config: DictConfig, plot_flag: bool=True) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        plot_flag: bool. If True, plot the covariance matrix.

        Returns
        -------
        None.
    """

    # Ground truth
    f_norm = instantiate(config['input']['prof']['normalization']) \
        if hasattr(config['input']['prof'], 'normalization') else identity
    xt = f_norm(np.array(instantiate(config['input']['prof']['load']), dtype=np.float32))

    # Background
    f_norm = instantiate(config['input']['background']['normalization']) \
        if hasattr(config['input']['background'], 'normalization') else identity
    xb = f_norm(np.array(instantiate(config['input']['background']['load']), dtype=np.float32))

    # Cloud filter
    if hasattr(config['input'], 'cloud_filter'):
        cloud_filter = instantiate(config['input']['cloud_filter']['load'])
        xb = xb[cloud_filter] if xb.shape[0] == xt.shape[0] else xb
        xt = xt[cloud_filter]

    # Pressure filter
    if hasattr(config['input'], 'pressure_filter'):
        breakpoint()
        pressure_filter = instantiate(config['input']['pressure_filter']['load'])
        xb = xb[:, pressure_filter]
        xt = xt[:, pressure_filter]

    # Compute error
    xb_err = xb - xt

    # Background error
    f_norm = instantiate(config['input']['background_err']['normalization']) \
        if hasattr(config['input']['background_err'], 'normalization') else identity
    xb_err = f_norm(np.array(instantiate(config['input']['background_err']['load']), dtype=np.float64))

    # Pressure filter
    if hasattr(config['input'], 'pressure_filter'):
        pressure_filter = instantiate(config['input']['pressure_filter']['load'])
        xb_err = xb_err[:, pressure_filter]

    # Compute covariance matrix
    bb = np.cov(xb_err.reshape(xt.shape[0], -1) - np.mean(xb_err, axis=0).reshape(1, -1), rowvar=False)
    # Compute inverse covariance matrix
    bb_inv = np.linalg.inv(bb).astype(np.float64)

    # Save statistics to file
    filename = config['output']['path']
    np.save(config['output']['path'], bb_inv)

    # Plot covariance matrix if required
    if plot_flag:
        # From n_profiles, determine optimal layout for flexible_gridspec
        n_rows, n_cols = 1, 1
        # Create a flexible gridspec
        list_cols = [n_cols for _ in range(n_rows)]
        fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, bb_inv, title=f"Covariance matrix of profiles", img_range=(-1000, 1000), plt_origin='upper')
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

    # Compute background covariance matrix
    b_covariance_matrix(config.data.preparation.covariance.model)
    r_covariance_matrix(config.data.preparation.covariance.observation)

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