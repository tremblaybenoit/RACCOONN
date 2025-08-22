import numpy as np
import hydra
from omegaconf import DictConfig
from forward.utilities.instantiators import instantiate
from forward.utilities.logic import get_config_path
from inverse.utilities.plot import plot_map, save_plot, flexible_gridspec
import os
from inverse.data.transformations import identity


def r_covariance_matrix(config: DictConfig, variable: str, filename: str, plot_flag: bool = True) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        variable: Variable to compute covariance matrix for.
        filename: str. Name of the file to save the covariance matrix.
        plot_flag: bool. If True, plot the covariance matrix.

        Returns
        -------
        None.
    """

    prof = np.array(instantiate(config['train']['prof']['load']), dtype=np.float32)
    clrsky = np.logical_and(prof[:, 5, :].sum(axis=1) == 0, prof[:, 6, :].sum(axis=1) == 0)
    clrsky = np.logical_and(clrsky, prof[:, 7, :].sum(axis=1) == 0)

    yt = np.array(np.load(os.path.join(config['train']['path'], 'hofx.npy')), dtype=np.float32)[~clrsky]
    hx = np.array(instantiate(config['train'][variable]['load']), dtype=np.float32)[:, :10][~clrsky]
    y_err = yt - hx
    rr = np.cov(y_err - np.mean(y_err, axis=0), rowvar=False)
    rr_inv = np.linalg.inv(rr).astype(np.float32)

    # Save statistics to file
    np.save(filename, rr_inv)

    # Plot covariance matrix if required
    if plot_flag:
        # From n_profiles, determine optimal layout for flexible_gridspec
        n_rows, n_cols = 1, 1
        # Create a flexible gridspec
        list_cols = [n_cols for _ in range(n_rows)]
        fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, rr_inv, title=f"Covariance matrix of {variable}", plt_origin='upper')  # ,  img_range=(-100, 100))
        save_plot(fig, filename=filename.replace('.npy', '.png'))

    return


def b_covariance_matrix(config: DictConfig, variable: str, filename: str, plot_flag: bool = True) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        variable: Variable to compute covariance matrix for.
        filename: str. Name of the file to save the covariance matrix.
        plot_flag: bool. If True, plot the covariance matrix.

        Returns
        -------
        None.
    """

    # Extract cloudy profiles
    prof = np.array(instantiate(config['train']['prof']['load']), dtype=np.float32)
    clrsky = np.logical_and(prof[:, 5, :].sum(axis=1) == 0, prof[:, 6, :].sum(axis=1) == 0)
    clrsky = np.logical_and(clrsky, prof[:, 7, :].sum(axis=1) == 0)
    # Apply normalization if specified
    f_norm = instantiate(config['train'][variable]['normalization']) \
        if hasattr(config['train'][variable], 'normalization') else identity
    xt = f_norm(np.array(instantiate(config['train'][variable]['load']), dtype=np.float32))[~clrsky, ...]

    xb = np.nanmean(xt, axis=0, keepdims=True)  # Compute mean and replace NaN with 0
    xb_err = xb - xt
    bb = np.cov(xb_err.reshape(xt.shape[0], -1) - np.mean(xb_err, axis=0).reshape(1, -1), rowvar=False)
    # Add a small value to the diagonal for numerical stability, but only where the diagonal is zero
    for i in range(bb.shape[0]):
        if bb[i, i] == 0:
            bb[i, i] += 1e-6
    bb_inv = np.linalg.inv(bb).astype(np.float32)

    # xt = np.array(instantiate(config['train'][variable]['load']), dtype=np.float32)
    # xt = np.where(xt > 0, xt, np.nan)  # Replace negative values with NaN
    # xb = np.nanmean(xt, axis=0, keepdims=True)  # Compute mean and replace NaN with 0
    # xb_err = np.nan_to_num(xb, nan=0.0).astype(np.float32) - np.nan_to_num(xt, nan=0.0).astype(np.float32)  # Compute error from the mean
    # xb_err_c = xb_err - np.nanmean(xb_err, axis=0, keepdims=True)
    # bb = np.cov(xb_err_c.reshape(xt.shape[0], -1), rowvar=False)
    # bb_inv = np.linalg.inv(bb + 1.e-9 * np.eye(bb.shape[0])).astype(np.float32)

    # xt = np.array(instantiate(config['train'][variable]['load']), dtype=np.float32)
    # xt = np.where(xt > 0, xt, np.nan)  # Replace negative values with NaN
    # xb = np.nanmean(xt, axis=0, keepdims=True)  # Compute mean and replace NaN with 0
    # xb_err = np.nan_to_num(xb, nan=0.0).astype(np.float32) - np.nan_to_num(xt, nan=0.0).astype(
    #     np.float32)  # Compute error from the mean
    # xb_err_c = xb_err - np.nanmean(xb_err, axis=0, keepdims=True)
    # bb = np.cov(xb_err_c.reshape(xt.shape[0], -1), rowvar=False)
    # bb_inv = np.linalg.inv(bb + 1.e-6 * np.eye(bb.shape[0])).astype(np.float32)
    # breakpoint()

    # Save statistics to file
    np.save(filename, bb_inv)

    # Plot covariance matrix if required
    if plot_flag:
        # From n_profiles, determine optimal layout for flexible_gridspec
        n_rows, n_cols = 1, 1
        # Create a flexible gridspec
        list_cols = [n_cols for _ in range(n_rows)]
        fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, bb_inv, title=f"Covariance matrix of {variable}", img_range=(-1000, 1000), plt_origin='upper')
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
    b_covariance_matrix(config.data.sets, variable='prof', filename=config.data.sets.background.covariance.path)
    r_covariance_matrix(config.data.sets, variable='hofx', filename=config.data.sets.observations.covariance.path)

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