from typing import Union, Dict, List
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from forward.utilities.instantiators import instantiate
from forward.utilities.logic import get_config_path
from inverse.utilities.plot import save_plot, flexible_gridspec, plot_vertical_profiles


def statistics(data: Union[np.ndarray, torch.Tensor], axis: Union[int, tuple] = None) \
        -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        data: np.ndarray or torch.Tensor. Dataset to compute statistics on.
        axis: int or tuple. Axis to compute statistics along.

        Returns
        -------
        Dictionary containing statistics of the dataset.
    """

    # Convert torch tensor to numpy array
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Create dictionary to store statistics
    stats = {"mean": np.nanmean(data, axis=axis), "stdev": np.nanstd(data, axis=axis),
             "min": np.nanmin(data, axis=axis), "max": np.nanmax(data, axis=axis),
             "median": np.nanmedian(data, axis=axis), "variance": np.nanvar(data, axis=axis),
             "n_samples": np.sum(~np.isnan(data), axis=axis)}

    # Compute statistics
    return stats


def compute_histograms(config: DictConfig, variables: List[str]) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.
        variables: List[str]. List of variables to compute statistics for
                   (e.g. ['prof', 'surf', 'meta', 'hofx', 'lat', 'lon', 'scans', 'pressure']).

        Returns
        -------
        None.
    """

    # Read dataset
    pressure = np.array(instantiate(config.input.data['pressure'].load)).astype(np.float32)
    data = np.array(instantiate(config.input.data['prof'].load)).astype(np.float32)
    clrsky = np.logical_and(data[:, 5, :].sum(axis=1) == 0, data[:, 6, :].sum(axis=1) == 0)
    clrsky = np.logical_and(clrsky, data[:, 7, :].sum(axis=1) == 0)
    data_clrsky = data[clrsky, :, :]
    data_cloudy = data[~clrsky, :, :]

    # Save clearsky mask to file
    # instantiate(config.input.data['clrsky'].save, clrsky)

    # Compute statistics for clear-sky and cloudy data
    stats_all = statistics(data, axis=0)
    stats_clrsky = statistics(data_clrsky, axis=0)
    stats_cloudy = statistics(data_cloudy, axis=0)

    # Plot mean profiles
    _, n_profiles, n_levels = data.shape

    # Create a flexible gridspec
    # From n_profiles, determine optimal layout for flexible_gridspec
    n_rows = int(np.ceil(np.sqrt(n_profiles)))
    n_cols = int(np.ceil(n_profiles / n_rows))
    # Create a flexible gridspec
    list_cols = [n_cols for _ in range(n_rows)]
    """
    # Create a figure with flexible gridspec
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = variables[i] if i < len(variables) else f"Profile {i+1}"
        plot_vertical_profiles(ax, [data_clrsky[:, i, :], data_cloudy[:, i, :]], title=plot_title,
                               # err=[np.zeros((data.shape[-1])), np.zeros((data.shape[-1])), np.zeros((data.shape[-1]))],
                               y=pressure, y_label=variables[i], label=['Cleasky', 'Cloudy'], x_label='Profile value (units)')

    save_plot(fig, 'clrsky_cloudy.png')

    # Create a figure with flexible gridspec
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = variables[i] if i < len(variables) else f"Profile {i + 1}"
        plot_vertical_profiles(ax, [data_clrsky[:, i, :]], title=plot_title,
                               # err=[np.zeros((data.shape[-1])), np.zeros((data.shape[-1])), np.zeros((data.shape[-1]))],
                               y=pressure, y_label=variables[i], label=['Cleasky'],
                               x_label='Profile value (units)')

    save_plot(fig, 'clrsky.png')

    # Create a figure with flexible gridspec
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = variables[i] if i < len(variables) else f"Profile {i + 1}"
        plot_vertical_profiles(ax, [data_cloudy[:, i, :]], title=plot_title,
                               # err=[np.zeros((data.shape[-1])), np.zeros((data.shape[-1])), np.zeros((data.shape[-1]))],
                               y=pressure, y_label=variables[i], label=['Cloudy'],
                               x_label='Profile value (units)')

    save_plot(fig, 'cloudy.png')
    """

    # Create a figure with flexible gridspec
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = variables[i] if i < len(variables) else f"Profile {i + 1}"
        plot_vertical_profiles(ax, [stats_clrsky['min'][i, :], stats_cloudy['min'][i, :],
                                    stats_clrsky['mean'][i, :], stats_cloudy['mean'][i, :],
                                    stats_clrsky['max'][i, :], stats_cloudy['max'][i, :],
                                    ], title=plot_title,
                               # err=[np.zeros((data.shape[-1])), np.zeros((data.shape[-1])), np.zeros((data.shape[-1]))],
                               y=pressure, y_label=variables[i], label=['Clearsky - Min', 'Cloudy - Min',
                                                                        'Clearsky - Mean', 'Cloudy - Mean',
                                                                        'Clearsky - Max', 'Cloudy - Max'],
                               x_label='Profile value (units)')

    save_plot(fig, 'stats.png')

    # Set up histogram plot


    # Loop over different profile variables



    # Compute histograms per variable

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

    # Compute radiance statistics per filter per instrument
    compute_histograms(config.data.sets.preparation.statistics, variables=config.model.parameters.data.prof_vars)

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
