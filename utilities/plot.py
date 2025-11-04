import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density
from  forward.evaluation.metrics import rmse
from typing import Union, Callable, Optional
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Colors
colors = {'blue': '#1f77b4',
          'orange': '#ff7f0e',
          'green': '#2ca02c',
          'red': '#d62728',
          'purple': '#9467bd',
          'brown': '#8c564b',
          'pink': '#e377c2',
          'gray': '#7f7f7f',
          'olive': '#bcbd22',
          'cyan': '#17becf'}
# Density colormap (for scatterplots)
white_viridis = LinearSegmentedColormap.from_list('white_viridis',
                                                       [(0, '#ffffff'), (1e-20, '#440053'), (0.2, '#404388'),
                                                        (0.4, '#2a788e'), (0.6, '#21a784'), (0.8, '#78d151'),
                                                        (1, '#fde624')], N=256)


def flexible_gridspec(cell_widths: list[float], cell_heights: list[float], lefts: list[float], rights: list[float],
                      bottoms: list[float], tops: list[float]) -> tuple[plt.Figure, Callable]:
    """
    Create a figure with a flexible grid layout, allowing per-row/column cell sizes and per-row/column paddings.

    Parameters
    ----------
    cell_widths: list of floats, width of each column in inches.
    cell_heights: list of floats, height of each row in inches.
    lefts: list of floats, left padding for each column.
    rights: list of floats, right padding for each column.
    bottoms: list of floats, bottom padding for each row.
    tops: list of floats, top padding for each row.

    Returns
    -------
    fig, get_ax(row, col)
    """

    # Number of rows and columns
    nrows = len(cell_heights)
    ncols = len(cell_widths)

    # Compute total width and height
    fig_width = sum(cell_widths) + sum(lefts) + sum(rights)
    fig_height = sum(cell_heights) + sum(bottoms) + sum(tops)

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    # Precompute x/y positions for each cell
    x_starts = []
    x = 0
    for col in range(ncols):
        x += lefts[col]
        x_starts.append(x)
        x += cell_widths[col] + rights[col]
    y_starts = []
    y = fig_height
    for row in range(nrows):
        y -= tops[row]
        y_starts.append(y - cell_heights[row])
        y -= cell_heights[row] + bottoms[row]

    def get_ax(row: int, col: int) -> plt.Axes:
        """Get axes for the specified row and column."""
        x0 = x_starts[col] / fig_width
        y0 = y_starts[row] / fig_height
        w = cell_widths[col] / fig_width
        h = cell_heights[row] / fig_height
        return fig.add_axes((x0, y0, w, h))

    return fig, get_ax


def apply_colorbar(ax, plot, font_size: float=13, label: str='Density', label_pad: float=15.5,
                   orientation: str="horizontal", rotation: float=0, side: str='bottom', vmin: float=-1, vmax: float=1,
                   size: float=0.01, pad: float=0.08, ticks=None, tickw: float=1, tickl: float=2.5, tickdir: str='out') \
        -> plt.colorbar:
    """
    Add a floating colorbar to the given axes, using fig.add_axes, without shrinking the plot.
    If horizontal, colorbar width matches plot width and side can be 'bottom' or 'top'.
    If vertical, colorbar height matches plot height and side can be 'left' or 'right'.

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to add the colorbar to.
    plot : matplotlib.cm.ScalarMappable. The plot to which the colorbar applies (e.g., the result of a scatter or imshow).
    font_size : float. Font size for the colorbar.
    label : str. Label for the colorbar.
    label_pad : float. Padding for the colorbar label.
    orientation : str. Orientation of the colorbar ('horizontal' or 'vertical').
    rotation : float. Rotation of the colorbar label.
    side : str. Side of the colorbar ('bottom', 'top', 'left', 'right').
    vmin : float. Minimum value for the colorbar.
    vmax : float. Maximum value for the colorbar.
    size : float. Thickness of the colorbar (in figure fraction).
    pad : float. Padding between the plot and the colorbar (in figure fraction).
    ticks : int. Number of ticks on the colorbar. If None, automatic ticks are used.
    tickw : float. Width of the colorbar ticks.
    tickl : float. Length of the colorbar ticks.
    tickdir : str. Direction of the colorbar ticks ('in', 'out', 'inout').

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar. The created colorbar.
    """

    # Get figure and position of the plot
    fig = ax.figure
    bbox = ax.get_position()

    # Depending on orientation, compute colorbar position and create it
    if orientation == 'horizontal':
        # Compute dimensions
        width = bbox.width
        height = size
        left = bbox.x0
        # Compute bottom position based on side
        if side == 'bottom':
            bottom = bbox.y0 - pad*bbox.height - height
        elif side == 'top':
            bottom = bbox.y0 + bbox.height + pad*bbox.height
        else:
            raise ValueError("For horizontal colorbar, side must be 'bottom' or 'top'")
        # Create colorbar axes
        cbar_ax = fig.add_axes([left, bottom, width, height])
        # Create colorbar
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation='horizontal')
        # Customize ticks
        cbar.ax.tick_params(axis='x', direction=tickdir, labelsize=font_size, width=tickw, length=tickl, bottom=(side=='bottom'), top=(side=='top'))
        # Set tick positions and label positions
        cbar.ax.xaxis.set_ticks_position(side)
        cbar.ax.xaxis.set_label_position(side)
        if label is not None:
            cbar.set_label(label, labelpad=label_pad, rotation=rotation, size=font_size)
        if ticks is not None:
            tick_values = np.linspace(vmin, vmax, ticks)
            cbar.set_ticks(tick_values)
            cbar.ax.set_xticklabels([f"{v:.2f}" for v in tick_values])
    else:
        # Compute dimensions
        height = bbox.height
        width = size
        bottom = bbox.y0
        # Compute left position based on side
        if side == 'right':
            left = bbox.x0 + bbox.width + pad
        elif side == 'left':
            left = bbox.x0 - pad - width
        else:
            raise ValueError("For vertical colorbar, side must be 'left' or 'right'")
        # Create colorbar axes
        cbar_ax = fig.add_axes([left, bottom, width, height])
        # Create colorbar
        cbar = fig.colorbar(plot, cax=cbar_ax, orientation='vertical')
        # Set ticks if specified
        if ticks is not None:
            tick_values = np.linspace(vmin, vmax, ticks)
            cbar.set_ticks(tick_values)
            cbar.ax.set_yticklabels([f"{v:.2f}" for v in tick_values])
        # Customize ticks
        cbar.ax.tick_params(axis='y', direction=tickdir, labelsize=font_size, width=tickw, length=tickl, left=(side=='left'), right=(side=='right'))
        # Set tick positions and label positions
        cbar.ax.yaxis.set_ticks_position(side)
        cbar.ax.yaxis.set_label_position(side)
        if label is not None:
            cbar.set_label(label, labelpad=label_pad, rotation=rotation if rotation is not None else (270 if side=='right' else 90), size=font_size)

    return cbar


def compute_min_max(data: np.ndarray, symmetric: bool=False) -> tuple[float, float]:
    """ Compute the minimum and maximum values of the data.

    Parameters
    ----------
    data : numpy.ndarray. Data to compute the min and max.
    symmetric : bool. If True, the min and max values are symmetric around 0.

    Returns
    -------
    min_val : float. Minimum value of the data.
    max_val : float. Maximum value of the data.
    """

    # Compute the min and max values
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    # Adopt a symmetric range if needed [-x, x]
    if symmetric and min_val * max_val < 0:
        max_val = np.nanmax([abs(min_val), abs(max_val)])
        min_val = -max_val

    # Ensure min and max are noth both zeroes
    if min_val == 0 and max_val == 0:
        min_val, max_val = -1.0, 1.0

    return min_val, max_val


def save_plot(fig, filename: str, fileformat: str='png', dpi: int=300):
    """ Save the figure to a file.
    Parameters
    ----------
    fig : matplotlib.figure.Figure. Figure to save.
    filename : str. Name of the file to save.
    fileformat : str. Format of the file to save. The default is 'png'.
    dpi : int. Dots per inch for the saved figure.

    Returns
    -------
    None.
    """

    # Save the figure
    fig.savefig(filename, format=fileformat, dpi=dpi)
    # Close the figure
    plt.close('all')


def plot_map(ax, img, img_alpha=1.0, img_norm='linear', img_coord=(0, 0), img_shape=None, img_pixel=(1, 1),
             img_ticks=None, img_labels=('y-axis', 'x-axis'), title=None, img_labelspad=(5, 3), img_tickw=1,
             img_tickl=2.5, img_tickdir='out', title_pad=1.005, cb_label=None, img_range=None,
             cb_cmap=None, cb_pad=0, cb_tickw=1, cb_tickl=2.5, cb_font=12, cb_dir='out', cb_rot=270, cb_labelpad=15.5,
             cb_side='right', cb_size=0.025, cb_ticks=5,
             plt_coord=None, plt_color='black', plt_linew=1, plt_lines='-', plt_symbl='', plt_origin='lower',
             vec=None, vec_coord=None, vec_step=1, vec_scale=1, vec_width=1, vec_hwidth=1, vec_hlength=1,
             vec_haxislength=1, vec_color='black', vec_qlength=1, vec_labelsep=0.05,
             vec_qdecimals=2, vec_qscale=1, vec_qunits='', font_size=13):
    """ Plot a map with optional vectors and colorbar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to plot on.
    img : numpy.ndarray. Image data to plot.
    img_alpha : float. Alpha value for the image.
    img_norm : str. Normalization for the image. Default is 'linear'.
    img_coord : tuple. Coordinates for the image. Default is (0, 0).
    img_shape : tuple. Shape of the image. Default is None.
    img_pixel : tuple. Pixel size of the image. Default is (1, 1).
    img_ticks : tuple. Ticks for the image. Default is None.
    img_labels : tuple. Labels for the image axes. Default is ('y-axis', 'x-axis').
    title : str. Title of the plot. Default is None.
    img_labelspad : tuple. Padding for the image labels. Default is (5, 3).
    img_tickw : float. Width of the ticks. Default is 1.
    img_tickl : float. Length of the ticks. Default is 2.5.
    img_tickdir : str. Direction of the ticks. Default is 'out'.
    title_pad : float. Padding for the title. Default is 1.005.
    cb_label : str. Label for the colorbar. Default is None.
    img_range : tuple. Range for the image. Default is None.
    cb_cmap : str. Colormap for the colorbar. Default is None.
    cb_pad : float. Padding for the colorbar. Default is 0.
    cb_tickw : float. Width of the colorbar ticks. Default is 1.
    cb_tickl : float. Length of the colorbar ticks. Default is 2.5.
    cb_font : int. Font size for the colorbar. Default is 12.
    cb_dir : str. Direction of the colorbar ticks. Default is 'out'.
    cb_rot : int. Rotation of the colorbar label. Default is 270.
    cb_labelpad : float. Padding for the colorbar label. Default is 15.5.
    cb_side : str. Side for the colorbar. Default is 'right'.
    cb_size : float. Size of the colorbar. Default is 0.05*4/(0.8+0.8+4).
    cb_ticks : int. Number of ticks on the colorbar. Default is 5.
    plt_coord : list. Coordinates for the plot. Default is None.
    plt_color : str. Color for the plot. Default is 'black'.
    plt_linew : float. Line width for the plot. Default is 1.
    plt_lines : str. Line style for the plot. Default is '-'.
    plt_symbl : str. Marker style for the plot. Default is ''.
    plt_origin : str. Origin for the plot. Default is 'lower'.
    vec : numpy.ndarray. Vectors to plot. Default is None.
    vec_coord : tuple. Coordinates for the vectors. Default is None.
    vec_step : int. Step size for the vectors. Default is 1.
    vec_scale : float. Scale for the vectors. Default is 1.
    vec_width : float. Width of the vectors. Default is 1.
    vec_hwidth : float. Head width of the vectors. Default is 1.
    vec_hlength : float. Head length of the vectors. Default is 1.
    vec_haxislength : float. Head axis length of the vectors. Default is 1.
    vec_color : str. Color for the vectors. Default is 'black'.
    vec_qlength : float. Length of the quiver key. Default is 1.
    vec_labelsep : float. Label separation for the quiver key. Default is 0.05.
    vec_qdecimals : int. Number of decimals for the quiver key. Default is 2.
    vec_qscale : float. Scale for the quiver key. Default is 1.
    vec_qunits : str. Units for the quiver key. Default is ''.
    font_size : int. Font size for the plot. Default is 13.

    Returns
    -------
    None.
    """

    # Extract image shape if not provided
    if img_shape is None:
        img_shape = img.shape

    # Extract subpatch
    img_plot = img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]]

    # Spatial extent
    extent = np.array([
        img_pixel[1] * img_coord[1],
        img_pixel[1] * (img_coord[1] + img_shape[1]),
        img_pixel[0] * img_coord[0],
        img_pixel[0] * (img_coord[0] + img_shape[0])
    ])

    # Colormap
    if img_range is None:
        img_range = compute_min_max(img_plot, symmetric=True)
    if cb_cmap is None:
        cmap = 'RdBu_r' if img_range[0] * img_range[1] < 0 else 'GnBu_r'
    else:
        cmap = cb_cmap

    # Plot image
    I = ax.imshow(
        img_plot, extent=extent, cmap=cmap, vmin=img_range[0], vmax=img_range[1],
        aspect=1, interpolation='none', alpha=img_alpha, origin=plt_origin, norm=img_norm
    )

    # Plot vectors
    if vec is not None and vec_coord is not None:
        q = ax.quiver(
            vec_coord[1][::vec_step, ::vec_step], vec_coord[0][::vec_step, ::vec_step],
            vec[1][::vec_step, ::vec_step], vec[0][::vec_step, ::vec_step],
            units='xy', scale=vec_scale, width=vec_width, headwidth=vec_hwidth,
            headlength=vec_hlength, headaxislength=vec_haxislength, pivot='tail', scale_units='xy',
            color=vec_color
        )
        qk_label = str(np.around(vec_qlength, decimals=vec_qdecimals))
        ax.quiverkey(
            q, 0.9, 0.05, vec_qlength * vec_qscale, qk_label + f' {vec_qunits}',
            labelpos='E', coordinates='axes', fontproperties={'size': str(cb_font)}, labelsep=vec_labelsep
        )

    # Overplot patch boundaries
    if plt_coord is not None:
        for loop in range(len(plt_coord)):
            x, y = zip(*plt_coord[loop])
            ax.plot(x, y, color=plt_color, linewidth=plt_linew, linestyle=plt_lines, marker=plt_symbl)

    # Set axis ticks
    ax.get_yaxis().set_tick_params(which='both', direction=img_tickdir, width=img_tickw, length=img_tickl,
                                   labelsize=cb_font, left=True, right=True)
    ax.get_xaxis().set_tick_params(which='both', direction=img_tickdir, width=img_tickw, length=img_tickl,
                                   labelsize=cb_font, bottom=True, top=True)
    if img_ticks is not None:
        ax.get_yaxis().set_major_locator(plt.MultipleLocator(img_ticks[0]))
        ax.get_xaxis().set_major_locator(plt.MultipleLocator(img_ticks[1]))


    # Set axis labels
    ax.set_ylabel(img_labels[0], fontsize=cb_font, labelpad=img_labelspad[0])
    ax.set_xlabel(img_labels[1], fontsize=cb_font, labelpad=img_labelspad[1])


    # Title
    if title is not None:
        ax.set_title(title, fontsize=font_size, y=title_pad, wrap=True)

    # Set colorbar
    if I is not None:
        apply_colorbar(ax, I, font_size=cb_font, label=cb_label, label_pad=cb_labelpad, orientation='vertical',
                       rotation=cb_rot, side=cb_side, vmin=img_range[0], vmax=img_range[1], size=cb_size, pad=cb_pad,
                       ticks=cb_ticks, tickw=cb_tickw, tickl=cb_tickl, tickdir=cb_dir)


def plot_vertical_profiles(ax, data, err=None, font_size=13, title='Mean vertical profile ± std', title_pad=1.005, y=None,
                           x_label=r'Profile value ($\sigma$)', y_label='Height (levels)', y_labelpad=5, x_labelpad=3,
                           x_range: tuple=None, y_range: tuple=None, x_nticks=6, y_nticks=6, y_invert=True, tickw=1,
                           tickl=2.5, tickdir='out', color=None, linew=0.5, lines='-', label = None, alpha=0.2,
                           grid=True, grid_linew=0.5, x_ascale='linear', y_ascale='linear', lg_loc='best', lg_font=10,
                           lg_ncol=1, lg_npoints=1, lg_scale=4.0, lg_spacing=0.05):
    """ Plot a vertical profile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to plot on.
    data : numpy.ndarray. Data to plot. Shape should be (n_samples, n_profiles, n_levels).
    err : numpy.ndarray. Error data to plot. Shape should be (n_samples, n_profiles, n_levels). If None, standard deviation is used.
    font_size : int. Font size for the plot.
    title : str. Title of the plot.
    title_pad : float. Padding for the title.
    y: float. Vertical levels.
    x_label : str. Label for the x-axis.
    y_label : str. Label for the y-axis.
    y_labelpad : float. Padding for the y-axis label.
    x_labelpad : float. Padding for the x-axis label.
    x_range : tuple. Range for the x-axis. If None, computed from data.
    y_range : tuple. Range for the y-axis. If None, computed from data.
    x_nticks : int. Number of ticks on the x-axis.
    y_nticks : int. Number of ticks on the y-axis.
    y_invert : bool. If True, invert the y-axis.
    tickw : float. Width of the ticks.
    tickl : float. Length of the ticks.
    tickdir : str. Direction of the ticks. Default is 'out'.
    color : str. Color for the plot line. If None, defaults to blue.
    linew : float. Line width for the plot line.
    lines : str. Line style for the plot line. Default is '-'.
    label : str. Label for the plot line. Default is None.
    alpha : float. Alpha value for the shaded area around the mean line.
    grid : bool. If True, show grid.
    grid_linew : float. Line width of the grid.
    x_ascale : str. Scale for the x-axis. Default is 'linear'.
    y_ascale : str. Scale for the y-axis. Default is 'linear'.
    lg_loc : str. Location of the legend.
    lg_font : int. Font size for the legend.
    lg_ncol : int. Number of columns in the legend.
    lg_npoints : int. Number of points in the legend.
    lg_scale : float. Scale for the legend markers.
    lg_spacing : float. Spacing between legend entries.

    Returns
    -------
    None.
    """

    # Shapes
    if isinstance(data, list):
        n_profiles = len(data)
        n_levels = data[0].shape[0]
        # Compute mean and standard deviation
        data_mean = np.stack([np.mean(data[i], axis=0) if data[i].ndim > 1 else data[i] for i in range(n_profiles)], axis=0)
        data_std = np.stack([np.std(data[i], axis=0) if data[i].ndim > 1 else np.zeros_like(data[i]) for i in range(n_profiles)], axis=0)
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            n_profiles = data.shape[0]
            n_levels = data.shape[1]
            data_mean = data
            data_std = np.zeros_like(data) if err is None else err
        else:
            n_profiles = data.shape[1]
            n_levels = data.shape[2]
            # Compute mean and standard deviation
            data_mean = data.mean(axis=0)
            data_std = data.std(axis=0) if err is None else err
    else:
        raise ValueError("Data should be a list or a 3D array with shape (n_samples, n_profiles, n_levels) or a 2D array with shape (n_profiles, n_levels).")
    heights = np.arange(n_levels) if y is None else y

    # Compute min and max values
    if x_range is None:
        x_range = compute_min_max(
            np.stack([data_mean - data_std if np.min(data_mean) < 0 else np.clip(data_mean - data_std, 0, None),
                      data_mean + data_std]), symmetric=False)
        x_range = (0.95* x_range[0] if x_range[0] >= 0 else 1.05 * x_range[0],
                   1.05 * x_range[1] if x_range[1] >= 0 else 0.95 * x_range[1])
    if y_range is None:
        y_range = compute_min_max(heights, symmetric=False)

    # Loop over profiles
    for profile in range(n_profiles):
        # Plot mean and standard deviation
        lcolor = colors[list(colors.keys())[profile]] if color is None else color[profile]
        ax.plot(data_mean[profile], heights, color=lcolor, label=label[profile], linewidth=linew, linestyle=lines)
        ax.fill_betweenx(heights, data_mean[profile] - data_std[profile], data_mean[profile] + data_std[profile],
                         alpha=alpha, color=lcolor)
    if y is not None:
        ax.plot([np.min(x_range), np.max(x_range)], [100., 100.],
                label='100 hpa level', color='gray', linewidth=0.5, linestyle='--')

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xscale(x_ascale)
    ax.set_yscale(y_ascale)
    if y_invert:
        ax.invert_yaxis()

    # Add grid
    ax.grid(grid, linewidth=grid_linew)

    # Set axis ticks
    ax.get_yaxis().set_tick_params(which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size,
                                   left=True, right=True)
    ax.get_xaxis().set_tick_params(which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size,
                                   bottom=True, top=True)
    if x_nticks is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(x_nticks))
    if y_nticks is not None:
        ax.yaxis.set_major_locator(plt.MaxNLocator(y_nticks))

    # Set axis labels
    ax.set_ylabel(y_label, fontsize=font_size, labelpad=y_labelpad)
    ax.set_xlabel(x_label, fontsize=font_size, labelpad=x_labelpad)
    # Set title
    ax.set_title(title, fontsize=font_size, y=title_pad)
    # Set legend
    ax.legend(loc=lg_loc, fontsize=lg_font, labelspacing=lg_spacing, numpoints=lg_npoints, ncol=lg_ncol,
              markerscale=lg_scale, fancybox=False)


def fig_vertical_profiles(sources: list[np.ndarray], label: list[str], y: np.ndarray=None,
                          x_label: str = None, y_label: Union[list, str]=None, color: Union[list, str]=None,
                          title: Union[list, str] = None, x_range=None):
    """
    Plot vertical profiles for target and prediction data using flexible gridspec.

    Parameters
    ----------
    sources : list of numpy.ndarray. List containing target and prediction data. Each array should have shape (n_samples, n_profiles, n_levels).
    label : list of str. Labels for the target and prediction data.
    y : np.ndarray, optional. Vertical levels. If None, levels will be generated as 0, 1, ..., n_levels-1.
    y_label : Union[list[str, ...], str], optional. Labels for the vertical levels. If None, default labels will be used.
    x_label : str, optional. Label for the x-axis. If None, default label will be used.
    color : Union[list[str, ...], str], optional. Colors for the target and prediction data. If None, default colors will be used.
    title : str, optional. Title for the plots. If None, default titles will be used.
    x_range : tuple, optional. Range for the x-axis. If None, computed from data.

    Returns
    -------
    fig : matplotlib.figure.Figure. Figure with vertical profiles for target and prediction.
    """

    # Check shapes
    n_samples, n_profiles, n_levels = sources[0].shape
    for src in sources:
        if src.shape != (n_samples, n_profiles, n_levels):
            raise ValueError("All source arrays must have the same shape (n_samples, n_profiles, n_levels).")

    # If no colors, assign default colors
    if color is None:
        color = [colors[list(colors.keys())[c]] for c in range(len(sources))]

    # From n_profiles, determine optimal layout for flexible_gridspec
    n_rows = int(np.ceil(np.sqrt(n_profiles)))
    n_cols = int(np.ceil(n_profiles / n_rows))
    # Create a flexible gridspec
    cell_widths = [4.0] * n_cols
    cell_heights = [4.0] * n_rows
    lefts = [0.75] * n_cols
    rights = [0.75] * n_cols
    bottoms = [0.75] * n_rows
    tops = [0.75] * n_rows
    fig, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        stacked = np.stack([src[:, i, :] for src in sources], axis=1)
        plot_title = f'Vertical profile #{i + 1}' if title is None else title[i]
        plot_vertical_profiles(ax, stacked, title=plot_title, y=y, y_label=y_label, label=label,
                               x_label=x_label, x_range=x_range, color=color)

    return fig


def plot_rmse_bars(ax, values, positions, height=0.3, colors=None, labels=None, x_range=None, x_ascale='linear',
                   font_size=13, title='RMSE by channel', title_pad=1.005, x_label='RMSE (units)', y_label='Channels',
                   y_labelpad=5, x_labelpad=3, tickw=1, tickl=2.5, tickdir='out', lg_loc='upper right', lg_font=10,
                   lg_ncol=1, lg_npoints=1, lg_scale=4.0, lg_spacing=0.05, **bar_kwargs):
    """
    Bar plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to plot on.
    values : numpy.ndarray. Values to plot. Must have shape (n_groups, n_values).
    positions : numpy.ndarray. Positions of the bars on the y-axis. Must have shape (n_groups,).
    height : float. Height of the bars.
    colors : list or None. Colors for the bars. If None, defaults to a single color.
    labels : list or None. Labels for the bars. If None, no labels are shown.
    x_range : tuple or None. Range for the y-axis. If None, computed from data.
    x_ascale : str. Scale for the y-axis. Default is 'linear'.
    font_size : int. Font size for the plot.
    title : str. Title of the plot.
    title_pad : float. Padding for the title.
    x_label : str. Label for the x-axis.
    y_label : str. Label for the y-axis.
    y_labelpad : float. Padding for the y-axis label.
    x_labelpad : float. Padding for the x-axis label.
    tickw : float. Width of the ticks.
    tickl : float. Length of the ticks.
    tickdir : str. Direction of the ticks. Default is 'out'.
    lg_loc : str. Location of the legend.
    lg_font : int. Font size for the legend.
    lg_ncol : int. Number of columns in the legend.
    lg_npoints : int. Number of points in the legend.
    lg_scale : float. Scale for the legend markers.
    lg_spacing : float. Spacing between legend entries.
    bar_kwargs : dict. Additional keyword arguments for the bar plot.

    Return
    ------
    None
    """

    # Compute min and max values
    if x_range is None:
        x_range = compute_min_max(values, symmetric=False)
        x_range = (0, 1.2*x_range[1])
    # Aspect ratio
    # ax.set_aspect(1)

    n = len(values)
    for i in range(n):
        bars = ax.barh(positions + i * height, values[i], color=None if colors is None else colors[i],
                       height=height, align='edge', label=None if labels is None else labels[i], **bar_kwargs)
        bar_font_size = 10 if n <= 2 else 7
        ax.bar_label(bars, fmt="%.2f", fontsize=bar_font_size)

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_xscale(x_ascale)

    # Set axis ticks
    ax.get_yaxis().set_tick_params(which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size,
                                   left=True, right=True)
    ax.get_xaxis().set_tick_params(which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size,
                                   bottom=True, top=True)

    # Set axis labels
    ax.set_ylabel(y_label, fontsize=font_size, labelpad=y_labelpad)
    ax.set_xlabel(x_label, fontsize=font_size, labelpad=x_labelpad)
    # Set title
    ax.set_title(title, fontsize=font_size, y=title_pad)

    # Set legend
    if labels is not None:
        ax.legend(loc=lg_loc, fontsize=lg_font, labelspacing=lg_spacing, numpoints=lg_npoints, ncol=lg_ncol,
                  markerscale=lg_scale, fancybox=False)



def fig_rmse_bars(target, pred, clrsky, figname=None, channels=None, height=0.3, colors=None, labels=None,
                  x_range=None, y_label=None, x_label=None, title=None):
    """
    Plot raw and normalized RMSE bars side by side using a flexible gridspec.

    Parameters
    ----------
    target : numpy.ndarray. Target data. Shape should be (n_samples, n_channels).
    pred : numpy.ndarray. Prediction data. Shape should be (n_samples, n_channels).
    clrsky : numpy.ndarray. Boolean array indicating clear sky samples. Shape should be (n_samples,).
    figname : str or None. If provided, the figure will be saved to this filename.
    channels : list or None. List of channel indices to plot. If None, defaults to channels 7 to 16.
    height : float. Height of the bars. Default is 0.3.
    colors : list or None. List of colors for the bars. If None, defaults to ['#D81B60', '#1E88E5'].
    labels : list or None. List of labels for the bars. If None, defaults to ['Cloudy', 'Clear Sky'].
    x_range : list or None. List of x-axis ranges for the two plots. If None, defaults to [None, None].
    y_label : list or None. List of y-axis labels for the two plots. If None, defaults to ['Channels', 'Channels'].
    x_label : list or None. List of x-axis labels for the two plots.
              If None, defaults to ['RMSE (K)', 'RMSE (Standard Deviations)']. Default is None.
    title : list or None. List of titles for the two plots.
            If None, defaults to ['(a) Forward model errors', '(b) Normalized forward model errors'].

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # Axis
    channels = channels if channels is not None else np.arange(7, 17)
    y_label = y_label if y_label is not None else ['Channels', 'Channels']
    x_label = x_label if x_label is not None else [f'RMSE (K)', f'RMSE (Standard Deviations)']
    x_range = x_range if x_range is not None else [None, None]  # [(0, 1.6), (0, 2.0)]
    title = title if title is not None else ['(a) Forward model errors', '(b) Normalized forward model errors']
    # Colors and labels
    colors = colors if colors is not None else ['#D81B60', '#1E88E5']
    labels = labels if labels is not None else ['Cloudy', 'Clear Sky']

    # Prepare error values
    err_cloudy = pred[~clrsky, :10] - target[~clrsky, :10]
    err_clear = pred[clrsky, :10] - target[clrsky, :10]
    norm_cloudy = err_cloudy / pred[~clrsky, 10:]
    norm_clear = err_clear / pred[clrsky, 10:]

    # Create flexible grid (2 columns)
    cell_widths = [4.0, 4.0]
    cell_heights = [4.0]
    lefts = [0.75, 0.75]
    rights = [0.75, 0.75]
    bottoms = [0.75]
    tops = [0.75]
    fig, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)
    ax0 = get_axes(0, 0)
    ax1 = get_axes(0, 1)

    # Plot raw RMSE
    values_raw = [rmse(err_cloudy, axis=0), rmse(err_clear, axis=0)]
    plot_rmse_bars(ax0, values_raw, channels, height=height, colors=colors, labels=labels, x_range=x_range[0],
                   title=title[0], x_label=x_label[0], y_label=y_label[0])

    # Plot normalized RMSE
    values_norm = [rmse(norm_cloudy, axis=0), rmse(norm_clear, axis=0)]
    plot_rmse_bars(ax1, values_norm, channels, height=height, colors=colors, labels=labels, x_range=x_range[1],
                   title=title[1], x_label=x_label[1], y_label=y_label[1])

    # Save plot
    if figname:
        save_plot(fig, figname)

    return fig


def plot_rmse_bars2(ax, values, positions, height=0.175, colors=None, labels=None, x_range=None, **kwargs):
    """
    Bar plot for four groups: Cloudy, Clear Sky, Day, Night.

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to plot on.
    values : numpy.ndarray. Values to plot. Must have shape (4, n_values).
    positions : numpy.ndarray. Positions of the bars on the y-axis. Must have shape (n_values,).
    height : float. Height of the bars.
    colors : list or None. Colors for the bars. If None, defaults to ['#D81B60', '#1E88E5', '#FFC107', 'r'].
    labels : list or None. Labels for the bars. If None, defaults to ['Cloudy', 'Clear Sky', 'Day', 'Night'].
    x_range : tuple or None. Range for the y-axis. If None, computed from data.
    kwargs : dict. Additional keyword arguments for the bar plot.

    Return
    ------
    None.
    """
    # Defaults for 4 groups
    if colors is None:
        colors = ['#D81B60', '#1E88E5', '#FFC107', 'r']
    if labels is None:
        labels = ['Cloudy', 'Clear Sky', 'Day', 'Night']
    # Delegate to existing implementation
    return plot_rmse_bars(ax, values, positions, height=height, colors=colors, labels=labels, x_range=x_range, **kwargs)


def fig_rmse_bars2(target, pred, clrsky, daytime, figname=None, channels=None, height=0.175, colors=None, labels=None,
                   x_range=None, y_label=None, x_label=None, title=None):
    """
    Create figure with two panels: raw RMSE and normalized RMSE for four groups:
    Cloudy, Clear Sky, Day, Night. Returns matplotlib Figure.

    Parameters
    ----------
    target : numpy.ndarray. Target data. Shape should be (n_samples, n_channels).
    pred : numpy.ndarray. Prediction data. Shape should be (n_samples, n_channels).
    clrsky : numpy.ndarray. Boolean array indicating clear sky samples. Shape should be (n_samples,).
    daytime : numpy.ndarray. Boolean array indicating daytime samples. Shape should be (n_samples,).
    figname : str or None. If provided, the figure will be saved to this filename.
    channels : list or None. List of channel indices to plot. If None, defaults to channels 7 to 16.
    height : float. Height of the bars. Default is 0.175.
    colors : list or None. List of colors for the bars. If None, defaults to ['#D81B60', '#1E88E5', '#FFC107', 'r'].
    labels : list or None. List of labels for the bars. If None, defaults to ['Cloudy', 'Clear Sky', 'Day', 'Night'].
    x_range : list or None. List of x-axis ranges for the two plots. If None, defaults to [(0, 1.6), (0, 2.0)].
    y_label : list or None. List of y-axis labels for the two plots. If None, defaults to ['Channels', 'Channels'].
    x_label : list or None. List of x-axis labels for the two plots.
              If None, defaults to ['RMSE (K)', 'RMSE (Standard Deviations)']. Default is None.
    title : list or None. List of titles for the two plots.
            If None, defaults to ['(a) Forward model errors', '(b) Normalized forward model errors'].

    Returns
    -------
    fig : matplotlib.figure.Figure

    Parameters mirror fig_rmse_bars plus `daytime` mask.
    """
    # Defaults
    channels = channels if channels is not None else np.arange(7, 17)
    colors = colors if colors is not None else ['#D81B60', '#1E88E5', '#FFC107', 'r']
    labels = labels if labels is not None else ['Cloudy', 'Clear Sky', 'Day', 'Night']
    x_range = x_range if x_range is not None else [(0, 1.6), (0, 2.0)]
    y_label = y_label if y_label is not None else ['Channels', 'Channels']
    x_label = x_label if x_label is not None else [f'RMSE (K)', f'RMSE (Standard Deviations)']
    title = title if title is not None else [f'(a) Forward model errors', f'(b) Normalized forward model errors']

    # Ensure masks
    clr = np.asarray(clrsky, dtype=bool)
    day = np.asarray(daytime, dtype=bool)

    # Safe helpers
    def safe_rmse_from_slices(pred_slice, targ_slice):
        if pred_slice.size == 0 or targ_slice.size == 0:
            return np.full(10, np.nan)
        return rmse(pred_slice - targ_slice, axis=0)

    def safe_rmse_norm_from_slices(pred_slice, targ_slice, std_slice):
        if pred_slice.size == 0 or targ_slice.size == 0 or std_slice.size == 0:
            return np.full(10, np.nan)
        denom = np.where(std_slice == 0, np.nan, std_slice)
        return rmse((pred_slice - targ_slice) / denom, axis=0)

    # Extract arrays
    pred = pred
    targ = target
    std = pred[:, 10:] if pred.ndim == 2 and pred.shape[1] >= 20 else (pred[:, 10:] if pred.ndim >= 2 else None)

    # Compute raw errors
    err_cloudy = safe_rmse_from_slices(pred[~clr, :10], targ[~clr, :10])
    err_clear = safe_rmse_from_slices(pred[clr, :10], targ[clr, :10])
    err_day = safe_rmse_from_slices(pred[day, :10], targ[day, :10])
    err_night = safe_rmse_from_slices(pred[~day, :10], targ[~day, :10])

    # Compute normalized errors (protect zero std)
    norm_cloudy = safe_rmse_norm_from_slices(pred[~clr, :10], targ[~clr, :10], std[~clr]) if std is not None else np.full(10, np.nan)
    norm_clear = safe_rmse_norm_from_slices(pred[clr, :10], targ[clr, :10], std[clr]) if std is not None else np.full(10, np.nan)
    norm_day = safe_rmse_norm_from_slices(pred[day, :10], targ[day, :10], std[day]) if std is not None else np.full(10, np.nan)
    norm_night = safe_rmse_norm_from_slices(pred[~day, :10], targ[~day, :10], std[~day]) if std is not None else np.full(10, np.nan)

    values_raw = [err_cloudy, err_clear, err_day, err_night]
    values_norm = [norm_cloudy, norm_clear, norm_day, norm_night]

    # Create flexible grid (2 columns)
    cell_widths = [4.0, 4.0]
    cell_heights = [4.0]
    lefts = [0.75, 0.75]
    rights = [0.75, 0.75]
    bottoms = [0.75]
    tops = [0.75]
    fig, get_axes = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)
    ax0 = get_axes(0, 0)
    ax1 = get_axes(0, 1)

    # Plot raw and normalized using the wrapper
    plot_rmse_bars2(ax0, values_raw, channels, height=height, colors=colors, labels=labels, x_range=x_range[0],
                    font_size=13, title=title[0], x_label=x_label[0], y_label=y_label[0])
    plot_rmse_bars2(ax1, values_norm, channels, height=height, colors=colors, labels=labels, x_range=x_range[1],
                    font_size=13, title=title[1], x_label=x_label[1], y_label=y_label[1])

    # Save plot
    if figname:
        save_plot(fig, figname)

    return fig


def histedges_equalN(x, nbin):
    """
    Compute histogram edges that split data `x` into `nbin` bins with approximately equal counts.

    Parameters
    ----------
    x : array_like
        1D array of numeric values to bin.
    nbin : int
        Number of histogram bins desired.

    Returns
    -------
    numpy.ndarray
        Array of length `nbin + 1` containing bin edges.
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


def plot_errs_by_channel_vert(ax: plt.Axes, ypred: np.ndarray, bt: np.ndarray, j: int,
                              nbins: int = 10, xmax_percentile: float = 99.0, xmax_factor: float = 2.0,
                              beta: float = 0.5, font_size: int = 13, title: Optional[str] = None,
                              title_pad: float = 1.005, x_range: Optional[list] = None,
                              x_label: Optional[str] = None, y_label: Optional[str] = None,
                              y_labelpad=5, x_labelpad=3,
                              tickw: float = 1.0, tickl: float = 2.5, tickdir: str = 'out',
                              lg_loc='lower right', lg_font=10,
                              lg_ncol=1, lg_npoints=1, lg_scale=4.0, lg_spacing=0.05):
    """
    Plot binned predicted/std vs actual error statistics for one channel with 95% CI.
    Layout and styling mimic `plot_vertical_profiles` style.
    """

    # Extract arrays, flatten
    err = (ypred[:, j] - bt[:, j]).ravel()
    # predicted std expected at column 10 + j if available
    std_col = 10 + j
    if ypred.ndim == 2 and ypred.shape[1] > std_col:
        std_pred = ypred[:, std_col].ravel()
    else:
        std_pred = np.full_like(err, np.nan)

    # xmax from percentile of absolute actual errors
    try:
        xmax = float(np.nanpercentile(np.abs(err), xmax_percentile)) * xmax_factor
    except Exception:
        xmax = np.nan

    if not np.isfinite(xmax) or xmax <= 0:
        xmax = np.nanmax(np.abs(std_pred[np.isfinite(std_pred)])) if np.any(np.isfinite(std_pred)) else 1.0

    # candidate bins: linear and equal-count, then blend
    bins1 = np.linspace(0.0, xmax, nbins + 1)
    masked_std = std_pred[np.isfinite(std_pred) & (std_pred < xmax)]
    bins2 = histedges_equalN(masked_std, nbins)
    edges = bins1 * beta + bins2 * (1.0 - beta)

    # Histogram counts and per-bin statistics
    count, _ = np.histogram(std_pred, bins=edges)
    std_pred_mean = np.full(nbins, np.nan)
    std_pred_std = np.full(nbins, np.nan)
    std_actual = np.full(nbins, np.nan)

    for i in range(nbins):
        mask = (std_pred >= edges[i]) & (std_pred < edges[i+1])
        sel = std_pred[mask]
        if sel.size > 0:
            std_pred_mean[i] = np.nanmean(sel)
            std_pred_std[i] = np.nanstd(sel)
            err_sel = err[mask]
            std_actual[i] = rmse(err_sel) if err_sel.size > 0 else np.nan

    # uncertainties: protect divisions by zero / nan
    with np.errstate(invalid='ignore', divide='ignore'):
        denom_x = np.sqrt(count)
        denom_x[denom_x == 0] = np.nan
        xerr = 1.96 * std_pred_std / denom_x

        denom_y = np.sqrt(2.0 * count)
        denom_y[denom_y == 0] = np.nan
        yerr = 1.96 * std_actual / denom_y

    # Main plot (errorbar)
    eb = ax.errorbar(std_pred_mean, std_actual, xerr=xerr, yerr=yerr, color='k',
                     marker='.', linestyle='', capsize=2, capthick=0.5, markersize=4, linewidth=0.5,
                     label='NN - 95% CI')

    # y=x reference
    max_val = np.nanmax(std_actual) if np.isfinite(np.nanmax(std_actual)) else np.nanmax(std_pred_mean)
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = xmax if np.isfinite(xmax) and xmax > 0 else 1.0

    # Axis styling similar to plot_vertical_profiles
    if isinstance(x_range, tuple):
        xlim = x_range
    elif isinstance(x_range, list):
        xlim = x_range[j]
    else:
        xlim = [0.0, 1.125*max_val]
    ylim = xlim
    yx = ax.plot(xlim, ylim, 'r--', linewidth=0.6, label='y=x')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=font_size, labelpad=x_labelpad)
    else:
        ax.set_xlabel('Predicted Error StDev (K)', fontsize=font_size, labelpad=x_labelpad)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size, labelpad=y_labelpad)
    else:
        ax.set_ylabel('Actual Error StDev (K)', fontsize=font_size, labelpad=5)
    ax.tick_params(axis='both', which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size - 2,
                   bottom=True, top=True, left=True, right=True)

    # Secondary axis: cumulative distribution of |predicted std|
    ax2 = ax.twinx()
    valid = std_pred[np.isfinite(std_pred)]
    if valid.size > 0:
        xs = np.sort(np.abs(valid))
        ys = np.linspace(0.0, 1.0, xs.size)
        cdf_line, = ax2.plot(xs, ys, linestyle=':', color='b', label='CDF')
        ax2.set_ylim(0.0, 1.0)
    else:
        cdf_line = None
    ax2.tick_params(axis='y', which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size - 2)
    ax2.set_ylabel('Cumulative distribution function (no units)', fontsize=font_size, labelpad=5)

    # Title / channel label
    title = title if title is not None else f'Channel {j+1}'
    ax.set_title(title, fontsize=font_size, y=title_pad)

    # Set legend
    # Legend composition
    handles = []
    labels = []
    if eb is not None:
        # errorbar's first element is Line2D for markers in many mpl versions
        try:
            h = eb[0]
        except Exception:
            h = None
        if h is not None:
            handles.append(h); labels.append('NN')
    if yx:
        handles.append(yx[0]); labels.append('y=x')
    if cdf_line is not None:
        handles.append(cdf_line); labels.append('CDF')
    if handles:
        ax.legend(handles, labels, loc=lg_loc, fontsize=lg_font, labelspacing=lg_spacing, numpoints=lg_npoints,
                  ncol=lg_ncol, markerscale=lg_scale, fancybox=False)


def plot_errs_by_channel(ax: plt.Axes, ypred: np.ndarray, bt: np.ndarray, j: int,
                         yref: np.ndarray = None,
                         nbins: int = 10, xmax_percentile: float = 99.0, xmax_factor: float = 2.0,
                         beta: float = 0.5, font_size: int = 13, title: Optional[str] = None,
                         title_pad: float = 1.005, x_range: Optional[list] = None,
                         x_label: Optional[str] = None, y_label: Optional[str] = None,
                         y_labelpad=5, x_labelpad=3,
                         tickw: float = 1.0, tickl: float = 2.5, tickdir: str = 'out',
                         lg_loc='lower right', lg_font=10,
                         lg_ncol=1, lg_npoints=1, lg_scale=2.0, lg_spacing=0.05):
    """
    Plot binned predicted/std vs actual error statistics for one channel.
    If `yref` is provided it is treated as a reference: edges/xmax are computed from the reference
    (if possible) and both reference and prediction are overplotted using the same bins.
    """

    # helper to extract err and predicted std column
    def _extract_err_std(y, bt, col_j):
        err = (y[:, col_j] - bt[:, col_j]).ravel()
        std_col = 10 + col_j
        if y.ndim == 2 and y.shape[1] > std_col:
            std = y[:, std_col].ravel()
        else:
            std = np.full_like(err, np.nan)
        return err, std

    # extract for prediction
    err_p, std_p = _extract_err_std(ypred, bt, j)

    # extract for reference if provided
    if yref is not None:
        err_r, std_r = _extract_err_std(yref, bt, j)
    else:
        err_r, std_r = None, None

    # choose arrays to compute bins/xmax: prefer reference when available and finite
    use_err = None
    use_std = None
    if err_r is not None and np.any(np.isfinite(err_r)):
        use_err = err_r
    else:
        use_err = err_p

    if std_r is not None and np.any(np.isfinite(std_r)):
        use_std = std_r
    else:
        use_std = std_p

    # compute xmax from chosen errors
    try:
        xmax = float(np.nanpercentile(np.abs(use_err), xmax_percentile)) * xmax_factor
    except Exception:
        xmax = np.nan
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = np.nanmax(np.abs(use_std[np.isfinite(use_std)])) if np.any(np.isfinite(use_std)) else 1.0

    # candidate bins: linear and equal-count, then blend
    bins1 = np.linspace(0.0, xmax, nbins + 1)
    masked_std = use_std[np.isfinite(use_std) & (use_std < xmax)]
    bins2 = histedges_equalN(masked_std, nbins) if masked_std.size > 0 else bins1.copy()
    edges = bins1 * beta + bins2 * (1.0 - beta)

    # per-bin stats function
    def _per_bin_stats(std_arr, err_arr):
        n = nbins
        count, _ = np.histogram(std_arr, bins=edges)
        mean = np.full(n, np.nan)
        sdev = np.full(n, np.nan)
        actual = np.full(n, np.nan)
        for i in range(n):
            mask = (std_arr >= edges[i]) & (std_arr < edges[i+1])
            sel = std_arr[mask]
            if sel.size > 0:
                mean[i] = np.nanmean(sel)
                sdev[i] = np.nanstd(sel)
                err_sel = err_arr[mask]
                actual[i] = rmse(err_sel) if err_sel.size > 0 else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            denom_x = np.sqrt(count).astype(float)
            denom_x[denom_x == 0] = np.nan
            xerr = 1.96 * sdev / denom_x
            denom_y = np.sqrt(2.0 * count).astype(float)
            denom_y[denom_y == 0] = np.nan
            yerr = 1.96 * actual / denom_y
        return mean, actual, xerr, yerr, count

    # compute stats for reference (if any) and prediction
    if err_r is not None:
        mean_r, actual_r, xerr_r, yerr_r, count_r = _per_bin_stats(std_r, err_r)
    else:
        mean_r = actual_r = xerr_r = yerr_r = count_r = None

    mean_p, actual_p, xerr_p, yerr_p, count_p = _per_bin_stats(std_p, err_p)

    # Main plotting: reference first (if present), then prediction
    handles = []
    labels = []

    # y=x reference using combined extent
    all_actuals = np.hstack([a for a in [actual_r, actual_p] if a is not None])
    all_means = np.hstack([m for m in [mean_r, mean_p] if m is not None])
    max_val = np.nanmax(all_actuals) if np.isfinite(np.nanmax(all_actuals)) else np.nanmax(all_means)
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = xmax if np.isfinite(xmax) and xmax > 0 else 1.0

    if isinstance(x_range, tuple):
        xlim = x_range
    elif isinstance(x_range, list):
        xlim = x_range[j]
    else:
        xlim = [0.0, 1.125 * max_val]
    ylim = xlim
    yx = ax.plot(xlim, ylim, 'k--', linewidth=0.6, label='y=x')
    handles.append(yx[0]); labels.append('y=x')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if mean_r is not None:
        label = 'Paper - 95% CI'
        h_ref = ax.errorbar(mean_r, actual_r, xerr=xerr_r, yerr=yerr_r, color=colors['blue'],
                            marker='.', linestyle='', capsize=2, capthick=1, markersize=6, linewidth=1,
                            label=label)
        try:
            handles.append(h_ref[0])
            labels.append(label)
        except Exception:
            pass

    label = 'Model - 95% CI'
    eb = ax.errorbar(mean_p, actual_p, xerr=xerr_p, yerr=yerr_p, color=colors['red'],
                     marker='.', linestyle='', capsize=2, capthick=1, markersize=6, linewidth=1,
                     label=label)
    try:
        handles.append(eb[0])
        labels.append(label)
    except Exception:
        pass

    # Labels and ticks
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=font_size, labelpad=x_labelpad)
    else:
        ax.set_xlabel('Predicted Error StDev (K)', fontsize=font_size, labelpad=x_labelpad)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size, labelpad=y_labelpad)
    else:
        ax.set_ylabel('Actual Error StDev (K)', fontsize=font_size, labelpad=5)
    ax.tick_params(axis='both', which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size - 2,
                   bottom=True, top=True, left=True, right=True)

    # Secondary axis: CDFs for absolute predicted std (plot both if available)
    ax2 = ax.twinx()
    cdf_handles = []
    if std_r is not None:
        valid_r = std_r[np.isfinite(std_r)]
        if valid_r.size > 0:
            xs = np.sort(np.abs(valid_r))
            ys = np.linspace(0.0, 1.0, xs.size)
            h_r, = ax2.plot(xs, ys, linestyle=':', color=colors['blue'], label='Paper - CDF')
            cdf_handles.append(h_r)
    valid_p = std_p[np.isfinite(std_p)]
    if valid_p.size > 0:
        xs = np.sort(np.abs(valid_p))
        ys = np.linspace(0.0, 1.0, xs.size)
        h_p, = ax2.plot(xs, ys, linestyle=':', color=colors['red'], label='Model - CDF')
        cdf_handles.append(h_p)

    if cdf_handles:
        # include CDF in legend set
        handles.extend(cdf_handles)
        labels.extend([h.get_label() for h in cdf_handles])
        ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis='y', which='both', direction=tickdir, width=tickw, length=tickl, labelsize=font_size - 2)
    ax2.set_ylabel('Cumulative distribution function (no units)', fontsize=font_size, labelpad=5)

    # Title
    title = title if title is not None else f'Channel {j+1}'
    ax.set_title(title, fontsize=font_size, y=title_pad)

    # Legend
    if handles:
        ax.legend(handles, labels, loc=lg_loc, fontsize=lg_font, labelspacing=lg_spacing, numpoints=lg_npoints,
                  ncol=lg_ncol, markerscale=lg_scale, fancybox=False)


def fig_errs_by_channel(target: np.ndarray, pred: np.ndarray, ref: np.ndarray=None, figname: Optional[str] = None,
                        nbins: int = 10, xmax_percentile: float = 99.0, xmax_factor: float = 2.0,
                        beta: float = 0.5, channels: Optional[Union[list, np.ndarray]] = None,
                        font_size: int = 13, x_range: Optional[list] = None,
                        x_label: Union[str, list] = 'Predicted Error StDev (K)',
                        y_label: Union[str, list] = 'Actual Error StDev (K)',
                        title: Optional[Union[str, list]] = None,
                        orientation: str = 'vertical'):
    """
    Create a multi-panel layout and plot errors per channel using plot_errs_by_channel_vert.
    orientation: 'vertical' (default, 2 columns) or 'horizontal' (2 rows).
    """
    if channels is None:
        channels = np.arange(7, 17)
    channels = list(channels)
    nchan = len(channels)

    # Determine layout from orientation
    orientation = orientation.lower()
    if orientation not in ('vertical', 'horizontal'):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if orientation == 'vertical':
        ncols = 2
        nrows = int(np.ceil(nchan / ncols))
    else:  # horizontal -> 2 rows
        nrows = 2
        ncols = int(np.ceil(nchan / nrows))

    # cell sizes and paddings (per-column/row)
    cell_widths = [4.0] * ncols
    cell_heights = [4.0] * nrows
    lefts = [0.75] * ncols
    rights = [0.75] * ncols
    bottoms = [0.75] * nrows
    tops = [0.75] * nrows

    fig, get_ax = flexible_gridspec(cell_widths, cell_heights, lefts, rights, bottoms, tops)

    def pick(val, idx, default=None):
        if val is None:
            return default
        if isinstance(val, (list, tuple, np.ndarray)):
            return val[idx] if idx < len(val) else default
        return val

    for idx in range(nchan):
        row = idx // ncols
        col = idx % ncols
        ax = get_ax(row, col)

        per_title = pick(title, idx, f'Channel {channels[idx]}')
        per_xlabel = pick(x_label, idx, None)
        per_ylabel = pick(y_label, idx, None)

        # call per-channel plotting function (keeps original j indexing as before)
        if ref is not None:
            plot_errs_by_channel(ax, pred, target, j=idx,
                                 yref=ref,
                                 nbins=nbins,
                                 xmax_percentile=xmax_percentile,
                                 xmax_factor=xmax_factor,
                                 beta=beta,
                                 font_size=font_size,
                                 title=per_title,
                                 x_label=per_xlabel,
                                 y_label=per_ylabel,
                                 x_range=x_range)
        else:
            plot_errs_by_channel_vert(ax, pred, target, j=idx, nbins=nbins,
                                      xmax_percentile=xmax_percentile, xmax_factor=xmax_factor,
                                      beta=beta, font_size=font_size, title=per_title,
                                      x_label=per_xlabel, y_label=per_ylabel, x_range=x_range)

    # Save plot
    if figname:
        save_plot(fig, figname)

    return fig


if __name__ == "__main__":

    print("No example currently implemented for this module.")
    