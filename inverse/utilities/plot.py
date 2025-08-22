import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density
from  forward.evaluation.metrics import rmse
from typing import Union
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


def flexible_gridspec(row_col_counts: list[int], cell_width: float=4.0, cell_height: float=4.0,
                      left: float=0.8, right: float=0.8, bottom: float=0.58, top: float=0.42) \
        -> tuple[plt.Figure, callable]:
    """ Create a figure with the flexible grid layout.

    Parameters
    ----------
    row_col_counts: list of ints, number of columns in each row.
    cell_width, cell_height: size of each cell in inches.
    left, right, bottom, top: margins in inches.

    Returns
    -------
    fig, get_ax(row, col)
    """

    # Extract the number of rows and columns
    nrows = len(row_col_counts)
    max_cols = max(row_col_counts)
    # Compute total width and height
    fig_width = max_cols * (cell_width + left + right)
    fig_height = nrows * (cell_height + top + bottom)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    # Precompute x/y positions for each cell
    x_starts = []
    for cols in row_col_counts:
        row_x = [(i+1)*left + i*(cell_width + right) for i in range(cols)]
        x_starts.append(row_x)
    y_starts = [(nrows-r)*bottom + (nrows - 1 - r) * (cell_height + top) for r in range(nrows)]

    # Axes layout for each cell
    def get_ax(row: int, col: int) -> plt.Axes:
        """ Establish layout for gridspec.

        Parameters
        ----------
        row: int, row index.
        col: int, column index.

        Returns
        -------
        ax: matplotlib.axes.Axes. Axes for the specified row and column.
        """

        # Compute normalized properties for [left, bottom, width, height]
        x0 = x_starts[row][col] / fig_width
        y0 = y_starts[row] / fig_height
        w = cell_width / fig_width
        h = cell_height / fig_height

        # Create the axes
        return fig.add_axes((x0, y0, w, h))

    return fig, get_ax


def apply_colorbar(ax, plot, size=0.05*4/(0.8+0.8+4),
                   font_size=13, label='Density', label_pad=15.5, pad=0.05,
                   axis="y", orientation="vertical", rotation=270, side='right',
                   ticks=None, tickw=1, tickl=2.5, tickdir='out') -> plt.colorbar:
    """ Add a colorbar to the given axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to add the colorbar to.
    plot : matplotlib.collections.Collection. The plot to which the colorbar is associated.
    size : float. Size of the colorbar.
    font_size : int. Font size for the colorbar ticks.
    label : str. Label for the colorbar.
    label_pad : float. Padding for the colorbar label.
    pad : float. Padding for the colorbar.
    axis : str. Axis for the colorbar. Default is 'y'.
    orientation : str. Orientation of the colorbar. Default is 'vertical'.
    rotation : int. Rotation of the colorbar label. Default is 270.
    side : str. Side for the colorbar. Default is 'right'.
    ticks : int. Number of ticks on the colorbar.
    tickw : float. Width of the ticks.
    tickl : float. Length of the ticks.
    tickdir : str. Direction of the ticks. Default is 'out'.

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar. The colorbar object.
    """

    # Create a divider for the axes
    divider = make_axes_locatable(ax)

    # Add a new axis for the colorbar
    cax = divider.append_axes(side, size=size, pad=pad)

    # Create the colorbar
    cb = colorbar(plot, extend='neither', cax=cax)
    # Set colorbar ticks
    cb.ax.tick_params(axis=axis, direction=tickdir, labelsize=font_size, width=tickw, length=tickl)
    if ticks is not None:
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(ticks))

    # Force scientific notation
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((0, 0))
    cb.formatter.set_useMathText(True)

    # Set colorbar label
    if label is not None:
        # Set rotation based on orientation
        if rotation is None:
            rotation = 270 if orientation == 'vertical' else 0
        cb.set_label(label, labelpad=label_pad, rotation=rotation, size=font_size)

    return cb


def compute_min_max(data, symmetric=False):
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


def save_plot(fig, filename, fileformat='png', dpi=300):
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


def scatterplot(fig, ax, x, y, font_size=13, projection=None, title='Scatterplot', title_pad=1.005,
                x_label='Reference', y_label='Inference', y_labelpad=5, x_labelpad=3, xy_symmetric=True,
                x_range: tuple=None, y_range: tuple=None, x_nticks=6, y_nticks=6, tickw=1, tickl=2.5, tickdir='out',
                marker='.', markersize=0.9, grid=True, grid_linew=0.5, x_ascale='linear', y_ascale='linear',
                ref_label='Reference (1:1)', ref_color='black', ref_linew=0.5, ref_lines='--',
                fit=None, fit_color=None, fit_linew=0.25, fit_lines='-',
                lg_loc='upper left', lg_font=10, lg_ncol=1, lg_npoints=1, lg_scale=4.0, lg_spacing=0.05,
                cb_label='Density', cb_size=0.2, cb_ticks=5, cb_axis="y", cb_pad=0, cb_tickw=1,
                cb_tickl=2.5, cb_dir='out', cb_rot=270, cb_labelpad=16, cb_side='right'):
    """ Create a scatterplot with optional density projection.

    Parameters
    ----------
    fig : matplotlib.figure.Figure. Figure to plot on.
    ax : matplotlib.axes.Axes. Axes to plot on.
    x : numpy.ndarray. X data.
    y : numpy.ndarray. Y data.
    font_size : int. Font size for the plot.
    projection : str. Projection type. Default is None. Use 'scatter_density' for density plots.
    title : str. Title of the plot.
    title_pad : float. Padding for the title.
    x_label : str. Label for the x-axis.
    y_label : str. Label for the y-axis.
    y_labelpad : float. Padding for the y-axis label.
    x_labelpad : float. Padding for the x-axis label.
    xy_symmetric : bool. If True, the x and y axes are symmetric.
    x_range : tuple. Range for the x-axis. If None, computed from data.
    y_range : tuple. Range for the y-axis. If None, computed from data.
    x_nticks : int. Number of ticks on the x-axis.
    y_nticks : int. Number of ticks on the y-axis.
    tickw : float. Width of the ticks.
    tickl : float. Length of the ticks.
    tickdir : str. Direction of the ticks. Default is 'out'.
    marker : str. Marker style for the scatterplot.
    markersize : float. Size of the markers.
    grid : bool. If True, show grid.
    grid_linew : float. Line width of the grid.
    x_ascale : str. Scale for the x-axis. Default is 'linear'.
    y_ascale : str. Scale for the y-axis. Default is 'linear'.
    ref_label : str. Label for the reference line.
    ref_color : str. Color for the reference line.
    ref_linew : float. Line width for the reference line.
    ref_lines : str. Line style for the reference line.
    fit : bool. If True, fit a line to the data.
    fit_color : str. Color for the fit line.
    fit_linew : float. Line width for the fit line.
    fit_lines : str. Line style for the fit line.
    lg_loc : str. Location of the legend.
    lg_font : int. Font size for the legend.
    lg_ncol : int. Number of columns in the legend.
    lg_npoints : int. Number of points in the legend.
    lg_scale : float. Scale for the legend markers.
    lg_spacing : float. Spacing between legend entries.
    cb_label : str. Label for the colorbar.
    cb_size : float. Size of the colorbar.
    cb_ticks : int. Number of ticks on the colorbar.
    cb_axis : str. Axis for the colorbar. Default is 'y'.
    cb_pad : float. Padding for the colorbar.
    cb_tickw : float. Width of the colorbar ticks.
    cb_tickl : float. Length of the colorbar ticks.
    cb_dir : str. Direction of the colorbar ticks. Default is 'out'.
    cb_rot : int. Rotation of the colorbar label.
    cb_labelpad : float. Padding for the colorbar label.
    cb_side : str. Side for the colorbar. Default is 'right'.

    Returns
    -------
    ax : matplotlib.axes.Axes. Axes with the scatterplot.
    """


    # Compute min and max values
    if x_range is None:
        x_range = compute_min_max(x, symmetric=xy_symmetric)
    if y_range is None:
        y_range = compute_min_max(y, symmetric=xy_symmetric)
    # Adopt a symmetric range
    if xy_symmetric:
        x_range = compute_min_max(x_range + y_range, symmetric=xy_symmetric)
        y_range = x_range
    # Aspect ratio
    ax.set_aspect(1)

    # Regular scatterplot
    if projection is None:
        scat = ax.scatter(x.flatten(), y.flatten(), c=colors['blue'], marker=marker, s=markersize)
    # Density scatterplot
    elif projection == 'scatter_density':
        pos = ax.get_position()
        fig.delaxes(ax)
        ax = fig.add_axes(pos, projection='scatter_density')
        scat = ax.scatter_density(x.flatten(), y.flatten(), cmap=white_viridis)
    else:
        raise ValueError("Projection not supported. Use None or 'scatter_density'.")

    # Plot reference 1:1 line
    ax.plot(x_range, x_range, label=ref_label, color=ref_color, linewidth=ref_linew, linestyle=ref_lines)

    # Compute and plot linear fit
    if fit:
        slope, y0 = np.polyfit(x.flatten(), y.flatten(), 1)
        # Plot linear fit
        fit_label = f"y = {slope:.3f}x + {y0:.3f}" if y0 > 0 else f"y = {slope:.3f}x - {abs(y0):.3f}"
        ax.plot(np.array(x_range), slope * np.array(x_range) + y0, label=fit_label,
                color=fit_color, linewidth=fit_linew, linestyle=fit_lines)

    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xscale(x_ascale)
    ax.set_yscale(y_ascale)

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

    # Set colorbar
    if projection == 'scatter_density':
        apply_colorbar(ax, scat, size=cb_size, font_size=font_size, label=cb_label, label_pad=cb_labelpad,
                       pad=cb_pad, axis=cb_axis, orientation='vertical', rotation=cb_rot, side=cb_side,
                       ticks=cb_ticks, tickw=cb_tickw, tickl=cb_tickl, tickdir=cb_dir)


def histogram(ax, data, bins=50, font_size=13, title='Histogram', title_pad=1.005,
                x_label='Value', y_label='Frequency', x_labelpad=3, y_labelpad=5,
                x_range=None, y_range=None, x_nticks=6, y_nticks=6,
                tickw=1, tickl=2.5, tickdir='out', grid=True, grid_linew=0.5,
                x_ascale='linear', y_ascale='linear'):

    # Create a histogram of the data
    hist, bin_edges = np.histogram(data, bins=bins, range=x_range)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Plot the histogram
    ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], color=colors['blue'], alpha=0.7)
    # Set axis limits
    if x_range is None:
        x_range = compute_min_max(data, symmetric=True)
    if y_range is None:
        y_range = (0, np.nanmax(hist) * 1.1)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xscale(x_ascale)
    ax.set_yscale(y_ascale)
    # Add grid
    if grid:
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


def fig_histograms(data, bins=50):

    """ Create a figure with histograms of the data.

    Parameters
    ----------
    data : numpy.ndarray. Data to create histograms for.
    bins : int. Number of bins for the histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure. Figure with histograms.
    ax : matplotlib.axes.Axes. Axes with histograms.
    """

    # Extract dimensions
    n_samples, n_profiles, n_levels = data.shape
    # Create a flexible gridspec
    n_rows = int(np.ceil(np.sqrt(n_profiles)))
    n_cols = int(np.ceil(n_profiles / n_rows))
    list_cols = [n_cols for _ in range(n_rows)]
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the histograms for each channel
        plot_title = f"Profile {i+1} Histogram"
        histogram(ax, data[:, i, :].flatten(), bins=bins, font_size=13, title=plot_title,
                  x_label='Value', y_label='Frequency', x_range=None, y_range=None,
                  x_nticks=6, y_nticks=6, tickw=1, tickl=2.5, tickdir='out',
                  grid=True, grid_linew=0.5, x_ascale='linear', y_ascale='linear')

    return fig


def plot_map(ax, img, img_alpha=1.0, img_norm='linear', img_coord=(0, 0), img_shape=None, img_pixel=(1, 1),
             img_ticks=None, img_labels=('y-axis', 'x-axis'), title=None, img_labelspad=(5, 3), img_tickw=1,
             img_tickl=2.5, img_tickdir='out', title_pad=1.005, cb_label=None, img_range=None, cb_cmap=None,
             cb_pad=0, cb_tickw=1, cb_tickl=2.5, cb_font=12, cb_dir='out', cb_rot=270, cb_labelpad=15.5,
             cb_side='right', cb_size=0.2, cb_ticks=5, cb_axis="y",
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
    cb_axis : str. Axis for the colorbar. Default is 'y'.
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
    ax.get_yaxis().set_tick_params(
        which='both', direction=img_tickdir, width=img_tickw, length=img_tickl,
        labelsize=cb_font, left=True, right=True
    )
    ax.get_xaxis().set_tick_params(
        which='both', direction=img_tickdir, width=img_tickw, length=img_tickl,
        labelsize=cb_font, bottom=True, top=True
    )
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
        apply_colorbar(ax, I, size=cb_size, font_size=cb_font, label=cb_label, label_pad=cb_labelpad,
                       pad=cb_pad, axis=cb_axis, orientation='vertical', rotation=cb_rot, side=cb_side,
                       ticks=cb_ticks, tickw=cb_tickw, tickl=cb_tickl, tickdir=cb_dir)


def plot_vertical_profile(ax, data, font_size=13, title='Mean vertical profile ± std', title_pad=1.005, y=None,
                          x_label=r'Profile value ($\sigma$)', y_label='Height (levels)', y_labelpad=5, x_labelpad=3,
                          x_range: tuple=None, y_range: tuple=None, x_nticks=6, y_nticks=6, y_invert=True, tickw=1,
                          tickl=2.5, tickdir='out', color=None, linew=0.5, lines='-', label = None, alpha=0.2,
                          grid=True, grid_linew=0.5, x_ascale='linear', y_ascale='linear', lg_loc='best', lg_font=10,
                          lg_ncol=1, lg_npoints=1, lg_scale=4.0, lg_spacing=0.05):
    """ Plot a vertical profile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes. Axes to plot on.
    data : numpy.ndarray. Data to plot. Shape should be (n_samples, n_levels).
    font_size : int. Font size for the plot.
    title : str. Title of the plot.
    title_pad : float. Padding for the title.
    y : numpy.ndarray. Y data. If None, computed from data.
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
        n_levels = data[0].shape[0]
        data_mean = np.stack([np.mean(data[i], axis=0) if data[i].ndim > 1 else data[i] for i in range(len(data))], axis=0)
        data_std = np.stack([np.std(data[i], axis=0) if data[i].ndim > 1 else np.zeros_like(data[i]) for i in range(len(data))], axis=0)
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            n_levels = data.shape[0]
            data_mean = data
            data_std = np.zeros_like(data)
        elif data.ndim == 2:
            n_levels = data.shape[1]
            # Compute mean and standard deviation
            data_mean = data.mean(axis=0)
            data_std = data.std(axis=0)
        else:
            raise ValueError("Data should be 1D or 2D array with shape (n_samples, n_levels) or (n_levels,).")
        heights = np.arange(n_levels) if y is None else y

    # Compute min and max values
    if x_range is None:
        x_range = compute_min_max(np.stack([data_mean - data_std if np.min(data_mean) < 0 else np.clip(data_mean - data_std, 0, None),
                                            data_mean + data_std]), symmetric=True)
        x_range = (0.95 * x_range[0] if x_range[0] >= 0 else 1.05 * x_range[0],
                   1.05 * x_range[1] if x_range[1] >= 0 else 0.95 * x_range[1])
    if y_range is None:
        y_range = compute_min_max(heights, symmetric=False)

    # Plot mean and standard deviation
    color = colors['blue'] if color is None else color
    ax.plot(data_mean, heights, color=color, label=label, linewidth=linew, linestyle=lines)
    ax.fill_betweenx(heights, data_mean - data_std, data_mean + data_std, alpha=alpha, color=color)

    # Plot reference 1:1 line
    if x_range[0] * x_range[1] < 0:
        ax.plot([np.mean(x_range), np.mean(x_range)], [np.min(heights), np.max(heights)],
                label='Reference', color='black', linewidth=0.5, linestyle='--')
        if y is not None:
            ax.plot([np.min(x_range), np.max(x_range)], [100., 100.],
                    label='100 hpa', color='gray', linewidth=0.5, linestyle='--')

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


def fig_vertical_profile(data: np.ndarray, y: np.ndarray=None, y_label: Union[list, str]=None,
                         title: Union[list, str] = None):
    """ Plot vertical profiles for a given 3D data array using flexible gridspec.

    Parameters
    ----------
    data : np.ndarray. Data to plot. Shape should be (n_samples, n_profiles, n_levels).
    y : np.ndarray, optional. Vertical levels. If None, levels will be generated as 0, 1, ..., n_levels-1.
    y_label : Union[list[str, ...], str], optional. Labels for the vertical levels. If None, default labels will be used.
    title : str, optional. Title for the plots. If None, default titles will be used.

    Returns
    -------
    fig : matplotlib.figure.Figure. Figure with vertical profiles for each channel.
    """

    # Check shape
    if data.ndim != 3:
        raise ValueError("Data should be a 3D array with shape (n_samples, n_profiles, n_levels).")

    # Extract dimensions
    n_samples, n_profiles, n_levels = data.shape
    # Create a flexible gridspec
    n_rows = int(np.ceil(np.sqrt(n_profiles)))
    n_cols = int(np.ceil(n_profiles / n_rows))
    list_cols = [n_cols for _ in range(n_rows)]
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = f'Vertical profile #{i + 1}' if title is None else title[i]
        plot_vertical_profile(ax, data[:, i, :], y=y, y_label=y_label, title=plot_title, label=['Errors'])

    return fig


def fig_vertical_profiles(target: np.ndarray, pred: np.ndarray, y: np.ndarray=None, y_label: Union[list, str]=None,
                          title: Union[list, str] = None, x_range=None):
    """
    Plot vertical profiles for target and prediction data using flexible gridspec.

    Parameters
    ----------
    target : np.ndarray. Target data to plot. Shape should be (n_samples, n_profiles, n_levels).
    pred : np.ndarray. Prediction data to plot. Shape should be (n_samples, n_profiles, n_levels).
    y : np.ndarray, optional. Vertical levels. If None, levels will be generated as 0, 1, ..., n_levels-1.
    y_label : Union[list[str, ...], str], optional. Labels for the vertical levels. If None, default labels will be used.
    title : str, optional. Title for the plots. If None, default titles will be used.

    Returns
    -------
    fig : matplotlib.figure.Figure. Figure with vertical profiles for target and prediction.
    """

    # Check shapes
    if target.ndim != 3 or pred.ndim != 3:
        raise ValueError("Both target and prediction should be 3D arrays with shape (n_samples, n_profiles, n_levels).")

    # Extract dimensions
    n_samples, n_profiles, n_levels = target.shape

    # Create a flexible gridspec
    # From n_profiles, determine optimal layout for flexible_gridspec
    n_rows = int(np.ceil(np.sqrt(n_profiles)))
    n_cols = int(np.ceil(n_profiles / n_rows))
    # Create a flexible gridspec
    list_cols = [n_cols for _ in range(n_rows)]
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = f'Vertical profile #{i + 1}' if title is None else title[i]
        plot_vertical_profiles(ax, np.stack([target[:, i, :], pred[:, i, :]], axis=1), title=plot_title,
                               y=y, y_label=y_label, label=['Target', 'Prediction'], x_label='Profile value (units)',
                               x_range=x_range)

    return fig


def fig_vertical_profiles_background(target: np.ndarray, pred: np.ndarray, background: np.ndarray, background_err: np.ndarray,
                                     y: np.ndarray = None, y_label: Union[list, str] = None, title: Union[list, str] = None,
                                     x_range=None):
    """
    Plot vertical profiles for target and prediction data using flexible gridspec.

    Parameters
    ----------
    target : np.ndarray. Target data to plot. Shape should be (n_samples, n_profiles, n_levels).
    pred : np.ndarray. Prediction data to plot. Shape should be (n_samples, n_profiles, n_levels).
    title : str, optional. Title for the plots. If None, default titles will be used.
    background : np.ndarray. Background data to plot. Shape should be (n_samples, n_profiles, n_levels).
    background_err : np.ndarray. Background error data to plot. Shape should be (n_samples, n_profiles, n_levels).
    y : np.ndarray, optional. Vertical levels. If None, levels will be generated as 0, 1, ..., n_levels-1.
    y_label : Union[list[str, ...], str], optional. Labels for the vertical levels. If None, default labels will be used.

    Returns
    -------
    fig : matplotlib.figure.Figure. Figure with vertical profiles for target and prediction.
    """

    # Check shapes
    if target.ndim != 3 or pred.ndim != 3:
        raise ValueError("Both target and prediction should be 3D arrays with shape (n_samples, n_profiles, n_levels).")

    # Extract dimensions
    n_samples, n_profiles, n_levels = target.shape

    # Create a flexible gridspec
    # From n_profiles, determine optimal layout for flexible_gridspec
    n_rows = int(np.ceil(np.sqrt(n_profiles)))
    n_cols = int(np.ceil(n_profiles / n_rows))
    # Create a flexible gridspec
    list_cols = [n_cols for _ in range(n_rows)]
    fig, get_axes = flexible_gridspec(list_cols, cell_width=4, cell_height=4)

    # Loop over profiles
    for i in range(n_profiles):
        ax = get_axes(i // n_cols, i % n_cols)
        # Plot the vertical profile for each channel
        plot_title = f'Vertical profile #{i + 1}' if title is None else title[i]
        plot_vertical_profiles(ax, np.stack([background[:, i, :], target[:, i, :], pred[:, i, :]], axis=1),
                               err=np.stack([background_err[:, i, :].mean(axis=0), target[:, i, :].std(axis=0),
                                             pred[:, i, :].std(axis=0)], axis=0), y=y, y_label=y_label,
                               title=plot_title, color = [colors['green'], colors['blue'], colors['orange']],
                               label=['Background', 'Target', 'Prediction'], x_label='Profile value (units)',
                               x_range=x_range)

    return fig


def plot_rmse_bars(ax, values, positions, height=0.3, colors=None, labels=None, x_range=None, x_ascale='linear',
                   font_size=13, title='RMSE by channel', title_pad=1.005, x_label='RMSE (units)', y_label='Channels',
                   y_labelpad=5, x_labelpad=3, tickw=1, tickl=2.5, tickdir='out', lg_loc='best', lg_font=10,
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
        ax.bar_label(bars, fmt="%.2f", fontsize=10)

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

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    # Axis
    channels = channels if channels is not None else np.arange(7, 17)
    y_label = y_label if y_label is not None else ['Channels', 'Channels']
    x_label = x_label if x_label is not None else [f'RMSE (K)', f'RMSE (Standard Deviations)']
    x_range = x_range if x_range is not None else [None, None]  #[(0, 1.6), (0, 2.0)]
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
    fig, get_ax = flexible_gridspec([2])
    ax0 = get_ax(0, 0)
    ax1 = get_ax(0, 1)

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


if __name__ == "__main__":

    ex_path = "E:\\Data\\ISSI_Team_Flows\\Matthias\\SSD_25x8Mm_16_pdmp_1_ISSI_Flows\\2D"  # os.path.abspath("../E/Data/ISSI_Team_Flows/Matthias/SSD_25x8Mm_16_pdmp_1_ISSI_Flows/2D/")
    ex_slice_type = 'yz'
    ex_slice = [192]  # [192, 400]
    ex_iter = [15000]  # [0, 3900, 4200]
    ex_vars = ['I500', 'Bz']  # ['I500', 'vx', 'vy', 'vz', 'Bx', 'By', 'Bz']
    ex_x_min, ex_x_max, ex_y_min, ex_y_max = 0, 1536, 0, 1536

    # Initialize the dataset
    # dataset = MURaMQSDataset(ex_path, dataset=ex_slice_type)

    # Measure slice_reader
    # data1 = dataset.read(ex_iter, ex_slice, ex_vars,
    #                      x_min=ex_x_min, x_max=ex_x_max, y_min=ex_y_min, y_max=ex_y_max)
    # vx = data1[0, :, :, 0, 1]
    # vy = data1[0, :, :, 0, 2]
    # vz = data1[0, :, :, 0, 3]
    # ic = (data1[0, :, :, 0, 0]-np.mean(data1[0, :, :, 0, 0]))/np.std(data1[0, :, :, 0, 0])
    # bz = data1[0, :, :, 0, 1]
    # Make a perturbed version of bz
    # ic2 = (100+0.1 * np.random.randn(*ic.shape))*ic

    # breakpoint()

    # Plotting example
    # figure, get_axes = flexible_gridspec([2, 1])  # 2 columns in the first row, 1 in the second

    # Second subplot: scatter_density
    # ax0 = get_axes(0, 0)
    # scatterplot(figure, ax0, ic, ic2, projection='scatter_density', title='Density Scatter', fit=True, xy_symmetric=False,
    #             x_range=(-3, 3))

    # ax2 = get_axes(0, 1)
    # plot_map(ax2, data1[0, :, :, 0, 0], cb_cmap='hot')

    # ax1 = get_axes(1, 0)
    # scatterplot(figure, ax1, bz, 3.0 * bz, projection='scatter_density', title='Density Scatter', fit=True)
    # plot_map(ax1, bz, img_range=(-100, 100), cb_cmap='Greys_r')

    # save_plot(figure, filename='plotting_example.png')
