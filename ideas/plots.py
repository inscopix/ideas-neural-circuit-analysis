import logging
import os
from typing import Literal

import bokeh
import cv2
import isx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from beartype import beartype
from beartype.typing import List, Optional, Union
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from matplotlib.ticker import FuncFormatter
from scipy import ndimage

# from ideas import io, measures, tracking
from ideas import io, measures
from ideas.exceptions import IdeasError
from ideas.types import NumpyFloat2DArray
from ideas.utils import _find_coord_start, get_file_size

logger = logging.getLogger()


def scale_figsize_by_fov(
    shape: tuple, base: tuple = (10, 10), padding: tuple = 0.2
) -> tuple:
    """Scale a base figure plot size by the shape of the input image data.
    Using the same figsize for different sized images leads to bad looking plots.
    This will dynamically resize the shape of the plot according to the aspect ratio
    of the image.

    :param shape tuple: The shape of the image
    :param base tuple: The base figsize of the plot.
    :param padding tuple: The percent of the smaller side to pad in the output fig shape.
        Otherwise the scaled figsize will be too tight on the smaller edge.

    :return tuple: The base fig size scaled by the aspect ratio of the image.
    """

    shape_contrib = (shape / np.max(shape)) + padding
    # threshold to ignore effect of padding on long side
    shape_contrib[shape_contrib > 1.0] = 1.0
    figsize = np.array(base) * shape_contrib
    return figsize.astype(int)


def _get_figsize(width, height, trajectory_x, trajectory_y):
    """Get the figure size based on the width and height of the image."""
    if width and height:
        shape = (width, height)
    else:
        # norm data to get shape of region containing all points
        norm_points = np.round((trajectory_x, trajectory_y)).astype(int)
        norm_points -= np.min(norm_points, axis=1)[:, None]
        shape = np.max(norm_points, axis=1) + 1

    return scale_figsize_by_fov(shape=shape, base=(10, 10))


def _find_zone_max(zones: pd.DataFrame):
    """
    Find the maximum x and y values of the zones.
    """
    x_upper = 0
    y_upper = 0
    for _, zone in zones.iterrows():
        if zone["Type"] == "ellipse":
            x_upper = max(x_upper, zone["X 0"] + zone[" Minor Axis"] / 2)
            y_upper = max(y_upper, zone["Y 0"] + zone["Major Axis"] / 2)
        elif zone["Type"] == "polygon" or zone["Type"] == "rectangle":
            num_points = _find_coord_start(zone, len(zone) / 2) + 1
            for i in range(num_points):
                x_upper = max(x_upper, zone[f"X {i}"])
                y_upper = max(y_upper, zone[f"Y {i}"])

    return x_upper, y_upper


def plot_trajectory(
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    preview_filename: Optional[str] = None,
    trajectory_x,
    trajectory_y,
    traj_cmap="flare",
    title: str = "Trajectory",
    x_label: str = "X (pixels)",
    y_label: str = "Y (pixels)",
    hue: Optional[Union[np.array, pd.Series]] = None,
    hue_label: str = "Frame Number",
    norm_hue: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
    invert_yaxis: bool = True,
    output_format: str = "png",
):
    """Generate a preview of trajectory in a FOV extracted from a behavior movie.
    The trajectory is colored by user-specified metadata, which by default is the frame number (i.e., time).

    Parameters:
    - ax: Optional matplotlib axes to use for plotting
    - fig: Optional matplotlib figure to use for plotting
    - preview_filename: Optional filename to save the plot as.
    - trajectory_x: An array of x-coordinates for the trajectory.
    - trajectory_y: An array of y-coordinates for the trajectory.
    - traj_cmap: The colormap for the trajectory plot (default: "flare").
    - title: The title of the plot (default: "Trajectory").
    - x_label: The label for the x-axis (default: "X (pixels)").
    - y_label: The label for the y-axis (default: "Y (pixels)").
    - hue: Optional array of values to use for coloring the trajectory.
        If empty, then default to the trajectory index, i.e., frame number or time
    - hue_label: The label for the colorbar (default: "Frame Number")
    - norm_hue: If true, the colorbar is normalized.
        Helpful if the colorbar is heavily skewed on one end.
    - width: Optional width to set for the plot.
        Used to render the full FOV of the behavior movie.
    - height: Optional height to set for the plot.
        Used to render the full FOV of the behavior movie.
    - invert_yaxis: If true, the y-axis is reflected. By default, the origin of matplotlib
        plots is the top-left corner. However, most behavior movies have an origin in the
        bottom-left corner, which is why the y-axis needs to be reflected.
    - output_format: The format to save the plot as, either "png" (default) or "svg".
    """

    if ax is None:
        figsize = _get_figsize(width, height, trajectory_x, trajectory_y)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if width and height:
        # show the full fov if dims are given
        plt.xlim(0, width)
        plt.ylim(0, height)

    if hue is None:
        hue = trajectory_x.index

    # this connects each point with a line that is colored by time
    points = np.array([trajectory_x, trajectory_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = None
    if norm_hue:
        norm = plt.Normalize(0, len(hue))
    lc = LineCollection(segments, cmap=traj_cmap, norm=norm)
    lc.set_array(hue)
    lc.set_linewidth(2)
    trajectory = ax.add_collection(lc)

    # Plot each point as a scatter point
    sns.scatterplot(
        x=trajectory_x,
        y=trajectory_y,
        ax=ax,
        hue=hue,
        palette=traj_cmap,
        legend=False,
        size=1,
    )

    # Add labels
    plt.gca().set_aspect("equal")
    divider = make_axes_locatable(ax)
    cbar_cax = divider.append_axes("right", size="1.5%", pad=0.1)
    fig.colorbar(
        trajectory,
        ax=ax,
        label=hue_label,
        cax=cbar_cax,
        orientation="vertical",
    )
    # By default, the origin of matplotlib plots is the top-left corner.
    # However, most behavior movies have an origin in the bottom-left corner,
    # which is why the y-axis needs to be reflected.
    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if preview_filename:
        plt.savefig(
            preview_filename,
            format=output_format,
            transparent=(output_format == "svg"),
        )

    return ax


# def plot_trajectory_with_zones(
#     *,
#     zones: Union[pd.DataFrame, List[tracking.Zone]],
#     preview_filename: Optional[str] = None,
#     trajectory_x: np.array,
#     trajectory_y: np.array,
#     traj_cmap: str = "flare",
#     title: str = "Trajectory with Zones",
#     x_label: str = "X (pixels)",
#     y_label: str = "Y (pixels)",
#     hue: Optional[Union[np.array, pd.Series]] = None,
#     hue_label: str = "Frame Number",
#     norm_hue: bool = True,
#     width: Optional[int] = None,
#     height: Optional[int] = None,
#     invert_yaxis: bool = True,
#     zone_cmap: str = "crest",
#     output_format: str = "png",
# ):
#     """
#     Plot the trajectory with zones.

#     Parameters:
#     - zones: A pandas DataFrame, or list of zone objects, containing zone information.
#     - preview_filename: Optional filename to save the plot as.
#     - trajectory_x: An array of x-coordinates for the trajectory.
#     - trajectory_y: An array of y-coordinates for the trajectory.
#     - traj_cmap: The colormap for the trajectory plot (default: "flare").
#     - title: The title of the plot (default: "Trajectory with Zones").
#     - x_label: The label for the x-axis (default: "X (pixels)").
#     - y_label: The label for the y-axis (default: "Y (pixels)").
#     - hue: Optional array of values to use for coloring the trajectory.
#         If empty, then default to the trajectory index, i.e., frame number or time
#     - hue_label: The label for the colorbar (default: "Frame Number")
#     - norm_hue: If true, the colorbar is normalized.
#         Helpful if the colorbar is heavily skewed on one end.
#     - width: Optional width to set for the plot.
#         Used to render the full FOV of the behavior movie.
#     - height: Optional height to set for the plot.
#         Used to render the full FOV of the behavior movie.
#     - invert_yaxis: If true, the y-axis is reflected. By default, the origin of matplotlib
#         plots is the top-left corner. However, most behavior movies have an origin in the
#         bottom-left corner, which is why the y-axis needs to be reflected.
#     - zone_cmap: The colormap for the zone plot (default: "crest").
#     - output_format: The format to save the plot as, either "png" (default) or "svg".
#     """

#     if not width or not height:
#         # Approximate dimensions of the original FOV
#         # find the max x and y values of the zones
#         x_upper, y_upper = _find_zone_max(zones)

#         # find the max x and y values of the zones and trajectory
#         x_upper = max(x_upper, trajectory_x.max())
#         y_upper = max(y_upper, trajectory_y.max())

#         width = x_upper * 1.1
#         height = y_upper * 1.1

#     # Create zones object
#     if isinstance(zones, pd.DataFrame):
#         z = tracking.read_zones_from_dict(zones)
#     else:
#         z = zones

#     # Plot the trajectory
#     ax = plot_trajectory(
#         trajectory_x=trajectory_x,
#         trajectory_y=trajectory_y,
#         traj_cmap=traj_cmap,
#         title=title,
#         x_label=x_label,
#         y_label=y_label,
#         hue=hue,
#         hue_label=hue_label,
#         norm_hue=norm_hue,
#         width=width,
#         height=height,
#         invert_yaxis=invert_yaxis,
#     )
#     # Draw the zones on the plot
#     ax = tracking.plot_zones_on_ax(
#         z,
#         ax=ax,
#         cmap=zone_cmap,
#         title=title,
#         x_label=x_label,
#         y_label=y_label,
#     )

#     if preview_filename:
#         plt.savefig(
#             preview_filename,
#             format=output_format,
#             transparent=(output_format == "svg"),
#         )

#     return ax


@beartype
def plot_correlation_matrix(
    ax: matplotlib.axes._axes.Axes,
    corr_matrix: NumpyFloat2DArray,
    cmap="bwr",
    output_format="png",
) -> None:
    """Plot the correlation matrix of raw traces.
    Neurons are reordered so that weight in the
    correlation matrix is concentrated along the diagonal.

    :param ax: Matplotlib axis to plot on.
    :param corr_matrix: 2D numpy array representing the correlation matrix.
    :param cmap: Colormap to use for the plot.
    :param output_format: Plot format, either "png" (default) or "svg".
    """
    plt.sca(ax)

    if output_format == "svg":
        # Use pcolormesh for proper alignment in vector output
        flipped_corr_matrix = corr_matrix[::-1, :]
        num_cells = flipped_corr_matrix.shape[0]
        x_edges = np.arange(num_cells)
        y_edges = np.arange(num_cells)

        ax.pcolormesh(
            x_edges,
            y_edges,
            flipped_corr_matrix,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            shading="nearest",
        )
    else:
        # Default to matshow for raster-based output
        ax.matshow(corr_matrix, vmin=-1, vmax=1, cmap=cmap)


@beartype
def plot_footprints(
    footprints_x: List,
    footprints_y: List,
    figure,  # bokeh or matplotlib figure
    *,
    colors: Optional[List] = None,
    edge_colors: Optional[List] = None,
    fill_alpha: float = 0.5,
    line_alpha: float = 0.5,
    legend_label: str = "footprints",
    show_legend: bool = False,
    colormap: str = "jet",
) -> None:
    """draw footprints from cellset file

    ### Arguments

    - footprints_x: List of List of footprints (x co-ordinates)
    - footprints_y: List of List of footprints (y co-ordinates)
    - figure: where to plot? bokeh figure or matplotlib ax

    This function draws footprints using some contours.
    This does not directly compute footprints, use
    cell_set_to_convex_hulls or cell_set_to_contours
    for that. Those functions will also make clear
    the format of footprints_x & footprints_y
    that this function expects

    """

    N = len(footprints_x)

    if "bokeh" in str(type(figure)):
        palette = bokeh.palettes.Turbo256
        ranks = np.linspace(0, 255, N).astype(int)

        if colors is None:
            colors = [palette[r] for r in ranks]

        if show_legend:
            figure.multi_polygons(
                xs=footprints_x,
                ys=footprints_y,
                line_alpha=line_alpha,
                fill_alpha=fill_alpha,
                color=colors,
                hover_alpha=1,
                legend_label=legend_label,
            )
        else:
            figure.multi_polygons(
                xs=footprints_x,
                ys=footprints_y,
                line_alpha=line_alpha,
                fill_alpha=fill_alpha,
                color=colors,
                hover_alpha=1,
            )

    elif "matplotlib" in str(type(figure)):
        # I can't think of any other way to reliably check if
        # this is a matplotlib object

        if colors is None:
            # generate a randomized colormap
            cmap = matplotlib.colormaps[colormap]
            ranks = np.arange(0, N)
            colors = [cmap(r / N) for r in ranks]

        if edge_colors is None:
            edge_colors = colors
            linewidth = 3
        else:
            linewidth = 1
        plt.sca(figure)

        for i in range(N):
            plt.fill(
                footprints_x[i][0][0],
                footprints_y[i][0][0],
                fill=True,
                facecolor=colors[i],
                alpha=0.5,
                edgecolor=edge_colors[i],
                linewidth=linewidth,
            )

    else:
        raise ValueError(
            "Unknown canvas type. Should be matplotlib axis/bokeh figure"
        )


UnitsType = Literal["microns", "um", "centimeters", "cm"]
Locations = Literal["lower left", "lower right", "upper left", "upper right"]


@beartype
def add_scalebar(
    axis: matplotlib.axes.Axes,
    microns_per_pixel: float,
    scalebar_size: float,
    units: UnitsType = "microns",
    location: Locations = "lower left",
    fontsize: int = 12,
    scalebar_color: str = "white",
    text_color: str = "white",
    x_pixels_from_edge: Optional[int] = None,
    y_pixels_from_edge: Optional[int] = None,
    y_buffer: Optional[int] = None,
    scalebar_thickness: float = 3.0,
):
    """
    Adds a scalebar to an image
    Args:
        axis (matplotlib axis): Axis on which to add scalebar
        microns_per_pixel (float): number of microns per pixel
        scalebar_size (float): length of scalebar in desired units
        units (str, optional): Units of scalebar. Can be "microns", "um", "centimeters" or "cm". Defaults to "microns".
        location (str, optional): Where to location scalebar, Can be "lower left", "lower right", "upper left", "upper right": . Defaults to "lower left".
        fontsize (int, optional): fontsize for text. Defaults to 12.
        scalebar_color (str, optional): color of scalebar. Defaults to "white".
        text_color (str, optional): color of text. Defaults to "white".
        x_pixels_from_edge (float, optional): How many pixels to place from edge in x-dimension. Will attempt to auto-determine if None is passed. Defaults to None.
        y_pixels_from_edge (float, optional): How many pixels to place from edge in y-dimesion. Will attempt to auto-determine if None is passed. Defaults to None.
        y_buffer (int, optional): Number of pixels between text and scalebar. Auto-calculated if None. Defaults to None.
        scalebar_thickness (int, optional): Scalebar thickness. Defaults to 3.
    """
    # get units
    if units == "microns" or units == "um":
        units_text = "$\mu m$"
    elif units == "centimeters" or units == "cm":
        units_text = "$cm$"

    # get location
    # Note that each location makes some assumptions.
    # These can be overrode with x_pixels_from_edge and y_pixels_from_edge
    if location == "lower left":
        x_pixels_from_edge = (
            10 if x_pixels_from_edge is None else x_pixels_from_edge
        )
        y_pixels_from_edge = (
            30 if y_pixels_from_edge is None else y_pixels_from_edge
        )
        x_pos = axis.get_xlim()[0] + x_pixels_from_edge
        y_pos = axis.get_ylim()[0] - y_pixels_from_edge
    elif location == "lower right":
        x_pixels_from_edge = (
            10 if x_pixels_from_edge is None else x_pixels_from_edge
        )
        y_pixels_from_edge = (
            30 if y_pixels_from_edge is None else y_pixels_from_edge
        )
        x_pos = (
            axis.get_xlim()[1]
            - x_pixels_from_edge
            - scalebar_size / microns_per_pixel
        )
        y_pos = axis.get_ylim()[0] - y_pixels_from_edge
    elif location == "upper left":
        x_pixels_from_edge = (
            10 if x_pixels_from_edge is None else x_pixels_from_edge
        )
        y_pixels_from_edge = (
            10 if y_pixels_from_edge is None else y_pixels_from_edge
        )
        x_pos = axis.get_xlim()[0] + x_pixels_from_edge
        y_pos = axis.get_ylim()[1] + y_pixels_from_edge
    elif location == "upper right":
        x_pixels_from_edge = (
            10 if x_pixels_from_edge is None else x_pixels_from_edge
        )
        y_pixels_from_edge = (
            10 if y_pixels_from_edge is None else y_pixels_from_edge
        )
        x_pos = (
            axis.get_xlim()[1]
            - x_pixels_from_edge
            - scalebar_size / microns_per_pixel
        )
        y_pos = axis.get_ylim()[1] + y_pixels_from_edge

    if y_buffer is None:
        y_buffer = -1 * (axis.get_ylim()[0] - axis.get_ylim()[1]) / 300

    # add the scalebar using figrid
    scalebar(
        axis=axis,
        x_pos=x_pos,
        y_pos=y_pos,
        x_length=scalebar_size / microns_per_pixel,
        x_text="{} {}".format(scalebar_size, units_text),
        scalebar_color=scalebar_color,
        text_color=text_color,
        fontsize=fontsize,
        y_buffer=y_buffer,
        linewidth=scalebar_thickness,
    )


# # Preview functions brought over from ideas-python-utils
# @beartype
# def save_zones_preview(
#     *,
#     zones_file: Union[str, List[dict], pd.DataFrame],
#     output_preview_filename: str = "zones_preview.png",
#     image_source: Optional[Union[str, np.ndarray]] = None,
# ):
#     """Create and save a preview of the zones in a zones file. Optionally plot the zones on top of an image.

#     Args:
#         zones_file (str): Path to the zones file if it is a csv, a dataframe, or a list of dicts if its an roi input.
#         output_preview_filename (str): Path to save the preview image.
#         image_source (str or np.array): Path to a movie or image file or an image array to plot the zones on top of.
#     """

#     if isinstance(zones_file, list):
#         # Check for key unique to frontend ROIS
#         if "groupKey" in zones_file[0].keys():
#             z = tracking.get_zones_from_json(zones_file)
#         # Check for key unique to self zone style
#         elif "Name" in zones_file[0].keys():
#             z = tracking.read_zones_from_dict(zones_file)
#         else:
#             raise ValueError("Unknown format of zones provided.")
#     # Check if csv has already been loaded
#     elif isinstance(zones_file, pd.DataFrame):
#         z = tracking.read_zones_from_dict(zones_file.to_dict(orient="records"))
#     # Check if csv file path has been provided
#     elif isinstance(zones_file, str):
#         z = tracking.read_zones_from_csv(zones_file)

#     if image_source is not None:
#         if isinstance(image_source, str):
#             if image_source.endswith(".avi") or image_source.endswith(".mp4"):
#                 cap = cv2.VideoCapture(image_source)
#                 if cap.isOpened():
#                     ret, image = cap.read()
#                     cap.release()
#                 else:
#                     raise ValueError("Could not open video file.")
#             elif (
#                 image_source.endswith(".png")
#                 or image_source.endswith(".jpg")
#                 or image_source.endswith(".jpeg")
#                 or image_source.endswith(".tif")
#             ):
#                 image = plt.imread(image_source)
#         elif isinstance(image_source, np.ndarray):
#             image = image_source

#         else:
#             raise ValueError(
#                 "Invalid image source. Must be a path to a movie or image file or an image array."
#             )

#         tracking.plot_zones_on_im(z, image)
#         plt.imsave(output_preview_filename, image)
#     else:
#         fig, ax = plt.subplots()
#         tracking.plot_zones_on_ax(z, ax)
#         # Invert y-axis to match orientation in IDPS (y=0 at top of image and y=max at bottom)
#         ax.invert_yaxis()
#         fig.savefig(output_preview_filename)

#     # Plot the zones
#     fig, ax = plt.subplots()
#     tracking.plot_zones_on_ax(z, ax)

#     # Plot the image if provided
#     if image_source is not None:
#         if isinstance(image_source, str):
#             image = plt.imread(image_source)
#         else:
#             image = image_source

#         plt.imshow(image, cmap="gray", alpha=0.5)

#     plt.savefig(output_preview_filename)
#     plt.close()


def _format_yticks(tick_val, pos):
    return f"{tick_val:.1f}"


def _cellset_to_traces_skew_order(cell_set_file: str):
    """Small helper function to compute traces and skew_order
    from a cellset
    """
    traces = io.cell_set_to_traces(cell_set_file)
    status = io.cell_set_to_status(cell_set_file)

    skew = measures.matrix_skew(traces)
    skew_order = np.argsort(skew)

    # only show accepted/undecided neurons
    num_neurons = len(status)
    accepted_neurons = np.arange(0, num_neurons)
    accepted_neurons = accepted_neurons[status != "rejected"]

    skew_order_accepted = np.argsort(skew[status != "rejected"])
    accepted_neurons = accepted_neurons[skew_order_accepted]
    num_ok_neurons = len(accepted_neurons)

    show_these_neurons = np.unique(
        np.linspace(0, num_ok_neurons - 1, 20, dtype=int)
    )
    show_these_neurons = accepted_neurons[show_these_neurons]

    return traces, skew_order, show_these_neurons


def save_neural_traces_preview(
    *,
    cell_set_file: str,
    output_preview_filename: str,
) -> None:
    """Save a preview of the input neural traces to a PNG file

    all traces are plotted. rejected/undecided cells are plotted
    in gray. accepted cells are plotted in colors.
    """
    logger.info("Started generating cellset preview...")

    traces, skew_order, show_these_neurons = _cellset_to_traces_skew_order(
        cell_set_file
    )

    time = io.cell_set_to_time(cell_set_file)
    status = io.cell_set_to_status(cell_set_file)

    logger.info(f"Cellset has {len(status)} cells")

    traces = scipy.stats.zscore(traces, axis=0, nan_policy="omit")

    num_neurons = traces.shape[1]
    inches_per_neuron = 0.5

    num_neurons_to_show = len(show_these_neurons)

    # if the number of neurons is too small, then the title gets
    # clipped. a way around this is to account for the space
    # that the title needs.
    if num_neurons_to_show > 10:
        fig_height = num_neurons_to_show * inches_per_neuron
    else:
        fig_height = 10 * inches_per_neuron

    colors = cm.rainbow(np.linspace(0, 1, num_neurons))
    fig, ax = plt.subplots(figsize=(7, fig_height))

    y_offset = 7
    y = 0

    yticklabels = []
    for i, color in zip(skew_order, colors):
        if i not in show_these_neurons:
            continue
        plt.plot(time, y + traces[:, i], lw=0.5, color=color)

        yticklabels.append(i)
        y += y_offset

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cell ID")
    ax.set_xlim([0, time[-1]])
    ax.set_ylim((-y_offset, (num_neurons_to_show + 1) * y_offset))
    ax.set_yticks(np.arange(num_neurons_to_show) * y_offset)
    ax.set_yticklabels(yticklabels)
    ax.set_title(
        f"{num_neurons_to_show} cell traces ordered by SNR.",
        fontsize=12,
        color="gray",
    )
    ax.tick_params(axis="y", colors="gray")
    ax.tick_params(axis="x", colors="gray")
    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_color("gray")
    plt.tight_layout()

    logger.info("Saving traces figure...")
    fig.savefig(output_preview_filename, dpi=300)

    logger.info(
        f"Traces preview saved ({os.path.basename(output_preview_filename)}, size: {get_file_size(output_preview_filename)})"
    )


def save_footprints_preview(
    *,
    cell_set_file: str,
    output_preview_filename: str,
) -> None:
    """Save a PNG image showing footprints of cells

    rejected/undecided cells shown in gray

    (unless all cells are undecided, in which case they
    are shown in color)
    """
    logger.info("Started making cellset footprints preview...")

    _, skew_order, show_these_neurons = _cellset_to_traces_skew_order(
        cell_set_file
    )

    x, y = io.cell_set_to_contours(cell_set_file, threshold=2.0)
    status = io.cell_set_to_status(cell_set_file)

    fig, ax = plt.subplots(figsize=(12, 8))

    num_neurons = len(x)

    colors = cm.rainbow(np.linspace(0, 1, num_neurons))

    # i literally know of no easier way of doing this
    # to avoid going mad over this, here is this ugly
    # code that is correct and works
    use_colors = ["#d1d1d1" for _ in range(num_neurons)]
    for i, color in zip(skew_order, colors):
        if i in show_these_neurons:
            use_colors[i] = color

    # fully purge rejected neurons
    show_x = [
        xx for xx, this_status in zip(x, status) if this_status != "rejected"
    ]
    show_y = [
        xx for xx, this_status in zip(y, status) if this_status != "rejected"
    ]
    show_colors = [
        color
        for color, this_status in zip(use_colors, status)
        if this_status != "rejected"
    ]

    plot_footprints(show_x, show_y, ax, colors=show_colors, fill_alpha=0.25)

    # plot text labels showing IDs of cells
    # only for cells that we show traces for

    i = -1
    for xx, yy in zip(x, y):
        i += 1
        if i not in show_these_neurons:
            continue
        try:
            cx = xx[0][0].mean()
            cy = yy[0][0].mean()

            plt.text(cx, cy, i, horizontalalignment="center")
        except Exception:
            # this can fail if cellsets have all-zero
            # footprints, which is a known bug in IDPS
            # the best thing to do is simply ignore this
            # because there is nothing we can do here

            logger.info(f"Cell {i} has undefined location.")

    # figure out the dimensions of the frame
    # we want the figure to maintain frame dims
    cell_set = isx.CellSet.read(cell_set_file)
    xy_max = cell_set.get_cell_image_data(0).shape
    ax.set_ylim(0, xy_max[0])
    ax.set_xlim(0, xy_max[1])
    ax.set_aspect("equal", "box")
    ax.set_title(
        "Footprints of identified cells.",
        fontsize=12,
        color="gray",
    )
    for spine in ["bottom", "top", "left", "right"]:
        ax.spines[spine].set_color("gray")
    ax.tick_params(axis="y", colors="gray")
    ax.tick_params(axis="x", colors="gray")
    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    # invert y-axis to match orientation in IDPS (y=0 at top of image and y=max at bottom)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(
        output_preview_filename,
        dpi=300,
    )

    logger.info(
        f"Footprints preview saved ({os.path.basename(output_preview_filename)}, size: {get_file_size(output_preview_filename)})"
    )


def save_experiment_annotations_preview(
    df: pd.DataFrame, output_preview_filename: str, top_n_states: int = 10
) -> None:
    """Save a preview of an experiment annotations dataframe to a PNG file.
       By default the top ten most common states are shown on a bar chart.

    :param df: input dataframe from which to extract a preview
    :param output_preview_filename: path to the output json file
    :param num_states: top most common states to include in the bar chart
    """

    common_states = df["state"].value_counts().nlargest(top_n_states)
    states = common_states.index
    counts = common_states.values

    fig, ax = plt.subplots()

    states = [state[:30] for state in states]
    ax.bar(states, counts)
    ax.set_title("Most Common States")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], label_type="edge")
    ax.margins(y=0.1)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()

    fig.savefig(
        output_preview_filename,
        dpi=300,
    )


def arrange_axes_to_ordinal_values(d):
    """Rearrange the grid spec layout from matplotlib.subplot_mosaic into ordinal values. For
    example, If there are 2 subplots created using matplotlib.subplot_mosaic(), individual axis
    will be referenced as ax['a'] and ax['b']. This function converts these indices such that
    they can be referenced as ax[0] and ax[1].
    Parameters :
        d: dictionary of axis and corresponding
    """
    temp = np.array(list(d.items()))
    idx = np.argsort(np.asarray(list(d.keys())))

    return temp[idx, -1]


def plot_shaded_hist(
    values, ax, title, xlabel, palette="coolwarm", hist_lims=None, norm=None
):
    """
    Plots a shaded histogram with a customizable diverging color palette.

    Args:
    values (array-like): The data values to be plotted in the histogram.
    ax (matplotlib.axes.Axes): The axes object to plot the histogram on.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    palette (str or list, optional): The color palette to use for shading the histogram bars.
                                     Default is "coolwarm".
    hist_lims (tuple, optional): The limits for the x-axis of the histogram. Default is None.
    norm (matplotlib.colors.Normalize, optional): The normalization for the color mapping. Default is None.

    Returns:
    cmap (matplotlib.colors.Colormap): The colormap used for shading the histogram bars.
    norm (matplotlib.colors.Normalize): The normalization used for the color mapping.
    """
    if isinstance(palette, str):
        # Interpret the palette as a colormap
        try:
            cmap = sns.color_palette(palette, as_cmap=True)
        except ValueError:
            IdeasError(
                "Invalid palette. Please provide a valid colormap or a list of colors"
            )
    elif isinstance(palette, list):
        cmap = LinearSegmentedColormap.from_list(name="test", colors=palette)

    # Set the number of bins to 30 or the number of differences, whichever is smaller
    bin_num = min(30, len(values))

    # Plot the histogram of trace changes vertically
    _, bins, patches = ax.hist(values, bins=bin_num)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_max = np.max(np.abs(bins))

    # If no normalization is provided, normalize the colormap to the maximum bin value
    if norm is None:
        norm = plt.Normalize(-bin_max, bin_max)

    # Overwrite the face color of each patch with the colormap
    for c, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", cmap(norm(c)))

    # Set the x-axis limits
    if hist_lims is not None:
        ax.set_xlim(hist_lims)
    else:
        ax.set_xlim([-bin_max, bin_max])

    ax.axvline(0, color="black", label="Zero")
    # add lines median
    ax.axvline(np.median(values), color="black", linestyle=":", label="Median")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add just the vlines as a legend
    ax.legend(
        loc="upper right",
        handles=[
            plt.Line2D([0], [0], color="black", lw=2, label="Zero"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                lw=2,
                linestyle=":",
                label="Median",
            ),
        ],
        frameon=False,
    )
    return cmap, norm


def plot_paired_lines(
    *,
    ax: plt.Axes,
    df: pd.DataFrame,
    category_names: List[str],
    num_values: int,
    category_column: str = "Epoch",
    value_column: str = "Activity",
    line_color: str = "grey",
    line_alpha: float = 0.07,
):
    """Plots lines connecting the values in a dataframe between categories.


    - ax (plt.Axes): Axes object to plot on.
    - df (pd.DataFrame): DataFrame containing the data.
    - category_names (List[str]): List of category names to plot.
    - num_values (int): Number of values to plot.
    - category_column (str, optional): Column name for categories. Default is "Epoch".
    - value_column (str, optional): Column name for values. Default is "Activity".
    - line_color (str, optional): Color of the lines. Default is "grey".
    - line_alpha (float, optional): Alpha value for line transparency. Default is 0.07.
    """
    # Iterate over the categories
    for i in range(len(category_names) - 1):
        # get the data for each category
        full_category_1 = df[df[category_column] == category_names[i]][
            value_column
        ]
        full_category_2 = df[df[category_column] == category_names[i + 1]][
            value_column
        ]

        # Plot a line for each value
        for j in range(num_values):
            ax.plot(
                [i, i + 1],
                [full_category_1.values[j], full_category_2.values[j]],
                color=line_color,
                alpha=line_alpha,
            )


class EventSetPreview(object):
    """Class for generating eventset preview.
    This class is responsible for generating a preview from an eventset file.
    """

    def __init__(
        self,
        input_eventset_filepath: str,
        output_png_filepath: str,
        markercolor: str = "k",
        markersize: float = 1.5,
        figsize: tuple = (14, 12),
        axis_label_fontsize: float = 15,
        background_color: str = "dark_background",
        foreground_color: str = "white",
        event_rate_num_bins: int = 30,
        inter_event_interval_bins: int = 30,
        gaussian_filter_smoothing_sigma_time: float = 0.5,
        lineplot_linewidth: float = 1.5,
    ):
        """Initialize features that will be derived from event set and eventually plotted
        Parameters:
            input_eventset_filepath: Filepath of the eventset file
            output_png_filepath: Filepath of the preview png file
            markercolor: color of the matplotlib marker
            markersize: size of matpotlib marker
            figsize: size of the matplotlib figure
            axis_label_fontsize: matplotlib axis label font size
            background_color: Background color of matplotlib subplots
            foreground_color: Foreground color of symbols and texts within the matplotlib subplots
            event_rate_num_bins: Number of bins for the mean event rate histogram
            inter_event_interval_bins: Number of bins for the mean inter-event interval histogram
            gaussian_filter_smoothing_sigma_time: Sigma of gaussian filter (in seconds) used for
            smoothing the average event rate
            lineplot_linewidth: Linewidth
        """
        self.filepath = input_eventset_filepath
        self.output_png_filepath = output_png_filepath

        # Raise an exception if the events file does not exist
        if not os.path.exists(self.filepath):
            raise IdeasError(
                "Input file {0} does not exist".format(
                    os.path.basename(self.filepath)
                )
            )

        # Raise an exception the input file is not an isxd eventset
        try:
            self.eventset = isx.EventSet.read(self.filepath)
        except Exception as err:
            raise IdeasError("Only isxd eventsets are supported") from err

        self.num_samples = self.eventset.timing.num_samples
        self.sampling_rate = np.around(
            1 / self.eventset.timing.period.secs_float, 2
        )
        self.markercolor = markercolor
        self.markersize = markersize
        self.axis_label_fontsize = axis_label_fontsize
        self.figsize = figsize
        self.background_color = background_color
        self.foreground_color = foreground_color
        self.event_rate_num_bins = event_rate_num_bins
        self.inter_event_interval_bins = inter_event_interval_bins
        self.gaussian_filter_smoothing_sigma_time = (
            gaussian_filter_smoothing_sigma_time
        )
        self.lineplot_linewidth = lineplot_linewidth

    def _events_to_raster(self, event_times: np.array, timing) -> np.array:
        """Return raster from event timings
        Paramaters:
            event_times: times of events
            timing: isx.timing object
        Returns:
            raster: A single of rasters
        """
        samp_vect = [i for i in range(timing.num_samples)]
        event_times_stamps = event_times / timing.period.to_usecs()
        raster = np.zeros_like(samp_vect)

        if len(event_times) == 0:
            return raster

        else:
            raster[event_times_stamps.astype(int)] = 1
            return raster

    def compute_rasters(self) -> np.array:
        """Extract events from an eventset file and returns rasters
        Parameters:
            filepath: Filepath of the event set file
        Returns:
            rasters: numpy array of rasters derived from events
        """
        # Initializing the rasters
        rasters = np.empty([self.eventset.num_cells, self.num_samples])

        # Populating the rasters
        for cellnum in range(self.eventset.num_cells):
            offs, _ = self.eventset.get_cell_data(cellnum)
            raster = self._events_to_raster(offs, self.eventset.timing)
            rasters[cellnum] = raster

        return rasters

    def compute_mean_inter_event_interval(self, rasters: np.array) -> np.array:
        """Compute mean inter-event interval for each neuron
        Parameters:
            rasters: numpy array of rasters derived from events
        Returns:
            mean_inter_event_interval: Mean interval-event interval of each neuron
        """
        mean_inter_event_interval = np.empty(rasters.shape[0])
        for i in range(rasters.shape[0]):
            # Location of where the raster values = 1, this is equivalent to presence of an event
            event_location = np.where(rasters[i, :])[0]
            if event_location.size > 1:
                mean_inter_event_interval[i] = np.mean(
                    np.diff(event_location) * (1 / self.sampling_rate)
                )
            else:
                mean_inter_event_interval[i] = np.nan

        return mean_inter_event_interval[~np.isnan(mean_inter_event_interval)]

    def generate_preview(self):
        """Plot and save event set preview. Preview generates 4 subplots
        1. Raster
        2. A time series of mean event rate across neurons
        3. Histogram of mean event rate of neurons across the entire recording
        4. Histogram of mean inter-event interval
        """
        # Designing the figure layout
        fig, ax = plt.subplot_mosaic(
            """
                                    aaaa
                                    aaaa
                                    aaaa
                                    bbbb
                                    bbbb
                                    ccdd
                                    ccdd
                                    """,
            figsize=self.figsize,
        )
        fig.tight_layout(pad=4)
        ax = arrange_axes_to_ordinal_values(ax)
        t = np.linspace(
            0, (self.num_samples - 1) / self.sampling_rate, self.num_samples
        )

        # Check whether there is atleast 1 event in file
        rasters = self.compute_rasters()

        # Populate the axis if there is at least 2 event in the file
        with plt.style.context(self.background_color):
            if np.sum(rasters.ravel()) > 0:
                # Raster plot
                # show as heatmap image instead of event plot in order to reduce preview file size
                num_cells = rasters.shape[0]
                rasters_img = np.zeros((num_cells, self.num_samples))
                for idx in range(num_cells):
                    offsets = np.where(rasters[idx, :] == 1)[0]
                    rasters_img[idx, offsets] = 1
                ax[0].imshow(
                    rasters_img,
                    aspect="auto",
                    interpolation="none",
                    origin="lower",
                    cmap="gray",
                    extent=(t[0], t[-1], -0.5, num_cells - 0.5),
                )
                ax[0].set_xlabel("Time (s)", fontsize=self.axis_label_fontsize)
                ax[0].set_ylabel("Cell #", fontsize=self.axis_label_fontsize)
                ax[0].set_yticks(np.linspace(0, num_cells - 1, 5).astype(int))

                # Displaying the average event rate across neurons
                mean_event_rate_across_neurons = (
                    np.nanmean(rasters, axis=0) * self.sampling_rate
                )
                mean_event_rate_across_neurons = ndimage.gaussian_filter1d(
                    input=mean_event_rate_across_neurons,
                    sigma=(
                        self.gaussian_filter_smoothing_sigma_time
                        * self.sampling_rate
                    ).astype(int),
                )
                ax[1].plot(
                    t,
                    mean_event_rate_across_neurons,
                    color=self.foreground_color,
                    linewidth=self.lineplot_linewidth,
                )
                ax[1].set_xlabel("Time (s)", fontsize=self.axis_label_fontsize)
                ax[1].set_ylabel(
                    "Mean event rate (Hz)", fontsize=self.axis_label_fontsize
                )
                ax[1].set_ylim((0, None))
                ax[1].set_xlim((t[0] - t[1], int(t[-1]) + 1))

                # Computing the mean event rate across time
                mean_ER = np.mean(rasters, axis=1) * self.sampling_rate
                ax[2].hist(
                    mean_ER,
                    bins=np.linspace(
                        0, mean_ER.max(), self.event_rate_num_bins
                    ),
                    facecolor="white",
                    edgecolor="k",
                )
                ax[2].set_xlabel(
                    "Mean event rate across time (Hz)",
                    fontsize=self.axis_label_fontsize,
                )
                ax[2].set_ylabel(
                    "Cell count", fontsize=self.axis_label_fontsize
                )
                ax[2].set_xlim(0, mean_ER.max())

                # Computing the mean inter-event interval
                mean_interval_event_interval = (
                    self.compute_mean_inter_event_interval(rasters)
                )
                if mean_interval_event_interval.size > 0:
                    # Making sure that there is atleast 1 entry so that the plot can be generated
                    ax[3].hist(
                        mean_interval_event_interval,
                        bins=np.linspace(
                            0,
                            mean_interval_event_interval.max(),
                            self.inter_event_interval_bins,
                        ),
                        facecolor="white",
                        edgecolor="k",
                    )
                    ax[3].set_xlim(0, mean_interval_event_interval.max())
                else:
                    ax[3].text(
                        0.1,
                        0.5,
                        "There are no cells that have more than 1 event",
                        fontsize=15,
                    )
                    ax[3].set_ylim((0, 1))
                    ax[3].set_xlim((0, 1))

                # In case there is no data point, at least the axes labels will be shown
                ax[3].set_xlabel(
                    "Mean inter-event interval (s)",
                    fontsize=self.axis_label_fontsize,
                )
                ax[3].set_ylabel(
                    "Cell count", fontsize=self.axis_label_fontsize
                )

                # Making sure that the aesthethics of all the subplots are the same
                for axis in ax:
                    axis.spines["top"].set_visible(False)
                    axis.spines["right"].set_visible(False)
                    axis.set_facecolor("k")
                    axis.xaxis.label.set_color(self.foreground_color)
                    axis.yaxis.label.set_color(self.foreground_color)
                    axis.tick_params(colors=self.foreground_color)
                    axis.spines["bottom"].set_color(self.foreground_color)
                    axis.spines["left"].set_color(self.foreground_color)

            # Saving the preview
            plt.savefig(self.output_png_filepath, dpi=300)


# Function from figrid (https://github.com/dougollerenshaw/figrid). Copied here to avoid dependency on figrid


def scalebar(
    axis,
    x_pos,
    y_pos,
    x_length=None,
    y_length=None,
    x_text=None,
    y_text=None,
    x_buffer=0.25,
    y_buffer=0.25,
    scalebar_color="black",
    text_color="black",
    fontsize=10,
    linewidth=3,
):
    """
    add a scalebar
    input params:
        axis: axis on which to add scalebar
        x_pos: x position, in pixels
        y_pos: y position, in pixels
    """
    if x_length is not None:
        axis.plot(
            [x_pos, x_pos + x_length],
            [y_pos, y_pos],
            color=scalebar_color,
            linewidth=linewidth,
        )
        axis.text(
            x_pos + x_length / 2,
            y_pos - y_buffer,
            x_text,
            color=text_color,
            fontsize=fontsize,
            ha="center",
            va="top",
        )

    if y_length is not None:
        axis.plot(
            [x_pos, x_pos],
            [y_pos, y_pos + y_length],
            color=scalebar_color,
            linewidth=linewidth,
        )

        axis.text(
            x_pos - x_buffer,
            y_pos + y_length / 2,
            y_text,
            color=text_color,
            fontsize=fontsize,
            ha="right",
            va="center",
        )
