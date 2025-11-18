import copy
import pathlib
import logging
import os
import shutil
import math
from collections import OrderedDict
from typing import List, Optional
import h5py
import isx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import utils.config as config
from scipy import stats
from skimage.measure import find_contours
# from toolbox.utils.data_model import IdeasFile, IdeasGroup, IdeasPreviewFile
from ideas.exceptions import IdeasError
from utils.footprint_utils import compute_cell_centroid
# from toolbox.utils.output_manifest import (
#     save_metadata_manifest,
#     save_output_manifest,
# )
from ideas.analysis.utils import (
    # compute_sampling_rate,
    get_file_size,
    # read_isxd_metadata,
    _sort_isxd_files_by_start_time,
)
from utils.utils import (
    validate_cellset_series_compatibility,
    get_num_cells_by_status,
)

from utils.metadata import (
    read_isxd_metadata
)
logger = logging.getLogger()

PLOT_PARAMS = {}


def nanmean_iterative(event_windows, cell_indices):
    """Compute mean of event windows across specified cells.
       Iterative and memory-efficient version of np.nanmean.

    Assumptions:
        - The cell indices array is not empty.
        - Data is to be averaged over axis=2.

    :param event_windows: array of event windows (num_windows, num_timepoints, num_cells)
    :param cell_indices: list of cell indices to average
    :param axis: axis over which to average
    """
    if len(cell_indices) == 0 or event_windows.shape[-1] == 0:
        raise IdeasError(
            "Cannot compute the mean of 0 elements",
        )

    # initialize average that will be updated as data gets processed
    avg = np.zeros(
        (event_windows.shape[0], event_windows.shape[1]),
        dtype="float64",
    )
    for n, idx in enumerate(cell_indices):
        # replace nan values with current average, effectively ignoring them
        data = event_windows[:, :, idx].copy()
        nan_indices = np.isnan(data)
        data[nan_indices] = avg[nan_indices]

        # update average
        avg *= n / (n + 1)
        avg += data / (n + 1)

    return avg.astype("float32")


def extract_event_windows(traces, event_indices, time_window):
    """Splice out traces around each event.
    - The length of the trace is defined by the time window.

    :param traces: input traces <1d array for single trace, or 2d array for multiple traces>
    :param event_indices: indices of the individual events <1d array>
    :param time_window: number of frames to consider before (pre) and after (post)
                        each event <tuple (pre,post)>

    :return event_windows: window of time around each event
            <array(num_events, num_timepoints)> when given a single trace
            <array(num_events, num_timepoints_per_window, num_cells)> for multiple traces
    """
    num_events = len(event_indices)
    pre, post = time_window
    num_timepoints_per_window = pre + post + 1

    if traces.ndim == 1:
        event_windows = np.zeros(
            (num_events, num_timepoints_per_window), dtype="float32"
        )
    elif traces.ndim == 2:
        num_cells = traces.shape[1]
        event_windows = np.zeros(
            (num_events, num_timepoints_per_window, num_cells), dtype="float32"
        )
    else:
        raise IdeasError(
            "Event windows can only be extracted from a one- or two-dimensional traces array.",
        )

    for i, event_index in enumerate(event_indices):
        window_start_index = event_index - pre
        window_end_index = event_index + post + 1
        event_windows[i] = traces[window_start_index:window_end_index]

    return event_windows


def extract_mean_event_windows_per_event_shuffle(
    traces, event_indices_shuffles, time_window
):
    """Given shuffled event times, splice out window around each event,
    and compute mean event window per shuffle.

    traces: input traces <1d array for single trace, or 2d array for multiple traces>
    :param event_indices_shuffles: shuffles of event indices <2d array (num_shuffles, num_events)>
    :param time_window: number of frames to consider before (pre) and after (post)
                        each event <tuple (pre,post)>

    :return event_windows: mean event window for each shuffle
                           - <array(num_shuffles, num_timepoints)> for single input trace
                           - <array(num_shuffles, num_timepoints_per_window, num_cells)> for
                              multiple input traces
    """
    pre, post = time_window
    num_timepoints_per_window = pre + post + 1
    num_shuffles = len(event_indices_shuffles)

    if traces.ndim == 1:
        event_windows = np.zeros(
            (num_shuffles, num_timepoints_per_window), dtype="float32"
        )
    elif traces.ndim == 2:
        num_cells = traces.shape[1]
        event_windows = np.zeros(
            (num_shuffles, num_timepoints_per_window, num_cells),
            dtype="float32",
        )
    else:
        raise IdeasError(
            "Event windows can only be extracted from a one- or two-dimensional traces array.",
        )

    for i in range(num_shuffles):
        # extract event windows for the current shuffle
        shuffle_event_windows = extract_event_windows(
            traces, event_indices_shuffles[i], time_window
        )

        # compute the mean event window of the current shuffle by averaging
        # all event windows across time
        event_windows[i] = np.nanmean(
            shuffle_event_windows, axis=0, dtype="float32"
        )
        del shuffle_event_windows

    return event_windows


def compute_post_minus_pre(
    event_windows, pre_event_indices, post_event_indices
):
    """Compute the average post- minus pre-event activity for the given event windows.
    - Given a window of time around a given event (i.e. an event window),
      compute the mean activity before (pre) and subtract that value from
      the mean activity after (post) the event.
      => post_minus_pre = mean(post) - mean(pre)

    :param event_windows: activity traces around each event
    :param pre_event_indices: indices to use for pre-event activity
    :param post_event_indices: indices to use for post-event activity

    :return: average post- minus pre-event activity for the given event windows
    """
    if event_windows.ndim == 1:
        post_minus_pre = np.nanmean(
            event_windows[post_event_indices], dtype="float32"
        ) - np.nanmean(event_windows[pre_event_indices], dtype="float32")
    elif event_windows.ndim == 2:
        post_minus_pre = np.nanmean(
            event_windows[post_event_indices], axis=0, dtype="float32"
        ) - np.nanmean(
            event_windows[pre_event_indices], axis=0, dtype="float32"
        )
    else:
        raise IdeasError(
            "Post minus pre event activity difference can only be computed "
            "from a one- or two-dimensional array of event windows.",
        )
    return post_minus_pre


def generate_post_minus_pre_shuffled_dist(
    event_windows, pre_event_indices, post_event_indices
):
    """Generate a post minus pre distribution from the event windows collected from shuffled
    event indices.
    - Each value in the output list corresponds to the average of the trace
    obtained by subtraction pre- from post-event activity in a given event window.

    :param event_windows: activity traces around each event
                          <array (num_events, num_timepoints_per_event)> for a single cell
                          <array (num_events, num_timepoints_per_event, num_cells)> for many cells
    :param pre_event_indices: indices to use for pre-event activity
    :param post_event_indices: indices to use for post-event activity

    :return: list of post minus pre value
              <array (num_shuffles)> for single cell
              <array (num_shuffles, num_cells)> for multiple cells
    """
    num_shuffles = event_windows.shape[0]

    if event_windows.ndim == 2:
        # (num_events, num_timepoints_per_event)
        null_dist = np.zeros(num_shuffles, dtype="float32")
    elif event_windows.ndim == 3:
        # (num_events, num_timepoints_per_event, num_cells)
        num_cells = event_windows.shape[2]
        null_dist = np.zeros((num_shuffles, num_cells), dtype="float32")
    else:
        raise IdeasError(
            "Post minus pre event activity difference can only be computed "
            "from a two- or three-dimensional array of event windows.",
        )

    for i in range(num_shuffles):
        null_dist[i] = compute_post_minus_pre(
            event_windows[i], pre_event_indices, post_event_indices
        )

    return null_dist


def draw_footprints(
    footprints_x: List,
    footprints_y: List,
    *,
    color,
    fill_alpha: float = 0.125,
    label: str = "footprints",
):
    """Draw footprints from cellset file

    ### Arguments

    - footprints_x: List of List of footprints (x co-ordinates)
    - footprints_y: List of List of footprints (y co-ordinates)


    This function draws footprints using contours.
    This does not directly compute footprints, use
    cell_set_to_contours
    for that. Those functions will also make clear
    the format of footprints_x & footprints_y
    that this function expects

    """
    N = len(footprints_x)

    for i in range(1, N):
        plt.fill(
            footprints_x[i][0][0],
            footprints_y[i][0][0],
            fill=True,
            facecolor=color,
            alpha=fill_alpha,
            edgecolor=color,
        )

    plt.fill(
        footprints_x[0][0][0],
        footprints_y[0][0][0],
        fill=True,
        facecolor=color,
        alpha=fill_alpha,
        edgecolor=color,
        label=label,
    )


def footprints_to_contours(
    footprints: np.array,
    *,
    threshold: float = 2.0,
) -> (List, List):
    """Generate list of contours from a cell set file

    ### Arguments:

    - footprints: footprints matrix
    - threshold: to consider footprints within cell

    threshold is a parameter that divides the maximum of
    each cell's footprint, and defines the threshold to
    draw a contour around


    ### Returns

    - contours_x: list of list of arrays of contours
    - contours_y: list of list of arrays of contours

    This convoluted format is so that this can be
    fed directly into bokeh.multi_polygons

    """
    # generate contours of all cells
    contours_x = []
    contours_y = []

    for i in range(footprints.shape[0]):
        footprint = footprints[i, :, :]

        if footprint.max() == 0:
            # we can't compute footprints here

            contours_x.append([[0]])
            contours_y.append([[0]])
            continue

        xy = find_contours(np.fliplr(footprint), footprint.max() / threshold)

        contours_x.append([[xy[0][:, 0]]])
        contours_y.append([[xy[0][:, 1]]])

    return (contours_x, contours_y)


def compute_statistical_metrics(
    true_dist: np.array,
    null_dist: np.array,
    significance_threshold: float,
):
    """Compute metrics for every cell by comparing their post minus pre value
    to their own null distribution.

    Metrics include the following:
        - z-score
        - p-value
        - modulation classification (positive modulation, negative modulation, or no modulation)
        - rank (from most up-modulated to most down-modulated)
        - mean of the null distribution (mu)
        - standard deviation of the null distribution (std)

    :param true_dist: true post-pre value for each cell <1d array>
    :param null_dist: null distribution of post minus pre values for each cell
                      <2d array: (num_shuffles, num_cells)>
    :param significance_threshold: significance threshold to use for statistical tests

    :return zscores: list of z-scores (one per cell)
    :return pvalues: list of p-values (one per cell)
    :return modulations: list of modulation signs
                        (-1 for negative, +1 for positive, 0 for no modulation)
    :return ranks: indices of the z-scores sorted in descending order
                   (from most up- to most down- modulated)
    """
    pvalues = np.zeros_like(true_dist)
    modulations = np.zeros_like(true_dist)

    # zscore
    null_dist_mus = np.nanmean(null_dist, axis=0, dtype="float32")
    null_dist_stds = np.nanstd(null_dist, axis=0, dtype="float32")
    zscores = (true_dist - null_dist_mus) / null_dist_stds

    for i in range(true_dist.shape[0]):
        # estimate p-values by comparing to null dist
        frac_above_null = np.mean(true_dist[i] > null_dist[:, i])
        frac_below_null = 1 - frac_above_null

        # dividing alpha by 2 for 2 single-tailed comparisons
        if frac_below_null < (significance_threshold / 2):
            modulations[i] = 1
        elif frac_above_null < (significance_threshold / 2):
            modulations[i] = -1

        pvalues[i] = np.min([frac_above_null, frac_below_null])

    # compute ranks and reverse order to get descending order
    # i.e. from most up-modulated to most down-modulated
    ranks = np.argsort(zscores)[::-1]

    return zscores, pvalues, modulations, ranks, null_dist_mus, null_dist_stds


def plot_population_mean_event_window(
    x_values,
    x_limits,
    population_mean,
    population_sem,
    shuffled_mean,
    shuffled_conf_interval_low,
    shuffled_conf_interval_high,
    significance_threshold,
    event_type,
    output_filename,
    color="#1f77b4",
    plot_limits=None,
    epoch_name=None,
):
    """Plot the population mean event window data.

    :param x_values: values along the x-axis
    :param x_limits: limits of the x-axis to use for plots
    :param population_mean: population mean event window
    :param population_sem: standard error of the mean of the population mean event window
    :param shuffled_mean: population mean event window across all shuffles
    :param shuffled_conf_interval_low: lower bound on population mean event window across shuffles
    :param shuffled_conf_interval_high: upper bound on population mean event window across shuffles
    :param significance_threshold: threshold used to define the statistical confidence interval
    :param event_type: type of event for which analysis was performed
    :param output_file_basename: path to the output image without a file extension
    :param color: color to use for the population activity line
    :param plot_limits: y-axis range (z-score) applied to the event-aligned population activity plot
    :param epoch_name: name of the epoch
    """
    fig, ax = plt.subplots()

    # plot event-aligned population data
    if not all(np.isnan(population_sem)):
        ax.plot(
            x_values,
            population_mean,
            label="mean activity ± sem (true data)",
            color=color,
        )
        ax.fill_between(
            x_values,
            population_mean + population_sem,
            population_mean - population_sem,
            alpha=config.PLOT_SHADING_ALPHA,
            color=color,
        )
    else:
        # When there is only one event, only mean is plotted because sem is non-sensical in this
        # case
        ax.plot(
            x_values,
            population_mean,
            label="mean activity (true data)",
            color=color,
        )

    # plot shuffled event-aligned data
    if any(shuffled_mean):
        ax.plot(
            x_values,
            shuffled_mean,
            color="k",
            label="mean activity (shuffled data)",
        )

    if any(shuffled_conf_interval_low):
        if significance_threshold is None:
            label = "confidence interval (shuffled data)"
        else:
            confidence_interval = (1 - significance_threshold) * 100
            label = "{0:.2f}% confidence interval (shuffled data)".format(
                confidence_interval
            )

        ax.plot(
            x_values,
            shuffled_conf_interval_low,
            color="k",
            ls="--",
            label=label,
        )

    if any(shuffled_conf_interval_high):
        ax.plot(x_values, shuffled_conf_interval_high, color="k", ls="--")

    # add vertical line at x=0
    # this line represents the reference event time around which we want to observe patterns
    ax.axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

    # add labels
    plot_title = "Event-Aligned Population Activity"

    if event_type:
        subtitle = f"\n(event type: {event_type}"
        if epoch_name:
            subtitle += f", epoch: {epoch_name})"
        else:
            subtitle += ")"
        plot_title += subtitle
    else:
        if epoch_name:
            plot_title += f"\n(epoch: {epoch_name})"

    ax.set_title(plot_title, fontsize=config.PLOT_TITLE_FONT_SIZE)

    ax.set_xlabel(
        "Time from Event (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )
    ax.set_ylabel(
        "Neural Activity (z-score)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )

    # adjust plot limits
    global PLOT_PARAMS
    if plot_limits not in [None, "auto"]:
        y_limits = [float(lim) for lim in plot_limits.split(",")]
        ax.set_ylim(y_limits)
    elif (
        "population_activity_plot_limits" in PLOT_PARAMS
        and PLOT_PARAMS["population_activity_plot_limits"] != "auto"
    ):
        y_limits = [
            float(lim)
            for lim in PLOT_PARAMS["population_activity_plot_limits"].split(
                ","
            )
        ]
        ax.set_ylim(y_limits)
    ax.set_xlim(x_limits)
    ax.margins(x=0)

    # add legend to plot
    ax.legend()

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_single_neurons_heatmap(
    single_cell_data,
    x_limits,
    event_type,
    output_filename,
    color_limits=None,
    cmap=None,
    epoch_name=None,
):
    """Plot heatmap showing event-aligned activity modulation per neuron.

    :param single_cell_data: single cell data to plot as a heatmap
           Formatted as (num_cells, num_timepoints).
    :param x_limits: limits of the x-axis to use for plots
    :param event_type: type of event for which analysis was performed
    :param output_filename: path to the output image data
    :param color_limits: colormap range (z-score) applied to the activity heatmap
    :param cmap: colormap applied to the activity heatmap
    :param epoch_name: name of the epoch
    """
    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )
    fig, ax = plt.subplots()

    num_cells, num_timepoints = single_cell_data.shape

    # Compute x edges to match `imshow()`
    time_edges = np.linspace(x_limits[0], x_limits[1], num_timepoints + 1)
    cells_edges = np.arange(num_cells + 1)
    # Flip data vertically to match `imshow()`
    single_cell_data_flipped = np.flipud(single_cell_data)

    # Use pcolormesh with corrected edges
    axis_image = ax.pcolormesh(
        time_edges,
        cells_edges,
        single_cell_data_flipped,
        cmap=cmap,
        shading="flat",
    )

    # adjust plot limits
    global PLOT_PARAMS
    if color_limits not in [None, "auto"]:
        limits = [float(lim) for lim in color_limits.split(",")]
        min_lim, max_lim = limits
        lim_warn_msg = "User-specified norm limits for image scaling"
        " could not be applied to the activity heatmap"
    elif (
        "activity_heatmap_color_limits" in PLOT_PARAMS
        and PLOT_PARAMS["activity_heatmap_color_limits"] != "auto"
    ):
        limits = [
            float(lim)
            for lim in PLOT_PARAMS["activity_heatmap_color_limits"].split(",")
        ]
        min_lim, max_lim = limits
        lim_warn_msg = "User-specified norm limits for image scaling"
        " could not be applied to the activity heatmap"
    else:
        abs_max = np.max(np.abs(single_cell_data))
        min_lim, max_lim = -abs_max, abs_max
        lim_warn_msg = "Automatic norm limits for image scaling"
        " could not be applied to the activity heatmap"

    try:
        axis_image.set_clim(min_lim, max_lim)
    except Exception:
        logger.warning(lim_warn_msg)

    # Add color bar
    plt.colorbar(axis_image, ax=ax, label="z-score")

    # Add vertical reference line
    ax.axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

    # Add labels and title
    plot_title = "Event-Aligned Activity Heatmap"

    if event_type:
        subtitle = "\n(event type: {0}".format(event_type)
        if epoch_name:
            subtitle += ", epoch: {0})".format(epoch_name)
        else:
            subtitle += ")"
        plot_title += subtitle
    else:
        if epoch_name:
            plot_title += "\n(epoch: {0})".format(epoch_name)

    ax.set_title(plot_title, fontsize=config.PLOT_TITLE_FONT_SIZE)
    ax.set_xlabel(
        "Time from Event (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )
    ax.set_ylabel("Neuron Number", fontsize=config.PLOT_LABEL_FONT_SIZE)

    # Save as SVG without dpi (avoids rasterization)
    fig.tight_layout()
    fig.savefig(
        output_filename, format="svg", bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def plot_single_cell_modulation(
    x,
    x_limits,
    mean_up,
    sem_up,
    num_cells_up_modulated,
    mean_down,
    sem_down,
    num_cells_down_modulated,
    mean_non,
    sem_non,
    num_cells_non_modulated,
    event_type,
    output_filename,
    modulation_colors=None,
    subtitle=None,
    plot_limits=None,
    epoch_name=None,
):
    """Plot event-aligned activity by activity-modulation group.
    - Mean activity plotted for a given group when the group contains at least 1 cell.
    - Standard error of the mean activity plotted when the group contains at least 2 cells.
      - For 1 cell, uncorrected sample variance is 0 and corrected sample variance is
      undefined, so nothing to plot either way.

    :param x: values to use along the x-axis of the plot
    :param x_limits: limits of the x-axis to use for plots
    :param mean_up: mean trace for up-modulated neurons
    :param sem_up: standard error of the mean for up-modulated neurons
    :param num_cells_up_modulated: number of up-modulated cells
    :param mean_down: mean trace for down-modulated neurons
    :param sem_down: standard error of the mean for down-modulated neurons
    :param num_cells_down_modulated: number of down-modulated cells
    :param mean_non: mean trace for non-modulated neurons
    :param sem_non: standard error of the mean for non-modulated neurons
    :param num_cells_non_modulated: number of non-modulated cells
    :param event_type: type of event for which analysis was performed
    :param output_filename: path to the output image data
    :param modulation_colors: string list with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param subtitle: plot subtitle
    :param plot_limits: y-axis range (z-score) applied to the event-aligned activity by modulation plot
    :param epoch_name: name of the epoch
    """
    # Provide a default value for modulation_colors if not passed

    modulation_colors = validate_modulation_colors(modulation_colors)

    # Dynamic color mapping
    up_color, down_color, non_color = modulation_colors

    fig, ax = plt.subplots()

    # plot data for up-modulated neurons
    if num_cells_up_modulated > 0:
        ax.plot(
            x,
            mean_up,
            color=up_color,
            label="up modulated (n={0})".format(num_cells_up_modulated),
        )
    if num_cells_up_modulated > 1:
        ax.fill_between(
            x,
            mean_up + sem_up,
            mean_up - sem_up,
            alpha=config.PLOT_SHADING_ALPHA,
            color=up_color,
        )

    # plot data for down-modulated neurons
    if num_cells_down_modulated > 0:
        ax.plot(
            x,
            mean_down,
            color=down_color,
            label="down modulated (n={0})".format(num_cells_down_modulated),
        )
    if num_cells_down_modulated > 1:
        ax.fill_between(
            x,
            mean_down + sem_down,
            mean_down - sem_down,
            alpha=config.PLOT_SHADING_ALPHA,
            color=down_color,
        )

    # plot data for non-modulated neurons
    if num_cells_non_modulated > 0:
        ax.plot(
            x,
            mean_non,
            color=non_color,
            label="non modulated (n={0})".format(num_cells_non_modulated),
        )
    if num_cells_non_modulated > 1:
        ax.fill_between(
            x,
            mean_non + sem_non,
            mean_non - sem_non,
            alpha=config.PLOT_SHADING_ALPHA,
            color=non_color,
        )

    # plot vertical line at x=0, the reference event time
    ax.axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

    # set x limits
    ax.set_xlim(x_limits)

    # add labels
    plot_title = "Event-Aligned Activity by Modulation Group"

    if event_type:
        subtitle_text = "\n(event type: {0}".format(event_type)
        if epoch_name:
            subtitle_text += ", epoch: {0})".format(epoch_name)
        else:
            subtitle_text += ")"
        plot_title += subtitle_text

    if subtitle:
        plot_title += "\n{0}".format(subtitle)

    ax.set_title(plot_title, fontsize=config.PLOT_TITLE_FONT_SIZE)

    ax.set_xlabel(
        "Time from Event (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )
    ax.set_ylabel(
        "Neural Activity (z-score)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )

    # adjust plot limits
    global PLOT_PARAMS
    if plot_limits not in [None, "auto"]:
        y_limits = [float(lim) for lim in plot_limits.split(",")]
        ax.set_ylim(y_limits)
    elif (
        "activity_by_modulation_plot_limits" in PLOT_PARAMS
        and PLOT_PARAMS["activity_by_modulation_plot_limits"] != "auto"
    ):
        y_limits = [
            float(lim)
            for lim in PLOT_PARAMS["activity_by_modulation_plot_limits"].split(
                ","
            )
        ]
        ax.set_ylim(y_limits)
    ax.margins(x=0)

    # add legend to figure
    ax.legend()

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_cell_map(
    footprints: np.array,
    modulations: np.array,
    event_type,
    output_filename,
    modulation_colors=None,
    epoch_name=None,
):
    """Plot event-aligned activity by activity-modulation group.

    :param cell_centroids: list of cell centroid coordinates [(x1,y1),(x2,y2),...]
    :param modulations: list of modulation per cell (-1: down, 0: non, +1: up)
    :param footprint_dimensions: dimensions in pixels of an individual footprint
                                 (width, height)
     :param event_type: type of event for which analysis was performed
    :param output_filename: path to the output image data
    :param modulation_colors: string list with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param epoch_name: name of the epoch
    """
    fig, ax = plt.subplots()

    modulation_colors = validate_modulation_colors(modulation_colors)

    # Dynamic color mapping
    modulation_to_color_mapping = {
        1: modulation_colors[0],  # Up-modulated color
        -1: modulation_colors[1],  # Down-modulated color
        0: modulation_colors[2],  # Non-modulated color
    }
    modulation_to_name_mapping = {
        1: "up modulated",
        -1: "down modulated",
        0: "non modulated",
    }

    for modulation in [1, -1, 0]:
        cell_indices = np.where((np.array(modulations) == modulation))[0]

        num_cells = len(cell_indices)

        if num_cells == 0:
            continue

        x, y = footprints_to_contours(footprints[cell_indices, :, :])

        draw_footprints(
            x,
            y,
            color=modulation_to_color_mapping[modulation],
            label="{0} (n={1})".format(
                modulation_to_name_mapping[modulation], num_cells
            ),
        )

    # add labels and legend
    plot_title = "Cell Map"

    if event_type:
        subtitle = "\n(event type: {0}".format(event_type)
        if epoch_name:
            subtitle += ", epoch: {0})".format(epoch_name)
        else:
            subtitle += ")"
        plot_title += subtitle

    ax.set_title(plot_title, fontsize=config.PLOT_TITLE_FONT_SIZE)

    ax.set_xlabel("x (pixels)", fontsize=config.PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel("y (pixels)", fontsize=config.PLOT_LABEL_FONT_SIZE)
    ax.legend()

    ax.set_xlim((0, footprints.shape[1]))
    ax.set_ylim((0, footprints.shape[2]))

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def validate_modulation_colors(modulation_colors):
    """Validate modulation_colors input for compatibility with ax.plot and plt.fill.
    Defaults to predefined colors if the input is invalid or incompatible.

    :param modulation_colors: list of str or str
        List of colors for modulation groups or a comma-separated string.

    :returns:
        list of str
        A valid list of colors, defaulting to the predefined ones if invalid.
    :raises:
        IdeasError: Custom exception with specific error codes and messages.
    """
    # Default colors
    default_modulation_colors = [
        config.PLOT_UP_MODULATED_COLOR,
        config.PLOT_DOWN_MODULATED_COLOR,
        config.PLOT_NON_MODULATED_COLOR,
    ]

    try:
        # Process modulation_colors if it's a string
        if isinstance(modulation_colors, str):
            modulation_colors = modulation_colors.split(",")
            modulation_colors = [x.strip() for x in modulation_colors]

        # Check if modulation_colors is a list or tuple
        if not isinstance(modulation_colors, (list, tuple)):
            raise IdeasError(
                "modulation_colors must be a list, tuple, or a comma-separated string.",
            )

        # Ensure the input contains exactly three colors
        if len(modulation_colors) != 3:
            raise IdeasError(
                "modulation_colors must contain exactly three colors.",
            )

        # Validate each color
        for color in modulation_colors:
            if not isinstance(color, str) or not mcolors.is_color_like(color):
                raise IdeasError(
                    f"Invalid modulation color: '{color}' is not recognized by Matplotlib.",
                )

        # Optional: Ensure no duplicates
        if len(set(modulation_colors)) != len(modulation_colors):
            raise IdeasError(
                "Duplicate colors detected in modulation_colors.",
            )

    except Exception:
        logger.warning(
            "Using default modulation colors: %s", default_modulation_colors
        )
        return default_modulation_colors

    # Return validated colors if all checks pass
    return modulation_colors


def bin_traces_update_period(
    standardized_traces_df,
    traces_timepoints,
    period,
    temporal_downsampling_factor,
):
    """Apply binning to traces and timepoints using a bin size as a multiple of period.

    :Parameters:
        standardized_traces_df (np.array):
            2D NumPy array of standardized traces (rows: timepoints, columns: features).
        traces_timepoints (np.array): 1D NumPy array of timepoints corresponding to traces.
        period (float): Original sampling period (time per data point).
        temporal_downsampling_factor (int):
            Multiplier to determine bin size (bin_size = temporal_downsampling_factor * period).

    :Returns:
        tuple: (binned_traces, binned_timepoints, updated_period).
             binned_traces (np.array): Binned traces with reduced time resolution.
             binned_timepoints (np.array): Binned timepoints.
             updated_period (float): Updated period after binning.
    """
    try:
        # Compute bin size
        bin_size = temporal_downsampling_factor * period

        # Compute bin size in indices
        bin_size_in_idxs = max(1, int(temporal_downsampling_factor))
        num_bins = (
            len(traces_timepoints) + bin_size_in_idxs - 1
        ) // bin_size_in_idxs  # Ensure all data is included

        # Ensure enough data for binning
        if len(traces_timepoints) < bin_size_in_idxs:
            logger.warning(
                "Not enough data points for binning. Returning original data."
            )
            return standardized_traces_df, traces_timepoints, period

        # Apply binning to traces (Y-values)
        binned_traces = np.array(
            [
                np.nanmean(
                    standardized_traces_df[
                        i
                        * bin_size_in_idxs : min(
                            (i + 1) * bin_size_in_idxs, len(traces_timepoints)
                        )
                    ],
                    axis=0,
                )
                for i in range(num_bins)
            ]
        )

        # Apply binning to timepoints (X-values)
        binned_timepoints = np.array(
            [
                np.nanmean(
                    traces_timepoints[
                        i
                        * bin_size_in_idxs : min(
                            (i + 1) * bin_size_in_idxs, len(traces_timepoints)
                        )
                    ]
                )
                for i in range(num_bins)
            ]
        )

        # Update period after binning
        updated_period = bin_size
        logger.info(
            f"The period of the activity traces was updated to {updated_period:.6f}"
        )
        return binned_traces, binned_timepoints, updated_period

    except Exception:
        logger.warning(
            "An error occurred while downsampling the traces. No downsampling was applied. "
            "Processing will continue using the standardized traces."
        )
        return standardized_traces_df, traces_timepoints, period


def validate_peri_event_workflow_parameters(params):
    """Validate the input parameters for the peri-event analysis workflow.

    :param params: dictionary of processing parameters
    """
    # ensure mandatory top-level parameters are specified
    mandatory_parameters = [
        "event_types",
        "visual_window",
        "statistical_window",
        "num_shuffles",
        "significance_threshold",
        "seed",
    ]
    for p in mandatory_parameters:
        if p not in params:
            raise IdeasError(
                "'{0}' must be specified in the peri-event workflow parameters".format(
                    p
                ),
            )

    # validate visual window
    if (
        "pre" not in params["visual_window"]
        or "post" not in params["visual_window"]
    ):
        raise IdeasError(
            "The 'pre' and 'post' keys must be specified under the 'visual_window' parameter",
        )

    if (
        params["visual_window"]["pre"] == 0
        and params["visual_window"]["post"] == 0
    ):
        raise IdeasError(
            "The visual window cannot be empty. 'pre' and 'post' cannot both be equal to 0.",
        )

    if params["visual_window"]["pre"] > 0:
        raise IdeasError(
            "The visual window 'pre' parameter must be specified as a number smaller than 0.",
        )

    if params["visual_window"]["post"] < 0:
        raise IdeasError(
            "The visual window 'post' parameter must be specified as a number greater than 0.",
        )

    # validate statistical window format
    if (len(params["statistical_window"]["pre"]) != 2) or (
        len(params["statistical_window"]["post"]) != 2
    ):
        raise IdeasError(
            "The 'pre' and 'post' parameters of the statistical window must each be specified"
            " as a list of 2 values.",
        )

    # ensure statistical window is not empty or invalid for either pre or post
    for side in ["pre", "post"]:
        if (
            params["statistical_window"][side][0]
            >= params["statistical_window"][side][1]
        ):
            raise IdeasError(
                "The '{0}' statistical window specified is invalid. ".format(
                    side
                )
                + "The end time must be larger than the start time.",
            )

    # ensure pre and post statistical windows are not identical
    if (
        params["statistical_window"]["pre"]
        == params["statistical_window"]["post"]
    ):
        raise IdeasError(
            "The 'pre' and 'post' statistical windows cannot be identical.",
        )

    # ensure statistical window is within the visual window
    if (
        params["statistical_window"]["pre"][0] < params["visual_window"]["pre"]
        or params["statistical_window"]["post"][1]
        > params["visual_window"]["post"]
    ):
        raise IdeasError(
            "The statistical window must be contained within the visual window.",
        )

    # validate number of shuffles
    if params["num_shuffles"] < 1:
        raise IdeasError(
            "The number of shuffles must be equal to or greater than 1",
        )

    if not isinstance(params["num_shuffles"], int):
        raise IdeasError(
            "The number of shuffles must be specified as an integer value.",
        )

    # validate significance threshold
    if (
        params["significance_threshold"] < 0
        or params["significance_threshold"] > 1
    ):
        raise IdeasError(
            "The p-value must be between 0 and 1",
        )

    # validate seed
    if not isinstance(params["seed"], int):
        raise IdeasError(
            "The random seed must be specified as an integer value",
        )

    if params["seed"] < 0 or params["seed"] > (2**32) - 1:
        raise IdeasError(
            "The random seed must be between 0 and 2^32 - 1.",
        )

    # validate optional plot params
    for param_key, param_name in [
        (
            "population_activity_plot_limits",
            "Population activity (y-axis range)",
        ),
        ("activity_heatmap_color_limits", "Activity heatmap (colormap range)"),
        (
            "activity_by_modulation_plot_limits",
            "Activity by modulation group (y-axis range)",
        ),
    ]:
        if param_key in params:
            if params[param_key] != "auto":
                # ensure param is formatted as 'x,y'
                try:
                    min_lim, max_lim = params[param_key].split(",")
                    float(min_lim)
                    float(max_lim)
                except Exception:
                    raise IdeasError(
                        f"Visualization parameter '{param_name}' must be specified as"
                        f" 'auto' or 'min,max' (e.g. -1,1) where the minimum and "
                        f"maximum are not equal.",
                    )

                # ensure x != y
                if min_lim == max_lim:
                    raise IdeasError(
                        f"Visualization parameter '{param_name}' must be specified as"
                        f" 'auto' or 'min,max' (e.g. -1,1) where the minimum and "
                        f"maximum are not equal.",
                    )


def peri_event_population_analysis(
    traces_df,
    event_indices,
    event_indices_shuffles,
    visual_window,
    statistical_window_indices,
    significance_threshold,
    x_values,
    x_limits,
    event_type,
    output_dir,
    single_group_color=None,
):
    """Perform peri-event analysis at the population level.

    :param traces_df: dataframe containing individual cell traces
                      (rows=timepoints, columns=cells)
    :param event_indices: list of event indices
    :param event_indices_shuffles: shuffles of event indices to use to construct
                                   the null distribution
    :param visual_window: window of time to use for visualization purposes (pre, post)
    :param statistical_window_indices: window of time to use for statistical tests
                                      {'pre': [pre_event_indices], 'post': [post_event_indices]}
    :param significance_threshold: significance threshold to use for statistical tests
    :param x_values: values of the x-axis to use for plots
    :param x_limits: limits of the x-axis to use for plots
    :param event_type: type of event for which analysis was performed
    :param output_dir: path to the output directory
    :param single_group_color: the mean and sem trace color for event-aligned population activity
    """
    # compute mean population activity
    # by averaging the traces of all neurons together across time
    single_group_color = (
        single_group_color
        if isinstance(single_group_color, str)
        and mcolors.is_color_like(single_group_color)
        else (
            logger.warning(
                f"Invalid activity trace color '{single_group_color}', defaulting to '#1f77b4'"
            )
            or "#1f77b4"
        )
    )
    mean_trace = traces_df.mean(axis=1)

    # Step 1: Compute true population mean activity across event windows
    # (i.e. windows around each event)
    # extract short traces around each event
    event_windows = extract_event_windows(
        mean_trace, event_indices, visual_window
    )

    # - Compute mean event window, i.e. we average all event windows across time.
    # - Also compute the standard error of the mean (sem) associated with each timepoint.
    mean_event_window = np.nanmean(event_windows, axis=0, dtype="float32")

    if event_windows.shape[0] > 1:
        sem_event_window = stats.sem(
            event_windows,
            axis=0,
            nan_policy="omit",
        ).astype("float32")
    else:
        # no SEM to compute if dealing with a single event
        sem_event_window = np.full_like(
            mean_event_window, math.nan, dtype="float32"
        )

    # Step 2: Construct the null distribution
    # - The null distribution is constructed by shuffling the events a number of times.
    # - For each shuffle, we then compute the mean event window as done above in step 1.
    # - We end up with 'num_shuffles' such mean event windows.
    # - Finally, we average all mean event windows and compute the associated confidence interval.

    # compute mean event window for N shuffles of the event indices
    shuffled_mean_event_windows = extract_mean_event_windows_per_event_shuffle(
        mean_trace, event_indices_shuffles, visual_window
    )

    # compute the mean event window across all shuffles
    shuffled_mean_event_window = np.nanmean(
        shuffled_mean_event_windows, axis=0, dtype="float32"
    )

    # - determine the lower and upper bound of the confidence interval
    #   for the mean event window across all shuffles
    # - this is achieved by computing the appropriate percentiles for each timepoint
    #   giving us a lower and upper bound for the mean event window of the distribution
    shuffled_mean_event_window_conf_interval_low = np.nanpercentile(
        shuffled_mean_event_windows, 100 * (significance_threshold / 2), axis=0
    )
    shuffled_mean_event_window_conf_interval_high = np.nanpercentile(
        shuffled_mean_event_windows,
        100 - (100 * significance_threshold / 2),
        axis=0,
    )

    # plot population mean event window +- sem
    # versus the shuffled mean event window and associated confidence interval
    plot_basename = os.path.join(
        output_dir, "event_aligned_population_activity"
    )
    plot_preview_filename = (
        plot_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )
    plot_population_mean_event_window(
        x_values,
        x_limits,
        mean_event_window,
        sem_event_window,
        shuffled_mean_event_window,
        shuffled_mean_event_window_conf_interval_low,
        shuffled_mean_event_window_conf_interval_high,
        significance_threshold,
        event_type,
        plot_preview_filename,
        color=single_group_color,
    )

    # Step 3: Evaluate statistical significance of the difference between
    # true data and null distribution

    # - Compute mean activity difference between timepoints after the event (post)
    # and timepoints before (pre) the event.
    # - This is achieved by subtracting pre-event activity from post-event activity
    #   over the STATISTICAL window, and averaging the resulting trace to get a single
    #   value representing the "true" difference between pre- and post- event activity.
    true_post_minus_pre = compute_post_minus_pre(
        mean_event_window,
        statistical_window_indices["pre"],
        statistical_window_indices["post"],
    )

    # generate post-pre null distribution
    post_minus_pre_null_dist = generate_post_minus_pre_shuffled_dist(
        shuffled_mean_event_windows,
        statistical_window_indices["pre"],
        statistical_window_indices["post"],
    )

    # compute statistics for true versus null distribution
    (
        zscores,
        pvalues,
        modulations,
        ranks,
        post_minus_pre_null_dist_mu,
        post_minus_pre_null_dist_std,
    ) = compute_statistical_metrics(
        np.array([true_post_minus_pre]),
        np.array(post_minus_pre_null_dist).reshape(
            (len(post_minus_pre_null_dist), 1)
        ),
        significance_threshold,
    )

    # return data of interest
    output_data = {
        "mean": mean_event_window,
        "sem": sem_event_window,
        "shuffled_mean": shuffled_mean_event_window,
        "shuffled_lower_conf": shuffled_mean_event_window_conf_interval_low,
        "shuffled_upper_conf": shuffled_mean_event_window_conf_interval_high,
        "true_mean_post-pre": true_post_minus_pre,
        "shuffled_mean_post-pre": post_minus_pre_null_dist_mu[0],
        "shuffled_std_post-pre": post_minus_pre_null_dist_std[0],
        "z-score": zscores[0],
        "p-value": pvalues[0],
        "modulation": modulations[0],
    }

    # # create population activity preview file
    # population_activity_preview_file = IdeasPreviewFile(
    #     name="Event-aligned population activity figure",
    #     help="Event-aligned average population activity line plot",
    #     file_path=os.path.abspath(plot_preview_filename),
    #     file_format="svg",
    # )

    return output_data


def peri_event_single_cell_analysis(
    traces_df,
    footprints,
    event_indices,
    event_indices_shuffles,
    visual_window,
    statistical_window_indices,
    significance_threshold,
    x_values,
    x_limits,
    event_type,
    output_dir,
    modulation_colors=None,
    cmap=None,
):
    """Perform peri-event analysis at the single-cell level.

    :param traces_df: dataframe containing individual cell traces
                      (rows=timepoints, columns=cells)
    :param footprints: footprints array (num_cells, width, height)
    :param event_indices: list of event indices
    :param event_indices_shuffles: shuffles of event indices to use to
                                   construct the null distribution
    :param visual_window: window of time to use for visualization purposes (pre, post)
    :param statistical_window_indices: window of time to use for statistical tests
                                      {'pre': [pre_event_indices], 'post': [post_event_indices]}
    :param significance_threshold: significance threshold to use for statistical tests
    :param x_values: values of the x-axis to use for plots
    :param x_limits: limits of the x-axis to use for plots
    :param event_type: type of event for which analysis was performed
    :param output_dir: path to the output directory
    :param modulation_colors: string list with color inputs
    :param cmap: colormap applied to the activity heatmap
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    """
    # Provide a default value for modulation_colors if not passed
    modulation_colors = validate_modulation_colors(modulation_colors)

    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )
    # extract short traces around each event
    # - this step returns an array with shape (num_events, num_timepoints_per_window, num_cells)
    event_windows = extract_event_windows(
        traces_df, event_indices, visual_window
    )

    # compute mean & sem event window per cell, i.e. average activity across all windows for each
    # cell - this step returns an array with shape (num_timepoints_per_window, num_cells)
    mean_event_windows = np.nanmean(event_windows, axis=0, dtype="float32")

    if event_windows.shape[0] > 1:
        sem_event_windows = stats.sem(
            event_windows, axis=0, nan_policy="omit"
        ).astype("float32")
    else:
        # no SEM to compute if dealing with a single event
        sem_event_windows = np.full_like(
            mean_event_windows, math.nan, dtype="float32"
        )

    # - Compute mean activity difference between timepoints after the event (post)
    # and timepoints before (pre) the event.
    # - This is achieved by subtracting pre-event activity from post-event activity
    #   over the STATISTICAL window, and averaging the resulting trace to get a single
    #   value representing the "true" difference between pre- and post- event activity.
    # - The above procedure is repeated for all cells.
    # - This step returns a 1D array with length equal to 'num_cells', which contains
    #   the mean post_minus_pre value across all events.
    true_post_minus_pre_dist = compute_post_minus_pre(
        mean_event_windows,
        statistical_window_indices["pre"],
        statistical_window_indices["post"],
    )

    # compute mean event window for N shuffles of the event indices for each cell
    # - this returns an array with shape (num_shuffles, num_timepoints_per_window, num_cells)
    shuffled_mean_event_windows = extract_mean_event_windows_per_event_shuffle(
        traces_df, event_indices_shuffles, visual_window
    )

    # generate post-pre null distribution
    post_minus_pre_null_dist = generate_post_minus_pre_shuffled_dist(
        shuffled_mean_event_windows,
        statistical_window_indices["pre"],
        statistical_window_indices["post"],
    )

    # compute single-cell metrics
    (
        zscores,
        pvalues,
        modulations,
        ranks,
        post_minus_pre_null_dist_mu,
        post_minus_pre_null_dist_std,
    ) = compute_statistical_metrics(
        true_post_minus_pre_dist,
        post_minus_pre_null_dist,
        significance_threshold,
    )

    # data transposed from (num_timepoints_per_window, num_cells)
    # to (num_cells, num_timepoints_per_window) for plotting purposes
    ordered_mean_event_windows = mean_event_windows.T[ranks]

    # generate heatmap showing event-aligned activity modulation per neuron
    heatmap_data_basename = os.path.join(
        output_dir, "event_aligned_single_cell_activity_heatmap"
    )
    heatmap_preview_filename = (
        heatmap_data_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )
    plot_single_neurons_heatmap(
        ordered_mean_event_windows,
        x_limits,
        event_type,
        heatmap_preview_filename,
        cmap=cmap,
    )

    # compute mean and sem for up-, down-, and non- modulated traces
    up_modulated_indices = np.where(modulations == 1)[0]
    down_modulated_indices = np.where(modulations == -1)[0]
    non_modulated_indices = np.where(modulations == 0)[0]

    num_cells_up_modulated = len(up_modulated_indices)
    num_cells_down_modulated = len(down_modulated_indices)
    num_cells_non_modulated = len(non_modulated_indices)

    # up modulated MEAN
    if num_cells_up_modulated > 0:
        mean_up_modulated = np.nanmean(
            mean_event_windows[:, up_modulated_indices],
            axis=1,
            dtype="float32",
        )
    else:
        mean_up_modulated = np.empty(
            mean_event_windows.shape[0], dtype="float32"
        )
        mean_up_modulated.fill(np.nan)
    # up modulated SEM
    if num_cells_up_modulated > 1:
        sem_up_modulated = stats.sem(
            mean_event_windows[:, up_modulated_indices],
            axis=1,
            nan_policy="omit",
        ).astype("float32")
    else:
        sem_up_modulated = np.empty(
            mean_event_windows.shape[0], dtype="float32"
        )
        sem_up_modulated.fill(np.nan)

    # down modulated MEAN
    if num_cells_down_modulated > 0:
        mean_down_modulated = np.nanmean(
            mean_event_windows[:, down_modulated_indices],
            axis=1,
            dtype="float32",
        )
    else:
        mean_down_modulated = np.empty(
            mean_event_windows.shape[0], dtype="float32"
        )
        mean_down_modulated.fill(np.nan)
    # down modulated SEM
    if num_cells_down_modulated > 1:
        sem_down_modulated = stats.sem(
            mean_event_windows[:, down_modulated_indices],
            axis=1,
            nan_policy="omit",
        ).astype("float32")
    else:
        sem_down_modulated = np.empty(
            mean_event_windows.shape[0], dtype="float32"
        )
        sem_down_modulated.fill(np.nan)

    # non modulated MEAN
    if num_cells_non_modulated > 0:
        mean_non_modulated = np.nanmean(
            mean_event_windows[:, non_modulated_indices],
            axis=1,
            dtype="float32",
        )
    else:
        mean_non_modulated = np.empty(mean_event_windows.shape[0])
        mean_non_modulated.fill(np.nan)
    # non modulated SEM
    if num_cells_non_modulated > 1:
        sem_non_modulated = stats.sem(
            mean_event_windows[:, non_modulated_indices],
            axis=1,
            nan_policy="omit",
        ).astype("float32")
    else:
        sem_non_modulated = np.empty(
            mean_event_windows.shape[0], dtype="float32"
        )
        sem_non_modulated.fill(np.nan)

    # compute stats for each modulation group
    # up-modulated
    up_modulated_true_post_minus_pre = None
    up_modulated_zscores = [None]
    up_modulated_pvalues = [None]
    up_modulated_modulations = [None]
    up_modulated_post_minus_pre_null_dist_mu = [None]
    up_modulated_post_minus_pre_null_dist_std = [None]

    if len(up_modulated_indices) > 0:
        up_modulated_true_post_minus_pre = compute_post_minus_pre(
            mean_up_modulated,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        up_modulated_shuffled_mean_event_windows = nanmean_iterative(
            shuffled_mean_event_windows, up_modulated_indices
        )

        up_modulated_post_minus_pre_null_dist = (
            generate_post_minus_pre_shuffled_dist(
                up_modulated_shuffled_mean_event_windows,
                statistical_window_indices["pre"],
                statistical_window_indices["post"],
            )
        )

        (
            up_modulated_zscores,
            up_modulated_pvalues,
            up_modulated_modulations,
            up_modulated_ranks,
            up_modulated_post_minus_pre_null_dist_mu,
            up_modulated_post_minus_pre_null_dist_std,
        ) = compute_statistical_metrics(
            np.array([up_modulated_true_post_minus_pre]),
            np.array(up_modulated_post_minus_pre_null_dist).reshape(
                (len(up_modulated_post_minus_pre_null_dist), 1)
            ),
            significance_threshold,
        )

    # down-modulated
    down_modulated_true_post_minus_pre = None
    down_modulated_zscores = [None]
    down_modulated_pvalues = [None]
    down_modulated_modulations = [None]
    down_modulated_post_minus_pre_null_dist_mu = [None]
    down_modulated_post_minus_pre_null_dist_std = [None]

    if len(down_modulated_indices) > 0:
        down_modulated_true_post_minus_pre = compute_post_minus_pre(
            mean_down_modulated,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        down_modulated_shuffled_mean_event_windows = nanmean_iterative(
            shuffled_mean_event_windows, down_modulated_indices
        )

        down_modulated_post_minus_pre_null_dist = (
            generate_post_minus_pre_shuffled_dist(
                down_modulated_shuffled_mean_event_windows,
                statistical_window_indices["pre"],
                statistical_window_indices["post"],
            )
        )

        (
            down_modulated_zscores,
            down_modulated_pvalues,
            down_modulated_modulations,
            down_modulated_ranks,
            down_modulated_post_minus_pre_null_dist_mu,
            down_modulated_post_minus_pre_null_dist_std,
        ) = compute_statistical_metrics(
            np.array([down_modulated_true_post_minus_pre]),
            np.array(down_modulated_post_minus_pre_null_dist).reshape(
                (len(down_modulated_post_minus_pre_null_dist), 1)
            ),
            significance_threshold,
        )

    # non-modulated
    non_modulated_true_post_minus_pre = None
    non_modulated_zscores = [None]
    non_modulated_pvalues = [None]
    non_modulated_modulations = [None]
    non_modulated_post_minus_pre_null_dist_mu = [None]
    non_modulated_post_minus_pre_null_dist_std = [None]

    if len(non_modulated_indices) > 0:
        non_modulated_true_post_minus_pre = compute_post_minus_pre(
            mean_non_modulated,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        non_modulated_shuffled_mean_event_windows = nanmean_iterative(
            shuffled_mean_event_windows, non_modulated_indices
        )

        non_modulated_post_minus_pre_null_dist = (
            generate_post_minus_pre_shuffled_dist(
                non_modulated_shuffled_mean_event_windows,
                statistical_window_indices["pre"],
                statistical_window_indices["post"],
            )
        )

        (
            non_modulated_zscores,
            non_modulated_pvalues,
            non_modulated_modulations,
            non_modulated_ranks,
            non_modulated_post_minus_pre_null_dist_mu,
            non_modulated_post_minus_pre_null_dist_std,
        ) = compute_statistical_metrics(
            np.array([non_modulated_true_post_minus_pre]),
            np.array(non_modulated_post_minus_pre_null_dist).reshape(
                (len(non_modulated_post_minus_pre_null_dist), 1)
            ),
            significance_threshold,
        )

    # plot (mean +- sem) per modulation group (up, down, non)
    modulation_plot_basename = os.path.join(
        output_dir, "event_aligned_activity_by_modulation"
    )
    modulation_plot_preview_filename = (
        modulation_plot_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )
    plot_single_cell_modulation(
        x_values,
        x_limits,
        mean_up_modulated,
        sem_up_modulated,
        num_cells_up_modulated,
        mean_down_modulated,
        sem_down_modulated,
        num_cells_down_modulated,
        mean_non_modulated,
        sem_non_modulated,
        num_cells_non_modulated,
        event_type,
        modulation_plot_preview_filename,
        modulation_colors=modulation_colors,
    )

    # compute centroid coordinates for every cell
    num_cells_footprints = footprints.shape[0]
    cell_centroids = np.zeros((num_cells_footprints, 2), dtype="float32")
    for i in range(num_cells_footprints):
        cell_centroid = compute_cell_centroid(footprints[i])
        if cell_centroid is None:
            cell_centroids[i] = [-1, -1]
            logger.warning(
                "centroid of cell {0} could not be located".format(i)
            )
        else:
            cell_centroids[i] = cell_centroid

    # generate cell map colored by modulation group
    cell_map_basename = os.path.join(output_dir, "cell_map")
    cell_map_preview_filename = (
        cell_map_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )

    plot_cell_map(
        footprints,
        modulations,
        event_type,
        cell_map_preview_filename,
        modulation_colors=modulation_colors,
    )

    # return data of interest
    output_data = {
        "up_modulated": {
            "mean": mean_up_modulated,
            "sem": sem_up_modulated,
            "num_cells": num_cells_up_modulated,
            "true_mean_post-pre": up_modulated_true_post_minus_pre,
            "shuffled_mean_post-pre": up_modulated_post_minus_pre_null_dist_mu[
                0
            ],
            "shuffled_std_post-pre": up_modulated_post_minus_pre_null_dist_std[
                0
            ],
            "z-score": up_modulated_zscores[0],
            "p-value": up_modulated_pvalues[0],
            "modulation": up_modulated_modulations[0],
        },
        "down_modulated": {
            "mean": mean_down_modulated,
            "sem": sem_down_modulated,
            "num_cells": num_cells_down_modulated,
            "true_mean_post-pre": down_modulated_true_post_minus_pre,
            "shuffled_mean_post-pre": down_modulated_post_minus_pre_null_dist_mu[
                0
            ],
            "shuffled_std_post-pre": down_modulated_post_minus_pre_null_dist_std[
                0
            ],
            "z-score": down_modulated_zscores[0],
            "p-value": down_modulated_pvalues[0],
            "modulation": down_modulated_modulations[0],
        },
        "non_modulated": {
            "mean": mean_non_modulated,
            "sem": sem_non_modulated,
            "num_cells": num_cells_non_modulated,
            "true_mean_post-pre": non_modulated_true_post_minus_pre,
            "shuffled_mean_post-pre": non_modulated_post_minus_pre_null_dist_mu[
                0
            ],
            "shuffled_std_post-pre": non_modulated_post_minus_pre_null_dist_std[
                0
            ],
            "z-score": non_modulated_zscores[0],
            "p-value": non_modulated_pvalues[0],
            "modulation": non_modulated_modulations[0],
        },
        "cell": {  # data for individual cells
            "mean": mean_event_windows.T,  # transposed to return (num_cells, num_points_per_window)
            "sem": sem_event_windows.T,  # transposed to return (num_cells, num_points_per_window)
            "true_mean_post-pre": true_post_minus_pre_dist,
            "shuffled_mean_post-pre": post_minus_pre_null_dist_mu,
            "shuffled_std_post-pre": post_minus_pre_null_dist_std,
            "z-score": zscores,
            "p-value": pvalues,
            "modulation": modulations,
            "rank": ranks,
        },
    }

    # create preview file for single-cell activity heatmap
    # activity_heatmap_preview_file = IdeasPreviewFile(
    #     name="Event-aligned single-cell activity figure",
    #     help="Event-aligned single-cell activity heatmap",
    #     file_path=os.path.abspath(heatmap_preview_filename),
    #     file_format="svg",
    # )

    # # create preview for event-aligned activity by modulation group
    # activity_by_modulation_preview_file = IdeasPreviewFile(
    #     name="Event-aligned sub-population activity figure",
    #     help="Event-aligned average sub-population activity line plot "
    #     "(up-, down-, and non-modulated neurons)",
    #     file_path=os.path.abspath(modulation_plot_preview_filename),
    #     file_format="svg",
    # )

    # # cell map
    # cell_map_preview_file = IdeasPreviewFile(
    #     name="Spatial organization of modulation",
    #     help="Cell map visualizing spatial organization of modulation",
    #     file_path=os.path.abspath(cell_map_preview_filename),
    #     file_format="svg",
    # )

    return (
        output_data,
        # activity_heatmap_preview_file,
        # activity_by_modulation_preview_file,
        # cell_map_preview_file,
    )


def save_event_aligned_traces_to_csv(
    event_aligned_data, x_values, valid_cells, output_filename
):
    """Save event-aligned traces to a csv file.
    :param event_aligned_data: dictionary containing event-aligned data
    :param x_values: values along the x-axis
    :param valid_cells: list of cell names that were processed
    :param output_filename: path to the output csv file
    """
    # create dictionary to hold activity data
    output_activity_dict = OrderedDict()

    # add time data to activity dictionary
    output_activity_dict["Time"] = x_values

    # population and modulation subgroups
    for group_name in [
        "population",
        "up_modulated",
        "down_modulated",
        "non_modulated",
    ]:
        mean_label = "{0}_mean".format(group_name)
        sem_label = "{0}_sem".format(group_name)
        shuffled_mean_label = "{0}_shuffled_mean".format(group_name)
        shuffled_lower_conf_label = "{0}_shuffled_lower_conf".format(
            group_name
        )
        shuffled_upper_conf_label = "{0}_shuffled_upper_conf".format(
            group_name
        )

        if group_name == "population":
            # add population data
            output_activity_dict[mean_label] = event_aligned_data[group_name][
                "mean"
            ]
            output_activity_dict[sem_label] = event_aligned_data[group_name][
                "sem"
            ]
            output_activity_dict[shuffled_mean_label] = event_aligned_data[
                group_name
            ]["shuffled_mean"]
            output_activity_dict[
                shuffled_lower_conf_label
            ] = event_aligned_data[group_name]["shuffled_lower_conf"]
            output_activity_dict[
                shuffled_upper_conf_label
            ] = event_aligned_data[group_name]["shuffled_upper_conf"]
        else:
            # add group data
            output_activity_dict[mean_label] = event_aligned_data[
                "single_cell"
            ][group_name]["mean"]
            output_activity_dict[sem_label] = event_aligned_data[
                "single_cell"
            ][group_name]["sem"]

    # add single-cell data
    for i, cell_name in enumerate(valid_cells):
        mean_label = "{0}_mean".format(cell_name)
        sem_label = "{0}_sem".format(cell_name)
        output_activity_dict[mean_label] = event_aligned_data["single_cell"][
            "cell"
        ]["mean"][i]
        output_activity_dict[sem_label] = event_aligned_data["single_cell"][
            "cell"
        ]["sem"][i]

    # convert activity dict to dataframe
    output_activity_dataframe = pd.DataFrame(output_activity_dict)
    output_activity_dataframe.to_csv(output_filename, index=False)


def save_event_aligned_statistics_to_csv(
    event_aligned_data, valid_cells, output_filename
):
    """Save event-aligned statistics to a csv file.
    :param event_aligned_data: dictionary containing event-aligned data
    :param valid_cells: list of cell names for cells that were processed
    :param output_filename: path to the output csv file
    """
    # create dictionary to hold statistics data
    output_dict = OrderedDict()

    # add 1 column per metric, rows will correspond to cells or groups
    output_dict["name"] = [
        "population",
        "up_modulated",
        "down_modulated",
        "non_modulated",
    ] + valid_cells

    # gather populate metric data
    for col_name in [
        "true_mean_post-pre",
        "shuffled_mean_post-pre",
        "shuffled_std_post-pre",
        "z-score",
        "p-value",
        "modulation",
    ]:
        output_dict[col_name] = [
            event_aligned_data["population"][col_name],
            event_aligned_data["single_cell"]["up_modulated"][col_name],
            event_aligned_data["single_cell"]["down_modulated"][col_name],
            event_aligned_data["single_cell"]["non_modulated"][col_name],
        ] + list(event_aligned_data["single_cell"]["cell"][col_name])

    # convert stats dict to dataframe
    output_stats_dataframe = pd.DataFrame(output_dict)

    # save stats dataframe to disk
    output_stats_dataframe.to_csv(output_filename, index=False)


def peri_event_analysis_for_single_event_type(
    traces_df,
    traces_timepoints,
    footprints,
    events_data,
    event_type,
    valid_cells,
    visual_window,
    statistical_window,
    significance_threshold,
    x_values,
    x_limits,
    period,
    num_shuffles,
    seed,
    modulation_colors,
    single_group_color,
    cmap,
    output_dir,
    input_cellset_files,
    input_events_h5_file,
):
    """Perform peri-event analysis for a single event type.

    :param traces_df: dataframe containing individual cell traces (rows=timepoints, columns=cells)
    :param traces_timepoints: timestamps for the cellular activity traces
    :param footprints: footprints array (num_cells, width, height)
    :param events_data: h5 file object containing the events data
    :param event_type: string representing the event type to analyze
    :param valid_cells: list of cell names to process
    :param visual_window: window of time to use for visualization purposes (pre, post)
    :param statistical_window: window of time to use for statistical tests
                               ([pre_start, pre_end], [post_start, post_end])
    :param significance_threshold: significance threshold to use for statistical tests
    :param x_values: values of the x-axis to use for plots
    :param x_limits: limits of the x-axis to use for plots
    :param period: duration of a sample in the input traces
    :param num_shuffles: number of random shuffles of the event times to perform to
                         construct the null distribution
    :param seed: seed for the random generator used to shuffle event indices
    :param modulation_colors: string list with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param output_dir: path to the output directory for the event type being processed
    :param input_cellset_files: list of paths to the cell set files
    :param input_events_h5_file: path to the events file
    :param single_group_color: the mean and sem trace color for event-aligned population activity
    :param cmap: colormap applied to the activity heatmap
    """
    single_group_color = (
        single_group_color
        if isinstance(single_group_color, str)
        and mcolors.is_color_like(single_group_color)
        else (
            logger.warning(
                f"Invalid activity trace color '{single_group_color}', defaulting to '#1f77b4'"
            )
            or "#1f77b4"
        )
    )
    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )
    # read event times
    try:
        event_times_by_type = events_data["events"][event_type]
    except KeyError:
        raise IdeasError(
            "The event type '{0}' does not exist in the events file.".format(
                event_type
            ),
        )

    try:
        event_times = event_times_by_type["Time"][:]
    except KeyError:
        raise IdeasError(
            "The events file does not contain timestamps for event type '{0}'.".format(
                event_type
            ),
        )

    # convert event times to the corresponding trace time indices
    # (trace time indices correspond to frame indices in the movie they were derived from)
    event_indices = ((event_times / period).round()).astype(int)

    # retain only events for which we can extract the full time window around them
    # without exceeding the recording length
    pre_visual, post_visual = visual_window
    max_time_index = len(traces_df) - 1

    # determine event indices
    # (assume input files form a series when the number of cell set files is greater than one)
    is_series = len(input_cellset_files) > 1
    if is_series:
        # input data consists of a series

        # compute start index of each item in the series
        series_start_indices = [0]
        for f in input_cellset_files[:-1]:
            cs = isx.CellSet.read(f)
            series_start_indices.append(
                series_start_indices[-1] + cs.timing.num_samples
            )

        # - loop over items in the series, each time considering whether events exceed the
        #   temporal boundaries of the item
        series_item_indices = series_start_indices + [max_time_index]

        # - Compute the indices of the beginning and end of each item in the series
        #   assuming a continuous timeline across the series.
        series_item_endpoints = []
        for row_start_index, row_end_index in zip(
            series_item_indices, series_item_indices[1:]
        ):
            # retrieve start and end times of this portion of the series
            subset_traces_start_time = traces_timepoints[row_start_index]
            subset_traces_end_time = traces_timepoints[row_end_index - 1]

            # convert start and end times to the corresponding indices across the entire series
            subset_traces_start_index = int(
                round(subset_traces_start_time / period)
            )
            subset_traces_end_index = int(
                round(subset_traces_end_time / period)
            )

            series_item_endpoints.append(
                (subset_traces_start_index, subset_traces_end_index)
            )

        # compare event indices to the indices of each item in the series
        # and retain those for which we have a complete window
        filtered_event_row_indices = []
        for event_index in event_indices:
            for item_start_index, item_end_index in series_item_endpoints:
                if (event_index - pre_visual >= item_start_index) and (
                    event_index + post_visual <= item_end_index - 1
                ):
                    # map event index to the row index in the parquet traces table
                    # with the closest timestamp
                    event_time = event_index * period
                    event_index_in_table = (
                        np.abs(traces_timepoints - event_time)
                    ).argmin()

                    filtered_event_row_indices.append(event_index_in_table)
                    continue

        # update event indices so that they now represent the row
        # indices to use in the parquet table
        event_indices = filtered_event_row_indices
    else:
        # here we know we are NOT dealing with series data
        event_indices = np.array(
            [
                event_index
                for event_index in event_indices
                if event_index - pre_visual >= 0
                and event_index + post_visual <= max_time_index
            ]
        )

    if len(event_indices) < 1:
        raise IdeasError(
            "There are no events for which a complete time window can be constructed.",
        )

    # prepare shuffles of event indices across the entire timeline of the input trace
    num_events = len(event_indices)
    event_indices_shuffles = []

    # seed random generator to allow reproducibility
    np.random.seed(seed)

    # create a one hot vector with event times
    one_hot_triggers = np.zeros(len(traces_df), bool)
    one_hot_triggers[event_indices] = 1

    # Generate shuffled event data.
    # - When dealing with a small number of events, some shuffles may not have any valid event so
    #   we must reshuffle the data until at least one valid event is obtained.
    # - To prevent an infinite loop, we set a maximum number of iterations.
    num_iterations = 0
    max_num_iterations = 2 * num_shuffles
    while (
        len(event_indices_shuffles) < num_shuffles
        and num_iterations < max_num_iterations
    ):
        # circulate event times
        # this preserves inter-event intervals
        shuffle_indices = np.roll(
            one_hot_triggers, np.random.randint(0, len(traces_df))
        )

        shuffle_indices = np.where(shuffle_indices)[0]

        # filter out events for which we cannot construct a complete window
        shuffle_indices = np.array(
            [
                event_index
                for event_index in shuffle_indices
                if event_index - pre_visual >= 0
                and event_index + post_visual <= max_time_index
            ]
        )

        if len(shuffle_indices) > 0:
            event_indices_shuffles.append(shuffle_indices)

        num_iterations += 1

    if (
        len(event_indices_shuffles) != num_shuffles
        and num_iterations == max_num_iterations
    ):
        logger.warning(
            f"{len(event_indices_shuffles)} shuffles were successfully generated and "
            f"will be used to construct the null distribution"
        )

    # dict to hold output data
    output_data = {}

    # identify pre- and post- event indices for statistical window
    visual_window_indices = np.arange(-visual_window[0], visual_window[1] + 1)
    stats_pre_event_indices = np.where(
        (visual_window_indices >= -statistical_window[0][0])
        & (visual_window_indices <= -statistical_window[0][1])
    )[0]
    stats_post_event_indices = np.where(
        (visual_window_indices >= statistical_window[1][0])
        & (visual_window_indices <= statistical_window[1][1])
    )[0]
    statistical_window_indices = {
        "pre": stats_pre_event_indices,
        "post": stats_post_event_indices,
    }

    # run the peri-event analysis at the POPULATION level
    (
        output_data["population"]
    ) = peri_event_population_analysis(
        traces_df,
        event_indices,
        event_indices_shuffles,
        visual_window,
        statistical_window_indices,
        significance_threshold,
        x_values,
        x_limits,
        event_type,
        output_dir,
        single_group_color=single_group_color,
    )

    logger.info("Population activity analysis completed")

    # run the peri-event analysis at the SINGLE-CELL level
    (
        output_data["single_cell"],
        # single_cell_activity_heatmap_preview_file,
        # activity_by_modulation_preview_file,
        # cell_map_preview_file,
    ) = peri_event_single_cell_analysis(
        traces_df,
        footprints,
        event_indices,
        event_indices_shuffles,
        visual_window,
        statistical_window_indices,
        significance_threshold,
        x_values,
        x_limits,
        event_type,
        output_dir,
        modulation_colors=modulation_colors,
        cmap=cmap,
    )

    logger.info("Single-cell activity analysis completed")

    # save event-aligned neural activity TRACES
    output_traces_csv_filename = os.path.join(
        output_dir, "event_aligned_activity.TRACES.csv"
    )
    save_event_aligned_traces_to_csv(
        output_data, x_values, valid_cells, output_traces_csv_filename
    )

    logger.info(
        f"Event-aligned traces saved "
        f"({os.path.basename(output_traces_csv_filename)}, "
        f"size: {get_file_size(output_traces_csv_filename)})"
    )

    # save event-aligned neural activity STATISTICS
    output_stats_csv_filename = os.path.join(
        output_dir, "event_aligned_activity.STATISTICS.csv"
    )
    save_event_aligned_statistics_to_csv(
        output_data, valid_cells, output_stats_csv_filename
    )

    logger.info(
        f"Event-aligned statistics saved "
        f"({os.path.basename(output_stats_csv_filename)}, "
        f"size: {get_file_size(output_stats_csv_filename)})"
    )

    # create output files/objects/groups and specify relationships

    # source cell set FILES
    # source_cellset_files = [
    #     IdeasFile(
    #         file_key="input_cellset_files",
    #         file_path=os.path.abspath(f),
    #         file_type=FileType.CELL_SET.value[1],
    #         file_format=FileFormat.ISXD_FILE.value[1],
    #         file_structure=FileStructure.BINARY.value[1],
    #         file_category=FileCategory.SOURCE.value[1],
    #     )
    #     for f in input_cellset_files
    # ]

    # # source events FILE
    # source_events_file = IdeasFile(
    #     file_key="input_events_h5_file",
    #     file_path=os.path.abspath(input_events_h5_file),
    #     file_type=FileType.TIMESTAMP_EVENTS.value[1],
    #     file_format=FileFormat.H5_FILE.value[1],
    #     file_structure=FileStructure.SPARSE_TIME_SERIES.value[1],
    #     file_category=FileCategory.SOURCE.value[1],
    # )

    # # event-aligned METADATA
    # first_cs_metadata = read_isxd_metadata(input_cellset_files[0])
    # aligned_traces_metadata = {
    #     config.IDEAS_METADATA_KEY: {
    #         "dataset": {"signal": [{"name": "calcium", "units": "z-score"}]},
    #         "metrics": {
    #             "num_up_modulated_cells": output_data["single_cell"][
    #                 "up_modulated"
    #             ]["num_cells"],
    #             "num_down_modulated_cells": output_data["single_cell"][
    #                 "down_modulated"
    #             ]["num_cells"],
    #             "num_non_modulated_cells": output_data["single_cell"][
    #                 "non_modulated"
    #             ]["num_cells"],
    #             "num_valid_events": num_events,
    #         },
    #         "timingInfo": {
    #             "blank": [],
    #             "cropped": [],
    #             "dropped": [],
    #             "numTimes": len(x_values),
    #             "period": {"num": int(period * 1e6), "den": 1000000},
    #             "sampling_rate": compute_sampling_rate(
    #                 period_num=int(period * 1e6), period_den=1000000
    #             ),
    #             "start": {
    #                 "secsSinceEpoch": {
    #                     "num": int(-visual_window[0] * period * 1e6),
    #                     "den": 1000000,
    #                 },
    #                 "utcOffset": 0,
    #             },
    #             "end": {
    #                 "secsSinceEpoch": {
    #                     "num": int(visual_window[1] * period * 1e6),
    #                     "den": 1000000,
    #                 },
    #                 "utcOffset": 0,
    #             },
    #         },
    #         "spacingInfo": first_cs_metadata["spacingInfo"],
    #     }
    # }

    # # event-aligned traces FILE
    # event_aligned_traces_file = IdeasFile(
    #     file_key="event_aligned_activity_traces_event_aligned_activity_traces_csv_file",
    #     file_path=os.path.abspath(output_traces_csv_filename),
    #     file_type=FileType.EVENT_ALIGNED_NEURAL_DATA.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TIME_SERIES.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     parent_ids=[f.file_id for f in source_cellset_files],
    #     preview_files=[
    #         population_activity_preview_file,
    #         single_cell_activity_heatmap_preview_file,
    #     ],
    #     add_metadata=aligned_traces_metadata,
    # )

    # # statistics METADATA
    # stats_metadata = copy.deepcopy(aligned_traces_metadata)
    # del stats_metadata["ideas"]["dataset"]["signal"]

    # # statistics FILE
    # statistics_file = IdeasFile(
    #     file_key="event_aligned_statistics_event_aligned_activity_statistics_csv_file",
    #     file_path=os.path.abspath(output_stats_csv_filename),
    #     file_type=FileType.STATISTICS.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TABLE.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     parent_ids=[
    #         source_cellset_files[
    #             0
    #         ].file_id,  # footprints retrieved from first cell set file
    #         event_aligned_traces_file.file_id,
    #     ],
    #     preview_files=[
    #         activity_by_modulation_preview_file,
    #         cell_map_preview_file,
    #     ],
    #     add_metadata=stats_metadata,
    # )

    # # peri-event analysis GROUP
    # peri_event_analysis_group = IdeasGroup(
    #     group_key="peri_event_analysis_output",
    #     group_type=GroupType.TOOL_OUTPUT.value[1],
    #     files=[
    #         *source_cellset_files,
    #         source_events_file,
    #         event_aligned_traces_file,
    #         statistics_file,
    #     ],
    # )
    # return peri_event_analysis_group


def run_peri_event_workflow(
    input_cellset_files: List[pathlib.Path],
    input_events_h5_file: pathlib.Path,
    event_type: str,
    visual_window_pre: float = -2.0,
    visual_window_post: float = 2.0,
    statistical_window_pre_start: float = -1.0,
    statistical_window_pre_end: float = 1.0,
    statistical_window_post_start: float = 0.0,
    statistical_window_post_end: float = 1.0,
    num_shuffles: int = 1000,
    significance_threshold: float = 0.05,
    seed: int = 0,
    modulation_colors: str = "green, blue, black",
    single_group_color: str = "#1f77b4",
    cmap: str = "coolwarm",
    temporal_downsampling_factor: int = 1,
    population_activity_plot_limits: str = "auto",
    activity_heatmap_color_limits: str = "auto",
    activity_by_modulation_plot_limits: str = "auto",
    output_dir: Optional[str] = None,
):
    """Peri-event analysis workflow.

    :param input_cellset_files: list of paths to the cell set files
    :param input_events_h5_file: path to the events file
    :param event_type: string representing an event type (currently only supports 1 event type)
    :param visual_window_pre: time in seconds before each event to use for visualization
    :param visual_window_post: time in seconds after each event to use for visualization
    :param statistical_window_pre_start: start of time range in seconds before each event
     to use for statistical tests
    :param statistical_window_pre_end: end of time range in seconds before each event
     to use for statistical tests
    :param statistical_window_post_start: start of time range in seconds after each event
     to use for statistical tests
    :param statistical_window_post_end: end of time range in seconds after each event
     to use for statistical tests
    :param num_shuffles: number of random shuffles of the event times to perform
    :param significance_threshold: threshold to use for statistical significance tests
    :param seed: seed for the random generator used to shuffle event indices
    :param modulation_colors: comma separated strings with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param single_group_color: the mean and sem trace color for event-aligned population activity
    :param cmap: colormap applied to the activity heatmap
    :param population_activity_plot_limits: y-axis range (z-score) applied to the event-aligned
     population activity plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_heatmap_color_limits: colormap range (z-score) applied to the activity heatmap
     specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_by_modulation_plot_limits: y-axis range (z-score) applied to the event-aligned
     activity by modulation plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param output_dir: path to the output directory
    """
    logger.info("Starting the peri-event analysis workflow")

    modulation_colors = validate_modulation_colors(modulation_colors)
    single_group_color = (
        single_group_color
        if isinstance(single_group_color, str)
        and mcolors.is_color_like(single_group_color)
        else (
            logger.warning(
                f"Invalid activity trace color '{single_group_color}', defaulting to '#1f77b4'"
            )
            or "#1f77b4"
        )
    )
    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )

    # set output directory to the current working directory if unspecified
    if output_dir is None:
        output_dir = os.getcwd()

    # validate input files exist
    for f in [
        *input_cellset_files,
        input_events_h5_file,
    ]:
        if not os.path.exists(f):
            raise IdeasError(
                "Input file '{0}' does not exist.".format(os.path.basename(f)),
            )

    # construct parameter dictionary
    parameters = {
        "event_types": [event_type],
        "visual_window": {
            "pre": visual_window_pre,
            "post": visual_window_post,
        },
        "statistical_window": {
            "pre": [statistical_window_pre_start, statistical_window_pre_end],
            "post": [
                statistical_window_post_start,
                statistical_window_post_end,
            ],
        },
        "num_shuffles": num_shuffles,
        "significance_threshold": significance_threshold,
        "seed": seed,
        "population_activity_plot_limits": population_activity_plot_limits,
        "activity_heatmap_color_limits": activity_heatmap_color_limits,
        "activity_by_modulation_plot_limits": activity_by_modulation_plot_limits,
    }

    # validate processing parameters
    validate_peri_event_workflow_parameters(parameters)

    # store plot params globally
    global PLOT_PARAMS
    PLOT_PARAMS[
        "population_activity_plot_limits"
    ] = population_activity_plot_limits
    PLOT_PARAMS[
        "activity_heatmap_color_limits"
    ] = activity_heatmap_color_limits
    PLOT_PARAMS[
        "activity_by_modulation_plot_limits"
    ] = activity_by_modulation_plot_limits
    PLOT_PARAMS["modulation_colors"] = modulation_colors

    # sort input isxd cell sets in chronological order
    input_cellset_files = _sort_isxd_files_by_start_time(input_cellset_files)

    # validate series compatibility of input cell sets
    validate_cellset_series_compatibility(input_cellset_files)

    # load cell sets
    cellsets = [isx.CellSet.read(f) for f in input_cellset_files]

    # retrieve cell statuses
    (
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
    ) = get_num_cells_by_status(input_cellset_files[0])

    if num_accepted_cells > 0:
        logger.info(f"Detected {num_accepted_cells} accepted cells to analyze")
        cell_indices_to_process = [
            i
            for i in range(cellsets[0].num_cells)
            if cellsets[0].get_cell_status(i) == "accepted"
        ]
    elif num_undecided_cells > 0:
        logger.info(
            f"No accepted cells were detected. "
            f"Detected {num_undecided_cells} undecided cells to analyze"
        )
        cell_indices_to_process = [
            i
            for i in range(cellsets[0].num_cells)
            if cellsets[0].get_cell_status(i) == "undecided"
        ]
    else:
        raise IdeasError(
            "There are no accepted or undecided cells to process",
        )

    cell_names_to_process = [
        cellsets[0].get_cell_name(i) for i in cell_indices_to_process
    ]

    # ensure there is at least one non-rejected cell
    if len(cell_names_to_process) < 1:
        raise IdeasError(
            "There are no non-rejected cells in the input cell sets.",
        )

    # retrieve relevant processing parameters from metadata
    first_cell_set_metadata = read_isxd_metadata(input_cellset_files[0])
    period = (
        first_cell_set_metadata["timingInfo"]["period"]["num"]
        / first_cell_set_metadata["timingInfo"]["period"]["den"]
    )

    # load traces and footprints for non-rejected cells
    traces_dict = {}
    footprints = np.zeros(
        (
            len(cell_names_to_process),
            cellsets[0].spacing.num_pixels[1],
            cellsets[0].spacing.num_pixels[0],
        ),
        dtype="float32",
    )

    # collect footprints and traces of non-rejected cells
    for i, cell_index in enumerate(cell_indices_to_process):
        footprints[i] = cellsets[0].get_cell_image_data(cell_index).T
        cell_name = cellsets[0].get_cell_name(cell_index)
        traces_dict[cell_name] = np.concatenate(
            [cs.get_cell_trace_data(cell_index) for cs in cellsets]
        )
    traces_df = pd.DataFrame.from_dict(traces_dict)

    # standardize traces (z-scores)
    standardized_traces_df = (traces_df - traces_df.mean()) / traces_df.std(
        ddof=0
    )

    # construct traces timepoints
    traces_timepoints = []
    for i, cs in enumerate(cellsets):
        item_timepoints = np.array(
            [
                n * cs.timing.period.secs_float
                for n in range(cs.timing.num_samples)
            ]
        )

        if i > 0:
            item_timepoints += (
                traces_timepoints[-1]
                + cellsets[i - 1].timing.period.secs_float
            )

        traces_timepoints.extend(item_timepoints)

    # remove cell set objects
    del cellsets[:]

    # bin trace, timestamps and update period to the new bin_size
    logger.info(
        f"Downsampling activity traces with a factor of {temporal_downsampling_factor}"
    )
    (
        standardized_traces_df,
        traces_timepoints,
        period,
    ) = bin_traces_update_period(
        standardized_traces_df,
        traces_timepoints,
        period,
        temporal_downsampling_factor,
    )

    # visual window as (pre, post)
    visual_window = (
        parameters["visual_window"]["pre"],
        parameters["visual_window"]["post"],
    )
    # statistical window as ([pre_start, pre_end], [post_start, post_end])
    statistical_window = (
        parameters["statistical_window"]["pre"],
        parameters["statistical_window"]["post"],
    )
    num_shuffles = parameters["num_shuffles"]
    significance_threshold = parameters["significance_threshold"]

    # convert visual and statistical windows to the equivalent number of frames
    visual_window_frames = (
        int(np.ceil(abs(visual_window[0]) / period)),
        int(np.ceil(abs(visual_window[1]) / period)),
    )
    statistical_window_frames = (
        [
            int(round(abs(statistical_window[0][0]) / period)),
            int(round(abs(statistical_window[0][1]) / period)),
        ],
        [
            int(round(abs(statistical_window[1][0]) / period)),
            int(round(abs(statistical_window[1][1]) / period)),
        ],
    )

    # define values to use along the x-axis, i.e. time in seconds from event
    num_x_values = visual_window_frames[0] + 1 + visual_window_frames[1]
    x_values = np.linspace(
        -visual_window_frames[0] * period,
        visual_window_frames[1] * period,
        num_x_values,
    )
    x_limits = (visual_window[0], visual_window[1])

    # ensure time window does not exceed length of traces
    num_timepoints = first_cell_set_metadata["timingInfo"]["numTimes"]
    visual_window_length = (
        visual_window_frames[0] + 1 + visual_window_frames[1]
    )
    if visual_window_length > num_timepoints:
        raise IdeasError(
            "The visual window must be temporally shorter than the input traces.",
        )

    # ensure that the statistical window is contained within the visual window
    if (
        statistical_window_frames[0][0] > visual_window_frames[0]
        or statistical_window_frames[1][1] > visual_window_frames[1]
    ):
        raise IdeasError(
            "The statistical window must be contained within the visual window.",
        )
    # retrieve event types to consider from parameters
    events_data = h5py.File(input_events_h5_file, "r")
    try:
        event_types = parameters["event_types"]
    except KeyError:
        # if event types is not specified, process all event types in the h5 file
        event_types = list(events_data["events"].keys())

    # if empty list provided, process all event types in the h5 file
    if len(event_types) == 0:
        event_types = list(events_data["events"].keys())

    # retain unique event types to avoid processing same event type more than once
    event_types = np.unique(event_types).tolist()

    # perform peri-event analysis for each event type independently
    for event_type_name in event_types:
        # create output directory to store outputs for this event type
        event_type_output_dir = os.path.join(
            output_dir, "event_type_{0}".format(event_type)
        )

        if not os.path.exists(event_type_output_dir):
            os.makedirs(event_type_output_dir)
        else:
            shutil.rmtree(event_type_output_dir)
            os.mkdir(event_type_output_dir)

        # run peri-event analysis
        peri_event_analysis_for_single_event_type(
            traces_df=standardized_traces_df,
            traces_timepoints=traces_timepoints,
            footprints=footprints,
            events_data=events_data,
            event_type=event_type_name,
            valid_cells=cell_names_to_process,
            visual_window=visual_window_frames,
            statistical_window=statistical_window_frames,
            significance_threshold=significance_threshold,
            x_values=x_values,
            x_limits=x_limits,
            period=period,
            num_shuffles=num_shuffles,
            seed=parameters["seed"],
            modulation_colors=PLOT_PARAMS["modulation_colors"],
            single_group_color=single_group_color,
            cmap=cmap,
            output_dir=event_type_output_dir,
            input_cellset_files=input_cellset_files,
            input_events_h5_file=input_events_h5_file,
        )

        # # save output manifest to disk
        # save_output_manifest(peri_event_analysis_group, output_dir)

        # # save output metadata manifest to disk
        # peri_event_analysis_group.add_group_metadata(
        #     key="event_type", value=event_type_name
        # )
        # save_metadata_manifest(peri_event_analysis_group, output_dir)

    logger.info("Peri-event analysis workflow completed")
