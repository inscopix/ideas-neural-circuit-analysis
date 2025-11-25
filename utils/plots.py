import logging
from math import ceil
from typing import List, Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
from beartype import beartype
from ideas.analysis import plots
from matplotlib.ticker import FixedLocator, FixedFormatter

from utils.utils import Comp, Rescale, save_optimized_svg
from utils import config
from utils.utils import get_num_cells_by_status
from ideas.analysis import io
from ideas.exceptions import IdeasError

logger = logging.getLogger(__name__)

# Consistent font settings
TITLE_FONT = {"fontsize": 13}
SUPTITLE_FONT = {"fontsize": 14}
LABEL_FONT = {"fontsize": 12}

# Default arguments for boxplots for previews
BOX_PLOT_ARGS = {
    "fill": False,
    "linewidth": 2.5,
    "gap": 0.1,
}


def plot_trace_preview(
    traces,
    epochs,
    boundaries,
    period,
    epoch_colors,
    epoch_names,
    spacing: float = 5.01,
):
    """Plot the first 10 cells and highlight the epochs."""
    time = np.arange(0, traces.shape[0] * period, period)
    num_cells = min(traces.shape[1], 50)

    # account for off by one error
    if len(time) != traces.shape[0]:
        time = time[:-1]
    fig, ax = plt.subplots(figsize=(15, 10))

    for y in range(num_cells):
        x = y * spacing + scipy.stats.zscore(traces[:, y], nan_policy="omit")
        ax.plot(time, x, linewidth=1, color="black")

    for idx, epoch in enumerate(epochs):
        ax.axvspan(
            epoch[0],
            epoch[1],
            color=epoch_colors[idx],
            alpha=0.5,
            label=epoch_names[idx],
        )
        ax.set_ylabel("Cell", fontdict=LABEL_FONT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Time (s)", fontdict=LABEL_FONT)
    ax.set_title(
        f"Traces of the first {num_cells} cells ", fontdict=TITLE_FONT
    )

    # add cellset boundaries as vertical dashed lines if several cellsets;
    # "boundaries" is a list of temporal boundaries between cellsets,
    # including 0 and last global time point, in seconds;
    # below, we only need to get boundaries across cellsets in order to place
    # the vertical dashed lines, which is why we remove the first (0) and last
    # (last global time point) items from this "boundaries" list
    if len(boundaries) > 2:
        for idx, boundary in enumerate(boundaries[1:-1]):
            if idx == 0:
                label = "cellset boundary"
            else:
                label = None
            ax.axvline(boundary, linestyle="--", color="k", label=label)

    # remove y tick labels
    ax.get_yaxis().set_ticks([])
    ax.set_yticklabels([])

    # get fig handles
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[: len(epochs) + 1],
        labels[: len(epochs) + 1],
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )
    fig.tight_layout()
    fig.savefig(
        "Trace_Preview.svg",
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


def _create_group_preview(
    df, metric, epoch_names, epoch_colors, group_name, save_name
):
    """Create a group preview plot comparing different epochs based on a
    specified metric.

    :Parameters
        df (pd.DataFrame): DataFrame containing the data to
        be plotted. Must include columns 'Epoch' and the specified metric.
        metric (str): The metric to be used for comparison.
        epoch_names (list of str): List of epoch names to be compared.
        epoch_colors (list of str): List of colors corresponding to each epoch for the boxplot.
        group_name (str): Name of the group for saving the plot file.
    :Returns
        None:
        The function saves the generated
        plot as a SVG file named '{group_name}_{metric}_preview.svg'.
    """
    num_plots = ceil((len(epoch_names) * (len(epoch_names) - 1)) / 2) + 1

    fig, ax = plt.subplots(nrows=num_plots, figsize=(7, num_plots * 5))

    _clean_ax(ax[0])
    sns.boxplot(
        ax=ax[0],
        data=df,
        x="Epoch",
        y=f"{metric} Activity",
        hue="Epoch",
        palette=epoch_colors,
        **BOX_PLOT_ARGS,
    )
    ax[0].set_ylabel(f"{metric} Activity", fontdict=LABEL_FONT)
    ax[0].set_xlabel("Epoch", fontdict=LABEL_FONT)
    ax[0].set_title(
        f"{group_name} {metric} Activity Comparison", fontdict=TITLE_FONT
    )
    counter = 1
    for i in range(len(epoch_names)):
        for j in range(i, len(epoch_names) - 1):
            a_epoch = df[df["Epoch"] == epoch_names[i]]
            b_epoch = df[df["Epoch"] == epoch_names[j + 1]]

            diff = (
                b_epoch[f"{metric} Activity"].values
                - a_epoch[f"{metric} Activity"].values
            )
            cool_warm, norm = plots.plot_shaded_hist(
                values=diff,
                ax=ax[counter],
                title=f"{epoch_names[j + 1]} - {epoch_names[i]}",
                xlabel="Difference in Activity",
            )

            # add colorbar
            sm = plt.cm.ScalarMappable(cmap=cool_warm, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax[counter], label="Difference in Activity")
            counter += 1
    fig.tight_layout()
    fig.savefig(
        f"{save_name}_{metric}_preview.svg",
        dpi=300,
        transparent=True,
    )


def _plot_combined_data(
    data,
    corr_data,
    epoch_colors,
    group_name,
    save_name,
):
    """Plot the combined trace and event data for a group.

    :Args
        trace_data (pd.DataFrame): Combined DataFrame of all trace data.
        event_data (pd.DataFrame): Combined DataFrame of all event data.
        group_name (str): Name of the group for which data is being combined.
    """
    # Plot Trace Data
    fig, ax = plt.subplots(ncols=2)

    _clean_ax(ax[1])
    # Plot Trace Data
    _create_group_preview(
        df=data,
        metric="Trace",
        epoch_names=data.Epoch.unique(),
        epoch_colors=epoch_colors,
        group_name=group_name,
        save_name=save_name,
    )

    # Plot Event Data
    if "Eventrate Activity" in data.columns:
        _create_group_preview(
            df=data,
            metric="Eventrate",
            epoch_names=data.Epoch.unique(),
            epoch_colors=epoch_colors,
            group_name=group_name,
            save_name=save_name,
        )

    # Plot Correlation Data
    if corr_data is not None:
        fig, ax = plt.subplots(nrows=2)
        ax[0].set_title(f"{group_name} Correlation Data", fontdict=TITLE_FONT)
        _clean_ax(ax[0])

        _clean_ax(ax[1])
        ax[1].spines["bottom"].set_visible(False)
        ax[1].spines["top"].set_visible(True)
        sns.boxplot(
            ax=ax[0],
            data=corr_data,
            x="Epoch",
            y="Average Positive Correlation",
            palette=epoch_colors,
            hue="Epoch",
            **BOX_PLOT_ARGS,
        )
        ax[0].set_ylabel("Average Positive Correlation", fontdict=LABEL_FONT)
        ax[0].set_xlabel("Epoch", fontdict=LABEL_FONT)
        ax[0].set_ylim(
            0, np.max(corr_data["Average Positive Correlation"]) * 1.1
        )
        ax[0].set_xticklabels([])
        ax[0].set_xlabel("")
        sns.boxplot(
            ax=ax[1],
            data=corr_data,
            x="Epoch",
            y="Average Negative Correlation",
            palette=epoch_colors,
            hue="Epoch",
            **BOX_PLOT_ARGS,
        )
        ax[1].set_ylabel("Average Negative Correlation")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylim(
            np.min(corr_data["Average Negative Correlation"]) * 1.1, 0
        )
        ax[1].set_ylabel("Average Negative Correlation", fontdict=LABEL_FONT)
        ax[1].set_xlabel("Epoch", fontdict=LABEL_FONT)
        fig.tight_layout()
        fig.savefig(
            f"{save_name}_correlation_preview.svg",
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )


def _plot_average_timecourse(
    ax,
    time,
    period,
    avg_data,
    epochs,
    boundaries,
    epoch_colors,
    epoch_names,
    data_name,
):
    ax.plot(time, avg_data, color="black", linewidth=2)
    ax.set_xlabel("Time (s)", fontdict=LABEL_FONT)

    # Highlight the trace for each epoch
    average_epoch_data = []
    for idx, epoch in enumerate(epochs):
        start_bin = int(epoch[0] / period)
        end_bin = int(epoch[1] / period)
        start = time[start_bin]
        # check if the end bin is out of bounds
        if end_bin >= len(avg_data):
            end = time[-1]
        else:
            end = time[end_bin]

        ax.axvspan(
            start,
            end,
            color=epoch_colors[idx],
            alpha=0.15,
            label=epoch_names[idx],
        )
        average_epoch_data.append(avg_data[start_bin:end_bin])

    # add cellset boundaries as vertical dashed lines if several cellsets
    if len(boundaries) > 2:
        for idx, boundary in enumerate(boundaries[1:-1]):
            if idx == 0:
                label = "cellset boundary"
            else:
                label = None
            ax.axvline(boundary, linestyle="--", color="k", label=label)

    ax.set_ylabel(
        f"Average {data_name}\nPopulation Activity", fontdict=LABEL_FONT
    )
    # ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim([0, time[-1]])

    return average_epoch_data


def _plot_timecourse(
    data_name,
    data,
    epochs,
    boundaries,
    epoch_names,
    epoch_colors,
    period,
    raw_data,
    scale_method,
    heatmap="auto",
):
    """Create function will plot the average timecourse of activity across the
    entire recording as well as plot a single view overview of the data.

    For traces it will be a heatmap, and for events it will be a raster
    plot.
    """
    avg_data = np.nanmean(data, axis=1)

    # Plotting average timecourse
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 4), width_ratios=[10, 1]
    )
    # convert bin values from frames to seconds
    time = np.arange(0, data.shape[0]) * period
    # create single time course trace
    average_epoch_data = _plot_average_timecourse(
        ax=ax[0],
        time=time,
        period=period,
        avg_data=avg_data,
        epochs=epochs,
        boundaries=boundaries,
        epoch_colors=epoch_colors,
        epoch_names=epoch_names,
        data_name=data_name,
    )
    if (
        scale_method == Rescale.FRACTIONAL_CHANGE.value
        and data_name == "Traces"
    ):
        ax[0].axhline(1, color="grey", linestyle="--")
    elif scale_method == Rescale.STANDARDIZE_EPOCH.value:
        ax[0].axhline(0, color="grey", linestyle="--")
    ax[0].legend(loc="upper right")

    # Plot barplot of the average activity for each epoch using ax.bar()
    mean_vals = np.array([float(np.nanmean(x)) for x in average_epoch_data])
    std_vals = np.array([float(np.nanstd(x)) for x in average_epoch_data])
    x_positions = np.arange(len(mean_vals))

    ax[1].bar(
        x_positions,
        mean_vals,
        color=epoch_colors,
        edgecolor="black",
        alpha=0.75,
    )

    # Add error bars
    ax[1].errorbar(
        x_positions,
        mean_vals,
        yerr=std_vals,
        fmt="none",
        ecolor="black",
        capsize=5,
    )

    if scale_method == Rescale.NONE.value:
        label = f"Average {data_name}\nPopulation Activity"
    elif scale_method == Rescale.NORMALIZE.value:
        label = f"Average Normalized \n{data_name}\nPopulation Activity"
    elif (
        scale_method == Rescale.STANDARDIZE.value
        or scale_method == Rescale.STANDARDIZE_EPOCH.value
    ):
        label = f"Average Standardized \n{data_name}\nPopulation Activity"
    elif scale_method == Rescale.FRACTIONAL_CHANGE.value:
        label = f"Average Fractional Change \n{data_name}\nPopulation Activity"
    else:
        logger.warning(f"Unknown scale method {scale_method}")
        label = f"Average {data_name}\nPopulation Activity"
    ax[1].set_ylabel(label, fontdict=LABEL_FONT)
    ax[0].set_ylabel(label, fontdict=LABEL_FONT)
    # Set tick positions and labels properly using FixedLocator and FixedFormatter:
    tick_positions = np.arange(len(epoch_names))
    ax[1].xaxis.set_major_locator(FixedLocator(tick_positions))
    ax[1].xaxis.set_major_formatter(FixedFormatter(epoch_names))

    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)

    fig.tight_layout()
    fig.savefig(
        f"{data_name}_Population_Timecourse.svg",
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )

    # plot the heatmap/raster plot
    fig, ax = plt.subplots(nrows=2, figsize=(8, 8), height_ratios=[1, 7])
    # Plot the average timecourse on top
    average_epoch_data = _plot_average_timecourse(
        ax=ax[0],
        time=time,
        period=period,
        avg_data=avg_data,
        epochs=epochs,
        boundaries=boundaries,
        epoch_colors=epoch_colors,
        epoch_names=epoch_names,
        data_name=data_name,
    )
    ax[0].spines["bottom"].set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_xticklabels([])
    ax[0].set_xlabel(None)

    if scale_method == Rescale.NONE.value:
        label = f"{data_name} Activity"
    elif scale_method == Rescale.NORMALIZE.value:
        label = f"Normalized\n{data_name}\nActivity"
    elif scale_method == Rescale.STANDARDIZE.value:
        label = f"Standardized\n{data_name}\nActivity"
    elif scale_method == Rescale.FRACTIONAL_CHANGE.value:
        label = f"Fractional\nChange in\n{data_name}\nActivity"
        if data_name == "Traces":
            ax[0].axhline(1, color="grey", linestyle="--")
    elif scale_method == Rescale.STANDARDIZE_EPOCH.value:
        label = f"Standardized\n{data_name}\nActivity"
        ax[0].axhline(0, color="grey", linestyle="--")
    else:
        logger.warning(f"Unknown scale method {scale_method}")
        label = f"{data_name} Activity"
    ax[0].set_ylabel(label, fontdict=LABEL_FONT)
    if data_name == "Traces":
        ax[0].set_xlim([0, time[-1]])
        if heatmap == "auto" or heatmap is None:
            sns.heatmap(
                data.T,
                ax=ax[1],
                cmap="viridis",
                cbar_kws={
                    "label": label,
                    "orientation": "horizontal",
                    "location": "bottom",
                    "pad": 0.1,
                },
            )
        else:
            sns.heatmap(
                data.T,
                vmin=heatmap[0],
                vmax=heatmap[1],
                ax=ax[1],
                cmap="viridis",
                cbar_kws={
                    "label": label,
                    "orientation": "horizontal",
                    "location": "bottom",
                    "pad": 0.1,
                },
            )
        ax[1].set_ylabel("Cell #", fontdict=LABEL_FONT)
        ax[1].set_yticks(np.linspace(0, data.shape[1], 5))
        ax[1].set_yticklabels(
            [int(label) for label in np.linspace(0, data.shape[1], 5)]
        )

        ax[1].set_xlabel("Time (s)", fontdict=LABEL_FONT)
        ax[1].set_xticks(np.linspace(0, len(time), 5))
        ax[1].set_xticklabels(
            [int(label) for label in np.linspace(0, time[-1], 5)]
        )
        ax[1].tick_params(axis="x", rotation=0)

        # converting boundaries from seconds to samples, to enable proper x-axis location on heatmap
        length_traces_samp = data.shape[0]
        length_traces_s = boundaries[-1]
        boundaries = [
            np.min(
                (
                    int(x * length_traces_samp / length_traces_s),
                    length_traces_samp,
                )
            )
            for x in boundaries
        ]

    elif data_name == "Eventrate":
        ax[1].eventplot(raw_data, color="black", alpha=0.15)
        for idx, epoch in enumerate(epochs):
            e_events = []
            for cell in raw_data:
                e_cell_events = [
                    event
                    for event in cell
                    if event >= epoch[0] and event <= epoch[1]
                ]
                e_events.append(e_cell_events)

            ax[1].eventplot(
                e_events,
                color=epoch_colors[idx],
                label=epoch_names[idx],
            )

        max_val = data.shape[0] * period
        ax[1].set_ylabel("Cell #", fontdict=LABEL_FONT)
        ax[1].set_xlabel("Time (s)", fontdict=LABEL_FONT)
        ax[1].set_xlim([0, max_val])
        ax[1].set_ylim([0, data.shape[1]])
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
    # add cellset boundaries as vertical dashed lines if several cellsets
    if len(boundaries) > 2:
        for idx, boundary in enumerate(boundaries[1:-1]):
            if idx == 0:
                label = "cellset boundary"
            else:
                label = None
            ax[1].axvline(boundary, linestyle="--", color="k", label=label)
    save_optimized_svg(
        fig,
        f"{data_name}_Single_Cell_Timecourse.svg",
        max_size_mb=10,
        # Use more padding for eventrate plots, otherwise axis labels get cut off
        pad_inches=(0.3 if data_name == "Eventrate" else 0.1),
    )
    # save timecourse data
    np.save(f"{data_name}_timecourse_data", data)


def _plot_correlation(
    *,
    epochs,
    epoch_colors,
    df,
    ylabel,
):
    fig, ax = plt.subplots(nrows=2, figsize=(1.5 * len(epochs), 6))
    # Plot the positive correlation
    sns.barplot(
        data=df[df["Activity"] > 0],
        x="Epoch",
        y="Activity",
        ax=ax[0],
        palette=epoch_colors,
        hue="Epoch",
        errorbar="sd",
    )
    ax[0].set_ylabel(f"Average Positive {ylabel}", fontdict=LABEL_FONT)
    ax[0].set_xticks([])
    ax[0].set_xlabel("")
    ax[0].spines["bottom"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    # Plot the negative correlation
    sns.barplot(
        data=df[df["Activity"] < 0],
        x="Epoch",
        y="Activity",
        ax=ax[1],
        palette=epoch_colors,
        hue="Epoch",
        errorbar="sd",
    )

    ax[1].set_ylabel(f"Average Negative {ylabel}", fontdict=LABEL_FONT)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        "Average_Correlations.svg",
        dpi=300,
        transparent=True,
    )


def _plot_box_and_strip(
    epochs,
    epoch_names,
    epoch_colors,
    df,
    num_cells,
    ylabel,
    save_name,
):
    """Plot a combination of boxplot and stripplot for given epochs and
    activity data.

    :Parameters
        epochs (list): List of epoch identifiers.
        epoch_names (list): List of names corresponding to each epoch.
        epoch_colors (list): List of colors corresponding to each epoch.
        df (pandas.DataFrame): DataFrame containing the activity
        data with columns 'Epoch' and 'Activity'.
        num_cells (int): Number of cells to be plotted.
        ylabel (str): Label for the y-axis.
        save_name (str): File path to save the resulting plot.
    :Returns
        None
    """
    fig, ax = plt.subplots(figsize=(2.5 * len(epochs), 6))
    # Plot the epoch-epoch change
    plots.plot_paired_lines(
        df=df,
        category_names=epoch_names,
        num_values=num_cells,
        ax=ax,
    )
    df = df.reset_index()
    # Plot the boxplot
    sns.boxplot(
        data=df,
        x="Epoch",
        y="Activity",
        ax=ax,
        palette=epoch_colors,
        linewidth=3,
        fliersize=0,
        zorder=2,
        hue="Epoch",
    )
    # Plot the stripplot
    sns.stripplot(
        data=df,
        x="Epoch",
        y="Activity",
        ax=ax,
        color="black",
        size=3,
        alpha=0.2,
        jitter=0.02,
        zorder=1,
        legend=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, fontdict=LABEL_FONT)

    fig.tight_layout()
    fig.savefig(
        save_name,
        dpi=300,
        transparent=True,
    )


def _clean_ax(ax):
    """Remove spines from an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_difference_cellmap(
    epoch_names,
    epoch_colors,
    x,
    y,
    df,
    num_cells,
    ylabel,
    save_name,
    scale_method,
    max_num_comparisons=20,
):
    """Plot a cellmap showing the differences in activity between epochs.

    :Parameters
        epoch_names (list): List of names corresponding to each epoch.
        epoch_colors (list): List of colors corresponding to each epoch.
        x (array-like): X coordinates of the cells.
        y (array-like): Y coordinates of the cells.
        df (DataFrame): DataFrame containing the activity data with an 'Epoch' column.
        num_cells (int): Number of cells.
        ylabel (str): Label for the y-axis.
        save_name (str): Path to save the resulting figure.
        scale_method (str): Method to scale the differences.
        max_num_comparisons: The maximum number of comparisons to render in a single figure.
        Options are 'none', 'normalize', 'standardize'.

    :Returns
        None
    """
    num_comparisons = ceil((len(epoch_names) * (len(epoch_names) - 1)) / 2)
    if num_comparisons > max_num_comparisons:
        logger.warning(
            f"Total number of comparisons to plot for {save_name} is {num_comparisons}. "
            "This is too large to render in a single preview. "
            f"Limiting number of comparisons to {max_num_comparisons}"
        )
        num_comparisons = max_num_comparisons

    fig, ax = plt.subplots(
        nrows=num_comparisons,
        ncols=2,
        figsize=(10, num_comparisons * 5),
        width_ratios=[4, 6],
    )
    df = df.reset_index()
    counter = 0
    for i in range(len(epoch_names)):
        for j in range(i, len(epoch_names) - 1):
            a_epoch = df[df["Epoch"] == epoch_names[i]]
            b_epoch = df[df["Epoch"] == epoch_names[j + 1]]
            diff = b_epoch["Activity"].values - a_epoch["Activity"].values

            if num_comparisons == 1:
                ax0 = ax[0]
                ax1 = ax[1]
            else:
                ax0 = ax[counter, 0]
                ax1 = ax[counter, 1]

            _clean_ax(ax0)
            _clean_ax(ax1)
            cool_warm, norm = plots.plot_shaded_hist(
                values=diff,
                ax=ax0,
                title=f"{epoch_names[j + 1]} - {epoch_names[i]}",
                xlabel="Difference",
            )
            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cool_warm, norm=norm)
            sm.set_array([])
            if scale_method == Rescale.NONE.value:
                label = "Difference"
            elif scale_method == Rescale.NORMALIZE.value:
                label = "Difference of Normalized Activity"
            elif (
                scale_method == Rescale.STANDARDIZE.value
                or scale_method == Rescale.STANDARDIZE_EPOCH.value
            ):
                label = "Difference of Standardized Activity"
            elif scale_method == Rescale.FRACTIONAL_CHANGE.value:
                label = "Difference of Fractional Change"
            else:
                logger.warning(f"Unknown scale method {scale_method}")
                label = "Difference"
            fig.colorbar(sm, ax=ax1, label=label)

            # Plot the footprints
            # Create a color map that goes from green to grey to blue

            max_val = np.max(np.abs(diff))

            normalized_diff = (diff + max_val) / (2 * max_val)
            colors = [cool_warm(val) for val in normalized_diff]

            plots.plot_footprints(
                x,
                y,
                figure=ax1,
                colors=colors,
            )
            ax1.set_title("Cellmap of differences", fontdict=TITLE_FONT)
            ax1.spines["bottom"].set_visible(False)
            ax1.spines["left"].set_visible(False)
            ax1.set_xticks([])
            ax1.set_yticks([])

            # invert y-axis to match IDPS/IDEAS orientation (y=0 at top, y=max at bottom)
            ax1.invert_yaxis()

            counter += 1

            if counter >= num_comparisons:
                break

        if counter >= num_comparisons:
            break

    fig.tight_layout()
    fig.savefig(
        save_name,
        dpi=300,
        transparent=True,
    )


def _plot_comparison(
    data, epoch_colors, aov, pairwise, metric, group_name, save_name
):
    """Plot a comparison of a single ANOVA of activity data across epochs for
    two groups using a boxplot.

    :Parameters
        data (pd.DataFrame): The data to plot, must
        contain columns 'Epoch', 'Activity', and 'Group'.
        g1_color (str): Color for the first group in the plot.
        g2_color (str): Color for the second group in the plot.
    :Returns
        None: The function saves the plot as a SVG file with the specified name.
    """
    # First plot trace data
    fig, ax = plt.subplots(figsize=(10, 6))
    _clean_ax(ax)
    sns.boxplot(
        ax=ax,
        data=data,
        x="Epoch",
        y=metric,
        hue="Epoch",
        palette=epoch_colors,
        **BOX_PLOT_ARGS,
    )

    _create_one_way_legend(aov, pairwise, ax)

    if metric == "Eventrate Activity":
        plt.ylabel("Event rate (Hz)", fontdict=LABEL_FONT)
    else:
        plt.ylabel(metric, fontdict=LABEL_FONT)
    plt.xlabel("Epoch", fontdict=LABEL_FONT)
    plt.title(f"{group_name} {metric} Comparison", **TITLE_FONT)
    fig.tight_layout()
    metric = metric.replace(" ", "_")
    fig.savefig(
        f"{save_name}_{metric}_comparison.svg",
        dpi=300,
        transparent=True,
    )


def _plot_correlation_comparison(
    data,
    epoch_colors,
    pos_aov,
    pos_pairwise,
    neg_aov,
    neg_pairwise,
    group_name,
    save_name,
):
    """Plot a comparison of average positive and negative correlations across
    epochs for two groups.

    :Parameters
        data (pd.DataFrame): DataFrame containing the data to plot. Must include columns 'Epoch',
                            'Average Positive Correlation',
                            'Average Negative Correlation', and 'Group'.
        name (str): The base name for the saved plot file.
        g1_color (str): Color for the first group in the plot.
        g2_color (str): Color for the second group in the plot.
    :Returns
        None: The function saves the plot as a SVG file with the specified name.
    """
    # Plot Data
    fig, ax = plt.subplots(nrows=2, figsize=(10, 8))
    _clean_ax(ax[0])
    _clean_ax(ax[1])
    ax[0].set_title(f"{group_name} Correlation Data", fontdict=TITLE_FONT)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["top"].set_visible(True)
    sns.boxplot(
        ax=ax[0],
        data=data,
        x="Epoch",
        y="Average Positive Correlation",
        hue="Epoch",
        palette=epoch_colors,
        **BOX_PLOT_ARGS,
    )
    ax[0].set_xticklabels([])
    ax[0].set_xlabel("")
    ax[0].set_ylim(0, np.max(data["Average Positive Correlation"]) * 1.1)

    _create_one_way_legend(pos_aov, pos_pairwise, ax[0])

    sns.boxplot(
        ax=ax[1],
        data=data,
        x="Epoch",
        y="Average Negative Correlation",
        hue="Epoch",
        palette=epoch_colors,
        **BOX_PLOT_ARGS,
    )
    ax[1].set_ylim(np.min(data["Average Negative Correlation"]) * 1.1, 0)

    _create_one_way_legend(neg_aov, neg_pairwise, ax[1])

    plt.xlabel("Epoch", fontdict=LABEL_FONT)
    fig.tight_layout()
    fig.savefig(
        f"{save_name}_correlation_data_comparison.svg",
        dpi=300,
        transparent=True,
    )


def _create_one_way_legend(aov, pairwise, ax):
    """Create a legend for a one-way ANOVA plot.

    :Parameters
        aov (DataFrame): A DataFrame containing the ANOVA results with at least a "p-unc" column.
        pairwise (DataFrame): A DataFrame containing pairwise comparison
        results with at least "A", "B", and "p-corr" columns.
        ax (matplotlib.axes.Axes): The matplotlib Axes object where the legend will be added.
    :Returns
        None
    """
    sources = ["ANOVA"]
    A = pairwise["A"].values
    B = pairwise["B"].values

    pair_source = [f"{a}x{b}" for a, b in zip(A, B)]
    sources.extend(pair_source)

    if "p-corr" in pairwise:
        pvals = [aov["p-unc"].values[0], *pairwise["p-corr"].values]
    else:
        pvals = [aov["p-unc"].values[0]]

    _format_labels(sources, pvals, ax)


def _format_labels(keys, p_values, ax, limit=0.001):
    """Format and set the legend labels for a given matplotlib axis based on
    provided keys and p-values.

    :Parameters
        keys (list of str): A list of source names to be used as labels.
        p_values (list of float): A list of p-values corresponding to each source.
        ax (matplotlib.axes.Axes): The matplotlib axis object where the legend will be set.
        limit (float): The p-value limit below which the label will be formatted as "p<0.001".
    :Returns
        None
    """
    new_label = []
    for source, p_val in zip(keys, p_values):
        if p_val < limit:
            new_label.append(f"{source}: p<0.001")
        else:
            new_label.append(f"{source}: p={p_val:.3f}")

    # get current handles and labels

    handles, labels = ax.get_legend_handles_labels()

    handles = handles + [plt.Line2D([0], [0], color="white", lw=0)] * len(
        new_label
    )
    labels = labels + new_label

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        handles=handles,
        labels=labels,
        frameon=False,
    )


def _create_pairwise_legend(pairwise, ax):
    """Create a legend for pairwise comparisons on a given axis."""
    # Skip if pairwise doesn't contain required columns
    if not all(col in pairwise.columns for col in ["Contrast", "A", "B"]):
        logger.warning(
            "Pairwise comparison data is missing required columns. Skipping legend creation."
        )
        return

    contrast = pairwise["Contrast"].values
    A = pairwise["A"].values
    B = pairwise["B"].values

    # Get the corrected p-values if available, otherwise use uncorrected p-values
    # this is for if only two groups are present, no correction is needed
    if "p-corr" in pairwise:
        p_vals = pairwise["p-corr"].values
    else:
        p_vals = pairwise["p-unc"].values

    sources = [f"{c}: {a}x{b}" for c, a, b in zip(contrast, A, B)]
    _format_labels(sources, p_vals, ax)


def _plot_mixed_comparisons(data, g1_color, g2_color, aov, pairwise, metric):
    """Plot mixed comparisons including ANOVA and pairwise comparisons.

    :Parameters
        data (pd.DataFrame): The data to be plotted, containing columns for 'Epoch',
                            the metric activity, and 'Group'.
        g1_color (str): Color for the first group in the plot.
        g2_color (str): Color for the second group in the plot.
        aov (pd.DataFrame): DataFrame containing ANOVA results with 'Source' and 'p-unc' columns.
        pairwise (pd.DataFrame): DataFrame containing pairwise comparison results with 'A', 'B',
                                and 'p-corr' columns.
        metric (str): The metric to be plotted (e.g., 'Accuracy', 'Loss').
    :Returns
        None: The function saves the plots as SVG files.
    """
    # AOV figure first
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{metric} Activity Comparison", fontdict=TITLE_FONT)
    _clean_ax(ax)
    sns.boxplot(
        ax=ax,
        data=data,
        x="Epoch",
        y=f"{metric} Activity",
        hue="Group",
        palette=[g1_color, g2_color],
        **BOX_PLOT_ARGS,
    )

    # Safety check for source column
    if "Source" not in aov.columns:
        logger.warning(
            "Source column not found in ANOVA results. Using default labels."
        )
        sources = ["Analysis"]
    else:
        sources = aov["Source"].values

    aov_p_vals = aov["p-unc"].values
    _format_labels(sources, aov_p_vals, ax)
    fig.tight_layout()
    fig.savefig(
        f"mixed_ANOVA_{metric}_comparison.svg",
        dpi=300,
        transparent=True,
    )

    # Safety check for Contrast column
    if "Contrast" not in pairwise.columns:
        logger.warning(
            "Contrast column not found in pairwise results. Skipping pairwise legend."
        )
    else:
        reduced_pairwise = pairwise[pairwise["Contrast"] == "Epoch * Group"]
        _create_pairwise_legend(reduced_pairwise, ax)

    fig.tight_layout()
    fig.savefig(
        f"mixed_pairwise_{metric}_comparison.svg",
        dpi=300,
        transparent=True,
    )


def _plot_mixed_corr_comparisons(
    combined_data,
    g1_color,
    g2_color,
    pos_aov,
    pos_pairwise,
    neg_aov,
    neg_pairwise,
):
    """Plot mixed correlation comparisons using boxplots for both positive and
    negative correlations.

    This function generates two figures:
    1. ANOVA comparison figure for positive and negative correlations.
    2. Pairwise comparison figure for positive and negative correlations.
    :Parameters
        combined_data (pd.DataFrame): DataFrame containing the combined data for plotting.
        g1_color (str): Color for the first group in the boxplot.
        g2_color (str): Color for the second group in the boxplot.
        pos_aov (pd.DataFrame): DataFrame containing ANOVA results for positive correlations.
        pos_pairwise (pd.DataFrame): DataFrame containing pairwise comparison
        results for positive correlations.
        neg_aov (pd.DataFrame): DataFrame containing ANOVA results for negative correlations.
        neg_pairwise (pd.DataFrame): DataFrame containing pairwise comparison
        results for negative correlations.
    :Returns
        None
    """
    # AOV figure first
    # Positive Correlation
    fig, ax = plt.subplots(nrows=2, figsize=(10, 12))
    _clean_ax(ax[0])
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[0].set_title("Correlation Data Comparison", fontdict=TITLE_FONT)
    sns.boxplot(
        ax=ax[0],
        data=combined_data,
        x="Epoch",
        y="Average Positive Correlation",
        hue="Group",
        palette=[g1_color, g2_color],
        **BOX_PLOT_ARGS,
    )

    # Safety check for source column in pos_aov
    if "Source" not in pos_aov.columns:
        logger.warning(
            "Source column not found in positive ANOVA results. Using default labels."
        )
        sources = ["Analysis"]
    else:
        sources = pos_aov["Source"].values

    aov_p_vals = pos_aov["p-unc"].values
    _format_labels(sources, aov_p_vals, ax[0])

    # Negative Correlation
    sns.boxplot(
        ax=ax[1],
        data=combined_data,
        x="Epoch",
        y="Average Negative Correlation",
        hue="Group",
        palette=[g1_color, g2_color],
        **BOX_PLOT_ARGS,
    )

    # Safety check for source column in neg_aov
    if "Source" not in neg_aov.columns:
        logger.warning(
            "Source column not found in negative ANOVA results. Using default labels."
        )
        sources = ["Analysis"]
    else:
        sources = neg_aov["Source"].values

    aov_p_vals = neg_aov["p-unc"].values
    _format_labels(sources, aov_p_vals, ax[1])

    fig.tight_layout()
    fig.savefig(
        "mixed_correlation_ANOVA_comparison.svg",
        dpi=300,
        transparent=True,
    )

    # Pairwise figure
    # Positive Correlation
    if "Contrast" in pos_pairwise.columns:
        reduced_pos_pairwise = pos_pairwise[
            pos_pairwise["Contrast"] == "Epoch * Group"
        ]
        _create_pairwise_legend(reduced_pos_pairwise, ax[0])

        # Negative Correlation
        reduced_neg_pairwise = neg_pairwise[
            neg_pairwise["Contrast"] == "Epoch * Group"
        ]
        _create_pairwise_legend(reduced_neg_pairwise, ax[1])
    else:
        logger.warning(
            "Contrast column not found in pairwise results. Skipping pairwise legend."
        )

    fig.tight_layout()
    fig.savefig(
        "mixed_correlation_pairwise_comparison.svg",
        dpi=300,
        transparent=True,
    )


def _plot_state_time(behavior, column_name, state_names, state_colors, period):
    """Plot the time spent in each state and the fraction of time spent in each state.

    This function creates a two-panel figure:
    1. A bar plot showing the absolute time spent in each state
    2. A stacked bar plot showing the fraction of total time spent in each state

    :Parameters
        behavior (pd.DataFrame): DataFrame containing the behavioral data with state information
        column_name (str): Name of the column in behavior DataFrame that contains state information
        state_names (list of str): List of state names to plot
        state_colors (list of str): List of colors corresponding to each state for plotting
        period (float): Time period between frames in seconds, used to convert frame counts to time

    :Returns
        None: The function saves the plot as 'time_in_state_preview.svg'
    """
    # Set consistent font size
    plt.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(6, 5), width_ratios=[5, 1]
    )
    frames_in_state = (
        behavior[column_name].value_counts().reindex(state_names, fill_value=0)
    )
    # convert to seconds
    frames_in_state = frames_in_state * period
    total_frames = len(behavior) * period
    frames_in_state.plot(kind="bar", ax=ax[0], color=state_colors)
    ax[0].set_title("Time spent in each state", fontdict=TITLE_FONT)
    ax[0].set_ylabel("Time (s)", fontdict=LABEL_FONT)
    ax[0].set_xlabel("")
    # Capitalize x-axis tick labels for consistency
    ax[0].set_xticklabels(
        [label.get_text().capitalize() for label in ax[0].get_xticklabels()],
        rotation=0,
    )
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    # plot fraction of time spent in each state
    fraction_in_state = {
        state: frames_in_state[state] / total_frames for state in state_names
    }

    colors = {state: color for state, color in zip(state_names, state_colors)}
    # plot fractions of neurons in different states
    plot_neuron_fractions(
        fractions=fraction_in_state,
        label="states",
        bottom=0,
        colors=colors,
        xlabel="",
        ylabel="Fraction of total time",
        ax=ax[1],
        plot_legend=False,
    )
    # Adjust y-axis to show only the range needed and make it more evident
    max_fraction = sum(fraction_in_state.values())
    ax[1].set_ylim([0, max(max_fraction * 1.1, 0.1)])
    # Make y-axis more evident by showing the left spine
    ax[1].spines["left"].set_visible(True)

    fig.tight_layout()
    fig.savefig("time_in_state_preview.svg", dpi=300, transparent=True)


def _plot_traces(
    *,
    traces: np.array,
    behavior: pd.DataFrame,
    data: dict,
    column_name: str,
    filename: str,
    spacing: float = 5.01,
    state_colors: List[str],
    state_names: List[str],
    period: float = 1.0,
    boundaries: Optional[List[float]] = None,
) -> None:
    """Plot all traces in given input matrix, and colors them by behavior
    :param traces: neural activity of individual cells
                   (2D array <num_timepoints x num_cells>)
    :param behavior: binary 2D array indicating the presence/
    absence of behaviors at each timepoint
                     (2D array <num_timepoints x num_behaviors>)
    :param plot_filename: path to the plot file <str>
    :param figsize: dimensions of the figure <int, int>
    :param spacing: spacing factor use to shift the traces
    relative to one another on the y-axis <float>
    :param period: Time period between frames in seconds
    """
    # Set consistent font size
    plt.rcParams.update({"font.size": 12})

    # remove minimum for traces.
    traces = np.copy(traces)
    num_neurons = traces.shape[1]
    traces -= np.nanmin(traces, axis=0)

    # Create time axis in seconds
    time_axis = np.arange(traces.shape[0]) * period

    # Traces figure
    trace_fig, ax_traces = plt.subplots(
        nrows=2, ncols=1, figsize=(15, 10), height_ratios=[1, 8]
    )

    locations = [np.zeros(traces.shape[0]) for _ in state_names]

    for idx, state in enumerate(state_names):
        locations[idx][behavior[column_name] == state] = 1

    # Plot the first 50 traces
    num_to_plot = min(50, num_neurons)

    for y in range(num_to_plot):
        x = y * spacing + scipy.stats.zscore(traces[:, y], nan_policy="omit")
        ax_traces[1].plot(time_axis, x, linewidth=1, color="black")

    # Plot vspan for all states
    for idx, _ in enumerate(state_names):
        starts = np.where(np.diff(locations[idx]) == 1)[0]
        ends = np.where(np.diff(locations[idx]) == -1)[0]
        # Adjust lengths if state starts at frame 0 or ends at last frame
        if locations[idx][0] == 1:
            starts = np.insert(starts, 0, -1)
        if locations[idx][-1] == 1:
            ends = np.append(ends, len(locations[idx]) - 1)

        for start, end in zip(starts, ends):
            # Increment start by 1 for correct axvspan range
            plot_start = (start + 1) * period
            plot_end = end * period
            ax_traces[0].axvspan(
                plot_start,
                plot_end,
                color=state_colors[idx],
                alpha=0.4,
                label=(
                    state_names[idx] if start == starts[0] else ""
                ),  # Label only first instance
            )
            ax_traces[1].axvspan(
                plot_start,
                plot_end,
                color=state_colors[idx],
                alpha=0.4,
                label=(
                    state_names[idx] if start == starts[0] else ""
                ),  # Label only first instance
            )

    # Set x-limits in seconds
    ax_traces[1].set_xlim([0, traces.shape[0] * period])
    ax_traces[1].set_ylim([-5, 15 + num_to_plot * spacing])

    ax_traces[1].set_yticks([])
    ax_traces[1].set_xlabel("Time (s)", fontdict=LABEL_FONT)
    ax_traces[1].set_ylabel("Cell index", fontdict=LABEL_FONT)
    ax_traces[1].spines.right.set_visible(False)
    ax_traces[1].spines.top.set_visible(False)

    ax_traces[1].set_title(
        f"Traces of the first {num_to_plot} cells", fontdict=TITLE_FONT
    )

    # Calculate mean activity of population and plot
    mean_activity = np.nanmean(traces, axis=1)
    ax_traces[0].plot(time_axis, mean_activity, color="black")
    ax_traces[0].set_xlim([0, traces.shape[0] * period])
    ax_traces[0].set_title("Population Average Activity", fontdict=TITLE_FONT)
    ax_traces[0].set_ylabel("Mean Activity", fontdict=LABEL_FONT)
    ax_traces[0].spines.right.set_visible(False)
    ax_traces[0].spines.top.set_visible(False)
    ax_traces[0].spines.bottom.set_visible(False)
    ax_traces[0].set_xticks([])

    # Add cellset boundaries as vertical dashed lines if several cellsets
    if boundaries is not None and len(boundaries) > 2:
        for idx, boundary in enumerate(boundaries[1:-1]):
            if idx == 0:
                label = "cellset boundary"
            else:
                label = None
            ax_traces[0].axvline(
                boundary, linestyle="--", color="k", label=label
            )
            ax_traces[1].axvline(
                boundary, linestyle="--", color="k", label=label
            )

    # Create manual handles for the legend
    handles = []
    for state, color in zip(state_names, state_colors):
        handles.append(plt.Line2D([0], [0], color=color, lw=4, label=state))

    # Add legend to the top panel with padding for save function
    ax_traces[0].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    save_optimized_svg(trace_fig, filename, max_size_mb=10, pad_inches=0.3)
    plt.close(trace_fig)


def plot_neuron_fractions(
    fractions,
    label,
    bottom,
    colors,
    xlabel,
    ax,
    plot_legend=True,
    ylabel="Fraction of cells significantly modulated",
):
    """Plot the fractions of neurons for different states.
    :Parameters
    fractions (dict): A dictionary containing the fractions of neurons for different states.
    data (dict): A dictionary containing the data for each state.
    bottom (float): The bottom position for the bars.
    colors (dict): A dictionary mapping each state to its corresponding color.
    xlabel (str): The label for the x-axis.
    ax (matplotlib.axes.Axes): The axes object to plot the fractions on.
    :Returns
    None
    """
    for state, value in fractions.items():
        ax.bar(
            label,
            value,
            width=0.15,
            label=state,
            bottom=bottom,
            color=colors[state],
        )
        bottom += value
    if plot_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper right")

    ax.set_ylabel(ylabel, fontdict=LABEL_FONT)
    ax.set_xlabel(xlabel, fontdict=LABEL_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _plot_raster(
    *,
    events: np.array,
    event_timeseries: np.array,
    behavior: pd.DataFrame,
    column_name: str,
    period: float,
    state_colors: List[str],
    state_names: List[str],
    filename: str,
    boundaries: Optional[List[float]] = None,
) -> None:
    """Plot all traces in given input matrix, and colors them by behavior
    :param traces: neural activity of individual cells
                   (2D array <num_timepoints x num_cells>)
    :param behavior: binary 2D array indicating the presence/
    absence of behaviors at each timepoint
                     (2D array <num_timepoints x num_behaviors>)
    :param plot_filename: path to the plot file <str>
    :param figsize: dimensions of the figure <int, int>
    :param spacing: spacing factor use to shift the traces
    relative to one another on the y-axis <float>
    """
    # Set consistent font size
    plt.rcParams.update({"font.size": 12})

    # Raster figure
    trace_fig, ax = plt.subplots(
        nrows=2, ncols=1, figsize=(15, 10), height_ratios=[1, 8]
    )
    ax[1].eventplot(
        events, color="black", linelengths=0.5, linewidths=0.8, alpha=0.5
    )

    ax[1].set_ylabel("Cell index", fontdict=LABEL_FONT)
    ax[1].set_xlabel("Time (s)", fontdict=LABEL_FONT)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    locations = [np.zeros(event_timeseries.shape[0]) for _ in state_names]

    for idx, state in enumerate(state_names):
        locations[idx][behavior[column_name] == state] = 1

    # Plot vspan for all states
    for idx, _ in enumerate(state_names):
        starts = np.where(np.diff(locations[idx]) == 1)[0]
        ends = np.where(np.diff(locations[idx]) == -1)[0]

        # Handle edge cases for states starting at frame 0 or ending at last frame
        if locations[idx][0] == 1:
            starts = np.insert(starts, 0, -1)
        if locations[idx][-1] == 1:
            ends = np.append(ends, len(locations[idx]) - 1)

        for start, end in zip(starts, ends):
            # Use consistent coordinate calculation with _plot_traces
            plot_start = (start + 1) * period
            plot_end = end * period

            ax[0].axvspan(
                plot_start,
                plot_end,
                color=state_colors[idx],
                alpha=0.4,
                label=(
                    state_names[idx] if start == starts[0] else ""
                ),  # Label only first instance
            )
            ax[1].axvspan(
                plot_start,
                plot_end,
                color=state_colors[idx],
                alpha=0.4,
                label=(
                    state_names[idx] if start == starts[0] else ""
                ),  # Label only first instance
            )
    ax[1].set_ylim([0, len(events)])
    ax[1].set_xlim([0, event_timeseries.shape[0] * period])
    ax[1].spines.right.set_visible(False)
    ax[1].spines.top.set_visible(False)

    ax[1].set_title("Raster plot of events", fontdict=TITLE_FONT)

    # Calculate mean activity of population and plot
    mean_activity = np.nanmean(event_timeseries, axis=1)
    # Smooth the mean activity with a rolling window of 1 second
    window = int(1 / period)
    mean_activity = np.convolve(
        mean_activity, np.ones(window) / window, mode="same"
    )
    # Create time axis for mean activity
    time_axis = np.arange(event_timeseries.shape[0]) * period
    ax[0].plot(time_axis, mean_activity, color="black")
    ax[0].set_xlim([0, event_timeseries.shape[0] * period])
    ax[0].set_title("Population Average Activity", fontdict=TITLE_FONT)
    ax[0].set_ylabel("Mean Event Rate (Hz)", fontdict=LABEL_FONT)
    ax[0].spines.right.set_visible(False)
    ax[0].spines.top.set_visible(False)
    ax[0].spines.bottom.set_visible(False)
    ax[0].set_xticks([])

    # Add cellset boundaries as vertical dashed lines if several cellsets
    if boundaries is not None and len(boundaries) > 2:
        for idx, boundary in enumerate(boundaries[1:-1]):
            if idx == 0:
                label = "cellset boundary"
            else:
                label = None
            ax[0].axvline(boundary, linestyle="--", color="k", label=label)
            ax[1].axvline(boundary, linestyle="--", color="k", label=label)

    # Create manual handles for the legend (consistent with _plot_traces)
    handles = []
    for state, color in zip(state_names, state_colors):
        handles.append(plt.Line2D([0], [0], color=color, lw=4, label=state))

    # Add legend to the top panel with better positioning
    ax[0].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )

    save_optimized_svg(trace_fig, filename, max_size_mb=10, pad_inches=0.3)
    plt.close(trace_fig)


def _plot_population_average(
    *,
    data: dict,
    filename: str,
    state_colors: List[str],
    ylabel="Mean Activity",
    xlabel="State",
) -> None:
    """Plot the population average activity as a box plot"""
    # Set consistent font size
    plt.rcParams.update({"font.size": 12})

    # Get states and figure dimensions
    states = list(data.keys())
    fig_width = max(len(states), 4)

    # Create a figure with no axes initially
    fig = plt.figure(figsize=(fig_width, 6.5))

    # Define the plot area position: [left, bottom, width, height]
    # Leave extra room on left for y-axis label and bottom for x-tick labels
    ax_box = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    num_neurons = len(data[states[0]]["mean_activity"])

    # Create list of dataframes to concatenate
    dfs_to_concat = []

    for state in states:
        dfs_to_concat.append(
            pd.DataFrame(
                {
                    "Epoch": [state] * num_neurons,
                    "Cell": np.arange(num_neurons),
                    "Activity": data[state]["mean_activity"],
                }
            )
        )

    # Concatenate non-empty dataframes with consistent dtypes
    df = pd.concat(dfs_to_concat, ignore_index=True)

    # make sure there are no duplicate indices
    df.reset_index(drop=True, inplace=True)

    plots.plot_paired_lines(
        ax=ax_box, df=df, category_names=states, num_values=num_neurons
    )

    sns.boxplot(
        data=df,
        x="Epoch",
        y="Activity",
        ax=ax_box,
        hue="Epoch",
        palette=state_colors,
        linewidth=3,
        fliersize=0,
        zorder=2,
    )
    sns.stripplot(
        data=df,
        x="Epoch",
        y="Activity",
        ax=ax_box,
        color="black",
        size=3,
        alpha=0.2,
        jitter=0.02,
        zorder=1,
        legend=False,
    )
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.set_ylabel(ylabel, fontdict=LABEL_FONT, labelpad=10)
    ax_box.set_xlabel(xlabel, fontdict=LABEL_FONT, labelpad=10)

    # Fix ticklabels warning by setting ticks explicitly before labels
    # Get current ticklabels
    labels = [item.get_text() for item in ax_box.get_xticklabels()]
    # Set ticks at proper positions
    ax_box.set_xticks(range(len(labels)))
    # Then set ticklabels with rotation - add extra padding for rotated labels
    ax_box.set_xticklabels(labels, rotation=90)
    ax_box.tick_params(axis="x", pad=10)

    fig.savefig(filename, dpi=300, transparent=True, bbox_inches="tight")

    return df


@beartype
def plot_modulated_neuron_footprints(
    data: dict,
    x,
    y,
    filename: str,
    method: str = Comp.NOT_STATE.value,
    baseline_state: Optional[str] = None,
    up_modulation_color="green",
    down_modulation_color="blue",
    non_modulation_color="gray",
    plot_type: str = "map",
) -> None:
    """Make a plot of footprints of all cells, and color
    them by whether they are up or down modulated in different
    contexts.
    data: dictionary with modulation data, output of
    _make_ modulation_data
    :param x: vector containing the x position of each cell
              (1D array <num_cells>)
    :param y: vector containing the y position of each cell
              (1D array <num_cells>)
    :param plot_filename: path to the plot file
    """
    # Set consistent font size
    plt.rcParams.update({"font.size": 12})

    num_neurons = len(x)

    if method == Comp.PAIRWISE.value:
        num_comparisons = int(len(data) * (len(data) - 1) / 2)
    elif method == Comp.BASELINE.value:
        num_comparisons = len(data) - 1
    else:
        num_comparisons = len(data)
    fig, axs = plt.subplots(
        nrows=num_comparisons,
        ncols=3,
        figsize=(14, 8 * num_comparisons),
        width_ratios=[1, 7, 0.3],
    )
    axs = axs.flatten()

    counter = 0

    if method == Comp.PAIRWISE.value:
        for s1 in data.keys():
            first_state = data[s1]
            # Get the comparisons for this state
            comparisons = [
                c for c in first_state.keys() if c != "mean_activity"
            ]
            for s2 in comparisons:
                counter = plot_comparison_row(
                    fig,
                    axs,
                    plot_type,
                    first_state[s2],
                    f"{s1} vs {s2}",
                    counter,
                    x,
                    y,
                    num_neurons,
                    up_modulation_color,
                    down_modulation_color,
                    non_modulation_color,
                    TITLE_FONT,
                    LABEL_FONT,
                )
                if counter == 2:
                    axs[counter].legend(
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        frameon=False,
                    )
    else:
        for case in data.keys():
            if method == Comp.NOT_STATE.value:
                title = f"{case} vs not during {case}"
            elif method == Comp.BASELINE.value:
                if case == baseline_state:
                    continue
                title = f"{case} vs {baseline_state}"
            elif method == Comp.NOT_DEFINED.value:
                # Skip the not_defined state itself when using NOT_DEFINED method
                if case == "not_defined":
                    continue
                title = f"{case} vs not defined"
            else:
                raise IdeasError("Invalid method")
            counter = plot_comparison_row(
                fig,
                axs,
                plot_type,
                data[case],
                title,
                counter,
                x,
                y,
                num_neurons,
                up_modulation_color,
                down_modulation_color,
                non_modulation_color,
                TITLE_FONT,
                LABEL_FONT,
            )
            if counter == 2:
                axs[counter].legend(
                    loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
                )

    fig.tight_layout()

    # save figure to disk
    fig.savefig(filename, dpi=300, transparent=True)


def plot_comparison_row(
    fig,
    axs,
    plot_type,
    data,
    case,
    counter,
    x,
    y,
    num_neurons,
    up_modulation_color,
    down_modulation_color,
    non_modulation_color,
    TITLE_FONT,
    LABEL_FONT,
):
    """Plot a comparison row of modulation levels and significance maps for a given case.
    :Parameters
    fig (matplotlib.figure.Figure): The figure object to plot on.
    axs (numpy.ndarray): Array of axes objects to plot on.
    data (dict): Dictionary containing modulation scores and neuron modulation data.
    case (str): The case name to be plotted.
    counter (int): The current index of the subplot.
    x (numpy.ndarray): X coordinates of neurons.
    y (numpy.ndarray): Y coordinates of neurons.
    num_neurons (int): Total number of neurons.
    up_modulation_color (str): Color for up-modulated neurons.
    down_modulation_color (str): Color for down-modulated neurons.
    non_modulation_color (str): Color for non-modulated neurons.
    :Returns
    int: Updated counter after plotting.
    """
    fractions = dict()
    # First plot modulation levels
    norm = plt.Normalize(-1, 1)

    cmap, _ = plots.plot_shaded_hist(
        values=data["modulation_scores"],
        ax=axs[counter + 1],
        title=f"Modulation in {case}",
        palette=[
            down_modulation_color,
            non_modulation_color,
            up_modulation_color,
        ],
        xlabel="Modulation Score",
        norm=norm,
    )

    # plot raw modulation level footprints
    max_val = np.max(
        np.abs(
            data["modulation_scores"],
        )
    )

    normalized_diff = (data["modulation_scores"] + max_val) / (2 * max_val)
    colors = [cmap(val) for val in normalized_diff]

    edge_colors = [non_modulation_color for _ in x]
    for neuron in data["up_modulated_neurons"]:
        edge_colors[neuron] = up_modulation_color

    for neuron in data["down_modulated_neurons"]:
        edge_colors[neuron] = down_modulation_color

    if plot_type == "map":
        # clear axis
        axs[counter + 1].cla()

        plots.plot_footprints(
            x,
            y,
            figure=axs[counter + 1],
            colors=colors,
            edge_colors=edge_colors,
        )
        axs[counter + 1].set_aspect("equal", "box")
        axs[counter + 1].set_ylabel(case, fontdict=LABEL_FONT)
        axs[counter + 1].spines["top"].set_visible(False)
        axs[counter + 1].spines["right"].set_visible(False)
        axs[counter + 1].spines["left"].set_visible(False)
        axs[counter + 1].spines["bottom"].set_visible(False)
        axs[counter + 1].set_xticks([])
        axs[counter + 1].set_yticks([])

        axs[counter + 1].axis("off")

        axs[counter + 1].set_aspect("equal", "box")

        n_up = len(data["up_modulated_neurons"])
        n_down = len(data["down_modulated_neurons"])

        fractions["Down-modulated"] = n_down / num_neurons
        fractions["Up-modulated"] = n_up / num_neurons

        colors = dict()
        colors["Down-modulated"] = down_modulation_color
        colors["Up-modulated"] = up_modulation_color

        plot_neuron_fractions(
            fractions=fractions,
            label=case,
            bottom=0,
            colors=colors,
            xlabel="",
            ax=axs[counter],
            plot_legend=False,
        )
        # Adjust y-axis to show only the range needed and make it more evident
        max_fraction = sum(fractions.values())
        axs[counter].set_ylim([0, max(max_fraction * 1.1, 0.1)])
        axs[counter].set_ylabel(
            "Fraction of cells significantly modulated", fontsize=12
        )
        # Make y-axis more evident by showing the left spine
        axs[counter].spines["left"].set_visible(True)

    # add colorbar

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axs[counter + 1],
        cax=axs[counter + 2],
        label="Modulation Score",
    )
    cbar.set_label("Modulation Score", fontsize=12)

    axs[counter + 1].set_title(f"Modulation in {case}", fontdict=TITLE_FONT)
    # plot fractions of neurons in different states
    # remove fraction
    if plot_type != "map":
        axs[counter].cla()
        axs[counter].axis("off")
        axs[counter + 1].set_xlabel("Modulation Score", fontdict=LABEL_FONT)
        axs[counter + 1].set_xlim([-1, 1])
    counter += 3

    return counter

#### COMPARE PERI-EVENT ACROSS EPOCHS



def plot_population_mean_event_window_across_epochs(
    x_values,
    x_limits,
    data,
    event_type,
    output_filename,
    plot_limits=None,
    epoch_data=None,
):
    """Plot the population mean event window data for each epoch on a single figure.

    :param x_values: values along the x-axis
    :param x_limits: limits of the x-axis to use for plots
    :param data: population data for each epoch in a dictionary structure
    :param event_type: type of event for which analysis was performed
    :param output_file_basename: path to the output image without a file extension
    :param color: color to use for the population activity line
    :param plot_limits: y-axis range (z-score) applied to the event-aligned population activity plot
    :param epoch_data: dictionary containing data relevant to each epoch (keys are the epoch names)
    """
    fig, ax = plt.subplots()

    epoch_names = list(epoch_data.keys())
    for epoch_name in epoch_names:
        # extract epoch data
        population_mean = data[epoch_name]["mean"]
        population_sem = data[epoch_name]["sem"]
        color = epoch_data[epoch_name]["color"]

        # plot event-aligned population data
        if not all(np.isnan(population_sem)):
            ax.plot(
                x_values,
                population_mean,
                label=f"{epoch_name} (mean ± sem)",
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
            # When there is only one event, only mean is plotted because sem is
            # non-sensical in this case
            ax.plot(
                x_values,
                population_mean,
                label=f"{epoch_name} (mean)",
                color=color,
            )

        # add vertical line at x=0
        # this line represents the reference event time around which we want to
        # observe patterns
        ax.axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

    # add labels
    plot_title = "Event-Aligned Population Activity"

    if event_type:
        subtitle = "\n(event type: {0})".format(event_type)
        plot_title += subtitle

    ax.set_title(plot_title, fontsize=config.PLOT_TITLE_FONT_SIZE)

    ax.set_xlabel(
        "Time from Event (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )
    ax.set_ylabel(
        "Neural Activity (z-score)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )

    # adjust plot limits
    if plot_limits not in [None, "auto"]:
        y_limits = [float(lim) for lim in plot_limits.split(",")]
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


def plot_event_aligned_activity_by_modulation_group_comparing_epochs(
    x,
    x_limits,
    data,
    event_type,
    epoch_data=None,
    plot_limits=None,
    output_dir=None,
    group_name=None,
):
    """Plot event-aligned activity by activity-modulation group in each epoch.

    - Mean activity plotted for a given group when the group contains at least 1 cell.
    - Standard error of the mean activity plotted when the group contains at least 2 cells.
      - For 1 cell, uncorrected sample variance is 0 and corrected sample variance is
      undefined, so nothing to plot either way.

    :param x: values to use along the x-axis of the plot
    :param x_limits: limits of the x-axis to use for plots
    :param data: dictionary containing all single cell and group data
    :param event_type: type of event for which analysis was performed
    :param epoch_data: dictionary containing data relevant to each epoch (keys are the
                       epoch names)
    :param plot_limits: y-axis range (z-score) applied to the event-aligned activity
                        by modulation plot
    :param output_dir: path to the output directory
    :param group_name: name of the group (applicable when combining data within a group)
    """
    if output_dir is None:
        output_dir = os.getcwd()

    epoch_names = list(epoch_data.keys())

    output_files = []
    for group, group_title in [
        ("up_modulated", "Up-Modulated"),
        ("down_modulated", "Down-Modulated"),
        ("non_modulated", "Non-Modulated"),
    ]:
        # initialize plot
        fig, ax = plt.subplots()

        total_num_cells_plotted = 0
        for epoch_name in epoch_names:
            # collect number of cells and traces
            num_cells = data[epoch_name][group]["num_cells"]
            mean_activity = data[epoch_name][group]["mean"]
            sem_activity = data[epoch_name][group]["sem"]
            total_num_cells_plotted += num_cells

            # plot data
            epoch_color = epoch_data[epoch_name]["color"]
            if num_cells > 0:
                ax.plot(
                    x,
                    mean_activity,
                    color=epoch_color,
                    label=f"{epoch_name} (n={num_cells})",
                )
            if num_cells > 1:
                ax.fill_between(
                    x,
                    mean_activity + sem_activity,
                    mean_activity - sem_activity,
                    alpha=config.PLOT_SHADING_ALPHA,
                    color=epoch_color,
                )

        if total_num_cells_plotted == 0:
            logger.warning(
                f"No {group.replace('_', '-')} cells found in any epoch. "
                f"The preview comparing the activity of {group.replace('_', '-')} "
                f"cells across epochs will not be generated."
            )
            continue

        # plot vertical line at x=0, the reference event time
        ax.axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

        # set x limits
        ax.set_xlim(x_limits)

        # add labels
        plot_title = f"Event-Aligned Activity of {group_title} Cells"
        if event_type:
            plot_title += "\n(event type: {0})".format(event_type)
        ax.set_title(plot_title, fontsize=config.PLOT_TITLE_FONT_SIZE)

        ax.set_xlabel(
            "Time from Event (seconds)",
            fontsize=config.PLOT_LABEL_FONT_SIZE,
        )
        ax.set_ylabel(
            "Neural Activity (z-score)",
            fontsize=config.PLOT_LABEL_FONT_SIZE,
        )

        # adjust plot limits
        if plot_limits not in [None, "auto"]:
            y_limits = [float(lim) for lim in plot_limits.split(",")]
            ax.set_ylim(y_limits)

        ax.margins(x=0)

        # add legend to figure
        ax.legend()

        # save figure to disk
        output_file_basename = (
            f"event_aligned_activity_{group}.svg"
            if group_name is None
            else f"event_aligned_activity_{group}_{group_name}.svg"
        )
        output_filename = os.path.join(
            output_dir,
            output_file_basename.replace(" ", ""),
        )

        fig.tight_layout()
        fig.savefig(
            output_filename,
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)

        output_files.append((group_title, output_filename))

    return output_files


def plot_population_activity(
    mean_trace,
    event_indices,
    epoch_data,
    output_filename,
    plot_limits=None,
    boundaries=None,
):
    """Plot mean population activity over the duration of the recording.
    - Highlight epoch periods.
    - Overlay events onto the timeline.

    :param mean_trace: mean peri-event trace
    :param event_indices: indices of the individual events
    :param epoch_data: dictionary containing data relevant to each epoch (keys are the epoch names)
    :param output_filename: path to the output file
    :param plot_limits: y-axis limits
    :param boundaries: boundaries between the input cell sets
    """
    # plot population activity
    period = epoch_data[list(epoch_data.keys())[0]]["recording_period"]
    fig, ax = plt.subplots()
    time = np.arange(len(mean_trace)) * period

    ax.plot(
        time,
        mean_trace,
        color="black",
        linewidth=2,
    )

    # highlight epoch periods
    for epoch_name, epoch_data_item in epoch_data.items():
        epoch_color = epoch_data_item["color"]
        start_index, end_index = epoch_data_item["indices"]
        ax.axvspan(
            start_index * period,
            end_index * period,
            color=epoch_color,
            alpha=0.15,
            label=epoch_name,
        )

    # overlay events
    ymin, ymax = ax.get_ylim()
    ax.scatter(
        x=np.array(event_indices) * period,
        y=[ymin] * len(event_indices),
        c="gray",
        s=5,
        label="events",
    )

    if boundaries is not None:
        # draw vertical lines to delineate the input cell sets
        if len(boundaries) > 2:
            for idx, boundary in enumerate(boundaries[1:-1]):
                if idx == 0:
                    label = "cellset boundary"
                else:
                    label = None
                ax.axvline(boundary, linestyle="--", color="k", label=label)

    # adjust figure settings
    ax.margins(x=0)
    ax.legend()
    ax.set_title(
        "Mean Population Activity", fontsize=config.PLOT_TITLE_FONT_SIZE
    )
    ax.set_xlabel("Time (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE)
    ax.set_ylabel(
        "Neural Activity (z-score)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )

    if plot_limits not in [None, "auto"]:
        y_limits = [float(lim) for lim in plot_limits.split(",")]
        ax.set_ylim(y_limits)

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


def plot_number_of_events_per_epoch(epoch_data, output_filename):
    """Plot the number of epochs per epoch as a bar chart.

    :param epoch_data: dictionary containing data relevant to each epoch (keys are the epoch names)
    :param output_filename: path to the output file
    """
    fig, ax = plt.subplots()

    epoch_names = epoch_data.keys()
    colors = []
    num_events = []
    for epoch_name, epoch_data_item in epoch_data.items():
        colors.append(epoch_data_item["color"])
        num_events_in_epoch = len(epoch_data_item["event_indices"])
        num_events.append(num_events_in_epoch)

    # plot data
    plt.bar(x=epoch_names, height=num_events, color=colors)

    # set figure labels
    plt.ylabel("Number of Events")
    plt.title("Number of Events Per Epoch")

    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_post_minus_pre_per_epoch_bar_chart(data, epoch_data, output_filename):
    """Plot the mean+-sem post-pre activity per epoch as a bar chart.

    :param epoch_data: dictionary containing data relevant to each epoch (keys are the epoch names)
    :param output_filename: path to the output file
    """
    fig, ax = plt.subplots()

    epoch_names = epoch_data.keys()
    colors = []
    post_minus_pre_means = []
    post_minus_pre_sems = []
    for epoch_name, epoch_data_item in epoch_data.items():
        colors.append(epoch_data_item["color"])

        post_minus_pre_dist = data[epoch_name]["cell"]["true_mean_post-pre"]
        post_mins_pre_mean = np.nanmean(post_minus_pre_dist)
        post_mins_pre_sem = stats.sem(post_minus_pre_dist, nan_policy="omit")

        post_minus_pre_means.append(post_mins_pre_mean)
        post_minus_pre_sems.append(post_mins_pre_sem)

    # plot data
    plt.bar(
        x=epoch_names,
        height=post_minus_pre_means,
        yerr=post_minus_pre_sems,
        color=colors,
        capsize=3,
    )
    # draw horizontal line at y=0
    ax.axhline(0, linewidth=1, color="black")

    # set figure labels
    plt.ylabel("Post-Pre Activity")
    plt.title("Mean Post-Pre Activity Per Epoch")
    fig.tight_layout()

    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_number_of_modulated_cells_per_epoch(
    data, epoch_data, output_filename
):
    """Plot the number of modulated cells per epoch as a bar chart.

    :param data: dictionary containing population data and individual cell data
    :param epoch_data: dictionary containing data relevant to each epoch (keys are the epoch names)
    :param output_filename: path to the output file
    """
    fig, ax = plt.subplots(layout="constrained")

    data_to_plot = {}
    for epoch_name, epoch_data_item in epoch_data.items():
        data_to_plot[epoch_name] = {
            "measurements": [
                data[epoch_name]["up_modulated"]["num_cells"],
                data[epoch_name]["down_modulated"]["num_cells"],
                data[epoch_name]["non_modulated"]["num_cells"],
            ],
            "color": epoch_data_item["color"],
        }

    x = np.arange(3)  # label locations (up/down/non)
    width = 0.25  # width of the bars
    multiplier = 0

    for label, item in data_to_plot.items():
        measurements = item["measurements"]
        color = item["color"]
        offset = width * multiplier
        rects = ax.bar(
            x + offset, measurements, width, label=label, color=color
        )
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # adjust figure settings
    ax.set_xticks(
        x + width, ["Up-Modulated", "Down-Modulated", "Non-Modulated"]
    )
    plt.ylabel("Number of Cells")
    plt.title("Number of Modulated Cells Per Epoch")
    ax.legend()

    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def plot_post_minus_pre_activity_differences_with_cell_map(
    cell_set_files, data, epoch_data, output_dir, group_name=None
):
    """Plot the post-pre activity differences between a pair of epochs along with a
    color-coded cell map.
    :param cell_set_files: list of paths to the isxd cell set files
    :param data: dictionary containing population data and individual cell data
    :param epoch_data: dictionary containing data relevant to each epoch (keys are the epoch names)
    :param output_dir: path to the output directory
    :param group_name: name of the group
    """
    epoch_names = list(epoch_data.keys())

    # format data into a dataframe used for plotting
    dfs = []
    for epoch_name in epoch_names:
        post_minus_pre_dist = data["single_cell"][epoch_name]["cell"][
            "true_mean_post-pre"
        ]
        num_cells = len(post_minus_pre_dist)

        df = pd.DataFrame(
            {
                "Epoch": [epoch_name] * num_cells,
                "Cell": np.arange(num_cells),
                "Activity": post_minus_pre_dist,
            }
        )
        dfs.append(df)

    df = pd.concat(dfs)

    if cell_set_files in [None, []]:
        logger.info(
            "No cell set files available, skipping the post-pre activity "
            "differences histogram and cell map preview"
        )
        post_minus_pre_activity_differences_preview_files = None
    else:
        (
            num_accepted_cells,
            num_undecided_cells,
            num_rejected_cells,
        ) = get_num_cells_by_status(cell_set_files[0])
        status = io.cell_set_to_status(cell_set_files)

        # get contours for non-rejected neurons
        try:
            x, y = io.cell_set_to_contours(cell_set_files)
        except Exception as err:
            raise IdeasError(
                f"Failed to extract cell contours from cell set files: {err}",
            )

        if num_accepted_cells > 0:
            cell_status_filter = "accepted"
        elif num_undecided_cells > 0:
            cell_status_filter = "undecided"

        # x and y coordinates of the cells
        x = [cx for cx, stat in zip(x, status) if stat == cell_status_filter]
        y = [cx for cx, stat in zip(y, status) if stat == cell_status_filter]
        num_cells = len(x)

        # plot difference distribution and cell map
        # post_minus_pre_activity_differences_preview_files = (
        _plot_difference_cellmap2(
            epoch_names=list(epoch_data.keys()),
            x=x,
            y=y,
            df=df,
            output_dir=output_dir,
        )
        # )

    # plot data using box plot
    epoch_colors = []
    for epoch_name in epoch_names:
        color = epoch_data[epoch_name]["color"]
        epoch_colors.append(color)

    boxplot_file_basename = (
        "post_minus_pre_boxplot.svg"
        if group_name is None
        else f"post_minus_pre_boxplot_{group_name}.svg"
    )
    post_minus_pre_boxplot_preview_filename = os.path.join(
        output_dir, boxplot_file_basename.replace(" ", "")
    )
    _plot_box_and_strip2(
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        df=df,
        num_cells=num_cells,
        ylabel="Post-Pre",
        output_filename=post_minus_pre_boxplot_preview_filename,
    )

    # post_minus_pre_boxplot_preview_file = IdeasPreviewFile(
    #     name="Post-pre distribution across the epochs",
    #     help=(
    #         "Distribution of post-pre activity across epochs displayed using a "
    #         "box plot. Lines connect the same cells together."
    #     ),
    #     file_path=os.path.abspath(post_minus_pre_boxplot_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # return (
    #     post_minus_pre_activity_differences_preview_files,
    #     # post_minus_pre_boxplot_preview_file,
    # )


def _clean_ax(ax):
    """Remove spines from an axis.

    :param ax: matplotlib axis
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_difference_cellmap_single_epoch_pair(
    epoch1_name,
    epoch2_name,
    x,
    y,
    df,
    save_name,
):
    """Plot a cellmap showing the differences in activity between a given pair of epochs.

    :Parameters
        epoch1_name (str): name of the first epoch
        epoch2_name (str): name of the second epoch
        x (array-like): X coordinates of the cells.
        y (array-like): Y coordinates of the cells.
        df (DataFrame): DataFrame containing the activity data with an 'Epoch' column.
        save_name (str): Path to save the resulting figure.
    """
    # instantiate figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
        width_ratios=[4, 6],
    )

    # extract data for the 2 epochs being compared
    df = df.reset_index()
    a_epoch = df[df["Epoch"] == epoch1_name]
    b_epoch = df[df["Epoch"] == epoch2_name]
    diff = b_epoch["Activity"].values - a_epoch["Activity"].values

    # clean axes
    ax0 = ax[0]
    ax1 = ax[1]
    _clean_ax(ax0)
    _clean_ax(ax1)

    # plot histogram (left panel)
    cool_warm, norm = plots.plot_shaded_hist(
        values=diff,
        ax=ax0,
        title=f"{epoch1_name} - {epoch2_name}",
        xlabel="Difference",
    )

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cool_warm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, label="Difference")

    # plot footprints (right panel)
    max_val = np.max(np.abs(diff))
    normalized_diff = (diff + max_val) / (2 * max_val)
    colors = [cool_warm(val) for val in normalized_diff]

    plots.plot_footprints(
        x,
        y,
        figure=ax1,
        colors=colors,
    )

    ax1.set_title("Cellmap of differences")
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # invert y-axis to match IDPS/IDEAS orientation (y=0 at top, y=max at bottom)
    ax1.invert_yaxis()

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        save_name,
        dpi=300,
        transparent=True,
    )
    plt.close(fig)


def _plot_difference_cellmap2(
    epoch_names,
    x,
    y,
    df,
    output_dir,
):
    """Plot a cellmap showing the differences in activity between each pair of epochs.

    :Parameters
        epoch_names (list): List of names corresponding to each epoch.
        x (array-like): X coordinates of the cells.
        y (array-like): Y coordinates of the cells.
        df (DataFrame): DataFrame containing the activity data with an 'Epoch' column.
        output_dir (str): Path to the output directory.
    """
    # post_minus_pre_activity_differences_preview_files = []
    for i in range(len(epoch_names)):
        for j in range(i, len(epoch_names) - 1):
            epoch_name1 = epoch_names[i]
            epoch_name2 = epoch_names[j + 1]

            output_filename = os.path.join(
                output_dir,
                f"post_minus_pre_differences_{epoch_name1}_{epoch_name2}.svg".replace(
                    " ", ""
                ),
            )

            _plot_difference_cellmap_single_epoch_pair(
                epoch1_name=epoch_name1,
                epoch2_name=epoch_name2,
                x=x,
                y=y,
                df=df,
                save_name=output_filename,
            )

            # post_minus_pre_activity_differences_preview_files.append(
            #     IdeasPreviewFile(
            #         name=f"Post-pre differences between {epoch_name1} and {epoch_name2}",
            #         help=(
            #             f"Pairwise difference of post-pre activity between epochs "
            #             f"{epoch_name1} and {epoch_name2}. The left panel presents "
            #             f"the data as a histogram. The right panel contains a cell "
            #             f"map colored by the magnitude of the difference in "
            #             f"post-pre activity between the epochs."
            #         ),
            #         file_path=os.path.abspath(output_filename),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            # )

    # return post_minus_pre_activity_differences_preview_files

def _plot_mixed_comparisons2(
    data,
    g1_color,
    g2_color,
    preview_filename,
):
    """Boxplot comparing the groups across the epochs.

    :Parameters
        data (pd.DataFrame): The data to be plotted, containing columns for
                            'Epoch', the metric activity, and 'Group'.
        g1_color (str): Color for the first group in the plot.
        g2_color (str): Color for the second group in the plot.
        anova_preview_filename: path to the preview file
    :Returns
        None: The function saves the plots as SVG files.
    """
    # AOV figure first
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Post-Pre Activity Comparison")
    _clean_ax(ax)
    sns.boxplot(
        ax=ax,
        data=data,
        x="epoch",
        y="true_mean_post-pre",
        hue="group",
        palette=[g1_color, g2_color],
        **BOX_PLOT_ARGS,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Post-Pre Activity")

    fig.tight_layout()
    fig.savefig(
        preview_filename,
        dpi=300,
        transparent=True,
    )


def _plot_box_and_strip2(
    epoch_names,
    epoch_colors,
    df,
    num_cells,
    ylabel,
    output_filename,
    epoch_column_name="Epoch",
    value_column_name="Activity",
):
    """Plot a combination of boxplot and stripplot for given epochs and
    activity data.

    :Parameters
        epoch_names (list): List of names corresponding to each epoch.
        epoch_colors (list): List of colors corresponding to each epoch.
        df (pandas.DataFrame): DataFrame containing the activity
        data with columns 'Epoch' and 'Activity'.
        num_cells (int): Number of cells to be plotted.
        ylabel (str): Label for the y-axis.
        output_dir (str): Path to the output directory.
    """
    fig, ax = plt.subplots(figsize=(2.5 * len(epoch_names), 6))

    # Plot the epoch-epoch change
    plots.plot_paired_lines(
        df=df,
        category_names=epoch_names,
        num_values=num_cells,
        ax=ax,
    )
    df = df.reset_index()

    # Plot the boxplot
    sns.boxplot(
        data=df,
        x=epoch_column_name,
        y=value_column_name,
        ax=ax,
        palette=epoch_colors,
        linewidth=3,
        fliersize=0,
        zorder=2,
        hue=epoch_column_name,
    )

    # Plot the stripplot
    sns.stripplot(
        data=df,
        x=epoch_column_name,
        y=value_column_name,
        ax=ax,
        color="black",
        size=3,
        alpha=0.2,
        jitter=0.02,
        zorder=1,
        legend=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.set_title(
        "Distribution of Post-Pre Activity Per Epoch",
        fontsize=config.PLOT_TITLE_FONT_SIZE,
    )

    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        transparent=True,
    )
    plt.close(fig)
