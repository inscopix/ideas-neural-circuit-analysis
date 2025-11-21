import json
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import seaborn as sns
# from ideas_commons.constants import (
#     FileCategory,
#     FileFormat,
#     FileStructure,
#     FileType,
#     GroupType,
# )
from scipy import stats

import utils.config as config
from analysis.peri_event_workflow import (
    plot_population_mean_event_window,
    plot_single_cell_modulation,
    plot_single_neurons_heatmap,
    validate_modulation_colors,
)
# from toolbox.utils.data_model import (
#     IdeasFile,
#     IdeasGroup,
#     IdeasPreviewFile,
# )
from ideas.exceptions import IdeasError
# from toolbox.utils.output_manifest import save_output_manifest
from utils.stats_utils import ttest
from ideas.analysis.utils import (
    get_file_size,
)
from utils.utils import compute_sampling_rate

from ideas.tools import log
from ideas.tools.types import IdeasFile
from ideas.tools import outputs

logger = log.get_logger()

PROCESSING_PARAMS = {}
PLOT_PARAMS = {}


def validate_group_colors(group_colors):
    """Validate group_colors input for compatibility with ax.plot and plt.fill.
    Defaults to predefined colors if the input is invalid or incompatible.

    :param group_colors: list of str or str
        List of colors for two groups or a comma-separated string.

    :returns:
        list of str
        A valid list of colors, defaulting to the predefined ones if invalid.
    :raises:
        IdeasError: Custom exception with specific error codes and messages.
    """
    # Default colors for two groups
    default_group_colors = ["#1f77b4", "#ff7f0e"]

    try:
        # Process group_colors if it's a string
        if isinstance(group_colors, str):
            group_colors = group_colors.split(",")
            group_colors = [x.strip() for x in group_colors]

        # Check if group_colors is a list or tuple
        if not isinstance(group_colors, (list, tuple)):
            raise IdeasError(
                "group_colors must be a list, tuple, or a comma-separated string.",
            )

        # Ensure the input contains exactly two colors
        if len(group_colors) != 2:
            raise IdeasError(
                "group_colors must contain exactly two colors.",
            )

        # Validate each color
        for color in group_colors:
            if not isinstance(color, str) or not mcolors.is_color_like(color):
                raise IdeasError(
                    f"Invalid group color: '{color}' is not recognized by Matplotlib.",
                )

        # Optional: Ensure no duplicates
        if len(set(group_colors)) != len(group_colors):
            raise IdeasError(
                "Duplicate colors detected in group_colors.",
            )

    except Exception:
        logger.warning("Using default group colors: %s", default_group_colors)
        return default_group_colors

    # Return validated colors if all checks pass
    return group_colors


def validate_combine_compare_peri_event_data_parameters(
    group1_traces_files: List[str],
    group1_stats_files: List[str],
    group1_name: str,
    group2_traces_files: List[str],
    group2_stats_files: List[str],
    group2_name: str,
    comparison_type: str,
    data_pairing: str,
    significance_threshold: float,
    tolerance: float,
    average_method: str,
    population_activity_plot_limits: str,
    activity_heatmap_color_limits: str,
    activity_by_modulation_plot_limits: str,
):
    """Validate the input parameter to the combine-and-compare peri-event data tool.

    :param group1_traces_files: peri-event analysis traces files from the first group
    :param group1_stats_files: peri-event analysis statistics files from the first group
    :param group1_name: name of the first group
    :param group2_traces_files: peri-event analysis traces files from the second group
    :param group2_stats_files: peri-event analysis statistics files from the second group
    :param group2_name: name of the second group
    :param comparison_type: type of statistical test to perform
    :param data_pairing: indicates whether observations should be paired for
                         statistical comparison
    :param significance_threshold: p-value threshold for classifying neurons
                                   as up- or down-modulated
    :param tolerance: maximum shift in seconds between the time windows
                      of the input files
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording
    :param population_activity_plot_limits: y-axis range (z-score) applied to the event-aligned
     population activity plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_heatmap_color_limits: colormap range (z-score) applied to the activity heatmap
     specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_by_modulation_plot_limits: y-axis range (z-score) applied to the event-aligned
     activity by modulation plot specified as 'min,max' (e.g. -1,1) or 'auto'
    """
    # validate group 1
    if not isinstance(group1_traces_files, list) or not isinstance(
        group1_stats_files, list
    ):
        raise IdeasError(
            "The first input group must consist of a list of files.",
        )

    if len(group1_traces_files) < 2 or len(group1_stats_files) < 2:
        raise IdeasError(
            "The first input group must contain at least two peri-event analyses.",
        )

    if len(group1_traces_files) != len(group1_stats_files):
        raise IdeasError(
            "The number of traces and statistics files must match for the first input group.",
        )

    # validate group 2
    if group2_traces_files is None:
        group2_traces_files = []

    if group2_stats_files is None:
        group2_stats_files = []

    if not isinstance(group2_traces_files, list) or not isinstance(
        group2_stats_files, list
    ):
        raise IdeasError(
            "The second input group must consist of a list of files.",
        )

    if len(group2_traces_files) != 0 or len(group2_stats_files) != 0:
        if len(group2_traces_files) < 2 or len(group2_stats_files) < 2:
            raise IdeasError(
                "The second input group must contain at least two peri-event analysis.",
            )

    if len(group2_traces_files) != len(group2_stats_files):
        raise IdeasError(
            "The number of traces and statistics files must match for the second input group.",
        )

    # validate parameters
    if (group2_traces_files or group2_stats_files) and group2_name is None:
        raise IdeasError(
            "The second input group must have a group name.",
        )

    if group1_name == group2_name:
        raise IdeasError(
            "Group names cannot be identical.",
        )

    valid_comparison_types = (
        "two_tailed",
        "one_tailed_less",
        "one_tailed_greater",
    )
    if not (group2_traces_files or group2_stats_files):
        valid_comparison_types = valid_comparison_types + (None,)

    if comparison_type not in valid_comparison_types:
        raise IdeasError(
            "The comparison type must be 'two_tailed', 'one_tailed_less', or 'one_tailed_greater'.",
        )

    valid_data_pairings = ("unpaired", "paired")
    if not (group2_traces_files or group2_stats_files):
        valid_data_pairings = valid_data_pairings + (None,)

    if data_pairing not in valid_data_pairings:
        raise IdeasError(
            "The data pairing parameter must be set to 'paired' or 'unpaired'.",
        )

    if significance_threshold is not None and (
        significance_threshold < 0 or significance_threshold > 1
    ):
        raise IdeasError(
            "Significance threshold must be between 0 and 1.",
        )

    if tolerance < 0:
        raise IdeasError(
            "Tolerance must be greater than 0.",
        )

    if average_method not in ["neurons", "recordings"]:
        raise IdeasError(
            "An invalid average method was supplied. Must be 'neurons' or 'recordings'.",
        )

    # validate optional plot params
    for param_value, param_name in [
        (
            population_activity_plot_limits,
            "Population activity (y-axis range)",
        ),
        (activity_heatmap_color_limits, "Activity heatmap (colormap range)"),
        (
            activity_by_modulation_plot_limits,
            "Activity by modulation group (y-axis range)",
        ),
    ]:
        if param_value is not None:
            if param_value != "auto":
                # ensure param is formatted as 'x,y'
                try:
                    min_lim, max_lim = param_value.split(",")
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


def generate_output_manifest(
    group1_traces_files: List[str],
    group1_stats_files: List[str],
    group2_traces_files: List[str],
    group2_stats_files: List[str],
    output_files: List,
    output_dir: str,
    output_group_key: str = "combine_compare_peri_event_data_output",
):
    """Construct and save output manifest and metadata to disk.

    :param group1_traces_files: peri-event analysis traces files from the first group
    :param group1_stats_files: peri-event analysis statistics files from the first group
    :param group2_traces_files: peri-event analysis traces files from the second group
    :param group2_stats_files: peri-event analysis statistics files from the second group
    :param output_files: list of IdeasFile to include in the output manifest
    :param output_dir: path to the output directory
    """
    pass
    # # GROUP 1
    # # source traces FILES (group 1)
    # source_traces_files_group1 = [
    #     IdeasFile(
    #         file_key="group1_traces_files",
    #         file_path=os.path.abspath(f),
    #         file_type=FileType.EVENT_ALIGNED_NEURAL_DATA.value[1],
    #         file_format=FileFormat.CSV_FILE.value[1],
    #         file_structure=FileStructure.TIME_SERIES.value[1],
    #         file_category=FileCategory.SOURCE.value[1],
    #     )
    #     for f in group1_traces_files
    # ]

    # # source statistics FILES (group 1)
    # source_stats_files_group1 = [
    #     IdeasFile(
    #         file_key="group1_stats_files",
    #         file_path=os.path.abspath(f),
    #         file_type=FileType.STATISTICS.value[1],
    #         file_format=FileFormat.CSV_FILE.value[1],
    #         file_structure=FileStructure.TABLE.value[1],
    #         file_category=FileCategory.SOURCE.value[1],
    #     )
    #     for f in group1_stats_files
    # ]

    # files = [
    #     *source_traces_files_group1,
    #     *source_stats_files_group1,
    # ]

    # # GROUP 2
    # if len(group2_traces_files) > 0:
    #     # source traces FILES (group 2)
    #     source_traces_files_group2 = [
    #         IdeasFile(
    #             file_key="group2_traces_files",
    #             file_path=os.path.abspath(f),
    #             file_type=FileType.EVENT_ALIGNED_NEURAL_DATA.value[1],
    #             file_format=FileFormat.CSV_FILE.value[1],
    #             file_structure=FileStructure.TIME_SERIES.value[1],
    #             file_category=FileCategory.SOURCE.value[1],
    #         )
    #         for f in group2_traces_files
    #     ]

    #     # source statistics FILES (group 2)
    #     source_stats_files_group2 = [
    #         IdeasFile(
    #             file_key="group2_stats_files",
    #             file_path=os.path.abspath(f),
    #             file_type=FileType.STATISTICS.value[1],
    #             file_format=FileFormat.CSV_FILE.value[1],
    #             file_structure=FileStructure.TABLE.value[1],
    #             file_category=FileCategory.SOURCE.value[1],
    #         )
    #         for f in group2_stats_files
    #     ]

    #     files.extend(
    #         [
    #             *source_traces_files_group2,
    #             *source_stats_files_group2,
    #         ]
    #     )

    # # combine input and output files into a single list
    # files.extend(output_files)

    # # combine-and-compare peri-event data GROUP
    # combine_compare_peri_event_data_group = IdeasGroup(
    #     group_key=output_group_key,
    #     group_type=GroupType.TOOL_OUTPUT.value[1],
    #     files=files,
    # )

    # # save output manifest & metadata to disk
    # save_output_manifest(combine_compare_peri_event_data_group, output_dir)


def generate_subpopulation_activity_plot(
    traces_timeline: List[float],
    modulated_cells_dict: dict,
    output_filename: str,
    activity_by_modulation_plot_limits: str = None,
    modulation_colors=None,
    average_method=None,
    epoch_name=None,
):
    """Plot subpopulation activity by modulation group.

    :param traces_timeline: x-values to use in the plot
    :param modulated_cells_dict: dictionary containing data per modulation group (up/down/non)
    :param output_filename: path to the output file
    :param modulation_colors: string list with color inputs
    :param average_method: method used for averaging the data (by neuron vs by recording)
                           if none, defaults to recording
    :param epoch_name: name of the epoch
    """
    modulation_colors = validate_modulation_colors(modulation_colors)
    # determine number of samples based on averaging method
    global PROCESSING_PARAMS
    average_method = PROCESSING_PARAMS.get("average_method", average_method)

    if average_method == "neurons":
        num_up_modulated = modulated_cells_dict["up_modulated"]["num_cells"]
        num_down_modulated = modulated_cells_dict["down_modulated"][
            "num_cells"
        ]
        num_non_modulated = modulated_cells_dict["non_modulated"]["num_cells"]
    else:
        num_up_modulated = modulated_cells_dict["up_modulated"]["num_samples"]
        num_down_modulated = modulated_cells_dict["down_modulated"][
            "num_samples"
        ]
        num_non_modulated = modulated_cells_dict["non_modulated"][
            "num_samples"
        ]

    # plot data
    if epoch_name is not None:
        subtitle = f"(n={average_method}, epoch: {epoch_name})"
    else:
        subtitle = f"(n={average_method})"

    # set plot limits
    global PLOT_PARAMS
    plot_limits = PLOT_PARAMS.get("activity_by_modulation_plot_limits", None)
    if activity_by_modulation_plot_limits is not None:
        plot_limits = activity_by_modulation_plot_limits

    plot_single_cell_modulation(
        x=traces_timeline,
        x_limits=(traces_timeline[0], traces_timeline[-1]),
        mean_up=modulated_cells_dict["up_modulated"]["mean"],
        sem_up=modulated_cells_dict["up_modulated"]["sem"],
        num_cells_up_modulated=num_up_modulated,
        mean_down=modulated_cells_dict["down_modulated"]["mean"],
        sem_down=modulated_cells_dict["down_modulated"]["sem"],
        num_cells_down_modulated=num_down_modulated,
        mean_non=modulated_cells_dict["non_modulated"]["mean"],
        sem_non=modulated_cells_dict["non_modulated"]["sem"],
        num_cells_non_modulated=num_non_modulated,
        event_type=None,
        modulation_colors=modulation_colors,
        output_filename=output_filename,
        subtitle=subtitle,
        plot_limits=plot_limits,
    )


def extract_modulation_group_data(
    traces_dataframes: pd.DataFrame,
    stats_dataframes: pd.DataFrame,
    average_method: str,
):
    """Extract, group, and average neurons across input files by modulation group.

    :param traces_dataframes: list of dataframes containing peri-event traces
    :param stats_dataframes: list of dataframes containing peri-event statistics
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording

    :return modulated_cells_dict: dictionary containing (mean, sem, num_cells)
                                  for each modulation group
    """
    names_to_exclude = [
        "population",
        "up_modulated",
        "down_modulated",
        "non_modulated",
    ]

    # store traces per modulation
    up_modulated_traces = None
    down_modulated_traces = None
    non_modulated_traces = None

    # store fractions per modulation group for each input recording
    up_modulated_fractions = []
    down_modulated_fractions = []
    non_modulated_fractions = []

    # store number of cells per modulation group
    total_num_up_modulated_cells = 0
    total_num_down_modulated_cells = 0
    total_num_non_modulated_cells = 0

    # for each traces file
    for traces_df, stats_df in zip(traces_dataframes, stats_dataframes):
        up_modulated_filter = (stats_df["modulation"] == 1) & (
            ~stats_df["name"].isin(names_to_exclude)
        )
        down_modulated_filter = (stats_df["modulation"] == -1) & (
            ~stats_df["name"].isin(names_to_exclude)
        )
        non_modulated_filter = (stats_df["modulation"] == 0) & (
            ~stats_df["name"].isin(names_to_exclude)
        )

        up_modulated_cell_names = (
            stats_df[up_modulated_filter]["name"]
            .apply(lambda x: x + "_mean")
            .tolist()
        )
        down_modulated_cell_names = (
            stats_df[down_modulated_filter]["name"]
            .apply(lambda x: x + "_mean")
            .tolist()
        )
        non_modulated_cell_names = (
            stats_df[non_modulated_filter]["name"]
            .apply(lambda x: x + "_mean")
            .tolist()
        )

        # compute fractions of up/down/non modulated cells in each input recording
        num_up_modulated_cells = len(up_modulated_cell_names)
        num_down_modulated_cells = len(down_modulated_cell_names)
        num_non_modulated_cells = len(non_modulated_cell_names)
        total_num_cells = (
            num_up_modulated_cells
            + num_down_modulated_cells
            + num_non_modulated_cells
        )

        up_modulated_fractions.append(num_up_modulated_cells / total_num_cells)
        down_modulated_fractions.append(
            num_down_modulated_cells / total_num_cells
        )
        non_modulated_fractions.append(
            num_non_modulated_cells / total_num_cells
        )

        # update cell counts
        total_num_up_modulated_cells += num_up_modulated_cells
        total_num_down_modulated_cells += num_down_modulated_cells
        total_num_non_modulated_cells += num_non_modulated_cells

        # aggregate activity per group
        if average_method == "neurons":
            if up_modulated_traces is not None:
                up_modulated_traces = pd.concat(
                    [up_modulated_traces, traces_df[up_modulated_cell_names]],
                    axis=1,
                )
            else:
                up_modulated_traces = traces_df[up_modulated_cell_names]

            if down_modulated_traces is not None:
                down_modulated_traces = pd.concat(
                    [
                        down_modulated_traces,
                        traces_df[down_modulated_cell_names],
                    ],
                    axis=1,
                )
            else:
                down_modulated_traces = traces_df[down_modulated_cell_names]

            if non_modulated_traces is not None:
                non_modulated_traces = pd.concat(
                    [
                        non_modulated_traces,
                        traces_df[non_modulated_cell_names],
                    ],
                    axis=1,
                )
            else:
                non_modulated_traces = traces_df[non_modulated_cell_names]
        else:
            # average method is recordings
            if up_modulated_traces is not None:
                # iteratively stacks the means of up modulated cells per recording
                up_modulated_traces = np.column_stack(
                    [
                        up_modulated_traces,
                        np.nanmean(traces_df[up_modulated_cell_names], axis=1),
                    ]
                )
            else:
                # initializes up modulated cells by obtaining the mean in the first recording
                up_modulated_traces = np.nanmean(
                    traces_df[up_modulated_cell_names], axis=1
                )

            if down_modulated_traces is not None:
                # iteratively stacks the means of down modulated cells per recording
                down_modulated_traces = np.column_stack(
                    [
                        down_modulated_traces,
                        np.nanmean(
                            traces_df[down_modulated_cell_names], axis=1
                        ),
                    ]
                )
            else:
                # initializes down modulated cells by obtaining the mean in the first recording
                down_modulated_traces = np.nanmean(
                    traces_df[down_modulated_cell_names], axis=1
                )

            if non_modulated_traces is not None:
                # iteratively stacks the means of non modulated cells per recording
                non_modulated_traces = np.column_stack(
                    [
                        non_modulated_traces,
                        np.nanmean(
                            traces_df[non_modulated_cell_names], axis=1
                        ),
                    ]
                )
            else:
                # initializes non modulated cells by obtaining the mean in the first recording
                non_modulated_traces = np.nanmean(
                    traces_df[non_modulated_cell_names], axis=1
                )

    with warnings.catch_warnings():
        # ignore warnings that may occur when computing the mean or sem of an empty array
        warnings.simplefilter("ignore", category=RuntimeWarning)

        up_modulated_mean = np.nanmean(up_modulated_traces, axis=1)
        down_modulated_mean = np.nanmean(down_modulated_traces, axis=1)
        non_modulated_mean = np.nanmean(non_modulated_traces, axis=1)

        up_modulated_sem = stats.sem(
            up_modulated_traces, axis=1, nan_policy="omit"
        )
        down_modulated_sem = stats.sem(
            down_modulated_traces, axis=1, nan_policy="omit"
        )
        non_modulated_sem = stats.sem(
            non_modulated_traces, axis=1, nan_policy="omit"
        )

    modulated_cells_dict = {
        "up_modulated": {
            "num_samples": up_modulated_traces.shape[1],
            "num_cells": total_num_up_modulated_cells,
            "mean": up_modulated_mean,
            "sem": up_modulated_sem,
            "fractions": up_modulated_fractions,
        },
        "down_modulated": {
            "num_samples": down_modulated_traces.shape[1],
            "num_cells": total_num_down_modulated_cells,
            "mean": down_modulated_mean,
            "sem": down_modulated_sem,
            "fractions": down_modulated_fractions,
        },
        "non_modulated": {
            "num_samples": non_modulated_traces.shape[1],
            "num_cells": total_num_non_modulated_cells,
            "mean": non_modulated_mean,
            "sem": non_modulated_sem,
            "fractions": non_modulated_fractions,
        },
    }
    return modulated_cells_dict


def generate_modulation_pie_chart(
    num_cells_up_modulated: int,
    num_cells_down_modulated: int,
    num_cells_non_modulated: int,
    output_filename: str,
    modulation_colors=None,
    epoch_name=None,
):
    """Plot the fraction of up/down/non modulated neurons.

    :param num_cells_up_modulated: number of up-modulated cells
    :param num_cells_down_modulated: number of down-modulated cells
    :param num_cells_non_modulated: number of non-modulated cells
    :param output_filename: path to the output file
    :param modulation_colors: string list with color inputs
    :param epoch_name: name of the epoch
    """
    modulation_colors = validate_modulation_colors(modulation_colors)

    # Dynamic color mapping
    up_color, down_color, non_color = modulation_colors

    fig, ax = plt.subplots()

    # initialize data
    sizes = []
    labels = []
    colors = []
    total_num_cells = (
        num_cells_up_modulated
        + num_cells_down_modulated
        + num_cells_non_modulated
    )

    # only include data for which the number of cells is greater than 0
    if num_cells_up_modulated > 0:
        sizes.append(num_cells_up_modulated)
        labels.append(
            f"up modulated ("
            f"{num_cells_up_modulated / total_num_cells * 100:.2f}%, "
            f"n={num_cells_up_modulated})"
        )
        colors.append(up_color)

    if num_cells_down_modulated > 0:
        sizes.append(num_cells_down_modulated)
        labels.append(
            f"down modulated "
            f"({num_cells_down_modulated / total_num_cells * 100:.2f}%, "
            f"n={num_cells_down_modulated})"
        )
        colors.append(down_color)

    if num_cells_non_modulated > 0:
        sizes.append(num_cells_non_modulated)
        labels.append(
            f"non modulated ({num_cells_non_modulated / total_num_cells * 100:.2f}%"
            f", n={num_cells_non_modulated})"
        )
        colors.append(non_color)

    # create pie chart
    patches, _ = ax.pie(
        sizes,
        colors=colors,
        wedgeprops={"alpha": config.PLOT_SHADING_ALPHA},
    )

    # add labels to plot
    ax.legend(handles=patches, labels=labels, loc="lower center")

    # set title and subtitle
    title = "Fraction of Neurons Per Modulation Group"

    if epoch_name is not None:
        subtitle = f"\n(n=neurons, epoch: {epoch_name})"
    else:
        subtitle = "\n(n=neurons)"
    title += subtitle

    ax.set_title(
        title,
        fontsize=config.PLOT_TITLE_FONT_SIZE,
    )

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        output_filename, dpi=300, bbox_inches="tight", transparent=True
    )


def generate_activity_heatmap(
    traces_timeline: List[float],
    df_traces: pd.DataFrame,
    df_stats: pd.DataFrame,
    output_filename: str,
    cmap=None,
):
    """Plot individual cell activity heatmap over the event window.

    :param traces_timeline: x-values to use for the plot
    :param df_traces: dataframe containing traces from all input files (num_timepoints x num_cells)
    :param df_stats: dataframe containing statistics from all input files
    :param output_filename: path to the output file
    :param cmap: colormap applied to the activity heatmap
    """
    # compute ranks and reverse order to get descending order
    # i.e. from most up-modulated to most down-modulated
    zscores = df_stats["z-score"].tolist()
    ranks = np.argsort(zscores)[::-1]

    # transpose dataframe to match expected format for plotting (num_cells, num_timepoints)
    # and sort rows by their modulation ranking
    ordered_traces = df_traces.T.iloc[ranks]

    global PLOT_PARAMS
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
    plot_single_neurons_heatmap(
        single_cell_data=ordered_traces,
        x_limits=(traces_timeline[0], traces_timeline[-1]),
        event_type=None,
        output_filename=output_filename,
        color_limits=PLOT_PARAMS.get("activity_heatmap_color_limits", None),
        cmap=cmap,
    )


def generate_population_activity_plot(
    traces_timeline: List[float],
    population_mean: List[float],
    population_sem: List[float],
    output_filename: str,
    color: str = "#1f77b4",
):
    """Plot population activity across all input recordings.

    :param traces_timeline: x-values to use for the plot
    :param population_mean: population mean activity at each time point
                            (1D array of size num_timepoints)
    :param population_sem: population standard error of the mean activity at each time point
                           (1D array of size num_timepoints)
    :param df_traces: dataframe containing traces from all input files
                      (num_timepoints x num_cells)
    :param output_filename: path to the output file
    :param color: color to use for the population activity line
    """
    global PLOT_PARAMS
    plot_population_mean_event_window(
        x_values=traces_timeline,
        x_limits=(traces_timeline[0], traces_timeline[-1]),
        population_mean=population_mean,
        population_sem=population_sem,
        shuffled_mean=[None] * len(population_mean),
        shuffled_conf_interval_low=[None] * len(population_mean),
        shuffled_conf_interval_high=[None] * len(population_mean),
        significance_threshold=None,
        event_type=None,
        output_filename=output_filename,
        color=color,
        plot_limits=PLOT_PARAMS.get("population_activity_plot_limits", None),
    )


def plot_comparison_of_population_activity(
    group1_data: dict,
    group1_name: str,
    group2_data: dict,
    group2_name: str,
    output_filename: str,
    group_colors=None,
):
    """Plot the population mean activity for each group.

    :param x_values: values along the x-axis
    :param group1_data: data dictionary for the first group
    :param group1_name: name of the first group
    :param group2_data: data dictionary for the first group
    :param group2_name: name of the second group
    :param output_file_basename: path to the output image without a file extension
    :param group_colors: comma separated strings with color inputs
    """
    group_colors = validate_group_colors(group_colors)
    fig, ax = plt.subplots()

    # plot event-aligned population data for group 1
    if not all(np.isnan(group1_data["population"]["sem"])):
        ax.plot(
            group1_data["x_values"],
            group1_data["population"]["mean"],
            label=f"mean activity ± sem ({group1_name})",
            color=group_colors[0],
        )
        ax.fill_between(
            group1_data["x_values"],
            np.array(group1_data["population"]["mean"])
            + np.array(group1_data["population"]["sem"]),
            np.array(group1_data["population"]["mean"])
            - np.array(group1_data["population"]["sem"]),
            color=group_colors[0],
            alpha=config.PLOT_SHADING_ALPHA,
        )
    else:
        # When there is only one event, only mean is plotted because sem is non-sensical in this
        # case
        ax.plot(
            group1_data["x_values"],
            group1_data["population"]["mean"],
            label=f"mean activity ({group1_name})",
            color=group_colors[0],
        )

    # plot event-aligned population data for group 2
    if not all(np.isnan(group2_data["population"]["sem"])):
        ax.plot(
            group2_data["x_values"],
            group2_data["population"]["mean"],
            label=f"mean activity ± sem ({group2_name})",
            color=group_colors[1],
        )
        ax.fill_between(
            group2_data["x_values"],
            np.array(group2_data["population"]["mean"])
            + np.array(group2_data["population"]["sem"]),
            np.array(group2_data["population"]["mean"])
            - np.array(group2_data["population"]["sem"]),
            color=group_colors[1],
            alpha=config.PLOT_SHADING_ALPHA,
        )
    else:
        # When there is only one event, only mean is plotted because sem is non-sensical in this
        # case
        ax.plot(
            group2_data["x_values"],
            group2_data["population"]["mean"],
            label=f"mean activity ({group2_name})",
            color=group_colors[1],
        )

    # add vertical line at x=0
    # this line represents the reference event time around which we want to observe patterns
    ax.axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

    # add labels
    ax.set_title(
        "Event-Aligned Population Activity",
        fontsize=config.PLOT_TITLE_FONT_SIZE,
    )
    ax.set_xlabel(
        "Time from Event (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )
    ax.set_ylabel(
        "Neural Activity (z-score)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )

    ax.set_xlim((group1_data["x_values"][0], group1_data["x_values"][-1]))
    ax.margins(x=0)
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


def plot_comparison_of_subpopulation_activity(
    group1_data: dict,
    group1_name: str,
    group2_data: dict,
    group2_name: str,
    output_filename: str,
    group_colors=None,
):
    """Plot the sub-population mean activity for each group.

    :param x_values: values along the x-axis
    :param group1_data: data dictionary for the first group
    :param group1_name: name of the first group
    :param group2_data: data dictionary for the first group
    :param group2_name: name of the second group
    :param output_file_basename: path to the output image without a file extension
    :param These colors represent the two comparison groups.
    """
    group_colors = validate_group_colors(group_colors)
    fig, ax = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10, 4)
    )

    # plot data for each modulation group
    for i, modulation_group in enumerate(
        ["up_modulated", "down_modulated", "non_modulated"]
    ):
        # plot data from GROUP 1
        if group1_data[modulation_group]["num_cells"] > 0:
            ax[i].plot(
                group1_data["x_values"],
                group1_data[modulation_group]["mean"],
                label=f"{group1_name}",
                color=group_colors[0],
            )
        if group1_data[modulation_group]["num_cells"] > 1:
            ax[i].fill_between(
                group1_data["x_values"],
                np.array(group1_data[modulation_group]["mean"])
                + np.array(group1_data[modulation_group]["sem"]),
                np.array(group1_data[modulation_group]["mean"])
                - np.array(group1_data[modulation_group]["sem"]),
                alpha=config.PLOT_SHADING_ALPHA,
                color=group_colors[0],
            )

        # plot data from GROUP 2
        if group2_data[modulation_group]["num_cells"] > 0:
            ax[i].plot(
                group2_data["x_values"],
                group2_data[modulation_group]["mean"],
                label=f"{group2_name}",
                color=group_colors[1],
            )
        if group2_data[modulation_group]["num_cells"] > 1:
            ax[i].fill_between(
                group2_data["x_values"],
                np.array(group2_data[modulation_group]["mean"])
                + np.array(group2_data[modulation_group]["sem"]),
                np.array(group2_data[modulation_group]["mean"])
                - np.array(group2_data[modulation_group]["sem"]),
                alpha=config.PLOT_SHADING_ALPHA,
                color=group_colors[1],
            )

    # add labels
    fig.suptitle(
        "Event-Aligned Activity by Modulation Group",
        fontsize=config.PLOT_TITLE_FONT_SIZE,
    )
    ax[0].set_title("Up-Modulated", fontsize=config.PLOT_TITLE_FONT_SIZE)
    ax[1].set_title("Down-Modulated", fontsize=config.PLOT_TITLE_FONT_SIZE)
    ax[2].set_title("Non-Modulated", fontsize=config.PLOT_TITLE_FONT_SIZE)

    ax[1].set_xlabel(
        "Time from Event (seconds)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )
    ax[0].set_ylabel(
        "Neural Activity (z-score)", fontsize=config.PLOT_LABEL_FONT_SIZE
    )

    for i in range(3):
        # add vertical line representing the event time
        ax[i].axvline(0, color=config.PLOT_EVENT_REF_LINE_COLOR)

        # set x limits
        ax[i].set_xlim(
            (group1_data["x_values"][0], group1_data["x_values"][-1])
        )

        # add legend
        ax[i].margins(x=0)

        # display legend if there is at least one group
        _, labels = ax[i].get_legend_handles_labels()
        if len(labels) > 0:
            ax[i].legend()

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


def plot_fractions_of_modulated_neurons(
    group1_data: dict,
    group1_name: str,
    group2_data: dict,
    group2_name: str,
    output_filename: str,
    group_colors=None,
):
    """Plot and compare the fractions of modulated neurons for each group.

    :param group1_data: data dictionary for the first group
    :param group1_name: name of the first group
    :param group2_data: data dictionary for the first group
    :param group2_name: name of the second group
    :param output_filename: path to the output file
    :param group_colors: comma separated strings with color inputs
    These colors represent the two comparison groups.
    """
    group_colors = validate_group_colors(group_colors)

    fig, ax = plt.subplots()

    # reorganize data into a dataframe for plotting
    modulation_dict = {
        "up_modulated": "Up-Modulated",
        "down_modulated": "Down-Modulated",
        "non_modulated": "Non-Modulated",
    }
    data = []
    for modulation in ["up_modulated", "down_modulated", "non_modulated"]:
        # group 1
        for i in range(len(group1_data[modulation]["fractions"])):
            data.append(
                {
                    "group": group1_name,
                    "modulation": modulation_dict[modulation],
                    "fraction": group1_data[modulation]["fractions"][i],
                }
            )

        # group 2
        for i in range(len(group2_data[modulation]["fractions"])):
            data.append(
                {
                    "group": group2_name,
                    "modulation": modulation_dict[modulation],
                    "fraction": group2_data[modulation]["fractions"][i],
                }
            )
    df = pd.DataFrame(data)

    # plot data
    sns.barplot(
        data=df,
        x="modulation",
        y="fraction",
        hue="group",
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        errorbar="se",
        palette=group_colors,
        ax=ax,
    )

    sns.stripplot(
        data=df,
        x="modulation",
        y="fraction",
        hue="group",
        dodge=True,
        palette="dark:0",
        alpha=0.35,
        ax=ax,
    )

    # remove extra legend handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])

    # add labels
    ax.set_title(
        "Fraction of Neurons in Each Modulation Group",
        fontsize=config.PLOT_TITLE_FONT_SIZE,
    )
    ax.set_ylabel("Fraction of Neurons", fontsize=config.PLOT_LABEL_FONT_SIZE)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=config.PLOT_LABEL_FONT_SIZE)

    # save figure to disk
    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


def assign_modulation(row, threshold, score_col, p_col):
    """Assign a modulation category (up/down/non) to each neuron.

    :param row: single row from a dataframe
    :param threshold: p-value threshold below which neurons are considered significantly modulated
    :param score_col: name of the column containing the score data
    :param p_col: name of the column containing the p-value data

    :return: the new modulation classification
    """
    if row[p_col] < (threshold / 2):
        if row[score_col] < 0:
            return -1.0
        elif row[score_col] > 0:
            return 1.0
    return 0.0


def reclassify_neurons(df: pd.DataFrame, significance_threshold: float):
    """Reclassify neurons by modulation based on the provided significance threshold.

    :param df: input dataframe
    :param significance_threshold: p-value threshold
    """
    df["modulation"] = df.apply(
        lambda row: assign_modulation(
            row=row,
            threshold=significance_threshold,
            score_col="z-score",
            p_col="p-value",
        ),
        axis=1,
    )
    return df


def combine_peri_event_data(
    traces_files: List[str],
    stats_files: List[str],
    significance_threshold: float,
    group_name: str,
    output_dir: str,
    group_id: int,
    average_method: str,
    tolerance: float,
    color=None,
    modulation_colors=None,
    cmap=None,
):
    """Combine multiple peri-event analyses together.

    :param traces_files: peri-event analysis traces files
    :param stats_files: peri-event analysis statistics files
    :param significance_threshold: p-value threshold for classifying
                                   neurons as up- or down-modulated
    :param group_name: name of the group
    :param output_dir: path to the output directory
    :param group_id: identifier of the group being combined (1 or 2)
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording
    :param color: plot color for the single group
    :param modulation_colors: string list with color inputs
    :param cmap: colormap applied to the activity heatmap
    """
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
    # read csv files into pandas dataframes
    traces_dataframes = [pd.read_csv(f) for f in traces_files]
    stats_dataframes = [pd.read_csv(f) for f in stats_files]

    # verify and adjust time windows based on specified tolerance,
    # i.e. we allow the number of timepoints in time windows to differ slightly
    # and we match them by trimming the excess time points in the longer ones
    # to match the shorter one
    time_windows_match = True
    min_time_window_index = 0
    min_time_window_length = len(traces_dataframes[min_time_window_index])
    for i, df in enumerate(traces_dataframes):
        if len(df) != min_time_window_length:
            time_windows_match = False
        if len(df) < min_time_window_length:
            min_time_window_index = i
            min_time_window_length = len(df)

    if not time_windows_match:
        logger.warning(
            f"The time windows of the input traces files do not match. "
            f"Shortest time window containing {min_time_window_length} time points "
            f"identified in file '{os.path.basename(traces_files[min_time_window_index])}'"
        )

        # retrieve first time point of shortest time window
        time_window_first_timepoint = traces_dataframes[min_time_window_index][
            "Time"
        ].tolist()[0]

        # trim time windows to match shortest one
        for i, df in enumerate(traces_dataframes):
            if len(df) > min_time_window_length:
                logger.warning(
                    f"Trimming and aligning time points from input file "
                    f"'{os.path.basename(traces_files[i])}' "
                    f"to match shortest time window."
                )

                # try to align based on closest time point
                closest_index = (
                    (df["Time"] - time_window_first_timepoint)
                    .abs()
                    .argsort()
                    .tolist()[0]
                )
                if (
                    len(
                        df[
                            closest_index : closest_index
                            + min_time_window_length
                        ]
                    )
                    == min_time_window_length
                ):
                    traces_dataframes[i] = df.iloc[
                        closest_index : closest_index + min_time_window_length
                    ].reset_index(drop=True)
                else:
                    # if alignment based on closest time point fails (e.g. lack of data points),
                    # align based on first time point.
                    traces_dataframes[i] = df.iloc[
                        :min_time_window_length
                    ].reset_index(drop=True)

                # compute time shift between the time window in
                # current file vs file with shortest time window
                time_delta = abs(
                    traces_dataframes[i].iloc[0]["Time"]
                    - time_window_first_timepoint
                )
                logger.warning(
                    f"The time windows of the input traces files "
                    f"'{os.path.basename(traces_files[min_time_window_index])}' "
                    f"and '{os.path.basename(traces_files[i])}' "
                    f"differ by {time_delta} seconds."
                )

                # compare time difference with tolerance
                if time_delta > tolerance:
                    raise IdeasError(
                        f"The difference in the time windows of the input traces files "
                        f"'{os.path.basename(traces_files[min_time_window_index])}' and "
                        f"'{os.path.basename(traces_files[i])}' "
                        f"exceeds the specified tolerance.",
                    )

    # ensure timelines match across the input traces files
    traces_timeline = traces_dataframes[0]["Time"].tolist()
    for i, df in enumerate(traces_dataframes[1:]):
        # time windows should have the same size in all input files
        if len(df["Time"]) != len(traces_timeline):
            raise IdeasError(
                f"The time windows of the input traces files '{os.path.basename(traces_files[0])}'"
                f" and '{os.path.basename(traces_files[i + 1])}' do not match.",
            )

    # combine traces data
    for i in range(len(traces_files)):
        # retain per-cell data only and drop the other columns
        cols_to_drop = [
            "Time",
            "population_mean",
            "population_sem",
            "population_shuffled_mean",
            "population_shuffled_lower_conf",
            "population_shuffled_upper_conf",
            "up_modulated_mean",
            "up_modulated_sem",
            "down_modulated_mean",
            "down_modulated_sem",
            "non_modulated_mean",
            "non_modulated_sem",
        ]
        traces_dataframes[i].drop(
            columns=cols_to_drop, axis=1, inplace=True, errors="ignore"
        )

    df_traces = pd.concat(traces_dataframes, axis=1)

    # exclude sem columns, not needed for plotting
    sem_cols = [c for c in df_traces.columns if "sem" in c]
    df_traces.drop(columns=sem_cols, axis=1, inplace=True)

    # combine statistics data
    for i in range(len(stats_files)):
        # add filename to distinguish cells from different input files
        stats_dataframes[i]["file"] = os.path.basename(stats_files[i])
        # drop first 4 rows (population, up_modulated, down_modulated, non_modulated)
        # to retain per-cell data only
        stats_dataframes[i] = (
            stats_dataframes[i].iloc[4:].reset_index(drop=True)
        )

        # reclassify neurons based on significance threshold supplied
        if significance_threshold is not None and significance_threshold > 0:
            stats_dataframes[i] = reclassify_neurons(
                stats_dataframes[i], significance_threshold
            )

    df_stats = pd.concat(stats_dataframes, axis=0)

    output_stats_filename = os.path.join(
        output_dir,
        f"event_aligned_statistics_{group_name}.csv".replace(" ", "_"),
    )
    df_stats.to_csv(output_stats_filename, index=False)
    logger.info(
        f"Event-aligned statistics saved "
        f"({os.path.basename(output_stats_filename)}, "
        f"size: {get_file_size(output_stats_filename)})"
    )

    # generate single-cell activity heatmap
    single_cell_heatmap_preview_filename = os.path.join(
        output_dir,
        f"event_aligned_single_cell_activity_heatmap_{group_name}".replace(
            " ", "_"
        )
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    generate_activity_heatmap(
        traces_timeline=traces_timeline,
        df_traces=df_traces,
        df_stats=df_stats,
        output_filename=single_cell_heatmap_preview_filename,
        cmap=cmap,
    )

    # generate population activity data
    if average_method == "neurons":
        population_mean = np.nanmean(df_traces, axis=1)
        population_sem = stats.sem(df_traces, axis=1, nan_policy="omit")
    else:
        # drop cell sem columns from all recordings
        simplified_traces_dataframes = [
            traces_df.drop(
                columns=[c for c in traces_df.columns if "sem" in c],
                axis=1,
            )
            for traces_df in traces_dataframes
        ]

        # compute means of cell population data over reach recording
        combined_population_means = np.column_stack(
            [
                np.nanmean(traces_df, axis=1)
                for traces_df in simplified_traces_dataframes
            ]
        )
        population_mean = np.nanmean(combined_population_means, axis=1)
        population_sem = stats.sem(
            combined_population_means, axis=1, nan_policy="omit"
        )

    df_traces["Time"] = traces_timeline
    df_traces["population_mean"] = population_mean
    df_traces["population_sem"] = population_sem

    population_activity_preview_filename = os.path.join(
        output_dir,
        f"event_aligned_population_activity_{group_name}".replace(" ", "_")
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    generate_population_activity_plot(
        traces_timeline=traces_timeline,
        population_mean=df_traces["population_mean"],
        population_sem=df_traces["population_sem"],
        output_filename=population_activity_preview_filename,
        color=color,
    )

    # separate cells based on their modulation group (up/down/non)
    # and compute group data (num cells, mean, sem)
    combined_cell_data = extract_modulation_group_data(
        traces_dataframes,
        stats_dataframes,
        average_method,
    )

    # add population data to output data
    combined_cell_data["population"] = {
        "mean": df_traces["population_mean"].tolist(),
        "sem": df_traces["population_sem"].tolist(),
        "num_cells": len(df_traces.columns),
    }

    # add timeline to output data
    combined_cell_data["x_values"] = traces_timeline

    df_traces["up_modulated_mean"] = combined_cell_data["up_modulated"]["mean"]
    df_traces["up_modulated_sem"] = combined_cell_data["up_modulated"]["sem"]
    df_traces["down_modulated_mean"] = combined_cell_data["down_modulated"][
        "mean"
    ]
    df_traces["down_modulated_sem"] = combined_cell_data["down_modulated"][
        "sem"
    ]
    df_traces["non_modulated_mean"] = combined_cell_data["non_modulated"][
        "mean"
    ]
    df_traces["non_modulated_sem"] = combined_cell_data["non_modulated"]["sem"]

    # generate subpopulation activity plot
    subpopulation_activity_preview_filename = os.path.join(
        output_dir,
        f"event_aligned_activity_by_modulation_{group_name}".replace(" ", "_")
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    generate_subpopulation_activity_plot(
        traces_timeline=traces_timeline,
        modulated_cells_dict=combined_cell_data,
        output_filename=subpopulation_activity_preview_filename,
        modulation_colors=modulation_colors,
    )

    # generate pie chart showing fraction of up/down/non-modulated neurons
    pie_chart_modulation_preview_filename = os.path.join(
        output_dir,
        f"fraction_of_modulated_neurons_{group_name}".replace(" ", "_")
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    generate_modulation_pie_chart(
        num_cells_up_modulated=combined_cell_data["up_modulated"]["num_cells"],
        num_cells_down_modulated=combined_cell_data["down_modulated"][
            "num_cells"
        ],
        num_cells_non_modulated=combined_cell_data["non_modulated"][
            "num_cells"
        ],
        output_filename=pie_chart_modulation_preview_filename,
        modulation_colors=modulation_colors,
    )

    # save traces data to disk
    output_traces_filename = os.path.join(
        output_dir,
        f"event_aligned_activity_{group_name}.csv".replace(" ", "_"),
    )

    # reorder columns and save to csv
    df_traces = pd.concat(
        [df_traces.iloc[:, -9:], df_traces.iloc[:, :-9]], axis=1
    )
    df_traces.to_csv(output_traces_filename, index=False)

    logger.info(
        f"Event-aligned traces saved "
        f"({os.path.basename(output_traces_filename)}, "
        f"size: {get_file_size(output_traces_filename)})"
    )

    # construct output manifest entries for the combined data
    # PREVIEWS
    # population_activity_preview_file = IdeasPreviewFile(
    #     name="Event-aligned population activity figure",
    #     help="Event-aligned average population activity line plot",
    #     file_path=os.path.abspath(population_activity_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # activity_heatmap_preview_file = IdeasPreviewFile(
    #     name="Event-aligned single-cell activity figure",
    #     help="Event-aligned single-cell activity heatmap",
    #     file_path=os.path.abspath(single_cell_heatmap_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # activity_by_modulation_preview_file = IdeasPreviewFile(
    #     name="Event-aligned sub-population activity figure",
    #     help="Event-aligned average sub-population activity line plot "
    #     "(up-, down-, and non-modulated neurons)",
    #     file_path=os.path.abspath(subpopulation_activity_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # fraction_neurons_per_subpopulation_preview_file = IdeasPreviewFile(
    #     name="Fraction of neurons in each sub-population",
    #     help="Pie chart depicting the fraction of neurons in each sub-population "
    #     "(up-, down-, and non-modulated neurons)",
    #     file_path=os.path.abspath(pie_chart_modulation_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # # define event-aligned metadata
    event_aligned_metadata = [
        {
            "key": "ideas.metrics.num_up_modulated_cells",
            "name": "Number of up-modulated cells",
            "value": combined_cell_data["up_modulated"][
                "num_cells"
            ]
        },
        {
            "key": "ideas.metrics.num_down_modulated_cells",
            "name": "Number of down-modulated cells",
            "value": combined_cell_data["down_modulated"][
                "num_cells"
            ]
        },
        {
            "key": "ideas.metrics.num_non_modulated_cells",
            "name": "Number of non-modulated cells",
            "value": combined_cell_data["non_modulated"][
                "num_cells"
            ]
        },
        {
            "key": "ideas.timingInfo.numTime",
            "name": "Number of timepoints",
            "value": len(traces_timeline)
        },
        {
            "key": "ideas.timingInfo.sampling_rate",
            "name": "Sampling Rate (Hz)",
            "value": compute_sampling_rate(
                period_num=abs(traces_timeline[0] - traces_timeline[1]),
                period_den=1,
            )
        }
    ]

    # event_aligned_metadata = {
    #     config.IDEAS_METADATA_KEY: {
    #         "metrics": {
    #             "num_up_modulated_cells": combined_cell_data["up_modulated"][
    #                 "num_cells"
    #             ],
    #             "num_down_modulated_cells": combined_cell_data[
    #                 "down_modulated"
    #             ]["num_cells"],
    #             "num_non_modulated_cells": combined_cell_data["non_modulated"][
    #                 "num_cells"
    #             ],
    #         },
    #         "timingInfo": {
    #             "numTimes": len(traces_timeline),
    #             "sampling_rate": compute_sampling_rate(
    #                 period_num=abs(traces_timeline[0] - traces_timeline[1]),
    #                 period_den=1,
    #             ),
    #         },
    #     }
    # }

    # event-aligned traces FILE
    # event_aligned_traces_file = IdeasFile(
    #     file_key=f"group{group_id}_event_aligned_activity"
    #     f"_traces_event_aligned_activity_traces_csv_file",
    #     file_path=os.path.abspath(output_traces_filename),
    #     file_type=FileType.COMBINED_PERI_EVENT_TRACES.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TIME_SERIES.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     preview_files=[
    #         population_activity_preview_file,
    #         activity_heatmap_preview_file,
    #     ],
    #     add_metadata=event_aligned_metadata,
    # )

    # # define statistics metadata
    # statistics_metadata = {
    #     config.IDEAS_METADATA_KEY: {
    #         "metrics": {
    #             "num_up_modulated_cells": combined_cell_data["up_modulated"][
    #                 "num_cells"
    #             ],
    #             "num_down_modulated_cells": combined_cell_data[
    #                 "down_modulated"
    #             ]["num_cells"],
    #             "num_non_modulated_cells": combined_cell_data["non_modulated"][
    #                 "num_cells"
    #             ],
    #         },
    #         "timingInfo": {
    #             "numTimes": len(traces_timeline),
    #             "sampling_rate": compute_sampling_rate(
    #                 period_num=abs(traces_timeline[0] - traces_timeline[1]),
    #                 period_den=1,
    #             ),
    #         },
    #     }
    # }

    # statistics FILE
    # statistics_file = IdeasFile(
    #     file_key=f"group{group_id}_event_aligned_statistics_event"
    #     f"_aligned_activity_statistics_csv_file",
    #     file_path=os.path.abspath(output_stats_filename),
    #     file_type=FileType.COMBINED_PERI_EVENT_STATISTICS.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TABLE.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     preview_files=[
    #         activity_by_modulation_preview_file,
    #         fraction_neurons_per_subpopulation_preview_file,
    #     ],
    #     add_metadata=statistics_metadata,
    # )

    return combined_cell_data, event_aligned_metadata


def generate_comparison_plots(
    group1_data: dict,
    group1_name: str,
    group2_data: dict,
    group2_name: str,
    output_dir: str,
    group_colors=None,
):
    """Generate plots to compare data from groups 1 and 2.

    :param group1_data: data dictionary for the first group
    :param group1_name: name of the first group
    :param group2_data: data dictionary for the first group
    :param group2_name: name of the second group
    :param output_dir: path to the output directory
    :param group_colors: comma separated strings with color inputs
    These colors represent the two comparison groups.
    """
    group_colors = validate_group_colors(group_colors)

    # Avg Pop comparison
    avg_population_comparison_preview_filename = os.path.join(
        output_dir,
        "event_aligned_population_activity_comparison"
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    plot_comparison_of_population_activity(
        group1_data=group1_data,
        group1_name=group1_name,
        group2_data=group2_data,
        group2_name=group2_name,
        output_filename=avg_population_comparison_preview_filename,
        group_colors=group_colors,
    )
    # avg_population_comparison_preview_file = IdeasPreviewFile(
    #     name="Event-aligned population activity figure",
    #     help="Event-aligned average population activity line plot",
    #     file_path=os.path.abspath(avg_population_comparison_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # Sup Pop comparison (one subplot for each modulation group)
    sub_population_comparison_preview_filename = os.path.join(
        output_dir,
        "event_aligned_subpopulation_activity_comparison"
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    plot_comparison_of_subpopulation_activity(
        group1_data=group1_data,
        group1_name=group1_name,
        group2_data=group2_data,
        group2_name=group2_name,
        output_filename=sub_population_comparison_preview_filename,
        group_colors=group_colors,
    )
    # sub_population_comparison_preview_file = IdeasPreviewFile(
    #     name="Event-aligned sub-population activity figure",
    #     help="Event-aligned average sub-population activity line plot",
    #     file_path=os.path.abspath(sub_population_comparison_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # Barplot (up, down, non)
    fractions_comparison_preview_filename = os.path.join(
        output_dir,
        "modulated_fractions_comparison"
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
    )
    plot_fractions_of_modulated_neurons(
        group1_data=group1_data,
        group1_name=group1_name,
        group2_data=group2_data,
        group2_name=group2_name,
        output_filename=fractions_comparison_preview_filename,
        group_colors=group_colors,
    )
    # fractions_comparison_preview_file = IdeasPreviewFile(
    #     name="Fraction of neurons in each modulation group",
    #     help="Fraction of neurons in each modulation group",
    #     file_path=os.path.abspath(fractions_comparison_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # return (
    #     avg_population_comparison_preview_file,
    #     sub_population_comparison_preview_file,
    #     fractions_comparison_preview_file,
    # )


def combine_compare_peri_event_data(
    group1_traces_files: List[str],
    group1_stats_files: List[str],
    group1_name: str,
    group2_traces_files: List[str],
    group2_stats_files: List[str],
    group2_name: str,
    comparison_type: str,
    data_pairing: str,
    significance_threshold: float,
    average_method: str,
    tolerance: float,
    group_colors: str = "#1f77b4, #ff7f0e",
    modulation_colors: str = "green, blue, black",
    cmap: str = "coolwarm",
    population_activity_plot_limits: str = "auto",
    activity_heatmap_color_limits: str = "auto",
    activity_by_modulation_plot_limits: str = "auto",
    output_dir: str = None,
):
    """Combine and compare peri-event analysis data.

    :param group1_traces_files: peri-event analysis traces files from the first group
    :param group1_stats_files: peri-event analysis statistics files from the first group
    :param group1_name: name of the first group
    :param group2_traces_files: peri-event analysis traces files from the second group
    :param group2_stats_files: peri-event analysis statistics files from the second group
    :param group2_name: name of the second group
    :param comparison_type: type of statistical test to perform
    :param data_pairing: indicates whether observations should be paired for
                         statistical comparison
    :param significance_threshold: p-value threshold for classifying neurons as
                                   up- or down-modulated
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording
    :param group_colors: comma separated strings with color inputs
    These colors represent the two comparison groups.
    :param modulation_colors: comma separated strings with color inputs
    :param cmap: colormap applied to the activity heatmap
    :param population_activity_plot_limits: y-axis range (z-score) applied to the event-aligned
     population activity plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_heatmap_color_limits: colormap range (z-score) applied to the activity heatmap
     specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_by_modulation_plot_limits: y-axis range (z-score) applied to the event-aligned
     activity by modulation plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param output_dir: path to the output directory
    """
    logger.info("Starting the combination and comparison of peri-event data")

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

    # validate input parameters
    validate_combine_compare_peri_event_data_parameters(
        group1_traces_files=group1_traces_files,
        group1_stats_files=group1_stats_files,
        group1_name=group1_name,
        group2_traces_files=group2_traces_files,
        group2_stats_files=group2_stats_files,
        group2_name=group2_name,
        comparison_type=comparison_type,
        data_pairing=data_pairing,
        significance_threshold=significance_threshold,
        tolerance=tolerance,
        average_method=average_method,
        population_activity_plot_limits=population_activity_plot_limits,
        activity_heatmap_color_limits=activity_heatmap_color_limits,
        activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
    )

    group_colors = validate_group_colors(group_colors)

    # remove unsupported characters from group names
    unsupported_chars = ["$"]
    placeholder_char = "�"
    for c in unsupported_chars:
        if c in group1_name:
            group1_name = group1_name.replace(c, placeholder_char)
            logger.warning(
                f"Invalid character '{c}' found in first "
                f"group name replaced with '{placeholder_char}'"
            )
        if group2_name is not None and c in group2_name:
            group2_name = group2_name.replace(c, placeholder_char)
            logger.warning(
                f"Invalid character '{c}' found in second "
                f"group name replaced with '{placeholder_char}'"
            )

    # store processing params globally
    global PROCESSING_PARAMS
    PROCESSING_PARAMS["average_method"] = average_method

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

    if group2_traces_files is None:
        group2_traces_files = []

    if group2_stats_files is None:
        group2_stats_files = []

    # set output directory to the current working directory if unspecified
    if output_dir is None:
        output_dir = os.getcwd()

    # define list of files to include in the output manifest
    output_files = []
    output_metadata = {}

    # combine data from group 1
    if len(group1_traces_files) > 1:
        # (
        #     group1_data,
        #     # group1_traces_manifest_files,
        #     # group1_statistics_manifest_files,
        # ) 
        group1_data, group1_md = combine_peri_event_data(
            traces_files=group1_traces_files,
            stats_files=group1_stats_files,
            significance_threshold=significance_threshold,
            group_name=group1_name,
            output_dir=output_dir,
            group_id=1,
            average_method=average_method,
            tolerance=tolerance,
            color=group_colors[0],
            modulation_colors=modulation_colors,
            cmap=cmap,
        )
        output_metadata["group1_event_aligned_activity_traces"] = group1_md
        output_metadata["group1_event_aligned_statistics"] = group1_md

        # output_files.extend(
        #     [group1_traces_manifest_files, group1_statistics_manifest_files]
        # )
        logger.info("Data from the first input group combined")

    # combine data from group 2
    if len(group2_traces_files) > 1:
        # (
        #     group2_data,
        #     # group2_traces_manifest_files,
        #     # group2_statistics_manifest_files,
        # ) = 
        group2_data, group2_md = combine_peri_event_data(
            traces_files=group2_traces_files,
            stats_files=group2_stats_files,
            significance_threshold=significance_threshold,
            group_name=group2_name,
            output_dir=output_dir,
            group_id=2,
            average_method=average_method,
            tolerance=tolerance,
            color=group_colors[1],
            modulation_colors=modulation_colors,
            cmap=cmap,
        )

        output_metadata["group2_event_aligned_activity_traces"] = group2_md
        output_metadata["group2_event_aligned_statistics"] = group2_md
        # output_files.extend(
        #     [group2_traces_manifest_files, group2_statistics_manifest_files]
        # )
        logger.info("Data from the second input group combined")

    # compare groups 1 and 2
    if len(group1_traces_files) > 1 and len(group2_traces_files) > 1:
        stats_dict = {
            "up_modulated": {},
            "down_modulated": {},
            "non_modulated": {},
        }
        for modulation_group in [
            "up_modulated",
            "down_modulated",
            "non_modulated",
        ]:
            t_statistic, p_value = ttest(
                group1_data=group1_data[modulation_group]["fractions"],
                group2_data=group2_data[modulation_group]["fractions"],
                comparison_type=comparison_type,
                data_pairing=data_pairing,
            )
            stats_dict[modulation_group]["t_statistic"] = t_statistic
            stats_dict[modulation_group]["p_value"] = p_value

        logger.info("Data from the input groups compared")

        # save comparison data to disk
        comparison_dicts = []
        for modulation_group in [
            "up_modulated",
            "down_modulated",
            "non_modulated",
        ]:
            comparison_dicts.append(
                {
                    "modulation": modulation_group,
                    f"n_{group1_name}": len(
                        group1_data["up_modulated"]["fractions"]
                    ),
                    f"n_{group2_name}": len(
                        group2_data["up_modulated"]["fractions"]
                    ),
                    "t_statistic": stats_dict[modulation_group]["t_statistic"],
                    "p_value": stats_dict[modulation_group]["p_value"],
                }
            )
        df_comparison = pd.DataFrame(comparison_dicts)

        output_comparison_filename = os.path.join(
            output_dir, "comparison_data.csv"
        )
        df_comparison.to_csv(output_comparison_filename, index=False)

        logger.info(
            f"Comparison data saved "
            f"({os.path.basename(output_comparison_filename)}, "
            f"size: {get_file_size(output_comparison_filename)})"
        )

        # plot comparison data
        # (
        #     avg_population_comparison_preview_file,
        #     sub_population_comparison_preview_file,
        #     fractions_comparison_preview_file,
        # ) = 
        generate_comparison_plots(
            group1_data=group1_data,
            group1_name=group1_name,
            group2_data=group2_data,
            group2_name=group2_name,
            output_dir=output_dir,
            group_colors=group_colors,
        )
        logger.info("Comparison plots generated")

        # # define comparison file metadata
        output_metadata["comparison_data"] = [
            group1_md[3], group1_md[4]
        ]
        # comparison_metadata = {
        #     config.IDEAS_METADATA_KEY: {
        #         "timingInfo": group1_traces_manifest_files.file_metadata[
        #             "add"
        #         ]["ideas"]["timingInfo"]
        #     }
        # }

    #     # add comparison data and plots to output manifest
    #     comparison_data_file = IdeasFile(
    #         file_key="comparison_data_comparison_data_csv_file",
    #         file_path=os.path.abspath(output_comparison_filename),
    #         file_type=FileType.PERI_EVENT_COMPARISON_DATA.value[1],
    #         file_format=FileFormat.CSV_FILE.value[1],
    #         file_structure=FileStructure.TABLE.value[1],
    #         file_category=FileCategory.RESULT.value[1],
    #         preview_files=[
    #             avg_population_comparison_preview_file,
    #             sub_population_comparison_preview_file,
    #             fractions_comparison_preview_file,
    #         ],
    #         add_metadata=comparison_metadata,
    #     )

    #     output_files.append(comparison_data_file)

    # # generate output manifest & metadata
    # generate_output_manifest(
    #     group1_traces_files=group1_traces_files,
    #     group1_stats_files=group1_stats_files,
    #     group2_traces_files=group2_traces_files,
    #     group2_stats_files=group2_stats_files,
    #     output_files=output_files,
    #     output_dir=output_dir,
    # )

    
    with open(os.path.join(output_dir, "output_metadata.json"), "w") as f:
        json.dump(output_metadata, f)

    logger.info("Combination and comparison of peri-event data completed")


def combine_compare_peri_event_data_ideas_wrapper(
    group1_traces_files: List[IdeasFile],
    group1_stats_files: List[IdeasFile],
    group1_name: str,
    group2_traces_files: List[IdeasFile],
    group2_stats_files: List[IdeasFile],
    group2_name: str,
    comparison_type: str,
    data_pairing: str,
    significance_threshold: float,
    average_method: str,
    tolerance: float,
    group_colors: str = "#1f77b4, #ff7f0e",
    modulation_colors: str = "green, blue, black",
    cmap: str = "coolwarm",
    population_activity_plot_limits: str = "auto",
    activity_heatmap_color_limits: str = "auto",
    activity_by_modulation_plot_limits: str = "auto",
):
    combine_compare_peri_event_data(
        group1_traces_files=group1_traces_files,
        group1_stats_files=group1_stats_files,
        group1_name=group1_name,
        group2_traces_files=group2_traces_files,
        group2_stats_files=group2_stats_files,
        group2_name=group2_name,
        comparison_type=comparison_type,
        data_pairing=data_pairing,
        significance_threshold=significance_threshold,
        average_method=average_method,
        tolerance=tolerance,
        group_colors=group_colors,
        modulation_colors=modulation_colors,
        cmap=cmap,
        population_activity_plot_limits=population_activity_plot_limits,
        activity_heatmap_color_limits=activity_heatmap_color_limits,
        activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        with outputs.register(raise_missing_file=False) as output_data:
            for group_name in [group1_name, group2_name]:
                group_name = group_name.replace(" ", "_")
                subdir_base = "group1" if group_name == group1_name else "group2"
                
                output_file = output_data.register_file(
                    f"event_aligned_activity_{group_name}.csv",
                    subdir=f"{subdir_base}_event_aligned_activity_traces",
                ).register_preview(
                    f"event_aligned_single_cell_activity_heatmap_{group_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                    caption="Event-aligned single-cell activity heatmap"
                ).register_preview(
                    f"event_aligned_population_activity_{group_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                    caption="Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons)"
                )
                for md in metadata.get(f"{subdir_base}_event_aligned_activity_traces", {}):
                    output_file.register_metadata(**md) 


                output_file = output_data.register_file(
                    f"event_aligned_statistics_{group_name}.csv",
                    subdir=f"{subdir_base}_event_aligned_statistics",
                ).register_preview(
                    f"event_aligned_activity_by_modulation_{group_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                    caption="Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons)",
                ).register_preview(
                    f"fraction_of_modulated_neurons_{group_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                    caption="Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons)",
                )
                for md in metadata.get(f"{subdir_base}_event_aligned_statistics", {}):
                    output_file.register_metadata(**md)

            output_file = output_data.register_file(
                "comparison_data.csv",
                subdir="comparison_data"
            ).register_preview(
                f"modulated_fractions_comparison{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                caption="Fraction of neurons in each modulation group"
            ).register_preview(
                f"event_aligned_subpopulation_activity_comparison{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                caption="Event-aligned average sub-population activity line plot",
            ).register_preview(
                f"event_aligned_population_activity_comparison{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                caption="Event-aligned average population activity line plot",
            )
            for md in metadata.get("comparison_data", {}):
                output_file.register_metadata(**md)
        
        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")
    
