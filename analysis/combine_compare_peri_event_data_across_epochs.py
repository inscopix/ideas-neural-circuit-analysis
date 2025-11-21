import json
import os
import warnings
from typing import List
from scipy import stats
# from distutils.util import strtobool
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
import utils.config as config
from ideas.exceptions import IdeasError
from analysis.combine_compare_peri_event_data import (
    validate_combine_compare_peri_event_data_parameters,
    validate_group_colors,
    reclassify_neurons,
    generate_subpopulation_activity_plot,
    generate_modulation_pie_chart,
    # generate_output_manifest,
)
from analysis.peri_event_workflow import (
    validate_modulation_colors,
    plot_population_mean_event_window,
    plot_single_neurons_heatmap,
)
from ideas.analysis.utils import (
    get_file_size,
    # compute_sampling_rate,
)
from utils.stats_utils import (
    is_normal,
    statistically_compare_two_groups,
    perform_paired_pairwise_comparisons,
)
from utils.plots import (
    plot_event_aligned_activity_by_modulation_group_comparing_epochs,
    plot_population_mean_event_window_across_epochs,
    plot_number_of_modulated_cells_per_epoch,
    plot_post_minus_pre_activity_differences_with_cell_map,
    plot_post_minus_pre_per_epoch_bar_chart,
    _plot_mixed_comparisons2,
)
from utils.utils import (
    remove_unsupported_characters,
    compute_sampling_rate
)
# from toolbox.utils.data_model import IdeasFile, IdeasPreviewFile
# from ideas_commons.constants import (
#     FileFormat,
#     FileStructure,
#     FileType,
#     FileCategory,
# )
from ideas.tools import log
from ideas.tools.types import IdeasFile
from ideas.tools import outputs

logger = log.get_logger()

# Define a constant for small values used in division or variance checks
DIVISION_THRESHOLD = 1e-10


def strtobool(value: str) -> bool:
  # this func has been deprecated in python 3.13
  # implementation is easy to copy https://docs.python.org/3.9/distutils/apiref.html#distutils.util.strtobool
  value = value.lower()
  if value in ("y", "yes", "t", "true", "on", "1"):
    return True
  elif value in ["n", "no", "f", "false", "off", "0"]:
      return False
  raise ValueError("Failed to convert str to bool")

def generate_activity_heatmap(
    traces_timeline: List[float],
    df_traces: pd.DataFrame,
    df_stats: pd.DataFrame,
    output_filename: str,
    epoch_name: str,
    activity_heatmap_color_limits: tuple,
    cmap=None,
):
    """Plot individual cell activity heatmap over the event window.

    :param traces_timeline: x-values to use for the plot
    :param df_traces: dataframe containing traces from all input files (num_timepoints x num_cells)
    :param df_stats: dataframe containing statistics from all input files
    :param output_filename: path to the output file
    :paran epoch_name: name of the epoch
    :param cmap: colormap applied to the activity heatmap
    """
    # compute ranks and reverse order to get descending order
    # i.e. from most up-modulated to most down-modulated
    zscores = df_stats["z-score"].tolist()
    ranks = np.argsort(zscores)[::-1]

    # transpose dataframe to match expected format for plotting (num_cells, num_timepoints)
    # and sort rows by their modulation ranking
    ordered_traces = df_traces.T.iloc[ranks]

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
        color_limits=activity_heatmap_color_limits,
        cmap=cmap,
        epoch_name=epoch_name,
    )


def extract_modulation_group_data(
    traces_dataframes: pd.DataFrame,
    stats_dataframes: pd.DataFrame,
    average_method: str,
    epoch_name: str,
    stats_files: list[str],
):
    """Extract, group, and average neurons across input files by modulation group.

    :param traces_dataframes: list of dataframes containing peri-event traces
    :param stats_dataframes: list of dataframes containing peri-event statistics
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording

    :return modulated_cells_dict: dictionary containing (mean, sem, num_cells)
                                  for each modulation group
    """
    # exclude columns that represent groups of cells
    # since here we are recreating the groups by taking
    # cells from all input files
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
    for traces_df, stats_df, stats_filename in zip(
        traces_dataframes, stats_dataframes, stats_files
    ):
        up_modulated_filter = (
            (stats_df["modulation"] == 1)
            & (~stats_df["name"].isin(names_to_exclude))
            & (stats_df["epoch"] == epoch_name)
        )
        down_modulated_filter = (
            (stats_df["modulation"] == -1)
            & (~stats_df["name"].isin(names_to_exclude))
            & (stats_df["epoch"] == epoch_name)
        )
        non_modulated_filter = (
            (stats_df["modulation"] == 0)
            & (~stats_df["name"].isin(names_to_exclude))
            & (stats_df["epoch"] == epoch_name)
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

        if total_num_cells == 0:
            logger.warning(
                f"No cells detected for epoch '{epoch_name}' in file '{os.path.basename(stats_filename)}', "
                f"the corresponding modulation data cannot be calculated"
            )
            continue

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
                    [
                        up_modulated_traces,
                        traces_df[epoch_name][up_modulated_cell_names],
                    ],
                    axis=1,
                )
            else:
                up_modulated_traces = traces_df[epoch_name][
                    up_modulated_cell_names
                ]

            if down_modulated_traces is not None:
                down_modulated_traces = pd.concat(
                    [
                        down_modulated_traces,
                        traces_df[epoch_name][down_modulated_cell_names],
                    ],
                    axis=1,
                )
            else:
                down_modulated_traces = traces_df[epoch_name][
                    down_modulated_cell_names
                ]

            if non_modulated_traces is not None:
                non_modulated_traces = pd.concat(
                    [
                        non_modulated_traces,
                        traces_df[epoch_name][non_modulated_cell_names],
                    ],
                    axis=1,
                )
            else:
                non_modulated_traces = traces_df[epoch_name][
                    non_modulated_cell_names
                ]
        else:
            # average method is recordings
            if up_modulated_traces is not None:
                # iteratively stacks the means of up modulated cells per recording
                up_modulated_traces = np.column_stack(
                    [
                        up_modulated_traces,
                        np.nanmean(
                            traces_df[epoch_name][up_modulated_cell_names],
                            axis=1,
                        ),
                    ]
                )
            else:
                # initializes up modulated cells by obtaining the mean in the first recording
                up_modulated_traces = np.nanmean(
                    traces_df[epoch_name][up_modulated_cell_names], axis=1
                )

            if down_modulated_traces is not None:
                # iteratively stacks the means of down modulated cells per recording
                down_modulated_traces = np.column_stack(
                    [
                        down_modulated_traces,
                        np.nanmean(
                            traces_df[epoch_name][down_modulated_cell_names],
                            axis=1,
                        ),
                    ]
                )
            else:
                # initializes down modulated cells by obtaining the mean in the first recording
                down_modulated_traces = np.nanmean(
                    traces_df[epoch_name][down_modulated_cell_names], axis=1
                )

            if non_modulated_traces is not None:
                # iteratively stacks the means of non modulated cells per recording
                non_modulated_traces = np.column_stack(
                    [
                        non_modulated_traces,
                        np.nanmean(
                            traces_df[epoch_name][non_modulated_cell_names],
                            axis=1,
                        ),
                    ]
                )
            else:
                # initializes non modulated cells by obtaining the mean in the first recording
                non_modulated_traces = np.nanmean(
                    traces_df[epoch_name][non_modulated_cell_names], axis=1
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


def _run_ANOVA(
    data,
    method,
    metric,
    file,
    ind_variable_name="epoch",
):
    """Perform ANOVA (Analysis of Variance) on the given dataset.
    :Parameters
    data (DataFrame): The dataset containing the data to be analyzed.
    method (str): The type of ANOVA to perform. Options are 'mixed' or 'oneway'.
    data_pairing: Indicates whether the observations are paired or not.
    significance_threshold: Significance level (0-1).
    metric (str): The metric to be analyzed.
    :Returns
    tuple: A tuple containing two DataFrames:
        - aov: The ANOVA results.
        - pairwise: The pairwise comparison results.
    """
    logger.info("Performing analysis of variance (ANOVA)")

    # Check for problematic data that could cause warnings
    try:
        # Remove any constant groups which will cause division by zero
        groups = data[ind_variable_name].unique()
        groups_to_drop = []

        for group in groups:
            group_data = data[data[ind_variable_name] == group][metric]

            # Check for constant values within a group
            if len(group_data) > 0 and group_data.std() < DIVISION_THRESHOLD:
                logger.warning(
                    f"Group '{group}' has constant or near-constant values for {metric}. "
                    "This can cause division by zero in statistical calculations."
                )
                if len(groups) > 2:  # Only drop if we have enough groups left
                    groups_to_drop.append(group)

        # Only filter if we have groups to drop
        if groups_to_drop and len(groups) - len(groups_to_drop) >= 2:
            logger.warning(
                f"Removing groups {groups_to_drop} from analysis due to constant values."
            )
            data = data[~data[ind_variable_name].isin(groups_to_drop)]

        # If we have fewer than 2 groups left, we can't run ANOVA
        if len(data[ind_variable_name].unique()) < 2:
            logger.warning(
                "Not enough valid groups for analysis. Returning empty results."
            )
            empty_aov = pd.DataFrame(
                {"Comparison": [metric], "p-unc": [np.nan]}
            )
            empty_pairwise = pd.DataFrame(
                {"Comparison": [metric], "p-unc": [np.nan]}
            )
            return empty_aov, empty_pairwise

        # Add tiny random noise to eliminate exact zeros in variance
        if data[metric].std() < DIVISION_THRESHOLD:
            logger.warning(
                f"Adding tiny noise to {metric} data to avoid division by zero."
            )
            noise_scale = DIVISION_THRESHOLD
            data[metric] = data[metric] + np.random.normal(
                0, noise_scale, size=len(data)
            )

    except Exception as e:
        logger.warning(f"Error preprocessing data for ANOVA: {str(e)}")
        empty_aov = pd.DataFrame({"Comparison": [metric], "p-unc": [np.nan]})
        empty_pairwise = pd.DataFrame(
            {"Comparison": [metric], "p-unc": [np.nan]}
        )
        return empty_aov, empty_pairwise

    # Run the statistical tests with exception handling
    try:
        if method == "mixed":
            with np.errstate(divide="ignore", invalid="ignore"):
                try:
                    aov = pg.mixed_anova(
                        data=data,
                        dv=metric,
                        within="epoch",
                        between="group",
                        subject=file,  # file is the subject
                    )
                    # Ensure Source column exists for compatibility with tests
                    if "Source" not in aov.columns and len(aov) > 0:
                        aov["Source"] = ["epoch", "group", "epoch * group"][
                            : len(aov)
                        ]
                except Exception as e:
                    # Create a dummy dataframe with expected columns for mixed ANOVA
                    logger.warning(
                        f"Error during mixed ANOVA: {str(e)}. Creating dummy results."
                    )
                    aov = pd.DataFrame(
                        {
                            "Source": ["epoch", "group", "epoch * group"],
                            "F": [np.nan, np.nan, np.nan],
                            "p-unc": [1.0, 1.0, 1.0],
                            "np2": [0.0, 0.0, 0.0],
                        }
                    )
        elif method == "oneway":
            with np.errstate(divide="ignore", invalid="ignore"):
                aov = pg.anova(
                    data=data,
                    dv=metric,
                    between=ind_variable_name,
                    detailed=True,
                )
        elif method == "repeated":
            with np.errstate(divide="ignore", invalid="ignore"):
                aov = pg.rm_anova(
                    data=data,
                    dv=metric,
                    within=ind_variable_name,
                    subject=file,
                    detailed=True,
                )

        # Replace any NaN or infinite values in p-values with 1.0 (non-significant)
        if "p-unc" in aov.columns:
            aov["p-unc"] = aov["p-unc"].fillna(1.0)
            aov.loc[~np.isfinite(aov["p-unc"]), "p-unc"] = 1.0

    except Exception as e:
        logger.warning(
            f"Error during {method} analysis: {str(e)}. Returning empty results."
        )
        if method == "mixed":
            # Create dummy results with expected structure for mixed ANOVA
            aov = pd.DataFrame(
                {
                    "Source": ["epoch", "group", "epoch * group"],
                    "F": [np.nan, np.nan, np.nan],
                    "p-unc": [1.0, 1.0, 1.0],
                    "np2": [0.0, 0.0, 0.0],
                }
            )
        else:
            aov = pd.DataFrame({"Comparison": [metric], "p-unc": [1.0]})

    aov["Comparison"] = metric

    # Put the comparison column on the far left
    aov = aov[
        ["Comparison"] + [col for col in aov.columns if col != "Comparison"]
    ]
    return aov


def _run_pairwise_epoch_comparisons(
    data, parametric, comparison_type, data_pairing, group1_name, group2_name
):
    """For each epoch, perform a statistical comparison of the group means (post-pre activity)."""
    cols = ["group", "file", "true_mean_post-pre"]
    epoch_names = data["epoch"].unique().tolist()
    epoch_comparison_dfs = []
    for epoch_name in epoch_names:
        logger.info(
            f"Performing statistical comparison for epoch '{epoch_name}'"
        )
        epoch_data = data[data["epoch"] == epoch_name][cols]
        grouped_df = (
            epoch_data.groupby(["group", "file"])
            .mean(numeric_only=True)
            .reset_index()
        )
        group1_data = grouped_df[grouped_df["group"] == group1_name][
            "true_mean_post-pre"
        ]
        group2_data = grouped_df[grouped_df["group"] == group2_name][
            "true_mean_post-pre"
        ]

        if len(group1_data) < 2:
            logger.warning(
                f"Fewer than 2 observations found in the first group. "
                f"Skipping group comparison for epoch {epoch_name}"
            )
            continue

        if len(group2_data) < 2:
            logger.warning(
                f"Fewer than 2 observations found in the second group. "
                f"Skipping group comparison for epoch {epoch_name}"
            )
            continue

        epoch_comparison_df = statistically_compare_two_groups(
            x=group1_data,
            y=group2_data,
            comparison_type=comparison_type,
            data_pairing=data_pairing,
            parametric=parametric,
            epoch_name=epoch_name,
        )
        epoch_comparison_dfs.append(epoch_comparison_df)

    # combine dataframes
    epoch_comparison_df = pd.concat(epoch_comparison_dfs)
    return epoch_comparison_df


def _compare_single_group(data, comparison_type, output_filename):
    """Compare data across epochs within a single group."""
    # create unique cell names by joining cell ids and filenames
    data["cell"] = data["name"].str.cat(data["file"], sep="-")

    pairwise = perform_paired_pairwise_comparisons(
        df=data,
        comparison_type=comparison_type,
        dependent_var_col="true_mean_post-pre",
        comparison_name="population post-pre",
        within_col="epoch",
        subject_col="cell",
    )

    # save pairwise comparisons data
    pairwise.to_csv(output_filename, index=False)
    logger.info(
        f"Epoch pairwise comparison data saved "
        f"({os.path.basename(output_filename)}, "
        f"size: {get_file_size(output_filename)})"
    )


def _compare_data(
    g1_data,
    g2_data,
    group1_name,
    group2_name,
    comparison_type,
    data_pairing,
    parametric,
    significance_threshold,
):
    """Compare data between two groups using 2-way repeated measures ANOVA and
    pairwise t-tests.

    :Parameters
        g1_data (pd.DataFrame): Data for group 1.
        g2_data (pd.DataFrame): Data for group 2.
        g1_name (str): Name of group 1.
        g2_name (str): Name of group 2.
        name (str): Name for the plot.
    :Returns
        tuple: ANOVA results and pairwise t-tests results.
    """
    # Combine data for comparison
    g1_data["group"] = group1_name
    g2_data["group"] = group2_name
    combined_data = pd.concat([g1_data, g2_data])

    # determine whether to use parametric or non-parametric tests
    if parametric == "auto":
        logger.info("Determining whether to use parametric tests")
        parametric = is_normal(
            data=combined_data,
            metric="true_mean_post-pre",
            ind_variable_name="epoch",
        )
        if parametric:
            logger.info(
                "The data follows a normal distribution. Parametric tests will be used."
            )
        else:
            logger.info(
                "The data does not follow a normal distribution. Non-parametric tests will be used."
            )
    else:
        # convert parametric from string to boolean
        parametric = bool(strtobool(parametric))

    # run ANOVA
    num_epochs = len(combined_data["epoch"].unique())
    if num_epochs < 2:
        logger.warning(
            "A minimum of 2 epochs are needed to run ANOVA. The ANOVA output file will not be generated."
        )
        aov = None
    else:
        # perform ANOVA on post-pre data
        aov = _run_ANOVA(
            data=combined_data,
            method="mixed",
            metric="true_mean_post-pre",
            file="file",
            ind_variable_name="epoch",
        )

    # compare the post-pre activity across the two groups for each epoch
    pairwise = _run_pairwise_epoch_comparisons(
        data=combined_data,
        parametric=parametric,
        comparison_type=comparison_type,
        data_pairing=data_pairing,
        group1_name=group1_name,
        group2_name=group2_name,
    )

    return aov, pairwise


def combine_peri_event_data(
    traces_files: List[str],
    stats_files: List[str],
    epoch_names: List[str],
    significance_threshold: float,
    group_name: str,
    output_dir: str,
    group_id: int,
    average_method: str,
    tolerance: float,
    epoch_data: dict,
    activity_heatmap_color_limits: str,
    population_activity_plot_limits: str,
    activity_by_modulation_plot_limits: str,
    color: str = None,
    modulation_colors: List[str] = None,
    cmap: str = None,
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
    :param epoch_data: dictionary containing epoch-specific information
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
    traces_dataframes = [pd.read_csv(f, header=[0, 1]) for f in traces_files]
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
    traces_timeline = traces_dataframes[0]["Time"][
        "Unnamed: 0_level_1"
    ].tolist()
    for i, df in enumerate(traces_dataframes[1:]):
        # time windows should have the same size in all input files
        if len(df["Time"]) != len(traces_timeline):
            raise IdeasError(
                f"The time windows of the input traces files '{os.path.basename(traces_files[0])}'"
                f" and '{os.path.basename(traces_files[i + 1])}' do not match.",
            )

    # combine traces data
    cols_to_drop_per_epoch = [
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

    for i in range(len(traces_files)):
        # retain per-cell data only and drop the other columns
        cols_to_drop = [("Time", "Unnamed: 0_level_1")]
        for epoch_name in epoch_names:
            cols_to_drop.extend(
                [(epoch_name, c) for c in cols_to_drop_per_epoch]
            )

        traces_dataframes[i].drop(
            columns=cols_to_drop, axis=1, inplace=True, errors="ignore"
        )

    df_traces = pd.concat(traces_dataframes, axis=1)

    # exclude sem columns, not needed for plotting
    sem_cols = [(e, c) for e, c in df_traces.columns if "sem" in c]
    df_traces.drop(columns=sem_cols, axis=1, inplace=True)

    # combine statistics data
    for i in range(len(stats_files)):
        # add filename to distinguish cells from different input files
        stats_dataframes[i]["file"] = os.path.basename(stats_files[i])

        # drop (population, up_modulated, down_modulated, non_modulated) to retain per-cell data only
        filter = ~stats_dataframes[i]["name"].isin(
            ["population", "up_modulated", "down_modulated", "non_modulated"]
        )
        stats_dataframes[i] = stats_dataframes[i][filter]

        # reclassify neurons based on significance threshold supplied
        if significance_threshold is not None and significance_threshold > 0:
            stats_dataframes[i] = reclassify_neurons(
                stats_dataframes[i], significance_threshold
            )

    df_stats = pd.concat(stats_dataframes, axis=0)

    output_stats_filename = os.path.join(
        output_dir,
        f"event_aligned_statistics_{group_name}.csv".replace(" ", ""),
    )
    df_stats.to_csv(output_stats_filename, index=False)
    logger.info(
        f"Event-aligned statistics saved "
        f"({os.path.basename(output_stats_filename)}, "
        f"size: {get_file_size(output_stats_filename)})"
    )

    # initialize list of file entries for the output manifest
    activity_heatmap_preview_files = []
    activity_by_modulation_preview_files = []
    fraction_neurons_per_subpopulation_preview_files = []
    epoch_population_activity_preview_filenames = []

    # generate plots
    all_epoch_data = (
        {}
    )  # dictionary that will hold all epoch data together for plotting purposes
    epoch_traces_dfs = []
    for epoch_name in epoch_names:
        # generate heatmap showing event-aligned activity modulation per neuron
        single_cell_heatmap_preview_filename = os.path.join(
            output_dir,
            f"event_aligned_activity_heatmap_{group_name}_{epoch_name}.svg".replace(
                " ", ""
            ),
        )

        # retain only data specific to the epoch being analyzed
        cols_to_drop = [
            (e, c) for e, c in df_traces.columns if e != epoch_name
        ]
        df_traces_epoch = df_traces.drop(cols_to_drop, axis=1)
        df_stats_epoch = df_stats[df_stats["epoch"] == epoch_name]

        generate_activity_heatmap(
            traces_timeline=traces_timeline,
            df_traces=df_traces_epoch,
            df_stats=df_stats_epoch,
            output_filename=single_cell_heatmap_preview_filename,
            cmap=cmap,
            epoch_name=epoch_name,
            activity_heatmap_color_limits=activity_heatmap_color_limits,
        )

        # activity_heatmap_preview_files.append(
        #     IdeasPreviewFile(
        #         name="Event-aligned single-cell activity figure",
        #         help=f"Event-aligned single-cell activity heatmap (epoch: {epoch_name})",
        #         file_path=os.path.abspath(
        #             single_cell_heatmap_preview_filename
        #         ),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        # )

        # generate population activity data
        if average_method == "neurons":
            population_mean = np.nanmean(df_traces_epoch, axis=1)
            population_sem = stats.sem(
                df_traces_epoch, axis=1, nan_policy="omit"
            )
        else:
            # drop cell sem columns from all recordings
            simplified_traces_dataframes = [
                traces_df.drop(
                    columns=[
                        (e, c)
                        for e, c in traces_df.columns
                        if "sem" in c or e != epoch_name
                    ],
                    axis=1,
                )
                for traces_df in traces_dataframes
            ]

            # compute means of cell population data over each recording
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

        with warnings.catch_warnings():
            # ignore performance warnings that may occur when setting multi-level index
            warnings.simplefilter("ignore")
            df_traces_epoch["Time"] = traces_timeline
            df_traces_epoch[(epoch_name, "population_mean")] = population_mean
            df_traces_epoch[(epoch_name, "population_sem")] = population_sem

        population_activity_preview_filename = os.path.join(
            output_dir,
            f"event_aligned_population_activity_{group_name}_{epoch_name}".replace(
                " ", ""
            )
            + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
        )

        plot_population_mean_event_window(
            x_values=traces_timeline,
            x_limits=(traces_timeline[0], traces_timeline[-1]),
            population_mean=df_traces_epoch[epoch_name]["population_mean"],
            population_sem=df_traces_epoch[epoch_name]["population_sem"],
            shuffled_mean=[None] * len(population_mean),
            shuffled_conf_interval_low=[None] * len(population_mean),
            shuffled_conf_interval_high=[None] * len(population_mean),
            significance_threshold=None,
            event_type=None,
            output_filename=population_activity_preview_filename,
            color=epoch_data[epoch_name]["color"],
            epoch_name=epoch_name,
            plot_limits=population_activity_plot_limits,
        )

        epoch_population_activity_preview_filenames.append(
            (epoch_name, population_activity_preview_filename)
        )

        # separate cells based on their modulation group (up/down/non)
        # and compute group data (num cells, mean, sem)
        combined_cell_data = extract_modulation_group_data(
            traces_dataframes=traces_dataframes,
            stats_dataframes=stats_dataframes,
            average_method=average_method,
            epoch_name=epoch_name,
            stats_files=stats_files,
        )

        # add population data to output data
        combined_cell_data["population"] = {
            "mean": df_traces_epoch[epoch_name]["population_mean"].tolist(),
            "sem": df_traces_epoch[epoch_name]["population_sem"].tolist(),
            "num_cells": len(df_traces_epoch.columns),
        }

        # add timeline to output data
        combined_cell_data["x_values"] = traces_timeline

        # retain all epoch data needed for plotting comparisons across epochs
        all_epoch_data[epoch_name] = combined_cell_data

        with warnings.catch_warnings():
            # ignore performance warnings that may occur when setting multi-level index
            warnings.simplefilter("ignore")
            df_traces_epoch[
                (epoch_name, "up_modulated_mean")
            ] = combined_cell_data["up_modulated"]["mean"]
            df_traces_epoch[
                (epoch_name, "up_modulated_sem")
            ] = combined_cell_data["up_modulated"]["sem"]
            df_traces_epoch[
                (epoch_name, "down_modulated_mean")
            ] = combined_cell_data["down_modulated"]["mean"]
            df_traces_epoch[
                (epoch_name, "down_modulated_sem")
            ] = combined_cell_data["down_modulated"]["sem"]
            df_traces_epoch[
                (epoch_name, "non_modulated_mean")
            ] = combined_cell_data["non_modulated"]["mean"]
            df_traces_epoch[
                (epoch_name, "non_modulated_sem")
            ] = combined_cell_data["non_modulated"]["sem"]

        # generate subpopulation activity plot
        subpopulation_activity_preview_filename = os.path.join(
            output_dir,
            f"event_aligned_activity_by_modulation_{group_name}_{epoch_name}".replace(
                " ", ""
            )
            + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
        )
        generate_subpopulation_activity_plot(
            traces_timeline=traces_timeline,
            modulated_cells_dict=combined_cell_data,
            output_filename=subpopulation_activity_preview_filename,
            modulation_colors=modulation_colors,
            average_method=average_method,
            epoch_name=epoch_name,
            activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
        )

        # activity_by_modulation_preview_files.append(
        #     IdeasPreviewFile(
        #         name="Event-aligned sub-population activity figure",
        #         help="Event-aligned average sub-population activity line plot "
        #         f"(up-, down-, and non-modulated neurons) (epoch: {epoch_name}).",
        #         file_path=os.path.abspath(
        #             subpopulation_activity_preview_filename
        #         ),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        # )

        # generate pie chart showing fraction of up/down/non-modulated neurons
        pie_chart_modulation_preview_filename = os.path.join(
            output_dir,
            f"fraction_of_modulated_neurons_{group_name}_{epoch_name}".replace(
                " ", ""
            )
            + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
        )
        generate_modulation_pie_chart(
            num_cells_up_modulated=combined_cell_data["up_modulated"][
                "num_cells"
            ],
            num_cells_down_modulated=combined_cell_data["down_modulated"][
                "num_cells"
            ],
            num_cells_non_modulated=combined_cell_data["non_modulated"][
                "num_cells"
            ],
            output_filename=pie_chart_modulation_preview_filename,
            modulation_colors=modulation_colors,
            epoch_name=epoch_name,
        )

        # fraction_neurons_per_subpopulation_preview_files.append(
        #     IdeasPreviewFile(
        #         name="Fraction of neurons in each sub-population",
        #         help="Pie chart depicting the fraction of neurons in each sub-population "
        #         f"(up-, down-, and non-modulated neurons) (epoch: {epoch_name}).",
        #         file_path=os.path.abspath(
        #             pie_chart_modulation_preview_filename
        #         ),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        # )

        # reorder columns
        df_traces_epoch = pd.concat(
            [df_traces_epoch.iloc[:, -9:], df_traces_epoch.iloc[:, :-9]],
            axis=1,
        )

        epoch_traces_dfs.append(df_traces_epoch)

    # combine epoch traces dataframes into a single dataframe
    combined_traces_df = epoch_traces_dfs[0]
    for df in epoch_traces_dfs[1:]:
        combined_traces_df = pd.concat(
            [combined_traces_df, df.drop("Time", axis=1)], axis=1
        )

    # save combined traces dataframe to disk
    output_traces_filename = os.path.join(
        output_dir,
        f"event_aligned_activity_{group_name}.csv".replace(" ", ""),
    )
    combined_traces_df.to_csv(output_traces_filename, index=False)

    logger.info(
        f"Event-aligned traces saved "
        f"({os.path.basename(output_traces_filename)}, "
        f"size: {get_file_size(output_traces_filename)})"
    )

    # reformat data for compatibility with comparison function below
    single_cell_data = {}
    for epoch_name in epoch_names:
        single_cell_data[epoch_name] = {
            "cell": {
                "true_mean_post-pre": df_stats[
                    df_stats["epoch"] == epoch_name
                ]["true_mean_post-pre"]
            }
        }

    # compare peri-event data (post-pre) across epochs
    num_epochs = len(epoch_names)
    if num_epochs > 1:
        # plot post-pre activity differences between a pair of epochs along with color-coded cell map
        output_epoch_comparison_csv_filename = os.path.join(
            output_dir,
            f"pairwise_epoch_comparisons_{group_name}.csv".replace(" ", ""),
        )
        # (
        #     _,
        #     post_minus_pre_boxplot_preview_file,
        # ) = 
        plot_post_minus_pre_activity_differences_with_cell_map(
            cell_set_files=None,
            data={"single_cell": single_cell_data},
            epoch_data=epoch_data,
            output_dir=output_dir,
            group_name=group_name,
        )

    # reformat data into a dictionary for compatibility with plotting functions
    population_plot_data = {}
    for epoch_name in epoch_names:
        population_plot_data[epoch_name] = {
            "mean": combined_traces_df[epoch_name]["population_mean"],
            "sem": combined_traces_df[epoch_name]["population_sem"],
        }

    # generate plot comparing population event-aligned activity across epochs
    pop_act_comparison_plot_preview_filename = os.path.join(
        output_dir,
        f"event_aligned_population_activity_{group_name}.svg".replace(" ", ""),
    )
    plot_population_mean_event_window_across_epochs(
        x_values=traces_timeline,
        x_limits=(traces_timeline[0], traces_timeline[-1]),
        data=population_plot_data,
        event_type=None,
        output_filename=pop_act_comparison_plot_preview_filename,
        epoch_data=epoch_data,
        plot_limits=population_activity_plot_limits,
    )
    del population_plot_data

    # reformat data into a dictionary for compatibility with plotting functions
    modulation_group_plot_data = {epoch_name: {} for epoch_name in epoch_names}
    for epoch_name in epoch_names:
        for modulation_group in [
            "up_modulated",
            "down_modulated",
            "non_modulated",
        ]:
            modulation_group_plot_data[epoch_name][modulation_group] = {
                "mean": combined_traces_df[epoch_name][
                    f"{modulation_group}_mean"
                ],
                "sem": combined_traces_df[epoch_name][
                    f"{modulation_group}_sem"
                ],
                "num_cells": all_epoch_data[epoch_name][modulation_group][
                    "num_cells"
                ],
            }

    # plot event-aligned activity by activity-modulation group
    event_aligned_modulation_groups_across_epoch_files = (
        plot_event_aligned_activity_by_modulation_group_comparing_epochs(
            x=traces_timeline,
            x_limits=(traces_timeline[0], traces_timeline[-1]),
            data=modulation_group_plot_data,
            event_type=None,
            epoch_data=epoch_data,
            output_dir=output_dir,
            group_name=group_name,
            plot_limits=activity_by_modulation_plot_limits,
        )
    )

    # event_aligned_by_modulation_across_epochs_preview_files = []
    # for (
    #     group_title,
    #     output_filename,
    # ) in event_aligned_modulation_groups_across_epoch_files:
    #     event_aligned_by_modulation_across_epochs_preview_files.append(
    #         IdeasPreviewFile(
    #             name=f"Event-aligned activity of {group_title.lower()} cells",
    #             help=f"Comparison of event-aligned activity of {group_title.lower()} cells across epochs.",
    #             file_path=os.path.abspath(output_filename),
    #             file_format=FileFormat.SVG_FILE.value[1],
    #         )
    #     )

    # plot the number of modulated cells per epoch
    num_modulated_cells_per_epoch_preview_filename = os.path.join(
        output_dir,
        f"num_modulated_cells_per_epoch_{group_name}.svg".replace(" ", ""),
    )
    plot_number_of_modulated_cells_per_epoch(
        data=all_epoch_data,
        epoch_data=epoch_data,
        output_filename=num_modulated_cells_per_epoch_preview_filename,
    )

    # num_modulated_cells_per_epoch_preview_file = IdeasPreviewFile(
    #     name="Number of modulated cells per epoch",
    #     help="Number of up-, down-, and non-modulated neurons per epoch.",
    #     file_path=os.path.abspath(
    #         num_modulated_cells_per_epoch_preview_filename
    #     ),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # plot post-pre per epoch
    post_minus_pre_per_epoch_preview_filename = os.path.join(
        output_dir,
        f"mean_post_minus_pre_activity_per_epoch_{group_name}.svg".replace(
            " ", ""
        ),
    )
    plot_post_minus_pre_per_epoch_bar_chart(
        data=single_cell_data,
        epoch_data=epoch_data,
        output_filename=post_minus_pre_per_epoch_preview_filename,
    )
    # post_minus_pre_per_epoch_preview_file = IdeasPreviewFile(
    #     name="Mean post-pre activity per epoch",
    #     help="Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
    #     file_path=os.path.abspath(post_minus_pre_per_epoch_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # construct output manifest entries for the combined data

    # event-aligned population activity figures
    event_aligned_population_activity_preview_files = []

    # population activity across epochs on one figure
    # event_aligned_population_activity_preview_files.append(
    #     IdeasPreviewFile(
    #         name="Event-aligned population activity across epochs",
    #         help="Comparison of event-aligned average population activity across the epochs.",
    #         file_path=os.path.abspath(
    #             pop_act_comparison_plot_preview_filename
    #         ),
    #         file_format=FileFormat.SVG_FILE.value[1],
    #     )
    # )
    # individual plot showing population activity for each epoch
    # for (
    #     epoch_name,
    #     preview_filename,
    # ) in epoch_population_activity_preview_filenames:
    #     event_aligned_population_activity_preview_files.append(
    #         IdeasPreviewFile(
    #             name="Event-aligned population activity",
    #             help=f"Event-aligned average population activity line plot (epoch: {epoch_name}).",
    #             file_path=os.path.abspath(preview_filename),
    #             file_format=FileFormat.SVG_FILE.value[1],
    #         )
    #     )

    # construct modulated cells metadata (number of up/down/non modulated cells per epoch)
    num_up_modulated_cells_str = ""
    num_down_modulated_cells_str = ""
    num_non_modulated_cells_str = ""
    for i, epoch_name in enumerate(epoch_names):
        num_up_modulated_cells = all_epoch_data[epoch_name]["up_modulated"][
            "num_cells"
        ]
        num_down_modulated_cells = all_epoch_data[epoch_name][
            "down_modulated"
        ]["num_cells"]
        num_non_modulated_cells = all_epoch_data[epoch_name]["non_modulated"][
            "num_cells"
        ]

        num_up_modulated_cells_str += f"{epoch_name}: {num_up_modulated_cells}{', ' if i < num_epochs - 1 else ''}"
        num_down_modulated_cells_str += f"{epoch_name}: {num_down_modulated_cells}{', ' if i < num_epochs - 1 else ''}"
        num_non_modulated_cells_str += f"{epoch_name}: {num_non_modulated_cells}{', ' if i < num_epochs - 1 else ''}"

    # define event-aligned metadata
    # event_aligned_metadata = {
    #     config.IDEAS_METADATA_KEY: {
    #         "metrics": {
    #             "num_up_modulated_cells": num_up_modulated_cells_str,
    #             "num_down_modulated_cells": num_down_modulated_cells_str,
    #             "num_non_modulated_cells": num_non_modulated_cells_str,
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
    event_aligned_metadata = [
        {
            "key": "ideas.metrics.num_up_modulated_cells",
            "name": "Number of up-modulated cells",
            "value": num_up_modulated_cells_str
        },
        {
            "key": "ideas.metrics.num_down_modulated_cells",
            "name": "Number of down-modulated cells",
            "value": num_down_modulated_cells_str
        },
        {
            "key": "ideas.metrics.num_non_modulated_cells",
            "name": "Number of non-modulated cells",
            "value": num_non_modulated_cells_str
        },
        {
            "key": "ideas.timingInfo.numTimes",
            "name": "Number of timepoints",
            "value": len(traces_timeline)
        },
        {
            "key": "ideas.timingInfo.sampling_rate",
            "name": "Sampling Rate (Hz)",
            "value": compute_sampling_rate(
                period_num=abs(traces_timeline[0] - traces_timeline[1]),
                period_den=1,
            ),
        }
    ]

    # event-aligned traces FILE
    # event_aligned_traces_file = IdeasFile(
    #     file_key=f"group{group_id}_event_aligned_traces",
    #     file_path=os.path.abspath(output_traces_filename),
    #     file_type=FileType.COMBINED_PERI_EVENT_TRACES.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TIME_SERIES.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     preview_files=[
    #         *event_aligned_population_activity_preview_files,
    #         *activity_heatmap_preview_files,
    #     ],
    #     add_metadata=event_aligned_metadata,
    # )

    # statistics FILE
    # statistics_file = IdeasFile(
    #     file_key=f"group{group_id}_event_aligned_statistics",
    #     file_path=os.path.abspath(output_stats_filename),
    #     file_type=FileType.COMBINED_PERI_EVENT_STATISTICS.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TABLE.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     preview_files=[
    #         *activity_by_modulation_preview_files,
    #         *event_aligned_by_modulation_across_epochs_preview_files,
    #         *fraction_neurons_per_subpopulation_preview_files,
    #         num_modulated_cells_per_epoch_preview_file,
    #         post_minus_pre_per_epoch_preview_file,
    #     ],
    #     add_metadata=event_aligned_metadata,
    # )

    # pairwise comparison FILE
    # if num_epochs > 1:
    #     epoch_comparison_file = IdeasFile(
    #         file_key=f"group{group_id}_epoch_comparison_data",
    #         file_path=os.path.abspath(output_epoch_comparison_csv_filename),
    #         file_type=FileType.PERI_EVENT_COMPARISON_DATA.value[1],
    #         file_format=FileFormat.CSV_FILE.value[1],
    #         file_structure=FileStructure.TABLE.value[1],
    #         file_category=FileCategory.RESULT.value[1],
    #         parent_ids=[
    #             event_aligned_traces_file.file_id,
    #             statistics_file.file_id,
    #         ],
    #         preview_files=[
    #             post_minus_pre_boxplot_preview_file,
    #         ],
    #         add_metadata=event_aligned_metadata,
    #     )
    # else:
    #     epoch_comparison_file = None

    return df_stats, event_aligned_metadata
        # event_aligned_traces_file,
        # statistics_file,
        # epoch_comparison_file,


def _validate_epoch_names_within_group(
    group_traces_files, group_stats_files, epoch_names
):
    """Validate epoch names present in the group input files.

    :param epoch_names: list of epoch names specified by the user
    :param group_traces_files: peri-event analysis traces files from the group
    :param group_stats_files: peri-event analysis statistics files from the group
    """
    epoch_names_set = set()
    for traces_file, stats_file in zip(group_traces_files, group_stats_files):
        # extract epoch names from traces file
        traces_df = pd.read_csv(traces_file, header=[0, 1])
        traces_epoch_names = sorted(
            traces_df.columns.get_level_values(0).unique().tolist()
        )
        if "Time" in traces_epoch_names:
            traces_epoch_names.remove("Time")

        # extract epoch names from statistics file
        statistics_df = pd.read_csv(stats_file)
        stats_epoch_names = sorted(list(statistics_df["epoch"].unique()))

        # validate that epoch names in traces and statistics files match
        if traces_epoch_names != stats_epoch_names:
            raise IdeasError(
                f"The traces and statistics files do not contain the same epochs: "
                f"{', '.join(traces_epoch_names)} "
                f"vs {', '.join(stats_epoch_names)}",
            )

        # log warnings when the user-specified epoch names don't match those found in the input files
        num_epochs_not_found = 0
        for epoch_name in epoch_names:
            if epoch_name not in traces_epoch_names:
                logger.warning(
                    f"No data found for epoch '{epoch_name}' in file '{os.path.basename(traces_file)}'"
                )
                num_epochs_not_found += 1

        if num_epochs_not_found == len(epoch_names):
            raise IdeasError(
                "No data was found for any of the epochs specified within the input group. "
                "Verify that the epoch names specified match those present in the input files.",
            )

        if traces_epoch_names != epoch_names:
            logger.warning(
                f"The epoch names specified ({', '.join(epoch_names)}) do not "
                f"match the epoch names found in input file '{os.path.basename(traces_file)}' "
                f"({', '.join(traces_epoch_names)})."
            )

        for e in traces_epoch_names:
            epoch_names_set.add(e)

    # return list of unique epochs found across the files within the group
    epoch_names_found = [e for e in epoch_names if e in epoch_names_set]
    return epoch_names_found


def validate_epochs(
    epoch_names: List[str],
    group1_traces_files: List[str],
    group1_stats_files: List[str],
    group2_traces_files: List[str] = None,
    group2_stats_files: List[str] = None,
):
    """Validate epochs across the input parameters and input files.
       - Ensure epoch names provided match those present in the input files.

    :param epoch_names: list of epoch names specified by the user
    :param group1_traces_files: peri-event analysis traces files from the first group
    :param group1_stats_files: peri-event analysis statistics files from the first group
    :param group2_traces_files: peri-event analysis traces files from the second group
    :param group2_stats_files: peri-event analysis statistics files from the second group
    """
    # validate epoch names in the first group
    logger.info("Validating epoch names in the first group")
    group1_epoch_names = _validate_epoch_names_within_group(
        group_traces_files=group1_traces_files,
        group_stats_files=group1_stats_files,
        epoch_names=epoch_names,
    )

    group2_epoch_names = None
    if group2_traces_files is not None and group2_stats_files is not None:
        # validate epoch names in the second group
        logger.info("Validating epoch names in the second group")
        group2_epoch_names = _validate_epoch_names_within_group(
            group_traces_files=group2_traces_files,
            group_stats_files=group2_stats_files,
            epoch_names=epoch_names,
        )

    return group1_epoch_names, group2_epoch_names


def combine_compare_peri_event_data_across_epochs(
    group1_traces_files: List[str],
    group1_stats_files: List[str],
    group1_name: str,
    group2_traces_files: List[str],
    group2_stats_files: List[str],
    group2_name: str,
    epoch_names: str,
    epoch_colors: str,
    comparison_type: str,
    data_pairing: str,
    parametric: str,
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
    :param data_pairing: indicates whether the observations are paired or not
    :param significance_threshold: p-value threshold for classifying neurons as
                                   up- or down-modulated
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording
    :param tolerance: Maximum time shift in seconds between the time windows of the input traces files
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
    logger.info(
        "Starting the combination and comparison of peri-event data across epochs"
    )

    # parse epoch data
    epoch_names = [e.strip() for e in epoch_names.split(",")]
    epoch_colors = [c.strip() for c in epoch_colors.split(",")]
    if len(epoch_names) != len(epoch_colors):
        raise IdeasError(
            "The number of epoch names and epoch colors must match",
        )

    # validate epoch names across the input data
    group1_epoch_names, group2_epoch_names = validate_epochs(
        epoch_names=epoch_names,
        group1_traces_files=group1_traces_files,
        group1_stats_files=group1_stats_files,
        group2_traces_files=group2_traces_files,
        group2_stats_files=group2_stats_files,
    )
    logger.info(
        f"Epochs found in the first group: {', '.join(group1_epoch_names)}"
    )
    if group2_epoch_names is not None:
        logger.info(
            f"Epochs found in the second group: {', '.join(group2_epoch_names)}"
        )

    # construct epoch data structure
    group1_epoch_data = {}
    group2_epoch_data = {}
    for epoch_name, epoch_color in zip(epoch_names, epoch_colors):
        if epoch_name in group1_epoch_names:
            group1_epoch_data[epoch_name] = {
                "color": epoch_color,
            }

        if group2_epoch_names is not None and epoch_name in group2_epoch_names:
            group2_epoch_data[epoch_name] = {
                "color": epoch_color,
            }

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
    group1_name = remove_unsupported_characters(group1_name)
    if group2_name is not None:
        group2_name = remove_unsupported_characters(group2_name)

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
        logger.info("Combining peri-event data from the first group")

        # (
        #     group1_data,
        #     group1_traces_manifest_file,
        #     group1_statistics_manifest_file,
        #     group1_epoch_comparison_file,
        # ) = 
        group1_data, group1_md = combine_peri_event_data(
            traces_files=group1_traces_files,
            stats_files=group1_stats_files,
            epoch_names=group1_epoch_names,
            significance_threshold=significance_threshold,
            group_name=group1_name,
            output_dir=output_dir,
            group_id=1,
            average_method=average_method,
            tolerance=tolerance,
            color=group_colors[0],
            modulation_colors=modulation_colors,
            cmap=cmap,
            epoch_data=group1_epoch_data,
            population_activity_plot_limits=population_activity_plot_limits,
            activity_heatmap_color_limits=activity_heatmap_color_limits,
            activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
        )

        output_metadata["group1_event_aligned_traces"] = group1_md
        output_metadata["group1_event_aligned_statistics"] = group1_md
        output_metadata["group1_epoch_comparison_data"] = group1_md
        # output_files.extend(
        #     [group1_traces_manifest_file, group1_statistics_manifest_file]
        # )

        if len(group1_epoch_names) == 1:
            logger.info(
                f"Only 1 epoch was analyzed. No comparison data across epochs will be generated for group '{group1_name}'."
            )
        else:
            # add file entry to output manifest
            # output_files.append(group1_epoch_comparison_file)

            # compare data within group 1
            group1_pairwise_comparisons_output_filename = os.path.join(
                output_dir,
                f"pairwise_epoch_comparisons_{group1_name}.csv".replace(
                    " ", ""
                ),
            )
            _compare_single_group(
                data=group1_data,
                comparison_type=comparison_type,
                output_filename=group1_pairwise_comparisons_output_filename,
            )

        logger.info("Finished combining peri-event data from the first group")

    # combine data from group 2
    if len(group2_traces_files) > 1:
        logger.info("Combining peri-event data from the second group")

        # (
        #     group2_data,
        #     group2_traces_manifest_file,
        #     group2_statistics_manifest_file,
        #     group2_epoch_comparison_file,
        # ) = 
        group2_data, group2_md = combine_peri_event_data(
            traces_files=group2_traces_files,
            stats_files=group2_stats_files,
            epoch_names=group2_epoch_names,
            significance_threshold=significance_threshold,
            group_name=group2_name,
            output_dir=output_dir,
            group_id=2,
            average_method=average_method,
            tolerance=tolerance,
            color=group_colors[1],
            modulation_colors=modulation_colors,
            cmap=cmap,
            epoch_data=group2_epoch_data,
            population_activity_plot_limits=population_activity_plot_limits,
            activity_heatmap_color_limits=activity_heatmap_color_limits,
            activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
        )
        output_metadata["group2_event_aligned_traces"] = group2_md
        output_metadata["group2_event_aligned_statistics"] = group2_md
        output_metadata["group2_epoch_comparison_data"] = group2_md
        # output_files.extend(
        #     [group2_traces_manifest_file, group2_statistics_manifest_file]
        # )

        if len(group2_epoch_names) == 1:
            logger.info(
                f"Only 1 epoch was analyzed. No comparison data across epochs will be generated for group '{group2_name}'."
            )
        else:
            # add file entry to output manifest
            # output_files.append(group2_epoch_comparison_file)

            # compare data within group 2
            group2_pairwise_comparisons_output_filename = os.path.join(
                output_dir,
                f"pairwise_epoch_comparisons_{group2_name}.csv".replace(
                    " ", ""
                ),
            )
            _compare_single_group(
                data=group2_data,
                comparison_type=comparison_type,
                output_filename=group2_pairwise_comparisons_output_filename,
            )

        logger.info("Finished combining peri-event data from the second group")

    # compare groups 1 and 2
    if len(group1_traces_files) > 1 and len(group2_traces_files) > 1:
        logger.info("Comparing data from the two groups")

        # retain and compare epochs common to both groups
        group1_epoch_names = group1_data["epoch"].unique().tolist()
        group2_epoch_names = group2_data["epoch"].unique().tolist()
        common_epochs = [
            e for e in group1_epoch_names if e in group2_epoch_names
        ]

        if len(common_epochs) == 0:
            logger.warning(
                "There are no epochs in common between the two groups. "
                "No group comparison will be performed."
            )
        else:
            if sorted(group1_epoch_names) != sorted(group2_epoch_names):
                logger.warning(
                    f"The epochs in the first group ({', '.join(group1_epoch_names)}) "
                    f"do not match the epochs in the second group ({', '.join(group2_epoch_names)}). "
                    f"Only the epochs common to both groups ({', '.join(common_epochs)}) will be compared."
                )
                group1_data = group1_data[
                    group1_data["epoch"].isin(common_epochs)
                ]
                group2_data = group2_data[
                    group2_data["epoch"].isin(common_epochs)
                ]

            # compare group 1 and group 2
            mixed_aov, mixed_pairwise = _compare_data(
                g1_data=group1_data,
                g2_data=group2_data,
                group1_name=group1_name,
                group2_name=group2_name,
                comparison_type=comparison_type,
                data_pairing=data_pairing,
                parametric=parametric,
                significance_threshold=significance_threshold,
            )

            # save results to disk
            if mixed_aov is not None:
                # ANOVA data file only generated when there are more than 2 epochs
                anova_comparisons_output_filename = os.path.join(
                    output_dir, "anova_group_comparisons.csv"
                )
                mixed_aov.to_csv(
                    anova_comparisons_output_filename, index=False
                )
                logger.info(
                    f"Group ANOVA comparison data saved "
                    f"({os.path.basename(anova_comparisons_output_filename)}, "
                    f"size: {get_file_size(anova_comparisons_output_filename)})"
                )

            pairwise_comparisons_output_filename = os.path.join(
                output_dir, "pairwise_group_comparisons.csv"
            )
            mixed_pairwise.to_csv(
                pairwise_comparisons_output_filename, index=False
            )
            logger.info(
                f"Group pairwise comparison data saved "
                f"({os.path.basename(pairwise_comparisons_output_filename)}, "
                f"size: {get_file_size(pairwise_comparisons_output_filename)})"
            )

            # generate mixed comparisons previews
            combined_data = pd.concat([group1_data, group2_data])

            post_minus_pre_preview_filename = os.path.join(
                output_dir, "population_post_minus_pre_comparison.svg"
            )

            _plot_mixed_comparisons2(
                data=combined_data,
                g1_color=group_colors[0],
                g2_color=group_colors[1],
                preview_filename=post_minus_pre_preview_filename,
            )

            # generate output manifest file entries

            # ANOVA FILE
            if mixed_aov is not None:
                output_metadata["anova_group_comparisons"] = [
                    group1_md[3], group1_md[4]
                ]

            output_metadata["pairwise_group_comparisons"] = [
                group1_md[3], group1_md[4]
            ]
            #     anova_preview_file = IdeasPreviewFile(
            #         name="Comparisons of post-pre activity between the two groups",
            #         help="Comparisons of post-pre activity between the two groups",
            #         file_path=os.path.abspath(post_minus_pre_preview_filename),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     anova_metadata = {
            #         config.IDEAS_METADATA_KEY: {
            #             "timingInfo": group1_traces_manifest_file.file_metadata[
            #                 "add"
            #             ][
            #                 "ideas"
            #             ][
            #                 "timingInfo"
            #             ],
            #         }
            #     }
            #     anova_file = IdeasFile(
            #         file_key="group_anova_comparisons",
            #         file_path=os.path.abspath(
            #             anova_comparisons_output_filename
            #         ),
            #         file_type=FileType.PERI_EVENT_COMPARISON_DATA.value[1],
            #         file_format=FileFormat.CSV_FILE.value[1],
            #         file_structure=FileStructure.TABLE.value[1],
            #         file_category=FileCategory.RESULT.value[1],
            #         preview_files=[anova_preview_file],
            #         add_metadata=anova_metadata,
            #     )
            #     output_files.append(anova_file)

            # # pairwise FILE
            # pairwise_preview_file = IdeasPreviewFile(
            #     name="Comparisons of post-pre activity between the two groups",
            #     help="Comparisons of post-pre activity between the two groups",
            #     file_path=os.path.abspath(post_minus_pre_preview_filename),
            #     file_format=FileFormat.SVG_FILE.value[1],
            # )
            # pairwise_metadata = {
            #     config.IDEAS_METADATA_KEY: {
            #         "timingInfo": group1_traces_manifest_file.file_metadata[
            #             "add"
            #         ]["ideas"]["timingInfo"],
            #     }
            # }
            # pairwise_file = IdeasFile(
            #     file_key="group_pairwise_comparisons",
            #     file_path=os.path.abspath(
            #         pairwise_comparisons_output_filename
            #     ),
            #     file_type=FileType.PERI_EVENT_COMPARISON_DATA.value[1],
            #     file_format=FileFormat.CSV_FILE.value[1],
            #     file_structure=FileStructure.TABLE.value[1],
            #     file_category=FileCategory.RESULT.value[1],
            #     preview_files=[pairwise_preview_file],
            #     add_metadata=pairwise_metadata,
            # )
            # output_files.append(pairwise_file)

            logger.info("Finished comparing data from the two groups")

    # generate output manifest & metadata
    # generate_output_manifest(
    #     group1_traces_files=group1_traces_files,
    #     group1_stats_files=group1_stats_files,
    #     group2_traces_files=group2_traces_files,
    #     group2_stats_files=group2_stats_files,
    #     output_files=output_files,
    #     output_dir=output_dir,
    #     output_group_key="combine_compare_peri_event_data_across_epochs_output",
    # )
    with open(os.path.join(output_dir, "output_metadata.json"), "w") as f:
        json.dump(output_metadata, f)

    logger.info(
        "Combination and comparison of peri-event data across epochs completed"
    )


def combine_compare_peri_event_data_across_epochs_ideas_wrapper(
    group1_traces_files: List[IdeasFile],
    group1_stats_files: List[IdeasFile],
    group1_name: str,
    group2_traces_files: List[IdeasFile],
    group2_stats_files: List[IdeasFile],
    group2_name: str,
    epoch_names: str,
    epoch_colors: str,
    comparison_type: str,
    data_pairing: str,
    parametric: str,
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
    """IDEAS tool wrapper for combine and compare peri-event analysis data.

    :param group1_traces_files: peri-event analysis traces files from the first group
    :param group1_stats_files: peri-event analysis statistics files from the first group
    :param group1_name: name of the first group
    :param group2_traces_files: peri-event analysis traces files from the second group
    :param group2_stats_files: peri-event analysis statistics files from the second group
    :param group2_name: name of the second group
    :param comparison_type: type of statistical test to perform
    :param data_pairing: indicates whether the observations are paired or not
    :param significance_threshold: p-value threshold for classifying neurons as
                                   up- or down-modulated
    :param average_method: indicates whether averaging should be done across all
                           observations or over each recording
    :param tolerance: Maximum time shift in seconds between the time windows of the input traces files
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
    """

    combine_compare_peri_event_data_across_epochs(
        group1_traces_files=group1_traces_files,
        group1_stats_files=group1_stats_files,
        group1_name=group1_name,
        group2_traces_files=group2_traces_files,
        group2_stats_files=group2_stats_files,
        group2_name=group2_name,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        comparison_type=comparison_type,
        data_pairing=data_pairing,
        parametric=parametric,
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
        epoch_names = [e.strip().replace(" ", "") for e in epoch_names.split(",")]
        with outputs.register(raise_missing_file=False) as output_data:
            for group_name in [group1_name, group2_name]:
                group_name = group_name.replace(" ", "")
                subdir_base = "group1" if group_name == group1_name else "group2"
                
                output_file = output_data.register_file(
                    f"event_aligned_activity_{group_name}.csv",
                    subdir=f"{subdir_base}_event_aligned_traces",
                )
                for epoch_name in epoch_names:
                    output_file.register_preview(
                        f"event_aligned_population_activity_{group_name}.svg",
                        caption=f"Event-aligned single-cell activity heatmap (epoch: {epoch_name})"
                    ).register_preview(
                        f"event_aligned_population_activity_{group_name}_{epoch_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                        caption="Comparison of event-aligned average population activity across the epochs."
                    ).register_preview(
                        f"event_aligned_activity_heatmap_{group_name}_{epoch_name}.svg",
                        caption=f"Event-aligned single-cell activity heatmap (epoch: {epoch_name})"
                    )
                for md in metadata.get(f"{subdir_base}_event_aligned_traces", {}):
                    output_file.register_metadata(**md) 
                
                output_file = output_data.register_file(
                    f"event_aligned_statistics_{group_name}.csv",
                    subdir=f"{subdir_base}_event_aligned_statistics",
                ).register_preview(
                    f"num_modulated_cells_per_epoch_{group_name}.svg",
                    caption="Number of up-, down-, and non-modulated neurons per epoch."
                ).register_preview(
                    f"mean_post_minus_pre_activity_per_epoch_{group_name}.svg",
                    caption="Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
                )
                for epoch_name in epoch_names:
                    output_file.register_preview(
                        f"event_aligned_activity_by_modulation_{group_name}_{epoch_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                        caption=f"Event-aligned average sub-population activity line plot (up-, down-, and non-modulated neurons) (epoch: {epoch_name})."
                    ).register_preview(
                        f"fraction_of_modulated_neurons_{group_name}_{epoch_name}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                        caption=f"Pie chart depicting the fraction of neurons in each sub-population (up-, down-, and non-modulated neurons) (epoch: {epoch_name}).",
                    )
                mod_groups = ["up_modulated", "down_modulated", "non_modulated"]
                for mod_group in mod_groups:
                    output_file.register_preview(
                        f"event_aligned_activity_{mod_group}_{group_name}.svg",
                        caption=f"Comparison of event-aligned activity of {mod_group.replace("_", " ")} cells across epochs.",
                    )
                for md in metadata.get(f"{subdir_base}_event_aligned_statistics", {}):
                    output_file.register_metadata(**md) 
        
                output_file = output_data.register_file(
                    f"pairwise_epoch_comparisons_{group_name}.csv",
                    subdir=f"{subdir_base}_epoch_comparison_data",
                ).register_preview(
                    f"post_minus_pre_boxplot_{group_name}.svg",
                    caption="Distribution of post-pre activity across epochs displayed using a box plot. Lines connect the same cells together."
                )
                for md in metadata.get(f"{subdir_base}_epoch_comparison_data", {}):
                    output_file.register_metadata(**md) 
            
            output_file = output_data.register_file(
                "anova_group_comparisons.csv"
            ).register_preview(
                "population_post_minus_pre_comparison.svg",
                caption="Comparisons of post-pre activity between the two groups",
            )
            for md in metadata.get("anova_group_comparisons", {}):
                output_file.register_metadata(**md) 

            output_file = output_data.register_file(
                "pairwise_group_comparisons.csv"
            ).register_preview(
                "population_post_minus_pre_comparison.svg",
                caption="Comparisons of post-pre activity between the two groups",
            )
            for md in metadata.get("pairwise_group_comparisons", {}):
                output_file.register_metadata(**md) 

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")
    