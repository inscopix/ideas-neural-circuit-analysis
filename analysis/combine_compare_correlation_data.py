import logging
import os
from typing import List, Optional, Tuple, Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from ideas_commons.constants import (
#     FileCategory,
#     FileFormat,
#     FileStructure,
#     FileType,
#     GroupType,
# )

import utils.config as config

# from toolbox.utils.data_model import (
#     IdeasFile,
#     IdeasGroup,
#     IdeasPreviewFile,
# )
# from toolbox.utils.exceptions import ExitStatus, ToolException
from ideas.exceptions import IdeasError
# from toolbox.utils.output_manifest import save_output_manifest
from ideas.utils import (
    get_file_size,
    _set_up_logger
)
from analysis.combine_compare_population_data import (
    match_subjects,
    validate_colors,
)
from utils.population_data_validation import (
    validate_file_group,
    validate_state_names_and_colors,
    validate_group_names,
    validate_subject_id_format,
)
from utils.combine_compare_population_data_utils import (
    calculate_state_lmm_stats,
    calculate_group_anova_stats,
    calculate_mod_stats_direct,
)
from utils.visualization_helpers import (
    plot_state_lmm_comparison,
    plot_group_anova_comparison,
    create_boxplot_preview,
    create_cdf_preview,
)
from utils.statistical_formatting import (
    _get_mixed_pairwise_sig,
    _cleanup_final_csv_columns,
)
from utils.statistical_validation import (
    _suppress_pingouin_warnings,
)

_set_up_logger()
logger = logging.getLogger(__name__)


def _expand_statistic_name(statistic: str) -> str:
    """Expand abbreviated statistic names to full forms.

    Parameters
    ----------
    statistic : str
        The abbreviated statistic name (e.g., "max", "min", "mean")

    Returns
    -------
    str
        The expanded statistic name (e.g., "Maximum Correlation",
        "Minimum Correlation", "Mean Correlation")

    """
    statistic_mapping = {
        "max": "Maximum Correlation",
        "min": "Minimum Correlation",
        "mean": "Mean Correlation",
        "average": "Average Correlation",
    }
    return statistic_mapping.get(statistic.lower(), statistic.capitalize())


def combine_compare_correlation_data(
    group1_correlation_files: List[str],
    state_names: str,
    state_colors: str,
    group1_name: Optional[str] = None,
    group1_color: Optional[str] = None,
    group2_correlation_files: Optional[List[str]] = None,
    group2_name: Optional[str] = None,
    group2_color: Optional[str] = None,
    statistic: str = "max",
    multiple_correction: str = "bonf",
    effect_size: str = "cohen",
    data_pairing: str = "unpaired",
    subject_matching: str = "number",
    significance_threshold: float = 0.05,
    output_dir: str = None,
) -> None:
    """Combine and compare correlation data.

    :param group1_correlation_files: correlation data files from the first group
    :param state_names: comma-separated string of state names to include
    :param state_colors: comma-separated string of colors for states
    :param group1_name: name of the first group
    :param group1_color: color for the first group
    :param group2_correlation_files: correlation data files from the second group
    :param group2_name: name of the second group
    :param group2_color: color for the second group
    :param statistic: type of statistical test to perform
    :param multiple_correction: method for multiple comparison correction
    :param effect_size: method to calculate effect size
    :param data_pairing: type of data pairing for comparison ("paired" or "unpaired")
    :param subject_matching: method for matching subjects between groups for paired analysis
    :param significance_threshold: threshold for statistical significance
    :param output_dir: path to the output directory

    Expected H5 File Structure:
    --------------------------
    Each H5 file should contain:
    - Top-level datasets where each dataset name represents a "state" (experimental condition)
    - Each dataset should be a 2D correlation matrix where each element represents
      the correlation between two cells/neurons
    - The size of matrices may differ between files (representing different numbers of cells)
    - Example dataset shapes: (198, 198), (330, 330), etc.
    """
    logger.info("Starting the combination and comparison of correlation data")

    # validate input parameters
    validate_combine_compare_correlation_data_parameters(
        group1_correlation_files=group1_correlation_files,
        group1_name=group1_name,
        group2_correlation_files=group2_correlation_files,
        group2_name=group2_name,
        state_names=state_names,
        state_colors=state_colors,
    )

    # Parse state names and colors
    filter_state_names = [
        s.strip() for s in state_names.split(",") if s.strip()
    ]
    state_colors_list = validate_colors(state_colors, "state")

    # Create state_color_map dictionary
    state_color_map = {
        state: color
        for state, color in zip(filter_state_names, state_colors_list)
    }

    # Set default group names first
    group1_name = group1_name or "Group 1"

    # Log the states we'll be filtering for
    logger.info(f"Filtering for states that match: {filter_state_names}")

    if group2_correlation_files is None:
        group2_correlation_files = []

    # set output directory to the current working directory if unspecified
    if output_dir is None:
        output_dir = os.getcwd()

    # define list of files to include in the output manifest
    output_files = []
    # Apply subject matching if doing paired analysis with two groups
    if data_pairing == "paired" and group2_correlation_files:
        # Get matched file pairs
        matched_files = match_subjects(
            group1_files=group1_correlation_files,
            group2_files=group2_correlation_files,
            match_method=subject_matching,
        )

        # Update file lists with matched pairs
        group1_correlation_files = [pair[0] for pair in matched_files]
        group2_correlation_files = [pair[1] for pair in matched_files]

    # Step 1: Combine data from group 1 and perform 1-way ANOVA
    # Initialize DataFrames to avoid NameError
    avg_df1 = pd.DataFrame()
    stat_df1 = pd.DataFrame()
    avg_df2 = pd.DataFrame()
    stat_df2 = pd.DataFrame()

    # Validate minimum file requirements after subject matching
    if len(group1_correlation_files) < 2:
        raise IdeasError(
            f"After subject matching, group 1 has only {len(group1_correlation_files)} file(s). "
            f"At least 2 files per group are required for meaningful statistical analysis.",
        )

    if group2_correlation_files and len(group2_correlation_files) < 2:
        raise IdeasError(
            f"After subject matching, group 2 has only {len(group2_correlation_files)} file(s). "
            f"At least 2 files per group are required for meaningful statistical analysis.",
        )

    if len(group1_correlation_files) >= 2:
        # Read and process group 1 data
        data1 = []
        # Create group prefix for consistent subject ID handling
        group1_prefix = (
            f"{group1_name.replace(' ', '_')}_"
            if data_pairing == "unpaired" and group2_correlation_files
            else ""
        )
        for i, f in enumerate(group1_correlation_files):
            correlations, keys = read_h5(f, filter_state_names)
            if correlations:
                data1.append(
                    {
                        "file": os.path.basename(f),
                        "group_id": 1,
                        "subject_id": f"subject_{i + 1}",  # Base subject ID
                        # (i+1 for human-readable numbering starting from 1)
                        "normalized_subject_id": f"{group1_prefix}subject_{i + 1}",
                        # Group-prefixed ID for statistical analysis
                        # (ensures uniqueness across groups in unpaired analysis)
                        "data": correlations,
                    }
                )

        if not data1:
            raise IdeasError(
                f"No valid data found in any of the group 1 correlation files after filtering "
                f"for states: {filter_state_names}. Please check your state names.",
            )

        # Calculate average correlations and cell measures
        avg_df1 = average_correlations(data1)
        avg_df1["file"] = avg_df1["file"].apply(lambda x: os.path.basename(x))
        avg_df1["group_name"] = group1_name
        # Add subject_id and normalized_subject_id to avg_df1
        for d in data1:
            mask = avg_df1["file"] == os.path.basename(d["file"])
            avg_df1.loc[mask, "subject_id"] = d["subject_id"]
            avg_df1.loc[mask, "normalized_subject_id"] = d[
                "normalized_subject_id"
            ]

        stat_df1 = measure_cells(data1, statistic)
        stat_df1["group_name"] = group1_name

        # Add subject_id and normalized_subject_id to stat_df1
        for d in data1:
            mask = stat_df1["file"] == os.path.basename(d["file"])
            stat_df1.loc[mask, "subject_id"] = d["subject_id"]
            stat_df1.loc[mask, "normalized_subject_id"] = d[
                "normalized_subject_id"
            ]

        # Calculate total cells for group 1
        total_cells_group1 = (
            stat_df1.groupby("file")["cell"].nunique().sum()
            if not stat_df1.empty
            else 0
        )
        message = (
            f"Total unique cells calculated for Group 1: {total_cells_group1}"
        )
        logger.info(message)

    if group2_correlation_files and len(group2_correlation_files) >= 1:
        # Read and process group 2 data
        data2 = []
        # Create group prefix for consistent subject ID handling
        group2_prefix = (
            f"{group2_name.replace(' ', '_')}_"
            if data_pairing == "unpaired"
            else ""
        )
        for i, f in enumerate(group2_correlation_files):
            correlations, keys = read_h5(f, filter_state_names)
            if correlations:
                data2.append(
                    {
                        "file": os.path.basename(f),
                        "group_id": 2,
                        "subject_id": f"subject_{i + 1}",  # Base subject ID
                        # (i+1 for human-readable numbering starting from 1)
                        "normalized_subject_id": f"{group2_prefix}subject_{i + 1}",
                        # Group-prefixed ID # for statistical analysis
                        # (ensures uniqueness across groups in unpaired analysis)
                        "data": correlations,
                    }
                )

        if not data2:
            raise IdeasError(
                f"No valid data found in any of the group 2 correlation files after filtering "
                f"for states: {filter_state_names}. Please check your state names.",
            )

        # Calculate average correlations and cell measures
        avg_df2 = average_correlations(data2)
        avg_df2["file"] = avg_df2["file"].apply(lambda x: os.path.basename(x))
        avg_df2["group_name"] = group2_name
        # Add subject_id and normalized_subject_id to avg_df2
        for d in data2:
            mask = avg_df2["file"] == os.path.basename(d["file"])
            avg_df2.loc[mask, "subject_id"] = d["subject_id"]
            avg_df2.loc[mask, "normalized_subject_id"] = d[
                "normalized_subject_id"
            ]

        stat_df2 = measure_cells(data2, statistic)
        stat_df2["group_name"] = group2_name
        # Add subject_id and normalized_subject_id to stat_df2
        for d in data2:
            mask = stat_df2["file"] == os.path.basename(d["file"])
            stat_df2.loc[mask, "subject_id"] = d["subject_id"]
            stat_df2.loc[mask, "normalized_subject_id"] = d[
                "normalized_subject_id"
            ]

        # Calculate total cells for group 2
        total_cells_group2 = (
            stat_df2.groupby("file")["cell"].nunique().sum()
            if not stat_df2.empty
            else 0
        )
        message = (
            f"Total unique cells calculated for Group 2: {total_cells_group2}"
        )
        logger.info(message)

    # Merge DataFrames for combined analysis
    # Combine average correlations - safely handle concatenation
    if group2_correlation_files and not avg_df2.empty:
        # Only concatenate if we have both groups with data
        if not avg_df1.empty:
            combined_avg = pd.concat([avg_df1, avg_df2], ignore_index=True)
        else:
            combined_avg = avg_df2.copy()
    else:
        combined_avg = avg_df1.copy()

    # Combine cell measures - safely handle concatenation
    if group2_correlation_files and not stat_df2.empty:
        # Only concatenate if we have both groups with data
        if not stat_df1.empty:
            combined_stat = pd.concat([stat_df1, stat_df2], ignore_index=True)
        else:
            combined_stat = stat_df2.copy()
    else:
        combined_stat = stat_df1.copy()

    # --- Save Combined Data ---
    # Save group 1 combined average correlation data
    group1_avg_df = combined_avg[combined_avg["group_name"] == group1_name]
    group1_avg_previews = []
    if not group1_avg_df.empty:
        group1_combined_avg_file_path = os.path.join(
            output_dir,
            f"{group1_name}_combined_average_correlation.csv",
        )
        # Clean the dataframe before saving to remove internal columns
        group1_avg_df_clean = _cleanup_final_csv_columns(
            group1_avg_df, "mixed"
        )
        group1_avg_df_clean.to_csv(group1_combined_avg_file_path, index=False)
        message = (
            f"Group 1 combined average correlation data saved"
            f" ({os.path.basename(group1_combined_avg_file_path)},"
            f" size: {get_file_size(group1_combined_avg_file_path)})"
        )
        logger.info(message)

        # Compute separate y-limits for positive and negative correlation boxplots
        positive_values = []
        negative_values = []

        if not group1_avg_df.empty:
            positive_values.append(
                group1_avg_df["positive_correlation"].dropna()
            )
            negative_values.append(
                group1_avg_df["negative_correlation"].dropna()
            )

        if group2_correlation_files and group2_name:
            group2_avg_df = combined_avg[
                combined_avg["group_name"] == group2_name
            ]
            if not group2_avg_df.empty:
                positive_values.append(
                    group2_avg_df["positive_correlation"].dropna()
                )
                negative_values.append(
                    group2_avg_df["negative_correlation"].dropna()
                )

        # Calculate separate limits for positive and negative correlations
        if positive_values:
            positive_concat = pd.concat(positive_values)
            pos_min = positive_concat.min()
            pos_max = positive_concat.max()
            pos_pad = 0.05 * (pos_max - pos_min) if pos_max > pos_min else 0.1
            positive_y_limits = (pos_min - pos_pad, pos_max + pos_pad)
        else:
            positive_y_limits = None

        if negative_values:
            negative_concat = pd.concat(negative_values)
            neg_min = negative_concat.min()
            neg_max = negative_concat.max()
            neg_pad = 0.05 * (neg_max - neg_min) if neg_max > neg_min else 0.1
            negative_y_limits = (neg_min - neg_pad, neg_max + neg_pad)

        # Create boxplot previews using helper function
        pos_avg_filename = os.path.join(
            output_dir,
            f"{group1_name}_avg_positive_correlation_boxplot.svg",
        )
        pos_avg_success = create_boxplot_preview(
            data_df=group1_avg_df,
            col_name="positive_correlation",
            group_name=group1_name,
            identifier="avg_positive",
            title_prefix="Average Positive Correlation",
            filename=pos_avg_filename,
            state_color_map=state_color_map,
            filter_state_names=filter_state_names,
            data_type="correlation",
            y_limits=positive_y_limits,
        )
        # if pos_avg_success:
        #     pos_avg_preview = IdeasPreviewFile(
        #         name=f"{group1_name} Avg Positive Correlation Boxplot",
        #         help=(
        #             f"Box and whisker plot displaying the distribution of average positive "
        #             f"correlations across experimental states for {group1_name}. "
        #             f"The plot shows median values, quartiles, and outliers."
        #         ),
        #         file_path=os.path.abspath(pos_avg_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        #     group1_avg_previews.append(pos_avg_preview)

        neg_avg_filename = os.path.join(
            output_dir,
            f"{group1_name}_avg_negative_correlation_boxplot.svg",
        )
        neg_avg_success = create_boxplot_preview(
            data_df=group1_avg_df,
            col_name="negative_correlation",
            group_name=group1_name,
            identifier="avg_negative",
            title_prefix="Average Negative Correlation",
            filename=neg_avg_filename,
            state_color_map=state_color_map,
            filter_state_names=filter_state_names,
            data_type="correlation",
            y_limits=negative_y_limits,
        )
        # if neg_avg_success:
        #     neg_avg_preview = IdeasPreviewFile(
        #         name=f"{group1_name} Avg Negative Correlation Boxplot",
        #         help=(
        #             f"Box and whisker plot displaying the distribution of average negative "
        #             f"correlations across experimental states for {group1_name}. "
        #             f"The plot shows median values, quartiles, and outliers."
        #         ),
        #         file_path=os.path.abspath(neg_avg_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        #     group1_avg_previews.append(neg_avg_preview)

        avg_metadata_g1 = {
            config.IDEAS_METADATA_KEY: {
                "dataset": {
                    "states": list(group1_avg_df.state.unique()),
                    "group_name": group1_name,
                },
                "metrics": {
                    "num_recordings": len(group1_avg_df.file.unique())
                },
            }
        }
        # group1_combined_avg_file = IdeasFile(
        #     file_key="group1_combined_avg_correlation_data",
        #     file_path=os.path.abspath(group1_combined_avg_file_path),
        #     file_type=FileType.COMBINED_CORRELATION_DATA.value[1],
        #     file_format=FileFormat.CSV_FILE.value[1],
        #     file_structure=FileStructure.TABLE.value[1],
        #     file_category=FileCategory.RESULT.value[1],
        #     preview_files=group1_avg_previews,
        #     add_metadata=avg_metadata_g1,
        # )
        # output_files.append(group1_combined_avg_file)

    # Save group 1 combined statistic correlation data
    group1_stat_df = combined_stat[combined_stat["group_name"] == group1_name]
    group1_stat_previews = []
    if not group1_stat_df.empty:
        group1_combined_stat_file_path = os.path.join(
            output_dir,
            f"{group1_name}_combined_{statistic}_correlation.csv",
        )
        # Clean the dataframe before saving to remove internal columns
        group1_stat_df_clean = _cleanup_final_csv_columns(
            group1_stat_df, "mixed"
        )
        group1_stat_df_clean.to_csv(
            group1_combined_stat_file_path, index=False
        )
        message = (
            f"Group 1 combined {statistic} correlation data saved"
            f" ({os.path.basename(group1_combined_stat_file_path)},"
            f" size: {get_file_size(group1_combined_stat_file_path)})"
        )
        logger.info(message)

        # Create CDF preview for single cell distribution
        stat_cdf_filename = os.path.join(
            output_dir,
            f"{group1_name}_{statistic}_correlation_cdf.svg",
        )
        stat_cdf_success = create_cdf_preview(
            data_df=group1_stat_df,
            col_name=f"{statistic}_correlation",
            group_name=group1_name,
            identifier=statistic,
            title_prefix=_expand_statistic_name(statistic),
            filename=stat_cdf_filename,
            state_color_map=state_color_map,
            filter_state_names=filter_state_names,
            data_type="correlation",
        )
        # if stat_cdf_success:
        #     stat_preview = IdeasPreviewFile(
        #         name=f"{group1_name} {statistic.capitalize()} Correlation CDF",
        #         help=(
        #             f"Cumulative distribution function (CDF) displaying "
        #             f"{statistic} correlations across experimental states for "
        #             f"{group1_name}. The CDF plot shows the probability "
        #             f"distribution of correlation values."
        #         ),
        #         file_path=os.path.abspath(stat_cdf_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        #     group1_stat_previews.append(stat_preview)

        # Add an additional boxplot with individual data points for better visualization
        stat_boxplot_filename = os.path.join(
            output_dir,
            f"{group1_name}_{statistic}_correlation_boxplot.svg",
        )
        stat_boxplot_success = create_boxplot_preview(
            data_df=group1_stat_df,
            col_name=f"{statistic}_correlation",
            group_name=group1_name,
            identifier=statistic,
            title_prefix=_expand_statistic_name(statistic),
            filename=stat_boxplot_filename,
            state_color_map=state_color_map,
            filter_state_names=filter_state_names,
            data_type="correlation",
            include_points=True,
        )
        # if stat_boxplot_success:
        #     stat_detailed_preview = IdeasPreviewFile(
        #         name=f"{group1_name} {statistic.capitalize()} Correlation Boxplot",
        #         help=(
        #             f"Box and whisker plot with individual data points "
        #             f"displaying the distribution of {statistic} correlations across "
        #             f"experimental states for {group1_name}. The plot shows "
        #             f"median values, quartiles, outliers, and individual data "
        #             f"points."
        #         ),
        #         file_path=os.path.abspath(stat_boxplot_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        #     group1_stat_previews.append(stat_detailed_preview)

        total_cells_group1 = (
            group1_stat_df.groupby("file")["cell"].nunique().sum()
            if not group1_stat_df.empty
            else 0
        )
        stat_metadata_g1 = {
            config.IDEAS_METADATA_KEY: {
                "dataset": {
                    "states": list(group1_stat_df.state.unique()),
                    "group_name": group1_name,
                },
                "metrics": {
                    "total_num_cells": int(total_cells_group1),
                    "num_recordings": len(group1_stat_df.file.unique()),
                    "statistic_type": statistic,
                },
            }
        }
        # group1_combined_stat_file = IdeasFile(
        #     file_key="group1_combined_cell_correlation_data",
        #     file_path=os.path.abspath(group1_combined_stat_file_path),
        #     file_type=FileType.COMBINED_CORRELATION_DATA.value[1],
        #     file_format=FileFormat.CSV_FILE.value[1],
        #     file_structure=FileStructure.TABLE.value[1],
        #     file_category=FileCategory.RESULT.value[1],
        #     preview_files=group1_stat_previews,
        #     add_metadata=stat_metadata_g1,
        # )
        # output_files.append(group1_combined_stat_file)

    # Save group 2 data if it exists
    if group2_correlation_files and group2_name:
        # Save group 2 combined average correlation data
        group2_avg_df = combined_avg[combined_avg["group_name"] == group2_name]
        group2_avg_previews = []
        if not group2_avg_df.empty:
            group2_combined_avg_file_path = os.path.join(
                output_dir,
                f"{group2_name}_combined_average_correlation.csv",
            )
            # Clean the dataframe before saving to remove internal columns
            group2_avg_df_clean = _cleanup_final_csv_columns(
                group2_avg_df, "mixed"
            )
            group2_avg_df_clean.to_csv(
                group2_combined_avg_file_path, index=False
            )
            message = (
                f"Group 2 combined average correlation data saved"
                f" ({os.path.basename(group2_combined_avg_file_path)},"
                f" size: {get_file_size(group2_combined_avg_file_path)})"
            )
            logger.info(message)

            # Compute separate y-limits for positive and negative correlation boxplots
            positive_values = []
            negative_values = []

            if not group2_avg_df.empty:
                positive_values.append(
                    group2_avg_df["positive_correlation"].dropna()
                )
                negative_values.append(
                    group2_avg_df["negative_correlation"].dropna()
                )

            # Calculate separate limits for positive and negative correlations
            if positive_values:
                positive_concat = pd.concat(positive_values)
                pos_min = positive_concat.min()
                pos_max = positive_concat.max()
                pos_pad = (
                    0.05 * (pos_max - pos_min) if pos_max > pos_min else 0.1
                )
                positive_y_limits = (pos_min - pos_pad, pos_max + pos_pad)
            else:
                positive_y_limits = None

            if negative_values:
                negative_concat = pd.concat(negative_values)
                neg_min = negative_concat.min()
                neg_max = negative_concat.max()
                neg_pad = (
                    0.05 * (neg_max - neg_min) if neg_max > neg_min else 0.1
                )
                negative_y_limits = (neg_min - neg_pad, neg_max + neg_pad)

            # Create boxplot previews using helper function
            pos_avg_filename_g2 = os.path.join(
                output_dir,
                f"{group2_name}_avg_positive_correlation_boxplot.svg",
            )
            pos_avg_success_g2 = create_boxplot_preview(
                data_df=group2_avg_df,
                col_name="positive_correlation",
                group_name=group2_name,
                identifier="avg_positive",
                title_prefix="Average Positive Correlation",
                filename=pos_avg_filename_g2,
                state_color_map=state_color_map,
                filter_state_names=filter_state_names,
                data_type="correlation",
                y_limits=positive_y_limits,
            )
            # if pos_avg_success_g2:
            #     pos_avg_preview_g2 = IdeasPreviewFile(
            #         name=f"{group2_name} Avg Positive Correlation Boxplot",
            #         help=(
            #             f"Box and whisker plot displaying the distribution of average "
            #             f"positive correlations across experimental states for "
            #             f"{group2_name}. The plot shows median values, quartiles, "
            #             f"and outliers."
            #         ),
            #         file_path=os.path.abspath(pos_avg_filename_g2),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group2_avg_previews.append(pos_avg_preview_g2)

            neg_avg_filename_g2 = os.path.join(
                output_dir,
                f"{group2_name}_avg_negative_correlation_boxplot.svg",
            )
            neg_avg_success_g2 = create_boxplot_preview(
                data_df=group2_avg_df,
                col_name="negative_correlation",
                group_name=group2_name,
                identifier="avg_negative",
                title_prefix="Average Negative Correlation",
                filename=neg_avg_filename_g2,
                state_color_map=state_color_map,
                filter_state_names=filter_state_names,
                data_type="correlation",
                y_limits=negative_y_limits,
            )
            # if neg_avg_success_g2:
            #     neg_avg_preview_g2 = IdeasPreviewFile(
            #         name=f"{group2_name} Avg Negative Correlation Boxplot",
            #         help=(
            #             f"Box and whisker plot displaying the distribution of average "
            #             f"negative correlations across experimental states for "
            #             f"{group2_name}. The plot shows median values, quartiles, "
            #             f"and outliers."
            #         ),
            #         file_path=os.path.abspath(neg_avg_filename_g2),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group2_avg_previews.append(neg_avg_preview_g2)

            avg_metadata_g2 = {
                config.IDEAS_METADATA_KEY: {
                    "dataset": {
                        "states": list(group2_avg_df.state.unique()),
                        "group_name": group2_name,
                    },
                    "metrics": {
                        "num_recordings": len(group2_avg_df.file.unique())
                    },
                }
            }
            # group2_combined_avg_file = IdeasFile(
            #     file_key="group2_combined_avg_correlation_data",
            #     file_path=os.path.abspath(group2_combined_avg_file_path),
            #     file_type=FileType.COMBINED_CORRELATION_DATA.value[1],
            #     file_format=FileFormat.CSV_FILE.value[1],
            #     file_structure=FileStructure.TABLE.value[1],
            #     file_category=FileCategory.RESULT.value[1],
            #     preview_files=group2_avg_previews,
            #     add_metadata=avg_metadata_g2,
            # )
            # output_files.append(group2_combined_avg_file)

        # Save group 2 combined statistic correlation data
        group2_stat_df = combined_stat[
            combined_stat["group_name"] == group2_name
        ]
        group2_stat_previews = []
        if not group2_stat_df.empty:
            group2_combined_stat_file_path = os.path.join(
                output_dir,
                f"{group2_name}_combined_{statistic}_correlation.csv",
            )
            # Clean the dataframe before saving to remove internal columns
            group2_stat_df_clean = _cleanup_final_csv_columns(
                group2_stat_df, "mixed"
            )
            group2_stat_df_clean.to_csv(
                group2_combined_stat_file_path, index=False
            )
            message = (
                f"Group 2 combined {statistic} correlation data saved"
                f" ({os.path.basename(group2_combined_stat_file_path)},"
                f" size: {get_file_size(group2_combined_stat_file_path)})"
            )
            logger.info(message)

            # Create CDF preview for single cell distribution
            stat_cdf_filename_g2 = os.path.join(
                output_dir,
                f"{group2_name}_{statistic}_correlation_cdf.svg",
            )
            stat_cdf_success_g2 = create_cdf_preview(
                data_df=group2_stat_df,
                col_name=f"{statistic}_correlation",
                group_name=group2_name,
                identifier=statistic,
                title_prefix=_expand_statistic_name(statistic),
                filename=stat_cdf_filename_g2,
                state_color_map=state_color_map,
                filter_state_names=filter_state_names,
                data_type="correlation",
            )
            # if stat_cdf_success_g2:
            #     stat_preview_g2 = IdeasPreviewFile(
            #         name=f"{group2_name} {statistic.capitalize()} Correlation CDF",
            #         help=(
            #             f"Cumulative distribution function (CDF) displaying "
            #             f"{statistic} correlations across experimental states for "
            #             f"{group2_name}. The CDF plot shows the probability "
            #             f"distribution of correlation values."
            #         ),
            #         file_path=os.path.abspath(stat_cdf_filename_g2),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group2_stat_previews.append(stat_preview_g2)

            # Add a boxplot with individual data points for additional visualization
            stat_boxplot_filename_g2 = os.path.join(
                output_dir,
                f"{group2_name}_{statistic}_correlation_boxplot.svg",
            )
            stat_boxplot_success_g2 = create_boxplot_preview(
                data_df=group2_stat_df,
                col_name=f"{statistic}_correlation",
                group_name=group2_name,
                identifier=statistic,
                title_prefix=_expand_statistic_name(statistic),
                filename=stat_boxplot_filename_g2,
                state_color_map=state_color_map,
                filter_state_names=filter_state_names,
                data_type="correlation",
                include_points=True,
            )
            # if stat_boxplot_success_g2:
            #     stat_detailed_preview_g2 = IdeasPreviewFile(
            #         name=f"{group2_name} {statistic.capitalize()} Correlation Boxplot",
            #         help=(
            #             f"Box and whisker plot with individual data points "
            #             f"displaying the distribution of {statistic} correlations "
            #             f"across experimental states for {group2_name}. The plot "
            #             f"shows median values, quartiles, outliers, and "
            #             f"individual data points."
            #         ),
            #         file_path=os.path.abspath(stat_boxplot_filename_g2),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group2_stat_previews.append(stat_detailed_preview_g2)

            total_cells_group2 = (
                group2_stat_df.groupby("file")["cell"].nunique().sum()
                if not group2_stat_df.empty
                else 0
            )
            stat_metadata_g2 = {
                config.IDEAS_METADATA_KEY: {
                    "dataset": {
                        "states": list(group2_stat_df.state.unique()),
                        "group_name": group2_name,
                    },
                    "metrics": {
                        "total_num_cells": int(total_cells_group2),
                        "num_recordings": len(group2_stat_df.file.unique()),
                        "statistic_type": statistic,
                    },
                }
            }
            # group2_combined_stat_file = IdeasFile(
            #     file_key="group2_combined_cell_correlation_data",
            #     file_path=os.path.abspath(group2_combined_stat_file_path),
            #     file_type=FileType.COMBINED_CORRELATION_DATA.value[1],
            #     file_format=FileFormat.CSV_FILE.value[1],
            #     file_structure=FileStructure.TABLE.value[1],
            #     file_category=FileCategory.RESULT.value[1],
            #     preview_files=group2_stat_previews,
            #     add_metadata=stat_metadata_g2,
            # )
            # output_files.append(group2_combined_stat_file)
    # --- End Save Combined Data ---

    has_single_group = not group2_correlation_files

    group_names = [group1_name]
    group_colors = [
        validate_colors(group1_color or "#1f77b4", color_type="group")[0]
    ]
    if group2_correlation_files:
        group2_name = group2_name or "Group 2"
        group2_color = validate_colors(
            group2_color or "#ff7f0e", color_type="group"
        )[0]
        group_names.append(group2_name)
        group_colors.append(group2_color)

    aov, pairwise, all_preview_files = calculate_and_plot_stats(
        avg_data=combined_avg,
        stat_data=combined_stat,
        statistic_name=statistic,
        states=filter_state_names,
        state_colors=state_color_map,
        group_names=group_names,
        group_colors=group_colors,
        output_dir=output_dir,
        has_single_group=has_single_group,
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        data_pairing=data_pairing,
        significance_threshold=significance_threshold,
    )

    # Add analysis parameters to the dataframes for comprehensive CSV output
    analysis_params = {
        "significance_threshold": significance_threshold,
        "multiple_correction": multiple_correction,
        "effect_size_method": effect_size,
        "data_pairing": data_pairing,
    }

    # Add parameters as additional columns to preserve analysis context
    for param, value in analysis_params.items():
        if not aov.empty:
            aov.loc[:, param] = value
        if not pairwise.empty:
            pairwise.loc[:, param] = value

    # Apply final cleanup to remove unwanted columns before saving
    aov_clean = _cleanup_final_csv_columns(aov, "anova")
    pairwise_clean = _cleanup_final_csv_columns(pairwise, "pairwise")

    # population data FILE
    aov_comparison_csv_file = os.path.join(
        output_dir,
        "ANOVA_comparisons.csv",
    )
    aov_clean.to_csv(aov_comparison_csv_file, index=False)

    pairwise_comparison_csv_file = os.path.join(
        output_dir,
        "pairwise_comparisons.csv",
    )
    pairwise_clean.to_csv(pairwise_comparison_csv_file, index=False)

    # Create comparison metadata similar to population data file
    # NOTE: Adjust the metadata keys and variable names (group1_combined_file, group2_combined_file)
    #       and metric names ('total_num_correlations') as needed for correlation data.
    if group2_correlation_files is None:
        group2_correlation_files = []

    if len(group2_correlation_files) > 0:
        comparison_metadata = {
            config.IDEAS_METADATA_KEY: {
                "dataset": {"states": filter_state_names},
                "metrics": {
                    "total_num_cells_group1": int(total_cells_group1),
                    "total_num_cells_group2": int(total_cells_group2),
                },
            }
        }
    else:
        # Define comparison metadata for a single group
        comparison_metadata = {
            config.IDEAS_METADATA_KEY: {
                "dataset": {"states": filter_state_names},
                "metrics": {
                    "total_num_cells_group1": int(total_cells_group1),
                },
            }
        }

    # aov_file = IdeasFile(
    #     file_key="aov_comparison_data",
    #     file_path=os.path.abspath(aov_comparison_csv_file),
    #     file_type=FileType.CORRELATION_COMPARISON_DATA.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TABLE.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     preview_files=all_preview_files,
    #     add_metadata=comparison_metadata,
    # )
    # output_files.append(aov_file)

    # pairwise_file = IdeasFile(
    #     file_key="pairwise_comparison_data",
    #     file_path=os.path.abspath(pairwise_comparison_csv_file),
    #     file_type=FileType.CORRELATION_COMPARISON_DATA.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TABLE.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     preview_files=all_preview_files,
    #     add_metadata=comparison_metadata,
    # )
    # output_files.append(pairwise_file)

    # logger.info("Combination and comparison of correlation data completed")
    # generate_output_manifest(
    #     group1_correlation_files=group1_correlation_files,
    #     group2_correlation_files=group2_correlation_files,
    #     output_files=output_files,
    #     output_dir=output_dir,
    # )


@_suppress_pingouin_warnings
def calculate_and_plot_stats(
    avg_data: pd.DataFrame,
    stat_data: pd.DataFrame,
    statistic_name: str,
    states: List[str],
    state_colors: Dict[str, str],
    group_names: List[str],
    group_colors: List[str],
    output_dir: str,
    has_single_group: bool,
    multiple_correction: str,
    effect_size: str,
    data_pairing: str,
    significance_threshold: float,
):
    """Calculate and plot statistics for correlation data.

    Parameters
    ----------
    avg_data : pd.DataFrame
        DataFrame containing average correlation data (positive & negative)
    stat_data : pd.DataFrame
        DataFrame containing cell-level statistic correlation data (e.g., max_correlation)
    statistic_name : str
        The name of the statistic column in stat_data (e.g., 'max_correlation')
    states : List[str]
        List of state names
    state_colors : Dict[str, str]
        Dictionary mapping state names to colors
    group_names : List[str]
        List of group names
    group_colors : List[str]
        List of colors for groups
    output_dir : str
        Directory to save output files
    has_single_group : bool
        Whether there is only one group
    multiple_correction : str
        Method for multiple comparison correction
    effect_size : str
        Method for effect size calculation
    data_pairing : str
        Type of data pairing
    significance_threshold : float
        Threshold for statistical significance

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[IdeasPreviewFile]]
        ANOVA results, pairwise comparison results, and list of preview files

    """
    aov = pd.DataFrame()
    pairwise = pd.DataFrame()
    preview_files = []

    # --- SUBJECT ID VALIDATION ---
    # Validate subject ID format for clean assignment rules
    # This is critical for ensuring accurate paired/unpaired statistical comparisons
    if avg_data is not None and not avg_data.empty:
        validate_subject_id_format(
            data=avg_data,
            data_pairing=data_pairing,
            group_names=group_names,
            context="average correlation data",
        )

    if stat_data is not None and not stat_data.empty:
        validate_subject_id_format(
            data=stat_data,
            data_pairing=data_pairing,
            group_names=group_names,
            context="statistical correlation data",
        )

    # ------------------------------------------------------------------
    # PART 1: Process average correlation data (positive & negative)
    # ------------------------------------------------------------------
    if avg_data is not None and not avg_data.empty:
        # Define columns for reshaping long format
        id_vars = ["file", "state", "subject_id", "group_name"]
        value_vars = ["positive_correlation", "negative_correlation"]

        # Validate required columns exist
        if not all(col in avg_data.columns for col in id_vars + value_vars):
            logger.warning(
                f"Average correlation data missing required columns {id_vars + value_vars}. "
                "Skipping average correlation analysis."
            )
            avg_data = pd.DataFrame()
        else:
            # Reshape from wide to long format
            mod_data_long = pd.melt(
                avg_data,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name="status",
                value_name="correlation_value",
            )

            # Prepare for statistical analysis
            mod_data_long["Comparison"] = "Average Correlation"
            mod_data_long["file"] = mod_data_long["file"].apply(
                os.path.basename
            )

            # Add this line to create 'group' column from 'group_name'
            if (
                "group_name" in mod_data_long.columns
                and "group" not in mod_data_long.columns
            ):
                mod_data_long["group"] = mod_data_long["group_name"]

            # Add normalized_subject_id column from subject_id for compatibility
            # with statistical functions
            if (
                "subject_id" in mod_data_long.columns
                and "normalized_subject_id" not in mod_data_long.columns
            ):
                # For unpaired analysis, prefix subject IDs with group names to ensure uniqueness
                if data_pairing == "unpaired" and not has_single_group:
                    mod_data_long["normalized_subject_id"] = (
                        mod_data_long["group_name"].str.replace(" ", "_")
                        + "_"
                        + mod_data_long["subject_id"].astype(str)
                    )
                else:
                    mod_data_long["normalized_subject_id"] = mod_data_long[
                        "subject_id"
                    ]

            # Verify all required columns after reshaping
            required_columns = [
                "status",
                "state",
                "Comparison",
                "correlation_value",
                "group_name",
                "subject_id",
                "file",
            ]
            missing_columns = [
                col
                for col in required_columns
                if col not in mod_data_long.columns
            ]

            if missing_columns:
                logger.warning(
                    f"Missing columns after reshaping: {missing_columns}. "
                    "Skipping average correlation analysis."
                )
            else:
                # Calculate statistics using the new direct function
                mod_aov, mod_pairwise = calculate_mod_stats_direct(
                    df=mod_data_long,
                    states=states,
                    correction=multiple_correction,
                    effect_size=effect_size,
                    data_pairing=data_pairing,
                    measure_column="correlation_value",
                    group_column="group_name",
                    comparison_description="Average Correlation",
                    status_values=[
                        "positive_correlation",
                        "negative_correlation",
                    ],
                    state_comparison_type="state_vs_not_state",
                    baseline_state=None,
                )

                # Add metadata for proper categorization
                if not mod_pairwise.empty:
                    mod_pairwise["state_comparison_type"] = (
                        "state" if has_single_group else "state-group"
                    )
                    mod_pairwise["analysis_level"] = "subject"
                    # Set Measure column to match status values for average correlation data
                    if "Status" in mod_pairwise.columns:
                        mod_pairwise["Measure"] = mod_pairwise["Status"]
                    pairwise = pd.concat(
                        [pairwise, mod_pairwise], ignore_index=True
                    )

                if not mod_aov.empty:
                    # Set Measure column to match status values for average correlation data
                    if "Status" in mod_aov.columns:
                        mod_aov["Measure"] = mod_aov["Status"]
                    aov = pd.concat([aov, mod_aov], ignore_index=True)

                # Generate and save visualization
                mod_preview_filename = os.path.join(
                    output_dir,
                    f"average_correlation_distribution{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                )

                # mod_preview_file = IdeasPreviewFile(
                #     name="Average Correlation Distribution",
                #     help=(
                #         "Figure displaying box and whisker plots of average positive and "
                #         "negative correlations across experimental states. The plot shows "
                #         "individual data points, median values, quartiles, and "
                #         "statistical significance indicators for both positive and "
                #         "negative correlation components. This analysis reveals how "
                #         "neural correlation patterns differ between experimental "
                #         "conditions, providing insights into functional connectivity "
                #         "changes across states."
                #     ),
                #     file_path=os.path.abspath(mod_preview_filename),
                #     file_format=FileFormat.SVG_FILE.value[1],
                # )

                # Create visualization using specialized function
                plot_average_correlation_data(
                    mod_data=mod_data_long,
                    states=states,
                    aov=mod_aov,
                    pairwise=mod_pairwise,
                    colors=list(state_colors.values()),
                    mod_filename=mod_preview_filename,
                    data_pairing=data_pairing,
                    group_colors=group_colors,
                    data_type="Average Correlation",
                    group_names=group_names,
                )

                # preview_files.append(mod_preview_file)

    # ------------------------------------------------------------------
    # PART 2: Process statistical correlation data (max, mean, etc.)
    # ------------------------------------------------------------------
    if stat_data is not None and not stat_data.empty:
        # Prepare data for LMM analysis
        act_data_processed = stat_data.copy()
        stat_col_name = f"{statistic_name}_correlation"

        if stat_col_name not in act_data_processed.columns:
            logger.warning(
                f"Statistic column '{stat_col_name}' not found in stat_data. "
                "Skipping statistical analysis."
            )
        else:
            # Rename columns to match expected input format
            # normalized_subject_id already validated by validate_subject_id_format() above

            act_data_processed = act_data_processed.rename(
                columns={
                    stat_col_name: "activity",
                    "group_name": "group",
                }
            )
            act_data_processed[
                "Comparison"
            ] = f"{statistic_name.capitalize()} Correlation"
            act_data_processed["file"] = act_data_processed["file"].apply(
                os.path.basename
            )

            # Validate required columns
            required_columns = [
                "activity",
                "state",
                "Comparison",
                "group",
                "normalized_subject_id",
                "file",
            ]
            missing_columns = [
                col
                for col in required_columns
                if col not in act_data_processed.columns
            ]

            if missing_columns:
                logger.warning(
                    f"Missing columns in processed data: {missing_columns}. "
                    "Skipping statistical analysis."
                )
            else:
                # Use LMM for all analyses to leverage nested cell structure
                # and higher statistical power
                act_aov, act_pairwise = calculate_state_lmm_stats(
                    df=act_data_processed,
                    states=states,
                    has_single_group=has_single_group,
                    data_pairing=data_pairing,
                    effect_size=effect_size,
                    correction=multiple_correction,
                    significance_threshold=significance_threshold,
                )

                # Add consistent metadata for comparison identification
                stat_comparison_name = (
                    f"{statistic_name.capitalize()} Correlation"
                )
                stat_measure_name = f"{statistic_name}_correlation"

                if not act_aov.empty:
                    act_aov["Comparison"] = stat_comparison_name
                    act_aov["Measure"] = stat_measure_name

                if not act_pairwise.empty:
                    act_pairwise["Comparison"] = stat_comparison_name
                    act_pairwise["Measure"] = stat_measure_name

                # Create state LMM visualization
                state_lmm_filename = os.path.join(
                    output_dir,
                    f"{statistic_name.lower()}_correlation_state_lmm"
                    f"{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                )

                # state_lmm_preview_file = IdeasPreviewFile(
                #     name=f"{statistic_name.capitalize()} Correlation State LMM Comparison",
                #     help=(
                #         f"Figure presenting linear mixed model (LMM) analysis "
                #         f"results for {statistic_name} correlation data across "
                #         f"experimental states. The plot displays mean correlation "
                #         f"values with error bars, individual data points, and "
                #         f"statistical significance markers. LMM analysis accounts for "
                #         f"nested cell structure within subjects and provides robust "
                #         f"statistical comparison between states while controlling for "
                #         f"subject-level variability in correlation patterns."
                #     ),
                #     file_path=os.path.abspath(state_lmm_filename),
                #     file_format=FileFormat.SVG_FILE.value[1],
                # )

                # Generate visualization
                plot_state_lmm_comparison(
                    df=act_data_processed,
                    data_type=f"{statistic_name.capitalize()} Correlation",
                    states=states,
                    state_colors=state_colors,
                    has_single_group=has_single_group,
                    group_names=group_names,
                    group_colors=group_colors,
                    state_lmm=act_aov,
                    state_pairwise=act_pairwise,
                    output_filename=state_lmm_filename,
                )

                # preview_files.append(state_lmm_preview_file)

                # Combine results
                if not act_aov.empty:
                    aov = pd.concat([aov, act_aov], ignore_index=True)

                if not act_pairwise.empty:
                    # Ensure metadata is consistent
                    if "state_comparison_type" not in act_pairwise.columns:
                        act_pairwise["state_comparison_type"] = "state"
                    if "analysis_level" not in act_pairwise.columns:
                        act_pairwise["analysis_level"] = "cell"

                    pairwise = pd.concat(
                        [pairwise, act_pairwise], ignore_index=True
                    )

                # For multiple groups, run group ANOVA for complementary global variance detection
                # Keep LMM pairwise tests only (since they're equivalent to ANOVA
                # pairwise but LMM has higher power)
                if not has_single_group:
                    (
                        group_aov,
                        group_pairwise,
                        group_df,
                    ) = calculate_group_anova_stats(
                        df=act_data_processed,
                        states=states,
                        correction=multiple_correction,
                        effect_size=effect_size,
                        data_pairing=data_pairing,
                        significance_threshold=significance_threshold,
                    )

                    # Add consistent metadata to ANOVA results
                    if not group_aov.empty:
                        group_aov["Comparison"] = stat_comparison_name
                        group_aov["Measure"] = stat_measure_name

                    # Add ANOVA main effects to complement LMM fixed effects
                    if not group_aov.empty:
                        aov = pd.concat([aov, group_aov], ignore_index=True)
                        logger.info(
                            f"Added {len(group_aov)} ANOVA main effects for "
                            f"complementary global variance detection"
                        )

                    # Skip ANOVA pairwise tests - they're redundant with LMM pairwise tests
                    logger.info(
                        "Skipping ANOVA pairwise tests "
                        "(redundant with LMM pairwise tests)"
                    )

                    # Create group comparison visualization using the group_df for plotting
                    if group_df is not None and not group_df.empty:
                        anova_type = (
                            "rm_anova"
                            if data_pairing == "paired"
                            else "mixed_anova"
                        )
                        group_anova_filename = os.path.join(
                            output_dir,
                            f"{statistic_name.lower()}_correlation_group_"
                            f"{anova_type}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                        )

                        # group_anova_preview_file = IdeasPreviewFile(
                        #     name=f"{statistic_name.capitalize()} "
                        #     "Correlation Group ANOVA Comparison",
                        #     help=(
                        #         f"Figure displaying analysis of variance (ANOVA) "
                        #         f"results comparing {statistic_name} correlation data "
                        #         f"between experimental groups across different states. "
                        #         f"The plot illustrates group differences with mean "
                        #         f"correlation values, error bars, individual data "
                        #         f"points, and significance indicators. ANOVA provides "
                        #         f"complementary global variance detection to LMM "
                        #         f"analysis, revealing main effects of group and "
                        #         f"state factors on correlation patterns."
                        #     ),
                        #     file_path=os.path.abspath(group_anova_filename),
                        #     file_format=FileFormat.SVG_FILE.value[1],
                        # )

                        # Generate group comparison visualization using LMM pairwise for plotting
                        plot_group_anova_comparison(
                            df=group_df,
                            data_type=f"{statistic_name.capitalize()} Correlation",
                            states=states,
                            group_names=group_names,
                            group_colors=group_colors,
                            data_pairing=data_pairing,
                            group_pairwise=act_pairwise,
                            output_filename=group_anova_filename,
                        )

                        # preview_files.append(group_anova_preview_file)

    # Ensure consistent values in critical columns
    if not aov.empty:
        aov.fillna(
            {"Comparison": "Unknown", "Measure": "Unknown"}, inplace=True
        )

    if not pairwise.empty:
        pairwise.fillna(
            {"Comparison": "Unknown", "Measure": "Unknown"}, inplace=True
        )

        # Apply final cleanup to remove unwanted columns like state_comparison_type
        pairwise = _cleanup_final_csv_columns(pairwise, "pairwise")

    return aov, pairwise, preview_files


@_suppress_pingouin_warnings
def plot_average_correlation_data(
    *,
    mod_data: pd.DataFrame,
    states: List[str],
    aov: pd.DataFrame,
    pairwise: pd.DataFrame,
    colors: List[str],
    mod_filename: str,
    data_pairing: str = "unpaired",
    group_names: Optional[List[str]] = None,
    group_colors: Optional[List[str]] = None,
    data_type: str = "Average Correlation",
) -> None:
    """Plot combined average correlation data across states.

    This function creates visualizations of average positive and negative correlation
    values across different states and potentially groups. It generates a plot file
    showing these patterns.

    Parameters
    ----------
    mod_data : pd.DataFrame
        DataFrame containing average correlation data in long format. Expected columns:
        'state', 'status' (e.g., 'positive_correlation', 'negative_correlation'),
        'correlation_value' (containing the average correlation value), 'group', 'subject_id'.
    states : List[str]
        List of state names
    aov : pd.DataFrame
        ANOVA results DataFrame
    pairwise : pd.DataFrame
        Pairwise comparison results DataFrame
    colors : List[str]
        List of colors for states (used if single group or as fallback)
    mod_filename : str
        Output filename for the plot
    data_pairing : str, optional
        Type of data pairing ("paired" or "unpaired"), by default "unpaired"
    group_names : List[str], optional
        List of group names in order, by default None
    group_colors : List[str], optional
        List of colors for groups in order, by default None
    data_type : str, optional
        Type of data being plotted, used for title/labels, by default "Average Correlation"

    """
    try:
        # Validate input data
        if mod_data is None or mod_data.empty:
            logger.warning(
                f"Cannot plot {data_type} data: Empty dataframe provided."
            )
            # Create an empty plot
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.text(
                0.5,
                0.5,
                f"No {data_type} data available",
                ha="center",
                va="center",
            )
            ax.set_title(f"{data_type.capitalize()} - No Data")
            fig.savefig(
                mod_filename,
                dpi=300,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            return

        if not states:
            raise IdeasError(
                "Cannot plot: No states provided",
            )

        # Validate required columns
        # 'correlation_value' now holds the average correlation value
        required_mod_cols = ["status", "state", "correlation_value"]
        missing_mod_cols = [
            col for col in required_mod_cols if col not in mod_data.columns
        ]
        if missing_mod_cols:
            if (
                "positive_correlation" in mod_data.columns
                or "negative_correlation" in mod_data.columns
            ):
                logger.error(
                    "Input data for plot_average_correlation_data seems to be in wide format."
                    f" Expected long format with columns: {required_mod_cols}"
                )
                raise IdeasError(
                    "Incorrect data format for plotting. Expected long format, received wide.",
                )
            else:
                raise IdeasError(
                    f"Missing required columns in average correlation data: {missing_mod_cols}."
                    " Expected columns include 'status' and 'correlation_value'"
                    " (containing correlation values).",
                )

        # Use 'status' column directly (e.g., 'positive_correlation', 'negative_correlation')
        correlation_types = mod_data["status"].unique()

        # Check if we need to split by group (if multiple groups exist)
        has_multiple_groups = (
            "group_name" in mod_data.columns
            and mod_data["group_name"].nunique() > 1
        )

        # Create a figure for correlation data - create panels for each correlation type
        fig, ax = plt.subplots(
            ncols=len(correlation_types),
            figsize=(5 * len(correlation_types), 5),
        )
        if len(correlation_types) == 1:
            ax = [ax]

        for idx, status in enumerate(correlation_types):
            temp_df = mod_data[mod_data["status"] == status]

            # Skip if we have no data for this status
            if temp_df.empty:
                ax[idx].text(
                    0.5,
                    0.5,
                    f"No {status.replace('_', ' ')} data",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                # Use status directly for title (e.g., 'Positive Correlation')
                title_status = status.replace("_", " ").title()
                ax[idx].set_title(title_status)
                continue

            # Plotting logic (scatter plot with connecting lines for paired data)
            if has_multiple_groups:
                # Get unique groups in the order they appear in the data
                unique_groups = list(temp_df["group_name"].unique())

                # Create a mapping of group names to their colors using the provided lists
                group_color_map = {
                    group: color
                    for group, color in zip(group_names, group_colors)
                    if group in unique_groups
                }

                # Create a scatter plot with group as the hue variable
                for i, state in enumerate(states):
                    state_data = temp_df[temp_df["state"] == state]

                    # Plot each group's data points
                    for j, group_name in enumerate(unique_groups):
                        group_data = state_data[
                            state_data["group_name"] == group_name
                        ]
                        if group_data.empty:
                            continue

                        x_pos = i + (j - len(unique_groups) / 2 + 0.5) * 0.3
                        color = group_color_map.get(
                            group_name, colors[i % len(colors)]
                        )

                        # Plot points (y-axis is the average correlation value)
                        ax[idx].scatter(
                            [x_pos] * len(group_data),
                            group_data["correlation_value"],
                            color=color,
                            s=80,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                            zorder=3,
                            label=group_name if i == 0 else None,
                        )

                        # Add mean line
                        mean_y = group_data["correlation_value"].mean()
                        ax[idx].plot(
                            [x_pos - 0.1, x_pos + 0.1],
                            [mean_y, mean_y],
                            color=color,
                            linewidth=3,
                            zorder=4,
                        )

                    # Add connecting lines for paired data
                    if (
                        data_pairing == "paired"
                        and "subject_id" in state_data.columns
                        and len(unique_groups) > 1
                    ):
                        # Group data by subject ID
                        for subject_id in state_data["subject_id"].unique():
                            subject_data = state_data[
                                state_data["subject_id"] == subject_id
                            ].sort_values(
                                by="group_name",
                                key=lambda col: col.map(
                                    {
                                        name: k
                                        for k, name in enumerate(unique_groups)
                                    }
                                ),
                            )

                            if len(subject_data) == len(unique_groups):
                                x_positions = []
                                y_values = []

                                # Iterate through groups in the defined order
                                for j, group_name in enumerate(unique_groups):
                                    group_row = subject_data[
                                        subject_data["group_name"]
                                        == group_name
                                    ]
                                    if not group_row.empty:
                                        x_pos = (
                                            i
                                            + (
                                                j
                                                - len(unique_groups) / 2
                                                + 0.5
                                            )
                                            * 0.3
                                        )
                                        x_positions.append(x_pos)
                                        y_values.append(
                                            group_row[
                                                "correlation_value"
                                            ].iloc[0]
                                        )

                                # Connect points if we have data for all groups for this subject
                                if len(x_positions) == len(unique_groups):
                                    ax[idx].plot(
                                        x_positions,
                                        y_values,
                                        color="gray",
                                        alpha=0.3,
                                        linewidth=1,
                                        zorder=1,
                                    )

                # Set x-ticks to state names
                ax[idx].set_xticks(range(len(states)))
                ax[idx].set_xticklabels(states)
                # Get significance title using the status (e.g., 'positive_correlation')
                sig_title = _get_mixed_pairwise_sig(pairwise, status)

                # Hide all "no significant difference" messages from plot titles
                no_sig_messages = [
                    "No significant differences between states",
                    "No significant differences",
                    "No pairwise data available",
                    "No p-value column found",
                    "Error retrieving significance info",
                ]

                # Format status for the main title
                status_title = status.replace("_", " ").title()
                if sig_title and not any(
                    no_sig_msg in sig_title for no_sig_msg in no_sig_messages
                ):
                    title = f"{status_title}\n{sig_title}"
                else:
                    title = status_title
                ax[idx].set_title(title)

                # Add legend for group if names are provided
                if group_names and idx == 0:
                    handles, labels = ax[idx].get_legend_handles_labels()
                    # Avoid duplicate labels
                    by_label = dict(zip(labels, handles))
                    if by_label:
                        ax[idx].legend(
                            by_label.values(),
                            by_label.keys(),
                            title="Group",
                            loc="center left",
                            bbox_to_anchor=(1, 0.5),
                            frameon=False,
                        )
            else:
                # Single group case
                for i, state in enumerate(states):
                    state_data = temp_df[temp_df["state"] == state]
                    if state_data.empty:
                        continue

                    # Use group color if provided, otherwise use state color
                    color = (
                        group_colors[0]
                        if (group_colors and len(group_colors) > 0)
                        else colors[i % len(colors)]
                    )

                    # Plot points
                    ax[idx].scatter(
                        [i] * len(state_data),
                        state_data["correlation_value"],
                        color=color,
                        s=80,
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=3,
                        label=(
                            group_names[0] if group_names and i == 0 else None
                        ),
                    )

                    # Add mean line
                    mean_y = state_data["correlation_value"].mean()
                    ax[idx].plot(
                        [i - 0.1, i + 0.1],
                        [mean_y, mean_y],
                        color=color,
                        linewidth=3,
                        zorder=4,
                    )

                    # Add connecting lines for paired data (across states within the single group)
                    if (
                        data_pairing == "paired"
                        and "subject_id" in temp_df.columns
                    ):
                        # Group by subject and plot lines connecting states for each subject
                        for subject_id in temp_df["subject_id"].unique():
                            subject_data = temp_df[
                                temp_df["subject_id"] == subject_id
                            ].copy()
                            # Sort by state according to the input 'states' list order
                            subject_data["state"] = pd.Categorical(
                                subject_data["state"],
                                categories=states,
                                ordered=True,
                            )
                            subject_data = subject_data.sort_values("state")

                            if len(subject_data) > 1:
                                state_indices = [
                                    states.index(s)
                                    for s in subject_data["state"]
                                ]
                                ax[idx].plot(
                                    state_indices,
                                    subject_data["correlation_value"],
                                    color="gray",
                                    alpha=0.3,
                                    linewidth=1,
                                    zorder=1,
                                )

                # Set x-ticks to state names
                ax[idx].set_xticks(range(len(states)))
                ax[idx].set_xticklabels(states)

                # Get significance title
                sig_title = _get_mixed_pairwise_sig(
                    pairwise, status, state_legend_show=True
                )

                # Hide all "no significant difference" messages from plot titles
                no_sig_messages = [
                    "No significant differences between states",
                    "No significant differences",
                    "No pairwise data available",
                    "No p-value column found",
                    "Error retrieving significance info",
                ]

                # Format status for the main title
                status_title = status.replace("_", " ").title()
                if sig_title and not any(
                    no_sig_msg in sig_title for no_sig_msg in no_sig_messages
                ):
                    title = f"{status_title}\n{sig_title}"
                else:
                    title = status_title
                ax[idx].set_title(title)

                # Add legend for group if name is provided
                if group_names and idx == 0:
                    handles, labels = ax[idx].get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    if by_label:
                        ax[idx].legend(
                            by_label.values(),
                            by_label.keys(),
                            title="Group",
                            loc="center left",
                            bbox_to_anchor=(1, 0.5),
                            frameon=False,
                        )

            ax[idx].spines["top"].set_visible(False)
            ax[idx].spines["right"].set_visible(False)
            # Update y-axis label to reflect it's showing correlation values
            ax[idx].set_ylabel(f"{data_type}")

        # Save the plot
        fig.tight_layout()
        fig.savefig(
            mod_filename,
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )

    except Exception as e:
        # Log the specific error that occurred
        logger.error(
            f"Error during plot_average_correlation_data execution: {type(e).__name__}: {str(e)}"
        )
        # Optionally re-raise or handle as needed, but logging helps diagnose
        # For the test, we might want to avoid re-raising if the goal is just to check savefig
        # If savefig should always be called,
        # ensure it happens before potential errors or is in a finally block
        raise IdeasError(
            f"Error plotting average correlation data: {str(e)}",
        )
        # If the function must save something even on error,
        # consider saving an error placeholder plot here


def validate_combine_compare_correlation_data_parameters(
    group1_correlation_files: List[str],
    group1_name: str,
    group2_correlation_files: List[str],
    group2_name: str,
    state_names: str = None,
    state_colors: str = None,
) -> None:
    """Validate the input parameter to the combine-and-compare correlation data tool.

    :param group1_correlation_files: correlation data files from the first group
    :param group1_name: name of the first group
    :param group2_correlation_files: correlation data files from the second group
    :param group2_name: name of the second group
    :param state_names: comma-separated string of state names to filter
    :param state_colors: comma-separated string of colors for visualization
    """
    # Validate group 1 files using generalized function
    validate_file_group(
        files=group1_correlation_files,
        group_name="first input group",
        file_type="correlation",
        min_files=2,
        allow_test_files=True,
    )

    # Validate group 2 if specified
    group2_specified = group2_name is not None or (
        group2_correlation_files is not None
        and len(group2_correlation_files) > 0
    )

    if group2_specified:
        # If group 2 is specified (either by name or by providing files),
        # then both name and sufficient files are required.
        if group2_name is None:
            raise IdeasError(
                "The second input group must have a group name if files are provided.",
            )
        if (
            group2_correlation_files is None
            or len(group2_correlation_files) == 0
        ):
            raise IdeasError(
                "The second input group must contain correlation files if group name is provided.",
            )

        # Validate group 2 files using generalized function
        validate_file_group(
            files=group2_correlation_files,
            group_name="second input group",
            file_type="correlation",
            min_files=2,
            allow_test_files=True,
        )

    # Validate group names using generalized function
    validate_group_names(
        group1_name=group1_name,
        group2_name=group2_name,
        require_group1=False,
        require_group2_if_provided=group2_specified,
    )

    # Validate state names and colors using generalized function
    validate_state_names_and_colors(
        state_names=state_names,
        state_colors=state_colors,
        allow_empty=False,
    )


def read_h5(
    file_path: str,
    filter_states: Optional[List[str]] = None,
    verbose_logging: bool = False,
) -> Tuple[Dict[str, List], List[str]]:
    """Read data from an h5 file.

    :param file_path: Path to the H5 file
    :param filter_states: Optional list of state names to include (if None, include all)
    :param verbose_logging: If True, log detailed information about file structure
    :return: Tuple of (data dictionary, keys list)

    Expected H5 file structure:
    --------------------------
    - Top-level datasets named by state/condition (e.g., 'immobile', 'mobile', 'other')
    - Each dataset should be a 2D correlation matrix where matrix[i,j] represents
      the correlation between cell i and cell j
    - Example: A file with 3 states and 200 cells would have 3 datasets,
      each with shape (200, 200)
    """
    data = {}
    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())

            if verbose_logging:
                logger.info(
                    f"H5 file structure for {os.path.basename(file_path)}:"
                )
                # logger.info(f"  Found {len(keys)} top-level keys: {keys}")

                # Log detailed info about each dataset
                for key in keys:
                    dataset = f[key]
                    if isinstance(dataset, h5py.Dataset):
                        shape = dataset.shape
                        dtype = dataset.dtype
                        # Get statistics safely (handling non-numeric or empty datasets)
                        stats = {}
                        try:
                            if dataset.size > 0 and np.issubdtype(
                                dtype, np.number
                            ):
                                arr = dataset[()]
                                stats = {
                                    "min": np.nanmin(arr),
                                    "max": np.nanmax(arr),
                                    "mean": np.nanmean(arr),
                                    "has_nans": np.any(np.isnan(arr)),
                                }
                        except Exception as e:
                            stats = {"error": str(e)}

                        logger.info(
                            f"    Dataset '{key}': shape={shape}, dtype={dtype}, stats={stats}"
                        )
                    elif isinstance(dataset, h5py.Group):
                        group_keys = list(dataset.keys())
                        logger.info(
                            f"    Group '{key}' with {len(group_keys)} items: {group_keys}"
                        )
                    else:
                        logger.info(
                            f"    '{key}': unknown type {type(dataset)}"
                        )

            # If filter_states is provided, only include matching states
            if filter_states:
                # Use exact matching only for state names
                included_keys = []
                for key in keys:
                    # Check if this key exactly matches any of the requested states
                    if any(
                        state.lower() == key.lower() for state in filter_states
                    ):
                        included_keys.append(key)

                # If no matches found, log a warning but DO NOT use all keys
                # This ensures consistency across files
                if not included_keys:
                    logger.warning(
                        f"No states matching {filter_states} found in {file_path}. "
                        f"Available states: {keys}. This file may be excluded from analysis."
                    )
                    # Return empty data instead of using all keys to maintain filtering consistency
                    return {}, {}
            else:
                included_keys = keys

            # Only read data for included keys
            for key in included_keys:
                try:
                    # Check if this key points to a Dataset (not a Group)
                    if isinstance(f[key], h5py.Dataset):
                        # Check if the dataset contains numeric data we can process
                        if np.issubdtype(f[key].dtype, np.number):
                            # Validate that the dataset is a square matrix (for correlation data)
                            dataset = f[key][()]
                            if (
                                len(dataset.shape) != 2
                                or dataset.shape[0] != dataset.shape[1]
                            ):
                                logger.warning(
                                    f"Dataset '{key}' in {file_path} is not a square matrix. "
                                    f"Shape: {dataset.shape}. Expected a square correlation matrix."
                                )
                                continue

                            data[key] = list(dataset)
                            if verbose_logging:
                                logger.info(
                                    f"  Loaded '{key}' as list with {len(data[key])} items"
                                )
                        else:
                            logger.warning(
                                f"Error: Dataset '{key}' has non-numeric dtype "
                                f"{f[key].dtype}. Skipping."
                            )
                    else:
                        # It's a Group or something else, not a Dataset
                        logger.warning(
                            f"Error: '{key}' is a {type(f[key]).__name__}, not a Dataset. Skipping."
                        )
                        # Include the key in the result with an empty list
                        # This ensures the key appears in the returned keys list
                        data[key] = []
                except Exception as e:
                    logger.warning(
                        f"Error processing key '{key}': {str(e)}. Skipping."
                    )

        return data, list(data.keys())
    except (IOError, OSError) as e:
        raise IdeasError(
            f"Error reading h5 file {file_path}: {str(e)}",
        )


def average_correlations(data: List[dict]) -> pd.DataFrame:
    """Calculate the average positive and negative correlations for each state across
    multiple recordings, excluding diagonal elements.

    This function processes correlation data from a list of recordings. For each state
    within a recording, it gathers all off-diagonal correlation values, and then
    calculates the overall average of the positive off-diagonal values and the overall
    average of the negative off-diagonal values.

    Parameters
    ----------
    data:  List[dict]
        A list of dictionaries where each dictionary
        represents a recording. Each dictionary contains
        the following keys:
        - "data": A dictionary where keys represent states
          (e.g., experimental conditions), and values
          are lists of lists/arrays representing correlation matrices.
          Example: {'state1': [[0.9, 0.1], [0.1, 0.8]], 'state2': ...}
        - "file": A string representing the filename or identifier for the recording.
        - "subject_id": An integer representing the subject identifier.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the average positive and negative off-diagonal correlations
        for each state in each recording, with columns:
        - file: The filename or identifier for the recording
        - state: The state name
        - positive_correlation: The average positive off-diagonal correlation
        - negative_correlation: The average negative off-diagonal correlation
        - subject_id: The subject identifier

    """
    outputs = []
    for recording in data:
        file = recording["file"]
        subject_id = recording["subject_id"]
        correlations = recording["data"]

        for state, corr_list in correlations.items():
            all_off_diagonal_values = []

            if not corr_list:
                logger.warning(
                    f"Empty correlation list found in file {file} for state {state}"
                )
                outputs.append(
                    {
                        "file": file,
                        "state": state,
                        "positive_correlation": np.nan,
                        "negative_correlation": np.nan,
                        "subject_id": subject_id,
                    }
                )
                continue

            # Collect all off-diagonal values for the state
            for cell_idx, corr_row in enumerate(corr_list):
                if corr_row is None or len(corr_row) <= cell_idx:
                    logger.warning(
                        "Skipping invalid corr_row at index"
                        f" {cell_idx} in file {file}, state {state}"
                    )
                    continue

                # Create a boolean mask to exclude the diagonal element
                mask = np.ones(len(corr_row), dtype=bool)
                # Ensure cell_idx is within bounds before masking
                if cell_idx < len(mask):
                    mask[cell_idx] = False
                    off_diagonal_corr = corr_row[mask]
                    all_off_diagonal_values.extend(off_diagonal_corr)
                else:
                    logger.warning(
                        f"cell_idx {cell_idx} out of bounds for mask length {len(mask)} "
                        f"in file {file}, state {state}. Skipping row."
                    )

            # Process the collected off-diagonal values for the state
            if not all_off_diagonal_values:
                logger.warning(
                    f"No valid off-diagonal values found in file {file} for state {state}"
                )
                pos_corr = np.nan
                neg_corr = np.nan
            else:
                # Flatten and filter NaNs
                valid_correlations = np.array(all_off_diagonal_values)
                valid_correlations = valid_correlations[
                    ~np.isnan(valid_correlations)
                ]

                if len(valid_correlations) == 0:
                    logger.warning(
                        "No valid (non-NaN) off-diagonal values"
                        f" after filtering in file {file} for state {state}"
                    )
                    pos_corr = np.nan
                    neg_corr = np.nan
                else:
                    # Calculate overall mean positive and negative correlations
                    positive_mask = valid_correlations > 0
                    negative_mask = valid_correlations < 0

                    pos_corr = (
                        np.mean(valid_correlations[positive_mask])
                        if np.any(positive_mask)
                        else np.nan
                    )
                    neg_corr = (
                        np.mean(valid_correlations[negative_mask])
                        if np.any(negative_mask)
                        else np.nan
                    )

            # Append results for this state
            outputs.append(
                {
                    "file": file,
                    "state": state,
                    "positive_correlation": pos_corr,
                    "negative_correlation": neg_corr,
                    "subject_id": subject_id,
                }
            )

    # Handle case where outputs is empty
    if not outputs:
        return pd.DataFrame(
            columns=[
                "file",
                "state",
                "positive_correlation",
                "negative_correlation",
                "subject_id",
            ]
        )

    # Create DataFrame from the list of dictionaries
    output_df = pd.DataFrame(outputs)

    message = (
        f"Average off-diagonal correlations computed: {output_df.shape[0]}"
        f" rows (expected: num_files x num_states)"
    )
    logger.info(message)
    return output_df


def measure_cells(data: List[dict], correlation_type: str) -> pd.DataFrame:
    """Calculate the mean, min, or max correlation for each cell across multiple recordings.

    This function processes data from a list of recordings, where each recording contains
    state-specific correlation data for multiple cells. The function calculates the specified
    correlation statistic (mean, min, or max) for each cell in each state of a recording and
    aggregates the results into a single pandas DataFrame.

    Parameters
    ----------
    data : List[dict]
        A list of dictionaries where each dictionary represents a recording. Each dictionary
        contains the following keys:
        - "data": A dictionary where keys represent states and values are lists of lists/arrays,
          representing **square** correlation matrices for that state.
          Example: {'state1': [[0.9, 0.1], [0.1, 0.8]], 'state2': ...}
        - "file": A string representing the filename or identifier for the recording.
    correlation_type : str
        The type of correlation statistic to calculate. Must be one of the following:
        - "max": Calculate the maximum correlation value for each cell.
        - "min": Calculate the minimum correlation value for each cell.
        - "mean": Calculate the mean correlation value for each cell.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the calculated correlation statistics for each cell
        in each state of the recordings. The DataFrame includes the following columns:
        - "file": The filename or identifier for the recording.
        - "state": The state corresponding to the correlation data.
        - "cell": The index of the cell in the state.
        - "<correlation_type>_correlation": The calculated correlation statistic
          (e.g., "max_correlation", "min_correlation", or "mean_correlation").

    Raises
    ------
    IdeasError
        If `correlation_type` is not one of "max", "min", or "mean".
        If the correlation matrices are not square (rows must equal columns).

    """
    # Validate correlation type before proceeding
    if correlation_type not in ["max", "min", "mean"]:
        raise IdeasError(
            "Correlation type must be 'max', 'min', or 'mean'.",
        )

    output_rows = []
    # Run through each recording
    for i in range(len(data)):
        for state in data[i]["data"]:
            try:
                # Validate that we have data for this state
                if not data[i]["data"][state] or not isinstance(
                    data[i]["data"][state], list
                ):
                    logger.warning(
                        f"Error: Invalid or empty data for state '{state}' "
                        f"in file {data[i]['file']}. Skipping."
                    )
                    continue

                # Validate that the data structure is square
                # (each row has same number of columns as rows)
                for cell_idx, cell_data in enumerate(data[i]["data"][state]):
                    if cell_data is None or not hasattr(cell_data, "__len__"):
                        logger.warning(
                            f"Error: Invalid cell data for cell {cell_idx}, "
                            f"state '{state}' in file {data[i]['file']}. Skipping."
                        )
                        continue

                    # Check if we have proper correlation matrix rows
                    # (array should have same length as num of cells)
                    if len(cell_data) != len(data[i]["data"][state]):
                        logger.warning(
                            f"Error: Non-square correlation matrix for state '{state}' "
                            f"in file {data[i]['file']}. Row {cell_idx} has length "
                            f"{len(cell_data)} but expected {len(data[i]['data'][state])}. "
                            f"Skipping this state."
                        )
                        continue

                for cell in range(len(data[i]["data"][state])):
                    try:
                        cell_data = data[i]["data"][state][cell]
                        # Check if cell data is valid and not empty
                        if (
                            cell_data is None
                            or not hasattr(cell_data, "__len__")
                            or len(cell_data) == 0
                        ):
                            logger.warning(
                                f"Error: Invalid cell data for cell {cell}, "
                                f"state '{state}' in file {data[i]['file']}. Skipping."
                            )
                            continue
                        # Calculate statistic based on correlation_type
                        if correlation_type == "max":
                            cell_measure = np.nanmax(cell_data)
                        elif correlation_type == "min":
                            cell_measure = np.nanmin(cell_data)
                        elif correlation_type == "mean":
                            cell_measure = np.nanmean(cell_data)

                        # Check if the result is a valid number
                        if np.isnan(cell_measure) or np.isinf(cell_measure):
                            logger.warning(
                                f"Error: {correlation_type} calculation resulted in "
                                f"{cell_measure} for cell {cell}, state '{state}' "
                                f"in file {data[i]['file']}. Skipping."
                            )
                            continue
                        # save to a parsable format
                        output_rows.append(
                            {
                                "file": data[i]["file"],
                                "state": state,
                                "cell": cell,
                                f"{correlation_type}_correlation": cell_measure,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error processing cell {cell} in state '{state}' "
                            f"for file {data[i]['file']}: {str(e)}"
                        )
            except Exception as e:
                logger.warning(
                    f"Error processing state '{state}' in file {data[i]['file']}: "
                    f"{str(e)}"
                )
    # Create DataFrame once at the end
    output_df = pd.DataFrame(output_rows)
    message = (
        f"Cell measures computed: {output_df.shape[0]} cells"
        f" (expected: sum(num_cells_per_file × num_states))"
    )
    logger.info(message)
    # Check if we have any data after all the filtering
    if output_df.empty:
        logger.warning(
            "Error: No valid correlation data found after processing."
        )
    return output_df


def _clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_output_manifest(
    group1_correlation_files: List[str],
    group2_correlation_files: List[str],
    output_files: List,
    output_dir: str,
) -> None:
    """Construct and save output manifest and metadata to disk.

    :param group1_correlation_files: correlation data files from the first group
    :param group2_correlation_files: correlation data files from the second group,
    :param output_files: list of IdeasFile to include in the output manifest
    :param output_dir: path to the output directory
    """
    pass
    # # GROUP 1
    # # source correlation FILES (group 1)
    # source_correlation_files_group1 = [
    #     IdeasFile(
    #         file_key="group1_correlation_files",
    #         file_path=os.path.abspath(f),
    #         file_type=FileType.CORRELATION_DATA.value[1],
    #         file_format=FileFormat.H5_FILE.value[1],
    #         file_structure=FileStructure.TABLE.value[1],
    #         file_category=FileCategory.SOURCE.value[1],
    #     )
    #     for f in group1_correlation_files
    # ]

    # # initialize files to list of files in group 1
    # files = source_correlation_files_group1

    # # GROUP 2
    # if len(group2_correlation_files) > 0:
    #     # source correlation FILES (group 2)
    #     source_correlation_files_group2 = [
    #         IdeasFile(
    #             file_key="group2_correlation_files",
    #             file_path=os.path.abspath(f),
    #             file_type=FileType.CORRELATION_DATA.value[1],
    #             file_format=FileFormat.H5_FILE.value[1],
    #             file_structure=FileStructure.TABLE.value[1],
    #             file_category=FileCategory.SOURCE.value[1],
    #         )
    #         for f in group2_correlation_files
    #     ]
    #     # add group 2 files to list of output files
    #     files.extend(source_correlation_files_group2)

    # # add output files to the list of files
    # files.extend(output_files)

    # # combine-and-compare correlation GROUP
    # combine_compare_correlation_group = IdeasGroup(
    #     group_key="combine_compare_correlation_output",
    #     group_type=GroupType.TOOL_OUTPUT.value[1],
    #     files=files,
    # )
    # # save output manifest & metadata to disk
    # save_output_manifest(combine_compare_correlation_group, output_dir)


def create_preview_file(
    name: str,
    help_text: str,
    file_path: str,
    # file_format: str = FileFormat.SVG_FILE.value[1],
):
    """Create an IdeasPreviewFile with the given parameters.

    Parameters
    ----------
    name : str
        Display name for the preview file
    help_text : str
        Description text providing context about the preview
    file_path : str
        Path where the preview file is or will be saved
    file_format : str, optional
        Format of the file, defaults to SVG

    Returns
    -------
    IdeasPreviewFile
        Configured preview file object ready for inclusion in IdeasFile

    """
    # Validate the output directory exists (if specified)
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        logger.warning(
            f"Output directory for preview file does not exist: {output_dir}"
        )
    # return IdeasPreviewFile(
    #     name=name,
    #     help=help_text,
    #     file_path=file_path,
    #     file_format=file_format,
    # )
