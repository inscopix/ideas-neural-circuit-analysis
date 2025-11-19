import fnmatch
import json
import os
import re
import time
from typing import List, Optional, Tuple, Dict, Union

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
from ideas.exceptions import IdeasError
from ideas.tools.log import get_logger
from ideas.tools.types import IdeasFile
from ideas.tools import outputs
# from toolbox.utils.output_manifest import save_output_manifest

# Import utilities for this tool
from utils.combine_compare_population_data_utils import (
    calculate_mod_stats_direct,
    calculate_state_lmm_stats,
    calculate_group_anova_stats,
    detect_baseline_state,
    detect_state_comparison_type,
    validate_user_specified_comparison_method,
)
from utils.population_data_validation import (
    validate_colors,
    extract_and_validate_states,
    validate_group_names,
    validate_state_names_and_colors,
    validate_file_group,
    validate_subject_id_format,
)
from utils.statistical_formatting import (
    _cleanup_final_csv_columns,
)
from utils.visualization_helpers import (
    plot_modulation_distribution,
    plot_combined_modulation_data,
    plot_state_lmm_comparison,
    plot_group_anova_comparison,
    create_boxplot_preview,
)
from analysis.combine_compare_peri_event_data import assign_modulation


logger = get_logger()


def _combine_group_data_frames(
    df1: Optional[pd.DataFrame],
    df2: Optional[pd.DataFrame],
    group1_name: str,
    group2_name: Optional[str],
) -> Optional[pd.DataFrame]:
    """Combine dataframes from two groups, adding a 'group' column."""
    dfs_to_combine = []

    if df1 is not None:
        df1_copy = df1.copy()
        df1_copy["group"] = group1_name
        dfs_to_combine.append(df1_copy)

    if df2 is not None and group2_name:
        df2_copy = df2.copy()
        df2_copy["group"] = group2_name
        dfs_to_combine.append(df2_copy)

    return (
        pd.concat(dfs_to_combine, ignore_index=True)
        if dfs_to_combine
        else None
    )


def combine_compare_population_data(
    group1_population_activity_files: List[str],
    state_names: str,
    state_colors: str,
    group1_population_events_files: Optional[List[str]] = None,
    group1_name: Optional[str] = None,
    group1_color: Optional[str] = None,
    group2_population_activity_files: Optional[List[str]] = None,
    group2_population_events_files: Optional[List[str]] = None,
    group2_name: Optional[str] = None,
    group2_color: Optional[str] = None,
    modulation_colors: Optional[str] = "red, blue",
    data_pairing: str = "unpaired",
    subject_matching: str = "number",
    significance_threshold: float = 0.05,
    multiple_correction: str = "bonf",
    effect_size: str = "cohen",
    state_comparison_method: str = "auto",
    output_dir: Optional[str] = None,
) -> None:
    """Combine and compare population activity data from two groups.

    This function processes population activity data (and optionally event data) from one or two
    groups, performs statistical analyses to identify differences between states within each group,
    and between the two groups. It generates visualizations of the results and saves all output
    files to the specified directory.

    Parameters
    ----------
    group1_population_activity_files : List[str]
        List of file paths for group 1 population activity data.
    state_names : str
        Comma-separated string of state names to include in the analysis.
    state_colors : str
        Comma-separated string of colors to use for visualizing different states.
    group1_population_events_files : List[str], optional
        List of file paths for group 1 population events data.
    group1_name : str, optional
        Name of group 1, used for labeling outputs.
    group1_color : str, optional
        Color associated with group 1 for visualizations.
    group2_population_activity_files : List[str], optional
        List of file paths for group 2 population activity data.
    group2_population_events_files : List[str], optional
        List of file paths for group 2 population events data.
    group2_name : str, optional
        Name of group 2, used for labeling outputs.
    group2_color : str, optional
        Color associated with group 2 for visualizations.
    modulation_colors : str, optional
        Comma-separated string of colors for up and down modulation. Default is "red, blue".
    data_pairing : str, optional
        Type of data pairing for comparison ("paired" or "unpaired").
    subject_matching : str, optional
        Method for matching subjects between groups for paired analysis.
    significance_threshold : float, optional
        Significance threshold for statistical tests. Default is 0.05.
    multiple_correction : str, optional
        Method for multiple comparison correction. Default is "bonf".
    effect_size : str, optional
        Method for calculating effect size. Default is "cohen".
    state_comparison_method : str, optional
        Method for determining state comparison type. Default is "auto".
    output_dir : str, optional
        Directory to save output files. Default is None,
        which sets it to the current working directory.

    Returns
    -------
    None
        All output files are saved to the specified output directory.

    Raises
    ------
    IdeasError
        If input parameters are invalid or if there are issues processing the data.

    """
    try:
        # Initialize output directory
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        # Initialize lists for output files
        output_files = []

        # Set default group names first
        group1_name = group1_name or "Group 1"
        group2_name = (
            group2_name or "Group 2"
            if group2_population_activity_files
            else None
        )

        # Validate group names using centralized validation
        validate_group_names(
            group1_name=group1_name,
            group2_name=group2_name,
            require_group1=True,
            require_group2_if_provided=bool(group2_population_activity_files),
        )

        # Log pairing mode and subject matching method
        if group2_population_activity_files is not None:
            if data_pairing == "paired":
                logger.info(
                    "Running PAIRED analysis between groups (requires matched samples)"
                )
                logger.debug(
                    f"Using '{subject_matching}' method for subject matching"
                )
            else:
                logger.info(
                    "Running UNPAIRED analysis between groups (standard comparison)"
                )

        # Parse and validate state names and colors using centralized function
        (
            state_names_list,
            original_state_colors_list,
        ) = validate_state_names_and_colors(
            state_names=state_names,
            state_colors=state_colors,
            allow_empty=False,
        )

        # Validate file groups using centralized function
        validate_file_group(
            group1_population_activity_files, "group 1", "activity"
        )

        if group1_population_events_files:
            validate_file_group(
                group1_population_events_files,
                "group 1",
                "events",
                min_files=1,
            )

        if group2_population_activity_files:
            validate_file_group(
                group2_population_activity_files, "group 2", "activity"
            )

            if group2_population_events_files:
                validate_file_group(
                    group2_population_events_files,
                    "group 2",
                    "events",
                    min_files=1,
                )

        # Parse modulation colors if provided
        if modulation_colors:
            modulation_colors_list = validate_colors(
                modulation_colors, color_type="modulation"
            )
        else:
            modulation_colors_list = ["red", "blue"]

        # Validate group colors
        group1_color = validate_colors(
            group1_color or "blue", color_type="group"
        )[0]
        group_names = [group1_name]
        group_colors = [group1_color]

        if group2_population_activity_files:
            group2_color = validate_colors(
                group2_color or "red", color_type="group"
            )[0]
            group_names.append(group2_name)
            group_colors.append(group2_color)

        # Apply subject matching if doing paired analysis with two groups
        if data_pairing == "paired" and group2_population_activity_files:
            # Get matched file pairs
            matched_files = match_subjects(
                group1_files=group1_population_activity_files,
                group2_files=group2_population_activity_files,
                match_method=subject_matching,
            )

            # Update file lists with matched pairs
            group1_population_activity_files = [
                pair[0] for pair in matched_files
            ]
            group2_population_activity_files = [
                pair[1] for pair in matched_files
            ]

            # Update event files if present
            if (
                group1_population_events_files
                and group2_population_events_files
            ):
                matched_event_files = match_subjects(
                    group1_files=group1_population_events_files,
                    group2_files=group2_population_events_files,
                    match_method=subject_matching,
                )
                group1_population_events_files = [
                    pair[0] for pair in matched_event_files
                ]
                group2_population_events_files = [
                    pair[1] for pair in matched_event_files
                ]

            # Validate minimum file requirements after subject matching
            for files, group_name in [
                (group1_population_activity_files, "group 1"),
                (group2_population_activity_files, "group 2"),
            ]:
                if len(files) < 2:
                    raise IdeasError(
                        f"After subject matching, {group_name} has only {len(files)} file(s). "
                        f"At least 2 files per group are required for "
                        f"meaningful statistical analysis.",
                    )

        # SINGLE POINT DETECTION: Detect comparison type and baseline state
        # once from first file
        # This ensures consistency throughout the entire analysis pipeline
        logger.info(
            "Detecting comparison type and " "baseline state from raw data..."
        )
        first_activity_file = group1_population_activity_files[0]

        # Initialize with safe defaults
        global_state_comparison_type = "state_vs_not_state"
        global_baseline_state = None

        try:
            first_df = pd.read_csv(first_activity_file)
            detected_state_comparison_type = detect_state_comparison_type(
                first_df
            )
            detected_baseline_state, _, _ = detect_baseline_state(first_df)

            # Validate states against user-specified or detected comparison type
            (
                validated_global_states,
                confirmed_state_comparison_type,
                confirmed_baseline_state,
            ) = validate_user_specified_comparison_method(
                first_df, state_names_list, state_comparison_method
            )

            # Use confirmed values (may be refined by validation)
            global_state_comparison_type = (
                confirmed_state_comparison_type
                if confirmed_state_comparison_type is not None
                else detected_state_comparison_type
            )
            global_baseline_state = (
                confirmed_baseline_state
                if confirmed_baseline_state is not None
                else detected_baseline_state
            )

        except Exception as e:
            logger.error(
                f"Critical failure: Unable to detect comparison type and baseline state "
                f"from first file {first_activity_file}: {str(e)}"
            )
            raise IdeasError(
                f"Failed to detect comparison type and baseline state from input data. "
                f"This is required for consistent analysis. Error: {str(e)}. "
                f"Please verify that the input files contain valid population activity data "
                f"with proper state columns.",
            )

        # Extract and validate states from all input files
        final_validated_states = extract_and_validate_states(
            group1_population_activity_files=group1_population_activity_files,
            group2_population_activity_files=group2_population_activity_files,
            group1_population_events_files=group1_population_events_files,
            group2_population_events_files=group2_population_events_files,
            state_names=state_names_list,
            reference_df=first_df,
            global_state_comparison_type=global_state_comparison_type,
            global_baseline_state=global_baseline_state,
        )

        # Create global state-to-color mapping from user input
        global_state_color_map = {}
        for i, state in enumerate(state_names_list):
            if i < len(original_state_colors_list):
                global_state_color_map[state] = original_state_colors_list[i]
            else:
                # Fallback for extra states
                global_state_color_map[state] = original_state_colors_list[
                    i % len(original_state_colors_list)
                ]

        logger.info(f"Global state-to-color mapping: {global_state_color_map}")

        # For modulation plots, get colors for non-baseline states only
        if (
            global_state_comparison_type == "state_vs_baseline"
            and global_baseline_state
        ):
            modulation_states = [
                s for s in final_validated_states if s != global_baseline_state
            ]
            modulation_state_colors_list = [
                global_state_color_map.get(s, "#1f77b4")
                for s in modulation_states
            ]
            logger.info(
                "Modulation plot colors (baseline excluded):"
                f" {dict(zip(modulation_states, modulation_state_colors_list))}"
            )
        else:
            # For non-baseline comparisons, use colors for all validated states
            modulation_state_colors_list = [
                global_state_color_map.get(s, "#1f77b4")
                for s in final_validated_states
            ]

        # For activity plots, use the complete mapping (will be applied during visualization)
        activity_state_colors_list = global_state_color_map

        group1_population_file = None
        # Process group 1 data
        (
            group1_population_md,
            group1_df,
            act_mod_group1,
            ev_mod_group1,
            act_group1,
            ev_group1,
            act_raw_mod_group1,
            ev_raw_mod_group1,
        ) = combine_population_data(
            population_activity_files=group1_population_activity_files,
            event_activity_files=group1_population_events_files,
            significance_threshold=significance_threshold,
            group_name=group1_name,
            states=final_validated_states,
            state_colors=modulation_state_colors_list,
            modulation_colors=modulation_colors_list,
            output_dir=output_dir,
            group_id=1,
            multiple_correction=multiple_correction,
            effect_size=effect_size,
            data_pairing=data_pairing,
            global_state_comparison_type=global_state_comparison_type,
            global_baseline_state=global_baseline_state,
        )

        # Initialize group 2 variables with default values
        group2_population_file = None
        group2_df = pd.DataFrame()
        act_mod_group2 = pd.DataFrame()
        ev_mod_group2 = pd.DataFrame()
        act_group2 = pd.DataFrame()
        ev_group2 = pd.DataFrame()
        act_raw_mod_group2 = None
        ev_raw_mod_group2 = None

        # Process group 2 data if provided
        if group2_population_activity_files:
            (
                group2_population_md,
                group2_df,
                act_mod_group2,
                ev_mod_group2,
                act_group2,
                ev_group2,
                act_raw_mod_group2,
                ev_raw_mod_group2,
            ) = combine_population_data(
                population_activity_files=group2_population_activity_files,
                event_activity_files=group2_population_events_files,
                significance_threshold=significance_threshold,
                group_name=group2_name,
                states=final_validated_states,
                state_colors=modulation_state_colors_list,
                modulation_colors=modulation_colors_list,
                output_dir=output_dir,
                group_id=2,
                multiple_correction=multiple_correction,
                effect_size=effect_size,
                data_pairing=data_pairing,
                global_state_comparison_type=global_state_comparison_type,
                global_baseline_state=global_baseline_state,
            )

        # Final validation: ensure we have meaningful data for analysis
        # This catches any cases where files might have been reduced during processing
        if len(group1_population_activity_files) < 2:
            raise IdeasError(
                f"Group 1 has only {len(group1_population_activity_files)} file(s) "
                f"available for analysis. At least 2 files per group are required "
                f"for meaningful statistical analysis.",
            )

        if (
            group2_population_activity_files
            and len(group2_population_activity_files) < 2
        ):
            raise IdeasError(
                f"Group 2 has only {len(group2_population_activity_files)} file(s) "
                f"available for analysis. At least 2 files per group are required "
                f"for meaningful statistical analysis.",
            )

        # Process and visualize group 1 data
        group1_preview_files = []
        group2_preview_files = []
        process_and_visualize_mod_his(
            mod_data=act_mod_group1,
            raw_mod=act_raw_mod_group1,
            group_name=group1_name,
            data_type="activity",
            preview_files=group1_preview_files,
            output_dir=output_dir,
            states=final_validated_states,
            modulation_colors=modulation_colors_list,
        )
        process_and_visualize_mod_his(
            mod_data=ev_mod_group1,
            raw_mod=ev_raw_mod_group1,
            group_name=group1_name,
            data_type="events",
            preview_files=group1_preview_files,
            output_dir=output_dir,
            states=final_validated_states,
            modulation_colors=modulation_colors_list,
        )
        # group1_population_file.preview = group1_preview_files
        # output_files.append([group1_population_file])

        # Process and visualize group 2 data if present
        if group2_population_activity_files:
            group2_preview_files = []
            process_and_visualize_mod_his(
                mod_data=act_mod_group2,
                raw_mod=act_raw_mod_group2,
                group_name=group2_name,
                data_type="activity",
                preview_files=group2_preview_files,
                output_dir=output_dir,
                states=final_validated_states,
                modulation_colors=modulation_colors_list,
            )
            process_and_visualize_mod_his(
                mod_data=ev_mod_group2,
                raw_mod=ev_raw_mod_group2,
                group_name=group2_name,
                data_type="events",
                preview_files=group2_preview_files,
                output_dir=output_dir,
                states=final_validated_states,
                modulation_colors=modulation_colors_list,
            )
            # group2_population_file.preview = group2_preview_files
            # output_files.append([group2_population_file])

        # Combine activity data
        combined_act_mod_long = _combine_group_data_frames(
            act_mod_group1, act_mod_group2, group1_name, group2_name
        )
        combined_act_long = _combine_group_data_frames(
            act_group1, act_group2, group1_name, group2_name
        )

        # Process event data if available
        combined_ev_mod_long = None
        combined_ev_long = None
        if group1_population_events_files or (
            group2_population_events_files and group2_population_activity_files
        ):
            combined_ev_mod_long = _combine_group_data_frames(
                ev_mod_group1, ev_mod_group2, group1_name, group2_name
            )
            combined_ev_long = _combine_group_data_frames(
                ev_group1, ev_group2, group1_name, group2_name
            )

        # Process activity data
        has_single_group = not group2_population_activity_files

        (
            combined_aov,
            combined_pairwise,
            preview_files,
        ) = calculate_and_plot_stats(
            mod_data=combined_act_mod_long,
            act_data=combined_act_long,
            data_type="activity",
            states=final_validated_states,
            state_colors=activity_state_colors_list,
            group_names=group_names,
            group_colors=group_colors,
            output_dir=output_dir,
            has_single_group=has_single_group,
            multiple_correction=multiple_correction,
            effect_size=effect_size,
            data_pairing=data_pairing,
            significance_threshold=significance_threshold,
            modulation_colors=modulation_colors_list,
            group1_population_file=group1_population_file,
            group2_population_file=group2_population_file,
            global_state_comparison_type=global_state_comparison_type,
            global_baseline_state=global_baseline_state,
        )

        # Safely assign comparison labels to non-empty DataFrames
        if not combined_aov.empty:
            combined_aov.loc[:, "Comparison"] = "trace_activity"
        if not combined_pairwise.empty:
            combined_pairwise.loc[:, "Comparison"] = "trace_activity"

        # Process event data
        if combined_ev_mod_long is not None:
            ev_aov, ev_pairwise, ev_preview_files = calculate_and_plot_stats(
                mod_data=combined_ev_mod_long,
                act_data=combined_ev_long,
                data_type="events",
                states=final_validated_states,
                state_colors=activity_state_colors_list,
                group_names=group_names,
                group_colors=group_colors,
                output_dir=output_dir,
                has_single_group=has_single_group,
                multiple_correction=multiple_correction,
                effect_size=effect_size,
                data_pairing=data_pairing,
                significance_threshold=significance_threshold,
                modulation_colors=modulation_colors_list,
                group1_population_file=group1_population_file,
                group2_population_file=group2_population_file,
                global_state_comparison_type=global_state_comparison_type,
                global_baseline_state=global_baseline_state,
            )

            # Safely assign comparison labels to non-empty DataFrames
            if not ev_aov.empty:
                ev_aov.loc[:, "Comparison"] = "event_rate"
            if not ev_pairwise.empty:
                ev_pairwise.loc[:, "Comparison"] = "event_rate"

            # Combine event stats with activity stats
            combined_aov = pd.concat([combined_aov, ev_aov], ignore_index=True)
            combined_pairwise = pd.concat(
                [combined_pairwise, ev_pairwise], ignore_index=True
            )
            preview_files.extend(ev_preview_files)

        final_aov = combined_aov.dropna(axis=1, how="all").copy()
        final_pairwise = combined_pairwise.dropna(axis=1, how="all").copy()

        # Add analysis parameters to the dataframes for comprehensive CSV output
        analysis_params = {
            "significance_threshold": significance_threshold,
            "multiple_correction": multiple_correction,
            "effect_size_method": effect_size,
            "data_pairing": data_pairing,
        }

        # Add parameters as additional columns to preserve analysis context
        for param, value in analysis_params.items():
            if not final_aov.empty:
                final_aov.loc[:, param] = value
            if not final_pairwise.empty:
                final_pairwise.loc[:, param] = value

        output_aov_filename = os.path.join(
            output_dir,
            "aov_comparisons.csv",
        )
        output_pairwise_filename = os.path.join(
            output_dir,
            "pairwise_comparisons.csv",
        )

        # Apply final cleanup to remove unwanted columns before saving
        final_aov = _cleanup_final_csv_columns(final_aov, "anova")
        final_pairwise = _cleanup_final_csv_columns(final_pairwise, "pairwise")

        # save aov and pairwise data to disk
        final_aov.to_csv(output_aov_filename, index=False)
        final_pairwise.to_csv(output_pairwise_filename, index=False)

        if group2_population_activity_files is None:
            group2_population_activity_files = []

        if (
            len(group2_population_activity_files) > 1
            and group2_population_file is not None
        ):
            comparison_metadata = [
                {
                    "key" : "ideas.data.states",
                    "name": "States",
                    "value": final_validated_states
                },
                {
                    "key" : "ideas.metrics.total_num_cells_group1",
                    "name": "Number of cells (first group)",
                    "value": group1_population_md[1]["value"]
                },
                {
                    "key" : "ideas.metrics.total_num_cells_group2",
                    "name": "Number of cells (second group)",
                    "value": group2_population_md[1]["value"]
                }
            ]
            # comparison_metadata = {
            #     config.IDEAS_METADATA_KEY: {
            #         "dataset": {"states": final_validated_states},
            #         "metrics": {
            #             "total_num_cells_group1": group1_population_file.file_metadata[
            #                 "add"
            #             ][
            #                 "ideas"
            #             ][
            #                 "metrics"
            #             ][
            #                 "total_num_cells"
            #             ],
            #             "total_num_cells_group2": group2_population_file.file_metadata[
            #                 "add"
            #             ][
            #                 "ideas"
            #             ][
            #                 "metrics"
            #             ][
            #                 "total_num_cells"
            #             ],
            #         },
            #     }
            # }

            # create filepaths for aov and pairwise
        else:
            # define comparison metadata
            comparison_metadata = [
                {
                    "key" : "ideas.data.states",
                    "name": "States",
                    "value": final_validated_states
                },
                {
                    "key" : "ideas.metrics.total_num_cells_group1",
                    "name": "Number of cells (first group)",
                    "value": group1_population_md[1]["value"]
                },
            ]
            # comparison_metadata = {
            #     config.IDEAS_METADATA_KEY: {
            #         "dataset": {"states": final_validated_states},
            #         "metrics": {
            #             "total_num_cells_group1": group1_population_file.file_metadata[
            #                 "add"
            #             ][
            #                 "ideas"
            #             ][
            #                 "metrics"
            #             ][
            #                 "total_num_cells"
            #             ],
            #         },
            #     }
            # }

        # # add comparison data and plots to output manifest
        # aov_comparison_file = IdeasFile(
        #     file_key="aov_comparison_data",
        #     file_path=os.path.abspath(output_aov_filename),
        #     file_type=FileType.POPULATION_ACTIVITY_COMPARISON_DATA.value[1],
        #     file_format=FileFormat.CSV_FILE.value[1],
        #     file_structure=FileStructure.TABLE.value[1],
        #     file_category=FileCategory.RESULT.value[1],
        #     preview_files=preview_files,
        #     add_metadata=comparison_metadata,
        # )

        # output_files.extend([aov_comparison_file])

        # pairwise_comparison_file = IdeasFile(
        #     file_key="pairwise_comparison_data",
        #     file_path=os.path.abspath(output_pairwise_filename),
        #     file_type=FileType.POPULATION_ACTIVITY_COMPARISON_DATA.value[1],
        #     file_format=FileFormat.CSV_FILE.value[1],
        #     file_structure=FileStructure.TABLE.value[1],
        #     file_category=FileCategory.RESULT.value[1],
        #     preview_files=preview_files,
        #     add_metadata=comparison_metadata,
        # )

        # output_files.extend([pairwise_comparison_file])

        # After extracting act_data and ev_data, add barplots for mean activity and event-rate
        # Group 1: Activity
        if act_group1 is not None and not act_group1.empty:
            barplot_filename = os.path.join(
                output_dir,
                f"{group1_name}_mean_activity_barplot{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
            )
            # Get actual component states from the data
            component_states = sorted(act_group1["state"].unique())
            barplot_success = create_boxplot_preview(
                data_df=act_group1,
                col_name="activity",
                group_name=group1_name,
                identifier="mean_activity",
                title_prefix="Mean Activity",
                filename=barplot_filename,
                state_color_map=activity_state_colors_list,
                filter_state_names=component_states,
                data_type="activity",
                y_limits=None,
            )
            # if barplot_success:
            #     barplot_preview = IdeasPreviewFile(
            #         name=f"{group1_name} Mean Activity Barplot",
            #         help=(
            #             f"Box and whisker plot displaying the distribution of mean neural activity "
            #             f"across experimental states for {group1_name}. The plot shows "
            #             f"median values, quartiles, and outliers for each state."
            #         ),
            #         file_path=os.path.abspath(barplot_filename),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group1_preview_files.append(barplot_preview)
        # Group 1: Event-rate
        if ev_group1 is not None and not ev_group1.empty:
            barplot_filename = os.path.join(
                output_dir,
                f"{group1_name}_mean_event_rate_barplot{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
            )
            # Get actual component states from the data
            component_states = sorted(ev_group1["state"].unique())
            barplot_success = create_boxplot_preview(
                data_df=ev_group1,
                col_name="activity",
                group_name=group1_name,
                identifier="mean_event_rate",
                title_prefix="Mean Event Rate",
                filename=barplot_filename,
                state_color_map=activity_state_colors_list,
                filter_state_names=component_states,
                data_type="event_rate",
                y_limits=None,
            )
            # if barplot_success:
            #     barplot_preview = IdeasPreviewFile(
            #         name=f"{group1_name} Mean Event Rate Barplot",
            #         help=(
            #             f"Box and whisker plot displaying the distribution of mean neural event "
            #             f"rates across experimental states for {group1_name}. The plot "
            #             f"shows median values, quartiles, and outliers for each state."
            #         ),
            #         file_path=os.path.abspath(barplot_filename),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group1_preview_files.append(barplot_preview)
        # Group 2: Activity
        if act_group2 is not None and not act_group2.empty:
            barplot_filename = os.path.join(
                output_dir,
                f"{group2_name}_mean_activity_barplot{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
            )
            # Get actual component states from the data
            component_states = sorted(act_group2["state"].unique())
            barplot_success = create_boxplot_preview(
                data_df=act_group2,
                col_name="activity",
                group_name=group2_name,
                identifier="mean_activity",
                title_prefix="Mean Activity",
                filename=barplot_filename,
                state_color_map=activity_state_colors_list,
                filter_state_names=component_states,
                data_type="activity",
                y_limits=None,
            )
            # if barplot_success:
            #     barplot_preview = IdeasPreviewFile(
            #         name=f"{group2_name} Mean Activity Barplot",
            #         help=(
            #             f"Box and whisker plot displaying the distribution of mean neural activity "
            #             f"across experimental states for {group2_name}. The plot shows "
            #             f"median values, quartiles, and outliers for each state."
            #         ),
            #         file_path=os.path.abspath(barplot_filename),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group2_preview_files.append(barplot_preview)
        # Group 2: Event-rate
        if ev_group2 is not None and not ev_group2.empty:
            barplot_filename = os.path.join(
                output_dir,
                f"{group2_name}_mean_event_rate_barplot{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
            )
            # Get actual component states from the data
            component_states = sorted(ev_group2["state"].unique())
            barplot_success = create_boxplot_preview(
                data_df=ev_group2,
                col_name="activity",
                group_name=group2_name,
                identifier="mean_event_rate",
                title_prefix="Mean Event Rate",
                filename=barplot_filename,
                state_color_map=activity_state_colors_list,
                filter_state_names=component_states,
                data_type="event_rate",
                y_limits=None,
            )
            # if barplot_success:
            #     barplot_preview = IdeasPreviewFile(
            #         name=f"{group2_name} Mean Event Rate Barplot",
            #         help=(
            #             f"Box and whisker plot displaying the distribution of mean neural event "
            #             f"rates across experimental states for {group2_name}. The plot "
            #             f"shows median values, quartiles, and outliers for each state."
            #         ),
            #         file_path=os.path.abspath(barplot_filename),
            #         file_format=FileFormat.SVG_FILE.value[1],
            #     )
            #     group2_preview_files.append(barplot_preview)

        # generate_output_manifest(
        #     group1_population_files=group1_population_activity_files,
        #     group2_population_files=group2_population_activity_files,
        #     output_files=output_files,
        #     output_dir=output_dir,
        # )

        output_metadata = {
            f"population_activity_data_{group1_name}": group1_population_md,
            "aov_comparisons.csv": comparison_metadata,
            "pairwise_comparisons.csv": comparison_metadata,
        }
        if group2_name:
            output_metadata[f"population_activity_data_{group2_name}"] = group2_population_md
        
        with open(os.path.join(output_dir, "output_metadata.json"), "w") as f:
            json.dump(output_metadata, f, indent=4)

    except Exception as e:
        logger.error(f"Error in combine_compare_population_data: {str(e)}")
        raise IdeasError(
            f"Error combining and comparing population data: {str(e)}",
        )


def combine_population_data(
    *,
    population_activity_files: List[str],
    event_activity_files: List[str],
    significance_threshold: float,
    group_name: str,
    states: List[str],
    state_colors: List[str],
    modulation_colors: Optional[List[str]] = None,
    output_dir: str,
    group_id: int,
    multiple_correction: str,
    effect_size: str,
    data_pairing: str = "unpaired",
    global_state_comparison_type: Optional[str] = None,
    global_baseline_state: Optional[str] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """Combine population activity data from multiple recordings together.

    Parameters
    ----------
    population_activity_files : List[str]
        List of file paths for population activity data
    event_activity_files : List[str]
        List of file paths for event activity data
    significance_threshold : float
        Threshold for significance testing
    group_name : str
        Name of the group
    states : List[str]
        List of state names
    state_colors : List[str]
        List of colors for each state
    modulation_colors : List[str], optional
        List of colors for modulation states
    output_dir : str
        Directory to save output files
    group_id : int
        ID of the group
    multiple_correction : str
        Method for multiple comparison correction
    effect_size : str
        Method for effect size calculation
    data_pairing : str, optional
        Data pairing method, defaults to "unpaired"
    global_state_comparison_type : Optional[str]
        Global state comparison type to use for all files
    global_baseline_state : Optional[str]
        Global baseline state to use for all files

    Returns
    -------
    Tuple
        Combined data and analysis results

    """
    start_time = time.time()
    try:
        # Initialize lists to store dataframes
        population_dataframes = []
        event_dataframes = []
        total_cells = 0

        # Read and process population activity files
        for i, file_path in enumerate(population_activity_files):
            try:
                df = pd.read_csv(file_path)
                population_dataframes.append(df)
            except Exception as e:
                raise IdeasError(
                    f"Error reading population activity file {file_path}: {str(e)}",
                )

        # Read and process event activity files if provided
        if event_activity_files:
            for file_path in event_activity_files:
                try:
                    if file_path and os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        if not df.empty:
                            event_dataframes.append(df)
                        else:
                            logger.warning(
                                f"Event activity file {file_path} is empty"
                            )
                    else:
                        logger.warning(
                            f"Event activity file {file_path} does not exist"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error reading event activity file {file_path}: {str(e)}"
                    )
                    continue

        # ========================================================================
        # CLEAN SUBJECT ID ASSIGNMENT - Deterministic, Group-Name Based, No Fallbacks
        # ========================================================================
        #
        # UNPAIRED ANALYSIS:
        #   - Format: "{Group_Name}_subject_N" (e.g., "Control_subject_1", "Treatment_subject_1")
        #   - Purpose: Ensures subjects from different groups are never confused
        #   - Statistical effect: Distinct subjects across groups for independent samples
        #
        # PAIRED ANALYSIS:
        #   - Format: "subject_N" (e.g., "subject_1", "subject_1")
        #   - Purpose: Identical subject IDs across groups enable proper matching
        #   - Statistical effect: Same subjects compared across conditions
        #
        # NO FALLBACKS: Assignment is completely deterministic based on file position
        # ========================================================================

        group_prefix = (
            f"{group_name.replace(' ', '_')}_"
            if data_pairing == "unpaired"
            else ""
        )

        # Log clean subject ID assignment strategy
        if data_pairing == "unpaired":
            logger.info(
                f"Clean subject ID assignment for {group_name}: "
                f"Using group-prefixed format '{group_prefix}subject_N' for unpaired analysis"
            )
        else:
            logger.info(
                f"Clean subject ID assignment for {group_name}: "
                f"Using standard format 'subject_N' for paired analysis"
            )

        # Process activity files with clean subject ID assignment
        for i in range(len(population_activity_files)):
            filename = os.path.basename(population_activity_files[i])
            population_dataframes[i]["file"] = filename

            # CLEAN SUBJECT ID ASSIGNMENT - deterministic, no fallbacks
            # subject_id: clean base identifier (subject_1, subject_2, etc.)
            population_dataframes[i]["subject_id"] = f"subject_{i + 1}"

            # normalized_subject_id: includes group prefix for unpaired analysis
            # Unpaired: "Group_1_subject_1", "Group_2_subject_1" (distinct across groups)
            # Paired: "subject_1", "subject_1" (identical across groups for matching)
            population_dataframes[i][
                "normalized_subject_id"
            ] = f"{group_prefix}subject_{i + 1}"

            # Store total cell count for percentage calculations
            cell_count = len(population_dataframes[i])
            population_dataframes[i]["total_cell_count"] = cell_count
            total_cells += cell_count

            # Identify the comparison type
            population_dataframes[i]["Comparison"] = "trace_activity"

            # Ensure the significance threshold is the same for all recordings
            if (
                significance_threshold is not None
                and significance_threshold > 0
            ):
                population_dataframes[i] = reclassify_neurons(
                    population_dataframes[i], states, significance_threshold
                )
        # Process event files with identical subject ID assignment to activity files
        if event_activity_files:
            for i in range(len(event_activity_files)):
                filename = os.path.basename(event_activity_files[i])

                # Validate event dataframe exists
                if i >= len(event_dataframes):
                    logger.warning(
                        f"Missing event dataframe for {filename}. Skipping processing."
                    )
                    continue

                event_dataframes[i]["file"] = filename
                event_dataframes[i]["total_cell_count"] = len(
                    event_dataframes[i]
                )

                # CLEAN SUBJECT ID ASSIGNMENT - identical to activity files for consistency
                # Must match activity files exactly for proper data alignment
                event_dataframes[i]["subject_id"] = f"subject_{i + 1}"
                event_dataframes[i][
                    "normalized_subject_id"
                ] = f"{group_prefix}subject_{i + 1}"

                # Identify the comparison type
                event_dataframes[i]["Comparison"] = "event_rate"

                # Reclassify neurons based on the specified significance threshold
                if (
                    significance_threshold is not None
                    and significance_threshold > 0
                ):
                    event_dataframes[i] = reclassify_neurons(
                        event_dataframes[i], states, significance_threshold
                    )
        # Combine the activity and event rate dataframes to simplify data handling
        df_activity = pd.concat(population_dataframes, ignore_index=True)

        # Use global detection values instead of detecting again
        if global_state_comparison_type is not None:
            state_comparison_type = global_state_comparison_type
            baseline_state = global_baseline_state
        else:
            # Fallback detection only if global values not provided
            # (should not happen in normal flow)
            logger.warning(
                "Global detection values not provided, performing fallback detection"
            )
            state_comparison_type = detect_state_comparison_type(df_activity)
            baseline_state, _, _ = detect_baseline_state(df_activity)
            logger.info(
                f"Fallback detection complete - Type: {state_comparison_type}, "
                f"Baseline: {baseline_state}"
            )

        # Handle event data processing more gracefully
        if event_dataframes:
            try:
                df_events = pd.concat(event_dataframes, ignore_index=True)
                df_population = pd.concat(
                    [df_activity, df_events], ignore_index=True
                )
                logger.info(
                    f"Successfully processed {len(event_dataframes)}"
                    f" event data files for {group_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not process {group_name} event data:"
                    f" {str(e)}. Using activity data only."
                )
                df_population = df_activity
                event_dataframes = []
        else:
            logger.info(
                f"No valid event data files found for {group_name}. Using activity data only."
            )
            df_population = df_activity

        # save combined population data to disk
        output_population_filename = os.path.join(
            output_dir,
            f"population_activity_data_{group_name}.csv".replace(" ", "_"),
        )
        # Apply cleanup to remove unwanted columns before saving
        df_population_clean = _cleanup_final_csv_columns(
            df_population, "mixed"
        )
        df_population_clean.to_csv(output_population_filename, index=False)

        # extract and organize modulation data for downstream analysis
        act_mod_data, act_raw_mod = extract_combined_modulation_data(
            df_activity, states, state_comparison_type, baseline_state
        )
        act_mod_data["group"] = group_name
        act_mod_data["group_id"] = group_id

        # For paired analysis calculations, ensure we have percentage metrics for modulation
        if "total_cell_count" in df_activity.columns:
            # Ensure modulation data has total_cell_count information
            if "file" in act_mod_data.columns:
                act_mod_data = act_mod_data.merge(
                    df_activity[
                        ["file", "total_cell_count"]
                    ].drop_duplicates(),
                    on="file",
                    how="left",
                )
                # Calculate percentage of modulated cells
                act_mod_data["percent_cells"] = np.round(
                    (
                        act_mod_data["num_cells"]
                        / act_mod_data["total_cell_count"]
                    )
                    * 100,
                    2,
                )

        ev_mod_data = None
        ev_raw_mod = None
        if event_dataframes:
            try:
                ev_mod_data, ev_raw_mod = extract_combined_modulation_data(
                    df_events, states, state_comparison_type, baseline_state
                )
                ev_mod_data["group"] = group_name
                ev_mod_data["group_id"] = group_id

                # Add similar percentage calculation for event modulation data
                if (
                    "file" in ev_mod_data.columns
                    and "total_cell_count" in df_events.columns
                ):
                    ev_mod_data = ev_mod_data.merge(
                        df_events[
                            ["file", "total_cell_count"]
                        ].drop_duplicates(),
                        on="file",
                        how="left",
                    )
                    # Calculate percentage of modulated cells for events
                    ev_mod_data["percent_cells"] = np.round(
                        (
                            ev_mod_data["num_cells"]
                            / ev_mod_data["total_cell_count"]
                        )
                        * 100,
                        2,
                    )

                else:
                    logger.warning(
                        "Missing total_cell_count column in event data,"
                        "skipping percentage calculations"
                    )
            except Exception as e:
                logger.error(
                    f"Error processing event modulation data: {str(e)}"
                )
                raise IdeasError(
                    f"Error processing event modulation data: {str(e)}",
                )

        # Re-order the activity data into a form more usable for stats
        act_data = extract_combined_activity_data(
            df_activity, states, data_type="activity"
        )
        act_data["group"] = group_name
        act_data["group_id"] = group_id

        ev_data = None
        if event_dataframes:
            try:
                ev_data = extract_combined_activity_data(
                    df_events, states, data_type="events"
                )
                ev_data["group"] = group_name
                ev_data["group_id"] = group_id
            except Exception as e:
                logger.warning(
                    f"Error processing event activity data: {str(e)}"
                )

        # Detect comparison type and baseline state for metadata
        # (Already detected earlier, just use the values)

        population_data_metadata = [
            {
                "key": "ideas.dataset.states",
                "name": "States",
                "value": states
            },
            {
                "key": "ideas.metrics.total_num_cells",
                "name": "Number of cells",
                "value": total_cells
            }
        ]

        # population_data_metadata = {
        #     config.IDEAS_METADATA_KEY: {
        #         "dataset": {
        #             "states": states,
        #             "state_comparison_type": state_comparison_type,
        #             "baseline_state": baseline_state,
        #         },
        #         "metrics": {
        #             "total_num_cells": total_cells,
        #             "total_subjects": len(population_activity_files),
        #         },
        #         "analysis_parameters": {
        #             "significance_threshold": significance_threshold,
        #             "multiple_correction": multiple_correction,
        #             "effect_size": effect_size,
        #             "data_pairing": data_pairing,
        #         },
        #         "processing_info": {
        #             "file_count": len(population_activity_files),
        #             "group_name": group_name,
        #             "group_id": group_id,
        #         },
        #     }
        # }

        # # Create population activity file
        # file_key = f"group{group_id}_population_activity_csv_file"
        # population_activity_file = IdeasFile(
        #     file_key=file_key,
        #     file_path=os.path.abspath(output_population_filename),
        #     file_type=FileType.COMBINED_POPULATION_ACTIVITY_DATA.value[1],
        #     file_format=FileFormat.CSV_FILE.value[1],
        #     file_structure=FileStructure.TABLE.value[1],
        #     file_category=FileCategory.RESULT.value[1],
        #     preview_files=[],
        #     add_metadata=population_data_metadata,
        # )

        logger.info(
            f"{group_name} group neural analysis complete"
            f" ({round(time.time() - start_time, 2)} seconds)"
        )

        return (
            population_data_metadata,
            df_population,
            act_mod_data,
            ev_mod_data,
            act_data,
            ev_data,
            act_raw_mod,
            ev_raw_mod,
        )
    except Exception as e:
        logger.error(f"Error in combine_population_data: {str(e)}")
        raise IdeasError(
            f"Error combining population data: {str(e)}",
        )


def process_and_visualize_mod_his(
    mod_data: pd.DataFrame,
    raw_mod: pd.DataFrame,
    group_name: str,
    data_type: str,
    preview_files: List,
    output_dir: str,
    states: List[str],
    modulation_colors: List[str],
) -> None:
    """Process and visualize data for a specific data type.

    Parameters
    ----------
    mod_data : pd.DataFrame
        DataFrame containing modulation data
    raw_mod : pd.DataFrame
        DataFrame containing raw modulation data
    group_name : str
        Name of the group
    data_type : str
        Type of data being processed
    preview_files : List
        List to store preview files
    output_dir : str
        Directory to save output files
    states : List[str]
        List of state names
    modulation_colors : List[str]
        List of colors for modulation states

    Returns
    -------
    None

    """
    if raw_mod is not None:
        filename = os.path.join(
            output_dir,
            f"{data_type}_modulation_distribution_{group_name.replace(' ', '_')}"
            f"{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
        )

        # preview_file = IdeasPreviewFile(
        #     name=f"{data_type.capitalize()} Modulation Distribution",
        #     help=(
        #         f"Stacked bar chart displaying the proportion of neurons with "
        #         f"increased, decreased, and unchanged {data_type} for each "
        #         f"experimental state in the {group_name} group."
        #     ),
        #     file_path=os.path.abspath(filename),
        #     file_format=FileFormat.SVG_FILE.value[1],
        # )
        # preview_files.append(preview_file)

        # Generate the visualization
        plot_modulation_distribution(
            modulation_scores=raw_mod,
            modulation_colors=modulation_colors,
            states=states,
            output_filename=filename,
            group_name=group_name,
            data_type=data_type,
        )


def calculate_and_plot_stats(
    mod_data: pd.DataFrame,
    act_data: pd.DataFrame,
    data_type: str,
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
    modulation_colors: Optional[List[str]] = None,
    group1_population_file = None,
    group2_population_file = None,
    global_state_comparison_type: Optional[str] = None,
    global_baseline_state: Optional[str] = None,
):
    """Calculate and plot statistics for modulation and activity data.

    Parameters
    ----------
    mod_data : pd.DataFrame
        Modulation data
    act_data : pd.DataFrame
        Activity data
    data_type : str
        Type of data ("activity" or "events")
    states : List[str]
        List of state names
    state_colors : List[str]
        List of colors for states
    group_names : List[str]
        List of group names
    group_colors : List[str]
        List of colors for groups
    output_dir : str
        Output directory
    has_single_group : bool
        Whether analysis is for single group
    multiple_correction : str
        Multiple comparison correction method
    effect_size : str
        Effect size measure
    data_pairing : str
        Data pairing method
    significance_threshold : float
        Significance threshold
    modulation_colors : List[str], optional
        List of colors for modulation states
    group1_population_file : Optional[IdeasFile]
        Group 1 population file metadata
    group2_population_file : Optional[IdeasFile]
        Group 2 population file metadata
    global_state_comparison_type : Optional[str]
        Global state comparison type to use for all files
    global_baseline_state : Optional[str]
        Global baseline state to use for all files

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[IdeasPreviewFile]]
        ANOVA results, pairwise results, and preview files

    """
    aov = pd.DataFrame()
    pairwise = pd.DataFrame()
    preview_files = []

    # --- SUBJECT ID VALIDATION ---
    # Validate subject ID format for clean assignment rules
    # This is critical for ensuring accurate paired/unpaired statistical comparisons
    if mod_data is not None and not mod_data.empty:
        validate_subject_id_format(
            data=mod_data,
            data_pairing=data_pairing,
            group_names=group_names,
            context="modulation data",
        )

    if act_data is not None and not act_data.empty:
        validate_subject_id_format(
            data=act_data,
            data_pairing=data_pairing,
            group_names=group_names,
            context="activity data",
        )

    if mod_data is not None and not mod_data.empty:
        # Validate required inputs - fail fast if critical data is missing
        if "status" not in mod_data.columns:
            raise IdeasError(
                f"Missing required column 'status' in {data_type} modulation data. "
                f"This indicates an issue in data processing upstream.",
            )

        if global_state_comparison_type is None:
            raise IdeasError(
                f"global_state_comparison_type must be provided for {data_type} analysis. "
                f"This should have been detected and passed from the main function.",
            )

        # Use provided global detection values (required parameters)
        detected_state_comparison_type = global_state_comparison_type
        detected_baseline_state = global_baseline_state
        validated_states_for_plotting = states

        # For calculate_mod_stats, we pass the `states` it should analyze.
        # These are the `states` originally passed to combine_compare_population_data,
        # which extract_and_validate_states refined based on raw data.
        mod_aov, mod_pairwise = calculate_mod_stats_direct(
            df=mod_data,
            states=states,
            correction=multiple_correction,
            effect_size=effect_size,
            data_pairing=data_pairing,
            significance_threshold=significance_threshold,
            state_comparison_type=detected_state_comparison_type,
            baseline_state=detected_baseline_state,
            use_weighted_proportions=False,
        )

        # Add metadata to identify comparison type and analysis level
        if not mod_pairwise.empty:
            # Note: state_comparison_type removed - will be cleaned up before CSV saving
            mod_pairwise.loc[:, "analysis_level"] = "subject"
            pairwise = pd.concat([pairwise, mod_pairwise], ignore_index=True)

        if not mod_aov.empty:
            # Add similar metadata for ANOVA results if needed
            aov = pd.concat([aov, mod_aov], ignore_index=True)

        # Generate modulation plot
        mod_preview_filename = os.path.join(
            output_dir,
            f"{data_type}_modulation_distribution{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
        )

        # Create IDEASPreviewFile for modulation plot
        # mod_preview_file = IdeasPreviewFile(
        #     name=f"{data_type.capitalize()} Modulation Distribution",
        #     help=(
        #         f"Stacked bar chart displaying the proportion of neurons with "
        #         f"increased, decreased, and unchanged {data_type} for each "
        #         f"experimental state across {', '.join(group_names)} groups."
        #     ),
        #     file_path=os.path.abspath(mod_preview_filename),
        #     file_format=FileFormat.SVG_FILE.value[1],
        # )

        # Use the already detected comparison type and baseline state for plotting
        # This ensures consistency with what was used for calculate_mod_stats
        state_comparison_type_for_plot = detected_state_comparison_type
        baseline_state_for_plot = detected_baseline_state

        # For modulation plots, filter out baseline states and their corresponding colors
        if state_comparison_type_for_plot == "pairwise" and modulation_colors:
            # For pairwise comparisons, use modulation colors as before
            modulation_plot_colors = modulation_colors
            modulation_plot_states = states
        else:
            # For non-pairwise, filter out baseline states for modulation plots
            if baseline_state_for_plot and baseline_state_for_plot in states:
                # Remove baseline state for modulation plots
                modulation_plot_states = [
                    s for s in states if s != baseline_state_for_plot
                ]
            else:
                # No baseline filtering needed
                modulation_plot_states = states

            # Convert state color mapping to list for modulation plots
            modulation_plot_colors = [
                state_colors.get(state, "#1f77b4")
                for state in modulation_plot_states
            ]

        plot_combined_modulation_data(
            mod_data=mod_data,
            states=modulation_plot_states,
            aov=mod_aov,
            pairwise=mod_pairwise,
            colors=modulation_plot_colors,
            mod_filename=mod_preview_filename,
            data_pairing=data_pairing,
            group_colors=group_colors,
            data_type=data_type,
            group_names=group_names,
            state_comparison_type=state_comparison_type_for_plot,
            baseline_state=baseline_state_for_plot,
            validated_states=validated_states_for_plotting,
        )

        # preview_files.append(mod_preview_file)

        if act_data is not None and not act_data.empty:
            required_columns = ["activity", "state", "Comparison", "group"]
            missing_columns = [
                col for col in required_columns if col not in act_data.columns
            ]
            if missing_columns:
                raise IdeasError(
                    f"Missing required columns in {data_type} activity data: {missing_columns}. "
                    f"This indicates an issue in data processing upstream.",
                )

            # For activity analysis, we need to handle pairwise states differently
            # Extract individual component states from pairwise comparisons
            if detected_state_comparison_type == "pairwise":
                activity_states = []
                for state in states:
                    if " vs " in state:
                        # Extract individual states from pairwise comparison
                        state1, state2 = state.split(" vs ")
                        activity_states.extend(
                            [state1.strip(), state2.strip()]
                        )
                    else:
                        activity_states.append(state)
                # Remove duplicates while preserving order
                activity_states = list(dict.fromkeys(activity_states))
                activity_colors = state_colors
                logger.info(
                    f"Using individual component states for activity analysis: {activity_states}"
                )
            else:
                # For non-pairwise comparisons, use ALL states including baseline
                activity_states = states.copy()

                # Explicitly add baseline state for activity analysis
                # if it exists and not already included
                if (
                    detected_baseline_state
                    and detected_baseline_state not in activity_states
                ):
                    activity_states.append(detected_baseline_state)
                    logger.info(
                        f"Added baseline state '{detected_baseline_state}' to activity analysis"
                    )

                # For activity analysis, use global state-to-color mapping directly
                activity_colors = state_colors

            act_aov, act_pairwise = calculate_state_lmm_stats(
                df=act_data,
                states=activity_states,
                has_single_group=has_single_group,
                data_pairing=data_pairing,
                effect_size=effect_size,
                correction=multiple_correction,
                measure_name=data_type,
                significance_threshold=significance_threshold,
            )

        # Create state LMM visualization
        state_lmm_filename = os.path.join(
            output_dir,
            f"{data_type.lower()}_state_lmm{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
        )

        # Create IDEASPreviewFile for state LMM plot
        # state_lmm_preview_file = IdeasPreviewFile(
        #     name=f"{data_type.capitalize()} State LMM Comparison",
        #     help=(
        #         f"Statistical comparison of {data_type} data across experimental "
        #         f"states using Linear Mixed Models (LMM). The plot displays mean "
        #         f"values with error bars and individual data points."
        #     ),
        #     file_path=os.path.abspath(state_lmm_filename),
        #     file_format=FileFormat.SVG_FILE.value[1],
        # )

        plot_state_lmm_comparison(
            df=act_data,
            data_type=data_type,
            states=activity_states,
            state_colors=activity_colors,
            has_single_group=has_single_group,
            group_names=group_names,
            group_colors=group_colors,
            state_lmm=act_aov,
            state_pairwise=act_pairwise,
            output_filename=state_lmm_filename,
        )

        # preview_files.append(state_lmm_preview_file)

        # For single group, use state LMM results directly
        if has_single_group:
            if not act_aov.empty:
                aov = pd.concat([aov, act_aov], ignore_index=True)
            if not act_pairwise.empty:
                pairwise = pd.concat(
                    [pairwise, act_pairwise], ignore_index=True
                )
        else:
            # For multiple groups, calculate group comparison statistics

            (
                group_aov,
                group_pairwise,
                group_df,
            ) = calculate_group_anova_stats(
                df=act_data,
                states=activity_states,
                correction=multiple_correction,
                effect_size=effect_size,
                data_pairing=data_pairing,
                measure_name=data_type,
                significance_threshold=significance_threshold,
            )

            # Add ANOVA main effects to complement LMM fixed effects
            if not group_aov.empty:
                aov = pd.concat([aov, group_aov], ignore_index=True)

            # Skip ANOVA pairwise tests - they're redundant with LMM pairwise tests
            logger.info(
                "Skipping ANOVA pairwise tests "
                "(redundant with LMM pairwise tests)"
            )

            # Create group comparison visualization using LMM pairwise for plotting
            if group_df is not None and not group_df.empty:
                anova_type = (
                    "rm_anova" if data_pairing == "paired" else "mixed_anova"
                )
                group_anova_filename = os.path.join(
                    output_dir,
                    f"{data_type.lower()}_group_{anova_type}"
                    f"{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                )

                # Create IDEASPreviewFile for group ANOVA plot
                # group_anova_preview_file = IdeasPreviewFile(
                #     name=f"{data_type.capitalize()} Group ANOVA Comparison",
                #     help=(
                #         f"Statistical comparison of {data_type} data between "
                #         f"experimental groups across different states using Analysis "
                #         f"of Variance (ANOVA). The plot shows group differences with "
                #         f"mean values, error bars, and individual data points."
                #     ),
                #     file_path=os.path.abspath(group_anova_filename),
                #     file_format=FileFormat.SVG_FILE.value[1],
                # )

                # Generate group comparison visualization using LMM pairwise
                # instead of ANOVA pairwise
                plot_group_anova_comparison(
                    df=group_df,
                    data_type=data_type,
                    states=activity_states,
                    group_names=group_names,
                    group_colors=group_colors,
                    data_pairing=data_pairing,
                    group_pairwise=act_pairwise,
                    output_filename=group_anova_filename,
                )

                # preview_files.append(group_anova_preview_file)

            # Combine state LMM results (avoiding duplication with ANOVA pairwise)
            if not act_aov.empty:
                aov = pd.concat([aov, act_aov], ignore_index=True)
            if not act_pairwise.empty:
                pairwise = pd.concat(
                    [pairwise, act_pairwise], ignore_index=True
                )

    return aov, pairwise, preview_files


def adjust_state_colors_for_state_comparison_type(
    state_colors_list: List[str],
    states: List[str],
    state_comparison_type: str,
) -> List[str]:
    """Adjust state colors based on comparison type and number of states.

    Parameters
    ----------
    state_colors_list : List[str]
        Input list of state colors (may be empty)
    states : List[str]
        List of validated states
    state_comparison_type : str
        Detected comparison type ("pairwise", "state_vs_baseline", etc.)

    Returns
    -------
    List[str]
        Adjusted list of state colors appropriate for the comparison type

    """
    default_state_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Determine if this is a pairwise comparison
    is_pairwise_comparison = state_comparison_type == "pairwise" or any(
        " vs " in state for state in states
    )

    if not state_colors_list:
        # Generate default colors based on comparison type
        if is_pairwise_comparison:
            # For pairwise, use one color per comparison pair
            num_pairs = len(states)
            state_colors_list = default_state_colors[: max(1, num_pairs)]
            logger.info(
                f"No state colors provided for pairwise comparison. "
                f"Using {len(state_colors_list)} default colors: {state_colors_list}"
            )
        else:
            # For non-pairwise, use one color per state
            state_colors_list = default_state_colors[: len(states)]
            logger.warning(
                f"No state colors provided. Using default colors for {len(states)} states: "
                f"{state_colors_list}"
            )
    elif is_pairwise_comparison:
        # For pairwise comparisons, ensure we have enough colors for all pairs
        num_pairs = len(states)

        if len(state_colors_list) < num_pairs:
            # Extend colors by cycling through provided colors
            original_colors = state_colors_list.copy()
            while len(state_colors_list) < num_pairs:
                additional_needed = num_pairs - len(state_colors_list)
                state_colors_list.extend(original_colors[:additional_needed])
            logger.info(
                f"Extended state colors to {num_pairs} colors for pairwise comparisons: "
                f"{state_colors_list}"
            )
        elif len(state_colors_list) > num_pairs:
            # Trim to exact number needed
            state_colors_list = state_colors_list[:num_pairs]
            logger.info(
                f"Using {num_pairs} colors for pairwise comparisons: {state_colors_list}"
            )
        else:
            logger.info(
                f"Using provided {len(state_colors_list)} colors for {num_pairs} "
                f"pairwise comparisons: {state_colors_list}"
            )
    else:
        # For non-pairwise comparisons, adjust colors to match number of states
        if len(state_colors_list) < len(states):
            # Extend with default colors
            additional_colors = default_state_colors[
                len(state_colors_list) : len(states)
            ]
            state_colors_list.extend(additional_colors)
            logger.warning(
                f"Extended state colors to match {len(states)} non-pairwise states. "
                f"Final colors: {state_colors_list}"
            )
        elif len(state_colors_list) > len(states):
            # Truncate to match number of states
            state_colors_list = state_colors_list[: len(states)]
            logger.warning(
                f"Truncated state colors to match {len(states)} non-pairwise states. "
                f"Final colors: {state_colors_list}"
            )

    # Ensure we always return at least one color
    if not state_colors_list:
        state_colors_list = [default_state_colors[0]]

    return state_colors_list


def match_subjects(
    group1_files: List[str],
    group2_files: List[str],
    match_method: str = "number",
) -> List[Tuple[str, str]]:
    """Match subjects between two groups based on the specified method.

    This function matches subjects between two groups using various matching strategies.
    It supports three matching methods:
    1. "number": Extracts and matches numbers from filenames
    2. "filename": Matches exact filenames
    3. "order": Matches files based on their order in the lists

    Parameters
    ----------
    group1_files : List[str]
        List of file paths for group 1 subjects
    group2_files : List[str]
        List of file paths for group 2 subjects
    match_method : str, optional
        Method for matching subjects between groups:
        - "number": Match by extracting numbers from filenames (default)
        - "filename": Match by exact filename
        - "order": Match by order in file lists

    Returns
    -------
    List[Tuple[str, str]]
        List of matched file path pairs (group1_file, group2_file)

    Examples
    --------
    >>> files1 = ["subject_1.csv", "subject_2.csv"]
    >>> files2 = ["patient1.csv", "patient2.csv"]
    >>> matches = match_subjects(files1, files2, "number")
    >>> print(matches)
    [("subject_1.csv", "patient1.csv"), ("subject_2.csv", "patient2.csv")]

    Raises
    ------
    IdeasError
        If input parameters are invalid or if no matches can be found

    """
    if not group1_files or not group2_files:
        raise IdeasError(
            "Both group1_files and group2_files must be non-empty lists",
        )

    if match_method not in ["number", "filename", "order"]:
        raise IdeasError(
            f"Invalid match_method: '{match_method}'. Must be one of: number, filename, order",
        )

    matched_pairs = []
    logger.info(f"Attempting to match subjects using {match_method} method")

    def extract_numbers(filepath: str) -> Optional[int]:
        """Extract all digits from a filename as a single number.

        Examples
        --------
        - pairwise_group1_subject1.csv → 11 (from group1 + subject1)
        - data_001_session2.csv → 0012 (from 001 + 2)
        - subject_5.csv → 5

        """
        filename = os.path.basename(filepath)
        # Remove file extension for cleaner matching
        name_without_ext = os.path.splitext(filename)[0]

        # Extract all digits and concatenate them
        all_digits = re.sub(r"[^\d]", "", name_without_ext)

        if not all_digits:
            logger.debug(f"No digits found in filename: {filename}")
            return None

        # Convert to integer
        number = int(all_digits)
        logger.debug(
            f"Extracted number {number} from all digits in filename: {filename}"
        )
        return number

    def create_lookup_dict(
        files: List[str], method: str
    ) -> Dict[Union[int, str], str]:
        """Create a lookup dictionary based on the matching method."""
        lookup = {}
        for file in files:
            if method == "number":
                num = extract_numbers(file)
                if num is not None:
                    lookup[num] = file
            elif method == "filename":
                lookup[os.path.basename(file)] = file
        return lookup

    try:
        if match_method == "number":
            # Create lookup dictionaries with extracted numbers as keys
            group1_dict = create_lookup_dict(group1_files, "number")
            group2_dict = create_lookup_dict(group2_files, "number")

            # Log extracted numbers for debugging
            group1_numbers = set(group1_dict.keys())
            group2_numbers = set(group2_dict.keys())
            logger.debug(
                f"Group 1 extracted numbers: {sorted(group1_numbers) if group1_numbers else 'None'}"
            )
            logger.debug(
                f"Group 2 extracted numbers: {sorted(group2_numbers) if group2_numbers else 'None'}"
            )

            # Find common numbers between the two groups
            common_numbers = group1_numbers.intersection(group2_numbers)

            if not common_numbers:
                # Provide more specific information about why matching failed
                if not group1_numbers and not group2_numbers:
                    logger.warning(
                        "No numbers found in filenames from either group. "
                        "Filename examples: Group 1: {}, Group 2: {}. "
                        "Falling back to order matching.".format(
                            [os.path.basename(f) for f in group1_files[:3]],
                            [os.path.basename(f) for f in group2_files[:3]],
                        )
                    )
                elif not group1_numbers:
                    logger.warning(
                        f"No numbers found in Group 1 filenames. "
                        f"Group 1 examples: {[os.path.basename(f) for f in group1_files[:3]]}. "
                        "Falling back to order matching."
                    )
                elif not group2_numbers:
                    logger.warning(
                        f"No numbers found in Group 2 filenames. "
                        f"Group 2 examples: {[os.path.basename(f) for f in group2_files[:3]]}. "
                        "Falling back to order matching."
                    )
                else:
                    # Show which files produced which numbers for clarity
                    group1_file_number_map = {
                        os.path.basename(file): extract_numbers(file)
                        for file in group1_files
                    }
                    group2_file_number_map = {
                        os.path.basename(file): extract_numbers(file)
                        for file in group2_files
                    }

                    logger.warning(
                        f"No matching numbers found between groups. "
                        f"Group 1 numbers: {sorted(group1_numbers)} from files "
                        f"{dict(sorted(group1_file_number_map.items(), key=lambda x: x[1] or 0))}, "
                        f"Group 2 numbers: {sorted(group2_numbers)} from files "
                        f"{dict(sorted(group2_file_number_map.items(), key=lambda x: x[1] or 0))}. "
                        "Falling back to order matching."
                    )
                return match_subjects(group1_files, group2_files, "order")

            # Create pairs of matched files
            matched_pairs = [
                (group1_dict[num], group2_dict[num]) for num in common_numbers
            ]
            # Sort by number for consistency
            matched_pairs.sort(key=lambda x: extract_numbers(x[0]))

        elif match_method == "filename":
            # Create lookup dictionaries with filenames as keys
            group1_dict = create_lookup_dict(group1_files, "filename")
            group2_dict = create_lookup_dict(group2_files, "filename")

            # Find common filenames
            common_filenames = set(group1_dict.keys()).intersection(
                set(group2_dict.keys())
            )

            if not common_filenames:
                logger.warning(
                    "No matching filenames found between groups. Falling back to number matching."
                )
                return match_subjects(group1_files, group2_files, "number")

            # Create pairs of matched files
            matched_pairs = [
                (group1_dict[name], group2_dict[name])
                for name in common_filenames
            ]
            # Sort by filename for consistency
            matched_pairs.sort(key=lambda x: os.path.basename(x[0]))

        else:
            # Pair files based on their order in the lists
            # Take the smaller number of files to ensure equal pairs
            min_files = min(len(group1_files), len(group2_files))
            matched_pairs = [
                (group1_files[i], group2_files[i]) for i in range(min_files)
            ]

        # Log matching results
        if matched_pairs:
            logger.info(
                f"Successfully matched {len(matched_pairs)} "
                f"subject pairs using {match_method} matching"
            )
            for i, (f1, f2) in enumerate(matched_pairs[:5]):
                logger.info(
                    f"Match {i+1}: {os.path.basename(f1)} ↔ {os.path.basename(f2)}"
                )
            if len(matched_pairs) > 5:
                logger.debug(f"... and {len(matched_pairs)-5} more pairs")

            # Validate minimum requirements for statistical analysis
            if len(matched_pairs) < 2:
                logger.error(
                    f"Insufficient matched pairs for statistical analysis: "
                    f"{len(matched_pairs)} pair(s) found. "
                    f"At least 2 matched pairs are required for meaningful "
                    f"statistical comparison. "
                    f"Available files - Group 1: {len(group1_files)}, Group 2: {len(group2_files)}"
                )
                raise IdeasError(
                    f"Subject matching produced only {len(matched_pairs)} matched pair(s). "
                    f"At least 2 matched pairs are required for statistical analysis. "
                    f"Check that your files have consistent naming patterns "
                    f"for {match_method} matching, "
                    f"or try a different subject_matching method.",
                )
        else:
            logger.warning(
                f"No subject pairs could be matched using {match_method} matching"
            )
            # If all matching methods failed, force order matching as a last resort
            if match_method != "order":
                logger.warning("Forcing order-based matching as last resort")
                return match_subjects(group1_files, group2_files, "order")
            else:
                # Even order matching failed - this shouldn't happen unless one group is empty
                raise IdeasError(
                    f"All subject matching methods failed. "
                    f"Group 1 has {len(group1_files)} files, "
                    f"Group 2 has {len(group2_files)} files. "
                    f"Both groups must have at least 2 files for statistical analysis.",
                )

        return matched_pairs

    except Exception as e:
        logger.error(f"Error in match_subjects: {str(e)}")
        raise IdeasError(
            f"Error matching subjects: {str(e)}",
        )


def reclassify_neurons(
    df: pd.DataFrame, states: List[str], significance_threshold: float
) -> pd.DataFrame:
    """Reclassify neurons into the appropriate modulation group based on a new p-value threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing modulation scores and p-values
    states : List[str]
        List of state names to process
    significance_threshold : float
        P-value threshold for determining significance

    Returns
    -------
    pd.DataFrame
        Updated dataframe with reclassified neurons

    Notes
    -----
    This function goes through each state and assigns modulation status to neurons
    based on their modulation scores and p-values, using the specified significance threshold.

    """
    if df is None or df.empty:
        raise IdeasError(
            "Cannot reclassify neurons: Empty dataframe",
        )

    try:
        # Create a copy to avoid modifying the original dataframe
        updated_df = df.copy()

        # Collect modulation counts for summary log instead of individual logs
        modulation_summary = {}

        for state in states:
            score_col = f"modulation scores in {state}"
            p_col = f"p-values in {state}"
            mod_col = f"modulation in {state}"

            # Check if required columns exist
            if score_col not in updated_df.columns:
                logger.debug(f"Column '{score_col}' not found in dataframe")
                continue

            if p_col not in updated_df.columns:
                logger.debug(f"Column '{p_col}' not found in dataframe")
                continue

            # Apply the modulation assignment function to each row
            updated_df[mod_col] = updated_df.apply(
                lambda row: assign_modulation(
                    row=row,
                    threshold=significance_threshold,
                    score_col=score_col,
                    p_col=p_col,
                ),
                axis=1,
            )

            # Collect counts but don't log individually
            if mod_col in updated_df.columns:
                up_mod = (updated_df[mod_col] == 1).sum()
                down_mod = (updated_df[mod_col] == -1).sum()
                non_mod = (updated_df[mod_col] == 0).sum()
                modulation_summary[state] = (up_mod, down_mod, non_mod)

        # Log a summary of modulation counts for all states
        for state, (up, down, non) in modulation_summary.items():
            logger.debug(
                f"{state.capitalize()} state modulation: {up} "
                f"neurons increased activity, {down} decreased, {non} unchanged"
            )

        return updated_df
    except Exception as e:
        logger.error(f"Error in reclassify_neurons: {str(e)}")
        raise IdeasError(
            f"Error reclassifying neurons: {str(e)}",
        )


def extract_combined_activity_data(
    df: pd.DataFrame, states: List[str], data_type: str = "activity"
) -> pd.DataFrame:
    """Extract and reorganize activity data from combined population data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the combined population data
    states : List[str]
        List of state names to extract data from combined population data
    data_type : str, optional
        Type of data being processed ("activity" or "events"), by default "activity"

    Returns
    -------
    pd.DataFrame
        Reorganized dataframe with activity data by state

    Notes
    -----
    This function extracts activity measurements for each state from columns matching
    patterns like "mean * in {state}" and organizes them into a long-format dataframe
    suitable for statistical analysis and visualization. For state vs baseline comparisons,
    it automatically includes the baseline state in activity analysis.

    """
    try:
        if df is None or df.empty:
            raise IdeasError(
                "Cannot extract activity data: Empty input dataframe",
            )

        if not states:
            raise IdeasError(
                "Cannot extract activity data: No states provided",
            )

        # Detect baseline state and get all available states for activity analysis
        baseline_state, mod_states, mean_states = detect_baseline_state(df)

        # For activity analysis, we want to include ALL states with mean activity data
        # This includes both modulation states and baseline states
        states_for_activity = states.copy()

        # Add baseline state if detected and not already in the list
        if baseline_state and baseline_state not in states_for_activity:
            states_for_activity.append(baseline_state)
            logger.info(
                f"Added baseline state '{baseline_state}' to activity analysis"
            )

        # Also check for any other states in mean activity columns that might have been missed
        # This handles all possible scaling methods from population_activity.py
        found_additional_states = []
        for col in df.columns:
            # Check if this is a mean activity column with any scaling method
            is_mean_activity_col = (
                col.startswith("mean ")
                and " in " in col
                and any(
                    unit_type in col.lower()
                    for unit_type in [
                        "activity",
                        "fractional change",
                        "z-score",
                        "normalized",
                        "event rate",
                    ]
                )
            )

            if is_mean_activity_col:
                state = col.split(" in ")[-1].strip()
                if state and state not in states_for_activity:
                    states_for_activity.append(state)
                    found_additional_states.append(state)

        output_df = pd.DataFrame()

        # Process each comparison type (trace_activity, event_rate, etc.)
        for comp in df.Comparison.unique():
            # Get subset of data for this comparison type
            m_df = df[df["Comparison"] == comp]

            # Process each state (including baseline for activity analysis)
            for state in states_for_activity:
                # Find columns that match the pattern "mean * in {state}"
                mean_cols = [
                    c
                    for c in m_df.columns
                    if fnmatch.fnmatch(c, f"mean * in {state}")
                ]

                if not mean_cols:
                    # Hide messages for pairwise comparison states (containing "vs")
                    if " vs " not in state:
                        logger.info(
                            f"No mean columns found for state '{state}' in comparison '{comp}'"
                        )
                    continue

                if len(mean_cols) > 1:
                    logger.warning(
                        "Multiple mean columns found "
                        f"for state '{state}': {mean_cols}. Using first."
                    )

                # Use the first matching column
                mean_col = mean_cols[0]

                # Extract the unit information from the column name
                # Remove 'mean ' prefix and ' in {state}' suffix to get the unit
                unit = (
                    mean_col.replace("mean ", "")
                    .replace(f" in {state}", "")
                    .strip()
                )

                # Create a dictionary with the extracted data
                # This ensures proper subject identification for statistical analysis
                state_data = {
                    "name": m_df[
                        "normalized_subject_id"
                    ],  # Use clean subject ID, not cell names
                    "file": m_df["file"],
                    "Comparison": comp,
                    "state": state,
                    "activity": m_df[mean_col],
                    "unit": unit,
                    "Measure": unit,
                }

                # Add baseline information if this is a baseline comparison
                if baseline_state:
                    state_data["baseline_state"] = baseline_state
                    state_data["is_baseline"] = state == baseline_state

                # Add subject IDs if they exist in the input dataframe
                if "subject_id" in m_df.columns:
                    state_data["subject_id"] = m_df["subject_id"]
                if "normalized_subject_id" in m_df.columns:
                    state_data["normalized_subject_id"] = m_df[
                        "normalized_subject_id"
                    ]
                if "total_cell_count" in m_df.columns:
                    state_data["total_cell_count"] = m_df["total_cell_count"]

                # Add this state's data to the output dataframe
                state_df = pd.DataFrame(state_data)
                output_df = pd.concat([output_df, state_df], ignore_index=True)

        if output_df.empty:
            raise IdeasError(
                "Cannot extract activity data: No columns match 'mean * in <state>'",
            )

        # Log summary of what was processed
        if found_additional_states:
            logger.info(
                f"Found additional states in activity columns: {found_additional_states}"
            )
        if baseline_state:
            logger.info(
                f"Included baseline state '{baseline_state}' in activity analysis"
            )

        return output_df

    except Exception as e:
        logger.error(f"Error in extract_combined_activity_data: {str(e)}")
        raise IdeasError(
            f"Error extracting combined {data_type} data: {str(e)}",
        )


def extract_combined_modulation_data(
    df: pd.DataFrame,
    states: List[str],
    state_comparison_type: Optional[str] = None,
    baseline_state: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract, group, and average neurons across input files by modulation group.

    This function processes modulation data from population activity files, extracting
    modulation scores, p-values, and modulation status. It supports multiple comparison types:
    - Pairwise comparisons (state1 vs state2)
    - State vs baseline comparisons
    - Single state modulation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing population activity data
    states : List[str]
        List of state names to extract data from combined population data
    state_comparison_type : Optional[str]
        Pre-detected comparison type to avoid re-detection on processed data
    baseline_state : Optional[str]
        Pre-detected baseline state if applicable

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing:
        - Modulation data summarized by state and modulation status
        - Raw modulation scores for visualization

    """
    try:
        if df is None or df.empty:
            raise IdeasError(
                "Cannot extract modulation data: Empty input dataframe",
            )

        if not states:
            raise IdeasError(
                "Cannot extract modulation data: No states provided",
            )

        # Map from modulation status to numeric code
        status_map = {
            "up_modulated": 1,
            "down_modulated": -1,
            "non_modulated": 0,
        }

        # Initialize lists to collect data
        data_dict = []
        raw_modulation = []

        # Require state_comparison_type to be provided - no fallback detection
        if state_comparison_type is None:
            raise IdeasError(
                (
                    "state_comparison_type must be provided to "
                    "extract_combined_modulation_data. Please ensure the calling "
                    "function provides the detected comparison type."
                ),
            )

        # Use provided comparison type and baseline state - validated by caller
        validated_states = states

        if not validated_states:
            logger.warning("No valid states found for modulation extraction")
            raise IdeasError(
                "No valid states found for modulation extraction. Check that the requested "
                "states exist in the data files.",
            )

        # Process each comparison type (trace_activity, event_rate, etc.)
        for comp in df.Comparison.unique():
            # Get subset of data for this comparison type
            c_df = df[df["Comparison"] == comp]

            # Process each file separately
            for filename, group_df in c_df.groupby("file"):
                # Process each validated state
                for state in validated_states:
                    mod_col = f"modulation in {state}"

                    # Check if the modulation column exists
                    if mod_col not in group_df.columns:
                        logger.info(
                            f"No modulation column '{mod_col}' found in {filename}"
                        )
                        continue

                    # For pairwise comparisons, extract state1 and state2
                    if state_comparison_type == "pairwise" and " vs " in state:
                        state1, state2 = state.split(" vs ")
                    else:
                        state1, state2 = None, None

                    # Count neurons in each modulation category
                    for status, code in status_map.items():
                        # Get the count matching this code
                        count = (group_df[mod_col] == code).sum()

                        # Get subject ID - already validated by validate_subject_id_format()
                        subject_id = group_df["normalized_subject_id"].iloc[0]

                        # Create a data record with clean subject ID
                        # This ensures subject IDs are group-prefixed for unpaired analysis
                        # and clean for paired analysis, NOT filename-based
                        record = {
                            "name": subject_id,
                            "file": filename,
                            "Comparison": comp,
                            "state": state,
                            "status": status,
                            "Measure": status,
                            "num_cells": count,
                            "normalized_subject_id": subject_id,
                            # Note: state_comparison_type removed to keep CSV clean
                            "baseline_state": baseline_state,
                        }

                        # Add pairwise-specific fields
                        if (
                            state_comparison_type == "pairwise"
                            and state1
                            and state2
                        ):
                            record["state1"] = state1
                            record["state2"] = state2

                        data_dict.append(record)

                    # Get the raw modulation scores
                    score_col = f"modulation scores in {state}"
                    if score_col in group_df.columns:
                        raw_mod_record = {
                            "state": state,
                            "modulation": [group_df[score_col].values],
                            # Note: state_comparison_type removed to keep CSV clean
                            "baseline_state": baseline_state,
                        }

                        # Add pairwise-specific fields
                        if (
                            state_comparison_type == "pairwise"
                            and state1
                            and state2
                        ):
                            raw_mod_record["state1"] = state1
                            raw_mod_record["state2"] = state2

                        raw_modulation.append(raw_mod_record)
                    else:
                        logger.info(
                            f"No modulation scores column '{score_col}' found in {filename}"
                        )

        if not data_dict:
            logger.warning("No modulation data extracted")
            return pd.DataFrame(), pd.DataFrame()

        if not raw_modulation:
            logger.warning("No raw modulation scores extracted")

        return pd.DataFrame(data_dict), pd.DataFrame(raw_modulation)
    except Exception as e:
        logger.error(f"Error in extract_combined_modulation_data: {str(e)}")
        raise IdeasError(
            f"Error extracting combined modulation data: {str(e)}",
        )

def combine_compare_population_data_ideas_wrapper(
    group1_population_activity_files: List[IdeasFile],
    state_names: str,
    state_colors: str,
    group1_population_events_files: Optional[List[IdeasFile]] = None,
    group1_name: Optional[str] = None,
    group1_color: Optional[str] = None,
    group2_population_activity_files: Optional[List[IdeasFile]] = None,
    group2_population_events_files: Optional[List[IdeasFile]] = None,
    group2_name: Optional[str] = None,
    group2_color: Optional[str] = None,
    modulation_colors: Optional[str] = "red, blue",
    data_pairing: str = "unpaired",
    subject_matching: str = "number",
    significance_threshold: float = 0.05,
    multiple_correction: str = "bonf",
    effect_size: str = "cohen",
    state_comparison_method: str = "auto",
) -> None:
    """Combine and compare population activity data from two groups.

    This function processes population activity data (and optionally event data) from one or two
    groups, performs statistical analyses to identify differences between states within each group,
    and between the two groups. It generates visualizations of the results and saves all output
    files to the specified directory.

    Parameters
    ----------
    group1_population_activity_files : List[str]
        List of file paths for group 1 population activity data.
    state_names : str
        Comma-separated string of state names to include in the analysis.
    state_colors : str
        Comma-separated string of colors to use for visualizing different states.
    group1_population_events_files : List[str], optional
        List of file paths for group 1 population events data.
    group1_name : str, optional
        Name of group 1, used for labeling outputs.
    group1_color : str, optional
        Color associated with group 1 for visualizations.
    group2_population_activity_files : List[str], optional
        List of file paths for group 2 population activity data.
    group2_population_events_files : List[str], optional
        List of file paths for group 2 population events data.
    group2_name : str, optional
        Name of group 2, used for labeling outputs.
    group2_color : str, optional
        Color associated with group 2 for visualizations.
    modulation_colors : str, optional
        Comma-separated string of colors for up and down modulation. Default is "red, blue".
    data_pairing : str, optional
        Type of data pairing for comparison ("paired" or "unpaired").
    subject_matching : str, optional
        Method for matching subjects between groups for paired analysis.
    significance_threshold : float, optional
        Significance threshold for statistical tests. Default is 0.05.
    multiple_correction : str, optional
        Method for multiple comparison correction. Default is "bonf".
    effect_size : str, optional
        Method for calculating effect size. Default is "cohen".
    state_comparison_method : str, optional
        Method for determining state comparison type. Default is "auto".

    Returns
    -------
    None
        All output files are saved to the specified output directory.

    Raises
    ------
    IdeasError
        If input parameters are invalid or if there are issues processing the data.

    """
    combine_compare_population_data(
        group1_population_activity_files=group1_population_activity_files,
        state_names=state_names,
        state_colors=state_colors,
        group1_population_events_files=group1_population_events_files,
        group1_name=group1_name,
        group1_color=group1_color,
        group2_population_activity_files=group2_population_activity_files,
        group2_population_events_files=group2_population_events_files,
        group2_name=group2_name,
        group2_color=group2_color,
        modulation_colors=modulation_colors,
        data_pairing=data_pairing,
        subject_matching=subject_matching,
        significance_threshold=significance_threshold,
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        state_comparison_method=state_comparison_method
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        with outputs.register(raise_missing_file=False) as output_data:
            anova_type = (
                "rm_anova" if data_pairing == "paired" else "mixed_anova"
            )
            for group_name in [group1_name, group2_name]:
                subdir_base = "group1" if group_name == group1_name else "group2"
                data_types = ["activity", "events"]
                
                output_file = output_data.register_file(
                    f"population_activity_data_{group_name}.csv",
                    subdir=f"{subdir_base}_population_activity_data",
                )
                for md in metadata.get(f"population_activity_data_{group_name}", {}):
                    output_file.register_metadata(**md) 

                for data_type in data_types:
                    output_file.register_preview(
                        f"{data_type}_modulation_distribution_{group_name}.preview.svg",
                        caption=(
                            f"Stacked bar chart displaying the proportion of neurons with "
                            f"increased, decreased, and unchanged {data_type} for each "
                            f"experimental state in the {group_name} group."
                        ),
                    ).register_preview(
                        f"{group_name}_mean_{data_type}_barplot{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                        caption=(
                            f"Box and whisker plot displaying the distribution of mean neural {data_type} "
                            f"across experimental states for {group1_name}. The plot shows "
                            f"median values, quartiles, and outliers for each state."
                        ),
                    )

                output_names = ["aov_comparisons.csv", "pairwise_comparisons.csv"]
                for output_name in output_names:
                    output_file = output_data.register_file(
                        output_name,
                    )
                    for md in metadata.get(output_name, {}):
                        output_file.register_metadata(**md)
                    
                    for data_type in data_types:
                        output_file.register_preview(
                            f"{data_type}_modulation_distribution{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                            caption=(
                                f"Stacked bar chart displaying the proportion of neurons with "
                                f"increased, decreased, and unchanged {data_type} for each "
                                f"experimental state across {group1_name}, {group2_name}  groups."
                            ),
                        ).register_preview(
                            f"{data_type}_state_lmm{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                            caption=(
                                f"Statistical comparison of {data_type} data across experimental "
                                f"states using Linear Mixed Models (LMM). The plot displays mean "
                                f"values with error bars and individual data points."
                            ),
                        ).register_preview(
                            f"{data_type}_group_{anova_type}{config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION}",
                            caption=(
                                f"Statistical comparison of {data_type} data between "
                                f"experimental groups across different states using Analysis "
                                f"of Variance (ANOVA). The plot shows group differences with "
                                f"mean values, error bars, and individual data points."
                            ),
                        )  

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")