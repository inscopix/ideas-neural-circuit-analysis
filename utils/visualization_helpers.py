"""Visualization helpers for population data analysis.

This module contains functions for plotting and visualizing population data analysis results
in the IDEAS toolbox, including modulation plots, state comparisons, and statistical visualizations.
"""

import logging
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np
import pandas as pd
from ideas.exceptions import IdeasError
import utils.config as config
from utils.statistical_validation import (
    _suppress_pingouin_warnings,
)
from utils.statistical_formatting import (
    _get_mixed_pairwise_sig,
)
from ideas import plots

logger = logging.getLogger(__name__)


def plot_modulation_distribution(
    modulation_scores: pd.DataFrame,
    modulation_colors: List[str],
    states: List[str],
    output_filename: str,
    group_name: Optional[str] = None,
    data_type: str = "activity",
    dimension_label: Optional[str] = None,
) -> None:
    """Plot distribution of modulation scores.

    Parameters
    ----------
    modulation_scores : pd.DataFrame
        DataFrame containing modulation score data
    modulation_colors : List[str]
        List of colors for modulation visualization
    states : List[str]
        List of state names
    output_filename : str
        Output filename for the plot
    group_name : Optional[str]
        Name of the group for the legend. If None, no legend is shown.
    data_type : str, optional
        Type of data being plotted ("activity" or "events"). Defaults to "activity".
    dimension_label : Optional[str], optional
        Human-readable label for the comparison dimension (e.g., "State" or "Epoch").
        Defaults to "State".

    Returns
    -------
    None
        Plot file is saved to the specified filename

    """
    # Normalize dimension label for plot titles
    label_prefix = dimension_label.strip() if dimension_label else "State"

    # Check if this is a pairwise comparison
    is_pairwise = False
    if "state_comparison_type" in modulation_scores.columns:
        is_pairwise = any(
            modulation_scores["state_comparison_type"] == "pairwise"
        )
    else:
        # If no state_comparison_type column, detect from state names
        if "state" in modulation_scores.columns:
            state_samples = modulation_scores["state"].dropna().unique()
            is_pairwise = any(
                [" vs " in str(state) for state in state_samples]
            )

    # Determine which states to plot based on pairwise status
    available_states = []
    if is_pairwise:
        # For pairwise, we look for state pairs
        if "state" in modulation_scores.columns:
            available_states = modulation_scores["state"].unique().tolist()

            # Filter the states based on the requested list
            if states:
                filtered_states = []
                for pair in available_states:
                    # Check if the full pair is in the requested states (exact match)
                    if pair in states:
                        filtered_states.append(pair)
                    # Also check if any individual state from the pair is in the requested states
                    elif " vs " in str(pair):
                        state1, state2 = str(pair).split(" vs ")
                        if (
                            state1.strip() in states
                            or state2.strip() in states
                        ):
                            filtered_states.append(pair)

                # If no filtering worked, use all available states as fallback
                if not filtered_states:
                    logger.warning(
                        f"No pairwise states found for requested states {states}"
                    )
                    logger.info(
                        f"Available pairwise states: {available_states}"
                    )
                    filtered_states = available_states

                available_states = filtered_states
    else:
        # For non-pairwise, use the original method
        if "state" in modulation_scores.columns:
            available_states = modulation_scores["state"].unique().tolist()

    # Filter the states list to only include those present in the data
    if is_pairwise:
        # For pairwise, filter based on actual pairs available
        filtered_states = available_states
    else:
        # For non-pairwise, filter based on requested states
        filtered_states = [
            state for state in available_states if state in states
        ]

    # Log information about which states are being plotted
    if len(filtered_states) < len(states) and not is_pairwise:
        missing_states = set(states) - set(filtered_states)
        logger.warning(
            f"States not found in modulation data: {missing_states}"
        )

    # If no states are available, create an empty plot with a message
    if not filtered_states:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(
            0.5,
            0.5,
            f"No modulation data available for the requested {label_prefix.lower()}s: {states}",
            horizontalalignment="center",
            verticalalignment="center",
            wrap=True,
        )
        ax.axis("off")
        fig.savefig(
            output_filename,
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        logger.warning(
            f"No modulation data available for any of the requested states: {states}"
        )
        return

    # Limit the number of panels to 24 for pairwise comparisons
    MAX_PANELS = 24
    original_count = len(filtered_states)

    if is_pairwise and len(filtered_states) > MAX_PANELS:
        logger.warning(
            f"Too many pairwise comparisons ({original_count}) for modulation distribution plot. "
            f"Limiting to first {MAX_PANELS} panels to maintain readability. "
            f"Consider reducing the number of states or using a subset of comparisons."
        )
        filtered_states = filtered_states[:MAX_PANELS]

    # Log the final number of panels being created
    if is_pairwise:
        logger.info(
            f"Creating modulation distribution plot with {len(filtered_states)}"
            " pairwise comparison panels"
        )

    # Create a figure with the correct number of subplots based on available states
    fig, ax = plt.subplots(
        nrows=len(filtered_states),
        ncols=1,
        figsize=(5, 5 * len(filtered_states)),
    )

    # Set x-axis label based on data type
    xlabel = (
        "Activity Modulation"
        if data_type == "activity"
        else "Event Rate Modulation"
    )

    # Determine the comparison type for title formatting
    is_baseline_comparison = False
    baseline_state = None

    if "state_comparison_type" in modulation_scores.columns:
        if any(
            modulation_scores["state_comparison_type"] == "state_vs_baseline"
        ):
            is_baseline_comparison = True
            if "baseline_state" in modulation_scores.columns:
                baseline_values = (
                    modulation_scores["baseline_state"].dropna().unique()
                )
                if len(baseline_values) > 0:
                    baseline_state = baseline_values[0]

    # Make ax iterable if only one state
    if len(filtered_states) == 1:
        ax = [ax]

    for i, state in enumerate(filtered_states):
        # Get data for this state
        state_data = modulation_scores[modulation_scores["state"] == state]

        # Safely concatenate modulation values
        all_vals = []
        for mod_array in state_data["modulation"].values:
            if isinstance(mod_array, (list, np.ndarray)):
                if len(mod_array) == 1 and isinstance(
                    mod_array[0], np.ndarray
                ):
                    all_vals.extend(mod_array[0])
                else:
                    all_vals.extend(mod_array)
            else:
                logger.warning(
                    f"Unexpected data type in modulation scores: {type(mod_array)}"
                )
                continue

        vals = np.array(all_vals)

        if len(vals) == 0:
            logger.warning(f"No valid modulation values for {label_prefix.lower()} '{state}'")
            ax[i].text(
                0.5,
                0.5,
                f"No valid modulation values for {state}",
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax[i].set_title(f"{label_prefix}: {state}")
            continue

        if modulation_colors and len(modulation_colors) >= 2:
            palette = [
                modulation_colors[0],
                "whitesmoke",
                modulation_colors[1],
            ]
        else:
            palette = "coolwarm"

        # Format title based on comparison type
        if is_pairwise:
            # For pairwise comparison, state already contains "vs"
            title_state = state
        elif is_baseline_comparison:
            # For baseline comparison
            vs_baseline_text = " vs. " + (
                baseline_state if baseline_state else "Baseline"
            )
            title_state = f"{state}{vs_baseline_text}"
        else:
            # For regular state
            title_state = state

        plots.plot_shaded_hist(
            vals,
            ax[i],
            state,
            xlabel=xlabel,
            hist_lims=[-1, 1],
            palette=palette,
        )

        if group_name:
            ax[i].set_title(f"{label_prefix}: {title_state} - {group_name}")
        else:
            ax[i].set_title(f"{label_prefix}: {title_state}")

    # Add a note to the figure if panels were limited
    if is_pairwise and original_count > MAX_PANELS:
        fig.suptitle(
            "Modulation Distribution ("
            f"showing {len(filtered_states)} of {original_count} comparisons)",
            fontsize=12,
            y=0.98,
        )

    fig.tight_layout()
    fig.savefig(
        output_filename,
        dpi=300,
        format="svg",
        bbox_inches="tight",
        transparent=True,
    )


def plot_combined_modulation_data(
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
    data_type: str = "activity",
    state_comparison_type: Optional[str] = None,
    baseline_state: Optional[str] = None,
    validated_states: Optional[List[str]] = None,
) -> None:
    """Plot combined population modulation data.

    This function creates visualizations of modulation data across states.
    It generates a plot file showing modulation patterns for up and down modulated cells.

    Parameters
    ----------
    mod_data : pd.DataFrame
        DataFrame containing modulation data
    states : List[str]
        List of state names
    aov : pd.DataFrame
        ANOVA results DataFrame
    pairwise : pd.DataFrame
        Pairwise comparison results DataFrame
    colors : List[str]
        List of colors for states
    mod_filename : str
        Output filename for modulation plot
    data_pairing : str, optional
        Type of data pairing ("paired" or "unpaired"), by default "unpaired"
    group_names : List[str], optional
        List of group names in order, by default None
    group_colors : List[str], optional
        List of colors for groups in order, by default None
    data_type : str, optional
        Type of data being plotted ("activity" or "events"), by default "activity"
    state_comparison_type : str, optional
        Type of comparison being performed, by default None
    baseline_state : str, optional
        Name of baseline state if applicable, by default None
    validated_states : List[str], optional
        Pre-validated list of states to plot, by default None

    """
    # Constants for positioning
    X_OFFSET_FACTOR = 0.4
    MEAN_LINE_WIDTH = 0.1

    try:
        # Validate input data
        if mod_data is None or mod_data.empty:
            PlottingHelpers.create_empty_plot(
                mod_filename, data_type, "No modulation data available"
            )
            return

        # Determine final states to use
        final_states = PlottingHelpers.determine_final_states(
            validated_states, mod_data, states
        )

        # Exit if no states are available
        if not final_states:
            PlottingHelpers.create_empty_plot(
                mod_filename,
                data_type,
                f"No modulation data available for states: {states}",
            )
            return

        # Validate required columns
        PlottingHelpers.validate_required_columns(mod_data)

        # Ignore unmodulated cells for the plot
        clean_data = mod_data[mod_data["status"] != "non_modulated"]

        # Check if we need to split by group (if multiple groups exist)
        has_multiple_groups = (
            "group" in mod_data.columns and mod_data["group"].nunique() > 1
        )

        # Create a figure for modulation data - only create panels for existing modulation types
        modulation_types = clean_data["status"].unique()

        # Handle case where no modulation types exist
        if len(modulation_types) == 0:
            PlottingHelpers.create_empty_plot(
                mod_filename,
                data_type,
                "No modulated cells found for plotting",
            )
            return

        fig, ax = plt.subplots(
            ncols=len(modulation_types), figsize=(5 * len(modulation_types), 5)
        )
        if len(modulation_types) == 1:
            ax = [ax]

        for idx, status in enumerate(modulation_types):
            temp_df = clean_data[clean_data["status"] == status]

            # Skip if we have no data for this status
            if temp_df.empty:
                PlottingHelpers.handle_empty_status_data(ax[idx], status)
                continue

            # Plot data based on group configuration
            if has_multiple_groups:
                _plot_multi_group_data(
                    ax[idx],
                    temp_df,
                    final_states,
                    group_names,
                    group_colors,
                    data_pairing,
                    X_OFFSET_FACTOR,
                    MEAN_LINE_WIDTH,
                )
            else:
                _plot_single_group_data(
                    ax[idx],
                    temp_df,
                    final_states,
                    colors,
                    group_colors,
                    group_names,
                    MEAN_LINE_WIDTH,
                )

            # Configure axis appearance and labels
            _configure_axis(
                ax[idx],
                final_states,
                state_comparison_type,
                status,
                baseline_state,
                pairwise,
                data_type,
                group_names,
                has_multiple_groups,
                idx,
                data_pairing,
                len(modulation_types),
            )

        # Save modulation plot
        fig.tight_layout()
        fig.savefig(
            mod_filename,
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )

    except Exception as e:
        logger.error(
            f"Error during plot_combined_modulation_data execution: {type(e).__name__}: {str(e)}"
        )
        raise IdeasError(
            f"Error plotting combined population data: {str(e)}",
        )


class PlottingHelpers:
    """Unified helper class for modulation plotting functions."""

    @staticmethod
    def create_empty_plot(filename: str, data_type: str, message: str) -> None:
        """Create an empty plot with a message when no data is available."""
        logger.warning(
            f"Cannot plot combined modulation data for {data_type}: {message}"
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_title(f"{data_type.capitalize()} Modulation - No Data")
        ax.axis("off")
        fig.savefig(
            filename,
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )

    @staticmethod
    def determine_final_states(
        validated_states: Optional[List[str]],
        mod_data: pd.DataFrame,
        states: List[str],
    ) -> List[str]:
        """Determine the final list of states to use for plotting."""
        if validated_states is not None:
            return validated_states

        if "state" in mod_data.columns:
            return mod_data["state"].unique().tolist()
        else:
            return states

    @staticmethod
    def validate_required_columns(mod_data: pd.DataFrame) -> None:
        """Validate that required columns are present in the modulation data."""
        required_mod_cols = ["status", "state", "num_cells"]
        missing_mod_cols = [
            col for col in required_mod_cols if col not in mod_data.columns
        ]
        if missing_mod_cols:
            raise IdeasError(
                f"Missing required columns in modulation data: {missing_mod_cols}",
            )

    @staticmethod
    def handle_empty_status_data(ax: Axes, status: str) -> None:
        """Handle the case where no data exists for a particular modulation status."""
        ax.text(
            0.5,
            0.5,
            f"No {status.replace('_', ' ')} cells",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_title(status.replace("_", " ").title())

    @staticmethod
    def format_xticklabels(
        final_states: List[str], state_comparison_type: Optional[str]
    ) -> List[str]:
        """Format x-tick labels based on comparison type."""
        if state_comparison_type == "pairwise":
            # For pairwise, use shorter labels for clarity
            xticklabels = []
            for state_pair in final_states:
                if " vs " in state_pair:
                    parts = state_pair.split(" vs ")
                    xticklabels.append(f"{parts[0]} vs\n{parts[1]}")
                else:
                    xticklabels.append(state_pair)
            return xticklabels
        else:
            return final_states

    @staticmethod
    def create_title(
        status: str,
        state_comparison_type: Optional[str],
        baseline_state: Optional[str],
        pairwise: pd.DataFrame,
        final_states: List[str],
        has_multiple_groups: bool,
        data_pairing: str = "unpaired",
        data_type: str = "activity",
    ) -> str:
        """Create the title for the modulation plot based on comparison type and status."""
        # Format status name and include data type for clarity
        status_title = status.replace("_", " ").title()
        data_type_title = data_type.capitalize()

        # Create clean title with clear comparison type indication and data type
        if state_comparison_type == "pairwise":
            title = f"{status_title} Cells ({data_type_title})\n(Pairwise Contrasts)"
        elif state_comparison_type == "state_vs_baseline":
            if baseline_state:
                title = f"{status_title} Cells ({data_type_title})\n(vs. {baseline_state})"
            else:
                title = (
                    f"{status_title} Cells ({data_type_title})\n(vs. Baseline)"
                )
        else:
            title = f"{status_title} Cells ({data_type_title})"

        # Add group information if relevant
        if has_multiple_groups:
            if data_pairing == "paired":
                title += "\n(Paired Group Comparison)"
            else:
                title += "\n(Group Comparison)"

        # Add significance information using _get_mixed_pairwise_sig
        if pairwise is not None and not pairwise.empty:
            try:
                sig_info = _get_mixed_pairwise_sig(
                    pairwise,
                    status=status,
                    state_legend_show=(not has_multiple_groups),
                )

                # Hide all "no significant difference" messages from plot titles
                no_sig_messages = [
                    "No significant differences between states",
                    "No significant differences",
                    "No pairwise data available",
                    "No p-value column found",
                    "Error retrieving significance info",
                ]

                if (
                    sig_info
                    and sig_info.strip()
                    and not any(
                        no_sig_msg in sig_info
                        for no_sig_msg in no_sig_messages
                    )
                ):
                    title += f"\n{sig_info}"
            except Exception as e:
                logger.debug(f"Could not extract significance info: {str(e)}")

        return title


@_suppress_pingouin_warnings
def plot_state_lmm_comparison(
    *,
    df: pd.DataFrame,
    data_type: str,
    states: List[str],
    state_colors: Dict[str, str],
    has_single_group: bool,
    group_names: Optional[List[str]] = None,
    group_colors: Optional[List[str]] = None,
    state_lmm: pd.DataFrame = None,
    state_pairwise: pd.DataFrame = None,
    output_dir: str = None,
    output_filename: Optional[str] = None,
):
    """Create visualization for state comparison using LMM results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing activity data
    data_type : str
        Type of data being visualized (e.g., 'activity' or 'events')
    states : List[str]
        List of state names
    state_colors : Dict[str, str]
        Mapping from state names to their colors
    has_single_group : bool
        Whether analysis is for single group or multiple groups
    group_names : List[str], optional
        List of group names in order
    group_colors : List[str], optional
        List of colors for groups in order
    state_lmm : pd.DataFrame, optional
        State LMM results for title
    state_pairwise : pd.DataFrame, optional
        State pairwise comparison results for title
    output_dir : str, optional
        Directory to save output files
    output_filename : str, optional
        Output filename for the plot

    Returns
    -------
    Optional[IdeasPreviewFile]
        Preview file object if output_dir is provided and empty data, otherwise None

    """
    try:
        # Add check for empty dataframe
        if df is None or df.empty:
            logger.warning(
                f"Cannot plot state LMM comparison for {data_type}: "
                f"Empty dataframe provided."
            )
            # Create an empty plot
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title(
                f"{data_type.capitalize()} State Comparison - No Data"
            )
            # Save placeholder if output dir provided
            if output_dir:
                placeholder_filename = os.path.join(
                    output_dir,
                    f"{data_type}_state_lmm_nodata"
                    + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
                )
                plt.savefig(placeholder_filename, bbox_inches="tight")
            return None

        fig, ax = plt.subplots(figsize=(7, 7))

        # Use violin plot for state comparison
        if not has_single_group and group_names and group_colors:
            # Get unique groups in the order specified by group_names
            unique_groups = [
                g for g in group_names if g in df["group"].unique()
            ]

            # Create a mapping of group names to their colors using the provided lists
            group_color_map = {
                group: color
                for group, color in zip(group_names, group_colors)
                if group in unique_groups
            }

            # Apply same state ordering logic as single group
            actual_states_in_data = (
                set(df["state"].unique())
                if "state" in df.columns
                else set(states)
            )

            # Start with user-provided states that are in the data (preserving user order)
            final_states = [
                state for state in states if state in actual_states_in_data
            ]

            # Add any additional states from data that aren't in user list
            # (like auto-added baseline)
            additional_states = [
                state for state in actual_states_in_data if state not in states
            ]
            final_states.extend(additional_states)

            # Create violin plot with group colors and proper state ordering
            sns.violinplot(
                data=df,
                x="state",
                y="activity",
                hue="group",
                ax=ax,
                palette=group_color_map,
                order=final_states,
                fill=True,
                linewidth=1,
                inner="point",
                density_norm="width",
                legend=True,
            )

            # Add legend for groups
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    title="Group",
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    frameon=False,
                )
        else:
            # Single group case - use violin plot with state colors
            # Get all states actually present in the data
            actual_states_in_data = (
                set(df["state"].unique())
                if "state" in df.columns
                else set(states)
            )

            # Start with user-provided states that are in the data (preserving user order)
            final_states = [
                state for state in states if state in actual_states_in_data
            ]

            # Add any additional states from data that aren't in user list
            # (like auto-added baseline)
            additional_states = [
                state for state in actual_states_in_data if state not in states
            ]
            final_states.extend(additional_states)

            # Use global state color mapping - simple and clean
            final_colors = [
                state_colors.get(state, "#1f77b4") for state in final_states
            ]

            sns.violinplot(
                data=df,
                x="state",
                y="activity",
                hue="state",
                palette=final_colors,
                order=final_states,
                fill=True,
                linewidth=1,
                inner="point",
                density_norm="width",
                legend=False,
            )
        y_label = f"Single-Cell {data_type}"
        ax.set_ylabel(y_label)
        ax.set_xlabel("State")

        # Create title with significance information
        title = _get_clean_activity_title(data_type, has_single_group)
        if state_pairwise is not None and not state_pairwise.empty:
            try:
                sig_info = _get_mixed_pairwise_sig(
                    state_pairwise, state_legend_show=True
                )

                # Hide all "no significant difference" messages from plot titles
                no_sig_messages = [
                    "No significant differences between states",
                    "No significant differences",
                    "No pairwise data available",
                    "No p-value column found",
                    "Error retrieving significance info",
                ]

                if (
                    sig_info
                    and sig_info.strip()
                    and not any(
                        no_sig_msg in sig_info
                        for no_sig_msg in no_sig_messages
                    )
                ):
                    title += f"\n{sig_info}"
            except Exception as e:
                logger.debug(f"Could not extract significance info: {str(e)}")

        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Save state comparison LMM plot if output directory or filename provided
        if output_dir or output_filename:
            # Determine the final filename
            if output_filename:
                save_path = output_filename
            else:
                # Construct filename using output_dir if filename is not provided
                save_path = os.path.join(
                    output_dir,
                    f"{data_type}_state_lmm"
                    + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
                )

            # Save the plot using the determined save path
            plt.savefig(save_path, bbox_inches="tight")

        return None

    except Exception as e:
        logger.error(f"Error in plot_state_lmm_comparison: {str(e)}")
        return None


@_suppress_pingouin_warnings
def plot_group_anova_comparison(
    *,
    df: pd.DataFrame,
    data_type: str,
    states: List[str],
    group_names: List[str],
    group_colors: List[str],
    data_pairing: str = "unpaired",
    group_pairwise: pd.DataFrame = None,
    output_dir: str = None,
    output_filename: Optional[str] = None,
):
    """Create visualization for group comparison using ANOVA results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing activity data
    data_type : str
        Type of data being visualized (e.g., 'activity' or 'events')
    states : List[str]
        List of state names
    group_names : List[str]
        List of group names in order
    group_colors : List[str]
        List of colors for groups in order
    data_pairing : str, optional
        Data pairing method, defaults to "unpaired"
    group_pairwise : pd.DataFrame, optional
        Group ANOVA results for title
    output_dir : str, optional
        Directory to save output files
    output_filename : str, optional
        Output filename for the plot

    Returns
    -------
    Optional[IdeasPreviewFile]
        Preview file object if output_dir is provided and empty data, otherwise None

    """
    try:
        # Add check for empty dataframe
        if df is None or df.empty:
            logger.warning(
                f"Cannot plot group ANOVA comparison for {data_type}: Empty dataframe provided."
            )
            # Create an empty plot
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title(
                f"{data_type.capitalize()} Group Comparison - No Data"
            )
            # Save placeholder if output dir provided
            if output_dir:
                placeholder_filename = os.path.join(
                    output_dir,
                    f"{data_type}_group_"
                    f"{'rm_anova' if data_pairing == 'paired' else 'mixed_anova'}_nodata"
                    + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
                )
                plt.savefig(placeholder_filename, bbox_inches="tight")
            return None

        fig, ax = plt.subplots(figsize=(7, 7))

        # Get unique groups in the order specified by group_names
        unique_groups = [g for g in group_names if g in df["group"].unique()]

        # Create a mapping of group names to their colors using the provided lists
        group_color_map = {
            group: color
            for group, color in zip(group_names, group_colors)
            if group in unique_groups
        }

        # Create a scatter plot with group as the hue variable
        for i, state in enumerate(states):
            state_data = df[df["state"] == state]

            # Plot each group's data points in the order specified by group_names
            for j, group_name in enumerate(unique_groups):
                group_data = state_data[state_data["group"] == group_name]
                if group_data.empty:
                    continue

                x_pos = i + (j - 0.5) * 0.4
                color = group_color_map[group_name]

                # Plot points
                ax.scatter(
                    [x_pos] * len(group_data),
                    group_data["activity"],
                    color=color,
                    s=80,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                    label=group_name if i == 0 else None,
                )

                # Add mean line
                mean_y = group_data["activity"].mean()
                ax.plot(
                    [x_pos - 0.1, x_pos + 0.1],
                    [mean_y, mean_y],
                    color=color,
                    linewidth=3,
                    zorder=4,
                )

                # Add connecting lines between groups within the same state for paired data
                if (
                    data_pairing == "paired"
                    and "normalized_subject_id" in state_data.columns
                ):
                    # Group data by subject ID
                    for subject_id in state_data[
                        "normalized_subject_id"
                    ].unique():
                        subject_data = state_data[
                            state_data["normalized_subject_id"] == subject_id
                        ]

                        if len(subject_data) == 2:
                            # Get x positions and y values for both groups
                            # in the order specified by group_names
                            x_positions = []
                            y_values = []

                            # Iterate through groups in the order specified by group_names
                            for k, group_name in enumerate(unique_groups):
                                group_data = subject_data[
                                    subject_data["group"] == group_name
                                ]
                                if not group_data.empty:
                                    x_pos = i + (k - 0.5) * 0.4
                                    x_positions.append(x_pos)
                                    y_values.append(
                                        group_data["activity"].iloc[0]
                                    )

                            # Only connect points if we have exactly two points to connect
                            if len(x_positions) == 2 and len(y_values) == 2:
                                # Connect points with a line
                                ax.plot(
                                    x_positions,
                                    y_values,
                                    color="gray",
                                    alpha=0.3,
                                    linewidth=1,
                                    zorder=1,
                                )

        # Set x-ticks to state names
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states)

        # Add legend for groups
        if len(unique_groups) > 1:
            # Create custom legend handles with correct colors in the same order as plotting
            legend_handles = []
            legend_labels = []
            # Use the same order as in the plotting loop
            for j, group_name in enumerate(unique_groups):
                color = group_color_map[group_name]
                legend_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markersize=10,
                    )
                )
                legend_labels.append(group_name)
            ax.legend(
                legend_handles,
                legend_labels,
                title="Group",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
            )
        y_label = f"Subject-averaged {data_type}"
        ax.set_ylabel(y_label)
        ax.set_xlabel("State")

        # Create title with significance information
        title = _get_clean_activity_title(data_type, False, data_pairing)
        if group_pairwise is not None and not group_pairwise.empty:
            try:
                sig_info = _get_mixed_pairwise_sig(
                    group_pairwise,
                    state_legend_show=False,
                )

                # Hide all "no significant difference" messages from plot titles
                no_sig_messages = [
                    "No significant differences between states",
                    "No significant differences",
                    "No pairwise data available",
                    "No p-value column found",
                    "Error retrieving significance info",
                ]

                if (
                    sig_info
                    and sig_info.strip()
                    and not any(
                        no_sig_msg in sig_info
                        for no_sig_msg in no_sig_messages
                    )
                ):
                    title += f"\n{sig_info}"
            except Exception as e:
                logger.debug(f"Could not extract significance info: {str(e)}")

        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Save group comparison plot if output directory or filename provided
        if output_dir or output_filename:
            # Determine the final filename
            if output_filename:
                save_path = output_filename
            else:
                # Construct filename using output_dir
                # if filename is not provided
                filename_base = (
                    f"{data_type}_group_"
                    f"{'rm_anova' if data_pairing == 'paired' else 'mixed_anova'}"
                )
                save_path = os.path.join(
                    output_dir,
                    filename_base + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION,
                )

            # Save the plot using the determined save path
            plt.savefig(save_path, bbox_inches="tight")

        return None

    except Exception as e:
        logger.error(f"Error in plot_group_anova_comparison: {str(e)}")
        return None


@_suppress_pingouin_warnings
def _plot_multi_group_data(
    ax: Axes,
    temp_df: pd.DataFrame,
    final_states: List[str],
    group_names: Optional[List[str]],
    group_colors: Optional[List[str]],
    data_pairing: str,
    x_offset_factor: float,
    mean_line_width: float,
) -> None:
    """Plot data for multiple groups with appropriate positioning and connecting lines."""
    # Get unique groups in the order they appear in the data
    unique_groups = list(temp_df["group"].unique())

    # Create a mapping of group names to their colors using the provided lists
    if group_names and group_colors:
        group_color_map = {
            group: color
            for group, color in zip(group_names, group_colors)
            if group in unique_groups
        }
    else:
        # Fallback to default colors if not provided
        default_colors = ["blue", "red", "green", "orange", "purple", "brown"]
        group_color_map = {
            group: default_colors[i % len(default_colors)]
            for i, group in enumerate(unique_groups)
        }

    # Plot each state
    for i, state in enumerate(final_states):
        state_data = temp_df[temp_df["state"] == state]

        if state_data.empty:
            continue

        # Plot each group's data points in the order specified by group_names
        for j, group_name in enumerate(unique_groups):
            group_data = state_data[state_data["group"] == group_name]
            if group_data.empty:
                continue

            # Calculate position offset for this group
            x_pos = i + (j - len(unique_groups) / 2 + 0.5) * x_offset_factor
            color = group_color_map.get(group_name, "gray")

            # Plot individual data points
            ax.scatter(
                [x_pos] * len(group_data),
                group_data["num_cells"],
                color=color,
                s=60,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label=group_name if i == 0 else None,
            )

            # Add mean line
            mean_y = group_data["num_cells"].mean()
            ax.plot(
                [x_pos - mean_line_width, x_pos + mean_line_width],
                [mean_y, mean_y],
                color=color,
                linewidth=3,
                zorder=4,
            )

            # Add connecting lines for paired data
            if (
                data_pairing == "paired"
                and "normalized_subject_id" in state_data.columns
            ):
                # Connect corresponding subjects across groups within the same state
                for subject_id in state_data["normalized_subject_id"].unique():
                    subject_data = state_data[
                        state_data["normalized_subject_id"] == subject_id
                    ]

                    if len(subject_data) >= 2:
                        subject_groups = subject_data.groupby("group")
                        group_positions = []
                        group_values = []

                        for group_name in unique_groups:
                            if group_name in subject_groups.groups:
                                group_subset = subject_groups.get_group(
                                    group_name
                                )
                                if not group_subset.empty:
                                    group_idx = unique_groups.index(group_name)
                                    x_pos = (
                                        i
                                        + (
                                            group_idx
                                            - len(unique_groups) / 2
                                            + 0.5
                                        )
                                        * x_offset_factor
                                    )
                                    group_positions.append(x_pos)
                                    group_values.append(
                                        group_subset["num_cells"].iloc[0]
                                    )

                        # Connect points with a line if we have exactly 2 points
                        if (
                            len(group_positions) == 2
                            and len(group_values) == 2
                        ):
                            ax.plot(
                                group_positions,
                                group_values,
                                color="gray",
                                alpha=0.3,
                                linewidth=1,
                                zorder=1,
                            )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(final_states)))
    ax.set_xticklabels(final_states)

    # Add legend for groups if multiple groups exist
    if len(unique_groups) > 1:
        ax.legend(
            title="Group",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )


def _plot_single_group_data(
    ax: Axes,
    temp_df: pd.DataFrame,
    final_states: List[str],
    state_colors: List[str],
    group_colors: Optional[List[str]],
    group_names: Optional[List[str]],
    mean_line_width: float,
) -> None:
    """Plot data for a single group with state-based coloring."""
    # Check if this is a pairwise comparison by looking for " vs " in state names
    is_pairwise = any(" vs " in state for state in final_states)

    if is_pairwise:
        # For pairwise comparisons, use consistent colors for all comparisons
        if group_colors and len(group_colors) >= 1:
            consistent_color = group_colors[0]
        elif state_colors and len(state_colors) >= 1:
            consistent_color = state_colors[0]
        else:
            consistent_color = "#1f77b4"

        # For pairwise, all states get the same color
        state_color_map = {state: consistent_color for state in final_states}
    else:
        # For non-pairwise comparisons, use individual state colors
        # Ensure we have enough colors, use defaults if needed
        if len(state_colors) >= len(final_states):
            state_color_map = {
                state: color
                for state, color in zip(
                    final_states, state_colors[: len(final_states)]
                )
            }
        else:
            # Fallback to default colors if not enough state colors provided
            default_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
            ]
            state_color_map = {
                state: (
                    state_colors[i]
                    if i < len(state_colors)
                    else default_colors[i % len(default_colors)]
                )
                for i, state in enumerate(final_states)
            }

    # Plot each state
    for i, state in enumerate(final_states):
        state_data = temp_df[temp_df["state"] == state]

        if state_data.empty:
            continue

        color = state_color_map.get(state, "gray")

        # Plot data points
        ax.scatter(
            [i] * len(state_data),
            state_data["num_cells"],
            color=color,
            s=60,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

        # Add mean line
        mean_y = state_data["num_cells"].mean()
        ax.plot(
            [i - mean_line_width, i + mean_line_width],
            [mean_y, mean_y],
            color=color,
            linewidth=3,
            zorder=4,
        )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(final_states)))
    ax.set_xticklabels(final_states)


def _configure_axis(
    ax: Axes,
    final_states: List[str],
    state_comparison_type: Optional[str],
    status: str,
    baseline_state: Optional[str],
    pairwise: pd.DataFrame,
    data_type: str,
    group_names: Optional[List[str]],
    has_multiple_groups: bool,
    panel_idx: int,
    data_pairing: str,
    total_panels: int,
) -> None:
    """Configure axis appearance, labels, and titles."""
    # Format x-tick labels based on comparison type
    formatted_labels = PlottingHelpers.format_xticklabels(
        final_states, state_comparison_type
    )
    ax.set_xticklabels(formatted_labels, rotation=45, ha="right")

    # Set y-axis label
    y_label = f"Number of {status.replace('_', ' ').title()} Cells"
    ax.set_ylabel(y_label)

    # Set x-axis label for all panels (since they're side-by-side, not stacked)
    if state_comparison_type == "pairwise":
        ax.set_xlabel("State Comparisons")
    elif state_comparison_type == "state_vs_baseline":
        ax.set_xlabel(f"States (vs. {baseline_state or 'Baseline'})")
    else:
        ax.set_xlabel("States")

    # Create and set title
    title = PlottingHelpers.create_title(
        status,
        state_comparison_type,
        baseline_state,
        pairwise,
        final_states,
        has_multiple_groups,
        data_pairing,
        data_type,
    )
    ax.set_title(title)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _get_clean_activity_title(
    data_type: str, has_single_group: bool, data_pairing: str = "unpaired"
) -> str:
    """Generate clean title for activity plots without statistical results.

    Parameters
    ----------
    data_type : str
        Type of data being plotted (e.g., 'activity' or 'events')
    has_single_group : bool
        Whether analysis is for single group
    data_pairing : str, optional
        Type of data pairing ("paired" or "unpaired"), defaults to "unpaired"

    Returns
    -------
    str
        Clean title without statistical results

    """
    if has_single_group:
        return f"Single-Cell {data_type.capitalize()}"
    else:
        if data_pairing == "paired":
            return f"Subject-averaged {data_type.capitalize()}\n(Paired Group Comparison)"
        else:
            return f"Subject-averaged {data_type.capitalize()}\n(Group Comparison)"


def plot_single_state_group_comparison(
    *,
    df: pd.DataFrame,
    state_name: str,
    data_type: str,
    group_names: List[str],
    group_colors: List[str],
    data_pairing: str = "unpaired",
    group_stats: pd.DataFrame = None,
    output_filename: Optional[str] = None,
) -> None:
    """Create visualization for single state, multiple group comparison.

    This function handles the edge case where only one experimental state is present
    but multiple groups need to be compared visually.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing activity data for single state
    state_name : str
        Name of the single state being visualized
    data_type : str
        Type of data being visualized (e.g., 'activity' or 'events')
    group_names : List[str]
        List of group names in order
    group_colors : List[str]
        List of colors for groups in order
    data_pairing : str, optional
        Type of data pairing ("paired" or "unpaired"), by default "unpaired"
    group_stats : pd.DataFrame, optional
        Statistical results for title information
    output_filename : str, optional
        Output filename for the plot

    Returns
    -------
    None
        Plot file is saved to the specified filename

    """
    try:
        # Validate input data
        if df is None or df.empty:
            logger.warning(
                f"Cannot plot single state comparison for {data_type}: Empty dataframe provided."
            )
            if output_filename:
                _create_empty_plot_file(
                    output_filename,
                    f"No {data_type} data available for state '{state_name}'",
                )
            return

        # Check if we have groups to compare
        if "group" not in df.columns or df["group"].nunique() < 2:
            logger.warning(
                f"Single state '{state_name}': No groups to compare for visualization"
            )
            if output_filename:
                _create_empty_plot_file(
                    output_filename,
                    f"No groups to compare for state '{state_name}'",
                )
            return

        fig, ax = plt.subplots(figsize=(6, 5))

        # Get unique groups in the order specified by group_names
        unique_groups = [g for g in group_names if g in df["group"].unique()]

        # Create a mapping of group names to their colors
        group_color_map = {
            group: color
            for group, color in zip(group_names, group_colors)
            if group in unique_groups
        }

        # Create violin plot or box plot for single state comparison
        if len(df) > 10:
            sns.violinplot(
                data=df,
                x="group",
                y="activity",
                order=unique_groups,
                palette=[
                    group_color_map.get(g, "#1f77b4") for g in unique_groups
                ],
                ax=ax,
                fill=True,
                linewidth=1,
                inner="point",
                density_norm="width",
                width=0.7,  # Make plots slightly narrower for more spacing
            )
        else:
            sns.boxplot(
                data=df,
                x="group",
                y="activity",
                order=unique_groups,
                hue="state",
                palette=group_color_map,
                legend=False,
                ax=ax,
                width=0.6,
            )

        # Add individual data points for better visualization
        for i, group_name in enumerate(unique_groups):
            group_data = df[df["group"] == group_name]
            if not group_data.empty:
                color = group_color_map.get(group_name, "#1f77b4")

                # Add scatter points with slight jitter
                jitter = np.random.normal(0, 0.02, len(group_data))
                ax.scatter(
                    i + jitter,
                    group_data["activity"],
                    color=color,
                    s=40,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )

        # Add connecting lines for paired data
        if (
            data_pairing == "paired"
            and "normalized_subject_id" in df.columns
            and len(unique_groups) == 2
        ):
            for subject_id in df["normalized_subject_id"].unique():
                subject_data = df[df["normalized_subject_id"] == subject_id]

                if len(subject_data) == 2:
                    # Get data for both groups for this subject
                    group_positions = []
                    group_values = []

                    for j, group_name in enumerate(unique_groups):
                        group_subset = subject_data[
                            subject_data["group"] == group_name
                        ]
                        if not group_subset.empty:
                            group_positions.append(j)
                            group_values.append(
                                group_subset["activity"].iloc[0]
                            )

                    # Connect points if we have exactly 2 points
                    if len(group_positions) == 2 and len(group_values) == 2:
                        ax.plot(
                            group_positions,
                            group_values,
                            color="gray",
                            alpha=0.3,
                            linewidth=1,
                            zorder=1,
                        )

        # Configure axis appearance
        ax.set_xlabel("Group")
        ax.set_ylabel(f"{data_type.capitalize()}")

        # Create title with statistical information if available
        title = f"Single State Analysis: {state_name}"
        if data_pairing == "paired":
            title += "\n(Paired Group Comparison)"
        else:
            title += "\n(Group Comparison)"

        # Add statistical significance information if available
        if group_stats is not None and not group_stats.empty:
            # Look for significant results
            if "p-unc" in group_stats.columns:
                sig_results = group_stats[group_stats["p-unc"] < 0.05]
                if not sig_results.empty:
                    title += "\n(p < 0.05)"

        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Save the plot if filename provided
        if output_filename:
            fig.tight_layout()
            fig.savefig(
                output_filename,
                dpi=300,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            logger.info(
                f"Single state group comparison plot saved: {output_filename}"
            )

        plt.close(fig)

    except Exception as e:
        logger.error(f"Error in plot_single_state_group_comparison: {str(e)}")
        if output_filename:
            _create_empty_plot_file(
                output_filename,
                f"Error plotting single state comparison: {str(e)}",
            )


def _create_empty_plot_file(filename: str, message: str) -> None:
    """Create an empty plot file with a message."""
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.axis("off")
        fig.savefig(
            filename,
            dpi=300,
            format="svg",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating empty plot file: {str(e)}")


def create_boxplot_preview(
    data_df: pd.DataFrame,
    col_name: str,
    group_name: str,
    identifier: str,
    title_prefix: str,
    filename: str,
    state_color_map: dict,
    filter_state_names: list,
    data_type: str = "activity",
    y_limits: tuple = None,
    include_points: bool = False,
    dimension_label: Optional[str] = None,
) -> bool:
    """Create and save boxplot to the specified filename.

    This is a consolidated function that can create both simple boxplots and
    detailed boxplots with individual data points.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data to plot
    col_name : str
        Column name containing the data to plot
    group_name : str
        Name of the group for labeling
    identifier : str
        Identifier for the plot (used in title)
    title_prefix : str
        Prefix for the plot title
    filename : str
        Full path to save the plot file
    state_color_map : dict
        Dictionary mapping state names to colors
    filter_state_names : list
        List of state names to include in the plot
    data_type : str, optional
        Type of data being plotted (e.g., "activity", "correlation",
        "event_rate"), by default "activity"
    y_limits : tuple, optional
        Tuple of (min, max) for y-axis limits
    include_points : bool, optional
        Whether to include individual data points on the boxplot
    dimension_label : Optional[str], optional
        Label to use for the comparison dimension axis (defaults to "State")

    Returns
    -------
    bool
        True if plot was created successfully, False otherwise

    """
    axis_label = (
        dimension_label.strip() if dimension_label else "State"
    )

    if col_name in data_df.columns and not data_df[col_name].isnull().all():
        try:
            fig, ax = plt.subplots(figsize=(8 if include_points else 6, 6))

            # Create boxplot using seaborn
            sns.boxplot(
                data=data_df,
                x="state",
                y=col_name,
                hue="state",
                palette=state_color_map,
                ax=ax,
                width=0.6,  # Make boxes slightly narrower for more spacing
            )

            # Add individual data points if requested
            if include_points:
                sns.stripplot(
                    data=data_df,
                    x="state",
                    y=col_name,
                    color="black",
                    alpha=0.3,
                    size=4,
                    ax=ax,
                )

            # Clean up the plot
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel(title_prefix, fontsize=12)
            ax.set_xlabel(axis_label, fontsize=12)
            ax.set_title(
                f"{group_name}: {title_prefix} by {axis_label}",
                fontsize=14,
            )

            # Set y-axis limits if provided
            if y_limits is not None:
                ax.set_ylim(y_limits)

            # Rotate x-axis labels if needed
            if len(filter_state_names) > 3:
                ax.tick_params(axis="x", rotation=45)

            fig.tight_layout()

            # Save to the provided filename
            fig.savefig(
                filename,
                dpi=300,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            # plt.close(fig)

            return True

        except Exception as e:
            logger.error(
                f"Error creating boxplot for {group_name} {title_prefix}: {str(e)}"
            )
            return False
    else:
        logger.warning(f"No valid data found for {col_name} in {group_name}")
        return False


def create_cdf_preview(
    data_df: pd.DataFrame,
    col_name: str,
    group_name: str,
    identifier: str,
    title_prefix: str,
    filename: str,
    state_color_map: dict,
    filter_state_names: list,
    data_type: str = "activity",
    dimension_label: Optional[str] = None,
) -> bool:
    """Create and save CDF plot to the specified filename.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data to plot
    col_name : str
        Column name containing the data to plot
    group_name : str
        Name of the group for labeling
    identifier : str
        Identifier for the plot (used in title)
    title_prefix : str
        Prefix for the plot title
    filename : str
        Full path to save the plot file
    state_color_map : dict
        Dictionary mapping state names to colors
    filter_state_names : list
        List of state names to include in the plot
    data_type : str, optional
        Type of data being plotted (e.g., "activity", "correlation",
        "event_rate"), by default "activity"
    dimension_label : Optional[str], optional
        Label to use for the comparison dimension axis (defaults to "State")

    Returns
    -------
    bool
        True if plot was created successfully, False otherwise

    """
    axis_label = (
        dimension_label.strip() if dimension_label else "State"
    )

    if col_name in data_df.columns and not data_df[col_name].isnull().all():
        try:
            fig, ax = plt.subplots(figsize=(5, 4))

            # Create palette mapping state names to colors
            palette = [
                state_color_map.get(s, "gray") for s in filter_state_names
            ]

            sns.ecdfplot(
                data=data_df,
                x=col_name,
                hue="state",
                palette=palette,
                ax=ax,
            )

            # Clean up the plot
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel("Cumulative Probability", fontsize=12)
            ax.set_xlabel(title_prefix, fontsize=12)
            ax.set_title(
                f"{group_name}: {title_prefix} CDF by {axis_label}",
                fontsize=14,
            )

            # Rotate x-axis labels if needed
            if len(filter_state_names) > 3:
                ax.tick_params(axis="x", rotation=45)

            fig.tight_layout()

            # Save to the provided filename
            fig.savefig(
                filename,
                dpi=300,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            plt.close(fig)

            return True

        except Exception as e:
            logger.error(
                f"Error creating CDF plot for {group_name} {title_prefix}: {str(e)}"
            )
            return False
    else:
        logger.warning(f"No valid data found for {col_name} in {group_name}")
        return False
