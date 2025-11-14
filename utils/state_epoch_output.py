"""State-epoch output generation utilities.

This module provides utilities for generating plots, CSVs, and other
output files from state-epoch baseline analysis results.
"""

import json
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from beartype.typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass

from utils.state_epoch_results import StateEpochResults

# Consolidated imports to reduce redundancy in functions
from utils.plots import (
    plot_modulated_neuron_footprints,
    _plot_population_average,
    _plot_state_time,
    _plot_traces,
    _plot_raster,
)
from utils.utils import Comp

# Data preparation imports from appropriate modules
from utils.state_epoch_data import (
    load_and_filter_cell_contours,
)
from utils.state_epoch_results import (
    extract_common_modulation_data,
    prepare_event_modulation_data,
    DEFAULT_ALPHA,
)

# Plotting utilities imports
from utils.plotting_utils import (
    save_figure_with_cleanup,
    validate_data_availability,
)

# Import existing functions from correlations.py to avoid duplication
from analysis.correlations import (
    _correlations_to_csv,
)


@dataclass
class ColorScheme:
    """Centralized color scheme management."""

    state_colors: List[str]
    epoch_colors: List[str]
    modulation_colors: Optional[List[str]] = None
    correlation_colors: Optional[List[str]] = None

    def get_modulation_colors(self) -> Tuple[str, str, str]:
        """Get standardized modulation colors with fallbacks."""
        if self.modulation_colors and len(self.modulation_colors) >= 3:
            return (
                self.modulation_colors[0],
                self.modulation_colors[1],
                self.modulation_colors[2],
            )
        else:
            up_color = (
                self.modulation_colors[0] if self.modulation_colors else "red"
            )
            down_color = (
                self.modulation_colors[1]
                if self.modulation_colors and len(self.modulation_colors) > 1
                else "blue"
            )
            non_color = (
                self.modulation_colors[2]
                if self.modulation_colors and len(self.modulation_colors) > 2
                else "gray"
            )
            return up_color, down_color, non_color

    def get_state_color(self, state: str, states: List[str]) -> str:
        """Get color for a specific state."""
        try:
            state_idx = states.index(state)
            return self.state_colors[state_idx]
        except (ValueError, IndexError):
            return "gray"


# Configure matplotlib to reduce memory warnings
matplotlib.rcParams["figure.max_open_warning"] = 50  # Increase threshold
plt.ioff()  # Turn off interactive mode to prevent figure accumulation

logger = logging.getLogger(__name__)

# Consistent font settings
LABEL_FONT = {"fontsize": 12}
TITLE_FONT = {"fontsize": 13}

# Preview file constants aligned with existing tools (matching reference.py)
TIME_IN_STATE_PREVIEW = "time_in_state_preview.svg"
POPULATION_AVERAGE_PREVIEW = "population_average_preview.svg"
SPATIAL_CORRELATION_PREVIEW = "spatial_correlation_preview.svg"
SPATIAL_CORRELATION_MAP_PREVIEW = "spatial_correlation_map_preview.svg"
MODULATION_FOOTPRINT_PREVIEW = "modulation_footprint_preview.svg"
MODULATION_HISTOGRAM_PREVIEW = "modulation_histogram_preview.svg"
CORRELATION_MATRICES_PREVIEW = "correlation_matrices_preview.svg"
AVERAGE_CORRELATIONS_PREVIEW = "average_correlations_preview.svg"
CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW = (
    "correlation_statistic_distribution_preview.svg"
)
EVENT_AVERAGE_PREVIEW = "event_average_preview.svg"
EVENT_MODULATION_PREVIEW = "event_modulation_preview.svg"
EVENT_MODULATION_HISTOGRAM_PREVIEW = "event_modulation_histogram_preview.svg"

# Separate overlay preview files
TRACE_STATE_OVERLAY = "trace_state_overlay.svg"
EVENT_STATE_OVERLAY = "event_state_overlay.svg"

# Data file constants
ACTIVITY_PER_STATE_EPOCH_DATA_CSV = "activity_per_state_epoch_data.csv"
CORRELATIONS_PER_STATE_EPOCH_DATA_CSV = "correlations_per_state_epoch_data.csv"
MODULATION_VS_BASELINE_DATA_CSV = "modulation_vs_baseline_data.csv"
AVERAGE_CORRELATIONS_CSV = "average_correlations.csv"
RAW_CORRELATIONS_H5_NAME = "pairwise_correlation_heatmaps.h5"
RAW_CORRELATIONS_ZIP_NAME = "spatial_analysis_pairwise_correlations.zip"


class StateEpochOutputGenerator:
    """Generates outputs for state-epoch baseline analysis.

    All preview functions consistently show state-epoch combinations with
    appropriate overlays and color coding for comprehensive analysis.
    """

    def __init__(
        self,
        output_dir: str,
        states: List[str],
        epochs: List[str],
        state_colors: List[str],
        epoch_colors: List[str],
        baseline_state: str,
        baseline_epoch: str,
        alpha: float = DEFAULT_ALPHA,
        n_shuffle: int = 1000,
        epoch_periods: Optional[List[tuple]] = None,
        correlation_statistic: str = "max",
    ):
        """Initialize output generator.

        :Args
        ----
            output_dir: Directory for output files.
            states: List of state names.
            epochs: List of epoch names.
            state_colors: Colors for states.
            epoch_colors: Colors for epochs.
            baseline_state: Baseline state name.
            baseline_epoch: Baseline epoch name.
            alpha: Significance level.
            n_shuffle: Number of permutations for resampling-based tests.
            epoch_periods: Optional list of epoch time periods as tuples.
            correlation_statistic: Per-cell correlation statistic ("max",
                "min", or "mean") used for distribution previews.

        """
        self.output_dir = output_dir
        self.states = states
        self.epochs = epochs
        self.baseline_state = baseline_state
        self.baseline_epoch = baseline_epoch
        self.alpha = alpha
        self.n_shuffle = n_shuffle
        self.epoch_periods = epoch_periods
        self.correlation_statistic = self._normalize_correlation_statistic(
            correlation_statistic
        )

        # Create centralized color scheme
        self.color_scheme = ColorScheme(
            state_colors=state_colors, epoch_colors=epoch_colors
        )

        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def _get_output_path(self, filename: str) -> str:
        """Get full output path for a filename."""
        if self.output_dir:
            return os.path.join(self.output_dir, filename)
        return filename

    @staticmethod
    def _normalize_correlation_statistic(
        correlation_statistic: Optional[str],
    ) -> str:
        """Normalize correlation statistic selection."""
        if correlation_statistic is None:
            logger.warning(
                "No correlation_statistic provided; defaulting to 'max'."
            )
            return "max"

        normalized = str(correlation_statistic).strip().lower()
        if normalized not in {"max", "min", "mean"}:
            logger.warning(
                "Unsupported correlation_statistic '%s'; defaulting to 'max'.",
                correlation_statistic,
            )
            return "max"
        return normalized

    def generate_all_outputs(
        self,
        results: StateEpochResults,
        modulation_results: Dict[str, Any],
        cell_info: Dict[str, Any],
        traces: Optional[np.ndarray] = None,
        events: Optional[np.ndarray] = None,
        annotations_df: Optional["pd.DataFrame"] = None,
        column_name: str = "state",
    ) -> None:
        """Generate all output files.

        :Args
            results: StateEpochResults object containing analysis results.
            modulation_results: Dictionary containing modulation analysis results.
            cell_info: Dictionary with cell information including names and metadata.
            traces: Optional full trace data array for visualization.
            events: Optional full event data array for event analysis.
            annotations_df: Optional annotations DataFrame with behavioral state
                information for timeline reconstruction.
            column_name: Column name containing state labels in annotations.

        """
        logger.info("Generating state-epoch analysis outputs...")

        # Generate individual CSV files
        self._save_activity_summary_csv(results, cell_info)
        self._save_correlation_summary_csv(results, cell_info)
        self._save_modulation_results_csv(modulation_results, cell_info)

        self._save_average_correlations_csv(
            results
        )  # Correlations.py style format
        self._save_raw_correlations_h5(
            results
        )  # Raw correlation matrices in H5 format
        self._save_raw_correlations_zip(
            results, cell_info
        )  # Raw correlation data in ZIP

        # Generate preview plots
        self._generate_core_previews(
            results,
            modulation_results,
            cell_info,
            traces,
            events,
            annotations_df,
            column_name,
        )

        # Save comprehensive metadata file
        self._save_output_metadata(results, cell_info, modulation_results)

        logger.info("Output generation completed")

    def _save_activity_summary_csv(
        self, results: StateEpochResults, cell_info: Dict[str, Any]
    ) -> None:
        """Save activity summary CSV file with comprehensive trace and event data.

        Generate the activity_per_state_epoch_data.csv file with both
        trace-based activity and event-based metrics for comprehensive analysis.
        """
        data_rows = []
        cell_names = cell_info.get("cell_names", [])

        for state, epoch in results.get_all_combinations():
            # Get trace-based activity data
            mean_activity = results.get_activity_vector(state, epoch)
            if mean_activity is None or len(mean_activity) == 0:
                continue

            # Get additional activity metrics and event data
            combination_data = results.get_combination_results(state, epoch)
            std_activity = (
                combination_data.get("std_activity")
                if combination_data
                else None
            )
            median_activity = (
                combination_data.get("median_activity")
                if combination_data
                else None
            )
            activity_cv = (
                combination_data.get("activity_cv")
                if combination_data
                else None
            )

            # Get event-based metrics (matching trace statistics)
            event_rates = results.get_event_rates(state, epoch)

            # Get additional event statistics to match trace statistics
            event_std = (
                combination_data.get("event_std") if combination_data else None
            )
            event_median = (
                combination_data.get("event_median")
                if combination_data
                else None
            )
            event_cv = (
                combination_data.get("event_cv") if combination_data else None
            )

            for cell_idx, activity in enumerate(mean_activity):
                cell_name = (
                    cell_names[cell_idx]
                    if cell_idx < len(cell_names)
                    else f"cell_{cell_idx}"
                )

                # Extract event statistics for this cell if available
                cell_event_rate = (
                    event_rates[cell_idx]
                    if event_rates is not None and cell_idx < len(event_rates)
                    else np.nan
                )
                cell_event_std = (
                    event_std[cell_idx]
                    if event_std is not None and cell_idx < len(event_std)
                    else np.nan
                )
                cell_event_median = (
                    event_median[cell_idx]
                    if event_median is not None
                    and cell_idx < len(event_median)
                    else np.nan
                )
                cell_event_cv = (
                    event_cv[cell_idx]
                    if event_cv is not None and cell_idx < len(event_cv)
                    else np.nan
                )

                data_rows.append(
                    {
                        "name": cell_name,
                        "cell_index": cell_idx,
                        "state": state,
                        "epoch": epoch,
                        # Trace-based activity metrics
                        "mean_trace_activity": activity,
                        "std_trace_activity": (
                            std_activity[cell_idx]
                            if std_activity is not None
                            and cell_idx < len(std_activity)
                            else np.nan
                        ),
                        "median_trace_activity": (
                            median_activity[cell_idx]
                            if median_activity is not None
                            and cell_idx < len(median_activity)
                            else np.nan
                        ),
                        "trace_activity_cv": (
                            activity_cv[cell_idx]
                            if activity_cv is not None
                            and cell_idx < len(activity_cv)
                            else np.nan
                        ),
                        # Event-based activity metrics (matching trace structure)
                        "mean_event_rate": cell_event_rate,
                        "std_event_rate": cell_event_std,
                        "median_event_rate": cell_event_median,
                        "event_rate_cv": cell_event_cv,
                    }
                )

        if data_rows:
            df = pd.DataFrame(data_rows)
            output_path = self._get_output_path(
                ACTIVITY_PER_STATE_EPOCH_DATA_CSV
            )
            df.to_csv(output_path, index=False)
            logger.info(
                f"Saved activity summary with trace and event data to {output_path}"
            )
        else:
            logger.warning("No activity data available for CSV generation")

    def _save_correlation_summary_csv(
        self, results: StateEpochResults, cell_info: Dict[str, Any]
    ) -> None:
        """Save correlation summary CSV file with comprehensive trace and event correlations.

        Includes both per-cell statistics (max/min/mean) and population-level
        statistics (positive/negative) for both trace and event data.
        """
        # Build using centralized results methods only
        data_rows = []
        cell_names = cell_info.get("cell_names", [])

        for state, epoch in results.get_all_combinations():
            corr_stats = results.get_correlation_stats(state, epoch)
            if corr_stats is None:
                continue

            max_correlations = corr_stats["max_per_cell"]
            min_correlations = corr_stats["min_per_cell"]
            mean_correlations = corr_stats["mean_per_cell"]

            # Get trace correlation matrix to calculate positive/negative averages
            corr_matrix = results.get_correlation_matrix(state, epoch)
            pos_corr_trace, neg_corr_trace = (
                self._calculate_pos_neg_correlations(corr_matrix)
            )

            # Try to get event correlation matrix (if events were analyzed)
            combination_data = results.get_combination_results(state, epoch)
            event_corr_matrix = None
            if combination_data:
                event_corr_matrix = combination_data.get(
                    "event_correlation_matrix"
                )

            # Calculate event correlation statistics if available
            if event_corr_matrix is not None and event_corr_matrix.size > 0:
                event_corr_stats = self._calculate_correlation_stats(
                    event_corr_matrix
                )
                max_event_correlations = event_corr_stats["max_per_cell"]
                min_event_correlations = event_corr_stats["min_per_cell"]
                mean_event_correlations = event_corr_stats["mean_per_cell"]
                pos_corr_event, neg_corr_event = (
                    self._calculate_pos_neg_correlations(event_corr_matrix)
                )
            else:
                # No event correlations available - use NaN
                max_event_correlations = np.full(len(max_correlations), np.nan)
                min_event_correlations = np.full(len(max_correlations), np.nan)
                mean_event_correlations = np.full(
                    len(max_correlations), np.nan
                )
                pos_corr_event = np.nan
                neg_corr_event = np.nan

            for cell_idx in range(len(max_correlations)):
                cell_name = (
                    cell_names[cell_idx]
                    if cell_idx < len(cell_names)
                    else f"cell_{cell_idx}"
                )
                data_rows.append(
                    {
                        "name": cell_name,
                        "cell_index": cell_idx,
                        "state": state,
                        "epoch": epoch,
                        # Per-cell trace correlation statistics
                        "max_trace_correlation": max_correlations[cell_idx],
                        "min_trace_correlation": min_correlations[cell_idx],
                        "mean_trace_correlation": mean_correlations[cell_idx],
                        # Per-cell event correlation statistics
                        "max_event_correlation": max_event_correlations[
                            cell_idx
                        ],
                        "min_event_correlation": min_event_correlations[
                            cell_idx
                        ],
                        "mean_event_correlation": mean_event_correlations[
                            cell_idx
                        ],
                        # Population-level trace correlations
                        "positive_trace_correlation": pos_corr_trace,
                        "negative_trace_correlation": neg_corr_trace,
                        # Population-level event correlations
                        "positive_event_correlation": pos_corr_event,
                        "negative_event_correlation": neg_corr_event,
                    }
                )

        if data_rows:
            df = pd.DataFrame(data_rows)
            output_path = self._get_output_path(
                CORRELATIONS_PER_STATE_EPOCH_DATA_CSV
            )
            df.to_csv(output_path, index=False)
            logger.info(
                f"Saved correlation summary with trace and event data to {output_path}"
            )

    @staticmethod
    def _calculate_correlation_stats(
        corr_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculate per-cell correlation statistics from a correlation matrix.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Correlation matrix (n_cells x n_cells)

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - max_per_cell: Maximum correlation for each cell
            - min_per_cell: Minimum correlation for each cell
            - mean_per_cell: Mean correlation for each cell
            - mean_correlation: Mean of all off-diagonal correlations (population-level)

        """
        n_cells = corr_matrix.shape[0]
        max_per_cell = np.full(n_cells, np.nan)
        min_per_cell = np.full(n_cells, np.nan)
        mean_per_cell = np.full(n_cells, np.nan)

        try:
            # Calculate per-cell statistics (max, min, mean)
            for i in range(n_cells):
                # Get row excluding diagonal
                mask = np.ones(n_cells, dtype=bool)
                mask[i] = False
                off_diag = corr_matrix[i, mask]
                valid = off_diag[~np.isnan(off_diag)]

                if len(valid) > 0:
                    max_per_cell[i] = np.max(valid)
                    min_per_cell[i] = np.min(valid)
                    mean_per_cell[i] = np.mean(valid)

            # Calculate mean of all off-diagonal correlations (population-level)
            upper_triangle = np.triu(corr_matrix, k=1)
            valid_corrs = upper_triangle[~np.isnan(upper_triangle)]
            mean_corr = (
                np.mean(valid_corrs) if len(valid_corrs) > 0 else np.nan
            )

        except Exception as e:
            logger.warning(f"Failed to calculate correlation stats: {e}")
            mean_corr = np.nan

        return {
            "max_per_cell": max_per_cell,
            "min_per_cell": min_per_cell,
            "mean_per_cell": mean_per_cell,
            "mean_correlation": mean_corr,
        }

    @staticmethod
    def _calculate_pos_neg_correlations(
        corr_matrix: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """Calculate average positive and negative off-diagonal correlations.

        This matches the calculation in combine_compare_correlation_data tool.

        Parameters
        ----------
        corr_matrix : Optional[np.ndarray]
            Correlation matrix (n_cells x n_cells)

        Returns
        -------
        Tuple[float, float]
            (positive_correlation, negative_correlation)
            Returns (np.nan, np.nan) if matrix is None or calculation fails

        """
        if corr_matrix is None or corr_matrix.size == 0:
            return np.nan, np.nan

        try:
            # Collect all off-diagonal values
            n_cells = corr_matrix.shape[0]
            all_off_diagonal = []

            for i in range(n_cells):
                # Get row excluding diagonal element
                mask = np.ones(n_cells, dtype=bool)
                mask[i] = False
                off_diag_values = corr_matrix[i, mask]
                all_off_diagonal.extend(off_diag_values)

            # Convert to array and filter NaNs
            all_off_diagonal = np.array(all_off_diagonal)
            valid_correlations = all_off_diagonal[~np.isnan(all_off_diagonal)]

            if len(valid_correlations) == 0:
                return np.nan, np.nan

            # Calculate averages for positive and negative correlations
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

            return pos_corr, neg_corr

        except Exception as e:
            logger.warning(f"Failed to calculate pos/neg correlations: {e}")
            return np.nan, np.nan

    def _save_modulation_results_csv(
        self, modulation_results: Dict[str, Any], cell_info: Dict[str, Any]
    ) -> None:
        """Save modulation results CSV file with comprehensive trace and event data."""
        if "error" in modulation_results:
            logger.warning(
                f"Cannot save modulation results: {modulation_results['error']}"
            )
            return

        data_rows = []
        cell_names = cell_info.get("cell_names", [])

        # Get activity modulation (trace-based)
        activity_modulation = modulation_results.get("activity_modulation", {})
        # Get event modulation if available
        event_modulation = modulation_results.get("event_modulation", {})

        # Combine all state-epoch keys from both types
        all_state_epoch_keys = set(activity_modulation.keys()) | set(
            event_modulation.keys()
        )

        for state, epoch in all_state_epoch_keys:
            # Get trace modulation data
            trace_mod_data = activity_modulation.get((state, epoch), {})
            trace_modulation_index = trace_mod_data.get("modulation_index", [])
            trace_p_values = trace_mod_data.get("p_values", [])
            trace_significant = trace_mod_data.get("significant", [])
            trace_index_len = (
                len(trace_modulation_index)
                if trace_modulation_index is not None
                else 0
            )
            trace_p_len = (
                len(trace_p_values) if trace_p_values is not None else 0
            )
            trace_significant_len = (
                len(trace_significant) if trace_significant is not None else 0
            )

            # Get event modulation data
            event_mod_data = event_modulation.get((state, epoch), {})
            event_modulation_index = event_mod_data.get("modulation_index", [])
            event_p_values = event_mod_data.get("p_values", [])
            event_significant = event_mod_data.get("significant", [])
            event_index_len = (
                len(event_modulation_index)
                if event_modulation_index is not None
                else 0
            )
            event_p_len = (
                len(event_p_values) if event_p_values is not None else 0
            )
            event_significant_len = (
                len(event_significant) if event_significant is not None else 0
            )

            # Determine number of cells (use the larger of the two if sizes differ)
            n_cells = max(trace_index_len, event_index_len)

            if n_cells == 0:
                continue  # Skip if no data available

            for cell_idx in range(n_cells):
                cell_name = (
                    cell_names[cell_idx]
                    if cell_idx < len(cell_names)
                    else f"cell_{cell_idx}"
                )

                # Create trace modulation categorical indicator
                trace_value = (
                    trace_modulation_index[cell_idx]
                    if trace_modulation_index is not None
                    and trace_index_len > cell_idx
                    else None
                )
                trace_modulation_categorical = 0  # Default: non-modulated
                trace_is_significant = (
                    trace_significant is not None
                    and trace_significant_len > cell_idx
                    and bool(trace_significant[cell_idx])
                )
                if trace_is_significant and trace_value is not None:
                    if trace_value > 0:
                        trace_modulation_categorical = 1  # Up-modulated
                    elif trace_value < 0:
                        trace_modulation_categorical = -1  # Down-modulated

                # Create event modulation categorical indicator
                event_value = (
                    event_modulation_index[cell_idx]
                    if event_modulation_index is not None
                    and event_index_len > cell_idx
                    else None
                )
                event_modulation_categorical = 0  # Default: non-modulated
                event_is_significant = (
                    event_significant is not None
                    and event_significant_len > cell_idx
                    and bool(event_significant[cell_idx])
                )
                if event_is_significant and event_value is not None:
                    if event_value > 0:
                        event_modulation_categorical = 1  # Up-modulated
                    elif event_value < 0:
                        event_modulation_categorical = -1  # Down-modulated

                trace_p_value = (
                    trace_p_values[cell_idx]
                    if trace_p_values is not None and trace_p_len > cell_idx
                    else np.nan
                )
                event_p_value = (
                    event_p_values[cell_idx]
                    if event_p_values is not None and event_p_len > cell_idx
                    else np.nan
                )

                row = {
                    "name": cell_name,
                    "cell_index": cell_idx,
                    "state": state,
                    "epoch": epoch,
                    "baseline_state": self.baseline_state,
                    "baseline_epoch": self.baseline_epoch,
                    # Trace-based modulation columns
                    f"trace_modulation_scores in {state}-{epoch}": (
                        trace_value if trace_value is not None else np.nan
                    ),
                    f"trace_p_values in {state}-{epoch}": (trace_p_value),
                    f"trace_modulation in {state}-{epoch}": trace_modulation_categorical,
                    # Event-based modulation columns
                    f"event_modulation_scores in {state}-{epoch}": (
                        event_value if event_value is not None else np.nan
                    ),
                    f"event_p_values in {state}-{epoch}": (event_p_value),
                    f"event_modulation in {state}-{epoch}": event_modulation_categorical,
                }
                data_rows.append(row)

        df = pd.DataFrame(data_rows)
        output_path = self._get_output_path(MODULATION_VS_BASELINE_DATA_CSV)
        df.to_csv(output_path, index=False)
        logger.info(
            f"Saved modulation results with trace and event data to {output_path}"
        )

    def _save_average_correlations_csv(
        self, results: StateEpochResults
    ) -> None:
        """Save average correlations CSV file with both trace and event data."""
        data_rows = []

        for state, epoch in results.get_all_combinations():
            combination_results = results.get_combination_results(state, epoch)
            if combination_results is None:
                continue

            # Initialize row with state-epoch identifier
            row_data = {
                "state": f"{state}-{epoch}",
            }

            # Process trace correlations
            correlation_matrix = combination_results.get("correlation_matrix")
            if correlation_matrix is not None:
                # Calculate positive and negative trace correlations
                trace_corr_data = correlation_matrix[
                    np.triu_indices(correlation_matrix.shape[0], k=1)
                ]

                # Handle empty arrays to avoid warnings
                pos_trace_data = trace_corr_data[trace_corr_data > 0]
                pos_trace_corr = (
                    np.nanmean(pos_trace_data)
                    if len(pos_trace_data) > 0
                    else np.nan
                )

                neg_trace_data = trace_corr_data[trace_corr_data < 0]
                neg_trace_corr = (
                    np.nanmean(neg_trace_data)
                    if len(neg_trace_data) > 0
                    else np.nan
                )

                row_data["positive_trace_correlation"] = pos_trace_corr
                row_data["negative_trace_correlation"] = neg_trace_corr
            else:
                row_data["positive_trace_correlation"] = np.nan
                row_data["negative_trace_correlation"] = np.nan

            # Process event correlations
            event_correlation_matrix = combination_results.get(
                "event_correlation_matrix"
            )
            if event_correlation_matrix is not None:
                # Calculate positive and negative event correlations
                event_corr_data = event_correlation_matrix[
                    np.triu_indices(event_correlation_matrix.shape[0], k=1)
                ]

                # Handle empty arrays and NaN values
                pos_event_data = event_corr_data[
                    (event_corr_data > 0) & ~np.isnan(event_corr_data)
                ]
                pos_event_corr = (
                    np.nanmean(pos_event_data)
                    if len(pos_event_data) > 0
                    else np.nan
                )

                neg_event_data = event_corr_data[
                    (event_corr_data < 0) & ~np.isnan(event_corr_data)
                ]
                neg_event_corr = (
                    np.nanmean(neg_event_data)
                    if len(neg_event_data) > 0
                    else np.nan
                )

                row_data["positive_event_correlation"] = pos_event_corr
                row_data["negative_event_correlation"] = neg_event_corr
            else:
                row_data["positive_event_correlation"] = np.nan
                row_data["negative_event_correlation"] = np.nan

            data_rows.append(row_data)

        # Create DataFrame with consistent column order
        column_order = [
            "state",
            "positive_trace_correlation",
            "negative_trace_correlation",
            "positive_event_correlation",
            "negative_event_correlation",
        ]
        df = pd.DataFrame(data_rows).reindex(columns=column_order)

        output_path = self._get_output_path(AVERAGE_CORRELATIONS_CSV)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved average correlations CSV to {output_path}")

    def _save_raw_correlations_h5(self, results: StateEpochResults) -> None:
        """Save raw correlation matrices to H5 file.

        Saves both trace and event correlation matrices in a single H5 file
        with hierarchical structure: trace/{state-epoch} and event/{state-epoch}.
        """
        import h5py

        # Collect both trace and event correlation matrices
        trace_correlation_matrices = {}
        event_correlation_matrices = {}

        # Use centralized results accessor
        for state, epoch in results.get_all_combinations():
            key = f"{state}-{epoch}"

            # Get trace correlation matrix
            corr_matrix = results.get_correlation_matrix(state, epoch)
            if corr_matrix is not None:
                trace_correlation_matrices[key] = corr_matrix

            # Get event correlation matrix
            combination_results = results.get_combination_results(state, epoch)
            if combination_results:
                event_corr_matrix = combination_results.get(
                    "event_correlation_matrix"
                )
                if event_corr_matrix is not None:
                    event_correlation_matrices[key] = event_corr_matrix

        # Save to H5 file if we have any correlation matrices
        if trace_correlation_matrices or event_correlation_matrices:
            output_path = self._get_output_path(RAW_CORRELATIONS_H5_NAME)

            logger.info(f"Saving correlation matrices to {output_path}")
            with h5py.File(output_path, "w") as f:
                # Create groups for trace and event correlations
                if trace_correlation_matrices:
                    trace_group = f.create_group("trace")
                    for key, matrix in trace_correlation_matrices.items():
                        trace_group.create_dataset(key, data=matrix)
                    logger.info(
                        f"Saved {len(trace_correlation_matrices)} trace correlation matrices"
                    )

                if event_correlation_matrices:
                    event_group = f.create_group("event")
                    for key, matrix in event_correlation_matrices.items():
                        event_group.create_dataset(key, data=matrix)
                    logger.info(
                        f"Saved {len(event_correlation_matrices)} event correlation matrices"
                    )
        else:
            logger.warning("No correlation matrices to save to H5")

    def _save_raw_correlations_zip(
        self, results: StateEpochResults, cell_info: Dict[str, Any]
    ) -> None:
        """Save raw correlation matrices and cell name pairs to ZIP file."""
        # Collect correlation matrices and prepare data for correlations.py function
        correlation_matrices = {}
        cell_names = cell_info.get("cell_names", [])

        # Use centralized results accessor
        for state, epoch in results.get_all_combinations():
            corr_matrix = results.get_correlation_matrix(state, epoch)
            if corr_matrix is not None:
                key = f"{state}-{epoch}"
                correlation_matrices[key] = corr_matrix

        if correlation_matrices:
            # Change to output directory temporarily if needed
            current_dir = os.getcwd()
            try:
                if self.output_dir:
                    os.chdir(self.output_dir)

                # Use existing function from correlations.py
                # Note: This creates basic sort indices (identity mapping)
                sort_indices = {
                    key: np.arange(len(cell_names))
                    for key in correlation_matrices.keys()
                }

                _correlations_to_csv(
                    correlation_matrix=correlation_matrices,
                    cell_names=cell_names,
                    sort_indices=sort_indices,
                    positions=None,  # Position data handling can be added later if needed
                )
                logger.info(
                    "Saved raw correlations ZIP using correlations.py function"
                )
            finally:
                # Always restore original directory
                os.chdir(current_dir)
        else:
            logger.warning("No correlation matrices to save to ZIP")

    def _save_output_metadata(
        self,
        results: StateEpochResults,
        cell_info: Dict[str, Any],
        modulation_results: Dict[str, Any],
    ) -> None:
        """Save comprehensive output metadata JSON file."""
        from pathlib import Path

        # Get basic stats
        all_combinations = results.get_all_combinations()
        num_combinations = len(all_combinations)
        num_cells = len(cell_info.get("cell_names", []))

        # Create base values for all output files
        base_values = {
            "num_cells": num_cells,
            "num_states": len(self.states),
            "num_epochs": len(self.epochs),
            "num_combinations": num_combinations,
            "states": self.states,
            "epochs": self.epochs,
            "state_colors": self.color_scheme.state_colors,
            "epoch_colors": self.color_scheme.epoch_colors,
            "baseline_state": self.baseline_state,
            "baseline_epoch": self.baseline_epoch,
            "alpha": self.alpha,
            "n_shuffle": self.n_shuffle,
        }

        # Comprehensive metadata combining file descriptions and analysis details
        metadata = {
            # CSV data files with enhanced analysis information
            Path(ACTIVITY_PER_STATE_EPOCH_DATA_CSV).stem: {
                **base_values,
                "file_type": "activity_data",
                "description": "Mean activity per cell for each state-epoch combination",
                "analysis_type": "state_epoch_baseline_analysis",
                "combinations_analyzed": num_combinations,
                "activity_metrics": [
                    "mean_activity",
                    "std_activity",
                    "median_activity",
                    "cv_activity",
                ],
            },
            Path(CORRELATIONS_PER_STATE_EPOCH_DATA_CSV).stem: {
                **base_values,
                "file_type": "correlation_statistics",
                "description": "Correlation statistics per cell across state-epoch combinations",
                "analysis_type": "correlation_analysis",
                "correlation_method": "pearson",
            },
            Path(MODULATION_VS_BASELINE_DATA_CSV).stem: {
                **base_values,
                "file_type": "modulation_analysis",
                "description": "Modulation relative to baseline state-epoch combination",
                "analysis_type": "modulation_analysis",
                "baseline_reference": {
                    "state": self.baseline_state,
                    "epoch": self.baseline_epoch,
                },
                "modulation_method": "baseline_comparison",
            },
            Path(AVERAGE_CORRELATIONS_CSV).stem: {
                **base_values,
                "file_type": "correlation_summary",
                "description": "Average positive and negative correlations per state-epoch",
                "analysis_type": "correlation_analysis",
                "correlation_method": "pearson",
            },
            Path(CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW).stem: {
                **base_values,
                "file_type": "correlation_distribution_preview",
                "description": (
                    f"CDF and box plot of per-cell {self.correlation_statistic} "
                    "correlations across state-epoch combinations"
                ),
                "analysis_type": "correlation_analysis",
                "correlation_statistic": self.correlation_statistic,
            },
            # Raw correlation data files
            Path(RAW_CORRELATIONS_H5_NAME).stem: {
                **base_values,
                "file_type": "raw_correlation_matrices",
                "description": "Raw correlation matrices in HDF5 format",
                "format": "HDF5",
                "analysis_type": "correlation_analysis",
            },
            Path(RAW_CORRELATIONS_ZIP_NAME).stem: {
                **base_values,
                "file_type": "raw_correlation_data",
                "description": "Raw correlation matrices and cell pairs in ZIP archive",
                "format": "ZIP",
                "analysis_type": "correlation_analysis",
            },
            Path(TIME_IN_STATE_PREVIEW).stem: {
                **base_values,
                "file_type": "state_time_summary",
                "description": (
                    "Duration and fractional time spent in each behavioral state"
                ),
                "analysis_type": "state_time_analysis",
            },
        }

        # Save comprehensive metadata
        output_path = self._get_output_path("output_metadata.json")
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved comprehensive output metadata to {output_path}")

    def _generate_core_previews(
        self,
        results: StateEpochResults,
        modulation_results: Dict[str, Any],
        cell_info: Dict[str, Any],
        traces: Optional[np.ndarray] = None,
        events: Optional[np.ndarray] = None,
        annotations_df: Optional["pd.DataFrame"] = None,
        column_name: str = "state",
        modulation_colors: Optional[List[str]] = None,
        correlation_colors: Optional[List[str]] = None,
    ) -> None:
        """Generate all preview plots matching reference.py implementation.

        :Args
        ----
            results: StateEpochResults object containing analysis results.
            modulation_results: Modulation analysis results.
            cell_info: Cell information dictionary.
            traces: Optional full trace data array.
            events: Optional full event data array.
            annotations_df: Optional annotations DataFrame with behavioral state
                information.
            column_name: Column name containing state labels.
            modulation_colors: Optional list of colors for modulation visualization.
            correlation_colors: Optional list of colors for correlation visualization.

        """
        logger.info("Generating preview visualizations")

        # Update color scheme with optional colors
        if modulation_colors:
            self.color_scheme.modulation_colors = modulation_colors
        if correlation_colors:
            self.color_scheme.correlation_colors = correlation_colors

        # Generate all preview files
        self._safe_execute_preview(
            self._create_trace_preview,
            "trace preview",
            results,
            cell_info,
            traces,
            events,
            annotations_df=annotations_df,
            column_name=column_name,
        )

        self._safe_execute_preview(
            self._create_state_time_preview,
            "state time preview",
            annotations_df=annotations_df,
            column_name=column_name,
            cell_info=cell_info,
        )

        self._safe_execute_preview(
            self._create_population_average_plot,
            "population average preview",
            results,
            cell_info,
        )

        # Correlation previews
        self._safe_execute_preview(
            self._create_spatial_correlation_preview,
            "spatial correlation preview",
            results,
            cell_info,
        )

        self._safe_execute_preview(
            self._create_spatial_correlation_map_preview,
            "spatial correlation map preview",
            results,
            cell_info,
        )

        self._safe_execute_preview(
            self._create_correlation_matrices_preview,
            "correlation matrices preview",
            results,
            correlation_colors=correlation_colors,
        )

        self._safe_execute_preview(
            self._create_average_correlations_preview,
            "average correlations preview",
            results,
            correlation_colors=correlation_colors,
        )

        self._safe_execute_preview(
            self._create_correlation_statistic_distribution_preview,
            "correlation statistic distribution preview",
            results,
        )

        # Modulation previews
        self._safe_execute_preview(
            self._create_modulation_plot_with_type,
            "modulation footprint preview",
            modulation_results,
            cell_info,
            "map",
            MODULATION_FOOTPRINT_PREVIEW,
            "modulation footprint preview",
            modulation_colors,
        )

        self._safe_execute_preview(
            self._create_modulation_plot_with_type,
            "modulation histogram preview",
            modulation_results,
            cell_info,
            "hist",
            MODULATION_HISTOGRAM_PREVIEW,
            "modulation histogram preview",
            modulation_colors,
        )

        # Event-related previews if events available
        if events is not None:
            self._safe_execute_preview(
                self._create_event_preview,
                "event preview",
                results,
                cell_info,
                events,
                annotations_df=annotations_df,
                column_name=column_name,
            )

            self._safe_execute_preview(
                self._create_event_average_preview,
                "event average preview",
                results,
                cell_info,
                events,
            )

            self._safe_execute_preview(
                self._create_event_modulation_plot_with_type,
                "event modulation preview",
                results,
                modulation_results,
                cell_info,
                events,
                "map",
                EVENT_MODULATION_PREVIEW,
                modulation_colors,
            )

            self._safe_execute_preview(
                self._create_event_modulation_plot_with_type,
                "event modulation histogram preview",
                results,
                modulation_results,
                cell_info,
                events,
                "hist",
                EVENT_MODULATION_HISTOGRAM_PREVIEW,
                modulation_colors,
            )

    def _safe_execute_preview(
        self, preview_func, error_msg: str, *args, **kwargs
    ) -> None:
        """Safely execute a preview function with error handling.

        Parameters
        ----------
        preview_func : callable
            Function to execute
        error_msg : str
            Error message identifier for logging
        *args
            Arguments to pass to the function
        **kwargs
            Keyword arguments to pass to the function

        """
        try:
            preview_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Could not create {error_msg}: {e}")

    def _get_population_data_from_results(
        self, results: StateEpochResults, data_key: str = "mean_activity"
    ) -> Dict[str, Dict[str, Any]]:
        """Get population data directly from StateEpochResults without re-extraction.

        This leverages the existing data infrastructure instead of duplicating extraction logic.

        Args
        ----
            results: StateEpochResults object with pre-computed data
            data_key: Key to extract ('mean_activity' or 'event_rates')

        Returns
        -------
            Dictionary ready for population plotting

        """
        # Use results object's built-in iteration and data access
        data_structure = {}

        for state, epoch in results.get_all_combinations():
            combination_results = results.get_combination_results(state, epoch)

            if combination_results and data_key in combination_results:
                data_values = combination_results[data_key]
                if data_values is not None and (
                    data_key != "mean_activity" or len(data_values) > 0
                ):
                    combination_label = f"{state}_{epoch}"
                    data_structure[combination_label] = {
                        "mean_activity": data_values  # Standard key for plotting
                    }

        return data_structure

    def _plot_population_data_with_colors(
        self,
        results: StateEpochResults,
        data_structure: Dict[str, Dict[str, Any]],
        filename_constant: str,
        ylabel: str,
    ) -> None:
        """Plot population data with consistent state colors.

        :Args
        ----
            results: State-epoch analysis results
            data_structure: Data structure from _create_population_data_structure
            filename_constant: Output filename constant
            ylabel: Y-axis label for the plot

        """
        if not data_structure:
            return

        # Create consistent state colors for each combination
        state_colors = [
            self.color_scheme.get_state_color(state, self.states)
            for state, epoch in results.get_all_combinations()
            if f"{state}_{epoch}"
            in data_structure  # Only colors for data that exists
        ]

        # Use population activity tool's function with consistent state colors
        _plot_population_average(
            data=data_structure,
            filename=str(self._get_output_path(filename_constant)),
            state_colors=state_colors[
                : len(data_structure)
            ],  # Ensure correct length
            ylabel=ylabel,
            xlabel="State-Epoch",
        )

    def _create_population_average_plot(
        self, results: StateEpochResults, cell_info: Dict[str, Any]
    ) -> None:
        """Create population average plot leveraging existing StateEpochResults data."""
        try:
            # Get data directly from results object (no re-extraction needed)
            activity_data = self._get_population_data_from_results(
                results, "mean_activity"
            )

            if not activity_data:
                logger.warning(
                    "No activity data available for population average plot"
                )
                return

            # Plot with consistent colors
            self._plot_population_data_with_colors(
                results=results,
                data_structure=activity_data,
                filename_constant=POPULATION_AVERAGE_PREVIEW,
                ylabel="Mean Activity",
            )

        except Exception as e:
            logger.warning(f"Could not create population average plot: {e}")

    def _create_state_time_preview(
        self,
        annotations_df: Optional["pd.DataFrame"],
        column_name: str,
        cell_info: Dict[str, Any],
    ) -> None:
        """Create time-in-state summary plots mirroring population activity tool."""
        if annotations_df is None:
            logger.info(
                "Annotations unavailable for state time preview; skipping."
            )
            return

        period = cell_info.get("period", 1.0)
        try:
            _plot_state_time(
                annotations_df,
                column_name,
                self.states,
                self.color_scheme.state_colors[: len(self.states)],
                period,
                filename=self._get_output_path(TIME_IN_STATE_PREVIEW),
                epoch_names=self.epochs,
                epoch_periods=self.epoch_periods,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create state time preview: {exc}"
            ) from exc

    def _create_trace_preview(
        self,
        results: StateEpochResults,
        cell_info: Dict[str, Any],
        traces: Optional[np.ndarray] = None,
        events: Optional[np.ndarray] = None,
        annotations_df: Optional["pd.DataFrame"] = None,
        column_name: str = "state",
    ) -> None:
        """Create trace preview following epoch_activity.py pattern."""
        try:
            logger.info("Creating trace preview")

            if not validate_data_availability(
                traces, "traces", "No traces data available for trace preview"
            ):
                return

            # Get epoch information and create state-aware colors
            period = cell_info.get("period", 1.0)
            boundaries = cell_info.get("boundaries", [])

            # For trace preview, use actual temporal epochs, not state combinations
            epoch_names = self.epochs

            # Use epoch periods - must be provided by caller
            epochs = self.epoch_periods
            if epochs is None:
                # Create default epochs if none provided
                total_time = traces.shape[0] * period
                num_epochs = len(epoch_names)
                if num_epochs > 0:
                    epoch_duration = total_time / num_epochs
                    epochs = [
                        (i * epoch_duration, (i + 1) * epoch_duration)
                        for i in range(num_epochs)
                    ]
                else:
                    epochs = [(0, total_time)]
                    epoch_names = ["full_recording"]

            # Create separate overlay previews
            if annotations_df is not None:
                logger.info(
                    "Creating trace preview with state overlays"
                )

                # 1. Create state overlay preview (like population_activity.py)
                self._plot_trace_preview_with_state_overlays(
                    traces=traces,
                    behavior=annotations_df,
                    column_name=column_name,
                    boundaries=boundaries,
                    period=period,
                )

                # 2. Epoch overlay preview removed (not needed for state-epoch analysis)
                # self._plot_trace_preview_with_epoch_overlays(
                #     traces=traces,
                #     epochs=epochs,
                #     boundaries=boundaries,
                #     period=period,
                #     epoch_colors=epoch_colors[: len(epochs)],
                #     epoch_names=epoch_names[: len(epochs)],
                # )

            logger.info(
                f"Created trace state overlay preview: {TRACE_STATE_OVERLAY}"
            )

        except Exception as e:
            logger.warning(f"Could not create trace preview: {e}")

    def _create_correlation_matrices_preview(
        self,
        results: StateEpochResults,
        correlation_colors: Optional[List[str]] = None,
    ) -> None:
        """Create correlation matrices preview plot."""
        # Collect correlation matrices from all combinations
        correlation_matrices = {}

        for state, epoch in results.get_all_combinations():
            combination_results = results.get_combination_results(state, epoch)
            if (
                combination_results
                and "correlation_matrix" in combination_results
            ):
                key = f"{state}-{epoch}"
                correlation_matrices[key] = combination_results[
                    "correlation_matrix"
                ]

        if correlation_matrices:
            from analysis.correlations import plot_correlation_matrices

            # Use default colors if not provided
            if not correlation_colors:
                correlation_colors = ["red", "blue"]

            # Save current directory and change to output directory if needed
            current_dir = os.getcwd()
            try:
                if self.output_dir:
                    os.chdir(self.output_dir)

                plot_correlation_matrices(
                    correlation_matrix=correlation_matrices,
                    correlation_colors=correlation_colors,
                    out_file_name=CORRELATION_MATRICES_PREVIEW,
                )
            finally:
                os.chdir(current_dir)

    def _create_spatial_correlation_preview(
        self, results: StateEpochResults, cell_info: Dict[str, Any]
    ) -> None:
        """Create spatial correlation preview similar to correlations tool."""
        try:
            logger.info("Creating spatial correlation preview")

            # Get cell positions
            try:
                from analysis.correlations import (
                    cell_set_to_positions_mapping,
                )

                positions = cell_set_to_positions_mapping(
                    files=cell_info.get("cell_set_files", []),
                    cell_names=cell_info.get("cell_names", []),
                )
            except Exception as e:
                logger.warning(
                    f"Could not get cell positions for spatial correlation: {e}"
                )
                return

            if not positions:
                logger.warning(
                    "No spatial positions available for spatial correlation preview"
                )
                return

            # Use the plot_spatial_correlations function from correlations tool
            from analysis.correlations import (
                plot_spatial_correlations,
                _extract_triu_data_for_spatial_analysis,
            )

            # Extract correlation matrices from results
            correlation_matrix = {}
            for state, epoch in results.get_all_combinations():
                combination_results = results.get_combination_results(
                    state, epoch
                )
                if (
                    combination_results
                    and "correlation_matrix" in combination_results
                ):
                    state_epoch_key = f"{state}_{epoch}"
                    correlation_matrix[state_epoch_key] = combination_results[
                        "correlation_matrix"
                    ]

            if not correlation_matrix:
                logger.warning(
                    "No correlation matrices available for spatial correlation preview"
                )
                return

            # Extract triu data for spatial analysis
            empty_sort_indices = {
                state: np.arange(len(cell_info.get("cell_names", [])))
                for state in correlation_matrix.keys()
            }
            triu_data = _extract_triu_data_for_spatial_analysis(
                correlation_matrix=correlation_matrix,
                cell_names=cell_info.get("cell_names", []),
                sort_indices=empty_sort_indices,
                positions=positions,
            )

            if triu_data:
                output_path = self._get_output_path(
                    SPATIAL_CORRELATION_PREVIEW
                )
                # Create shortened labels inline for better spacing
                shortened_triu_data = {
                    key.replace("_", "-"): df for key, df in triu_data.items()
                }
                plot_spatial_correlations(
                    triu_data=shortened_triu_data,
                    out_file_name=str(output_path),
                )
                logger.info(
                    f"Created spatial correlation preview: {output_path}"
                )
            else:
                logger.warning("No spatial correlation data available")

        except Exception as e:
            logger.warning(
                f"Could not create spatial correlation preview: {e}"
            )

    def _create_spatial_correlation_map_preview(
        self, results: StateEpochResults, cell_info: Dict[str, Any]
    ) -> None:
        """Create spatial correlation map preview similar to correlations tool."""
        try:
            logger.info("Creating spatial correlation map preview")

            # Get cell positions
            try:
                from analysis.correlations import (
                    cell_set_to_positions_mapping,
                )

                positions = cell_set_to_positions_mapping(
                    files=cell_info.get("cell_set_files", []),
                    cell_names=cell_info.get("cell_names", []),
                )
            except Exception as e:
                logger.warning(
                    f"Could not get cell positions for spatial correlation map: {e}"
                )
                return

            if not positions:
                logger.warning(
                    "No spatial positions available for spatial correlation map preview"
                )
                return

            # Use the plot_correlation_spatial_map function from correlations tool
            from analysis.correlations import plot_correlation_spatial_map

            # Extract correlation matrices from results
            correlation_matrix = {}
            for state, epoch in results.get_all_combinations():
                combination_results = results.get_combination_results(
                    state, epoch
                )
                if (
                    combination_results
                    and "correlation_matrix" in combination_results
                ):
                    state_epoch_key = f"{state}_{epoch}"
                    correlation_matrix[state_epoch_key] = combination_results[
                        "correlation_matrix"
                    ]

            if not correlation_matrix:
                logger.warning(
                    "No correlation matrices available for spatial correlation map preview"
                )
                return

            output_path = self._get_output_path(
                SPATIAL_CORRELATION_MAP_PREVIEW
            )
            plot_correlation_spatial_map(
                correlation_matrix=correlation_matrix,
                positions=positions,
                cell_names=cell_info.get("cell_names", []),
                correlation_threshold=0.5,
                correlation_colors=["red", "blue"],
                max_lines=500,
                out_file_name=str(output_path),
            )
            logger.info(
                f"Created spatial correlation map preview: {output_path}"
            )

        except Exception as e:
            logger.warning(
                f"Could not create spatial correlation map preview: {e}"
            )

    def _create_average_correlations_preview(
        self,
        results: StateEpochResults,
        correlation_colors: Optional[List[str]] = None,
    ) -> None:
        """Create average correlations preview with proper state-epoch labeling."""
        try:
            # Extract correlation matrices from results
            correlation_matrices = {}
            for state, epoch in results.get_all_combinations():
                combination_results = results.get_combination_results(
                    state, epoch
                )
                if (
                    combination_results
                    and "correlation_matrix" in combination_results
                ):
                    key = f"{state}_{epoch}"
                    correlation_matrices[key] = combination_results[
                        "correlation_matrix"
                    ]

            if not correlation_matrices:
                logger.warning(
                    "No correlation matrices available for average correlations preview"
                )
                return

            # Generate enough colors for all state-epoch combinations
            num_combinations = len(correlation_matrices)
            if num_combinations == 0:
                return

            # Create consistent colors for each state-epoch combination using state colors
            colors_for_avg_corr = [
                self.color_scheme.get_state_color(state, self.states)
                for state, _epoch in results.get_all_combinations()
            ]

            # Create our own version of average correlations plot with better spacing
            self._plot_average_correlations_with_state_epoch_labels(
                correlation_matrices, colors_for_avg_corr
            )

        except Exception as e:
            logger.warning(
                f"Could not create average correlations preview: {e}"
            )

    def _determine_combination_order(
        self,
        results: StateEpochResults,
        valid_labels: Optional[Set[str]] = None,
    ) -> List[Tuple[str, str]]:
        """Determine ordered list of combinations respecting configured state order."""
        ordered: List[Tuple[str, str]] = []

        for state in self.states:
            for epoch in self.epochs:
                label = f"{state}-{epoch}"
                if valid_labels is not None and label not in valid_labels:
                    continue
                if results.has_combination(state, epoch):
                    ordered.append((state, epoch))

        for state, epoch in results.get_all_combinations():
            label = f"{state}-{epoch}"
            if valid_labels is not None and label not in valid_labels:
                continue
            if (state, epoch) not in ordered:
                ordered.append((state, epoch))

        return ordered

    @staticmethod
    def _select_correlation_statistic_array(
        stats: Dict[str, Any], correlation_statistic: str
    ) -> Optional[np.ndarray]:
        """Select per-cell correlation statistic array from stats dictionary."""
        stat_map = {
            "max": stats.get("max_per_cell"),
            "min": stats.get("min_per_cell"),
            "mean": stats.get("mean_per_cell"),
        }
        return stat_map.get(correlation_statistic)

    def _collect_correlation_statistic_values(
        self,
        results: StateEpochResults,
        correlation_statistic: str,
    ) -> pd.DataFrame:
        """Collect per-cell correlation statistic values for plotting."""
        rows: List[Dict[str, Any]] = []

        for state, epoch in self._determine_combination_order(results):
            stats = results.get_correlation_stats(state, epoch)
            if not stats:
                continue

            values = self._select_correlation_statistic_array(
                stats, correlation_statistic
            )
            if values is None:
                continue

            finite_mask = np.isfinite(values)
            valid_values = values[finite_mask]
            if valid_values.size == 0:
                continue

            label = f"{state}-{epoch}"
            for value in valid_values:
                rows.append(
                    {
                        "state": state,
                        "epoch": epoch,
                        "state_epoch": label,
                        "correlation_value": float(value),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "state",
                    "epoch",
                    "state_epoch",
                    "correlation_value",
                ]
            )

        return pd.DataFrame(rows)

    def _create_correlation_statistic_distribution_preview(
        self,
        results: StateEpochResults,
    ) -> None:
        """Create CDF and box plot for the selected per-cell correlation statistic."""
        correlation_df = self._collect_correlation_statistic_values(
            results, self.correlation_statistic
        )

        if correlation_df.empty:
            logger.warning(
                "No per-cell correlation statistics available for distribution preview"
            )
            return

        valid_labels = set(correlation_df["state_epoch"].unique())
        combination_order = self._determine_combination_order(
            results, valid_labels
        )

        if not combination_order:
            logger.warning(
                "No valid state-epoch combinations for correlation statistic preview"
            )
            return

        labels = [f"{state}-{epoch}" for state, epoch in combination_order]
        palette = {
            label: self.color_scheme.get_state_color(state, self.states)
            for (state, epoch), label in zip(combination_order, labels)
        }

        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # CDF plot
        statistic_label = self.correlation_statistic.capitalize()
        for label in labels:
            subset = correlation_df[correlation_df["state_epoch"] == label]
            if subset.empty:
                continue
            sns.ecdfplot(
                data=subset,
                x="correlation_value",
                ax=axes[0],
                label=label,
                color=palette[label],
            )

        axes[0].set_xlim((-1, 1))
        axes[0].set_xticks([-1, -0.5, 0, 0.5, 1])
        axes[0].set_xlabel("Correlation", fontdict=LABEL_FONT)
        axes[0].set_ylabel(
            f"Cumulative probability of per-cell {statistic_label} correlation",
            fontdict=LABEL_FONT,
        )
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].tick_params(
            axis="both", which="major", labelsize=LABEL_FONT["fontsize"]
        )
        if len(labels) > 1:
            axes[0].legend(
                loc="lower right", fontsize=LABEL_FONT["fontsize"] - 2
            )

        # Box plot
        sns.boxplot(
            data=correlation_df,
            x="state_epoch",
            y="correlation_value",
            ax=axes[1],
            order=labels,
            palette=palette,
            linewidth=3,
            fliersize=0,
        )
        sns.stripplot(
            data=correlation_df,
            x="state_epoch",
            y="correlation_value",
            order=labels,
            ax=axes[1],
            color="black",
            size=2,
            alpha=0.2,
            jitter=0.05,
        )
        axes[1].set_xlabel("State-Epoch", fontdict=LABEL_FONT)
        axes[1].set_ylabel(
            f"Per-cell {statistic_label} correlation", fontdict=LABEL_FONT
        )
        axes[1].tick_params(
            axis="both", which="major", labelsize=LABEL_FONT["fontsize"]
        )
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].set_ylim((-1.05, 1.05))

        fig.tight_layout()
        save_figure_with_cleanup(
            fig,
            self._get_output_path(CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW),
            "correlation statistic distribution preview",
        )

    def _create_modulation_plot_with_type(
        self,
        modulation_results: Dict[str, Any],
        cell_info: Dict[str, Any],
        plot_type: str = "map",
        filename: str = None,
        title: str = "modulation",
        modulation_colors: Optional[List[str]] = None,
    ) -> None:
        """Create modulation plot (footprint or histogram) with unified logic.

        Parameters
        ----------
        modulation_results : Dict[str, Any]
            Dictionary containing modulation analysis results
        cell_info : Dict[str, Any]
            Dictionary containing cell information
        plot_type : str, default="map"
            Type of plot ("map" for spatial footprint or "hist" for histogram)
        filename : str, optional
            Output filename constant
        title : str, default="modulation"
            Description for logging
        modulation_colors : List[str], optional
            Optional custom colors for modulation states

        """
        try:
            logger.info(f"Creating {title}")

            # Load and filter cell contours using helper
            x, y = load_and_filter_cell_contours(cell_info)

            # Get modulation source data
            if "activity_modulation" in modulation_results:
                modulation_source = modulation_results["activity_modulation"]
            else:
                logger.warning(
                    f"No activity_modulation data available for {title}"
                )
                modulation_source = None

            if not validate_data_availability(
                modulation_source, "modulation source"
            ):
                return

            # Extract and process modulation data using helper
            modulation_data = extract_common_modulation_data(
                modulation_source, len(x), alpha=self.alpha
            )

            # Add baseline entry with real baseline activity data
            baseline_key = f"{self.baseline_state}_{self.baseline_epoch}"
            if baseline_key not in modulation_data and len(x) > 0:
                try:
                    if (
                        "baseline_mean_activity" in modulation_results
                        and modulation_results["baseline_mean_activity"]
                        is not None
                    ):
                        baseline_activity = modulation_results[
                            "baseline_mean_activity"
                        ]
                        # Ensure baseline_activity is an array and get its length safely
                        baseline_activity = np.atleast_1d(baseline_activity)
                        activity_size = len(baseline_activity)

                        modulation_data[baseline_key] = {
                            "mean_activity": baseline_activity,
                            "modulation_scores": np.zeros(activity_size),
                            "p_val": np.ones(activity_size),
                            "up_modulated_neurons": np.array([]),
                            "down_modulated_neurons": np.array([]),
                        }
                except Exception as e:
                    logger.debug(f"Could not create baseline data: {e}")
                    pass  # Skip if baseline data cannot be retrieved

            if len(modulation_data) >= 1 and len(x) > 0:
                # Set default filename based on plot type if not provided
                if filename is None:
                    if plot_type == "footprint":
                        filename = "modulation_footprint_preview.svg"
                    elif plot_type == "histogram":
                        filename = "modulation_histogram_preview.svg"
                    else:
                        filename = f"modulation_{plot_type}_preview.svg"
                output_path = self._get_output_path(filename)

                # Set modulation colors using color scheme
                if modulation_colors:
                    self.color_scheme.modulation_colors = modulation_colors
                up_color, down_color, non_color = (
                    self.color_scheme.get_modulation_colors()
                )

                plot_modulated_neuron_footprints(
                    data=modulation_data,
                    x=x,
                    y=y,
                    filename=str(output_path),
                    plot_type=plot_type,
                    up_modulation_color=up_color,
                    down_modulation_color=down_color,
                    non_modulation_color=non_color,
                    method=Comp.BASELINE.value,
                    baseline_state=baseline_key,
                )
                logger.info(f"Created {title}: {output_path}")
            else:
                logger.warning(f"No modulation data available for {title}")

        except Exception as e:
            logger.warning(f"Could not create {title}: {e}")

    def _plot_average_correlations_with_state_epoch_labels(
        self, correlation_matrices: Dict[str, np.ndarray], colors: List[str]
    ) -> None:
        """Create average correlations plot with proper state-epoch labels and spacing."""
        import pandas as pd
        import seaborn as sns

        # Collect individual correlation data points for each state-epoch combination
        pos_data = []
        neg_data = []

        for state_epoch in correlation_matrices.keys():
            corr_matrix = correlation_matrices[state_epoch]
            corr_data = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]

            # Create clean state-epoch label with no extra text
            display_label = state_epoch.replace("_", "-")

            # Separate positive and negative correlations
            pos_values = corr_data[corr_data > 0]
            neg_values = corr_data[corr_data < 0]

            # Add individual data points for scatter plot
            for val in pos_values:
                if not np.isnan(val):
                    pos_data.append([display_label, val])

            for val in neg_values:
                if not np.isnan(val):
                    neg_data.append([display_label, val])

        # Create DataFrames for positive and negative correlations
        pos_df = (
            pd.DataFrame(pos_data, columns=["state", "positive"])
            if pos_data
            else pd.DataFrame(columns=["state", "positive"])
        )
        neg_df = (
            pd.DataFrame(neg_data, columns=["state", "negative"])
            if neg_data
            else pd.DataFrame(columns=["state", "negative"])
        )

        # Get all unique state-epoch combinations for color mapping
        all_states = list(correlation_matrices.keys())
        colors = colors[: len(all_states)]

        # Create figure with more space for labels and rotated text
        fig, ax = plt.subplots(figsize=(12, 10), nrows=2)

        # Use local LABEL_FONT for consistent styling

        # Plot positive correlations with box plot and scatter overlay (like trace/event plots)
        if not pos_df.empty:
            sns.boxplot(
                data=pos_df,
                x="state",
                y="positive",
                ax=ax[0],
                palette=colors,
                hue="state",
                linewidth=3,
                fliersize=0,
                legend=False,
            )
            sns.stripplot(
                data=pos_df,
                x="state",
                y="positive",
                ax=ax[0],
                color="black",
                size=3,
                alpha=0.2,
                jitter=0.02,
                legend=False,
            )

        # Plot negative correlations with box plot and scatter overlay (like trace/event plots)
        if not neg_df.empty:
            sns.boxplot(
                data=neg_df,
                x="state",
                y="negative",
                ax=ax[1],
                palette=colors,
                hue="state",
                linewidth=3,
                fliersize=0,
                legend=False,
            )
            sns.stripplot(
                data=neg_df,
                x="state",
                y="negative",
                ax=ax[1],
                color="black",
                size=3,
                alpha=0.2,
                jitter=0.02,
                legend=False,
            )

        ax[0].set_ylabel("Average Positive Correlation", fontdict=LABEL_FONT)
        ax[0].set_xlabel("State-Epoch", fontdict=LABEL_FONT)
        ax[1].set_ylabel("Average Negative Correlation", fontdict=LABEL_FONT)
        ax[1].set_xlabel("State-Epoch", fontdict=LABEL_FONT)

        # Remove spines
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)

        # Rotate x-axis labels to prevent overlap and add padding
        for axis in [ax[0], ax[1]]:
            labels = [item.get_text() for item in axis.get_xticklabels()]
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.tick_params(axis="x", pad=5)

        # Adjust layout with more space for rotated labels
        fig.tight_layout(pad=3.0, h_pad=1.0)
        fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.2)

        # Save figure using helper function
        save_figure_with_cleanup(
            fig,
            self._get_output_path(AVERAGE_CORRELATIONS_PREVIEW),
            "average correlations preview",
        )

        # CSV is saved by _save_average_correlations_csv; avoid duplicate writes here

    def _create_event_preview(
        self,
        results: StateEpochResults,
        cell_info: Dict[str, Any],
        events: np.ndarray,
        annotations_df: Optional["pd.DataFrame"] = None,
        column_name: str = "state",
    ) -> None:
        """Create event preview following epoch_activity.py pattern."""
        try:
            logger.info("Creating event preview")

            # Get epoch information and create state-aware colors
            period = cell_info.get("period", 1.0)
            boundaries = cell_info.get("boundaries", [])

            # For event preview, use actual temporal epochs, not state combinations
            epoch_names = self.epochs

            # Use epoch periods - must be provided by caller
            epochs = self.epoch_periods
            if epochs is None:
                # Create default epochs if none provided
                total_time = events.shape[0] * period
                num_epochs = len(epoch_names)
                if num_epochs > 0:
                    epoch_duration = total_time / num_epochs
                    epochs = [
                        (i * epoch_duration, (i + 1) * epoch_duration)
                        for i in range(num_epochs)
                    ]
                else:
                    epochs = [(0, total_time)]
                    epoch_names = ["full_recording"]

            # Convert event data to offsets format (time indices where events occurred)
            offsets = []
            for cell_idx in range(events.shape[1]):
                # Find timepoints where events occurred for this cell
                event_indices = np.where(events[:, cell_idx] > 0)[0]
                # Convert indices to time offsets in seconds
                event_times = event_indices * period
                offsets.append(event_times)

            # Create separate overlay previews
            if annotations_df is not None:
                logger.info(
                    "Creating event preview with state overlays"
                )

                # 1. Create state overlay event preview (like population_activity.py)
                self._plot_event_preview_with_state_overlays(
                    events=offsets,  # Use offset data for raster plot
                    event_timeseries=events,
                    behavior=annotations_df,
                    column_name=column_name,
                    boundaries=boundaries,
                    period=period,
                )

                # 2. Epoch overlay event preview removed (not needed for state-epoch analysis)
                # self._plot_event_preview_with_epoch_overlays(
                #     events=offsets,
                #     event_timeseries=events,
                #     epochs=epochs,
                #     boundaries=boundaries,
                #     period=period,
                #     epoch_colors=epoch_colors[: len(epochs)],
                #     epoch_names=epoch_names[: len(epochs)],
                # )
            logger.info(
                f"Created event state overlay preview: {EVENT_STATE_OVERLAY}"
            )

        except Exception as e:
            logger.warning(f"Could not create event preview: {e}")

    def _create_event_average_preview(
        self,
        results: StateEpochResults,
        cell_info: Dict[str, Any],
        events: np.ndarray,
    ) -> None:
        """Create event average preview leveraging existing StateEpochResults data."""
        try:
            # Get data directly from results object (no re-extraction needed)
            event_data = self._get_population_data_from_results(
                results, "event_rates"
            )

            if not event_data:
                logger.warning(
                    "No event data available for event average preview"
                )
                return

            # Plot with consistent colors
            self._plot_population_data_with_colors(
                results=results,
                data_structure=event_data,
                filename_constant=EVENT_AVERAGE_PREVIEW,
                ylabel="Event Rate (Hz)",
            )

        except Exception as e:
            logger.warning(f"Could not create event average preview: {e}")

    def _create_event_modulation_plot_with_type(
        self,
        results: StateEpochResults,
        modulation_results: Dict[str, Any],
        cell_info: Dict[str, Any],
        events: np.ndarray,
        plot_type: str,
        filename: str,
        modulation_colors: Optional[List[str]] = None,
    ) -> None:
        """Unified event modulation plotting function."""
        try:
            # Prepare event modulation data using helper
            event_data = prepare_event_modulation_data(
                results,
                modulation_results,
                cell_info,
                self.baseline_state,
                self.baseline_epoch,
                alpha=self.alpha,
            )
            if not validate_data_availability(
                event_data, "event data", "No baseline activity data available"
            ):
                return

            # event_data already prepared by helper; proceed to plotting

            # Load and filter cell contours using helper
            try:
                x, y = load_and_filter_cell_contours(cell_info)

            except Exception as e:
                logger.warning(f"Could not get cell contours: {e}")
                return

            # Use population activity tool's function

            # Get modulation colors via centralized color scheme
            if modulation_colors:
                self.color_scheme.modulation_colors = modulation_colors
            up_color, down_color, non_color = (
                self.color_scheme.get_modulation_colors()
            )

            plot_modulated_neuron_footprints(
                data=event_data,
                x=x,
                y=y,
                filename=str(self._get_output_path(filename)),
                plot_type=plot_type,
                up_modulation_color=up_color,
                down_modulation_color=down_color,
                non_modulation_color=non_color,
                method=Comp.BASELINE.value,
                baseline_state=f"{self.baseline_state}_{self.baseline_epoch}",
            )

        except Exception as e:
            logger.warning(
                f"Could not create event modulation {plot_type} preview: {e}"
            )

    def _plot_trace_preview_with_state_overlays(
        self,
        traces: np.ndarray,
        behavior: "pd.DataFrame",
        column_name: str,
        boundaries: List[float],
        period: float,
    ) -> None:
        """Plot trace preview with state colors applied directly to traces.

        Uses population_activity.py pattern.

        This reuses the existing _plot_traces function from population_activity.py.
        """
        # Use the existing _plot_traces function from population_activity.py
        _plot_traces(
            traces=traces,
            behavior=behavior,
            data=None,  # No additional data needed
            column_name=column_name,
            filename=TRACE_STATE_OVERLAY,
            state_colors=self.color_scheme.state_colors,
            state_names=self.states,
            period=period,
            boundaries=boundaries,
        )

    def _plot_event_preview_with_state_overlays(
        self,
        events: List[np.ndarray],
        event_timeseries: np.ndarray,
        behavior: "pd.DataFrame",
        column_name: str,
        boundaries: List[float],
        period: float,
    ) -> None:
        """Plot event preview with state colors applied directly to raster.

        Uses population_activity.py pattern.

        This reuses the existing _plot_raster function from population_activity.py.
        """
        # Use the existing _plot_raster function from population_activity.py
        _plot_raster(
            events=events,
            event_timeseries=event_timeseries,
            behavior=behavior,
            column_name=column_name,
            period=period,
            state_colors=self.color_scheme.state_colors,
            state_names=self.states,
            filename=EVENT_STATE_OVERLAY,
            boundaries=boundaries,
        )

