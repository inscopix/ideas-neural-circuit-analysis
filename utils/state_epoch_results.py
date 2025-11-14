"""State-epoch results management utilities.

This module provides utilities for storing, organizing, and accessing
analysis results from state-epoch baseline analysis.
"""

import logging
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Dict, Any, Tuple
from ideas.exceptions import IdeasError
from ideas import measures

# Import modulation functions from population_activity.py to avoid duplication
from analysis.population_activity import (
    find_two_state_modulated_neurons,
)

# Analysis constants
STRONG_CORRELATION_THRESHOLD = 0.5
DEFAULT_ALPHA = 0.05
DIVISION_SAFETY_EPSILON = 1e-10
MIN_TIMEPOINTS_FOR_RELIABLE_ANALYSIS = 10
MIN_ACTIVITY_THRESHOLD = 1e-10
MIN_EVENT_RATE_THRESHOLD = 1e-10

logger = logging.getLogger(__name__)


@beartype
def _compute_correlation_statistics(
    correlation_matrix: np.ndarray,
    strong_threshold: float = STRONG_CORRELATION_THRESHOLD,
) -> Dict[str, Any]:
    """Compute statistical metrics from a correlation matrix.

    Args
    ----
        correlation_matrix: Pre-computed correlation matrix
        strong_threshold: Threshold for strong correlations

    Returns
    -------
        Dictionary with correlation statistics

    """
    # Get upper triangle for analysis (exclude diagonal)
    triu_indices = np.triu_indices(correlation_matrix.shape[0], k=1)
    triu_values = correlation_matrix[triu_indices]

    # Filter out NaN values for valid statistics
    valid_values = triu_values[~np.isnan(triu_values)]

    # Check if we have valid data
    if len(valid_values) == 0:
        # Return all NaN statistics when no valid data
        return {
            "mean_correlation": np.nan,
            "std_correlation": np.nan,
            "median_correlation": np.nan,
            "max_correlation": np.nan,
            "min_correlation": np.nan,
            "mean_positive_correlation": np.nan,
            "mean_negative_correlation": np.nan,
            "positive_corr_fraction": 0.0,
            "negative_corr_fraction": 0.0,
            "strong_positive_corr_count": 0,
            "strong_negative_corr_count": 0,
            "correlation_range": np.nan,
        }

    pos_corr = valid_values[valid_values > 0]
    neg_corr = valid_values[valid_values < 0]

    return {
        "mean_correlation": np.mean(valid_values),
        "std_correlation": np.std(valid_values),
        "median_correlation": np.median(valid_values),
        "max_correlation": np.max(valid_values),
        "min_correlation": np.min(valid_values),
        "mean_positive_correlation": (
            np.mean(pos_corr) if len(pos_corr) > 0 else np.nan
        ),
        "mean_negative_correlation": (
            np.mean(neg_corr) if len(neg_corr) > 0 else np.nan
        ),
        "positive_corr_fraction": (
            len(pos_corr) / len(valid_values) if len(valid_values) > 0 else 0
        ),
        "negative_corr_fraction": (
            len(neg_corr) / len(valid_values) if len(valid_values) > 0 else 0
        ),
        "strong_positive_corr_count": np.sum(valid_values > strong_threshold),
        "strong_negative_corr_count": np.sum(valid_values < -strong_threshold),
        "correlation_range": np.max(valid_values) - np.min(valid_values),
    }


@beartype
def _calculate_correlation_metrics(
    traces: np.ndarray, strong_threshold: float = STRONG_CORRELATION_THRESHOLD
) -> Dict[str, Any]:
    """Calculate correlation metrics for traces.

    Args
    ----
        traces: Neural activity traces (timepoints x cells)
        strong_threshold: Threshold for strong correlations

    Returns
    -------
        Dictionary with correlation metrics

    """
    if traces.shape[1] <= 1:
        return {"correlation_matrix": None}

    try:
        # Use existing correlation computation logic
        correlation_matrix = measures.correlation_matrix(
            traces, fill_diagonal=0.0
        )

        # Compute statistics using shared function
        stats = _compute_correlation_statistics(
            correlation_matrix, strong_threshold
        )

        # Combine matrix and statistics
        result = {"correlation_matrix": correlation_matrix}
        result.update(stats)
        return result

    except Exception as e:
        # Log the error but don't fail the entire analysis
        logger.warning(f"Correlation calculation failed: {e}")
        return {"correlation_matrix": None}


@beartype
def _calculate_activity_metrics(
    traces: np.ndarray, division_epsilon: float = DIVISION_SAFETY_EPSILON
) -> Dict[str, Any]:
    """Calculate basic activity metrics for traces.

    Args
    ----
        traces: Neural activity traces (timepoints x cells)
        division_epsilon: Small value to prevent division by zero

    Returns
    -------
        Dictionary with activity metrics

    """
    return {
        "mean_activity": np.nanmean(traces, axis=0),
        "std_activity": np.nanstd(traces, axis=0),
        "median_activity": np.nanmedian(traces, axis=0),
        "activity_cv": np.nanstd(traces, axis=0)
        / (np.nanmean(traces, axis=0) + division_epsilon),
        "total_activity": np.nansum(traces, axis=0),
    }


@beartype
def _calculate_event_metrics(
    events: Optional[np.ndarray],
    include_correlations: bool = True,
    strong_threshold: float = STRONG_CORRELATION_THRESHOLD,
) -> Dict[str, Any]:
    """Calculate event-based metrics including correlations.

    Args
    ----
        events: Event data (timepoints x cells) or None
        include_correlations: Whether to compute event correlations
        strong_threshold: Threshold for strong correlations

    Returns
    -------
        Dictionary with event metrics including correlations if requested

    """
    if events is None:
        return {}

    try:
        # Calculate per-cell event statistics (matching trace statistics)
        event_rates = np.nanmean(events, axis=0)
        event_std = np.nanstd(events, axis=0)
        event_median = np.nanmedian(events, axis=0)

        # Calculate coefficient of variation for events (with epsilon for safety)
        event_cv = event_std / (event_rates + DIVISION_SAFETY_EPSILON)

        result = {
            "event_rates": event_rates,  # Per-cell mean
            "event_std": event_std,  # Per-cell std dev
            "event_median": event_median,  # Per-cell median
            "event_cv": event_cv,  # Per-cell coefficient of variation
            "mean_event_rate": np.nanmean(event_rates),  # Population mean
        }

        # Calculate event correlations if requested and we have enough cells
        if include_correlations and events.shape[1] > 1:
            try:
                # Convert events to float for correlation calculation
                # (measures.correlation_matrix requires floating point arrays)
                events_float = events.astype(np.float64)

                # Check for sparse event data (cells with no variance)
                event_std = np.std(events_float, axis=0)
                active_cells = np.sum(event_std > MIN_EVENT_RATE_THRESHOLD)
                total_cells = events.shape[1]

                if active_cells < 2:
                    logger.warning(
                        f"Only {active_cells} of {total_cells} cells have event activity. "
                        "Event correlations will be mostly undefined (NaN). "
                        "This is expected for sparse event data."
                    )

                # Use existing correlation computation logic
                # Note: Cells with no events will produce NaN correlations
                event_correlation_matrix = measures.correlation_matrix(
                    events_float, fill_diagonal=0.0
                )

                # Compute statistics using shared function
                event_corr_stats = _compute_correlation_statistics(
                    event_correlation_matrix, strong_threshold
                )

                # Add event correlation results with event_ prefix
                result["event_correlation_matrix"] = event_correlation_matrix
                result["event_mean_correlation"] = event_corr_stats[
                    "mean_correlation"
                ]
                result["event_std_correlation"] = event_corr_stats[
                    "std_correlation"
                ]
                result["event_median_correlation"] = event_corr_stats[
                    "median_correlation"
                ]
                result["event_max_correlation"] = event_corr_stats[
                    "max_correlation"
                ]
                result["event_min_correlation"] = event_corr_stats[
                    "min_correlation"
                ]
                result["event_mean_positive_correlation"] = event_corr_stats[
                    "mean_positive_correlation"
                ]
                result["event_mean_negative_correlation"] = event_corr_stats[
                    "mean_negative_correlation"
                ]
                result["event_positive_corr_fraction"] = event_corr_stats[
                    "positive_corr_fraction"
                ]
                result["event_negative_corr_fraction"] = event_corr_stats[
                    "negative_corr_fraction"
                ]
                result["event_strong_positive_corr_count"] = event_corr_stats[
                    "strong_positive_corr_count"
                ]
                result["event_strong_negative_corr_count"] = event_corr_stats[
                    "strong_negative_corr_count"
                ]
                result["event_correlation_range"] = event_corr_stats[
                    "correlation_range"
                ]

                # Store metadata about event sparsity for diagnostics
                result["event_active_cells"] = int(active_cells)
                result["event_sparse_data"] = active_cells < total_cells * 0.5

            except Exception as e:
                logger.warning(f"Event correlation calculation failed: {e}")
                # Continue with event rates even if correlation fails

        return result

    except Exception as e:
        logger.warning(f"Event analysis failed: {e}")
        return {}


class StateEpochResults:
    """Stores and manages results from state-epoch analysis."""

    def __init__(self):
        """Initialize empty results container."""
        self.combination_results = {}  # {(state, epoch): results_dict}

    def add_combination_results(
        self, state: str, epoch: str, results: Dict[str, Any]
    ) -> None:
        """Add results for a specific state-epoch combination.

        :Args
        ----
            state: Behavioral state name.
            epoch: Time epoch name.
            results: Dictionary containing analysis results.

        """
        key = (state, epoch)
        self.combination_results[key] = results
        logger.debug(f"Added results for {state}-{epoch}")

    def get_combination_results(
        self, state: str, epoch: str
    ) -> Optional[Dict[str, Any]]:
        """Get results for a specific state-epoch combination."""
        key = (state, epoch)
        return self.combination_results.get(key)

    def get_activity_vector(
        self, state: str, epoch: str
    ) -> Optional[np.ndarray]:
        """Return mean activity vector for a combination, if available.

        :Args
        ----
            state: State name
            epoch: Epoch name

        Returns
        -------
            Activity array or None if unavailable

        """
        results = self.get_combination_results(state, epoch)
        if not results:
            return None
        return results.get("mean_activity")

    def get_event_rates(self, state: str, epoch: str) -> Optional[np.ndarray]:
        """Return event rate vector for a combination, if available.

        :Args
        ----
            state: State name
            epoch: Epoch name

        Returns
        -------
            Event rates array or None if unavailable

        """
        results = self.get_combination_results(state, epoch)
        if not results:
            return None
        return results.get("event_rates")

    def get_correlation_stats(
        self, state: str, epoch: str
    ) -> Optional[Dict[str, Any]]:
        """Return correlation statistics and per-cell extrema for a combination.

        Computes summary statistics using the shared helper and returns
        per-cell max/min/mean as convenience, centralizing extraction here.

        :Args
        ----
            state: State name
            epoch: Epoch name

        Returns
        -------
            Dictionary with keys: 'mean_correlation', 'max_per_cell', 'min_per_cell',
            'mean_per_cell' or None if correlation matrix is unavailable

        """
        results = self.get_combination_results(state, epoch)
        if not results or "correlation_matrix" not in results:
            return None
        correlation_matrix = results.get("correlation_matrix")
        if correlation_matrix is None:
            return None

        stats = _compute_correlation_statistics(correlation_matrix)

        # Calculate per-cell statistics (excluding diagonal)
        n_cells = correlation_matrix.shape[0]
        max_per_cell = np.full(n_cells, np.nan)
        min_per_cell = np.full(n_cells, np.nan)
        mean_per_cell = np.full(n_cells, np.nan)

        for i in range(n_cells):
            # Get row excluding diagonal
            mask = np.ones(n_cells, dtype=bool)
            mask[i] = False
            off_diag = correlation_matrix[i, mask]
            valid = off_diag[~np.isnan(off_diag)]

            if len(valid) > 0:
                max_per_cell[i] = np.max(valid)
                min_per_cell[i] = np.min(valid)
                mean_per_cell[i] = np.mean(valid)

        return {
            "mean_correlation": stats["mean_correlation"],
            "max_per_cell": max_per_cell,
            "min_per_cell": min_per_cell,
            "mean_per_cell": mean_per_cell,
        }

    def get_correlation_matrix(
        self, state: str, epoch: str
    ) -> Optional[np.ndarray]:
        """Return raw correlation matrix for a combination, if available."""
        results = self.get_combination_results(state, epoch)
        if not results:
            return None
        return results.get("correlation_matrix")

    def get_baseline_data(
        self, baseline_state: str, baseline_epoch: str
    ) -> Dict[str, Any]:
        """Get baseline data for modulation analysis with validation.

        Args
        ----
            baseline_state: Name of baseline state
            baseline_epoch: Name of baseline epoch

        Returns
        -------
            Dictionary containing essential baseline data for modulation

        Raises
        ------
            IdeasError: If baseline data not found or invalid

        """
        baseline_results = self._validate_baseline_exists(
            baseline_state, baseline_epoch
        )
        self._validate_baseline_data_quality(
            baseline_results, baseline_state, baseline_epoch
        )
        return self._extract_baseline_data(baseline_results)

    def get_all_states(self) -> List[str]:
        """Get list of all states in results."""
        return list(set(state for state, _ in self.combination_results.keys()))

    def get_all_epochs(self) -> List[str]:
        """Get list of all epochs in results."""
        return list(set(epoch for _, epoch in self.combination_results.keys()))

    def get_all_combinations(self) -> List[Tuple[str, str]]:
        """Get list of all state-epoch combinations."""
        return list(self.combination_results.keys())

    def has_combination(self, state: str, epoch: str) -> bool:
        """Check if results exist for a state-epoch combination."""
        return (state, epoch) in self.combination_results

    def _validate_baseline_exists(
        self, baseline_state: str, baseline_epoch: str
    ) -> Dict[str, Any]:
        """Check that baseline combination exists and is valid.

        Args
        ----
            baseline_state: Name of baseline state
            baseline_epoch: Name of baseline epoch

        Returns
        -------
            Baseline results dictionary

        Raises
        ------
            IdeasError: If baseline is not available or failed

        """
        baseline_key = (baseline_state, baseline_epoch)

        # Check existence
        if baseline_key not in self.combination_results:
            available_combinations = list(self.combination_results.keys())
            raise IdeasError(
                f"Baseline combination '{baseline_state}'-'{baseline_epoch}' is not available. "
                f"Available: {available_combinations}"
            )

        baseline_results = self.combination_results[baseline_key]

        # Check for analysis failure
        if isinstance(baseline_results, dict) and "error" in baseline_results:
            raise IdeasError(
                f"Baseline '{baseline_state}'-'{baseline_epoch}' analysis failed: "
                f"{baseline_results['error']}"
            )

        return baseline_results

    def _validate_baseline_data_quality(
        self,
        baseline_results: Dict[str, Any],
        baseline_state: str,
        baseline_epoch: str,
    ) -> None:
        """Validate baseline has required data for modulation analysis.

        Args
        ----
            baseline_results: Baseline analysis results
            baseline_state: Name of baseline state for error messages
            baseline_epoch: Name of baseline epoch for error messages

        Raises
        ------
            IdeasError: If required data is missing

        """
        baseline_id = f"'{baseline_state}'-'{baseline_epoch}'"

        # Check required trace data
        if baseline_results.get("mean_activity") is None:
            raise IdeasError(f"Baseline {baseline_id} missing activity data")

        if baseline_results.get("traces") is None:
            raise IdeasError(f"Baseline {baseline_id} missing trace data")

        # Check event data availability (optional but recommended)
        events = baseline_results.get("events")
        event_rates = baseline_results.get("event_rates")

        if events is None and event_rates is None:
            logger.warning(
                f"Baseline {baseline_id} missing event data - "
                "event-based modulation analysis will be unavailable"
            )
        elif events is not None and event_rates is None:
            logger.warning(
                f"Baseline {baseline_id} has events but no event_rates computed - "
                "calculating event_rates from events"
            )
            # Calculate event rates if events are available but rates aren't
            try:
                baseline_results["event_rates"] = np.nanmean(events, axis=0)
            except Exception as e:
                logger.warning(
                    f"Failed to calculate event rates for baseline: {e}"
                )

        # Quality warnings for traces
        num_timepoints = baseline_results.get("num_timepoints", 0)
        if num_timepoints < MIN_TIMEPOINTS_FOR_RELIABLE_ANALYSIS:
            logger.warning(
                f"Baseline {baseline_id} has only {num_timepoints} timepoints - "
                "may affect modulation reliability"
            )

        mean_activity = baseline_results.get("mean_activity")
        if mean_activity is not None and np.allclose(
            mean_activity, 0, atol=MIN_ACTIVITY_THRESHOLD
        ):
            logger.warning(
                f"Baseline {baseline_id} has very low activity levels"
            )

        # Quality warnings for events
        if event_rates is not None:
            mean_event_rate = np.nanmean(event_rates)
            if np.allclose(mean_event_rate, 0, atol=MIN_EVENT_RATE_THRESHOLD):
                logger.warning(
                    f"Baseline {baseline_id} has very low event rates"
                )

    def _extract_baseline_data(
        self, baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract essential baseline data for modulation calculation.

        Args
        ----
            baseline_results: Full baseline analysis results

        Returns
        -------
            Dictionary with essential baseline data

        """
        return {
            "mean_activity": baseline_results["mean_activity"],
            "traces": baseline_results["traces"],
            "events": baseline_results.get("events"),
            "event_rates": baseline_results.get("event_rates"),
            "mean_event_rate": baseline_results.get("mean_event_rate"),
            "correlation_matrix": baseline_results.get("correlation_matrix"),
            "num_timepoints": baseline_results.get("num_timepoints", 0),
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all combinations."""
        summary = {
            "num_combinations": len(self.combination_results),
            "states": self.get_all_states(),
            "epochs": self.get_all_epochs(),
            "combinations_with_data": [],
            "combinations_with_trace_data": [],
            "combinations_with_event_data": [],
            "combinations_with_errors": [],
        }

        for state, epoch in self.combination_results.keys():
            results = self.combination_results[(state, epoch)]
            if isinstance(results, dict) and "error" in results:
                summary["combinations_with_errors"].append((state, epoch))
            else:
                # Check for different types of data
                has_traces = results.get("traces") is not None
                has_events = (
                    results.get("events") is not None
                    or results.get("event_rates") is not None
                )

                if has_traces or has_events:
                    summary["combinations_with_data"].append((state, epoch))

                if has_traces:
                    summary["combinations_with_trace_data"].append(
                        (state, epoch)
                    )

                if has_events:
                    summary["combinations_with_event_data"].append(
                        (state, epoch)
                    )

        return summary


@beartype
def analyze_state_epoch_combination(
    state_epoch_data: Optional[Dict[str, Any]],
    state: str,
    epoch: str,
    cell_info: Dict[str, Any],
    include_correlations: bool,
    include_population_activity: bool,
    include_event_analysis: bool,
    alpha: float,
    n_shuffle: int,
) -> Dict[str, Any]:
    """Analyze a single state-epoch combination.

    Args
    ----
        state_epoch_data: Data for this state-epoch combination
        state: State name
        epoch: Epoch name
        cell_info: Cell information dictionary
        include_correlations: Whether to compute correlations
        include_population_activity: Whether to compute population activity
        include_event_analysis: Whether to analyze events
        alpha: Significance level
        n_shuffle: Number of permutations for statistical tests

    Returns
    -------
        Dictionary containing analysis results

    """
    if state_epoch_data is None:
        return {"error": "No data available"}

    traces = state_epoch_data["traces"]
    events = state_epoch_data.get("events")

    # Additional safety checks for trace data
    if traces is None or traces.size == 0:
        return {"error": f"No trace data available for {state}-{epoch}"}

    if traces.shape[0] < MIN_TIMEPOINTS_FOR_RELIABLE_ANALYSIS:
        logger.warning(
            f"Very few timepoints ({traces.shape[0]}) for {state}-{epoch} - "
            "analysis may be unreliable"
        )

    results = {
        "state": state,
        "epoch": epoch,
        "num_timepoints": traces.shape[0],
        "num_cells": traces.shape[1],
        "traces": traces,
        "events": events,
        "alpha": alpha,
        "n_shuffle": n_shuffle,
    }

    # Calculate activity metrics
    results.update(_calculate_activity_metrics(traces))

    # Calculate population-level metrics if requested
    if include_population_activity:
        results["population_mean"] = np.nanmean(results["mean_activity"])
        results["population_std"] = np.nanstd(results["mean_activity"])

    # Calculate correlations if requested
    if include_correlations:
        correlation_results = _calculate_correlation_metrics(traces)
        if correlation_results.get("correlation_matrix") is None:
            # Provide detailed context for debugging if this was unexpected
            if (
                traces.shape[1] > 1
            ):  # Only warn if we expected correlations to work
                n_timepoints, n_cells = traces.shape[0], traces.shape[1]
                logger.warning(
                    f"Correlation calculation failed for {state}-{epoch}. "
                    f"traces shape=({n_timepoints}, {n_cells})"
                )
            # Don't add correlation_matrix if it's None (preserve original behavior)
            correlation_results = {
                k: v
                for k, v in correlation_results.items()
                if not (k == "correlation_matrix" and v is None)
            }
        results.update(correlation_results)

    # Analyze events if requested and available
    if include_event_analysis:
        event_results = _calculate_event_metrics(
            events, include_correlations=include_correlations
        )
        if not event_results and events is not None:
            logger.warning(f"Event analysis failed for {state}-{epoch}")
        results.update(event_results)

    return results


@beartype
def calculate_baseline_modulation(
    results: StateEpochResults,
    baseline_state: str,
    baseline_epoch: str,
    cell_info: Dict[str, Any],
    alpha: float,
    n_shuffle: int,
) -> Dict[str, Any]:
    """Calculate activity and event modulation relative to baseline for all combinations.

    This function calculates both activity-based and event-based modulation
    when the respective data types are available in the results.

    Args
    ----
        results: StateEpochResults object containing all combination results.
        baseline_state: Name of baseline state.
        baseline_epoch: Name of baseline epoch.
        cell_info: Cell information dictionary.
        alpha: Significance level for statistical tests.
        n_shuffle: Number of permutations for statistical tests.

    Returns
    -------
        Dictionary containing modulation results with:
            - activity_modulation: Activity-based modulation data
            - event_modulation: Event-based modulation data (if available)
            - baseline data for both activity and events

    """
    try:
        baseline_data = results.get_baseline_data(
            baseline_state, baseline_epoch
        )
    except IdeasError as e:
        logger.error(f"Baseline data error: {e}")
        return {"error": str(e)}

    baseline_mean = baseline_data["mean_activity"]

    # Check if baseline has event data for integrated event modulation
    baseline_event_rates = baseline_data.get("event_rates")
    baseline_events = baseline_data.get("events")
    has_event_data = (
        baseline_event_rates is not None or baseline_events is not None
    )

    # Calculate baseline event rates if needed
    if (
        has_event_data
        and baseline_event_rates is None
        and baseline_events is not None
    ):
        try:
            baseline_event_rates = np.nanmean(baseline_events, axis=0)
        except Exception as e:
            logger.warning(f"Failed to calculate baseline event rates: {e}")
            has_event_data = False

    modulation_results = {
        "baseline_state": baseline_state,
        "baseline_epoch": baseline_epoch,
        "baseline_mean_activity": baseline_mean,
        "cell_names": cell_info.get("cell_names", []),
        "activity_modulation": {},  # Activity-based modulation
        "significant_cells": {},
        "modulation_summary": {},
    }

    # Add event modulation structure if event data is available
    if has_event_data:
        modulation_results["event_modulation"] = {}
        modulation_results["baseline_event_rates"] = baseline_event_rates
        modulation_results["baseline_mean_event_rate"] = np.nanmean(
            baseline_event_rates
        )
        logger.info(
            f"Event modulation analysis enabled for baseline {baseline_state}-{baseline_epoch}"
        )

    # Calculate modulation for each combination
    for state, epoch in results.get_all_combinations():
        if state == baseline_state and epoch == baseline_epoch:
            continue  # Skip baseline itself

        combination_results = results.get_combination_results(state, epoch)
        if (
            combination_results is None
            or combination_results.get("mean_activity") is None
        ):
            continue

        # Use consistent modulation calculation from population_activity.py
        baseline_traces = baseline_data.get("traces")
        test_traces = combination_results.get("traces")

        # Require traces for proper modulation analysis - consistent with population_activity.py
        if baseline_traces is None or test_traces is None:
            logger.warning(
                f"Skipping modulation for {state}-{epoch}: missing trace data. "
                f"Baseline traces available: {baseline_traces is not None}, "
                f"Test traces available: {test_traces is not None}"
            )
            continue

        # Use the same approach as population_activity.py for two-state comparison
        # Create boolean masks for the time periods
        baseline_mask = np.ones(baseline_traces.shape[0], dtype=bool)
        test_mask = np.ones(test_traces.shape[0], dtype=bool)

        # Combine traces for modulation calculation (proper approach)
        all_traces = np.vstack([baseline_traces, test_traces])
        combined_mask_baseline = np.concatenate(
            [baseline_mask, np.zeros(test_traces.shape[0], dtype=bool)]
        )
        combined_mask_test = np.concatenate(
            [np.zeros(baseline_traces.shape[0], dtype=bool), test_mask]
        )

        try:
            # Use population_activity.py functions for robust modulation calculation
            modulation_results_detailed = find_two_state_modulated_neurons(
                traces=all_traces,
                when1=combined_mask_test,  # Test state periods
                when2=combined_mask_baseline,  # Baseline state periods
                alpha=alpha,
                n_shuffle=n_shuffle,
            )

            modulation_index = modulation_results_detailed["modulation_scores"]
            p_values = modulation_results_detailed["p_val"]
            # Note: find_two_state_modulated_neurons already applies multiple comparisons correction
            p_values_corrected = p_values
            # Store the correctly-classified neurons from permutation test
            # (these use proper alpha/2 threshold for two one-tailed comparisons)
            up_modulated_neurons = modulation_results_detailed[
                "up_modulated_neurons"
            ]
            down_modulated_neurons = modulation_results_detailed[
                "down_modulated_neurons"
            ]

        except Exception as e:
            # Follow population_activity.py error handling pattern
            exception_type = type(e).__name__
            logger.error(
                f"Error calculating modulation for {state}-{epoch} vs baseline "
                f"{baseline_state}-{baseline_epoch}: {exception_type}: {str(e)}. "
                f"Trace shape: {all_traces.shape}, "
                f"Test mask sum: {np.sum(combined_mask_test)}, "
                f"Baseline mask sum: {np.sum(combined_mask_baseline)}."
            )
            raise IdeasError(
                f"Modulation calculation failed for {state}-{epoch}: {str(e)}"
            ) from e

        combination_key = (state, epoch)  # Use tuple key to match reference.py
        modulation_results["activity_modulation"][combination_key] = {
            "modulation_index": modulation_index,
            "p_values": p_values,
            "p_values_corrected": p_values_corrected,
            "up_modulated_neurons": up_modulated_neurons,
            "down_modulated_neurons": down_modulated_neurons,
            "significant": (
                (p_values_corrected < alpha)
                if p_values_corrected is not None
                else np.zeros(len(modulation_index), dtype=bool)
            ),
            "n_significant": (
                np.sum(p_values_corrected < alpha)
                if p_values_corrected is not None
                else 0
            ),
            "fraction_significant": (
                np.mean(p_values_corrected < alpha)
                if p_values_corrected is not None
                else 0
            ),
        }

        # Calculate event modulation if event data is available
        if has_event_data:
            test_event_rates = combination_results.get("event_rates")
            test_events = combination_results.get("events")

            # Calculate event rates if not available
            if test_event_rates is None and test_events is not None:
                try:
                    test_event_rates = np.nanmean(test_events, axis=0)
                    combination_results["event_rates"] = test_event_rates
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate event rates for {state}-{epoch}: {e}"
                    )
                    test_event_rates = None

            if test_event_rates is not None:
                # Calculate event modulation using the same robust approach as activity
                try:
                    event_modulation_scores = _calculate_modulation_scores(
                        test_event_rates, baseline_event_rates
                    )

                    # Use simple statistical approach for event modulation (conservative)
                    # For now, use a simplified approach - could be enhanced later
                    event_p_values = np.ones(
                        len(event_modulation_scores)
                    )  # Conservative p-values

                    # Apply significance threshold
                    event_significant = (
                        np.abs(event_modulation_scores) > 0.1
                    )  # Simple threshold

                    modulation_results["event_modulation"][combination_key] = {
                        "modulation_index": event_modulation_scores,
                        "p_values": event_p_values,
                        "p_values_corrected": event_p_values,  # No correction for now
                        "significant": event_significant,
                        "n_significant": np.sum(event_significant),
                        "fraction_significant": np.mean(event_significant),
                    }

                except Exception as e:
                    logger.warning(
                        f"Event modulation calculation failed for {state}-{epoch}: {e}"
                    )

    return modulation_results


@beartype
def _extract_activity_data(
    mod_data: Dict[str, Any],
    data_type: str,
    activity_key: str,
    cell_count: int,
) -> np.ndarray:
    """Extract activity data from modulation data.

    Args
    ----
        mod_data: Modulation data dictionary
        data_type: Type of data ("activity" or "event")
        activity_key: Key for activity data
        cell_count: Number of cells for fallback

    Returns
    -------
        Activity data array

    """
    if data_type == "event":
        # For events, try to get event rates or calculate from events
        activity_data = np.array(mod_data.get("event_rates", []))
        if len(activity_data) == 0:
            # Try to get from mean_event_rate
            mean_rate = mod_data.get("mean_event_rate")
            if mean_rate is not None:
                activity_data = np.full(cell_count, mean_rate)
            else:
                activity_data = np.array([])
    else:
        # For traces, get mean activity
        activity_data = np.array(mod_data.get("mean_activity", []))

    return activity_data


def extract_common_modulation_data(
    modulation_source: Dict[str, Any],
    cell_count: int,
    alpha: float = DEFAULT_ALPHA,
    data_type: str = "activity",
) -> Dict[str, Dict[str, Any]]:
    """Extract and process common modulation data structure.

    Args
    ----
        modulation_source: Source modulation data
        cell_count: Number of cells
        alpha: Significance threshold for modulation detection
        data_type: Type of data being processed ("activity" or "event")

    Returns
    -------
        Processed modulation data dictionary

    """
    modulation_data = {}
    # Get appropriate keys based on data type
    if data_type == "event":
        modulation_key, activity_key = "event_modulation", "event_rates"
    else:
        modulation_key, activity_key = "activity_modulation", "mean_activity"

    # Get the appropriate modulation source
    if modulation_key in modulation_source:
        source_data = modulation_source[modulation_key]
    else:
        # Fallback to direct source if specific key not found
        source_data = modulation_source

    for state_epoch, mod_data in source_data.items():
        try:
            modulation_index = np.array(mod_data.get("modulation_index", []))
            p_values = np.array(mod_data.get("p_values", []))
        except Exception as e:
            logger.warning(
                f"Error converting {data_type} modulation data to arrays for {state_epoch}: {e}"
            )
            continue

        # Extract activity data
        activity_data = _extract_activity_data(
            mod_data, data_type, activity_key, cell_count
        )

        # Convert tuple key to string key for compatibility
        if isinstance(state_epoch, tuple):
            state_epoch_str = f"{state_epoch[0]}_{state_epoch[1]}"
        else:
            state_epoch_str = str(state_epoch)

        # Find modulated neurons
        # Use pre-classified neurons from permutation test if available
        # (they use correct alpha/2 logic for two one-tailed comparisons)
        try:
            if (
                "up_modulated_neurons" in mod_data
                and "down_modulated_neurons" in mod_data
            ):
                # Use correctly pre-classified neurons from permutation test
                up_neurons = mod_data["up_modulated_neurons"]
                down_neurons = mod_data["down_modulated_neurons"]
                if not isinstance(up_neurons, np.ndarray):
                    up_neurons = np.array(up_neurons)
                if not isinstance(down_neurons, np.ndarray):
                    down_neurons = np.array(down_neurons)
            elif len(modulation_index) == 0 or len(p_values) == 0:
                up_neurons, down_neurons = np.array([]), np.array([])
            else:
                # Fallback: Ensure both arrays are 1D
                modulation_index = np.atleast_1d(modulation_index)
                p_values = np.atleast_1d(p_values)

                # WARNING: This fallback uses minimum p-value which is conservative
                # but not statistically ideal for directional classification
                # Ideally all paths should use permutation test results
                up_mask = (modulation_index > 0) & (p_values < alpha / 2)
                down_mask = (modulation_index < 0) & (p_values < alpha / 2)

                up_neurons = np.where(up_mask)[0]
                down_neurons = np.where(down_mask)[0]
        except Exception as e:
            logger.warning(
                f"Error processing {data_type} modulation data for {state_epoch}: {e}"
            )
            up_neurons, down_neurons = np.array([]), np.array([])

        modulation_data[state_epoch_str] = {
            activity_key: activity_data,
            "modulation_scores": modulation_index,
            "p_val": p_values,
            "up_modulated_neurons": up_neurons,
            "down_modulated_neurons": down_neurons,
        }

    return modulation_data


def _get_safe_cell_count(cell_info: Dict[str, Any]) -> int:
    """Get safe cell count with fallback logic.

    Args
    ----
        cell_info: Cell information dictionary

    Returns
    -------
        Number of valid cells

    Raises
    ------
        IdeasError: If no valid cells found

    """
    num_accepted_cells = cell_info.get("num_accepted_cells", 0)
    num_undecided_cells = cell_info.get("num_undecided_cells", 0)

    if num_accepted_cells > 0:
        return num_accepted_cells
    elif num_undecided_cells > 0:
        return num_undecided_cells
    else:
        raise IdeasError(
            "No valid cells found. Both accepted and undecided cell counts are zero or missing."
        )


def _extract_baseline_array(
    results: "StateEpochResults",
    baseline_state: str,
    baseline_epoch: str,
    data_key: str,
    data_type_name: str,
) -> Optional[np.ndarray]:
    """Extract specific data array from baseline results.

    Args
    ----
        results: State-epoch analysis results
        baseline_state: Baseline state name
        baseline_epoch: Baseline epoch name
        data_key: Key to extract from baseline results (e.g., "mean_activity", "event_rates")
        data_type_name: Human-readable name for logging (e.g., "activity", "event rates")

    Returns
    -------
        Baseline data array or None if not available

    """
    baseline_results = results.get_combination_results(
        baseline_state, baseline_epoch
    )
    if baseline_results is None:
        logger.warning(
            f"No baseline data found for {baseline_state}-{baseline_epoch}"
        )
        return None

    baseline_data = baseline_results.get(data_key)
    if baseline_data is None:
        logger.warning(f"No baseline {data_type_name} data available")
        return None

    return baseline_data


def _calculate_modulation_scores(
    test_data: np.ndarray, baseline_data: np.ndarray
) -> np.ndarray:
    """Calculate bounded modulation scores.

    Args
    ----
        test_data: Test condition data
        baseline_data: Baseline condition data

    Returns
    -------
        Bounded modulation scores [-1, 1]

    """
    # Calculate modulation index (bounded -1 to 1): (test - baseline) / (test + baseline)
    modulation_scores = (test_data - baseline_data) / (
        test_data + baseline_data + DIVISION_SAFETY_EPSILON
    )

    # Ensure modulation scores are bounded between -1 and 1
    return np.clip(modulation_scores, -1.0, 1.0)


def _get_p_values_with_fallback(
    modulation_results: Dict[str, Any],
    state: str,
    epoch: str,
    modulation_scores: np.ndarray,
) -> np.ndarray:
    """Get p-values with conservative fallback.

    Args
    ----
        modulation_results: Modulation analysis results
        state: State name
        epoch: Epoch name
        modulation_scores: Modulation scores for fallback shape

    Returns
    -------
        P-values array

    """
    modulation_key = (state, epoch)
    modulation_data = modulation_results.get("activity_modulation", {}).get(
        modulation_key, {}
    )
    p_values = modulation_data.get("p_values")

    if p_values is None:
        # If no statistical test results available, mark as non-significant
        p_values = np.ones_like(modulation_scores)
        logger.warning(
            f"No p-values available for {state}-{epoch}, using non-significant values"
        )

    return p_values


def _create_modulation_summary(
    modulation_scores: np.ndarray, p_values: np.ndarray, alpha: float
) -> Dict[str, np.ndarray]:
    """Create modulation summary with significant neurons.

    Args
    ----
        modulation_scores: Modulation scores
        p_values: P-values (typically minimum p-value from two-tailed test)
        alpha: Significance threshold

    Returns
    -------
        Dictionary with modulation summary

    Notes
    -----
        This is a fallback function for when full permutation test results
        are not available. It uses alpha/2 for directional classification
        to maintain consistency with two one-tailed comparison logic.
        However, when permutation test results with correctly pre-classified
        neurons are available, those should be used instead.

    """
    return {
        "modulation_scores": modulation_scores,
        "p_val": p_values,
        "up_modulated_neurons": np.where(
            (modulation_scores > 0) & (p_values < alpha / 2)
        )[0],
        "down_modulated_neurons": np.where(
            (modulation_scores < 0) & (p_values < alpha / 2)
        )[0],
    }


def _prepare_modulation_data(
    results: "StateEpochResults",
    modulation_results: Dict[str, Any],
    cell_info: Dict[str, Any],
    baseline_state: str,
    baseline_epoch: str,
    alpha: float,
    data_key: str,
    data_type_name: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Prepare modulation data for plotting using common patterns.

    Args
    ----
        results: State-epoch analysis results
        modulation_results: Modulation analysis results
        cell_info: Cell information dictionary
        baseline_state: Baseline state name
        baseline_epoch: Baseline epoch name
        alpha: Significance threshold for modulation detection
        data_key: Key to extract data (e.g., "mean_activity", "event_rates")
        data_type_name: Human-readable name for logging

    Returns
    -------
        Modulation data dictionary or None if baseline not available

    """
    # Get safe cell count
    num_cells = _get_safe_cell_count(cell_info)

    # Get baseline data
    baseline_data = _extract_baseline_array(
        results, baseline_state, baseline_epoch, data_key, data_type_name
    )
    if baseline_data is None:
        return None

    # Create modulation data structure
    modulation_data = {}

    # Process each state-epoch combination
    for state, epoch in results.get_all_combinations():
        combination_results = results.get_combination_results(state, epoch)
        if combination_results is None:
            continue

        combination_label = f"{state}_{epoch}"
        test_data = combination_results.get(data_key, np.zeros(num_cells))
        modulation_data[combination_label] = {data_key: test_data}

        # Add modulation data for non-baseline combinations
        if f"{state}_{epoch}" != f"{baseline_state}_{baseline_epoch}":
            # Calculate modulation scores
            modulation_scores = _calculate_modulation_scores(
                test_data, baseline_data
            )

            # Get p-values with fallback
            p_values = _get_p_values_with_fallback(
                modulation_results, state, epoch, modulation_scores
            )

            # Add modulation summary
            modulation_summary = _create_modulation_summary(
                modulation_scores, p_values, alpha
            )
            modulation_data[combination_label].update(modulation_summary)

    return modulation_data


def prepare_event_modulation_data(
    results: "StateEpochResults",
    modulation_results: Dict[str, Any],
    cell_info: Dict[str, Any],
    baseline_state: str,
    baseline_epoch: str,
    alpha: float = DEFAULT_ALPHA,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Prepare event modulation data structure for plotting.

    Args
    ----
        results: State-epoch analysis results
        modulation_results: Modulation analysis results
        cell_info: Cell information dictionary
        baseline_state: Baseline state name
        baseline_epoch: Baseline epoch name
        alpha: Significance threshold for modulation detection

    Returns
    -------
        Event modulation data dictionary or None if baseline not available

    """
    return _prepare_modulation_data(
        results=results,
        modulation_results=modulation_results,
        cell_info=cell_info,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        alpha=alpha,
        data_key="event_rates",
        data_type_name="event rates",
    )
