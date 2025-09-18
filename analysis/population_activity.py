"""Define a tool that analyzes modulation of neural circuit
activity in different states.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import List, Optional
from ideas import io
from ideas.exceptions import IdeasError
from ideas.types import NumpyFloatArray
from ideas.utils import _set_up_logger

from utils.plots import (
    _plot_population_average,
    _plot_raster,
    _plot_state_time,
    _plot_traces,
    plot_modulated_neuron_footprints,
)
from utils.utils import (
    Comp,
    Rescale,
    _check_states_valid,
    _fractional_change_states,
    _get_cellset_data,
    _norm_2D_array,
    _standardize_2D_array,
    _standardize_baseline,
    _status_to_cell_names,
)
from utils.validation import _validate_correlation_params

_set_up_logger()
logger = logging.getLogger()


# define output file locs and constants
MODULATION_CSV_NAME = "trace_population_data.csv"
EVENT_MODULATION_CSV_NAME = "event_population_data.csv"
MINIMUM_STATE_LENGTH = 10

trace_preview = "trace_preview.svg"
event_preview = "event_preview.svg"
activity_average_preview = "activity_average_preview.svg"
event_average_preview = "event_average_preview.svg"
modulation_preview = "modulation_preview.svg"
modulation_histogram_preview = "modulation_histogram_preview.svg"
event_modulation_preview = "event_modulation_preview.svg"
event_modulation_histogram_preview = "event_modulation_histogram_preview.svg"

MIN_ACTIVITY_THRESHOLD = 1e-6


@beartype
def _apply_rescaling(
    traces: NumpyFloatArray,
    behavior: pd.DataFrame,
    column_name: str,
    rescale: str,
    baseline_state: Optional[str] = None,
) -> NumpyFloatArray:
    """Apply the selected rescaling method to the traces.

    :param traces: Neural activity traces
    :param behavior: Behavioral data
    :param column_name: Column in behavior containing state information
    :param rescale: Rescaling method to apply
    :param baseline_state: State to use as baseline (for methods that require it)
    :return: Rescaled traces
    """
    if rescale == Rescale.NONE.value:
        return traces
    elif rescale == Rescale.NORMALIZE.value:
        return _norm_2D_array(traces)
    elif rescale == Rescale.STANDARDIZE.value:
        return _standardize_2D_array(traces)
    elif rescale == Rescale.FRACTIONAL_CHANGE.value:
        if baseline_state is None:
            raise IdeasError(
                "Baseline state must be specified for fractional change rescaling"
            )
        return _fractional_change_states(
            traces, behavior, column_name, baseline_state
        )
    elif rescale == Rescale.STANDARDIZE_BASELINE.value:
        if baseline_state is None:
            raise IdeasError(
                "Baseline state must be specified for standardize baseline rescaling"
            )
        return _standardize_baseline(
            traces, behavior, column_name, baseline_state
        )
    else:
        raise IdeasError(f"Rescale method {rescale} is not supported")


@beartype
def _make_modulation_data(
    *,
    traces: NumpyFloatArray,
    behavior: pd.DataFrame,
    states: list,
    column_name: str,
    n_shuffle: int = 1000,
    alpha: float = 0.05,
    method: str = Comp.NOT_STATE.value,
    baseline_state: Optional[str] = None,
    rescale: str = Rescale.NONE.value,
) -> dict:
    """Calculate modulation data based on the specified method.

    Modulation data quantifies how neural activity differs between behavioral states,
    including statistical significance of these differences.

    :param traces: Neural activity traces
    :param behavior: Behavioral data
    :param states: List of state names to analyze
    :param column_name: Column in behavior containing state information
    :param n_shuffle: Number of shuffles for permutation testing
    :param alpha: Significance threshold
    :param method: Comparison method
    :param baseline_state: State to use as baseline (for methods that require it)
    :param rescale: Method to rescale the data
    :return: Dictionary with structure {state_name: {metrics}} where metrics include:
             - mean_activity: Average activity during that state
             - modulation_scores: Modulation values for each cell
             - p_val: Statistical significance for each cell
             - up_modulated_neurons: Indices of significantly upmodulated neurons
             - down_modulated_neurons: Indices of significantly downmodulated neurons
             For PAIRWISE comparisons, the structure is nested with {state1: {state2: {metrics}}}
    """
    # Apply the appropriate rescaling method
    scaled_traces = _apply_rescaling(
        traces, behavior, column_name, rescale, baseline_state
    )

    # Initialize the results dictionary
    data = {}

    # Calculate mean activity for all states first (common to all methods)
    for state in states:
        data[state] = {
            "mean_activity": np.nanmean(
                scaled_traces[behavior[column_name] == state, :], axis=0
            )
        }

    # Apply the appropriate comparison method
    if method == Comp.NOT_STATE.value:
        # Compare each state to not-that-state
        for state in states:
            state_mask = (behavior[column_name] == state).values
            modulation_results = find_modulated_neurons(
                traces, state_mask, n_shuffle=n_shuffle, alpha=alpha
            )
            # Update the existing state dictionary with modulation results
            data[state].update(modulation_results)

    elif method == Comp.PAIRWISE.value:
        # Compare each state to every other state
        for s1_idx in range(len(states)):
            for s2_idx in range(s1_idx + 1, len(states)):
                state1, state2 = states[s1_idx], states[s2_idx]

                state1_mask = (behavior[column_name] == state1).values
                state2_mask = (behavior[column_name] == state2).values

                modulation_results = find_two_state_modulated_neurons(
                    traces,
                    state1_mask,
                    state2_mask,
                    n_shuffle=n_shuffle,
                    alpha=alpha,
                )

                # Store comparison results in the first state's dictionary
                data[state1][state2] = modulation_results

    elif method == Comp.BASELINE.value:
        # Compare each state to the baseline state
        if baseline_state is None:
            raise IdeasError(
                "Baseline state must be specified when using 'state vs baseline' method."
            )
        if baseline_state not in states:
            raise IdeasError(
                f"Baseline state '{baseline_state}' must be one of the specified states: {states}"
            )

        baseline_mask = (behavior[column_name] == baseline_state).values

        # Compare each non-baseline state to the baseline
        for state in states:
            if state != baseline_state:
                state_mask = (behavior[column_name] == state).values

                try:
                    modulation_results = find_two_state_modulated_neurons(
                        traces,
                        state_mask,
                        baseline_mask,
                        n_shuffle=n_shuffle,
                        alpha=alpha,
                    )
                    # Update the existing state dictionary with modulation results
                    data[state].update(modulation_results)
                except Exception as e:
                    exception_type = type(e).__name__
                    logger.warning(
                        f"Error calculating modulation for state '{state}'"
                        f" vs baseline '{baseline_state}': "
                        f"{exception_type}: {str(e)}. Trace shape: {traces.shape}, "
                        f"State mask sum: {np.sum(state_mask)}, Baseline mask sum:"
                        f" {np.sum(baseline_mask)}. "
                        "Setting modulation scores to empty arrays."
                    )
                    # Initialize with empty arrays to prevent downstream errors
                    num_cells = traces.shape[1]
                    data[state]["modulation_scores"] = np.zeros(num_cells)
                    data[state]["p_val"] = np.ones(num_cells)
                    data[state]["modulation"] = np.zeros(num_cells)
                    data[state]["up_modulated_neurons"] = np.array(
                        [], dtype=int
                    )
                    data[state]["down_modulated_neurons"] = np.array(
                        [], dtype=int
                    )

    elif method == Comp.NOT_DEFINED.value:
        # Find frames that don't belong to any defined state
        not_state_mask = np.ones(len(behavior), dtype=bool)
        for state in states:
            not_state_mask = not_state_mask & (behavior[column_name] != state)

        # Convert to numpy array to avoid pandas Series type error
        not_state_mask_array = (
            not_state_mask.values
            if hasattr(not_state_mask, "values")
            else np.array(not_state_mask)
        )

        # Check if there's enough data for the "not defined" state
        if np.sum(not_state_mask_array) < MINIMUM_STATE_LENGTH:
            raise IdeasError(
                "Not enough data to create a 'not_defined' state."
                f"In this session, there are fewer than {MINIMUM_STATE_LENGTH}"
                " frames where the animal is not in any state."
                " We cannot accurately compare activity between states."
            )

        # Compare each state to the "not defined" state
        for state in states:
            state_mask = (behavior[column_name] == state).values

            try:
                modulation_results = find_two_state_modulated_neurons(
                    traces,
                    state_mask,
                    not_state_mask_array,
                    n_shuffle=n_shuffle,
                    alpha=alpha,
                )
                data[state].update(modulation_results)
            except Exception as e:
                logger.warning(
                    f"Error calculating modulation for state '{state}' vs not defined: {str(e)}"
                    " Setting modulation scores to empty arrays."
                )
                # Initialize with empty arrays to prevent downstream errors
                num_cells = traces.shape[1]
                data[state]["modulation_scores"] = np.zeros(num_cells)
                data[state]["p_val"] = np.ones(num_cells)
                data[state]["modulation"] = np.zeros(num_cells)
                data[state]["up_modulated_neurons"] = np.array([], dtype=int)
                data[state]["down_modulated_neurons"] = np.array([], dtype=int)
    else:
        raise IdeasError(f"Method {method} is not supported")

    return data


@beartype
def _calculate_modulation_scores(
    traces: np.ndarray, mask1: np.ndarray, mask2: np.ndarray
) -> np.ndarray:
    """Calculate modulation scores between two states.

    Improved to handle edge cases with near-zero denominators.

    :param traces: neural activity of individual cells (2D array <num_timepoints x num_cells>)
    :param mask1: boolean array set to True for time indices of first state
    :param mask2: boolean array set to True for time indices of second state

    :return: array of modulation scores (1D array <num_cells>)
    """
    # Check if there are any frames in either state
    if np.sum(mask1) == 0 or np.sum(mask2) == 0:
        logger.warning(
            "One of the state arrays has no True values. This would lead to NaN modulation scores."
            " Returning zero modulation."
        )
        return np.zeros(traces.shape[1])

    # Create a copy to avoid modifying the input data
    traces_copy = np.copy(traces)

    # Remove minimum for traces to ensure values are positive
    traces_copy -= np.nanmin(traces_copy, axis=0)

    # Calculate mean activity for each state
    x = np.nanmean(traces_copy[mask1, :], axis=0)
    y = np.nanmean(traces_copy[mask2, :], axis=0)

    # Calculate modulation, handling edge cases
    denominator = x + y

    # Avoid division by zero or very small values
    # Using a threshold instead of exact zero check
    denominator_is_small = denominator < MIN_ACTIVITY_THRESHOLD

    # Initialize modulation array
    mod = np.zeros_like(denominator)

    # Calculate modulation only where denominator is above threshold
    valid_indices = ~denominator_is_small
    if np.any(valid_indices):
        mod[valid_indices] = (
            x[valid_indices] - y[valid_indices]
        ) / denominator[valid_indices]

    # For very small denominators, check if numerator is also small
    small_denom_indices = denominator_is_small
    if np.any(small_denom_indices):
        # For small denominators, check if one state has significantly higher activity
        numerator = x[small_denom_indices] - y[small_denom_indices]
        # If numerator is significantly different from zero, assign maximum modulation
        numerator_significant = np.abs(numerator) > MIN_ACTIVITY_THRESHOLD
        if np.any(numerator_significant):
            mod[small_denom_indices] = np.sign(numerator) * 1.0
            logger.warning(
                "Assigned maximum modulation for "
                f"{np.sum(numerator_significant)} cells with small denominators"
            )

    # Handle NaN and Inf values
    nan_inf_mask = ~np.isfinite(mod)
    nan_inf_count = np.sum(nan_inf_mask)
    if nan_inf_count > 0:
        logger.warning(
            f"{nan_inf_count} NaN/Inf values detected in modulation calculation."
            " Setting to 0. This may indicate cells with no activity in both states."
        )

    # Replace NaN and Inf values with 0
    mod[nan_inf_mask] = 0

    # Ensure modulation scores are bounded between -1 and 1
    mod = np.clip(mod, -1.0, 1.0)

    return mod


@beartype
def compute_two_state_modulation(
    traces: np.ndarray,
    when1: np.ndarray,
    when2: np.ndarray,
) -> np.ndarray:
    """Compute modulation value for neural activity traces between two states.

    :param traces: neural activity of individual cells
    (2D array <num_timepoints x num_cells>)
    :param when1: boolean array set to True for time indices of first state
    :param when2: boolean array set to True for time indices of second state

    :return: array of modulation scores (1D array <num_cells>)
    """
    return _calculate_modulation_scores(traces, when1, when2)


@beartype
def compute_modulation(
    traces: NumpyFloatArray,
    when,
) -> NumpyFloatArray:
    """Compute modulation value for neural activity traces.

    :param traces: neural activity of individual cells (2D array
        <num_timepoints x num_cells>)
    :param when: boolean array set to True for time indices when a
        behavior occured, False otherwise(1D array <num_timepoints>)
    :return: array of modulation scores (1D array <num_cells>)

    Modulation calculation steps:
    1. Subtract the minimum activity for each cell (column) to ensure non-negative values
    2. Calculate mean activity during the behavior ('when' is True)
    3. Calculate mean activity outside the behavior ('when' is False)
    4. Compute modulation as: (mean_when - mean_not_when) / (mean_when + mean_not_when)
    """
    return _calculate_modulation_scores(traces, when, ~when)


@beartype
def population_activity(
    *,
    cell_set_files: List[str],
    event_set_files: Optional[List[str]] = None,
    annotations_file: List[str],
    concatenate: bool = True,
    trace_scale_method: str = Rescale.NONE.value,
    event_scale_method: str = Rescale.NONE.value,
    column_name: str = "state",
    state_names: str,
    state_colors: str,
    method: str = Comp.NOT_STATE.value,
    baseline_state: Optional[str] = None,
    modulation_colors: str = "tab:red,tab:blue",
    n_shuffle: int = 1000,
    alpha: float = 0.05,
) -> None:
    """Calculate population activity during behavioral states.

    This function analyzes neural population activity across different behavioral states
    and generates visualizations of state-dependent modulation.

    :param cell_set_files: List of paths to isxd cell set files
    :param event_set_files: Optional list of paths to event set files
    :param annotations_file: Path to an annotations file (in IDEAS format)
    :param concatenate: Whether to concatenate multiple cell sets
    :param trace_scale_method: Method to scale trace data ("none", "normalize", "standardize", etc.)
    :param event_scale_method: Method to scale event data
    :param column_name: Column name in annotations file containing state information
    :param state_names: Comma-separated list of state names to analyze
    :param state_colors: Comma-separated list of colors for each state
    :param method: Comparison method (e.g., NOT_STATE, PAIRWISE, BASELINE)
    :param baseline_state: Name of state to use as baseline (required for some methods)
    :param modulation_colors: Comma-separated list of colors for up/down modulated cells
    :param n_shuffle: Number of shuffles for statistical significance testing
    :param alpha: Significance threshold for modulation
    """
    # unpack, because we always have exactly 1 file
    annotations_file = annotations_file[0]

    logger.info("State-based population analysis started")

    state_names = state_names.split(",")
    state_names = [state.strip() for state in state_names]

    state_colors = state_colors.split(",")
    state_colors = [color.strip() for color in state_colors]

    # Validate that state_names and state_colors have the same length
    if len(state_names) != len(state_colors):
        raise IdeasError(
            f"Number of state names ({len(state_names)}) "
            f"must match number of state colors ({len(state_colors)})"
        )

    # Add validation for empty state_names
    if not any(state for state in state_names):
        raise IdeasError(
            "No valid state names provided. Please specify at least one state name."
        )

    # Identify baseline state for comparison if method is state vs baseline
    if method == Comp.BASELINE.value and baseline_state is not None:
        try:
            # Verify baseline state exists in state_names
            state_names.index(baseline_state)
            logger.info(
                f"Using '{baseline_state}' as baseline state for modulation comparison"
            )
        except ValueError as err:
            raise IdeasError(
                f"Baseline state '{baseline_state}' not found in state_names {state_names}"
            ) from err

    # Log what states we'll analyze
    logger.info(f"Will analyze the following states: {state_names}")

    _validate_correlation_params(
        method=method,
        state_names=state_names,
        baseline_state=baseline_state,
        trace_scale_method=trace_scale_method,
        event_scale_method=event_scale_method,
    )

    # get sampling rate
    (
        traces,
        status,
        boundaries,
        period,
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
        cell_names,
    ) = _get_cellset_data(cell_set_files)

    # get behavior
    behavior = pd.read_parquet(annotations_file)
    # check that behavior and traces are the same length
    if len(behavior) != np.shape(traces)[0]:
        raise IdeasError(
            f"Mismatch between length of annotations and"
            f" traces. Annotations has {len(behavior)} samples, while"
            f" traces from provided cellset(s) have"
            f" {np.shape(traces)[0]} samples"
        )
    if trace_scale_method == Rescale.FRACTIONAL_CHANGE.value:
        unit = "fractional change"
        ylabel = "Mean Fractional Change in Activity"
    elif (
        trace_scale_method == Rescale.STANDARDIZE.value
        or trace_scale_method == Rescale.STANDARDIZE_BASELINE.value
    ):
        unit = "z-score"
        ylabel = "Mean Activity Z-score "
    elif trace_scale_method == Rescale.NORMALIZE.value:
        unit = "Normalized Activity"
        ylabel = "Mean Normalized Activity"
    else:
        unit = "Activity (a.u.)"
        ylabel = "Mean Activity (a.u.)"
    # Create preview looking at time spent in each state and a stacked bar
    # chart of fraction of time spent in each state
    _plot_state_time(
        behavior,
        column_name,
        state_names,
        state_colors,
        period,
    )

    _check_states_valid(
        states=state_names,
        behavior=behavior,
        column_name=column_name,
    )
    logger.info("Data loaded and parameters validated")

    logger.info("Modulation Data Created, creating plots")
    _plot_traces(
        traces=traces,
        behavior=behavior,
        data=None,
        column_name=column_name,
        filename=trace_preview,
        state_colors=state_colors,
        state_names=state_names,
        period=period,
    )

    # Only use baseline_state when method is BASELINE
    effective_baseline_state = (
        baseline_state if method == Comp.BASELINE.value else None
    )

    # Exception: always pass baseline_state for rescaling methods that require it
    rescale_needs_baseline = (
        trace_scale_method == Rescale.FRACTIONAL_CHANGE.value
        or trace_scale_method == Rescale.STANDARDIZE_BASELINE.value
    )

    data = _make_modulation_data(
        traces=traces,
        behavior=behavior,
        states=state_names,
        column_name=column_name,
        n_shuffle=n_shuffle,
        alpha=alpha,
        method=method,
        baseline_state=(
            baseline_state
            if rescale_needs_baseline
            else effective_baseline_state
        ),
        rescale=trace_scale_method,
    )

    _plot_population_average(
        data=data,
        filename=activity_average_preview,
        state_colors=state_colors,
        ylabel=ylabel,
    )

    # footprints

    # get contours for non-rejected neurons
    x, y = io.cell_set_to_contours(cell_set_files)
    mod_colors = modulation_colors.split(",")
    mod_colors = [color.strip() for color in mod_colors]

    up_modulation_color = mod_colors[0]
    down_modulation_color = mod_colors[1]
    non_modulation_color = [_ / 255 for _ in [225, 225, 225]]

    if num_accepted_cells > 0:
        cell_status_filter = "accepted"
    elif num_undecided_cells > 0:
        cell_status_filter = "undecided"

    x = [cx for cx, stat in zip(x, status) if stat == cell_status_filter]
    y = [cx for cx, stat in zip(y, status) if stat == cell_status_filter]

    plot_modulated_neuron_footprints(
        data=data,
        x=x,
        y=y,
        filename=modulation_preview,
        up_modulation_color=up_modulation_color,
        down_modulation_color=down_modulation_color,
        non_modulation_color=non_modulation_color,
        method=method,
        baseline_state=baseline_state,
    )

    plot_modulated_neuron_footprints(
        data=data,
        x=x,
        y=y,
        filename=modulation_histogram_preview,
        up_modulation_color=up_modulation_color,
        down_modulation_color=down_modulation_color,
        non_modulation_color=non_modulation_color,
        method=method,
        baseline_state=baseline_state,
        plot_type="hist",
    )

    # save data to CSV on disk
    logger.info("Saving data to CSV")
    _modulation_data_to_csv(
        data=data,
        filename=MODULATION_CSV_NAME,
        cell_names=cell_names,
        method=method,
        unit=unit,
        baseline_state=baseline_state,
    )

    # Create meta data

    activity_key = Path(MODULATION_CSV_NAME).stem
    activity_values = {
        "num_cells": len(data[state_names[0]]["mean_activity"]),
        "num_states": len(state_names),
        "states": state_names,
        "state_colors": state_colors,
        "method": method,
        "baseline_state": baseline_state,
        "modulation_colors": modulation_colors,
        "n_shuffle": n_shuffle,
        "alpha": alpha,
        "scale_method": trace_scale_method,
    }

    logger.info("Trace analysis complete, starting event analysis...")

    if event_set_files is not None and len(event_set_files) > 0:
        try:
            offsets, amplitudes = io.event_set_to_events(
                event_set_files, concatenate
            )
        except Exception as e:
            logger.error(
                f"Error in event set processing: {e}. Skipping event set processing"
            )
            raise e

        # Make appropriate labels for figures
        if event_scale_method == Rescale.FRACTIONAL_CHANGE.value:
            unit = "fractional change"
            ylabel = "Mean Fractional Change in Event Rate"
        elif (
            event_scale_method == Rescale.STANDARDIZE.value
            or event_scale_method == Rescale.STANDARDIZE_BASELINE.value
        ):
            unit = "z-score"
            ylabel = "Mean Event Rate Z-score"
        elif event_scale_method == Rescale.NORMALIZE.value:
            unit = "normalized Event Rate"
            ylabel = "Mean Normalized Event Rate"
        else:
            unit = "event rate (Hz)"
            ylabel = "Mean Event Rate (Hz)"

        event_timeseries = io.offsets_to_timeseries(
            offsets=offsets,
            status=status,
            timeseries_shape=traces.shape,
            period=period,
        )
        logger.info("Event data loaded")
        event_data = _make_modulation_data(
            traces=event_timeseries,
            behavior=behavior,
            states=state_names,
            column_name=column_name,
            n_shuffle=n_shuffle,
            alpha=alpha,
            method=method,
            baseline_state=baseline_state,
            rescale=event_scale_method,
        )
        logger.info("Event modulation data created, creating plots...")
        _plot_raster(
            events=offsets,
            event_timeseries=event_timeseries,
            behavior=behavior,
            column_name=column_name,
            period=period,
            state_colors=state_colors,
            state_names=state_names,
            filename=event_preview,
        )
        _plot_population_average(
            data=event_data,
            filename=event_average_preview,
            state_colors=state_colors,
            ylabel=ylabel,
        )

        plot_modulated_neuron_footprints(
            data=event_data,
            x=x,
            y=y,
            filename=event_modulation_preview,
            up_modulation_color=up_modulation_color,
            down_modulation_color=down_modulation_color,
            non_modulation_color=non_modulation_color,
            method=method,
            baseline_state=baseline_state,
        )

        plot_modulated_neuron_footprints(
            data=event_data,
            x=x,
            y=y,
            filename=event_modulation_histogram_preview,
            up_modulation_color=up_modulation_color,
            down_modulation_color=down_modulation_color,
            non_modulation_color=non_modulation_color,
            method=method,
            baseline_state=baseline_state,
            plot_type="hist",
        )
        logger.info("Event analysis complete, saving data to CSV")
        _modulation_data_to_csv(
            data=event_data,
            filename=EVENT_MODULATION_CSV_NAME,
            cell_names=_status_to_cell_names(status),
            method=method,
            unit=unit,
            baseline_state=baseline_state,
        )

        # Create meta data

        event_key = Path(EVENT_MODULATION_CSV_NAME).stem
        event_values = {
            "num_cells": len(data[state_names[0]]["mean_activity"]),
            "num_states": len(state_names),
            "states": state_names,
            "state_colors": state_colors,
            "method": method,
            "baseline_state": baseline_state,
            "modulation_colors": modulation_colors,
            "n_shuffle": n_shuffle,
            "alpha": alpha,
            "scale_method": event_scale_method,
        }

        # Save meta data
        meta_data = {
            activity_key: activity_values,
            event_key: event_values,
        }

        with open("output_metadata.json", "w") as f:
            json.dump(meta_data, f)
        logger.info("All analyses complete.")


# helper functions
@beartype
def _perform_permutation_test(
    traces: np.ndarray,
    compute_func,
    compute_args: list,
    n_shuffle: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """Perform permutation testing to find statistically significant modulation.

    :param traces: activity of individual cells
    :param compute_func: function used to compute modulation scores
    :param compute_args: arguments to pass to compute_func
    :param n_shuffle: number of shuffles for permutation test
    :param alpha: statistical significance threshold

    :return dict: dictionary containing up/down modulated neurons with
        corresponding p-values and modulation scores
    """
    np.random.seed(1984)

    # Compute modulation scores from the actual data
    modulation_scores = compute_func(*compute_args)

    # Bootstrap by circular permutation
    modulation_scores_bs = np.zeros((n_shuffle, len(modulation_scores)))
    for i in range(n_shuffle):
        # Create shuffled versions of all boolean masks
        shuffled_args = []
        for arg in compute_args[1:]:
            if isinstance(arg, np.ndarray) and arg.dtype == bool:
                shuffled_args.append(
                    np.roll(arg.copy(), np.random.randint(traces.shape[0]))
                )
            else:
                shuffled_args.append(arg)

        # Compute modulation with shuffled masks
        modulation_scores_bs[i, :] = compute_func(traces, *shuffled_args)

    # Calculate p-values
    p_val_1 = np.mean(modulation_scores < modulation_scores_bs, axis=0)
    p_val_2 = np.mean(modulation_scores > modulation_scores_bs, axis=0)
    p_val = np.minimum(p_val_1, p_val_2)

    # Identify significantly modulated neurons
    up_neurons = np.where((modulation_scores > 0) & (p_val < alpha))[0]
    down_neurons = np.where((modulation_scores < 0) & (p_val < alpha))[0]

    return {
        "up_modulated_neurons": up_neurons,
        "down_modulated_neurons": down_neurons,
        "p_val": p_val,
        "modulation_scores": modulation_scores,
    }


@beartype
def find_modulated_neurons(
    traces: np.ndarray,
    when: np.ndarray,
    alpha: float = 0.01,
    n_shuffle: int = 1000,
) -> dict:
    """Find modulated neurons and compute modulation score and statistical
    significance.

    :param traces: activity of individual cells
    :param when: boolean array set to True for time indices when a
        behavior occurred, False otherwise
    :param alpha: statistical significance threshold
    :param n_shuffle: number of shuffles for permutation test
    :return dict: dictionary containing up/down modulated neurons with
        corresponding p-values and modulation scores
    """
    return _perform_permutation_test(
        traces=traces,
        compute_func=compute_modulation,
        compute_args=[traces, when],
        n_shuffle=n_shuffle,
        alpha=alpha,
    )


@beartype
def find_two_state_modulated_neurons(
    traces: np.ndarray,
    when1: np.ndarray,
    when2: np.ndarray,
    alpha: float = 0.05,
    n_shuffle: int = 1000,
) -> dict:
    """Find modulated neurons and compute modulation
    score and statistical significance.

    :param traces: activity of individual cells
    :param when1: boolean array set to True for time indices of first state
    :param when2: boolean array set to True for time indices of second state
    :param alpha: statistical significance threshold
    :param n_shuffle: number of shuffles for permutation test

    :return dict: dictionary containing up/down
    modulated neurons with
    corresponding p-values and modulation scores
    """
    return _perform_permutation_test(
        traces=traces,
        compute_func=compute_two_state_modulation,
        compute_args=[traces, when1, when2],
        n_shuffle=n_shuffle,
        alpha=alpha,
    )


@beartype
def _modulation_data_to_csv(
    *,
    data: dict,
    filename: str,
    cell_names: List[str],
    method: str = Comp.NOT_STATE.value,
    baseline_state: Optional[str] = None,
    unit: str = "activity (a.u.)",
) -> None:
    """Convert modulation data to a CSV file for user inspection.

    :param data: Modulation data dictionary
    :param filename: Output filename
    :param cell_names: List of cell names/identifiers
    :param method: Comparison method used
    :param baseline_state: Baseline state (if applicable)
    :param unit: Unit of measurement for the data
    """
    # Initialize dictionaries to store columns for modulation data and mean activity
    df_data_modulation = {"name": cell_names}
    df_data_mean_activity = {}

    # Process all states first to extract mean activity data
    for state in data.keys():
        if "mean_activity" in data[state]:
            df_data_mean_activity[f"mean {unit} in {state}"] = data[state][
                "mean_activity"
            ]

    # Process modulation data based on the method
    if method == Comp.PAIRWISE.value:
        # Handle pairwise comparisons
        for s1 in data.keys():
            for s2 in data[s1].keys():
                if s2 != "mean_activity" and isinstance(data[s1][s2], dict):
                    _add_modulation_columns(
                        df_data_modulation, data[s1][s2], f"{s1} vs {s2}"
                    )
    elif method == Comp.BASELINE.value:
        # Handle baseline comparisons
        for state in data.keys():
            if state != baseline_state and all(
                key in data[state]
                for key in [
                    "modulation_scores",
                    "p_val",
                    "up_modulated_neurons",
                ]
            ):
                _add_modulation_columns(df_data_modulation, data[state], state)
    else:
        # Handle other methods (NOT_STATE, NOT_DEFINED)
        for state in data.keys():
            if all(
                key in data[state]
                for key in [
                    "modulation_scores",
                    "p_val",
                    "up_modulated_neurons",
                ]
            ):
                _add_modulation_columns(df_data_modulation, data[state], state)

    # Combine dictionaries - modulation data first, then mean activity data
    df_data = {**df_data_modulation, **df_data_mean_activity}

    # Create DataFrame and save to CSV
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)


def _add_modulation_columns(
    df_data: dict, state_data: dict, label: str
) -> None:
    """Add modulation columns to the dataframe dictionary.

    :param df_data: Dictionary to add columns to
    :param state_data: State data containing modulation information
    :param label: Label to use for the column names
    """
    n_neurons = len(state_data["p_val"])
    modulated_neurons = np.zeros(n_neurons)

    # Mark up-modulated neurons with 1, down-modulated with -1
    modulated_neurons[state_data["up_modulated_neurons"]] = 1
    modulated_neurons[state_data["down_modulated_neurons"]] = -1

    # Add columns to the dataframe dictionary
    df_data[f"modulation scores in {label}"] = state_data["modulation_scores"]
    df_data[f"p-values in {label}"] = state_data["p_val"]
    df_data[f"modulation in {label}"] = modulated_neurons
