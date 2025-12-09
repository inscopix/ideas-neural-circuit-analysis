import logging

import numpy as np
from ideas.exceptions import IdeasError

from utils.utils import _epoch_time_to_index, DIVISION_THRESHOLD

logger = logging.getLogger(__name__)


def _validate_correlation_params(
    *,
    method,
    state_names,
    baseline_state,
    trace_scale_method,
    event_scale_method,
):
    """Validate the parameters for the correlation calculation."""
    if method == "state vs baseline":
        assert baseline_state is not None, "Baseline state must be provided"
        assert (
            baseline_state in state_names
        ), "Baseline state must be one of the states"
        assert (
            len(state_names) > 1
        ), "If using state vs baseline, must have at least 1 non-baseline state"

    if method == "pairwise":
        assert (
            len(state_names) > 1
        ), "If using pairwise, must have at least 2 states to compare"

    if (
        trace_scale_method == "standardize_baseline"
        or trace_scale_method == "fractional_change"
    ):
        assert baseline_state is not None, "Baseline state must be provided"

    if (
        event_scale_method == "standardize_baseline"
        or event_scale_method == "fractional_change"
    ):
        assert baseline_state is not None, "Baseline state must be provided"


def _validate_files(files):
    """Validate that the number of files is consistent across groups.

    :Parameters
        files: The files to validate
    """
    expected_files = None

    for file in files:
        if file is not None:
            if expected_files is None:
                # Traces should always exist and be first
                expected_files = len(file)
            else:
                # Events and Correlations should be the same length as traces
                assert len(file) == expected_files, (
                    "The number of trace, event, and correlation files must be"
                    " the same for each group."
                )


def _validate_epoch_name_strings(epoch_names):
    """Clean string inputs to catch common user errors, and ensure that norm
    epoch name exists in the epoch names.

    :Parameters
        epoch_names: List or comma-separated string of epoch names

    :Returns
        list: Cleaned list of epoch names
    """
    if isinstance(epoch_names, str):
        epoch_names = epoch_names.split(",")

    epoch_names = [name.strip() for name in epoch_names]
    return epoch_names


def _check_epochs_valid(
    *,
    traces,
    epochs,
    period,
):
    """Check if the epochs are valid.

    :Parameters
        traces: Array of neural traces
        epochs: List of epoch time ranges
        period: Sampling period

    :Raises
        IdeasError: If epochs are invalid
    """
    epoch_idx = _epoch_time_to_index(epochs, period)

    for epoch in epoch_idx:
        # Make sure start is less than end
        if epoch[0] >= epoch[1]:
            raise IdeasError(
                "The start time of the epoch must be less than the end time."
            )
        # Make sure the epoch positive
        if epoch[0] < 0 or epoch[1] < 0:
            raise IdeasError("The epoch must be positive.")
        # Make sure the epoch is within the range of the traces
        if epoch[0] > traces.shape[0] or epoch[1] > traces.shape[0]:
            raise IdeasError(
                "The epoch is not within the range of the traces."
                f" ({epoch[0]}, {epoch[1]}) vs {traces.shape[0]}"
            )

        # Check if epoch has enough data points for statistical analysis
        epoch_length = epoch[1] - epoch[0]
        if epoch_length < 5:
            logger.warning(
                f"Epoch ({epoch[0]}, {epoch[1]}) has only {epoch_length} data points, "
                "which may be too few for reliable statistical analysis."
            )

        # Check for variance in each epoch to avoid divide-by-zero errors
        epoch_data = traces[epoch[0] : epoch[1], :]
        if np.any(np.nanstd(epoch_data, axis=0) < DIVISION_THRESHOLD):
            logger.warning(
                f"Epoch ({epoch[0]}, {epoch[1]}) contains variables with near-zero variance, "
                "which may cause statistical calculation warnings."
            )

    # make sure no epochs intersect
    # TODO add these checks, but I dont think anything will break if they're not here

    # Check for overall statistical issues
    _validate_statistical_suitability(traces, epochs, period)


def _validate_statistical_suitability(traces, epochs, period):
    """Check if the data is suitable for statistical analysis.

    :Parameters
        traces: Array of neural traces
        epochs: List of epoch time ranges
        period: Sampling period

    :Returns
        None: Issues are logged as warnings
    """
    epoch_idx = _epoch_time_to_index(epochs, period)

    # Check if any cells have constant values across all epochs
    for cell_idx in range(traces.shape[1]):
        cell_data = []
        for epoch in epoch_idx:
            cell_data.append(traces[epoch[0] : epoch[1], cell_idx])

        # Check for near-constant variance across epochs
        variances = [np.nanvar(data) for data in cell_data]
        if np.all(np.array(variances) < DIVISION_THRESHOLD):
            logger.warning(
                f"Cell {cell_idx} has near-zero variance across all epochs, "
                "which may cause statistical calculation warnings."
            )

    # Check if all epochs have enough distinct values
    for i, epoch in enumerate(epoch_idx):
        epoch_data = traces[epoch[0] : epoch[1], :]
        unique_counts = np.array(
            [
                len(np.unique(epoch_data[:, j][~np.isnan(epoch_data[:, j])]))
                for j in range(epoch_data.shape[1])
            ]
        )

        if np.any(unique_counts < 2):
            logger.warning(
                f"Epoch {i} contains cells with fewer than 2 distinct values, "
                "which can cause problems in statistical calculations."
            )


def _check_num_epochs(
    *,
    epochs,
    epoch_names,
    epoch_colors,
):
    """Check if the number of epochs, epoch names, and epoch colors are the
    same.

    :Parameters
        epochs: List of epoch time ranges
        epoch_names: List of epoch names
        epoch_colors: List of epoch colors

    :Raises
        IdeasError: If the number of epochs, names, and colors do not match
    """
    if len(epochs) != len(epoch_names) or len(epochs) != len(epoch_colors):
        raise IdeasError(
            "The number of epochs, epoch names, and epoch colors must be the"
            " same."
        )

    if len(epochs) < 2:
        raise IdeasError("There must be at least 2 epochs.")


def _validate_events(traces: np.ndarray, offsets: list, indices: list) -> bool:
    """Make sure traces and events are the same length.

    :Parameters
        traces: Array of neural traces
        offsets: List of event offsets
        indices: List of event indices

    :Returns
        bool: True if events are valid, False otherwise
    """
    if traces.shape[1] != len(offsets):
        logger.warning(
            f"Number of cells in cell set ({traces.shape[1]}) does not match"
            f" number of cells in event set ({len(offsets)})"
        )
        return False

    if len(indices) != traces.shape[1]:
        logger.warning(
            f"Number of cells in cell set ({traces.shape[1]}) does not match"
            f" number of cells in event set ({len(indices)})"
        )
        return False

    return True
