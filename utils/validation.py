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


def _validate_traces(traces):
    """Validate that traces are not all the same value.

    Traces are guaranteed to not be all Nan, make sure that traces are not
    all the same value, because correlation cannot be calculated if traces are
    all the same value.

    :Parameters
        traces: Array of neural traces

    :Returns
        bool: True if traces are valid, False otherwise
    """
    for i in range(traces.shape[1]):
        # Check if any trace is constant (all values are the same)
        if np.all(traces[:, i] == traces[0, i]):
            logger.warning(
                "This file contains a trace that is only a single value,"
                " correlation cannot be calculated."
            )
            return False

        # Check for traces with extremely low variance
        if np.nanstd(traces[:, i]) < DIVISION_THRESHOLD:
            logger.warning(
                "This file contains a trace with extremely low variance,"
                " which may cause statistical calculation warnings."
            )

    return True


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


def validate_timeseries_data(traces, min_variance_threshold=1e-6):
    """Validate time series data for common issues.

    :Parameters
    ----------
    traces : numpy.ndarray
        2D array of time series data (time x cells)
    min_variance_threshold : float
        Minimum variance threshold for detecting constant signals

    :Returns
    -------
    dict
        Dictionary of validation results with warnings and flags
    """
    result = {
        "valid": True,
        "warnings": [],
        "flags": {
            "has_nan": False,
            "has_inf": False,
            "has_constant": False,
            "has_low_variance": False,
        },
    }

    # Check basic properties
    if traces.size == 0:
        result["valid"] = False
        result["warnings"].append("Empty array provided")
        return result

    # Check for NaN values
    nan_mask = np.isnan(traces)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        result["flags"]["has_nan"] = True
        result["warnings"].append(
            f"Found {nan_count} NaN values ({nan_count / traces.size * 100:.2f}% of data)"
        )

    # Check for inf values
    inf_mask = np.isinf(traces)
    inf_count = np.sum(inf_mask)
    if inf_count > 0:
        result["flags"]["has_inf"] = True
        result["warnings"].append(f"Found {inf_count} infinite values")

    # Check for constant signals
    for i in range(traces.shape[1]):
        col_data = traces[:, i]
        if np.all(col_data == col_data[0]):
            result["flags"]["has_constant"] = True
            result["warnings"].append(
                f"Column {i} has constant value {col_data[0]}"
            )

    # Check for low variance
    variances = np.nanvar(traces, axis=0)
    low_var_cols = np.where(variances < min_variance_threshold)[0]
    if len(low_var_cols) > 0:
        result["flags"]["has_low_variance"] = True
        result["warnings"].append(
            f"Found {len(low_var_cols)} columns with variance below {min_variance_threshold}"
        )

    return result


def validate_anova_assumptions(data, dv, factors, subject=None):
    """Validate that data meets assumptions for ANOVA analysis.

    :Parameters
    data : pandas.DataFrame
        Data to analyze
    dv : str
        Name of dependent variable
    factors : list
        List of factor column names
    subject : str, optional
        Name of subject column for repeated measures

    :Returns
    dict
        Dictionary of validation results and warnings
    """
    results = {"valid": True, "warnings": []}

    # Check for missing values
    missing = data[dv].isna().sum()
    if missing > 0:
        results["warnings"].append(
            f"Found {missing} missing values in dependent variable"
        )

    # Check sample size adequacy
    for factor in factors:
        group_counts = data.groupby(factor).size()
        if (group_counts < 3).any():
            results["warnings"].append(
                f"Some groups in factor '{factor}' have fewer"
                f" than 3 observations: {dict(group_counts[group_counts < 3])}"
            )

    # Check for variance homogeneity
    try:
        import scipy.stats as stats

        if len(factors) == 1:
            # One-way ANOVA
            groups = [
                data[data[factors[0]] == val][dv].dropna()
                for val in data[factors[0]].unique()
            ]
            if all(
                len(g) > 1 for g in groups
            ):  # Need at least 2 observations per group
                _, p = stats.levene(*groups)
                if p < 0.05:
                    results["warnings"].append(
                        f"Levene test suggests heterogeneity of variance (p={p:.4f})"
                    )

        # Check for extreme outliers
        z_scores = np.abs(stats.zscore(data[dv].dropna()))
        outliers = np.where(z_scores > 3)[0]
        if len(outliers) > 0:
            results["warnings"].append(
                f"Found {len(outliers)} extreme outliers (z-score > 3)"
            )

    except Exception as e:
        results["warnings"].append(
            f"Could not perform full assumption testing: {str(e)}"
        )

    return results
