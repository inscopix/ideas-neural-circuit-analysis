"""State-epoch data management utilities.

This module provides utilities for loading, processing, and validating
neural data across behavioral states and time epochs.
"""

import logging
import os
import pandas as pd
import numpy as np
from beartype import beartype
from beartype.typing import List, Optional, Dict, Any, Tuple
from ideas.exceptions import IdeasError
from utils.utils import (
    _epoch_time_to_index,
    _get_cellset_data,
    event_set_to_events,
    _parse_string_to_tuples,
    _validate_epochs_param,
    _redefine_epochs,
)
from utils.validation import _validate_events

logger = logging.getLogger(__name__)


def load_and_filter_cell_contours(
    cell_info: Dict[str, Any]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load cell contours and apply intelligent status filtering.

    Uses the same fallback logic as other functions: accepted cells first,
    then undecided cells if no accepted cells are available.

    Args
    ----
        cell_info: Dictionary containing cell information including:
                  - cell_set_files: List of cell set file paths
                  - cell_status_filter: Optional explicit filter
                    ("accepted", "undecided", "rejected")
                  - num_accepted_cells: Number of accepted cells (for auto-detection)
                  - num_undecided_cells: Number of undecided cells (for auto-detection)

    Returns
    -------
        Tuple of (filtered_x_contours, filtered_y_contours)

    Raises
    ------
        IdeasError: If cell files are missing or no valid cells are found

    """
    from ideas.io import cell_set_to_contours, cell_set_to_status

    # Validate input files
    cell_set_files = cell_info.get("cell_set_files", [])
    if not cell_set_files:
        raise IdeasError("No cell set files provided for contour loading")

    try:
        # Get cell contours
        x_all, y_all = cell_set_to_contours(files=cell_set_files)

        # Get cell status
        status = cell_set_to_status(files=cell_set_files)

        if len(x_all) != len(status) or len(y_all) != len(status):
            raise IdeasError(
                f"Mismatch between contour count ({len(x_all)}, {len(y_all)}) "
                f"and status count ({len(status)})"
            )
    except Exception as e:
        raise IdeasError(f"Failed to load cell contours or status: {e}") from e

    # Determine cell status filter
    explicit_filter = cell_info.get("cell_status_filter")

    if explicit_filter in ["accepted", "undecided", "rejected"]:
        # Use explicit filter when provided
        cell_status_filter = explicit_filter
        logger.info(f"Using explicit cell status filter: {cell_status_filter}")
    else:
        # Auto-detect based on cell counts (consistent with other functions)
        # Only consider accepted and undecided as valid for analysis
        num_accepted_cells = cell_info.get("num_accepted_cells", 0)
        num_undecided_cells = cell_info.get("num_undecided_cells", 0)

        if num_accepted_cells > 0:
            cell_status_filter = "accepted"
            logger.info(
                f"Auto-selected 'accepted' cells ({num_accepted_cells} available)"
            )
        elif num_undecided_cells > 0:
            cell_status_filter = "undecided"
            logger.info(
                f"Auto-selected 'undecided' cells ({num_undecided_cells} available)"
            )
        else:
            raise IdeasError(
                "No valid cells found for contour filtering. "
                "Both accepted and undecided cell counts are zero or missing."
            )

    # Apply status filtering
    x = [x_all[i] for i, s in enumerate(status) if s == cell_status_filter]
    y = [y_all[i] for i, s in enumerate(status) if s == cell_status_filter]

    # Validate filtered results
    if len(x) == 0 or len(y) == 0:
        raise IdeasError(
            f"No cells found with status '{cell_status_filter}'. "
            f"Available statuses: {set(status)}"
        )

    logger.info(
        f"Filtered to {len(x)} cells with status '{cell_status_filter}'"
    )
    return x, y


@beartype
def validate_input_files_exist(file_paths: List[str]) -> None:
    """Validate that all input files exist."""
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")


@beartype
def scale_data(
    data: np.ndarray,
    method: str = "none",
    behavior: Optional[pd.DataFrame] = None,
    column_name: str = "state",
    baseline_state: Optional[str] = None,
) -> np.ndarray:
    """Scale data using specified method.

    Args
    ----
        data: Input data array
        method: Scaling method ("none", "normalize", "standardize",
            "fractional_change", "standardize_baseline")
        behavior: Optional behavioral data DataFrame (required for baseline methods)
        column_name: Column name in behavior containing state information
            (required for baseline methods)
        baseline_state: State to use as baseline (required for baseline methods)

    Returns
    -------
        Scaled data array

    Raises
    ------
        IdeasError: If method is unknown or required parameters are missing

    """
    if method == "none":
        return data
    elif method == "normalize":
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    elif method == "standardize":
        return (data - np.nanmean(data)) / np.nanstd(data)
    elif method == "fractional_change":
        if behavior is None or baseline_state is None:
            raise IdeasError(
                "Behavior data and baseline_state must be specified for fractional change scaling"
            )
        # Import the helper function from utils
        from utils.utils import _fractional_change_states

        return _fractional_change_states(
            data, behavior, column_name, baseline_state
        )
    elif method == "standardize_baseline":
        if behavior is None or baseline_state is None:
            raise IdeasError(
                "Behavior data and baseline_state must be specified for "
                "standardize baseline scaling"
            )
        # Import the helper function from utils
        from utils.utils import _standardize_baseline

        return _standardize_baseline(
            data, behavior, column_name, baseline_state
        )
    else:
        raise IdeasError(f"Unknown scaling method: {method}")


def check_epochs_valid_state_epoch(
    *,
    num_samples: int,
    epochs: List[Tuple[float, float]],
    period: float,
    epoch_names: List[str],
) -> None:
    """Check if epochs are valid for state-epoch analysis.

    This function provides more detailed error messages and logging
    specific to state-epoch analysis compared to the generic validation.

    Args
    ----
        num_samples: Number of samples in the data
        epochs: List of epoch time ranges (start, end) in seconds
        period: Sampling period in seconds
        epoch_names: List of epoch names for error reporting

    Raises
    ------
        IdeasError: If epochs are invalid

    """
    epoch_idx = _epoch_time_to_index(epochs, period)

    for i, epoch in enumerate(epoch_idx):
        epoch_name = epoch_names[i] if i < len(epoch_names) else f"epoch_{i}"

        if epoch[0] >= epoch[1]:
            raise IdeasError(
                f"The start time of epoch '{epoch_name}' must be less than the end time. "
                f"Got start={epoch[0]}, end={epoch[1]}"
            )

        if epoch[0] < 0 or epoch[1] < 0:
            raise IdeasError(
                f"Epoch '{epoch_name}' times must be positive. "
                f"Got start={epoch[0]}, end={epoch[1]}"
            )

        if epoch[0] > num_samples:
            raise IdeasError(
                f"Epoch '{epoch_name}' start exceeds trace length. "
                f"Start index {epoch[0]} > {num_samples}"
            )

        if epoch[1] > num_samples:
            logger.info(
                f"Epoch '{epoch_name}' end index {epoch[1]} exceeds trace length {num_samples}. "
                f"Will be clamped to data length."
            )


def check_num_epochs_state_epoch(
    *,
    epochs: List[Tuple[float, float]],
    epoch_names: List[str],
    epoch_colors: List[str],
) -> None:
    """Check if the number of epochs, epoch names, and epoch colors match.

    Args
    ----
        epochs: List of epoch time ranges
        epoch_names: List of epoch names
        epoch_colors: List of epoch colors

    Raises
    ------
        IdeasError: If counts don't match

    """
    if len(epochs) != len(epoch_names) or len(epochs) != len(epoch_colors):
        raise IdeasError(
            f"The number of epochs ({len(epochs)}), epoch names ({len(epoch_names)}), "
            f"and epoch colors ({len(epoch_colors)}) must be the same."
        )


class StateEpochDataManager:
    """Manages loading and processing of multi-file, multi-epoch data."""

    def __init__(
        self,
        cell_set_files: List[str],
        event_set_files: Optional[List[str]],
        annotations_file: Optional[List[str]],
        concatenate: bool,
        use_registered_cellsets: bool,
        registration_method: str,
        # Validation parameters (for built-in validation)
        epochs: str,
        epoch_names: List[str],
        epoch_colors: List[str],
        state_names: List[str],
        state_colors: List[str],
        baseline_state: str,
        baseline_epoch: str,
        define_epochs_by: str,
        tolerance: float = 1e-4,
        sort_by_time: bool = True,
        allow_epoch_only_mode: bool = False,
    ):
        """Initialize StateEpochDataManager with file paths and validation parameters.

        Args:
        ----
        cell_set_files : List[str]
            List of cellset file paths
        event_set_files : Optional[List[str]]
            Optional list of eventset file paths
        annotations_file : Optional[List[str]]
            List of annotation file paths. Only the first file is used when provided.
        concatenate : bool
            Whether to concatenate multiple files
        use_registered_cellsets : bool
            Whether to use registered cellsets
        registration_method : str
            Method for cell registration
        epochs : str
            Epoch time periods as string representation of tuples
        epoch_names : List[str]
            List of epoch names
        epoch_colors : List[str]
            List of epoch colors
        state_names : List[str]
            List of state names
        state_colors : List[str]
            List of state colors
        baseline_state : str
            Baseline state name
        baseline_epoch : str
            Baseline epoch name
        define_epochs_by : str
            Method for defining epochs
        tolerance : float
            Tolerance for temporal alignment and file concatenation
        sort_by_time : bool
            Whether to sort cellsets by time when concatenating
        allow_epoch_only_mode : bool
            Enables fallback to dummy annotations when annotations_file is not supplied.

        """
        # File and processing configuration
        self.cell_set_files = cell_set_files
        self.event_set_files = event_set_files
        self.annotations_file = annotations_file
        self.concatenate = concatenate
        self.use_registered_cellsets = use_registered_cellsets
        self.registration_method = registration_method
        self.tolerance = tolerance
        self.sort_by_time = sort_by_time
        self.allow_epoch_only_mode = allow_epoch_only_mode

        # Validation parameters
        self.epochs = epochs
        self.epoch_names = epoch_names
        self.epoch_colors = epoch_colors
        self.state_names = state_names
        self.state_colors = state_colors
        self.baseline_state = baseline_state
        self.baseline_epoch = baseline_epoch
        self.define_epochs_by = define_epochs_by

        # Internal state for validation and data loading
        self._data_loaded = False
        self._validated = False
        self._traces = None
        self._cell_info = None
        self._epoch_periods = None
        self._cellset_metadata = None

    def _validate_parameters(self) -> None:
        """Validate input parameters before data loading."""
        # Validate parameter counts
        if len(self.state_names) != len(self.state_colors):
            raise IdeasError(
                f"Number of states ({len(self.state_names)}) must match number of "
                f"state colors ({len(self.state_colors)})"
            )
        if len(self.epoch_names) != len(self.epoch_colors):
            raise IdeasError(
                f"Number of epochs ({len(self.epoch_names)}) must match number of "
                f"epoch colors ({len(self.epoch_colors)})"
            )

        # Validate baseline parameters
        if self.baseline_state not in self.state_names:
            raise IdeasError(
                f"Baseline state '{self.baseline_state}' not found in states: {self.state_names}"
            )
        if self.baseline_epoch not in self.epoch_names:
            raise IdeasError(
                f"Baseline epoch '{self.baseline_epoch}' not found in epochs: {self.epoch_names}"
            )

    def _validate_and_load_cellsets(self) -> None:
        """Validate parameters and load cellset data."""
        if self._data_loaded:
            return

        logger.info("Validating parameters and loading cellset data...")

        # 1. Quick parameter validation (no data loading)
        self._validate_parameters()

        # 2. Load cellset data
        try:
            (
                self._traces,
                status,
                boundaries,
                period,
                num_accepted_cells,
                num_undecided_cells,
                num_rejected_cells,
                cell_names,
            ) = _get_cellset_data(
                cell_set_files=self.cell_set_files,
                sort_by_time=self.sort_by_time,
                tolerance=self.tolerance,
                concat=self.concatenate,
            )
        except IdeasError as err:
            logger.error(err)
            raise err

        # 3. Validate epochs using loaded data
        try:
            # Process epochs parameter
            epochs = _validate_epochs_param(
                epochs=self.epochs, define_epochs_by=self.define_epochs_by
            )

            # Redefine epochs if needed (must be done before parsing)
            epochs = _redefine_epochs(
                define_epochs_by=self.define_epochs_by,
                epochs=epochs,
                boundaries=boundaries,
            )

            # Parse the epoch string to tuples (with proper error handling)
            try:
                self._epoch_periods = _parse_string_to_tuples(epochs)
            except ValueError as e:
                raise IdeasError(
                    "The specified Epoch Periods could not be parsed. "
                    "Verify that the epoch periods are specified as comma-separated"
                    " tuples of the form (start, end) "
                    "and that no comma is missing between and within the tuples."
                ) from e

            # Validate epoch counts and timing
            if self._epoch_periods:
                check_num_epochs_state_epoch(
                    epochs=self._epoch_periods,
                    epoch_names=self.epoch_names,
                    epoch_colors=self.epoch_colors,
                )

                if self.define_epochs_by != "files":
                    check_epochs_valid_state_epoch(
                        num_samples=self._traces.shape[0],
                        epochs=self._epoch_periods,
                        period=period,
                        epoch_names=self.epoch_names,
                    )

        except IdeasError:
            # Re-raise IdeasError as-is
            raise
        except Exception as e:
            raise IdeasError(f"Failed to validate epochs: {e}") from e

        # 4. Create cell_info dictionary
        self._cell_info = {
            "cell_names": cell_names,
            "cell_status": status,
            "num_accepted_cells": num_accepted_cells,
            "num_undecided_cells": num_undecided_cells,
            "num_rejected_cells": num_rejected_cells,
            "boundaries": boundaries,
            "period": period,
            "cell_set_files": self.cell_set_files,
            "is_registered": self.use_registered_cellsets,
        }

        # 5. Store cellset metadata for event processing
        self._cellset_metadata = {
            "status": status,
            "boundaries": boundaries,
            "period": period,
            "num_accepted_cells": num_accepted_cells,
            "num_undecided_cells": num_undecided_cells,
            "num_rejected_cells": num_rejected_cells,
            "cell_names": cell_names,
        }

        self._data_loaded = True
        self._validated = True
        logger.info("Cellset data loaded and validated successfully")

    def get_epoch_periods(self) -> List[Tuple[float, float]]:
        """Get validated epoch periods."""
        self._validate_and_load_cellsets()
        return self._epoch_periods

    def get_cell_info(self) -> Dict[str, Any]:
        """Get cell information."""
        self._validate_and_load_cellsets()
        return self._cell_info.copy()

    def load_data(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame, Dict[str, Any]]:
        """Load all data with built-in validation.

        This method validates all parameters and loads cellset data, then loads
        events and annotations. All validation happens automatically.

        Returns
        -------
            Tuple of (traces, events, annotations_df, cell_info)

        """
        # 1. Validate and load cellset data (includes parameter validation)
        self._validate_and_load_cellsets()

        # 2. Load events using validated cellset data
        events = self._load_events()

        # 3. Load annotations (consistent with other tools - use first file only)
        # Note: Multiple annotation files are accepted but only the first is used
        if self.annotations_file and len(self.annotations_file) > 0:
            annotation_path = self.annotations_file[0]
            if annotation_path and str(annotation_path).strip():
                if str(annotation_path).endswith(".csv"):
                    annotations_df = pd.read_csv(annotation_path)
                elif str(annotation_path).endswith(".parquet"):
                    annotations_df = pd.read_parquet(annotation_path)
                else:
                    raise IdeasError(
                        "Unsupported file extension for annotations file. "
                        "Expected .csv or .parquet."
                    )
            else:
                annotation_path = None
        else:
            annotation_path = None

        if annotation_path is None:
            if not self.allow_epoch_only_mode:
                raise IdeasError(
                    "annotations_file must contain at least one valid file path."
                )
            # Create dummy annotations for epoch-only analysis
            num_timepoints = self._traces.shape[0]
            annotations_df = pd.DataFrame(
                {
                    "dummy_state": ["epoch_activity"] * num_timepoints,
                    "time": [
                        i * self._cellset_metadata["period"]
                        for i in range(num_timepoints)
                    ],
                }
            )

        # 4. Validate data consistency
        self._validate_data_consistency(self._traces, events, annotations_df)

        return self._traces, events, annotations_df, self._cell_info

    def _load_events(self) -> Optional[np.ndarray]:
        """Load and process events using validated cellset data."""
        if not self.event_set_files or len(self.event_set_files) == 0:
            logger.info("No event set files provided.")
            return None

        logger.info("Event set files provided. Processing events.")

        # Extract metadata from validated cellset data
        status = self._cellset_metadata["status"]
        period = self._cellset_metadata["period"]
        num_accepted_cells = self._cellset_metadata["num_accepted_cells"]
        num_undecided_cells = self._cellset_metadata["num_undecided_cells"]

        try:
            offsets, amplitudes = event_set_to_events(self.event_set_files)
        except IdeasError as err:
            logger.info("Error processing event set files.")
            logger.error(err)
            logger.warning("Proceeding without events due to loading error.")
            return None

        # Keep a copy of the original offsets
        original_offsets = list(offsets)
        valid_events = False
        final_offset_indices = None

        # Determine cell status filter (same logic as epoch_activity.py)
        if num_accepted_cells > 0:
            cell_status_filter = "accepted"
        elif num_undecided_cells > 0:
            cell_status_filter = "undecided"
        else:
            logger.warning(
                "No valid cells found for event processing. Proceeding without events."
            )
            return None

        # --- Attempt 1: Validate with unfiltered offsets ---
        logger.info("Attempting event validation with unfiltered offsets.")
        try:
            unfiltered_indices = [
                np.array([int(offset / period) for offset in cell])
                for cell in original_offsets
            ]
            if _validate_events(
                self._traces, original_offsets, unfiltered_indices
            ):
                logger.info("Unfiltered event validation successful.")
                valid_events = True
                final_offset_indices = unfiltered_indices
            else:
                logger.warning("Unfiltered event validation failed.")
        except Exception as e:
            logger.warning(
                f"Error during unfiltered event validation step: {e}"
            )

        # --- Attempt 2: Validate with filtered offsets (if Attempt 1 failed) ---
        if not valid_events:
            logger.info("Attempting event validation with filtered offsets.")
            try:
                # Filter the original offsets based on status
                filtered_offsets = [
                    cell
                    for cell, stat in zip(original_offsets, status)
                    if stat == cell_status_filter
                ]
                # Recalculate indices using the filtered offsets
                filtered_indices = [
                    np.array([int(offset / period) for offset in cell])
                    for cell in filtered_offsets
                ]

                if _validate_events(
                    self._traces, filtered_offsets, filtered_indices
                ):
                    logger.info("Filtered event validation successful.")
                    valid_events = True
                    final_offset_indices = filtered_indices
                else:
                    logger.warning("Filtered event validation failed.")
            except Exception as e:
                logger.warning(
                    f"Error during filtered event validation step: {e}"
                )

        # --- Final Check ---
        if not valid_events:
            logger.warning(
                "Traces and events do not match after attempting both "
                "unfiltered and filtered validation. Proceeding without events."
            )
            return None

        # Use the successfully validated offsets and indices
        offset_indices = final_offset_indices

        # Convert the indices to timeseries
        original_shape = self._traces.shape
        event_timeseries = np.zeros(original_shape)
        for idx, cell in enumerate(offset_indices):
            for event in cell:
                if (
                    event < original_shape[0] and idx < original_shape[1]
                ):  # Ensure we don't exceed time or cell bounds
                    event_timeseries[event, idx] = 1 / period

        # Check event rate for nans and replace with 0
        if np.isnan(event_timeseries).any():
            logger.warning("Event rate contains NaNs. Replacing with 0.")
        events = np.nan_to_num(event_timeseries)

        logger.info("Event timeseries created.")
        return events

    def _validate_data_consistency(
        self,
        traces: np.ndarray,
        events: Optional[np.ndarray],
        annotations_df: pd.DataFrame,
    ) -> None:
        """Validate that data dimensions are consistent."""
        if traces.shape[0] != len(annotations_df):
            raise IdeasError(
                f"Trace timepoints ({traces.shape[0]}) don't match "
                f"annotation timepoints ({len(annotations_df)})"
            )

        if events is not None and events.shape != traces.shape:
            raise IdeasError(
                f"Event dimensions {events.shape} don't match trace dimensions {traces.shape}"
            )

    def extract_state_epoch_data(
        self,
        annotations_df: pd.DataFrame,
        traces: np.ndarray,
        events: Optional[np.ndarray],
        state: str,
        epoch: str,
        column_name: str,
        parsed_epochs: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Extract data for a specific state-epoch combination.

        Args
        ----
            annotations_df: Behavioral annotations DataFrame
            traces: Neural trace data
            events: Optional event data
            state: Target behavioral state
            epoch: Target time epoch
            column_name: Column name for state information
            parsed_epochs: List of all epoch names

        Returns
        -------
            Dictionary containing extracted data or None if no data found

        """
        # Ensure validation and data loading has occurred
        self._validate_and_load_cellsets()

        try:
            # Find epoch index
            if epoch not in parsed_epochs:
                logger.warning(f"Epoch '{epoch}' not found in parsed epochs")
                return None

            epoch_idx = parsed_epochs.index(epoch)

            if self._epoch_periods and epoch_idx < len(self._epoch_periods):
                start_time, end_time = self._epoch_periods[epoch_idx]
                period = self._cellset_metadata["period"]
                start_idx = int(start_time / period) if period else 0
                end_idx = int(end_time / period) if period else traces.shape[0]

                # Clamp indices to data bounds
                start_idx = max(0, start_idx)
                end_idx = min(traces.shape[0], end_idx)

                # Extract epoch data
                epoch_annotations = annotations_df.iloc[start_idx:end_idx]
                epoch_traces = traces[start_idx:end_idx, :]
                epoch_events = (
                    events[start_idx:end_idx, :]
                    if events is not None
                    else None
                )

                # Filter by state
                state_mask = epoch_annotations[column_name] == state
                if not state_mask.any():
                    logger.warning(
                        f"No data found for state '{state}' in epoch '{epoch}'"
                    )
                    return None

                state_traces = epoch_traces[state_mask, :]
                state_events = (
                    epoch_events[state_mask, :]
                    if epoch_events is not None
                    else None
                )

                return {
                    "traces": state_traces,
                    "events": state_events,
                    "annotations": epoch_annotations[state_mask],
                    "num_timepoints": state_traces.shape[0],
                    "state": state,
                    "epoch": epoch,
                }
            else:
                logger.warning(f"No epoch period defined for epoch '{epoch}'")
                return None

        except Exception as e:
            logger.error(f"Error extracting data for {state}-{epoch}: {e}")
            return None
