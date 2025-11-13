import ast
import json
import logging
import os
from enum import Enum
from pathlib import Path

import isx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import List, Union, Optional
from ideas import io
from ideas.exceptions import IdeasError
from matplotlib.spines import Spine
from ideas.validation import event_set_series
from utils.metadata import read_isxd_metadata
from ideas.utils import _sort_isxd_files_by_start_time
import warnings

logger = logging.getLogger()

# Define a constant for small values used in division or variance checks
DIVISION_THRESHOLD = 1e-10


def get_num_cells_by_status(cellset_filename: str):
    """Count the number of cells for each cell status"""
    cell_set = isx.CellSet.read(cellset_filename)

    num_accepted_cells = 0
    num_undecided_cells = 0
    num_rejected_cells = 0

    for index in range(cell_set.num_cells):
        cell_status = cell_set.get_cell_status(index)
        if cell_status == "accepted":
            num_accepted_cells += 1
        elif cell_status == "undecided":
            num_undecided_cells += 1
        elif cell_status == "rejected":
            num_rejected_cells += 1

    return num_accepted_cells, num_undecided_cells, num_rejected_cells

class Metric(Enum):
    """Metric is an enumeration that defines various types of metrics used in
    the toolbox.

    :Attributes
        TRACE_ACTIVITY (str): Represents the trace activity metric.
        EVENTRATE_ACTIVITY (str): Represents the event rate activity metric.
        AVG_POS_CORRELATION (str): Represents the average positive correlation metric.
        AVG_NEG_CORRELATION (str): Represents the average negative correlation metric.
    """

    TRACE = "Trace Activity"
    EVENT = "Eventrate Activity"
    POS_CORR = "Average Positive Correlation"
    NEG_CORR = "Average Negative Correlation"


class Rescale(Enum):
    """Rescale is an enumeration that defines various
    types of rescaling methods used in the toolbox.
    :Attributes
        NONE (str): Represents no rescaling.
        NORMALIZE (str): Represents normalization rescaling.
        STANDARDIZE (str): Represents standardization rescaling.
        STANDARDIZE_BASELINE (str): Represents standardization based on a baseline.
        STANDARDIZE_EPOCH (str): Represents standardization based on the first epoch.
        FRACTIONAL_CHANGE (str): Represents rescaling based on fractional change.
    """

    NONE = "none"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    STANDARDIZE_BASELINE = "standardize_baseline"
    STANDARDIZE_EPOCH = "standardize_epoch"
    FRACTIONAL_CHANGE = "fractional_change"


class Comp(Enum):
    """Comp is an enumeration class that defines different types of comparisons.
    :Attributes
        NOT_STATE (str): Represents a comparison between a state and a non-state.
        NOT_DEFINED (str): Represents a comparison between a state and an undefined state.
        PAIRWISE (str): Represents a pairwise comparison.
        BASELINE (str): Represents a comparison between a state and a baseline.
    """

    PAIRWISE = "pairwise"
    NOT_STATE = "state vs not state"
    NOT_DEFINED = "state vs not defined"
    BASELINE = "state vs baseline"


def _parse_string_to_tuples(s):
    """Parse a string representation of tuples into a list of tuples.

    :Args
        s (str): The string representation of tuples.
    :Returns
        list: A list of tuples.
    :Example
        >>> _parse_string_to_tuples('(1,2),(3,4),(5,6)')
        [(1, 2), (3, 4), (5, 6)]
        ("Example 1" , "ExaMple 2" ) , ( "example 1","Example 3")
    """
    # Make it agnostic to any spaces
    s = s.replace(" ", "").split("),(")
    s = [str.strip("()") for str in s]
    return [ast.literal_eval(f"({pair})") for pair in s]


def _parse_string_to_list_of_tuples(s):
    """Parse a string representation of lists of tuples into a list of several tuples."""
    # Make it agnostic to any spaces
    s = s.replace(" ", "").split("],[")
    s = [str.strip("[]") for str in s]
    return s


@beartype
def _save_experiment_annotations_preview_and_metadata(
    parquet_file_name: str,
    output_preview_filename: str,
    top_n_states: int = 10,
) -> None:
    """Save a preview of an experiment annotations dataframe to a SVG file.
       By default the top ten most common states are shown on a bar chart.

    :param df: input dataframe from which to extract a preview
    :param output_preview_filename: path to the output json file
    :param num_states: top most common states to include in the bar chart
    """
    df = pd.read_parquet(parquet_file_name)

    common_states = df["state"].value_counts().nlargest(top_n_states)

    # sometimes no annotation frame can be mapped to an isxd frame
    # so the state is set to an empty string
    # don't display those states
    common_states.drop("", inplace=True, errors="ignore")

    states = common_states.index
    counts = common_states.values

    fig, ax = plt.subplots()

    states = [state[:30] for state in states]
    ax.bar(states, counts)
    ax.set_title("Most Common States")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], label_type="edge")
    ax.margins(y=0.1)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()

    fig.savefig(
        output_preview_filename,
        dpi=300,
        transparent=True,
    )
    plt.close(fig)

    key_values = {
        "dataset": {
            "states": list(df["state"].unique()),
        },
        "metrics": {
            "num_rows": len(df),
            "num_columns": len(df.columns),
        },
        "column_names": list(df.columns),
    }

    file_key = Path(parquet_file_name).stem
    metadata = {file_key: key_values}

    with open("output_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)


@beartype
def _check_columns_in_df(
    *,
    df: pd.DataFrame,
    columns: tuple,
) -> None:
    """Check that the columns are in the dataframe."""
    for column in columns:
        if column not in df:
            raise IdeasError(
                "The annotations data frame does not contain"
                f" the column requested: {column}. The columns are:"
                f" {list(df.columns)}",
            )


@beartype
def _get_cellset_data(
    cell_set_files: List[str],
    sort_by_time: bool = True,
    tolerance: float = 1e-6,
    concat: bool = True,
):
    """Get traces for non-rejected neurons
    traces with NaNs in them are automatically marked
    as a rejected
    """
    traces = io.cell_set_to_traces(
        files=cell_set_files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
        flush_concatenate=concat,
    )
    status = io.cell_set_to_status(
        files=cell_set_files,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
    )

    cellset = isx.CellSet.read(cell_set_files[0])
    period = cellset.timing.period.secs_float

    boundaries = _get_cellset_boundaries(
        cell_set_files=cell_set_files,
        period=period,
    )

    # reject cells whose traces contain NaN values
    is_nan = np.all(np.isnan(traces), axis=0)
    num_nan_cells = np.sum(is_nan)
    if num_nan_cells > 0:
        logger.info(
            f"Number of traces that contain only NaN values: {num_nan_cells}. "
            f"The corresponding cells will be rejected."
        )
        status[is_nan] = "rejected"

    # count number of cells for each status
    num_accepted_cells = 0
    num_undecided_cells = 0
    num_rejected_cells = 0
    for s in status:
        if s == "accepted":
            num_accepted_cells += 1
        elif s == "undecided":
            num_undecided_cells += 1
        elif s == "rejected":
            num_rejected_cells += 1

    if num_accepted_cells > 0:
        logger.info(f"Detected {num_accepted_cells} accepted cells to analyze")
        target_status = "accepted"
    elif num_undecided_cells > 0:
        logger.info(
            f"No accepted cells were detected. "
            f"Detected {num_undecided_cells} undecided cells to analyze"
        )
        target_status = "undecided"
    else:
        raise IdeasError("There are no accepted or undecided cells to process")

    cell_filter = status == target_status
    traces = traces[:, cell_filter]

    cellset = isx.CellSet.read(cell_set_files[0])
    cell_names = [
        cellset.get_cell_name(i)
        for i in range(cellset.num_cells)
        if status[i] == target_status
    ]

    if traces.shape[1] == 0:
        raise IdeasError(
            "This cell set had 0 accepted/undecided cells, so we cannot use any data here"
        )

    return (
        traces,
        status,
        boundaries,
        period,
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
        cell_names,
    )


def _check_states_valid(
    *,
    states,
    behavior: pd.DataFrame,
    column_name: str,
):
    """Check that states are valid, column names are valid, etc"""
    valid_states = list(behavior[column_name].unique())

    if column_name not in behavior:
        raise IdeasError(
            "The annotations data frame does not contain the"
            f" column requested: {column_name}."
            f" The columns are: {list(behavior.columns)}",
        )

    if len(states) == 0:
        raise IdeasError(
            """This tool expects at least 1 state to be defined.
Instead, this tool received no states.""",
        )

    for state in states:
        if state not in behavior[column_name].unique():
            raise IdeasError(
                f"The state to be analyzed: {state}, does not"
                f" exist. Valid states are: {valid_states}"
            )


def _bin_data(data, bin_size, period):
    """Bin the data based on the bin size."""
    if (bin_size is not None) and (bin_size > 0):
        bin_size_in_idxs = int(bin_size / period)
        # bin and average the data
        bins = np.arange(0, data.shape[0], bin_size_in_idxs)
        new_data = np.zeros((len(bins), data.shape[1]))
        for i, bin_start in enumerate(bins):
            bin_end = bin_start + bin_size_in_idxs
            new_data[i] = np.nanmean(data[bin_start:bin_end], axis=0)
        return new_data
    else:
        return data


def _norm_2D_array(array):
    """Normalize a 2D array to the range [0, 1]."""
    min_vals = np.nanmin(array, axis=0)
    max_vals = np.nanmax(array, axis=0)
    return (array - min_vals) / (max_vals - min_vals)


def _standardize_2D_array(array):
    """Standardize a 2D array to have a mean of 0 and a standard deviation of
    1.
    """
    return (array - np.nanmean(array, axis=0)) / np.nanstd(array, axis=0)


def _fractional_change_2D_array(array, epochs, period):
    """Calculate the fractional change in the array compared to the first epoch
    defined.
    """
    # Convert the period to an
    # make the array non-negative by adding the minimum value
    array += np.abs(np.nanmin(array))
    epoch_idx = _epoch_time_to_index(epochs, period)
    start, end = epoch_idx[0]
    # get the baseline average value

    baseline = np.nanmean(array[start:end], axis=0)
    # replace any zeros with 1 to avoid division by zero
    baseline[baseline == 0] = 1

    # calculate the fractional change
    return array / baseline


def _fractional_change_states(traces, behavior, column_name, baseline_state):
    """Calculate the fractional change in the traces compared to the baseline state."""
    # make traces non-negative
    traces += np.abs(np.nanmin(traces))
    # get the baseline state
    baseline_state_traces = traces[behavior[column_name] == baseline_state, :]
    baseline = np.nanmean(baseline_state_traces, axis=0)
    # replace any zeros with 1 to avoid division by zero
    baseline[baseline == 0] = 1
    # calculate the fractional change
    return traces / baseline


def _standardize_to_first_epoch(array, epochs, period):
    """Standardize the array to the first epoch defined."""
    epoch_idx = _epoch_time_to_index(epochs, period)
    start, end = epoch_idx[0]
    # Compute mean and std for the first epoch
    mean_val = np.nanmean(array[start:end], axis=0)
    std_val = np.nanstd(array[start:end], axis=0)
    # Avoid division by zero
    std_val = np.where(std_val == 0, 1, std_val)
    return (array - mean_val) / std_val


def _standardize_baseline(traces, behavior, column_name, baseline_state):
    """Standardize the array to the baseline state."""
    # get the baseline state
    baseline_state_traces = traces[behavior[column_name] == baseline_state, :]
    return (traces - np.nanmean(baseline_state_traces, axis=0)) / np.nanstd(
        baseline_state_traces, axis=0
    )


def _process_files(
    files: List[str], data_type: str, df: pd.DataFrame, epoch_names: List[str]
) -> pd.DataFrame:
    """Process a list of files, validates their content, and merges them into
    a single DataFrame.

    :Args
        files (List[str]): List of file paths to be processed.
        data_type (str): Type of data being processed (e.g., 'traces', 'events').
        df (pd.DataFrame): Existing DataFrame to merge the processed data into.
        epoch_names (List[str]): List of epoch names to validate in the data.

    :Returns
        pd.DataFrame: Merged DataFrame containing the processed data from all files.

    :Raises
        AssertionError: If the data does not contain the
        required 'Epoch' column or specified epoch names.
        AssertionError: If the number of rows or cell IDs
        do not match between the existing DataFrame and the new data.
    """
    temp_df = pd.DataFrame()
    for file in files:
        # Read in the data
        cur_df = pd.read_csv(file)

        # Validate Epochs in data
        assert (
            "Epoch" in cur_df.columns
        ), f"The {data_type} data does not contain an 'Epoch' column."

        # Make sure epochs are strings even if they're numbers
        cur_df["Epoch"] = cur_df["Epoch"].astype(str)

        for epoch in epoch_names:
            assert (
                epoch in cur_df["Epoch"].unique()
            ), f"The {data_type} data does not contain the epoch '{epoch}',"
            f" available epochs are {cur_df['Epoch'].unique()}."

        # Add file identifier
        cur_df["File"] = os.path.basename(file)

        # rename Activity column to match data type
        cur_df.rename(
            columns={
                "File": f"{data_type} file",
                "Activity": f"{data_type} Activity",
            },
            inplace=True,
        )

        # Concatenate DataFrames and reset index
        temp_df = pd.concat([temp_df, cur_df], ignore_index=True)

    # Check if DataFrame is empty
    if temp_df.empty:
        logger.warning("The DataFrame is empty.")
    else:
        # Remove duplicate index labels (if any)
        temp_df = temp_df[~temp_df.index.duplicated(keep="first")]

    # Check if df is empty
    if df.empty:
        df = temp_df
    else:
        # join the data
        # Check that the length of the data is the same
        assert len(df) == len(
            temp_df
        ), "The number of rows for traces and events data is not the same for each file."

        assert (
            df["Cell"].all() == temp_df["Cell"].all()
        ), "The cell IDs for traces and events data are not the same for each file."

        df[f"{data_type} Activity"] = temp_df[f"{data_type} Activity"]
        df[f"{data_type} file"] = temp_df[f"{data_type} file"]

    return df


def _combine_data(
    traces: List[str],
    events: List[str],
    corr: List[str],
    group_name: str,
    epoch_names: List[str],
    metadata: dict,
):
    """Combine trace and event data from multiple CSV files into two
    DataFrames.

    :Args
        traces (List[str]): List of file paths to trace CSV files.
        events (List[str]): List of file paths to event CSV files.
        group_name (str): Name of the group for which data is being combined.
        epoch_names (List[str]): List of epoch names to validate against the data.

    :Returns
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - trace_data: Combined DataFrame of all trace data.
            - event_data: Combined DataFrame of all event data.
    """
    data = pd.DataFrame()

    data = _process_files(traces, "Trace", data, epoch_names)

    if events is not None:
        data = _process_files(events, "Eventrate", data, epoch_names)

    data.to_csv(f"{group_name}_combined_data.csv", index=False)
    data_key = f"{group_name}_combined_data"
    # Add metadata
    metadata[data_key] = {
        "num_epochs": len(epoch_names),
        "epoch_names": epoch_names,
        "num_files": len(traces),
    }
    # Combining correlation data is a little different

    if corr is not None:
        corr_data = pd.DataFrame(
            columns=[
                "Epoch",
                "File",
                "Average Positive Correlation",
                "Average Negative Correlation",
            ]
        )

        for corr_file in corr:
            try:
                # read in numpy file
                loaded_data = np.load(corr_file, allow_pickle=True)

                # Handle different possible data formats
                if isinstance(loaded_data, dict):
                    corr_dict = loaded_data
                elif hasattr(loaded_data, "item"):
                    try:
                        corr_dict = loaded_data.item()
                        if not isinstance(corr_dict, dict):
                            logger.warning(
                                f"File {corr_file} could not be converted to dictionary. Skipping."
                            )
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Error converting {corr_file} to dictionary: {e}. Skipping."
                        )
                        continue
                else:
                    logger.warning(
                        f"Unsupported correlation data format in {corr_file}. Skipping."
                    )
                    continue

                for epoch in corr_dict.keys():
                    vals = corr_dict[epoch]
                    if not isinstance(vals, np.ndarray):
                        try:
                            vals = np.array(vals, dtype=float)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Cannot convert"
                                f" correlation values to array for epoch {epoch}. Skipping."
                            )
                            continue

                    # Handle empty arrays
                    pos_vals = vals[vals > 0]
                    neg_vals = vals[vals < 0]

                    avg_pos_corr = (
                        np.nanmean(pos_vals) if len(pos_vals) > 0 else np.nan
                    )
                    avg_neg_corr = (
                        np.nanmean(neg_vals) if len(neg_vals) > 0 else np.nan
                    )

                    temp_df = pd.DataFrame(
                        {
                            "Epoch": [epoch],
                            "File": [os.path.basename(corr_file)],
                            "Average Positive Correlation": [avg_pos_corr],
                            "Average Negative Correlation": [avg_neg_corr],
                        }
                    )
                    # drop columns with all-NA entries in temp_df
                    temp_df = temp_df.dropna(axis="columns", how="all")
                    # drop columns with all-NA entries in corr_data
                    corr_data = corr_data.dropna(axis="columns", how="all")
                    # add the data to the dataframe
                    corr_data = pd.concat(
                        [corr_data, temp_df], ignore_index=True
                    )
            except Exception as e:
                logger.warning(
                    f"Error processing correlation file {corr_file}: {str(e)}"
                )
                continue

        # save data
        corr_data.to_csv(f"{group_name}_correlation_data.csv", index=False)

        corr_key = f"{group_name}_correlation_data"

        metadata[corr_key] = {
            "num_epochs": len(epoch_names),
            "epoch_names": epoch_names,
            "num_files": len(corr),
        }
    else:
        corr_data = None

    return data, corr_data, metadata


def _epoch_time_to_index(epochs, period):
    """Convert epoch time to index based on a given period.

    :Parameters
    epochs (list): A list tuples for the start and end of each epoch.
    period (int): The period to divide the epoch time by.

    :Returns
    list: A list of indices corresponding to the converted epoch time.
    """
    return [
        (int((epoch[0] / period)), int(epoch[1] / period)) for epoch in epochs
    ]


@beartype
def event_set_to_events(
    files: Union[str, List[str]],  # , flush_concatenate: bool = True
):
    # -> (NumpyFloat2DArray, NumpyFloatVector):
    """Read event set file and get all offsets and amplitudes

    :Returns
    - all_offsets: list (length of num cells) of lists (length of num events) of offsets
    - all_amplitudes: lists (length of num cells) of lists (length of num events) of amplitudes
    """
    # Ensure that files is a list
    if isinstance(files, str):
        files = [files]

    # Ensure that files are in a series, and in order
    files = event_set_series(files)
    # initialize time offset
    time_offset = 0

    # Initalize storage for all offsets and amplitudes
    # Each cell gets its own list of offsets and amplitudes
    all_offsets = []
    all_amplitudes = []

    # Go through each file in the series
    for file in files:
        # load the event set
        eventset = isx.EventSet.read(file)

        # get the number of cells This is assumed to be the same accross all files
        num_cells = eventset.num_cells
        for i in range(num_cells):
            # get the offsets and amplitudes for each cell
            # offsets is a uint in microseconds since start of recording
            offsets, amplitudes = eventset.get_cell_data(i)
            # convert offsets to seconds
            offsets = offsets / 1e6
            # add the time offset to the offsets
            offsets += time_offset
            # offsets and amplitudes are stored in an array, convert to list and add
            # values to the correct cell in the storage list
            if len(all_offsets) < i + 1:
                all_offsets.append(offsets.tolist())
                all_amplitudes.append(amplitudes.tolist())
            else:
                all_offsets[i].extend(offsets.tolist())
                all_amplitudes[i].extend(amplitudes.tolist())
        # update the time offset
        time_offset += (
            eventset.timing.num_samples * eventset.timing.period.secs_float
        )

    return all_offsets, all_amplitudes


# SVG size estimation factors - heuristic values for different plot elements
# These factors are used to estimate SVG file size before saving
QUADMESH_FACTOR = 200  # For correlation matrix heatmaps
LINECOLLECTION_SEGMENT_FACTOR = 2500  # For line plots and spatial connections
POLYCOLLECTION_FACTOR = 35000  # For hexbin plots (many hexagonal patches)
SCATTER_POINT_FACTOR = 800  # For scatter plots (conservative estimation)


def estimate_svg_size(fig) -> int:
    """Estimate the size of an SVG file in bytes based on plot elements.

    This function analyzes matplotlib figure elements and estimates the resulting
    SVG file size using empirically determined factors.

    :param fig: Matplotlib figure object to analyze
    :returns: Estimated size in bytes
    """
    import logging
    from matplotlib.collections import QuadMesh, LineCollection, PolyCollection

    logger = logging.getLogger()
    estimated_size = 0

    for ax in fig.get_axes():
        # Check for mesh/collection objects (used for correlation matrices)
        for collection in ax.collections:
            if isinstance(collection, QuadMesh) and hasattr(
                collection, "_coordinates"
            ):
                mesh_shape = collection._coordinates.shape
                if len(mesh_shape) >= 3:
                    cell_count = (mesh_shape[0] - 1) * (mesh_shape[1] - 1)
                    estimated_size += cell_count * QUADMESH_FACTOR
            elif isinstance(collection, LineCollection):
                # Estimate complexity for LineCollection (e.g., eventplot, spatial maps)
                try:
                    # get_segments() returns a list of (N, 2) arrays
                    num_segments = len(collection.get_segments())
                    # Each segment has a start and end point,
                    # but complexity might scale with segment count
                    estimated_size += (
                        num_segments * LINECOLLECTION_SEGMENT_FACTOR
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not get segments from LineCollection: {e}"
                    )
            elif isinstance(collection, PolyCollection):
                # Estimate complexity for PolyCollection (e.g., hexbin plots, scatter plots)
                try:
                    # get_paths() returns a list of Path objects
                    num_polygons = len(collection.get_paths())
                    # Each polygon can have multiple vertices, complexity scales with polygon count
                    estimated_size += num_polygons * POLYCOLLECTION_FACTOR
                except Exception as e:
                    logger.warning(
                        f"Could not get paths from PolyCollection: {e}"
                    )
            elif hasattr(collection, "get_offsets"):
                # PathCollection (scatter plots) - detect by presence of get_offsets method
                try:
                    num_points = len(collection.get_offsets())
                    # Each scatter point becomes an SVG element, complexity scales with point count
                    estimated_size += num_points * SCATTER_POINT_FACTOR
                except Exception as e:
                    logger.warning(
                        f"Could not get offsets from scatter plot: {e}"
                    )

    return estimated_size


def save_optimized_svg(
    fig,
    output_filename: str,
    max_size_mb: float = 10,
    dpi: int = 300,
    pad_inches: float = 0.1,
) -> int:
    """Save a figure as an SVG, with adaptive rasterization for large correlation matrices.

    This function analyzes the content and applies rasterization when needed.

    :param fig: Matplotlib figure object.
    :param output_filename: Output filename for the SVG.
    :param max_size_mb: Maximum allowed file size in MB before rasterization.
    :param dpi: Resolution for rasterized elements.
    :param pad_inches: Amount of padding in inches around the figure when bbox_inches is 'tight'.
    :returns: Estimated size in bytes before optimization
    """
    # Import required modules
    from matplotlib.collections import (
        Collection,
    )
    from matplotlib.image import AxesImage

    # Ensure layout is optimized - handle colorbars and complex layouts gracefully
    try:
        # Temporarily suppress the specific tight_layout warning for complex layouts
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
                category=UserWarning,
            )
            fig.tight_layout()
    except ValueError:
        # tight_layout sometimes fails with colorbars or complex subplot layouts
        logger.debug("tight_layout failed, using constrained_layout instead")
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            # If all layout options fail, proceed without layout optimization
            logger.debug(
                "Layout optimization skipped due to incompatible axes"
            )
            pass

    # Estimate the size using the extracted function
    estimated_size_bytes = estimate_svg_size(fig)
    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
    should_rasterize = estimated_size_mb > max_size_mb

    if should_rasterize:
        logger.info(
            f"Large matrix detected (est. {estimated_size_mb:.1f}MB)"
            f" - using rasterization for {output_filename}"
        )

        # Apply rasterization to elements
        for ax in fig.get_axes():
            # Rasterize images and matrices
            for img in ax.get_images():
                if hasattr(img, "set_rasterized"):
                    img.set_rasterized(True)

            # Rasterize QuadMesh objects
            for collection in ax.collections:
                if hasattr(collection, "set_rasterized"):
                    collection.set_rasterized(True)

            # Rasterize lines
            for line in ax.lines:
                if hasattr(line, "set_rasterized"):
                    line.set_rasterized(True)

            # Other heavy elements
            for artist in ax.get_children():
                if (
                    isinstance(artist, Spine)
                    or artist in ax.get_xticklabels()
                    or artist in ax.get_yticklabels()
                ):
                    continue

                if hasattr(artist, "set_rasterized"):
                    if isinstance(artist, Collection) or isinstance(
                        artist, AxesImage
                    ):
                        artist.set_rasterized(True)

        # Save with rasterization
        fig.savefig(
            output_filename,
            format="svg",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )
    else:
        # Save as pure vector SVG
        fig.savefig(
            output_filename,
            format="svg",
            transparent=True,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )

    return estimated_size_bytes


@beartype
def _validate_epochs_param(
    *,
    epochs: Optional[Union[list, str]],
    define_epochs_by: str,
) -> str:
    """Validate epochs parameter by ensuring it is a string. When defining
    epochs by local file time, the list of lists of tuples is interpreted
    as a list i.o. a string by BE, hence the need for this present function.
    """
    # if epochs is a list, reformat it properly as a list of lists
    # (one per input cell set) of tuples (one per epoch)
    if isinstance(epochs, list):
        all_epochs_list = []
        for file_epochs in epochs:
            file_epochs_list = []
            for epoch in file_epochs:
                file_epochs_list.append(tuple(epoch))
            all_epochs_list.append(file_epochs_list)

        logger.info(
            "The 'epochs' parameter was interpreted as a list, ",
            "and got automatically replaced by a string, ",
            f"as {str(all_epochs_list)}.",
        )

        epochs = str(all_epochs_list)

    if epochs is None and define_epochs_by == "files":
        epochs = ""

    if not isinstance(epochs, str):
        raise IdeasError(
            "The 'epochs' parameters should be a string, ",
            f"but is a {type(epochs)} instead.",
        )

    return epochs


@beartype
def _redefine_epochs(
    *,
    define_epochs_by: str,
    epochs: str,
    boundaries: List[int],
) -> str:
    """Redefine epochs, either by setting one epoch per input cell set, or by
    adding elapsed global time to epochs defined in local file time.
    """
    if define_epochs_by in ["files", "local file time"]:
        # map epochs to cellsets
        if define_epochs_by == "files":
            n_epochs = len(boundaries) - 1
            epochs = ""
            for idx in range(n_epochs):
                epochs += f"({boundaries[idx]}, "
                epochs += f"{boundaries[idx + 1]}), "
            epochs = epochs[:-2]
            logger.info(
                "Automatically defined epochs based on input cell set files: "
                f"{epochs}."
            )

        # add previous duration to each "local file time"-based epoch
        else:
            epochs_local_str = epochs
            epochs_local_list = _parse_string_to_list_of_tuples(epochs)
            epochs_global = []
            for epochs_local, offset in zip(epochs_local_list, boundaries):
                epochs_local = _parse_string_to_tuples(epochs_local)
                for epoch in epochs_local:
                    epoch_global = tuple([x + offset for x in epoch])
                    epochs_global.append(epoch_global)
            epochs = str(epochs_global).strip("[]")
            logger.info(
                "Automatically converted local epochs into global epochs: "
                f"{epochs_local_str} -> {epochs}."
            )

    return epochs


def _get_cellset_boundaries(
    cell_set_files,
    period,
):
    """Get a list of temporal boundaries between cellsets,
    including 0 and last global time point, in seconds.
    """
    # get cellset lengths; cs stands for cellset
    cs_length_list = [
        isx.CellSet.read(x).timing.num_samples for x in cell_set_files
    ]

    # get cumulative cellset lengths
    cumul_cs_length_list = np.cumsum([0] + cs_length_list)

    # convert length (in samples) to duration (in seconds)
    boundaries = [int(np.floor(x * period)) for x in cumul_cs_length_list]

    # return a list of temporal boundaries, in seconds
    return boundaries


def remove_unsupported_characters(
    input_string: str, unsupported_chars=["$"], placeholder_char="�"
):
    """Remove unsupported characters from a given string and replace them for a placeholder character.

    :param input_string: string from which to remove unsupported characters
    :param unsupported_chars: list of characters to remove
    :param placeholder_char: character to insert in place of unsupported characters
    """
    output_string = input_string
    for c in unsupported_chars:
        if c in output_string:
            output_string = output_string.replace(c, placeholder_char)
            logger.warning(
                f"Invalid character '{c}' found and replaced with '{placeholder_char}'"
            )
    return output_string

def compute_end_time(timing_info):
    """Compute end time of an isxd file.
    :param isxd_timing_info: json object containing isxd timing information
    :return: end time json object of the form
             {
                "secsSinceEpoch": {"num": 123456, "den": 1000}
                "utcOffset": 0
             }
    """
    # extract start time and period contained in the timing info object
    start_time = (
        timing_info["start"]["secsSinceEpoch"]["num"]
        / timing_info["start"]["secsSinceEpoch"]["den"]
    )
    period = timing_info["period"]["num"] / timing_info["period"]["den"]

    # compute end time
    end_time = start_time + (period * timing_info["numTimes"])
    end_time_den = timing_info["start"]["secsSinceEpoch"]["den"]
    end_time_num = round(end_time * end_time_den)

    # return properly formatted end time
    return {
        "secsSinceEpoch": {"den": end_time_den, "num": end_time_num},
        "utcOffset": timing_info["start"]["utcOffset"],
    }


def validate_cellset_series_compatibility(input_files):
    """Validate that input files form a valid series.
    :param input_files: list of paths to isxd cell set files
    :return: True if files form a valid cell set series, False otherwise
    """
    # sort input files by their start time
    input_files = _sort_isxd_files_by_start_time(input_files)

    # extract metadata from first input file
    first_cell_set = isx.CellSet.read(input_files[0])
    first_cell_set_metadata = read_isxd_metadata(input_files[0])
    num_cells = first_cell_set.num_cells
    cell_statuses = [
        first_cell_set.get_cell_status(i) for i in range(num_cells)
    ]
    del first_cell_set

    # ensure cell set contains at least 1 cell
    if num_cells < 1:
        raise IdeasError(
            "No cell found in '{0}'.".format(os.path.basename(input_files[0])),
        )

    # initialize series timing and spacing info to those of first cell set
    series_spacing_info = first_cell_set_metadata["spacingInfo"]
    series_timing_info = first_cell_set_metadata["timingInfo"]
    series_timing_info["end"] = compute_end_time(
        first_cell_set_metadata["timingInfo"]
    )

    # loop over all other cell sets and ensure they can be combined into a series
    for f in input_files[1:]:
        # read isxd cell set and corresponding metadata
        isxd_cell_set = isx.CellSet.read(f)
        isxd_metadata = read_isxd_metadata(f)

        # ensure number of cells match across input files
        if isxd_cell_set.num_cells != num_cells:
            raise IdeasError(
                "The number of cells differ across the input files",
            )

        # ensure cell statuses match across input files
        isxd_cell_statuses = [
            isxd_cell_set.get_cell_status(i)
            for i in range(isxd_cell_set.num_cells)
        ]
        if isxd_cell_statuses != cell_statuses:
            raise IdeasError(
                "Individual cell statuses differ across the input files",
            )

        # verify SPACING info
        # validate that the movies have the same spatial dimensions
        if isxd_metadata["spacingInfo"] != series_spacing_info:
            raise IdeasError(
                "All input movies must have the same spatial dimensions",
            )

        # verify TIMING info
        # ensure current item does not temporally overlap with any other item
        item_start_time = (
            isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["num"]
            / isxd_metadata["timingInfo"]["start"]["secsSinceEpoch"]["den"]
        )

        series_end_time = (
            series_timing_info["end"]["secsSinceEpoch"]["num"]
            / series_timing_info["end"]["secsSinceEpoch"]["den"]
        )

        if item_start_time <= series_end_time:
            raise IdeasError(
                "Items that temporally overlap cannot be combined into a series.",
            )

        # validate that the period is nearly identical for all input files
        # TODO

        # validate that all input files share the same microscope identifier
        # TODO: if not, log this as a warning but allow processing to continue
