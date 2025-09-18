import json
import logging
from pathlib import Path

import h5py
from typing import Literal, Optional, Union
import os
import zipfile
import tempfile
import shutil
from ideas.io import cell_set_to_positions

# Add isx import
import isx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from beartype import beartype
from beartype.typing import List, Dict, Tuple, Any
from ideas import measures, plots
from ideas.utils import _set_up_logger
from ideas.exceptions import IdeasError
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from utils.utils import (
    _check_columns_in_df,
    _check_states_valid,
    _get_cellset_data,
    save_optimized_svg,
    DIVISION_THRESHOLD,
)
from utils.plots import (
    SUPTITLE_FONT,
    TITLE_FONT,
    LABEL_FONT,
)

logger = logging.getLogger()
MINIMUM_STATE_LENGTH = 10
statistic_types = Literal["max", "mean", "min"]

# mapper from what the user wants to how
# we compute these stats
statistic_mapper = {
    "max": np.nanmax,
    "mean": np.nanmean,
    "min": np.nanmin,
}

STAT_CORRELATIONS_CSV_NAME = "correlation_statistic_comparison.csv"
AVG_CORRELATIONS_CSV_NAME = "average_correlations.csv"
RAW_CORRELATIONS_H5_NAME = "pairwise_correlation_heatmaps.h5"
RAW_CORRELATIONS_ZIP_NAME = "spatial_analysis_pairwise_correlations.zip"
SPATIAL_CORRELATION_SVG_NAME = "spatial_correlation.svg"
SPATIAL_MAP_SVG_NAME = "spatial_correlation_map.svg"

_set_up_logger()

# @beartype
def _average_correlations(
    correlation_matrix: Dict[str, np.ndarray], colors: List[str]
) -> None:
    """Find average positive and average negative correlations in each state

    :Args
        correlation_matrix: Dictionary mapping state names to correlation matrices
        colors: List of colors for plotting

    :Returns
        None
    """
    # Collect all data first in a list
    data = []
    for state in correlation_matrix.keys():
        corr_matrix = correlation_matrix[state]
        corr_data = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
        pos_corr = np.nanmean(corr_data[corr_data > 0])
        neg_corr = np.nanmean(corr_data[corr_data < 0])
        data.append([state, pos_corr, neg_corr])

    # Create DataFrame all at once
    avg_df = pd.DataFrame(data, columns=["state", "positive", "negative"])

    # Check if we have enough colors for all states
    if len(avg_df) > len(colors):
        raise IdeasError(
            f"Only {len(colors)} colors were provided,"
            f" which is not enough to represent the {len(avg_df)} states."
        )

    # Use only the colors needed for the actual states
    colors = colors[: len(avg_df)]

    # Create figure with more space for labels
    fig, ax = plt.subplots(figsize=(10, 8), nrows=2)

    sns.barplot(
        data=avg_df,
        x="state",
        y="positive",
        ax=ax[0],
        palette=colors,
        hue="state",
        errorbar="sd",
        legend=False,
    )
    sns.barplot(
        data=avg_df,
        x="state",
        y="negative",
        ax=ax[1],
        palette=colors,
        hue="state",
        errorbar="sd",
        legend=False,
    )

    ax[0].set_ylabel("Average Positive Correlation", fontdict=LABEL_FONT)
    ax[1].set_ylabel("Average Negative Correlation", fontdict=LABEL_FONT)
    ax[1].set_xlabel("State", fontdict=LABEL_FONT)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    # Adjust layout with more space for labels
    fig.tight_layout(pad=2.0, h_pad=0.5)

    # Ensure y-axis labels have enough space
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    fig.savefig("average_correlations_preview.svg", dpi=300, transparent=True)

    avg_df.to_csv(AVG_CORRELATIONS_CSV_NAME, index=False)


def cell_set_to_positions_mapping(
    files: Union[str, List[str]],
    *,
    threshold: float = 4.0,
    cell_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Return cell positions as a mapping from cell names to positions

    :Args
        files: Cell set file path(s)
        threshold: Threshold for footprint processing
        cell_names: Optional list of cell names to extract positions for (if None, all cells)

    :Returns
        Dictionary mapping cell names to (x, y) position tuples
    """
    if isinstance(files, str):
        files = [files]

    # Get positions using the core function
    try:
        # Read the first cell set file to get cell names
        cellset = isx.CellSet.read(files[0])

        # Get cell names for all cells
        all_cell_names = [
            cellset.get_cell_name(i) for i in range(cellset.num_cells)
        ]

        # Get positions from core function
        positions_x, positions_y = cell_set_to_positions(
            files, threshold=threshold
        )

        # Initialize positions dictionary
        positions = {}

        # Filter by cell names if provided
        target_cells = cell_names if cell_names is not None else all_cell_names

        cells_found = 0
        for cell_name in target_cells:
            if cell_name in all_cell_names:
                i = all_cell_names.index(cell_name)
                if not np.isnan(positions_x[i]) and not np.isnan(
                    positions_y[i]
                ):
                    positions[cell_name] = (
                        int(positions_x[i]),
                        int(positions_y[i]),
                    )
                    cells_found += 1

    except Exception as e:
        logger.warning(f"Error extracting positions from cell set: {e}")
        positions = {}

    return positions


# @beartype
def correlation_tool(
    *,
    cell_set_files: List[str],
    annotations_file: Optional[List[str]] = None,
    column_name: str = "state",
    state_names: Optional[str] = "",
    state_colors: str = "blue, orange, grey",
    correlation_colors: str = "red, blue",
    statistic: statistic_types = "max",
    include_positions: bool = True,
    correlation_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Analyze pairwise Pearson correlations between neural traces across behavioral states.

    This tool computes correlation matrices between all neurons and analyzes how these
    correlations change across different behavioral states (e.g., exploring familiar vs.
    novel objects). It generates visualization of correlation matrices, statistics about
    correlation distributions, and exports raw correlation data for further analysis.

    :Parameters
        cell_set_files: List of cell set files to analyze
        annotations_file: Optional list of annotation files
        column_name: Column in annotation file that contains state information
        state_names: Comma-separated string of state names to analyze
        state_colors: Comma-separated string of colors for each state
        correlation_colors: Comma-separated string of colors for
        correlation matrix (negative, positive)
        statistic: Type of statistic to compute (max, mean, or min)
        include_positions: Whether to include cell position information in output files
        correlation_threshold: Threshold for correlation strength in spatial map

    :Note
        Frames are labeled in the following ways:
        1. With no annotation file, all frames are analyzed as "all times"
        2. With valid annotations, frames belonging to specified states maintain their labels
        3. With valid annotations, frames not in any specified state are labeled as "not_defined"
    """
    logger.info("State analysis: correlation tool started")
    logger.info("Checking inputs...")

    # Validate statistic parameter
    if statistic not in statistic_mapper:
        raise IdeasError(
            f"Statistic {statistic} not supported. Choose from:"
            f" {', '.join(statistic_mapper.keys())}"
        )

    # Parse state names
    states = []
    if state_names:
        states = [
            state.strip() for state in state_names.split(",") if state.strip()
        ]

    # Log analysis plan
    if states:
        logger.info(f"Will analyze the following states: {states}")
    else:
        logger.info(
            "No specific states provided, will analyze entire recording"
        )

    # Parse color parameters
    colors = state_colors.split(",")
    colors = [color.strip() for color in colors]
    if len(colors) < len(states):
        # Ensure we have enough colors by adding gray for any missing ones
        colors.extend(["grey"] * (len(states) - len(colors)))
    else:
        # Only add gray if we might need it for "not_defined" state
        colors.append("grey")

    corr_colors = correlation_colors
    if "," in correlation_colors:
        corr_colors = correlation_colors.split(",")
        corr_colors = [color.strip() for color in corr_colors]

    # Check if states specified but no annotation file
    if len(states) > 0 and annotations_file is None:
        raise IdeasError(
            """States specified, but no annotation
file provided. This is an error. If you meant to compute
correlations on all times, do not specify state A or B"""
        )

    num_nan_frames_end = 0
    # get annotations & traces
    if (
        annotations_file is None
        or len(annotations_file) == 0
        or annotations_file[0] is None
        or len(states) == 0
    ):
        logger.info(
            """
No annotations provided or no states specified.
Will compute correlation matrix for entire recording"""
        )
        annotations = None
        # Ensure states is just ["all times"] for downstream processing
        states = ["all times"]
    else:
        logger.info(
            f"""Annotations provided, will be used to
 compute correlations in states: {', '.join(states)}"""
        )
        try:
            annotations = pd.read_parquet(annotations_file[0]).copy()
        except Exception as e:
            logger.error(f"Failed to read annotation file: {e}")
            logger.info("Computing correlations for entire recording instead")
            annotations = None
            states = ["all times"]
            # Skip the rest of the annotation processing

        if annotations is not None:
            try:
                _check_columns_in_df(df=annotations, columns=(column_name,))
            except Exception as e:
                logger.error(
                    f"Column '{column_name}' not found in annotation file: {e}"
                )
                logger.info(
                    "Computing correlations for entire recording instead"
                )
                annotations = None
                states = ["all times"]
                # Skip the rest of the annotation processing

        if annotations is not None:
            # detect if there are nan mapped frames at the end of the input annotation
            # this occurs when the behavior annotations ends earlier than the cell
            # set that it's mapped to
            # these frames are dropped from the input annotations and cell traces if they occur
            n = annotations.shape[0]
            na_mask = annotations[column_name] == ""
            for i in range(1, n + 1):
                if na_mask[n - i]:
                    # consecutive nan mapped frame, continue
                    num_nan_frames_end += 1
                else:
                    # real value encountered, end of sequence
                    break

            if num_nan_frames_end:
                logger.info(
                    f"Found {num_nan_frames_end} frames at end of"
                    " input annotations with NaN mapped frames, trimming from analysis"
                )
                annotations = annotations.drop(
                    index=range(n - num_nan_frames_end, n)
                )

            # Check if specified states exist in the annotations
            available_states = annotations[column_name].unique()
            if not any(state in available_states for state in states):
                logger.warning(
                    f"None of the specified states {states} found in annotation data."
                    f" Available states are: {available_states}."
                    " Computing correlations for entire recording instead."
                )
                annotations = None
                states = ["all times"]
            else:
                # clean up annotations to focus on what we need
                annotations, states = _relabel_annotations(
                    df=annotations,
                    states=states,
                    column_name=column_name,
                )

    (
        traces,
        status,
        _,
        _,
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
        cell_names,
    ) = _get_cellset_data(cell_set_files)

    if num_nan_frames_end:
        traces = np.delete(traces, np.arange(n - num_nan_frames_end, n), 0)
        logger.info("Trimming end of traces")
    logger.info("Computing correlations...")
    # dict of raw correlation matrices
    correlation_matrix = _compute_correlation_matrices(
        traces=traces,
        annotations=annotations,
        states=states,
        column_name=column_name,
    )

    # Check if we got any correlation matrices back
    if not correlation_matrix:
        logger.warning(
            "No valid correlation matrices were computed. Check your state names."
        )
        return {}

    # Get the actual states found in correlation matrices (may be different from requested)
    actual_states = list(correlation_matrix.keys())
    logger.info(f"Generated correlation matrices for states: {actual_states}")

    # plot all correlation matrices (will be a preview)
    # plot_correlation_matrices(correlation_matrix, correlation_colors)
    # logger.info("Saving raw correlation matrices...")
    # save raw correlations to h5
    _correlations_to_h5(correlation_matrix=correlation_matrix)
    sort_indices = plot_correlation_matrices(correlation_matrix, corr_colors)

    logger.info("Saving correlation matrices and cell name pairs...")
    # Extract cell positions if requested
    positions = None
    if include_positions and cell_set_files:
        try:
            positions = cell_set_to_positions_mapping(
                cell_set_files[0], cell_names=cell_names
            )
            logger.info(f"Extracted positions for {len(positions)} cells")
        except Exception as e:
            logger.warning(f"Could not extract cell positions: {e}")

    # save raw correlations and cell name pairs to zip file
    _correlations_to_csv(
        correlation_matrix=correlation_matrix,
        cell_names=cell_names,
        sort_indices=sort_indices,
        positions=positions,
    )

    logger.info("Computing average correlations...")
    _average_correlations(correlation_matrix=correlation_matrix, colors=colors)

    # compute max/min/mean correlations
    df = _correlations_to_stat(
        correlation_matrix=correlation_matrix,
        statistic=statistic,
        cell_names=cell_names,
    )

    # When plotting and exporting statistics, use the actual states we computed
    plot_measure_correlations(
        df=df,
        statistic=statistic,
        states=actual_states,
        colors=colors[: len(actual_states)],
    )
    # save to disk
    df.to_csv(STAT_CORRELATIONS_CSV_NAME, index=False)

    # Create spatial correlation analysis if position data is available
    if positions and include_positions:
        logger.info("Creating spatial correlation analysis...")
        # Extract triu data with positions for each state
        triu_data = _extract_triu_data_for_spatial_analysis(
            correlation_matrix=correlation_matrix,
            cell_names=cell_names,
            sort_indices=sort_indices,
            positions=positions,
        )

        # Plot spatial correlations if data is available
        if triu_data:
            # Generate the spatial correlation visualization
            plot_spatial_correlations(triu_data=triu_data)
            logger.info(
                f"Spatial correlation analysis saved to {SPATIAL_CORRELATION_SVG_NAME}"
            )

            # Generate the spatial correlation map
            plot_correlation_spatial_map(
                correlation_matrix=correlation_matrix,
                positions=positions,
                cell_names=cell_names,
                correlation_threshold=correlation_threshold,
                correlation_colors=corr_colors,
                max_lines=500,
            )
            logger.info(
                f"Spatial correlation map saved to {SPATIAL_MAP_SVG_NAME}"
            )
        else:
            logger.warning(
                "Could not create spatial correlation analysis: missing position data"
            )

    stat_key = Path(STAT_CORRELATIONS_CSV_NAME).stem
    avg_key = Path(AVG_CORRELATIONS_CSV_NAME).stem
    raw_h5_key = Path(RAW_CORRELATIONS_H5_NAME).stem
    raw_zip_key = Path(RAW_CORRELATIONS_ZIP_NAME).stem
    spatial_corr_key = Path(SPATIAL_CORRELATION_SVG_NAME).stem
    spatial_map_key = Path(SPATIAL_MAP_SVG_NAME).stem

    # Update metadata with actual states found
    values = {
        "num_cells": len(df),
        "num_states": len(actual_states),
        "states": actual_states,
        "colors": colors[: len(actual_states)],
    }

    stat_values = {
        "num_cells": len(df),
        "num_states": len(actual_states),
        "states": actual_states,
        "colors": colors[: len(actual_states)],
        "statistic": statistic,
    }

    spatial_values = {
        "num_cells": len(df),
        "num_states": len(actual_states),
        "states": actual_states,
    }

    values = [
        {"key": k, "name": k, "value": v} for k, v in values.items()
    ]
    stat_values = [
        {"key": k, "name": k, "value": v} for k, v in stat_values.items()
    ]
    spatial_values = [
        {"key": k, "name": k, "value": v} for k, v in spatial_values.items()
    ]

    # metadata = {
    #     stat_key: stat_values,
    #     avg_key: values,
    #     raw_h5_key: values,
    #     raw_zip_key: values,
    #     spatial_corr_key: spatial_values,
    #     spatial_map_key: spatial_values,
    # }

    # with open("output_metadata.json", "w") as file:
    #     json.dump(metadata, file, indent=2)


    output_data = [
        {
            "file": AVG_CORRELATIONS_CSV_NAME,
            "previews": [
                {
                    "file": "average_correlations_preview.svg",
                    "caption": "Mean positive (top) and negative (bottom) correlations across behavioral states. These barplots show how average correlation values differ between states, providing insight into overall network connectivity patterns."
                }
            ],
            "metadata": values
        },
        {
            "file": RAW_CORRELATIONS_H5_NAME,
            "previews": [
                {
                    "file": "correlation_matrices.svg",
                    "caption": "Pairwise Pearson correlation matrices between neural activity across behavioral states. Neurons are hierarchically clustered to reveal functional organization, with color intensity representing correlation strength from -1 (negative) to +1 (positive)."
                }
            ],
            "metadata": values
        },
        {
            "file": RAW_CORRELATIONS_ZIP_NAME,
            "previews": [
                {
                    "file": "spatial_correlation.svg",
                    "caption": "Relationship between spatial distance and neural correlation across different behavioral states. Left panels show scatter plots of pairwise neural correlation versus physical distance between cell centroids, with linear regression line (gray). Right panels show density plots displaying the distribution of correlation values as a function of distance. This visualization reveals how functional relationships between neurons relate to their spatial arrangement."
                },
                {
                    "file": "spatial_correlation_map.svg",
                    "caption": "Spatial map of neural correlations across behavioral states. Gray dots show all neurons with known positions. Colored lines connect neuron pairs above the correlation threshold, with line color indicating correlation strength and direction. Bold black dots highlight neurons with very strong correlations (|r| > 0.7)."
                }
            ],
            "metadata": spatial_values
        },
        {
            "file": STAT_CORRELATIONS_CSV_NAME,
            "previews": [
                {
                    "file": "correlation_plot.svg",
                    "caption": "Distribution of correlation values across behavioral states. Shows cumulative distribution functions of correlation values and boxplot comparisons between states, illustrating the proportion of neurons with correlations below each threshold."
                }
            ],
            "metadata": stat_values
        }
    ]

    with open("output_data.json", "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info("State Analysis: correlation tool completed")

    # Return an empty dictionary to satisfy the return type
    return {}


# @beartype
def plot_correlation_matrices(
    correlation_matrix: Dict[str, np.ndarray],
    correlation_colors: Union[List[str], str],
    out_file_name: str = "correlation_matrices.svg",
) -> Dict[str, np.ndarray]:
    """Create a plot of correlation matrices side by side.

    :Parameters
        correlation_matrix: Dictionary mapping state names to correlation matrices
        correlation_colors: Colors for correlation matrix (either list of 2 colors or colormap name)
        out_file_name: Output file name for the plot

    :Returns
        Dict[str, np.ndarray]: Dictionary mapping state names to sorting indices
    """
    logger.info("Making figure showing raw correlation matrices...")

    states = correlation_matrix.keys()
    sort_indices = {}

    # Create colormap based on input type
    if isinstance(correlation_colors, list):
        # Create custom colormap - "mycmap" is just an arbitrary name required by the API
        # The actual colormap is stored in the cmap variable and the name isn't used later
        cmap = LinearSegmentedColormap.from_list(
            "mycmap", [correlation_colors[0], "white", correlation_colors[1]]
        )
    elif isinstance(correlation_colors, str):
        cmap = correlation_colors
    else:
        cmap = "bwr"

    # Setup figure with GridSpec for better layout management
    fig = plt.figure(figsize=(6, 5 * len(states) + 1))

    # Define grid with extra row for colorbar
    gs = GridSpec(len(states) + 1, 1, height_ratios=[1] * len(states) + [0.05])

    ax = [fig.add_subplot(gs[i, 0]) for i in range(len(states))]

    plt.suptitle(
        "Pairwise Pearson correlation between neurons",
        y=1.0,
        **SUPTITLE_FONT,
    )

    for i, key in enumerate(states):
        corr_matrix = correlation_matrix[key]
        ax[i].set_title(f"Correlations during {key}", fontdict=TITLE_FONT)

        # check for nans in the correlation matrix
        # and replace them with 0s
        if np.isnan(corr_matrix).any():
            logger.warning(
                f"Found nan values in the correlation matrix for state '{key}', "
                f"replacing them with 0's for plotting"
            )
            corr_matrix = np.nan_to_num(corr_matrix)

        idx, _ = measures._get_new_matrix_index(corr_matrix)
        sort_indices[key] = idx
        plots.plot_correlation_matrix(
            ax[i],
            corr_matrix[idx, :][:, idx],
            cmap=cmap,
            output_format="svg",
        )

    # Create colorbar in the last row
    cax = fig.add_subplot(gs[-1, 0])

    plt.colorbar(
        ax[-1].collections[0],
        cax=cax,
        orientation="horizontal",
        label="Correlation",
        ticks=[-1, 0, 1],
    )
    cax.set_xlabel("Correlation", fontdict=LABEL_FONT)

    # Adjust layout for better spacing
    fig.subplots_adjust(
        hspace=0.5, bottom=0.1, top=0.98, left=0.15, right=0.85
    )
    # Save optimized SVG
    save_optimized_svg(
        fig,
        out_file_name,
        max_size_mb=10,
    )

    return sort_indices


# @beartype
def _correlations_to_csv(
    correlation_matrix: Dict[str, np.ndarray],
    cell_names: List[str] = None,
    sort_indices: Dict[str, np.ndarray] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Save raw correlation matrices and cell name pair correlations to CSV files and zip them.

    Exports two types of files:
    1. Raw correlation matrices with original ordering
    2. Upper triangle (triu) matrices with cell name pairs for memory efficiency

    All files are packaged into a single zip file with a README.

    :Args
        correlation_matrix: Dictionary mapping state names to correlation matrices
        cell_names: List of cell names
        sort_indices: Dictionary mapping state names to sorting indices for each state
        positions: Optional dictionary mapping cell names to (x, y) position coordinates

    :Returns
        None
    """
    zip_filename = RAW_CORRELATIONS_ZIP_NAME
    temp_dir = tempfile.mkdtemp(prefix="corr_matrices_")

    # Add diagnostic info about positions
    if positions:
        logger.info(f"Including positions for {len(positions)} cells")

    try:
        csv_files = []

        # Save each raw correlation matrix to a CSV file
        for state, matrix in correlation_matrix.items():
            # Create a clean filename from the state name
            safe_state = state.replace(" ", "_").replace("/", "_")
            csv_filename = f"correlation_matrix_{safe_state}.csv"
            csv_path = os.path.join(temp_dir, csv_filename)

            # Convert the matrix to a DataFrame with row and column indices
            df = pd.DataFrame(matrix)
            df.to_csv(csv_path)
            csv_files.append((csv_path, csv_filename))

            # If cell names and sort indices are provided, also save cell name pair correlations
            if (
                cell_names is not None
                and sort_indices is not None
                and state in sort_indices
            ):
                idx = sort_indices[state]

                # Sort cell names using the same index
                sorted_cell_names = [cell_names[i] for i in idx]

                # Sort correlation matrix
                sorted_matrix = matrix[idx, :][:, idx]

                # Perform verification checks more efficiently
                _verify_correlation_matrix(
                    sorted_matrix, matrix, idx, sorted_cell_names
                )

                # Upper triangle only (more memory-efficient)
                rows_triu = []
                n = len(sorted_cell_names)
                positions_included = 0

                for i in range(n):
                    for j in range(i + 1, n):
                        # Get indices in the original unsorted matrix
                        orig_i = idx[i]
                        orig_j = idx[j]

                        # Get original cell names
                        cell_i = cell_names[orig_i]
                        cell_j = cell_names[orig_j]

                        # Create a dictionary with cell names and correlation
                        row = {
                            "cell_name_1": cell_i,
                            "cell_name_2": cell_j,
                            "correlation": sorted_matrix[i, j],
                        }

                        # Add position information if available - use original cell names for lookup
                        if positions:
                            if cell_i in positions and cell_j in positions:
                                positions_included += 1
                                x1, y1 = positions[cell_i]
                                x2, y2 = positions[cell_j]

                                # Calculate Euclidean distance
                                distance = np.sqrt(
                                    (x1 - x2) ** 2 + (y1 - y2) ** 2
                                )

                                # Add position data to row
                                row.update(
                                    {
                                        "centroid_x1": x1,
                                        "centroid_y1": y1,
                                        "centroid_x2": x2,
                                        "centroid_y2": y2,
                                        "distance": distance,
                                    }
                                )

                        rows_triu.append(row)

                # Save upper triangle with cell names to the temp directory
                triu_filename = f"correlation_matrix_{safe_state}_triu.csv"
                triu_path = os.path.join(temp_dir, triu_filename)
                triu_df = pd.DataFrame(rows_triu)

                # Log position inclusion stats
                if positions:
                    positions_included = sum(
                        1 for row in rows_triu if "centroid_x1" in row
                    )

                    # Verify that the output dataframe actually has the position columns
                    if (
                        "centroid_x1" not in triu_df.columns
                        and positions_included > 0
                    ):
                        logger.warning(
                            "Position columns missing from output data"
                        )

                triu_df.to_csv(triu_path, index=False)
                csv_files.append((triu_path, triu_filename))

        # Create a zip file containing all CSV files
        logger.info(
            f"Creating zip file with all correlation matrices: {zip_filename}"
        )
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path, arcname in csv_files:
                zipf.write(file_path, arcname=arcname)

        # Add a README to the zip file explaining the contents
        _add_readme_to_zip(temp_dir, zip_filename, positions is not None)

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def _verify_correlation_matrix(
    sorted_matrix: np.ndarray,
    original_matrix: np.ndarray,
    indices: np.ndarray,
    cell_names: List[str],
) -> None:
    """Verify that the sorted correlation matrix is consistent with the original matrix.

    Checks:
    1. Diagonal values should be near zero
    2. Values should be preserved after sorting

    :Args
        sorted_matrix: The sorted correlation matrix
        original_matrix: The original correlation matrix before sorting
        indices: The indices used for sorting
        cell_names: List of cell names corresponding to rows/columns

    :Returns
        None
    """
    # Verify diagonal values
    for i in range(len(cell_names)):
        if abs(sorted_matrix[i, i]) > DIVISION_THRESHOLD:
            logger.warning(
                f"Unexpected diagonal value for cell {cell_names[i]}: {sorted_matrix[i, i]}"
            )

    # Verify that sorting preserves correlation values (check a sample of values)
    n = len(indices)
    num_checks = min(100, n * n)

    if n <= 10:
        for i in range(n):
            for j in range(n):
                orig_value = original_matrix[indices[i], indices[j]]
                sorted_value = sorted_matrix[i, j]
                if abs(orig_value - sorted_value) > DIVISION_THRESHOLD:
                    logger.warning(
                        "Correlation value mismatch after"
                        f" sorting: original={orig_value}, sorted={sorted_value}"
                    )
    else:
        rng = np.random.RandomState(42)
        for _ in range(num_checks):
            i, j = rng.randint(0, n, 2)
            orig_value = original_matrix[indices[i], indices[j]]
            sorted_value = sorted_matrix[i, j]
            if abs(orig_value - sorted_value) > DIVISION_THRESHOLD:
                logger.warning(
                    "Correlation value mismatch"
                    f" after sorting: original={orig_value}, sorted={sorted_value}"
                )


def _add_readme_to_zip(
    temp_dir: str, zip_filename: str, include_positions: bool
) -> None:
    """Add a README file to the zip archive explaining its contents.

    :Args
        temp_dir: Temporary directory where files are stored
        zip_filename: Name of the zip file to add the README to
        include_positions: Whether to include position information in the README

    :Returns
        None
    """
    readme_path = os.path.join(temp_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("This ZIP file contains the following correlation data:\n\n")
        f.write(
            "1. correlation_matrix_*.csv: Raw correlation matrices for each state\n"
        )
        f.write(
            "2. correlation_matrix_*_triu.csv:"
            "Upper triangle of correlation matrices with cell name pairs\n"
        )
        if include_positions:
            f.write(
                "The position columns (centroid_x1, centroid_y1, centroid_x2, centroid_y2)"
                "represent the spatial coordinates of each cell in the pair.\n"
                "Distance column represents the Euclidean distance between cell pairs.\n"
            )

    with zipfile.ZipFile(zip_filename, "a", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(readme_path, arcname="README.txt")


def _compute_correlation_matrices(
    *,
    traces: np.ndarray,
    annotations: Optional[pd.DataFrame] = None,
    states: List[str] = None,
    column_name: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Compute correlation matrices for certain states, and for
    the entire duration of the recording.

    :Parameters
        traces: Array of neural traces
        annotations: DataFrame of behavioral annotations
        states: List of state names to analyze
        column_name: Column in annotations DataFrame that contains state information
    :Returns
        Dict[str, np.ndarray]: Dictionary mapping state names to correlation matrices
    :Raises
        IdeasError: If ISXD data and annotations do not match in length
    """
    correlation_matrix = dict()

    if annotations is None or column_name is None:
        # Compute across all times
        correlation_matrix["all times"] = measures.correlation_matrix(
            traces,
            fill_diagonal=0.0,
        )
        return correlation_matrix

    # at this point we are using annotations,
    # so check that they match

    if len(annotations) != traces.shape[0]:
        raise IdeasError(
            f"""ISXD data and annotations do not match.
The length of the ISXD data is {traces.shape[0]},
but the length of the annotations data is {len(annotations)}"""
        )

    # Filter states to only include those that actually exist in the annotation data
    available_states = annotations[column_name].unique()
    valid_states = [state for state in states if state in available_states]

    if len(valid_states) < len(states):
        invalid_states = [
            state for state in states if state not in available_states
        ]
        logger.warning(
            f"Skipping states not found in annotation data: {invalid_states}"
        )

    # If no valid states left, warn user
    if not valid_states:
        logger.warning(
            "None of the specified states were found in annotation data!"
        )

    for state in valid_states:
        traces_in_state = traces[annotations[column_name] == state, :]

        if traces_in_state.shape[0] < MINIMUM_STATE_LENGTH:
            logger.warning(
                f"State '{state}' has fewer than {MINIMUM_STATE_LENGTH} frames. "
                "Correlation calculation may be unreliable."
            )

        correlation_matrix[state] = measures.correlation_matrix(
            traces_in_state,
            fill_diagonal=0.0,
        )

    return correlation_matrix


# @beartype
def plot_measure_correlations(
    *,
    df: pd.DataFrame,
    statistic: statistic_types,
    states: List[str],
    colors: Optional[List[str]] = None,
) -> None:
    """Make figure showing max/mean/min correlations in states

    :Args
        df: DataFrame containing correlation statistics
        statistic: Type of statistic plotted (max, mean, or min)
        states: List of state names
        colors: List of colors for each state

    :Returns
        None
    """
    logger.info(f"Making figure showing {statistic} correlations...")

    single_state = len(states) == 1
    if single_state:
        # no states, we're plotting correlation at all times
        fig, ax = plt.subplots(figsize=(5, 5))

        # to ensure compatibility with rest of code
        ax = [ax]

    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Check if we have enough colors for all states
    if len(states) > len(colors):
        raise IdeasError(
            f"Only {len(colors)} colors were provided,"
            f" which is not enough to represent the {len(states)} states."
        )

    # Use only the colors needed for the actual states
    colors = colors[: len(states)]

    # plot CDFs of max correlations for each state
    sns.ecdfplot(
        data=df,
        palette=colors,
        ax=ax[0],
    )
    ax[0].set_xlim((-1, 1))
    ax[0].set_xticks([-1, -0.5, 0, 0.5, 1])

    # clean formatting of the plot
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_ylabel(
        f"Proportion of cells with {statistic}correlation <= x",
        fontdict=LABEL_FONT,
    )
    ax[0].set_xlabel("Correlation", fontdict=LABEL_FONT)

    # Set consistent font for tick labels
    ax[0].tick_params(
        axis="both", which="major", labelsize=LABEL_FONT["fontsize"]
    )

    # Only create boxplot when we have multiple states or enough room
    if not single_state:
        # For a more reliable boxplot with correct labels, reshape data first
        plot_data = pd.melt(
            df,
            id_vars=["name"],
            value_vars=states,
            var_name="State",
            value_name=f"{statistic} correlation",
        )

        # plot boxplot with the melted data frame
        sns.boxplot(
            data=plot_data,
            x="State",
            y=f"{statistic} correlation",
            hue="State",
            ax=ax[1],
            palette=colors,
            fill=False,
            linewidth=3,
            order=states,
            legend=False,
        )

        ax[1].set_ylabel(f"{statistic} correlations", fontdict=LABEL_FONT)
        ax[1].set_xlabel("State", fontdict=LABEL_FONT)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)

        # Set consistent font for tick labels and rotate x-axis labels to prevent overlap
        ax[1].tick_params(
            axis="both", which="major", labelsize=LABEL_FONT["fontsize"]
        )
        ax[1].tick_params(axis="x", rotation=45)

    out_file_name = "correlation_plot.svg"

    # Adjust layout to prevent text cutoff
    fig.tight_layout()

    fig.savefig(
        out_file_name,
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )


# @beartype
def _correlations_to_stat(
    correlation_matrix: Dict[str, np.ndarray],
    statistic: statistic_types,
    cell_names: List[str],
) -> pd.DataFrame:
    """Convert a dictionary of raw correlation matrices to a table
    with statistics of correlations (max/min/mean)

    :Args
        correlation_matrix: Dictionary mapping state names to correlation matrices
        statistic: Type of statistic to compute (max, mean, or min)
        cell_names: List of cell names

    :Returns
        DataFrame containing correlation statistics for each cell
    """
    df = dict()
    df["name"] = cell_names

    for state in correlation_matrix.keys():
        df[state] = statistic_mapper[statistic](
            correlation_matrix[state], axis=0
        )

    df = pd.DataFrame(df)

    return df


# @beartype
def _correlations_to_h5(correlation_matrix: dict) -> None:
    """Save a dict with raw correlation matrices to a h5 file"""
    filename = RAW_CORRELATIONS_H5_NAME

    logger.info(f"Saving raw correlation to {filename}.")
    with h5py.File(filename, "w") as f:
        for state in correlation_matrix.keys():
            f.create_dataset(state, data=correlation_matrix[state])


# @beartype
def _relabel_annotations(
    *,
    df: pd.DataFrame,
    states: Optional[List[str]],
    column_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Make sure states are appropriately labelled
    for the different cases this tool can be called under

    :Args
        df: DataFrame of behavioral annotations
        states: List of state names
        column_name: Column in DataFrame that contains state information

    :Returns
        Tuple of (modified DataFrame, updated list of states)

    :Note
        Frames not in any specified state are labeled as "not_defined"
        if there are enough of them (>= MINIMUM_STATE_LENGTH)
    """
    np.random.seed(0)

    # safety checks
    for state in states:
        if state is None:
            continue

        _check_states_valid(
            states=[state],
            behavior=df,
            column_name=column_name,
        )

        if np.sum(df[column_name] == state) < MINIMUM_STATE_LENGTH:
            raise IdeasError(
                f"""
In this session, there are fewer than {MINIMUM_STATE_LENGTH}
frames where the animal is in state: `{state}`.
We cannot accurately estimate correlations here. """,
            )

    # Create a mask for frames not in any specified state
    not_in_any_state = np.ones(len(df), dtype=bool)
    for state in states:
        not_in_any_state[df[column_name] == state] = False

    # Count frames that don't belong to any specified state
    num_unlabeled_frames = np.sum(not_in_any_state)
    logger.info(
        f"Found {num_unlabeled_frames} frames not in any specified state"
    )

    # If there are enough unlabeled frames, create a "not_defined" state for them
    if num_unlabeled_frames >= MINIMUM_STATE_LENGTH:
        logger.info(
            f"Adding 'not_defined' state for {num_unlabeled_frames} unlabeled frames"
        )

        # Ensure column is categorical before adding new category
        if not isinstance(df[column_name].dtype, pd.CategoricalDtype):
            df[column_name] = df[column_name].astype("category")

        # Check if "not_defined" category already exists
        if "not_defined" not in df[column_name].cat.categories:
            # set "not_defined" as a category for the state column
            df[column_name] = df[column_name].cat.add_categories("not_defined")

        df.loc[not_in_any_state, column_name] = "not_defined"


        states.append("not_defined")
    else:
        logger.info(
            f"Too few unlabeled frames ({num_unlabeled_frames} < {MINIMUM_STATE_LENGTH}),"
            " ignoring them"
        )

    return df, states


# @beartype
def plot_spatial_correlations(
    triu_data: Dict[str, pd.DataFrame],
    out_file_name: str = SPATIAL_CORRELATION_SVG_NAME,
) -> None:
    """Create a visualization showing the relationship
    between spatial distance and correlation values.
    """
    # Determine how many states we have
    states = list(triu_data.keys())
    n_states = len(states)

    # Create figure with space for multiple subplots
    fig = plt.figure(figsize=(8, 8 + n_states * 1.5))
    gs = GridSpec(n_states + 1, 2, height_ratios=[2] * n_states + [0.05])

    # Main title
    plt.suptitle(
        "Relationship between spatial distance and neural correlation",
        y=0.99,
        x=0.5,
        ha="center",
        **SUPTITLE_FONT,
    )

    # Prepare for plotting
    scatter_axes = []
    density_axes = []

    # Track min/max distance for consistent x-axis limits
    min_distance = float("inf")
    max_distance = 0

    # Process data for each state and create initial plots
    for i, state in enumerate(states):
        df = triu_data[state]

        # Skip if missing position data
        if "distance" not in df.columns:
            logger.warning(
                f"No position data found for state {state}, skipping spatial analysis"
            )
            continue

        # Update min/max distance
        min_distance = min(min_distance, df["distance"].min())
        max_distance = max(max_distance, df["distance"].max())

        # Create scatter plot
        ax_scatter = fig.add_subplot(gs[i, 0])
        scatter_axes.append(ax_scatter)

        # Create density plot (2D histogram)
        ax_density = fig.add_subplot(gs[i, 1])
        density_axes.append(ax_density)

        # Store plotting data for second pass
        triu_data[state] = df

    # Second pass - create actual plots with consistent scales
    for i, state in enumerate(states):
        if i >= len(scatter_axes):
            continue

        df = triu_data[state]

        # Skip if missing position data
        if "distance" not in df.columns:
            continue

        # Scatter plot with trend line
        ax_scatter = scatter_axes[i]
        sns.scatterplot(
            data=df,
            x="distance",
            y="correlation",
            alpha=0.3,
            s=10,
            ax=ax_scatter,
        )

        # Add trend line
        sns.regplot(
            data=df,
            x="distance",
            y="correlation",
            scatter=False,
            ax=ax_scatter,
            color="gray",
            line_kws={"linewidth": 2},
        )

        # Format scatter plot
        ax_scatter.set_title(
            f"Correlation vs Distance ({state})", fontdict=TITLE_FONT
        )
        ax_scatter.set_ylabel("Correlation", fontdict=LABEL_FONT)
        ax_scatter.set_xlabel("Distance (pixels)", fontdict=LABEL_FONT)
        ax_scatter.set_xlim(max(0, min_distance - 5), max_distance + 5)
        ax_scatter.set_ylim(-1.05, 1.05)
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)

        # Calculate Pearson correlation between distance and correlation
        pearson_r = df["distance"].corr(df["correlation"])
        # Add text annotation showing the correlation value
        ax_scatter.text(
            0.05,
            0.95,
            f"Spatial-correlation r = {pearson_r:.3f}",
            transform=ax_scatter.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # 2D Density plot (hexbin)
        ax_density = density_axes[i]
        hb = ax_density.hexbin(
            df["distance"],
            df["correlation"],
            gridsize=30,
            cmap="viridis",
            mincnt=1,
        )

        # Format density plot
        ax_density.set_title(
            f"Density of Correlation vs Distance ({state})",
            fontdict=TITLE_FONT,
        )
        ax_density.set_xlabel("Distance (pixels)", fontdict=LABEL_FONT)
        ax_density.set_ylabel("Correlation", fontdict=LABEL_FONT)
        ax_density.set_xlim(max(0, min_distance - 5), max_distance + 5)
        ax_density.set_ylim(-1.05, 1.05)
        ax_density.spines["top"].set_visible(False)
        ax_density.spines["right"].set_visible(False)

    # Create shared colorbar at the bottom
    if density_axes:
        cax = fig.add_subplot(gs[-1, :])
        cbar = plt.colorbar(
            hb, cax=cax, orientation="horizontal", label="Count"
        )
        cbar.set_label("Count", fontdict=LABEL_FONT)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save figure using optimized SVG function
    save_optimized_svg(fig, out_file_name)


# @beartype
def _extract_triu_data_for_spatial_analysis(
    correlation_matrix: Dict[str, np.ndarray],
    cell_names: List[str] = None,
    sort_indices: Dict[str, np.ndarray] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Extract triangular matrix data with spatial information for visualization.

    :Args
        correlation_matrix: Dictionary mapping state names to correlation matrices
        cell_names: List of cell names
        sort_indices: Dictionary mapping state names to sorting indices
        positions: Dictionary mapping cell_names to (x, y) position coordinates

    :Returns
        Dictionary mapping state names to DataFrames with triu data
    """
    triu_data = {}

    # Skip if missing required data
    if not positions or not cell_names or not sort_indices:
        logger.warning(
            "Missing required data (positions, cell names, or sort indices) for spatial analysis"
        )
        return triu_data

    for state, matrix in correlation_matrix.items():
        if state not in sort_indices:
            continue

        idx = sort_indices[state]
        sorted_cell_names = [cell_names[i] for i in idx]
        sorted_matrix = matrix[idx, :][:, idx]

        # Upper triangle only
        rows_triu = []
        n = len(sorted_cell_names)

        for i in range(n):
            for j in range(i + 1, n):
                orig_i = idx[i]
                orig_j = idx[j]

                cell_i = cell_names[orig_i]
                cell_j = cell_names[orig_j]

                row = {
                    "cell_name_1": cell_i,
                    "cell_name_2": cell_j,
                    "correlation": sorted_matrix[i, j],
                }

                # Add position information if available
                if cell_i in positions and cell_j in positions:
                    x1, y1 = positions[cell_i]
                    x2, y2 = positions[cell_j]

                    # Calculate Euclidean distance
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                    # Add position data to row
                    row.update(
                        {
                            "centroid_x1": x1,
                            "centroid_y1": y1,
                            "centroid_x2": x2,
                            "centroid_y2": y2,
                            "distance": distance,
                        }
                    )

                rows_triu.append(row)

        # Convert to DataFrame
        if rows_triu:
            triu_data[state] = pd.DataFrame(rows_triu)

    return triu_data


# @beartype
def plot_correlation_spatial_map(
    correlation_matrix: Dict[str, np.ndarray],
    positions: Dict[str, Tuple[float, float]],
    cell_names: List[str],
    correlation_threshold: float = 0.5,
    correlation_colors: Union[List[str], str] = "red, blue",
    max_lines: int = 1000,
    out_file_name: str = SPATIAL_MAP_SVG_NAME,
) -> None:
    """Create a spatial map of cells with lines showing correlation strength."""
    # Create colormap based on input type (same logic as plot_correlation_matrices)
    if isinstance(correlation_colors, list):
        # Create custom colormap - "mycmap" is just an arbitrary name required by the API
        # The actual colormap is stored in the cmap variable and the name isn't used later
        cmap = LinearSegmentedColormap.from_list(
            "mycmap", [correlation_colors[0], "white", correlation_colors[1]]
        )
    elif isinstance(correlation_colors, str):
        # Check if it's a comma-separated string of colors
        if "," in correlation_colors:
            colors = [color.strip() for color in correlation_colors.split(",")]
            # Create custom colormap - "mycmap" is just an arbitrary name required by the API
            # The actual colormap is stored in the cmap variable and the name isn't used later
            cmap = LinearSegmentedColormap.from_list(
                "mycmap", [colors[0], "white", colors[1]]
            )
        else:
            # It's a matplotlib colormap name
            cmap = correlation_colors
    else:
        logger.info(
            "Correlation colors parameter not recognized, defaulting to 'coolwarm' colormap"
        )
        cmap = "coolwarm"

    # Check if we have enough position data
    cells_with_positions = [name for name in cell_names if name in positions]
    num_cells_with_positions = len(cells_with_positions)

    if num_cells_with_positions < 5:
        logger.warning(
            f"Only {num_cells_with_positions} cells have position data, skipping spatial map"
        )
        return

    # Setup figure with multiple panels for different states
    states = list(correlation_matrix.keys())
    n_states = len(states)

    # Adjust figure size to be more compact, narrower width for equal aspect plots
    # Width: 6 inches works well for spatial plots that need equal aspect ratio
    # Height: 4.5 inches per state provides enough space for scatter plots + labels
    fig = plt.figure(figsize=(6, 4.5 * n_states))

    # Use single column layout like plot_correlation_matrices, with shared colorbar at bottom
    # Make panel spacing dynamic too for consistency
    panel_spacing = 0.3 + (0.1 / n_states)
    gs = GridSpec(
        n_states + 1,
        1,
        height_ratios=[1] * n_states + [0.05],
        hspace=panel_spacing,
    )

    # Add title using suptitle with dynamic positioning based on number of states
    title_y = 1.0 - (0.01 + 0.005 * n_states)
    plt.suptitle(
        "Spatial Map of Neural Correlations\n"
        f"Showing correlations with |r| ≥ {correlation_threshold}",
        y=title_y,
        **SUPTITLE_FONT,
    )

    # Store line collections for shared colorbar
    line_collections = []

    # Process each state
    for i, state in enumerate(states):
        # Create axis for this state
        ax = fig.add_subplot(gs[i, 0])
        ax.set_title(f"{state}", fontdict=TITLE_FONT)

        # Extract cell coordinates
        xy_coords = np.array(
            [positions[name] for name in cells_with_positions]
        )

        # Plot all cells as dots
        ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c="gray", s=10, alpha=0.5)

        # Create the connections (lines) between correlated cell pairs
        lines = []
        line_colors = []

        # Extract high correlation pairs
        high_corr_pairs = []

        # For each pair of cells
        for idx1, name1 in enumerate(cells_with_positions):
            for idx2, name2 in enumerate(cells_with_positions):
                if idx1 >= idx2:
                    continue

                # Get actual indices in the correlation matrix
                try:
                    orig_idx1 = cell_names.index(name1)
                    orig_idx2 = cell_names.index(name2)

                    # Get correlation value
                    corr_value = correlation_matrix[state][
                        orig_idx1, orig_idx2
                    ]

                    # Only include if absolute correlation is above threshold
                    if abs(corr_value) >= correlation_threshold:
                        high_corr_pairs.append(
                            (idx1, idx2, corr_value, abs(corr_value))
                        )
                except (ValueError, IndexError):
                    # Skip if cells aren't found in the correlation matrix
                    continue

        # Sort by absolute correlation (highest first) and limit the number of lines
        high_corr_pairs.sort(key=lambda x: x[3], reverse=True)
        if len(high_corr_pairs) > max_lines:
            high_corr_pairs = high_corr_pairs[:max_lines]

        # Create lines and colors
        for idx1, idx2, corr_value, _ in high_corr_pairs:
            # Get coordinates for the cell pair
            coord1 = xy_coords[idx1]
            coord2 = xy_coords[idx2]

            # Add the line segment
            lines.append([coord1, coord2])

            # Set color based on correlation (red for positive, blue for negative)
            line_colors.append(corr_value)

        # Create LineCollection for efficient plotting
        if lines:
            lc = LineCollection(
                lines,
                cmap=cmap,
                norm=plt.Normalize(-1, 1),
                linewidths=1.5,
                alpha=0.6,
                zorder=1,
            )
            lc.set_array(np.array(line_colors))
            ax.add_collection(lc)
            line_collections.append(lc)
        else:
            # Create a dummy collection for consistent colorbar
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=-1, vmax=1)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            line_collections.append(sm)

        # Highlight cells with strong connections
        # Find cells that participate in high-correlation pairs
        highlighted_cells = set()
        for idx1, idx2, corr_value, _ in high_corr_pairs:
            if abs(corr_value) > 0.7:
                highlighted_cells.add(idx1)
                highlighted_cells.add(idx2)

        # Highlight these cells with a larger marker
        if highlighted_cells:
            highlight_coords = xy_coords[list(highlighted_cells)]
            ax.scatter(
                highlight_coords[:, 0],
                highlight_coords[:, 1],
                c="black",
                s=30,
                alpha=0.8,
                zorder=2,
            )

        # Set axis properties
        if i == n_states - 1:
            ax.set_xlabel("X position (pixels)", fontdict=LABEL_FONT)
        ax.set_ylabel("Y position (pixels)", fontdict=LABEL_FONT)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    # Create shared colorbar at the bottom (same approach as plot_correlation_matrices)
    if line_collections:
        cax = fig.add_subplot(gs[-1, 0])
        plt.colorbar(
            line_collections[0],
            cax=cax,
            orientation="horizontal",
            label="Correlation",
            ticks=[-1, 0, 1],
        )
        cax.set_xlabel("Correlation", fontdict=LABEL_FONT)

    # Adjust layout for better spacing with dynamic top margin based on number of states
    top_margin = 0.95 - (0.01 + 0.005 * n_states)
    fig.subplots_adjust(bottom=0.15, top=top_margin, left=0.20, right=0.80)

    # Save figure using optimized SVG function
    save_optimized_svg(fig, out_file_name)
