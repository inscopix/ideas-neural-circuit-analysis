import json
import logging
from pathlib import Path

import numpy as np
from beartype.typing import List, Optional
from ideas.exceptions import IdeasError
from ideas.analysis.io import (
    cell_set_to_contours,
)

from utils.algorithms import (
    _calculate_correlation,
    _calculate_population_activity,
)
from utils.plots import (
    _plot_timecourse,
    plot_trace_preview,
)
from utils.utils import (
    _bin_data,
    _fractional_change_2D_array,
    _get_cellset_data,
    _norm_2D_array,
    _parse_string_to_tuples,
    _standardize_2D_array,
    _standardize_to_first_epoch,
    event_set_to_events,
    _validate_epochs_param,
    _redefine_epochs,
)
from utils.validation import (
    _check_epochs_valid,
    _check_num_epochs,
    _validate_epoch_name_strings,
    _validate_events,
)

from ideas.tools.types import IdeasFile
from ideas.tools import log
from ideas.tools import outputs

logger = log.get_logger()


def run(
    *,
    cell_set_files: List[str],
    event_set_files: Optional[List[str]] = None,
    define_epochs_by: str,
    epoch_names: str,
    epochs: Optional[str] = None,
    epoch_colors: str,
    bin_size: Optional[float] = None,
    trace_scale_method: Optional[str] = "none",
    event_scale_method: Optional[str] = "none",
    sort_by_time: Optional[bool] = True,
    tolerance: Optional[float] = 1e-4,
    heatmap: Optional[tuple] = "auto",
):
    """Compare the activity of neurons between any number of epochs. The tool
    requires a cell set file and an event set file for each epoch. The tool
    will compare the activity of all nuerons between each epoch and run an
    ANOVA test to determine if the activity is significantly different. The
    tool will also calculate the modulation index for each neuron and determine
    if the neuron is up-modulated, down-modulated, or non-modulated compared
    pairwise between epochs. This tool will output a plot of the average
    timecourse of activity for each epoch, a matrix of scatter plots of the
    activity between epochs, a box plot of the activity between epochs with
    ANOVA, and a matrix of cell maps showing the modulation index of each
    neuron.

    This tool will compute these for traces and events if provided.

    :Raises
        IdeasError: If there is an error in reading the cell set files or event set files.

    :Returns
        None

        - A plot of the average timecourse of activity for each epoch as "Traces_Timecourse.svg".
        - A matrix of scatter plots of the activity between epochs
          as "Population_Activity_Overview.svg".
        - A box plot of the activity between epochs with ANOVA as "Population_Activity.svg".
        - A matrix of cell maps showing the modulation index of each neuron as "Modulation_Map.svg".
    :Saves
        - A CSV file of the modulation data as "modulation.csv".
    """
    try:
        (
            traces,
            status,
            boundaries,
            period,
            num_accepted_cells,
            num_undecided_cells,
            num_rejected_cells,
            cell_names,
        ) = _get_cellset_data(
            cell_set_files=cell_set_files,
            sort_by_time=sort_by_time,
            tolerance=tolerance,
            concat=True,  # Explicitly set concatenation (consistent with state_epoch tool)
        )
    except IdeasError as err:
        logger.error(err)
        raise err

    # automatically create or modify epochs
    epochs = _validate_epochs_param(
        epochs=epochs,
        define_epochs_by=define_epochs_by,
    )
    epochs = _redefine_epochs(
        define_epochs_by=define_epochs_by,
        epochs=epochs,
        boundaries=boundaries,
    )

    # get the epochs from the input string
    try:
        epochs = _parse_string_to_tuples(epochs)
    except ValueError as e:
        raise IdeasError(
            "The specified Epoch Periods could not be parsed. "
            "Verify that the epoch periods are specified as comma-separated"
            " tuples of the form (start, end) "
            "and that no comma is missing between and within the tuples."
        ) from e

    epoch_colors = epoch_colors.split(",")
    epoch_colors = [color.strip() for color in epoch_colors]
    if isinstance(heatmap, str):
        heatmap = heatmap.lower()
    if heatmap != "auto" and heatmap is not None:
        # make sure it is not already a list
        if isinstance(heatmap, str):
            heatmap = heatmap.split(",")
            heatmap = [val.strip().strip("()") for val in heatmap]
        # Unify
    epoch_names = _validate_epoch_name_strings(epoch_names)

    _check_num_epochs(
        epochs=epochs,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
    )
    _check_epochs_valid(
        traces=traces,
        epochs=epochs,
        period=period,
    )

    # preview the traces
    plot_trace_preview(
        traces=traces,
        epochs=epochs,
        boundaries=boundaries,
        period=period,
        epoch_colors=epoch_colors,
        epoch_names=epoch_names,
    )

    # Start processing the data

    # bin the data
    original_shape = traces.shape
    traces = _bin_data(traces, bin_size, period)
    if bin_size is not None and bin_size > 0:
        functional_period = bin_size
    else:
        functional_period = period
    # now rescale the data if appropriate
    if trace_scale_method == "normalize":
        traces = _norm_2D_array(traces)
        ylabel = "Normalized Trace Activity"
    elif trace_scale_method == "standardize":
        traces = _standardize_2D_array(traces)
        ylabel = "Standardized Trace Activity"
    elif trace_scale_method == "fractional_change":
        traces = _fractional_change_2D_array(traces, epochs, functional_period)
        ylabel = "Fractional Change in Trace Activity"
    elif trace_scale_method == "standardize_epoch":
        traces = _standardize_to_first_epoch(traces, epochs, functional_period)
        ylabel = "Standardized Trace Activity"
    else:
        ylabel = "Trace Activity"

    # get contours for non-rejected neurons
    try:
        x, y = cell_set_to_contours(
            files=cell_set_files,
            sort_by_time=sort_by_time,
            tolerance=tolerance,
        )
    except IdeasError as err:
        logger.error(err)
        raise err

    if num_accepted_cells > 0:
        cell_status_filter = "accepted"
    elif num_undecided_cells > 0:
        cell_status_filter = "undecided"
    else:
        logger.warning(
            "No valid cells found. Using 'accepted' as fallback filter."
        )
        cell_status_filter = (
            "accepted"  # Fallback to prevent undefined variable
        )

    x = [cx for cx, stat in zip(x, status) if stat == cell_status_filter]
    y = [cx for cx, stat in zip(y, status) if stat == cell_status_filter]

    # plot average activity between epochs using single cell metrics
    _calculate_population_activity(
        data_name="Traces",
        data=traces,
        epochs=epochs,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        period=functional_period,
        x=x,
        y=y,
        ylabel=ylabel,
        scale_method=trace_scale_method,
        cell_names=cell_names,
    )
    # plot time course of activity for the population.

    _plot_timecourse(
        data_name="Traces",
        data=traces,
        epochs=epochs,
        boundaries=boundaries,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        period=functional_period,
        raw_data=None,
        scale_method=trace_scale_method,
        heatmap=heatmap,
    )
    _calculate_correlation(
        data_name="Traces",
        data=traces,
        epochs=epochs,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        period=functional_period,
        output_format="svg",
    )
    # Create metadata for trace output
    trace_timecourse_key = Path("Traces_timecourse_data.npy").stem
    trace_activity_key = Path("Traces_activity_data.csv").stem
    trace_correlation_key = Path("Traces_correlation_data.npy").stem

    trace_timecourse_values = {
        "epoch_lengths": [epoch[1] - epoch[0] for epoch in epochs],
        "epoch_names": epoch_names,
        "epoch_colors": epoch_colors,
        "sampling_rate": 1 / functional_period,
        "scale_method": trace_scale_method,
        "num_cells": traces.shape[1],
    }
    trace_activity_values = {
        "epoch_lengths": [epoch[1] - epoch[0] for epoch in epochs],
        "epoch_names": epoch_names,
        "epoch_colors": epoch_colors,
        "num_cells": traces.shape[1],
        "scale_method": trace_scale_method,
    }
    trace_correlation_values = {
        "epoch_lengths": [epoch[1] - epoch[0] for epoch in epochs],
        "epoch_names": epoch_names,
        "epoch_colors": epoch_colors,
        "num_cells": traces.shape[1],
    }
    metadata = {
        trace_timecourse_key: trace_timecourse_values,
        trace_activity_key: trace_activity_values,
        trace_correlation_key: trace_correlation_values,
    }

    # PROCESS EVENTS IF PROVIDED
    if event_set_files is not None and len(event_set_files) > 0:
        logger.info("Event set files provided. Processing events.")
        try:
            offsets, amplitudes = event_set_to_events(event_set_files)
        except IdeasError as err:
            logger.info("Error processing event set files.")
            logger.error(err)
            raise err

        # Keep a copy of the original offsets
        original_offsets = list(offsets)  # Ensure it's a copy
        valid_events = False
        final_offsets = None
        final_offset_indices = None

        # --- Attempt 1: Validate with unfiltered offsets ---
        logger.info("Attempting event validation with unfiltered offsets.")
        try:
            unfiltered_indices = [
                np.array([int(offset / period) for offset in cell])
                for cell in original_offsets
            ]
            if _validate_events(traces, original_offsets, unfiltered_indices):
                logger.info("Unfiltered event validation successful.")
                valid_events = True
                final_offsets = original_offsets
                final_offset_indices = unfiltered_indices
            else:
                logger.warning("Unfiltered event validation failed.")
        except Exception as e:
            logger.warning(
                f"Error during unfiltered event validation step: {e}"
            )
            # valid_events remains False

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
                    traces, filtered_offsets, filtered_indices
                ):
                    logger.info("Filtered event validation successful.")
                    valid_events = True
                    final_offsets = filtered_offsets
                    final_offset_indices = filtered_indices
                else:
                    logger.warning("Filtered event validation failed.")
            except Exception as e:
                logger.warning(
                    f"Error during filtered event validation step: {e}"
                )
                # valid_events remains False

        # --- Final Check ---
        if not valid_events:
            logger.warning(
                "Traces and events do not match"
                " after attempting both unfiltered and filtered validation."
                " Skipping event analysis."
            )
            return  # Exit event processing

        # Use the successfully validated offsets and indices for further processing
        offsets = final_offsets
        offset_indices = final_offset_indices

        # convert the indices to timeseries
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
        event_timeseries = np.nan_to_num(event_timeseries)

        # Bin the data
        event_timeseries = _bin_data(event_timeseries, bin_size, period)

        # now rescale the event_timeseries if appropriate
        if event_scale_method == "normalize":
            event_timeseries = _norm_2D_array(event_timeseries)
            ylabel = "Normalized Event Rate"
        elif event_scale_method == "standardize":
            event_timeseries = _standardize_2D_array(event_timeseries)
            ylabel = "Standardized Event Rate"
        elif event_scale_method == "fractional_change":
            # Changed: update event_timeseries instead of traces
            event_timeseries = _fractional_change_2D_array(
                event_timeseries, epochs, functional_period
            )
            ylabel = "Fractional Change in Event Rate"
        elif event_scale_method == "standardize_epoch":
            # Changed: update event_timeseries instead of traces
            event_timeseries = _standardize_to_first_epoch(
                event_timeseries, epochs, functional_period
            )
            ylabel = "Standardized Event Rate"
        else:
            ylabel = "Event Rate (Hz)"

        # check event rate for nans and replace with 0
        if np.isnan(event_timeseries).any():
            logger.warning("Event rate contains NaNs. Replacing with 0.")
        event_timeseries = np.nan_to_num(event_timeseries)

        logger.info("Event timeseries created.")
        # plot average event rate between epochs

        _calculate_population_activity(
            data_name="Eventrate",
            data=event_timeseries,
            epochs=epochs,
            x=x,
            y=y,
            epoch_names=epoch_names,
            epoch_colors=epoch_colors,
            period=functional_period,
            ylabel=ylabel,
            scale_method=event_scale_method,
            cell_names=cell_names,
        )
        logger.info("Population analysis complete.")
        # plot time course of activity
        _plot_timecourse(
            data_name="Eventrate",
            data=event_timeseries,
            epochs=epochs,
            boundaries=boundaries,
            epoch_names=epoch_names,
            epoch_colors=epoch_colors,
            raw_data=offsets,
            period=functional_period,
            scale_method=event_scale_method,
        )
        logger.info("Event analysis complete.")

        logger.info("Creating metadata")
        events_timecourse_key = Path("Eventrate_timecourse_data.npy").stem
        events_activity_key = Path("Eventrate_activity_data.csv").stem

        events_timecourse_values = {
            "epoch_lengths": [epoch[1] - epoch[0] for epoch in epochs],
            "epoch_names": epoch_names,
            "epoch_colors": epoch_colors,
            "sampling_rate": 1 / functional_period,
            "num_cells": traces.shape[1],
        }
        events_activity_values = {
            "epoch_lengths": [epoch[1] - epoch[0] for epoch in epochs],
            "epoch_names": epoch_names,
            "epoch_colors": epoch_colors,
            "num_cells": traces.shape[1],
        }
        metadata[events_timecourse_key] = events_timecourse_values
        metadata[events_activity_key] = events_activity_values

    # Save metadata
    with open("output_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)


def epoch_activity_ideas_wrapper(
    *,
    cell_set_files: List[IdeasFile],
    event_set_files: Optional[List[IdeasFile]] = None,
    define_epochs_by: str,
    epoch_names: str,
    epochs: Optional[str] = None,
    epoch_colors: str,
    bin_size: Optional[float] = None,
    trace_scale_method: Optional[str] = "none",
    event_scale_method: Optional[str] = "none",
    sort_by_time: Optional[bool] = True,
    tolerance: Optional[float] = 1e-4,
    heatmap: Optional[tuple] = "auto",
):
    run(
        cell_set_files=cell_set_files,
        event_set_files=event_set_files,
        define_epochs_by=define_epochs_by,
        epoch_names=epoch_names,
        epochs=epochs,
        epoch_colors=epoch_colors,
        bin_size=bin_size,
        trace_scale_method=trace_scale_method,
        event_scale_method=event_scale_method,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
        heatmap=heatmap,
    )
    
    try:
        logger.info("Registering output data")
        output_prefix = outputs.input_paths_to_output_prefix(cell_set_files, event_set_files)
        metadata = outputs._load_and_remove_output_metadata()
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "Traces_timecourse_data.npy",
                prefix=output_prefix,
                subdir="Traces_timecourse_data"
            ).register_preview(
                "Trace_Preview.svg",
                caption="Preview of calcium traces for up to the first 20 cells with epoch periods overlaid"
            ).register_preview(
                "Trace_Preview.svg",
                caption="Preview of calcium traces for up to the first 20 cells with epoch periods overlaid"
            ).register_preview(
                "Traces_Single_Cell_Timecourse.svg",
                caption="Top panel is the trace activity averaged across the entire population. Bottom panel is a heat map showing the activity of individual cell trace activities across the entire recording"
            ).register_preview(
                "Traces_Population_Timecourse.svg",
                caption="Left panel is trace activity averaged across the entire population. Right panel is the average trace activity averaged across the entire population in each epoch. Error bars represent the standard deviation"
            ).register_metadata_dict(
                **metadata["Traces_timecourse_data"]
            )

            output_data.register_file(
                "Traces_activity_data.csv",
                prefix=output_prefix,
                subdir="Traces_activity_data"
            ).register_preview(
                "Population_Traces_differences.svg",
                caption="Each row is a histogram of the pairwise difference of trace activity levels between different epochs. On The right of each row is the magnitude of change at each individual cell"
            ).register_preview(
                "Population_Traces.svg",
                caption="Average trace activity across epochs displayed using a box plot overlaid over a strip plot with lines connecting cells across epochs"
            ).register_metadata_dict(
                **metadata["Traces_activity_data"]
            )

            output_data.register_file(
                "Population_Traces_Correlation.npy",
                prefix=output_prefix,
                subdir="Population_Traces_Correlation"
            ).register_preview(
                "Correlations.svg",
                caption="Correlation heatmaps for each epoch with similar neurons clustered together"
            ).register_preview(
                "Average_Correlations.svg",
                caption="Summary plot showing the amplitude of average positive and negative correlations"
            ).register_metadata_dict(
                **metadata["Traces_correlation_data"]
            )
            
            output_data.register_file(
                "Eventrate_timecourse_data.npy",
                prefix=output_prefix,
                subdir="Eventrate_timecourse_data"
            ).register_preview(
                "Eventrate_Population_Timecourse.svg",
                caption="Left panel is the event rate averaged across the entire population. Right panel is the average event rate averaged across the entire population in each epoch. Error bars represent the standard deviation"
            ).register_preview(
                "Eventrate_Single_Cell_Timecourse.svg",
                caption="Top panel is the event rate averaged across the entire population. Bottom panel is a raster plot showing the event timestamps  of individual cells across the entire recording"
            ).register_metadata_dict(
                **metadata["Eventrate_timecourse_data"]
            )

            output_data.register_file(
                "Eventrate_activity_data.csv",
                prefix=output_prefix,
                subdir="Eventrate_activity_data"
            ).register_preview(
                "Population_Eventrate.svg",
                caption="Average event rate across epochs displayed using a box plot overlaid over a strip plot with lines connecting cells across epochs"
            ).register_preview(
                "Population_Eventrate_differences.svg",
                caption="Each row is a histogram of the pairwise difference of event rate between different epochs. On The right of each row is the magnitude of change at each individual cell"
            ).register_metadata_dict(
                **metadata["Eventrate_activity_data"]
            )

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")
