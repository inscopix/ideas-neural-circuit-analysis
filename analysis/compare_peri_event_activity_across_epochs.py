import json
import copy
import logging
import os
import math
from collections import OrderedDict
from typing import List
import h5py
import isx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.config as config
# from ideas_commons.constants import (
#     FileCategory,
#     FileFormat,
#     FileStructure,
#     FileType,
#     GroupType,
# )
from scipy import stats
# from toolbox.utils.data_model import IdeasFile, IdeasGroup, IdeasPreviewFile
from ideas.exceptions import IdeasError
from ideas.tools.log import get_logger
from utils.footprint_utils import compute_cell_centroid
from utils.stats_utils import perform_paired_pairwise_comparisons
from ideas.analysis.utils import (
    get_file_size,
)
from utils.utils import (
    validate_cellset_series_compatibility,
    get_num_cells_by_status
)
from utils.utils import (
    _parse_string_to_tuples,
    _redefine_epochs,
    _get_cellset_boundaries,
)
from utils.metadata import read_isxd_metadata
from analysis.peri_event_workflow import (
    nanmean_iterative,
    extract_event_windows,
    extract_mean_event_windows_per_event_shuffle,
    compute_post_minus_pre,
    generate_post_minus_pre_shuffled_dist,
    compute_statistical_metrics,
    plot_population_mean_event_window,
    plot_single_neurons_heatmap,
    plot_single_cell_modulation,
    plot_cell_map,
    validate_modulation_colors,
    bin_traces_update_period,
    validate_peri_event_workflow_parameters,
)
from utils.plots import (
    plot_population_mean_event_window_across_epochs,
    plot_event_aligned_activity_by_modulation_group_comparing_epochs,
    plot_population_activity,
    plot_number_of_events_per_epoch,
    plot_post_minus_pre_per_epoch_bar_chart,
    plot_number_of_modulated_cells_per_epoch,
    plot_post_minus_pre_activity_differences_with_cell_map,
)


logger = get_logger()


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


def _check_num_epochs(
    *,
    epochs,
    epoch_names,
    epoch_colors,
):
    """Check if the number of epochs, epoch names, and epoch colors are the same.

    :Parameters
        epochs: List of epoch time ranges
        epoch_names: List of epoch names
        epoch_colors: List of epoch colors
    """
    if len(epochs) != len(epoch_names) or len(epochs) != len(epoch_colors):
        raise IdeasError(
            "The number of epochs, epoch names, and epoch colors must be the same.",
        )


def _check_epochs_valid(
    *,
    num_samples,
    epochs,
    period,
):
    """Check if the epochs are valid.

    :Parameters
        num_samples: Number of samples in the input traces
        epochs: List of epoch time ranges
        period: Sampling period
    """
    epoch_idx = _epoch_time_to_index(epochs, period)

    for epoch in epoch_idx:
        # Make sure start is less than end
        if epoch[0] >= epoch[1]:
            raise IdeasError(
                "The start time of the epoch must be less than the end time.",
            )

        # Make sure the epoch positive
        if epoch[0] < 0 or epoch[1] < 0:
            raise IdeasError(
                "The epoch must be positive.",
            )
        # Make sure the epoch is within the range of the traces
        if epoch[0] > num_samples or epoch[1] > num_samples:
            raise IdeasError(
                f"The epoch is not within the range of the traces. ({epoch[0]}, {epoch[1]}) vs {num_samples}",
            )


def compare_peri_event_activity_across_epochs(
    input_cellset_files: List[str],
    input_events_h5_file: str,
    event_type: str,
    define_epochs_by: str,
    epoch_names: str,
    epoch_periods: str,
    epoch_colors: str,
    epoch_min_events: int,
    visual_window_pre: float = -2.0,
    visual_window_post: float = 2.0,
    statistical_window_pre_start: float = -1.0,
    statistical_window_pre_end: float = 1.0,
    statistical_window_post_start: float = 0.0,
    statistical_window_post_end: float = 1.0,
    num_shuffles: int = 1000,
    significance_threshold: float = 0.05,
    seed: int = 0,
    comparison_type: str = "two_tailed",
    modulation_colors: str = "green, blue, black",
    cmap: str = "coolwarm",
    temporal_downsampling_factor: int = 1,
    population_activity_plot_limits: str = "auto",
    activity_heatmap_color_limits: str = "auto",
    activity_by_modulation_plot_limits: str = "auto",
    output_dir=None,
):
    """Peri-event analysis workflow.

    :param input_cellset_files: list of paths to the cell set files
    :param input_events_h5_file: path to the events file
    :param event_type: string representing an event type (currently only supports 1 event type)
    :param epoch_names: list of epoch names
    :param epoch_periods: list of epoch periods specified as list of tuples encoded as a string (e.g. "(1,10), (11,20)")
    :param epoch_colors: list of epoch colors
    :param epoch_min_events: minimum number of events per epoch
    :param visual_window_pre: time in seconds before each event to use for visualization
    :param visual_window_post: time in seconds after each event to use for visualization
    :param statistical_window_pre_start: start of time range in seconds before each event
     to use for statistical tests
    :param statistical_window_pre_end: end of time range in seconds before each event
     to use for statistical tests
    :param statistical_window_post_start: start of time range in seconds after each event
     to use for statistical tests
    :param statistical_window_post_end: end of time range in seconds after each event
     to use for statistical tests
    :param num_shuffles: number of random shuffles of the event times to perform
    :param significance_threshold: threshold to use for statistical significance tests
    :param seed: seed for the random generator used to shuffle event indices
    :param modulation_colors: comma separated strings with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param cmap: colormap applied to the activity heatmap
    :param temporal_downsampling_factor: downsampling factor to apply to the traces during the analysis
    :param population_activity_plot_limits: y-axis range (z-score) applied to the event-aligned
     population activity plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_heatmap_color_limits: colormap range (z-score) applied to the activity heatmap
     specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_by_modulation_plot_limits: y-axis range (z-score) applied to the event-aligned
     activity by modulation plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param output_dir: path to the output directory
    """
    logger.info("Starting peri-event analysis across epochs")

    # automatically create or modify epochs
    cellset = isx.CellSet.read(input_cellset_files[0])
    period = cellset.timing.period.secs_float

    boundaries = _get_cellset_boundaries(
        cell_set_files=input_cellset_files,
        period=period,
    )

    epoch_periods = _redefine_epochs(
        epochs=epoch_periods,
        define_epochs_by=define_epochs_by,
        boundaries=boundaries,
    )
    del cellset

    # parse epoch data
    epochs = _parse_string_to_tuples(epoch_periods)
    epoch_colors = [c.strip() for c in epoch_colors.split(",")]
    epoch_names = [e.strip() for e in epoch_names.split(",")]

    # retrieve timing metadata
    num_samples = sum(
        [isx.CellSet.read(f).timing.num_samples for f in input_cellset_files]
    )
    period = isx.CellSet.read(input_cellset_files[0]).timing.period.secs_float

    # validate epoch data
    _check_num_epochs(
        epochs=epochs,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
    )
    _check_epochs_valid(
        num_samples=num_samples,
        epochs=epochs,
        period=period,
    )

    # construct epoch data structure
    epoch_data = {}
    for epoch, epoch_name, epoch_color in zip(
        epochs, epoch_names, epoch_colors
    ):
        epoch_indices = _epoch_time_to_index([epoch], period)[0]
        logger.info(
            f"Converted epoch period for epoch '{epoch_name}' to corresponding indices: {epoch} -> {epoch_indices}"
        )

        epoch_data[epoch_name] = {
            "period": epoch,
            "indices": epoch_indices,
            "color": epoch_color,
            "recording_period": period,
        }

    # run peri-event workflow
    run_peri_event_workflow(
        input_cellset_files=input_cellset_files,
        input_events_h5_file=input_events_h5_file,
        event_type=event_type,
        visual_window_pre=visual_window_pre,
        visual_window_post=visual_window_post,
        statistical_window_pre_start=statistical_window_pre_start,
        statistical_window_pre_end=statistical_window_pre_end,
        statistical_window_post_start=statistical_window_post_start,
        statistical_window_post_end=statistical_window_post_end,
        num_shuffles=num_shuffles,
        significance_threshold=significance_threshold,
        seed=seed,
        comparison_type=comparison_type,
        modulation_colors=modulation_colors,
        cmap=cmap,
        temporal_downsampling_factor=temporal_downsampling_factor,
        population_activity_plot_limits=population_activity_plot_limits,
        activity_heatmap_color_limits=activity_heatmap_color_limits,
        activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
        epoch_data=epoch_data,
        epoch_min_events=epoch_min_events,
        output_dir=output_dir,
    )

    logger.info("Peri-event analysis across epochs completed")


def run_peri_event_workflow(
    input_cellset_files: List[str],
    input_events_h5_file: str,
    event_type: str,
    visual_window_pre: float = -2.0,
    visual_window_post: float = 2.0,
    statistical_window_pre_start: float = -1.0,
    statistical_window_pre_end: float = 1.0,
    statistical_window_post_start: float = 0.0,
    statistical_window_post_end: float = 1.0,
    num_shuffles: int = 1000,
    significance_threshold: float = 0.05,
    seed: int = 0,
    comparison_type: str = "two_tailed",
    modulation_colors: str = "green, blue, black",
    cmap: str = "coolwarm",
    temporal_downsampling_factor: int = 1,
    population_activity_plot_limits: str = "auto",
    activity_heatmap_color_limits: str = "auto",
    activity_by_modulation_plot_limits: str = "auto",
    epoch_data=None,
    epoch_min_events=1,
    output_dir=None,
):
    """Peri-event analysis workflow across epochs.

    :param input_cellset_files: list of paths to the cell set files
    :param input_events_h5_file: path to the events file
    :param event_type: string representing an event type (currently only supports 1 event type)
    :param visual_window_pre: time in seconds before each event to use for visualization
    :param visual_window_post: time in seconds after each event to use for visualization
    :param statistical_window_pre_start: start of time range in seconds before each event
     to use for statistical tests
    :param statistical_window_pre_end: end of time range in seconds before each event
     to use for statistical tests
    :param statistical_window_post_start: start of time range in seconds after each event
     to use for statistical tests
    :param statistical_window_post_end: end of time range in seconds after each event
     to use for statistical tests
    :param num_shuffles: number of random shuffles of the event times to perform
    :param significance_threshold: threshold to use for statistical significance tests
    :param seed: seed for the random generator used to shuffle event indices
    :param modulation_colors: comma separated strings with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param cmap: colormap applied to the activity heatmap
    :param temporal_downsampling_factor: downsampling factor to apply to the traces during the analysis
    :param population_activity_plot_limits: y-axis range (z-score) applied to the event-aligned
     population activity plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_heatmap_color_limits: colormap range (z-score) applied to the activity heatmap
     specified as 'min,max' (e.g. -1,1) or 'auto'
    :param activity_by_modulation_plot_limits: y-axis range (z-score) applied to the event-aligned
     activity by modulation plot specified as 'min,max' (e.g. -1,1) or 'auto'
    :param epoch_data: dictionary containing epoch-specific information (keys are the epoch names)
    :param epoch_min_events: minimum number of events per epoch
    :param output_dir: path to the output directory
    """
    logger.info("Starting the peri-event analysis workflow")

    modulation_colors = validate_modulation_colors(modulation_colors)
    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )

    # set output directory to the current working directory if unspecified
    if output_dir is None:
        output_dir = os.getcwd()

    # validate input files exist
    for f in [
        *input_cellset_files,
        input_events_h5_file,
    ]:
        if not os.path.exists(f):
            raise IdeasError(
                "Input file '{0}' does not exist.".format(os.path.basename(f)),
            )

    # construct parameter dictionary
    parameters = {
        "event_types": [event_type],
        "visual_window": {
            "pre": visual_window_pre,
            "post": visual_window_post,
        },
        "statistical_window": {
            "pre": [statistical_window_pre_start, statistical_window_pre_end],
            "post": [
                statistical_window_post_start,
                statistical_window_post_end,
            ],
        },
        "num_shuffles": num_shuffles,
        "significance_threshold": significance_threshold,
        "seed": seed,
        "population_activity_plot_limits": population_activity_plot_limits,
        "activity_heatmap_color_limits": activity_heatmap_color_limits,
        "activity_by_modulation_plot_limits": activity_by_modulation_plot_limits,
    }

    # validate processing parameters
    validate_peri_event_workflow_parameters(parameters)

    # validate series compatibility of input cell sets
    validate_cellset_series_compatibility(input_cellset_files)

    # load cell sets
    cellsets = [isx.CellSet.read(f) for f in input_cellset_files]

    # extract cell set boundaries
    boundaries = _get_cellset_boundaries(
        cell_set_files=input_cellset_files,
        period=cellsets[0].timing.period.secs_float,
    )

    # retrieve cell statuses
    (
        num_accepted_cells,
        num_undecided_cells,
        num_rejected_cells,
    ) = get_num_cells_by_status(input_cellset_files[0])

    if num_accepted_cells > 0:
        logger.info(f"Detected {num_accepted_cells} accepted cells to analyze")
        cell_indices_to_process = [
            i
            for i in range(cellsets[0].num_cells)
            if cellsets[0].get_cell_status(i) == "accepted"
        ]
    elif num_undecided_cells > 0:
        logger.info(
            f"No accepted cells were detected. "
            f"Detected {num_undecided_cells} undecided cells to analyze"
        )
        cell_indices_to_process = [
            i
            for i in range(cellsets[0].num_cells)
            if cellsets[0].get_cell_status(i) == "undecided"
        ]
    else:
        raise IdeasError(
            "There are no accepted or undecided cells to process",
        )

    cell_names_to_process = [
        cellsets[0].get_cell_name(i) for i in cell_indices_to_process
    ]

    # ensure there is at least one non-rejected cell
    if len(cell_names_to_process) < 1:
        raise IdeasError(
            "There are no non-rejected cells in the input cell sets.",
        )

    # retrieve relevant processing parameters from metadata
    first_cell_set_metadata = read_isxd_metadata(input_cellset_files[0])
    period = (
        first_cell_set_metadata["timingInfo"]["period"]["num"]
        / first_cell_set_metadata["timingInfo"]["period"]["den"]
    )

    # load traces and footprints for non-rejected cells
    traces_dict = {}
    footprints = np.zeros(
        (
            len(cell_names_to_process),
            cellsets[0].spacing.num_pixels[1],
            cellsets[0].spacing.num_pixels[0],
        ),
        dtype="float32",
    )

    # collect footprints and traces of non-rejected cells
    for i, cell_index in enumerate(cell_indices_to_process):
        footprints[i] = cellsets[0].get_cell_image_data(cell_index).T
        cell_name = cellsets[0].get_cell_name(cell_index)
        traces_dict[cell_name] = np.concatenate(
            [cs.get_cell_trace_data(cell_index) for cs in cellsets]
        )
    traces_df = pd.DataFrame.from_dict(traces_dict)

    # standardize traces (z-scores)
    standardized_traces_df = (traces_df - traces_df.mean()) / traces_df.std(
        ddof=0
    )

    # construct traces timepoints
    traces_timepoints = []
    for i, cs in enumerate(cellsets):
        item_timepoints = np.array(
            [
                n * cs.timing.period.secs_float
                for n in range(cs.timing.num_samples)
            ]
        )

        if i > 0:
            item_timepoints += (
                traces_timepoints[-1]
                + cellsets[i - 1].timing.period.secs_float
            )

        traces_timepoints.extend(item_timepoints)

    # remove cell set objects
    del cellsets[:]

    # bin trace, timestamps and update period to the new bin_size
    logger.info(
        f"Downsampling activity traces with a factor of {temporal_downsampling_factor}"
    )
    (
        standardized_traces_df,
        traces_timepoints,
        period,
    ) = bin_traces_update_period(
        standardized_traces_df=standardized_traces_df,
        traces_timepoints=traces_timepoints,
        period=period,
        temporal_downsampling_factor=temporal_downsampling_factor,
    )

    # visual window as (pre, post)
    visual_window = (
        parameters["visual_window"]["pre"],
        parameters["visual_window"]["post"],
    )
    # statistical window as ([pre_start, pre_end], [post_start, post_end])
    statistical_window = (
        parameters["statistical_window"]["pre"],
        parameters["statistical_window"]["post"],
    )
    num_shuffles = parameters["num_shuffles"]
    significance_threshold = parameters["significance_threshold"]

    # convert visual and statistical windows to the equivalent number of frames
    visual_window_frames = (
        int(np.ceil(abs(visual_window[0]) / period)),
        int(np.ceil(abs(visual_window[1]) / period)),
    )
    statistical_window_frames = (
        [
            int(round(abs(statistical_window[0][0]) / period)),
            int(round(abs(statistical_window[0][1]) / period)),
        ],
        [
            int(round(abs(statistical_window[1][0]) / period)),
            int(round(abs(statistical_window[1][1]) / period)),
        ],
    )

    # define values to use along the x-axis, i.e. time in seconds from event
    num_x_values = visual_window_frames[0] + 1 + visual_window_frames[1]
    x_values = np.linspace(
        -visual_window_frames[0] * period,
        visual_window_frames[1] * period,
        num_x_values,
    )
    x_limits = (visual_window[0], visual_window[1])

    # ensure time window does not exceed length of traces
    num_timepoints = first_cell_set_metadata["timingInfo"]["numTimes"]
    visual_window_length = (
        visual_window_frames[0] + 1 + visual_window_frames[1]
    )
    if visual_window_length > num_timepoints:
        raise IdeasError(
            "The visual window must be temporally shorter than the input traces.",
        )

    # ensure that the statistical window is contained within the visual window
    if (
        statistical_window_frames[0][0] > visual_window_frames[0]
        or statistical_window_frames[1][1] > visual_window_frames[1]
    ):
        raise IdeasError(
            "The statistical window must be contained within the visual window.",
        )
    # retrieve event types to consider from parameters
    events_data = h5py.File(input_events_h5_file, "r")
    try:
        event_types = parameters["event_types"]
    except KeyError:
        # if event types is not specified, process all event types in the h5 file
        event_types = list(events_data["events"].keys())
        logger.warning(
            f"Event types set to {event_types} based on the event types contained in the input events file"
        )

    # if empty list provided, retrieve all event types from the h5 file
    if len(event_types) == 0:
        event_types = list(events_data["events"].keys())

    # retain unique event types to avoid processing same event type more than once
    event_types = np.unique(event_types).tolist()
    event_type_name = event_types[0]
    logger.info(f"Processing data for event type '{event_type_name}'")

    # perform peri-event analysis for one event type
    peri_event_analysis_for_single_event_type(
        traces_df=standardized_traces_df,
        traces_timepoints=traces_timepoints,
        footprints=footprints,
        events_data=events_data,
        event_type=event_type_name,
        valid_cells=cell_names_to_process,
        visual_window=visual_window_frames,
        statistical_window=statistical_window_frames,
        significance_threshold=significance_threshold,
        x_values=x_values,
        x_limits=x_limits,
        period=period,
        num_shuffles=num_shuffles,
        seed=parameters["seed"],
        cmap=cmap,
        output_dir=output_dir,
        input_cellset_files=input_cellset_files,
        input_events_h5_file=input_events_h5_file,
        epoch_data=epoch_data,
        epoch_min_events=epoch_min_events,
        comparison_type=comparison_type,
        # visualize parameters
        modulation_colors=modulation_colors,
        population_activity_plot_limits=population_activity_plot_limits,
        activity_heatmap_color_limits=activity_heatmap_color_limits,
        activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
        boundaries=boundaries,
    )

    # save output manifest to disk
    # save_output_manifest(peri_event_analysis_group, output_dir)

    # save output metadata manifest to disk
    # peri_event_analysis_group.add_group_metadata(
    #     key="event_type", value=event_type_name
    # )
    # save_metadata_manifest(peri_event_analysis_group, output_dir)

    logger.info("Peri-event analysis workflow completed")


def compare_post_minus_pre_across_epochs(
    single_cell_data, epoch_data, comparison_type, output_filename
):
    """Compare the post-pre values across epochs using pairwise t-tests.
    :param single_cell_data: dictionary containing single-cell data
    :param epoch_data: dictionary containing epoch-specific data
    :output_filename: path to the output csv file
    """
    dfs = []
    for epoch_name in epoch_data.keys():
        # population
        epoch_population_true_post_minus_pre_dist = single_cell_data[
            epoch_name
        ]["cell"]["true_mean_post-pre"]
        num_cells = len(epoch_population_true_post_minus_pre_dist)

        # construct epoch dataframe
        df_epoch = pd.DataFrame(
            {
                "cell": np.arange(num_cells),
                "epoch": [epoch_name] * num_cells,
                "population_post_minus_pre": epoch_population_true_post_minus_pre_dist,
            }
        )
        dfs.append(df_epoch)

    # combine dataframes from all epochs together
    df = pd.concat(dfs)

    pairwise = perform_paired_pairwise_comparisons(
        df=df,
        comparison_type=comparison_type,
        dependent_var_col="population_post_minus_pre",
        comparison_name="population post-pre",
        within_col="epoch",
        subject_col="cell",
    )

    # save comparisons to disk
    pairwise.to_csv(output_filename, index=False)

    logger.info(
        f"Pairwise epoch comparison data saved "
        f"({os.path.basename(output_filename)}, "
        f"size: {get_file_size(output_filename)})"
    )


def peri_event_analysis_for_single_event_type(
    traces_df,
    traces_timepoints,
    footprints,
    events_data,
    event_type,
    valid_cells,
    visual_window,
    statistical_window,
    significance_threshold,
    x_values,
    x_limits,
    period,
    num_shuffles,
    seed,
    modulation_colors,
    population_activity_plot_limits,
    activity_heatmap_color_limits,
    activity_by_modulation_plot_limits,
    cmap,
    output_dir,
    input_cellset_files,
    input_events_h5_file,
    comparison_type,
    epoch_data=None,
    epoch_min_events=1,
    boundaries=None,
):
    """Perform peri-event analysis for a single event type.

    :param traces_df: dataframe containing individual cell traces (rows=timepoints, columns=cells)
    :param traces_timepoints: timestamps for the cellular activity traces
    :param footprints: footprints array (num_cells, width, height)
    :param events_data: h5 file object containing the events data
    :param event_type: string representing the event type to analyze
    :param valid_cells: list of cell names to process
    :param visual_window: window of time to use for visualization purposes (pre, post)
    :param statistical_window: window of time to use for statistical tests
                               ([pre_start, pre_end], [post_start, post_end])
    :param significance_threshold: significance threshold to use for statistical tests
    :param x_values: values of the x-axis to use for plots
    :param x_limits: limits of the x-axis to use for plots
    :param period: duration of a sample in the input traces
    :param num_shuffles: number of random shuffles of the event times to perform to
                         construct the null distribution
    :param seed: seed for the random generator used to shuffle event indices
    :param modulation_colors: string list with color inputs
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param output_dir: path to the output directory for the event type being processed
    :param input_cellset_files: list of paths to the cell set files
    :param input_events_h5_file: path to the events file
    :param cmap: colormap applied to the activity heatmap
    :param output_dir: path to the output directory
    :param input_cellset_files: list of paths to the input isxd cell set files
    :param input_events_h5_file: path to the input h5 file containing timestamp events
    :param epoch_data: dictionary containing epoch-specific information (keys are the epoch names)
    :param epoch_min_events: minimum number of events per epoch
    """
    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )
    # read event times
    try:
        event_times_by_type = events_data["events"][event_type]
    except KeyError:
        raise IdeasError(
            "The event type '{0}' does not exist in the events file.".format(
                event_type
            ),
        )

    try:
        event_times = event_times_by_type["Time"][:]
    except KeyError:
        raise IdeasError(
            "The events file does not contain timestamps for event type '{0}'.".format(
                event_type
            ),
        )

    # convert event times to the corresponding trace time indices
    # (trace time indices correspond to frame indices in the movie they were derived from)
    event_indices = ((event_times / period).round()).astype(int)

    # retain only events for which we can extract the full time window around them
    # without exceeding the recording length
    pre_visual, post_visual = visual_window
    max_time_index = len(traces_df) - 1

    # determine event indices
    # (assume input files form a series when the number of cell set files is greater than one)
    is_series = len(input_cellset_files) > 1
    if is_series:
        # input data consists of a series

        # compute start index of each item in the series
        series_start_indices = [0]
        for f in input_cellset_files[:-1]:
            cs = isx.CellSet.read(f)
            series_start_indices.append(
                series_start_indices[-1] + cs.timing.num_samples
            )

        # - loop over items in the series, each time considering whether events exceed the
        #   temporal boundaries of the item
        series_item_indices = series_start_indices + [max_time_index]

        # - Compute the indices of the beginning and end of each item in the series
        #   assuming a continuous timeline across the series.
        series_item_endpoints = []
        for row_start_index, row_end_index in zip(
            series_item_indices, series_item_indices[1:]
        ):
            # retrieve start and end times of this portion of the series
            subset_traces_start_time = traces_timepoints[row_start_index]
            subset_traces_end_time = traces_timepoints[row_end_index - 1]

            # convert start and end times to the corresponding indices across the entire series
            subset_traces_start_index = int(
                round(subset_traces_start_time / period)
            )
            subset_traces_end_index = int(
                round(subset_traces_end_time / period)
            )

            series_item_endpoints.append(
                (subset_traces_start_index, subset_traces_end_index)
            )

        # compare event indices to the indices of each item in the series
        # and retain those for which we have a complete window
        filtered_event_row_indices = []
        for event_index in event_indices:
            for item_start_index, item_end_index in series_item_endpoints:
                if (event_index - pre_visual >= item_start_index) and (
                    event_index + post_visual <= item_end_index - 1
                ):
                    # map event index to the row index in the parquet traces table
                    # with the closest timestamp
                    event_time = event_index * period
                    event_index_in_table = (
                        np.abs(traces_timepoints - event_time)
                    ).argmin()

                    filtered_event_row_indices.append(event_index_in_table)
                    continue

        # update event indices so that they now represent the row
        # indices to use in the parquet table
        event_indices = filtered_event_row_indices
    else:
        # here we know we are NOT dealing with series data
        event_indices = np.array(
            [
                event_index
                for event_index in event_indices
                if event_index - pre_visual >= 0
                and event_index + post_visual <= max_time_index
            ]
        )

    if len(event_indices) < 1:
        raise IdeasError(
            "There are no events for which a complete time window can be constructed.",
        )

    # prepare shuffles of event indices across the entire timeline of the input trace
    event_indices_shuffles = []

    # seed random generator to allow reproducibility
    np.random.seed(seed)

    # create a one hot vector with event times
    one_hot_triggers = np.zeros(len(traces_df), bool)
    one_hot_triggers[event_indices] = 1

    # Generate shuffled event data.
    # - When dealing with a small number of events, some shuffles may not have any valid event so
    #   we must reshuffle the data until at least one valid event is obtained.
    # - To prevent an infinite loop, we set a maximum number of iterations.
    num_iterations = 0
    max_num_iterations = 2 * num_shuffles
    while (
        len(event_indices_shuffles) < num_shuffles
        and num_iterations < max_num_iterations
    ):
        # circulate event times
        # this preserves inter-event intervals
        shuffle_indices = np.roll(
            one_hot_triggers, np.random.randint(0, len(traces_df))
        )

        shuffle_indices = np.where(shuffle_indices)[0]

        # filter out events for which we cannot construct a complete window
        shuffle_indices = np.array(
            [
                event_index
                for event_index in shuffle_indices
                if event_index - pre_visual >= 0
                and event_index + post_visual <= max_time_index
            ]
        )

        if len(shuffle_indices) > 0:
            event_indices_shuffles.append(shuffle_indices)

        num_iterations += 1

    if (
        len(event_indices_shuffles) != num_shuffles
        and num_iterations == max_num_iterations
    ):
        logger.warning(
            f"{len(event_indices_shuffles)} shuffles were successfully generated and "
            f"will be used to construct the null distribution"
        )

    # dict to hold output data
    output_data = {}

    # identify pre- and post- event indices for statistical window
    visual_window_indices = np.arange(-visual_window[0], visual_window[1] + 1)
    stats_pre_event_indices = np.where(
        (visual_window_indices >= -statistical_window[0][0])
        & (visual_window_indices <= -statistical_window[0][1])
    )[0]
    stats_post_event_indices = np.where(
        (visual_window_indices >= statistical_window[1][0])
        & (visual_window_indices <= statistical_window[1][1])
    )[0]
    statistical_window_indices = {
        "pre": stats_pre_event_indices,
        "post": stats_post_event_indices,
    }

    # convert event indices to numpy array for easier manipulation
    event_indices = np.array(event_indices)

    # identify valid events within each epoch
    epochs_to_exclude = []
    for epoch_name, epoch_data_item in epoch_data.items():
        # convert epoch period to corresponding frame indices
        epoch_indices = _epoch_time_to_index(
            [epoch_data_item["period"]], period
        )[0]

        # retain only event indices within the epoch
        event_indices_within_epoch = event_indices[
            (event_indices - pre_visual > epoch_indices[0])
            & (event_indices + post_visual < epoch_indices[1])
        ]
        event_indices_shuffles_within_epoch = [
            event_indices_shuffle[
                (event_indices_shuffle - pre_visual > epoch_indices[0])
                & (event_indices_shuffle + post_visual < epoch_indices[1])
            ]
            for event_indices_shuffle in event_indices_shuffles
        ]

        # - Exclude epochs for which we do not have a sufficient number of valid events.
        # - An event is considered valid if a full visual window can be constructed around it.
        num_events_in_epoch = len(event_indices_within_epoch)
        logger.info(
            f"Found {num_events_in_epoch} valid events in epoch '{epoch_name}'"
        )
        if num_events_in_epoch == 0:
            logger.warning(
                f"No valid events found in epoch '{epoch_name}'. The epoch will be excluded from the analysis."
            )
            epochs_to_exclude.append(epoch_name)
            continue
        elif num_events_in_epoch < epoch_min_events:
            logger.warning(
                f"{num_events_in_epoch} events found in epoch '{epoch_name}', but the minimum number of events per epoch is {epoch_min_events}. "
                f"The epoch will be excluded from the analysis."
            )
            epochs_to_exclude.append(epoch_name)
            continue

        epoch_data[epoch_name]["indices"] = epoch_indices
        epoch_data[epoch_name]["event_indices"] = event_indices_within_epoch
        epoch_data[epoch_name][
            "event_indices_shuffles"
        ] = event_indices_shuffles_within_epoch

    # remove epochs without events
    for epoch_to_exclude in epochs_to_exclude:
        del epoch_data[epoch_to_exclude]

    # ensure we have at least one epoch to process
    num_epochs = len(epoch_data.keys())
    if num_epochs < 1:
        raise IdeasError(
            "There are no epochs to analyze. "
            "This may occur if all epochs were excluded from the analysis "
            "due to an insufficient number of events.",
        )

    # log the epochs that will be processed
    logger.info(
        f"The following epochs will be processed: {list(epoch_data.keys())}"
    )

    # run the peri-event analysis at the POPULATION level
    logger.info("Population activity analysis started")
    (
        output_data["population"],
        # population_activity_preview_file,
        # event_aligned_population_activity_preview_files,
        # num_events_per_epoch_preview_file,
    ) = peri_event_population_analysis(
        traces_df=traces_df,
        visual_window=visual_window,
        statistical_window_indices=statistical_window_indices,
        significance_threshold=significance_threshold,
        x_values=x_values,
        x_limits=x_limits,
        event_type=event_type,
        event_indices=event_indices,
        output_dir=output_dir,
        epoch_data=epoch_data,
        population_activity_plot_limits=population_activity_plot_limits,
        boundaries=boundaries,
    )
    logger.info("Population activity analysis completed")

    # run the peri-event analysis at the SINGLE-CELL level
    logger.info("Single-cell activity analysis started")
    (
        output_data["single_cell"],
        # single_cell_activity_heatmap_preview_files,
        # activity_by_modulation_preview_files,
        # cell_map_preview_files,
        # num_modulated_cells_per_epoch_preview_file,
        # event_aligned_by_modulation_across_epochs_preview_files,
        # post_minus_pre_per_epoch_preview_file,
    ) = peri_event_single_cell_analysis(
        traces_df=traces_df,
        footprints=footprints,
        visual_window=visual_window,
        statistical_window_indices=statistical_window_indices,
        significance_threshold=significance_threshold,
        x_values=x_values,
        x_limits=x_limits,
        event_type=event_type,
        output_dir=output_dir,
        modulation_colors=modulation_colors,
        activity_heatmap_color_limits=activity_heatmap_color_limits,
        activity_by_modulation_plot_limits=activity_by_modulation_plot_limits,
        cmap=cmap,
        epoch_data=epoch_data,
    )
    logger.info("Single-cell activity analysis completed")

    # compare peri-event data (post-pre) across epochs
    if num_epochs <= 1:
        logger.info(
            "Only 1 epoch was analyzed. No comparison data across epochs will be generated."
        )
    else:
        output_epoch_comparison_csv_filename = os.path.join(
            output_dir, "pairwise_epoch_comparisons.csv"
        )
        compare_post_minus_pre_across_epochs(
            single_cell_data=output_data["single_cell"],
            epoch_data=epoch_data,
            comparison_type=comparison_type,
            output_filename=output_epoch_comparison_csv_filename,
        )
        logger.info("Comparison of post-pre data across epochs completed")

        # plot post-pre activity differences between a pair of epochs along with color-coded cell map
        # (
        #     post_minus_pre_activity_differences_preview_files,
        #     post_minus_pre_boxplot_preview_file,
        # ) = 
        plot_post_minus_pre_activity_differences_with_cell_map(
            cell_set_files=input_cellset_files,
            data=output_data,
            epoch_data=epoch_data,
            output_dir=output_dir,
        )

    # save event-aligned neural activity TRACES
    output_traces_csv_filename = os.path.join(
        output_dir, "event_aligned_activity.TRACES.csv"
    )
    save_event_aligned_traces_to_csv(
        event_aligned_data=output_data,
        x_values=x_values,
        valid_cells=valid_cells,
        output_filename=output_traces_csv_filename,
    )

    logger.info(
        f"Event-aligned traces saved "
        f"({os.path.basename(output_traces_csv_filename)}, "
        f"size: {get_file_size(output_traces_csv_filename)})"
    )

    # save event-aligned neural activity STATISTICS
    output_stats_csv_filename = os.path.join(
        output_dir, "event_aligned_activity.STATISTICS.csv"
    )
    save_event_aligned_statistics_to_csv(
        output_data, valid_cells, output_stats_csv_filename
    )

    logger.info(
        f"Event-aligned statistics saved "
        f"({os.path.basename(output_stats_csv_filename)}, "
        f"size: {get_file_size(output_stats_csv_filename)})"
    )

    # create output files/objects/groups and specify relationships
    # source cell set FILES
    # source_cellset_files = [
    #     IdeasFile(
    #         file_key="input_cellset_files",
    #         file_path=os.path.abspath(f),
    #         file_type=FileType.CELL_SET.value[1],
    #         file_format=FileFormat.ISXD_FILE.value[1],
    #         file_structure=FileStructure.BINARY.value[1],
    #         file_category=FileCategory.SOURCE.value[1],
    #     )
    #     for f in input_cellset_files
    # ]

    # # source events FILE
    # source_events_file = IdeasFile(
    #     file_key="input_events_h5_file",
    #     file_path=os.path.abspath(input_events_h5_file),
    #     file_type=FileType.TIMESTAMP_EVENTS.value[1],
    #     file_format=FileFormat.H5_FILE.value[1],
    #     file_structure=FileStructure.SPARSE_TIME_SERIES.value[1],
    #     file_category=FileCategory.SOURCE.value[1],
    # )

    # # construct events metadata (number of valid events per epoch)
    # num_events_str = ""
    # for i, (epoch_name, epoch_data_item) in enumerate(epoch_data.items()):
    #     num_events_str += f"{epoch_name}: {len(epoch_data_item['event_indices'])}{', ' if i < num_epochs - 1 else ''}"

    # # construct modulated cells metadata (number of up/down/non modulated cells per epoch)
    # num_up_modulated_cells_str = ""
    # num_down_modulated_cells_str = ""
    # num_non_modulated_cells_str = ""
    # for i, epoch_name in enumerate(epoch_data.keys()):
    #     num_up_modulated_cells = output_data["single_cell"][epoch_name][
    #         "up_modulated"
    #     ]["num_cells"]
    #     num_down_modulated_cells = output_data["single_cell"][epoch_name][
    #         "down_modulated"
    #     ]["num_cells"]
    #     num_non_modulated_cells = output_data["single_cell"][epoch_name][
    #         "non_modulated"
    #     ]["num_cells"]

    #     num_up_modulated_cells_str += f"{epoch_name}: {num_up_modulated_cells}{', ' if i < num_epochs - 1 else ''}"
    #     num_down_modulated_cells_str += f"{epoch_name}: {num_down_modulated_cells}{', ' if i < num_epochs - 1 else ''}"
    #     num_non_modulated_cells_str += f"{epoch_name}: {num_non_modulated_cells}{', ' if i < num_epochs - 1 else ''}"

    # # event-aligned METADATA
    # first_cs_metadata = read_isxd_metadata(input_cellset_files[0])
    # aligned_traces_metadata = {
    #     config.IDEAS_METADATA_KEY: {
    #         "dataset": {"signal": [{"name": "calcium", "units": "z-score"}]},
    #         "metrics": {
    #             "num_valid_events": num_events_str,
    #             "num_up_modulated_cells": num_up_modulated_cells_str,
    #             "num_down_modulated_cells": num_down_modulated_cells_str,
    #             "num_non_modulated_cells": num_non_modulated_cells_str,
    #         },
    #         "timingInfo": {
    #             "blank": [],
    #             "cropped": [],
    #             "dropped": [],
    #             "numTimes": len(x_values),
    #             "period": {"num": int(period * 1e6), "den": 1000000},
    #             "sampling_rate": compute_sampling_rate(
    #                 period_num=int(period * 1e6), period_den=1000000
    #             ),
    #             "start": {
    #                 "secsSinceEpoch": {
    #                     "num": int(-visual_window[0] * period * 1e6),
    #                     "den": 1000000,
    #                 },
    #                 "utcOffset": 0,
    #             },
    #             "end": {
    #                 "secsSinceEpoch": {
    #                     "num": int(visual_window[1] * period * 1e6),
    #                     "den": 1000000,
    #                 },
    #                 "utcOffset": 0,
    #             },
    #         },
    #         "spacingInfo": first_cs_metadata["spacingInfo"],
    #     }
    # }

    # # event-aligned traces FILE
    # event_aligned_traces_file = IdeasFile(
    #     file_key="event_aligned_traces",
    #     file_path=os.path.abspath(output_traces_csv_filename),
    #     file_type=FileType.EVENT_ALIGNED_NEURAL_DATA.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TIME_SERIES.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     parent_ids=[f.file_id for f in source_cellset_files],
    #     preview_files=[
    #         population_activity_preview_file,
    #         *event_aligned_population_activity_preview_files,
    #         *single_cell_activity_heatmap_preview_files,
    #     ],
    #     add_metadata=aligned_traces_metadata,
    # )

    # # statistics METADATA
    # stats_metadata = copy.deepcopy(aligned_traces_metadata)
    # del stats_metadata["ideas"]["dataset"]["signal"]

    # # statistics FILE
    # statistics_file = IdeasFile(
    #     file_key="event_aligned_statistics",
    #     file_path=os.path.abspath(output_stats_csv_filename),
    #     file_type=FileType.STATISTICS.value[1],
    #     file_format=FileFormat.CSV_FILE.value[1],
    #     file_structure=FileStructure.TABLE.value[1],
    #     file_category=FileCategory.RESULT.value[1],
    #     parent_ids=[
    #         source_cellset_files[
    #             0
    #         ].file_id,  # footprints retrieved from first cell set file
    #         event_aligned_traces_file.file_id,
    #     ],
    #     preview_files=[
    #         *activity_by_modulation_preview_files,
    #         *cell_map_preview_files,
    #         *event_aligned_by_modulation_across_epochs_preview_files,
    #         num_modulated_cells_per_epoch_preview_file,
    #         num_events_per_epoch_preview_file,
    #         post_minus_pre_per_epoch_preview_file,
    #     ],
    #     add_metadata=stats_metadata,
    # )

    # # gather output files to bundle under a single output group
    # output_files = [
    #     *source_cellset_files,
    #     source_events_file,
    #     event_aligned_traces_file,
    #     statistics_file,
    # ]

    # # epoch comparison METADATA
    # if num_epochs > 1:
    #     epoch_comparison_metadata = copy.deepcopy(stats_metadata)

    #     # epoch comparison FILE
    #     epoch_comparison_file = IdeasFile(
    #         file_key="event_aligned_epoch_comparison_data",
    #         file_path=os.path.abspath(output_epoch_comparison_csv_filename),
    #         file_type=FileType.PERI_EVENT_COMPARISON_DATA.value[1],
    #         file_format=FileFormat.CSV_FILE.value[1],
    #         file_structure=FileStructure.TABLE.value[1],
    #         file_category=FileCategory.RESULT.value[1],
    #         parent_ids=[
    #             source_cellset_files[
    #                 0
    #             ].file_id,  # footprints retrieved from first cell set file
    #             event_aligned_traces_file.file_id,
    #             statistics_file.file_id,
    #         ],
    #         preview_files=[
    #             *post_minus_pre_activity_differences_preview_files,
    #             post_minus_pre_boxplot_preview_file,
    #         ],
    #         add_metadata=epoch_comparison_metadata,
    #     )
    #     output_files.append(epoch_comparison_file)

    # # peri-event analysis across epochs GROUP
    # peri_event_analysis_group = IdeasGroup(
    #     group_key="compare_peri_event_activity_across_epochs_output",
    #     group_type=GroupType.TOOL_OUTPUT.value[1],
    #     files=output_files,
    # )
    # return peri_event_analysis_group

    output_metadata = {}
    output_metadata_path = os.path.join(output_dir, "output_metadata.json")
    if os.path.exists(output_metadata_path):
        with open(output_metadata_path, "r") as f:
            output_metadata = json.load(f)

    with open(output_metadata_path, "w") as f:
        json.dump(output_metadata, f)


def peri_event_population_analysis(
    traces_df,
    visual_window,
    statistical_window_indices,
    significance_threshold,
    x_values,
    x_limits,
    event_type,
    event_indices,
    output_dir,
    population_activity_plot_limits,
    epoch_data=None,
    boundaries=None,
):
    """Perform peri-event analysis at the population level.

    :param traces_df: dataframe containing individual cell traces
                      (rows=timepoints, columns=cells)
    :param event_indices: list of event indices
    :param visual_window: window of time to use for visualization purposes (pre, post)
    :param statistical_window_indices: window of time to use for statistical tests
                                      {'pre': [pre_event_indices], 'post': [post_event_indices]}
    :param significance_threshold: significance threshold to use for statistical tests
    :param x_values: values of the x-axis to use for plots
    :param x_limits: limits of the x-axis to use for plots
    :param event_type: type of event for which analysis was performed
    :param output_dir: path to the output directory
    :param epoch_data: dictionary containing epoch-specific information (keys are the epoch names)
    """
    # compute mean population activity
    # by averaging the traces of all neurons together across time
    mean_trace = traces_df.mean(axis=1)

    # perform population analysis for each epoch
    epoch_population_activity_preview_filenames = []
    output_data = {}
    for epoch_name, epoch_data_item in epoch_data.items():
        logger.info(f"Processing population data for epoch '{epoch_name}'")

        # retrieve event indices within epoch
        epoch_event_indices = epoch_data_item["event_indices"]
        epoch_event_indices_shuffles = epoch_data_item[
            "event_indices_shuffles"
        ]

        # Step 1: Compute true population mean activity across event windows
        # (i.e. windows around each event)
        # extract short traces around each event
        event_windows = extract_event_windows(
            mean_trace, epoch_event_indices, visual_window
        )

        # - Compute mean event window, i.e. we average all event windows across time.
        # - Also compute the standard error of the mean (sem) associated with each timepoint.
        mean_event_window = np.nanmean(event_windows, axis=0, dtype="float32")

        if event_windows.shape[0] > 1:
            sem_event_window = stats.sem(
                event_windows,
                axis=0,
                nan_policy="omit",
            ).astype("float32")
        else:
            # no SEM to compute if dealing with a single event
            sem_event_window = np.full_like(
                mean_event_window, math.nan, dtype="float32"
            )

        # Step 2: Construct the null distribution
        # - The null distribution is constructed by shuffling the events a number of times.
        # - For each shuffle, we then compute the mean event window as done above in step 1.
        # - We end up with 'num_shuffles' such mean event windows.
        # - Finally, we average all mean event windows and compute the associated confidence interval.

        # compute mean event window for N shuffles of the event indices
        shuffled_mean_event_windows = (
            extract_mean_event_windows_per_event_shuffle(
                mean_trace, epoch_event_indices_shuffles, visual_window
            )
        )

        # compute the mean event window across all shuffles
        shuffled_mean_event_window = np.nanmean(
            shuffled_mean_event_windows, axis=0, dtype="float32"
        )

        # - determine the lower and upper bound of the confidence interval
        #   for the mean event window across all shuffles
        # - this is achieved by computing the appropriate percentiles for each timepoint
        #   giving us a lower and upper bound for the mean event window of the distribution
        shuffled_mean_event_window_conf_interval_low = np.nanpercentile(
            shuffled_mean_event_windows,
            100 * (significance_threshold / 2),
            axis=0,
        )
        shuffled_mean_event_window_conf_interval_high = np.nanpercentile(
            shuffled_mean_event_windows,
            100 - (100 * significance_threshold / 2),
            axis=0,
        )

        # plot population mean event window +- sem
        # versus the shuffled mean event window and associated confidence interval
        plot_basename = os.path.join(
            output_dir,
            f"event_aligned_population_activity_{epoch_name.replace(' ', '')}",
        )
        plot_preview_filename = (
            plot_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
        )

        epoch_population_activity_preview_filenames.append(
            (epoch_name, plot_preview_filename)
        )

        plot_population_mean_event_window(
            x_values=x_values,
            x_limits=x_limits,
            population_mean=mean_event_window,
            population_sem=sem_event_window,
            shuffled_mean=shuffled_mean_event_window,
            shuffled_conf_interval_low=shuffled_mean_event_window_conf_interval_low,
            shuffled_conf_interval_high=shuffled_mean_event_window_conf_interval_high,
            significance_threshold=significance_threshold,
            event_type=event_type,
            output_filename=plot_preview_filename,
            color=epoch_data[epoch_name]["color"],
            epoch_name=epoch_name,
            plot_limits=population_activity_plot_limits,
        )

        # Step 3: Evaluate statistical significance of the difference between
        # true data and null distribution

        # - Compute mean activity difference between timepoints after the event (post)
        # and timepoints before (pre) the event.
        # - This is achieved by subtracting pre-event activity from post-event activity
        #   over the STATISTICAL window, and averaging the resulting trace to get a single
        #   value representing the "true" difference between pre- and post- event activity.
        true_post_minus_pre = compute_post_minus_pre(
            mean_event_window,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        # generate post-pre null distribution
        post_minus_pre_null_dist = generate_post_minus_pre_shuffled_dist(
            shuffled_mean_event_windows,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        # compute statistics for true versus null distribution
        (
            zscores,
            pvalues,
            modulations,
            ranks,
            post_minus_pre_null_dist_mu,
            post_minus_pre_null_dist_std,
        ) = compute_statistical_metrics(
            np.array([true_post_minus_pre]),
            np.array(post_minus_pre_null_dist).reshape(
                (len(post_minus_pre_null_dist), 1)
            ),
            significance_threshold,
        )

        # construct output data
        output_data[epoch_name] = {
            "mean": mean_event_window,
            "sem": sem_event_window,
            "shuffled_mean": shuffled_mean_event_window,
            "shuffled_lower_conf": shuffled_mean_event_window_conf_interval_low,
            "shuffled_upper_conf": shuffled_mean_event_window_conf_interval_high,
            "true_mean_post-pre": true_post_minus_pre,
            "shuffled_mean_post-pre": post_minus_pre_null_dist_mu[0],
            "shuffled_std_post-pre": post_minus_pre_null_dist_std[0],
            "z-score": zscores[0],
            "p-value": pvalues[0],
            "modulation": modulations[0],
        }

    # generate population-level comparison plot
    pop_act_comparison_plot_basename = os.path.join(
        output_dir, "event_aligned_population_activity"
    )
    pop_act_comparison_plot_preview_filename = (
        pop_act_comparison_plot_basename
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )
    plot_population_mean_event_window_across_epochs(
        x_values=x_values,
        x_limits=x_limits,
        data=output_data,
        event_type=event_type,
        output_filename=pop_act_comparison_plot_preview_filename,
        epoch_data=epoch_data,
        plot_limits=population_activity_plot_limits,
    )

    # plot population activity over the entire recording
    # and overlay epoch periods and events
    population_activity_plot_basename = os.path.join(
        output_dir, "population_activity"
    )
    population_activity_plot_preview_filename = (
        population_activity_plot_basename
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )
    plot_population_activity(
        mean_trace=mean_trace,
        event_indices=event_indices,
        epoch_data=epoch_data,
        output_filename=population_activity_plot_preview_filename,
        plot_limits=population_activity_plot_limits,
        boundaries=boundaries,
    )

    # plot the number of events per epoch
    num_events_per_epoch_plot_basename = os.path.join(
        output_dir, "event_count_per_epoch"
    )
    num_events_per_epoch_plot_preview_filename = (
        num_events_per_epoch_plot_basename
        + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
    )
    plot_number_of_events_per_epoch(
        epoch_data=epoch_data,
        output_filename=num_events_per_epoch_plot_preview_filename,
    )

    # create preview entries for output manifest

    # # population activity over the entire recording
    # population_activity_preview_file = IdeasPreviewFile(
    #     name="Mean population activity",
    #     help="Mean population activity over time. "
    #     "Shaded areas represent the different epochs. "
    #     "Individual events are displayed underneath the population activity trace.",
    #     file_path=os.path.abspath(population_activity_plot_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # # event-aligned population activity figures
    # event_aligned_population_activity_preview_files = []
    # # population activity across epochs on one figure
    # event_aligned_population_activity_preview_files.append(
    #     IdeasPreviewFile(
    #         name="Event-aligned population activity across epochs",
    #         help="Comparison of event-aligned average population activity across the epochs.",
    #         file_path=os.path.abspath(
    #             pop_act_comparison_plot_preview_filename
    #         ),
    #         file_format=FileFormat.SVG_FILE.value[1],
    #     )
    # )
    # # individual plot showing population activity for each epoch
    # for (
    #     epoch_name,
    #     preview_filename,
    # ) in epoch_population_activity_preview_filenames:
    #     event_aligned_population_activity_preview_files.append(
    #         IdeasPreviewFile(
    #             name="Event-aligned population activity",
    #             help=f"Event-aligned average population activity line plot (epoch: {epoch_name}).",
    #             file_path=os.path.abspath(preview_filename),
    #             file_format=FileFormat.SVG_FILE.value[1],
    #         )
    #     )

    # # preview showing number events per epoch
    # num_events_per_epoch_preview_file = IdeasPreviewFile(
    #     name="Number of events per epoch",
    #     help="Number of events in each epoch.",
    #     file_path=os.path.abspath(num_events_per_epoch_plot_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    return (
        output_data,
        # population_activity_preview_file,
        # event_aligned_population_activity_preview_files,
        # num_events_per_epoch_preview_file,
    )


def peri_event_single_cell_analysis(
    traces_df,
    footprints,
    visual_window,
    statistical_window_indices,
    significance_threshold,
    x_values,
    x_limits,
    event_type,
    output_dir,
    activity_heatmap_color_limits,
    activity_by_modulation_plot_limits,
    modulation_colors=None,
    cmap=None,
    epoch_data=None,
):
    """Perform peri-event analysis at the single-cell level.

    :param traces_df: dataframe containing individual cell traces
                      (rows=timepoints, columns=cells)
    :param footprints: footprints array (num_cells, width, height)
    :param event_indices: list of event indices
    :param event_indices_shuffles: shuffles of event indices to use to
                                   construct the null distribution
    :param visual_window: window of time to use for visualization purposes (pre, post)
    :param statistical_window_indices: window of time to use for statistical tests
                                      {'pre': [pre_event_indices], 'post': [post_event_indices]}
    :param significance_threshold: significance threshold to use for statistical tests
    :param x_values: values of the x-axis to use for plots
    :param x_limits: limits of the x-axis to use for plots
    :param event_type: type of event for which analysis was performed
    :param output_dir: path to the output directory
    :param modulation_colors: string list with color inputs
    :param cmap: colormap applied to the activity heatmap
    These colors represent [up-modulated, down-modulated, non-modulated] groups.
    :param epoch_data: dictionary containing epoch-specific information (keys are the epoch names)
    """
    # Provide a default value for modulation_colors if not passed
    modulation_colors = validate_modulation_colors(modulation_colors)

    cmap = (
        cmap
        if isinstance(cmap, str) and cmap in plt.colormaps()
        else (
            logger.warning(
                f"Invalid activity heatmap colormap '{cmap}', defaulting to 'coolwarm'"
            )
            or "coolwarm"
        )
    )

    # initialize output preview file lists
    # activity_heatmap_preview_files = []
    # activity_by_modulation_preview_files = []
    # cell_map_preview_files = []

    # perform population analysis for each epoch
    output_data = {}
    for epoch_name, epoch_data_item in epoch_data.items():
        logger.info(f"Processing single-cell data for epoch '{epoch_name}'")

        # retrieve event indices within epoch
        event_indices = epoch_data_item["event_indices"]
        event_indices_shuffles = epoch_data_item["event_indices_shuffles"]

        # extract short traces around each event
        # - this step returns an array with shape (num_events, num_timepoints_per_window, num_cells)
        event_windows = extract_event_windows(
            traces=traces_df,
            event_indices=event_indices,
            time_window=visual_window,
        )

        # compute mean & sem event window per cell, i.e. average activity across all windows for each
        # cell - this step returns an array with shape (num_timepoints_per_window, num_cells)
        mean_event_windows = np.nanmean(event_windows, axis=0, dtype="float32")

        if event_windows.shape[0] > 1:
            sem_event_windows = stats.sem(
                event_windows, axis=0, nan_policy="omit"
            ).astype("float32")
        else:
            # no SEM to compute if dealing with a single event
            sem_event_windows = np.full_like(
                mean_event_windows, math.nan, dtype="float32"
            )

        # - Compute mean activity difference between timepoints after the event (post)
        # and timepoints before (pre) the event.
        # - This is achieved by subtracting pre-event activity from post-event activity
        #   over the STATISTICAL window, and averaging the resulting trace to get a single
        #   value representing the "true" difference between pre- and post- event activity.
        # - The above procedure is repeated for all cells.
        # - This step returns a 1D array with length equal to 'num_cells', which contains
        #   the mean post_minus_pre value across all events.
        true_post_minus_pre_dist = compute_post_minus_pre(
            mean_event_windows,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        # compute mean event window for N shuffles of the event indices for each cell
        # - this returns an array with shape (num_shuffles, num_timepoints_per_window, num_cells)
        shuffled_mean_event_windows = (
            extract_mean_event_windows_per_event_shuffle(
                traces_df, event_indices_shuffles, visual_window
            )
        )

        # generate post-pre null distribution
        post_minus_pre_null_dist = generate_post_minus_pre_shuffled_dist(
            shuffled_mean_event_windows,
            statistical_window_indices["pre"],
            statistical_window_indices["post"],
        )

        # compute single-cell metrics
        (
            zscores,
            pvalues,
            modulations,
            ranks,
            post_minus_pre_null_dist_mu,
            post_minus_pre_null_dist_std,
        ) = compute_statistical_metrics(
            true_dist=true_post_minus_pre_dist,
            null_dist=post_minus_pre_null_dist,
            significance_threshold=significance_threshold,
        )

        # data transposed from (num_timepoints_per_window, num_cells)
        # to (num_cells, num_timepoints_per_window) for plotting purposes
        ordered_mean_event_windows = mean_event_windows.T[ranks]

        # generate heatmap showing event-aligned activity modulation per neuron
        heatmap_data_basename = os.path.join(
            output_dir,
            f"event_aligned_single_cell_activity_heatmap_{epoch_name.replace(' ', '')}",
        )
        heatmap_preview_filename = (
            heatmap_data_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
        )
        plot_single_neurons_heatmap(
            single_cell_data=ordered_mean_event_windows,
            x_limits=x_limits,
            event_type=event_type,
            output_filename=heatmap_preview_filename,
            cmap=cmap,
            epoch_name=epoch_name,
            color_limits=activity_heatmap_color_limits,
        )

        # compute mean and sem for up-, down-, and non- modulated traces
        up_modulated_indices = np.where(modulations == 1)[0]
        down_modulated_indices = np.where(modulations == -1)[0]
        non_modulated_indices = np.where(modulations == 0)[0]

        num_cells_up_modulated = len(up_modulated_indices)
        num_cells_down_modulated = len(down_modulated_indices)
        num_cells_non_modulated = len(non_modulated_indices)

        # up modulated MEAN
        if num_cells_up_modulated > 0:
            mean_up_modulated = np.nanmean(
                mean_event_windows[:, up_modulated_indices],
                axis=1,
                dtype="float32",
            )
        else:
            mean_up_modulated = np.empty(
                mean_event_windows.shape[0], dtype="float32"
            )
            mean_up_modulated.fill(np.nan)
        # up modulated SEM
        if num_cells_up_modulated > 1:
            sem_up_modulated = stats.sem(
                mean_event_windows[:, up_modulated_indices],
                axis=1,
                nan_policy="omit",
            ).astype("float32")
        else:
            sem_up_modulated = np.empty(
                mean_event_windows.shape[0], dtype="float32"
            )
            sem_up_modulated.fill(np.nan)

        # down modulated MEAN
        if num_cells_down_modulated > 0:
            mean_down_modulated = np.nanmean(
                mean_event_windows[:, down_modulated_indices],
                axis=1,
                dtype="float32",
            )
        else:
            mean_down_modulated = np.empty(
                mean_event_windows.shape[0], dtype="float32"
            )
            mean_down_modulated.fill(np.nan)
        # down modulated SEM
        if num_cells_down_modulated > 1:
            sem_down_modulated = stats.sem(
                mean_event_windows[:, down_modulated_indices],
                axis=1,
                nan_policy="omit",
            ).astype("float32")
        else:
            sem_down_modulated = np.empty(
                mean_event_windows.shape[0], dtype="float32"
            )
            sem_down_modulated.fill(np.nan)

        # non modulated MEAN
        if num_cells_non_modulated > 0:
            mean_non_modulated = np.nanmean(
                mean_event_windows[:, non_modulated_indices],
                axis=1,
                dtype="float32",
            )
        else:
            mean_non_modulated = np.empty(mean_event_windows.shape[0])
            mean_non_modulated.fill(np.nan)
        # non modulated SEM
        if num_cells_non_modulated > 1:
            sem_non_modulated = stats.sem(
                mean_event_windows[:, non_modulated_indices],
                axis=1,
                nan_policy="omit",
            ).astype("float32")
        else:
            sem_non_modulated = np.empty(
                mean_event_windows.shape[0], dtype="float32"
            )
            sem_non_modulated.fill(np.nan)

        # compute stats for each modulation group
        # up-modulated
        up_modulated_true_post_minus_pre = None
        up_modulated_zscores = [None]
        up_modulated_pvalues = [None]
        up_modulated_modulations = [None]
        up_modulated_post_minus_pre_null_dist_mu = [None]
        up_modulated_post_minus_pre_null_dist_std = [None]

        if len(up_modulated_indices) > 0:
            up_modulated_true_post_minus_pre = compute_post_minus_pre(
                event_windows=mean_up_modulated,
                pre_event_indices=statistical_window_indices["pre"],
                post_event_indices=statistical_window_indices["post"],
            )

            up_modulated_shuffled_mean_event_windows = nanmean_iterative(
                event_windows=shuffled_mean_event_windows,
                cell_indices=up_modulated_indices,
            )

            up_modulated_post_minus_pre_null_dist = (
                generate_post_minus_pre_shuffled_dist(
                    event_windows=up_modulated_shuffled_mean_event_windows,
                    pre_event_indices=statistical_window_indices["pre"],
                    post_event_indices=statistical_window_indices["post"],
                )
            )

            (
                up_modulated_zscores,
                up_modulated_pvalues,
                up_modulated_modulations,
                up_modulated_ranks,
                up_modulated_post_minus_pre_null_dist_mu,
                up_modulated_post_minus_pre_null_dist_std,
            ) = compute_statistical_metrics(
                true_dist=np.array([up_modulated_true_post_minus_pre]),
                null_dist=np.array(
                    up_modulated_post_minus_pre_null_dist
                ).reshape((len(up_modulated_post_minus_pre_null_dist), 1)),
                significance_threshold=significance_threshold,
            )

        # down-modulated
        down_modulated_true_post_minus_pre = None
        down_modulated_zscores = [None]
        down_modulated_pvalues = [None]
        down_modulated_modulations = [None]
        down_modulated_post_minus_pre_null_dist_mu = [None]
        down_modulated_post_minus_pre_null_dist_std = [None]

        if len(down_modulated_indices) > 0:
            down_modulated_true_post_minus_pre = compute_post_minus_pre(
                event_windows=mean_down_modulated,
                pre_event_indices=statistical_window_indices["pre"],
                post_event_indices=statistical_window_indices["post"],
            )

            down_modulated_shuffled_mean_event_windows = nanmean_iterative(
                event_windows=shuffled_mean_event_windows,
                cell_indices=down_modulated_indices,
            )

            down_modulated_post_minus_pre_null_dist = (
                generate_post_minus_pre_shuffled_dist(
                    event_windows=down_modulated_shuffled_mean_event_windows,
                    pre_event_indices=statistical_window_indices["pre"],
                    post_event_indices=statistical_window_indices["post"],
                )
            )

            (
                down_modulated_zscores,
                down_modulated_pvalues,
                down_modulated_modulations,
                down_modulated_ranks,
                down_modulated_post_minus_pre_null_dist_mu,
                down_modulated_post_minus_pre_null_dist_std,
            ) = compute_statistical_metrics(
                true_dist=np.array([down_modulated_true_post_minus_pre]),
                null_dist=np.array(
                    down_modulated_post_minus_pre_null_dist
                ).reshape((len(down_modulated_post_minus_pre_null_dist), 1)),
                significance_threshold=significance_threshold,
            )

        # non-modulated
        non_modulated_true_post_minus_pre = None
        non_modulated_zscores = [None]
        non_modulated_pvalues = [None]
        non_modulated_modulations = [None]
        non_modulated_post_minus_pre_null_dist_mu = [None]
        non_modulated_post_minus_pre_null_dist_std = [None]

        if len(non_modulated_indices) > 0:
            non_modulated_true_post_minus_pre = compute_post_minus_pre(
                event_windows=mean_non_modulated,
                pre_event_indices=statistical_window_indices["pre"],
                post_event_indices=statistical_window_indices["post"],
            )

            non_modulated_shuffled_mean_event_windows = nanmean_iterative(
                event_windows=shuffled_mean_event_windows,
                cell_indices=non_modulated_indices,
            )

            non_modulated_post_minus_pre_null_dist = (
                generate_post_minus_pre_shuffled_dist(
                    event_windows=non_modulated_shuffled_mean_event_windows,
                    pre_event_indices=statistical_window_indices["pre"],
                    post_event_indices=statistical_window_indices["post"],
                )
            )

            (
                non_modulated_zscores,
                non_modulated_pvalues,
                non_modulated_modulations,
                non_modulated_ranks,
                non_modulated_post_minus_pre_null_dist_mu,
                non_modulated_post_minus_pre_null_dist_std,
            ) = compute_statistical_metrics(
                true_dist=np.array([non_modulated_true_post_minus_pre]),
                null_dist=np.array(
                    non_modulated_post_minus_pre_null_dist
                ).reshape((len(non_modulated_post_minus_pre_null_dist), 1)),
                significance_threshold=significance_threshold,
            )

        # plot (mean +- sem) per modulation group (up, down, non)
        modulation_plot_basename = os.path.join(
            output_dir,
            f"event_aligned_activity_by_modulation_{epoch_name.replace(' ', '')}",
        )
        modulation_plot_preview_filename = (
            modulation_plot_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
        )
        plot_single_cell_modulation(
            x=x_values,
            x_limits=x_limits,
            mean_up=mean_up_modulated,
            sem_up=sem_up_modulated,
            num_cells_up_modulated=num_cells_up_modulated,
            mean_down=mean_down_modulated,
            sem_down=sem_down_modulated,
            num_cells_down_modulated=num_cells_down_modulated,
            mean_non=mean_non_modulated,
            sem_non=sem_non_modulated,
            num_cells_non_modulated=num_cells_non_modulated,
            event_type=event_type,
            output_filename=modulation_plot_preview_filename,
            modulation_colors=modulation_colors,
            epoch_name=epoch_name,
            plot_limits=activity_by_modulation_plot_limits,
        )

        # compute centroid coordinates for every cell
        num_cells_footprints = footprints.shape[0]
        cell_centroids = np.zeros((num_cells_footprints, 2), dtype="float32")
        for i in range(num_cells_footprints):
            cell_centroid = compute_cell_centroid(footprints[i])
            if cell_centroid is None:
                cell_centroids[i] = [-1, -1]
                logger.warning(
                    "centroid of cell {0} could not be located".format(i)
                )
            else:
                cell_centroids[i] = cell_centroid

        # generate cell map colored by modulation group
        cell_map_basename = os.path.join(
            output_dir, f"cell_map_{epoch_name.replace(' ', '')}"
        )
        cell_map_preview_filename = (
            cell_map_basename + config.OUTPUT_PREVIEW_SVG_FILE_EXTENSION
        )

        plot_cell_map(
            footprints=footprints,
            modulations=modulations,
            event_type=event_type,
            output_filename=cell_map_preview_filename,
            modulation_colors=modulation_colors,
            epoch_name=epoch_name,
        )

        # construct output data
        output_data[epoch_name] = {
            "up_modulated": {
                "mean": mean_up_modulated,
                "sem": sem_up_modulated,
                "num_cells": num_cells_up_modulated,
                "true_mean_post-pre": up_modulated_true_post_minus_pre,
                "shuffled_mean_post-pre": up_modulated_post_minus_pre_null_dist_mu[
                    0
                ],
                "shuffled_std_post-pre": up_modulated_post_minus_pre_null_dist_std[
                    0
                ],
                "z-score": up_modulated_zscores[0],
                "p-value": up_modulated_pvalues[0],
                "modulation": up_modulated_modulations[0],
                "indices": up_modulated_indices,
            },
            "down_modulated": {
                "mean": mean_down_modulated,
                "sem": sem_down_modulated,
                "num_cells": num_cells_down_modulated,
                "true_mean_post-pre": down_modulated_true_post_minus_pre,
                "shuffled_mean_post-pre": down_modulated_post_minus_pre_null_dist_mu[
                    0
                ],
                "shuffled_std_post-pre": down_modulated_post_minus_pre_null_dist_std[
                    0
                ],
                "z-score": down_modulated_zscores[0],
                "p-value": down_modulated_pvalues[0],
                "modulation": down_modulated_modulations[0],
                "indices": down_modulated_indices,
            },
            "non_modulated": {
                "mean": mean_non_modulated,
                "sem": sem_non_modulated,
                "num_cells": num_cells_non_modulated,
                "true_mean_post-pre": non_modulated_true_post_minus_pre,
                "shuffled_mean_post-pre": non_modulated_post_minus_pre_null_dist_mu[
                    0
                ],
                "shuffled_std_post-pre": non_modulated_post_minus_pre_null_dist_std[
                    0
                ],
                "z-score": non_modulated_zscores[0],
                "p-value": non_modulated_pvalues[0],
                "modulation": non_modulated_modulations[0],
                "indices": non_modulated_indices,
            },
            "cell": {  # data for individual cells
                "mean": mean_event_windows.T,  # transposed to return (num_cells, num_points_per_window)
                "sem": sem_event_windows.T,  # transposed to return (num_cells, num_points_per_window)
                "true_mean_post-pre": true_post_minus_pre_dist,
                "shuffled_mean_post-pre": post_minus_pre_null_dist_mu,
                "shuffled_std_post-pre": post_minus_pre_null_dist_std,
                "z-score": zscores,
                "p-value": pvalues,
                "modulation": modulations,
                "rank": ranks,
            },
        }

        # create preview entries for output manifest

        # # create preview file for single-cell activity heatmap
        # activity_heatmap_preview_files.append(
        #     IdeasPreviewFile(
        #         name="Event-aligned single-cell activity figure",
        #         help=f"Event-aligned single-cell activity heatmap (epoch: {epoch_name})",
        #         file_path=os.path.abspath(heatmap_preview_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        # )

        # # create preview for event-aligned activity by modulation group
        # activity_by_modulation_preview_files.append(
        #     IdeasPreviewFile(
        #         name="Event-aligned sub-population activity figure",
        #         help="Event-aligned average sub-population activity line plot "
        #         f"(up-, down-, and non-modulated neurons) (epoch: {epoch_name}).",
        #         file_path=os.path.abspath(modulation_plot_preview_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        # )

        # # cell map
        # cell_map_preview_files.append(
        #     IdeasPreviewFile(
        #         name="Spatial organization of modulation",
        #         help=f"Cell map visualizing spatial organization of modulation (epoch: {epoch_name}).",
        #         file_path=os.path.abspath(cell_map_preview_filename),
        #         file_format=FileFormat.SVG_FILE.value[1],
        #     )
        # )

    # plot mean event-aligned activity of up/down/non modulated cells in each epoch together
    event_aligned_modulation_groups_across_epoch_files = (
        plot_event_aligned_activity_by_modulation_group_comparing_epochs(
            x=x_values,
            x_limits=x_limits,
            data=output_data,
            event_type=event_type,
            epoch_data=epoch_data,
            output_dir=output_dir,
            plot_limits=activity_by_modulation_plot_limits,
        )
    )

    # event_aligned_by_modulation_across_epochs_preview_files = []
    # for (
    #     group_title,
    #     output_filename,
    # ) in event_aligned_modulation_groups_across_epoch_files:
    #     event_aligned_by_modulation_across_epochs_preview_files.append(
    #         IdeasPreviewFile(
    #             name=f"Event-aligned activity of {group_title.lower()} cells",
    #             help=f"Comparison of event-aligned activity of {group_title.lower()} cells across epochs.",
    #             file_path=os.path.abspath(output_filename),
    #             file_format=FileFormat.SVG_FILE.value[1],
    #         )
    #     )

    # plot post-pre per epoch
    post_minus_pre_per_epoch_preview_filename = os.path.join(
        output_dir, "mean_post_minus_pre_activity_per_epoch.svg"
    )
    plot_post_minus_pre_per_epoch_bar_chart(
        data=output_data,
        epoch_data=epoch_data,
        output_filename=post_minus_pre_per_epoch_preview_filename,
    )
    # post_minus_pre_per_epoch_preview_file = IdeasPreviewFile(
    #     name="Mean post-pre activity per epoch",
    #     help="Comparison of mean post-pre activity across the epochs. The error bars represent the standard error of the mean.",
    #     file_path=os.path.abspath(post_minus_pre_per_epoch_preview_filename),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    # plot the number of modulated cells per epoch
    num_modulated_cells_per_epoch_preview_filename = os.path.join(
        output_dir, "num_modulated_cells_per_epoch.svg"
    )
    plot_number_of_modulated_cells_per_epoch(
        data=output_data,
        epoch_data=epoch_data,
        output_filename=num_modulated_cells_per_epoch_preview_filename,
    )
    # num_modulated_cells_per_epoch_preview_file = IdeasPreviewFile(
    #     name="Number of modulated cells per epoch",
    #     help="Number of up-, down-, and non-modulated neurons per epoch.",
    #     file_path=os.path.abspath(
    #         num_modulated_cells_per_epoch_preview_filename
    #     ),
    #     file_format=FileFormat.SVG_FILE.value[1],
    # )

    return (
        output_data,
        # activity_heatmap_preview_files,
        # activity_by_modulation_preview_files,
        # cell_map_preview_files,
        # num_modulated_cells_per_epoch_preview_file,
        # event_aligned_by_modulation_across_epochs_preview_files,
        # post_minus_pre_per_epoch_preview_file,
    )


def save_event_aligned_traces_to_csv(
    event_aligned_data, x_values, valid_cells, output_filename
):
    """Save event-aligned traces to a csv file.
    :param event_aligned_data: dictionary containing event-aligned data
    :param x_values: values along the x-axis
    :param valid_cells: list of cell names that were processed
    :param output_filename: path to the output csv file
    """
    # retrieve epoch names and initialize epoch-specific data
    epoch_names = list(event_aligned_data["population"].keys())

    # initialize output data and columns with 'Time' values
    data = [x_values]
    columns = [("Time", "")]

    # for each epoch, gather population/group/single-cell data
    for epoch_name in epoch_names:

        # population and modulation subgroups
        for group_name in [
            "population",
            "up_modulated",
            "down_modulated",
            "non_modulated",
        ]:
            if group_name == "population":
                # Add population data

                population_mean_label = "{0}_mean".format(group_name)
                population_sem_label = "{0}_sem".format(group_name)
                shuffled_mean_label = "{0}_shuffled_mean".format(group_name)
                shuffled_lower_conf_label = "{0}_shuffled_lower_conf".format(
                    group_name
                )
                shuffled_upper_conf_label = "{0}_shuffled_upper_conf".format(
                    group_name
                )

                # population mean
                columns.append((epoch_name, population_mean_label))
                data.append(event_aligned_data[group_name][epoch_name]["mean"])

                # population sem
                columns.append((epoch_name, population_sem_label))
                data.append(event_aligned_data[group_name][epoch_name]["sem"])

                # shuffled mean
                columns.append((epoch_name, shuffled_mean_label))
                data.append(
                    event_aligned_data[group_name][epoch_name]["shuffled_mean"]
                )

                # shuffled lower conf
                columns.append((epoch_name, shuffled_lower_conf_label))
                data.append(
                    event_aligned_data[group_name][epoch_name][
                        "shuffled_lower_conf"
                    ]
                )

                # shuffled upper conf
                columns.append((epoch_name, shuffled_upper_conf_label))
                data.append(
                    event_aligned_data[group_name][epoch_name][
                        "shuffled_upper_conf"
                    ]
                )
            else:
                # Add group data

                subgroup_mean_label = "{0}_mean".format(group_name)
                subgroup_sem_label = "{0}_sem".format(group_name)

                # group mean
                columns.append((epoch_name, subgroup_mean_label))
                data.append(
                    event_aligned_data["single_cell"][epoch_name][group_name][
                        "mean"
                    ]
                )

                # group sem
                columns.append((epoch_name, subgroup_sem_label))
                data.append(
                    event_aligned_data["single_cell"][epoch_name][group_name][
                        "sem"
                    ]
                )

        # single-cell data
        for i, cell_name in enumerate(valid_cells):
            single_cell_mean_label = "{0}_mean".format(cell_name)
            single_cell_sem_label = "{0}_sem".format(cell_name)

            # single cell mean
            columns.append((epoch_name, single_cell_mean_label))
            data.append(
                event_aligned_data["single_cell"][epoch_name]["cell"]["mean"][
                    i
                ]
            )

            # single cell sem
            columns.append((epoch_name, single_cell_sem_label))
            data.append(
                event_aligned_data["single_cell"][epoch_name]["cell"]["sem"][i]
            )

    # convert activity dict to dataframe

    # transpose data as the DataFrame constructor expects a list of rows
    data = np.array(
        data
    ).T  # transpose data as the DataFrame constructor expects a list of rows

    # convert column tuples to multi-level column indices
    columns = pd.MultiIndex.from_tuples(columns)

    output_activity_dataframe = pd.DataFrame(data, columns=columns)
    output_activity_dataframe.to_csv(output_filename, index=False)


def save_event_aligned_statistics_to_csv(
    event_aligned_data, valid_cells, output_filename
):
    """Save event-aligned statistics to a csv file.
    :param event_aligned_data: dictionary containing event-aligned data
    :param valid_cells: list of cell names for cells that were processed
    :param output_filename: path to the output csv file
    """
    # retrieve epoch names and initialize epoch-specific data
    epoch_names = list(event_aligned_data["population"].keys())

    # initialize list of dataframes to concatenante, 1 per epoch
    dfs = []

    # for each epoch, extract statistics contained within the input event-aligned data
    for epoch_name in epoch_names:

        # create dictionary to hold statistics data
        epoch_dict = OrderedDict()

        # add 1 column per metric, rows will correspond to cells or groups
        epoch_dict["name"] = [
            "population",
            "up_modulated",
            "down_modulated",
            "non_modulated",
        ] + valid_cells

        # add epoch column containing the epoch name
        epoch_dict["epoch"] = [epoch_name] * len(epoch_dict["name"])

        # gather populate metric data
        for col_name in [
            "true_mean_post-pre",
            "shuffled_mean_post-pre",
            "shuffled_std_post-pre",
            "z-score",
            "p-value",
            "modulation",
        ]:
            epoch_dict[col_name] = [
                event_aligned_data["population"][epoch_name][col_name],
                event_aligned_data["single_cell"][epoch_name]["up_modulated"][
                    col_name
                ],
                event_aligned_data["single_cell"][epoch_name][
                    "down_modulated"
                ][col_name],
                event_aligned_data["single_cell"][epoch_name]["non_modulated"][
                    col_name
                ],
            ] + list(
                event_aligned_data["single_cell"][epoch_name]["cell"][col_name]
            )

        # convert dictionary to dataframe
        epoch_dataframe = pd.DataFrame(epoch_dict)
        dfs.append(epoch_dataframe)

    # combine epoch dataframes into a single dataframe
    output_stats_dataframe = pd.concat(dfs)

    # save stats dataframe to disk
    output_stats_dataframe.to_csv(output_filename, index=False)
