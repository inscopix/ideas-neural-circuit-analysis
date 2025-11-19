"""Combined State-Epoch Analysis Tool with Baseline Comparison.

Analyzes neural activity patterns across behavioral states and time epochs,
integrating correlation and population activity analysis with baseline comparisons.
"""

import os
import json
import logging
import pathlib
from beartype.typing import List, Optional
from ideas.exceptions import IdeasError
from utils.state_epoch_data import (
    StateEpochDataManager,
    validate_input_files_exist,
    scale_data,
)
from utils.state_epoch_results import (
    StateEpochResults,
    analyze_state_epoch_combination,
    calculate_baseline_modulation,
)
from utils.state_epoch_output import StateEpochOutputGenerator
# from ideas.tools.log import get_logger
from ideas.tools.log import get_logger
from ideas.tools import outputs


logger = get_logger()


# @beartype
def state_epoch_baseline_analysis(
    *,
    # Core inputs (consistent with existing tools)
    cell_set_files: List[pathlib.Path],
    event_set_files: Optional[List[pathlib.Path]] = None,
    annotations_file: Optional[List[pathlib.Path]] = None,
    # State definition
    column_name: str = "state",
    state_names: str,  # "rest, exploration, feeding"
    state_colors: str = "gray, blue, orange",
    # Epoch definition
    define_epochs_by: str = "global file time",
    epochs: str = "(0, 600), (650, 950), (950, 1250)",
    epoch_names: str,  # "baseline, training, test"
    epoch_colors: str = "lightgray, lightblue, lightgreen",
    # Baseline specification (user-defined)
    baseline_state: str = "rest",  # Which state to use as baseline
    baseline_epoch: str = "baseline",  # Which epoch to use as baseline
    # Data preprocessing
    concatenate: bool = True,
    trace_scale_method: str = "none",
    event_scale_method: str = "none",
    # Analysis options
    include_correlations: bool = True,
    include_population_activity: bool = True,
    include_event_analysis: bool = True,
    # Registration support (for CaImAn MSR outputs)
    use_registered_cellsets: bool = False,
    registration_method: str = "auto_detect",  # "auto_detect", "caiman_msr"
    # Statistical parameters
    alpha: float = 0.05,
    n_shuffle: int = 1000,
    # Data processing parameters
    tolerance: float = 1e-4,
    sort_by_time: bool = True,
    output_dir: str = ""
) -> None:
    """Perform combined state-epoch analysis with baseline comparison.

    This tool analyzes neural activity patterns across behavioral states
    and time epochs, calculating:
    1. Activity/event rates per state-epoch combination
    2. Correlations per state-epoch combination
    3. Modulation indices relative to baseline state-epoch

    Args:
    ----
        cell_set_files: List of cellset files (.isxd, .h5)
        event_set_files: Optional list of eventset files (.isxd)
        annotations_file: Optional list of annotation files (.parquet) - only first file is used.
            If None, will use epoch-only analysis mode with dummy state
        column_name: Column name for state annotations
        state_names: Comma-separated state names
        state_colors: Comma-separated color names for states
        define_epochs_by: Method for defining epochs ("global file time",
            "files", "local file time")
        epochs: Epoch time periods as a string representation of tuples
        epoch_names: Comma-separated epoch names
        epoch_colors: Comma-separated color names for epochs
        baseline_state: State to use as baseline for modulation
        baseline_epoch: Epoch to use as baseline for modulation
        concatenate: Whether to concatenate multiple files
        trace_scale_method: Method for scaling traces
            ("none", "normalize", "standardize", "fractional_change", "standardize_baseline")
        event_scale_method: Method for scaling events
            ("none", "normalize", "standardize", "fractional_change", "standardize_baseline")
        include_correlations: Whether to compute correlation analysis
        include_population_activity: Whether to compute population activity
        include_event_analysis: Whether to analyze events (if provided)
        use_registered_cellsets: Whether to use registered cellset data
        registration_method: Method for cell registration
        alpha: Significance level for statistical tests
        n_shuffle: Number of permutations for statistical tests
        tolerance: Tolerance for temporal alignment and file concatenation
        sort_by_time: Whether to sort cellsets by time when concatenating
        output_dir: Directory for output files (empty for current directory)

    Raises
    ------
        IdeasError: If inputs are invalid or analysis fails
        FileNotFoundError: If input files don't exist

    """
    logger.info("Starting state-epoch baseline analysis...")

    # Validate inputs
    try:
        validate_input_files_exist(cell_set_files)
        if event_set_files:
            validate_input_files_exist(event_set_files)
        if annotations_file:
            validate_input_files_exist(annotations_file)
    except FileNotFoundError as e:
        raise IdeasError(f"Input validation failed: {e}") from e

    # Parse parameter strings
    states = [s.strip() for s in state_names.split(",")]
    parsed_epochs = [e.strip() for e in epoch_names.split(",")]
    state_color_list = [c.strip() for c in state_colors.split(",")]
    epoch_color_list = [c.strip() for c in epoch_colors.split(",")]

    # Handle epoch-only mode (consistent with correlations.py)
    epoch_only_mode = (
        annotations_file is None
        or len(annotations_file) == 0
        or annotations_file[0] is None
        or len(states) == 0
        or (len(states) == 1 and states[0].strip() == "")
    )

    if epoch_only_mode:
        logger.info(
            "No annotations provided or no states specified. "
            "Using epoch-only analysis mode with dummy state 'epoch_activity'."
        )
        # Use dummy state for epoch-only analysis (consistent with epoch_activity.py)
        states = ["epoch_activity"]
        state_color_list = ["gray"]  # Default color for dummy state
        baseline_state = "epoch_activity"  # Update baseline to match
        column_name = "dummy_state"  # Use dummy column name

    data_manager = StateEpochDataManager(
        cell_set_files=cell_set_files,
        event_set_files=event_set_files,
        annotations_file=annotations_file,
        concatenate=concatenate,
        use_registered_cellsets=use_registered_cellsets,
        registration_method=registration_method,
        # Validation parameters (passed directly to data manager)
        epochs=epochs,
        epoch_names=parsed_epochs,
        epoch_colors=epoch_color_list,
        state_names=states,
        state_colors=state_color_list,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        define_epochs_by=define_epochs_by,
        tolerance=tolerance,
        sort_by_time=sort_by_time,
    )

    # Load all data with automatic validation
    traces, events, annotations_df, cell_info = data_manager.load_data()

    logger.info(
        f"Analyzing {len(states)} states x {len(parsed_epochs)} epochs "
        f"(baseline: {baseline_state} in {baseline_epoch})"
    )

    # Scale data if requested
    if trace_scale_method != "none":
        traces = scale_data(
            traces,
            method=trace_scale_method,
            behavior=annotations_df,
            column_name=column_name,
            baseline_state=baseline_state,
        )
    if events is not None and event_scale_method != "none":
        events = scale_data(
            events,
            method=event_scale_method,
            behavior=annotations_df,
            column_name=column_name,
            baseline_state=baseline_state,
        )

    # Initialize results container
    results = StateEpochResults()

    # Analyze each state-epoch combination
    logger.info("Analyzing state-epoch combinations...")
    for state in states:
        for epoch in parsed_epochs:
            logger.info(f"Processing {state}-{epoch}...")

            # Extract data for this combination
            state_epoch_data = data_manager.extract_state_epoch_data(
                annotations_df=annotations_df,
                traces=traces,
                events=events,
                state=state,
                epoch=epoch,
                column_name=column_name,
                parsed_epochs=parsed_epochs,
            )

            # Analyze the combination
            combination_results = analyze_state_epoch_combination(
                state_epoch_data=state_epoch_data,
                state=state,
                epoch=epoch,
                cell_info=cell_info,
                include_correlations=include_correlations,
                include_population_activity=include_population_activity,
                include_event_analysis=include_event_analysis,
                alpha=alpha,
                n_shuffle=n_shuffle,
            )

            # Store results
            results.add_combination_results(state, epoch, combination_results)

    # Validate baseline by attempting to get baseline data
    logger.info("Validating baseline availability...")
    try:
        baseline_data = results.get_baseline_data(
            baseline_state, baseline_epoch
        )
        num_timepoints = baseline_data.get("num_timepoints", 0)
        logger.info(
            f"Baseline validation passed: {baseline_state}-{baseline_epoch} "
            f"available with {num_timepoints} timepoints"
        )

        # Additional validation for baseline data quality
        if num_timepoints == 0:
            logger.warning(
                f"Baseline {baseline_state}-{baseline_epoch} has no timepoints"
            )
        elif num_timepoints < 10:  # MIN_TIMEPOINTS_FOR_RELIABLE_ANALYSIS
            logger.warning(
                f"Baseline {baseline_state}-{baseline_epoch} has very few timepoints "
                f"({num_timepoints}) - modulation analysis may be unreliable"
            )

    except IdeasError as e:
        logger.error(f"Baseline validation failed: {e}")
        raise e

    # Calculate baseline modulation
    logger.info("Calculating baseline modulation...")
    modulation_results = calculate_baseline_modulation(
        results=results,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        cell_info=cell_info,
        alpha=alpha,
        n_shuffle=n_shuffle,
    )

    # Generate outputs
    logger.info("Generating outputs...")
    output_generator = StateEpochOutputGenerator(
        output_dir=output_dir,
        states=states,
        epochs=parsed_epochs,
        state_colors=state_color_list,
        epoch_colors=epoch_color_list,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        alpha=alpha,
        n_shuffle=n_shuffle,
        epoch_periods=data_manager.get_epoch_periods(),
    )

    output_generator.generate_all_outputs(
        results=results,
        modulation_results=modulation_results,
        cell_info=cell_info,
        traces=traces,
        events=events,
        annotations_df=annotations_df,
        column_name=column_name,
    )

    logger.info("Saving output data")

    output_metadata = {}
    output_metadata_path = os.path.join(output_dir, "output_metadata.json")
    if os.path.exists(output_metadata_path):
        with open(output_metadata_path, "r") as f:
            output_metadata = json.load(f)
        os.remove(os.path.join(output_dir, "output_metadata.json"))

    # generate basename for output files based on input cell sets
    output_file_basename = ""
    for cell_set_file in cell_set_files:
        cell_set_file_name = os.path.basename(cell_set_file)
        cell_set_file_name, _ = os.path.splitext(cell_set_file_name)
        cell_set_file_name = cell_set_file_name.replace(" ", "_")

        if output_file_basename:
            output_file_basename += "_"
        output_file_basename += cell_set_file_name + "_"

    try:
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "activity_per_state_epoch_data.csv",
                subdir="activity_per_state_epoch_data",
                prefix=output_file_basename
            ).register_preview(
                "population_average_preview.svg",
                caption="Average neural activity across state-epoch combinations. Bar plot showing mean activity ± standard error for each state-epoch combination."
            ).register_preview(
                "event_average_preview.svg",
                caption="Average event rates across state-epoch combinations. Box plots showing event frequency distributions when event data is available."
            ).register_preview(
                "trace_state_overlay.svg",
                caption="Trace preview with state-colored traces showing neural activity patterns colored by behavioral state."
            ).register_preview(
                "trace_epoch_overlay.svg",
                caption="Trace preview with epoch overlay boxes showing neural activity patterns with epoch time periods highlighted."
            ).register_preview(
                "event_state_overlay.svg",
                caption="Event raster plot with state-colored events showing event patterns colored by behavioral state."
            ).register_preview(
                "event_epoch_overlay.svg",
                caption="Combined event visualization with population average activity and raster plot showing event patterns with epoch time periods highlighted, formatted similar to Event_State_Overlay."
            ).register_metadata_dict(
                **output_metadata["activity_per_state_epoch_data"]
            )

            output_data.register_file(
                "correlations_per_state_epoch_data.csv",
                subdir="correlations_per_state_epoch_data",
                prefix=output_file_basename
            ).register_preview(
                "spatial_correlation_preview.svg",
                caption="Relationship between spatial distance and neural correlation across different state-epoch combinations. Scatter plots show pairwise neural correlation versus physical distance between cell centroids."
            ).register_preview(
                "spatial_correlation_map_preview.svg",
                caption="Spatial map of neural correlations across state-epoch combinations. Colored lines connect neuron pairs above the correlation threshold, with line color indicating correlation strength."
            ).register_preview(
                "correlation_matrices_preview.svg",
                caption="Pairwise Pearson correlation matrices between neurons for each state-epoch combination. Shows how neural correlations change across different conditions."
            ).register_preview(
                "average_correlations_preview.svg",
                caption="Bar plots showing average positive and negative correlations for each state-epoch combination. Provides summary statistics of correlation patterns."
            ).register_metadata_dict(
                **output_metadata["correlations_per_state_epoch_data"]
            )
            
            output_data.register_file(
                "modulation_vs_baseline_data.csv",
                subdir="modulation_vs_baseline_data",
                prefix=output_file_basename
            ).register_preview(
                "modulation_histogram_preview.svg",
                caption="Distribution of modulation scores across neurons relative to baseline. Histograms show the modulation scores for all neurons compared to the baseline state-epoch combination."
            ).register_preview(
                "modulation_footprint_preview.svg",
                caption="Spatial distribution of modulated neurons relative to baseline. Cell footprints are colored by modulation direction and significance relative to the baseline state-epoch combination."
            ).register_preview(
                "event_modulation_preview.svg",
                caption="Spatial footprints of event-modulated neurons relative to baseline. Cell maps colored by event modulation significance when event data is available."
            ).register_preview(
                "event_modulation_histogram_preview.svg",
                caption="Distribution of event modulation scores across neurons relative to baseline. Histograms showing event modulation when event data is available."
            ).register_metadata_dict(
                **output_metadata["modulation_vs_baseline_data"]
            )

            output_data.register_file("average_correlations.csv", prefix=output_file_basename).register_metadata_dict(
                **output_metadata["average_correlations"]
            )
            output_data.register_file("pairwise_correlation_heatmaps.h5", prefix=output_file_basename).register_metadata_dict(
                **output_metadata["pairwise_correlation_heatmaps"]
            )
            output_data.register_file("spatial_analysis_pairwise_correlations.zip", prefix=output_file_basename).register_metadata_dict(
                **output_metadata["spatial_analysis_pairwise_correlations"]
            )
    except Exception:
        logger.exception("Failed to generate output data!")

    logger.info("State-epoch baseline analysis completed successfully")


analyze = state_epoch_baseline_analysis
