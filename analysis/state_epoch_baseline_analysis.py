"""Combined State-Epoch Analysis Tool with Baseline Comparison.

Analyzes neural activity patterns across behavioral states and time epochs,
integrating correlation and population activity analysis with baseline comparisons.
"""

import json
import logging
import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from beartype.typing import List, Optional, Iterable, Tuple
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
from utils.state_epoch_output import (
    StateEpochOutputGenerator,
    STATE_EPOCH_TIME_PREVIEW,
    TRACE_POPULATION_AVERAGE_PREVIEW,
    TRACE_STATE_OVERLAY,
    EVENT_POPULATION_AVERAGE_PREVIEW,
    EVENT_STATE_OVERLAY,
    TRACE_MODULATION_FOOTPRINT_PREVIEW,
    TRACE_MODULATION_HISTOGRAM_PREVIEW,
    EVENT_MODULATION_PREVIEW,
    EVENT_MODULATION_HISTOGRAM_PREVIEW,
    CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
    EVENT_CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
    AVERAGE_CORRELATIONS_PREVIEW,
    EVENT_AVERAGE_CORRELATIONS_PREVIEW,
    CORRELATION_MATRICES_PREVIEW,
    EVENT_CORRELATION_MATRICES_PREVIEW,
    SPATIAL_CORRELATION_PREVIEW,
    SPATIAL_CORRELATION_MAP_PREVIEW,
    EVENT_SPATIAL_CORRELATION_PREVIEW,
    EVENT_SPATIAL_CORRELATION_MAP_PREVIEW,
    CORRELATIONS_PER_STATE_EPOCH_DATA_CSV,
    AVERAGE_CORRELATIONS_CSV,
    RAW_CORRELATIONS_H5_NAME,
    RAW_CORRELATIONS_ZIP_NAME,
)
from ideas.utils import _set_up_logger
from ideas.outputs import OutputData
from utils.utils import Comp


_set_up_logger()
logger = logging.getLogger(__name__)


SUPPORTED_STATE_COMPARISON_METHOD = Comp.BASELINE.value

USEFUL_OUTPUT_METADATA_KEYS = {
    # "analysis_type",
    # "file_type",
    # "description",
    "num_cells",
    "num_states",
    "num_epochs",
    "num_combinations",
    "combinations_analyzed",
    "baseline_state",
    "baseline_epoch",
    # "baseline_reference",
    "alpha",
    "n_shuffle",
    "correlation_method",
    "correlation_statistic",
    "modulation_method",
    # "activity_metrics",
    # "format",
}


def _extract_useful_metadata(metadata: dict) -> dict:
    """Return only the metadata fields we want to register with IDEAS."""
    if not metadata:
        return {}

    return {
        key: value
        for key, value in metadata.items()
        if key in USEFUL_OUTPUT_METADATA_KEYS
        and value not in (None, "", [], {})
    }


def _collect_available_previews(
    output_dir: str,
    preview_definitions: Iterable[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Return preview tuples that exist on disk."""
    available_previews: List[Tuple[str, str]] = []
    for preview_name, caption in preview_definitions:
        preview_path = os.path.join(output_dir, preview_name)
        if os.path.exists(preview_path):
            available_previews.append((preview_name, caption))
    return available_previews

def _process_output_file(
    *,
    output_data: OutputData,
    output_dir: str,
    output_metadata: dict,
    file: str,
    output_file_basename: str,
    preview_files: Optional[List[tuple]] = None,
) -> None:
    """Attach an output file and its previews to the OutputData object."""

    try:
        file_path = os.path.join(output_dir, file)
        filename = pathlib.Path(file_path).name
        basename = pathlib.Path(file_path).stem

        final_file_path = file_path
        target_dir = os.path.join(output_dir, basename)

        def _should_skip_prefix(preview_name: str) -> bool:
            lowered = preview_name.lower()
            return (
                lowered.startswith("group1_")
                or lowered.startswith("group2_")
            )

        if preview_files:
            os.makedirs(target_dir, exist_ok=True)
            final_file_path = os.path.join(
                target_dir,
                f"{output_file_basename}_{filename}",
            )
            os.rename(file_path, final_file_path)

        output_file = output_data.add_file(final_file_path)

        metadata = _extract_useful_metadata(output_metadata.get(basename, {}))
        for key, value in metadata.items():
            output_file.add_metadata(
                key=key,
                value=str(value),
                name=key.title(),
            )

        if preview_files:
            for preview_name, caption in preview_files:
                preview_path = os.path.join(output_dir, preview_name)
                if not os.path.exists(preview_path):
                    logger.warning(
                        "Preview '%s' not found for output '%s'; skipping attachment.",
                        preview_name,
                        filename,
                    )
                    continue

                if _should_skip_prefix(preview_name):
                    new_preview_filename = preview_name
                else:
                    new_preview_filename = (
                        f"{output_file_basename}_{preview_name}"
                    )

                new_preview_path = os.path.join(
                    target_dir,
                    new_preview_filename,
                )
                os.rename(preview_path, new_preview_path)
                output_file.add_preview(new_preview_path, caption)

    except Exception:  # noqa: BLE001
        logger.exception("failed to process file")

    return basename
# Analysis feature flags (not exposed via the main function)
@dataclass(frozen=True)
class StateEpochAnalysisFeatureFlags:
    """Configuration for optional analysis features."""

    include_correlations: bool = True
    include_population_activity: bool = True
    include_event_analysis: bool = True
    # Registration support (for CaImAn MSR outputs)
    use_registered_cellsets: bool = False
    registration_method: str = "auto_detect"  # "auto_detect", "caiman_msr"
    # Controls whether annotations_file can be omitted (epoch-only mode)
    allow_epoch_only_mode: bool = False


_FEATURE_FLAGS = StateEpochAnalysisFeatureFlags()

def get_state_epoch_analysis_feature_flags() -> StateEpochAnalysisFeatureFlags:
    """Return a copy of the current feature flag configuration."""
    return replace(_FEATURE_FLAGS)


def configure_state_epoch_analysis_feature_flags(
    **overrides: object,
) -> None:
    """Update the global feature flag configuration."""
    global _FEATURE_FLAGS
    if overrides:
        _FEATURE_FLAGS = replace(_FEATURE_FLAGS, **overrides)
    else:
        _FEATURE_FLAGS = StateEpochAnalysisFeatureFlags()


@contextmanager
def temporary_state_epoch_analysis_feature_flags(**overrides: object):
    """Temporarily override feature flags within a context."""
    global _FEATURE_FLAGS
    original_flags = _FEATURE_FLAGS
    if overrides:
        _FEATURE_FLAGS = replace(_FEATURE_FLAGS, **overrides)
    try:
        yield
    finally:
        _FEATURE_FLAGS = original_flags


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
    correlation_statistic: str = "max",
    method: str = SUPPORTED_STATE_COMPARISON_METHOD,
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
    include_event_correlation_preview: bool = False,
    # Statistical parameters
    alpha: float = 0.05,
    n_shuffle: int = 1000,
    # Data processing parameters
    tolerance: float = 1e-4,
    sort_by_time: bool = True,
    # Output parameters
    output_dir: str = "",
) -> None:
    """Perform combined state-epoch analysis with baseline comparison.

    This tool analyzes neural activity patterns across behavioral states
    and time epochs, calculating:
    1. Activity/event rates per state-epoch combination
    2. Correlations per state-epoch combination
    3. Modulation indices relative to baseline state-epoch

    Feature toggles for correlations, population activity, event analysis,
    and registered cellset support are configured via
    configure_state_epoch_analysis_feature_flags instead of function
    arguments.

    Args:
    ----
        cell_set_files: List of cellset files (.isxd, .h5)
        event_set_files: Optional list of eventset files (.isxd)
        annotations_file: List of annotation files (.parquet or .csv). Only the first file
            is used. When allow_epoch_only_mode feature flag is enabled, this parameter
            may be omitted to run in epoch-only mode.
        column_name: Column name for state annotations
        state_names: Comma-separated state names
        state_colors: Comma-separated color names for states
        correlation_statistic: Per-cell correlation statistic to summarize
            ("max", "min", or "mean") in distribution previews
        method: State comparison method (currently only "state vs baseline")
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
        include_event_correlation_preview: Whether to generate additional event-based
            correlation previews alongside the standard trace outputs when event
            correlations are available.
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
    normalized_method = method.strip().lower() if method else ""
    if normalized_method != SUPPORTED_STATE_COMPARISON_METHOD:
        raise IdeasError(
            "Unsupported state comparison method "
            f"'{method}'. This tool only supports '{SUPPORTED_STATE_COMPARISON_METHOD}' "
            "to maintain parity with the standard-python implementation."
        )
    logger.info(
        "State comparison method: %s", SUPPORTED_STATE_COMPARISON_METHOD
    )
    feature_flags = get_state_epoch_analysis_feature_flags()

    if include_event_correlation_preview:
        logger.info("Event correlation outputs enabled when event data is available.")

    annotations_file_list = annotations_file or []
    
    # Check for None entries first before other validation
    if any(path is None for path in annotations_file_list):
        raise IdeasError(
            "annotations_file cannot contain empty or null entries."
        )
    
    has_annotations = (
        bool(annotations_file_list)
        and annotations_file_list[0] is not None
        and str(annotations_file_list[0]).strip() != ""
    )

    if not has_annotations and not feature_flags.allow_epoch_only_mode:
        raise IdeasError(
            "annotations_file is required unless allow_epoch_only_mode is enabled."
        )

    # Validate inputs
    try:
        validate_input_files_exist(cell_set_files)
        if event_set_files:
            validate_input_files_exist(event_set_files)
        if has_annotations:
            validate_input_files_exist(
                [str(p) for p in annotations_file_list if p is not None]
            )
    except FileNotFoundError as e:
        raise IdeasError(f"Input validation failed: {e}") from e

    # Parse parameter strings
    states = [s.strip() for s in state_names.split(",")]
    parsed_epochs = [e.strip() for e in epoch_names.split(",")]
    state_color_list = [c.strip() for c in state_colors.split(",")]
    epoch_color_list = [c.strip() for c in epoch_colors.split(",")]

    epoch_only_mode = feature_flags.allow_epoch_only_mode and (
        not has_annotations
        or len(states) == 0
        or all(state == "" for state in states)
    )

    if epoch_only_mode:
        logger.info(
            "Allowing epoch-only mode with dummy state 'epoch_activity' because "
            "allow_epoch_only_mode feature flag is enabled."
        )
        states = ["epoch_activity"]
        state_color_list = ["gray"]
        baseline_state = "epoch_activity"
        column_name = "dummy_state"
    elif len(states) == 0 or all(state == "" for state in states):
        raise IdeasError(
            "state_names must include at least one valid state when annotations are provided."
        )

    data_manager = StateEpochDataManager(
        cell_set_files=cell_set_files,
        event_set_files=event_set_files,
        annotations_file=annotations_file_list if has_annotations else None,
        concatenate=concatenate,
        use_registered_cellsets=feature_flags.use_registered_cellsets,
        registration_method=feature_flags.registration_method,
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
        allow_epoch_only_mode=feature_flags.allow_epoch_only_mode,
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
                include_correlations=feature_flags.include_correlations,
                include_population_activity=feature_flags.include_population_activity,
                include_event_analysis=feature_flags.include_event_analysis,
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
        correlation_statistic=correlation_statistic,
        include_event_correlation_preview=include_event_correlation_preview,
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
        os.remove(output_metadata_path)

    output_file_basename = ""
    for cell_set_file in cell_set_files:
        cell_set_file_name = os.path.basename(cell_set_file)
        cell_set_file_name, _ = os.path.splitext(cell_set_file_name)
        cell_set_file_name = cell_set_file_name.replace(" ", "_")

        if output_file_basename:
            output_file_basename += "_"
        output_file_basename += cell_set_file_name

    with OutputData() as output_data:
        _process_output_file(
            output_data=output_data,
            output_dir=output_dir,
            output_metadata=output_metadata,
            file="activity_per_state_epoch_data.csv",
            output_file_basename=output_file_basename,
            preview_files=[
                (
                    STATE_EPOCH_TIME_PREVIEW,
                    "Time spent in each state-epoch combination, with per-state totals shown for context.",
                ),
                (
                    TRACE_POPULATION_AVERAGE_PREVIEW,
                    "Average trace activity across state-epoch combinations. Bar plot showing mean activity ± standard error for each state-epoch combination."
                ),

                (
                    TRACE_STATE_OVERLAY,
                    "Trace preview with state-colored traces showing neural activity patterns colored by behavioral state."
                ),
                (
                    EVENT_POPULATION_AVERAGE_PREVIEW,
                    "Average event rates across state-epoch combinations. Box plots showing event frequency distributions when event data is available."
                ),
                (
                    EVENT_STATE_OVERLAY,
                    "Event raster plot with state-colored events showing event patterns colored by behavioral state."
                ),
            ],
        )

        correlation_preview_files = [
            (
                CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
                "Distribution of the selected per-cell correlation statistic across neurons and state-epoch combinations.",
            ),
        ]
        if include_event_correlation_preview:
            correlation_preview_files.extend(
                _collect_available_previews(
                    output_dir,
                    [
                        (
                            EVENT_CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
                            "Distribution of the selected per-cell event correlation statistic across neurons and state-epoch combinations.",
                        ),
                    ],
                )
            )

        _process_output_file(
            output_data=output_data,
            output_dir=output_dir,
            output_metadata=output_metadata,
            file=CORRELATIONS_PER_STATE_EPOCH_DATA_CSV,
            output_file_basename=output_file_basename,
            preview_files=correlation_preview_files,
        )

        _process_output_file(
            output_data=output_data,
            output_dir=output_dir,
            output_metadata=output_metadata,
            file="modulation_vs_baseline_data.csv",
            output_file_basename=output_file_basename,
            preview_files=[
                (
                    TRACE_MODULATION_HISTOGRAM_PREVIEW,
                    "Distribution of modulation scores across neurons relative to baseline. Histograms show the modulation scores for all neurons compared to the baseline state-epoch combination."
                ),
                (
                    TRACE_MODULATION_FOOTPRINT_PREVIEW,
                    "Spatial distribution of modulated neurons relative to baseline. Cell footprints are colored by modulation direction and significance relative to the baseline state-epoch combination."
                ),
                (
                    EVENT_MODULATION_HISTOGRAM_PREVIEW,
                    "Distribution of event modulation scores across neurons relative to baseline. Histograms showing event modulation when event data is available."
                ),
                (
                    EVENT_MODULATION_PREVIEW,
                    "Spatial footprints of event-modulated neurons relative to baseline. Cell maps colored by event modulation significance when event data is available."
                ),

            ]
        )

        average_correlation_previews = [
            (
                AVERAGE_CORRELATIONS_PREVIEW,
                "Bar plots showing average positive and negative correlations for each state-epoch combination. Provides summary statistics of correlation patterns.",
            ),
        ]
        if include_event_correlation_preview:
            average_correlation_previews.extend(
                _collect_available_previews(
                    output_dir,
                    [
                        (
                            EVENT_AVERAGE_CORRELATIONS_PREVIEW,
                            "Bar plots showing average positive and negative event correlations for each state-epoch combination.",
                        ),
                    ],
                )
            )

        _process_output_file(
            output_data=output_data,
            output_dir=output_dir,
            output_metadata=output_metadata,
            file=AVERAGE_CORRELATIONS_CSV,
            output_file_basename=output_file_basename,
            preview_files=average_correlation_previews,
        )

        correlation_matrix_previews = [
            (
                CORRELATION_MATRICES_PREVIEW,
                "Pairwise Pearson correlation matrices between neurons for each state-epoch combination. Shows how neural correlations change across different conditions.",
            ),
        ]
        if include_event_correlation_preview:
            correlation_matrix_previews.extend(
                _collect_available_previews(
                    output_dir,
                    [
                        (
                            EVENT_CORRELATION_MATRICES_PREVIEW,
                            "Pairwise event correlation matrices between neurons for each state-epoch combination when event data is available.",
                        ),
                    ],
                )
            )

        _process_output_file(
            output_data=output_data,
            output_dir=output_dir,
            output_metadata=output_metadata,
            file=RAW_CORRELATIONS_H5_NAME,
            output_file_basename=output_file_basename,
            preview_files=correlation_matrix_previews,
        )

        spatial_correlation_previews = [
            (
                SPATIAL_CORRELATION_PREVIEW,
                "Relationship between spatial distance and neural correlation across different state-epoch combinations. Scatter plots show pairwise neural correlation versus physical distance between cell centroids.",
            ),
            (
                SPATIAL_CORRELATION_MAP_PREVIEW,
                "Spatial map of neural correlations across state-epoch combinations. Colored lines connect neuron pairs above the correlation threshold, with line color indicating correlation strength.",
            ),
        ]
        if include_event_correlation_preview:
            spatial_correlation_previews.extend(
                _collect_available_previews(
                    output_dir,
                    [
                        (
                            EVENT_SPATIAL_CORRELATION_PREVIEW,
                            "Relationship between spatial distance and event-based neural correlation across state-epoch combinations.",
                        ),
                        (
                            EVENT_SPATIAL_CORRELATION_MAP_PREVIEW,
                            "Spatial map of event-based neural correlations when event data is available.",
                        ),
                    ],
                )
            )

        _process_output_file(
            output_data=output_data,
            output_dir=output_dir,
            output_metadata=output_metadata,
            file=RAW_CORRELATIONS_ZIP_NAME,
            output_file_basename=output_file_basename,
            preview_files=spatial_correlation_previews,
        )
    logger.info("State-epoch baseline analysis completed successfully")


analyze = state_epoch_baseline_analysis
