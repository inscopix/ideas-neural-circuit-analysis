"""Combined State-Epoch Analysis Tool for Two-Group Comparison.

Analyzes and compares neural activity patterns between two groups across
behavioral states or time epochs using pre-computed results from state_epoch_baseline_analysis.

Output Files Structure
------------------------------------------------------------------
Files with previews are organized in subdirectories for better organization:

1. Per-Group Data Files (in subdirectories with previews):
   - <group_name>_combined_activity_data/<comparison>_<group_name>_combined_activity_data.csv
   - <group_name>_combined_trace_correlation_data/<comparison>_<group_name>_combined_trace_correlation_data.csv
   - <group_name>_combined_modulation_data/<comparison>_<group_name>_combined_modulation_data.csv
   - Where <comparison> is typically <group1>_vs_<group2>_<dimension> (e.g., "Control_vs_Treatment_epochs")
   - Preview SVG files are in .previews/ subdirectories within each folder

2. Statistical Comparison Files (Trace-specific, in subdirectories with previews):
   - trace_aov_comparisons/<comparison>_trace_aov_comparisons.csv (main effects and interactions)
   - trace_pairwise_comparisons/<comparison>_trace_pairwise_comparisons.csv (post-hoc tests)

3. Statistical Comparison Files (Event-specific, in subdirectories with previews):
   - event_aov_comparisons/<comparison>_event_aov_comparisons.csv (main effects and interactions)
   - event_pairwise_comparisons/<comparison>_event_pairwise_comparisons.csv (post-hoc tests)

4. Per-Group SVG Preview Files (in .previews/ subdirectories):
   - <group_name>_trace_activity_boxplot.svg
   - <group_name>_event_activity_boxplot.svg
   - <group_name>_{stat}_trace_correlation_boxplot.svg / <group_name>_{stat}_event_correlation_boxplot.svg
     where ``stat`` is the selected correlation statistic (max, min, or mean)
   - <group_name>_{stat}_trace_correlation_cdf.svg / <group_name>_{stat}_event_correlation_cdf.svg
   - <group_name>_positive_trace_population_boxplot.svg
   - <group_name>_negative_trace_population_boxplot.svg
   - <group_name>_positive_event_population_boxplot.svg
   - <group_name>_negative_event_population_boxplot.svg

5. Comparison SVG Files (in .previews/ subdirectories, conditional on comparison_dimension):
   - states_comparison_trace_activity.svg OR epochs_comparison_trace_activity.svg
   - states_comparison_trace_correlation.svg OR epochs_comparison_trace_correlation.svg
   - states_comparison_event_activity.svg OR epochs_comparison_event_activity.svg
   - states_comparison_event_correlation.svg OR epochs_comparison_event_correlation.svg

"""

import json
import logging
import os
import pathlib
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Any,
    Set,
    Union,
    Deque,
    Iterable,
)

import numpy as np
import pandas as pd
from ideas.exceptions import IdeasError  # type: ignore[import-not-found]

from ideas.outputs import OutputData  # type: ignore[import-not-found]
from analysis.output_registration import (
    collect_available_previews,
    register_output_file,
    PreviewPrefixRules,
)

from utils.state_epoch_comparison_utils import (
    calculate_state_epoch_comparison_stats,
    calculate_state_epoch_lmm_stats,
    _require_supported_multiple_correction_method,
    match_subjects,
    plot_state_epoch_comparison,
    validate_modulation_colors,
)
from utils.statistical_formatting import (
    _cleanup_final_csv_columns,
    _finalize_statistical_output,
)
from utils.population_data_validation import (
    validate_group_names,
    validate_file_group,
    validate_subject_id_format,
)
from utils.visualization_helpers import (
    create_boxplot_preview as _base_create_boxplot_preview,
    create_cdf_preview as _base_create_cdf_preview,
    plot_modulation_distribution,
)


logger = logging.getLogger(__name__)


# Metadata filtering for output registration
# Controls which metadata fields are included in the registered output files
USEFUL_OUTPUT_METADATA_KEYS = {
    # "num_states",
    # "num_epochs",
    # "num_combinations",
    "states",
    "epochs",
    "baseline_state",
    "baseline_epoch",
    "comparison_dimension",
    "data_pairing",
    "correlation_statistic",
    "multiple_correction",
    "effect_size",
    # "num_groups",
    # "group_names",
    # "group_colors",
    # "measure_source",
    # "selected_measures",
    # "selected_measure_sources",
    "group_comparison_type",
    "parametric",
    # Optional fields that might be present
    # "file_type",
    # "description",
    # "analysis_type",
    # "test_type",
    # "file_exists",
}


def _extract_useful_metadata(metadata: dict) -> dict:
    """Return only the metadata fields we want to register with IDEAS.
    
    Filters the comprehensive metadata dictionary to include only fields
    that are relevant for output registration and user information.
    
    Parameters
    ----------
    metadata : dict
        Full metadata dictionary
        
    Returns
    -------
    dict
        Filtered metadata containing only useful fields
    """
    if not metadata:
        return {}

    return {
        key: value
        for key, value in metadata.items()
        if key in USEFUL_OUTPUT_METADATA_KEYS
        and value not in (None, "", [], {})
    }


# Measure column mapping dictionary for comprehensive trace/event analysis
# Defines all possible measure columns organized by category and data source
#
# IMPORTANT DATA LEVEL NOTES:
# - activity: Cell-level (unique per cell) - LMM valid
# - correlation_percell: Cell-level (unique per cell) - LMM valid
# - correlation_population: STATE-EPOCH level (same for all cells in condition) - LMM INVALID
# - modulation: Cell-level (unique per cell) - LMM valid
#
# Population correlation columns represent aggregates computed across all cells
# in a state-epoch combination, not individual cell properties. They should NOT
# be analyzed with LMM due to pseudoreplication (identical values treated as
# independent observations). Use ANOVA results instead.
MEASURE_COLUMNS = {
    "activity": {
        "trace": ["mean_trace_activity"],
        "event": ["mean_event_rate"],
    },
    "correlation_percell": {
        "trace": {
            "max": ["max_trace_correlation"],
            "min": ["min_trace_correlation"],
            "mean": ["mean_trace_correlation"],
        },
        "event": {
            "max": ["max_event_correlation"],
            "min": ["min_event_correlation"],
            "mean": ["mean_event_correlation"],
        },
    },
    "correlation_population": {
        "trace": {
            "positive": ["positive_trace_correlation"],
            "negative": ["negative_trace_correlation"],
        },
        "event": {
            "positive": ["positive_event_correlation"],
            "negative": ["negative_event_correlation"],
        },
    },
    "modulation": {
        "trace": ["trace_modulation"],
        "event": ["event_modulation"],
    },
}

# Output file name constants
GROUP_COMBINED_ACTIVITY_SUFFIX = "combined_activity_data.csv"
GROUP_COMBINED_TRACE_CORRELATION_SUFFIX = "combined_trace_correlation_data.csv"
GROUP_COMBINED_MODULATION_SUFFIX = "combined_modulation_data.csv"
GROUP_OUTPUT_SUFFIXES = {
    "activity": GROUP_COMBINED_ACTIVITY_SUFFIX,
    "correlation": GROUP_COMBINED_TRACE_CORRELATION_SUFFIX,
    "modulation": GROUP_COMBINED_MODULATION_SUFFIX,
}

# Statistical comparison files (trace-specific)
TRACE_ANOVA_COMPARISONS_CSV = "trace_aov_comparisons.csv"
TRACE_PAIRWISE_COMPARISONS_CSV = "trace_pairwise_comparisons.csv"

# Statistical comparison files (event-specific)
EVENT_ANOVA_COMPARISONS_CSV = "event_aov_comparisons.csv"
EVENT_PAIRWISE_COMPARISONS_CSV = "event_pairwise_comparisons.csv"

# LMM statistical comparison files
TRACE_LMM_COMPARISONS_CSV = "trace_lmm_comparisons.csv"
TRACE_LMM_PAIRWISE_COMPARISONS_CSV = "trace_lmm_pairwise_comparisons.csv"
EVENT_LMM_COMPARISONS_CSV = "event_lmm_comparisons.csv"
EVENT_LMM_PAIRWISE_COMPARISONS_CSV = "event_lmm_pairwise_comparisons.csv"

# Trace-specific comparison preview files
STATES_COMPARISON_TRACE_ACTIVITY_SVG = "states_comparison_trace_activity.svg"
STATES_COMPARISON_TRACE_CORRELATION_SVG = (
    "states_comparison_trace_correlation.svg"
)
STATES_COMPARISON_TRACE_POSITIVE_CORRELATION_SVG = (
    "states_comparison_trace_positive_correlation.svg"
)
STATES_COMPARISON_TRACE_NEGATIVE_CORRELATION_SVG = (
    "states_comparison_trace_negative_correlation.svg"
)
EPOCHS_COMPARISON_TRACE_ACTIVITY_SVG = "epochs_comparison_trace_activity.svg"
EPOCHS_COMPARISON_TRACE_CORRELATION_SVG = (
    "epochs_comparison_trace_correlation.svg"
)
EPOCHS_COMPARISON_TRACE_POSITIVE_CORRELATION_SVG = (
    "epochs_comparison_trace_positive_correlation.svg"
)
EPOCHS_COMPARISON_TRACE_NEGATIVE_CORRELATION_SVG = (
    "epochs_comparison_trace_negative_correlation.svg"
)

# Event-specific comparison preview files
STATES_COMPARISON_EVENT_ACTIVITY_SVG = "states_comparison_event_activity.svg"
STATES_COMPARISON_EVENT_CORRELATION_SVG = (
    "states_comparison_event_correlation.svg"
)
STATES_COMPARISON_EVENT_POSITIVE_CORRELATION_SVG = (
    "states_comparison_event_positive_correlation.svg"
)
STATES_COMPARISON_EVENT_NEGATIVE_CORRELATION_SVG = (
    "states_comparison_event_negative_correlation.svg"
)
EPOCHS_COMPARISON_EVENT_ACTIVITY_SVG = "epochs_comparison_event_activity.svg"
EPOCHS_COMPARISON_EVENT_CORRELATION_SVG = (
    "epochs_comparison_event_correlation.svg"
)
EPOCHS_COMPARISON_EVENT_POSITIVE_CORRELATION_SVG = (
    "epochs_comparison_event_positive_correlation.svg"
)
EPOCHS_COMPARISON_EVENT_NEGATIVE_CORRELATION_SVG = (
    "epochs_comparison_event_negative_correlation.svg"
)

# Modulation-specific comparison preview files
STATES_COMPARISON_TRACE_MODULATION_SVG = (
    "states_comparison_trace_modulation.svg"
)
EPOCHS_COMPARISON_TRACE_MODULATION_SVG = (
    "epochs_comparison_trace_modulation.svg"
)
STATES_COMPARISON_EVENT_MODULATION_SVG = (
    "states_comparison_event_modulation.svg"
)
EPOCHS_COMPARISON_EVENT_MODULATION_SVG = (
    "epochs_comparison_event_modulation.svg"
)


def _normalize_parametric_value(value: Union[str, bool, None]) -> str:
    """Return a normalized string representation for the parametric flag."""
    if isinstance(value, str):
        lowered_value = value.strip().lower()
        if lowered_value in {"true", "yes", "1"}:
            return "True"
        if lowered_value in {"false", "no", "0"}:
            return "False"
        if lowered_value == "auto":
            return "auto"
        return value.strip()
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "auto"
    return str(value).strip()


def combine_compare_state_epoch_data(
    *,
    # Group 1 inputs (from state_epoch_baseline_analysis outputs)
    group1_activity_csv_files: List[pathlib.Path],
    group1_correlation_csv_files: Optional[List[pathlib.Path]] = None,
    group1_modulation_csv_files: Optional[List[pathlib.Path]] = None,
    group1_name: Optional[str] = None,
    group1_color: Optional[str] = None,
    # Group 2 inputs (from state_epoch_baseline_analysis outputs)
    group2_activity_csv_files: Optional[List[pathlib.Path]] = None,
    group2_correlation_csv_files: Optional[List[pathlib.Path]] = None,
    group2_modulation_csv_files: Optional[List[pathlib.Path]] = None,
    group2_name: Optional[str] = None,
    group2_color: Optional[str] = None,
    # Comparison mode
    comparison_dimension: str = "states",  # "states" or "epochs"
    # Unified trace/event selection
    measure_source: str = "trace",
    # Dimension colors (optional user-defined colors)
    state_colors: Optional[List[str]] = None,
    epoch_colors: Optional[List[str]] = None,
    modulation_colors: Optional[str] = "green,blue,black",
    # Comparison parameters
    data_pairing: str = "unpaired",
    subject_matching: str = "order",
    correlation_statistic: str = "max",
    significance_threshold: Optional[float] = None,
    multiple_correction: str = "bonf",
    multiple_correction_scope: str = "global",
    effect_size: str = "cohen",
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
    # Output parameters
    output_dir: str = "",
    enable_lmm_analysis: bool = True,
    save_lmm_outputs: bool = False,
) -> None:
    """Perform combined state-epoch analysis with two-group comparison.

    This tool compares neural activity patterns between two groups across
    behavioral states or time epochs using CSV/H5 outputs from state_epoch_baseline_analysis.
    Comparisons can be made across states OR across epochs (not both simultaneously).

    IMPORTANT: This tool currently focuses on state OR epoch comparisons separately,
    following the same logic as the combine_compare_population_data and
    combine_compare_correlation_data tools. This design choice maintains consistency
    during toolbox merging and ensures statistical approaches align across all
    combine/compare tools. Cross-epoch-state comparisons will be addressed in a
    future update.

    BASELINE HANDLING: The baseline state and epoch are automatically extracted from
    the input modulation CSV files (computed during state_epoch_baseline_analysis).
    These cannot be overridden by the user to ensure consistency with the pre-computed
    modulation classifications in the input data.

    CORRELATION INPUT FORMATS: Correlation data can be provided in two formats:
    - CSV files (correlations_per_state_epoch_data.csv): Per-cell correlation statistics
    - H5 files (pairwise_correlation_heatmaps.h5): Raw correlation matrices
    Both formats are generated by state_epoch_baseline_analysis and produce identical
    analysis results. H5 files are automatically converted to the same format as CSV files.

    Parameters
    ----------
    group1_activity_csv_files : List[pathlib.Path]
        List of activity_per_state_epoch_data.csv files for group 1
    group1_correlation_csv_files : Optional[List[pathlib.Path]]
        Optional list of correlations_per_state_epoch_data.csv OR pairwise_correlation_heatmaps.h5
        files for group 1. CSV files must include positive_trace_correlation and
        negative_trace_correlation columns generated by state_epoch_baseline_analysis;
        optional event correlation columns will be detected automatically. H5 files should
        contain trace and/or event groups with correlation matrices for each state-epoch
        combination. If not provided, correlation analysis will be skipped.
        NOTE: If group 2 activity files are provided, and group 1 has correlation files,
        then group 2 must also have matching correlation files.
    group1_modulation_csv_files : Optional[List[pathlib.Path]]
        Optional list of modulation_vs_baseline_data.csv files for group 1. If not provided,
        modulation analysis will be skipped.
        NOTE: If group 2 activity files are provided, and group 1 has modulation files,
        then group 2 must also have matching modulation files.
    group1_name : Optional[str]
        Name of group 1 for labeling
    group1_color : Optional[str]
        Color for group 1 visualizations
    group2_activity_csv_files : Optional[List[pathlib.Path]]
        List of activity_per_state_epoch_data.csv files for group 2
    group2_correlation_csv_files : Optional[List[pathlib.Path]]
        Optional list of correlations_per_state_epoch_data.csv OR pairwise_correlation_heatmaps.h5
        files for group 2. CSV files must include positive_trace_correlation and
        negative_trace_correlation columns generated by state_epoch_baseline_analysis;
        optional event correlation columns will be detected automatically. H5 files should
        contain trace and/or event groups with correlation matrices for each state-epoch
        combination. If group 2 activity files are provided and group 1 has correlation files,
        this must also be provided with matching file counts.
    group2_modulation_csv_files : Optional[List[pathlib.Path]]
        Optional list of modulation_vs_baseline_data.csv files for group 2. If group 2 activity
        files are provided and group 1 has modulation files, this must also be provided with
        matching file counts.
    group2_name : Optional[str]
        Name of group 2 for labeling
    group2_color : Optional[str]
        Color for group 2 visualizations
    comparison_dimension : str
        Dimension to compare: "states" or "epochs"
    measure_source : str
        Data source to analyze across activity, correlation, and modulation
        (default: "trace", case-insensitive):
        - "trace": Use trace-based measures when available (falls back to event if
          trace columns are missing)
        - "event": Use event-based measures exclusively
        - "both": Analyze both trace and event measures separately
    state_colors : Optional[List[str]]
        Optional list of hex color codes for states (e.g., ["#FF0000", "#00FF00", "#0000FF"]).
        Colors will be assigned to states in the order they appear in the data.
        If not provided, colors will be extracted from input CSV files or auto-generated.
    epoch_colors : Optional[List[str]]
        Optional list of hex color codes for epochs (e.g., ["#0000FF", "#FFFF00", "#FF00FF"]).
        Colors will be assigned to epochs in the order they appear in the data.
        If not provided, colors will be extracted from input CSV files or auto-generated.
    modulation_colors : Optional[str]
        Comma-separated list of matplotlib compatible colors representing the up-, down- and
        non-modulated neurons, respectively (default: "green,blue,black")
    data_pairing : str
        Type of data pairing ("paired" or "unpaired")
    subject_matching : str
        Method for matching subjects between groups
    correlation_statistic : str
        Which statistic to use from correlation data: "max", "min", or "mean"
        (default: "max"). This applies to both trace and event datasets.
    significance_threshold : Optional[float]
        Significance threshold for statistical tests. If None (default), uses
        pre-computed modulation classifications from state_epoch_baseline_analysis.
        If a value is provided, neurons will be reclassified using this threshold
        with proper alpha/2 correction for two one-tailed comparisons.
    multiple_correction : str
        Method for multiple comparison correction ("bonf", "fdr_bh", "none")
    multiple_correction_scope : str
        Scope of multiple comparison correction (default: "global"):
        - "global": Apply correction across ALL pairwise tests from all strata
          (e.g., all state comparisons across all epochs). This is the most
          statistically rigorous approach and maintains proper family-wise error
          rate control. RECOMMENDED for confirmatory analyses.
        - "within_stratum": Apply correction only within each stratum
          (e.g., separately for each epoch when comparing states). This is less
          conservative and may be appropriate for exploratory analyses.

        IMPORTANT: With stratified comparisons (states within epochs or vice versa),
        global correction prevents inflation of Type I error rate. For example,
        comparing 3 states across 4 epochs yields 12 total tests. Global correction
        accounts for all 12 tests, while within-stratum correction only accounts
        for 3 tests per epoch, leading to inflated false positive rates.
    effect_size : str
        Method for calculating effect size
    group_comparison_type : str
        Type of statistical test to perform (default: "two_tailed"):
        - "two_tailed": Two-tailed test (tests for differences in either direction)
        - "one_tailed_less": One-tailed test (tests if first group is less than second)
        - "one_tailed_greater": One-tailed test (tests if first group is greater than second)
    parametric : str
        Indicates whether to perform a parametric test (default: "auto"):
        - "auto": Automatically select parametric test if data follows normal
          distribution and has at least 8 observations per group. Otherwise,
          use non-parametric test.
        - "True": Always use parametric tests (t-test, ANOVA). If parametric tests
          fail (e.g., due to small sample size, zero variance, or non-normality),
          an informative IdeasError will be raised suggesting alternative parametric
          values rather than automatically falling back to non-parametric tests.
        - "False": Always use non-parametric tests (Mann-Whitney U, Wilcoxon,
          Friedman, Kruskal-Wallis)
    output_dir : str
        Directory for output files
    enable_lmm_analysis : bool
        Whether to run the linear mixed model (LMM) workflow alongside the
        classical ANOVA-style analysis (default: False)
    save_lmm_outputs : bool
        When True, LMM summary and pairwise tables are written to CSV files in
        addition to being returned in-memory. Defaults to False so LMM results
        are available for evaluation without generating new files.

    Raises
    ------
    IdeasError
        If inputs are invalid, analysis fails, or when parametric="True" is
        explicitly requested but parametric tests fail due to data characteristics
        (e.g., insufficient sample size, zero variance, or normality violations).
        The error message will include data diagnostics and recommendations for
        using parametric="auto" or parametric="False"

    """
    logger.info("Starting combine and compare state-epoch analysis...")

    # Log the comparison mode for clarity
    if comparison_dimension == "states":
        logger.info(
            "Analysis mode: Comparing across STATES "
            "Epochs will be analyzed separately within each state."
        )
    elif comparison_dimension == "epochs":
        logger.info(
            "Analysis mode: Comparing across EPOCHS "
            "States will be analyzed separately within each epoch."
        )

    if data_pairing == "paired" and not group2_activity_csv_files:
        logger.warning(
            "Paired analysis requested but only one group provided. "
            "Defaulting data_pairing to 'unpaired'."
        )
        data_pairing = "unpaired"

    pairing_description = (
        "matched subjects across groups"
        if data_pairing == "paired"
        else "independent samples"
    )
    logger.info(
        f"Data pairing mode: {data_pairing.upper()} ({pairing_description})"
    )

    measure_source = measure_source.lower().strip()
    correlation_statistic = str(correlation_statistic).strip().lower()
    subject_matching = _normalize_subject_matching(subject_matching)
    multiple_correction_scope = _normalize_multiple_correction_scope(
        multiple_correction_scope
    )
    state_colors = _normalize_color_list(state_colors)
    epoch_colors = _normalize_color_list(epoch_colors)

    # Apply default labels before validation so validation receives the resolved values
    # Convert string paths to pathlib.Path objects
    group1_activity_csv_files = [
        pathlib.Path(f) for f in group1_activity_csv_files
    ]
    if group1_correlation_csv_files:
        group1_correlation_csv_files = [
            pathlib.Path(f) for f in group1_correlation_csv_files
        ]
    if group1_modulation_csv_files:
        group1_modulation_csv_files = [
            pathlib.Path(f) for f in group1_modulation_csv_files
        ]

    if group2_activity_csv_files:
        group2_activity_csv_files = [
            pathlib.Path(f) for f in group2_activity_csv_files
        ]
    if group2_correlation_csv_files:
        group2_correlation_csv_files = [
            pathlib.Path(f) for f in group2_correlation_csv_files
        ]
    if group2_modulation_csv_files:
        group2_modulation_csv_files = [
            pathlib.Path(f) for f in group2_modulation_csv_files
        ]

    if not group1_name:
        group1_name = "Group 1"
    if group2_activity_csv_files and not group2_name:
        group2_name = "Group 2"

    if not group1_color:
        group1_color = "blue"
    if group2_activity_csv_files and not group2_color:
        group2_color = "red"

    # Validate modulation colors
    modulation_colors = validate_modulation_colors(modulation_colors)
    logger.info(f"Using modulation colors: {modulation_colors}")

    # Validate inputs
    _validate_inputs(
        group1_activity_csv_files=group1_activity_csv_files,
        group1_correlation_csv_files=group1_correlation_csv_files,
        group1_modulation_csv_files=group1_modulation_csv_files,
        group1_name=group1_name,
        group2_activity_csv_files=group2_activity_csv_files,
        group2_correlation_csv_files=group2_correlation_csv_files,
        group2_modulation_csv_files=group2_modulation_csv_files,
        group2_name=group2_name,
        comparison_dimension=comparison_dimension,
        correlation_statistic=correlation_statistic,
    )

    group_names = [group1_name]
    if group2_name:
        group_names.append(group2_name)

    group_colors = [group1_color]
    if group2_color:
        group_colors.append(group2_color)

    logger.info(
        f"Comparing {len(group_names)} groups across {comparison_dimension}"
    )

    # Apply subject matching when running paired analysis with two groups
    if data_pairing == "paired" and group2_activity_csv_files:
        logger.info(
            "Paired analysis detected - applying subject matching using "
            f"method: '{subject_matching}'"
        )

        # Check that if group 1 has correlation/modulation, group 2 must have them too
        if group1_correlation_csv_files and not group2_correlation_csv_files:
            raise IdeasError(
                "Paired analysis with group 1 correlation files requires "
                "correlation CSV files for group 2"
            )
        if group1_modulation_csv_files and not group2_modulation_csv_files:
            raise IdeasError(
                "Paired analysis with group 1 modulation files requires "
                "modulation CSV files for group 2"
            )

        (
            group1_activity_csv_files,
            group2_activity_csv_files,
            matched_activity_pairs,
        ) = _match_group_file_lists(
            group1_files=group1_activity_csv_files,
            group2_files=group2_activity_csv_files,
            subject_matching=subject_matching,
            file_type="activity",
        )

        # Match correlation files only if both groups have them
        if group1_correlation_csv_files and group2_correlation_csv_files:
            (
                group1_correlation_csv_files,
                group2_correlation_csv_files,
                _,
            ) = _match_group_file_lists(
                group1_files=group1_correlation_csv_files,
                group2_files=group2_correlation_csv_files,
                subject_matching=subject_matching,
                file_type="correlation",
                reference_pairs=matched_activity_pairs,
            )

        # Match modulation files only if both groups have them
        if group1_modulation_csv_files and group2_modulation_csv_files:
            (
                group1_modulation_csv_files,
                group2_modulation_csv_files,
                _,
            ) = _match_group_file_lists(
                group1_files=group1_modulation_csv_files,
                group2_files=group2_modulation_csv_files,
                subject_matching=subject_matching,
                file_type="modulation",
                reference_pairs=matched_activity_pairs,
            )

    # Load and process group 1 data
    logger.info("Loading Group 1 data...")
    group1_data = _load_group_data(
        activity_csv_files=group1_activity_csv_files,
        correlation_csv_files=group1_correlation_csv_files,
        modulation_csv_files=group1_modulation_csv_files,
        group_name=group1_name,
        group_id=1,
        data_pairing=data_pairing,
    )

    # Load and process group 2 data if provided
    group2_data = None
    if group2_activity_csv_files:
        logger.info("Loading Group 2 data...")
        group2_data = _load_group_data(
            activity_csv_files=group2_activity_csv_files,
            correlation_csv_files=group2_correlation_csv_files,
            modulation_csv_files=group2_modulation_csv_files,
            group_name=group2_name,
            group_id=2,
            data_pairing=data_pairing,
        )

    # Extract states, epochs, and baseline from data
    # Baseline is ALWAYS extracted from input files (not user-configurable)
    states, epochs, baseline_state, baseline_epoch = (
        _extract_metadata_from_data(
            group1_data,
            group2_data,
        )
    )

    logger.info(f"Group 1 - States: {states}")
    logger.info(f"Group 1 - Epochs: {epochs}")
    logger.info(f"Group 1 - Baseline: {baseline_state}-{baseline_epoch}")

    # Reclassify neurons if significance_threshold is explicitly provided
    # (otherwise use pre-computed classifications from state_epoch_baseline_analysis)
    if significance_threshold is not None:
        logger.info(
            f"Reclassifying neurons with significance threshold: {significance_threshold}"
        )
        group1_data = _reclassify_state_epoch_neurons(
            group1_data,
            states,
            epochs,
            baseline_state,
            baseline_epoch,
            significance_threshold,
        )
        if group2_data is not None:
            group2_data = _reclassify_state_epoch_neurons(
                group2_data,
                states,
                epochs,
                baseline_state,
                baseline_epoch,
                significance_threshold,
            )
    else:
        logger.info(
            "Using pre-computed modulation classifications from state_epoch_baseline_analysis"
        )

    # Perform statistical comparison
    logger.info("Performing statistical comparison between groups...")
    comparison_results = _perform_statistical_comparison_csv(
        group1_data=group1_data,
        group2_data=group2_data,
        group_names=group_names,
        comparison_dimension=comparison_dimension,
        states=states,
        epochs=epochs,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        data_pairing=data_pairing,
        measure_source=measure_source,
        correlation_statistic=correlation_statistic,
        significance_threshold=significance_threshold,
        multiple_correction=multiple_correction,
        multiple_correction_scope=multiple_correction_scope,
        effect_size=effect_size,
        group_comparison_type=group_comparison_type,
        parametric=parametric,
        output_dir=output_dir,
        enable_lmm_analysis=enable_lmm_analysis,
        save_lmm_outputs=save_lmm_outputs,
    )
    multiple_correction = comparison_results.get(
        "multiple_correction",
        multiple_correction,
    )

    # Generate combined outputs
    logger.info("Generating combined outputs...")
    _generate_combined_outputs_csv(
        group1_data=group1_data,
        group2_data=group2_data,
        comparison_results=comparison_results,
        group_names=group_names,
        group_colors=group_colors,
        states=states,
        epochs=epochs,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        comparison_dimension=comparison_dimension,
        data_pairing=data_pairing,
        output_dir=output_dir,
        correlation_statistic=correlation_statistic,
        state_colors=state_colors,
        epoch_colors=epoch_colors,
        modulation_colors=modulation_colors,
    )

    # Save comprehensive output metadata
    logger.info("Saving output metadata...")
    resolved_selected_measures = comparison_results.get(
        "selected_measures", {}
    )
    resolved_selected_sources = comparison_results.get("selected_sources", {})
    _save_output_metadata(
        group1_data=group1_data,
        group2_data=group2_data,
        group_names=group_names,
        group_colors=group_colors,
        states=states,
        epochs=epochs,
        baseline_state=baseline_state,
        baseline_epoch=baseline_epoch,
        comparison_dimension=comparison_dimension,
        data_pairing=data_pairing,
        correlation_statistic=correlation_statistic,
        significance_threshold=significance_threshold,
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        group_comparison_type=group_comparison_type,
        parametric=parametric,
        output_dir=output_dir,
        measure_source=measure_source,
        selected_measures=resolved_selected_measures,
        selected_sources=resolved_selected_sources,
    )

    logger.info("Registering output files...")
    _register_combined_tool_outputs(
        output_dir=output_dir,
        group_names=group_names,
        comparison_dimension=comparison_dimension,
        correlation_statistic=correlation_statistic,
    )


def _validate_inputs(
    group1_activity_csv_files: List[pathlib.Path],
    group1_correlation_csv_files: Optional[List[pathlib.Path]],
    group1_modulation_csv_files: Optional[List[pathlib.Path]],
    group1_name: str,
    group2_activity_csv_files: Optional[List[pathlib.Path]],
    group2_correlation_csv_files: Optional[List[pathlib.Path]],
    group2_modulation_csv_files: Optional[List[pathlib.Path]],
    group2_name: Optional[str],
    comparison_dimension: str,
    correlation_statistic: str,
) -> None:
    """Validate input parameters.

    Note: Validates logical constraints (names, dimensions, etc.) before file existence
    to provide clear error messages for parameter issues. This validation order ensures
    users receive immediate feedback on configuration errors before expensive file
    operations are attempted.

    Validation Order:
        1. Group naming and high-level requirements
        2. Group-specific logical constraints (optional group requirements)
        3. Cross-modal file count alignment
        4. File existence and format checks (most expensive, done last)
    """
    # NOTE: `comparison_dimension` and `correlation_statistic` choices are constrained
    # via toolbox_info.json ToolChoiceParam definitions, so only non-UI-enforced
    # validations are performed here.

    def _require_matching_length(
        reference_files: Sequence[pathlib.Path],
        comparison_files: Sequence[pathlib.Path],
        *,
        group_label: str,
        reference_label: str,
        comparison_label: str,
    ) -> None:
        if len(reference_files) != len(comparison_files):
            raise IdeasError(
                f"{group_label} {reference_label} and {comparison_label} file counts must match. "
                f"{reference_label.capitalize()}: {len(reference_files)}, "
                f"{comparison_label.capitalize()}: {len(comparison_files)}"
            )

    group2_files_provided = bool(group2_activity_csv_files)

    # STEP 1: Validate group naming rules before file checks
    try:
        validate_group_names(
            group1_name=group1_name,
            group2_name=group2_name if group2_files_provided else None,
            require_group1=True,
            require_group2_if_provided=True,
        )
    except ValueError as e:
        raise IdeasError(str(e)) from e

    # STEP 2: Validate optional group 2 metadata requirements
    if group2_files_provided and not group2_name:
        raise IdeasError(
            "Group 2 name must be provided if group 2 files are specified"
        )

    # STEP 3: Validate group 1 modality file counts before checking existence
    # Correlation and modulation files are optional, but if provided must match activity count
    if group1_correlation_csv_files:
        _require_matching_length(
            group1_activity_csv_files,
            group1_correlation_csv_files,
            group_label="Group 1",
            reference_label="activity",
            comparison_label="correlation",
        )
    if group1_modulation_csv_files:
        _require_matching_length(
            group1_activity_csv_files,
            group1_modulation_csv_files,
            group_label="Group 1",
            reference_label="activity",
            comparison_label="modulation",
        )

    # STEP 4: Validate group 2 modality file counts and requirements
    # If group 2 is provided and group 1 has correlation/modulation, group 2 must have matching ones
    if group2_files_provided:
        if group1_correlation_csv_files and not group2_correlation_csv_files:
            raise IdeasError(
                "Group 2 correlation CSV files must be provided when "
                "group 2 activity files are specified and group 1 has correlation files"
            )

        if group1_modulation_csv_files and not group2_modulation_csv_files:
            raise IdeasError(
                "Group 2 modulation CSV files must be provided when "
                "group 2 activity files are specified and group 1 has modulation files"
            )

        if group2_correlation_csv_files:
            _require_matching_length(
                group2_activity_csv_files,
                group2_correlation_csv_files,
                group_label="Group 2",
                reference_label="activity",
                comparison_label="correlation",
            )
        if group2_modulation_csv_files:
            _require_matching_length(
                group2_activity_csv_files,
                group2_modulation_csv_files,
                group_label="Group 2",
                reference_label="activity",
                comparison_label="modulation",
            )

    # STEP 5: Validate file existence and formats (last, after all logical checks)
    try:
        validate_file_group(
            files=group1_activity_csv_files,
            group_name="Group 1",
            file_type="activity CSV",
            min_files=2,
        )
        if group1_correlation_csv_files:
            validate_file_group(
                files=group1_correlation_csv_files,
                group_name="Group 1",
                file_type="correlation CSV",
                min_files=2,
            )
        if group1_modulation_csv_files:
            validate_file_group(
                files=group1_modulation_csv_files,
                group_name="Group 1",
                file_type="modulation CSV",
                min_files=2,
            )
    except ValueError as e:
        raise IdeasError(str(e)) from e

    # Validate group 2 file existence if provided
    if group2_files_provided:
        try:
            validate_file_group(
                files=group2_activity_csv_files,
                group_name="Group 2",
                file_type="activity CSV",
                min_files=2,
            )
            if group2_correlation_csv_files:
                validate_file_group(
                    files=group2_correlation_csv_files,
                    group_name="Group 2",
                    file_type="correlation CSV",
                    min_files=2,
                )
            if group2_modulation_csv_files:
                validate_file_group(
                    files=group2_modulation_csv_files,
                    group_name="Group 2",
                    file_type="modulation CSV",
                    min_files=2,
                )
        except ValueError as e:
            raise IdeasError(str(e)) from e

    # Check if files actually exist
    file_lists_to_check = [
        (group1_activity_csv_files, "Group 1 activity CSV"),
    ]
    if group1_correlation_csv_files:
        file_lists_to_check.append(
            (group1_correlation_csv_files, "Group 1 correlation CSV/H5")
        )
    if group1_modulation_csv_files:
        file_lists_to_check.append(
            (group1_modulation_csv_files, "Group 1 modulation CSV")
        )

    for file_list, file_type in file_lists_to_check:
        for file_path in file_list:
            if not file_path.exists():
                raise IdeasError(f"{file_type} file not found: {file_path}")

            # Validate correlation file formats (CSV or H5 only)
            if "correlation" in file_type.lower():
                if file_path.suffix.lower() not in [".csv", ".h5"]:
                    raise IdeasError(
                        f"Invalid correlation file format: {file_path}. "
                        "Must be .csv or .h5 file."
                    )

    if group2_activity_csv_files:
        file_lists_to_check_g2 = [
            (group2_activity_csv_files, "Group 2 activity CSV"),
        ]
        if group2_correlation_csv_files:
            file_lists_to_check_g2.append(
                (group2_correlation_csv_files, "Group 2 correlation CSV/H5")
            )
        if group2_modulation_csv_files:
            file_lists_to_check_g2.append(
                (group2_modulation_csv_files, "Group 2 modulation CSV")
            )

        for file_list, file_type in file_lists_to_check_g2:
            for file_path in file_list:
                if not file_path.exists():
                    raise IdeasError(
                        f"{file_type} file not found: {file_path}"
                    )

                # Validate correlation file formats (CSV or H5 only)
                if "correlation" in file_type.lower():
                    if file_path.suffix.lower() not in [".csv", ".h5"]:
                        raise IdeasError(
                            f"Invalid correlation file format: {file_path}. "
                            "Must be .csv or .h5 file."
                        )


def _derive_subject_key(file_path: pathlib.Path) -> Optional[str]:
    """Derive a subject key by extracting all digits from the file name.

    The state-epoch toolbox aligns its matching behavior with the standard-python
    toolbox by using digit-based identifiers. All digits found in the file name
    (excluding the file extension) are concatenated together. Leading zeros are
    preserved so that identifiers like ``subject001`` remain distinct from
    ``subject1`` when necessary.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the subject-specific file.

    Returns
    -------
    Optional[str]
        Concatenated string of digits if any are found; ``None`` otherwise.

    """
    stem_without_extension = file_path.with_suffix("").name
    digits = re.findall(r"\d+", stem_without_extension)
    if digits:
        return "".join(digits)

    return None


def _match_subject_pairs(
    *,
    group1_files: List[pathlib.Path],
    group2_files: List[pathlib.Path],
    subject_matching: str,
    file_type: str,
) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Return matched subject file pairs between two groups."""
    if not group2_files:
        raise IdeasError(
            f"Paired analysis requires group 2 {file_type} files for subject matching"
        )

    try:
        matched_pairs_raw = match_subjects(
            [str(path) for path in group1_files],
            [str(path) for path in group2_files],
            match_method=subject_matching,
        )
    except ValueError as exc:
        raise IdeasError(
            f"Failed to match {file_type} files between groups using "
            f"subject_matching='{subject_matching}': {exc}"
        ) from exc

    matched_pairs = [
        (pathlib.Path(pair[0]), pathlib.Path(pair[1]))
        for pair in matched_pairs_raw
    ]

    if len(matched_pairs) < 2:
        raise IdeasError(
            f"Subject matching produced only {len(matched_pairs)} matched {file_type} pair(s). "
            "At least 2 matched pairs are required for paired analysis."
        )

    logger.info(
        f"Matched {len(matched_pairs)} {file_type} file pairs using '{subject_matching}'"
    )

    return matched_pairs


def _match_group_file_lists(
    *,
    group1_files: List[pathlib.Path],
    group2_files: List[pathlib.Path],
    subject_matching: str,
    file_type: str,
    reference_pairs: Optional[List[Tuple[pathlib.Path, pathlib.Path]]] = None,
) -> Tuple[
    List[pathlib.Path],
    List[pathlib.Path],
    List[Tuple[pathlib.Path, pathlib.Path]],
]:
    """Validate subject matching for provided file lists and align ordering.

    Parameters
    ----------
    group1_files : List[pathlib.Path]
        Group 1 file paths for a specific data type.
    group2_files : List[pathlib.Path]
        Group 2 file paths for the same data type.
    subject_matching : str
        Subject matching strategy to use.
    file_type : str
        Human-readable file type label for logging and error messages.
    reference_pairs : Optional[List[Tuple[pathlib.Path, pathlib.Path]]]
        Optional list of matched file pairs derived from activity files. When provided,
        the routine validates counts and subject identifiers against the reference pairs.

    Returns
    -------
    Tuple[List[pathlib.Path], List[pathlib.Path], List[Tuple[pathlib.Path, pathlib.Path]]]
        File lists reordered to match the subject matching results along with the paired tuples.

    Raises
    ------
    IdeasError
        If subject matching fails or produces insufficient matches.

    """
    expected_count = (
        len(reference_pairs) if reference_pairs is not None else None
    )

    group1_count = len(group1_files)
    group2_count = len(group2_files)

    if group1_count != group2_count:
        raise IdeasError(
            f"Paired analysis requires equal counts of {file_type} files per group. "
            f"Group 1 has {group1_count}, Group 2 has {group2_count}."
        )

    if expected_count is not None and group1_count != expected_count:
        raise IdeasError(
            f"Mismatch between {file_type} file count ({group1_count}) and activity matches "
            f"({expected_count})."
        )

    normalized_reference: List[Tuple[Optional[str], Optional[str]]] = []
    reference_contains_unknown = False
    matched_pairs: Optional[List[Tuple[pathlib.Path, pathlib.Path]]] = None

    if reference_pairs is not None:
        normalized_reference = [
            (
                _derive_subject_key(pair[0]),
                _derive_subject_key(pair[1]),
            )
            for pair in reference_pairs
        ]

        reference_contains_unknown = any(
            None in normalized_pair for normalized_pair in normalized_reference
        )

        def _build_lookup(
            file_list: List[pathlib.Path],
        ) -> Tuple[Dict[str, pathlib.Path], Set[str], bool]:
            lookup: Dict[str, pathlib.Path] = {}
            duplicates: Set[str] = set()
            contains_unknown = False
            for file_path in file_list:
                subject_key = _derive_subject_key(file_path)
                if subject_key is None:
                    contains_unknown = True
                    continue
                if subject_key in lookup:
                    duplicates.add(subject_key)
                lookup[subject_key] = file_path
            return lookup, duplicates, contains_unknown

        if reference_contains_unknown:
            logger.warning(
                "Unable to align %s files using activity reference pairs because "
                "one or more reference identifiers lack numeric components.",
                file_type,
            )
        else:
            group1_lookup, group1_duplicates, group1_contains_unknown = (
                _build_lookup(group1_files)
            )
            group2_lookup, group2_duplicates, group2_contains_unknown = (
                _build_lookup(group2_files)
            )

            if group1_contains_unknown or group2_contains_unknown:
                logger.warning(
                    "Unable to align %s files using activity reference pairs because "
                    "one or more file paths lack numeric identifiers.",
                    file_type,
                )
            elif group1_duplicates or group2_duplicates:
                duplicate_details: List[str] = []
                if group1_duplicates:
                    duplicate_details.append(
                        f"Group 1 duplicates: {sorted(group1_duplicates)}"
                    )
                if group2_duplicates:
                    duplicate_details.append(
                        f"Group 2 duplicates: {sorted(group2_duplicates)}"
                    )
                logger.warning(
                    "Unable to align %s files using activity reference pairs due to "
                    "duplicate subject identifiers (%s). Falling back to independent "
                    "matching.",
                    file_type,
                    "; ".join(duplicate_details),
                )
            else:
                missing_pairs = [
                    keys
                    for keys in normalized_reference
                    if keys[0] not in group1_lookup
                    or keys[1] not in group2_lookup
                ]
                if missing_pairs:
                    raise IdeasError(
                        "Subject mismatch detected between activity reference pairs and "
                        f"{file_type} files: {missing_pairs}"
                    )

                matched_pairs = [
                    (
                        group1_lookup[keys[0]],
                        group2_lookup[keys[1]],
                    )
                    for keys in normalized_reference
                ]
                logger.info(
                    "Aligned %s files using activity reference pairs; reused matched "
                    "subject ordering from activity data.",
                    file_type,
                )

    if matched_pairs is None:
        matched_pairs = _match_subject_pairs(
            group1_files=group1_files,
            group2_files=group2_files,
            subject_matching=subject_matching,
            file_type=file_type,
        )

    matched_count = len(matched_pairs)

    if matched_count < 2:
        raise IdeasError(
            f"Subject matching produced only {matched_count} matched {file_type} pair(s). "
            "At least 2 matched pairs are required for paired analysis."
        )

    if matched_count != group1_count:
        raise IdeasError(
            f"Subject matching identified {matched_count} matched {file_type} pair(s), "
            f"but {group1_count} {file_type} files were provided per group."
        )

    provided_pairs = list(zip(group1_files, group2_files))

    normalized_provided = [
        (_derive_subject_key(pair[0]), _derive_subject_key(pair[1]))
        for pair in provided_pairs
    ]
    normalized_matched = [
        (_derive_subject_key(pair[0]), _derive_subject_key(pair[1]))
        for pair in matched_pairs
    ]

    if normalized_provided != normalized_matched:
        logger.warning(
            "Provided %s file order does not align with subject matching results. "
            "Reordering to matched subject order.",
            file_type,
        )

    output_pairs = matched_pairs

    if reference_pairs is not None:
        matched_contains_unknown = any(
            None in normalized_pair for normalized_pair in normalized_matched
        )

        if reference_contains_unknown or matched_contains_unknown:
            logger.warning(
                "Unable to validate %s files against activity reference pairs because "
                "one or more identifiers lack numeric components.",
                file_type,
            )
        else:
            matched_lookup: Dict[
                Tuple[Optional[str], Optional[str]],
                Deque[Tuple[pathlib.Path, pathlib.Path]],
            ] = defaultdict(deque)
            for normalized_pair, pair in zip(
                normalized_matched,
                matched_pairs,
            ):
                matched_lookup[normalized_pair].append(pair)

            missing_pairs: List[Tuple[Optional[str], Optional[str]]] = []
            ordered_pairs: List[Tuple[pathlib.Path, pathlib.Path]] = []

            for keys in normalized_reference:
                candidates = matched_lookup.get(keys)
                if not candidates:
                    missing_pairs.append(keys)
                    continue
                ordered_pairs.append(candidates.popleft())

            if missing_pairs:
                raise IdeasError(
                    "Subject mismatch detected between activity reference pairs and "
                    f"{file_type} files: {missing_pairs}"
                )

            output_pairs = ordered_pairs

            if normalized_reference != normalized_matched:
                logger.info(
                    "Reordered %s files to align with activity subject matching reference.",
                    file_type,
                )

    return (
        [pair[0] for pair in output_pairs],
        [pair[1] for pair in output_pairs],
        output_pairs,
    )


def _classify_measure_type(measure_column: str) -> str:
    """Classify measure column into activity, correlation, or modulation.

    Parameters
    ----------
    measure_column : str
        Column name to classify

    Returns
    -------
    str
        One of: "activity", "correlation", or "modulation"

    Raises
    ------
    IdeasError
        If measure column cannot be classified into a supported type

    """
    measure_lower = measure_column.lower()
    if "activity" in measure_lower or "rate" in measure_lower:
        return "activity"
    elif "correlation" in measure_lower:
        return "correlation"
    elif "modulation" in measure_lower:
        return "modulation"
    else:
        raise IdeasError(
            f"Cannot classify measure type for column '{measure_column}'. "
            "Expected the name to include 'activity', 'rate', 'correlation', "
            "or 'modulation'."
        )


def _classify_data_source(measure_column: str) -> str:
    """Classify measure column as trace-based or event-based.

    Parameters
    ----------
    measure_column : str
        Column name to classify

    Returns
    -------
    str
        Either "trace" or "event"

    Raises
    ------
    IdeasError
        If column name does not contain 'trace' or 'event'

    """
    measure_lower = measure_column.lower()
    if "trace" in measure_lower:
        return "trace"
    elif "event" in measure_lower:
        return "event"
    else:
        raise IdeasError(
            f"Cannot classify data source for column '{measure_column}'. "
            f"Column name must contain 'trace' or 'event' to indicate data source."
        )


def _is_population_correlation(measure_column: str) -> bool:
    """Check if measure column is a population correlation (state-epoch aggregate).

    Population correlation columns (positive/negative correlations) are computed
    as aggregates across all cells in a state-epoch combination, resulting in
    identical values for all cells. These columns are NOT cell-level data and
    should NOT be analyzed with LMM (which assumes nested cell-level observations).

    Parameters
    ----------
    measure_column : str
        Column name to check

    Returns
    -------
    bool
        True if this is a population correlation column, False otherwise

    """
    measure_lower = measure_column.lower()
    # Population correlation columns are identified by "positive" or "negative"
    # combined with "correlation"
    return ("positive" in measure_lower or "negative" in measure_lower) and (
        "correlation" in measure_lower
    )


def _normalize_color_list(
    color_values: Optional[Union[str, Sequence[str]]]
) -> Optional[List[str]]:
    """Return a cleaned list of color strings from user input."""
    if color_values is None:
        return None

    if isinstance(color_values, str):
        tokens = re.split(r"[;,]", color_values)
        cleaned = [token.strip() for token in tokens if token.strip()]
        return cleaned or None

    normalized: List[str] = []
    for value in color_values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned:
            normalized.append(cleaned)

    return normalized or None


def _normalize_subject_matching(subject_matching: str) -> str:
    """Map UI-friendly subject matching names to implementation keywords."""
    normalized = (subject_matching or "").strip().lower()
    alias_map = {
        "name": "filename",
    }
    return alias_map.get(normalized, normalized or "order")


def _normalize_multiple_correction_scope(scope: Optional[str]) -> str:
    """Translate UI scope choices to the internal keywords."""
    if not scope:
        return "global"

    normalized = scope.strip().lower()
    alias_map = {
        "per_condition": "within_stratum",
        "within-stratum": "within_stratum",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    if normalized in {"global", "within_stratum"}:
        return normalized
    logger.warning(
        "Unknown multiple_correction_scope '%s'. Defaulting to 'global'.",
        scope,
    )
    return "global"


def _resolve_measure_metadata(
    measure_column: str,
    measure_type: str,
    data_source: str,
) -> Tuple[str, str]:
    """Resolve standardized display and measure identifiers.

    Parameters
    ----------
    measure_column : str
        Original column name used for analysis
    measure_type : str
        Classified measure type (activity, correlation, modulation)
    data_source : str
        Data source associated with the measure (trace or event)

    Returns
    -------
    Tuple[str, str]
        (measure_display_label, measure_identifier) aligned with standard-python
        toolbox outputs

    """
    column_lower = measure_column.lower()
    measure_type_lower = measure_type.lower()
    data_source_lower = data_source.lower()

    if measure_type_lower == "activity":
        if data_source_lower == "event":
            return "Event Rate", "event_rate"
        return "Activity", "activity"

    if measure_type_lower == "modulation":
        return "Modulation", "modulation"

    if measure_type_lower == "correlation":
        if "max" in column_lower:
            return "Max Correlation", "max_correlation"
        if "min" in column_lower:
            return "Min Correlation", "min_correlation"
        if "mean" in column_lower:
            return "Mean Correlation", "mean_correlation"
        if "positive" in column_lower:
            return "Positive Correlation", "positive_correlation"
        if "negative" in column_lower:
            return "Negative Correlation", "negative_correlation"
        return "Correlation", "correlation"

    return measure_column, measure_column


def _select_measures_for_analysis(
    combined_data: pd.DataFrame,
    measure_source: str,
    correlation_statistic: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Select which measure columns to analyze based on user parameters.

    This function implements comprehensive measure selection logic with auto-detection
    and explicit control for trace vs event data across activity, correlation, and
    modulation analyses.

    Parameters
    ----------
    combined_data : pd.DataFrame
        Combined data containing all potential measure columns
    measure_source : str
        Data source selection: "trace", "event", or "both"
    correlation_statistic : str
        Which statistic to use from correlations: "max", "min", or "mean"

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, List[str]]]
        First element contains dictionaries with keys "activity",
        "correlation", and "modulation" where each value is the list of
        column names to analyze. Second element contains dictionaries mapping
        each measure type to the resolved data sources ("trace" and/or "event")
        that will drive downstream processing.

    """
    correlation_statistic = str(correlation_statistic).strip().lower()
    selected_measures = {
        "activity": [],
        "correlation": [],
        "modulation": [],
    }
    selected_source_sets: Dict[str, Set[str]] = {
        "activity": set(),
        "correlation": set(),
        "modulation": set(),
    }

    available_columns = list(combined_data.columns)
    available_set = set(available_columns)

    allowed_correlation_stats: Set[str] = set()
    for correlation_data_type in ("trace", "event"):
        allowed_correlation_stats.update(
            MEASURE_COLUMNS["correlation_percell"][
                correlation_data_type
            ].keys()
        )
    if correlation_statistic not in allowed_correlation_stats:
        raise IdeasError(
            "Unsupported correlation_statistic "
            f"'{correlation_statistic}'. Expected one of "
            f"{sorted(allowed_correlation_stats)}."
        )

    def _add_unique(target_list: List[str], column: str) -> None:
        if column not in target_list:
            target_list.append(column)

    def _select_activity_for_source(source: str) -> bool:
        added = False
        for column in MEASURE_COLUMNS["activity"][source]:
            if column in available_set:
                _add_unique(selected_measures["activity"], column)
                selected_source_sets["activity"].add(source)
                added = True
        return added

    def _add_correlation_for_type(data_type: str) -> Tuple[bool, bool]:
        added_any = False
        per_cell_added = False

        stat_options = [correlation_statistic]
        population_stats = ["positive", "negative"]

        for stat_option in stat_options:
            per_cell_candidates = MEASURE_COLUMNS["correlation_percell"][
                data_type
            ][stat_option]
            for column in per_cell_candidates:
                if column in available_set:
                    _add_unique(selected_measures["correlation"], column)
                    logger.info(
                        "Selected %s per-cell correlation measure (%s): %s",
                        data_type,
                        stat_option,
                        column,
                    )
                    added_any = True
                    per_cell_added = True
                    selected_source_sets["correlation"].add(data_type)
                    break

        for direction in population_stats:
            for column in MEASURE_COLUMNS["correlation_population"][data_type][
                direction
            ]:
                if column in available_set:
                    _add_unique(selected_measures["correlation"], column)
                    logger.info(
                        f"Selected {data_type} population correlation measure: {column}"
                    )
                    added_any = True
                    selected_source_sets["correlation"].add(data_type)
                    break

        return added_any, per_cell_added

    def _find_modulation_columns(data_source: str) -> List[str]:
        matched: List[str] = []

        base_column = f"{data_source}_modulation"
        if base_column in available_set:
            _add_unique(matched, base_column)
            return matched

        def _should_skip(column_name: str) -> bool:
            lower = column_name.lower()
            return (
                "scores" in lower or "p_values" in lower or "p-values" in lower
            )

        if data_source == "trace":
            for column in available_columns:
                lower = column.lower()
                if "trace_modulation in " in lower and not _should_skip(
                    column
                ):
                    _add_unique(matched, column)

        elif data_source == "event":
            for column in available_columns:
                lower = column.lower()
                if "event_modulation in " in lower and not _should_skip(
                    column
                ):
                    _add_unique(matched, column)

        return matched

    def _select_modulation_for_source(source: str) -> bool:
        modulation_columns = _find_modulation_columns(source)
        if modulation_columns:
            selected_measures["modulation"].extend(modulation_columns)
            selected_source_sets["modulation"].add(source)
            return True
        return False

    source_mode = measure_source.lower().strip()
    if source_mode not in {"trace", "event", "both"}:
        raise IdeasError(
            f"Invalid measure_source '{measure_source}'. Expected 'trace', 'event', or 'both'."
        )

    # === ACTIVITY MEASURE SELECTION ===
    if source_mode == "trace":
        activity_added = _select_activity_for_source("trace")
        if not activity_added:
            logger.info(
                "Trace activity columns not found; falling back to event activity columns"
            )
            _select_activity_for_source("event")
    elif source_mode == "event":
        _select_activity_for_source("event")
    elif source_mode == "both":
        _select_activity_for_source("trace")
        _select_activity_for_source("event")

    # === CORRELATION MEASURE SELECTION ===
    if source_mode == "trace":
        trace_added, trace_per_cell_added = _add_correlation_for_type("trace")
        if not trace_per_cell_added:
            if trace_added:
                logger.info(
                    "Trace per-cell correlation columns not found; adding event "
                    "correlations for requested statistic"
                )
            else:
                logger.info(
                    "Trace correlation columns not found; falling back to event correlations"
                )
            _add_correlation_for_type("event")
    elif source_mode == "event":
        _add_correlation_for_type("event")
    elif source_mode == "both":
        _add_correlation_for_type("trace")
        _add_correlation_for_type("event")

    # === MODULATION MEASURE SELECTION ===
    if source_mode == "trace":
        trace_mod_added = _select_modulation_for_source("trace")
        if not trace_mod_added:
            logger.info(
                "Trace modulation columns not found; falling back to event modulation"
            )
            _select_modulation_for_source("event")
    elif source_mode == "event":
        _select_modulation_for_source("event")
    elif source_mode == "both":
        _select_modulation_for_source("trace")
        _select_modulation_for_source("event")

    selected_sources = {
        key: sorted(list(value)) for key, value in selected_source_sets.items()
    }

    return selected_measures, selected_sources


def _load_correlation_data_from_h5(
    h5_file_path: pathlib.Path,
    activity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Load correlation data from H5 file and convert to CSV-compatible DataFrame.

    Parameters
    ----------
    h5_file_path : pathlib.Path
        Path to H5 file containing correlation matrices
    activity_df : pd.DataFrame
        Activity DataFrame to extract cell names, states, and epochs

    Returns
    -------
    pd.DataFrame
        DataFrame with same structure as correlations_per_state_epoch_data.csv

    """
    import h5py

    logger.info(f"Loading correlation data from H5 file: {h5_file_path}")

    # Extract unique states, epochs, and cell info from activity data
    states = activity_df["state"].unique().tolist()
    epochs = activity_df["epoch"].unique().tolist()

    # Preserve original cell indices (may be non-contiguous) and associated names
    cell_metadata = (
        activity_df[["cell_index", "name"]]
        .drop_duplicates(subset="cell_index", keep="first")
        .sort_values("cell_index")
    )
    cell_indices = cell_metadata["cell_index"].tolist()
    cell_idx_to_name = dict(
        zip(cell_metadata["cell_index"], cell_metadata["name"])
    )

    correlation_rows = []

    with h5py.File(h5_file_path, "r") as f:
        # Check if H5 file has hierarchical structure (trace/event groups)
        has_groups = "trace" in f.keys() or "event" in f.keys()

        if has_groups:
            # New format with trace/event groups
            trace_group = f.get("trace", {})
            event_group = f.get("event", {})

            # Process all state-epoch combinations
            for state in states:
                for epoch in epochs:
                    key = f"{state}-{epoch}"

                    # Load trace correlation matrix
                    trace_corr_matrix = None
                    if key in trace_group:
                        trace_corr_matrix = trace_group[key][:]

                    # Load event correlation matrix
                    event_corr_matrix = None
                    if key in event_group:
                        event_corr_matrix = event_group[key][:]

                    # Calculate per-cell statistics for each cell
                    if (
                        trace_corr_matrix is not None
                        or event_corr_matrix is not None
                    ):
                        for matrix_idx, cell_idx in enumerate(cell_indices):
                            row_data = {
                                "name": cell_idx_to_name.get(
                                    cell_idx, f"cell_{cell_idx}"
                                ),
                                "cell_index": cell_idx,
                                "state": state,
                                "epoch": epoch,
                            }

                            # Calculate trace correlation statistics
                            if (
                                trace_corr_matrix is not None
                                and matrix_idx < trace_corr_matrix.shape[0]
                            ):
                                cell_corrs = np.asarray(
                                    trace_corr_matrix[matrix_idx, :]
                                )
                                # Exclude self-correlation (diagonal)
                                other_corrs = np.delete(cell_corrs, matrix_idx)
                                other_corrs = other_corrs[
                                    ~np.isnan(other_corrs)
                                ]

                                if other_corrs.size > 0:
                                    row_data["max_trace_correlation"] = float(
                                        np.max(other_corrs)
                                    )
                                    row_data["min_trace_correlation"] = float(
                                        np.min(other_corrs)
                                    )
                                    row_data["mean_trace_correlation"] = float(
                                        np.mean(other_corrs)
                                    )

                                    # Calculate positive/negative correlations
                                    pos_corrs = other_corrs[other_corrs > 0]
                                    neg_corrs = other_corrs[other_corrs < 0]

                                    row_data["positive_trace_correlation"] = (
                                        float(np.mean(pos_corrs))
                                        if pos_corrs.size > 0
                                        else np.nan
                                    )
                                    row_data["negative_trace_correlation"] = (
                                        float(np.mean(neg_corrs))
                                        if neg_corrs.size > 0
                                        else np.nan
                                    )
                                else:
                                    row_data["max_trace_correlation"] = np.nan
                                    row_data["min_trace_correlation"] = np.nan
                                    row_data["mean_trace_correlation"] = np.nan
                                    row_data["positive_trace_correlation"] = (
                                        np.nan
                                    )
                                    row_data["negative_trace_correlation"] = (
                                        np.nan
                                    )
                            else:
                                row_data["max_trace_correlation"] = np.nan
                                row_data["min_trace_correlation"] = np.nan
                                row_data["mean_trace_correlation"] = np.nan
                                row_data["positive_trace_correlation"] = np.nan
                                row_data["negative_trace_correlation"] = np.nan

                            # Calculate event correlation statistics
                            if (
                                event_corr_matrix is not None
                                and matrix_idx < event_corr_matrix.shape[0]
                            ):
                                cell_corrs = np.asarray(
                                    event_corr_matrix[matrix_idx, :]
                                )
                                # Exclude self-correlation (diagonal)
                                other_corrs = np.delete(cell_corrs, matrix_idx)
                                other_corrs = other_corrs[
                                    ~np.isnan(other_corrs)
                                ]

                                if other_corrs.size > 0:
                                    row_data["max_event_correlation"] = float(
                                        np.max(other_corrs)
                                    )
                                    row_data["min_event_correlation"] = float(
                                        np.min(other_corrs)
                                    )
                                    row_data["mean_event_correlation"] = float(
                                        np.mean(other_corrs)
                                    )

                                    # Calculate positive/negative correlations
                                    pos_corrs = other_corrs[other_corrs > 0]
                                    neg_corrs = other_corrs[other_corrs < 0]

                                    row_data["positive_event_correlation"] = (
                                        float(np.mean(pos_corrs))
                                        if pos_corrs.size > 0
                                        else np.nan
                                    )
                                    row_data["negative_event_correlation"] = (
                                        float(np.mean(neg_corrs))
                                        if neg_corrs.size > 0
                                        else np.nan
                                    )
                                else:
                                    row_data["max_event_correlation"] = np.nan
                                    row_data["min_event_correlation"] = np.nan
                                    row_data["mean_event_correlation"] = np.nan
                                    row_data["positive_event_correlation"] = (
                                        np.nan
                                    )
                                    row_data["negative_event_correlation"] = (
                                        np.nan
                                    )
                            else:
                                row_data["max_event_correlation"] = np.nan
                                row_data["min_event_correlation"] = np.nan
                                row_data["mean_event_correlation"] = np.nan
                                row_data["positive_event_correlation"] = np.nan
                                row_data["negative_event_correlation"] = np.nan

                            correlation_rows.append(row_data)
        else:
            # Legacy format without groups is no longer supported; the structures diverged
            raise IdeasError(
                f"H5 file {h5_file_path} uses an unsupported legacy format without trace/event "
                "groups. Please regenerate correlation outputs with the updated toolbox to "
                "include the required hierarchy."
            )

    correlation_df = pd.DataFrame(correlation_rows)
    logger.info(
        f"Loaded {len(correlation_df)} correlation records from H5 file"
    )

    return correlation_df


def _load_group_data(
    activity_csv_files: List[pathlib.Path],
    correlation_csv_files: Optional[List[pathlib.Path]],
    modulation_csv_files: Optional[List[pathlib.Path]],
    group_name: str,
    group_id: int,
    data_pairing: str,
) -> Dict[str, Any]:
    """Load and combine data from CSV/H5 files for a single group.

    Parameters
    ----------
    activity_csv_files : List[pathlib.Path]
        List of activity_per_state_epoch_data.csv files
    correlation_csv_files : Optional[List[pathlib.Path]]
        Optional list of correlations_per_state_epoch_data.csv files OR
        pairwise_correlation_heatmaps.h5 files. If None, correlation data
        will not be loaded.
        (CSV must include positive_trace_correlation and negative_trace_correlation
        columns generated by state_epoch_baseline_analysis; optional event
        correlation columns will be detected automatically.
        H5 files should contain trace and/or event groups with correlation matrices.)
    modulation_csv_files : Optional[List[pathlib.Path]]
        Optional list of modulation_vs_baseline_data.csv files. If None,
        modulation data will not be loaded.
    group_name : str
        Name of the group
    group_id : int
        Numeric ID of the group
    data_pairing : str
        Type of data pairing ("paired" or "unpaired")

    Returns
    -------
    Dict[str, Any]
        Dictionary containing combined data for the group

    """
    logger.info(f"Loading {len(activity_csv_files)} subjects for {group_name}")

    # ========================================================================
    # CLEAN SUBJECT ID ASSIGNMENT - Consistent with Population and Correlation Tools
    # ========================================================================
    # This follows the exact pattern used in combine_compare_population_data and
    # combine_compare_correlation_data for consistency during toolbox merging.
    #
    # UNPAIRED ANALYSIS:
    #   - Format: "{Group_Name}_subject_N" (ensures subjects from different groups are distinct)
    #   - Example: "Group_1_subject_1", "Group_2_subject_1" (different subjects)
    # PAIRED ANALYSIS:
    #   - Format: "subject_N" (identical subject IDs across groups enable proper matching)
    #   - Example: "subject_1" in both groups (same subject, matched observations)
    # ========================================================================

    group_prefix = (
        f"{group_name.replace(' ', '_')}_"
        if data_pairing == "unpaired"
        else ""
    )

    if data_pairing == "unpaired":
        logger.info(
            f"Clean subject ID assignment for {group_name}: "
            f"Using group-prefixed format '{group_prefix}subject_N' for unpaired analysis"
        )
    else:
        logger.info(
            f"Clean subject ID assignment for {group_name}: "
            f"Using standard format 'subject_N' for paired analysis"
        )

    # Load activity data
    activity_dfs = []
    for i, file_path in enumerate(activity_csv_files):
        df = pd.read_csv(file_path)
        filename = file_path.name
        df["file"] = filename
        df["subject_id"] = f"subject_{i + 1}"
        df["normalized_subject_id"] = f"{group_prefix}subject_{i + 1}"
        df["group_name"] = group_name
        df["group_id"] = group_id
        activity_dfs.append(df)

    combined_activity = pd.concat(activity_dfs, ignore_index=True)

    # Load correlation data (CSV or H5 format) - only if provided
    combined_correlation = None
    if correlation_csv_files:
        correlation_dfs = []
        for i, file_path in enumerate(correlation_csv_files):
            # Check file extension to determine format
            if file_path.suffix.lower() == ".h5":
                # Load from H5 file using activity data for metadata
                activity_subject_df = (
                    activity_dfs[i]
                    if i < len(activity_dfs)
                    else combined_activity
                )
                df = _load_correlation_data_from_h5(
                    file_path, activity_subject_df
                )
            else:
                # Load from CSV file
                df = pd.read_csv(file_path)

            filename = file_path.name
            df["file"] = filename
            df["subject_id"] = f"subject_{i + 1}"
            df["normalized_subject_id"] = f"{group_prefix}subject_{i + 1}"
            df["group_name"] = group_name
            df["group_id"] = group_id
            correlation_dfs.append(df)

        combined_correlation = pd.concat(correlation_dfs, ignore_index=True)
        logger.info(f"Loaded correlation data for {group_name}")
    else:
        logger.info(
            f"Skipping correlation data for {group_name} (not provided)"
        )

    # Load modulation data - only if provided
    combined_modulation = None
    if modulation_csv_files:
        modulation_dfs = []
        for i, file_path in enumerate(modulation_csv_files):
            df = pd.read_csv(file_path)
            df = _materialize_modulation_columns(df)
            filename = file_path.name
            df["file"] = filename
            df["subject_id"] = f"subject_{i + 1}"
            df["normalized_subject_id"] = f"{group_prefix}subject_{i + 1}"
            df["group_name"] = group_name
            df["group_id"] = group_id
            modulation_dfs.append(df)

        combined_modulation = pd.concat(modulation_dfs, ignore_index=True)
        logger.info(f"Loaded modulation data for {group_name}")
    else:
        logger.info(
            f"Skipping modulation data for {group_name} (not provided)"
        )

    return {
        "activity": combined_activity,
        "correlation": combined_correlation,
        "modulation": combined_modulation,
        "group_name": group_name,
        "group_id": group_id,
        "n_subjects": len(activity_csv_files),
    }


def _extract_metadata_from_data(
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], str, str]:
    """Extract states, epochs, and baseline information from loaded data.

    The baseline state and epoch are ALWAYS extracted from the modulation data
    (computed during state_epoch_baseline_analysis) and cannot be user-specified.
    This ensures consistency with the pre-computed modulation classifications.

    Parameters
    ----------
    group1_data : Dict[str, Any]
        Data dictionary for group 1
    group2_data : Optional[Dict[str, Any]]
        Optional data dictionary for group 2. When provided, this function
        validates that both groups share identical states and epochs and that
        their baseline assignments match.

    Returns
    -------
    Tuple[List[str], List[str], str, str]
        Lists of states and epochs, and extracted baseline state and epoch

    Raises
    ------
    IdeasError
        If metadata extraction fails or produces invalid values

    """

    def _ordered_unique(series: pd.Series) -> List[Any]:
        ordered_values: List[Any] = []
        for raw_value in series:
            if pd.isna(raw_value):
                continue
            if isinstance(raw_value, str) and not raw_value.strip():
                continue
            if raw_value not in ordered_values:
                ordered_values.append(raw_value)
        return ordered_values

    def _extract_group_metadata(
        group_data: Dict[str, Any],
        group_label: str,
    ) -> Tuple[List[str], List[str], str, str]:
        activity_df = group_data["activity"]

        if activity_df.empty:
            raise IdeasError(
                f"Cannot extract metadata from empty activity data for {group_label}"
            )

        states_local = _ordered_unique(activity_df["state"])
        epochs_local = _ordered_unique(activity_df["epoch"])

        if not states_local:
            raise IdeasError(
                f"No valid states found in activity data for {group_label}"
            )
        if not epochs_local:
            raise IdeasError(
                f"No valid epochs found in activity data for {group_label}"
            )

        modulation_df = group_data["modulation"]

        if modulation_df is None or modulation_df.empty:
            logger.warning(
                "%s modulation data is empty - falling back to alphabetically "
                "first state/epoch for baseline selection",
                group_label,
            )
            baseline_state_local = (
                sorted(states_local)[0] if states_local else None
            )
            baseline_epoch_local = (
                sorted(epochs_local)[0] if epochs_local else None
            )

            if not baseline_state_local or not baseline_epoch_local:
                raise IdeasError(
                    f"Cannot determine baseline for {group_label}: modulation "
                    "data is empty and no states/epochs available"
                )

            logger.info(
                "%s baseline fallback: state='%s', epoch='%s'",
                group_label,
                baseline_state_local,
                baseline_epoch_local,
            )
            return (
                states_local,
                epochs_local,
                baseline_state_local,
                baseline_epoch_local,
            )

        if "baseline_state" not in modulation_df.columns:
            raise IdeasError(
                f"Missing 'baseline_state' column in modulation data for "
                f"{group_label}. Input files must be from "
                "state_epoch_baseline_analysis."
            )
        if "baseline_epoch" not in modulation_df.columns:
            raise IdeasError(
                f"Missing 'baseline_epoch' column in modulation data for "
                f"{group_label}. Input files must be from "
                "state_epoch_baseline_analysis."
            )

        baseline_state_series = modulation_df["baseline_state"].dropna()
        if baseline_state_series.empty:
            raise IdeasError(
                "Cannot extract baseline state: all values in 'baseline_state' "
                f"column are NaN for {group_label}. "
                "Input modulation files may be corrupted or incomplete."
            )

        baseline_state_local = baseline_state_series.iloc[0]

        unique_baseline_states = baseline_state_series.unique()
        if len(unique_baseline_states) > 1:
            raise IdeasError(
                "Inconsistent baseline states found in modulation data for "
                f"{group_label}: {unique_baseline_states.tolist()}. "
                "All rows must share the same baseline state."
            )

        baseline_epoch_series = modulation_df["baseline_epoch"].dropna()
        if baseline_epoch_series.empty:
            raise IdeasError(
                "Cannot extract baseline epoch: all values in 'baseline_epoch' "
                f"column are NaN for {group_label}. "
                "Input modulation files may be corrupted or incomplete."
            )

        baseline_epoch_local = baseline_epoch_series.iloc[0]

        unique_baseline_epochs = baseline_epoch_series.unique()
        if len(unique_baseline_epochs) > 1:
            raise IdeasError(
                "Inconsistent baseline epochs found in modulation data for "
                f"{group_label}: {unique_baseline_epochs.tolist()}. "
                "All rows must share the same baseline epoch."
            )

        return (
            states_local,
            epochs_local,
            baseline_state_local,
            baseline_epoch_local,
        )

    (
        states,
        epochs,
        baseline_state,
        baseline_epoch,
    ) = _extract_group_metadata(group1_data, "Group 1")

    if group2_data is not None:
        (
            states_g2,
            epochs_g2,
            baseline_state_g2,
            baseline_epoch_g2,
        ) = _extract_group_metadata(group2_data, "Group 2")

        if baseline_state_g2 != baseline_state:
            raise IdeasError(
                "Baseline state mismatch between groups. "
                f"Group 1: '{baseline_state}', Group 2: '{baseline_state_g2}'. "
                "Both groups must be analyzed with the same baseline state. "
                "Check the 'baseline_state' column in your modulation CSV "
                "files from state_epoch_baseline_analysis outputs."
            )

        if baseline_epoch_g2 != baseline_epoch:
            raise IdeasError(
                "Baseline epoch mismatch between groups. "
                f"Group 1: '{baseline_epoch}', Group 2: '{baseline_epoch_g2}'. "
                "Both groups must be analyzed with the same baseline epoch. "
                "Check the 'baseline_epoch' column in your modulation CSV "
                "files from state_epoch_baseline_analysis outputs."
            )

        states_diff_g1 = sorted(set(states) - set(states_g2))
        states_diff_g2 = sorted(set(states_g2) - set(states))
        if states_diff_g1 or states_diff_g2:
            raise IdeasError(
                "State mismatch between groups. "
                f"Group 1 states: {states}. Group 2 states: {states_g2}. "
                "Both groups must have identical state sets."
            )

        epochs_diff_g1 = sorted(set(epochs) - set(epochs_g2))
        epochs_diff_g2 = sorted(set(epochs_g2) - set(epochs))
        if epochs_diff_g1 or epochs_diff_g2:
            raise IdeasError(
                "Epoch mismatch between groups. "
                f"Group 1 epochs: {epochs}. Group 2 epochs: {epochs_g2}. "
                "Both groups must have identical epoch sets."
            )

        logger.info(
            "Validated metadata consistency across groups "
            "(baseline: %s-%s)",
            baseline_state,
            baseline_epoch,
        )

    return states, epochs, baseline_state, baseline_epoch


def _reclassify_state_epoch_neurons(
    group_data: Dict[str, Any],
    states: List[str],
    epochs: List[str],
    baseline_state: str,
    baseline_epoch: str,
    significance_threshold: float,
) -> Dict[str, Any]:
    """Reclassify neurons with a custom significance threshold.

    Parameters
    ----------
    group_data : Dict[str, Any]
        Group data dictionary containing modulation DataFrame
    states : List[str]
        List of state names
    epochs : List[str]
        List of epoch names
    baseline_state : str
        Baseline state name
    baseline_epoch : str
        Baseline epoch name
    significance_threshold : float
        New p-value threshold for reclassification

    Returns
    -------
    Dict[str, Any]
        Updated group data with reclassified neurons

    Notes
    -----
    This function uses the same alpha/2 logic as assign_modulation in
    combine_compare_peri_event_data.py for two one-tailed comparisons
    (testing both up- and down-modulation). This Bonferroni correction
    ensures consistency across all combine/compare tools.

    Mathematical basis:
        - For two one-tailed tests (up-mod and down-mod), we use alpha/2
        - If p < (threshold/2) AND score < 0: down-modulated
        - If p < (threshold/2) AND score > 0: up-modulated
        - Otherwise: non-modulated

    """
    modulation_df = group_data.get("modulation")
    if modulation_df is None or modulation_df.empty:
        logger.warning("No modulation data available for reclassification")
        return group_data

    # Create a copy to avoid modifying the original
    updated_data = group_data.copy()
    modulation_df = modulation_df.copy()

    # Track reclassification summary
    reclassification_summary = {}

    # Process each state-epoch combination (excluding baseline)
    for state in states:
        for epoch in epochs:
            if state == baseline_state and epoch == baseline_epoch:
                continue  # Skip baseline itself

            combination = f"{state}-{epoch}"
            column_templates = [
                (
                    f"trace_modulation_scores in {combination}",
                    f"trace_p_values in {combination}",
                    f"trace_modulation in {combination}",
                ),
                (
                    f"event_modulation_scores in {combination}",
                    f"event_p_values in {combination}",
                    f"event_modulation in {combination}",
                ),
                (
                    f"modulation scores in {combination}",
                    f"p-values in {combination}",
                    f"modulation in {combination}",
                ),
            ]

            for score_col, p_col, mod_col in column_templates:
                if score_col not in modulation_df.columns:
                    continue
                if p_col not in modulation_df.columns:
                    continue
                if mod_col not in modulation_df.columns:
                    continue

                # Reclassify neurons using alpha/2 for two one-tailed comparisons
                # This matches the logic in assign_modulation from
                # combine_compare_peri_event_data.py
                def classify_neuron(
                    row, p_column=p_col, score_column=score_col
                ):
                    """Classify a single neuron based on modulation score and p-value."""
                    if pd.isna(row[p_column]) or pd.isna(row[score_column]):
                        return 0.0  # Non-modulated if data is missing

                    if row[p_column] < (significance_threshold / 2):
                        if row[score_column] < 0:
                            return -1.0  # Down-modulated
                        elif row[score_column] > 0:
                            return 1.0  # Up-modulated
                    return 0.0  # Non-modulated

                modulation_df[mod_col] = modulation_df.apply(
                    classify_neuron, axis=1
                )

                up_mod = (modulation_df[mod_col] == 1).sum()
                down_mod = (modulation_df[mod_col] == -1).sum()
                non_mod = (modulation_df[mod_col] == 0).sum()
                reclassification_summary[f"{mod_col}"] = (
                    up_mod,
                    down_mod,
                    non_mod,
                )

    # Log summary of reclassification
    logger.info(
        f"Reclassified neurons for {updated_data['group_name']} "
        f"with threshold={significance_threshold}"
    )
    for combination, (up, down, non) in reclassification_summary.items():
        logger.debug(
            f"  {combination}: {up} up-modulated, {down} down-modulated, {non} non-modulated"
        )

    # Update the modulation DataFrame in the group data
    updated_data["modulation"] = modulation_df
    return updated_data


def _perform_statistical_comparison_csv(
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    group_names: List[str],
    comparison_dimension: str,
    states: List[str],
    epochs: List[str],
    baseline_state: str,
    baseline_epoch: str,
    data_pairing: str,
    *,
    measure_source: str = "trace",
    correlation_statistic: str,
    significance_threshold: Optional[float],
    multiple_correction: str,
    multiple_correction_scope: str = "global",
    effect_size: str,
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
    output_dir: str,
    enable_lmm_analysis: bool = True,
    save_lmm_outputs: bool = False,
) -> Dict[str, Any]:
    """Perform statistical comparison between groups using CSV data.

    This function performs statistical comparisons across states or epochs,
    following the same pattern as population and correlation tools.

    NOTE: For now, this focuses on state or epoch comparisons separately.
    Cross-epoch-state comparisons will be addressed later.

    enable_lmm_analysis allows evaluation of imbalanced designs with the linear
    mixed model workflow. When save_lmm_outputs is False (default) LMM results
    are retained in-memory without generating files.

    """
    # Use default significance threshold if None provided
    # This is only used for statistical tests between groups, NOT for neuron classification
    # (neuron classification uses pre-computed modulation or explicit reclassification)
    if significance_threshold is None:
        significance_threshold = 0.05
        logger.debug(
            "Using default significance threshold: 0.05 for statistical comparisons"
        )

    logger.info(f"Performing {comparison_dimension} comparison between groups")

    measure_source = measure_source.lower().strip()

    multiple_correction = _require_supported_multiple_correction_method(
        multiple_correction
    )

    # ========================================================================
    # SUBJECT ID VALIDATION - Consistency with Population and Correlation Tools
    # ========================================================================
    # Validate subject ID format for clean assignment rules
    # This is critical for ensuring accurate paired/unpaired statistical comparisons
    # and matches the validation pattern used in combine_compare_population_data
    # and combine_compare_correlation_data tools
    # ========================================================================

    should_validate_individual = not (
        data_pairing == "paired" and group2_data is not None
    )

    if should_validate_individual:
        if "activity" in group1_data and not group1_data["activity"].empty:
            validate_subject_id_format(
                data=group1_data["activity"],
                data_pairing=data_pairing,
                group_names=group_names,
                context=f"{group_names[0]} activity data",
            )

        if (
            "correlation" in group1_data
            and group1_data["correlation"] is not None
            and not group1_data["correlation"].empty
        ):
            validate_subject_id_format(
                data=group1_data["correlation"],
                data_pairing=data_pairing,
                group_names=group_names,
                context=f"{group_names[0]} correlation data",
            )

        if (
            "modulation" in group1_data
            and group1_data["modulation"] is not None
            and not group1_data["modulation"].empty
        ):
            validate_subject_id_format(
                data=group1_data["modulation"],
                data_pairing=data_pairing,
                group_names=group_names,
                context=f"{group_names[0]} modulation data",
            )

        if group2_data is not None:
            if "activity" in group2_data and not group2_data["activity"].empty:
                validate_subject_id_format(
                    data=group2_data["activity"],
                    data_pairing=data_pairing,
                    group_names=group_names,
                    context=f"{group_names[1]} activity data",
                )

            if (
                "correlation" in group2_data
                and group2_data["correlation"] is not None
                and not group2_data["correlation"].empty
            ):
                validate_subject_id_format(
                    data=group2_data["correlation"],
                    data_pairing=data_pairing,
                    group_names=group_names,
                    context=f"{group_names[1]} correlation data",
                )

            if (
                "modulation" in group2_data
                and group2_data["modulation"] is not None
                and not group2_data["modulation"].empty
            ):
                validate_subject_id_format(
                    data=group2_data["modulation"],
                    data_pairing=data_pairing,
                    group_names=group_names,
                    context=f"{group_names[1]} modulation data",
                )

    # Prepare combined DataFrames for each measure type
    if group2_data is None:
        logger.info("Single group analysis: comparing within group")
        combined_activity = group1_data["activity"].copy()
        combined_correlation = (
            group1_data["correlation"].copy()
            if group1_data["correlation"] is not None
            else None
        )
        combined_modulation = (
            group1_data["modulation"].copy()
            if group1_data["modulation"] is not None
            else None
        )
        has_single_group = True
    else:
        logger.info("Two group analysis: comparing between groups")
        combined_activity = pd.concat(
            [group1_data["activity"], group2_data["activity"]],
            ignore_index=True,
        )
        # Combine correlation data only if both groups have it
        if (
            group1_data["correlation"] is not None
            and group2_data["correlation"] is not None
        ):
            combined_correlation = pd.concat(
                [group1_data["correlation"], group2_data["correlation"]],
                ignore_index=True,
            )
        else:
            combined_correlation = None
        # Combine modulation data only if both groups have it
        if (
            group1_data["modulation"] is not None
            and group2_data["modulation"] is not None
        ):
            combined_modulation = pd.concat(
                [group1_data["modulation"], group2_data["modulation"]],
                ignore_index=True,
            )
        else:
            combined_modulation = None
        has_single_group = False

    combined_frames = {
        "activity": combined_activity,
        "correlation": combined_correlation,
        "modulation": combined_modulation,
    }

    # Validate subject identifiers for combined data where applicable
    for frame_name, frame in combined_frames.items():
        if frame is None or frame.empty:
            continue
        validate_subject_id_format(
            data=frame,
            data_pairing=data_pairing,
            group_names=group_names,
            context=f"Combined {frame_name} data",
        )

    # Determine which dimension we're comparing
    comparison_values = states if comparison_dimension == "states" else epochs
    comparison_col = "state" if comparison_dimension == "states" else "epoch"

    # Use comprehensive measure selection based on user parameters
    logger.info("Measure source selection: %s", measure_source)

    available_columns = set()
    for frame in combined_frames.values():
        if frame is None:
            continue
        available_columns.update(frame.columns)

    column_reference_df = pd.DataFrame(columns=sorted(available_columns))

    (
        selected_measures,
        selected_sources,
    ) = _select_measures_for_analysis(
        combined_data=column_reference_df,
        measure_source=measure_source,
        correlation_statistic=correlation_statistic,
    )

    logger.info(
        "Resolved measure sources: activity=%s, correlation=%s, modulation=%s",
        selected_sources.get("activity", []),
        selected_sources.get("correlation", []),
        selected_sources.get("modulation", []),
    )

    # Flatten all selected measures for analysis
    all_measure_columns = (
        selected_measures["activity"]
        + selected_measures["correlation"]
        + selected_measures["modulation"]
    )

    if not all_measure_columns:
        logger.error(
            f"No valid measure columns found for selected analysis types. "
            f"Available columns: {column_reference_df.columns.tolist()}"
        )
        return {
            "aov": pd.DataFrame(),
            "pairwise": pd.DataFrame(),
            "comparison_dimension": comparison_dimension,
            "error": "No valid measure columns found",
            "selected_measures": selected_measures,
            "selected_sources": selected_sources,
            "multiple_correction": multiple_correction,
        }

    logger.info(f"Selected measures for analysis: {all_measure_columns}")

    # Use the existing comparison utilities
    # Analyze each measure column separately and combine results
    all_aov_dfs: List[pd.DataFrame] = []
    all_pairwise_dfs: List[pd.DataFrame] = []
    all_lmm_dfs: List[pd.DataFrame] = []
    all_lmm_pairwise_dfs: List[pd.DataFrame] = []
    all_lmm_output_dfs: List[pd.DataFrame] = []

    try:
        for measure_column in all_measure_columns:
            # Determine measure type and data source for labeling
            measure_type = _classify_measure_type(measure_column)
            data_source = _classify_data_source(measure_column)

            source_frame = combined_frames.get(measure_type)
            if source_frame is None or source_frame.empty:
                logger.info(
                    f"Skipping {measure_column} because {measure_type} data is unavailable"
                )
                continue

            if measure_column not in source_frame.columns:
                logger.info(
                    f"Skipping {measure_column} because column is missing from {measure_type} data"
                )
                continue

            required_cols = [
                "group_name",
                "normalized_subject_id",
                measure_column,
            ]
            missing_cols = [
                col for col in required_cols if col not in source_frame.columns
            ]
            if missing_cols:
                logger.info(
                    f"Skipping {measure_column} due to missing columns: {missing_cols}"
                )
                continue

            logger.info(
                f"Analyzing {data_source} {measure_type}: {measure_column}"
            )

            # The utility function returns (aov_df, pairwise_df) as a tuple
            aov_df, pairwise_df = calculate_state_epoch_comparison_stats(
                df=source_frame,
                comparison_dimension=comparison_dimension,
                comparison_values=comparison_values,
                has_single_group=has_single_group,
                data_pairing=data_pairing,
                significance_threshold=significance_threshold,
                multiple_correction=multiple_correction,
                multiple_correction_scope=multiple_correction_scope,
                effect_size_method=effect_size,
                group_comparison_type=group_comparison_type,
                parametric=parametric,
                measure_column=measure_column,
            )

            _, measure_label = _resolve_measure_metadata(
                measure_column=measure_column,
                measure_type=measure_type,
                data_source=data_source,
            )

            lmm_df = pd.DataFrame()
            lmm_pairwise_df = pd.DataFrame()

            # Skip LMM for population correlation columns
            # These are state-epoch aggregates (not cell-level data) and would
            # result in pseudoreplication if analyzed with LMM
            if enable_lmm_analysis and not _is_population_correlation(
                measure_column
            ):
                lmm_df, lmm_pairwise_df = calculate_state_epoch_lmm_stats(
                    df=source_frame,
                    comparison_dim=comparison_col,
                    comparison_values=comparison_values,
                    has_single_group=has_single_group,
                    data_pairing=data_pairing,
                    effect_size=effect_size,
                    correction=multiple_correction,
                    measure_name=measure_column,
                    significance_threshold=significance_threshold,
                    group_comparison_type=group_comparison_type,
                    parametric=parametric,
                )
            elif enable_lmm_analysis and _is_population_correlation(
                measure_column
            ):
                logger.info(
                    f"Skipping LMM for {measure_column}: population correlation columns "
                    f"are state-epoch aggregates (not cell-level data). Use ANOVA results instead."
                )

            # Add metadata columns for comprehensive result tracking
            if aov_df is not None and not aov_df.empty:
                aov_df = aov_df.copy()
                aov_df["Measure"] = measure_label
                aov_df["Measure_Type"] = (
                    measure_type  # "activity", "correlation", "modulation"
                )
                aov_df["Data_Source"] = data_source  # "trace" or "event"
                all_aov_dfs.append(aov_df)

            if pairwise_df is not None and not pairwise_df.empty:
                pairwise_df = pairwise_df.copy()
                pairwise_df["Measure"] = measure_label
                pairwise_df["Measure_Type"] = measure_type
                pairwise_df["Data_Source"] = data_source
                all_pairwise_dfs.append(pairwise_df)

            if lmm_df is not None and not lmm_df.empty:
                lmm_df = lmm_df.copy()
                lmm_df = lmm_df.rename(
                    columns={
                        "Coefficient": "coefficient",
                        "SE": "std_error",
                        "t": "t_value",
                        "p-unc": "p_value",
                    }
                )
                lmm_df["Measure"] = measure_label
                lmm_df["Measure_Type"] = measure_type
                lmm_df["Data_Source"] = data_source
                lmm_df["stat_method"] = "lmm"
                lmm_df["analysis_level"] = "subject"
                lmm_df["Comparison"] = (
                    f"{comparison_col.capitalize()} Comparison"
                )
                all_lmm_output_dfs.append(lmm_df.copy())
                lmm_df = _finalize_statistical_output(lmm_df, "anova")
                all_lmm_dfs.append(lmm_df)

            if lmm_pairwise_df is not None and not lmm_pairwise_df.empty:
                lmm_pairwise_df = lmm_pairwise_df.copy()
                lmm_pairwise_df["Measure"] = measure_label
                lmm_pairwise_df["Measure_Type"] = measure_type
                lmm_pairwise_df["Data_Source"] = data_source
                if "Comparison" not in lmm_pairwise_df.columns:
                    lmm_pairwise_df["Comparison"] = (
                        f"{comparison_col.capitalize()} Comparison"
                    )
                all_lmm_pairwise_dfs.append(lmm_pairwise_df)

        # Combine all results
        if all_aov_dfs:
            aov_df = pd.concat(all_aov_dfs, ignore_index=True)
        else:
            aov_df = pd.DataFrame()

        if all_pairwise_dfs:
            pairwise_df = pd.concat(all_pairwise_dfs, ignore_index=True)
        else:
            pairwise_df = pd.DataFrame()

        if all_lmm_dfs:
            lmm_df = pd.concat(all_lmm_dfs, ignore_index=True)
        else:
            lmm_df = pd.DataFrame()

        if all_lmm_pairwise_dfs:
            lmm_pairwise_df = pd.concat(
                all_lmm_pairwise_dfs, ignore_index=True
            )
        else:
            lmm_pairwise_df = pd.DataFrame()

        # Add analysis parameters to the dataframes for comprehensive CSV output
        # (consistent with combine_compare_population_data and combine_compare_correlation_data)
        normalized_group_comparison_type = (
            str(group_comparison_type).strip()
            if group_comparison_type is not None
            else ""
        )

        normalized_parametric = _normalize_parametric_value(parametric)

        analysis_params = {
            "significance_threshold": significance_threshold,
            "multiple_correction": multiple_correction,
            "effect_size_method": effect_size,
            "data_pairing": data_pairing,
            "group_comparison_type": normalized_group_comparison_type,
            "parametric": normalized_parametric,
        }

        # Add parameters as additional columns to preserve analysis context
        for param, value in analysis_params.items():
            if not aov_df.empty:
                aov_df.loc[:, param] = value
            if not pairwise_df.empty:
                pairwise_df.loc[:, param] = value
            if not lmm_df.empty:
                lmm_df.loc[:, param] = value
            if not lmm_pairwise_df.empty:
                lmm_pairwise_df.loc[:, param] = value

        # Preserve pre-cleanup copies (retain Data_Source column for file splitting)
        output_aov_df = aov_df.copy() if not aov_df.empty else pd.DataFrame()
        output_pairwise_df = (
            pairwise_df.copy() if not pairwise_df.empty else pd.DataFrame()
        )
        if all_lmm_output_dfs:
            output_lmm_df = pd.concat(
                all_lmm_output_dfs,
                ignore_index=True,
            )
        else:
            output_lmm_df = pd.DataFrame()
        output_lmm_pairwise_df = (
            lmm_pairwise_df.copy()
            if not lmm_pairwise_df.empty
            else pd.DataFrame()
        )

        # Apply final cleanup to remove unwanted columns before saving
        # (consistent with standard-python toolbox)
        if not aov_df.empty:
            aov_df = _cleanup_final_csv_columns(aov_df, "anova")
        if not pairwise_df.empty:
            pairwise_df = _cleanup_final_csv_columns(pairwise_df, "pairwise")
        if not lmm_df.empty:
            lmm_df = _cleanup_final_csv_columns(lmm_df, "anova")
        if not lmm_pairwise_df.empty:
            lmm_pairwise_df = _cleanup_final_csv_columns(
                lmm_pairwise_df, "pairwise"
            )

        # Files saved directly in output_dir (or current dir if output_dir is empty)
        # Use current directory if output_dir is empty (for Docker runs)
        save_dir = output_dir if output_dir else "."

        # ========================================================================
        # SAVE SEPARATE TRACE AND EVENT STATISTICAL COMPARISON FILES
        # ========================================================================
        # Split statistical results by Data_Source for separate trace/event analysis
        # This enables cleaner organization when both trace and event data are analyzed
        # ========================================================================

        trace_aov_df = pd.DataFrame()
        trace_pairwise_df = pd.DataFrame()
        event_aov_df = pd.DataFrame()
        event_pairwise_df = pd.DataFrame()
        trace_lmm_df = pd.DataFrame()
        trace_lmm_pairwise_df = pd.DataFrame()
        event_lmm_df = pd.DataFrame()
        event_lmm_pairwise_df = pd.DataFrame()

        # Split ANOVA results by data source
        if not output_aov_df.empty and "Data_Source" in output_aov_df.columns:
            trace_mask = output_aov_df["Data_Source"] == "trace"
            event_mask = output_aov_df["Data_Source"] == "event"

            if trace_mask.any():
                trace_aov_df = output_aov_df[trace_mask].copy()
                trace_aov_path = os.path.join(
                    save_dir, TRACE_ANOVA_COMPARISONS_CSV
                )
                trace_aov_df.to_csv(trace_aov_path, index=False)
                logger.info(
                    f"Saved trace-specific ANOVA results to {trace_aov_path} "
                    f"({len(trace_aov_df)} rows)"
                )

            if event_mask.any():
                event_aov_df = output_aov_df[event_mask].copy()
                event_aov_path = os.path.join(
                    save_dir, EVENT_ANOVA_COMPARISONS_CSV
                )
                event_aov_df.to_csv(event_aov_path, index=False)
                logger.info(
                    f"Saved event-specific ANOVA results to {event_aov_path} "
                    f"({len(event_aov_df)} rows)"
                )

        # Split pairwise results by data source
        if (
            not output_pairwise_df.empty
            and "Data_Source" in output_pairwise_df.columns
        ):
            trace_mask = output_pairwise_df["Data_Source"] == "trace"
            event_mask = output_pairwise_df["Data_Source"] == "event"

            if trace_mask.any():
                trace_pairwise_df = output_pairwise_df[trace_mask].copy()
                trace_pairwise_path = os.path.join(
                    save_dir, TRACE_PAIRWISE_COMPARISONS_CSV
                )
                trace_pairwise_df.to_csv(trace_pairwise_path, index=False)
                logger.info(
                    f"Saved trace-specific pairwise results to {trace_pairwise_path} "
                    f"({len(trace_pairwise_df)} rows)"
                )

            if event_mask.any():
                event_pairwise_df = output_pairwise_df[event_mask].copy()
                event_pairwise_path = os.path.join(
                    save_dir, EVENT_PAIRWISE_COMPARISONS_CSV
                )
                event_pairwise_df.to_csv(event_pairwise_path, index=False)
                logger.info(
                    f"Saved event-specific pairwise results to {event_pairwise_path} "
                    f"({len(event_pairwise_df)} rows)"
                )

        if save_lmm_outputs:
            if (
                not output_lmm_df.empty
                and "Data_Source" in output_lmm_df.columns
            ):
                trace_mask = output_lmm_df["Data_Source"] == "trace"
                event_mask = output_lmm_df["Data_Source"] == "event"

                if trace_mask.any():
                    trace_lmm_df = output_lmm_df[trace_mask].copy()
                    trace_lmm_path = os.path.join(
                        save_dir, TRACE_LMM_COMPARISONS_CSV
                    )
                    trace_lmm_df.to_csv(trace_lmm_path, index=False)
                    logger.info(
                        "Saved trace-specific LMM results to %s (%d rows)",
                        trace_lmm_path,
                        len(trace_lmm_df),
                    )

                if event_mask.any():
                    event_lmm_df = output_lmm_df[event_mask].copy()
                    event_lmm_path = os.path.join(
                        save_dir, EVENT_LMM_COMPARISONS_CSV
                    )
                    event_lmm_df.to_csv(event_lmm_path, index=False)
                    logger.info(
                        "Saved event-specific LMM results to %s (%d rows)",
                        event_lmm_path,
                        len(event_lmm_df),
                    )

            if (
                not output_lmm_pairwise_df.empty
                and "Data_Source" in output_lmm_pairwise_df.columns
            ):
                trace_mask = output_lmm_pairwise_df["Data_Source"] == "trace"
                event_mask = output_lmm_pairwise_df["Data_Source"] == "event"

                if trace_mask.any():
                    trace_lmm_pairwise_df = output_lmm_pairwise_df[
                        trace_mask
                    ].copy()
                    trace_lmm_pairwise_path = os.path.join(
                        save_dir, TRACE_LMM_PAIRWISE_COMPARISONS_CSV
                    )
                    trace_lmm_pairwise_df.to_csv(
                        trace_lmm_pairwise_path, index=False
                    )
                    logger.info(
                        "Saved trace-specific LMM pairwise results to %s (%d rows)",
                        trace_lmm_pairwise_path,
                        len(trace_lmm_pairwise_df),
                    )

                if event_mask.any():
                    event_lmm_pairwise_df = output_lmm_pairwise_df[
                        event_mask
                    ].copy()
                    event_lmm_pairwise_path = os.path.join(
                        save_dir, EVENT_LMM_PAIRWISE_COMPARISONS_CSV
                    )
                    event_lmm_pairwise_df.to_csv(
                        event_lmm_pairwise_path, index=False
                    )
                    logger.info(
                        "Saved event-specific LMM pairwise results to %s (%d rows)",
                        event_lmm_pairwise_path,
                        len(event_lmm_pairwise_df),
                    )

        # Return results in dict format for consistency with calling code
        return {
            "aov": aov_df if aov_df is not None else pd.DataFrame(),
            "pairwise": (
                pairwise_df if pairwise_df is not None else pd.DataFrame()
            ),
            "lmm": lmm_df if lmm_df is not None else pd.DataFrame(),
            "lmm_pairwise": (
                lmm_pairwise_df
                if lmm_pairwise_df is not None
                else pd.DataFrame()
            ),
            "comparison_dimension": comparison_dimension,
            "selected_measures": selected_measures,
            "selected_sources": selected_sources,
            "multiple_correction": multiple_correction,
        }

    except Exception as e:
        logger.warning(f"Statistical comparison failed: {e}")
        # Return empty results structure
        return {
            "aov": pd.DataFrame(),
            "pairwise": pd.DataFrame(),
            "lmm": pd.DataFrame(),
            "lmm_pairwise": pd.DataFrame(),
            "comparison_dimension": comparison_dimension,
            "error": str(e),
            "selected_measures": selected_measures,
            "selected_sources": selected_sources,
            "multiple_correction": multiple_correction,
        }


def _generate_combined_outputs_csv(
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    comparison_results: Dict[str, Any],
    group_names: List[str],
    group_colors: List[str],
    states: List[str],
    epochs: List[str],
    baseline_state: str,
    baseline_epoch: str,
    comparison_dimension: str,
    data_pairing: str,
    output_dir: str,
    correlation_statistic: str,
    state_colors: Optional[List[str]] = None,
    epoch_colors: Optional[List[str]] = None,
    modulation_colors: Optional[List[str]] = None,
) -> None:
    """Generate combined visualization and output files from CSV data.

    Follows standard-python toolbox naming conventions:
    - Per-group files: {group_name}_combined_*_data.csv
    - All files saved directly in output_dir (no subdirectories)
    - Consistent with combine_compare_population_data and combine_compare_correlation_data
    - Respects correlation_statistic when generating per-cell preview files
    """
    logger.info("Generating combined comparison outputs...")

    if not output_dir:
        output_dir = "."

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    group_file_prefixes = _resolve_group_file_prefixes(group_names)  # ["group1", "group2"] for files/dirs
    
    # Create sanitized group name prefixes for human-readable preview filenames
    group_preview_prefixes = [
        _sanitize_filename_component(name, f"group{i+1}")
        for i, name in enumerate(group_names)
    ]

    default_selection: Dict[str, List[str]] = {
        "activity": [],
        "correlation": [],
        "modulation": [],
    }
    selected_measures = {
        key: comparison_results.get("selected_measures", {}).get(key, [])
        for key in default_selection
    }
    selected_sources_raw = comparison_results.get("selected_sources", {})
    selected_source_sets: Dict[str, Set[str]] = {
        key: set(selected_sources_raw.get(key, []))
        for key in default_selection
    }

    logger.info(
        "Using resolved measure configuration: activity=%s, correlation=%s, modulation=%s",
        selected_source_sets["activity"] or {"none"},
        selected_source_sets["correlation"] or {"none"},
        selected_source_sets["modulation"] or {"none"},
    )

    # ========================================================================
    # SAVE PER-GROUP DATA FILES (following standard-python toolbox pattern)
    # ========================================================================
    # Format: {group_name}_combined_{data_type}_data.csv
    # Matches pattern from combine_compare_correlation_data.py
    # ========================================================================

    # Save Group 1 activity data using result key name (rename rules will handle final naming)
    group1_activity_clean = _cleanup_final_csv_columns(
        group1_data["activity"], "mixed"
    )
    group1_activity_filename = _group_output_filename(
        group_file_prefixes,
        0,
        GROUP_COMBINED_ACTIVITY_SUFFIX,
    )
    group1_activity_path = os.path.join(output_dir, group1_activity_filename)
    group1_activity_clean.to_csv(group1_activity_path, index=False)
    logger.info(
        f"Saved {group_names[0]} activity data to {group1_activity_path}"
    )

    # Save Group 1 correlation data using result key name (if provided)
    if group1_data["correlation"] is not None:
        group1_correlation_clean = _cleanup_final_csv_columns(
            group1_data["correlation"], "mixed"
        )
        group1_correlation_filename = _group_output_filename(
            group_file_prefixes,
            0,
            GROUP_COMBINED_TRACE_CORRELATION_SUFFIX,
        )
        group1_correlation_path = os.path.join(
            output_dir,
            group1_correlation_filename,
        )
        group1_correlation_clean.to_csv(group1_correlation_path, index=False)
        logger.info(
            f"Saved {group_names[0]} correlation data to {group1_correlation_path}"
        )
    else:
        logger.info(
            f"Skipping {group_names[0]} correlation data save (not provided)"
        )

    # Save Group 1 modulation data using result key name (if provided)
    if group1_data["modulation"] is not None:
        group1_modulation_clean = _cleanup_final_csv_columns(
            group1_data["modulation"], "mixed"
        )
        group1_modulation_filename = _group_output_filename(
            group_file_prefixes,
            0,
            GROUP_COMBINED_MODULATION_SUFFIX,
        )
        group1_modulation_path = os.path.join(
            output_dir,
            group1_modulation_filename,
        )
        group1_modulation_clean.to_csv(group1_modulation_path, index=False)
        logger.info(
            f"Saved {group_names[0]} modulation data to {group1_modulation_path}"
        )
    else:
        logger.info(
            f"Skipping {group_names[0]} modulation data save (not provided)"
        )

    # Save Group 2 data if present using result key names
    if group2_data is not None and len(group_names) > 1:
        # Save Group 2 activity data using result key name
        group2_activity_clean = _cleanup_final_csv_columns(
            group2_data["activity"], "mixed"
        )
        group2_activity_filename = _group_output_filename(
            group_file_prefixes,
            1,
            GROUP_COMBINED_ACTIVITY_SUFFIX,
        )
        group2_activity_path = os.path.join(
            output_dir,
            group2_activity_filename,
        )
        group2_activity_clean.to_csv(group2_activity_path, index=False)
        logger.info(
            f"Saved {group_names[1]} activity data to {group2_activity_path}"
        )

        # Save Group 2 correlation data using result key name (if provided)
        if group2_data["correlation"] is not None:
            group2_correlation_clean = _cleanup_final_csv_columns(
                group2_data["correlation"], "mixed"
            )
            group2_correlation_filename = _group_output_filename(
                group_file_prefixes,
                1,
                GROUP_COMBINED_TRACE_CORRELATION_SUFFIX,
            )
            group2_correlation_path = os.path.join(
                output_dir,
                group2_correlation_filename,
            )
            group2_correlation_clean.to_csv(
                group2_correlation_path, index=False
            )
            logger.info(
                f"Saved {group_names[1]} correlation data to {group2_correlation_path}"
            )
        else:
            logger.info(
                f"Skipping {group_names[1]} correlation data save (not provided)"
            )

        # Save Group 2 modulation data using result key name (if provided)
        if group2_data["modulation"] is not None:
            group2_modulation_clean = _cleanup_final_csv_columns(
                group2_data["modulation"], "mixed"
            )
            group2_modulation_filename = _group_output_filename(
                group_file_prefixes,
                1,
                GROUP_COMBINED_MODULATION_SUFFIX,
            )
            group2_modulation_path = os.path.join(
                output_dir,
                group2_modulation_filename,
            )
            group2_modulation_clean.to_csv(group2_modulation_path, index=False)
            logger.info(
                f"Saved {group_names[1]} modulation data to {group2_modulation_path}"
            )
        else:
            logger.info(
                f"Skipping {group_names[1]} modulation data save (not provided)"
            )

    # ========================================================================
    # GENERATE SUMMARY STATISTICS (saved to output_dir)
    # ========================================================================

    # Combine data for summary
    if group2_data is None:
        combined_activity = group1_data["activity"].copy()
        combined_correlation = (
            group1_data["correlation"].copy()
            if group1_data["correlation"] is not None
            else None
        )
        combined_modulation = (
            group1_data["modulation"].copy()
            if group1_data["modulation"] is not None
            else None
        )
    else:
        combined_activity = pd.concat(
            [group1_data["activity"], group2_data["activity"]],
            ignore_index=True,
        )
        # Combine correlation only if both groups have it
        if (
            group1_data["correlation"] is not None
            and group2_data["correlation"] is not None
        ):
            combined_correlation = pd.concat(
                [group1_data["correlation"], group2_data["correlation"]],
                ignore_index=True,
            )
        else:
            combined_correlation = None
        # Combine modulation only if both groups have it
        if (
            group1_data["modulation"] is not None
            and group2_data["modulation"] is not None
        ):
            combined_modulation = pd.concat(
                [group1_data["modulation"], group2_data["modulation"]],
                ignore_index=True,
            )
        else:
            combined_modulation = None

    preview_group1_data = _filter_group_data_by_sources(
        group1_data,
        selected_source_sets,
    )
    preview_group2_data = _filter_group_data_by_sources(
        group2_data,
        selected_source_sets,
    )

    plot_modulation_df = _filter_measure_dataframe(
        combined_modulation,
        selected_source_sets["modulation"],
        "modulation",
    )

    modulation_measures_selected = bool(selected_measures["modulation"])

    trace_activity_selected = "trace" in selected_source_sets["activity"]
    event_activity_selected = "event" in selected_source_sets["activity"]
    trace_correlation_selected = "trace" in selected_source_sets["correlation"]
    event_correlation_selected = "event" in selected_source_sets["correlation"]

    # ========================================================================
    # GENERATE PER-GROUP PREVIEW PLOTS (matching standard-python toolbox)
    # ========================================================================
    # Following the same pattern as combine_compare_population_data and
    # combine_compare_correlation_data, generate per-group boxplot previews

    _generate_per_group_previews(
        group1_data=preview_group1_data,
        group2_data=preview_group2_data,
        group_names=group_names,
        states=states,
        epochs=epochs,
        output_dir=output_dir,
        correlation_statistic=correlation_statistic,
        comparison_dimension=comparison_dimension,
        state_colors=state_colors,
        epoch_colors=epoch_colors,
        modulation_colors=modulation_colors,
    )

    # ========================================================================
    # GENERATE COMPARISON PLOTS (saved to output_dir)
    # ========================================================================
    # Generate comparison visualizations for combined data
    # Check if trace or event statistical files exist

    # Define save_dir for all file checks
    save_dir = output_dir if output_dir else "."

    # Check if any statistical CSV files exist
    trace_anova_path = os.path.join(save_dir, TRACE_ANOVA_COMPARISONS_CSV)
    trace_pairwise_path = os.path.join(
        save_dir, TRACE_PAIRWISE_COMPARISONS_CSV
    )
    event_anova_path = os.path.join(save_dir, EVENT_ANOVA_COMPARISONS_CSV)
    event_pairwise_path = os.path.join(
        save_dir, EVENT_PAIRWISE_COMPARISONS_CSV
    )

    stats_files_exist = (
        os.path.exists(trace_anova_path)
        or os.path.exists(trace_pairwise_path)
        or os.path.exists(event_anova_path)
        or os.path.exists(event_pairwise_path)
    )

    if stats_files_exist:
        try:
            # Skip generic activity/correlation plots - these are now generated
            # as source-specific plots (trace_activity, event_activity, etc.)
            # to avoid duplication
            logger.info("Generating source-specific comparison plots...")

            # Generate modulation plots with source-specific naming
            if modulation_measures_selected and not plot_modulation_df.empty:
                # Generate trace modulation plot
                trace_modulation = _filter_data_by_source(
                    plot_modulation_df, "trace", measure_type="modulation"
                )
                if not trace_modulation.empty:
                    plot_state_epoch_comparison(
                        data=trace_modulation,
                        group_names=group_names,
                        group_colors=group_colors,
                        states=states,
                        epochs=epochs,
                        baseline_state=baseline_state,
                        baseline_epoch=baseline_epoch,
                        comparison_dimension=comparison_dimension,
                        data_pairing=data_pairing,
                        output_dir=output_dir,
                        data_type="trace_modulation",
                    )

                # Generate event modulation plot
                event_modulation = _filter_data_by_source(
                    plot_modulation_df, "event", measure_type="modulation"
                )
                if not event_modulation.empty:
                    plot_state_epoch_comparison(
                        data=event_modulation,
                        group_names=group_names,
                        group_colors=group_colors,
                        states=states,
                        epochs=epochs,
                        baseline_state=baseline_state,
                        baseline_epoch=baseline_epoch,
                        comparison_dimension=comparison_dimension,
                        data_pairing=data_pairing,
                        output_dir=output_dir,
                        data_type="event_modulation",
                    )
            else:
                logger.info(
                    "Skipping modulation comparison plots - no modulation measures "
                    "selected or data unavailable"
                )

            logger.info("Generated modulation comparison plots")

            # ================================================================
            # GENERATE TRACE-SPECIFIC COMPARISON PLOTS
            # ================================================================
            # Check if trace-specific CSV files exist
            trace_anova_path = os.path.join(
                save_dir, TRACE_ANOVA_COMPARISONS_CSV
            )
            trace_pairwise_path = os.path.join(
                save_dir, TRACE_PAIRWISE_COMPARISONS_CSV
            )
            trace_files_exist = os.path.exists(
                trace_anova_path
            ) or os.path.exists(trace_pairwise_path)

            if trace_files_exist:
                if trace_activity_selected or trace_correlation_selected:
                    logger.info(
                        "Generating trace-specific comparison plots..."
                    )

                    trace_activity = _filter_data_by_source(
                        combined_activity, "trace", measure_type="activity"
                    )
                    trace_correlation = _filter_data_by_source(
                        combined_correlation,
                        "trace",
                        measure_type="correlation",
                    )

                    if trace_activity_selected and not trace_activity.empty:
                        plot_state_epoch_comparison(
                            data=trace_activity,
                            group_names=group_names,
                            group_colors=group_colors,
                            states=states,
                            epochs=epochs,
                            baseline_state=baseline_state,
                            baseline_epoch=baseline_epoch,
                            comparison_dimension=comparison_dimension,
                            data_pairing=data_pairing,
                            output_dir=output_dir,
                            data_type="trace_activity",
                        )

                    if (
                        trace_correlation_selected
                        and not trace_correlation.empty
                    ):
                        plot_state_epoch_comparison(
                            data=trace_correlation,
                            group_names=group_names,
                            group_colors=group_colors,
                            states=states,
                            epochs=epochs,
                            baseline_state=baseline_state,
                            baseline_epoch=baseline_epoch,
                            comparison_dimension=comparison_dimension,
                            data_pairing=data_pairing,
                            output_dir=output_dir,
                            data_type="trace_correlation",
                            correlation_statistic=correlation_statistic,
                        )

                        if (
                            "positive_trace_correlation"
                            in trace_correlation.columns
                            and trace_correlation["positive_trace_correlation"]
                            .notna()
                            .any()
                        ):
                            plot_state_epoch_comparison(
                                data=trace_correlation,
                                group_names=group_names,
                                group_colors=group_colors,
                                states=states,
                                epochs=epochs,
                                baseline_state=baseline_state,
                                baseline_epoch=baseline_epoch,
                                comparison_dimension=comparison_dimension,
                                data_pairing=data_pairing,
                                output_dir=output_dir,
                                data_type="trace_positive_correlation",
                                correlation_statistic=correlation_statistic,
                            )

                        if (
                            "negative_trace_correlation"
                            in trace_correlation.columns
                            and trace_correlation["negative_trace_correlation"]
                            .notna()
                            .any()
                        ):
                            plot_state_epoch_comparison(
                                data=trace_correlation,
                                group_names=group_names,
                                group_colors=group_colors,
                                states=states,
                                epochs=epochs,
                                baseline_state=baseline_state,
                                baseline_epoch=baseline_epoch,
                                comparison_dimension=comparison_dimension,
                                data_pairing=data_pairing,
                                output_dir=output_dir,
                                data_type="trace_negative_correlation",
                                correlation_statistic=correlation_statistic,
                            )

                    logger.info("Generated trace-specific comparison plots")
                else:
                    logger.info(
                        "Skipping trace-specific comparison plots - trace measures not selected"
                    )

            # ================================================================
            # GENERATE EVENT-SPECIFIC COMPARISON PLOTS
            # ================================================================
            # Check if event-specific CSV files exist
            event_anova_path = os.path.join(
                save_dir, EVENT_ANOVA_COMPARISONS_CSV
            )
            event_pairwise_path = os.path.join(
                save_dir, EVENT_PAIRWISE_COMPARISONS_CSV
            )
            event_files_exist = os.path.exists(
                event_anova_path
            ) or os.path.exists(event_pairwise_path)

            if event_files_exist:
                if event_activity_selected or event_correlation_selected:
                    logger.info(
                        "Generating event-specific comparison plots..."
                    )

                    event_activity = _filter_data_by_source(
                        combined_activity, "event", measure_type="activity"
                    )
                    event_correlation = _filter_data_by_source(
                        combined_correlation,
                        "event",
                        measure_type="correlation",
                    )

                    if event_activity_selected and not event_activity.empty:
                        plot_state_epoch_comparison(
                            data=event_activity,
                            group_names=group_names,
                            group_colors=group_colors,
                            states=states,
                            epochs=epochs,
                            baseline_state=baseline_state,
                            baseline_epoch=baseline_epoch,
                            comparison_dimension=comparison_dimension,
                            data_pairing=data_pairing,
                            output_dir=output_dir,
                            data_type="event_activity",
                        )

                    if (
                        event_correlation_selected
                        and not event_correlation.empty
                    ):
                        plot_state_epoch_comparison(
                            data=event_correlation,
                            group_names=group_names,
                            group_colors=group_colors,
                            states=states,
                            epochs=epochs,
                            baseline_state=baseline_state,
                            baseline_epoch=baseline_epoch,
                            comparison_dimension=comparison_dimension,
                            data_pairing=data_pairing,
                            output_dir=output_dir,
                            data_type="event_correlation",
                            correlation_statistic=correlation_statistic,
                        )

                        if (
                            "positive_event_correlation"
                            in event_correlation.columns
                            and event_correlation["positive_event_correlation"]
                            .notna()
                            .any()
                        ):
                            plot_state_epoch_comparison(
                                data=event_correlation,
                                group_names=group_names,
                                group_colors=group_colors,
                                states=states,
                                epochs=epochs,
                                baseline_state=baseline_state,
                                baseline_epoch=baseline_epoch,
                                comparison_dimension=comparison_dimension,
                                data_pairing=data_pairing,
                                output_dir=output_dir,
                                data_type="event_positive_correlation",
                                correlation_statistic=correlation_statistic,
                            )

                        if (
                            "negative_event_correlation"
                            in event_correlation.columns
                            and event_correlation["negative_event_correlation"]
                            .notna()
                            .any()
                        ):
                            plot_state_epoch_comparison(
                                data=event_correlation,
                                group_names=group_names,
                                group_colors=group_colors,
                                states=states,
                                epochs=epochs,
                                baseline_state=baseline_state,
                                baseline_epoch=baseline_epoch,
                                comparison_dimension=comparison_dimension,
                                data_pairing=data_pairing,
                                output_dir=output_dir,
                                data_type="event_negative_correlation",
                                correlation_statistic=correlation_statistic,
                            )

                    logger.info("Generated event-specific comparison plots")
                else:
                    logger.info(
                        "Skipping event-specific comparison plots - event measures not selected"
                    )

        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    else:
        logger.info(
            "Skipping comparison plots - no statistical comparison files found"
        )


def _filter_measure_dataframe(
    df: Optional[pd.DataFrame],
    allowed_sources: Set[str],
    measure_type: str,
) -> pd.DataFrame:
    """Return a copy of df containing only columns matching allowed sources."""
    if df is None:
        return pd.DataFrame()

    filtered_df = df.copy()
    if filtered_df.empty:
        return filtered_df

    allowed = set(allowed_sources)
    columns_to_drop: List[str] = []

    for column in filtered_df.columns:
        column_lower = column.lower()

        if measure_type == "activity":
            if "trace" in column_lower:
                if "trace" not in allowed:
                    columns_to_drop.append(column)
            elif "event" in column_lower or "rate" in column_lower:
                if "event" not in allowed:
                    columns_to_drop.append(column)
            elif (
                "activity" in column_lower or "rate" in column_lower
            ) and not allowed:
                columns_to_drop.append(column)
        elif measure_type == "correlation":
            if "correlation" in column_lower:
                if "trace" in column_lower and "trace" not in allowed:
                    columns_to_drop.append(column)
                elif "event" in column_lower and "event" not in allowed:
                    columns_to_drop.append(column)
                elif (
                    "trace" not in column_lower
                    and "event" not in column_lower
                    and not allowed
                ):
                    columns_to_drop.append(column)
        elif measure_type == "modulation":
            if "modulation" in column_lower:
                if "trace" in column_lower and "trace" not in allowed:
                    columns_to_drop.append(column)
                elif "event" in column_lower and "event" not in allowed:
                    columns_to_drop.append(column)
                elif (
                    "trace" not in column_lower
                    and "event" not in column_lower
                    and not allowed
                ):
                    columns_to_drop.append(column)

    if columns_to_drop:
        filtered_df = filtered_df.drop(
            columns=columns_to_drop, errors="ignore"
        )

    return filtered_df


def _generate_dimension_variants(value: Any) -> List[str]:
    """Return normalized string variants for a dimension value."""
    if value is None or pd.isna(value):
        return []

    normalized = str(value).strip()
    if not normalized:
        return []

    variants: List[str] = [normalized]
    underscored = normalized.replace(" ", "_")
    if underscored != normalized:
        variants.append(underscored)

    return variants


def _lookup_modulation_value(row: pd.Series, prefix: str) -> Any:
    """Lookup modulation-related value for the row based on state-epoch combination."""
    state_variants = _generate_dimension_variants(row.get("state"))
    epoch_variants = _generate_dimension_variants(row.get("epoch"))

    if not state_variants or not epoch_variants:
        return np.nan

    for state_variant in state_variants:
        for epoch_variant in epoch_variants:
            column_name = f"{prefix}{state_variant}-{epoch_variant}"
            if column_name in row.index:
                return row[column_name]

    return np.nan


def _materialize_modulation_columns(
    modulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create base modulation columns from per-combination columns when necessary."""
    if modulation_df is None or modulation_df.empty:
        return modulation_df

    if (
        "state" not in modulation_df.columns
        or "epoch" not in modulation_df.columns
    ):
        return modulation_df

    result = modulation_df.copy()

    for source in ("trace", "event"):
        base_column = f"{source}_modulation"
        base_prefix = f"{base_column} in "
        if base_column not in result.columns:
            if any(
                column.startswith(base_prefix) for column in result.columns
            ):
                result[base_column] = result.apply(
                    lambda row, prefix=base_prefix: _lookup_modulation_value(
                        row, prefix
                    ),
                    axis=1,
                )
                result[base_column] = pd.to_numeric(
                    result[base_column], errors="coerce"
                )

        score_column = f"{source}_modulation_scores"
        score_prefix = f"{score_column} in "
        if score_column not in result.columns:
            if any(
                column.startswith(score_prefix) for column in result.columns
            ):
                result[score_column] = result.apply(
                    lambda row, prefix=score_prefix: _lookup_modulation_value(
                        row, prefix
                    ),
                    axis=1,
                )
                result[score_column] = pd.to_numeric(
                    result[score_column], errors="coerce"
                )

        pval_column = f"{source}_p_values"
        pval_prefix = f"{pval_column} in "
        if pval_column not in result.columns:
            if any(
                column.startswith(pval_prefix) for column in result.columns
            ):
                result[pval_column] = result.apply(
                    lambda row, prefix=pval_prefix: _lookup_modulation_value(
                        row, prefix
                    ),
                    axis=1,
                )
                result[pval_column] = pd.to_numeric(
                    result[pval_column], errors="coerce"
                )

    return result


def _filter_group_data_by_sources(
    group_data: Optional[Dict[str, Any]],
    selected_sources: Dict[str, Set[str]],
) -> Optional[Dict[str, Any]]:
    """Build a group data copy containing only trace/event columns that were selected."""
    if group_data is None:
        return None

    filtered_data = dict(group_data)

    for measure_type in ["activity", "correlation", "modulation"]:
        df = group_data.get(measure_type)
        allowed_sources = selected_sources.get(measure_type, set())
        filtered_data[measure_type] = _filter_measure_dataframe(
            df=df,
            allowed_sources=allowed_sources,
            measure_type=measure_type,
        )

    return filtered_data


def _filter_data_by_source(
    data: Optional[pd.DataFrame], data_source: str, measure_type: str
) -> pd.DataFrame:
    """Filter dataframe to include only columns related to specified data source.

    Parameters
    ----------
    data : Optional[pd.DataFrame]
        Combined dataframe with both trace and event measures. If None, returns
        an empty dataframe.
    data_source : str
        Data source to filter: "trace" or "event"
    measure_type : str
        Type of measure: "activity", "correlation", or "modulation"

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with only trace or event measures

    """
    if data is None:
        return pd.DataFrame()

    if data.empty:
        return data

    # Identify columns that belong to the specified data source
    relevant_columns = []

    # Always keep these structural columns
    structural_cols = [
        "state",
        "epoch",
        "group_name",
        "normalized_subject_id",
        "state_color",
        "state_hex_color",
        "epoch_color",
        "epoch_hex_color",
        "cell_index",
        "name",
        "subject_id",
        "group_id",
    ]

    for col in data.columns:
        if col in structural_cols:
            relevant_columns.append(col)
        elif data_source in col.lower():
            relevant_columns.append(col)

    if not relevant_columns:
        logger.warning(
            f"No {data_source} columns found in {measure_type} data"
        )
        return pd.DataFrame()

    filtered_data = data[relevant_columns].copy()

    if measure_type == "modulation":
        has_modulation_column = any(
            "modulation" in column.lower() for column in filtered_data.columns
        )
        if not has_modulation_column:
            logger.info(
                "Skipping %s modulation data - no modulation columns present",
                data_source,
            )
            return pd.DataFrame()

    if measure_type == "correlation":
        has_corr_column = any(
            "correlation" in column.lower()
            for column in filtered_data.columns
            if column not in structural_cols
        )
        if not has_corr_column:
            logger.info(
                "Skipping %s correlation data - no correlation columns present",
                data_source,
            )
            return pd.DataFrame()

    if measure_type == "activity":
        has_activity_column = any(
            any(keyword in column.lower() for keyword in ["activity", "rate"])
            for column in filtered_data.columns
            if column not in structural_cols
        )
        if not has_activity_column:
            logger.info(
                "Skipping %s activity data - no activity or rate columns present",
                data_source,
            )
            return pd.DataFrame()

    return filtered_data


def _resolve_color_map(
    items: List[str],
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    user_colors: Optional[List[str]],
    *,
    item_label: str,
    item_column: str,
    color_columns: List[str],
) -> Dict[str, str]:
    """Resolve deterministic colors for a set of items (states or epochs)."""
    color_map: Dict[str, str] = {}

    if user_colors and isinstance(user_colors, list):
        for index, item in enumerate(items):
            if index < len(user_colors):
                color = user_colors[index]
                if isinstance(color, str) and color.strip():
                    color_map[item] = color.strip()
                    logger.debug(
                        "Using user-provided color for %s '%s': %s",
                        item_label,
                        item,
                        color.strip(),
                    )

    def _update_from_dataframe(df: pd.DataFrame) -> None:
        if df is None or df.empty or item_column not in df.columns:
            return

        for color_column in color_columns:
            if color_column not in df.columns:
                continue
            non_null = df[[item_column, color_column]].dropna()
            for item_value, color_value in non_null.values:
                if (
                    isinstance(item_value, str)
                    and item_value in items
                    and isinstance(color_value, str)
                    and color_value.strip()
                ):
                    color_map.setdefault(item_value, color_value.strip())

    for group_data in [group1_data, group2_data]:
        if not group_data:
            continue
        for key in ["activity", "modulation", "correlation"]:
            _update_from_dataframe(group_data.get(key))

    missing_items = [item for item in items if item not in color_map]
    if missing_items:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        cmap = colormaps.get_cmap("tab20").resampled(len(missing_items))
        denominator = max(len(missing_items) - 1, 1)
        for index, item in enumerate(missing_items):
            fractional_index = index / denominator if denominator else 0.0
            color_map[item] = to_hex(cmap(fractional_index))

    logger.info(
        "Resolved %s color map: %s",
        item_label,
        {item: color_map[item] for item in items},
    )

    return color_map


def _resolve_state_color_map(
    states: List[str],
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    user_state_colors: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build a deterministic state-to-color mapping from available data."""
    return _resolve_color_map(
        items=states,
        group1_data=group1_data,
        group2_data=group2_data,
        user_colors=user_state_colors,
        item_label="state",
        item_column="state",
        color_columns=[
            "state_color",
            "state_hex_color",
            "state_rgb",
        ],
    )


def _resolve_epoch_color_map(
    epochs: List[str],
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    user_epoch_colors: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Build a deterministic epoch-to-color mapping from available data."""
    return _resolve_color_map(
        items=epochs,
        group1_data=group1_data,
        group2_data=group2_data,
        user_colors=user_epoch_colors,
        item_label="epoch",
        item_column="epoch",
        color_columns=[
            "epoch_color",
            "epoch_hex_color",
            "epoch_rgb",
        ],
    )


def _compute_correlation_y_limits(
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
]:
    """Compute shared y-limits for correlation previews.

    Returns
    -------
    Tuple of 4 optional y-limit tuples:
        - per_cell_max_limits: For max correlations (per-cell)
        - per_cell_min_limits: For min correlations (per-cell)
        - population_positive_limits: For positive correlations (population)
        - population_negative_limits: For negative correlations (population)

    """
    per_cell_max_values = []
    per_cell_min_values = []
    population_positive_values = []
    population_negative_values = []

    for group_data in [group1_data, group2_data]:
        if not group_data:
            continue
        correlation_df = group_data.get("correlation")
        if correlation_df is None or correlation_df.empty:
            continue

        # Per-cell max correlations (typically positive)
        max_cols = [
            col
            for col in correlation_df.columns
            if "max" in col.lower() and "correlation" in col.lower()
        ]
        for col in max_cols:
            per_cell_max_values.append(correlation_df[col].dropna())

        # Per-cell min correlations (typically negative)
        min_cols = [
            col
            for col in correlation_df.columns
            if "min" in col.lower() and "correlation" in col.lower()
        ]
        for col in min_cols:
            per_cell_min_values.append(correlation_df[col].dropna())

        # Population-level positive correlations
        positive_cols = [
            col
            for col in correlation_df.columns
            if "positive" in col.lower() and "correlation" in col.lower()
        ]
        for col in positive_cols:
            population_positive_values.append(correlation_df[col].dropna())

        # Population-level negative correlations
        negative_cols = [
            col
            for col in correlation_df.columns
            if "negative" in col.lower() and "correlation" in col.lower()
        ]
        for col in negative_cols:
            population_negative_values.append(correlation_df[col].dropna())

    def _range_from_series(
        series_list: List[pd.Series],
        label: str = "",
    ) -> Optional[Tuple[float, float]]:
        if not series_list:
            return None
        concatenated = pd.concat(series_list)
        if concatenated.empty:
            return None

        # Remove NaN values but keep zeros (they're valid correlation values)
        valid_data = concatenated.dropna()

        if valid_data.empty:
            logger.info(
                f"No valid {label} correlation values found - "
                "using auto-scaling for y-limits"
            )
            return None

        min_val = valid_data.min()
        max_val = valid_data.max()

        if pd.isna(min_val) or pd.isna(max_val):
            return None

        # For correlation data, ensure padding provides reasonable scale
        range_val = max_val - min_val

        if range_val < 0.01:
            # Very narrow range - use fixed padding
            padding = 0.1
        else:
            # Normal case - use 15% padding for better visibility
            # (increased from 10% to prevent cutoff)
            padding = 0.15 * range_val

        # Calculate limits with padding
        y_min = min_val - padding
        y_max = max_val + padding

        # Only clamp to [-1, 1] if we exceed bounds, but preserve all actual data
        # Check if actual data exceeds correlation bounds (shouldn't happen but just in case)
        if min_val < -1.0:
            logger.warning(
                f"Found {label} correlation values < -1.0 (min={min_val:.3f}). "
                "This may indicate data issues."
            )
            y_min = min(y_min, -1.0)
        else:
            y_min = max(y_min, -1.05)  # Allow slight overshoot for visibility

        if max_val > 1.0:
            logger.warning(
                f"Found {label} correlation values > 1.0 (max={max_val:.3f}). "
                "This may indicate data issues."
            )
            y_max = max(y_max, 1.0)
        else:
            y_max = min(y_max, 1.05)  # Allow slight overshoot for visibility

        logger.debug(
            f"Computed {label} y-limits: [{y_min:.3f}, {y_max:.3f}] "
            f"(data range: [{min_val:.3f}, {max_val:.3f}])"
        )

        return y_min, y_max

    per_cell_max_limits = _range_from_series(
        per_cell_max_values, label="per-cell max"
    )
    per_cell_min_limits = _range_from_series(
        per_cell_min_values, label="per-cell min"
    )
    population_positive_limits = _range_from_series(
        population_positive_values, label="population positive"
    )
    population_negative_limits = _range_from_series(
        population_negative_values, label="population negative"
    )

    # Log summary of computed shared y-limits
    logger.info(
        "Computed shared y-limits for correlation boxplots (across all groups):"
    )
    if per_cell_max_limits:
        logger.info(
            f"  Per-cell max correlations: "
            f"[{per_cell_max_limits[0]:.3f}, {per_cell_max_limits[1]:.3f}]"
        )
    if per_cell_min_limits:
        logger.info(
            f"  Per-cell min correlations: "
            f"[{per_cell_min_limits[0]:.3f}, {per_cell_min_limits[1]:.3f}]"
        )
    if population_positive_limits:
        logger.info(
            f"  Population positive correlations: "
            f"[{population_positive_limits[0]:.3f}, {population_positive_limits[1]:.3f}]"
        )
    if population_negative_limits:
        logger.info(
            f"  Population negative correlations: "
            f"[{population_negative_limits[0]:.3f}, {population_negative_limits[1]:.3f}]"
        )

    return (
        per_cell_max_limits,
        per_cell_min_limits,
        population_positive_limits,
        population_negative_limits,
    )


@dataclass(frozen=True)
class GroupPreviewMetric:
    """Configuration describing a per-group preview to generate."""

    dataset: str
    column: str
    suffix: str
    title: str
    data_type: str
    y_limit_key: Optional[str] = None
    stat_key: Optional[str] = None
    cdf_suffix: Optional[str] = None
    cdf_title: Optional[str] = None
    population_category: Optional[str] = None


GROUP_PREVIEW_METRICS: Tuple[GroupPreviewMetric, ...] = (
    GroupPreviewMetric(
        dataset="activity",
        column="mean_trace_activity",
        suffix="trace_activity_boxplot.svg",
        title="Mean Trace Activity",
        data_type="activity",
    ),
    GroupPreviewMetric(
        dataset="activity",
        column="mean_event_rate",
        suffix="event_activity_boxplot.svg",
        title="Mean Event Rate",
        data_type="event_rate",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="max_trace_correlation",
        suffix="max_trace_correlation_boxplot.svg",
        title="Max Trace Correlation (Per Cell)",
        data_type="correlation",
        y_limit_key="per_cell_max",
        stat_key="max",
        cdf_suffix="max_trace_correlation_cdf.svg",
        cdf_title="Max Trace Correlation",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="min_trace_correlation",
        suffix="min_trace_correlation_boxplot.svg",
        title="Min Trace Correlation (Per Cell)",
        data_type="correlation",
        y_limit_key="per_cell_min",
        stat_key="min",
        cdf_suffix="min_trace_correlation_cdf.svg",
        cdf_title="Min Trace Correlation",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="mean_trace_correlation",
        suffix="mean_trace_correlation_boxplot.svg",
        title="Mean Trace Correlation (Per Cell)",
        data_type="correlation",
        stat_key="mean",
        cdf_suffix="mean_trace_correlation_cdf.svg",
        cdf_title="Mean Trace Correlation",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="max_event_correlation",
        suffix="max_event_correlation_boxplot.svg",
        title="Max Event Correlation (Per Cell)",
        data_type="correlation",
        y_limit_key="per_cell_max",
        stat_key="max",
        cdf_suffix="max_event_correlation_cdf.svg",
        cdf_title="Max Event Correlation",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="min_event_correlation",
        suffix="min_event_correlation_boxplot.svg",
        title="Min Event Correlation (Per Cell)",
        data_type="correlation",
        y_limit_key="per_cell_min",
        stat_key="min",
        cdf_suffix="min_event_correlation_cdf.svg",
        cdf_title="Min Event Correlation",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="mean_event_correlation",
        suffix="mean_event_correlation_boxplot.svg",
        title="Mean Event Correlation (Per Cell)",
        data_type="correlation",
        stat_key="mean",
        cdf_suffix="mean_event_correlation_cdf.svg",
        cdf_title="Mean Event Correlation",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="positive_trace_correlation",
        suffix="positive_trace_population_boxplot.svg",
        title="Positive Trace Correlation (Population)",
        data_type="correlation",
        y_limit_key="population_positive",
        population_category="positive",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="negative_trace_correlation",
        suffix="negative_trace_population_boxplot.svg",
        title="Negative Trace Correlation (Population)",
        data_type="correlation",
        y_limit_key="population_negative",
        population_category="negative",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="positive_event_correlation",
        suffix="positive_event_population_boxplot.svg",
        title="Positive Event Correlation (Population)",
        data_type="correlation",
        y_limit_key="population_positive",
        population_category="positive",
    ),
    GroupPreviewMetric(
        dataset="correlation",
        column="negative_event_correlation",
        suffix="negative_event_population_boxplot.svg",
        title="Negative Event Correlation (Population)",
        data_type="correlation",
        y_limit_key="population_negative",
        population_category="negative",
    ),
)


def _generate_group_metric_previews(
    *,
    group_index: int,
    group_label: str,
    group_data: Optional[Dict[str, Any]],
    output_dir: str,
    dimension_column: str,
    dimension_color_map: Dict[str, str],
    dimension_filter_list: List[str],
    group_file_prefixes: Sequence[str],
    metrics: Sequence[GroupPreviewMetric],
    y_limit_lookup: Dict[str, Optional[Tuple[float, float]]],
    allowed_per_cell_stats: Set[str],
    include_positive_population: bool,
    include_negative_population: bool,
    dimension_suffix: Optional[str] = None,
    group_preview_prefixes: Optional[Sequence[str]] = None,
) -> None:
    """Generate boxplot/CDF previews for the provided group.
    
    Args:
        group_preview_prefixes: Optional list of prefixes for preview filenames.
            If provided, uses real group names for preview files (e.g., "Control", "Treatment").
            If None, falls back to group_file_prefixes.
    """
    if not group_data:
        return

    # Use real group names for preview filenames if provided
    preview_prefixes = group_preview_prefixes if group_preview_prefixes is not None else group_file_prefixes

    dataset_cache: Dict[str, Optional[pd.DataFrame]] = {}

    for spec in metrics:
        if not include_positive_population and spec.population_category == "positive":
            continue
        if not include_negative_population and spec.population_category == "negative":
            continue
        if spec.stat_key and spec.stat_key not in allowed_per_cell_stats:
            continue

        if spec.dataset not in dataset_cache:
            dataset_cache[spec.dataset] = group_data.get(spec.dataset)
        dataset_df = dataset_cache[spec.dataset]

        if dataset_df is None or getattr(dataset_df, "empty", True):
            continue
        if spec.column not in dataset_df.columns:
            continue

        filename = _group_output_filename(
            preview_prefixes,
            group_index,
            spec.suffix,
            dimension_suffix=dimension_suffix,
        )
        boxplot_path = os.path.join(output_dir, filename)

        success = _create_dimension_boxplot_preview(
            dimension_column=dimension_column,
            data_df=dataset_df,
            col_name=spec.column,
            group_name=group_label,
            identifier=spec.column,
            title_prefix=spec.title,
            filename=boxplot_path,
            state_color_map=dimension_color_map,
            filter_state_names=dimension_filter_list,
            data_type=spec.data_type,
            y_limits=y_limit_lookup.get(spec.y_limit_key),
        )
        if success:
            logger.info(
                "Generated %s %s preview: %s",
                group_label,
                spec.title,
                boxplot_path,
            )

        if not (spec.cdf_suffix and spec.cdf_title):
            continue

        cdf_filename = _group_output_filename(
            preview_prefixes,
            group_index,
            spec.cdf_suffix,
            dimension_suffix=dimension_suffix,
        )
        cdf_path = os.path.join(output_dir, cdf_filename)
        cdf_success = _create_dimension_cdf_preview(
            dimension_column=dimension_column,
            data_df=dataset_df,
            col_name=spec.column,
            group_name=group_label,
            identifier=spec.column,
            title_prefix=spec.cdf_title,
            filename=cdf_path,
            state_color_map=dimension_color_map,
            filter_state_names=dimension_filter_list,
            data_type=spec.data_type,
        )
        if cdf_success:
            logger.info(
                "Generated %s %s CDF preview: %s",
                group_label,
                spec.cdf_title,
                cdf_path,
            )


def _generate_modulation_histograms_for_group(
    *,
    group_index: int,
    group_label: str,
    group_data: Optional[Dict[str, Any]],
    output_dir: str,
    dimension_column: str,
    dimension_filter_list: List[str],
    dimension_label: str,
    group_file_prefixes: Sequence[str],
    modulation_colors: Optional[List[str]],
    dimension_suffix: Optional[str] = None,
    group_preview_prefixes: Optional[Sequence[str]] = None,
) -> None:
    """Generate modulation histograms for a single group when available.
    
    Args:
        group_preview_prefixes: Optional list of prefixes for preview filenames.
            If provided, uses real group names for preview files (e.g., "Control", "Treatment").
            If None, falls back to group_file_prefixes.
    """
    if (
        not group_data
        or "modulation" not in group_data
        or group_data["modulation"] is None
        or group_data["modulation"].empty
    ):
        return

    # Use real group names for preview filenames if provided
    preview_prefixes = group_preview_prefixes if group_preview_prefixes is not None else group_file_prefixes

    modulation_df = group_data["modulation"]
    palette = modulation_colors if modulation_colors else ["green", "blue", "black"]

    for score_column, suffix, data_type in (
        ("trace_modulation_scores", "trace_modulation_distribution.svg", "activity"),
        ("event_modulation_scores", "event_modulation_distribution.svg", "events"),
    ):
        if score_column not in modulation_df.columns:
            continue

        distribution_df = _prepare_modulation_distribution_data(
            modulation_df=modulation_df,
            score_column=score_column,
            dimension_column=dimension_column,
        )
        filename = _group_output_filename(
            preview_prefixes,
            group_index,
            suffix,
            dimension_suffix=dimension_suffix,
        )
        output_path = os.path.join(output_dir, filename)
        try:
            plot_modulation_distribution(
                modulation_scores=distribution_df,
                modulation_colors=palette,
                states=dimension_filter_list,
                output_filename=output_path,
                group_name=group_label,
                data_type=data_type,
                dimension_label=dimension_label,
            )
            logger.info(
                "Generated %s %s modulation histogram",
                group_label,
                data_type,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not generate %s %s modulation histogram: %s",
                group_label,
                data_type,
                exc,
            )


def _generate_per_group_previews(
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    group_names: List[str],
    states: List[str],
    epochs: List[str],
    output_dir: str,
    correlation_statistic: str,
    comparison_dimension: str,
    state_colors: Optional[List[str]] = None,
    epoch_colors: Optional[List[str]] = None,
    modulation_colors: Optional[List[str]] = None,
) -> None:
    """Generate per-group preview plots matching standard-python toolbox pattern.

    Creates boxplot preview files for each group, following the same naming
    convention as combine_compare_population_data and combine_compare_correlation_data.

    Parameters
    ----------
    group1_data : Dict[str, Any]
        Data dictionary for group 1
    group2_data : Optional[Dict[str, Any]]
        Data dictionary for group 2 (if present)
    group_names : List[str]
        Names of groups
    states : List[str]
        List of state names
    epochs : List[str]
        List of epoch names
    output_dir : str
        Output directory path
    correlation_statistic : str
        Correlation statistic to visualize: "max", "min", or "mean"
    comparison_dimension : str
        Dimension for comparison: "states" or "epochs"
    state_colors : Optional[List[str]]
        User-provided state colors list
    epoch_colors : Optional[List[str]]
        User-provided epoch colors list
    modulation_colors : Optional[List[str]]
        Validated list of three colors for modulation plots:
        [up-modulated, down-modulated, non-modulated]

    """
    # Resolve group names and generate filesystem-safe prefixes
    group_names = list(group_names) if group_names else ["Group 1"]
    if group2_data is not None and len(group_names) < 2:
        group_names.append("Group 2")

    group_file_prefixes = _resolve_group_file_prefixes(group_names)  # ["group1", "group2"] for files/dirs
    
    # Create sanitized group name prefixes for human-readable preview filenames
    group_preview_prefixes = [
        _sanitize_filename_component(name, f"group{i+1}")
        for i, name in enumerate(group_names)
    ]

    # Resolve color maps based on comparison dimension
    if comparison_dimension == "states":
        dimension_color_map = _resolve_state_color_map(
            states=states,
            group1_data=group1_data,
            group2_data=group2_data,
            user_state_colors=state_colors,
        )
        dimension_filter_list = states
        dimension_column = "state"
    else:  # epochs
        dimension_color_map = _resolve_epoch_color_map(
            epochs=epochs,
            group1_data=group1_data,
            group2_data=group2_data,
            user_epoch_colors=epoch_colors,
        )
        dimension_filter_list = epochs
        dimension_column = "epoch"

    dimension_label = "State" if dimension_column == "state" else "Epoch"
    dimension_suffix = (comparison_dimension or "").strip().lower() or None

    (
        per_cell_max_limits,
        per_cell_min_limits,
        population_positive_limits,
        population_negative_limits,
    ) = _compute_correlation_y_limits(
        group1_data=group1_data,
        group2_data=group2_data,
    )

    correlation_statistic = str(correlation_statistic).lower().strip()

    allowed_per_cell_stats = {correlation_statistic}

    include_positive_population = True
    include_negative_population = True

    y_limit_lookup = {
        "per_cell_max": per_cell_max_limits,
        "per_cell_min": per_cell_min_limits,
        "population_positive": population_positive_limits,
        "population_negative": population_negative_limits,
    }

    try:
        group1_label = group_names[0]
        _generate_group_metric_previews(
            group_index=0,
            group_label=group1_label,
            group_data=group1_data,
            output_dir=output_dir,
            dimension_column=dimension_column,
            dimension_color_map=dimension_color_map,
            dimension_filter_list=dimension_filter_list,
            group_file_prefixes=group_file_prefixes,
            metrics=GROUP_PREVIEW_METRICS,
            y_limit_lookup=y_limit_lookup,
            allowed_per_cell_stats=allowed_per_cell_stats,
            include_positive_population=include_positive_population,
            include_negative_population=include_negative_population,
            dimension_suffix=dimension_suffix,
            group_preview_prefixes=group_preview_prefixes,  # Use real group names for previews
        )

        if group2_data is not None and len(group_names) > 1:
            group2_label = group_names[1]
            _generate_group_metric_previews(
                group_index=1,
                group_label=group2_label,
                group_data=group2_data,
                output_dir=output_dir,
                dimension_column=dimension_column,
                dimension_color_map=dimension_color_map,
                dimension_filter_list=dimension_filter_list,
                group_file_prefixes=group_file_prefixes,
                metrics=GROUP_PREVIEW_METRICS,
                y_limit_lookup=y_limit_lookup,
                allowed_per_cell_stats=allowed_per_cell_stats,
                include_positive_population=include_positive_population,
                include_negative_population=include_negative_population,
                dimension_suffix=dimension_suffix,
                group_preview_prefixes=group_preview_prefixes,  # Use real group names for previews
            )

        _generate_modulation_histograms_for_group(
            group_index=0,
            group_label=group1_label,
            group_data=group1_data,
            output_dir=output_dir,
            dimension_column=dimension_column,
            dimension_filter_list=dimension_filter_list,
            dimension_label=dimension_label,
            group_file_prefixes=group_file_prefixes,
            modulation_colors=modulation_colors,
            dimension_suffix=dimension_suffix,
            group_preview_prefixes=group_preview_prefixes,  # Use real group names for previews
        )

        if group2_data is not None and len(group_names) > 1:
            _generate_modulation_histograms_for_group(
                group_index=1,
                group_label=group_names[1],
                group_data=group2_data,
                output_dir=output_dir,
                dimension_column=dimension_column,
                dimension_filter_list=dimension_filter_list,
                dimension_label=dimension_label,
                group_file_prefixes=group_file_prefixes,
                modulation_colors=modulation_colors,
                dimension_suffix=dimension_suffix,
                group_preview_prefixes=group_preview_prefixes,  # Use real group names for previews
            )

        logger.info("Completed per-group preview generation")

    except Exception as e:
        logger.warning(f"Could not generate all per-group previews: {e}")


def _prepare_dimension_plot_dataframe(
    data_df: Optional[pd.DataFrame],
    dimension_column: str,
) -> Optional[pd.DataFrame]:
    """Return dataframe with plotting dimension exposed through the 'state' column."""
    if data_df is None or data_df.empty:
        return data_df

    normalized_column = (dimension_column or "state").strip()
    if not normalized_column or normalized_column == "state":
        return data_df

    if normalized_column not in data_df.columns:
        logger.warning(
            "Dimension column '%s' not found in dataframe; defaulting to 'state'",
            normalized_column,
        )
        return data_df

    plot_df = data_df.copy()
    plot_df["state"] = plot_df[normalized_column]
    return plot_df


def _create_dimension_boxplot_preview(
    *,
    data_df: Optional[pd.DataFrame],
    dimension_column: str,
    **kwargs: Any,
) -> bool:
    """Support epoch comparisons for boxplot previews."""
    plot_df = _prepare_dimension_plot_dataframe(data_df, dimension_column)
    dimension_label = (
        "State"
        if (dimension_column or "state").strip() == "state"
        else "Epoch"
    )
    return _base_create_boxplot_preview(
        data_df=plot_df,
        dimension_label=dimension_label,
        **kwargs,
    )


def _create_dimension_cdf_preview(
    *,
    data_df: Optional[pd.DataFrame],
    dimension_column: str,
    **kwargs: Any,
) -> bool:
    """Support epoch comparisons for CDF previews."""
    plot_df = _prepare_dimension_plot_dataframe(data_df, dimension_column)
    dimension_label = (
        "State"
        if (dimension_column or "state").strip() == "state"
        else "Epoch"
    )
    return _base_create_cdf_preview(
        data_df=plot_df,
        dimension_label=dimension_label,
        **kwargs,
    )


# Expose wrappers under original helper names for existing call sites
create_boxplot_preview = _create_dimension_boxplot_preview
create_cdf_preview = _create_dimension_cdf_preview


def _prepare_modulation_distribution_data(
    modulation_df: pd.DataFrame,
    score_column: str,
    dimension_column: str,
    group_mode: str = "dimension",
) -> pd.DataFrame:
    """Aggregate modulation scores into arrays for histogram plotting.

    Parameters
    ----------
    modulation_df : pd.DataFrame
        Modulation dataframe containing score columns and dimension metadata.
    score_column : str
        Column containing the modulation score values to aggregate.
    dimension_column : str
        Column indicating the primary comparison dimension ("state" or "epoch").
    group_mode : str, optional
        Grouping mode for aggregation. Supported values:
        - "dimension" (default): aggregate across the provided ``dimension_column``.
        - "state_epoch": aggregate per unique state-epoch combination.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "state": Display label for the group (state, epoch, or state-epoch).
        - "modulation": List containing numpy arrays of modulation scores.
        - Optional metadata columns copied from the source dataframe.

    """
    if modulation_df is None or modulation_df.empty:
        return pd.DataFrame(columns=["state", "modulation"])

    if score_column not in modulation_df.columns:
        logger.info(
            "Skipping modulation histogram: column '%s' not found in modulation data",
            score_column,
        )
        return pd.DataFrame(columns=["state", "modulation"])

    grouping_column = str(dimension_column or "").strip()
    if not grouping_column:
        grouping_column = "state"

    if grouping_column not in modulation_df.columns:
        logger.warning(
            "Cannot group modulation scores by '%s' - column not found in modulation data. "
            "Skipping modulation histogram generation.",
            grouping_column,
        )
        return pd.DataFrame(columns=["state", "modulation"])

    required_columns = [grouping_column, score_column]
    filtered_df = modulation_df.dropna(subset=required_columns)

    if filtered_df.empty:
        logger.warning(
            "No valid modulation scores available after filtering column '%s'",
            score_column,
        )
        return pd.DataFrame(columns=["state", "modulation"])

    grouping_mode = str(group_mode or "dimension").strip().lower()
    supported_modes = {"dimension", "state_epoch"}
    if grouping_mode not in supported_modes:
        logger.warning(
            "Unsupported group_mode '%s'; defaulting to 'dimension'",
            group_mode,
        )
        grouping_mode = "dimension"

    working_df = filtered_df.copy()

    def _normalize_label(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    use_state_epoch = grouping_mode == "state_epoch"
    if use_state_epoch:
        missing_columns = {"state", "epoch"} - set(working_df.columns)
        if missing_columns:
            logger.warning(
                "Cannot group modulation scores by state-epoch combination; "
                "missing columns: %s. Falling back to '%s' grouping.",
                sorted(missing_columns),
                dimension_column,
            )
            use_state_epoch = False

    if use_state_epoch:
        state_labels = working_df["state"].apply(_normalize_label)
        epoch_labels = working_df["epoch"].apply(_normalize_label)

        def _compose_label(state_label: str, epoch_label: str) -> str:
            if state_label and epoch_label:
                return f"{state_label} :: {epoch_label}"
            if state_label:
                return state_label
            if epoch_label:
                return epoch_label
            return "Unknown"

        working_df["_group_label"] = [
            _compose_label(state_label, epoch_label)
            for state_label, epoch_label in zip(state_labels, epoch_labels)
        ]
    else:
        working_df["_group_label"] = working_df[grouping_column].apply(
            _normalize_label
        )

    working_df = working_df[working_df["_group_label"] != ""]
    if working_df.empty:
        logger.warning(
            "No valid modulation scores available after resolving group labels."
        )
        return pd.DataFrame(columns=["state", "modulation"])

    grouped_scores = (
        working_df.groupby("_group_label")[score_column]
        .apply(lambda values: [values.to_numpy()])
        .reset_index()
        .rename(
            columns={
                "_group_label": "state",
                score_column: "modulation",
            }
        )
    )

    def _select_metadata_value(
        label: str,
        values: pd.Series,
        metadata_column: str,
    ) -> Any:
        representative: Any = None
        normalized_seen: Set[Any] = set()
        conflicts: List[Any] = []

        for raw_value in values:
            if isinstance(raw_value, str):
                stripped_value = raw_value.strip()
                normalized_value = stripped_value.casefold()
                display_value: Any = stripped_value
            else:
                stripped_value = raw_value
                normalized_value = stripped_value
                display_value = stripped_value

            if pd.isna(stripped_value):
                continue
            if isinstance(stripped_value, str) and not stripped_value:
                continue

            if representative is None:
                representative = display_value
                normalized_seen.add(normalized_value)
                continue

            if normalized_value in normalized_seen:
                continue

            normalized_seen.add(normalized_value)
            conflicts.append(display_value)

        if conflicts:
            conflict_list = (
                [representative] + conflicts
                if representative is not None
                else conflicts
            )
            logger.warning(
                "Conflicting '%s' values found for modulation grouping '%s': %s. "
                "Using '%s'.",
                metadata_column,
                label,
                conflict_list,
                representative,
            )

        return representative

    metadata_columns = [
        "state_comparison_type",
        "baseline_state",
        "baseline_epoch",
    ]

    for metadata_column in metadata_columns:
        if metadata_column not in working_df.columns:
            continue

        metadata_subset = working_df[["_group_label", metadata_column]].dropna(
            subset=["_group_label", metadata_column]
        )
        if metadata_subset.empty:
            continue

        resolved_rows: List[Dict[str, Any]] = []
        for label, column_values in metadata_subset.groupby("_group_label"):
            resolved_value = _select_metadata_value(
                label, column_values[metadata_column], metadata_column
            )
            if resolved_value is None or (
                isinstance(resolved_value, str) and not resolved_value
            ):
                continue
            resolved_rows.append(
                {"state": label, metadata_column: resolved_value}
            )

        if not resolved_rows:
            continue

        metadata_mapping = pd.DataFrame(resolved_rows)

        grouped_scores = grouped_scores.merge(
            metadata_mapping,
            on="state",
            how="left",
        )

    if use_state_epoch:
        label_mapping = (
            working_df[["_group_label", "state", "epoch"]]
            .dropna(subset=["_group_label"])
            .drop_duplicates(subset=["_group_label"])
            .rename(
                columns={
                    "_group_label": "state",
                    "state": "original_state",
                    "epoch": "original_epoch",
                }
            )
        )

        grouped_scores = grouped_scores.merge(
            label_mapping,
            on="state",
            how="left",
        )

    return grouped_scores


def _save_output_metadata(
    group1_data: Dict[str, Any],
    group2_data: Optional[Dict[str, Any]],
    group_names: List[str],
    group_colors: List[str],
    states: List[str],
    epochs: List[str],
    baseline_state: str,
    baseline_epoch: str,
    comparison_dimension: str,
    data_pairing: str,
    correlation_statistic: str,
    significance_threshold: Optional[float],
    multiple_correction: str,
    effect_size: str,
    group_comparison_type: str,
    parametric: Union[str, bool],
    output_dir: str,
    measure_source: str = "trace",
    selected_measures: Optional[Dict[str, List[str]]] = None,
    selected_sources: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Save comprehensive output metadata JSON file.

    Args:
    ----
    group1_data : Dict[str, Any]
        Combined data for group 1
    group2_data : Optional[Dict[str, Any]]
        Combined data for group 2 (optional)
    group_names : List[str]
        List of group names
    group_colors : List[str]
        List of group colors
    states : List[str]
        List of state names
    epochs : List[str]
        List of epoch names
    baseline_state : str
        Baseline state name
    baseline_epoch : str
        Baseline epoch name
    comparison_dimension : str
        Dimension being compared (states or epochs)
    data_pairing : str
        Type of statistical pairing (paired or unpaired)
    correlation_statistic : str
        Correlation statistic used
    significance_threshold : Optional[float]
        Significance threshold for reclassification
    multiple_correction : str
        Multiple comparison correction method
    effect_size : str
        Effect size measure used
    group_comparison_type : str
        Type of group comparison being performed
    parametric : Union[str, bool]
        Whether parametric statistical tests are used
    output_dir : str
        Output directory path
    measure_source : str
        Data source selection used for analysis ("trace", "event", or "both")
    selected_measures : Optional[Dict[str, List[str]]]
        Mapping of measure types to analyzed column names
    selected_sources : Optional[Dict[str, List[str]]]
        Mapping of measure types to resolved data sources (trace/event)

    """
    if selected_measures is None:
        selected_measures = {
            "activity": [],
            "correlation": [],
            "modulation": [],
        }
    if selected_sources is None:
        selected_sources = {
            "activity": [],
            "correlation": [],
            "modulation": [],
        }

    normalized_measure_source = measure_source.lower().strip()

    normalized_selected_measures = {
        key: sorted(selected_measures.get(key, []))
        for key in ["activity", "correlation", "modulation"]
    }
    normalized_selected_sources = {
        key: sorted(selected_sources.get(key, []))
        for key in ["activity", "correlation", "modulation"]
    }

    normalized_group_comparison_type = (
        str(group_comparison_type).strip()
        if group_comparison_type is not None
        else ""
    )
    normalized_parametric = _normalize_parametric_value(parametric)

    def _count_unique_entries(
        df: pd.DataFrame, candidate_columns: List[str]
    ) -> int:
        if df is None or df.empty:
            return 0
        for column in candidate_columns:
            if column in df.columns:
                unique_count = df[column].dropna().nunique()
                if unique_count:
                    return unique_count
        return 0

    activity_df_g1 = group1_data.get("activity", pd.DataFrame())
    num_cells_g1 = _count_unique_entries(
        activity_df_g1, ["name", "cell_name", "cell_id"]
    )
    num_subjects_g1 = _count_unique_entries(
        activity_df_g1, ["normalized_subject_id", "subject_id"]
    )

    num_cells_g2 = 0
    num_subjects_g2 = 0
    if group2_data is not None:
        activity_df_g2 = group2_data.get("activity", pd.DataFrame())
        num_cells_g2 = _count_unique_entries(
            activity_df_g2, ["name", "cell_name", "cell_id"]
        )
        num_subjects_g2 = _count_unique_entries(
            activity_df_g2, ["normalized_subject_id", "subject_id"]
        )

    group_file_prefixes = _resolve_group_file_prefixes(group_names)
    normalized_dimension_suffix = (
        (comparison_dimension or "").strip().lower() or None
    )

    def _group_filename(idx: int, suffix: str) -> str:
        return _group_output_filename(
            group_file_prefixes,
            idx,
            suffix,
            dimension_suffix=None,  # Group files don't use dimension suffix
        )

    # Create base values for all output files
    base_values = {
        "num_states": len(states),
        "num_epochs": len(epochs),
        "num_combinations": len(states) * len(epochs),
        "states": states,
        "epochs": epochs,
        "baseline_state": baseline_state,
        "baseline_epoch": baseline_epoch,
        "comparison_dimension": comparison_dimension,
        "data_pairing": data_pairing,
        "correlation_statistic": correlation_statistic,
        "multiple_correction": multiple_correction,
        "effect_size": effect_size,
        "num_groups": len(group_names),
        "group_names": group_names,
        "group_colors": group_colors,
        "measure_source": normalized_measure_source,
        "selected_measures": normalized_selected_measures,
        "selected_measure_sources": normalized_selected_sources,
        "group_comparison_type": normalized_group_comparison_type,
        "parametric": normalized_parametric,
    }

    # Comprehensive metadata combining file descriptions and analysis details
    # Use Path().stem to extract keys from constants
    group1_activity_filename = _group_filename(0, GROUP_COMBINED_ACTIVITY_SUFFIX)
    group1_correlation_filename = _group_filename(
        0, GROUP_COMBINED_TRACE_CORRELATION_SUFFIX
    )
    group1_modulation_filename = _group_filename(
        0, GROUP_COMBINED_MODULATION_SUFFIX
    )

    metadata = {
        # Group 1 combined data files
        Path(group1_activity_filename).stem: {
            **base_values,
            "file_type": "combined_activity_data",
            "description": f"Combined mean activity data for {group_names[0]}",
            "analysis_type": "group_comparison",
            "num_cells": num_cells_g1,
            "num_subjects": num_subjects_g1,
        },
        Path(group1_correlation_filename).stem: {
            **base_values,
            "file_type": "combined_correlation_data",
            "description": f"Combined correlation statistics for {group_names[0]}",
            "analysis_type": "group_comparison",
            "num_cells": num_cells_g1,
            "num_subjects": num_subjects_g1,
        },
        Path(group1_modulation_filename).stem: {
            **base_values,
            "file_type": "combined_modulation_data",
            "description": f"Combined modulation data for {group_names[0]}",
            "analysis_type": "group_comparison",
            "num_cells": num_cells_g1,
            "num_subjects": num_subjects_g1,
        },
    }

    def _output_file_exists(filename: str) -> bool:
        target_path = (
            Path(output_dir) / filename if output_dir else Path(filename)
        )
        return target_path.exists()

    # Statistical comparison files (trace-specific)
    metadata[Path(TRACE_ANOVA_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Trace ANOVA results comparing groups across state-epoch combinations",
        "analysis_type": "statistical_test",
        "test_type": ("mixed_anova" if data_pairing == "paired" else "anova"),
        "file_exists": _output_file_exists(TRACE_ANOVA_COMPARISONS_CSV),
    }

    metadata[Path(TRACE_PAIRWISE_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Trace pairwise comparisons between groups",
        "analysis_type": "statistical_test",
        "test_type": "pairwise_t_test",
        "file_exists": _output_file_exists(TRACE_PAIRWISE_COMPARISONS_CSV),
    }

    # Statistical comparison files (event-specific)
    metadata[Path(EVENT_ANOVA_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Event ANOVA results comparing groups across state-epoch combinations",
        "analysis_type": "statistical_test",
        "test_type": ("mixed_anova" if data_pairing == "paired" else "anova"),
        "file_exists": _output_file_exists(EVENT_ANOVA_COMPARISONS_CSV),
    }

    metadata[Path(EVENT_PAIRWISE_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Event pairwise comparisons between groups",
        "analysis_type": "statistical_test",
        "test_type": "pairwise_t_test",
        "file_exists": _output_file_exists(EVENT_PAIRWISE_COMPARISONS_CSV),
    }

    metadata[Path(TRACE_LMM_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Trace LMM fixed effects comparing groups across state-epoch combinations",
        "analysis_type": "statistical_test",
        "test_type": "linear_mixed_model",
        "file_exists": _output_file_exists(TRACE_LMM_COMPARISONS_CSV),
    }

    metadata[Path(TRACE_LMM_PAIRWISE_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Trace LMM-derived pairwise comparisons between groups",
        "analysis_type": "statistical_test",
        "test_type": "lmm_pairwise",
        "file_exists": _output_file_exists(TRACE_LMM_PAIRWISE_COMPARISONS_CSV),
    }

    metadata[Path(EVENT_LMM_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Event LMM fixed effects comparing groups across state-epoch combinations",
        "analysis_type": "statistical_test",
        "test_type": "linear_mixed_model",
        "file_exists": _output_file_exists(EVENT_LMM_COMPARISONS_CSV),
    }

    metadata[Path(EVENT_LMM_PAIRWISE_COMPARISONS_CSV).stem] = {
        **base_values,
        "file_type": "statistical_comparison",
        "description": "Event LMM-derived pairwise comparisons between groups",
        "analysis_type": "statistical_test",
        "test_type": "lmm_pairwise",
        "file_exists": _output_file_exists(EVENT_LMM_PAIRWISE_COMPARISONS_CSV),
    }

    # Add group 2 metadata if present
    if group2_data is not None:
        group2_activity_filename = _group_filename(
            1, GROUP_COMBINED_ACTIVITY_SUFFIX
        )
        group2_correlation_filename = _group_filename(
            1, GROUP_COMBINED_TRACE_CORRELATION_SUFFIX
        )
        group2_modulation_filename = _group_filename(
            1, GROUP_COMBINED_MODULATION_SUFFIX
        )
        metadata[Path(group2_activity_filename).stem] = {
            **base_values,
            "file_type": "combined_activity_data",
            "description": f"Combined mean activity data for {group_names[1]}",
            "analysis_type": "group_comparison",
            "num_cells": num_cells_g2,
            "num_subjects": num_subjects_g2,
        }
        metadata[Path(group2_correlation_filename).stem] = {
            **base_values,
            "file_type": "combined_correlation_data",
            "description": f"Combined correlation statistics for {group_names[1]}",
            "analysis_type": "group_comparison",
            "num_cells": num_cells_g2,
            "num_subjects": num_subjects_g2,
        }
        metadata[Path(group2_modulation_filename).stem] = {
            **base_values,
            "file_type": "combined_modulation_data",
            "description": f"Combined modulation data for {group_names[1]}",
            "analysis_type": "group_comparison",
            "num_cells": num_cells_g2,
            "num_subjects": num_subjects_g2,
        }

    # Add significance threshold if reclassification was performed
    if significance_threshold is not None:
        for key in metadata:
            if "combined_" in key or "comparison" in key:
                metadata[key][
                    "significance_threshold"
                ] = significance_threshold

    # Save comprehensive metadata
    output_path = os.path.join(output_dir, "output_metadata.json")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved comprehensive output metadata to {output_path}")


MANUAL_PREVIEW_MAP: Dict[str, List[Tuple[str, str]]] = {}


def _load_output_metadata_dict(output_dir: str) -> Dict[str, Any]:
    """Load and remove the temporary output_metadata.json file."""
    target_dir = output_dir if output_dir else "."
    metadata_path = os.path.join(target_dir, "output_metadata.json")
    if not os.path.exists(metadata_path):
        logger.warning("Output metadata file not found at %s", metadata_path)
        return {}

    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    try:
        os.remove(metadata_path)
    except OSError:
        logger.warning("Unable to remove metadata file at %s", metadata_path)
    return metadata


def _sanitize_filename_component(
    raw_value: Optional[str],
    fallback: str,
) -> str:
    """Convert a raw string value into a filesystem-safe filename component.
    
    Replaces all non-alphanumeric characters with underscores and strips
    leading/trailing underscores. Returns the fallback if the result is empty.
    
    Args:
        raw_value: The raw string to sanitize (e.g., "Group 1: Control")
        fallback: The fallback value if sanitization produces an empty string
        
    Returns:
        A filesystem-safe string (e.g., "Group_1_Control")
        
    Examples:
        >>> _sanitize_filename_component("Wild-Type Mice", "Group1")
        'Wild_Type_Mice'
        >>> _sanitize_filename_component("", "Group1")
        'Group1'
        >>> _sanitize_filename_component(None, "Group1")
        'Group1'
    """
    if raw_value is None:
        return fallback
    clean = re.sub(r"[^A-Za-z0-9]+", "_", str(raw_value)).strip("_")
    return clean or fallback


def _resolve_group_file_prefixes(group_names: Sequence[str]) -> List[str]:
    """Generate unique, filesystem-safe filename prefixes for each group.
    
    Always uses generic group1, group2, etc. prefixes for consistency with
    tool spec patterns and output registration. Group names are used for
    display purposes in captions and metadata, not in file paths.
    
    Args:
        group_names: Sequence of group display names (used only to determine count)
        
    Returns:
        List of generic prefixes: ["group1", "group2", ...]
        
    Examples:
        >>> _resolve_group_file_prefixes(["Control", "Treatment"])
        ['group1', 'group2']
        >>> _resolve_group_file_prefixes(["Wild-Type"])
        ['group1']
        >>> _resolve_group_file_prefixes([])
        ['group1']
    """
    num_groups = max(len(group_names) if group_names else 1, 1)
    return [f"group{idx}" for idx in range(1, num_groups + 1)]


def _group_output_filename(
    group_prefixes: Sequence[str],
    group_index: int,
    suffix: str,
    dimension_suffix: Optional[str] = None,
) -> str:
    """Construct the full filename for a group's output file.
    
    Args:
        group_prefixes: Pre-computed sanitized prefixes for each group
        group_index: Zero-based index of the group (0 for Group 1, 1 for Group 2)
        suffix: Filename suffix (e.g., "combined_activity_data.csv")
        dimension_suffix: Optional label ("states", "epochs") inserted before the suffix
        
    Returns:
        Complete filename like "Control_combined_activity_data.csv"
        Falls back to "Group_N_suffix" if index is out of bounds.
        
    Examples:
        >>> prefixes = ["Control", "Treatment"]
        >>> _group_output_filename(prefixes, 0, "combined_activity_data.csv")
        'Control_combined_activity_data.csv'
        >>> _group_output_filename(prefixes, 2, "test.csv")  # Out of bounds
        'Group_3_test.csv'
    """
    if group_index < len(group_prefixes):
        prefix = group_prefixes[group_index]
    else:
        prefix = f"Group_{group_index + 1}"
        
    filename = prefix

    if dimension_suffix:
        clean_dimension = _sanitize_filename_component(
            dimension_suffix,
            dimension_suffix,
        )
        if clean_dimension:
            filename = f"{filename}_{clean_dimension}"

    if suffix:
        filename = f"{filename}_{suffix}"

    return filename


def _derive_output_file_basename(
    group_names: List[str],
    comparison_dimension: str,
) -> str:
    """Return a deterministic basename for relocating outputs with previews."""

    if not group_names:
        group_names = ["Group_1"]

    sanitized_names: List[str] = []
    for idx, raw_name in enumerate(group_names, start=1):
        fallback = f"Group_{idx}"
        sanitized_names.append(
            _sanitize_filename_component(raw_name, fallback)
        )

    if len(sanitized_names) == 1:
        base = sanitized_names[0]
    else:
        base = "_vs_".join(sanitized_names)

    dimension_suffix = comparison_dimension.strip().lower()
    if dimension_suffix:
        base = f"{base}_{dimension_suffix}"
    return base


def _build_group_activity_previews(
    group_index: int,
    group_label: str,
    group_file_prefixes: Sequence[str],
    dimension_suffix: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return activity preview definitions for a given group."""
    label = group_label or f"Group {group_index}"
    zero_based_index = max(group_index - 1, 0)
    trace_activity_filename = _group_output_filename(
        group_file_prefixes,
        zero_based_index,
        "trace_activity_boxplot.svg",
        dimension_suffix=dimension_suffix,
    )
    event_activity_filename = _group_output_filename(
        group_file_prefixes,
        zero_based_index,
        "event_activity_boxplot.svg",
        dimension_suffix=dimension_suffix,
    )
    return [
        (
            trace_activity_filename,
            f"{label} trace activity distributions across state-epoch combinations.",
        ),
        (
            event_activity_filename,
            f"{label} event rate distributions across state-epoch combinations.",
        ),
    ]


def _build_group_correlation_previews(
    group_index: int,
    group_label: str,
    group_file_prefixes: Sequence[str],
    dimension_suffix: Optional[str] = None,
    allowed_stats: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    """Return correlation preview definitions for a given group."""
    label = group_label or f"Group {group_index}"
    zero_based_index = max(group_index - 1, 0)
    stat_map = {
        "max": "maximum",
        "min": "minimum",
        "mean": "mean",
    }
    data_sources = {
        "trace": "trace",
        "event": "event",
    }

    normalized_allowed_stats: Set[str]
    if allowed_stats:
        normalized_allowed_stats = {
            str(stat).strip().lower() for stat in allowed_stats if stat
        }
        normalized_allowed_stats &= set(stat_map.keys())
        if not normalized_allowed_stats:
            normalized_allowed_stats = set(stat_map.keys())
    else:
        normalized_allowed_stats = set(stat_map.keys())

    def _per_cell_filename(stat: str, trace_or_event: str, suffix: str) -> str:
        return _group_output_filename(
            group_file_prefixes,
            zero_based_index,
            f"{stat}_{trace_or_event}_correlation_{suffix}.svg",
            dimension_suffix=dimension_suffix,
        )

    previews: List[Tuple[str, str]] = []
    for stat_key, stat_text in stat_map.items():
        if stat_key not in normalized_allowed_stats:
            continue
        for source_key, source_text in data_sources.items():
            previews.append(
                (
                    _per_cell_filename(stat_key, source_key, "boxplot"),
                    f"{label} {source_text} {stat_text} correlation boxplots.",
                )
            )
            previews.append(
                (
                    _per_cell_filename(stat_key, source_key, "cdf"),
                    f"{label} {source_text} {stat_text} correlation ECDFs.",
                )
            )

    for direction_key, direction_text in {
        "positive": "positive",
        "negative": "negative",
    }.items():
        for source_key, source_text in data_sources.items():
            previews.append(
                (
                    _group_output_filename(
                        group_file_prefixes,
                        zero_based_index,
                        f"{direction_key}_{source_key}_population_boxplot.svg",
                        dimension_suffix=dimension_suffix,
                    ),
                    f"{label} {source_text} {direction_text} population correlations.",
                )
            )

    return previews


def _build_group_modulation_previews(
    group_index: int,
    group_label: str,
    group_file_prefixes: Sequence[str],
    dimension_suffix: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return modulation preview definitions for a given group."""
    label = group_label or f"Group {group_index}"
    zero_based_index = max(group_index - 1, 0)
    return [
        (
            _group_output_filename(
                group_file_prefixes,
                zero_based_index,
                "trace_modulation_distribution.svg",
                dimension_suffix=dimension_suffix,
            ),
            f"{label} trace modulation classifications relative to baseline.",
        ),
        (
            _group_output_filename(
                group_file_prefixes,
                zero_based_index,
                "event_modulation_distribution.svg",
                dimension_suffix=dimension_suffix,
            ),
            f"{label} event modulation classifications relative to baseline.",
        ),
    ]


def _build_trace_comparison_previews(
    comparison_dimension: str,
) -> List[Tuple[str, str]]:
    """Return comparison preview definitions for trace-level statistics."""
    if comparison_dimension == "states":
        return [
            (
                STATES_COMPARISON_TRACE_ACTIVITY_SVG,
                "Trace activity comparison across behavioral states.",
            ),
            (
                STATES_COMPARISON_TRACE_CORRELATION_SVG,
                "Trace correlation summary across behavioral states.",
            ),
            (
                STATES_COMPARISON_TRACE_POSITIVE_CORRELATION_SVG,
                "Positive trace correlations compared across states.",
            ),
            (
                STATES_COMPARISON_TRACE_NEGATIVE_CORRELATION_SVG,
                "Negative trace correlations compared across states.",
            ),
            (
                STATES_COMPARISON_TRACE_MODULATION_SVG,
                "Trace modulation prevalence across states.",
            ),
        ]

    return [
        (
            EPOCHS_COMPARISON_TRACE_ACTIVITY_SVG,
            "Trace activity comparison across task epochs.",
        ),
        (
            EPOCHS_COMPARISON_TRACE_CORRELATION_SVG,
            "Trace correlation summary across task epochs.",
        ),
        (
            EPOCHS_COMPARISON_TRACE_POSITIVE_CORRELATION_SVG,
            "Positive trace correlations compared across epochs.",
        ),
        (
            EPOCHS_COMPARISON_TRACE_NEGATIVE_CORRELATION_SVG,
            "Negative trace correlations compared across epochs.",
        ),
        (
            EPOCHS_COMPARISON_TRACE_MODULATION_SVG,
            "Trace modulation prevalence across epochs.",
        ),
    ]


def _build_event_comparison_previews(
    comparison_dimension: str,
) -> List[Tuple[str, str]]:
    """Return comparison preview definitions for event-level statistics."""
    if comparison_dimension == "states":
        return [
            (
                STATES_COMPARISON_EVENT_ACTIVITY_SVG,
                "Event rate comparison across behavioral states.",
            ),
            (
                STATES_COMPARISON_EVENT_CORRELATION_SVG,
                "Event correlation summary across behavioral states.",
            ),
            (
                STATES_COMPARISON_EVENT_POSITIVE_CORRELATION_SVG,
                "Positive event correlations compared across states.",
            ),
            (
                STATES_COMPARISON_EVENT_NEGATIVE_CORRELATION_SVG,
                "Negative event correlations compared across states.",
            ),
            (
                STATES_COMPARISON_EVENT_MODULATION_SVG,
                "Event modulation prevalence across states.",
            ),
        ]

    return [
        (
            EPOCHS_COMPARISON_EVENT_ACTIVITY_SVG,
            "Event rate comparison across task epochs.",
        ),
        (
            EPOCHS_COMPARISON_EVENT_CORRELATION_SVG,
            "Event correlation summary across task epochs.",
        ),
        (
            EPOCHS_COMPARISON_EVENT_POSITIVE_CORRELATION_SVG,
            "Positive event correlations compared across epochs.",
        ),
        (
            EPOCHS_COMPARISON_EVENT_NEGATIVE_CORRELATION_SVG,
            "Negative event correlations compared across epochs.",
        ),
        (
            EPOCHS_COMPARISON_EVENT_MODULATION_SVG,
            "Event modulation prevalence across epochs.",
        ),
    ]


def _register_combined_tool_outputs(
    *,
    output_dir: str,
    group_names: List[str],
    comparison_dimension: str,
    correlation_statistic: Optional[str] = None,
) -> None:
    """Register output files and previews using the shared state-epoch utilities."""

    # Normalize empty output_dir to current directory for file operations
    registration_dir = output_dir if output_dir else "."
    output_metadata = _load_output_metadata_dict(output_dir)
    output_file_basename = _derive_output_file_basename(
        group_names,
        comparison_dimension,
    )

    resolved_group_names = group_names or ["Group 1"]
    group_file_prefixes = _resolve_group_file_prefixes(resolved_group_names)  # ["group1", "group2"] for files/dirs
    
    # Create sanitized group name prefixes for human-readable preview filenames
    group_preview_prefixes = [
        _sanitize_filename_component(name, f"group{i+1}")
        for i, name in enumerate(resolved_group_names)
    ]
    
    normalized_dimension_suffix = (
        (comparison_dimension or "").strip().lower() or None
    )

    def _group_file(idx: int, suffix: str) -> str:
        # Don't include dimension_suffix for group files - they're created without it
        return _group_output_filename(
            group_file_prefixes,
            idx,
            suffix,
            dimension_suffix=None,
        )

    trace_comparison_previews = _build_trace_comparison_previews(
        comparison_dimension
    )
    event_comparison_previews = _build_event_comparison_previews(
        comparison_dimension
    )
    comparison_preview_skip_prefixes = ("states", "epochs")

    def _prerelocate_previews_to_subdir() -> None:
        """Move all SVG preview files to .previews/ subdirectory before OutputData discovers them."""
        if not registration_dir or not os.path.exists(registration_dir):
            logger.warning("Registration dir is empty or doesn't exist, skipping preview relocation")
            return
        
        preview_subdir = os.path.join(registration_dir, ".previews")
        os.makedirs(preview_subdir, exist_ok=True)
        
        # Move all .svg files to .previews/ subdirectory
        svg_files_moved = 0
        for filename in os.listdir(registration_dir):
            if filename.lower().endswith('.svg'):
                src_path = os.path.join(registration_dir, filename)
                dst_path = os.path.join(preview_subdir, filename)
                if os.path.isfile(src_path):
                    os.replace(src_path, dst_path)
                    svg_files_moved += 1
                    logger.debug(f"Pre-relocated preview {filename} to .previews/")
        
        logger.info(f"Pre-relocated {svg_files_moved} SVG preview files to .previews/ subdirectory")

    def _resolve_manual_previews(filename: str) -> List[Tuple[str, str]]:
        """Return statically configured preview definitions for the given file."""
        if not MANUAL_PREVIEW_MAP:
            return []

        basename = Path(filename).name
        stem = Path(filename).stem

        collected: List[Tuple[str, str]] = []
        for key in (filename, basename, stem):
            manual_entries = MANUAL_PREVIEW_MAP.get(key)
            if not manual_entries:
                continue
            for entry in manual_entries:
                if not entry:
                    continue
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    preview_name = str(entry[0]).strip()
                    caption = (
                        str(entry[1]).strip()
                        if entry[1] is not None
                        else ""
                    )
                    if preview_name:
                        collected.append((preview_name, caption))
                else:
                    logger.debug(
                        "Skipping manual preview entry for '%s': %s",
                        filename,
                        entry,
                    )
        return collected

    def _register_file(
        filename: str,
        preview_defs: List[Tuple[str, str]],
        skip_preview_prefixes: Optional[Sequence[str]] = None,
        attach_output_basename: bool = True,
    ):
        """Register an output file with previews, following state_epoch_baseline pattern.
        
        This function:
        1. Checks if the CSV file exists
        2. Filters preview_defs to only existing files using collect_available_previews
        3. Adds any manually configured previews
        4. Registers the file with filtered previews attached
        """
        file_path = os.path.join(registration_dir, filename)
        if not os.path.exists(file_path):
            logger.debug("Skipping registration for %s (not found)", file_path)
            return

        # Filter to only previews that actually exist on disk
        previews = (
            collect_available_previews(registration_dir, preview_defs)
            if preview_defs
            else []
        )
        
        # Add any manual preview entries from MANUAL_PREVIEW_MAP
        manual_preview_entries = _resolve_manual_previews(filename)
        previews.extend(manual_preview_entries)
        
        prefix_rules = PreviewPrefixRules(
            output_basename=output_file_basename,
            skip_preview_prefixes=tuple(skip_preview_prefixes or ()),
        )
        register_output_file(
            output_data=output_data,
            output_dir=registration_dir,
            output_metadata=output_metadata,
            file=filename,
            output_file_basename=output_file_basename,
            preview_files=previews,
            attach_output_basename=attach_output_basename,
            preview_prefix_rules=prefix_rules,
            metadata_filter=_extract_useful_metadata,
            logger_instance=logger,
        )

    normalized_stats: Set[str]
    if correlation_statistic:
        normalized_stats = {
            str(correlation_statistic).strip().lower(),
        }
    else:
        normalized_stats = {"max", "min", "mean"}
    normalized_stats &= {"max", "min", "mean"}
    if not normalized_stats:
        normalized_stats = {"max", "min", "mean"}

    # Move all preview SVG files to .previews/ subdirectory BEFORE OutputData discovers them
    _prerelocate_previews_to_subdir()

    with OutputData() as output_data:
        # Group 1 outputs
        group1_label = resolved_group_names[0]
        _register_file(
            _group_file(0, GROUP_COMBINED_ACTIVITY_SUFFIX),
            _build_group_activity_previews(
                1,
                group1_label,
                group_preview_prefixes,  # Use real group names for previews
                dimension_suffix=normalized_dimension_suffix,
            ),
            skip_preview_prefixes=[group_preview_prefixes[0]],
            attach_output_basename=False,  # Clean filenames - subdirectory provides context
        )
        _register_file(
            _group_file(0, GROUP_COMBINED_TRACE_CORRELATION_SUFFIX),
            _build_group_correlation_previews(
                1,
                group1_label,
                group_preview_prefixes,  # Use real group names for previews
                dimension_suffix=normalized_dimension_suffix,
                allowed_stats=normalized_stats,
            ),
            skip_preview_prefixes=[group_preview_prefixes[0]],
            attach_output_basename=False,
        )
        _register_file(
            _group_file(0, GROUP_COMBINED_MODULATION_SUFFIX),
            _build_group_modulation_previews(
                1,
                group1_label,
                group_preview_prefixes,  # Use real group names for previews
                dimension_suffix=normalized_dimension_suffix,
            ),
            skip_preview_prefixes=[group_preview_prefixes[0]],
            attach_output_basename=False,
        )

        # Group 2 outputs (if provided)
        if len(resolved_group_names) > 1:
            group2_label = resolved_group_names[1]
            _register_file(
                _group_file(1, GROUP_COMBINED_ACTIVITY_SUFFIX),
                _build_group_activity_previews(
                    2,
                    group2_label,
                    group_preview_prefixes,  # Use real group names for previews
                    dimension_suffix=normalized_dimension_suffix,
                ),
                skip_preview_prefixes=[group_preview_prefixes[1]],
                attach_output_basename=False,
            )
            _register_file(
                _group_file(1, GROUP_COMBINED_TRACE_CORRELATION_SUFFIX),
                _build_group_correlation_previews(
                    2,
                    group2_label,
                    group_preview_prefixes,  # Use real group names for previews
                    dimension_suffix=normalized_dimension_suffix,
                    allowed_stats=normalized_stats,
                ),
                skip_preview_prefixes=[group_preview_prefixes[1]],
                attach_output_basename=False,
            )
            _register_file(
                _group_file(1, GROUP_COMBINED_MODULATION_SUFFIX),
                _build_group_modulation_previews(
                    2,
                    group2_label,
                    group_preview_prefixes,  # Use real group names for previews
                    dimension_suffix=normalized_dimension_suffix,
                ),
                skip_preview_prefixes=[group_preview_prefixes[1]],
                attach_output_basename=False,
            )

        # Statistical outputs
        _register_file(
            TRACE_ANOVA_COMPARISONS_CSV,
            trace_comparison_previews,
            skip_preview_prefixes=comparison_preview_skip_prefixes,
            attach_output_basename=False,
        )
        _register_file(
            TRACE_PAIRWISE_COMPARISONS_CSV,
            [],
            attach_output_basename=False,
        )
        _register_file(
            EVENT_ANOVA_COMPARISONS_CSV,
            event_comparison_previews,
            skip_preview_prefixes=comparison_preview_skip_prefixes,
            attach_output_basename=False,
        )
        _register_file(
            EVENT_PAIRWISE_COMPARISONS_CSV,
            [],
            attach_output_basename=False,
        )

        # LMM outputs (only saved when requested)
        _register_file(
            TRACE_LMM_COMPARISONS_CSV,
            [],
            attach_output_basename=False,
        )
        _register_file(
            TRACE_LMM_PAIRWISE_COMPARISONS_CSV,
            [],
            attach_output_basename=False,
        )
        _register_file(
            EVENT_LMM_COMPARISONS_CSV,
            [],
            attach_output_basename=False,
        )
        _register_file(
            EVENT_LMM_PAIRWISE_COMPARISONS_CSV,
            [],
            attach_output_basename=False,
        )

# function alias
compare = combine_compare_state_epoch_data
