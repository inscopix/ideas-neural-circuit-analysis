"""Statistical comparison utilities for state-epoch analysis.

This module provides statistical analysis and visualization functions for
comparing neural activity data across groups in state-epoch analyses.

UPDATED for future toolbox merging:
- Now uses pingouin-based statistical functions from statistical_validation.py
- Integrates standardized formatting from statistical_formatting.py
- Replaces basic scipy.stats implementations with validated pingouin functions
- Ensures consistency with ideas-toolbox-standard-python approach

Key improvements:
- Enhanced data validation before statistical tests
- Better handling of edge cases and missing data
- Standardized output formats across both toolboxes
- Proper effect size calculations and multiple comparison corrections
"""

import logging
import os
import re
import warnings
import traceback
from itertools import product
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ideas.exceptions import IdeasError

from utils.statistical_validation import (
    _suppress_pingouin_warnings,
    _safe_anova,
    _safe_pairwise_ttests,
    _validate_measure_column_flexible,
    DEFAULT_SIGNIFICANCE_THRESHOLD,
)
from utils.statistical_formatting import (
    _standardize_anova_output,
    _add_pairing_columns,
    _finalize_statistical_output,
)
from utils.combine_compare_population_data_utils import (
    _choose_model_type,
    _fit_simple_poisson_glmm,
)

logger = logging.getLogger(__name__)

# Module-level set for deduplicating LMM structure logging
_global_lmm_structure_logged = set()

_SUPPORTED_MULTIPLE_CORRECTION_METHODS = {"bonf", "fdr_bh", "none"}
_MULTIPLE_CORRECTION_ALIASES = {
    "fdr": "fdr_bh",
    "bonferroni": "bonf",
}

# Statistical power validation thresholds
MIN_SUBJECTS_REQUIRED = 2
MIN_SUBJECTS_CAUTIOUS = 3
MIN_SUBJECTS_RECOMMENDED = 5
MIN_OBSERVATIONS_LMM = 3
MIN_OBSERVATIONS_CAUTIOUS_LMM = 6
IMBALANCED_DESIGN_RATIO_THRESHOLD = 10


def _require_supported_multiple_correction_method(
    method: Optional[str],
) -> str:
    """Return a supported multiple comparison method or raise IdeasError."""
    if method is None:
        return "bonf"

    method_str = str(method).strip().lower()
    if not method_str:
        return "bonf"

    canonical_method = _MULTIPLE_CORRECTION_ALIASES.get(method_str, method_str)
    if canonical_method in _SUPPORTED_MULTIPLE_CORRECTION_METHODS:
        return canonical_method

    valid_options = sorted(
        _SUPPORTED_MULTIPLE_CORRECTION_METHODS
        | set(_MULTIPLE_CORRECTION_ALIASES.keys())
    )
    raise IdeasError(
        f"Unsupported multiple_correction value: '{method}'. Valid options: {valid_options}"
    )


def validate_modulation_colors(modulation_colors):
    """Validate modulation_colors input for compatibility with plotting functions."""
    default_modulation_colors = ["green", "blue", "black"]

    try:
        if modulation_colors is None:
            logger.info(
                "Using default modulation colors: %s",
                default_modulation_colors,
            )
            return default_modulation_colors

        if isinstance(modulation_colors, str):
            modulation_colors = modulation_colors.split(",")
            modulation_colors = [x.strip() for x in modulation_colors]

        if not isinstance(modulation_colors, (list, tuple)):
            raise IdeasError(
                "modulation_colors must be a list, tuple, or a comma-separated string."
            )

        if len(modulation_colors) != 3:
            raise IdeasError(
                "modulation_colors must contain exactly three colors."
            )

        for color in modulation_colors:
            if not isinstance(color, str) or not mcolors.is_color_like(color):
                raise IdeasError(
                    f"Invalid color value: {color}. Must be a valid matplotlib color."
                )

        if len(set(modulation_colors)) != len(modulation_colors):
            raise IdeasError("Duplicate colors detected in modulation_colors.")

    except IdeasError as error:
        logger.warning(
            "Invalid modulation_colors provided (%s). Using default colors: %s",
            error,
            default_modulation_colors,
        )
        return default_modulation_colors
    except Exception as error:
        logger.warning(
            "Error validating modulation_colors (%s). Using default colors.",
            error,
        )
        return default_modulation_colors

    return modulation_colors


def _auto_select_parametric(
    data: pd.DataFrame,
    dv_column: str,
    min_sample_size: int = 8,
    grouping_column: Optional[str] = None,
) -> bool:
    """Automatically select parametric vs non-parametric tests."""
    if data is None or data.empty or dv_column not in data.columns:
        logger.debug(
            "Auto-select parametric: insufficient data, using non-parametric tests."
        )
        return False

    cleaned_series = data[dv_column].dropna()
    n_samples = len(cleaned_series)

    if n_samples < min_sample_size:
        logger.info(
            "Auto-select parametric: fewer than %d observations found (n=%d). "
            "Non-parametric tests will be used.",
            min_sample_size,
            n_samples,
        )
        return False

    try:
        normality_data = data.copy()
        columns_needed = [dv_column]

        if grouping_column and grouping_column in normality_data.columns:
            columns_needed.append(grouping_column)

        normality_data = normality_data[columns_needed].dropna()

        if grouping_column and grouping_column in normality_data.columns:
            normality_result = pg.normality(
                data=normality_data,
                dv=dv_column,
                group=grouping_column,
                alpha=DEFAULT_SIGNIFICANCE_THRESHOLD,
            )
        else:
            normality_result = pg.normality(
                data=normality_data,
                dv=dv_column,
                alpha=DEFAULT_SIGNIFICANCE_THRESHOLD,
            )

        parametric = normality_result["normal"].all()

        if parametric:
            logger.info(
                "Auto-select parametric: data appears normally distributed. "
                "Parametric tests will be used."
            )
        else:
            logger.info(
                "Auto-select parametric: data is not normally distributed. "
                "Non-parametric tests will be used."
            )

        return parametric

    except Exception as error:
        logger.warning(
            "Auto-select parametric: normality test failed (%s). "
            "Defaulting to non-parametric tests.",
            error,
        )
        return False


def _resolve_parametric_setting(
    data: pd.DataFrame,
    dv_column: str,
    parametric: Union[str, bool],
    grouping_column: Optional[str] = None,
) -> bool:
    """Resolve parametric setting by handling strings and auto-selection."""
    if isinstance(parametric, str):
        normalized = parametric.strip().lower()
        if normalized == "auto":
            return _auto_select_parametric(
                data,
                dv_column,
                grouping_column=grouping_column,
            )
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        logger.warning(
            "Invalid parametric value '%s'. Using auto-selection.", parametric
        )
        return _auto_select_parametric(
            data,
            dv_column,
            grouping_column=grouping_column,
        )
    return bool(parametric)


def _calculate_subject_averages_state_epoch(
    data: pd.DataFrame,
    subject_col: str,
    measure_col: str,
    group_cols: List[str],
    context: str = "",
) -> pd.DataFrame:
    """Calculate subject-level averages from cell-level data to prevent pseudoreplication.

    This function aggregates multiple cell-level observations per subject into single
    subject-level values. This is CRITICAL for preventing pseudoreplication in
    statistical analyses where individual cells from the same subject are treated
    as independent observations.

    **Pseudoreplication Prevention:**
    - Cell-level data: Multiple cells per subject per condition (NOT independent)
    - Subject-level data: One average per subject per condition (independent samples)
    - Statistical tests require independent observations; cells from same subject violate this

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with cell-level data (one row per cell)
    subject_col : str
        Column name containing subject IDs
    measure_col : str
        Column name containing the measurement values to average
    group_cols : List[str]
        List of grouping columns (e.g., ["state", "epoch", "group_name"])
    context : str, optional
        Context string for logging purposes

    Returns
    -------
    pd.DataFrame
        DataFrame with subject-level averages (one row per subject per condition)

    Notes
    -----
    Example transformation:
        BEFORE (cell-level, N=100 cells from 10 subjects):
            subject_id | state | cell_id | mean_activity
            subject_1  | run   | C001    | 0.5
            subject_1  | run   | C002    | 0.6
            subject_1  | rest  | C001    | 0.2
            ...

        AFTER (subject-level, N=20 observations):
            subject_id | state | mean_activity
            subject_1  | run   | 0.55  (averaged across cells)
            subject_1  | rest  | 0.25  (averaged across cells)
            ...

    This ensures each subject contributes equally to statistical comparisons,
    regardless of the number of cells recorded per subject.

    """
    if data is None or data.empty:
        logger.warning(
            f"Empty data provided for subject averaging in {context}"
        )
        return pd.DataFrame()

    # Log input data structure for transparency
    n_total_cells = len(data)
    n_subjects = (
        data[subject_col].nunique() if subject_col in data.columns else 0
    )

    logger.debug(
        f"Subject averaging ({context}): {n_total_cells} cell-level observations "
        f"from {n_subjects} subjects"
    )

    # Determine aggregation columns
    agg_cols = [subject_col] + group_cols

    # Perform aggregation to subject level
    grouped = (
        data.groupby(agg_cols, observed=True)[measure_col].mean().reset_index()
    )

    # Log data reduction for pseudoreplication prevention
    n_subject_obs = len(grouped)
    reduction_pct = n_subject_obs / n_total_cells if n_total_cells > 0 else 0

    logger.debug(
        f"Subject averaging ({context}): {n_subject_obs} subject-level observations "
        f"({reduction_pct:.1%} of cell-level data) - prevents pseudoreplication"
    )

    return grouped


def _filter_subjects_with_complete_levels(
    data: pd.DataFrame,
    subject_col: str,
    factor_levels: Dict[str, List[Any]],
    require_full_cross: bool,
    context: str,
    enforce_complete_levels: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter subjects lacking complete repeated-measures coverage.

    Parameters
    ----------
    data : pd.DataFrame
        Subject-level dataframe.
    subject_col : str
        Column containing subject identifiers.
    factor_levels : Dict[str, List[Any]]
        Mapping of factor column names to the levels that should be present
        for each subject.
    require_full_cross : bool
        Whether each subject must have the full Cartesian product of factor
        levels (used for paired group comparisons).
    context : str
        Context string for logging.
    enforce_complete_levels : bool, optional
        Whether subjects missing required factor levels should be excluded.
        When False, the function retains subjects with partial coverage and
        returns their identifiers for diagnostic purposes. Defaults to True.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Filtered dataframe and list of excluded subjects.

    """
    if data is None or data.empty or not factor_levels:
        return data, []

    cleaned_levels: Dict[str, List[Any]] = {}
    for factor, levels in factor_levels.items():
        if factor not in data.columns:
            continue
        unique_levels = [
            level
            for level in dict.fromkeys(levels)  # preserve order
            if pd.notna(level) and level in data[factor].unique()
        ]
        cleaned_levels[factor] = unique_levels

    if not cleaned_levels:
        return data, []

    factor_order = list(cleaned_levels.keys())
    expected_combos: Optional[Set[Tuple[Any, ...]]] = None
    missing_combos: Optional[Set[Tuple[Any, ...]]] = None
    if require_full_cross and len(factor_order) >= 2:
        level_products = [cleaned_levels[factor] for factor in factor_order]
        if any(not levels for levels in level_products):
            logger.debug(
                "Skipping repeated-measures filtering due to missing factor levels "
                f"in {context}"
            )
            return pd.DataFrame(), []
        expected_combos = set(product(*level_products))
        actual_combos = set(
            tuple(row[factor] for factor in factor_order)
            for _, row in data[factor_order].drop_duplicates().iterrows()
        )
        missing_combos = expected_combos - actual_combos
        if missing_combos:
            logger.warning(
                "Missing required factor combinations in %s analysis: %s.",
                context,
                list(missing_combos)[:5],
            )
            if enforce_complete_levels:
                logger.warning(
                    "Repeated-measures statistics cannot be computed with the current data."
                )
                return pd.DataFrame(), []

    kept_frames: List[pd.DataFrame] = []
    excluded_subjects: List[str] = []
    incomplete_subjects: List[str] = []

    for subject_id, subject_df in data.groupby(subject_col, sort=False):
        subject_levels_ok = True

        for factor, levels in cleaned_levels.items():
            subject_levels = subject_df[factor].dropna().unique()
            if set(levels) - set(subject_levels):
                subject_levels_ok = False
                break

        # Cache subject_combos computation to avoid redundancy
        subject_combos = None
        if subject_levels_ok and expected_combos is not None:
            subject_combos = set(
                tuple(row[factor] for factor in factor_order)
                for _, row in subject_df[factor_order]
                .drop_duplicates()
                .iterrows()
            )
            if subject_combos != expected_combos:
                subject_levels_ok = False

        has_complete_levels = subject_levels_ok

        if has_complete_levels:
            kept_frames.append(subject_df)
        else:
            if enforce_complete_levels:
                excluded_subjects.append(subject_id)
            else:
                incomplete_subjects.append(subject_id)
                kept_frames.append(subject_df)

    if not kept_frames:
        if enforce_complete_levels and excluded_subjects:
            logger.warning(
                "Excluded all subjects from %s due to incomplete factor coverage.",
                context,
            )
            return pd.DataFrame(), excluded_subjects
        return pd.DataFrame(), incomplete_subjects

    filtered = pd.concat(kept_frames, ignore_index=True)

    if enforce_complete_levels:
        if excluded_subjects:
            logger.info(
                "Excluded %d subjects with incomplete data for %s: %s",
                len(excluded_subjects),
                context,
                excluded_subjects[:5],
            )
        return filtered, excluded_subjects

    if incomplete_subjects:
        logger.info(
            "Proceeding with imbalanced design for %s. Subjects missing coverage: %s",
            context,
            incomplete_subjects[:5],
        )
    if missing_combos:
        logger.info(
            "Imbalanced design detected for %s with missing combinations: %s",
            context,
            list(missing_combos)[:5],
        )

    return filtered, incomplete_subjects


def calculate_state_epoch_comparison_stats(
    *,
    df: pd.DataFrame,
    comparison_dimension: str,
    comparison_values: List[str],
    has_single_group: bool,
    data_pairing: str = "unpaired",
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
    multiple_correction: str = "bonf",
    multiple_correction_scope: str = "global",
    effect_size_method: str = "cohen",
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
    measure_column: str = "mean_activity",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate comparison statistics for state-epoch data.

    **Data Level Handling:**
    This function automatically aggregates cell-level data to subject-level data
    before performing statistical tests. This is CRITICAL for preventing pseudoreplication.

    - **Input**: Cell-level data (one row per cell per subject per condition)
    - **Statistical Analysis**: Subject-level data (one row per subject per condition)
    - **Why**: Cells from the same subject are NOT independent observations

    This approach matches combine_compare_population_data and combine_compare_correlation_data
    tools exactly, ensuring consistent statistical rigor across all IDEAS toolbox tools.

    **Analysis Levels:**
    1. **Single Group**: Repeated measures ANOVA across states/epochs
    2. **Two-Group Unpaired**: Mixed ANOVA (within=states/epochs, between=groups)
    3. **Two-Group Paired**: Repeated measures ANOVA (within=states/epochs AND groups)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with **cell-level** data containing columns:
        - state, epoch: Condition identifiers
        - group_name: Group identifier (for multi-group analysis)
        - normalized_subject_id: Subject identifier (critical for aggregation)
        - measure_column: Measurement values (e.g., mean_activity)
    comparison_dimension : str
        Dimension to compare: "states" or "epochs"
    comparison_values : List[str]
        List of states or epochs to compare
    has_single_group : bool
        Whether analysis is for single group or multiple groups
    data_pairing : str
        Data pairing type ("paired" or "unpaired")
    significance_threshold : float
        Significance threshold for tests
    multiple_correction : str
        Multiple comparison correction method ("bonf", "fdr_bh", "none")
    multiple_correction_scope : str
        Scope of multiple comparison correction:
        - "global": Apply correction across ALL tests from all strata (default, most conservative)
        - "within_stratum": Apply correction only within each stratum (less conservative)
        When comparing states across epochs (or vice versa), global correction ensures
        proper family-wise error rate control across the entire analysis.
    effect_size_method : str
        Effect size calculation method ("cohen", "hedges")
    group_comparison_type : str
        Type of statistical test to perform (default: "two_tailed"):
        - "two_tailed": Two-tailed test (tests for differences in either direction)
        - "one_tailed_less": One-tailed test (tests if first group is less than second)
        - "one_tailed_greater": One-tailed test (tests if first group is greater than second)
    parametric : str
        Indicates whether to perform a parametric test (default: "auto"):
        - "auto": Automatically select parametric test if data follows normal
          distribution and has at least 8 observations per group
        - "True": Always use parametric tests (t-test, ANOVA)
        - "False": Always use non-parametric tests (Mann-Whitney U, Wilcoxon)
    measure_column : str
        Column name containing the measurement values

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - ANOVA results (main effects and interactions)
        - Pairwise comparison results (post-hoc tests)

    Notes
    -----
    **Pseudoreplication Prevention Example:**

    Input (cell-level, pseudoreplicated):
        subject_1, state_A, cell_1: 0.5
        subject_1, state_A, cell_2: 0.6  <- NOT independent from cell_1
        subject_1, state_B, cell_1: 0.3
        subject_1, state_B, cell_2: 0.4

    Statistical Analysis (subject-level, independent):
        subject_1, state_A: 0.55 (averaged across cells)
        subject_1, state_B: 0.35 (averaged across cells)

    This ensures proper degrees of freedom and valid p-values.

    """
    if df is None or df.empty:
        logger.warning("No data provided for comparison")
        return pd.DataFrame(), pd.DataFrame()

    if not comparison_values:
        logger.warning(f"No {comparison_dimension} provided for comparison")
        return pd.DataFrame(), pd.DataFrame()

    # Validate required columns
    required_cols = [
        comparison_dimension[:-1],  # "state" or "epoch"
        "group_name",
        "normalized_subject_id",
        measure_column,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame()

    # Normalize multiple comparison method to ensure consistent downstream metadata
    multiple_correction = _require_supported_multiple_correction_method(
        multiple_correction
    )

    # Prepare data for analysis
    comp_col = comparison_dimension[:-1]  # "state" or "epoch"
    aov_results = []
    pairwise_results = []

    # Group by the non-comparison dimension
    other_dimension = "epoch" if comp_col == "state" else "state"

    if other_dimension in df.columns:
        # Analyze each level of the other dimension separately
        other_values = df[other_dimension].unique()
    else:
        # No other dimension, analyze everything together
        other_values = [None]

    for other_val in other_values:
        # Filter data
        if other_val is not None:
            subset = df[df[other_dimension] == other_val].copy()
            context = f"{other_dimension}={other_val}"
        else:
            subset = df.copy()
            context = "all data"

        if subset.empty:
            continue

        if comparison_values:
            filtered_subset = subset[
                subset[comp_col].isin(comparison_values)
            ].copy()
            if filtered_subset.empty:
                logger.debug(
                    f"No data for requested {comp_col} values {comparison_values} "
                    f"in {context}"
                )
                continue

            missing_levels = [
                value
                for value in comparison_values
                if value not in filtered_subset[comp_col].unique()
            ]
            if missing_levels:
                logger.info(
                    f"Missing {comp_col} levels {missing_levels} in {context}; "
                    "they will be skipped"
                )

            subset = filtered_subset

        # Perform ANOVA-style comparison
        if has_single_group:
            # Within-group comparison across states/epochs
            aov_result = _perform_within_group_anova(
                data=subset,
                comparison_col=comp_col,
                measure_col=measure_column,
                subject_col="normalized_subject_id",
                context=context,
            )
            if not aov_result.empty:
                if other_val is not None:
                    aov_result[other_dimension] = other_val
                aov_results.append(aov_result)

            # Pairwise comparisons
            pairwise_result = _perform_pairwise_tests(
                data=subset,
                comparison_col=comp_col,
                comparison_values=comparison_values,
                measure_col=measure_column,
                subject_col="normalized_subject_id",
                group_col=None,
                data_pairing=data_pairing,
                correction=multiple_correction,
                effect_size=effect_size_method,
                group_comparison_type=group_comparison_type,
                parametric=parametric,
                context=context,
            )
            if not pairwise_result.empty:
                if other_val is not None:
                    pairwise_result[other_dimension] = other_val
                pairwise_results.append(pairwise_result)

        else:
            # Between-group comparison
            aov_result = _perform_between_group_anova(
                data=subset,
                comparison_col=comp_col,
                measure_col=measure_column,
                subject_col="normalized_subject_id",
                group_col="group_name",
                data_pairing=data_pairing,
                context=context,
            )
            if not aov_result.empty:
                if other_val is not None:
                    aov_result[other_dimension] = other_val
                aov_results.append(aov_result)

            # Pairwise comparisons between groups
            pairwise_result = _perform_pairwise_tests(
                data=subset,
                comparison_col=comp_col,
                comparison_values=comparison_values,
                measure_col=measure_column,
                subject_col="normalized_subject_id",
                group_col="group_name",
                data_pairing=data_pairing,
                correction=multiple_correction,
                effect_size=effect_size_method,
                group_comparison_type=group_comparison_type,
                parametric=parametric,
                context=context,
            )
            if not pairwise_result.empty:
                if other_val is not None:
                    pairwise_result[other_dimension] = other_val
                pairwise_results.append(pairwise_result)

    # Combine results
    combined_aov = (
        pd.concat(aov_results, ignore_index=True)
        if aov_results
        else pd.DataFrame()
    )
    combined_pairwise = (
        pd.concat(pairwise_results, ignore_index=True)
        if pairwise_results
        else pd.DataFrame()
    )

    # Apply global multiple comparison correction if requested
    if not combined_pairwise.empty and multiple_correction_scope == "global":
        logger.info(
            f"Applying {multiple_correction_scope} multiple comparison correction "
            f"across all {len(combined_pairwise)} pairwise tests"
        )
        combined_pairwise = _apply_global_multiple_comparison_correction(
            pairwise_df=combined_pairwise,
            correction_method=multiple_correction,
        )
    elif (
        not combined_pairwise.empty
        and multiple_correction_scope == "within_stratum"
    ):
        logger.info(
            f"Using within-stratum multiple comparison correction "
            f"(correction applied separately within each {other_dimension})"
        )
        # Add metadata to indicate within-stratum correction
        combined_pairwise["correction_scope"] = "within_stratum"
        if other_dimension in combined_pairwise.columns:
            combined_pairwise["n_tests_corrected"] = combined_pairwise.groupby(
                other_dimension
            )["p-unc"].transform("count")
    elif not combined_pairwise.empty:
        logger.warning(
            f"Unknown multiple_correction_scope: '{multiple_correction_scope}'. "
            f"Valid options: 'global', 'within_stratum'. Using 'global' as default."
        )
        combined_pairwise = _apply_global_multiple_comparison_correction(
            pairwise_df=combined_pairwise,
            correction_method=multiple_correction,
        )

    return combined_aov, combined_pairwise


@_suppress_pingouin_warnings
def _perform_within_group_anova(
    data: pd.DataFrame,
    comparison_col: str,
    measure_col: str,
    subject_col: str,
    context: str,
) -> pd.DataFrame:
    """Perform within-subjects ANOVA (repeated measures) using pingouin.

    This function handles SINGLE GROUP analysis, consistent with
    combine_compare_population_data and combine_compare_correlation_data tools.

    Statistical Approach:
    ---------------------
    - Within-subject factor: comparison dimension (states or epochs)
    - Statistical test: Repeated measures ANOVA
    - Use case: One group compared across multiple states/epochs
    - Same subjects measured under different conditions

    Example:
    --------
    Comparing a single group of subjects across 3 behavioral states
    (running, resting, grooming) to see if neural activity differs by state.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with subject and comparison dimension columns
    comparison_col : str
        Name of comparison column ("state" or "epoch")
    measure_col : str
        Name of measurement column
    subject_col : str
        Name of subject ID column
    context : str
        Context string for logging

    Returns
    -------
    pd.DataFrame
        Standardized ANOVA results

    """
    try:
        # Validate data first
        is_valid, error_msg, validated_data = (
            _validate_measure_column_flexible(
                data, measure_col, f"within-group ANOVA ({context})"
            )
        )

        if not is_valid:
            logger.debug(
                f"Within-group ANOVA validation failed for {context}: {error_msg}"
            )
            return pd.DataFrame()

        # ===================================================================
        # CRITICAL: Convert cell-level to subject-level data
        # ===================================================================
        # Aggregates multiple cells per subject into single subject-level value
        # This prevents pseudoreplication (treating non-independent cells as independent)
        # Matches combine_compare_population_data and combine_compare_correlation_data
        # ===================================================================
        grouped = _calculate_subject_averages_state_epoch(
            data=validated_data,
            subject_col=subject_col,
            measure_col=measure_col,
            group_cols=[comparison_col],
            context=f"within-group ANOVA ({context})",
        )

        factor_levels = {
            comparison_col: grouped[comparison_col].dropna().unique().tolist()
        }
        grouped, _ = _filter_subjects_with_complete_levels(
            grouped,
            subject_col=subject_col,
            factor_levels=factor_levels,
            require_full_cross=False,
            context=f"within-group ANOVA ({context})",
        )
        if grouped.empty:
            logger.warning(
                "Insufficient complete repeated-measures data for within-group ANOVA "
                "in %s",
                context,
            )
            return pd.DataFrame()

        # Get unique comparison values after filtering
        comp_values = grouped[comparison_col].unique()
        if len(comp_values) < 2:
            logger.warning(
                f"Not enough {comparison_col} levels for ANOVA in {context}"
            )
            return pd.DataFrame()

        # Use pingouin for repeated measures ANOVA
        aov_result = _safe_anova(
            data=grouped,
            dv=measure_col,
            within=comparison_col,
            subject=subject_col,
            detailed=True,
        )

        if aov_result is None or aov_result.empty:
            return pd.DataFrame()

        # Add stat_method metadata (consistent with standard-python toolbox)
        # Single group analysis always uses repeated measures ANOVA
        aov_result["stat_method"] = "rm_anova"

        # Standardize the output
        metadata = {
            "Comparison": f"{comparison_col.capitalize()} Comparison",
            "Measure": measure_col,
        }

        standardized_result = _standardize_anova_output(
            aov_result,
            metadata=metadata,
            analysis_level="subject",
            state_comparison_type=comparison_col,
            measure_name=measure_col,
        )

        standardized_result["Context"] = context

        # Apply final formatting
        final_result = _finalize_statistical_output(
            standardized_result, "anova"
        )

        return final_result

    except Exception as e:
        logger.error(f"Error in within-group ANOVA for {context}: {e}")
        return pd.DataFrame()


@_suppress_pingouin_warnings
def _perform_between_group_anova(
    data: pd.DataFrame,
    comparison_col: str,
    measure_col: str,
    subject_col: str,
    group_col: str,
    data_pairing: str,
    context: str,
) -> pd.DataFrame:
    """Perform between-groups ANOVA (or mixed/repeated measures ANOVA) using pingouin.

    This function implements the three standard analysis patterns consistent with
    combine_compare_population_data and combine_compare_correlation_data tools:

    1. TWO-GROUP UNPAIRED (Mixed ANOVA):
       - Within-subject factor: comparison dimension (states or epochs)
       - Between-subject factor: groups
       - Statistical test: Mixed ANOVA (within + between factors)
       - Use case: Different subjects in each group

    2. TWO-GROUP PAIRED (Repeated Measures ANOVA):
       - Within-subject factors: comparison dimension AND groups
       - Statistical test: Repeated measures ANOVA with 2 within-subject factors
       - Use case: Same subjects measured in both groups (e.g., pre/post treatment)

    3. SINGLE GROUP (handled by _perform_within_group_anova instead):
       - Within-subject factor: comparison dimension only
       - Statistical test: Repeated measures ANOVA
       - Use case: One group compared across states/epochs

    Parameters
    ----------
    data : pd.DataFrame
        Input data with subject, group, and comparison dimension columns
    comparison_col : str
        Name of comparison column ("state" or "epoch")
    measure_col : str
        Name of measurement column
    subject_col : str
        Name of subject ID column
    group_col : str
        Name of group column
    data_pairing : str
        Type of data pairing ("paired" or "unpaired")
    context : str
        Context string for logging

    Returns
    -------
    pd.DataFrame
        Standardized ANOVA results

    """
    try:
        # Validate data first
        is_valid, error_msg, validated_data = (
            _validate_measure_column_flexible(
                data, measure_col, f"between-group ANOVA ({context})"
            )
        )

        if not is_valid:
            logger.debug(
                f"Between-group ANOVA validation failed for {context}: {error_msg}"
            )
            return pd.DataFrame()

        # ===================================================================
        # CRITICAL: Convert cell-level to subject-level data
        # ===================================================================
        # Aggregates multiple cells per subject into single subject-level value
        # This prevents pseudoreplication (treating non-independent cells as independent)
        # For two-group analysis: aggregates by [subject, group, comparison_dimension]
        # Matches combine_compare_population_data and combine_compare_correlation_data
        # ===================================================================
        grouped = _calculate_subject_averages_state_epoch(
            data=validated_data,
            subject_col=subject_col,
            measure_col=measure_col,
            group_cols=[group_col, comparison_col],
            context=f"between-group ANOVA ({context})",
        )

        factor_levels = {
            comparison_col: grouped[comparison_col].dropna().unique().tolist()
        }
        require_full_cross = False
        if data_pairing == "paired":
            factor_levels[group_col] = (
                grouped[group_col].dropna().unique().tolist()
            )
            require_full_cross = True

        grouped, _ = _filter_subjects_with_complete_levels(
            grouped,
            subject_col=subject_col,
            factor_levels=factor_levels,
            require_full_cross=require_full_cross,
            context=f"between-group ANOVA ({context})",
        )
        if grouped.empty:
            logger.warning(
                "Insufficient complete repeated-measures data for between-group ANOVA "
                "in %s",
                context,
            )
            return pd.DataFrame()

        # Get unique values
        groups_list = grouped[group_col].unique()
        comp_values = grouped[comparison_col].unique()

        if len(groups_list) < 2:
            logger.warning(
                f"Not enough groups for between-group ANOVA in {context}"
            )
            return pd.DataFrame()

        # ===================================================================
        # DETERMINE ANOVA TYPE BASED ON DATA STRUCTURE AND PAIRING
        # Consistent with combine_compare_population_data and
        # combine_compare_correlation_data statistical approaches
        # ===================================================================

        stat_method = None

        if len(comp_values) >= 2:
            # Multiple comparison values (states or epochs)

            if data_pairing == "paired":
                # TWO-GROUP PAIRED: Repeated measures ANOVA with 2 within-subject factors
                # Both comparison dimension (states/epochs) AND groups are within-subject
                # Example: Same subjects measured at different timepoints (groups) across states
                # Statistical model: within=[comparison_col, group_col]
                aov_result = _safe_anova(
                    data=grouped,
                    dv=measure_col,
                    within=[comparison_col, group_col],
                    subject=subject_col,
                    detailed=True,
                )
                stat_method = "rm_anova"
            else:
                # TWO-GROUP UNPAIRED: Mixed ANOVA
                # Within-subject: comparison dimension (states/epochs)
                # Between-subject: groups (different subjects in each group)
                # Example: Control vs Treatment groups measured across states
                # Statistical model: within=comparison_col, between=group_col
                aov_result = _safe_anova(
                    data=grouped,
                    dv=measure_col,
                    within=comparison_col,
                    between=group_col,
                    subject=subject_col,
                )
                stat_method = "mixed_anova"
        else:
            # Single comparison value - direct group comparison
            # Simple between-subjects ANOVA (only comparing groups)
            # This handles edge case of single state/epoch with multiple groups
            if data_pairing == "paired":
                aov_result = _safe_anova(
                    data=grouped,
                    dv=measure_col,
                    within=group_col,
                    subject=subject_col,
                    detailed=True,
                )
                stat_method = "rm_anova"
            else:
                aov_result = _safe_anova(
                    data=grouped,
                    dv=measure_col,
                    between=group_col,
                    detailed=True,
                )
                stat_method = "anova"

        if aov_result is None or aov_result.empty:
            return pd.DataFrame()

        # Add stat_method metadata (consistent with standard-python toolbox)
        # Set specific stat_method based on the ANOVA type used
        if not stat_method:
            if data_pairing == "paired":
                stat_method = "rm_anova"
            else:
                stat_method = "mixed_anova"

        aov_result["stat_method"] = stat_method

        # Standardize the output
        metadata = {
            "Comparison": f"Group x {comparison_col.capitalize()} Comparison",
            "Measure": measure_col,
        }

        standardized_result = _standardize_anova_output(
            aov_result,
            metadata=metadata,
            analysis_level="subject",
            state_comparison_type="group",
            measure_name=measure_col,
        )

        standardized_result["Context"] = context

        # Apply final formatting
        final_result = _finalize_statistical_output(
            standardized_result, "anova"
        )

        return final_result

    except Exception as e:
        logger.error(f"Error in between-group ANOVA for {context}: {e}")
        return pd.DataFrame()


def _format_interaction_comparisons(
    pairwise_df: pd.DataFrame, comparison_col: str
) -> pd.DataFrame:
    """Format A and B columns for interaction term comparisons.

    For interaction comparisons (e.g., epoch * group_name), pingouin returns
    the between-subject factor levels in A and B columns, and the within-subject
    factor level in a separate column (named after comparison_col, e.g., 'epoch').

    This function combines them to make it clear what's being compared, e.g.:
    - A: "Control" becomes "Control-baseline"
    - B: "Treatment" becomes "Treatment-baseline"

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise comparison results from pingouin
    comparison_col : str
        Name of the within-subject factor column (e.g., 'epoch', 'state')

    Returns
    -------
    pd.DataFrame
        Pairwise results with formatted A and B columns for interaction terms

    """
    if pairwise_df is None or pairwise_df.empty:
        return pairwise_df

    df = pairwise_df.copy()

    # Check if we have the necessary columns
    if (
        "Contrast" not in df.columns
        or "A" not in df.columns
        or "B" not in df.columns
    ):
        return df

    # Identify interaction rows
    interaction_mask = df["Contrast"].astype(str).str.contains("\\*", na=False)

    if not interaction_mask.any():
        return df

    # Check if the within-subject factor column exists
    if comparison_col not in df.columns:
        logger.warning(
            f"Cannot format interaction comparisons: '{comparison_col}' column not found. "
            f"Available columns: {list(df.columns)}"
        )
        return df

    # Format A and B columns for interaction rows
    for idx in df[interaction_mask].index:
        within_level = df.at[idx, comparison_col]

        # Only format if we have a valid within-subject factor level
        if (
            pd.notna(within_level)
            and within_level != ""
            and within_level != "-"
        ):
            current_a = str(df.at[idx, "A"])
            current_b = str(df.at[idx, "B"])

            # Combine group name with within-subject factor level using hyphen separator
            df.at[idx, "A"] = f"{current_a}-{within_level}"
            df.at[idx, "B"] = f"{current_b}-{within_level}"

            logger.debug(
                f"Formatted interaction comparison: {current_a} vs {current_b} "
                f"at {comparison_col}={within_level} -> "
                f"{df.at[idx, 'A']} vs {df.at[idx, 'B']}"
            )

    return df


@_suppress_pingouin_warnings
def _perform_pairwise_tests(
    data: pd.DataFrame,
    comparison_col: str,
    comparison_values: List[str],
    measure_col: str,
    subject_col: str,
    group_col: Optional[str],
    data_pairing: str,
    correction: str,
    effect_size: str,
    group_comparison_type: str,
    parametric: str,
    context: str,
    allow_partial_subjects: bool = False,
) -> pd.DataFrame:
    """Perform pairwise t-tests with multiple comparison correction using pingouin."""
    try:
        # Validate data first
        is_valid, error_msg, validated_data = (
            _validate_measure_column_flexible(
                data, measure_col, f"pairwise tests ({context})"
            )
        )

        if not is_valid:
            logger.debug(
                f"Pairwise tests validation failed for {context}: {error_msg}"
            )
            return pd.DataFrame()

        # ===================================================================
        # CRITICAL: Convert cell-level to subject-level data
        # ===================================================================
        # Aggregates multiple cells per subject into single subject-level value
        # This prevents pseudoreplication in pairwise comparisons
        # Matches combine_compare_population_data and combine_compare_correlation_data
        # ===================================================================
        if group_col:
            # Two-group analysis: aggregate by subject, group, and comparison dimension
            grouped = _calculate_subject_averages_state_epoch(
                data=validated_data,
                subject_col=subject_col,
                measure_col=measure_col,
                group_cols=[group_col, comparison_col],
                context=f"pairwise tests ({context})",
            )
        else:
            # Single-group analysis: aggregate by subject and comparison dimension
            grouped = _calculate_subject_averages_state_epoch(
                data=validated_data,
                subject_col=subject_col,
                measure_col=measure_col,
                group_cols=[comparison_col],
                context=f"pairwise tests ({context})",
            )

        if comparison_values:
            grouped = grouped[
                grouped[comparison_col].isin(comparison_values)
            ].copy()
            if grouped.empty:
                logger.debug(
                    f"No data available for requested {comparison_col} values "
                    f"{comparison_values} in pairwise tests ({context})"
                )
                return pd.DataFrame()

        factor_levels = {
            comparison_col: grouped[comparison_col].dropna().unique().tolist()
        }
        require_full_cross = bool(group_col) and data_pairing == "paired"
        if require_full_cross and group_col in grouped.columns:
            factor_levels[group_col] = (
                grouped[group_col].dropna().unique().tolist()
            )

        grouped, coverage_subjects = _filter_subjects_with_complete_levels(
            grouped,
            subject_col=subject_col,
            factor_levels=factor_levels,
            require_full_cross=require_full_cross,
            context=f"pairwise tests ({context})",
            enforce_complete_levels=not allow_partial_subjects,
        )

        if grouped.empty:
            logger.debug(
                "Pairwise tests skipped: no subjects with complete repeated-measures "
                f"data in {context}"
            )
            if allow_partial_subjects and coverage_subjects:
                logger.debug(
                    "Subjects with partial coverage retained for diagnostics: %s",
                    coverage_subjects[:5],
                )
            return pd.DataFrame()

        def _explicit_parametric_requested(value: Union[str, bool]) -> bool:
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0", "auto"}:
                    return False
                return False
            return bool(value)

        explicit_parametric = _explicit_parametric_requested(parametric)

        # Use pingouin for pairwise tests
        within_factors: Union[str, List[str], None]
        if group_col and data_pairing == "paired":
            # Include only within-subject factors with at least two levels
            within_candidates: List[str] = []

            if comparison_col in grouped.columns:
                if grouped[comparison_col].nunique() >= 2:
                    within_candidates.append(comparison_col)
            if group_col in grouped.columns:
                if grouped[group_col].nunique() >= 2:
                    within_candidates.append(group_col)

            if not within_candidates:
                logger.debug(
                    "Pairwise tests skipped: insufficient levels for within-subject "
                    f"factors in {context}"
                )
                return pd.DataFrame()
            if len(within_candidates) == 1:
                within_factors = within_candidates[0]
            else:
                within_factors = within_candidates
        else:
            within_factors = comparison_col

        if isinstance(within_factors, list):
            potential_within = next(
                (
                    factor
                    for factor in within_factors
                    if isinstance(factor, str)
                ),
                None,
            )
        else:
            potential_within = (
                within_factors if isinstance(within_factors, str) else None
            )

        grouping_column = None
        if potential_within and potential_within in grouped.columns:
            grouping_column = potential_within
        elif group_col and group_col in grouped.columns:
            grouping_column = group_col

        resolved_parametric = _resolve_parametric_setting(
            grouped,
            measure_col,
            parametric,
            grouping_column=grouping_column,
        )

        pairwise_kwargs: Dict[str, Any] = {
            "data": grouped,
            "dv": measure_col,
            "within": within_factors,
            "subject": subject_col,
            "padjust": correction,
            "effsize": effect_size,
            "tail": group_comparison_type,
            "parametric": resolved_parametric,
        }

        if group_col and not (data_pairing == "paired"):
            pairwise_kwargs["between"] = group_col

        pairwise_result = _safe_pairwise_ttests(**pairwise_kwargs)

        missing_pairwise_results = (
            pairwise_result is None or pairwise_result.empty
        )

        if missing_pairwise_results:
            n_subjects = (
                grouped[subject_col].nunique()
                if subject_col in grouped.columns
                else len(grouped)
            )
            n_conditions = (
                grouped[comparison_col].nunique()
                if comparison_col in grouped.columns
                else 0
            )

            subjects_per_group = None
            if group_col and group_col in grouped.columns:
                subjects_per_group = grouped.groupby(group_col)[
                    subject_col
                ].nunique()

            has_min_subjects = n_subjects >= 2
            has_multiple_conditions = n_conditions >= 2
            has_group_support = not group_col or (
                subjects_per_group is not None
                and not subjects_per_group.empty
                and (subjects_per_group >= 2).all()
            )

            should_raise_parametric_error = (
                explicit_parametric
                and has_min_subjects
                and has_multiple_conditions
                and has_group_support
            )

            if should_raise_parametric_error:
                error_msg = (
                    f"Parametric pairwise tests failed for {context} ({measure_col}).\n\n"
                    f"Data characteristics:\n"
                    f"  - Number of subjects: {n_subjects}\n"
                    f"  - Number of conditions: {n_conditions}\n"
                    f"  - Total observations: {len(grouped)}\n"
                )

                if subjects_per_group is not None:
                    group_details = "\n".join(
                        f"    - {group_name}: {count} subjects"
                        for group_name, count in subjects_per_group.items()
                    )
                    error_msg += f"  - Subjects per group:\n{group_details}\n"

                error_msg += (
                    "\nCommon reasons for parametric test failure:\n"
                    "  1. Small sample size (n < 8 per group)\n"
                    "  2. Zero variance in data (all values identical)\n"
                    "  3. Insufficient observations per condition\n"
                    "  4. Data violates normality assumptions\n\n"
                    "Recommendations:\n"
                    "  - Use parametric='auto' to automatically select "
                    "appropriate tests\n"
                    "  - Use parametric='False' to explicitly use non-parametric "
                    "tests (Mann-Whitney, Wilcoxon)\n"
                    "  - Check your data for quality issues (missing values, "
                    "outliers)\n"
                    "  - Ensure sufficient sample sizes for parametric "
                    "assumptions\n"
                )
                raise IdeasError(error_msg)

            logger.debug(
                f"Pairwise tests returned no results for {context} ({measure_col}). "
                f"Subjects={n_subjects}, conditions={n_conditions}"
            )
            return pd.DataFrame()

        # Format interaction comparison A and B columns to include within-subject factor level
        pairwise_result = _format_interaction_comparisons(
            pairwise_result, comparison_col
        )

        # Add context and standardize columns
        pairwise_result["Context"] = context
        pairwise_result["Measure"] = measure_col
        pairwise_result["Comparison"] = (
            f"{comparison_col.capitalize()} Comparison"
        )

        # Add pairing columns for consistency with standard-python toolbox
        state_comparison_type = "group" if group_col else comparison_col
        pairwise_result = _add_pairing_columns(
            pairwise_result,
            state_comparison_type=state_comparison_type,
            data_pairing=data_pairing,
        )

        # Apply final formatting
        final_result = _finalize_statistical_output(
            pairwise_result, "pairwise"
        )

        return final_result

    except Exception as e:
        logger.error(f"Error in pairwise tests for {context}: {e}")
        return pd.DataFrame()


# Note: Effect size calculation and WITHIN-STRATUM multiple comparison correction
# are now handled by pingouin's built-in functions (_safe_pairwise_ttests, _safe_anova).
# However, GLOBAL/CROSS-STRATUM correction still requires the standalone utility below
# to properly control family-wise error rate across all strata (see function docstring).


def _apply_global_multiple_comparison_correction(
    pairwise_df: pd.DataFrame,
    correction_method: str = "bonf",
) -> pd.DataFrame:
    """Apply multiple comparison correction globally across all strata.

    This function addresses the stratified analysis multiple comparison problem:
    when comparing across states separately for each epoch (or vice versa),
    the total number of tests is the product of comparisons in both dimensions.

    CRITICAL ISSUE ADDRESSED:
    -------------------------
    When performing stratified comparisons (e.g., comparing 3 states within each
    of 4 epochs), the current within-stratum correction only accounts for tests
    within each stratum. This function applies correction across ALL tests to
    maintain proper family-wise error rate control.

    Example Problem:
    ----------------
    - Comparing 3 states across 4 epochs between 2 groups
    - WITHOUT global correction:
      * Epoch 1: Bonferroni for 3 tests (alpha = 0.05/3 = 0.0167)
      * Epoch 2: Bonferroni for 3 tests (alpha = 0.05/3 = 0.0167)
      * Epoch 3: Bonferroni for 3 tests (alpha = 0.05/3 = 0.0167)
      * Epoch 4: Bonferroni for 3 tests (alpha = 0.05/3 = 0.0167)
      * Total: 12 tests, but each only corrected for 3!
      * Actual family-wise error rate: ~18% (not 5%)

    - WITH global correction:
      * All 12 tests corrected together (alpha = 0.05/12 = 0.00417)
      * True family-wise error rate: 5%

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Combined pairwise comparison results from all strata.
        Expected to contain 'p-unc' (uncorrected p-values) column from pingouin.
    correction_method : str
        Multiple comparison correction method:
        - "bonf": Bonferroni correction (most conservative)
        - "fdr_bh": Benjamini-Hochberg False Discovery Rate
        - "none": No correction (returns input unchanged)

    Returns
    -------
    pd.DataFrame
        Pairwise results with globally corrected p-values.
        Adds columns:
        - 'p-corr': Globally corrected p-values
        - 'correction_scope': Labels correction as "global"
        - 'n_tests_corrected': Number of tests in correction family

    Notes
    -----
    This function uses the uncorrected p-values ('p-unc') stored by pingouin
    to recompute correction across the entire test family. This approach ensures
    proper correction without "double-correcting":

    - Within-stratum: p_uncorrected × n_within_tests
    - Global (this function): p_uncorrected × n_total_tests (REPLACES within-stratum)

    Existing corrected p-value columns are removed so the returned dataframe
    exposes only the globally corrected values via 'p-corr'. This prevents
    accidentally using within-stratum corrections when global correction was
    intended.

    """
    if pairwise_df.empty:
        logger.debug("Empty pairwise dataframe; skipping global correction")
        return pairwise_df

    # Normalize correction method to canonical form
    correction_method = _require_supported_multiple_correction_method(
        correction_method
    )

    if correction_method == "none":
        logger.info(
            "Global correction set to 'none'; returning unadjusted p-values"
        )
        result = pairwise_df.copy()
        result = result.drop(
            columns=[
                "p-corr",
                "p-corr-within-stratum",
                "p-corr-global",
                "correction_method",
            ],
            errors="ignore",
        )

        numeric_p_values = pd.to_numeric(result["p-unc"], errors="coerce")
        valid_mask_none = numeric_p_values.between(
            0.0,
            1.0,
            inclusive="both",
        )

        result["p-corr"] = numeric_p_values
        result["correction_scope"] = "global"
        result["n_tests_corrected"] = int(valid_mask_none.sum())

        return result

    # Check for required columns
    if "p-unc" not in pairwise_df.columns:
        logger.warning(
            "No uncorrected p-values ('p-unc') found in pairwise results. "
            "Cannot apply global correction. Returning original results."
        )
        return pairwise_df

    # Normalize p-values to numeric and identify valid entries
    p_series = pd.to_numeric(pairwise_df["p-unc"], errors="coerce")
    p_uncorrected = p_series.to_numpy()

    invalid_mask = (
        ~np.isfinite(p_uncorrected)
        | (p_uncorrected < 0.0)
        | (p_uncorrected > 1.0)
    )
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        logger.info(
            f"Excluding {n_invalid} invalid p-values (NaN or outside [0, 1]) "
            "from global multiple comparison correction"
        )

    valid_mask = ~invalid_mask
    n_valid_tests = int(valid_mask.sum())

    # Prepare container for corrected values (default NaN)
    p_corrected_global = np.full_like(p_uncorrected, np.nan, dtype=float)

    if correction_method == "bonf":
        correction_name = "Bonferroni"
        if n_valid_tests:
            corrected_valid = np.minimum(
                p_uncorrected[valid_mask] * n_valid_tests,
                1.0,
            )
            p_corrected_global[valid_mask] = corrected_valid
    elif correction_method == "fdr_bh":
        correction_name = "FDR (Benjamini-Hochberg)"
        if n_valid_tests:
            valid_p_values = p_uncorrected[valid_mask]
            logger.debug(
                "Applying manual Benjamini-Hochberg FDR correction across all strata"
            )
            # Using manual Benjamini-Hochberg implementation
            corrected_valid = _apply_manual_bh(valid_p_values)

            p_corrected_global[valid_mask] = corrected_valid
    else:
        # This should never happen due to normalization, but keep as safeguard
        logger.warning(
            f"Unexpected correction method after normalization: '{correction_method}'. "
            f"Using 'bonf' as fallback."
        )
        correction_name = "Bonferroni (fallback)"
        if n_valid_tests:
            corrected_valid = np.minimum(
                p_uncorrected[valid_mask] * n_valid_tests,
                1.0,
            )
            p_corrected_global[valid_mask] = corrected_valid

    # Create result dataframe with global corrections
    result = pairwise_df.copy()
    # Remove any pre-existing corrected columns to avoid conflicting metadata
    result = result.drop(
        columns=[
            "p-corr",
            "p-corr-within-stratum",
            "p-corr-global",
            "correction_method",
        ],
        errors="ignore",
    )

    # Set p-corr as the primary corrected p-value (global correction)
    result["p-corr"] = p_corrected_global

    # Add metadata to document correction approach
    result["correction_scope"] = "global"
    result["n_tests_corrected"] = n_valid_tests

    logger.info(
        f"Applied global {correction_name} correction across {n_valid_tests} "
        "valid tests from all strata"
    )

    return result


def _apply_manual_bh(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg correction manually for fallback paths."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    adjusted_sorted = np.empty(n, dtype=float)
    min_adjusted = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted_value = min(sorted_p[i] * n / rank, 1.0)
        min_adjusted = min(adjusted_value, min_adjusted)
        adjusted_sorted[i] = min_adjusted

    adjusted = np.empty(n, dtype=float)
    adjusted[sorted_indices] = adjusted_sorted

    return adjusted


def _combine_warning_messages(messages: List[str]) -> str:
    """Join warning messages while preserving order and removing duplicates.

    Parameters
    ----------
    messages : List[str]
        List of warning messages to combine

    Returns
    -------
    str
        Combined warning messages separated by semicolons

    """
    if not messages:
        return ""
    seen: Dict[str, None] = {}
    combined: List[str] = []
    for msg in messages:
        if not msg:
            continue
        trimmed = str(msg).strip()
        if not trimmed:
            continue
        if trimmed in seen:
            continue
        seen[trimmed] = None
        combined.append(trimmed)
    return "; ".join(combined)


def _merge_text_column_with_message(
    df_to_update: pd.DataFrame, column: str, message: str
) -> pd.DataFrame:
    """Ensure column exists in dataframe and append message if provided.

    Parameters
    ----------
    df_to_update : pd.DataFrame
        Dataframe to update
    column : str
        Column name to add or update
    message : str
        Message to append to the column

    Returns
    -------
    pd.DataFrame
        Updated dataframe with message appended to column

    """
    if df_to_update is None or df_to_update.empty:
        return df_to_update

    df_local = df_to_update.copy()
    if column not in df_local.columns:
        df_local[column] = "NA"

    series = df_local[column]

    # Convert categorical columns to object before introducing new fill values
    if pd.api.types.is_categorical_dtype(series):
        series = series.astype("object")

    series = series.where(series.notna(), "NA")
    series = (
        series.astype(str)
        .str.strip()
        .replace({"": "NA", "nan": "NA", "None": "NA", "none": "NA"})
    )

    df_local[column] = series

    if message:
        df_local[column] = df_local[column].apply(
            lambda existing: (
                message
                if existing in {"NA", "nan", ""}
                else f"{existing}; {message}"
            )
        )

    return df_local


def match_subjects(
    group1_files: List[str],
    group2_files: List[str],
    match_method: str = "number",
) -> List[Tuple[str, str]]:
    """Match subjects between two groups based on the specified method.

    Parameters
    ----------
    group1_files : List[str]
        List of file paths for group 1 subjects
    group2_files : List[str]
        List of file paths for group 2 subjects
    match_method : str
        Method for matching subjects ("number", "filename", or "order")

    Returns
    -------
    List[Tuple[str, str]]
        List of matched file path pairs (group1_file, group2_file)

    Raises
    ------
    ValueError
        If input parameters are invalid or if no matches can be found

    """
    if not group1_files or not group2_files:
        raise ValueError(
            "Both group1_files and group2_files must be non-empty lists"
        )

    normalized_method = (match_method or "").strip().lower()
    if normalized_method == "name":
        normalized_method = "filename"

    if normalized_method not in ["number", "filename", "order"]:
        raise ValueError(
            f"Invalid match_method: '{match_method}'. Must be one of: number, filename, order"
        )

    matched_pairs = []
    logger.info(f"Attempting to match subjects using {normalized_method} method")

    def extract_numbers(filepath: str) -> Optional[int]:
        """Extract all digits from a filename as a single number."""
        filename = os.path.basename(filepath)
        name_without_ext = os.path.splitext(filename)[0]
        all_digits = re.sub(r"[^\d]", "", name_without_ext)

        if not all_digits:
            logger.debug(f"No digits found in filename: {filename}")
            return None

        return int(all_digits)

    def create_lookup_dict(
        files: List[str], method: str
    ) -> Dict[Union[int, str], str]:
        """Create a lookup dictionary based on the matching method."""
        lookup = {}
        for file in files:
            if method == "number":
                num = extract_numbers(file)
                if num is not None:
                    lookup[num] = file
            elif method == "filename":
                lookup[os.path.basename(file)] = file
        return lookup

    try:
        if normalized_method == "number":
            group1_dict = create_lookup_dict(group1_files, "number")
            group2_dict = create_lookup_dict(group2_files, "number")

            group1_numbers = set(group1_dict.keys())
            group2_numbers = set(group2_dict.keys())

            common_numbers = group1_numbers.intersection(group2_numbers)

            if not common_numbers:
                logger.warning(
                    "No matching numbers found between groups. Falling back to order matching."
                )
                return match_subjects(group1_files, group2_files, "order")

            matched_pairs = [
                (group1_dict[num], group2_dict[num]) for num in common_numbers
            ]
            matched_pairs.sort(key=lambda x: extract_numbers(x[0]))

        elif normalized_method == "filename":
            group1_dict = create_lookup_dict(group1_files, "filename")
            group2_dict = create_lookup_dict(group2_files, "filename")

            common_filenames = set(group1_dict.keys()).intersection(
                set(group2_dict.keys())
            )

            if not common_filenames:
                logger.warning(
                    "No matching filenames found between groups. Falling back to number matching."
                )
                return match_subjects(group1_files, group2_files, "number")

            matched_pairs = [
                (group1_dict[name], group2_dict[name])
                for name in common_filenames
            ]
            matched_pairs.sort(key=lambda x: os.path.basename(x[0]))

        else:  # order
            min_files = min(len(group1_files), len(group2_files))
            matched_pairs = [
                (group1_files[i], group2_files[i]) for i in range(min_files)
            ]

        if matched_pairs:
            logger.info(
                f"Successfully matched {len(matched_pairs)} subject pairs "
                f"using {normalized_method} matching"
            )
            if len(matched_pairs) < 2:
                raise ValueError(
                    f"Subject matching produced only {len(matched_pairs)} matched pair(s). "
                    f"At least 2 matched pairs are required for statistical analysis."
                )
        else:
            if normalized_method != "order":
                logger.warning("Forcing order-based matching as last resort")
                return match_subjects(group1_files, group2_files, "order")
            else:
                raise ValueError(
                    f"All subject matching methods failed. "
                    f"Group 1 has {len(group1_files)} files, Group 2 has {len(group2_files)} files."
                )

        return matched_pairs

    except Exception as e:
        logger.error(f"Error in match_subjects: {str(e)}")
        raise ValueError(f"Error matching subjects: {str(e)}") from e


# =============================================================================
# LINEAR MIXED MODEL (LMM) ANALYSIS FUNCTIONS
# =============================================================================
# Note: Generic LMM helper functions (_choose_model_type, _validate_statistical_power,
# _fit_simple_poisson_glmm) are imported from combine_compare_population_data_utils.py
# =============================================================================


def _validate_statistical_power_state_epoch(
    df: pd.DataFrame,
    subject_column: str,
    measure_name: str,
    comparison_dim: str,
) -> Tuple[bool, str]:
    """Check statistical power for state-epoch analysis with flexible comparison dimension.

    This is a state-epoch specific wrapper around _validate_statistical_power that supports
    both 'state' and 'epoch' as comparison dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    subject_column : str
        Subject ID column name
    measure_name : str
        Name of the measure being analyzed
    comparison_dim : str
        Comparison dimension ('state' or 'epoch')

    Returns
    -------
    Tuple[bool, str]
        (has_concerns flag, warning message)

    """
    warnings_list = []
    n_subjects = df[subject_column].nunique()
    n_conditions = (
        df[comparison_dim].nunique() if comparison_dim in df.columns else 1
    )

    if n_subjects < MIN_SUBJECTS_REQUIRED:
        warnings_list.append(
            f"Insufficient subjects ({n_subjects}) for analysis"
        )
        return True, "; ".join(warnings_list)
    elif n_subjects < MIN_SUBJECTS_CAUTIOUS:
        warnings_list.append(
            f"Low subject count ({n_subjects}) - interpret with caution"
        )
    elif n_subjects < MIN_SUBJECTS_RECOMMENDED:
        warnings_list.append(
            f"Moderate subject count ({n_subjects}) - limited generalizability"
        )

    obs_per_subject = df.groupby(subject_column).size()
    min_obs = obs_per_subject.min()
    max_obs = obs_per_subject.max()

    condition_counts = (
        df.groupby(subject_column)[comparison_dim].nunique()
        if comparison_dim in df.columns
        else pd.Series(dtype=int)
    )
    min_conditions_per_subject = (
        condition_counts.min() if not condition_counts.empty else n_conditions
    )

    if min_conditions_per_subject < n_conditions:
        warnings_list.append(
            f"Incomplete data: some subjects have {min_conditions_per_subject}"
            f" {comparison_dim} levels but {n_conditions} exist overall"
        )

    if max_obs / min_obs > IMBALANCED_DESIGN_RATIO_THRESHOLD:
        warnings_list.append(
            f"Highly imbalanced design (ratio: {max_obs/min_obs:.1f}) - "
            f"results may be unreliable"
        )

    return len(warnings_list) > 0, "; ".join(warnings_list)


@_suppress_pingouin_warnings
def _perform_state_epoch_lmm_analysis(
    df: pd.DataFrame,
    dv: str,
    comparison_dim: str,
    subject: str,
    between: Optional[str] = None,
    effect_size: str = "hedges",
    correction: str = "bonf",
    has_single_group: bool = True,
    measure_name: str = "activity",
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
) -> Tuple[pd.DataFrame, bool, List[str]]:
    """Perform Linear Mixed Model analysis on state-epoch comparison data.

    This function performs LMM analysis with proper random effects structure
    (subject as random intercept). It automatically detects and handles both
    cell-level data (multiple observations per subject-condition) and subject-level
    data (one observation per subject-condition). For both types, it handles
    single-group and multi-group comparisons, with the comparison dimension
    being either 'state' or 'epoch'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with either cell-level or subject-level data
    dv : str
        Dependent variable column name
    comparison_dim : str
        Comparison dimension: 'state' or 'epoch'
    subject : str
        Subject ID column name (used for random intercepts)
    between : str, optional
        Between-subject factor (usually 'group')
    effect_size : str, optional
        Effect size measure, by default "hedges"
    correction : str, optional
        Multiple comparison correction, by default "bonf"
    has_single_group : bool, optional
        Whether analysis is for single group, by default True
    measure_name : str, optional
        Name of the measure being analyzed, by default "activity"
    group_comparison_type : str, optional
        Type of statistical tail to use for post-hoc tests, by default "two_tailed"
    parametric : str, optional
        Parametric testing mode for post-hoc tests, by default "auto"

    Returns
    -------
    Tuple[pd.DataFrame, bool, List[str]]
        LMM fixed effects results, reliability warning flag, and warning messages

    """
    # Initialize warning messages list
    lmm_warning_messages = []

    try:
        # Enhanced detection of nested data structure
        obs_per_subject_condition = df.groupby(
            [subject] + [comparison_dim] + ([between] if between else [])
        ).size()

        has_nested_data = (obs_per_subject_condition > 1).any()
        max_obs_per_condition = obs_per_subject_condition.max()

        # Also check for explicit cell column as additional indicator
        has_cell_column = "cell" in df.columns
        if has_cell_column:
            cells_per_subject_condition = df.groupby(
                [subject] + [comparison_dim] + ([between] if between else [])
            )["cell"].nunique()
            has_multiple_cells = (cells_per_subject_condition > 1).any()
        else:
            has_multiple_cells = False

        # Determine if we have nested structure
        has_nested_structure = has_nested_data or has_multiple_cells

        # Calculate basic data info
        n_subjects = df[subject].nunique()
        n_observations = len(df)

        # Use module-level deduplication for LMM data structure
        structure_key = f"lmm_structure_{n_subjects}_{n_observations}_{has_nested_structure}"
        if structure_key not in _global_lmm_structure_logged:
            logger.info(
                f"LMM data structure: {n_subjects} subjects, {n_observations} observations"
            )
            if has_nested_structure:
                logger.info(
                    f"  - Cell-level data (max {max_obs_per_condition} per subject-condition) "
                    f"- using proper random effects"
                )
            else:
                logger.info("  - Subject-level data - using standard LMM")
            _global_lmm_structure_logged.add(structure_key)
        else:
            logger.debug(
                f"LMM data structure (deduplicated): {n_subjects} subjects, "
                f"{n_observations} observations"
            )

        # Drop rows with missing values in critical columns
        critical_cols = [comparison_dim, subject]
        if between is not None:
            critical_cols.append(between)

        df = df.dropna(subset=critical_cols)

        if df.empty:
            logger.warning(
                f"No {measure_name} data remaining after removing missing values"
            )
            return pd.DataFrame(), False, lmm_warning_messages

        # Ensure all factor columns are strings
        df[comparison_dim] = df[comparison_dim].astype(str)
        if between is not None:
            df[between] = df[between].astype(str)
        df[subject] = df[subject].astype(str)

        # Handle name column if needed for other purposes
        if "name" in df.columns:
            df["name"] = df["name"].fillna("unknown").astype(str)

        # Check for sufficient unique values in each factor
        for col in critical_cols:
            if col != subject and df[col].nunique() < 2:
                logger.warning(
                    f"Factor column '{col}' has less than 2 unique values for {measure_name}"
                )
                return pd.DataFrame(), False, lmm_warning_messages
            elif col == subject and df[col].nunique() < 2:
                logger.warning(
                    f"Subject column '{col}' has less than 2 unique values for {measure_name}"
                )
                return pd.DataFrame(), False, lmm_warning_messages

        # Relaxed minimum requirements for LMM - allow analysis to proceed with warnings
        if len(df) < MIN_OBSERVATIONS_LMM:
            logger.warning(
                f"Very low observation count for {measure_name} LMM: {len(df)} < "
                f"{MIN_OBSERVATIONS_LMM} observations. Analysis may be unreliable."
            )
            return pd.DataFrame(), False, lmm_warning_messages
        elif len(df) < MIN_OBSERVATIONS_CAUTIOUS_LMM:
            logger.warning(
                f"Low observation count for {measure_name} LMM: {len(df)} observations. "
                f"Results should be interpreted with caution."
            )
            lmm_warning_messages.append(
                "Low observation count - results may be less reliable"
            )

        # Ensure the dependent variable is numeric and handle NaN values
        df[dv] = pd.to_numeric(df[dv], errors="coerce")
        df = df.dropna(subset=[dv])

        if len(df) < MIN_OBSERVATIONS_LMM:
            logger.warning(
                f"Insufficient {measure_name} data after cleaning dependent "
                f"variable"
            )
            return pd.DataFrame(), False, lmm_warning_messages

        # Check for sufficient variation in the data
        if df[dv].std() == 0:
            logger.warning(
                f"No variation in {measure_name} dependent variable"
            )
            return pd.DataFrame(), False, lmm_warning_messages

        # Check for minimum observations per subject - relaxed for single group analysis
        obs_per_subject = df.groupby(subject).size()
        min_obs_per_subject = obs_per_subject.min()

        if min_obs_per_subject < 1:
            logger.warning(
                f"No valid {measure_name} observations for some subjects"
            )
            return pd.DataFrame(), False, lmm_warning_messages
        elif min_obs_per_subject == 1 and has_single_group:
            logger.warning(
                f"Only 1 observation per subject for {measure_name} LMM. "
                f"Results may be less reliable than with multiple observations per subject."
            )
            # Continue with analysis despite low observations per subject in single group case

        # Choose appropriate model type based on data characteristics
        model_type = _choose_model_type(df, dv, measure_name)

        # Apply data transformations if needed
        if model_type == "logit_transform":
            # Apply logit transformation for proportional data
            # Add small epsilon to handle 0 and 1 values
            epsilon = 1e-6
            df[dv] = df[dv].clip(epsilon, 1 - epsilon)
            df[f"{dv}_logit"] = np.log(df[dv] / (1 - df[dv]))
            dv = f"{dv}_logit"
        elif model_type == "poisson_glmm":
            # For Poisson GLMM, ensure integer values and log info
            df[dv] = np.round(df[dv]).astype(int)

        # Build the formula for the LMM dynamically (fixed effects part)
        formula = f"{dv} ~ 1"
        all_factors = [comparison_dim] + ([between] if between else [])

        # Add main effects
        if all_factors:
            formula += " + " + " + ".join(all_factors)

        # Add interaction term if we have both within and between factors
        if comparison_dim and between:
            formula += f" + {comparison_dim}:{between}"

        logger.debug(f"LMM formula: {formula} | Random: (1|{subject})")

        # =============================================================================
        # MODEL FITTING WITH DATA-TYPE-SPECIFIC APPROACH
        # =============================================================================

        # Fit the appropriate model based on data type
        result = None
        if model_type == "poisson_glmm":
            # Try Poisson GLMM first, fallback to LMM if it fails
            poisson_result, model_data = _fit_simple_poisson_glmm(
                df, formula, subject, dv
            )
            if poisson_result is not None:
                # Use Poisson results
                result = poisson_result
                df = model_data
                model = None
                methods = []
                logger.info(f"Using Poisson GLM results for {measure_name}")
            else:
                # Fallback to standard LMM
                model = MixedLM.from_formula(
                    formula=formula,
                    data=df,
                    groups=df[subject],
                )
        else:
            # Standard LMM or logit-transformed LMM
            model = MixedLM.from_formula(
                formula=formula,
                data=df,
                groups=df[subject],
            )

        # Try different optimization methods with increased max iterations
        # (Skip this if we already have Poisson results)
        if model_type != "poisson_glmm" or result is None:
            methods = ["lbfgs", "bfgs"]
            convergence_issues = []
            fit_warnings = []
        else:
            methods = []
            convergence_issues = []
            fit_warnings = []

        # Catch warnings specifically during the fitting process
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", UserWarning)
            try:
                warnings.simplefilter("always", ConvergenceWarning)
            except ImportError:
                pass

            # Only attempt fitting if we have a valid model object
            for method in methods:
                if model is None:
                    logger.warning(
                        f"Model object is None, cannot fit using {method} for {measure_name}"
                    )
                    break
                try:
                    logger.debug(
                        f"Attempting LMM fit for {measure_name} using {method} optimizer"
                    )
                    # Use reml=True by default for better variance estimates
                    result = model.fit(method=method, maxiter=1000, reml=True)
                    if result.converged:
                        break
                    else:
                        convergence_issues.append(
                            f"Method {method} did not converge (converged=False)"
                        )
                        logger.debug(
                            f"Method {method} failed to converge for {measure_name}"
                        )
                except np.linalg.LinAlgError as lae:
                    convergence_issues.append(
                        f"Method {method} failed with LinAlgError: {str(lae)}"
                    )
                    logger.debug(
                        f"Method {method} failed with linear algebra error for {measure_name}"
                    )
                except Exception as e:
                    convergence_issues.append(
                        f"Method {method} failed: {type(e).__name__}: {str(e)}"
                    )
                    logger.debug(
                        f"Method {method} failed with exception for {measure_name}: {str(e)}"
                    )
                    continue

            # If first attempts failed, try others
            if result is None or not result.converged:
                if model is None:
                    logger.warning(
                        (
                            "Model object is None, cannot try additional "
                            f"optimization methods for {measure_name}"
                        )
                    )
                else:
                    logger.debug(
                        f"Trying additional optimization methods for {measure_name}"
                    )
                    methods_extra = ["cg", "powell"]
                    for method in methods_extra:
                        try:
                            logger.debug(
                                f"Attempting LMM fit for {measure_name} using {method} optimizer"
                            )
                            result = model.fit(
                                method=method, maxiter=1000, reml=True
                            )
                            if result.converged:
                                logger.info(
                                    f"LMM converged successfully using {method}"
                                )
                                break
                            else:
                                convergence_issues.append(
                                    f"Method {method} did not converge (converged=False)"
                                )
                        except np.linalg.LinAlgError as lae:
                            convergence_issues.append(
                                f"Method {method} failed with LinAlgError: {str(lae)}"
                            )
                        except Exception as e:
                            convergence_issues.append(
                                f"Method {method} failed: {type(e).__name__}: {str(e)}"
                            )
                            continue
                        # Break outer loop if converged
                        if result is not None and result.converged:
                            break

            # Store caught warnings relevant to statsmodels LMM
            fit_warnings.extend(
                [
                    w
                    for w in caught_warnings
                    if "statsmodels" in str(w.filename)
                ]
            )

        # Enhanced warning detection and reporting
        singular_cov_warn = any(
            "covariance is singular" in str(w.message).lower()
            for w in fit_warnings
        )
        boundary_warn = any(
            "on the boundary" in str(w.message).lower() for w in fit_warnings
        )

        # Check for specific convergence/reliability issues
        model_converged = result is not None and result.converged
        has_invalid_se = False

        # Relaxed check: Detect singular covariance but continue with warning
        has_singular_covariance = singular_cov_warn
        if (
            result is not None
            and hasattr(result, "cov_params")
            and result.cov_params is not None
        ):
            try:
                # Safely check condition number - lower threshold for better detection
                if (
                    hasattr(result.cov_params, "shape")
                    and len(result.cov_params.shape) >= 2
                ):
                    cond_num = np.linalg.cond(result.cov_params)
                    # More sensitive detection: use 1e8 instead of 1e12
                    has_singular_covariance = (
                        has_singular_covariance or cond_num > 1e8
                    )

                    # Additional checks for singular covariance
                    # Check for near-zero determinant
                    try:
                        det = np.linalg.det(result.cov_params)
                        if abs(det) < 1e-12:
                            has_singular_covariance = True
                    except Exception:
                        pass

                    # Check for invalid standard errors
                    if hasattr(result, "bse") and result.bse is not None:
                        invalid_se_check = (
                            np.isnan(result.bse).any()
                            or np.isinf(result.bse).any()
                            or (result.bse <= 0).any()
                        )
                        has_invalid_se = invalid_se_check
                        if has_invalid_se:
                            has_singular_covariance = True
            except Exception as e:
                logger.debug(
                    f"Could not fully check covariance properties: {str(e)}"
                )

        # Overall reliability check
        has_reliability_concerns = (
            not model_converged or has_singular_covariance or has_invalid_se
        )

        # Build comprehensive warning messages
        if not model_converged:
            lmm_warning_messages.append(
                f"LMM did not converge for {measure_name} - results may be unreliable"
            )
            if convergence_issues:
                logger.debug(
                    f"Convergence issues for {measure_name}: {convergence_issues[:2]}"
                )

        if has_singular_covariance:
            lmm_warning_messages.append(
                f"Singular covariance matrix detected for {measure_name} - "
                f"standard errors may be unreliable"
            )

        if has_invalid_se:
            lmm_warning_messages.append(
                f"Invalid standard errors detected for {measure_name}"
            )

        if boundary_warn:
            lmm_warning_messages.append(
                f"Variance component on boundary for {measure_name} - "
                f"model may be overparameterized"
            )

        # Format the LMM result into a standardized DataFrame
        if result is not None:
            lmm_df = pd.DataFrame(
                {
                    "Source": result.params.index,
                    "Coefficient": result.params.values,
                    "SE": result.bse.values,
                    "t": result.tvalues.values,
                    "p-unc": result.pvalues.values,
                    "Measure": measure_name,
                }
            )

            # Add convergence status
            lmm_df["converged"] = result.converged

            logger.info(
                f"LMM analysis completed for {measure_name} "
                f"(converged={result.converged}, "
                f"concerns={has_reliability_concerns})"
            )

            return lmm_df, has_reliability_concerns, lmm_warning_messages
        else:
            logger.warning(f"LMM analysis failed for {measure_name}")
            return pd.DataFrame(), True, lmm_warning_messages

    except Exception as e:
        logger.error(
            f"Error in LMM analysis for {measure_name}: {str(e)}\n{traceback.format_exc()}"
        )
        lmm_warning_messages.append(f"LMM analysis error: {str(e)}")
        return pd.DataFrame(), True, lmm_warning_messages


@_suppress_pingouin_warnings
def calculate_state_epoch_lmm_stats(
    df: pd.DataFrame,
    comparison_dim: str,
    comparison_values: List[str],
    has_single_group: bool,
    data_pairing: str = "unpaired",
    effect_size: str = "hedges",
    correction: str = "bonf",
    measure_name: str = "activity",
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Linear Mixed Model statistics for state-epoch comparison data.

    This function performs Linear Mixed Model analysis on state-epoch data with
    proper random effects (subject as random intercept). It automatically detects
    and handles both cell-level data (multiple cells per subject) and subject-level
    data (aggregated to one value per subject per condition). It's designed to work
    with both state and epoch comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing state-epoch data (cell-level or subject-level)
    comparison_dim : str
        Comparison dimension: 'state' or 'epoch'
    comparison_values : List[str]
        List of comparison values (states or epochs) to analyze
    has_single_group : bool
        Whether analysis is for a single group or multiple groups
    data_pairing : str, optional
        Data pairing type ("paired" or "unpaired"), by default "unpaired"
    effect_size : str, optional
        Effect size measure, by default "hedges"
    correction : str, optional
        Multiple comparison correction method, by default "bonf"
    measure_name : str, optional
        Name of the measure being analyzed, by default "activity"
    significance_threshold : float, optional
        Significance threshold, by default DEFAULT_SIGNIFICANCE_THRESHOLD
    group_comparison_type : str, optional
        Statistical tail to use for pairwise post-hoc tests ("two_tailed",
        "one_tailed_less", or "one_tailed_greater"), by default "two_tailed"
    parametric : str, optional
        Parametric testing mode for post-hoc tests ("auto", "True", or "False"),
        by default "auto"

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        LMM fixed effects results and pairwise comparison results

    """
    try:
        # Limit analysis to explicitly requested comparison values when provided.
        # This mirrors the ANOVA workflow and prevents unexpected factor levels
        # from influencing the LMM while still permitting partial subject overlap.
        if comparison_values:
            available_values = df[comparison_dim].dropna().unique().tolist()
            filtered_values = [
                value
                for value in comparison_values
                if value in available_values
            ]
            if not filtered_values:
                logger.warning(
                    "No matching %s values found for requested comparison set %s",
                    comparison_dim,
                    comparison_values,
                )
                return pd.DataFrame(), pd.DataFrame()
            if set(filtered_values) != set(comparison_values):
                missing_values = set(comparison_values) - set(filtered_values)
                logger.info(
                    "Skipping missing %s levels in LMM analysis: %s",
                    comparison_dim,
                    sorted(missing_values),
                )
            df = df[df[comparison_dim].isin(filtered_values)].copy()
        else:
            filtered_values = df[comparison_dim].dropna().unique().tolist()

        # Input validation with enhanced logging
        if df.empty:
            logger.warning(f"Empty dataframe for {measure_name}")
            return pd.DataFrame(), pd.DataFrame()

        if "normalized_subject_id" not in df.columns:
            logger.error(
                f"Missing normalized_subject_id column for {measure_name}"
            )
            return pd.DataFrame(), pd.DataFrame()

        if comparison_dim not in df.columns:
            logger.error(f"Missing {comparison_dim} column for {measure_name}")
            return pd.DataFrame(), pd.DataFrame()

        # Determine the dependent variable column
        dv_column = "activity"
        if "activity" in df.columns:
            dv_column = "activity"
        elif measure_name in df.columns:
            dv_column = measure_name
        else:
            # Look for numeric columns that might be the DV
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            value_cols = [
                col
                for col in numeric_cols
                if "value" in col.lower() or "measure" in col.lower()
            ]
            if value_cols:
                dv_column = value_cols[0]
                logger.info(
                    f"Auto-detected dependent variable for {measure_name}: "
                    f"{dv_column}"
                )
            else:
                logger.error(
                    f"Could not determine dependent variable column for "
                    f"{measure_name}"
                )
                return pd.DataFrame(), pd.DataFrame()

        # Check statistical power and add warnings to results if needed
        has_concerns, power_warning = _validate_statistical_power_state_epoch(
            df, "normalized_subject_id", measure_name, comparison_dim
        )

        between_factor = None if has_single_group else "group_name"

        (
            lmm_result,
            has_reliability_concerns,
            lmm_warning_messages,
        ) = _perform_state_epoch_lmm_analysis(
            df=df,
            dv=dv_column,
            comparison_dim=comparison_dim,
            subject="normalized_subject_id",
            between=between_factor,
            effect_size=effect_size,
            correction=correction,
            has_single_group=has_single_group,
            measure_name=measure_name,
            group_comparison_type=group_comparison_type,
            parametric=parametric,
        )

        warning_messages: List[str] = []
        if has_concerns and power_warning:
            warning_messages.append(str(power_warning))
        warning_messages.extend(lmm_warning_messages)
        warning_text = _combine_warning_messages(warning_messages)

        reliability_note = ""
        if has_reliability_concerns:
            reliability_note = (
                "LMM flagged potential reliability concerns; "
                "interpret results with caution."
            )

        # Use existing ANOVA-based pairwise comparison function for consistency
        # LMM gives us overall fixed effects, but pairwise comparisons are still
        # useful and can be computed using subject-averaged data
        if not lmm_result.empty and not has_single_group:
            # Perform pairwise comparisons using the existing function
            pairwise_result = _perform_pairwise_tests(
                data=df,
                subject_col="normalized_subject_id",
                group_col="group_name",
                comparison_col=comparison_dim,
                comparison_values=filtered_values,
                measure_col=dv_column,
                correction=correction,
                effect_size=effect_size,
                data_pairing=data_pairing,
                group_comparison_type=group_comparison_type,
                parametric=parametric,
                context=f"{comparison_dim} LMM pairwise",
                allow_partial_subjects=True,
            )
        else:
            pairwise_result = pd.DataFrame()

        if not lmm_result.empty:
            lmm_result = _merge_text_column_with_message(
                lmm_result, "statistical_warnings", warning_text
            )
            lmm_result = _merge_text_column_with_message(
                lmm_result, "analysis_notes", reliability_note
            )

        if not pairwise_result.empty:
            pairwise_result = _merge_text_column_with_message(
                pairwise_result, "statistical_warnings", warning_text
            )
            pairwise_result = _merge_text_column_with_message(
                pairwise_result, "analysis_notes", reliability_note
            )

        return lmm_result, pairwise_result

    except Exception as e:
        logger.error(
            f"Error in calculate_state_epoch_lmm_stats for {measure_name}: "
            f"{str(e)}\n{traceback.format_exc()}"
        )
        return pd.DataFrame(), pd.DataFrame()


def _detect_measure_column(
    data: pd.DataFrame, data_type: str, correlation_statistic: str = "mean"
) -> str:
    """Detect the appropriate measure column based on data type and available columns.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing measure columns
    data_type : str
        Type of data: "activity", "correlation", "modulation", etc.
    correlation_statistic : str, optional
        Which correlation statistic to use: "max", "min", or "mean"
        (default: "mean")

    Returns
    -------
    str
        Name of the detected measure column

    Raises
    ------
    IdeasError
        If no appropriate measure column can be found

    """
    available_columns = set(data.columns)

    normalized_type = data_type.lower().strip()
    # Normalize common synonyms so that "modulated" variants map to modulation logic
    if "modulated" in normalized_type and "modulation" not in normalized_type:
        normalized_type = normalized_type.replace("modulated", "modulation")
    if "activity" in normalized_type:
        base_type = "activity"
    elif "correlation" in normalized_type:
        base_type = "correlation"
    elif "modulation" in normalized_type:
        base_type = "modulation"
    else:
        base_type = normalized_type

    # Define column priority lists based on data type
    if base_type == "activity":
        # Priority order: trace activity, event rate
        candidate_columns = [
            "mean_trace_activity",
            "mean_activity",
            "mean_event_rate",
            "median_trace_activity",
        ]
    elif base_type == "correlation":
        # Determine correlation column based on statistic and data type
        # Extract source (trace/event) from data_type if present
        if "trace" in normalized_type:
            source = "trace"
        elif "event" in normalized_type:
            source = "event"
        else:
            source = None

        normalized_stat = str(correlation_statistic).strip().lower()
        if normalized_stat not in {"max", "min", "mean"}:
            normalized_stat = "mean"

        # Build candidate list based on data_type and correlation_statistic
        if "positive" in normalized_type:
            if source == "trace":
                candidate_columns = ["positive_trace_correlation"]
            elif source == "event":
                candidate_columns = ["positive_event_correlation"]
            else:
                candidate_columns = [
                    "positive_trace_correlation",
                    "positive_correlation",
                    "positive_event_correlation",
                ]
        elif "negative" in normalized_type:
            if source == "trace":
                candidate_columns = ["negative_trace_correlation"]
            elif source == "event":
                candidate_columns = ["negative_event_correlation"]
            else:
                candidate_columns = [
                    "negative_trace_correlation",
                    "negative_correlation",
                    "negative_event_correlation",
                ]
        elif normalized_stat == "max":
            if source == "trace":
                candidate_columns = ["max_trace_correlation"]
            elif source == "event":
                candidate_columns = ["max_event_correlation"]
            else:
                candidate_columns = [
                    "max_trace_correlation",
                    "max_correlation",
                    "max_event_correlation",
                ]
        elif normalized_stat == "min":
            if source == "trace":
                candidate_columns = ["min_trace_correlation"]
            elif source == "event":
                candidate_columns = ["min_event_correlation"]
            else:
                candidate_columns = [
                    "min_trace_correlation",
                    "min_correlation",
                    "min_event_correlation",
                ]
        else:  # mean or default
            if source == "trace":
                candidate_columns = ["mean_trace_correlation"]
            elif source == "event":
                candidate_columns = ["mean_event_correlation"]
            else:
                candidate_columns = [
                    "mean_trace_correlation",
                    "mean_correlation",
                    "mean_event_correlation",
                ]

        # Add fallback options
        candidate_columns.extend(
            [
                "mean_trace_correlation",
                "mean_correlation",
                "mean_event_correlation",
                "max_trace_correlation",
                "max_event_correlation",
                "min_trace_correlation",
                "min_event_correlation",
                "positive_trace_correlation",
                "positive_correlation",
                "positive_event_correlation",
                "negative_trace_correlation",
                "negative_correlation",
                "negative_event_correlation",
            ]
        )
    elif base_type == "modulation":
        # Handle modulation counts (up/down) as well as modulation scores
        if "up" in normalized_type and "count" in normalized_type:
            # Up-modulated cell counts
            if "trace" in normalized_type:
                candidate_columns = ["trace_up_modulation_number"]
            elif "event" in normalized_type:
                candidate_columns = ["event_up_modulation_number"]
            else:
                candidate_columns = [
                    "trace_up_modulation_number",
                    "event_up_modulation_number",
                ]
        elif "down" in normalized_type and "count" in normalized_type:
            # Down-modulated cell counts
            if "trace" in normalized_type:
                candidate_columns = ["trace_down_modulation_number"]
            elif "event" in normalized_type:
                candidate_columns = ["event_down_modulation_number"]
            else:
                candidate_columns = [
                    "trace_down_modulation_number",
                    "event_down_modulation_number",
                ]
        else:
            # Priority order: trace or event modulation scores
            candidate_columns = [
                "trace_modulation",
                "event_modulation",
            ]
    else:
        # For unknown types, look for common patterns
        candidate_columns = [
            "mean_trace_activity",
            "mean_activity",
            "mean_trace_correlation",
            "mean_correlation",
        ]

    # Identify numeric columns (including nullable numeric dtypes)
    numeric_columns = set(
        col
        for col, dtype in data.dtypes.items()
        if pd.api.types.is_numeric_dtype(dtype)
    )

    # Find the first available candidate column that is numeric
    for col in candidate_columns:
        if col in available_columns and col in numeric_columns:
            return col

    # If no standard columns found, look for any column with relevant keywords
    if base_type == "activity":
        for col in available_columns & numeric_columns:
            if "activity" in col.lower() or "rate" in col.lower():
                return col
    elif base_type == "correlation":
        for col in available_columns & numeric_columns:
            if "correlation" in col.lower():
                return col
    elif base_type == "modulation":
        for col in available_columns & numeric_columns:
            if "modulation" in col.lower():
                return col

    # If still no column found, raise error
    raise IdeasError(
        f"Unable to determine measure column for {data_type} plot. "
        f"Available columns: {list(data.columns)}"
    )


def plot_state_epoch_comparison(
    data: pd.DataFrame,
    group_names: List[str],
    group_colors: List[str],
    states: List[str],
    epochs: List[str],
    baseline_state: str,
    baseline_epoch: str,
    comparison_dimension: str,
    data_pairing: str = "unpaired",
    state_colors: Optional[Dict[str, str]] = None,
    epoch_colors: Optional[Dict[str, str]] = None,
    pairwise_results: Optional[pd.DataFrame] = None,
    output_dir: str = None,
    output_filename: str = None,
    data_type: str = "activity",
    correlation_statistic: str = "mean",
) -> None:
    """Create comparison plot for state-epoch data.

    This visualization function maintains consistency with combine_compare_population_data
    and combine_compare_correlation_data tools by:
    - Supporting single group, two-group unpaired, and two-group paired analyses
    - Using bar plots with error bars (SEM) for group comparisons

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot with columns: state, epoch, group_name, normalized_subject_id, measure values
    group_names : List[str]
        Names of groups
    group_colors : List[str]
        Colors for groups
    states : List[str]
        List of state names
    epochs : List[str]
        List of epoch names
    baseline_state : str
        Baseline state name
    baseline_epoch : str
        Baseline epoch name
    comparison_dimension : str
        Dimension to compare: "states" or "epochs"
    data_pairing : str
        Type of data pairing ("paired" or "unpaired"), default "unpaired".
        Preserved for API compatibility with previous scatter plot styling.
    state_colors : Optional[Dict[str, str]]
        Colors for states
    epoch_colors : Optional[Dict[str, str]]
        Colors for epochs
    pairwise_results : Optional[pd.DataFrame]
        Pairwise comparison results for significance annotations
    output_dir : str
        Output directory path
    output_filename : str
        Output file path (overrides output_dir if specified)
    data_type : str
        Type of data being plotted
    correlation_statistic : str
        Statistic to use for correlation data, default "mean"

    """
    try:
        if data is None or data.empty:
            logger.warning(f"No data to plot for {data_type}")
            return

        def _ordered_unique(values: Iterable[Any]) -> List[Any]:
            """Return values in first-seen order, skipping NaN-like entries."""
            ordered: List[Any] = []
            for value in values:
                if pd.isna(value):
                    continue
                if value not in ordered:
                    ordered.append(value)
            return ordered

        def _apply_preferred_order(
            preferred: Optional[Iterable[Any]],
            available: List[Any],
        ) -> List[Any]:
            if preferred:
                filtered: List[Any] = []
                for value in preferred:
                    if value in available and value not in filtered:
                        filtered.append(value)
                if filtered:
                    return filtered
            return available

        # Determine what to plot on x-axis
        comp_col = comparison_dimension[:-1]  # "state" or "epoch"
        other_col = "epoch" if comp_col == "state" else "state"

        available_comp_values = _ordered_unique(data[comp_col].tolist())
        preferred_comp_values = states if comp_col == "state" else epochs
        comp_values = _apply_preferred_order(
            preferred_comp_values,
            available_comp_values,
        )
        if not comp_values:
            logger.warning(
                "No valid %s values found to plot for %s", comp_col, data_type
            )
            return

        # Get unique values
        if other_col in data.columns:
            available_other_values = _ordered_unique(data[other_col].tolist())
            preferred_other = epochs if comp_col == "state" else states
            ordered_other = _apply_preferred_order(
                preferred_other,
                available_other_values,
            )
            other_values = ordered_other if ordered_other else [None]
        else:
            other_values = [None]

        # Create subplots if we have multiple levels of other dimension
        n_subplots = len(other_values) if other_values[0] is not None else 1
        fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
        if n_subplots == 1:
            axes = [axes]

        # Collect all values across all subplots for consistent y-axis scaling
        global_min = float("inf")
        global_max = float("-inf")
        subplot_data = []

        for idx, other_val in enumerate(other_values):
            ax = axes[idx]

            # Filter data
            if other_val is not None:
                subset = data[data[other_col] == other_val]
                title = f"{data_type.capitalize()} - {other_col}={other_val}"
            else:
                subset = data
                title = f"{data_type.capitalize()}"

            # Plot data for each group
            x_positions = np.arange(len(comp_values))
            width = 0.35 if len(group_names) > 1 else 0.5

            # Detect the measure column to use
            measure_column = _detect_measure_column(
                subset, data_type, correlation_statistic
            )

            # Store data for this subplot
            subplot_means_sems = []

            for g_idx, group_name in enumerate(group_names):
                group_data = subset[subset["group_name"] == group_name]

                means = []
                sems = []
                for comp_val in comp_values:
                    comp_data = group_data[group_data[comp_col] == comp_val]

                    # Calculate subject-level means first to avoid pseudoreplication
                    if "normalized_subject_id" in comp_data.columns:
                        subject_means = comp_data.groupby(
                            "normalized_subject_id"
                        )[measure_column].mean()
                        values = subject_means.values
                    else:
                        values = comp_data[measure_column].values

                    mean_val = np.mean(values) if len(values) > 0 else 0
                    sem_val = stats.sem(values) if len(values) > 1 else 0

                    means.append(mean_val)
                    sems.append(sem_val)

                    # Track global min/max for y-axis scaling using finite values only
                    if len(values) > 0 and np.isfinite(mean_val):
                        lower_bound = mean_val - sem_val
                        upper_bound = mean_val + sem_val

                        if np.isfinite(lower_bound):
                            global_min = min(global_min, lower_bound)

                        if np.isfinite(upper_bound):
                            global_max = max(global_max, upper_bound)

                subplot_means_sems.append((means, sems, group_name, g_idx))

            # Store subplot data for later plotting
            subplot_data.append(
                (
                    ax,
                    subplot_means_sems,
                    x_positions,
                    width,
                    subset,
                    comp_values,
                    comp_col,
                    measure_column,
                    title,
                )
            )

        # Now plot all subplots with consistent y-axis limits
        # Add 10% padding to y-axis range
        if np.isfinite(global_min) and np.isfinite(global_max):
            y_range = global_max - global_min
            if np.isfinite(y_range) and y_range != 0:
                padding = 0.1 * y_range
            else:
                padding = 0.1 if np.isfinite(global_min) else 0

            y_min = global_min - padding
            y_max = global_max + padding
        else:
            y_min, y_max = None, None

        for (
            ax,
            subplot_means_sems,
            x_positions,
            width,
            _subset,
            comp_values,
            comp_col,
            _measure_column,
            title,
        ) in subplot_data:
            # Plot bars for each group
            for means, sems, group_name, g_idx in subplot_means_sems:
                offset = width * (g_idx - 0.5) if len(group_names) > 1 else 0
                ax.bar(
                    x_positions + offset,
                    means,
                    width,
                    yerr=sems,
                    label=group_name,
                    color=group_colors[g_idx],
                    alpha=0.7,
                    capsize=5,
                )

            # Set consistent y-axis limits across all subplots
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)

            ax.set_xlabel(comp_col.capitalize())
            ax.set_ylabel(data_type.capitalize())
            ax.set_title(title)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(comp_values, rotation=45, ha="right")
            if len(group_names) > 1:
                ax.legend()

            # Clean up spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()

        # Determine output filename
        if output_filename:
            save_path = output_filename
        elif output_dir:
            # Create output directory if needed
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(
                output_dir,
                f"{comparison_dimension}_comparison_{data_type}.svg",
            )
        else:
            save_path = None

        if save_path:
            fig.savefig(
                save_path,
                dpi=300,
                format="svg",
                bbox_inches="tight",
                transparent=True,
            )
            logger.info(f"Saved comparison plot: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    except Exception as e:
        logger.error(f"Error creating comparison plot: {e}")
