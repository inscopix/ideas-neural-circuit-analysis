"""Statistical validation utilities for population data analysis.

This module contains functions for validating and cleaning statistical data
to ensure robust analysis in the IDEAS toolbox population analysis pipeline.
"""

import inspect
import logging
import warnings
import numpy as np
import pandas as pd
import pingouin as pg
from typing import Optional, Tuple, Union, List, Any, Callable


logger = logging.getLogger(__name__)

# Default significance threshold
DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05

_PAIRWISE_TESTS_PARAMS: Tuple[str, ...]
try:
    _PAIRWISE_TESTS_PARAMS = tuple(
        inspect.signature(pg.pairwise_tests).parameters
    )
except (ValueError, TypeError):
    logger.warning(
        "Unable to inspect pingouin.pairwise_tests signature. "
        "Tail parameter compatibility will be limited."
    )
    _PAIRWISE_TESTS_PARAMS = ()

_TAIL_ARGUMENT_MAPPING = {
    "two_tailed": "two-sided",
    "one_tailed_less": "less",
    "one_tailed_greater": "greater",
    "two-sided": "two-sided",
    "less": "less",
    "greater": "greater",
}


def _suppress_pingouin_warnings(func: Callable) -> Callable:
    """Decorate to suppress specific warnings from the pingouin library.

    Parameters
    ----------
    func : callable
        Function to wrap with warning suppression

    Returns
    -------
    callable
        Wrapped function with warning suppression

    """

    def wrapper(*args, **kwargs) -> Any:
        with warnings.catch_warnings():
            # Suppress specific warnings from pingouin
            warnings.filterwarnings(
                "ignore", message="DataFrame.groupby with axis=1 is deprecated"
            )
            warnings.filterwarnings(
                "ignore",
                message="DataFrameGroupBy.diff with axis=1 is deprecated",
            )
            warnings.filterwarnings(
                "ignore",
                message="x and y are equals. Cannot compute T or p-value",
            )
            # Suppress scipy warnings about divide by zero
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in _nct_sf",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in _nct_cdf",
                category=RuntimeWarning,
            )
            # Suppress pingouin eigenvalue/sphericity calculation warnings
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in divide",
                category=RuntimeWarning,
            )
            # Suppress scipy precision loss warning
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Precision loss occurred in moment calculation due to "
                    "catastrophic cancellation"
                ),
                category=RuntimeWarning,
            )
            # Suppress pingouin invalid value warnings
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in multiply",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in sqrt",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in subtract",
                category=RuntimeWarning,
            )
            return func(*args, **kwargs)

    return wrapper


def _validate_statistical_data(data: pd.Series, name: str = "data") -> bool:
    """Validate data for statistical analysis to prevent pingouin warnings.

    Parameters
    ----------
    data : pd.Series
        Data to validate
    name : str, optional
        Name of the data for logging, by default "data"

    Returns
    -------
    bool
        True if data is valid for statistical analysis, False otherwise

    """
    if data is None or len(data) == 0:
        logger.debug(f"_validate_statistical_data: {name} is empty")
        return False

    # Check for non-finite values
    if not np.isfinite(data).all():
        non_finite_count = (~np.isfinite(data)).sum()
        logger.info(
            f"_validate_statistical_data: {name} contains {non_finite_count} non-finite values"
        )
        return False

    # Check for constant data (zero variance)
    if data.std() == 0:
        logger.info(
            f"_validate_statistical_data: {name} has zero variance (all values are {data.iloc[0]})"
        )
        return False

    # Check for sufficient sample size
    if len(data) < 2:
        logger.info(
            f"_validate_statistical_data: {name} has insufficient sample size ({len(data)})"
        )
        return False

    return True


def _clean_statistical_data(data: pd.Series, name: str = "data") -> pd.Series:
    """Clean data for statistical analysis by removing invalid values.

    Parameters
    ----------
    data : pd.Series
        Data to clean
    name : str, optional
        Name of the data for logging, by default "data"

    Returns
    -------
    pd.Series
        Cleaned data with invalid values removed

    """
    original_length = len(data)
    cleaned_data = data[np.isfinite(data)]

    if len(cleaned_data) < original_length:
        removed_count = original_length - len(cleaned_data)
        logger.info(
            f"_clean_statistical_data: Removed {removed_count} non-finite values from {name}"
        )

    return cleaned_data


@_suppress_pingouin_warnings
def _safe_ttest(
    data1: pd.Series, data2: pd.Series, paired: bool = False, **kwargs
) -> Optional[pd.DataFrame]:
    """Safely perform t-test with data validation to prevent pingouin warnings.

    Parameters
    ----------
    data1 : pd.Series
        First group data
    data2 : pd.Series
        Second group data
    paired : bool, optional
        Whether to perform paired t-test, by default False
    **kwargs
        Additional arguments passed to pg.ttest

    Returns
    -------
    Optional[pd.DataFrame]
        T-test results or None if data is invalid

    """
    # Clean and validate data
    clean_data1 = _clean_statistical_data(data1, "group1")
    clean_data2 = _clean_statistical_data(data2, "group2")

    # Validate both groups
    group1_valid = _validate_statistical_data(clean_data1, "group1")
    group2_valid = _validate_statistical_data(clean_data2, "group2")

    if not group1_valid or not group2_valid:
        validation_details = []
        if not group1_valid:
            validation_details.append(f"group1 invalid (n={len(clean_data1)})")
        if not group2_valid:
            validation_details.append(f"group2 invalid (n={len(clean_data2)})")

        logger.debug(
            f"_safe_ttest: Data validation failed ({', '.join(validation_details)}), "
            f"skipping t-test"
        )
        return None

    # For paired tests, ensure equal length
    if paired and len(clean_data1) != len(clean_data2):
        min_length = min(len(clean_data1), len(clean_data2))
        if min_length < 2:
            logger.info("_safe_ttest: Insufficient paired data after cleaning")
            return None
        clean_data1 = clean_data1.iloc[:min_length]
        clean_data2 = clean_data2.iloc[:min_length]
        logger.info(
            f"_safe_ttest: Truncated paired data to {min_length} samples"
        )

    try:
        result = pg.ttest(clean_data1, clean_data2, paired=paired, **kwargs)

        # Validate and clean results
        if result is not None and not result.empty:
            for col in ["T", "p-val", "dof"]:
                if (
                    col in result.columns
                    and not np.isfinite(result[col]).all()
                ):
                    logger.debug(
                        f"_safe_ttest: Invalid values detected in {col} column "
                        f"(automatically corrected)"
                    )
                    # Replace invalid values with appropriate defaults
                    if col == "p-val":
                        result[col] = result[col].fillna(1.0)
                    elif col == "T":
                        result[col] = result[col].fillna(0.0)
                    elif col == "dof":
                        result[col] = result[col].fillna(
                            len(clean_data1) + len(clean_data2) - 2
                        )

        return result

    except Exception as e:
        logger.debug(f"_safe_ttest: Error in t-test calculation: {str(e)}")
        return None


def _validate_pairwise_test_data(
    data: pd.DataFrame,
    dv: str,
    within: Optional[Union[str, List[str]]],
    between: Optional[str],
) -> Tuple[bool, str, pd.DataFrame]:
    """Validate and clean data for pairwise tests.

    Parameters
    ----------
    data : pd.DataFrame
        Input data for validation
    dv : str
        Dependent variable column name
    within : Optional[Union[str, List[str]]]
        Within-subject factors
    between : Optional[str]
        Between-subject factors

    Returns
    -------
    Tuple[bool, str, pd.DataFrame]
        (is_valid, error_message, cleaned_data)

    """
    if data is None or data.empty:
        return False, "Empty data provided", data

    # Validate dependent variable
    if dv and dv in data.columns:
        original_len = len(data)
        cleaned_data = data[np.isfinite(data[dv])]

        if cleaned_data.empty:
            return (
                False,
                "All values are non-finite in dependent variable",
                cleaned_data,
            )

        if len(cleaned_data) < original_len:
            logger.debug(
                f"_validate_pairwise_test_data: Removed "
                f"{original_len - len(cleaned_data)} non-finite values"
            )

        if cleaned_data[dv].var() == 0:
            return False, "Zero variance in dependent variable", cleaned_data
    else:
        cleaned_data = data
        if dv:
            return (
                False,
                f"Dependent variable '{dv}' not found in data",
                cleaned_data,
            )

    # Validate grouping variables
    grouping_factors = []
    if within:
        if isinstance(within, str):
            grouping_factors.append(within)
        else:
            grouping_factors.extend(within)

    if between:
        grouping_factors.append(between)

    for factor in grouping_factors:
        if (
            factor in cleaned_data.columns
            and cleaned_data[factor].nunique() < 2
        ):
            return False, f"Insufficient groups in {factor}", cleaned_data

    return True, "", cleaned_data


@_suppress_pingouin_warnings
def _safe_pairwise_ttests(
    data: pd.DataFrame, **kwargs
) -> Optional[pd.DataFrame]:
    """Safely perform pairwise t-tests with enhanced validation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data for pairwise t-tests
    **kwargs
        Arguments passed to pg.pairwise_tests

    Returns
    -------
    Optional[pd.DataFrame]
        Pairwise t-test results or None if data is invalid

    """
    try:
        # Extract key parameters for validation
        dv = kwargs.get("dv")
        within = kwargs.get("within")
        between = kwargs.get("between")

        # Use validation helper function
        is_valid, error_msg, cleaned_data = _validate_pairwise_test_data(
            data, dv, within, between
        )
        if not is_valid:
            logger.debug(f"_safe_pairwise_ttests: {error_msg}")
            return None

        pairwise_kwargs = kwargs.copy()

        grouping_column = _infer_normality_group_column(
            cleaned_data, within, between
        )

        if "parametric" in pairwise_kwargs:
            pairwise_kwargs["parametric"] = _resolve_pairwise_parametric(
                cleaned_data,
                dv,
                pairwise_kwargs.get("parametric"),
                grouping_column,
            )

        normalized_tail = _normalize_tail_argument(
            pairwise_kwargs.pop("tail", None)
        )

        if normalized_tail:
            if "alternative" in _PAIRWISE_TESTS_PARAMS:
                pairwise_kwargs["alternative"] = normalized_tail
            elif "tail" in _PAIRWISE_TESTS_PARAMS:
                pairwise_kwargs["tail"] = normalized_tail
            else:
                logger.debug(
                    "_safe_pairwise_ttests: Tail argument ignored because "
                    "pingouin.pairwise_tests does not accept 'alternative' or 'tail'"
                )

        # Call pingouin with cleaned data
        result = pg.pairwise_tests(data=cleaned_data, **pairwise_kwargs)

        # Validate and clean results
        if result is not None and not result.empty:
            # Convert pingouin's "-" values in state column to empty strings for clean CSV output
            if "state" in result.columns:
                result.loc[result["state"] == "-", "state"] = ""

            effect_size_cols = ["cohen-d", "hedges", "eta-sq", "np2"]
            for col in effect_size_cols:
                if col in result.columns:
                    mask = ~np.isfinite(result[col])
                    if mask.any():
                        result.loc[mask, col] = np.nan

        return result

    except Exception as e:
        logger.debug(
            f"_safe_pairwise_ttests: Error in pairwise t-tests: {str(e)}"
        )
        return None


def _normalize_tail_argument(tail_value: Optional[str]) -> Optional[str]:
    """Normalize user-facing tail strings to pingouin-compatible values."""
    if tail_value is None:
        return None

    normalized = str(tail_value).strip().lower()
    mapped_value = _TAIL_ARGUMENT_MAPPING.get(normalized)

    if mapped_value is None:
        logger.warning(
            "_safe_pairwise_ttests: Unsupported tail value '%s'. "
            "Defaulting to two-sided comparison.",
            tail_value,
        )
        return "two-sided"

    return mapped_value


def _infer_normality_group_column(
    data: pd.DataFrame,
    within: Optional[Union[str, List[str]]],
    between: Optional[str],
) -> Optional[str]:
    """Infer an appropriate grouping column for normality testing."""
    candidate_columns: List[str] = []
    if isinstance(within, str):
        candidate_columns.append(within)
    elif isinstance(within, list):
        candidate_columns.extend(
            [factor for factor in within if isinstance(factor, str)]
        )

    if between:
        candidate_columns.append(between)

    for column in candidate_columns:
        if column in data.columns and data[column].nunique() >= 2:
            return column
    return None


def _auto_select_pairwise_parametric(
    data: pd.DataFrame,
    dv: Optional[str],
    grouping_column: Optional[str],
    min_sample_size: int = 8,
) -> bool:
    """Automatically determine whether parametric tests are appropriate."""
    if data is None or data.empty or dv is None or dv not in data.columns:
        logger.debug(
            "Auto-select parametric (pairwise): insufficient data, "
            "using non-parametric tests."
        )
        return False

    cleaned_series = data[dv].dropna()
    n_samples = len(cleaned_series)

    if n_samples < min_sample_size:
        logger.info(
            "Auto-select parametric (pairwise): fewer than %d observations "
            "found (n=%d). Non-parametric tests will be used.",
            min_sample_size,
            n_samples,
        )
        return False

    try:
        columns = [dv]
        if grouping_column and grouping_column in data.columns:
            columns.append(grouping_column)

        normality_data = data[columns].dropna()

        if normality_data.empty:
            logger.debug(
                "Auto-select parametric (pairwise): no data available after "
                "cleaning, defaulting to non-parametric tests."
            )
            return False

        if grouping_column and grouping_column in normality_data.columns:
            normality_result = pg.normality(
                data=normality_data,
                dv=dv,
                group=grouping_column,
                alpha=DEFAULT_SIGNIFICANCE_THRESHOLD,
            )
        else:
            normality_result = pg.normality(
                data=normality_data,
                dv=dv,
                alpha=DEFAULT_SIGNIFICANCE_THRESHOLD,
            )

        parametric = normality_result["normal"].all()

        if parametric:
            logger.info(
                "Auto-select parametric (pairwise): data appears normally "
                "distributed. Parametric tests will be used."
            )
        else:
            logger.info(
                "Auto-select parametric (pairwise): data is not normally "
                "distributed. Non-parametric tests will be used."
            )

        return bool(parametric)

    except Exception as error:
        logger.warning(
            "Auto-select parametric (pairwise): normality test failed (%s). "
            "Defaulting to non-parametric tests.",
            error,
        )
        return False


def _resolve_pairwise_parametric(
    data: pd.DataFrame,
    dv: Optional[str],
    parametric_value: Union[str, bool, None],
    grouping_column: Optional[str],
) -> bool:
    """Resolve parametric flag, handling string inputs and auto-selection."""
    if isinstance(parametric_value, str):
        normalized = parametric_value.strip().lower()
        if normalized == "auto" or not normalized:
            return _auto_select_pairwise_parametric(
                data, dv, grouping_column
            )
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        logger.warning(
            "_safe_pairwise_ttests: Invalid parametric value '%s'. "
            "Using auto-selection.",
            parametric_value,
        )
        return _auto_select_pairwise_parametric(data, dv, grouping_column)

    if parametric_value is None:
        return _auto_select_pairwise_parametric(data, dv, grouping_column)

    return bool(parametric_value)


def _get_group_column(df: pd.DataFrame) -> Optional[str]:
    """Get the appropriate group column name from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for group columns

    Returns
    -------
    Optional[str]
        Group column name or None if not found

    """
    for col in ["group", "group_name"]:
        if col in df.columns:
            return col
    return None


def _has_sufficient_paired_data(
    df: pd.DataFrame, subject_id_col: str, group_column: str = "group"
) -> bool:
    """Check if there's sufficient data for paired analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing processed event rate data
    subject_id_col : str
        Column name for subject identification
    group_column : str, optional
        Column name for group identification, by default "group"

    Returns
    -------
    bool
        True if there's sufficient data for paired analysis, False otherwise

    """
    # Try to find the appropriate group column
    actual_group_col = _get_group_column(df)
    if actual_group_col is None:
        # If no group column, just count total unique subjects
        return df[subject_id_col].nunique() >= 2 and df["state"].nunique() >= 1

    # Check subject counts per group
    subjects_per_group = df.groupby(actual_group_col)[subject_id_col].nunique()
    states_count = df["state"].nunique()

    # We need at least 2 subjects per group and at least 1 state
    return (
        df[subject_id_col].nunique() >= 2
        and all(count >= 2 for count in subjects_per_group.values)
        and states_count >= 1
    )


def _validate_anova_data(
    group_df: pd.DataFrame,
    analysis_type: str = "ANOVA",
    measure_column: str = "activity",
) -> Tuple[bool, str]:
    """Validate data for ANOVA analysis (both paired and mixed).

    Parameters
    ----------
    group_df : pd.DataFrame
        Subject-averaged data for analysis
    analysis_type : str, optional
        Type of analysis ("paired_rm_anova", "mixed_anova", or "ANOVA")
    measure_column : str, optional
        Column name containing the measure values, by default "activity"

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)

    """
    # Common validations for all ANOVA types
    if len(group_df) < 4:
        return (
            False,
            f"Insufficient data for {analysis_type}: less than 4 observations",
        )

    # Check if measure column exists
    if measure_column not in group_df.columns:
        return (
            False,
            f"Measure column '{measure_column}' not found in data for {analysis_type}",
        )

    if group_df[measure_column].var() == 0:
        return (
            False,
            f"Insufficient data for {analysis_type}: zero variance in {measure_column}",
        )

    # Check for group column
    group_col = _get_group_column(group_df)
    if group_col is None or group_df[group_col].nunique() < 2:
        return (
            False,
            f"Insufficient data for {analysis_type}: less than 2 groups",
        )

    # Check for required columns
    if "normalized_subject_id" not in group_df.columns:
        return (
            False,
            f"Required column 'normalized_subject_id' not found in data for {analysis_type}",
        )

    if "state" not in group_df.columns:
        return (
            False,
            f"Required column 'state' not found in data for {analysis_type}",
        )

    n_subjects = group_df["normalized_subject_id"].nunique()
    if n_subjects < 2:
        return (
            False,
            f"Insufficient data for {analysis_type}: less than 2 subjects ({n_subjects})",
        )

    # Type-specific validations
    if analysis_type == "paired_rm_anova":
        unique_states = group_df["state"].unique()
        is_pairwise = any(" vs " in str(state) for state in unique_states)

        if is_pairwise:
            if group_df["state"].nunique() < 1:
                return (
                    False,
                    "Insufficient data for paired rm_anova: no pairwise comparisons",
                )
            logger.info(
                f"Pairwise comparison data detected for paired analysis with "
                f"{group_df['state'].nunique()} comparisons"
            )
        else:
            if group_df["state"].nunique() < 2:
                return (
                    False,
                    "Insufficient data for paired rm_anova: less than 2 states",
                )

    elif analysis_type == "mixed_anova":
        group_state_counts = group_df.groupby([group_col, "state"]).size()
        if group_state_counts.min() < 1:
            return (
                False,
                "Insufficient data for mixed ANOVA: empty group-state combinations",
            )

    return True, ""


def _validate_paired_anova_data(
    group_df: pd.DataFrame, measure_column: str = "activity"
) -> Tuple[bool, str]:
    """Validate data for paired ANOVA analysis."""
    return _validate_anova_data(group_df, "paired_rm_anova", measure_column)


def _validate_mixed_anova_data(
    group_df: pd.DataFrame, measure_column: str = "activity"
) -> Tuple[bool, str]:
    """Validate data for mixed ANOVA analysis."""
    return _validate_anova_data(group_df, "mixed_anova", measure_column)


def _clean_non_finite_values(
    group_df: pd.DataFrame,
    analysis_type: str,
    measure_column: str = "activity",
) -> pd.DataFrame:
    """Clean non-finite values from data.

    Parameters
    ----------
    group_df : pd.DataFrame
        Data to clean
    analysis_type : str
        Type of analysis for logging
    measure_column : str, optional
        Column name to clean, by default "activity"

    Returns
    -------
    pd.DataFrame
        Cleaned data

    """
    if measure_column not in group_df.columns:
        logger.warning(
            f"Column '{measure_column}' not found in data for "
            f"{analysis_type}, returning original data"
        )
        return group_df

    if not np.isfinite(group_df[measure_column]).all():
        logger.warning(
            f"Data validation issue for {analysis_type}: non-finite values in {measure_column}"
        )
        return group_df[np.isfinite(group_df[measure_column])]
    return group_df


def _validate_modulation_data(
    df: pd.DataFrame,
    measure_column: str = "num_cells",
    status_column: str = "status",
    state_column: str = "state",
    group_column: Optional[str] = None,
) -> Tuple[bool, str, pd.DataFrame]:
    """Validate data for modulation analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data for modulation analysis
    measure_column : str, optional
        Column containing the measurement values, by default "num_cells"
    status_column : str, optional
        Column containing modulation status, by default "status"
    state_column : str, optional
        Column containing state information, by default "state"
    group_column : Optional[str], optional
        Column containing group information, by default None

    Returns
    -------
    Tuple[bool, str, pd.DataFrame]
        (is_valid, error_message, cleaned_data)

    """
    if df is None or df.empty:
        return False, "Empty data provided", df

    # Check for required columns
    required_cols = [measure_column, status_column, state_column]
    if group_column:
        required_cols.append(group_column)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}", df

    # Clean and validate measure column
    cleaned_df = df.copy()
    original_len = len(cleaned_df)

    # Ensure measure column is numeric
    cleaned_df[measure_column] = pd.to_numeric(
        cleaned_df[measure_column], errors="coerce"
    )

    # Remove non-finite values
    cleaned_df = cleaned_df[np.isfinite(cleaned_df[measure_column])]

    if cleaned_df.empty:
        return (
            False,
            f"All values are non-finite in {measure_column}",
            cleaned_df,
        )

    if len(cleaned_df) < original_len:
        removed_count = original_len - len(cleaned_df)
        logger.debug(
            f"Removed {removed_count} non-finite values from {measure_column}"
        )

    # Check for sufficient data
    if len(cleaned_df) < 2:
        return (
            False,
            f"Insufficient data after cleaning: {len(cleaned_df)} < 2",
            cleaned_df,
        )

    # Check for variation in measure
    if cleaned_df[measure_column].var() == 0:
        return False, f"Zero variance in {measure_column}", cleaned_df

    # Validate grouping factors
    if (
        status_column in cleaned_df.columns
        and cleaned_df[status_column].nunique() < 1
    ):
        return (
            False,
            f"No valid modulation statuses in {status_column}",
            cleaned_df,
        )

    if (
        state_column in cleaned_df.columns
        and cleaned_df[state_column].nunique() < 1
    ):
        return False, f"No valid states in {state_column}", cleaned_df

    if group_column and cleaned_df[group_column].nunique() < 2:
        return (
            False,
            f"Insufficient groups in {group_column}: need >= 2",
            cleaned_df,
        )

    return True, "", cleaned_df


def _validate_modulation_group_data(
    df: pd.DataFrame,
    measure_column: str = "num_cells",
    group_column: str = "group",
    state_column: str = "state",
    data_pairing: str = "unpaired",
) -> Tuple[bool, str, pd.DataFrame]:
    """Validate data for group comparisons in modulation analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input data for group comparison
    measure_column : str, optional
        Column containing the measurement values, by default "num_cells"
    group_column : str, optional
        Column containing group information, by default "group"
    state_column : str, optional
        Column containing state information, by default "state"
    data_pairing : str, optional
        Type of pairing ("paired" or "unpaired"), by default "unpaired"

    Returns
    -------
    Tuple[bool, str, pd.DataFrame]
        (is_valid, error_message, cleaned_data)

    """
    # First run basic modulation validation
    is_valid, error_msg, cleaned_df = _validate_modulation_data(
        df,
        measure_column=measure_column,
        group_column=group_column,
        state_column=state_column,
    )

    if not is_valid:
        return is_valid, error_msg, cleaned_df

    # Additional group-specific validation
    groups = cleaned_df[group_column].unique()
    if len(groups) < 2:
        return (
            False,
            f"Need at least 2 groups, found {len(groups)}",
            cleaned_df,
        )

    # Check sample sizes per group
    group_counts = cleaned_df[group_column].value_counts()
    min_group_size = 2 if data_pairing == "unpaired" else 1

    if group_counts.min() < min_group_size:
        return (
            False,
            f"Insufficient data in some groups (min: {group_counts.min()}, need: {min_group_size})",
            cleaned_df,
        )

    # For paired data, check subject consistency
    if (
        data_pairing == "paired"
        and "normalized_subject_id" in cleaned_df.columns
    ):
        subject_group_counts = cleaned_df.groupby("normalized_subject_id")[
            group_column
        ].nunique()
        if not all(count == len(groups) for count in subject_group_counts):
            return (
                False,
                "Not all subjects have data in all groups for paired analysis",
                cleaned_df,
            )

    return True, "", cleaned_df


@_suppress_pingouin_warnings
def _safe_anova(
    data: pd.DataFrame,
    dv: str,
    between: Optional[str] = None,
    within: Optional[Union[str, List[str]]] = None,
    subject: Optional[str] = None,
    detailed: bool = True,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Safely perform ANOVA with data validation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data for ANOVA
    dv : str
        Dependent variable column name
    between : Optional[str], optional
        Between-subject factor, by default None
    within : Optional[Union[str, List[str]]], optional
        Within-subject factor(s), by default None
    subject : Optional[str], optional
        Subject ID column, by default None
    detailed : bool, optional
        Whether to return detailed results, by default True
    **kwargs
        Additional arguments passed to pingouin ANOVA functions

    Returns
    -------
    Optional[pd.DataFrame]
        ANOVA results or None if data is invalid

    """
    try:
        # Validate data structure
        is_valid, error_msg, cleaned_data = _validate_pairwise_test_data(
            data, dv, within, between
        )
        if not is_valid:
            logger.debug(f"_safe_anova: {error_msg}")
            return None

        # Determine ANOVA type and call appropriate function
        if within and between and subject:
            # Mixed ANOVA
            result = pg.mixed_anova(
                data=cleaned_data,
                dv=dv,
                within=within,
                between=between,
                subject=subject,
                **kwargs,
            )
        elif within and subject:
            # Repeated measures ANOVA
            result = pg.rm_anova(
                data=cleaned_data,
                dv=dv,
                within=within,
                subject=subject,
                detailed=detailed,
                **kwargs,
            )
        elif between:
            # One-way ANOVA
            result = pg.anova(
                data=cleaned_data,
                dv=dv,
                between=between,
                detailed=detailed,
                **kwargs,
            )
        else:
            logger.warning("_safe_anova: No valid ANOVA design specified")
            return None

        # Validate results
        if result is not None and not result.empty:
            # Check for invalid values in key columns
            numeric_cols = (
                ["F", "p-unc", "np2"]
                if "np2" in result.columns
                else ["F", "p-unc"]
            )
            for col in numeric_cols:
                if (
                    col in result.columns
                    and not np.isfinite(result[col]).all()
                ):
                    logger.debug(
                        f"_safe_anova: Invalid values detected in {col} "
                        f"column (automatically corrected)"
                    )
                    # Replace invalid F values with 0, p-values with 1
                    if col == "F":
                        result[col] = result[col].fillna(0.0)
                    elif col == "p-unc":
                        result[col] = result[col].fillna(1.0)
                    elif col == "np2":
                        result[col] = result[col].fillna(0.0)

        return result

    except Exception as e:
        logger.debug(f"_safe_anova: Error in ANOVA calculation: {str(e)}")
        return None


@_suppress_pingouin_warnings
def _safe_ttest_modulation(
    data: pd.DataFrame,
    measure_column: str,
    group_column: str,
    groups: List[str],
    paired: bool = False,
) -> Optional[pd.DataFrame]:
    """Safely perform t-test for modulation data with validation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing modulation measurements
    measure_column : str
        Column containing the measurement values
    group_column : str
        Column containing group labels
    groups : List[str]
        List of exactly 2 group names to compare
    paired : bool, optional
        Whether to perform paired t-test, by default False

    Returns
    -------
    Optional[pd.DataFrame]
        T-test results or None if data is invalid

    """
    if len(groups) != 2:
        logger.debug(
            f"_safe_ttest_modulation: Need exactly 2 groups, got {len(groups)}"
        )
        return None

    try:
        # Extract data for each group
        group1_data = data[data[group_column] == groups[0]][
            measure_column
        ].dropna()
        group2_data = data[data[group_column] == groups[1]][
            measure_column
        ].dropna()

        # Use existing safe t-test function
        return _safe_ttest(group1_data, group2_data, paired=paired)

    except Exception as e:
        logger.debug(
            f"_safe_ttest_modulation: Error in modulation t-test: {str(e)}"
        )
        return None


def _validate_measure_column_flexible(
    df: pd.DataFrame,
    measure_column: str,
    analysis_type: str = "statistical analysis",
) -> Tuple[bool, str, pd.DataFrame]:
    """Flexible validation for different measure columns (not just 'activity').

    Parameters
    ----------
    df : pd.DataFrame
        Input data for validation
    measure_column : str
        Name of the column containing measurements
    analysis_type : str, optional
        Type of analysis for logging, by default "statistical analysis"

    Returns
    -------
    Tuple[bool, str, pd.DataFrame]
        (is_valid, error_message, cleaned_data)

    """
    if df is None or df.empty:
        return False, "Empty data provided", df

    if measure_column not in df.columns:
        return (
            False,
            f"Measure column '{measure_column}' not found in data",
            df,
        )

    # Create cleaned copy
    cleaned_df = df.copy()
    original_len = len(cleaned_df)

    # Ensure measure column is numeric
    cleaned_df[measure_column] = pd.to_numeric(
        cleaned_df[measure_column], errors="coerce"
    )

    # Remove non-finite values
    cleaned_df = cleaned_df[np.isfinite(cleaned_df[measure_column])]

    if cleaned_df.empty:
        return (
            False,
            f"All values are non-finite in {measure_column}",
            cleaned_df,
        )

    if len(cleaned_df) < original_len:
        removed_count = original_len - len(cleaned_df)
        logger.debug(
            f"Removed {removed_count} non-finite values from {measure_column}"
        )

    # Check for variation
    if cleaned_df[measure_column].var() == 0:
        return (
            False,
            f"Zero variance in {measure_column} for {analysis_type}",
            cleaned_df,
        )

    # Check minimum sample size
    if len(cleaned_df) < 2:
        return (
            False,
            f"Insufficient data for {analysis_type}: {len(cleaned_df)} < 2",
            cleaned_df,
        )

    return True, "", cleaned_df
