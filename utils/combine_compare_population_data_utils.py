import logging
import warnings
import traceback
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

from ideas.exceptions import IdeasError
from utils.statistical_validation import (
    _suppress_pingouin_warnings,
    _validate_pairwise_test_data,
    _safe_pairwise_ttests,
    _validate_modulation_data,
    _validate_modulation_group_data,
    _safe_anova,
    _safe_ttest_modulation,
    _validate_measure_column_flexible,
)
from utils.statistical_formatting import (
    _add_pairing_columns,
    _finalize_statistical_output,
    _standardize_anova_output,
)

warnings.filterwarnings(
    "ignore", message="Group .* has less than .* valid samples"
)

DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05
logger = logging.getLogger(__name__)


def _compute_effect_sizes(
    lmm_result, data, dv_column, subject_column="normalized_subject_id"
):
    """Compute standardized effect sizes for LMM results."""
    if lmm_result.empty:
        return lmm_result

    try:
        if "state" in data.columns and len(data["state"].unique()) >= 2:
            states = data["state"].unique()
            if len(states) == 2:
                state1_data = data[data["state"] == states[0]][dv_column]
                state2_data = data[data["state"] == states[1]][dv_column]

                if len(state1_data) > 0 and len(state2_data) > 0:
                    pooled_std = np.sqrt(
                        (
                            (len(state1_data) - 1) * state1_data.var()
                            + (len(state2_data) - 1) * state2_data.var()
                        )
                        / (len(state1_data) + len(state2_data) - 2)
                    )
                    cohens_d = (
                        state2_data.mean() - state1_data.mean()
                    ) / pooled_std
                    state_mask = lmm_result["Source"].str.contains(
                        "state", case=False, na=False
                    )
                    lmm_result.loc[state_mask, "cohens_d"] = cohens_d

        # Add partial eta-squared approximation: eta² ≈ t² / (t² + df)
        for idx, row in lmm_result.iterrows():
            if not pd.isna(row.get("t_value")) and not pd.isna(row.get("df2")):
                try:
                    t_val = float(row["t_value"])
                    df = float(row["df2"])
                    if df > 0:
                        partial_eta_sq = (t_val**2) / (t_val**2 + df)
                        lmm_result.loc[
                            idx, "partial_eta_squared"
                        ] = partial_eta_sq
                except (ValueError, TypeError):
                    continue

        logger.debug("Added effect sizes to LMM results")
    except Exception as e:
        logger.warning(f"Could not compute effect sizes: {e}")

    return lmm_result


def _check_model_assumptions(model_result, data, dv_column):
    """Check basic LMM assumptions with simple diagnostics."""
    diagnostics = {
        "normality_pvalue": np.nan,
        "homoscedasticity_pvalue": np.nan,
        "assumption_warnings": [],
    }

    try:
        if model_result is None or not hasattr(model_result, "resid"):
            return diagnostics

        residuals = model_result.resid
        if len(residuals) < 3:
            return diagnostics

        if len(residuals) <= 5000:
            _, normality_p = stats.shapiro(residuals)
            diagnostics["normality_pvalue"] = normality_p
            if normality_p < 0.05:
                diagnostics["assumption_warnings"].append(
                    f"Residuals may not be normally distributed (p={normality_p:.3f})"
                )

        if hasattr(model_result, "fittedvalues"):
            fitted = model_result.fittedvalues
            abs_resid = np.abs(residuals)
            if len(fitted) > 3:
                corr, homo_p = stats.pearsonr(fitted, abs_resid)
                diagnostics["homoscedasticity_pvalue"] = homo_p
                if homo_p < 0.05:
                    diagnostics["assumption_warnings"].append(
                        f"Potential heteroscedasticity detected (p={homo_p:.3f})"
                    )

        if diagnostics["assumption_warnings"]:
            logger.info(
                f"Model assumption warnings: {diagnostics['assumption_warnings']}"
            )

    except Exception as e:
        logger.debug(f"Could not perform assumption checks: {e}")

    return diagnostics


def _choose_model_type(data, dv_column, measure_name):
    """Choose appropriate model type based on data characteristics."""
    try:
        values = data[dv_column].dropna()
        if len(values) == 0:
            return "gaussian_lmm"

        if "event" in measure_name.lower() or "count" in measure_name.lower():
            is_integers = np.allclose(values, np.round(values))
            is_non_negative = (values >= 0).all()
            if is_integers and is_non_negative:
                if values.var() > values.mean() * 1.5:
                    logger.info(
                        f"Detected count data with overdispersion for {measure_name}"
                    )
                else:
                    logger.info(f"Detected count data for {measure_name}")
                return "poisson_glmm"

        is_proportional = (values >= 0).all() and (values <= 1).all()
        has_boundaries = (values == 0).any() or (values == 1).any()
        if is_proportional and has_boundaries:
            logger.info(
                f"Detected proportional data for {measure_name}, will use logit transform"
            )
            return "logit_transform"

    except Exception as e:
        logger.debug(f"Error in model type selection: {e}")

    return "gaussian_lmm"


def _validate_statistical_power(df, subject_column, measure_name):
    """Check statistical power and generate warnings for potential issues."""
    warnings = []
    n_subjects = df[subject_column].nunique()
    n_states = df["state"].nunique() if "state" in df.columns else 1

    if n_subjects < 2:
        warnings.append(f"Insufficient subjects ({n_subjects}) for analysis")
        return True, "; ".join(warnings)
    elif n_subjects < 3:
        warnings.append(
            f"Low subject count ({n_subjects}) - interpret with caution"
        )
    elif n_subjects < 5:
        warnings.append(
            f"Moderate subject count ({n_subjects}) - limited generalizability"
        )

    obs_per_subject = df.groupby(subject_column).size()
    min_obs = obs_per_subject.min()
    max_obs = obs_per_subject.max()

    if min_obs < n_states:
        warnings.append(
            f"Incomplete data: some subjects have {min_obs}"
            f" observations but {n_states} states exist"
        )

    if max_obs / min_obs > 10:
        warnings.append(
            f"Highly imbalanced design (ratio: {max_obs/min_obs:.1f}) - "
            f"results may be unreliable"
        )
    elif max_obs / min_obs > 3:
        warnings.append(
            f"Moderately imbalanced design (ratio: {max_obs/min_obs:.1f}) - "
            f"interpret with caution"
        )

    if measure_name in df.columns:
        measure_var = df[measure_name].var()
        if measure_var < 1e-10:
            warnings.append(
                f"Very low variance in {measure_name} (var={measure_var:.2e}) - "
                f"may indicate measurement issues"
            )
        elif measure_var < 1e-6:
            warnings.append(
                f"Low variance in {measure_name} (var={measure_var:.2e}) - "
                f"effect sizes may be inflated"
            )

    has_concerns = len(warnings) > 0
    warning_message = "; ".join(warnings) if warnings else ""
    return has_concerns, warning_message


def _improve_degrees_of_freedom(model_result, data, subject_column):
    """Improve degrees of freedom calculation using Satterthwaite approximation."""
    try:
        if model_result is None or not hasattr(model_result, "params"):
            return None

        n_subjects = data[subject_column].nunique()
        n_params = len(model_result.params)
        effective_df = max(2, n_subjects - n_params)

        if effective_df < 5:
            logger.debug(
                f"Low degrees of freedom ({effective_df}) - reduced precision"
            )

        logger.debug(
            f"Improved DF calculation: {effective_df} (from {n_subjects} subjects)"
        )
        return effective_df

    except Exception as e:
        logger.debug(f"Could not improve DF calculation: {e}")
        return None


def _fit_simple_poisson_glmm(df, formula, subject_column, dv):
    """Fit a simple Poisson GLMM implementation for count data."""
    try:
        logger.info("Using simplified Poisson GLM approach for count data")

        avg_data = (
            df.groupby([subject_column, "state"])
            .agg(
                {
                    dv: "mean",
                    **{
                        col: "first"
                        for col in df.columns
                        if col not in [dv, subject_column, "state"]
                    },
                }
            )
            .reset_index()
        )

        avg_data[dv] = np.round(avg_data[dv]).astype(int)
        avg_data[dv] = np.maximum(avg_data[dv], 0)
        simplified_formula = formula.split("|")[0].strip()

        model = GLM.from_formula(
            simplified_formula, data=avg_data, family=Poisson()
        )
        result = model.fit()

        logger.info("Poisson GLM fitted successfully for count data")
        return result, avg_data

    except Exception as e:
        logger.warning(
            f"Poisson GLMM fitting failed: {e}, falling back to standard LMM"
        )
        return None, df


@_suppress_pingouin_warnings
def calculate_state_lmm_stats(
    *,
    df: pd.DataFrame,
    states: List[str],
    has_single_group: bool,
    data_pairing: str = "unpaired",
    effect_size: str = "hedges",
    correction: str = "bonf",
    measure_name: str = "activity",
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate state LMM statistics for population activity data.

    This function performs Linear Mixed Model analysis on population activity data,
    using either cell-level data (with proper random effects) or subject-averaged data
    depending on the data structure detected.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing population activity data
    states : List[str]
        List of state names to analyze
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

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        LMM fixed effects results and pairwise comparison results

    """
    try:
        # Input validation with enhanced logging
        if df is None or df.empty:
            logger.warning(f"No {measure_name} data provided for LMM analysis")
            return pd.DataFrame(), pd.DataFrame()

        if not states:
            logger.error(f"No states provided for {measure_name} LMM analysis")
            return pd.DataFrame(), pd.DataFrame()

        logger.info(
            f"=== Starting LMM Analysis for {measure_name.upper()} Data ==="
        )

        # Enhanced input validation
        required_columns = ["state", "normalized_subject_id"]
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            logger.error(
                f"Missing required columns for {measure_name} LMM: {missing_columns}"
            )
            return pd.DataFrame(), pd.DataFrame()

        # Check minimum data requirements
        n_subjects = df["normalized_subject_id"].nunique()
        n_states = df["state"].nunique()
        n_observations = len(df)

        logger.info(f"LMM input data summary for {measure_name}:")
        logger.info(f"  • {n_observations} total observations")
        logger.info(f"  • {n_subjects} unique subjects")
        logger.info(
            f"  • {n_states} unique states: {sorted(df['state'].unique())}"
        )

        # Initialize warnings list
        statistical_warnings = []

        if n_subjects < 2:
            warning_msg = f"Insufficient subjects ({n_subjects}) for {measure_name} analysis"
            logger.warning(warning_msg)
            statistical_warnings.append(warning_msg)

        if n_states < 2:
            warning_msg = (
                f"Insufficient states ({n_states}) for {measure_name} LMM analysis. "
                f"Need at least 2 states for comparison."
            )
            logger.warning(warning_msg)
            statistical_warnings.append(warning_msg)
            # Only return empty if truly no comparison possible
            if n_states < 1:
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
        has_concerns, power_warning = _validate_statistical_power(
            df, "normalized_subject_id", measure_name
        )
        if has_concerns:
            # Only add to warnings list - don't duplicate logging
            statistical_warnings.append(power_warning)
            # Continue with analysis but include warnings in output

        logger.info(
            f"Running LMM analysis on {len(df)} {measure_name} observations "
            f"with DV: {dv_column}"
        )

        between_factor = None if has_single_group else "group"
        analysis_design = (
            "Single-group within-subjects"
            if has_single_group
            else "Multi-group mixed-design"
        )
        logger.info(f"LMM analysis design: {analysis_design}")

        (
            lmm_result,
            has_reliability_concerns,
            lmm_warning_messages,
        ) = _perform_lmm_analysis(
            df=df,
            dv=dv_column,
            within_factors=["state"],
            subject="normalized_subject_id",
            between=between_factor,
            effect_size=effect_size,
            correction=correction,
            has_single_group=has_single_group,
            measure_name=measure_name,
        )

        if lmm_warning_messages:
            statistical_warnings.extend(lmm_warning_messages)

        pairwise_result = pd.DataFrame()
        try:
            logger.info(
                f"=== Starting Pairwise Comparisons for {measure_name.upper()} ==="
            )

            logger.info(
                f"Aggregating {measure_name} data to subject-level for pairwise "
                f"tests to prevent pseudoreplication"
            )
            pairwise_data = _calculate_subject_averages(
                df=df.copy(),
                subject_id_col="normalized_subject_id",
                measure_column=dv_column,
                context="for pairwise tests",
            )

            if "activity_mean" in pairwise_data.columns:
                pairwise_data = pairwise_data.rename(
                    columns={"activity_mean": dv_column}
                )

            if has_single_group:
                logger.info(
                    f"Performing within-subject state comparisons for single "
                    f"group {measure_name} data"
                )
                logger.info(f"Will compare states: {states}")

                (
                    is_valid,
                    error_msg,
                    cleaned_pairwise_data,
                ) = _validate_pairwise_test_data(
                    pairwise_data, dv_column, "state", None
                )

                if not is_valid:
                    logger.warning(
                        f"Single group {measure_name} pairwise validation "
                        f"failed: {error_msg}"
                    )
                    return lmm_result, pd.DataFrame()

                pairwise_args = {
                    "data": cleaned_pairwise_data,
                    "dv": dv_column,
                    "within": "state",
                    "subject": "normalized_subject_id",
                    "padjust": correction,
                    "effsize": effect_size,
                }
                pairwise_result = _safe_pairwise_ttests(**pairwise_args)

            else:
                logger.info(
                    f"Performing comprehensive pairwise comparisons for "
                    f"multi-group {measure_name} data"
                )
                logger.info(
                    "Will test: state effects, group effects, and "
                    "state×group interactions"
                )

                (
                    is_valid,
                    error_msg,
                    cleaned_pairwise_data,
                ) = _validate_pairwise_test_data(
                    pairwise_data, dv_column, "state", "group"
                )

                if not is_valid:
                    logger.warning(
                        f"Multi-group {measure_name} pairwise validation "
                        f"failed: {error_msg}"
                    )
                    return lmm_result, pd.DataFrame()

                pairwise_args = {
                    "data": cleaned_pairwise_data,
                    "dv": dv_column,
                    "within": "state",
                    "between": "group",
                    "subject": "normalized_subject_id",
                    "padjust": correction,
                    "effsize": effect_size,
                }
                pairwise_result = _safe_pairwise_ttests(**pairwise_args)

            if pairwise_result is None:
                logger.debug(
                    f"{measure_name} pairwise tests returned None, using empty DataFrame"
                )
                pairwise_result = pd.DataFrame()
            elif not isinstance(pairwise_result, pd.DataFrame):
                logger.warning(
                    f"{measure_name} pairwise result was not a DataFrame, attempting conversion"
                )
                try:
                    pairwise_result = pd.DataFrame(pairwise_result)
                except Exception as df_conv_err:
                    logger.error(
                        f"Could not convert {measure_name} pairwise "
                        f"result to DataFrame: {df_conv_err}"
                    )
                    pairwise_result = pd.DataFrame()

            if not pairwise_result.empty:
                pairwise_result["Comparison"] = f"{measure_name}_comparison"
                pairwise_result["Measure"] = measure_name
                pairwise_result["analysis_level"] = "subject"
                pairwise_result["state_comparison_type"] = (
                    "state" if has_single_group else "mixed"
                )

                pairwise_result = _add_pairing_columns(
                    pairwise_result,
                    state_comparison_type="state"
                    if has_single_group
                    else "mixed",
                    data_pairing=data_pairing,
                )

                if has_single_group and "group" not in pairwise_result.columns:
                    pairwise_result["group"] = "Single Group"

                pairwise_result = _finalize_statistical_output(
                    pairwise_result, "pairwise"
                )
                logger.info(
                    f"Generated {len(pairwise_result)} pairwise comparisons for {measure_name}"
                )
            else:
                logger.warning(
                    f"{measure_name} pairwise analysis returned empty results"
                )

        except Exception as pairwise_error:
            logger.warning(
                f"{measure_name} pairwise comparison failed: {str(pairwise_error)}"
            )
            pairwise_result = pd.DataFrame()

        if statistical_warnings:
            warning_text = "; ".join(statistical_warnings)
            if not lmm_result.empty:
                lmm_result["statistical_warnings"] = warning_text
                # Preserve existing analysis_notes and append warning info
                if (
                    "analysis_notes" in lmm_result.columns
                    and not lmm_result["analysis_notes"].empty
                ):
                    existing_notes = (
                        lmm_result["analysis_notes"].iloc[0]
                        if len(lmm_result) > 0
                        else ""
                    )
                else:
                    existing_notes = ""

                if existing_notes:
                    combined_notes = (
                        f"{existing_notes} | Warning: Results generated with "
                        f"statistical concerns - see warnings"
                    )
                else:
                    combined_notes = "Results generated with statistical concerns - see warnings"
                lmm_result["analysis_notes"] = combined_notes

        if not lmm_result.empty:
            logger.info(
                f"LMM analysis for {measure_name}: {len(lmm_result)} results"
            )
        if not pairwise_result.empty:
            logger.info(
                f"LMM pairwise for {measure_name}: {len(pairwise_result)} comparisons"
            )

        return lmm_result, pairwise_result

    except Exception as e:
        logger.error("Error in %s LMM analysis: %s", measure_name, str(e))
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()


@_suppress_pingouin_warnings
def calculate_group_anova_stats(
    *,
    df: pd.DataFrame,
    states: List[str],
    correction: str,
    effect_size: str = "hedges",
    data_pairing: str = "unpaired",
    measure_name: str = "activity",
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate group comparison statistics using direct pingouin calls.

    This function handles group comparisons appropriately based on the number of states:
    - Single state + Single group: Raises IdeasError (no statistical comparison possible)
    - Single state + Multiple groups: Performs simple t-test or one-way ANOVA between groups
    - Multiple states + Multiple groups: Performs mixed/repeated measures ANOVA

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing activity data
    states : List[str]
        List of state names
    correction : str
        Method for multiple comparison correction
    effect_size : str, optional
        Effect size measure to use, by default "hedges"
    data_pairing : str, optional
        Data pairing method, defaults to "unpaired"
    measure_name : str, optional
        Name of the measure being analyzed (e.g., "activity"). Default is "activity".
    significance_threshold : float, optional
        Significance threshold, by default DEFAULT_SIGNIFICANCE_THRESHOLD

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Group comparison ANOVA and pairwise results
        Subject-averaged DataFrame used for ANOVA and plotting (group_df)

    Raises
    ------
    IdeasError
        When single state and fewer than 2 groups are provided (no comparison possible)

    """
    # Basic input validation - do these checks outside try block
    if df is None or df.empty:
        logger.debug("Input DataFrame is None or empty")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "normalized_subject_id" not in df.columns:
        logger.debug("No subject identifier found for group comparison")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "activity" not in df.columns:
        logger.debug("Required column 'activity' not found in data")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        # Calculate subject averages directly
        group_df = _calculate_subject_averages(
            df=df.copy(),
            subject_id_col="normalized_subject_id",
            measure_column="activity",
            context="for group comparisons",
        )

        if group_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Check statistical power and add warnings to results if needed
        has_concerns, power_warning = _validate_statistical_power(
            group_df, "normalized_subject_id", measure_name
        )
        statistical_warnings = []
        if has_concerns:
            # Only add to warnings list - don't duplicate logging
            statistical_warnings.append(power_warning)
            # Continue with analysis but include warnings in output

        # Validate the averaged data for statistical analysis
        is_valid, error_msg, group_df = _validate_measure_column_flexible(
            group_df, measure_column="activity", analysis_type="group ANOVA"
        )

        if not is_valid:
            logger.debug(f"Group ANOVA validation failed: {error_msg}")
            return pd.DataFrame(), pd.DataFrame(), group_df

    except Exception as e:
        logger.error(
            f"Error in data preparation: {type(e).__name__}: {str(e)}"
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Check if we have groups to compare - do this outside try block so IdeasError propagates
    if "group" not in group_df.columns or group_df["group"].nunique() < 2:
        # Check if this is a single state scenario
        if len(states) == 1 or group_df["state"].nunique() == 1:
            state_name = (
                group_df["state"].iloc[0] if not group_df.empty else "unknown"
            )
            logger.error(
                f"Cannot perform statistical comparison with single state "
                f"('{state_name}') and fewer than 2 groups. "
                f"Found {group_df['group'].nunique() if 'group' in group_df.columns else 0} "
                f"groups. Statistical comparisons require either multiple states "
                f"OR multiple groups."
            )
            # Return empty results instead of raising exception for consistency with modulation path
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            logger.debug("Insufficient groups for group comparison")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:

        # Handle single state scenarios differently
        n_states = group_df["state"].nunique()
        if n_states == 1:
            # Single state, multiple groups - use simple group comparison
            logger.info(
                "Single state detected - performing direct group comparison"
            )
            return _handle_single_state_group_comparison(
                group_df=group_df,
                correction=correction,
                effect_size=effect_size,
                data_pairing=data_pairing,
                measure_name=measure_name,
            )

        # Multiple states - standard ANOVA analysis
        aov_df = pd.DataFrame()
        pairwise_df = pd.DataFrame()

        try:
            if data_pairing == "paired":
                # Repeated measures ANOVA using safe function
                aov_df = _safe_anova(
                    data=group_df,
                    dv="activity",
                    within=["state", "group"],
                    subject="normalized_subject_id",
                )

                pairwise_df = _safe_pairwise_ttests(
                    data=group_df,
                    dv="activity",
                    within=["state", "group"],
                    subject="normalized_subject_id",
                    padjust=correction,
                    effsize=effect_size,
                )

            else:
                # Mixed ANOVA using safe function
                aov_df = _safe_anova(
                    data=group_df,
                    dv="activity",
                    within="state",
                    between="group",
                    subject="normalized_subject_id",
                )

                pairwise_df = _safe_pairwise_ttests(
                    data=group_df,
                    dv="activity",
                    within="state",
                    between="group",
                    subject="normalized_subject_id",
                    padjust=correction,
                    effsize=effect_size,
                )

        except Exception as e:
            logger.warning(f"Statistical analysis failed: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), group_df

        # Handle None results from safe functions
        if aov_df is None:
            aov_df = pd.DataFrame()
        if pairwise_df is None:
            pairwise_df = pd.DataFrame()

        # Add basic metadata
        if not aov_df.empty:
            aov_df["Comparison"] = (
                measure_name.capitalize()
                if measure_name
                else "Group Comparison"
            )
            aov_df["Measure"] = measure_name
            aov_df["state_comparison_type"] = "group"
            aov_df["analysis_level"] = "subject"
            # Set more specific stat_method based on the ANOVA type used
            if data_pairing == "paired":
                aov_df["stat_method"] = "rm_anova"
            else:
                aov_df["stat_method"] = "mixed_anova"

        if not pairwise_df.empty:
            pairwise_df["Comparison"] = (
                measure_name.capitalize()
                if measure_name
                else "Group Comparison"
            )
            pairwise_df["Measure"] = measure_name
            pairwise_df["state_comparison_type"] = "group"
            pairwise_df["analysis_level"] = "subject"

            # Add pairing columns and update stat_method
            try:
                pairwise_df = _add_pairing_columns(
                    pairwise_df,
                    state_comparison_type="group",
                    data_pairing=data_pairing,
                )
            except Exception as e:
                logger.warning(f"Could not add pairing columns: {str(e)}")

        # Apply final output standardization

        if not aov_df.empty:
            # Apply standardization first to extract df1/df2 from pingouin format
            try:
                metadata = {
                    "Comparison": (
                        measure_name.capitalize()
                        if measure_name
                        else "Group Comparison"
                    ),
                    "Measure": measure_name,
                }
                aov_df = _standardize_anova_output(
                    aov_df,
                    metadata=metadata,
                    analysis_level="subject",
                    state_comparison_type="group",
                    measure_name=measure_name,
                )
            except Exception as e:
                logger.warning(f"Could not standardize ANOVA output: {str(e)}")

            aov_df = _finalize_statistical_output(aov_df, "anova")
        if not pairwise_df.empty:
            pairwise_df = _finalize_statistical_output(pairwise_df, "pairwise")

        # Add statistical warnings to results if any exist (similar to LMM analysis)
        if statistical_warnings:
            warning_text = "; ".join(statistical_warnings)

            # Only add warnings to ANOVA results (not pairwise) to match LMM approach
            if not aov_df.empty:
                aov_df["statistical_warnings"] = warning_text
                aov_df[
                    "analysis_notes"
                ] = "Results generated with statistical concerns - see warnings"

            # Don't add warnings to pairwise results anymore - they go to ANOVA CSV via aov_df

            logger.warning(
                f"Statistical warnings added to group ANOVA results: {warning_text}"
            )

        return aov_df, pairwise_df, group_df

    except Exception as e:
        logger.error(
            f"Unexpected error in calculate_group_anova_stats: "
            f"{type(e).__name__}: {str(e)}"
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def _calculate_subject_averages(
    df: pd.DataFrame,
    subject_id_col: str,
    measure_column: str = "activity",
    context: str = "",
) -> pd.DataFrame:
    """Calculate subject-level averages from cell-level data to prevent pseudoreplication.

    This function aggregates multiple cell-level observations per subject into single
    subject-level values. This is critical for preventing pseudoreplication in
    statistical analyses where individual cells from the same subject are treated
    as independent observations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with cell-level data
    subject_id_col : str
        Column name containing subject IDs
    measure_column : str, optional
        Column name containing the measurement values, by default "activity"
    context : str, optional
        Context string for logging purposes, by default ""

    Returns
    -------
    pd.DataFrame
        DataFrame with subject-level averages

    """
    try:
        # Input validation (matching old_utils.py)
        if df is None or df.empty:
            logger.warning("Input DataFrame is empty")
            return pd.DataFrame()

        required_cols = [subject_id_col, "state", measure_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Create a copy and log input data summary (enhanced logging)
        df = df.copy()
        n_total_obs = len(df)
        n_subjects = df[subject_id_col].nunique()

        logger.debug(
            f"Subject averaging input for {measure_column}: "
            f"{n_total_obs} observations, {n_subjects} subjects"
        )

        df = df.dropna(subset=[subject_id_col, "state", measure_column])
        if df.empty:
            logger.warning("No valid data remaining after cleaning")
            return pd.DataFrame()

        df[subject_id_col] = df[subject_id_col].astype(str)
        df["state"] = df["state"].astype(str)
        df[measure_column] = pd.to_numeric(df[measure_column], errors="coerce")

        df = df.dropna(subset=[measure_column])
        if df.empty:
            logger.warning("No valid numeric data remaining after cleaning")
            return pd.DataFrame()

        group_cols = [subject_id_col]

        if "state" in df.columns:
            group_cols.append("state")

        if "group" in df.columns:
            df["group"] = df["group"].fillna("ungrouped").astype(str)
            group_cols.append("group")
        elif "group_name" in df.columns:
            df["group"] = df["group_name"].fillna("ungrouped").astype(str)
            group_cols.append("group")

        agg_dict = {measure_column: "mean"}

        if "num_cells" in df.columns:
            agg_dict["num_cells"] = "sum"
        if "file" in df.columns:
            agg_dict["file"] = "first"
        if "total_cell_count" in df.columns:
            agg_dict["total_cell_count"] = "first"

        logger.debug(f"Grouping by: {group_cols}")

        averaged_df = (
            df.groupby(group_cols, observed=True).agg(agg_dict).reset_index()
        )
        averaged_df["is_averaged"] = True

        if (
            "normalized_subject_id" in df.columns
            and subject_id_col != "normalized_subject_id"
        ):
            subject_mapping = df[
                [subject_id_col, "normalized_subject_id"]
            ].drop_duplicates()
            averaged_df = averaged_df.merge(
                subject_mapping, on=subject_id_col, how="left"
            )

        n_averaged_obs = len(averaged_df)
        context_msg = f" ({context})" if context else ""
        logger.info(
            f"Subject averaging completed{context_msg}: {len(df)} cells -> "
            f"{n_averaged_obs} subject averages"
        )
        logger.debug(
            f"Data reduction: {n_averaged_obs/n_total_obs:.1%} - "
            f"prevents pseudoreplication"
        )

        return averaged_df

    except Exception as e:
        logger.error(
            f"Error calculating subject averages: {type(e).__name__}: {str(e)}"
        )
        # Return empty DataFrame instead of re-raising to maintain consistent API
        return pd.DataFrame()


def detect_state_comparison_type(df: pd.DataFrame) -> str:
    """Detect the type of comparison present in the modulation data."""
    if df is None or df.empty:
        raise IdeasError(
            "Cannot detect comparison type: Empty dataframe provided",
        )

    modulation_columns = [
        col for col in df.columns if "modulation scores in " in col
    ]

    if not modulation_columns:
        raise IdeasError(
            "No modulation columns found in data structure",
        )

    baseline_state, mod_states, _ = detect_baseline_state(df)
    if baseline_state is not None:
        return "state_vs_baseline"

    state_names = [
        col.replace("modulation scores in ", "") for col in modulation_columns
    ]

    if any(" vs " in state for state in state_names):
        return "pairwise"

    return "state_vs_not_state"


def detect_baseline_state(
    df: pd.DataFrame,
) -> Tuple[str, List[str], List[str]]:
    """Detect baseline state from a dataframe with modulation data."""
    if df is None or df.empty:
        return None, [], []

    mod_states = []
    for col in df.columns:
        if "modulation scores in " in col:
            state = col.replace("modulation scores in ", "")
            mod_states.append(state)

    mean_states = []
    for col in df.columns:
        if (
            col.startswith("mean ")
            and " in " in col
            and any(
                unit in col.lower()
                for unit in [
                    "activity",
                    "fractional change",
                    "z-score",
                    "normalized",
                    "event rate",
                ]
            )
        ):
            state = col.split(" in ")[-1]
            if state not in mean_states:
                mean_states.append(state)

    if any(" vs " in state for state in mod_states):
        logger.debug(
            "detect_baseline_state: Pairwise comparison detected - "
            "no traditional baseline state applies"
        )
        return (None, mod_states, mean_states)

    if mean_states and mod_states:
        possible_baseline_states = set(mean_states) - set(mod_states)
        if len(possible_baseline_states) == 1:
            baseline_state = list(possible_baseline_states)[0]
            return (baseline_state, mod_states, mean_states)
        elif len(possible_baseline_states) > 1:
            raise IdeasError(
                "More than 1 baseline state detected, check the input data "
                "for accuracy",
            )

    return (None, mod_states, mean_states)


def validate_states_by_state_comparison_type(
    df: pd.DataFrame, requested_states: List[str]
) -> Tuple[List[str], str, Optional[str]]:
    """Validate requested states against available states for detected comparison type."""
    if df is None or df.empty:
        raise IdeasError(
            "Cannot validate states: Empty dataframe provided",
        )

    state_comparison_type = detect_state_comparison_type(df)
    baseline_state, mod_states, _ = detect_baseline_state(df)

    if state_comparison_type == "pairwise":
        available_states = [s for s in mod_states if " vs " in s]
        if not any(" vs " in s for s in requested_states):
            validated_states = []
            for pair in available_states:
                state1, state2 = pair.split(" vs ")
                if (
                    state1.strip() in requested_states
                    or state2.strip() in requested_states
                ):
                    validated_states.append(pair)
        else:
            validated_states = [
                s for s in requested_states if s in available_states
            ]
    elif state_comparison_type == "state_vs_baseline" and baseline_state:
        validated_states = [
            s
            for s in requested_states
            if s != baseline_state and s in mod_states
        ]
    else:
        available_states = [s for s in mod_states if " vs " not in s]
        validated_states = [
            s for s in requested_states if s in available_states
        ]

    if not validated_states:
        raise IdeasError(
            f"No valid states found for {state_comparison_type}. "
            f"Requested: {requested_states}, Available: {mod_states}",
        )

    return validated_states, state_comparison_type, baseline_state


def _perform_lmm_analysis(
    df: pd.DataFrame,
    dv: str,
    within_factors: List[str],
    subject: str,
    between: Optional[str] = None,
    effect_size: str = "hedges",
    correction: str = "bonf",
    has_single_group: bool = True,
    measure_name: str = "activity",
) -> Tuple[pd.DataFrame, bool, List[str]]:
    """Perform Linear Mixed Model analysis on raw cell-level data.

    This function performs only the main LMM analysis using raw cell-level data
    with proper random effects structure. Pairwise comparisons should be handled
    separately by group ANOVA functions to avoid duplication.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with cell-level data
    dv : str
        Dependent variable column name
    within_factors : List[str]
        List of within-subject factors
    subject : str
        Subject ID column name (used for random intercepts)
    between : str, optional
        Between-subject factor
    effect_size : str, optional
        Effect size measure, by default "hedges"
    correction : str, optional
        Multiple comparison correction, by default "bonf"
    has_single_group : bool, optional
        Whether analysis is for single group, by default True
    measure_name : str, optional
        Name of the measure being analyzed, by default "activity"

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
            [subject] + within_factors + ([between] if between else [])
        ).size()

        has_nested_data = (obs_per_subject_condition > 1).any()
        max_obs_per_condition = obs_per_subject_condition.max()

        # Also check for explicit cell column as additional indicator
        has_cell_column = "cell" in df.columns
        if has_cell_column:
            cells_per_subject_condition = df.groupby(
                [subject] + within_factors + ([between] if between else [])
            )["cell"].nunique()
            has_multiple_cells = (cells_per_subject_condition > 1).any()
        else:
            has_multiple_cells = False

        # Determine if we have nested structure
        has_nested_structure = has_nested_data or has_multiple_cells

        # Calculate basic data info
        n_subjects = df[subject].nunique()
        n_observations = len(df)

        logger.info(
            f"LMM data structure: {n_subjects} subjects, {n_observations} observations"
        )
        if has_nested_structure:
            logger.info(
                f"  • Cell-level data (max {max_obs_per_condition} per subject-condition) "
                f"- using proper random effects"
            )
        else:
            logger.info("  • Subject-level data - using standard LMM")

        # Drop rows with missing values in critical columns
        critical_cols = within_factors + [subject]
        if between is not None:
            critical_cols.append(between)

        df = df.dropna(subset=critical_cols)

        if df.empty:
            logger.warning(
                f"No {measure_name} data remaining after removing missing values"
            )
            return pd.DataFrame(), False, lmm_warning_messages

        # Ensure all factor columns are strings
        for factor in within_factors:
            df[factor] = df[factor].astype(str)
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
        min_observations = 3
        if len(df) < min_observations:
            logger.warning(
                f"Very low observation count for {measure_name} LMM: {len(df)} < "
                f"{min_observations} observations. Analysis may be unreliable."
            )
            return pd.DataFrame(), False, lmm_warning_messages
        elif len(df) < 6:
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

        if len(df) < min_observations:
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

        # =============================================================================
        # ENHANCEMENT 3: DATA-TYPE-SPECIFIC MODEL SELECTION
        # =============================================================================

        # Choose appropriate model type based on data characteristics
        model_type = _choose_model_type(df, dv, measure_name)
        logger.info(f"Selected model type: {model_type} for {measure_name}")

        # Apply data transformations if needed
        if model_type == "logit_transform":
            # Apply logit transformation for proportional data
            # Add small epsilon to handle 0 and 1 values
            epsilon = 1e-6
            df[dv] = df[dv].clip(epsilon, 1 - epsilon)
            df[f"{dv}_logit"] = np.log(df[dv] / (1 - df[dv]))
            dv = f"{dv}_logit"
            logger.info(f"Applied logit transformation to {measure_name}")
        elif model_type == "poisson_glmm":
            # For Poisson GLMM, ensure integer values and log info
            df[dv] = np.round(df[dv]).astype(int)
            logger.info(
                f"Prepared count data for Poisson GLMM for {measure_name}"
            )

        # **SIMPLIFIED: Only LMM Analysis on raw cell-level data**
        # Build the formula for the LMM dynamically (fixed effects part)
        formula = f"{dv} ~ 1"
        all_factors = within_factors + ([between] if between else [])

        # Add main effects
        if all_factors:
            formula += " + " + " + ".join(all_factors)

        # Add interaction terms
        fixed_interactions = []
        if len(within_factors) > 1:
            fixed_interactions.append(":".join(within_factors))
        if within_factors and between:
            for w_factor in within_factors:
                fixed_interactions.append(f"{w_factor}:{between}")
        # Only add complex interactions if we have multiple within factors AND between factor
        if len(within_factors) > 1 and between:
            fixed_interactions.append(":".join(within_factors + [between]))

        if fixed_interactions:
            formula += " + " + " + ".join(fixed_interactions)

        logger.info(f"LMM formula: {formula} | Random: (1|{subject})")

        # =============================================================================
        # MODEL FITTING WITH DATA-TYPE-SPECIFIC APPROACH
        # =============================================================================

        # Fit the appropriate model based on data type
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

        # Initialize result if not already set by Poisson GLMM
        if model_type != "poisson_glmm":
            result = None

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

            for method in methods:
                try:
                    logger.debug(
                        f"Attempting LMM fit for {measure_name} using {method} optimizer"
                    )
                    # Use reml=True by default for better variance estimates
                    result = model.fit(method=method, maxiter=1000, reml=True)
                    if result.converged:
                        logger.info(
                            f"LMM converged successfully using {method}"
                        )
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
        hessian_warn = any(
            "hessian matrix" in str(w.message).lower() for w in fit_warnings
        )

        # Check for specific convergence/reliability issues
        model_converged = result is not None and result.converged
        has_fit_warnings = singular_cov_warn or boundary_warn or hessian_warn
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
                    except (np.linalg.LinAlgError, ValueError, TypeError):
                        pass

                    # Check for near-zero eigenvalues
                    try:
                        eigenvals = np.linalg.eigvals(result.cov_params)
                        if np.any(eigenvals < 1e-12):
                            has_singular_covariance = True
                    except (np.linalg.LinAlgError, ValueError, TypeError):
                        pass

            except (np.linalg.LinAlgError, ValueError):
                # If covariance matrix is malformed, treat as singular
                has_singular_covariance = True

        if result is not None and result.bse is not None:
            has_invalid_se = (
                np.isnan(result.bse).any() or np.isinf(result.bse).any()
            )

        # Collect all warning messages for proper timing - consolidate singular covariance warnings
        all_issues = []
        if has_singular_covariance or singular_cov_warn:
            all_issues.append("singular covariance")
        if boundary_warn:
            all_issues.append("boundary issues")
        if hessian_warn:
            all_issues.append("hessian issues")
        if has_invalid_se:
            all_issues.append("invalid standard errors")

        # Generate appropriate warning messages
        if len(all_issues) == 1 and all_issues[0] == "singular covariance":
            # Standalone singular covariance warning
            warning_msg = (
                f"LMM singular covariance detected for {measure_name} - "
                f"interpret with caution"
            )
            lmm_warning_messages.append(warning_msg)
        elif len(all_issues) > 0:
            # Multiple issues including or excluding singular covariance
            warning_msg = (
                f"LMM fitting issues detected ({', '.join(all_issues)}) - "
                f"interpret with caution"
            )
            lmm_warning_messages.append(warning_msg)

        # Relaxed convergence status reporting - allow non-converged results with warnings
        if not model_converged:
            if result is None:
                logger.warning(f"LMM fitting failed for {measure_name}")
                return pd.DataFrame(), False, lmm_warning_messages
            else:
                warning_msg = f"LMM non-convergence for {measure_name} - interpret with caution"
                lmm_warning_messages.append(warning_msg)
                logger.debug(
                    f"LMM convergence issues: {'; '.join(convergence_issues)}"
                )

        # Summary of model fitting status
        if result is not None:
            df_residual = getattr(result, "df_resid", None)
            if (
                df_residual is None
                or np.isnan(df_residual)
                or np.isinf(df_residual)
            ):
                df_residual = max(1, len(df) - len(result.params))
                logger.debug(f"Using fallback df calculation: {df_residual}")

            # Only log if there are concerns, otherwise keep quiet
            if has_fit_warnings or has_invalid_se or not model_converged:
                logger.debug(
                    f"LMM for {measure_name}: {len(result.params)} params, df={df_residual}"
                )
            else:
                logger.debug(f"LMM fitted successfully for {measure_name}")

            # Display any warnings at the appropriate time and track reliability flag
        has_reliability_concerns = False
        if lmm_warning_messages:
            has_reliability_concerns = True
            for warning_msg in lmm_warning_messages:
                logger.warning(warning_msg)
        else:
            has_reliability_concerns = False

        if result is not None:
            # Extract fixed effects results robustly
            try:
                conf_int = result.conf_int()
                lmm_result = pd.DataFrame(
                    {
                        "Source": result.params.index,
                        "coefficient": result.params.values,
                        "std_error": result.bse.values,
                        "t_value": result.tvalues.values,
                        "p_value": result.pvalues.values,
                        "conf_int_lower": conf_int.iloc[:, 0].values,
                        "conf_int_upper": conf_int.iloc[:, 1].values,
                        "df2": df_residual,
                        "stat_test_name": "lmm",
                        "stat_method": "lmm",
                    }
                )

                # Add placeholder columns expected by standardization
                for col in [
                    "sum_squares",
                    "df1",
                    "mean_square",
                    "F_statistic",
                    "effect_size",
                    "sphericity",
                ]:
                    if col not in lmm_result.columns:
                        lmm_result[col] = np.nan

                # For LMM fixed effects, each parameter typically has 1 df
                lmm_result["df1"] = 1

                # Add metadata
                has_cell_data = (
                    "cell" in df.columns
                    and df.groupby([subject, "state"])["cell"].nunique().max()
                    > 1
                )

                # Add standardized metadata
                lmm_result["stat_test_name"] = "lmm_reml"
                metadata = {
                    "Comparison": (
                        measure_name.capitalize()
                        if measure_name
                        else "State Comparison"
                    ),
                    "Measure": measure_name,
                    "state_comparison_type": "state",
                }

                # Add explanatory notes for LMM Source column encoding
                lmm_explanation = (
                    "LMM Source column uses treatment coding: [T.value] indicates "
                    "treatment contrast vs reference level. Interactions shown as "
                    "factor1[T.level1]:factor2[T.level2]. Example: "
                    "'state[T.stimulus]:group[T.treatment]' = stimulus state vs "
                    "reference state, in treatment group vs reference group."
                )
                lmm_result["analysis_notes"] = lmm_explanation

                try:
                    lmm_result = _standardize_anova_output(
                        lmm_result,
                        metadata=metadata,
                        analysis_level=(
                            "subject/cell" if has_cell_data else "subject"
                        ),
                        state_comparison_type="state",
                        measure_name=measure_name,
                    )
                except Exception as e:
                    logger.error(
                        f"Error standardizing LMM output for {measure_name}: {str(e)}"
                    )
                    for key, value in metadata.items():
                        if key not in lmm_result.columns:
                            lmm_result[key] = value
                    lmm_result["analysis_level"] = (
                        "subject/cell" if has_cell_data else "subject"
                    )

                # =============================================================================
                # STATISTICAL ENHANCEMENTS - APPLY IMPROVEMENTS
                # =============================================================================

                # 1. Improve degrees of freedom calculation
                improved_df = _improve_degrees_of_freedom(result, df, subject)
                if improved_df is not None:
                    lmm_result["df2"] = improved_df
                    logger.debug(
                        f"Updated DF from {df_residual} to {improved_df}"
                    )

                # 2. Add effect size calculations (Cohen's d, partial eta-squared)
                lmm_result = _compute_effect_sizes(lmm_result, df, dv, subject)

                # 3. Check model assumptions (residual diagnostics)
                diagnostics = _check_model_assumptions(result, df, dv)
                if diagnostics["assumption_warnings"]:
                    has_reliability_concerns = True
                    for warning in diagnostics["assumption_warnings"]:
                        logger.warning(f"Model assumption issue: {warning}")

                # Add diagnostic info to results (as metadata columns)
                lmm_result["normality_p"] = diagnostics.get(
                    "normality_pvalue", np.nan
                )
                lmm_result["homoscedasticity_p"] = diagnostics.get(
                    "homoscedasticity_pvalue", np.nan
                )

                # Preserve important columns before standardization
                important_cols = [
                    "coefficient",
                    "std_error",
                    "t_value",
                    "conf_int_lower",
                    "conf_int_upper",
                ]
                preserved_data = {}
                for col in important_cols:
                    if col in lmm_result.columns:
                        preserved_data[col] = lmm_result[col].copy()

                # Apply final output standardization
                lmm_result = _finalize_statistical_output(lmm_result, "anova")

                # Restore preserved columns
                for col, data in preserved_data.items():
                    lmm_result[col] = data

                logger.info(
                    f"Enhanced LMM completed: {lmm_result.shape[0]} fixed effects "
                    f"results for {measure_name} with effect sizes and diagnostics"
                )
                return (
                    lmm_result,
                    has_reliability_concerns,
                    lmm_warning_messages,
                )

            except Exception as extract_err:
                logger.error(
                    f"Failed to extract LMM results for {measure_name}: "
                    f"{extract_err}"
                )
                return pd.DataFrame(), False, lmm_warning_messages
        else:
            return pd.DataFrame(), False, lmm_warning_messages

    except Exception as e:
        logger.error(
            f"Error in LMM analysis setup/execution for {measure_name}: "
            f"{type(e).__name__}: {str(e)}"
        )
        logger.error(traceback.format_exc())
        return pd.DataFrame(), False, lmm_warning_messages


def get_statistical_method_mapping() -> Dict[Tuple[str, str], str]:
    """Get a simple mapping of experimental conditions to statistical methods."""
    return {
        ("pairwise", "single_group"): "Descriptive statistics only",
        ("state_vs_baseline", "single_group"): "One-way ANOVA across states",
        ("state_vs_not_state", "single_group"): "One-way ANOVA across states",
        (
            "pairwise",
            "multiple_groups_paired",
        ): "Paired t-tests between groups",
        (
            "state_vs_baseline",
            "multiple_groups_paired",
        ): "Repeated measures ANOVA",
        (
            "state_vs_not_state",
            "multiple_groups_paired",
        ): "Repeated measures ANOVA",
        (
            "pairwise",
            "multiple_groups_unpaired",
        ): "Unpaired t-tests between groups",
        ("state_vs_baseline", "multiple_groups_unpaired"): "Mixed ANOVA",
        ("state_vs_not_state", "multiple_groups_unpaired"): "Mixed ANOVA",
    }


def determine_experimental_condition(
    df: pd.DataFrame, data_pairing: str, state_comparison_type: str
) -> Tuple[str, str]:
    """Determine experimental condition from data characteristics."""
    has_multiple_groups = False

    if "group" in df.columns and not df["group"].isnull().all():
        has_multiple_groups = df["group"].nunique() > 1
    elif "group_name" in df.columns and not df["group_name"].isnull().all():
        has_multiple_groups = df["group_name"].nunique() > 1

    if not has_multiple_groups:
        data_structure = "single_group"
    elif data_pairing == "paired":
        data_structure = "multiple_groups_paired"
    else:
        data_structure = "multiple_groups_unpaired"

    return state_comparison_type, data_structure


@_suppress_pingouin_warnings
def calculate_mod_stats_direct(
    df: pd.DataFrame,
    states: List[str],
    correction: str,
    effect_size: str,
    data_pairing: str = "unpaired",
    measure_column: str = "num_cells",
    group_column: str = "group_name",
    comparison_description: str = "Modulation",
    status_values: Optional[List[str]] = None,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD,
    state_comparison_type: Optional[str] = None,
    baseline_state: Optional[str] = None,
    use_weighted_proportions: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate modulation statistics using direct statistical function calls.

    This function completely bypasses all legacy code and calls statistical
    functions directly based on experimental conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing modulation information
    states : List[str]
        List of state names to analyze
    correction : str
        Multiple comparison correction method
    effect_size : str
        Effect size measure to use
    data_pairing : str, optional
        Data pairing type, by default "unpaired"
    measure_column : str, optional
        Column name containing the measure values, by default "num_cells"
    group_column : str, optional
        Column name containing group information, by default "group_name"
    comparison_description : str, optional
        Description of the comparison, by default "Modulation"
    status_values : Optional[List[str]], optional
        List of status values to analyze, by default None
    significance_threshold : float, optional
        Significance threshold, by default DEFAULT_SIGNIFICANCE_THRESHOLD
    state_comparison_type : Optional[str], optional
        Type of comparison being performed, by default None
    baseline_state : Optional[str], optional
        Baseline state if applicable, by default None
    use_weighted_proportions : bool, optional
        Whether to apply weighted proportions to address pseudoreplication, by default False.
        Should only be enabled for population data that has total_cell_count column.
        Must be disabled for correlation data analysis.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ANOVA results and pairwise comparison results

    """
    try:
        # Input validation
        if df is None or df.empty:
            logger.warning(
                f"No {comparison_description.lower()} data provided"
            )
            return pd.DataFrame(), pd.DataFrame()

        if not states or state_comparison_type is None:
            logger.error(
                "Missing required parameters for %s: states or state_comparison_type",
                comparison_description.lower(),
            )
            return pd.DataFrame(), pd.DataFrame()

        logger.info(
            "=== Starting Direct Statistical Analysis for %s Data ===",
            comparison_description.upper(),
        )

        # Determine experimental condition
        comp_type, data_structure = determine_experimental_condition(
            df, data_pairing, state_comparison_type
        )

        # Get the statistical method for this condition
        method_mapping = get_statistical_method_mapping()
        condition_key = (comp_type, data_structure)
        statistical_method = method_mapping.get(
            condition_key, "Unknown method"
        )

        # Enhanced data summary
        n_observations = len(df)
        n_subjects = (
            df.get(
                "normalized_subject_id", df.get("subject_id", pd.Series())
            ).nunique()
            if not df.empty
            else 0
        )
        n_states = df["state"].nunique() if "state" in df.columns else 0

        # Flexible group detection (check both "group" and "group_name")
        actual_group_column = None
        n_groups = 1

        if "group" in df.columns and not df["group"].isnull().all():
            actual_group_column = "group"
            n_groups = df["group"].nunique()
        elif (
            "group_name" in df.columns and not df["group_name"].isnull().all()
        ):
            actual_group_column = "group_name"
            n_groups = df["group_name"].nunique()
        elif (
            group_column in df.columns and not df[group_column].isnull().all()
        ):
            actual_group_column = group_column
            n_groups = df[group_column].nunique()

        # Validate group structure consistency
        if n_groups < 2 and data_structure.startswith("multiple_groups"):
            logger.warning("Data structure mismatch detected!")
            logger.warning(
                f"Expected multiple groups for '{comp_type}' comparison, "
                f"but found only {n_groups} group(s)"
            )

            if actual_group_column:
                logger.warning(
                    f"Found group column: '{actual_group_column}' with values: "
                    f"{df[actual_group_column].unique()}"
                )
            else:
                checked_columns = ["group", "group_name", group_column]
                available_columns = [
                    col for col in checked_columns if col in df.columns
                ]
                logger.warning(
                    "No valid group column found. Checked: %s", checked_columns
                )
                if available_columns:
                    logger.warning(
                        "Available columns contained: %s (but all null/invalid)",
                        available_columns,
                    )
                else:
                    logger.warning(
                        "None of the expected group columns exist in data"
                    )

            # Automatically switch to single group analysis
            logger.info("Automatically switching to single-group analysis...")
            data_structure = "single_group"
            condition_key = (comp_type, data_structure)
            statistical_method = method_mapping.get(
                condition_key, "Single-group descriptive statistics"
            )
            logger.info(f"Updated method: {statistical_method}")
            logger.info(f"Updated condition: {comp_type} + {data_structure}")

        logger.info(f"{comparison_description} analysis configuration:")
        logger.info(f"  • Statistical method: {statistical_method}")
        logger.info(f"  • Design: {data_structure}")
        logger.info(f"  • Comparison type: {comp_type}")
        logger.info(
            f"  • Data summary: {n_observations} observations, {n_subjects} subjects, "
            f"{n_states} states, {n_groups} groups"
        )

        # Add explanation for descriptive statistics only
        if "Descriptive statistics only" in statistical_method:
            logger.info(
                "Note: No inferential statistics will be computed for this configuration"
            )
            logger.info(
                "      Consider grouping data differently for statistical comparisons"
            )

        # Check statistical power and collect warnings for modulation analysis
        statistical_warnings = []
        if not df.empty and "normalized_subject_id" in df.columns:
            has_concerns, power_warning = _validate_statistical_power(
                df, "normalized_subject_id", comparison_description.lower()
            )
            if has_concerns:
                statistical_warnings.append(power_warning)

        # Process each modulation status separately
        all_anova_results = []
        all_pairwise_results = []

        mod_statuses = status_values or ["up_modulated", "down_modulated"]
        valid_statuses = [s for s in mod_statuses if s in df["status"].values]

        logger.info(
            f"Processing {len(valid_statuses)} modulation statuses: {valid_statuses}"
        )

        for status in valid_statuses:
            logger.info(
                f"--- Processing {status} {comparison_description.lower()} data ---"
            )
            status_df = df[df["status"] == status].copy()

            # **Apply weighted proportions if explicitly enabled**
            if use_weighted_proportions:
                logger.info(
                    "Applying weighted proportions for %s to prevent pseudoreplication",
                    status,
                )
                status_df = _calculate_weighted_modulation_proportions(
                    status_df
                )

            # Update measure_column to use proportions instead of raw counts when available
            if use_weighted_proportions and "proportion" in status_df.columns:
                current_measure_column = "proportion"
                logger.info(
                    f"Using weighted proportions for {status} analysis"
                )
            else:
                current_measure_column = measure_column
                if use_weighted_proportions:
                    logger.warning(
                        "Weighted proportions not available for %s, using raw counts",
                        status,
                    )

            # Basic validation for status data
            # For single group scenarios, don't validate group column requirements
            # since we're comparing states within the single group, not between groups
            validation_group_column = None
            if data_structure != "single_group" and actual_group_column:
                validation_group_column = actual_group_column

            is_valid, error_msg, status_df = _validate_modulation_data(
                status_df,
                measure_column=current_measure_column,
                status_column="status",
                state_column="state",
                group_column=validation_group_column,
            )

            if not is_valid:
                logger.debug(
                    f"Skipping {status} due to validation failure: {error_msg}"
                )
                continue

            if len(status_df) < 2:
                logger.debug(
                    f"Skipping {status} due to insufficient data after validation"
                )
                continue

            # Dispatch to direct statistical functions using weighted measure
            anova_results, pairwise_results = _dispatch_direct_analysis(
                condition_key,
                status_df,
                status,
                states,
                correction,
                effect_size,
                current_measure_column,
                data_pairing,
            )

            logger.info(
                f"Completed {status} analysis: {len(anova_results)} ANOVA, "
                f"{len(pairwise_results)} pairwise results"
            )

            # Add context about data completeness
            if hasattr(status_df, "state") and "state" in status_df.columns:
                available_states = status_df["state"].nunique()
                total_combinations = len(status_df)
                logger.info(
                    f"    Data completeness: {available_states} states, "
                    f"{total_combinations} observations for {status}"
                )
            all_anova_results.extend(anova_results)
            all_pairwise_results.extend(pairwise_results)

        # Create result DataFrames
        anova_df = (
            pd.DataFrame(all_anova_results)
            if all_anova_results
            else pd.DataFrame()
        )
        pairwise_df = (
            pd.DataFrame(all_pairwise_results)
            if all_pairwise_results
            else pd.DataFrame()
        )

        # Add comparison description
        for df_result in [anova_df, pairwise_df]:
            if not df_result.empty:
                df_result["Comparison"] = comparison_description

        # Add pairing columns and update stat_method
        if not pairwise_df.empty:
            try:
                pairwise_df = _add_pairing_columns(
                    pairwise_df,
                    state_comparison_type=comp_type,
                    data_pairing=data_pairing,
                )
            except Exception as e:
                logger.warning(
                    f"Could not add pairing columns for {comparison_description.lower()}: {str(e)}"
                )

        # Ensure p-corr column exists (handled at individual result level)
        # No need for global correction since pingouin handles this via padjust parameter
        if not pairwise_df.empty:
            if correction == "none":
                logger.info(
                    "No correction applied for %d %s pairwise results (p-corr = p-unc)",
                    len(pairwise_df),
                    comparison_description.lower(),
                )
            else:
                logger.info(
                    "Using pingouin built-in %s correction for %d %s pairwise results",
                    correction,
                    len(pairwise_df),
                    comparison_description.lower(),
                )

        # Apply final output standardization
        if not anova_df.empty:
            # Apply standardization first to extract df1/df2 from pingouin format
            try:
                metadata = {
                    "Comparison": comparison_description,
                    "Measure": comparison_description,
                }
                anova_df = _standardize_anova_output(
                    anova_df,
                    metadata=metadata,
                    analysis_level="subject",
                    state_comparison_type="state",
                    measure_name=comparison_description,
                )
            except Exception as e:
                logger.warning(
                    f"Could not standardize {comparison_description.lower()} "
                    f"ANOVA output: {str(e)}"
                )

            anova_df = _finalize_statistical_output(anova_df, "anova")
        if not pairwise_df.empty:
            pairwise_df = _finalize_statistical_output(pairwise_df, "pairwise")

        # Add statistical warnings to results if any exist (similar to LMM analysis)
        if statistical_warnings:
            warning_text = "; ".join(statistical_warnings)

            # Add warnings to both ANOVA and pairwise results for modulation analysis
            if not anova_df.empty:
                anova_df["statistical_warnings"] = warning_text
                anova_df[
                    "analysis_notes"
                ] = "Results generated with statistical concerns - see warnings"

        logger.info(
            f"=== {comparison_description.upper()} Analysis Results Summary ==="
        )
        logger.info(
            f"✓ Direct analysis completed: {len(all_anova_results)} ANOVA, "
            f"{len(all_pairwise_results)} pairwise results"
        )
        return anova_df, pairwise_df

    except Exception as e:
        logger.error(
            f"Direct {comparison_description.lower()} analysis failed: "
            f"{type(e).__name__}: {str(e)}"
        )
        return pd.DataFrame(), pd.DataFrame()


def _get_corrected_p_value(row: dict, correction: str) -> float:
    """Get corrected p-value, ensuring NaN consistency for failed tests.

    Parameters
    ----------
    row : dict
        Row data from statistical results
    correction : str
        Correction method used

    Returns
    -------
    float
        Corrected p-value or np.nan if test failed

    """
    # If p-unc is NaN (test failed), p-corr should also be NaN
    if pd.isna(row.get("p-unc", np.nan)):
        return np.nan

    # Handle p-corr column based on correction type
    # When padjust="none", pingouin doesn't create p-corr column
    if correction == "none":
        return row.get("p-unc", np.nan)
    else:
        return row.get("p-corr", row.get("p-unc", np.nan))


def _dispatch_direct_analysis(
    condition_key: Tuple[str, str],
    status_df: pd.DataFrame,
    status: str,
    states: List[str],
    correction: str,
    effect_size: str,
    measure_column: str,
    data_pairing: str,
) -> Tuple[List[Any], List[Any]]:
    """Dispatch to direct statistical analysis based on condition."""
    if condition_key == ("pairwise", "single_group"):
        # Check for single state + single group = no comparison possible
        if len(states) == 1 or (
            hasattr(status_df, "state") and status_df["state"].nunique() == 1
        ):
            state_name = (
                states[0]
                if len(states) == 1
                else (
                    status_df["state"].iloc[0]
                    if not status_df.empty
                    else "unknown"
                )
            )
            logger.info(f"No statistical tests performed for {status}:")
            logger.info(
                "  • Single group with pairwise states ('%s') detected",
                state_name,
            )
            logger.info(
                "  • Statistical comparisons require multiple groups OR multiple states per group"
            )
            logger.info(
                "  • Consider using descriptive statistics or different grouping"
            )
        else:
            logger.info(f"No statistical tests performed for {status}:")
            logger.info(
                "  • Single group design with pairwise comparison not supported"
            )
            logger.info(
                "  • Use state comparison within group or between-group analysis"
            )
        return [], []

    elif condition_key in [
        ("pairwise", "multiple_groups_paired"),
        ("pairwise", "multiple_groups_unpaired"),
    ]:
        # For pairwise modulation comparisons, use state comparison function
        return _direct_pairwise_state_comparison(
            status_df,
            status,
            states,
            measure_column,
            correction,
            effect_size,
            data_pairing,
        )

    elif condition_key in [
        ("state_vs_baseline", "single_group"),
        ("state_vs_not_state", "single_group"),
    ]:
        return _direct_single_group_anova(
            status_df,
            status,
            states,
            measure_column,
            correction,
            effect_size,
            "state_vs_not_state",
        )

    elif condition_key in [
        ("state_vs_baseline", "multiple_groups_paired"),
        ("state_vs_not_state", "multiple_groups_paired"),
    ]:
        # Check if we actually have only one state - if so, compare groups instead
        if len(states) == 1 or (
            hasattr(status_df, "state") and status_df["state"].nunique() == 1
        ):
            logger.info(
                "Single state detected for %s, using group comparison instead of RM ANOVA",
                condition_key,
            )
            return _direct_pairwise_multiple_groups(
                status_df,
                status,
                states,
                measure_column,
                correction,
                effect_size,
                data_pairing,
            )
        else:
            return _direct_multiple_groups_rm_anova(
                status_df,
                status,
                states,
                measure_column,
                correction,
                effect_size,
                "state_vs_not_state",
            )

    elif condition_key in [
        ("state_vs_baseline", "multiple_groups_unpaired"),
        ("state_vs_not_state", "multiple_groups_unpaired"),
    ]:
        # Check if we actually have only one state - if so, compare groups instead
        if len(states) == 1 or (
            hasattr(status_df, "state") and status_df["state"].nunique() == 1
        ):
            logger.info(
                "Single state detected for %s, using group comparison instead of mixed ANOVA",
                condition_key,
            )
            return _direct_pairwise_multiple_groups(
                status_df,
                status,
                states,
                measure_column,
                correction,
                effect_size,
                data_pairing,
            )
        else:
            return _direct_multiple_groups_mixed_anova(
                status_df,
                status,
                states,
                measure_column,
                correction,
                effect_size,
                "state_vs_not_state",
            )

    else:
        logger.warning(f"No direct handler for condition: {condition_key}")
        return [], []


# =============================================================================
# DIRECT STATISTICAL ANALYSIS FUNCTIONS (NO LEGACY CODE)
# =============================================================================


@_suppress_pingouin_warnings
def _direct_pairwise_multiple_groups(
    status_df: pd.DataFrame,
    status: str,
    states: List[str],
    measure_column: str,
    correction: str,
    effect_size: str,
    data_pairing: str,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Direct pairwise analysis for multiple groups using validated pingouin calls."""
    pairwise_results = []

    try:
        # Validate input data for group comparisons
        is_valid, error_msg, validated_df = _validate_modulation_group_data(
            status_df,
            measure_column=measure_column,
            group_column="group",
            state_column="state",
            data_pairing=data_pairing,
        )

        if not is_valid:
            logger.warning(
                f"Pairwise state comparison validation failed for {status}: {error_msg}"
            )
            logger.info(
                f"Skipping all state contrasts for {status} due to validation failure"
            )
            return [], pairwise_results

        # For each state, compare between groups
        for state in states:
            # Check if state column exists, handle both original and validated data
            if "state" in validated_df.columns:
                state_data = validated_df[validated_df["state"] == state]

            if len(state_data) < 2:
                continue

            groups = state_data["group"].unique()
            if len(groups) == 2:
                # Use safe t-test function with validation
                ttest_result = _safe_ttest_modulation(
                    data=state_data,
                    measure_column=measure_column,
                    group_column="group",
                    groups=list(groups),
                    paired=(data_pairing == "paired"),
                )

                if ttest_result is not None and not ttest_result.empty:
                    group1_data = state_data[state_data["group"] == groups[0]][
                        measure_column
                    ]
                    group2_data = state_data[state_data["group"] == groups[1]][
                        measure_column
                    ]

                    t_val = ttest_result["T"].iloc[0]
                    p_val = ttest_result["p-val"].iloc[0]
                    cohen_d = ttest_result.get(
                        "cohen-d", pd.Series([np.nan])
                    ).iloc[0]

                    result_dict = {
                        "Status": status,
                        "State": state,
                        "Contrast": f"{groups[0]} vs {groups[1]}",
                        "A": groups[0],
                        "B": groups[1],
                        "mean(A)": group1_data.mean(),
                        "mean(B)": group2_data.mean(),
                        "T": t_val,
                        "p-unc": p_val,
                        "cohen-d": cohen_d,
                        "analysis_level": "subject",
                        "state_comparison_type": "group",
                        "Measure": (
                            f"{status.replace('_modulated', '_modulation')}"
                            if status.endswith("_modulated")
                            else status
                        ),
                    }
                    pairwise_results.append(result_dict)

    except Exception as e:
        logger.error(f"Direct pairwise analysis failed: {str(e)}")

    return [], pairwise_results


@_suppress_pingouin_warnings
def _direct_single_group_anova(
    status_df: pd.DataFrame,
    status: str,
    states: List[str],
    measure_column: str,
    correction: str,
    effect_size: str,
    state_comparison_type: str = "state_vs_not_state",
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Direct single group ANOVA using validated pingouin calls.

    For state comparisons within a single group, we use repeated measures ANOVA
    since the same subjects/cells are measured across different states.
    """
    anova_results = []
    pairwise_results = []

    try:
        # Validate input data
        is_valid, error_msg, validated_df = _validate_modulation_data(
            status_df,
            measure_column=measure_column,
            status_column="status",
            state_column="state",
        )

        if not is_valid:
            logger.debug(
                f"Single group validation failed for {status}: {error_msg}"
            )
            return [], []

        if len(states) < 2 or validated_df["state"].nunique() < 2:
            logger.debug(
                f"Insufficient states for ANOVA: {validated_df['state'].nunique()}"
            )
            return [], []

        # Check if we have subject ID for repeated measures
        subject_col = None
        if "normalized_subject_id" in validated_df.columns:
            subject_col = "normalized_subject_id"
        elif "subject_id" in validated_df.columns:
            subject_col = "subject_id"
        elif "file" in validated_df.columns:
            # Use file as subject identifier if no explicit subject ID
            validated_df["subject_id"] = validated_df["file"]
            subject_col = "subject_id"

        # Determine statistical approach based on data structure
        if subject_col and validated_df[subject_col].nunique() >= 2:
            # Check data quality for repeated measures ANOVA
            n_subjects = validated_df[subject_col].nunique()
            n_states = validated_df["state"].nunique()

            # Check for sufficient data per subject-state combination
            obs_per_subj_state = validated_df.groupby(
                [subject_col, "state"]
            ).size()
            min_obs_per_comb = obs_per_subj_state.min()

            # Check for variance in the measure across subjects and states
            state_means = validated_df.groupby("state")[measure_column].mean()
            subject_means = validated_df.groupby(subject_col)[
                measure_column
            ].mean()

            # Log diagnostic information
            logger.debug(
                f"RM ANOVA diagnostics for {status}: {n_subjects} subjects, "
                f"{n_states} states"
            )
            logger.debug(
                f"Min observations per subject-state: {min_obs_per_comb}"
            )
            logger.debug(
                f"State variance: {state_means.var():.6f}, "
                f"Subject variance: {subject_means.var():.6f}"
            )

            if (
                n_subjects >= 2
                and n_states >= 2
                and min_obs_per_comb >= 1
                and state_means.var() > 1e-10
                and subject_means.var() > 1e-10
            ):

                # Use repeated measures ANOVA (within-subject design)
                # This is appropriate for state comparisons
                # where same subjects measured across states
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*divide by zero.*"
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*invalid value.*"
                    )

                    anova_result = _safe_anova(
                        data=validated_df,
                        dv=measure_column,
                        within="state",
                        subject=subject_col,
                        detailed=True,
                    )

                stat_method = "rm_anova"
                pairwise_paired = True

                # Additional validation of ANOVA result for sphericity issues
                if anova_result is not None and not anova_result.empty:
                    # Check if F statistics or p-values are problematic
                    f_vals = anova_result.get("F", pd.Series([]))
                    p_vals = anova_result.get("p-unc", pd.Series([]))

                    if (
                        f_vals.isna().all()
                        or np.isinf(f_vals).any()
                        or p_vals.isna().all()
                        or (p_vals == 0).all()
                    ):
                        logger.warning(
                            "RM ANOVA for %s shows signs of sphericity/degeneracy issues, "
                            "falling back to between-subject ANOVA",
                            status,
                        )
                        anova_result = None

            else:
                logger.debug(
                    "Data quality insufficient for RM ANOVA for %s, using between-subject ANOVA",
                    status,
                )
                anova_result = None

            # Fallback to between-subject if RM ANOVA failed
            if anova_result is None:
                anova_result = _safe_anova(
                    data=validated_df,
                    dv=measure_column,
                    between="state",
                    detailed=True,
                )
                stat_method = "one_way_anova"
                pairwise_paired = False
        else:
            # Fallback to between-subject ANOVA if no proper subject grouping
            anova_result = _safe_anova(
                data=validated_df,
                dv=measure_column,
                between="state",
                detailed=True,
            )
            stat_method = "one_way_anova"
            pairwise_paired = False

        if anova_result is not None and not anova_result.empty:
            # Use _standardize_anova_output for consistent df1/df2 extraction
            anova_result["stat_method"] = stat_method
            standardized_anova = _standardize_anova_output(
                anova_result,
                metadata={"Comparison": status, "Measure": status},
                analysis_level="subject",
                state_comparison_type="state",
                measure_name=status,
            )

            for idx, row in standardized_anova.iterrows():
                anova_results.append(
                    {
                        "Status": status,
                        "Measure": f"{status.replace('_modulated', '_modulation')}",
                        "Source": row.get("Source", ""),
                        "p_value": row.get("p_value", np.nan),
                        "F_statistic": row.get("F_statistic", np.nan),
                        "df1": row.get("df1", np.nan),
                        "df2": row.get("df2", np.nan),
                        "mean_square": row.get("mean_square", np.nan),
                        "sum_squares": row.get("sum_squares", np.nan),
                        "stat_method": stat_method,
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        # Keep original columns for compatibility
                        "p-unc": row.get("p-unc", np.nan),
                        "F": row.get("F", np.nan),
                        "DF": row.get("DF", np.nan),
                        "MS": row.get("MS", np.nan),
                        "SS": row.get("SS", np.nan),
                    }
                )

        # Use appropriate pairwise tests based on design
        if pairwise_paired and subject_col:
            # Paired comparisons for repeated measures design
            pairwise_result = _safe_pairwise_ttests(
                data=validated_df,
                dv=measure_column,
                within="state",
                subject=subject_col,
                padjust=correction,
                effsize=effect_size,
            )

            # Ensure the Paired column is correctly set for repeated measures
            if pairwise_result is not None and not pairwise_result.empty:
                pairwise_result["Paired"] = True
        else:
            # Unpaired comparisons for between-subject design
            pairwise_result = _safe_pairwise_ttests(
                data=validated_df,
                dv=measure_column,
                between="state",
                padjust=correction,
                effsize=effect_size,
            )

            # Ensure the Paired column is correctly set for between-subject design
            if pairwise_result is not None and not pairwise_result.empty:
                pairwise_result["Paired"] = False

        if pairwise_result is not None and not pairwise_result.empty:
            for idx, row in pairwise_result.iterrows():
                p_corr_value = _get_corrected_p_value(row, correction)

                pairwise_results.append(
                    {
                        "Status": status,
                        "Measure": f"{status.replace('_modulated', '_modulation')}",
                        "Contrast": row.get("Contrast", ""),
                        "A": row.get("A", ""),
                        "B": row.get("B", ""),
                        "T": row.get("T", np.nan),
                        "p-unc": row.get("p-unc", np.nan),
                        "p-corr": p_corr_value,
                        "hedges": row.get("hedges", np.nan),
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        "Paired": row.get("Paired", True),
                        "Parametric": row.get("Parametric", True),
                        # Preserve pingouin's state information
                        "state": row.get("state", np.nan),
                    }
                )

    except Exception as e:
        logger.error(f"Direct single group ANOVA failed: {str(e)}")

    return anova_results, pairwise_results


@_suppress_pingouin_warnings
def _direct_multiple_groups_rm_anova(
    status_df: pd.DataFrame,
    status: str,
    states: List[str],
    measure_column: str,
    correction: str,
    effect_size: str,
    state_comparison_type: str = "state_vs_not_state",
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Direct repeated measures ANOVA using validated pingouin calls."""
    anova_results = []
    pairwise_results = []

    try:
        # Validate input data for repeated measures design
        is_valid, error_msg, validated_df = _validate_modulation_group_data(
            status_df,
            measure_column=measure_column,
            group_column="group",
            state_column="state",
            data_pairing="paired",
        )

        if not is_valid:
            logger.debug(
                f"RM ANOVA validation failed for {status}: {error_msg}"
            )
            return [], []

        required_cols = [
            "state",
            "group",
            measure_column,
            "normalized_subject_id",
        ]

        if not all(col in validated_df.columns for col in required_cols):
            logger.warning(
                "Missing required columns for RM ANOVA: %s",
                [
                    col
                    for col in required_cols
                    if col not in validated_df.columns
                ],
            )
            return [], []

        # Use safe repeated measures ANOVA
        rm_anova_result = _safe_anova(
            data=validated_df,
            dv=measure_column,
            within=["state", "group"],
            subject="normalized_subject_id",
            detailed=True,
        )

        if rm_anova_result is not None and not rm_anova_result.empty:
            # Use _standardize_anova_output for consistent df1/df2 extraction
            rm_anova_result["stat_method"] = "rm_anova"
            standardized_anova = _standardize_anova_output(
                rm_anova_result,
                metadata={"Comparison": status, "Measure": status},
                analysis_level="subject",
                state_comparison_type="state",
                measure_name=status,
            )

            for idx, row in standardized_anova.iterrows():
                anova_results.append(
                    {
                        "Status": status,
                        "Measure": f"{status.replace('_modulated', '_modulation')}",
                        "Source": row.get("Source", ""),
                        "p_value": row.get("p_value", np.nan),
                        "F_statistic": row.get("F_statistic", np.nan),
                        "df1": row.get("df1", np.nan),
                        "df2": row.get("df2", np.nan),
                        "mean_square": row.get("mean_square", np.nan),
                        "sum_squares": row.get("sum_squares", np.nan),
                        "stat_method": "rm_anova",
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        # Keep original columns for compatibility
                        "p-unc": row.get("p-unc", np.nan),
                        "F": row.get("F", np.nan),
                        "DF": row.get("DF", np.nan),
                        "MS": row.get("MS", np.nan),
                        "SS": row.get("SS", np.nan),
                    }
                )

        # Use safe pairwise tests
        pairwise_result = _safe_pairwise_ttests(
            data=validated_df,
            dv=measure_column,
            within=["state", "group"],
            subject="normalized_subject_id",
            padjust=correction,
            effsize=effect_size,
        )

        # Ensure the Paired column is correctly set for repeated measures
        if pairwise_result is not None and not pairwise_result.empty:
            pairwise_result["Paired"] = True

        if pairwise_result is not None and not pairwise_result.empty:
            for idx, row in pairwise_result.iterrows():
                p_corr_value = _get_corrected_p_value(row, correction)

                pairwise_results.append(
                    {
                        "Status": status,
                        "Measure": f"{status.replace('_modulated', '_modulation')}",
                        "Contrast": row.get("Contrast", ""),
                        "A": row.get("A", ""),
                        "B": row.get("B", ""),
                        "T": row.get("T", np.nan),
                        "p-unc": row.get("p-unc", np.nan),
                        "p-corr": p_corr_value,
                        "hedges": row.get("hedges", np.nan),
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        "Paired": row.get("Paired", True),
                        "Parametric": row.get("Parametric", True),
                        # Preserve pingouin's state information
                        "state": row.get("state", np.nan),
                    }
                )

    except Exception as e:
        logger.error(f"Direct RM ANOVA failed: {str(e)}")

    return anova_results, pairwise_results


@_suppress_pingouin_warnings
def _direct_multiple_groups_mixed_anova(
    status_df: pd.DataFrame,
    status: str,
    states: List[str],
    measure_column: str,
    correction: str,
    effect_size: str,
    state_comparison_type: str = "state_vs_not_state",
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Direct mixed ANOVA using validated pingouin calls."""
    anova_results = []
    pairwise_results = []

    try:
        # Validate input data for mixed design
        is_valid, error_msg, validated_df = _validate_modulation_group_data(
            status_df,
            measure_column=measure_column,
            group_column="group",
            state_column="state",
            data_pairing="unpaired",
        )

        if not is_valid:
            logger.debug(
                f"Mixed ANOVA validation failed for {status}: {error_msg}"
            )
            return [], []

        required_cols = [
            "state",
            "group",
            measure_column,
            "normalized_subject_id",
        ]

        if not all(col in validated_df.columns for col in required_cols):
            logger.warning(
                "Missing required columns for Mixed ANOVA: %s",
                [
                    col
                    for col in required_cols
                    if col not in validated_df.columns
                ],
            )
            return [], []

        # Use safe mixed ANOVA
        mixed_anova_result = _safe_anova(
            data=validated_df,
            dv=measure_column,
            within="state",
            between="group",
            subject="normalized_subject_id",
        )

        if mixed_anova_result is not None and not mixed_anova_result.empty:
            # Use _standardize_anova_output for consistent df1/df2 extraction
            mixed_anova_result["stat_method"] = "mixed_anova"
            standardized_anova = _standardize_anova_output(
                mixed_anova_result,
                metadata={"Comparison": status, "Measure": status},
                analysis_level="subject",
                state_comparison_type="state",
                measure_name=status,
            )

            for idx, row in standardized_anova.iterrows():
                anova_results.append(
                    {
                        "Status": status,
                        "Measure": f"{status.replace('_modulated', '_modulation')}",
                        "Source": row.get("Source", ""),
                        "p_value": row.get("p_value", np.nan),
                        "F_statistic": row.get("F_statistic", np.nan),
                        "df1": row.get("df1", np.nan),
                        "df2": row.get("df2", np.nan),
                        "mean_square": row.get("mean_square", np.nan),
                        "sum_squares": row.get("sum_squares", np.nan),
                        "stat_method": "mixed_anova",
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        # Keep original columns for compatibility
                        "p-unc": row.get("p-unc", np.nan),
                        "F": row.get("F", np.nan),
                        "DF": row.get("DF", np.nan),
                        "MS": row.get("MS", np.nan),
                        "SS": row.get("SS", np.nan),
                    }
                )

        # Use safe pairwise tests
        pairwise_result = _safe_pairwise_ttests(
            data=validated_df,
            dv=measure_column,
            within="state",
            between="group",
            subject="normalized_subject_id",
            padjust=correction,
            effsize=effect_size,
        )

        # Ensure the Paired column is correctly set for repeated measures
        if pairwise_result is not None and not pairwise_result.empty:
            pairwise_result["Paired"] = True

        if pairwise_result is not None and not pairwise_result.empty:
            for idx, row in pairwise_result.iterrows():
                p_corr_value = _get_corrected_p_value(row, correction)

                pairwise_results.append(
                    {
                        "Status": status,
                        "Measure": f"{status.replace('_modulated', '_modulation')}",
                        "Contrast": row.get("Contrast", ""),
                        "A": row.get("A", ""),
                        "B": row.get("B", ""),
                        "T": row.get("T", np.nan),
                        "p-unc": row.get("p-unc", np.nan),
                        "p-corr": p_corr_value,
                        "hedges": row.get("hedges", np.nan),
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        "Paired": row.get("Paired", True),
                        "Parametric": row.get("Parametric", True),
                        # Preserve pingouin's state information
                        "state": row.get("state", np.nan),
                    }
                )

    except Exception as e:
        logger.error(f"Direct Mixed ANOVA failed: {str(e)}")

    return anova_results, pairwise_results


def _handle_single_state_group_comparison(
    *,
    group_df: pd.DataFrame,
    correction: str,
    effect_size: str,
    data_pairing: str,
    measure_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Handle single state, multiple groups comparison using direct group comparison.

    For single state scenarios, we perform simple group comparisons rather than
    trying to use mixed ANOVA with within="state" which doesn't make sense.

    Parameters
    ----------
    group_df : pd.DataFrame
        Subject-averaged DataFrame with single state and multiple groups
    correction : str
        Multiple comparison correction method
    effect_size : str
        Effect size measure to use
    data_pairing : str
        Data pairing method ("paired" or "unpaired")
    measure_name : str
        Name of the measure being analyzed

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ANOVA results, pairwise results, and subject-averaged DataFrame

    """
    try:

        aov_df = pd.DataFrame()
        pairwise_df = pd.DataFrame()

        # Get unique groups
        groups = group_df["group"].unique()
        n_groups = len(groups)

        if n_groups == 2:
            # Simple t-test between two groups
            logger.info("Performing t-test between 2 groups for single state")

            # Use safe t-test function
            ttest_result = _safe_ttest_modulation(
                data=group_df,
                measure_column="activity",
                group_column="group",
                groups=list(groups),
                paired=(data_pairing == "paired"),
            )

            if ttest_result is not None and not ttest_result.empty:
                # Extract group data for means
                group1_data = group_df[group_df["group"] == groups[0]][
                    "activity"
                ]
                group2_data = group_df[group_df["group"] == groups[1]][
                    "activity"
                ]

                state_name = group_df["state"].iloc[0]

                # Format as pairwise result
                pairwise_result = {
                    "State": state_name,
                    "Contrast": f"{groups[0]} vs {groups[1]}",
                    "A": groups[0],
                    "B": groups[1],
                    "mean(A)": group1_data.mean(),
                    "mean(B)": group2_data.mean(),
                    "T": ttest_result["T"].iloc[0],
                    "p-unc": ttest_result["p-val"].iloc[0],
                    "cohen-d": ttest_result.get(
                        "cohen-d", pd.Series([np.nan])
                    ).iloc[0],
                    "Comparison": (
                        measure_name.capitalize()
                        if measure_name
                        else "Group Comparison"
                    ),
                    "Measure": measure_name,
                    "state_comparison_type": "group",
                    "analysis_level": "subject",
                }
                pairwise_df = pd.DataFrame([pairwise_result])

                # Add pairing columns and update stat_method
                try:
                    pairwise_df = _add_pairing_columns(
                        pairwise_df,
                        state_comparison_type="group",
                        data_pairing=data_pairing,
                    )
                except Exception as e:
                    logger.warning(f"Could not add pairing columns: {str(e)}")

        elif n_groups > 2:
            # One-way ANOVA between groups (no within-state factor)
            logger.info(
                f"Performing one-way ANOVA between {n_groups} groups for single state"
            )

            # Use safe ANOVA function - just between groups, no within factor
            aov_result = _safe_anova(
                data=group_df,
                dv="activity",
                between="group",
                detailed=True,
            )

            if aov_result is not None and not aov_result.empty:
                state_name = group_df["state"].iloc[0]

                # Format ANOVA result
                aov_rows = []
                for idx, row in aov_result.iterrows():
                    aov_rows.append(
                        {
                            "State": state_name,
                            "Source": row.get("Source", "group"),
                            "p-unc": row.get("p-unc", np.nan),
                            "F": row.get("F", np.nan),
                            "DF1": row.get("ddof1", np.nan),
                            "DF2": row.get("ddof2", np.nan),
                            "MS": row.get("MS", np.nan),
                            "SS": row.get("SS", np.nan),
                            "Comparison": (
                                measure_name.capitalize()
                                if measure_name
                                else "Group Comparison"
                            ),
                            "Measure": measure_name,
                            "state_comparison_type": "group",
                            "analysis_level": "subject",
                        }
                    )
                aov_df = pd.DataFrame(aov_rows)

            # Pairwise comparisons between groups
            pairwise_result = _safe_pairwise_ttests(
                data=group_df,
                dv="activity",
                between="group",
                padjust=correction,
                effsize=effect_size,
            )

            if pairwise_result is not None and not pairwise_result.empty:
                state_name = group_df["state"].iloc[0]

                # Add metadata to pairwise results
                pairwise_result["State"] = state_name
                pairwise_result["Comparison"] = (
                    measure_name.capitalize()
                    if measure_name
                    else "Group Comparison"
                )
                pairwise_result["Measure"] = measure_name
                pairwise_result["state_comparison_type"] = "group"
                pairwise_result["analysis_level"] = "subject"

                pairwise_df = pairwise_result

                # Add pairing columns and update stat_method
                try:
                    pairwise_df = _add_pairing_columns(
                        pairwise_df,
                        state_comparison_type="group",
                        data_pairing=data_pairing,
                    )
                except Exception as e:
                    logger.warning(f"Could not add pairing columns: {str(e)}")

        # Apply final output standardization

        if not aov_df.empty:
            aov_df = _finalize_statistical_output(aov_df, "anova")
        if not pairwise_df.empty:
            pairwise_df = _finalize_statistical_output(pairwise_df, "pairwise")

        logger.info(
            f"Single state group comparison completed: {len(aov_df)} ANOVA, "
            f"{len(pairwise_df)} pairwise results"
        )
        return aov_df, pairwise_df, group_df

    except Exception as e:
        logger.error(
            f"Single state group comparison failed: {type(e).__name__}: {str(e)}"
        )
        return pd.DataFrame(), pd.DataFrame(), group_df


def _log_missing_state_contrasts(
    status: str,
    expected_pairs: List[Tuple[str, str]],
    processed_pairs: List[Tuple[str, str]],
    skipped_pairs: Dict[Tuple[str, str], str],
) -> None:
    """Log comprehensive information about missing state contrast combinations."""
    missing_pairs = [
        pair for pair in expected_pairs if pair not in processed_pairs
    ]

    if missing_pairs:
        logger.debug(
            f"Missing {len(missing_pairs)} state contrasts for {status}:"
        )
        for pair in missing_pairs:
            reason = skipped_pairs.get(pair, "Unknown reason")
            logger.debug(f"  - '{pair}': {reason}")

        logger.info(
            f"State contrast summary for {status}: Expected: {len(expected_pairs)}, "
            f"Processed: {len(processed_pairs)}, Missing: {len(missing_pairs)}"
        )

        if processed_pairs:
            logger.info(f"  Successfully processed: {processed_pairs}")


@_suppress_pingouin_warnings
def _direct_pairwise_state_comparison(
    status_df: pd.DataFrame,
    status: str,
    states: List[str],
    measure_column: str,
    correction: str,
    effect_size: str,
    data_pairing: str,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Direct pairwise state comparison for modulation data.

    For pairwise modulation comparisons where states are in "state A vs state B" format,
    this function compares Group 1 vs Group 2 for each state pair.
    The output shows A="state A", B="state B" (comparing states between groups).
    """
    pairwise_results = []

    # Track missing combinations for comprehensive logging
    expected_pairs = states.copy()
    processed_pairs = []
    skipped_pairs = {}

    try:
        # Validate input data for group comparisons
        is_valid, error_msg, validated_df = _validate_modulation_group_data(
            status_df,
            measure_column=measure_column,
            group_column="group",
            state_column="state",
            data_pairing=data_pairing,
        )

        if not is_valid:
            logger.warning(
                f"Pairwise state comparison validation failed for {status}: {error_msg}"
            )
            # Mark all pairs as skipped due to validation failure
            for pair in expected_pairs:
                skipped_pairs[pair] = f"Validation failure: {error_msg}"
            _log_missing_state_contrasts(
                status, expected_pairs, processed_pairs, skipped_pairs
            )
            return [], pairwise_results

        logger.info(
            f"Processing pairwise state comparisons for {status}: expecting "
            f"{len(expected_pairs)} state contrasts"
        )

        # For each pairwise state comparison (e.g., "baseline vs active")
        for state_pair in states:
            if " vs " not in state_pair:
                logger.warning(
                    f"State '{state_pair}' is not in expected pairwise format 'A vs B'"
                )
                skipped_pairs[
                    state_pair
                ] = "Invalid format: missing ' vs ' separator"
                continue

            # Extract individual states from the pair
            state_a, state_b = state_pair.split(" vs ")
            state_a = state_a.strip()
            state_b = state_b.strip()

            # Get data for this state pair
            state_data = validated_df[validated_df["state"] == state_pair]

            if len(state_data) < 2:
                skipped_pairs[
                    state_pair
                ] = f"Insufficient data: only {len(state_data)} records found, need ≥2"
                continue

            groups = state_data["group"].unique()
            if len(groups) == 2:
                # Use safe t-test function with validation
                ttest_result = _safe_ttest_modulation(
                    data=state_data,
                    measure_column=measure_column,
                    group_column="group",
                    groups=list(groups),
                    paired=(data_pairing == "paired"),
                )

                if ttest_result is not None and not ttest_result.empty:
                    group1_data = state_data[state_data["group"] == groups[0]][
                        measure_column
                    ]
                    group2_data = state_data[state_data["group"] == groups[1]][
                        measure_column
                    ]

                    t_val = ttest_result["T"].iloc[0]
                    p_val = ttest_result["p-val"].iloc[0]
                    cohen_d = ttest_result.get(
                        "cohen-d", pd.Series([np.nan])
                    ).iloc[0]

                    result_dict = {
                        "Status": status,
                        "State": "",
                        "Contrast": f"{state_a} vs {state_b}",
                        "A": state_a,
                        "B": state_b,
                        "mean(A)": group1_data.mean(),
                        "mean(B)": group2_data.mean(),
                        "T": t_val,
                        "p-unc": p_val,
                        "cohen-d": cohen_d,
                        "analysis_level": "subject",
                        "state_comparison_type": "state",
                        "state_pairing": "not_applicable",
                        "group_pairing": data_pairing,
                        "Measure": (
                            f"{status.replace('_modulated', '_modulation')}"
                            if status.endswith("_modulated")
                            else status
                        ),
                    }
                    pairwise_results.append(result_dict)
                    processed_pairs.append(state_pair)
                else:
                    skipped_pairs[
                        state_pair
                    ] = "T-test failed: invalid or empty statistical result"
            else:
                skipped_pairs[state_pair] = (
                    f"Insufficient groups: found {len(groups)} groups "
                    f"{list(groups)}, need exactly 2"
                )

        # Add missing state contrasts as rows with failure reasons
        for state_pair in expected_pairs:
            if state_pair not in processed_pairs:
                reason = skipped_pairs.get(state_pair, "Unknown reason")

                # Extract states for A and B columns
                if " vs " in state_pair:
                    state_a, state_b = state_pair.split(" vs ")
                    state_a = state_a.strip()
                    state_b = state_b.strip()
                else:
                    state_a = state_pair
                    state_b = "unknown"

                # Create a row indicating the failure
                # Use np.nan for numeric columns to maintain data type consistency
                failure_dict = {
                    "Status": status,
                    "State": "",
                    "Contrast": state_pair,
                    "A": state_a,
                    "B": state_b,
                    "mean(A)": np.nan,
                    "mean(B)": np.nan,
                    "T": np.nan,
                    "p-unc": np.nan,
                    "cohen-d": np.nan,
                    "analysis_level": "subject",
                    "state_comparison_type": "state",
                    "state_pairing": "not_applicable",
                    "group_pairing": data_pairing,
                    "Measure": (
                        f"{status.replace('_modulated', '_modulation')}"
                        if status.endswith("_modulated")
                        else status
                    ),
                    "failure_reason": reason,
                }

                # Log warning about failed statistical test
                logger.warning(
                    f"Statistical test could not be computed for {status} {state_pair}: {reason}"
                )
                pairwise_results.append(failure_dict)

        # Log comprehensive summary of missing state contrasts
        _log_missing_state_contrasts(
            status, expected_pairs, processed_pairs, skipped_pairs
        )

    except Exception as e:
        logger.error(f"Direct pairwise state comparison failed: {str(e)}")
        # Mark any unprocessed pairs as failed due to exception and add them as rows
        for pair in expected_pairs:
            if pair not in processed_pairs and pair not in skipped_pairs:
                reason = f"Exception during processing: {str(e)}"
                skipped_pairs[pair] = reason

                # Add failure row for exception cases
                if " vs " in pair:
                    state_a, state_b = pair.split(" vs ")
                    state_a = state_a.strip()
                    state_b = state_b.strip()
                else:
                    state_a = pair
                    state_b = "unknown"

                # Use np.nan for numeric columns to maintain data type consistency
                failure_dict = {
                    "Status": status,
                    "State": "",
                    "Contrast": pair,
                    "A": state_a,
                    "B": state_b,
                    "mean(A)": np.nan,
                    "mean(B)": np.nan,
                    "T": np.nan,
                    "p-unc": np.nan,
                    "cohen-d": np.nan,
                    "analysis_level": "subject",
                    "state_comparison_type": "state",
                    "state_pairing": "not_applicable",
                    "group_pairing": data_pairing,
                    "Measure": (
                        f"{status.replace('_modulated', '_modulation')}"
                        if status.endswith("_modulated")
                        else status
                    ),
                    "failure_reason": reason,
                }

                # Log warning about failed statistical test due to exception
                logger.warning(
                    f"Statistical test could not be computed for {status} {pair}: {reason}"
                )
                pairwise_results.append(failure_dict)

        _log_missing_state_contrasts(
            status, expected_pairs, processed_pairs, skipped_pairs
        )

    return [], pairwise_results


def _calculate_weighted_modulation_proportions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate weighted modulation proportions to address pseudoreplication."""
    try:
        subject_id_col = None
        for col_name in ["normalized_subject_id", "subject_id"]:
            if col_name in df.columns:
                subject_id_col = col_name
                break

        if subject_id_col is None:
            logger.warning(
                "Missing required subject identifier column - "
                "returning original data"
            )
            return df

        required_cols = [
            subject_id_col,
            "state",
            "status",
            "num_cells",
            "total_cell_count",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(
                f"Missing required columns for weighted calculation: "
                f"{missing_cols} - returning original data"
            )
            return df

        df_weighted = df.copy()
        df_weighted["proportion"] = (
            df_weighted["num_cells"] / df_weighted["total_cell_count"]
        )
        df_weighted["proportion"] = df_weighted["proportion"].clip(0, 1)
        df_weighted["proportion"] = df_weighted["proportion"].fillna(0)
        df_weighted["proportion"] = df_weighted["proportion"].replace(
            [np.inf, -np.inf], 0
        )

        logger.info(
            f"Calculated weighted proportions for {len(df_weighted)} "
            f"modulation observations using {subject_id_col}"
        )
        return df_weighted

    except Exception as e:
        logger.error(
            f"Error calculating weighted proportions: {str(e)} - "
            f"returning original data"
        )
        return df
