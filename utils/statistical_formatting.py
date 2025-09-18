"""Statistical formatting utilities for population data analysis.

This module contains functions for formatting and extracting significance information
from statistical analysis results in the IDEAS toolbox.

Key functions:
- _standardize_anova_output: Standardize ANOVA results with proper df1/df2 extraction
- _fill_missing_state_information: Fill missing state info while preserving pingouin data
- _add_pairing_columns: Add pairing metadata based on comparison type
- _finalize_statistical_output: Complete output formatting pipeline
"""

import logging
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_stat_method_for_row(row):
    """Determine statistical method based on row data."""
    parametric = row.get("Parametric", True)
    contrast = str(row.get("Contrast", "")).strip()

    # Determine pairing based on the specific contrast type
    if "state * group" in contrast or "state:group" in contrast:
        # For state*group interactions, use group_pairing (comparing groups within states)
        is_paired = row.get("group_pairing") == "paired"
    elif contrast == "group":
        # For pure group comparisons, use group_pairing
        is_paired = row.get("group_pairing") == "paired"
    elif contrast == "state":
        # For pure state comparisons, use state_pairing
        is_paired = row.get("state_pairing") == "paired"
    elif " vs " in contrast:
        # For pairwise state comparisons (e.g., "center vs quad 4"), use group_pairing
        # since we're comparing groups on their response to this state contrast
        is_paired = row.get("group_pairing") == "paired"
    else:
        # Fallback: use OR logic for unrecognized contrasts
        is_paired = (
            row.get("state_pairing") == "paired"
            or row.get("group_pairing") == "paired"
        )

    if parametric:
        return "paired_ttest" if is_paired else "unpaired_ttest"
    else:
        return "wilcoxon_signed_rank" if is_paired else "mann_whitney_u"


def _standardize_anova_output(
    aov_df: pd.DataFrame,
    metadata: Dict[str, str],
    analysis_level: str = "cell",
    state_comparison_type: str = "state",
    measure_name: Optional[str] = None,
) -> pd.DataFrame:
    """Standardize ANOVA output format across different statistical functions."""
    if aov_df is None or aov_df.empty:
        return pd.DataFrame()

    try:
        df = aov_df.copy()

        # Column mapping for degrees of freedom and other ANOVA outputs
        column_mapping = {
            "p-unc": "p_value",
            "F": "F_statistic",
            "MS": "mean_square",
            "SS": "sum_squares",
            "DF": "df1",
            "hedges": "effect_size",
            "cohen-d": "effect_size",
            "eta2": "effect_size",
            "ddof1": "df1",
            "ddof2": "df2",
            "DF1": "df1",
            "DF2": "df2",
        }

        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Extract df2 from error terms if needed
        if "df2" not in df.columns or df["df2"].isna().all():
            df2_values = []
            error_df = None

            # First pass: look for explicit error terms to extract error_df
            for idx, row in df.iterrows():
                source = str(row.get("Source", "")).lower()
                if any(
                    term in source for term in ["error", "residual", "within"]
                ):
                    error_df = row.get("DF", row.get("df1", np.nan))
                    break

            # If no explicit error term found, try to infer df2 from context
            if error_df is None:
                # For repeated measures ANOVA, estimate error df
                # Common pattern: df2 = (n_subjects - 1) * (n_conditions - 1)
                # or use a reasonable default based on df1 values
                df1_values = df.get("df1", df.get("DF", pd.Series([np.nan])))
                if not df1_values.isna().all():
                    # Use the maximum df1 as a reasonable estimate for df2
                    # This is a heuristic but better than leaving it empty
                    max_df1 = df1_values.max()
                    if pd.notna(max_df1) and max_df1 > 0:
                        # For repeated measures, error df is often larger than effect df
                        error_df = max(4, int(max_df1 * 2))

            # Second pass: assign df2 values for each row
            for idx, row in df.iterrows():
                source = str(row.get("Source", "")).lower()
                if any(
                    term in source for term in ["error", "residual", "within"]
                ):
                    # This is an error term, use its DF directly
                    df2_val = row.get("DF", row.get("df1", np.nan))
                    df2_values.append(df2_val)
                else:
                    # This is an effect term, use the estimated error_df
                    if error_df is not None:
                        df2_values.append(error_df)
                    else:
                        df2_values.append(np.nan)

            # Only assign if we actually have values
            if len(df2_values) == len(df):
                df["df2"] = df2_values
            else:
                # Fallback: create df2 column with estimated values
                if error_df is not None:
                    df["df2"] = error_df
                else:
                    df["df2"] = np.nan

        # Ensure df1 is properly set
        if "df1" in df.columns:
            missing_df1 = df["df1"].isna()
            if missing_df1.any():
                df.loc[missing_df1, "df1"] = df.loc[missing_df1, "DF"].fillna(
                    1
                )

        # Add metadata
        df["Comparison"] = metadata.get("Comparison", "Unknown Comparison")

        # Set Measure column - prioritize measure_name parameter, then metadata
        measure_value = measure_name or metadata.get(
            "Measure", "Unknown Measure"
        )

        # Only set Measure if it doesn't already exist or is empty/NA
        # This preserves Status-based Measure values for correlation data
        if (
            "Measure" not in df.columns
            or df["Measure"].isna().all()
            or (df["Measure"] == "").all()
        ):
            df["Measure"] = measure_value
        elif (
            measure_value
            and (
                df["Measure"].isna()
                | (df["Measure"] == "")
                | (df["Measure"] == "NA")
            ).any()
        ):
            # Fill only empty/NA values while preserving existing non-empty values
            missing_mask = (
                df["Measure"].isna()
                | (df["Measure"] == "")
                | (df["Measure"] == "NA")
            )
            df.loc[missing_mask, "Measure"] = measure_value

        df["analysis_level"] = analysis_level
        df["state_comparison_type"] = state_comparison_type

        # Only fill non-numeric columns with "NA", keep NaN for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in df.columns:
            if col not in numeric_cols:
                df[col] = df[col].fillna("NA")

        return df

    except Exception as e:
        logger.error(f"Error in _standardize_anova_output: {str(e)}")
        return pd.DataFrame()


def _get_mixed_pairwise_sig(
    pairwise: pd.DataFrame,
    status: Optional[str] = None,
    state_legend_show: bool = False,
) -> str:
    """Extract and format significance information from pairwise comparison data."""
    if pairwise is None or pairwise.empty:
        return "No pairwise data available"

    try:
        # Find status column
        status_col = "Measure"

        # Filter by status if provided
        if status and status_col:
            filtered_pairwise = pairwise[pairwise[status_col] == status]
        else:
            filtered_pairwise = (
                pairwise[pairwise[status_col].isnull()]
                if status_col
                else pairwise
            )

        if filtered_pairwise.empty:
            return "No significant differences"

        # Find p-value column
        p_col = (
            "p-corr"
            if "p-corr" in filtered_pairwise.columns
            else "p-unc"
            if "p-unc" in filtered_pairwise.columns
            else None
        )
        if not p_col:
            return "No p-value column found"

        # Generate result text
        result_text = ""
        for _, row in filtered_pairwise.iterrows():
            contrast = row.get("Contrast", "")
            if contrast in ["state * group", "state * group_name"]:
                state = row.get("state", "")
                if state:
                    result_text += f"{contrast} ({state}): p={row.get(p_col, np.nan):.3f}\n"
            elif (
                contrast == "state"
                and state_legend_show
                and "A" in row
                and "B" in row
            ):
                result_text += (
                    f"{row.get('A', '')} × {row.get('B', '')}: "
                    f"p={row.get(p_col, np.nan):.3f}\n"
                )

        return (
            result_text
            if result_text
            else "No significant differences between states"
        )

    except Exception as e:
        logger.warning(f"Error in _get_mixed_pairwise_sig: {str(e)}")
        return "Error retrieving significance info"


def _compute_missing_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing effect sizes from available statistics."""
    if df is None or df.empty:
        return df

    df = df.copy()
    if "effect_size" not in df.columns:
        df["effect_size"] = "NA"

    # Find rows needing effect size computation
    needs_filling = (df["effect_size"].isna()) | (df["effect_size"] == "NA")

    # Try existing effect size columns first
    for effect_col in ["hedges", "cohen-d", "cohens_d", "eta2", "np2"]:
        if effect_col in df.columns and needs_filling.any():
            valid_mask = (
                df[effect_col].notna()
                & (df[effect_col] != "NA")
                & needs_filling
            )
            if valid_mask.any():
                df.loc[valid_mask, "effect_size"] = df.loc[
                    valid_mask, effect_col
                ]
                needs_filling = (df["effect_size"].isna()) | (
                    df["effect_size"] == "NA"
                )

    # Compute from T-statistic if still needed
    if needs_filling.any() and "T" in df.columns:
        try:
            t_vals = pd.to_numeric(df["T"], errors="coerce")
            is_paired = (
                "state_pairing" in df.columns
                and (df["state_pairing"] == "paired").any()
            ) or ("Paired" in df.columns and df["Paired"].any())
            cohens_d = (
                t_vals / np.sqrt(4) if is_paired else t_vals * np.sqrt(0.5)
            )
            mask_computable = (
                needs_filling
                & t_vals.notna()
                & (t_vals != 0)
                & np.isfinite(t_vals)
            )
            if mask_computable.any():
                df.loc[mask_computable, "effect_size"] = cohens_d.where(
                    mask_computable & pd.notna(cohens_d), "NA"
                )
        except Exception as e:
            logger.debug(
                f"Could not compute effect size from t-statistic: {e}"
            )

    # Only fill non-numeric columns with "NA", keep NaN for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in df.columns:
        if col not in numeric_cols:
            df[col] = df[col].fillna("NA")

    return df


def _add_pairing_columns(
    results_df: pd.DataFrame,
    state_comparison_type: str,
    data_pairing: str = "unpaired",
) -> pd.DataFrame:
    """Add pairing columns and update stat_method based on pairing information."""
    if results_df is None or results_df.empty:
        return results_df

    df = results_df.copy()

    # Default values
    df["state_pairing"] = "not_applicable"
    df["group_pairing"] = "not_applicable"

    # Determine if we have group comparisons
    has_group_comparisons = False
    if "A" in df.columns and "B" in df.columns:
        group_mask = df["A"].astype(str).str.contains(
            "group", case=False, na=False
        ) | df["B"].astype(str).str.contains("group", case=False, na=False)
        has_group_comparisons = group_mask.any()

    if "Contrast" in df.columns:
        group_contrast_mask = df["Contrast"].astype(str).str.contains(
            "group", case=False, na=False
        ) | df["Contrast"].astype(str).str.contains(
            "state \\* group", case=False, na=False
        )
        if group_contrast_mask.any():
            has_group_comparisons = True

    # Set pairing based on comparison types and actual content
    if state_comparison_type == "pairwise":
        # For pairwise state comparisons, we're comparing groups on state contrasts
        df["state_pairing"] = "not_applicable"
        df["group_pairing"] = data_pairing
    elif state_comparison_type in [
        "state",
        "state_vs_not_state",
        "state_vs_baseline",
    ]:
        df["state_pairing"] = "paired"
        if has_group_comparisons:
            df["group_pairing"] = data_pairing
    elif state_comparison_type == "group":
        df["group_pairing"] = data_pairing
        # Even for group state_comparison_type, check if we have state comparisons in the data
        if (
            "Contrast" in df.columns
            and "A" in df.columns
            and "B" in df.columns
        ):
            # Check for state comparisons based on contrast and A/B columns
            state_comparison_mask = (
                (df["Contrast"] == "state")
                # Check if A and B contain state names (not group names)
                | (
                    ~df["A"]
                    .astype(str)
                    .str.contains("group", case=False, na=False)
                    & ~df["B"]
                    .astype(str)
                    .str.contains("group", case=False, na=False)
                    & (df["Contrast"] != "group")
                    & (
                        ~df["Contrast"]
                        .astype(str)
                        .str.contains("group", case=False, na=False)
                    )
                )
            )
            df.loc[state_comparison_mask, "state_pairing"] = "paired"

            # Handle "state * group" interactions - these involve state comparisons
            interaction_mask = (
                df["Contrast"]
                .astype(str)
                .str.contains("state \\* group", case=False, na=False)
            )
            df.loc[interaction_mask, "state_pairing"] = "paired"
    elif (
        state_comparison_type in ["state-group", "mixed"]
        or has_group_comparisons
    ):
        df["state_pairing"] = "paired"
        df["group_pairing"] = data_pairing
    else:
        if has_group_comparisons:
            df["group_pairing"] = data_pairing
        # Check for state comparisons in any other case
        if (
            "Contrast" in df.columns
            and "A" in df.columns
            and "B" in df.columns
        ):
            state_comparison_mask = (df["Contrast"] == "state") | (
                ~df["A"]
                .astype(str)
                .str.contains("group", case=False, na=False)
                & ~df["B"]
                .astype(str)
                .str.contains("group", case=False, na=False)
                & (df["Contrast"] != "group")
            )
            df.loc[state_comparison_mask, "state_pairing"] = "paired"

            # Handle "state * group" interactions
            interaction_mask = (
                df["Contrast"]
                .astype(str)
                .str.contains("state \\* group", case=False, na=False)
            )
            df.loc[interaction_mask, "state_pairing"] = "paired"

    # Ensure consistency for group comparisons - all group comparisons
    # should have the same state_pairing
    if "Contrast" in df.columns:
        group_comparison_mask = df["Contrast"] == "group"
        if group_comparison_mask.any():
            # For pure group comparisons (not state*group interactions),
            # state_pairing should be "not_applicable"
            df.loc[group_comparison_mask, "state_pairing"] = "not_applicable"

    # Remove old Paired column and update stat_method
    if "Paired" in df.columns:
        df = df.drop(columns=["Paired"])

    df["stat_method"] = df.apply(_get_stat_method_for_row, axis=1)
    df = _compute_missing_effect_sizes(df)
    df = _infer_stat_method(df)

    # Only fill non-numeric columns with "NA", keep NaN for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in df.columns:
        if col not in numeric_cols:
            df[col] = df[col].fillna("NA")

    return df


def _finalize_statistical_output(
    df: pd.DataFrame, output_type: str = "mixed"
) -> pd.DataFrame:
    """Finalize statistical output with consistent formatting."""
    if df is None or df.empty:
        return df

    df = df.copy()

    # Filter out rows with no meaningful statistical data before processing
    if output_type in ["anova", "mixed"]:
        # For ANOVA, require at least one of: F_statistic, p_value, or coefficient (for LMM)
        has_stats = (
            (
                df.get("F_statistic", pd.Series([np.nan])).notna()
                & (df.get("F_statistic", pd.Series([np.nan])) != "")
            )
            | (
                df.get("F", pd.Series([np.nan])).notna()
                & (df.get("F", pd.Series([np.nan])) != "")
            )
            | (
                df.get("p_value", pd.Series([np.nan])).notna()
                & (df.get("p_value", pd.Series([np.nan])) != "")
            )
            | (
                df.get("p-unc", pd.Series([np.nan])).notna()
                & (df.get("p-unc", pd.Series([np.nan])) != "")
            )
            | (
                df.get("coefficient", pd.Series([np.nan])).notna()
                & (df.get("coefficient", pd.Series([np.nan])) != "")
            )
        )
        if has_stats.any():
            df = df[has_stats].copy()
        else:
            # No meaningful statistical data - return empty DataFrame
            return pd.DataFrame()

    elif output_type == "pairwise":
        # For pairwise, require at least one of: T, p-unc, or coefficient
        has_stats = (
            (
                df.get("T", pd.Series([np.nan])).notna()
                & (df.get("T", pd.Series([np.nan])) != "")
            )
            | (
                df.get("p-unc", pd.Series([np.nan])).notna()
                & (df.get("p-unc", pd.Series([np.nan])) != "")
            )
            | (
                df.get("coefficient", pd.Series([np.nan])).notna()
                & (df.get("coefficient", pd.Series([np.nan])) != "")
            )
        )
        if has_stats.any():
            df = df[has_stats].copy()
        else:
            # No meaningful statistical data - return empty DataFrame
            return pd.DataFrame()

    # If DataFrame becomes empty after filtering, return empty
    if df.empty:
        return df

    # Ensure required columns exist
    required_cols = _get_required_columns(output_type)
    for col in required_cols:
        if col not in df.columns:
            df[col] = "NA"

    # Unified processing pipeline - replaces multiple separate functions
    df = _standardize_missing_values(df)
    df = _fill_essential_metadata(df, output_type)
    df = _infer_stat_method(df)

    # Fill missing state information for pairwise results
    if output_type in ["pairwise", "mixed"]:
        df = _fill_missing_state_information(df)

    return _cleanup_final_csv_columns(df, output_type)


def _get_required_columns(output_type: str) -> list:
    """Get required columns based on output type."""
    base_cols = ["analysis_level"]

    if output_type in ["anova", "mixed"]:
        base_cols.extend(
            [
                "Source",
                "sum_squares",
                "df1",
                "df2",
                "mean_square",
                "F_statistic",
                "p_value",
                "effect_size",
            ]
        )

    if output_type in ["pairwise", "mixed"]:
        base_cols.extend(
            [
                "Contrast",
                "A",
                "B",
                "T",
                "p-unc",
                "p-corr",
                "effect_size",
                "state_pairing",
                "group_pairing",
                "stat_method",
            ]
        )

    return base_cols


# Removed redundant functions - functionality consolidated into:
# - _standardize_missing_values()
# - _fill_essential_metadata()
# - _infer_stat_method()


def _infer_stat_method(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified statistical method inference."""
    if df is None or df.empty:
        return df

    df = df.copy()
    if "stat_method" not in df.columns:
        df["stat_method"] = "unknown"

    # Only process rows with missing stat_method
    missing_mask = (
        (df["stat_method"] == "NA")
        | df["stat_method"].isna()
        | (df["stat_method"] == "")
    )

    if not missing_mask.any():
        return df

    # Simple pattern-based inference with priority order
    inference_rules = [
        # (condition_columns, value_pattern, method_name)
        (
            ["stat_test_name"],
            lambda row: row.get("stat_test_name", "").strip(),
            None,
        ),
        (
            ["coefficient", "t_value"],
            lambda row: (
                "lmm_reml"
                if pd.notna(row.get("coefficient"))
                and pd.notna(row.get("t_value"))
                else None
            ),
            None,
        ),
        (
            ["F_statistic"],
            lambda row: "anova" if pd.notna(row.get("F_statistic")) else None,
            None,
        ),
        (["F"], lambda row: "anova" if pd.notna(row.get("F")) else None, None),
        (
            ["T", "state_pairing"],
            lambda row: (
                "paired_ttest"
                if pd.notna(row.get("T"))
                and row.get("state_pairing") == "paired"
                else None
            ),
            None,
        ),
        (
            ["T"],
            lambda row: "unpaired_ttest" if pd.notna(row.get("T")) else None,
            None,
        ),
    ]

    for cols, func, _ in inference_rules:
        if all(col in df.columns for col in cols):
            for idx in df[missing_mask].index:
                row = df.loc[idx]
                result = func(row)
                if result and result != "NA" and result.strip():
                    df.loc[idx, "stat_method"] = result
                    # Update missing mask
                    missing_mask = (
                        (df["stat_method"] == "NA")
                        | df["stat_method"].isna()
                        | (df["stat_method"] == "")
                    )
                    if not missing_mask.any():
                        break
        if not missing_mask.any():
            break

    # Final fallback
    df.loc[missing_mask, "stat_method"] = "unknown"
    return df


def _cleanup_final_csv_columns(
    df: pd.DataFrame, output_type: str
) -> pd.DataFrame:
    """Remove non-critical columns for cleaner final CSV output."""
    if df is None or df.empty:
        return df

    df = df.copy()

    # Define essential columns for each output type
    if output_type == "anova":
        essential_cols = [
            "Source",
            "p_value",
            "F_statistic",
            "df1",
            "df2",
            "mean_square",
            "sum_squares",
            "effect_size",
            "stat_method",
            "Measure",
            "Comparison",
            "analysis_level",
            "statistical_warnings",
            "analysis_notes",
            # LMM-specific columns that should be preserved
            "coefficient",
            "std_error",
            "t_value",
            "conf_int_lower",
            "conf_int_upper",
            # Analysis parameter columns
            "significance_threshold",
            "multiple_correction",
            "effect_size_method",
            "data_pairing",
        ]
    elif output_type == "pairwise":
        essential_cols = [
            "A",
            "B",
            "Contrast",
            "state",
            "T",
            "p-unc",
            "p-corr",
            "effect_size",
            "stat_method",
            "Measure",
            "Comparison",
            "analysis_level",
            "state_pairing",
            "group_pairing",
            "failure_reason",
            "statistical_warnings",
            "analysis_notes",
            # Analysis parameter columns
            "significance_threshold",
            "multiple_correction",
            "effect_size_method",
            "data_pairing",
        ]
    elif output_type == "mixed":
        # For mixed data (like population CSVs), include all columns except unwanted ones
        # Be defensive about raw data that might have inconsistent structures
        excluded_cols = [
            "Status",
            "state_comparison_type",
            "normalized_subject_id",
        ]
        essential_cols = [
            col for col in df.columns if col not in excluded_cols
        ]

        # For mixed data, be more permissive about empty values since it might be raw data
        mixed_preserve = [
            "analysis_level",
            "Measure",
            "Comparison",
            "stat_method",
            "effect_size",
            "state",
            "file",
            "name",
            "activity",
            "num_cells",
            "status",
        ]
    else:
        # Default case - exclude specific unwanted columns
        essential_cols = [
            col
            for col in df.columns
            if col not in ["Status", "state_comparison_type"]
        ]

    # Always preserve these columns even if they have empty values - base set for all output types
    always_preserve = [
        "analysis_level",
        "Measure",
        "Comparison",
        "stat_method",
        "effect_size",
        "state",
        "statistical_warnings",
        "analysis_notes",
        # Analysis parameter columns should always be preserved
        "significance_threshold",
        "multiple_correction",
        "effect_size_method",
        "data_pairing",
    ]

    # For mixed output type, extend the preserve list
    if output_type == "mixed" and "mixed_preserve" in locals():
        always_preserve.extend(
            [col for col in mixed_preserve if col not in always_preserve]
        )

    # Select meaningful columns
    meaningful_cols = []
    for col in essential_cols:
        if col in df.columns:
            if col in always_preserve:
                meaningful_cols.append(col)
            else:
                meaningful_values = (
                    ~df[col].isna() & (df[col] != "NA") & (df[col] != "")
                ).sum()
                if meaningful_values > 0:
                    meaningful_cols.append(col)

    # Return cleaned DataFrame with fallback
    if meaningful_cols:
        final_df = df[meaningful_cols].copy()
    else:
        if output_type == "anova":
            fallback_cols = [
                "Source",
                "p_value",
                "stat_method",
                "analysis_level",
            ]
        else:
            fallback_cols = [
                "A",
                "B",
                "p-unc",
                "stat_method",
                "analysis_level",
            ]

        available_cols = [col for col in fallback_cols if col in df.columns]
        final_df = df[available_cols].copy() if available_cols else df.copy()

    # Always remove internal-only columns if they somehow made it through
    internal_only_cols = ["state_comparison_type", "normalized_subject_id"]
    cols_to_remove = [
        col for col in internal_only_cols if col in final_df.columns
    ]
    if cols_to_remove:
        final_df = final_df.drop(columns=cols_to_remove)

    return final_df


def _fill_missing_state_information(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing state information in pairwise results."""
    if df is None or df.empty or "state" not in df.columns:
        return df

    df = df.copy()

    # Find rows with missing state information (preserve pingouin's "-" for non-applicable cases)
    missing_state_mask = (
        df["state"].isna()
        | (df["state"] == "NA")
        | (df["state"] == "")
        | (df["state"].astype(str).str.strip() == "")
    )

    # Convert pingouin's "-" values to "across_states" for group comparisons
    if "Contrast" in df.columns:
        pingouin_dash_mask = df["state"] == "-"
        group_comparison_mask = (
            pingouin_dash_mask
            & (
                df["Contrast"]
                .astype(str)
                .str.contains("group", case=False, na=False)
            )
            & (
                ~df["Contrast"]
                .astype(str)
                .str.contains("\\*", case=False, na=False)
            )
        )
        if group_comparison_mask.any():
            df.loc[group_comparison_mask, "state"] = "across_states"
            logger.debug(
                f"Converted {group_comparison_mask.sum()} pingouin '-' "
                f"group comparisons to 'across_states'"
            )

    if not missing_state_mask.any():
        return df

    # Fill missing state information using context clues
    if "Contrast" in df.columns and "A" in df.columns and "B" in df.columns:
        # Group comparisons
        group_comparison_mask = (
            missing_state_mask
            & (
                df["Contrast"]
                .astype(str)
                .str.contains("group", case=False, na=False)
            )
            & (
                ~df["Contrast"]
                .astype(str)
                .str.contains("\\*", case=False, na=False)
            )
        )
        if group_comparison_mask.any():
            df.loc[group_comparison_mask, "state"] = "across_states"

        # State comparisons - these are comparisons across states, not within a specific state
        state_comparison_mask = (
            missing_state_mask
            & (
                df["Contrast"]
                .astype(str)
                .str.contains("state", case=False, na=False)
            )
            & (
                ~df["Contrast"]
                .astype(str)
                .str.contains("group", case=False, na=False)
            )
        )
        if state_comparison_mask.any():
            # For pure state comparisons, use empty string for clean CSV output
            df.loc[state_comparison_mask, "state"] = ""

    # Final fallback for any remaining missing states
    # Exclude empty strings that were intentionally set for state comparisons
    still_missing = df["state"].isna() | (df["state"] == "NA")
    if still_missing.any():
        df.loc[still_missing, "state"] = "unspecified"

    return df


def _standardize_missing_values(
    df: pd.DataFrame, numeric_cols: List[str] = None
) -> pd.DataFrame:
    """Unified function to handle missing values across all column types."""
    if df is None or df.empty:
        return df

    df = df.copy()

    # Default numeric columns
    if numeric_cols is None:
        numeric_cols = [
            "sum_squares",
            "df1",
            "df2",
            "mean_square",
            "F_statistic",
            "p_value",
            "effect_size",
            "T",
            "p-unc",
            "p-corr",
            "coefficient",
            "std_error",
            "t_value",
            "conf_int_lower",
            "conf_int_upper",
        ]

    # Convert all columns to object type first for consistent handling
    for col in df.columns:
        if df[col].dtype in ["int64", "float64", "bool"]:
            df[col] = df[col].astype(object)

    # Process each column
    for col in df.columns:
        # Create mask for empty/missing values
        mask_empty = (
            df[col].isna()
            | (df[col] == "")
            | (df[col].astype(str).str.strip() == "")
        )

        if col in numeric_cols:
            # For numeric columns: convert "NA" strings and invalid values to NaN
            # This ensures empty cells in CSV output instead of "NA" strings
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Keep as numeric type to preserve NaN values properly
        else:
            # For non-numeric columns: standardize to "NA"
            df.loc[mask_empty, col] = "NA"

    return df


def _fill_essential_metadata(
    df: pd.DataFrame, output_type: str
) -> pd.DataFrame:
    """Fill essential metadata columns with intelligent defaults."""
    if df is None or df.empty:
        return df

    df = df.copy()

    # Handle analysis_level
    if (
        "analysis_level" not in df.columns
        or (df["analysis_level"] == "NA").any()
    ):
        if "analysis_level" not in df.columns:
            df["analysis_level"] = "NA"

        mask_empty = df["analysis_level"] == "NA"
        if mask_empty.any():
            if output_type == "pairwise":
                df.loc[mask_empty, "analysis_level"] = "subject"
            elif (
                "stat_method" in df.columns
                and df["stat_method"].str.contains("lmm", na=False).any()
            ):
                lmm_mask = mask_empty & df["stat_method"].str.contains(
                    "lmm", na=False
                )
                df.loc[lmm_mask, "analysis_level"] = "subject/cell"
                df.loc[mask_empty & ~lmm_mask, "analysis_level"] = "subject"
            else:
                df.loc[mask_empty, "analysis_level"] = "subject"

    # Ensure other essential columns exist
    essential_defaults = {
        "stat_method": "unknown",
        "Measure": "Unknown Measure",
        "Comparison": "Unknown Comparison",
    }

    for col, default in essential_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            mask_empty = (df[col] == "NA") | df[col].isna() | (df[col] == "")
            df.loc[mask_empty, col] = default

    return df
