"""Combine/compare *epoch_activity* outputs across subjects and groups.

This tool is the "epochs" analogue of `combine_compare_state_epoch_data`. After
`analysis/epoch_activity.py` was refactored to emit state-epoch-compatible long-form
CSVs (`activity_per_epoch_data.csv`, `correlations_per_epoch_data.csv`,
`modulation_vs_baseline_data.csv`), the old epoch combine/compare implementation
became redundant.

This module now delegates the heavy lifting to the shared, battle-tested workflow in
`analysis/combine_compare_state_epoch_data.py`, fixing `comparison_dimension="epochs"`.
"""

import re
from typing import List, Optional

from ideas.exceptions import IdeasError  # type: ignore[import-not-found]
from ideas.tools.types import IdeasFile

from analysis.combine_compare_state_epoch_data import combine_compare_state_epoch_data


def _parse_csv_list(value: Optional[str]) -> List[str]:
    """Parse a comma/semicolon-separated string into a list of strings.

    Returns an empty list if value is None or empty.
    """
    if value is None:
        return []
    tokens = re.split(r"[;,]", str(value))
    return [t.strip() for t in tokens if t.strip()]


def run_cc_epochs(
    *,
    # Group 1 (epoch_activity outputs)
    group1_activity_csv_files: List[IdeasFile],
    group1_correlation_csv_files: Optional[List[IdeasFile]] = None,
    group1_modulation_csv_files: Optional[List[IdeasFile]] = None,
    group1_name: Optional[str] = None,
    group1_color: Optional[str] = None,
    # Group 2 (epoch_activity outputs)
    group2_activity_csv_files: Optional[List[IdeasFile]] = None,
    group2_correlation_csv_files: Optional[List[IdeasFile]] = None,
    group2_modulation_csv_files: Optional[List[IdeasFile]] = None,
    group2_name: Optional[str] = None,
    group2_color: Optional[str] = None,
    # Epoch configuration (required)
    epoch_names: str,
    epoch_colors: Optional[str] = None,
    # Analysis controls
    measure_source: str = "trace",
    modulation_colors: Optional[str] = "red, blue, gray",
    data_pairing: str = "unpaired",
    subject_matching: str = "order",
    correlation_statistic: str = "max",
    multiple_correction: str = "bonf",
    multiple_correction_scope: str = "global",
    effect_size: str = "cohen",
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
    enable_lmm_analysis: bool = True,
    save_lmm_outputs: bool = False,
    output_dir: str = "",
) -> None:
    """Combine and compare epoch-activity outputs across epochs.

    Inputs should be *outputs from the refactored* `analysis/epoch_activity.py`:
    - `activity_per_epoch_data.csv`
    - `correlations_per_epoch_data.csv` (optional; or `pairwise_correlation_heatmaps.h5`)
    - `modulation_vs_baseline_data.csv` (optional)

    The underlying workflow is shared with the state-epoch tool, so outputs follow the
    same structure as `combine_compare_state_epoch_data`, but with comparisons performed
    across epochs.

    Args:
        epoch_names: Required comma-separated epoch names matching epochs in input CSVs.
    """
    # Validate epoch_names is provided and non-empty
    preferred_epoch_order = _parse_csv_list(epoch_names)
    if not preferred_epoch_order:
        raise IdeasError(
            "epoch_names is required and must contain at least one epoch name."
        )

    # Validate epoch_colors matches epoch_names count if both provided
    if epoch_colors is not None:
        colors = _parse_csv_list(epoch_colors)
        if colors and len(colors) != len(preferred_epoch_order):
            raise IdeasError("The number of epoch names and colors must be the same.")

    # Validate minimum subject count for statistical tests (ANOVA requires variance)
    if len(group1_activity_csv_files) < 2:
        raise IdeasError(
            "Group 1 must have at least 2 activity files for statistical analysis. "
            "ANOVA and other statistical tests require multiple subjects to estimate variance."
        )
    if group2_activity_csv_files is not None and len(group2_activity_csv_files) < 2:
        raise IdeasError(
            "Group 2 must have at least 2 activity files when provided. "
            "ANOVA and other statistical tests require multiple subjects to estimate variance."
        )

    # Validate that group names do not contain special characters
    if group1_name and re.search(r'[\/:*?"<>|]', group1_name):
        raise IdeasError(
            "Group1 name contains special characters which are not allowed."
        )
    if group2_name and re.search(r'[\/:*?"<>|]', group2_name):
        raise IdeasError(
            "Group2 name contains special characters which are not allowed."
        )

    combine_compare_state_epoch_data(
        group1_activity_csv_files=group1_activity_csv_files,
        group1_correlation_csv_files=group1_correlation_csv_files,
        group1_modulation_csv_files=group1_modulation_csv_files,
        group1_name=group1_name,
        group1_color=group1_color,
        group2_activity_csv_files=group2_activity_csv_files,
        group2_correlation_csv_files=group2_correlation_csv_files,
        group2_modulation_csv_files=group2_modulation_csv_files,
        group2_name=group2_name,
        group2_color=group2_color,
        comparison_dimension="epochs",
        measure_source=measure_source,
        epoch_colors=epoch_colors,
        modulation_colors=modulation_colors,
        data_pairing=data_pairing,
        subject_matching=subject_matching,
        correlation_statistic=correlation_statistic,
        multiple_correction=multiple_correction,
        multiple_correction_scope=multiple_correction_scope,
        effect_size=effect_size,
        group_comparison_type=group_comparison_type,
        parametric=parametric,
        output_dir=output_dir,
        enable_lmm_analysis=enable_lmm_analysis,
        save_lmm_outputs=save_lmm_outputs,
        preferred_epoch_order=preferred_epoch_order,
    )


def run_cc_epochs_ideas_wrapper(
    *,
    group1_activity_csv_files: List[IdeasFile],
    group1_correlation_csv_files: Optional[List[IdeasFile]] = None,
    group1_modulation_csv_files: Optional[List[IdeasFile]] = None,
    group1_name: Optional[str] = None,
    group1_color: Optional[str] = None,
    group2_activity_csv_files: Optional[List[IdeasFile]] = None,
    group2_correlation_csv_files: Optional[List[IdeasFile]] = None,
    group2_modulation_csv_files: Optional[List[IdeasFile]] = None,
    group2_name: Optional[str] = None,
    group2_color: Optional[str] = None,
    epoch_names: str,
    epoch_colors: Optional[str] = None,
    measure_source: str = "trace",
    modulation_colors: Optional[str] = "red, blue, gray",
    data_pairing: str = "unpaired",
    subject_matching: str = "order",
    correlation_statistic: str = "max",
    multiple_correction: str = "bonf",
    multiple_correction_scope: str = "global",
    effect_size: str = "cohen",
    group_comparison_type: str = "two_tailed",
    parametric: str = "auto",
    enable_lmm_analysis: bool = True,
    save_lmm_outputs: bool = False,
    output_dir: str = "",
) -> None:
    """IDEAS wrapper for run_cc_epochs. Output registration handled internally."""
    run_cc_epochs(
        group1_activity_csv_files=group1_activity_csv_files,
        group1_correlation_csv_files=group1_correlation_csv_files,
        group1_modulation_csv_files=group1_modulation_csv_files,
        group1_name=group1_name,
        group1_color=group1_color,
        group2_activity_csv_files=group2_activity_csv_files,
        group2_correlation_csv_files=group2_correlation_csv_files,
        group2_modulation_csv_files=group2_modulation_csv_files,
        group2_name=group2_name,
        group2_color=group2_color,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        measure_source=measure_source,
        modulation_colors=modulation_colors,
        data_pairing=data_pairing,
        subject_matching=subject_matching,
        correlation_statistic=correlation_statistic,
        multiple_correction=multiple_correction,
        multiple_correction_scope=multiple_correction_scope,
        effect_size=effect_size,
        group_comparison_type=group_comparison_type,
        parametric=parametric,
        enable_lmm_analysis=enable_lmm_analysis,
        save_lmm_outputs=save_lmm_outputs,
        output_dir=output_dir,
    )
