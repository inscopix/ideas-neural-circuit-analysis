from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from ideas.exceptions import IdeasError

from analysis.comb_comp_epochs import run_cc_epochs
from data.test_data.create_cc_epoch_dummy_data import create_dummy_epoch_activity_output


def find_output_file(output_dir: Path, filename: str) -> Optional[Path]:
    """Find an output file, checking both flat and subdirectory layouts.

    The combine_compare workflow may output files either directly in output_dir
    or in a subdirectory with the same name as the file stem.

    Args:
        output_dir: The base output directory.
        filename: The filename to search for (e.g., "group1_combined_activity_data.csv").

    Returns:
        Path to the file if found, None otherwise.
    """
    # Check flat layout first
    flat_path = output_dir / filename
    if flat_path.exists():
        return flat_path

    # Check subdirectory layout (filename stem as subdirectory)
    stem = Path(filename).stem
    subdir_path = output_dir / stem / filename
    if subdir_path.exists():
        return subdir_path

    # Search recursively as fallback
    matches = list(output_dir.rglob(f"*{filename}"))
    return matches[0] if matches else None


def assert_output_file_exists(output_dir: Path, filename: str) -> Path:
    """Assert that an output file exists and return its path.

    Raises AssertionError with helpful message if not found.
    """
    path = find_output_file(output_dir, filename)
    assert path is not None, f"Expected output file not found: {filename}"
    return path


@pytest.fixture
def epoch_data_files(tmp_path):
    """Generate dummy data files for two groups."""
    g1_dir = tmp_path / "group1"
    g1_dir.mkdir()
    g2_dir = tmp_path / "group2"
    g2_dir.mkdir()

    epochs = ["Baseline", "Drug", "Washout"]

    # Group 1 files (5 subjects - minimum for robust statistical tests)
    g1_files = {"activity": [], "correlation": [], "modulation": []}
    for i in range(5):
        a, c, m = create_dummy_epoch_activity_output(
            g1_dir, f"subject{i + 1}", epochs, n_cells=10
        )
        g1_files["activity"].append(str(a))
        g1_files["correlation"].append(str(c))
        g1_files["modulation"].append(str(m))

    # Group 2 files (5 subjects - minimum for robust statistical tests)
    g2_files = {"activity": [], "correlation": [], "modulation": []}
    for i in range(5):
        a, c, m = create_dummy_epoch_activity_output(
            g2_dir, f"subject{i + 1}", epochs, n_cells=12
        )
        g2_files["activity"].append(str(a))
        g2_files["correlation"].append(str(c))
        g2_files["modulation"].append(str(m))

    return g1_files, g2_files, epochs


def test_run_cc_epochs_integration(epoch_data_files, tmp_path):
    """Integration test running the full workflow with dummy files."""
    g1_files, g2_files, epochs = epoch_data_files

    # Output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run the tool
    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_correlation_csv_files=g1_files["correlation"],
        group1_modulation_csv_files=g1_files["modulation"],
        group1_name="Control",
        group1_color="blue",
        group2_activity_csv_files=g2_files["activity"],
        group2_correlation_csv_files=g2_files["correlation"],
        group2_modulation_csv_files=g2_files["modulation"],
        group2_name="Treatment",
        group2_color="red",
        epoch_names=", ".join(epochs),
        epoch_colors="gray, red, blue",
        # Options
        measure_source="trace",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
        save_lmm_outputs=False,
    )

    # Verify outputs exist
    # output_metadata.json is consumed/deleted during output registration
    # assert (output_dir / "output_metadata.json").exists()

    # Check for group comparison files (trace)
    trace_aov_path = assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")
    trace_pairwise_path = assert_output_file_exists(
        output_dir, "trace_pairwise_comparisons.csv"
    )

    # Check for combined data files (organized by group)
    assert_output_file_exists(output_dir, "group1_combined_activity_data.csv")
    assert_output_file_exists(output_dir, "group2_combined_activity_data.csv")

    # Verify content of ANOVA results
    anova_df = pd.read_csv(trace_aov_path)
    assert not anova_df.empty
    # Should contain "Epoch" effect
    assert "Epoch" in anova_df["Source"].values or "epoch" in anova_df["Source"].values

    # Verify pairwise has comparisons
    pairwise_df = pd.read_csv(trace_pairwise_path)
    assert not pairwise_df.empty
    same_mask = (
        pairwise_df["A"].astype(str).str.strip()
        == pairwise_df["B"].astype(str).str.strip()
    )
    assert not same_mask.any()


def test_run_cc_epochs_missing_modulation_ok(epoch_data_files, tmp_path):
    """Test that modulation files are optional."""
    g1_files, g2_files, epochs = epoch_data_files
    output_dir = tmp_path / "output_no_mod"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_name="Group1",
        group2_activity_csv_files=g2_files["activity"],
        group2_name="Group2",
        epoch_names=",".join(epochs),
        epoch_colors="gray, red, blue",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Should succeed and produce activity comparisons
    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")


def test_run_cc_epochs_validation_error_mismatch(epoch_data_files):
    """Test that mismatched epoch/color counts raise error."""
    g1_files, _, _ = epoch_data_files

    with pytest.raises(IdeasError, match="number of epoch names and colors"):
        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            epoch_names="A, B, C",
            epoch_colors="red, blue",  # mismatch
        )


def test_run_cc_epochs_invalid_group1_name(epoch_data_files):
    """Test that group1 name with special characters raises error."""
    g1_files, _, epochs = epoch_data_files

    with pytest.raises(IdeasError, match="Group1 name contains special characters"):
        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="Group1/invalid*chars",
            epoch_names=",".join(epochs),
        )


def test_run_cc_epochs_invalid_group2_name(epoch_data_files):
    """Test that group2 name with special characters raises error."""
    g1_files, g2_files, epochs = epoch_data_files

    with pytest.raises(IdeasError, match="Group2 name contains special characters"):
        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="ValidGroup1",
            group2_activity_csv_files=g2_files["activity"],
            group2_name="Group2:invalid<chars>",
            epoch_names=",".join(epochs),
        )


def test_run_cc_epochs_insufficient_group1_files(epoch_data_files):
    """Test that group1 with less than 2 files raises error."""
    g1_files, _, epochs = epoch_data_files

    with pytest.raises(IdeasError, match="Group 1 must have at least 2 activity files"):
        run_cc_epochs(
            group1_activity_csv_files=[g1_files["activity"][0]],  # Only 1 file
            group1_name="Group1",
            epoch_names=",".join(epochs),
        )


def test_run_cc_epochs_insufficient_group2_files(epoch_data_files):
    """Test that group2 with less than 2 files raises error when provided."""
    g1_files, g2_files, epochs = epoch_data_files

    with pytest.raises(IdeasError, match="Group 2 must have at least 2 activity files"):
        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="Group1",
            group2_activity_csv_files=[g2_files["activity"][0]],  # Only 1 file
            group2_name="Group2",
            epoch_names=",".join(epochs),
        )


# -----------------------------------------------------------------------------
# Additional test coverage
# -----------------------------------------------------------------------------


def test_run_cc_epochs_single_group(epoch_data_files, tmp_path, cleanup_plots):
    """Test with only Group 1 (no Group 2)."""
    g1_files, _, epochs = epoch_data_files
    output_dir = tmp_path / "output_single_group"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_correlation_csv_files=g1_files["correlation"],
        group1_name="OnlyGroup",
        group1_color="purple",
        # No group2 files
        group2_activity_csv_files=None,
        epoch_names=",".join(epochs),
        epoch_colors="gray, red, blue",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Should produce single-group comparisons
    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")
    assert_output_file_exists(output_dir, "trace_pairwise_comparisons.csv")

    # Group 1 combined data should exist
    assert_output_file_exists(output_dir, "group1_combined_activity_data.csv")

    # Group 2 should NOT exist
    assert find_output_file(output_dir, "group2_combined_activity_data.csv") is None


def test_run_cc_epochs_measure_source_event(epoch_data_files, tmp_path, cleanup_plots):
    """Test with measure_source='event'."""
    g1_files, g2_files, epochs = epoch_data_files
    output_dir = tmp_path / "output_event"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_name="Control",
        group2_activity_csv_files=g2_files["activity"],
        group2_name="Treatment",
        epoch_names=",".join(epochs),
        measure_source="event",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Should produce event-based comparisons
    assert_output_file_exists(output_dir, "event_aov_comparisons.csv")
    assert_output_file_exists(output_dir, "event_pairwise_comparisons.csv")


def test_run_cc_epochs_measure_source_both(epoch_data_files, tmp_path, cleanup_plots):
    """Test with measure_source='both'."""
    g1_files, g2_files, epochs = epoch_data_files
    output_dir = tmp_path / "output_both"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_name="Control",
        group2_activity_csv_files=g2_files["activity"],
        group2_name="Treatment",
        epoch_names=",".join(epochs),
        measure_source="both",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Should produce both trace and event comparisons
    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")
    assert_output_file_exists(output_dir, "event_aov_comparisons.csv")


def test_run_cc_epochs_correlation_only(epoch_data_files, tmp_path, cleanup_plots):
    """Test with correlation files but no modulation files."""
    g1_files, g2_files, epochs = epoch_data_files
    output_dir = tmp_path / "output_corr_only"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_correlation_csv_files=g1_files["correlation"],
        group1_name="Control",
        group2_activity_csv_files=g2_files["activity"],
        group2_correlation_csv_files=g2_files["correlation"],
        group2_name="Treatment",
        # No modulation files
        group1_modulation_csv_files=None,
        group2_modulation_csv_files=None,
        epoch_names=",".join(epochs),
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Should produce activity and correlation comparisons
    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")

    # Check correlation combined data exists
    assert_output_file_exists(output_dir, "group1_combined_trace_correlation_data.csv")


def test_run_cc_epochs_different_correction_methods(
    epoch_data_files, tmp_path, cleanup_plots
):
    """Test with different multiple correction methods."""
    g1_files, _, epochs = epoch_data_files

    correction_methods = ["bonf", "fdr_bh", "none"]

    for method in correction_methods:
        output_dir = tmp_path / f"output_{method}"
        output_dir.mkdir()

        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="TestGroup",
            epoch_names=",".join(epochs),
            multiple_correction=method,
            output_dir=str(output_dir),
            enable_lmm_analysis=False,
        )

        # Verify output exists
        assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")

        # Verify pairwise results reflect the correction method
        pairwise_path = assert_output_file_exists(
            output_dir, "trace_pairwise_comparisons.csv"
        )
        pairwise_df = pd.read_csv(pairwise_path)
        assert not pairwise_df.empty, f"Empty pairwise results for method: {method}"


def test_run_cc_epochs_different_effect_sizes(
    epoch_data_files, tmp_path, cleanup_plots
):
    """Test with different effect size methods."""
    g1_files, g2_files, epochs = epoch_data_files

    effect_size_methods = ["cohen", "hedges", "eta_squared", "cles"]

    for effect in effect_size_methods:
        output_dir = tmp_path / f"output_effect_{effect}"
        output_dir.mkdir()

        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="Control",
            group2_activity_csv_files=g2_files["activity"],
            group2_name="Treatment",
            epoch_names=",".join(epochs),
            effect_size=effect,
            output_dir=str(output_dir),
            enable_lmm_analysis=False,
        )

        assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")


def test_run_cc_epochs_many_epochs(tmp_path, cleanup_plots):
    """Test with 5 epochs."""
    epochs = ["Baseline", "Early", "Mid", "Late", "Recovery"]

    g1_dir = tmp_path / "group1_many"
    g1_dir.mkdir()

    g1_activity_files = []
    for i in range(5):
        a, _, _ = create_dummy_epoch_activity_output(
            g1_dir, f"subject{i + 1}", epochs, n_cells=8
        )
        g1_activity_files.append(str(a))

    output_dir = tmp_path / "output_many_epochs"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_activity_files,
        group1_name="ManyEpochs",
        epoch_names=",".join(epochs),
        epoch_colors="gray, blue, green, orange, red",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Check pairwise comparisons exist
    pairwise_path = assert_output_file_exists(
        output_dir, "trace_pairwise_comparisons.csv"
    )
    pairwise_df = pd.read_csv(pairwise_path)
    assert not pairwise_df.empty

    # Verify the pairwise results have the expected structure
    assert "A" in pairwise_df.columns
    assert "B" in pairwise_df.columns
    assert "Contrast" in pairwise_df.columns

    # Verify that epoch names appear in the comparison columns
    all_compared = set(pairwise_df["A"].unique()) | set(pairwise_df["B"].unique())
    assert len(all_compared) >= 2, "At least 2 epochs should be compared"

    # Check ANOVA also ran successfully
    aov_path = assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")
    aov_df = pd.read_csv(aov_path)
    assert not aov_df.empty


def test_run_cc_epochs_minimal_epochs(tmp_path, cleanup_plots):
    """Test with minimal 2 epochs."""
    epochs = ["Before", "After"]

    g1_dir = tmp_path / "group1_minimal"
    g1_dir.mkdir()

    g1_activity_files = []
    for i in range(5):
        a, _, _ = create_dummy_epoch_activity_output(
            g1_dir, f"subject{i + 1}", epochs, n_cells=5
        )
        g1_activity_files.append(str(a))

    output_dir = tmp_path / "output_minimal_epochs"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_activity_files,
        group1_name="MinimalEpochs",
        epoch_names=",".join(epochs),
        epoch_colors="gray, red",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Should succeed with 2 epochs
    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")
    pairwise_path = assert_output_file_exists(
        output_dir, "trace_pairwise_comparisons.csv"
    )
    pairwise_df = pd.read_csv(pairwise_path)
    assert not pairwise_df.empty


def test_run_cc_epochs_no_epoch_colors(epoch_data_files, tmp_path, cleanup_plots):
    """Test that epoch_colors is optional."""
    g1_files, _, epochs = epoch_data_files
    output_dir = tmp_path / "output_no_colors"
    output_dir.mkdir()

    # epoch_colors=None should work (uses defaults)
    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_name="NoColors",
        epoch_names=",".join(epochs),
        epoch_colors=None,  # No colors provided
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")


def test_run_cc_epochs_epoch_names_required(epoch_data_files):
    """Test that epoch_names is required and cannot be empty."""
    g1_files, _, _ = epoch_data_files

    # Empty epoch_names should raise error
    with pytest.raises(IdeasError, match="epoch_names is required"):
        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="TestGroup",
            epoch_names="",  # Empty string
        )


def test_run_cc_epochs_epoch_names_whitespace_only(epoch_data_files):
    """Test that whitespace-only epoch_names raises error."""
    g1_files, _, _ = epoch_data_files

    with pytest.raises(IdeasError, match="epoch_names is required"):
        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_name="TestGroup",
            epoch_names="   ,  ,  ",  # Whitespace only
        )


def test_run_cc_epochs_correlation_statistic_options(
    epoch_data_files, tmp_path, cleanup_plots
):
    """Test different correlation_statistic options."""
    g1_files, _, epochs = epoch_data_files

    stats = ["max", "min", "mean"]

    for stat in stats:
        output_dir = tmp_path / f"output_corr_stat_{stat}"
        output_dir.mkdir()

        run_cc_epochs(
            group1_activity_csv_files=g1_files["activity"],
            group1_correlation_csv_files=g1_files["correlation"],
            group1_name="TestGroup",
            epoch_names=",".join(epochs),
            correlation_statistic=stat,
            output_dir=str(output_dir),
            enable_lmm_analysis=False,
        )

        assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")


def test_run_cc_epochs_large_dataset(tmp_path, cleanup_plots):
    """Test with larger dataset (100 cells per subject)."""
    epochs = ["Epoch1", "Epoch2", "Epoch3"]

    g1_dir = tmp_path / "group1_large"
    g1_dir.mkdir()

    g1_activity_files = []
    for i in range(5):  # 5 subjects with 100 cells each
        a, _, _ = create_dummy_epoch_activity_output(
            g1_dir, f"subject{i + 1}", epochs, n_cells=100
        )
        g1_activity_files.append(str(a))

    output_dir = tmp_path / "output_large"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_activity_files,
        group1_name="LargeGroup",
        epoch_names=",".join(epochs),
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # Verify combined data has expected size
    g1_act_file = assert_output_file_exists(
        output_dir, "group1_combined_activity_data.csv"
    )
    combined_df = pd.read_csv(g1_act_file)
    # 5 subjects * 100 cells * 3 epochs = 1500 rows
    assert len(combined_df) == 1500


def test_run_cc_epochs_without_events(tmp_path, cleanup_plots):
    """Test with activity data that has no event columns."""
    epochs = ["Before", "After"]

    g1_dir = tmp_path / "group1_no_events"
    g1_dir.mkdir()

    g1_activity_files = []
    for i in range(5):
        a, _, _ = create_dummy_epoch_activity_output(
            g1_dir, f"subject{i + 1}", epochs, n_cells=10, has_events=False
        )
        g1_activity_files.append(str(a))

    output_dir = tmp_path / "output_no_events"
    output_dir.mkdir()

    # Should work with trace-only data
    run_cc_epochs(
        group1_activity_csv_files=g1_activity_files,
        group1_name="TraceOnly",
        epoch_names=",".join(epochs),
        measure_source="trace",  # Only trace, no events
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    assert_output_file_exists(output_dir, "trace_aov_comparisons.csv")


def test_run_cc_epochs_output_structure(epoch_data_files, tmp_path, cleanup_plots):
    """Verify the complete output structure."""
    g1_files, g2_files, epochs = epoch_data_files
    output_dir = tmp_path / "output_structure"
    output_dir.mkdir()

    run_cc_epochs(
        group1_activity_csv_files=g1_files["activity"],
        group1_correlation_csv_files=g1_files["correlation"],
        group1_modulation_csv_files=g1_files["modulation"],
        group1_name="Control",
        group2_activity_csv_files=g2_files["activity"],
        group2_correlation_csv_files=g2_files["correlation"],
        group2_modulation_csv_files=g2_files["modulation"],
        group2_name="Treatment",
        epoch_names=",".join(epochs),
        measure_source="trace",
        output_dir=str(output_dir),
        enable_lmm_analysis=False,
    )

    # List all files created
    output_files = list(output_dir.rglob("*"))
    output_names = [f.name for f in output_files if f.is_file()]

    # Check key outputs exist
    expected_patterns = [
        "trace_aov_comparisons.csv",
        "trace_pairwise_comparisons.csv",
        "group1_combined_activity_data.csv",
        "group2_combined_activity_data.csv",
    ]

    for pattern in expected_patterns:
        assert any(pattern in name for name in output_names), (
            f"Missing output: {pattern}"
        )
