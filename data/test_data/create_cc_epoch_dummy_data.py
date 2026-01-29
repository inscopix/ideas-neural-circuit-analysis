"""Generate dummy epoch_activity outputs for testing.

This module provides utilities for creating test data that mimics the output
of the epoch_activity analysis tool. It can be used both as a standalone script
to generate persistent test files and as an importable module for test fixtures.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.config import EPOCH_ONLY_DUMMY_STATE


def create_dummy_epoch_activity_output(
    output_dir: Path,
    file_prefix: str,
    epochs: List[str],
    n_cells: int = 10,
    has_events: bool = True,
    seed: int | None = None,
) -> Tuple[Path, Path, Path]:
    """Generate dummy CSV outputs mimicking epoch_activity.py.

    Args:
        output_dir: Directory to write output files.
        file_prefix: Prefix for output filenames (e.g., "subject1").
        epochs: List of epoch names.
        n_cells: Number of cells to generate per epoch.
        has_events: Whether to include event-related columns.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of paths: (activity_file, correlation_file, modulation_file)
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure output dir exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Activity Data
    activity_rows = []
    for epoch in epochs:
        for i in range(n_cells):
            mean_trace = np.random.rand() * 10
            std_trace = np.random.rand()
            median_trace = mean_trace + np.random.randn() * 0.1
            trace_cv = std_trace / mean_trace if mean_trace else 0.0

            mean_event = np.random.rand() * 2 if has_events else np.nan
            std_event = np.random.rand() if has_events else np.nan
            median_event = (
                mean_event + np.random.randn() * 0.05 if has_events else np.nan
            )
            event_cv = std_event / mean_event if has_events and mean_event else np.nan

            row = {
                "name": f"C{i}",
                "cell_index": i,
                "state": EPOCH_ONLY_DUMMY_STATE,
                "epoch": epoch,
                "mean_trace_activity": mean_trace,
                "std_trace_activity": std_trace,
                "median_trace_activity": median_trace,
                "trace_activity_cv": trace_cv,
                "mean_event_rate": mean_event,
                "std_event_rate": std_event,
                "median_event_rate": median_event,
                "event_rate_cv": event_cv,
            }
            activity_rows.append(row)

    activity_df = pd.DataFrame(activity_rows)
    activity_file = output_dir / f"{file_prefix}_activity_per_epoch_data.csv"
    activity_df.to_csv(activity_file, index=False)

    # 2. Correlation Data
    corr_rows = []
    for epoch in epochs:
        # Population correlations with per-epoch variance
        pos_pop_corr = np.random.rand() * 0.5 + 0.3  # 0.3-0.8 range
        neg_pop_corr = -np.random.rand() * 0.5 - 0.1  # -0.6 to -0.1 range
        pos_event_corr = (np.random.rand() * 0.5 + 0.2) if has_events else np.nan
        neg_event_corr = (-np.random.rand() * 0.5 - 0.1) if has_events else np.nan

        for i in range(n_cells):
            # Add per-cell variance to ensure non-zero variance after aggregation
            row = {
                "name": f"C{i}",
                "cell_index": i,
                "state": EPOCH_ONLY_DUMMY_STATE,
                "epoch": epoch,
                "max_trace_correlation": np.random.rand() * 0.8 + 0.1,  # 0.1-0.9
                "min_trace_correlation": -np.random.rand() * 0.8 - 0.1,  # -0.9 to -0.1
                "mean_trace_correlation": np.random.rand() * 0.4 - 0.1,  # -0.1 to 0.3
                "positive_trace_correlation": pos_pop_corr + np.random.randn() * 0.05,
                "negative_trace_correlation": neg_pop_corr + np.random.randn() * 0.05,
                "max_event_correlation": (np.random.rand() * 0.7 + 0.2)
                if has_events
                else np.nan,
                "min_event_correlation": (-np.random.rand() * 0.3)
                if has_events
                else np.nan,
                "mean_event_correlation": (np.random.rand() * 0.3)
                if has_events
                else np.nan,
                "positive_event_correlation": (
                    pos_event_corr + np.random.randn() * 0.05
                )
                if has_events
                else np.nan,
                "negative_event_correlation": (
                    neg_event_corr + np.random.randn() * 0.05
                )
                if has_events
                else np.nan,
            }
            corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    corr_file = output_dir / f"{file_prefix}_correlations_per_epoch_data.csv"
    corr_df.to_csv(corr_file, index=False)

    # 3. Modulation Data
    mod_rows = []
    baseline_epoch = next(
        (epoch for epoch in epochs if "baseline" in epoch.lower()),
        epochs[0],
    )

    for epoch in epochs:
        if epoch == baseline_epoch:
            # Epoch-only baseline rows are not included in modulation outputs.
            continue
        for i in range(n_cells):
            val = np.random.rand()
            if val > 0.66:
                # up-modulated
                score = np.random.rand() * 0.8 + 0.5  # 0.5-1.3
                trace_modulation = 1
            elif val < 0.33:
                # down-modulated
                score = -np.random.rand() * 0.8 - 0.5  # -1.3 to -0.5
                trace_modulation = -1
            else:
                # non-modulated
                score = np.random.randn() * 0.2
                trace_modulation = 0
            p_val = (
                np.random.rand() * 0.1
                if trace_modulation != 0
                else np.random.rand() * 0.5 + 0.5
            )

            combo_label = f"{EPOCH_ONLY_DUMMY_STATE}-{epoch}"
            row = {
                "name": f"C{i}",
                "cell_index": i,
                "state": EPOCH_ONLY_DUMMY_STATE,
                "epoch": epoch,
                "baseline_state": EPOCH_ONLY_DUMMY_STATE,
                "baseline_epoch": baseline_epoch,
                f"trace_modulation_scores in {combo_label}": score,
                f"trace_p_values in {combo_label}": p_val,
                f"trace_modulation in {combo_label}": trace_modulation,
                f"event_modulation_scores in {combo_label}": (
                    score if has_events else np.nan
                ),
                f"event_p_values in {combo_label}": (p_val if has_events else np.nan),
                f"event_modulation in {combo_label}": (
                    trace_modulation if has_events else 0
                ),
            }
            mod_rows.append(row)

    mod_df = pd.DataFrame(mod_rows)
    mod_file = output_dir / f"{file_prefix}_modulation_vs_baseline_data.csv"
    mod_df.to_csv(mod_file, index=False)

    return activity_file, corr_file, mod_file


def main():
    """Generate persistent test data files for manual testing."""
    import shutil

    base_dir = Path("data/test_data/cc_epoch_dummy")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)

    epochs = ["early drug", "BASELINE", "late-Drug"]

    # Group 1 - 5 subjects for robust statistical tests
    g1_dir = base_dir / "group1"
    for i in range(1, 6):
        create_dummy_epoch_activity_output(
            g1_dir, f"subject{i}", epochs, n_cells=10 + i, seed=42 + i
        )
        print(f"Created Group 1 subject{i} files in {g1_dir}")

    # Group 2 - 5 subjects for robust statistical tests
    g2_dir = base_dir / "group2"
    for i in range(1, 6):
        create_dummy_epoch_activity_output(
            g2_dir, f"subject{i}", epochs, n_cells=11 + i, seed=100 + i
        )
        print(f"Created Group 2 subject{i} files in {g2_dir}")

    print(f"\nAll test data created in {base_dir}")


if __name__ == "__main__":
    main()
