import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from analysis.epoch_activity import (
    _EPOCH_ONLY_STATE_NAME,
    _parse_csv_list,
    _validate_epoch_name_strings,
    run,
)
from utils.state_epoch_data import StateEpochDataManager, scale_data
from utils.state_epoch_results import (
    StateEpochResults,
    analyze_state_epoch_combination,
    calculate_baseline_modulation,
)
from utils.utils import _bin_data

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CELL_SETS = [str(DATA_DIR / "input_cellset.isxd")]
EVENT_SETS = [str(DATA_DIR / "input_cellset-ED.isxd")]

DEFAULT_PARAMS = dict(
    cell_set_files=CELL_SETS,
    event_set_files=EVENT_SETS,
    define_epochs_by="global file time",
    epoch_names="epoch1, epoch2, epoch3",
    epochs="(0, 2), (2.1, 5), (5.1, 9.9)",
    epoch_colors="tab:grey, tab:blue, tab:cyan",
    bin_size=1,
    trace_scale_method="fractional_change",
    event_scale_method="standardize_epoch",
    modulation_colormap="red, blue, gray",
)


@pytest.fixture(scope="module")
def epoch_activity_output_dir(tmp_path_factory):
    """Run the refactored epoch activity workflow once and cache outputs."""
    tmp_dir = tmp_path_factory.mktemp("epoch_activity_outputs")
    cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        run(**DEFAULT_PARAMS)
    finally:
        os.chdir(cwd)
    return tmp_dir


@pytest.fixture(scope="module")
def epoch_activity_reference():
    """Build reference StateEpochResults mirroring the refactored workflow."""
    parsed_epoch_names = _validate_epoch_name_strings(DEFAULT_PARAMS["epoch_names"])
    baseline_epoch = parsed_epoch_names[0]

    data_manager = StateEpochDataManager(
        cell_set_files=[str(p) for p in DEFAULT_PARAMS["cell_set_files"]],
        event_set_files=[str(p) for p in DEFAULT_PARAMS["event_set_files"]]
        if DEFAULT_PARAMS["event_set_files"]
        else None,
        annotations_file=None,
        concatenate=True,
        epochs=DEFAULT_PARAMS["epochs"],
        epoch_names=parsed_epoch_names,
        epoch_colors=_parse_csv_list(DEFAULT_PARAMS["epoch_colors"]),
        state_names=[_EPOCH_ONLY_STATE_NAME],
        state_colors=["gray"],
        baseline_state=_EPOCH_ONLY_STATE_NAME,
        baseline_epoch=baseline_epoch,
        define_epochs_by=DEFAULT_PARAMS["define_epochs_by"],
        tolerance=1e-4,
        sort_by_time=True,
        allow_epoch_only_mode=True,
    )

    traces, events, _, cell_info = data_manager.load_data()
    epoch_periods = data_manager.get_epoch_periods()
    original_period = float(cell_info.get("period", 1.0))

    traces = _bin_data(traces, DEFAULT_PARAMS["bin_size"], original_period)
    events = (
        _bin_data(events, DEFAULT_PARAMS["bin_size"], original_period)
        if events is not None
        else None
    )

    if DEFAULT_PARAMS["bin_size"] and DEFAULT_PARAMS["bin_size"] > 0:
        functional_period = float(DEFAULT_PARAMS["bin_size"])
    else:
        functional_period = original_period

    cell_info = dict(cell_info)
    cell_info["period"] = functional_period
    if (
        "boundaries" in cell_info
        and DEFAULT_PARAMS["bin_size"]
        and DEFAULT_PARAMS["bin_size"] > 0
    ):
        bin_size_in_idxs = int(float(DEFAULT_PARAMS["bin_size"]) / original_period)
        if bin_size_in_idxs > 0:
            cell_info["boundaries"] = [
                int(b / bin_size_in_idxs) for b in cell_info.get("boundaries", [])
            ]

    if DEFAULT_PARAMS["trace_scale_method"]:
        traces = scale_data(
            traces,
            method=str(DEFAULT_PARAMS["trace_scale_method"]),
            behavior=None,
            baseline_state=None,
            epochs=epoch_periods,
            period=functional_period,
            baseline_epoch=baseline_epoch,
            epoch_names=parsed_epoch_names,
        )

    if events is not None and DEFAULT_PARAMS["event_scale_method"]:
        events = scale_data(
            events,
            method=str(DEFAULT_PARAMS["event_scale_method"]),
            behavior=None,
            baseline_state=None,
            epochs=epoch_periods,
            period=functional_period,
            baseline_epoch=baseline_epoch,
            epoch_names=parsed_epoch_names,
        )

    num_timepoints = traces.shape[0]
    annotations_df = pd.DataFrame(
        {
            "dummy_state": [_EPOCH_ONLY_STATE_NAME] * num_timepoints,
            "time": np.arange(num_timepoints, dtype=float) * functional_period,
        }
    )

    alpha = 0.05
    n_shuffle = 1000
    results = StateEpochResults()

    for epoch_name, (start_s, end_s) in zip(parsed_epoch_names, epoch_periods):
        start_idx = max(0, int(start_s / functional_period))
        end_idx = min(traces.shape[0], int(end_s / functional_period))

        if end_idx <= start_idx:
            combination_data = None
        else:
            combination_data = {
                "traces": traces[start_idx:end_idx, :],
                "events": (
                    events[start_idx:end_idx, :] if events is not None else None
                ),
                "annotations": annotations_df.iloc[start_idx:end_idx],
                "num_timepoints": end_idx - start_idx,
                "state": _EPOCH_ONLY_STATE_NAME,
                "epoch": epoch_name,
            }

        combination_results = analyze_state_epoch_combination(
            state_epoch_data=combination_data,
            state=_EPOCH_ONLY_STATE_NAME,
            epoch=epoch_name,
            cell_info=cell_info,
            include_correlations=True,
            include_population_activity=True,
            include_event_analysis=events is not None,
            alpha=alpha,
            n_shuffle=n_shuffle,
        )
        results.add_combination_results(
            _EPOCH_ONLY_STATE_NAME, epoch_name, combination_results
        )

    modulation_results = calculate_baseline_modulation(
        results=results,
        baseline_state=_EPOCH_ONLY_STATE_NAME,
        baseline_epoch=baseline_epoch,
        cell_info=cell_info,
        alpha=alpha,
        n_shuffle=n_shuffle,
    )

    return {
        "results": results,
        "modulation_results": modulation_results,
        "cell_info": cell_info,
        "epoch_periods": epoch_periods,
        "functional_period": functional_period,
        "parsed_epoch_names": parsed_epoch_names,
    }


def _sort_activity_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["epoch", "cell_index"]).reset_index(drop=True)


def test_activity_csv_matches_state_epoch_results(
    epoch_activity_output_dir, epoch_activity_reference
):
    """Ensure long-form activity CSV mirrors the underlying combination results."""
    actual_df = _sort_activity_df(
        pd.read_csv(Path(epoch_activity_output_dir) / "activity_per_epoch_data.csv")
    )

    results = epoch_activity_reference["results"]
    cell_names = epoch_activity_reference["cell_info"].get("cell_names", [])

    expected_rows = []
    for state, epoch in sorted(results.get_all_combinations()):
        combination = results.get_combination_results(state, epoch)
        mean_activity = results.get_activity_vector(state, epoch)
        std_activity = combination.get("std_activity")
        median_activity = combination.get("median_activity")
        activity_cv = combination.get("activity_cv")

        for cell_idx, cell_name in enumerate(cell_names):
            expected_rows.append(
                {
                    "state": state,
                    "epoch": epoch,
                    "cell_index": cell_idx,
                    "name": cell_name,
                    "mean_trace_activity": mean_activity[cell_idx],
                    "std_trace_activity": std_activity[cell_idx],
                    "median_trace_activity": median_activity[cell_idx],
                    "trace_activity_cv": activity_cv[cell_idx],
                }
            )

    expected_df = _sort_activity_df(pd.DataFrame(expected_rows))

    assert list(actual_df["name"]) == list(expected_df["name"])
    for column in [
        "mean_trace_activity",
        "std_trace_activity",
        "median_trace_activity",
        "trace_activity_cv",
    ]:
        np.testing.assert_allclose(
            actual_df[column].to_numpy(),
            expected_df[column].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )


def test_output_metadata_contains_parity_fields(
    epoch_activity_output_dir, epoch_activity_reference
):
    """Confirm metadata exposes comparison-friendly summary statistics."""
    metadata_path = Path(epoch_activity_output_dir) / "output_metadata.json"
    metadata = json.loads(metadata_path.read_text())

    activity_meta = metadata["activity_per_epoch_data"]
    corr_meta = metadata["correlations_per_epoch_data"]

    parsed_epoch_names = epoch_activity_reference["parsed_epoch_names"]
    num_cells = len(epoch_activity_reference["cell_info"].get("cell_names", []))

    assert activity_meta["num_cells"] == num_cells
    assert activity_meta["states"] == [_EPOCH_ONLY_STATE_NAME]
    assert activity_meta["epochs"] == parsed_epoch_names
    assert activity_meta["baseline_state"] == _EPOCH_ONLY_STATE_NAME
    assert activity_meta["baseline_epoch"] == parsed_epoch_names[0]
    assert activity_meta["epoch_comparison_method"] == "epoch_vs_baseline"
    assert activity_meta["analysis_type"] == "state_epoch_baseline_analysis"

    # Correlation metadata should mirror global stats as well
    assert corr_meta["num_cells"] == num_cells
    assert corr_meta["num_epochs"] == len(parsed_epoch_names)
    assert corr_meta["analysis_type"] == "correlation_analysis"


def _sort_correlation_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["epoch", "cell_index"]).reset_index(drop=True)


def _pos_neg_triangle_means(corr_matrix: Optional[np.ndarray]) -> tuple[float, float]:
    """Replicate state_epoch_output.py's pos/neg correlation summary logic."""
    if corr_matrix is None or corr_matrix.size == 0:
        return np.nan, np.nan
    tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
    tri = tri[~np.isnan(tri)]
    if tri.size == 0:
        return np.nan, np.nan
    pos = tri[tri > 0]
    neg = tri[tri < 0]
    pos_mean = float(np.nanmean(pos)) if pos.size > 0 else np.nan
    neg_mean = float(np.nanmean(neg)) if neg.size > 0 else np.nan
    return pos_mean, neg_mean


def test_correlations_csv_matches_state_epoch_results(
    epoch_activity_output_dir, epoch_activity_reference
):
    """Ensure correlations CSV matches correlation matrices in the results."""
    actual_df = _sort_correlation_df(
        pd.read_csv(Path(epoch_activity_output_dir) / "correlations_per_epoch_data.csv")
    )

    results: StateEpochResults = epoch_activity_reference["results"]
    cell_names = epoch_activity_reference["cell_info"].get("cell_names", [])

    expected_rows = []
    for state, epoch in sorted(results.get_all_combinations()):
        trace_stats = results.get_correlation_stats(state, epoch)
        corr_matrix = results.get_correlation_matrix(state, epoch)
        pos_trace, neg_trace = _pos_neg_triangle_means(corr_matrix)

        combination = results.get_combination_results(state, epoch) or {}
        event_corr_matrix = combination.get("event_correlation_matrix")
        pos_event, neg_event = _pos_neg_triangle_means(event_corr_matrix)

        # If correlations are unavailable (e.g. 1 cell), skip (not the case in test data).
        if trace_stats is None:
            continue

        for cell_idx, cell_name in enumerate(cell_names):
            # Event stats are computed directly from the matrix in the output code.
            if event_corr_matrix is None:
                max_event = np.nan
                min_event = np.nan
                mean_event = np.nan
            else:
                # Per-cell stats excluding diagonal, matching StateEpochResults.get_correlation_stats.
                n_cells = event_corr_matrix.shape[0]
                mask = np.ones(n_cells, dtype=bool)
                mask[cell_idx] = False
                off_diag = event_corr_matrix[cell_idx, mask]
                valid = off_diag[~np.isnan(off_diag)]
                if valid.size > 0:
                    max_event = float(np.max(valid))
                    min_event = float(np.min(valid))
                    mean_event = float(np.mean(valid))
                else:
                    max_event = np.nan
                    min_event = np.nan
                    mean_event = np.nan

            expected_rows.append(
                {
                    "name": cell_name,
                    "cell_index": cell_idx,
                    "state": state,
                    "epoch": epoch,
                    "max_trace_correlation": trace_stats["max_per_cell"][cell_idx],
                    "min_trace_correlation": trace_stats["min_per_cell"][cell_idx],
                    "mean_trace_correlation": trace_stats["mean_per_cell"][cell_idx],
                    "max_event_correlation": max_event,
                    "min_event_correlation": min_event,
                    "mean_event_correlation": mean_event,
                    "positive_trace_correlation": pos_trace,
                    "negative_trace_correlation": neg_trace,
                    "positive_event_correlation": pos_event,
                    "negative_event_correlation": neg_event,
                }
            )

    expected_df = _sort_correlation_df(pd.DataFrame(expected_rows))

    assert list(actual_df["name"]) == list(expected_df["name"])
    for column in [
        "max_trace_correlation",
        "min_trace_correlation",
        "mean_trace_correlation",
        "positive_trace_correlation",
        "negative_trace_correlation",
        "max_event_correlation",
        "min_event_correlation",
        "mean_event_correlation",
        "positive_event_correlation",
        "negative_event_correlation",
    ]:
        np.testing.assert_allclose(
            actual_df[column].to_numpy(),
            expected_df[column].to_numpy(),
            rtol=1e-6,
            atol=1e-8,
            equal_nan=True,
        )


def test_modulation_vs_baseline_csv_matches_modulation_results(
    epoch_activity_output_dir, epoch_activity_reference
):
    """Ensure modulation CSV encodes the baseline-vs-epoch modulation results."""
    df = pd.read_csv(
        Path(epoch_activity_output_dir) / "modulation_vs_baseline_data.csv"
    )

    modulation_results = epoch_activity_reference["modulation_results"]
    assert "error" not in modulation_results

    baseline_epoch = modulation_results["baseline_epoch"]
    baseline_state = modulation_results["baseline_state"]

    # Baseline should not appear as an epoch row (baseline is skipped in modulation calc).
    assert not ((df["state"] == baseline_state) & (df["epoch"] == baseline_epoch)).any()

    # Expected combinations are all non-baseline combinations with data.
    activity_mod = modulation_results.get("activity_modulation", {})
    event_mod = modulation_results.get("event_modulation", {})
    all_keys = set(activity_mod.keys()) | set(event_mod.keys())
    assert len(all_keys) > 0

    # Basic invariants on schema and value ranges
    assert set(
        ["name", "cell_index", "state", "epoch", "baseline_state", "baseline_epoch"]
    ).issubset(df.columns)
    assert (df["baseline_state"] == baseline_state).all()
    assert (df["baseline_epoch"] == baseline_epoch).all()

    # Validate a few per-key dynamic columns exist and match underlying arrays.
    # (The output CSV repeats values per row in the dynamic column corresponding to that row's state-epoch.)
    for state, epoch in sorted(all_keys):
        sub = df[(df["state"] == state) & (df["epoch"] == epoch)].sort_values(
            "cell_index"
        )
        assert len(sub) > 0

        trace_col = f"trace_modulation_scores in {state}-{epoch}"
        trace_p_col = f"trace_p_values in {state}-{epoch}"
        trace_cat_col = f"trace_modulation in {state}-{epoch}"
        event_col = f"event_modulation_scores in {state}-{epoch}"
        event_p_col = f"event_p_values in {state}-{epoch}"
        event_cat_col = f"event_modulation in {state}-{epoch}"

        for c in [
            trace_col,
            trace_p_col,
            trace_cat_col,
            event_col,
            event_p_col,
            event_cat_col,
        ]:
            assert c in df.columns

        trace_data = activity_mod.get((state, epoch), {})
        trace_scores = np.asarray(trace_data.get("modulation_index", []), dtype=float)
        trace_p = np.asarray(trace_data.get("p_values", []), dtype=float)
        trace_sig = np.asarray(trace_data.get("significant", []), dtype=bool)

        # trace: if present, match values
        if trace_scores.size > 0:
            np.testing.assert_allclose(
                sub[trace_col].to_numpy(),
                trace_scores[: len(sub)],
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                sub[trace_p_col].to_numpy(),
                trace_p[: len(sub)],
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )
            # categorical: derived from significant + sign
            expected_cat = np.zeros(len(sub), dtype=int)
            sig = trace_sig[: len(sub)]
            vals = trace_scores[: len(sub)]
            expected_cat[(sig) & (vals > 0)] = 1
            expected_cat[(sig) & (vals < 0)] = -1
            np.testing.assert_array_equal(sub[trace_cat_col].to_numpy(), expected_cat)

            # score range sanity (modulation clipped to [-1, 1] in calculation)
            assert np.nanmin(sub[trace_col].to_numpy()) >= -1.0 - 1e-9
            assert np.nanmax(sub[trace_col].to_numpy()) <= 1.0 + 1e-9

        event_data = event_mod.get((state, epoch), {})
        event_scores = np.asarray(event_data.get("modulation_index", []), dtype=float)
        event_p = np.asarray(event_data.get("p_values", []), dtype=float)
        event_sig = np.asarray(event_data.get("significant", []), dtype=bool)

        if event_scores.size > 0:
            np.testing.assert_allclose(
                sub[event_col].to_numpy(),
                event_scores[: len(sub)],
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                sub[event_p_col].to_numpy(),
                event_p[: len(sub)],
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            )
            expected_cat = np.zeros(len(sub), dtype=int)
            sig = event_sig[: len(sub)]
            vals = event_scores[: len(sub)]
            expected_cat[(sig) & (vals > 0)] = 1
            expected_cat[(sig) & (vals < 0)] = -1
            np.testing.assert_array_equal(sub[event_cat_col].to_numpy(), expected_cat)
