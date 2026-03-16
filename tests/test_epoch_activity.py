import json
import os
from pathlib import Path

import isx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from ideas.exceptions import IdeasError

from analysis.epoch_activity import run
from utils.utils import (
    _get_cellset_boundaries,
    _redefine_epochs,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
cell_sets = [str(DATA_DIR / "input_cellset.isxd")]
event_sets = [str(DATA_DIR / "input_cellset-ED.isxd")]

HAS_ISX_EVENTSET = True

valid_inputs = [
    # valid inputs
    dict(
        cell_set_files=cell_sets,
        event_set_files=event_sets,
        define_epochs_by="global file time",
        epoch_names="epoch1, epoch2, epoch3",
        epochs="(0, 2), (2.1, 5), (5.1, 9.9)",
        epoch_colors="tab:grey, tab:blue, tab:cyan",
        bin_size=1,
        trace_scale_method="fractional_change",
        event_scale_method="standardize_epoch",
        modulation_colormap="red, blue, gray",
    ),
]
expected_output_files = [
    # Core long-form outputs (state-epoch-compatible)
    "activity_per_epoch_data.csv",
    "correlations_per_epoch_data.csv",
    "modulation_vs_baseline_data.csv",
    "average_correlations.csv",
    "pairwise_correlation_heatmaps.h5",
    "spatial_analysis_pairwise_correlations.zip",
]
expected_preview_files = [
    "time_in_epoch_preview.svg",
    "trace_population_average_preview.svg",
    "trace_epoch_overlay.svg",
]

if HAS_ISX_EVENTSET:
    expected_preview_files.extend(
        ["event_population_average_preview.svg", "event_epoch_overlay.svg"]
    )


def _load_output_data(output_dir: Path) -> dict:
    output_data_path = Path(output_dir) / "output_data.json"
    return json.loads(output_data_path.read_text())


def _collect_output_paths(output_data: dict) -> list[Path]:
    return [Path(entry["file"]) for entry in output_data.get("output_files", [])]


def _collect_preview_paths(output_data: dict) -> list[Path]:
    previews = []
    for entry in output_data.get("output_files", []):
        for preview in entry.get("previews", []):
            previews.append(Path(preview["file"]))
    return previews


def _find_output_path(output_data: dict, filename: str, output_dir: Path) -> Path:
    for path in _collect_output_paths(output_data):
        if str(path).endswith(filename):
            return output_dir / path
    raise AssertionError(f"Output file ending with {filename} not registered")


def _metadata_for_output(output_data: dict, filename: str) -> dict:
    for entry in output_data.get("output_files", []):
        file_path = Path(entry.get("file", ""))
        if str(file_path).endswith(filename):
            return {m["key"]: m["value"] for m in entry.get("metadata", [])}
    raise AssertionError(f"No metadata found for output {filename}")


@pytest.mark.parametrize("params", valid_inputs)
def test_epoch_activity(params, tmp_path):
    """Check that code runs without error with valid inputs."""
    run(**params, output_dir=tmp_path)
    output_data = _load_output_data(tmp_path)
    actual_output_files = _collect_output_paths(output_data)
    actual_preview_files = _collect_preview_paths(output_data)

    # check that the output files are created
    for output_file in expected_output_files:
        matches = [p for p in actual_output_files if str(p).endswith(output_file)]
        assert matches, f"Missing registered output ending with {output_file}"
        for match in matches:
            assert (Path(tmp_path) / match).exists()

    for preview_file in expected_preview_files:
        matches = [p for p in actual_preview_files if str(p).endswith(preview_file)]
        assert matches, f"Missing preview ending with {preview_file}"
        for match in matches:
            assert (Path(tmp_path) / match).exists()

    # validate number of cells in output files
    # validate number of cells in long-form output files
    for f in ["activity_per_epoch_data.csv", "correlations_per_epoch_data.csv"]:
        df = pd.read_csv(_find_output_path(output_data, f, Path(tmp_path)))
        assert len(df) == 76 * 3  # 76 accepted cells, 3 epochs


def test_epoch_activity_baseline_epoch_can_be_non_first(tmp_path):
    """baseline_epoch should be configurable (not always the first epoch)."""
    params = valid_inputs[0].copy()
    params["baseline_epoch"] = "epoch2"
    params["epoch_comparison_method"] = "epoch_vs_baseline"
    run(**params, output_dir=tmp_path)
    output_data = _load_output_data(tmp_path)
    activity_meta = _metadata_for_output(output_data, "activity_per_epoch_data.csv")
    assert activity_meta["baseline_epoch"] == "epoch2"
    assert activity_meta["epoch_comparison_method"] == "epoch_vs_baseline"


def test_epoch_activity_invalid_epoch_comparison_method_raises(tmp_path):
    params = valid_inputs[0].copy()
    params["epoch_comparison_method"] = "pairwise"
    with pytest.raises(IdeasError):
        run(**params, output_dir=tmp_path)


def test_epoch_activity_invalid_modulation_colormap_raises(tmp_path):
    """modulation_colormap must be exactly three comma-separated colors."""
    params = valid_inputs[0].copy()
    params["modulation_colormap"] = "red, blue"  # missing third
    with pytest.raises(IdeasError):
        run(**params, output_dir=tmp_path)


def test_epoch_activity_multiple_regions_requires_second_cell_set_files(tmp_path):
    params = valid_inputs[0].copy()
    params["region_selection"] = "multiple_regions"
    with pytest.raises(IdeasError):
        run(**params, output_dir=tmp_path)


def test_epoch_activity_multiple_regions_runs_in_region_subdirs(tmp_path):
    params = valid_inputs[0].copy()
    params["region_selection"] = "multiple_regions"
    params["second_cell_set_files"] = cell_sets
    params["second_event_set_files"] = event_sets
    run(**params, output_dir=tmp_path)

    first_region_dir = Path(tmp_path) / "single_brain_region"
    second_region_dir = Path(tmp_path) / "second_brain_region"
    for region_dir in [first_region_dir, second_region_dir]:
        assert (region_dir / "output_data.json").exists()
        output_data = _load_output_data(region_dir)
        assert _find_output_path(output_data, "activity_per_epoch_data.csv", region_dir).exists()
        assert _find_output_path(
            output_data, "correlations_per_epoch_data.csv", region_dir
        ).exists()
        assert _find_output_path(
            output_data, "modulation_vs_baseline_data.csv", region_dir
        ).exists()


def test_epoch_activity_multiple_regions_adds_brain_region_column(tmp_path):
    params = valid_inputs[0].copy()
    params["region_selection"] = "multiple_regions"
    params["second_cell_set_files"] = cell_sets
    params["second_event_set_files"] = event_sets
    params["first_brain_region_name"] = "Region A"
    params["second_brain_region_name"] = "Region B"
    run(**params, output_dir=tmp_path)

    region_expectations = {
        Path(tmp_path) / "single_brain_region": "Region A",
        Path(tmp_path) / "second_brain_region": "Region B",
    }
    csvs_to_check = [
        "activity_per_epoch_data.csv",
        "correlations_per_epoch_data.csv",
        "modulation_vs_baseline_data.csv",
        "average_correlations.csv",
    ]
    for region_dir, expected_name in region_expectations.items():
        output_data = _load_output_data(region_dir)
        for csv_name in csvs_to_check:
            csv_path = _find_output_path(output_data, csv_name, region_dir)
            df = pd.read_csv(csv_path)
            assert "brain_region" in df.columns
            assert set(df["brain_region"].dropna().unique()) == {expected_name}


def test_plot_traces(cleanup_plots):
    """Test the _plot_traces function with various state transition scenarios."""
    from utils.plots import _plot_traces

    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create test data
    num_frames = 100
    num_cells = 10
    traces = np.random.rand(num_frames, num_cells)

    # Test case 1: Normal state transitions
    behavior1 = pd.DataFrame({"state": ["A"] * 30 + ["B"] * 40 + ["A"] * 30})
    _plot_traces(
        traces=traces,
        behavior=behavior1,
        data={},
        column_name="state",
        filename="test_traces_normal.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 2: State starts at frame 0 (edge case fix test)
    behavior2 = pd.DataFrame({"state": ["A"] * 50 + ["B"] * 50})
    _plot_traces(
        traces=traces,
        behavior=behavior2,
        data={},
        column_name="state",
        filename="test_traces_start0.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 3: State ends at last frame (edge case fix test)
    behavior3 = pd.DataFrame({"state": ["B"] * 50 + ["A"] * 50})
    _plot_traces(
        traces=traces,
        behavior=behavior3,
        data={},
        column_name="state",
        filename="test_traces_end_last.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 4: Multiple state transitions
    behavior4 = pd.DataFrame(
        {"state": ["A"] * 20 + ["B"] * 20 + ["A"] * 20 + ["B"] * 20 + ["A"] * 20}
    )
    _plot_traces(
        traces=traces,
        behavior=behavior4,
        data={},
        column_name="state",
        filename="test_traces_multiple.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 5: Three states
    behavior5 = pd.DataFrame({"state": ["A"] * 30 + ["B"] * 30 + ["C"] * 40})
    _plot_traces(
        traces=traces,
        behavior=behavior5,
        data={},
        column_name="state",
        filename="test_traces_three_states.svg",
        spacing=5.01,
        state_colors=["red", "blue", "green"],
        state_names=["A", "B", "C"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 6: Single state throughout (edge case)
    behavior6 = pd.DataFrame({"state": ["A"] * 100})
    _plot_traces(
        traces=traces,
        behavior=behavior6,
        data={},
        column_name="state",
        filename="test_traces_single_state.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 7: State starts and ends with same state (edge case)
    behavior7 = pd.DataFrame({"state": ["A"] * 1 + ["B"] * 98 + ["A"] * 1})
    _plot_traces(
        traces=traces,
        behavior=behavior7,
        data={},
        column_name="state",
        filename="test_traces_boundary_same.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 8: Custom period to test time coordinate calculation
    behavior8 = pd.DataFrame({"state": ["A"] * 25 + ["B"] * 50 + ["A"] * 25})
    _plot_traces(
        traces=traces,
        behavior=behavior8,
        data={},
        column_name="state",
        filename="test_traces_custom_period.svg",
        spacing=5.01,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        period=0.5,  # Test custom period
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Verify that all test files were created
    expected_files = [
        "test_traces_normal.svg",
        "test_traces_start0.svg",
        "test_traces_end_last.svg",
        "test_traces_multiple.svg",
        "test_traces_three_states.svg",
        "test_traces_single_state.svg",
        "test_traces_boundary_same.svg",
        "test_traces_custom_period.svg",
    ]

    for file in expected_files:
        assert os.path.exists(file), f"Expected file {file} was not created"

    # Clean up test files
    for file in expected_files:
        os.remove(file)


def test_plot_raster(cleanup_plots):
    """Test the _plot_raster function with various state transition scenarios."""
    from utils.plots import _plot_raster

    # Set random seed for reproducible tests
    np.random.seed(42)

    # Create test data
    num_frames = 100
    num_cells = 10

    # Create synthetic event data (list of arrays with event times for each cell)
    events = []
    for _cell in range(num_cells):
        # Random events for each cell
        event_times = np.sort(np.random.uniform(0, 100, np.random.randint(5, 20)))
        events.append(event_times)

    # Create event timeseries (binary matrix)
    event_timeseries = np.zeros((num_frames, num_cells))
    for cell_idx, cell_events in enumerate(events):
        for event_time in cell_events:
            frame_idx = int(event_time)
            if frame_idx < num_frames:
                event_timeseries[frame_idx, cell_idx] = 1

    # Test case 1: Normal state transitions
    behavior1 = pd.DataFrame({"state": ["A"] * 30 + ["B"] * 40 + ["A"] * 30})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior1,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_normal.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 2: State starts at frame 0 (edge case fix test)
    behavior2 = pd.DataFrame({"state": ["A"] * 50 + ["B"] * 50})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior2,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_start0.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 3: State ends at last frame (edge case fix test)
    behavior3 = pd.DataFrame({"state": ["B"] * 50 + ["A"] * 50})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior3,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_end_last.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 4: Multiple state transitions
    behavior4 = pd.DataFrame(
        {"state": ["A"] * 20 + ["B"] * 20 + ["A"] * 20 + ["B"] * 20 + ["A"] * 20}
    )
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior4,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_multiple.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 5: Three states
    behavior5 = pd.DataFrame({"state": ["A"] * 30 + ["B"] * 30 + ["C"] * 40})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior5,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue", "green"],
        state_names=["A", "B", "C"],
        filename="test_raster_three_states.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 6: Single state throughout (edge case)
    behavior6 = pd.DataFrame({"state": ["A"] * 100})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior6,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_single_state.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 7: State starts and ends with same state (edge case)
    behavior7 = pd.DataFrame({"state": ["A"] * 1 + ["B"] * 98 + ["A"] * 1})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior7,
        column_name="state",
        period=1.0,
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_boundary_same.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Test case 8: Custom period to test time coordinate calculation
    behavior8 = pd.DataFrame({"state": ["A"] * 25 + ["B"] * 50 + ["A"] * 25})
    _plot_raster(
        events=events,
        event_timeseries=event_timeseries,
        behavior=behavior8,
        column_name="state",
        period=0.5,  # Test custom period
        state_colors=["red", "blue"],
        state_names=["A", "B"],
        filename="test_raster_custom_period.svg",
    )
    plt.close("all")  # Close figures to prevent memory leaks

    # Verify that all test files were created
    expected_files = [
        "test_raster_normal.svg",
        "test_raster_start0.svg",
        "test_raster_end_last.svg",
        "test_raster_multiple.svg",
        "test_raster_three_states.svg",
        "test_raster_single_state.svg",
        "test_raster_boundary_same.svg",
        "test_raster_custom_period.svg",
    ]

    for file in expected_files:
        assert os.path.exists(file), f"Expected file {file} was not created"

    # Clean up test files
    for file in expected_files:
        os.remove(file)


@pytest.mark.parametrize(
    "define_epochs_by,epochs,cell_set_files,expected_epochs",
    [
        [
            "files",
            "(0, 10), (15, 25)",
            cell_sets * 2,
            "(0, 745), (745, 1490)",
        ],
        [
            "files",
            None,
            cell_sets * 2,
            "(0, 745), (745, 1490)",
        ],
        [
            "global file time",
            "(0, 300), (745, 1045)",
            cell_sets * 2,
            "(0, 300), (745, 1045)",
        ],
        [
            "global file time",
            None,
            cell_sets * 2,
            "(0, 745), (745, 1490)",
        ],
        [
            "local file time",
            "[(0, 300)], [(0, 300), (400, 700)]",
            cell_sets * 2,
            "(0, 300), (745, 1045), (1145, 1445)",
        ],
        [
            "local file time",
            None,
            cell_sets * 2,
            "(0, 745), (745, 1490)",
        ],
    ],
)
def test_redefine_epochs(
    define_epochs_by,
    epochs,
    cell_set_files,
    expected_epochs,
):
    """
    Test the redefine epochs function.
    """

    cellset = isx.CellSet.read(cell_set_files[0])
    period = cellset.timing.period.secs_float

    boundaries = _get_cellset_boundaries(
        cell_set_files=cell_set_files,
        period=period,
    )

    epochs = _redefine_epochs(
        define_epochs_by=define_epochs_by,
        epochs=epochs,
        boundaries=boundaries,
    )

    assert epochs == expected_epochs, (
        f"epochs {epochs} do not match expected epochs {expected_epochs}"
    )
