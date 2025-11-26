import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.epoch_activity import run
from utils.utils import (
    _redefine_epochs,
    _get_cellset_boundaries,
)

cell_sets = ["/ideas/data/input_cellset.isxd"]
event_sets = ["/ideas/data/input_cellset-ED.isxd"]

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
        heatmap="0, 3",
    ),
]
expected_output_files = [
    "Eventrate_activity_data.csv",
    "Eventrate_timecourse_data.npy",
    "Eventrate_Population_Timecourse.svg",
    "Eventrate_Single_Cell_Timecourse.svg",
    "Population_Eventrate_differences.svg",
    "Population_Eventrate.svg",
    "Population_Traces_Correlation.npy",
    "Population_Traces_differences.svg",
    "Population_Traces.svg",
    "Trace_Preview.svg",
    "Traces_activity_data.csv",
    "Traces_timecourse_data.npy",
    "Traces_Population_Timecourse.svg",
    "Traces_Single_Cell_Timecourse.svg",
]


@pytest.mark.parametrize("params", valid_inputs)
def test_epoch_activity(params, tmp_path):
    """Check that code runs without error with valid inputs."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    run(**params)
    try:  # When running locally
        actual_output_files = os.listdir(tmp_path)
    except FileNotFoundError:  # when running on IDEAS
        actual_output_files = os.listdir()

    # check that the output files are created
    for output_file in expected_output_files:
        assert output_file in actual_output_files

    # validate number of cells in output files
    for f in ["Eventrate_activity_data.csv", "Traces_activity_data.csv"]:
        df = pd.read_csv(f)
        assert len(df) == 76 * 3  # 76 accepted cells, 3 epochs

    os.chdir(cwd)

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
        {
            "state": ["A"] * 20
            + ["B"] * 20
            + ["A"] * 20
            + ["B"] * 20
            + ["A"] * 20
        }
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
        event_times = np.sort(
            np.random.uniform(0, 100, np.random.randint(5, 20))
        )
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
        {
            "state": ["A"] * 20
            + ["B"] * 20
            + ["A"] * 20
            + ["B"] * 20
            + ["A"] * 20
        }
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
            "global file time",
            "(0, 300), (745, 1045)",
            cell_sets * 2,
            "(0, 300), (745, 1045)",
        ],
        [
            "local file time",
            "[(0, 300)], [(0, 300), (400, 700)]",
            cell_sets * 2,
            "(0, 300), (745, 1045), (1145, 1445)",
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
    import isx

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

    assert (
        epochs == expected_epochs
    ), f"epochs {epochs} do not match expected epochs {expected_epochs}"
