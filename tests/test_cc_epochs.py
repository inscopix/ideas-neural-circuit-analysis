import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open

from analysis.comb_comp_epochs import run_cc_epochs
from ideas.exceptions import IdeasError

# Real file paths for integration tests
g1_trace_files = ["data/Traces1.csv", "data/Traces2.csv"]
g2_trace_files = ["data/Traces3.csv", "data/Traces4.csv"]
g1_event_files = ["data/Eventrate1.csv", "data/Eventrate2.csv"]
g2_event_files = ["data/Eventrate3.csv", "data/Eventrate4.csv"]
g1_corr_files = ["data/corr1.npy", "data/corr2.npy"]
g2_corr_files = ["data/corr3.npy", "data/corr4.npy"]

valid_inputs = [
    # valid inputs
    dict(
        group1_traces=g1_trace_files,
        group2_traces=g2_trace_files,
        group1_events=g1_event_files,
        group2_events=g2_event_files,
        group1_corr=g1_corr_files,
        group2_corr=g2_corr_files,
        epoch_names="early drug, BASELINE, late-Drug",
        group1_name="Group 1",
        group2_name="Group 2",
        epoch_colors="blue, orange, red",
        group1_color="blue",
        group2_color="red",
        multiple_correction="bonf",
        effect_size="cohen",
    ),
]

expected_output_files = [
    "ANOVA_results.csv",
    "group_1_combined_data.csv",
    "group_1_correlation_data_comparison.svg",
    "group_1_correlation_data.csv",
    "group_1_correlation_preview.svg",
    "group_1_Eventrate_Activity_comparison.svg",
    "group_1_Eventrate_preview.svg",
    "group_1_Trace_Activity_comparison.svg",
    "group_1_Trace_preview.svg",
    "group_2_combined_data.csv",
    "group_2_correlation_data_comparison.svg",
    "group_2_correlation_data.csv",
    "group_2_correlation_preview.svg",
    "group_2_Eventrate_Activity_comparison.svg",
    "group_2_Eventrate_preview.svg",
    "group_2_Trace_Activity_comparison.svg",
    "group_2_Trace_preview.svg",
    "mixed_ANOVA_Eventrate_comparison.svg",
    "mixed_ANOVA_Trace_comparison.svg",
    "mixed_correlation_ANOVA_comparison.svg",
    "mixed_correlation_pairwise_comparison.svg",
    "mixed_pairwise_Eventrate_comparison.svg",
    "mixed_pairwise_Trace_comparison.svg",
    "pairwise_results.csv",
]
outputs_to_load = {
    "ANOVA_results.csv": (28, 11),
    "group_1_combined_data.csv": (1713, 6),
    "group_1_correlation_data.csv": (6, 4),
    "group_2_combined_data.csv": (1731, 6),
    "group_2_correlation_data.csv": (6, 4),
    "pairwise_results.csv": (52, 18),
}


# Mock fixtures for isolated unit tests
@pytest.fixture
def mock_files_setup():
    return {
        # Group 1 files
        "g1_trace_files": [
            "/mock/data/traces_g1_1.csv",
            "/mock/data/traces_g1_2.csv",
            "/mock/data/traces_g1_3.csv",
        ],
        "g1_event_files": [
            "/mock/data/events_g1_1.csv",
            "/mock/data/events_g1_2.csv",
            "/mock/data/events_g1_3.csv",
        ],
        "g1_corr_files": [
            "/mock/data/corr_g1_1.npy",
            "/mock/data/corr_g1_2.npy",
            "/mock/data/corr_g1_3.npy",
        ],
        # Group 2 files
        "g2_trace_files": [
            "/mock/data/traces_g2_1.csv",
            "/mock/data/traces_g2_2.csv",
            "/mock/data/traces_g2_3.csv",
        ],
        "g2_event_files": [
            "/mock/data/events_g2_1.csv",
            "/mock/data/events_g2_2.csv",
            "/mock/data/events_g2_3.csv",
        ],
        "g2_corr_files": [
            "/mock/data/corr_g2_1.npy",
            "/mock/data/corr_g2_2.npy",
            "/mock/data/corr_g2_3.npy",
        ],
    }


@pytest.fixture
def mock_trace_data():
    """Generate mock trace data dataframe with multiple epochs"""
    np.random.seed(42)
    epochs = ["Epoch1", "Epoch2", "Epoch3"]
    cells = [f"Cell{i}" for i in range(10)]

    data = []
    for epoch in epochs:
        for cell in cells:
            # Different means for different epochs
            if epoch == "Epoch1":
                value = np.random.normal(5, 1)
            elif epoch == "Epoch2":
                value = np.random.normal(6, 1)
            else:
                value = np.random.normal(4, 1)

            data.append(
                {
                    "Epoch": epoch,
                    "Cell": cell,
                    "Activity": value,
                    "File": "mock_file.csv",
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def mock_event_data():
    """Generate mock event data dataframe with multiple epochs"""
    np.random.seed(43)
    epochs = ["Epoch1", "Epoch2", "Epoch3"]
    cells = [f"Cell{i}" for i in range(10)]

    data = []
    for epoch in epochs:
        for cell in cells:
            # Different means for different epochs
            if epoch == "Epoch1":
                value = np.random.normal(2, 0.5)
            elif epoch == "Epoch2":
                value = np.random.normal(3, 0.5)
            else:
                value = np.random.normal(1.5, 0.5)

            data.append(
                {
                    "Epoch": epoch,
                    "Cell": cell,
                    "Activity": value,
                    "File": "mock_event_file.csv",
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def mock_correlation_data():
    """Generate mock correlation matrices for testing"""
    np.random.seed(44)
    epochs = ["Epoch1", "Epoch2", "Epoch3"]
    num_cells = 10

    # Create a dictionary with correlation matrices for each epoch
    correlation_mtx = {}
    for epoch in epochs:
        if epoch == "Epoch1":
            base_corr = 0.3
        elif epoch == "Epoch2":
            base_corr = 0.5
        else:
            base_corr = 0.2

        # Create a mock correlation matrix with some structure
        matrix = np.zeros((num_cells, num_cells))
        for i in range(num_cells):
            for j in range(num_cells):
                if i != j:  # Zero diagonal
                    matrix[i, j] = base_corr + np.random.normal(0, 0.2)

        correlation_mtx[epoch] = matrix

    return correlation_mtx


@pytest.fixture
def mock_csv_read(mock_trace_data, mock_event_data):
    def mock_read_csv(file_path, *args, **kwargs):
        if "traces" in file_path:
            return mock_trace_data.copy()
        elif "events" in file_path:
            return mock_event_data.copy()
        return pd.DataFrame()

    return mock_read_csv


@pytest.fixture
def mock_numpy_load(mock_correlation_data):
    def mock_load(file_path, *args, **kwargs):
        if "corr" in file_path:
            return mock_correlation_data
        return None

    return mock_load


# Integration test with real file paths
@pytest.mark.parametrize("params", valid_inputs)
def test_cc_epochs_integration(params, cleanup_plots):
    """Check that code runs without error with valid inputs."""
    run_cc_epochs(**params)
    try:  # When running locally
        actual_output_files = os.listdir("outputs/comb_comp_epochs__run")
    except FileNotFoundError:
        actual_output_files = os.listdir()

    # check that the output files are created
    for output_file in expected_output_files:
        assert output_file in actual_output_files

    # check that the output files are correct shape
    for output_file, shape in outputs_to_load.items():
        df = pd.read_csv(output_file)
        assert df.shape == shape

    # Clean up
    for output_file in expected_output_files:
        os.remove(output_file)


# Unit tests with mocks


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())  # Mock figure saving
@patch("matplotlib.pyplot.close", MagicMock())  # Mock figure closing
@patch("pandas.DataFrame.to_csv", MagicMock())  # Mock CSV saving
@patch("numpy.save", MagicMock())  # Mock numpy saving
@patch("json.dump", MagicMock())  # Mock json saving
@patch("builtins.open", mock_open())  # Mock file opening
def test_cc_epochs_full_data(
    mock_files_setup, mock_csv_read, mock_numpy_load, cleanup_plots
):
    """Test run_cc_epochs with complete data for both groups"""
    with patch("pandas.read_csv", mock_csv_read):
        with patch("numpy.load", mock_numpy_load):
            # Run the function with all data
            run_cc_epochs(
                group1_traces=mock_files_setup["g1_trace_files"],
                group2_traces=mock_files_setup["g2_trace_files"],
                group1_events=mock_files_setup["g1_event_files"],
                group2_events=mock_files_setup["g2_event_files"],
                group1_corr=mock_files_setup["g1_corr_files"],
                group2_corr=mock_files_setup["g2_corr_files"],
                epoch_names="Epoch1, Epoch2, Epoch3",
                group1_name="Group1",
                group2_name="Group2",
                epoch_colors="blue, red, green",
                multiple_correction="bonf",
                effect_size="cohen",
            )
            # We mainly verify that it runs without exceptions


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())
@patch("matplotlib.pyplot.close", MagicMock())
@patch("pandas.DataFrame.to_csv", MagicMock())
@patch("numpy.save", MagicMock())
@patch("json.dump", MagicMock())
@patch("builtins.open", mock_open())
def test_cc_epochs_no_events(
    mock_files_setup, mock_csv_read, mock_numpy_load, cleanup_plots
):
    """Test run_cc_epochs without event data"""
    with patch("pandas.read_csv", mock_csv_read):
        with patch("numpy.load", mock_numpy_load):
            # Run without event files
            run_cc_epochs(
                group1_traces=mock_files_setup["g1_trace_files"],
                group2_traces=mock_files_setup["g2_trace_files"],
                group1_events=None,
                group2_events=None,
                group1_corr=mock_files_setup["g1_corr_files"],
                group2_corr=mock_files_setup["g2_corr_files"],
                epoch_names="Epoch1, Epoch2, Epoch3",
                group1_name="Group1",
                group2_name="Group2",
                epoch_colors="blue, red, green",
                multiple_correction="bonf",
                effect_size="cohen",
            )
            # We mainly verify that it runs without exceptions


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())
@patch("matplotlib.pyplot.close", MagicMock())
@patch("pandas.DataFrame.to_csv", MagicMock())
@patch("numpy.save", MagicMock())
@patch("json.dump", MagicMock())
@patch("builtins.open", mock_open())
def test_cc_epochs_no_correlation(
    mock_files_setup, mock_csv_read, mock_numpy_load, cleanup_plots
):
    """Test run_cc_epochs without correlation data"""
    with patch("pandas.read_csv", mock_csv_read):
        with patch("numpy.load", mock_numpy_load):
            # Run without correlation files
            run_cc_epochs(
                group1_traces=mock_files_setup["g1_trace_files"],
                group2_traces=mock_files_setup["g2_trace_files"],
                group1_events=mock_files_setup["g1_event_files"],
                group2_events=mock_files_setup["g2_event_files"],
                group1_corr=None,
                group2_corr=None,
                epoch_names="Epoch1, Epoch2, Epoch3",
                group1_name="Group1",
                group2_name="Group2",
                epoch_colors="blue, red, green",
                multiple_correction="bonf",
                effect_size="cohen",
            )
            # We mainly verify that it runs without exceptions


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())
@patch("matplotlib.pyplot.close", MagicMock())
@patch("pandas.DataFrame.to_csv", MagicMock())
@patch("numpy.save", MagicMock())
@patch("json.dump", MagicMock())
@patch("builtins.open", mock_open())
def test_cc_epochs_one_group(
    mock_files_setup, mock_csv_read, mock_numpy_load, cleanup_plots
):
    """Test run_cc_epochs with only one group"""
    with patch("pandas.read_csv", mock_csv_read):
        with patch("numpy.load", mock_numpy_load):
            # Run with only group 1
            run_cc_epochs(
                group1_traces=mock_files_setup["g1_trace_files"],
                group2_traces=None,
                group1_events=mock_files_setup["g1_event_files"],
                group2_events=None,
                group1_corr=mock_files_setup["g1_corr_files"],
                group2_corr=None,
                epoch_names="Epoch1, Epoch2, Epoch3",
                group1_name="Group1",
                group2_name=None,
                epoch_colors="blue, red, green",
                multiple_correction="bonf",
                effect_size="cohen",
            )
            # We mainly verify that it runs without exceptions


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())
@patch("matplotlib.pyplot.close", MagicMock())
@patch("pandas.DataFrame.to_csv", MagicMock())
@patch("numpy.save", MagicMock())
@patch("json.dump", MagicMock())
@patch("builtins.open", mock_open())
def test_cc_epochs_minimal_epochs(
    mock_files_setup, mock_csv_read, cleanup_plots
):
    """Test run_cc_epochs with minimal number of epochs (2)"""

    # Override mock_csv_read for this test
    def minimal_mock_read_csv(file_path, *args, **kwargs):
        np.random.seed(42)
        epochs = ["Epoch1", "Epoch2"]
        cells = [f"Cell{i}" for i in range(5)]

        data = []
        for epoch in epochs:
            for cell in cells:
                value = (
                    np.random.normal(5, 1)
                    if epoch == "Epoch1"
                    else np.random.normal(6, 1)
                )

                data.append(
                    {
                        "Epoch": epoch,
                        "Cell": cell,
                        "Activity": value,
                        "File": "mock_file.csv",
                    }
                )

        return pd.DataFrame(data)

    with patch("pandas.read_csv", minimal_mock_read_csv):
        # Mock minimal correlation data
        def minimal_mock_numpy_load(file_path, *args, **kwargs):
            epochs = ["Epoch1", "Epoch2"]
            num_cells = 5
            correlation_mtx = {}

            for epoch in epochs:
                base_corr = 0.3 if epoch == "Epoch1" else 0.5
                matrix = np.zeros((num_cells, num_cells))
                for i in range(num_cells):
                    for j in range(num_cells):
                        if i != j:
                            matrix[i, j] = base_corr + np.random.normal(0, 0.2)
                correlation_mtx[epoch] = matrix

            return correlation_mtx

        with patch("numpy.load", minimal_mock_numpy_load):
            # Run with minimal epochs
            run_cc_epochs(
                group1_traces=mock_files_setup["g1_trace_files"][:2],
                group2_traces=mock_files_setup["g2_trace_files"][:2],
                group1_events=None,
                group2_events=None,
                group1_corr=None,
                group2_corr=None,
                epoch_names="Epoch1, Epoch2",
                group1_name="Group1",
                group2_name="Group2",
                epoch_colors="blue, red",
                multiple_correction="bonf",
                effect_size="cohen",
            )
            # We mainly verify that it runs without exceptions


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())
@patch("matplotlib.pyplot.close", MagicMock())
@patch("pandas.DataFrame.to_csv", MagicMock())
@patch("numpy.save", MagicMock())
@patch("json.dump", MagicMock())
@patch("builtins.open", mock_open())
def test_cc_epochs_many_epochs(mock_files_setup, mock_csv_read, cleanup_plots):
    """Test run_cc_epochs with larger number of epochs (5)"""

    # Override mock_csv_read for this test
    def many_epochs_mock_read_csv(file_path, *args, **kwargs):
        np.random.seed(42)
        epochs = ["Epoch1", "Epoch2", "Epoch3", "Epoch4", "Epoch5"]
        cells = [f"Cell{i}" for i in range(5)]

        data = []
        for epoch in epochs:
            base_value = {
                "Epoch1": 5,
                "Epoch2": 6,
                "Epoch3": 4,
                "Epoch4": 7,
                "Epoch5": 3,
            }[epoch]
            for cell in cells:
                value = np.random.normal(base_value, 1)
                data.append(
                    {
                        "Epoch": epoch,
                        "Cell": cell,
                        "Activity": value,
                        "File": "mock_file.csv",
                    }
                )

        return pd.DataFrame(data)

    with patch("pandas.read_csv", many_epochs_mock_read_csv):
        # Run with more epochs
        run_cc_epochs(
            group1_traces=mock_files_setup["g1_trace_files"][:2],
            group2_traces=None,
            group1_events=None,
            group2_events=None,
            group1_corr=None,
            group2_corr=None,
            epoch_names="Epoch1, Epoch2, Epoch3, Epoch4, Epoch5",
            group1_name="Group1",
            group2_name=None,
            epoch_colors="blue, red, green, yellow, purple",
            multiple_correction="bonf",
            effect_size="cohen",
        )
        # We mainly verify that it runs without exceptions


@patch("os.path.basename", lambda x: "mock_basename.csv")
@patch("matplotlib.pyplot.savefig", MagicMock())
@patch("matplotlib.pyplot.close", MagicMock())
@patch("pandas.DataFrame.to_csv", MagicMock())
@patch("numpy.save", MagicMock())
@patch("json.dump", MagicMock())
@patch("builtins.open", mock_open())
def test_cc_epochs_different_corrections(
    mock_files_setup, mock_csv_read, mock_numpy_load, cleanup_plots
):
    """Test run_cc_epochs with different multiple comparison correction methods"""
    with patch("pandas.read_csv", mock_csv_read):
        with patch("numpy.load", mock_numpy_load):
            correction_methods = ["bonf", "holm", "fdr_bh"]

            for method in correction_methods:
                # Run with different correction method
                run_cc_epochs(
                    group1_traces=mock_files_setup["g1_trace_files"][:2],
                    group2_traces=None,
                    group1_events=None,
                    group2_events=None,
                    group1_corr=None,
                    group2_corr=None,
                    epoch_names="Epoch1, Epoch2, Epoch3",
                    group1_name="Group1",
                    group2_name=None,
                    epoch_colors="blue, red, green",
                    multiple_correction=method,
                    effect_size="cohen",
                )
                # We mainly verify that it runs without exceptions


# Error case tests
def test_cc_epochs_insufficient_traces():
    """Test run_cc_epochs with insufficient trace files (should raise error)"""
    with pytest.raises(
        IdeasError, match="Group1 must have more than 1 trace file"
    ):
        run_cc_epochs(
            group1_traces=["/mock/data/single_trace.csv"],
            group2_traces=None,
            group1_events=None,
            group2_events=None,
            group1_corr=None,
            group2_corr=None,
            epoch_names="Epoch1, Epoch2, Epoch3",
            group1_name="Group1",
            group2_name=None,
            epoch_colors="blue, red, green",
            multiple_correction="bonf",
            effect_size="cohen",
        )


def test_cc_epochs_inconsistent_epochs_colors():
    """Test run_cc_epochs with inconsistent epoch names and colors (should raise error)"""
    with pytest.raises(
        IdeasError,
        match="The number of epoch names and colors must be the same",
    ):
        run_cc_epochs(
            group1_traces=["/mock/data/trace1.csv", "/mock/data/trace2.csv"],
            group2_traces=None,
            group1_events=None,
            group2_events=None,
            group1_corr=None,
            group2_corr=None,
            epoch_names="Epoch1, Epoch2, Epoch3",
            group1_name="Group1",
            group2_name=None,
            epoch_colors="blue, red",  # Only 2 colors for 3 epochs
            multiple_correction="bonf",
            effect_size="cohen",
        )


def test_cc_epochs_invalid_group_name():
    """Test run_cc_epochs with invalid group name containing special characters"""
    with pytest.raises(
        IdeasError, match="Group1 name contains special characters"
    ):
        run_cc_epochs(
            group1_traces=["/mock/data/trace1.csv", "/mock/data/trace2.csv"],
            group2_traces=None,
            group1_events=None,
            group2_events=None,
            group1_corr=None,
            group2_corr=None,
            epoch_names="Epoch1, Epoch2, Epoch3",
            group1_name="Group1/invalid*chars",
            group2_name=None,
            epoch_colors="blue, red, green",
            multiple_correction="bonf",
            effect_size="cohen",
        )


def test_cc_epochs_medium_neuronal_data(tmp_path, monkeypatch, cleanup_plots):
    """Test run_cc_epochs with medium-sized neuronal dataset (50 cells).

    This test verifies that the combination and comparison of epochs function
    works correctly with a moderate amount of data (50 cells). It creates
    artificial trace data, event rate data, and correlation matrices for testing.

    Args:
        tmp_path: Pytest fixture for temporary directory
        monkeypatch: Pytest fixture for modifying the environment
    """
    import pandas as pd
    import numpy as np
    import pytest
    from toolbox.tools.comb_comp_epochs import run_cc_epochs
    from ideas.exceptions import IdeasError

    # Generate medium data: 50 cells across 3 epochs (each simulated with one observation per cell)
    epochs = ["Epoch1", "Epoch2", "Epoch3"]
    num_cells = 50
    data = []
    for epoch in epochs:
        for cell in range(num_cells):
            value = np.random.normal(loc=5.0, scale=1.0)
            data.append(
                {
                    "Epoch": epoch,
                    "Cell": cell,
                    "Activity": value,
                    "File": "medium_trace.csv",
                }
            )
    df_trace = pd.DataFrame(data)
    trace_file = tmp_path / "medium_trace.csv"
    df_trace.to_csv(trace_file, index=False)

    # Generate matching event data
    df_event = df_trace.copy()
    df_event["Activity"] = np.random.normal(
        loc=2.0, scale=0.5, size=len(df_event)
    )
    event_file = tmp_path / "medium_event.csv"
    df_event.to_csv(event_file, index=False)

    # Generate correlation data saved as a numpy file
    # Each epoch's matrix now includes both positive and negative correlations
    corr_data = {}
    for epoch in epochs:
        # Create matrix with both positive and negative values
        matrix = np.random.uniform(-0.5, 0.5, (num_cells, num_cells))
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(matrix, 1.0)
        corr_data[epoch] = matrix

    corr_file = tmp_path / "medium_corr.npy"
    np.save(corr_file, corr_data)

    # Mock plt.savefig to avoid actual file saving
    with patch("matplotlib.pyplot.savefig"):
        with patch("matplotlib.pyplot.close"):
            try:
                run_cc_epochs(
                    group1_traces=[str(trace_file), str(trace_file)],
                    group2_traces=None,  # Test single group scenario
                    group1_events=[str(event_file), str(event_file)],
                    group2_events=None,
                    group1_corr=[str(corr_file), str(corr_file)],
                    group2_corr=None,
                    epoch_names="Epoch1, Epoch2, Epoch3",
                    group1_name="GroupMedium",
                    group2_name=None,
                    epoch_colors="blue, red, green",
                    multiple_correction="bonf",
                    effect_size="cohen",
                )
            except IdeasError as e:
                pytest.fail(f"run_cc_epochs failed with medium data: {e}")


def test_cc_epochs_large_neuronal_data(tmp_path, monkeypatch, cleanup_plots):
    """Test run_cc_epochs with large-sized neuronal dataset (200 cells).

    This test verifies that the combination and comparison of epochs function
    works correctly with a large amount of data (200 cells). It creates
    artificial trace data, event rate data, and correlation matrices with
    both positive and negative correlations to ensure robust testing.

    Args:
        tmp_path: Pytest fixture for temporary directory
        monkeypatch: Pytest fixture for modifying the environment
    """
    import pandas as pd
    import numpy as np
    import pytest
    from toolbox.tools.comb_comp_epochs import run_cc_epochs
    from ideas.exceptions import IdeasError

    # Generate large data: 200 cells across 3 epochs.
    epochs = ["Epoch1", "Epoch2", "Epoch3"]
    num_cells = 200
    data = []
    for epoch in epochs:
        for cell in range(num_cells):
            value = np.random.normal(loc=5.0, scale=1.0)
            data.append(
                {
                    "Epoch": epoch,
                    "Cell": cell,
                    "Activity": value,
                    "File": "large_trace.csv",
                }
            )
    df_trace = pd.DataFrame(data)
    trace_file = tmp_path / "large_trace.csv"
    df_trace.to_csv(trace_file, index=False)

    # Generate matching event data
    df_event = df_trace.copy()
    df_event["Activity"] = np.random.normal(
        loc=2.0, scale=0.5, size=len(df_event)
    )
    event_file = tmp_path / "large_event.csv"
    df_event.to_csv(event_file, index=False)

    # Generate correlation data with both positive and negative values
    corr_data = {}
    for epoch in epochs:
        # Create matrix with both positive and negative values
        matrix = np.random.uniform(-0.5, 0.5, (num_cells, num_cells))
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        # Set diagonal to 1
        np.fill_diagonal(matrix, 1.0)
        corr_data[epoch] = matrix

    corr_file = tmp_path / "large_corr.npy"
    np.save(corr_file, corr_data)

    # Mock plt.savefig to avoid actual file saving
    with patch("matplotlib.pyplot.savefig"):
        with patch("matplotlib.pyplot.close"):
            try:
                run_cc_epochs(
                    group1_traces=[str(trace_file), str(trace_file)],
                    group2_traces=None,
                    group1_events=[str(event_file), str(event_file)],
                    group2_events=None,
                    group1_corr=[str(corr_file), str(corr_file)],
                    group2_corr=None,
                    epoch_names="Epoch1, Epoch2, Epoch3",
                    group1_name="GroupLarge",
                    group2_name=None,
                    epoch_colors="blue, red, green",
                    multiple_correction="bonf",
                    effect_size="cohen",
                )
            except IdeasError as e:
                pytest.fail(f"run_cc_epochs failed with large data: {e}")


@pytest.fixture
def generate_neuronal_data():
    """Fixture to generate neural trace, event, and correlation data of different sizes.

    Returns a function that creates test data with specified parameters.
    """

    def _generate(
        tmp_path,
        num_cells,
        num_epochs=3,
        trace_value_range=(3, 7),
        event_value_range=(1, 3),
        correlation_range=(0.1, 0.9),
        file_count=2,
    ):
        """Generate test data files for neuronal analysis.

        Args:
            tmp_path: Pytest fixture for temporary directory
            num_cells: Number of cells to simulate
            num_epochs: Number of epochs to include
            trace_value_range: Range of values for trace activity
            event_value_range: Range of values for event rates
            correlation_range: Range for correlation values
            file_count: Number of duplicate files to create (for ANOVA test requirements)

        Returns:
            dict: Paths to generated files and metadata
        """
        np.random.seed(
            42 + num_cells
        )  # Vary seed by size to avoid identical patterns
        epochs = [f"Epoch{i + 1}" for i in range(num_epochs)]

        # Generate trace data
        trace_files = []
        event_files = []
        corr_files = []

        for file_idx in range(file_count):
            # Create unique trace data with specific patterns per epoch
            trace_data = []
            for epoch_idx, epoch in enumerate(epochs):
                # Make each epoch have a distinct mean value
                mean_value = trace_value_range[0] + (
                    trace_value_range[1] - trace_value_range[0]
                ) * (epoch_idx / max(1, num_epochs - 1))

                for cell in range(num_cells):
                    value = np.random.normal(loc=mean_value, scale=1.0)
                    trace_data.append(
                        {
                            "Epoch": epoch,
                            "Cell": cell,
                            "Activity": value,
                            "File": f"trace_file_{file_idx}.csv",
                        }
                    )

            df_trace = pd.DataFrame(trace_data)
            trace_file = tmp_path / f"trace_{file_idx}.csv"
            df_trace.to_csv(trace_file, index=False)
            trace_files.append(str(trace_file))

            # Create matching event data with its own pattern
            event_data = []
            for epoch_idx, epoch in enumerate(epochs):
                mean_value = event_value_range[0] + (
                    event_value_range[1] - event_value_range[0]
                ) * (epoch_idx / max(1, num_epochs - 1))

                for cell in range(num_cells):
                    value = np.random.normal(loc=mean_value, scale=0.5)
                    event_data.append(
                        {
                            "Epoch": epoch,
                            "Cell": cell,
                            "Activity": value,
                            "File": f"event_file_{file_idx}.csv",
                        }
                    )

            df_event = pd.DataFrame(event_data)
            event_file = tmp_path / f"event_{file_idx}.csv"
            df_event.to_csv(event_file, index=False)
            event_files.append(str(event_file))

            # Generate correlation data with realistic structure
            corr_data = {}
            for epoch in epochs:
                # Create correlation matrix with community structure
                matrix = np.zeros((num_cells, num_cells))

                # Create communities (groups of correlated cells)
                community_size = min(10, num_cells // 3 or 1)
                for i in range(num_cells):
                    for j in range(i + 1, num_cells):
                        # Higher correlation for cells in same community
                        if (i // community_size) == (j // community_size):
                            matrix[i, j] = np.random.uniform(
                                correlation_range[0] + 0.2,
                                correlation_range[1],
                            )
                        else:
                            # Include some negative correlations to ensure both pos/neg data exists
                            matrix[i, j] = np.random.uniform(
                                -0.5, correlation_range[0] + 0.1
                            )

                        # Symmetric matrix
                        matrix[j, i] = matrix[i, j]

                # Ensure diagonal is 1.0 (self-correlation)
                np.fill_diagonal(matrix, 1.0)
                corr_data[epoch] = matrix

            corr_file = tmp_path / f"corr_{file_idx}.npy"
            np.save(corr_file, corr_data)
            corr_files.append(str(corr_file))

        return {
            "trace_files": trace_files,
            "event_files": event_files,
            "corr_files": corr_files,
            "epochs": epochs,
            "num_cells": num_cells,
        }

    return _generate


# Replace the simple medium and large data tests with more comprehensive tests
def test_cc_epochs_medium_data(
    tmp_path, generate_neuronal_data, monkeypatch, cleanup_plots
):
    """Test run_cc_epochs with medium-sized dataset (50 cells)."""
    # Generate medium size data (50 cells, 3 epochs)
    data = generate_neuronal_data(
        tmp_path,
        num_cells=50,
        trace_value_range=(4, 8),
        event_value_range=(1.5, 3.5),
    )

    # Directory to check output files
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.chdir(output_dir)

    # Run the function
    run_cc_epochs(
        group1_traces=data["trace_files"],
        group1_events=data["event_files"],
        group1_corr=data["corr_files"],
        group2_traces=None,  # Test single group scenario
        group2_events=None,
        group2_corr=None,
        epoch_names=",".join(data["epochs"]),
        epoch_colors="blue,red,green",
        group1_name="MediumGroup",
        group1_color="blue",
        multiple_correction="bonf",
        effect_size="cohen",
    )

    # Verify files were created
    expected_files = [
        "ANOVA_results.csv",
        "pairwise_results.csv",
        "group_1_combined_data.csv",
        "group_1_correlation_data.csv",
    ]

    for filename in expected_files:
        assert os.path.exists(filename), f"Output file {filename} not created"

    # Check structure of key output files
    anova_df = pd.read_csv("ANOVA_results.csv")
    pairwise_df = pd.read_csv("pairwise_results.csv")
    combined_df = pd.read_csv("group_1_combined_data.csv")

    # Verify ANOVA and pairwise comparison results make sense
    assert "p-unc" in anova_df.columns
    assert "p-corr" in pairwise_df.columns
    assert "F" in anova_df.columns

    # Check that combined data has the right structure
    # Account for file_count (default=2) in the data generation function
    file_count = len(data["trace_files"])
    assert (
        len(combined_df) == 50 * len(data["epochs"]) * file_count
    )  # num_cells * num_epochs * file_count


def test_cc_epochs_large_data(
    tmp_path, generate_neuronal_data, monkeypatch, cleanup_plots
):
    """Test run_cc_epochs with large-sized dataset (200 cells)."""
    # Generate large size data (200 cells, 3 epochs)
    data = generate_neuronal_data(
        tmp_path,
        num_cells=200,
        trace_value_range=(3, 9),
        event_value_range=(1.0, 4.0),
    )

    # Directory to check output files
    output_dir = tmp_path / "output_large"
    output_dir.mkdir()
    monkeypatch.chdir(output_dir)

    # Run the function testing multiple groups
    run_cc_epochs(
        group1_traces=data["trace_files"],
        group1_events=data["event_files"],
        group1_corr=data["corr_files"],
        group2_traces=data[
            "trace_files"
        ],  # Test two group scenario with same data
        group2_events=data["event_files"],
        group2_corr=data["corr_files"],
        epoch_names=",".join(data["epochs"]),
        epoch_colors="blue,red,green",
        group1_name="LargeGroup1",
        group2_name="LargeGroup2",
        group1_color="blue",
        group2_color="red",
        multiple_correction="fdr_bh",  # Test different correction method
        effect_size="cohen",
    )

    # Check group comparison outputs specifically
    expected_files = [
        "group_1_combined_data.csv",
        "group_2_combined_data.csv",
        "mixed_ANOVA_Trace_comparison.svg",
        "mixed_pairwise_Trace_comparison.svg",
    ]

    for filename in expected_files:
        assert os.path.exists(filename), f"Output file {filename} not created"

    # Check structure of key output files
    anova_df = pd.read_csv("ANOVA_results.csv")

    # Verify results include group comparison
    group_comparisons = anova_df[anova_df["Source"] == "Group"]
    assert len(group_comparisons) > 0, "No group comparisons found in results"


def test_cc_epochs_different_epochs_count(
    tmp_path, generate_neuronal_data, monkeypatch, cleanup_plots
):
    """Test run_cc_epochs with different numbers of epochs."""
    # Generate data with 5 epochs
    data = generate_neuronal_data(tmp_path, num_cells=30, num_epochs=5)

    # Directory to check output files
    output_dir = tmp_path / "output_epochs"
    output_dir.mkdir()
    monkeypatch.chdir(output_dir)

    # Run function with 5 epochs
    run_cc_epochs(
        group1_traces=data["trace_files"],
        group1_events=data["event_files"],
        group1_corr=data["corr_files"],
        group2_traces=None,
        group2_events=None,
        group2_corr=None,
        epoch_names=",".join(data["epochs"]),
        epoch_colors="blue,green,red,purple,orange",
        group1_name="MultiEpochGroup",
        group1_color="blue",
        multiple_correction="bonf",
        effect_size="cohen",
    )

    # Verify files
    pairwise_df = pd.read_csv("pairwise_results.csv")

    # Print columns and sample data for debugging
    print(f"Columns: {list(pairwise_df.columns)}")
    print(f"First few rows:\n{pairwise_df.head()}")

    # Check that pairwise file exists and has data
    assert len(pairwise_df) > 0, "Pairwise results file is empty"

    # Verify that epoch comparisons exist
    # First check if we have A and B columns that would contain epoch names
    if "A" in pairwise_df.columns and "B" in pairwise_df.columns:
        # Extract rows that compare across epochs
        epoch_pairs = set()
        for _, row in pairwise_df.iterrows():
            if isinstance(row["A"], str) and isinstance(row["B"], str):
                if row["A"] in data["epochs"] and row["B"] in data["epochs"]:
                    epoch_pairs.add((row["A"], row["B"]))

        # Check if we have a reasonable number of comparisons
        assert (
            len(epoch_pairs) > 0
        ), "No epoch comparisons found in A/B columns"
        print(f"Found {len(epoch_pairs)} unique epoch pairs in A/B columns")

    else:
        # If there are no A/B columns, look for epoch names in the Comparison column
        found_comparisons = 0
        for epoch1 in data["epochs"]:
            for epoch2 in data["epochs"]:
                if epoch1 != epoch2:
                    pattern = f"{epoch1}.*{epoch2}|{epoch2}.*{epoch1}"
                    matches = (
                        pairwise_df["Comparison"]
                        .str.contains(pattern, regex=True)
                        .sum()
                    )
                    found_comparisons += matches

        assert (
            found_comparisons > 0
        ), "No epoch comparisons found in Comparison column"
        print(
            f"Found {found_comparisons} epoch comparisons in Comparison column"
        )

    # Verify that we have results relating to Activity measurements
    activity_mentions = pairwise_df.apply(
        lambda row: any("Activity" in str(cell) for cell in row), axis=1
    ).sum()

    assert activity_mentions > 0, "No Activity measurements found in results"
    print(f"Found {activity_mentions} rows mentioning Activity")

    # Also check the ANOVA results
    anova_df = pd.read_csv("ANOVA_results.csv")
    assert len(anova_df) > 0, "ANOVA results file is empty"

    # Verify epoch-related results exist in ANOVA file
    epoch_effect = any("Epoch" in str(source) for source in anova_df["Source"])
    assert epoch_effect, "No epoch effects found in ANOVA results"
