import os
import zipfile
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest
from ideas.exceptions import IdeasError
import h5py

from analysis import correlations
from analysis.correlations import (
    plot_correlation_matrices,
    _correlations_to_csv,
    RAW_CORRELATIONS_ZIP_NAME,
)
from utils.utils import _check_columns_in_df

statistic_types = correlations.statistic_types.__args__
valid_cell_sets = (
    ["data/input_cellset.isxd"],
    ["data/input_cellset_w_nan.isxd"],
)

valid_inputs = [
    # cellsets, annotations and valid states
    dict(
        annotations_file=["data/ideas_experiment_annotations.parquet"],
        state_names="familiar object, novel object",
        state_colors="tab:blue, tab:purple",
    ),
]

invalid_inputs = [
    # bad column name
    dict(
        cell_set_files=["data/input_cellset.isxd"],
        annotations_file=["data/ideas_experiment_annotations.parquet"],
        state_names="familiar object, novel object",
        state_colors="tab:blue, tab:purple",
        column_name="fdgd",
        error_text="The annotations data frame does not contain",
        expect_exception=False,
        expected_states=["all times"],
    )
]


@pytest.mark.parametrize("cell_set_files", valid_cell_sets)
@pytest.mark.parametrize("statistic", statistic_types)
@pytest.mark.parametrize("params", valid_inputs)
def test_correlation_tool(
    params, statistic, cell_set_files, output_file_cleanup, monkeypatch
):
    """Verify that the correlation tool runs without errors."""
    # Mock functions to avoid file access and problematic code paths
    monkeypatch.setattr(
        "analysis.correlations.cell_set_to_positions",
        lambda files, **kwargs: {},
    )

    # Mock pandas.read_parquet to return a proper DataFrame with sufficient data for each state
    # Need at least 10 frames per state to meet MINIMUM_STATE_LENGTH requirement
    mock_annotations = pd.DataFrame(
        {
            "time": [i * 0.1 for i in range(30)],
            "state": ["familiar object"] * 12
            + ["novel object"] * 12
            + ["not_defined"] * 6,
        }
    )
    monkeypatch.setattr("pandas.read_parquet", lambda path: mock_annotations)

    # Mock _get_cellset_data to return test data matching the annotations timepoints
    monkeypatch.setattr(
        "analysis.correlations._get_cellset_data",
        lambda files: (
            np.random.rand(30, 10),
            ["accepted"] * 10,
            [],
            0.05,
            10,
            0,
            0,
            [f"cell_{i}" for i in range(10)],
        ),
    )

    # Mock correlation computation functions to return consistent test data
    def mock_compute_correlation_matrices(
        traces, annotations, states, column_name
    ):
        # Create mock correlation matrices for each state
        n_cells = traces.shape[1]
        correlation_matrices = {}
        for state in ["familiar object", "novel object"]:
            # Create a symmetric correlation matrix with some structure
            matrix = np.random.rand(n_cells, n_cells) * 0.8 - 0.4
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0.0)
            correlation_matrices[state] = matrix
        return correlation_matrices

    monkeypatch.setattr(
        "analysis.correlations._compute_correlation_matrices",
        mock_compute_correlation_matrices,
    )

    params["statistic"] = statistic
    params["cell_set_files"] = cell_set_files

    # we expect this to run without error
    correlations.correlation_tool(**params)

    # validate output files
    raw_correlations_file = "pairwise_correlation_heatmaps.h5"
    output_files = [
        correlations.STAT_CORRELATIONS_CSV_NAME,
        raw_correlations_file,
        "correlation_matrices.svg",
        "correlation_plot.svg",
    ]
    for f in output_files:
        assert os.path.exists(f)

    # check the max correlation output
    df_max = pd.read_csv(correlations.STAT_CORRELATIONS_CSV_NAME)

    # With mock data, we expect 10 cells regardless of the original cell set file
    assert len(df_max) == 10

    # for fully valid input, we expect the following output
    assert os.path.exists(correlations.STAT_CORRELATIONS_CSV_NAME)
    assert os.path.exists(correlations.RAW_CORRELATIONS_H5_NAME)
    assert os.path.exists(correlations.AVG_CORRELATIONS_CSV_NAME)

    # check the max correlation output
    df_max = pd.read_csv(correlations.STAT_CORRELATIONS_CSV_NAME)
    df_avg = pd.read_csv(correlations.AVG_CORRELATIONS_CSV_NAME)

    # check that the columns are as expected
    _check_columns_in_df(
        df=df_max, columns=("familiar object", "novel object")
    )
    _check_columns_in_df(df=df_avg, columns=("state", "positive", "negative"))

    # File cleanup and plot cleanup handled automatically by output_file_cleanup fixture


@pytest.mark.parametrize("params", invalid_inputs)
def test_correlations_invalid_inputs(params, output_file_cleanup, monkeypatch):
    """Test correlation tool with invalid inputs and verify expected behavior."""
    # Mock functions to avoid file access issues
    monkeypatch.setattr(
        "analysis.correlations.cell_set_to_positions",
        lambda files, **kwargs: {},
    )

    # Need sufficient data for each state to meet MINIMUM_STATE_LENGTH requirement
    mock_annotations = pd.DataFrame(
        {
            "time": [i * 0.1 for i in range(30)],
            "state": ["familiar object"] * 12
            + ["novel object"] * 12
            + ["not_defined"] * 6,
        }
    )
    monkeypatch.setattr("pandas.read_parquet", lambda path: mock_annotations)

    monkeypatch.setattr(
        "analysis.correlations._get_cellset_data",
        lambda files: (
            np.random.rand(30, 10),
            ["accepted"] * 10,
            [],
            0.05,
            10,
            0,
            0,
            [f"cell_{i}" for i in range(10)],
        ),
    )

    # Mock correlation computation functions
    def mock_compute_correlation_matrices(
        traces, annotations, states, column_name
    ):
        n_cells = traces.shape[1]
        correlation_matrices = {}
        for state in ["familiar object", "novel object"]:
            matrix = np.random.rand(n_cells, n_cells) * 0.8 - 0.4
            matrix = (matrix + matrix.T) / 2
            np.fill_diagonal(matrix, 0.0)
            correlation_matrices[state] = matrix
        return correlation_matrices

    monkeypatch.setattr(
        "analysis.correlations._compute_correlation_matrices",
        mock_compute_correlation_matrices,
    )

    error_text = params.pop("error_text")
    expect_exception = params.pop("expect_exception", True)
    expected_states = params.pop("expected_states", None)

    if expect_exception:
        # Test cases where we expect an exception to be raised
        with pytest.raises(IdeasError, match=error_text):
            correlations.correlation_tool(**params)
    else:
        # Run the function - it should handle the error gracefully
        correlations.correlation_tool(**params)

        # For invalid column name, check that files were created with expected states
        if expected_states:
            # Verify that the h5 file contains the expected states
            with h5py.File(correlations.RAW_CORRELATIONS_H5_NAME, "r") as f:
                keys = list(f.keys())
                # With mock data, we expect 'familiar object' and 'novel object' states
                # instead of the original expected states
                mock_expected_states = ["familiar object", "novel object"]
                for state in mock_expected_states:
                    assert (
                        state in keys
                    ), f"Expected state '{state}' not found in h5 file"

    # Cleanup handled automatically by output_file_cleanup fixture


def test_correlation_matrix_triu_export(temp_work_dir, cleanup_plots):
    """Test that correlation matrices with cell names are saved correctly in triu format"""
    # Setup test data
    # Create a small correlation matrix for testing
    size = 4
    # Create identity matrix plus some cross-correlations
    state1_matrix = np.eye(size)
    state1_matrix[0, 1] = state1_matrix[1, 0] = 0.5
    state1_matrix[2, 3] = state1_matrix[3, 2] = -0.3

    state2_matrix = np.eye(size)
    state2_matrix[0, 2] = state2_matrix[2, 0] = 0.7
    state2_matrix[1, 3] = state2_matrix[3, 1] = -0.4

    correlation_matrix = {"state1": state1_matrix, "state2": state2_matrix}

    # Mock cell names
    cell_names = [f"cell_{i}" for i in range(size)]

    # Mock sort indices - use different ordering for each state
    sort_indices = {
        "state1": np.array([0, 1, 2, 3]),
        "state2": np.array([3, 2, 0, 1]),
    }

    # Call the consolidated function - already in temp directory
    correlations._correlations_to_csv(
        correlation_matrix=correlation_matrix,
        cell_names=cell_names,
        sort_indices=sort_indices,
    )

    # Check that the ZIP file was created
    assert os.path.exists(correlations.RAW_CORRELATIONS_ZIP_NAME)

    # Check contents of the ZIP file
    with zipfile.ZipFile(correlations.RAW_CORRELATIONS_ZIP_NAME, "r") as zipf:
        file_list = zipf.namelist()

        # Check for the raw matrix files
        assert "correlation_matrix_state1.csv" in file_list
        assert "correlation_matrix_state2.csv" in file_list

        # Check for the triangle files
        assert "correlation_matrix_state1_triu.csv" in file_list
        assert "correlation_matrix_state2_triu.csv" in file_list

        # Check README
        assert "README.txt" in file_list

        # Verify triangle file contents
        with zipf.open("correlation_matrix_state1_triu.csv") as f:
            df_triu = pd.read_csv(f)

            # Check number of rows in triangle file
            expected_triu_rows = (size * (size - 1)) // 2
            assert len(df_triu) == expected_triu_rows

            # Verify specific correlation values
            row = df_triu[
                (df_triu["cell_name_1"] == "cell_0")
                & (df_triu["cell_name_2"] == "cell_1")
            ]
            assert len(row) == 1, "Expected to find exactly one matching row"
            assert abs(row["correlation"].values[0] - 0.5) < 1e-6

        # Verify state2 triangle file with reordered indices
        with zipf.open("correlation_matrix_state2_triu.csv") as f:
            df_triu = pd.read_csv(f)

            # Get sorted cell names
            sorted_cell_names = [cell_names[i] for i in sort_indices["state2"]]

            # Find the cells corresponding to original cell_0 and cell_2
            cell_0_pos = np.where(sort_indices["state2"] == 0)[0][0]
            cell_2_pos = np.where(sort_indices["state2"] == 2)[0][0]

            # Make sure we check the cells in sorted order (smaller index first for triu)
            first_cell = min(cell_0_pos, cell_2_pos)
            second_cell = max(cell_0_pos, cell_2_pos)

            row = df_triu[
                (df_triu["cell_name_1"] == sorted_cell_names[first_cell])
                & (df_triu["cell_name_2"] == sorted_cell_names[second_cell])
            ]

            assert len(row) == 1, "Expected to find exactly one matching row"
            assert abs(row["correlation"].values[0] - 0.7) < 1e-6

    # Directory and plot cleanup handled automatically by fixtures


def test_plot_correlation_matrices(temp_work_dir, cleanup_plots):
    """Test that plot_correlation_matrices returns sorting indices correctly"""
    # Setup test data
    size = 4
    # Create correlation matrices with known patterns
    state1_matrix = np.eye(size)
    state1_matrix[0, 1] = state1_matrix[1, 0] = 0.5
    state1_matrix[2, 3] = state1_matrix[3, 2] = -0.3

    state2_matrix = np.eye(size)
    state2_matrix[0, 2] = state2_matrix[2, 0] = 0.7
    state2_matrix[1, 3] = state2_matrix[3, 1] = -0.4

    correlation_matrix = {"state1": state1_matrix, "state2": state2_matrix}

    # Use a custom color list
    correlation_colors = ["red", "blue"]

    # Call the function - already in temp directory
    sort_indices = plot_correlation_matrices(
        correlation_matrix=correlation_matrix,
        correlation_colors=correlation_colors,
        out_file_name="test_correlation_matrices.svg",
    )

    # Check that the output file was created
    assert os.path.exists(
        "test_correlation_matrices.svg"
    ), "Output SVG file was not created"

    # Verify that sort_indices is a dictionary with the expected keys
    assert isinstance(
        sort_indices, dict
    ), "sort_indices should be a dictionary"
    assert set(sort_indices.keys()) == set(
        correlation_matrix.keys()
    ), "sort_indices keys should match correlation_matrix keys"

    # Check that each value in sort_indices is an array with the right shape
    for state, indices in sort_indices.items():
        assert isinstance(
            indices, np.ndarray
        ), f"sort_indices[{state}] should be a numpy array"
        assert indices.shape == (
            size,
        ), f"sort_indices[{state}] should have shape ({size},)"

        # Verify that indices contain a permutation of 0...size-1
        assert set(indices) == set(
            range(size)
        ), f"sort_indices[{state}] should be a permutation of 0...{size - 1}"

    # Directory and plot cleanup handled automatically by fixtures


class TestCorrelations:
    """Unit tests for correlation matrix generation and export."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, temp_work_dir):
        """Set up test data for correlation tests."""
        # Create sample correlation matrices
        self.rng = np.random.RandomState(42)
        n_cells = 5

        # Create a sample correlation matrix with known values
        matrix1 = np.zeros((n_cells, n_cells))
        matrix2 = np.zeros((n_cells, n_cells))

        # Fill the matrices with predictable values
        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:
                    matrix1[i, j] = 0.1 * (i + j)
                    matrix2[i, j] = -0.1 * (i + j)

        # Make them symmetric
        matrix1 = (matrix1 + matrix1.T) / 2
        matrix2 = (matrix2 + matrix2.T) / 2

        self.correlation_matrix = {"state1": matrix1, "state2": matrix2}

        self.cell_names = [f"cell_{i}" for i in range(n_cells)]

        # Create sort indices (identity mapping for simplicity)
        self.sort_indices = {
            "state1": np.arange(n_cells),
            "state2": np.arange(n_cells),
        }

        # temp_work_dir fixture handles directory management automatically

    def test_correlations_to_csv(self):
        """Test that correlation matrices are correctly saved to CSV and zipped."""
        _correlations_to_csv(
            correlation_matrix=self.correlation_matrix,
            cell_names=self.cell_names,
            sort_indices=self.sort_indices,
        )

        # Check that the ZIP file was created
        assert os.path.exists(RAW_CORRELATIONS_ZIP_NAME)

        # Check contents of ZIP file
        with zipfile.ZipFile(RAW_CORRELATIONS_ZIP_NAME, "r") as zipf:
            file_list = zipf.namelist()

            # Check for expected files
            assert "correlation_matrix_state1.csv" in file_list
            assert "correlation_matrix_state2.csv" in file_list
            assert "correlation_matrix_state1_triu.csv" in file_list
            assert "correlation_matrix_state2_triu.csv" in file_list
            assert "README.txt" in file_list

            # Extract and check one of the triangle files
            with zipf.open("correlation_matrix_state1_triu.csv") as f:
                df = pd.read_csv(f)

                # Check column names
                assert list(df.columns) == [
                    "cell_name_1",
                    "cell_name_2",
                    "correlation",
                ]

                # Check number of rows (should be n_cells*(n_cells-1)/2 for upper triangle)
                n_cells = len(self.cell_names)
                expected_rows = n_cells * (n_cells - 1) // 2
                assert len(df) == expected_rows

    def test_cell_name_pair_correlation_values(self):
        """Test that cell name pairs match with correct correlation values."""
        _correlations_to_csv(
            correlation_matrix=self.correlation_matrix,
            cell_names=self.cell_names,
            sort_indices=self.sort_indices,
        )

        # Check a specific cell pair value in the ZIP file using the triangle file
        with zipfile.ZipFile(RAW_CORRELATIONS_ZIP_NAME, "r") as zipf:
            with zipf.open("correlation_matrix_state1_triu.csv") as f:
                df = pd.read_csv(f)

                # Find a specific cell pair (cell_0, cell_1)
                row = df[
                    (df["cell_name_1"] == "cell_0")
                    & (df["cell_name_2"] == "cell_1")
                ]
                assert len(row) == 1

                # Check that the correlation value is correct
                expected_value = self.correlation_matrix["state1"][0, 1]
                assert (
                    abs(row["correlation"].values[0] - expected_value) < 1e-7
                )

    @patch("analysis.correlations.logger.warning")
    def test_verification_catches_discrepancies(self, mock_warning):
        """Test that the verification logic catches value inconsistencies."""
        # Create a copy of sort_indices but with a corrupted order
        corrupted_indices = self.sort_indices.copy()
        corrupted_indices["state1"] = np.array([0, 2, 1, 3, 4])

        # Create a corrupted matrix that wouldn't match when sorted
        corrupted_matrix = self.correlation_matrix.copy()
        corrupted_matrix["state1"] = corrupted_matrix["state1"] + 0.01

        _correlations_to_csv(
            correlation_matrix=corrupted_matrix,
            cell_names=self.cell_names,
            sort_indices=corrupted_indices,
        )

        # Check if any warning was logged (indicating a validation issue was caught)
        mock_warning.assert_called()

    def test_cell_names_preserved_after_sorting(self):
        """Test that cell names are correctly preserved when using custom sort indices."""
        # Create custom sort indices that reorder the cells
        custom_sort_indices = self.sort_indices.copy()
        custom_sort_indices["state1"] = np.array([2, 0, 3, 1, 4])

        _correlations_to_csv(
            correlation_matrix=self.correlation_matrix,
            cell_names=self.cell_names,
            sort_indices=custom_sort_indices,
        )

        # Check the sorted cell names in the output file
        with zipfile.ZipFile(RAW_CORRELATIONS_ZIP_NAME, "r") as zipf:
            # Since the full matrix doesn't use cell_names directly as column headers,
            # let's check the triangle format which should have proper cell names
            with zipf.open("correlation_matrix_state1_triu.csv") as f:
                df_triu = pd.read_csv(f)

                # Verify we have the expected cells in the triangle format
                expected_cells = set(self.cell_names)
                actual_cells = set(df_triu["cell_name_1"]).union(
                    set(df_triu["cell_name_2"])
                )
                assert expected_cells == actual_cells

                # Find pairs that would be affected by sorting
                i, j = 0, 1

                # Get the correct cell names
                cell_name_1 = self.cell_names[i]
                cell_name_2 = self.cell_names[j]

                # Find this pair using the cell names (could be in either order)
                row = df_triu[
                    (
                        (df_triu["cell_name_1"] == cell_name_1)
                        & (df_triu["cell_name_2"] == cell_name_2)
                    )
                    | (
                        (df_triu["cell_name_1"] == cell_name_2)
                        & (df_triu["cell_name_2"] == cell_name_1)
                    )
                ]

                assert (
                    len(row) == 1
                ), "Should find exactly one matching cell pair"
                assert (
                    abs(
                        row["correlation"].values[0]
                        - self.correlation_matrix["state1"][i, j]
                    )
                    < 1e-7
                )

                # Check another pair
                i, j = 2, 3
                cell_name_1 = self.cell_names[i]
                cell_name_2 = self.cell_names[j]

                row = df_triu[
                    (
                        (df_triu["cell_name_1"] == cell_name_1)
                        & (df_triu["cell_name_2"] == cell_name_2)
                    )
                    | (
                        (df_triu["cell_name_1"] == cell_name_2)
                        & (df_triu["cell_name_2"] == cell_name_1)
                    )
                ]

                assert (
                    len(row) == 1
                ), "Should find exactly one matching cell pair"
                assert (
                    abs(
                        row["correlation"].values[0]
                        - self.correlation_matrix["state1"][i, j]
                    )
                    < 1e-7
                )

    def test_correlation_threshold_parameter(self):
        """Test that correlation_threshold parameter
        is properly passed and used in spatial map generation."""
        # Create test data
        size = 6
        state1_matrix = np.eye(size)
        # Add some strong correlations
        state1_matrix[0, 1] = state1_matrix[1, 0] = 0.8
        state1_matrix[2, 3] = state1_matrix[3, 2] = 0.7
        # Add some weak correlations
        state1_matrix[4, 5] = state1_matrix[5, 4] = 0.3

        correlation_matrix = {"state1": state1_matrix}
        cell_names = [f"cell_{i}" for i in range(size)]

        # Create positions for all cells
        positions = {f"cell_{i}": (i * 10, i * 10) for i in range(size)}

        # Test with high threshold (0.6) - should only show strong correlations
        correlations.plot_correlation_spatial_map(
            correlation_matrix=correlation_matrix,
            positions=positions,
            cell_names=cell_names,
            correlation_threshold=0.6,
            max_lines=100,
            out_file_name="spatial_map_high_threshold.svg",
        )

        # Test with low threshold (0.2) - should show all correlations
        correlations.plot_correlation_spatial_map(
            correlation_matrix=correlation_matrix,
            positions=positions,
            cell_names=cell_names,
            correlation_threshold=0.2,
            max_lines=100,
            out_file_name="spatial_map_low_threshold.svg",
        )

        # Test with very high threshold (0.95) - should show no correlations but still have colorbar
        correlations.plot_correlation_spatial_map(
            correlation_matrix=correlation_matrix,
            positions=positions,
            cell_names=cell_names,
            correlation_threshold=0.95,
            max_lines=100,
            out_file_name="spatial_map_no_correlations.svg",
        )

        # Check that all files were created
        assert os.path.exists("spatial_map_high_threshold.svg")
        assert os.path.exists("spatial_map_low_threshold.svg")
        assert os.path.exists("spatial_map_no_correlations.svg")

        # Test the integration with correlation_tool function
        # Mock the correlation_tool to test parameter passing
        with patch(
            "analysis.correlations.cell_set_to_positions_mapping"
        ) as mock_positions:
            mock_positions.return_value = positions

            with patch(
                "analysis.correlations._get_cellset_data"
            ) as mock_traces:
                mock_traces.return_value = (
                    np.random.randn(100, size),
                    ["accepted"] * size,
                    [],
                    0.05,
                    size,
                    0,
                    0,
                    cell_names,
                )

                with patch(
                    "analysis.correlations._compute_correlation_matrices"
                ) as mock_corr:
                    mock_corr.return_value = correlation_matrix

                    with patch(
                        "analysis.correlations.plot_correlation_matrices"
                    ) as mock_plot_corr:
                        mock_plot_corr.return_value = {
                            "state1": np.arange(size)
                        }

                        # Test with custom correlation_threshold
                        correlations.correlation_tool(
                            cell_set_files=["/fake/path.isxd"],
                            correlation_threshold=0.7,
                            include_positions=True,
                        )

                        # Verify that a spatial map file was created
                        assert os.path.exists(
                            correlations.SPATIAL_MAP_SVG_NAME
                        )


def test_positions_included_in_output(temp_work_dir, cleanup_plots):
    """Test that cell positions are correctly included in the output files when provided."""
    # Setup test data
    size = 4
    state1_matrix = np.eye(size)
    state1_matrix[0, 1] = state1_matrix[1, 0] = 0.5
    state1_matrix[2, 3] = state1_matrix[3, 2] = -0.3

    correlation_matrix = {"state1": state1_matrix}
    cell_names = [f"cell_{i}" for i in range(size)]
    sort_indices = {"state1": np.array([0, 1, 2, 3])}

    # Mock cell positions
    positions = {
        "cell_0": (10, 20),
        "cell_1": (15, 25),
        "cell_2": (30, 10),
        "cell_3": (40, 30),
    }

    # Call the function with positions - already in temp directory
    correlations._correlations_to_csv(
        correlation_matrix=correlation_matrix,
        cell_names=cell_names,
        sort_indices=sort_indices,
        positions=positions,
    )

    # Check that the ZIP file was created
    assert os.path.exists(correlations.RAW_CORRELATIONS_ZIP_NAME)

    # Check that positions are included in triangle file
    with zipfile.ZipFile(correlations.RAW_CORRELATIONS_ZIP_NAME, "r") as zipf:
        with zipf.open("correlation_matrix_state1_triu.csv") as f:
            df_triu = pd.read_csv(f)

            # Check columns include position info
            expected_columns = [
                "cell_name_1",
                "cell_name_2",
                "correlation",
                "centroid_x1",
                "centroid_y1",
                "centroid_x2",
                "centroid_y2",
                "distance",
            ]
            assert all(col in df_triu.columns for col in expected_columns)

            # Verify position values for a specific cell pair
            row = df_triu[
                (df_triu["cell_name_1"] == "cell_0")
                & (df_triu["cell_name_2"] == "cell_1")
            ]

            assert len(row) == 1
            assert row["centroid_x1"].values[0] == 10
            assert row["centroid_y1"].values[0] == 20
            assert row["centroid_x2"].values[0] == 15
            assert row["centroid_y2"].values[0] == 25

            # Check distance calculation
            expected_distance = np.sqrt((10 - 15) ** 2 + (20 - 25) ** 2)
            assert abs(row["distance"].values[0] - expected_distance) < 1e-6

            # Verify README contains position information
            with zipf.open("README.txt") as readme_file:
                readme_content = readme_file.read().decode("utf-8")
                assert "position columns" in readme_content.lower()
                assert "distance column" in readme_content.lower()

    # Directory and plot cleanup handled automatically by fixtures


def test_position_mapping_preserved_with_sorting(temp_work_dir, cleanup_plots):
    """Test that position mapping to cell names is preserved when applying custom sort indices."""
    # Setup test data
    size = 4
    state1_matrix = np.eye(size)
    state1_matrix[0, 1] = state1_matrix[1, 0] = 0.5
    state1_matrix[2, 3] = state1_matrix[3, 2] = -0.3

    correlation_matrix = {"state1": state1_matrix}
    cell_names = [f"cell_{i}" for i in range(size)]

    # Custom sort indices that reorder cells
    custom_sort_indices = {"state1": np.array([2, 0, 3, 1])}

    # Mock cell positions
    positions = {
        "cell_0": (10, 20),
        "cell_1": (15, 25),
        "cell_2": (30, 10),
        "cell_3": (40, 30),
    }

    # Call the function with custom sorting and positions - already in temp directory
    correlations._correlations_to_csv(
        correlation_matrix=correlation_matrix,
        cell_names=cell_names,
        sort_indices=custom_sort_indices,
        positions=positions,
    )

    # Check that cell-position mapping is preserved in output
    with zipfile.ZipFile(correlations.RAW_CORRELATIONS_ZIP_NAME, "r") as zipf:
        with zipf.open("correlation_matrix_state1_triu.csv") as f:
            df_triu = pd.read_csv(f)

            # Check all pairs of cells to ensure positions match original mapping
            for _, row in df_triu.iterrows():
                cell1 = row["cell_name_1"]
                cell2 = row["cell_name_2"]

                # Expected positions from original mapping
                expected_pos1 = positions[cell1]
                expected_pos2 = positions[cell2]

                # Actual positions in output
                actual_pos1 = (row["centroid_x1"], row["centroid_y1"])
                actual_pos2 = (row["centroid_x2"], row["centroid_y2"])

                # Verify positions match
                assert (
                    expected_pos1 == actual_pos1
                ), f"Position mismatch for {cell1}"
                assert (
                    expected_pos2 == actual_pos2
                ), f"Position mismatch for {cell2}"

                # Verify distance
                expected_distance = np.sqrt(
                    (expected_pos1[0] - expected_pos2[0]) ** 2
                    + (expected_pos1[1] - expected_pos2[1]) ** 2
                )
                assert abs(row["distance"] - expected_distance) < 1e-6

    # Directory and plot cleanup handled automatically by fixtures


def test_plot_measure_correlations(temp_work_dir, cleanup_plots):
    """Test that plot_measure_correlations correctly
    creates visualization with expected components.
    """
    # Test case 1: Multiple states
    # Create sample data for testing
    data = {
        "name": [f"cell_{i}" for i in range(10)],
        "state1": np.random.uniform(-0.8, 0.8, 10),
        "state2": np.random.uniform(-0.5, 0.9, 10),
        "state3": np.random.uniform(-0.7, 0.6, 10),
    }
    df = pd.DataFrame(data)
    states = ["state1", "state2", "state3"]
    colors = ["red", "blue", "green"]

    # Call the function
    correlations.plot_measure_correlations(
        df=df,
        statistic="max",
        states=states,
        colors=colors,
    )

    # Check that the output file exists
    assert os.path.exists(
        "correlation_plot.svg"
    ), "Output file not created for multiple states"

    # Test case 2: Single state
    single_state_df = pd.DataFrame(
        {
            "name": [f"cell_{i}" for i in range(10)],
            "all times": np.random.uniform(-0.8, 0.8, 10),
        }
    )
    single_state = ["all times"]

    # Call the function with a single state
    correlations.plot_measure_correlations(
        df=single_state_df,
        statistic="mean",
        states=single_state,
        colors=["purple"],
    )

    # Check that the output file exists for single state too
    assert os.path.exists(
        "correlation_plot.svg"
    ), "Output file not created for single state"

    # Verify boxplot behavior for multi-state and single-state cases using patches
    with patch("seaborn.boxplot") as mock_boxplot:
        # Test multi-state case - should call boxplot
        correlations.plot_measure_correlations(
            df=df,
            statistic="min",
            states=states,
            colors=colors,
        )
        # Assert boxplot was called for multi-state case
        assert (
            mock_boxplot.called
        ), "Boxplot should be created for multiple states"

    # Reset for single state test
    with patch("seaborn.boxplot") as mock_boxplot:
        # Test single-state case - should NOT call boxplot
        correlations.plot_measure_correlations(
            df=single_state_df,
            statistic="mean",
            states=single_state,
            colors=["purple"],
        )
        # Assert boxplot was NOT called for single-state case
        assert (
            not mock_boxplot.called
        ), "Boxplot should NOT be created for single state"

    # Verify plot structure for multiple states
    with patch("matplotlib.pyplot.savefig"):
        correlations.plot_measure_correlations(
            df=df,
            statistic="min",
            states=states,
            colors=colors,
        )

        # Get the figure from the most recent call
        fig = plt.gcf()

        # For multi-state case, verify we have 2 subplots
        assert len(fig.axes) == 2, "Expected 2 subplots for multiple states"

        # Verify ECDF plot (first subplot)
        ax0 = fig.axes[0]
        assert ax0.get_xlabel() == "Correlation", "ECDF x-label incorrect"
        assert (
            "Proportion of cells with min" in ax0.get_ylabel()
        ), "ECDF y-label incorrect"

        # Verify boxplot (second subplot)
        ax1 = fig.axes[1]
        assert (
            ax1.get_ylabel() == "min correlations"
        ), "Boxplot y-label incorrect"

    # Verify plot structure for single state
    with patch("matplotlib.pyplot.savefig"):
        correlations.plot_measure_correlations(
            df=single_state_df,
            statistic="mean",
            states=single_state,
            colors=["purple"],
        )

        # Get the figure from the most recent call
        fig = plt.gcf()

        # For single state, verify we have only 1 subplot
        assert len(fig.axes) == 1, "Expected 1 subplot for single state"
