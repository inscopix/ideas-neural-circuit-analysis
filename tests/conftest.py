"""Shared pytest fixtures for toolbox tests.

This module provides reusable fixtures for common test cleanup operations
including file cleanup, directory management, and plot cleanup.
"""

import os
import tempfile
import shutil
from typing import List, Union
import pytest
import matplotlib
import matplotlib.pyplot as plt


@pytest.fixture
def cleanup_files():
    """Fixture to track and clean up files created during tests.

    Returns a function that can be called to register files for cleanup.
    All registered files will be automatically cleaned up after the test.

    Usage:
        def test_something(cleanup_files):
            register_cleanup = cleanup_files
            # Create some files
            register_cleanup("output.csv")
            register_cleanup(["file1.svg", "file2.json"])
            # Files will be automatically cleaned up after test
    """
    files_to_cleanup = []

    def register_cleanup(files: Union[str, List[str]]):
        """Register files for cleanup after test completion."""
        if isinstance(files, str):
            files_to_cleanup.append(files)
        else:
            files_to_cleanup.extend(files)

    yield register_cleanup

    # Cleanup phase
    for file_path in files_to_cleanup:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except (OSError, PermissionError):
            # Ignore cleanup errors - test already completed
            pass


@pytest.fixture
def temp_work_dir():
    """Fixture that creates a temporary working directory and changes to it.

    Automatically restores the original working directory after the test.
    The temporary directory is automatically cleaned up.

    Usage:
        def test_something(temp_work_dir):
            # Already in a temporary directory
            # Create files, they'll be cleaned up automatically
            pass
    """
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp(prefix="toolbox_test_")

    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        # Restore original directory
        os.chdir(original_dir)
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except (OSError, PermissionError):
            # Ignore cleanup errors
            pass


@pytest.fixture
def cleanup_plots():
    """Fixture to ensure matplotlib plots are properly closed after tests.

    This prevents memory leaks and interference between tests that create plots.

    Usage:
        def test_plotting(cleanup_plots):
            plt.figure()
            plt.plot([1, 2, 3])
            # Plot will be automatically closed after test
    """
    yield
    # Close all matplotlib figures
    plt.close("all")
    # Clear the matplotlib backend state
    matplotlib.pyplot.clf()


@pytest.fixture
def comprehensive_cleanup(cleanup_files, temp_work_dir, cleanup_plots):
    """Create a combined fixture that provides comprehensive cleanup for tests.

    Combines file cleanup, temporary directory management, and plot cleanup
    into a single convenient fixture.

    Returns
    -------
    tuple
        (register_cleanup_function, temp_directory_path)

    Usage:
        def test_complex_operation(comprehensive_cleanup):
            register_cleanup, temp_dir = comprehensive_cleanup
            # Work in temporary directory with automatic cleanup
            register_cleanup("additional_file.txt")
            plt.figure()  # Will be cleaned up
            # Everything cleaned up automatically

    """
    yield cleanup_files, temp_work_dir


@pytest.fixture
def output_file_cleanup():
    """Fixture specifically for toolbox output files that are commonly created.

    Automatically cleans up common output files that toolbox functions create.

    Usage:
        def test_toolbox_function(output_file_cleanup):
            # Run toolbox function that creates standard output files
            toolbox.some_function()
            # Standard output files will be cleaned up automatically
    """
    # Common output files created by toolbox functions
    common_output_files = [
        # Population activity outputs
        "activity_average_preview.svg",
        "trace_population_average_preview.svg",
        "modulation_histogram_preview.svg",
        "trace_modulation_histogram_preview.svg",
        "modulation_preview.svg",
        "time_in_state_preview.svg",
        "time_in_state_epoch_preview.svg",
        "trace_preview.svg",
        "trace_modulation_footprint_preview.svg",
        "event_population_average_preview.svg",
        "event_modulation_preview.svg",
        "event_modulation_histogram_preview.svg",
        "event_preview.svg",
        "trace_population_data.csv",
        "event_population_data.csv",
        # Correlation outputs
        "correlation_statistic_comparison.csv",
        "pairwise_correlation_heatmaps.h5",
        "average_correlations.csv",
        "average_correlations_preview.svg",
        "correlation_matrices.svg",
        "correlation_plot.svg",
        "correlation_spatial_map.svg",
        "spatial_analysis_pairwise_correlations.zip",
        # Trace-only correlation outputs
        "correlations_per_state_epoch_data.csv",
        "average_correlations.csv",
        "trace_average_correlations_preview.svg",
        "trace_correlation_matrices_preview.svg",
        "trace_correlation_statistic_distribution_preview.svg",
        "trace_spatial_correlation_preview.svg",
        "trace_spatial_correlation_map_preview.svg",
        # Generic common outputs
        "output.csv",
        "output.svg",
        "output.png",
        "output.json",
        "output.h5",
        "output_metadata.json",
        "output_data.json",
        "exit_status.txt",
    ]

    yield

    # Cleanup phase
    for file_path in common_output_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except (OSError, PermissionError):
            # Ignore cleanup errors
            pass


@pytest.fixture
def preserve_working_directory():
    """Fixture to preserve the current working directory.

    Some tests need to change directories but should restore the original
    directory afterward. This fixture ensures the directory is always restored.

    Usage:
        def test_with_directory_change(preserve_working_directory):
            os.chdir("/some/other/directory")
            # Original directory will be restored automatically
    """
    original_dir = os.getcwd()
    yield
    os.chdir(original_dir)


@pytest.fixture(scope="function")
def isolated_test_environment(
    cleanup_files, temp_work_dir, cleanup_plots, output_file_cleanup
):
    """Complete isolation fixture for tests that need full cleanup.

    Provides a completely isolated test environment with:
    - Temporary working directory
    - File cleanup registration
    - Plot cleanup
    - Common output file cleanup

    This is the most comprehensive fixture and should be used for integration
    tests or tests that create many different types of outputs.

    Usage:
        def test_full_workflow(isolated_test_environment):
            register_cleanup, temp_dir = isolated_test_environment
            # Run complex operations with complete isolation
            # Everything is cleaned up automatically
    """
    yield cleanup_files, temp_work_dir


@pytest.fixture
def mock_matplotlib_close():
    """Fixture that mocks matplotlib.pyplot.close to prevent actual plot closing.

    Useful for tests where you want to verify plot creation without actually
    displaying or closing plots.

    Usage:
        def test_plot_creation(mock_matplotlib_close):
            # matplotlib.pyplot.close calls are mocked
            create_plot()  # Won't actually close plots
    """
    with pytest.MonkeyPatch().context() as m:
        m.setattr("matplotlib.pyplot.close", lambda *args, **kwargs: None)
        yield


class CleanupHelper:
    """Helper class for manual cleanup operations.

    Provides utility methods for common cleanup tasks that can be used
    in tests that need more control over when cleanup occurs.
    """

    @staticmethod
    def remove_files(*file_paths: str) -> None:
        """Remove multiple files, ignoring errors."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except (OSError, PermissionError):
                pass

    @staticmethod
    def remove_directories(*dir_paths: str) -> None:
        """Remove multiple directories recursively, ignoring errors."""
        for dir_path in dir_paths:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            except (OSError, PermissionError):
                pass

    @staticmethod
    def clean_matplotlib() -> None:
        """Clean up matplotlib state."""
        plt.close("all")
        matplotlib.pyplot.clf()

    @staticmethod
    def clean_working_directory(original_dir: str) -> None:
        """Restore working directory to original location."""
        try:
            os.chdir(original_dir)
        except (OSError, FileNotFoundError):
            pass


@pytest.fixture
def cleanup_helper():
    """Fixture that provides the CleanupHelper class for manual cleanup operations."""
    return CleanupHelper
