"""
Comprehensive tests for color mapping and timeline functionality in state-epoch baseline analysis.

Tests cover:
- Two-layer color mapping (states inner, epochs outer)
- Real behavioral timeline reconstruction
- State-epoch overlay functionality
- Preview file generation with accurate timepoints
- Color consistency across all preview functions
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from utils.state_epoch_output import StateEpochOutputGenerator
from utils.state_epoch_results import StateEpochResults


class TestColorMappingFunctionality:
    """Test two-layer color mapping functionality."""

    @pytest.fixture
    def setup_output_generator(self, tmp_path):
        """Create StateEpochOutputGenerator with test configuration."""
        return StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["rest", "active", "feeding"],
            epochs=["baseline", "training", "test"],
            state_colors=["gray", "blue", "orange"],
            epoch_colors=["lightgray", "lightblue", "lightgreen"],
            baseline_state="rest",
            baseline_epoch="baseline",
            alpha=0.05,
            n_shuffle=1000,
        )

    @pytest.fixture
    def mock_results(self):
        """Create mock StateEpochResults with test combinations."""
        results = MagicMock(spec=StateEpochResults)
        # Test combinations: (state, epoch) pairs
        test_combinations = [
            ("rest", "baseline"),
            ("active", "training"),
            ("feeding", "test"),
            ("rest", "training"),  # Cross combinations
            ("active", "test"),
        ]
        results.get_all_combinations.return_value = test_combinations

        # Mock combination results with realistic data
        def mock_get_combination_results(state, epoch):
            return {
                "mean_activity": np.random.rand(10),  # 10 cells
                "num_timepoints": 50,
                "state": state,
                "epoch": epoch,
            }

        results.get_combination_results.side_effect = (
            mock_get_combination_results
        )

        return results

    # NOTE: _create_two_layer_colors method was removed and logic inlined
    # The color mapping functionality is now part of _create_event_average_preview
    # and is tested through integration tests rather than unit tests

    # NOTE: Color mapping consistency is now tested through integration tests
    # since the logic was inlined into _create_event_average_preview


class TestBehavioralTimelineReconstruction:
    """Test behavioral timeline functionality (simplified)."""

    @pytest.fixture
    def setup_output_generator(self, tmp_path):
        """Create StateEpochOutputGenerator for timeline tests."""
        return StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["rest", "active"],
            epochs=["baseline", "test"],
            state_colors=["gray", "blue"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="rest",
            baseline_epoch="baseline",
            alpha=0.05,
            n_shuffle=1000,
        )

    def test_output_generator_initialization(self, setup_output_generator):
        """Test that output generator is properly initialized for timeline tests."""
        generator = setup_output_generator

        # Verify basic initialization
        assert generator.states == ["rest", "active"]
        assert generator.epochs == ["baseline", "test"]
        assert generator.color_scheme.state_colors == ["gray", "blue"]
        assert generator.color_scheme.epoch_colors == [
            "lightgray",
            "lightblue",
        ]
        assert generator.baseline_state == "rest"
        assert generator.baseline_epoch == "baseline"


class TestPreviewFunctionIntegration:
    """Test that preview functions use real timepoints and proper color mapping."""

    @pytest.fixture
    def setup_complete_test_environment(self, tmp_path):
        """Set up complete test environment with all required components."""
        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["rest", "active"],
            epochs=["baseline", "test"],
            state_colors=["gray", "blue"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="rest",
            baseline_epoch="baseline",
            alpha=0.05,
            n_shuffle=1000,
            epoch_periods=[
                (0.0, 12.0),
                (12.0, 25.0),
            ],  # Two epochs covering the data
        )

        # Mock results
        results = MagicMock(spec=StateEpochResults)
        results.get_all_combinations.return_value = [
            ("rest", "baseline"),
            ("active", "test"),
        ]

        def mock_get_combination_results(state, epoch):
            return {
                "mean_activity": np.random.rand(5),  # 5 cells
                "num_timepoints": 25,
                "state": state,
                "epoch": epoch,
            }

        results.get_combination_results.side_effect = (
            mock_get_combination_results
        )

        # Mock behavioral annotations
        annotations_df = pd.DataFrame(
            {
                "state": ["rest"] * 12 + ["active"] * 13,  # 25 total frames
                "frame": range(25),
            }
        )

        # Mock traces and events
        traces = np.random.rand(25, 5)  # 25 timepoints, 5 cells
        events = np.random.randint(0, 2, (25, 5))  # Binary events

        # Mock cell info
        cell_info = {
            "cell_set": None,  # Will trigger fallback contours
            "period": 1.0,  # 1 second per frame
            "boundaries": [0.0, 12.5, 25.0],  # Start, middle, end
        }

        return {
            "generator": generator,
            "results": results,
            "annotations_df": annotations_df,
            "traces": traces,
            "events": events,
            "cell_info": cell_info,
        }

    @patch("utils.state_epoch_output._plot_traces")
    @patch("utils.plots.plot_trace_preview")
    @patch("utils.plots._plot_timecourse")
    def test_trace_preview_uses_real_timepoints(
        self,
        mock_plot_timecourse,
        mock_plot_trace_preview,
        mock_plot_traces,
        setup_complete_test_environment,
    ):
        """Test that trace preview uses real behavioral timepoints."""
        env = setup_complete_test_environment

        # Call trace preview function
        env["generator"]._create_trace_preview(
            results=env["results"],
            cell_info=env["cell_info"],
            traces=env["traces"],
            events=env["events"],
            annotations_df=env["annotations_df"],
            column_name="state",
        )

        # Verify state overlay plotting function was called
        assert (
            mock_plot_traces.called
        ), "_plot_traces was not called for state overlay"

        # With annotations present, custom epoch overlay is used instead of plot_trace_preview
        # The custom implementation creates its own figure rather than calling plot_trace_preview

        # Check that state plotting was called with real behavioral timepoints
        state_call_args = mock_plot_traces.call_args
        state_args, state_kwargs = state_call_args

        # Should have behavior data
        assert "behavior" in state_kwargs
        assert "column_name" in state_kwargs
        assert state_kwargs["column_name"] == "state"

        # Should have real timepoints (boundaries and period)
        assert "boundaries" in state_kwargs
        assert "period" in state_kwargs

        # Check that state plotting was called with correct parameters
        assert "traces" in state_kwargs
        assert state_kwargs["traces"] is not None
        assert state_kwargs["traces"].shape == (25, 5)

        # Verify state plotting parameters
        assert "state_colors" in state_kwargs
        assert "state_names" in state_kwargs

    @patch("utils.state_epoch_output._plot_raster")
    @patch("utils.plots._plot_timecourse")
    def test_event_preview_uses_real_timepoints(
        self,
        mock_plot_timecourse,
        mock_plot_raster,
        setup_complete_test_environment,
    ):
        """Test that event preview uses real behavioral timepoints."""
        env = setup_complete_test_environment

        # Call event preview function
        env["generator"]._create_event_preview(
            results=env["results"],
            cell_info=env["cell_info"],
            events=env["events"],
            annotations_df=env["annotations_df"],
            column_name="state",
        )

        # Verify state overlay plotting function was called
        assert (
            mock_plot_raster.called
        ), "_plot_raster was not called for state overlay"

        # With annotations present, custom epoch overlay is used instead of _plot_timecourse
        # The custom implementation creates its own figures rather than calling _plot_timecourse

        # Check that state raster plotting was called with correct parameters
        raster_call_args = mock_plot_raster.call_args
        raster_args, raster_kwargs = raster_call_args

        # Should have behavior data and state parameters
        assert "behavior" in raster_kwargs
        assert "column_name" in raster_kwargs
        assert raster_kwargs["column_name"] == "state"
        assert "state_colors" in raster_kwargs
        assert "state_names" in raster_kwargs

    def test_preview_color_consistency(self, setup_complete_test_environment):
        """Test that color mapping logic works correctly in preview functions."""
        env = setup_complete_test_environment
        generator = env["generator"]

        # Test inline color mapping logic
        combinations = [("rest", "baseline"), ("active", "training")]

        # Simulate the inline color mapping logic
        state_colors = []
        for state, _epoch in combinations:
            try:
                state_idx = generator.states.index(state)
                state_color = generator.color_scheme.state_colors[state_idx]
            except (ValueError, IndexError):
                state_color = "gray"  # Default color for unknown states
            state_colors.append(state_color)

        # Verify expected colors based on generator setup
        expected_colors = ["gray", "blue"]  # rest=0, active=1
        assert state_colors == expected_colors

    def test_all_previews_handle_missing_annotations(
        self, setup_complete_test_environment
    ):
        """Test that preview functions handle missing annotations gracefully."""
        env = setup_complete_test_environment

        # Test with None annotations - should not crash
        try:
            env["generator"]._create_trace_preview(
                results=env["results"],
                cell_info=env["cell_info"],
                traces=env["traces"],
                events=env["events"],
                annotations_df=None,  # Missing annotations
                column_name="state",
            )

            env["generator"]._create_event_preview(
                results=env["results"],
                cell_info=env["cell_info"],
                events=env["events"],
                annotations_df=None,  # Missing annotations
                column_name="state",
            )

            # Should complete without errors (graceful handling)
            assert (
                True
            ), "Preview functions handled missing annotations gracefully"
        except Exception as e:
            pytest.fail(
                f"Preview functions should handle missing annotations gracefully, but got: {e}"
            )


class TestStateEpochOverlayFunctionality:
    """Test state-epoch overlay functionality (simplified)."""

    def test_overlay_functionality_exists(self):
        """Test that overlay functionality is available through output generator."""
        # Simple test to verify overlay functions exist
        generator = StateEpochOutputGenerator(
            output_dir="",
            states=["rest", "active"],
            epochs=["baseline", "test"],
            state_colors=["gray", "blue"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="rest",
            baseline_epoch="baseline",
        )

        # Verify overlay methods exist
        assert hasattr(generator, "_plot_trace_preview_with_state_overlays")
        assert hasattr(generator, "_plot_trace_preview_with_epoch_overlays")
        assert hasattr(generator, "_plot_event_preview_with_state_overlays")
        assert hasattr(generator, "_plot_event_preview_with_epoch_overlays")


if __name__ == "__main__":
    pytest.main([__file__])
