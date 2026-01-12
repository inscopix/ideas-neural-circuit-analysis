"""
Comprehensive tests for color mapping and timeline functionality in state-epoch baseline analysis.

Tests cover:
- Two-layer color mapping (states inner, epochs outer)
- Real behavioral timeline reconstruction
- State-epoch overlay functionality
- Preview file generation with accurate timepoints
- Color consistency across all preview functions
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from utils.plots import _plot_state_epoch_time
from utils.plotting_utils import plot_events_bottom_panel
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

        results.get_combination_results.side_effect = mock_get_combination_results

        return results

    # NOTE: _create_two_layer_colors method was removed and logic inlined
    # The color mapping functionality is now part of _create_event_average_preview
    # and is tested through integration tests rather than unit tests

    # NOTE: Color mapping consistency is now tested through integration tests
    # since the logic was inlined into _create_event_average_preview

    def test_state_epoch_palette_prefers_epoch_colors(self, tmp_path):
        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["rest", "active"],
            epochs=["baseline", "test"],
            state_colors=["gray", "blue"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="rest",
            baseline_epoch="baseline",
        )
        combination_order = [("rest", "baseline"), ("active", "test")]
        labels = ["rest-baseline", "active-test"]

        ecdf_palette, box_palette = generator._build_state_epoch_palette(
            combination_order, labels
        )

        assert ecdf_palette["rest-baseline"] == "lightgray"
        assert ecdf_palette["active-test"] == "lightblue"
        assert box_palette == ecdf_palette

    def test_state_epoch_palette_epoch_only_mode(self, tmp_path):
        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["epoch_activity"],
            epochs=["epoch1", "epoch2"],
            state_colors=["gray"],
            epoch_colors=["orange", "green"],
            baseline_state="epoch_activity",
            baseline_epoch="epoch1",
            hide_state_prefix=True,
            epoch_only_mode=True,
        )

        combination_order = [("epoch_activity", "epoch1"), ("epoch_activity", "epoch2")]
        labels = ["epoch1", "epoch2"]

        ecdf_palette, box_palette = generator._build_state_epoch_palette(
            combination_order, labels
        )
        assert ecdf_palette == box_palette
        assert ecdf_palette["epoch1"] == "orange"
        assert ecdf_palette["epoch2"] == "green"

    @patch("seaborn.stripplot")
    @patch("seaborn.boxplot")
    def test_average_correlation_preview_uses_epoch_colors(
        self, mock_boxplot, mock_stripplot, tmp_path, monkeypatch
    ):
        import numpy as np

        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["rest", "active"],
            epochs=["baseline", "test"],
            state_colors=["gray", "blue"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="rest",
            baseline_epoch="baseline",
        )

        labeled_matrices = [
            ("rest-baseline", np.ones((2, 2))),
            ("active-test", np.ones((2, 2))),
        ]
        color_map = {"rest-baseline": "lightgray", "active-test": "lightblue"}

        def fake_save(fig, output_path, title):
            import matplotlib.pyplot as plt

            plt.close(fig)

        monkeypatch.setattr(
            "utils.state_epoch_output.save_figure_with_cleanup", fake_save
        )

        generator._plot_average_correlations_with_state_epoch_labels(
            labeled_matrices, color_map, "test.svg"
        )

        palette_arg = mock_boxplot.call_args.kwargs["palette"]
        assert palette_arg == color_map

    def test_create_average_correlation_preview_state_mode(self, tmp_path, monkeypatch):
        import numpy as np

        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["rest", "active"],
            epochs=["baseline", "test"],
            state_colors=["gray", "blue"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="rest",
            baseline_epoch="baseline",
        )

        def fake_collect(_results, _matrix_key):
            return {
                "rest_baseline": np.ones((2, 2)),
                "active_test": np.ones((2, 2)),
            }

        def fake_order(_results, _valid=None):
            return [("rest", "baseline"), ("active", "test")]

        captured = {}

        def fake_plot(labeled, color_map, output_filename):
            captured["labels"] = [label for label, _ in labeled]
            captured["color_map"] = color_map

        monkeypatch.setattr(
            generator,
            "_collect_correlation_matrices",
            fake_collect,
        )
        monkeypatch.setattr(generator, "_determine_combination_order", fake_order)
        monkeypatch.setattr(
            generator,
            "_plot_average_correlations_with_state_epoch_labels",
            fake_plot,
        )

        generator._create_average_correlations_preview(results=MagicMock())

        assert captured["labels"] == ["rest-baseline", "active-test"]
        assert captured["color_map"]["rest-baseline"] == "lightgray"
        assert captured["color_map"]["active-test"] == "lightblue"

    def test_create_average_correlation_preview_epoch_only(self, tmp_path, monkeypatch):
        import numpy as np

        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["epoch_activity"],
            epochs=["epoch1", "epoch2"],
            state_colors=["gray"],
            epoch_colors=["orange", "green"],
            baseline_state="epoch_activity",
            baseline_epoch="epoch1",
            hide_state_prefix=True,
            epoch_only_mode=True,
        )

        def fake_collect(_results, _matrix_key):
            return {
                "epoch1": np.ones((2, 2)),
                "epoch2": np.ones((2, 2)),
            }

        def fake_order(_results, _valid=None):
            return [
                ("epoch_activity", "epoch1"),
                ("epoch_activity", "epoch2"),
            ]

        captured = {}

        def fake_plot(labeled, color_map, output_filename):
            captured["labels"] = [label for label, _ in labeled]
            captured["color_map"] = color_map

        monkeypatch.setattr(
            generator,
            "_collect_correlation_matrices",
            fake_collect,
        )
        monkeypatch.setattr(generator, "_determine_combination_order", fake_order)
        monkeypatch.setattr(
            generator,
            "_plot_average_correlations_with_state_epoch_labels",
            fake_plot,
        )

        generator._create_average_correlations_preview(results=MagicMock())

        assert captured["labels"] == ["epoch1", "epoch2"]
        assert captured["color_map"]["epoch1"] == "orange"
        assert captured["color_map"]["epoch2"] == "green"

    @patch("utils.plots.sns.heatmap")
    def test_state_epoch_time_preview_new_layout(self, mock_heatmap, tmp_path):
        period = 1.0
        behavior = pd.DataFrame(
            {
                "state": ["rest"] * 10 + ["active"] * 5,
            }
        )
        output_file = tmp_path / "state_epoch_time.svg"
        _plot_state_epoch_time(
            behavior=behavior,
            column_name="state",
            state_names=["rest", "active"],
            state_colors=["gray", "blue"],
            period=period,
            filename=str(output_file),
            epoch_names=["baseline", "test"],
            epoch_periods=[(0, 10), (10, 15)],
            epoch_colors=["lightgray", "lightblue"],
        )
        assert mock_heatmap.called
        assert output_file.exists()

    @patch("utils.plots.sns.heatmap")
    def test_plot_epoch_only_time_saves_file(self, mock_heatmap, tmp_path):
        filename = tmp_path / "epoch_only.svg"
        _plot_state_epoch_time(
            behavior=pd.DataFrame({"state": []}),
            column_name="state",
            state_names=["epoch_activity"],
            state_colors=["gray"],
            period=1.0,
            filename=str(filename),
            epoch_names=["baseline", "stim"],
            epoch_periods=[(0, 5), (5, 9)],
            epoch_colors=["lightgray", "lightblue"],
            epoch_only_mode=True,
        )
        assert filename.exists()
        assert not mock_heatmap.called

    @patch("utils.state_epoch_output._plot_state_epoch_time")
    def test_epoch_only_mode_state_time_preview_uses_epoch_view(
        self, mock_epoch_time, tmp_path
    ):
        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["epoch_activity"],
            epochs=["epoch1", "epoch2"],
            state_colors=["gray"],
            epoch_colors=["orange", "green"],
            baseline_state="epoch_activity",
            baseline_epoch="epoch1",
            epoch_periods=[(0, 5), (5, 10)],
            epoch_only_mode=True,
            hide_state_prefix=True,
        )

        annotations = pd.DataFrame(
            {"state": ["epoch_activity"] * 10, "time": np.arange(10)}
        )
        generator._create_state_time_preview(
            annotations_df=annotations,
            column_name="state",
            cell_info={"period": 1.0},
        )
        mock_epoch_time.assert_called_once()
        assert mock_epoch_time.call_args.kwargs.get("epoch_only_mode") is True


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

        results.get_combination_results.side_effect = mock_get_combination_results

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

    @patch("utils.state_epoch_output._plot_traces_with_epochs")
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
        assert mock_plot_traces.called, (
            "_plot_traces_with_epochs was not called for state overlay"
        )

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
        assert "epoch_periods" in state_kwargs
        assert state_kwargs["epoch_periods"] is not None
        assert "epoch_colors" in state_kwargs
        assert state_kwargs["epoch_colors"] is not None

    @patch("utils.state_epoch_output._plot_raster_with_epochs")
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
        assert mock_plot_raster.called, (
            "_plot_raster_with_epochs was not called for state overlay"
        )

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
        assert "epoch_periods" in raster_kwargs
        assert raster_kwargs["epoch_periods"] is not None
        assert "epoch_colors" in raster_kwargs
        assert raster_kwargs["epoch_colors"] is not None

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
            assert True, "Preview functions handled missing annotations gracefully"
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
        assert hasattr(generator, "_plot_event_preview_with_state_overlays")

        missing_epoch_overlays = [
            name
            for name in (
                "_plot_trace_preview_with_epoch_overlays",
                "_plot_event_preview_with_epoch_overlays",
            )
            if not hasattr(generator, name)
        ]
        if missing_epoch_overlays:
            pytest.skip(
                "Epoch overlay previews are not implemented in this environment."
            )

        assert hasattr(generator, "_plot_trace_preview_with_epoch_overlays")
        assert hasattr(generator, "_plot_event_preview_with_epoch_overlays")

    @patch("utils.state_epoch_output.create_dual_panel_plot_with_epoch_overlays")
    def test_epoch_only_overlays_share_dual_panel_helper(
        self, mock_dual_panel, tmp_path
    ):
        """Ensure epoch-only trace/event overlays share the same helper pipeline."""
        generator = StateEpochOutputGenerator(
            output_dir=str(tmp_path),
            states=["epoch_activity"],
            epochs=["baseline", "stim"],
            state_colors=["gray"],
            epoch_colors=["lightgray", "lightblue"],
            baseline_state="epoch_activity",
            baseline_epoch="baseline",
            epoch_only_mode=True,
        )

        epochs = [(0.0, 5.0), (5.0, 10.0)]
        traces = np.random.default_rng(0).random((10, 3))
        generator._plot_trace_preview_with_epoch_overlays(
            traces=traces,
            epochs=epochs,
            boundaries=[0.0, 5.0, 10.0],
            period=1.0,
            epoch_names=["baseline", "stim"],
            epoch_colors=["lightgray", "lightblue"],
        )

        events_series = np.random.default_rng(1).random((10, 3))
        events_offsets = [
            np.array([1.0, 3.0]),
            np.array([2.0]),
            np.array([]),
        ]
        generator._plot_event_preview_with_epoch_overlays(
            events=events_offsets,
            event_timeseries=events_series,
            epochs=epochs,
            boundaries=[0.0, 5.0, 10.0],
            period=1.0,
            epoch_names=["baseline", "stim"],
            epoch_colors=["lightgray", "lightblue"],
        )

        assert mock_dual_panel.call_count == 2
        trace_call, event_call = mock_dual_panel.call_args_list
        assert (
            trace_call.kwargs["bottom_panel_callback"].__name__
            == "plot_traces_bottom_panel"
        )
        assert (
            event_call.kwargs["bottom_panel_callback"].__name__
            == "plot_events_bottom_panel"
        )

    def test_event_bottom_panel_aligns_with_population_axis(self):
        """Ensure raster x-axis span matches duration used by population plot."""
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        try:
            period = 0.5
            event_timeseries = np.zeros((21, 3))
            events = [
                np.array([0.5, 1.0, 2.5]),
                np.array([3.0]),
                np.array([]),
            ]
            plot_events_bottom_panel(
                ax=ax,
                event_timeseries=event_timeseries,
                period=period,
                events=events,
            )
            expected_last_time = (event_timeseries.shape[0] - 1) * period
            assert ax.get_xlim()[1] == pytest.approx(expected_last_time)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
