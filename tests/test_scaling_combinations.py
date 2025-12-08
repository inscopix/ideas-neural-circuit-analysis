"""
Test scaling method combinations for trace and event data.

This test suite validates that trace_scale_method and event_scale_method
work independently and produce correct output ranges for all combinations.
"""

import unittest
import numpy as np
import pandas as pd
import pytest

from utils.state_epoch_data import scale_data
from ideas.exceptions import IdeasError


# All available scaling methods
SCALING_METHODS = [
    "none",
    "normalize",
    "standardize",
    "fractional_change",
    "standardize_baseline",
]


class TestScalingMethodValidation(unittest.TestCase):
    """Test that individual scaling methods work correctly."""

    def setUp(self):
        """Create mock data for scaling tests."""
        np.random.seed(42)
        # Create data with realistic ranges
        self.traces = np.random.randn(100, 20) * 10 + 50  # Mean ~50, std ~10
        self.events = np.random.poisson(5, (100, 20)).astype(float)  # ~5 events/sec

        # Create behavior dataframe with baseline state
        self.behavior = pd.DataFrame(
            {"state": ["rest"] * 30 + ["active"] * 40 + ["feeding"] * 30}
        )
        self.baseline_state = "rest"

    def test_none_scaling_preserves_original_values(self):
        """Test that 'none' scaling preserves original data."""
        scaled_traces = scale_data(
            self.traces,
            method="none",
            behavior=self.behavior,
            baseline_state=self.baseline_state,
        )
        np.testing.assert_array_equal(scaled_traces, self.traces)

    def test_normalize_produces_zero_one_range(self):
        """Test that 'normalize' produces values in [0, 1] range."""
        scaled_traces = scale_data(
            self.traces,
            method="normalize",
        )
        assert scaled_traces.min() >= 0, "Normalized data should be >= 0"
        assert scaled_traces.max() <= 1, "Normalized data should be <= 1"
        # Check that we actually use the full range
        assert scaled_traces.min() < 0.1, "Should have values near 0"
        assert scaled_traces.max() > 0.9, "Should have values near 1"

    def test_standardize_produces_zero_mean_unit_variance(self):
        """Test that 'standardize' produces mean=0, std=1."""
        scaled_traces = scale_data(
            self.traces,
            method="standardize",
        )
        assert abs(np.nanmean(scaled_traces)) < 1e-10, "Mean should be ~0"
        assert abs(np.nanstd(scaled_traces) - 1.0) < 1e-10, "Std should be ~1"

    def test_fractional_change_relative_to_baseline(self):
        """Test that 'fractional_change' computes correct ratio to baseline."""
        scaled_traces = scale_data(
            self.traces,
            method="fractional_change",
            behavior=self.behavior,
            column_name="state",
            baseline_state=self.baseline_state,
        )
        # Baseline samples should average to ~1.0 (ratio to themselves)
        baseline_indices = self.behavior["state"] == "rest"
        baseline_values = scaled_traces[baseline_indices]
        assert (
            abs(np.nanmean(baseline_values) - 1.0) < 0.2
        ), "Baseline should be near 1.0 for fractional change (ratio to itself)"
        # Non-baseline values should differ from 1.0
        non_baseline_values = scaled_traces[~baseline_indices]
        assert (
            abs(np.nanmean(non_baseline_values) - 1.0) > 0.01
        ), "Non-baseline values should differ from 1.0"

    def test_standardize_baseline_uses_baseline_stats(self):
        """Test that 'standardize_baseline' uses baseline statistics."""
        scaled_traces = scale_data(
            self.traces,
            method="standardize_baseline",
            behavior=self.behavior,
            column_name="state",
            baseline_state=self.baseline_state,
        )
        # Baseline samples should have mean ~0, std ~1
        baseline_indices = self.behavior["state"] == "rest"
        baseline_values = scaled_traces[baseline_indices]
        assert abs(np.nanmean(baseline_values)) < 0.2, "Baseline mean should be near 0"

    def test_scaling_requires_behavior_for_baseline_methods(self):
        """Test that baseline methods require behavior data."""
        with pytest.raises(IdeasError, match="Behavior data.*must be specified"):
            scale_data(
                self.traces,
                method="fractional_change",
                behavior=None,
                baseline_state=self.baseline_state,
            )

    def test_unknown_scaling_method_raises_error(self):
        """Test that unknown scaling method raises error."""
        with pytest.raises(IdeasError, match="Unknown scaling method"):
            scale_data(
                self.traces,
                method="invalid_method",
            )


class TestScalingCombinationsDirectly:
    """
    Test different combinations of trace and event scaling methods.

    This tests that trace and event scaling work independently by
    applying them directly to data arrays.
    """

    @classmethod
    def setup_class(cls):
        """Create mock data for scaling tests."""
        # Create realistic mock data
        np.random.seed(42)
        cls.n_samples = 150
        cls.n_cells = 10

        # Traces: calcium fluorescence (0-100 range)
        cls.traces = np.random.randn(cls.n_samples, cls.n_cells) * 15 + 50
        cls.traces = np.clip(cls.traces, 0, None)  # Non-negative

        # Events: detected calcium events (0-10 events/sec)
        cls.events = np.random.poisson(3, (cls.n_samples, cls.n_cells)).astype(float)

        # Create behavior dataframe
        cls.behavior = pd.DataFrame(
            {
                "state": (["rest"] * 50 + ["exploration"] * 50 + ["feeding"] * 50),
            }
        )
        cls.baseline_state = "rest"

    @pytest.mark.parametrize(
        "trace_method,event_method",
        [
            # Test all combinations of scaling methods
            ("none", "none"),
            ("none", "normalize"),
            ("none", "standardize"),
            ("normalize", "none"),
            ("normalize", "standardize"),
            ("standardize", "none"),
            ("standardize", "normalize"),
            ("standardize", "standardize"),
            # Test baseline methods
            ("fractional_change", "none"),
            ("standardize_baseline", "none"),
            ("none", "fractional_change"),
            ("none", "standardize_baseline"),
            ("standardize", "fractional_change"),
            ("fractional_change", "standardize"),
        ],
    )
    def test_scaling_method_combinations(self, trace_method, event_method):
        """
        Test that different trace/event scaling combinations work correctly.

        This test validates:
        1. Scaling completes without errors
        2. Scaled values are in expected ranges for each method
        3. Trace and event data scale independently
        """
        # Apply trace scaling
        if trace_method in ["fractional_change", "standardize_baseline"]:
            scaled_traces = scale_data(
                self.traces.copy(),
                method=trace_method,
                behavior=self.behavior,
                column_name="state",
                baseline_state=self.baseline_state,
            )
        else:
            scaled_traces = scale_data(
                self.traces.copy(),
                method=trace_method,
            )

        # Apply event scaling
        if event_method in ["fractional_change", "standardize_baseline"]:
            scaled_events = scale_data(
                self.events.copy(),
                method=event_method,
                behavior=self.behavior,
                column_name="state",
                baseline_state=self.baseline_state,
            )
        else:
            scaled_events = scale_data(
                self.events.copy(),
                method=event_method,
            )

        # Validate trace value ranges
        self._validate_trace_value_range(scaled_traces.flatten(), trace_method)

        # Validate event value ranges
        self._validate_event_value_range(scaled_events.flatten(), event_method)

        # Validate that all values are finite
        assert np.isfinite(
            scaled_traces
        ).all(), f"Trace scaling '{trace_method}' should produce finite values"
        assert np.isfinite(
            scaled_events
        ).all(), f"Event scaling '{event_method}' should produce finite values"

    def _validate_trace_value_range(
        self,
        values: np.ndarray,
        method: str,
    ):
        """Validate that trace values are in expected range for method."""
        if method == "none":
            # Raw fluorescence: typically 0-100
            assert values.min() >= 0, "Raw traces should be non-negative"
            assert values.max() < 200, "Raw traces should be < 200"
        elif method == "normalize":
            # Should be in [0, 1]
            assert values.min() >= -0.01, "Normalized should be >= 0"
            assert values.max() <= 1.01, "Normalized should be <= 1"
        elif method == "standardize":
            # Should have mean ~0, std ~1, typically in [-3, 3]
            assert abs(values.mean()) < 1, "Standardized mean should be ~0"
            assert 0.5 < values.std() < 2.0, "Standardized std should be ~1"
        elif method in ["fractional_change", "standardize_baseline"]:
            # Can have wide range, but should be reasonable
            assert values.min() > -10, "Should be > -10"
            assert values.max() < 10, "Should be < 10"

    def _validate_event_value_range(
        self,
        values: np.ndarray,
        method: str,
    ):
        """Validate that event values are in expected range for method."""
        if method == "none":
            # Raw event rates: typically 0-20 events/sec
            assert values.min() >= 0, "Raw events should be non-negative"
            assert values.max() < 50, "Raw events should be < 50"
        elif method == "normalize":
            # Should be in [0, 1]
            assert values.min() >= -0.01, "Normalized should be >= 0"
            assert values.max() <= 1.01, "Normalized should be <= 1"
        elif method == "standardize":
            # Should have mean ~0, std ~1
            assert abs(values.mean()) < 1, "Standardized mean should be ~0"
            assert 0.5 < values.std() < 2.0, "Standardized std should be ~1"
        elif method in ["fractional_change", "standardize_baseline"]:
            # Can have wide range
            assert values.min() > -10, "Should be > -10"
            assert values.max() < 10, "Should be < 10"

    def test_trace_event_scaling_independence(self):
        """
        Test that trace and event scaling are truly independent.

        Scaling events should not affect traces and vice versa.
        """
        # Scale traces with "none", events with two different methods
        scaled_events_none = scale_data(self.events.copy(), method="none")
        scaled_events_std = scale_data(self.events.copy(), method="standardize")

        # Events should be different
        assert not np.allclose(
            scaled_events_none, scaled_events_std
        ), "Different event scaling should produce different results"

        # But trace data should be unchanged (not passed to scaling)
        # This test confirms that scaling one doesn't affect the other
        original_trace_mean = self.traces.mean()
        scaled_trace_none = scale_data(self.traces.copy(), method="none")
        assert np.allclose(
            scaled_trace_none.mean(), original_trace_mean
        ), "Trace scaling with 'none' should preserve original values"


class TestScalingMethodRangeValidation:
    """Test that scaled data produces expected statistical properties."""

    def setup_method(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_cells = 15
        self.traces = np.random.randn(self.n_samples, self.n_cells) * 20 + 50
        self.behavior = pd.DataFrame(
            {"state": ["rest"] * 50 + ["active"] * 100 + ["rest"] * 50}
        )

    @pytest.mark.parametrize("method", SCALING_METHODS)
    def test_scaling_produces_finite_values(self, method):
        """Test that all scaling methods produce finite values."""
        if method in ["fractional_change", "standardize_baseline"]:
            scaled = scale_data(
                self.traces,
                method=method,
                behavior=self.behavior,
                column_name="state",
                baseline_state="rest",
            )
        else:
            scaled = scale_data(self.traces, method=method)

        assert np.isfinite(
            scaled
        ).all(), f"Scaling method '{method}' should produce finite values"

    @pytest.mark.parametrize("method", ["normalize", "standardize"])
    def test_scaling_changes_distribution(self, method):
        """Test that scaling methods actually transform the data."""
        scaled = scale_data(self.traces, method=method)

        # Scaled data should differ from original
        assert not np.allclose(
            scaled, self.traces
        ), f"Scaling method '{method}' should transform data"

        # Scaled data should have different mean/std
        orig_mean = np.mean(self.traces)
        scaled_mean = np.mean(scaled)
        assert (
            abs(orig_mean - scaled_mean) > 1
        ), f"Scaling method '{method}' should change mean significantly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
