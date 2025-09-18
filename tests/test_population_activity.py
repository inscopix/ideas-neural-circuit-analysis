import os
import re

import pytest
import numpy as np
import pandas as pd
from beartype.roar import BeartypeCallHintParamViolation
from ideas.exceptions import IdeasError
# from ideas_data import fetch

from utils.utils import Comp
from analysis.population_activity import (
    population_activity,
    compute_modulation,
    compute_two_state_modulation,
    find_modulated_neurons,
    find_two_state_modulated_neurons,
    _make_modulation_data,
    _modulation_data_to_csv,
)

series_annotations = "data/2023-05-10-13-55-53_video-camera-1-annotations.series.parquet"

series_cell_sets = [
    "data/2023-05-10-13-55-53_video-PP-BP-MC-CNMFE-full.isxd",
    "data/2023-05-10-14-15-44_video-PP-BP-MC-CNMFE-full.isxd",
    "data/2023-05-10-14-36-23_video-PP-BP-MC-CNMFE-full.isxd",
]

n_shuffles = 1
cell_set_file = "data/input_cellset.isxd"
event_set_file = "data/input_cellset-ED.isxd"
annotations_file = "data/ideas_experiment_annotations.parquet"


valid_inputs = [
    # valid inputs
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=[event_set_file],
        annotations_file=[annotations_file],
        concatenate=True,
        trace_scale_method="none",
        event_scale_method="none",
        state_names="center, quad 1, quad 2, quad 4",
        state_colors="tab:gray, tab:blue, tab:purple, tab:green",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="pairwise",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # valid inputs, single state
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        trace_scale_method="none",
        event_scale_method="none",
        state_names="center",
        state_colors="tab:blue",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs not state",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # test scale methods
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        trace_scale_method="standardize",
        event_scale_method="normalize",
        state_names="center",
        state_colors="tab:blue",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs not state",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # test baseline scale methods
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        trace_scale_method="standardize_baseline",
        event_scale_method="fractional_change",
        state_names="center",
        state_colors="tab:blue",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs not state",
        baseline_state="center",
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # test remaining methods
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        trace_scale_method="none",
        event_scale_method="none",
        state_names="center, quad 1, quad 2, quad 4",
        state_colors="tab:gray, tab:blue, tab:purple, tab:green",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs not defined",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # make sure baseline works for events as well
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=[event_set_file],
        annotations_file=[annotations_file],
        concatenate=True,
        trace_scale_method="none",
        event_scale_method="none",
        state_names="center, quad 1, quad 2, quad 4",
        state_colors="tab:gray, tab:blue, tab:purple, tab:green",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs baseline",
        baseline_state="center",
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # Color types
    # hex
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        state_names="center, quad 1, quad 2, quad 4",
        state_colors="#000000, #0000FF, #800080, #008000",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs baseline",
        baseline_state="center",
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
    # series inputs
    dict(
        cell_set_files=series_cell_sets,
        event_set_files=None,
        annotations_file=[series_annotations],
        concatenate=True,
        state_names="other, move, rest",
        state_colors="tab:gray, tab:blue, tab:purple",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs baseline",
        baseline_state="rest",
        n_shuffle=n_shuffles,
        alpha=0.05,
    ),
]

invalid_inputs = [
    # invalid states
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        state_names="bad state, does not exist",
        state_colors="tab:gray, tab:blue, tab:purple, tab:green",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="pairwise",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
        error=IdeasError,
        error_text="Number of state names (2) must match number of state colors (4)",
    ),
    # no states
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[annotations_file],
        concatenate=True,
        state_names=None,
        state_colors="tab:gray, tab:blue, tab:purple, tab:green",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="state vs not defined",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
        error=BeartypeCallHintParamViolation,
        error_text="Function analysis.population_activity.population_activity() "
        "parameter state_names=\"None\" violates type hint <class 'str'>, "
        'as <class "builtins.NoneType"> "None" not instance of str.',
    ),
    # in-congruent files
    dict(
        cell_set_files=[cell_set_file],
        event_set_files=None,
        annotations_file=[series_annotations],
        concatenate=True,
        state_names="center, quad 1, quad 2, quad 4",
        state_colors="tab:gray, tab:blue, tab:purple, tab:green",
        modulation_colors="tab:red, tab:blue",
        column_name="state",
        method="pairwise",
        baseline_state=None,
        n_shuffle=n_shuffles,
        alpha=0.05,
        error=IdeasError,
        error_text="Mismatch between length of annotations",
    ),
]


@pytest.mark.parametrize("params", valid_inputs)
def test_population_activity(params, output_file_cleanup):
    """Check that code runs without error with valid inputs."""

    params["n_shuffle"] = n_shuffles

    population_activity(**params)

    # check that the output files are created
    activity_previews = [
        "activity_average_preview.svg",
        "modulation_histogram_preview.svg",
        "modulation_preview.svg",
        "time_in_state_preview.svg",
        "trace_preview.svg",
    ]
    event_previews = [
        "event_average_preview.svg",
        "event_modulation_preview.svg",
        "event_modulation_histogram_preview.svg",
        "event_preview.svg",
    ]

    trace_file = "trace_population_data.csv"
    event_file = "event_population_data.csv"

    for output_file in activity_previews:
        assert os.path.exists(output_file)
    assert os.path.exists(trace_file)

    if params["event_set_files"] is not None:
        for output_file in event_previews:
            assert os.path.exists(output_file)
        assert os.path.exists(event_file)

    # All output files automatically cleaned up by output_file_cleanup fixture


@pytest.mark.parametrize("params", invalid_inputs)
def test_population_activity_invalid_inputs(params, output_file_cleanup):
    """Check that with invalid inputs the expected error is raised."""

    params["n_shuffle"] = n_shuffles

    error = params.pop("error")
    error_text = params.pop("error_text")

    with pytest.raises(error, match=re.escape(error_text)):
        population_activity(**params)

    # Any output files created before error automatically cleaned up by fixture


# Add unit tests for individual functions
class TestComputeModulation:
    """Test the compute_modulation function with various input scenarios."""

    def test_normal_case(self):
        """Test with typical input."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
        )
        when = np.array([True, False, True])
        result = compute_modulation(traces, when)

        # Expected calculation:
        # After subtracting min by column: [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]]
        # For when=True: mean of first column is (0+4)/2 = 2.0
        # For when=False: mean of first column is 2.0
        # For when=True: mean of second column is (0+4)/2 = 2.0
        # For when=False: mean of second column is 2.0
        # Modulation scores: (2-2)/(2+2) = 0.0 for both columns

        assert len(result) == 2
        assert np.all(np.isclose(result, 0.0))

    def test_all_true(self):
        """Test when all values in 'when' are True."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
        )
        when = np.array([True, True, True])
        # For all True, the modulation becomes (x-0)/(x+0) which is undefined/NaN
        # Our function should handle this gracefully
        result = compute_modulation(traces, when)
        assert len(result) == 2
        # The function should return 0s for this case as there's no meaningful modulation
        assert np.all(np.isclose(result, 0))

    def test_all_false(self):
        """Test when all values in 'when' are False."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
        )
        when = np.array([False, False, False])
        # For all False, the modulation becomes (0-y)/(0+y) which is -1
        # Our function should handle this gracefully
        result = compute_modulation(traces, when)
        assert len(result) == 2
        # The function should return 0s for this case as there's no meaningful modulation
        assert np.all(np.isclose(result, 0))

    def test_nan_handling(self):
        """Test handling of NaN values."""
        traces = np.array(
            [[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
        )
        when = np.array([True, False, True])
        result = compute_modulation(traces, when)
        # The function should handle NaNs and not propagate them
        assert not np.isnan(np.sum(result))

    def test_input_not_modified(self):
        """Test that input traces are not modified."""
        traces = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64
        )
        traces_orig = traces.copy()
        when = np.array([True, False, True])
        compute_modulation(traces, when)
        # The function should not modify the input array
        assert np.array_equal(traces, traces_orig)

    def test_nonzero_modulation(self):
        """Test with input that should produce non-zero modulation scores."""
        traces = np.array(
            [[1.0, 10.0], [5.0, 7.0], [3.0, 9.0]], dtype=np.float64
        )
        when = np.array([True, False, True])
        result = compute_modulation(traces, when)

        # Expected calculation:
        # After subtracting min by column: [[0.0, 3.0], [4.0, 0.0], [2.0, 2.0]]
        # For when=True: mean of first column is (0+2)/2 = 1.0
        # For when=False: mean of first column is 4.0
        # For when=True: mean of second column is (3+2)/2 = 2.5
        # For when=False: mean of second column is 0.0
        # Modulation scores: (1-4)/(1+4) = -0.6 and (2.5-0)/(2.5+0) = 1.0

        assert len(result) == 2
        assert np.isclose(result[0], -0.6)
        assert np.isclose(result[1], 1.0)

    def test_edge_case_extreme_values(self):
        """Test with extreme values to ensure numerical stability."""
        # Very large values
        traces = np.array(
            [[1e6, 2e6], [3e6, 4e6], [5e6, 6e6]], dtype=np.float64
        )
        when = np.array([True, False, True])
        result = compute_modulation(traces, when)
        assert np.all(np.isfinite(result))

        # Very small values
        traces = np.array(
            [[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]], dtype=np.float64
        )
        when = np.array([True, False, True])
        result = compute_modulation(traces, when)
        assert np.all(np.isfinite(result))

    def test_mathematical_correctness(self):
        """Test mathematical correctness of modulation calculation
        with a manually calculated example.
        """
        traces = np.array(
            [[2.0, 10.0], [6.0, 3.0], [10.0, 8.0]], dtype=np.float64
        )
        when = np.array([True, False, True])

        # Manual calculation:
        # Min values by column: [2.0, 3.0]
        # Adjusted traces: [[0.0, 7.0], [4.0, 0.0], [8.0, 5.0]]
        # Mean when=True: [4.0, 6.0]
        # Mean when=False: [4.0, 0.0]
        # Modulation scores: (4-4)/(4+4) = 0.0, (6-0)/(6+0) = 1.0

        expected = np.array([0.0, 1.0])
        result = compute_modulation(traces, when)
        assert np.allclose(result, expected)

    def test_large_dataset_modulation(self):
        """Test modulation computation with a large dataset to ensure stability."""
        # Create a large dataset with 1000 timepoints and 100 neurons
        np.random.seed(42)
        n_timepoints, n_neurons = 1000, 100
        traces = np.random.normal(size=(n_timepoints, n_neurons))

        # Create a pattern where some neurons respond strongly to a specific state
        # First 25 neurons are up-modulated during when=True
        traces[:300, :25] += 3.0
        # Next 25 neurons are down-modulated during when=True
        traces[300:, 25:50] += 3.0
        # Remaining neurons have no clear pattern

        when = np.zeros(n_timepoints, dtype=bool)
        when[:300] = True

        result = compute_modulation(traces, when)

        # Verify expected modulation patterns
        assert len(result) == n_neurons
        assert np.all(np.isfinite(result))

        # Verify that we can detect the artificial modulation pattern
        assert np.mean(result[:25]) > 0.3
        assert np.mean(result[25:50]) < -0.3

        # Middle group should have modulation scores closer to 0
        assert np.abs(np.mean(result[50:])) < 0.2

    def test_extreme_signal_to_noise(self):
        """Test modulation calculation with different signal-to-noise ratios."""
        np.random.seed(123)
        n_timepoints, n_neurons = 500, 50

        # Base traces with noise
        traces = np.random.normal(0, 1, size=(n_timepoints, n_neurons))

        # Add strong signal to first 10 neurons during first half
        signal_strength = 5.0
        traces[:250, :10] += signal_strength

        when = np.zeros(n_timepoints, dtype=bool)
        when[:250] = True

        result_strong = compute_modulation(traces, when)

        # Create another set with much weaker signal
        traces_weak = np.random.normal(0, 1, size=(n_timepoints, n_neurons))
        weak_signal = 0.2
        traces_weak[:250, :10] += weak_signal

        result_weak = compute_modulation(traces_weak, when)

        # Strong signal should produce clear modulation
        assert np.mean(result_strong[:10]) > 0.5

        # Weak signal should still show positive modulation but less pronounced
        assert np.mean(result_weak[:10]) > 0
        assert np.mean(result_weak[:10]) < np.mean(result_strong[:10])


class TestComputeTwoStateModulation:
    """Test the compute_two_state_modulation function with various input scenarios."""

    def test_normal_case(self):
        """Test with typical input."""
        traces = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        when1 = np.array([True, False, True])
        when2 = np.array([False, True, False])
        result = compute_two_state_modulation(traces, when1, when2)
        assert len(result) == 2
        # Expected calculation:
        # After subtracting min by column: [[0, 0], [2, 2], [4, 4]]
        # For when1: mean of rows [0,2] for first column is (0+4)/2 = 2.0
        # For when2: mean of row [1] for first column is 2.0
        # For when1: mean of rows [0,2] for second column is (0+4)/2 = 2.0
        # For when2: mean of row [1] for second column is 2.0
        # Modulation scores: (2-2)/(2+2) = 0.0 for both columns
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 0.0)

    def test_input_not_modified(self):
        """Test that input data is not modified."""
        traces = np.random.rand(10, 5).astype(np.float64)
        when1 = np.random.random(10) > 0.5
        when2 = np.random.random(10) > 0.5
        when1_orig = when1.copy()
        when2_orig = when2.copy()
        compute_two_state_modulation(traces, when1, when2)
        assert np.array_equal(when1, when1_orig)
        assert np.array_equal(when2, when2_orig)

    def test_edge_cases(self):
        """Test edge cases with all True/False arrays."""
        traces = np.array([[1, 2], [3, 4], [5, 6]])

        # Case 1: when1 all True, when2 all False
        when1 = np.array([True, True, True])
        when2 = np.array([False, False, False])
        result1 = compute_two_state_modulation(traces, when1, when2)
        assert len(result1) == 2

        # Case 2: when1 all False, when2 all True
        when1 = np.array([False, False, False])
        when2 = np.array([True, True, True])
        result2 = compute_two_state_modulation(traces, when1, when2)
        assert len(result2) == 2
        # Results should be opposite signs
        assert np.allclose(result1, -result2)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        traces = np.array([[1, np.nan], [3, 4], [5, 6]])
        when1 = np.array([True, False, True])
        when2 = np.array([False, True, False])
        result = compute_two_state_modulation(traces, when1, when2)
        # The function should handle NaNs by converting to 0
        assert not np.isnan(np.sum(result))


class TestFindModulatedNeurons:
    """Test the find_modulated_neurons function for statistical detection of modulated neurons."""

    def test_normal_operation(self):
        """Test the function operates correctly with valid input."""
        traces = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        when = np.array([True, False, True])
        result = find_modulated_neurons(traces, when, alpha=0.05, n_shuffle=10)
        assert "up_modulated_neurons" in result
        assert "down_modulated_neurons" in result
        assert "p_val" in result
        assert "modulation_scores" in result
        assert len(result["modulation_scores"]) == 2

    def test_input_not_modified(self):
        """Test that input when array is not modified."""
        traces = np.random.rand(10, 5).astype(np.float64)
        when = np.random.random(10) > 0.5
        when_orig = when.copy()
        find_modulated_neurons(traces, when, n_shuffle=10)
        # Since find_modulated_neurons modifies 'when' in the loop, we need to check
        # that the original 'when' array still contains the same information even if
        # the array reference has been changed
        assert np.array_equal(when, when_orig)

    def test_statistical_significance(self):
        """Test that statistical significance is correctly determined."""
        # Create traces with clear modulation pattern
        traces = np.zeros((100, 3))
        # First neuron: clearly up-modulated in when=True
        traces[:50, 0] = 10.0
        # Second neuron: clearly down-modulated in when=True
        traces[50:, 1] = 10.0
        # Third neuron: not modulated
        traces[:, 2] = 5.0

        when = np.zeros(100, dtype=bool)
        when[:50] = True

        result = find_modulated_neurons(traces, when, n_shuffle=50, alpha=0.05)

        # First neuron should be up-modulated
        assert 0 in result["up_modulated_neurons"]
        # Second neuron should be down-modulated
        assert 1 in result["down_modulated_neurons"]
        # Third neuron should not be modulated
        assert 2 not in result["up_modulated_neurons"]
        assert 2 not in result["down_modulated_neurons"]

    def test_consistent_results_with_seed(self):
        """Test that results are consistent with the same random seed."""
        traces = np.random.rand(50, 10)
        when = np.random.random(50) > 0.5

        # Run twice with the same seed
        result1 = find_modulated_neurons(traces, when, n_shuffle=50)
        result2 = find_modulated_neurons(traces, when, n_shuffle=50)

        # Results should be identical
        assert np.array_equal(
            result1["up_modulated_neurons"], result2["up_modulated_neurons"]
        )
        assert np.array_equal(
            result1["down_modulated_neurons"],
            result2["down_modulated_neurons"],
        )
        assert np.array_equal(result1["p_val"], result2["p_val"])
        assert np.array_equal(
            result1["modulation_scores"], result2["modulation_scores"]
        )

    def test_large_scale_statistical_significance(self):
        """Test statistical significance with a large-scale realistic dataset."""
        # Create a large dataset with 1000 timepoints and 200 neurons
        np.random.seed(42)
        n_timepoints, n_neurons = 1000, 200
        traces = np.random.normal(size=(n_timepoints, n_neurons))

        # Create a controlled pattern:
        # - First 40 neurons: strongly up-modulated during state
        # - Next 40 neurons: strongly down-modulated during state
        # - Next 40 neurons: weakly up-modulated (may not all be detected)
        # - Next 40 neurons: weakly down-modulated (may not all be detected)
        # - Remaining 40 neurons: no modulation (random)

        # Create strong modulation
        traces[:300, :40] += 2.0
        traces[300:, 40:80] += 2.0

        # Create weak modulation
        traces[:300, 80:120] += 0.5
        traces[300:, 120:160] += 0.5

        when = np.zeros(n_timepoints, dtype=bool)
        when[:300] = True

        # Run with a substantial number of shuffles
        result = find_modulated_neurons(
            traces, when, n_shuffle=200, alpha=0.05
        )

        # Strong modulation should be detected with high accuracy
        detected_up_strong = np.intersect1d(
            result["up_modulated_neurons"], np.arange(40)
        )
        detected_down_strong = np.intersect1d(
            result["down_modulated_neurons"], np.arange(40, 80)
        )

        # Check detection rate for strongly modulated neurons (should be high)
        strong_up_detection_rate = len(detected_up_strong) / 40
        strong_down_detection_rate = len(detected_down_strong) / 40

        # We expect high detection rates for strongly modulated neurons
        assert (
            strong_up_detection_rate > 0.8
        ), f"Only detected {strong_up_detection_rate * 100:.1f}% of strongly up-modulated neurons"
        assert (
            strong_down_detection_rate > 0.8
        ), f"Only detected {strong_down_detection_rate} down modulated neurons, expected > 0.8"

        # False positives in non-modulated neurons (160-199) should be rare
        non_modulated_range = np.arange(160, 200)
        false_positives = np.intersect1d(
            np.union1d(
                result["up_modulated_neurons"],
                result["down_modulated_neurons"],
            ),
            non_modulated_range,
        )
        false_positive_rate = len(false_positives) / 40

        # False positive rate should be close to alpha
        assert (
            false_positive_rate < 0.1
        ), f"False positive rate {false_positive_rate * 100:.1f}% exceeds reasonable threshold"


class TestFindTwoStateModulatedNeurons:
    """Test the find_two_state_modulated_neurons
    function for detection between two explicitly defined states
    """

    def test_normal_operation(self):
        """Test the function operates correctly with valid input."""
        traces = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        when1 = np.array([True, False, True])
        when2 = np.array([False, True, False])
        result = find_two_state_modulated_neurons(
            traces, when1, when2, alpha=0.05, n_shuffle=10
        )

        assert "up_modulated_neurons" in result
        assert "down_modulated_neurons" in result
        assert "p_val" in result
        assert "modulation_scores" in result
        assert len(result["modulation_scores"]) == 2

    def test_inputs_not_modified(self):
        """Test that input arrays are not modified."""
        traces = np.random.rand(10, 5).astype(np.float64)
        when1 = np.random.random(10) > 0.5
        when2 = np.random.random(10) > 0.5
        when1_orig = when1.copy()
        when2_orig = when2.copy()
        find_two_state_modulated_neurons(traces, when1, when2, n_shuffle=10)
        # The original boolean arrays should remain unchanged
        assert np.array_equal(when1, when1_orig)
        assert np.array_equal(when2, when2_orig)

    def test_known_modulation(self):
        """Test with traces that have known modulation patterns."""
        # Create data where first column is clearly modulated up in state1
        # and second column is clearly modulated down
        traces = np.zeros((20, 2), dtype=np.float64)
        traces[0:10, 0] = 10
        traces[10:20, 1] = 10

        when1 = np.zeros(20, dtype=bool)
        when1[0:10] = True

        when2 = np.zeros(20, dtype=bool)
        when2[10:20] = True

        result = find_two_state_modulated_neurons(
            traces, when1, when2, alpha=0.05, n_shuffle=50
        )
        # First neuron should be up-modulated in state1 vs state2
        assert 0 in result["up_modulated_neurons"]
        # Second neuron should be down-modulated in state1 vs state2
        assert 1 in result["down_modulated_neurons"]


# Add this test only if you can access the functions needed
class TestMakeModulationData:
    """Test the _make_modulation_data function
    for generating modulation data with various methods and parameters
    """

    def test_basic_operation(self):
        """Test the function operates correctly with valid input."""
        traces = np.random.rand(10, 5)
        behavior = pd.DataFrame({"state": ["A"] * 5 + ["B"] * 5})
        states = ["A", "B"]
        # Check that each state has the expected keys
        result = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            n_shuffle=10,
            method=Comp.NOT_STATE.value,
        )
        # Check that we have data for each state
        assert "A" in result
        assert "B" in result
        # Check that each state has the expected keys
        for state in states:
            assert "mean_activity" in result[state]
            assert "p_val" in result[state]
            assert "modulation_scores" in result[state]

    def test_different_methods(self):
        """Test different comparison methods."""
        traces = np.random.rand(15, 3)
        behavior = pd.DataFrame({"state": ["A"] * 5 + ["B"] * 5 + ["C"] * 5})
        states = ["A", "B", "C"]

        # Test NOT_STATE method
        result1 = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            n_shuffle=10,
            method=Comp.NOT_STATE.value,
        )
        assert "A" in result1
        assert "B" in result1
        assert "C" in result1

        # Test PAIRWISE method
        result2 = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            n_shuffle=10,
            method=Comp.PAIRWISE.value,
        )
        assert "A" in result2
        assert "B" in result2
        assert "C" in result2

        # Test BASELINE method
        result3 = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            n_shuffle=10,
            method=Comp.BASELINE.value,
            baseline_state="A",
        )
        assert "A" in result3
        assert "B" in result3
        assert "C" in result3

    def test_scaling_methods(self):
        """Test different scaling methods."""
        traces = np.random.rand(15, 3)
        behavior = pd.DataFrame({"state": ["A"] * 5 + ["B"] * 5 + ["C"] * 5})
        states = ["A", "B", "C"]

        # Test normalize scaling
        result1 = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            rescale="normalize",
            n_shuffle=10,
            method=Comp.NOT_STATE.value,
        )

        # Test standardize scaling
        result2 = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            rescale="standardize",
            n_shuffle=10,
            method=Comp.NOT_STATE.value,
        )
        # Both should produce results with the same structure
        for state in states:
            assert "mean_activity" in result1[state]
            assert "mean_activity" in result2[state]

    def test_error_handling(self, output_file_cleanup):
        """Test error handling for invalid parameters."""
        traces = np.random.rand(15, 3)
        behavior = pd.DataFrame({"state": ["A"] * 5 + ["B"] * 5 + ["C"] * 5})
        states = ["A", "B", "C"]

        # Test invalid rescale method
        with pytest.raises(
            IdeasError, match="Rescale method invalid_method is not supported"
        ):
            _make_modulation_data(
                traces=traces,
                behavior=behavior,
                states=states,
                column_name="state",
                rescale="invalid_method",
                n_shuffle=10,
                method=Comp.NOT_STATE.value,
            )

        # Test invalid comparison method
        with pytest.raises(
            IdeasError, match="Method invalid_method is not supported"
        ):
            _make_modulation_data(
                traces=traces,
                behavior=behavior,
                states=states,
                column_name="state",
                n_shuffle=10,
                method="invalid_method",
            )

    def test_not_defined_method(self):
        """Test state vs not defined method."""
        traces = np.random.rand(25, 3)
        # Use enough NaN values (at least 10) to meet the MINIMUM_STATE_LENGTH requirement
        behavior = pd.DataFrame(
            {"state": ["A"] * 5 + ["B"] * 5 + [np.nan] * 15}
        )
        states = ["A", "B"]

        result = _make_modulation_data(
            traces=traces,
            behavior=behavior,
            states=states,
            column_name="state",
            n_shuffle=10,
            method=Comp.NOT_DEFINED.value,
        )
        # Check that we have results for each state compared to "not defined" state
        assert "A" in result
        assert "B" in result

        # Check that the structure of results is as expected
        for state in states:
            assert "up_modulated_neurons" in result[state]
            assert "down_modulated_neurons" in result[state]
            assert "p_val" in result[state]
            assert "modulation_scores" in result[state]

    def test_not_enough_frames_raises_error(self, output_file_cleanup):
        """Test that error is raised when not enough frames are available."""
        traces = np.random.rand(15, 3)
        behavior = pd.DataFrame({"state": ["A"] * 7 + ["B"] * 8})
        states = ["A", "B"]

        with pytest.raises(
            IdeasError, match="Not enough data to create a 'not_defined' state"
        ):
            _make_modulation_data(
                traces=traces,
                behavior=behavior,
                states=states,
                column_name="state",
                n_shuffle=10,
                method=Comp.NOT_DEFINED.value,
            )

    def test_with_different_rescale_methods(self):
        """Test all rescale methods in _make_modulation_data."""
        from utils.utils import Rescale

        # Use larger dataset with 500 timepoints and 50 neurons
        np.random.seed(42)
        n_timepoints, n_neurons = 500, 50
        traces = np.random.gamma(2, 1, size=(n_timepoints, n_neurons))

        # Create different states with distinct activity patterns
        state_a_indices = np.arange(0, 150)
        state_b_indices = np.arange(150, 300)
        state_c_indices = np.arange(300, 500)

        # Make some neurons respond differently in different states
        traces[state_a_indices, :15] *= 2.0
        traces[state_b_indices, 15:30] *= 2.0
        traces[state_c_indices, 30:45] *= 2.0

        states = ["A"] * 150 + ["B"] * 150 + ["C"] * 200
        behavior = pd.DataFrame({"state": states})
        state_names = ["A", "B", "C"]
        column_name = "state"

        # Test all scaling methods with large dataset
        for rescale_method in [
            Rescale.NONE.value,
            Rescale.NORMALIZE.value,
            Rescale.STANDARDIZE.value,
        ]:
            result = _make_modulation_data(
                traces=traces,
                behavior=behavior,
                states=state_names,
                column_name=column_name,
                rescale=rescale_method,
                n_shuffle=50,
                method=Comp.NOT_STATE.value,
            )

            # Verify structure and content validity
            for state in state_names:
                assert "mean_activity" in result[state]
                assert "p_val" in result[state]
                assert "modulation_scores" in result[state]
                assert len(result[state]["modulation_scores"]) == n_neurons

                # Check that values are finite
                assert np.all(np.isfinite(result[state]["mean_activity"]))
                assert np.all(np.isfinite(result[state]["modulation_scores"]))

        # Test baseline-dependent scaling methods with large dataset
        baseline_state = "A"
        for rescale_method in [
            Rescale.FRACTIONAL_CHANGE.value,
            Rescale.STANDARDIZE_BASELINE.value,
        ]:
            result = _make_modulation_data(
                traces=traces,
                behavior=behavior,
                states=state_names,
                column_name=column_name,
                rescale=rescale_method,
                n_shuffle=50,
                method=Comp.BASELINE.value,
                baseline_state=baseline_state,
            )

            # Verification for non-baseline states
            for state in [s for s in state_names if s != baseline_state]:
                assert "mean_activity" in result[state]
                assert "modulation_scores" in result[state]
                assert len(result[state]["modulation_scores"]) == n_neurons

                # For fractional change, values should reflect relative changes
                if rescale_method == Rescale.FRACTIONAL_CHANGE.value:
                    # For state B, neurons 15-30 should show positive modulation vs baseline
                    if state == "B":
                        assert (
                            np.mean(result[state]["mean_activity"][15:30])
                            > 0.5
                        )


class TestModulationDataToCSV:
    """Test the _modulation_data_to_csv function for generating correctly formatted CSV output."""

    def test_pairwise_method(self, temp_work_dir):
        """Test CSV generation with pairwise method."""
        data = {
            "state1": {
                "mean_activity": np.array([1.0, 2.0, 3.0]),
                "state2": {
                    "up_modulated_neurons": np.array([0]),
                    "down_modulated_neurons": np.array([1]),
                    "p_val": np.array([0.01, 0.02, 0.03]),
                    "modulation_scores": np.array([0.5, -0.5, 0.1]),
                },
            }
        }

        cell_names = ["cell1", "cell2", "cell3"]
        filename = "test_data.csv"

        _modulation_data_to_csv(
            data=data,
            filename=filename,
            cell_names=cell_names,
            method=Comp.PAIRWISE.value,
            baseline_state=None,
            unit="test_unit",
        )
        # Check if file exists
        assert os.path.exists(filename)

        # Read the CSV and verify contents
        df = pd.read_csv(filename)
        assert len(df) == 3
        assert "modulation scores in state1 vs state2" in df.columns
        assert "p-values in state1 vs state2" in df.columns
        assert "modulation in state1 vs state2" in df.columns
        assert "mean test_unit in state1" in df.columns

    def test_not_state_method(self, temp_work_dir):
        """Test CSV generation with state vs not state method."""
        data = {
            "state1": {
                "up_modulated_neurons": np.array([0]),
                "down_modulated_neurons": np.array([1]),
                "p_val": np.array([0.01, 0.02, 0.03]),
                "modulation_scores": np.array([0.5, -0.5, 0.1]),
                "mean_activity": np.array([1.0, 2.0, 3.0]),
            }
        }

        cell_names = ["cell1", "cell2", "cell3"]
        filename = "test_data.csv"

        _modulation_data_to_csv(
            data=data,
            filename=filename,
            cell_names=cell_names,
            method=Comp.NOT_STATE.value,
            baseline_state=None,
            unit="test_unit",
        )
        # Check if file exists
        assert os.path.exists(filename)

        # Read the CSV and verify contents
        df = pd.read_csv(filename)
        assert len(df) == 3
        assert "modulation scores in state1" in df.columns
        assert "p-values in state1" in df.columns
        assert "modulation in state1" in df.columns
        assert "mean test_unit in state1" in df.columns

    def test_csv_structure_with_different_methods(self, temp_work_dir):
        """Test that CSV structure is correct with different comparison methods."""
        cell_names = ["cell1", "cell2", "cell3"]

        # Test NOT_STATE method
        data_not_state = {
            "state1": {
                "up_modulated_neurons": np.array([0]),
                "down_modulated_neurons": np.array([1]),
                "p_val": np.array([0.01, 0.02, 0.03]),
                "modulation_scores": np.array([0.5, -0.5, 0.1]),
                "mean_activity": np.array([1.0, 2.0, 3.0]),
            }
        }

        filename_not_state = "not_state.csv"
        _modulation_data_to_csv(
            data=data_not_state,
            filename=filename_not_state,
            cell_names=cell_names,
            method=Comp.NOT_STATE.value,
            baseline_state=None,
        )

        df_not_state = pd.read_csv(filename_not_state)
        assert "modulation in state1" in df_not_state.columns

        # Test BASELINE method
        data_baseline = {
            "state1": {"mean_activity": np.array([1.0, 2.0, 3.0])},
            "state2": {
                "up_modulated_neurons": np.array([0]),
                "down_modulated_neurons": np.array([1]),
                "p_val": np.array([0.01, 0.02, 0.03]),
                "modulation_scores": np.array([0.5, -0.5, 0.1]),
                "mean_activity": np.array([2.0, 1.0, 3.0]),
            },
        }

        filename_baseline = "baseline.csv"
        _modulation_data_to_csv(
            data=data_baseline,
            filename=filename_baseline,
            cell_names=cell_names,
            method=Comp.BASELINE.value,
            baseline_state="state1",
        )

        df_baseline = pd.read_csv(filename_baseline)
        assert "modulation in state2" in df_baseline.columns
        assert "mean activity (a.u.) in state2" in df_baseline.columns


class TestValidationFunctions:
    """Test parameter validation functions to ensure proper error handling with invalid inputs."""

    def test_validate_params_valid(self):
        """Test valid parameter combinations."""
        from utils.validation import _validate_correlation_params

        # Valid parameters for "state vs not state" method
        _validate_correlation_params(
            method=Comp.NOT_STATE.value,
            state_names=["state1", "state2"],
            baseline_state=None,
            trace_scale_method="none",
            event_scale_method="none",
        )

        # Valid parameters for "state vs baseline" method
        _validate_correlation_params(
            method=Comp.BASELINE.value,
            state_names=["state1", "state2"],
            baseline_state="state1",
            trace_scale_method="none",
            event_scale_method="none",
        )

    def test_validate_params_invalid(self):
        """Test invalid parameter combinations."""
        from utils.validation import _validate_correlation_params
        import pytest

        # Invalid: baseline method without baseline state
        with pytest.raises(
            AssertionError, match="Baseline state must be provided"
        ):
            _validate_correlation_params(
                method=Comp.BASELINE.value,
                state_names=["state1", "state2"],
                baseline_state=None,
                trace_scale_method="none",
                event_scale_method="none",
            )

        # Invalid: baseline method with invalid baseline state
        with pytest.raises(
            AssertionError, match="Baseline state must be one of the states"
        ):
            _validate_correlation_params(
                method=Comp.BASELINE.value,
                state_names=["state1", "state2"],
                baseline_state="invalid_state",
                trace_scale_method="none",
                event_scale_method="none",
            )

        # Invalid: pairwise method with only one state
        with pytest.raises(
            AssertionError,
            match="If using pairwise, must have at least 2 states",
        ):
            _validate_correlation_params(
                method=Comp.PAIRWISE.value,
                state_names=["state1"],
                baseline_state=None,
                trace_scale_method="none",
                event_scale_method="none",
            )

        # Invalid: standardize_baseline without baseline state
        with pytest.raises(
            AssertionError, match="Baseline state must be provided"
        ):
            _validate_correlation_params(
                method=Comp.NOT_STATE.value,
                state_names=["state1", "state2"],
                baseline_state=None,
                trace_scale_method="standardize_baseline",
                event_scale_method="none",
            )


def test_modulation_reproducibility():
    """Test that modulation calculations produce the same results across repeated runs."""
    # Create synthetic test data
    np.random.seed(42)
    num_timepoints, num_cells = 1000, 20
    traces = np.random.normal(0, 1, (num_timepoints, num_cells))

    # Create state masks
    when1 = np.zeros(num_timepoints, dtype=bool)
    when2 = np.zeros(num_timepoints, dtype=bool)
    when1[100:300] = True
    when2[600:800] = True

    # Run the function multiple times
    results1 = find_two_state_modulated_neurons(traces, when1, when2)
    results2 = find_two_state_modulated_neurons(traces, when1, when2)

    # Assert results are identical
    np.testing.assert_array_equal(
        results1["modulation_scores"], results2["modulation_scores"]
    )
    np.testing.assert_array_equal(results1["p_val"], results2["p_val"])
    np.testing.assert_array_equal(
        results1["up_modulated_neurons"], results2["up_modulated_neurons"]
    )
    np.testing.assert_array_equal(
        results1["down_modulated_neurons"], results2["down_modulated_neurons"]
    )
