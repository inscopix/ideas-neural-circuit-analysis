"""Unit tests for combine_compare_state_epoch_data.py module.

This module tests functionality for combining and comparing state/epoch data,
with a focus on the new modulation count columns feature.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import types
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Provide lightweight stubs for optional ideas.* dependencies used by analysis code.
# Only register stubs when the real package (or submodule) is unavailable.
try:
    import ideas as ideas_module  # type: ignore
except ImportError:  # pragma: no cover - executed only when ideas pkg missing
    ideas_module = types.ModuleType("ideas")
    sys.modules["ideas"] = ideas_module

try:
    from ideas import outputs as _ideas_outputs  # type: ignore
except ImportError:  # pragma: no cover - executed only when outputs missing
    outputs_module = types.ModuleType("ideas.outputs")

    class OutputData:  # pragma: no cover - simple stub
        pass

    outputs_module.OutputData = OutputData
    ideas_module.outputs = outputs_module
    sys.modules["ideas.outputs"] = outputs_module

try:
    from ideas import plots as _ideas_plots  # type: ignore
except ImportError:  # pragma: no cover - executed only when plots missing
    plots_module = types.ModuleType("ideas.plots")

    def plot_shaded_hist(*_args, **_kwargs):  # pragma: no cover - simple stub
        return None

    plots_module.plot_shaded_hist = plot_shaded_hist
    ideas_module.plots = plots_module
    sys.modules["ideas.plots"] = plots_module

try:
    from ideas.exceptions import IdeasError  # type: ignore
except ImportError:  # pragma: no cover - executed only when exceptions missing
    class IdeasError(Exception):  # simple stub
        pass

    exceptions_module = types.ModuleType("ideas.exceptions")
    exceptions_module.IdeasError = IdeasError
    ideas_module.exceptions = exceptions_module
    sys.modules["ideas.exceptions"] = exceptions_module

from analysis.combine_compare_state_epoch_data import (  # noqa: E402
    _add_modulation_count_columns,
)
from utils.state_epoch_comparison_utils import _detect_measure_column  # noqa: E402


class TestAddModulationCountColumns:
    """Tests for the _add_modulation_count_columns function."""

    def test_add_counts_with_trace_modulation_only(self):
        """Test adding count columns when only trace_modulation is present."""
        # Create sample data with trace modulation
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1', 'subject1', 'subject2', 'subject2'],
            'state': ['awake', 'awake', 'awake', 'awake', 'awake'],
            'epoch': ['baseline', 'baseline', 'baseline', 'baseline', 'baseline'],
            'cell_id': [1, 2, 3, 1, 2],
            'trace_modulation': [1, -1, 0, 1, 1]  # 2 up, 1 down, 1 non-modulated
        })

        result = _add_modulation_count_columns(df)

        # Check that count columns were added
        assert 'trace_up_modulation_number' in result.columns
        assert 'trace_down_modulation_number' in result.columns

        # Verify counts for subject1 (2 up, 1 down)
        subject1_rows = result[result['normalized_subject_id'] == 'subject1']
        assert (subject1_rows['trace_up_modulation_number'] == 1).all()
        assert (subject1_rows['trace_down_modulation_number'] == 1).all()

        # Verify counts for subject2 (2 up, 0 down)
        subject2_rows = result[result['normalized_subject_id'] == 'subject2']
        assert (subject2_rows['trace_up_modulation_number'] == 2).all()
        assert (subject2_rows['trace_down_modulation_number'] == 0).all()

    def test_add_counts_with_event_modulation_only(self):
        """Test adding count columns when only event_modulation is present."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1', 'subject1'],
            'state': ['awake', 'awake', 'awake'],
            'epoch': ['baseline', 'baseline', 'baseline'],
            'cell_id': [1, 2, 3],
            'event_modulation': [-1, -1, 1]  # 1 up, 2 down
        })

        result = _add_modulation_count_columns(df)

        # Check that event count columns were added
        assert 'event_up_modulation_number' in result.columns
        assert 'event_down_modulation_number' in result.columns

        # Verify counts
        assert (result['event_up_modulation_number'] == 1).all()
        assert (result['event_down_modulation_number'] == 2).all()

    def test_add_counts_with_both_trace_and_event_modulation(self):
        """Test adding count columns with both trace and event modulation."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1', 'subject1'],
            'state': ['awake', 'awake', 'awake'],
            'epoch': ['baseline', 'baseline', 'baseline'],
            'cell_id': [1, 2, 3],
            'trace_modulation': [1, 1, -1],  # 2 up, 1 down
            'event_modulation': [1, -1, -1]  # 1 up, 2 down
        })

        result = _add_modulation_count_columns(df)

        # Check that all count columns were added
        assert 'trace_up_modulation_number' in result.columns
        assert 'trace_down_modulation_number' in result.columns
        assert 'event_up_modulation_number' in result.columns
        assert 'event_down_modulation_number' in result.columns

        # Verify trace counts
        assert (result['trace_up_modulation_number'] == 2).all()
        assert (result['trace_down_modulation_number'] == 1).all()

        # Verify event counts
        assert (result['event_up_modulation_number'] == 1).all()
        assert (result['event_down_modulation_number'] == 2).all()

    def test_add_counts_with_multiple_states_and_epochs(self):
        """Test that counts are computed correctly per state/epoch grouping."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 6,
            'state': ['awake', 'awake', 'awake', 'sleep', 'sleep', 'sleep'],
            'epoch': ['baseline', 'baseline', 'stim', 'baseline', 'baseline', 'stim'],
            'cell_id': [1, 2, 1, 1, 2, 1],
            'trace_modulation': [1, -1, 1, -1, -1, 1]
        })

        result = _add_modulation_count_columns(df)

        # Check awake-baseline: 1 up, 1 down
        awake_baseline = result[
            (result['state'] == 'awake') & (result['epoch'] == 'baseline')
        ]
        assert (awake_baseline['trace_up_modulation_number'] == 1).all()
        assert (awake_baseline['trace_down_modulation_number'] == 1).all()

        # Check awake-stim: 1 up, 0 down
        awake_stim = result[
            (result['state'] == 'awake') & (result['epoch'] == 'stim')
        ]
        assert (awake_stim['trace_up_modulation_number'] == 1).all()
        assert (awake_stim['trace_down_modulation_number'] == 0).all()

        # Check sleep-baseline: 0 up, 2 down
        sleep_baseline = result[
            (result['state'] == 'sleep') & (result['epoch'] == 'baseline')
        ]
        assert (sleep_baseline['trace_up_modulation_number'] == 0).all()
        assert (sleep_baseline['trace_down_modulation_number'] == 2).all()

        # Check sleep-stim: 1 up, 0 down
        sleep_stim = result[
            (result['state'] == 'sleep') & (result['epoch'] == 'stim')
        ]
        assert (sleep_stim['trace_up_modulation_number'] == 1).all()
        assert (sleep_stim['trace_down_modulation_number'] == 0).all()

    def test_add_counts_with_empty_dataframe(self):
        """Test that empty dataframes are handled correctly."""
        df = pd.DataFrame()

        result = _add_modulation_count_columns(df)

        # Should return empty dataframe unchanged
        assert result.empty

    def test_add_counts_with_none_input(self):
        """Test that None input is handled correctly."""
        result = _add_modulation_count_columns(None)

        # Should return None
        assert result is None

    def test_add_counts_with_missing_required_columns(self):
        """Test that function handles missing required columns gracefully."""
        # Missing 'state' column
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1'],
            'epoch': ['baseline', 'baseline'],
            'trace_modulation': [1, -1]
        })

        result = _add_modulation_count_columns(df)

        # Should return dataframe unchanged (with warning logged)
        assert 'trace_up_modulation_number' not in result.columns
        assert 'trace_down_modulation_number' not in result.columns

    def test_add_counts_with_all_up_modulated(self):
        """Test with all cells up-modulated."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 3,
            'state': ['awake'] * 3,
            'epoch': ['baseline'] * 3,
            'cell_id': [1, 2, 3],
            'trace_modulation': [1, 1, 1]  # All up-modulated
        })

        result = _add_modulation_count_columns(df)

        assert (result['trace_up_modulation_number'] == 3).all()
        assert (result['trace_down_modulation_number'] == 0).all()

    def test_add_counts_with_all_down_modulated(self):
        """Test with all cells down-modulated."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 3,
            'state': ['awake'] * 3,
            'epoch': ['baseline'] * 3,
            'cell_id': [1, 2, 3],
            'trace_modulation': [-1, -1, -1]  # All down-modulated
        })

        result = _add_modulation_count_columns(df)

        assert (result['trace_up_modulation_number'] == 0).all()
        assert (result['trace_down_modulation_number'] == 3).all()

    def test_add_counts_with_all_non_modulated(self):
        """Test with all cells non-modulated."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 3,
            'state': ['awake'] * 3,
            'epoch': ['baseline'] * 3,
            'cell_id': [1, 2, 3],
            'trace_modulation': [0, 0, 0]  # All non-modulated
        })

        result = _add_modulation_count_columns(df)

        assert (result['trace_up_modulation_number'] == 0).all()
        assert (result['trace_down_modulation_number'] == 0).all()

    def test_add_counts_preserves_original_data(self):
        """Test that original data columns are preserved."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1'],
            'state': ['awake', 'awake'],
            'epoch': ['baseline', 'baseline'],
            'cell_id': [1, 2],
            'trace_modulation': [1, -1],
            'other_column': ['a', 'b']
        })

        result = _add_modulation_count_columns(df)

        # Check that original columns are preserved
        assert 'cell_id' in result.columns
        assert 'other_column' in result.columns
        assert (result['cell_id'] == df['cell_id']).all()
        assert (result['other_column'] == df['other_column']).all()

    def test_add_counts_with_nan_values(self):
        """Test handling of NaN values in modulation columns."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 4,
            'state': ['awake'] * 4,
            'epoch': ['baseline'] * 4,
            'cell_id': [1, 2, 3, 4],
            'trace_modulation': [1, -1, np.nan, 0]  # 1 up, 1 down, 1 NaN, 1 non-modulated
        })

        result = _add_modulation_count_columns(df)

        # NaN values should not be counted as up or down modulated
        assert (result['trace_up_modulation_number'] == 1).all()
        assert (result['trace_down_modulation_number'] == 1).all()

    def test_add_counts_with_multiple_subjects(self):
        """Test that counts are computed separately per subject."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1', 'subject2', 'subject2', 'subject2'],
            'state': ['awake'] * 5,
            'epoch': ['baseline'] * 5,
            'cell_id': [1, 2, 1, 2, 3],
            'trace_modulation': [1, 1, -1, -1, 1]  # subject1: 2 up, subject2: 1 up, 2 down
        })

        result = _add_modulation_count_columns(df)

        # Check subject1: 2 up, 0 down
        subject1_rows = result[result['normalized_subject_id'] == 'subject1']
        assert (subject1_rows['trace_up_modulation_number'] == 2).all()
        assert (subject1_rows['trace_down_modulation_number'] == 0).all()

        # Check subject2: 1 up, 2 down
        subject2_rows = result[result['normalized_subject_id'] == 'subject2']
        assert (subject2_rows['trace_up_modulation_number'] == 1).all()
        assert (subject2_rows['trace_down_modulation_number'] == 2).all()

    def test_add_counts_returns_integer_dtype(self):
        """Test that count columns have integer dtype."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1'],
            'state': ['awake', 'awake'],
            'epoch': ['baseline', 'baseline'],
            'cell_id': [1, 2],
            'trace_modulation': [1, -1]
        })

        result = _add_modulation_count_columns(df)

        # Check that count columns are integers
        assert result['trace_up_modulation_number'].dtype == np.int64
        assert result['trace_down_modulation_number'].dtype == np.int64
    
    def test_add_counts_separate_groups_with_shared_subject_ids(self):
        """Counts should remain separate when groups share normalized IDs (paired design)."""
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1', 'subject1'],
            'state': ['awake', 'awake'],
            'epoch': ['baseline', 'baseline'],
            'group_name': ['group1', 'group2'],
            'group_id': [1, 2],
            'cell_id': [1, 1],
            'trace_modulation': [1, -1],
        })
        
        result = _add_modulation_count_columns(df)
        
        group1_rows = result[result['group_name'] == 'group1']
        group2_rows = result[result['group_name'] == 'group2']
        
        assert (group1_rows['trace_up_modulation_number'] == 1).all()
        assert (group1_rows['trace_down_modulation_number'] == 0).all()
        
        assert (group2_rows['trace_up_modulation_number'] == 0).all()
        assert (group2_rows['trace_down_modulation_number'] == 1).all()


class TestDetectMeasureColumnForModulationCounts:
    """Tests for the updated _detect_measure_column function with modulation counts."""

    def test_detect_trace_up_modulated_counts(self):
        """Test detection of trace up-modulated cell counts."""
        df = pd.DataFrame({
            'trace_up_modulation_number': [5, 3, 7],
            'trace_down_modulation_number': [2, 1, 3]
        })

        # Test various type strings that should map to trace_up_modulation_number
        for data_type in ['trace_up_modulated_counts', 'trace_up_modulation_counts']:
            result = _detect_measure_column(df, data_type)
            assert result == 'trace_up_modulation_number', f"Failed for data_type: {data_type}"

    def test_detect_trace_down_modulated_counts(self):
        """Test detection of trace down-modulated cell counts."""
        df = pd.DataFrame({
            'trace_up_modulation_number': [5, 3, 7],
            'trace_down_modulation_number': [2, 1, 3]
        })

        # Test various type strings that should map to trace_down_modulation_number
        for data_type in ['trace_down_modulated_counts', 'trace_down_modulation_counts']:
            result = _detect_measure_column(df, data_type)
            assert result == 'trace_down_modulation_number', f"Failed for data_type: {data_type}"

    def test_detect_event_up_modulated_counts(self):
        """Test detection of event up-modulated cell counts."""
        df = pd.DataFrame({
            'event_up_modulation_number': [5, 3, 7],
            'event_down_modulation_number': [2, 1, 3]
        })

        # Test various type strings that should map to event_up_modulation_number
        for data_type in ['event_up_modulated_counts', 'event_up_modulation_counts']:
            result = _detect_measure_column(df, data_type)
            assert result == 'event_up_modulation_number', f"Failed for data_type: {data_type}"

    def test_detect_event_down_modulated_counts(self):
        """Test detection of event down-modulated cell counts."""
        df = pd.DataFrame({
            'event_up_modulation_number': [5, 3, 7],
            'event_down_modulation_number': [2, 1, 3]
        })

        # Test various type strings that should map to event_down_modulation_number
        for data_type in ['event_down_modulated_counts', 'event_down_modulation_counts']:
            result = _detect_measure_column(df, data_type)
            assert result == 'event_down_modulation_number', f"Failed for data_type: {data_type}"

    def test_detect_prefers_trace_when_both_available_up(self):
        """Test that trace is preferred when both trace and event up-modulated counts are available."""
        df = pd.DataFrame({
            'trace_up_modulation_number': [5, 3, 7],
            'event_up_modulation_number': [2, 1, 3]
        })

        # Without specifying trace or event, should prefer trace
        result = _detect_measure_column(df, 'up_modulated_counts')
        assert result == 'trace_up_modulation_number'

    def test_detect_prefers_trace_when_both_available_down(self):
        """Test that trace is preferred when both trace and event down-modulated counts are available."""
        df = pd.DataFrame({
            'trace_down_modulation_number': [5, 3, 7],
            'event_down_modulation_number': [2, 1, 3]
        })

        # Without specifying trace or event, should prefer trace
        result = _detect_measure_column(df, 'down_modulated_counts')
        assert result == 'trace_down_modulation_number'

    def test_detect_falls_back_to_event_up(self):
        """Test fallback to event when trace up-modulated counts not available."""
        df = pd.DataFrame({
            'event_up_modulation_number': [5, 3, 7]
        })

        result = _detect_measure_column(df, 'up_modulated_counts')
        assert result == 'event_up_modulation_number'

    def test_detect_falls_back_to_event_down(self):
        """Test fallback to event when trace down-modulated counts not available."""
        df = pd.DataFrame({
            'event_down_modulation_number': [5, 3, 7]
        })

        result = _detect_measure_column(df, 'down_modulated_counts')
        assert result == 'event_down_modulation_number'

    def test_detect_raises_when_column_not_found(self):
        """Test that an IdeasError is raised when no matching count column exists."""
        df = pd.DataFrame({
            'some_other_column': [5, 3, 7]
        })

        with pytest.raises(IdeasError):
            _detect_measure_column(df, 'trace_up_modulated_counts')

    def test_detect_case_insensitive_matching(self):
        """Test that detection works with various case combinations."""
        df = pd.DataFrame({
            'trace_up_modulation_number': [5, 3, 7]
        })

        # Test case variations
        for data_type in ['TRACE_UP_MODULATED_COUNTS', 'Trace_Up_Modulated_Counts']:
            result = _detect_measure_column(df, data_type)
            assert result == 'trace_up_modulation_number', f"Failed for data_type: {data_type}"

    def test_detect_with_underscore_variations(self):
        """Test detection with various underscore patterns."""
        df = pd.DataFrame({
            'trace_up_modulation_number': [5, 3, 7]
        })

        # Various forms that should all work
        for data_type in ['trace_up_modulated_counts', 'traceupmodulatedcounts']:
            result = _detect_measure_column(df, data_type)
            assert result == 'trace_up_modulation_number', f"Failed for data_type: {data_type}"


class TestModulationCountsIntegration:
    """Integration tests for modulation counts workflow."""

    def test_full_workflow_trace_modulation(self):
        """Test complete workflow: add counts, then detect columns."""
        # Start with modulation data
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 3,
            'state': ['awake'] * 3,
            'epoch': ['baseline'] * 3,
            'cell_id': [1, 2, 3],
            'trace_modulation': [1, 1, -1]  # 2 up, 1 down
        })

        # Add count columns
        df_with_counts = _add_modulation_count_columns(df)

        # Verify columns were added
        assert 'trace_up_modulation_number' in df_with_counts.columns
        assert 'trace_down_modulation_number' in df_with_counts.columns

        # Verify detection works
        up_col = _detect_measure_column(df_with_counts, 'trace_up_modulated_counts')
        assert up_col == 'trace_up_modulation_number'

        down_col = _detect_measure_column(df_with_counts, 'trace_down_modulated_counts')
        assert down_col == 'trace_down_modulation_number'

        # Verify counts are correct
        assert (df_with_counts['trace_up_modulation_number'] == 2).all()
        assert (df_with_counts['trace_down_modulation_number'] == 1).all()

    def test_full_workflow_event_modulation(self):
        """Test complete workflow with event modulation."""
        # Start with event modulation data
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 4,
            'state': ['awake'] * 4,
            'epoch': ['baseline'] * 4,
            'cell_id': [1, 2, 3, 4],
            'event_modulation': [-1, -1, -1, 1]  # 1 up, 3 down
        })

        # Add count columns
        df_with_counts = _add_modulation_count_columns(df)

        # Verify columns were added
        assert 'event_up_modulation_number' in df_with_counts.columns
        assert 'event_down_modulation_number' in df_with_counts.columns

        # Verify detection works
        up_col = _detect_measure_column(df_with_counts, 'event_up_modulated_counts')
        assert up_col == 'event_up_modulation_number'

        down_col = _detect_measure_column(df_with_counts, 'event_down_modulated_counts')
        assert down_col == 'event_down_modulation_number'

        # Verify counts are correct
        assert (df_with_counts['event_up_modulation_number'] == 1).all()
        assert (df_with_counts['event_down_modulation_number'] == 3).all()

    def test_counts_update_after_reclassification(self):
        """Test that counts can be recomputed after modulation reclassification."""
        # Initial data
        df = pd.DataFrame({
            'normalized_subject_id': ['subject1'] * 3,
            'state': ['awake'] * 3,
            'epoch': ['baseline'] * 3,
            'cell_id': [1, 2, 3],
            'trace_modulation': [1, 1, -1]  # 2 up, 1 down
        })

        # Add counts
        df_with_counts = _add_modulation_count_columns(df)
        assert (df_with_counts['trace_up_modulation_number'] == 2).all()
        assert (df_with_counts['trace_down_modulation_number'] == 1).all()

        # Simulate reclassification: change one up-modulated to down-modulated
        df_with_counts.loc[df_with_counts['cell_id'] == 1, 'trace_modulation'] = -1

        # Recompute counts
        df_updated = _add_modulation_count_columns(df_with_counts)

        # Verify updated counts (1 up, 2 down)
        assert (df_updated['trace_up_modulation_number'] == 1).all()
        assert (df_updated['trace_down_modulation_number'] == 2).all()

