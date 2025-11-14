"""Comprehensive validation script for test data consistency.

This script validates that all test data files are consistent with the expected
output format from state_epoch_baseline_analysis and can be properly loaded by
combine_compare_state_epoch_data.

Checks include:
1. File existence and structure
2. Required columns presence
3. Data type consistency
4. Row count validation
5. Sparse data scenario validation
6. Cross-file consistency checks
"""

import pandas as pd
from pathlib import Path
from typing import List


class TestDataValidator:
    """Validator for state-epoch test data."""

    # Expected dimensions
    N_CELLS = 150
    N_STATES = 3
    N_EPOCHS = 3
    N_COMBINATIONS = N_STATES * N_EPOCHS
    N_NON_BASELINE = N_COMBINATIONS - 1

    # Expected file names
    ACTIVITY_FILE = "activity_per_state_epoch_data.csv"
    CORRELATION_FILE = "correlations_per_state_epoch_data.csv"
    MODULATION_FILE = "modulation_vs_baseline_data.csv"

    # Required columns
    REQUIRED_ACTIVITY_COLS = [
        'name', 'cell_index', 'state', 'epoch',
        'mean_trace_activity', 'std_trace_activity',
        'median_trace_activity', 'trace_activity_cv',
        'mean_event_rate'
    ]

    REQUIRED_CORRELATION_COLS = [
        'name', 'cell_index', 'state', 'epoch',
        'max_trace_correlation', 'min_trace_correlation',
        'mean_trace_correlation',
        'positive_trace_correlation', 'negative_trace_correlation'
    ]

    OPTIONAL_EVENT_CORRELATION_COLS = [
        'max_event_correlation', 'min_event_correlation',
        'mean_event_correlation',
        'positive_event_correlation', 'negative_event_correlation'
    ]

    REQUIRED_MODULATION_COLS = [
        'name', 'cell_index', 'state', 'epoch',
        'baseline_state', 'baseline_epoch'
    ]

    def __init__(self, test_data_dir: Path):
        """Initialize validator with test data directory."""
        self.test_data_dir = test_data_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_all(self) -> bool:
        """Validate all test data and return success status."""
        print("=" * 70)
        print("Test Data Validation")
        print("=" * 70)

        # Find all subject directories
        subject_dirs = [
            d for d in self.test_data_dir.iterdir()
            if d.is_dir() and d.name.startswith('dummy_')
        ]

        if not subject_dirs:
            self.errors.append("No subject directories found!")
            return False

        print(f"\nFound {len(subject_dirs)} subject directories")

        # Validate each subject
        for subject_dir in sorted(subject_dirs):
            self._validate_subject(subject_dir)

        # Print results
        self._print_results()

        return len(self.errors) == 0

    def _validate_subject(self, subject_dir: Path) -> None:
        """Validate a single subject's data files."""
        subject_name = subject_dir.name
        print(f"\n{subject_name}:")
        print("-" * 70)

        # Check if this is a sparse data scenario
        is_sparse_no_events = 'no_events' in subject_name
        is_sparse_partial = 'partial_events' in subject_name
        is_sparse = is_sparse_no_events or is_sparse_partial

        # Check file existence
        activity_path = subject_dir / self.ACTIVITY_FILE
        correlation_path = subject_dir / self.CORRELATION_FILE
        modulation_path = subject_dir / self.MODULATION_FILE

        if not activity_path.exists():
            self.errors.append(
                f"{subject_name}: Missing {self.ACTIVITY_FILE}"
            )
            return

        if not correlation_path.exists():
            self.errors.append(
                f"{subject_name}: Missing {self.CORRELATION_FILE}"
            )
            return

        if not modulation_path.exists():
            self.errors.append(
                f"{subject_name}: Missing {self.MODULATION_FILE}"
            )
            return

        # Load data files
        try:
            activity_df = pd.read_csv(activity_path)
            correlation_df = pd.read_csv(correlation_path)
            modulation_df = pd.read_csv(modulation_path)
        except Exception as e:
            self.errors.append(
                f"{subject_name}: Failed to load CSV files: {e}"
            )
            return

        # Validate activity file
        self._validate_activity_file(
            subject_name, activity_df, is_sparse
        )

        # Validate correlation file
        self._validate_correlation_file(
            subject_name, correlation_df, is_sparse_no_events, is_sparse_partial
        )

        # Validate modulation file
        self._validate_modulation_file(
            subject_name, modulation_df, is_sparse_no_events
        )

        # Cross-file consistency checks
        self._validate_cross_file_consistency(
            subject_name, activity_df, correlation_df, modulation_df
        )

    def _validate_activity_file(
        self, subject: str, df: pd.DataFrame, is_sparse: bool
    ) -> None:
        """Validate activity CSV file."""
        # Check row count
        expected_rows = self.N_CELLS * self.N_COMBINATIONS
        if len(df) != expected_rows:
            self.errors.append(
                f"{subject}: Activity file has {len(df)} rows, "
                f"expected {expected_rows}"
            )
        else:
            print(f"  SUCCESS: Activity rows: {len(df)}")

        # Check required columns
        missing_cols = [
            col for col in self.REQUIRED_ACTIVITY_COLS
            if col not in df.columns
        ]
        if missing_cols:
            self.errors.append(
                f"{subject}: Activity file missing columns: {missing_cols}"
            )
        else:
            print("  SUCCESS: All required activity columns present")

        # Check data types
        if 'cell_index' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['cell_index']):
                self.warnings.append(
                    f"{subject}: cell_index should be numeric"
                )

        # Check for NaN in required columns
        required_data_cols = [
            'mean_trace_activity', 'mean_event_rate'
        ]
        for col in required_data_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    self.warnings.append(
                        f"{subject}: Activity {col} has {nan_count} NaN values"
                    )

    def _validate_correlation_file(
        self,
        subject: str,
        df: pd.DataFrame,
        is_sparse_no_events: bool,
        is_sparse_partial: bool
    ) -> None:
        """Validate correlation CSV file."""
        # Check row count
        expected_rows = self.N_CELLS * self.N_COMBINATIONS
        if len(df) != expected_rows:
            self.errors.append(
                f"{subject}: Correlation file has {len(df)} rows, "
                f"expected {expected_rows}"
            )
        else:
            print(f"  SUCCESS: Correlation rows: {len(df)}")

        # Check required trace correlation columns
        missing_cols = [
            col for col in self.REQUIRED_CORRELATION_COLS
            if col not in df.columns
        ]
        if missing_cols:
            self.errors.append(
                f"{subject}: Correlation file missing columns: {missing_cols}"
            )
        else:
            print("  SUCCESS: All required trace correlation columns present")

        # Check event correlation columns
        has_event_cols = all(
            col in df.columns
            for col in self.OPTIONAL_EVENT_CORRELATION_COLS
        )

        if is_sparse_no_events:
            # Should NOT have event correlation columns
            if has_event_cols:
                self.errors.append(
                    f"{subject}: Sparse 'no_events' subject should not have "
                    "event correlation columns"
                )
            else:
                print("  SUCCESS: Event correlation columns properly removed "
                      "(sparse scenario)")
        elif has_event_cols:
            print("  SUCCESS: Event correlation columns present")

            # Check for NaN values in partial sparse scenario
            if is_sparse_partial:
                nan_count = df['max_event_correlation'].isna().sum()
                if nan_count > 0:
                    print(f"  SUCCESS: Partial event data has {nan_count} "
                          "NaN values (sparse scenario)")
                else:
                    self.warnings.append(
                        f"{subject}: Partial events scenario should have some "
                        "NaN values"
                    )
        else:
            self.info.append(
                f"{subject}: Event correlation columns missing (optional)"
            )

    def _validate_modulation_file(
        self, subject: str, df: pd.DataFrame, is_sparse_no_events: bool
    ) -> None:
        """Validate modulation CSV file."""
        # Check row count (excludes baseline combination)
        expected_rows = self.N_CELLS * self.N_NON_BASELINE
        if len(df) != expected_rows:
            self.errors.append(
                f"{subject}: Modulation file has {len(df)} rows, "
                f"expected {expected_rows}"
            )
        else:
            print(f"  SUCCESS: Modulation rows: {len(df)}")

        # Check required columns
        missing_cols = [
            col for col in self.REQUIRED_MODULATION_COLS
            if col not in df.columns
        ]
        if missing_cols:
            self.errors.append(
                f"{subject}: Modulation file missing columns: {missing_cols}"
            )
        else:
            print("  SUCCESS: All required modulation columns present")

        # Check for trace modulation columns
        trace_mod_cols = [
            col for col in df.columns
            if col.startswith('trace_modulation')
        ]
        if not trace_mod_cols:
            self.errors.append(
                f"{subject}: No trace modulation columns found"
            )
        else:
            print(f"  SUCCESS: Found {len(trace_mod_cols)} trace modulation columns")

        # Check for event modulation columns
        event_mod_cols = [
            col for col in df.columns
            if col.startswith('event_modulation')
        ]

        if is_sparse_no_events:
            if event_mod_cols:
                self.errors.append(
                    f"{subject}: Sparse 'no_events' subject should not have "
                    "event modulation columns"
                )
            else:
                print("  SUCCESS: Event modulation columns properly removed "
                      "(sparse scenario)")
        elif event_mod_cols:
            print(f"  SUCCESS: Found {len(event_mod_cols)} event modulation columns")
        else:
            self.info.append(
                f"{subject}: No event modulation columns (may be sparse)"
            )

        # Validate baseline consistency
        if 'baseline_state' in df.columns and 'baseline_epoch' in df.columns:
            baseline_states = df['baseline_state'].unique()
            baseline_epochs = df['baseline_epoch'].unique()

            if len(baseline_states) != 1:
                self.errors.append(
                    f"{subject}: Multiple baseline states: {baseline_states}"
                )

            if len(baseline_epochs) != 1:
                self.errors.append(
                    f"{subject}: Multiple baseline epochs: {baseline_epochs}"
                )

    def _validate_cross_file_consistency(
        self,
        subject: str,
        activity_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        modulation_df: pd.DataFrame
    ) -> None:
        """Validate consistency across CSV files."""
        # Check cell counts match
        activity_cells = activity_df['name'].nunique()
        correlation_cells = correlation_df['name'].nunique()

        if activity_cells != correlation_cells:
            self.errors.append(
                f"{subject}: Cell count mismatch between activity "
                f"({activity_cells}) and correlation ({correlation_cells})"
            )

        if activity_cells != self.N_CELLS:
            self.warnings.append(
                f"{subject}: Expected {self.N_CELLS} cells, "
                f"found {activity_cells}"
            )

        # Check state-epoch combinations match
        activity_combos = set(
            zip(activity_df['state'], activity_df['epoch'])
        )
        correlation_combos = set(
            zip(correlation_df['state'], correlation_df['epoch'])
        )

        if activity_combos != correlation_combos:
            self.warnings.append(
                f"{subject}: State-epoch combinations differ between "
                "activity and correlation files"
            )

    def _print_results(self) -> None:
        """Print validation results."""
        print("\n" + "=" * 70)
        print("Validation Results")
        print("=" * 70)

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ❌ {error}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")

        if self.info:
            print(f"\nINFO ({len(self.info)}):")
            for info_msg in self.info:
                print(f"  ℹ️  {info_msg}")

        if not self.errors and not self.warnings:
            print("\n✅ All validation checks passed!")
        elif not self.errors:
            print("\n⚠️  Validation passed with warnings")
        else:
            print(f"\n❌ Validation failed with {len(self.errors)} error(s)")


def main():
    """Run validation on all test data."""
    test_data_dir = Path(__file__).parent

    validator = TestDataValidator(test_data_dir)
    success = validator.validate_all()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
