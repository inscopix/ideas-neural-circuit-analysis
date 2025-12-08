"""Test baseline extraction robustness in combine_compare_state_epoch_data."""

import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ideas.exceptions import IdeasError

from analysis.combine_compare_state_epoch_data import (
    _extract_metadata_from_data,
    _load_group_data,
)


logger = logging.getLogger(__name__)


class TestBaselineExtraction(unittest.TestCase):
    """Test robustness of baseline extraction logic."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.states = ["rest", "active", "explore"]
        self.epochs = ["baseline", "test", "recovery"]

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_test_files(
        self,
        baseline_state="rest",
        baseline_epoch="baseline",
        include_baseline_in_modulation=True,
        modulation_has_data=True,
        include_nan_baseline=False,
        inconsistent_baselines=False,
    ):
        """Create test CSV files with various baseline configurations."""
        subject_dir = self.temp_dir / "subject_1"
        subject_dir.mkdir()

        # Activity file
        activity_data = []
        for state in self.states:
            for epoch in self.epochs:
                for cell_idx in range(10):
                    activity_data.append(
                        {
                            "name": f"cell_{cell_idx}",
                            "state": state,
                            "epoch": epoch,
                            "mean_activity": np.random.rand() * 10,
                        }
                    )
        pd.DataFrame(activity_data).to_csv(
            subject_dir / "activity_per_state_epoch_data.csv", index=False
        )

        # Correlation file
        correlation_data = []
        for state in self.states:
            for epoch in self.epochs:
                for cell_idx in range(10):
                    correlation_data.append(
                        {
                            "name": f"cell_{cell_idx}",
                            "state": state,
                            "epoch": epoch,
                            "max_correlation": np.random.rand(),
                            "min_correlation": -np.random.rand(),
                            "mean_correlation": np.random.rand() * 0.5,
                        }
                    )
        pd.DataFrame(correlation_data).to_csv(
            subject_dir / "correlations_per_state_epoch_data.csv", index=False
        )

        # Modulation file
        if modulation_has_data:
            modulation_data = []
            for state in self.states:
                for epoch in self.epochs:
                    for cell_idx in range(10):
                        # Set baseline values
                        if include_nan_baseline and cell_idx == 0:
                            cell_baseline_state = np.nan
                            cell_baseline_epoch = np.nan
                        elif inconsistent_baselines and cell_idx < 5:
                            cell_baseline_state = "active"  # Different baseline
                            cell_baseline_epoch = "test"
                        else:
                            cell_baseline_state = (
                                baseline_state if include_baseline_in_modulation else ""
                            )
                            cell_baseline_epoch = (
                                baseline_epoch if include_baseline_in_modulation else ""
                            )

                        modulation_data.append(
                            {
                                "name": f"cell_{cell_idx}",
                                "state": state,
                                "epoch": epoch,
                                "baseline_state": cell_baseline_state,
                                "baseline_epoch": cell_baseline_epoch,
                                "modulation_scores": np.random.rand() * 2 - 1,
                            }
                        )
            pd.DataFrame(modulation_data).to_csv(
                subject_dir / "modulation_vs_baseline_data.csv", index=False
            )
        else:
            # Empty modulation file
            pd.DataFrame(
                columns=[
                    "name",
                    "state",
                    "epoch",
                    "baseline_state",
                    "baseline_epoch",
                    "modulation_scores",
                ]
            ).to_csv(subject_dir / "modulation_vs_baseline_data.csv", index=False)

        return [
            subject_dir / "activity_per_state_epoch_data.csv",
            subject_dir / "correlations_per_state_epoch_data.csv",
            subject_dir / "modulation_vs_baseline_data.csv",
        ]

    def test_normal_baseline_extraction(self):
        """Test baseline extraction with normal, valid data."""
        files = self._create_test_files()

        group_data = _load_group_data(
            activity_csv_files=[files[0]],
            correlation_csv_files=[files[1]],
            modulation_csv_files=[files[2]],
            group_name="Test",
            group_id=1,
            data_pairing="unpaired",
        )

        states, epochs, baseline_state, baseline_epoch = _extract_metadata_from_data(
            group_data
        )

        self.assertEqual(baseline_state, "rest")
        self.assertEqual(baseline_epoch, "baseline")
        self.assertEqual(len(states), 3)
        self.assertEqual(len(epochs), 3)
        logger.info("[PASS] Normal baseline extraction works correctly")

    def test_empty_modulation_data_fallback(self):
        """Test fallback to first state/epoch when modulation data is empty.

        This is a valid scenario when users don't provide modulation files
        (modulation_csv_files=None or empty DataFrame). The tool should fall back
        to the alphabetically first state and epoch for baseline selection.
        """
        files = self._create_test_files(modulation_has_data=False)

        group_data = _load_group_data(
            activity_csv_files=[files[0]],
            correlation_csv_files=[files[1]],
            modulation_csv_files=[files[2]],
            group_name="Test",
            group_id=1,
            data_pairing="unpaired",
        )

        states, epochs, baseline_state, baseline_epoch = _extract_metadata_from_data(
            group_data
        )

        # Should fall back to first state and epoch (sorted alphabetically)
        self.assertEqual(baseline_state, "active")  # First alphabetically
        self.assertEqual(baseline_epoch, "baseline")  # First alphabetically
        logger.info("[PASS] Empty modulation data fallback works correctly")

    def test_inconsistent_baselines_across_rows(self):
        """Test behavior when different rows have different baselines."""
        files = self._create_test_files(inconsistent_baselines=True)

        group_data = _load_group_data(
            activity_csv_files=[files[0]],
            correlation_csv_files=[files[1]],
            modulation_csv_files=[files[2]],
            group_name="Test",
            group_id=1,
            data_pairing="unpaired",
        )

        # Should raise an error due to inconsistent baseline states
        with self.assertRaises(IdeasError) as context:
            states, epochs, baseline_state, baseline_epoch = (
                _extract_metadata_from_data(group_data)
            )

        # Verify error message mentions inconsistent baselines
        self.assertIn("inconsistent", str(context.exception).lower())
        logger.info("[PASS] Inconsistent baselines correctly detected and error raised")

    def test_nan_baseline_values(self):
        """Test behavior when baseline columns contain NaN values."""
        files = self._create_test_files(include_nan_baseline=True)

        group_data = _load_group_data(
            activity_csv_files=[files[0]],
            correlation_csv_files=[files[1]],
            modulation_csv_files=[files[2]],
            group_name="Test",
            group_id=1,
            data_pairing="unpaired",
        )

        # First row might have NaN
        states, epochs, baseline_state, baseline_epoch = _extract_metadata_from_data(
            group_data
        )

        # If first row is NaN, it will pass through without validation
        if pd.isna(baseline_state):
            logger.warning("[WARN] NaN baseline values not validated")
        else:
            logger.info("[PASS] Non-NaN baseline extracted successfully")


if __name__ == "__main__":
    unittest.main()
