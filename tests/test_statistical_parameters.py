"""Test that statistical parameters are correctly passed through all functions."""

import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ideas.exceptions import IdeasError
from analysis.combine_compare_state_epoch_data import (
    combine_compare_state_epoch_data,
)


logger = logging.getLogger(__name__)


class TestStatisticalParameterPassing(unittest.TestCase):
    """Verify statistical parameters are passed correctly through call chain."""

    def setUp(self):
        """Create temporary directory and test files."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.states = ["rest", "active"]
        self.epochs = ["baseline", "test"]
        self.n_subjects = 3

        # Create test data for both groups
        self.group1_files = self._create_test_files("group1", 1)
        self.group2_files = self._create_test_files("group2", 2)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def _create_test_files(self, group_name, group_id):
        """Create test CSV files for a group."""
        activity_files = []
        correlation_files = []
        modulation_files = []

        for subject_idx in range(self.n_subjects):
            subject_dir = self.temp_dir / f"{group_name}_subject_{subject_idx}"
            subject_dir.mkdir()

            # Activity data
            activity_data = []
            for state in self.states:
                for epoch in self.epochs:
                    for cell_idx in range(10):
                        activity_data.append(
                            {
                                "name": f"cell_{cell_idx}",
                                "state": state,
                                "epoch": epoch,
                                "mean_trace_activity": np.random.rand() * 10
                                + (group_id * 2),
                                "mean_event_rate": np.random.rand() * 5,
                            }
                        )

            activity_file = subject_dir / "activity_per_state_epoch_data.csv"
            pd.DataFrame(activity_data).to_csv(activity_file, index=False)
            activity_files.append(activity_file)

            # Correlation data
            correlation_data = []
            for state in self.states:
                for epoch in self.epochs:
                    for cell_idx in range(10):
                        correlation_data.append(
                            {
                                "name": f"cell_{cell_idx}",
                                "state": state,
                                "epoch": epoch,
                                "max_trace_correlation": np.random.rand(),
                                "min_trace_correlation": -np.random.rand(),
                                "mean_trace_correlation": np.random.rand() * 0.5,
                            }
                        )

            correlation_file = subject_dir / "correlations_per_state_epoch_data.csv"
            pd.DataFrame(correlation_data).to_csv(correlation_file, index=False)
            correlation_files.append(correlation_file)

            # Modulation data
            modulation_data = []
            for state in self.states:
                for epoch in self.epochs:
                    for cell_idx in range(10):
                        modulation_data.append(
                            {
                                "name": f"cell_{cell_idx}",
                                "state": state,
                                "epoch": epoch,
                                "baseline_state": "rest",
                                "baseline_epoch": "baseline",
                                "modulation_scores": np.random.rand() * 2 - 1,
                            }
                        )

            modulation_file = subject_dir / "modulation_vs_baseline_data.csv"
            pd.DataFrame(modulation_data).to_csv(modulation_file, index=False)
            modulation_files.append(modulation_file)

        return activity_files, correlation_files, modulation_files

    @patch("analysis.combine_compare_state_epoch_data.calculate_state_epoch_lmm_stats")
    @patch(
        "analysis.combine_compare_state_epoch_data.calculate_state_epoch_comparison_stats"
    )
    def test_significance_threshold_passed(self, mock_calc_stats, mock_calc_lmm):
        """Test that custom significance_threshold is passed through."""
        # Setup mock to return empty dataframes
        mock_calc_stats.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_calc_lmm.return_value = (pd.DataFrame(), pd.DataFrame())

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        # Call with custom significance threshold
        combine_compare_state_epoch_data(
            group1_activity_csv_files=self.group1_files[0],
            group1_correlation_csv_files=self.group1_files[1],
            group1_modulation_csv_files=self.group1_files[2],
            group1_name="Group 1",
            group2_activity_csv_files=self.group2_files[0],
            group2_correlation_csv_files=self.group2_files[1],
            group2_modulation_csv_files=self.group2_files[2],
            group2_name="Group 2",
            comparison_dimension="states",
            significance_threshold=0.01,  # Custom value
            output_dir=str(output_dir),
            enable_lmm_analysis=True,
        )

        # Verify the function was called with correct significance_threshold
        mock_calc_stats.assert_called()
        call_kwargs = mock_calc_stats.call_args[1]
        self.assertEqual(call_kwargs["significance_threshold"], 0.01)
        mock_calc_lmm.assert_called()
        logger.info(
            "[PASS] significance_threshold=0.01 correctly passed to utility function"
        )

    @patch("analysis.combine_compare_state_epoch_data.calculate_state_epoch_lmm_stats")
    @patch(
        "analysis.combine_compare_state_epoch_data.calculate_state_epoch_comparison_stats"
    )
    def test_multiple_correction_passed(self, mock_calc_stats, mock_calc_lmm):
        """Test that custom multiple_correction is passed through."""
        mock_calc_stats.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_calc_lmm.return_value = (pd.DataFrame(), pd.DataFrame())

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        # Call with custom multiple correction method
        combine_compare_state_epoch_data(
            group1_activity_csv_files=self.group1_files[0],
            group1_correlation_csv_files=self.group1_files[1],
            group1_modulation_csv_files=self.group1_files[2],
            group1_name="Group 1",
            group2_activity_csv_files=self.group2_files[0],
            group2_correlation_csv_files=self.group2_files[1],
            group2_modulation_csv_files=self.group2_files[2],
            group2_name="Group 2",
            comparison_dimension="states",
            multiple_correction="fdr_bh",  # Custom value
            output_dir=str(output_dir),
            enable_lmm_analysis=True,
        )

        # Verify the function was called with correct multiple_correction
        mock_calc_stats.assert_called()
        call_kwargs = mock_calc_stats.call_args[1]
        self.assertEqual(call_kwargs["multiple_correction"], "fdr_bh")
        mock_calc_lmm.assert_called()
        logger.info(
            "[PASS] multiple_correction='fdr_bh' correctly passed to utility function"
        )

    @patch("analysis.combine_compare_state_epoch_data.calculate_state_epoch_lmm_stats")
    @patch(
        "analysis.combine_compare_state_epoch_data.calculate_state_epoch_comparison_stats"
    )
    def test_multiple_correction_unknown_defaults_to_bonf(
        self, mock_calc_stats, mock_calc_lmm
    ):
        """Unsupported multiple_correction values should raise IdeasError."""
        mock_calc_stats.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_calc_lmm.return_value = (pd.DataFrame(), pd.DataFrame())

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        with self.assertRaises(IdeasError):
            combine_compare_state_epoch_data(
                group1_activity_csv_files=self.group1_files[0],
                group1_correlation_csv_files=self.group1_files[1],
                group1_modulation_csv_files=self.group1_files[2],
                group1_name="Group 1",
                group2_activity_csv_files=self.group2_files[0],
                group2_correlation_csv_files=self.group2_files[1],
                group2_modulation_csv_files=self.group2_files[2],
                group2_name="Group 2",
                comparison_dimension="states",
                multiple_correction="bogus_method",
                output_dir=str(output_dir),
            )

        mock_calc_stats.assert_not_called()
        mock_calc_lmm.assert_not_called()
        logger.info("[PASS] Invalid multiple_correction inputs trigger IdeasError")

    @patch("analysis.combine_compare_state_epoch_data.calculate_state_epoch_lmm_stats")
    @patch(
        "analysis.combine_compare_state_epoch_data.calculate_state_epoch_comparison_stats"
    )
    def test_effect_size_passed(self, mock_calc_stats, mock_calc_lmm):
        """Test that custom effect_size is passed through."""
        mock_calc_stats.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_calc_lmm.return_value = (pd.DataFrame(), pd.DataFrame())

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        # Call with custom effect size method
        combine_compare_state_epoch_data(
            group1_activity_csv_files=self.group1_files[0],
            group1_correlation_csv_files=self.group1_files[1],
            group1_modulation_csv_files=self.group1_files[2],
            group1_name="Group 1",
            group2_activity_csv_files=self.group2_files[0],
            group2_correlation_csv_files=self.group2_files[1],
            group2_modulation_csv_files=self.group2_files[2],
            group2_name="Group 2",
            comparison_dimension="states",
            effect_size="hedges",  # Custom value
            output_dir=str(output_dir),
            enable_lmm_analysis=True,
        )

        # Verify the function was called with correct effect_size
        mock_calc_stats.assert_called()
        call_kwargs = mock_calc_stats.call_args[1]
        self.assertEqual(call_kwargs["effect_size_method"], "hedges")
        mock_calc_lmm.assert_called()
        logger.info("[PASS] effect_size='hedges' correctly passed to utility function")

    @patch("analysis.combine_compare_state_epoch_data.calculate_state_epoch_lmm_stats")
    @patch(
        "analysis.combine_compare_state_epoch_data.calculate_state_epoch_comparison_stats"
    )
    def test_all_parameters_passed_together(self, mock_calc_stats, mock_calc_lmm):
        """Test that all three statistical parameters are passed correctly together."""
        mock_calc_stats.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_calc_lmm.return_value = (pd.DataFrame(), pd.DataFrame())

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        # Call with all custom parameters
        combine_compare_state_epoch_data(
            group1_activity_csv_files=self.group1_files[0],
            group1_correlation_csv_files=self.group1_files[1],
            group1_modulation_csv_files=self.group1_files[2],
            group1_name="Group 1",
            group2_activity_csv_files=self.group2_files[0],
            group2_correlation_csv_files=self.group2_files[1],
            group2_modulation_csv_files=self.group2_files[2],
            group2_name="Group 2",
            comparison_dimension="states",
            significance_threshold=0.001,  # Custom
            multiple_correction="fdr_bh",  # Custom
            effect_size="glass",  # Custom
            output_dir=str(output_dir),
            enable_lmm_analysis=True,
        )

        # Verify all three parameters were passed correctly
        mock_calc_stats.assert_called()
        call_kwargs = mock_calc_stats.call_args[1]

        self.assertEqual(call_kwargs["significance_threshold"], 0.001)
        self.assertEqual(call_kwargs["multiple_correction"], "fdr_bh")
        self.assertEqual(call_kwargs["effect_size_method"], "glass")
        mock_calc_lmm.assert_called()

        logger.info(
            "[PASS] All three statistical parameters passed correctly together:"
        )
        logger.info("  - significance_threshold=0.001")
        logger.info("  - multiple_correction='fdr_bh'")
        logger.info("  - effect_size='glass'")

    @patch("analysis.combine_compare_state_epoch_data.calculate_state_epoch_lmm_stats")
    @patch(
        "analysis.combine_compare_state_epoch_data.calculate_state_epoch_comparison_stats"
    )
    def test_default_parameters_used(self, mock_calc_stats, mock_calc_lmm):
        """Test that default values are used when parameters not specified."""
        mock_calc_stats.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_calc_lmm.return_value = (pd.DataFrame(), pd.DataFrame())

        output_dir = self.temp_dir / "output"
        output_dir.mkdir()

        # Call without specifying statistical parameters (use defaults)
        combine_compare_state_epoch_data(
            group1_activity_csv_files=self.group1_files[0],
            group1_correlation_csv_files=self.group1_files[1],
            group1_modulation_csv_files=self.group1_files[2],
            group1_name="Group 1",
            group2_activity_csv_files=self.group2_files[0],
            group2_correlation_csv_files=self.group2_files[1],
            group2_modulation_csv_files=self.group2_files[2],
            group2_name="Group 2",
            comparison_dimension="states",
            # Not specifying: significance_threshold, multiple_correction, effect_size
            output_dir=str(output_dir),
            enable_lmm_analysis=True,
        )

        # Verify default values were passed
        mock_calc_stats.assert_called()
        call_kwargs = mock_calc_stats.call_args[1]

        self.assertEqual(call_kwargs["significance_threshold"], 0.05)  # Default
        self.assertEqual(call_kwargs["multiple_correction"], "bonf")  # Default
        self.assertEqual(call_kwargs["effect_size_method"], "cohen")  # Default
        mock_calc_lmm.assert_called()

        logger.info("[PASS] Default statistical parameters used correctly:")
        logger.info("  - significance_threshold=0.05 (default)")
        logger.info("  - multiple_correction='bonf' (default)")
        logger.info("  - effect_size='cohen' (default)")


if __name__ == "__main__":
    unittest.main()
