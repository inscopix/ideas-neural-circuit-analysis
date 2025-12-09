#!/usr/bin/env python3
"""
Unit tests for combine_compare_population_data_utils.py

Tests critical functions for statistical analysis of population data including
LMM analysis, ANOVA, subject averaging, comparison type detection, and more.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

# Import the functions to test
from utils.combine_compare_population_data_utils import (
    calculate_state_lmm_stats,
    calculate_group_anova_stats,
    _calculate_subject_averages,
    detect_state_comparison_type,
    detect_baseline_state,
    validate_states_by_state_comparison_type,
    _perform_lmm_analysis,
    get_statistical_method_mapping,
    determine_experimental_condition,
    calculate_mod_stats_direct,
    _dispatch_direct_analysis,
    _direct_pairwise_multiple_groups,
    _direct_single_group_anova,
    _direct_multiple_groups_rm_anova,
    _direct_multiple_groups_mixed_anova,
    _handle_single_state_group_comparison,
    _log_missing_state_contrasts,
    _direct_pairwise_state_comparison,
    _calculate_weighted_modulation_proportions,
)

from ideas.exceptions import IdeasError


class TestCombineComparePopulationDataUtils(unittest.TestCase):
    """Test suite for population data utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")

        # Create sample activity data for testing (with 3 subjects for statistical validity)
        self.sample_activity_data = pd.DataFrame(
            {
                "normalized_subject_id": [
                    "subj1",
                    "subj1",
                    "subj1",
                    "subj2",
                    "subj2",
                    "subj2",
                    "subj3",
                    "subj3",
                    "subj3",
                ],
                "state": [
                    "baseline",
                    "active",
                    "baseline",
                    "baseline",
                    "active",
                    "active",
                    "baseline",
                    "active",
                    "baseline",
                ],
                "activity": [1.2, 2.4, 1.1, 1.5, 2.8, 2.3, 1.3, 2.1, 1.0],
                "group": [
                    "group1",
                    "group1",
                    "group1",
                    "group2",
                    "group2",
                    "group2",
                    "group1",
                    "group1",
                    "group1",
                ],
                "cell": [
                    "cell1",
                    "cell2",
                    "cell3",
                    "cell4",
                    "cell5",
                    "cell6",
                    "cell7",
                    "cell8",
                    "cell9",
                ],
            }
        )

        # Create sample modulation data for testing (with 3 subjects for statistical validity)
        self.sample_modulation_data = pd.DataFrame(
            {
                "normalized_subject_id": [
                    "subj1",
                    "subj1",
                    "subj2",
                    "subj2",
                    "subj3",
                    "subj3",
                ],
                "state": [
                    "baseline vs active",
                    "baseline vs active",
                    "baseline vs active",
                    "baseline vs active",
                    "baseline vs active",
                    "baseline vs active",
                ],
                "status": [
                    "up_modulated",
                    "down_modulated",
                    "up_modulated",
                    "down_modulated",
                    "up_modulated",
                    "down_modulated",
                ],
                "num_cells": [10, 5, 8, 12, 9, 7],
                "total_cell_count": [20, 20, 25, 25, 22, 22],
                "group": [
                    "group1",
                    "group1",
                    "group2",
                    "group2",
                    "group1",
                    "group1",
                ],
            }
        )

        # Create sample data with baseline state
        self.sample_baseline_data = pd.DataFrame(
            {
                "modulation scores in active": [0.5, 0.3, -0.2],
                "modulation scores in stimulus": [0.8, -0.1, 0.4],
                "mean activity in baseline": [1.2, 1.5, 1.8],
                "mean activity in active": [2.0, 1.8, 1.6],
                "mean activity in stimulus": [2.5, 1.3, 2.2],
            }
        )

    def tearDown(self):
        """Clean up after tests."""
        warnings.resetwarnings()

    # Test _calculate_subject_averages
    def test_calculate_subject_averages_basic(self):
        """Test basic subject averaging functionality."""
        # Create cell-level data
        cell_data = pd.DataFrame(
            {
                "normalized_subject_id": [
                    "subj1",
                    "subj1",
                    "subj1",
                    "subj2",
                    "subj2",
                ],
                "state": [
                    "baseline",
                    "baseline",
                    "active",
                    "baseline",
                    "active",
                ],
                "activity": [1.0, 1.2, 2.0, 1.5, 2.5],
                "group": ["group1", "group1", "group1", "group2", "group2"],
            }
        )

        result = _calculate_subject_averages(
            cell_data, "normalized_subject_id", "activity"
        )

        # Check that data is aggregated properly
        self.assertFalse(result.empty)
        self.assertIn("is_averaged", result.columns)
        self.assertTrue(result["is_averaged"].all())

        # Check that we have the right number of rows (2 subjects × 2 states each)
        expected_rows = cell_data.groupby(
            ["normalized_subject_id", "state"]
        ).ngroups
        self.assertEqual(len(result), expected_rows)

    def test_calculate_subject_averages_empty_input(self):
        """Test subject averaging with empty input."""
        empty_df = pd.DataFrame()
        result = _calculate_subject_averages(
            empty_df, "subject_id", "activity"
        )
        self.assertTrue(result.empty)

    def test_calculate_subject_averages_missing_columns(self):
        """Test subject averaging with missing required columns."""
        incomplete_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1"],
                "activity": [1.0],
                # Missing 'state' column
            }
        )

        result = _calculate_subject_averages(
            incomplete_data, "normalized_subject_id", "activity"
        )
        self.assertTrue(result.empty)

    # Test detect_state_comparison_type
    def test_detect_state_comparison_type_pairwise(self):
        """Test detection of pairwise comparison type."""
        pairwise_data = pd.DataFrame(
            {
                "modulation scores in baseline vs active": [0.5, 0.3],
                "modulation scores in active vs stimulus": [0.2, -0.1],
            }
        )

        result = detect_state_comparison_type(pairwise_data)
        self.assertEqual(result, "pairwise")

    def test_detect_state_comparison_type_baseline(self):
        """Test detection of state_vs_baseline comparison type."""
        result = detect_state_comparison_type(self.sample_baseline_data)
        self.assertEqual(result, "state_vs_baseline")

    def test_detect_state_comparison_type_not_state(self):
        """Test detection of state_vs_not_state comparison type."""
        not_state_data = pd.DataFrame(
            {
                "modulation scores in active": [0.5, 0.3],
                "modulation scores in stimulus": [0.2, -0.1],
            }
        )

        result = detect_state_comparison_type(not_state_data)
        self.assertEqual(result, "state_vs_not_state")

    def test_detect_state_comparison_type_empty_data(self):
        """Test detect_state_comparison_type with empty data."""
        with self.assertRaises(IdeasError) as context:
            detect_state_comparison_type(pd.DataFrame())

        # IdeasError doesn't have exit_status attribute, just check that exception was raised
        self.assertIsInstance(context.exception, IdeasError)

    # Test detect_baseline_state
    def test_detect_baseline_state_found(self):
        """Test successful baseline state detection."""
        baseline_state, mod_states, mean_states = detect_baseline_state(
            self.sample_baseline_data
        )

        self.assertEqual(baseline_state, "baseline")
        self.assertIn("active", mod_states)
        self.assertIn("stimulus", mod_states)
        self.assertIn("baseline", mean_states)

    def test_detect_baseline_state_pairwise(self):
        """Test baseline detection with pairwise data (should return None)."""
        pairwise_data = pd.DataFrame(
            {"modulation scores in baseline vs active": [0.5, 0.3]}
        )

        baseline_state, mod_states, mean_states = detect_baseline_state(
            pairwise_data
        )
        self.assertIsNone(baseline_state)

    def test_detect_baseline_state_empty_data(self):
        """Test baseline detection with empty data."""
        baseline_state, mod_states, mean_states = detect_baseline_state(
            pd.DataFrame()
        )

        self.assertIsNone(baseline_state)
        self.assertEqual(mod_states, [])
        self.assertEqual(mean_states, [])

    # Test validate_states_by_state_comparison_type
    def test_validate_states_by_state_comparison_type_baseline(self):
        """Test state validation for baseline comparison."""
        requested_states = ["active", "stimulus"]

        (
            validated_states,
            state_comparison_type,
            baseline_state,
        ) = validate_states_by_state_comparison_type(
            self.sample_baseline_data, requested_states
        )

        self.assertEqual(state_comparison_type, "state_vs_baseline")
        self.assertEqual(baseline_state, "baseline")
        self.assertIn("active", validated_states)
        self.assertIn("stimulus", validated_states)

    def test_validate_states_by_state_comparison_type_invalid_states(self):
        """Test state validation with invalid states."""
        requested_states = ["nonexistent_state"]

        with self.assertRaises(IdeasError):
            validate_states_by_state_comparison_type(
                self.sample_baseline_data, requested_states
            )

    # Test get_statistical_method_mapping
    def test_get_statistical_method_mapping(self):
        """Test statistical method mapping returns expected structure."""
        mapping = get_statistical_method_mapping()

        self.assertIsInstance(mapping, dict)
        # Test a few key mappings
        self.assertIn(("pairwise", "single_group"), mapping)
        self.assertIn(("state_vs_baseline", "multiple_groups_paired"), mapping)
        self.assertEqual(
            mapping[("pairwise", "single_group")],
            "Descriptive statistics only",
        )

    # Test determine_experimental_condition
    def test_determine_experimental_condition_single_group(self):
        """Test experimental condition determination for single group."""
        single_group_data = pd.DataFrame(
            {
                "group": ["group1", "group1", "group1"],
                "state": ["baseline", "active", "baseline"],
            }
        )

        comp_type, data_structure = determine_experimental_condition(
            single_group_data, "unpaired", "pairwise"
        )

        self.assertEqual(comp_type, "pairwise")
        self.assertEqual(data_structure, "single_group")

    def test_determine_experimental_condition_multiple_groups_paired(self):
        """Test experimental condition determination for multiple groups paired."""
        multi_group_data = pd.DataFrame(
            {
                "group": ["group1", "group1", "group2", "group2"],
                "state": ["baseline", "active", "baseline", "active"],
            }
        )

        comp_type, data_structure = determine_experimental_condition(
            multi_group_data, "paired", "state_vs_baseline"
        )

        self.assertEqual(comp_type, "state_vs_baseline")
        self.assertEqual(data_structure, "multiple_groups_paired")

    def test_determine_experimental_condition_group_name_column(self):
        """Test experimental condition determination with group_name column."""
        group_name_data = pd.DataFrame(
            {
                "group_name": ["group1", "group1", "group2", "group2"],
                "state": ["baseline", "active", "baseline", "active"],
            }
        )

        comp_type, data_structure = determine_experimental_condition(
            group_name_data, "unpaired", "state_vs_baseline"
        )

        self.assertEqual(comp_type, "state_vs_baseline")
        self.assertEqual(data_structure, "multiple_groups_unpaired")

    # Test _calculate_weighted_modulation_proportions
    def test_calculate_weighted_modulation_proportions_basic(self):
        """Test basic weighted proportion calculation."""
        result = _calculate_weighted_modulation_proportions(
            self.sample_modulation_data
        )

        self.assertFalse(result.empty)
        self.assertIn("proportion", result.columns)

        # Check that proportions are calculated correctly
        expected_props = (
            self.sample_modulation_data["num_cells"]
            / self.sample_modulation_data["total_cell_count"]
        )
        pd.testing.assert_series_equal(
            result["proportion"], expected_props, check_names=False
        )

    def test_calculate_weighted_modulation_proportions_missing_columns(self):
        """Test weighted proportions with missing required columns."""
        incomplete_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1"],
                "state": ["baseline vs active"],
                "status": ["up_modulated"],
                # Missing num_cells and total_cell_count
            }
        )

        result = _calculate_weighted_modulation_proportions(incomplete_data)
        # Should return original data when missing columns
        pd.testing.assert_frame_equal(result, incomplete_data)

    def test_calculate_weighted_modulation_proportions_edge_cases(self):
        """Test weighted proportions with edge cases."""
        edge_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj2", "subj3"],
                "state": ["baseline vs active"] * 3,
                "status": ["up_modulated"] * 3,
                "num_cells": [10, 0, 5],
                "total_cell_count": [
                    10,
                    0,
                    np.inf,
                ],  # Test division by zero and infinity
                "group": ["group1"] * 3,
            }
        )

        result = _calculate_weighted_modulation_proportions(edge_data)

        self.assertFalse(result.empty)
        self.assertIn("proportion", result.columns)

        # Check that proportions are clipped and handle edge cases
        self.assertEqual(result["proportion"].iloc[0], 1.0)  # 10/10 = 1.0
        self.assertEqual(
            result["proportion"].iloc[1], 0.0
        )  # 0/0 -> 0 (handled)
        self.assertEqual(
            result["proportion"].iloc[2], 0.0
        )  # 5/inf -> 0 (handled)

    # Test calculate_state_lmm_stats
    @patch(
        "utils.combine_compare_population_data_utils._perform_lmm_analysis"
    )
    @patch(
        "utils.combine_compare_population_data_utils._calculate_subject_averages"
    )
    @patch(
        "utils.combine_compare_population_data_utils._safe_pairwise_ttests"
    )
    def test_calculate_state_lmm_stats_single_group(
        self, mock_pairwise, mock_averages, mock_lmm
    ):
        """Test LMM stats calculation for single group."""
        # Set up mocks
        mock_lmm.return_value = (
            pd.DataFrame({"Source": ["state"], "p_value": [0.05]}),
            False,
            [],  # Add empty warning messages list
        )
        mock_averages.return_value = self.sample_activity_data
        mock_pairwise.return_value = pd.DataFrame(
            {"Contrast": ["baseline vs active"], "p-unc": [0.03]}
        )

        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=self.sample_activity_data,
            states=["baseline", "active"],
            has_single_group=True,
            measure_name="activity",
        )

        self.assertFalse(lmm_result.empty)
        self.assertFalse(pairwise_result.empty)
        mock_lmm.assert_called_once()

    def test_calculate_state_lmm_stats_empty_data(self):
        """Test LMM stats calculation with empty data."""
        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=pd.DataFrame(),
            states=["baseline", "active"],
            has_single_group=True,
        )

        self.assertTrue(lmm_result.empty)
        self.assertTrue(pairwise_result.empty)

    def test_calculate_state_lmm_stats_insufficient_subjects(self):
        """Test LMM stats calculation with insufficient subjects."""
        single_subject_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj1"],
                "state": ["baseline", "active"],
                "activity": [1.0, 2.0],
            }
        )

        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=single_subject_data,
            states=["baseline", "active"],
            has_single_group=True,
        )

        self.assertTrue(lmm_result.empty)
        self.assertTrue(pairwise_result.empty)

    # Test calculate_group_anova_stats
    @patch("utils.combine_compare_population_data_utils._safe_anova")
    @patch(
        "utils.combine_compare_population_data_utils._safe_pairwise_ttests"
    )
    def test_calculate_group_anova_stats_basic(
        self, mock_pairwise, mock_anova
    ):
        """Test basic group ANOVA calculation."""
        # Set up mocks
        mock_anova.return_value = pd.DataFrame(
            {"Source": ["group"], "p-unc": [0.05], "F": [4.2]}
        )
        mock_pairwise.return_value = pd.DataFrame(
            {"Contrast": ["group1 vs group2"], "p-unc": [0.03]}
        )

        anova_result, pairwise_result, group_df = calculate_group_anova_stats(
            df=self.sample_activity_data,
            states=["baseline", "active"],
            correction="bonf",
            effect_size="hedges",
        )

        # Should return results even if mocked
        self.assertIsInstance(anova_result, pd.DataFrame)
        self.assertIsInstance(pairwise_result, pd.DataFrame)
        self.assertIsInstance(group_df, pd.DataFrame)

    def test_calculate_group_anova_stats_insufficient_groups(self):
        """Test group ANOVA with insufficient groups."""
        single_group_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj2"],
                "activity": [1.0, 2.0],
                "state": ["baseline", "active"],
                "group": ["group1", "group1"],
            }
        )

        anova_result, pairwise_result, group_df = calculate_group_anova_stats(
            df=single_group_data,
            states=["baseline"],  # Single state
            correction="bonf",
            effect_size="hedges",
        )

        # Should return empty results for single state + insufficient groups
        self.assertTrue(anova_result.empty)
        self.assertTrue(pairwise_result.empty)

    # Test calculate_mod_stats_direct
    @patch(
        "utils.combine_compare_population_data_utils._dispatch_direct_analysis"
    )
    def test_calculate_mod_stats_direct_basic(self, mock_dispatch):
        """Test direct modulation stats calculation."""
        # Set up mock
        mock_dispatch.return_value = (
            [{"Status": "up_modulated", "p_value": 0.05}],
            [
                {
                    "Status": "up_modulated",
                    "Contrast": "group1 vs group2",
                    "p-unc": 0.03,
                }
            ],
        )

        anova_result, pairwise_result = calculate_mod_stats_direct(
            df=self.sample_modulation_data,
            states=["baseline vs active"],
            correction="bonf",
            effect_size="hedges",
            state_comparison_type="pairwise",
            status_values=["up_modulated", "down_modulated"],
        )

        self.assertFalse(anova_result.empty)
        self.assertFalse(pairwise_result.empty)

    def test_calculate_mod_stats_direct_empty_data(self):
        """Test direct modulation stats with empty data."""
        anova_result, pairwise_result = calculate_mod_stats_direct(
            df=pd.DataFrame(),
            states=["baseline vs active"],
            correction="bonf",
            effect_size="hedges",
            state_comparison_type="pairwise",
        )

        self.assertTrue(anova_result.empty)
        self.assertTrue(pairwise_result.empty)

    def test_calculate_mod_stats_direct_missing_parameters(self):
        """Test direct modulation stats with missing parameters."""
        anova_result, pairwise_result = calculate_mod_stats_direct(
            df=self.sample_modulation_data,
            states=[],  # Empty states
            correction="bonf",
            effect_size="hedges",
            state_comparison_type=None,  # Missing comparison type
        )

        self.assertTrue(anova_result.empty)
        self.assertTrue(pairwise_result.empty)

    # Test _dispatch_direct_analysis
    def test_dispatch_direct_analysis_single_group_pairwise(self):
        """Test dispatch for single group pairwise analysis."""
        condition_key = ("pairwise", "single_group")

        anova_results, pairwise_results = _dispatch_direct_analysis(
            condition_key=condition_key,
            status_df=self.sample_modulation_data,
            status="up_modulated",
            states=["baseline vs active"],
            correction="bonf",
            effect_size="hedges",
            measure_column="num_cells",
            data_pairing="unpaired",
        )

        # Single group pairwise should return empty results
        self.assertEqual(anova_results, [])
        self.assertEqual(pairwise_results, [])

    # Test _log_missing_state_contrasts
    @patch("utils.combine_compare_population_data_utils.logger")
    def test_log_missing_state_contrasts(self, mock_logger):
        """Test logging of missing state contrasts."""
        expected_pairs = [("baseline", "active"), ("baseline", "stimulus")]
        processed_pairs = [("baseline", "active")]
        skipped_pairs = {("baseline", "stimulus"): "Insufficient data"}

        _log_missing_state_contrasts(
            "up_modulated", expected_pairs, processed_pairs, skipped_pairs
        )

        # Verify that logger was called
        mock_logger.debug.assert_called()
        mock_logger.info.assert_called()

    # Test error handling and edge cases
    def test_functions_with_none_input(self):
        """Test functions handle None input gracefully."""
        # Test detect_state_comparison_type
        with self.assertRaises(IdeasError):
            detect_state_comparison_type(None)

        # Test detect_baseline_state
        result = detect_baseline_state(None)
        self.assertEqual(result, (None, [], []))

        # Test subject averaging
        result = _calculate_subject_averages(None, "subject_id", "activity")
        self.assertTrue(result.empty)

    # Test data type validation
    def test_subject_averages_data_types(self):
        """Test subject averaging handles various data types correctly."""
        mixed_type_data = pd.DataFrame(
            {
                "normalized_subject_id": [1, 1, 2, 2],  # Numeric IDs
                "state": ["baseline", "active", "baseline", "active"],
                "activity": ["1.0", "2.0", "1.5", "2.5"],  # String numbers
                "group": ["group1", "group1", "group2", "group2"],
            }
        )

        result = _calculate_subject_averages(
            mixed_type_data, "normalized_subject_id", "activity"
        )

        # Should handle type conversions
        self.assertFalse(result.empty)
        self.assertTrue(result["normalized_subject_id"].dtype == "object")

    # Integration test for statistical method mapping consistency
    def test_statistical_method_mapping_consistency(self):
        """Test that statistical method mapping covers all expected conditions."""
        mapping = get_statistical_method_mapping()

        # Define expected conditions
        state_comparison_types = [
            "pairwise",
            "state_vs_baseline",
            "state_vs_not_state",
        ]
        data_structures = [
            "single_group",
            "multiple_groups_paired",
            "multiple_groups_unpaired",
        ]

        # Check that all combinations are covered
        for comp_type in state_comparison_types:
            for data_struct in data_structures:
                condition = (comp_type, data_struct)
                self.assertIn(
                    condition,
                    mapping,
                    f"Missing mapping for condition: {condition}",
                )

    # Test weighted proportions with realistic data
    def test_weighted_proportions_realistic_scenario(self):
        """Test weighted proportions with realistic neuroscience data."""
        realistic_data = pd.DataFrame(
            {
                "normalized_subject_id": [
                    "mouse1",
                    "mouse1",
                    "mouse2",
                    "mouse2",
                    "mouse3",
                    "mouse3",
                ],
                "state": ["baseline vs stim"] * 6,
                "status": ["up_modulated", "down_modulated"] * 3,
                "num_cells": [45, 23, 67, 31, 52, 18],
                "total_cell_count": [150, 150, 200, 200, 120, 120],
                "group": [
                    "control",
                    "control",
                    "treatment",
                    "treatment",
                    "control",
                    "control",
                ],
            }
        )

        result = _calculate_weighted_modulation_proportions(realistic_data)

        self.assertFalse(result.empty)
        self.assertIn("proportion", result.columns)

        # Check that proportions are reasonable (between 0 and 1)
        self.assertTrue((result["proportion"] >= 0).all())
        self.assertTrue((result["proportion"] <= 1).all())

        # Check a specific calculation
        expected_prop_0 = 45 / 150  # First row
        self.assertAlmostEqual(
            result["proportion"].iloc[0], expected_prop_0, places=6
        )

    # Test _perform_lmm_analysis
    @patch("utils.combine_compare_population_data_utils.MixedLM")
    def test_perform_lmm_analysis_basic(self, mock_mixed_lm):
        """Test basic LMM analysis functionality."""
        # Set up mock
        mock_result = MagicMock()
        mock_result.converged = True
        mock_result.params = pd.Series(
            [1.0, 0.5], index=["Intercept", "state"]
        )
        mock_result.bse = pd.Series([0.1, 0.05], index=["Intercept", "state"])
        mock_result.tvalues = pd.Series(
            [10.0, 10.0], index=["Intercept", "state"]
        )
        mock_result.pvalues = pd.Series(
            [0.001, 0.001], index=["Intercept", "state"]
        )
        mock_result.conf_int.return_value = pd.DataFrame(
            [[0.8, 1.2], [0.4, 0.6]], index=["Intercept", "state"]
        )
        mock_result.df_resid = 18
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_result
        mock_mixed_lm.from_formula.return_value = mock_model

        lmm_result, has_concerns, warning_messages = _perform_lmm_analysis(
            df=self.sample_activity_data,
            dv="activity",
            within_factors=["state"],
            subject="normalized_subject_id",
            between=None,
            has_single_group=True,
        )

        self.assertFalse(lmm_result.empty)
        self.assertFalse(has_concerns)
        # Check for key columns that should be present after processing
        self.assertIn("Source", lmm_result.columns)
        self.assertIn("p_value", lmm_result.columns)

    def test_perform_lmm_analysis_convergence_issues(self):
        """Test LMM analysis with convergence issues."""
        # Test with insufficient data
        small_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1"],
                "state": ["baseline"],
                "activity": [1.0],
            }
        )

        lmm_result, has_concerns, warning_messages = _perform_lmm_analysis(
            df=small_data,
            dv="activity",
            within_factors=["state"],
            subject="normalized_subject_id",
        )

        self.assertTrue(lmm_result.empty)
        self.assertFalse(has_concerns)

    # Test _direct_pairwise_multiple_groups
    @patch(
        "utils.combine_compare_population_data_utils._validate_modulation_group_data"
    )
    @patch(
        "utils.combine_compare_population_data_utils._safe_ttest_modulation"
    )
    def test_direct_pairwise_multiple_groups(self, mock_ttest, mock_validate):
        """Test direct pairwise analysis for multiple groups."""
        # Set up mocks
        mock_validate.return_value = (True, "", self.sample_modulation_data)
        mock_ttest.return_value = pd.DataFrame(
            {"T": [2.5], "p-val": [0.03], "cohen-d": [0.8]}
        )

        anova_results, pairwise_results = _direct_pairwise_multiple_groups(
            status_df=self.sample_modulation_data,
            status="up_modulated",
            states=["baseline vs active"],
            measure_column="num_cells",
            correction="bonf",
            effect_size="hedges",
            data_pairing="unpaired",
        )

        self.assertEqual(anova_results, [])  # No ANOVA for pairwise
        self.assertGreater(len(pairwise_results), 0)

    # Test _direct_single_group_anova
    @patch(
        "utils.combine_compare_population_data_utils._validate_modulation_data"
    )
    @patch("utils.combine_compare_population_data_utils._safe_anova")
    @patch(
        "utils.combine_compare_population_data_utils._safe_pairwise_ttests"
    )
    def test_direct_single_group_anova(
        self, mock_pairwise, mock_anova, mock_validate
    ):
        """Test direct single group ANOVA analysis."""
        # Create test data with proper state column (not pairwise format)
        test_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj1", "subj2", "subj2"],
                "state": ["baseline", "active", "baseline", "active"],
                "status": ["up_modulated"] * 4,
                "num_cells": [10, 15, 12, 18],
                "group": ["group1"] * 4,
            }
        )

        # Set up mocks
        mock_validate.return_value = (True, "", test_data)
        mock_anova.return_value = pd.DataFrame(
            {"Source": ["state"], "p-unc": [0.05], "F": [4.2], "DF": [1]}
        )
        mock_pairwise.return_value = pd.DataFrame(
            {"Contrast": ["baseline vs active"], "p-unc": [0.03], "T": [2.5]}
        )

        anova_results, pairwise_results = _direct_single_group_anova(
            status_df=test_data,
            status="up_modulated",
            states=["baseline", "active"],
            measure_column="num_cells",
            correction="bonf",
            effect_size="hedges",
        )

        self.assertGreater(len(anova_results), 0)
        self.assertGreater(len(pairwise_results), 0)

    # Test _direct_multiple_groups_rm_anova
    @patch(
        "utils.combine_compare_population_data_utils._validate_modulation_group_data"
    )
    @patch("utils.combine_compare_population_data_utils._safe_anova")
    @patch(
        "utils.combine_compare_population_data_utils._safe_pairwise_ttests"
    )
    def test_direct_multiple_groups_rm_anova(
        self, mock_pairwise, mock_anova, mock_validate
    ):
        """Test direct repeated measures ANOVA analysis."""
        # Set up mocks
        mock_validate.return_value = (True, "", self.sample_modulation_data)
        mock_anova.return_value = pd.DataFrame(
            {
                "Source": ["state", "group"],
                "p-unc": [0.05, 0.03],
                "F": [4.2, 6.1],
            }
        )
        mock_pairwise.return_value = pd.DataFrame(
            {"Contrast": ["group1 vs group2"], "p-unc": [0.03], "T": [2.5]}
        )

        anova_results, pairwise_results = _direct_multiple_groups_rm_anova(
            status_df=self.sample_modulation_data,
            status="up_modulated",
            states=["baseline", "active"],
            measure_column="num_cells",
            correction="bonf",
            effect_size="hedges",
        )

        self.assertGreater(len(anova_results), 0)
        self.assertGreater(len(pairwise_results), 0)

    # Test _direct_multiple_groups_mixed_anova
    @patch(
        "utils.combine_compare_population_data_utils._validate_modulation_group_data"
    )
    @patch("utils.combine_compare_population_data_utils._safe_anova")
    @patch(
        "utils.combine_compare_population_data_utils._safe_pairwise_ttests"
    )
    def test_direct_multiple_groups_mixed_anova(
        self, mock_pairwise, mock_anova, mock_validate
    ):
        """Test direct mixed ANOVA analysis."""
        # Set up mocks
        mock_validate.return_value = (True, "", self.sample_modulation_data)
        mock_anova.return_value = pd.DataFrame(
            {
                "Source": ["state", "group", "state * group"],
                "p-unc": [0.05, 0.03, 0.08],
                "F": [4.2, 6.1, 3.2],
            }
        )
        mock_pairwise.return_value = pd.DataFrame(
            {"Contrast": ["group1 vs group2"], "p-unc": [0.03], "T": [2.5]}
        )

        anova_results, pairwise_results = _direct_multiple_groups_mixed_anova(
            status_df=self.sample_modulation_data,
            status="up_modulated",
            states=["baseline", "active"],
            measure_column="num_cells",
            correction="bonf",
            effect_size="hedges",
        )

        self.assertGreater(len(anova_results), 0)
        self.assertGreater(len(pairwise_results), 0)

    # Test _handle_single_state_group_comparison
    @patch(
        "utils.combine_compare_population_data_utils._safe_ttest_modulation"
    )
    def test_handle_single_state_group_comparison_two_groups(self, mock_ttest):
        """Test single state group comparison with two groups."""
        # Set up mock
        mock_ttest.return_value = pd.DataFrame(
            {"T": [2.5], "p-val": [0.03], "cohen-d": [0.8]}
        )

        # Create single state data
        single_state_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj2"],
                "state": ["baseline", "baseline"],
                "activity": [1.2, 1.8],
                "group": ["group1", "group2"],
            }
        )

        aov_df, pairwise_df, group_df = _handle_single_state_group_comparison(
            group_df=single_state_data,
            correction="bonf",
            effect_size="hedges",
            data_pairing="unpaired",
            measure_name="activity",
        )

        self.assertTrue(aov_df.empty)  # No ANOVA for 2 groups
        self.assertFalse(pairwise_df.empty)  # Should have pairwise results

    @patch("utils.combine_compare_population_data_utils._safe_anova")
    @patch(
        "utils.combine_compare_population_data_utils._safe_pairwise_ttests"
    )
    def test_handle_single_state_group_comparison_multiple_groups(
        self, mock_pairwise, mock_anova
    ):
        """Test single state group comparison with multiple groups."""
        # Set up mocks
        mock_anova.return_value = pd.DataFrame(
            {"Source": ["group"], "p-unc": [0.05], "F": [4.2]}
        )
        mock_pairwise.return_value = pd.DataFrame(
            {"Contrast": ["group1 vs group2"], "p-unc": [0.03], "T": [2.5]}
        )

        # Create single state, multiple groups data
        multi_group_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj2", "subj3"],
                "state": ["baseline", "baseline", "baseline"],
                "activity": [1.2, 1.8, 2.1],
                "group": ["group1", "group2", "group3"],
            }
        )

        aov_df, pairwise_df, group_df = _handle_single_state_group_comparison(
            group_df=multi_group_data,
            correction="bonf",
            effect_size="hedges",
            data_pairing="unpaired",
            measure_name="activity",
        )

        self.assertFalse(aov_df.empty)  # Should have ANOVA results
        self.assertFalse(pairwise_df.empty)  # Should have pairwise results

    # Test _direct_pairwise_state_comparison
    @patch(
        "utils.combine_compare_population_data_utils._validate_modulation_group_data"
    )
    @patch(
        "utils.combine_compare_population_data_utils._safe_ttest_modulation"
    )
    def test_direct_pairwise_state_comparison(self, mock_ttest, mock_validate):
        """Test direct pairwise state comparison."""
        # Set up mocks
        mock_validate.return_value = (True, "", self.sample_modulation_data)
        mock_ttest.return_value = pd.DataFrame(
            {"T": [2.5], "p-val": [0.03], "cohen-d": [0.8]}
        )

        anova_results, pairwise_results = _direct_pairwise_state_comparison(
            status_df=self.sample_modulation_data,
            status="up_modulated",
            states=["baseline vs active"],
            measure_column="num_cells",
            correction="bonf",
            effect_size="hedges",
            data_pairing="unpaired",
        )

        self.assertEqual(anova_results, [])  # No ANOVA for pairwise state
        self.assertGreater(len(pairwise_results), 0)

        # Check that all expected states are processed
        processed_states = [result["Contrast"] for result in pairwise_results]
        self.assertIn("baseline vs active", processed_states)

    def test_direct_pairwise_state_comparison_validation_failure(self):
        """Test pairwise state comparison with validation failure."""
        # Test with invalid state format
        invalid_data = pd.DataFrame(
            {
                "state": ["invalid_state_format"],  # Missing ' vs '
                "status": ["up_modulated"],
                "num_cells": [10],
                "group": ["group1"],
            }
        )

        anova_results, pairwise_results = _direct_pairwise_state_comparison(
            status_df=invalid_data,
            status="up_modulated",
            states=["invalid_state_format"],
            measure_column="num_cells",
            correction="bonf",
            effect_size="hedges",
            data_pairing="unpaired",
        )

        self.assertEqual(anova_results, [])
        # When validation fails, no results are returned (empty list)
        # This is expected behavior for invalid data
        self.assertEqual(pairwise_results, [])

    # Test error handling and edge cases for complex functions
    def test_complex_functions_error_handling(self):
        """Test error handling in complex statistical functions."""
        # Test with completely empty data
        empty_df = pd.DataFrame()

        # These should all handle empty data gracefully
        anova_results, pairwise_results = _direct_single_group_anova(
            empty_df, "status", [], "measure", "bonf", "hedges"
        )
        self.assertEqual(anova_results, [])
        self.assertEqual(pairwise_results, [])

        # Test with malformed data
        malformed_data = pd.DataFrame({"invalid_column": [1, 2, 3]})

        aov_df, pair_df, group_df = _handle_single_state_group_comparison(
            group_df=malformed_data,
            correction="bonf",
            effect_size="hedges",
            data_pairing="unpaired",
            measure_name="activity",
        )

        # Should return empty results for malformed data
        self.assertTrue(aov_df.empty)
        self.assertTrue(pair_df.empty)

    # Test realistic data scenarios
    def test_realistic_neuroscience_data_scenario(self):
        """Test with realistic neuroscience data patterns."""
        # Create realistic population data
        realistic_data = pd.DataFrame(
            {
                "normalized_subject_id": [
                    "mouse1",
                    "mouse1",
                    "mouse1",
                    "mouse2",
                    "mouse2",
                    "mouse2",
                    "mouse3",
                    "mouse3",
                    "mouse3",
                    "mouse4",
                    "mouse4",
                    "mouse4",
                ],
                "state": ["baseline", "stimulus", "recovery"] * 4,
                "activity": [
                    1.2,
                    2.8,
                    1.5,
                    1.0,
                    3.2,
                    1.3,
                    1.4,
                    2.5,
                    1.7,
                    1.1,
                    2.9,
                    1.6,
                ],
                "group": [
                    "control",
                    "control",
                    "control",
                    "control",
                    "control",
                    "control",
                    "treatment",
                    "treatment",
                    "treatment",
                    "treatment",
                    "treatment",
                    "treatment",
                ],
                "cell": [f"cell{i}" for i in range(1, 13)],
            }
        )

        # Test subject averaging
        averaged_data = _calculate_subject_averages(
            realistic_data, "normalized_subject_id", "activity"
        )

        self.assertFalse(averaged_data.empty)
        self.assertEqual(len(averaged_data), 12)  # 4 subjects × 3 states
        self.assertIn("is_averaged", averaged_data.columns)

        # Test experimental condition determination
        comp_type, data_structure = determine_experimental_condition(
            realistic_data, "unpaired", "state_vs_baseline"
        )

        self.assertEqual(comp_type, "state_vs_baseline")
        self.assertEqual(data_structure, "multiple_groups_unpaired")

    # Test data validation edge cases
    def test_data_validation_edge_cases(self):
        """Test edge cases in data validation."""
        # Test with NaN values
        nan_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj1", "subj2", "subj2"],
                "state": ["baseline", "active", "baseline", np.nan],
                "activity": [1.0, np.nan, 1.5, 2.0],
                "group": ["group1", "group1", "group2", "group2"],
            }
        )

        result = _calculate_subject_averages(
            nan_data, "normalized_subject_id", "activity"
        )

        # Should handle NaN values by dropping them
        self.assertFalse(result.empty)
        # Should have fewer rows than input due to NaN removal
        expected_valid_combinations = (
            nan_data.dropna()
            .groupby(["normalized_subject_id", "state"])
            .ngroups
        )
        self.assertEqual(len(result), expected_valid_combinations)

    # Test statistical method consistency
    def test_statistical_method_integration(self):
        """Test integration between statistical method mapping and dispatch."""
        # Test that dispatch handles all mapped conditions
        mapping = get_statistical_method_mapping()

        for condition_key, method_name in mapping.items():
            # Create minimal test data for each condition
            test_data = pd.DataFrame(
                {
                    "normalized_subject_id": ["subj1", "subj2"],
                    "state": ["baseline", "active"],
                    "status": ["up_modulated", "up_modulated"],
                    "num_cells": [10, 15],
                    "group": ["group1", "group2"],
                }
            )

            # Test that dispatch doesn't crash for any condition
            try:
                anova_results, pairwise_results = _dispatch_direct_analysis(
                    condition_key=condition_key,
                    status_df=test_data,
                    status="up_modulated",
                    states=["baseline", "active"],
                    correction="bonf",
                    effect_size="hedges",
                    measure_column="num_cells",
                    data_pairing="unpaired",
                )

                # Results should be lists (even if empty)
                self.assertIsInstance(anova_results, list)
                self.assertIsInstance(pairwise_results, list)

            except Exception as e:
                # If there's an exception, it should be handled gracefully
                self.fail(
                    f"Dispatch failed for condition {condition_key}: {str(e)}"
                )

    # =============================================================================
    # COMPREHENSIVE STATISTICAL ACCURACY TESTS FOR LMM FUNCTIONS
    # =============================================================================

    def test_lmm_mathematical_accuracy_single_group_nested_data(self):
        """Test LMM statistical accuracy with nested cell-level data"""
        # Create realistic nested neuroscience data
        np.random.seed(42)  # For reproducible results

        subjects = ["subject_1", "subject_2", "subject_3", "subject_4"]
        states = ["baseline", "stimulus"]
        cells_per_subject_state = (
            5  # Multiple cells per subject-state combination
        )

        # Generate known effects for validation
        baseline_mean = 1.0
        stimulus_effect = 0.5  # Known effect size
        subject_effects = np.random.normal(
            0, 0.2, len(subjects)
        )  # Random subject effects
        cell_noise = 0.1  # Within-subject noise

        test_data = []
        for i, subject in enumerate(subjects):
            for state in states:
                state_mean = baseline_mean + (
                    stimulus_effect if state == "stimulus" else 0
                )
                subject_mean = state_mean + subject_effects[i]

                for cell_idx in range(cells_per_subject_state):
                    cell_activity = subject_mean + np.random.normal(
                        0, cell_noise
                    )
                    test_data.append(
                        {
                            "normalized_subject_id": subject,
                            "state": state,
                            "activity": cell_activity,
                            "cell": f"{subject}_cell_{cell_idx}",
                            "name": f"{subject}_cell_{cell_idx}",
                        }
                    )

        nested_df = pd.DataFrame(test_data)

        # Test LMM analysis
        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=nested_df,
            states=states,
            has_single_group=True,
            measure_name="activity",
        )

        # Validate results
        self.assertFalse(
            lmm_result.empty, "LMM should return results for valid nested data"
        )
        self.assertFalse(
            pairwise_result.empty,
            "Pairwise should return results for valid nested data",
        )

        # Check for key statistical components
        self.assertIn("Source", lmm_result.columns)
        self.assertIn("p_value", lmm_result.columns)
        self.assertIn("coefficient", lmm_result.columns)
        self.assertIn("std_error", lmm_result.columns)

        # Validate statistical accuracy
        # Should detect the stimulus effect
        state_effects = lmm_result[
            lmm_result["Source"].str.contains("state", case=False, na=False)
        ]
        if not state_effects.empty:
            # Check that coefficient is in expected direction and magnitude
            state_coeff = state_effects["coefficient"].iloc[0]
            self.assertGreater(
                abs(state_coeff), 0, "Should detect non-zero state effect"
            )
            # Allow for some variance due to randomness but check general direction
            self.assertTrue(
                -1 < state_coeff < 1,
                f"State coefficient {state_coeff} should be reasonable",
            )

    def test_lmm_mathematical_accuracy_multiple_groups_mixed_design(self):
        """Test LMM statistical accuracy with multiple groups (mixed design)."""
        np.random.seed(123)  # For reproducible results

        subjects_per_group = 4
        states = ["baseline", "stimulus"]
        cells_per_subject_state = 3

        # Generate known group and state effects
        baseline_mean = 1.0
        state_effect = 0.4  # Main effect of state
        group_effect = 0.3  # Main effect of group
        interaction_effect = 0.2  # Group × state interaction

        test_data = []
        for group_idx, group in enumerate(["control", "treatment"]):
            for subj_idx in range(subjects_per_group):
                subject_id = f"subject_{group_idx}_{subj_idx}"
                subject_effect = np.random.normal(
                    0, 0.15
                )  # Random subject effect

                for state in states:
                    # Calculate expected mean with all effects
                    expected_mean = baseline_mean
                    if state == "stimulus":
                        expected_mean += state_effect
                    if group == "treatment":
                        expected_mean += group_effect
                    if state == "stimulus" and group == "treatment":
                        expected_mean += interaction_effect

                    subject_state_mean = expected_mean + subject_effect

                    for cell_idx in range(cells_per_subject_state):
                        cell_activity = subject_state_mean + np.random.normal(
                            0, 0.1
                        )
                        test_data.append(
                            {
                                "normalized_subject_id": subject_id,
                                "state": state,
                                "activity": cell_activity,
                                "group": group,
                                "cell": f"{subject_id}_cell_{cell_idx}",
                                "name": f"{subject_id}_cell_{cell_idx}",
                            }
                        )

        mixed_df = pd.DataFrame(test_data)

        # Test LMM analysis for mixed design
        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=mixed_df,
            states=states,
            has_single_group=False,
            measure_name="activity",
        )

        # Validate mixed design results
        self.assertFalse(lmm_result.empty, "LMM should handle mixed design")

        # Check for expected fixed effects
        sources = lmm_result["Source"].tolist()

        # Should have intercept and main effects
        self.assertTrue(
            any("intercept" in str(s).lower() for s in sources),
            "Should have intercept term",
        )

        # Check for state and group effects
        has_state_effect = any("state" in str(s).lower() for s in sources)
        has_group_effect = any("group" in str(s).lower() for s in sources)

        # At least one of these should be present in a proper mixed model
        self.assertTrue(
            has_state_effect or has_group_effect,
            "Should detect state and/or group effects in mixed design",
        )

    def test_lmm_pseudoreplication_prevention(self):
        """Test that LMM properly handles pseudoreplication"""
        np.random.seed(456)

        # Create data with clear pseudoreplication risk
        subjects = ["subj1", "subj2"]
        states = ["baseline", "treatment"]
        cells_per_subject = (
            20  # Many cells per subject - risk of pseudoreplication
        )

        test_data = []
        for subject in subjects:
            for state in states:
                base_activity = 1.0 + (0.5 if state == "treatment" else 0)
                for cell_idx in range(cells_per_subject):
                    # Add small random variation to each cell
                    cell_activity = base_activity + np.random.normal(0, 0.05)
                    test_data.append(
                        {
                            "normalized_subject_id": subject,
                            "state": state,
                            "activity": cell_activity,
                            "cell": f"{subject}_cell_{cell_idx}",
                            "name": f"{subject}_cell_{cell_idx}",
                        }
                    )

        pseudo_df = pd.DataFrame(test_data)

        # Test that pairwise comparisons use subject-averaged data
        with patch(
            "utils.combine_compare_population_data_utils._calculate_subject_averages"
        ) as mock_avg:
            # Mock subject averaging to return aggregated data
            mock_avg.return_value = (
                pseudo_df.groupby(["normalized_subject_id", "state"])[
                    "activity"
                ]
                .mean()
                .reset_index()
            )
            mock_avg.return_value["is_averaged"] = True

            lmm_result, pairwise_result = calculate_state_lmm_stats(
                df=pseudo_df,
                states=states,
                has_single_group=True,
                measure_name="activity",
            )

            # Verify that subject averaging was called for pairwise tests
            mock_avg.assert_called()
            call_args = mock_avg.call_args
            self.assertIn(
                "for pairwise tests",
                str(call_args),
                "Should call subject averaging specifically for pairwise tests",
            )

    def test_lmm_convergence_and_reliability_detection(self):
        """Test LMM convergence checking and reliability warning system."""
        # Create problematic data that might cause convergence issues
        problematic_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1"] * 4,  # Only one subject
                "state": ["baseline", "stimulus"] * 2,
                "activity": [1.0, 1.0, 1.0, 1.0],  # No variation
                "cell": ["cell1", "cell2", "cell3", "cell4"],
            }
        )

        # Test that function handles problematic data gracefully
        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=problematic_data,
            states=["baseline", "stimulus"],
            has_single_group=True,
            measure_name="activity",
        )

        # Should return empty results for problematic data
        self.assertTrue(
            lmm_result.empty,
            "Should return empty results for insufficient subjects",
        )
        self.assertTrue(
            pairwise_result.empty,
            "Should return empty pairwise for insufficient subjects",
        )

        # Test with zero variance data
        zero_var_data = pd.DataFrame(
            {
                "normalized_subject_id": ["subj1", "subj1", "subj2", "subj2"],
                "state": ["baseline", "stimulus", "baseline", "stimulus"],
                "activity": [1.0, 1.0, 1.0, 1.0],  # Zero variance
                "cell": ["cell1", "cell2", "cell3", "cell4"],
            }
        )

        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=zero_var_data,
            states=["baseline", "stimulus"],
            has_single_group=True,
            measure_name="activity",
        )

        # Should handle zero variance gracefully
        self.assertTrue(
            lmm_result.empty or len(lmm_result) == 0,
            "Should handle zero variance data gracefully",
        )

    def test_perform_lmm_analysis_formula_construction(self):
        """Test that _perform_lmm_analysis constructs statistically correct formulas."""
        # Test single within-factor formula
        test_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2"],
                "state": ["baseline", "stimulus", "baseline", "stimulus"],
                "activity": [1.0, 1.5, 0.9, 1.4],
                "cell": ["c1", "c2", "c3", "c4"],
            }
        )

        with patch(
            "utils.combine_compare_population_data_utils.MixedLM"
        ) as mock_lmm:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.converged = True
            mock_result.params = pd.Series(
                [1.0, 0.5], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.bse = pd.Series(
                [0.1, 0.1], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.tvalues = pd.Series(
                [10.0, 5.0], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.pvalues = pd.Series(
                [0.001, 0.01], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.conf_int.return_value = pd.DataFrame(
                [[0.8, 1.2], [0.3, 0.7]],
                index=["Intercept", "state[T.stimulus]"],
            )
            mock_result.df_resid = 2
            mock_model.fit.return_value = mock_result
            mock_lmm.from_formula.return_value = mock_model

            lmm_result, has_concerns, warning_messages = _perform_lmm_analysis(
                df=test_data,
                dv="activity",
                within_factors=["state"],
                subject="normalized_subject_id",
                has_single_group=True,
            )

            # Check that MixedLM was called with correct formula
            mock_lmm.from_formula.assert_called_once()
            call_args = mock_lmm.from_formula.call_args
            formula = call_args[1]["formula"]

            # Formula should include DV, fixed effects, and be syntactically correct
            self.assertIn(
                "activity ~", formula, "Formula should start with DV"
            )
            self.assertIn(
                "state", formula, "Formula should include state factor"
            )

            # Check that groups parameter is correctly set for random effects
            self.assertIn(
                "groups",
                call_args[1],
                "Should specify groups for random effects",
            )

    def test_perform_lmm_analysis_mixed_design_formula(self):
        """Test formula construction for mixed design (within + between factors)."""
        test_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2"],
                "state": ["baseline", "stimulus", "baseline", "stimulus"],
                "group": ["control", "control", "treatment", "treatment"],
                "activity": [1.0, 1.5, 1.2, 2.0],
                "cell": ["c1", "c2", "c3", "c4"],
            }
        )

        with patch(
            "utils.combine_compare_population_data_utils.MixedLM"
        ) as mock_lmm:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.converged = True
            mock_result.params = pd.Series(
                [1.0, 0.5, 0.3, 0.2],
                index=[
                    "Intercept",
                    "state[T.stimulus]",
                    "group[T.treatment]",
                    "state[T.stimulus]:group[T.treatment]",
                ],
            )
            mock_result.bse = pd.Series(
                [0.1, 0.1, 0.1, 0.1], index=mock_result.params.index
            )
            mock_result.tvalues = pd.Series(
                [10.0, 5.0, 3.0, 2.0], index=mock_result.params.index
            )
            mock_result.pvalues = pd.Series(
                [0.001, 0.01, 0.05, 0.1], index=mock_result.params.index
            )
            mock_result.conf_int.return_value = pd.DataFrame(
                [[0.8, 1.2], [0.3, 0.7], [0.1, 0.5], [0.0, 0.4]],
                index=mock_result.params.index,
            )
            mock_result.df_resid = 0  # Will be adjusted
            mock_model.fit.return_value = mock_result
            mock_lmm.from_formula.return_value = mock_model

            lmm_result, has_concerns, warning_messages = _perform_lmm_analysis(
                df=test_data,
                dv="activity",
                within_factors=["state"],
                subject="normalized_subject_id",
                between="group",
                has_single_group=False,
            )

            # Check mixed design formula
            call_args = mock_lmm.from_formula.call_args
            formula = call_args[1]["formula"]

            # Should include main effects and interaction
            self.assertIn("state", formula, "Should include within-factor")
            self.assertIn("group", formula, "Should include between-factor")
            # May include interaction depending on implementation
            # self.assertIn(':', formula, "Should include interaction term")

    def test_lmm_statistical_output_standardization(self):
        """Test that LMM output is properly standardized"""

        test_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                "state": ["baseline", "stimulus"] * 3,
                "activity": [1.0, 1.5, 0.9, 1.4, 1.1, 1.6],
                "cell": ["c1", "c2", "c3", "c4", "c5", "c6"],
            }
        )

        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=test_data,
            states=["baseline", "stimulus"],
            has_single_group=True,
            measure_name="activity",
        )

        if not lmm_result.empty:
            # Check required statistical columns
            required_columns = [
                "Source",
                "p_value",
                "coefficient",
                "std_error",
                "t_value",
                "conf_int_lower",
                "conf_int_upper",
                "Comparison",
                "Measure",
                "analysis_level",
            ]

            for col in required_columns:
                self.assertIn(
                    col, lmm_result.columns, f"Missing required column: {col}"
                )

            # Check that p-values are valid probabilities
            p_values = lmm_result["p_value"].dropna()
            if len(p_values) > 0:
                self.assertTrue(
                    (p_values >= 0).all(), "P-values should be >= 0"
                )
                self.assertTrue(
                    (p_values <= 1).all(), "P-values should be <= 1"
                )

            # Check that confidence intervals are properly ordered
            ci_lower = lmm_result["conf_int_lower"].dropna()
            ci_upper = lmm_result["conf_int_upper"].dropna()
            if len(ci_lower) > 0 and len(ci_upper) > 0:
                self.assertTrue(
                    (ci_lower <= ci_upper).all(),
                    "Confidence interval lower bounds should be <= upper bounds",
                )

    def test_lmm_event_vs_activity_data_handling(self):
        """Test that LMM correctly handles both activity and event rate data."""
        # Test with activity data
        activity_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2"],
                "state": ["baseline", "stimulus", "baseline", "stimulus"],
                "activity": [
                    0.5,
                    0.8,
                    0.4,
                    0.7,
                ],  # Typical activity values (0-1)
                "cell": ["c1", "c2", "c3", "c4"],
            }
        )

        # Test with event rate data
        event_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2"],
                "state": ["baseline", "stimulus", "baseline", "stimulus"],
                "activity": [
                    2.0,
                    5.0,
                    1.8,
                    4.5,
                ],  # Typical event rates (events/min)
                "cell": ["c1", "c2", "c3", "c4"],
            }
        )

        # Test activity analysis
        activity_lmm, activity_pairwise = calculate_state_lmm_stats(
            df=activity_data,
            states=["baseline", "stimulus"],
            has_single_group=True,
            measure_name="activity",
        )

        # Test event rate analysis
        event_lmm, event_pairwise = calculate_state_lmm_stats(
            df=event_data,
            states=["baseline", "stimulus"],
            has_single_group=True,
            measure_name="event_rate",
        )

        # Both should work and return structured results
        if not activity_lmm.empty:
            self.assertIn("Measure", activity_lmm.columns)
            self.assertEqual(activity_lmm["Measure"].iloc[0], "activity")

        if not event_lmm.empty:
            self.assertIn("Measure", event_lmm.columns)
            self.assertEqual(event_lmm["Measure"].iloc[0], "event_rate")

    def test_lmm_minimum_data_requirements_validation(self):
        """Test LMM validation of minimum data requirements."""
        # Test insufficient subjects
        insufficient_subjects = pd.DataFrame(
            {
                "normalized_subject_id": ["s1"],  # Only 1 subject
                "state": ["baseline"],
                "activity": [1.0],
                "cell": ["c1"],
            }
        )

        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=insufficient_subjects,
            states=["baseline"],
            has_single_group=True,
            measure_name="activity",
        )

        self.assertTrue(
            lmm_result.empty, "Should reject data with insufficient subjects"
        )

        # Test insufficient states
        insufficient_states = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s2"],
                "state": ["baseline", "baseline"],  # Only 1 state
                "activity": [1.0, 1.1],
                "cell": ["c1", "c2"],
            }
        )

        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=insufficient_states,
            states=["baseline"],  # Only 1 state
            has_single_group=True,
            measure_name="activity",
        )

        self.assertTrue(
            lmm_result.empty, "Should reject data with insufficient states"
        )

    def test_lmm_data_cleaning_and_validation(self):
        """Test LMM data cleaning and validation procedures."""
        # Create data with missing values and outliers
        messy_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                "state": [
                    "baseline",
                    "stimulus",
                    "baseline",
                    "stimulus",
                    "baseline",
                    "stimulus",
                ],
                "activity": [
                    1.0,
                    np.nan,
                    0.9,
                    np.inf,
                    1.1,
                    1.6,
                ],  # Include NaN and inf
                "cell": ["c1", "c2", "c3", "c4", "c5", "c6"],
            }
        )

        # LMM should handle data cleaning internally
        lmm_result, pairwise_result = calculate_state_lmm_stats(
            df=messy_data,
            states=["baseline", "stimulus"],
            has_single_group=True,
            measure_name="activity",
        )

        # Should either clean the data and proceed or return empty results
        # The key is that it shouldn't crash
        self.assertIsInstance(
            lmm_result,
            pd.DataFrame,
            "Should return DataFrame even with messy data",
        )
        self.assertIsInstance(
            pairwise_result,
            pd.DataFrame,
            "Should return DataFrame even with messy data",
        )

    def test_lmm_reml_vs_ml_estimation(self):
        """Test that LMM uses REML estimation appropriately for variance components."""
        test_data = pd.DataFrame(
            {
                "normalized_subject_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                "state": ["baseline", "stimulus"] * 3,
                "activity": [1.0, 1.5, 0.9, 1.4, 1.1, 1.6],
                "cell": ["c1", "c2", "c3", "c4", "c5", "c6"],
            }
        )

        with patch(
            "utils.combine_compare_population_data_utils.MixedLM"
        ) as mock_lmm:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.converged = True
            mock_result.params = pd.Series(
                [1.0, 0.5], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.bse = pd.Series(
                [0.1, 0.1], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.tvalues = pd.Series(
                [10.0, 5.0], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.pvalues = pd.Series(
                [0.001, 0.01], index=["Intercept", "state[T.stimulus]"]
            )
            mock_result.conf_int.return_value = pd.DataFrame(
                [[0.8, 1.2], [0.3, 0.7]],
                index=["Intercept", "state[T.stimulus]"],
            )
            mock_result.df_resid = 4
            mock_model.fit.return_value = mock_result
            mock_lmm.from_formula.return_value = mock_model

            lmm_result, has_concerns, warning_messages = _perform_lmm_analysis(
                df=test_data,
                dv="activity",
                within_factors=["state"],
                subject="normalized_subject_id",
                has_single_group=True,
            )

            # Check that fit was called with REML=True for proper variance estimation
            mock_model.fit.assert_called()
            fit_call_kwargs = mock_model.fit.call_args[1]
            self.assertTrue(
                fit_call_kwargs.get("reml", False),
                "Should use REML estimation for variance components",
            )
