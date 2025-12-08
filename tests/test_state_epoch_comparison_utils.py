"""Regression tests for `utils.state_epoch_comparison_utils` helpers.

These tests are a trimmed version of the toolbox suite and focus on the
behaviours that integrate with the neural-circuit toolbox (ANOVAs, pairwise
tests, subject matching, plotting helpers, and correction utilities).
"""

import os
import tempfile
import unittest
from typing import List

import numpy as np
import pandas as pd

from utils.state_epoch_comparison_utils import (
    IMBALANCED_DESIGN_RATIO_THRESHOLD,
    MIN_OBSERVATIONS_CAUTIOUS_LMM,
    MIN_OBSERVATIONS_LMM,
    MIN_SUBJECTS_CAUTIOUS,
    MIN_SUBJECTS_RECOMMENDED,
    MIN_SUBJECTS_REQUIRED,
    calculate_state_epoch_comparison_stats,
    match_subjects,
    plot_state_epoch_comparison,
    _apply_global_multiple_comparison_correction,
    _perform_between_group_anova,
    _perform_pairwise_tests,
    _perform_within_group_anova,
    _require_supported_multiple_correction_method,
)


class _BaseStatsTest(unittest.TestCase):
    """Shared fixtures for comparison testing."""

    STATES = ["rest", "active"]
    EPOCHS = ["baseline", "training"]

    def setUp(self):
        np.random.seed(42)
        self.simple_df = self._build_dataset(groups=["Control"])

    def _build_dataset(self, groups: List[str]) -> pd.DataFrame:
        records = []
        for group in groups:
            for subj_idx in range(4):
                subject = f"{group}_subject_{subj_idx+1}"
                for state in self.STATES:
                    for epoch in self.EPOCHS:
                        records.append(
                            {
                                "group_name": group,
                                "normalized_subject_id": subject,
                                "state": state,
                                "epoch": epoch,
                                "mean_activity": np.random.randn() * 0.25
                                + (1.0 if group == "Control" else 1.5),
                            }
                        )
        return pd.DataFrame(records)


class TestAnovaHelpers(_BaseStatsTest):
    def test_within_group_anova_returns_expected_columns(self):
        result = _perform_within_group_anova(
            data=self.simple_df,
            comparison_col="state",
            measure_col="mean_activity",
            subject_col="normalized_subject_id",
            context="unit",
        )
        self.assertFalse(result.empty)
        self.assertTrue({"F_statistic", "p_value"}.issubset(result.columns))

    def test_between_group_anova_detects_group_column(self):
        df = self._build_dataset(groups=["Control", "Treatment"])
        result = _perform_between_group_anova(
            data=df,
            comparison_col="state",
            measure_col="mean_activity",
            subject_col="normalized_subject_id",
            group_col="group_name",
            data_pairing="unpaired",
            context="between",
        )
        self.assertFalse(result.empty)
        self.assertIn("Source", result.columns)
        self.assertTrue(
            any(
                src in {"group_effect", "group_name"}
                for src in result["Source"].unique()
            )
        )

    def test_calculate_state_epoch_comparison_stats_runs(self):
        df = self._build_dataset(groups=["Control", "Treatment"])
        aov, pairwise = calculate_state_epoch_comparison_stats(
            df=df,
            comparison_dimension="states",
            comparison_values=self.STATES,
            has_single_group=False,
            data_pairing="unpaired",
            significance_threshold=0.05,
            multiple_correction="bonf",
            effect_size_method="cohen",
            group_comparison_type="two_tailed",
            parametric="auto",
            measure_column="mean_activity",
        )
        self.assertIsInstance(aov, pd.DataFrame)
        self.assertIsInstance(pairwise, pd.DataFrame)


class TestPairwiseAndCorrection(_BaseStatsTest):
    def test_pairwise_tests_between_groups(self):
        df = self._build_dataset(groups=["Control", "Treatment"])
        pairwise = _perform_pairwise_tests(
            data=df,
            comparison_col="state",
            comparison_values=self.STATES,
            measure_col="mean_activity",
            subject_col="normalized_subject_id",
            group_col="group_name",
            data_pairing="unpaired",
            correction="bonf",
            effect_size="cohen",
            group_comparison_type="two_tailed",
            parametric="auto",
            context="states",
        )
        self.assertTrue({"Contrast", "p-unc"}.issubset(pairwise.columns))

    def test_global_correction_metadata_added(self):
        df = pd.DataFrame(
            {"Contrast": ["A-B", "A-C", "B-C"], "p-unc": [0.01, 0.02, 0.2]}
        )
        corrected = _apply_global_multiple_comparison_correction(
            df, correction_method="bonf"
        )
        self.assertTrue({"p-corr", "correction_scope"}.issubset(corrected.columns))
        self.assertEqual(corrected["correction_scope"].iloc[0], "global")

    def test_multiple_correction_aliases(self):
        self.assertEqual(
            _require_supported_multiple_correction_method("bonferroni"), "bonf"
        )
        self.assertEqual(_require_supported_multiple_correction_method("fdr"), "fdr_bh")


class TestSubjectMatching(unittest.TestCase):
    def test_match_subjects_by_order(self):
        group1 = ["subj1.isxd", "subj2.isxd", "subj3.isxd"]
        group2 = ["other1.isxd", "other2.isxd"]
        matches = match_subjects(group1, group2, match_method="order")
        self.assertEqual(matches, list(zip(group1[:2], group2)))

    def test_match_subjects_requires_min_pairs(self):
        with self.assertRaises(ValueError):
            match_subjects(["file1.isxd"], ["file2.isxd"], match_method="order")


class TestPlotting(unittest.TestCase):
    def test_plot_state_epoch_comparison_writes_svg(self):
        df = pd.DataFrame(
            {
                "group_name": ["Group 1"] * 4,
                "state": ["rest", "active"] * 2,
                "epoch": ["baseline", "training"] * 2,
                "mean_activity": np.random.rand(4),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "plot.svg")
            plot_state_epoch_comparison(
                data=df,
                comparison_dimension="states",
                states=["rest", "active"],
                epochs=["baseline", "training"],
                baseline_state="rest",
                baseline_epoch="baseline",
                group_names=["Group 1"],
                group_colors=["blue"],
                state_colors={"rest": "gray", "active": "orange"},
                epoch_colors={"baseline": "lightgray", "training": "skyblue"},
                pairwise_results=None,
                output_dir=tmpdir,
                output_filename=out_file,
                data_type="activity",
            )
            self.assertTrue(os.path.exists(out_file))
            self.assertGreater(os.path.getsize(out_file), 0)


class TestStatisticalConstants(unittest.TestCase):
    def test_threshold_relationships(self):
        self.assertLess(MIN_SUBJECTS_REQUIRED, MIN_SUBJECTS_CAUTIOUS)
        self.assertLess(MIN_SUBJECTS_CAUTIOUS, MIN_SUBJECTS_RECOMMENDED)
        self.assertLess(MIN_OBSERVATIONS_LMM, MIN_OBSERVATIONS_CAUTIOUS_LMM)
        self.assertGreater(IMBALANCED_DESIGN_RATIO_THRESHOLD, 0)


if __name__ == "__main__":
    unittest.main()
