"""Consistency checks across higher-level combine/compare utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.combine_compare_state_epoch_data import (
    _load_group_data,
    _perform_statistical_comparison_csv,
)


def _write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


class TestCombineCompareConsistency(unittest.TestCase):
    """Ensure helper utilities behave coherently for minimal datasets."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        base = self.tmpdir.name

        def p(name):
            return Path(base) / name

        # Create tiny per-subject CSVs
        activity_rows = []
        correlation_rows = []
        modulation_rows = []
        for idx in range(2):
            for state in ["rest", "active"]:
                activity_rows.append(
                    {
                        "name": f"cell_{idx}",
                        "state": state,
                        "epoch": "baseline",
                        "mean_activity": np.random.rand(),
                    }
                )
                correlation_rows.append(
                    {
                        "name": f"cell_{idx}",
                        "state": state,
                        "epoch": "baseline",
                        "max_trace_correlation": 0.1,
                        "min_trace_correlation": -0.1,
                        "mean_trace_correlation": 0.0,
                    }
                )
            modulation_rows.append(
                {
                    "name": f"cell_{idx}",
                    "state": "active",
                    "epoch": "baseline",
                    "baseline_state": "rest",
                    "baseline_epoch": "baseline",
                    "modulation_scores": 0.1,
                }
            )

        self.activity_file = p("activity.csv")
        self.correlation_file = p("corr.csv")
        self.modulation_file = p("mod.csv")
        _write_csv(activity_rows, self.activity_file)
        _write_csv(correlation_rows, self.correlation_file)
        _write_csv(modulation_rows, self.modulation_file)

        self.group_data = _load_group_data(
            activity_csv_files=[self.activity_file],
            correlation_csv_files=[self.correlation_file],
            modulation_csv_files=[self.modulation_file],
            group_name="Group 1",
            group_id=1,
            data_pairing="unpaired",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_group_data_has_expected_tables(self):
        self.assertIn("activity", self.group_data)
        self.assertIn("correlation", self.group_data)
        self.assertIn("modulation", self.group_data)
        self.assertFalse(self.group_data["activity"].empty)

    def test_statistical_comparison_returns_dataframes(self):
        results = _perform_statistical_comparison_csv(
            group1_data=self.group_data,
            group2_data=None,
            group_names=["Group 1"],
            comparison_dimension="states",
            states=["rest", "active"],
            epochs=["baseline"],
            baseline_state="rest",
            baseline_epoch="baseline",
            data_pairing="unpaired",
            correlation_statistic="mean",
            significance_threshold=0.05,
            multiple_correction="bonf",
            effect_size="cohen",
            output_dir=self.tmpdir.name,
        )
        self.assertIn("aov", results)
        self.assertIn("pairwise", results)
        self.assertIsInstance(results["aov"], pd.DataFrame)
        self.assertIsInstance(results["pairwise"], pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
