"""Unit tests for multiple comparison correction helpers."""

import unittest

import numpy as np
import pandas as pd

from utils.state_epoch_comparison_utils import (
    _apply_global_multiple_comparison_correction,
)


class TestMultipleComparisonCorrection(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Contrast": ["A-B", "A-C", "B-C"],
                "p-unc": [0.01, 0.015, 0.02],
                "A": ["A", "A", "B"],
                "B": ["B", "C", "C"],
            }
        )

    def test_bonferroni_scales_by_number_of_tests(self):
        corrected = _apply_global_multiple_comparison_correction(
            self.df.copy(), correction_method="bonf"
        )
        expected = np.minimum(np.array([0.01, 0.015, 0.02]) * 3, 1.0)
        np.testing.assert_allclose(corrected["p-corr"].values, expected)

    def test_fdr_is_less_conservative_than_bonferroni(self):
        corrected_fdr = _apply_global_multiple_comparison_correction(
            self.df.copy(), correction_method="fdr_bh"
        )
        corrected_bonf = _apply_global_multiple_comparison_correction(
            self.df.copy(), correction_method="bonf"
        )
        self.assertLess(
            corrected_fdr["p-corr"].iloc[0], corrected_bonf["p-corr"].iloc[0]
        )

    def test_metadata_columns_present(self):
        corrected = _apply_global_multiple_comparison_correction(
            self.df.copy(), correction_method="bonf"
        )
        self.assertTrue({"p-corr", "correction_scope"}.issubset(corrected.columns))
        self.assertEqual(corrected["correction_scope"].iloc[0], "global")


if __name__ == "__main__":
    unittest.main()
