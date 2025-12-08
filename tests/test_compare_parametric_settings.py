"""Tests for feature flag helpers controlling parametric settings."""

import unittest

from analysis.state_epoch_baseline_analysis import (
    configure_state_epoch_analysis_feature_flags,
    get_state_epoch_analysis_feature_flags,
    temporary_state_epoch_analysis_feature_flags,
)


class TestFeatureFlagConfiguration(unittest.TestCase):
    def tearDown(self):
        configure_state_epoch_analysis_feature_flags()

    def test_configure_updates_global_flags(self):
        configure_state_epoch_analysis_feature_flags(
            include_correlations=False, include_event_analysis=False
        )
        flags = get_state_epoch_analysis_feature_flags()
        self.assertFalse(flags.include_correlations)
        self.assertFalse(flags.include_event_analysis)

    def test_configure_without_overrides_resets_defaults(self):
        configure_state_epoch_analysis_feature_flags(include_population_activity=False)
        configure_state_epoch_analysis_feature_flags()  # reset
        flags = get_state_epoch_analysis_feature_flags()
        self.assertTrue(flags.include_population_activity)

    def test_temporary_context_restores_previous_state(self):
        configure_state_epoch_analysis_feature_flags(include_correlations=False)
        with temporary_state_epoch_analysis_feature_flags(
            include_correlations=True, include_event_analysis=False
        ):
            flags = get_state_epoch_analysis_feature_flags()
            self.assertTrue(flags.include_correlations)
            self.assertFalse(flags.include_event_analysis)
        flags_after = get_state_epoch_analysis_feature_flags()
        self.assertFalse(flags_after.include_correlations)
        self.assertTrue(flags_after.include_event_analysis)


if __name__ == "__main__":
    unittest.main()
