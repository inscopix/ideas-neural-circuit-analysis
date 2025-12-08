"""Tests for statistical validation utilities."""

from typing import Dict, Any

import pandas as pd
from unittest.mock import patch

from utils.statistical_validation import (
    _safe_pairwise_ttests,
    _PAIRWISE_TESTS_PARAMS,
)


def _build_mock_pairwise_data() -> pd.DataFrame:
    """Create a minimal pairwise result dataframe."""
    return pd.DataFrame(
        {
            "A": ["rest"],
            "B": ["active"],
            "T": [0.0],
            "p-unc": [1.0],
            "state": ["rest"],
        }
    )


def _build_input_data() -> pd.DataFrame:
    """Create input data for pairwise testing."""
    return pd.DataFrame(
        {
            "activity": [1.0, 1.1, 0.9, 1.2],
            "state": ["rest", "rest", "active", "active"],
            "group_name": ["Group 1", "Group 2", "Group 1", "Group 2"],
            "normalized_subject_id": ["subj_1", "subj_2", "subj_1", "subj_2"],
        }
    )


def test_safe_pairwise_ttests_maps_one_tailed_less() -> None:
    """Verify that one-tailed comparisons are mapped to Pingouin's alternative arg."""
    input_data = _build_input_data()
    expected_result = _build_mock_pairwise_data()
    captured_kwargs: Dict[str, Any] = {}

    def _mock_pairwise_tests(*, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        captured_kwargs.update(kwargs)
        return expected_result

    with patch(
        "utils.statistical_validation.pg.pairwise_tests",
        side_effect=_mock_pairwise_tests,
    ) as mock_pairwise:
        result = _safe_pairwise_ttests(
            data=input_data,
            dv="activity",
            within="state",
            between="group_name",
            subject="normalized_subject_id",
            padjust="bonf",
            effsize="cohen",
            tail="one_tailed_less",
            parametric="auto",
        )

    assert mock_pairwise.called, "pairwise_tests should be invoked once"
    if "alternative" in _PAIRWISE_TESTS_PARAMS:
        assert captured_kwargs.get("alternative") == "less"
        assert "tail" not in captured_kwargs
    else:
        assert captured_kwargs.get("tail") == "less"

    assert captured_kwargs.get("parametric") is False
    pd.testing.assert_frame_equal(result, expected_result)
