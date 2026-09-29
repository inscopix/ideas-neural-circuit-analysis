import pytest
import pandas as pd
from ideas.exceptions import IdeasError

from toolbox.tools import sync_annotations


def test_get_start_tsc_from_isxd_metadata_valid(monkeypatch):
    """Return startTsc when metadata contains extraProperties.startTsc."""
    monkeypatch.setattr(
        sync_annotations,
        "_extract_footer",
        lambda _: {"extraProperties": {"startTsc": 123456}},
    )

    assert (
        sync_annotations._get_start_tsc_from_isxd_metadata("input.isxd")
        == 123456
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"extraProperties": None},
        {"extraProperties": {}},
        {"extraProperties": {"startTsc": None}},
    ],
)
def test_get_start_tsc_from_isxd_metadata_missing_raises(
    monkeypatch, metadata
):
    """Raise IdeasError when extraProperties.startTsc is missing."""
    monkeypatch.setattr(
        sync_annotations, "_extract_footer", lambda _: metadata
    )

    with pytest.raises(IdeasError, match="Could not find startTsc metadata"):
        sync_annotations._get_start_tsc_from_isxd_metadata("input.isxd")


def test_get_alignment_method_and_start_tsc_no_hardware_column():
    """Return time-based alignment when hardware counter column is absent."""
    method, start_tsc = sync_annotations._get_alignment_method_and_start_tsc(
        annotations_df=pd.DataFrame({"time": [0.0], "state": ["a"]}),
        isxd_file_for_tsc="input.isxd",
        use_hardware_tsc_alignment=True,
    )
    assert method == sync_annotations.START_TIME_ALIGNMENT_METHOD
    assert start_tsc is None


def test_get_alignment_method_and_start_tsc_disabled_hardware_alignment():
    """Return time-based alignment when hardware alignment is disabled."""
    method, start_tsc = sync_annotations._get_alignment_method_and_start_tsc(
        annotations_df=pd.DataFrame(
            {
                "time": [0.0],
                "state": ["a"],
                sync_annotations.HARDWARE_COUNTER_TIME_COLUMN: [1],
            }
        ),
        isxd_file_for_tsc="input.isxd",
        use_hardware_tsc_alignment=False,
    )
    assert method == sync_annotations.START_TIME_ALIGNMENT_METHOD
    assert start_tsc is None


def test_get_alignment_method_and_start_tsc_hardware_alignment(
    monkeypatch,
):
    """Return hardware_tsc alignment and resolved start tsc when enabled."""
    monkeypatch.setattr(
        sync_annotations,
        "_get_start_tsc_from_isxd_metadata",
        lambda _: 123456,
    )
    method, start_tsc = sync_annotations._get_alignment_method_and_start_tsc(
        annotations_df=pd.DataFrame(
            {
                "time": [0.0],
                "state": ["a"],
                sync_annotations.HARDWARE_COUNTER_TIME_COLUMN: [1],
            }
        ),
        isxd_file_for_tsc="input.isxd",
        use_hardware_tsc_alignment=True,
    )
    assert method == sync_annotations.HARDWARE_TSC_ALIGNMENT_METHOD
    assert start_tsc == 123456


def test_get_alignment_method_and_start_tsc_missing_start_tsc_raises(
    monkeypatch,
):
    """Raise IdeasError when hardware alignment is enabled but startTsc is missing."""

    def _mock_raise(_):
        raise IdeasError("Could not find startTsc metadata")

    monkeypatch.setattr(
        sync_annotations, "_get_start_tsc_from_isxd_metadata", _mock_raise
    )

    with pytest.raises(IdeasError, match="Could not find startTsc metadata"):
        sync_annotations._get_alignment_method_and_start_tsc(
            annotations_df=pd.DataFrame(
                {
                    "time": [0.0],
                    "state": ["a"],
                    sync_annotations.HARDWARE_COUNTER_TIME_COLUMN: [1],
                }
            ),
            isxd_file_for_tsc="input.isxd",
            use_hardware_tsc_alignment=True,
        )


def test_sync_csv_to_annotations_missing_start_tsc_raises_when_enabled(
    monkeypatch,
):
    """Raise during sync when strict hardware alignment cannot resolve startTsc."""
    input_df = pd.DataFrame(
        {
            "time": [0.0],
            "state": ["a"],
            sync_annotations.HARDWARE_COUNTER_TIME_COLUMN: [123456],
        }
    )

    monkeypatch.setattr(
        sync_annotations, "movie_series", lambda input_files: input_files
    )
    monkeypatch.setattr(
        sync_annotations,
        "_get_start_time_from_isxd_metadata",
        lambda _: 0.0,
    )
    monkeypatch.setattr(sync_annotations.pd, "read_csv", lambda _: input_df)

    def _mock_raise(_):
        raise IdeasError("Could not find startTsc metadata")

    monkeypatch.setattr(
        sync_annotations, "_get_start_tsc_from_isxd_metadata", _mock_raise
    )

    with pytest.raises(IdeasError, match="Could not find startTsc metadata"):
        sync_annotations.sync_csv_to_annotations(
            isxd_files=["input.isxd"],
            annotations_files=["input.csv"],
            use_hardware_tsc_alignment=True,
        )
