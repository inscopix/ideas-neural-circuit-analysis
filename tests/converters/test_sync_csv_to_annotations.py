import os
import re

import numpy as np
import pandas as pd
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from ideas.analysis import io
from ideas.analysis.validation import _check_columns_in_df
from ideas.exceptions import IdeasError

from converters.sync_annotations import (
    ALIGNMENT_METHOD_COLUMN,
    _get_start_tsc_from_isxd_metadata,
    _map_frames,
    sync_csv_to_annotations,
)

annotations_files = [
    "data/cellset_series_1-annotations.csv",
    "data/cellset_series_2-annotations.csv",
]

cell_set_files = [
    "data/cellset_series_1.isxd",
    "data/cellset_series_2.isxd",
]


time_column = "time"
state_column = "state"


valid_inputs = [
    # series inputs
    dict(
        isxd_files=cell_set_files,
        annotations_files=annotations_files,
    ),
    # single inputs
    dict(
        isxd_files=[cell_set_files[0]],
        annotations_files=[annotations_files[0]],
    ),
    # cell set series with a single annotations file
    dict(
        isxd_files=cell_set_files,
        annotations_files=[annotations_files[0]],
    ),
]

invalid_inputs = [
    # wrong type
    dict(
        isxd_files=cell_set_files[0],
        annotations_files=annotations_files[0],
        error=BeartypeCallHintParamViolation,
        error_text="violates type hint",
    ),
    # wrong combination of series and not
    dict(
        isxd_files=[cell_set_files[0]],
        annotations_files=annotations_files,
        error=Exception,
        error_text="Expected to get the same number of isxd files",
    ),
    # invalid column info
    dict(
        isxd_files=cell_set_files,
        annotations_files=annotations_files,
        time_column="dsfds",
        error=IdeasError,
        error_text="Data frame does not contain\n the column requested: dsfds\n\nThe columns in this data frame are:\n['time', 'state']",
    ),
]


@pytest.mark.parametrize(
    [
        "isxd_times",
        "annotations_times",
        "isxd_epoch_start_times",
        "annotations_epoch_start_times",
        "expected_annotations_mapped_frames",
        "expected_isxd_times_series",
        "expected_annotations_times_series",
    ],
    [
        (
            # annotations start after isxd
            [pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])],
            [pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])],
            [10],
            [10.5],
            pd.Series([np.nan, np.nan, np.nan, 0, 0, 0, 0, 1, 1, 2, 2]),
            pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            pd.Series([0.5, 0.75, 1.0, 1.25, 1.5]),
        ),
        (
            # annotations start before isxd
            [pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])],
            [pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])],
            [10],
            [9.5],
            pd.Series([2, 2, 3, 3, 4, 4, 4, 4, np.nan, np.nan, np.nan]),
            pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            pd.Series([-0.5, -0.25, 0.0, 0.25, 0.5]),
        ),
        (
            # series test
            [
                pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                pd.Series([0.0, 0.1, 0.2, 0.3, 0.4]),
            ],
            [pd.Series([0.0, 0.25, 0.5]), pd.Series([0.0, 0.25])],
            [10, 11],
            [10.1, 11.1],
            pd.Series([0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4]),
            pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4]),
            pd.Series([0.1, 0.35, 0.6, 1.1, 1.35]),
        ),
    ],
)
def test_map_frames(
    isxd_times,
    annotations_times,
    isxd_epoch_start_times,
    annotations_epoch_start_times,
    expected_annotations_mapped_frames,
    expected_isxd_times_series,
    expected_annotations_times_series,
):
    """Tests _map_frames util function."""

    (
        isxd_times_series,
        annotations_time_series,
        annotations_mapped_frames,
    ) = _map_frames(
        ref_recordings_times=isxd_times,
        align_recordings_times=annotations_times,
        ref_epoch_start_times=isxd_epoch_start_times,
        align_epoch_start_times=annotations_epoch_start_times,
    )

    # very tiny floating point differences occur sometimes which is why assert_allclose is used
    np.testing.assert_allclose(isxd_times_series, expected_isxd_times_series)
    np.testing.assert_allclose(
        annotations_time_series, expected_annotations_times_series
    )
    np.testing.assert_allclose(
        annotations_mapped_frames, expected_annotations_mapped_frames
    )


@pytest.mark.parametrize("params", valid_inputs)
def test_converter_valid_inputs(params):
    """Tests the basic converter."""

    sync_csv_to_annotations(**params)

    output_file = "annotations.parquet"

    # check that output file is produced
    if not os.path.exists(output_file):
        raise RuntimeError("output file not generated")

    df = pd.read_parquet(output_file)

    # check that the dataframe has the correct columns
    _check_columns_in_df(
        df=df,
        columns=("time", "state", ALIGNMENT_METHOD_COLUMN),
    )
    assert set(df[ALIGNMENT_METHOD_COLUMN].unique()) == {"start_time"}

    # check that the time is the same in
    # the annotations and cellset file
    time = io.cell_set_to_time(params["isxd_files"])

    assert np.max(df["time"] - time) == 0, (
        "Time in annotations and cellset is not the same"
    )


@pytest.mark.parametrize("params", invalid_inputs)
def test_converter_invalid_inputs(params):
    """Tests the basic converter."""

    error = params.pop("error")
    error_text = params.pop("error_text")
    with pytest.raises(error, match=re.escape(error_text)):
        sync_csv_to_annotations(**params)


@pytest.mark.parametrize(
    [
        "annotations_files",
        "cell_set_files",
        "time_offsets",
    ],
    [
        (
            # series inputs, no time delays between isxd and annotations
            annotations_files,
            cell_set_files,
            [0.0, 0.0, 0.0],
        ),
        (
            # series inputs, time delays between isxd and annotations
            annotations_files,
            cell_set_files,
            [3.0, 30.0, -10.0],
        ),
        (
            # single input
            [annotations_files[0]],
            [cell_set_files[0]],
            [3.0],
        ),
    ],
)
def test_converter_with_hardware_counter_valid_inputs(
    annotations_files,
    cell_set_files,
    time_offsets,
    hardware_counter_column="Hardware counter (us)",
    time_column="time",
    state_column="state",
):
    """Tests the basic converter."""

    tmp_annotations_files = []
    input_dfs = []
    for annotations_file, cell_set_file, time_offset in zip(
        annotations_files, cell_set_files, time_offsets
    ):
        # read the start tsc of the cell set
        start_tsc = _get_start_tsc_from_isxd_metadata(cell_set_file)

        annotations_file_path, annotations_file_name = os.path.split(annotations_file)
        tmp_annotations_file = os.path.join(
            annotations_file_path, f"tmp_{annotations_file_name}"
        )

        # add hardware tsc values to the input df for testing
        input_df = pd.read_csv(annotations_file)
        input_df[hardware_counter_column] = input_df[time_column] * 1e6
        input_df[hardware_counter_column] += start_tsc + (time_offset * 1e6)
        input_df[hardware_counter_column] = (
            input_df[hardware_counter_column].round().astype(int)
        )
        input_df.to_csv(tmp_annotations_file, index=False)

        input_dfs.append(input_df)
        tmp_annotations_files.append(tmp_annotations_file)

    input_df = pd.concat(input_dfs, ignore_index=True)
    sync_csv_to_annotations(
        annotations_files=tmp_annotations_files,
        isxd_files=cell_set_files,
        time_column=time_column,
    )

    output_file = "annotations.parquet"

    # check that output file is produced
    if not os.path.exists(output_file):
        raise RuntimeError("output file not generated")

    output_df = pd.read_parquet(output_file)

    # check that the dataframe has the correct columns
    _check_columns_in_df(
        df=output_df,
        columns=(
            time_column,
            state_column,
            hardware_counter_column,
            ALIGNMENT_METHOD_COLUMN,
        ),
    )
    assert set(output_df[ALIGNMENT_METHOD_COLUMN].unique()) == {"hardware_tsc"}

    # check that the time is the same in
    # the annotations and cellset file
    time = io.cell_set_to_time(cell_set_files)
    assert np.max(output_df[time_column] - time) == 0, (
        "Time in annotations and cellset is not the same"
    )

    # verify frames are mapped correctly
    for i in range(output_df.shape[0]):
        if pd.isna(output_df.iloc[i]["mapped frame"]):
            assert pd.isna(output_df.iloc[i]["Hardware counter (us)"])
        else:
            assert (
                output_df.iloc[i]["Hardware counter (us)"]
                == input_df.iloc[output_df.iloc[i]["mapped frame"]][
                    "Hardware counter (us)"
                ]
            )

    # clean-up
    for tmp_annotations_file in tmp_annotations_files:
        os.remove(tmp_annotations_file)


@pytest.mark.parametrize(
    (
        "annotations_files",
        "cell_set_files",
        "time_column",
        "state_column",
        "gpio_ref_file",
        "gpio_ref_channel",
        "gpio_ref_threshold",
        "expected_results",
    ),
    [
        # time-reference with positive channel threshold
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green.gpio",
            "BNC Sync Output",
            1,
            [
                # first row with mapped frame
                {
                    "row": 2,
                    "frame": 0,
                    "time": 0.08912491798400879,
                    "state": "",
                },
                # first row with mapped state
                {
                    "row": 11,
                    "frame": 14,
                    "time": 0.5571229179840088,
                    "state": "entry",
                },
                # last row with mapped frame
                {
                    "row": 224,
                    "frame": 332,
                    "time": 11.16112091798401,
                    "state": "",
                },
            ],
        ),
        # time-reference with no threshold
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green.gpio",
            "BNC Sync Output",
            None,
            [
                # first row with mapped frame
                {
                    "row": 0,
                    "frame": 0,
                    "time": -0.0010750293731689453,
                    "state": "",
                },
                # first row with mapped state
                {
                    "row": 10,
                    "frame": 15,
                    "time": 0.4989219706268311,
                    "state": "occupying",
                },
                # last row with mapped frame
                {
                    "row": 222,
                    "frame": 332,
                    "time": 11.070920970626831,
                    "state": "",
                },
            ],
        ),
        # time-reference with zero channel threshold
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green.gpio",
            "BNC Sync Output",
            0,
            [
                # first row with mapped frame
                {
                    "row": 0,
                    "frame": 0,
                    "time": -0.0010750293731689453,
                    "state": "",
                },
                # first row with mapped state
                {
                    "row": 10,
                    "frame": 15,
                    "time": 0.4989219706268311,
                    "state": "occupying",
                },
                # last row with mapped frame
                {
                    "row": 222,
                    "frame": 332,
                    "time": 11.070920970626831,
                    "state": "",
                },
            ],
        ),
    ],
)
def test_converter_with_gpio_ref(
    annotations_files,
    cell_set_files,
    time_column,
    state_column,
    gpio_ref_file,
    gpio_ref_channel,
    gpio_ref_threshold,
    expected_results,
):
    """test converter with gpio time-reference"""
    sync_csv_to_annotations(
        annotations_files=annotations_files,
        isxd_files=cell_set_files,
        time_column=time_column,
        state_column=state_column,
        gpio_ref_file=gpio_ref_file,
        gpio_ref_channel=gpio_ref_channel,
        gpio_ref_threshold=gpio_ref_threshold,
    )

    output_file = "annotations.parquet"

    # check that output file is produced
    if not os.path.exists(output_file):
        raise RuntimeError("output file not generated")

    output_df = pd.read_parquet(output_file)

    # check that the dataframe has the correct columns
    _check_columns_in_df(
        df=output_df,
        columns=("time", "state", ALIGNMENT_METHOD_COLUMN),
    )
    assert set(output_df[ALIGNMENT_METHOD_COLUMN].unique()) == {"start_time"}

    # check that the time is the same in
    # the annotations and cellset file
    time = io.cell_set_to_time(cell_set_files)
    assert np.max(output_df["time"] - time) == 0, (
        "Time in annotations and cellset is not the same"
    )

    # validate rows of output
    for expected_result in expected_results:
        assert (
            output_df.iloc[expected_result["row"]]["mapped frame"]
            == expected_result["frame"]
        )
        np.testing.assert_allclose(
            output_df.iloc[expected_result["row"]]["mapped time since start (s)"],
            expected_result["time"],
        )

        if expected_result["state"] == "":
            assert pd.isnull(output_df.iloc[expected_result["row"]]["state"])
        else:
            assert (
                output_df.iloc[expected_result["row"]]["state"]
                == expected_result["state"]
            )


@pytest.mark.parametrize(
    (
        "annotations_files",
        "cell_set_files",
        "time_column",
        "state_column",
        "gpio_ref_file",
        "gpio_ref_channel",
        "gpio_ref_threshold",
        "expected_error_message",
    ),
    [
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green.gpio",
            None,
            None,
            "Must provide channel name with gpio time reference file",
        ),
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green.gpio",
            "BND Sync Output",
            None,
            "Could not find channel BND Sync Output in input gpio file",
        ),
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green.gpio",
            "BNC Sync Output",
            500,
            "Unable to find timestamp in input gpio channel "
            "BNC Sync Output with a value greater than or equal to 500",
        ),
    ],
)
def test_converter_with_gpio_ref_invalid(
    annotations_files,
    cell_set_files,
    time_column,
    state_column,
    gpio_ref_file,
    gpio_ref_channel,
    gpio_ref_threshold,
    expected_error_message,
):
    """test converter with invalid inputs for gpio time-reference"""
    with pytest.raises(IdeasError, match=expected_error_message):
        sync_csv_to_annotations(
            annotations_files=annotations_files,
            isxd_files=cell_set_files,
            time_column=time_column,
            state_column=state_column,
            gpio_ref_file=gpio_ref_file,
            gpio_ref_channel=gpio_ref_channel,
            gpio_ref_threshold=gpio_ref_threshold,
        )


@pytest.mark.parametrize(
    (
        "annotations_files",
        "cell_set_files",
        "time_column",
        "state_column",
        "manual_time_offset",
        "expected_results",
    ),
    [
        # positive time offset
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            1.2345,
            [
                # first row with mapped frame
                {
                    "row": 25,
                    "frame": 0,
                    "time": 1.2344999313354492,
                    "state": "",
                },
                # first row with mapped state
                {
                    "row": 34,
                    "frame": 14,
                    "time": 1.7024979313354494,
                    "state": "entry",
                },
                # last row with mapped frame
                {
                    "row": 240,
                    "frame": 323,
                    "time": 12.00649693133545,
                    "state": "",
                },
            ],
        ),
        # negative time offset
        pytest.param(
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-test-1.csv"
            ],
            [
                "/ideas/data/Group-20240708-135659_2024-07-11-11-56-35_video_green-PP-ROI.isxd"
            ],
            "Frame Timestamp (s)",
            "Zone Event",
            -0.581,
            [
                # first row with mapped frame and state
                {
                    "row": 0,
                    "frame": 17,
                    "time": -0.012953089645385774,
                    "state": "exit",
                },
                # last row with mapped frame
                {
                    "row": 210,
                    "frame": 332,
                    "time": 10.490995910354615,
                    "state": "",
                },
            ],
        ),
    ],
)
def test_converter_with_manual_time_offset(
    annotations_files,
    cell_set_files,
    time_column,
    state_column,
    manual_time_offset,
    expected_results,
):
    """test converter with manual time offset"""
    sync_csv_to_annotations(
        annotations_files=annotations_files,
        isxd_files=cell_set_files,
        time_column=time_column,
        state_column=state_column,
        manual_time_offset=manual_time_offset,
    )

    output_file = "annotations.parquet"

    # check that output file is produced
    if not os.path.exists(output_file):
        raise RuntimeError("output file not generated")

    output_df = pd.read_parquet(output_file)

    # check that the dataframe has the correct columns
    _check_columns_in_df(
        df=output_df,
        columns=("time", "state", ALIGNMENT_METHOD_COLUMN),
    )
    assert set(output_df[ALIGNMENT_METHOD_COLUMN].unique()) == {"start_time"}

    # check that the time is the same in
    # the annotations and cellset file
    time = io.cell_set_to_time(cell_set_files)
    assert np.max(output_df["time"] - time) == 0, (
        "Time in annotations and cellset is not the same"
    )

    # validate rows of output
    for expected_result in expected_results:
        assert (
            output_df.iloc[expected_result["row"]]["mapped frame"]
            == expected_result["frame"]
        )
        np.testing.assert_allclose(
            output_df.iloc[expected_result["row"]]["mapped time since start (s)"],
            expected_result["time"],
        )

        if expected_result["state"] == "":
            assert pd.isnull(output_df.iloc[expected_result["row"]]["state"])
        else:
            assert (
                output_df.iloc[expected_result["row"]]["state"]
                == expected_result["state"]
            )
