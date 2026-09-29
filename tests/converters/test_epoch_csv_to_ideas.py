import os

import pandas as pd
from pandas.testing import assert_series_equal

import pytest
from ideas.exceptions import IdeasError

from converters.sync_annotations import (
    PARQUET_FILENAME,
    sync_boris_to_annotations,
)
from ideas.analysis.validation import _check_columns_in_df

cell_set_file = ["/ideas/data/input_cellset.isxd"]
cell_set_series_file = [
    "/ideas/data/cellset_series_1.isxd",
    "/ideas/data/cellset_series_2.isxd",
]
tsv_annotations_file = "/ideas/data/boris_annotations.tsv"
csv_annotations_file = "/ideas/data/boris_annotations.csv"
state_column_name = "Behavior"
start_column_name = "Start (s)"
stop_column_name = "Stop (s)"
ignore_behaviors = ("immobile", "rearing")


test_items = [
    # valid inputs
    (
        cell_set_file,
        tsv_annotations_file,
        state_column_name,
        start_column_name,
        stop_column_name,
        ignore_behaviors,
        [
            (0, [0, 0.0, "not_defined"]),
            (2, [2, 0.199844, "Fourth Quadrant"]),
            (11, [11, 1.099142, "Fourth Quadrant"]),
            (51, [51, 5.096022, "First Quadrant"]),
            (7455, [7455, 744.91851, "not_defined"]),
        ],
        None,
    ),
    # valid inputs, no behavior ignored
    (
        cell_set_file,
        tsv_annotations_file,
        state_column_name,
        start_column_name,
        stop_column_name,
        (),
        [
            (0, [0, 0.0, "not_defined"]),
            (2, [2, 0.199844, "Fourth Quadrant"]),
            (11, [11, 1.099142, "immobile"]),
            (51, [51, 5.096022, "First Quadrant"]),
            (7455, [7455, 744.91851, "not_defined"]),
        ],
        None,
    ),
    # valid inputs, csv
    (
        cell_set_file,
        csv_annotations_file,
        state_column_name,
        start_column_name,
        stop_column_name,
        ignore_behaviors,
        [
            (0, [0, 0.0, "not_defined"]),
            (2, [2, 0.199844, "Fourth Quadrant"]),
            (11, [11, 1.099142, "Fourth Quadrant"]),
            (51, [51, 5.096022, "First Quadrant"]),
            (7455, [7455, 744.91851, "not_defined"]),
        ],
        None,
    ),
    # valid inputs, cell set series (first file only)
    (
        [cell_set_series_file[0]],
        tsv_annotations_file,
        state_column_name,
        start_column_name,
        stop_column_name,
        ignore_behaviors,
        [
            (0, [0, 0.0, "not_defined"]),
            (2, [2, 0.199844, "Fourth Quadrant"]),
            (35, [35, 3.49727, "center"]),
            (293, [293, 29.277146, "not_defined"]),
        ],
        None,
    ),
    # valid inputs, cell set series (all files)
    (
        cell_set_series_file,
        tsv_annotations_file,
        state_column_name,
        start_column_name,
        stop_column_name,
        ignore_behaviors,
        [
            (0, [0, 0.0, "not_defined"]),
            (2, [2, 0.199844, "Fourth Quadrant"]),
            (35, [35, 3.49727, "center"]),
            (293, [293, 29.277146, "not_defined"]),
            (587, [587, 58.654214, "not_defined"]),
        ],
        None,
    ),
    # bad columns inputs
    (
        cell_set_file,
        tsv_annotations_file,
        "dsfdfs",
        start_column_name,
        stop_column_name,
        ignore_behaviors,
        None,
        IdeasError,
    ),
    # bad columns inputs
    (
        cell_set_file,
        tsv_annotations_file,
        "dsfdfs",
        "dfdf",
        stop_column_name,
        ignore_behaviors,
        None,
        IdeasError,
    ),
    # bad columns inputs
    (
        cell_set_file,
        tsv_annotations_file,
        state_column_name,
        start_column_name,
        "dfsd",
        ignore_behaviors,
        None,
        IdeasError,
    ),
]


@pytest.mark.parametrize(
    "cell_set_file, annotations_file,state_column_name,"
    " start_column_name,stop_column_name,ignore_behaviors,expected_rows,error",
    test_items,
)
def test_converter(
    cell_set_file,
    annotations_file,
    state_column_name,
    start_column_name,
    stop_column_name,
    ignore_behaviors,
    expected_rows,
    error,
):
    """Run the converter and check that it runs without error."""

    if os.path.exists(PARQUET_FILENAME):
        os.remove(PARQUET_FILENAME)

    if error is None:
        sync_boris_to_annotations(
            cell_set_file=cell_set_file,
            annotations_file=annotations_file,
            state_column_name=state_column_name,
            start_column_name=start_column_name,
            stop_column_name=stop_column_name,
            ignore_behaviors=ignore_behaviors,
        )

        # check that output file is created
        if not os.path.exists(PARQUET_FILENAME):
            raise RuntimeError("output file not generated")

        df = pd.read_parquet(PARQUET_FILENAME)

        # check that the dataframe has the correct columns
        _check_columns_in_df(
            df=df,
            columns=("time", "state"),
        )

        if expected_rows:
            # validate selected rows with expected data
            for idx, expected_row in expected_rows:
                assert_series_equal(
                    df.iloc[idx],
                    pd.Series(
                        expected_row,
                        index=["frame", "time", "state"],
                        name=idx,
                    ),
                )

        if len(ignore_behaviors) == 0:
            # nothing ignored
            assert (
                "immobile" in df["state"].unique()
            ), "Expected to see immobile in the parquet table, because no states are ignored"
        else:
            # make sure we don't see ignored behaviors here
            for thing in ignore_behaviors:
                assert (
                    thing not in df["state"]
                ), f"Expected not to see {thing} in the list of states"

    else:
        with pytest.raises(error):
            sync_boris_to_annotations(
                cell_set_file=cell_set_file,
                annotations_file=annotations_file,
                state_column_name=state_column_name,
                start_column_name=start_column_name,
                stop_column_name=stop_column_name,
                ignore_behaviors=ignore_behaviors,
            )
    # clean up
    if os.path.exists(PARQUET_FILENAME):
        os.remove(PARQUET_FILENAME)
