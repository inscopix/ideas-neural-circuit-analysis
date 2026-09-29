import json
import os

import pandas as pd

from converters.parquet_to_csv import convert

output_filename = "experiment_annotations.csv"
output_metadata = "output_metadata.json"
output_metadata_source = "/ideas/data/pqt_output_metadata.json"


parquet_files = ["/ideas/data/ideas_experiment_annotations.parquet"]


def test_parquet_to_csv(
    parquet_files=parquet_files,
):
    """Tests the basic converter."""

    convert(parquet_files)

    # check that output file is produced
    if not os.path.exists(output_filename):
        raise RuntimeError("output file not generated")

    df = pd.read_csv(output_filename)

    # check that the dataframe has the correct columns
    assert df.columns.to_list() == [
        "time",
        "state",
    ], "columns not equal"

    assert len(df) == 7456, "number of rows not equal"

    # check that metadata is produced
    if not os.path.exists(output_metadata):
        raise RuntimeError("metadata not generated")

    with open(output_metadata, "r") as f:
        metadata = json.load(f)

    assert (
        metadata["experiment_annotations"]
        == {
            "metrics": {
                "num_rows": 7456,
                "num_columns": 2
            },
            "column_names": [
                "time",
                "state"
            ],
            "dataset": {
                "states": "not_defined, quad 4, center, quad 1, familiar object, novel object, quad 2, quad 3"
            }
        }
    )
