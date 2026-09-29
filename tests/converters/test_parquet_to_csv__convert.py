import json
import os

import pandas as pd

from toolbox.tools.parquet_to_csv import convert

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

    with open(output_metadata_source, "r") as f:
        metadata_source = json.load(f)
    assert (
        metadata["experiment_annotations"]
        == metadata_source["file_metadata"][0]["add"]["ideas"]
    )
