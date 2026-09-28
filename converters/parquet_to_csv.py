import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger()

output_filename = "experiment_annotations.csv"


def convert(parquet_files):
    """Convert parquet file to csv format.

    :Args
        parquet_files (list): List of parquet files

    :Returns
        None
    """
    # Load parquet file
    df = pd.read_parquet(parquet_files[0])  # Load only the first parquet file

    # Export to CSV
    df.to_csv(output_filename, index=False)

    # Export metadata
    output_key = Path(output_filename).stem
    output_values = {
        "metrics": {
            "num_rows": len(df),
            "num_columns": len(df.columns),
        },
        "column_names": df.columns.to_list(),
    }
    # Add unique values for "state" column
    if "state" in df.columns:
        output_values["dataset"] = {
            "states": ", ".join(df["state"].unique().astype(str))
        }

    metadata = {output_key: output_values}

    with open("output_metadata.json", "w") as f:
        json.dump(metadata, f)

    logger.info("Successfully exported timestamps to CSV")
