import json
from pathlib import Path
from typing import List

import pandas as pd
from ideas.tools.types import IdeasFile
from ideas.tools import log
from ideas.tools import outputs

logger = log.get_logger()

output_filename = "experiment_annotations.csv"


def convert(parquet_files: List[IdeasFile]):
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

def convert_ideas_wrapper(parquet_files: List[IdeasFile]):
    """IDEAS wrapper for tool to convert parquet file to csv format."""
    
    convert(
        parquet_files=parquet_files
    )

    output_prefix = outputs.input_paths_to_output_prefix(
        parquet_files
    )
    metadata = outputs._load_and_remove_output_metadata()
    metadata = metadata["experiment_annotations"]
    with outputs.register(raise_missing_file=False) as output_data:
        output_file = output_data.register_file(
            "experiment_annotations.csv",
            prefix=output_prefix,
            subdir="experiment_annotations"
        )
        
        if output_file:
            output_file.register_metadata(
                key="ideas.metrics.num_rows",
                name="Number of rows",
                value=metadata["metrics"]["num_rows"]
            ).register_metadata(
                key="ideas.metrics.num_columns",
                name="Number of columns",
                value=metadata["metrics"]["num_columns"]
            ).register_metadata(
                key="ideas.column_names",
                name="Column Names",
                value=metadata["column_names"]
            )

            if "dataset" in metadata:
                output_file.register_metadata(
                    key="ideas.dataset.states",
                    name="States",
                    value=metadata["dataset"]["states"]
                )
