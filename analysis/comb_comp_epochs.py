import json
import logging
import re

import pandas as pd
from beartype.typing import List, Optional

from utils.algorithms import (
    _compare_correlation,
    _compare_data,
    _compare_single_group,
    _unify_results,
)
from utils.plots import _plot_combined_data
from utils.utils import _combine_data
from utils.validation import _validate_files
from ideas.exceptions import IdeasError

from ideas.tools import log
from ideas.tools.types import IdeasFile
from ideas.tools import outputs

logger = log.get_logger()


def run_cc_epochs(
    group1_traces: List[str],
    epoch_names: str,
    epoch_colors: str,
    group1_name: Optional[str] = None,
    group2_traces: Optional[List[str]] = None,
    group2_name: Optional[str] = None,
    group1_events: Optional[List[str]] = None,
    group2_events: Optional[List[str]] = None,
    group1_corr: Optional[List[str]] = None,
    group2_corr: Optional[List[str]] = None,
    group1_color: Optional[str] = "blue",
    group2_color: Optional[str] = "orange",
    multiple_correction: Optional[str] = "bonf",
    effect_size: Optional[str] = "cohen",
) -> None:
    """Combine and compares epoch data between two groups.

    group1_traces (List[str]): List of file paths to trace CSV files for group 1.
        group2_traces (List[str]): List of file paths to trace CSV files for group 2.
        group1_events (List[str]): List of file paths to event CSV files for group 1.
        group2_events (List[str]): List of file paths to event CSV files for group 2.
        group1_corr (List[str]): List of file paths to correlation CSV files for group 1.
        group2_corr (List[str]): List of file paths to correlation CSV files for group 2.
        epoch_names (str): Comma-separated string of epoch names.
        group1_name (str): Name of group 1.
        group2_name (str): Name of group 2.
        epoch_colors (str): Comma-separated string of colors for each epoch.
        group1_color (Optional[str], optional): Color for group 1. Defaults to "blue".
        group2_color (Optional[str], optional): Color for group 2. Defaults to "orange".
        multiple_correction (Optional[str], optional):
        Method for multiple correction. Defaults to "bonf".
        effect_size (Optional[str], optional):
        Method for effect size calculation. Defaults to "cohen".
    :Raises
        IdeasError: If the number of trace, event, and correlation
        files are not the same for each group.
        IdeasError: If there are fewer than two files for any group.
        IdeasError: If the number of epoch names and colors are not the same.
        IdeasError: If group names contain special characters.
    :Returns
        None
    """
    # Input Validation

    # Check that the number of trace, event, and correlation files is the same for each group
    _validate_files(
        files=[group1_traces, group1_events, group1_corr],
    )
    _validate_files(
        files=[group2_traces, group2_events, group2_corr],
    )

    # Validate that each group has more than one file for ANOVA
    if len(group1_traces) < 2:
        raise IdeasError(
            "Group1 must have more than 1 trace file to run ANOVA."
        )
    if group2_traces is not None and len(group2_traces) < 2:
        raise IdeasError(
            "Group2 must have more than 1 trace file to run ANOVA."
        )

    # Validate that group names do not contain special characters
    if group1_name and re.search(r'[\/:*?"<>|]', group1_name):
        raise IdeasError(
            "Group1 name contains special characters which are not allowed."
        )
    if group2_name and re.search(r'[\/:*?"<>|]', group2_name):
        raise IdeasError(
            "Group2 name contains special characters which are not allowed."
        )

    epoch_names = epoch_names.split(",")
    epoch_names = [name.strip() for name in epoch_names]

    epoch_colors = epoch_colors.replace(" ", "").split(",")
    if len(epoch_names) != len(epoch_colors):
        raise IdeasError(
            "The number of epoch names and colors must be the same."
        )

    # initialize metadata
    metadata = {}

    # Combine Data
    g1_data, g1_corr_data, metadata = _combine_data(
        traces=group1_traces,
        events=group1_events,
        corr=group1_corr,
        group_name="group_1",
        epoch_names=epoch_names,
        metadata=metadata,
    )
    _plot_combined_data(
        data=g1_data,
        corr_data=g1_corr_data,
        group_name=group1_name,
        save_name="group_1",
        epoch_colors=epoch_colors,
    )
    aov, pairwise = _compare_single_group(
        data=g1_data,
        corr_data=g1_corr_data,
        multiple_correction=multiple_correction,
        effect_size=effect_size,
        epoch_colors=epoch_colors,
        save_name="group_1",
        group_name=group1_name,
    )
    aov["Group"] = group1_name
    pairwise["Group"] = group1_name
    if group2_traces is not None:
        g2_data, g2_corr_data, metadata = _combine_data(
            traces=group2_traces,
            events=group2_events,
            corr=group2_corr,
            group_name="group_2",
            epoch_names=epoch_names,
            metadata=metadata,
        )
        _plot_combined_data(
            data=g2_data,
            corr_data=g2_corr_data,
            group_name=group2_name,
            save_name="group_2",
            epoch_colors=epoch_colors,
        )
        aov2, pairwise2 = _compare_single_group(
            data=g2_data,
            corr_data=g2_corr_data,
            multiple_correction=multiple_correction,
            effect_size=effect_size,
            epoch_colors=epoch_colors,
            save_name="group_2",
            group_name=group2_name,
        )
        aov2["Group"] = group2_name
        pairwise2["Group"] = group2_name

        aov = pd.concat([aov, aov2])
        pairwise = pd.concat([pairwise, pairwise2])

        # compare group 1 and group 2
        mixed_aov, mixed_pairwise = _compare_data(
            g1_data=g1_data,
            g2_data=g2_data,
            g1_name=group1_name,
            g2_name=group2_name,
            g1_color=group1_color,
            g2_color=group2_color,
            multiple_correction=multiple_correction,
            effect_size=effect_size,
        )
    else:
        # create variables but set them to None
        g2_data = g2_corr_data = None

    if g1_corr_data is not None and g2_corr_data is not None:
        mixed_corr_aov, mixed_corr_pairwise = _compare_correlation(
            g1_data=g1_corr_data,
            g2_data=g2_corr_data,
            g1_name=group1_name,
            g2_name=group2_name,
            g1_color=group1_color,
            g2_color=group2_color,
            name="Correlation",
            multiple_correction=multiple_correction,
            effect_size=effect_size,
        )
        mixed_aov = pd.concat([mixed_aov, mixed_corr_aov])
        mixed_pairwise = pd.concat([mixed_pairwise, mixed_corr_pairwise])
        aov, pairwise = _unify_results(
            aov, pairwise, mixed_aov, mixed_pairwise
        )

    aov.to_csv("ANOVA_results.csv", index=False)
    pairwise.to_csv("pairwise_results.csv", index=False)

    metadata["ANOVA_results"] = {
        "num_epochs": len(epoch_names),
        "epoch_names": epoch_names,
        "group_names": [group1_name, group2_name],
    }
    metadata["pairwise_results"] = {
        "num_epochs": len(epoch_names),
        "epoch_names": epoch_names,
        "group_names": [group1_name, group2_name],
        "correction_method": multiple_correction,
        "effect_size": effect_size,
        "alternative": "two-sided",
    }

    # Create metadata
    with open("output_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

def run_cc_epochs_ideas_wrapper(
    group1_traces: List[IdeasFile],
    epoch_names: str,
    epoch_colors: str,
    group1_name: Optional[str] = None,
    group2_traces: Optional[List[IdeasFile]] = None,
    group2_name: Optional[str] = None,
    group1_events: Optional[List[IdeasFile]] = None,
    group2_events: Optional[List[IdeasFile]] = None,
    group1_corr: Optional[List[IdeasFile]] = None,
    group2_corr: Optional[List[IdeasFile]] = None,
    group1_color: Optional[str] = "blue",
    group2_color: Optional[str] = "orange",
    multiple_correction: Optional[str] = "bonf",
    effect_size: Optional[str] = "cohen",
) -> None:
    """Combine and compares epoch data between two groups.

    group1_traces (List[str]): List of file paths to trace CSV files for group 1.
        group2_traces (List[str]): List of file paths to trace CSV files for group 2.
        group1_events (List[str]): List of file paths to event CSV files for group 1.
        group2_events (List[str]): List of file paths to event CSV files for group 2.
        group1_corr (List[str]): List of file paths to correlation CSV files for group 1.
        group2_corr (List[str]): List of file paths to correlation CSV files for group 2.
        epoch_names (str): Comma-separated string of epoch names.
        group1_name (str): Name of group 1.
        group2_name (str): Name of group 2.
        epoch_colors (str): Comma-separated string of colors for each epoch.
        group1_color (Optional[str], optional): Color for group 1. Defaults to "blue".
        group2_color (Optional[str], optional): Color for group 2. Defaults to "orange".
        multiple_correction (Optional[str], optional):
        Method for multiple correction. Defaults to "bonf".
        effect_size (Optional[str], optional):
        Method for effect size calculation. Defaults to "cohen".
    :Raises
        IdeasError: If the number of trace, event, and correlation
        files are not the same for each group.
        IdeasError: If there are fewer than two files for any group.
        IdeasError: If the number of epoch names and colors are not the same.
        IdeasError: If group names contain special characters.
    :Returns
        None
    """
    run_cc_epochs(
        group1_traces=group1_traces,
        epoch_names=epoch_names,
        epoch_colors=epoch_colors,
        group1_name=group1_name,
        group2_traces=group2_traces,
        group2_name=group2_name,
        group1_events=group1_events,
        group2_events=group2_events,
        group1_corr=group1_corr,
        group2_corr=group2_corr,
        group1_color=group1_color,
        group2_color=group2_color,
        multiple_correction=multiple_correction,
        effect_size=effect_size,
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        with outputs.register(raise_missing_file=False) as output_data:
            group_names = [("group_1", group1_name), ("group_2", group2_name)]
            metrics = ["Trace", "Eventrate"]

            for group_id, group_name in group_names:
                output_file = output_data.register_file(
                    f"{group_id}_combined_data.csv",
                    subdir=f"{group_id}_combined_data",
                    prefix=f"{group_name}_",
                    name="combined_trace_event_data.csv"
                ).register_metadata_dict(
                    **metadata[f"{group_id}_combined_data"]
                )

                for metric in metrics:
                    output_file.register_preview(
                        f"{group_id}_{metric}_preview.svg",
                        caption=f"Box plots comparing {metric.lower()} activity across epochs in {group_name}, and histograms comparing pairwise differences in {metric.lower()} activity between each epoch combination"
                    )
            
                output_file = output_data.register_file(
                    f"{group_id}_correlation_data.csv",
                    subdir=f"{group_id}_correlation_data",
                    prefix=f"{group_name}_",
                    name="combined_correlation_data.csv"
                ).register_preview(
                    f"{group_id}_correlation_preview.svg",
                    caption=f"Box plots comparing average positive (top) and negative (bottom) correlations between epochs in {group_name}",
                ).register_metadata_dict(
                    **metadata[f"{group_id}_correlation_data"]
                )

            anova_file = output_data.register_file(
                "ANOVA_results.csv",
                prefix=f"{group1_name}_{group2_name}_",
                name="anova_results.csv"
            ).register_preview(
                "mixed_correlation_ANOVA_comparison.svg",
                caption="Box plots comparing average positive (top) and negative (bottom) correlations between each group and epoch",
            ).register_metadata_dict(
                **metadata["ANOVA_results"]
            )

            for metric in metrics:
                anova_file.register_preview(
                    f"mixed_ANOVA_{metric}_comparison.svg",
                    caption=f"Box plots comparing {metric} activity between each epoch and group",
                )

            pairwise_file = output_data.register_file(
                "pairwise_results.csv",
                prefix=f"{group1_name}_{group2_name}_",
            ).register_preview(
                "mixed_correlation_pairwise_comparison.svg",
                caption="Box plots comparing average positive (top) and negative (bottom) correlations between each group and epoch"
            ).register_metadata_dict(
                **metadata["pairwise_results"]
            )

            for metric in metrics:
                pairwise_file.register_preview(
                    f"mixed_pairwise_{metric}_comparison.svg",
                    caption=f"Box plots comparing {metric} activity between each epoch and group",
                )
            
            for output_file in [anova_file, pairwise_file]:
                for group_id, group_name in group_names:
                    for metric in metrics:
                        output_file.register_preview(
                            f"{group_id}_{metric}_Activity_comparison.svg",
                            caption=f"Box plots comparing {metric.lower()} activity between epochs in group 1"
                        )
                    output_file.register_preview(
                        f"{group_id}_correlation_data_comparison.svg",
                        caption=f"Box plots comparing average positive (top) and negative (bottom) correlations between epochs in {group_name}",
                    )

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")
