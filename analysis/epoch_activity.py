"""Epoch activity tool (epoch-only analysis).

This module compares neural activity across user-defined *time epochs* using one or more
cell set files, and (optionally) event set files.

This implementation predates the newer state/epoch baseline pipeline in
`analysis/state_epoch_baseline_analysis.py`. It is being refactored to follow the same
high-level structure (parse/validate → load data → analyze → generate outputs → write
metadata) while keeping the existing public entrypoints (`run` and
`epoch_activity_ideas_wrapper`) stable.
"""

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from beartype import beartype
from beartype.typing import List, Optional, Union
from ideas.exceptions import IdeasError
from ideas.tools import log, outputs
from ideas.tools.types import IdeasFile

from utils.state_epoch_data import StateEpochDataManager, scale_data
from utils.state_epoch_output import (
    ACTIVITY_PER_STATE_EPOCH_DATA_CSV,
    AVERAGE_CORRELATIONS_CSV,
    AVERAGE_CORRELATIONS_PREVIEW,
    CORRELATION_MATRICES_PREVIEW,
    CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
    CORRELATIONS_PER_STATE_EPOCH_DATA_CSV,
    EVENT_AVERAGE_CORRELATIONS_PREVIEW,
    EVENT_CORRELATION_MATRICES_PREVIEW,
    EVENT_CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
    EVENT_MODULATION_HISTOGRAM_PREVIEW,
    EVENT_MODULATION_PREVIEW,
    EVENT_POPULATION_AVERAGE_PREVIEW,
    EVENT_SPATIAL_CORRELATION_MAP_PREVIEW,
    EVENT_SPATIAL_CORRELATION_PREVIEW,
    EVENT_STATE_OVERLAY,
    MODULATION_VS_BASELINE_DATA_CSV,
    RAW_CORRELATIONS_H5_NAME,
    RAW_CORRELATIONS_ZIP_NAME,
    SPATIAL_CORRELATION_MAP_PREVIEW,
    SPATIAL_CORRELATION_PREVIEW,
    STATE_EPOCH_TIME_PREVIEW,
    TRACE_MODULATION_FOOTPRINT_PREVIEW,
    TRACE_MODULATION_HISTOGRAM_PREVIEW,
    TRACE_POPULATION_AVERAGE_PREVIEW,
    TRACE_STATE_OVERLAY,
    StateEpochOutputGenerator,
)
from utils.state_epoch_results import (
    StateEpochResults,
    analyze_state_epoch_combination,
    calculate_baseline_modulation,
)
from utils.utils import (
    _bin_data,
)
from utils.validation import _validate_epoch_name_strings

logger = log.get_logger()

# Epoch-only output filenames (no "state" wording)
ACTIVITY_PER_EPOCH_DATA_CSV = "activity_per_epoch_data.csv"
CORRELATIONS_PER_EPOCH_DATA_CSV = "correlations_per_epoch_data.csv"
TIME_IN_EPOCH_PREVIEW = "time_in_epoch_preview.svg"
TRACE_EPOCH_OVERLAY = "trace_epoch_overlay.svg"
EVENT_EPOCH_OVERLAY = "event_epoch_overlay.svg"

_EPOCH_FILENAME_OVERRIDES = {
    ACTIVITY_PER_STATE_EPOCH_DATA_CSV: ACTIVITY_PER_EPOCH_DATA_CSV,
    CORRELATIONS_PER_STATE_EPOCH_DATA_CSV: CORRELATIONS_PER_EPOCH_DATA_CSV,
    STATE_EPOCH_TIME_PREVIEW: TIME_IN_EPOCH_PREVIEW,
    TRACE_STATE_OVERLAY: TRACE_EPOCH_OVERLAY,
    EVENT_STATE_OVERLAY: EVENT_EPOCH_OVERLAY,
}

_EPOCH_ONLY_STATE_NAME = "epoch_activity"
_BRAIN_REGION_COLUMN = "brain_region"
_DEFAULT_FIRST_BRAIN_REGION_NAME = "Brain Region 1"
_DEFAULT_SECOND_BRAIN_REGION_NAME = "Brain Region 2"
_MERGED_REGION_CSV_OUTPUTS = [
    ACTIVITY_PER_EPOCH_DATA_CSV,
    CORRELATIONS_PER_EPOCH_DATA_CSV,
    MODULATION_VS_BASELINE_DATA_CSV,
    AVERAGE_CORRELATIONS_CSV,
]

_USEFUL_OUTPUT_METADATA_KEYS = {
    "num_cells",
    "num_epochs",
    "baseline_epoch",
    "epoch_comparison_method",
    "modulation_colormap",
    "alpha",
    "n_shuffle",
    "correlation_method",
    "correlation_statistic",
    "modulation_method",
}


def _extract_useful_metadata(metadata: dict) -> dict:
    """Return only the metadata fields we want to register with IDEAS."""
    if not metadata:
        return {}
    return {
        key: value
        for key, value in metadata.items()
        if key in _USEFUL_OUTPUT_METADATA_KEYS and value not in (None, "", [], {})
    }


def _parse_csv_list(value: str) -> List[str]:
    """Parse a comma-separated string into a stripped list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split(",")]


def _normalize_region_selection(region_selection: str) -> str:
    """Normalize region selection aliases to canonical values."""
    normalized = str(region_selection or "").strip().lower()
    if normalized in {"", "single_brain_region", "single", "single_region"}:
        return "single_brain_region"
    if normalized in {
        "multiple_regions",
        "multiple_region",
        "multi_region",
        "multi_regions",
    }:
        return "multiple_regions"
    raise IdeasError(
        "region_selection must be 'single_brain_region' or 'multiple_regions'. "
        f"Got region_selection='{region_selection}'."
    )


def _append_brain_region_column_to_csv(csv_path: Path, brain_region_name: str) -> None:
    """Append (or overwrite) a brain_region column in a CSV file."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if _BRAIN_REGION_COLUMN in df.columns:
        df[_BRAIN_REGION_COLUMN] = brain_region_name
    else:
        df.insert(0, _BRAIN_REGION_COLUMN, brain_region_name)
    df.to_csv(csv_path, index=False)


def _append_brain_region_column_to_epoch_csv_outputs(
    output_dir: Path, brain_region_name: str
) -> None:
    """Add brain_region to core CSV outputs without altering output file layout."""
    for csv_name in _MERGED_REGION_CSV_OUTPUTS:
        _append_brain_region_column_to_csv(output_dir / csv_name, brain_region_name)


def _registered_output_path(
    output_data: dict, output_dir: Path, file_name: str
) -> Optional[Path]:
    """Resolve an output file path from output_data registration by suffix match."""
    for entry in output_data.get("output_files", []):
        file_path = Path(str(entry.get("file", "")))
        if file_path.name.endswith(file_name):
            return output_dir / file_path
    return None


def _merge_multi_region_csv_outputs(
    primary_output_dir: Path, secondary_output_dir: Path
) -> None:
    """Merge row-based CSV outputs from both regions into primary output paths."""
    primary_output_data_path = primary_output_dir / "output_data.json"
    secondary_output_data_path = secondary_output_dir / "output_data.json"
    if not primary_output_data_path.exists() or not secondary_output_data_path.exists():
        return

    primary_output_data = json.loads(primary_output_data_path.read_text())
    secondary_output_data = json.loads(secondary_output_data_path.read_text())

    for csv_name in _MERGED_REGION_CSV_OUTPUTS:
        primary_csv = _registered_output_path(primary_output_data, primary_output_dir, csv_name)
        secondary_csv = _registered_output_path(
            secondary_output_data, secondary_output_dir, csv_name
        )
        if (
            primary_csv is None
            or secondary_csv is None
            or not primary_csv.exists()
            or not secondary_csv.exists()
        ):
            continue
        primary_df = pd.read_csv(primary_csv)
        secondary_df = pd.read_csv(secondary_csv)
        pd.concat([primary_df, secondary_df], ignore_index=True).to_csv(
            primary_csv, index=False
        )


def _merge_multi_region_binary_outputs(
    primary_output_dir: Path,
    secondary_output_dir: Path,
    primary_region_file_tag: str,
    secondary_region_file_tag: str,
) -> None:
    """Merge H5/ZIP raw correlation outputs so both regions are retained."""
    primary_output_data_path = primary_output_dir / "output_data.json"
    secondary_output_data_path = secondary_output_dir / "output_data.json"
    if not primary_output_data_path.exists() or not secondary_output_data_path.exists():
        return

    primary_output_data = json.loads(primary_output_data_path.read_text())
    secondary_output_data = json.loads(secondary_output_data_path.read_text())

    primary_h5 = _registered_output_path(
        primary_output_data, primary_output_dir, RAW_CORRELATIONS_H5_NAME
    )
    secondary_h5 = _registered_output_path(
        secondary_output_data, secondary_output_dir, RAW_CORRELATIONS_H5_NAME
    )
    if (
        primary_h5 is not None
        and secondary_h5 is not None
        and primary_h5.exists()
        and secondary_h5.exists()
    ):
        with h5py.File(primary_h5, "a") as primary_h5_file, h5py.File(
            secondary_h5, "r"
        ) as secondary_h5_file:
            existing_keys = list(primary_h5_file.keys())
            for key in existing_keys:
                if key.startswith(f"{primary_region_file_tag}_"):
                    continue
                destination_key = f"{primary_region_file_tag}_{key}"
                duplicate_idx = 2
                while destination_key in primary_h5_file:
                    destination_key = f"{primary_region_file_tag}_{key}_{duplicate_idx}"
                    duplicate_idx += 1
                primary_h5_file.copy(key, primary_h5_file, name=destination_key)
                del primary_h5_file[key]

            for key in secondary_h5_file.keys():
                destination_key = f"{secondary_region_file_tag}_{key}"
                duplicate_idx = 2
                while destination_key in primary_h5_file:
                    destination_key = (
                        f"{secondary_region_file_tag}_{key}_{duplicate_idx}"
                    )
                    duplicate_idx += 1
                secondary_h5_file.copy(key, primary_h5_file, name=destination_key)

    primary_zip = _registered_output_path(
        primary_output_data, primary_output_dir, RAW_CORRELATIONS_ZIP_NAME
    )
    secondary_zip = _registered_output_path(
        secondary_output_data, secondary_output_dir, RAW_CORRELATIONS_ZIP_NAME
    )
    if (
        primary_zip is not None
        and secondary_zip is not None
        and primary_zip.exists()
        and secondary_zip.exists()
    ):
        with zipfile.ZipFile(primary_zip, "r") as original_zip:
            original_members = [m for m in original_zip.infolist() if not m.is_dir()]
            should_prefix_primary = any(
                not m.filename.startswith(f"{primary_region_file_tag}_")
                for m in original_members
            )
            if should_prefix_primary:
                rewritten_entries = [
                    (m.filename, original_zip.read(m.filename)) for m in original_members
                ]
                rewritten_path = primary_zip.with_suffix(".tmp.zip")
                with zipfile.ZipFile(
                    rewritten_path, "w", zipfile.ZIP_DEFLATED
                ) as rewritten_zip:
                    used_names = set()
                    for name, data in rewritten_entries:
                        new_name = f"{primary_region_file_tag}_{name}"
                        duplicate_idx = 2
                        while new_name in used_names:
                            new_name = (
                                f"{primary_region_file_tag}_{duplicate_idx}_{name}"
                            )
                            duplicate_idx += 1
                        rewritten_zip.writestr(new_name, data)
                        used_names.add(new_name)
                rewritten_path.replace(primary_zip)

        with zipfile.ZipFile(primary_zip, "a", zipfile.ZIP_DEFLATED) as primary_zip_file:
            existing_members = set(primary_zip_file.namelist())
            with zipfile.ZipFile(secondary_zip, "r") as secondary_zip_file:
                for member in secondary_zip_file.infolist():
                    if member.is_dir():
                        continue
                    destination_name = f"{secondary_region_file_tag}_{member.filename}"
                    duplicate_idx = 2
                    while destination_name in existing_members:
                        destination_name = (
                            f"{secondary_region_file_tag}_{duplicate_idx}_{member.filename}"
                        )
                        duplicate_idx += 1
                    with secondary_zip_file.open(member, "r") as src:
                        primary_zip_file.writestr(destination_name, src.read())
                    existing_members.add(destination_name)


def _region_labeled_caption(caption: str, region_label: str) -> str:
    caption_text = str(caption or "").strip()
    if caption_text:
        return f"{region_label}: {caption_text}"
    return region_label


def _region_file_tag(region_label: str, fallback_idx: int) -> str:
    """Create a stable filename-safe tag from a region label."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(region_label or "").strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        normalized = f"region{fallback_idx}"
    return normalized


def _normalize_brain_region_name(region_label: Optional[str]) -> Optional[str]:
    """Return a trimmed region label or None when not meaningfully provided."""
    if region_label is None:
        return None
    normalized = str(region_label).strip()
    return normalized or None


def _merge_multi_region_preview_registration(
    primary_output_dir: Path,
    secondary_output_dir: Path,
    primary_region_label: Optional[str],
    primary_region_file_tag: str,
    secondary_region_label: str,
    secondary_region_file_tag: str,
) -> None:
    """Register secondary-region previews under the same output entries."""
    primary_output_data_path = primary_output_dir / "output_data.json"
    secondary_output_data_path = secondary_output_dir / "output_data.json"
    if not primary_output_data_path.exists() or not secondary_output_data_path.exists():
        return

    primary_output_data = json.loads(primary_output_data_path.read_text())
    secondary_output_data = json.loads(secondary_output_data_path.read_text())

    primary_entries_by_name = {}
    for entry in primary_output_data.get("output_files", []):
        output_name = Path(str(entry.get("file", ""))).name
        if output_name:
            primary_entries_by_name[output_name] = entry
        if primary_region_label:
            for preview in entry.get("previews", []):
                preview_rel_path = Path(str(preview.get("file", "")))
                preview_src = primary_output_dir / preview_rel_path
                destination_name = (
                    f"{primary_region_file_tag}_{preview_rel_path.name}"
                    if not preview_rel_path.name.startswith(
                        f"{primary_region_file_tag}_"
                    )
                    else preview_rel_path.name
                )
                destination_rel_path = preview_rel_path.parent / destination_name
                destination_path = primary_output_dir / destination_rel_path
                if preview_src.exists() and preview_src != destination_path:
                    if not destination_path.exists():
                        preview_src.replace(destination_path)
                    preview["file"] = str(destination_rel_path)
                preview["caption"] = _region_labeled_caption(
                    preview.get("caption", ""), primary_region_label
                )

    for secondary_entry in secondary_output_data.get("output_files", []):
        output_name = Path(str(secondary_entry.get("file", ""))).name
        primary_entry = primary_entries_by_name.get(output_name)
        if primary_entry is None:
            continue

        for preview in secondary_entry.get("previews", []):
            preview_rel_path = Path(str(preview.get("file", "")))
            preview_source_path = secondary_output_dir / preview_rel_path
            if not preview_source_path.exists():
                continue

            destination_dir = primary_output_dir / preview_rel_path.parent
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_name = f"{secondary_region_file_tag}_{preview_rel_path.name}"
            destination_rel_path = preview_rel_path.parent / destination_name
            destination_path = primary_output_dir / destination_rel_path

            duplicate_idx = 2
            while destination_path.exists():
                destination_name = (
                    f"{secondary_region_file_tag}_{duplicate_idx}_{preview_rel_path.name}"
                )
                destination_rel_path = preview_rel_path.parent / destination_name
                destination_path = primary_output_dir / destination_rel_path
                duplicate_idx += 1

            shutil.copy2(preview_source_path, destination_path)
            primary_entry.setdefault("previews", []).append(
                {
                    **preview,
                    "file": str(destination_rel_path),
                    "caption": _region_labeled_caption(
                        preview.get("caption", ""), secondary_region_label
                    ),
                }
            )

    primary_output_data_path.write_text(json.dumps(primary_output_data, indent=4))


def _get_metadata_item(metadata: List[dict], key: str) -> Optional[dict]:
    for item in metadata:
        if item.get("key") == key:
            return item
    return None


def _upsert_metadata_item(metadata: List[dict], name: str, key: str, value) -> None:
    existing_item = _get_metadata_item(metadata, key)
    if existing_item is not None:
        existing_item["name"] = name
        existing_item["value"] = value
        return
    metadata.append({"name": name, "key": key, "value": value})


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_region_count_value(value: Optional[str]) -> dict:
    counts = {}
    if value is None:
        return counts
    for part in str(value).split(","):
        token = part.strip()
        if ":" not in token:
            continue
        name, count = token.split(":", 1)
        parsed_count = _to_int(count.strip())
        if parsed_count is not None:
            counts[name.strip()] = parsed_count
    return counts


def _merge_multi_region_metadata_values(
    primary_output_dir: Path,
    secondary_output_dir: Path,
    primary_region_label: str,
    secondary_region_label: str,
) -> None:
    """Merge metadata values so merged outputs reflect all regions, not just primary."""
    primary_output_data_path = primary_output_dir / "output_data.json"
    secondary_output_data_path = secondary_output_dir / "output_data.json"
    if not primary_output_data_path.exists() or not secondary_output_data_path.exists():
        return

    primary_output_data = json.loads(primary_output_data_path.read_text())
    secondary_output_data = json.loads(secondary_output_data_path.read_text())

    primary_by_name = {}
    for entry in primary_output_data.get("output_files", []):
        output_name = Path(str(entry.get("file", ""))).name
        if output_name:
            primary_by_name[output_name] = entry

    for secondary_entry in secondary_output_data.get("output_files", []):
        output_name = Path(str(secondary_entry.get("file", ""))).name
        primary_entry = primary_by_name.get(output_name)
        if primary_entry is None:
            continue

        primary_metadata = primary_entry.setdefault("metadata", [])
        secondary_metadata = secondary_entry.get("metadata", [])

        primary_num_cells_item = _get_metadata_item(primary_metadata, "num_cells")
        secondary_num_cells_item = _get_metadata_item(secondary_metadata, "num_cells")
        primary_num_cells = (
            _to_int(primary_num_cells_item.get("value"))
            if primary_num_cells_item is not None
            else None
        )
        secondary_num_cells = (
            _to_int(secondary_num_cells_item.get("value"))
            if secondary_num_cells_item is not None
            else None
        )

        if primary_num_cells is not None and secondary_num_cells is not None:
            _upsert_metadata_item(
                primary_metadata,
                "Num Cells",
                "num_cells",
                primary_num_cells + secondary_num_cells,
            )

        region_counts_item = _get_metadata_item(
            primary_metadata, "num_cells_by_brain_region"
        )
        region_counts = _parse_region_count_value(
            region_counts_item.get("value") if region_counts_item else None
        )
        if primary_num_cells is not None and primary_region_label not in region_counts:
            region_counts[primary_region_label] = primary_num_cells
        if secondary_num_cells is not None:
            region_counts[secondary_region_label] = secondary_num_cells
        if region_counts:
            formatted_counts = ", ".join(
                f"{region}: {count}" for region, count in region_counts.items()
            )
            _upsert_metadata_item(
                primary_metadata,
                "Num Cells by Brain Region",
                "num_cells_by_brain_region",
                formatted_counts,
            )

    primary_output_data_path.write_text(json.dumps(primary_output_data, indent=4))


def _register_multi_region_metadata(
    output_dir: Path, region_labels: List[str]
) -> None:
    """Add merged-region metadata entries to all registered outputs."""
    output_data_path = output_dir / "output_data.json"
    if not output_data_path.exists():
        return

    cleaned_labels = [str(label).strip() for label in region_labels if str(label).strip()]
    if not cleaned_labels:
        return

    output_data = json.loads(output_data_path.read_text())
    metadata_entries = [
        {
            "name": "Number of Brain Regions",
            "key": "num_brain_regions",
            "value": len(cleaned_labels),
        },
        {
            "name": "Brain Regions",
            "key": "brain_regions",
            "value": ", ".join(cleaned_labels),
        },
    ]

    for output_file in output_data.get("output_files", []):
        existing = output_file.get("metadata", [])
        existing = [
            item
            for item in existing
            if item.get("key") not in {"num_brain_regions", "brain_regions"}
        ]
        output_file["metadata"] = [*existing, *metadata_entries]

    output_data_path.write_text(json.dumps(output_data, indent=4))


@beartype
def run(
    *,
    cell_set_files: List[Union[str, Path]],
    event_set_files: Optional[List[Union[str, Path]]] = None,
    region_selection: str = "single_brain_region",
    second_cell_set_files: Optional[List[Union[str, Path]]] = None,
    second_event_set_files: Optional[List[Union[str, Path]]] = None,
    first_brain_region_name: Optional[str] = _DEFAULT_FIRST_BRAIN_REGION_NAME,
    second_brain_region_name: Optional[str] = _DEFAULT_SECOND_BRAIN_REGION_NAME,
    brain_region_name: Optional[str] = None,
    define_epochs_by: str,
    epoch_names: str,
    baseline_epoch: Optional[str] = None,
    epoch_comparison_method: str = "epoch_vs_baseline",
    epochs: Optional[str] = None,
    epoch_colors: str,
    bin_size: Optional[Union[int, float]] = None,
    trace_scale_method: Optional[str] = "none",
    event_scale_method: Optional[str] = "none",
    sort_by_time: Optional[bool] = True,
    tolerance: Optional[float] = 1e-4,
    modulation_colormap: Optional[str] = None,
    alpha: float = 0.05,
    n_shuffle: int = 1000,
    include_event_correlation_preview: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
):
    """Analyze neural activity across user-defined time epochs (epoch-only mode).

    This refactored implementation reuses the state-epoch baseline pipeline in an
    epoch-only configuration (a single dummy state), so outputs follow the same
    organization and schemas as `state_epoch_baseline_analysis`.

    Key outputs (epoch-only filenames):
    - `activity_per_epoch_data.csv`
    - `correlations_per_epoch_data.csv`
    - `modulation_vs_baseline_data.csv` (baseline epoch configurable; defaults to first epoch)
    - `average_correlations.csv`
    - `pairwise_correlation_heatmaps.h5`
    - `spatial_analysis_pairwise_correlations.zip`

    :Args
    ----
        output_dir: Directory where output files will be written. If None (default),
            writes to the current working directory.
    """
    region_mode = _normalize_region_selection(region_selection)
    if region_mode == "multiple_regions":
        if not second_cell_set_files:
            raise IdeasError(
                "second_cell_set_files is required when region_selection is "
                "'multiple_regions'."
            )

        base_output_dir = Path(output_dir) if output_dir is not None else Path(".")
        base_output_dir.mkdir(parents=True, exist_ok=True)
        first_region_name = _normalize_brain_region_name(first_brain_region_name)
        if not first_region_name:
            first_region_name = _DEFAULT_FIRST_BRAIN_REGION_NAME
        second_region_name = _normalize_brain_region_name(second_brain_region_name)
        if not second_region_name:
            second_region_name = _DEFAULT_SECOND_BRAIN_REGION_NAME
        second_region_events = (
            second_event_set_files
            if second_event_set_files is not None
            else event_set_files
        )

        shared_kwargs = dict(
            define_epochs_by=define_epochs_by,
            epoch_names=epoch_names,
            baseline_epoch=baseline_epoch,
            epoch_comparison_method=epoch_comparison_method,
            epochs=epochs,
            epoch_colors=epoch_colors,
            bin_size=bin_size,
            trace_scale_method=trace_scale_method,
            event_scale_method=event_scale_method,
            sort_by_time=sort_by_time,
            tolerance=tolerance,
            modulation_colormap=modulation_colormap,
            alpha=alpha,
            n_shuffle=n_shuffle,
            include_event_correlation_preview=include_event_correlation_preview,
            region_selection="single_brain_region",
        )
        region_runs = [
            {
                "cell_set_files": cell_set_files,
                "event_set_files": event_set_files,
                "brain_region_name": first_region_name,
            },
            {
                "cell_set_files": second_cell_set_files,
                "event_set_files": second_region_events,
                "brain_region_name": second_region_name,
            },
        ]

        primary_region = region_runs[0]
        run(
            cell_set_files=primary_region["cell_set_files"],
            event_set_files=primary_region["event_set_files"],
            output_dir=base_output_dir,
            brain_region_name=primary_region["brain_region_name"],
            **shared_kwargs,
        )

        for region_idx, region in enumerate(region_runs[1:], start=2):
            with tempfile.TemporaryDirectory() as secondary_region_tmp:
                secondary_output_dir = Path(secondary_region_tmp)
                run(
                    cell_set_files=region["cell_set_files"],
                    event_set_files=region["event_set_files"],
                    output_dir=secondary_output_dir,
                    brain_region_name=region["brain_region_name"],
                    **shared_kwargs,
                )
                region_file_tag = _region_file_tag(
                    region["brain_region_name"], fallback_idx=region_idx
                )
                _merge_multi_region_csv_outputs(base_output_dir, secondary_output_dir)
                _merge_multi_region_binary_outputs(
                    base_output_dir,
                    secondary_output_dir,
                    primary_region_file_tag=_region_file_tag(
                        primary_region["brain_region_name"], fallback_idx=1
                    ),
                    secondary_region_file_tag=region_file_tag,
                )
                _merge_multi_region_metadata_values(
                    base_output_dir,
                    secondary_output_dir,
                    primary_region_label=primary_region["brain_region_name"],
                    secondary_region_label=region["brain_region_name"],
                )
                _merge_multi_region_preview_registration(
                    primary_output_dir=base_output_dir,
                    secondary_output_dir=secondary_output_dir,
                    primary_region_label=(
                        primary_region["brain_region_name"] if region_idx == 2 else None
                    ),
                    primary_region_file_tag=_region_file_tag(
                        primary_region["brain_region_name"], fallback_idx=1
                    ),
                    secondary_region_label=region["brain_region_name"],
                    secondary_region_file_tag=region_file_tag,
                )
        _register_multi_region_metadata(
            base_output_dir, [region["brain_region_name"] for region in region_runs]
        )
        return

    parsed_epoch_names = _validate_epoch_name_strings(epoch_names)
    parsed_epoch_colors = _parse_csv_list(epoch_colors)

    if len(parsed_epoch_names) == 0:
        raise IdeasError("epoch_names must include at least one epoch name.")
    if len(parsed_epoch_names) != len(parsed_epoch_colors):
        raise IdeasError(
            f"Number of epoch names ({len(parsed_epoch_names)}) must match "
            f"number of epoch colors ({len(parsed_epoch_colors)})."
        )

    # Epoch-only mode uses a single dummy state.
    states = [_EPOCH_ONLY_STATE_NAME]
    state_colors = ["gray"]
    baseline_state = _EPOCH_ONLY_STATE_NAME

    # Resolve output directory
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(".")

    resolved_baseline_epoch: str
    if baseline_epoch is None:
        resolved_baseline_epoch = parsed_epoch_names[0]
    else:
        baseline_epoch_str = str(baseline_epoch).strip()
        if baseline_epoch_str == "" or baseline_epoch_str.lower() == "auto":
            resolved_baseline_epoch = parsed_epoch_names[0]
        else:
            resolved_baseline_epoch = baseline_epoch_str
    if resolved_baseline_epoch not in parsed_epoch_names:
        raise IdeasError(
            "baseline_epoch must be one of the provided epoch_names. "
            f"Got baseline_epoch='{resolved_baseline_epoch}'. "
            f"Available epochs: {parsed_epoch_names}."
        )

    comparison_method = str(epoch_comparison_method or "").strip().lower()
    if comparison_method in {"", "auto"}:
        comparison_method = "epoch_vs_baseline"
    if comparison_method in {"epoch-vs-baseline", "epoch_vs_baseline", "baseline"}:
        comparison_method = "epoch_vs_baseline"
    if comparison_method != "epoch_vs_baseline":
        raise IdeasError(
            "Only epoch_vs_baseline is currently supported for epoch_comparison_method. "
            f"Got epoch_comparison_method='{epoch_comparison_method}'."
        )

    parsed_modulation_colormap = (
        _parse_csv_list(modulation_colormap)
        if modulation_colormap not in (None, "", [], {})
        else None
    )
    if parsed_modulation_colormap is not None and len(parsed_modulation_colormap) > 0:
        if len(parsed_modulation_colormap) != 3:
            raise IdeasError(
                "modulation_colormap must be three comma-separated colors in the order "
                "(up, down, non). Example: 'red, blue, gray'."
            )
    column_name = "dummy_state"

    # Load + validate inputs using the shared state-epoch data manager.
    data_manager = StateEpochDataManager(
        cell_set_files=[str(p) for p in cell_set_files],
        event_set_files=[str(p) for p in event_set_files] if event_set_files else None,
        annotations_file=None,
        concatenate=True,
        epochs=epochs,
        epoch_names=parsed_epoch_names,
        epoch_colors=parsed_epoch_colors,
        state_names=states,
        state_colors=state_colors,
        baseline_state=baseline_state,
        baseline_epoch=resolved_baseline_epoch,
        define_epochs_by=define_epochs_by,
        tolerance=float(tolerance) if tolerance is not None else 1e-4,
        sort_by_time=bool(sort_by_time) if sort_by_time is not None else True,
        allow_epoch_only_mode=True,
    )

    traces, events, _, cell_info = data_manager.load_data()
    epoch_periods = data_manager.get_epoch_periods()
    original_period = float(cell_info.get("period", 1.0))

    # Bin data (epoch_activity behavior) and update effective sampling period.
    traces = _bin_data(traces, bin_size, original_period)
    events = (
        _bin_data(events, bin_size, original_period) if events is not None else None
    )
    functional_period = (
        float(bin_size) if (bin_size is not None and bin_size > 0) else original_period
    )

    # Create epoch-only annotations aligned to the processed (binned) data length.
    num_timepoints = traces.shape[0]
    annotations_df = pd.DataFrame(
        {
            column_name: [_EPOCH_ONLY_STATE_NAME] * num_timepoints,
            "time": np.arange(num_timepoints, dtype=float) * functional_period,
        }
    )

    # Adjust cell_info period/boundaries for downstream plotting.
    cell_info = dict(cell_info)
    cell_info["period"] = functional_period
    if "boundaries" in cell_info and bin_size is not None and bin_size > 0:
        bin_size_in_idxs = int(float(bin_size) / original_period)
        if bin_size_in_idxs > 0:
            try:
                cell_info["boundaries"] = [
                    int(b / bin_size_in_idxs) for b in cell_info.get("boundaries", [])
                ]
            except Exception:
                # keep original boundaries if any unexpected type issues occur
                pass

    # Scale data using shared helpers (epoch-based scaling supported).
    if trace_scale_method:
        traces = scale_data(
            traces,
            method=str(trace_scale_method),
            behavior=None,
            baseline_state=None,
            epochs=epoch_periods,
            period=functional_period,
            baseline_epoch=resolved_baseline_epoch,
            epoch_names=parsed_epoch_names,
        )
    if events is not None and event_scale_method:
        events = scale_data(
            events,
            method=str(event_scale_method),
            behavior=None,
            baseline_state=None,
            epochs=epoch_periods,
            period=functional_period,
            baseline_epoch=resolved_baseline_epoch,
            epoch_names=parsed_epoch_names,
        )

    # Analyze each epoch as a (state, epoch) combination with the dummy state.
    results = StateEpochResults()

    for epoch_name, (start_s, end_s) in zip(parsed_epoch_names, epoch_periods):
        start_idx = max(0, int(start_s / functional_period))
        end_idx = min(traces.shape[0], int(end_s / functional_period))
        if end_idx <= start_idx:
            combination_data = None
        else:
            epoch_annotations = annotations_df.iloc[start_idx:end_idx]
            combination_data = {
                "traces": traces[start_idx:end_idx, :],
                "events": events[start_idx:end_idx, :] if events is not None else None,
                "annotations": epoch_annotations,
                "num_timepoints": end_idx - start_idx,
                "state": _EPOCH_ONLY_STATE_NAME,
                "epoch": epoch_name,
            }

        combination_results = analyze_state_epoch_combination(
            state_epoch_data=combination_data,
            state=_EPOCH_ONLY_STATE_NAME,
            epoch=epoch_name,
            cell_info=cell_info,
            include_correlations=True,
            include_population_activity=True,
            include_event_analysis=events is not None,
            alpha=alpha,
            n_shuffle=n_shuffle,
        )
        results.add_combination_results(
            _EPOCH_ONLY_STATE_NAME, epoch_name, combination_results
        )

    # Baseline modulation (baseline epoch is the first epoch by default).
    modulation_results = calculate_baseline_modulation(
        results=results,
        baseline_state=baseline_state,
        baseline_epoch=resolved_baseline_epoch,
        cell_info=cell_info,
        alpha=alpha,
        n_shuffle=n_shuffle,
    )

    # Generate outputs using the same organization as the state-epoch baseline tool.
    normalized_brain_region_name = _normalize_brain_region_name(brain_region_name)

    output_generator = StateEpochOutputGenerator(
        output_dir=str(output_dir) if output_dir is not None else "",
        states=states,
        epochs=parsed_epoch_names,
        state_colors=state_colors,
        epoch_colors=parsed_epoch_colors,
        baseline_state=baseline_state,
        baseline_epoch=resolved_baseline_epoch,
        epoch_comparison_method=comparison_method,
        alpha=alpha,
        n_shuffle=n_shuffle,
        epoch_periods=epoch_periods,
        correlation_statistic="max",
        include_event_correlation_preview=include_event_correlation_preview,
        trace_scale_method=trace_scale_method,
        event_scale_method=event_scale_method,
        filename_overrides=_EPOCH_FILENAME_OVERRIDES,
        hide_state_prefix=True,
        epoch_only_mode=True,
        brain_region_name=normalized_brain_region_name,
    )
    output_generator.generate_all_outputs(
        results=results,
        modulation_results=modulation_results,
        cell_info=cell_info,
        traces=traces,
        events=events,
        annotations_df=annotations_df,
        column_name=column_name,
        modulation_colors=parsed_modulation_colormap,
    )
    if normalized_brain_region_name:
        _append_brain_region_column_to_epoch_csv_outputs(
            resolved_output_dir, normalized_brain_region_name
        )

    has_event_correlation_data = False
    if events is not None:
        for state, epoch in results.get_all_combinations():
            combination_results = results.get_combination_results(state, epoch)
            if (
                combination_results
                and combination_results.get("event_correlation_matrix") is not None
            ):
                has_event_correlation_data = True
                break

    logger.info("Registering output data")
    try:
        output_metadata_path = resolved_output_dir / "output_metadata.json"
        if output_metadata_path.exists():
            output_metadata = json.loads(output_metadata_path.read_text())
            output_metadata_path.unlink()
        else:
            output_metadata = outputs._load_and_remove_output_metadata()
        prefix_args = [cell_set_files]
        if event_set_files:
            prefix_args.append(event_set_files)
        output_prefix = outputs.input_paths_to_output_prefix(*prefix_args)
        has_event_data = bool(event_set_files)

        with outputs.register(
            output_dir=resolved_output_dir, raise_missing_file=False
        ) as output_data:
            activity_reg = output_data.register_file(
                resolved_output_dir / ACTIVITY_PER_EPOCH_DATA_CSV,
                subdir=Path(ACTIVITY_PER_EPOCH_DATA_CSV).stem,
                prefix=output_prefix,
            )
            if activity_reg:
                activity_reg.register_preview(
                    TIME_IN_EPOCH_PREVIEW,
                    caption="Time spent in each epoch.",
                ).register_preview(
                    TRACE_POPULATION_AVERAGE_PREVIEW,
                    caption="Average trace activity across epochs.",
                ).register_preview(
                    TRACE_EPOCH_OVERLAY,
                    caption="Trace preview with epoch overlay.",
                )
                if has_event_data:
                    activity_reg.register_preview(
                        EVENT_POPULATION_AVERAGE_PREVIEW,
                        caption="Average event rates across epochs (when event data is available).",
                    ).register_preview(
                        EVENT_EPOCH_OVERLAY,
                        caption="Event raster plot with epoch overlay (when event data is available).",
                    )
                activity_reg.register_metadata_dict(
                    **_extract_useful_metadata(
                        output_metadata.get(Path(ACTIVITY_PER_EPOCH_DATA_CSV).stem, {})
                    )
                )

            corr_reg = output_data.register_file(
                resolved_output_dir / CORRELATIONS_PER_EPOCH_DATA_CSV,
                subdir=Path(CORRELATIONS_PER_EPOCH_DATA_CSV).stem,
                prefix=output_prefix,
            )
            if corr_reg:
                corr_reg.register_preview(
                    CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
                    caption="Distribution of per-cell correlation statistic across epochs.",
                )
                if (
                    has_event_data
                    and has_event_correlation_data
                    and include_event_correlation_preview
                ):
                    corr_reg.register_preview(
                        EVENT_CORRELATION_STATISTIC_DISTRIBUTION_PREVIEW,
                        caption="Distribution of per-cell event correlation statistic across epochs (when event data is available).",
                    )
                corr_reg.register_metadata_dict(
                    **_extract_useful_metadata(
                        output_metadata.get(
                            Path(CORRELATIONS_PER_EPOCH_DATA_CSV).stem, {}
                        )
                    )
                )

            mod_reg = output_data.register_file(
                resolved_output_dir / MODULATION_VS_BASELINE_DATA_CSV,
                subdir=Path(MODULATION_VS_BASELINE_DATA_CSV).stem,
                prefix=output_prefix,
            )
            if mod_reg:
                mod_reg.register_preview(
                    TRACE_MODULATION_HISTOGRAM_PREVIEW,
                    caption="Distribution of trace modulation scores relative to baseline epoch.",
                ).register_preview(
                    TRACE_MODULATION_FOOTPRINT_PREVIEW,
                    caption="Spatial distribution of trace-modulated neurons relative to baseline epoch.",
                )
                if has_event_data:
                    mod_reg.register_preview(
                        EVENT_MODULATION_HISTOGRAM_PREVIEW,
                        caption="Distribution of event modulation scores relative to baseline epoch (when event data is available).",
                    ).register_preview(
                        EVENT_MODULATION_PREVIEW,
                        caption="Spatial footprints of event-modulated neurons relative to baseline epoch (when event data is available).",
                    )
                mod_reg.register_metadata_dict(
                    **_extract_useful_metadata(
                        output_metadata.get(
                            Path(MODULATION_VS_BASELINE_DATA_CSV).stem, {}
                        )
                    )
                )

            avg_corr_reg = output_data.register_file(
                resolved_output_dir / AVERAGE_CORRELATIONS_CSV,
                subdir=Path(AVERAGE_CORRELATIONS_CSV).stem,
                prefix=output_prefix,
            )
            if avg_corr_reg:
                avg_corr_reg.register_preview(
                    AVERAGE_CORRELATIONS_PREVIEW,
                    caption="Average positive and negative correlations per epoch.",
                )
                if (
                    has_event_data
                    and has_event_correlation_data
                    and include_event_correlation_preview
                ):
                    avg_corr_reg.register_preview(
                        EVENT_AVERAGE_CORRELATIONS_PREVIEW,
                        caption="Average positive and negative event correlations per epoch (when event data is available).",
                    )
                avg_corr_reg.register_metadata_dict(
                    **_extract_useful_metadata(
                        output_metadata.get(Path(AVERAGE_CORRELATIONS_CSV).stem, {})
                    )
                )

            raw_h5_reg = output_data.register_file(
                resolved_output_dir / RAW_CORRELATIONS_H5_NAME,
                subdir=Path(RAW_CORRELATIONS_H5_NAME).stem,
                prefix=output_prefix,
            )
            if raw_h5_reg:
                raw_h5_reg.register_preview(
                    CORRELATION_MATRICES_PREVIEW,
                    caption="Pairwise correlation matrices for each epoch (trace).",
                )
                if (
                    has_event_data
                    and has_event_correlation_data
                    and include_event_correlation_preview
                ):
                    raw_h5_reg.register_preview(
                        EVENT_CORRELATION_MATRICES_PREVIEW,
                        caption="Pairwise correlation matrices for each epoch (events, when available).",
                    )
                raw_h5_reg.register_metadata_dict(
                    **_extract_useful_metadata(
                        output_metadata.get(Path(RAW_CORRELATIONS_H5_NAME).stem, {})
                    )
                )

            raw_zip_reg = output_data.register_file(
                resolved_output_dir / RAW_CORRELATIONS_ZIP_NAME,
                subdir=Path(RAW_CORRELATIONS_ZIP_NAME).stem,
                prefix=output_prefix,
            )
            if raw_zip_reg:
                raw_zip_reg.register_preview(
                    SPATIAL_CORRELATION_PREVIEW,
                    caption="Spatial distance vs correlation relationships per epoch (trace).",
                ).register_preview(
                    SPATIAL_CORRELATION_MAP_PREVIEW,
                    caption="Spatial map of correlations per epoch (trace).",
                )
                if (
                    has_event_data
                    and has_event_correlation_data
                    and include_event_correlation_preview
                ):
                    raw_zip_reg.register_preview(
                        EVENT_SPATIAL_CORRELATION_PREVIEW,
                        caption="Spatial distance vs event-correlation relationships per epoch (when available).",
                    ).register_preview(
                        EVENT_SPATIAL_CORRELATION_MAP_PREVIEW,
                        caption="Spatial map of event correlations per epoch (when available).",
                    )
                raw_zip_reg.register_metadata_dict(
                    **_extract_useful_metadata(
                        output_metadata.get(Path(RAW_CORRELATIONS_ZIP_NAME).stem, {})
                    )
                )

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def epoch_activity_ideas_wrapper(
    *,
    cell_set_files: List[IdeasFile],
    event_set_files: Optional[List[IdeasFile]] = None,
    region_selection: str = "single_brain_region",
    second_cell_set_files: Optional[List[IdeasFile]] = None,
    second_event_set_files: Optional[List[IdeasFile]] = None,
    first_brain_region_name: Optional[str] = _DEFAULT_FIRST_BRAIN_REGION_NAME,
    second_brain_region_name: Optional[str] = _DEFAULT_SECOND_BRAIN_REGION_NAME,
    brain_region_name: Optional[str] = None,
    define_epochs_by: str,
    epoch_names: str,
    baseline_epoch: Optional[str] = None,
    epochs: Optional[str] = None,
    epoch_colors: str,
    bin_size: Optional[Union[int, float]] = None,
    trace_scale_method: Optional[str] = "none",
    event_scale_method: Optional[str] = "none",
    sort_by_time: Optional[bool] = True,
    tolerance: Optional[float] = 1e-4,
    modulation_colormap: Optional[str] = None,
    epoch_comparison_method: str = "epoch_vs_baseline",
    alpha: float = 0.05,
    n_shuffle: int = 1000,
    include_event_correlation_preview: bool = False,
):
    run(
        cell_set_files=cell_set_files,
        event_set_files=event_set_files,
        region_selection=region_selection,
        second_cell_set_files=second_cell_set_files,
        second_event_set_files=second_event_set_files,
        first_brain_region_name=first_brain_region_name,
        second_brain_region_name=second_brain_region_name,
        brain_region_name=brain_region_name,
        define_epochs_by=define_epochs_by,
        epoch_names=epoch_names,
        baseline_epoch=baseline_epoch,
        epoch_comparison_method=epoch_comparison_method,
        epochs=epochs,
        epoch_colors=epoch_colors,
        bin_size=bin_size,
        trace_scale_method=trace_scale_method,
        event_scale_method=event_scale_method,
        sort_by_time=sort_by_time,
        tolerance=tolerance,
        modulation_colormap=modulation_colormap,
        alpha=alpha,
        n_shuffle=n_shuffle,
        include_event_correlation_preview=include_event_correlation_preview,
    )
