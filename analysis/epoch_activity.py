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
import matplotlib.pyplot as plt
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
_SUBJECT_ID_COLUMN = "subject_id"
_DEFAULT_FIRST_BRAIN_REGION_NAME = "Brain Region 1"
_DEFAULT_SECOND_BRAIN_REGION_NAME = "Brain Region 2"
_MERGED_REGION_CSV_OUTPUTS = [
    ACTIVITY_PER_EPOCH_DATA_CSV,
    CORRELATIONS_PER_EPOCH_DATA_CSV,
    MODULATION_VS_BASELINE_DATA_CSV,
    AVERAGE_CORRELATIONS_CSV,
]
_TRACE_EVENT_OUTPUTS_SUBDIR = "trace_event_analysis_outputs"
_RAW_TRACE_EVENT_ACTIVITY_CSV = "raw_trace_event_data.csv"
_CELL_CORR_HEATMAP_PREVIEW = "cell_correlation_heatmap_preview.svg"

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


def _append_subject_id_column_to_csv(csv_path: Path, subject_id: str) -> None:
    """Append (or overwrite) a subject_id column in a CSV file."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if _SUBJECT_ID_COLUMN in df.columns:
        df[_SUBJECT_ID_COLUMN] = subject_id
    else:
        df.insert(0, _SUBJECT_ID_COLUMN, subject_id)
    df.to_csv(csv_path, index=False)


def _append_brain_region_column_to_epoch_csv_outputs(
    output_dir: Path, brain_region_name: str
) -> None:
    """Add brain_region to core CSV outputs without altering output file layout."""
    for csv_name in _MERGED_REGION_CSV_OUTPUTS:
        _append_brain_region_column_to_csv(output_dir / csv_name, brain_region_name)


def _append_subject_id_column_to_epoch_csv_outputs(
    output_dir: Path, subject_id: str
) -> None:
    """Add subject_id to core CSV outputs without altering output file layout."""
    for csv_name in _MERGED_REGION_CSV_OUTPUTS:
        _append_subject_id_column_to_csv(output_dir / csv_name, subject_id)


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
        primary_csv = _registered_output_path(
            primary_output_data, primary_output_dir, csv_name
        )
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
        with (
            h5py.File(primary_h5, "a") as primary_h5_file,
            h5py.File(secondary_h5, "r") as secondary_h5_file,
        ):
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
                    (m.filename, original_zip.read(m.filename))
                    for m in original_members
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

        with zipfile.ZipFile(
            primary_zip, "a", zipfile.ZIP_DEFLATED
        ) as primary_zip_file:
            existing_members = set(primary_zip_file.namelist())
            with zipfile.ZipFile(secondary_zip, "r") as secondary_zip_file:
                for member in secondary_zip_file.infolist():
                    if member.is_dir():
                        continue
                    destination_name = f"{secondary_region_file_tag}_{member.filename}"
                    duplicate_idx = 2
                    while destination_name in existing_members:
                        destination_name = f"{secondary_region_file_tag}_{duplicate_idx}_{member.filename}"
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


def _normalize_subject_id(subject_id: Optional[str]) -> Optional[str]:
    """Return a trimmed subject ID or None when not meaningfully provided."""
    if subject_id is None:
        return None
    normalized = str(subject_id).strip()
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
                destination_name = f"{secondary_region_file_tag}_{duplicate_idx}_{preview_rel_path.name}"
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


def _register_multi_region_metadata(output_dir: Path, region_labels: List[str]) -> None:
    """Add merged-region metadata entries to all registered outputs."""
    output_data_path = output_dir / "output_data.json"
    if not output_data_path.exists():
        return

    cleaned_labels = [
        str(label).strip() for label in region_labels if str(label).strip()
    ]
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


def _register_subject_metadata(output_dir: Path, subject_id: Optional[str]) -> None:
    """Add subject_id metadata entries to all registered outputs."""
    normalized_subject_id = _normalize_subject_id(subject_id)
    if normalized_subject_id is None:
        return
    output_data_path = output_dir / "output_data.json"
    if not output_data_path.exists():
        return

    output_data = json.loads(output_data_path.read_text())
    metadata_item = {
        "name": "Subject ID",
        "key": _SUBJECT_ID_COLUMN,
        "value": normalized_subject_id,
    }
    for output_file in output_data.get("output_files", []):
        existing = output_file.get("metadata", [])
        existing = [item for item in existing if item.get("key") != _SUBJECT_ID_COLUMN]
        output_file["metadata"] = [*existing, metadata_item]

    output_data_path.write_text(json.dumps(output_data, indent=4))


def _load_multi_region_raw_data(
    region_runs: List[dict],
    define_epochs_by: str,
    epoch_names: str,
    epochs: Optional[str],
) -> dict:
    """Load raw traces/events for each region for cross-region metrics."""
    raw_data = {}
    parsed_epoch_names = _validate_epoch_name_strings(epoch_names)
    parsed_epoch_colors = [f"C{i}" for i in range(len(parsed_epoch_names))]
    baseline_epoch = parsed_epoch_names[0] if parsed_epoch_names else "epoch_1"
    for region in region_runs:
        manager = StateEpochDataManager(
            cell_set_files=region["cell_set_files"],
            event_set_files=region["event_set_files"],
            annotations_file=None,
            concatenate=True,
            define_epochs_by=define_epochs_by,
            epochs=epochs,
            epoch_names=parsed_epoch_names,
            epoch_colors=parsed_epoch_colors,
            state_names=[_EPOCH_ONLY_STATE_NAME],
            state_colors=["gray"],
            baseline_state=_EPOCH_ONLY_STATE_NAME,
            baseline_epoch=baseline_epoch,
            allow_epoch_only_mode=True,
        )
        traces, events, _annotations_df, cell_info = manager.load_data()
        raw_data[region["brain_region_name"]] = {
            "traces": traces,
            "events": events,
            "cell_names": list(cell_info.get("cell_names", [])),
            "epoch_periods": manager.get_epoch_periods(),
            "period": float(cell_info.get("period", 1.0)),
            "epoch_names": parsed_epoch_names,
        }
    return raw_data


def _safe_corrcoef(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute Pearson correlation safely for 1D arrays."""
    if values_a.size < 2 or values_b.size < 2:
        return float("nan")
    if np.allclose(np.std(values_a), 0.0) or np.allclose(np.std(values_b), 0.0):
        return float("nan")
    return float(np.corrcoef(values_a, values_b)[0, 1])


def _epoch_slices_from_periods(
    epoch_periods: List[tuple], period: float, n_timepoints: int
) -> List[tuple]:
    slices = []
    for start_s, end_s in epoch_periods:
        start_idx = max(0, int(float(start_s) / float(period)))
        end_idx = min(n_timepoints, int(float(end_s) / float(period)))
        if end_idx > start_idx:
            slices.append((start_idx, end_idx))
    return slices


def _compute_multi_region_raw_trace_correlation_df(
    raw_region_data: dict, region_labels: List[str]
) -> pd.DataFrame:
    """Compute full-length raw-trace pairwise correlations across all available regions."""
    rows: List[dict] = []
    valid_regions = [
        region
        for region in region_labels
        if region in raw_region_data
        and raw_region_data[region].get("traces") is not None
    ]
    region_arrays = {}
    region_names = {}

    for region in valid_regions:
        traces = raw_region_data[region]["traces"]
        if traces is None or traces.size == 0:
            continue
        cell_names = [
            str(name) for name in raw_region_data[region].get("cell_names", [])
        ]
        if not cell_names:
            cell_names = [f"cell_{idx}" for idx in range(traces.shape[1])]
        n_cells = min(traces.shape[1], len(cell_names))
        if n_cells < 2:
            continue
        region_arrays[region] = traces[:, :n_cells]
        region_names[region] = cell_names[:n_cells]

    valid_regions = [region for region in valid_regions if region in region_arrays]
    for region in valid_regions:
        traces = region_arrays[region]
        names = region_names[region]
        for i in range(traces.shape[1]):
            x = traces[:, i]
            for j in range(i + 1, traces.shape[1]):
                y = traces[:, j]
                rows.append(
                    {
                        "comparison_type": "within_region",
                        "modality": "trace",
                        "epoch": "__all_raw__",
                        "region_a": region,
                        "region_b": region,
                        "cell_a": names[i],
                        "cell_b": names[j],
                        "n_overlap_timepoints": int(len(x)),
                        "corr_value": _safe_corrcoef(x, y),
                    }
                )

    for idx_a, region_a in enumerate(valid_regions):
        traces_a = region_arrays[region_a]
        names_a = region_names[region_a]
        for region_b in valid_regions[idx_a + 1 :]:
            traces_b = region_arrays[region_b]
            names_b = region_names[region_b]
            n_overlap = min(traces_a.shape[0], traces_b.shape[0])
            if n_overlap < 2:
                continue
            sub_a = traces_a[:n_overlap, :]
            sub_b = traces_b[:n_overlap, :]
            for i in range(sub_a.shape[1]):
                x = sub_a[:, i]
                for j in range(sub_b.shape[1]):
                    y = sub_b[:, j]
                    rows.append(
                        {
                            "comparison_type": "across_region",
                            "modality": "trace",
                            "epoch": "__all_raw__",
                            "region_a": region_a,
                            "region_b": region_b,
                            "cell_a": names_a[i],
                            "cell_b": names_b[j],
                            "n_overlap_timepoints": int(n_overlap),
                            "corr_value": _safe_corrcoef(x, y),
                        }
                    )

    return pd.DataFrame(rows)


def _build_raw_trace_event_activity_df(
    raw_region_data: dict, region_labels: List[str], subject_id: Optional[str] = None
) -> pd.DataFrame:
    """Build a single long-form raw table across brain regions and epochs."""
    if not raw_region_data:
        return pd.DataFrame()

    rows = []
    for region_label in region_labels:
        region_data = raw_region_data.get(region_label)
        if region_data is None:
            continue

        traces = region_data["traces"]
        if traces is None or traces.size == 0:
            continue
        events = region_data.get("events")
        cell_names = [str(name) for name in region_data.get("cell_names", [])]
        if not cell_names:
            cell_names = [f"cell_{idx}" for idx in range(traces.shape[1])]
        epoch_names = region_data.get("epoch_names", [])
        epoch_slices = _epoch_slices_from_periods(
            epoch_periods=region_data["epoch_periods"],
            period=region_data["period"],
            n_timepoints=traces.shape[0],
        )

        for epoch_idx, (start_idx, end_idx) in enumerate(epoch_slices):
            epoch_label = (
                str(epoch_names[epoch_idx])
                if epoch_idx < len(epoch_names)
                else f"epoch_{epoch_idx + 1}"
            )
            n_timepoints = end_idx - start_idx
            if n_timepoints <= 0:
                continue

            trace_segment = traces[start_idx:end_idx, :]
            trace_frame = pd.DataFrame(trace_segment, columns=cell_names)
            trace_frame.insert(0, "epoch_frame_index", np.arange(n_timepoints))
            trace_long = trace_frame.melt(
                id_vars="epoch_frame_index",
                var_name="name",
                value_name="trace_value",
            )
            has_event_input = events is not None and events.size > 0
            if has_event_input:
                event_segment = events[start_idx:end_idx, :]
                event_frame = pd.DataFrame(event_segment, columns=cell_names)
                event_frame.insert(0, "epoch_frame_index", np.arange(n_timepoints))
                event_long = event_frame.melt(
                    id_vars="epoch_frame_index",
                    var_name="name",
                    value_name="event_value",
                )
                trace_long["event_value"] = event_long["event_value"].to_numpy(
                    dtype=float
                )
            else:
                trace_long["event_value"] = np.nan

            trace_long[_BRAIN_REGION_COLUMN] = region_label
            if subject_id is not None:
                trace_long[_SUBJECT_ID_COLUMN] = subject_id
            trace_long["epoch"] = epoch_label
            trace_long["time_s"] = (
                start_idx + trace_long["epoch_frame_index"].astype(float)
            ) * float(region_data["period"])
            trace_long["epoch_time_s"] = np.nan
            if has_event_input:
                trace_long["epoch_time_s"] = trace_long["epoch_frame_index"].astype(
                    float
                ) * float(region_data["period"])
            rows.append(trace_long)

    if not rows:
        return pd.DataFrame()
    output_columns = [
        _BRAIN_REGION_COLUMN,
        "name",
        "epoch",
        "time_s",
        "epoch_time_s",
        "trace_value",
        "event_value",
    ]
    if subject_id is not None:
        output_columns = [_SUBJECT_ID_COLUMN, *output_columns]
    return pd.concat(rows, ignore_index=True)[output_columns]


def _create_raw_trace_event_previews(
    output_dir: Path,
    cell_corr_df: pd.DataFrame,
) -> List[str]:
    """Create SVG previews for raw trace/event output."""
    preview_paths: List[str] = []
    if not cell_corr_df.empty:
        trace_rows = cell_corr_df[cell_corr_df["modality"] == "trace"].copy()
        if not trace_rows.empty:
            matrix_source = trace_rows[trace_rows["epoch"] == "__all_raw__"]
            if matrix_source.empty:
                matrix_source = trace_rows
            matrix_rows = matrix_source[
                ["region_a", "region_b", "cell_a", "cell_b", "corr_value"]
            ].dropna(subset=["corr_value"])
            if not matrix_rows.empty:
                matrix_rows = matrix_rows.assign(
                    key_a=matrix_rows["region_a"].astype(str)
                    + "::"
                    + matrix_rows["cell_a"].astype(str),
                    key_b=matrix_rows["region_b"].astype(str)
                    + "::"
                    + matrix_rows["cell_b"].astype(str),
                )
                pair_summary = (
                    matrix_rows.groupby(["key_a", "key_b"], as_index=False)[
                        "corr_value"
                    ]
                    .median()
                    .rename(columns={"corr_value": "corr_median"})
                )

                region_order = pd.concat(
                    [matrix_rows["region_a"], matrix_rows["region_b"]],
                    ignore_index=True,
                ).drop_duplicates()
                cells_by_region = {}
                for region in region_order:
                    region_cells = pd.concat(
                        [
                            matrix_rows.loc[
                                matrix_rows["region_a"] == region, "cell_a"
                            ].astype(str),
                            matrix_rows.loc[
                                matrix_rows["region_b"] == region, "cell_b"
                            ].astype(str),
                        ],
                        ignore_index=True,
                    ).drop_duplicates()
                    cells_by_region[str(region)] = sorted(region_cells.tolist())

                ordered_keys = [
                    f"{region}::{cell}"
                    for region in cells_by_region
                    for cell in cells_by_region[region]
                ]
                if ordered_keys:
                    key_to_index = {key: idx for idx, key in enumerate(ordered_keys)}
                    corr_matrix = np.full(
                        (len(ordered_keys), len(ordered_keys)), np.nan, dtype=float
                    )
                    np.fill_diagonal(corr_matrix, 1.0)

                    for row in pair_summary.itertuples(index=False):
                        i = key_to_index.get(row.key_a)
                        j = key_to_index.get(row.key_b)
                        if i is None or j is None:
                            continue
                        corr_matrix[i, j] = float(row.corr_median)
                        corr_matrix[j, i] = float(row.corr_median)

                    fig_size = min(18, max(8, 0.16 * len(ordered_keys)))
                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                    image = ax.imshow(
                        corr_matrix,
                        cmap="coolwarm",
                        vmin=-1,
                        vmax=1,
                        interpolation="nearest",
                    )
                    ax.set_title(
                        "Pairwise cell trace correlation matrix (within + across regions)"
                    )

                    if len(ordered_keys) <= 60:
                        labels = [key.split("::", 1)[1] for key in ordered_keys]
                        ax.set_xticks(range(len(ordered_keys)))
                        ax.set_xticklabels(labels, rotation=90, fontsize=6)
                        ax.set_yticks(range(len(ordered_keys)))
                        ax.set_yticklabels(labels, fontsize=6)
                    else:
                        ax.set_xticks([])
                        ax.set_yticks([])

                    offset = 0
                    for region, cells in cells_by_region.items():
                        block_size = len(cells)
                        if block_size == 0:
                            continue
                        center = offset + (block_size - 1) / 2
                        ax.text(
                            center,
                            -1.4,
                            region,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            clip_on=False,
                        )
                        ax.text(
                            -1.4,
                            center,
                            region,
                            ha="right",
                            va="center",
                            fontsize=8,
                            clip_on=False,
                        )
                        offset += block_size
                        if offset < len(ordered_keys):
                            ax.axhline(offset - 0.5, color="black", linewidth=0.8)
                            ax.axvline(offset - 0.5, color="black", linewidth=0.8)

                    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Median correlation across epochs")
                    path = output_dir / _CELL_CORR_HEATMAP_PREVIEW
                    fig.tight_layout()
                    fig.savefig(path, dpi=150, transparent=True)
                    plt.close(fig)
                    preview_paths.append(str(path))
    return preview_paths


def _generate_raw_trace_event_output_and_register(
    output_dir: Path,
    region_labels: List[str],
    raw_region_data: Optional[dict] = None,
    subject_id: Optional[str] = None,
) -> None:
    """Generate a unified raw trace/event CSV and register matrix heatmap preview."""
    output_data_path = output_dir / "output_data.json"
    if not output_data_path.exists():
        return
    output_data = json.loads(output_data_path.read_text())
    if not raw_region_data:
        return
    raw_df = _build_raw_trace_event_activity_df(
        raw_region_data, region_labels, subject_id=subject_id
    )
    if raw_df.empty:
        return

    trace_event_dir = output_dir / _TRACE_EVENT_OUTPUTS_SUBDIR
    trace_event_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = trace_event_dir / _RAW_TRACE_EVENT_ACTIVITY_CSV
    raw_df.to_csv(raw_csv_path, index=False)

    cell_corr_df = _compute_multi_region_raw_trace_correlation_df(
        raw_region_data=raw_region_data,
        region_labels=region_labels,
    )

    preview_paths = _create_raw_trace_event_previews(
        output_dir=trace_event_dir,
        cell_corr_df=cell_corr_df,
    )

    output_files = output_data.get("output_files", [])
    output_files = [
        f
        for f in output_files
        if not str(f.get("file", "")).endswith("cross_region_analysis_outputs.zip")
        and not str(f.get("file", "")).endswith(_RAW_TRACE_EVENT_ACTIVITY_CSV)
        and not str(f.get("file", "")).endswith("cross_region_raw_trace_event_data.csv")
    ]
    output_files.append(
        {
            "file": str(
                Path(_TRACE_EVENT_OUTPUTS_SUBDIR) / _RAW_TRACE_EVENT_ACTIVITY_CSV
            ),
            "previews": [
                {
                    "file": str(Path(_TRACE_EVENT_OUTPUTS_SUBDIR) / Path(preview).name),
                    "caption": "Raw trace/event preview.",
                }
                for preview in preview_paths
            ],
            "metadata": [
                {
                    "name": "Number of Brain Regions",
                    "key": "num_brain_regions",
                    "value": len(region_labels),
                },
                {
                    "name": "Brain Regions",
                    "key": "brain_regions",
                    "value": ", ".join(region_labels),
                },
                {
                    "name": "Raw Trace/Event Rows",
                    "key": "raw_trace_event_row_count",
                    "value": int(len(raw_df)),
                },
            ],
        }
    )
    output_data["output_files"] = output_files
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
    subject_id: Optional[str] = None,
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
    _generate_trace_event_output: bool = True,
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
    normalized_subject_id = _normalize_subject_id(subject_id)
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
            _generate_trace_event_output=False,
            region_selection="single_brain_region",
            subject_id=normalized_subject_id,
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
        raw_region_data = _load_multi_region_raw_data(
            region_runs=region_runs,
            define_epochs_by=define_epochs_by,
            epoch_names=epoch_names,
            epochs=epochs,
        )
        _generate_raw_trace_event_output_and_register(
            output_dir=base_output_dir,
            region_labels=[region["brain_region_name"] for region in region_runs],
            raw_region_data=raw_region_data,
            subject_id=normalized_subject_id,
        )
        _register_subject_metadata(base_output_dir, normalized_subject_id)
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
    if normalized_subject_id:
        _append_subject_id_column_to_epoch_csv_outputs(
            resolved_output_dir, normalized_subject_id
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
    if _generate_trace_event_output:
        single_region_label = (
            normalized_brain_region_name
            if normalized_brain_region_name is not None
            else _DEFAULT_FIRST_BRAIN_REGION_NAME
        )
        single_region_raw_data = _load_multi_region_raw_data(
            region_runs=[
                {
                    "cell_set_files": cell_set_files,
                    "event_set_files": event_set_files,
                    "brain_region_name": single_region_label,
                }
            ],
            define_epochs_by=define_epochs_by,
            epoch_names=epoch_names,
            epochs=epochs,
        )
        _generate_raw_trace_event_output_and_register(
            output_dir=resolved_output_dir,
            region_labels=[single_region_label],
            raw_region_data=single_region_raw_data,
            subject_id=normalized_subject_id,
        )
    _register_subject_metadata(resolved_output_dir, normalized_subject_id)


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
    subject_id: Optional[str] = None,
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
        subject_id=subject_id,
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
