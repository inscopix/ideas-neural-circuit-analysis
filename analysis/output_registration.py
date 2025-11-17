"""Shared helpers for registering state-epoch analysis outputs.

This module centralizes preview scanning, prefix normalization, and file
registration so multi-tool workflows (baseline + combined analyzers) can
reuse the same logic without duplicating helper functions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ideas.outputs import OutputData

PreviewDefinition = Tuple[str, str]
MetadataMapping = Mapping[str, Dict[str, Any]]

logger = logging.getLogger(__name__)


def collect_available_previews(
    output_dir: str,
    preview_definitions: Iterable[PreviewDefinition],
) -> List[PreviewDefinition]:
    """Return preview tuples that actually exist on disk.
    
    Checks both the root output_dir and .previews/ subdirectory.
    """
    available_previews: List[PreviewDefinition] = []
    for preview_name, caption in preview_definitions:
        # Check root directory first
        preview_path = os.path.join(output_dir, preview_name)
        if os.path.exists(preview_path):
            available_previews.append((preview_name, caption))
            continue
        
        # Also check .previews/ subdirectory
        preview_subdir_path = os.path.join(output_dir, ".previews", preview_name)
        if os.path.exists(preview_subdir_path):
            available_previews.append((preview_name, caption))
    return available_previews


@dataclass(frozen=True)
class PreviewPrefixRules:
    """Rules that determine how preview filenames receive prefixes.

    Attributes
    ----------
    output_basename:
        Base prefix to attach to previews (typically the relocated output basename).
    skip_if_prefixed_with:
        Lowercase prefixes that indicate a preview already carries the correct
        context (e.g., ``group1_`` previews keep their original name).
    skip_preview_prefixes:
        Additional caller-provided prefixes (group names, dimension tags, etc.)
        that should bypass automatic prefixing.
    dimension_labels:
        Labels that, when appearing as the trailing portion of ``output_basename``,
        should be removed if the preview filename already begins with the same
        label. This avoids redundant names such as
        ``Control_vs_Treatment_states_states_comparison.svg``.
    """

    output_basename: str = ""
    skip_if_prefixed_with: Sequence[str] = ("group1_", "group2_")
    skip_preview_prefixes: Sequence[str] = ()
    dimension_labels: Sequence[str] = ("states", "epochs")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_basename",
            str(self.output_basename or "").strip(),
        )
        object.__setattr__(
            self,
            "_skip_patterns",
            tuple(
                str(pattern).strip().lower()
                for pattern in self.skip_if_prefixed_with
                if pattern
            ),
        )
        object.__setattr__(
            self,
            "_custom_skip_prefixes",
            tuple(
                str(prefix).strip().lower()
                for prefix in self.skip_preview_prefixes
                if prefix
            ),
        )
        object.__setattr__(
            self,
            "_dimension_labels",
            tuple(
                str(label).strip().lower()
                for label in self.dimension_labels
                if label
            ),
        )

    def should_skip_prefix(self, preview_name: str) -> bool:
        """Return True if the preview already contains an identifying prefix."""
        lowered = preview_name.lower()
        if any(lowered.startswith(pattern) for pattern in self._skip_patterns):
            return True

        for prefix in self._custom_skip_prefixes:
            if lowered == prefix or lowered.startswith(f"{prefix}_"):
                return True
        return False

    def _normalized_prefix(self, preview_name: str) -> str:
        """Return the prefix to attach, trimming redundant dimension suffixes."""
        if not self._basename:
            return ""
        prefix = self._basename

        last_sep = prefix.rfind("_")
        if last_sep <= 0:
            return prefix

        suffix = prefix[last_sep + 1 :].lower()
        if not suffix or suffix not in self._dimension_labels:
            return prefix

        preview_head = preview_name.split("_", 1)[0].lower()
        if preview_head != suffix:
            return prefix
        return prefix[:last_sep]

    def apply(self, preview_name: str) -> str:
        """Return the preview filename after applying prefix rules."""
        if self.should_skip_prefix(preview_name):
            return preview_name

        prefix = self._normalized_prefix(preview_name)
        if not prefix:
            return preview_name
        return f"{prefix}_{preview_name}"


def register_output_file(
    *,
    output_data: OutputData,
    output_dir: str,
    output_metadata: MetadataMapping,
    file: str,
    output_file_basename: str,
    preview_files: Optional[List[PreviewDefinition]] = None,
    attach_output_basename: bool = True,
    metadata_filter: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    preview_prefix_rules: Optional[PreviewPrefixRules] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> str:
    """Attach an output file and its previews to the OutputData registry.

    Matches the old process_output_file behavior:
    - If preview_files is provided (even if empty list): relocate to subdirectory
    - If preview_files is None: keep file in root directory
    """

    active_logger = logger_instance or logger
    previews: List[PreviewDefinition] = preview_files or []
    prefix_rules = preview_prefix_rules or PreviewPrefixRules(
        output_basename=output_file_basename
    )
    base_output_dir = output_dir or "."

    try:
        file_path = os.path.join(base_output_dir, file)
        filename = Path(file_path).name
        basename = Path(file_path).stem
        
        active_logger.debug(
            "register_output_file: file=%s, output_file_basename=%s, preview_files=%s",
            file,
            output_file_basename,
            "None" if preview_files is None else f"list[{len(preview_files)}]",
        )

        if preview_files is not None:
            # Create subdirectory named after the file's basename
            target_dir = os.path.join(base_output_dir, basename)
            os.makedirs(target_dir, exist_ok=True)
            
            # Use original filename (subdirectory provides context, no need for prefix)
            relocated_filename = filename if not attach_output_basename else f"{output_file_basename}_{filename}"
            final_file_path = os.path.join(target_dir, relocated_filename)
            active_logger.debug(
                "Relocating file from %s to %s",
                file_path,
                final_file_path,
            )
            os.rename(file_path, final_file_path)
        else:
            target_dir = base_output_dir
            final_file_path = file_path

        # Normalize path to remove leading "./" for consistency with IDEAS platform
        normalized_file_path = final_file_path.lstrip("./") if final_file_path.startswith("./") else final_file_path
        output_file = output_data.add_file(normalized_file_path)

        # Process previews
        for preview_name, caption in previews:
            # Check root directory first, then .previews/ subdirectory
            preview_path = os.path.join(base_output_dir, preview_name)
            if not os.path.exists(preview_path):
                preview_path = os.path.join(base_output_dir, ".previews", preview_name)
            
            if not os.path.exists(preview_path):
                active_logger.warning(
                    "Preview '%s' not found for output '%s'; skipping attachment.",
                    preview_name,
                    filename,
                )
                continue

            if preview_files is not None:
                # Relocate previews to the same subdirectory (use original name unless prefix requested)
                final_preview_name = preview_name if not attach_output_basename else f"{output_file_basename}_{preview_name}"
                new_preview_path = os.path.join(target_dir, final_preview_name)
            else:
                new_preview_path = preview_path
            
            if new_preview_path != preview_path:
                os.rename(preview_path, new_preview_path)
            
            # Normalize preview path to remove leading "./" for consistency
            normalized_preview_path = new_preview_path.lstrip("./") if new_preview_path.startswith("./") else new_preview_path
            output_file.add_preview(normalized_preview_path, caption)

        # Add metadata
        raw_metadata = dict(output_metadata.get(basename, {}))
        if metadata_filter:
            filtered_metadata = metadata_filter(raw_metadata)
        else:
            filtered_metadata = {
                key: value
                for key, value in raw_metadata.items()
                if value not in (None, "", [], {}, ())
            }

        for key, value in filtered_metadata.items():
            output_file.add_metadata(
                key=key,
                value=str(value),
                name=key.title(),
            )

        return basename

    except Exception:  # noqa: BLE001
        active_logger.exception(
            "failed to register output file '%s' in directory '%s'",
            file,
            output_dir,
        )
        return Path(file).stem


__all__ = [
    "collect_available_previews",
    "PreviewPrefixRules",
    "register_output_file",
]

