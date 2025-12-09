"""Data model classes for IDEAS file and metadata structures.

This module provides classes for representing IDEAS files, series, groups,
and objects with their associated metadata and preview files.
"""

import os
import uuid
from enum import Enum


def generate_unique_id():
    """Generate a unique identifier using UUID4.

    Returns
    -------
        Unique identifier string

    """
    return str(uuid.uuid4())


class MetadataInstruction(Enum):
    """Metadata instructions"""

    ADD = 1, "add"
    INHERIT = 2, "inherit"


class IdeasPreviewFile:
    """Individual preview file"""

    def __init__(
        self, name, help, file_path, file_format, order=None, data_files=None
    ):
        """Construct preview file"""
        self.name = name
        self.help = help
        self.file_path = file_path
        self.file_format = file_format
        self.order = order
        self.data_files = data_files

    def to_dict(self):
        """Convert preview file to a dictionary representation"""
        d = {
            "name": self.name,
            "help": self.help,
            "file_path": self.file_path,
            "file_format": self.file_format,
        }

        if self.order is not None:
            d["order"] = self.order

        if self.data_files is not None:
            d["data"] = [f.to_dict() for f in self.data_files]

        return d


class IdeasMetadataFile:
    """Individual metadata file"""

    def __init__(self, file_path, file_format, metadata_type):
        """Construct metadata file"""
        self.file_path = file_path
        self.file_format = file_format
        self.metadata_type = metadata_type

    def to_dict(self):
        """Convert metadata file to a dictionary representation"""
        return {
            "file_path": self.file_path,
            "file_format": self.file_format,
            "metadata_type": self.metadata_type,
        }


class IdeasSeries:
    """Individual series"""

    def __init__(
        self,
        series_key,
        series_type,
        parent_ids=None,
        preview_files=None,
        add_metadata=None,
        inherit_metadata=None,
    ):
        """Construct series on IDEAS"""
        self.series_key = series_key
        self.series_type = series_type
        self.series_id = generate_unique_id()
        self.parent_ids = parent_ids
        self.preview = preview_files

        # define series name based on its type
        if self.series_type == "cell_set":
            self.series_name = "Cell Set"
        elif self.series_type == "neural_events":
            self.series_name = "Neural Events"
        elif self.series_type == "miniscope_movie":
            self.series_name = "Miniscope Movie"
        else:
            self.series_name = "N/A"

        # construct series metadata
        self.series_metadata = {
            "series_id": self.series_id,
            MetadataInstruction.ADD.value[1]: (
                {} if add_metadata is None else add_metadata
            ),
            MetadataInstruction.INHERIT.value[1]: (
                [] if inherit_metadata is None else inherit_metadata
            ),
        }

        if (
            "ideas"
            not in self.series_metadata[MetadataInstruction.ADD.value[1]]
        ):
            self.series_metadata[MetadataInstruction.ADD.value[1]][
                "ideas"
            ] = {}

        if (
            "dataset"
            not in self.series_metadata[MetadataInstruction.ADD.value[1]][
                "ideas"
            ]
        ):
            self.series_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
                "dataset"
            ] = {}

    def to_dict(self):
        """Convert series to a dictionary representation"""
        d = {
            "series_key": self.series_key,
            "series_name": self.series_name,
            "series_id": self.series_id,
            "series_type": self.series_type,
        }

        if self.parent_ids is not None:
            d["parent_ids"] = self.parent_ids

        if self.preview is not None:
            d["preview"] = [f.to_dict() for f in self.preview]

        return d


class IdeasFile:
    """Individual file"""

    def __init__(
        self,
        file_key,
        file_path,
        file_type,
        file_format,
        file_structure,
        file_category,
        series=None,
        series_order=None,
        parent_ids=None,
        preview_files=None,
        add_metadata=None,
        inherit_metadata=None,
    ):
        """Construct file on IDEAS"""
        self.file_key = file_key
        self.file_name = os.path.basename(file_path)
        self.file_id = generate_unique_id()
        self.file_path = file_path
        self.file_type = file_type
        self.file_format = file_format
        self.file_structure = file_structure
        self.file_category = file_category
        self.parent_ids = parent_ids
        self.preview = preview_files
        self.series = series
        self.series_order = series_order

        # construct file metadata
        self.file_metadata = {
            "file_id": self.file_id,
            MetadataInstruction.ADD.value[1]: (
                {} if add_metadata is None else add_metadata
            ),
            MetadataInstruction.INHERIT.value[1]: (
                [] if inherit_metadata is None else inherit_metadata
            ),
        }

        if "ideas" not in self.file_metadata[MetadataInstruction.ADD.value[1]]:
            self.file_metadata[MetadataInstruction.ADD.value[1]]["ideas"] = {}
        if (
            "dataset"
            not in self.file_metadata[MetadataInstruction.ADD.value[1]][
                "ideas"
            ]
        ):
            self.file_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
                "dataset"
            ] = {}
        self.file_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
            "dataset"
        ]["file_type"] = self.file_type
        self.file_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
            "dataset"
        ]["file_format"] = self.file_format
        self.file_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
            "dataset"
        ]["file_structure"] = self.file_structure

    def to_dict(self):
        """Convert file to a dictionary representation"""
        d = {
            "file_key": self.file_key,
            "file_name": self.file_name,
            "file_id": self.file_id,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_format": self.file_format,
            "file_structure": self.file_structure,
            "file_category": self.file_category,
        }

        if self.series is not None:
            d["series"] = {
                "series_id": self.series.series_id,
                "series_order": self.series_order,
            }

        if self.parent_ids is not None:
            d["parent_ids"] = self.parent_ids

        if self.preview is not None:
            d["preview"] = [f.to_dict() for f in self.preview]

        return d


class IdeasGroup:
    """Collection of objects and files"""

    def __init__(
        self,
        group_key,
        group_type,
        files=None,
        series=None,
        add_metadata=None,
        inherit_metadata=None,
    ):
        """Construct group on IDEAS"""
        self.group_key = group_key
        self.group_type = group_type
        self.group_id = generate_unique_id()

        if files is not None:
            self.files = {f.file_id: f for f in files}
        else:
            self.files = {}

        if series is not None:
            self.series = {s.series_id: s for s in series}
        else:
            self.series = {}

        self.group_metadata = {
            "group_id": self.group_id,
            MetadataInstruction.ADD.value[1]: (
                {} if add_metadata is None else add_metadata
            ),
            MetadataInstruction.INHERIT.value[1]: (
                [] if inherit_metadata is None else inherit_metadata
            ),
        }

        if (
            "ideas"
            not in self.group_metadata[MetadataInstruction.ADD.value[1]]
        ):
            self.group_metadata[MetadataInstruction.ADD.value[1]]["ideas"] = {}
        if (
            "dataset"
            not in self.group_metadata[MetadataInstruction.ADD.value[1]][
                "ideas"
            ]
        ):
            self.group_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
                "dataset"
            ] = {}
        self.group_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
            "dataset"
        ]["group_type"] = self.group_type

    def to_dict(self):
        """Convert group to a dictionary representation"""
        d = {
            "group_key": self.group_key,
            "group_type": self.group_type,
            "group_id": self.group_id,
        }

        if self.series is not None and len(self.series) != 0:
            d["series"] = [s.to_dict() for sid, s in self.series.items()]
        else:
            d["series"] = []

        if self.files is not None and len(self.files) != 0:
            d["files"] = [f.to_dict() for fid, f in self.files.items()]
        else:
            d["files"] = []

        return d

    def add_group_metadata(self, key, value):
        """Add group-level metadata.

        Args:
            key: Metadata key
            value: Metadata value associated with the key

        Raises
        ------
        KeyError
            If the metadata key is already defined

        """
        if key in self.group_metadata[MetadataInstruction.ADD.value[1]]:
            raise KeyError("Group metadata key is already defined")
        self.group_metadata[MetadataInstruction.ADD.value[1]][key] = value


class IdeasObject:
    """Collection of files"""

    def __init__(
        self,
        object_key,
        object_type,
        object_files=None,
        is_series=None,
        add_metadata=None,
        inherit_metadata=None,
    ):
        """Construct object on IDEAS"""
        self.object_key = object_key
        self.object_type = object_type
        self.object_id = generate_unique_id()
        self.is_series = is_series

        # add files that belong to object
        if object_files is not None:
            self.object_data = {f.file_id: f for f in object_files}
        else:
            self.object_data = {}

        # construct object metadata
        self.object_metadata = {
            "object_id": self.object_id,
            MetadataInstruction.ADD.value[1]: (
                {} if add_metadata is None else add_metadata
            ),
            MetadataInstruction.INHERIT.value[1]: (
                [] if inherit_metadata is None else inherit_metadata
            ),
        }

        if (
            "ideas"
            not in self.object_metadata[MetadataInstruction.ADD.value[1]]
        ):
            self.object_metadata[MetadataInstruction.ADD.value[1]][
                "ideas"
            ] = {}
        if (
            "dataset"
            not in self.object_metadata[MetadataInstruction.ADD.value[1]][
                "ideas"
            ]
        ):
            self.object_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
                "dataset"
            ] = {}
        self.object_metadata[MetadataInstruction.ADD.value[1]]["ideas"][
            "dataset"
        ]["object_type"] = self.object_type

        # set object name
        if object_files and len(object_files) == 1 and not is_series:
            # if object contain a single file, name object after file
            self.object_name = object_files[0].file_name
        elif is_series:
            # if object represent a series, name object after first file with series suffix
            self.object_name = object_files[0].file_name + "-series"
        elif object_files and len(object_files) > 1:
            # if object contains multiple files, name it after the first file
            self.object_name = object_files[0].file_name
        else:
            # in last resort, name object after object key
            self.object_name = object_key

    def to_dict(self):
        """Convert object to a dictionary representation"""
        d = {
            "object_key": self.object_key,
            "object_name": self.object_name,
            "object_type": self.object_type,
            "object_id": self.object_id,
        }

        if self.is_series is not None:
            d["is_series"] = self.is_series

        if self.object_data is not None and len(self.object_data) != 0:
            d["object_data"] = [
                f.to_dict() for fid, f in self.object_data.items()
            ]

        return d
