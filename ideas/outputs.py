from dataclasses import dataclass, field
import json
from typing import TypeVar, List
import os

OUTPUT_DATA_JSON_FILENAME = "output_data.json"
OUTPUT_DATA_SCHEMA_VERSION = "1.0.0"

Tag = TypeVar("Tag", bound=str)

@dataclass
class Preview:
    file: str
    caption: str

    def to_dict(self):
        return {
            "file": self.file,
            "caption": self.caption
        }

    @classmethod
    def from_dict(cls, instance):
        return cls(
            file=instance["file"],
            caption=instance["caption"]
        )
    
@dataclass
class Metadata:
    key: str
    name: str
    value: str

    def to_dict(self):
        return {
            "key": self.key,
            "name": self.name,
            "value": self.value
        }
    
    @classmethod
    def from_dict(cls, instance):
        return cls(
            key=instance["key"],
            name=instance["name"],
            value=instance["value"]
        )

@dataclass
class OutputFile:
    file: str
    previews: List[Preview] = field(default_factory=list)
    metadata: List[Metadata] = field(default_factory=list)
    tags: List[Tag] = field(default_factory=list)

    def to_dict(self):
        return {
            "file": self.file,
            "previews": [p.to_dict() for p in self.previews],
            "metadata": [m.to_dict() for m in self.metadata],
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, instance):
        return cls(
            file=instance["file"],
            previews=[Preview.from_dict(p) for p in instance["previews"]],
            metadata=[Metadata.from_dict(m) for m in instance["metadata"]],
            tags=[Tag(t) for t in instance["tags"]],
        )

    def add_preview(self, file: str, caption: str):
        self.previews.append(
            Preview(
                file=file,
                caption=caption
            )
        )

    def add_metadata(self, key: str, name: str, value: str):
        self.metadata.append(
            Metadata(
                key=key,
                name=name,
                value=value
            )
        )

    def add_tag(self, tag: str):
        self.tags.append(tag)


class OutputData:
    output_files: List[OutputFile]
    append: bool

    def __init__(self, append=False):
        self.filename = OUTPUT_DATA_JSON_FILENAME
        self.output_files = []
        self.append = append
    
    def open(self):
        print(f"open output data: {os.path.exists(self.filename)}")
        if self.append and os.path.exists(self.filename):
            with open(self.filename, "r") as file:
                output_data = json.load(file)
                # todo: validate schema here
                for output_file in output_data["output_files"]:
                    self.output_files.append(OutputFile.from_dict(output_file))

    def close(self):
        output_data = {
            "schema_version": OUTPUT_DATA_SCHEMA_VERSION,
            "output_files": [f.to_dict() for f in self.output_files]
        }
        with open(self.filename, "w") as file:
            json.dump(output_data, file, indent=4)

        print(f"close output data: {os.path.exists(self.filename)}\n{output_data}")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def add_file(self, file: str, previews: List[Preview] = [], metadata: List[Metadata] = [], tags: List[Tag] = []):
        output_file = OutputFile(
            file=file,
            previews=previews,
            metadata=metadata,
            tags=tags
        )
        self.output_files.append(output_file)
        return output_file
