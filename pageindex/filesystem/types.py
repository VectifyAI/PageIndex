from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class FileEntry:
    file_ref: str
    external_id: Optional[str]
    storage_uri: str
    title: str
    descriptor: str
    content_type: str
    source_type: Optional[str]
    fingerprint: str
    text_artifact_path: str
    raw_artifact_path: Optional[str]
    pageindex_doc_id: Optional[str]
    pageindex_tree_status: str
    metadata: dict[str, Any]
    folder_path: str
    metadata_status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetadataField:
    name: str
    description: str = ""
    source: str = "manual"


@dataclass(frozen=True)
class PIFSQueryScope:
    path: str
    folder_path: str
    metadata_filter: dict[str, str] = field(default_factory=dict)
    metadata_axis: Optional[str] = None
