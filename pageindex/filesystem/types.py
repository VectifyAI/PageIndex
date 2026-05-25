from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SearchResult:
    reference_id: str
    file_ref: str
    external_id: Optional[str]
    title: str
    snippet: str
    folder_path: str
    folder_paths: list[str]
    metadata: dict[str, Any]
    source_path: str = ""
    id: Optional[str] = None
    document_id: Optional[str] = None
    name: str = ""
    description: str = ""
    status: str = ""
    pageNum: Optional[int] = None
    createdAt: Optional[str] = None
    folderId: Optional[str] = None
    derived_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_generation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenResult:
    reference_id: str
    file_ref: str
    start_line: int
    end_line: int
    text: str
    external_id: Optional[str] = None
    folder_path: str = ""
    source_path: str = ""


@dataclass(frozen=True)
class FolderEntry:
    folder_id: str
    parent_id: Optional[str]
    name: str
    path: str
    kind: str


@dataclass(frozen=True)
class FileEntry:
    file_ref: str
    external_id: Optional[str]
    storage_uri: str
    source_path: str
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
    derived_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_generation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetadataField:
    name: str
    field_type: str
    description: str = ""
    indexed: bool = True
    faceted: bool = False
    sortable: bool = False
    source: str = "manual"


@dataclass(frozen=True)
class CommandResult:
    command: str
    data: Any
    text: str
