from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import quote, unquote, urlparse

from ._projection_topology import (
    projection_database_pair,
    projection_database_path_present,
    projection_database_paths,
)
from .metadata import MetadataQueryEngine
from .store import (
    SQLiteFileSystemStore,
    fingerprint,
    make_file_ref,
    normalize_path,
)
from .types import PIFSQueryScope

if TYPE_CHECKING:
    from ..client import PageIndexClient
    from .semantic_projection import _EmbeddingCacheKey

PROJECTION_INDEX_STATUSES = {
    "not_indexed",
    "pending_index",
    "generated",
    "ready",
    "failed",
}

DEFAULT_EMBEDDING_DIMENSIONS = 1024
PAGEINDEX_DOCUMENT_SUFFIXES = {".pdf", ".md", ".markdown"}
PAGEINDEX_DOCUMENT_CONTENT_TYPES = {
    "application/pdf",
    "text/markdown",
    "text/x-markdown",
    "application/markdown",
}
ADD_FILE_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


@dataclass
class _RegistrationRollbackSnapshot:
    preexisting_pageindex_doc_ids: set[str]
    artifact_baselines: dict[Path, bytes | None] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    new_records: list[dict[str, Any]] = field(default_factory=list)
    created_folder_paths: list[str] = field(default_factory=list)
    new_metadata_fields: set[str] = field(default_factory=set)
    new_cache_keys: set[_EmbeddingCacheKey] = field(default_factory=set)
    catalog_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    membership_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata_value_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    projection_rows: dict[
        str,
        tuple[dict[str, Any], dict[str, Any]],
    ] = field(default_factory=dict)


def strip_pageindex_text_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_pageindex_text_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_pageindex_text_fields(item)
            for key, item in value.items()
            if key != "text"
        }
    return value


class PageIndexFileSystem:
    def __init__(
        self,
        workspace: Union[str, Path],
        *,
        summary_projection_index_dir: Union[str, Path, None] = None,
        summary_projection_embedding_model: str = "text-embedding-3-small",
        summary_projection_embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        summary_projection_embedding_timeout: float = 60,
        summary_projection_embedding_api_key: str | None = None,
        summary_projection_embedding_base_url: str | None = None,
    ):
        self.workspace = Path(workspace).expanduser()
        self.summary_projection_index_dir = (
            Path(summary_projection_index_dir).expanduser()
            if summary_projection_index_dir is not None
            else self.workspace / "artifacts" / "projection_indexes"
        )
        summary_path, cache_path = projection_database_paths(
            self.summary_projection_index_dir
        )
        catalog_path = self.workspace / "filesystem.sqlite"
        catalog_present = catalog_path.exists() or catalog_path.is_symlink()
        summary_present = projection_database_path_present(summary_path)
        cache_present = projection_database_path_present(cache_path)
        if catalog_present or summary_present or cache_present:
            SQLiteFileSystemStore.validate_existing_database(catalog_path)
        database_pair = projection_database_pair(self.summary_projection_index_dir)
        if database_pair is not None:
            from .semantic_projection import validate_projection_topology
            from ._workspace_consistency import validate_workspace_consistency

            validate_projection_topology(self.summary_projection_index_dir)
            validate_workspace_consistency(
                catalog_path,
                database_pair[0],
            )
        elif catalog_present:
            from ._workspace_consistency import validate_catalog_without_projection

            validate_catalog_without_projection(catalog_path)
        self.store = SQLiteFileSystemStore(self.workspace)
        self.metadata = MetadataQueryEngine(self.store)
        self.summary_projection: Any | None = None
        self.summary_projection_embedding_model = summary_projection_embedding_model
        self.summary_projection_embedding_dimensions = summary_projection_embedding_dimensions
        self.summary_projection_embedding_timeout = summary_projection_embedding_timeout
        self.summary_projection_embedding_api_key = summary_projection_embedding_api_key
        self.summary_projection_embedding_base_url = summary_projection_embedding_base_url

    def register_file(
        self,
        *,
        storage_uri: str,
        folder_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        external_id: Optional[str] = None,
        title: Optional[str] = None,
        content: str = "",
        content_type: str | None = None,
        source_type: Optional[str] = None,
    ) -> str:
        return self.register_files(
            [
                {
                    "storage_uri": storage_uri,
                    "folder_path": folder_path,
                    "metadata": metadata,
                    "external_id": external_id,
                    "title": title,
                    "content": content,
                    "content_type": content_type,
                    "source_type": source_type,
                }
            ]
        )[0]

    def add_file(
        self,
        physical_path: Union[str, Path],
        virtual_target: Union[str, Path],
    ) -> dict[str, Any]:
        source = Path(physical_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        suffix = source.suffix.lower()
        content_type = ADD_FILE_CONTENT_TYPES.get(suffix)
        if content_type is None:
            supported = ", ".join(sorted(ADD_FILE_CONTENT_TYPES))
            raise ValueError(
                f"Unsupported file type: {suffix or '<none>'}; supported: {supported}"
            )

        folder_path, filename, virtual_path = self._resolve_add_target(
            virtual_target,
            physical_basename=source.name,
            physical_suffix=suffix,
        )
        if self.store.file_basename_exists_in_folder(folder_path, filename):
            raise FileExistsError(f"File already exists at {virtual_path}")
        projection = self._ensure_summary_projection()
        add_created_folder_paths = self._add_created_folder_paths(folder_path)
        file_ref = make_file_ref(virtual_path.strip("/"))
        uploads_dir = self.workspace / "artifacts" / "uploads"
        final_dir = uploads_dir / file_ref
        final_path = final_dir / filename
        final_dir_created = False
        catalog_inserted = False
        records: list[dict[str, Any]] = []
        cache_keys: set[_EmbeddingCacheKey] = set()
        preexisting_cache_keys: set[_EmbeddingCacheKey] = set()
        preexisting_pageindex_doc_ids = self._pageindex_cache_doc_ids()

        uploads_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".add-{file_ref}-", dir=uploads_dir) as tmp:
            temp_path = Path(tmp) / filename
            try:
                shutil.copy2(source, temp_path)
                if final_dir.exists():
                    raise FileExistsError(
                        f"Workspace artifact already exists for {virtual_path}: {final_dir}"
                    )
                final_dir.mkdir(parents=True)
                final_dir_created = True
                os.replace(temp_path, final_path)

                record = self._prepare_file_record(
                    {
                        "storage_uri": final_path.as_uri(),
                        "folder_path": folder_path,
                        "metadata": {},
                        "external_id": None,
                        "title": filename,
                        "content": self._add_file_content(final_path, content_type),
                        "content_type": content_type,
                    }
                )
                records = [record]
                cache_keys = projection.cache_keys_for_records(records)
                preexisting_cache_keys = projection.existing_cache_keys(cache_keys)
                self._require_add_pageindex_ready(record)
                self._register_custom_metadata_fields(records)
                self.store.insert_files(records)
                catalog_inserted = True
                if self._complete_summary_projection_index(record):
                    self.store.update_file_metadata_status(
                        record["file_ref"],
                        metadata=record["metadata"],
                        metadata_status=record["metadata_status"],
                    )
                self._require_summary_projection_ready(record, operation="add")
                self._sync_owned_raw_artifact(record)
                self._ensure_add_semantic_retrieval_ready()
            except Exception:
                if catalog_inserted:
                    self._cleanup_catalog_record(file_ref)
                self._cleanup_summary_projection_records(records)
                self._cleanup_summary_projection_cache(
                    cache_keys - preexisting_cache_keys
                )
                self._cleanup_failed_register_artifacts(records)
                self._cleanup_pageindex_cache(records, preexisting_pageindex_doc_ids)
                self._cleanup_created_folders(add_created_folder_paths)
                if final_dir_created:
                    shutil.rmtree(final_dir, ignore_errors=True)
                raise

        info = self.store.file_info(file_ref)
        info["path"] = virtual_path
        return info

    def register_files(self, files: list[dict[str, Any]]) -> list[str]:
        files = [
            {
                **file,
                "metadata": self._validated_register_metadata(file.get("metadata")),
            }
            for file in files
        ]
        rollback = _RegistrationRollbackSnapshot(
            preexisting_pageindex_doc_ids=self._pageindex_cache_doc_ids()
        )
        try:
            for file in files:
                rollback.records.append(
                    self._prepare_file_record(
                        file,
                        artifact_baselines=rollback.artifact_baselines,
                    )
                )
            projection = self._ensure_summary_projection() if rollback.records else None
            self._capture_existing_registration_rows(rollback, projection)
            rollback.new_records = [
                record
                for record in rollback.records
                if record["file_ref"] not in rollback.catalog_rows
            ]
            rollback.created_folder_paths = sorted(
                {
                    path
                    for record in rollback.records
                    for path in self._add_created_folder_paths(record["folder_path"])
                },
                key=lambda path: (path.count("/"), path),
            )
            rollback.new_metadata_fields = {
                name
                for name in self._custom_metadata_field_names(rollback.records)
                if not self.store.metadata_field_exists(name)
            }
            if projection is not None:
                batch_cache_keys = projection.cache_keys_for_records(rollback.records)
                rollback.new_cache_keys = (
                    batch_cache_keys - projection.existing_cache_keys(batch_cache_keys)
                )
            self._register_custom_metadata_fields(rollback.records)
            self.store.insert_files(rollback.records)
            for record in rollback.records:
                try:
                    if self._complete_summary_projection_index(record):
                        self.store.update_file_metadata_status(
                            record["file_ref"],
                            metadata=record["metadata"],
                            metadata_status=record["metadata_status"],
                        )
                    self._require_summary_projection_ready(
                        record,
                        operation="registration",
                    )
                    self._sync_owned_raw_artifact(record)
                except KeyError:
                    continue
        except Exception:
            self._cleanup_summary_projection_records(rollback.new_records)
            self._restore_existing_registration_projection(rollback)
            self._cleanup_summary_projection_cache(rollback.new_cache_keys)
            for record in rollback.new_records:
                self._cleanup_catalog_record(str(record["file_ref"]))
            self._restore_existing_registration_catalog(rollback)
            self._cleanup_created_folders(rollback.created_folder_paths)
            self._cleanup_new_metadata_fields(rollback.new_metadata_fields)
            self._cleanup_pageindex_cache(
                rollback.records,
                rollback.preexisting_pageindex_doc_ids,
            )
            self._restore_registration_artifact_baselines(
                rollback.artifact_baselines
            )
            raise
        return [record["file_ref"] for record in rollback.records]

    def _ensure_summary_projection(self) -> Any:
        return self._open_summary_projection(create=True)

    def _ensure_add_semantic_retrieval_ready(self) -> None:
        projection = self._open_summary_projection(create=False)
        if not projection.available:
            raise RuntimeError("pifs add failed to make the Summary Projection available")

    def _summary_embedding_profile(self) -> Any:
        from .semantic_projection import SummaryEmbeddingProfile

        return SummaryEmbeddingProfile(
            base_url=self.summary_projection_embedding_base_url,
            model=self.summary_projection_embedding_model,
            dimensions=self.summary_projection_embedding_dimensions,
            timeout=self.summary_projection_embedding_timeout,
            api_key=self.summary_projection_embedding_api_key,
        )

    def _open_summary_projection(self, *, create: bool) -> Any:
        if self.summary_projection is None:
            from .semantic_projection import SummaryProjection

            self.summary_projection = SummaryProjection(
                self.summary_projection_index_dir,
                profile=self._summary_embedding_profile(),
                create=create,
            )
        return self.summary_projection

    def _add_created_folder_paths(self, folder_path: str) -> list[str]:
        paths = self._folder_ancestor_paths(folder_path)
        return [path for path in paths if not self.store.folder_exists(path)]

    @staticmethod
    def _folder_ancestor_paths(folder_path: str) -> list[str]:
        normalized = normalize_path(folder_path)
        if normalized == "/":
            return []
        segments = [segment for segment in normalized.strip("/").split("/") if segment]
        paths: list[str] = []
        for index in range(1, len(segments) + 1):
            paths.append("/" + "/".join(segments[:index]))
        return paths

    def resolve_query_scope(self, path: str) -> PIFSQueryScope:
        normalized = normalize_path(path)
        if self._folder_exists(normalized):
            return PIFSQueryScope(path=normalized, folder_path=normalized)

        parts = [part for part in normalized.strip("/").split("/") if part]
        folder_path = "/"
        remainder = parts
        for index in range(len(parts), -1, -1):
            candidate = "/" + "/".join(parts[:index]) if index else "/"
            if self._folder_exists(candidate):
                folder_path = candidate
                remainder = parts[index:]
                break

        if not remainder:
            return PIFSQueryScope(path=normalized, folder_path=folder_path)

        metadata_filter: dict[str, str] = {}
        index = 0
        while index < len(remainder):
            segment = remainder[index]
            if not segment.startswith("@"):
                if index == 0:
                    raise KeyError(f"Unknown folder path: {normalized}")
                raise ValueError(
                    "Metadata axes must come after the physical folder prefix; "
                    "inspect the physical folder first, then append @field/value segments. "
                    "Use the path returned by tree for values containing '/'."
                )
            axis_segment = segment[1:]
            if "=" in axis_segment:
                raise ValueError(
                    "Metadata virtual paths use @field/value; run tree <scope>/@field "
                    "and copy the returned path."
                )
            field = unquote(axis_segment)
            self.metadata.validate_field_name(field)
            if not self.store.metadata_field_exists(field):
                raise ValueError("Unknown metadata axis; run tree <scope> to inspect available @field axes.")
            if field in metadata_filter:
                raise ValueError(
                    "A metadata field can appear only once in a scope path; "
                    "choose one value or use browse --where for advanced predicates."
                )
            value_index = index + 1
            if value_index == len(remainder):
                return PIFSQueryScope(
                    path=normalized,
                    folder_path=folder_path,
                    metadata_filter=metadata_filter,
                    metadata_axis=field,
                )
            encoded_value = remainder[value_index]
            if encoded_value.startswith("@"):
                raise ValueError(
                    "Metadata axis inspection must be the final path segment; "
                    "choose a value with @field/value before appending another axis."
                )
            metadata_filter[field] = unquote(encoded_value)
            index += 2

        return PIFSQueryScope(
            path=normalized,
            folder_path=folder_path,
            metadata_filter=metadata_filter,
        )

    def merge_scope_filter(
        self,
        scope: PIFSQueryScope,
        metadata_filter: dict[str, Any] | str | None,
    ) -> dict[str, Any] | None:
        parsed = self.metadata.parse_filter(metadata_filter)
        if scope.metadata_axis is not None:
            raise ValueError(
                "Metadata axis paths require @field/value; run tree <scope>/@field to inspect values."
            )
        if not parsed:
            return dict(scope.metadata_filter) or None
        overlap = set(scope.metadata_filter).intersection(self.metadata.filter_fields(parsed))
        if overlap:
            raise ValueError(
                "Do not constrain the same metadata field in both the path and --where; "
                "move the predicate into one place."
            )
        if not scope.metadata_filter:
            return parsed
        return {**scope.metadata_filter, **parsed}

    def scope_file_count(self, scope: PIFSQueryScope) -> int:
        return self.store.count_files(
            scope={"folder_path": scope.folder_path, "recursive": True},
            metadata_filter=scope.metadata_filter or None,
        )

    def scope_folders(
        self,
        scope: PIFSQueryScope,
        *,
        max_depth: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.store.find_folders(
            scope.folder_path,
            metadata_filter=scope.metadata_filter or None,
            limit=limit,
            max_depth=max_depth,
            include_self=False,
        )

    def scope_metadata_axes(self, scope: PIFSQueryScope) -> list[dict[str, Any]]:
        return self.store.list_metadata_axes(
            scope={"folder_path": scope.folder_path, "recursive": True},
            metadata_filter=scope.metadata_filter or None,
            exclude_fields=set(scope.metadata_filter),
        )

    def scope_metadata_values(
        self,
        scope: PIFSQueryScope,
        *,
        page: int,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        if scope.metadata_axis is None:
            return [], False
        rows = self.store.list_metadata_values(
            scope.metadata_axis,
            scope={"folder_path": scope.folder_path, "recursive": True},
            metadata_filter=scope.metadata_filter or None,
            limit=page_size + 1,
            offset=(page - 1) * page_size,
        )
        has_more = len(rows) > page_size
        return rows[:page_size], has_more

    def scope_files(self, scope: PIFSQueryScope, *, limit: int) -> list[dict[str, Any]]:
        leaf_items = self._scope_file_leaf_items(scope, limit=self.scope_file_count(scope) + 1)
        locator_leaf_by_file_ref = self._scope_locator_leaf_by_file_ref(leaf_items)
        files = []
        for row, leaf in leaf_items:
            locator_leaf = locator_leaf_by_file_ref[row["file_ref"]]
            files.append(
                {
                    "path": self._scope_file_locator(scope, locator_leaf),
                    "locator_leaf": locator_leaf,
                    "type": "file",
                    "file_ref": row["file_ref"],
                    "external_id": row["external_id"],
                    "title": leaf,
                    "pageindex_tree_status": row["pageindex_tree_status"],
                    "metadata": row["metadata"],
                }
            )
        files = sorted(
            files,
            key=lambda item: (
                str(item["title"]).lower(),
                item["path"],
                item["file_ref"],
            ),
        )
        return files[:limit]

    def scope_file_locator(self, scope: PIFSQueryScope, file_ref: str, leaf: str) -> str:
        leaf_items = self._scope_file_leaf_items(scope, limit=self.scope_file_count(scope) + 1)
        return self._scope_file_locator(
            scope,
            self._scope_locator_leaf_by_file_ref(leaf_items).get(file_ref, leaf),
        )

    def _scope_file_leaf_items(self, scope: PIFSQueryScope, *, limit: int) -> list[tuple[dict[str, Any], str]]:
        recursive = bool(scope.metadata_filter)
        rows = self.store.list_files(
            scope={"folder_path": scope.folder_path, "recursive": recursive},
            metadata_filter=scope.metadata_filter or None,
            limit=limit,
        )
        items = []
        for row in rows:
            folder_paths = [
                folder["path"]
                for folder in self.store.folder_memberships(row["file_ref"])
            ]
            folder_path = self._preferred_folder_path(
                folder_paths,
                scope.folder_path,
                row["folder_path"],
            )
            leaf = self.store.membership_display_name(row["file_ref"], folder_path) or row["title"]
            items.append((row, leaf))
        return items

    @staticmethod
    def _scope_leaf_counts(leaf_items: list[tuple[dict[str, Any], str]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, leaf in leaf_items:
            counts[leaf] = counts.get(leaf, 0) + 1
        return counts

    @classmethod
    def _scope_locator_leaf_by_file_ref(cls, leaf_items: list[tuple[dict[str, Any], str]]) -> dict[str, str]:
        leaf_counts = cls._scope_leaf_counts(leaf_items)
        reserved = {leaf for leaf, count in leaf_counts.items() if count == 1}
        used: set[str] = set()
        next_suffix: dict[str, int] = {}
        locator_leaf_by_file_ref: dict[str, str] = {}
        for row, leaf in sorted(leaf_items, key=lambda item: (item[1].lower(), item[1], item[0]["file_ref"])):
            if leaf_counts[leaf] == 1:
                locator_leaf = leaf
            else:
                suffix = next_suffix.get(leaf, 1)
                locator_leaf = f"{leaf}~{suffix}"
                while locator_leaf in used or locator_leaf in reserved:
                    suffix += 1
                    locator_leaf = f"{leaf}~{suffix}"
                next_suffix[leaf] = suffix + 1
            used.add(locator_leaf)
            locator_leaf_by_file_ref[row["file_ref"]] = locator_leaf
        return locator_leaf_by_file_ref

    def scope_stat(self, path: str) -> dict[str, Any]:
        scope = self.resolve_query_scope(path)
        data = {
            "path": scope.path,
            "folder_path": scope.folder_path,
            "metadata_filter": dict(scope.metadata_filter),
            "file_count": self.scope_file_count(scope),
            "available_axes": [item["name"] for item in self.scope_metadata_axes(scope)],
        }
        if scope.metadata_axis is not None:
            data["metadata_axis"] = scope.metadata_axis
        return data

    @staticmethod
    def encode_scope_segment(segment: Any) -> str:
        value = str(segment)
        if value in {".", ".."}:
            return value.replace(".", "%2E")
        return quote(value, safe="")

    def browse_semantic_files(
        self,
        path: str,
        query: str,
        *,
        recursive: bool = False,
        page: int = 1,
        metadata_filter: Optional[dict[str, Any] | str] = None,
    ) -> dict[str, Any]:
        page_size = 10
        path = normalize_path(path)
        query_scope = self.resolve_query_scope(path)
        self.store.folder_info(query_scope.folder_path)
        query_text = self._query_text(query).strip()
        if not query_text:
            raise ValueError("browse requires a query")
        if page < 1:
            raise ValueError("browse --page must be at least 1")
        if self.summary_projection is None:
            index_path = self.summary_projection_index_dir / "summary.sqlite"
            if not index_path.exists():
                raise ValueError("browse Summary Projection is not available")
        projection = self._open_summary_projection(create=False)
        if not projection.available:
            raise ValueError("browse Summary Projection is not available")
        parsed_filter = self.merge_scope_filter(query_scope, metadata_filter)
        effective_recursive = recursive or bool(query_scope.metadata_filter)
        scope = {"folder_path": query_scope.folder_path, "recursive": effective_recursive}
        scope_file_refs = self.store.file_refs_for_scope(
            scope=scope,
            metadata_filter=parsed_filter,
        )
        offset = (page - 1) * page_size
        needed = offset + page_size + 1
        candidates = (
            projection.search(
                query_text,
                limit=needed,
                file_refs=scope_file_refs,
            )
            if scope_file_refs
            else []
        )
        scope_file_ref_set = set(scope_file_refs)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            file_ref = candidate.file_ref
            if file_ref in seen:
                continue
            if file_ref not in scope_file_ref_set:
                continue
            if not self.store.file_matches(
                file_ref,
                scope=scope,
                metadata_filter=parsed_filter,
            ):
                continue
            entry = self.store.get_file(file_ref)
            folder_paths = [
                folder["path"]
                for folder in self.store.folder_memberships(file_ref)
            ]
            folder_path = self._preferred_folder_path(
                folder_paths,
                query_scope.folder_path,
                entry.folder_path,
            )
            display_title = self.store.membership_display_name(file_ref, folder_path) or entry.title
            try:
                if query_scope.metadata_filter:
                    stable_path = self.scope_file_locator(query_scope, file_ref, display_title)
                else:
                    stable_path = self._stable_file_locator(
                        file_ref,
                        entry,
                        folder_path=folder_path,
                    )
            except RuntimeError:
                continue
            seen.add(file_ref)
            rank = len(rows) + 1
            rows.append(
                {
                    "rank": rank,
                    "similarity": candidate.similarity,
                    "path": stable_path,
                    "file_ref": file_ref,
                    "external_id": entry.external_id,
                    "title": display_title,
                    "pageindex_tree_status": entry.pageindex_tree_status,
                    "folder_path": folder_path,
                    "folder_paths": folder_paths,
                    "summary": str((entry.metadata or {}).get("summary") or ""),
                    "metadata": entry.metadata,
                }
            )
            if len(rows) >= needed:
                break
        page_rows = rows[offset : offset + page_size]
        payload = {
            "mode": "files",
            "retrieval": "summary",
            "query": query,
            "scope": query_scope.path,
            "recursive": effective_recursive,
            "page": page,
            "page_size": page_size,
            "has_more": len(rows) > offset + page_size,
            "data": page_rows,
        }
        if metadata_filter is not None:
            payload["where"] = self._metadata_filter_payload(metadata_filter)
        return payload

    def folder_info(self, path: str = "/") -> dict[str, Any]:
        return self.store.folder_info(path)

    def create_folder(
        self,
        path: str,
        kind: str = "manual",
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.store.create_folder(
            path,
            kind=kind,
            description=description,
            metadata=metadata,
        )

    def attach_file_to_folder(
        self,
        file_ref: str,
        folder_path_or_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.store.attach_file_to_folder(file_ref, folder_path_or_id, metadata=metadata)

    def attach_files_to_folders(self, items: list[dict[str, Any]]) -> None:
        self.store.attach_files_to_folders(items)

    def set_metadata(
        self,
        target: str,
        metadata: dict[str, Any],
        *,
        clear: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if "summary" in metadata:
            raise ValueError("setmeta cannot edit PageIndex summary")
        file_ref = self._resolve_target(target)
        info = self.store.file_info(file_ref)
        replacement = {} if clear else dict(metadata)
        for name in replacement:
            self.metadata.validate_field_name(str(name))
        existing = dict(info.get("metadata") or {})
        summary = existing.get("summary")
        if summary:
            replacement["summary"] = summary
        self._register_custom_metadata_fields([{"metadata": replacement}])
        self.store.update_file_metadata_status(
            file_ref,
            metadata=replacement,
            metadata_status=dict(info.get("metadata_status") or {}),
        )
        updated = self.store.file_info(file_ref)
        entry = self.store.get_file(file_ref)
        folder_paths = [folder["path"] for folder in updated.get("folders", [])]
        folder_path = self._preferred_folder_path(
            folder_paths,
            entry.folder_path,
            entry.folder_path,
        )
        updated["path"] = self._stable_file_locator(
            file_ref,
            entry,
            folder_path=folder_path,
        )
        return updated

    def pageindex_structure(
        self,
        target: str,
    ) -> dict[str, Any]:
        file_ref = self._resolve_target(target)
        entry = self.store.get_file(file_ref)
        self._require_pageindex_document_file(entry, "cat --structure")
        client, doc_id = self._pageindex_client_doc_for_entry(entry)
        if doc_id is None:
            return self._structural_unavailable(
                "structure",
                entry,
                message=(
                    "PageIndex structure is not cached for this file in the "
                    "PageIndexClient workspace."
                ),
            )
        structure = self._client_json(client.get_document_structure(doc_id))
        if isinstance(structure, dict) and structure.get("error"):
            return self._structural_unavailable(
                "structure",
                entry,
                message=str(structure["error"]),
            )
        return {
            "mode": "structure",
            "file_ref": file_ref,
            "external_id": entry.external_id,
            "status": entry.pageindex_tree_status,
            "available": True,
            "pageindex_doc_id": doc_id,
            "structure": strip_pageindex_text_fields(structure),
        }

    def pageindex_pages(self, target: str, pages: str) -> dict[str, Any]:
        file_ref = self._resolve_target(target)
        entry = self.store.get_file(file_ref)
        self._require_pageindex_document_file(entry, "cat --page")
        client, doc_id = self._pageindex_client_doc_for_entry(entry)
        if doc_id is None:
            return self._structural_unavailable(
                "page",
                entry,
                pages=pages,
                message=(
                    "PageIndex page content is not cached for this file in the "
                    "PageIndexClient workspace."
                ),
            )
        page_entries = self._client_json(client.get_page_content(doc_id, pages))
        if isinstance(page_entries, dict) and page_entries.get("error"):
            return self._structural_unavailable(
                "page",
                entry,
                pages=pages,
                message=str(page_entries["error"]),
            )
        if not isinstance(page_entries, list) or not page_entries:
            return self._structural_unavailable(
                "page",
                entry,
                pages=pages,
                message="Requested PageIndex page content is not cached for this file.",
            )
        text = "\n\n".join(str(page.get("content") or "") for page in page_entries)
        return {
            "mode": "page",
            "file_ref": file_ref,
            "external_id": entry.external_id,
            "status": entry.pageindex_tree_status,
            "available": True,
            "pageindex_doc_id": doc_id,
            "pages": pages,
            "data": page_entries,
            "text": text,
        }

    def _require_pageindex_document_file(self, entry: Any, command: str) -> None:
        if self._file_format(entry) in {"pdf", "markdown", "pageindex"}:
            return
        raise ValueError(
            f"{command} is only supported for PDF/Markdown PageIndex files; "
            f"got title={entry.title!r}, content_type={entry.content_type!r}. "
            "Use grep <query> <file> for single-document lexical evidence."
        )

    @classmethod
    def _file_format(cls, entry: Any) -> str:
        if getattr(entry, "pageindex_doc_id", None) or entry.pageindex_tree_status != "not_built":
            return "pageindex"
        file_format = cls._content_format(getattr(entry, "title", ""), entry.content_type)
        if file_format != "unsupported":
            return file_format
        return "unsupported"

    @classmethod
    def _content_format(cls, filename: Any, content_type: str | None) -> str:
        suffix = Path(str(filename or "")).suffix.lower()
        normalized_content_type = cls._normalized_content_type(content_type)
        if suffix == ".pdf" or normalized_content_type == "application/pdf":
            return "pdf"
        if (
            suffix in PAGEINDEX_DOCUMENT_SUFFIXES
            or normalized_content_type in PAGEINDEX_DOCUMENT_CONTENT_TYPES
        ):
            return "markdown"
        return "unsupported"

    @staticmethod
    def _normalized_content_type(content_type: str | None) -> str:
        return str(content_type or "").split(";", 1)[0].strip().lower()

    @property
    def pageindex_client_workspace(self) -> Path:
        return self.workspace / "artifacts" / "pageindex_client"

    def _pageindex_client(self) -> PageIndexClient:
        from ..client import PageIndexClient

        workspace = self.pageindex_client_workspace
        workspace.mkdir(parents=True, exist_ok=True)
        metadata_index = workspace / "_meta.json"
        if not metadata_index.exists():
            metadata_index.write_text("{}\n", encoding="utf-8")
        return PageIndexClient(workspace=str(workspace))

    def _pageindex_client_doc_for_entry(self, entry: Any) -> tuple[PageIndexClient, str | None]:
        client = self._pageindex_client()
        if not entry.pageindex_doc_id:
            return client, None
        if entry.pageindex_doc_id not in client.documents:
            return client, None
        return client, entry.pageindex_doc_id

    def _registration_pageindex_pointer(
        self,
        *,
        storage_uri: str,
        title: str,
        content_type: str,
    ) -> tuple[str | None, str, dict[str, Any] | None]:
        if self._content_format(title, content_type) not in {"pdf", "markdown"}:
            return None, "not_built", None
        client = self._pageindex_client()
        local_path = self._canonical_storage_uri_path(storage_uri)
        cached_doc_id = self._find_cached_pageindex_doc_id(client, local_path)
        if cached_doc_id:
            return cached_doc_id, "built", None
        if local_path is None:
            return None, "failed", self._pageindex_tree_failure_record(
                source="PageIndexFileSystem.registration",
                error_type="UnresolvableStorageUri",
                message=(
                    "storage_uri must resolve to a local file path for "
                    "PDF/Markdown registration."
                ),
            )
        try:
            doc_id = client.index(local_path)
            return doc_id, "built", None
        except Exception as exc:
            return None, "failed", self._pageindex_tree_failure_record(
                source="PageIndexClient.index",
                error_type=exc.__class__.__name__,
                message=str(exc) or exc.__class__.__name__,
            )

    @staticmethod
    def _pageindex_tree_failure_record(
        *,
        source: str,
        error_type: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "owner": "pageindex",
            "source": source,
            "error_type": error_type,
            "message": message,
        }

    def _find_cached_pageindex_doc_id(
        self,
        client: PageIndexClient,
        local_path: str | None,
    ) -> str | None:
        if local_path is None:
            return None
        for doc_id, doc in client.documents.items():
            if self._canonical_path(doc.get("path")) == local_path:
                return doc_id
        return None

    def _canonical_storage_uri_path(self, storage_uri: str) -> str | None:
        parsed = urlparse(storage_uri)
        if parsed.scheme == "file":
            return self._canonical_path(unquote(parsed.path))
        if storage_uri and not parsed.scheme:
            return self._canonical_path(storage_uri)
        return None

    @staticmethod
    def _title_from_storage_uri(storage_uri: str) -> str:
        parsed = urlparse(str(storage_uri or ""))
        path = unquote(parsed.path) if parsed.scheme else str(storage_uri or "")
        return Path(path).name

    @classmethod
    def _infer_content_type(cls, *, title: str, storage_uri: str) -> str:
        for filename in (title, cls._title_from_storage_uri(storage_uri)):
            suffix = Path(str(filename or "")).suffix.lower()
            if suffix == ".pdf":
                return "application/pdf"
            if suffix in PAGEINDEX_DOCUMENT_SUFFIXES:
                return "text/markdown"
        return "application/octet-stream"

    @staticmethod
    def _canonical_path(path: Any) -> str | None:
        if not path:
            return None
        return str(Path(os.path.expanduser(str(path))).resolve(strict=False))

    @staticmethod
    def _client_json(payload: str) -> Any:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {"error": f"Invalid PageIndexClient JSON response: {payload}"}

    @classmethod
    def _resolve_add_target(
        cls,
        virtual_target: Union[str, Path],
        *,
        physical_basename: str,
        physical_suffix: str,
    ) -> tuple[str, str, str]:
        raw_target = str(virtual_target).strip()
        if not raw_target:
            raise ValueError("pifs add target is required")
        normalized = normalize_path(raw_target)
        posix_target = PurePosixPath(normalized)
        raw_looks_like_folder = raw_target.replace("\\", "/").endswith("/")
        target_suffix = posix_target.suffix.lower()
        if raw_looks_like_folder or target_suffix not in ADD_FILE_CONTENT_TYPES:
            folder_path = normalized
            filename = physical_basename
        else:
            if target_suffix != physical_suffix:
                raise ValueError(
                    "pifs add target file extension must match the physical file extension"
                )
            folder_path = normalize_path(str(posix_target.parent))
            filename = posix_target.name
        cls._validate_add_filename(filename)
        virtual_path = cls._join_virtual_file_path(folder_path, filename)
        return folder_path, filename, virtual_path

    @staticmethod
    def _validate_add_filename(filename: str) -> None:
        if not filename or filename in {".", ".."}:
            raise ValueError("pifs add target filename is required")
        if "/" in filename or "\\" in filename:
            raise ValueError("pifs add target filename must be a basename")

    @staticmethod
    def _join_virtual_file_path(folder_path: str, filename: str) -> str:
        folder_path = normalize_path(folder_path)
        if folder_path == "/":
            return f"/{filename}"
        return f"{folder_path}/{filename}"

    def _add_file_content(self, path: Path, content_type: str) -> str:
        if self._content_format(path.name, content_type) == "markdown":
            return path.read_text(encoding="utf-8")
        return ""

    def _require_add_pageindex_ready(self, record: dict[str, Any]) -> None:
        if self._content_format(record["title"], record["content_type"]) not in {
            "pdf",
            "markdown",
        }:
            return
        if record.get("pageindex_tree_status") == "built" and record.get("pageindex_doc_id"):
            return
        message = self._pageindex_tree_failure_message(record.get("metadata_status")) or (
            "PageIndex tree was not built"
        )
        raise RuntimeError(f"pifs add failed to build PageIndex tree: {message}")

    def _require_summary_projection_ready(
        self,
        record: dict[str, Any],
        *,
        operation: str,
    ) -> None:
        summary_projection = (record.get("metadata_status") or {}).get("summary_projection")
        if not summary_projection or not summary_projection.get("requested"):
            raise RuntimeError(
                f"PIFS {operation} requires a requested summary projection index"
            )
        if summary_projection.get("status") != "ready":
            detail = summary_projection.get("error") or summary_projection.get("status")
            raise RuntimeError(
                f"PIFS {operation} failed to build summary projection index: {detail}"
            )

    def _prepare_file_record(
        self,
        file: dict[str, Any],
        *,
        artifact_baselines: dict[Path, bytes | None] | None = None,
    ) -> dict[str, Any]:
        storage_uri = file["storage_uri"]
        metadata = self._validated_register_metadata(file.get("metadata"))
        external_id = file.get("external_id")
        content = file.get("content") or ""
        folder_path = normalize_path(file.get("folder_path") or "/")
        title = str(
            file.get("title")
            or metadata.get("title")
            or self._title_from_storage_uri(storage_uri)
            or external_id
            or ""
        ).strip()
        if not title:
            raise ValueError("file title is required")
        content_type = file.get("content_type") or self._infer_content_type(
            title=title,
            storage_uri=storage_uri,
        )
        if self._content_format(title, content_type) not in {"pdf", "markdown"}:
            raise ValueError("PIFS registration supports PageIndex-backed PDF/Markdown files only")
        file_ref = make_file_ref(
            str(external_id or self._join_virtual_file_path(folder_path, title).strip("/"))
        )
        if artifact_baselines is not None:
            self._capture_registration_artifact_baselines(
                file_ref,
                file,
                artifact_baselines,
            )
        (
            pageindex_doc_id,
            pageindex_tree_status,
            pageindex_tree_failure,
        ) = self._registration_pageindex_pointer(
            storage_uri=storage_uri,
            title=title,
            content_type=content_type,
        )
        if pageindex_tree_status != "built" or not pageindex_doc_id:
            message = self._pageindex_tree_failure_message(
                {"pageindex_tree": pageindex_tree_failure}
            ) or "PageIndex tree was not built"
            raise RuntimeError(f"PIFS registration requires PageIndex extraction: {message}")
        pageindex_summary = self._pageindex_doc_description(pageindex_doc_id)
        if not pageindex_summary:
            raise RuntimeError("PIFS registration requires PageIndex doc_description")
        metadata["summary"] = pageindex_summary
        artifact_content = self._registration_text_artifact_content(
            title=title,
            content_type=content_type,
            pageindex_doc_id=pageindex_doc_id,
            pageindex_tree_status=pageindex_tree_status,
            fallback_content=content,
        )
        source_type = file.get("source_type")
        metadata_status = self._metadata_status_state(metadata=metadata)
        self._attach_pageindex_tree_failure(metadata_status, pageindex_tree_failure)
        indexed_metadata = SQLiteFileSystemStore.indexed_metadata_values(metadata)
        text_artifact_path = file.get("text_artifact_path")
        owns_text_artifact = text_artifact_path is None
        if text_artifact_path is None:
            text_artifact_path = self.store.write_text_artifact(file_ref, artifact_content)
        raw_artifact_path = file.get("raw_artifact_path")
        owns_raw_artifact = False
        if raw_artifact_path is None and file.get("write_raw_artifact", True):
            raw_artifact_path = self.store.raw_dir / f"{file_ref}.json"
            owns_raw_artifact = True
        return {
            "file_ref": file_ref,
            "external_id": external_id,
            "storage_uri": storage_uri,
            "title": title,
            "descriptor": title,
            "content_type": content_type,
            "source_type": source_type,
            "fingerprint": fingerprint(artifact_content),
            "text_artifact_path": str(text_artifact_path),
            "raw_artifact_path": str(raw_artifact_path) if raw_artifact_path is not None else None,
            "pageindex_doc_id": pageindex_doc_id,
            "pageindex_tree_status": pageindex_tree_status,
            "metadata": metadata,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
            "metadata_status": metadata_status,
            "metadata_status_json": json.dumps(metadata_status, ensure_ascii=False),
            "indexed_metadata": indexed_metadata,
            "folder_path": folder_path,
            "_pifs_owned_text_artifact": owns_text_artifact,
            "_pifs_owned_raw_artifact": owns_raw_artifact,
        }

    def _registration_text_artifact_content(
        self,
        *,
        title: str,
        content_type: str,
        pageindex_doc_id: str | None,
        pageindex_tree_status: str,
        fallback_content: str,
    ) -> str:
        if self._content_format(title, content_type) not in {"pdf", "markdown"}:
            return fallback_content
        if pageindex_tree_status != "built" or not pageindex_doc_id:
            return fallback_content
        return self._pageindex_extracted_text(pageindex_doc_id) or fallback_content

    def _pageindex_extracted_text(self, doc_id: str) -> str:
        client = self._pageindex_client()
        if doc_id not in client.documents:
            return ""
        client._ensure_doc_loaded(doc_id)
        doc = client.documents.get(doc_id) or {}
        return self._pageindex_pages_text(doc.get("pages"))

    def _pageindex_doc_description(self, doc_id: str) -> str:
        client = self._pageindex_client()
        if doc_id not in client.documents:
            return ""
        client._ensure_doc_loaded(doc_id)
        doc = client.documents.get(doc_id) or {}
        return str(doc.get("doc_description") or "").strip()

    @staticmethod
    def _pageindex_pages_text(pages: Any) -> str:
        if not isinstance(pages, list):
            return ""
        parts: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            content = str(page.get("content") or "").strip()
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    @staticmethod
    def _raw_artifact_payload(
        *,
        folder_path: str,
        metadata: dict[str, Any],
        metadata_status: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "folder_path": folder_path,
            "metadata": metadata,
            "metadata_status": metadata_status,
        }

    def _sync_owned_raw_artifact(self, record: dict[str, Any]) -> None:
        raw_artifact_path = record.get("raw_artifact_path")
        if self._managed_raw_artifact_path(
            str(record["file_ref"]),
            raw_artifact_path,
        ) is None:
            return
        record["raw_artifact_path"] = str(
            self.store.write_raw_artifact(
                record["file_ref"],
                self._raw_artifact_payload(
                    folder_path=record["folder_path"],
                    metadata=record["metadata"],
                    metadata_status=record["metadata_status"],
                ),
            )
        )

    def _record_from_file_entry(self, entry: Any) -> dict[str, Any]:
        metadata_status = self._metadata_status_state(metadata=entry.metadata)
        self._attach_pageindex_tree_failure(
            metadata_status,
            entry.metadata_status.get("pageindex_tree"),
        )
        return {
            "file_ref": entry.file_ref,
            "external_id": entry.external_id,
            "storage_uri": entry.storage_uri,
            "title": entry.title,
            "descriptor": entry.descriptor,
            "content_type": entry.content_type,
            "source_type": entry.source_type,
            "fingerprint": entry.fingerprint,
            "text_artifact_path": entry.text_artifact_path,
            "raw_artifact_path": entry.raw_artifact_path,
            "pageindex_doc_id": entry.pageindex_doc_id,
            "pageindex_tree_status": entry.pageindex_tree_status,
            "metadata": dict(entry.metadata),
            "metadata_json": json.dumps(entry.metadata, ensure_ascii=False),
            "metadata_status": metadata_status,
            "metadata_status_json": json.dumps(metadata_status, ensure_ascii=False),
            "indexed_metadata": SQLiteFileSystemStore.indexed_metadata_values(entry.metadata),
            "folder_path": entry.folder_path,
        }

    def _complete_summary_projection_index(self, record: dict[str, Any]) -> bool:
        metadata_status = record["metadata_status"]
        summary_index = metadata_status.get("summary_projection")
        if not summary_index or not summary_index.get("requested"):
            return False
        summary = str(record.get("metadata", {}).get("summary") or "").strip()
        if not summary:
            return False
        if self.summary_projection is None:
            raise RuntimeError("PIFS Summary Projection is not open")
        try:
            result = self.summary_projection.upsert_summary(record)
        except Exception as exc:
            summary_index["status"] = "failed"
            summary_index["error"] = str(exc)
            self._refresh_record_metadata_status(record)
            raise RuntimeError(
                f"PIFS failed to build summary projection index: {exc}"
            ) from exc
        summary_index.clear()
        summary_index.update({"requested": True, **result})
        self._refresh_record_metadata_status(record)
        return True

    @staticmethod
    def _unlink_artifact(path: Any) -> None:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            return

    def _cleanup_failed_register_artifacts(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            if record.get("_pifs_owned_text_artifact"):
                self._unlink_artifact(record["text_artifact_path"])
            if record.get("_pifs_owned_raw_artifact") and record.get("raw_artifact_path"):
                self._unlink_artifact(record["raw_artifact_path"])

    def _capture_registration_artifact_baselines(
        self,
        file_ref: str,
        file: dict[str, Any],
        baselines: dict[Path, bytes | None],
    ) -> None:
        paths = []
        if file.get("text_artifact_path") is None:
            paths.append(self.store.text_dir / f"{file_ref}.txt")
        raw_artifact_path = file.get("raw_artifact_path")
        if raw_artifact_path is None and file.get("write_raw_artifact", True):
            raw_artifact_path = self.store.raw_dir / f"{file_ref}.json"
        managed_raw_path = self._managed_raw_artifact_path(file_ref, raw_artifact_path)
        if managed_raw_path is not None:
            paths.append(managed_raw_path)
        try:
            existing = self.store.get_file(file_ref)
        except KeyError:
            existing = None
        if existing is not None and existing.pageindex_doc_id:
            paths.extend(
                [
                    self.pageindex_client_workspace / "_meta.json",
                    self.pageindex_client_workspace
                    / f"{existing.pageindex_doc_id}.json",
                ]
            )
        for path in paths:
            if path not in baselines:
                baselines[path] = path.read_bytes() if path.is_file() else None

    def _managed_raw_artifact_path(
        self,
        file_ref: str,
        raw_artifact_path: Any,
    ) -> Path | None:
        if not raw_artifact_path:
            return None
        default_path = self.store.raw_dir / f"{file_ref}.json"
        if Path(raw_artifact_path).expanduser().resolve(strict=False) != (
            default_path.resolve(strict=False)
        ):
            return None
        return default_path

    def _restore_registration_artifact_baselines(
        self,
        baselines: dict[Path, bytes | None],
    ) -> None:
        for path, content in baselines.items():
            if content is None:
                self._unlink_artifact(path)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def _cleanup_catalog_record(self, file_ref: str) -> None:
        try:
            self.store.delete_file(file_ref)
        except Exception:
            return

    def _capture_existing_registration_rows(
        self,
        snapshot: _RegistrationRollbackSnapshot,
        projection: Any | None,
    ) -> None:
        file_refs = sorted(
            {
                str(record.get("file_ref") or "")
                for record in snapshot.records
                if str(record.get("file_ref") or "")
            }
        )
        with self.store.connect() as connection:
            for file_ref in file_refs:
                file_row = connection.execute(
                    "SELECT * FROM files WHERE file_ref = ?",
                    (file_ref,),
                ).fetchone()
                if file_row is None:
                    continue
                snapshot.catalog_rows[file_ref] = dict(file_row)
                snapshot.membership_rows[file_ref] = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM file_folders WHERE file_ref = ? ORDER BY folder_id",
                        (file_ref,),
                    )
                ]
                snapshot.metadata_value_rows[file_ref] = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM metadata_values WHERE file_ref = ? "
                        "ORDER BY field_id, value_text, created_at",
                        (file_ref,),
                    )
                ]
        if not snapshot.catalog_rows:
            return
        if projection is None:
            raise RuntimeError(
                "PIFS registration cannot snapshot existing files without a Summary Projection"
            )
        with projection.index.connect(read_only=True) as connection:
            for file_ref in sorted(snapshot.catalog_rows):
                doc = connection.execute(
                    "SELECT * FROM semantic_index_docs WHERE file_ref = ?",
                    (file_ref,),
                ).fetchone()
                vector = (
                    None
                    if doc is None
                    else connection.execute(
                        "SELECT rowid, source_type, embedding "
                        "FROM semantic_index_vec WHERE rowid = ?",
                        (doc["rowid"],),
                    ).fetchone()
                )
                if doc is None or vector is None:
                    raise RuntimeError(
                        "PIFS registration found an incomplete existing Summary Projection; "
                        "migrate this workspace before retrying."
                    )
                vector_row = dict(vector)
                vector_row["embedding"] = bytes(vector_row["embedding"])
                snapshot.projection_rows[file_ref] = (dict(doc), vector_row)

    def _restore_existing_registration_catalog(
        self,
        snapshot: _RegistrationRollbackSnapshot,
    ) -> None:
        if not snapshot.catalog_rows:
            return
        with self.store.connect() as connection:
            for file_ref in sorted(snapshot.catalog_rows):
                connection.execute(
                    "DELETE FROM metadata_values WHERE file_ref = ?", (file_ref,)
                )
                connection.execute(
                    "DELETE FROM file_folders WHERE file_ref = ?", (file_ref,)
                )
                connection.execute("DELETE FROM files WHERE file_ref = ?", (file_ref,))
                self._insert_registration_snapshot_row(
                    connection,
                    "files",
                    snapshot.catalog_rows[file_ref],
                )
                for row in snapshot.membership_rows[file_ref]:
                    self._insert_registration_snapshot_row(
                        connection,
                        "file_folders",
                        row,
                    )
                for row in snapshot.metadata_value_rows[file_ref]:
                    self._insert_registration_snapshot_row(
                        connection,
                        "metadata_values",
                        row,
                    )

    def _restore_existing_registration_projection(
        self,
        snapshot: _RegistrationRollbackSnapshot,
    ) -> None:
        if not snapshot.projection_rows:
            return
        if self.summary_projection is None:
            raise RuntimeError(
                "PIFS registration cannot restore an unopened Summary Projection"
            )
        with self.summary_projection.index.connect() as connection:
            for file_ref in sorted(snapshot.projection_rows):
                current = connection.execute(
                    "SELECT rowid FROM semantic_index_docs WHERE file_ref = ?",
                    (file_ref,),
                ).fetchall()
                for row in current:
                    connection.execute(
                        "DELETE FROM semantic_index_vec WHERE rowid = ?",
                        (row["rowid"],),
                    )
                connection.execute(
                    "DELETE FROM semantic_index_docs WHERE file_ref = ?",
                    (file_ref,),
                )
                doc, vector = snapshot.projection_rows[file_ref]
                self._insert_registration_snapshot_row(
                    connection,
                    "semantic_index_docs",
                    doc,
                )
                self._insert_registration_snapshot_row(
                    connection,
                    "semantic_index_vec",
                    vector,
                )

    @staticmethod
    def _insert_registration_snapshot_row(
        connection: Any,
        table: str,
        row: dict[str, Any],
    ) -> None:
        columns = list(row)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders})",
            [row[column] for column in columns],
        )

    def _cleanup_summary_projection_records(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        projection = self.summary_projection
        if projection is None:
            return
        for record in records:
            file_ref = str(record.get("file_ref") or "")
            if not file_ref:
                continue
            try:
                projection.delete_summary(file_ref)
            except Exception:
                continue

    def _cleanup_summary_projection_cache(
        self,
        keys: set[_EmbeddingCacheKey],
    ) -> None:
        if not keys or self.summary_projection is None:
            return
        try:
            self.summary_projection.delete_cache_keys(keys)
        except Exception:
            return

    def _cleanup_created_folders(self, folder_paths: list[str]) -> None:
        for folder_path in reversed(folder_paths):
            try:
                self.store.delete_empty_folder(folder_path)
            except Exception:
                continue

    def _cleanup_new_metadata_fields(self, names: set[str]) -> None:
        for name in sorted(names):
            try:
                self.store.delete_metadata_field_if_unreferenced(name)
            except Exception:
                continue

    def _pageindex_cache_doc_ids(self) -> set[str]:
        workspace = self.pageindex_client_workspace
        doc_ids = {path.stem for path in workspace.glob("*.json") if path.name != "_meta.json"}
        meta_path = workspace / "_meta.json"
        if not meta_path.exists():
            return doc_ids
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return doc_ids
        if isinstance(payload, dict):
            doc_ids.update(str(doc_id) for doc_id in payload)
        return doc_ids

    def _cleanup_pageindex_cache(
        self,
        records: list[dict[str, Any]],
        preexisting_doc_ids: set[str],
    ) -> None:
        doc_ids = sorted(self._pageindex_cache_doc_ids() - preexisting_doc_ids)
        for record in records:
            doc_id = str(record.get("pageindex_doc_id") or "").strip()
            if doc_id and doc_id not in preexisting_doc_ids:
                doc_ids.append(doc_id)
        doc_ids = sorted(set(doc_ids))
        if not doc_ids:
            return
        workspace = self.pageindex_client_workspace
        for doc_id in doc_ids:
            try:
                (workspace / f"{doc_id}.json").unlink()
            except FileNotFoundError:
                pass
            except Exception:
                continue
        meta_path = workspace / "_meta.json"
        if not meta_path.exists():
            return
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        changed = False
        for doc_id in doc_ids:
            if doc_id in payload:
                payload.pop(doc_id, None)
                changed = True
        if not changed:
            return
        try:
            meta_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _refresh_record_metadata_status(
        self,
        record: dict[str, Any],
        *,
        explicit_status: str | None = None,
    ) -> None:
        metadata_status = record["metadata_status"]
        metadata_status["status"] = explicit_status or metadata_status.get("status") or "generated"
        self._refresh_summary_projection_status(metadata_status, record["metadata"])
        record["metadata_json"] = json.dumps(record["metadata"], ensure_ascii=False)
        record["metadata_status_json"] = json.dumps(metadata_status, ensure_ascii=False)
        record["indexed_metadata"] = SQLiteFileSystemStore.indexed_metadata_values(record["metadata"])

    @classmethod
    def _structural_unavailable(
        cls,
        mode: str,
        entry: Any,
        *,
        message: str,
        pages: str | None = None,
    ) -> dict[str, Any]:
        pageindex_tree_error = cls._pageindex_tree_failure_message(entry.metadata_status)
        if pageindex_tree_error and entry.pageindex_tree_status == "failed":
            message = f"PageIndex tree build failed: {pageindex_tree_error}"
        result = {
            "mode": mode,
            "file_ref": entry.file_ref,
            "external_id": entry.external_id,
            "status": entry.pageindex_tree_status,
            "available": False,
            "message": message,
        }
        if pageindex_tree_error:
            result["pageindex_tree_error"] = pageindex_tree_error
        if pages is not None:
            result["pages"] = pages
        return result

    @staticmethod
    def _attach_pageindex_tree_failure(
        metadata_status: dict[str, Any],
        pageindex_tree_failure: Any,
    ) -> None:
        if isinstance(pageindex_tree_failure, dict) and pageindex_tree_failure:
            metadata_status["pageindex_tree"] = dict(pageindex_tree_failure)

    @staticmethod
    def _pageindex_tree_failure_message(metadata_status: Any) -> str | None:
        if not isinstance(metadata_status, dict):
            return None
        pageindex_tree = metadata_status.get("pageindex_tree")
        if not isinstance(pageindex_tree, dict):
            return None
        if pageindex_tree.get("status") != "failed":
            return None
        message = str(pageindex_tree.get("message") or "").strip()
        error_type = str(pageindex_tree.get("error_type") or "").strip()
        if error_type and message:
            return f"{error_type}: {message}"
        return message or error_type or None

    def _resolve_target(self, target: str) -> str:
        try:
            return self.store.resolve_file_ref(target)
        except KeyError:
            if not str(target).strip().startswith("/"):
                raise
        normalized = normalize_path(target)
        try:
            return self._resolve_scope_file_locator(normalized)
        except KeyError:
            pass
        try:
            scope = self.resolve_query_scope(normalized)
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError(self._scope_file_required_message(scope.path))
        raise KeyError(f"Unknown file target: {target}")

    def _resolve_scope_file_locator(self, target: str) -> str:
        prefix, _, leaf = target.rstrip("/").rpartition("/")
        if not leaf:
            raise KeyError(f"Unknown file target: {target}")
        leaf = unquote(leaf)
        scope_path = prefix or "/"
        scope = self.resolve_query_scope(scope_path)
        if scope.metadata_axis is not None:
            raise ValueError(self._scope_file_required_message(target))
        # ponytail: scans the scoped leaf list; add an indexed leaf lookup if huge scoped collisions matter.
        matches = [
            item
            for item in self.scope_files(scope, limit=self.scope_file_count(scope) + 1)
            if item["locator_leaf"] == leaf
        ]
        if not matches:
            raise KeyError(f"Unknown file target: {target}")
        if len(matches) > 1:
            raise KeyError(
                f"Ambiguous file target: {target}. Use tree {scope.path} or "
                f'browse {scope.path} "<query>" to copy a disambiguated file locator.'
            )
        return matches[0]["file_ref"]

    @staticmethod
    def _scope_file_required_message(path: str) -> str:
        return (
            f"{path} is a scope, not a file locator; use tree {path} or "
            f'browse {path} "<query>" to select a file leaf.'
        )

    @staticmethod
    def _metadata_filter_payload(metadata_filter: Any) -> str:
        if isinstance(metadata_filter, str):
            return metadata_filter
        return json.dumps(
            metadata_filter,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _stable_file_locator(
        self,
        file_ref: str,
        entry: Any,
        *,
        folder_path: str | None = None,
    ) -> str:
        folder_path = normalize_path(folder_path or getattr(entry, "folder_path", None) or "/")
        title = str(
            self.store.membership_display_name(file_ref, folder_path)
            or getattr(entry, "title", "")
            or ""
        ).strip()
        if not title:
            raise RuntimeError(f"browse cannot build a virtual path for {file_ref}: missing title")
        return self.scope_file_locator(
            self.resolve_query_scope(folder_path),
            file_ref,
            title,
        )

    @staticmethod
    def _scope_file_locator(scope: PIFSQueryScope, leaf: Any) -> str:
        return PageIndexFileSystem._join_virtual_file_path(
            scope.path,
            PageIndexFileSystem.encode_scope_segment(str(leaf).strip("/")),
        )

    @staticmethod
    def _validated_register_metadata(metadata: Any) -> dict[str, Any]:
        if metadata is None:
            validated = {}
        elif not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        else:
            validated = dict(metadata)
        try:
            json.dumps(validated, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        if "summary" in validated:
            raise ValueError("summary is managed by PageIndex doc_description")
        return validated

    def _register_custom_metadata_fields(self, records: list[dict[str, Any]]) -> None:
        fields = {
            name: {}
            for name in self._custom_metadata_field_names(records)
            if not self.store.metadata_field_exists(name)
        }
        if fields:
            self.metadata.register_schema({"fields": fields}, source="user")

    def _custom_metadata_field_names(self, records: list[dict[str, Any]]) -> set[str]:
        fields = set()
        for record in records:
            for name in SQLiteFileSystemStore.indexed_metadata_values(
                record.get("metadata", {})
            ):
                if self.metadata.FIELD_RE.match(str(name)):
                    fields.add(str(name))
        return fields

    @staticmethod
    def _metadata_status_state(*, metadata: dict[str, Any]) -> dict[str, Any]:
        state = {
            "status": "generated",
            "summary_projection": {
                "requested": True,
                "status": "not_indexed",
                "owner": "pifs",
                "source": "index",
            },
        }
        PageIndexFileSystem._refresh_summary_projection_status(state, metadata)
        return state

    @staticmethod
    def _refresh_summary_projection_status(
        metadata_status: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        summary_index = metadata_status.get("summary_projection")
        if not summary_index or not summary_index.get("requested"):
            return
        if "summary" not in metadata:
            return
        if summary_index.get("status", "not_indexed") == "not_indexed":
            summary_index["status"] = "pending_index"

    def _folder_exists(self, path: str) -> bool:
        try:
            self.store.folder_info(path)
            return True
        except KeyError:
            return False

    @staticmethod
    def _query_text(query: Union[str, list[str], None]) -> str:
        if query is None:
            return ""
        if isinstance(query, list):
            return " ".join(str(item) for item in query)
        return str(query)

    @staticmethod
    def _preferred_folder_path(
        folder_paths: list[str],
        scope_path: Optional[str],
        fallback: str,
    ) -> str:
        if scope_path:
            scoped = [
                path
                for path in folder_paths
                if path == scope_path or path.startswith(f"{scope_path.rstrip('/')}/")
            ]
            if scoped:
                return sorted(scoped, key=lambda item: (len(item), item))[0]
        non_root = [path for path in folder_paths if path != "/"]
        if non_root:
            return sorted(non_root, key=lambda item: (len(item), item))[0]
        return fallback
