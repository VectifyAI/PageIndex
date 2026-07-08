from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import quote, unquote, urlparse

from .metadata import MetadataQueryEngine
from .store import (
    SQLiteFileSystemStore,
    fingerprint,
    make_file_ref,
    metadata_text,
    normalize_path,
)
from .types import OpenResult, PIFSQueryScope, SearchResult

if TYPE_CHECKING:
    from ..client import PageIndexClient

PROJECTION_INDEX_STATUSES = {
    "not_indexed",
    "pending_index",
    "generated",
    "ready",
    "failed",
}

DEFAULT_EMBEDDING_DIMENSIONS = 1024
SEMANTIC_RETRIEVAL_CHANNELS = ("summary",)
SEMANTIC_PROJECTION_INDEX_NAMES = {
    "summary": "summary",
}
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
        summary_projection_embedding_provider: str = "openai",
        summary_projection_embedding_model: str = "text-embedding-3-small",
        summary_projection_embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        summary_projection_embedding_timeout: float = 60,
        summary_projection_embedding_api_key: str | None = None,
        summary_projection_embedding_base_url: str | None = None,
    ):
        self.workspace = Path(workspace).expanduser()
        self.store = SQLiteFileSystemStore(self.workspace)
        self.metadata = MetadataQueryEngine(self.store)
        self.semantic_retrieval_backend: Any | None = None
        self.summary_projection_indexer: Any | None = None
        self.summary_projection_index_dir = (
            Path(summary_projection_index_dir).expanduser()
            if summary_projection_index_dir is not None
            else self.workspace / "artifacts" / "projection_indexes"
        )
        self.summary_projection_embedding_provider = summary_projection_embedding_provider
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

    def register(self, **kwargs: Any) -> str:
        self._ensure_register_completion_defaults()
        return self.register_file(**kwargs)

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
        self._ensure_add_completion_defaults()
        add_created_folder_paths = self._add_created_folder_paths(folder_path)
        file_ref = make_file_ref(virtual_path.strip("/"))
        uploads_dir = self.workspace / "artifacts" / "uploads"
        final_dir = uploads_dir / file_ref
        final_path = final_dir / filename
        final_dir_created = False
        catalog_inserted = False
        records: list[dict[str, Any]] = []
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
                self._require_add_summary_projection_ready(record)
                self._sync_owned_raw_artifact(record)
                self._ensure_add_semantic_retrieval_ready()
            except Exception:
                if catalog_inserted:
                    self._cleanup_add_catalog_record(file_ref)
                self._cleanup_add_summary_projection(records)
                self._cleanup_failed_register_artifacts(records)
                self._cleanup_add_pageindex_cache(records, preexisting_pageindex_doc_ids)
                self._cleanup_add_created_folders(add_created_folder_paths)
                if final_dir_created:
                    shutil.rmtree(final_dir, ignore_errors=True)
                raise

        info = self.store.file_info(file_ref)
        info["path"] = virtual_path
        return info

    def register_files(self, files: list[dict[str, Any]]) -> list[str]:
        records = [self._prepare_file_record(file) for file in files]
        preexisting_file_refs = self._existing_file_refs(records)
        new_records = [
            record for record in records if record["file_ref"] not in preexisting_file_refs
        ]
        try:
            self._register_custom_metadata_fields(records)
            self.store.insert_files(records)
            for record in records:
                try:
                    if self._complete_summary_projection_index(record):
                        self.store.update_file_metadata_status(
                            record["file_ref"],
                            metadata=record["metadata"],
                            metadata_status=record["metadata_status"],
                        )
                    self._sync_owned_raw_artifact(record)
                except KeyError:
                    continue
        except Exception:
            self._cleanup_add_summary_projection(new_records)
            for record in new_records:
                self._cleanup_add_catalog_record(str(record["file_ref"]))
            self._cleanup_failed_register_artifacts(records)
            raise
        return [record["file_ref"] for record in records]

    def _ensure_register_completion_defaults(self) -> None:
        if self.summary_projection_indexer is None:
            from .semantic_projection import SummaryProjectionIndexer

            self.summary_projection_indexer = SummaryProjectionIndexer.from_provider(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
                embedding_api_key=self.summary_projection_embedding_api_key,
                embedding_base_url=self.summary_projection_embedding_base_url,
            )
        if self.semantic_retrieval_backend is None:
            self.configure_semantic_projection_retrieval(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
                embedding_api_key=self.summary_projection_embedding_api_key,
                embedding_base_url=self.summary_projection_embedding_base_url,
            )

    def _ensure_add_completion_defaults(self) -> None:
        if self.summary_projection_indexer is None:
            from .semantic_projection import SummaryProjectionIndexer

            self.summary_projection_indexer = SummaryProjectionIndexer.from_provider(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
                embedding_api_key=self.summary_projection_embedding_api_key,
                embedding_base_url=self.summary_projection_embedding_base_url,
            )

    def _ensure_add_semantic_retrieval_ready(self) -> None:
        indexer = self.summary_projection_indexer
        if indexer is None:
            raise RuntimeError("pifs add requires a summary projection indexer")
        from .semantic_projection import SemanticProjectionSearchBackend

        index_dir = Path(getattr(indexer, "index_dir", self.summary_projection_index_dir))
        embedder = getattr(indexer, "embedder", None)
        if embedder is None:
            self.configure_semantic_projection_retrieval(
                index_dir,
                embedding_provider=str(
                    getattr(
                        indexer,
                        "embedding_provider",
                        self.summary_projection_embedding_provider,
                    )
                ),
                embedding_model=str(
                    getattr(indexer, "embedding_model", self.summary_projection_embedding_model)
                ),
                embedding_dimensions=int(
                    getattr(
                        indexer,
                        "embedding_dimensions",
                        self.summary_projection_embedding_dimensions,
                    )
                ),
                embedding_timeout=self.summary_projection_embedding_timeout,
                embedding_api_key=self.summary_projection_embedding_api_key,
                embedding_base_url=self.summary_projection_embedding_base_url,
            )
        else:
            embedding_cache = getattr(indexer, "embedding_cache", None)
            self.semantic_retrieval_backend = SemanticProjectionSearchBackend(
                index_dir,
                embedder=embedder,
                embedding_provider=str(
                    getattr(
                        indexer,
                        "embedding_provider",
                        self.summary_projection_embedding_provider,
                    )
                ),
                embedding_model=str(
                    getattr(indexer, "embedding_model", self.summary_projection_embedding_model)
                ),
                embedding_dimensions=int(
                    getattr(
                        indexer,
                        "embedding_dimensions",
                        self.summary_projection_embedding_dimensions,
                    )
                ),
                embedding_cache_path=getattr(embedding_cache, "db_path", None),
            )
        if "summary" not in self.semantic_retrieval_channels():
            raise RuntimeError("pifs add failed to configure summary semantic retrieval")

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

    def configure_existing_projection_retrieval(self) -> bool:
        """Attach semantic retrieval to already-built projection indexes.

        Register-time generation owns building the index files. Opening an
        existing workspace should still expose semantic retrieval when the
        configured embedding dimensions match the existing index.
        """
        if self.semantic_retrieval_backend is not None:
            return bool(self.semantic_retrieval_channels())
        index_config = self._existing_projection_index_config()
        if index_config is None:
            return False
        existing_dimension = int(index_config.get("dimension") or 0)
        if existing_dimension != self.summary_projection_embedding_dimensions:
            raise RuntimeError(
                "summary projection index dimension mismatch: "
                f"{index_config.get('db_path') or self.summary_projection_index_dir} "
                f"was built with dimension {existing_dimension}, but configured "
                "summary_projection_embedding_dimensions is "
                f"{self.summary_projection_embedding_dimensions}. Rebuild the "
                "projection index or use a matching embedding configuration."
            )
        self.configure_semantic_projection_retrieval(
            self.summary_projection_index_dir,
            embedding_provider=self.summary_projection_embedding_provider,
            embedding_model=self.summary_projection_embedding_model,
            embedding_dimensions=self.summary_projection_embedding_dimensions,
            embedding_timeout=self.summary_projection_embedding_timeout,
            embedding_api_key=self.summary_projection_embedding_api_key,
            embedding_base_url=self.summary_projection_embedding_base_url,
        )
        return bool(self.semantic_retrieval_channels())

    def _existing_projection_index_config(self) -> dict[str, Any] | None:
        for channel in SEMANTIC_RETRIEVAL_CHANNELS:
            index_name = SEMANTIC_PROJECTION_INDEX_NAMES.get(channel)
            if not index_name:
                continue
            index_path = self.summary_projection_index_dir / f"{index_name}.sqlite"
            if not index_path.exists():
                continue
            from .semantic_index import SQLiteVecSemanticIndex

            try:
                info = SQLiteVecSemanticIndex(index_path).info()
            except Exception:
                continue
            if int(info.get("document_count") or 0) <= 0:
                continue
            metadata = dict(info.get("metadata") or {})
            if metadata.get("channel") and metadata.get("channel") != channel:
                continue
            return info
        return None

    def browse(
        self,
        path: str = "/",
        recursive: bool = False,
        limit: int = 100,
        max_depth: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        return self.store.list_folder(
            path,
            recursive=recursive,
            limit=limit,
            max_depth=max_depth,
        )

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
                    "inspect the physical folder first, then append @field/value buckets. "
                    "Use the path returned by tree for values containing '/'."
                )
            field = unquote(segment[1:])
            self.metadata.validate_field_name(field)
            if not self.store.metadata_field_exists(field):
                raise ValueError("Unknown metadata axis; run tree <scope> to inspect available @field axes.")
            if field in metadata_filter:
                raise ValueError(
                    "A metadata field can appear only once in a scope path; "
                    "choose one value or use browse --where for advanced predicates."
                )
            if index + 1 >= len(remainder):
                return PIFSQueryScope(
                    path=normalized,
                    folder_path=folder_path,
                    metadata_filter=metadata_filter,
                    metadata_axis=field,
                )
            value_segment = remainder[index + 1]
            if value_segment.startswith("@"):
                raise ValueError(
                    "Metadata axis paths require @field/value; run tree <scope>/@field to inspect values."
                )
            metadata_filter[field] = unquote(value_segment)
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
                    "name": locator_leaf,
                    "type": "file",
                    "file_ref": row["file_ref"],
                    "document_id": row["document_id"],
                    "title": leaf,
                    "metadata": row["metadata"],
                }
            )
        files = sorted(files, key=lambda item: (str(item["name"]).lower(), item["path"], item["file_ref"]))
        return files[:limit]

    def scope_file_locator(self, scope: PIFSQueryScope, file_ref: str, leaf: str) -> str:
        leaf_items = self._scope_file_leaf_items(scope, limit=self.scope_file_count(scope) + 1)
        return self._scope_file_locator(
            scope,
            self._scope_locator_leaf_by_file_ref(leaf_items).get(file_ref, leaf),
        )

    def _scope_file_leaf_items(self, scope: PIFSQueryScope, *, limit: int) -> list[tuple[dict[str, Any], str]]:
        recursive = bool(scope.metadata_filter)
        rows = self.store.search_files(
            None,
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

    @staticmethod
    def _disambiguated_scope_leaf(leaf: str, file_ref: str, leaf_counts: dict[str, int]) -> str:
        if leaf_counts.get(leaf, 0) <= 1:
            return leaf
        return f"{leaf}~{file_ref}"

    @classmethod
    def _scope_locator_leaf_by_file_ref(cls, leaf_items: list[tuple[dict[str, Any], str]]) -> dict[str, str]:
        leaf_counts = cls._scope_leaf_counts(leaf_items)
        used: set[str] = set()
        locator_leaf_by_file_ref: dict[str, str] = {}
        for row, leaf in sorted(leaf_items, key=lambda item: (item[1].lower(), item[1], item[0]["file_ref"])):
            base = cls._disambiguated_scope_leaf(leaf, row["file_ref"], leaf_counts)
            locator_leaf = base
            suffix = 2
            while locator_leaf in used:
                locator_leaf = f"{base}~{suffix}"
                suffix += 1
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
        return quote(str(segment), safe="")

    def browse_semantic_files(
        self,
        path: str,
        query: str,
        *,
        retrieval_query: str | None = None,
        recursive: bool = False,
        space: str = "summary",
        page: int = 1,
        page_size: int = 10,
        metadata_filter: Optional[dict[str, Any] | str] = None,
    ) -> dict[str, Any]:
        path = normalize_path(path)
        query_scope = self.resolve_query_scope(path)
        self.store.folder_info(query_scope.folder_path)
        query_text = self._query_text(retrieval_query or query).strip()
        if not query_text:
            raise ValueError("browse requires a query")
        if page < 1:
            raise ValueError("browse --page must be at least 1")
        if page_size < 1:
            raise ValueError("browse page_size must be at least 1")
        if space not in SEMANTIC_RETRIEVAL_CHANNELS:
            raise ValueError(
                "Unsupported browse --space: "
                f"{space}. Supported spaces: {', '.join(SEMANTIC_RETRIEVAL_CHANNELS)}"
            )
        available_spaces = self.semantic_retrieval_channels()
        if space not in available_spaces:
            available = ", ".join(available_spaces) if available_spaces else "none"
            raise ValueError(
                f"browse --space {space} is not available; available spaces: {available}"
            )
        search_channel = getattr(self.semantic_retrieval_backend, "search_channel", None)
        if search_channel is None:
            available = ", ".join(available_spaces) if available_spaces else "none"
            raise ValueError(
                f"browse --space {space} is not available; available spaces: {available}"
            )
        parsed_filter = self.merge_scope_filter(query_scope, metadata_filter)
        effective_recursive = recursive or bool(query_scope.metadata_filter)
        scope = {"folder_path": query_scope.folder_path, "recursive": effective_recursive}
        scope_file_refs = self.store.file_refs_for_scope(
            scope=scope,
            metadata_filter=parsed_filter,
        )
        offset = (page - 1) * page_size
        needed = offset + page_size + 1
        semantic_filters = {"file_ref": scope_file_refs}
        candidates = (
            search_channel(
                space,
                query_text,
                limit=needed,
                filters=semantic_filters,
            )
            if scope_file_refs
            else []
        )
        scope_file_ref_set = set(scope_file_refs)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                file_ref = self.store.resolve_file_ref(candidate.document_id)
            except KeyError:
                continue
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
                    "similarity": self._semantic_candidate_similarity(candidate),
                    "score": self._semantic_candidate_score(candidate),
                    "path": stable_path,
                    "file_ref": file_ref,
                    "document_id": entry.external_id,
                    "external_id": entry.external_id,
                    "title": display_title,
                    "original_title": entry.title,
                    "folder_path": folder_path,
                    "folder_paths": folder_paths,
                    "summary": str((entry.metadata or {}).get("summary") or ""),
                    "snippet": str(getattr(candidate, "snippet", "") or entry.descriptor),
                    "metadata": entry.metadata,
                    "metadata_status": entry.metadata_status,
                    "sources": list(getattr(candidate, "sources", []) or []),
                }
            )
            if len(rows) >= needed:
                break
        page_rows = rows[offset : offset + page_size]
        payload = {
            "mode": "files",
            "retrieval": f"{space}_vector",
            "query": query,
            "scope": query_scope.path,
            "recursive": effective_recursive,
            "space": space,
            "available_spaces": list(available_spaces),
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

    def find_folders(
        self,
        path: str = "/",
        metadata_filter: Optional[dict[str, Any] | str] = None,
        limit: int = 100,
        max_depth: int | None = None,
        include_self: bool = False,
    ) -> list[dict[str, Any]]:
        parsed_filter = self.metadata.parse_filter(metadata_filter)
        return self.store.find_folders(
            path,
            metadata_filter=parsed_filter,
            limit=limit,
            max_depth=max_depth,
            include_self=include_self,
        )

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
        return self.store.file_info(file_ref)

    def search(
        self,
        query: Union[str, list[str], None] = None,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any] | str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        parsed_filter = self.metadata.parse_filter(metadata_filter)
        rows = self.store.search_files(
            query,
            scope=scope,
            metadata_filter=parsed_filter,
            limit=limit,
        )
        results = []
        scope_path = self._scope_folder_path(scope)
        for row in rows:
            folder_paths = [
                folder["path"]
                for folder in self.store.folder_memberships(row["file_ref"])
            ]
            folder_path = self._preferred_folder_path(folder_paths, scope_path, row["folder_path"])
            display_title = self.store.membership_display_name(row["file_ref"], folder_path) or row["title"]
            results.append(
                SearchResult(
                    file_ref=row["file_ref"],
                    external_id=row["external_id"],
                    title=display_title,
                    snippet=row["snippet"],
                    folder_path=folder_path,
                    folder_paths=folder_paths,
                    metadata=row["metadata"],
                    metadata_status=row["metadata_status"],
                    id=row["id"],
                    document_id=row["document_id"],
                    name=display_title,
                    description=row["description"],
                    status=row["status"],
                    pageNum=row["pageNum"],
                    createdAt=row["createdAt"],
                    folderId=row["folderId"],
                )
            )
        return results

    def configure_semantic_projection_retrieval(
        self,
        index_dir: Union[str, Path],
        *,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_timeout: float = 60,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        fetch_multiplier: int = 100,
    ) -> Any:
        from .semantic_projection import SemanticProjectionSearchBackend

        self.semantic_retrieval_backend = SemanticProjectionSearchBackend.from_provider(
            index_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_timeout=embedding_timeout,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            fetch_multiplier=fetch_multiplier,
        )
        return self.semantic_retrieval_backend

    @property
    def has_semantic_retrieval_backend(self) -> bool:
        return self.semantic_retrieval_backend is not None

    def semantic_retrieval_channels(self) -> tuple[str, ...]:
        backend = self.semantic_retrieval_backend
        if backend is None:
            return ()
        available_channels = getattr(backend, "available_channels", None)
        if callable(available_channels):
            raw_channels = available_channels()
        else:
            raw_channels = getattr(backend, "semantic_tool_channels", ())
        available = set(raw_channels or ())
        return tuple(channel for channel in SEMANTIC_RETRIEVAL_CHANNELS if channel in available)

    def has_semantic_channel(self, channel: str) -> bool:
        return channel in self.semantic_retrieval_channels()

    def retrieval_capabilities(self) -> dict[str, Any]:
        semantic_channels = ["summary"] if self.has_semantic_channel("summary") else []
        semantic_commands = ["browse"] if semantic_channels else []
        return {
            "lexical": {
                "grep_recursive": False,
                "grep_recursive_semantic_prefilter": False,
                "find_maxdepth": False,
            },
            "semantic": {
                "backend_configured": self.semantic_retrieval_backend is not None,
                "channels": semantic_channels,
                "commands": semantic_commands,
            },
        }

    def open(self, target: str, location: str = "all") -> OpenResult:
        file_ref = self._resolve_target(target)
        entry = self.store.get_file(file_ref)
        if self._file_format(entry) in {"pdf", "markdown", "pageindex"}:
            raise ValueError(
                "open() text artifact reads are not supported for PDF/Markdown PageIndex files; "
                "use pageindex_structure() or pageindex_pages()."
            )
        if str(location).strip().lower() in {"all", "full", "*"}:
            return self._open_all(file_ref)
        start, end = self._parse_line_range(location)
        return self._open_lines(file_ref, start, end)

    def cat_text_artifact(self, target: str, location: str = "all") -> OpenResult:
        file_ref = self._resolve_target(target)
        entry = self.store.get_file(file_ref)
        self._require_text_artifact_file(entry, "cat --all")
        if str(location).strip().lower() in {"all", "full", "*"}:
            return self._open_all(file_ref)
        start, end = self._parse_line_range(location)
        return self._open_lines(file_ref, start, end)

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

    def _stat(self, target: str) -> dict[str, Any]:
        file_ref = self._resolve_target(target)
        return self.store.file_info(file_ref)

    def _require_text_artifact_file(self, entry: Any, command: str) -> None:
        if self._file_format(entry) == "text":
            return
        raise ValueError(
            f"{command} is only supported for txt/text files; "
            f"got title={entry.title!r}, content_type={entry.content_type!r}. "
            "Use cat <path|file_ref|document_id> --structure, "
            "or cat <path|file_ref|document_id> --page for PDF/Markdown PageIndex files."
        )

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

        return PageIndexClient(workspace=str(self.pageindex_client_workspace))

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

    def _create_folder(self, path: str) -> str:
        return self.create_folder(path)

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

    def _require_add_summary_projection_ready(self, record: dict[str, Any]) -> None:
        summary_projection = (record.get("metadata_status") or {}).get("summary_projection")
        if not summary_projection or not summary_projection.get("requested"):
            raise RuntimeError("pifs add requires a requested summary projection index")
        if summary_projection.get("status") != "ready":
            detail = summary_projection.get("error") or summary_projection.get("status")
            raise RuntimeError(
                f"pifs add failed to build summary projection index: {detail}"
            )

    def _prepare_file_record(self, file: dict[str, Any]) -> dict[str, Any]:
        storage_uri = file["storage_uri"]
        metadata = file.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        self._validate_register_metadata(metadata)
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
        fts_content = file.get("fts_content", artifact_content)
        source_type = file.get("source_type")
        metadata_status = self._metadata_status_state(metadata=metadata)
        self._attach_pageindex_tree_failure(metadata_status, pageindex_tree_failure)
        indexed_metadata = SQLiteFileSystemStore.indexed_metadata_values(metadata)
        searchable_metadata = indexed_metadata
        text_artifact_path = file.get("text_artifact_path")
        owns_text_artifact = text_artifact_path is None
        if text_artifact_path is None:
            text_artifact_path = self.store.write_text_artifact(file_ref, artifact_content)
        raw_artifact_path = file.get("raw_artifact_path")
        owns_raw_artifact = False
        if raw_artifact_path is None and file.get("write_raw_artifact", True):
            raw_artifact_path = self.store.raw_dir / f"{file_ref}.json"
            owns_raw_artifact = True
        descriptor = self._build_descriptor(title, metadata)
        return {
            "file_ref": file_ref,
            "external_id": external_id,
            "storage_uri": storage_uri,
            "title": title,
            "descriptor": descriptor,
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
            "metadata_text": metadata_text(searchable_metadata),
            "folder_path": folder_path,
            "content": fts_content,
            "skip_fts": bool(file.get("skip_fts", False)),
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
        return self._pageindex_extracted_text(pageindex_doc_id)

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
        if not raw_artifact_path:
            return
        default_raw_artifact_path = self.store.raw_dir / f"{record['file_ref']}.json"
        if Path(raw_artifact_path).expanduser().resolve(strict=False) != (
            default_raw_artifact_path.resolve(strict=False)
        ):
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
        content = self.store.read_text(entry.file_ref)
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
            "metadata_text": metadata_text(
                SQLiteFileSystemStore.indexed_metadata_values(entry.metadata)
            ),
            "folder_path": entry.folder_path,
            "content": content,
            "skip_fts": False,
        }

    def _complete_summary_projection_index(self, record: dict[str, Any]) -> bool:
        metadata_status = record["metadata_status"]
        summary_index = metadata_status.get("summary_projection")
        if not summary_index or not summary_index.get("requested"):
            return False
        summary = str(record.get("metadata", {}).get("summary") or "").strip()
        if not summary:
            return False
        if self.summary_projection_indexer is None:
            self._refresh_record_metadata_status(record)
            return True
        try:
            result = self.summary_projection_indexer.upsert_summary(record)
        except Exception as exc:
            summary_index["status"] = "failed"
            summary_index["error"] = str(exc)
            self._refresh_record_metadata_status(record)
            return True
        summary_index.clear()
        summary_index.update({"requested": True, **result})
        if summary_index.get("status") != "ready":
            summary_index["status"] = "ready"
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

    def _cleanup_add_catalog_record(self, file_ref: str) -> None:
        try:
            self.store.delete_file(file_ref)
        except Exception:
            return

    def _existing_file_refs(self, records: list[dict[str, Any]]) -> set[str]:
        existing: set[str] = set()
        for record in records:
            file_ref = str(record.get("file_ref") or "")
            if not file_ref:
                continue
            try:
                self.store.get_file(file_ref)
            except KeyError:
                continue
            existing.add(file_ref)
        return existing

    def _cleanup_add_summary_projection(self, records: list[dict[str, Any]]) -> None:
        indexer = self.summary_projection_indexer
        if indexer is None:
            return
        delete_summary = getattr(indexer, "delete_summary", None)
        for record in records:
            file_ref = str(record.get("file_ref") or "")
            if not file_ref:
                continue
            try:
                if callable(delete_summary):
                    delete_summary(file_ref)
                    continue
                index = getattr(indexer, "index", None)
                delete_file_refs = getattr(index, "delete_file_refs", None)
                if callable(delete_file_refs):
                    delete_file_refs([file_ref])
            except Exception:
                continue

    def _cleanup_add_created_folders(self, folder_paths: list[str]) -> None:
        for folder_path in reversed(folder_paths):
            try:
                self.store.delete_empty_folder(folder_path)
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

    def _cleanup_add_pageindex_cache(
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
        record["metadata_text"] = metadata_text(record["indexed_metadata"])

    def _open_lines(self, file_ref: str, start: int, end: int) -> OpenResult:
        entry = self.store.get_file(file_ref)
        lines = self.store.read_text(file_ref).splitlines()
        start = max(1, start)
        end = min(max(start, end), len(lines))
        text = "\n".join(lines[start - 1:end])
        return OpenResult(
            file_ref=file_ref,
            start_line=start,
            end_line=end,
            text=text,
            external_id=entry.external_id,
            folder_path=entry.folder_path,
        )

    def _open_all(self, file_ref: str) -> OpenResult:
        entry = self.store.get_file(file_ref)
        text = self.store.read_text(file_ref)
        line_count = len(text.splitlines())
        return OpenResult(
            file_ref=file_ref,
            start_line=1,
            end_line=line_count,
            text=text,
            external_id=entry.external_id,
            folder_path=entry.folder_path,
        )

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
            if item["name"] == leaf
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
    def _semantic_candidate_score(candidate: Any) -> float | None:
        try:
            return float(getattr(candidate, "score"))
        except (AttributeError, TypeError, ValueError):
            return None

    @classmethod
    def _semantic_candidate_similarity(cls, candidate: Any) -> float:
        distances: list[float] = []
        for source in getattr(candidate, "sources", []) or []:
            if not isinstance(source, dict) or source.get("distance") is None:
                continue
            try:
                distances.append(float(source["distance"]))
            except (TypeError, ValueError):
                continue
        if distances:
            distance = max(min(distances), 0.0)
            return round(max(0.0, min(1.0, 1.0 / (1.0 + distance))), 4)
        score = cls._semantic_candidate_score(candidate)
        if score is None:
            return 0.0
        return round(max(0.0, min(1.0, score)), 4)

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
        target = self._join_virtual_file_path(folder_path, title.strip("/"))
        try:
            resolved_file_ref = self.store.resolve_file_ref(target)
        except KeyError as exc:
            raise RuntimeError(
                f"browse produced an unresolved virtual path for {file_ref}: {target}"
            ) from exc
        if resolved_file_ref != file_ref:
            raise RuntimeError(
                "browse produced a non-idempotent virtual path: "
                f"{target} resolved to {resolved_file_ref}, expected {file_ref}"
            )
        return target

    @staticmethod
    def _scope_file_locator(scope: PIFSQueryScope, leaf: Any) -> str:
        return PageIndexFileSystem._join_virtual_file_path(
            scope.path,
            PageIndexFileSystem.encode_scope_segment(str(leaf).strip("/")),
        )

    @staticmethod
    def _build_descriptor(title: str, metadata: dict[str, Any]) -> str:
        source = metadata.get("source_type") or metadata.get("repo") or metadata.get("channel")
        return f"{title} ({source})" if source else title

    @staticmethod
    def _validate_register_metadata(metadata: dict[str, Any]) -> None:
        if "summary" in metadata:
            raise ValueError("summary is managed by PageIndex doc_description")

    def _register_custom_metadata_fields(self, records: list[dict[str, Any]]) -> None:
        fields = {}
        for record in records:
            for name in SQLiteFileSystemStore.indexed_metadata_values(
                record.get("metadata", {})
            ):
                if self.metadata.FIELD_RE.match(str(name)):
                    fields[str(name)] = {}
        if fields:
            self.metadata.register_schema({"fields": fields}, source="user")

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

    @staticmethod
    def _scope_folder_path(scope: Optional[dict[str, Any]]) -> Optional[str]:
        if not scope:
            return None
        path = scope.get("folder_path") or scope.get("path")
        return normalize_path(path) if path else None

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

    @staticmethod
    def _parse_line_range(location: str) -> tuple[int, int]:
        value = str(location).strip()
        if "-" in value:
            left, right = value.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(value)
        if start < 1 or end < start:
            raise ValueError(f"Invalid line range: {location}")
        return start, end
