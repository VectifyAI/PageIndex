from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import unquote, urlparse

from .metadata import MetadataQueryEngine
from .metadata_generation import (
    MetadataGenerationBackend,
    MetadataGenerationError,
    MetadataGenerationInput,
    MetadataGenerationResult,
    MetadataGenerator,
)
from .store import (
    SQLiteFileSystemStore,
    fingerprint,
    make_file_ref,
    metadata_text,
    normalize_path,
)
from .types import OpenResult, SearchResult

if TYPE_CHECKING:
    from ..client import PageIndexClient
    from .semantic_projection import SummaryProjectionIndexer

DEFAULT_METADATA_GENERATION_FIELDS = {
    "summary": True,
    "doc_type": True,
    "domain": True,
    "topic": True,
    "entity": False,
    "relation": False,
}

DEFAULT_METADATA_FIELD_TYPES = {
    "summary": "string",
    "doc_type": "string",
    "domain": "string",
    "topic": "string",
    "entity": "string",
    "relation": "string",
}

METADATA_STATUSES = {
    "skipped",
    "pending_submit",
    "pending_generate",
    "generated",
    "failed",
}

PROJECTION_INDEX_STATUSES = {
    "not_indexed",
    "pending_index",
    "generated",
    "ready",
    "failed",
}

DEFAULT_EMBEDDING_DIMENSIONS = 1024
SEMANTIC_RETRIEVAL_CHANNELS = ("summary", "entity", "relation")
SEMANTIC_PROJECTION_INDEX_NAMES = {
    "summary": "summary_only_vector",
    "entity": "entity_vectors",
    "relation": "relation_vectors",
}
PAGEINDEX_DOCUMENT_SUFFIXES = {".pdf", ".md", ".markdown"}
PAGEINDEX_DOCUMENT_CONTENT_TYPES = {
    "application/pdf",
    "text/markdown",
    "text/x-markdown",
    "application/markdown",
}
TEXT_ARTIFACT_SUFFIXES = {".txt", ".text"}
TEXT_ARTIFACT_CONTENT_TYPES = {"text/plain"}
ADD_FILE_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
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
        semantic_retrieval_backend: Any | None = None,
        metadata_generator: MetadataGenerationBackend | None = None,
        metadata_provider: str = "openai",
        metadata_model: str | None = None,
        metadata_base_url: str | None = None,
        metadata_max_text_chars: int = 24000,
        summary_projection_indexer: SummaryProjectionIndexer | None = None,
        summary_projection_index: bool = True,
        summary_projection_index_dir: Union[str, Path, None] = None,
        summary_projection_embedding_provider: str = "openai",
        summary_projection_embedding_model: str = "text-embedding-3-small",
        summary_projection_embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        summary_projection_embedding_timeout: float = 60,
    ):
        self.workspace = Path(workspace).expanduser()
        self.store = SQLiteFileSystemStore(self.workspace)
        self.metadata = MetadataQueryEngine(self.store)
        self.semantic_retrieval_backend = semantic_retrieval_backend
        self.metadata_generator = metadata_generator
        self.metadata_provider = metadata_provider
        self.metadata_model = metadata_model
        self.metadata_base_url = metadata_base_url
        self.metadata_max_text_chars = metadata_max_text_chars
        self.summary_projection_indexer = summary_projection_indexer
        self.summary_projection_index = summary_projection_index
        self.summary_projection_index_dir = (
            Path(summary_projection_index_dir).expanduser()
            if summary_projection_index_dir is not None
            else self.workspace / "artifacts" / "projection_indexes"
        )
        self.summary_projection_embedding_provider = summary_projection_embedding_provider
        self.summary_projection_embedding_model = summary_projection_embedding_model
        self.summary_projection_embedding_dimensions = summary_projection_embedding_dimensions
        self.summary_projection_embedding_timeout = summary_projection_embedding_timeout

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
        metadata_policy: Optional[dict[str, Any]] = None,
        metadata_status: Optional[str] = None,
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
                    "metadata_policy": metadata_policy,
                    "metadata_status": metadata_status,
                }
            ]
        )[0]

    def register(self, **kwargs: Any) -> str:
        if not self._register_uses_deferred_metadata(kwargs.get("metadata_policy")):
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
        if not self.summary_projection_index:
            raise RuntimeError("pifs add requires the summary projection index")

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
                        "metadata_policy": self._add_metadata_policy(),
                    }
                )
                records = [record]
                self._require_add_pageindex_ready(record)
                self._generate_register_metadata(record)
                self._require_add_metadata_ready(record)
                self._register_generation_policy_schema(records)
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
            for record in records:
                self._generate_register_metadata(record)
            self._register_generation_policy_schema(records)
            self.store.insert_files(records)
            for record in records:
                if self._complete_summary_projection_index(record):
                    self.store.update_file_metadata_status(
                        record["file_ref"],
                        metadata=record["metadata"],
                        metadata_status=record["metadata_status"],
                    )
                self._sync_owned_raw_artifact(record)
        except Exception:
            self._cleanup_add_summary_projection(new_records)
            for record in new_records:
                self._cleanup_add_catalog_record(str(record["file_ref"]))
            self._cleanup_failed_register_artifacts(records)
            raise
        return [record["file_ref"] for record in records]

    def batch_generate(self, *, limit: int | None = None) -> dict[str, Any]:
        if self.metadata_generator is None:
            raise MetadataGenerationError(
                "metadata_generator is required to generate pending PIFS metadata"
            )
        rows = self.store.list_pending_metadata_status(limit=limit)
        generated = 0
        failed = 0
        file_refs: list[str] = []
        for row in rows:
            record = self._record_from_file_entry(row)
            self._generate_register_metadata(record, force=True)
            self._complete_summary_projection_index(record)
            self._register_generation_policy_schema([record])
            self.store.update_file_metadata_status(
                record["file_ref"],
                metadata=record["metadata"],
                metadata_status=record["metadata_status"],
            )
            self._sync_owned_raw_artifact(record)
            file_refs.append(record["file_ref"])
            if record["metadata_status"]["status"] == "failed":
                failed += 1
            else:
                generated += 1
        return {
            "processed": len(rows),
            "generated": generated,
            "failed": failed,
            "file_refs": file_refs,
        }

    def _ensure_register_completion_defaults(self) -> None:
        if self.metadata_generator is None:
            self.metadata_generator = MetadataGenerator(
                provider=self.metadata_provider,
                model=self.metadata_model,
                base_url=self.metadata_base_url,
                max_text_chars=self.metadata_max_text_chars,
        )
        if self.summary_projection_index and self.summary_projection_indexer is None:
            from .semantic_projection import SummaryProjectionIndexer

            self.summary_projection_indexer = SummaryProjectionIndexer.from_provider(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
            )
        if self.summary_projection_index and self.semantic_retrieval_backend is None:
            self.configure_semantic_projection_retrieval(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
            )

    def _ensure_add_completion_defaults(self) -> None:
        if self.metadata_generator is None:
            self.metadata_generator = MetadataGenerator(
                provider=self.metadata_provider,
                model=self.metadata_model,
                base_url=self.metadata_base_url,
                max_text_chars=self.metadata_max_text_chars,
        )
        if self.summary_projection_index and self.summary_projection_indexer is None:
            from .semantic_projection import SummaryProjectionIndexer

            self.summary_projection_indexer = SummaryProjectionIndexer.from_provider(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
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

    @staticmethod
    def _register_uses_deferred_metadata(policy: Any) -> bool:
        if not isinstance(policy, dict):
            return False
        return bool(policy.get("batch")) or policy.get("mode") == "batch"

    @classmethod
    def default_metadata_policy(cls) -> dict[str, Any]:
        return {
            "fields": dict(DEFAULT_METADATA_GENERATION_FIELDS),
            "projection_indexes": {"summary": True},
            "batch": False,
        }

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
        self.store.folder_info(path)
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
        parsed_filter = self.metadata.parse_filter(metadata_filter)
        scope = {"folder_path": path, "recursive": recursive}
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
                path,
                entry.folder_path,
            )
            display_title = self.store.membership_display_name(file_ref, folder_path) or entry.title
            try:
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
            "scope": path,
            "recursive": recursive,
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
        fetch_multiplier: int = 100,
    ) -> Any:
        from .semantic_projection import SemanticProjectionSearchBackend

        self.semantic_retrieval_backend = SemanticProjectionSearchBackend.from_provider(
            index_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_timeout=embedding_timeout,
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
        semantic_channels = self.semantic_retrieval_channels()
        semantic_commands = ["browse"] if semantic_channels else []
        return {
            "lexical": {
                "grep_recursive": True,
                "grep_recursive_semantic_prefilter": False,
                "grep_recursive_guard": "bounded broad-folder notice",
                "find_maxdepth": True,
            },
            "semantic": {
                "backend_configured": self.semantic_retrieval_backend is not None,
                "channels": list(semantic_channels),
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
            "Use cat <path|file_ref|document_id> --all for txt/text files."
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
        if suffix in TEXT_ARTIFACT_SUFFIXES:
            return "text"
        if normalized_content_type in TEXT_ARTIFACT_CONTENT_TYPES:
            return "text"
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
            if suffix in TEXT_ARTIFACT_SUFFIXES:
                return "text/plain"
        return "text/plain"

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

    def _metadata_schema(self) -> dict[str, Any]:
        return self.metadata.export_schema()

    def _register_metadata_schema(self, schema: dict[str, Any]) -> None:
        self.metadata.register_schema(schema)

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

    @staticmethod
    def _add_metadata_policy() -> dict[str, Any]:
        return {
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
                "entity": False,
                "relation": False,
            },
            "projection_indexes": {"summary": True},
            "batch": False,
        }

    def _add_file_content(self, path: Path, content_type: str) -> str:
        if self._content_format(path.name, content_type) in {"markdown", "text"}:
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

    @staticmethod
    def _require_add_metadata_ready(record: dict[str, Any]) -> None:
        metadata = record.get("metadata") or {}
        summary = str(metadata.get("summary") or "").strip()
        if not summary:
            raise MetadataGenerationError(
                "pifs add requires synchronous generated summary metadata"
            )
        status = record.get("metadata_status") or {}
        summary_status = (status.get("fields") or {}).get("summary") or {}
        if summary_status.get("status") != "generated":
            raise MetadataGenerationError(
                "pifs add requires generated summary metadata before registration"
            )
        if status.get("status") == "failed":
            raise MetadataGenerationError(
                "pifs add metadata generation failed before registration"
            )

    def _require_add_summary_projection_ready(self, record: dict[str, Any]) -> None:
        if not self.summary_projection_index:
            return
        summary_projection = (
            (record.get("metadata_status") or {})
            .get("projection_indexes", {})
            .get("summary")
        )
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
        artifact_content = self._registration_text_artifact_content(
            title=title,
            content_type=content_type,
            pageindex_doc_id=pageindex_doc_id,
            pageindex_tree_status=pageindex_tree_status,
            fallback_content=content,
        )
        fts_content = file.get("fts_content", artifact_content)
        source_type = file.get("source_type")
        metadata_policy = self._normalize_metadata_policy(
            file.get("metadata_policy"),
            metadata=metadata,
        )
        metadata_status = self._metadata_status_state(
            metadata_policy,
            metadata=metadata,
            status=file.get("metadata_status"),
        )
        self._attach_pageindex_tree_failure(metadata_status, pageindex_tree_failure)
        indexed_metadata = SQLiteFileSystemStore.indexed_metadata_values(metadata)
        searchable_metadata = dict(metadata)
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
        metadata_policy = self._normalize_metadata_policy(
            entry.metadata_status.get("policy", {}),
            metadata=entry.metadata,
        )
        metadata_status = self._metadata_status_state(
            metadata_policy,
            metadata=entry.metadata,
            status=entry.metadata_status.get("status"),
        )
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
            "metadata_text": metadata_text(entry.metadata),
            "folder_path": entry.folder_path,
            "content": content,
            "skip_fts": False,
        }

    def _generate_register_metadata(self, record: dict[str, Any], *, force: bool = False) -> None:
        status = record["metadata_status"]
        policy = status.get("policy", {})
        if self._metadata_policy_is_batch(policy) and not force:
            self._mark_requested_generation_status(record, "pending_submit")
            return
        fields = self._metadata_fields_to_generate(record)
        if not fields:
            return
        if self.metadata_generator is None:
            if self._metadata_policy_requires_sync(policy):
                raise MetadataGenerationError(
                    "metadata_generator is required for synchronous PIFS metadata generation; "
                    "set metadata_policy batch=true to defer"
                )
            return
        try:
            result = self.metadata_generator.generate(
                MetadataGenerationInput(
                    file_ref=record["file_ref"],
                    external_id=record.get("external_id"),
                    title=record["title"],
                    content_type=record["content_type"],
                    source_type=record.get("source_type"),
                    text=Path(record["text_artifact_path"]).read_text(encoding="utf-8"),
                    metadata=dict(record.get("metadata") or {}),
                    text_artifact_path=record.get("text_artifact_path"),
                ),
                fields=fields,
            )
            if isinstance(result, dict):
                result = MetadataGenerationResult(values=result)
        except Exception as exc:
            self._apply_metadata_status_failures(record, fields, str(exc))
            return
        failures = dict(result.failures)
        for field in fields:
            if field in result.values:
                record["metadata"][field] = result.values[field]
                status["fields"][field] = {
                    "requested": True,
                    "status": "generated",
                    "owner": "pifs",
                    "source": "llm",
                }
            else:
                failures.setdefault(field, "metadata generator did not return field")
        for field, reason in failures.items():
            status["fields"][field] = {
                "requested": True,
                "status": "failed",
                "owner": "pifs",
                "source": "llm",
                "error": str(reason),
            }
        self._refresh_record_metadata_status(record)

    def _complete_summary_projection_index(self, record: dict[str, Any]) -> bool:
        metadata_status = record["metadata_status"]
        summary_index = metadata_status.get("projection_indexes", {}).get("summary")
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

    @staticmethod
    def _metadata_policy_is_batch(policy: dict[str, Any]) -> bool:
        return bool(policy.get("batch")) or policy.get("mode") == "batch"

    @staticmethod
    def _metadata_policy_requires_sync(policy: dict[str, Any]) -> bool:
        return policy.get("batch") is False or policy.get("mode") == "sync"

    def _metadata_fields_to_generate(self, record: dict[str, Any]) -> list[str]:
        fields: list[str] = []
        for name, state in record["metadata_status"].get("fields", {}).items():
            if not state.get("requested"):
                continue
            if state.get("status") == "generated" and name in record["metadata"]:
                continue
            fields.append(name)
        return fields

    def _mark_requested_generation_status(self, record: dict[str, Any], status: str) -> None:
        for name, field in record["metadata_status"].get("fields", {}).items():
            if field.get("requested") and field.get("status") != "generated":
                record["metadata_status"]["fields"][name] = {
                    "requested": True,
                    "status": status,
                    "owner": "pifs",
                    "source": "llm",
                }
        self._refresh_record_metadata_status(record, explicit_status=status)

    def _apply_metadata_status_failures(
        self,
        record: dict[str, Any],
        fields: list[str],
        reason: str,
    ) -> None:
        for field in fields:
            record["metadata_status"]["fields"][field] = {
                "requested": True,
                "status": "failed",
                "owner": "pifs",
                "source": "llm",
                "error": reason,
            }
        self._refresh_record_metadata_status(record, explicit_status="failed")

    def _refresh_record_metadata_status(
        self,
        record: dict[str, Any],
        *,
        explicit_status: str | None = None,
    ) -> None:
        metadata_status = record["metadata_status"]
        statuses = [
            field.get("status")
            for field in metadata_status.get("fields", {}).values()
            if field.get("requested") and field.get("status")
        ]
        metadata_status["status"] = explicit_status or self._aggregate_metadata_status(statuses)
        self._refresh_projection_index_statuses(metadata_status, record["metadata"])
        record["metadata_json"] = json.dumps(record["metadata"], ensure_ascii=False)
        record["metadata_status_json"] = json.dumps(metadata_status, ensure_ascii=False)
        record["indexed_metadata"] = SQLiteFileSystemStore.indexed_metadata_values(record["metadata"])
        record["metadata_text"] = metadata_text(record["metadata"])

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
        return self.store.resolve_file_ref(target)

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
    def _build_descriptor(title: str, metadata: dict[str, Any]) -> str:
        source = metadata.get("source_type") or metadata.get("repo") or metadata.get("channel")
        return f"{title} ({source})" if source else title

    @staticmethod
    def _validate_register_metadata(metadata: dict[str, Any]) -> None:
        pifs_owned_fields = set(DEFAULT_METADATA_GENERATION_FIELDS)
        conflicts = sorted(pifs_owned_fields.intersection(metadata))
        if conflicts:
            raise ValueError(
                "metadata contains PIFS-owned generated field(s): "
                + ", ".join(conflicts)
                + "; configure metadata_policy instead of passing generated fields"
            )

    def _register_generation_policy_schema(self, records: list[dict[str, Any]]) -> None:
        pifs_fields: dict[str, dict[str, str]] = {}
        user_fields: dict[str, dict[str, str]] = {}
        for record in records:
            policy_fields = record["metadata_status"]["policy"]["fields"]
            generated_names = {str(name) for name, requested in policy_fields.items() if requested}
            for name, requested in policy_fields.items():
                if requested:
                    pifs_fields[name] = {
                        "type": DEFAULT_METADATA_FIELD_TYPES.get(
                            name,
                            self._infer_metadata_field_type(
                                record.get("metadata", {}).get(name)
                            ),
                        )
                    }
            for name, value in record.get("metadata", {}).items():
                if name in generated_names:
                    pifs_fields.setdefault(name, {"type": self._infer_metadata_field_type(value)})
                else:
                    user_fields.setdefault(name, {"type": self._infer_metadata_field_type(value)})
        if pifs_fields:
            self.metadata.register_schema({"fields": pifs_fields}, source="pifs")
        if user_fields:
            self.metadata.register_schema({"fields": user_fields}, source="user")

    @classmethod
    def _normalize_metadata_policy(
        cls,
        policy: Optional[dict[str, Any]],
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        fields = dict(DEFAULT_METADATA_GENERATION_FIELDS)
        field_statuses: dict[str, str] = {}
        projection_indexes: dict[str, bool] | None = None
        projection_index_statuses: dict[str, str] = {}
        mode = None
        batch = None
        top_level_status = None
        if policy is not None:
            if not isinstance(policy, dict):
                raise ValueError("metadata_policy must be a JSON object")
            raw_fields = policy.get("fields")
            if raw_fields is None:
                raw_fields = {
                    name: declaration
                    for name, declaration in policy.items()
                    if name not in {"batch", "mode", "status", "projection_indexes"}
                }
            if not isinstance(raw_fields, dict):
                raise ValueError("metadata_policy fields must be a JSON object")
            for name, declaration in raw_fields.items():
                name = str(name)
                if isinstance(declaration, bool):
                    fields[name] = declaration
                    continue
                if isinstance(declaration, dict):
                    fields[name] = bool(
                        declaration.get("enabled", declaration.get("requested", True))
                    )
                    field_status = declaration.get("status")
                    if field_status is not None:
                        cls._validate_metadata_status(str(field_status))
                        field_statuses[name] = str(field_status)
                    continue
                raise ValueError(f"Invalid metadata generation policy for field: {name}")
            mode = policy.get("mode")
            if "batch" in policy:
                batch = bool(policy["batch"])
            elif mode == "batch":
                batch = True
            top_level_status = policy.get("status")
            if top_level_status is not None:
                cls._validate_metadata_status(str(top_level_status))
            if "projection_indexes" in policy:
                projection_indexes, projection_index_statuses = (
                    cls._normalize_projection_index_policy(policy["projection_indexes"])
                )
        normalized: dict[str, Any] = {
            "fields": fields,
            "projection_indexes": (
                projection_indexes
                if projection_indexes is not None
                else {"summary": bool(fields.get("summary", False))}
            ),
        }
        if field_statuses:
            normalized["field_statuses"] = field_statuses
        if projection_index_statuses:
            normalized["projection_index_statuses"] = projection_index_statuses
        if mode:
            normalized["mode"] = str(mode)
        if batch is not None:
            normalized["batch"] = batch
        if top_level_status:
            normalized["status"] = str(top_level_status)
        return normalized

    @classmethod
    def _metadata_status_state(
        cls,
        policy: dict[str, Any],
        *,
        metadata: dict[str, Any],
        status: Optional[str],
    ) -> dict[str, Any]:
        explicit_status = status or policy.get("status")
        if explicit_status is not None:
            explicit_status = str(explicit_status)
            cls._validate_metadata_status(explicit_status)
        field_statuses = policy.get("field_statuses", {})
        fields: dict[str, dict[str, Any]] = {}
        for name, requested in policy["fields"].items():
            if not requested:
                fields[name] = {
                    "requested": False,
                    "status": "skipped",
                    "owner": "pifs",
                    "source": "llm",
                }
                continue
            field_status = field_statuses.get(name)
            if field_status is None:
                field_status = explicit_status
            if field_status is None:
                field_status = "generated" if name in metadata else "pending_generate"
            fields[name] = {
                "requested": True,
                "status": field_status,
                "owner": "pifs",
                "source": "llm",
            }

        requested_statuses = [
            item["status"]
            for item in fields.values()
            if item.get("requested") and item.get("status")
        ]
        aggregate_status = explicit_status or cls._aggregate_metadata_status(requested_statuses)
        policy_summary = {
            "fields": dict(policy["fields"]),
            "projection_indexes": dict(policy.get("projection_indexes", {})),
        }
        if "mode" in policy:
            policy_summary["mode"] = policy["mode"]
        if "batch" in policy:
            policy_summary["batch"] = policy["batch"]
        state = {
            "status": aggregate_status,
            "policy": policy_summary,
            "fields": fields,
            "projection_indexes": {},
        }
        projection_statuses = policy.get("projection_index_statuses", {})
        for name, requested in policy.get("projection_indexes", {}).items():
            if not requested:
                continue
            state["projection_indexes"][name] = {
                "requested": True,
                "status": projection_statuses.get(name, "not_indexed"),
                "owner": "pifs",
                "source": "index",
            }
        cls._refresh_projection_index_statuses(state, metadata)
        return state

    @staticmethod
    def _aggregate_metadata_status(statuses: list[str]) -> str:
        if not statuses:
            return "generated"
        for status in ("failed", "pending_submit", "pending_generate"):
            if status in statuses:
                return status
        return "generated"

    @staticmethod
    def _validate_metadata_status(status: str) -> None:
        if status not in METADATA_STATUSES:
            raise ValueError(f"Unsupported metadata status: {status}")

    @classmethod
    def _normalize_projection_index_policy(
        cls,
        projection_policy: Any,
    ) -> tuple[dict[str, bool], dict[str, str]]:
        if projection_policy is None:
            return {}, {}
        if not isinstance(projection_policy, dict):
            raise ValueError("metadata_policy projection_indexes must be a JSON object")
        projection_indexes: dict[str, bool] = {}
        projection_index_statuses: dict[str, str] = {}
        for name, declaration in projection_policy.items():
            name = str(name)
            if isinstance(declaration, bool):
                projection_indexes[name] = declaration
                continue
            if isinstance(declaration, dict):
                projection_indexes[name] = bool(
                    declaration.get("enabled", declaration.get("requested", True))
                )
                status = declaration.get("status")
                if status is not None:
                    status = str(status)
                    cls._validate_projection_index_status(status)
                    projection_index_statuses[name] = status
                continue
            raise ValueError(f"Invalid projection index policy for index: {name}")
        return projection_indexes, projection_index_statuses

    @staticmethod
    def _validate_projection_index_status(status: str) -> None:
        if status not in PROJECTION_INDEX_STATUSES:
            raise ValueError(f"Unsupported projection index status: {status}")

    @classmethod
    def _refresh_projection_index_statuses(
        cls,
        metadata_status: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        summary_index = metadata_status.get("projection_indexes", {}).get("summary")
        if not summary_index or not summary_index.get("requested"):
            return
        if "summary" not in metadata:
            return
        if summary_index.get("status", "not_indexed") == "not_indexed":
            summary_index["status"] = "pending_index"

    @staticmethod
    def _infer_metadata_field_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float)):
            return "number"
        return "string"

    @staticmethod
    def _scope_folder_path(scope: Optional[dict[str, Any]]) -> Optional[str]:
        if not scope:
            return None
        path = scope.get("folder_path") or scope.get("path")
        return normalize_path(path) if path else None

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
