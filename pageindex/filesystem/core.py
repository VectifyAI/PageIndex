from __future__ import annotations

import json
import os
from pathlib import Path
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
from .semantic_folder_policy import (
    SEMANTIC_FOLDER_BASE_FIELDS,
    SEMANTIC_FOLDER_ROOT,
    SEMANTIC_FOLDER_SYSTEM_FIELDS,
    canonical_semantic_folder_field_name,
    is_semantic_folder_forbidden_field,
    semantic_folder_allowed_extension_fields,
)
from .store import (
    SQLiteFileSystemStore,
    fingerprint,
    make_file_ref,
    metadata_text,
    normalize_path,
)
from .structural_read import (
    flatten_pageindex_structure_nodes,
    first_node_location,
    find_pageindex_node,
    strip_pageindex_text_fields,
)
from .types import OpenResult, SearchResult

if TYPE_CHECKING:
    from ..client import PageIndexClient
    from .projection_indexing import SummaryProjectionIndexer

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

SEMANTIC_RETRIEVAL_CHANNELS = ("summary", "entity", "relation")
SEMANTIC_GREP_CHANNELS = ("entity", "relation")
PAGEINDEX_DOCUMENT_SUFFIXES = {".pdf", ".md", ".markdown"}
PAGEINDEX_DOCUMENT_CONTENT_TYPES = {
    "application/pdf",
    "text/markdown",
    "text/x-markdown",
    "application/markdown",
}
TEXT_ARTIFACT_SUFFIXES = {".txt", ".text"}
TEXT_ARTIFACT_CONTENT_TYPES = {"text/plain"}


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
        summary_projection_embedding_dimensions: int = 256,
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
        source_path: str,
        folder_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        external_id: Optional[str] = None,
        title: Optional[str] = None,
        content: str = "",
        content_type: str = "text/plain",
        source_type: Optional[str] = None,
        metadata_policy: Optional[dict[str, Any]] = None,
        metadata_status: Optional[str] = None,
    ) -> str:
        return self.register_files(
            [
                {
                    "storage_uri": storage_uri,
                    "source_path": source_path,
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

    def register_files(self, files: list[dict[str, Any]]) -> list[str]:
        records = [self._prepare_file_record(file) for file in files]
        for record in records:
            self._generate_register_metadata(record)
            self._complete_summary_projection_index(record)
            self._sync_owned_raw_artifact(record)
        self._register_generation_policy_schema(records)
        self.store.insert_files(records)
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
            from .projection_indexing import SummaryProjectionIndexer

            self.summary_projection_indexer = SummaryProjectionIndexer.from_provider(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
            )
        if self.summary_projection_index and self.semantic_retrieval_backend is None:
            self.configure_hybrid_projection_retrieval(
                self.summary_projection_index_dir,
                embedding_provider=self.summary_projection_embedding_provider,
                embedding_model=self.summary_projection_embedding_model,
                embedding_dimensions=self.summary_projection_embedding_dimensions,
                embedding_timeout=self.summary_projection_embedding_timeout,
            )

    def configure_existing_projection_retrieval(self) -> bool:
        """Attach semantic retrieval to already-built projection indexes.

        Register-time generation owns building the index files. Opening an
        existing workspace should still expose the corresponding read commands,
        such as search-summary, without forcing a re-register step.
        """
        if self.semantic_retrieval_backend is not None:
            return bool(self.semantic_retrieval_channels())
        index_config = self._existing_projection_index_config()
        if index_config is None:
            return False
        metadata = dict(index_config.get("metadata") or {})
        embedding_provider = str(
            metadata.get("embedding_provider")
            or self.summary_projection_embedding_provider
        )
        embedding_model = str(
            metadata.get("embedding_model")
            or self.summary_projection_embedding_model
        )
        embedding_dimensions = int(
            metadata.get("embedding_dimensions")
            or index_config.get("dimension")
            or self.summary_projection_embedding_dimensions
        )
        self.configure_hybrid_projection_retrieval(
            self.summary_projection_index_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_timeout=self.summary_projection_embedding_timeout,
        )
        return bool(self.semantic_retrieval_channels())

    def _existing_projection_index_config(self) -> dict[str, Any] | None:
        from .hybrid_projection import INDEX_BY_CHANNEL
        from .semantic_index import SQLiteVecSemanticIndex

        for channel in SEMANTIC_RETRIEVAL_CHANNELS:
            index_name = INDEX_BY_CHANNEL.get(channel)
            if not index_name:
                continue
            index_path = self.summary_projection_index_dir / f"{index_name}.sqlite"
            if not index_path.exists():
                continue
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

    def folder_info(self, path: str = "/") -> dict[str, Any]:
        return self.store.folder_info(path)

    def find_folders(
        self,
        path: str = "/",
        metadata_filter: Optional[dict[str, Any] | str] = None,
        limit: int = 100,
        max_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        parsed_filter = self.metadata.parse_filter(metadata_filter)
        return self.store.find_folders(
            path,
            metadata_filter=parsed_filter,
            limit=limit,
            max_depth=max_depth,
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

    def apply_semantic_folder_projection(
        self,
        projection_plan: dict[str, Any],
        *,
        file_ref_by_document_id: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Attach registered files to a Semantic Folder Projection.

        Registration remains the explicit folder placement step. This method is
        the separate product API for adding derived `/semantic/...` memberships.
        """
        folders = list(projection_plan.get("folders") or [])
        memberships = list(projection_plan.get("memberships") or [])
        policy_raw = projection_plan.get("policy")
        policy = policy_raw if isinstance(policy_raw, dict) else {}
        allowed_extension_fields = semantic_folder_allowed_extension_fields(
            policy.get("allowed_extension_fields", [])
        )
        for folder in folders:
            self._validate_semantic_folder_projection_item(folder, allowed_extension_fields)
        for membership in memberships:
            self._validate_semantic_folder_projection_item(membership, allowed_extension_fields)

        for folder in folders:
            folder_metadata = folder.get("metadata")
            self.create_folder(
                self._validate_semantic_folder_projection_path(str(folder["path"])),
                kind=str(folder.get("kind") or "semantic_projection"),
                description=str(folder.get("description") or ""),
                metadata=folder_metadata if isinstance(folder_metadata, dict) else {},
            )

        items: list[dict[str, Any]] = []
        file_ref_by_document_id = file_ref_by_document_id or {}
        for membership in memberships:
            document_id = self._semantic_folder_projection_document_id(membership)
            file_ref = file_ref_by_document_id.get(document_id)
            if not file_ref:
                file_ref = self.store.resolve_file_ref(document_id)
            metadata = (
                dict(membership.get("folder_metadata"))
                if isinstance(membership.get("folder_metadata"), dict)
                else {}
            )
            metadata.update(
                {
                    "projection": "Semantic Folder Projection",
                    "field": membership.get("field", ""),
                    "value": membership.get("value", ""),
                    "mount_kind": membership.get(
                        "mount_kind",
                        "semantic_folder_projection",
                    ),
                }
            )
            items.append(
                {
                    "file_ref": file_ref,
                    "folder": self._validate_semantic_folder_projection_path(
                        str(membership["folder_path"])
                    ),
                    "metadata": metadata,
                }
            )
        self.attach_files_to_folders(items)
        return {
            "projection": "Semantic Folder Projection",
            "folders_applied": len(folders),
            "memberships_attached": len(items),
        }

    def search(
        self,
        query: Union[str, list[str], None] = None,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any] | str] = None,
        limit: int = 10,
        semantic: bool = True,
    ) -> list[SearchResult]:
        parsed_filter = self.metadata.parse_filter(metadata_filter)
        if semantic and self._should_use_semantic_retrieval(query, scope):
            semantic_results = self._semantic_search(
                query,
                scope=scope,
                metadata_filter=parsed_filter,
                limit=limit,
            )
            if semantic_results:
                return semantic_results
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
            results.append(
                SearchResult(
                    file_ref=row["file_ref"],
                    external_id=row["external_id"],
                    title=row["title"],
                    snippet=row["snippet"],
                    folder_path=folder_path,
                    folder_paths=folder_paths,
                    metadata=row["metadata"],
                    metadata_status=row["metadata_status"],
                    source_path=row["source_path"],
                    id=row["id"],
                    document_id=row["document_id"],
                    name=row["name"],
                    description=row["description"],
                    status=row["status"],
                    pageNum=row["pageNum"],
                    createdAt=row["createdAt"],
                    folderId=row["folderId"],
                )
            )
        return results

    def search_semantic_channel(
        self,
        channel: str,
        query: Union[str, list[str], None],
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any] | str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        parsed_filter = self.metadata.parse_filter(metadata_filter)
        if (
            self.semantic_retrieval_backend is None
            or not self.has_semantic_channel(channel)
            or not self._query_text(query)
        ):
            return []
        return self._semantic_search(
            query,
            scope=scope,
            metadata_filter=parsed_filter,
            limit=limit,
            channel=channel,
        )

    def configure_hybrid_projection_retrieval(
        self,
        index_dir: Union[str, Path],
        *,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 256,
        embedding_timeout: float = 60,
        per_channel_limit: int = 100,
        fetch_multiplier: int = 100,
    ) -> Any:
        from .hybrid_projection import HybridProjectionSearchBackend

        self.semantic_retrieval_backend = HybridProjectionSearchBackend.from_provider(
            index_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_timeout=embedding_timeout,
            per_channel_limit=per_channel_limit,
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
        semantic_commands = [f"search-{channel}" for channel in semantic_channels]
        semantic_grep_channels = [
            channel for channel in SEMANTIC_GREP_CHANNELS if channel in semantic_channels
        ]
        if semantic_grep_channels:
            semantic_commands.append("semantic-grep")
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
                "semantic_grep_channels": semantic_grep_channels,
            },
        }

    def find(
        self,
        target: str,
        patterns: Union[str, list[str]],
        limit: int = 20,
    ) -> list[OpenResult]:
        file_ref = self._resolve_target(target)
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        lowered_patterns = [pattern.lower() for pattern in patterns if pattern]
        if not lowered_patterns:
            return []
        text = self.store.read_text(file_ref)
        lines = text.splitlines()
        matches = []
        for i, line in enumerate(lines, 1):
            haystack = line.lower()
            if any(pattern in haystack for pattern in lowered_patterns):
                start = max(1, i - 1)
                end = min(len(lines), i + 1)
                matches.append(self._open_lines(file_ref, start, end))
                if len(matches) >= limit:
                    break
        return matches

    def open(self, target: str, location: str = "all") -> OpenResult:
        file_ref = self._resolve_target(target)
        entry = self.store.get_file(file_ref)
        if self._file_format(entry) in {"pdf", "markdown", "pageindex"}:
            raise ValueError(
                "open() text artifact reads are not supported for PDF/Markdown PageIndex files; "
                "use pageindex_structure(), pageindex_pages(), or pageindex_node()."
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
        *,
        offset: int = 0,
        limit: int = 25,
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
        node_rows = flatten_pageindex_structure_nodes(structure)
        offset = max(0, offset)
        limit = max(0, limit)
        window = node_rows[offset : offset + limit] if limit else []
        next_offset = offset + len(window)
        has_more = next_offset < len(node_rows)
        return {
            "mode": "structure",
            "file_ref": file_ref,
            "external_id": entry.external_id,
            "source_path": entry.source_path,
            "status": entry.pageindex_tree_status,
            "available": True,
            "pageindex_doc_id": doc_id,
            "structure": window,
            "structure_pagination": {
                "offset": offset,
                "limit": limit,
                "returned_nodes": len(window),
                "total_nodes": len(node_rows),
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
            },
        }

    def pageindex_node(self, target: str, node_id: str) -> dict[str, Any]:
        file_ref = self._resolve_target(target)
        entry = self.store.get_file(file_ref)
        self._require_pageindex_document_file(entry, "cat --node")
        client, doc_id = self._pageindex_client_doc_for_entry(entry)
        if doc_id is None:
            return self._structural_unavailable(
                "node",
                entry,
                node_id=node_id,
                message=(
                    "PageIndex structure is not cached for this file in the "
                    "PageIndexClient workspace."
                ),
            )
        client._ensure_doc_loaded(doc_id)
        doc = client.documents.get(doc_id, {})
        node = find_pageindex_node(doc.get("structure", []), node_id)
        if node is None:
            return self._structural_unavailable(
                "node",
                entry,
                node_id=node_id,
                message="PageIndex node was not found in the cached structure.",
            )
        text = str(node.get("text") or "")
        if not text:
            location = first_node_location(node)
            if location:
                content = self._client_json(client.get_page_content(doc_id, location))
                if isinstance(content, list):
                    text = "\n\n".join(str(page.get("content") or "") for page in content)
        if not text:
            return self._structural_unavailable(
                "node",
                entry,
                node_id=node_id,
                message="Cached PageIndex node has no text content.",
            )
        return {
            "mode": "node",
            "file_ref": file_ref,
            "external_id": entry.external_id,
            "source_path": entry.source_path,
            "status": entry.pageindex_tree_status,
            "available": True,
            "pageindex_doc_id": doc_id,
            "node_id": node_id,
            "node": strip_pageindex_text_fields(node),
            "text": text,
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
            "source_path": entry.source_path,
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
            f"got source_path={entry.source_path!r}, content_type={entry.content_type!r}. "
            "Use cat <path|file_ref|document_id> --structure, "
            "cat <path|file_ref|document_id> --page, or "
            "cat <path|file_ref|document_id> --node for PDF/Markdown PageIndex files."
        )

    def _require_pageindex_document_file(self, entry: Any, command: str) -> None:
        if self._file_format(entry) in {"pdf", "markdown", "pageindex"}:
            return
        raise ValueError(
            f"{command} is only supported for PDF/Markdown PageIndex files; "
            f"got source_path={entry.source_path!r}, content_type={entry.content_type!r}. "
            "Use cat <path|file_ref|document_id> --all for txt/text files."
        )

    @classmethod
    def _file_format(cls, entry: Any) -> str:
        suffix = Path(str(entry.source_path or "")).suffix.lower()
        content_type = cls._normalized_content_type(entry.content_type)
        if suffix == ".pdf" or content_type == "application/pdf":
            return "pdf"
        if suffix in PAGEINDEX_DOCUMENT_SUFFIXES or content_type in PAGEINDEX_DOCUMENT_CONTENT_TYPES:
            return "markdown"
        if suffix in TEXT_ARTIFACT_SUFFIXES:
            return "text"
        if entry.pageindex_doc_id or entry.pageindex_tree_status != "not_built":
            return "pageindex"
        if content_type in TEXT_ARTIFACT_CONTENT_TYPES:
            return "text"
        return "unsupported"

    @classmethod
    def _source_format(cls, source_path: Any, content_type: str | None) -> str:
        suffix = Path(str(source_path or "")).suffix.lower()
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
        source_path: str,
        content_type: str,
    ) -> tuple[str | None, str]:
        if self._source_format(source_path, content_type) not in {"pdf", "markdown"}:
            return None, "not_built"
        client = self._pageindex_client()
        source = self._canonical_source_path(storage_uri=storage_uri, source_path=source_path)
        cached_doc_id = self._find_cached_pageindex_doc_id(client, source)
        if cached_doc_id:
            return cached_doc_id, "built"
        if source is None:
            return None, "failed"
        try:
            doc_id = client.index(source)
            return doc_id, "built"
        except Exception:
            return None, "failed"

    def _find_cached_pageindex_doc_id(
        self,
        client: PageIndexClient,
        source_path: str | None,
    ) -> str | None:
        if source_path is None:
            return None
        for doc_id, doc in client.documents.items():
            if self._canonical_path(doc.get("path")) == source_path:
                return doc_id
        return None

    def _canonical_source_path(self, *, storage_uri: str, source_path: str) -> str | None:
        parsed = urlparse(storage_uri)
        if parsed.scheme == "file":
            return self._canonical_path(unquote(parsed.path))
        if storage_uri and not parsed.scheme:
            return self._canonical_path(storage_uri)
        if Path(source_path).expanduser().is_absolute():
            return self._canonical_path(source_path)
        return None

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

    def _prepare_file_record(self, file: dict[str, Any]) -> dict[str, Any]:
        storage_uri = file["storage_uri"]
        raw_source_path = str(file["source_path"])
        source_path = raw_source_path.strip("/")
        metadata = file.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        legacy_value_key = "derived_" + "metadata"
        legacy_policy_key = "metadata_" + "generation_policy"
        legacy_status_key = "metadata_" + "generation_status"
        if legacy_value_key in file:
            raise ValueError("legacy generated metadata map has been removed; put values in metadata")
        if legacy_policy_key in file:
            raise ValueError("legacy metadata policy key has been renamed to metadata_policy")
        if legacy_status_key in file:
            raise ValueError("legacy metadata status key has been renamed to metadata_status")
        self._validate_register_metadata(metadata)
        external_id = file.get("external_id")
        content = file.get("content") or ""
        content_type = file.get("content_type") or "text/plain"
        pageindex_doc_id, pageindex_tree_status = self._registration_pageindex_pointer(
            storage_uri=storage_uri,
            source_path=raw_source_path,
            content_type=content_type,
        )
        artifact_content = self._registration_text_artifact_content(
            source_path=raw_source_path,
            content_type=content_type,
            pageindex_doc_id=pageindex_doc_id,
            pageindex_tree_status=pageindex_tree_status,
            fallback_content=content,
        )
        fts_content = file.get("fts_content", artifact_content)
        source_type = file.get("source_type") or self._infer_source_type(source_path)
        metadata_policy = self._normalize_metadata_policy(
            file.get("metadata_policy"),
            metadata=metadata,
        )
        metadata_status = self._metadata_status_state(
            metadata_policy,
            metadata=metadata,
            status=file.get("metadata_status"),
        )
        indexed_metadata = SQLiteFileSystemStore.indexed_metadata_values(metadata)
        searchable_metadata = dict(metadata)
        folder_path = normalize_path(file.get("folder_path") or "/")
        title = file.get("title") or metadata.get("title") or Path(source_path).stem
        file_ref = make_file_ref(external_id or source_path)
        text_artifact_path = file.get("text_artifact_path") or self.store.write_text_artifact(
            file_ref,
            artifact_content,
        )
        raw_artifact_path = file.get("raw_artifact_path")
        if raw_artifact_path is None and file.get("write_raw_artifact", True):
            raw_artifact_path = self.store.write_raw_artifact(
                file_ref,
                self._raw_artifact_payload(
                    storage_uri=storage_uri,
                    source_path=source_path,
                    folder_path=folder_path,
                    metadata=metadata,
                    metadata_status=metadata_status,
                ),
            )
        descriptor = self._build_descriptor(title, metadata)
        return {
            "file_ref": file_ref,
            "external_id": external_id,
            "storage_uri": storage_uri,
            "source_path": source_path,
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
        }

    def _registration_text_artifact_content(
        self,
        *,
        source_path: str,
        content_type: str,
        pageindex_doc_id: str | None,
        pageindex_tree_status: str,
        fallback_content: str,
    ) -> str:
        if self._source_format(source_path, content_type) not in {"pdf", "markdown"}:
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
        page_text = self._pageindex_pages_text(doc.get("pages"))
        if page_text:
            return page_text
        return self._pageindex_structure_text(doc.get("structure", []))

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

    @classmethod
    def _pageindex_structure_text(cls, structure: Any) -> str:
        parts: list[str] = []
        cls._collect_pageindex_node_text(structure, parts)
        return "\n\n".join(parts)

    @classmethod
    def _collect_pageindex_node_text(cls, node: Any, parts: list[str]) -> None:
        if isinstance(node, list):
            for item in node:
                cls._collect_pageindex_node_text(item, parts)
            return
        if not isinstance(node, dict):
            return
        text = str(node.get("text") or "").strip()
        if text:
            parts.append(text)
        cls._collect_pageindex_node_text(node.get("nodes", []), parts)

    @staticmethod
    def _raw_artifact_payload(
        *,
        storage_uri: str,
        source_path: str,
        folder_path: str,
        metadata: dict[str, Any],
        metadata_status: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "storage_uri": storage_uri,
            "source_path": source_path,
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
                    storage_uri=record["storage_uri"],
                    source_path=record["source_path"],
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
        return {
            "file_ref": entry.file_ref,
            "external_id": entry.external_id,
            "storage_uri": entry.storage_uri,
            "source_path": entry.source_path,
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
                    source_path=record["source_path"],
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

    def _complete_summary_projection_index(self, record: dict[str, Any]) -> None:
        metadata_status = record["metadata_status"]
        summary_index = metadata_status.get("projection_indexes", {}).get("summary")
        if not summary_index or not summary_index.get("requested"):
            return
        summary = str(record.get("metadata", {}).get("summary") or "").strip()
        if not summary:
            return
        if self.summary_projection_indexer is None:
            self._refresh_record_metadata_status(record)
            return
        try:
            result = self.summary_projection_indexer.upsert_summary(record)
        except Exception as exc:
            summary_index["status"] = "failed"
            summary_index["error"] = str(exc)
            self._refresh_record_metadata_status(record)
            return
        summary_index.clear()
        summary_index.update({"requested": True, **result})
        if summary_index.get("status") != "ready":
            summary_index["status"] = "ready"
        self._refresh_record_metadata_status(record)

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
            source_path=entry.source_path,
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
            source_path=entry.source_path,
        )

    @staticmethod
    def _structural_unavailable(
        mode: str,
        entry: Any,
        *,
        message: str,
        node_id: str | None = None,
        pages: str | None = None,
    ) -> dict[str, Any]:
        result = {
            "mode": mode,
            "file_ref": entry.file_ref,
            "external_id": entry.external_id,
            "source_path": entry.source_path,
            "status": entry.pageindex_tree_status,
            "available": False,
            "message": message,
        }
        if node_id is not None:
            result["node_id"] = node_id
        if pages is not None:
            result["pages"] = pages
        return result

    def _resolve_target(self, target: str) -> str:
        return self.store.resolve_file_ref(target)

    def _should_use_semantic_retrieval(
        self,
        query: Union[str, list[str], None],
        scope: Optional[dict[str, Any]],
    ) -> bool:
        if self.semantic_retrieval_backend is None:
            return False
        if not self._query_text(query):
            return False
        if not scope:
            return True
        return bool(scope.get("recursive", True))

    def _semantic_search(
        self,
        query: Union[str, list[str], None],
        *,
        scope: Optional[dict[str, Any]],
        metadata_filter: Optional[dict[str, Any]],
        limit: int,
        channel: str | None = None,
    ) -> list[SearchResult]:
        if self.semantic_retrieval_backend is None:
            return []
        filters = self._semantic_filters_for_scope(scope)
        fetch_limit = max(limit * 10, 50)
        query_text = self._query_text(query)
        if channel:
            search_channel = getattr(self.semantic_retrieval_backend, "search_channel", None)
            if search_channel is None:
                return []
            candidates = search_channel(
                channel,
                query_text,
                limit=fetch_limit,
                filters=filters,
            )
        else:
            candidates = self.semantic_retrieval_backend.search(
                query_text,
                limit=fetch_limit,
                filters=filters,
            )
        results: list[SearchResult] = []
        seen: set[str] = set()
        scope_path = self._scope_folder_path(scope)
        for candidate in candidates:
            try:
                file_ref = self.store.resolve_file_ref(candidate.document_id)
            except KeyError:
                continue
            if file_ref in seen:
                continue
            if not self.store.file_matches(file_ref, scope=scope, metadata_filter=metadata_filter):
                continue
            seen.add(file_ref)
            entry = self.store.get_file(file_ref)
            folder_paths = [
                folder["path"]
                for folder in self.store.folder_memberships(file_ref)
            ]
            folder_path = self._preferred_folder_path(folder_paths, scope_path, entry.folder_path)
            results.append(
                SearchResult(
                    file_ref=file_ref,
                    external_id=entry.external_id,
                    title=entry.title,
                    snippet=candidate.snippet or entry.descriptor,
                    folder_path=folder_path,
                    folder_paths=folder_paths,
                    metadata=entry.metadata,
                    metadata_status=entry.metadata_status,
                    source_path=entry.source_path,
                    id=entry.external_id or file_ref,
                    document_id=entry.external_id,
                    name=entry.title,
                    description=entry.descriptor,
                    status=entry.pageindex_tree_status,
                    pageNum=None,
                    createdAt=None,
                    folderId=None,
                )
            )
            if len(results) >= limit:
                break
        return results

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
    def _infer_source_type(source_path: str) -> Optional[str]:
        parts = [part for part in Path(source_path).parts if part not in ("", ".")]
        return parts[0] if parts else None

    @staticmethod
    def _scope_folder_path(scope: Optional[dict[str, Any]]) -> Optional[str]:
        if not scope:
            return None
        path = scope.get("folder_path") or scope.get("path")
        return normalize_path(path) if path else None

    @classmethod
    def _semantic_filters_for_scope(cls, scope: Optional[dict[str, Any]]) -> dict[str, Any]:
        path = cls._scope_folder_path(scope)
        if not path or path == "/":
            return {}
        source_type = cls._source_type_filter_from_path(path)
        return {"source_type": source_type} if source_type else {}

    @staticmethod
    def _source_type_filter_from_path(path: str) -> str:
        segments = [segment for segment in path.strip("/").split("/") if segment]
        if not segments:
            return ""
        if segments[0] == SEMANTIC_FOLDER_ROOT.strip("/"):
            segments = segments[1:]
        if not segments:
            return ""
        first_segment = segments[0]
        if first_segment.startswith("source_type="):
            return first_segment.split("=", 1)[1].replace("-", "_")
        if path.startswith(f"{SEMANTIC_FOLDER_ROOT}/"):
            return ""
        return ""

    @classmethod
    def _validate_semantic_folder_projection_item(
        cls,
        item: dict[str, Any],
        allowed_extension_fields: set[str],
    ) -> None:
        path = item.get("folder_path") or item.get("path")
        if not path:
            raise ValueError("Semantic Folder Projection items must include a folder path")
        cls._validate_semantic_folder_projection_path(str(path))
        allowed_fields = (
            SEMANTIC_FOLDER_BASE_FIELDS
            | SEMANTIC_FOLDER_SYSTEM_FIELDS
            | allowed_extension_fields
        )
        if item.get("dataset_doc_uuid"):
            raise ValueError(
                "dataset_doc_uuid is not allowed in Semantic Folder Projection memberships; "
                "use file_key or file_ref"
            )
        fields = []
        explicit_field = cls._canonical_semantic_folder_field_name(item.get("field"))
        if explicit_field:
            fields.append(explicit_field)
        fields.extend(cls._semantic_folder_projection_fields_from_path(str(path)))
        for payload_key in ("metadata", "folder_metadata"):
            cls._validate_semantic_folder_projection_metadata_payload(
                item.get(payload_key),
                allowed_fields,
            )
        for field in fields:
            if is_semantic_folder_forbidden_field(field) or field not in allowed_fields:
                raise ValueError(f"Field is not allowed for Semantic Folder Projection: {field}")

    @staticmethod
    def _validate_semantic_folder_projection_path(path: str) -> str:
        normalized = normalize_path(path)
        if normalized != SEMANTIC_FOLDER_ROOT and not normalized.startswith(
            f"{SEMANTIC_FOLDER_ROOT}/"
        ):
            raise ValueError("Semantic Folder Projection paths must be under /semantic")
        return normalized

    @classmethod
    def _semantic_folder_projection_fields_from_path(cls, path: str) -> list[str]:
        normalized = cls._validate_semantic_folder_projection_path(path)
        fields: list[str] = []
        for segment in normalized.strip("/").split("/")[1:]:
            if "=" not in segment:
                continue
            field = cls._canonical_semantic_folder_field_name(
                segment.split("=", 1)[0]
            )
            if field:
                fields.append(field)
        return fields

    @classmethod
    def _validate_semantic_folder_projection_metadata_payload(
        cls,
        payload: Any,
        allowed_fields: set[str],
    ) -> None:
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_text = str(key)
                key_field = cls._canonical_semantic_folder_field_name(key)
                if is_semantic_folder_forbidden_field(key_field):
                    raise ValueError(
                        "Forbidden metadata field in Semantic Folder Projection payload: "
                        f"{key_text}"
                    )
                if key_field in {"field", "source_field", "metadata_field"}:
                    field = cls._canonical_semantic_folder_field_name(value)
                    if field and (
                        is_semantic_folder_forbidden_field(field)
                        or field not in allowed_fields
                    ):
                        raise ValueError(
                            f"Field is not allowed for Semantic Folder Projection: {field}"
                        )
                cls._validate_semantic_folder_projection_metadata_payload(value, allowed_fields)
        elif isinstance(payload, list):
            for item in payload:
                cls._validate_semantic_folder_projection_metadata_payload(item, allowed_fields)
        elif isinstance(payload, str):
            field = cls._canonical_semantic_folder_field_name(payload)
            if is_semantic_folder_forbidden_field(field):
                raise ValueError(
                    "Forbidden metadata field label in Semantic Folder Projection payload: "
                    f"{payload}"
                )

    @staticmethod
    def _canonical_semantic_folder_field_name(value: Any) -> str:
        return canonical_semantic_folder_field_name(value)

    @staticmethod
    def _semantic_folder_projection_document_id(membership: dict[str, Any]) -> str:
        for key in ("file_key", "file_ref", "document_ref"):
            value = str(membership.get(key) or "").strip()
            if value:
                return value
        raise ValueError("Semantic Folder Projection membership is missing file_key or file_ref")

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
