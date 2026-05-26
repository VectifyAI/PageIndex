from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import unquote, urlparse

from ..client import PageIndexClient
from .metadata import MetadataQueryEngine
from .metadata_generation import (
    MetadataGenerationError,
    MetadataGenerationInput,
    MetadataGenerationResult,
    MetadataGenerator,
    OpenAIMetadataGenerator,
)
from .projection_indexing import SummaryProjectionIndexer
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
    first_node_location,
    find_pageindex_node,
    strip_pageindex_text_fields,
)
from .types import OpenResult, SearchResult

DEFAULT_METADATA_GENERATION_FIELDS = {
    "summary": True,
    "doc_type": True,
    "domain": True,
    "topic": True,
    "entity": False,
    "relation": False,
}

DEFAULT_DERIVED_METADATA_FIELD_TYPES = {
    "summary": "string",
    "doc_type": "string",
    "domain": "string",
    "topic": "string",
    "entity": "string",
    "relation": "string",
}

METADATA_GENERATION_STATUSES = {
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
        metadata_generator: MetadataGenerator | None = None,
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
        derived_metadata: Optional[dict[str, Any]] = None,
        metadata_generation_policy: Optional[dict[str, Any]] = None,
        metadata_generation_status: Optional[str] = None,
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
                    "derived_metadata": derived_metadata,
                    "metadata_generation_policy": metadata_generation_policy,
                    "metadata_generation_status": metadata_generation_status,
                }
            ]
        )[0]

    def register(self, **kwargs: Any) -> str:
        if not self._register_uses_deferred_metadata(kwargs.get("metadata_generation_policy")):
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
        rows = self.store.list_pending_metadata_generation(limit=limit)
        generated = 0
        failed = 0
        file_refs: list[str] = []
        for row in rows:
            record = self._record_from_file_entry(row)
            self._generate_register_metadata(record, force=True)
            self._complete_summary_projection_index(record)
            self._register_generation_policy_schema([record])
            self.store.update_file_metadata_generation(
                record["file_ref"],
                derived_metadata=record["derived_metadata"],
                metadata_generation=record["metadata_generation"],
            )
            self._sync_owned_raw_artifact(record)
            file_refs.append(record["file_ref"])
            if record["metadata_generation"]["status"] == "failed":
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
            self.metadata_generator = OpenAIMetadataGenerator()
        if self.summary_projection_index and self.summary_projection_indexer is None:
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

    @staticmethod
    def _register_uses_deferred_metadata(policy: Any) -> bool:
        if not isinstance(policy, dict):
            return False
        return bool(policy.get("batch")) or policy.get("mode") == "batch"

    @classmethod
    def default_metadata_generation_policy(cls) -> dict[str, Any]:
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
                    reference_id=row["external_id"] or row["file_ref"],
                    file_ref=row["file_ref"],
                    external_id=row["external_id"],
                    title=row["title"],
                    snippet=row["snippet"],
                    folder_path=folder_path,
                    folder_paths=folder_paths,
                    metadata=row["metadata"],
                    derived_metadata=row["derived_metadata"],
                    metadata_generation=row["metadata_generation"],
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
        reference_id: str,
        patterns: Union[str, list[str]],
        limit: int = 20,
    ) -> list[OpenResult]:
        file_ref = self._resolve_reference(reference_id)
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
                matches.append(self._open_lines(reference_id, file_ref, start, end))
                if len(matches) >= limit:
                    break
        return matches

    def open(self, reference_id: str, location: str = "all") -> OpenResult:
        file_ref = self._resolve_reference(reference_id)
        entry = self.store.get_file(file_ref)
        if self._file_format(entry) in {"pdf", "markdown", "pageindex"}:
            raise ValueError(
                "open() text artifact reads are not supported for PDF/Markdown PageIndex files; "
                "use pageindex_structure(), pageindex_pages(), or pageindex_node()."
            )
        if str(location).strip().lower() in {"all", "full", "*"}:
            return self._open_all(reference_id, file_ref)
        start, end = self._parse_line_range(location)
        return self._open_lines(reference_id, file_ref, start, end)

    def cat_text_artifact(self, reference_id: str, location: str = "all") -> OpenResult:
        file_ref = self._resolve_reference(reference_id)
        entry = self.store.get_file(file_ref)
        self._require_text_artifact_file(entry, "cat --all")
        if str(location).strip().lower() in {"all", "full", "*"}:
            return self._open_all(reference_id, file_ref)
        start, end = self._parse_line_range(location)
        return self._open_lines(reference_id, file_ref, start, end)

    def pageindex_structure(self, reference_id: str) -> dict[str, Any]:
        file_ref = self._resolve_reference(reference_id)
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
            "source_path": entry.source_path,
            "status": entry.pageindex_tree_status,
            "available": True,
            "pageindex_doc_id": doc_id,
            "structure": strip_pageindex_text_fields(structure),
        }

    def pageindex_node(self, reference_id: str, node_id: str) -> dict[str, Any]:
        file_ref = self._resolve_reference(reference_id)
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

    def pageindex_pages(self, reference_id: str, pages: str) -> dict[str, Any]:
        file_ref = self._resolve_reference(reference_id)
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
        file_ref = self._resolve_reference(target)
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
        derived_metadata = file.get("derived_metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        if not isinstance(derived_metadata, dict):
            raise ValueError("derived_metadata must be a JSON object")
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
        generation_policy = self._normalize_metadata_generation_policy(
            file.get("metadata_generation_policy"),
            derived_metadata=derived_metadata,
        )
        generation_state = self._metadata_generation_state(
            generation_policy,
            derived_metadata=derived_metadata,
            status=file.get("metadata_generation_status"),
        )
        indexed_metadata = SQLiteFileSystemStore.indexed_metadata_values(
            metadata,
            derived_metadata,
            generation_state,
        )
        searchable_metadata = self._merge_metadata_values(metadata, derived_metadata)
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
                    derived_metadata=derived_metadata,
                    metadata_generation=generation_state,
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
            "derived_metadata": derived_metadata,
            "derived_metadata_json": json.dumps(derived_metadata, ensure_ascii=False),
            "metadata_generation": generation_state,
            "metadata_generation_json": json.dumps(generation_state, ensure_ascii=False),
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
        derived_metadata: dict[str, Any],
        metadata_generation: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "storage_uri": storage_uri,
            "source_path": source_path,
            "folder_path": folder_path,
            "metadata": metadata,
            "derived_metadata": derived_metadata,
            "metadata_generation": metadata_generation,
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
                    derived_metadata=record["derived_metadata"],
                    metadata_generation=record["metadata_generation"],
                ),
            )
        )

    def _record_from_file_entry(self, entry: Any) -> dict[str, Any]:
        content = self.store.read_text(entry.file_ref)
        generation_policy = self._normalize_metadata_generation_policy(
            entry.metadata_generation.get("policy", {}),
            derived_metadata=entry.derived_metadata,
        )
        generation_state = self._metadata_generation_state(
            generation_policy,
            derived_metadata=entry.derived_metadata,
            status=entry.metadata_generation.get("status"),
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
            "derived_metadata": dict(entry.derived_metadata),
            "derived_metadata_json": json.dumps(entry.derived_metadata, ensure_ascii=False),
            "metadata_generation": generation_state,
            "metadata_generation_json": json.dumps(generation_state, ensure_ascii=False),
            "indexed_metadata": SQLiteFileSystemStore.indexed_metadata_values(
                entry.metadata,
                entry.derived_metadata,
                generation_state,
            ),
            "metadata_text": metadata_text(self._merge_metadata_values(entry.metadata, entry.derived_metadata)),
            "folder_path": entry.folder_path,
            "content": content,
            "skip_fts": False,
        }

    def _generate_register_metadata(self, record: dict[str, Any], *, force: bool = False) -> None:
        generation = record["metadata_generation"]
        policy = generation.get("policy", {})
        if self._metadata_generation_is_batch(policy) and not force:
            self._mark_requested_generation_status(record, "pending_submit")
            return
        fields = self._metadata_fields_to_generate(record)
        if not fields:
            return
        if self.metadata_generator is None:
            if self._metadata_generation_requires_sync(policy):
                raise MetadataGenerationError(
                    "metadata_generator is required for synchronous PIFS metadata generation; "
                    "set metadata_generation_policy batch=true to defer"
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
            self._apply_metadata_generation_failures(record, fields, str(exc))
            return
        failures = dict(result.failures)
        for field in fields:
            if field in result.values:
                record["derived_metadata"][field] = result.values[field]
                generation["fields"][field] = {"requested": True, "status": "generated"}
            else:
                failures.setdefault(field, "metadata generator did not return field")
        for field, reason in failures.items():
            generation["fields"][field] = {
                "requested": True,
                "status": "failed",
                "error": str(reason),
            }
        self._refresh_record_metadata_generation(record)

    def _complete_summary_projection_index(self, record: dict[str, Any]) -> None:
        generation = record["metadata_generation"]
        summary_index = generation.get("projection_indexes", {}).get("summary")
        if not summary_index or not summary_index.get("requested"):
            return
        summary = str(record.get("derived_metadata", {}).get("summary") or "").strip()
        if not summary:
            return
        if self.summary_projection_indexer is None:
            self._refresh_record_metadata_generation(record)
            return
        try:
            result = self.summary_projection_indexer.upsert_summary(record)
        except Exception as exc:
            summary_index["status"] = "failed"
            summary_index["error"] = str(exc)
            self._refresh_record_metadata_generation(record)
            return
        summary_index.clear()
        summary_index.update({"requested": True, **result})
        if summary_index.get("status") != "ready":
            summary_index["status"] = "ready"
        self._refresh_record_metadata_generation(record)

    @staticmethod
    def _metadata_generation_is_batch(policy: dict[str, Any]) -> bool:
        return bool(policy.get("batch")) or policy.get("mode") == "batch"

    @staticmethod
    def _metadata_generation_requires_sync(policy: dict[str, Any]) -> bool:
        return policy.get("batch") is False or policy.get("mode") == "sync"

    def _metadata_fields_to_generate(self, record: dict[str, Any]) -> list[str]:
        fields: list[str] = []
        for name, state in record["metadata_generation"].get("fields", {}).items():
            if not state.get("requested"):
                continue
            if state.get("status") == "generated" and name in record["derived_metadata"]:
                continue
            fields.append(name)
        return fields

    def _mark_requested_generation_status(self, record: dict[str, Any], status: str) -> None:
        for name, field in record["metadata_generation"].get("fields", {}).items():
            if field.get("requested") and field.get("status") != "generated":
                record["metadata_generation"]["fields"][name] = {
                    "requested": True,
                    "status": status,
                }
        self._refresh_record_metadata_generation(record, explicit_status=status)

    def _apply_metadata_generation_failures(
        self,
        record: dict[str, Any],
        fields: list[str],
        reason: str,
    ) -> None:
        for field in fields:
            record["metadata_generation"]["fields"][field] = {
                "requested": True,
                "status": "failed",
                "error": reason,
            }
        self._refresh_record_metadata_generation(record, explicit_status="failed")

    def _refresh_record_metadata_generation(
        self,
        record: dict[str, Any],
        *,
        explicit_status: str | None = None,
    ) -> None:
        generation = record["metadata_generation"]
        statuses = [
            field.get("status")
            for field in generation.get("fields", {}).values()
            if field.get("requested") and field.get("status")
        ]
        generation["status"] = explicit_status or self._aggregate_generation_status(statuses)
        self._refresh_projection_index_statuses(generation, record["derived_metadata"])
        record["derived_metadata_json"] = json.dumps(record["derived_metadata"], ensure_ascii=False)
        record["metadata_generation_json"] = json.dumps(generation, ensure_ascii=False)
        record["indexed_metadata"] = SQLiteFileSystemStore.indexed_metadata_values(
            record["metadata"],
            record["derived_metadata"],
            generation,
        )
        record["metadata_text"] = metadata_text(
            self._merge_metadata_values(record["metadata"], record["derived_metadata"])
        )

    def _open_lines(self, reference_id: str, file_ref: str, start: int, end: int) -> OpenResult:
        entry = self.store.get_file(file_ref)
        lines = self.store.read_text(file_ref).splitlines()
        start = max(1, start)
        end = min(max(start, end), len(lines))
        text = "\n".join(lines[start - 1:end])
        return OpenResult(
            reference_id=reference_id,
            file_ref=file_ref,
            start_line=start,
            end_line=end,
            text=text,
            external_id=entry.external_id,
            folder_path=entry.folder_path,
            source_path=entry.source_path,
        )

    def _open_all(self, reference_id: str, file_ref: str) -> OpenResult:
        entry = self.store.get_file(file_ref)
        text = self.store.read_text(file_ref)
        line_count = len(text.splitlines())
        return OpenResult(
            reference_id=reference_id,
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

    def _resolve_reference(self, reference_id: str) -> str:
        return self.store.resolve_file_ref(reference_id)

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
                    reference_id=entry.external_id or file_ref,
                    file_ref=file_ref,
                    external_id=entry.external_id,
                    title=entry.title,
                    snippet=candidate.snippet or entry.descriptor,
                    folder_path=folder_path,
                    folder_paths=folder_paths,
                    metadata=entry.metadata,
                    derived_metadata=entry.derived_metadata,
                    metadata_generation=entry.metadata_generation,
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

    def _register_generation_policy_schema(self, records: list[dict[str, Any]]) -> None:
        fields: dict[str, dict[str, str]] = {}
        for record in records:
            policy_fields = record["metadata_generation"]["policy"]["fields"]
            for name, requested in policy_fields.items():
                if requested:
                    fields[name] = {
                        "type": DEFAULT_DERIVED_METADATA_FIELD_TYPES.get(
                            name,
                            self._infer_metadata_field_type(
                                record.get("derived_metadata", {}).get(name)
                            ),
                        )
                    }
            for name, value in record.get("derived_metadata", {}).items():
                fields.setdefault(name, {"type": self._infer_metadata_field_type(value)})
        if fields:
            self.metadata.register_schema({"fields": fields}, source="derived")

    @classmethod
    def _normalize_metadata_generation_policy(
        cls,
        policy: Optional[dict[str, Any]],
        *,
        derived_metadata: dict[str, Any],
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
                raise ValueError("metadata_generation_policy must be a JSON object")
            raw_fields = policy.get("fields")
            if raw_fields is None:
                raw_fields = {
                    name: declaration
                    for name, declaration in policy.items()
                    if name not in {"batch", "mode", "status", "projection_indexes"}
                }
            if not isinstance(raw_fields, dict):
                raise ValueError("metadata_generation_policy fields must be a JSON object")
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
                        cls._validate_metadata_generation_status(str(field_status))
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
                cls._validate_metadata_generation_status(str(top_level_status))
            if "projection_indexes" in policy:
                projection_indexes, projection_index_statuses = (
                    cls._normalize_projection_index_policy(policy["projection_indexes"])
                )
        for name in derived_metadata:
            fields.setdefault(name, True)
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
    def _metadata_generation_state(
        cls,
        policy: dict[str, Any],
        *,
        derived_metadata: dict[str, Any],
        status: Optional[str],
    ) -> dict[str, Any]:
        explicit_status = status or policy.get("status")
        if explicit_status is not None:
            explicit_status = str(explicit_status)
            cls._validate_metadata_generation_status(explicit_status)
        field_statuses = policy.get("field_statuses", {})
        fields: dict[str, dict[str, Any]] = {}
        for name, requested in policy["fields"].items():
            if not requested:
                fields[name] = {"requested": False}
                continue
            field_status = field_statuses.get(name)
            if field_status is None:
                field_status = explicit_status
            if field_status is None:
                field_status = "generated" if name in derived_metadata else "pending_generate"
            fields[name] = {"requested": True, "status": field_status}

        requested_statuses = [
            item["status"]
            for item in fields.values()
            if item.get("requested") and item.get("status")
        ]
        aggregate_status = explicit_status or cls._aggregate_generation_status(requested_statuses)
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
            }
        cls._refresh_projection_index_statuses(state, derived_metadata)
        return state

    @staticmethod
    def _aggregate_generation_status(statuses: list[str]) -> str:
        if not statuses:
            return "generated"
        for status in ("failed", "pending_submit", "pending_generate"):
            if status in statuses:
                return status
        return "generated"

    @staticmethod
    def _validate_metadata_generation_status(status: str) -> None:
        if status not in METADATA_GENERATION_STATUSES:
            raise ValueError(f"Unsupported metadata generation status: {status}")

    @classmethod
    def _normalize_projection_index_policy(
        cls,
        projection_policy: Any,
    ) -> tuple[dict[str, bool], dict[str, str]]:
        if projection_policy is None:
            return {}, {}
        if not isinstance(projection_policy, dict):
            raise ValueError("metadata_generation_policy projection_indexes must be a JSON object")
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
        generation: dict[str, Any],
        derived_metadata: dict[str, Any],
    ) -> None:
        summary_index = generation.get("projection_indexes", {}).get("summary")
        if not summary_index or not summary_index.get("requested"):
            return
        if "summary" not in derived_metadata:
            return
        if summary_index.get("status", "not_indexed") == "not_indexed":
            summary_index["status"] = "pending_index"

    @classmethod
    def _merge_metadata_values(
        cls,
        metadata: dict[str, Any],
        derived_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(metadata)
        for name, value in derived_metadata.items():
            if name not in merged:
                merged[name] = value
                continue
            if merged[name] == value:
                continue
            merged[name] = cls._merge_metadata_value(merged[name], value)
        return merged

    @staticmethod
    def _merge_metadata_value(raw_value: Any, derived_value: Any) -> Any:
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        derived_values = derived_value if isinstance(derived_value, list) else [derived_value]
        merged = list(values)
        for item in derived_values:
            if item not in merged:
                merged.append(item)
        return merged

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
