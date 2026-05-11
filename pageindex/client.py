# pageindex/client.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator

from typing_extensions import deprecated

from .collection import Collection
from .config import IndexConfig
from .errors import PageIndexAPIError
from .parser.protocol import DocumentParser

_LEGACY_SDK_MSG = (
    "Legacy compatibility — new code should prefer the Collection-based API "
    "(PageIndexClient.collection(...))."
)
_legacy_sdk = deprecated(_LEGACY_SDK_MSG, category=PendingDeprecationWarning)


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """PageIndex client — supports both local and cloud modes.

    Args:
        api_key: PageIndex cloud API key. When provided, cloud mode is used
            and local-only params (model, storage_path, index_config, …) are ignored.
        model: LLM model for indexing (local mode only, default: gpt-4o-2024-11-20).
        retrieve_model: LLM model for agent QA (local mode only, default: same as model).
        storage_path: Directory for SQLite DB and files (local mode only, default: ./.pageindex).
        storage: Custom StorageEngine instance (local mode only).
        index_config: Advanced indexing parameters (local mode only, optional).
            Pass an IndexConfig instance or a dict. Defaults are sensible for most use cases.

    Usage:
        # Local mode (auto-detected when no api_key)
        client = PageIndexClient(model="gpt-5.4")

        # Cloud mode (auto-detected when api_key provided)
        client = PageIndexClient(api_key="your-api-key")

        # Or use LocalClient / CloudClient for explicit mode selection
    """

    BASE_URL = "https://api.pageindex.ai"

    def __init__(self, api_key: str | None = None, model: str = None,
                 retrieve_model: str = None, storage_path: str = None,
                 storage=None, index_config: IndexConfig | dict = None):
        if api_key == "":
            import logging
            logging.getLogger(__name__).warning(
                "PageIndexClient received an empty api_key; falling back to local mode. "
                "Pass api_key=None to silence this warning, or provide a real key for cloud mode."
            )
            api_key = None
        if api_key is not None:
            self._init_cloud(api_key)
        else:
            self._init_local(model, retrieve_model, storage_path, storage, index_config)

    def _init_cloud(self, api_key: str):
        from .backend.cloud import CloudBackend
        from .cloud_api import LegacyCloudAPI
        self._backend = CloudBackend(api_key=api_key)
        self._legacy_cloud_api = LegacyCloudAPI(api_key=api_key, base_url=self.BASE_URL)

    def _init_local(self, model: str = None, retrieve_model: str = None,
                    storage_path: str = None, storage=None,
                    index_config: IndexConfig | dict = None):
        self._legacy_cloud_api = None

        # Build IndexConfig: merge model/retrieve_model with index_config
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        if isinstance(index_config, IndexConfig):
            opt = index_config.model_copy(update=overrides)
        elif isinstance(index_config, dict):
            merged = {**index_config, **overrides}  # explicit model/retrieve_model win
            opt = IndexConfig(**merged)
        else:
            opt = IndexConfig(**overrides) if overrides else IndexConfig()

        self._validate_llm_provider(opt.model)

        storage_path = Path(storage_path or ".pageindex").resolve()
        storage_path.mkdir(parents=True, exist_ok=True)

        from .storage.sqlite import SQLiteStorage
        from .backend.local import LocalBackend
        storage_engine = storage or SQLiteStorage(str(storage_path / "pageindex.db"))
        self._backend = LocalBackend(
            storage=storage_engine,
            files_dir=str(storage_path / "files"),
            model=opt.model,
            retrieve_model=_normalize_retrieve_model(opt.retrieve_model or opt.model),
            index_config=opt,
        )

    @staticmethod
    def _validate_llm_provider(model: str) -> None:
        """Validate model and check API key via litellm. Warns if key seems missing."""
        try:
            import litellm
            litellm.model_cost_map_url = ""
            _, provider, _, _ = litellm.get_llm_provider(model=model)
        except Exception:
            return

        key = litellm.get_api_key(llm_provider=provider, dynamic_api_key=None)
        if not key:
            import os
            common_var = f"{provider.upper()}_API_KEY"
            if not os.getenv(common_var):
                from .errors import PageIndexError
                raise PageIndexError(
                    f"API key not configured for provider '{provider}' (model: {model}). "
                    f"Set the {common_var} environment variable."
                )

    def collection(self, name: str = "default") -> Collection:
        """Get or create a collection. Defaults to 'default'."""
        self._backend.get_or_create_collection(name)
        return Collection(name=name, backend=self._backend)

    def list_collections(self) -> list[str]:
        return self._backend.list_collections()

    def delete_collection(self, name: str) -> None:
        self._backend.delete_collection(name)

    def register_parser(self, parser: DocumentParser) -> None:
        """Register a custom document parser. Only available in local mode."""
        if not hasattr(self._backend, 'register_parser'):
            from .errors import PageIndexError
            raise PageIndexError("Custom parsers are not supported in cloud mode")
        self._backend.register_parser(parser)

    def _require_cloud_api(self):
        if self._legacy_cloud_api is None:
            from .errors import PageIndexAPIError
            raise PageIndexAPIError(
                "This method is part of the pageindex 0.2.x cloud SDK API. "
                "Initialize with api_key to use it."
            )
        return self._legacy_cloud_api

    # ── pageindex 0.2.x cloud SDK compatibility (prefer Collection API for new code) ──
    @_legacy_sdk
    def submit_document(
        self,
        file_path: str,
        mode: str | None = None,
        beta_headers: list[str] | None = None,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``client.collection(...).add(path)``."""
        return self._require_cloud_api().submit_document(
            file_path=file_path,
            mode=mode,
            beta_headers=beta_headers,
            folder_id=folder_id,
        )

    @_legacy_sdk
    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``collection.get_page_content(doc_id, pages)``."""
        return self._require_cloud_api().get_ocr(doc_id=doc_id, format=format)

    @_legacy_sdk
    def get_tree(self, doc_id: str, node_summary: bool = False) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``collection.get_document_structure(doc_id)``."""
        return self._require_cloud_api().get_tree(doc_id=doc_id, node_summary=node_summary)

    @_legacy_sdk
    def is_retrieval_ready(self, doc_id: str) -> bool:
        """Legacy SDK compatibility — Collection API handles readiness internally."""
        return self._require_cloud_api().is_retrieval_ready(doc_id=doc_id)

    @_legacy_sdk
    def submit_query(self, doc_id: str, query: str, thinking: bool = False) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``collection.query(question, doc_ids=[doc_id])``."""
        return self._require_cloud_api().submit_query(
            doc_id=doc_id,
            query=query,
            thinking=thinking,
        )

    @_legacy_sdk
    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        """Legacy SDK compatibility — Collection API returns answers synchronously."""
        return self._require_cloud_api().get_retrieval(retrieval_id=retrieval_id)

    @_legacy_sdk
    def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        doc_id: str | list[str] | None = None,
        temperature: float | None = None,
        stream_metadata: bool = False,
        enable_citations: bool = False,
    ) -> dict[str, Any] | Iterator[str] | Iterator[dict[str, Any]]:
        """Legacy SDK compatibility — prefer ``collection.query(...)``."""
        return self._require_cloud_api().chat_completions(
            messages=messages,
            stream=stream,
            doc_id=doc_id,
            temperature=temperature,
            stream_metadata=stream_metadata,
            enable_citations=enable_citations,
        )

    @_legacy_sdk
    def get_document(self, doc_id: str) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``collection.get_document(doc_id)``."""
        return self._require_cloud_api().get_document(doc_id=doc_id)

    @_legacy_sdk
    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``collection.delete_document(doc_id)``."""
        return self._require_cloud_api().delete_document(doc_id=doc_id)

    @_legacy_sdk
    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``collection.list_documents()``."""
        return self._require_cloud_api().list_documents(
            limit=limit,
            offset=offset,
            folder_id=folder_id,
        )

    @_legacy_sdk
    def create_folder(
        self,
        name: str,
        description: str | None = None,
        parent_folder_id: str | None = None,
    ) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``client.collection(name)`` (auto-creates)."""
        return self._require_cloud_api().create_folder(
            name=name,
            description=description,
            parent_folder_id=parent_folder_id,
        )

    @_legacy_sdk
    def list_folders(self, parent_folder_id: str | None = None) -> dict[str, Any]:
        """Legacy SDK compatibility — prefer ``client.list_collections()``."""
        return self._require_cloud_api().list_folders(parent_folder_id=parent_folder_id)


class LocalClient(PageIndexClient):
    """Local mode — indexes and queries documents on your machine.

    Args:
        model: LLM model for indexing (default: gpt-4o-2024-11-20)
        retrieve_model: LLM model for agent QA (default: same as model)
        storage_path: Directory for SQLite DB and files (default: ./.pageindex)
        storage: Custom StorageEngine instance (default: SQLiteStorage)
        index_config: Advanced indexing parameters. Pass an IndexConfig instance
            or a dict. All fields have sensible defaults — most users don't need this.

    Example::

        # Simple — defaults are fine
        client = LocalClient(model="gpt-5.4")

        # Advanced — tune indexing parameters
        from pageindex.config import IndexConfig
        client = LocalClient(
            model="gpt-5.4",
            index_config=IndexConfig(toc_check_page_num=30),
        )
    """

    def __init__(self, model: str = None, retrieve_model: str = None,
                 storage_path: str = None, storage=None,
                 index_config: IndexConfig | dict = None):
        self._init_local(model, retrieve_model, storage_path, storage, index_config)


class CloudClient(PageIndexClient):
    """Cloud mode — fully managed by PageIndex cloud service. No LLM key needed."""

    def __init__(self, api_key: str):
        self._init_cloud(api_key)
