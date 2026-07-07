# pageindex/__init__.py
# Upstream exports (backward compatibility). Import from the canonical
# pageindex.index.* modules directly so `import pageindex` does NOT trip the
# top-level deprecation shims (pageindex.page_index / .page_index_md / .utils).
from .index.page_index import *
from .index.page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content

# SDK exports
from .client import PageIndexClient, LocalClient, CloudClient
from .config import IndexConfig, set_llm_params
from .collection import Collection
from .types import DocumentInfo, DocumentDetail, PageContent
from .parser.protocol import ContentNode, ParsedDocument, DocumentParser
from .storage.protocol import StorageEngine
from .events import QueryEvent
from .errors import (
    PageIndexError,
    PageIndexAPIError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    IndexingError,
    CloudAPIError,
    FileTypeError,
)

__all__ = [
    "PageIndexClient",
    "LocalClient",
    "CloudClient",
    "IndexConfig",
    "set_llm_params",
    "Collection",
    "DocumentInfo",
    "DocumentDetail",
    "PageContent",
    "ContentNode",
    "ParsedDocument",
    "DocumentParser",
    "StorageEngine",
    "QueryEvent",
    "PageIndexError",
    "PageIndexAPIError",
    "CollectionNotFoundError",
    "DocumentNotFoundError",
    "IndexingError",
    "CloudAPIError",
    "FileTypeError",
]
