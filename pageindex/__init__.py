# pageindex/__init__.py
# Load .env explicitly, before anything else, so environment-based credentials
# (OPENAI_API_KEY for local mode, PAGEINDEX_API_KEY that callers read via
# os.environ for cloud mode) are populated by PageIndex itself — not left to
# litellm's incidental dotenv loading, which would vanish if litellm changes or
# its import is ever made lazy.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

# Backward compatibility: honor CHATGPT_API_KEY as an alias for OPENAI_API_KEY
# (kept from the pre-SDK pageindex.utils). Runs after load_dotenv so a value in
# .env is picked up too; only fills OPENAI_API_KEY when it isn't already set.
import os as _os
if not _os.getenv("OPENAI_API_KEY") and _os.getenv("CHATGPT_API_KEY"):
    _os.environ["OPENAI_API_KEY"] = _os.getenv("CHATGPT_API_KEY")

# Upstream exports (backward compatibility). Import from the canonical
# pageindex.index.* modules directly so `import pageindex` does NOT trip the
# top-level deprecation shims (pageindex.page_index / .page_index_md / .utils).
from .index.page_index import *  # noqa: E402
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
    # Legacy top-level exports (pre-SDK API), kept so `from pageindex import *`
    # still binds them.
    "page_index",
    "md_to_tree",
    "get_document",
    "get_document_structure",
    "get_page_content",
]
