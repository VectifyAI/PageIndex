# pageindex/__init__.py
# Load .env first so env-based credentials (e.g. OPENAI_API_KEY) are set.
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

# Backward compatibility: honor CHATGPT_API_KEY as an alias for OPENAI_API_KEY.
import os as _os
_chatgpt_key = _os.getenv("CHATGPT_API_KEY")
if not _os.getenv("OPENAI_API_KEY") and _chatgpt_key:
    _os.environ["OPENAI_API_KEY"] = _chatgpt_key

from typing import TYPE_CHECKING as _TYPE_CHECKING
if _TYPE_CHECKING:
    # Static-only bindings for the lazy legacy names — real signatures for IDEs.
    from .index.page_index import page_index, page_index_main, tree_parser
    from .index.page_index_md import md_to_tree
    from .index.utils import ConfigLoader, llm_completion, llm_acompletion
    from .retrieve import get_document, get_document_structure, get_page_content

# SDK exports — must stay light: no LLM/indexing imports here.
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
    "page_index_main",
    "tree_parser",
    "ConfigLoader",
    "llm_completion",
    "llm_acompletion",
    "md_to_tree",
    "get_document",
    "get_document_structure",
    "get_page_content",
]

# Legacy (pre-SDK) exports resolve lazily (PEP 562) so cloud-only
# `import pageindex` never pays for litellm/PyPDF2.
_LAZY_LEGACY = {
    "md_to_tree": ".index.page_index_md",
    "get_document": ".retrieve",
    "get_document_structure": ".retrieve",
    "get_page_content": ".retrieve",
}


def __getattr__(name):
    if name.startswith("_"):
        # dunder probes (pickle, IPython) must not trigger the heavy import
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    if name in ("utils", "page_index_md"):
        return importlib.import_module(f".{name}", __name__)
    module = _LAZY_LEGACY.get(name)
    if module is not None:
        value = getattr(importlib.import_module(module, __name__), name)
        globals()[name] = value
        return value
    # Remaining pre-SDK names live in index.page_index (star-exports index.utils).
    legacy = importlib.import_module(".index.page_index", __name__)
    try:
        value = getattr(legacy, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | {"utils", "page_index_md"})
