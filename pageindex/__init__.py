import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

_OPTIONAL_CORE_IMPORTS = {"litellm", "openai", "PyPDF2", "pymupdf"}

try:
    from .page_index import *
    from .page_index_md import md_to_tree
    from .retrieve import get_document, get_document_structure, get_page_content
    from .client import PageIndexClient
except ModuleNotFoundError as exc:
    if exc.name not in _OPTIONAL_CORE_IMPORTS:
        raise


def __getattr__(name: str):
    if name == "PageIndexFileSystem":
        from .filesystem import PageIndexFileSystem

        return PageIndexFileSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
