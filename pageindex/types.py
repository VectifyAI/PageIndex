# pageindex/types.py
# TypedDicts describing the plain-dict shapes the SDK returns, so callers get
# key/field discovery in their IDE without any runtime cost (these are dicts).
from __future__ import annotations

from typing import Any, TypedDict


class DocumentInfo(TypedDict):
    """A document as returned by ``list_documents()``."""
    doc_id: str
    doc_name: str
    doc_description: str
    doc_type: str


class _DocumentDetailRequired(DocumentInfo):
    """``structure`` is always present — split into its own (default
    total=True) base so the total=False below only applies to the genuinely
    optional, backend-specific fields below. A single
    ``class DocumentDetail(DocumentInfo, total=False): structure: ...`` would
    incorrectly mark ``structure`` optional too, since total=False applies to
    the whole class body, not just the fields declared after it.
    """
    structure: list[dict[str, Any]]


class DocumentDetail(_DocumentDetailRequired, total=False):
    """A document with its tree, as returned by ``get_document()``.

    ``structure`` is always present; ``file_path`` is local-only and
    ``status`` is cloud-only, hence total=False for those two only.
    """
    file_path: str  # local backend only
    status: str     # cloud backend only


class PageContent(TypedDict, total=False):
    """One page of content, as returned by ``get_page_content()``."""
    page: int
    content: str
    images: list[dict[str, Any]]
