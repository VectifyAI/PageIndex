# pageindex/collection.py
from __future__ import annotations
import os
import warnings
from typing import AsyncIterator
from .events import QueryEvent
from .backend.protocol import Backend
from .types import DocumentInfo, DocumentDetail, PageContent


def _multidoc_acked() -> bool:
    return os.getenv("PAGEINDEX_EXPERIMENTAL_MULTIDOC", "").lower() in ("1", "true", "yes")


_MULTIDOC_WARNING = (
    "Querying the entire collection (no doc_ids) is experimental — a naive "
    "first implementation that lets the agent pick docs from auto-generated "
    "descriptions. Better cross-document retrieval is on the way. Pass "
    "doc_ids=[...] for reliable results, or set "
    "PAGEINDEX_EXPERIMENTAL_MULTIDOC=1 to silence this warning."
)


class QueryStream:
    """Wraps backend.query_stream() as an async iterable object."""

    def __init__(self, backend: Backend, collection: str, question: str,
                 doc_ids: list[str] | None = None):
        self._backend = backend
        self._collection = collection
        self._question = question
        self._doc_ids = doc_ids

    async def stream_events(self) -> AsyncIterator[QueryEvent]:
        async for event in self._backend.query_stream(
            self._collection, self._question, self._doc_ids
        ):
            yield event

    def __aiter__(self):
        return self.stream_events()


class Collection:
    def __init__(self, name: str, backend: Backend):
        self._name = name
        self._backend = backend

    @property
    def name(self) -> str:
        return self._name

    def add(self, file_path: str) -> str:
        """Index a document (PDF or Markdown) into this collection.

        Returns the ``doc_id``. Re-adding byte-identical content returns the
        existing doc_id (content-hash dedup); change ``IndexConfig`` won't
        force a re-index — delete the doc first if you need a fresh tree.
        """
        return self._backend.add_document(self._name, file_path)

    def list_documents(self) -> list[DocumentInfo]:
        """List every document in this collection.

        Each item has ``doc_id``, ``doc_name``, ``doc_description``, ``doc_type``.
        """
        return self._backend.list_documents(self._name)

    def get_document(self, doc_id: str, include_text: bool = False) -> DocumentDetail:
        """Return a document's metadata plus its tree under ``structure``.

        ``include_text=True`` fills each node's text from cached pages (local
        backend only; can be large — avoid for LLM contexts). Raises
        ``DocumentNotFoundError`` if the doc_id is unknown.
        """
        return self._backend.get_document(self._name, doc_id, include_text=include_text)

    def get_document_structure(self, doc_id: str) -> list:
        """Return the document's hierarchical tree (a list of node dicts)."""
        return self._backend.get_document_structure(self._name, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> list[PageContent]:
        """Return content for specific pages.

        ``pages`` is a range/list spec: ``"5-7"``, ``"3,8"``, or ``"12"``.
        Each returned item has ``page`` and ``content`` (and ``images`` when
        present). For Markdown docs, "page" numbers map to line ranges.
        """
        return self._backend.get_page_content(self._name, doc_id, pages)

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and its stored files/artifacts.

        Raises ``DocumentNotFoundError`` if the doc_id is unknown.
        """
        self._backend.delete_document(self._name, doc_id)

    def query(self, question: str,
              doc_ids: str | list[str] | None = None,
              stream: bool = False) -> str | QueryStream:
        """Query documents in this collection.

        - stream=False: returns answer string (sync)
        - stream=True: returns async iterable of QueryEvent

        ``doc_ids`` can be a single doc id (``str``) or a list. ``None`` queries
        the entire collection (experimental).

        Usage:
            answer = col.query("question", doc_ids=doc_id)            # single
            answer = col.query("question", doc_ids=[d1, d2])          # multi
            async for event in col.query("question", doc_ids=doc_id, stream=True):
                ...

        Passing doc_ids=None queries the entire collection — this is
        experimental; emits a UserWarning unless PAGEINDEX_EXPERIMENTAL_MULTIDOC
        is set.
        """
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        elif doc_ids == []:
            raise ValueError(
                "doc_ids cannot be empty; pass None to query the whole collection"
            )
        if doc_ids is None:
            # One list_documents call serves both the empty-collection guard
            # (always) and the multi-doc warning (only when not acknowledged).
            docs = self._backend.list_documents(self._name)
            if not docs:
                raise ValueError(
                    f"Cannot query collection '{self._name}': it is empty. "
                    "Add documents with col.add(...) first."
                )
            if len(docs) > 1 and not _multidoc_acked():
                warnings.warn(_MULTIDOC_WARNING, UserWarning, stacklevel=2)
        if stream:
            return QueryStream(self._backend, self._name, question, doc_ids)
        return self._backend.query(self._name, question, doc_ids)
