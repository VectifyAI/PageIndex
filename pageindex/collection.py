# pageindex/collection.py
from __future__ import annotations
import os
import warnings
from typing import AsyncIterator
from .events import QueryEvent
from .backend.protocol import Backend


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
        return self._backend.add_document(self._name, file_path)

    def list_documents(self) -> list[dict]:
        return self._backend.list_documents(self._name)

    def get_document(self, doc_id: str, include_text: bool = False) -> dict:
        return self._backend.get_document(self._name, doc_id, include_text=include_text)

    def get_document_structure(self, doc_id: str) -> list:
        return self._backend.get_document_structure(self._name, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> list:
        return self._backend.get_page_content(self._name, doc_id, pages)

    def delete_document(self, doc_id: str) -> None:
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
