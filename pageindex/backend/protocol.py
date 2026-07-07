from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Any, AsyncIterator, runtime_checkable

from ..events import QueryEvent
from ..types import DocumentInfo, DocumentDetail, PageContent


@dataclass
class AgentTools:
    """Structured container for agent tool configuration (local mode only)."""
    function_tools: list[Any] = field(default_factory=list)
    mcp_servers: list[Any] = field(default_factory=list)


@runtime_checkable
class Backend(Protocol):
    # Collection management
    def create_collection(self, name: str) -> None: ...
    def get_or_create_collection(self, name: str) -> None: ...
    def list_collections(self) -> list[str]: ...
    def delete_collection(self, name: str) -> None: ...

    # Document management
    def add_document(self, collection: str, file_path: str) -> str: ...
    def get_document(self, collection: str, doc_id: str, include_text: bool = False) -> DocumentDetail: ...
    def get_document_structure(self, collection: str, doc_id: str) -> list: ...
    def get_page_content(self, collection: str, doc_id: str, pages: str) -> list[PageContent]: ...
    def list_documents(self, collection: str) -> list[DocumentInfo]: ...
    def delete_document(self, collection: str, doc_id: str) -> None: ...

    # Query — doc_ids accepts a single id or a list; implementations should
    # normalize internally (a bare str is treated as a single-element list).
    def query(self, collection: str, question: str,
              doc_ids: str | list[str] | None = None) -> str: ...
    # query_stream is an async generator: calling it returns an async iterator
    # WITHOUT awaiting, so it is declared as a plain def returning
    # AsyncIterator (not `async def`, which would be a coroutine).
    def query_stream(self, collection: str, question: str,
                     doc_ids: str | list[str] | None = None) -> AsyncIterator[QueryEvent]: ...


@runtime_checkable
class SupportsParserRegistration(Protocol):
    """Capability protocol: a backend that accepts custom document parsers
    (local mode). Cloud backends don't implement this."""
    def register_parser(self, parser: Any) -> None: ...
