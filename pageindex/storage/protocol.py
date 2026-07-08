from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageEngine(Protocol):
    """Persistence contract for collections and their documents."""

    def create_collection(self, name: str) -> None:
        """Create a new collection; error if it already exists."""

    def get_or_create_collection(self, name: str) -> None:
        """Create the collection if absent; no-op if it already exists."""

    def list_collections(self) -> list[str]:
        """Return all collection names."""

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its documents."""

    def save_document(self, collection: str, doc_id: str, doc: dict) -> None:
        """Persist a document under a collection."""

    def find_document_by_hash(self, collection: str, file_hash: str) -> str | None:
        """Return the doc_id with this file hash in the collection, or None."""

    def get_document(self, collection: str, doc_id: str) -> dict:
        """Return a document's metadata."""

    def get_document_structure(self, collection: str, doc_id: str) -> list:
        """Return a document's tree structure."""

    def get_pages(self, collection: str, doc_id: str) -> list | None:
        """Return cached page content, or None if not cached."""

    def list_documents(self, collection: str) -> list[dict]:
        """Return metadata for all documents in a collection."""

    def delete_document(self, collection: str, doc_id: str) -> None:
        """Delete a single document from a collection."""

    def close(self) -> None:
        """Release any underlying resources (connections, handles)."""
