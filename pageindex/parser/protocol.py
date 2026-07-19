from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ContentNode:
    """Universal content unit produced by parsers."""
    content: str
    tokens: int
    title: str | None = None
    index: int | None = None
    level: int | None = None
    images: list[dict] | None = None  # [{"path": str, "width": int, "height": int}, ...]


@dataclass
class ParsedDocument:
    """Unified parser output. Always a flat list of ContentNode."""
    doc_name: str
    nodes: list[ContentNode]
    # Doc-level fields merged into the stored document record at index time.
    # The built-in storage only persists keys it has columns for
    # (currently page_count / line_count); other keys are dropped.
    metadata: dict | None = None


@runtime_checkable
class DocumentParser(Protocol):
    def supported_extensions(self) -> list[str]:
        """Return the file extensions this parser handles (e.g. ['.pdf'])."""

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """Parse a file into a ParsedDocument (a flat list of ContentNode)."""
