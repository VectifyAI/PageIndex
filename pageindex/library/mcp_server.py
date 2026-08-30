"""stdio MCP server exposing the library to Claude Code.

Register once:  claude mcp add --scope user books -- <fork>/.venv/bin/books mcp
"""
from __future__ import annotations

from typing import Callable

from ..local_store import DocStore
from .config import LibraryConfig
from .digest import render_book_digest, render_node_digest
from .ingest import find_book
from .pages import parse_page_spec

MAX_PAGES_PER_CALL = 40


def _trim(node: dict, depth: int | None, level: int) -> dict:
    out = {"node_id": node.get("node_id"), "title": node.get("title", ""),
           "pages": f"{node.get('start_index')}-{node.get('end_index')}",
           "summary": node.get("summary", "")}
    if node.get("key_items"):
        out["key_items"] = node["key_items"]
    children = node.get("nodes") or []
    if children and (depth is None or level + 1 < depth):
        out["nodes"] = [_trim(c, depth, level + 1) for c in children]
    return out


def build_tools(cfg: LibraryConfig) -> dict[str, Callable]:
    store = DocStore(str(cfg.storage_path))

    def list_books() -> list[dict]:
        """List every indexed book with its one-paragraph description. Call this first."""
        rows = []
        for meta in sorted(store.list_metas(), key=lambda m: m["name"]):
            extra = meta.get("metadata") or {}
            rows.append({"doc_id": meta["id"], "name": meta["name"],
                         "title": extra.get("title"), "pages": meta.get("pageNum"),
                         "profile": extra.get("profile"), "status": meta.get("status"),
                         "description": meta.get("description")})
        return rows

    def get_structure(book: str, depth: int | None = None) -> dict:
        """A book's table of contents: node ids, titles, page ranges and routing
        summaries. `book` is a doc_id, file name or part of the title. Use depth=1
        or 2 first on long books, then drill into a node's children."""
        meta = find_book(store, book)
        tree = store.get_tree(meta["id"]) or []
        return {"book": (meta.get("metadata") or {}).get("title") or meta["name"],
                "doc_id": meta["id"], "description": meta.get("description"),
                "nodes": [_trim(n, depth, 0) for n in tree]}

    def get_pages(book: str, pages: str) -> str:
        """Full text of specific pages ("12", "3,7", "40-52"), at most 40 per call.
        Read a node's page range after choosing it from get_structure. Cite as
        (Book title, p. N)."""
        meta = find_book(store, book)
        wanted = parse_page_spec(pages, meta["pageNum"])
        if len(wanted) > MAX_PAGES_PER_CALL:
            raise ValueError(f"Request at most {MAX_PAGES_PER_CALL} pages per call")
        stored = store.get_pages(meta["id"]) or []
        return "\n\n".join(f"--- Page {n} ---\n{stored[n - 1].get('markdown', '')}"
                           for n in wanted)

    def get_digest(book: str, node_id: str | None = None) -> str:
        """Markdown digest of a whole book or of one node (falls back to routing
        summaries where no digest has been generated yet)."""
        meta = find_book(store, book)
        tree = store.get_tree(meta["id"]) or []
        if node_id is None:
            return render_book_digest(meta, tree)
        return render_node_digest(meta, tree, node_id)

    return {"list_books": list_books, "get_structure": get_structure,
            "get_pages": get_pages, "get_digest": get_digest}


def build_server(cfg: LibraryConfig):
    from mcp.server.mcpserver import MCPServer
    server = MCPServer("books", instructions=(
        "Personal book library. Start with list_books, then get_structure on the "
        "relevant book, then get_pages on tight page ranges. Always cite page numbers."))
    for name, func in build_tools(cfg).items():
        server.tool(name=name)(func)
    return server


def main() -> None:
    build_server(LibraryConfig.load()).run("stdio")
