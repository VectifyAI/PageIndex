"""Render a book's tree (with digest / summary fields) to Markdown."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..utils import structure_to_list


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "untitled"


def _book_title(meta: dict) -> str:
    return (meta.get("metadata") or {}).get("title") or meta.get("name", "")


def _pages(node: dict) -> str:
    a, b = node.get("start_index"), node.get("end_index")
    return f"p. {a}" if a == b else f"pp. {a}–{b}"


def _body(node: dict) -> str:
    if node.get("digest"):
        return node["digest"].strip()
    if node.get("summary"):
        return f"_{node['summary'].strip()}_"
    return "_(no summary yet)_"


def _section(node: dict, level: int, out: list[str]) -> None:
    out.append(f"{'#' * min(level, 6)} {node.get('title', '')} ({_pages(node)})\n")
    out.append(_body(node) + "\n")
    if node.get("key_items"):
        out.append("Includes: " + "; ".join(node["key_items"]) + "\n")
    for child in node.get("nodes") or []:
        _section(child, level + 1, out)


def render_book_digest(meta: dict, tree: list[dict]) -> str:
    out = [f"# {_book_title(meta)}\n"]
    if meta.get("description"):
        out.append(meta["description"].strip() + "\n")
    out.append(f"_{meta.get('pageNum')} pages · source: {meta.get('name')}_\n")
    for node in tree:
        _section(node, 2, out)
    return "\n".join(out)


def render_node_digest(meta: dict, tree: list[dict], node_id: str) -> str:
    node = next((n for n in structure_to_list(tree) if n.get("node_id") == node_id), None)
    if node is None:
        raise KeyError(f"No node {node_id!r} in {_book_title(meta)}")
    out = [f"# {_book_title(meta)} — {node.get('title', '')} ({_pages(node)})\n",
           _body(node) + "\n"]
    if node.get("key_items"):
        out.append("Includes: " + "; ".join(node["key_items"]) + "\n")
    for child in node.get("nodes") or []:
        _section(child, 2, out)
    return "\n".join(out)


def write_digest(cfg, store, doc_id: str, node_id: str | None = None) -> Path:
    meta = store.get_meta(doc_id)
    if meta is None:
        raise KeyError(f"Unknown doc_id: {doc_id!r}")
    tree = store.get_tree(doc_id) or []
    folder = cfg.digests_dir / slugify(_book_title(meta))
    folder.mkdir(parents=True, exist_ok=True)
    if node_id is None:
        path = folder / "book.md"
        text = render_book_digest(meta, tree)
    else:
        text = render_node_digest(meta, tree, node_id)
        node = next(n for n in structure_to_list(tree) if n.get("node_id") == node_id)
        path = folder / f"{node_id}-{slugify(node.get('title', ''))}.md"
    path.write_text(text, encoding="utf-8")
    return path
