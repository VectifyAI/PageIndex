"""Add a PDF to the library: LLM-free tree, profile splitter, store, then
checkpointed summaries and a description."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from ..flash.api import _optimize as optimize_structure
from ..flash.main import extract_toc
from ..local_store import DocStore
from ..utils import structure_to_list, write_node_id
from .config import LibraryConfig
from .pages import leaf_spans
from .splitters import apply_diary_profile
from .summaries import describe_book, summarize_book


class IngestError(RuntimeError):
    pass


def file_doc_id(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return "pi-" + digest.hexdigest()[:16]


def _now_iso() -> str:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(microsecond=now.microsecond // 1000 * 1000).isoformat()


def add_book(pdf_path: str, cfg: LibraryConfig, *, profile: str | None = None,
             model: str | None = None, summaries: bool = True, force: bool = False,
             metadata: dict | None = None, log=print) -> dict:
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)
    profile = profile or cfg.profile
    model = model or cfg.index_model
    store = DocStore(str(cfg.storage_path))
    doc_id = file_doc_id(pdf_path)
    existing = store.get_meta(doc_id)
    if existing is not None and not force:
        log(f"already indexed as {doc_id} ({existing['name']}); use --force to redo")
        return {"doc_id": doc_id, "name": existing["name"],
                "title": (existing.get("metadata") or {}).get("title"),
                "nodes": len(structure_to_list(store.get_tree(doc_id) or [])),
                "pages": existing["pageNum"], "status": existing["status"],
                "summary": None, "long_leaves": []}
    if existing is not None and force:
        old_nodes = structure_to_list(store.get_tree(doc_id) or [])
        summary_count = sum(1 for n in old_nodes if n.get("summary"))
        digest_count = sum(1 for n in old_nodes if n.get("digest"))
        if summary_count or digest_count:
            log(f"--force discards {summary_count} summaries and {digest_count} digests "
                f"from the existing index")

    log(f"extracting structure from {os.path.basename(pdf_path)} …")
    result = extract_toc(pdf_path, use_embedded_toc=True)
    structure = result.get("structure") or []
    page_texts = result.get("page_texts") or []
    if not structure:
        raise IngestError("PageIndex Flash found no structure in this PDF; try a profile "
                          "or upstream mode='standard' (needs an API model).")
    optimize_structure(structure, page_texts, False, None)
    if profile == "diary":
        structure = apply_diary_profile(structure, page_texts)
    write_node_id(structure)
    long_leaves = [span for span in leaf_spans(structure)
                   if span[2] - span[1] + 1 > cfg.max_leaf_pages]
    for title, start, end in long_leaves:
        log(f"warning: leaf {title!r} spans {end - start + 1} pages ({start}-{end})")

    title = result.get("doc_title") or os.path.splitext(os.path.basename(pdf_path))[0]
    meta = {
        "id": doc_id, "name": os.path.basename(pdf_path), "description": None,
        "status": "indexed", "createdAt": _now_iso(), "pageNum": len(page_texts),
        "folderId": None, "mode": "flash",
        "metadata": {"title": title, "profile": profile, "source": pdf_path,
                     "sha256": doc_id[3:], "summary_tier_done": False,
                     "digest_tier_done": False, **(metadata or {})},
    }
    pages = [{"page_index": i + 1, "markdown": text} for i, text in enumerate(page_texts)]
    with store.lock():
        store.save_document(doc_id, meta, structure, pages)
    node_count = len(structure_to_list(structure))
    log(f"indexed {doc_id}: {node_count} nodes over {len(page_texts)} pages")

    stats = None
    status = "indexed"
    if summaries:
        log(f"summaries via {model} …")
        stats = summarize_book(store, doc_id, tier="summary", model=model, force=force)
        log(f"summaries: {stats['generated']} generated, {stats['skipped']} skipped, "
            f"{stats['failed']} failed")
        if stats["failed"] == 0:
            describe_book(store, doc_id, model=model)
            status = "completed"
            store.update_meta(doc_id, status=status)
    return {"doc_id": doc_id, "name": meta["name"], "title": title, "nodes": node_count,
            "pages": len(page_texts), "status": status, "summary": stats,
            "long_leaves": long_leaves}


def find_book(store: DocStore, query: str) -> dict:
    metas = store.list_metas()
    for meta in metas:
        if meta["id"] == query or meta["name"] == query:
            return meta
    needle = query.lower()
    hits = [m for m in metas
            if needle in m["name"].lower()
            or needle in ((m.get("metadata") or {}).get("title") or "").lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(f"No book matches {query!r}. Known: "
                          + ", ".join(sorted(m["name"] for m in metas)))
    raise LookupError(f"{query!r} is ambiguous: " + ", ".join(m["name"] for m in hits))
