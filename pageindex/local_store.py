"""On-disk document store behind PageIndexClient's local mode.

Layout under the storage directory:

    manifest.json             all documents' metadata in one file
    docs/<doc_id>/doc.json    document metadata (small; read by get)
    docs/<doc_id>/tree.json   the PageIndex tree structure
    docs/<doc_id>/pages.json  extracted page text: [{"page_index": 1, "markdown": ...}, ...]

Every file is written atomically (temp file, fsync, os.replace). ``doc.json``
is the existence marker in both directions: it is written last on save and
unlinked first on delete, so a document exists exactly while it is present.
Directories without it (a crashed save or delete) are ignored everywhere.

The manifest is a cache, never a second source of truth: writers update it
best-effort, and ``list_metas`` serves an entry only after confirming the
document's ``doc.json`` still exists (documents are immutable, so presence
implies the cached content is valid). Anything missing from the cache is
re-read from the doc.json files and the manifest rewritten. No locks
anywhere.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except json.JSONDecodeError:
        logger.warning("Unreadable JSON at %s; treating it as absent", path)
        return None


def _is_safe_id(value: str) -> bool:
    """Reject ids that could escape the store — callers pass arbitrary
    strings, and delete_document removes a directory tree."""
    return (
        isinstance(value, str)
        and value not in ("", ".", "..")
        and os.path.basename(value) == value
        and "\\" not in value
    )


class DocStore:
    def __init__(self, storage_dir: str):
        self._root = Path(storage_dir).expanduser()
        self._docs = self._root / "docs"
        self._manifest = self._root / "manifest.json"

    def _doc_dir(self, doc_id: str) -> Path | None:
        if not _is_safe_id(doc_id):
            return None
        return self._docs / doc_id

    # ── manifest cache ──
    def _read_manifest(self) -> dict:
        data = _read_json(self._manifest)
        docs = data.get("docs") if isinstance(data, dict) else None
        return docs if isinstance(docs, dict) else {}

    def _write_manifest(self, docs: dict) -> None:
        try:
            _write_json_atomic(self._manifest, {"docs": docs})
        except OSError:
            pass  # cache only — listings rebuild it from the doc.json files

    # ── documents ──
    def save_document(self, doc_id: str, meta: dict, tree: list, pages: list) -> None:
        doc_dir = self._doc_dir(doc_id)
        if doc_dir is None:
            raise ValueError(f"Invalid doc_id: {doc_id!r}")
        doc_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(doc_dir / "tree.json", tree)
        _write_json_atomic(doc_dir / "pages.json", pages)
        _write_json_atomic(doc_dir / "doc.json", meta)
        manifest = self._read_manifest()
        manifest[doc_id] = meta
        self._write_manifest(manifest)

    def _read_doc_file(self, doc_id: str, name: str):
        doc_dir = self._doc_dir(doc_id)
        if doc_dir is None or not (doc_dir / "doc.json").is_file():
            return None
        return _read_json(doc_dir / name)

    def get_meta(self, doc_id: str) -> dict | None:
        doc_dir = self._doc_dir(doc_id)
        if doc_dir is None or not (doc_dir / "doc.json").is_file():
            return None
        meta = _read_json(doc_dir / "doc.json")
        if meta is None:
            # doc.json exists but is unreadable — the manifest holds a copy
            meta = self._read_manifest().get(doc_id)
        return meta

    def get_tree(self, doc_id: str) -> list | None:
        return self._read_doc_file(doc_id, "tree.json")

    def get_pages(self, doc_id: str) -> list | None:
        return self._read_doc_file(doc_id, "pages.json")

    def list_metas(self) -> list[dict]:
        if not self._docs.is_dir():
            return []
        with os.scandir(self._docs) as entries:
            dir_names = {entry.name for entry in entries if entry.is_dir()}
        cached = self._read_manifest()
        fresh = {}
        for name in dir_names:
            if not (self._docs / name / "doc.json").is_file():
                continue
            meta = cached.get(name)
            if meta is None:
                meta = _read_json(self._docs / name / "doc.json")
            if meta is not None:
                fresh[name] = meta
        if fresh != cached:
            self._write_manifest(fresh)
        return list(fresh.values())

    def delete_document(self, doc_id: str) -> bool:
        doc_dir = self._doc_dir(doc_id)
        if doc_dir is None:
            return False
        try:
            (doc_dir / "doc.json").unlink()
            existed = True
        except (FileNotFoundError, NotADirectoryError):
            existed = False
        if doc_dir.is_dir():
            shutil.rmtree(doc_dir, ignore_errors=True)
        manifest = self._read_manifest()
        if manifest.pop(doc_id, None) is not None:
            self._write_manifest(manifest)
        return existed
