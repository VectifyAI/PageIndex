"""On-disk document store behind PageIndexClient's local mode.

Layout under the storage directory:

    manifest.json             all documents' metadata in one file, for
                              one-read listings and human inspection
    docs/<doc_id>/doc.json    document metadata (small; read by get)
    docs/<doc_id>/tree.json   the PageIndex tree structure
    docs/<doc_id>/pages.json  extracted page text: [{"page_index": 1, "markdown": ...}, ...]

Every file is written atomically (temp file + os.replace). ``doc.json`` is
written last, so its presence marks a completely stored document; directories
without it (e.g. from a crashed indexing run) are ignored everywhere.

The manifest is a cache, never a second source of truth: writers update it
best-effort, and ``list_metas`` trusts it only while its id set matches the
``docs/`` directory names (documents are immutable, so matching names imply
valid content). On any mismatch — a concurrently lost update, a crash, a
corrupt or deleted manifest — it is rebuilt from the doc.json files. No
locks anywhere.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError):
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
        try:
            data = _read_json(self._manifest)
        except ValueError:
            return {}
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
        return self._read_doc_file(doc_id, "doc.json")

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
        if set(cached) == dir_names:
            return list(cached.values())
        fresh = {}
        for name in dir_names:
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
        existed = (doc_dir / "doc.json").is_file()
        if doc_dir.is_dir():
            shutil.rmtree(doc_dir)
        manifest = self._read_manifest()
        if manifest.pop(doc_id, None) is not None:
            self._write_manifest(manifest)
        return existed
