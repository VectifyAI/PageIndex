"""On-disk document store behind PageIndexClient's local mode.

Layout under the storage directory:

    docs/<doc_id>/doc.json    document metadata (small; read by list/get)
    docs/<doc_id>/tree.json   the PageIndex tree structure
    docs/<doc_id>/pages.json  extracted page text: [{"page_index": 1, "markdown": ...}, ...]

Every file is written atomically (temp file + os.replace). ``doc.json`` is
written last, so its presence marks a completely stored document; directories
without it (e.g. from a crashed indexing run) are ignored everywhere.
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
    """True when the id is usable as a single path component under the store.

    Ids the store hands out are uuid4 strings, but callers can pass any string
    to lookups (and to delete_document, which removes a directory tree), so a
    traversal like ``..`` or ``a/b`` must never leave the storage directory.
    """
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

    def _doc_dir(self, doc_id: str) -> Path | None:
        if not _is_safe_id(doc_id):
            return None
        return self._docs / doc_id

    # ── documents ──
    def save_document(self, doc_id: str, meta: dict, tree: list, pages: list) -> None:
        doc_dir = self._doc_dir(doc_id)
        if doc_dir is None:
            raise ValueError(f"Invalid doc_id: {doc_id!r}")
        doc_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(doc_dir / "tree.json", tree)
        _write_json_atomic(doc_dir / "pages.json", pages)
        _write_json_atomic(doc_dir / "doc.json", meta)

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
        metas = []
        for entry in self._docs.iterdir():
            meta = _read_json(entry / "doc.json")
            if meta is not None:
                metas.append(meta)
        return metas

    def delete_document(self, doc_id: str) -> bool:
        doc_dir = self._doc_dir(doc_id)
        if doc_dir is None:
            return False
        existed = (doc_dir / "doc.json").is_file()
        if doc_dir.is_dir():
            shutil.rmtree(doc_dir)
        return existed
