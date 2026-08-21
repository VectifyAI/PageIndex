"""On-disk document store behind PageIndexClient's local mode."""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        # errors=: a lone surrogate (os.fsdecode'd path in metadata, an
        # LLM-written \ud83d escape) must not crash the store after a whole
        # indexing run — it lands as U+FFFD instead.
        with open(tmp, "w", encoding="utf-8", errors="replace") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError,
            PermissionError):
        return None
    except ValueError:
        logger.warning("Unreadable JSON at %s; treating it as absent", path)
        return None


def _is_safe_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and value not in ("", ".", "..")
        and os.path.basename(value) == value
        and "\\" not in value
    )


def _is_valid_meta(meta, doc_id: str) -> bool:
    if not isinstance(meta, dict) or meta.get("id") != doc_id:
        return False
    page_num = meta.get("pageNum")
    return (
        isinstance(meta.get("name"), str)
        and (meta.get("description") is None
             or isinstance(meta.get("description"), str))
        and isinstance(meta.get("status"), str)
        and isinstance(meta.get("createdAt"), str)
        and isinstance(page_num, int)
        and not isinstance(page_num, bool)
        and page_num >= 0
        and (meta.get("folderId") is None
             or isinstance(meta.get("folderId"), str))
        and (meta.get("metadata") is None
             or isinstance(meta.get("metadata"), dict))
        and (meta.get("mode") is None or isinstance(meta.get("mode"), str))
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
            pass

    @contextmanager
    def lock(self):
        """Cross-process mutex for check-then-write sequences (name
        uniquing before save). fcntl is absent on Windows, where the
        pre-existing best-effort behavior stays."""
        try:
            import fcntl
        except ImportError:
            yield
            return
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self._root / ".lock", "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

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
        if not _is_valid_meta(meta, doc_id):
            meta = self._read_manifest().get(doc_id)
        return meta if _is_valid_meta(meta, doc_id) else None

    def get_tree(self, doc_id: str) -> list | None:
        return self._read_doc_file(doc_id, "tree.json")

    def get_pages(self, doc_id: str) -> list | None:
        return self._read_doc_file(doc_id, "pages.json")

    def list_metas(self) -> list[dict]:
        if not self._docs.is_dir():
            return []
        with os.scandir(self._docs) as entries:
            dir_names = {entry.name for entry in entries
                         if entry.is_dir() and _is_safe_id(entry.name)}
        cached = self._read_manifest()
        fresh = {}
        for name in dir_names:
            if not (self._docs / name / "doc.json").is_file():
                continue
            meta = cached.get(name)
            if not _is_valid_meta(meta, name):
                meta = _read_json(self._docs / name / "doc.json")
            if _is_valid_meta(meta, name):
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
        except OSError:
            if not (doc_dir / "doc.json").is_dir():
                raise
            existed = False
        if doc_dir.is_dir():
            shutil.rmtree(doc_dir, ignore_errors=True)
        manifest = self._read_manifest()
        if manifest.pop(doc_id, None) is not None:
            self._write_manifest(manifest)
        return existed
