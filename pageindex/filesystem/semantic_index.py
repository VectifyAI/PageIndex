from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import sqlite_vec


class SemanticIndexError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticIndexRecord:
    file_ref: str
    vector: list[float]
    text: str
    external_id: str | None = None
    source_type: str = ""
    source_path: str = ""
    title: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SemanticSearchResult:
    file_ref: str
    distance: float
    external_id: str | None
    source_type: str
    source_path: str
    title: str
    text_hash: str
    metadata: dict[str, Any]


class RebuildableSemanticIndex(Protocol):
    def reset(self, *, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        ...

    def upsert_many(self, records: list[SemanticIndexRecord]) -> int:
        ...

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        fetch_multiplier: int = 20,
    ) -> list[SemanticSearchResult]:
        ...

    def info(self) -> dict[str, Any]:
        ...


class SQLiteVecSemanticIndex:
    """Rebuildable local semantic index backed by sqlite-vec.

    This is intentionally separate from the PIFS catalog tables. The catalog
    remains source of truth; this file is a rebuildable recall index.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self, *, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        if dimension <= 0:
            raise SemanticIndexError("semantic index dimension must be positive")
        with self.connect() as conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS semantic_index_vec;
                DROP TABLE IF EXISTS semantic_index_docs;
                DROP TABLE IF EXISTS semantic_index_config;
                CREATE TABLE semantic_index_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE semantic_index_docs (
                    rowid INTEGER PRIMARY KEY,
                    file_ref TEXT NOT NULL UNIQUE,
                    external_id TEXT,
                    source_type TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    text_hash TEXT NOT NULL,
                    text_chars INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX idx_semantic_index_docs_file_ref
                  ON semantic_index_docs(file_ref);
                CREATE INDEX idx_semantic_index_docs_external_id
                  ON semantic_index_docs(external_id);
                CREATE INDEX idx_semantic_index_docs_source_type
                  ON semantic_index_docs(source_type);
                """
            )
            conn.execute(
                "CREATE VIRTUAL TABLE semantic_index_vec USING "
                f"vec0(source_type TEXT partition key, embedding float[{dimension}])"
            )
            config = {
                "dimension": str(dimension),
                "adapter": "sqlite-vec",
                "adapter_version": sqlite_vec.__version__,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            }
            conn.executemany(
                "INSERT INTO semantic_index_config(key, value) VALUES (?, ?)",
                sorted(config.items()),
            )
            conn.commit()

    def upsert_many(self, records: list[SemanticIndexRecord]) -> int:
        if not records:
            return 0
        dimension = self.dimension()
        with self.connect() as conn:
            inserted = 0
            for record in records:
                if len(record.vector) != dimension:
                    raise SemanticIndexError(
                        f"vector dimension mismatch for {record.file_ref}: "
                        f"expected {dimension}, got {len(record.vector)}"
                    )
                rowid = self._upsert_doc(conn, record)
                conn.execute("DELETE FROM semantic_index_vec WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO semantic_index_vec(rowid, source_type, embedding) VALUES (?, ?, ?)",
                    (
                        rowid,
                        record.source_type,
                        sqlite_vec.serialize_float32(record.vector),
                    ),
                )
                inserted += 1
            conn.commit()
            return inserted

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        fetch_multiplier: int = 20,
    ) -> list[SemanticSearchResult]:
        dimension = self.dimension()
        if len(vector) != dimension:
            raise SemanticIndexError(
                f"query vector dimension mismatch: expected {dimension}, got {len(vector)}"
            )
        fetch_k = min(4096, max(limit, limit * max(fetch_multiplier, 1)))
        source_types = _source_type_filters(filters or {})
        with self.connect() as conn:
            rows = []
            if source_types:
                for source_type in source_types:
                    rows.extend(
                        conn.execute(
                            """
                            SELECT
                                d.file_ref,
                                d.external_id,
                                d.source_type,
                                d.source_path,
                                d.title,
                                d.text_hash,
                                d.metadata_json,
                                v.distance
                            FROM semantic_index_vec v
                            JOIN semantic_index_docs d ON d.rowid = v.rowid
                            WHERE v.embedding MATCH ? AND k = ? AND v.source_type = ?
                            ORDER BY v.distance
                            """,
                            (sqlite_vec.serialize_float32(vector), fetch_k, source_type),
                        ).fetchall()
                    )
                rows.sort(key=lambda row: float(row["distance"]))
            else:
                rows = conn.execute(
                    """
                    SELECT
                        d.file_ref,
                        d.external_id,
                        d.source_type,
                        d.source_path,
                        d.title,
                        d.text_hash,
                        d.metadata_json,
                        v.distance
                    FROM semantic_index_vec v
                    JOIN semantic_index_docs d ON d.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                    """,
                    (sqlite_vec.serialize_float32(vector), fetch_k),
                ).fetchall()
        results: list[SemanticSearchResult] = []
        for row in rows:
            metadata = _json_obj(row["metadata_json"])
            if not _matches_filters(row, metadata, filters or {}):
                continue
            results.append(
                SemanticSearchResult(
                    file_ref=row["file_ref"],
                    distance=float(row["distance"]),
                    external_id=row["external_id"],
                    source_type=row["source_type"],
                    source_path=row["source_path"],
                    title=row["title"],
                    text_hash=row["text_hash"],
                    metadata=metadata,
                )
            )
            if len(results) >= limit:
                break
        return results

    def info(self) -> dict[str, Any]:
        with self.connect() as conn:
            config = {
                row["key"]: row["value"]
                for row in conn.execute(
                    "SELECT key, value FROM semantic_index_config ORDER BY key"
                ).fetchall()
            }
            count = conn.execute("SELECT COUNT(*) FROM semantic_index_docs").fetchone()[0]
        parsed_metadata: dict[str, Any]
        try:
            parsed_metadata = json.loads(config.get("metadata", "{}"))
        except json.JSONDecodeError:
            parsed_metadata = {}
        return {
            "db_path": str(self.db_path),
            "adapter": config.get("adapter", "sqlite-vec"),
            "adapter_version": config.get("adapter_version", ""),
            "dimension": int(config.get("dimension", "0") or 0),
            "document_count": count,
            "metadata": parsed_metadata,
        }

    def dimension(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT value FROM semantic_index_config WHERE key = 'dimension'"
            ).fetchone()
        if row is None:
            raise SemanticIndexError(
                f"semantic index is not initialized; call reset() first: {self.db_path}"
            )
        return int(row["value"])

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _upsert_doc(conn: sqlite3.Connection, record: SemanticIndexRecord) -> int:
        existing = conn.execute(
            "SELECT rowid FROM semantic_index_docs WHERE file_ref = ?",
            (record.file_ref,),
        ).fetchone()
        metadata_json = json.dumps(record.metadata or {}, ensure_ascii=False, sort_keys=True)
        text_hash = SQLiteVecSemanticIndex.text_hash(record.text)
        if existing is None:
            cursor = conn.execute(
                """
                INSERT INTO semantic_index_docs(
                    file_ref, external_id, source_type, source_path, title,
                    text_hash, text_chars, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.file_ref,
                    record.external_id,
                    record.source_type,
                    record.source_path,
                    record.title,
                    text_hash,
                    len(record.text),
                    metadata_json,
                ),
            )
            return int(cursor.lastrowid)
        rowid = int(existing["rowid"])
        conn.execute(
            """
            UPDATE semantic_index_docs
            SET external_id = ?,
                source_type = ?,
                source_path = ?,
                title = ?,
                text_hash = ?,
                text_chars = ?,
                metadata_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE rowid = ?
            """,
            (
                record.external_id,
                record.source_type,
                record.source_path,
                record.title,
                text_hash,
                len(record.text),
                metadata_json,
                rowid,
            ),
        )
        return rowid


def _json_obj(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _matches_filters(
    row: sqlite3.Row,
    metadata: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    for key, expected in filters.items():
        actual = row[key] if key in row.keys() else metadata.get(key)
        if isinstance(expected, list):
            if str(actual) not in {str(item) for item in expected}:
                return False
        elif str(actual) != str(expected):
            return False
    return True


def _source_type_filters(filters: dict[str, Any]) -> list[str]:
    value = filters.get("source_type")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []
