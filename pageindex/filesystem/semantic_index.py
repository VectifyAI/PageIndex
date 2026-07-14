from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlite_vec

from ._embedding_identity import normalize_base_url, normalize_model
from ._sqlite_schema import sqlite_schema_signature


class SemanticIndexError(RuntimeError):
    pass


SCHEMA_VERSION = 2
SUMMARY_TABLES = {
    "semantic_index_config",
    "semantic_index_docs",
    "semantic_index_vec",
}
SUMMARY_CONFIG_KEYS = {"adapter", "adapter_version", "dimension", "metadata"}
VEC0_DECLARATION_RE = re.compile(
    r"""
    ^\s*CREATE\s+VIRTUAL\s+TABLE\s+[\"`\[]?semantic_index_vec[\"`\]]?
    \s+USING\s+vec0\s*\(\s*
    source_type\s+TEXT\s+PARTITION\s+KEY\s*,\s*
    embedding\s+FLOAT\s*\[\s*(\d+)\s*\]\s*
    (?:distance_metric\s*=\s*([A-Za-z0-9_-]+)\s*)?
    \)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class SemanticIndexRecord:
    file_ref: str
    vector: list[float]
    text: str
    external_id: str | None = None
    source_type: str = ""
    title: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SemanticSearchResult:
    file_ref: str
    distance: float
    external_id: str | None
    source_type: str
    title: str
    text_hash: str
    metadata: dict[str, Any]


class SQLiteVecSemanticIndex:
    """Rebuildable local semantic index backed by sqlite-vec.

    This is intentionally separate from the PIFS catalog tables. The catalog
    remains source of truth; this file is a rebuildable recall index.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def reset(self, *, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        if dimension <= 0:
            raise SemanticIndexError("semantic index dimension must be positive")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            self._create_schema(conn, dimension=dimension, metadata=metadata or {})

    def validate(self, expected_identity: dict[str, Any]) -> dict[str, Any]:
        info = self.validate_schema()
        if info["metadata"] != expected_identity:
            raise SemanticIndexError(
                "Incompatible PIFS Summary Embedding Profile; migrate the workspace or "
                "use the base URL, model, and dimensions that created this projection."
            )
        if int(info["dimension"]) != int(expected_identity["dimensions"]):
            raise SemanticIndexError(
                "Incompatible PIFS Summary Projection dimensions; migrate the workspace "
                "or use the matching embedding profile."
            )
        return info

    def validate_schema(self) -> dict[str, Any]:
        try:
            with self.connect(read_only=True) as conn:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                tables = self._logical_tables(conn)
                config = self._config(conn)
                info = self._info_from_connection(conn, config)
                actual = self._schema_signature(conn, tables)
                dimension = int(info["dimension"])
                self._validate_projection_rows(conn, dimension=dimension)
            metadata = info["metadata"]
            if (
                version != SCHEMA_VERSION
                or tables != SUMMARY_TABLES
                or set(config) != SUMMARY_CONFIG_KEYS
                or config["adapter"] != "sqlite-vec"
                or not config["adapter_version"]
                or dimension <= 0
                or set(metadata) != {"base_url", "model", "dimensions"}
                or not isinstance(metadata["base_url"], str)
                or not metadata["base_url"].strip()
                or normalize_base_url(metadata["base_url"]) != metadata["base_url"]
                or not isinstance(metadata["model"], str)
                or not metadata["model"].strip()
                or normalize_model(metadata["model"]) != metadata["model"]
                or isinstance(metadata["dimensions"], bool)
                or int(metadata["dimensions"]) != dimension
                or actual["vec0"] is None
                or actual["vec0"]["dimension"] != dimension
                or actual["vec0"]["distance_metric"] != "l2"
            ):
                raise self._incompatible_schema_error()
            with self._memory_connection() as expected_conn:
                self._create_schema(
                    expected_conn,
                    dimension=dimension,
                    metadata=metadata,
                )
                expected = self._schema_signature(
                    expected_conn,
                    self._logical_tables(expected_conn),
                )
            if actual != expected:
                raise self._incompatible_schema_error()
        except SemanticIndexError:
            raise
        except (json.JSONDecodeError, sqlite3.Error, TypeError, ValueError) as exc:
            raise self._incompatible_schema_error() from exc
        return info

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

    def delete_file_refs(self, file_refs: list[str]) -> int:
        refs = [str(file_ref) for file_ref in file_refs if str(file_ref)]
        if not refs:
            return 0
        placeholders = ", ".join("?" for _ in refs)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT rowid
                FROM semantic_index_docs
                WHERE file_ref IN ({placeholders})
                """,
                refs,
            ).fetchall()
            rowids = [int(row["rowid"]) for row in rows]
            if not rowids:
                return 0
            rowid_placeholders = ", ".join("?" for _ in rowids)
            conn.execute(
                f"DELETE FROM semantic_index_vec WHERE rowid IN ({rowid_placeholders})",
                rowids,
            )
            conn.execute(
                f"DELETE FROM semantic_index_docs WHERE rowid IN ({rowid_placeholders})",
                rowids,
            )
            conn.commit()
        return len(rowids)

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
        raw_filters = filters or {}
        source_types = _source_type_filters(raw_filters)
        file_refs = _file_ref_filters(raw_filters)
        if file_refs == []:
            return []
        with self.connect() as conn:
            if file_refs is not None:
                _install_file_ref_filter_table(conn, file_refs)
            rows = []
            if source_types:
                for source_type in source_types:
                    fetch_k = self._search_fetch_k(
                        conn,
                        limit,
                        fetch_multiplier,
                        exact_file_ref_filter=file_refs is not None,
                        source_type=source_type,
                    )
                    if fetch_k <= 0:
                        continue
                    rows.extend(
                        conn.execute(
                            f"""
                            SELECT
                                d.file_ref,
                                d.external_id,
                                d.source_type,
                                d.title,
                                d.text_hash,
                                d.metadata_json,
                                v.distance
                            FROM semantic_index_vec v
                            JOIN semantic_index_docs d ON d.rowid = v.rowid
                            WHERE v.embedding MATCH ? AND k = ? AND v.source_type = ?
                              {_file_ref_filter_sql(file_refs)}
                            ORDER BY v.distance
                            """,
                            (sqlite_vec.serialize_float32(vector), fetch_k, source_type),
                        ).fetchall()
                    )
                rows.sort(key=lambda row: float(row["distance"]))
            else:
                fetch_k = self._search_fetch_k(
                    conn,
                    limit,
                    fetch_multiplier,
                    exact_file_ref_filter=file_refs is not None,
                )
                if fetch_k <= 0:
                    return []
                rows = conn.execute(
                    f"""
                    SELECT
                        d.file_ref,
                        d.external_id,
                        d.source_type,
                        d.title,
                        d.text_hash,
                        d.metadata_json,
                        v.distance
                    FROM semantic_index_vec v
                    JOIN semantic_index_docs d ON d.rowid = v.rowid
                    WHERE v.embedding MATCH ? AND k = ?
                      {_file_ref_filter_sql(file_refs)}
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
                    title=row["title"],
                    text_hash=row["text_hash"],
                    metadata=metadata,
                )
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _search_fetch_k(
        conn: sqlite3.Connection,
        limit: int,
        fetch_multiplier: int,
        *,
        exact_file_ref_filter: bool,
        source_type: str | None = None,
    ) -> int:
        if exact_file_ref_filter:
            where = []
            params: list[Any] = []
            if source_type is not None:
                where.append("source_type = ?")
                params.append(source_type)
            where_sql = "WHERE " + " AND ".join(where) if where else ""
            return int(
                conn.execute(
                    f"SELECT COUNT(*) FROM semantic_index_docs {where_sql}",
                    params,
                ).fetchone()[0]
            )
        return min(4096, max(limit, limit * max(fetch_multiplier, 1)))

    def info(self) -> dict[str, Any]:
        with self.connect(read_only=True) as conn:
            return self._info_from_connection(conn, self._config(conn))

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

    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            conn = sqlite3.connect(
                f"{self.db_path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self._load_extension(conn)
        return conn

    @staticmethod
    def _load_extension(conn: sqlite3.Connection) -> None:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    @classmethod
    def _memory_connection(cls) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        cls._load_extension(conn)
        return conn

    @staticmethod
    def _create_schema(
        conn: sqlite3.Connection,
        *,
        dimension: int,
        metadata: dict[str, Any],
    ) -> None:
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
                title TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL,
                text_chars INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
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
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        }
        conn.executemany(
            "INSERT INTO semantic_index_config(key, value) VALUES (?, ?)",
            sorted(config.items()),
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    @staticmethod
    def _logical_tables(conn: sqlite3.Connection) -> set[str]:
        return {
            str(row[1])
            for row in conn.execute("PRAGMA table_list")
            if row[2] in {"table", "virtual"}
            and not str(row[1]).startswith("sqlite_")
            and not str(row[1]).startswith("semantic_index_vec_")
        }

    @staticmethod
    def _config(conn: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in conn.execute(
                "SELECT key, value FROM semantic_index_config ORDER BY key"
            )
        }

    def _info_from_connection(
        self,
        conn: sqlite3.Connection,
        config: dict[str, str],
    ) -> dict[str, Any]:
        metadata = json.loads(config.get("metadata", "{}"))
        if not isinstance(metadata, dict):
            raise self._incompatible_schema_error()
        return {
            "db_path": str(self.db_path),
            "adapter": config.get("adapter", ""),
            "adapter_version": config.get("adapter_version", ""),
            "dimension": int(config.get("dimension", "0") or 0),
            "document_count": int(
                conn.execute("SELECT COUNT(*) FROM semantic_index_docs").fetchone()[0]
            ),
            "metadata": metadata,
        }

    @staticmethod
    def _schema_signature(
        conn: sqlite3.Connection,
        tables: set[str],
    ) -> dict[str, Any]:
        signature = sqlite_schema_signature(conn, tables)
        vec_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'semantic_index_vec'"
        ).fetchone()
        match = VEC0_DECLARATION_RE.fullmatch(str(vec_row[0] if vec_row else ""))
        signature["vec0"] = (
            {
                "partition_key": ("source_type", "text"),
                "embedding_type": "float",
                "dimension": int(match.group(1)),
                "distance_metric": str(match.group(2) or "l2").lower(),
            }
            if match
            else None
        )
        return signature

    @classmethod
    def _validate_projection_rows(
        cls,
        conn: sqlite3.Connection,
        *,
        dimension: int,
    ) -> None:
        docs = {
            int(row["rowid"]): str(row["source_type"])
            for row in conn.execute(
                "SELECT rowid, source_type FROM semantic_index_docs"
            )
        }
        vectors: dict[int, str] = {}
        expected_bytes = dimension * 4
        for row in conn.execute(
            "SELECT rowid, source_type, embedding FROM semantic_index_vec"
        ):
            rowid = int(row["rowid"])
            if len(bytes(row["embedding"])) != expected_bytes:
                raise cls._incompatible_schema_error()
            vectors[rowid] = str(row["source_type"])
        if docs != vectors:
            raise cls._incompatible_schema_error()

    @staticmethod
    def _incompatible_schema_error() -> SemanticIndexError:
        return SemanticIndexError(
            "Incompatible PIFS Summary Projection schema; migrate this workspace "
            "with pifs-data/scripts/migrate_pifs_workspace.py before opening it."
        )

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
                    file_ref, external_id, source_type, title,
                    text_hash, text_chars, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.file_ref,
                    record.external_id,
                    record.source_type,
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
        actual_key = "file_ref" if key == "file_refs" else key
        actual = row[actual_key] if actual_key in row.keys() else metadata.get(actual_key)
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


def _file_ref_filters(filters: dict[str, Any]) -> list[str] | None:
    if "file_ref" in filters:
        value = filters.get("file_ref")
    elif "file_refs" in filters:
        value = filters.get("file_refs")
    else:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _install_file_ref_filter_table(conn: sqlite3.Connection, file_refs: list[str]) -> None:
    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS semantic_index_filter_file_refs (
            file_ref TEXT PRIMARY KEY
        )
        """
    )
    conn.execute("DELETE FROM semantic_index_filter_file_refs")
    conn.executemany(
        "INSERT OR IGNORE INTO semantic_index_filter_file_refs(file_ref) VALUES (?)",
        [(file_ref,) for file_ref in file_refs],
    )


def _file_ref_filter_sql(file_refs: list[str] | None) -> str:
    if file_refs is None:
        return ""
    return (
        "AND EXISTS ("
        "SELECT 1 FROM semantic_index_filter_file_refs scope_refs "
        "WHERE scope_refs.file_ref = d.file_ref"
        ")"
    )
