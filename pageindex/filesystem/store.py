from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from ._sqlite_schema import regular_table_names, sqlite_schema_signature
from .types import FileEntry, MetadataField

SCHEMA_VERSION = 2
CATALOG_TABLES = {
    "files",
    "folders",
    "file_folders",
    "metadata_fields",
    "metadata_values",
}


class SQLiteFileSystemStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace / "filesystem.sqlite"
        self.text_dir = self.workspace / "artifacts" / "text"
        self.raw_dir = self.workspace / "artifacts" / "raw"
        self.pageindex_client_dir = self.workspace / "artifacts" / "pageindex_client"
        self.initialize_schema()
        for path in (self.text_dir, self.raw_dir, self.pageindex_client_dir):
            path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize_schema(self) -> None:
        if self.db_path.exists() or self.db_path.is_symlink():
            self.validate_existing_database(self.db_path)
            return
        with self.connect() as conn:
            self._create_current_schema(conn)
            self.ensure_folder(conn, "/")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @classmethod
    def validate_existing_database(cls, db_path: str | Path) -> None:
        db_path = Path(db_path)
        if not db_path.is_file() or db_path.stat().st_size <= 0:
            raise cls._incompatible_schema_error()
        try:
            with cls._readonly_connection(db_path) as conn:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                actual = cls._schema_signature(conn)
            with sqlite3.connect(":memory:") as expected_conn:
                expected_conn.row_factory = sqlite3.Row
                cls._create_current_schema(expected_conn)
                expected = cls._schema_signature(expected_conn)
        except sqlite3.Error as exc:
            raise cls._incompatible_schema_error() from exc
        if version != SCHEMA_VERSION or actual != expected:
            raise cls._incompatible_schema_error()
        from ._workspace_consistency import validate_catalog_root

        validate_catalog_root(db_path)

    @staticmethod
    def _create_current_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_ref TEXT PRIMARY KEY,
                external_id TEXT,
                storage_uri TEXT NOT NULL,
                title TEXT NOT NULL,
                descriptor TEXT NOT NULL,
                content_type TEXT NOT NULL,
                source_type TEXT,
                fingerprint TEXT NOT NULL,
                text_artifact_path TEXT NOT NULL,
                raw_artifact_path TEXT,
                pageindex_doc_id TEXT,
                pageindex_tree_status TEXT NOT NULL DEFAULT 'not_built',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                metadata_status_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                deleted_at TEXT
            );

            CREATE TABLE IF NOT EXISTS folders (
                folder_id TEXT PRIMARY KEY,
                parent_id TEXT,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT 'manual',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(parent_id) REFERENCES folders(folder_id)
            );

            CREATE TABLE IF NOT EXISTS file_folders (
                file_ref TEXT NOT NULL,
                folder_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (file_ref, folder_id),
                FOREIGN KEY(file_ref) REFERENCES files(file_ref) ON DELETE CASCADE,
                FOREIGN KEY(folder_id) REFERENCES folders(folder_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS metadata_fields (
                field_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            );

            CREATE TABLE IF NOT EXISTS metadata_values (
                file_ref TEXT NOT NULL,
                field_id TEXT NOT NULL,
                value_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(file_ref) REFERENCES files(file_ref) ON DELETE CASCADE,
                FOREIGN KEY(field_id) REFERENCES metadata_fields(field_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_files_external_id ON files(external_id);
            CREATE INDEX IF NOT EXISTS idx_files_source_type ON files(source_type);
            CREATE INDEX IF NOT EXISTS idx_folders_parent_id ON folders(parent_id);
            CREATE INDEX IF NOT EXISTS idx_file_folders_folder ON file_folders(folder_id);
            CREATE INDEX IF NOT EXISTS idx_metadata_values_field_text ON metadata_values(field_id, value_text);
            """
        )

    @staticmethod
    def _readonly_connection(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _schema_signature(conn: sqlite3.Connection) -> dict[str, Any]:
        tables = regular_table_names(conn)
        return sqlite_schema_signature(conn, tables)

    @staticmethod
    def _incompatible_schema_error() -> RuntimeError:
        return RuntimeError(
            "Incompatible PIFS catalog schema; migrate this workspace with "
            "pifs-data/scripts/migrate_pifs_workspace.py before opening it "
            f"(expected exact schema version {SCHEMA_VERSION} with tables "
            f"{sorted(CATALOG_TABLES)})."
        )

    @staticmethod
    def _json_object(value: Any) -> dict[str, Any]:
        try:
            parsed = json.loads(value or "{}") if isinstance(value, str) else value
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def insert_file(self, record: dict[str, Any]) -> None:
        self.insert_files([record])

    def insert_files(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self.connect() as conn:
            conn.execute("PRAGMA temp_store = MEMORY")
            folder_cache: dict[tuple[str, str], str] = {}
            file_rows = []
            membership_rows = []
            file_ref_rows = []
            metadata_rows = []
            pending_folder_titles: dict[tuple[str, str], str] = {}
            metadata_field_ids = {
                row["name"]: row["field_id"]
                for row in conn.execute(
                    "SELECT name, field_id FROM metadata_fields"
                ).fetchall()
            }
            for record in records:
                folder_cache_key = (record["folder_path"], record.get("folder_kind", "manual"))
                folder_id = folder_cache.get(folder_cache_key)
                if folder_id is None:
                    folder_id = self.ensure_folder(
                        conn,
                        record["folder_path"],
                        kind=record.get("folder_kind", "manual"),
                    )
                    folder_cache[folder_cache_key] = folder_id
                self._ensure_title_available_in_folder(
                    conn,
                    folder_id=folder_id,
                    file_ref=record["file_ref"],
                    title=record["title"],
                )
                title_key = (folder_id, str(record["title"]))
                existing_file_ref = pending_folder_titles.get(title_key)
                if existing_file_ref and existing_file_ref != record["file_ref"]:
                    target = self._virtual_file_target(conn, folder_id, str(record["title"]))
                    raise FileExistsError(f"File already exists at {target}")
                pending_folder_titles[title_key] = record["file_ref"]
                file_rows.append(self._file_insert_values(record))
                membership_rows.append(
                    (
                        record["file_ref"],
                        folder_id,
                        json.dumps(record.get("folder_metadata") or {}, ensure_ascii=False),
                    )
                )
                file_ref_rows.append((record["file_ref"],))
                metadata_rows.extend(
                    self._metadata_insert_values(
                        record["file_ref"],
                        record.get("indexed_metadata", record["metadata"]),
                        metadata_field_ids,
                    )
                )
            conn.executemany(self._file_insert_sql(), file_rows)
            conn.executemany(
                """
                INSERT OR REPLACE INTO file_folders(file_ref, folder_id, metadata_json)
                VALUES (?, ?, ?)
                """,
                membership_rows,
            )
            conn.executemany("DELETE FROM metadata_values WHERE file_ref = ?", file_ref_rows)
            if metadata_rows:
                conn.executemany(
                    """
                    INSERT INTO metadata_values(
                        file_ref, field_id, value_text
                    ) VALUES (?, ?, ?)
                    """,
                    metadata_rows,
                )

    @staticmethod
    def _file_insert_sql() -> str:
        columns = [
            "file_ref",
            "external_id",
            "storage_uri",
            "title",
            "descriptor",
            "content_type",
            "source_type",
            "fingerprint",
            "text_artifact_path",
            "raw_artifact_path",
            "pageindex_doc_id",
            "pageindex_tree_status",
            "metadata_json",
            "metadata_status_json",
        ]
        columns.extend(["deleted_at", "updated_at"])
        placeholders = ", ".join(["?"] * (len(columns) - 2) + ["NULL", "CURRENT_TIMESTAMP"])
        update_assignments = [
            f"{column} = excluded.{column}"
            for column in columns
            if column not in {"file_ref", "deleted_at", "updated_at"}
        ]
        update_assignments.append("updated_at = CURRENT_TIMESTAMP")
        return f"""
            INSERT INTO files ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(file_ref) DO UPDATE SET
                {", ".join(update_assignments)}
        """

    @staticmethod
    def _file_insert_values(record: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = [
            record["file_ref"],
            record["external_id"],
            record["storage_uri"],
            record["title"],
            record["descriptor"],
            record["content_type"],
            record["source_type"],
            record["fingerprint"],
            record["text_artifact_path"],
            record["raw_artifact_path"],
            record.get("pageindex_doc_id"),
            record.get("pageindex_tree_status", "not_built"),
            record["metadata_json"],
            record.get("metadata_status_json", "{}"),
        ]
        return tuple(values)

    def _metadata_insert_values(
        self,
        file_ref: str,
        metadata: dict[str, Any],
        metadata_field_ids: dict[str, str],
    ) -> list[tuple[Any, ...]]:
        values = []
        for name, value in metadata.items():
            if not self._valid_field_name(name):
                continue
            field_id = metadata_field_ids.get(name)
            if field_id is None:
                continue
            for item in self._metadata_value_items(value):
                values.append(
                    (
                        file_ref,
                        field_id,
                        item,
                    )
                )
        return values

    def create_folder(
        self,
        path: str,
        *,
        kind: str = "manual",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.connect() as conn:
            return self.ensure_folder(
                conn,
                path,
                kind=kind,
                description=description,
                metadata=metadata,
            )

    def attach_file_to_folder(
        self,
        file_ref: str,
        folder_path_or_id: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            resolved_file_ref = self._resolve_file_ref(conn, file_ref)
            folder_id = self._resolve_or_create_folder(conn, folder_path_or_id)
            self._ensure_title_available_in_folder(
                conn,
                folder_id=folder_id,
                file_ref=resolved_file_ref,
                title=self._file_title(conn, resolved_file_ref),
            )
            conn.execute(
                """
                INSERT INTO file_folders(file_ref, folder_id, metadata_json)
                VALUES (?, ?, ?)
                ON CONFLICT(file_ref, folder_id) DO UPDATE SET
                    metadata_json = excluded.metadata_json
                """,
                (
                    resolved_file_ref,
                    folder_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )

    def attach_files_to_folders(self, items: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            for item in items:
                resolved_file_ref = self._resolve_file_ref(conn, item["file_ref"])
                folder_id = self._resolve_or_create_folder(conn, item["folder"])
                self._ensure_title_available_in_folder(
                    conn,
                    folder_id=folder_id,
                    file_ref=resolved_file_ref,
                    title=self._file_title(conn, resolved_file_ref),
                )
                conn.execute(
                    """
                    INSERT INTO file_folders(file_ref, folder_id, metadata_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(file_ref, folder_id) DO UPDATE SET
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        resolved_file_ref,
                        folder_id,
                        json.dumps(item.get("metadata") or {}, ensure_ascii=False),
                    ),
                )

    def membership_display_name(self, file_ref: str, folder_path: str) -> str | None:
        folder_path = normalize_path(folder_path)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT ff.metadata_json, f.title
                FROM file_folders ff
                JOIN folders fo ON fo.folder_id = ff.folder_id
                JOIN files f ON f.file_ref = ff.file_ref
                WHERE ff.file_ref = ?
                  AND fo.path = ?
                  AND f.deleted_at IS NULL
                LIMIT 1
                """,
                (file_ref, folder_path),
            ).fetchone()
        if row is None:
            return None
        metadata = self._json_object(row["metadata_json"])
        return str(metadata.get("display_name") or row["title"] or "").strip() or None

    def _ensure_title_available_in_folder(
        self,
        conn: sqlite3.Connection,
        *,
        folder_id: str,
        file_ref: str,
        title: str,
    ) -> None:
        row = conn.execute(
            """
            SELECT f.file_ref, fo.path
            FROM files f
            JOIN file_folders ff ON ff.file_ref = f.file_ref
            JOIN folders fo ON fo.folder_id = ff.folder_id
            WHERE f.deleted_at IS NULL
              AND ff.folder_id = ?
              AND f.title = ?
              AND f.file_ref != ?
            LIMIT 1
            """,
            (folder_id, title, file_ref),
        ).fetchone()
        if row:
            raise FileExistsError(
                f"File already exists at {self._virtual_file_target(conn, folder_id, title)}"
            )

    @staticmethod
    def _virtual_file_target(
        conn: sqlite3.Connection,
        folder_id: str,
        title: str,
    ) -> str:
        row = conn.execute(
            "SELECT path FROM folders WHERE folder_id = ?",
            (folder_id,),
        ).fetchone()
        folder_path = normalize_path(row["path"] if row else "/")
        return f"/{title}" if folder_path == "/" else f"{folder_path}/{title}"

    @staticmethod
    def _file_title(conn: sqlite3.Connection, file_ref: str) -> str:
        row = conn.execute(
            "SELECT title FROM files WHERE file_ref = ? AND deleted_at IS NULL",
            (file_ref,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown file target: {file_ref}")
        return str(row["title"])

    def replace_metadata_values(
        self,
        conn: sqlite3.Connection,
        file_ref: str,
        metadata: dict[str, Any],
    ) -> None:
        conn.execute("DELETE FROM metadata_values WHERE file_ref = ?", (file_ref,))
        for name, value in metadata.items():
            if not self._valid_field_name(name):
                continue
            field_id = self._registered_field_id(conn, name)
            if field_id is None:
                continue
            for item in self._metadata_value_items(value):
                conn.execute(
                    """
                    INSERT INTO metadata_values(
                        file_ref, field_id, value_text
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        file_ref,
                        field_id,
                        item,
                    ),
                )

    @staticmethod
    def _registered_field_id(conn: sqlite3.Connection, name: str) -> str | None:
        row = conn.execute(
            """
            SELECT field_id
            FROM metadata_fields
            WHERE name = ?
            """,
            (name,),
        ).fetchone()
        return None if row is None else row["field_id"]

    def upsert_metadata_fields(
        self,
        fields: Iterable[MetadataField],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = conn is None
        if conn is None:
            conn = self.connect()
        try:
            for field in fields:
                conn.execute(
                    """
                    INSERT INTO metadata_fields(
                        field_id, name, description, source, updated_at
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(name) DO UPDATE SET
                        description = excluded.description,
                        source = excluded.source,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        self.field_id(field.name),
                        field.name,
                        field.description,
                        field.source,
                    ),
                )
            if owns_connection:
                conn.commit()
        finally:
            if owns_connection:
                conn.close()

    def metadata_field_exists(self, name: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM metadata_fields WHERE name = ?",
                (name,),
            ).fetchone()
        return row is not None

    def delete_metadata_field_if_unreferenced(self, name: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT field_id FROM metadata_fields WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                return False
            referenced = conn.execute(
                "SELECT 1 FROM metadata_values WHERE field_id = ? LIMIT 1",
                (row["field_id"],),
            ).fetchone()
            if referenced is not None:
                return False
            cursor = conn.execute(
                "DELETE FROM metadata_fields WHERE field_id = ?",
                (row["field_id"],),
            )
            return cursor.rowcount > 0

    def list_metadata_fields(self) -> list[MetadataField]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT name, description, source
                FROM metadata_fields
                ORDER BY name
                """
            ).fetchall()
        return [
            MetadataField(
                name=row["name"],
                description=row["description"],
                source=row["source"],
            )
            for row in rows
        ]

    def list_folder(
        self,
        path: str = "/",
        recursive: bool = False,
        limit: int = 100,
        max_depth: int | None = None,
    ) -> dict[str, Any]:
        path = normalize_path(path)
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        with self.connect() as conn:
            folder = self._folder_by_path(conn, path)
            if folder is None:
                raise KeyError(f"Unknown folder path: {path}")
            if recursive:
                folder_depth_clause = ""
                folder_depth_params: list[Any] = []
                if max_depth is not None:
                    if max_depth == 0:
                        folder_depth_clause = "AND 0"
                    else:
                        folder_depth_clause = (
                            f"AND ({self._folder_depth_sql('fo.path')} - ?) <= ?"
                        )
                        folder_depth_params = [self._folder_depth(path), max_depth]
                folder_rows = conn.execute(
                    f"""
                    SELECT
                        fo.folder_id,
                        fo.parent_id,
                        fo.name,
                        fo.path,
                        fo.description,
                        fo.kind,
                        fo.metadata_json,
                        fo.created_at,
                        fo.updated_at,
                        (
                            SELECT COUNT(DISTINCT child_ff.file_ref)
                            FROM file_folders child_ff
                            JOIN files child_file
                              ON child_file.file_ref = child_ff.file_ref
                             AND child_file.deleted_at IS NULL
                            WHERE child_ff.folder_id = fo.folder_id
                        ) AS file_count,
                        (
                            SELECT COUNT(*)
                            FROM folders child_folder
                            WHERE child_folder.parent_id = fo.folder_id
                        ) AS children_count
                    FROM folders fo
                    WHERE fo.path != ? AND (fo.path LIKE ? ESCAPE '\\')
                      {folder_depth_clause}
                    ORDER BY fo.path
                    LIMIT ?
                    """,
                    (path, self._descendant_like(path), *folder_depth_params, limit),
                ).fetchall()
                file_rows = self._file_rows_for_scope(
                    conn,
                    path,
                    True,
                    limit,
                    max_depth=max_depth,
                )
            else:
                folder_rows = conn.execute(
                    """
                    SELECT
                        fo.folder_id,
                        fo.parent_id,
                        fo.name,
                        fo.path,
                        fo.description,
                        fo.kind,
                        fo.metadata_json,
                        fo.created_at,
                        fo.updated_at,
                        (
                            SELECT COUNT(DISTINCT child_ff.file_ref)
                            FROM file_folders child_ff
                            JOIN files child_file
                              ON child_file.file_ref = child_ff.file_ref
                             AND child_file.deleted_at IS NULL
                            WHERE child_ff.folder_id = fo.folder_id
                        ) AS file_count,
                        (
                            SELECT COUNT(*)
                            FROM folders child_folder
                            WHERE child_folder.parent_id = fo.folder_id
                        ) AS children_count
                    FROM folders fo
                    WHERE fo.parent_id = ?
                    ORDER BY lower(fo.name), fo.name
                    LIMIT ?
                    """,
                    (folder["folder_id"], limit),
                ).fetchall()
                file_rows = self._file_rows_for_scope(conn, path, False, limit)
        return {
            "folders": [self._folder_row_to_dict(row) for row in folder_rows],
            "files": [self._file_summary(row) for row in file_rows],
        }

    def folder_info(self, path: str = "/") -> dict[str, Any]:
        path = normalize_path(path)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    fo.folder_id,
                    fo.parent_id,
                    fo.name,
                    fo.path,
                    fo.description,
                    fo.kind,
                    fo.metadata_json,
                    fo.created_at,
                    fo.updated_at,
                    (
                        SELECT COUNT(DISTINCT child_ff.file_ref)
                        FROM file_folders child_ff
                        JOIN files child_file
                          ON child_file.file_ref = child_ff.file_ref
                         AND child_file.deleted_at IS NULL
                        WHERE child_ff.folder_id = fo.folder_id
                    ) AS file_count,
                    (
                        SELECT COUNT(*)
                        FROM folders child_folder
                        WHERE child_folder.parent_id = fo.folder_id
                    ) AS children_count
                FROM folders fo
                WHERE fo.path = ?
                """,
                (path,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown folder path: {path}")
        return self._folder_row_to_dict(row)

    def find_folders(
        self,
        path: str = "/",
        *,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 100,
        max_depth: int | None = None,
        include_self: bool = False,
    ) -> list[dict[str, Any]]:
        path = normalize_path(path)
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        metadata_clause = f"AND {' AND '.join(metadata_sql)}" if metadata_sql else ""
        folder_depth_clause = ""
        folder_depth_params: list[Any] = []
        if max_depth is not None:
            if max_depth == 0 and not include_self:
                folder_depth_clause = "AND 0"
            else:
                folder_depth_clause = f"AND ({self._folder_depth_sql('fo.path')} - ?) <= ?"
                folder_depth_params = [self._folder_depth(path), max_depth]
        folder_scope_clause = (
            "(fo.path = ? OR fo.path LIKE ? ESCAPE '\\')"
            if include_self
            else "fo.path != ? AND fo.path LIKE ? ESCAPE '\\'"
        )
        sql = f"""
            SELECT *
            FROM (
                SELECT
                    fo.folder_id,
                    fo.parent_id,
                    fo.name,
                    fo.path,
                    fo.description,
                    fo.kind,
                    fo.metadata_json,
                    fo.created_at,
                    fo.updated_at,
                    (
                        SELECT COUNT(DISTINCT child_ff.file_ref)
                        FROM file_folders child_ff
                        JOIN files child_file
                          ON child_file.file_ref = child_ff.file_ref
                         AND child_file.deleted_at IS NULL
                        WHERE child_ff.folder_id = fo.folder_id
                    ) AS file_count,
                    (
                        SELECT COUNT(*)
                        FROM folders child_folder
                        WHERE child_folder.parent_id = fo.folder_id
                    ) AS children_count,
                    (
                        SELECT COUNT(DISTINCT f.file_ref)
                        FROM files f
                        JOIN file_folders matched_ff
                          ON matched_ff.file_ref = f.file_ref
                        JOIN folders matched_folder
                          ON matched_folder.folder_id = matched_ff.folder_id
                        WHERE f.deleted_at IS NULL
                          AND (
                              matched_folder.folder_id = fo.folder_id
                              OR matched_folder.path LIKE {self._descendant_like_sql_expr("fo.path")} ESCAPE '\\'
                          )
                          {metadata_clause}
                    ) AS matched_files
                FROM folders fo
                WHERE {folder_scope_clause}
                  {folder_depth_clause}
            )
            WHERE matched_files > 0
            ORDER BY path
            LIMIT ?
        """
        params = [
            *metadata_params,
            path,
            self._descendant_like(path),
            *folder_depth_params,
            limit,
        ]
        with self.connect() as conn:
            folder = self._folder_by_path(conn, path)
            if folder is None:
                raise KeyError(f"Unknown folder path: {path}")
            rows = conn.execute(sql, params).fetchall()
        return [self._folder_row_to_dict(row) for row in rows]

    def list_files(
        self,
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        selects = [
            "f.file_ref",
            "f.external_id",
            "f.title",
            "f.descriptor",
            "f.pageindex_tree_status",
            "f.metadata_json",
            "f.metadata_status_json",
            "f.created_at",
            """
            (
                SELECT display_folder.folder_id
                FROM file_folders display_ff
                JOIN folders display_folder
                  ON display_folder.folder_id = display_ff.folder_id
                WHERE display_ff.file_ref = f.file_ref
                ORDER BY display_folder.path
                LIMIT 1
            ) AS folder_id
            """,
            """
            (
                SELECT display_folder.path
                FROM file_folders display_ff
                JOIN folders display_folder
                  ON display_folder.folder_id = display_ff.folder_id
                WHERE display_ff.file_ref = f.file_ref
                ORDER BY display_folder.path
                LIMIT 1
            ) AS folder_path
            """,
        ]
        where = ["f.deleted_at IS NULL"]
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(scope)
        if scope_sql:
            where.append(scope_sql)
            params.extend(scope_params)
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        where.extend(metadata_sql)
        params.extend(metadata_params)
        sql = f"""
            SELECT {", ".join(selects)}
            FROM files f
            WHERE {" AND ".join(where)}
            ORDER BY lower(f.title), f.title, f.file_ref
            LIMIT ?
        """
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._file_summary(row) for row in rows]

    def file_refs_for_scope(
        self,
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        where = ["f.deleted_at IS NULL"]
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(scope)
        if scope_sql:
            where.append(scope_sql)
            params.extend(scope_params)
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        where.extend(metadata_sql)
        params.extend(metadata_params)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT f.file_ref
                FROM files f
                WHERE {" AND ".join(where)}
                ORDER BY f.file_ref
                """,
                params,
            ).fetchall()
        return [row["file_ref"] for row in rows]

    def count_files(
        self,
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> int:
        where = ["f.deleted_at IS NULL"]
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(scope)
        if scope_sql:
            where.append(scope_sql)
            params.extend(scope_params)
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        where.extend(metadata_sql)
        params.extend(metadata_params)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(DISTINCT f.file_ref) AS file_count
                FROM files f
                WHERE {" AND ".join(where)}
                """,
                params,
            ).fetchone()
        return int(row["file_count"] if row is not None else 0)

    def list_metadata_axes(
        self,
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        exclude_fields: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        where = ["f.deleted_at IS NULL", "mv.value_text != ''"]
        params: list[Any] = []
        scope_sql, scope_params = self._scope_sql(scope)
        if scope_sql:
            where.append(scope_sql)
            params.extend(scope_params)
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        where.extend(metadata_sql)
        params.extend(metadata_params)
        if exclude_fields:
            placeholders = ", ".join("?" for _ in sorted(exclude_fields))
            where.append(f"mf.name NOT IN ({placeholders})")
            params.extend(sorted(exclude_fields))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT mf.name, COUNT(DISTINCT mv.value_text) AS value_count
                FROM metadata_values mv
                JOIN metadata_fields mf ON mf.field_id = mv.field_id
                JOIN files f ON f.file_ref = mv.file_ref
                WHERE {" AND ".join(where)}
                GROUP BY mf.name
                ORDER BY mf.name
                """,
                params,
            ).fetchall()
        return [
            {"name": row["name"], "value_count": int(row["value_count"] or 0)}
            for row in rows
        ]

    def list_metadata_values(
        self,
        field: str,
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
        limit: int = 51,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where = [
            "f.deleted_at IS NULL",
            "mf.name = ?",
            "mv.value_text != ''",
        ]
        params: list[Any] = [field]
        scope_sql, scope_params = self._scope_sql(scope)
        if scope_sql:
            where.append(scope_sql)
            params.extend(scope_params)
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        where.extend(metadata_sql)
        params.extend(metadata_params)
        params.extend([limit, offset])
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT mv.value_text AS value, COUNT(DISTINCT f.file_ref) AS file_count
                FROM metadata_values mv
                JOIN metadata_fields mf ON mf.field_id = mv.field_id
                JOIN files f ON f.file_ref = mv.file_ref
                WHERE {" AND ".join(where)}
                GROUP BY mv.value_text
                ORDER BY file_count DESC, mv.value_text
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
        return [
            {"value": row["value"], "file_count": int(row["file_count"] or 0)}
            for row in rows
        ]

    def _metadata_filter_sql(self, metadata_filter: Optional[dict[str, Any]]) -> tuple[list[str], list[Any]]:
        if not metadata_filter:
            return [], []
        clause, params = self._compile_metadata_filter(metadata_filter)
        return [clause] if clause else [], params

    def _compile_metadata_filter(self, metadata_filter: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses = []
        params: list[Any] = []
        for key, condition in metadata_filter.items():
            if key in {"$and", "$or"}:
                child_clauses = []
                child_params: list[Any] = []
                for item in condition:
                    child_clause, item_params = self._compile_metadata_filter(item)
                    if child_clause:
                        child_clauses.append(f"({child_clause})")
                        child_params.extend(item_params)
                if child_clauses:
                    joiner = " AND " if key == "$and" else " OR "
                    clauses.append(joiner.join(child_clauses))
                    params.extend(child_params)
                continue
            field_clause, field_params = self._compile_metadata_field_filter(key, condition)
            clauses.append(field_clause)
            params.extend(field_params)
        return " AND ".join(f"({clause})" for clause in clauses), params

    def _compile_metadata_field_filter(self, field: str, condition: Any) -> tuple[str, list[Any]]:
        if not isinstance(condition, dict) or not any(str(key).startswith("$") for key in condition):
            condition = {"$eq": condition}
        operator, expected = next(iter(condition.items()))
        field_id = self.field_id(field)
        if operator == "$eq":
            return (
                """
                EXISTS (
                    SELECT 1 FROM metadata_values mv
                    WHERE mv.file_ref = f.file_ref
                      AND mv.field_id = ?
                      AND mv.value_text = ?
                )
                """,
                [field_id, self._metadata_compare_text(expected)],
            )
        if operator == "$ne":
            return (
                """
                NOT EXISTS (
                    SELECT 1 FROM metadata_values mv
                    WHERE mv.file_ref = f.file_ref
                      AND mv.field_id = ?
                      AND mv.value_text = ?
                )
                """,
                [field_id, self._metadata_compare_text(expected)],
            )
        if operator == "$in":
            values = list(expected)
            if not values:
                return "0", []
            placeholders = ", ".join("?" for _ in values)
            return (
                f"""
                EXISTS (
                    SELECT 1 FROM metadata_values mv
                    WHERE mv.file_ref = f.file_ref
                      AND mv.field_id = ?
                      AND mv.value_text IN ({placeholders})
                )
                """,
                [field_id, *(self._metadata_compare_text(item) for item in values)],
            )
        if operator == "$contains":
            return (
                """
                EXISTS (
                    SELECT 1 FROM metadata_values mv
                    WHERE mv.file_ref = f.file_ref
                      AND mv.field_id = ?
                      AND lower(mv.value_text) LIKE lower(?) ESCAPE '\\'
                )
                """,
                [field_id, self._contains_like(self._metadata_compare_text(expected))],
            )
        raise ValueError(f"Unsupported metadata operator: {operator}")

    def get_file(self, file_ref: str) -> FileEntry:
        with self.connect() as conn:
            row = self._file_entry_row(conn, file_ref)
        if row is None:
            raise KeyError(f"Unknown file_ref: {file_ref}")
        return self._file_entry(row)

    def update_file_metadata_status(
        self,
        file_ref: str,
        *,
        metadata: dict[str, Any],
        metadata_status: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            row = self._file_entry_row(conn, file_ref)
            if row is None:
                raise KeyError(f"Unknown file_ref: {file_ref}")
            indexed_metadata = self.indexed_metadata_values(metadata)
            conn.execute(
                """
                UPDATE files
                SET metadata_json = ?,
                    metadata_status_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_ref = ? AND deleted_at IS NULL
                """,
                (
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(metadata_status, ensure_ascii=False),
                    file_ref,
                ),
            )
            self.replace_metadata_values(
                conn,
                file_ref,
                indexed_metadata,
            )

    def delete_file(self, target: str) -> None:
        with self.connect() as conn:
            file_ref = self._resolve_file_ref(conn, target)
            conn.execute("DELETE FROM metadata_values WHERE file_ref = ?", (file_ref,))
            conn.execute("DELETE FROM files WHERE file_ref = ?", (file_ref,))

    def folder_exists(self, path: str) -> bool:
        path = normalize_path(path)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM folders WHERE path = ?",
                (path,),
            ).fetchone()
        return row is not None

    def delete_empty_folder(self, path: str) -> bool:
        path = normalize_path(path)
        if path == "/":
            return False
        with self.connect() as conn:
            folder = self._folder_by_path(conn, path)
            if folder is None:
                return False
            has_files = conn.execute(
                """
                SELECT 1
                FROM file_folders
                WHERE folder_id = ?
                LIMIT 1
                """,
                (folder["folder_id"],),
            ).fetchone()
            if has_files is not None:
                return False
            has_children = conn.execute(
                """
                SELECT 1
                FROM folders
                WHERE parent_id = ?
                LIMIT 1
                """,
                (folder["folder_id"],),
            ).fetchone()
            if has_children is not None:
                return False
            conn.execute("DELETE FROM folders WHERE folder_id = ?", (folder["folder_id"],))
            return True

    def resolve_file_ref(self, target: str) -> str:
        with self.connect() as conn:
            return self._resolve_file_ref(conn, target)

    def _resolve_file_ref(self, conn: sqlite3.Connection, target: str) -> str:
        target = str(target).strip()
        if not target:
            raise KeyError("Empty file target")
        row = conn.execute(
            "SELECT file_ref FROM files WHERE file_ref = ? AND deleted_at IS NULL",
            (target,),
        ).fetchone()
        if row:
            return row["file_ref"]
        row = conn.execute(
            "SELECT file_ref FROM files WHERE external_id = ? AND deleted_at IS NULL",
            (target,),
        ).fetchone()
        if row:
            return row["file_ref"]
        virtual_file_ref = self._resolve_virtual_file_ref(conn, target)
        if virtual_file_ref:
            return virtual_file_ref
        raise KeyError(f"Unknown file target: {target}")

    def _resolve_virtual_file_ref(self, conn: sqlite3.Connection, target: str) -> str | None:
        virtual_target = normalize_path(target)
        rows = conn.execute(
            """
            WITH virtual_matches AS (
                SELECT
                    f.file_ref,
                    f.external_id,
                    f.title,
                    COALESCE(
                        NULLIF(json_extract(ff.metadata_json, '$.display_name'), ''),
                        f.title
                    ) AS display_title,
                    pf.path AS folder_path,
                    (CASE WHEN pf.path = '/' THEN '/' ELSE pf.path || '/' END)
                        || ltrim(
                            COALESCE(
                                NULLIF(json_extract(ff.metadata_json, '$.display_name'), ''),
                                f.title
                            ),
                            '/'
                        ) AS title_virtual_path
                FROM files f
                JOIN file_folders ff ON ff.file_ref = f.file_ref
                JOIN folders pf ON pf.folder_id = ff.folder_id
                WHERE f.deleted_at IS NULL
            )
            SELECT
                file_ref,
                external_id,
                display_title AS title,
                MIN(folder_path) AS folder_path
            FROM virtual_matches
            WHERE title_virtual_path = ?
            GROUP BY file_ref, external_id, display_title
            ORDER BY file_ref
            LIMIT 2
            """,
            (virtual_target,),
        ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            matches = "; ".join(self._virtual_match_summary(row) for row in rows)
            raise KeyError(f"Ambiguous file target: {target}. Matches: {matches}")
        return rows[0]["file_ref"]

    @staticmethod
    def _virtual_match_summary(row: sqlite3.Row) -> str:
        external_id = row["external_id"] or "-"
        return (
            f"file_ref={row['file_ref']} external_id={external_id} "
            f"folder={row['folder_path']} title={row['title']!r}"
        )

    def ensure_folder(
        self,
        conn: sqlite3.Connection | None,
        path: str,
        *,
        kind: str = "manual",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        owns_connection = conn is None
        if conn is None:
            conn = self.connect()
        try:
            normalized = normalize_path(path)
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
            if normalized == "/":
                folder_id = self.folder_id("/")
                existing = conn.execute(
                    "SELECT folder_id FROM folders WHERE path = '/'"
                ).fetchone()
                if existing is not None and not description and metadata_json == "{}":
                    if owns_connection:
                        conn.commit()
                    return folder_id
                self._upsert_folder_row(
                    conn,
                    folder_id=folder_id,
                    parent_id=None,
                    name="/",
                    path="/",
                    kind=kind,
                    description=description,
                    metadata_json=metadata_json,
                )
                if owns_connection:
                    conn.commit()
                return folder_id
            parent_id = self.ensure_folder(conn, str(Path(normalized).parent), kind=kind)
            name = normalized.rsplit("/", 1)[-1]
            folder_id = self.folder_id(normalized)
            self._upsert_folder_row(
                conn,
                folder_id=folder_id,
                parent_id=parent_id,
                name=name,
                path=normalized,
                kind=kind,
                description=description,
                metadata_json=metadata_json,
            )
            if owns_connection:
                conn.commit()
            return folder_id
        finally:
            if owns_connection:
                conn.close()

    def _upsert_folder_row(
        self,
        conn: sqlite3.Connection,
        *,
        folder_id: str,
        parent_id: str | None,
        name: str,
        path: str,
        kind: str,
        description: str,
        metadata_json: str,
    ) -> None:
        columns = self._columns(conn, "folders")
        insert_columns = ["folder_id", "parent_id", "name", "path", "description", "kind", "metadata_json"]
        values: list[Any] = [folder_id, parent_id, name, path, description, kind, metadata_json]
        if "source" in columns:
            insert_columns.append("source")
            values.append("system")
        if "sort_order" in columns:
            insert_columns.append("sort_order")
            values.append(0)
        placeholders = ", ".join("?" for _ in values)
        update_assignments = [
            "parent_id = excluded.parent_id",
            "name = excluded.name",
            "kind = excluded.kind",
            "updated_at = CURRENT_TIMESTAMP",
        ]
        if description:
            update_assignments.append("description = excluded.description")
        if metadata_json != "{}":
            update_assignments.append("metadata_json = excluded.metadata_json")
        conn.execute(
            f"""
            INSERT INTO folders({", ".join(insert_columns)})
            VALUES ({placeholders})
            ON CONFLICT(path) DO UPDATE SET
                {", ".join(update_assignments)}
            """,
            values,
        )

    def _resolve_or_create_folder(self, conn: sqlite3.Connection, folder_path_or_id: str) -> str:
        target = str(folder_path_or_id).strip()
        if not target:
            raise KeyError("Empty folder target")
        row = conn.execute(
            "SELECT folder_id FROM folders WHERE folder_id = ?",
            (target,),
        ).fetchone()
        if row:
            return row["folder_id"]
        row = conn.execute(
            "SELECT folder_id FROM folders WHERE path = ?",
            (normalize_path(target),),
        ).fetchone()
        if row:
            return row["folder_id"]
        return self.ensure_folder(conn, target)

    def read_text(self, file_ref: str) -> str:
        entry = self.get_file(file_ref)
        return Path(entry.text_artifact_path).read_text(encoding="utf-8")

    def write_text_artifact(self, file_ref: str, content: str) -> Path:
        path = self.text_dir / f"{file_ref}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def update_pageindex_pointer(
        self,
        file_ref: str,
        *,
        pageindex_doc_id: str | None,
        pageindex_tree_status: str,
    ) -> None:
        with self.connect() as conn:
            resolved = self._resolve_file_ref(conn, file_ref)
            conn.execute(
                """
                UPDATE files
                SET pageindex_doc_id = ?,
                    pageindex_tree_status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE file_ref = ? AND deleted_at IS NULL
                """,
                (pageindex_doc_id, pageindex_tree_status, resolved),
            )

    def write_raw_artifact(self, file_ref: str, metadata: dict[str, Any]) -> Path:
        path = self.raw_dir / f"{file_ref}.json"
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def file_info(self, target: str) -> dict[str, Any]:
        file_ref = self.resolve_file_ref(target)
        entry = self.get_file(file_ref)
        info = self._file_entry_to_dict(entry)
        info["folders"] = self.folder_memberships(file_ref)
        return info

    def file_matches(
        self,
        file_ref: str,
        *,
        scope: Optional[dict[str, Any]] = None,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> bool:
        where = ["f.file_ref = ?", "f.deleted_at IS NULL"]
        params: list[Any] = [file_ref]
        scope_sql, scope_params = self._scope_sql(scope)
        if scope_sql:
            where.append(scope_sql)
            params.extend(scope_params)
        metadata_sql, metadata_params = self._metadata_filter_sql(metadata_filter)
        where.extend(metadata_sql)
        params.extend(metadata_params)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM files f
                WHERE {" AND ".join(where)}
                LIMIT 1
                """,
                params,
            ).fetchone()
        return row is not None

    def folder_memberships(self, file_ref: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    fo.folder_id,
                    fo.parent_id,
                    fo.name,
                    fo.path,
                    fo.description,
                    fo.kind,
                    fo.metadata_json AS folder_metadata_json,
                    ff.metadata_json AS membership_metadata_json,
                    ff.created_at
                FROM file_folders ff
                JOIN folders fo ON fo.folder_id = ff.folder_id
                WHERE ff.file_ref = ?
                ORDER BY fo.path
                """,
                (file_ref,),
            ).fetchall()
        return [
            {
                "folder_id": row["folder_id"],
                "parent_id": row["parent_id"],
                "name": row["name"],
                "path": row["path"],
                "kind": row["kind"],
                "description": row["description"],
                "folder_metadata": json.loads(row["folder_metadata_json"] or "{}"),
                "metadata": json.loads(row["membership_metadata_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def count_files_in_folder(self, path: str, *, recursive: bool = True) -> int:
        path = normalize_path(path)
        with self.connect() as conn:
            folder = self._folder_by_path(conn, path)
            if folder is None:
                raise KeyError(f"Unknown folder path: {path}")
            if recursive:
                row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT f.file_ref) AS count
                    FROM files f
                    JOIN file_folders ff ON ff.file_ref = f.file_ref
                    JOIN folders fo ON fo.folder_id = ff.folder_id
                    WHERE f.deleted_at IS NULL
                      AND (fo.path = ? OR fo.path LIKE ? ESCAPE '\\')
                    """,
                    (path, self._descendant_like(path)),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT COUNT(DISTINCT f.file_ref) AS count
                    FROM files f
                    JOIN file_folders ff ON ff.file_ref = f.file_ref
                    JOIN folders fo ON fo.folder_id = ff.folder_id
                    WHERE f.deleted_at IS NULL
                      AND fo.path = ?
                    """,
                    (path,),
                ).fetchone()
        return int(row["count"] or 0)

    def file_basename_exists_in_folder(self, path: str, basename: str) -> bool:
        path = normalize_path(path)
        basename = str(basename).strip()
        if not basename:
            return False
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM files f
                JOIN file_folders ff ON ff.file_ref = f.file_ref
                JOIN folders fo ON fo.folder_id = ff.folder_id
                WHERE f.deleted_at IS NULL
                  AND fo.path = ?
                  AND f.title = ?
                LIMIT 1
                """,
                (
                    path,
                    basename,
                ),
            ).fetchone()
        return row is not None

    def folder_subtree_thresholds(
        self,
        path: str,
        *,
        depth_limit: int,
        file_limit: int,
    ) -> dict[str, Any]:
        path = normalize_path(path)
        with self.connect() as conn:
            folder = self._folder_by_path(conn, path)
            if folder is None:
                raise KeyError(f"Unknown folder path: {path}")
            base_depth = self._folder_depth(path)
            deep_folder = conn.execute(
                """
                SELECT path
                FROM folders
                WHERE path != ?
                  AND path LIKE ? ESCAPE '\\'
                  AND (
                    CASE
                      WHEN TRIM(path, '/') = '' THEN 0
                      ELSE LENGTH(TRIM(path, '/')) - LENGTH(REPLACE(TRIM(path, '/'), '/', '')) + 1
                    END
                  ) - ? > ?
                LIMIT 1
                """,
                (path, self._descendant_like(path), base_depth, depth_limit),
            ).fetchone()
            file_rows = conn.execute(
                """
                SELECT DISTINCT f.file_ref
                FROM files f
                JOIN file_folders ff ON ff.file_ref = f.file_ref
                JOIN folders fo ON fo.folder_id = ff.folder_id
                WHERE f.deleted_at IS NULL
                  AND (fo.path = ? OR fo.path LIKE ? ESCAPE '\\')
                LIMIT ?
                """,
                (path, self._descendant_like(path), file_limit + 1),
            ).fetchall()
        return {
            "depth_limit": depth_limit,
            "file_limit": file_limit,
            "folder_depth_exceeds_limit": deep_folder is not None,
            "file_count_exceeds_limit": len(file_rows) > file_limit,
            "sampled_file_count": len(file_rows),
            "sample_deep_folder_path": deep_folder["path"] if deep_folder is not None else "",
        }

    def _file_entry_row(self, conn: sqlite3.Connection, file_ref: str) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT
                f.file_ref,
                f.external_id,
                f.storage_uri,
                f.title,
                f.descriptor,
                f.content_type,
                f.source_type,
                f.fingerprint,
                f.text_artifact_path,
                f.raw_artifact_path,
                f.pageindex_doc_id,
                f.pageindex_tree_status,
                f.metadata_json,
                f.metadata_status_json,
                COALESCE(
                    (
                        SELECT display_folder.path
                        FROM file_folders display_ff
                        JOIN folders display_folder
                          ON display_folder.folder_id = display_ff.folder_id
                        WHERE display_ff.file_ref = f.file_ref
                        ORDER BY display_folder.path
                        LIMIT 1
                    ),
                    '/'
                ) AS folder_path
            FROM files f
            WHERE f.file_ref = ? AND f.deleted_at IS NULL
            """,
            (file_ref,),
        ).fetchone()

    def _file_rows_for_scope(
        self,
        conn: sqlite3.Connection,
        path: str,
        recursive: bool,
        limit: int,
        max_depth: int | None = None,
    ) -> list[sqlite3.Row]:
        sql = """
            SELECT *
            FROM (
                SELECT
                    f.file_ref,
                    f.external_id,
                    f.title,
                    f.descriptor,
                    f.pageindex_tree_status,
                    f.metadata_json,
                    f.metadata_status_json,
                    f.created_at,
                    pf.folder_id AS folder_id,
                    pf.path AS folder_path,
                    COALESCE(
                        NULLIF(json_extract(ff.metadata_json, '$.display_name'), ''),
                        f.title
                    ) AS display_title,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.file_ref
                        ORDER BY pf.path, pf.folder_id
                    ) AS membership_rank
                FROM files f
                JOIN file_folders ff ON ff.file_ref = f.file_ref
                JOIN folders pf ON pf.folder_id = ff.folder_id
                WHERE f.deleted_at IS NULL
        """
        params: list[Any]
        if recursive:
            sql += " AND (pf.path = ? OR pf.path LIKE ? ESCAPE '\\')"
            params = [path, self._descendant_like(path)]
            if max_depth is not None:
                if max_depth <= 0:
                    sql += " AND 0"
                else:
                    sql += f" AND ({self._folder_depth_sql('pf.path')} - ?) <= ?"
                    params.extend([self._folder_depth(path), max_depth - 1])
        else:
            sql += " AND pf.path = ?"
            params = [path]
        sql += """
            )
            WHERE membership_rank = 1
            ORDER BY lower(display_title), display_title, lower(folder_path), folder_path, file_ref
            LIMIT ?
        """
        params.append(limit)
        return conn.execute(sql, params).fetchall()

    def _scope_sql(self, scope: Optional[dict[str, Any]]) -> tuple[str, list[Any]]:
        if not scope:
            return "", []
        recursive = scope.get("recursive", True)
        max_depth = scope.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
            if max_depth < 0:
                raise ValueError("max_depth must be non-negative")
        folder_id = scope.get("folder_id")
        if folder_id:
            if folder_id == "root":
                folder_path = "/"
            else:
                if recursive:
                    if max_depth == 0:
                        return "0", []
                    depth_clause = ""
                    depth_params: list[Any] = []
                    if max_depth is not None:
                        depth_clause = (
                            "AND "
                            f"({self._folder_depth_sql('scope_folder.path')} - "
                            f"{self._folder_depth_sql('base_folder.path')}) <= ?"
                        )
                        depth_params = [max_depth - 1]
                    return (
                        f"""
                        EXISTS (
                            SELECT 1
                            FROM file_folders scope_ff
                            JOIN folders scope_folder
                              ON scope_folder.folder_id = scope_ff.folder_id
                            JOIN folders base_folder
                              ON base_folder.folder_id = ?
                            WHERE scope_ff.file_ref = f.file_ref
                              AND (
                                scope_folder.folder_id = base_folder.folder_id
                                OR scope_folder.path LIKE {self._descendant_like_sql_expr("base_folder.path")} ESCAPE '\\'
                              )
                              {depth_clause}
                        )
                        """,
                        [folder_id, *depth_params],
                    )
                return (
                    """
                    EXISTS (
                        SELECT 1
                        FROM file_folders scope_ff
                        WHERE scope_ff.file_ref = f.file_ref
                          AND scope_ff.folder_id = ?
                    )
                    """,
                    [folder_id],
                )
        elif scope.get("folder_path") or scope.get("path"):
            folder_path = normalize_path(scope.get("folder_path") or scope.get("path"))
        else:
            return "", []
        if recursive and max_depth == 0:
            return "0", []
        path_clause = (
            "(scope_folder.path = ? OR scope_folder.path LIKE ? ESCAPE '\\')"
            if recursive
            else "scope_folder.path = ?"
        )
        params = [folder_path, self._descendant_like(folder_path)] if recursive else [folder_path]
        depth_clause = ""
        if recursive and max_depth is not None:
            depth_clause = f"AND ({self._folder_depth_sql('scope_folder.path')} - ?) <= ?"
            params.extend([self._folder_depth(folder_path), max_depth - 1])
        return (
            f"""
            EXISTS (
                SELECT 1
                FROM file_folders scope_ff
                JOIN folders scope_folder
                  ON scope_folder.folder_id = scope_ff.folder_id
                WHERE scope_ff.file_ref = f.file_ref
                  AND {path_clause}
                  {depth_clause}
            )
            """,
            params,
        )

    def _folder_by_path(self, conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT
                folder_id,
                parent_id,
                name,
                path,
                description,
                kind,
                metadata_json,
                created_at,
                updated_at
            FROM folders
            WHERE path = ?
            """,
            (path,),
        ).fetchone()

    @classmethod
    def _descendant_like(cls, path: str) -> str:
        return "/%" if path == "/" else f"{cls._like_escape(path)}/%"

    @staticmethod
    def _descendant_like_sql_expr(path_expr: str) -> str:
        escaped_expr = SQLiteFileSystemStore._like_escape_sql_expr(path_expr)
        return f"CASE WHEN {path_expr} = '/' THEN '/%' ELSE {escaped_expr} || '/%' END"

    @staticmethod
    def _contains_like(value: str) -> str:
        return f"%{SQLiteFileSystemStore._like_escape(value)}%"

    @staticmethod
    def _like_escape(value: str) -> str:
        return (
            value.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    @staticmethod
    def _like_escape_sql_expr(value_expr: str) -> str:
        return (
            f"replace(replace(replace({value_expr}, '\\', '\\\\'), "
            "'%', '\\%'), '_', '\\_')"
        )

    @staticmethod
    def _folder_depth(path: str) -> int:
        stripped = normalize_path(path).strip("/")
        return 0 if not stripped else len(stripped.split("/"))

    @staticmethod
    def _folder_depth_sql(path_expr: str) -> str:
        return (
            "(CASE "
            f"WHEN TRIM({path_expr}, '/') = '' THEN 0 "
            f"ELSE LENGTH(TRIM({path_expr}, '/')) "
            f"- LENGTH(REPLACE(TRIM({path_expr}, '/'), '/', '')) + 1 "
            "END)"
        )

    @classmethod
    def _folder_row_to_dict(cls, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "folder_id": row["folder_id"],
            "parent_id": row["parent_id"],
            "name": row["name"],
            "description": cls._row_value(row, "description", ""),
            "path": row["path"],
            "kind": row["kind"],
            "metadata": json.loads(cls._row_value(row, "metadata_json", "{}") or "{}"),
            "created_at": cls._row_value(row, "created_at"),
            "updated_at": cls._row_value(row, "updated_at"),
            "file_count": cls._row_value(row, "file_count", 0),
            "children_count": cls._row_value(row, "children_count", 0),
            "matched_files": cls._row_value(row, "matched_files", 0),
        }

    @classmethod
    def _file_summary(cls, row: sqlite3.Row) -> dict[str, Any]:
        display_title = cls._row_value(row, "display_title", row["title"])
        return {
            "file_ref": row["file_ref"],
            "external_id": row["external_id"],
            "title": display_title,
            "descriptor": cls._row_value(row, "descriptor", row["title"]),
            "pageindex_tree_status": cls._row_value(
                row, "pageindex_tree_status", "not_built"
            ),
            "created_at": cls._row_value(row, "created_at"),
            "folder_id": cls._row_value(row, "folder_id"),
            "folder_path": row["folder_path"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "metadata_status": json.loads(
                cls._row_value(row, "metadata_status_json", "{}") or "{}"
            ),
        }

    @staticmethod
    def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
        return row[key] if key in row.keys() else default

    @staticmethod
    def _file_entry(row: sqlite3.Row) -> FileEntry:
        return FileEntry(
            file_ref=row["file_ref"],
            external_id=row["external_id"],
            storage_uri=row["storage_uri"],
            title=row["title"],
            descriptor=row["descriptor"],
            content_type=row["content_type"],
            source_type=row["source_type"],
            fingerprint=row["fingerprint"],
            text_artifact_path=row["text_artifact_path"],
            raw_artifact_path=row["raw_artifact_path"],
            pageindex_doc_id=row["pageindex_doc_id"],
            pageindex_tree_status=row["pageindex_tree_status"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            folder_path=row["folder_path"],
            metadata_status=json.loads(
                SQLiteFileSystemStore._row_value(row, "metadata_status_json", "{}") or "{}"
            ),
        )

    @classmethod
    def _file_entry_to_dict(cls, entry: FileEntry) -> dict[str, Any]:
        return {
            "file_ref": entry.file_ref,
            "external_id": entry.external_id,
            "title": entry.title,
            "descriptor": entry.descriptor,
            "content_type": entry.content_type,
            "source_type": entry.source_type,
            "fingerprint": entry.fingerprint,
            "pageindex_doc_id": entry.pageindex_doc_id,
            "pageindex_tree_status": entry.pageindex_tree_status,
            "metadata": entry.metadata,
            "metadata_status": entry.metadata_status,
            "folder_path": entry.folder_path,
        }

    @staticmethod
    def _metadata_value_items(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = []
            for item in value:
                items.extend(SQLiteFileSystemStore._metadata_value_items(item))
            return items
        return [SQLiteFileSystemStore._metadata_compare_text(value)]

    @staticmethod
    def _metadata_compare_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return "" if value is None else str(value)

    @staticmethod
    def indexed_metadata_values(metadata: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in metadata.items() if key != "summary"}

    @staticmethod
    def _valid_field_name(name: str) -> bool:
        return re.match(r"^[A-Za-z][A-Za-z0-9_]*$", str(name)) is not None

    @staticmethod
    def folder_id(path: str) -> str:
        normalized = normalize_path(path)
        if normalized == "/":
            return "folder_root"
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
        return f"folder_{digest}"

    @staticmethod
    def field_id(name: str) -> str:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
        return f"field_{digest}"


def normalize_path(path: str | Path | None) -> str:
    if path is None:
        return "/"
    if str(path).strip().lower() == "root":
        return "/"
    parts = [part for part in str(path).replace("\\", "/").split("/") if part and part != "."]
    if ".." in parts:
        raise ValueError("PIFS paths must not contain '..'")
    return "/" + "/".join(parts) if parts else "/"


def make_file_ref(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"file_{digest}"


def fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
