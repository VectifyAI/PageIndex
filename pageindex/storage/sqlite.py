import json
import sqlite3
import threading
from pathlib import Path

from ..errors import CollectionAlreadyExistsError
from .._validation import validate_collection_name


def _validate_collection_name(name: str) -> None:
    # SQLiteStorage is a public StorageEngine and may be used without
    # LocalBackend, so enforce the shared contract at this boundary too.
    validate_collection_name(name)


class SQLiteStorage:
    def __init__(self, db_path: str):
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        # Bumped by close(). A thread caches its connection in thread-local
        # storage, so after close() every OTHER thread's thread-local still
        # points at a now-closed connection. Comparing the cached generation
        # against this counter lets _get_conn detect that and reconnect, instead
        # of handing back a closed connection (sqlite3.ProgrammingError). close()
        # can only touch its OWN thread-local, so this is the only way to
        # invalidate the others consistently.
        self._generation = 0
        # Serializes the (fast) write operations within this process so
        # concurrent indexing threads don't collide on WAL's single writer
        # ("database is locked"). Reads stay concurrent; the expensive LLM
        # indexing runs outside this lock. busy_timeout above covers the
        # cross-process case.
        self._write_lock = threading.Lock()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection.

        Reconnects if this thread has no connection yet OR its cached connection
        was invalidated by a close() on another thread (generation mismatch).
        """
        if (not hasattr(self._local, "conn")
                or getattr(self._local, "generation", None) != self._generation):
            # Each thread gets its own connection (threading.local), so
            # statements never race. check_same_thread=False exists solely so
            # close() can close every tracked connection from whichever thread
            # calls it — with the default True those closes raise
            # ProgrammingError and the connections leak.
            # isolation_level=None -> autocommit: a plain SELECT (e.g. the
            # dedup hash lookup) never leaves a lingering read snapshot that a
            # later write on the same connection would conflict with
            # (SQLITE_BUSY_SNAPSHOT, which busy_timeout can't retry). Each
            # statement is its own transaction, so busy_timeout can actually
            # wait for the WAL single-writer lock under concurrency.
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False,
                                   isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
            self._local.generation = self._generation
            with self._conn_lock:
                self._connections.append(conn)
        return self._local.conn

    def _init_schema(self):
        conn = self._get_conn()
        # SQLite cannot ALTER a CHECK constraint. Rebuild the small parent table
        # transactionally when opening a v1 database; documents keep referring to
        # the same table name and are verified after foreign keys are re-enabled.
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY NOT NULL
                        CHECK(
                            length(name) BETWEEN 1 AND 255
                            -- SQLite length(TEXT) stops at the first NUL, while
                            -- Python and the cloud API count it as a character.
                            -- The Python boundary still enforces the 255 limit.
                            OR (instr(name, char(0)) > 0 AND name <> '')
                        ),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            schema_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'collections'"
            ).fetchone()
            schema_sql = schema_row[0] if schema_row else ""
            schema_upper = schema_sql.upper()
            if "BETWEEN 1 AND 128" in schema_upper or "NAME GLOB" in schema_upper:
                conn.execute("""
                    CREATE TABLE _pageindex_collections_v2 (
                        name TEXT PRIMARY KEY NOT NULL
                            CHECK(
                                length(name) BETWEEN 1 AND 255
                                OR (instr(name, char(0)) > 0 AND name <> '')
                            ),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    INSERT INTO _pageindex_collections_v2 (name, created_at)
                    SELECT name, created_at FROM collections
                """)
                conn.execute("DROP TABLE collections")
                conn.execute("ALTER TABLE _pageindex_collections_v2 RENAME TO collections")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
                    doc_name TEXT,
                    doc_description TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    doc_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER,
                    line_count INTEGER,
                    structure JSON,
                    pages JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collection_name, file_hash)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_collection ON documents(collection_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(collection_name, file_hash)"
            )
            if schema_version < 2:
                conn.execute("PRAGMA user_version = 2")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys=ON")

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(
                f"Foreign-key violations after SQLite schema migration: {violations!r}"
            )

        # DBs created before these columns existed: add them in place.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
        for col_name, col_def in (
            ("page_count", "INTEGER"),
            ("line_count", "INTEGER"),
            ("status", "TEXT NOT NULL DEFAULT 'completed'"),
        ):
            if col_name not in cols:
                try:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_def}")
                except sqlite3.OperationalError:
                    pass  # concurrent open of the same legacy DB already added it
        conn.commit()

    def create_collection(self, name: str) -> None:
        _validate_collection_name(name)
        with self._write_lock:
            conn = self._get_conn()
            try:
                conn.execute("INSERT INTO collections (name) VALUES (?)", (name,))
            except sqlite3.IntegrityError as e:
                raise CollectionAlreadyExistsError(f"Collection '{name}' already exists") from e
            conn.commit()

    def get_or_create_collection(self, name: str) -> None:
        _validate_collection_name(name)
        with self._write_lock:
            conn = self._get_conn()
            # INSERT OR IGNORE works with older SQLite versions, but it can also
            # suppress CHECK failures. Verify the row so ignored non-duplicate
            # constraints never falsely report success.
            conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (name,))
            if conn.execute(
                "SELECT 1 FROM collections WHERE name = ?", (name,)
            ).fetchone() is None:
                raise sqlite3.IntegrityError(
                    f"Collection {name!r} was rejected by the SQLite schema"
                )
            conn.commit()

    def list_collections(self) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute("SELECT name FROM collections ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def delete_collection(self, name: str) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            conn.commit()

    def save_document(self, collection: str, doc_id: str, doc: dict) -> None:
        # Plain INSERT (doc_id is a fresh uuid, never pre-existing). A duplicate
        # (collection_name, file_hash) raises sqlite3.IntegrityError, which the
        # caller uses to resolve a concurrent add-of-same-file race.
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO documents
                   (doc_id, collection_name, doc_name, doc_description, file_path, file_hash, doc_type, status, page_count, line_count, structure, pages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, collection, doc.get("doc_name"), doc.get("doc_description"),
                 doc.get("file_path"), doc.get("file_hash"), doc["doc_type"],
                 doc["status"],
                 doc.get("page_count"), doc.get("line_count"),
                 json.dumps(doc.get("structure", [])),
                 json.dumps(doc.get("pages")) if doc.get("pages") else None),
            )
            conn.commit()

    def find_document_by_hash(self, collection: str, file_hash: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT doc_id FROM documents WHERE collection_name = ? AND file_hash = ?",
            (collection, file_hash),
        ).fetchone()
        return row[0] if row else None

    def get_document(self, collection: str, doc_id: str) -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT doc_id, doc_name, doc_description, file_path, doc_type, status, page_count, line_count FROM documents WHERE doc_id = ? AND collection_name = ?",
            (doc_id, collection),
        ).fetchone()
        if not row:
            return {}
        doc = {"doc_id": row[0], "doc_name": row[1], "doc_description": row[2],
               "file_path": row[3], "doc_type": row[4], "status": row[5]}
        doc.update({k: v for k, v in (("page_count", row[6]), ("line_count", row[7])) if v is not None})
        return doc

    def get_document_structure(self, collection: str, doc_id: str) -> list:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT structure FROM documents WHERE doc_id = ? AND collection_name = ?",
            (doc_id, collection),
        ).fetchone()
        if not row:
            return []
        return json.loads(row[0])

    def get_pages(self, collection: str, doc_id: str) -> list | None:
        """Return cached page content, or None if not cached."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT pages FROM documents WHERE doc_id = ? AND collection_name = ?",
            (doc_id, collection),
        ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    def list_documents(self, collection: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT doc_id, doc_name, doc_description, doc_type FROM documents WHERE collection_name = ? ORDER BY created_at DESC, doc_id ASC",
            (collection,),
        ).fetchall()
        return [{"doc_id": r[0], "doc_name": r[1], "doc_description": r[2] or "", "doc_type": r[3]} for r in rows]

    def delete_document(self, collection: str, doc_id: str) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM documents WHERE doc_id = ? AND collection_name = ?",
                (doc_id, collection),
            )
            conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Close all tracked SQLite connections across all threads."""
        with self._conn_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            # Invalidate every thread's cached connection. close() can only
            # del its OWN thread-local, so the bump is what makes _get_conn on
            # any other thread reconnect instead of reusing a closed handle.
            self._generation += 1
        if hasattr(self._local, "conn"):
            del self._local.conn

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
