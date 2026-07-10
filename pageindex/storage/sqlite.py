import json
import re
import sqlite3
import threading
from pathlib import Path

from ..errors import CollectionAlreadyExistsError, PageIndexError

# Mirrors LocalBackend's own collection-name rule. SQLiteStorage enforces this
# itself (not just relying on LocalBackend's pre-check or the schema's CHECK
# constraint below) because it's a public StorageEngine that can be used
# directly, bypassing LocalBackend entirely.
# Matched with .fullmatch() (not .match()): a $-anchored .match() would accept a
# trailing newline ("papers\n") because $ matches just before a final \n.
_COLLECTION_NAME_RE = re.compile(r'[a-zA-Z0-9_-]{1,128}')


def _validate_collection_name(name: str) -> None:
    if not _COLLECTION_NAME_RE.fullmatch(name):
        raise PageIndexError(f"Invalid collection name: {name!r}. Must be 1-128 chars of [a-zA-Z0-9_-].")


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
        conn.execute("PRAGMA user_version = 1")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS collections (
                -- GLOB '*' is "any characters", not a regex quantifier over the
                -- preceding class — '[a-zA-Z0-9_-]*' alone only constrains the
                -- FIRST character. The second GLOB (NOT ... '*[^...]*') checks
                -- every remaining character too, so this is real defense-in-depth
                -- for direct SQLiteStorage use (bypassing _validate_collection_name
                -- above), not just a first-character gate.
                name TEXT PRIMARY KEY CHECK(
                    length(name) BETWEEN 1 AND 128
                    AND name GLOB '[a-zA-Z0-9_-]*'
                    AND name NOT GLOB '*[^a-zA-Z0-9_-]*'
                ),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
                doc_name TEXT,
                doc_description TEXT,
                file_path TEXT,
                file_hash TEXT,
                doc_type TEXT NOT NULL,
                structure JSON,
                pages JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(collection_name, file_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_docs_collection ON documents(collection_name);
            CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(collection_name, file_hash);
        """)
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
            conn.execute("INSERT OR IGNORE INTO collections (name) VALUES (?)", (name,))
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
                   (doc_id, collection_name, doc_name, doc_description, file_path, file_hash, doc_type, structure, pages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, collection, doc.get("doc_name"), doc.get("doc_description"),
                 doc.get("file_path"), doc.get("file_hash"), doc["doc_type"],
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
            "SELECT doc_id, doc_name, doc_description, file_path, doc_type FROM documents WHERE doc_id = ? AND collection_name = ?",
            (doc_id, collection),
        ).fetchone()
        if not row:
            return {}
        return {"doc_id": row[0], "doc_name": row[1], "doc_description": row[2],
                "file_path": row[3], "doc_type": row[4]}

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
            "SELECT doc_id, doc_name, doc_description, doc_type FROM documents WHERE collection_name = ? ORDER BY created_at",
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
