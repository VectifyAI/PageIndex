import pytest
from pageindex.storage.sqlite import SQLiteStorage

@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(str(tmp_path / "test.db"))

def test_create_and_list_collections(storage):
    storage.create_collection("papers")
    assert "papers" in storage.list_collections()

def test_get_or_create_collection_idempotent(storage):
    storage.get_or_create_collection("papers")
    storage.get_or_create_collection("papers")
    assert storage.list_collections().count("papers") == 1

def test_delete_collection(storage):
    storage.create_collection("papers")
    storage.delete_collection("papers")
    assert "papers" not in storage.list_collections()


def test_create_duplicate_collection_raises_pageindex_error(storage):
    """A raw sqlite3.IntegrityError leaking out breaks `except PageIndexError`
    catch-alls; must be translated to a proper SDK exception."""
    from pageindex.errors import CollectionAlreadyExistsError, PageIndexError
    storage.create_collection("papers")
    with pytest.raises(CollectionAlreadyExistsError):
        storage.create_collection("papers")
    # also catchable via the SDK's generic base class
    storage.create_collection("other")
    with pytest.raises(PageIndexError):
        storage.create_collection("other")


@pytest.mark.parametrize("bad_name", [
    "a/../../etc/passwd", "/etc/passwd", "a$(whoami)", ".hidden",
    "a b", "válid", "", "a" * 129,
])
def test_create_collection_rejects_invalid_names_at_the_python_layer(storage, bad_name):
    """SQLiteStorage must validate collection names itself — it's a public
    StorageEngine that can be used directly, bypassing LocalBackend's own
    regex check entirely."""
    from pageindex.errors import PageIndexError
    with pytest.raises(PageIndexError):
        storage.create_collection(bad_name)


def test_sql_check_constraint_also_rejects_invalid_names_directly(storage):
    """Defense-in-depth: even bypassing SQLiteStorage's own Python validation
    and inserting via raw SQL, the schema's CHECK constraint must reject a
    name that isn't ENTIRELY [a-zA-Z0-9_-] — not just its first character
    (GLOB '*' is a wildcard, not a regex quantifier over the preceding class,
    so 'name GLOB [a-zA-Z0-9_-]*' alone only constrains the first character)."""
    import sqlite3
    conn = storage._get_conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO collections (name) VALUES (?)", ("a/../../etc/passwd",))


def test_malicious_collection_name_rejected_through_local_backend_too(tmp_path):
    """End-to-end via the normal LocalBackend entry point: a path-traversal-
    shaped collection name must never reach add_document's
    files_dir / collection path construction. Three independent layers now
    reject it (LocalBackend's own regex, SQLiteStorage's regex, and the SQL
    CHECK constraint) — this pins the outermost one."""
    from pageindex.backend.local import LocalBackend
    from pageindex.errors import PageIndexError

    storage = SQLiteStorage(str(tmp_path / "t.db"))
    backend = LocalBackend(storage=storage, files_dir=str(tmp_path / "files"), model="gpt-4o")
    with pytest.raises(PageIndexError):
        backend.create_collection("a/../../escape_me")
    assert not (tmp_path / "escape_me").exists()

def test_save_and_get_document(storage):
    storage.create_collection("papers")
    doc = {
        "doc_name": "test.pdf", "doc_description": "A test",
        "file_path": "/tmp/test.pdf", "doc_type": "pdf",
        "structure": [{"title": "Intro", "node_id": "0001"}],
    }
    storage.save_document("papers", "doc-1", doc)
    result = storage.get_document("papers", "doc-1")
    assert result["doc_name"] == "test.pdf"
    assert result["doc_type"] == "pdf"

def test_get_document_structure(storage):
    storage.create_collection("papers")
    structure = [{"title": "Ch1", "node_id": "0001", "nodes": []}]
    storage.save_document("papers", "doc-1", {
        "doc_name": "test.pdf", "doc_type": "pdf",
        "file_path": "/tmp/test.pdf", "structure": structure,
    })
    result = storage.get_document_structure("papers", "doc-1")
    assert result[0]["title"] == "Ch1"

def test_list_documents(storage):
    storage.create_collection("papers")
    storage.save_document("papers", "doc-1", {"doc_name": "p1.pdf", "doc_type": "pdf", "file_path": "/tmp/p1.pdf", "structure": []})
    storage.save_document("papers", "doc-2", {"doc_name": "p2.pdf", "doc_type": "pdf", "file_path": "/tmp/p2.pdf", "structure": []})
    docs = storage.list_documents("papers")
    assert len(docs) == 2

def test_delete_document(storage):
    storage.create_collection("papers")
    storage.save_document("papers", "doc-1", {"doc_name": "test.pdf", "doc_type": "pdf", "file_path": "/tmp/test.pdf", "structure": []})
    storage.delete_document("papers", "doc-1")
    assert len(storage.list_documents("papers")) == 0

def test_delete_collection_cascades_documents(storage):
    storage.create_collection("papers")
    storage.save_document("papers", "doc-1", {"doc_name": "test.pdf", "doc_type": "pdf", "file_path": "/tmp/test.pdf", "structure": []})
    storage.delete_collection("papers")
    assert "papers" not in storage.list_collections()


def test_close_closes_connections_created_in_other_threads(storage):
    """Regression: with check_same_thread=True, close() from another thread
    raised ProgrammingError (swallowed) and leaked every worker connection."""
    import sqlite3
    import threading

    conns = {}

    def worker():
        conns["worker"] = storage._get_conn()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    storage.close()  # main thread closes the worker's connection too
    with pytest.raises(sqlite3.ProgrammingError):
        conns["worker"].execute("SELECT 1")


def test_duplicate_file_hash_in_collection_raises(storage):
    """UNIQUE(collection_name, file_hash) guards the add-same-file race."""
    import sqlite3
    storage.create_collection("papers")
    doc = {"doc_name": "a", "doc_type": "pdf", "file_hash": "HASH1", "structure": []}
    storage.save_document("papers", "doc-1", doc)
    with pytest.raises(sqlite3.IntegrityError):
        storage.save_document("papers", "doc-2", {**doc, "doc_name": "b"})
    # same hash in a DIFFERENT collection is fine
    storage.create_collection("other")
    storage.save_document("other", "doc-3", {**doc})


def test_concurrent_read_then_write_no_database_locked(storage):
    """Regression: concurrent add (read hash -> write) hit 'database is locked'
    under WAL. Fixed via autocommit + busy_timeout + write lock. All writers
    must succeed (dedup via UNIQUE), none raise OperationalError."""
    import sqlite3, threading, uuid, time
    storage.create_collection("c")
    errs = []

    def worker():
        try:
            storage.list_collections()
            storage.find_document_by_hash("c", "SAME")  # read snapshot
            time.sleep(0.001)                            # widen the window
            try:
                storage.save_document("c", str(uuid.uuid4()),
                    {"doc_name": "d", "doc_type": "pdf", "file_hash": "SAME", "structure": []})
            except sqlite3.IntegrityError:
                pass  # expected: lost the dedup race
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errs, f"concurrent write errored: {errs}"
    assert len(storage.list_documents("c")) == 1  # dedup held
