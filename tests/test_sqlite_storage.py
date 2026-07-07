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
