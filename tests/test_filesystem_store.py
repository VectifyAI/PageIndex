import json

import pytest

from tests.pifs_markdown_fixture import register_markdown


def test_insert_files_does_not_disable_sqlite_synchronous(tmp_path):
    from pageindex.filesystem.store import SQLiteFileSystemStore

    statements = []

    class RecordingStore(SQLiteFileSystemStore):
        def connect(self):
            conn = super().connect()
            conn.set_trace_callback(statements.append)
            return conn

    store = RecordingStore(tmp_path / "workspace")
    statements.clear()

    store.insert_files(
        [
            {
                "file_ref": "ref_report",
                "external_id": "doc_report",
                "storage_uri": "file:///tmp/report.pdf",
                "folder_path": "/documents",
                "title": "Report",
                "descriptor": "documents/report.pdf",
                "content_type": "application/pdf",
                "source_type": "documents",
                "fingerprint": "fingerprint",
                "text_artifact_path": "artifacts/text/ref_report.txt",
                "raw_artifact_path": None,
                "metadata": {},
                "metadata_json": json.dumps({}),
            }
        ]
    )

    assert not any(
        statement.upper().replace(" ", "") == "PRAGMASYNCHRONOUS=OFF"
        for statement in statements
    )


def test_register_file_rejects_parent_directory_segments(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")

    with pytest.raises(ValueError, match="must not contain '\\.\\.'"):
        filesystem.register_file(
            storage_uri="file:///tmp/report.txt",
            folder_path="/a/../b",
            external_id="doc_bad",
            title="Bad",
            content="bad body",
        )


def test_file_upsert_preserves_created_at(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    file_ref = register_markdown(filesystem, tmp_path, "doc_report", "/documents", text="version one")
    with filesystem.store.connect() as conn:
        conn.execute(
            "UPDATE files SET created_at = '2001-02-03 04:05:06' WHERE file_ref = ?",
            (file_ref,),
        )
    register_markdown(filesystem, tmp_path, "doc_report", "/documents", text="version two")

    with filesystem.store.connect() as conn:
        row = conn.execute(
            "SELECT created_at, storage_uri, deleted_at FROM files WHERE file_ref = ?",
            (file_ref,),
        ).fetchone()

    assert row["created_at"] == "2001-02-03 04:05:06"
    assert row["storage_uri"].endswith("/doc_report.md")
    assert row["deleted_at"] is None


def test_file_upsert_preserves_soft_delete_tombstone(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    file_ref = register_markdown(filesystem, tmp_path, "doc_report", "/documents", text="version one")
    with filesystem.store.connect() as conn:
        conn.execute(
            "UPDATE files SET deleted_at = '2001-02-03 04:05:06' WHERE file_ref = ?",
            (file_ref,),
        )
    register_markdown(filesystem, tmp_path, "doc_report", "/documents", text="version two")

    with filesystem.store.connect() as conn:
        row = conn.execute(
            "SELECT storage_uri, deleted_at FROM files WHERE file_ref = ?",
            (file_ref,),
        ).fetchone()

    assert row["storage_uri"].endswith("/doc_report.md")
    assert row["deleted_at"] == "2001-02-03 04:05:06"
    assert filesystem.store.list_files() == []


def test_listing_uses_one_consistent_folder_membership_row(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    file_ref = register_markdown(
        filesystem,
        tmp_path,
        "doc_shared",
        "/a",
        title="Original.md",
        text="shared body",
    )
    filesystem.store.attach_file_to_folder(
        file_ref,
        "/b",
        metadata={"display_name": "Alpha"},
    )

    listing = filesystem.store.list_folder("/", recursive=True, limit=10)
    row = next(item for item in listing["files"] if item["external_id"] == "doc_shared")

    assert row["folder_path"] == "/a"
    assert row["title"] == "Original.md"
