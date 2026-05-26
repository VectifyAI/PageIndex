import json


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
                "source_path": "documents/report.pdf",
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
                "metadata_text": "",
                "content": "",
                "skip_fts": True,
            }
        ]
    )

    assert not any(
        statement.upper().replace(" ", "") == "PRAGMASYNCHRONOUS=OFF"
        for statement in statements
    )
