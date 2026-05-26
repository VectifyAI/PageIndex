from pathlib import Path

import pytest


class SummaryGenerator:
    def generate(self, document, *, fields):
        return {field: "Generated registration summary." for field in fields}


class RecordingSummaryIndexer:
    def __init__(self):
        self.upserted = []

    def upsert_summary(self, record):
        self.upserted.append(dict(record))
        return {"status": "ready"}


def test_register_insert_failure_cleans_owned_artifacts_and_skips_projection(
    tmp_path: Path, monkeypatch
):
    from pageindex.filesystem import PageIndexFileSystem

    workspace = tmp_path / "workspace"
    source = tmp_path / "source.txt"
    source.write_text("Plain text content for registration.", encoding="utf-8")
    indexer = RecordingSummaryIndexer()
    filesystem = PageIndexFileSystem(
        workspace=workspace,
        metadata_generator=SummaryGenerator(),
        summary_projection_indexer=indexer,
    )

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path="docs/source.txt",
            folder_path="/documents",
            external_id="doc_insert_failure",
            title="Insert failure",
            content=source.read_text(encoding="utf-8"),
            metadata_policy={
                "fields": {
                    "summary": True,
                    "doc_type": False,
                    "domain": False,
                    "topic": False,
                }
            },
        )

    assert indexer.upserted == []
    assert list((workspace / "artifacts" / "raw").glob("*.json")) == []
    assert list((workspace / "artifacts" / "text").glob("*.txt")) == []
