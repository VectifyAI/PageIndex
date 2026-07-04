import json
from pathlib import Path

import pytest


class StaticEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_summary_indexer(workspace: Path):
    from pageindex.filesystem.semantic_projection import SummaryProjectionIndexer

    return SummaryProjectionIndexer(
        workspace / "artifacts" / "projection_indexes",
        embedder=StaticEmbedder(),
        embedding_provider="test",
        embedding_model="static",
        embedding_dimensions=3,
    )


def make_filesystem(workspace: Path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(
        workspace=workspace,
        summary_projection_embedding_provider="test",
        summary_projection_embedding_model="static",
        summary_projection_embedding_dimensions=3,
    )
    filesystem.summary_projection_indexer = make_summary_indexer(workspace)
    return filesystem


@pytest.fixture(autouse=True)
def fake_pageindex_index(monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        path = Path(file_path)
        doc_id = f"doc_{path.stem}"
        text = path.read_text(encoding="utf-8")
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(path.resolve()),
            "doc_name": path.name,
            "doc_description": f"Summary for {path.name}: {text[:60]}",
            "line_count": len(text.splitlines()),
            "structure": [
                {
                    "title": path.stem,
                    "node_id": "0001",
                    "line_num": 1,
                    "text": text,
                    "nodes": [],
                }
            ],
            "pages": [{"page": 1, "content": text}],
        }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)


def write_pageindex_client_doc(workspace: Path, doc_id: str, doc: dict) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / f"{doc_id}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        doc_id: {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
            "line_count": doc.get("line_count"),
        }
    }
    (workspace / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_add_text_folder_target_copies_artifact_indexes_summary_and_is_readable(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor

    source = tmp_path / "filing.md"
    source.write_text("alpha filing text for pifs add", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    info = filesystem.add_file(str(source), "/documents/reports")

    assert info["path"] == "/documents/reports/filing.md"
    assert info["folder_path"] == "/documents/reports"
    assert filesystem.folder_info("/documents/reports")["path"] == "/documents/reports"
    entry = filesystem.store.get_file(info["file_ref"])
    assert entry.storage_uri != source.as_uri()
    assert "/artifacts/uploads/" in entry.storage_uri
    copied_path = Path(entry.storage_uri.removeprefix("file://"))
    assert copied_path.read_text(encoding="utf-8") == "alpha filing text for pifs add"
    assert copied_path.resolve() != source.resolve()

    executor = PIFSCommandExecutor(filesystem)
    rendered = json.loads(executor.execute("grep alpha /documents/reports/filing.md"))

    assert rendered["data"]["matches"] == [
        {"line": 1, "text": "alpha filing text for pifs add"}
    ]
    assert info["metadata"]["summary"].startswith("Summary for filing.md")
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 1


def test_add_rejects_same_folder_same_basename_without_overwrite(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor

    source = tmp_path / "conflict.md"
    source.write_text("first body", encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    filesystem.add_file(source, "/documents")
    source.write_text("second body must not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        filesystem.add_file(source, "/documents")

    executor = PIFSCommandExecutor(filesystem)
    rendered = json.loads(executor.execute("grep first /documents/conflict.md"))
    assert rendered["data"]["matches"] == [{"line": 1, "text": "first body"}]


def test_add_rejects_unsupported_type_before_registration(tmp_path):
    source = tmp_path / "payload.json"
    source.write_text('{"unsupported": true}', encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    with pytest.raises(ValueError, match="Unsupported file type"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert not list((tmp_path / "workspace" / "artifacts" / "uploads").glob("**/*"))


def test_add_configures_semantic_retrieval_in_same_filesystem_instance(tmp_path):
    source = tmp_path / "semantic.md"
    source.write_text("alpha semantic recall text", encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    assert filesystem.semantic_retrieval_channels() == ()

    filesystem.add_file(source, "/documents")

    assert filesystem.semantic_retrieval_channels() == ("summary",)
    results = filesystem.browse_semantic_files(
        "/documents",
        "semantic recall",
        recursive=True,
        page_size=5,
    )
    assert [item["path"] for item in results["data"]] == ["/documents/semantic.md"]


def test_add_markdown_builds_pageindex_tree_from_copied_artifact(tmp_path, monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor

    indexed_paths = []

    def fake_index(self, file_path, mode="auto"):
        indexed_paths.append(Path(file_path))
        doc_id = "doc_added_md"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "notes.md",
            "doc_description": "summary",
            "line_count": 3,
            "structure": [
                {
                    "title": "Notes",
                    "node_id": "0001",
                    "line_num": 1,
                    "text": "# Notes\n\ncopied markdown body",
                    "nodes": [],
                }
            ],
        }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\ncopied markdown body", encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    info = filesystem.add_file(source, "/documents")
    executor = PIFSCommandExecutor(filesystem)
    structure = json.loads(executor.execute("cat /documents/notes.md --structure"))
    entry = filesystem.store.get_file(info["file_ref"])

    assert structure["data"]["document"]["available"] is True
    assert structure["data"]["structure"][0]["title"] == "Notes"
    assert indexed_paths == [Path(entry.storage_uri.removeprefix("file://"))]
    assert indexed_paths[0].resolve() != source.resolve()


def test_add_failure_does_not_leave_visible_catalog_or_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "atomic.md"
    source.write_text("atomic body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_markdown_insert_failure_removes_pageindex_cache(tmp_path, monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = "doc_failed_add_md"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "failed.md",
            "doc_description": "summary",
            "line_count": 3,
            "structure": [
                {
                    "title": "Failed",
                    "node_id": "0001",
                    "line_num": 1,
                    "text": "# Failed\n\nbody",
                    "nodes": [],
                }
            ],
        }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "failed.md"
    source.write_text("# Failed\n\nbody", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.add_file(source, "/documents/reports")

    pageindex_workspace = workspace / "artifacts" / "pageindex_client"
    assert not (pageindex_workspace / "doc_failed_add_md.json").exists()
    meta_path = pageindex_workspace / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "doc_failed_add_md" not in meta
    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert listing["folders"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_markdown_index_failure_removes_pageindex_cache_delta(tmp_path, monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = "doc_partial_before_raise"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "partial.md",
            "doc_description": "summary",
            "line_count": 3,
            "structure": [{"title": "Partial", "node_id": "0001", "nodes": []}],
        }
        self.documents[doc_id] = doc
        self._save_doc(doc_id)
        raise RuntimeError("index failed after cache write")

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "partial.md"
    source.write_text("# Partial\n\nbody", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)
    pageindex_workspace = workspace / "artifacts" / "pageindex_client"

    with pytest.raises(RuntimeError, match="requires PageIndex extraction"):
        filesystem.add_file(source, "/documents/reports")

    assert not (pageindex_workspace / "doc_partial_before_raise.json").exists()
    meta_path = pageindex_workspace / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "doc_partial_before_raise" not in meta
    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert listing["folders"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_markdown_failure_preserves_unrelated_pageindex_cache(tmp_path, monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = "doc_failed_add_md"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "failed.md",
            "doc_description": "summary",
            "line_count": 3,
            "structure": [{"title": "Failed", "node_id": "0001", "nodes": []}],
        }
        self.documents[doc_id] = doc
        self._save_doc(doc_id)
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "failed.md"
    source.write_text("# Failed\n\nbody", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)
    pageindex_workspace = workspace / "artifacts" / "pageindex_client"
    write_pageindex_client_doc(
        pageindex_workspace,
        "doc_unrelated",
        {
            "id": "doc_unrelated",
            "type": "md",
            "path": str((tmp_path / "unrelated.md").resolve()),
            "doc_name": "unrelated.md",
            "doc_description": "summary",
            "line_count": 1,
            "structure": [{"title": "Unrelated", "node_id": "0001", "nodes": []}],
        },
    )

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.add_file(source, "/documents")

    assert not (pageindex_workspace / "doc_failed_add_md.json").exists()
    assert (pageindex_workspace / "doc_unrelated.json").exists()
    meta = json.loads((pageindex_workspace / "_meta.json").read_text(encoding="utf-8"))
    assert "doc_failed_add_md" not in meta
    assert "doc_unrelated" in meta


def test_add_failure_after_summary_vector_rolls_back_catalog_and_vector(
    tmp_path, monkeypatch
):
    source = tmp_path / "post_vector.md"
    source.write_text("post vector rollback body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_status_update(*args, **kwargs):
        raise RuntimeError("metadata status update failed")

    monkeypatch.setattr(filesystem.store, "update_file_metadata_status", fail_status_update)

    with pytest.raises(RuntimeError, match="metadata status update failed"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_failure_removes_nested_folders_created_only_for_add(tmp_path, monkeypatch):
    source = tmp_path / "nested.md"
    source.write_text("nested rollback body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_status_update(*args, **kwargs):
        raise RuntimeError("metadata status update failed")

    monkeypatch.setattr(filesystem.store, "update_file_metadata_status", fail_status_update)

    with pytest.raises(RuntimeError, match="metadata status update failed"):
        filesystem.add_file(source, "/documents/reports")

    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert listing["folders"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_failure_preserves_preexisting_parent_folder(tmp_path, monkeypatch):
    source = tmp_path / "nested.md"
    source.write_text("nested rollback body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)
    filesystem.create_folder("/documents")

    def fail_status_update(*args, **kwargs):
        raise RuntimeError("metadata status update failed")

    monkeypatch.setattr(filesystem.store, "update_file_metadata_status", fail_status_update)

    with pytest.raises(RuntimeError, match="metadata status update failed"):
        filesystem.add_file(source, "/documents/reports")

    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert [folder["path"] for folder in listing["folders"]] == ["/documents"]
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0


def test_cli_add_uses_workspace_and_prints_added_file(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    source = tmp_path / "cli.md"
    source.write_text("cli body", encoding="utf-8")
    calls = []

    class FakeAddFileSystem:
        def __init__(self, workspace, **_kwargs):
            self.workspace = Path(workspace)

        def configure_existing_projection_retrieval(self):
            return False

        def add_file(self, physical_path, virtual_target):
            calls.append((self.workspace, physical_path, virtual_target))
            return {
                "file_ref": "file_cli",
                "path": "/documents/cli.md",
            }

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeAddFileSystem)

    status = cli.main(["--workspace", str(tmp_path / "workspace"), "add", str(source), "/documents"])

    assert status == 0
    assert calls == [(tmp_path / "workspace", str(source), "/documents")]
    assert capsys.readouterr().out == (
        "added: /documents/cli.md\n"
        "file_ref: file_cli\n"
    )
