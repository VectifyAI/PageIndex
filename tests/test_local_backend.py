# tests/sdk/test_local_backend.py
import asyncio
import json
import pytest
from pathlib import Path
from pageindex.backend.local import LocalBackend
from pageindex.storage.sqlite import SQLiteStorage
from pageindex.errors import FileTypeError, DocumentNotFoundError


@pytest.fixture
def backend(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "test.db"))
    files_dir = tmp_path / "files"
    return LocalBackend(storage=storage, files_dir=str(files_dir), model="gpt-4o")


def test_collection_lifecycle(backend):
    backend.get_or_create_collection("papers")
    assert "papers" in backend.list_collections()
    backend.delete_collection("papers")
    assert "papers" not in backend.list_collections()


def test_list_documents_empty(backend):
    backend.get_or_create_collection("papers")
    assert backend.list_documents("papers") == []


def test_unsupported_file_type_raises(backend, tmp_path):
    backend.get_or_create_collection("papers")
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("hello")
    with pytest.raises(FileTypeError):
        backend.add_document("papers", str(bad_file))


def test_register_custom_parser(backend):
    from pageindex.parser.protocol import ParsedDocument, ContentNode

    class TxtParser:
        def supported_extensions(self):
            return [".txt"]
        def parse(self, file_path, **kwargs):
            text = Path(file_path).read_text()
            return ParsedDocument(doc_name="test", nodes=[
                ContentNode(content=text, tokens=len(text.split()), title="Content", index=1, level=1)
            ])

    backend.register_parser(TxtParser())
    # Now .txt should be supported (won't raise FileTypeError)
    assert backend._resolve_parser("test.txt") is not None


# ── Scoped-mode agent tools ──────────────────────────────────────────────────

@pytest.fixture
def populated_backend(backend):
    """Backend with a 'papers' collection containing two stub docs."""
    backend.get_or_create_collection("papers")
    for did, name, desc in [
        ("d1", "alpha.pdf", "About alpha."),
        ("d2", "beta.pdf", "About beta."),
    ]:
        backend._storage.save_document("papers", did, {
            "doc_name": name, "doc_description": desc,
            "doc_type": "pdf", "file_path": f"/tmp/{name}", "structure": [],
        })
    return backend


def _invoke_tool(tool, args: dict) -> str:
    """Run a FunctionTool synchronously with a minimal ToolContext."""
    from agents.tool_context import ToolContext
    ctx = ToolContext(context=None, tool_name=tool.name,
                      tool_call_id="test", tool_arguments=json.dumps(args))
    return asyncio.run(tool.on_invoke_tool(ctx, json.dumps(args)))


def test_open_mode_includes_list_documents(populated_backend):
    tools = populated_backend.get_agent_tools("papers", doc_ids=None)
    names = {t.name for t in tools.function_tools}
    assert names == {"list_documents", "get_document", "get_document_structure", "get_page_content"}


def test_scoped_mode_excludes_list_documents(populated_backend):
    tools = populated_backend.get_agent_tools("papers", doc_ids=["d1"])
    names = {t.name for t in tools.function_tools}
    assert "list_documents" not in names
    assert names == {"get_document", "get_document_structure", "get_page_content"}


def test_scoped_mode_rejects_out_of_scope_doc_id(populated_backend):
    tools = populated_backend.get_agent_tools("papers", doc_ids=["d1"])
    by_name = {t.name: t for t in tools.function_tools}
    out = json.loads(_invoke_tool(by_name["get_document"], {"doc_id": "d2"}))
    assert "error" in out
    assert "not in scope" in out["error"]
    assert out["allowed_doc_ids"] == ["d1"]


def test_scoped_mode_allows_in_scope_doc_id(populated_backend):
    tools = populated_backend.get_agent_tools("papers", doc_ids=["d1"])
    by_name = {t.name: t for t in tools.function_tools}
    out = json.loads(_invoke_tool(by_name["get_document"], {"doc_id": "d1"}))
    assert out.get("doc_name") == "alpha.pdf"


def test_wrap_with_doc_context_single(populated_backend):
    from pageindex.agent import wrap_with_doc_context
    docs = populated_backend._scoped_docs("papers", ["d1"])
    wrapped = wrap_with_doc_context(docs, "what is this?")
    assert "d1: alpha.pdf — About alpha." in wrapped
    assert "specified the following document" in wrapped
    assert "<docs>" in wrapped and "</docs>" in wrapped
    assert "User question: what is this?" in wrapped


def test_wrap_with_doc_context_multi(populated_backend):
    from pageindex.agent import wrap_with_doc_context
    docs = populated_backend._scoped_docs("papers", ["d1", "d2"])
    wrapped = wrap_with_doc_context(docs, "compare them")
    assert "d1: alpha.pdf — About alpha." in wrapped
    assert "d2: beta.pdf — About beta." in wrapped
    assert "specified the following documents" in wrapped
    assert "<docs>" in wrapped and "</docs>" in wrapped
    assert "User question: compare them" in wrapped


def test_scoped_docs_raises_on_missing(populated_backend):
    with pytest.raises(DocumentNotFoundError, match="nonexistent"):
        populated_backend._scoped_docs("papers", ["d1", "nonexistent"])


def test_normalize_doc_ids():
    assert LocalBackend._normalize_doc_ids("d1") == ["d1"]
    assert LocalBackend._normalize_doc_ids(["d1", "d2"]) == ["d1", "d2"]
    assert LocalBackend._normalize_doc_ids(None) is None


def test_normalize_doc_ids_rejects_empty_list():
    with pytest.raises(ValueError, match="cannot be empty"):
        LocalBackend._normalize_doc_ids([])


# ── error taxonomy: missing docs raise DocumentNotFoundError ─────────────────

def test_get_document_missing_raises(backend):
    backend.get_or_create_collection("papers")
    with pytest.raises(DocumentNotFoundError, match="ghost"):
        backend.get_document("papers", "ghost")


def test_delete_document_missing_raises(backend):
    backend.get_or_create_collection("papers")
    with pytest.raises(DocumentNotFoundError, match="ghost"):
        backend.delete_document("papers", "ghost")


def test_delete_collection_rejects_path_traversal(backend, tmp_path):
    # Regression: an unvalidated name like "../.." would rmtree outside files_dir.
    from pageindex.errors import PageIndexError
    canary = tmp_path / "canary.txt"
    canary.write_text("still here")
    with pytest.raises(PageIndexError, match="Invalid collection name"):
        backend.delete_collection("../..")
    assert canary.exists()


def test_add_document_missing_file_raises_file_not_found(backend, tmp_path):
    backend.get_or_create_collection("papers")
    with pytest.raises(FileNotFoundError):
        backend.add_document("papers", str(tmp_path / "nope.pdf"))


def test_add_document_unknown_collection_fails_fast(backend, tmp_path):
    from pageindex.errors import CollectionNotFoundError
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    # Collection never created -> must raise before any parse/LLM work.
    with pytest.raises(CollectionNotFoundError, match="does not exist"):
        backend.add_document("ghost-collection", str(pdf))


def test_add_document_race_returns_existing_id(backend, tmp_path, monkeypatch):
    """If the pre-check misses but the INSERT hits UNIQUE (concurrent add),
    add_document must clean up and return the winner's doc_id, not duplicate."""
    import pageindex.backend.local as local_mod
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 body")
    backend.get_or_create_collection("papers")

    # Pretend a winning add already stored this content under "winner-id".
    file_hash = backend._file_hash(str(pdf))
    backend._storage.save_document("papers", "winner-id", {
        "doc_name": "doc", "doc_type": "pdf", "file_hash": file_hash, "structure": [],
    })
    # Pre-check misses (returns None) so we reach the INSERT; the post-conflict
    # lookup then returns the winner's id.
    calls = {"n": 0}
    def fake_find(col, h):
        calls["n"] += 1
        return None if calls["n"] == 1 else "winner-id"
    monkeypatch.setattr(backend._storage, "find_document_by_hash", fake_find)
    # avoid real parsing/LLM: stub parser + build_index
    monkeypatch.setattr(backend, "_resolve_parser", lambda p: type("P", (), {
        "parse": lambda self, fp, **k: type("PD", (), {"doc_name": "doc", "nodes": []})()
    })())
    monkeypatch.setattr(local_mod, "build_index", lambda parsed, model=None, opt=None: {"structure": [], "doc_description": ""})

    result = backend.add_document("papers", str(pdf))
    assert result == "winner-id"
