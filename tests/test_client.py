"""SDK surface tests: PageIndexClient in local and cloud mode."""
import json
import re
import shutil
import sys
import types

import pytest

import pageindex
import pageindex.flash
import pageindex.utils
from pageindex import PageIndexClient, PageIndexAPIError
from pageindex.local_api import LocalAPI

# `from .page_index import *` shadows the submodule with the function of the
# same name, so the module must come from sys.modules.
page_index_module = sys.modules["pageindex.page_index"]


STRUCTURE = [
    {
        "title": "Root Section", "node_id": "0000",
        "start_index": 1, "end_index": 2,
        "summary": "root summary", "text": "root text",
        "nodes": [
            {"title": "Child Section", "node_id": "0001",
             "start_index": 2, "end_index": 2,
             "summary": "child summary", "text": "child text"},
        ],
    },
]


@pytest.fixture
def local_client(tmp_path):
    return PageIndexClient(storage_path=str(tmp_path / "store"))


@pytest.fixture
def indexed_doc(local_client, sample_pdf, monkeypatch):
    """A document indexed through a stubbed standard pipeline."""
    def fake_page_index_main(doc, opt=None, logger=None):
        assert opt.if_add_node_summary == "yes"
        assert opt.if_add_node_text == "yes"
        assert logger is not None
        return {"doc_name": "sample.pdf",
                "doc_description": "A test document.",
                "structure": json.loads(json.dumps(STRUCTURE))}
    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    return local_client.submit_document(sample_pdf)["doc_id"]


# ── constructor ──

def test_empty_api_key_raises():
    with pytest.raises(PageIndexAPIError, match="empty string"):
        PageIndexClient(api_key="")


def test_cloud_rejects_local_args():
    with pytest.raises(ValueError, match="model, storage_path"):
        PageIndexClient(api_key="k", model="m", storage_path="/tmp/x")


def test_local_client_does_not_touch_disk(tmp_path):
    storage = tmp_path / "store"
    PageIndexClient(storage_path=str(storage))
    assert not storage.exists()


def test_explicit_mode_clients(tmp_path):
    from pageindex import PageIndexCloudClient, PageIndexLocalClient

    for bad_key in (None, ""):
        with pytest.raises(PageIndexAPIError, match="requires a PageIndex API key"):
            PageIndexCloudClient(bad_key)
    cloud = PageIndexCloudClient("k")
    assert cloud.api_key == "k" and isinstance(cloud, PageIndexClient)

    local = PageIndexLocalClient(model="m", storage_path=str(tmp_path / "s"))
    assert local.model == "m" and isinstance(local, PageIndexClient)
    with pytest.raises(TypeError):
        PageIndexLocalClient("k")


# ── local: indexing and reading ──

def test_submit_and_get_tree(local_client, indexed_doc, tmp_path, monkeypatch):
    tree = local_client.get_tree(indexed_doc, node_summary=True)
    assert tree["status"] == "completed"
    assert tree["retrieval_ready"] is True
    root = tree["result"][0]
    assert root["page_index"] == 1
    assert "start_index" not in root and "end_index" not in root
    assert root["prefix_summary"] == "root summary"
    assert "summary" not in root
    child = root["nodes"][0]
    assert child["summary"] == "child summary"
    assert child["text"] == "child text"

    no_summary = local_client.get_tree(indexed_doc)["result"][0]
    assert "summary" not in no_summary and "prefix_summary" not in no_summary


def test_submit_does_not_create_cwd_logs(local_client, sample_pdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def fake_page_index_main(doc, opt=None, logger=None):
        logger.info({"probe": True})
        return {"doc_name": "sample.pdf", "doc_description": None,
                "structure": json.loads(json.dumps(STRUCTURE))}
    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    local_client.submit_document(sample_pdf)
    assert not (tmp_path / "logs").exists()


def test_submit_flash(local_client, sample_pdf, monkeypatch):
    calls = {}
    def fake_flash(pdf, summary=True, summary_model=None, **kwargs):
        calls["summary"] = summary
        calls["summary_model"] = summary_model
        return {"doc_name": "sample.pdf",
                "structure": [{"title": "Flash Root", "start_index": 1,
                               "end_index": 2, "summary": "s", "nodes": []}]}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", fake_flash)
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda model, prompt, **kw: "Flash description.")
    doc_id = local_client.submit_document(sample_pdf, mode="flash")["doc_id"]
    assert calls == {"summary": True, "summary_model": local_client.summary_model}
    root = local_client.get_tree(doc_id)["result"][0]
    assert root["node_id"] == "0000"
    assert "Hello page one" in root["text"]
    assert local_client.get_document(doc_id)["description"] == "Flash description."


def test_submit_rejections(local_client, sample_pdf, tmp_path):
    with pytest.raises(FileNotFoundError):
        local_client.submit_document(str(tmp_path / "missing.pdf"))
    (tmp_path / "notes.txt").write_text("hi")
    with pytest.raises(PageIndexAPIError, match="only PDF"):
        local_client.submit_document(str(tmp_path / "notes.txt"))
    with pytest.raises(PageIndexAPIError, match="unknown local processing mode"):
        local_client.submit_document(sample_pdf, mode="mcp")
    with pytest.raises(PageIndexAPIError, match="folders"):
        local_client.submit_document(sample_pdf, folder_id="f1")
    with pytest.raises(PageIndexAPIError, match="beta_headers"):
        local_client.submit_document(sample_pdf, beta_headers=["block_reference"])


def test_submit_with_metadata(local_client, sample_pdf, monkeypatch):
    monkeypatch.setattr(
        page_index_module, "page_index_main",
        lambda doc, opt=None, logger=None: {
            "doc_name": "sample.pdf", "doc_description": None,
            "structure": json.loads(json.dumps(STRUCTURE))})
    tags = {"project": "alpha", "year": 2026}
    doc_id = local_client.submit_document(sample_pdf, metadata=tags)["doc_id"]
    assert local_client.get_tree(doc_id)["metadata"] == tags
    assert local_client.get_ocr(doc_id)["metadata"] == tags
    assert local_client.list_documents()["documents"][0]["metadata"] == tags
    assert "metadata" not in local_client.get_document(doc_id)


def test_submit_metadata_validation(local_client, sample_pdf, monkeypatch):
    indexed = []
    monkeypatch.setattr(page_index_module, "page_index_main",
                        lambda *args, **kwargs: indexed.append(1))
    with pytest.raises(PageIndexAPIError, match="metadata must be a dict"):
        local_client.submit_document(sample_pdf, metadata=["not", "a", "dict"])
    with pytest.raises(PageIndexAPIError, match="valid JSON"):
        local_client.submit_document(sample_pdf, metadata={"x": object()})
    assert indexed == []


def test_blank_pdf_rejected(local_client, tmp_path):
    from conftest import build_pdf
    blank = tmp_path / "blank.pdf"
    blank.write_bytes(build_pdf(["", ""]))
    with pytest.raises(PageIndexAPIError, match="All pages are blank"):
        local_client.submit_document(str(blank))


def test_get_ocr(local_client, indexed_doc):
    page = local_client.get_ocr(indexed_doc)
    assert page["result"][0] == {"page_index": 1,
                                 "markdown": "Hello page one about apples"}
    raw = local_client.get_ocr(indexed_doc, format="raw")
    assert raw["result"] == ("Hello page one about apples\n\n"
                             "Second page about bananas")
    node = local_client.get_ocr(indexed_doc, format="node")
    assert node["result"] == [
        {"title": "Root Section", "level": 1, "page_index": 1, "text": "root text"},
        {"title": "Child Section", "level": 2, "page_index": 2, "text": "child text"},
    ]
    with pytest.raises(ValueError):
        local_client.get_ocr(indexed_doc, format="bogus")


def test_document_management(local_client, indexed_doc):
    assert indexed_doc.startswith("pi-")
    doc = local_client.get_document(indexed_doc)
    assert doc["id"] == indexed_doc
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3}000)?",
                        doc["createdAt"])
    assert doc["name"] == "sample.pdf"
    assert doc["description"] == "A test document."
    assert doc["status"] == "completed"
    assert doc["pageNum"] == 2
    assert doc["folderId"] is None

    listing = local_client.list_documents()
    assert listing["total"] == 1
    assert listing["limit"] == 50 and listing["offset"] == 0
    assert listing["documents"][0]["id"] == indexed_doc

    assert local_client.is_retrieval_ready(indexed_doc) is True

    assert local_client.delete_document(indexed_doc) == {
        "message": "Document deleted successfully."}
    with pytest.raises(PageIndexAPIError, match="Document not found"):
        local_client.delete_document(indexed_doc)
    assert local_client.is_retrieval_ready(indexed_doc) is False


def test_manifest_write_through_and_self_heal(local_client, indexed_doc, tmp_path):
    manifest_path = tmp_path / "store" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["docs"][indexed_doc]["name"] == "sample.pdf"

    # corrupt cache → listings rebuild it from the doc.json files
    manifest_path.write_text("{broken")
    assert local_client.list_documents()["total"] == 1
    assert indexed_doc in json.loads(manifest_path.read_text())["docs"]

    # missing cache → same
    manifest_path.unlink()
    assert local_client.list_documents()["total"] == 1

    # doc dir removed behind the store's back → healed, not served stale
    shutil.rmtree(tmp_path / "store" / "docs" / indexed_doc)
    assert local_client.list_documents()["total"] == 0

    local_client_meta = json.loads(manifest_path.read_text())
    assert local_client_meta == {"docs": {}}


def test_manifest_updated_on_delete(local_client, indexed_doc, tmp_path):
    local_client.delete_document(indexed_doc)
    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest == {"docs": {}}


def test_manifest_picks_up_external_doc(local_client, indexed_doc, tmp_path):
    # a doc whose manifest update was lost (e.g. concurrent writer) still lists
    docs_dir = tmp_path / "store" / "docs"
    external_id = "11111111-1111-4111-8111-111111111111"
    shutil.copytree(docs_dir / indexed_doc, docs_dir / external_id)
    meta_path = docs_dir / external_id / "doc.json"
    meta = json.loads(meta_path.read_text())
    meta["id"] = external_id
    meta_path.write_text(json.dumps(meta))

    ids = {d["id"] for d in local_client.list_documents()["documents"]}
    assert ids == {indexed_doc, external_id}


def test_manifest_ignores_incomplete_dir(local_client, indexed_doc, tmp_path):
    (tmp_path / "store" / "docs" / "crashed-save").mkdir()
    listing = local_client.list_documents()
    assert listing["total"] == 1
    assert listing["documents"][0]["id"] == indexed_doc


def test_torn_delete_never_lists_ghost(local_client, indexed_doc, tmp_path):
    # crash mid-delete: doc.json gone, dir and manifest entry remain
    doc_dir = tmp_path / "store" / "docs" / indexed_doc
    (doc_dir / "doc.json").unlink()

    assert local_client.list_documents()["total"] == 0
    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest == {"docs": {}}
    with pytest.raises(PageIndexAPIError):
        local_client.get_document(indexed_doc)
    with pytest.raises(PageIndexAPIError, match="Document not found"):
        local_client.delete_document(indexed_doc)
    assert not doc_dir.exists()


def test_corrupt_doc_json_is_contained(local_client, indexed_doc, sample_pdf, tmp_path):
    second = local_client.submit_document(sample_pdf)["doc_id"]
    (tmp_path / "store" / "docs" / indexed_doc / "doc.json").write_text("{truncated")

    # manifest still holds a good copy of the meta — served consistently
    assert local_client.get_document(indexed_doc)["id"] == indexed_doc
    assert local_client.list_documents()["total"] == 2

    # without the manifest copy, the doc is treated as absent, not a crash
    (tmp_path / "store" / "manifest.json").unlink()
    listing = local_client.list_documents()
    assert listing["total"] == 1
    assert listing["documents"][0]["id"] == second
    with pytest.raises(PageIndexAPIError):
        local_client.get_document(indexed_doc)
    assert local_client.is_retrieval_ready(indexed_doc) is False


def test_invalid_utf8_is_contained(local_client, indexed_doc, tmp_path):
    doc_json = tmp_path / "store" / "docs" / indexed_doc / "doc.json"
    doc_json.write_bytes(b'{"id": "\xff\xfe broken')

    # the manifest copy keeps serving, consistently across list and get
    assert local_client.get_document(indexed_doc)["id"] == indexed_doc
    assert local_client.list_documents()["total"] == 1

    # even with the manifest corrupted the same way: no crash, self-heals
    (tmp_path / "store" / "manifest.json").write_bytes(b"\xff\xfe")
    assert local_client.list_documents()["total"] == 0
    with pytest.raises(PageIndexAPIError):
        local_client.get_document(indexed_doc)


def test_corrupt_data_files_fail_loud(local_client, indexed_doc, tmp_path):
    doc_dir = tmp_path / "store" / "docs" / indexed_doc
    (doc_dir / "tree.json").write_bytes(b"\xff\xfe")
    with pytest.raises(PageIndexAPIError, match="unreadable"):
        local_client.get_tree(indexed_doc)
    assert local_client.is_retrieval_ready(indexed_doc) is False

    (doc_dir / "pages.json").write_text("{broken")
    with pytest.raises(PageIndexAPIError, match="unreadable"):
        local_client.get_ocr(indexed_doc)

    # the metadata itself is intact, so listings stay honest
    assert local_client.list_documents()["total"] == 1


def test_delete_survives_marker_tamper(local_client, tmp_path):
    tampered = tmp_path / "store" / "docs" / "tampered" / "doc.json"
    tampered.mkdir(parents=True)
    with pytest.raises(PageIndexAPIError, match="Document not found"):
        local_client.delete_document("tampered")
    assert not tampered.parent.exists()


def test_list_documents_validation(local_client):
    with pytest.raises(ValueError):
        local_client.list_documents(limit=0)
    with pytest.raises(ValueError):
        local_client.list_documents(offset=-1)
    with pytest.raises(PageIndexAPIError, match="folders"):
        local_client.list_documents(folder_id="f1")


def test_missing_document_errors(local_client):
    with pytest.raises(PageIndexAPIError):
        local_client.get_tree("nope")
    with pytest.raises(PageIndexAPIError):
        local_client.get_document("nope")
    assert local_client.is_retrieval_ready("nope") is False


def test_traversal_ids_are_contained(local_client, indexed_doc, tmp_path):
    store_root = tmp_path / "store"
    with pytest.raises(PageIndexAPIError):
        local_client.get_document("../../etc")
    with pytest.raises(PageIndexAPIError):
        local_client.delete_document("..")
    assert (store_root / "docs").exists()


def test_folders_are_cloud_only(local_client):
    with pytest.raises(PageIndexAPIError, match="cloud-only"):
        local_client.create_folder("team")
    with pytest.raises(PageIndexAPIError, match="cloud-only"):
        local_client.list_folders()


# ── local: retrieval endpoints are cloud-only ──

def test_retrieval_endpoints_cloud_only(local_client):
    with pytest.raises(PageIndexAPIError, match="use chat_completions"):
        local_client.submit_query("any", "q")
    with pytest.raises(PageIndexAPIError, match="use chat_completions"):
        local_client.get_retrieval("any")


def test_missing_llm_key_fails_fast(local_client, indexed_doc, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PageIndexAPIError, match="OPENAI_API_KEY is not set"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}], doc_id=indexed_doc)


def test_chat_tree_search_failure(local_client, indexed_doc, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda model, prompt, **kw: "")
    with pytest.raises(PageIndexAPIError, match="no output"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}], doc_id=indexed_doc)


# ── local: chat completions ──

def _fake_completion(content="Bananas are on page two.", finish="stop"):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content),
            finish_reason=finish)],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                    total_tokens=15),
    )


def _fake_stream():
    def chunk(content=None, finish=None, usage=None):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=content),
                finish_reason=finish)] if content or finish else [],
            usage=usage)
    return iter([
        chunk(content="Bananas "),
        chunk(content="page two."),
        chunk(finish="stop",
              usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                          total_tokens=15)),
    ])


@pytest.fixture
def chat_ready(local_client, indexed_doc, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        pageindex.utils, "llm_completion",
        lambda model, prompt, **kw: json.dumps({"thinking": "t", "node_list": ["0001"]}))
    return indexed_doc


def test_chat_completions(local_client, chat_ready, monkeypatch):
    captured = {}
    def fake_chat_llm(self, messages, temperature, stream):
        captured["messages"] = messages
        captured["temperature"] = temperature
        return _fake_completion()
    monkeypatch.setattr(LocalAPI, "_chat_llm", fake_chat_llm)

    response = local_client.chat_completions(
        messages=[{"role": "user", "content": "What about bananas?"}],
        doc_id=chat_ready, temperature=0.2)
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "Bananas are on page two."
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 15
    assert captured["temperature"] == 0.2
    assert captured["messages"][0]["role"] == "system"
    assert "child text" in captured["messages"][0]["content"]
    assert captured["messages"][-1] == {"role": "user", "content": "What about bananas?"}


def test_chat_completions_stream(local_client, chat_ready, monkeypatch):
    monkeypatch.setattr(LocalAPI, "_chat_llm",
                        lambda self, messages, temperature, stream: _fake_stream())
    pieces = list(local_client.chat_completions(
        messages=[{"role": "user", "content": "q"}], doc_id=chat_ready, stream=True))
    assert pieces == ["Bananas ", "page two."]

    monkeypatch.setattr(LocalAPI, "_chat_llm",
                        lambda self, messages, temperature, stream: _fake_stream())
    chunks = list(local_client.chat_completions(
        messages=[{"role": "user", "content": "q"}], doc_id=chat_ready,
        stream=True, stream_metadata=True))
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["total_tokens"] == 15


def test_chat_validation(local_client, chat_ready):
    with pytest.raises(PageIndexAPIError, match="doc_id is required"):
        local_client.chat_completions(messages=[{"role": "user", "content": "q"}])
    with pytest.raises(PageIndexAPIError, match="cannot be empty"):
        local_client.chat_completions(messages=[], doc_id=chat_ready)
    with pytest.raises(PageIndexAPIError, match="First message"):
        local_client.chat_completions(
            messages=[{"role": "assistant", "content": "hi"}], doc_id=chat_ready)
    with pytest.raises(PageIndexAPIError, match="System messages"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"},
                      {"role": "system", "content": "s"}], doc_id=chat_ready)
    with pytest.raises(PageIndexAPIError, match="temperature"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}], doc_id=chat_ready,
            temperature=1.5)
    with pytest.raises(PageIndexAPIError, match="enable_citations"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}], doc_id=chat_ready,
            enable_citations=True)
    with pytest.raises(PageIndexAPIError, match="Document not found or access denied"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}], doc_id=[chat_ready, "nope"])
    with pytest.raises(PageIndexAPIError, match="cannot be empty"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}], doc_id=[])


# ── cloud mode: request wiring ──

class FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", content=b"{}",
                 lines=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text
        self.content = content
        self._lines = lines or []

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        pass


def _patch_requests(monkeypatch, handler):
    """Replace cloud_api's requests module with per-verb fakes."""
    fake = types.SimpleNamespace(
        post=lambda url, **kw: handler("POST", url, kw),
        get=lambda url, **kw: handler("GET", url, kw),
        delete=lambda url, **kw: handler("DELETE", url, kw),
        Response=FakeResponse,
    )
    monkeypatch.setattr("pageindex.cloud_api.requests", fake)


@pytest.fixture
def cloud(monkeypatch):
    client = PageIndexClient(api_key="secret")
    calls = []
    class Fake:
        payload = {}
    def handler(method, url, kw):
        calls.append({"method": method, "url": url, **kw})
        return FakeResponse(Fake.payload)
    _patch_requests(monkeypatch, handler)
    return client, calls, Fake


def test_cloud_request_wiring(cloud, sample_pdf):
    client, calls, fake = cloud

    fake.payload = {"doc_id": "pi-1"}
    assert client.submit_document(sample_pdf) == {"doc_id": "pi-1"}
    assert calls[-1]["url"] == "https://api.pageindex.ai/doc/"
    assert calls[-1]["headers"] == {"api_key": "secret"}
    assert calls[-1]["data"] == {"if_retrieval": True}
    assert "timeout" not in calls[-1]

    client.submit_document(sample_pdf, metadata={"project": "alpha"})
    assert calls[-1]["data"]["metadata"] == json.dumps({"project": "alpha"})

    fake.payload = {"status": "processing", "retrieval_ready": False}
    client.get_tree("pi-1", node_summary=True)
    assert calls[-1]["url"].endswith("/doc/pi-1/?type=tree&summary=True")
    assert calls[-1]["timeout"] == 30
    assert client.is_retrieval_ready("pi-1") is False

    client.get_ocr("pi/../1")
    assert "/doc/pi%2F..%2F1/" in calls[-1]["url"]

    client.BASE_URL = "https://staging.example"
    client.api_key = "other"
    client.get_document("pi-1")
    assert calls[-1]["url"] == "https://staging.example/doc/pi-1/metadata/"
    assert calls[-1]["headers"] == {"api_key": "other"}


def test_cloud_error_and_empty_delete(cloud, monkeypatch):
    client, calls, fake = cloud
    _patch_requests(monkeypatch,
                    lambda m, url, kw: FakeResponse(status_code=401, text="denied"))
    with pytest.raises(PageIndexAPIError,
                       match="Failed to get document metadata: denied"):
        client.get_document("pi-1")

    _patch_requests(monkeypatch, lambda m, url, kw: FakeResponse(content=b""))
    assert client.delete_document("pi-1") == {}


def test_cloud_chat_stream_parsing(cloud, monkeypatch):
    client, calls, fake = cloud
    lines = [
        b'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}',
        b'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        b"",
        b'data: {"object": "chat.completion.citations", "citations": []}',
        b'data: {"choices": [{"delta": {"content": " there"}}]}',
        b"data: [DONE]",
    ]
    _patch_requests(monkeypatch, lambda m, url, kw: FakeResponse(lines=lines))
    pieces = list(client.chat_completions(
        messages=[{"role": "user", "content": "q"}], stream=True))
    assert pieces == ["Hi", " there"]

    chunks = list(client.chat_completions(
        messages=[{"role": "user", "content": "q"}], stream=True,
        stream_metadata=True))
    assert {"object": "chat.completion.citations", "citations": []} in chunks
