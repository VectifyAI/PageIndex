"""SDK surface tests: PageIndexClient in local and cloud mode."""
import asyncio
import importlib
import json
import re
import shutil
import types

import pytest

import pageindex.flash
import pageindex.utils
from pageindex import PageIndexClient, PageIndexAPIError

page_index_module = importlib.import_module("pageindex.page_index_classic")


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
    def fake_page_index_main(doc, opt=None, logger=None, page_list=None):
        assert opt.if_add_node_summary == "yes"
        assert opt.if_add_node_text == "yes"
        assert logger is not None
        assert page_list is not None
        assert all(isinstance(t, tuple) and len(t) == 2 for t in page_list)
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
    with pytest.raises(PageIndexAPIError, match="model, storage_path"):
        PageIndexClient(api_key="k", model="m", storage_path="/tmp/x")


def test_local_client_does_not_touch_disk(tmp_path):
    storage = tmp_path / "store"
    PageIndexClient(storage_path=str(storage))
    assert not storage.exists()


def test_retrieve_model_carries_agents_sdk_prefix(tmp_path):
    def resolved(retrieve_model):
        return PageIndexClient(retrieve_model=retrieve_model,
                               storage_path=str(tmp_path / "s")).retrieve_model

    assert resolved("anthropic/claude-sonnet-4-6") == "litellm/anthropic/claude-sonnet-4-6"
    for already_routable in ("gpt-4o", "openai/gpt-4o", "litellm/anthropic/claude-sonnet-4-6"):
        assert resolved(already_routable) == already_routable


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
    assert child["text"] == "Second page about bananas"

    no_summary = local_client.get_tree(indexed_doc)["result"][0]
    assert "summary" not in no_summary and "prefix_summary" not in no_summary


def test_get_tree_include_text_false(local_client, indexed_doc):
    tree = local_client.get_tree(indexed_doc, include_text=False)
    root = tree["result"][0]
    assert "text" not in root
    assert "text" not in root["nodes"][0]
    assert root["page_index"] == 1

    with_text = local_client.get_tree(indexed_doc)["result"][0]
    assert "text" in with_text


def test_get_document_structure(local_client, indexed_doc):
    result = local_client.get_document_structure(indexed_doc)
    assert isinstance(result, list)
    root = result[0]
    assert "text" not in root
    assert "text" not in root["nodes"][0]
    assert "prefix_summary" in root
    assert root["nodes"][0]["summary"] == "child summary"


def test_get_page_content(local_client, indexed_doc):
    pages = local_client.get_page_content(indexed_doc, "1")
    assert len(pages) == 1
    assert pages[0]["page_index"] == 1
    assert "Hello page one" in pages[0]["markdown"]

    pages = local_client.get_page_content(indexed_doc, "1-2")
    assert len(pages) == 2

    pages = local_client.get_page_content(indexed_doc, "2,1")
    assert [p["page_index"] for p in pages] == [1, 2]

    assert local_client.get_page_content(indexed_doc, "99") == []

    with pytest.raises(ValueError):
        local_client.get_page_content(indexed_doc, "abc")


def test_submit_does_not_create_cwd_logs(local_client, sample_pdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def fake_page_index_main(doc, opt=None, logger=None, page_list=None):
        logger.info({"probe": True})
        return {"doc_name": "sample.pdf", "doc_description": None,
                "structure": json.loads(json.dumps(STRUCTURE))}
    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    local_client.submit_document(sample_pdf)
    assert not (tmp_path / "logs").exists()


def test_submit_duplicate_name_gets_suffix(local_client, sample_pdf, monkeypatch):
    """Mirror the cloud upload: a second submit of the same file name is
    stored as name_1, not as a same-name duplicate."""
    def fake_page_index_main(doc, opt=None, logger=None):
        return {"doc_name": "sample.pdf", "doc_description": "d",
                "structure": json.loads(json.dumps(STRUCTURE))}
    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    first = local_client.submit_document(sample_pdf)["doc_id"]
    second = local_client.submit_document(sample_pdf)["doc_id"]
    names = {d["id"]: d["name"]
             for d in local_client.list_documents()["documents"]}
    assert names[first] == "sample.pdf"
    assert names[second] == "sample_1.pdf"


def test_submit_duplicate_name_exhaustion(local_client, monkeypatch):
    api = local_client._api
    metas = ([{"name": "x.pdf"}]
             + [{"name": f"x_{num}.pdf"} for num in range(1, 100)])
    monkeypatch.setattr(api._store, "list_metas", lambda: metas)
    with pytest.raises(PageIndexAPIError, match="Too many files"):
        api._unique_doc_name("x.pdf")


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


def test_llm_completion_missing_key_raises_immediately(monkeypatch):
    import openai
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pageindex.utils, "_openai_sync_client", None)
    monkeypatch.setattr(pageindex.utils, "_openai_async_client", None)
    with pytest.raises(openai.OpenAIError):
        pageindex.utils.llm_completion("gpt-4o", "probe")
    with pytest.raises(openai.OpenAIError):
        asyncio.run(pageindex.utils.llm_acompletion("gpt-4o", "probe"))


def test_submit_missing_llm_key_fails_loud(local_client, sample_pdf, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pageindex.utils, "_openai_sync_client", None)
    def first_llm_call(*args, **kwargs):
        return pageindex.utils.llm_completion("gpt-4o", "probe")
    monkeypatch.setattr(page_index_module, "page_index_main", first_llm_call)
    monkeypatch.setattr(pageindex.flash, "page_index_flash", first_llm_call)
    for kwargs in ({}, {"mode": "flash"}):
        with pytest.raises(PageIndexAPIError, match="OPENAI_API_KEY"):
            local_client.submit_document(sample_pdf, **kwargs)
    assert local_client.list_documents()["total"] == 0


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


def test_corrupt_pdf_raises_api_error(local_client, tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage with no xref or trailer")
    with pytest.raises(PageIndexAPIError, match="could not read PDF"):
        local_client.submit_document(str(bad))


def test_encrypted_pdf_raises_api_error(local_client, sample_pdf, tmp_path):
    from PyPDF2 import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(sample_pdf).pages:
        writer.add_page(page)
    writer.encrypt("secret")
    enc = tmp_path / "enc.pdf"
    with open(enc, "wb") as f:
        writer.write(f)
    with pytest.raises(PageIndexAPIError, match="could not read PDF"):
        local_client.submit_document(str(enc))


def test_submit_explicit_standard_mode(local_client, sample_pdf, monkeypatch):
    calls = []

    def fake_page_index_main(doc, opt=None, logger=None, page_list=None):
        calls.append(doc)
        return {
            "doc_name": "sample.pdf",
            "doc_description": "A test document.",
            "structure": json.loads(json.dumps(STRUCTURE)),
        }

    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    doc_id = local_client.submit_document(sample_pdf, mode="standard")["doc_id"]

    assert calls == [sample_pdf]
    assert local_client._api._store.get_meta(doc_id)["mode"] == "standard"


@pytest.mark.parametrize("mode", ["standard", "flash"])
def test_submit_from_running_event_loop(
    local_client, sample_pdf, monkeypatch, mode
):
    def fake_index(*args):
        asyncio.run(asyncio.sleep(0))
        return json.loads(json.dumps(STRUCTURE)), "A test document."

    monkeypatch.setattr(local_client._api, f"_index_{mode}", fake_index)

    async def submit():
        return local_client.submit_document(sample_pdf, mode=mode)

    doc_id = asyncio.run(submit())["doc_id"]
    assert local_client.get_document(doc_id)["status"] == "completed"


def test_submit_with_metadata(local_client, sample_pdf, monkeypatch):
    monkeypatch.setattr(
        page_index_module, "page_index_main",
        lambda doc, opt=None, logger=None, page_list=None: {
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
        {"title": "Root Section", "level": 1, "page_index": 1,
         "text": "Hello page one about applesSecond page about bananas"},
        {"title": "Child Section", "level": 2, "page_index": 2,
         "text": "Second page about bananas"},
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


@pytest.mark.parametrize("bad_entry", ["corrupt-entry", {"id": "wrong"}])
def test_manifest_invalid_entry_self_heals(
    local_client, indexed_doc, tmp_path, bad_entry
):
    manifest_path = tmp_path / "store" / "manifest.json"
    manifest_path.write_text(json.dumps({"docs": {indexed_doc: bad_entry}}))

    listing = local_client.list_documents()

    assert listing["total"] == 1
    assert listing["documents"][0]["id"] == indexed_doc
    healed = json.loads(manifest_path.read_text())
    assert healed["docs"][indexed_doc]["name"] == "sample.pdf"


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


def test_get_tree_fails_loud_on_broken_pages(local_client, indexed_doc, tmp_path):
    doc_dir = tmp_path / "store" / "docs" / indexed_doc
    (doc_dir / "pages.json").write_text("{broken")
    with pytest.raises(PageIndexAPIError, match="unreadable"):
        local_client.get_tree(indexed_doc)


def test_get_tree_fails_loud_on_empty_pages(local_client, indexed_doc, tmp_path):
    doc_dir = tmp_path / "store" / "docs" / indexed_doc
    (doc_dir / "pages.json").write_text("[]")
    with pytest.raises(PageIndexAPIError, match="no page content"):
        local_client.get_tree(indexed_doc)


def test_data_file_as_directory_fails_loud(local_client, indexed_doc, tmp_path):
    tree_path = tmp_path / "store" / "docs" / indexed_doc / "tree.json"
    tree_path.unlink()
    tree_path.mkdir()
    with pytest.raises(PageIndexAPIError, match="unreadable"):
        local_client.get_tree(indexed_doc)


def test_list_documents_skips_unsafe_directory_names(
    local_client, indexed_doc, tmp_path
):
    bad_dir = tmp_path / "store" / "docs" / "bad\\name"
    bad_dir.mkdir()
    (bad_dir / "doc.json").write_text("{}")
    listing = local_client.list_documents()
    assert [d["id"] for d in listing["documents"]] == [indexed_doc]


def test_generate_doc_description_error_boundary(monkeypatch):
    def raiser(exc):
        def _f(*args, **kwargs):
            raise exc
        return _f
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        raiser(RuntimeError("retries exhausted")))
    assert pageindex.utils.generate_doc_description([]) == ""
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        raiser(ValueError("provider rejected the model")))
    with pytest.raises(ValueError):
        pageindex.utils.generate_doc_description([])


def test_generate_summaries_all_failed_raises(monkeypatch):
    async def boom(model, prompt):
        raise ValueError("bad key")
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", boom)
    structure = [{"title": "A", "text": "t1",
                  "nodes": [{"title": "B", "text": "t2"}]}]
    with pytest.raises(RuntimeError, match="all nodes"):
        asyncio.run(pageindex.utils.generate_summaries_for_structure(structure))


def test_generate_summaries_partial_failure_absorbed(monkeypatch):
    async def flaky(model, prompt):
        if "t1" in prompt:
            raise ValueError("transient")
        return "ok"
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", flaky)
    structure = [{"title": "A", "text": "t1",
                  "nodes": [{"title": "B", "text": "t2"}]}]
    result = asyncio.run(pageindex.utils.generate_summaries_for_structure(structure))
    summaries = {n["title"]: n["summary"]
                 for n in pageindex.utils.structure_to_list(result)}
    assert summaries == {"A": "", "B": "ok"}


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


def test_chat_completions_cloud_only(local_client):
    with pytest.raises(PageIndexAPIError, match="not yet supported in local mode"):
        local_client.chat_completions(
            messages=[{"role": "user", "content": "q"}])


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
    assert calls[-1]["url"].endswith("/doc/pi-1/?type=tree&summary=True&include_text=true")
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
