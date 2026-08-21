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
    return local_client.submit_document(sample_pdf, mode="standard")["doc_id"]


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


def test_retrieve_model_stays_as_configured(tmp_path):
    """The public attribute keeps the caller's spelling — ``litellm/`` is
    Agents SDK routing grammar, applied at the config door
    (openai_agent_config), never baked into ``chat_model``: handed to the
    Anthropic SDK or a raw request, the prefixed form is a 404."""
    def resolved(retrieve_model):
        return PageIndexClient(retrieve_model=retrieve_model,
                               storage_path=str(tmp_path / "s")).retrieve_model

    for as_configured in ("anthropic/claude-sonnet-4-6", "gpt-4o",
                          "openai/gpt-4o",
                          "litellm/anthropic/claude-sonnet-4-6"):
        assert resolved(as_configured) == as_configured


def test_model_resolution_covers_every_generation(tmp_path):
    """New names win over old, specific over general, ``model`` sets every
    role, and the built-in defaults close each chain. One row per released
    surface: 0.2.8 (model only), 0.3.0.dev (model + retrieve_model),
    0.2.10.dev (all three legacy names), the current pair, plus the
    umbrella and mixed forms."""
    from pageindex.utils import DEFAULT_CHAT_MODEL, DEFAULT_INDEX_MODEL
    cases = [
        ({}, DEFAULT_INDEX_MODEL, DEFAULT_INDEX_MODEL, DEFAULT_CHAT_MODEL),
        ({"model": "m"}, "m", "m", "m"),
        ({"model": "m", "retrieve_model": "r"}, "m", "m", "r"),
        ({"model": "m", "summary_model": "s", "retrieve_model": "r"},
         "m", "s", "r"),
        ({"index_model": "i", "chat_model": "c"}, "i", "i", "c"),
        ({"model": "m", "index_model": "i"}, "i", "i", "m"),
        ({"summary_model": "s"},
         DEFAULT_INDEX_MODEL, "s", DEFAULT_CHAT_MODEL),
    ]
    for kwargs, index, summary, chat in cases:
        client = PageIndexClient(storage_path=str(tmp_path / "s"), **kwargs)
        assert (client.index_model, client.model) == (index, index), kwargs
        assert client.summary_model == summary, kwargs
        assert client.chat_model == chat, kwargs
        assert client.retrieve_model == client.chat_model, kwargs


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

    with pytest.raises(PageIndexAPIError):
        local_client.get_page_content(indexed_doc, "abc")


def test_get_page_content_span_bomb_rejected(local_client, indexed_doc):
    """An absurd range must be rejected arithmetically, not expanded into
    a billion integers in the caller's process (the tool layer already
    refused; the public client method did not)."""
    with pytest.raises(PageIndexAPIError, match="spans more than 10000"):
        local_client.get_page_content(indexed_doc, "1-1000001")
    # At the bound itself the spec still parses.
    assert local_client.get_page_content(indexed_doc, "5-10004") == []


def test_submit_does_not_create_cwd_logs(local_client, sample_pdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def fake_page_index_main(doc, opt=None, logger=None, page_list=None):
        logger.info({"probe": True})
        return {"doc_name": "sample.pdf", "doc_description": None,
                "structure": json.loads(json.dumps(STRUCTURE))}
    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    local_client.submit_document(sample_pdf, mode="standard")
    assert not (tmp_path / "logs").exists()


def test_submit_duplicate_name_gets_suffix(local_client, sample_pdf, monkeypatch):
    """Mirror the cloud upload: a second submit of the same file name is
    stored as name_1, not as a same-name duplicate."""
    def fake_page_index_main(doc, opt=None, logger=None, page_list=None):
        return {"doc_name": "sample.pdf", "doc_description": "d",
                "structure": json.loads(json.dumps(STRUCTURE))}
    monkeypatch.setattr(page_index_module, "page_index_main", fake_page_index_main)
    first = local_client.submit_document(sample_pdf, mode="standard")
    assert first["name"] == "sample.pdf"
    with pytest.warns(UserWarning, match='stored as "sample_1.pdf"'):
        second = local_client.submit_document(sample_pdf, mode="standard")
    assert second["name"] == "sample_1.pdf"
    names = {d["id"]: d["name"]
             for d in local_client.list_documents()["documents"]}
    assert names[first["doc_id"]] == "sample.pdf"
    assert names[second["doc_id"]] == "sample_1.pdf"


def test_submit_duplicate_name_exhaustion(local_client, monkeypatch):
    api = local_client._api
    metas = ([{"name": "x.pdf"}]
             + [{"name": f"x_{num}.pdf"} for num in range(1, 100)])
    monkeypatch.setattr(api._store, "list_metas", lambda: metas)
    with pytest.raises(PageIndexAPIError, match="Too many files"):
        api._unique_doc_name("x.pdf")


def test_submit_name_exhaustion_rejects_before_indexing(
    local_client, sample_pdf, monkeypatch,
):
    api = local_client._api
    metas = ([{"name": "sample.pdf"}]
             + [{"name": f"sample_{num}.pdf"} for num in range(1, 100)])
    monkeypatch.setattr(api._store, "list_metas", lambda: metas)
    monkeypatch.setattr(
        page_index_module, "page_index_main",
        lambda *args, **kwargs: pytest.fail(
            "indexer ran despite name exhaustion"),
    )
    with pytest.raises(PageIndexAPIError, match="Too many files"):
        local_client.submit_document(sample_pdf, mode="standard")


def test_submit_flash(local_client, sample_pdf, monkeypatch):
    calls = {}
    def fake_flash(pdf, summary=True, summary_model=None, **kwargs):
        calls["summary"] = summary
        calls["summary_model"] = summary_model
        calls["optimize"] = kwargs.get("optimize")
        calls["optimize_model"] = kwargs.get("optimize_model")
        return {"doc_name": "sample.pdf",
                "structure": [{"title": "Flash Root", "start_index": 1,
                               "end_index": 2, "summary": "s", "nodes": []}]}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", fake_flash)
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda model, prompt, **kw: "Flash description.")
    doc_id = local_client.submit_document(sample_pdf, mode="flash")["doc_id"]
    assert calls == {"summary": True, "summary_model": local_client.summary_model,
                     "optimize": "full",
                     "optimize_model": local_client.summary_model}
    root = local_client.get_tree(doc_id)["result"][0]
    assert root["node_id"] == "0000"
    assert "Hello page one" in root["text"]
    assert local_client.get_document(doc_id)["description"] == "Flash description."


def test_submit_defaults_to_flash(local_client, sample_pdf, monkeypatch):
    monkeypatch.setattr(
        pageindex.flash, "page_index_flash",
        lambda pdf, **kwargs: {
            "doc_name": "sample.pdf",
            "structure": [{"title": "Flash Root", "start_index": 1,
                           "end_index": 2, "summary": "s", "nodes": []}]})
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda model, prompt, **kw: "Flash description.")
    doc_id = local_client.submit_document(sample_pdf)["doc_id"]
    assert local_client._api._store.get_meta(doc_id)["mode"] == "flash"


def test_submit_rejects_structure_beyond_stored_pages(local_client, sample_pdf,
                                                      monkeypatch):
    """A tree spanning pages the store lacks fails submit instead of saving a doc whose reads IndexError."""
    monkeypatch.setattr(
        pageindex.flash, "page_index_flash",
        lambda pdf, **kwargs: {
            "doc_name": "sample.pdf",
            "structure": [{"title": "Root", "start_index": 1,
                           "end_index": 3, "summary": "s", "nodes": []}]})
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda model, prompt, **kw: "d.")
    with pytest.raises(PageIndexAPIError, match="pages 1-3 outside"):
        local_client.submit_document(sample_pdf)
    assert local_client._api._store.list_metas() == []


def test_submit_survives_pypdf2_lone_surrogates(local_client, sample_pdf,
                                                monkeypatch):
    """PyPDF2 decodes broken ToUnicode with surrogatepass; the store gets U+FFFD, not a utf-8-fatal str."""
    import PyPDF2
    monkeypatch.setattr(PyPDF2.PageObject, "extract_text",
                        lambda self: "\ud83dello broken")
    monkeypatch.setattr(
        pageindex.flash, "page_index_flash",
        lambda p, **kwargs: {
            "structure": [{"title": "T", "start_index": 1,
                           "end_index": 1, "summary": "s", "nodes": []}]})
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda model, prompt, **kw: "d.")
    doc_id = local_client.submit_document(sample_pdf)["doc_id"]
    markdown = local_client.get_ocr(doc_id)["result"][0]["markdown"]
    assert "\ud83d" not in markdown
    assert markdown.startswith("�ello")


def test_page_index_flash_rejects_unknown_optimize():
    from pageindex.flash import page_index_flash
    with pytest.raises(ValueError, match="optimize must be"):
        page_index_flash("never-opened.pdf", optimize="off")


def test_llm_completion_refuses_unknown_provider(monkeypatch):
    """A first segment LiteLLM does not know (a HuggingFace repo id like
    Qwen/...) is refused with the openai/ escape before the retry loop,
    instead of burning it on per-call 400s."""
    import litellm
    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: pytest.fail("reached the wire"))
    with pytest.raises(Exception, match="not a LiteLLM provider"):
        pageindex.utils.llm_completion("Qwen/my-model", "probe")


def test_submit_missing_llm_key_fails_loud(local_client, sample_pdf, monkeypatch):
    import litellm  # first import may load a .env; delenv after it
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("pageindex.utils.time.sleep", lambda s: None)
    def first_llm_call(*args, **kwargs):
        return pageindex.utils.llm_completion("gpt-4o", "probe")
    monkeypatch.setattr(page_index_module, "page_index_main", first_llm_call)
    monkeypatch.setattr(pageindex.flash, "page_index_flash", first_llm_call)
    for kwargs in ({"mode": "standard"}, {"mode": "flash"}):
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


def test_write_json_atomic_replaces_lone_surrogates(tmp_path):
    """A lone surrogate (an os.fsdecode'd path in metadata, an LLM-written
    \\ud83d escape) must not crash the store's writer after a whole
    indexing run — it lands as the encoder's replacement character."""
    from pageindex.local_store import _read_json, _write_json_atomic

    path = tmp_path / "doc.json"
    _write_json_atomic(path, {"src": "bad-\udcff-path"})
    assert _read_json(path)["src"] == "bad-?-path"


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
    doc_id = local_client.submit_document(sample_pdf, mode="standard", metadata=tags)["doc_id"]
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
    with pytest.warns(UserWarning):  # same-name resubmit → stored as sample_1.pdf
        second = local_client.submit_document(sample_pdf, mode="standard")["doc_id"]
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


def test_generate_doc_description_propagates_failures(monkeypatch):
    """No swallow: a dead model fails the run instead of storing ''."""
    def raiser(exc):
        def _f(*args, **kwargs):
            raise exc
        return _f
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        raiser(RuntimeError("retries exhausted")))
    with pytest.raises(RuntimeError):
        pageindex.utils.generate_doc_description([])
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


def test_generate_summaries_unrecoverable_raises(monkeypatch):
    """A per-node 401 must abort, not store a blank node as completed."""
    class Denied(Exception):
        status_code = 401

    async def deny_t1(model, prompt):
        if "t1" in prompt:
            raise Denied("key rejected")
        return "ok"
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", deny_t1)
    structure = [{"title": "A", "text": "t1",
                  "nodes": [{"title": "B", "text": "t2"}]}]
    with pytest.raises(Denied):
        asyncio.run(pageindex.utils.generate_summaries_for_structure(structure))


def test_summarize_tree_child_unrecoverable_raises(monkeypatch):
    """A 401 on a leaf must abort the run, not store a blank subtree as
    completed: the child gather's exceptions are checked, not discarded."""
    class Denied(Exception):
        status_code = 401

    async def deny_alpha(model, prompt):
        if "alpha" in prompt:
            raise Denied("key rejected")
        return '{"points": [], "summary": "ok"}'
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", deny_alpha)
    pdf_pages = [("alpha " * 5, 5), ("beta " * 5, 5)]
    structure = [{"title": "R", "start_index": 1, "end_index": 2,
                  "nodes": [
                      {"title": "A", "start_index": 1, "end_index": 1},
                      {"title": "B", "start_index": 2, "end_index": 2}]}]
    with pytest.raises(Denied):
        asyncio.run(pageindex.utils.summarize_tree(
            structure, pdf_pages, small_node_tokens=0))


def test_summarize_tree_fails_loud_when_every_model_call_fails(monkeypatch):
    """A failure foreign to the retry ladder (not LLMRetriesExhausted)
    blanks per node; the asked-and-never-answered backstop must still fail
    the run loud — a raw-text short leaf cannot vouch for it."""
    async def exhausted(model, prompt):
        raise RuntimeError("LLM call failed after 10 attempts")
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", exhausted)
    pdf_pages = [("tiny", 1), ("beta " * 300, 300)]
    structure = [{"title": "R", "start_index": 1, "end_index": 2,
                  "nodes": [
                      {"title": "A", "start_index": 1, "end_index": 1},
                      {"title": "B", "start_index": 2, "end_index": 2}]}]
    with pytest.raises(RuntimeError, match="every summary call failed"):
        asyncio.run(pageindex.utils.summarize_tree(structure, pdf_pages))


def test_summarize_tree_all_short_leaves_need_no_model(monkeypatch):
    """A tree whose every node summarizes from raw text makes zero model
    calls and must not be mistaken for a failed run."""
    async def unexpected(model, prompt):
        raise AssertionError("no model call expected")
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", unexpected)
    structure = [{"title": "A", "start_index": 1, "end_index": 1}]
    out = asyncio.run(pageindex.utils.summarize_tree(structure, [("tiny", 1)]))
    assert out[0]["summary"] == "tiny"


def test_summarize_tree_partial_exhaustion_fails_loud(monkeypatch):
    """One lucky call must not vouch for a model that then went away: a
    ladder-exhausted node raises instead of silently blanking."""
    calls = {"n": 0}

    async def flaky(model, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"points": ["p"], "summary": "ok"}'
        raise pageindex.utils.LLMRetriesExhausted(
            "LLM completion failed after 10 retries", status_code=500)
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", flaky)
    pdf_pages = [("alpha " * 300, 300), ("beta " * 300, 300)]
    structure = [{"title": "A", "start_index": 1, "end_index": 1},
                 {"title": "B", "start_index": 2, "end_index": 2}]
    with pytest.raises(pageindex.utils.LLMRetriesExhausted):
        asyncio.run(pageindex.utils.summarize_tree(structure, pdf_pages))


def test_generate_summaries_partial_exhaustion_fails_loud(monkeypatch):
    calls = {"n": 0}

    async def flaky(model, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "fine"
        raise pageindex.utils.LLMRetriesExhausted(
            "LLM completion failed after 10 retries", status_code=500)
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", flaky)
    structure = [{"title": "A", "text": "t1",
                  "nodes": [{"title": "B", "text": "t2"}]}]
    with pytest.raises(pageindex.utils.LLMRetriesExhausted):
        asyncio.run(
            pageindex.utils.generate_summaries_for_structure(structure))


def test_expand_exhausted_ladder_fails_loud(monkeypatch):
    """Keyless/broken-model expand must kill the run, not degrade to
    no_children after burning the retry ladder on every node."""
    import pageindex.tree_optimize as tree_optimize

    async def exhausted(model, prompt):
        raise pageindex.utils.LLMRetriesExhausted(
            "LLM completion failed after 10 retries", status_code=500)
    monkeypatch.setattr(tree_optimize, "llm_acompletion", exhausted)
    structure = [{"title": "T", "start_index": 1, "end_index": 8,
                  "node_id": "0001", "nodes": []}]
    pages = ["heading\nbody text"] * 8
    lines = [["heading", "body text"]] * 8
    with pytest.raises(pageindex.utils.LLMRetriesExhausted):
        asyncio.run(tree_optimize.optimize(structure, pages, lines,
                                           model="m", do_expand=True))


def test_expand_absorbs_per_prompt_rejection(monkeypatch):
    """A 400-exhausted node (context_length_exceeded) stays collapsed and
    the run survives — the documented per-prompt absorption."""
    import pageindex.tree_optimize as tree_optimize

    async def rejected(model, prompt):
        raise pageindex.utils.LLMRetriesExhausted(
            "LLM completion failed after 10 retries", status_code=400)
    monkeypatch.setattr(tree_optimize, "llm_acompletion", rejected)
    structure = [{"title": "T", "start_index": 1, "end_index": 8,
                  "node_id": "0001", "nodes": []}]
    pages = ["heading\nbody text"] * 8
    lines = [["heading", "body text"]] * 8
    outcome = asyncio.run(tree_optimize.optimize(structure, pages, lines,
                                                 model="m", do_expand=True))
    assert outcome["expands"] == 0


def test_llm_completion_backend_reaches_litellm(monkeypatch):
    """No cache params of our own; backend keys reach litellm and win the merge."""
    import litellm
    captured = {}

    def fake_completion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        message = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert pageindex.utils.llm_completion("gpt-4o", "probe") == "ok"
    assert "cache_control_injection_points" not in captured
    token = pageindex.utils._llm_backend.set(
        {"api_key": "x", "max_retries": 3})
    try:
        pageindex.utils.llm_completion("gpt-4o", "probe")
    finally:
        pageindex.utils._llm_backend.reset(token)
    assert captured["api_key"] == "x"
    assert captured["max_retries"] == 3


def test_backend_overrides_reserved_kwargs_without_retry(monkeypatch):
    """A backend key colliding with our own kwargs wins the merge instead
    of raising TypeError through the retry ladder."""
    import litellm
    calls = {"n": 0}
    captured = {}

    def fake_completion(**kwargs):
        calls["n"] += 1
        captured.clear()
        captured.update(kwargs)
        message = types.SimpleNamespace(content="ok")
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        return types.SimpleNamespace(choices=[choice])
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(pageindex.utils.time, "sleep", lambda s: None)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    token = pageindex.utils._llm_backend.set({"drop_params": False})
    try:
        assert pageindex.utils.llm_completion("gpt-4o", "probe") == "ok"
    finally:
        pageindex.utils._llm_backend.reset(token)
    assert captured["drop_params"] is False
    assert calls["n"] == 1


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


def test_chat_completions_local_needs_openai_agents(local_client, monkeypatch):
    """Local chat is implemented (see test_local_chat.py); without
    openai-agents installed it raises the actionable install error."""
    import sys
    monkeypatch.setitem(sys.modules, "agents", None)
    with pytest.raises(PageIndexAPIError, match="pip install openai-agents"):
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


def test_cloud_chat_accepts_query_string(cloud):
    client, calls, fake = cloud
    fake.payload = {"choices": [{"message": {"content": "ok"}}]}
    client.chat_completions("What status?")
    assert calls[-1]["json"]["messages"] == [
        {"role": "user", "content": "What status?"}]
    with pytest.raises(PageIndexAPIError, match="non-empty string"):
        client.chat_completions("   ")


def test_parse_pages_overlap_counts_union():
    from pageindex.client import _parse_pages
    pages = _parse_pages("1-5000,2000-9000")
    assert len(pages) == 9000 and pages[0] == 1 and pages[-1] == 9000
    with pytest.raises(PageIndexAPIError, match="spans more than"):
        _parse_pages("1-10001")
    # one parser with the tool layer now: page 0 is rejected, not passed
    # on — surfaced as the documented SDK error type
    with pytest.raises(PageIndexAPIError, match="positive"):
        _parse_pages("0-3")


# ── backend: the indexing lane ──

def test_backend_scopes_the_index_lane(tmp_path, monkeypatch):
    """index_backend reaches the indexing lane's LiteLLM call kwargs
    verbatim — bare and provider-prefixed models alike — scoped to the
    operation, bypassing the env pre-check."""
    pytest.importorskip("litellm")
    import litellm
    from types import SimpleNamespace
    from pageindex.local_api import LocalAPI
    from pageindex.utils import _llm_backend, llm_completion

    api = LocalAPI(storage_path=str(tmp_path / "s"), model="m",
                   summary_model="s",
                   index_backend={"api_key": "ik", "api_base": "http://b"})
    assert api._with_backend(_llm_backend.get) == {"api_key": "ik",
                                                   "api_base": "http://b"}
    assert _llm_backend.get() is None

    reply = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="ok"), finish_reason="stop")])
    captured = {}
    monkeypatch.setattr(litellm, "completion",
                        lambda **kw: (captured.update(kw), reply)[1])
    monkeypatch.setattr(litellm, "validate_environment",
                        lambda *a, **k: pytest.fail("env pre-check ran"))
    for model, wire in (("anthropic/claude-x", "anthropic/claude-x"),
                        ("gpt-4o", "openai/gpt-4o"),
                        ("my-finetune-v2", "openai/my-finetune-v2")):
        captured.clear()
        api._with_backend(lambda: llm_completion(model, "p"))
        assert captured["model"] == wire
        assert captured["api_key"] == "ik"
        assert captured["api_base"] == "http://b"


def test_index_lane_makes_no_key_prejudgment(monkeypatch):
    """Credentials are LiteLLM's call at completion time: keyless
    environments reach the wire untouched for every provider shape."""
    pytest.importorskip("litellm")
    import litellm
    from types import SimpleNamespace
    from pageindex.utils import llm_completion

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reply = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="ok"), finish_reason="stop")])
    monkeypatch.setattr(litellm, "completion", lambda **kw: reply)
    monkeypatch.setattr(litellm, "validate_environment",
                        lambda *a, **k: pytest.fail("env pre-check ran"))
    assert llm_completion("my-finetune-v2", "p") == "ok"
    assert llm_completion("ollama/llama3", "p") == "ok"
    assert llm_completion("bedrock/anthropic.claude-sonnet", "p") == "ok"


def test_llm_completion_surfaces_litellms_credential_verdict(monkeypatch):
    """LiteLLM's own missing-credentials error (a retryable-shaped 500)
    rides the retry loop and lands verbatim in the terminal error."""
    pytest.importorskip("litellm")
    import litellm  # noqa: F401 — first import may load a .env; delenv after
    from pageindex.utils import llm_completion

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("pageindex.utils.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_completion("gpt-4o", "probe")

    async def _nosleep(s):
        pass

    monkeypatch.setattr("pageindex.utils.asyncio.sleep", _nosleep)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(pageindex.utils.llm_acompletion("gpt-4o", "probe"))


def test_litellm_model_normalizes_without_key_prejudgment(monkeypatch):
    """_litellm_model only normalizes and provider-checks — a keyless
    environment changes nothing for any spelling."""
    pytest.importorskip("litellm")
    import litellm  # noqa: F401 — first import may load a .env; delenv after
    from pageindex.utils import _litellm_model

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _litellm_model("litellm/gpt-4o", None) == "openai/gpt-4o"
    assert _litellm_model("gpt-4o", None) == "openai/gpt-4o"


def test_custom_provider_map_passes_provider_precheck(monkeypatch):
    """litellm appends custom_provider_map providers to provider_list only
    at completion time, so the pre-check must consult the map itself."""
    pytest.importorskip("litellm")
    import litellm
    from pageindex.utils import _litellm_model

    monkeypatch.setattr(litellm, "custom_provider_map",
                        [{"provider": "my-llm", "custom_handler": object()}])
    assert _litellm_model("my-llm/model-a", None) == "my-llm/model-a"


def test_backend_args_are_local_only():
    with pytest.raises(PageIndexAPIError, match="chat_backend"):
        PageIndexClient(api_key="pi-k", chat_backend={"api_key": "x"})
    with pytest.raises(PageIndexAPIError, match="index_backend"):
        PageIndexClient(api_key="pi-k", index_backend={"api_key": "x"})


def test_chat_wraps_answerless_cloud_reply(monkeypatch):
    """A cloud reply without choices (filtered / malformed) surfaces as
    the SDK's error, not a bare IndexError/KeyError."""
    client = PageIndexClient(api_key="pi-k")
    for reply in ({"id": "x", "object": "chat.completion", "choices": []},
                  {"id": "x"}):
        monkeypatch.setattr(client, "chat_completions",
                            lambda *a, _r=reply, **k: _r)
        with pytest.raises(PageIndexAPIError, match="carries no answer"):
            client.chat("hi")


def test_retrieve_model_assignment_still_works(local_client):
    """0.2.9 allowed `client.retrieve_model = ...`; the legacy property
    keeps the write path as an alias for chat_model."""
    local_client.retrieve_model = "gpt-x"
    assert local_client.chat_model == "gpt-x"
    assert local_client.retrieve_model == "gpt-x"


def test_concurrent_same_name_submits_store_unique_names(local_client,
                                                         sample_pdf,
                                                         monkeypatch):
    """Name uniquing runs under the store lock at save time, so two
    clients indexing the same filename concurrently cannot both store it
    — a stored duplicate would shadow the older doc_id forever."""
    import threading
    import time as time_mod

    def slow_flash(pdf, **kwargs):
        time_mod.sleep(0.1)  # both threads index before either saves
        return {"structure": [{"title": "T", "start_index": 1,
                               "end_index": 2, "summary": "s", "nodes": []}]}

    monkeypatch.setattr(pageindex.flash, "page_index_flash", slow_flash)
    monkeypatch.setattr(pageindex.utils, "llm_completion",
                        lambda *a, **k: "d")
    results = []
    workers = [threading.Thread(
        target=lambda: results.append(local_client.submit_document(sample_pdf)))
        for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert {r["name"] for r in results} == {"sample.pdf", "sample_1.pdf"}



def test_format_tree_node_keeps_key_items():
    """key_items from the merge optimization survive the get_tree formatter."""
    from pageindex.local_api import _format_tree_node

    node = {"title": "Chapter 1", "node_id": "0000", "start_index": 1,
            "summary": "s",
            "key_items": ["1.1 Alpha", "1.2 Beta", "1.3 Gamma"]}
    out = _format_tree_node(node, node_summary=True)
    assert out["key_items"] == ["1.1 Alpha", "1.2 Beta", "1.3 Gamma"]
    assert "key_items" not in _format_tree_node(
        {"title": "t", "node_id": "0001", "start_index": 1}, False)
