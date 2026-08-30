import pytest

from pageindex.library import mcp_server
from pageindex.library.config import LibraryConfig


@pytest.fixture
def tools(home, store, sample_tree, sample_pages):
    sample_tree[0]["summary"] = "c1"
    sample_tree[0]["nodes"][0]["summary"] = "sa"
    sample_tree[0]["nodes"][0]["digest"] = "DIGEST A"
    meta = {"id": "pi-1", "name": "Vol I.pdf", "description": "Essays.", "status": "completed",
            "createdAt": "2026-08-30T00:00:00.000000", "pageNum": 6, "folderId": None,
            "metadata": {"title": "Complete Works I", "profile": "nonfiction"}, "mode": "flash"}
    store.save_document("pi-1", meta, sample_tree,
                        [{"page_index": i + 1, "markdown": t} for i, t in enumerate(sample_pages)])
    return mcp_server.build_tools(LibraryConfig.load())


def test_list_books(tools):
    rows = tools["list_books"]()
    assert rows == [{"doc_id": "pi-1", "name": "Vol I.pdf", "title": "Complete Works I",
                     "pages": 6, "profile": "nonfiction", "status": "completed",
                     "description": "Essays."}]


def test_get_structure_full_and_depth(tools):
    out = tools["get_structure"]("Complete Works")
    assert out["book"] == "Complete Works I" and out["doc_id"] == "pi-1"
    top = out["nodes"][0]
    assert top["node_id"] == "0000" and top["pages"] == "1-4" and top["summary"] == "c1"
    assert top["nodes"][0]["title"] == "Section A" and "digest" not in top["nodes"][0]
    shallow = tools["get_structure"]("Vol I", depth=1)
    assert "nodes" not in shallow["nodes"][0]


def test_get_pages_formats_and_limits(tools):
    text = tools["get_pages"]("Vol I", "2-3")
    assert text.startswith("--- Page 2 ---\nPage 2 text word2.")
    assert "--- Page 3 ---" in text and "word4" not in text
    with pytest.raises(ValueError, match="outside"):
        tools["get_pages"]("Vol I", "7")


def test_get_pages_caps_at_max(tools, monkeypatch):
    monkeypatch.setattr(mcp_server, "MAX_PAGES_PER_CALL", 2)
    with pytest.raises(ValueError, match="at most 2"):
        tools["get_pages"]("Vol I", "1-3")


def test_get_digest_node_and_book(tools):
    assert "DIGEST A" in tools["get_digest"]("Vol I", "0001")
    assert tools["get_digest"]("Vol I").startswith("# Complete Works I")


def test_unknown_book_message(tools):
    with pytest.raises(LookupError, match="No book"):
        tools["get_structure"]("nope")


def test_build_server_registers_tools():
    server = mcp_server.build_server(LibraryConfig.load())
    import asyncio
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"list_books", "get_structure", "get_pages", "get_digest"}
