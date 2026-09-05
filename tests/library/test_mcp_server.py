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


def test_list_digests_excludes_book_folders_and_reads_the_first_heading(tools, home):
    digests = home / "digests"
    book_folder = digests / "complete-works-i"
    book_folder.mkdir(parents=True)
    (book_folder / "book.md").write_text("# Complete Works I\n\nbook digest", encoding="utf-8")
    topic_folder = digests / "the-goal-and-the-perfect-state"
    topic_folder.mkdir(parents=True)
    (topic_folder / "synthesis.md").write_text(
        "# The Goal, in Babuji's own words\n\nbody", encoding="utf-8")
    rows = tools["list_digests"]()
    assert rows == [{"topic": "the-goal-and-the-perfect-state",
                     "title": "The Goal, in Babuji's own words",
                     "path": str(topic_folder / "synthesis.md")}]


def test_list_digests_empty_when_no_digests_dir(tools):
    assert tools["list_digests"]() == []


def test_build_server_registers_tools():
    server = mcp_server.build_server(LibraryConfig.load())
    import asyncio
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"list_books", "get_structure", "get_pages", "get_digest", "list_digests"}


def test_mcp_tool_error_carries_the_original_message_through_the_real_transport(home, store):
    """Regression: the raw functions raise LookupError/ValueError/KeyError, but
    the installed mcp package strips any exception that isn't its own ToolError
    down to a generic "Error executing tool <name>", discarding the message
    find_book worked to produce. This must go through MCPServer.call_tool (the
    real client-facing path), not the plain function object, or it would not
    catch the regression."""
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    server = mcp_server.build_server(LibraryConfig.load())
    with pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool("get_structure", {"book": "nonexistent"}))
    assert not isinstance(exc.value, UnexpectedToolError)
    assert "No book matches 'nonexistent'" in str(exc.value)
