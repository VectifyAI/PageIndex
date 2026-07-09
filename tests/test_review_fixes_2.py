"""Regression tests for the second review pass (xhigh code-review of
2d46d68..8f536cb): the Markdown text-stripping fix's fallout, plus the other
directly-fixable findings from that pass."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from pageindex.config import IndexConfig


def _md_backend(tmp_path):
    from pageindex.backend.local import LocalBackend
    from pageindex.storage.sqlite import SQLiteStorage

    backend = LocalBackend(
        storage=SQLiteStorage(str(tmp_path / "t.db")),
        files_dir=str(tmp_path / "f"), model="gpt-4o",
        index_config=IndexConfig(if_add_node_summary=False, if_add_doc_description=False),
    )
    backend.get_or_create_collection("c")
    return backend


def _write_md(tmp_path, name="doc.md"):
    path = tmp_path / name
    path.write_text("# Title\nfirst section body\n\n## Sub\nsecond section body\n")
    return str(path)


# ── #1: get_document(include_text=True) must fill text for Markdown nodes ────
def test_get_document_include_text_fills_markdown_nodes(tmp_path):
    backend = _md_backend(tmp_path)
    doc_id = backend.add_document("c", _write_md(tmp_path))

    def _texts(nodes):
        for n in nodes:
            yield n.get("text")
            if n.get("nodes"):
                yield from _texts(n["nodes"])

    without = backend.get_document("c", doc_id, include_text=False)
    assert not any(_texts(without["structure"]))

    with_text = backend.get_document("c", doc_id, include_text=True)
    texts = list(_texts(with_text["structure"]))
    assert texts, "expected at least one node"
    assert any(t for t in texts), "Markdown nodes must get real text, not all empty"
    assert any("first section body" in t or "second section body" in t for t in texts if t)


# ── #2: get_page_content's Markdown fallback re-derives from the source file ──
def test_get_page_content_markdown_fallback_reads_from_file(tmp_path):
    backend = _md_backend(tmp_path)
    md_path = _write_md(tmp_path)
    doc_id = backend.add_document("c", md_path)

    # Simulate a StorageEngine that doesn't cache pages (protocol explicitly
    # allows get_pages() to return None) by clearing the cached pages column.
    conn = backend._storage._get_conn()
    conn.execute("UPDATE documents SET pages = NULL WHERE doc_id = ?", (doc_id,))

    result = backend.get_page_content("c", doc_id, "1")
    assert result and result[0]["content"], "fallback must return real text, not empty"
    assert "first section body" in result[0]["content"]


# ── #3: keyless provider allowlist covers other local LiteLLM providers ──────
@pytest.mark.parametrize("model", [
    "ollama/llama3", "lm_studio/x", "xinference/llama2", "llamafile/x",
    "triton/x", "oobabooga/x", "openai_like/x", "docker_model_runner/x",
])
def test_validate_llm_provider_accepts_more_keyless_providers(model):
    from pageindex.client import LocalClient
    LocalClient._validate_llm_provider(model)  # must not raise


# ── #4: agent-tool closures consistently raise/error on a missing doc ────────
def test_agent_tools_consistently_report_missing_doc(tmp_path):
    import json
    import asyncio as _asyncio
    from agents.tool_context import ToolContext

    backend = _md_backend(tmp_path)
    # Open-mode tools (doc_ids=None) so we probe not-found handling directly,
    # not the separate out-of-scope rejection path.
    tools = backend.get_agent_tools("c", doc_ids=None)
    by_name = {t.name: t for t in tools.function_tools}

    for name in ("get_document", "get_document_structure", "get_page_content"):
        tool = by_name[name]
        kwargs = {"doc_id": "ghost"}
        if name == "get_page_content":
            kwargs["pages"] = "1"
        raw_args = json.dumps(kwargs)
        ctx = ToolContext(context=None, tool_name=name, tool_call_id="1", tool_arguments=raw_args)
        out = _asyncio.run(tool.on_invoke_tool(ctx, raw_args))
        parsed = json.loads(out)
        assert "error" in parsed and "ghost" in parsed["error"], f"{name} did not report not-found consistently: {parsed}"


# ── #6: cloud delete_collection preserves the "folders unavailable" sentinel ──
def test_cloud_delete_collection_preserves_unavailable_sentinel(monkeypatch):
    from pageindex.backend.cloud import CloudBackend

    backend = CloudBackend(api_key="pi-test")
    backend._folder_id_cache["papers"] = None  # folders-unavailable sentinel
    called = []
    monkeypatch.setattr(backend, "_request", lambda *a, **k: called.append(a) or {})
    backend.delete_collection("papers")
    assert not called, "no DELETE should fire when folder_id is the unavailable sentinel"
    assert "papers" in backend._folder_id_cache and backend._folder_id_cache["papers"] is None


def test_cloud_delete_collection_still_clears_real_folder_id(monkeypatch):
    from pageindex.backend.cloud import CloudBackend

    backend = CloudBackend(api_key="pi-test")
    backend._folder_id_cache["papers"] = "folder-123"
    monkeypatch.setattr(backend, "_request", lambda *a, **k: {})
    backend.delete_collection("papers")
    assert "papers" not in backend._folder_id_cache


# ── #7: remove_structure_text is skipped when text was never added ───────────
def _mock_content_based_pipeline(monkeypatch, structure):
    """content_based's real path (_content_based_pipeline) drives real LLM
    calls (TOC detection etc.) regardless of if_add_node_summary — a prior
    version of these two tests didn't mock this out, fell through to it, and
    made real network calls (with a dummy key: 10 retries before failing;
    with a real key: real billable requests) on every run."""
    from pageindex.index import pipeline

    async def fake(page_list, opt):
        return structure

    monkeypatch.setattr(pipeline, "_content_based_pipeline", fake)


def test_build_index_skips_text_strip_when_no_text_was_added(monkeypatch):
    from pageindex.index import pipeline
    from pageindex.parser.protocol import ContentNode, ParsedDocument

    calls = []
    # build_index() imports remove_structure_text locally (`from .utils import
    # ...` inside the function body), so patch it on the utils module itself.
    import pageindex.index.utils as utils_mod
    monkeypatch.setattr(utils_mod, "remove_structure_text", lambda s: calls.append(s) or s)
    _mock_content_based_pipeline(monkeypatch, [{"title": "T", "start_index": 1, "end_index": 1}])

    nodes = [ContentNode(content="page one text", tokens=5, index=1)]
    parsed = ParsedDocument(doc_name="d", nodes=nodes)
    opt = IndexConfig(if_add_node_summary=False, if_add_doc_description=False,
                      if_add_node_text=False)
    pipeline.build_index(parsed, opt=opt)
    assert calls == [], "remove_structure_text must not run when no text was ever added"


def test_build_index_still_strips_text_when_summary_added_it(monkeypatch):
    from pageindex.index import pipeline
    from pageindex.parser.protocol import ContentNode, ParsedDocument

    calls = []
    import pageindex.index.utils as utils_mod
    monkeypatch.setattr(utils_mod, "remove_structure_text", lambda s: calls.append(s) or s)
    _mock_content_based_pipeline(monkeypatch, [{"title": "T", "start_index": 1, "end_index": 1}])
    # Summary generation itself would otherwise make a real LLM call.
    monkeypatch.setattr(
        utils_mod, "generate_summaries_for_structure",
        AsyncMock(side_effect=lambda structure, model=None: [
            n.__setitem__("summary", "fake") for n in structure
        ]),
    )

    nodes = [ContentNode(content="page one text", tokens=5, index=1)]
    parsed = ParsedDocument(doc_name="d", nodes=nodes)
    opt = IndexConfig(if_add_node_summary=True, if_add_doc_description=False,
                      if_add_node_text=False)
    pipeline.build_index(parsed, opt=opt)
    assert len(calls) == 1, "text WAS added for summary generation, so it must still be stripped"


# ── #9/#10: run_pageindex._cli_bool and the shim's md_to_tree are the same
#            object as the canonical implementation (no drift possible) ──────
def test_cli_bool_is_the_canonical_coerce_bool():
    import run_pageindex
    from pageindex.index.page_index_md import _coerce_bool
    assert run_pageindex._cli_bool is _coerce_bool


# ── #11: retrieve._get_md_page_content delegates to the canonical function ───
def test_retrieve_md_page_content_delegates_to_canonical():
    from pageindex import retrieve
    structure = [{"line_num": 5, "text": "five", "nodes": [
        {"line_num": 40, "text": "forty", "nodes": []},
    ]}]
    out = retrieve._get_md_page_content({"structure": structure}, [5])
    assert [r["page"] for r in out] == [5]
