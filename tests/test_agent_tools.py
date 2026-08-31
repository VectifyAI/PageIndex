"""Agent tools layer: cloud-contract parity and behavior against a seeded
local store (no LLM calls; one live parity test gated on PAGEINDEX_API_KEY)."""
import asyncio
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import pytest

import pageindex.agent_tools as agent_tools_module
import pageindex.client as client_module
from pageindex import PageIndexAPIError, PageIndexCloudClient, PageIndexLocalClient
from pageindex.agent_tools import (
    AGENT_INSTRUCTIONS,
    TOOL_CONTRACT,
    call_tool,
    tool_names,
)
from pageindex.local_store import DocStore

SNAPSHOT_PATH = Path(__file__).parent / "data" / "cloud_mcp_contract.json"


def seed_doc(storage_path, doc_id, name, *, created_at="2026-08-01T10:00:00.123000",
             description="A test document", metadata=None, tree=None, pages=None,
             page_num=None):
    pages = pages if pages is not None else [
        {"page_index": 1, "markdown": "Page one text about apples"},
        {"page_index": 2, "markdown": "Page two text about bananas"},
    ]
    tree = tree if tree is not None else [{
        "title": "Doc", "node_id": "0000", "start_index": 1, "end_index": 2,
        "summary": "root summary", "text": "ROOT TEXT",
        "nodes": [
            {"title": "Intro", "node_id": "0001", "start_index": 1,
             "end_index": 1, "summary": "intro summary", "text": "INTRO TEXT"},
            {"title": "Body", "node_id": "0002", "start_index": 2,
             "end_index": 2, "summary": "body summary", "text": "BODY TEXT"},
        ],
    }]
    meta = {
        "id": doc_id, "name": name, "description": description,
        "status": "completed", "createdAt": created_at,
        "pageNum": page_num if page_num is not None else len(pages),
        "folderId": None, "metadata": metadata, "mode": "standard",
    }
    DocStore(storage_path).save_document(doc_id, meta, tree, pages)
    return doc_id


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "store")


@pytest.fixture
def client(store_path):
    return PageIndexLocalClient(storage_path=store_path)


def run(client, name, **arguments):
    text, is_error = call_tool(client, name, arguments)
    return json.loads(text), is_error


# ── contract parity ──

def test_contract_edits_are_deliberate():
    """The committed snapshot cannot detect drift from the live cloud
    server — both copies live in this repo. It exists so a TOOL_CONTRACT
    edit must touch two files in one change, never land by accident."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert snapshot["tools"] == TOOL_CONTRACT


def test_tool_surface_and_docstrings(client):
    import inspect
    from pageindex.agent_tools import _LOCAL_HIDDEN_PARAMS, _local_schema
    tools = client.agent_tools()
    assert [tool.__name__ for tool in tools] == list(tool_names())
    with_management = client.agent_tools(include_management=True)
    assert [tool.__name__ for tool in with_management][-1] == "remove_document"
    for tool in with_management:
        exposed = list(_local_schema(tool.__name__)["properties"])
        assert list(inspect.signature(tool).parameters) == exposed
        for param in exposed:
            assert param in tool.__doc__
        # Cloud-only params are hidden, not documented-then-retracted:
        # strict-schema frameworks cannot express the dead-end calls at all.
        # (The description may still mention them as cloud capabilities.)
        args_section = tool.__doc__.split("Args:", 1)[1]
        for hidden in _LOCAL_HIDDEN_PARAMS.get(tool.__name__, ()):
            assert f"{hidden}:" not in args_section
    docs = {tool.__name__: tool.__doc__ for tool in tools}
    # Tools whose cloud description has no cloud-only content keep it
    # verbatim; browse_documents serves the localized guidance.
    assert docs["get_document"].startswith(
        TOOL_CONTRACT["get_document"]["description"])
    assert docs["browse_documents"].startswith(
        "Primary document retrieval tool")


def test_local_schema_structure_matches_contract():
    """The local surface is the contract minus the documented cloud-only
    params; the surviving params' names, types, defaults, bounds, and
    required stay byte-identical — localization may only touch description
    strings."""
    import copy
    from pageindex.agent_tools import _LOCAL_HIDDEN_PARAMS, _local_schema

    def stripped(schema, drop=()):
        schema = copy.deepcopy(schema)
        for param in drop:
            schema["properties"].pop(param, None)
        for spec in schema["properties"].values():
            spec.pop("description", None)
        return schema

    for name, contract in TOOL_CONTRACT.items():
        hidden = _LOCAL_HIDDEN_PARAMS.get(name, ())
        assert not (set(hidden) & set(contract["schema"].get("required", []))), name
        assert stripped(_local_schema(name)) == stripped(contract["schema"],
                                                         drop=hidden), name


def test_local_guidance_references_only_local_tools(client):
    """Local descriptions must not send the agent to tools that are not
    registered here (the cloud text names search_documents,
    get_folder_structure, and get_document_image)."""
    registered = set(tool_names(include_management=True))
    for tool in client.agent_tools(include_management=True):
        named = set(re.findall(r"\b(\w+)\(", tool.__doc__))
        assert named <= registered, (tool.__name__, named - registered)


def test_local_guidance_points_cloud_only_capabilities_at_cloud(client):
    tools = client.agent_tools(include_management=True)
    browse = tools[0].__doc__
    assert "not supported in local mode yet" in browse
    assert "PageIndex cloud" in browse
    # Capability-phrase guard, all docstrings: cloud-only language must not
    # drift back in via a contract refresh. browse alone keeps exactly one
    # sort="relevance" mention — the sanctioned pointer to the cloud.
    for tool in tools:
        doc = tool.__doc__
        for phrase in ("shared-with-me", "sub-folder", "get_folder_structure",
                       "search_documents", "get_document_image"):
            assert phrase not in doc, (tool.__name__, phrase)
        expected = 1 if tool.__name__ == "browse_documents" else 0
        assert doc.count('sort="relevance"') == expected, tool.__name__


# ── browse_documents ──

def test_browse_documents_shape(client, store_path):
    seed_doc(store_path, "pi-a", "older.pdf", created_at="2026-08-01T10:00:00.123000")
    seed_doc(store_path, "pi-b", "newer.pdf", created_at="2026-08-02T10:00:00.456000",
             metadata={"team": "research", "year": 2026, "nested": {"x": 1}})
    payload, is_error = run(client, "browse_documents")
    assert not is_error
    assert payload["success"] is True
    assert payload["folders"] == []
    assert payload["has_more"] is False
    assert payload["next_offset"] is None
    names = [doc["name"] for doc in payload["documents"]]
    assert names == ["newer.pdf", "older.pdf"]
    newer = payload["documents"][0]
    assert newer["status"] == "completed"
    assert newer["created_at"] == "2026-08-02T10:00:00.456Z"
    assert newer["metadata"] == {"team": "research", "year": 2026}
    assert "folder_id" not in newer
    assert "next_steps" in payload

    flat, _ = run(client, "browse_documents", recursive=True)
    assert "folders" not in flat


def test_browse_documents_pagination(client, store_path):
    for index in range(3):
        seed_doc(store_path, f"pi-{index}", f"doc{index}.pdf",
                 created_at=f"2026-08-0{index + 1}T10:00:00.000000")
    first, _ = run(client, "browse_documents", limit=2)
    assert [d["name"] for d in first["documents"]] == ["doc2.pdf", "doc1.pdf"]
    assert first["has_more"] is True and first["next_offset"] == 2
    assert "page through the rest" in json.dumps(first["next_steps"])
    second, _ = run(client, "browse_documents", limit=2, offset=2)
    assert [d["name"] for d in second["documents"]] == ["doc0.pdf"]
    assert second["has_more"] is False
    # No paging advice when there is nothing left to page through.
    assert "page through the rest" not in json.dumps(second["next_steps"])


def test_browse_documents_relevance_unsupported(client, store_path):
    """Semantic ranking is cloud-side; like folders, local answers with an
    honest error instead of a keyword imitation."""
    seed_doc(store_path, "pi-a", "attention.pdf",
             description="Transformers and attention mechanisms")
    payload, is_error = run(client, "browse_documents", sort="relevance",
                            query="attention transformers")
    assert is_error and payload["errorCode"] == "INVALID_INPUT"
    assert "not supported in local mode" in payload["error"]

    stray_query, is_error = run(client, "browse_documents", query="x")
    assert is_error and "not supported in local mode" in stray_query["error"]
    bad_sort, is_error = run(client, "browse_documents", sort="banana")
    assert is_error and bad_sort["errorCode"] == "INVALID_INPUT"
    # The invalid-sort guidance must not prescribe the cloud-only value.
    assert 'Use sort="relevance"' not in json.dumps(bad_sort)
    assert "local mode" in bad_sort["error"]


def test_expand_pages_enforces_contract_pattern():
    """int() alone is far laxer than the published pages pattern; an
    out-of-contract spelling must reject, never resolve to another page."""
    from pageindex.agent_tools import _PageSpecError, _expand_pages
    assert _expand_pages("1-3, 7") == [1, 2, 3, 7]
    for bad in ["1_0", "+5", "٥", "１", " 1", "1 - 3"]:
        with pytest.raises(_PageSpecError) as excinfo:
            _expand_pages(bad)
        assert excinfo.value.code == "invalid"


def test_browse_documents_empty_and_folder_error(client):
    payload, is_error = run(client, "browse_documents")
    assert not is_error
    assert payload["documents"] == []
    assert "submit_document" in json.dumps(payload)

    folder, is_error = run(client, "browse_documents", folder_id="folder-123")
    assert is_error and folder["errorCode"] == "INVALID_INPUT"


# ── get_document ──

def test_get_document(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf", metadata={"team": "research"})
    payload, is_error = run(client, "get_document", doc_name="report.pdf")
    assert not is_error
    assert payload["name"] == "report.pdf"
    assert payload["status"] == "completed"
    assert payload["page_count"] == 2
    assert payload["folder_id"] is None
    assert payload["created_at"].endswith("Z")
    assert payload["metadata"] == {"team": "research"}
    assert any("short document" in option
               for option in payload["next_steps"]["options"])


def test_get_document_not_found_suggests_similar(client, store_path):
    seed_doc(store_path, "pi-a", "annual-report.pdf")
    payload, is_error = run(client, "get_document", doc_name="anual-report.pdf")
    assert is_error
    assert payload["errorCode"] == "NOT_FOUND"
    assert "annual-report.pdf" in payload["similar_files"]
    assert "Did you mean" in payload["error"]


def test_get_document_duplicate_names_resolve_newest(client, store_path):
    seed_doc(store_path, "pi-old", "same.pdf", description="old copy",
             created_at="2026-08-01T10:00:00.000000")
    seed_doc(store_path, "pi-new", "same.pdf", description="new copy",
             created_at="2026-08-02T10:00:00.000000")
    payload, _ = run(client, "get_document", doc_name="same.pdf")
    assert payload["description"] == "new copy"


# ── get_document_structure ──

def test_structure_strips_text_and_orders_keys(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_document_structure", doc_name="report.pdf")
    assert not is_error
    assert payload["doc_name"] == "report.pdf"
    assert "pagination" not in payload and "total_parts" not in payload
    serialized = json.dumps(payload["structure"])
    assert "ROOT TEXT" not in serialized and "INTRO TEXT" not in serialized
    # Cloud structure node shape: start_index/end_index/summary (live-verified).
    root = payload["structure"][0]
    assert list(root)[:4] == ["title", "node_id", "start_index", "end_index"]
    assert root["summary"] == "root summary"
    assert (root["start_index"], root["end_index"]) == (1, 2)
    assert root["nodes"][0]["summary"] == "intro summary"
    assert root["nodes"][0]["end_index"] == 1


def test_structure_multipart_pagination(client, store_path):
    big_tree = [{
        "title": f"Chapter {index}", "node_id": f"{index:04d}",
        "start_index": index + 1, "end_index": index + 1,
        "summary": "s" * 4000, "text": "T",
    } for index in range(60)]
    seed_doc(store_path, "pi-big", "big.pdf", tree=big_tree,
             pages=[{"page_index": 1, "markdown": "x"}])
    first, _ = run(client, "get_document_structure", doc_name="big.pdf")
    assert first["total_parts"] > 1
    assert first["pagination"] == {
        "part": 1, "total_parts": first["total_parts"], "has_more": True,
    }
    titles = []
    for part in range(1, first["total_parts"] + 1):
        payload, _ = run(client, "get_document_structure", doc_name="big.pdf",
                         part=part)
        # Every part of one paginated response is a list — a consumer that
        # iterates part 1 must not silently iterate dict keys on part 2.
        assert isinstance(payload["structure"], list)
        titles.extend(node["title"] for node in payload["structure"])
        assert payload["pagination"]["has_more"] == (part < first["total_parts"])
    assert titles == [f"Chapter {index}" for index in range(60)]

    clamped, _ = run(client, "get_document_structure", doc_name="big.pdf",
                     part=999)
    assert clamped["pagination"]["part"] == first["total_parts"]


def test_split_structure_chunks_never_change_type():
    """A single-node group used to come out as a bare dict while its
    sibling parts were lists — same response sequence, flipping JSON type."""
    from pageindex.agent_tools import _split_structure
    small = {"title": "s", "node_id": "0001"}
    big = {"title": "b", "node_id": "0002",
           "nodes": [{"title": f"c{index}", "summary": "x" * 40}
                     for index in range(10)]}
    chunks = _split_structure([small, small, big], 200)
    assert len(chunks) > 1
    assert all(isinstance(chunk, list) for chunk in chunks)
    # Unsplit structures keep their natural shape (cloud fallback parity).
    assert _split_structure(small, 10_000) == [small]
    assert _split_structure([small], 10_000) == [[small]]


# ── get_page_content ──

def test_page_content(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages="1-2")
    assert not is_error
    assert payload["total_pages"] == 2
    assert payload["requested_pages"] == "1-2"
    assert payload["returned_pages"] == "1-2"
    assert payload["content"] == [
        {"page": 1, "text": "Page one text about apples"},
        {"page": 2, "text": "Page two text about bananas"},
    ]


def test_page_content_out_of_range(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    mixed, is_error = run(client, "get_page_content", doc_name="report.pdf",
                          pages="1,99")
    assert not is_error
    assert mixed["returned_pages"] == "1"
    assert "out of range" in mixed["next_steps"]["summary"]

    all_out, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages="99")
    assert is_error and all_out["errorCode"] == "INVALID_INPUT"
    assert all_out["max_pages"] == 2


def test_out_of_range_pages_reported_as_ranges(client, store_path):
    """Spans compress — enumerating them one by one buries the response."""
    seed_doc(store_path, "pi-a", "report.pdf")
    partial, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages="1,5-9")
    assert not is_error
    assert "Pages 5-9 were out of range" in partial["next_steps"]["summary"]

    spread, is_error = run(client, "get_page_content", doc_name="report.pdf",
                           pages="1,5,9")
    assert not is_error
    assert "Pages 5,9 were out of range" in spread["next_steps"]["summary"]

    all_out, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages="5-9")
    assert is_error
    assert all_out["error"].endswith("you requested pages: 5-9")
    assert all_out["requested_pages"] == "5-9"


@pytest.mark.parametrize("bad_spec", ["abc", "5-3", "1,,2", "-3", ""])
def test_page_content_invalid_spec(client, store_path, bad_spec):
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages=bad_spec)
    assert is_error and payload["errorCode"] == "INVALID_INPUT"


def test_page_content_zero_page_rejected(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages="0")
    assert is_error
    assert "positive integers" in payload["error"]


def test_page_content_preserves_blank_pages(client, store_path):
    pages = [
        {"page_index": 1, "markdown": ""},
        {"page_index": 2, "markdown": "content"},
    ]
    seed_doc(store_path, "pi-a", "blanks.pdf", pages=pages)
    payload, is_error = run(client, "get_page_content", doc_name="blanks.pdf",
                            pages="1-2")
    assert not is_error
    assert payload["content"][0] == {"page": 1, "text": ""}
    assert payload["content"][1] == {"page": 2, "text": "content"}


def test_created_at_accepts_z_suffixed_input(client, store_path):
    seed_doc(store_path, "pi-a", "cloudlike.pdf",
             created_at="2026-08-01T10:00:00.123Z")
    payload, _ = run(client, "browse_documents")
    assert payload["documents"][0]["created_at"] == "2026-08-01T10:00:00.123Z"


def test_page_content_char_budget(client, store_path):
    # Escape-dense pages: raw length fits the budget, JSON-serialized
    # length does not — the budget must count emitted characters.
    pages = [
        {"page_index": 1, "markdown": '"' * 30_000},
        {"page_index": 2, "markdown": '"' * 20_000},
    ]
    seed_doc(store_path, "pi-a", "huge.pdf", pages=pages)
    payload, is_error = run(client, "get_page_content", doc_name="huge.pdf",
                            pages="1-2")
    assert not is_error
    assert payload["returned_pages"] == "1"
    assert "size limits" in payload["next_steps"]["summary"]
    assert any("For remaining pages, request: 2" in option
               for option in payload["next_steps"]["options"])
    emitted = json.dumps(payload, ensure_ascii=False)
    assert len(emitted) <= agent_tools_module.TOOL_RESPONSE_CHAR_LIMIT


def test_page_content_reports_truncation_and_out_of_range_together(
        client, store_path):
    """Size truncation must not hide behind the out-of-range report (or
    vice versa) — the agent otherwise believes it holds every in-range
    page."""
    pages = [
        {"page_index": 1, "markdown": "x" * 96_000},
        {"page_index": 2, "markdown": "short"},
    ]
    seed_doc(store_path, "pi-a", "huge.pdf", pages=pages)
    payload, is_error = run(client, "get_page_content", doc_name="huge.pdf",
                            pages="1-2,99")
    assert not is_error
    assert payload["returned_pages"] == "1"
    summary = payload["next_steps"]["summary"]
    assert "size limits" in summary and "out of range" in summary


# ── remove_document (management-gated) ──

def test_remove_document(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "remove_document",
                            doc_names=["report.pdf", "ghost.pdf"])
    assert not is_error
    assert payload["results"] == [
        {"doc_name": "report.pdf", "status": "deleted"},
        {"doc_name": "ghost.pdf", "status": "not_found"},
    ]
    assert client.list_documents()["total"] == 0


def test_remove_document_rejects_non_string_names_before_deleting(client,
                                                                  store_path):
    """A rejection envelope must mean nothing was destroyed — the bad
    element is caught before the delete loop starts."""
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "remove_document",
                            doc_names=["report.pdf", 123])
    assert is_error and payload["errorCode"] == "INVALID_INPUT"
    assert client.list_documents()["total"] == 1


def test_remove_document_partial_failure_keeps_results(client, store_path,
                                                       monkeypatch):
    """A non-API error mid-batch must not discard the entries for documents
    already irreversibly deleted — a generic INTERNAL_ERROR envelope would
    tell the agent nothing was removed and to retry."""
    seed_doc(store_path, "pi-a", "a.pdf")
    seed_doc(store_path, "pi-b", "b.pdf")
    real = client.delete_document

    def flaky(doc_id):
        if doc_id == "pi-b":
            raise OSError(13, "Permission denied")
        return real(doc_id)

    monkeypatch.setattr(client, "delete_document", flaky)
    payload, is_error = run(client, "remove_document",
                            doc_names=["a.pdf", "b.pdf"])
    assert not is_error
    assert payload["results"] == [
        {"doc_name": "a.pdf", "status": "deleted"},
        {"doc_name": "b.pdf", "status": "failed",
         "error": "[Errno 13] Permission denied"},
    ]


def test_management_tools_hidden_by_default(client):
    assert "remove_document" not in [t.__name__ for t in client.agent_tools()]


# ── doc_id scope (the local chat surfaces' allowlist) ──

def test_call_tool_doc_scope_limits_every_lookup(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "payroll.pdf",
             created_at="2026-08-02T10:00:00.123000")

    text, is_error = call_tool(client, "browse_documents", {},
                               doc_ids=["pi-a"])
    browse = json.loads(text)
    assert not is_error
    assert [doc["name"] for doc in browse["documents"]] == ["report.pdf"]
    assert browse["has_more"] is False

    text, is_error = call_tool(client, "get_page_content",
                               {"doc_name": "payroll.pdf", "pages": "1"},
                               doc_ids="pi-a")
    assert is_error and json.loads(text)["errorCode"] == "NOT_FOUND"

    text, is_error = call_tool(client, "get_document",
                               {"doc_name": "report.pdf"}, doc_ids="pi-a")
    assert not is_error

    # An empty allowlist scopes to nothing — it must not read as "unscoped".
    text, is_error = call_tool(client, "browse_documents", {}, doc_ids=[])
    assert not is_error and json.loads(text)["documents"] == []


def test_call_tool_scope_channel_not_injectable(client, store_path):
    """Model arguments cannot smuggle an allowlist: underscore keys are
    stripped before binding."""
    seed_doc(store_path, "pi-a", "report.pdf")
    text, is_error = call_tool(client, "browse_documents",
                               {"_allowed_ids": ["pi-none"]})
    assert not is_error
    assert json.loads(text)["documents"]


# ── error containment ──

def test_tools_never_raise(client, store_path, monkeypatch):
    seed_doc(store_path, "pi-a", "report.pdf")
    monkeypatch.setattr(client._api._store, "get_tree",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    payload, is_error = run(client, "get_document_structure",
                            doc_name="report.pdf")
    assert is_error
    assert "boom" in payload["error"]


def test_unknown_argument_becomes_error_envelope(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_document", doc_name="report.pdf",
                            bogus=True)
    assert is_error and payload["errorCode"] == "INVALID_INPUT"


def test_execution_type_error_is_internal_not_invalid_input(client, store_path,
                                                            monkeypatch):
    """Only bind-time TypeErrors are argument errors; a TypeError raised
    mid-execution must not masquerade as an input rejection."""
    seed_doc(store_path, "pi-a", "report.pdf")
    monkeypatch.setattr(client._api._store, "get_tree",
                        lambda *a, **k: (_ for _ in ()).throw(
                            TypeError("wrong shape")))
    payload, is_error = run(client, "get_document_structure",
                            doc_name="report.pdf")
    assert is_error and payload["errorCode"] == "INTERNAL_ERROR"
    assert "wrong shape" in payload["error"]


def test_unknown_tool_envelope_uses_standard_formatting(client):
    text, is_error = call_tool(client, "nope", {})
    assert is_error
    assert text == json.dumps(json.loads(text), ensure_ascii=False)


# ── framework adapters ──

def test_as_openai_tools_missing_dependency(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    with pytest.raises(PageIndexAPIError, match="openai-agents"):
        client.as_openai_tools()


def test_as_openai_tools_local_in_process(client):
    pytest.importorskip("agents")
    tools = client.as_openai_tools()
    assert [tool.name for tool in tools] == list(tool_names())


def test_as_openai_tools_cloud_default_uses_bridge(monkeypatch):
    pytest.importorskip("agents")
    from agents import FunctionTool
    import pageindex.mcp_bridge as mcp_bridge
    monkeypatch.setattr(mcp_bridge, "McpBridge", _FakeBridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    tools = cloud.as_openai_tools()
    assert all(isinstance(tool, FunctionTool) for tool in tools)
    assert [tool.name for tool in tools] == ["search_documents", "get_document"]


def test_as_openai_tools_cloud_hosted_opt_in():
    pytest.importorskip("agents")
    from agents import HostedMCPTool
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    tools = cloud.as_openai_tools(hosted=True)
    assert len(tools) == 1
    assert isinstance(tools[0], HostedMCPTool)
    config = tools[0].tool_config
    assert config["server_url"] == "https://api.pageindex.ai/mcp?tools=read"
    assert config["headers"] == {"Authorization": "Bearer pi-test-key"}
    assert config["server_label"] == "pageindex"


def test_as_openai_tools_local_ignores_hosted(client):
    pytest.importorskip("agents")
    assert ([tool.name for tool in client.as_openai_tools(hosted=True)]
            == [tool.name for tool in client.as_openai_tools()]
            == list(tool_names()))


def test_as_openai_tools_schemas_pass_through_verbatim(client):
    """The contract schema goes to the model as-is — regenerating it from a
    Python signature dropped items/enum/pattern/bounds."""
    pytest.importorskip("agents")
    from pageindex.agent_tools import _local_schema
    tools = {tool.name: tool
             for tool in client.as_openai_tools(include_management=True)}
    assert (tools["remove_document"].params_json_schema
            == _local_schema("remove_document"))
    pages = tools["get_page_content"].params_json_schema["properties"]["pages"]
    assert pages["pattern"] and pages["minLength"] == 1
    assert all(tool.strict_json_schema is False for tool in tools.values())


def test_as_openai_tools_invocation_runs_call_tool(client, store_path):
    pytest.importorskip("agents")
    seed_doc(store_path, "pi-a", "report.pdf")
    tool = {t.name: t for t in client.as_openai_tools()}["get_document"]
    out = asyncio.run(tool.on_invoke_tool(
        None, json.dumps({"doc_name": "report.pdf", "folder_id": None})))
    payload = json.loads(out)
    assert payload["success"] is True and payload["name"] == "report.pdf"


def test_as_openai_tools_malformed_args_answer_the_model(client, store_path):
    """strict_json_schema is off, so a truncated or non-object argument
    string is reachable; raising here aborted the caller's whole run —
    the model must get the guided envelope back and retry instead."""
    pytest.importorskip("agents")
    seed_doc(store_path, "pi-a", "report.pdf")
    tool = {t.name: t for t in client.as_openai_tools()}["get_document"]
    for bad in ('{not json', '[1, 2]', '"x"', 'null'):
        out = asyncio.run(tool.on_invoke_tool(None, bad))
        payload = json.loads(out)
        assert payload["errorCode"] == "INVALID_INPUT"
        assert "JSON object" in payload["error"]


def test_as_openai_tools_cloud_object_params_survive(monkeypatch):
    """An object-typed server parameter used to abort the whole build with
    agents.exceptions.UserError; array items used to degrade to {}."""
    pytest.importorskip("agents")
    import pageindex.mcp_bridge as mcp_bridge

    schema = {
        "type": "object",
        "properties": {
            "filters": {"type": "object", "additionalProperties": False},
            "paths": {"type": "array",
                      "items": {"type": "string", "minLength": 1}},
        },
        "required": ["paths"],
    }

    class _ObjBridge:
        def __init__(self, url, headers):
            pass

        def list_tools(self):
            return [{"name": "get_document_image",
                     "description": "d",
                     "annotations": {"readOnlyHint": True},
                     "inputSchema": schema}]

        def call_tool(self, name, arguments):
            return json.dumps({"success": True}), False

    monkeypatch.setattr(mcp_bridge, "McpBridge", _ObjBridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    tools = cloud.as_openai_tools()
    assert len(tools) == 1
    assert tools[0].params_json_schema == schema
    assert tools[0].params_json_schema is not schema  # copied, not aliased


def test_as_claude_mcp_cloud_needs_no_framework(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    # The URL is the gate: default → read-only endpoint, management opt-in
    # → the full tool set.
    assert cloud.as_claude_mcp() == {
        "type": "http",
        "url": "https://api.pageindex.ai/mcp?tools=read",
        "headers": {"Authorization": "Bearer pi-test-key"},
    }
    assert (cloud.as_claude_mcp(include_management=True)["url"]
            == "https://api.pageindex.ai/mcp")


def test_as_claude_mcp_local_missing_dependency(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(PageIndexAPIError, match="claude-agent-sdk"):
        client.as_claude_mcp()


def test_as_claude_mcp_local_when_installed(client):
    pytest.importorskip("claude_agent_sdk")
    server = client.as_claude_mcp()
    assert server is not None
    if isinstance(server, dict):
        assert server.get("type") != "http"


def test_claude_agent_config_is_sugar_over_the_explicit_form(
        cloud_with_fake_bridge):
    cloud, _ = cloud_with_fake_bridge
    config = cloud.claude_agent_config()
    assert config["system_prompt"] == "SERVER GUIDANCE"
    server = config["mcp_servers"]["pageindex"]
    assert server["type"] == "http"
    assert server["url"] == "https://api.pageindex.ai/mcp?tools=read"
    # Pre-approval only: the URL is the gate.
    assert config["allowed_tools"] == ["mcp__pageindex"]
    renamed = cloud.claude_agent_config(server_name="docs",
                                        include_management=True)
    assert set(renamed["mcp_servers"]) == {"docs"}
    assert renamed["mcp_servers"]["docs"]["url"] == "https://api.pageindex.ai/mcp"
    assert renamed["allowed_tools"] == ["mcp__docs"]


def test_claude_agent_config_local(client, store_path):
    pytest.importorskip("claude_agent_sdk")
    seed_doc(store_path, "pi-a", "report.pdf")
    config = client.claude_agent_config(doc_id="pi-a")
    assert "report.pdf" in config["system_prompt"]
    assert config["allowed_tools"] == ["mcp__pageindex"]
    assert config["mcp_servers"]["pageindex"]["name"] == "pageindex"
    # The SDK server's declared identity follows the registration key.
    renamed = client.claude_agent_config(server_name="docs")
    assert renamed["mcp_servers"]["docs"]["name"] == "docs"
    assert renamed["allowed_tools"] == ["mcp__docs"]


def test_claude_agent_config_forwards_a_claude_chat_model(cloud_with_fake_bridge):
    # chat_model says who answers; the Claude Agent SDK takes Anthropic's
    # own name, so LiteLLM's routing prefix is stripped.
    for name in ("anthropic/claude-sonnet-4-6", "claude-sonnet-4-6",
                 "litellm/anthropic/claude-sonnet-4-6"):
        cloud = PageIndexCloudClient(api_key="pi-test-key", chat_model=name)
        assert cloud.claude_agent_config()["model"] == "claude-sonnet-4-6"
    # Names are sent as written — no model map gates them.
    for name in ("anthropic/claude-3-5-sonnet-latest",
                 "claude-3-5-sonnet-latest"):
        cloud = PageIndexCloudClient(api_key="pi-test-key", chat_model=name)
        assert cloud.claude_agent_config()["model"] == "claude-3-5-sonnet-latest"


def test_claude_agent_config_carries_any_chosen_chat_model(
        cloud_with_fake_bridge):
    # A model you set is sent as written — the destination judges the id.
    for name in ("gpt-4.1", "openrouter/anthropic/claude-sonnet-4-6"):
        cloud = PageIndexCloudClient(api_key="pi-test-key", chat_model=name)
        assert cloud.claude_agent_config()["model"] == name
        # An explicit model may be the SDK's own name (an alias included)...
        assert cloud.claude_agent_config(model="sonnet")["model"] == "sonnet"
        # ... or the client's spelling, read exactly like chat_model.
        for spelling in ("anthropic/claude-sonnet-4-6",
                         "litellm/anthropic/claude-sonnet-4-6"):
            assert (cloud.claude_agent_config(model=spelling)["model"]
                    == "claude-sonnet-4-6")
    # Explicitly writing the stock value is a choice too: it carries.
    cloud = PageIndexCloudClient(api_key="pi-test-key", chat_model="gpt-5.6-sol")
    assert cloud.claude_agent_config()["model"] == "gpt-5.6-sol"


def test_claude_agent_config_managed_chat_sets_no_model(cloud_with_fake_bridge):
    cloud, _ = cloud_with_fake_bridge
    assert "model" not in cloud.claude_agent_config()


def test_claude_agent_config_local_chat_model(store_path):
    pytest.importorskip("claude_agent_sdk")
    local = PageIndexLocalClient(storage_path=store_path,
                                 chat_model="anthropic/claude-sonnet-4-6")
    assert local.claude_agent_config()["model"] == "claude-sonnet-4-6"
    assert "model" not in PageIndexLocalClient(
        storage_path=store_path).claude_agent_config()


def test_openai_agent_config_local(client, store_path):
    pytest.importorskip("agents")
    from agents import Agent
    seed_doc(store_path, "pi-a", "report.pdf")
    config = client.openai_agent_config(doc_id="pi-a")
    assert config["name"] == "PageIndex"
    assert "report.pdf" in config["instructions"]
    assert [tool.name for tool in config["tools"]] == list(tool_names())
    assert config["model"] == client.retrieve_model
    assert client.openai_agent_config(model="gpt-x")["model"] == "gpt-x"
    assert Agent(**client.openai_agent_config()).name == "PageIndex"
    assert client.openai_agent_config(name="Researcher")["name"] == "Researcher"


def test_openai_agent_config_model_speaks_the_agents_sdk_grammar(tmp_path):
    """The bundle's model string is resolved by the Agents SDK's own
    prefix grammar, which refuses unknown prefixes — the constructor's
    normalized litellm/ spelling is what must reach this door."""
    pytest.importorskip("agents")
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  retrieve_model="anthropic/claude-x")
    assert (client.openai_agent_config()["model"]
            == "litellm/anthropic/claude-x")
    # The per-call override speaks the same grammar as chat_model.
    assert (client.openai_agent_config(model="anthropic/claude-y")["model"]
            == "litellm/anthropic/claude-y")
    assert (client.openai_agent_config(model="litellm/groq/llama-x")["model"]
            == "litellm/groq/llama-x")


def test_openai_agent_config_carries_cache_marks_for_litellm_claude(tmp_path):
    """LiteLLM-routed Claude gets the same cache marks the engine
    attaches in chat_completions(); OpenAI-bound models stay unmarked
    (their caching is server-side, and LiteLLM seeds nothing on its
    own)."""
    pytest.importorskip("agents")
    from pageindex.local_chat import _cache_extra_args
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  chat_model="anthropic/claude-x")
    settings = client.openai_agent_config()["model_settings"]
    assert settings.extra_args == _cache_extra_args("anthropic/claude-x")
    assert "cache_control_injection_points" in settings.extra_args
    assert "model_settings" not in client.openai_agent_config(model="gpt-x")
    # The per-call override is marked by its own routing, not the default's.
    marked = client.openai_agent_config(model="bedrock/claude-y")
    assert "cache_control_injection_points" in marked["model_settings"].extra_args


def test_openai_agent_config_marks_bare_claude_behind_litellm_prefix(tmp_path):
    """In this lane litellm/<bare-claude> routes to Anthropic — the Agents
    SDK strips the prefix and LiteLLM resolves the bare name — unlike the
    chat lane, whose wire treats bare names as OpenAI shorthand. The marks
    follow this lane's routing, not the chat lane's."""
    pytest.importorskip("agents")
    pytest.importorskip("litellm")
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"))
    cfg = client.openai_agent_config(model="litellm/claude-sonnet-4-5")
    assert cfg["model"] == "litellm/claude-sonnet-4-5"
    assert "cache_control_injection_points" in cfg["model_settings"].extra_args
    # Without the prefix the SDK's default OpenAI provider serves the name.
    assert "model_settings" not in client.openai_agent_config(
        model="claude-sonnet-4-5")
    assert "model_settings" not in client.openai_agent_config(
        model="litellm/gpt-4o")


def test_openai_agent_config_merges_caller_model_settings(tmp_path):
    """Caller model_settings merge on top of the bundled cache marks
    (caller fields win, extra_args dict-merge); with no marks the
    caller's object rides through verbatim."""
    pytest.importorskip("agents")
    from agents import ModelSettings
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  chat_model="anthropic/claude-x")
    mine = ModelSettings(temperature=0.2, extra_args={"top_k": 5})
    merged = client.openai_agent_config(model_settings=mine)["model_settings"]
    assert merged.temperature == 0.2
    assert merged.extra_args["top_k"] == 5
    assert "cache_control_injection_points" in merged.extra_args
    verbatim = client.openai_agent_config(model="gpt-x", model_settings=mine)
    assert verbatim["model_settings"] is mine


def test_plain_functions_answer_bad_arguments_with_the_envelope(client,
                                                                store_path):
    """agent_tools() functions must not raise into a framework loop:
    cloud-only parameters pruned from the local signature come back as
    the guided envelope, and the schema-bearing signature survives."""
    import inspect
    seed_doc(store_path, "pi-a", "report.pdf")
    tools = {f.__name__: f for f in client.agent_tools()}
    fn = tools["get_document"]
    assert "doc_name" in inspect.signature(fn).parameters
    payload = json.loads(fn(doc_name="report.pdf", folder_id="root"))
    assert payload["errorCode"] == "INVALID_INPUT"
    ok = json.loads(fn(doc_name="report.pdf"))
    assert not ok.get("errorCode")


def test_non_string_doc_name_stays_not_found(client, store_path):
    """A type-loose model argument must not turn NOT_FOUND into the
    retry-inviting INTERNAL_ERROR (strict_json_schema is off on the
    OpenAI adapter, so nothing upstream validates the type)."""
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_document", doc_name=5)
    assert is_error and payload["errorCode"] == "NOT_FOUND"


@pytest.mark.parametrize("repair", ["_repair_litellm_types",
                                    "_mute_litellm_bridge_usage_warning"])
def test_openai_agent_config_repairs_litellm_types(tmp_path, monkeypatch,
                                                   repair):
    """The BYO path resolves its model through LiteLLM in the caller's
    process, outside our completion helpers — the LiteLLM repairs must
    run at config time, and only for LiteLLM-routed models."""
    pytest.importorskip("agents")
    import pageindex.utils
    calls = []
    monkeypatch.setattr(pageindex.utils, repair,
                        lambda: calls.append(True))
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  chat_model="anthropic/claude-x")
    client.openai_agent_config()
    assert calls
    calls.clear()
    client.openai_agent_config(model="gpt-plain")
    assert not calls


def test_openai_agent_config_cloud_omits_model(cloud_with_fake_bridge):
    pytest.importorskip("agents")
    cloud, _ = cloud_with_fake_bridge
    config = cloud.openai_agent_config()
    assert "model" not in config
    assert config["instructions"] == "SERVER GUIDANCE"
    assert [tool.name for tool in config["tools"]] == ["search_documents",
                                                       "get_document"]


def test_anthropic_runner_config_accepts_the_litellm_spelling(client):
    pytest.importorskip("anthropic")
    config = client.anthropic_runner_config(
        model="anthropic/claude-3-opus-20240229")
    assert config["model"] == "claude-3-opus-20240229"
    assert config["max_tokens"] == 4096   # resolved on the stripped id


def test_anthropic_runner_config_carries_a_claude_chat_model(client, store_path):
    pytest.importorskip("anthropic")
    local = PageIndexLocalClient(storage_path=store_path,
                                 chat_model="anthropic/claude-3-opus-20240229")
    config = local.anthropic_runner_config()
    assert config["model"] == "claude-3-opus-20240229"
    assert config["max_tokens"] == 4096
    # The stock default was never chosen: nothing to send.
    with pytest.raises(PageIndexAPIError, match="needs a model"):
        client.anthropic_runner_config()


def test_anthropic_runner_config_shapes(client, store_path):
    pytest.importorskip("anthropic")
    import anthropic
    from anthropic.lib.tools import BetaAsyncFunctionTool
    seed_doc(store_path, "pi-a", "report.pdf")
    config = client.anthropic_runner_config(model="claude-3-opus-20240229",
                                            doc_id="pi-a")
    assert config["max_tokens"] == 4096
    assert config["max_iterations"] == 10
    assert config["cache_control"] == {"type": "ephemeral"}
    assert "report.pdf" in config["system"]
    assert [tool.name for tool in config["tools"]] == list(tool_names())
    assert (client.anthropic_runner_config(model="claude-sonnet-4-5")
            ["max_tokens"] == 8192)
    override = client.anthropic_runner_config(model="claude-sonnet-4-5",
                                              max_tokens=99, max_turns=3)
    assert override["max_tokens"] == 99 and override["max_iterations"] == 3
    async_tools = client.anthropic_runner_config(
        model="claude-sonnet-4-5", asynchronous=True)["tools"]
    assert all(isinstance(tool, BetaAsyncFunctionTool)
               for tool in async_tools)
    # The kwargs must construct a real runner (construction is offline —
    # requests start on iteration), pinning tool_runner's parameter names.
    runner = anthropic.Anthropic(api_key="test").beta.messages.tool_runner(
        **client.anthropic_runner_config(model="claude-sonnet-4-5"),
        messages=[{"role": "user", "content": "q"}])
    assert runner is not None


def test_anthropic_runner_config_cloud(cloud_with_fake_bridge):
    pytest.importorskip("anthropic")
    cloud, _ = cloud_with_fake_bridge
    config = cloud.anthropic_runner_config(model="claude-sonnet-4-5")
    assert config["system"] == "SERVER GUIDANCE"
    assert [tool.name for tool in config["tools"]] == ["search_documents",
                                                       "get_document"]


# ── config helpers: doc_id is structural in the tools, not just prompted ──

def test_openai_agent_config_doc_scope_enforced_in_tools(client, store_path):
    pytest.importorskip("agents")
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "payroll.pdf",
             created_at="2026-08-02T10:00:00.123000")
    tools = {tool.name: tool
             for tool in client.openai_agent_config(doc_id="pi-a")["tools"]}
    out = asyncio.run(tools["get_page_content"].on_invoke_tool(
        None, json.dumps({"doc_name": "payroll.pdf", "pages": "1"})))
    assert json.loads(out)["errorCode"] == "NOT_FOUND"
    out = asyncio.run(tools["browse_documents"].on_invoke_tool(None, "{}"))
    assert [doc["name"]
            for doc in json.loads(out)["documents"]] == ["report.pdf"]


def test_anthropic_runner_config_doc_scope_enforced_in_tools(client,
                                                             store_path):
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import ToolError
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "payroll.pdf",
             created_at="2026-08-02T10:00:00.123000")
    config = client.anthropic_runner_config(model="claude-sonnet-4-5",
                                            doc_id="pi-a")
    tools = {tool.name: tool for tool in config["tools"]}
    with pytest.raises(ToolError, match="NOT_FOUND"):
        tools["get_page_content"].call({"doc_name": "payroll.pdf",
                                        "pages": "1"})
    browse = json.loads(tools["browse_documents"].call({}))
    assert [doc["name"] for doc in browse["documents"]] == ["report.pdf"]


def test_claude_agent_config_doc_scope_enforced_in_tools(client, store_path,
                                                         monkeypatch):
    """claude_agent_config(doc_id=...) must wire scope all the way into the
    handlers it registers — an out-of-scope document returns NOT_FOUND. The
    assertion has to drive those handlers, or build_claude_mcp's doc_ids
    pass-through goes unguarded."""
    claude_agent_sdk = pytest.importorskip("claude_agent_sdk")
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "payroll.pdf",
             created_at="2026-08-02T10:00:00.123000")

    registered = {}
    create_server = claude_agent_sdk.create_sdk_mcp_server

    def capture(**kwargs):
        registered.update(kwargs)
        return create_server(**kwargs)

    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", capture)
    config = client.claude_agent_config(doc_id="pi-a")
    assert "report.pdf" in config["system_prompt"]
    handlers = {spec.name: spec.handler for spec in registered["tools"]}
    result = asyncio.run(handlers["get_page_content"](
        {"doc_name": "payroll.pdf", "pages": "1"}))
    assert result.get("is_error")
    assert json.loads(result["content"][0]["text"])["errorCode"] == "NOT_FOUND"
    browse = asyncio.run(handlers["browse_documents"]({}))
    listed = json.loads(browse["content"][0]["text"])["documents"]
    assert [doc["name"] for doc in listed] == ["report.pdf"]


def test_openai_agent_config_scoped_shadow_check(client, store_path):
    """The bundles' tools resolve names inside the allowlist, so a same-name
    document outside the target set must not block — only an in-set
    duplicate shadows."""
    pytest.importorskip("agents")
    seed_doc(store_path, "pi-old", "report.pdf")
    seed_doc(store_path, "pi-new", "report.pdf",
             created_at="2026-08-02T10:00:00.123000")
    config = client.openai_agent_config(doc_id="pi-old")
    assert "report.pdf" in config["instructions"]
    with pytest.raises(PageIndexAPIError, match="shadowed"):
        client.openai_agent_config(doc_id=["pi-old", "pi-new"])


def test_anthropic_runner_config_scoped_shadow_check(client, store_path):
    pytest.importorskip("anthropic")
    seed_doc(store_path, "pi-old", "report.pdf")
    seed_doc(store_path, "pi-new", "report.pdf",
             created_at="2026-08-02T10:00:00.123000")
    config = client.anthropic_runner_config(model="claude-sonnet-4-5",
                                            doc_id="pi-old")
    assert "report.pdf" in config["system"]


def test_claude_agent_config_scoped_shadow_check(client, store_path):
    pytest.importorskip("claude_agent_sdk")
    seed_doc(store_path, "pi-old", "report.pdf")
    seed_doc(store_path, "pi-new", "report.pdf",
             created_at="2026-08-02T10:00:00.123000")
    config = client.claude_agent_config(doc_id="pi-old")
    assert "report.pdf" in config["system_prompt"]


def test_anthropic_runner_config_thinking_lifts_max_tokens(client):
    pytest.importorskip("anthropic")
    config = client.anthropic_runner_config(
        model="claude-sonnet-4-5",
        thinking={"type": "enabled", "budget_tokens": 10000})
    assert config["max_tokens"] == 10000 + 8192
    assert config["thinking"] == {"type": "enabled", "budget_tokens": 10000}
    assert "thinking" not in client.anthropic_runner_config(
        model="claude-sonnet-4-5")


def test_bridge_invoker_reraises_auth_failures():
    class Revoked:
        def call_tool(self, name, arguments):
            raise PageIndexAPIError("HTTP 401", status_code=401)

    invoke = agent_tools_module._bridge_invoker(Revoked(), "get_document", {})
    with pytest.raises(PageIndexAPIError, match="401"):
        invoke({})
    # Transport blips stay contained in the retryable envelope.

    class Down:
        def call_tool(self, name, arguments):
            raise PageIndexAPIError("HTTP 503", status_code=503)

    text, is_error = agent_tools_module._bridge_invoker(
        Down(), "get_document", {})({})
    assert is_error
    assert json.loads(text)["errorCode"] == "INTERNAL_ERROR"


def test_cloud_bridge_gates_the_endpoint(monkeypatch):
    """Instructions come from the same endpoint the tools register: the
    read-gated URL by default, the full one with include_management."""
    created = []

    class FakeBridge:
        def __init__(self, url, headers):
            created.append(url)

        def instructions(self):
            return "SERVED"

    monkeypatch.setattr("pageindex.mcp_bridge.McpBridge", FakeBridge)
    cloud = PageIndexCloudClient(api_key="pi-k")
    assert cloud.agent_instructions() == "SERVED"
    assert created == [f"{cloud.BASE_URL}/mcp?tools=read"]
    cloud.agent_instructions(include_management=True)
    assert created[1:] == [f"{cloud.BASE_URL}/mcp"]
    cloud.agent_instructions()
    cloud.agent_instructions(include_management=True)
    assert len(created) == 2  # cached per gate


def test_doc_scope_rejected_on_cloud_openai():
    pytest.importorskip("agents")
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="server-side"):
        cloud.as_openai_tools(doc_id="pi-a")
    # The hosted branch returns before _tool_specs — it must reject too,
    # not silently drop the allowlist.
    with pytest.raises(PageIndexAPIError, match="server-side"):
        cloud.as_openai_tools(hosted=True, doc_id="pi-a")


def test_doc_scope_rejected_on_cloud_anthropic():
    pytest.importorskip("anthropic")
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="server-side"):
        cloud.as_anthropic_tools(doc_id="pi-a")


def test_doc_scope_rejected_on_cloud_claude():
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="server-side"):
        cloud.as_claude_mcp(doc_id="pi-a")


def test_as_anthropic_tools_missing_dependency(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(PageIndexAPIError, match="anthropic"):
        client.as_anthropic_tools()


def test_as_anthropic_tools_local_in_process(client, store_path):
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import BetaFunctionTool
    from pageindex.agent_tools import _local_description, _local_schema
    tools = client.as_anthropic_tools()
    # The sync flavor is load-bearing: the sync runner (and messages())
    # rejects async tools and vice versa.
    assert all(isinstance(tool, BetaFunctionTool) for tool in tools)
    assert [tool.name for tool in tools] == list(tool_names())
    browse = {tool.name: tool for tool in tools}["browse_documents"]
    assert browse.input_schema == _local_schema("browse_documents")
    assert browse.description == _local_description("browse_documents")
    seed_doc(store_path, "pi-a", "report.pdf")
    assert "report.pdf" in browse.call({})


def test_as_anthropic_tools_async_flavor(client, store_path):
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import BetaAsyncFunctionTool
    tools = client.as_anthropic_tools(asynchronous=True)
    assert all(isinstance(tool, BetaAsyncFunctionTool) for tool in tools)
    assert [tool.name for tool in tools] == list(tool_names())
    seed_doc(store_path, "pi-a", "report.pdf")
    browse = {tool.name: tool for tool in tools}["browse_documents"]
    assert "report.pdf" in asyncio.run(browse.call({}))


def test_as_anthropic_tools_local_management_opt_in(client):
    pytest.importorskip("anthropic")
    names = [tool.name
             for tool in client.as_anthropic_tools(include_management=True)]
    assert names == list(tool_names(include_management=True))
    assert "remove_document" in names


def test_as_anthropic_tools_local_failures_raise_toolerror(client, store_path):
    """Error envelopes surface as ToolError so the runner marks the
    tool_result is_error: true — a bare return would read as success."""
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import ToolError
    seed_doc(store_path, "pi-a", "report.pdf")
    tools = {tool.name: tool for tool in client.as_anthropic_tools()}
    with pytest.raises(ToolError) as excinfo:
        tools["get_document"].call({"doc_name": "ghost.pdf"})
    assert json.loads(excinfo.value.content)["errorCode"] == "NOT_FOUND"
    assert "report.pdf" in tools["browse_documents"].call({})


def test_as_anthropic_tools_cloud_iserror_raises_toolerror(
        cloud_with_fake_bridge):
    """The server's MCP isError marking must reach the runner's error
    channel, not arrive as a successful tool_result."""
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import ToolError
    cloud, created = cloud_with_fake_bridge
    tools = cloud.as_anthropic_tools()
    created["bridge"].call_tool = lambda name, arguments: (
        '{"error": "denied"}', True)
    with pytest.raises(ToolError) as excinfo:
        tools[0].call({"query": "q"})
    assert json.loads(excinfo.value.content)["error"] == "denied"


def test_as_anthropic_tools_cloud_schemas_pass_through(cloud_with_fake_bridge):
    pytest.importorskip("anthropic")
    cloud, created = cloud_with_fake_bridge
    tools = cloud.as_anthropic_tools()
    assert [tool.name for tool in tools] == ["search_documents", "get_document"]
    bridge = created["bridge"]
    assert tools[0].input_schema == bridge.tools[0]["inputSchema"]
    # Equal but not aliased: beta_tool stores the dict by reference, so the
    # builder must hand out copies of the bridge's cached metas.
    assert tools[0].input_schema is not bridge.tools[0]["inputSchema"]
    assert tools[0].description == bridge.tools[0]["description"]
    # Calls route over the bridge; None-valued arguments mean "omitted".
    out = tools[1].call({"doc_name": "x.pdf", "folder_id": None})
    assert bridge.calls == [("get_document", {"doc_name": "x.pdf"})]
    assert json.loads(out)["success"] is True


def test_as_anthropic_tools_cloud_async_flavor(cloud_with_fake_bridge):
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import BetaAsyncFunctionTool
    cloud, created = cloud_with_fake_bridge
    tools = cloud.as_anthropic_tools(asynchronous=True)
    assert all(isinstance(tool, BetaAsyncFunctionTool) for tool in tools)
    out = asyncio.run(tools[1].call({"doc_name": "x.pdf"}))
    assert created["bridge"].calls == [("get_document", {"doc_name": "x.pdf"})]
    assert json.loads(out)["success"] is True


def test_as_anthropic_tools_cloud_management_opt_in(cloud_with_fake_bridge):
    pytest.importorskip("anthropic")
    cloud, _ = cloud_with_fake_bridge
    names = [tool.name
             for tool in cloud.as_anthropic_tools(include_management=True)]
    assert names == ["search_documents", "get_document",
                     "remove_document", "unannotated_tool"]


def test_as_anthropic_tools_cloud_contains_bridge_errors(cloud_with_fake_bridge):
    """Bridge failures become error envelopes raised as ToolError — the
    runner turns that into a tool_result with is_error: true and the
    envelope as content."""
    pytest.importorskip("anthropic")
    from anthropic.lib.tools import ToolError
    cloud, created = cloud_with_fake_bridge
    tools = cloud.as_anthropic_tools()

    def boom(name, arguments):
        raise RuntimeError("bridge down")

    created["bridge"].call_tool = boom
    with pytest.raises(ToolError) as excinfo:
        tools[0].call({"query": "q"})
    payload = json.loads(excinfo.value.content)
    assert payload["errorCode"] == "INTERNAL_ERROR"
    assert "bridge down" in payload["error"]


def test_agent_tools_work_without_frameworks(client, store_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    monkeypatch.setitem(sys.modules, "anthropic", None)
    seed_doc(store_path, "pi-a", "report.pdf")
    browse = client.agent_tools()[0]
    assert "report.pdf" in browse()


# ── cloud agent_tools: MCP bridge ──

class _FakeBridge:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.calls = []
        read_only = {"readOnlyHint": True, "openWorldHint": False}
        self.tools = [
            {
                "name": "search_documents",
                "description": "ESCALATION tool — keyword search.",
                "annotations": read_only,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keyword query."},
                        "limit": {"type": "number", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_document",
                "description": "Check a document's status.",
                "annotations": read_only,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_name": {"type": "string"},
                        "folder_id": {"anyOf": [{"type": "string"},
                                                {"type": "null"}]},
                    },
                    "required": ["doc_name"],
                },
            },
            {
                "name": "remove_document",
                "description": "Permanently delete documents.",
                "annotations": {"readOnlyHint": False, "destructiveHint": True},
                "inputSchema": {
                    "type": "object",
                    "properties": {"doc_names": {"type": "array"}},
                    "required": ["doc_names"],
                },
            },
            {
                "name": "unannotated_tool",
                "description": "A tool the server sent without annotations.",
                "inputSchema": {"type": "object", "properties": {},
                                "required": []},
            },
        ]

    def list_tools(self):
        # mirrors the live server: the read endpoint serves the read subset
        if "tools=read" in self.url:
            return [tool for tool in self.tools
                    if (tool.get("annotations") or {}).get("readOnlyHint")]
        return self.tools

    def instructions(self):
        return "SERVER GUIDANCE"

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({"success": True, "tool": name,
                           "args": arguments}), False


@pytest.fixture
def cloud_with_fake_bridge(monkeypatch):
    import pageindex.mcp_bridge as mcp_bridge
    created = {}

    def factory(url, headers):
        created["bridge"] = _FakeBridge(url, headers)
        return created["bridge"]

    monkeypatch.setattr(mcp_bridge, "McpBridge", factory)
    return PageIndexCloudClient(api_key="pi-test-key"), created


def test_cloud_agent_tools_discover_live_tool_set(cloud_with_fake_bridge):
    cloud, created = cloud_with_fake_bridge
    tools = cloud.agent_tools()
    bridge = created["bridge"]
    # Default discovery rides the read-gated endpoint, matching the
    # instructions fetch and the hosted/MCP registrations.
    assert bridge.url == "https://api.pageindex.ai/mcp?tools=read"
    assert bridge.headers == {"Authorization": "Bearer pi-test-key"}
    # The endpoint is the only gate: whatever it serves is exposed verbatim.
    assert [t.__name__ for t in tools] == ["search_documents", "get_document"]
    assert "ESCALATION tool" in tools[0].__doc__


def test_cloud_agent_tools_management_gate(cloud_with_fake_bridge):
    cloud, _ = cloud_with_fake_bridge
    names = [t.__name__ for t in cloud.agent_tools(include_management=True)]
    assert names == ["search_documents", "get_document", "remove_document",
                     "unannotated_tool"]


def test_cloud_agent_tools_signatures_from_schema(cloud_with_fake_bridge):
    import inspect
    cloud, _ = cloud_with_fake_bridge
    search, get_document = cloud.agent_tools()
    params = inspect.signature(search).parameters
    assert list(params) == ["query", "limit"]
    assert params["query"].default is inspect.Parameter.empty
    assert params["limit"].default == 10
    assert search.__annotations__["query"] is str
    folder_param = inspect.signature(get_document).parameters["folder_id"]
    assert folder_param.default is None
    # The live server encodes nullables as anyOf; the annotation must still
    # come out Optional[str], not Any.
    from typing import Optional
    assert get_document.__annotations__["folder_id"] == Optional[str]


def test_cloud_agent_tools_proxy_and_drop_none(cloud_with_fake_bridge):
    cloud, created = cloud_with_fake_bridge
    _, get_document = cloud.agent_tools()
    result = json.loads(get_document("report.pdf"))
    assert result["tool"] == "get_document"
    assert result["args"] == {"doc_name": "report.pdf"}  # folder_id=None dropped
    assert created["bridge"].calls == [("get_document", {"doc_name": "report.pdf"})]


def test_cloud_agent_tools_null_description_survives():
    """A server may send description: null — .get(key, default) does not
    apply the default to it, and agent_tools() died with a TypeError while
    the _tool_specs path handled the same payload fine."""

    class _Bridge:
        def call_tool(self, name, arguments):
            return json.dumps({"success": True}), False

    tool = _synth(_Bridge(), {
        "name": "search_documents",
        "description": None,
        "inputSchema": {"type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]},
    })
    assert tool.__name__ == "search_documents"
    assert json.loads(tool(query="x"))["success"] is True


def test_cloud_agent_tools_call_errors_contained(cloud_with_fake_bridge):
    cloud, created = cloud_with_fake_bridge
    search, _ = cloud.agent_tools()
    created["bridge"].call_tool = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("network down"))
    payload = json.loads(search(query="x"))
    assert payload["errorCode"] == "INTERNAL_ERROR"
    assert "network down" in payload["error"]


def test_cloud_agent_tools_list_failure_raises(monkeypatch):
    import pageindex.mcp_bridge as mcp_bridge

    class _DeadBridge:
        def __init__(self, url, headers):
            pass

        def list_tools(self):
            raise PageIndexAPIError("Could not connect")

    monkeypatch.setattr(mcp_bridge, "McpBridge", _DeadBridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="Could not connect"):
        cloud.agent_tools()


def test_local_description_edit_actually_removes_the_image_sentence():
    """The local get_page_content description edits the contract text by
    exact string replace — a contract wording change must fail here, not
    silently ship the image-tool sentence to local models."""
    from pageindex.agent_tools import TOOL_CONTRACT, _LOCAL_DESCRIPTIONS
    local = _LOCAL_DESCRIPTIONS["get_page_content"]
    assert "get_document_image" not in local
    assert len(local) < len(TOOL_CONTRACT["get_page_content"]["description"])


def test_list_tools_pagination_is_bounded():
    """A server echoing its nextCursor terminates (no-progress guard);
    a cycling one hits the page cap instead of hanging forever."""
    from pageindex.mcp_bridge import McpBridge

    bridge = McpBridge("http://x/mcp", {})
    pages = {None: {"tools": [{"name": "a"}], "nextCursor": "c1"},
             "c1": {"tools": [{"name": "b"}], "nextCursor": "c1"}}
    bridge._request = lambda method, params=None: pages[
        (params or {}).get("cursor")]
    assert [t["name"] for t in bridge.list_tools()] == ["a", "b"]

    bridge._request = lambda method, params=None: {
        "tools": [],
        "nextCursor": {"c1": "c2"}.get((params or {}).get("cursor"), "c1")}
    with pytest.raises(PageIndexAPIError, match="did not terminate"):
        bridge.list_tools()


def test_mcp_bridge_protocol(monkeypatch):
    import requests as requests_mod
    from pageindex.mcp_bridge import McpBridge
    import pageindex.mcp_bridge as mcp_bridge

    posts = []

    class _Resp:
        def __init__(self, status, body=None, headers=None, text=""):
            self.status_code = status
            self._body = body
            self.headers = headers or {"Content-Type": "application/json"}
            self.text = text or (json.dumps(body) if body else "")
            self.content = self.text.encode("utf-8")

        def json(self):
            if self._body is None:
                raise ValueError("no body")
            return self._body

    session_alive = {"first": True}

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append({"payload": json, "headers": headers})
        method = json.get("method")
        rid = json.get("id")
        if method == "initialize":
            return _Resp(200, {"jsonrpc": "2.0", "id": rid,
                               "result": {"protocolVersion": "2025-06-18",
                                          "instructions": "SERVER GUIDANCE"}},
                         {"Content-Type": "application/json",
                          "Mcp-Session-Id": "sess-1"})
        if method == "notifications/initialized":
            return _Resp(202)
        if method == "tools/list":
            # SSE-framed response exercises the event-stream parser; the
            # em-dash guards UTF-8 decoding (SSE is UTF-8 by spec).
            body = {"jsonrpc": "2.0", "id": rid,
                    "result": {"tools": [{"name": "t1",
                                          "description": "reads — never writes"}],
                               "nextCursor": None}}
            import json as json_mod
            return _Resp(200, None,
                         {"Content-Type": "text/event-stream"},
                         f"event: message\ndata: {json_mod.dumps(body)}\n\n")
        if method == "tools/call":
            if session_alive["first"]:
                session_alive["first"] = False
                return _Resp(404, text="session expired")
            return _Resp(200, {"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": "hello"},
                            {"type": "text", "text": "world"}]}})
        raise AssertionError(f"unexpected method {method}")

    # Replace the module's own `requests` binding — patching the shared
    # requests module would leak the fake process-wide.
    monkeypatch.setattr(mcp_bridge, "requests", types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=fake_post),
        RequestException=requests_mod.RequestException))
    bridge = McpBridge("https://api.pageindex.ai/mcp",
                       {"Authorization": "Bearer k"})

    tools = bridge.list_tools()
    assert tools == [{"name": "t1", "description": "reads — never writes"}]
    # Captured during the handshake — serving it must not post again.
    posts_before = len(posts)
    assert bridge.instructions() == "SERVER GUIDANCE"
    assert len(posts) == posts_before
    list_headers = posts[-1]["headers"]
    assert list_headers["Mcp-Session-Id"] == "sess-1"
    assert list_headers["MCP-Protocol-Version"] == "2025-06-18"
    assert list_headers["Authorization"] == "Bearer k"

    # First tools/call 404s (expired session) → re-initialize → retry succeeds.
    text, is_error = bridge.call_tool("t1", {"a": 1})
    assert (text, is_error) == ("hello\nworld", False)
    methods = [p["payload"]["method"] for p in posts]
    assert methods.count("initialize") == 2
    # The expired session's negotiated state must not leak into the new
    # handshake.
    reinit = [p for p in posts if p["payload"].get("method") == "initialize"][1]
    assert "MCP-Protocol-Version" not in reinit["headers"]
    assert "Mcp-Session-Id" not in reinit["headers"]


def test_mcp_bridge_400_is_an_error_not_session_expiry(monkeypatch):
    """The spec's expired-session status is 404; a 400 is an ordinary bad
    request — treating it as expiry replayed the rejected call (running a
    management tool's side effect twice) behind a spurious re-initialize."""
    import requests as requests_mod
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex.mcp_bridge import McpBridge

    posts = []

    class _Resp:
        def __init__(self, status, body=None, headers=None, text=""):
            self.status_code = status
            self._body = body
            self.headers = headers or {"Content-Type": "application/json"}
            self.text = text or (json.dumps(body) if body else "")
            self.content = self.text.encode("utf-8")

        def json(self):
            if self._body is None:
                raise ValueError("no body")
            return self._body

    def fake_post(url, json=None, headers=None, timeout=None):
        posts.append(json.get("method"))
        rid = json.get("id")
        if json.get("method") == "initialize":
            return _Resp(200, {"jsonrpc": "2.0", "id": rid,
                               "result": {"protocolVersion": "2025-06-18"}},
                         {"Content-Type": "application/json",
                          "Mcp-Session-Id": "sess-1"})
        if json.get("method") == "notifications/initialized":
            return _Resp(202)
        return _Resp(400, text="unknown tool")

    monkeypatch.setattr(mcp_bridge, "requests", types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=fake_post),
        RequestException=requests_mod.RequestException))
    bridge = McpBridge("https://api.pageindex.ai/mcp",
                       {"Authorization": "Bearer k"})
    with pytest.raises(PageIndexAPIError, match="HTTP 400"):
        bridge.call_tool("nope", {})
    # Exactly one call attempt, no replay, no re-initialize; the live
    # session survives for the next request.
    assert posts.count("tools/call") == 1
    assert posts.count("initialize") == 1
    assert bridge._session_id == "sess-1"


def test_mcp_bridge_init_notification_bars_concurrent_requests(monkeypatch):
    """No thread may send a request between the initialize handshake and
    notifications/initialized — strict servers reject such requests with
    HTTP 400, which the bridge never replays. The notification's fake
    transport stalls to hold that window open; a racing thread would post
    its tools/list inside it."""
    import threading
    import requests as requests_mod
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex.mcp_bridge import McpBridge

    events = []
    events_lock = threading.Lock()
    in_notification = threading.Event()

    class _Resp:
        def __init__(self, status, body=None):
            self.status_code = status
            self._body = body
            self.headers = {"Content-Type": "application/json"}
            self.text = json.dumps(body) if body else ""
            self.content = self.text.encode("utf-8")

        def json(self):
            if self._body is None:
                raise ValueError("no body")
            return self._body

    def fake_post(url, json=None, headers=None, timeout=None):
        method = json.get("method")
        with events_lock:
            events.append(("start", method))
        if method == "notifications/initialized":
            in_notification.set()
            time.sleep(0.2)
        rid = json.get("id")
        if method == "initialize":
            resp = _Resp(200, {"jsonrpc": "2.0", "id": rid,
                               "result": {"protocolVersion": "2025-06-18"}})
        elif method == "notifications/initialized":
            resp = _Resp(202)
        else:
            resp = _Resp(200, {"jsonrpc": "2.0", "id": rid,
                               "result": {"tools": [], "nextCursor": None}})
        with events_lock:
            events.append(("end", method))
        return resp

    monkeypatch.setattr(mcp_bridge, "requests", types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=fake_post),
        RequestException=requests_mod.RequestException))
    bridge = McpBridge("https://api.pageindex.ai/mcp",
                       {"Authorization": "Bearer k"})

    errors = []

    def list_tools():
        try:
            bridge.list_tools()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=list_tools)
    first.start()
    assert in_notification.wait(5), "handshake never reached the notification"
    second = threading.Thread(target=list_tools)
    second.start()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors

    notified = events.index(("end", "notifications/initialized"))
    first_list = events.index(("start", "tools/list"))
    assert notified < first_list, (
        f"tools/list overtook notifications/initialized: {events}")
    assert events.count(("start", "initialize")) == 1


def test_mcp_bridge_blob_blocks_become_stubs():
    """Non-text content used to be json.dumps'd wholesale, handing the
    model the raw base64 payload of an image tool's response."""
    from pageindex.mcp_bridge import McpBridge

    bridge = McpBridge("https://api.pageindex.ai/mcp", {})
    blob = "A" * 8192  # ~6 KB decoded
    bridge._request = lambda method, params: {"content": [
        {"type": "text", "text": "Page 3 of report.pdf"},
        {"type": "image", "mimeType": "image/png", "data": blob},
        {"type": "resource",
         "resource": {"mimeType": "image/jpeg", "blob": blob}},
    ]}
    text, is_error = bridge.call_tool("get_document_image", {})
    assert not is_error
    assert "Page 3 of report.pdf" in text
    assert "AAAA" not in text
    assert "[image/png content omitted: ~6 KB]" in text
    assert "[image/jpeg content omitted: ~6 KB]" in text


# ── review-round regressions ──

def _synth(bridge, meta):
    """Build a tool function the way the cloud lane does: signature
    synthesis over a bridge invoker."""
    from pageindex.agent_tools import _bridge_invoker, _make_tool_function
    name = meta["name"]
    return _make_tool_function(name, meta.get("description"),
                               meta["inputSchema"],
                               _bridge_invoker(bridge, name,
                                               meta["inputSchema"]))


def test_synth_binding_error_names_the_tool():
    """Binding TypeErrors quote the function's __qualname__; the model used
    to see \"_synthesized() got an unexpected keyword argument\" and had no
    tool name to correct against."""
    from pageindex.agent_tools import TOOL_CONTRACT

    class _Bridge:
        def call_tool(self, name, args):
            return json.dumps(args), False

    meta = {"name": "browse_documents", "description": "d",
            "inputSchema": TOOL_CONTRACT["browse_documents"]["schema"]}
    fn = _synth(_Bridge(), meta)
    payload = json.loads(fn(bogus_param=1))
    assert "browse_documents" in payload["error"]
    assert "_synthesized" not in payload["error"]


def test_bridge_invoker_coerces_string_booleans():
    """Identical model output must behave the same on both dispatch paths:
    call_tool coerced "false" but the cloud bridge forwarded it verbatim,
    turning "don't wait" into a 3-minute wait on lenient servers."""
    from pageindex.agent_tools import TOOL_CONTRACT, _bridge_invoker

    seen = {}

    class _Bridge:
        def call_tool(self, name, args):
            seen.update(args)
            return "{}", False

    invoke = _bridge_invoker(_Bridge(), "get_document",
                             TOOL_CONTRACT["get_document"]["schema"])
    invoke({"doc_name": "q.pdf", "wait_for_completion": "false"})
    assert seen["wait_for_completion"] is False


def test_synth_optional_no_default_param_is_nullable():
    """A non-required, no-default schema param must annotate Optional, or
    strict schemas force the model to always send a value (browse.query)."""
    from pageindex.agent_tools import TOOL_CONTRACT
    from typing import get_args

    class _Bridge:
        def call_tool(self, name, args):
            return json.dumps(args), False

    meta = {"name": "browse_documents",
            "description": "d",
            "inputSchema": TOOL_CONTRACT["browse_documents"]["schema"]}
    fn = _synth(_Bridge(), meta)
    assert type(None) in get_args(fn.__annotations__["query"])


def test_synth_array_params_keep_their_item_type():
    """The schema→annotation round-trip flattened arrays to bare `list`;
    function_tool then emits {"type": "array", "items": {}}, which strict
    function calling rejects."""
    from typing import Optional

    class _Bridge:
        def call_tool(self, name, args):
            return json.dumps(args), False

    meta = {"name": "remove_documents", "description": "d",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "doc_ids": {"type": "array", "items": {"type": "string"}},
                    "tags": {"anyOf": [{"type": "array",
                                        "items": {"type": "integer"}},
                                       {"type": "null"}]},
                    "mixed": {"type": "array",
                              "items": {"type": ["string", "null"]}},
                },
                "required": ["doc_ids", "mixed"],
            }}
    fn = _synth(_Bridge(), meta)
    assert fn.__annotations__["doc_ids"] == list[str]
    assert fn.__annotations__["tags"] == Optional[list[int]]
    # A type-array in items (nullable elements) degrades to bare list —
    # it must not crash the build on an unhashable dict key.
    assert fn.__annotations__["mixed"] == list


def test_synth_escape_hatches():

    calls = []

    class _Bridge:
        def call_tool(self, name, args):
            calls.append((name, args))
            return "ok", False

    # Tool named "_invoke" must not recurse into itself.
    invoke_named = _synth(_Bridge(), {
        "name": "_invoke", "description": "d",
        "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}},
                        "required": ["x"]}})
    assert invoke_named("v") == "ok"
    assert calls[-1] == ("_invoke", {"x": "v"})

    # Param named "dict" must not shadow the builtin.
    dict_param = _synth(_Bridge(), {
        "name": "t", "description": "d",
        "inputSchema": {"type": "object", "properties": {"dict": {"type": "string"}},
                        "required": ["dict"]}})
    assert dict_param("v") == "ok"
    assert calls[-1] == ("t", {"dict": "v"})

    # Non-identifier tool name still gets a real signature.
    import inspect
    dashed = _synth(_Bridge(), {
        "name": "page-content.v2", "description": "d",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}},
                        "required": ["a"]}})
    assert dashed.__name__ == "page-content.v2"
    assert list(inspect.signature(dashed).parameters) == ["a"]
    assert dashed("v") == "ok"


def test_annotation_for_both_nullable_encodings():
    """Servers have emitted nullables as type-arrays and as anyOf unions;
    both must map to Optional, not degrade to Any."""
    from typing import Optional
    from pageindex.agent_tools import _annotation_for
    assert _annotation_for({"type": "string"}) is str
    assert _annotation_for({"type": ["string", "null"]}) == Optional[str]
    assert (_annotation_for({"anyOf": [{"type": "string"}, {"type": "null"}]})
            == Optional[str])


def test_cloud_agent_tools_trust_the_gated_endpoint(monkeypatch):
    """The ?tools=read endpoint is the only gate: what it serves is exposed
    verbatim, with no client-side annotation second-guessing."""
    import pageindex.mcp_bridge as mcp_bridge

    class _AllWriteBridge:
        def __init__(self, url, headers):
            pass

        def list_tools(self):
            return [{"name": "remove_document",
                     "annotations": {"readOnlyHint": False},
                     "inputSchema": {"type": "object", "properties": {}}}]

    monkeypatch.setattr(mcp_bridge, "McpBridge", _AllWriteBridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    assert len(cloud.agent_tools()) == 1
    assert len(cloud.agent_tools(include_management=True)) == 1


def test_extract_result_skips_non_dict_messages():
    """A 200 body of null, a batched array, or an SSE string frame must
    surface as the contract's PageIndexAPIError, not an AttributeError."""
    from pageindex.mcp_bridge import McpBridge

    bridge = McpBridge("http://unused", {})

    class _Resp:
        def __init__(self, text, content_type="application/json"):
            self.headers = {"Content-Type": content_type}
            self.status_code = 200
            self.content = text.encode("utf-8")

        def json(self):
            return json.loads(self.content)

    for body in ("null", '[{"jsonrpc": "2.0", "id": 1, "result": {}}]',
                 '"hello"'):
        with pytest.raises(PageIndexAPIError, match="no reply matching"):
            bridge._extract_result(_Resp(body), 1)

    sse = ('data: "noise"\n\n'
           'data: {"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n\n')
    result = bridge._extract_result(_Resp(sse, "text/event-stream"), 1)
    assert result == {"ok": True}


def test_bridge_call_tool_surfaces_iserror(monkeypatch):
    import requests as requests_mod
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex.mcp_bridge import McpBridge

    class _Resp:
        def __init__(self, status, body=None):
            self.status_code = status
            self._body = body
            self.headers = {"Content-Type": "application/json"}
            self.text = json.dumps(body) if body else ""
            self.content = self.text.encode("utf-8")

        def json(self):
            if self._body is None:
                raise ValueError("no body")
            return self._body

    def fake_post(url, json=None, headers=None, timeout=None):
        method = json.get("method")
        rid = json.get("id")
        if method == "initialize":
            return _Resp(200, {"jsonrpc": "2.0", "id": rid, "result": {}})
        if method == "notifications/initialized":
            return _Resp(202)
        return _Resp(200, {"jsonrpc": "2.0", "id": rid, "result": {
            "isError": True,
            "content": [{"type": "text", "text": '{"error": "denied"}'}]}})

    monkeypatch.setattr(mcp_bridge, "requests", types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=fake_post),
        RequestException=requests_mod.RequestException))
    bridge = McpBridge("https://api.pageindex.ai/mcp", {})
    assert bridge.call_tool("t", {}) == ('{"error": "denied"}', True)


def test_bridge_rejects_mismatched_reply_id(monkeypatch):
    """A result-bearing message with the wrong id must not be returned as
    this call's reply."""
    import requests as requests_mod
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex.mcp_bridge import McpBridge

    class _Resp:
        def __init__(self, status, body=None):
            self.status_code = status
            self._body = body
            self.headers = {"Content-Type": "application/json"}
            self.text = json.dumps(body) if body else ""
            self.content = self.text.encode("utf-8")

        def json(self):
            if self._body is None:
                raise ValueError("no body")
            return self._body

    def fake_post(url, json=None, headers=None, timeout=None):
        method = json.get("method")
        rid = json.get("id")
        if method == "initialize":
            return _Resp(200, {"jsonrpc": "2.0", "id": rid, "result": {}})
        if method == "notifications/initialized":
            return _Resp(202)
        return _Resp(200, {"jsonrpc": "2.0", "id": rid - 1,  # stale reply
                           "result": {"content": [{"type": "text",
                                                   "text": "old"}]}})

    monkeypatch.setattr(mcp_bridge, "requests", types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=fake_post),
        RequestException=requests_mod.RequestException))
    bridge = McpBridge("https://api.pageindex.ai/mcp", {})
    with pytest.raises(PageIndexAPIError, match="no reply matching"):
        bridge.call_tool("t", {})


def test_sse_crlf_multi_message():
    from pageindex.mcp_bridge import _parse_sse
    body = ('event: message\r\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\r\n\r\n'
            'event: message\r\ndata: {"jsonrpc":"2.0","id":7,"result":{"ok":true}}\r\n\r\n')
    messages = _parse_sse(body)
    assert len(messages) == 2
    assert messages[1]["result"] == {"ok": True}


def test_bridge_transport_error_is_pageindex_error(monkeypatch):
    import requests as requests_mod
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex.mcp_bridge import McpBridge

    def dead_post(*args, **kwargs):
        raise requests_mod.ConnectionError("dns down")

    monkeypatch.setattr(mcp_bridge, "requests", types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(post=dead_post),
        RequestException=requests_mod.RequestException))
    bridge = McpBridge("https://api.pageindex.ai/mcp", {})
    with pytest.raises(PageIndexAPIError, match="Could not reach"):
        bridge.list_tools()


def test_await_completion_preserves_metadata_over_null_refetch(monkeypatch):
    """A status refetch that nulls out metadata must not clobber the
    listing's copy (setdefault is a no-op on an existing None value)."""
    import pageindex.agent_tools as agent_tools_mod
    monkeypatch.setattr(agent_tools_mod, "time", types.SimpleNamespace(
        monotonic=time.monotonic, sleep=lambda seconds: None))

    class _Client:
        def get_document(self, doc_id):
            return {"id": doc_id, "status": "completed", "metadata": None}

    entry = {"id": "pi-x", "status": "processing",
             "metadata": {"team": "research"}}
    merged = agent_tools_mod._await_completion(_Client(), entry, True)
    assert merged["status"] == "completed"
    assert merged["metadata"] == {"team": "research"}


def test_browse_time_sort_uses_native_pagination(client, store_path, monkeypatch):
    """Time-sorted browsing must page through list_documents directly, not
    fetch the whole library to slice one window."""
    for index in range(3):
        seed_doc(store_path, f"pi-{index}", f"doc{index}.pdf",
                 created_at=f"2026-08-0{index + 1}T10:00:00.000000")
    calls = []
    original = client.list_documents

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(client, "list_documents", spy)
    payload, is_error = run(client, "browse_documents", limit=2)
    assert not is_error
    assert calls == [{"limit": 2, "offset": 0}]
    assert [d["name"] for d in payload["documents"]] == ["doc2.pdf", "doc1.pdf"]
    assert payload["has_more"] is True and payload["next_offset"] == 2


def test_all_documents_survives_short_pages_and_missing_total():
    """The full-library walk behind every name resolution must trust what
    actually arrives: a server capping page size, omitting `total`, or
    sending total: null silently truncated the library (or raised)."""
    from pageindex.agent_tools import _all_documents

    docs = [{"id": f"pi-{index}"} for index in range(120)]

    def make_client(total_field, page_cap):
        class _Client:
            calls = 0

            def list_documents(self, limit, offset):
                type(self).calls += 1
                page = {"documents": docs[offset:offset + min(limit,
                                                              page_cap)]}
                if total_field != "omit":
                    page["total"] = total_field
                return page
        return _Client()

    assert _all_documents(make_client(120, 50)) == docs     # short pages
    assert _all_documents(make_client("omit", 100)) == docs  # no total
    assert _all_documents(make_client(None, 100)) == docs   # total: null
    exact = make_client(120, 100)   # well-behaved server:
    assert _all_documents(exact) == docs
    assert type(exact).calls == 2   # ...total still saves the empty page

    # stop_ids ends the walk once every wanted id has been seen — a
    # doc-scoped chat turn must not page the whole library...
    early = make_client(120, 100)
    listed = _all_documents(early, stop_ids=frozenset({"pi-3"}))
    assert type(early).calls == 1
    assert any(doc["id"] == "pi-3" for doc in listed)
    # ...while an id the listing lacks still costs the full sweep.
    full = make_client(120, 100)
    assert _all_documents(full, stop_ids=frozenset({"pi-missing"})) == docs
    assert type(full).calls == 2


def test_null_arguments_mean_omitted(client, store_path):
    """Adapters that forward the model's null values verbatim (the Claude
    MCP handler) used to trip parameter validation — None ≡ omitted is
    enforced once, in call_tool."""
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "browse_documents", folder_id=None,
                            sort=None, query=None)
    assert not is_error
    assert [doc["name"] for doc in payload["documents"]] == ["report.pdf"]


def test_page_spec_span_bomb_rejected(client, store_path):
    """An absurd range must be rejected arithmetically, not expanded into
    billions of integers in the caller's process."""
    seed_doc(store_path, "pi-a", "report.pdf")
    payload, is_error = run(client, "get_page_content", doc_name="report.pdf",
                            pages="1-1000000000")
    assert is_error and payload["errorCode"] == "INVALID_INPUT"
    assert "Too many pages" in payload["error"]


def test_wait_tolerates_transient_network_failures(fake_cloud_client, monkeypatch):
    import requests as requests_mod
    cloud = fake_cloud_client(["processing", "completed"])
    original = cloud._api.get_document
    state = {"raised": False}

    def flaky(doc_id):
        if not state["raised"]:
            state["raised"] = True
            raise requests_mod.ConnectionError("network blip")
        return original(doc_id)

    monkeypatch.setattr(cloud._api, "get_document", flaky)
    assert cloud.submit_document("x.pdf", wait=True) == {"doc_id": "pi-fake"}


def test_failed_document_status_message(client, store_path):
    seed_doc(store_path, "pi-a", "broken.pdf")
    import pageindex.agent_tools as agent_tools_mod
    payload, is_error = agent_tools_mod._not_ready_error(
        "broken.pdf", "failed", "structure retrieval", timed_out=False)
    assert is_error
    assert "failed" in payload["error"]
    assert any("submit_document" in option
               for option in payload["next_steps"]["options"])


def test_hosted_gate_is_the_endpoint():
    """The URL is the gate on hosted mode too — no approval-flow gating,
    the read-only endpoint simply has no write tools."""
    pytest.importorskip("agents")
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    gated = cloud.as_openai_tools(hosted=True)[0].tool_config
    assert gated["server_url"] == "https://api.pageindex.ai/mcp?tools=read"
    assert gated["require_approval"] == "never"
    open_config = cloud.as_openai_tools(hosted=True,
                                        include_management=True)[0].tool_config
    assert open_config["server_url"] == "https://api.pageindex.ai/mcp"
    assert open_config["require_approval"] == "never"


def test_wait_tolerates_transient_poll_failures(fake_cloud_client, monkeypatch):
    cloud = fake_cloud_client(["processing", "completed"])
    original = cloud._api.get_document
    state = {"raised": False}

    def flaky(doc_id):
        if not state["raised"]:
            state["raised"] = True
            raise PageIndexAPIError("502")
        return original(doc_id)

    monkeypatch.setattr(cloud._api, "get_document", flaky)
    assert cloud.submit_document("x.pdf", wait=True) == {"doc_id": "pi-fake"}


import pageindex.utils  # noqa: F401 — its import loads .env
LIVE_KEY = os.getenv("PAGEINDEX_API_KEY")


@pytest.mark.skipif(not LIVE_KEY, reason="PAGEINDEX_API_KEY not set")
def test_live_cloud_contract_parity():
    """Real-drift detector: the frozen contract must match the live server
    on every shared tool, including the annotations the gates rely on."""
    from pageindex.mcp_bridge import McpBridge
    bridge = McpBridge("https://api.pageindex.ai/mcp",
                       {"Authorization": f"Bearer {LIVE_KEY}"})
    live = {t["name"]: t for t in bridge.list_tools()}
    for name, ours in TOOL_CONTRACT.items():
        real = live.get(name)
        assert real is not None, f"{name} missing from live tools/list"
        assert real.get("description") == ours["description"], name
        real_schema = real.get("inputSchema") or {}
        real_props = real_schema.get("properties") or {}
        assert set(real_props) == set(ours["schema"]["properties"]), name
        # Full per-param equality: a drifted type, default, enum, or bound
        # breaks calls just as surely as a renamed parameter.
        for param, spec in ours["schema"]["properties"].items():
            assert real_props[param] == spec, (name, param)
        assert (sorted(real_schema.get("required") or [])
                == sorted(ours["schema"].get("required", []))), name
        for key, value in (ours.get("annotations") or {}).items():
            assert (real.get("annotations") or {}).get(key) == value, (name, key)


@pytest.mark.skipif(not LIVE_KEY, reason="PAGEINDEX_API_KEY not set")
def test_live_cloud_envelope_field_parity(tmp_path):
    """Response-envelope drift alarm: every field the local tools emit must
    exist in the live cloud tool's response for the analogous call — a cloud
    rename of a shared field (has_more, next_offset, content, ...) fails
    here. Guidance wording is deliberately localized and not compared."""
    from pageindex.mcp_bridge import McpBridge
    bridge = McpBridge("https://api.pageindex.ai/mcp",
                       {"Authorization": f"Bearer {LIVE_KEY}"})
    cloud_browse = json.loads(
        bridge.call_tool("browse_documents", {"limit": 2})[0])
    assert cloud_browse.get("success") is True and cloud_browse["documents"]
    doc_name = cloud_browse["documents"][0]["name"]
    cloud = {
        "browse_documents": cloud_browse,
        "get_document": json.loads(bridge.call_tool(
            "get_document", {"doc_name": doc_name})[0]),
        "get_document_structure": json.loads(bridge.call_tool(
            "get_document_structure", {"doc_name": doc_name})[0]),
        "get_page_content": json.loads(bridge.call_tool(
            "get_page_content", {"doc_name": doc_name, "pages": "1"})[0]),
    }

    store = str(tmp_path / "store")
    local_client = PageIndexLocalClient(storage_path=store)
    seed_doc(store, "pi-parity", "parity.pdf")
    local = {
        "browse_documents": run(local_client, "browse_documents")[0],
        "get_document": run(local_client, "get_document",
                            doc_name="parity.pdf")[0],
        "get_document_structure": run(local_client, "get_document_structure",
                                      doc_name="parity.pdf")[0],
        "get_page_content": run(local_client, "get_page_content",
                                doc_name="parity.pdf", pages="1")[0],
    }

    for name in cloud:
        assert cloud[name].get("success") is True, name
        missing = set(local[name]) - set(cloud[name])
        assert not missing, (name, missing)
        assert (set(local[name]["next_steps"])
                <= set(cloud[name]["next_steps"]) | {"auto_retry"}), name

    local_doc = local["browse_documents"]["documents"][0]
    cloud_doc = cloud_browse["documents"][0]
    assert set(local_doc) - set(cloud_doc) <= {"metadata"}

    local_nodes = local["get_document_structure"]["structure"]
    cloud_nodes = cloud["get_document_structure"]["structure"]
    local_node = local_nodes[0] if isinstance(local_nodes, list) else local_nodes
    cloud_node = cloud_nodes[0] if isinstance(cloud_nodes, list) else cloud_nodes
    assert (set(local_node)
            <= set(cloud_node) | {"page_index", "prefix_summary"})

    assert (set(local["get_page_content"]["content"][0])
            <= set(cloud["get_page_content"]["content"][0]))


@pytest.mark.skipif(not LIVE_KEY, reason="PAGEINDEX_API_KEY not set")
def test_live_cloud_instructions_nonempty():
    """The empty-instructions guard raises for cloud clients; the real
    server must actually serve instructions in its initialize result."""
    from pageindex.mcp_bridge import McpBridge
    bridge = McpBridge("https://api.pageindex.ai/mcp",
                       {"Authorization": f"Bearer {LIVE_KEY}"})
    assert bridge.instructions()


# ── agent_instructions ──

def test_agent_instructions_default(client):
    text = client.agent_instructions()
    assert text == AGENT_INSTRUCTIONS
    assert "READING WORKFLOW" in text
    assert "browse_documents" in text
    assert "search_documents" not in text
    assert "get_folder_structure" not in text
    assert 'sort="relevance"' not in text  # cloud-side capability


def test_agent_instructions_with_doc_id(client, store_path):
    seed_doc(store_path, "pi-a", "report.pdf")
    text = client.agent_instructions(doc_id="pi-a")
    assert text.startswith(AGENT_INSTRUCTIONS)
    assert "The user has specified document: report.pdf" in text

    seed_doc(store_path, "pi-b", "other.pdf")
    multi = client.agent_instructions(doc_id=["pi-a", "pi-b"])
    assert "The user has specified documents: report.pdf, other.pdf" in multi

    with pytest.raises(PageIndexAPIError):
        client.agent_instructions(doc_id="pi-missing")


def test_local_instructions_name_only_local_tools():
    """The local instructions are trimmed from the cloud server's; every
    tool they name must exist in the local registry, or the trim drifted."""
    named = set(re.findall(r"\b(\w+)\(", AGENT_INSTRUCTIONS))
    assert named
    assert named <= set(tool_names(include_management=True))


def test_cloud_agent_instructions_served_live(monkeypatch):
    """Cloud clients serve the server's live instructions from the MCP
    initialize handshake — over the same bridge session as agent_tools()."""
    import pageindex.mcp_bridge as mcp_bridge
    created = []

    class _Bridge(_FakeBridge):
        def __init__(self, url, headers):
            super().__init__(url, headers)
            created.append(self)

        def instructions(self):
            return "LIVE CLOUD GUIDANCE"

    monkeypatch.setattr(mcp_bridge, "McpBridge", _Bridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    cloud.agent_tools()
    assert cloud.agent_instructions() == "LIVE CLOUD GUIDANCE"
    assert len(created) == 1


def test_cloud_bridge_cache_threadsafe_and_pickle_clean(monkeypatch):
    """One bridge per client even under concurrent first calls, and the
    bridge lives off the instance so cloud clients stay picklable."""
    import pickle
    import threading
    import time as time_mod
    import pageindex.mcp_bridge as mcp_bridge
    created = []

    class _Bridge(_FakeBridge):
        def __init__(self, url, headers):
            time_mod.sleep(0.01)  # widen the construction window
            super().__init__(url, headers)
            created.append(self)

        def instructions(self):
            return "LIVE"

    monkeypatch.setattr(mcp_bridge, "McpBridge", _Bridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    workers = ([threading.Thread(target=cloud.agent_tools) for _ in range(4)]
               + [threading.Thread(target=cloud.agent_instructions)
                  for _ in range(4)])
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(created) == 1
    pickle.dumps(cloud)


def test_cloud_bridge_rebuilds_on_credential_change(monkeypatch):
    """CloudAPI re-reads client.api_key on every REST call; the MCP half
    must not keep authenticating with a rotation-stale snapshot."""
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex.agent_tools import _cloud_bridge
    built = []

    class _Bridge(_FakeBridge):
        def __init__(self, url, headers):
            super().__init__(url, headers)
            built.append((url, dict(headers)))

    monkeypatch.setattr(mcp_bridge, "McpBridge", _Bridge)
    cloud = PageIndexCloudClient(api_key="pi-old")
    first = _cloud_bridge(cloud)
    assert _cloud_bridge(cloud) is first  # unchanged credentials: cached
    cloud.api_key = "pi-new"
    second = _cloud_bridge(cloud)
    assert second is not first
    assert built[-1][1]["Authorization"] == "Bearer pi-new"
    cloud.BASE_URL = "https://alt.example"
    assert _cloud_bridge(cloud) is not second
    assert built[-1][0] == "https://alt.example/mcp"


def test_cloud_agent_instructions_blank_or_nonstring_raises(monkeypatch):
    """Whitespace-only or non-string initialize.instructions must hit the
    same honest error as a missing one — never a blank system prompt."""
    import pageindex.mcp_bridge as mcp_bridge

    for bad in ("   \n\t  ", {"not": "a string"}):
        class _SilentBridge:
            def __init__(self, url, headers):
                pass

            def instructions(self, _value=bad):
                return _value

        monkeypatch.setattr(mcp_bridge, "McpBridge", _SilentBridge)
        cloud = PageIndexCloudClient(api_key="pi-test-key")
        with pytest.raises(PageIndexAPIError, match="no agent instructions"):
            cloud.agent_instructions()


def test_cloud_agent_instructions_empty_raises(monkeypatch):
    """An empty server response must raise, not silently substitute the
    subset guidance — same posture as the annotation-regression guard."""
    import pageindex.mcp_bridge as mcp_bridge

    class _SilentBridge:
        def __init__(self, url, headers):
            pass

        def instructions(self):
            return None

    monkeypatch.setattr(mcp_bridge, "McpBridge", _SilentBridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="no agent instructions"):
        cloud.agent_instructions()


# ── submit_document(wait=True) ──

class _FakeCloudAPI:
    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.polls = 0

    def submit_document(self, **kwargs):
        return {"doc_id": "pi-fake"}

    def get_document(self, doc_id):
        self.polls += 1
        status = (self._statuses.pop(0) if len(self._statuses) > 1
                  else self._statuses[0])
        return {"id": doc_id, "status": status}


@pytest.fixture
def fake_cloud_client(tmp_path, monkeypatch):
    monkeypatch.setattr(client_module, "time", types.SimpleNamespace(
        monotonic=time.monotonic, sleep=lambda seconds: None))

    def build(statuses):
        cloud = PageIndexLocalClient(storage_path=str(tmp_path / "unused"))
        cloud._api = _FakeCloudAPI(statuses)
        return cloud
    return build


def test_submit_wait_polls_until_completed(fake_cloud_client):
    cloud = fake_cloud_client(["processing", "processing", "completed"])
    result = cloud.submit_document("whatever.pdf", wait=True)
    assert result == {"doc_id": "pi-fake"}
    assert cloud._api.polls == 3


def test_submit_wait_raises_on_failed(fake_cloud_client):
    cloud = fake_cloud_client(["processing", "failed"])
    with pytest.raises(PageIndexAPIError, match="failed"):
        cloud.submit_document("whatever.pdf", wait=True)


def test_submit_wait_times_out(fake_cloud_client, monkeypatch):
    clock = {"now": 0.0}

    def fake_monotonic():
        clock["now"] += 700.0
        return clock["now"]

    monkeypatch.setattr(client_module, "time", types.SimpleNamespace(
        monotonic=fake_monotonic, sleep=lambda seconds: None))
    cloud = fake_cloud_client(["processing"])
    with pytest.raises(PageIndexAPIError, match="Timed out"):
        cloud.submit_document("whatever.pdf", wait=True)


def test_submit_without_wait_does_not_poll(fake_cloud_client):
    cloud = fake_cloud_client(["processing"])
    assert cloud.submit_document("whatever.pdf") == {"doc_id": "pi-fake"}
    assert cloud._api.polls == 0


def test_submit_warns_when_stored_name_differs(fake_cloud_client):
    cloud = fake_cloud_client(["processing"])
    cloud._api.submit_document = lambda **kwargs: {
        "doc_id": "pi-fake", "name": "whatever_1.pdf"}
    with pytest.warns(UserWarning, match='stored as "whatever_1.pdf"'):
        result = cloud.submit_document("docs/whatever.pdf")
    assert result["name"] == "whatever_1.pdf"


def test_submit_wait_poll_error_carries_doc_id(fake_cloud_client, monkeypatch):
    """A poll that dies on transient errors must keep the uploaded doc_id
    recoverable, like the timeout and failed branches do."""
    cloud = fake_cloud_client(["processing"])

    def boom(doc_id):
        raise PageIndexAPIError("Failed to get document metadata: 502")

    monkeypatch.setattr(cloud, "get_document", boom)
    with pytest.raises(PageIndexAPIError, match="pi-fake"):
        cloud.submit_document("whatever.pdf", wait=True)


def test_submit_wait_reraises_definite_poll_answers(fake_cloud_client,
                                                    monkeypatch):
    """A 401/403/404 poll answer is final: re-raised untouched, no retries, no keep-polling advice."""
    cloud = fake_cloud_client(["processing"])
    polls = {"n": 0}

    def denied(doc_id):
        polls["n"] += 1
        raise PageIndexAPIError("Failed to get document metadata: 401",
                                status_code=401)

    monkeypatch.setattr(cloud, "get_document", denied)
    with pytest.raises(PageIndexAPIError, match="401") as err:
        cloud.submit_document("whatever.pdf", wait=True)
    assert polls["n"] == 1
    assert "Processing continues" not in str(err.value)


def test_config_helpers_reject_empty_doc_id_on_cloud():
    """An explicitly empty scope must not silently widen to the whole
    library — cloud has no tool-layer allowlist to enforce it."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="doc_id is empty"):
        cloud.openai_agent_config(doc_id=[])
    with pytest.raises(PageIndexAPIError, match="doc_id is empty"):
        cloud.anthropic_runner_config(model="claude-sonnet-4-5", doc_id=[])
    with pytest.raises(PageIndexAPIError, match="doc_id is empty"):
        cloud.claude_agent_config(doc_id=[])


def test_doc_targeting_keeps_transport_errors_out_of_not_found():
    """A cloud 429/5xx during the doc_id fetch is an outage, not a missing
    document — only a definite not-found/denied batches into the
    "Documents not found" message; anything else propagates raw."""
    class Stub:
        def __init__(self, status):
            self.status = status

        def get_document(self, doc_id):
            raise PageIndexAPIError(
                f"Failed to get document metadata: {self.status}",
                status_code=self.status)

    for status in (429, 500):
        with pytest.raises(PageIndexAPIError, match=f"metadata: {status}"):
            agent_tools_module.doc_targeting_block(Stub(status), "pi-a")
    for status in (403, 404):
        with pytest.raises(PageIndexAPIError,
                           match="Documents not found or access denied: pi-a"):
            agent_tools_module.doc_targeting_block(Stub(status), "pi-a")


def test_call_tool_coerces_string_booleans(client, store_path, monkeypatch):
    """Models routinely send booleans as JSON strings — "false" must not
    read as True (a full wait_for_completion stall)."""
    seed_doc(store_path, "pi-1", "a.pdf")
    seen = {}
    real = agent_tools_module._await_completion

    def spy(spy_client, entry, wait):
        seen["wait"] = wait
        return real(spy_client, entry, wait)

    monkeypatch.setattr(agent_tools_module, "_await_completion", spy)
    run(client, "get_document", doc_name="a.pdf", wait_for_completion="false")
    assert seen["wait"] is False
    run(client, "get_document", doc_name="a.pdf", wait_for_completion="true")
    assert seen["wait"] is True


def test_call_tool_rejects_non_object_arguments(client):
    """A non-dict arguments value must come back as the guided envelope,
    never raise into the agent loop."""
    for bad in ([1, 2], "doc_name=a.pdf"):
        text, is_error = call_tool(client, "browse_documents", bad)
        payload = json.loads(text)
        assert is_error and payload["errorCode"] == "INVALID_INPUT"


def test_remove_document_repeated_name_deletes_once(client, store_path):
    seed_doc(store_path, "pi-1", "a.pdf")
    payload, is_error = run(client, "remove_document",
                            doc_names=["a.pdf", "a.pdf"])
    assert not is_error
    assert payload["results"] == [{"doc_name": "a.pdf", "status": "deleted"}]
    assert "1 of 1" in payload["next_steps"]["summary"]


def test_page_spec_cap_counts_distinct_pages():
    """Overlapping parts are normal tree output (a parent section plus its
    children) — the cap is on the union, not the sum."""
    pages, error = agent_tools_module._parse_page_spec("1-5000,2000-9000",
                                                       "a.pdf")
    assert error is None and pages is not None and len(pages) == 9000
    pages, error = agent_tools_module._parse_page_spec("1-10001", "a.pdf")
    assert pages is None and "Too many pages" in error[0]["error"]


def test_browse_documents_pages_by_rows_returned():
    """A backend that caps its page size must not make the cursor skip
    documents, and a null total must not crash (same guards as
    _all_documents)."""
    class _Capping:
        def list_documents(self, limit, offset):
            docs = [{"id": f"pi-{i}", "name": f"d{i}.pdf",
                     "status": "completed"}
                    for i in range(offset, min(offset + 5, 30))]
            return {"documents": docs, "total": 30}

    payload, is_error = agent_tools_module._browse_documents(_Capping(),
                                                             limit=10)
    assert not is_error
    assert payload["has_more"] is True and payload["next_offset"] == 5

    class _NullTotal:
        def list_documents(self, limit, offset):
            return {"documents": [{"name": "d.pdf", "status": "completed"}],
                    "total": None}

    payload, is_error = agent_tools_module._browse_documents(_NullTotal(),
                                                             limit=10)
    assert not is_error
    assert payload["has_more"] is False and payload["next_offset"] is None


def test_agent_instructions_carry_user_metadata(client, store_path):
    """The targeting block promises names and metadata; local get_document
    keeps the 7-key detail wire shape, so the tags come from the listing."""
    seed_doc(store_path, "pi-1", "report.pdf",
             metadata={"quarter": "Q3", "year": 2025})
    text = client.agent_instructions(doc_id="pi-1")
    assert '"quarter": "Q3"' in text and '"year": 2025' in text


# ── wait-poll resilience, instruction scoping, agent_tools doc_id ──

def test_await_completion_polls_through_transient_refetch_failure(monkeypatch):
    """A refetch that fails once must not end the wait early — the caller
    would report that 5-second exit as the full 3-minute timeout."""
    monkeypatch.setattr(agent_tools_module, "_TOOL_WAIT_INTERVAL", 0.0)
    calls = {"n": 0}

    class Flaky:
        def get_document(self, doc_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PageIndexAPIError("transient listing failure")
            return {"id": doc_id, "status": "completed"}

    result = agent_tools_module._await_completion(
        Flaky(), {"id": "pi-x", "status": "processing"}, wait=True)
    assert result["status"] == "completed"
    assert calls["n"] == 2


def test_agent_instructions_doc_id_shadow_check(client, store_path):
    seed_doc(store_path, "pi-old", "report.pdf",
             created_at="2026-08-01T10:00:00.123000")
    seed_doc(store_path, "pi-new", "report.pdf",
             created_at="2026-08-05T10:00:00.123000")
    # standalone instructions get the strict check; only the *_agent_config
    # bundles (which build the tools too) relax it
    with pytest.raises(PageIndexAPIError, match="shadowed"):
        client.agent_instructions(doc_id="pi-old")


def test_agent_tools_doc_id_scopes_the_functions(client, store_path):
    seed_doc(store_path, "pi-a", "alpha.pdf")
    seed_doc(store_path, "pi-b", "secret.pdf")
    funcs = {fn.__name__: fn for fn in client.agent_tools(doc_id="pi-a")}
    blocked = json.loads(funcs["get_page_content"](doc_name="secret.pdf",
                                                   pages="1"))
    assert "success" not in blocked
    assert blocked["errorCode"] == "NOT_FOUND"
    allowed = json.loads(funcs["get_page_content"](doc_name="alpha.pdf",
                                                   pages="1"))
    assert allowed["success"] is True


def test_agent_tools_doc_id_refused_on_cloud():
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="local tools only"):
        cloud.agent_tools(doc_id="pi-a")


def test_cloud_tool_list_empty_raises(monkeypatch):
    """An empty tools/list must raise like empty instructions does: a
    zero-tool agent answers from the model's own knowledge instead of
    the documents, with nothing to signal it."""
    pytest.importorskip("agents")
    import pageindex.mcp_bridge as mcp_bridge

    class _ToollessBridge:
        def __init__(self, url, headers):
            pass

        def list_tools(self):
            return []

    monkeypatch.setattr(mcp_bridge, "McpBridge", _ToollessBridge)
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="no tools"):
        cloud.as_openai_tools()


def test_claude_wire_reads_only_the_routing_prefix():
    from pageindex.client import _claude_wire
    assert _claude_wire("claude-sonnet-4-6", "t()") == (
        "claude-sonnet-4-6", "anthropic")
    assert _claude_wire("anthropic/claude-sonnet-4-6", "t()") == (
        "claude-sonnet-4-6", "anthropic")
    assert _claude_wire(
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", "t()") == (
        "anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock")
    assert _claude_wire(
        "litellm/bedrock/us.anthropic.claude-sonnet-4-6-v1:0", "t()") == (
        "us.anthropic.claude-sonnet-4-6-v1:0", "bedrock")
    assert _claude_wire("vertex_ai/claude-sonnet-4@20250514", "t()") == (
        "claude-sonnet-4@20250514", "vertex_ai")
    # Unknown prefixes and bare names ship verbatim on the direct route.
    assert _claude_wire("team/claude-prod", "t()") == (
        "team/claude-prod", "anthropic")
    assert _claude_wire("sonnet", "t()") == ("sonnet", "anthropic")
    with pytest.raises(PageIndexAPIError, match="must be a str"):
        _claude_wire(123, "t()")


def test_stock_chat_model_never_impersonates_a_choice(store_path):
    pytest.importorskip("anthropic")
    from pageindex import PageIndexClient
    # Flagged at construction; any spelling that names a model clears it.
    stock = PageIndexLocalClient(storage_path=store_path)
    assert stock._chat_model_stock
    with pytest.raises(PageIndexAPIError, match="needs a model"):
        stock.anthropic_runner_config()
    for kwargs in ({"chat_model": "gpt-5.6-sol"}, {"retrieve_model": "gpt-4o"},
                   {"model": "gpt-4.1-mini"}):
        assert not PageIndexLocalClient(
            storage_path=store_path, **kwargs)._chat_model_stock
    # chat="local" alone names no model; managed chat has none at all.
    assert PageIndexClient(api_key="pi-test-key",
                           chat="local")._chat_model_stock
    assert PageIndexCloudClient(api_key="pi-test-key")._chat_model_stock
    # Assignment is a choice: the flag follows the write paths.
    stock.chat_model = "anthropic/claude-opus-4-1"
    assert not stock._chat_model_stock
    assert stock.anthropic_runner_config()["model"] == "claude-opus-4-1"


def test_anthropic_runner_config_strips_the_route_prefix(store_path):
    pytest.importorskip("anthropic")
    local = PageIndexLocalClient(
        storage_path=store_path,
        chat_model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    config = local.anthropic_runner_config()
    assert config["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert config["max_tokens"] == 8192
