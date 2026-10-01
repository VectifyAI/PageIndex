"""Jev-gated hybrid tree navigation: the routing kernel (threshold gates,
budget, depth, malformed input), the hard-dependency semantics, and the
two surfaces - client.search() and the find_pages agent tool. No network:
the kernel tests construct JevRouter over a duck-typed jev; the surface
tests monkeypatch jev_router.SystemOneClient."""
import json
import re
from pathlib import Path

import pytest

import pageindex.jev_router as jev_router
from pageindex import (PageIndexAPIError, PageIndexClient,
                       PageIndexLocalClient)
from pageindex.agent_tools import (TOOL_CONTRACT, _expand_pages,
                                   _format_page_spec, call_tool, tool_names)
from pageindex.jev_router import (JevRouter, JevUnavailable,
                                  SystemOneClient)
from pageindex.local_store import DocStore

SNAPSHOT_PATH = Path(__file__).parent / "data" / "cloud_mcp_contract.json"

QUESTION = "relevant_child"


@pytest.fixture(autouse=True)
def _jev_key(monkeypatch):
    """Key presence for every test; the missing-key test delenvs."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-jev-key")


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "store")


@pytest.fixture
def client(store_path):
    return PageIndexClient(storage_path=store_path)


# -- jev fakes --

class _ScriptedJev:
    """Duck-typed SystemOneClient: one scripted distribution per call, in
    order; past the script it prunes (NONE takes all the mass), the safe
    Jev answer."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def ask_choice(self, state_text, criteria):
        self.calls.append({"state": state_text,
                           "criteria": dict(criteria)})
        if self.scripts:
            return self.scripts.pop(0)
        return {**{cid: 0.0 for cid in criteria}, "NONE": 1.0}


class _TopChild:
    """Always puts 0.9 on the first child id of whatever fan-out arrives."""

    def __init__(self, api_key=None, timeout=None):
        pass

    def ask_choice(self, state_text, criteria):
        first = next(cid for cid in criteria if cid != "NONE")
        return {**{cid: 0.0 for cid in criteria}, first: 0.9, "NONE": 0.1}


class _NoJev:
    def ask_choice(self, state_text, criteria):
        raise AssertionError("Jev ran where the kernel must not call it")


class _ExplodingClient:
    """Stands in for SystemOneClient in build_router(): a Jev that failed
    for good - ask_choice raises JevUnavailable."""

    def __init__(self, api_key, timeout=None):
        self.message = "Jev request failed: HTTP 503 (service unavailable)"

    def ask_choice(self, state_text, criteria):
        raise JevUnavailable(self.message)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload


def _answer(criteria, top=None, p=0.9):
    """A well-formed System One answer body over ``criteria``."""
    top = top or next(cid for cid in criteria if cid != "NONE")
    probabilities = {cid: 0.0 for cid in criteria if cid not in (top, "NONE")}
    probabilities[top] = p
    probabilities["NONE"] = round(1 - sum(probabilities.values()), 10)
    return {"model": jev_router.MODEL,
            "answers": {QUESTION: {"type": "choice", "choice": top,
                                   "confidence": 0.99,
                                   "probabilities": probabilities}}}


def _install_transport(monkeypatch, status_code=200):
    """A SystemOneClient whose session post is a recorder that answers
    each request's own criteria. Returns (client, seen)."""
    seen = {}

    def post(url, **kwargs):
        seen.update(url=url, payload=kwargs.get("json"),
                    headers=kwargs.get("headers") or {})
        criteria = seen["payload"]["questions"][QUESTION]["criteria"]
        body = _answer(criteria) if status_code == 200 else {}
        return _FakeResponse(body, status_code)

    jev = SystemOneClient("sk-test")
    monkeypatch.setattr(jev._session, "post", post)
    return jev, seen


# -- trees --

def _leaf(node_id, page, title=None):
    return {"title": title or f"Section {node_id}", "node_id": node_id,
            "start_index": page, "end_index": page,
            "summary": f"{node_id} summary"}


def _two_leaf_tree():
    """Root 0000 over leaves 0001 (page 1) and 0002 (page 2)."""
    return [{"title": "Doc", "node_id": "0000", "start_index": 1,
             "end_index": 2, "summary": "root summary",
             "nodes": [_leaf("0001", 1), _leaf("0002", 2)]}]


def _chain(depth):
    """A single-child chain ``depth`` levels deep: ids n0..n{depth}, root
    at depth 0 on page 1, deepest leaf on page depth+1."""
    node = _leaf(f"n{depth}", depth + 1)
    for i in range(depth, 0, -1):
        node = {"title": f"n{i - 1}", "node_id": f"n{i - 1}",
                "start_index": i, "end_index": depth + 1,
                "summary": f"n{i - 1} summary", "nodes": [node]}
    return [node]


def _fan(n):
    """One root over ``n`` leaf children, ids 0001..{n:04d}, page = id."""
    return [{"title": "Doc", "node_id": "0000", "start_index": 1,
             "end_index": n, "summary": "root summary",
             "nodes": [_leaf(f"{i:04d}", i) for i in range(1, n + 1)]}]


def seed_doc(storage_path, doc_id, name, tree=None):
    """Seed the store directly - no indexing pipeline, no litellm."""
    pages = [{"page_index": 1, "markdown": "apples"},
             {"page_index": 2, "markdown": "bananas"}]
    tree = tree if tree is not None else _two_leaf_tree()
    meta = {"id": doc_id, "name": name, "description": "A test document",
            "status": "completed", "createdAt": "2026-08-01T10:00:00.123000",
            "pageNum": 2, "folderId": None, "metadata": None,
            "mode": "standard"}
    DocStore(storage_path).save_document(doc_id, meta, tree, pages)
    return doc_id


# -- llm seam --

def _no_llm(monkeypatch):
    """Pin the escalation lane shut: any LLM use fails the test."""
    import pageindex.utils as utils

    def boom(*args, **kwargs):
        raise AssertionError("LLM escalation ran where it must not")

    monkeypatch.setattr(utils, "llm_completion", boom)


def _script_llm(monkeypatch, behavior):
    """Script the escalation lane; returns the recorded (model, prompt)
    calls."""
    import pageindex.utils as utils
    calls = []

    def fake(model, prompt, **kwargs):
        calls.append((model, prompt))
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior(prompt) if callable(behavior) else behavior

    monkeypatch.setattr(utils, "llm_completion", fake)
    return calls


# -- kernel --

def test_client_payload_shape(monkeypatch):
    jev, seen = _install_transport(monkeypatch)
    probabilities = jev.ask_choice(
        "state text", {"0001": "Intro", "0002": "Body", "NONE": "none"})
    assert seen["url"] == jev_router.ENDPOINT
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["payload"]["model"] == "jev-latest"
    question = seen["payload"]["questions"][QUESTION]
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "0001": "Intro", "0002": "Body", "NONE": "none"}
    assert probabilities == {"0001": 0.9, "0002": 0.0, "NONE": 0.1}


def test_high_confidence_single_descend(monkeypatch):
    _no_llm(monkeypatch)
    jev = _ScriptedJev([{"0001": 0.9, "0002": 0.05, "NONE": 0.05}])
    result = JevRouter(jev, "unused-model").route("Where is X?",
                                                  _two_leaf_tree())
    assert result["stats"] == {"jev_calls": 1, "llm_calls": 0, "pruned": 0,
                               "fallbacks": 0}
    assert [hit["node_id"] for hit in result["hits"]] == ["0001"]
    state, criteria = jev.calls[0]["state"], jev.calls[0]["criteria"]
    assert state == ("Node: Doc\n"
                     "User query: Where is X?\n"
                     "Candidate children (id: title):\n"
                     "0001: Section 0001\n"
                     "0002: Section 0002")
    assert criteria == {"0001": "Section 0001", "0002": "Section 0002",
                        "NONE": jev_router._NONE_DESCRIPTION}


def test_mid_band_expands_top2(monkeypatch):
    _no_llm(monkeypatch)
    jev = _ScriptedJev([{"0001": 0.6, "0002": 0.3, "NONE": 0.1}])
    result = JevRouter(jev, "unused-model").route("Where is X?",
                                                  _two_leaf_tree())
    assert result["stats"] == {"jev_calls": 1, "llm_calls": 0, "pruned": 0,
                               "fallbacks": 0}
    assert sorted(hit["node_id"] for hit in result["hits"]) == ["0001", "0002"]


def test_none_prunes_subtree(monkeypatch):
    _no_llm(monkeypatch)
    jev = _ScriptedJev([{"0001": 0.2, "0002": 0.2, "NONE": 0.6}])
    result = JevRouter(jev, "unused-model").route("Where is X?",
                                                  _two_leaf_tree())
    assert result["stats"]["pruned"] == 1
    assert result["hits"] == []


def test_low_confidence_escalates_to_llm(monkeypatch):
    jev = _ScriptedJev([{"0001": 0.45, "0002": 0.25, "NONE": 0.3}])
    calls = _script_llm(monkeypatch, '["0002"]')
    result = JevRouter(jev, "test-model").route("Where is X?",
                                                _two_leaf_tree())
    assert result["stats"] == {"jev_calls": 1, "llm_calls": 1, "pruned": 0,
                               "fallbacks": 0}
    assert [hit["node_id"] for hit in result["hits"]] == ["0002"]
    model, prompt = calls[0]
    assert model == "test-model"
    for needle in ("Where is X?", "Section 0001", "0002 summary"):
        assert needle in prompt


def test_llm_escalation_failure_expands_all(monkeypatch):
    from pageindex.utils import LLMRetriesExhausted
    jev = _ScriptedJev([{"0001": 0.45, "0002": 0.25, "NONE": 0.3}])
    calls = _script_llm(monkeypatch, LLMRetriesExhausted("ladder gave up"))
    result = JevRouter(jev, "test-model").route("Where is X?",
                                                _two_leaf_tree())
    assert result["stats"]["llm_calls"] == 1
    assert result["stats"]["fallbacks"] == 1
    assert sorted(hit["node_id"] for hit in result["hits"]) == ["0001", "0002"]


def test_jev_failure_propagates(monkeypatch):
    """Kernel-level half of the hard-dependency rule: a dead Jev aborts the
    walk - it is never absorbed into a silent full expansion."""
    _no_llm(monkeypatch)
    router = JevRouter(_ExplodingClient("k"), "unused-model")
    with pytest.raises(JevUnavailable):
        router.route("Where is X?", _two_leaf_tree())


def test_wide_fanout_skips_jev(monkeypatch):
    calls = _script_llm(monkeypatch, '["0001"]')
    result = JevRouter(_NoJev(), "test-model").route("Where is X?", _fan(300))
    assert result["stats"]["jev_calls"] == 0
    assert result["stats"]["llm_calls"] == 1
    assert [hit["node_id"] for hit in result["hits"]] == ["0001"]
    assert "300" in calls[0][1] or "Section 0300" in calls[0][1]


def test_budget_exhaustion_returns_partial(monkeypatch):
    _no_llm(monkeypatch)
    result = JevRouter(_TopChild(), "unused-model", MAX_JEV_CALLS=1).route(
        "Where is X?", _chain(3))
    # One decision spent at the root; the undispatched frontier (n1) joins
    # the hits - its start_index still covers the unreached leaves.
    assert result["stats"] == {"jev_calls": 1, "llm_calls": 0, "pruned": 0,
                               "fallbacks": 0}
    assert [hit["node_id"] for hit in result["hits"]] == ["n1"]


def test_depth_limit(monkeypatch):
    _no_llm(monkeypatch)
    result = JevRouter(_TopChild(), "unused-model").route("Where is X?",
                                                          _chain(13))
    # Depths 0..11 each spend a decision; the node at MAX_DEPTH is the hit.
    assert result["stats"]["jev_calls"] == 12
    assert [hit["node_id"] for hit in result["hits"]] == ["n12"]


def test_duplicate_node_id_defense(monkeypatch):
    _no_llm(monkeypatch)
    tree = [{"title": "Doc", "node_id": "0000", "start_index": 1,
             "end_index": 3, "summary": "root summary",
             "nodes": [
                 {"title": "A", "node_id": "0001", "start_index": 1,
                  "end_index": 2, "summary": "a summary",
                  "nodes": [_leaf("0002", 2)]},
                 {"title": "A again", "node_id": "0001",
                  "start_index": 1, "end_index": 3},
             ]}]
    result = JevRouter(_TopChild(), "unused-model").route("Where is X?",
                                                          tree)
    # The duplicate id collapses onto its first occurrence; the walk
    # terminates and the first subtree's leaf is the hit.
    assert result["stats"]["jev_calls"] == 2
    assert [hit["node_id"] for hit in result["hits"]] == ["0002"]


def test_page_spec_formatting():
    hits = [{"start_index": page} for page in (3, 4, 5, 7)]
    assert jev_router.page_spec(hits) == "3-5,7"
    # Feeds straight into get_page_content's pages grammar.
    assert _expand_pages("3-5,7") == [3, 4, 5, 7]
    assert jev_router.format_ranges([3, 4, 5, 7]) == _format_page_spec(
        [3, 4, 5, 7])


# -- surfaces: client.search and the find_pages tool --

def _patched_jev(monkeypatch, cls):
    """Route build_router() at a fake SystemOneClient class."""
    monkeypatch.setattr(jev_router, "SystemOneClient", cls)


def test_missing_key_raises(client, store_path, monkeypatch):
    """No TYPESAFE_API_KEY: search() raises with the configuration pointer
    and find_pages returns the error envelope - never a silent full
    expansion."""
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _no_llm(monkeypatch)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(PageIndexAPIError, match="TYPESAFE_API_KEY"):
        client.search(doc_id, "Where is X?")
    text, is_error = call_tool(client, "find_pages",
                               {"doc_name": "report.pdf", "query": "q"})
    payload = json.loads(text)
    assert is_error and "success" not in payload
    assert "TYPESAFE_API_KEY" in payload["error"]
    assert "pages" not in payload


def test_jev_http_failure_aborts(client, store_path, monkeypatch):
    """A Jev failure past the transport's retries aborts both surfaces -
    search() raises, the tool returns the error envelope, and no path
    silently expands the whole tree."""
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _no_llm(monkeypatch)
    _patched_jev(monkeypatch, _ExplodingClient)
    with pytest.raises(PageIndexAPIError, match="Jev"):
        client.search(doc_id, "Where is X?")
    text, is_error = call_tool(client, "find_pages",
                               {"doc_name": "report.pdf", "query": "q"})
    payload = json.loads(text)
    assert is_error and "success" not in payload
    assert "Jev" in payload["error"]
    assert "pages" not in payload


def test_search_cloud_mode_raises(monkeypatch):
    """search() is local-only, and refuses before touching the network."""
    cloud = PageIndexClient(api_key="k")
    with pytest.raises(PageIndexAPIError, match="local-only"):
        cloud.search("pi-x", "Where is X?")


def test_search_returns_page_spec(client, store_path, monkeypatch):
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _patched_jev(monkeypatch, _TopChild)
    result = client.search(doc_id, "Where is X?")
    assert result["doc_id"] == doc_id
    assert result["pages"] == "1"
    assert result["page_ranges"] == [{"start": 1, "end": 1,
                                      "title": "Section 0001",
                                      "node_id": "0001"}]
    assert result["navigated"] == {"jev_calls": 1, "llm_calls": 0,
                                   "pruned": 0, "fallbacks": 0}


def test_find_pages_envelope_and_next_steps(client, store_path, monkeypatch):
    seed_doc(store_path, "pi-a", "report.pdf")
    _patched_jev(monkeypatch, _TopChild)
    text, is_error = call_tool(
        client, "find_pages",
        {"doc_name": "report.pdf", "query": "Where is X?"})
    payload = json.loads(text)
    assert not is_error and payload["success"] is True
    assert payload["doc_name"] == "report.pdf"
    assert payload["pages"] == "1"
    assert payload["page_ranges"] == [{"start": 1, "end": 1,
                                       "title": "Section 0001",
                                       "node_id": "0001"}]
    assert payload["navigated"] == {"jev_calls": 1, "llm_calls": 0,
                                    "pruned": 0, "fallbacks": 0}
    assert "get_page_content" in payload["next_steps"]["options"][0]

    # Zero hits: success, but the pages spec is empty and the guidance
    # points back at the full structure walk.
    tree = [{"title": "Doc", "node_id": "0000", "start_index": 1,
             "end_index": 2, "summary": "root", "nodes": [
                 _leaf("0001", 1), _leaf("0002", 2)]}]
    seed_doc(store_path, "pi-b", "empty.pdf", tree=tree)
    monkeypatch.setattr(jev_router, "SystemOneClient", _ScriptedJev)
    import pageindex.agent_tools as agent_tools
    monkeypatch.setattr(
        agent_tools, "build_router",
        lambda model: JevRouter(_ScriptedJev([{"NONE": 0.6, "0001": 0.2,
                                               "0002": 0.2}]), model))
    text, is_error = call_tool(
        client, "find_pages",
        {"doc_name": "empty.pdf", "query": "Where is X?"})
    payload = json.loads(text)
    assert not is_error and payload["success"] is True
    assert payload["pages"] == ""
    assert payload["page_ranges"] == []
    assert "get_document_structure" in payload["next_steps"]["options"][0]

    # Unknown document: the standard NOT_FOUND envelope.
    text, is_error = call_tool(
        client, "find_pages", {"doc_name": "missing.pdf", "query": "q"})
    payload = json.loads(text)
    assert is_error and payload["errorCode"] == "NOT_FOUND"
    assert "similar_files" in payload


def test_find_pages_registered_surface(client):
    assert "find_pages" in tool_names()
    # Local-only by design: the cloud contract snapshot stays untouched.
    assert "find_pages" not in TOOL_CONTRACT
    assert json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))[
        "tools"] == TOOL_CONTRACT
    tools = client.agent_tools()
    fn = next(t for t in tools if t.__name__ == "find_pages")
    # The capability-phrase guard the contract tests enforce, applied to
    # the new docstring.
    for phrase in ("shared-with-me", "sub-folder", "get_folder_structure",
                   "search_documents", "get_document_image"):
        assert phrase not in fn.__doc__, phrase
    assert fn.__doc__.count('sort="relevance"') == 0
    named = set(re.findall(r"\b(\w+)\(", fn.__doc__))
    assert named <= set(tool_names(include_management=True)), named


# -- agent loop (mirrors tests/test_local_chat.py's fake backend harness) --

try:
    import agents  # noqa: F401
    _HAS_AGENTS = True
except ImportError:  # the e2e test below skips; everything else runs
    _HAS_AGENTS = False

needs_agents = pytest.mark.skipif(not _HAS_AGENTS,
                                  reason="openai-agents not installed")


def _msg_item(text):
    from openai.types.responses import (ResponseOutputMessage,
                                        ResponseOutputText)
    return ResponseOutputMessage(
        id="msg_1", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text,
                                    annotations=[])])


def _call_item(name, arguments, call_id="call_1"):
    from openai.types.responses import ResponseFunctionToolCall
    return ResponseFunctionToolCall(
        id="fc_1", type="function_call", call_id=call_id, name=name,
        arguments=json.dumps(arguments), status="completed")


if _HAS_AGENTS:
    from agents.models.interface import Model  # noqa: E402
else:  # pragma: no cover - placeholder so the class statement parses
    Model = object


class _FakeModel(Model):
    """Scripted backend: one list of output items per model turn."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.inputs = []
        self.instructions = []

    def _record(self, system_instructions, input):
        self.instructions.append(system_instructions)
        items = input if isinstance(input, list) else [input]
        self.inputs.append(
            [dict(item) if isinstance(item, dict) else item
             for item in items])

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing,
                           **kwargs):
        from agents.items import ModelResponse
        from agents.usage import Usage
        self._record(system_instructions, input)
        return ModelResponse(output=self.turns.pop(0),
                             usage=Usage(requests=1, input_tokens=10,
                                         output_tokens=5, total_tokens=15),
                             response_id=None)

    async def stream_response(self, system_instructions, input,
                              model_settings, tools, output_schema, handoffs,
                              tracing, **kwargs):
        # The e2e runs the non-streaming lane; the abstract method only
        # needs to exist.
        raise NotImplementedError("non-streaming fake")


@pytest.fixture
def fake_model(monkeypatch):
    def install(turns):
        import pageindex.local_chat as local_chat
        fake = _FakeModel(turns)

        def factory(protocol, model_name, backend=None):
            return fake

        monkeypatch.setattr(local_chat, "_openai_model", factory)
        return fake
    return install


@needs_agents
def test_agent_loop_end_to_end(client, store_path, fake_model, monkeypatch):
    """The managed agent loop drives find_pages for real: turn 1 calls the
    tool, the router runs against the fake Jev, and turn 2's input carries
    the page spec."""
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _patched_jev(monkeypatch, _TopChild)
    fake = fake_model([
        [_call_item("find_pages", {"doc_name": "report.pdf",
                                   "query": "Where is X?"})],
        [_msg_item("It is on page 1.")],
    ])
    result = client.chat_completions(
        [{"role": "user", "content": "Where is X in report.pdf?"}])
    assert result["choices"][0]["message"]["content"] == "It is on page 1."
    # Turn 2 carries the tool's envelope: the find_pages result reached the
    # model with the located page.
    tool_outputs = [item for item in fake.inputs[1]
                    if isinstance(item, dict)
                    and item.get("type") == "function_call_output"]
    output = tool_outputs[0]["output"]
    if isinstance(output, list):  # MCP content blocks
        output = output[0]["text"]
    envelope = json.loads(output)
    assert envelope["success"] is True and envelope["pages"] == "1"
    assert envelope["page_ranges"][0]["node_id"] == "0001"
