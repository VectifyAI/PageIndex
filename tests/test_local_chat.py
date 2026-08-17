"""Local chat surfaces: three protocols over fake backends — no network,
no LLM keys. Tool execution runs for real against a seeded local store."""
import asyncio
import json
import sys
import types

import httpx  # via the hard `openai` dependency
import pytest

import pageindex.local_chat as local_chat
from pageindex import (PageIndexAPIError, PageIndexCloudClient,
                       PageIndexLocalClient)
from pageindex.local_chat import CHAT_HEADER
from pageindex.local_store import DocStore


def seed_doc(storage_path, doc_id, name):
    pages = [{"page_index": 1, "markdown": "Page one text about apples"}]
    tree = [{"title": "Doc", "node_id": "0000", "start_index": 1,
             "end_index": 1, "summary": "root summary", "text": "ROOT"}]
    meta = {
        "id": doc_id, "name": name, "description": "A test document",
        "status": "completed", "createdAt": "2026-08-01T10:00:00.123000",
        "pageNum": 1, "folderId": None, "metadata": None, "mode": "standard",
    }
    DocStore(storage_path).save_document(doc_id, meta, tree, pages)
    return doc_id


@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "store")


@pytest.fixture
def client(store_path):
    return PageIndexLocalClient(storage_path=str(store_path))


# ── OpenAI engine fakes (chat_completions / responses) ──
# Section-scoped skips: each engine's tests skip independently, so a
# machine with only one extra installed still covers the other surface.

try:
    import agents  # noqa: F401
    _HAS_AGENTS = True
except ImportError:
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


def _usage():
    from agents.usage import Usage
    return Usage(requests=1, input_tokens=10, output_tokens=5,
                 total_tokens=15)


if _HAS_AGENTS:
    from agents.models.interface import Model  # noqa: E402
else:  # pragma: no cover - placeholder so the class statement parses
    Model = object


class FakeModel(Model):
    """Scripted backend: one list of output items per model turn."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.inputs = []
        self.instructions = []
        self.deltas_emitted = 0

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
        self._record(system_instructions, input)
        # Mimic the real model's transport hop when a test attaches one, so
        # the transport-level status recorder sees each turn.
        transport = getattr(getattr(self, "_client", None), "responses", None)
        if transport is not None:
            await transport.create()
        return ModelResponse(output=self.turns.pop(0), usage=_usage(),
                             response_id=None)

    async def stream_response(self, system_instructions, input,
                              model_settings, tools, output_schema, handoffs,
                              tracing, **kwargs):
        import asyncio as aio
        from openai.types.responses import (Response, ResponseCompletedEvent,
                                            ResponseTextDeltaEvent)
        from openai.types.responses.response_usage import (
            InputTokensDetails, OutputTokensDetails, ResponseUsage)
        block_from = getattr(self, "block_from", None)
        if block_from is not None and len(self.inputs) + 1 >= block_from:
            while True:  # released only by task cancellation
                await aio.sleep(0.01)
        self._record(system_instructions, input)
        output = self.turns.pop(0)
        sequence = 0
        if getattr(self, "emit_created", False):
            from openai.types.responses import ResponseCreatedEvent
            sequence += 1
            yield ResponseCreatedEvent(
                type="response.created", sequence_number=sequence,
                response=Response(
                    id="resp_backend_turn", created_at=0.0, model="fake",
                    object="response", output=[], parallel_tool_calls=False,
                    tool_choice="auto", tools=[], status="in_progress"))
        for item in output:
            if item.type == "message":
                pieces = getattr(self, "pieces", ("The ", "answer"))
                for piece in pieces:
                    sequence += 1
                    self.deltas_emitted += 1
                    yield ResponseTextDeltaEvent(
                        type="response.output_text.delta", delta=piece,
                        content_index=0, item_id=item.id, output_index=0,
                        logprobs=[], sequence_number=sequence)
        if getattr(self, "no_terminal", False):
            return  # backend died mid-stream: no terminal event
        sequence += 1
        yield ResponseCompletedEvent(
            type="response.completed", sequence_number=sequence,
            response=Response(
                id="resp_fake", created_at=0.0, model="fake",
                object="response", output=output, parallel_tool_calls=False,
                tool_choice="auto", tools=[],
                usage=ResponseUsage(
                    input_tokens=10, output_tokens=5, total_tokens=15,
                    input_tokens_details=InputTokensDetails(
                        cached_tokens=0, cache_write_tokens=0),
                    output_tokens_details=OutputTokensDetails(
                        reasoning_tokens=0))))


@pytest.fixture
def fake_model(monkeypatch):
    state = {}

    def install(turns):
        fake = FakeModel(turns)
        state["protocols"] = []

        def factory(protocol, model_name, backend=None):
            state["protocols"].append((protocol, model_name))
            state["backends"] = state.get("backends", []) + [backend]
            return fake

        monkeypatch.setattr(local_chat, "_openai_model", factory)
        return fake

    install.state = state
    return install


# ── chat_completions ──

@needs_agents
def test_chat_completions_end_to_end(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.chat_completions(
        [{"role": "user", "content": "What status?"}])
    assert result["id"].startswith("chatcmpl-")
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant",
                                               "content": "The answer"}
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 20, "completion_tokens": 10,
                               "total_tokens": 30,
                               "prompt_tokens_details": {"cached_tokens": 0},
                               "completion_tokens_details":
                                   {"reasoning_tokens": 0}}
    assert fake_model.state["protocols"][0][0] == "chat"
    # The tool ran for real: turn 2's input carries its output.
    turn2 = json.dumps(fake.inputs[1])
    assert "report.pdf" in turn2 and "completed" in turn2
    # Managed instructions: header + the local agent guidance.
    assert fake.instructions[0].startswith(CHAT_HEADER)
    assert "READING WORKFLOW" in fake.instructions[0]


@needs_agents
def test_chat_completions_system_and_doc_block(client, store_path, fake_model):
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("ok")]])
    client.chat_completions(
        [{"role": "system", "content": "Answer in French."},
         {"role": "user", "content": "hi"}],
        doc_id=doc_id)
    assert fake.instructions[0].endswith("Answer in French.")
    first_item = fake.inputs[0][0]
    assert "The user has specified document: report.pdf" in first_item["content"]


@needs_agents
def test_chat_completions_accepts_query_string(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("Answer")]])
    result = client.chat_completions("What status?")
    assert result["choices"][0]["message"]["content"] == "Answer"
    assert fake.inputs[0][-1] == {"role": "user", "content": "What status?"}
    with pytest.raises(PageIndexAPIError, match="non-empty string"):
        client.chat_completions("   ")


@needs_agents
def test_chat_completions_validation(client, store_path, fake_model):
    fake_model([[_msg_item("ok")]])
    with pytest.raises(PageIndexAPIError, match="cloud-only"):
        client.chat_completions([{"role": "user", "content": "x"}],
                                enable_citations=True)
    with pytest.raises(PageIndexAPIError, match="responses\\(\\) or messages"):
        client.chat_completions([{"role": "tool", "content": "x"}])
    with pytest.raises(PageIndexAPIError, match="must be a string"):
        client.chat_completions([{"role": "user", "content": [1]}])
    with pytest.raises(PageIndexAPIError, match="non-empty"):
        client.chat_completions([])
    with pytest.raises(PageIndexAPIError,
                       match="Documents not found or access denied: a, b"):
        client.chat_completions([{"role": "user", "content": "x"}],
                                doc_id=["a", "b"])


@needs_agents
def test_chat_completions_stream_modes(client, store_path, fake_model):
    fake_model([[_msg_item("The answer")]])
    pieces = list(client.chat_completions(
        [{"role": "user", "content": "q"}], stream=True))
    assert pieces == ["The ", "answer"]

    fake_model([[_msg_item("The answer")]])
    chunks = list(client.chat_completions(
        [{"role": "user", "content": "q"}], stream=True,
        stream_metadata=True))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant",
                                               "content": ""}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] == 15
    assert all(c["object"] == "chat.completion.chunk" for c in chunks[:-1])


def test_chat_completions_missing_framework(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    with pytest.raises(PageIndexAPIError, match="pip install openai-agents"):
        client.chat_completions([{"role": "user", "content": "x"}])


def test_cloud_guards():
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="local-mode parameters"):
        cloud.chat_completions([{"role": "user", "content": "x"}], model="m")
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               reasoning_effort="low")
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               extra_body={"service_tier": "auto"})
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat_completions([{"role": "user", "content": "x"}], top_p=0.9)
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               max_tokens=256)
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat("x", reasoning_effort="low")
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               backend={"api_key": "k"})
    with pytest.raises(PageIndexAPIError, match="local-mode"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               extra_headers={"x-beta": "1"})
    with pytest.raises(PageIndexAPIError, match="not available on PageIndex "
                                                "cloud yet"):
        cloud.responses("x")
    with pytest.raises(PageIndexAPIError, match="not available on PageIndex "
                                                "cloud yet"):
        cloud.messages([{"role": "user", "content": "x"}], model="m",
                       max_tokens=10)


@needs_agents
def test_anthropic_routed_models_mark_managed_prefix_for_cache(
        client, store_path, fake_model):
    fake_model([[_msg_item("ok")]])
    from pageindex.local_chat import _openai_agent
    marked = {"cache_control_injection_points": [
        {"location": "message", "role": "system"}]}
    for name in ("anthropic/claude-x", "litellm/anthropic/claude-x",
                 "bedrock/us.anthropic.claude-sonnet-5",
                 "vertex_ai/claude-sonnet-4-5"):
        agent = _openai_agent(client, "chat", name, "sys", None, None)
        assert agent.model_settings.extra_args == marked
    for name in ("gpt-5", "openai/Qwen/x", "litellm/groq/x",
                 "bedrock/meta.llama3-70b-instruct-v1:0",
                 "vertex_ai/gemini-2.5-pro"):
        agent = _openai_agent(client, "chat", name, "sys", None, None)
        assert agent.model_settings.extra_args is None


@needs_agents
def test_status_recorder_attaches_to_the_real_responses_model(monkeypatch):
    # Guards the private-attribute chain the recorder rides
    # (agent.model._client.responses.create): a vendor rename turns the
    # recorder into a silent no-op and truncation reports as completion.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import openai
    from agents.models.openai_responses import OpenAIResponsesModel
    backend = openai.AsyncOpenAI()
    model = OpenAIResponsesModel("gpt-test", openai_client=backend)
    original = backend.responses.create
    local_chat._record_response_status(types.SimpleNamespace(model=model), {})
    assert backend.responses.create is not original
    asyncio.run(backend.close())


@needs_agents
def test_cache_marker_reaches_the_anthropic_wire(client, store_path,
                                                 monkeypatch):
    # End-to-end guard for the injection flag: through the real
    # LitellmModel and litellm's request build, the marker must appear in
    # the HTTP body — a regression in either vendor hop silently reverts
    # anthropic-routed calls to full price.
    pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import (AsyncHTTPHandler,
                                                        HTTPHandler)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    captured = {}
    reply = {"id": "msg_01", "type": "message", "role": "assistant",
             "model": "claude-3-5-sonnet-20240620",
             "content": [{"type": "text", "text": "ok"}],
             "stop_reason": "end_turn", "stop_sequence": None,
             "usage": {"input_tokens": 10, "output_tokens": 2}}

    def _capture(url, kwargs):
        body = kwargs.get("json")
        if body is None and kwargs.get("data") is not None:
            body = json.loads(kwargs["data"])
        captured["url"] = str(url)
        captured["body"] = body
        return httpx.Response(200, json=reply,
                              request=httpx.Request("POST", str(url)))

    async def fake_apost(self, url=None, *args, **kwargs):
        return _capture(url, kwargs)

    def fake_post(self, url=None, *args, **kwargs):
        return _capture(url, kwargs)

    monkeypatch.setattr(AsyncHTTPHandler, "post", fake_apost)
    monkeypatch.setattr(HTTPHandler, "post", fake_post)
    result = client.chat_completions(
        "hi", model="anthropic/claude-3-5-sonnet-20240620")
    assert "/v1/messages" in captured["url"]
    assert '"cache_control"' in json.dumps(captured["body"])
    # The OpenAI cache-routing hint must not leak here: LiteLLM plants
    # extra_body as a literal field, and Anthropic rejects unknown fields.
    assert "extra_body" not in json.dumps(captured["body"])
    assert result["choices"][0]["message"]["content"] == "ok"


# ── chat (front door) ──

@needs_agents
def test_chat_returns_answer_string(client, store_path, fake_model):
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    assert client.chat("What status?", doc_id=doc_id) == "The answer"
    first_item = fake.inputs[0][0]
    assert "The user has specified document: report.pdf" in first_item["content"]


@needs_agents
def test_chat_stream_yields_text_chunks(client, store_path, fake_model):
    fake_model([[_msg_item("The answer")]])
    assert list(client.chat("q", stream=True)) == ["The ", "answer"]


@needs_agents
def test_chat_multi_turn_history(client, store_path, fake_model):
    fake = fake_model([[_msg_item("Chapter 4 covers pears")]])
    history = [
        {"role": "user", "content": "What about chapter 3?"},
        {"role": "assistant", "content": "Chapter 3 covers apples"},
        {"role": "user", "content": "And chapter 4?"},
    ]
    assert client.chat(history) == "Chapter 4 covers pears"
    assert fake.inputs[0][-3:] == history


@needs_agents
def test_chat_reasoning_effort_reaches_the_engine(client, store_path,
                                                  fake_model, monkeypatch):
    """The business door's one thinking knob rides chat_completions'
    channel unchanged; unset sends nothing."""
    seen = {}
    real = local_chat._openai_agent

    def spy(*args, **kwargs):
        agent = real(*args, **kwargs)
        seen["settings"] = agent.model_settings
        return agent

    monkeypatch.setattr(local_chat, "_openai_agent", spy)
    fake_model([[_msg_item("ok")]])
    client.chat("q", reasoning_effort="low")
    assert seen["settings"].extra_args["reasoning_effort"] == "low"
    fake_model([[_msg_item("ok")]])
    client.chat("q")
    assert seen["settings"].extra_args is None


def test_chat_cloud_unwraps_envelope(monkeypatch):
    cloud = PageIndexCloudClient(api_key="pi-test-key")

    def fake_cc(**kwargs):
        assert kwargs["messages"] == [{"role": "user", "content": "q"}]
        return {"choices": [{"message": {"role": "assistant",
                                        "content": "cloud answer"}}]}

    monkeypatch.setattr(cloud._api, "chat_completions", fake_cc)
    assert cloud.chat("q") == "cloud answer"


# ── responses ──

@needs_agents
def test_responses_end_to_end(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.responses("What status?")
    assert result["id"].startswith("resp_")
    assert result["object"] == "response"
    assert result["status"] == "completed"
    assert result["usage"] == {
        "input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 10,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 30}
    assert fake_model.state["protocols"][0][0] == "responses"
    assert [item.get("type", "message") for item in result["output"]] == [
        "function_call", "message"]
    assert [item.get("type", "message") for item in result["items"]] == [
        "function_call", "function_call_output", "message"]
    # The final item is the assistant answer.
    assert "The answer" in json.dumps(result["output"][-1])


@needs_agents
def test_responses_round_trip_extends_prefix(client, store_path, fake_model):
    """The cache contract: a round-tripped call's first model input must
    extend the previous call's final model input item-for-item."""
    seed_doc(store_path, "pi-a", "report.pdf")
    first = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.responses("What status?")

    second = fake_model([[_msg_item("Done")]])
    follow_up = ([{"role": "user", "content": "What status?"}]
                 + result["items"]
                 + [{"role": "user", "content": "and now?"}])
    client.responses(follow_up)
    previous_final = first.inputs[-1]
    assert second.inputs[0][:len(previous_final)] == previous_final


@needs_agents
def test_responses_round_trip_prefix_with_doc_id(client, store_path, fake_model):
    """Same contract with doc targeting: re-passing the same doc_id re-sets
    an identical leading block, so the prefix still extends item-for-item."""
    seed_doc(store_path, "pi-a", "report.pdf")
    first = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.responses("What status?", doc_id="pi-a")

    second = fake_model([[_msg_item("Done")]])
    follow_up = ([{"role": "user", "content": "What status?"}]
                 + result["items"]
                 + [{"role": "user", "content": "and now?"}])
    client.responses(follow_up, doc_id="pi-a")
    previous_final = first.inputs[-1]
    assert second.inputs[0][:len(previous_final)] == previous_final


@needs_agents
def test_doc_id_conversations_get_distinct_cache_keys(client, store_path,
                                                      fake_model,
                                                      monkeypatch):
    """The doc-targeting block is byte-identical for every conversation
    about a document — seeding the cache key on items[0] pooled them all
    under one prompt_cache_key."""
    seed_doc(store_path, "pi-a", "report.pdf")
    keys = []
    real = local_chat._conversation_cache_key

    def spy(model_name, instructions, items):
        key = real(model_name, instructions, items)
        keys.append(key)
        return key

    monkeypatch.setattr(local_chat, "_conversation_cache_key", spy)

    fake_model([[_msg_item("a")]])
    result = client.responses("What is the CAGR?", doc_id="pi-a")
    fake_model([[_msg_item("b")]])
    client.responses("Summarize section 3.", doc_id="pi-a")
    assert keys[0] != keys[1]  # unrelated conversations never pool

    fake_model([[_msg_item("c")]])
    follow_up = ([{"role": "user", "content": "What is the CAGR?"}]
                 + result["items"]
                 + [{"role": "user", "content": "and now?"}])
    client.responses(follow_up, doc_id="pi-a")
    assert keys[2] == keys[0]  # a continuation keeps its conversation's key

    fake_model([[_msg_item("d")]])
    client.chat_completions("What is the CAGR?", doc_id="pi-a")
    fake_model([[_msg_item("e")]])
    client.chat_completions("Summarize section 3.", doc_id="pi-a")
    assert keys[3] != keys[4]  # same property on the chat surface


@needs_agents
def test_responses_stream_passthrough(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    events = list(client.responses("q", stream=True))
    types = [event.get("type") for event in events]
    assert "response.output_text.delta" in types
    assert not [event for event in events
                if event.get("item", {}).get("type") == "function_call_output"]
    assert types[-1] == "response.completed"
    final = events[-1]["response"]
    assert final["status"] == "completed"
    assert final["usage"]["total_tokens"] == 30
    assert [item.get("type", "message") for item in final["output"]] == [
        "function_call", "message"]
    assert [item.get("type", "message") for item in final["items"]] == [
        "function_call", "function_call_output", "message"]
    # output_index addresses the logical response.output: turn 2's deltas
    # are re-based past turn 1's item instead of restarting at 0.
    last_delta = [event for event in events
                  if event.get("type") == "response.output_text.delta"][-1]
    assert (final["output"][last_delta["output_index"]]
            .get("type", "message") == "message")


@needs_agents
def test_responses_stream_opens_with_created(client, store_path, fake_model):
    """N per-turn openings collapse to one response.created, not zero —
    the logical stream must open with a response object carrying the same
    id the terminal event reports, and the terminal envelope reports the
    backend's tool-param echo, not assumed values."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.emit_created = True  # two turns emit two; one must pass through
    events = list(client.responses("q", stream=True))
    created = [e for e in events if e["type"] == "response.created"]
    assert len(created) == 1 and events[0] is created[0]
    assert events[0]["sequence_number"] == 1
    terminal = events[-1]
    assert terminal["type"] == "response.completed"
    assert created[0]["response"]["id"] == terminal["response"]["id"]
    assert terminal["response"]["parallel_tool_calls"] is False  # echo


@needs_agents
def test_responses_envelope_validates_as_official_response(client, store_path,
                                                           fake_model):
    """The conformance contract: the envelope parses with the official
    openai SDK types, and the transcript survives in the extension field."""
    from openai.types.responses import Response
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.responses("What status?")
    parsed = Response.model_validate(result)
    assert [item.type for item in parsed.output] == ["function_call",
                                                     "message"]
    assert parsed.model_dump()["items"] == result["items"]


@needs_agents
def test_responses_stream_events_validate_as_official_events(
        client, store_path, fake_model):
    """Every stream event, terminal envelope included, parses with the
    official event union."""
    from pydantic import TypeAdapter
    from openai.types.responses import ResponseStreamEvent
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    adapter = TypeAdapter(ResponseStreamEvent)
    events = list(client.responses("q", stream=True))
    assert events
    for event in events:
        adapter.validate_python(event)


# ── messages (Anthropic engine) ──

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

needs_anthropic = pytest.mark.skipif(not _HAS_ANTHROPIC,
                                     reason="anthropic not installed")


def _anthropic_message(content, stop_reason):
    return {
        "id": "msg_fake", "type": "message", "role": "assistant",
        "model": "claude-test", "content": content,
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture
def fake_anthropic(monkeypatch):
    state = {"calls": []}

    def install(responses):
        state["calls"].clear()

        def handler(request):
            state["calls"].append(json.loads(request.content))
            body = responses[len(state["calls"]) - 1]
            if isinstance(body, str):  # pre-rendered SSE
                return httpx.Response(
                    200, content=body.encode(),
                    headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=body)

        fake = anthropic.Anthropic(
            api_key="test",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        monkeypatch.setattr(local_chat, "_anthropic_client",
                            lambda backend=None: fake)
        return state["calls"]

    return install


@needs_anthropic
def test_messages_end_to_end(client, store_path, fake_anthropic):
    seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message(
            [{"type": "tool_use", "id": "tu_1", "name": "get_document",
              "input": {"doc_name": "report.pdf"}}], "tool_use"),
        _anthropic_message([{"type": "text", "text": "The answer"}],
                           "end_turn"),
    ])
    result = client.messages([{"role": "user", "content": "What status?"}],
                             model="claude-test", max_tokens=100)
    assert result["stop_reason"] == "end_turn"
    assert result["content"][0]["text"] == "The answer"
    assert result["usage"]["input_tokens"] == 20
    assert result["usage"]["output_tokens"] == 10
    # Full new-turn sequence, valid for verbatim history append.
    roles = [message["role"] for message in result["messages"]]
    assert roles == ["assistant", "user", "assistant"]
    tool_result = json.dumps(result["messages"][1])
    assert "tool_result" in tool_result and "report.pdf" in tool_result

    request = calls[0]
    assert request["system"][0]["text"].startswith(CHAT_HEADER)
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    browse = next(t for t in request["tools"]
                  if t["name"] == "browse_documents")
    assert "folder_id" not in browse["input_schema"]["properties"]
    # Native prefix continuation: request 2 extends request 1's messages.
    assert calls[1]["messages"][:len(calls[0]["messages"])] \
        == calls[0]["messages"]


@needs_anthropic
def test_messages_doc_block_and_system(client, store_path, fake_anthropic):
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    client.messages([{"role": "user", "content": "hi"}], model="claude-test",
                    max_tokens=100, doc_id=doc_id, system="Answer in French.")
    system = calls[0]["system"]
    assert "The user has specified document: report.pdf" in system[1]["text"]
    assert system[-1]["text"] == "Answer in French."


@needs_anthropic
def test_messages_stream_passthrough(client, store_path, fake_anthropic):
    sse = "\n".join([
        'event: message_start',
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}',
        "",
        'event: content_block_start',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        "",
        'event: content_block_delta',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The answer"}}',
        "",
        'event: content_block_stop',
        'data: {"type":"content_block_stop","index":0}',
        "",
        'event: message_delta',
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}',
        "",
        'event: message_stop',
        'data: {"type":"message_stop"}',
        "",
        "",
    ])
    fake_anthropic([sse])
    events = list(client.messages([{"role": "user", "content": "q"}],
                                  model="claude-test", max_tokens=100,
                                  stream=True))
    types = [event.type for event in events]
    assert "content_block_delta" in types and "message_stop" in types


@needs_anthropic
def test_messages_accepts_query_string(client, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    result = client.messages("What status?", model="claude-test")
    assert result["content"][0]["text"] == "ok"
    assert calls[0]["messages"] == [{"role": "user",
                                     "content": "What status?"}]
    # The wire-required budget is table-setting, not a user obligation.
    assert calls[0]["max_tokens"] == 8192
    with pytest.raises(PageIndexAPIError, match="non-empty string"):
        client.messages("   ", model="claude-test")


@needs_anthropic
def test_messages_validation(client, fake_anthropic):
    fake_anthropic([])
    with pytest.raises(PageIndexAPIError, match="non-empty"):
        client.messages([], model="claude-test", max_tokens=100)
    with pytest.raises(PageIndexAPIError,
                       match="Documents not found or access denied"):
        client.messages([{"role": "user", "content": "x"}],
                        model="claude-test", max_tokens=100, doc_id="ghost")


@needs_anthropic
def test_messages_raises_when_runner_params_unreadable(client, fake_anthropic,
                                                       monkeypatch):
    """The conversation is read back through set_messages_params (a mutator
    used as a reader); if a vendor change stops it delivering params, the
    envelope silently lost every tool turn — it must raise instead."""
    from anthropic.lib.tools import BetaToolRunner
    fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    monkeypatch.setattr(BetaToolRunner, "set_messages_params",
                        lambda self, params: None)
    with pytest.raises(PageIndexAPIError, match="anthropic version"):
        client.messages([{"role": "user", "content": "hi"}],
                        model="claude-test", max_tokens=100)


def test_messages_missing_framework(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(PageIndexAPIError, match="pageindex\\[anthropic\\]"):
        client.messages([{"role": "user", "content": "x"}],
                        model="claude-test", max_tokens=100)


# ── review-round regressions ──

def _anthropic_tool_use(tool_use_id="tu_1"):
    return {"type": "tool_use", "id": tool_use_id, "name": "get_document",
            "input": {"doc_name": "report.pdf"}}


@needs_agents
@pytest.mark.parametrize("surface", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
def test_max_turns_wrapped(client, store_path, fake_model, surface, streaming):
    """MaxTurnsExceeded is an engine-internal type; callers get the SDK's
    own error, with the engine exception kept as the cause — on every
    surface and both the non-stream and stream paths."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_call_item("get_document", {"doc_name": "report.pdf"}, "call_2")],
        [_msg_item("never reached")],
    ])
    with pytest.raises(PageIndexAPIError, match=r"max_turns \(1\)") as caught:
        result = getattr(client, surface)("q", max_turns=1, stream=streaming)
        if streaming:
            list(result)
    assert type(caught.value.__cause__).__name__ == "MaxTurnsExceeded"


@needs_agents
def test_max_turns_rejects_non_positive(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    with pytest.raises(PageIndexAPIError, match="positive integer"):
        client.chat_completions([{"role": "user", "content": "q"}],
                                max_turns=0)


def test_enable_citations_rejected_before_framework_check(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    with pytest.raises(PageIndexAPIError, match="cloud-only"):
        client.chat_completions([{"role": "user", "content": "x"}],
                                enable_citations=True)


@needs_agents
def test_chat_stream_role_chunk_even_with_empty_output(client, fake_model):
    fake_model([[]])
    chunks = list(client.chat_completions([{"role": "user", "content": "q"}],
                                          stream=True, stream_metadata=True))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant",
                                               "content": ""}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"


@needs_agents
def test_responses_stream_single_completed_monotonic_sequence(
        client, store_path, fake_model):
    """One logical response per call: per-turn backend lifecycle events are
    collapsed and sequence numbers never go backwards."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    events = list(client.responses("q", stream=True))
    completed = [event for event in events
                 if event.get("type") == "response.completed"]
    assert len(completed) == 1 and events[-1] is completed[0]
    sequences = [event["sequence_number"] for event in events
                 if "sequence_number" in event]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


@needs_agents
def test_responses_envelope_fields_and_cache_group(client, store_path,
                                                   fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_msg_item("ok")]])
    result = client.responses("q")
    names = {tool["name"] for tool in result["tools"]}
    assert names == {"browse_documents", "get_document",
                     "get_document_structure", "get_page_content"}
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["instructions"].startswith(CHAT_HEADER)
    # No transport echo attached in this fixture, so these are the
    # documented fallbacks (the OpenAI server defaults), not assertions
    # about the request.
    assert result["parallel_tool_calls"] is True
    assert result["tool_choice"] == "auto"


def test_sol_class_refusal_names_its_exits():
    """The chatcmpl+tools-while-reasoning 400 is a lane problem, not a
    retry problem — the wrapped error must name every exit."""
    err = local_chat._model_backend_error(Exception(
        "Error code: 400 - Function tools with reasoning_effort are not "
        "supported for gpt-5.6-sol in /v1/chat/completions."))
    assert "responses()" in str(err) and "litellm" in str(err)
    assert "pass reasoning_effort" in str(err)
    plain = local_chat._model_backend_error(Exception("rate limited"))
    assert "responses()" not in str(plain)


def test_conversation_cache_key_stable_per_conversation():
    """Cache-routing key, sent as the OpenAI prompt_cache_key. A
    conversation's continuations must share one key (same model /
    instructions / first item), and unrelated conversations must not pool
    under it."""
    turn1 = [{"role": "user", "content": "q"}]
    continuation = turn1 + [{"role": "assistant", "content": "a"},
                            {"role": "user", "content": "and?"}]
    key = local_chat._conversation_cache_key("m", "sys", turn1)
    assert key == local_chat._conversation_cache_key("m", "sys", continuation)
    assert key != local_chat._conversation_cache_key(
        "m", "sys", [{"role": "user", "content": "other"}])
    assert key != local_chat._conversation_cache_key("m2", "sys", turn1)
    assert key != local_chat._conversation_cache_key("m", "sys2", turn1)


@needs_agents
def test_agent_carries_prompt_cache_key_in_extra_body(monkeypatch):
    """The key must reach the wire: openai-agents 0.20 dropped the
    RunConfig.group_id -> prompt_cache_key derivation, so the agent's
    ModelSettings.extra_body is the delivery channel. OpenAI destinations
    only — prompt_cache_key is OpenAI's routing hint, and LiteLLM plants
    extra_body as a literal field in other providers' bodies (Anthropic
    rejects unknown fields); Claude routes keep their cache_control marker
    in extra_args instead."""
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(
        None, "chat", "anthropic/claude-x", "sys", None, None,
        doc_ids=None, cache_key="pageindex-k1")
    settings = agent.model_settings
    assert settings.extra_body is None
    assert "cache_control_injection_points" in settings.extra_args
    for name in ("gpt-test", "openai/gpt-test", "litellm/openai/gpt-test"):
        agent = local_chat._openai_agent(
            None, "chat", name, "sys", None, None,
            doc_ids=None, cache_key="pageindex-k2")
        assert agent.model_settings.extra_body == {
            "prompt_cache_key": "pageindex-k2"}, name
        assert agent.model_settings.extra_args is None


@needs_agents
def test_reasoning_passthrough_reaches_each_engine(monkeypatch):
    """Per-door native reasoning, forwarded verbatim. The chat door's
    effort rides extra_args — LiteLLM's own top-level kwarg on every
    supported openai-agents version, and the channel admits values outside
    the OpenAI enum ("none") — coexisting with the Claude cache marker.
    The responses door's object rides ModelSettings.reasoning, which the
    Responses model forwards verbatim. Unset sends nothing."""
    pytest.importorskip("litellm")
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, reasoning_effort="low")
    assert agent.model_settings.extra_args == {"reasoning_effort": "low"}
    assert agent.model_settings.reasoning is None
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", None, None,
                                     reasoning_effort="none")
    assert agent.model_settings.extra_args["reasoning_effort"] == "none"
    assert "cache_control_injection_points" in agent.model_settings.extra_args
    agent = local_chat._openai_agent(None, "responses", "gpt-test", "sys",
                                     None, None,
                                     reasoning={"effort": "low",
                                                "summary": "auto"})
    # ModelSettings coerces the dict into the typed openai Reasoning object.
    assert agent.model_settings.reasoning.effort == "low"
    assert agent.model_settings.reasoning.summary == "auto"
    assert agent.model_settings.extra_args is None
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None)
    assert agent.model_settings.reasoning is None
    assert agent.model_settings.extra_args is None


@needs_agents
def test_extra_body_passthrough_reaches_each_engine(monkeypatch):
    """Caller extras merge last — over the cache key on OpenAI
    destinations — and ride LiteLLM's own kwargs elsewhere, where
    extra_body would plant literal fields providers reject."""
    pytest.importorskip("litellm")
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, cache_key="pageindex-k",
                                     extra_body={"logit_bias": {"1": 5},
                                                 "prompt_cache_key": "mine"})
    assert agent.model_settings.extra_body == {
        "prompt_cache_key": "mine", "logit_bias": {"1": 5}}
    assert agent.model_settings.extra_args is None
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", None, None,
                                     reasoning_effort="low",
                                     extra_body={"top_k": 20})
    assert agent.model_settings.extra_body is None
    assert agent.model_settings.extra_args["top_k"] == 20
    assert agent.model_settings.extra_args["reasoning_effort"] == "low"
    assert "cache_control_injection_points" in agent.model_settings.extra_args
    agent = local_chat._openai_agent(None, "responses", "gpt-test", "sys",
                                     None, None,
                                     extra_body={"service_tier": "flex"})
    assert agent.model_settings.extra_body == {"service_tier": "flex"}


@needs_agents
def test_sampling_knobs_ride_model_settings(client, store_path, fake_model,
                                            monkeypatch):
    """top_p/max_tokens ride ModelSettings fields — the one channel clean
    on every lane (extra_body collides with LitellmModel's explicit
    kwargs). responses' max_output_tokens is the same field's wire name,
    echoed in the envelope."""
    seed_doc(store_path, "pi-a", "report.pdf")
    seen = {}
    real = local_chat._openai_agent

    def spy(*args, **kwargs):
        agent = real(*args, **kwargs)
        seen[args[1]] = agent.model_settings
        return agent

    monkeypatch.setattr(local_chat, "_openai_agent", spy)
    fake_model([[_msg_item("ok")]])
    client.chat_completions("q", top_p=0.9, max_tokens=256)
    assert seen["chat"].top_p == 0.9
    assert seen["chat"].max_tokens == 256
    fake_model([[_msg_item("ok")]])
    result = client.responses("q", max_output_tokens=321)
    assert seen["responses"].max_tokens == 321
    assert result["max_output_tokens"] == 321
    fake_model([[_msg_item("ok")]])
    assert client.responses("q")["max_output_tokens"] is None


@needs_agents
def test_responses_envelope_echoes_reasoning(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_msg_item("ok")]])
    result = client.responses("q", reasoning={"effort": "low"})
    assert result["reasoning"] == {"effort": "low"}
    fake_model([[_msg_item("ok")]])
    assert client.responses("q")["reasoning"] is None


@needs_agents
def test_responses_input_validation(client, fake_model):
    fake_model([])
    for bad in ("", "   ", [], [1], None):
        with pytest.raises(PageIndexAPIError, match="input must be"):
            client.responses(bad)


@needs_agents
def test_doc_id_scopes_tools_to_targeted_documents(client, store_path,
                                                   fake_model):
    """doc_id is enforcement, not just a prompt: name-addressed reads of
    out-of-scope documents fail and browse lists only the targeted set."""
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "payroll.pdf")
    fake = fake_model([
        [_call_item("get_page_content",
                    {"doc_name": "payroll.pdf", "pages": "1"})],
        [_call_item("browse_documents", {}, "call_2")],
        [_msg_item("done")],
    ])
    client.chat_completions("q", doc_id="pi-a")

    def tool_outputs(items):
        return [item["output"] for item in items
                if item.get("type") == "function_call_output"]

    assert "NOT_FOUND" in tool_outputs(fake.inputs[1])[-1]
    browse = json.loads(tool_outputs(fake.inputs[2])[-1])
    assert [doc["name"] for doc in browse["documents"]] == ["report.pdf"]


@needs_agents
def test_empty_doc_id_is_refused(client, store_path):
    """doc_id=[] fails loud on every local surface, like cloud already
    did: washing it to None would mean "everything", and the empty
    allowlist meant "nothing" — an agent confidently reporting the
    documents don't exist, with no signal the scope was empty."""
    seed_doc(store_path, "pi-a", "report.pdf")
    with pytest.raises(PageIndexAPIError, match="doc_id is empty"):
        client.chat_completions("q", doc_id=[])
    with pytest.raises(PageIndexAPIError, match="doc_id is empty"):
        client.as_openai_tools(doc_id=[])
    with pytest.raises(PageIndexAPIError, match="doc_id is empty"):
        client.agent_instructions(doc_id=[])


@needs_agents
def test_openai_model_resolves_provider_prefixes():
    """The chat lane is LiteLLM, full stop — model names mean what LiteLLM
    says they mean, bare names are the openai/ shorthand, and routing
    prefixes never leak as wire model names. responses stays OpenAI-SDK
    native."""
    pytest.importorskip("litellm")
    from agents.extensions.models.litellm_model import LitellmModel
    from agents.models.openai_responses import OpenAIResponsesModel

    model = local_chat._openai_model("chat", "litellm/anthropic/claude-x")
    assert isinstance(model, LitellmModel) and model.model == "anthropic/claude-x"
    model = local_chat._openai_model("chat", "anthropic/claude-x")
    assert isinstance(model, LitellmModel) and model.model == "anthropic/claude-x"
    model = local_chat._openai_model("chat", "gpt-5.2")
    assert isinstance(model, LitellmModel) and model.model == "openai/gpt-5.2"
    model = local_chat._openai_model("chat", "openai/gpt-5.2")
    assert isinstance(model, LitellmModel) and model.model == "openai/gpt-5.2"
    model = local_chat._openai_model("responses", "gpt-5.2")
    assert isinstance(model, OpenAIResponsesModel)
    assert str(model.model) == "gpt-5.2"
    model = local_chat._openai_model("responses", "openai/gpt-5.2")
    assert isinstance(model, OpenAIResponsesModel)
    assert str(model.model) == "gpt-5.2"
    # litellm/ is routing grammar, not a provider: it strips before the
    # provider-prefix guard, so an OpenAI model stays reachable.
    model = local_chat._openai_model("responses", "litellm/gpt-5.2")
    assert isinstance(model, OpenAIResponsesModel)
    assert str(model.model) == "gpt-5.2"


@needs_agents
def test_chat_refuses_unknown_litellm_provider():
    """A HuggingFace-style id (vLLM serving Qwen/...) must fail at build
    time with the openai/ escape, not inside LiteLLM at request time."""
    pytest.importorskip("litellm")
    for name in ("Qwen/Qwen2.5-7B-Instruct", "litellm/Qwen/Qwen2.5-7B-Instruct"):
        with pytest.raises(PageIndexAPIError, match="openai/Qwen"):
            local_chat._openai_model("chat", name)


@needs_agents
def test_responses_refuses_litellm_routed_models(store_path):
    """LiteLLM speaks chat.completions, not /responses — the responses
    protocol must refuse the silent downgrade, at agent-build time and
    before any backend call."""
    for name in ("anthropic/claude-x", "litellm/anthropic/claude-x"):
        with pytest.raises(PageIndexAPIError, match="Responses API"):
            local_chat._openai_model("responses", name)
    client = PageIndexLocalClient(storage_path=store_path,
                                  retrieve_model="anthropic/claude-x")
    with pytest.raises(PageIndexAPIError, match="chat_completions"):
        client.responses("q")


@needs_agents
def test_envelope_model_strips_litellm_routing_prefix(store_path, fake_model):
    """litellm/ is the SDK's routing marker, not a model name — the
    OpenAI-shaped envelopes must report the model the provider serves."""
    seed_doc(store_path, "pi-a", "report.pdf")
    client = PageIndexLocalClient(storage_path=store_path,
                                  retrieve_model="anthropic/claude-x")
    assert client.retrieve_model == "litellm/anthropic/claude-x"
    fake_model([[_msg_item("ok")]])
    result = client.chat_completions("q")
    assert result["model"] == "anthropic/claude-x"
    fake_model([[_msg_item("ok")]])
    chunks = list(client.chat_completions("q", stream=True,
                                          stream_metadata=True))
    assert {c["model"] for c in chunks} == {"anthropic/claude-x"}


@needs_agents
def test_envelope_model_strips_openai_routing_prefix(store_path, fake_model):
    """openai/ is the other routing marker — both OpenAI-shaped envelopes
    must report the name the provider actually serves."""
    seed_doc(store_path, "pi-a", "report.pdf")
    client = PageIndexLocalClient(storage_path=store_path,
                                  retrieve_model="openai/gpt-5.2")
    fake_model([[_msg_item("ok")]])
    result = client.chat_completions("q")
    assert result["model"] == "gpt-5.2"
    fake_model([[_msg_item("ok")]])
    result = client.responses("q")
    assert result["model"] == "gpt-5.2"


@needs_agents
def test_chat_missing_openai_key_fails_loud(monkeypatch):
    """A missing backend credential surfaces as the SDK's own error type,
    like every other precondition on the chat surfaces."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in ("gpt-4o", "openai/gpt-4o"):
        with pytest.raises(PageIndexAPIError, match="OPENAI_API_KEY"):
            local_chat._openai_model("chat", name)


@needs_agents
def test_record_response_status_captures_last_status():
    class _Dumpable:
        def __init__(self, data):
            self._data = data

        def model_dump(self, mode=None):
            return dict(self._data)

    async def create(*args, **kwargs):
        return types.SimpleNamespace(
            status="incomplete",
            incomplete_details=_Dumpable({"reason": "max_output_tokens"}),
            error=None)

    agent = types.SimpleNamespace(model=types.SimpleNamespace(
        _client=types.SimpleNamespace(
            responses=types.SimpleNamespace(create=create))))
    recorded = {}
    local_chat._record_response_status(agent, recorded)
    asyncio.run(agent.model._client.responses.create())
    assert recorded == {"status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "error": None}


@needs_agents
def test_responses_envelope_reports_backend_truncation(client, store_path,
                                                       fake_model):
    """A final turn the backend reports as status "incomplete" must not be
    dressed up as a clean completion."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("cut off mid-answer")]])

    async def create(*args, **kwargs):
        return types.SimpleNamespace(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            error=None)

    fake._client = types.SimpleNamespace(
        responses=types.SimpleNamespace(create=create))
    result = client.responses("q")
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["error"] is None


@needs_agents
def test_responses_envelope_reports_backend_tool_params(client, store_path,
                                                        fake_model):
    """tool_choice / parallel_tool_calls come from the backend's echo —
    the request sends neither, so the envelope must not assume values."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("ok")]])

    async def create(*args, **kwargs):
        return types.SimpleNamespace(status=None, tool_choice="none",
                                     parallel_tool_calls=False)

    fake._client = types.SimpleNamespace(
        responses=types.SimpleNamespace(create=create))
    result = client.responses("q")
    assert result["tool_choice"] == "none"
    assert result["parallel_tool_calls"] is False


@needs_agents
def test_chat_completions_wraps_framework_errors(client, store_path,
                                                 fake_model, monkeypatch):
    """Both chat_completions paths surface engine failures as the SDK's
    own error type, like responses()."""
    from agents.exceptions import ModelBehaviorError
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("never terminal")]])
    fake.no_terminal = True
    with pytest.raises(PageIndexAPIError, match="agent backend failed"):
        list(client.chat_completions("q", stream=True))

    fake = fake_model([[_msg_item("x")]])

    async def boom(*args, **kwargs):
        raise ModelBehaviorError("backend broke")

    monkeypatch.setattr(fake, "get_response", boom)
    with pytest.raises(PageIndexAPIError, match="agent backend failed"):
        client.chat_completions("q")


@needs_agents
def test_responses_stream_wraps_framework_errors(client, store_path,
                                                 fake_model):
    """A backend stream that dies without a terminal event surfaces as the
    SDK's own error type, not a raw openai-agents exception."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("never terminal")]])
    fake.no_terminal = True
    with pytest.raises(PageIndexAPIError, match="agent backend failed"):
        list(client.responses("q", stream=True))


class _TerminalModel(FakeModel):
    """Engine-faithful backend terminal: openai-agents yields the
    response.failed/response.incomplete lifecycle event, then raises."""
    terminal = "incomplete"

    async def stream_response(self, system_instructions, input,
                              model_settings, tools, output_schema,
                              handoffs, tracing, **kwargs):
        from agents.exceptions import ModelBehaviorError
        from openai.types.responses import (Response, ResponseFailedEvent,
                                            ResponseIncompleteEvent,
                                            ResponseTextDeltaEvent)
        from openai.types.responses.response import IncompleteDetails
        from openai.types.responses.response_error import ResponseError
        self._record(system_instructions, input)
        yield ResponseTextDeltaEvent(
            type="response.output_text.delta", delta="partial ",
            content_index=0, item_id="item_x", output_index=0,
            logprobs=[], sequence_number=1)
        response = Response(
            id="resp_fake", created_at=0.0, model="fake", object="response",
            output=[], parallel_tool_calls=False, tool_choice="auto",
            tools=[], status=self.terminal,
            incomplete_details=(IncompleteDetails(reason="max_output_tokens")
                                if self.terminal == "incomplete" else None),
            error=(ResponseError(code="server_error", message="boom")
                   if self.terminal == "failed" else None))
        event_type = (ResponseIncompleteEvent if self.terminal == "incomplete"
                      else ResponseFailedEvent)
        yield event_type(type=f"response.{self.terminal}", response=response,
                         sequence_number=2)
        raise ModelBehaviorError(f"terminal: {self.terminal}")


@needs_agents
@pytest.mark.parametrize("terminal", ["incomplete", "failed"])
def test_responses_stream_backend_terminal_states_are_events(
        client, store_path, monkeypatch, terminal):
    """response.failed / response.incomplete are protocol terminal states,
    not engine failures: the stream must end with the honest terminal
    event carrying the backend's status, not raise away the run."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = _TerminalModel([[]])
    fake.terminal = terminal
    monkeypatch.setattr(local_chat, "_openai_model",
                        lambda protocol, model_name, backend=None: fake)
    events = list(client.responses("q", stream=True))
    assert events[0]["type"] == "response.output_text.delta"
    last = events[-1]
    assert last["type"] == f"response.{terminal}"
    assert last["response"]["status"] == terminal
    if terminal == "incomplete":
        assert (last["response"]["incomplete_details"]
                == {"reason": "max_output_tokens"})
    else:
        assert last["response"]["error"]["message"] == "boom"
    numbers = [event["sequence_number"] for event in events]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)


@needs_agents
def test_provider_errors_wrap_as_sdk_errors(client, store_path, fake_model,
                                            monkeypatch):
    """Raw provider exceptions (network, auth, rate limit) surface as
    PageIndexAPIError on every OpenAI-engine path, never as openai types."""
    import openai
    seed_doc(store_path, "pi-a", "report.pdf")
    request = httpx.Request("POST", "https://backend.test")

    async def conn_err(*args, **kwargs):
        raise openai.APIConnectionError(request=request)

    async def conn_err_stream(*args, **kwargs):
        raise openai.APIConnectionError(request=request)
        yield  # unreached: makes this an async generator

    fake = fake_model([[_msg_item("x")], [_msg_item("x")]])
    monkeypatch.setattr(fake, "get_response", conn_err)
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        client.chat_completions("q")
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        client.responses("q")
    monkeypatch.setattr(fake, "stream_response", conn_err_stream)
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        list(client.chat_completions("q", stream=True))
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        list(client.responses("q", stream=True))


@needs_anthropic
def test_messages_provider_errors_wrap_as_sdk_errors(client, store_path,
                                                     monkeypatch):
    """Anthropic transport errors surface as PageIndexAPIError on both
    messages() paths, never as anthropic types."""
    seed_doc(store_path, "pi-a", "report.pdf")

    def handler(request):
        return httpx.Response(429, json={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "slow down"}})

    fake = anthropic.Anthropic(
        api_key="test", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None: fake)
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        client.messages("q", model="claude-test")
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        list(client.messages("q", model="claude-test", stream=True))


@needs_agents
def test_chat_stream_close_at_opening_chunk_cancels_run(client, store_path,
                                                        fake_model,
                                                        monkeypatch):
    """GeneratorExit at the opening chunk must still cancel the agent task:
    the first yield sits inside the generator's try/finally."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_msg_item("never")]])
    fake.block_from = 1  # turn 1 hangs until cancelled
    captured = {}

    def capture(agen_factory):
        captured["factory"] = agen_factory
        return iter(())  # drive the async generator by hand instead

    monkeypatch.setattr(local_chat, "_stream_sync", capture)
    client.chat_completions("q", stream=True, stream_metadata=True)

    async def drive():
        agen = captured["factory"]()
        first = await agen.__anext__()
        assert first["choices"][0]["delta"] == {"role": "assistant",
                                                "content": ""}
        await agen.aclose()
        deadline = asyncio.get_running_loop().time() + 2.0
        pending = []
        while asyncio.get_running_loop().time() < deadline:
            pending = [task for task in asyncio.all_tasks()
                       if task is not asyncio.current_task()
                       and not task.done()]
            if not pending:
                break
            await asyncio.sleep(0.01)
        return pending

    assert asyncio.run(drive()) == []


@needs_agents
def test_stream_abandonment_cancels_pending_turn(client, store_path,
                                                 fake_model, monkeypatch):
    """Closing the iterator cancels the run even while it is awaiting the
    backend: the blocked turn is torn down (pump thread exits) instead of
    running — and billing — to completion in the background. The pump
    thread is tracked directly — a process-global thread count would be
    flaky against litellm's background threads."""
    import threading
    seed_doc(store_path, "pi-a", "report.pdf")
    pumps = []
    real_thread = threading.Thread

    class _Tracking(real_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if getattr(kwargs.get("target"), "__name__", "") == "pump":
                pumps.append(self)

    monkeypatch.setattr(threading, "Thread", _Tracking)
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.block_from = 2  # turn 2 hangs until cancelled
    stream = client.chat_completions([{"role": "user", "content": "q"}],
                                     stream=True, stream_metadata=True)
    next(stream)  # the opening role chunk
    stream.close()
    assert len(pumps) == 1
    pumps[0].join(timeout=3.0)
    assert not pumps[0].is_alive()
    assert fake.deltas_emitted == 0  # turn 2 never produced output


@needs_anthropic
def test_messages_max_tokens_default_resolves_per_model(client, fake_anthropic):
    """The wire-required budget must not exceed the model's ceiling: the
    claude-3 generation caps output at 4096."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-3-opus-20240229")
    assert calls[0]["max_tokens"] == 4096
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-sonnet-4-5")
    assert calls[0]["max_tokens"] == 8192
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-3-opus-20240229", max_tokens=1234)
    assert calls[0]["max_tokens"] == 1234


@needs_anthropic
def test_messages_thinking_passes_through(client, fake_anthropic):
    """Anthropic-native thinking config, forwarded verbatim; unset sends
    nothing so the backend default applies."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-sonnet-4-5",
                    thinking={"type": "adaptive"})
    assert calls[0]["thinking"] == {"type": "adaptive"}
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-sonnet-4-5")
    assert "thinking" not in calls[0]


@needs_anthropic
def test_messages_extra_body_merges_into_the_wire_body(client, fake_anthropic):
    """The anthropic SDK merges extra_body keys into the request JSON —
    asserted on the captured wire body, not the SDK call."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-sonnet-4-5",
                    extra_body={"service_tier": "auto"})
    assert calls[0]["service_tier"] == "auto"
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.messages("q", model="claude-sonnet-4-5")
    assert "service_tier" not in calls[0]


@needs_anthropic
def test_messages_tool_error_flagged_and_scoped(client, store_path,
                                                fake_anthropic):
    """Through the real runner: a failed call reaches Claude as a
    tool_result with is_error true, and doc_id scoping makes out-of-scope
    documents unreachable by name."""
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "secret.pdf")
    calls = fake_anthropic([
        _anthropic_message([{"type": "tool_use", "id": "tu_1",
                             "name": "get_document",
                             "input": {"doc_name": "secret.pdf"}}],
                           "tool_use"),
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    client.messages("q", model="claude-test", doc_id="pi-a")
    tool_result = calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result.get("is_error") is True
    assert "NOT_FOUND" in json.dumps(tool_result["content"])


@needs_anthropic
def test_messages_envelope_json_and_no_internal_fields(client, store_path,
                                                       fake_anthropic):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_anthropic([
        _anthropic_message([_anthropic_tool_use()], "tool_use"),
        _anthropic_message([{"type": "text", "text": "The answer"}],
                           "end_turn"),
    ])
    result = client.messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100)
    dumped = json.dumps(result)  # the whole envelope must serialize
    assert "parsed_output" not in dumped


@needs_anthropic
def test_messages_max_turns_truncation_round_trippable(client, store_path,
                                                       fake_anthropic):
    """On a max_turns cut the runner has already appended the final turn —
    no duplicate append, and the history stays valid for continuation."""
    seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message([_anthropic_tool_use()], "tool_use"),
        _anthropic_message([_anthropic_tool_use("tu_2")], "tool_use"),
    ])
    result = client.messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100, max_turns=1)
    assert len(calls) == 1
    assert result["stop_reason"] == "tool_use"
    roles = [message["role"] for message in result["messages"]]
    assert roles == ["assistant", "user"]  # tool_use, tool_result — no dup
    assert json.dumps(result).count('"tu_1"') == \
        json.dumps(result["messages"][0]).count('"tu_1"') \
        + json.dumps(result["messages"][1]).count('"tu_1"') \
        + json.dumps(result["content"]).count('"tu_1"')
    json.dumps(result)


@needs_anthropic
def test_messages_tool_use_cut_by_max_tokens_not_duplicated(client,
                                                            store_path,
                                                            fake_anthropic):
    """A max_tokens turn with complete tool_use blocks still executes and
    is appended by the runner — keying the re-append guard on stop_reason
    duplicated the tool_use id and broke verbatim continuation."""
    seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message([_anthropic_tool_use()], "max_tokens"),
        _anthropic_message([_anthropic_tool_use("tu_2")], "tool_use"),
    ])
    result = client.messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100, max_turns=1)
    assert len(calls) == 1
    assert result["stop_reason"] == "max_tokens"
    roles = [message["role"] for message in result["messages"]]
    assert roles == ["assistant", "user"]  # tool_use, tool_result — no dup
    assert json.dumps(result["messages"]).count('"tu_1"') == 2  # use + result


@needs_anthropic
def test_messages_refusal_with_tool_use_stays_appendable(client, store_path,
                                                         fake_anthropic):
    """A refusal turn is never executed by the runner; its tool_use blocks
    have no tool_result and must not enter the appendable history."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_anthropic([
        _anthropic_message([{"type": "text", "text": "I can't help."},
                            _anthropic_tool_use()], "refusal"),
    ])
    result = client.messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100)
    assert result["stop_reason"] == "refusal"
    message, = result["messages"]
    assert message["role"] == "assistant"
    assert [block["type"] for block in message["content"]] == ["text"]
    assert message["content"][0]["text"] == "I can't help."
    # The envelope's own content still carries the full turn verbatim.
    assert [block["type"] for block in result["content"]] \
        == ["text", "tool_use"]


@needs_anthropic
def test_messages_default_cap(client, store_path, fake_anthropic):
    seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message([_anthropic_tool_use(f"tu_{index}")], "tool_use")
        for index in range(30)
    ])
    result = client.messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100)
    assert len(calls) == 10  # bounded like the OpenAI surfaces
    assert result["stop_reason"] == "tool_use"
    json.dumps(result)


@needs_anthropic
def test_messages_edge_validation(client, store_path, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    client.messages([{"role": "user", "content": "q"}], model="claude-test",
                    max_tokens=100, system="   ")
    assert all(block["text"].strip() for block in calls[0]["system"])
    with pytest.raises(PageIndexAPIError, match="message dicts"):
        client.messages(["not a dict"], model="claude-test", max_tokens=100)
    with pytest.raises(PageIndexAPIError, match="doc_id"):
        client.messages([{"role": "user", "content": "q"}],
                        model="claude-test", max_tokens=100, doc_id=123)


# ── backend + extra_headers: the chat doors ──

@needs_agents
def test_backend_connection_reaches_each_engine(monkeypatch):
    """api_key/base_url ride each engine's client construction; the
    LiteLLM lane's remaining keys ride its call kwargs; a backend key
    satisfies the responses lane's missing-key check."""
    pytest.importorskip("litellm")
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", None, None,
                                     backend={"api_key": "k1",
                                              "api_base": "http://lb",
                                              "api_version": "v9"})
    assert agent.model.api_key == "k1"
    assert agent.model.base_url == "http://lb"
    assert agent.model_settings.extra_args["api_version"] == "v9"
    assert "api_key" not in agent.model_settings.extra_args
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = local_chat._openai_agent(None, "responses", "gpt-test", "sys",
                                     None, None, backend={"api_key": "k2"})
    assert agent.model._client.api_key == "k2"
    # the LiteLLM endpoint spelling works on the SDK-constructed door too
    agent = local_chat._openai_agent(None, "responses", "gpt-test", "sys",
                                     None, None,
                                     backend={"api_key": "k3",
                                              "api_base": "http://rb"})
    assert str(agent.model._client.base_url).rstrip("/") == "http://rb"


def test_merged_backend_precedence():
    from types import SimpleNamespace
    stub = SimpleNamespace(chat_backend={"api_key": "a", "api_version": "v1"})
    assert local_chat._merged_backend(stub, {"api_key": "b"}) == {
        "api_key": "b", "api_version": "v1"}
    assert local_chat._merged_backend(SimpleNamespace(), None) is None


@needs_anthropic
def test_messages_backend_merges_and_reaches_the_client(client, fake_anthropic,
                                                        monkeypatch):
    real = local_chat._anthropic_client({"api_key": "kk",
                                         "base_url": "http://x"})
    assert real.api_key == "kk"
    assert str(real.base_url).rstrip("/") == "http://x"
    real = local_chat._anthropic_client({"api_key": "kk",
                                         "api_base": "http://y"})
    assert str(real.base_url).rstrip("/") == "http://y"

    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    fixture_client = local_chat._anthropic_client
    seen = {}
    monkeypatch.setattr(
        local_chat, "_anthropic_client",
        lambda backend=None: (seen.setdefault("backend", backend),
                              fixture_client())[1])
    client.chat_backend = {"base_url": "http://cb"}
    client.messages("q", model="claude-sonnet-4-5", backend={"api_key": "z"})
    assert seen["backend"] == {"base_url": "http://cb", "api_key": "z"}


@needs_anthropic
def test_messages_bad_backend_wraps_like_the_other_doors():
    with pytest.raises(PageIndexAPIError,
                       match="Anthropic backend is not configured"):
        local_chat._anthropic_client({"no_such_param": 1})


@needs_agents
def test_extra_headers_ride_model_settings(monkeypatch):
    """Both openai-agents doors merge ModelSettings.extra_headers into
    their requests (wire-probed: LiteLLM's chatcmpl adapters forward
    custom headers; its anthropic adapter owns anthropic-beta only)."""
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None,
                                     extra_headers={"x-beta": "1"})
    assert agent.model_settings.extra_headers == {"x-beta": "1"}
    agent = local_chat._openai_agent(None, "responses", "gpt-test", "sys",
                                     None, None,
                                     extra_headers={"x-beta": "2"})
    assert agent.model_settings.extra_headers == {"x-beta": "2"}
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None)
    assert agent.model_settings.extra_headers is None


@needs_anthropic
def test_messages_extra_headers_reach_the_wire(client, monkeypatch):
    import anthropic
    seen = {}

    def handler(request):
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx.Response(200, json=_anthropic_message(
            [{"type": "text", "text": "ok"}], "end_turn"))

    fake = anthropic.Anthropic(api_key="t", http_client=httpx.Client(
        transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None: fake)
    client.messages("q", model="claude-sonnet-4-5",
                    extra_headers={"anthropic-beta": "context-1m-2025"})
    assert seen["beta"] == "context-1m-2025"
