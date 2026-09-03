"""Local chat surfaces: three protocols over fake backends — no network,
no LLM keys. Tool execution runs for real against a seeded local store."""
import asyncio
import json
import sys
import types

import httpx  # via the hard `openai` dependency
import pytest

try:  # anthropic >= 1.0 validates http_client against httpx2
    import httpx2 as anthropic_httpx
except ImportError:  # older anthropic rides classic httpx
    anthropic_httpx = httpx

import pageindex.local_chat as local_chat
from pageindex import (PageIndexAPIError, PageIndexClient,
                       PageIndexCloudClient, PageIndexLocalClient)
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
        for index, piece in enumerate(getattr(self, "thinking_pieces", ())):
            from openai.types.responses import (
                ResponseReasoningSummaryTextDeltaEvent,
                ResponseReasoningTextDeltaEvent)
            sequence += 1
            # litellm (every chat-protocol backend) delivers thinking as
            # summary deltas; alternate so both accepted variants stay
            # covered, production shape first
            if index % 2:
                yield ResponseReasoningTextDeltaEvent(
                    type="response.reasoning_text.delta", delta=piece,
                    content_index=0, item_id="rs_1", output_index=0,
                    sequence_number=sequence)
            else:
                yield ResponseReasoningSummaryTextDeltaEvent(
                    type="response.reasoning_summary_text.delta",
                    delta=piece, item_id="rs_1", output_index=0,
                    summary_index=0, sequence_number=sequence)
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
    with pytest.raises(PageIndexAPIError, match="managed chat endpoint"):
        client.chat_completions([{"role": "user", "content": "x"}],
                                enable_citations=True)
    with pytest.raises(PageIndexAPIError, match="chat\\(protocol="):
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


def test_cloud_guards(monkeypatch):
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    # empty is unset, as on the local lane — the managed chat serves it
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kw: {"choices": [{"message": {"content": "ok"}}]})
    assert cloud.chat("x", model="", reasoning_effort="", extra_body={},
                      extra_headers={}, backend={}) == "ok"
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}], model="m")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               reasoning_effort="low")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               extra_body={"service_tier": "auto"})
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}], top_p=0.9)
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               max_tokens=256)
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat("x", reasoning_effort="low")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               backend={"api_key": "k"})
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               extra_headers={"x-beta": "1"})
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat("x", protocol="responses")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat([{"role": "user", "content": "x"}],
                   protocol="messages", model="m")
    # the hidden doors refuse the same way: door X is chat(protocol=X)
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud._responses("x")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud._messages("x", model="m")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat("x", instructions="be brief")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        cloud.chat("x", max_turns=2)


@needs_agents
def test_anthropic_routed_models_mark_managed_prefix_for_cache(
        client, store_path, fake_model):
    fake_model([[_msg_item("ok")]])
    from pageindex.local_chat import _openai_agent
    marked = {"cache_control_injection_points": [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1}]}
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


@needs_agents
def test_chat_process_weaves_thinking_and_tools(client, store_path,
                                                fake_model):
    """show_process=True keeps the plain text stream but weaves the run in:
    a "[thinking] " section per thinking burst, a "[tool_call] name args"
    line per call with its clipped result, and the answer unlabeled."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.thinking_pieces = ("Need the ", "report")
    text = "".join(client.chat("What status?", stream=True, show_process=True))
    assert text.startswith("[thinking] Need the report")
    assert '\n\n[tool_call] get_document {"doc_name": "report.pdf"}' in text
    assert '"report.pdf"}\n[tool_result] get_document: ' in text
    assert text.endswith("\n\nThe answer")
    assert "[answer]" not in text


def test_chat_process_requires_stream(client):
    with pytest.raises(PageIndexAPIError, match="requires stream=True"):
        client.chat("q", show_process=True)
    # falsy-but-not-False means ON (the ruled falsy-{} trap); the message
    # must say how to turn it off, not claim the caller passed True
    with pytest.raises(PageIndexAPIError,
                       match="only show_process=False"):
        client.chat("q", show_process=0)


def _cloud_chunk(content=None, meta=None, choices=True):
    chunk = {"id": "chatcmpl-x", "object": "chat.completion.chunk"}
    if choices:
        delta = {"content": content} if content is not None else {}
        chunk["choices"] = [{"index": 0, "delta": delta,
                             "finish_reason": None}]
    if meta:
        chunk["block_metadata"] = meta
    return chunk


def _managed_cloud_with_tool_stream(monkeypatch):
    """A cloud client whose managed stream serves the live wire shape:
    block_metadata-tagged text, a tool call, and untagged tail chunks."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    seen = {}

    def fake_cc(**kwargs):
        seen.update(kwargs)
        return iter([
            _cloud_chunk("", {"type": "text_block_start", "block_index": 1}),
            _cloud_chunk("Let me check.", {"type": "text", "block_index": 1}),
            _cloud_chunk(None, {"type": "text_stop", "block_index": 1}),
            _cloud_chunk(None, {"type": "mcp_tool_use_start",
                                "block_index": 2,
                                "tool_name": "get_document",
                                "server_name": "pageindex"}),
            _cloud_chunk('{"doc_name": ', {"type": "tool_use",
                                           "block_index": 2}),
            _cloud_chunk('"report.pdf"}', {"type": "tool_use",
                                           "block_index": 2}),
            _cloud_chunk(None, {"type": "tool_use_stop", "block_index": 2}),
            _cloud_chunk("", {"type": "text_block_start", "block_index": 3}),
            _cloud_chunk("The answer", {"type": "text", "block_index": 3}),
            _cloud_chunk(None, {"type": "text_stop", "block_index": 3}),
            _cloud_chunk(None),                    # finish_reason chunk
            _cloud_chunk(choices=False),           # usage chunk
        ])

    monkeypatch.setattr(cloud._api, "chat_completions", fake_cc)
    return cloud, seen


def test_chat_process_managed_weaves_what_the_wire_serves(monkeypatch):
    """Managed streams weave the endpoint's block_metadata: tool-call
    lines with name + accumulated arguments; no thinking, no results."""
    cloud, seen = _managed_cloud_with_tool_stream(monkeypatch)
    text = "".join(cloud.chat("What status?", stream=True))
    assert seen["stream_metadata"] is True
    assert text == ('Let me check.\n\n'
                    '[tool_call] get_document {"doc_name": "report.pdf"}\n\n'
                    'The answer')
    cloud, _ = _managed_cloud_with_tool_stream(monkeypatch)
    assert "".join(cloud.chat("q", stream=True, show_process=True)) == text
    cloud, _ = _managed_cloud_with_tool_stream(monkeypatch)
    no_calls = "".join(cloud.chat("q", stream=True,
                                  show_process={"tool_call": False}))
    assert no_calls == "Let me check.The answer"


def test_chat_process_managed_off_is_clean_answer(monkeypatch):
    """show_process=False on managed: answer text only — the tool-call
    JSON the wire interleaves into delta.content must not leak in."""
    cloud, _ = _managed_cloud_with_tool_stream(monkeypatch)
    plain = "".join(cloud.chat("q", stream=True, show_process=False))
    assert plain == "Let me check.The answer"
    assert "doc_name" not in plain


def test_chat_process_managed_open_block_never_leaks(monkeypatch):
    """Inside an open tool block nothing is answer text: argument chunks
    under an unexpected tag (or none) accumulate into the call instead
    of leaking into the show_process=False answer."""
    def cloud_with(tag):
        cloud = PageIndexCloudClient(api_key="pi-test-key")
        monkeypatch.setattr(cloud._api, "chat_completions", lambda **kw: iter([
            _cloud_chunk(None, {"type": "mcp_tool_use_start",
                                "tool_name": "get_document"}),
            _cloud_chunk('{"doc_name": ', {"type": "tool_use"}),
            _cloud_chunk('"report.pdf"}', tag),
            _cloud_chunk(None, {"type": "tool_use_stop"}),
            _cloud_chunk("The answer"),
        ]))
        return cloud

    for tag in ({"type": "input_json_delta"}, None):
        plain = "".join(cloud_with(tag).chat("q", stream=True,
                                             show_process=False))
        assert plain == "The answer"
        woven = "".join(cloud_with(tag).chat("q", stream=True))
        assert '[tool_call] get_document {"doc_name": "report.pdf"}' in woven


def test_chat_process_managed_non_string_argument_chunk(monkeypatch):
    """A non-string delta.content inside a tool block degrades to a
    raw-string argument instead of killing the stream."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    monkeypatch.setattr(cloud._api, "chat_completions", lambda **kw: iter([
        _cloud_chunk(None, {"type": "mcp_tool_use_start",
                            "tool_name": "get_document"}),
        _cloud_chunk({"partial_json": "{"}, {"type": "tool_use"}),
        _cloud_chunk(None, {"type": "tool_use_stop"}),
        _cloud_chunk("The answer"),
    ]))
    text = "".join(cloud.chat("q", stream=True))
    assert "[tool_call] get_document" in text
    assert text.endswith("The answer")


@needs_agents
def test_chat_process_dict_selects_parts(client, store_path, fake_model):
    """Each key hides exactly its own line kind; omitted keys default on,
    and {} means all defaults, not "off" (the falsy-dict trap)."""
    def run(process):
        seed_doc(store_path, "pi-a", "report.pdf")
        fake = fake_model([
            [_call_item("get_document", {"doc_name": "report.pdf"})],
            [_msg_item("The answer")],
        ])
        fake.thinking_pieces = ("Need the report",)
        return "".join(client.chat("What status?", stream=True,
                                   show_process=process))

    no_thinking = run({"thinking": False})
    assert "[thinking]" not in no_thinking
    assert "[tool_call] get_document" in no_thinking
    assert "[tool_result] get_document: " in no_thinking

    no_calls = run({"tool_call": False})
    assert "[tool_call]" not in no_calls
    assert "\n\n[tool_result] get_document: " in no_calls  # results stand alone
    assert "[thinking] Need the report" in no_calls

    calls_only = run({"tool_result": False})
    assert "[tool_call] get_document" in calls_only
    assert "[tool_result]" not in calls_only

    assert run({}) == run(True)


@needs_agents
def test_chat_process_max_chars_caps_lines(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    text = "".join(client.chat("What status?", stream=True,
                               show_process={"max_chars": 10}))
    line = next(ln for ln in text.splitlines()
                if ln.startswith("[tool_result] "))
    body = line.split(": ", 1)[1]
    assert body[10:].startswith("... (+")


@needs_agents
def test_chat_stream_events_typed_sequence(client, store_path, fake_model):
    """.events is the typed view: full data, parsed arguments, no clip."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.thinking_pieces = ("Need the report",)
    events = list(client.chat("What status?", stream=True).events)
    kinds = [ev["type"] for ev in events]
    assert kinds == ["thinking", "tool_call", "tool_result",
                     "thinking", "answer", "answer"]
    call = events[1]
    assert call["name"] == "get_document"
    assert call["arguments"] == {"doc_name": "report.pdf"}  # parsed
    assert call["call_id"] == "call_1"
    result = events[2]
    assert result["name"] == "get_document"
    assert result["call_id"] == "call_1"
    assert "... (+" not in str(result["output"])  # never clipped
    assert '"next_steps"' in result["output"]
    assert "".join(ev["delta"] for ev in events
                   if ev["type"] == "answer") == "The answer"


@needs_agents
def test_chat_stream_supports_next_and_one_view(client, store_path,
                                                fake_model):
    fake_model([[_msg_item("The answer")]])
    stream = client.chat("q", stream=True)
    assert next(stream) == "The "  # iterator protocol survives the wrapper
    with pytest.raises(PageIndexAPIError, match="one view"):
        next(stream.events)
    fake_model([[_msg_item("The answer")]])
    stream = client.chat("q", stream=True)
    assert next(stream.events)["type"] == "answer"
    with pytest.raises(PageIndexAPIError, match="one view"):
        next(stream)


@needs_agents
def test_chat_stream_events_read_is_inert(client, store_path, fake_model):
    """Reading .events claims nothing — only consuming does. Debugger
    panes, hasattr and getattr probing must not poison the text view."""
    fake_model([[_msg_item("The answer")]])
    stream = client.chat("q", stream=True)
    assert hasattr(stream, "events")  # introspection, not consumption
    stream.events
    assert "".join(stream) == "The answer"


def test_chat_stream_events_refusal_waits_for_consumption(monkeypatch):
    """On managed, .events read is inert — getattr(stream, 'events',
    None) must not explode — and the refusal raises on first
    consumption, leaving the text view usable."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kwargs: iter([_cloud_chunk("x")]))
    stream = cloud.chat("q", stream=True)
    events = getattr(stream, "events", None)  # the standard probing idiom
    assert events is not None
    with pytest.raises(PageIndexAPIError, match="managed chat endpoint"):
        next(events)
    assert "".join(stream) == "x"


@needs_agents
def test_chat_stream_shows_process_by_default(client, store_path,
                                              fake_model):
    """The text view weaves the process unless show_process=False."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.thinking_pieces = ("Need the report",)
    text = "".join(client.chat("What status?", stream=True))
    assert "[thinking] Need the report" in text
    assert "[tool_call] get_document" in text
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.thinking_pieces = ("Need the report",)
    plain = "".join(client.chat("What status?", stream=True,
                                show_process=False))
    assert plain == "The answer"


def test_weave_close_closes_source():
    closed = {}

    def source():
        try:
            yield {"type": "answer", "delta": "a"}
            yield {"type": "answer", "delta": "b"}
        finally:
            closed["yes"] = True

    woven = local_chat._weave(source(), None)
    assert next(woven) == "a"
    woven.close()
    assert closed.get("yes") is True


@needs_agents
def test_chat_stream_close_stops_the_run(client, store_path, fake_model):
    fake_model([[_msg_item("The answer")]])
    stream = client.chat("q", stream=True)
    assert next(stream) == "The "
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)


@needs_agents
def test_chat_stream_closed_before_consumption_stays_dead(client, store_path,
                                                          fake_model):
    """close() on an unconsumed stream: the run must never start — a
    later next() is StopIteration, .events is empty, the model untouched."""
    fake = fake_model([[_msg_item("The answer")]])
    stream = client.chat("q", stream=True)
    stream.close()
    with pytest.raises(StopIteration):
        next(stream)
    assert fake.inputs == []
    fake = fake_model([[_msg_item("The answer")]])
    stream = client.chat("q", stream=True)
    stream.close()
    assert list(stream.events) == []
    assert fake.inputs == []


def test_chat_stream_managed_cloud(monkeypatch):
    """Old-wire chunks (no block_metadata) are answer text under any
    show_process; .events needs the in-process agent."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    monkeypatch.setattr(
        cloud._api, "chat_completions",
        lambda **kwargs: iter([_cloud_chunk("cloud "),
                               _cloud_chunk("answer")]))
    assert list(cloud.chat("q", stream=True)) == ["cloud ", "answer"]
    with pytest.raises(PageIndexAPIError, match="managed chat endpoint"):
        next(cloud.chat("q", stream=True).events)


def test_chat_process_config_chokes(client):
    for bad, match in [
        ({"thinkng": False}, "Unknown show_process key"),
        ("thinking", "show_process must be True, False, or a dict"),
        ({"max_chars": True}, "positive int"),
        ({"max_chars": 0}, "positive int"),
        ({"thinking": 1}, "must be a bool"),
        ({1: True, "foo": 1}, "Unknown show_process key"),
    ]:
        with pytest.raises(PageIndexAPIError, match=match):
            client.chat("q", stream=True, show_process=bad)


def test_chat_process_blank_chat_model_refuses(client):
    client.chat_model = None
    with pytest.raises(PageIndexAPIError, match="chat_model is empty"):
        client.chat("q", stream=True, show_process=True)


def test_clip_flattens_and_caps():
    assert local_chat._clip("a\n  b\tc") == "a b c"
    assert local_chat._clip("x" * 250) == "x" * 200 + "... (+50 chars)"


@needs_agents
def test_chat_process_parallel_calls_pair_results(client, store_path,
                                                  fake_model):
    """A result nests only under its own call line; parallel-call results
    stand alone with their call's arguments echoed."""
    seed_doc(store_path, "pi-a", "report.pdf")
    seed_doc(store_path, "pi-b", "other.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"}),
         _call_item("get_document", {"doc_name": "other.pdf"},
                    call_id="call_2")],
        [_msg_item("The answer")],
    ])
    text = "".join(client.chat("What status?", stream=True,
                               show_process=True))
    # no result rides directly under a call that is not its own — the
    # bare paired form is absent, every result echoes its arguments
    assert "\n[tool_result] get_document: " not in text
    assert '\n\n[tool_result] get_document {"doc_name": "report.pdf"}: ' in text
    assert '\n[tool_result] get_document {"doc_name": "other.pdf"}: ' in text
    assert '\n\n[tool_result] get_document {"doc_name": "other.pdf"}' not in text
    for doc in ("report.pdf", "other.pdf"):
        line = next(ln for ln in text.splitlines()
                    if ln.startswith(f'[tool_result] get_document {{"doc_name": '
                                     f'"{doc}"}}: '))
        assert f'"{doc}"' in line.split("}: ", 1)[1]


@needs_agents
def test_chat_stream_drops_empty_deltas(client, store_path, fake_model):
    """Mid-stream empty deltas carry nothing and would only flip weave
    sections; the event source drops them, like chat_completions."""
    fake = fake_model([[_msg_item("The answer")]])
    fake.pieces = ("The ", "", "answer")
    assert list(client.chat("q", stream=True,
                            show_process=False)) == ["The ", "answer"]
    fake = fake_model([[_msg_item("The answer")]])
    fake.pieces = ("The ", "", "answer")
    fake.thinking_pieces = ("hm", "", "m")
    deltas = [ev["delta"] for ev in client.chat("q", stream=True).events
              if ev["type"] in ("answer", "thinking")]
    assert "" not in deltas


def test_chat_process_managed_validates_before_request(monkeypatch):
    """A bad show_process must choke before the billed request is sent."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    calls = []
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kwargs: calls.append(kwargs) or iter(()))
    with pytest.raises(PageIndexAPIError, match="Unknown show_process key"):
        cloud.chat("q", stream=True, show_process={"thinkng": False})
    assert calls == []


# ── responses ──

@needs_agents
def test_responses_end_to_end(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client._responses("What status?")
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
    result = client._responses("What status?")

    second = fake_model([[_msg_item("Done")]])
    follow_up = ([{"role": "user", "content": "What status?"}]
                 + result["items"]
                 + [{"role": "user", "content": "and now?"}])
    client._responses(follow_up)
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
    result = client._responses("What status?", doc_id="pi-a")

    second = fake_model([[_msg_item("Done")]])
    follow_up = ([{"role": "user", "content": "What status?"}]
                 + result["items"]
                 + [{"role": "user", "content": "and now?"}])
    client._responses(follow_up, doc_id="pi-a")
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

    def spy(model_name, instructions, doc_id, items):
        key = real(model_name, instructions, doc_id, items)
        keys.append(key)
        return key

    monkeypatch.setattr(local_chat, "_conversation_cache_key", spy)

    fake_model([[_msg_item("a")]])
    result = client._responses("What is the CAGR?", doc_id="pi-a")
    fake_model([[_msg_item("b")]])
    client._responses("Summarize section 3.", doc_id="pi-a")
    assert keys[0] != keys[1]  # unrelated conversations never pool

    fake_model([[_msg_item("c")]])
    follow_up = ([{"role": "user", "content": "What is the CAGR?"}]
                 + result["items"]
                 + [{"role": "user", "content": "and now?"}])
    client._responses(follow_up, doc_id="pi-a")
    assert keys[2] == keys[0]  # a continuation keeps its conversation's key

    fake_model([[_msg_item("d")]])
    client.chat_completions("What is the CAGR?", doc_id="pi-a")
    fake_model([[_msg_item("e")]])
    client.chat_completions("Summarize section 3.", doc_id="pi-a")
    assert keys[3] != keys[4]  # same property on the chat surface

    seed_doc(store_path, "pi-b", "contract.pdf")
    fake_model([[_msg_item("f")]])
    client._responses("What is the CAGR?", doc_id="pi-b")
    assert keys[5] != keys[0]  # same opener, different doc: no pooling


@needs_agents
def test_responses_stream_passthrough(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    events = list(client._responses("q", stream=True))
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
    events = list(client._responses("q", stream=True))
    created = [e for e in events if e["type"] == "response.created"]
    assert len(created) == 1 and events[0] is created[0]
    assert events[0]["sequence_number"] == 1
    terminal = events[-1]
    assert terminal["type"] == "response.completed"
    assert created[0]["response"]["id"] == terminal["response"]["id"]
    assert (created[0]["response"]["created_at"]
            == terminal["response"]["created_at"])  # one timestamp, not two
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
    result = client._responses("What status?")
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
    events = list(client._responses("q", stream=True))
    assert events
    for event in events:
        adapter.validate_python(event)


# ── messages (Anthropic engine) ──

try:
    import anthropic
    _HAS_ANTHROPIC = True
    # anthropic 1.1 made every non-tool_use stop reason terminal: the runner
    # no longer executes tool_use blocks from a max_tokens-cut turn.
    try:
        _ANTHROPIC_RUNS_CUT_TOOL_TURNS = tuple(
            int(piece) for piece in anthropic.__version__.split(".")[:2]
        ) < (1, 1)
    except ValueError:
        _ANTHROPIC_RUNS_CUT_TOOL_TURNS = False  # unparseable: assume current
except ImportError:
    _HAS_ANTHROPIC = False
    _ANTHROPIC_RUNS_CUT_TOOL_TURNS = False

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
                return anthropic_httpx.Response(
                    200, content=body.encode(),
                    headers={"content-type": "text/event-stream"})
            return anthropic_httpx.Response(200, json=body)

        fake = anthropic.Anthropic(
            api_key="test",
            http_client=anthropic_httpx.Client(
                transport=anthropic_httpx.MockTransport(handler)))
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
    result = client._messages([{"role": "user", "content": "What status?"}],
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
    client._messages([{"role": "user", "content": "hi"}], model="claude-test",
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
    events = list(client._messages([{"role": "user", "content": "q"}],
                                  model="claude-test", max_tokens=100,
                                  stream=True))
    types = [event.type for event in events]
    assert "content_block_delta" in types and "message_stop" in types


@needs_anthropic
def test_messages_accepts_query_string(client, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    result = client._messages("What status?", model="claude-test")
    assert result["content"][0]["text"] == "ok"
    assert calls[0]["messages"] == [{"role": "user",
                                     "content": "What status?"}]
    # The wire-required budget is table-setting, not a user obligation.
    assert calls[0]["max_tokens"] == 8192
    with pytest.raises(PageIndexAPIError, match="non-empty string"):
        client._messages("   ", model="claude-test")


@needs_anthropic
def test_messages_validation(client, fake_anthropic):
    fake_anthropic([])
    with pytest.raises(PageIndexAPIError, match="non-empty"):
        client._messages([], model="claude-test", max_tokens=100)
    with pytest.raises(PageIndexAPIError,
                       match="Documents not found or access denied"):
        client._messages([{"role": "user", "content": "x"}],
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
        client._messages([{"role": "user", "content": "hi"}],
                        model="claude-test", max_tokens=100)


def test_messages_missing_framework(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(PageIndexAPIError, match="pageindex\\[anthropic\\]"):
        client._messages([{"role": "user", "content": "x"}],
                        model="claude-test", max_tokens=100)


# ── review-round regressions ──

def _anthropic_tool_use(tool_use_id="tu_1"):
    return {"type": "tool_use", "id": tool_use_id, "name": "get_document",
            "input": {"doc_name": "report.pdf"}}


@needs_agents
@pytest.mark.parametrize("surface", ["chat_completions", "_responses", "chat"])
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
    # every door that takes max_turns validates it, the runner config too
    with pytest.raises(PageIndexAPIError, match="positive integer"):
        client.chat("q", max_turns=0, stream=True)
    with pytest.raises(PageIndexAPIError, match="positive integer"):
        client.anthropic_runner_config(model="claude-sonnet-4-5",
                                       max_turns=-1)


def test_enable_citations_rejected_before_framework_check(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)
    with pytest.raises(PageIndexAPIError, match="managed chat endpoint"):
        client.chat_completions([{"role": "user", "content": "x"}],
                                enable_citations=True)
    # A cloud own-model client is told the real gate — managed vs own
    # chat — never "cloud-only": it is on the cloud.
    cloud = PageIndexClient(api_key="pi-k", chat_model="m")
    with pytest.raises(PageIndexAPIError) as err:
        cloud.chat_completions([{"role": "user", "content": "x"}],
                               enable_citations=True)
    assert "cloud-only" not in str(err.value)
    assert "drop the chat model" in str(err.value)


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
    events = list(client._responses("q", stream=True))
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
    result = client._responses("q")
    names = {tool["name"] for tool in result["tools"]}
    assert names == {"browse_documents", "get_document",
                     "get_document_structure", "get_page_content"}
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["instructions"].startswith(CHAT_HEADER)
    # No transport echo attached here, so these are the fallbacks.
    assert result["parallel_tool_calls"] is True
    assert result["tool_choice"] == "auto"


def test_sol_class_refusal_names_its_exits():
    """The chatcmpl+tools-while-reasoning 400 is a lane problem, not a
    retry problem — the wrapped error must name every exit that is real
    for the caller's lane. Chat has three; a Responses-lane caller gets only
    the litellm upgrade (it IS the other lane, and its reasoning knob is
    ``reasoning``, not ``reasoning_effort``)."""
    refusal = Exception(
        "Error code: 400 - Function tools with reasoning_effort are not "
        "supported for gpt-5.6-sol in /v1/chat/completions.")
    chat = str(local_chat._model_backend_error(refusal, "chat"))
    assert "chat(protocol='responses')" in chat and "litellm" in chat
    assert "pass reasoning_effort" in chat
    resp = str(local_chat._model_backend_error(refusal, "responses"))
    assert "upgrade litellm" in resp
    assert "protocol='responses'" not in resp
    assert "pass reasoning_effort" not in resp
    plain = local_chat._model_backend_error(Exception("rate limited"), "chat")
    assert "protocol='responses'" not in str(plain)


def test_conversation_cache_key_stable_per_conversation():
    """Cache-routing key, sent as the OpenAI prompt_cache_key. A
    conversation's continuations must share one key (same model /
    instructions / doc targeting / first item), and unrelated
    conversations must not pool under it."""
    turn1 = [{"role": "user", "content": "q"}]
    continuation = turn1 + [{"role": "assistant", "content": "a"},
                            {"role": "user", "content": "and?"}]
    key = local_chat._conversation_cache_key("m", "sys", "d1", turn1)
    assert key == local_chat._conversation_cache_key(
        "m", "sys", "d1", continuation)
    assert key == local_chat._conversation_cache_key(
        "m", "sys", ["d1"], turn1)  # str and one-item list: same targeting
    assert key != local_chat._conversation_cache_key(
        "m", "sys", "d1", [{"role": "user", "content": "other"}])
    assert key != local_chat._conversation_cache_key("m2", "sys", "d1", turn1)
    assert key != local_chat._conversation_cache_key("m", "sys2", "d1", turn1)
    assert key != local_chat._conversation_cache_key("m", "sys", "d2", turn1)
    assert key != local_chat._conversation_cache_key("m", "sys", None, turn1)


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
    assert settings.extra_args["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1}]
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
    # "" is unset on this lane too, like the protocol lanes
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, reasoning_effort="")
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
    result = client._responses("q", max_output_tokens=321)
    assert seen["responses"].max_tokens == 321
    assert result["max_output_tokens"] == 321
    fake_model([[_msg_item("ok")]])
    assert client._responses("q")["max_output_tokens"] is None
    # the public door has no knob for these; extra_body carries them to the
    # wire, and the envelope must report what was sent, not the unset locals
    fake_model([[_msg_item("ok")]])
    result = client.chat("q", protocol="responses",
                         extra_body={"max_output_tokens": 321, "top_p": 0.9,
                                     "temperature": 0.2,
                                     "metadata": {"tag": "abc"}})
    assert result["max_output_tokens"] == 321
    assert result["top_p"] == 0.9
    assert result["temperature"] == 0.2
    assert result["metadata"] == {"tag": "abc"}
    sent = seen["responses"].extra_body
    assert {"max_output_tokens": 321, "top_p": 0.9, "temperature": 0.2,
            "metadata": {"tag": "abc"}}.items() <= sent.items()


@needs_agents
def test_responses_envelope_echoes_reasoning(client, store_path, fake_model,
                                             monkeypatch):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_msg_item("ok")]])
    result = client._responses("q", reasoning={"effort": "low"})
    assert result["reasoning"] == {"effort": "low"}
    fake_model([[_msg_item("ok")]])
    assert client._responses("q")["reasoning"] is None
    seen = {}
    real = local_chat._openai_agent

    def spy(*args, **kwargs):
        agent = real(*args, **kwargs)
        seen[args[1]] = agent.model_settings
        return agent

    monkeypatch.setattr(local_chat, "_openai_agent", spy)
    fake_model([[_msg_item("ok")]])
    result = client.chat("q", protocol="responses",
                         extra_body={"reasoning": {"effort": "high"}})
    assert result["reasoning"] == {"effort": "high"}
    assert seen["responses"].extra_body["reasoning"] == {"effort": "high"}
    # effort joins the caller's reasoning object on the one channel to the
    # wire; a separate reasoning= would be replaced whole by extra_body's
    fake_model([[_msg_item("ok")]])
    result = client.chat("q", protocol="responses", reasoning_effort="high",
                         extra_body={"reasoning": {"summary": "auto"}})
    assert seen["responses"].reasoning is None
    assert seen["responses"].extra_body["reasoning"] == {
        "summary": "auto", "effort": "high"}
    assert result["reasoning"] == {"summary": "auto", "effort": "high"}


@needs_agents
def test_responses_input_validation(client, fake_model):
    fake_model([])
    for bad in ("", "   ", [], [1], None):
        with pytest.raises(PageIndexAPIError, match="messages must be"):
            client._responses(bad)
        with pytest.raises(PageIndexAPIError, match="messages must be"):
            client.chat(bad, protocol="responses")


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
        client._responses("q")


@needs_agents
def test_envelope_model_strips_litellm_routing_prefix(store_path, fake_model):
    """litellm/ is the SDK's routing marker, not a model name — the
    OpenAI-shaped envelopes must report the model the provider serves."""
    seed_doc(store_path, "pi-a", "report.pdf")
    client = PageIndexLocalClient(storage_path=store_path,
                                  retrieve_model="anthropic/claude-x")
    assert client.retrieve_model == "anthropic/claude-x"
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
    result = client._responses("q")
    assert result["model"] == "gpt-5.2"


@needs_agents
def test_chat_model_builds_keyless_without_prejudgment(monkeypatch):
    """No key pre-judgment at model build: credentials are LiteLLM's call
    at run time, so a keyless build succeeds for every spelling."""
    pytest.importorskip("litellm")  # first import may load a .env; delenv after
    from agents.extensions.models.litellm_model import LitellmModel
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in ("gpt-4o", "openai/gpt-4o"):
        model = local_chat._openai_model("chat", name)
        assert isinstance(model, LitellmModel)


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
    result = client._responses("q")
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
    result = client._responses("q")
    assert result["tool_choice"] == "none"
    assert result["parallel_tool_calls"] is False


@needs_agents
def test_chat_completions_wraps_framework_errors(client, store_path,
                                                 fake_model, monkeypatch):
    """Both chat_completions paths surface engine failures as the SDK's
    own error type, like the protocol lanes."""
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
        list(client._responses("q", stream=True))


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
    events = list(client._responses("q", stream=True))
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
        client._responses("q")
    monkeypatch.setattr(fake, "stream_response", conn_err_stream)
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        list(client.chat_completions("q", stream=True))
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        list(client._responses("q", stream=True))


@needs_anthropic
def test_messages_provider_errors_wrap_as_sdk_errors(client, store_path,
                                                     monkeypatch):
    """Anthropic transport errors surface as PageIndexAPIError on both
    messages() paths, never as anthropic types."""
    seed_doc(store_path, "pi-a", "report.pdf")

    def handler(request):
        return anthropic_httpx.Response(429, json={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "slow down"}})

    fake = anthropic.Anthropic(
        api_key="test", max_retries=0,
        http_client=anthropic_httpx.Client(
            transport=anthropic_httpx.MockTransport(handler)))
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None: fake)
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        client._messages("q", model="claude-test")
    with pytest.raises(PageIndexAPIError, match="model backend failed"):
        list(client._messages("q", model="claude-test", stream=True))


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


@needs_agents
def test_chat_stream_abandonment_cancels_pending_turn(client, store_path,
                                                      fake_model,
                                                      monkeypatch):
    """chat(stream=True)'s teardown mirrors the completions lane: closing
    the stream mid-run cancels the blocked turn (pump thread exits)
    instead of letting it run — and bill — in the background, and the
    per-call backend client is closed before its loop ends."""
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
    closes = []

    class _Backend:
        _pageindex_caller_http = False

        async def close(self):
            closes.append(True)

    fake._client = _Backend()
    stream = client.chat("q", stream=True)
    assert next(stream).startswith("[tool_call] get_document")
    stream.close()
    assert len(pumps) == 1
    pumps[0].join(timeout=3.0)
    assert not pumps[0].is_alive()
    assert fake.deltas_emitted == 0  # turn 2 never produced output
    assert closes == [True]  # _aclose_backend ran on abandonment


@needs_anthropic
def test_messages_max_tokens_default_resolves_per_model(client, fake_anthropic):
    """The wire-required budget must not exceed the model's ceiling: the
    claude-3 generation caps output at 4096."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-3-opus-20240229")
    assert calls[0]["max_tokens"] == 4096
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-sonnet-4-5")
    assert calls[0]["max_tokens"] == 8192
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-3-opus-20240229", max_tokens=1234)
    assert calls[0]["max_tokens"] == 1234


@needs_anthropic
def test_messages_thinking_passes_through(client, fake_anthropic):
    """Anthropic-native thinking config, forwarded verbatim; unset sends
    nothing so the backend default applies."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-sonnet-4-5",
                    thinking={"type": "adaptive"})
    assert calls[0]["thinking"] == {"type": "adaptive"}
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-sonnet-4-5")
    assert "thinking" not in calls[0]


@needs_anthropic
def test_messages_extra_body_merges_into_the_wire_body(client, fake_anthropic):
    """The anthropic SDK merges extra_body keys into the request JSON —
    asserted on the captured wire body, not the SDK call."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-sonnet-4-5",
                    extra_body={"service_tier": "auto"})
    assert calls[0]["service_tier"] == "auto"
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-sonnet-4-5")
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
    client._messages("q", model="claude-test", doc_id="pi-a")
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
    result = client._messages([{"role": "user", "content": "q"}],
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
    result = client._messages([{"role": "user", "content": "q"}],
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
    """A max_tokens turn carrying complete tool_use blocks: anthropic < 1.1
    executes and appends it, 1.1+ treats it as terminal and never executes.
    The envelope keys on the runner's history rather than stop_reason
    (keying on stop_reason once duplicated the tool_use id), so on either
    side the id never duplicates and the appendable messages stay valid
    for verbatim continuation."""
    seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message([_anthropic_tool_use()], "max_tokens"),
        _anthropic_message([_anthropic_tool_use("tu_2")], "tool_use"),
    ])
    result = client._messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100, max_turns=1)
    assert len(calls) == 1
    assert result["stop_reason"] == "max_tokens"
    roles = [message["role"] for message in result["messages"]]
    if _ANTHROPIC_RUNS_CUT_TOOL_TURNS:
        assert roles == ["assistant", "user"]  # tool_use then tool_result
        assert json.dumps(result["messages"]).count('"tu_1"') == 2  # use + result
    else:
        # Unexecuted tool_use has no tool_result: stripped from the
        # appendable history, still visible in content.
        assert roles == []
        assert json.dumps(result["messages"]).count('"tu_1"') == 0
        assert [block["type"] for block in result["content"]] == ["tool_use"]


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
    result = client._messages([{"role": "user", "content": "q"}],
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
    result = client._messages([{"role": "user", "content": "q"}],
                             model="claude-test", max_tokens=100)
    assert len(calls) == 10  # bounded like the OpenAI surfaces
    assert result["stop_reason"] == "tool_use"
    json.dumps(result)


@needs_anthropic
def test_messages_edge_validation(client, store_path, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    client._messages([{"role": "user", "content": "q"}], model="claude-test",
                    max_tokens=100, system="   ")
    assert all(block["text"].strip() for block in calls[0]["system"])
    with pytest.raises(PageIndexAPIError, match="message dicts"):
        client._messages(["not a dict"], model="claude-test", max_tokens=100)
    with pytest.raises(PageIndexAPIError, match="doc_id"):
        client._messages([{"role": "user", "content": "q"}],
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
    # both endpoint spellings on one merged dict: normalization keeps the
    # later (per-call) key instead of the eager nested pop discarding it
    agent = local_chat._openai_agent(
        None, "chat", "anthropic/claude-x", "sys", None, None,
        backend=local_chat._merged_backend(
            types.SimpleNamespace(chat_backend={"base_url": "http://client"}),
            {"api_base": "http://call"}))
    assert agent.model.base_url == "http://call"


@needs_anthropic
def test_messages_top_level_cache_control(client, store_path, fake_anthropic):
    """The moving breakpoint rides every request so each turn re-reads the
    growing conversation; it stands down when the caller's own marks fill
    the four-breakpoint budget (a fifth is a live-verified 400)."""
    seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-test", max_tokens=50)
    assert calls[0]["cache_control"] == {"type": "ephemeral"}
    marked = [{"type": "text", "text": f"b{i}",
               "cache_control": {"type": "ephemeral"}} for i in range(3)]
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-test", max_tokens=50, system=marked)
    assert "cache_control" not in calls[0]


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
    client._messages("q", model="claude-sonnet-4-5", backend={"api_key": "z"})
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
        return anthropic_httpx.Response(200, json=_anthropic_message(
            [{"type": "text", "text": "ok"}], "end_turn"))

    fake = anthropic.Anthropic(
        api_key="t", http_client=anthropic_httpx.Client(
            transport=anthropic_httpx.MockTransport(handler)))
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None: fake)
    client._messages("q", model="claude-sonnet-4-5",
                    extra_headers={"anthropic-beta": "context-1m-2025"})
    assert seen["beta"] == "context-1m-2025"


@needs_agents
def test_chat_model_settings_request_stream_usage(monkeypatch):
    """Without include_usage the streamed run carries no usage at all and
    the terminal chunk reports zeros (agents forwards it as
    stream_options only on streaming calls)."""
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None)
    assert agent.model_settings.include_usage is True


@needs_anthropic
def test_messages_default_max_tokens_clears_thinking_budget(client, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "a"}], "end_turn"),
    ])
    client._messages("q", model="claude-test",
                    thinking={"type": "enabled", "budget_tokens": 10000})
    assert calls[0]["max_tokens"] == 10000 + 8192
    assert calls[0]["thinking"] == {"type": "enabled",
                                    "budget_tokens": 10000}
    calls = fake_anthropic([  # fresh fake: each run closes its client
        _anthropic_message([{"type": "text", "text": "b"}], "end_turn"),
    ])
    client._messages("q", model="claude-test", max_tokens=11000,
                    thinking={"type": "enabled", "budget_tokens": 10000})
    assert calls[0]["max_tokens"] == 11000  # explicit value passes through
    # the public door's spelling: thinking rides extra_body, same lift
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "c"}], "end_turn"),
    ])
    client.chat("q", protocol="messages", model="claude-test",
                extra_body={"thinking": {"type": "enabled",
                                         "budget_tokens": 10000}})
    assert calls[0]["max_tokens"] == 10000 + 8192
    assert calls[0]["thinking"] == {"type": "enabled",
                                    "budget_tokens": 10000}


@needs_anthropic
def test_anthropic_client_cached_per_backend(monkeypatch):
    """One real client per backend: construction pays ~45ms of SSL-context
    build and a cold connection pool each call otherwise."""
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    a = local_chat._anthropic_client({"api_key": "k"})
    assert local_chat._anthropic_client({"api_key": "k"}) is a
    assert local_chat._anthropic_client({"api_key": "k2"}) is not a


@needs_anthropic
def test_anthropic_client_construction_race_keeps_first(monkeypatch):
    """A constructor losing the store race must adopt the winner rather
    than evict a client other threads may already hold."""
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    winner = object()
    real = anthropic.Anthropic

    def racing(**kwargs):
        local_chat._ANTHROPIC_CLIENTS[(("api_key", "k"),)] = winner
        return real(**kwargs)
    monkeypatch.setattr(anthropic, "Anthropic", racing)
    assert local_chat._anthropic_client({"api_key": "k"}) is winner


@needs_anthropic
def test_messages_reuses_cached_client_across_runs(client, monkeypatch):
    """A cached backend client survives the per-run close: two consecutive
    runs ride the same client (a closed one refuses the second request),
    and the cache hit constructs nothing."""
    def handler(request):
        return anthropic_httpx.Response(
            200, json=_anthropic_message([{"type": "text", "text": "ok"}],
                                         "end_turn"))
    cached = anthropic.Anthropic(
        api_key="test",
        http_client=anthropic_httpx.Client(
            transport=anthropic_httpx.MockTransport(handler)))
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS",
                        {(("api_key", "test"),): cached})

    def boom(**kwargs):
        raise AssertionError("cache hit expected — no new construction")
    monkeypatch.setattr(anthropic, "Anthropic", boom)
    for _ in range(2):
        result = client._messages("q", model="claude-test", max_tokens=64,
                                 backend={"api_key": "test"})
        assert result["stop_reason"] == "end_turn"


def test_record_chat_finish_records_and_delegates():
    recorded = {}
    closed = {"n": 0}

    class Stream:
        def __init__(self):
            self.chunks = [
                types.SimpleNamespace(choices=[]),
                types.SimpleNamespace(choices=[types.SimpleNamespace(
                    finish_reason="content_filter")]),
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            return self.chunks.pop(0)

        async def aclose(self):
            closed["n"] += 1

    async def fetch(*args, **kwargs):
        return "shell", Stream()

    model = types.SimpleNamespace(_fetch_response=fetch)
    local_chat._record_chat_finish(types.SimpleNamespace(model=model),
                                   recorded)

    async def drive():
        shell, tee = await model._fetch_response()
        assert shell == "shell"
        async for _chunk in tee:
            pass
        await tee.aclose()

    asyncio.run(drive())
    assert recorded == {"finish_reason": "content_filter"}
    assert closed["n"] == 1
    # A model without the seam: silently a no-op.
    local_chat._record_chat_finish(
        types.SimpleNamespace(model=types.SimpleNamespace()), {})


@needs_agents
def test_chat_completions_reports_native_finish_reason(client, store_path,
                                                       monkeypatch):
    seed_doc(store_path, "pi-a", "report.pdf")

    class TruncatingModel(FakeModel):
        async def _fetch_response(self, *args, **kwargs):
            return types.SimpleNamespace(choices=[
                types.SimpleNamespace(finish_reason="length")])

        async def get_response(self, *args, **kwargs):
            await self._fetch_response()
            return await super().get_response(*args, **kwargs)

        async def stream_response(self, *args, **kwargs):
            await self._fetch_response()
            async for event in super().stream_response(*args, **kwargs):
                yield event

    fake = TruncatingModel([[_msg_item("cut ")], [_msg_item("cut ")]])
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: fake)
    result = client.chat_completions("q")
    assert result["choices"][0]["finish_reason"] == "length"
    chunks = list(client.chat_completions("q", stream=True,
                                          stream_metadata=True))
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"


@needs_agents
def test_chat_model_honors_litellm_routing_and_custom_providers(monkeypatch):
    """litellm/ spellings and custom_provider_map providers pass the
    provider allowlist; a name LiteLLM cannot route is still refused up
    front."""
    pytest.importorskip("litellm")
    import litellm  # first import may load a .env; delenv after it
    from agents.extensions.models.litellm_model import LitellmModel

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = local_chat._openai_model("chat", "litellm/gpt-4o")
    assert isinstance(model, LitellmModel) and model.model == "openai/gpt-4o"

    monkeypatch.setattr(litellm, "custom_provider_map",
                        [{"provider": "my-llm", "custom_handler": object()}])
    model = local_chat._openai_model("chat", "my-llm/model-a")
    assert isinstance(model, LitellmModel) and model.model == "my-llm/model-a"

    with pytest.raises(PageIndexAPIError, match="not a LiteLLM provider"):
        local_chat._openai_model("chat", "Qwen/my-model")


@needs_agents
def test_litellm_model_still_has_the_fetch_response_seam():
    # Guards the private seam _record_chat_finish rides (LitellmModel
    # ._fetch_response): a vendor rename turns the recorder into a silent
    # no-op and every truncated turn reports finish_reason "stop".
    pytest.importorskip("litellm")
    from agents.extensions.models.litellm_model import LitellmModel

    assert hasattr(LitellmModel, "_fetch_response")


@needs_agents
def test_litellm_lane_hides_the_bridge_usage_warning():
    """litellm's chat→Responses bridge stores a chat-shaped usage dict in a
    ResponseAPIUsage field and pydantic reports it on every streamed turn;
    building the lane's model hides exactly that message — any other
    mismatch still shows."""
    pytest.importorskip("litellm")
    import warnings
    from litellm.types.llms.openai import ResponsesAPIResponse

    bridged = ResponsesAPIResponse(id="r", created_at=0, output=[])
    bridged.usage = {"completion_tokens": 1, "prompt_tokens": 1,
                     "total_tokens": 2}
    other = ResponsesAPIResponse(id="r", created_at=0, output=[])
    other.created_at = "later"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        local_chat._openai_model("chat", "gpt-5.2")
        bridged.model_dump()
        other.model_dump()
    seen = [str(w.message) for w in caught]
    assert not any("ResponseAPIUsage" in m for m in seen)
    assert any("Expected `int`" in m for m in seen)


@needs_agents
def test_litellm_lane_mutes_the_provider_list_banner(monkeypatch, capsys):
    """litellm's OpenRouter adapter probes supports_reasoning() with the
    provider-stripped model name, so any model missing from its static map
    print()s a red "Provider List:" banner into the middle of the streamed
    answer; building the lane's model flips litellm's embedder switch."""
    litellm = pytest.importorskip("litellm")
    monkeypatch.setattr(litellm, "suppress_debug_info", False)
    local_chat._openai_model("chat", "openrouter/z-ai/not-in-the-map")
    assert litellm.suppress_debug_info is True
    with pytest.raises(litellm.BadRequestError):
        litellm.get_llm_provider("z-ai/not-in-the-map")
    assert "Provider List" not in capsys.readouterr().out


@needs_agents
def test_litellm_lane_gates_litellm_logging(monkeypatch):
    """litellm's own logger sprays WARNING chatter (remote-map fetch
    fallbacks, cost hiccups) onto stderr from inside requests; building
    the lane's model gates it to ERROR — unless the caller picked a
    level via LITELLM_LOG, which stays theirs."""
    pytest.importorskip("litellm")
    import logging
    monkeypatch.delenv("LITELLM_LOG", raising=False)
    gated = logging.getLogger("LiteLLM")
    gated.setLevel(logging.WARNING)
    local_chat._openai_model("chat", "openrouter/z-ai/not-in-the-map")
    assert gated.getEffectiveLevel() == logging.ERROR
    assert not gated.isEnabledFor(logging.WARNING)
    monkeypatch.setenv("LITELLM_LOG", "DEBUG")
    gated.setLevel(logging.WARNING)
    local_chat._openai_model("chat", "openrouter/z-ai/not-in-the-map")
    assert gated.getEffectiveLevel() == logging.WARNING


def test_openai_protocol_predicate_follows_litellm_routing():
    pytest.importorskip("litellm")
    for name in ("gpt-5", "openai/gpt-4o", "litellm/gpt-4o",
                 "azure/gpt-4o", "openrouter/openai/gpt-4o",
                 "deepseek/deepseek-chat", "groq/llama-3.3-70b-versatile",
                 "xai/grok-3"):
        assert local_chat._openai_protocol(name), name
    for name in ("anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-pro",
                 "bedrock/us.anthropic.claude-sonnet-5",
                 "vertex_ai/claude-sonnet-4-5"):
        assert not local_chat._openai_protocol(name), name


@needs_agents
def test_responses_model_marks_caller_owned_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    shared = httpx.AsyncClient()
    caller = local_chat._openai_model("responses", "gpt-test",
                                      {"http_client": shared})
    assert caller._client._pageindex_caller_http is True
    owned = local_chat._openai_model("responses", "gpt-test")
    assert owned._client._pageindex_caller_http is False

    async def run():
        await local_chat._aclose_backend(types.SimpleNamespace(model=caller))
        assert not shared.is_closed  # caller-owned transport survives
        await local_chat._aclose_backend(types.SimpleNamespace(model=owned))
        await shared.aclose()

    asyncio.run(run())


@needs_anthropic
def test_messages_keeps_caller_owned_http_client_open(client):
    body = _anthropic_message([{"type": "text", "text": "a"}], "end_turn")
    shared = anthropic_httpx.Client(transport=anthropic_httpx.MockTransport(
        lambda request: anthropic_httpx.Response(200, json=body)))
    out = client._messages("q", model="claude-test",
                          backend={"api_key": "t", "http_client": shared})
    assert out["content"][0]["text"] == "a"
    assert not shared.is_closed
    client._messages("q", model="claude-test",
                    backend={"api_key": "t", "http_client": shared})
    shared.close()


@needs_anthropic
def test_messages_without_credentials_raises_contract_error(client,
                                                            monkeypatch,
                                                            tmp_path):
    """No pre-check: the SDK's own request-time credential-resolution
    failure is translated into the contract's PageIndexAPIError — for a
    bare call, a credential-less backend dict, and the unset-env-var
    shape ({"api_key": None}) alike."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ant-auth profile fallback
    for backend in (None, {"timeout": 30}, {"api_key": None}):
        with pytest.raises(PageIndexAPIError,
                           match="Anthropic backend is not configured"):
            client._messages("q", model="claude-test", backend=backend)


def test_cache_extra_args_follow_wire_normalization():
    """Chat lane only: litellm/<bare-name> rides the OpenAI protocol on
    this wire (bare names get openai/), so it must carry no Anthropic
    cache marks; explicit anthropic routes keep them. The agents lane
    routes the same spelling to Anthropic and marks it — see
    test_openai_agent_config_marks_bare_claude_behind_litellm_prefix."""
    pytest.importorskip("litellm")
    assert local_chat._cache_extra_args("litellm/claude-sonnet-4-5") is None
    assert local_chat._cache_extra_args("anthropic/claude-x") is not None
    assert local_chat._cache_extra_args("litellm/anthropic/claude-x") is not None


@needs_agents
def test_bad_model_settings_wrap_as_contract_error(client):
    """A mistyped sampling param is the SDK's verdict (pydantic),
    translated into the contract's PageIndexAPIError like every other
    door failure."""
    with pytest.raises(PageIndexAPIError, match="Invalid model settings"):
        client.chat_completions("q", temperature="hot")


@needs_agents
def test_translate_run_error_routes_all_three_kinds():
    """The shared ladder every agent door delegates to: max_turns guidance
    first (a MaxTurnsExceeded is also an AgentsException), then the
    agents-framework wrap, then the model-backend wrap."""
    import openai
    from agents.exceptions import AgentsException, MaxTurnsExceeded

    assert "max_turns (3)" in str(
        local_chat._translate_run_error(MaxTurnsExceeded("over"), 3, "chat"))
    assert "agent backend failed" in str(
        local_chat._translate_run_error(AgentsException("boom"), None,
                                        "responses"))
    assert "model backend failed" in str(
        local_chat._translate_run_error(openai.OpenAIError("down"), None,
                                        "chat"))


def test_cache_marks_counts_system_and_message_blocks():
    """The counter guards the API's 4-breakpoint limit; marks live on
    system blocks and on message content blocks, never on plain strings."""
    system = [{"type": "text", "text": "s",
               "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "t"}]
    messages = [
        {"role": "user", "content": "plain strings carry no marks"},
        {"role": "user", "content": [
            {"type": "text", "text": "a",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b"}]},
    ]
    assert local_chat._cache_marks(system, messages) == 2
    assert local_chat._cache_marks([], []) == 0


def test_dump_block_omits_unset_response_defaults():
    """messages() tells the caller to append result["messages"] verbatim;
    a response-only default like tool_use's caller must not surface as an
    explicit null the request schema has no variant for."""
    pytest.importorskip("anthropic")
    from anthropic.types.beta import BetaToolUseBlock
    block = BetaToolUseBlock(id="tu_1", input={}, name="t", type="tool_use")
    assert local_chat._dump_block(block) == {
        "id": "tu_1", "input": {}, "name": "t", "type": "tool_use"}


def test_default_max_tokens_respects_output_ceilings():
    """A lifted thinking default must not overshoot the model's output
    ceiling — the wire rejects max_tokens above it; bool is not a budget."""
    lift = local_chat._default_max_tokens
    enabled = {"type": "enabled", "budget_tokens": 30000}
    assert lift("claude-opus-4-1", enabled) == 32000
    assert lift("claude-sonnet-4-5-20250929",
                {"type": "enabled", "budget_tokens": 60000}) == 64000
    assert lift("claude-opus-4-1",
                {"type": "enabled", "budget_tokens": 10000}) == 18192
    assert lift("claude-test",
                {"type": "enabled", "budget_tokens": 10000}) == 18192
    assert lift("claude-sonnet-4-5",
                {"type": "enabled", "budget_tokens": True}) == 8192


# ── own-model chat over cloud documents (the bridge) ──


class FakeBridge:
    """Stands in for the cloud MCP bridge: one read tool, live
    instructions, recorded calls."""

    def __init__(self):
        self.calls = []

    def instructions(self):
        return "CLOUD LIVE INSTRUCTIONS"

    def list_tools(self):
        return [{"name": "get_document",
                 "description": "Cloud get_document",
                 "inputSchema": {"type": "object", "properties": {
                     "doc_name": {"type": "string"}}}}]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return json.dumps({"status": "success",
                           "data": {"doc": "cloud-doc"}}), False


@pytest.fixture
def bridge_client(monkeypatch):
    import pageindex.agent_tools as agent_tools
    from pageindex import PageIndexClient
    bridge = FakeBridge()
    monkeypatch.setattr(agent_tools, "_cloud_bridge",
                        lambda client, gated=True: bridge)
    client = PageIndexClient(api_key="pi-k", chat_model="fake-model")
    return client, bridge


@needs_agents
def test_bridge_chat_runs_engine_over_cloud_tools(bridge_client, fake_model):
    """A cloud client with its own chat model runs the in-process agent:
    tools come from the live cloud MCP set, instructions from the MCP
    server — not the local subset."""
    client, bridge = bridge_client
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "r.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.chat_completions("What?")
    assert result["choices"][0]["message"]["content"] == "The answer"
    assert bridge.calls == [("get_document", {"doc_name": "r.pdf"})]
    assert fake.instructions[0].startswith(CHAT_HEADER)
    assert "CLOUD LIVE INSTRUCTIONS" in fake.instructions[0]
    assert "READING WORKFLOW" not in fake.instructions[0]
    # The tool result made it back into turn 2.
    assert "cloud-doc" in json.dumps(fake.inputs[1])


@needs_agents
def test_bridge_doc_id_targets_at_prompt_level(bridge_client, fake_model,
                                               monkeypatch):
    """On cloud tools there is no local allowlist: doc_id becomes the
    prompt-level targeting block only. (Had the tool layer received the
    doc_ids, _require_local_scope would raise on a cloud client — this
    call succeeding is the proof it did not.)"""
    client, _ = bridge_client
    monkeypatch.setattr(client, "get_document",
                        lambda doc_id: {"name": "r.pdf", "description": "d",
                                        "status": "completed",
                                        "metadata": None})
    monkeypatch.setattr(
        client, "list_documents",
        lambda **kw: {"documents": [{"id": "pi-a", "name": "r.pdf"}],
                      "total": 1})
    fake = fake_model([[_msg_item("ok")]])
    client.chat_completions("q", doc_id="pi-a")
    first = fake.inputs[0][0]
    assert "specified document" in first["content"]
    assert "r.pdf" in first["content"]


def test_bridge_gate_and_citations(monkeypatch):
    from pageindex import PageIndexClient
    client = PageIndexClient(api_key="pi-k", chat_model="m")
    with pytest.raises(PageIndexAPIError, match="drop the chat model"):
        client.chat_completions("x", enable_citations=True)

    called = {}

    def fake_responses(client_arg, *args, **kwargs):
        called["responses"] = client_arg
        return {}

    def fake_messages(client_arg, *args, **kwargs):
        called["messages"] = client_arg
        return {}

    monkeypatch.setattr(local_chat, "run_responses", fake_responses)
    monkeypatch.setattr(local_chat, "run_messages", fake_messages)
    client._responses("q")
    client._messages("q", model="mm")
    assert called["responses"] is client and called["messages"] is client


def test_bridge_auth_failure_error_teaches_architecture():
    """The misreading ("the cloud runs my model") surfaces as a missing
    provider key — that error is where the architecture gets spelled out,
    and only there: local clients and non-auth failures stay untouched."""
    from pageindex import PageIndexClient
    bridge = PageIndexClient(api_key="pi-k", chat_model="m")
    local = PageIndexLocalClient()
    exc = Exception("The api_key client option must be set")
    assert "your own provider credentials" in str(
        local_chat._model_backend_error(exc, "chat", bridge))
    assert "provider credentials" not in str(
        local_chat._model_backend_error(exc, "chat", local))
    assert "provider credentials" not in str(
        local_chat._model_backend_error(Exception("boom"), "chat", bridge))
    # 401-shaped failures carry no "api key" text on some providers
    # (Anthropic says x-api-key): the status code is the signal.
    denied = Exception("invalid x-api-key")
    denied.status_code = 401
    assert "your own provider credentials" in str(
        local_chat._model_backend_error(denied, "messages", bridge))


def test_bridge_auth_note_managed_exit_is_chat_lane_only():
    """The managed-chat exit ("drop the chat model") is real only for
    chat_completions(); the protocol lanes refuse a client
    without an own model, so on those lanes the note keeps the
    credentials advice and drops the exit that would send the caller in
    a circle."""
    from pageindex import PageIndexClient
    bridge = PageIndexClient(api_key="pi-k", chat_model="m")
    denied = Exception("invalid x-api-key")
    denied.status_code = 401
    chat = str(local_chat._model_backend_error(denied, "chat", bridge))
    assert "drop the chat model" in chat
    for lane in ("responses", "messages"):
        text = str(local_chat._model_backend_error(denied, lane, bridge))
        assert "your own provider credentials" in text
        assert "drop the chat model" not in text


@needs_anthropic
def test_messages_auth_failure_teaches_architecture(bridge_client,
                                                    monkeypatch):
    """The auth note must reach the messages door too — both its paths
    wrap provider failures through _model_backend_error."""
    client, _ = bridge_client

    def handler(request):
        return anthropic_httpx.Response(
            401, json={"type": "error",
                       "error": {"type": "authentication_error",
                                 "message": "invalid x-api-key"}})

    def fresh_fake(backend=None):
        # per call: run_messages closes a per-call transport it owns
        return anthropic.Anthropic(
            api_key="test",
            http_client=anthropic_httpx.Client(
                transport=anthropic_httpx.MockTransport(handler)))

    monkeypatch.setattr(local_chat, "_anthropic_client", fresh_fake)
    with pytest.raises(PageIndexAPIError, match="provider credentials"):
        client._messages("q", model="claude-test", max_tokens=100)
    with pytest.raises(PageIndexAPIError, match="provider credentials"):
        list(client._messages("q", model="claude-test", max_tokens=100,
                             stream=True))


@needs_anthropic
def test_messages_no_backend_leak_when_tool_build_fails(bridge_client,
                                                        monkeypatch):
    """build_anthropic_tools is network I/O on a bridge client — a
    failure there must not strand an opened per-call transport."""
    client, _ = bridge_client
    made = []

    class FakeAnthropic:
        def __init__(self):
            self.closed = False
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(
                tool_runner=lambda **kw: iter(())))

        def close(self):
            self.closed = True

    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None: made.append(FakeAnthropic())
                        or made[-1])

    def boom(client, doc_ids=None):
        raise PageIndexAPIError("Could not reach the PageIndex MCP server")

    monkeypatch.setattr(
        "pageindex.integrations.anthropic_sdk.build_anthropic_tools", boom)
    with pytest.raises(PageIndexAPIError, match="MCP server"):
        client._messages("q", model="claude-test", max_tokens=100)
    assert all(fake.closed for fake in made)


@needs_agents
def test_bridge_responses_lane_runs_cloud_tools(bridge_client, fake_model):
    """The Responses door on a bridge client: same engine, cloud tools,
    live instructions."""
    client, bridge = bridge_client
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "r.pdf"})],
        [_msg_item("Done")],
    ])
    result = client._responses("What?")
    assert result["object"] == "response" and result["status"] == "completed"
    assert "Done" in json.dumps(result["output"])
    assert bridge.calls == [("get_document", {"doc_name": "r.pdf"})]
    assert "CLOUD LIVE INSTRUCTIONS" in fake.instructions[0]
    assert fake_model.state["protocols"][0][0] == "responses"


@needs_anthropic
def test_bridge_messages_lane_runs_cloud_tools(bridge_client, fake_anthropic):
    """The Messages door on a bridge client: cloud MCP tools on the wire,
    live instructions in the cached system prefix."""
    client, bridge = bridge_client
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "The answer"}],
                           "end_turn"),
    ])
    result = client._messages("What?", model="claude-test", max_tokens=64)
    assert result["content"][0]["text"] == "The answer"
    wire = calls[0]
    assert wire["tools"][0]["name"] == "get_document"
    system_text = json.dumps(wire["system"])
    assert "CLOUD LIVE INSTRUCTIONS" in system_text
    assert "READING WORKFLOW" not in system_text


@needs_agents
def test_bridge_openai_agent_config_carries_configured_model(bridge_client):
    """A bridge client's openai_agent_config carries its chat_model —
    same semantics as local; a plain cloud client still omits model."""
    client, _ = bridge_client
    config = client.openai_agent_config()
    assert config["model"] == "fake-model"
    assert "CLOUD LIVE INSTRUCTIONS" in config["instructions"]


# ── chat(protocol=): the protocol doors behind the front door ──

def test_old_door_names_point_at_chat_protocol(client):
    for name in ("responses", "messages"):
        with pytest.raises(AttributeError, match=f"chat\\(protocol={name!r}"):
            getattr(client, name)
        assert not hasattr(client, name)
    with pytest.raises(AttributeError, match="no attribute 'no_such_thing'"):
        client.no_such_thing


def test_chat_protocol_responses_is_the_door(client, monkeypatch):
    seen = []
    monkeypatch.setattr(local_chat, "run_responses",
                        lambda c, input, **kw: seen.append((input, kw)) or "door")
    knobs = dict(doc_id="pi-a", model="gpt-x", max_turns=3,
                 backend={"api_key": "k"}, extra_headers={"x": "1"},
                 instructions="be brief")
    assert client.chat("q", protocol="responses", reasoning_effort="low",
                       extra_body={"seed": 1}, **knobs) == "door"
    assert client._responses("q", extra_body={"seed": 1,
                                              "reasoning": {"effort": "low"}},
                             **knobs) == "door"
    assert seen[0] == seen[1]
    assert seen[0][1]["reasoning"] is None
    # OpenAI's own effort field; the caller's extra_body still wins
    client.chat("q", protocol="responses", reasoning_effort="low",
                extra_body={"reasoning": {"effort": "high"}})
    assert seen[-1][1]["extra_body"] == {"reasoning": {"effort": "high"}}
    # a caller's other reasoning keys survive; only effort is ours
    client.chat("q", protocol="responses", reasoning_effort="low",
                extra_body={"reasoning": {"summary": "auto"}})
    assert seen[-1][1]["extra_body"] == {
        "reasoning": {"summary": "auto", "effort": "low"}}
    # "" is unset, like model="" and instructions=""
    client.chat("q", protocol="responses", reasoning_effort="",
                extra_body={"seed": 1})
    assert seen[-1][1]["extra_body"] == {"seed": 1}
    assert seen[-1][1]["reasoning"] is None
    # the default for the stream view is silently off on a protocol lane
    assert client.chat("q", protocol="responses", stream=True) == "door"
    assert seen[-1][1]["stream"] is True


def test_chat_protocol_messages_is_the_door(client, monkeypatch):
    seen = []
    monkeypatch.setattr(local_chat, "run_messages",
                        lambda c, messages, **kw: seen.append((messages, kw)) or "door")
    blocks = [{"type": "text", "text": "persona",
               "cache_control": {"type": "ephemeral"}}]
    knobs = dict(doc_id="pi-a", model="claude-x", max_turns=3,
                 backend={"api_key": "k"}, extra_headers={"anthropic-beta": "b"})
    history = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
    assert client.chat(history, protocol="messages", reasoning_effort="low",
                       instructions=blocks, extra_body={"top_k": 5},
                       **knobs) == "door"
    assert client._messages(history, system=blocks,
                            extra_body={"output_config": {"effort": "low"},
                                        "top_k": 5},
                            **knobs) == "door"
    assert seen[0] == seen[1]
    # Anthropic's own effort field; the caller's extra_body still wins
    client.chat("q", protocol="messages", model="claude-x",
                reasoning_effort="low",
                extra_body={"output_config": {"effort": "max"}})
    assert seen[-1][1]["extra_body"] == {"output_config": {"effort": "max"}}
    assert seen[-1][0] == "q"
    # a caller's other output_config keys survive; only effort is ours
    client.chat("q", protocol="messages", model="claude-x",
                reasoning_effort="low",
                extra_body={"output_config": {"format": {"type": "json"}}})
    assert seen[-1][1]["extra_body"] == {
        "output_config": {"format": {"type": "json"}, "effort": "low"}}
    # "" is unset, like model="" and instructions=""
    client.chat("q", protocol="messages", model="claude-x",
                reasoning_effort="")
    assert seen[-1][1]["extra_body"] is None


def test_chat_protocol_chokes(client, monkeypatch):
    monkeypatch.setattr(local_chat, "run_responses",
                        lambda c, input, **kw: "door")
    with pytest.raises(PageIndexAPIError, match="protocol selects"):
        client.chat("q", protocol="grpc")
    with pytest.raises(PageIndexAPIError, match="drop show_process"):
        client.chat("q", protocol="responses", stream=True, show_process=True)
    # the protocol refusal first: "add stream=True" is no remedy here
    with pytest.raises(PageIndexAPIError, match="drop show_process"):
        client.chat("q", protocol="responses", show_process=True)
    with pytest.raises(PageIndexAPIError, match="instructions blocks"):
        client.chat("q", protocol="responses",
                    instructions=[{"type": "text", "text": "x"}])
    with pytest.raises(PageIndexAPIError, match="instructions blocks"):
        client.chat("q", instructions=[{"type": "text", "text": "x"}])
    with pytest.raises(PageIndexAPIError, match="model="):
        client.chat("q", protocol="messages")
    with pytest.raises(PageIndexAPIError, match="model="):
        client.chat("q", protocol="messages", model="")


@needs_agents
def test_chat_instructions_precede_history_system_rows(client, store_path,
                                                        fake_model, monkeypatch):
    fake_model([[_msg_item("ok")], [_msg_item("ok")], [_msg_item("ok")]])
    seen = {}
    real = local_chat._managed_instructions

    def spy(c, extra):
        seen["extra"] = list(extra)
        return real(c, extra)

    monkeypatch.setattr(local_chat, "_managed_instructions", spy)
    answer = client.chat([{"role": "system", "content": "short"},
                          {"role": "user", "content": "q"}],
                         instructions="analyst")
    assert answer == "ok"
    assert seen["extra"] == ["analyst", "short"]
    client.chat("q", instructions="analyst")  # a bare question works too
    assert seen["extra"] == ["analyst"]
    client.chat("q", instructions="")  # blank configures nothing, no row
    assert seen["extra"] == []


def test_chat_answer_lane_forwards_the_promoted_knobs(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(local_chat, "run_chat_completions",
                        lambda c, messages, **kw: seen.update(kw) or {
                            "choices": [{"message": {"content": "a"}}]})
    knobs = dict(max_turns=2, backend={"api_key": "k"},
                 extra_headers={"x": "1"}, extra_body={"seed": 1})
    assert client.chat("q", **knobs) == "a"
    assert {k: seen[k] for k in knobs} == knobs
    streamed = {}
    monkeypatch.setattr(local_chat, "run_chat_stream",
                        lambda c, messages, **kw: streamed.update(kw) or "s")
    assert client.chat("q", stream=True, **knobs) == "s"
    assert {k: streamed[k] for k in knobs} == knobs
