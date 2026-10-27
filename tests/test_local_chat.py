"""Local chat surfaces: three protocols over fake backends — no network,
no LLM keys. Tool execution runs for real against a seeded local store."""
import asyncio
import contextlib
import copy
import dataclasses
import inspect
import json
import pickle
import sys
import types
import warnings

import yaml

import httpx  # via the hard `openai` dependency
import pytest
from openai.types.shared import Reasoning

try:  # anthropic >= 1.0 validates http_client against httpx2
    import httpx2 as anthropic_httpx
except ImportError:  # older anthropic rides classic httpx
    anthropic_httpx = httpx

import pageindex.local_chat as local_chat
from pageindex import (PageIndexAPIError, PageIndexClient,
                       PageIndexCloudClient, PageIndexLocalClient,
                       ChatObject)
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
        output = self.turns.pop(0)
        if isinstance(output, BaseException):
            raise output  # the backend fails this turn
        return ModelResponse(output=output, usage=_usage(), response_id=None)

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
        if isinstance(output, BaseException):
            raise output  # the backend fails this turn
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
    call = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 0,
                                      "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0}}
    assert result["usage"] == {
        "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
        "request_usage_entries": [call, call]}
    from openai.types import CompletionUsage
    typed = CompletionUsage.model_validate(result["usage"])
    assert typed.prompt_tokens_details is not None
    assert typed.prompt_tokens_details.cache_write_tokens == 0
    assert fake_model.state["protocols"][0][0] == "chat"
    # The tool ran for real: turn 2's input carries its output.
    turn2 = json.dumps(fake.inputs[1])
    assert "report.pdf" in turn2 and "completed" in turn2
    # Managed instructions: header + the local agent guidance.
    assert fake.instructions[0].startswith(CHAT_HEADER)
    assert "PERSISTENCE" in fake.instructions[0]


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
    with pytest.raises(PageIndexAPIError, match="tool_call_id"):
        client.chat_completions([{"role": "tool", "content": "x"}])
    with pytest.raises(PageIndexAPIError, match="needs content"):
        client.chat_completions([{"role": "assistant", "content": None}])
    with pytest.raises(PageIndexAPIError, match="needs content"):
        client.chat_completions([{"role": "assistant", "content": None,
                                  "reasoning_content": "x"}])
    with pytest.raises(PageIndexAPIError, match="Invalid assistant message"):
        client.chat_completions([{"role": "assistant",
                                  "tool_calls": [{"id": "c"}]}])
    with pytest.raises(PageIndexAPIError, match="Unsupported role"):
        client.chat_completions([{"role": "function", "content": "x"}])
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
        assert agent.model_settings.extra_args == {"max_retries": 0, **marked}
    for name in ("gpt-5", "openai/Qwen/x", "litellm/groq/x",
                 "bedrock/meta.llama3-70b-instruct-v1:0",
                 "vertex_ai/gemini-2.5-pro"):
        agent = _openai_agent(client, "chat", name, "sys", None, None)
        assert agent.model_settings.extra_args == {"max_retries": 0}


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


def _anthropic_wire(monkeypatch, client, model, messages: str | list = "hi",
                    **settings):
    """One chat_completions call via the real stack: (request, envelope)."""
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
    return captured, client.chat_completions(messages, model=model, **settings)


@needs_agents
def test_foundry_claude_gets_no_openai_fields(client, monkeypatch):
    """LiteLLM sends azure_ai's Claude to Foundry's Anthropic API, which
    refuses prompt_cache_key like Anthropic's own (400, extra inputs)."""
    pytest.importorskip("litellm")
    captured, _ = _anthropic_wire(
        monkeypatch, client, "azure_ai/claude-sonnet-5-5",
        backend={"api_base": "https://x.services.ai.azure.com",
                 "api_key": "k"})
    assert captured["url"].endswith("/anthropic/v1/messages")
    assert "prompt_cache_key" not in captured["body"]


@needs_agents
def test_cache_marker_reaches_the_anthropic_wire(client, store_path,
                                                 monkeypatch):
    pytest.importorskip("litellm")
    captured, result = _anthropic_wire(
        monkeypatch, client, "anthropic/claude-3-5-sonnet-20240620")
    assert "/v1/messages" in captured["url"]
    assert '"cache_control"' in json.dumps(captured["body"])
    # The OpenAI cache-routing hint must not leak here: LiteLLM plants
    # extra_body as a literal field, and Anthropic rejects unknown fields.
    assert "extra_body" not in json.dumps(captured["body"])
    assert result["choices"][0]["message"]["content"] == "ok"


@needs_agents
def test_chat_completions_messages_on_both_paths(client, store_path,
                                                 fake_model):
    """The run's new messages ride the envelope and the closing usage
    chunk alike."""
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")

    def run():
        return [[_call_item("get_document", {"doc_name": "report.pdf"})],
                [_msg_item("The answer")]]

    fake_model(run())
    envelope = client.chat_completions("q", doc_id=doc_id)
    fake_model(run())
    *_, closing = client.chat_completions("q", doc_id=doc_id, stream=True,
                                          stream_metadata=True)
    assert closing["messages"] == envelope["messages"]
    assert [m["role"] for m in envelope["messages"]] == [
        "assistant", "tool", "assistant"]
    assert envelope["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert envelope["messages"][-1] == {"role": "assistant",
                                        "content": "The answer"}


def _litellm_reply(content=None, tool_calls=None, **extra):
    import litellm
    return litellm.ModelResponse(
        choices=[litellm.Choices(
            finish_reason="tool_calls" if tool_calls else "stop",
            message=litellm.Message(content=content, role="assistant",
                                    tool_calls=tool_calls, **extra))],
        usage=litellm.Usage(prompt_tokens=10, completion_tokens=5,
                            total_tokens=15))


def _litellm_chunks(*deltas, finish="tool_calls"):
    """A streamed turn as LiteLLM's parser yields it."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    return [ModelResponseStream(choices=[StreamingChoices(
        delta=Delta(**delta), finish_reason=None)]) for delta in deltas] + [
        ModelResponseStream(choices=[StreamingChoices(
            delta=Delta(), finish_reason=finish)])]


_TWO_CALLS = [{"id": f"call_{n}", "type": "function",
               "function": {"name": "get_document",
                            "arguments": '{"doc_name": "report.pdf"}'}}
              for n in (1, 2)]


def _mock_litellm(monkeypatch, replies, marks=None):
    """Real LiteLLM answering each request with the next reply, after the
    SDK's own hook has filled it: a ModelResponse, a list of chunks to
    stream (an exception in it breaks the stream there), or an exception
    to raise. Returns each request's messages as sent; ``marks`` collects
    each request's cache injection points."""
    import litellm
    import litellm.main
    from litellm.integrations.custom_logger import CustomLogger
    sent = []
    replies = list(replies)
    streaming = litellm.main.async_mock_completion_streaming_obj

    class Answer(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            sent.append(json.loads(json.dumps(kwargs["messages"])))
            if marks is not None:
                marks.append(kwargs.get("cache_control_injection_points"))
            return {**kwargs, "mock_response": replies.pop(0)}

    async def stream(model_response, mock_response, **kwargs):
        if isinstance(mock_response, list):
            for chunk in mock_response:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
            return
        async for chunk in streaming(model_response,
                                     mock_response=mock_response, **kwargs):
            yield chunk

    monkeypatch.setattr(litellm.main, "async_mock_completion_streaming_obj",
                        stream)
    local_chat._register_fill()
    # LiteLLM copies litellm.callbacks into these lists on each call
    for name in ("input_callback", "success_callback", "failure_callback",
                 "_async_input_callback", "_async_success_callback",
                 "_async_failure_callback"):
        monkeypatch.setattr(litellm, name, list(getattr(litellm, name)))
    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, Answer()])
    return sent


def _two_turns(store_path, monkeypatch, client, model, first_reply,
               marks=None, **settings):
    """Two calls through the real LitellmModel and LiteLLM, the second
    appending the first's ``messages``: the first envelope (or closing
    chunk), and each request's messages as LiteLLM sends them."""
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    sent = _mock_litellm(monkeypatch, [first_reply, _litellm_reply("A1"),
                                       _litellm_reply("A2")], marks)

    def call(history):
        out = client.chat_completions(history, doc_id=doc_id, model=model,
                                      **settings)
        return list(out)[-1] if settings.get("stream") else out

    question = {"role": "user", "content": "q1"}
    first = call([question])
    call([question, *first["messages"], {"role": "user", "content": "q2"}])
    return first, sent


@needs_agents
@pytest.mark.parametrize("model, extra, settings", [
    ("gpt-test", {}, {}),
    # streamed turns are rebuilt by agents' own stream handler; OpenAI
    # streams open with an empty content delta
    ("gpt-test", {"content": ""}, {"stream": True, "stream_metadata": True}),
    ("anthropic/claude-sonnet-5-5", {
        "reasoning_content": "hmm", "thinking_blocks": [
            {"type": "thinking", "thinking": "hmm", "signature": "SIG1"}]},
     {"extra_body": {"reasoning": {"effort": "high"}}, "stream": True,
      "stream_metadata": True}),
    # LitellmModel sends parallel calls one per message here
    ("anthropic/claude-test", {}, {}),
    ("anthropic/claude-sonnet-5-5", {"thinking_blocks": [
        {"type": "thinking", "thinking": "hmm", "signature": "SIG1"}]},
     {"extra_body": {"reasoning": {"effort": "high"}}}),
    # replayed as reasoning_content beside the calls
    ("deepseek/deepseek-chat", {"reasoning_content": "think"}, {}),
])
def test_chat_completions_messages_resend_the_wire(client, store_path,
                                                   monkeypatch, model, extra,
                                                   settings):
    """The returned messages are what the model saw, each assistant
    message naming its model, so appending them opens the next call's
    first request with the previous call's last one, unchanged: the
    provider's cached prefix carries over."""
    pytest.importorskip("litellm")
    first, sent = _two_turns(store_path, monkeypatch, client, model,
                             _litellm_reply(tool_calls=_TWO_CALLS, **extra),
                             **settings)
    writers = {m.pop("model") for m in first["messages"]
               if m["role"] == "assistant"}
    assert writers == {model if "/" in model else f"openai/{model}"}
    assert first["messages"][:-1] == sent[1][len(sent[0]):]
    assert first["messages"][-1]["content"] == "A1"
    assert sent[2][:len(sent[1])] == sent[1]


@needs_agents
@pytest.mark.parametrize("model, extra", [
    ("gpt-test", {}),
    # the reply's own fields come back through the fill
    ("deepseek/deepseek-chat", {"reasoning_content": "think"}),
])
def test_messages_continuing_a_reply_keep_their_tool_call(client, store_path,
                                                          monkeypatch, model,
                                                          extra):
    """A history ending with a returned reply: the run's tool call stays
    its own assistant message, on the wire and in the returned messages,
    so they append back whole and the next call opens with this one's
    last request."""
    pytest.importorskip("litellm")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    sent = _mock_litellm(monkeypatch, [
        _litellm_reply("A0", **extra),
        _litellm_reply(tool_calls=_TWO_CALLS[:1]), _litellm_reply("A1"),
        _litellm_reply("A2")])
    question = {"role": "user", "content": "q"}
    first = client.chat_completions([question], doc_id=doc_id, model=model)
    out = client.chat_completions([question, *first["messages"]],
                                  doc_id=doc_id, model=model)
    assert [m["role"] for m in out["messages"]] == ["assistant", "tool",
                                                    "assistant"]
    assert out["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert sent[1][-1].get("reasoning_content") == extra.get(
        "reasoning_content")
    assert sent[2][:len(sent[1])] == sent[1]
    client.chat_completions([question, *first["messages"], *out["messages"],
                             {"role": "user", "content": "q2"}],
                            doc_id=doc_id, model=model)
    assert sent[3][:len(sent[2])] == sent[2]


@needs_agents
@pytest.mark.parametrize("model, extra, split, text, stream", [
    ("gpt-test", {}, False, None, False),
    ("gpt-test", {}, False, None, True),
    # LitellmModel sends each call in its own message
    ("anthropic/claude-test", {}, True, None, False),
    ("deepseek/deepseek-chat", {"reasoning_content": "think"}, False, None,
     False),
    ("gpt-test", {}, False, "Sorry.", False),
])
def test_messages_continuing_a_refusal_keep_their_tool_calls(
        client, store_path, monkeypatch, model, extra, split, text, stream):
    """A history ending with a refusal: the SDK folds the run's first tool
    calls into it on the wire, and the returned messages open with them,
    the refusal and its text left in the history, so they append back
    whole and the next call opens with this one's last request."""
    pytest.importorskip("litellm")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    sent = _mock_litellm(monkeypatch, [
        _litellm_reply(tool_calls=_TWO_CALLS, **extra), _litellm_reply("A1"),
        _litellm_reply("A2")])
    settings = {"stream": True, "stream_metadata": True} if stream else {}

    def call(history):
        out = client.chat_completions(history, doc_id=doc_id, model=model,
                                      **settings)
        return list(out)[-1] if stream else out

    history = [{"role": "user", "content": "q"},
               {"role": "assistant", "content": text, "refusal": "No."}]
    out = call(history)
    one = [("assistant", ["call_1", "call_2"]), ("tool", []), ("tool", [])]
    each = [("assistant", ["call_1"]), ("tool", []),
            ("assistant", ["call_2"]), ("tool", [])]
    assert [(m["role"], [c["id"] for c in m.get("tool_calls") or ()])
            for m in out["messages"]] == (each if split else one) + [
        ("assistant", [])]
    assert "refusal" not in out["messages"][0]
    assert out["messages"][0]["content"] is None
    assert out["messages"][0].get("reasoning_content") == extra.get(
        "reasoning_content")
    call([*history, *out["messages"], {"role": "user", "content": "q2"}])
    assert sent[2][:len(sent[1])] == sent[1]


@needs_agents
@pytest.mark.parametrize("settings", [
    {"reasoning_effort": "high"},
    {"extra_body": {"reasoning": {"effort": "high"}}},
])
def test_claude_thinking_goes_back_with_its_calls(client, store_path,
                                                  monkeypatch, settings):
    """Thinking goes back once, as thinking_blocks."""
    pytest.importorskip("litellm")
    thinking = [{"type": "thinking", "thinking": "hmm", "signature": "SIG1"}]
    first, sent = _two_turns(store_path, monkeypatch, client,
                             "anthropic/claude-sonnet-5-5",
                             _litellm_reply(tool_calls=_TWO_CALLS,
                                            thinking_blocks=thinking,
                                            reasoning_content="hmm"),
                             **settings)
    for messages in (first["messages"], sent[1], sent[2]):
        calls = [m for m in messages if m.get("tool_calls")]
        assert calls[0].get("thinking_blocks") == thinking


@needs_agents
@pytest.mark.parametrize("settings", [
    {}, {"extra_body": {"thinking": {"type": "adaptive"}}},
    {"stream": True, "stream_metadata": True}])
@pytest.mark.parametrize("calls", [1, 2])
def test_claude_thinking_goes_back_with_no_effort_set(client, store_path,
                                                      monkeypatch, settings,
                                                      calls):
    """Claude's thinking goes back with no effort set, on the first
    message of a reply sent one call per message."""
    pytest.importorskip("litellm")
    thinking = [{"type": "thinking", "thinking": "", "signature": "SIG1"}]
    first, sent = _two_turns(store_path, monkeypatch, client,
                             "anthropic/claude-sonnet-5-5",
                             _litellm_reply(tool_calls=_TWO_CALLS[:calls],
                                            thinking_blocks=thinking),
                             **settings)
    for messages in (first["messages"], sent[1], sent[2]):
        sent_calls = [m for m in messages if m.get("tool_calls")]
        assert sent_calls[0].get("thinking_blocks") == thinking
        assert all("thinking_blocks" not in m for m in sent_calls[1:])


@needs_agents
@pytest.mark.parametrize("model", [
    "moonshot/kimi-k2-thinking", "zai/glm-4.6", "dashscope/qwen3-test",
    "openrouter/qwen/qwen3-test-thinking", "deepseek/deepseek-chat"])
def test_reasoning_content_goes_back_to_its_model(client, store_path,
                                                  monkeypatch, model):
    """Each model's reasoning_content goes back to it."""
    pytest.importorskip("litellm")
    first, sent = _two_turns(store_path, monkeypatch, client, model,
                             _litellm_reply(tool_calls=_TWO_CALLS[:1],
                                            reasoning_content="think"))
    for messages in (first["messages"], sent[1], sent[2]):
        calls = [m for m in messages if m.get("tool_calls")]
        assert calls[0].get("reasoning_content") == "think"


@needs_agents
def test_reasoning_content_never_reaches_claude(client, monkeypatch):
    """Another model's reasoning_content stays out of Claude's request."""
    pytest.importorskip("litellm")
    history = [{"role": "user", "content": "q1"},
               {"role": "assistant", "content": None,
                "reasoning_content": "Kimi's own plan",
                "tool_calls": _TWO_CALLS[:1]},
               {"role": "tool", "tool_call_id": "call_1", "content": "r"},
               {"role": "assistant", "content": "A1"},
               {"role": "user", "content": "q2"}]
    captured, _ = _anthropic_wire(monkeypatch, client, "anthropic/claude-test",
                                  messages=history)
    assert captured["body"]["messages"]
    assert "Kimi's own plan" not in json.dumps(captured["body"])


def _history(writer=None):
    reply = {"role": "assistant", "content": None,
             "tool_calls": _TWO_CALLS[:1], "reasoning_content": "plan",
             "thinking_blocks": [{"type": "thinking", "thinking": "plan",
                                  "signature": "SIG"}],
             "id": "msg_1", "finish_reason": "tool_calls"}
    answer = {"role": "assistant", "content": "A1", "reasoning_content": "sum",
              "id": "msg_2", "finish_reason": "stop"}
    tag = {"model": writer} if writer else {}
    return [{"role": "user", "content": "q1"}, {**reply, **tag},
            {"role": "tool", "tool_call_id": "call_1", "content": "r"},
            {**answer, **tag}, {"role": "user", "content": "q2"}]


@needs_agents
@pytest.mark.parametrize("writer, model, settings", [
    (None, "gemini/gemini-test", {}),
    ("anthropic/claude-test", "gemini/gemini-test", {}),
    # the Agents SDK's own replays read the writer too
    ("anthropic/claude-test", "anthropic/claude-sonnet-5-5",
     {"reasoning_effort": "high"}),
    ("anthropic/claude-test", "deepseek/deepseek-chat", {})])
def test_history_fields_go_back_to_their_own_model_only(client, monkeypatch,
                                                        writer, model,
                                                        settings):
    """A history message's fields go back only to the model its ``model``
    names: another model's reasoning, and the keys of a message that
    names none, stay off the request, a text reply's too."""
    pytest.importorskip("litellm")
    sent = _mock_litellm(monkeypatch, [_litellm_reply("A2")])
    client.chat_completions(_history(writer), model=model, **settings)
    turn = next(m for m in sent[0] if m.get("tool_calls"))
    answer = next(m for m in sent[0] if m.get("content") == "A1")
    for message in (turn, answer):
        assert not {"reasoning_content", "thinking_blocks", "id",
                    "finish_reason", "model"} & set(message)


@needs_agents
@pytest.mark.parametrize("model, fields, settings", [
    ("deepseek/deepseek-chat", {"reasoning_content": "plan"}, {}),
    ("anthropic/claude-sonnet-5-5", {"thinking_blocks": [
        {"type": "thinking", "thinking": "plan", "signature": "SIG"}]},
     {"reasoning_effort": "high"})])
@pytest.mark.parametrize("tail", [[{"role": "user", "content": "q2"}], []])
def test_untagged_replies_keep_the_sdk_replay(client, monkeypatch, model,
                                              fields, settings, tail):
    """A reply that names no model goes in as it did before ``model``
    existed, a run going on from it too: with the reasoning the Agents SDK
    replays, which DeepSeek needs on a reply in the current turn and Claude
    binds each later thinking block to."""
    pytest.importorskip("litellm")
    sent = _mock_litellm(monkeypatch, [_litellm_reply("A2")])
    client.chat_completions([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "A1", **fields}, *tail],
        model=model, **settings)
    reply = next(m for m in sent[0] if m.get("content") == "A1")
    assert {key: reply.get(key) for key in fields} == fields


@needs_agents
def test_own_history_given_as_a_generator_goes_back(client, monkeypatch):
    pytest.importorskip("litellm")
    model = "moonshot/kimi-k2-thinking"
    sent = _mock_litellm(monkeypatch, [_litellm_reply("A2")])
    client.chat_completions(iter(_history(model)), model=model)
    turn = next(m for m in sent[0] if m.get("tool_calls"))
    assert turn["reasoning_content"] == "plan" and "model" not in turn


@needs_agents
@pytest.mark.parametrize("final, stream", [("", False), (None, True)])
def test_a_final_reply_with_no_text_adds_no_message(client, store_path,
                                                    monkeypatch, final,
                                                    stream):
    """A run that ends with no text after a tool result returns that
    result once, with no message after it."""
    pytest.importorskip("litellm")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _mock_litellm(monkeypatch, [_litellm_reply(tool_calls=_TWO_CALLS[:1]),
                                _litellm_reply(final)])
    out = client.chat_completions("q", doc_id=doc_id, model="gpt-test",
                                  stream=stream, stream_metadata=stream)
    out = list(out)[-1] if stream else out
    assert [m["role"] for m in out["messages"]] == ["assistant", "tool"]


@needs_agents
def test_reply_with_no_choices_records_nothing(client, monkeypatch):
    """The Agents SDK reads a reply with no choices as an empty answer."""
    pytest.importorskip("litellm")
    import litellm
    _mock_litellm(monkeypatch, [litellm.ModelResponse(choices=[])])
    out = client.chat_completions("q", model="gpt-test")
    assert out.choices[0].message.content == ""


def _refusal(field):
    from litellm.exceptions import BadRequestError
    return BadRequestError(message=f"property '{field}' is unsupported",
                           model="m", llm_provider="groq")


@needs_agents
@pytest.mark.parametrize("settings", [{}, {"stream": True}])
def test_refused_field_goes_again_without_it(client, store_path, monkeypatch,
                                             settings):
    """A field a 400 names goes again at once without it and stays out for
    the rest of the call; the next call tries it afresh."""
    pytest.importorskip("litellm")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    call = _litellm_reply(tool_calls=_TWO_CALLS[:1], reasoning_content="think")
    sent = _mock_litellm(monkeypatch, [
        call, _refusal("reasoning_content"),
        _litellm_reply(tool_calls=_TWO_CALLS[1:]), _litellm_reply("A"),
        call, _litellm_reply("A")])
    for _ in range(2):
        out = client.chat_completions("q", doc_id=doc_id,
                                      model="groq/qwen/qwen3-32b", **settings)
        if settings:
            assert "A" in "".join(out)
        else:
            assert out.choices[0].message.content == "A"
    assert [any(m.get("reasoning_content") for m in messages)
            for messages in sent] == [False, True, False, False, False, True]


@needs_agents
def test_missing_field_400_costs_one_retry_at_most(client, store_path,
                                                    monkeypatch):
    """A 400 naming a field the provider wants rather than refuses gets one
    retry without it; the provider's own error then surfaces, and the next
    call sends the field again."""
    pytest.importorskip("litellm")
    from litellm.exceptions import BadRequestError
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    model = "deepseek/deepseek-chat"
    call = _litellm_reply(tool_calls=_TWO_CALLS[:1], reasoning_content="think")
    missing = "Missing `reasoning_content` field in the assistant message"
    sent = _mock_litellm(monkeypatch, [
        call, *[BadRequestError(message=missing, model="m",
                                llm_provider="deepseek") for _ in range(2)],
        call, _litellm_reply("A")])
    with pytest.raises(PageIndexAPIError, match=missing):
        client.chat_completions("q", doc_id=doc_id, model=model)
    client.chat_completions("q", doc_id=doc_id, model=model)
    assert [any(m.get("reasoning_content") for m in messages)
            for messages in sent] == [False, True, False, False, True]


@needs_agents
def test_refused_fields_go_again_one_by_one(client, store_path, monkeypatch):
    """Each 400 refuses the field it names, the Agents SDK's own too
    (reasoning_content, for a DeepSeek name), until one goes through."""
    pytest.importorskip("litellm")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    model = "groq/deepseek-r1-distill-llama-70b"
    details = [{"type": "reasoning.text", "text": "plan"}]
    sent = _mock_litellm(monkeypatch, [
        _litellm_reply(tool_calls=_TWO_CALLS[:1], reasoning_content="plan",
                       provider_specific_fields={"reasoning_details": details}),
        _refusal("reasoning_content"), _refusal("reasoning_details"),
        _litellm_reply("A")])
    out = client.chat_completions("q", doc_id=doc_id, model=model)
    assert out.choices[0].message.content == "A"
    assert [sorted({"reasoning_content", "reasoning_details"} & set(m))
            for messages in sent[1:] for m in messages
            if m.get("tool_calls")] == [
        ["reasoning_content", "reasoning_details"], ["reasoning_details"], []]


def _aborted(*chunks):
    """A stream OpenRouter cuts off after chunks, as LiteLLM reads it."""
    from litellm.exceptions import APIConnectionError
    return [*chunks, APIConnectionError(
        message="The operation was aborted", llm_provider="openai",
        model="m")]


@needs_agents
@pytest.mark.parametrize("breaks", [1, 3, 4])
def test_stream_broken_before_output_goes_again(client, monkeypatch, breaks):
    """A stream the server breaks before any output goes again, three
    times at most."""
    pytest.importorskip("litellm")
    sent = _mock_litellm(monkeypatch, [_aborted() for _ in range(breaks)]
                         + [_litellm_reply("A")])
    out = client.chat_completions("q", model="gpt-test", stream=True)
    if breaks <= 3:
        assert "".join(out) == "A"
    else:
        with pytest.raises(PageIndexAPIError):
            "".join(out)
    assert len(sent) == min(breaks + 1, 4)


@needs_agents
def test_stream_broken_after_output_is_not_replayed(client, monkeypatch):
    """The runner won't replay a stream that already showed text, nor
    does the lane retry a request that wasn't streamed."""
    pytest.importorskip("litellm")
    from litellm.exceptions import InternalServerError
    sent = _mock_litellm(monkeypatch, [
        _aborted(*_litellm_chunks({"content": "Hi"})[:1]),
        InternalServerError(message="boom", llm_provider="openai", model="m"),
        _litellm_reply("A")])
    with pytest.raises(PageIndexAPIError):
        "".join(client.chat_completions("q", model="gpt-test", stream=True))
    with pytest.raises(PageIndexAPIError):
        client.chat_completions("q", model="gpt-test")
    assert len(sent) == 2


@needs_agents
def test_other_400s_are_not_retried(client, monkeypatch):
    """A 400 that doesn't name reasoning_content is not retried."""
    pytest.importorskip("litellm")
    import litellm
    from litellm.exceptions import BadRequestError
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        raise BadRequestError(message="bad request", model=kwargs["model"],
                              llm_provider="groq")

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    with pytest.raises(PageIndexAPIError):
        client.chat_completions("q", model="groq/never-sent-reasoning")
    assert len(calls) == 1


@needs_agents
@pytest.mark.parametrize("error", [
    # names fields this request didn't get back, or only as part of a name
    "messages.1.content.0: Invalid `signature` in `thinking` block",
    "reasoning_content_mode is unsupported"])
def test_unrelated_400s_refuse_nothing(client, store_path, monkeypatch,
                                       error):
    """A 400 naming no field the request carried is not retried."""
    pytest.importorskip("litellm")
    from litellm.exceptions import BadRequestError
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    call = _litellm_reply(tool_calls=_TWO_CALLS[:1], reasoning_content="think")
    sent = _mock_litellm(monkeypatch, [call, BadRequestError(
        message=error, model="m", llm_provider="anthropic")])
    with pytest.raises(PageIndexAPIError):
        client.chat_completions("q", doc_id=doc_id,
                                model="moonshot/kimi-k2-thinking")
    assert [any(m.get("reasoning_content") for m in messages)
            for messages in sent] == [False, True]


@needs_agents
def test_unset_effort_sends_none(client, monkeypatch):
    """No effort reaches the wire unless the caller sets one."""
    pytest.importorskip("litellm")
    model = "anthropic/claude-sonnet-5-5"
    unset, _ = _anthropic_wire(monkeypatch, client, model)
    assert not {"thinking", "output_config"} & set(unset["body"])
    chosen, _ = _anthropic_wire(monkeypatch, client, model,
                                reasoning_effort="high")
    assert chosen["body"]["output_config"] == {"effort": "high"}


@needs_agents
@pytest.mark.parametrize("effort", [{"reasoning_effort": "high"}, {}])
def test_streamed_claude_thinking_goes_back_once(client, store_path,
                                                 monkeypatch, effort):
    """Streamed thinking goes back once, not doubled."""
    pytest.importorskip("litellm")
    call = {"index": 0, **_TWO_CALLS[0]}
    first, sent = _two_turns(
        store_path, monkeypatch, client, "anthropic/claude-sonnet-5-5",
        _litellm_chunks(
            {"reasoning_content": "plan", "thinking_blocks": [
                {"type": "thinking", "thinking": "plan"}]},
            {"reasoning_content": " step", "thinking_blocks": [
                {"type": "thinking", "thinking": " step"}]},
            {"reasoning_content": "", "thinking_blocks": [
                {"type": "thinking", "thinking": "plan step",
                 "signature": "SIG1"}]},
            {"tool_calls": [call]}),
        **effort, stream=True, stream_metadata=True)
    thinking = [{"type": "thinking", "thinking": "plan step",
                 "signature": "SIG1"}]
    for messages in (first["messages"], sent[1], sent[2]):
        calls = [m for m in messages if m.get("tool_calls")]
        assert calls[0].get("thinking_blocks") == thinking


@needs_agents
@pytest.mark.parametrize("model, effort, extra_body", [
    ("anthropic/claude-test", "high", None),
    ("gemini/gemini-test", "disable", None),
    # extra_body's reasoning names no effort, so the argument's stays
    ("anthropic/claude-test", "high", {"reasoning": {"summary": "auto"}}),
    ("anthropic/claude-test", "high", {"reasoning": Reasoning(summary="auto")}),
    ("anthropic/claude-test", "high",
     {"reasoning": Reasoning(summary="auto").model_dump()}),
    ("anthropic/claude-test", "high", {"reasoning": None}),
])
def test_reasoning_effort_reaches_litellm_as_given(client, monkeypatch,
                                                   model, effort, extra_body):
    pytest.importorskip("litellm")
    import litellm
    seen = []

    async def acompletion(**kwargs):
        seen.append(kwargs.get("reasoning_effort"))
        return _litellm_reply("ok")

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    client.chat_completions("q", model=model, reasoning_effort=effort,
                            extra_body=extra_body)
    assert seen == [effort]


@needs_agents
def test_extra_body_settings_litellm_model_skips_reach_litellm(client,
                                                               monkeypatch):
    """Settings LitellmModel ignores reach LiteLLM as kwargs."""
    pytest.importorskip("litellm")
    import litellm
    seen = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        return _litellm_reply("ok")

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    extras = {"context_management": {"edits": []}, "verbosity": "low",
              "store": False, "retry": {"max_retries": 2}}
    client.chat_completions("q", model="anthropic/claude-test",
                            extra_body=extras)
    assert {key: seen[0].get(key) for key in extras} == extras


@needs_agents
def test_chat_completions_messages_keep_gemini_signatures(client, store_path,
                                                          monkeypatch):
    """Gemini thought signatures ride messages back in LiteLLM's own field,
    as the model was sent them, and stay on every turn's calls, so the next
    request opens with the previous one unchanged."""
    pytest.importorskip("litellm")
    signed = [{**_TWO_CALLS[0], "provider_specific_fields": {
        "thought_signature": "sig1"}}, _TWO_CALLS[1]]
    first, sent = _two_turns(store_path, monkeypatch, client,
                             "gemini/gemini-test",
                             _litellm_reply(tool_calls=signed))
    assert first["messages"][0]["tool_calls"][0][
        "provider_specific_fields"] == {"thought_signature": "sig1"}
    assert sent[2][:len(sent[1])] == sent[1]
    calls = next(m["tool_calls"] for m in sent[2] if m.get("tool_calls"))
    assert calls[0]["provider_specific_fields"] == {
        "thought_signature": "sig1"}


@needs_agents
def test_gemini_signatures_go_back_on_a_continued_run(client, store_path,
                                                      monkeypatch):
    """A run that goes on from a tool result sends the call's own thought
    signature, not the placeholder the Agents SDK puts on calls after the
    last user message, so it opens with the previous call's request."""
    pytest.importorskip("litellm")
    model = "gemini/gemini-test"
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    signed = [{**_TWO_CALLS[0], "provider_specific_fields": {
        "thought_signature": "sig1"}}]
    sent = _mock_litellm(monkeypatch, [
        _litellm_reply(tool_calls=signed), _litellm_reply("A1"),
        _litellm_reply("A2")])
    question = {"role": "user", "content": "q"}
    first = client.chat_completions([question], doc_id=doc_id, model=model)
    client.chat_completions([question, *first["messages"][:2]],
                            doc_id=doc_id, model=model)
    assert sent[2][-2]["tool_calls"][0]["provider_specific_fields"] == {
        "thought_signature": "sig1"}
    assert sent[2] == sent[1]


@needs_agents
@pytest.mark.parametrize("field, value", [
    ("reasoning_details", [{"type": "reasoning.encrypted", "data": "ENC"}]),
    ("reasoning_items", [{"id": "rs_1", "type": "reasoning",
                          "encrypted_content": "ENC", "summary": []}]),
])
def test_reply_fields_the_agents_sdk_drops_go_back(client, store_path,
                                                   monkeypatch, field, value):
    """What LiteLLM parsed from a reply and the Agents SDK drops
    (OpenRouter's reasoning_details, the Responses bridge's
    reasoning_items, ...) goes back with that reply on every later
    request, and rides the returned messages."""
    pytest.importorskip("litellm")
    first, sent = _two_turns(store_path, monkeypatch, client,
                             "openrouter/x-ai/grok-test",
                             _litellm_reply(tool_calls=_TWO_CALLS[:1],
                                            provider_specific_fields={
                                                field: value}))
    for messages in (first["messages"], sent[1], sent[2]):
        calls = [m for m in messages if m.get("tool_calls")]
        assert calls[0][field] == value


def _openai_chunk(delta, finish=None):
    from openai.types.chat import ChatCompletionChunk
    return ChatCompletionChunk.model_validate({
        "id": "c1", "object": "chat.completion.chunk", "created": 0,
        "model": "m", "choices": [{"index": 0, "delta": delta,
                                   "finish_reason": finish}]})


@needs_agents
def test_streamed_fields_litellm_drops_go_back(client, store_path,
                                               monkeypatch):
    """A delta carrying only a field LiteLLM doesn't know, as OpenRouter
    streams a signature or encrypted reasoning, still goes back."""
    pytest.importorskip("litellm")
    text = {"type": "reasoning.text", "text": "plan"}
    encrypted = {"type": "reasoning.encrypted", "data": "ENC"}
    first, sent = _two_turns(
        store_path, monkeypatch, client, "openrouter/x-ai/grok-test",
        [_openai_chunk({"role": "assistant", "reasoning_details": [text],
                        "trace": "pl"}),
         _openai_chunk({"tool_calls": [{"index": 0, **_TWO_CALLS[0]}],
                        "trace": "an"}),
         _openai_chunk({"reasoning_details": [encrypted]}),
         _openai_chunk({}, finish="tool_calls")],
        stream=True, stream_metadata=True)
    for messages in (first["messages"], sent[1], sent[2]):
        calls = [m for m in messages if m.get("tool_calls")]
        assert calls[0]["reasoning_details"] == [text, encrypted]
        assert calls[0]["trace"] == "plan"


def _model_rate_limited():
    """LiteLLM's 429 as a server sent it, asking for a 0.01 s wait."""
    from litellm.exceptions import RateLimitError
    request = httpx.Request("POST", "https://llm.example/v1/messages")
    response = httpx.Response(429, headers={"retry-after": "0.01"}, request=request)
    err = RateLimitError("slow down", llm_provider="anthropic", model="claude-test",
                         response=response)
    setattr(err, "litellm_response_headers", response.headers)
    err.__cause__ = httpx.HTTPStatusError("reply", request=request, response=response)
    return err


def _model_key_missing():
    """LiteLLM's missing-credentials 500, raised before any request."""
    from litellm.exceptions import InternalServerError
    return InternalServerError("Missing credentials", llm_provider="anthropic",
                               model="claude-test")


@needs_agents
@pytest.mark.parametrize("stream", [False, True])
def test_claude_thinking_reaches_the_anthropic_wire_once(client, store_path,
                                                         monkeypatch, stream):
    """Through LiteLLM's own Anthropic parsing and request building, with
    no effort set: the run's next request carries the signed thinking
    block once, a stream's closing repeat of its text dropped."""
    pytest.importorskip("litellm")
    from litellm.llms.custom_httpx.http_handler import (AsyncHTTPHandler,
                                                        HTTPHandler)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    thinking = {"type": "thinking", "thinking": "plan step",
                "signature": "SIG1"}
    call = {"type": "tool_use", "id": "toolu_1", "name": "get_document",
            "input": {"doc_name": "report.pdf"}}
    answer = {"type": "text", "text": "done"}
    replies = [_anthropic_message([thinking, call], "tool_use"),
               _anthropic_message([answer], "end_turn")]
    if stream:
        replies = [_anthropic_sse(message) for message in replies]
    bodies = []

    def reply(url, kwargs):
        bodies.append(json.loads(kwargs["data"]) if kwargs.get("data")
                      else kwargs.get("json"))
        request = httpx.Request("POST", str(url))
        body = replies.pop(0)
        if isinstance(body, str):
            return httpx.Response(200, content=body.encode(), request=request,
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=body, request=request)

    async def fake_apost(self, url=None, *args, **kwargs):
        return reply(url, kwargs)

    monkeypatch.setattr(AsyncHTTPHandler, "post", fake_apost)
    monkeypatch.setattr(HTTPHandler, "post",
                        lambda self, url=None, *args, **kwargs:
                        reply(url, kwargs))
    out = client.chat_completions("q", doc_id=doc_id,
                                  model="anthropic/claude-sonnet-5-5",
                                  stream=stream, stream_metadata=stream)
    result = list(out)[-1] if stream else out
    assert not {"thinking", "output_config"} & set(bodies[0])
    turn = next(m for m in bodies[1]["messages"] if m["role"] == "assistant")
    assert [b for b in turn["content"] if b["type"] == "thinking"] == [
        thinking]
    assert result["messages"][0]["thinking_blocks"] == [thinking]


def test_repeated_answers_keep_their_own_fields():
    """Records keep conversation order: a repeated answer (or a reused call
    id) gets its own reply's fields back, not the latest one's."""
    replies = local_chat._Replies("m")
    for n in (1, 2):
        replies.record({"role": "assistant", "content": "OK",
                        "reasoning_details": [n]})
    out = replies.fill([{"role": "assistant", "content": "OK"},
                        {"role": "user", "content": "again"},
                        {"role": "assistant", "content": "OK"}])
    assert [m.get("reasoning_details") for m in out
            if m["role"] == "assistant"] == [[1], [2]]


def test_fill_puts_the_reply_back_as_it_was():
    """A reply's fields go back as it had them, empty ones too, over what
    the Agents SDK wrote in their place; a field it lacks (None) stays out;
    LiteLLM's nested provider fields go on top."""
    call = {"id": "c1", "type": "function",
            "function": {"name": "f", "arguments": "{}"}}
    signed = {**call, "caller": {"type": "direct"},
              "provider_specific_fields": {"thought_signature": "SIG"}}
    thinking = [{"type": "thinking", "thinking": "a", "signature": "S"}]
    replies = local_chat._Replies("m")
    replies.record({"role": "assistant", "content": "",
                    "tool_calls": [signed], "thinking_blocks": thinking,
                    "reasoning_content": "", "audio": None,
                    "provider_specific_fields": {"reasoning_details": [],
                                                 "extra": "x"}})
    out = replies.fill([{
        "role": "assistant",
        "thinking_blocks": [{"type": "thinking", "thinking": "b",
                             "signature": "T"}],
        "tool_calls": [{**call, "provider_specific_fields": {
            "thought_signature": "skip_thought_signature_validator"}}]}])
    assert out == [{"role": "assistant", "content": None,
                    "thinking_blocks": thinking, "reasoning_content": "",
                    "reasoning_details": [], "extra": "x",
                    "tool_calls": [signed]}]


@needs_agents
def test_empty_reasoning_goes_back(client, store_path, monkeypatch):
    """A reply's empty reasoning_content goes back as it had it: DeepSeek's
    thinking mode refuses a request whose replies in the current turn lack
    the field, and its replies sometimes reason nothing."""
    pytest.importorskip("litellm")
    first, sent = _two_turns(store_path, monkeypatch, client,
                             "deepseek/deepseek-chat",
                             _litellm_reply(tool_calls=_TWO_CALLS[:1],
                                            reasoning_content=""))
    for messages in (first["messages"], sent[1], sent[2]):
        calls = [m for m in messages if m.get("tool_calls")]
        assert calls[0].get("reasoning_content") == ""


@needs_agents
def test_native_routes_keep_provider_fields_nested(client, store_path,
                                                   monkeypatch):
    """LiteLLM's own adapter reads provider_specific_fields nested (Gemini's
    thought_signatures, Anthropic's web_search_results), so they go back
    there; a Gemini call id matches without the signature LiteLLM folds in
    and the Agents SDK strips."""
    pytest.importorskip("litellm")
    fields = {"thought_signatures": ["SIG"]}
    first, sent = _two_turns(store_path, monkeypatch, client,
                             "gemini/gemini-test", _litellm_reply(
                                 tool_calls=[{**_TWO_CALLS[0],
                                              "id": "call_1__thought__SIG"}],
                                 provider_specific_fields=fields))
    for messages in (first["messages"], sent[1], sent[2]):
        turn = next(m for m in messages if m.get("tool_calls"))
        assert turn["provider_specific_fields"] == fields
        assert "thought_signatures" not in turn


def test_fields_the_sdk_set_count_as_sent():
    """A field the Agents SDK set itself is refusable like one put back."""
    replies = local_chat._Replies("m")
    replies.record({"role": "assistant", "content": "OK"}, own=False)
    message = {"role": "assistant", "content": "OK", "reasoning_content": "r"}
    replies.fill([message])
    assert replies.sent == {"reasoning_content"}
    replies.refused.add("reasoning_content")
    assert replies.fill([message]) == [{"role": "assistant", "content": "OK"}]


def test_unassembled_streams_are_skipped(monkeypatch):
    pytest.importorskip("litellm")
    import litellm

    def fail(chunks):
        raise ValueError("unassembled")

    monkeypatch.setattr(litellm, "stream_chunk_builder", fail)
    replies = local_chat._Replies("m")
    replies.streams.append((["chunk"], {"reasoning_details": [[1]]}))
    message = {"role": "assistant", "content": "OK"}
    assert replies.fill([message]) == [message]


def test_requests_outside_a_chat_run_pass_untouched():
    pytest.importorskip("litellm")
    local_chat._register_fill()
    kwargs = {"messages": [{"role": "assistant", "content": "OK"}]}
    assert asyncio.run(local_chat._fill_hook.async_pre_call_deployment_hook(
        kwargs, None)) is None


@needs_agents
def test_fill_hook_returns_when_callbacks_are_replaced(client, store_path,
                                                       monkeypatch):
    """A caller replacing litellm.callbacks mid-run doesn't turn the fill
    off: each request puts the hook back."""
    pytest.importorskip("litellm")
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _mock_litellm(monkeypatch, [_litellm_reply(tool_calls=_TWO_CALLS[:1]),
                                _litellm_reply("A")])

    class Replace(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            litellm.callbacks = [callback for callback in litellm.callbacks
                                 if callback is not local_chat._fill_hook]

    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, Replace()])
    messages = client.chat_completions("q", doc_id=doc_id,
                                       model="gpt-test")["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert all(m.get("model") == "openai/gpt-test" for m in messages
               if m["role"] == "assistant")


@needs_agents
def test_a_request_the_fill_missed_returns_the_sdks_messages(
        client, store_path, monkeypatch):
    """A request that went out without the fill (here a hook ahead of it
    drops the call id) leaves no record of what was sent, so the returned
    messages are the SDK's own conversion, tool turns whole."""
    pytest.importorskip("litellm")
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _mock_litellm(monkeypatch, [_litellm_reply(tool_calls=_TWO_CALLS[:1]),
                                _litellm_reply("A")])

    class Strip(CustomLogger):
        requests = 0

        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            Strip.requests += 1
            if Strip.requests == 2:
                return {key: value for key, value in kwargs.items()
                        if key != "litellm_call_id"}

    monkeypatch.setattr(litellm, "callbacks", [Strip(), *litellm.callbacks])
    messages = client.chat_completions("q", doc_id=doc_id,
                                       model="gpt-test")["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert not any("model" in m for m in messages)


@needs_agents
def test_another_callbacks_own_call_is_not_the_runs(client, store_path,
                                                    monkeypatch):
    """A LiteLLM call another callback makes during a chat request is
    neither filled nor recorded as the run's, so the returned messages
    keep the run's tool turns."""
    pytest.importorskip("litellm")
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    _mock_litellm(monkeypatch, [
        _litellm_reply(tool_calls=_TWO_CALLS[:1]), _litellm_reply("checked"),
        _litellm_reply("A"), _litellm_reply("checked")])
    check = {"role": "user", "content": "check"}

    class Guard(CustomLogger):
        async def async_pre_call_deployment_hook(self, kwargs, call_type):
            if kwargs["messages"][-1] != check:
                await litellm.acompletion(model="openai/guard",
                                          messages=[check], api_key="k")

    monkeypatch.setattr(litellm, "callbacks", [*litellm.callbacks, Guard()])
    messages = client.chat_completions("q", doc_id=doc_id,
                                       model="gpt-test")["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert all(m.get("model") == "openai/gpt-test" for m in messages
               if m["role"] == "assistant")
    assert messages[-1]["content"] == "A"


def test_gateways_get_cache_marks_for_claude():
    """Claude caches only on marks, so it gets them through a gateway too.
    Every other model caches by itself and gets none, as on OpenAI's own
    API, where marks turn on explicit caching and lose its automatic
    hits."""
    pytest.importorskip("litellm")
    relay = "https://openrouter.ai/api/v1"
    for name in ("openai/anthropic/claude-x", "litellm/openai/Claude-x"):
        assert local_chat._cache_extra_args(name, relay), name
    assert local_chat._cache_extra_args("openrouter/anthropic/claude-x")
    for name, base in (("openai/x-ai/grok-x", relay), ("gpt-x", relay),
                       ("litellm/openai/qwen/x", relay),
                       ("openai/deepseek-chat", "https://api.deepseek.com"),
                       ("openrouter/x-ai/grok-x", None), ("gpt-x", None),
                       ("openai/gpt-x", "https://api.openai.com/v1"),
                       ("openai/gpt-x", "https://us.api.openai.com/v1"),
                       ("gemini/gemini-x", relay)):
        assert local_chat._cache_extra_args(name, base) is None, (name, base)


@needs_agents
@pytest.mark.parametrize("model, backend, env, blocks", [
    ("openai/anthropic/claude-test",
     {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}, None,
     True),
    # the gateway LiteLLM reads from the environment
    ("openai/anthropic/claude-test", None, "https://openrouter.ai/api/v1",
     True),
    ("anthropic/claude-test", None, None, False),
])
def test_gateway_requests_send_user_text_as_blocks(client, store_path,
                                                   monkeypatch, model,
                                                   backend, env, blocks):
    """Through a gateway every request carries the marks with user text
    as content blocks: a gateway keeps a mark only inside a block and
    merges adjacent text messages, so a message keeps one shape whether
    or not it holds this request's mark. A Claude route sends the text as
    is, and LiteLLM places the marks itself."""
    pytest.importorskip("litellm")
    if env:
        monkeypatch.setenv("OPENAI_BASE_URL", env)
    marks = []
    first, sent = _two_turns(store_path, monkeypatch, client, model,
                             _litellm_reply(tool_calls=_TWO_CALLS[:1]),
                             marks, backend=backend)
    assert len(marks) == len(sent) == 3 and all(marks)
    users = [m for messages in sent for m in messages if m["role"] == "user"]
    assert users and all(isinstance(m["content"], list) == blocks
                         for m in users)
    assert sent[2][:len(sent[1])] == sent[1]


@needs_agents
def test_gateway_refusing_marks_goes_again_without_them(client, store_path,
                                                        monkeypatch):
    """A gateway that refuses cache_control gets the request again without
    the marks, and none for the rest of the call; the next call tries them
    afresh."""
    pytest.importorskip("litellm")
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    marks = []
    _mock_litellm(monkeypatch, [
        _refusal("cache_control"), _litellm_reply(tool_calls=_TWO_CALLS[:1]),
        _litellm_reply("A"), _refusal("cache_control"), _litellm_reply("A")],
        marks)
    backend = {"base_url": "https://gateway.example/v1", "api_key": "k"}
    for _ in range(2):
        out = client.chat_completions("q", doc_id=doc_id, backend=backend,
                                      model="openai/anthropic/claude-strict")
        assert out.choices[0].message.content == "A"
    assert [bool(points) for points in marks] == [True, False, False, True,
                                                  False]


@needs_agents
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("error, fails, calls", [
    (_model_rate_limited, 2, 3), (_model_rate_limited, 9, 4), (_model_key_missing, 9, 1)],
    ids=["recovers", "exhausted", "local"])
def test_chat_lane_retries_as_the_official_sdks(client, store_path, monkeypatch,
                                                stream, error, fails, calls):
    pytest.importorskip("litellm")
    import litellm
    from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    sent = []

    async def acompletion(**kwargs):
        sent.append(kwargs.get("max_retries"))
        if len(sent) <= fails:
            raise error()
        reply = _litellm_reply("ok")
        return MockResponseIterator(reply) if kwargs.get("stream") else reply

    monkeypatch.setattr(litellm, "acompletion", acompletion)

    def ask():
        out = client.chat_completions([{"role": "user", "content": "q"}], doc_id=doc_id,
                                      model="anthropic/claude-test", stream=stream)
        return list(out) if stream else out

    if fails < calls:
        ask()
    else:
        with pytest.raises(PageIndexAPIError):
            ask()
    assert sent == [0] * calls


@needs_agents
def test_chat_retry_waits_as_the_server_asks():
    pytest.importorskip("litellm")
    from agents.retry import RetryDecision
    from pageindex.retry import agent_policy
    assert agent_policy(types.SimpleNamespace(error=_model_rate_limited(), attempt=1)) == \
        RetryDecision(retry=True, delay=0.01)
    assert agent_policy(types.SimpleNamespace(error=_model_key_missing(), attempt=1)) is False


@needs_agents
def test_chat_retry_budget_follows_the_constant(monkeypatch):
    monkeypatch.setattr("pageindex.retry.CHAT_RETRIES", 5)
    assert local_chat._chat_retry(local_chat._Replies("m")).max_retries == 5


def test_a_stream_error_retries_by_its_status():
    """A server's error inside a stream has no httpx error behind it; its
    status decides: OpenRouter's abort goes again, a content-policy
    refusal (a 400 LiteLLM's Responses stream raises this way) does not."""
    pytest.importorskip("litellm")
    from litellm.exceptions import (APIConnectionError,
                                    ContentPolicyViolationError,
                                    MidStreamFallbackError)
    from pageindex.retry import should_retry

    def in_stream(error):
        return MidStreamFallbackError(message=str(error), model="m",
                                      llm_provider="openai",
                                      original_exception=error)

    assert should_retry(in_stream(APIConnectionError(
        message="The operation was aborted", llm_provider="openai",
        model="m")))
    assert not should_retry(in_stream(ContentPolicyViolationError(
        message="Invalid prompt", llm_provider="openai", model="m")))


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
@pytest.mark.parametrize("extra", [{"refusal": None},
                                   {"reasoning_content": "think"}])
def test_chat_empty_answer_row_stays_text(client, store_path, fake_model,
                                          extra):
    fake = fake_model([[_msg_item("ok")]])
    empty = {"role": "assistant", "content": ""}
    client.chat([{"role": "user", "content": "q1"}, {**empty, **extra},
                 {"role": "user", "content": "q2"}])
    assert fake.inputs[0][-2] == empty


@needs_agents
def test_chat_reasoning_effort_reaches_the_engine(client, store_path,
                                                  fake_model, monkeypatch):
    """The business door's one thinking knob rides chat_completions'
    channel unchanged; unset sends no effort."""
    seen = {}
    real = local_chat._openai_agent

    def spy(*args, **kwargs):
        agent = real(*args, **kwargs)
        seen["settings"] = agent.model_settings
        return agent

    monkeypatch.setattr(local_chat, "_openai_agent", spy)
    fake_model([[_msg_item("ok")]])
    client.chat("q", reasoning_effort="low")
    assert seen["settings"].reasoning.effort == "low"
    fake_model([[_msg_item("ok")]])
    client.chat("q")
    assert seen["settings"].extra_args == {"max_retries": 0}


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
        client.chat("q", show_process={})
    # an invalid value is refused as such, with or without stream=True —
    # never told to add stream=True first
    for kwargs in ({}, {"stream": True}):
        with pytest.raises(PageIndexAPIError,
                           match="must be True, False, or a dict"):
            client.chat("q", show_process=0, **kwargs)


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


@pytest.mark.parametrize("finish", ["length", "max_tokens", "refusal",
                                    "model_context_window_exceeded"])
def test_managed_chat_raises_on_an_unfinished_answer(monkeypatch, finish):
    """The managed answer lane holds an unfinished answer to the same
    rule: a stream raises after its text, a plain call carries the text
    on the error (no usage: it is not an own-model run)."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    cut = _cloud_chunk(None)
    cut["choices"][0]["finish_reason"] = finish

    def fake_cc(**kwargs):
        if kwargs.get("stream"):
            return iter([_cloud_chunk("The range was"), cut])
        return {"choices": [{"message": {"content": "The range was"},
                             "finish_reason": finish}]}

    monkeypatch.setattr(cloud._api, "chat_completions", fake_cc)
    text = []
    with pytest.raises(PageIndexAPIError, match=finish):
        for piece in cloud.chat("q", stream=True):
            text.append(piece)
    assert "".join(text) == "The range was"
    with pytest.raises(PageIndexAPIError, match=finish) as info:
        cloud.chat("q")
    assert info.value.partial_answer == "The range was"
    assert info.value.usage is None


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
    # results stand alone, each echoing its call's arguments
    assert '\n\n[tool_result] get_document {"doc_name": "report.pdf"}: ' in no_calls
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
    # The result as the framework recorded it: its text item.
    assert '"next_steps"' in result["output"]["text"]
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


@needs_agents
def test_chat_stream_events_survive_partial_reads(client, store_path,
                                                  fake_model):
    """Peek at one event, then read the rest: the dropped .events handle
    must not close the run underneath."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    stream = client.chat("What status?", stream=True)
    first = next(stream.events)
    rest = list(stream.events)
    assert first["type"] == "tool_call"
    assert [ev["type"] for ev in rest] == ["tool_result", "answer", "answer"]


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
    call = {"input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0,
                                     "cache_write_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15}
    assert result["usage"] == {
        "input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 10,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 30,
        "request_usage_entries": [call, call]}
    assert fake_model.state["protocols"][0][0] == "responses"
    assert [item.get("type", "message") for item in result["output"]] == [
        "function_call", "message"]
    assert [item.get("type", "message") for item in result["messages"]] == [
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
                 + result["messages"]
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
                 + result["messages"]
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

    def spy(model_name, instructions, doc_id, items, folder_id=None):
        key = real(model_name, instructions, doc_id, items, folder_id)
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
                 + result["messages"]
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


def test_folder_less_cache_key_is_the_pre_folder_key():
    """Adding folder_id to the seed must not rotate every existing
    conversation's prompt_cache_key on upgrade."""
    items = [{"role": "user", "content": "hi"}]
    key = local_chat._conversation_cache_key("m", "sys", "d1", items)
    assert key == "pageindex-b0ab095344ee8f89"
    assert local_chat._conversation_cache_key(
        "m", "sys", "d1", items, "f-1") != key


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
    assert [item.get("type", "message") for item in final["messages"]] == [
        "function_call", "function_call_output", "message"]
    # output_index addresses the logical response.output: turn 2's deltas
    # are re-based past turn 1's item instead of restarting at 0.
    last_delta = [event for event in events
                  if event.get("type") == "response.output_text.delta"][-1]
    assert (final["output"][last_delta["output_index"]]
            .get("type", "message") == "message")


def _cut_response():
    """A billed response cut at the output cap: its text, and a tool call
    it never finished, each after its reasoning."""
    from openai.types.responses import Response, ResponseReasoningItem
    from openai.types.responses.response import IncompleteDetails
    from openai.types.responses.response_usage import (
        InputTokensDetails, OutputTokensDetails, ResponseUsage)
    return Response(
        id="resp_cut", created_at=0.0, model="fake", object="response",
        output=[ResponseReasoningItem(id="rs_1", type="reasoning", summary=[]),
                _msg_item("cut"),
                ResponseReasoningItem(id="rs_2", type="reasoning", summary=[]),
                _call_item("get_document", {"doc_n": 1}, "call_9")],
        parallel_tool_calls=False, tool_choice="auto", tools=[],
        status="incomplete",
        incomplete_details=IncompleteDetails(reason="max_output_tokens"),
        usage=ResponseUsage(
            input_tokens=40, output_tokens=99, total_tokens=139,
            input_tokens_details=InputTokensDetails(
                cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0)))


@needs_agents
def test_responses_stream_counts_a_call_cut_short(client, store_path,
                                                  fake_model):
    """openai-agents raises on response.incomplete before its hooks or
    items see the call, yet that call was billed and its text streamed:
    the terminal envelope counts it and keeps the text, but not a
    half-written tool call, which has no output to pair with."""
    from openai.types.responses import ResponseIncompleteEvent
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([[_call_item("get_document", {"doc_name": "report.pdf"})],
                       [_msg_item("cut")]])
    first_turn = fake.stream_response

    async def stream_response(*args, **kwargs):
        if not fake.inputs:
            async for event in first_turn(*args, **kwargs):
                yield event
            return
        fake._record(args[0], args[1])
        yield ResponseIncompleteEvent(type="response.incomplete",
                                      sequence_number=1,
                                      response=_cut_response())

    fake.stream_response = stream_response
    final = list(client._responses("q", stream=True))[-1]
    assert final["type"] == "response.incomplete"
    usage = final["response"]["usage"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (50, 104)
    assert len(usage["request_usage_entries"]) == 2
    assert [item.get("id") for item in final["response"]["messages"]] == [
        "fc_1", None, "rs_1", "msg_1"]
    assert final["response"]["output"][-1]["content"][0]["text"] == "cut"


@needs_agents
def test_responses_counts_a_call_cut_short(client, store_path, fake_model):
    """The non-streamed twin: a response cut short ends the run once the
    transport returned it, before openai-agents' hooks see it, and the
    envelope keeps it, as above."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("never reached")]])
    replies = [types.SimpleNamespace(status="completed",
                                     incomplete_details=None, error=None),
               _cut_response()]

    async def create(*args, **kwargs):
        return replies.pop(0)

    fake._client = types.SimpleNamespace(
        responses=types.SimpleNamespace(create=create))
    result = client._responses("q")
    assert result["status"] == "incomplete"
    usage = result["usage"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (50, 104)
    assert len(usage["request_usage_entries"]) == 2
    assert [item.get("id") for item in result["messages"]] == [
        "fc_1", None, "rs_1", "msg_1"]
    assert result["output"][-1]["content"][0]["text"] == "cut"
    from openai.types.responses import Response
    Response.model_validate(result)  # still the official shape


def _same_thread(agen_factory):
    """_stream_sync without its pump thread: the generator advances only
    when the caller asks for the next event, so the caller's edits between
    events land at a known point."""
    loop = asyncio.new_event_loop()
    agen = agen_factory()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        loop.run_until_complete(agen.aclose())
        loop.close()


@needs_agents
def test_responses_in_progress_owns_its_snapshot_before_created_is_read(
        client, store_path, fake_model, monkeypatch):
    """in_progress copies the opening snapshot before created is handed
    out: an edit to created the moment it arrives stays out of it."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_msg_item("The answer")]]).emit_created = True
    monkeypatch.setattr(local_chat, "_stream_sync", _same_thread)
    # closed on a failed assert too: left to gc, the open run deadlocks
    with contextlib.closing(client._responses("q", stream=True)) as events:
        created = next(events)
        created["response"]["status"] = "edited"
        progress = next(events)
        assert progress["type"] == "response.in_progress"
        assert progress["response"]["status"] != "edited"
        list(events)


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
    in_progress = [e for e in events if e["type"] == "response.in_progress"]
    assert len(in_progress) == 1 and events[1] is in_progress[0]
    assert events[1]["sequence_number"] == 2
    assert events[1]["response"]["id"] == events[0]["response"]["id"]
    events[1]["response"]["status"] = "edited"  # each event owns its snapshot
    assert events[0]["response"]["status"] != "edited"
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
    assert parsed.model_dump()["messages"] == result["messages"]
    assert "items" not in parsed.model_dump()


@needs_agents
def test_responses_items_reads_messages_with_a_warning(client, store_path,
                                                       fake_model):
    """The transcript's old name still reads, warned, on the envelope and the
    stream's terminal one, and is never stored: keys and JSON carry it once."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_msg_item("The answer")], [_msg_item("The answer")]])
    for result in (client._responses("q"),
                   list(client._responses("q", stream=True))[-1]["response"]):
        with pytest.warns(DeprecationWarning, match='"items" is now "messages"'):
            assert result["items"] is result["messages"]
        with pytest.warns(DeprecationWarning):
            assert result.get("items") is result["messages"]
        with pytest.warns(DeprecationWarning):
            assert "items" in result
        assert "items" not in list(result)
        assert '"items"' not in json.dumps(result)


@needs_agents
def test_openai_lanes_read_by_key_and_by_attribute(client, store_path,
                                                   fake_model):
    """Chat Completions and Responses envelopes, chunks and events are dicts
    that also read as the official SDK objects: the same values both ways,
    an official field the dict lacks reads None, any other name raises."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_msg_item("The answer")]] * 4)
    r = client.chat_completions("q")
    assert isinstance(r, dict) and json.loads(json.dumps(r)) == r
    assert isinstance(r, ChatObject)
    # yaml writes it as the plain mapping it is, safe_dump included
    assert yaml.safe_load(yaml.safe_dump(r)) == r
    assert yaml.safe_load(yaml.dump(r)) == r
    for name in ("CSafeDumper", "CDumper"):  # libyaml's, when built
        if hasattr(yaml, name):
            assert yaml.safe_load(yaml.dump(r, Dumper=getattr(yaml, name))) == r
    bare = ChatObject({"a": 1})  # no official type behind it
    assert bare["a"] == 1 and not hasattr(bare, "a")
    assert getattr(bare, "a", None) is None
    assert (r.choices[0].message.content
            == r["choices"][0]["message"]["content"] == "The answer")
    assert r.system_fingerprint is None
    assert r.choices[0].message.refusal is None
    assert r.messages == r["messages"]
    with pytest.raises(AttributeError):
        r.choicez
    chunks = list(client.chat_completions("q", stream=True,
                                          stream_metadata=True))
    assert "".join(chunk.choices[0].delta.content or ""
                   for chunk in chunks if chunk.choices) == "The answer"
    assert chunks[-1].usage.total_tokens == chunks[-1]["usage"]["total_tokens"]
    r = client._responses("q")
    assert r.output_text == "The answer" and r.messages == r["messages"]
    with pytest.raises(AttributeError, match="read-only"):
        r.model = "other"  # would shadow r["model"], which stays the truth
    for clone in (copy.copy(r), copy.deepcopy(r), pickle.loads(pickle.dumps(r))):
        assert type(clone) is type(r) and clone == r
        assert clone.output_text == "The answer"
    events = list(client._responses("q", stream=True))
    assert [event.type for event in events] == [e["type"] for e in events]
    assert events[-1].response.output_text == "The answer"
    # as the SDK reads the wire: a value it doesn't know reads as sent
    odd = local_chat._ChatCompletion({"choices": [
        {"index": 0, "finish_reason": "max_tokens",
         "message": {"role": "assistant", "content": "cut"}}]})
    assert odd.choices[0].finish_reason == "max_tokens"
    assert odd.choices[0].message.content == "cut"
    future = local_chat._ResponsesEvent({"type": "response.future_event",
                                         "sequence_number": 1})
    assert future.type == "response.future_event"


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


# ── conformance: the official SDKs read each lane as their own wire ──

@needs_agents
def test_chat_completions_conforms_to_the_official_sdk(client, store_path,
                                                       monkeypatch):
    """Through the real LitellmModel, the envelope parses as ChatCompletion
    and the openai SDK's own stream accumulator rebuilds it from the
    chunks: same text, finish reason and usage."""
    import litellm
    from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
    from openai.lib.streaming.chat import ChatCompletionStreamState
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")

    def call(**settings):
        replies = [_litellm_reply(tool_calls=_TWO_CALLS[:1]),
                   _litellm_reply("The answer")]

        async def acompletion(**kwargs):
            reply = replies.pop(0)
            return (MockResponseIterator(reply) if kwargs.get("stream")
                    else reply)

        monkeypatch.setattr(litellm, "acompletion", acompletion)
        return client.chat_completions("q", doc_id=doc_id, model="gpt-test",
                                       **settings)

    envelope = ChatCompletion.model_validate(call())
    state = ChatCompletionStreamState()
    for chunk in call(stream=True, stream_metadata=True):
        state.handle_chunk(ChatCompletionChunk.model_validate(chunk))
    rebuilt = state.get_final_completion()
    assert rebuilt.choices[0].message.content == "The answer"
    assert (rebuilt.choices[0].message.content
            == envelope.choices[0].message.content)
    assert (rebuilt.choices[0].finish_reason
            == envelope.choices[0].finish_reason == "stop")
    assert rebuilt.usage and envelope.usage
    assert rebuilt.usage.total_tokens == envelope.usage.total_tokens == 30


_RESPONSES_USAGE = {"input_tokens": 10, "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0}}


def _responses_turn(response_id, item, status="completed"):
    return {"id": response_id, "object": "response", "created_at": 1,
            "status": status, "model": "gpt-test",
            "output": [item] if status == "completed" else [],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": _RESPONSES_USAGE if status == "completed" else None}


def _responses_sse(response_id, item):
    """One turn as the Responses API streams it: lifecycle, item, deltas."""
    opening = _responses_turn(response_id, item, "in_progress")
    events = [{"type": "response.created", "response": opening},
              {"type": "response.in_progress", "response": opening}]
    started = ({**item, "status": "in_progress", "content": []}
               if item["type"] == "message"
               else {**item, "status": "in_progress", "arguments": ""})
    events.append({"type": "response.output_item.added", "output_index": 0,
                   "item": started})
    if item["type"] == "function_call":
        events += [
            {"type": "response.function_call_arguments.delta",
             "item_id": item["id"], "output_index": 0,
             "delta": item["arguments"]},
            {"type": "response.function_call_arguments.done",
             "item_id": item["id"], "output_index": 0,
             "arguments": item["arguments"], "name": item["name"]}]
    else:
        part = {"type": "output_text", "text": "", "annotations": [],
                "logprobs": []}
        events.append({"type": "response.content_part.added",
                       "item_id": item["id"], "output_index": 0,
                       "content_index": 0, "part": part})
        events += [{"type": "response.output_text.delta",
                    "item_id": item["id"], "output_index": 0,
                    "content_index": 0, "delta": piece, "logprobs": []}
                   for piece in ("The ", "answer")]
        events += [
            {"type": "response.output_text.done", "item_id": item["id"],
             "output_index": 0, "content_index": 0, "text": "The answer",
             "logprobs": []},
            {"type": "response.content_part.done", "item_id": item["id"],
             "output_index": 0, "content_index": 0,
             "part": {**part, "text": "The answer"}}]
    events += [{"type": "response.output_item.done", "output_index": 0,
                "item": item},
               {"type": "response.completed",
                "response": _responses_turn(response_id, item)}]
    return "".join(
        f"event: {e['type']}\n"
        f"data: {json.dumps({**e, 'sequence_number': n})}\n\n"
        for n, e in enumerate(events, 1))


@needs_agents
def test_responses_conforms_to_the_official_sdk(client, store_path):
    """Over the real OpenAIResponsesModel and an HTTP-level fake of the
    Responses API, a two-turn run's envelope parses as Response and the
    openai SDK's stream accumulator takes its events: one logical
    response, opened by created then in_progress, monotonic sequence
    numbers, each delta on the item its output_index names."""
    from openai import omit
    from openai.lib.streaming.responses import ResponseStreamState
    from openai.types.responses import Response, ResponseStreamEvent
    from pydantic import TypeAdapter
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    call_item = {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "get_document",
                 "arguments": '{"doc_name": "report.pdf"}',
                 "status": "completed"}
    message_item = {"type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "The answer",
                                 "annotations": [], "logprobs": []}]}

    def call(stream):
        turns = [("resp_1", call_item), ("resp_2", message_item)]

        def handler(request):
            response_id, item = turns.pop(0)
            if json.loads(request.content).get("stream"):
                return httpx.Response(
                    200, content=_responses_sse(response_id, item).encode(),
                    headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=_responses_turn(response_id, item))

        backend = {"http_client": httpx.AsyncClient(
                       transport=httpx.MockTransport(handler)),
                   "api_key": "test", "base_url": "http://responses.test/v1"}
        return client.chat("q", protocol="responses", doc_id=doc_id,
                           model="gpt-test", stream=stream, backend=backend)

    assert Response.model_validate(call(False)).output_text == "The answer"
    events = list(call(True))
    adapter = TypeAdapter(ResponseStreamEvent)
    state = ResponseStreamState(input_tools=omit, text_format=omit)
    for event in events:
        # it asserts each delta lands on the item output_index names
        state.handle_event(adapter.validate_python(event))
    assert [e["type"] for e in events[:2]] == ["response.created",
                                               "response.in_progress"]
    sequence = [e["sequence_number"] for e in events]
    assert sequence == list(range(1, len(events) + 1))
    terminal = events[-1]["response"]
    assert terminal["id"] == events[0]["response"]["id"]
    assert [item["type"] for item in terminal["output"]] == [
        "function_call", "message"]
    assert Response.model_validate(terminal).output_text == "The answer"


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


def _anthropic_sse(message):
    """Render text, thinking and tool turns for a real streaming parser."""
    events = [{"type": "message_start", "message": {
        **message, "content": [], "stop_reason": None}}]
    for index, block in enumerate(message["content"]):
        if block["type"] == "tool_use":
            initial = {**block, "input": {}}
            deltas = [{"type": "input_json_delta",
                       "partial_json": json.dumps(block["input"])}]
        elif block["type"] == "thinking":
            # the text in two deltas, then its signature in its own
            initial = {**block, "thinking": "", "signature": ""}
            half = len(block["thinking"]) // 2
            deltas = [{"type": "thinking_delta", "thinking": text}
                      for text in (block["thinking"][:half],
                                   block["thinking"][half:])]
            deltas.append({"type": "signature_delta",
                           "signature": block["signature"]})
        else:
            initial = {**block, "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        events.extend([
            {"type": "content_block_start", "index": index,
             "content_block": initial},
            *({"type": "content_block_delta", "index": index, "delta": delta}
              for delta in deltas),
            {"type": "content_block_stop", "index": index},
        ])
    events.extend([
        {"type": "message_delta", "delta": {
            "stop_reason": message["stop_reason"], "stop_sequence": None},
         "usage": {"output_tokens": message["usage"]["output_tokens"]}},
        {"type": "message_stop"},
    ])
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                   for event in events)


@pytest.fixture
def fake_anthropic(monkeypatch):
    state = {"calls": []}

    def install(responses):
        state["calls"].clear()

        def handler(request):
            state["calls"].append(json.loads(request.content))
            body = responses[len(state["calls"]) - 1]
            if isinstance(body, anthropic_httpx.Response):  # an error reply
                return body
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
                            lambda backend=None, route="anthropic": fake)
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
    # Each turn is one sampling iteration of the API's own breakdown.
    assert result["usage"]["iterations"] == 2 * [{
        "type": "message", "model": "claude-test", "input_tokens": 10,
        "output_tokens": 5, "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation": None}]
    from anthropic.types.beta import BetaMessageIterationUsage, BetaUsage
    native = BetaUsage.model_validate(result["usage"]).iterations or []
    assert all(isinstance(it, BetaMessageIterationUsage) for it in native)
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
def test_messages_usage_sums_every_count_across_turns():
    from anthropic.types.beta import BetaUsage

    def turn(usage):
        return BetaUsage.model_validate(usage).model_dump(mode="json")

    first = turn({
        "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 4,
        "cache_creation": {"ephemeral_5m_input_tokens": 3,
                           "ephemeral_1h_input_tokens": 0},
        "output_tokens_details": {"thinking_tokens": 2},
        "server_tool_use": {"web_search_requests": 1,
                            "web_fetch_requests": 0},
        "iterations": [{"type": "compaction", "input_tokens": 7,
                        "output_tokens": 1, "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0}],
        "service_tier": "priority",
        "fallback_credit": {"status": {"type": "not_applied",
                                       "reason": "expired",
                                       "remove_to_redeem": ["speed"]}},
        "undeclared_count": 7, "cost": 0.25,
        "undeclared_usage": {"requests": 1, "by_tool": {"runs": 2}}})
    last = turn({
        "input_tokens": 20, "output_tokens": 6,
        "cache_creation": {"ephemeral_5m_input_tokens": 1,
                           "ephemeral_1h_input_tokens": 2},
        "output_tokens_details": {"thinking_tokens": 3},
        "server_tool_use": {"web_search_requests": 2,
                            "web_fetch_requests": 1},
        "iterations": [{"type": "message", "model": "claude-test",
                        "input_tokens": 20, "output_tokens": 6,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0}],
        "service_tier": "standard",
        "fallback_credit": {"status": {"type": "redeemed"}}, "cost": 0.5,
        "undeclared_usage": {"requests": 1, "by_tool": {"runs": 3}}})

    usage = local_chat._anthropic_usage([(first, "claude-test"),
                                         (last, "claude-test")])
    assert usage == {
        **last,
        "input_tokens": 30, "output_tokens": 11,
        "cache_read_input_tokens": 4, "cache_creation_input_tokens": None,
        "cache_creation": {"ephemeral_5m_input_tokens": 4,
                           "ephemeral_1h_input_tokens": 2},
        "output_tokens_details": {"thinking_tokens": 5},
        "server_tool_use": {"web_search_requests": 3,
                            "web_fetch_requests": 1},
        "iterations": first["iterations"] + last["iterations"],
        "undeclared_count": 7, "cost": 0.75,
        "undeclared_usage": {"requests": 2, "by_tool": {"runs": 5}},
    }
    assert usage["service_tier"] == "standard"
    assert usage["fallback_credit"] == {"status": {"type": "redeemed"}}


@needs_anthropic
def test_messages_stream_final_message_is_the_non_stream_result(
        client, store_path, fake_anthropic):
    seed_doc(store_path, "pi-a", "r.pdf")
    replies = [_anthropic_message([
        {"type": "tool_use", "id": tool_id, "name": "get_document",
         "input": {"doc_name": "r.pdf"}}], "tool_use")
        for tool_id in ("tu_1", "tu_2")]
    replies.append(_anthropic_message([{"type": "text", "text": "Done"}],
                                      "end_turn"))
    fake_anthropic([_anthropic_sse(reply) for reply in replies])
    question = [{"role": "user", "content": "q"}]
    stream = client._messages(question, model="claude-test", max_tokens=100,
                              stream=True)
    events = list(stream)
    assert events[-1].type == "message_stop"
    assert events[-1].message.content[0].text == "Done"
    final = stream.get_final_message()
    assert stream.get_final_text() == "Done"

    # Called mid-stream, it reads the rest of the run first.
    fake_anthropic([_anthropic_sse(reply) for reply in replies])
    early = client._messages(question, model="claude-test", max_tokens=100,
                             stream=True)
    next(early)
    early_usage = early.get_final_message()["usage"]

    fake_anthropic(replies)
    result = client._messages(question, model="claude-test", max_tokens=100)
    assert result["usage"]["input_tokens"] == 30
    assert final["usage"] == early_usage == result["usage"]
    assert final.usage.input_tokens == 30
    assert final.stop_reason == result["stop_reason"] == "end_turn"
    assert final == result
    # The same continuation transcript the non-stream envelope carries.
    assert [m["role"] for m in final.messages] == [
        "assistant", "user", "assistant", "user", "assistant"]
    assert final.messages == result["messages"]

    fake_anthropic([_anthropic_sse(_anthropic_message(
        [{"type": "text", "text": "Do"}], "max_tokens"))])
    cut = client._messages(question, model="claude-test", max_tokens=100,
                           stream=True).get_final_message()
    assert cut.stop_reason == "max_tokens"
    # a bare string can't mark the cut, so get_final_text() raises instead
    fake_anthropic([_anthropic_sse(_anthropic_message(
        [{"type": "text", "text": "Do"}], "max_tokens"))])
    with pytest.raises(PageIndexAPIError, match="unfinished") as err:
        client._messages(question, model="claude-test", max_tokens=100,
                         stream=True).get_final_text()
    assert err.value.partial_answer == "Do"
    assert err.value.usage["output_tokens"] == 5
    # every text block joins (citations split an answer); a stop sequence
    # is a finished answer
    fake_anthropic([_anthropic_sse(_anthropic_message(
        [{"type": "text", "text": "The "}, {"type": "text", "text": "answer"}],
        "stop_sequence"))])
    assert client._messages(question, model="claude-test", max_tokens=100,
                            stream=True).get_final_text() == "The answer"

    fake_anthropic([_anthropic_sse(_anthropic_message([], "end_turn"))])
    with pytest.raises(PageIndexAPIError, match="no text block"):
        client._messages(question, model="claude-test", max_tokens=100,
                         stream=True).get_final_text()


@needs_anthropic
def test_messages_stream_cut_mid_message_raises(client, store_path,
                                                fake_anthropic):
    seed_doc(store_path, "pi-a", "r.pdf")
    tool_turn = _anthropic_message([
        {"type": "tool_use", "id": "tu_1", "name": "get_document",
         "input": {"doc_name": "r.pdf"}}], "tool_use")
    answer = _anthropic_message([{"type": "text", "text": "Done"}], "end_turn")
    cut = _anthropic_sse(answer).rsplit("event: message_delta", 1)[0]
    fake_anthropic([_anthropic_sse(tool_turn), cut])
    stream = client._messages("q", model="claude-test", max_tokens=100,
                              stream=True)
    with pytest.raises(PageIndexAPIError, match="ended before") as err:
        stream.get_final_message()
    usage = err.value.usage
    assert usage is not None
    assert usage["input_tokens"] == 10 and len(usage["iterations"]) == 1
    with pytest.raises(PageIndexAPIError, match="no final message"):
        stream.get_final_message()

    # Only message_stop missing: the message itself is complete.
    unstopped = _anthropic_sse(answer).rsplit("event: message_stop", 1)[0]
    fake_anthropic([_anthropic_sse(tool_turn), unstopped])
    final = client._messages("q", model="claude-test", max_tokens=100,
                             stream=True).get_final_message()
    assert final["usage"]["input_tokens"] == 20
    assert final.messages[-1]["content"][0]["text"] == "Done"


@needs_agents
@pytest.mark.parametrize("run, key", [
    (lambda c: c.chat_completions("q"), "prompt_tokens"),
    (lambda c: list(c.chat_completions("q", stream=True)), "prompt_tokens"),
    (lambda c: c.chat("q"), "prompt_tokens"),
    (lambda c: list(c.chat("q", stream=True)), "prompt_tokens"),
    (lambda c: c._responses("q"), "input_tokens"),
    (lambda c: list(c._responses("q", stream=True)), "input_tokens"),
])
def test_failed_run_carries_the_usage_already_spent(client, store_path,
                                                    fake_model, run, key):
    """A provider error on turn 2: openai-agents attaches no run data to
    it, yet turn 1 was billed and is reported."""
    import openai
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_call_item("get_document", {"doc_name": "report.pdf"})],
                openai.APIConnectionError(
                    request=httpx.Request("POST", "https://backend"))])
    with pytest.raises(PageIndexAPIError, match="model backend failed") as err:
        run(client)
    usage = err.value.usage
    assert usage is not None
    assert usage[key] == 10 and usage["total_tokens"] == 15
    assert len(usage["request_usage_entries"]) == 1


def test_run_usage_with_no_finished_call_is_zero():
    assert local_chat._openai_usage([]) == {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
        "request_usage_entries": []}
    assert local_chat._responses_usage([])["request_usage_entries"] == []


def test_run_usage_maps_each_count_to_its_field():
    """Distinct non-zero counts, so a swapped column cannot pass."""
    call = types.SimpleNamespace(usage=types.SimpleNamespace(
        input_tokens=100, output_tokens=20,
        input_tokens_details=types.SimpleNamespace(cached_tokens=7,
                                                   cache_write_tokens=11),
        output_tokens_details=types.SimpleNamespace(reasoning_tokens=3)))
    chat = local_chat._openai_usage([call, call])
    assert chat["request_usage_entries"][0] == {
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 7, "cache_write_tokens": 11},
        "completion_tokens_details": {"reasoning_tokens": 3}}
    assert chat["prompt_tokens_details"] == {"cached_tokens": 14,
                                             "cache_write_tokens": 22}
    responses = local_chat._responses_usage([call])
    assert responses["input_tokens_details"] == {"cached_tokens": 7,
                                                 "cache_write_tokens": 11}
    assert responses["output_tokens_details"] == {"reasoning_tokens": 3}


@needs_agents
def test_max_turns_error_carries_the_usage_already_spent(client, store_path,
                                                         fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([[_call_item("get_document", {"doc_name": "report.pdf"})],
                [_msg_item("too late")]])
    with pytest.raises(PageIndexAPIError, match="max_turns") as err:
        client.chat_completions("q", max_turns=1)
    usage = err.value.usage
    assert usage is not None
    assert usage["prompt_tokens"] == 10
    assert len(usage["request_usage_entries"]) == 1


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
def test_messages_failed_run_carries_the_usage_already_spent(
        client, store_path, fake_anthropic, stream):
    seed_doc(store_path, "pi-a", "report.pdf")
    tool_turn = _anthropic_message(
        [{"type": "tool_use", "id": "tu_1", "name": "get_document",
          "input": {"doc_name": "report.pdf"}}], "tool_use")
    refused = anthropic_httpx.Response(400, json={
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "no"}})
    fake_anthropic([_anthropic_sse(tool_turn) if stream else tool_turn,
                    refused])
    with pytest.raises(PageIndexAPIError, match="model backend failed") as err:
        result = client._messages("q", model="claude-test", max_tokens=100,
                                  stream=stream)
        list(result) if stream else result
    usage = err.value.usage
    assert usage is not None
    assert usage["input_tokens"] == 10 and usage["output_tokens"] == 5
    assert len(usage["iterations"]) == 1

    fake_anthropic([refused])  # nothing finished before the failure
    with pytest.raises(PageIndexAPIError) as err:
        result = client._messages("q", model="claude-test", max_tokens=100,
                                  stream=stream)
        list(result) if stream else result
    assert err.value.usage == {"input_tokens": 0, "output_tokens": 0,
                               "iterations": []}


@needs_anthropic
def test_messages_turn_without_usage_counts_nothing(client, store_path,
                                                    fake_anthropic):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_anthropic([
        _anthropic_message([{"type": "tool_use", "id": "tu_1",
                             "name": "get_document",
                             "input": {"doc_name": "report.pdf"}}],
                           "tool_use"),
        {**_anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
         "usage": None}])
    result = client._messages("q", model="claude-test", max_tokens=100)
    assert result["usage"]["input_tokens"] == 10
    assert len(result["usage"]["iterations"]) == 1


def test_validation_error_has_no_usage(client):
    with pytest.raises(PageIndexAPIError) as err:
        client.chat_completions("q", max_turns=0)
    assert err.value.usage is None


@needs_anthropic
def test_messages_doc_block_and_system(client, store_path, fake_anthropic):
    """Doc block leads as a user message; system keeps the cached header."""
    doc_id = seed_doc(store_path, "pi-a", "report.pdf")
    calls = fake_anthropic([
        _anthropic_message(
            [{"type": "tool_use", "id": "tu_1", "name": "get_document",
              "input": {"doc_name": "report.pdf"}}], "tool_use"),
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn"),
    ])
    result = client._messages([{"role": "user", "content": "hi"}],
                              model="claude-test", max_tokens=100,
                              doc_id=doc_id, system="Answer in French.")
    first, second = calls[0]["messages"][:2]
    assert first["role"] == "user"
    assert "The user has specified document: report.pdf" in first["content"]
    assert second == {"role": "user", "content": "hi"}
    system = calls[0]["system"]
    assert [block["text"] for block in system[1:]] == ["Answer in French."]
    roles = [message["role"] for message in result["messages"]]
    assert roles == ["assistant", "user", "assistant"]


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
def test_messages_reads_by_key_and_by_attribute(client, store_path,
                                                fake_anthropic):
    """The Messages envelope, each stream event and get_final_message() are
    dicts that also read by attribute, as the Anthropic SDK's objects."""
    reply = _anthropic_message([{"type": "text", "text": "Done"}], "end_turn")
    question = [{"role": "user", "content": "q"}]
    fake_anthropic([reply])
    r = client._messages(question, model="claude-test", max_tokens=100)
    assert isinstance(r, dict) and json.loads(json.dumps(r)) == r
    assert r.content[0].text == r["content"][0]["text"] == "Done"
    assert r.stop_reason == "end_turn" and r.messages == r["messages"]
    with pytest.raises(AttributeError):
        r.contentz
    fake_anthropic([_anthropic_sse(reply)])
    stream = client._messages(question, model="claude-test", max_tokens=100,
                              stream=True)
    events = list(stream)
    assert all(isinstance(event, dict) for event in events)
    assert [event.type for event in events] == [e["type"] for e in events]
    assert (events[-1].message.content[0].text
            == events[-1]["message"]["content"][0]["text"] == "Done")
    for clone in (copy.deepcopy(events[-1]),
                  pickle.loads(pickle.dumps(events[-1]))):
        assert type(clone) is type(events[-1]) and clone == events[-1]
        assert clone.message.content[0].text == "Done"
    with pytest.raises(AttributeError, match="read-only"):
        events[-1].type = "other"
    # the dict is the truth: an edit through the keys shows by attribute
    delta = next(e for e in events if e["type"] == "content_block_delta")
    delta["delta"]["text"] = "[redacted]"
    assert delta.delta.text == "[redacted]"
    final = stream.get_final_message()
    assert final.usage.output_tokens == final["usage"]["output_tokens"]
    assert final.stop_reason == final["stop_reason"] == "end_turn"
    with pytest.raises(AttributeError):
        final.usagez


@needs_anthropic
def test_messages_conforms_to_the_official_sdk(client, store_path,
                                               fake_anthropic):
    """The envelope parses as the Anthropic SDK's Message; the stream is
    its tool runner's, one message sequence per turn, every raw event
    parsing as RawMessageStreamEvent beside the SDK's convenience events."""
    from anthropic.types import Message, RawMessageStreamEvent
    from pydantic import TypeAdapter
    seed_doc(store_path, "pi-a", "r.pdf")
    turns = [_anthropic_message([
        {"type": "tool_use", "id": "tu_1", "name": "get_document",
         "input": {"doc_name": "r.pdf"}}], "tool_use"),
        _anthropic_message([{"type": "text", "text": "Done"}], "end_turn")]
    fake_anthropic(turns)
    envelope = Message.model_validate(
        client._messages("q", model="claude-test", max_tokens=100))
    assert envelope.content[0].type == "text"
    fake_anthropic([_anthropic_sse(turn) for turn in turns])
    with warnings.catch_warnings():  # the dict views dump the SDK's objects
        warnings.filterwarnings("error", message="Pydantic serializer")
        events = list(client._messages("q", model="claude-test",
                                       max_tokens=100, stream=True))
    raw_types = {"message_start", "message_delta", "message_stop",
                 "content_block_start", "content_block_delta",
                 "content_block_stop"}
    adapter = TypeAdapter(RawMessageStreamEvent)
    raw = [event for event in events if event["type"] in raw_types]
    for event in raw:
        adapter.validate_python(event)
    assert {e["type"] for e in events} - raw_types <= {"text", "input_json"}
    starts = [n for n, e in enumerate(raw) if e["type"] == "message_start"]
    stops = [n for n, e in enumerate(raw) if e["type"] == "message_stop"]
    assert len(starts) == len(stops) == len(turns)
    assert all(start < stop for start, stop in zip(starts, stops))
    assert all(stop < start for stop, start in zip(stops, starts[1:]))


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
    with pytest.raises(PageIndexAPIError, match="anthropic version") as err:
        client._messages([{"role": "user", "content": "hi"}],
                        model="claude-test", max_tokens=100)
    usage = err.value.usage  # the finished turn was still billed
    assert usage is not None and usage["input_tokens"] == 10


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


@needs_agents
def test_litellm_import_failure_surfaces_its_cause(monkeypatch):
    pytest.importorskip("agents.extensions.models.litellm_model")
    monkeypatch.setitem(sys.modules, "litellm", None)
    with pytest.raises(PageIndexAPIError, match="failed to import") as info:
        local_chat._openai_model("chat", "anthropic/claude-x")
    assert "not installed" not in str(info.value)
    assert isinstance(info.value.__cause__, ImportError)


def test_sol_class_refusal_names_its_exits():
    """The tools-with-reasoning 400 names its exits on the chat lane only."""
    refusal = Exception(
        "Error code: 400 - Function tools with reasoning_effort are not "
        "supported for gpt-5.6-sol in /v1/chat/completions.")
    chat = str(local_chat._model_backend_error(refusal, "chat"))
    assert "chat(protocol='responses')" in chat
    assert "pass reasoning_effort" in chat
    assert "upgrade" not in chat
    resp = str(local_chat._model_backend_error(refusal, "responses"))
    assert resp == f"The model backend failed: {refusal}"
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
        assert agent.model_settings.extra_args == {"max_retries": 0}


@needs_agents
def test_reasoning_passthrough_reaches_each_engine(monkeypatch):
    """Each door forwards its native reasoning; unset sends nothing."""
    pytest.importorskip("litellm")
    from openai.types.shared import Reasoning
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, reasoning_effort="low")
    assert agent.model_settings.reasoning == Reasoning(effort="low")
    assert agent.model_settings.extra_args == {"max_retries": 0}
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", None, None,
                                     reasoning_effort="none")
    assert agent.model_settings.reasoning == Reasoning(effort="none")
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", None, None,
                                     reasoning_effort="disable")
    assert agent.model_settings.reasoning is None
    assert agent.model_settings.extra_args["reasoning_effort"] == "disable"
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
    assert agent.model_settings.extra_args == {"max_retries": 0}
    # "" is unset on this lane too, like the protocol lanes
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, reasoning_effort="")
    assert agent.model_settings.extra_args == {"max_retries": 0}


@needs_agents
def test_extra_body_model_settings_fields_ride_their_field(monkeypatch):
    """extra_body settings LitellmModel reads ride their field, caller wins."""
    pytest.importorskip("litellm")
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", 0.7, None,
                                     extra_body={"temperature": 0.2,
                                                 "top_k": 5})
    settings = agent.model_settings
    assert settings.temperature == 0.2
    assert settings.extra_args["top_k"] == 5
    assert "temperature" not in settings.extra_args
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None,
                                     extra_body={"temperature": 0.2})
    assert agent.model_settings.temperature is None
    assert agent.model_settings.extra_body == {"temperature": 0.2}
    with pytest.raises(PageIndexAPIError, match="Invalid model settings"):
        local_chat._openai_agent(None, "chat", "anthropic/claude-x", "sys",
                                 None, None,
                                 extra_body={"temperature": "hot"})


@needs_agents
def test_litellm_model_settings_list_matches_litellm_model(monkeypatch):
    """_LITELLM_MODEL_SETTINGS names exactly what LitellmModel forwards."""
    pytest.importorskip("litellm")
    import litellm
    from agents import ModelSettings, ModelTracing, function_tool
    from agents.extensions.models.litellm_model import LitellmModel
    from openai.types.shared import Reasoning
    samples = {
        "temperature": 0.3, "top_p": 0.5, "frequency_penalty": 0.1,
        "presence_penalty": 0.2, "tool_choice": "required",
        "parallel_tool_calls": False, "truncation": "auto", "max_tokens": 7,
        "reasoning": Reasoning(effort="low"), "verbosity": "low",
        "metadata": {"k": "v"}, "store": True,
        "prompt_cache_retention": "24h", "include_usage": True,
        "response_include": ["reasoning.encrypted_content"],
        "top_logprobs": 2, "extra_query": {"q": "1"}, "extra_body": {"x": 1},
        "extra_headers": {"X-Probe": "1"}, "extra_args": {"y": 2},
        "retry": {"max_retries": 1},
        "context_management": [{"type": "compaction",
                                "compact_threshold": 1000}],
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "preserve_raw_usage": True}
    assert {field.name for field in dataclasses.fields(ModelSettings)} == set(
        samples), "a ModelSettings field changed: update samples"
    seen = []

    class Sent(Exception):
        pass

    async def acompletion(**kwargs):
        seen.append(kwargs)
        raise Sent

    monkeypatch.setattr(litellm, "acompletion", acompletion)

    @function_tool
    def lookup(page: int) -> str:
        return str(page)

    model = LitellmModel("anthropic/claude-test", api_key="k")

    def sent_with(settings):
        async def run():
            async for _ in model.stream_response(
                    None, "q", settings, [lookup], None, [],
                    ModelTracing.DISABLED):
                pass

        seen.clear()
        with pytest.raises(Sent):
            asyncio.run(run())
        return seen[0]

    unset = sent_with(ModelSettings())
    forwarded = {name for name, value in samples.items()
                 if sent_with(ModelSettings(**{name: value})) != unset}
    assert forwarded - {"extra_body", "extra_args"} == (
        local_chat._LITELLM_MODEL_SETTINGS)


@needs_agents
def test_extra_body_passthrough_reaches_each_engine(monkeypatch):
    """Caller extras merge last — over the cache key on OpenAI
    destinations — and ride LiteLLM's own kwargs elsewhere, where
    extra_body would plant literal fields providers reject."""
    pytest.importorskip("litellm")
    from openai.types.shared import Reasoning
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    agent = local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, cache_key="pageindex-k",
                                     extra_body={"logit_bias": {"1": 5},
                                                 "prompt_cache_key": "mine"})
    assert agent.model_settings.extra_body == {
        "prompt_cache_key": "mine", "logit_bias": {"1": 5}}
    assert agent.model_settings.extra_args == {"max_retries": 0}
    agent = local_chat._openai_agent(None, "chat", "anthropic/claude-x",
                                     "sys", None, None,
                                     reasoning_effort="low",
                                     extra_body={"top_k": 20})
    assert agent.model_settings.extra_body is None
    assert agent.model_settings.extra_args["top_k"] == 20
    assert agent.model_settings.reasoning == Reasoning(effort="low")
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
        # function_call_output.output: the framework's structured text item
        return [item["output"][0]["text"] for item in items
                if item.get("type") == "function_call_output"]

    assert "NOT_FOUND" in tool_outputs(fake.inputs[1])[-1]
    browse = json.loads(tool_outputs(fake.inputs[2])[-1])
    assert [doc["name"] for doc in browse["documents"]] == ["report.pdf"]


@needs_agents
def test_malformed_tool_arguments_answer_the_model(client, store_path,
                                                    fake_model):
    """A truncated argument string reaches the tool (strict schemas are
    off). The tools are the framework's own MCP conversion, so its failure
    pipeline hands the model an error message and the run goes on: no
    aborted run, no SDK envelope."""
    from openai.types.responses import ResponseFunctionToolCall
    seed_doc(store_path, "pi-a", "report.pdf")
    bad_call = ResponseFunctionToolCall(
        id="fc_1", type="function_call", call_id="call_1",
        name="get_document", arguments="{not json", status="completed")
    fake = fake_model([[bad_call], [_msg_item("The answer")]])
    result = client.chat_completions("What?")
    assert result["choices"][0]["message"]["content"] == "The answer"
    outputs = [item["output"] for item in fake.inputs[1]
               if item.get("type") == "function_call_output"]
    # the wording is the framework's; 0.24 hides the parse detail
    assert outputs and outputs[-1]


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
        client.document_context([])


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
    # litellm/ is routing grammar, not a provider: it strips here too.
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
def test_responses_sends_model_names_as_written():
    """The responses lane is the OpenAI SDK, never LiteLLM: litellm/ and
    openai/ strip, any other name reaches the backend verbatim and the
    destination judges it — a gateway (OpenRouter) serves slashed ids."""
    from agents.models.openai_responses import OpenAIResponsesModel

    for name, wire in (("deepseek/deepseek-chat", "deepseek/deepseek-chat"),
                       ("anthropic/claude-x", "anthropic/claude-x"),
                       ("litellm/anthropic/claude-x", "anthropic/claude-x"),
                       ("openai/deepseek/deepseek-chat",
                        "deepseek/deepseek-chat")):
        model = local_chat._openai_model("responses", name)
        assert isinstance(model, OpenAIResponsesModel), name
        assert str(model.model) == wire, name


@needs_agents
def test_responses_lane_never_takes_litellm_settings(monkeypatch):
    """Whatever the name looks like, the responses lane's request is an
    OpenAI one: the cache key and caller extras ride extra_body, and
    LiteLLM's Claude cache marker stays out — responses.create() raises
    TypeError on that unknown kwarg."""
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: None)
    monkeypatch.setattr("pageindex.integrations.openai_agents.build_openai_tools",
                        lambda *a, **k: [])
    for name in ("anthropic/claude-x", "bedrock/us.anthropic.claude-x"):
        agent = local_chat._openai_agent(
            None, "responses", name, "sys", None, None,
            cache_key="pageindex-k", extra_body={"x_custom": 1})
        assert agent.model_settings.extra_args is None, name
        assert agent.model_settings.extra_body == {
            "prompt_cache_key": "pageindex-k", "x_custom": 1}, name


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

    reply = types.SimpleNamespace(
        status="incomplete",
        incomplete_details=_Dumpable({"reason": "max_output_tokens"}),
        error=None)

    async def create(*args, **kwargs):
        return reply

    agent = types.SimpleNamespace(model=types.SimpleNamespace(
        _client=types.SimpleNamespace(
            responses=types.SimpleNamespace(create=create))))
    recorded = {}
    local_chat._record_response_status(agent, recorded)
    from agents.exceptions import ModelBehaviorError
    with pytest.raises(ModelBehaviorError):  # a cut response ends the run
        asyncio.run(agent.model._client.responses.create())
    assert recorded == {"response": reply, "status": "incomplete",
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
        return _cut_response()

    fake._client = types.SimpleNamespace(
        responses=types.SimpleNamespace(create=create))
    result = client._responses("q")
    assert result["status"] == "incomplete"
    assert result["incomplete_details"] == {"reason": "max_output_tokens"}
    assert result["error"] is None


@needs_agents
def test_responses_cut_short_still_returns_the_envelope(store_path):
    """openai-agents 0.22+ raises on a response cut short, before its hooks
    or items see it; the run still ends in the envelope, as the stream
    does, with the partial text and both billed calls. FakeModel skips the
    transport fetch that raises, so the real model runs on a mock one."""
    seed_doc(store_path, "pi-a", "report.pdf")
    details = {"input_tokens_details": {"cached_tokens": 0},
               "output_tokens_details": {"reasoning_tokens": 0}}
    replies = [
        {"id": "resp_1", "status": "completed",
         "output": [_call_item("get_document",
                               {"doc_name": "report.pdf"}).model_dump()],
         "usage": {**details, "input_tokens": 10, "output_tokens": 5,
                   "total_tokens": 15}},
        {"id": "resp_2", "status": "incomplete",
         "incomplete_details": {"reason": "max_output_tokens"},
         "output": [_msg_item("The range was").model_dump()],
         "usage": {**details, "input_tokens": 40, "output_tokens": 99,
                   "total_tokens": 139}}]
    common = {"object": "response", "created_at": 0, "model": "gpt-test",
              "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={**common,
                                                  **replies.pop(0)})))
    client = PageIndexLocalClient(storage_path=store_path,
                                  chat_model="openai/gpt-test")
    envelope = client.chat("q", protocol="responses", backend={
        "api_key": "sk-test", "base_url": "http://mock.invalid/v1",
        "http_client": http})
    asyncio.run(http.aclose())
    assert envelope["status"] == "incomplete"
    assert [item["type"] for item in envelope["messages"]] == [
        "function_call", "function_call_output", "message"]
    assert envelope["output"][-1]["content"][0]["text"] == "The range was"
    usage = envelope["usage"]
    assert (usage["input_tokens"], usage["output_tokens"]) == (50, 104)
    assert len(usage["request_usage_entries"]) == 2


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
                        lambda backend=None, route="anthropic": fake)
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
def test_messages_max_tokens_default(client, fake_anthropic):
    """The wire-required budget defaults to 8192 whatever the model names;
    an explicit value passes through."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-3-opus-20240229")
    assert calls[0]["max_tokens"] == 8192
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-sonnet-4-5")
    assert calls[0]["max_tokens"] == 8192
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="claude-3-opus-20240229", max_tokens=1234)
    assert calls[0]["max_tokens"] == 1234


@needs_anthropic
def test_messages_accepts_the_litellm_spelling(client, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client._messages("q", model="anthropic/claude-3-opus-20240229")
    assert calls[0]["model"] == "claude-3-opus-20240229"


@needs_anthropic
def test_messages_carries_a_claude_chat_model(store_path, fake_anthropic):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    local = PageIndexLocalClient(storage_path=store_path,
                                 chat_model="anthropic/claude-3-opus-20240229")
    # Through the public door: chat() must not demand model= itself.
    local.chat("q", protocol="messages")
    assert calls[0]["model"] == "claude-3-opus-20240229"
    # The stock default was never chosen: nothing to send.
    with pytest.raises(PageIndexAPIError, match="needs a model"):
        PageIndexLocalClient(storage_path=store_path).chat(
            "q", protocol="messages")


@needs_anthropic
def test_messages_cleared_chat_model_gets_the_own_model_refusal(
        store_path, fake_anthropic):
    """'' and None both mean "configures nothing": clearing chat_model
    drops the client back to no-own-chat, and messages() refuses in its
    own voice — never model='' on the wire, never a NoneType crash."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "never"}], "end_turn")])
    local = PageIndexLocalClient(storage_path=store_path,
                                 chat_model="claude-sonnet-4-5")
    for cleared in ("", None):
        local.chat_model = cleared
        with pytest.raises(PageIndexAPIError, match="chat_model="):
            local._messages("q")
    assert calls == []


@needs_anthropic
def test_messages_route_prefix_needs_a_model_id(store_path, fake_anthropic):
    """A prefix-only name selects a channel and names nothing — sending
    model='' (or switching transports with no model chosen) is the worst
    of both; refuse it in this SDK's own voice."""
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "never"}], "end_turn")])
    local = PageIndexLocalClient(storage_path=store_path,
                                 chat_model="claude-sonnet-4-5")
    for name in ("bedrock/", "vertex_ai/", "azure_ai/", "anthropic/",
                 "litellm/"):
        with pytest.raises(PageIndexAPIError, match="no model id"):
            local._messages("q", model=name)
    assert calls == []


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


_PAUSED = ([{"type": "text", "text": "Paused part"}], "pause_turn")
_DONE = ([{"type": "text", "text": "Done"}], "end_turn")


@needs_anthropic
@pytest.mark.parametrize("turns,kept,texts", [
    pytest.param([_DONE], 1, ["Done"], id="2.2-final"),
    pytest.param([([{"type": "text", "text": "Partial"},
                    _anthropic_tool_use()], "max_tokens")], 1, ["Partial"],
                 id="2.2-cut-tool-use"),
    pytest.param([_PAUSED, _DONE], 2, ["Paused part", "Done"], id="2.2-paused"),
    pytest.param([_PAUSED, _DONE], 1, ["Paused part", "Done"],
                 id="2.1-paused"),
])
def test_messages_transcript_carries_each_turn_once(client, store_path,
                                                    monkeypatch, turns, kept,
                                                    texts):
    """Each turn appears once in the transcript; the loop runs unthreaded."""
    from anthropic.types.beta import BetaMessage
    seed_doc(store_path, "pi-a", "report.pdf")
    replies = [BetaMessage.model_validate(_anthropic_message(*turn))
               for turn in turns]
    seen = {}

    class _Runner:
        def __init__(self, messages):
            self.history = [*messages, *({"role": "assistant",
                                          "content": reply.content}
                                         for reply in replies[:kept])]

        def __iter__(self):
            return iter(replies)

        def generate_tool_call_response(self):
            return None

        def set_messages_params(self, callback):
            callback({"messages": self.history})

    class _Fake:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(*, messages, thread=None, **kwargs):
                    seen["thread"] = thread
                    return _Runner(messages)

        def close(self):
            pass

    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None, route="anthropic": _Fake())
    result = client._messages("q", model="claude-test", max_tokens=100)
    assert seen["thread"] is False
    assert [[(block["type"], block.get("text")) for block in message["content"]]
            for message in result["messages"]] == [[("text", text)]
                                                    for text in texts]
    assert {message["role"] for message in result["messages"]} == {"assistant"}


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


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
def test_messages_bedrock_moves_an_explicit_breakpoint(client, store_path,
                                                       fake_anthropic, stream):
    """Bedrock's InvokeModel integration rejects the top-level field (Opus
    4.6 and earlier): each turn's newest tool result carries the one moving
    breakpoint instead, and none leaks into the returned history."""
    seed_doc(store_path, "pi-a", "report.pdf")
    replies = [_anthropic_message([_anthropic_tool_use(tool_id)], "tool_use")
               for tool_id in ("tu_1", "tu_2")]
    replies.append(_anthropic_message([{"type": "text", "text": "Done"}],
                                      "end_turn"))
    calls = fake_anthropic([_anthropic_sse(reply) for reply in replies]
                          if stream else replies)
    result = client._messages("q", model="bedrock/anthropic.claude-sonnet-4-6",
                              max_tokens=50, stream=stream)
    if stream:
        list(result)

    def marked(call):
        return [block.get("tool_use_id")
                for message in call["messages"]
                if isinstance(message["content"], list)
                for block in message["content"] if block.get("cache_control")]

    assert all("cache_control" not in call for call in calls)
    assert [marked(call) for call in calls] == [[], ["tu_1"], ["tu_2"]]
    if not stream:
        assert "cache_control" not in json.dumps(result["messages"])


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
                                         "base_url": "http://x"}, "anthropic")
    assert real.api_key == "kk"
    assert str(real.base_url).rstrip("/") == "http://x"
    real = local_chat._anthropic_client({"api_key": "kk",
                                         "api_base": "http://y"}, "anthropic")
    assert str(real.base_url).rstrip("/") == "http://y"

    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    fixture_client = local_chat._anthropic_client
    seen = {}
    monkeypatch.setattr(
        local_chat, "_anthropic_client",
        lambda backend=None, route="anthropic": (
            seen.setdefault("backend", backend), fixture_client())[1])
    client.chat_backend = {"base_url": "http://cb"}
    client._messages("q", model="claude-sonnet-4-5", backend={"api_key": "z"})
    assert seen["backend"] == {"base_url": "http://cb", "api_key": "z"}


@needs_anthropic
def test_messages_bad_backend_wraps_like_the_other_doors():
    with pytest.raises(PageIndexAPIError,
                       match="Anthropic backend is not configured"):
        local_chat._anthropic_client({"no_such_param": 1}, "anthropic")


@needs_agents
def test_extra_headers_ride_model_settings(monkeypatch):
    """Both openai-agents doors send ModelSettings.extra_headers."""
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
                        lambda backend=None, route="anthropic": fake)
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
    a = local_chat._anthropic_client({"api_key": "k"}, "anthropic")
    assert local_chat._anthropic_client({"api_key": "k"}, "anthropic") is a
    assert local_chat._anthropic_client({"api_key": "k2"}, "anthropic") is not a


@needs_anthropic
def test_anthropic_client_cache_keys_on_the_route(monkeypatch):
    """Same backend, another route: a cached direct client must never
    carry a Bedrock id to api.anthropic.com."""
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    direct = local_chat._anthropic_client({"api_key": "k"}, "anthropic")
    bedrock = local_chat._anthropic_client({"api_key": "k"}, "bedrock")
    assert isinstance(bedrock, anthropic.AnthropicBedrock)
    assert not isinstance(direct, anthropic.AnthropicBedrock)


@needs_anthropic
def test_anthropic_client_construction_race_keeps_first(monkeypatch):
    """A constructor losing the store race must adopt the winner rather
    than evict a client other threads may already hold."""
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    winner = object()
    real = anthropic.Anthropic

    def racing(**kwargs):
        local_chat._ANTHROPIC_CLIENTS[
            ("anthropic", (("api_key", "k"),))] = winner
        return real(**kwargs)
    monkeypatch.setattr(anthropic, "Anthropic", racing)
    assert local_chat._anthropic_client({"api_key": "k"}, "anthropic") is winner


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
                        {("anthropic", (("api_key", "test"),)): cached})

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


class _CutModel(FakeModel):
    """Scripted turns whose native finish_reason is `finish`, as the real
    model's fetch shows it to the finish recorder."""

    def __init__(self, turns, finish="length"):
        super().__init__(turns)
        self.finish = finish

    async def _fetch_response(self, *args, **kwargs):
        return types.SimpleNamespace(choices=[
            types.SimpleNamespace(finish_reason=self.finish)])

    async def get_response(self, *args, **kwargs):
        await self._fetch_response()
        return await super().get_response(*args, **kwargs)

    async def stream_response(self, *args, **kwargs):
        await self._fetch_response()
        async for event in super().stream_response(*args, **kwargs):
            yield event


@needs_agents
def test_chat_completions_reports_native_finish_reason(client, store_path,
                                                       monkeypatch):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = _CutModel([[_msg_item("cut ")], [_msg_item("cut ")]])
    monkeypatch.setattr(local_chat, "_openai_model", lambda *a: fake)
    result = client.chat_completions("q")
    assert result["choices"][0]["finish_reason"] == "length"
    chunks = list(client.chat_completions("q", stream=True,
                                          stream_metadata=True))
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"


@needs_agents
@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_chat_raises_on_an_unfinished_answer(client, store_path, monkeypatch,
                                             finish):
    """A bare answer string cannot say it is unfinished, so chat() and
    chat_completions' text stream raise: a stream after the text it
    delivered, a non-streamed call with that text on the error; both with
    the run's usage."""
    seed_doc(store_path, "pi-a", "report.pdf")

    def cut_model(*args):
        model = _CutModel([[_msg_item("The range was")]], finish)
        model.pieces = ("The range ", "was")  # the streamed deltas
        return model

    monkeypatch.setattr(local_chat, "_openai_model", cut_model)
    with pytest.raises(PageIndexAPIError, match=finish) as info:
        client.chat("q")
    assert info.value.partial_answer == "The range was"
    assert info.value.usage["completion_tokens"] == 5
    text = []
    with pytest.raises(PageIndexAPIError, match=finish) as info:
        for piece in client.chat("q", stream=True, show_process=False):
            text.append(piece)
    assert "".join(text) == "The range was"
    assert info.value.usage["completion_tokens"] == 5
    with pytest.raises(PageIndexAPIError, match=finish):
        list(client.chat("q", stream=True).events)
    text = []
    with pytest.raises(PageIndexAPIError, match=finish) as info:
        for piece in client.chat_completions("q", stream=True):
            text.append(piece)
    assert "".join(text) == "The range was"
    assert info.value.usage["completion_tokens"] == 5


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
def test_litellm_lane_gates_litellm_logging(monkeypatch, caplog):
    """The lane defaults litellm's logger to ERROR; a level the host set stays."""
    pytest.importorskip("litellm")
    import logging
    monkeypatch.delenv("LITELLM_LOG", raising=False)
    caplog.set_level(logging.NOTSET, logger="LiteLLM")  # untouched
    gated = logging.getLogger("LiteLLM")
    local_chat._openai_model("chat", "openrouter/z-ai/not-in-the-map")
    assert gated.level == logging.ERROR
    assert not gated.isEnabledFor(logging.WARNING)
    gated.setLevel(logging.WARNING)
    local_chat._openai_model("chat", "openrouter/z-ai/not-in-the-map")
    assert gated.level == logging.WARNING


def test_provider_lookups_flip_the_switch_before_asking(monkeypatch, capsys):
    """The lane asks litellm which provider serves a model before the
    model is built, and openai_agent_config asks with no model built at
    all; a failed ask print()s the banner, so the switch flips at the ask."""
    litellm = pytest.importorskip("litellm")
    for lookup in (local_chat._openai_protocol,
                   local_chat._litellm_claude_marks):
        monkeypatch.setattr(litellm, "suppress_debug_info", False)
        assert not lookup("z-ai/not-in-the-map")
        assert litellm.suppress_debug_info is True
    assert "Provider List" not in capsys.readouterr().out


def test_quiet_litellm_honors_the_chosen_default(monkeypatch, caplog):
    """LITELLM_LOG picks the default, in both directions."""
    pytest.importorskip("litellm")
    import logging
    from pageindex.utils import _quiet_litellm
    gated = logging.getLogger("LiteLLM")
    for chosen in (logging.CRITICAL, logging.DEBUG):
        monkeypatch.setenv("LITELLM_LOG", logging.getLevelName(chosen))
        caplog.set_level(logging.NOTSET, logger="LiteLLM")
        _quiet_litellm()
        assert gated.level == chosen


def test_openai_protocol_predicate_follows_litellm_routing():
    pytest.importorskip("litellm")
    for name in ("gpt-5", "openai/gpt-4o", "litellm/gpt-4o",
                 "azure/gpt-4o", "openrouter/openai/gpt-4o",
                 "deepseek/deepseek-chat", "groq/llama-3.3-70b-versatile",
                 "xai/grok-3", "azure_ai/mistral-large"):
        assert local_chat._openai_protocol(name), name
    for name in ("anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-pro",
                 "bedrock/us.anthropic.claude-sonnet-5",
                 "vertex_ai/claude-sonnet-4-5", "azure_ai/claude-sonnet-5-5"):
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
def test_messages_without_credentials_raises_the_sdks_own_error(client,
                                                                monkeypatch,
                                                                tmp_path):
    """No pre-check and no translation: the SDK's own request-time
    credential-resolution failure propagates as itself (its text already
    names api_key / auth_token) — for a bare call, a credential-less
    backend dict, and the unset-env-var shape ({"api_key": None}) alike."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ant-auth profile fallback
    for backend in (None, {"timeout": 30}, {"api_key": None}):
        with pytest.raises(TypeError, match="authentication"):
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


def test_default_max_tokens_is_pure_arithmetic(capsys):
    """The thinking default is budget + 8192, full stop: the model's real
    ceiling is the API's to enforce (its 400 names both numbers), so no
    third-party lookup runs — no I/O, no hang, nothing printed, for any
    spelling. bool is not a budget."""
    lift = local_chat._default_max_tokens
    assert lift({"type": "enabled", "budget_tokens": 30000}) == 38192
    assert lift({"type": "enabled", "budget_tokens": 10000}) == 18192
    assert lift({"type": "enabled", "budget_tokens": True}) == 8192
    assert lift() == 8192
    assert capsys.readouterr().out == ""


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
        return [{"type": "text", "text": json.dumps(
            {"status": "success", "data": {"doc": "cloud-doc"}})}], False


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
    assert "PERSISTENCE" not in fake.instructions[0]
    # The tool result made it back into turn 2.
    assert "cloud-doc" in json.dumps(fake.inputs[1])


@needs_agents
def test_process_display_elides_image_payloads(bridge_client, fake_model):
    """A [tool_result] line shows an image item as it is, minus the base64
    payload; the model still receives the image itself."""
    client, bridge = bridge_client
    bridge.call_tool = lambda name, arguments: (
        [{"type": "text", "text": "page 1"},
         {"type": "image", "mimeType": "image/png", "data": "A" * 8192}],
        False)
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "r.pdf"})],
        [_msg_item("The answer")],
    ])
    text = "".join(client.chat("What?", stream=True))
    assert ('[tool_result] get_document: page 1 {"type": "image", '
            '"image_url": "data:image/png;base64,..."}') in text
    assert "AAAA" not in text
    assert "AAAA" in json.dumps(fake.inputs[1])  # the model got the image


@needs_agents
def test_bridge_doc_id_targets_at_prompt_level(bridge_client, fake_model,
                                               monkeypatch):
    """On cloud tools there is no local allowlist: doc_id becomes the
    prompt-level targeting block only."""
    client, _ = bridge_client
    monkeypatch.setattr(client, "get_document",
                        lambda doc_id: {"name": "r.pdf", "description": "d",
                                        "status": "completed",
                                        "metadata": None})
    fake = fake_model([[_msg_item("ok")]])
    client.chat_completions("q", doc_id="pi-a")
    first = fake.inputs[0][0]
    assert "specified document" in first["content"]
    assert "r.pdf" in first["content"]


def test_targeting_block_orders_folder_before_documents(bridge_client,
                                                        monkeypatch):
    """The folder block leads the document block, joined as the managed
    chat joins them; "root" and no folder place nothing of their own."""
    from pageindex.agent_tools import targeting_block
    client, _ = bridge_client
    monkeypatch.setattr(client, "list_folders", lambda: {"folders": [
        {"id": "f-1", "name": "Team", "description": None}]})
    monkeypatch.setattr(client, "get_document",
                        lambda doc_id: {"name": "r.pdf", "status": "completed"})
    both = targeting_block(client, "pi-a", "f-1")
    assert both is not None
    folder, doc = both.split("\n\n")
    assert folder.startswith("The user has specified folder: Team\n")
    assert 'Folder metadata: {"id": "f-1", "name": "Team"}\n' in folder
    assert doc.startswith("The user has specified document: r.pdf\n")
    assert targeting_block(client, "pi-a", "root") == doc
    assert targeting_block(client, None, "f-1") == folder
    assert targeting_block(client, None, None) is None


@needs_agents
def test_bridge_folder_id_targets_ahead_of_documents(bridge_client, fake_model,
                                                     monkeypatch):
    """folder_id is prompt-level targeting on cloud tools, one leading
    user message with the folder block ahead of the document block."""
    client, _ = bridge_client
    monkeypatch.setattr(client, "list_folders", lambda: {"folders": [
        {"id": "f-1", "name": "Team", "description": "shared"}]})
    monkeypatch.setattr(client, "get_document",
                        lambda doc_id: {"name": "r.pdf", "status": "completed"})
    fake = fake_model([[_msg_item("ok")]])
    with pytest.raises(PageIndexAPIError, match="not found"):
        client.chat_completions("q", folder_id="f-9")
    client.chat_completions("q", doc_id="pi-a", folder_id="f-1")
    first, question = fake.inputs[0][:2]
    assert first["content"].startswith("The user has specified folder: Team\n")
    assert "The user has specified document: r.pdf" in first["content"]
    assert question == {"role": "user", "content": "q"}


@needs_agents
def test_bridge_folder_id_reaches_the_protocol_lanes(bridge_client, fake_model,
                                                     monkeypatch):
    """chat(protocol="responses") threads folder_id to its engine."""
    client, _ = bridge_client
    monkeypatch.setattr(client, "list_folders", lambda: {"folders": [
        {"id": "f-1", "name": "Team"}]})
    fake = fake_model([[_msg_item("ok")]])
    client.chat("q", protocol="responses", folder_id="f-1")
    assert fake.inputs[0][0]["content"].startswith(
        "The user has specified folder: Team\n")
    assert fake.inputs[0][1] == {"role": "user", "content": "q"}


@needs_anthropic
def test_bridge_folder_id_reaches_the_messages_lane(bridge_client,
                                                    fake_anthropic,
                                                    monkeypatch):
    """chat(protocol="messages") threads folder_id to its engine."""
    client, _ = bridge_client
    monkeypatch.setattr(client, "list_folders", lambda: {"folders": [
        {"id": "f-1", "name": "Team"}]})
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    client.chat("q", protocol="messages", model="claude-test",
                folder_id="f-1")
    first, second = calls[0]["messages"][:2]
    assert first["content"].startswith("The user has specified folder: Team\n")
    assert second == {"role": "user", "content": "q"}


@needs_agents
def test_folder_id_is_cloud_only(client):
    """A local library has no folders: folder_id refuses before any model
    call, like the folder methods."""
    with pytest.raises(PageIndexAPIError, match="cloud-only"):
        client.chat_completions("q", folder_id="f-1")


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

    def fresh_fake(backend=None, route="anthropic"):
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
    """build_anthropic_tools is network I/O on a bridge client, and it
    runs before the transport exists — a failure there must construct no
    transport to strand."""
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
                        lambda backend=None, route="anthropic":
                        made.append(FakeAnthropic()) or made[-1])

    def boom(client, doc_ids=None, **kwargs):
        raise PageIndexAPIError("Could not reach the PageIndex MCP server")

    monkeypatch.setattr(
        "pageindex.integrations.anthropic_sdk.build_anthropic_tools", boom)
    with pytest.raises(PageIndexAPIError, match="MCP server"):
        client._messages("q", model="claude-test", max_tokens=100)
    # Tools are deliberately built BEFORE the transport, so a tool-build
    # failure must find no transport constructed at all — if this list is
    # ever non-empty, that ordering (and its no-leak guarantee) broke.
    assert not made


@needs_anthropic
def test_messages_no_backend_leak_when_runner_build_fails(client, monkeypatch):
    """A failure between transport construction and the runner handoff
    (the runner rejecting a passthrough kwarg, say) must not strand the
    per-call transport either."""
    made = []

    class FakeAnthropic:
        def __init__(self):
            self.closed = False

            def explode(**kw):
                raise TypeError("unexpected keyword argument 'thinking'")
            self.beta = types.SimpleNamespace(
                messages=types.SimpleNamespace(tool_runner=explode))

        def close(self):
            self.closed = True

    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None, route="anthropic":
                        made.append(FakeAnthropic()) or made[-1])
    with pytest.raises(TypeError, match="thinking"):
        client._messages("q", model="claude-test", max_tokens=100)
    assert made and all(fake.closed for fake in made)


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
    assert "PERSISTENCE" not in system_text


@needs_agents
def test_bridge_openai_agent_config_carries_configured_model(bridge_client):
    """A bridge client's openai_agent_config carries its chat_model —
    same semantics as local; a plain cloud client still omits model."""
    client, _ = bridge_client
    config = client.openai_agent_config()
    assert config["model"] == "fake-model"
    assert "CLOUD LIVE INSTRUCTIONS" in config["instructions"]


@needs_anthropic
def test_messages_routes_by_model_prefix(client, fake_anthropic, monkeypatch):
    calls = fake_anthropic([
        _anthropic_message([{"type": "text", "text": "ok"}], "end_turn")])
    fixture_client = local_chat._anthropic_client
    seen = {}
    monkeypatch.setattr(
        local_chat, "_anthropic_client",
        lambda backend=None, route="anthropic": (
            seen.setdefault("route", route), fixture_client())[1])
    client._messages(
        "q", model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert seen["route"] == "bedrock"
    assert calls[0]["model"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"


@needs_anthropic
def test_anthropic_client_route_picks_the_transport_class(monkeypatch):
    """The model's routing prefix selects the SDK client class — the
    bedrock/vertex ids only mean something to their own transports."""
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    # AnthropicBedrock defaults api_key from this env var, and refuses
    # api_key alongside AWS credential kwargs — a developer machine that
    # exports it must not fail this construction.
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    bedrock = local_chat._anthropic_client(
        {"aws_region": "us-east-1", "aws_access_key": "a",
         "aws_secret_key": "s"}, "bedrock")
    assert isinstance(bedrock, anthropic.AnthropicBedrock)
    direct = local_chat._anthropic_client({"api_key": "k"}, "anthropic")
    assert isinstance(direct, anthropic.Anthropic)
    assert not isinstance(direct, anthropic.AnthropicBedrock)


@needs_anthropic
def test_anthropic_client_vertex_route(monkeypatch):
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    vertex = local_chat._anthropic_client(
        {"region": "us-east5", "project_id": "p", "access_token": "t"},
        "vertex_ai")
    assert isinstance(vertex, anthropic.AnthropicVertex)


@needs_anthropic
def test_anthropic_client_foundry_route(monkeypatch):
    if not hasattr(anthropic, "AnthropicFoundry"):
        pytest.skip("this anthropic build has no Foundry client")
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    # AnthropicFoundry fills base_url from this and refuses it beside resource=.
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    foundry = local_chat._anthropic_client(
        {"resource": "r", "api_key": "k"}, "azure_ai")
    assert isinstance(foundry, anthropic.AnthropicFoundry)


@needs_anthropic
def test_anthropic_client_unconfigured_route_keeps_the_error_contract(
        monkeypatch):
    """Vertex and Foundry fail at construction (region, credentials) —
    those failures must wrap the constructors' documented refusals."""
    # skip decided first: a skip after assertions would discard the
    # vertex coverage those assertions already ran.
    if not hasattr(anthropic, "AnthropicFoundry"):
        pytest.skip("this anthropic build has no Foundry client")
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS", {})
    for name in ("CLOUD_ML_REGION", "GOOGLE_CLOUD_PROJECT",
                 "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_BASE_URL",
                 "ANTHROPIC_FOUNDRY_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PageIndexAPIError,
                       match="Anthropic backend is not configured"):
        local_chat._anthropic_client(None, "vertex_ai")
    with pytest.raises(PageIndexAPIError,
                       match="Anthropic backend is not configured"):
        local_chat._anthropic_client(None, "azure_ai")


@needs_anthropic
@pytest.mark.parametrize("model", [
    "claude-sonnet-4-6", "bedrock/anthropic.claude-sonnet-4-6-v1:0"])
def test_messages_refuses_anthropic_below_its_floor(client, monkeypatch,
                                                     model):
    import anthropic

    def transport(*args):
        raise AssertionError("the gate let the call through")

    monkeypatch.setattr(local_chat, "_anthropic_client", transport)
    monkeypatch.setattr(anthropic, "__version__", "0.107.1")
    with pytest.raises(PageIndexAPIError,
                       match=r"anthropic 0\.122\.0 or newer, found 0\.107\.1"):
        client._messages("q", model=model)


def _failing_runner_client(monkeypatch, exc):
    # A transport whose runner dies on first turn — the shape of a
    # request-time credential failure (auth resolves per request).
    class _FailingRunner:
        def __iter__(self):
            return self

        def __next__(self):
            raise exc

        def generate_tool_call_response(self):
            return None

    class _Fake:
        class beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs):
                    return _FailingRunner()

        def close(self):
            pass

    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None, route="anthropic": _Fake())


@needs_anthropic
def test_messages_request_failures_propagate_raw(client, monkeypatch):
    """No second-guessing the transports: a request-time failure that is
    not the anthropic SDK's own error type propagates as itself — the
    stacks' original text and traceback are the diagnostic, on every
    route and on both paths. (Construction-time misconfiguration still
    wraps, in _anthropic_client — that is the constructors' documented
    contract.)"""
    cases = (RuntimeError("could not resolve credentials from session"),
             TypeError("Could not resolve authentication method"),
             ModuleNotFoundError("No module named 'botocore'",
                                 name="botocore"))
    for exc in cases:
        _failing_runner_client(monkeypatch, exc)
        with pytest.raises(type(exc)):
            client._messages(
                "q", model="bedrock/anthropic.claude-sonnet-4-6-v1:0")
        _failing_runner_client(monkeypatch, exc)
        with pytest.raises(type(exc)):
            list(client._messages("q", stream=True,
                                  model="claude-sonnet-4-5"))


def test_route_tables_and_marks_agree():
    """One route concept, three tables — the client's route/env map, this
    module's client classes, and the marks predicate. A new route must
    land in every one; a miss is a silent env no-op or full-price turns."""
    from pageindex.client import _CLAUDE_ROUTES
    assert set(local_chat._ROUTE_CLIENTS) == {"anthropic", *_CLAUDE_ROUTES}
    for wire in ("anthropic/claude-opus-4-6",
                 *(f"{route}/claude-opus-4-6" for route in _CLAUDE_ROUTES)):
        assert local_chat._litellm_claude_marks(wire), wire
    # Claude-gated on the cloud routes: other models get no marks.
    assert local_chat._litellm_claude_marks(
        "azure_ai/Meta-Llama-3-70B-Instruct") is None


# ── chat(protocol=): the protocol doors behind the front door ──

def test_old_door_names_point_at_chat_protocol(client):
    for name in ("responses", "messages"):
        with pytest.raises(AttributeError, match=f"chat\\(protocol={name!r}"):
            getattr(client, name)
        assert not hasattr(client, name)
    with pytest.raises(AttributeError, match="no attribute 'no_such_thing'"):
        client.no_such_thing


def test_chat_protocol_chat_completions_is_the_door(client, monkeypatch):
    seen = []
    monkeypatch.setattr(local_chat, "run_chat_completions",
                        lambda c, messages, **kw: seen.append((messages, kw))
                        or "door")
    knobs = dict(doc_id="pi-a", model="gpt-x", max_turns=3,
                 reasoning_effort="low", backend={"api_key": "k"},
                 extra_headers={"x": "1"}, extra_body={"seed": 1})
    for streaming in (False, True):
        assert client.chat("q", protocol="chat_completions", stream=streaming,
                           **knobs) == "door"
        # the protocol's own stream is its chunk dicts, never text pieces
        assert client.chat_completions("q", stream=streaming,
                                       stream_metadata=True,
                                       **knobs) == "door"
        assert seen[-2] == seen[-1]
    assert seen[0][0] == [{"role": "user", "content": "q"}]
    client.chat("q", protocol="chat_completions", instructions="be brief")
    assert seen[-1][0] == [{"role": "system", "content": "be brief"},
                           {"role": "user", "content": "q"}]
    with pytest.raises(PageIndexAPIError, match="show_process"):
        client.chat("q", protocol="chat_completions", stream=True,
                    show_process=True)
    assert client.chat_completions("q") == "door"


def test_chat_protocol_chat_completions_serves_managed_cloud(monkeypatch):
    """Unlike the own-model protocols, the managed cloud chat speaks
    chat.completions itself, so the lane opens without a chat model;
    the own-model knobs still refuse there."""
    from pageindex import PageIndexClient
    cloud = PageIndexClient(api_key="pi-k")
    seen = []
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kw: seen.append(kw) or {"choices": []})
    assert cloud.chat("q", protocol="chat_completions") == {"choices": []}
    assert seen[-1] == {"messages": [{"role": "user", "content": "q"}],
                        "stream": False, "doc_id": None, "temperature": None,
                        "stream_metadata": True, "enable_citations": False,
                        "extra_body": None, "folder_id": None}
    cloud.chat("q", protocol="chat_completions",
               extra_body={"temperature": 0.2, "enable_citations": True})
    assert seen[-1]["extra_body"] == {"temperature": 0.2,
                                      "enable_citations": True}
    with pytest.raises(PageIndexAPIError, match="extra_body cannot carry"):
        cloud.chat("q", protocol="chat_completions",
                   extra_body={"messages": []})
    with pytest.raises(PageIndexAPIError, match="chat_model="):
        cloud.chat("q", protocol="chat_completions", model="m")


def test_chat_takes_only_messages_by_position():
    """chat_completions() puts stream before doc_id; chat() the reverse.
    A positional rewrite must fail loudly, never bind doc_id=True."""
    params = list(inspect.signature(PageIndexClient.chat).parameters.values())
    assert [p.name for p in params[:2]] == ["self", "messages"]
    assert {p.kind for p in params[2:]} == {inspect.Parameter.KEYWORD_ONLY}
    cloud = PageIndexClient(api_key="pi-k")
    with pytest.raises(TypeError, match="positional"):
        cloud.chat("q", True, "pi-1")


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


def test_extra_body_refuses_skeleton_keys():
    """The managed prompt, conversation and tools are the SDK's on every
    lane; extra_body merges last, so a caller's copy would silently
    replace them. Refused at the seam both openai-agents lanes share."""
    for key in ("system", "instructions", "input", "messages", "tools"):
        with pytest.raises(PageIndexAPIError, match="instructions="):
            local_chat._openai_agent(None, "chat", "gpt-test", "sys",
                                     None, None, extra_body={key: "x"})
    with pytest.raises(PageIndexAPIError, match="instructions="):
        local_chat._openai_agent(None, "responses", "gpt-test", "sys",
                                 None, None, extra_body={"input": "x"})


def test_extra_body_refuses_non_dicts_and_argument_keys():
    """The same gate: a non-dict would be splatted into the payload as
    fabricated fields; a field with its own argument rides that argument
    on every lane."""
    for bad in (["ab"], "messages", 5, [("a", 1)]):
        with pytest.raises(PageIndexAPIError,
                           match="extra_body must be a dict"):
            local_chat._refuse_skeleton(bad)
    backend = r"backend= \(API chat models only\)"
    for key, argument in (("stream", "stream="), ("doc_id", "doc_id="),
                          ("folder_id", "folder_id="), ("model", "model="),
                          ("reasoning_effort",
                           r"reasoning_effort= \(own chat models only\)"),
                          ("api_key", backend), ("base_url", backend),
                          ("api_base", backend)):
        with pytest.raises(PageIndexAPIError, match=(
                rf"extra_body cannot carry {key}: use {argument} instead")):
            local_chat._refuse_skeleton({key: True})
    with pytest.raises(PageIndexAPIError, match=(
            rf"cannot carry api_key, base_url: use {backend} instead")):
        local_chat._refuse_skeleton({"api_key": "k", "base_url": "u"})
    local_chat._refuse_skeleton(None)
    local_chat._refuse_skeleton({})
    local_chat._refuse_skeleton({"service_tier": "auto"})


def test_chat_refuses_bad_extra_body_before_any_lane(client, monkeypatch):
    """chat() and chat_completions() check extra_body before entering a
    lane, so no lane does I/O (or, on Responses, an effort merge) on a
    bad value."""
    entered = []
    for door in ("run_chat_completions", "run_responses", "run_messages"):
        monkeypatch.setattr(local_chat, door,
                            lambda c, *a, **kw: entered.append(1))
    for protocol, knobs in ((None, {}), ("chat_completions", {}),
                            ("responses", {}),
                            ("messages", {"model": "claude-x"})):
        with pytest.raises(PageIndexAPIError,
                           match="extra_body must be a dict"):
            client.chat("q", protocol=protocol, reasoning_effort="low",
                        extra_body=["ab"], **knobs)
        with pytest.raises(PageIndexAPIError, match="cannot carry stream"):
            client.chat("q", protocol=protocol, extra_body={"stream": True},
                        **knobs)
    with pytest.raises(PageIndexAPIError, match="cannot carry stream"):
        client.chat_completions("q", extra_body={"stream": True})
    assert entered == []


@needs_anthropic
def test_messages_extra_body_refuses_skeleton_before_transport(client,
                                                                monkeypatch):
    made = []
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda backend=None: made.append(1))
    with pytest.raises(PageIndexAPIError, match="instructions="):
        local_chat.run_messages(client, "q", model="claude-x",
                                route="anthropic", extra_body={"system": "x"})
    assert made == []


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


def _citing(bridge):
    """Teach a FakeBridge the cited_answer prompt, recording each fetch."""
    bridge.prompts = []

    def get_prompt(name, arguments=None):
        bridge.prompts.append((name, arguments))
        fmt = (arguments or {}).get("format", "markdown")
        return "Cited answers", [{"role": "user", "content": {
            "type": "text", "text": f"CITATIONS — {fmt}"}}]

    bridge.get_prompt = get_prompt


@needs_agents
def test_chat_citations_join_the_managed_prompt(bridge_client, fake_model):
    """citations=True on own-model chat: the server's cited_answer prompt
    (PageIndex chat's cite format) joins the system prompt after the
    managed prompt and before the caller's instructions; off fetches
    nothing."""
    client, bridge = bridge_client
    _citing(bridge)
    fake = fake_model([[_msg_item("ok")], [_msg_item("ok")]])
    assert client.chat("q", citations=True, instructions="analyst") == "ok"
    system = fake.instructions[0]
    assert (system.index("CLOUD LIVE INSTRUCTIONS")
            < system.index("CITATIONS — cite") < system.index("analyst"))
    assert bridge.prompts == [("cited_answer", {"format": "cite"})]
    client.chat("q")
    assert "CITATIONS" not in fake.instructions[1]
    assert len(bridge.prompts) == 1


def test_chat_citations_on_protocol_lanes(bridge_client, monkeypatch):
    """The protocol lanes carry the same guidance in their instructions /
    system: prepended to a string, a leading block before caller blocks."""
    client, bridge = bridge_client
    _citing(bridge)
    seen = []
    monkeypatch.setattr(local_chat, "run_responses",
                        lambda c, input, **kw: seen.append(kw) or "door")
    monkeypatch.setattr(local_chat, "run_messages",
                        lambda c, messages, **kw: seen.append(kw) or "door")
    client.chat("q", protocol="responses", citations=True,
                instructions="be brief")
    assert seen[-1]["instructions"] == "CITATIONS — cite\n\nbe brief"
    blocks = [{"type": "text", "text": "persona"}]
    client.chat("q", protocol="messages", model="claude-x", citations=True,
                instructions=blocks)
    assert seen[-1]["system"] == [
        {"type": "text", "text": "CITATIONS — cite"}, *blocks]
    client.chat("q", protocol="messages", model="claude-x", citations=True)
    assert seen[-1]["system"] == "CITATIONS — cite"


def test_chat_citations_managed(monkeypatch):
    """Managed chat: citations=True is the endpoint's enable_citations; a
    format name raises rather than silently meaning True."""
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    seen = []
    monkeypatch.setattr(
        cloud._api, "chat_completions",
        lambda **kw: seen.append(kw) or (
            iter([_cloud_chunk("ok")]) if kw.get("stream")
            else {"choices": [{"message": {"content": "ok"}}]}))
    assert cloud.chat("q", citations=True) == "ok"
    assert seen[-1]["enable_citations"] is True
    assert "".join(cloud.chat("q", stream=True, citations=True)) == "ok"
    assert seen[-1]["enable_citations"] is True
    cloud.chat("q")
    assert seen[-1]["enable_citations"] is False
    cloud.chat("q", protocol="chat_completions", citations=True)
    assert seen[-1]["enable_citations"] is True
    with pytest.raises(PageIndexAPIError, match="True or False"):
        cloud.chat("q", citations="cite")


def test_chat_citations_local_documents_use_the_frozen_copy(client, monkeypatch):
    """Local documents: the frozen copy joins the system prompt the same
    way (pages are all local content has); a format name raises rather
    than silently meaning cite."""
    from pageindex.agent_tools import LOCAL_CITATION_PROMPTS
    seen = []
    monkeypatch.setattr(
        local_chat, "run_chat_completions",
        lambda c, messages, **kw: seen.append(messages) or {
            "choices": [{"message": {"content": "ok"}}]})
    assert client.chat("q", citations=True, instructions="analyst") == "ok"
    assert seen[-1][0] == {"role": "system", "content":
                           LOCAL_CITATION_PROMPTS["cite"] + "\n\nanalyst"}
    client.chat("q")
    assert seen[-1][0]["role"] == "user"
    with pytest.raises(PageIndexAPIError, match="True or False"):
        client.chat("q", citations="markdown")


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


# ── a tool failure that survived the bridge's retries fails the run fast ──

@needs_agents
def test_translate_run_error_unwraps_a_tool_failure():
    """A failure the invoker re-raised leaves the run wrapped in the
    framework's exception; the caller gets it back with its status."""
    from agents.exceptions import AgentsException
    wrapped = AgentsException("Error invoking MCP tool get_document")
    wrapped.__cause__ = PageIndexAPIError("MCP request failed: HTTP 429",
                                          status_code=429)
    err = local_chat._translate_run_error(wrapped, None, "chat")
    assert err.status_code == 429 and "HTTP 429" in str(err)
    plain = local_chat._translate_run_error(AgentsException("boom"), None,
                                            "chat")
    assert plain.status_code is None and "agent backend failed" in str(plain)


def test_model_backend_error_keeps_the_status_code():
    limited = Exception("rate limited")
    limited.status_code = 429
    assert local_chat._model_backend_error(limited, "chat").status_code == 429
    assert local_chat._model_backend_error(
        Exception("x"), "chat").status_code is None


def _rate_limited(name, arguments):
    raise PageIndexAPIError("MCP request failed: HTTP 429", status_code=429)


@needs_agents
def test_bridge_chat_fails_fast_on_a_rate_limited_tool(bridge_client,
                                                        fake_model):
    """A 429 that survived the bridge's retries ends the run with its
    status — no second model turn over an error envelope."""
    client, bridge = bridge_client
    bridge.call_tool = _rate_limited
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "r.pdf"})],
        [_msg_item("never reached")],
    ])
    with pytest.raises(PageIndexAPIError, match="HTTP 429") as info:
        client.chat_completions("What?")
    assert info.value.status_code == 429
    assert len(fake.instructions) == 1


@needs_agents
def test_a_cut_response_runs_no_tools(bridge_client, fake_model):
    """openai-agents before 0.22 would run a cut turn's tool calls on their
    half-written arguments; the run ends at the cut instead, as the stream
    does."""
    client, bridge = bridge_client
    ran = []
    bridge.call_tool = lambda name, arguments: ran.append(name)
    fake = fake_model([[_call_item("get_document", {"doc_name": "r.pdf"})],
                       [_msg_item("never reached")]])

    async def create(*args, **kwargs):
        return _cut_response()

    fake._client = types.SimpleNamespace(
        responses=types.SimpleNamespace(create=create))
    result = client._responses("q")
    assert ran == []
    assert result["status"] == "incomplete"
    assert [item["type"] for item in result["messages"]] == ["reasoning", "message"]


@needs_agents
@pytest.mark.parametrize("run, key", [
    (lambda c: c.chat_completions("q"), "prompt_tokens"),
    (lambda c: list(c.chat_completions("q", stream=True)), "prompt_tokens"),
    (lambda c: c.chat("q"), "prompt_tokens"),
    (lambda c: list(c.chat("q", stream=True)), "prompt_tokens"),
    (lambda c: c._responses("q"), "input_tokens"),
    (lambda c: list(c._responses("q", stream=True)), "input_tokens"),
])
def test_tool_failure_keeps_the_usage_of_its_turn(bridge_client, fake_model,
                                                  run, key):
    """openai-agents records a streamed turn only after its tools ran, so
    the turn whose tool failed was missing from streamed usage."""
    client, bridge = bridge_client
    bridge.call_tool = _rate_limited
    fake_model([[_call_item("get_document", {"doc_name": "r.pdf"})],
                [_msg_item("never reached")]])
    with pytest.raises(PageIndexAPIError, match="HTTP 429") as info:
        run(client)
    usage = info.value.usage
    assert usage is not None and usage[key] == 10
    assert len(usage["request_usage_entries"]) == 1


@needs_anthropic
def test_messages_fail_fast_is_quiet(bridge_client, fake_anthropic, caplog):
    """The runner's own tool-error logging never reports the failure the
    lane is about to raise."""
    client, bridge = bridge_client
    bridge.call_tool = _rate_limited
    fake_anthropic([_anthropic_message(
        [{"type": "tool_use", "id": "tu_1", "name": "get_document",
          "input": {"doc_name": "r.pdf"}}], "tool_use")])
    with pytest.raises(PageIndexAPIError, match="HTTP 429"):
        client.chat("q", protocol="messages", model="claude-test",
                    extra_body={"max_tokens": 100})
    assert not [r for r in caplog.records if r.name.startswith("anthropic")]


@needs_anthropic
@pytest.mark.parametrize("stop_reason", ["tool_use", "max_tokens", "refusal"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("max_turns", [1, 2])
def test_messages_fail_fast_on_a_rate_limited_tool(
        bridge_client, fake_anthropic, stop_reason, stream, max_turns):
    """Real runners must surface executed tools' failures before another
    model call or a max_turns exit, and preserve terminal-turn policy."""
    client, bridge = bridge_client
    tool_calls = []

    def fail(name, arguments):
        tool_calls.append((name, arguments))
        return _rate_limited(name, arguments)

    bridge.call_tool = fail
    replies = [
        _anthropic_message([{"type": "tool_use", "id": "tu_1",
                             "name": "get_document",
                             "input": {"doc_name": "r.pdf"}}], stop_reason),
        _anthropic_message([{"type": "text", "text": "never reached"}],
                           "end_turn"),
    ]
    calls = fake_anthropic([_anthropic_sse(reply) for reply in replies]
                          if stream else replies)

    def run():
        result = client.chat("q", protocol="messages", model="claude-test",
                             extra_body={"max_tokens": 100}, stream=stream,
                             max_turns=max_turns)
        return list(result) if stream else result

    executes_tools = (stop_reason == "tool_use"
                      or (stop_reason == "max_tokens"
                          and _ANTHROPIC_RUNS_CUT_TOOL_TURNS))
    if executes_tools:
        with pytest.raises(PageIndexAPIError, match="HTTP 429") as info:
            run()
        assert info.value.status_code == 429
        usage = info.value.usage
        assert usage is not None and usage["input_tokens"] == 10  # billed
        assert tool_calls == [("get_document", {"doc_name": "r.pdf"})]
    else:
        run()
        assert tool_calls == []
    assert len(calls) == 1, "a second model turn ran"


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
def test_messages_runs_each_tool_once(bridge_client, fake_anthropic, stream):
    """Failure checks preserve normal tool execution across multiple turns."""
    client, bridge = bridge_client
    replies = [_anthropic_message([
        {"type": "tool_use", "id": tool_id, "name": "get_document",
         "input": {"doc_name": "r.pdf"}}], "tool_use")
        for tool_id in ("tu_1", "tu_2")]
    replies.append(_anthropic_message([{"type": "text", "text": "Done"}],
                                      "end_turn"))
    calls = fake_anthropic([_anthropic_sse(reply) for reply in replies]
                          if stream else replies)
    result = client.chat("q", protocol="messages", model="claude-test",
                         extra_body={"max_tokens": 100}, stream=stream)
    if stream:
        list(result)
    assert len(calls) == 3
    assert bridge.calls == [("get_document", {"doc_name": "r.pdf"})] * 2


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_messages_tool_failure_preserves_client_lifecycle(
        bridge_client, fake_anthropic, monkeypatch, stream, cached):
    """Failed runs close owned clients and leave cached clients reusable,
    with no failure state leaking into the next run."""
    client, bridge = bridge_client
    replies = [
        _anthropic_message([_anthropic_tool_use()], "tool_use"),
        _anthropic_message([_anthropic_tool_use()], "tool_use"),
        _anthropic_message([{"type": "text", "text": "Recovered"}],
                           "end_turn"),
    ]
    calls = fake_anthropic([_anthropic_sse(reply) for reply in replies]
                          if stream else replies)
    backend = local_chat._anthropic_client()
    monkeypatch.setattr(local_chat, "_ANTHROPIC_CLIENTS",
                        {"test": backend} if cached else {})
    original_call = bridge.call_tool
    bridge.call_tool = _rate_limited

    def run():
        result = client.chat("q", protocol="messages", model="claude-test",
                             extra_body={"max_tokens": 100}, stream=stream)
        return list(result) if stream else result

    try:
        with pytest.raises(PageIndexAPIError, match="HTTP 429"):
            run()
        assert len(calls) == 1
        assert backend.is_closed() is (not cached)
        if cached:
            bridge.call_tool = original_call
            run()
            assert len(calls) == 3
            assert len(bridge.calls) == 1
            assert not backend.is_closed()
    finally:
        backend.close()


@needs_agents
def test_bridge_chat_keeps_model_slips_model_visible(bridge_client,
                                                     fake_model):
    """Only the invoker's re-raised failures escape: a model-side slip (bad
    JSON arguments) still comes back to the model as text and the run
    goes on."""
    from openai.types.responses import ResponseFunctionToolCall
    client, bridge = bridge_client
    fake = fake_model([
        [ResponseFunctionToolCall(id="fc_1", type="function_call",
                                  call_id="call_1", name="get_document",
                                  arguments="{not json", status="completed")],
        [_msg_item("Recovered")],
    ])
    result = client.chat_completions("What?")
    assert result["choices"][0]["message"]["content"] == "Recovered"
    assert len(fake.instructions) == 2 and bridge.calls == []


# ── client-level instructions: after the managed base, on every surface ──

def test_client_instructions_follow_the_managed_base_everywhere(store_path):
    from pageindex.agent_tools import AGENT_INSTRUCTIONS
    client = PageIndexLocalClient(storage_path=store_path,
                                  instructions="PERSONA")
    assert client.agent_instructions() == AGENT_INSTRUCTIONS + "\n\nPERSONA"
    managed = local_chat._managed_instructions(client, ["CALL", "HISTORY"])
    marks = [managed.index(m) for m in
             (CHAT_HEADER, AGENT_INSTRUCTIONS, "PERSONA", "CALL", "HISTORY")]
    assert marks == sorted(marks)
    blocks = local_chat._anthropic_system(client, "CALL")
    assert blocks[0]["text"].endswith("\n\nPERSONA")
    assert blocks[1]["text"] == "CALL"
    plain = PageIndexLocalClient(storage_path=store_path)
    assert plain.agent_instructions() == AGENT_INSTRUCTIONS


@needs_agents
def test_chat_reaches_the_model_with_client_instructions(store_path,
                                                          fake_model):
    client = PageIndexLocalClient(storage_path=store_path,
                                  instructions="PERSONA")
    fake = fake_model([[_msg_item("ok")]])
    assert client.chat([{"role": "system", "content": "CALL"},
                        {"role": "user", "content": "hi"}]) == "ok"
    assert fake.instructions[0].endswith("\n\nPERSONA\n\nCALL")


def test_bridge_client_instructions_follow_the_live_instructions(
        bridge_client):
    client, _ = bridge_client
    client.instructions = "PERSONA"
    assert client.agent_instructions() == "CLOUD LIVE INSTRUCTIONS\n\nPERSONA"


@needs_agents
def test_openai_agent_config_carries_client_instructions(store_path):
    client = PageIndexLocalClient(storage_path=store_path,
                                  instructions="PERSONA")
    assert client.openai_agent_config()["instructions"].endswith("PERSONA")


def test_anthropic_runner_config_carries_client_instructions(store_path):
    pytest.importorskip("anthropic")
    client = PageIndexLocalClient(storage_path=store_path,
                                  instructions="PERSONA")
    assert client.anthropic_runner_config("claude-x")["system"].endswith(
        "PERSONA")


def test_claude_agent_config_carries_client_instructions(store_path):
    pytest.importorskip("claude_agent_sdk")
    client = PageIndexLocalClient(storage_path=store_path,
                                  instructions="PERSONA")
    assert client.claude_agent_config()["system_prompt"].endswith("PERSONA")


def test_managed_chat_sends_one_leading_system_row(monkeypatch):
    """One system row first: the client's, the call's, then the history's."""
    cloud = PageIndexCloudClient(api_key="pi-k", instructions="PERSONA")
    seen = {}
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kw: seen.update(kw) or {
                            "choices": [{"message": {"content": "ok"}}]})
    history = [{"role": "user", "content": "q1"},
               {"role": "assistant", "content": "a1"},
               {"role": "system", "content": "HISTORY"},
               {"role": "user", "content": "q2"}]
    assert cloud.chat(history, instructions="CALL") == "ok"
    assert seen["messages"] == [
        {"role": "system", "content": "PERSONA\n\nCALL\n\nHISTORY"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}]
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kw: seen.update(kw) or iter([]))
    assert list(cloud.chat("q", stream=True, show_process=False)) == []
    assert seen["messages"] == [{"role": "system", "content": "PERSONA"},
                                {"role": "user", "content": "q"}]
    cloud.chat_completions([{"role": "user", "content": "q"},
                            {"role": "developer", "content": "DEV"}])
    assert seen["messages"][0] == {"role": "system",
                                   "content": "PERSONA\n\nDEV"}


def test_managed_fold_leaves_non_system_rows_to_the_endpoint(monkeypatch):
    """Only system rows fold; blank ones drop; the rest is not validated."""
    cloud = PageIndexCloudClient(api_key="pi-k")
    seen = {}
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kw: seen.update(kw) or {
                            "choices": [{"message": {"content": "ok"}}]})
    leading = [{"role": "system", "content": "S"},
               {"role": "user", "content": "q"}]
    cloud.chat(leading)
    assert seen["messages"] == leading
    history = [{"role": "user", "content": [{"type": "text", "text": "q"}],
                "name": "ray"},
               {"role": "assistant", "content": None,
                "tool_calls": [{"id": "c"}]},
               {"role": "tool", "tool_call_id": "c", "content": "x"},
               {"role": "system", "content": "   "}]
    cloud.chat(history)
    assert seen["messages"] == history[:-1]


def test_chat_history_takes_any_iterable(monkeypatch):
    """Tuples and generators ride both lanes; the call's instructions land."""
    cloud = PageIndexCloudClient(api_key="pi-k")
    seen = {}
    monkeypatch.setattr(cloud._api, "chat_completions",
                        lambda **kw: seen.update(kw) or {
                            "choices": [{"message": {"content": "ok"}}]})
    row = {"role": "user", "content": "q"}
    cloud.chat(iter([row]), instructions="CALL")
    assert seen["messages"] == [{"role": "system", "content": "CALL"}, row]
    assert local_chat._split_chat_messages((row,)) == ([], [row])


def test_system_text_refuses_non_text_parts():
    text = {"type": "text", "text": "A"}
    assert local_chat._system_text(
        [text, {"type": "text", "text": "B"}]) == "A\nB"
    with pytest.raises(PageIndexAPIError, match="text parts"):
        local_chat._system_text(
            [text, {"type": "image_url", "image_url": {"url": "u"}}])


def test_managed_instructions_drop_blank_system_texts(store_path):
    client = PageIndexLocalClient(storage_path=store_path)
    assert (local_chat._managed_instructions(client, ["", "  ", "X"])
            == local_chat._managed_instructions(client, ["X"]))
