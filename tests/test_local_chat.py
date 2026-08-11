"""Local chat surfaces: three protocols over fake backends — no network,
no LLM keys. Tool execution runs for real against a seeded local store."""
import json
import sys
from pathlib import Path

import pytest

import pageindex.local_chat as local_chat
from pageindex import (PageIndexAPIError, PageIndexCloudClient,
                       PageIndexLocalClient)
from pageindex.local_chat import CHAT_HEADER
from pageindex.local_store import DocStore

sys.path.insert(0, str(Path(__file__).parent.parent))


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
pytestmark_openai = needs_agents


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

        def factory(protocol, model_name):
            state["protocols"].append((protocol, model_name))
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
                               "total_tokens": 30}
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
    with pytest.raises(PageIndexAPIError, match="pageindex\\[openai\\]"):
        client.chat_completions([{"role": "user", "content": "x"}])


def test_cloud_guards():
    cloud = PageIndexCloudClient(api_key="pi-test-key")
    with pytest.raises(PageIndexAPIError, match="local-mode parameters"):
        cloud.chat_completions([{"role": "user", "content": "x"}], model="m")
    with pytest.raises(PageIndexAPIError, match="not available on PageIndex "
                                                "cloud yet"):
        cloud.responses("x")
    with pytest.raises(PageIndexAPIError, match="not available on PageIndex "
                                                "cloud yet"):
        cloud.messages([{"role": "user", "content": "x"}], model="m",
                       max_tokens=10)


# ── responses ──

@needs_agents
def test_responses_end_to_end(client, store_path, fake_model):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    result = client.responses("What status?")
    assert result["id"].startswith("resp_")
    assert result["object"] == "response"
    assert result["status"] == "completed"
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 10,
                               "total_tokens": 30}
    assert fake_model.state["protocols"][0][0] == "responses"
    types = [item.get("type", "message") for item in result["output"]]
    assert "function_call" in types and "function_call_output" in types
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
                 + result["output"]
                 + [{"role": "user", "content": "and now?"}])
    client.responses(follow_up)
    previous_final = first.inputs[-1]
    assert second.inputs[0][:len(previous_final)] == previous_final


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
                 + result["output"]
                 + [{"role": "user", "content": "and now?"}])
    client.responses(follow_up, doc_id="pi-a")
    previous_final = first.inputs[-1]
    assert second.inputs[0][:len(previous_final)] == previous_final


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
    tool_events = [event for event in events
                   if event.get("type") == "response.output_item.done"
                   and event.get("item", {}).get("type")
                   == "function_call_output"]
    assert tool_events, types
    assert types[-1] == "response.completed"
    final = events[-1]["response"]
    assert final["status"] == "completed"
    assert final["usage"]["total_tokens"] == 30


# ── messages (Anthropic engine) ──

try:
    import anthropic
    import httpx
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
        monkeypatch.setattr(local_chat, "_anthropic_client", lambda: fake)
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
    assert calls[0]["max_tokens"] == 4096
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
def test_chat_completions_max_turns_wrapped(client, store_path, fake_model):
    """MaxTurnsExceeded is an engine-internal type; callers get the SDK's
    own error — on both the non-stream and stream paths."""
    seed_doc(store_path, "pi-a", "report.pdf")
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_call_item("get_document", {"doc_name": "report.pdf"}, "call_2")],
        [_msg_item("never reached")],
    ])
    with pytest.raises(PageIndexAPIError, match="max_turns"):
        client.chat_completions([{"role": "user", "content": "q"}],
                                max_turns=1)
    fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_call_item("get_document", {"doc_name": "report.pdf"}, "call_2")],
        [_msg_item("never reached")],
    ])
    with pytest.raises(PageIndexAPIError, match="max_turns"):
        list(client.chat_completions([{"role": "user", "content": "q"}],
                                     stream=True, max_turns=1))
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
    tool_done = next(event for event in events
                     if event.get("type") == "response.output_item.done"
                     and event["item"]["type"] == "function_call_output")
    assert "sequence_number" in tool_done and "output_index" in tool_done


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
    assert result["parallel_tool_calls"] is True
    assert result["tool_choice"] == "auto"
    # Stable cache group: without it openai-agents stamps each run with a
    # fresh prompt_cache_key, defeating round-trip cache routing.
    assert (local_chat._run_kwargs(None)["run_config"].group_id
            == "pageindex-local-chat")


@needs_agents
def test_responses_input_validation(client, fake_model):
    fake_model([])
    for bad in ("", "   ", [], [1], None):
        with pytest.raises(PageIndexAPIError, match="input must be"):
            client.responses(bad)


@needs_agents
def test_stream_abandonment_cancels_pending_turn(client, store_path,
                                                 fake_model):
    """Closing the iterator cancels the run even while it is awaiting the
    backend: the blocked turn is torn down (pump thread exits) instead of
    running — and billing — to completion in the background."""
    import threading
    import time as time_mod
    seed_doc(store_path, "pi-a", "report.pdf")
    baseline = threading.active_count()
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("The answer")],
    ])
    fake.block_from = 2  # turn 2 hangs until cancelled
    stream = client.chat_completions([{"role": "user", "content": "q"}],
                                     stream=True, stream_metadata=True)
    next(stream)  # the opening role chunk
    stream.close()
    deadline = time_mod.monotonic() + 3.0
    while (threading.active_count() > baseline
           and time_mod.monotonic() < deadline):
        time_mod.sleep(0.05)
    assert threading.active_count() <= baseline
    assert fake.deltas_emitted == 0  # turn 2 never produced output


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
