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

agents = pytest.importorskip("agents")


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


from agents.models.interface import Model  # noqa: E402


class FakeModel(Model):
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
        self._record(system_instructions, input)
        return ModelResponse(output=self.turns.pop(0), usage=_usage(),
                             response_id=None)

    async def stream_response(self, system_instructions, input,
                              model_settings, tools, output_schema, handoffs,
                              tracing, **kwargs):
        from openai.types.responses import (Response, ResponseCompletedEvent,
                                            ResponseTextDeltaEvent)
        from openai.types.responses.response_usage import (
            InputTokensDetails, OutputTokensDetails, ResponseUsage)
        self._record(system_instructions, input)
        output = self.turns.pop(0)
        sequence = 0
        for item in output:
            if item.type == "message":
                for piece in ("The ", "answer"):
                    sequence += 1
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

anthropic = pytest.importorskip("anthropic")
import httpx  # noqa: E402  (anthropic depends on httpx)


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
