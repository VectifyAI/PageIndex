"""Whole-run budgets over real runners and deterministic model transports."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

import pageindex.local_chat as local_chat
from pageindex import (PageIndexAPIError, PageIndexClient,
                       PageIndexTimeoutError)
from test_local_chat import (_HAS_AGENTS, needs_agents, needs_anthropic,
                             seed_doc,
                             client as client, fake_model as fake_model,
                             store_path as store_path,
                             _anthropic_message, _anthropic_sse,
                             _call_item, _msg_item)


@pytest.fixture(autouse=True)
def warm_model_imports(client, fake_model):
    if _HAS_AGENTS:
        fake_model([[_msg_item("warm")]])
        client.chat("warm")


@needs_agents
@pytest.mark.parametrize("protocol", [None, "chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_deadline_spans_turns_and_waits_for_tool(client, store_path,
                                              fake_model, protocol, stream,
                                              monkeypatch):
    import pageindex.agent_tools as tools

    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("unreachable")],
    ])
    finished = threading.Event()
    original = tools._IMPLEMENTATIONS["get_document"]

    def slow_tool(*args, **kwargs):
        time.sleep(0.8)
        try:
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setitem(tools._IMPLEMENTATIONS, "get_document", slow_tool)
    with pytest.raises(PageIndexAPIError) as caught:
        result = client.chat("q", protocol=protocol, stream=stream,
                             timeout=0.5)
        if stream:
            list(result)
    assert caught.value.code == "deadline_exceeded"
    assert len(fake.inputs) == 1
    assert finished.is_set()


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_deadline_validation_precedes_setup(client, timeout, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid timeout reached setup")

    monkeypatch.setattr(local_chat, "_chat_agent", unexpected)
    with pytest.raises(PageIndexAPIError, match="positive finite"):
        client.chat("q", timeout=timeout)


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stage", ["model", "tool"])
def test_messages_deadline_uses_real_async_runner(client, store_path,
                                                monkeypatch, stream, stage):
    import anthropic
    import pageindex.agent_tools as tools
    from test_local_chat import anthropic_httpx as httpx

    seed_doc(store_path, "pi-a", "report.pdf")
    requests = []
    stopped = threading.Event()

    async def handle(request):
        requests.append(json.loads(request.content))
        if stage == "model":
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        message = _anthropic_message([
            {"type": "tool_use", "id": "tool-1", "name": "get_document",
             "input": {"doc_name": "report.pdf"}}], "tool_use")
        if stream:
            return httpx.Response(
                200, text=_anthropic_sse(message),
                headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=message)

    backend = anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda *args, **kwargs: backend)
    original = tools._IMPLEMENTATIONS["get_document"]

    def slow_tool(*args, **kwargs):
        try:
            time.sleep(0.8)
            return original(*args, **kwargs)
        finally:
            stopped.set()

    monkeypatch.setitem(tools._IMPLEMENTATIONS, "get_document", slow_tool)
    with pytest.raises(PageIndexTimeoutError):
        result = client.chat("q", protocol="messages", model="claude-test",
                             stream=stream, timeout=0.5)
        if stream:
            list(result)
    assert len(requests) == 1
    assert stopped.is_set()
    assert backend.is_closed()


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
def test_messages_deadline_success_retains_output(client, monkeypatch, stream):
    import anthropic
    from test_local_chat import anthropic_httpx as httpx

    message = _anthropic_message([{"type": "text", "text": "answer"}],
                                  "end_turn")

    async def handle(request):
        if stream:
            return httpx.Response(
                200, text=_anthropic_sse(message),
                headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=message)

    backend = anthropic.AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(local_chat, "_anthropic_client",
                        lambda *args, **kwargs: backend)
    result = client.chat("q", protocol="messages", model="claude-test",
                         stream=stream, timeout=5)
    if stream:
        assert any(getattr(getattr(event, "delta", None), "text", None)
                   == "answer" for event in result)
        result.close()
    else:
        assert result["content"][0]["text"] == "answer"
        assert result["messages"][-1]["content"][0]["text"] == "answer"
    assert backend.is_closed()


@pytest.fixture
def cloud_transport(monkeypatch):
    import httpx

    original = httpx.AsyncClient
    clients = []

    def install(handler):
        def create(**kwargs):
            client = original(transport=httpx.MockTransport(handler), **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", create)
        return clients

    return install


@needs_agents
@pytest.mark.parametrize("options", [
    {"doc_id": "pi-a"}, {"folder_id": "folder-a"}, {}, {"citations": True},
], ids=["metadata", "folder", "initialize", "citation"])
def test_setup_network_is_inside_budget(fake_model, cloud_transport, options):
    stopped = threading.Event()

    async def handle(request):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    clients = cloud_transport(handle)
    fake = fake_model([[_msg_item("unreachable")]])
    cloud = PageIndexClient(api_key="test", chat_model="openai/test")
    with pytest.raises(PageIndexTimeoutError):
        cloud.chat("q", timeout=0.5, **options)
    assert stopped.is_set()
    assert fake.inputs == []
    assert clients and all(client.is_closed for client in clients)


@pytest.mark.parametrize("trickle", [False, True])
def test_cloud_response_body_is_cancelled_at_deadline(cloud_transport,
                                                    trickle):
    import httpx
    from pageindex import deadline

    stopped = threading.Event()

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b" "
                if trickle:
                    await asyncio.sleep(0.01)
                else:
                    await asyncio.Event().wait()

        async def aclose(self):
            stopped.set()

    async def handle(request):
        assert request.extensions["timeout"]["read"] <= 0.3
        return httpx.Response(200, stream=Body())

    clients = cloud_transport(handle)
    with deadline.active(deadline.Deadline(0.3)):
        with pytest.raises(PageIndexTimeoutError):
            deadline.get("https://test.invalid/metadata", timeout=30)
    assert stopped.is_set()
    assert all(client.is_closed for client in clients)


def test_mcp_backoff_stops_before_next_attempt(cloud_transport):
    import httpx
    from pageindex import deadline
    from pageindex.mcp_bridge import McpBridge

    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(503)

    cloud_transport(handle)
    bridge = McpBridge("https://test.invalid/mcp", {})
    with deadline.active(deadline.Deadline(0.5)):
        with pytest.raises(PageIndexTimeoutError):
            bridge.list_tools()
    assert len(requests) == 2


@needs_agents
@pytest.mark.parametrize("protocol", [None, "chat_completions", "responses"])
@pytest.mark.parametrize("consume", [False, True])
def test_unstarted_stream_releases_backend(client, fake_model, monkeypatch,
                                          protocol, consume):
    from pageindex import deadline

    fake = fake_model([[_msg_item("unreachable")]])
    closes = []

    class Backend:
        async def close(self):
            closes.append(True)

    fake._client = Backend()
    stream = client.chat("q", protocol=protocol, stream=True, timeout=5)
    if consume:
        clock = time.monotonic
        monkeypatch.setattr(deadline, "time",
                            SimpleNamespace(monotonic=lambda: clock() + 10))
        with pytest.raises(PageIndexTimeoutError):
            list(stream)
    else:
        stream.close()
        stream.close()
    assert fake.inputs == []
    assert closes == [True]


@needs_agents
def test_stream_deadline_runs_while_consumer_is_paused(client, fake_model,
                                                    monkeypatch):
    from openai.types.responses import ResponseTextDeltaEvent

    fake = fake_model([[_msg_item("unreachable")]])
    stopped = threading.Event()

    async def stream_response(*args, **kwargs):
        try:
            while True:
                yield ResponseTextDeltaEvent(
                    type="response.output_text.delta", delta="x",
                    content_index=0, item_id="msg", output_index=0,
                    sequence_number=1, logprobs=[])
                await asyncio.sleep(0)
        finally:
            stopped.set()

    monkeypatch.setattr(fake, "stream_response", stream_response)
    stream = client.chat("q", stream=True, timeout=0.5, show_process=False)
    assert next(stream) == "x"
    assert stopped.wait(2)
    with pytest.raises(PageIndexTimeoutError):
        list(stream)


@needs_agents
def test_sync_deadline_inside_an_event_loop(client, fake_model, monkeypatch):
    fake = fake_model([[_msg_item("unreachable")]])
    stopped = threading.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(fake, "get_response", blocked)

    async def caller():
        with pytest.raises(PageIndexTimeoutError):
            client.chat("q", timeout=0.3)

    asyncio.run(caller())
    assert stopped.is_set()


@needs_agents
@pytest.mark.parametrize("protocol", [None, "chat_completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_budget_is_shared_by_model_turns(client, store_path, fake_model,
                                       monkeypatch, protocol, stream):
    seed_doc(store_path, "pi-a", "report.pdf")
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"}, f"tool-{i}")]
        for i in range(3)
    ] + [[_msg_item("unreachable")]])
    response = fake.get_response
    stream_response = fake.stream_response
    stopped = threading.Event()

    async def slow_response(*args, **kwargs):
        try:
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            stopped.set()
            raise
        return await response(*args, **kwargs)

    async def slow_stream(*args, **kwargs):
        try:
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            stopped.set()
            raise
        async for event in stream_response(*args, **kwargs):
            yield event

    monkeypatch.setattr(fake, "get_response", slow_response)
    monkeypatch.setattr(fake, "stream_response", slow_stream)
    with pytest.raises(PageIndexTimeoutError) as caught:
        result = client.chat("q", protocol=protocol, stream=stream, timeout=1)
        if stream:
            list(result)
    assert 2 <= len(fake.inputs) <= 3
    assert stopped.is_set()
    assert caught.value.code == "deadline_exceeded"
    assert caught.value.timeout == 1


def test_mcp_lock_wait_uses_remaining_budget(cloud_transport):
    from pageindex import deadline
    from pageindex.mcp_bridge import McpBridge

    locked = threading.Event()
    release = threading.Event()
    bridge = McpBridge("https://test.invalid/mcp", {})

    def hold_lock():
        with bridge._lock:
            locked.set()
            release.wait(5)

    def unexpected(request):
        pytest.fail("request started after lock wait exhausted the budget")

    cloud_transport(unexpected)
    worker = threading.Thread(target=hold_lock)
    worker.start()
    try:
        assert locked.wait(2)
        with deadline.active(deadline.Deadline(0.1)):
            with pytest.raises(PageIndexTimeoutError):
                bridge.list_tools()
    finally:
        release.set()
        worker.join()


def test_managed_chat_refuses_deadline():
    client = PageIndexClient(api_key="test")
    with pytest.raises(PageIndexAPIError, match="own chat model"):
        client.chat("q", timeout=1)


@needs_anthropic
@pytest.mark.parametrize("status", [200, 503])
def test_messages_deadline_borrows_transport_and_disables_retries(client,
                                                               status):
    from test_local_chat import anthropic_httpx as httpx

    calls = []

    async def handle(request):
        calls.append(request)
        assert request.extensions["timeout"]["read"] == 0.1
        return httpx.Response(status, json=_anthropic_message(
            [{"type": "text", "text": "answer"}], "end_turn"))

    transport = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        options = {"http_client": transport, "timeout": 0.1, "max_retries": 3}
        if status == 503:
            with pytest.raises(PageIndexAPIError) as caught:
                client.chat("q", protocol="messages", model="claude-test",
                             timeout=5, backend=options)
            assert caught.value.status_code == 503
        else:
            result = client.chat("q", protocol="messages", model="claude-test",
                                 timeout=5, backend=options)
            assert result["content"][0]["text"] == "answer"
        assert len(calls) == 1
        assert not transport.is_closed
    finally:
        asyncio.run(transport.aclose())


@needs_agents
@pytest.mark.parametrize("protocol", [None, "chat_completions", "responses"])
def test_expiry_after_model_prevents_tool_dispatch(client, store_path,
                                                 fake_model, monkeypatch,
                                                 protocol):
    from pageindex import deadline
    import pageindex.agent_tools as tools

    seed_doc(store_path, "pi-a", "report.pdf")
    clock = [0.0]
    monkeypatch.setattr(deadline, "time",
                        SimpleNamespace(monotonic=lambda: clock[0]))
    fake = fake_model([
        [_call_item("get_document", {"doc_name": "report.pdf"})],
        [_msg_item("unreachable")],
    ])
    response = fake.get_response

    async def late_response(*args, **kwargs):
        result = await response(*args, **kwargs)
        clock[0] = 2.0
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("tool dispatched after expiry")

    monkeypatch.setattr(fake, "get_response", late_response)
    monkeypatch.setitem(tools._IMPLEMENTATIONS, "get_document", unexpected)
    with pytest.raises(PageIndexTimeoutError):
        client.chat("q", protocol=protocol, timeout=1)
    assert len(fake.inputs) == 1


def test_mcp_deadline_preserves_session_and_retries(cloud_transport):
    import httpx
    from pageindex import deadline
    from pageindex.mcp_bridge import McpBridge

    calls = []

    async def handle(request):
        payload = json.loads(request.content)
        method = payload["method"]
        calls.append(method)
        assert request.headers["Authorization"] == "Bearer test"
        if method == "initialize":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": payload["id"],
                "result": {"protocolVersion": "2025-06-18"}},
                headers={"Mcp-Session-Id": "session"})
        assert request.headers["Mcp-Session-Id"] == "session"
        if method == "notifications/initialized":
            return httpx.Response(204)
        if calls.count("tools/call") == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={
            "jsonrpc": "2.0", "id": payload["id"],
            "result": {"content": [{"type": "text", "text": "answer"}],
                       "isError": False}})

    clients = cloud_transport(handle)
    bridge = McpBridge("https://test.invalid/mcp",
                       {"Authorization": "Bearer test"})
    with deadline.active(deadline.Deadline(5)):
        blocks, is_error = bridge.call_tool("read", {"doc_name": "report"})
    assert blocks == [{"type": "text", "text": "answer"}]
    assert not is_error
    assert calls == ["initialize", "notifications/initialized",
                     "tools/call", "tools/call"]
    assert all(client.is_closed for client in clients)
