"""Session lifecycle tests with the real Runner/tools and scripted models."""
import asyncio
import json
import threading

import pytest

from pageindex import PageIndexAPIError, PageIndexCloudClient
from pageindex.integrations.memory import MemoryChatSession
import pageindex.local_chat as local_chat
from test_local_chat import (FakeModel, _call_item, _msg_item, client,
                             fake_model, needs_agents, seed_doc, store_path)

pytestmark = needs_agents


def test_followup_carries_real_tool_result_once_and_snapshots_are_detached(
        client, store_path, fake_model, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    model = fake_model([
        [_call_item("get_page_content", {"doc_name": "report.pdf", "pages": "1"})],
        [_msg_item("Apples")], [_msg_item("More about apples")],
    ])
    keys = []
    original = local_chat._openai_agent
    def record(*args, **kwargs):
        keys.append(kwargs["cache_key"])
        return original(*args, **kwargs)
    monkeypatch.setattr(local_chat, "_openai_agent", record)
    selection = [doc]
    session = MemoryChatSession(client, doc_id=selection, model="gpt-test",
                                instructions="Answer in Chinese.")
    selection.append("other")  # not an alias to the session's allowlist
    async def run():
        first = await session.chat("What fruit?")
        assert [m["role"] for m in session.messages] == ["user", "assistant", "tool", "assistant"]
        assert "Page one text about apples" in json.dumps(session.messages)
        snapshot = session.messages
        snapshot[2]["content"] = "changed outside"
        first["messages"][1]["content"] = "changed returned envelope"
        await session.chat("Tell me more.")
        assert "changed" not in json.dumps(session.messages)
        assert session.context["instructions"].endswith("Answer in Chinese.")
    asyncio.run(run())
    followup = model.inputs[2]
    assert sum(item.get("type") == "function_call" for item in followup) == 1
    assert sum(item.get("type") == "function_call_output" for item in followup) == 1
    assert "Page one text about apples" in json.dumps(followup)
    assert sum("The user has specified document:" in item.get("content", "")
               for item in followup if isinstance(item.get("content"), str)) == 1
    assert keys[0] == keys[1]
    assert session.doc_ids == (doc,)


def test_same_session_serializes_but_other_sessions_progress(client, store_path, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        class Gated(FakeModel):
            async def get_response(self, *args, **kwargs):
                if not self.inputs:
                    started.set()
                    await release.wait()
                return await super().get_response(*args, **kwargs)
        blocked = Gated([[_msg_item("one")], [_msg_item("two")]])
        other = FakeModel([[_msg_item("independent")]])
        monkeypatch.setattr(local_chat, "_openai_model",
                            lambda protocol, model_name, backend=None: blocked if model_name == "first" else other)
        first = MemoryChatSession(client, doc_id=doc, model="first")
        second = MemoryChatSession(client, doc_id=doc, model="second")
        pending = asyncio.create_task(first.chat("first question"))
        await asyncio.wait_for(started.wait(), 2)
        queued = asyncio.create_task(first.chat("second question"))
        separate = await asyncio.wait_for(second.chat("separate question"), 2)
        assert separate["choices"][0]["message"]["content"] == "independent"
        assert first.messages == [] and not queued.done()
        release.set()
        await asyncio.gather(pending, queued)
        assert [m["content"] for m in first.messages if m["role"] == "user"] == ["first question", "second question"]
        assert "separate question" not in json.dumps(first.messages)
        assert "first question" not in json.dumps(second.messages)
        assert "one" in json.dumps(blocked.inputs[1])
    asyncio.run(run())


def test_failure_after_tool_keeps_only_previous_complete_turn(client, store_path, fake_model):
    doc = seed_doc(store_path, "a", "report.pdf")
    model = fake_model([[_msg_item("first")],
        [_call_item("get_page_content", {"doc_name": "report.pdf", "pages": "1"})],
        RuntimeError("injected backend failure"), [_msg_item("retry")]])
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
    async def run():
        await session.chat("good")
        committed = session.messages
        with pytest.raises(RuntimeError, match="injected"):
            await session.chat("failed")
        assert session.messages == committed
        await session.chat("retry question")
        assert "failed" not in json.dumps(model.inputs[-1])
    asyncio.run(run())


def test_cancel_active_and_waiting_turns_does_not_commit(client, store_path, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    async def run():
        started = asyncio.Event()
        class Canceled(FakeModel):
            async def get_response(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()
        fake = Canceled([])
        closed = []
        class Backend:
            async def close(self):
                closed.append(True)
        fake._client = Backend()
        monkeypatch.setattr(local_chat, "_openai_model", lambda *a, **k: fake)
        session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
        active = asyncio.create_task(session.chat("active"))
        await asyncio.wait_for(started.wait(), 2)
        waiting = asyncio.create_task(session.chat("waiting"))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert session.messages == [] and session.context is None
        assert closed == [True]
        monkeypatch.setattr(local_chat, "_openai_model", lambda *a, **k: FakeModel([[_msg_item("ok")]]))
        await session.chat("next")
        assert [m["content"] for m in session.messages] == ["next", "ok"]
    asyncio.run(run())


def test_cancel_during_preparation_closes_backend(client, store_path, fake_model, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    model = fake_model([[_msg_item("unused")]])
    closed, entered, release = [], threading.Event(), threading.Event()
    class Backend:
        async def close(self):
            closed.append(True)
    model._client = Backend()
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
    prepare = session._prepare
    def blocked(history):
        result = prepare(history)
        entered.set()
        assert release.wait(2)
        return result
    monkeypatch.setattr(session, "_prepare", blocked)
    async def run():
        task = asyncio.create_task(session.chat("question"))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # a second cancel must not orphan the preparation thread
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.messages == [] and model.inputs == []
    asyncio.run(run())
    assert closed == [True]


def test_clear_is_serial_and_next_turn_has_no_old_history(client, store_path, fake_model):
    doc = seed_doc(store_path, "a", "report.pdf")
    model = fake_model([[_msg_item("old")], [_msg_item("fresh")]])
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
    async def run():
        await session.chat("old question")
        await session.clear()
        assert session.messages == [] and session.context is None
        await session.chat("new question")
        assert "old question" not in json.dumps(model.inputs[1])
    asyncio.run(run())


def test_clear_waits_for_an_active_turn(client, store_path, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Gated(FakeModel):
            async def get_response(self, *args, **kwargs):
                entered.set()
                await release.wait()
                return await super().get_response(*args, **kwargs)
        model = Gated([[_msg_item("finished")]])
        monkeypatch.setattr(local_chat, "_openai_model", lambda *a, **k: model)
        session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
        turn = asyncio.create_task(session.chat("first"))
        await asyncio.wait_for(entered.wait(), 2)
        clearing = asyncio.create_task(session.clear())
        await asyncio.sleep(0)
        assert not clearing.done()
        release.set()
        await asyncio.gather(turn, clearing)
        assert session.messages == [] and session.context is None
    asyncio.run(run())


def test_configuration_change_during_run_cannot_change_tools_or_commit(client, store_path, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    client.instructions = "original guidance"
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class Gated(FakeModel):
            async def get_response(self, *args, **kwargs):
                entered.set()
                await release.wait()
                return await super().get_response(*args, **kwargs)
        model = Gated([[_msg_item("completed under old config")]])
        monkeypatch.setattr(local_chat, "_openai_model", lambda *a, **k: model)
        session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
        task = asyncio.create_task(session.chat("first"))
        await asyncio.wait_for(entered.wait(), 2)
        client.instructions = "changed guidance"
        client.storage_path = "/different-store"
        release.set()
        with pytest.raises(PageIndexAPIError, match="configuration changed"):
            await task
        assert session.messages == [] and session.context is None
        assert model.instructions[0].endswith("original guidance")
        assert session._client.storage_path == str(store_path)
    asyncio.run(run())


def test_max_turns_failure_keeps_history_empty(client, store_path, fake_model):
    doc = seed_doc(store_path, "a", "report.pdf")
    fake_model([[_call_item("get_document", {"doc_name": "report.pdf"})]])
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test", max_turns=1)
    async def run():
        with pytest.raises(PageIndexAPIError, match="max_turns"):
            await session.chat("unfinished")
        assert session.messages == [] and session.context is None
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [("instructions", "new instruction"),
    ("chat_model", "changed"), ("storage_path", "/different-store"),
    ("chat_backend", {"base_url": "http://127.0.0.1:1"}), ("api_key", "changed")])
def test_source_configuration_change_refuses_new_turn(field, value, client, store_path, fake_model):
    doc = seed_doc(store_path, "a", "report.pdf")
    model = fake_model([[_msg_item("ok")]])
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
    async def run():
        await session.chat("first")
        old = session.messages
        setattr(client, field, value)
        with pytest.raises(PageIndexAPIError, match="configuration changed"):
            await session.chat("refused")
        assert session.messages == old and len(model.inputs) == 1
        await session.clear()  # clearing remains possible after a config change
        assert session.messages == []
    asyncio.run(run())


def test_local_document_allowlist_is_enforced(client, store_path, fake_model):
    doc = seed_doc(store_path, "a", "allowed.pdf")
    seed_doc(store_path, "b", "outside.pdf")
    model = fake_model([[_call_item("get_page_content", {"doc_name": "outside.pdf", "pages": "1"})],
                        [_msg_item("The document is unavailable")]])
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
    async def run():
        await session.chat("try outside scope")
        # The existing executor returns a denial as a tool result, allowing
        # the model to explain it. It must never return the outside text.
        output = next(item for item in model.inputs[-1] if item.get("type") == "function_call_output")
        assert "Page one text about apples" not in json.dumps(output)
        assert "not found" in json.dumps(output).lower()
    asyncio.run(run())


def test_truncated_turn_is_not_committed(client, store_path, fake_model, monkeypatch):
    doc = seed_doc(store_path, "a", "report.pdf")
    fake_model([[_msg_item("partial")]])
    monkeypatch.setattr(local_chat, "_record_chat_finish", lambda agent, record: record.update(finish_reason="length"))
    session = MemoryChatSession(client, doc_id=doc, model="gpt-test")
    async def run():
        with pytest.raises(PageIndexAPIError) as caught:
            await session.chat("too long")
        assert caught.value.partial_answer == "partial"
        assert session.messages == []
    asyncio.run(run())


@pytest.mark.parametrize("override", [dict(doc_id=None), dict(doc_id=[]), dict(doc_id=[""]),
    dict(doc_id={"a": 1}), dict(model=""), dict(model="claude-code/sonnet"),
    dict(protocol="responses"), dict(protocol="messages"), dict(instructions=[]), dict(max_turns=0)])
def test_unsupported_configuration_is_explicit(override, client):
    with pytest.raises(PageIndexAPIError):
        MemoryChatSession(client, **{**dict(doc_id="a", model="gpt-test"), **override})


def test_cloud_and_cross_loop_sessions_are_refused(client):
    with pytest.raises(PageIndexAPIError, match="local documents"):
        MemoryChatSession(PageIndexCloudClient("pi-dummy"), doc_id="a", model="gpt-test")
    session = MemoryChatSession(client, doc_id="a", model="gpt-test")
    asyncio.run(session.clear())
    with pytest.raises(RuntimeError, match="one event loop"):
        asyncio.run(session.clear())
