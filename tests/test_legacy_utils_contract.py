import sys
import asyncio
from types import SimpleNamespace

from pageindex import utils


def test_remove_fields_keeps_legacy_max_len():
    data = {
        "title": "A long title",
        "text": "hidden",
        "nodes": [{"summary": "abcdefghijklmnopqrstuvwxyz"}],
    }

    result = utils.remove_fields(data, fields=["text"], max_len=5)

    assert "text" not in result
    assert result["title"] == "A lon..."
    assert result["nodes"][0]["summary"] == "abcde..."


def test_create_node_mapping_keeps_legacy_page_ranges():
    tree = [
        {
            "node_id": "0001",
            "title": "Root",
            "page_index": 1,
            "nodes": [
                {"node_id": "0002", "title": "Child", "page_index": 3, "nodes": []},
            ],
        }
    ]

    plain = utils.create_node_mapping(tree)
    ranged = utils.create_node_mapping(tree, include_page_ranges=True, max_page=8)

    assert plain["0001"]["title"] == "Root"
    assert ranged["0001"]["start_index"] == 1
    assert ranged["0001"]["end_index"] == 3
    assert ranged["0002"]["start_index"] == 3
    assert ranged["0002"]["end_index"] == 8


def test_create_node_mapping_prefers_existing_start_end_ranges():
    tree = [
        {
            "node_id": "0001",
            "title": "Root",
            "start_index": 1,
            "end_index": 10,
            "nodes": [
                {"node_id": "0002", "title": "Child", "start_index": 3, "end_index": 5},
            ],
        }
    ]

    ranged = utils.create_node_mapping(tree, include_page_ranges=True, max_page=12)

    assert ranged["0001"]["start_index"] == 1
    assert ranged["0001"]["end_index"] == 10
    assert ranged["0002"]["start_index"] == 3
    assert ranged["0002"]["end_index"] == 5


def test_print_tree_keeps_legacy_exclude_fields(capsys):
    tree = [{"node_id": "0001", "title": "Root", "text": "hidden", "page_index": 1}]

    utils.print_tree(tree)

    out = capsys.readouterr().out
    assert "Root" in out
    assert "hidden" not in out
    assert "page_index" not in out


def test_call_llm_keeps_legacy_async_openai_contract(monkeypatch):
    calls = []
    closed = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=" answer ")
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeAsyncOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=FakeCompletions())

        # call_llm must open the client as an async context manager so it is
        # closed (no leaked HTTP connection pool).
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            closed.append(True)
            return False

    fake_openai = SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = asyncio.run(utils.call_llm(
        "hello",
        api_key="sk-test",
        model="gpt-test",
        temperature=0.2,
    ))

    assert result == "answer"
    assert closed == [True]  # client was closed
    assert calls == [{
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
    }]
