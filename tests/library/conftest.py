"""Fixtures shared by the library tests. No network, no LLM keys."""
import pytest

from pageindex.local_store import DocStore


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway library home; BOOKS_HOME points at it."""
    root = tmp_path / "book-library"
    root.mkdir()
    monkeypatch.setenv("BOOKS_HOME", str(root))
    return root


@pytest.fixture
def store(home):
    return DocStore(str(home / ".pageindex"))


@pytest.fixture
def sample_pages():
    """Six one-line pages; page i contains the word 'word<i>'."""
    return [f"Page {i} text word{i}." for i in range(1, 7)]


@pytest.fixture
def sample_tree():
    """Two chapters; chapter one has two sections."""
    return [
        {"title": "Chapter One", "node_id": "0000", "start_index": 1, "end_index": 4,
         "nodes": [
             {"title": "Section A", "node_id": "0001", "start_index": 2, "end_index": 3, "nodes": []},
             {"title": "Section B", "node_id": "0002", "start_index": 4, "end_index": 4, "nodes": []},
         ]},
        {"title": "Chapter Two", "node_id": "0003", "start_index": 5, "end_index": 6, "nodes": []},
    ]


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace both LLM entry points with a deterministic fake that records
    prompts. Replies with JSON {"summary": "..."} echoing the model name."""
    calls = []

    async def acompletion(model, prompt):
        calls.append((model, prompt))
        return '{"points": ["p"], "summary": "S%d via %s"}' % (len(calls), model)

    def completion(model, prompt, chat_history=None, return_finish_reason=False):
        calls.append((model, prompt))
        text = "D%d via %s" % (len(calls), model)
        return (text, "finished") if return_finish_reason else text

    monkeypatch.setattr("pageindex.utils.llm_acompletion", acompletion)
    monkeypatch.setattr("pageindex.utils.llm_completion", completion)
    return calls
