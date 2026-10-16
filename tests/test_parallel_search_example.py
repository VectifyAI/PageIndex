"""Run the example through the Agents runner, without model or network costs."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("agents")
from agents import Model, ModelResponse
from agents.usage import Usage
from mcp import types
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
)
from pageindex import PageIndexLocalClient
from pageindex.local_store import DocStore


spec = importlib.util.spec_from_file_location(
    "parallel_example", Path(__file__).parents[1] / "examples/parallel_search/agent.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ScriptedModel(Model):
    """Request real tool dispatch and inspect the results fed back by Runner."""
    def __init__(self):
        self.turn = 0
        self.inputs = []

    async def get_response(self, system_instructions, input, model_settings,
                           tools, output_schema, handoffs, tracing, **kwargs):
        self.inputs.append(input)
        calls = [
            ("get_page_content", {"doc_name": "report.pdf", "pages": "1"}),
            ("web_search", {"objective": "Public report context", "search_queries": ["public report context"]}),
            ("web_fetch", {"urls": ["https://example.org/report"]}),
        ]
        if self.turn < len(calls):
            name, arguments = calls[self.turn]
            assert name in [tool.name for tool in tools]
            output = [ResponseFunctionToolCall(
                type="function_call", name=name, arguments=json.dumps(arguments),
                call_id=f"call-{self.turn}", id=f"tool-{self.turn}")]
        else:
            received = json.dumps(input)
            assert "Document evidence" in received
            assert "Public context" in received
            assert "Fetched source" in received
            output = [ResponseOutputMessage(
                type="message", id="answer", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text="Combined evidence", annotations=[])])]
        self.turn += 1
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield


class FakeSearchServer(example.MCPServerStreamableHttp):
    def __init__(self):
        super().__init__(params={"url": example.SEARCH_URL})
        self.calls = []
        self.closed = False

    async def connect(self):
        pass

    async def cleanup(self):
        self.closed = True

    async def list_tools(self, *args, **kwargs):
        return [types.Tool.model_validate({
            "name": name, "description": "Web evidence",
            "inputSchema": {"type": "object", "properties": properties,
                            "required": list(properties), "additionalProperties": False},
        }) for name, properties in [
            ("web_search", {"objective": {"type": "string"}, "search_queries": {"type": "array", "items": {"type": "string"}}}),
            ("web_fetch", {"urls": {"type": "array", "items": {"type": "string"}}}),
        ]]

    async def call_tool(self, tool_name, arguments, **kwargs):
        self.calls.append((tool_name, arguments))
        text = "Public context" if tool_name == "web_search" else "Fetched source"
        return types.CallToolResult.model_validate({
            "content": [{"type": "text", "text": text}], "isError": False})


def test_runner_combines_local_and_web_evidence_and_closes(tmp_path, monkeypatch):
    storage = tmp_path / "library"
    DocStore(storage).save_document(
        "report", {"id": "report", "name": "report.pdf", "status": "completed",
                   "createdAt": "2026-01-01T00:00:00", "pageNum": 1},
        [{"title": "Report", "node_id": "0000", "start_index": 1, "end_index": 1}],
        [{"page_index": 1, "markdown": "Document evidence"}],
    )
    client = PageIndexLocalClient(storage_path=storage)
    server = FakeSearchServer()
    monkeypatch.setattr(example, "search_server", lambda: server)
    model = ScriptedModel()
    result = asyncio.run(example.answer(client, "Compare sources", model, "report"))
    assert result == "Combined evidence"
    assert [name for name, _ in server.calls] == ["web_search", "web_fetch"]
    assert "report.pdf" in json.dumps(model.inputs[0])
    assert server.closed


def test_connection_closes_when_runner_fails(tmp_path, monkeypatch):
    server = FakeSearchServer()
    monkeypatch.setattr(example, "search_server", lambda: server)

    async def fail(*args, **kwargs):
        raise RuntimeError("runner failed")

    monkeypatch.setattr(example.Runner, "run", fail)
    with pytest.raises(RuntimeError, match="runner failed"):
        asyncio.run(example.answer(PageIndexLocalClient(storage_path=tmp_path), "Question", "gpt-4.1"))
    assert server.closed


def test_missing_document_fails_before_remote_connection(tmp_path, monkeypatch):
    def unexpected_connection():
        pytest.fail("Missing document should be resolved before remote connection")

    monkeypatch.setattr(example, "search_server", unexpected_connection)
    with pytest.raises(Exception, match="not found|Not found|does not exist"):
        asyncio.run(example.answer(PageIndexLocalClient(storage_path=tmp_path), "Question", "gpt-4.1", "missing"))
