"""
MCP integration tests — exercises mcp_server.py via subprocess (stdin/stdout JSON-RPC).
Simulates how Claude Code calls the MCP server.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MCP_SERVER = str(Path(__file__).parent.parent / "mcp_server.py")
PYTHON = sys.executable

SAMPLE_TS = '''
export class Calculator {
    add(a: number, b: number): number {
        return this.validate(a) + this.validate(b);
    }
    validate(n: number): number {
        return isNaN(n) ? 0 : n;
    }
}

export function multiply(x: number, y: number): number {
    return x * y;
}
'''

SAMPLE_PY = '''
class Processor:
    def run(self, items):
        return [self.process(i) for i in items]

    def process(self, item):
        return item * 2

def main():
    p = Processor()
    p.run([1, 2, 3])
'''


def send_mcp_request(proc, request: dict) -> dict:
    """Send a single JSON-RPC request and read the response."""
    line = json.dumps(request) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()
    raw = proc.stdout.readline()
    return json.loads(raw.decode())


def mcp_initialize(proc) -> dict:
    return send_mcp_request(proc, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    })


def mcp_call_tool(proc, req_id: int, tool_name: str, arguments: dict) -> dict:
    return send_mcp_request(proc, {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    })


def mcp_list_tools(proc, req_id: int) -> dict:
    return send_mcp_request(proc, {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/list",
        "params": {},
    })


@pytest.fixture(scope="module")
def mcp_proc():
    """Start MCP server process, yield, then terminate."""
    proc = subprocess.Popen(
        [PYTHON, MCP_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Initialize
    resp = mcp_initialize(proc)
    assert "result" in resp, f"Init failed: {resp}"
    yield proc
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def sample_repo(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("repo")
    (tmp / "calc.ts").write_text(SAMPLE_TS)
    (tmp / "processor.py").write_text(SAMPLE_PY)
    return tmp


def test_mcp_list_tools(mcp_proc):
    resp = mcp_list_tools(mcp_proc, req_id=10)
    assert "result" in resp, f"Expected result, got: {resp}"
    tools = resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "index_code" in tool_names
    assert "search_code" in tool_names
    assert "index_document" in tool_names
    assert "search" in tool_names
    assert "list_documents" in tool_names


def test_index_code_valid_dir(mcp_proc, sample_repo):
    resp = mcp_call_tool(mcp_proc, req_id=20, tool_name="index_code", arguments={
        "dir_path": str(sample_repo),
        "repo_name": "mcp-test-repo",
    })
    assert "result" in resp, f"Unexpected: {resp}"
    content = resp["result"]["content"]
    assert len(content) == 1
    text = content[0]["text"]
    assert "mcp-test-repo" in text
    assert "files" in text or "symbols" in text


def test_index_code_invalid_dir(mcp_proc):
    resp = mcp_call_tool(mcp_proc, req_id=21, tool_name="index_code", arguments={
        "dir_path": "/nonexistent/path/that/does/not/exist",
    })
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    assert "not found" in text.lower() or "error" in text.lower()


def test_search_code_definition_mode(mcp_proc, sample_repo):
    # Ensure indexed first (index_code test runs before, but be safe)
    mcp_call_tool(mcp_proc, req_id=29, tool_name="index_code", arguments={
        "dir_path": str(sample_repo),
        "repo_name": "mcp-test-repo",
    })
    resp = mcp_call_tool(mcp_proc, req_id=30, tool_name="search_code", arguments={
        "query": "multiply",
        "mode": "definition",
        "repo_names": ["mcp-test-repo"],
    })
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    assert "multiply" in text


def test_search_code_impact_mode(mcp_proc, sample_repo):
    resp = mcp_call_tool(mcp_proc, req_id=31, tool_name="search_code", arguments={
        "query": "process",
        "mode": "impact",
        "repo_names": ["mcp-test-repo"],
    })
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    assert "Impact analysis" in text or "process" in text.lower()


def test_search_code_missing_repo(mcp_proc):
    resp = mcp_call_tool(mcp_proc, req_id=32, tool_name="search_code", arguments={
        "query": "anything",
        "mode": "definition",
        "repo_names": ["repo-that-does-not-exist"],
    })
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    # Should return graceful message, not crash
    assert len(text) > 0


def test_search_code_relevance_mode_fallback(mcp_proc, sample_repo):
    """relevance mode with LLM may fail — should fall back gracefully."""
    resp = mcp_call_tool(mcp_proc, req_id=33, tool_name="search_code", arguments={
        "query": "calculator add numbers",
        "mode": "relevance",
        "repo_names": ["mcp-test-repo"],
        "top_k": 3,
    })
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    # Either LLM worked or fallback ran — should never be an exception traceback
    assert "Traceback" not in text
    assert len(text) > 0


def test_list_documents(mcp_proc):
    resp = mcp_call_tool(mcp_proc, req_id=40, tool_name="list_documents", arguments={})
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    assert len(text) > 0


def test_response_json_schema(mcp_proc, sample_repo):
    """All tool responses must follow JSON-RPC 2.0 schema."""
    resp = mcp_call_tool(mcp_proc, req_id=50, tool_name="index_code", arguments={
        "dir_path": str(sample_repo),
        "repo_name": "schema-test",
    })
    assert resp.get("jsonrpc") == "2.0"
    assert "id" in resp
    assert "result" in resp
    result = resp["result"]
    assert "content" in result
    for item in result["content"]:
        assert item.get("type") == "text"
        assert isinstance(item.get("text"), str)
