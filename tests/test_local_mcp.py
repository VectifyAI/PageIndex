"""Local MCP protocol tests against a seeded store, without LLM calls."""
import asyncio
import json
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_client_server_memory_streams
from test_agent_tools import seed_doc

from pageindex import PageIndexCloudClient, PageIndexLocalClient
from pageindex.agent_tools import _local_schema, tool_names
from pageindex.local_mcp_server import LocalMcpServer


@pytest.fixture
def local_client(tmp_path):
    storage = str(tmp_path / "store")
    seed_doc(storage, "pi-a", "report.pdf")
    return PageIndexLocalClient(storage_path=storage)


def wire(result):
    """MCP 1.x and 2.x use different Python names but the same wire aliases."""
    return result.model_dump(by_alias=True, exclude_none=True)


def payload(result):
    return json.loads(wire(result)["content"][0]["text"])


async def round_trip(server, assertions):
    """Exercise real initialization and dispatch with a bounded lifetime."""
    with anyio.fail_after(10):
        async with create_client_server_memory_streams() as (client_streams, server_streams), \
            anyio.create_task_group() as tasks:
                tasks.start_soon(
                    server.run, *server_streams,
                    server.create_initialization_options(),
                )
                async with ClientSession(*client_streams) as session:
                    initialized = await session.initialize()
                    await assertions(session, initialized)
                tasks.cancel_scope.cancel()


def test_protocol_discovery_and_document_reads(local_client):
    async def assertions(session, initialized):
        assert initialized.instructions == local_client.agent_instructions()
        assert initialized.capabilities.tools is not None
        catalog = await session.list_tools()
        assert [tool.name for tool in catalog.tools] == list(tool_names())
        for tool in catalog.tools:
            definition = wire(tool)
            assert definition["inputSchema"] == _local_schema(tool.name)
            assert definition["annotations"]["readOnlyHint"] is True

        browsed = await session.call_tool("browse_documents", {})
        assert not wire(browsed)["isError"]
        assert payload(browsed)["documents"][0]["name"] == "report.pdf"

        for name, arguments in [
            ("get_document", {"doc_name": "report.pdf"}),
            ("get_document_structure", {"doc_name": "report.pdf"}),
            ("get_page_content", {"doc_name": "report.pdf", "pages": "1-2"}),
        ]:
            result = await session.call_tool(name, arguments)
            assert not wire(result)["isError"]
            assert payload(result)["success"] is True
            if name == "get_page_content":
                assert "apples" in wire(result)["content"][0]["text"]
                assert "bananas" in wire(result)["content"][0]["text"]

    asyncio.run(round_trip(LocalMcpServer(local_client), assertions))


def test_protocol_errors_and_management_gate(local_client):
    async def assertions(session, initialized):
        missing = await session.call_tool("get_document", {"doc_name": "missing.pdf"})
        assert wire(missing)["isError"] is True
        assert payload(missing)["errorCode"] == "NOT_FOUND"

        for name, arguments in [
            ("unknown_tool", {}),
            ("remove_document", {"doc_names": ["report.pdf"]}),
            ("get_page_content", {}),
        ]:
            result = await session.call_tool(name, arguments)
            assert wire(result)["isError"] is True

        # Calling a hidden management tool must not delete anything.
        assert local_client.get_document("pi-a")["name"] == "report.pdf"

    asyncio.run(round_trip(LocalMcpServer(local_client), assertions))


def test_management_opt_in_allows_deletion(local_client):
    async def assertions(session, initialized):
        catalog = await session.list_tools()
        assert [tool.name for tool in catalog.tools] == list(tool_names(True))
        result = await session.call_tool("remove_document", {"doc_names": ["report.pdf"]})
        assert not wire(result)["isError"]
        assert payload(result)["success"] is True
        browsed = await session.call_tool("browse_documents", {})
        assert payload(browsed)["documents"] == []

    asyncio.run(round_trip(LocalMcpServer(local_client, True), assertions))


def test_string_booleans_reach_the_tool_layer(local_client):
    """Models often send booleans as strings; the tool layer coerces them, as
    on cloud. SDK-side schema validation must not reject them first."""
    async def assertions(session, initialized):
        result = await session.call_tool(
            "get_document",
            {"doc_name": "report.pdf", "wait_for_completion": "false"},
        )
        assert not wire(result)["isError"], wire(result)["content"][0]["text"]
        assert payload(result)["success"] is True

    asyncio.run(round_trip(LocalMcpServer(local_client), assertions))


def test_cloud_client_is_rejected():
    with pytest.raises(TypeError, match="PageIndexLocalClient"):
        LocalMcpServer(PageIndexCloudClient(api_key="test-key"))


def test_invoker_runs_off_event_loop(local_client):
    import threading

    server = LocalMcpServer(local_client)
    main_thread = threading.get_ident()
    worker_threads = []

    def invoke(arguments):
        worker_threads.append(threading.get_ident())
        return [{"type": "text", "text": "ok"}], False

    server.invokers["browse_documents"] = invoke
    result = asyncio.run(server.call_tool(
        None, types.CallToolRequestParams(name="browse_documents", arguments={}),
    ))
    assert wire(result)["content"][0]["text"] == "ok"
    assert len(worker_threads) == 1 and worker_threads[0] != main_thread


def test_stdio_subprocess_round_trip(local_client):
    """Catch stdout contamination and startup/shutdown issues over real pipes."""
    source = """
import asyncio
import sys
from pageindex import PageIndexLocalClient
from pageindex.local_mcp_server import LocalMcpServer

async def main():
    server = LocalMcpServer(PageIndexLocalClient(storage_path=sys.argv[1]))
    await server.serve_stdio()

asyncio.run(main())
"""

    async def check():
        with anyio.fail_after(15):
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-c", source, str(local_client.storage_path)],
            )
            async with stdio_client(parameters) as streams, \
                ClientSession(*streams) as session:
                    await session.initialize()
                    assert [tool.name for tool in (await session.list_tools()).tools] == list(tool_names())
                    result = await session.call_tool("browse_documents", {})
                    assert payload(result)["documents"][0]["name"] == "report.pdf"

    asyncio.run(check())


def test_stdio_survives_stray_stdout_from_a_tool(local_client):
    """A tool that writes to stdout mid-call must not corrupt the JSON-RPC
    stream; no trailing newline, so the noise would fuse with the reply."""
    source = """
import asyncio
import sys
from pageindex import PageIndexLocalClient
from pageindex.local_mcp_server import LocalMcpServer

async def main():
    server = LocalMcpServer(PageIndexLocalClient(storage_path=sys.argv[1]))
    browse = server.invokers["browse_documents"]

    def noisy(arguments):
        print("stray tool output", end="", flush=True)
        return browse(arguments)

    server.invokers["browse_documents"] = noisy
    await server.serve_stdio()

asyncio.run(main())
"""

    async def check():
        with anyio.fail_after(15):
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-c", source, str(local_client.storage_path)],
            )
            async with stdio_client(parameters) as streams, \
                ClientSession(*streams) as session:
                    await session.initialize()
                    for _ in range(2):  # the second reply would carry the noise
                        result = await session.call_tool("browse_documents", {})
                        assert payload(result)["documents"][0]["name"] == "report.pdf"

    asyncio.run(check())
