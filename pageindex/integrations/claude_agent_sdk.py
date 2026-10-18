"""Claude Agent SDK adapter: one value for the mcp_servers slot.

An in-process SDK MCP server in both modes. Cloud clients get a relay that
answers the CLI's MCP session with the PageIndex MCP server's own (its
handshake and instructions, tools, resources), fetched live over this
SDK's MCP client, so the API key stays off the CLI's command line (where
a remote server config would carry it, readable by any local process);
include_management picks the endpoint (the read-only
``?tools=read`` URL by default). Local clients get a server over the same
tool contract, gated the same way at registration, its results passed
through as the MCP content they are.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from .._version import sdk_version
from ..errors import PageIndexAPIError

# Claude Code shows the model only a 2KB preview of an MCP result over
# 50,000 chars; PageIndex results run to ~100,000. 500,000 is the CLI's
# ceiling.
MAX_RESULT_CHARS = 500_000


def build_claude_mcp(client, include_management: bool = False, *,
                     server_name: str = "pageindex"):
    if getattr(client, "api_key", None):
        return _cloud_relay(client, include_management, server_name)

    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_claude_mcp in local mode requires the Claude Agent SDK — "
            "pip install claude-agent-sdk (or pip install 'pageindex[claude]')."
        ) from exc
    from ..agent_tools import TOOL_CONTRACT, _tool_specs

    def make_handler(invoke):
        async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            blocks, is_error = await asyncio.to_thread(invoke, arguments or {})
            result: dict[str, Any] = {"content": blocks}
            if is_error:
                result["is_error"] = True
            return result
        return handler

    def tool_kwargs(name: str) -> dict:
        try:
            from claude_agent_sdk import ToolAnnotations
        except ImportError:
            return {}
        return {"annotations": ToolAnnotations(
            **(TOOL_CONTRACT[name].get("annotations") or {}),
            maxResultSizeChars=MAX_RESULT_CHARS)}

    tools = [
        tool(name, description, schema,
             **tool_kwargs(name))(make_handler(invoke))
        for name, description, schema, invoke
        in _tool_specs(client, include_management)
    ]
    return create_sdk_mcp_server(name=server_name, version=sdk_version(),
                                 tools=tools)


def _cloud_relay(client, include_management: bool, server_name: str):
    from importlib.metadata import version
    try:
        release = tuple(int(n) for n in
                        re.findall(r"\d+", version("claude-agent-sdk"))[:3])
    except ImportError:  # not installed
        release = ()
    if release < (0, 2, 160):
        # older releases answer the CLI from a server's registered
        # handlers, which the relay lacks (before 0.2.140), or close the
        # CLI's stdin at the first result (before 0.2.160): the agent
        # would quietly run without the tools
        raise PageIndexAPIError(
            "as_claude_mcp on cloud requires claude-agent-sdk 0.2.160 or "
            "newer — pip install -U claude-agent-sdk.")
    import anyio.to_thread
    import mcp.types
    from mcp.server.lowlevel import Server
    from ..mcp_bridge import McpBridge

    # include_management picks the endpoint — the URL itself is the gate
    # (?tools=read serves only readOnlyHint-annotated tools).
    url = f"{client.BASE_URL}/mcp" + ("" if include_management else "?tools=read")
    headers = {"Authorization": f"Bearer {client.api_key}"}

    class Relay(Server):
        async def run(self, read_stream, write_stream, *args, **kwargs):
            bridge = McpBridge(url, headers)  # a cloud session per CLI session

            async def answer(request):
                reply = {"jsonrpc": "2.0", "id": request.id}
                try:
                    result = await anyio.to_thread.run_sync(
                        bridge.relay, request.method, request.params,
                        abandon_on_cancel=True)
                    outgoing = _session_message({**reply, "result": result or {}})
                except Exception as exc:  # one bad call must not end the session
                    # the server's own error, which Claude Code may act on
                    error = getattr(exc, "rpc_error", None)
                    if not (isinstance(error, dict) and isinstance(error.get("code"), int)
                            and isinstance(error.get("message"), str)):
                        error = {"code": -32603,  # JSON-RPC internal error
                                 "message": str(exc) or repr(exc)}
                    outgoing = _session_message({**reply, "error": error})
                await write_stream.send(outgoing)

            async with anyio.create_task_group() as tg:
                async for item in read_stream:
                    message = getattr(item, "message", None)
                    message = getattr(message, "root", message)  # mcp 1.x wraps it
                    # notifications stay here: the cloud session has its own
                    if isinstance(message, mcp.types.JSONRPCRequest):
                        tg.start_soon(answer, message)
                tg.cancel_scope.cancel()

    return {"type": "sdk", "name": server_name, "instance": Relay(server_name)}


def _session_message(payload: dict):
    import mcp.types
    from mcp.shared.message import SessionMessage
    adapter = getattr(mcp.types, "jsonrpc_message_adapter", None)  # mcp 2.x
    return SessionMessage(adapter.validate_python(payload) if adapter
                          else mcp.types.JSONRPCMessage.model_validate(payload))
