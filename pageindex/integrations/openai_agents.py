"""OpenAI Agents SDK adapter for the Agent(tools=...) slot.

Cloud clients default to the live read tool set via the MCP bridge; pass
hosted=True to use a single HostedMCPTool instead (the model connects to
the PageIndex cloud MCP server from OpenAI's side — the read-only
``?tools=read`` endpoint by default). Local clients get the in-process
tools.

Either way the tool set reaches the framework as an MCP server (an
in-process one over the bridge or the local store), and the FunctionTools
are the framework's own conversion of it: the schema goes to the model
verbatim, and tool results reach it in the framework's shapes (text as
text, images as images). The SDK carries MCP types and renders nothing.
"""
from __future__ import annotations

import asyncio

from ..errors import PageIndexAPIError


def build_mcp_server(client, include_management: bool = False, doc_ids=None):
    """The tool set as an in-process MCP server for the Agents SDK."""
    from agents.mcp import MCPServer
    from mcp import types as mcp_types
    from ..agent_tools import _tool_specs

    specs = _tool_specs(client, include_management, doc_ids)

    class _ToolServer(MCPServer):
        def __init__(self):
            super().__init__()
            self.tools = [mcp_types.Tool(name=name, description=description,
                                         inputSchema=schema)
                          for name, description, schema, _ in specs]
            self._invoke = {name: invoke for name, _, _, invoke in specs}

        @property
        def name(self) -> str:
            return "pageindex"

        async def connect(self):
            pass

        async def cleanup(self):
            pass

        async def list_tools(self, run_context=None, agent=None):
            return self.tools

        async def call_tool(self, tool_name: str, arguments, meta=None):
            blocks, is_error = await asyncio.to_thread(
                self._invoke[tool_name], arguments or {})
            return mcp_types.CallToolResult.model_validate(
                {"content": blocks, "isError": is_error})

        async def list_prompts(self):
            return mcp_types.ListPromptsResult(prompts=[])

        async def get_prompt(self, name: str, arguments=None):
            raise ValueError(f"No prompt named {name!r}")

    return _ToolServer()


def build_openai_tools(client, include_management: bool = False,
                       hosted: bool = False, doc_ids=None) -> list:
    try:
        from agents import HostedMCPTool
        from agents.mcp import MCPUtil
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_openai_tools requires the OpenAI Agents SDK — "
            "pip install openai-agents."
        ) from exc
    from ..agent_tools import _require_local_scope
    _require_local_scope(client, doc_ids)
    if getattr(client, "api_key", None) and hosted:
        # include_management picks the endpoint — the URL itself is the
        # gate (?tools=read serves only readOnlyHint-annotated tools), so
        # nothing needs the Responses API approval flow.
        suffix = "" if include_management else "?tools=read"
        return [HostedMCPTool(tool_config={
            "type": "mcp",
            "server_label": "pageindex",
            "server_url": f"{client.BASE_URL}/mcp{suffix}",
            "headers": {"Authorization": f"Bearer {client.api_key}"},
            "require_approval": "never",
        })]

    server = build_mcp_server(client, include_management, doc_ids)
    return [MCPUtil.to_function_tool(tool, server, False)
            for tool in server.tools]
