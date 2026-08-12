"""OpenAI Agents SDK adapter for the Agent(tools=...) slot.

Cloud clients default to the full live tool set as plain FunctionTools via
the MCP bridge; pass hosted=True to use a single HostedMCPTool instead
(the model connects to the PageIndex cloud MCP server from OpenAI's side).
Local clients get the in-process tools wrapped as FunctionTools. Tools are
built as FunctionTool directly so the contract/server JSON schema goes to
the model verbatim — function_tool() would regenerate it from a Python
signature, dropping items/enum/pattern/bounds and rejecting object-typed
parameters.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..errors import PageIndexAPIError


def build_openai_tools(client, include_management: bool = False,
                       hosted: bool = False, doc_ids=None) -> list:
    try:
        from agents import FunctionTool, HostedMCPTool
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_openai_tools requires the OpenAI Agents SDK — "
            "pip install openai-agents (or pip install 'pageindex[openai]')."
        ) from exc
    if getattr(client, "api_key", None) and hosted:
        # Same gate as the in-process path, enforced by OpenAI: tools the
        # server annotates read-only run freely, everything else goes
        # through the Responses API approval flow.
        require_approval = ("never" if include_management
                            else {"never": {"read_only": True}})
        return [HostedMCPTool(tool_config={
            "type": "mcp",
            "server_label": "pageindex",
            "server_url": f"{client.BASE_URL}/mcp",
            "headers": {"Authorization": f"Bearer {client.api_key}"},
            "require_approval": require_approval,
        })]
    from ..agent_tools import _tool_specs

    def wrap(name, description, schema, invoke):
        async def on_invoke_tool(ctx: Any, args_json: str) -> str:
            arguments = {key: value for key, value
                         in (json.loads(args_json) if args_json else {}).items()
                         if value is not None}
            text, _ = await asyncio.to_thread(invoke, arguments)
            return text

        return FunctionTool(name=name, description=description,
                            params_json_schema=schema,
                            on_invoke_tool=on_invoke_tool,
                            strict_json_schema=False)

    return [wrap(*spec)
            for spec in _tool_specs(client, include_management,
                                    doc_ids=doc_ids)]
