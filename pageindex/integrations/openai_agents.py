"""OpenAI Agents SDK adapter for the Agent(tools=...) slot.

Cloud clients default to the live read tool set as plain FunctionTools via
the MCP bridge; pass hosted=True to use a single HostedMCPTool instead
(the model connects to the PageIndex cloud MCP server from OpenAI's side —
the read-only ``?tools=read`` endpoint by default). Local clients get the
in-process tools wrapped as FunctionTools. Tools are built as FunctionTool
directly so the contract/server JSON schema goes to the model verbatim —
function_tool() would regenerate it from a Python signature, dropping
items/enum/pattern/bounds and rejecting object-typed parameters.
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
    from ..agent_tools import _require_local_scope, _tool_specs
    # The hosted branch returns before _tool_specs — reject cloud doc_ids
    # here so they are never silently dropped.
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
