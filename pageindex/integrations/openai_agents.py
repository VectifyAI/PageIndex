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
    from ..agent_tools import (_dumps, _failure, _require_local_scope,
                               _tool_specs)
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
            # strict_json_schema is off, so the provider never validates the
            # payload; a malformed or non-object argument string must come
            # back as the guided error envelope — raising here aborts the
            # caller's whole run (hand-built FunctionTools have no
            # failure_error_function to hand the error back to the model).
            try:
                parsed = json.loads(args_json) if args_json else {}
            except ValueError:
                parsed = None
            if not isinstance(parsed, dict):
                payload, _ = _failure(
                    f"Invalid arguments for {name}: expected a JSON object, "
                    f"got: {(args_json or '')[:200]!r}", None,
                    {"summary": "Malformed tool arguments",
                     "options": [f"Re-send the {name} call with a JSON "
                                 "object of its parameters"]},
                    "INVALID_INPUT")
                return _dumps(payload)
            arguments = {key: value for key, value in parsed.items()
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
