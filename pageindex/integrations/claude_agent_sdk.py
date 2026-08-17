"""Claude Agent SDK adapter: one value for the mcp_servers slot.

Cloud clients get the remote PageIndex MCP config — the framework connects
directly, and include_management picks the endpoint (the read-only
``?tools=read`` URL by default); local clients get an in-process SDK MCP
server over the same tool contract, gated the same way at registration.
"""
from __future__ import annotations

import asyncio
from typing import Any

from .._version import sdk_version
from ..errors import PageIndexAPIError


def build_claude_mcp(client, include_management: bool = False, doc_ids=None):
    from ..agent_tools import _require_local_scope
    _require_local_scope(client, doc_ids)
    if getattr(client, "api_key", None):
        # include_management picks the endpoint — the URL itself is the
        # gate (?tools=read serves only readOnlyHint-annotated tools).
        suffix = "" if include_management else "?tools=read"
        return {
            "type": "http",
            "url": f"{client.BASE_URL}/mcp{suffix}",
            "headers": {"Authorization": f"Bearer {client.api_key}"},
        }

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
            text, is_error = await asyncio.to_thread(invoke, arguments or {})
            result: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
            if is_error:
                result["is_error"] = True
            return result
        return handler

    def tool_kwargs(name: str) -> dict:
        annotations = TOOL_CONTRACT[name].get("annotations")
        if not annotations:
            return {}
        try:
            from claude_agent_sdk import ToolAnnotations
        except ImportError:
            return {}
        return {"annotations": ToolAnnotations(**annotations)}

    tools = [
        tool(name, description, schema,
             **tool_kwargs(name))(make_handler(invoke))
        for name, description, schema, invoke
        in _tool_specs(client, include_management, doc_ids)
    ]
    return create_sdk_mcp_server(name="pageindex", version=sdk_version(),
                                 tools=tools)
