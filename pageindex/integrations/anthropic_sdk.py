"""Anthropic SDK adapter for the tool runner's tools=... slot.

Cloud clients get one runnable tool per live cloud MCP tool — the server's
input schemas pass through verbatim (MCP inputSchema and Messages API
input_schema are the same shape), calls proxied over MCP. Local clients get
the in-process tools — the same set messages() runs internally.
"""
from __future__ import annotations

from typing import Any

from ..errors import PageIndexAPIError


def build_anthropic_tools(client, include_management: bool = False) -> list:
    try:
        from anthropic import beta_tool
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_anthropic_tools requires the Anthropic SDK tool runner "
            "(anthropic>=0.68.0) — pip install -U anthropic (or pip install "
            "'pageindex[anthropic]')."
        ) from exc

    if getattr(client, "api_key", None):
        from ..agent_tools import (_bridge_invoker, _cloud_bridge,
                                   _read_only_tools)
        bridge = _cloud_bridge(client)
        tools_meta = bridge.list_tools()
        if not include_management:
            tools_meta = _read_only_tools(tools_meta)

        def make_cloud(meta: dict):
            name = str(meta.get("name") or "tool")
            invoke = _bridge_invoker(bridge, name)

            def _fn(**kwargs: Any) -> str:
                return invoke(kwargs)

            _fn.__name__ = name
            return beta_tool(
                _fn, name=name, description=meta.get("description", ""),
                input_schema=meta.get("inputSchema")
                or {"type": "object", "properties": {}},
            )

        return [make_cloud(meta) for meta in tools_meta]

    from ..agent_tools import (_local_description, _local_schema, call_tool,
                               tool_names)

    def make_local(name: str):
        def _fn(**kwargs: Any) -> str:
            return call_tool(client, name, kwargs)[0]

        _fn.__name__ = name
        return beta_tool(_fn, name=name, description=_local_description(name),
                         input_schema=_local_schema(name))

    return [make_local(name) for name in tool_names(include_management)]
