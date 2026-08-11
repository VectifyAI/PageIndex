"""Anthropic SDK adapter for the tool runner's tools=... slot.

Cloud clients get one runnable tool per live cloud MCP tool — the server's
input schemas pass through verbatim (MCP inputSchema and Messages API
input_schema are the same shape), calls proxied over MCP. Local clients get
the in-process tools — the same set messages() runs internally.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any, Callable

from ..errors import PageIndexAPIError


def build_anthropic_tools(client, include_management: bool = False,
                          asynchronous: bool = False) -> list:
    try:
        from anthropic import beta_async_tool, beta_tool
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_anthropic_tools requires the Anthropic SDK tool runner "
            "(anthropic>=0.68.0) — pip install -U anthropic (or pip install "
            "'pageindex[anthropic]')."
        ) from exc

    def wrap(name: str, description: str, schema: dict,
             invoke: Callable[[dict], str]):
        """One runnable tool in the caller's flavor: the sync runner and the
        async runner each accept only their own kind, and the async variant
        moves the blocking bridge/store call into a worker thread so it
        never blocks the caller's event loop."""
        if asynchronous:
            async def _afn(**kwargs: Any) -> str:
                return await asyncio.to_thread(invoke, kwargs)

            _afn.__name__ = name
            return beta_async_tool(_afn, name=name, description=description,
                                   input_schema=schema)

        def _fn(**kwargs: Any) -> str:
            return invoke(kwargs)

        _fn.__name__ = name
        return beta_tool(_fn, name=name, description=description,
                         input_schema=schema)

    if getattr(client, "api_key", None):
        from ..agent_tools import (_bridge_invoker, _cloud_bridge,
                                   _read_only_tools)
        bridge = _cloud_bridge(client)
        tools_meta = bridge.list_tools()
        if not include_management:
            tools_meta = _read_only_tools(tools_meta)

        def make_cloud(meta: dict):
            name = str(meta.get("name") or "tool")
            # beta_tool keeps the schema dict by reference — hand out a copy,
            # as _local_schema already does for the local contract.
            schema = (copy.deepcopy(meta.get("inputSchema"))
                      or {"type": "object", "properties": {}})
            return wrap(name, meta.get("description", ""), schema,
                        _bridge_invoker(bridge, name))

        return [make_cloud(meta) for meta in tools_meta]

    from ..agent_tools import (_local_description, _local_schema, call_tool,
                               tool_names)

    def local_invoke(name: str) -> Callable[[dict], str]:
        def invoke(arguments: dict) -> str:
            return call_tool(client, name, arguments)[0]
        return invoke

    return [wrap(name, _local_description(name), _local_schema(name),
                 local_invoke(name))
            for name in tool_names(include_management)]
