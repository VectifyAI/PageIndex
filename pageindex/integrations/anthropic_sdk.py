"""Anthropic SDK adapter for the tool runner's tools=... slot.

Cloud clients get one runnable tool per live cloud MCP tool — the server's
input schemas pass through verbatim (MCP inputSchema and Messages API
input_schema are the same shape), calls proxied over MCP. Local clients get
the in-process tools — the same set messages() runs internally. Failed
calls raise ToolError so the runner emits the tool_result with
``is_error: true`` and the envelope as its content.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..errors import PageIndexAPIError


def build_anthropic_tools(client, include_management: bool = False,
                          asynchronous: bool = False, doc_ids=None) -> list:
    try:
        from anthropic import beta_async_tool, beta_tool
        from anthropic.lib.tools import ToolError
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_anthropic_tools requires the Anthropic SDK tool runner "
            "(anthropic>=0.108.0) — pip install -U anthropic (or pip install "
            "'pageindex[anthropic]')."
        ) from exc
    from ..agent_tools import _tool_specs

    def wrap(name, description, schema, invoke):
        """One runnable tool in the caller's flavor: the sync runner and the
        async runner each accept only their own kind, and the async variant
        moves the blocking bridge/store call into a worker thread so it
        never blocks the caller's event loop."""
        def run(kwargs: dict) -> str:
            text, is_error = invoke(kwargs)
            if is_error:
                raise ToolError(text)
            return text

        if asynchronous:
            async def _afn(**kwargs: Any) -> str:
                return await asyncio.to_thread(run, kwargs)

            _afn.__name__ = name
            return beta_async_tool(_afn, name=name, description=description,
                                   input_schema=schema)

        def _fn(**kwargs: Any) -> str:
            return run(kwargs)

        _fn.__name__ = name
        return beta_tool(_fn, name=name, description=description,
                         input_schema=schema)

    return [wrap(*spec)
            for spec in _tool_specs(client, include_management,
                                    doc_ids=doc_ids)]
