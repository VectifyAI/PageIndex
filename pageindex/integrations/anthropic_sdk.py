"""Anthropic SDK adapter for the tool runner's tools=... slot.

Cloud clients get one runnable tool per live cloud MCP tool — the server's
input schemas pass through verbatim (MCP inputSchema and Messages API
input_schema are the same shape), calls proxied over MCP. Local clients get
the in-process tools — the same set chat(protocol="messages") runs
internally. Failed
calls raise ToolError so the runner emits the tool_result with
``is_error: true`` and the envelope as its content; the failures the
invoker re-raises (auth, limits, unreachable server) propagate as
PageIndexAPIError, which a caller-owned runner flattens into an is_error
result carrying the exception text, or land in ``failures`` when one is
supplied, for chat(protocol="messages") to fail fast on between turns.

Tool results are MCP content, rendered by the Anthropic SDK's own MCP
conversion (text as text, images as image blocks); the SDK carries the
MCP types and renders nothing.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..errors import PageIndexAPIError


def build_anthropic_tools(client, include_management: bool = False,
                          asynchronous: bool = False, doc_ids=None,
                          failures: Optional[list] = None) -> list:
    """``failures`` records the invoker's re-raised failures for
    chat(protocol="messages") to fail fast on."""
    try:
        from anthropic import beta_async_tool, beta_tool
        from anthropic.lib.tools import ToolError
        from anthropic.lib.tools.mcp import mcp_content
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_anthropic_tools requires the Anthropic SDK tool runner "
            "(anthropic>=0.108.0) — pip install -U anthropic (or pip install "
            "'pageindex[anthropic]')."
        ) from exc
    from mcp.types import CallToolResult
    from ..agent_tools import _tool_specs

    def wrap(name, description, schema, invoke):
        """One runnable tool in the caller's flavor: the sync runner and the
        async runner each accept only their own kind, and the async variant
        moves the blocking bridge/store call into a worker thread so it
        never blocks the caller's event loop."""
        def run(kwargs: dict) -> list:
            try:
                blocks, is_error = invoke(kwargs)
            except PageIndexAPIError as exc:
                if failures is None:
                    raise
                failures.append(exc)
                # the runner logs a traceback for anything but ToolError;
                # the lane raises exc itself before the runner advances
                raise ToolError(str(exc)) from exc
            result = CallToolResult.model_validate(
                {"content": blocks, "isError": is_error})
            content = [mcp_content(block) for block in result.content]
            if is_error:
                raise ToolError(content)
            return content

        if asynchronous:
            async def _afn(**kwargs: Any) -> list:
                return await asyncio.to_thread(run, kwargs)

            _afn.__name__ = name
            return beta_async_tool(_afn, name=name, description=description,
                                   input_schema=schema)

        def _fn(**kwargs: Any) -> list:
            return run(kwargs)

        _fn.__name__ = name
        return beta_tool(_fn, name=name, description=description,
                         input_schema=schema)

    return [wrap(*spec)
            for spec in _tool_specs(client, include_management,
                                    doc_ids=doc_ids)]
