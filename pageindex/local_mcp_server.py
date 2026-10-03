"""MCP tools over an existing local PageIndex document store.

The server reuses the SDK's tool contract and dispatchers. It does not index
documents or run a chat model; an MCP host uses the tools to read documents
that were previously indexed with a PageIndexLocalClient.
"""

import asyncio
import io
import os
import sys

import anyio
from mcp import types
from mcp.server.lowlevel import Server

from ._version import sdk_version
from .agent_tools import TOOL_CONTRACT, _tool_specs


class LocalMcpServer(Server):
    """Serve local document tools through MCP 1.x or 2.x.

    Args:
        client: A PageIndexLocalClient connected to the indexed document store.
        include_management: Expose and allow remove_document when True.
            The default tool set is read-only.

    Constructing this object does not start a transport: await
    ``serve_stdio()``, or run the ``pageindex-mcp`` command.
    """

    def __init__(self, client, include_management: bool = False):
        from .client import PageIndexLocalClient

        if not isinstance(client, PageIndexLocalClient):
            raise TypeError("LocalMcpServer requires a PageIndexLocalClient")
        self.specs = _tool_specs(client, include_management)
        self.invokers = {
            name: invoke
            for name, _, _, invoke in self.specs
        }

        options = {
            "version": sdk_version(),
            "instructions": client.agent_instructions(
                include_management=include_management),
        }
        # MCP 1.x registers decorators on an instance; 2.x accepts callbacks.
        if hasattr(Server, "list_tools"):
            super().__init__("pageindex-local-mcp", **options)

            async def list_handler():
                return await self.list_tools(None, None)

            async def call_handler(name, arguments):
                params = types.CallToolRequestParams(name=name, arguments=arguments)
                return await self.call_tool(None, params)

            Server.list_tools(self)(list_handler)
            Server.call_tool(self, validate_input=False)(call_handler)
        else:
            super().__init__(
                "pageindex-local-mcp", **options,
                on_list_tools=self.list_tools,
                on_call_tool=self.call_tool,
            )

    async def list_tools(self, context, params):
        """Return registered local schemas and their MCP safety annotations.

        ``context`` and pagination ``params`` are supplied by MCP 2.x. The
        fixed local tool catalog fits in one response, so neither is needed.
        """
        return types.ListToolsResult(tools=[
            types.Tool(
                name=name,
                description=description,
                inputSchema=schema,
                annotations=types.ToolAnnotations(
                    **TOOL_CONTRACT[name].get("annotations", {})),
            )
            for name, description, schema, _ in self.specs
        ])

    async def call_tool(self, context, params):
        """Dispatch a registered tool off the event loop and preserve errors.

        ``params`` carries the tool name and arguments. Only registered
        invokers can execute, so a management tool remains disabled even if
        a client calls its name directly. ``context`` is unused.
        """
        invoke = self.invokers.get(params.name)
        if invoke is None:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text", text=f"Unknown or disabled tool: {params.name}")],
                isError=True
            )

        blocks, is_error = await asyncio.to_thread(invoke, params.arguments or {})
        return types.CallToolResult.model_validate(
            {"content": blocks, "isError": is_error})

    async def serve_stdio(self):
        """Serve over stdin/stdout until the host closes the pipe.

        While serving, fd 1 points at stderr and the protocol writes to a
        private duplicate of the original stdout, so stray output from tools,
        C extensions or child processes never corrupts the JSON-RPC stream.
        MCP 2.x's stdio_server diverts stdout itself; 1.x does not.
        """
        from mcp.server.stdio import stdio_server

        sys.stdout.flush()
        protocol_fd = os.dup(1)
        os.dup2(2, 1)
        protocol_out = io.TextIOWrapper(
            os.fdopen(protocol_fd, "wb", closefd=False), encoding="utf-8")
        try:
            async with stdio_server(stdout=anyio.wrap_file(protocol_out)) as (
                    read_stream, write_stream):
                await self.run(read_stream, write_stream,
                               self.create_initialization_options())
        finally:
            sys.stdout.flush()
            try:
                protocol_out.flush()
            except (OSError, ValueError):
                pass  # host already closed the pipe
            os.dup2(protocol_fd, 1)
            os.close(protocol_fd)


def main():
    """Launch the local stdio server from the ``pageindex-mcp`` command."""
    from argparse import ArgumentParser

    from .client import PageIndexLocalClient

    parser = ArgumentParser(description="Serve a local PageIndex document store over MCP stdio.")
    parser.add_argument("--storage-path", required=True,
                        help="Path to an existing indexed document store.")
    parser.add_argument("--management", action="store_true",
                        help="Enable document deletion (disabled by default).")
    args = parser.parse_args()
    if not os.path.isdir(args.storage_path):
        # A typo would otherwise serve an empty library without complaint.
        parser.error(f"--storage-path {args.storage_path!r} is not a directory")

    client = PageIndexLocalClient(storage_path=args.storage_path)
    mcp_server = LocalMcpServer(client, include_management=args.management)
    asyncio.run(mcp_server.serve_stdio())
