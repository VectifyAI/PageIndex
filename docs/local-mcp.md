# Local MCP server

`pageindex.local_mcp_server.LocalMcpServer` exposes an indexed local document store to
MCP clients. It reuses the SDK's tool descriptions, schemas, implementations,
and agent instructions. Both MCP Python SDK 1.x and 2.x are supported within
the package's declared dependency range.

The default tools are `browse_documents`, `get_document`,
`get_document_structure`, and `get_page_content`. Pass
`include_management=True` to also enable `remove_document`. A direct call to
a disabled management tool returns an MCP tool error and does not delete data.

## Prepare a document store

Install your source checkout in a virtual environment:

```bash
python -m pip install -e .
```

Index documents through `PageIndexLocalClient` before starting the server:

```python
from pageindex import PageIndexLocalClient

client = PageIndexLocalClient(storage_path="/absolute/path/to/index-store")
result = client.submit_document("report.pdf")
print(result["doc_id"])
```

Normal indexing uses an LLM and requires the configured provider's credentials
(for example, `OPENAI_API_KEY`). The storage path points to the SDK's document
store, not a directory of unindexed PDFs or a standalone structure JSON file.

Serving existing document content does not require an LLM call. The MCP host
supplies the model that chooses tools and answers questions. Content returned
to the host may be sent to that host's model provider.

## Run over stdio

The class does not start a transport automatically. There is currently no
`pageindex-mcp` console command. Save this launcher as `serve_local_mcp_server.py`
outside the package or in your application:

```python
"""Serve an existing PageIndex document store over MCP stdio."""
import asyncio
import sys

from pageindex import PageIndexLocalClient
from pageindex.local_mcp_server import LocalMcpServer


async def main():
    client = PageIndexLocalClient(storage_path=sys.argv[1])
    await LocalMcpServer(client).serve_stdio()


if __name__ == "__main__":
    asyncio.run(main())
```

Configure an MCP host that accepts `mcpServers` entries to launch it:

```json
{
  "mcpServers": {
    "pageindex-local": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": [
        "/absolute/path/to/serve_local_mcp_server.py",
        "/absolute/path/to/index-store"
      ]
    }
  }
}
```

Use absolute paths because desktop applications may have a different working
directory and PATH. The host starts and stops the subprocess. There is no
HTTP endpoint or listening port. Starting the launcher in a terminal leaves it
waiting for MCP messages rather than displaying an interactive prompt.

`serve_stdio()` points stdout at stderr while serving, so stray output from
tools or libraries cannot corrupt the MCP stream on either MCP SDK version.
Send application logs to stderr. The
server accepts only `PageIndexLocalClient`; a cloud client is rejected.

## Development checks

```bash
python -m pip install pytest
python -m pytest tests/test_local_mcp_server.py -q
```

The tests initialize real MCP sessions over in-memory streams and subprocess
stdio. They verify tool discovery, schemas and annotations, document reads,
error flags, management gating, and execution outside the event-loop thread.
Fixtures seed a temporary store directly, so no provider credentials or live
LLM requests are needed.

The console entry point and upload/indexing tools remain separate work. This
server provides tools only; it does not expose MCP resources or prompts.
