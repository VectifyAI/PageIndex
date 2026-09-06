"""Minimal MCP client (streamable HTTP) for the PageIndex cloud MCP server.

Backs the cloud branches of ``client.agent_tools()`` and
``client.agent_instructions()``: ``tools/list`` discovers the live tool set,
``tools/call`` executes a tool, and the ``initialize`` handshake carries the
server's agent instructions. Synchronous, requests-only.
Works against both stateful and stateless servers: a session id returned by
``initialize`` is echoed back, and a session-carrying request rejected with
HTTP 404 (the spec's expired-session status) re-initializes once and
retries; a 400 is an ordinary bad request and is never replayed.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

import requests

from ._version import sdk_version
from .errors import PageIndexAPIError

_PROTOCOL_VERSION = "2025-06-18"
_TIMEOUT = (10, 240)  # tools may wait server-side (wait_for_completion: 3 min)


def _parse_sse(text: str) -> list[dict]:
    """JSON-RPC messages out of a text/event-stream body."""
    messages = []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in text.split("\n\n"):
        data_lines = [line[5:].removeprefix(" ") for line in block.splitlines()
                      if line.startswith("data:")]
        if not data_lines:
            continue
        try:
            messages.append(json.loads("\n".join(data_lines)))
        except ValueError:
            continue
    return messages


class McpBridge:
    def __init__(self, url: str, headers: dict[str, str]):
        self._url = url
        self._auth_headers = dict(headers)
        self._session = requests.Session()  # agent tool calls come in bursts
        self._session_id: Optional[str] = None
        self._protocol_version: Optional[str] = None
        self._instructions: Optional[str] = None
        self._initialized = False
        self._lock = threading.RLock()
        self._next_id = 0

    # ── JSON-RPC over streamable HTTP ──

    def _post(self, payload: dict, session_id: Optional[str] = None,
              protocol_version: Optional[str] = None) -> requests.Response:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._auth_headers,
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        if protocol_version:
            headers["MCP-Protocol-Version"] = protocol_version
        try:
            return self._session.post(self._url, json=payload,
                                      headers=headers, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise PageIndexAPIError(
                f"Could not reach the PageIndex MCP server: {exc}"
            ) from exc

    def _extract_result(self, response: requests.Response, request_id: int) -> Any:
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            # SSE is UTF-8 by spec; requests guesses latin-1 for charset-less
            # text/* and would mojibake every non-ASCII character.
            messages = _parse_sse(response.content.decode("utf-8",
                                                          errors="replace"))
        else:
            try:
                messages = [response.json()]
            except ValueError as exc:
                raise PageIndexAPIError(
                    f"MCP server returned a non-JSON response "
                    f"(HTTP {response.status_code}).",
                    status_code=response.status_code,
                ) from exc
        # Strict id correlation only — accepting any result-bearing message
        # would return a stale or mis-correlated reply as this call's.
        reply = next((m for m in messages
                      if isinstance(m, dict) and m.get("id") == request_id),
                     None)
        if reply is None:
            raise PageIndexAPIError(
                "MCP server response contained no reply matching the request."
            )
        if "error" in reply:
            error = reply["error"] or {}
            raise PageIndexAPIError(
                f"MCP error {error.get('code')}: {error.get('message')}"
            )
        return reply.get("result")

    def _request(self, method: str, params: Optional[dict] = None,
                 _retry: bool = True) -> Any:
        self._ensure_initialized()
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            session_id = self._session_id
            protocol_version = self._protocol_version
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id,
                                   "method": method}
        if params is not None:
            payload["params"] = params
        response = self._post(payload, session_id, protocol_version)
        if response.status_code == 404 and session_id and _retry:
            # Session expired (stateful servers; the spec's 404): the server
            # refused the request at session validation, so replaying it is
            # safe. 400 is an ordinary bad request — replaying one would
            # re-run side effects. Reset only if no other thread has already
            # re-initialized, then retry once on the fresh session.
            with self._lock:
                if self._session_id == session_id:
                    self._initialized = False
                    self._session_id = None
                    self._protocol_version = None
            return self._request(method, params, _retry=False)
        if response.status_code >= 400:
            raise PageIndexAPIError(
                f"MCP request failed: HTTP {response.status_code} "
                f"({response.text[:200]})",
                status_code=response.status_code,
            )
        return self._extract_result(response, request_id)

    def _ensure_initialized(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._next_id += 1
            request_id = self._next_id
            response = self._post({
                "jsonrpc": "2.0", "id": request_id, "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "pageindex-python-sdk",
                                   "version": sdk_version()},
                },
            })
            if response.status_code >= 400:
                raise PageIndexAPIError(
                    f"Could not connect to the PageIndex MCP server: HTTP "
                    f"{response.status_code} ({response.text[:200]}). Check "
                    "your API key.",
                    status_code=response.status_code,
                )
            result = self._extract_result(response, request_id) or {}
            self._session_id = response.headers.get("Mcp-Session-Id")
            self._protocol_version = result.get("protocolVersion",
                                                _PROTOCOL_VERSION)
            self._instructions = result.get("instructions")
            self._initialized = True
            # Sent inside the lock so no concurrent thread can slip a
            # request between the handshake and this notification.
            try:
                self._post({"jsonrpc": "2.0",
                            "method": "notifications/initialized"},
                           self._session_id, self._protocol_version)
            except PageIndexAPIError:
                pass  # advisory; a server that required it fails the next request

    # ── public surface ──

    def instructions(self) -> Optional[str]:
        """The server's agent instructions from the initialize handshake."""
        self._ensure_initialized()
        return self._instructions

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor: Optional[str] = None
        # A server echoing its cursor (or cycling) must not hang the client:
        # no-progress terminates, the page cap turns a cycle into an error.
        for _ in range(50):
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params) or {}
            tools.extend(result.get("tools") or [])
            next_cursor = result.get("nextCursor")
            if not next_cursor or next_cursor == cursor:
                return tools
            cursor = next_cursor
        raise PageIndexAPIError(
            "MCP tools/list pagination did not terminate within 50 pages.")

    def call_tool(self, name: str, arguments: dict[str, Any],
                  ) -> "tuple[list[dict], bool]":
        """Returns (content, is_error): the MCP content blocks untouched,
        for each adapter to render what its framework carries (render_text
        is the text-only rendering), and the server's isError marking,
        which callers must carry to their framework's own error channel."""
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments}) or {}
        return list(result.get("content") or []), bool(result.get("isError"))


def render_text(blocks: list) -> str:
    """The text-only rendering of MCP content: text verbatim, base64
    payloads as a size stub (dumped whole they hand the model the raw
    blob)."""
    texts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif isinstance(block, dict) and isinstance(block.get("data"), str):
            kind = block.get("mimeType") or block.get("type") or "binary"
            size_kb = max(1, len(block["data"]) * 3 // 4096)
            texts.append(f"[{kind} content omitted: ~{size_kb} KB]")
        elif (isinstance(block, dict)
              and isinstance(block.get("resource"), dict)
              and isinstance(block["resource"].get("blob"), str)):
            # EmbeddedResource nests its base64 one level down.
            resource = block["resource"]
            kind = resource.get("mimeType") or "binary"
            size_kb = max(1, len(resource["blob"]) * 3 // 4096)
            texts.append(f"[{kind} content omitted: ~{size_kb} KB]")
        else:
            texts.append(json.dumps(block, ensure_ascii=False))
    return "\n".join(texts)
