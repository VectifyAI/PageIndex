"""Minimal MCP client (streamable HTTP) for the PageIndex cloud MCP server.

Backs the cloud branch of ``client.agent_tools()``: ``tools/list`` discovers
the live tool set, ``tools/call`` executes a tool. Synchronous, requests-only.
Works against both stateful and stateless servers: a session id returned by
``initialize`` is echoed back, and a request rejected after session expiry
re-initializes once and retries.
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
        self._session_id: Optional[str] = None
        self._protocol_version: Optional[str] = None
        self._initialized = False
        self._lock = threading.RLock()
        self._next_id = 0

    # ── JSON-RPC over streamable HTTP ──

    def _post(self, payload: dict) -> requests.Response:
        with self._lock:
            session_id = self._session_id
            protocol_version = self._protocol_version
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
            return requests.post(self._url, json=payload, headers=headers,
                                 timeout=_TIMEOUT)
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
                    f"(HTTP {response.status_code})."
                ) from exc
        reply = next((m for m in messages if m.get("id") == request_id),
                     next((m for m in messages
                           if "result" in m or "error" in m), None))
        if reply is None:
            raise PageIndexAPIError("MCP server response contained no reply.")
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
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id,
                                   "method": method}
        if params is not None:
            payload["params"] = params
        response = self._post(payload)
        if response.status_code in (400, 404) and self._initialized and _retry:
            # Session expired (stateful servers): start over, retry once.
            with self._lock:
                self._initialized = False
                self._session_id = None
                self._protocol_version = None
            return self._request(method, params, _retry=False)
        if response.status_code >= 400:
            raise PageIndexAPIError(
                f"MCP request failed: HTTP {response.status_code} "
                f"({response.text[:200]})"
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
                    "your API key."
                )
            result = self._extract_result(response, request_id) or {}
            self._session_id = response.headers.get("Mcp-Session-Id")
            self._protocol_version = result.get("protocolVersion",
                                                _PROTOCOL_VERSION)
            self._initialized = True
        try:
            self._post({"jsonrpc": "2.0",
                        "method": "notifications/initialized"})
        except PageIndexAPIError:
            pass  # advisory; a server that required it fails the next request

    # ── public surface ──

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor: Optional[str] = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params) or {}
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments}) or {}
        blocks = result.get("content") or []
        texts = [block.get("text", "") for block in blocks
                 if isinstance(block, dict) and block.get("type") == "text"]
        if len(texts) == len(blocks):
            return "\n".join(texts)
        return json.dumps(blocks, ensure_ascii=False)
