"""Claude Agent SDK adapter: one value for the mcp_servers slot.

An in-process SDK MCP server in both modes. Cloud clients get a relay that
answers the CLI's MCP session with the PageIndex MCP server's own (its
handshake and instructions, tools, resources), fetched live over this
SDK's MCP client, so the API key stays off the CLI's command line (where
a remote server config would carry it, readable by any local process);
include_management picks the endpoint (the read-only
``?tools=read`` URL by default). Local clients get a server over the same
tool contract, gated the same way at registration, its results passed
through as the MCP content they are.

Also the ``claude-code/`` route's transport: indexing and chat calls run
as Claude Code CLI runs on the user's own Claude subscription login.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
import weakref
from contextlib import aclosing
from typing import Any, Optional

from .._version import sdk_version
from ..agent_tools import TOOL_RESPONSE_CHAR_LIMIT
from ..errors import PageIndexAPIError

# Claude Code shows the model only a 2KB preview of an MCP result over
# 50,000 chars; PageIndex results run to about TOOL_RESPONSE_CHAR_LIMIT,
# so twice it keeps them whole.
MAX_RESULT_CHARS = TOOL_RESPONSE_CHAR_LIMIT * 2


def build_claude_mcp(client, include_management: bool = False, *,
                     server_name: str = "pageindex"):
    if getattr(client, "api_key", None):
        return _cloud_relay(client, include_management, server_name)
    return tool_server(client, include_management, server_name=server_name)


def tool_server(client, include_management: bool = False, *,
                server_name: str = "pageindex", doc_ids=None,
                failures: Optional[list] = None):
    """The agent tools as an in-process server in either mode, on cloud
    through the bridge: local as_claude_mcp()'s, and the claude-code/ chat
    engine's, whose system prompt carries the guidance. ``failures``
    records the invoker's re-raised failures (auth, limits, an unreachable
    server) for the engine to fail the run on; the CLI then gets the call
    back as an error."""
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except ImportError as exc:
        raise PageIndexAPIError(
            "The Claude Agent SDK is not installed — "
            "pip install claude-agent-sdk (or pip install 'pageindex[claude]')."
        ) from exc
    from ..agent_tools import _tool_specs

    def make_handler(invoke):
        async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                blocks, is_error = await asyncio.to_thread(invoke, arguments or {})
            except PageIndexAPIError as exc:
                if failures is None:
                    raise
                failures.append(exc)
                return {"content": [{"type": "text", "text": str(exc)}],
                        "is_error": True}
            result: dict[str, Any] = {"content": blocks}
            if is_error:
                result["is_error"] = True
            return result
        return handler

    def tool_kwargs(name: str) -> dict:
        try:
            from claude_agent_sdk import ToolAnnotations
        except ImportError:
            return {}
        return {"annotations": ToolAnnotations(
            **(hints.get(name) or {}), maxResultSizeChars=MAX_RESULT_CHARS)}

    hints: dict = {}  # readOnlyHint lets the CLI run the read tools in parallel
    specs = _tool_specs(client, include_management, doc_ids, annotations=hints)
    tools = [
        tool(name, description, schema,
             **tool_kwargs(name))(make_handler(invoke))
        for name, description, schema, invoke in specs
    ]
    return create_sdk_mcp_server(name=server_name, version=sdk_version(),
                                 tools=tools)


def _cloud_relay(client, include_management: bool, server_name: str):
    from importlib.metadata import version
    try:
        release = tuple(int(n) for n in
                        re.findall(r"\d+", version("claude-agent-sdk"))[:3])
    except ImportError:  # not installed
        release = ()
    if release < (0, 2, 160):
        # older releases answer the CLI from a server's registered
        # handlers, which the relay lacks (before 0.2.140), or close the
        # CLI's stdin at the first result (before 0.2.160): the agent
        # would quietly run without the tools
        raise PageIndexAPIError(
            "as_claude_mcp on cloud requires claude-agent-sdk 0.2.160 or "
            "newer — pip install -U claude-agent-sdk.")
    import anyio.to_thread
    import mcp.types
    from mcp.server.lowlevel import Server
    from ..mcp_bridge import McpBridge

    # include_management picks the endpoint — the URL itself is the gate
    # (?tools=read serves only readOnlyHint-annotated tools).
    url = f"{client.BASE_URL}/mcp" + ("" if include_management else "?tools=read")
    headers = {"Authorization": f"Bearer {client.api_key}"}

    class Relay(Server):
        async def run(self, read_stream, write_stream, *args, **kwargs):
            bridge = McpBridge(url, headers)  # a cloud session per CLI session

            async def answer(request):
                reply = {"jsonrpc": "2.0", "id": request.id}
                try:
                    result = await anyio.to_thread.run_sync(
                        bridge.relay, request.method, request.params,
                        abandon_on_cancel=True)
                    outgoing = _session_message({**reply, "result": result or {}})
                except Exception as exc:  # one bad call must not end the session
                    # the server's own error, which Claude Code may act on
                    error = getattr(exc, "rpc_error", None)
                    if not (isinstance(error, dict) and isinstance(error.get("code"), int)
                            and isinstance(error.get("message"), str)):
                        error = {"code": -32603,  # JSON-RPC internal error
                                 "message": str(exc) or repr(exc)}
                    outgoing = _session_message({**reply, "error": error})
                await write_stream.send(outgoing)

            async with anyio.create_task_group() as tg:
                async for item in read_stream:
                    message = getattr(item, "message", None)
                    message = getattr(message, "root", message)  # mcp 1.x wraps it
                    # notifications stay here: the cloud session has its own
                    if isinstance(message, mcp.types.JSONRPCRequest):
                        tg.start_soon(answer, message)
                tg.cancel_scope.cancel()

    return {"type": "sdk", "name": server_name, "instance": Relay(server_name)}


def _session_message(payload: dict):
    import mcp.types
    from mcp.shared.message import SessionMessage
    adapter = getattr(mcp.types, "jsonrpc_message_adapter", None)  # mcp 2.x
    return SessionMessage(adapter.validate_python(payload) if adapter
                          else mcp.types.JSONRPCMessage.model_validate(payload))


# ── the claude-code/ route ──

CLAUDE_CODE_PREFIX = "claude-code/"

# One CLI process per call, ~150 MB each (measured): indexing's 64-wide
# summary lane must not start 64 of them.
CLAUDE_CODE_CONCURRENCY = 8
_GATES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

# AssistantMessage.error kinds, for a CLI that reports no api_error_status.
_ERROR_STATUS = {"authentication_failed": 401, "billing_error": 402,
                 "rate_limit": 429, "invalid_request": 400,
                 "server_error": 500}


def is_claude_code(model) -> bool:
    return isinstance(model, str) and model.startswith(CLAUDE_CODE_PREFIX)


def claude_code_options(model: str, reasoning_effort=None, **options):
    """ClaudeAgentOptions for one run of ``claude-code/<id>``, isolated from
    the user's Claude Code setup: no built-in tools, settings, CLAUDE.md,
    hooks or account connectors, no session file, and the subscription
    login, never an API key from the environment (a .env
    ANTHROPIC_API_KEY would silently win over the login).
    ``reasoning_effort`` "none" turns thinking off; another level is the
    CLI's own effort (one it does not know, it warns about and ignores)."""
    wire = model[len(CLAUDE_CODE_PREFIX):]
    if not wire:
        raise PageIndexAPIError(
            f"{model!r} names the claude-code route but no model — e.g. "
            "'claude-code/sonnet'.", status_code=404)
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise PageIndexAPIError(
            "claude-code/ models run on the Claude Agent SDK — "
            "pip install 'pageindex[claude]'.") from exc
    if reasoning_effort == "none":
        options["thinking"] = {"type": "disabled"}
    elif reasoning_effort:
        options["effort"] = reasoning_effort
    # not a git repo, so the CLI adds no git status to the prompt
    cwd = os.path.join(tempfile.gettempdir(), "pageindex-claude-code")
    os.makedirs(cwd, exist_ok=True)
    return ClaudeAgentOptions(
        model=wire, tools=[], setting_sources=[], strict_mcp_config=True,
        cwd=cwd, extra_args={"no-session-persistence": None},
        env={"ANTHROPIC_API_KEY": "",
             # no session-title model call per run
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
             # the advisor consults a stronger model: extra quota
             "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1"},
        **options)


def _claude_query(prompt, options):
    """The CLI run — the seam tests replace with a fake message stream."""
    from claude_agent_sdk import query
    return query(prompt=prompt, options=options)


def _run_error(result, kind, max_turns) -> PageIndexAPIError:
    if result.subtype == "error_max_turns":
        from ..local_chat import _wrap_max_turns
        return _wrap_max_turns(max_turns)
    status = result.api_error_status or _ERROR_STATUS.get(kind)
    text = result.result or "; ".join(result.errors or []) or result.subtype
    hint = {401: " Run `claude` once and log in with your Claude "
                 "subscription.",
            429: " Your Claude subscription's usage limit is reached; it "
                 "resets on its own."}.get(status, "")
    return PageIndexAPIError(f"Claude Code failed: {text.rstrip('.')}.{hint}",
                             status_code=status)


async def claude_code_messages(prompt, options):
    """The run's SDK messages. A login billed to an API key is refused, and
    a failed run raises PageIndexAPIError with the HTTP status (None when
    the CLI itself failed)."""
    from claude_agent_sdk import (AssistantMessage, ClaudeSDKError,
                                  ResultMessage, SystemMessage)
    kind = error = None
    try:
        async with aclosing(_claude_query(prompt, options)) as messages:
            async for message in messages:
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    source = message.data.get("apiKeySource")
                    if source not in (None, "none"):
                        raise PageIndexAPIError(
                            "claude-code/ runs on your Claude subscription, "
                            "but Claude Code is signed in with an API key "
                            f"({source}). Log in with your subscription "
                            "(`claude`, then /login), or use an anthropic/ "
                            "model to bill the key.", status_code=403)
                elif isinstance(message, AssistantMessage) and message.error:
                    kind = message.error
                elif isinstance(message, ResultMessage) and message.is_error:
                    # read on: the CLI exits by itself and the SDK raises
                    # after it — stopped early, the process outlives the loop
                    error = _run_error(message, kind, options.max_turns)
                    continue
                yield message
    except ClaudeSDKError as exc:
        raise error or PageIndexAPIError(f"Claude Code failed: {exc}") from exc
    if error:
        raise error


async def claude_code_complete(model: str, prompt: str,
                               reasoning_effort=None) -> str:
    """One indexing prompt on the claude-code/ route: no tools, one turn.
    Every failure is final — the CLI already retried what it could — and
    only a 400 stays per-prompt, for the indexing callers to absorb."""
    from ..utils import LLMRetriesExhausted
    # ponytail: one cap per event loop; submits on separate threads each get their own
    gate = _GATES.setdefault(asyncio.get_running_loop(),
                             asyncio.Semaphore(CLAUDE_CODE_CONCURRENCY))
    reply = ""
    try:
        options = claude_code_options(model, reasoning_effort, max_turns=1)
        from claude_agent_sdk import ResultMessage
        async with gate:
            # read to the end: the CLI exits on its own after the result,
            # while one stopped early outlives an asyncio.run's loop
            async for message in claude_code_messages(prompt, options):
                if isinstance(message, ResultMessage):
                    reply = message.result or ""
    except Exception as exc:
        # anything unmapped is final too: a bare error would read as a
        # per-node miss and quietly leave fallback summaries
        raise LLMRetriesExhausted(
            str(exc), status_code=getattr(exc, "status_code", None)) from exc
    return reply
