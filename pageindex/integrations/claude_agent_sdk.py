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
import logging
import os
import re
import tempfile
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
    # the app's own Agent SDK sessions reap the same probe
    logging.getLogger("asyncio").addFilter(_drop_probe_reaped)
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


def _sdk_release() -> tuple:
    """The installed claude-agent-sdk's (major, minor, patch); () if none."""
    from importlib.metadata import version
    try:
        return tuple(int(n) for n in
                     re.findall(r"\d+", version("claude-agent-sdk"))[:3])
    except ImportError:  # not installed
        return ()


def _cloud_relay(client, include_management: bool, server_name: str):
    if _sdk_release() < (0, 2, 160):
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
_GATES: dict = {}  # event loop -> [its cap, calls inside]

# AssistantMessage.error kinds, for a CLI that reports no api_error_status.
_ERROR_STATUS = {"authentication_failed": 401, "billing_error": 402,
                 "rate_limit": 429, "invalid_request": 400,
                 "server_error": 500}


# A .env written for API models must not reroute a subscription run: an
# API key would bill it, a base URL would receive its login.
SUBSCRIPTION_ENV = {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": "",
                    "ANTHROPIC_BASE_URL": ""}


def is_claude_code(model) -> bool:
    return isinstance(model, str) and model.startswith(CLAUDE_CODE_PREFIX)


def claude_code_options(model: str, reasoning_effort=None, **options):
    """ClaudeAgentOptions for one run of ``claude-code/<id>``, isolated from
    the user's Claude Code setup: no built-in tools, settings, CLAUDE.md,
    hooks or account connectors, no session file, the subscription login
    with no API key or base URL from the environment, and prompts sent
    verbatim (an ``@/path`` in document text would otherwise make the CLI
    read that local file into the prompt).
    ``reasoning_effort`` "none" turns thinking off where the model allows
    it (elsewhere the model's default applies); another level is the CLI's
    effort, checked once connected against the levels Claude Code lists
    for the model (a model listing none drops it)."""
    wire = model[len(CLAUDE_CODE_PREFIX):]
    if not wire:
        raise PageIndexAPIError(
            f"{model!r} names the claude-code route but no model — e.g. "
            "'claude-code/sonnet'.", status_code=404)
    if _sdk_release() < (0, 2, 166):
        # the claude extra's floor, the oldest release the route is verified
        # on; before 0.2.158 the options don't even take verbatim_prompts
        raise PageIndexAPIError(
            "claude-code/ models run on claude-agent-sdk 0.2.166 or newer — "
            "pip install -U 'pageindex[claude]'.")
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
        verbatim_prompts=True, cwd=cwd,
        extra_args={"no-session-persistence": None},
        # a tool result comes back as one stdout line: room for page images
        # past the SDK's 1 MiB default
        max_buffer_size=16 * 2**20,
        env={**SUBSCRIPTION_ENV,
             # no session-title model call per run
             "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
             # the advisor consults a stronger model: extra quota
             "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1"},
        **options)


def _claude_client(options):
    """The CLI session — the seam tests replace with a fake client."""
    from claude_agent_sdk import ClaudeSDKClient
    return ClaudeSDKClient(options=options)


def _drop_probe_reaped(record) -> bool:
    """The SDK terminates its `claude -v` probe even after it has exited; on
    Python 3.12 and earlier that reaps the probe first, and asyncio's child
    watcher logs this for an exit code nobody reads. Matched on the ending
    every child watcher's wording shares, which LiteLLM's redaction filter on
    the same logger keeps when it renders the message first."""
    # ponytail: drops this one message process-wide, not only for the probe;
    # remove once the SDK stops terminating an exited probe
    return not (isinstance(record.msg, str)
                and record.msg.endswith("will report returncode 255"))


def _run_error(result, kind, max_turns) -> PageIndexAPIError:
    if result.subtype == "error_max_turns":
        from ..local_chat import _wrap_max_turns
        return _wrap_max_turns(max_turns)
    status = result.api_error_status or _ERROR_STATUS.get(kind)
    text = result.result or "; ".join(result.errors or []) or result.subtype
    hint = {401: " Log in with your Claude subscription: run `claude`, then "
                 "/login (install Claude Code first if needed).",
            429: " Your Claude subscription's usage limit is reached; it "
                 "resets on its own."}.get(status, "")
    return PageIndexAPIError(f"Claude Code failed: {text.rstrip('.')}.{hint}",
                             status_code=status)


async def claude_code_messages(prompt, options):
    """The run's SDK messages, through its ResultMessage. The login and the
    effort are checked before the prompt goes out, so refusing them bills
    nothing: one signed in with an API key, or routed to another provider,
    is refused (403). An effort outside the levels Claude Code lists for
    the model, which it would otherwise ignore, is refused too; a model
    listing none drops it. With no prompt only the checks run. A failed
    run raises PageIndexAPIError with the HTTP status (None when the CLI
    itself failed). The client disconnects inline however the run ends, so
    no CLI process outlives its event loop."""
    from claude_agent_sdk import AssistantMessage, ResultMessage
    logging.getLogger("asyncio").addFilter(_drop_probe_reaped)  # added once
    try:
        async with _claude_client(options) as client:
            info = (await client.get_server_info()) or {}
            account = info.get("account") or {}
            source = account.get("apiKeySource")
            provider = account.get("apiProvider")
            elsewhere = (f"signed in with an API key ({source})"
                         if source not in (None, "none") else
                         f"routed to {provider}"
                         if provider not in (None, "firstParty") else None)
            if elsewhere:
                raise PageIndexAPIError(
                    "claude-code/ runs on your Claude subscription, but "
                    f"Claude Code is {elsewhere}. Log in with your "
                    "subscription (`claude`, then /login), or use an API "
                    "model to bill that account.", status_code=403)
            models = info.get("models") or []
            if options.effort and models:
                listed = [m for m in models
                          if options.model in (m.get("value"), m.get("resolvedModel"))]
                # a model the CLI doesn't list: any listed model's levels
                levels = list(dict.fromkeys(
                    level for m in listed or models
                    for level in m.get("supportedEffortLevels") or []))
                if levels and options.effort not in levels:
                    # no status: a 400 would read as one prompt's miss in indexing
                    raise PageIndexAPIError(
                        f"claude-code/{options.model} takes reasoning_effort "
                        f"{', '.join(['none', *levels])}, not {options.effort!r}.")
            if prompt is None:
                return
            await client.query(prompt)
            kind = None
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage) and message.error:
                    kind = message.error
                elif isinstance(message, ResultMessage) and message.is_error:
                    raise _run_error(message, kind, options.max_turns)
                yield message
    except PageIndexAPIError:
        raise
    except Exception as exc:  # the SDK raises bare Exceptions too (control timeouts)
        raise PageIndexAPIError(f"Claude Code failed: {exc}") from exc


async def claude_code_check(model: str, reasoning_effort=None) -> None:
    """A run's login and effort checks alone, in a session that sends no
    prompt."""
    async for _ in claude_code_messages(
            None, claude_code_options(model, reasoning_effort, max_turns=1)):
        pass


async def claude_code_complete(model: str, prompt: str,
                               reasoning_effort=None) -> str:
    """One indexing prompt on the claude-code/ route: no tools, one turn.
    Every failure is final — the CLI already retried what it could — and
    only a 400 stays per-prompt, for the indexing callers to absorb."""
    from ..utils import LLMRetriesExhausted, _add_usage_counts
    loop = asyncio.get_running_loop()
    # ponytail: one cap per event loop; submits on separate threads each get their own
    entry = _GATES.setdefault(loop, [asyncio.Semaphore(CLAUDE_CODE_CONCURRENCY), 0])
    entry[1] += 1
    reply = ""
    try:
        options = claude_code_options(model, reasoning_effort, max_turns=1)
        from claude_agent_sdk import ResultMessage
        async with entry[0]:
            async for message in claude_code_messages(prompt, options):
                if isinstance(message, ResultMessage):
                    reply = message.result or ""
                    from ..local_chat import _claude_code_usage
                    _add_usage_counts(_claude_code_usage(message.usage))
    except Exception as exc:
        # anything unmapped is final too: a bare error would read as a
        # per-node miss and quietly leave fallback summaries
        raise LLMRetriesExhausted(
            str(exc), status_code=getattr(exc, "status_code", None)) from exc
    finally:
        entry[1] -= 1
        if not entry[1]:  # the cap holds its loop: the last call out drops both
            del _GATES[loop]
    return reply
