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
import contextvars
import glob
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager, suppress
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
    if _sdk_release() < (0, 2, 166):
        raise PageIndexAPIError(
            "as_claude_mcp requires claude-agent-sdk 0.2.166 or newer — "
            "pip install -U claude-agent-sdk.")
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
        from claude_agent_sdk import (ToolAnnotations, create_sdk_mcp_server,
                                      tool)
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

    hints: dict = {}  # readOnlyHint lets the CLI run the read tools in parallel
    specs = _tool_specs(client, include_management, doc_ids, annotations=hints)
    tools = [
        tool(name, description, schema, annotations=ToolAnnotations(
            **(hints.get(name) or {}), maxResultSizeChars=MAX_RESULT_CHARS),
        )(make_handler(invoke))
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

# One CLI process per call, ~150 MB each (measured): indexing's summary
# lane (SUMMARY_CONCURRENCY wide) must not start that many.
CLAUDE_CODE_CONCURRENCY = 16
# The workflow backend's default width: measured at 64, wider untested.
CLAUDE_CODE_WORKFLOW_LANES = 64
_GATES: dict = {}  # event loop -> [its cap, calls inside]

# AssistantMessage.error kinds, for a CLI that reports no api_error_status.
_ERROR_STATUS = {"authentication_failed": 401, "billing_error": 402,
                 "rate_limit": 429, "invalid_request": 400,
                 "server_error": 500}


# A .env written for API models must not reroute a subscription run: an
# API key would bill it, a base URL would receive its login.
SUBSCRIPTION_ENV = {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": "",
                    "ANTHROPIC_BASE_URL": ""}

# the CLI's stream watchdogs: unset, the byte one follows the event one
# (Claude Code 2.1.301)
_IDLE_TIMEOUTS = {"CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS", "CLAUDE_STREAM_IDLE_TIMEOUT_MS"}


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
    # not a git repo, so the CLI adds no git status to the prompt; the temp
    # dir resolved, as the CLI lists a symlinked cwd (macOS's /var) twice in
    # every prompt (Claude Code 2.1.301), but not a link planted at the name
    cwd = os.path.join(os.path.realpath(tempfile.gettempdir()), "pageindex-claude-code")
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
             "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
             # inherited, it outranks the effort given (Claude Code 2.1.301)
             **({"CLAUDE_CODE_EFFORT_LEVEL": ""} if reasoning_effort else {}),
             # a stream with no bytes, pings included, is dead: re-sent after
             # 30 s, not the CLI's 3 minutes, unless the caller set a timeout
             **({} if _IDLE_TIMEOUTS & os.environ.keys()
                else {"CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "30000"})},
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
    # drops this one message process-wide, not only for the probe;
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


async def _check_session(client, options) -> None:
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
            await _check_session(client, options)
            if prompt is None:
                return
            await client.query(prompt)
            kind = None
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage) and message.error:
                    kind = message.error
                elif isinstance(message, ResultMessage) and message.is_error:
                    from ..local_chat import _claude_code_usage, _with_usage
                    raise _with_usage(_run_error(message, kind, options.max_turns),
                                      _claude_code_usage(message.usage))
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


def _index_options(model, reasoning_effort):
    """claude_code_options for index calls, in a process or a workflow."""
    options = claude_code_options(model, reasoning_effort, max_turns=1)
    if reasoning_effort == "none":
        # where Claude Code keeps thinking on anyway (2.1.301: Haiku 5.5,
        # Sonnet 5.5, Opus 5.5, Fable), it thinks least at low
        options.effort = "low"
    # inherited, it outranks the effort; "" reads as unset (Claude Code 2.1.301)
    options.env["CLAUDE_CODE_EFFORT_LEVEL"] = ""
    return options


async def claude_code_complete(model: str, prompt: str,
                               reasoning_effort=None) -> str:
    """One indexing prompt on the claude-code/ route: no tools, one turn.
    Every failure is final — the CLI already retried what it could — and
    only a 400 stays per-prompt, for the indexing callers to absorb.
    Inside claude_code_workflows(), a call whose settings have a workflow
    runs as one of its agents, whose tokens the scope counts when it ends;
    a prompt too long for a hook to hand over, or a call the workflow
    loses, runs in a process as usual."""
    from ..utils import (LLMRetriesExhausted, _add_usage_counts, _llm_priority,
                         _PriorityGate)
    host = _HOST.get()
    # the CLI measures in UTF-16 code units, as JavaScript strings do
    if host is not None and len(prompt.encode("utf-16-le", "surrogatepass")) // 2 <= HOOK_CONTEXT_MAX:
        try:
            return await host.complete(model, reasoning_effort, prompt,
                                       _llm_priority.get())
        except _Lost:
            pass
    loop = asyncio.get_running_loop()
    # one cap per event loop; submits on separate threads each get their own.
    # The wider indexing lanes queue their calls here, highest priority first
    entry = _GATES.setdefault(loop, [_PriorityGate(CLAUDE_CODE_CONCURRENCY), 0])
    entry[1] += 1
    reply = ""
    try:
        options = _index_options(model, reasoning_effort)
        # one turn, never read back: a cache write would only cost more
        options.env["DISABLE_PROMPT_CACHING"] = "1"
        from claude_agent_sdk import ResultMessage
        async with entry[0].slot(_llm_priority.get()):
            async for message in claude_code_messages(prompt, options):
                if isinstance(message, ResultMessage):
                    reply = message.result or ""
                    from ..local_chat import _claude_code_usage
                    _add_usage_counts(_claude_code_usage(message.usage))
    except Exception as exc:
        _add_usage_counts(getattr(exc, "usage", None) or {})
        # anything unmapped is final too: a bare error would read as a
        # per-node miss and quietly leave fallback summaries
        raise LLMRetriesExhausted(
            str(exc), status_code=getattr(exc, "status_code", None)) from exc
    finally:
        entry[1] -= 1
        if not entry[1]:  # the cap holds its loop: the last call out drops both
            del _GATES[loop]
    return reply


# ── the claude-code/ workflow backend ──

WORKFLOW_START_S = 30      # a workflow whose agents never ask for a call is given up
WORKFLOW_RESULT_S = 60     # how long a workflow session's turn result is waited for
# The hooks reference caps additionalContext at this many characters: past
# it the CLI saves the text to a file and the agent sees its path and a
# 2,000-character preview, so it summarizes the preview or fakes a Read.
HOOK_CONTEXT_MAX = 10_000
# the open claude_code_workflows() scope's _Host, in the contexts it covers
_HOST: contextvars.ContextVar = contextvars.ContextVar("pageindex_claude_code_host",
                                                       default=None)
_LANE = "pageindex-lane-"  # lane k runs agents of type pageindex-lane-k
# an agent's answer when no hook handed it a call: its lane stops, so a CLI
# whose hooks stop reaching workflow agents ends the workflow, not spins it.
# The hook context comes after the agent's task text, so that text names it
# rather than its place: told "the task above", agents answered this instead
# (Claude Code 2.1.299 and 2.1.301)
_NO_TASK = "PAGEINDEX_NO_TASK"
_LANE_TASK = ("Do the task given as SubagentStart hook additional context. "
              f"If none was given, answer only: {_NO_TASK}")
# sent once the workflow stops: the result that answers it carries the
# session's model_usage, cumulative and counting the workflow's agents
_USAGE_PROMPT = "Answer only: done."
_LANES_JS = """export const meta = { name: 'pageindex-lanes', description: 'PageIndex index calls, one per agent' }
const lane = k => async () => {
  for (;;) if ((await agent(%s, { agentType: `%s${k}` }))?.includes('%s')) return
}
await parallel(Array.from({ length: %d }, (_, k) => lane(k)))
"""


class _Lost(Exception):
    """The workflow lost this call before answering it."""


@contextmanager
def claude_code_workflows(lanes=None, prewarm=()):
    """While open, claude-code/ index calls made in its context (its loops
    and their tasks) whose (model, effort) is in ``prewarm`` run as the
    agents of that setting's workflow, ``lanes`` wide (default:
    PAGEINDEX_CLAUDE_CODE_WORKFLOW, else CLAUDE_CODE_WORKFLOW_LANES; 0 keeps
    processes), in sessions on a thread of their own, each started now so
    its launch overlaps whatever runs before the first call. Other calls
    run in processes. A session that dies is replaced once if its agents
    had asked for calls and it never hit the usage limit; otherwise its
    calls go to processes. Leaving stops each workflow before its session
    closes, since a session closed first would let the agents waiting for
    a call run with none, and adds the sessions' token usage to the
    indexing tally; each session deletes what it left under the Claude
    Code config dir. An inner scope defers to the outer one."""
    prewarm = list(prewarm)
    if lanes is None:
        try:
            lanes = int(os.environ.get("PAGEINDEX_CLAUDE_CODE_WORKFLOW",
                                       CLAUDE_CODE_WORKFLOW_LANES))
        except ValueError:
            lanes = 0
    if lanes < 1 or not prewarm or _HOST.get() is not None:
        yield
        return
    host = _Host(lanes, prewarm)
    token = _HOST.set(host)
    try:
        for model, reasoning_effort in prewarm:
            host.workflow(model, reasoning_effort)
        yield
    finally:
        _HOST.reset(token)
        from ..local_chat import _claude_code_usage
        from ..utils import _add_usage_counts
        for usage in host.close():
            _add_usage_counts(_claude_code_usage(usage))


class _Host:
    """The thread whose event loop runs a scope's workflow sessions, one
    per setting the scope declared."""

    def __init__(self, lanes, settings):
        self.lanes = lanes
        self._settings = set(settings)
        self._workflows = {}  # (model, effort) -> its _Workflow
        self._started = []    # every _Workflow, a replaced one too
        self._replacements = set()
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name="pageindex-claude-code-workflows")
        self._thread.start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def workflow(self, model, reasoning_effort):
        """The session for these settings; _Lost when they get none."""
        key = (model, reasoning_effort)
        if key not in self._settings:
            raise _Lost

        async def start():  # built on the host's loop, which its tasks belong to
            return _Workflow(model, reasoning_effort, self.lanes)
        with self._lock:
            old = self._workflows.get(key)
            if old is not None and not old._dead:
                return old
            # one whose agents never asked would die again; a replacement had its chance
            if old is not None and (not old.worked() or old in self._replacements):
                raise _Lost
            workflow = self._workflows[key] = self._run(start()).result()
            self._started.append(workflow)
            if old is not None:
                self._replacements.add(workflow)
        return workflow

    async def complete(self, model, reasoning_effort, prompt, prio):
        workflow = self.workflow(model, reasoning_effort)
        return await asyncio.wrap_future(self._run(workflow.complete(prompt, prio)))

    def close(self):
        """Stop every session; returns each one's token usage, API-shaped."""
        async def drain():
            await asyncio.gather(*(w.aclose() for w in self._started),
                                 return_exceptions=True)
            rest = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in rest:
                task.cancel()
            await asyncio.gather(*rest, return_exceptions=True)
        self._run(drain()).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        return [w.usage() for w in self._started]


def _lane(agent_type):
    head, _, k = (agent_type or "").rpartition("-")
    return int(k) if head + "-" == _LANE and k.isdigit() else None


def _forget_session(session_id, env):
    """Delete what a session left in the Claude Code config dir: a workflow
    writes a journal of its agents' replies there, --no-session-persistence
    or not (Claude Code 2.1.301)."""
    if not re.fullmatch(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", session_id or ""):
        return
    root = (env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
            or os.path.join(os.path.expanduser("~"), ".claude"))
    for path in glob.glob(os.path.join(glob.escape(root), "projects", "*", session_id)):
        shutil.rmtree(path, ignore_errors=True)
        with suppress(OSError):  # its cwd's project, once empty: a cwd per session
            os.rmdir(os.path.dirname(path))


class _Workflow:
    """One Claude Code session running a workflow of ``lanes`` loops, each
    starting agents one after another. A SubagentStart hook hands each new
    agent the next call, highest priority first, waiting while there is
    none; a SubagentStop hook takes its reply. An agent that dies on an API
    error never reaches SubagentStop, so each lane has its own agent type: a
    lane starting an agent while its last call is unanswered lost that call."""

    def __init__(self, model, reasoning_effort, lanes):
        self.lanes = lanes
        self._queue = asyncio.PriorityQueue()
        self._seq = 0
        self._on_lane = {}  # lane -> future of the call its agent has
        self._asked = asyncio.Event()   # an agent asked for a call
        self._ended = asyncio.Event()   # the workflow ended, or the launch without one
        self._result = asyncio.Event()  # a turn's result arrived
        self._stop = asyncio.Event()
        self._dead = False
        self._limited = False           # hit the usage limit: agents would wait out the window
        self._session_id = None
        self._model_usage = {}          # the latest result's, cumulative
        self._options = self._session_options(model, reasoning_effort)
        self._run_task = asyncio.get_running_loop().create_task(self._run())

    def worked(self):
        """Its agents asked for calls, and it never hit the usage limit."""
        return self._asked.is_set() and not self._limited

    def usage(self):
        """The session's tokens on every model, its agents' included."""
        total = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0, "output_tokens_details": {"thinking_tokens": 0}}
        for counts in self._model_usage.values():
            total["input_tokens"] += counts.get("inputTokens") or 0
            total["output_tokens"] += counts.get("outputTokens") or 0
            total["cache_read_input_tokens"] += counts.get("cacheReadInputTokens") or 0
            total["cache_creation_input_tokens"] += counts.get("cacheCreationInputTokens") or 0
            total["output_tokens_details"]["thinking_tokens"] += counts.get("thinkingTokens") or 0
        return total

    async def complete(self, prompt, prio):
        if self._dead:
            raise _Lost
        fut = asyncio.get_running_loop().create_future()
        self._seq += 1
        self._queue.put_nowait((-prio, self._seq, prompt, fut))
        return await fut

    async def aclose(self):
        self._dead = True
        self._stop.set()
        await self._run_task

    def _session_options(self, model, reasoning_effort):
        import dataclasses
        options = _index_options(model, reasoning_effort)  # first: it names a missing SDK's fix
        from claude_agent_sdk import AgentDefinition, HookMatcher
        # its own, so no other user's or scope's script runs in its place
        cwd = os.path.realpath(tempfile.mkdtemp(prefix="pageindex-claude-code-"))
        script = f"pageindex-lanes-{self.lanes}.js"
        with open(os.path.join(cwd, script), "w") as f:
            f.write(_LANES_JS % (json.dumps(_LANE_TASK), _LANE, _NO_TASK, self.lanes))
        self._launch = (f"Run the workflow script at {script} with the Workflow tool, "
                        "passing that path as scriptPath and nothing else. "
                        "Then answer only: done.")
        agent = AgentDefinition(description="Answers one PageIndex indexing prompt.",
                                prompt="Do exactly what the task asks.", tools=[],
                                model="inherit")
        return dataclasses.replace(
            options, cwd=cwd, tools=["Workflow", "Read"], allowed_tools=["Workflow", "Read"],
            # unset, the CLI has a Sonnet check read every agent's reply
            # (Claude Code 2.1.301)
            permission_mode="default",
            agents={f"{_LANE}{k}": agent for k in range(self.lanes)},
            hooks={"SubagentStart": [HookMatcher(matcher=f"^{_LANE}", timeout=86400,
                                                 hooks=[self._on_start])],
                   "SubagentStop": [HookMatcher(matcher=f"^{_LANE}",
                                                hooks=[self._on_stop])]},
            # agents read the prompt prefix they share from the cache
            env={**options.env, "CLAUDE_CODE_PROMPT_CACHE_TTL": "5m",
                 "CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS": str(self.lanes),
                 # set, the first (the second with CLAUDE_CODE_SUBAGENT_MODEL_FORCE)
                 # puts every agent on another model; "" reads as unset (Claude
                 # Code 2.1.301)
                 "CLAUDE_CODE_WORKFLOW_SUBAGENT_MODEL": "", "CLAUDE_CODE_SUBAGENT_MODEL": ""})

    async def _on_start(self, data, _tool_use_id, _context):
        lane = _lane(data.get("agent_type"))
        if lane is None:
            return {}
        self._asked.set()
        lost = self._on_lane.pop(lane, None)
        if lost is not None and not lost.done():
            lost.set_exception(_Lost())
        while True:
            *_, prompt, fut = await self._queue.get()
            if not fut.done():  # its caller may have given up
                break
        self._on_lane[lane] = fut
        return {"hookSpecificOutput": {"hookEventName": "SubagentStart",
                                       "additionalContext": prompt}}

    async def _on_stop(self, data, _tool_use_id, _context):
        lane = _lane(data.get("agent_type"))
        fut = self._on_lane.pop(lane, None)
        reply = data.get("last_assistant_message") or ""
        if fut is not None and not fut.done():
            # it saw no call, and its lane stops, or the CLI gave up on its
            # stream (Claude Code 2.1.301): a process runs it
            if not reply or _NO_TASK in reply:
                fut.set_exception(_Lost())
            else:
                fut.set_result(reply)
        elif fut is None and lane is not None and not self._asked.is_set():
            self._stop.set()  # an agent ran with no call: SubagentStart doesn't reach them
        return {}

    async def _run(self):
        logging.getLogger("asyncio").addFilter(_drop_probe_reaped)  # added once
        try:
            async with _claude_client(self._options) as client:
                await _check_session(client, self._options)
                if self._stop.is_set():  # the scope closed first: no launch to bill
                    return
                await client.query(self._launch)
                reader = asyncio.create_task(self._read(client))
                waits = {reader, asyncio.create_task(self._ended.wait()),
                         asyncio.create_task(self._stop.wait())}
                loop = asyncio.get_running_loop()
                deadline = loop.time() + WORKFLOW_START_S
                while not any(task.done() for task in waits):
                    done, _ = await asyncio.wait(
                        waits, return_when=asyncio.FIRST_COMPLETED,
                        timeout=None if self._asked.is_set() else deadline - loop.time())
                    if not done and not self._asked.is_set():
                        break  # no agent ever asked: the hooks don't reach them
                self._lose_calls()  # to processes now, not after the clean-up below
                # not started yet, it still may: the launch turn's result follows its start
                if not self._task_id and not reader.done():
                    with suppress(asyncio.TimeoutError):  # one started meanwhile is stopped all the same
                        await asyncio.wait_for(self._result.wait(), WORKFLOW_RESULT_S)
                if self._task_id and (self._limited or not self._ended.is_set()):
                    await client.stop_task(self._task_id)
                if self._task_id and not reader.done() and not self._limited:
                    self._result.clear()
                    await client.query(_USAGE_PROMPT)
                    await asyncio.wait_for(self._result.wait(), WORKFLOW_RESULT_S)
                for task in waits:
                    task.cancel()
        except Exception:  # every call goes to a process, which reports it
            logging.getLogger(__name__).debug("claude-code/ workflow failed",
                                              exc_info=True)
        finally:
            self._lose_calls()
            _forget_session(self._session_id, self._options.env)
            shutil.rmtree(self._options.cwd, ignore_errors=True)

    def _lose_calls(self):
        """Dead from here on: every call it holds goes to a process."""
        self._dead = True
        for fut in self._on_lane.values():
            if not fut.done():
                fut.set_exception(_Lost())
        while not self._queue.empty():
            *_, fut = self._queue.get_nowait()
            if not fut.done():
                fut.set_exception(_Lost())

    _task_id = None

    async def _read(self, client):
        from claude_agent_sdk import (TERMINAL_TASK_STATUSES, RateLimitEvent, ResultMessage,
                                      TaskNotificationMessage, TaskStartedMessage,
                                      TaskUpdatedMessage)
        async for message in client.receive_messages():
            self._session_id = getattr(message, "session_id", None) or self._session_id
            if isinstance(message, TaskStartedMessage) and not self._task_id:
                self._task_id = message.task_id
            elif (isinstance(message, RateLimitEvent)
                  and message.rate_limit_info.status == "rejected"):
                self._limited = True  # its agents would wait for the window to reset
                self._ended.set()
            elif (isinstance(message, (TaskNotificationMessage, TaskUpdatedMessage))
                  and message.task_id == self._task_id
                  and message.status in TERMINAL_TASK_STATUSES):
                self._ended.set()  # stopped, or an agent cap, a crash
            elif isinstance(message, ResultMessage):
                self._model_usage = message.model_usage or self._model_usage
                self._result.set()
                if not self._task_id:
                    self._ended.set()  # the launch turn ended without starting it
