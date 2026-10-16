"""Own-model chat: document-QA agents over the local or cloud agent
tools, and the runs behind both ChatStream views, which also weave the
managed endpoint's chunk stream."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import queue
import threading
import time
import uuid
from typing import Any, Callable, Iterator, Mapping, Optional, Union

from .agent_tools import _base_instructions, targeting_block
from .chat_stream import ChatStream
from .errors import PageIndexAPIError, _pageindex_cause

CHAT_HEADER = (
    "You are PageIndex by Vectify AI, a document-focused assistant. "
    "Be concise, never use emojis, and do not expose tool names."
)


# ── shared: prompt, doc targeting, validation, sync bridges ──

def _managed_instructions(client, extra_system: list[str]) -> str:
    # Local: the built-in subset guidance. Own-model chat over cloud
    # documents: the live instructions the MCP server serves.
    base: str = _base_instructions(client)
    return "\n\n".join([CHAT_HEADER, base,
                        *[t for t in extra_system if t.strip()]])


def _system_text(content: Any) -> str:
    """Text of a system/developer message: a string, or text parts joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part["text"] for part in content
                 if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if texts and len(texts) == len(content):
            return "\n".join(texts)
    raise PageIndexAPIError(
        "system message content must be a string or a list of text parts."
    )


def _split_chat_messages(messages) -> "tuple[list[str], list[dict]]":
    """Validate the chat_completions surface's messages: system/developer
    content joins the managed instructions; user/assistant history passes
    through. Tool-history round-trips belong to chat(protocol="responses")
    or chat(protocol="messages")."""
    messages = list(messages)
    if not messages:
        raise PageIndexAPIError("messages must be a non-empty list.")
    system_texts: list[str] = []
    history: list[dict] = []
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            raise PageIndexAPIError(
                "Each message must be a dict with 'role' and 'content'.")
        role = message["role"]
        if role in ("system", "developer"):
            system_texts.append(_system_text(message.get("content")))
        elif role in ("user", "assistant"):
            content = message.get("content")
            if not isinstance(content, str):
                raise PageIndexAPIError(
                    "content must be a string on this lane; for "
                    "structured items use chat(protocol=\"responses\") "
                    "or chat(protocol=\"messages\")."
                )
            history.append({"role": role, "content": content})
        else:
            raise PageIndexAPIError(
                f"Unsupported role {role!r} on this lane. Tool history "
                "round-trips belong to chat(protocol=\"responses\") or "
                "chat(protocol=\"messages\")."
            )
    if not history:
        raise PageIndexAPIError("messages must contain a user or assistant "
                                "message.")
    return system_texts, history


def _run_sync(coro):
    from .utils import run_off_loop
    return run_off_loop(asyncio.run, coro)



class StreamIterator(Iterator[Any]):
    """Sync iterator wrapper that preserves full Iterator compatibility while
    exposing execution outcome and measurement metadata.

    Lifecycle and Teardown Semantics:
        - ``execution_outcome``: ``"in_progress"`` while streaming or suspended;
          ``"completed"`` on normal exhaustion; ``"failed"`` on exception;
          ``"cancelled"`` when explicitly closed before completion.
        - Closing an already completed or failed stream preserves its terminal
          outcome without overwriting it.
        - Supported local cleanup is executed via ``close()`` (or on garbage collection);
          in-flight remote backend requests cannot be aborted mid-turn. Remote execution
          may continue until the current turn finishes, but subsequent turns and tools
          will not run. Usage never delivered by the provider cannot be recovered or fabricated.
        - Properties ``measurements``, ``measurement_status``, and ``usage`` reflect
          the latest collected state and remain accessible after stream closure or exhaustion.
    """

    def __init__(
        self,
        iterable: Any,
        *,
        on_close: Optional[Callable[[], None]] = None,
        measurements: Optional[list[dict[str, Any]]] = None,
        get_status: Optional[Callable[[], str]] = None,
        get_usage: Optional[Callable[[], Optional[dict[str, Any]]]] = None,
    ) -> None:
        self._it = iter(iterable)
        self._on_close = on_close
        self._measurements = measurements if measurements is not None else []
        self._get_status = get_status
        self._get_usage = get_usage
        self._outcome = "in_progress"

    def __iter__(self) -> "StreamIterator":
        return self

    def __next__(self) -> Any:
        if self._outcome == "completed":
            raise StopIteration
        if self._outcome == "cancelled":
            raise StopIteration
        try:
            return next(self._it)
        except StopIteration:
            if self._outcome == "in_progress":
                self._outcome = "completed"
            raise
        except BaseException as exc:
            if not isinstance(exc, (GeneratorExit, StopIteration)):
                self._outcome = "failed"
            raise

    def close(self) -> None:
        if self._outcome == "in_progress":
            self._outcome = "cancelled"
        close_fn = getattr(self._it, "close", None)
        if close_fn is not None:
            try:
                close_fn()
            except Exception:
                pass
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    @property
    def execution_outcome(self) -> str:
        return self._outcome

    @property
    def measurements(self) -> list[dict[str, Any]]:
        return list(self._measurements)

    @property
    def measurement_status(self) -> str:
        if self._get_status is not None:
            return self._get_status()
        return _aggregate_measurement_status(self._measurements)

    @property
    def usage(self) -> Optional[dict[str, Any]]:
        if self._get_usage is not None:
            return self._get_usage()
        return None


_SENTINEL = object()


def _stream_sync(agen_factory) -> Iterator[Any]:
    """Drive an async generator from a background thread; yield synchronously.

    Closing the iterator cancels the run between items: the pump stops, and
    the async generator's cleanup cancels the underlying agent task, so no
    further model turns or tool executions start. An in-flight backend
    request cannot be aborted mid-turn.
    """
    items: "queue.Queue[Any]" = queue.Queue(maxsize=32)
    cancelled = threading.Event()

    def deliver(item) -> bool:
        while not cancelled.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def pump():
        async def consume():
            agen = agen_factory()

            async def drain():
                async for item in agen:
                    if not deliver(item):
                        break

            # The watchdog lets cancellation land even while drain() is
            # awaiting the backend — a plain async-for would only notice
            # between items.
            task = asyncio.ensure_future(drain())
            try:
                while not task.done():
                    if cancelled.is_set():
                        task.cancel()
                        break
                    await asyncio.sleep(0.05)
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            finally:
                await agen.aclose()

        try:
            asyncio.run(consume())
        except BaseException as exc:  # re-raised on the consumer thread
            deliver(exc)
            return
        deliver(_SENTINEL)

    threading.Thread(target=pump, daemon=True).start()
    try:
        while True:
            item = items.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancelled.set()


# ── OpenAI engine (chat_completions / responses) ──

def _require_openai_agents(method: str) -> None:
    try:
        import agents  # noqa: F401
    except ImportError as exc:
        raise PageIndexAPIError(
            f"{method} with your own chat model requires the OpenAI "
            "Agents SDK — "
            "pip install openai-agents. "
            "chat(protocol='messages') runs on the anthropic extra "
            "instead."
        ) from exc


def _sdk_backend(backend) -> dict:
    """chat_backend for an SDK constructor: LiteLLM takes either endpoint
    spelling, the openai and anthropic SDKs only ``base_url``."""
    return {("base_url" if key == "api_base" else key): value
            for key, value in (backend or {}).items()}


def _openai_model(protocol: str, model_name: str, backend=None):
    """The backend protocol driver — the seam tests replace with a fake."""
    if protocol == "responses":
        model_name = model_name.removeprefix("litellm/")
        import openai
        model_name = model_name.removeprefix("openai/")
        try:
            sdk_client = openai.AsyncOpenAI(**_sdk_backend(backend))
        except (openai.OpenAIError, TypeError) as exc:
            raise PageIndexAPIError(
                f"The OpenAI backend is not configured: {exc}") from exc
        # A caller-owned transport must survive the per-call close.
        sdk_client._pageindex_caller_http = "http_client" in (backend or {})
        from agents.models.openai_responses import OpenAIResponsesModel
        return OpenAIResponsesModel(model_name, openai_client=sdk_client)
    try:
        from agents.extensions.models.litellm_model import LitellmModel
        import litellm
    except ImportError:
        raise PageIndexAPIError(
            f"'{model_name}' routes through LiteLLM, but litellm is not "
            "installed. Run:  pip install -U litellm"
        )
    from .utils import (_litellm_model, _mute_litellm_bridge_usage_warning,
                        _quiet_litellm, _repair_litellm_types)
    _repair_litellm_types()
    _mute_litellm_bridge_usage_warning()
    _quiet_litellm()
    try:
        wire = _litellm_model(model_name)
    except litellm.NotFoundError as exc:
        raise PageIndexAPIError(str(exc)) from exc
    return LitellmModel(wire, api_key=(backend or {}).get("api_key"),
                        base_url=(backend or {}).get("base_url"))


def _reported_model(model_name: str) -> str:
    """The name the provider actually serves — routing prefixes stripped."""
    return model_name.removeprefix("litellm/").removeprefix("openai/")


def _litellm_claude_marks(wire: str) -> Optional[dict]:
    """Claude's prompt caching is opt-in per request: on Claude models
    routed through LiteLLM (Anthropic direct, Bedrock, Vertex — each
    live-verified — and Foundry, same injection), mark the managed system
    prefix and the newest message via LiteLLM's injection param so the
    loop's later turns and a conversation's next calls read them instead
    of repaying full price.
    ``wire`` is the name LiteLLM itself resolves — each lane strips its
    own routing prefixes first, because the lanes normalize differently
    (the chat wire treats bare names as OpenAI shorthand; the Agents SDK
    hands bare names to LiteLLM's own resolution)."""
    try:
        from litellm import get_llm_provider
        from .utils import _quiet_litellm
        _quiet_litellm()
        model, provider, _, _ = get_llm_provider(model=wire)
    except Exception:
        return None
    if provider == "anthropic" or (
            provider in ("bedrock", "vertex_ai", "azure_ai")
            and "claude" in model.lower()):
        # The stable prefix plus the newest message, so each turn re-reads
        # the turns before it. LiteLLM seeds nothing unprompted, so this
        # pair is the marks' sole source.
        return {"cache_control_injection_points": [
            {"location": "message", "role": "system"},
            {"location": "message", "index": -1}]}
    return None


def _cache_extra_args(model_name: str) -> Optional[dict]:
    """The chat lane's marks: normalized exactly as _litellm_model
    normalizes the wire (bare names get openai/), so this predicate
    cannot disagree with where chat_completions actually routes."""
    wire = model_name.removeprefix("litellm/")
    if "/" not in wire or wire.startswith("openai/"):
        return None
    return _litellm_claude_marks(wire)


def _openai_protocol(model_name: str) -> bool:
    """Destinations that speak the OpenAI protocol on the wire, where
    prompt_cache_key means something and extra_body lands in the request
    body. Resolution is LiteLLM's own (same as _cache_extra_args), so the
    answer follows actual routing; azure and openrouter ride the OpenAI
    protocol without appearing in openai_compatible_providers."""
    wire = model_name.removeprefix("litellm/")
    if "/" not in wire or wire.startswith("openai/"):
        return True
    try:
        import litellm
        from .utils import _quiet_litellm
        _quiet_litellm()
        _, provider, _, _ = litellm.get_llm_provider(model=wire)
    except Exception:
        return False
    return (provider in ("openai", "azure", "openrouter")
            or provider in getattr(litellm, "openai_compatible_providers",
                                   ()))


def _merged_backend(client, backend):
    """This call's connection overrides: the client's ``chat_backend``
    under the per-call dict, per-call keys winning."""
    merged = {**(getattr(client, "chat_backend", None) or {}),
              **(backend or {})}
    return merged or None


_SKELETON_KEYS = frozenset({"system", "instructions", "input", "messages",
                            "tools"})
_ARGUMENT_KEYS = frozenset({"stream", "doc_id"})


def _refuse_skeleton(extra_body) -> None:
    """The managed prompt, conversation and tools are the SDK's on every
    lane; extra_body merges last, so a caller's copy would replace them.
    Fields with their own argument select the SDK's parser and scope, so
    they are refused here too."""
    if extra_body is None:
        return
    if not isinstance(extra_body, Mapping):
        raise PageIndexAPIError(
            "extra_body must be a dict of request fields, got "
            f"{type(extra_body).__name__}.")
    hit = sorted(_SKELETON_KEYS.intersection(extra_body))
    if hit:
        raise PageIndexAPIError(
            f"extra_body cannot carry {', '.join(hit)}: the managed prompt, "
            "conversation and tools are the SDK's. Extend the prompt with "
            "instructions= or a leading system row; pass the conversation "
            "as messages, the first argument.")
    hit = sorted(_ARGUMENT_KEYS.intersection(extra_body))
    if hit:
        raise PageIndexAPIError(
            f"extra_body cannot carry {', '.join(hit)}: use "
            f"{' / '.join(key + '=' for key in hit)} instead.")


def _openai_agent(client, protocol: str, model_name: str, instructions: str,
                  temperature, top_p, doc_ids=None, cache_key=None,
                  reasoning=None, reasoning_effort=None, extra_body=None,
                  max_tokens=None, backend=None, extra_headers=None):
    _refuse_skeleton(extra_body)
    from agents import Agent, ModelSettings
    from .integrations.openai_agents import build_openai_tools
    # ModelSettings.extra_body is the one channel all three engines put on
    # the wire: LiteLLM drops the bare prompt_cache_key kwarg (wire-verified),
    # and both OpenAI model classes pass extra_body through verbatim. OpenAI
    # destinations only — LiteLLM plants extra_body as a literal field in
    # other providers' request bodies, which Anthropic rejects as unknown.
    # The responses lane is the OpenAI SDK whatever the name says.
    openai_backend = protocol == "responses" or _openai_protocol(model_name)
    # Chat-lane effort rides extra_args: LiteLLM takes it as its own
    # top-level kwarg on every supported openai-agents version, and the
    # channel admits values outside the OpenAI enum ("none").
    extra_args = _cache_extra_args(model_name) if protocol == "chat" else None
    if reasoning_effort:
        extra_args = {**(extra_args or {}),
                      "reasoning_effort": reasoning_effort}
    conn = _sdk_backend(backend) if backend else {}
    if conn and protocol == "chat":
        # LiteLLM takes connection params per call, except the two names
        # LitellmModel pins as its own keywords — those ride its constructor.
        lifted = {"api_key": conn.pop("api_key", None),
                  "base_url": conn.pop("base_url", None)}
        if conn:
            extra_args = {**(extra_args or {}), **conn}
        conn = {key: value for key, value in lifted.items()
                if value is not None}
    body = ({"prompt_cache_key": cache_key}
            if cache_key and openai_backend else None)
    # Caller extras merge last, so they win over ours; non-OpenAI
    # destinations take them as LiteLLM kwargs instead (see note above).
    routed: dict[str, Any] = {}
    if extra_body:
        if openai_backend:
            body = {**(body or {}), **extra_body}
        else:
            # openai-agents passes ModelSettings' own fields to litellm by
            # name beside **extra_args, so those ride their field.
            own = {field.name for field in dataclasses.fields(ModelSettings)}
            routed = {k: v for k, v in extra_body.items() if k in own}
            rest = {k: v for k, v in extra_body.items() if k not in own}
            if rest:
                extra_args = {**(extra_args or {}), **rest}
    from pydantic import ValidationError
    try:
        settings = ModelSettings(**{
            "temperature": temperature, "top_p": top_p,
            "max_tokens": max_tokens, "reasoning": reasoning,
            # Streamed runs otherwise carry no usage at all (agents forwards
            # this as stream_options only on streaming calls).
            "include_usage": True,
            "extra_body": body,
            "extra_headers": extra_headers,
            "extra_args": extra_args,
            **routed})
    except ValidationError as exc:
        raise PageIndexAPIError(f"Invalid model settings: {exc}") from exc
    return Agent(
        name="PageIndex",
        instructions=instructions,
        tools=build_openai_tools(client, doc_ids=doc_ids),
        model=_openai_model(protocol, model_name, conn or None),
        model_settings=settings,
    )


def _validate_max_turns(max_turns) -> None:
    if max_turns is not None and (not isinstance(max_turns, int)
                                  or max_turns < 1):
        raise PageIndexAPIError("max_turns must be a positive integer.")


def _conversation_cache_key(model_name: str, instructions: str, doc_id,
                            items, folder_id=None) -> str:
    """Stable per-conversation cache-routing key, sent as the OpenAI
    ``prompt_cache_key`` through ModelSettings.extra_body (openai-agents
    0.20 no longer derives it from RunConfig.group_id — verified against a
    captured wire). Keyed on the prefix identity — model, instructions,
    doc targeting, first conversation item — so a conversation's
    continuations share one route without pooling unrelated conversations.
    Callers pass the conversation's own items, never the SDK-prepended
    doc-targeting block: that block is byte-identical for every
    conversation about a document and would pool them all under one key.
    doc_id and folder_id carry the targeting identity instead — the same
    opening question against different documents is different
    conversations."""
    scope = [doc_id] if isinstance(doc_id, str) else doc_id
    seed = json.dumps([model_name, instructions, scope,
                       *([folder_id] if folder_id else []),
                       items[0] if items else None],
                      sort_keys=True, default=str)
    return "pageindex-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _attach_error_measurements(
    err: PageIndexAPIError,
    lane: str,
    measurements: Optional[list[dict[str, Any]]],
) -> None:
    if measurements is not None:
        copied = []
        for m in measurements:
            if isinstance(m, dict):
                item = dict(m)
                if isinstance(item.get("usage"), dict):
                    u_dict = dict(item["usage"])
                    for k, v in list(u_dict.items()):
                        if isinstance(v, dict):
                            u_dict[k] = dict(v)
                    item["usage"] = u_dict
                copied.append(item)
            else:
                copied.append(m)
        err.measurements = copied
        err.measurement_status = _aggregate_measurement_status(copied)
        if lane in ("chat", "chat_completions"):
            err.usage = _openai_usage(copied)
        elif lane == "responses":
            err.usage = _responses_usage(copied)
        elif lane == "messages":
            last_usage = (
                copied[-1].get("usage")
                if (copied and isinstance(copied[-1], dict))
                else {}
            )
            err.usage = (
                _anthropic_usage(copied, last_usage or {})
                if copied
                else None
            )
        else:
            err.usage = None
    else:
        if getattr(err, "measurements", None) is None:
            err.measurements = []
        if getattr(err, "measurement_status", None) is None:
            err.measurement_status = "unknown"
        if getattr(err, "usage", None) is None:
            err.usage = None
    err.execution_outcome = "failed"


def _model_backend_error(
    exc, lane: str, client=None, measurements: Optional[list[dict[str, Any]]] = None
) -> PageIndexAPIError:
    """Wrap a provider failure; the sol-class refusal (chatcmpl rejects
    function tools while reasoning is on) gets its documented exits
    appended, since the fix is a different route, not a retry. The exits
    are per-lane: of the chat lane's three, two are dead ends for a
    Responses-lane caller — it IS the other lane, and its reasoning knob is
    ``reasoning``, not ``reasoning_effort``. On a cloud client an
    auth-shaped failure gets the own-model architecture spelled out —
    the misreading it corrects ("the cloud runs my model") surfaces
    exactly here."""
    message = f"The model backend failed: {exc}"
    if "Function tools with reasoning_effort" in str(exc):
        message += (
            " — this model runs tools on the Responses lane: upgrade "
            "litellm (newer releases route it there automatically)"
        )
        message += (
            ", pass reasoning_effort (older litellm routes explicit "
            "efforts), or use chat(protocol='responses') instead."
            if lane == "chat"
            else "."
        )
    if (getattr(client, "api_key", None)
            and (getattr(exc, "status_code", None) == 401
                 or "api key" in str(exc).lower().replace("_", " "))):
        message += (
            " — note: your chat model runs in your process on your own "
            "provider credentials; the PageIndex api_key does not cover "
            "it. Set the provider key (or chat_backend)")
        message += (
            ", or drop the chat model configuration to use the managed "
            "cloud chat." if lane == "chat" else "."
        )
    err = PageIndexAPIError(message,
                            status_code=getattr(exc, "status_code", None))
    _attach_error_measurements(err, lane, measurements)
    return err


def _translate_run_error(
    exc, max_turns, lane, client=None, measurements: Optional[list[dict[str, Any]]] = None
) -> PageIndexAPIError:
    """The uncaught-run ladder every agent door shares."""
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    if isinstance(exc, MaxTurnsExceeded):
        err = _wrap_max_turns(max_turns)
    elif isinstance(exc, AgentsException):
        cause = _pageindex_cause(exc)
        if cause is not None:
            # a tool failure the invoker re-raised, wrapped on its way out
            err = PageIndexAPIError(str(cause), status_code=cause.status_code)
        else:
            err = PageIndexAPIError(f"The agent backend failed: {exc}")
    else:
        err = _model_backend_error(exc, lane, client, measurements=measurements)
    _attach_error_measurements(err, lane, measurements)
    return err


def _run_kwargs(max_turns) -> dict:
    # No traces — the caller opted into QA, not telemetry.
    from agents import RunConfig
    kwargs: dict = {"run_config": RunConfig(tracing_disabled=True)}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    return kwargs


def _dump_usage(usage: Any) -> Optional[dict[str, Any]]:
    if usage is None:
        return None

    def _to_json_compatible(val: Any) -> Any:
        if val is None or isinstance(val, (int, float, str, bool)):
            return val
        if hasattr(val, "model_dump"):
            try:
                dumped = val.model_dump(mode="json")
            except TypeError:
                dumped = val.model_dump()
            return _to_json_compatible(dumped)
        if dataclasses.is_dataclass(val) and not isinstance(val, type):
            if getattr(val, "requests", None) == 0 and not getattr(val, "request_usage_entries", None):
                return None
            return _to_json_compatible(dataclasses.asdict(val))
        if hasattr(val, "dict") and callable(val.dict):
            return _to_json_compatible(val.dict())
        if hasattr(val, "__dict__"):
            return _to_json_compatible(vars(val))
        if isinstance(val, dict):
            return {str(k): _to_json_compatible(v) for k, v in val.items()}
        if isinstance(val, (list, tuple, set)):
            return [_to_json_compatible(v) for v in val]
        return val

    res = _to_json_compatible(usage)
    return res if isinstance(res, dict) else None


def _extract_call_id(response: Any) -> Optional[str]:
    if response is None:
        return None
    call_id = getattr(response, "id", None) or getattr(response, "response_id", None)
    if call_id:
        return str(call_id)
    if isinstance(response, dict):
        call_id = response.get("id") or response.get("response_id")
        if call_id:
            return str(call_id)
    output = getattr(response, "output", None)
    if output and isinstance(output, list):
        for item in output:
            prov = getattr(item, "provider_data", None)
            if isinstance(prov, dict) and prov.get("response_id"):
                return str(prov["response_id"])
    return None



def _call_measurement_status(usage: Any) -> str:
    """Per-call measurement completeness status ('complete', 'partial', or 'unknown').

    - 'complete': both required base counters (input/prompt and output/completion)
      are reported as non-negative integers (>= 0). Explicit 0 is a valid measurement
      and distinct from absent usage.
    - 'partial': exactly one required base counter is reported.
    - 'unknown': neither required base counter is reported.
    Missing optional cache, reasoning, or lifetime details do not downgrade completeness.
    """
    dumped = _dump_usage(usage)
    if dumped is None:
        return "unknown"
    in_val = dumped.get("input_tokens")
    if in_val is None:
        in_val = dumped.get("prompt_tokens")
    out_val = dumped.get("output_tokens")
    if out_val is None:
        out_val = dumped.get("completion_tokens")
    has_in = isinstance(in_val, int) and not isinstance(in_val, bool) and in_val >= 0
    has_out = isinstance(out_val, int) and not isinstance(out_val, bool) and out_val >= 0
    if has_in and has_out:
        return "complete"
    if has_in or has_out:
        return "partial"
    return "unknown"


def _aggregate_measurement_status(raw_responses: Any) -> str:
    """Aggregate measurement completeness across contributing calls.

    - 'complete': at least one contributing call exists and every call is 'complete'.
    - 'partial': at least one contributing call is known, but not all calls are 'complete'.
    - 'unknown': no contributing calls exist or every contributing call is 'unknown'.
    An aggregate cannot be labeled 'complete' if any contributing call lacks required base measurements.
    """
    if not raw_responses:
        return "unknown"
    statuses = []
    for r in raw_responses:
        u = getattr(r, "usage", None) if not isinstance(r, dict) else r.get("usage")
        statuses.append(_call_measurement_status(u))
    if not statuses:
        return "unknown"
    if all(s == "complete" for s in statuses):
        return "complete"
    if all(s == "unknown" for s in statuses):
        return "unknown"
    return "partial"


def _emit_measurement(
    on_measurement: Optional[Callable[[dict[str, Any]], None]],
    seen: Any,
    call_id: Optional[str],
    turn_index: int,
    model: Optional[str],
    backend: str,
    usage: Any,
    collected_measurements: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Emit an individual model-call measurement record upon turn completion.

    Invoked immediately upon normal completion of each model call / turn.
    Retains distinct call identity (call_id, turn_index, model, backend),
    dumped usage, and per-call completeness status.
    """
    if turn_index in seen:
        return
    seen.add(turn_index)
    payload = {
        "call_id": call_id,
        "turn_index": turn_index,
        "model": model,
        "backend": backend,
        "usage": _dump_usage(usage),
        "status": _call_measurement_status(usage),
    }
    if collected_measurements is not None:
        collected_measurements.append(payload)
    if on_measurement is not None:
        on_measurement(payload)


def _record_response_status(
    agent,
    recorded: dict,
    on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
    seen: Optional[Any] = None,
    turn_counter: Optional[list[int]] = None,
    reported_model: Optional[str] = None,
    backend: str = "responses",
    collected_measurements: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Capture each turn's terminal Response status at the transport client:
    openai-agents' non-streaming path discards Response.status, so a final
    turn truncated at the output cap would otherwise report as a clean
    completion. No-op for backends without an OpenAI responses resource
    (the streaming path records from lifecycle events instead)."""
    responses = getattr(getattr(getattr(agent, "model", None), "_client", None),
                        "responses", None)
    create = getattr(responses, "create", None)
    if create is None:
        return

    if seen is None:
        seen = set()
    if turn_counter is None:
        turn_counter = [0]

    async def recording_create(*args, **kwargs):
        response = await create(*args, **kwargs)
        if response is not None:
            if getattr(response, "status", None):
                recorded["status"] = response.status
                for field in ("incomplete_details", "error"):
                    value = getattr(response, field, None)
                    recorded[field] = (value.model_dump(mode="json")
                                       if hasattr(value, "model_dump") else value)
            # The backend's echo of what it actually ran with.
            for field in ("tool_choice", "parallel_tool_calls"):
                value = getattr(response, field, None)
                if value is not None:
                    recorded[field] = (value.model_dump(mode="json")
                                       if hasattr(value, "model_dump") else value)
            turn_idx = turn_counter[0]
            turn_counter[0] += 1
            call_id = _extract_call_id(response)
            resp_model = getattr(response, "model", None) or reported_model
            usage = getattr(response, "usage", None)
            _emit_measurement(
                on_measurement, seen,
                call_id=call_id,
                turn_index=turn_idx,
                model=resp_model,
                backend=backend,
                usage=usage,
                collected_measurements=collected_measurements,
            )
        return response

    responses.create = recording_create


def _record_chat_finish(
    agent,
    recorded: dict,
    on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
    seen: Optional[Any] = None,
    turn_counter: Optional[list[int]] = None,
    reported_model: Optional[str] = None,
    backend: str = "chat_completions",
    collected_measurements: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Capture each turn's native finish_reason from the raw LiteLLM
    response: openai-agents' ModelResponse drops it, so a truncated or
    content-filtered final turn would otherwise report as a clean "stop".
    The chat protocol has no transport client to hook (cf.
    _record_response_status), so this wraps the model's response fetch;
    no-op if that private seam moves."""
    model = getattr(agent, "model", None)
    fetch = getattr(model, "_fetch_response", None)
    if fetch is None:
        return

    if seen is None:
        seen = set()
    if turn_counter is None:
        turn_counter = [0]

    def note(item) -> None:
        choices = getattr(item, "choices", None)
        finish = getattr(choices[0], "finish_reason", None) if choices else None
        if finish:
            recorded["finish_reason"] = finish

    class _Tee:
        """Iteration passthrough that notes each chunk; everything else
        (aclose/close/...) delegates to the provider stream itself."""

        def __init__(self, inner, turn_counter: list[int], seen: Any,
                     call_id: Optional[str] = None, initial_model: Optional[str] = None,
                     collected_measurements: Optional[list[dict[str, Any]]] = None):
            self._inner = inner
            self._turn_counter = turn_counter
            self._seen = seen
            self._call_id = call_id
            self._model = initial_model
            self._usage = None
            self._completed = False
            self._collected_measurements = collected_measurements

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                chunk = await self._inner.__anext__()
            except StopAsyncIteration:
                if not self._completed:
                    self._completed = True
                    turn_idx = self._turn_counter[0]
                    self._turn_counter[0] += 1
                    _emit_measurement(
                        on_measurement, self._seen,
                        call_id=self._call_id,
                        turn_index=turn_idx,
                        model=self._model or reported_model,
                        backend=backend,
                        usage=self._usage,
                        collected_measurements=self._collected_measurements,
                    )
                raise
            note(chunk)
            chunk_id = _extract_call_id(chunk)
            if chunk_id:
                self._call_id = chunk_id
            if getattr(chunk, "model", None):
                self._model = chunk.model
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                self._usage = chunk_usage
            return chunk

        async def aclose(self):
            aclose_fn = getattr(self._inner, "aclose", None)
            if aclose_fn is not None:
                await aclose_fn()

        def close(self):
            close_fn = getattr(self._inner, "close", None)
            if close_fn is not None:
                close_fn()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    async def recording_fetch(*args, **kwargs):
        result = await fetch(*args, **kwargs)
        if isinstance(result, tuple):
            response, stream = result
            call_id = _extract_call_id(response)
            resp_model = getattr(response, "model", None)
            return response, _Tee(stream, turn_counter, seen, call_id, resp_model,
                                  collected_measurements=collected_measurements)
        note(result)
        turn_idx = turn_counter[0]
        turn_counter[0] += 1
        _emit_measurement(
            on_measurement, seen,
            call_id=_extract_call_id(result),
            turn_index=turn_idx,
            model=getattr(result, "model", None) or reported_model,
            backend=backend,
            usage=getattr(result, "usage", None),
            collected_measurements=collected_measurements,
        )
        return result

    model._fetch_response = recording_fetch


async def _aclose_backend(agent) -> None:
    """Close the per-call AsyncOpenAI client before its event loop ends —
    otherwise httpx tears down pooled connections on a closed loop and
    emits 'Task exception was never retrieved' noise. A client built on a
    caller-owned http_client stays open."""
    backend = getattr(getattr(agent, "model", None), "_client", None)
    if getattr(backend, "_pageindex_caller_http", False):
        return
    close = getattr(backend, "close", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass


async def _run_closing(agent, coro):
    try:
        return await coro
    finally:
        await _aclose_backend(agent)


def _wrap_max_turns(max_turns) -> PageIndexAPIError:
    limit = max_turns if max_turns is not None else "the default limit"
    return PageIndexAPIError(
        f"The agent did not finish within max_turns ({limit}). Raise "
        "max_turns, or narrow the question."
    )


def _usage_sums(
    raw_responses,
) -> "Optional[tuple[int, int, Optional[int], Optional[int], Optional[int]]]":
    has_usage = False
    prompt = 0
    completion = 0
    cached: Optional[int] = None
    cache_write: Optional[int] = None
    reasoning: Optional[int] = None
    for r in raw_responses or ():
        usage = getattr(r, "usage", None) if not isinstance(r, dict) else r.get("usage")
        if usage is None:
            continue
        if getattr(usage, "requests", None) == 0 and not getattr(usage, "request_usage_entries", None):
            continue
        has_usage = True
        if isinstance(usage, dict):
            prompt += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            completion += usage.get("output_tokens") or usage.get("completion_tokens") or 0
            in_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
            if in_details and isinstance(in_details, dict):
                c = in_details.get("cached_tokens")
                if isinstance(c, int) and not isinstance(c, bool):
                    cached = c if cached is None else cached + c
                cw = in_details.get("cache_write_tokens")
                if isinstance(cw, int) and not isinstance(cw, bool):
                    cache_write = cw if cache_write is None else cache_write + cw
            out_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
            if out_details and isinstance(out_details, dict):
                rt = out_details.get("reasoning_tokens")
                if isinstance(rt, int) and not isinstance(rt, bool):
                    reasoning = rt if reasoning is None else reasoning + rt
        else:
            prompt += getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
            completion += getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                c = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
                if isinstance(c, int) and not isinstance(c, bool):
                    cached = c if cached is None else cached + c
                cw = details.get("cache_write_tokens") if isinstance(details, dict) else getattr(details, "cache_write_tokens", None)
                if isinstance(cw, int) and not isinstance(cw, bool):
                    cache_write = cw if cache_write is None else cache_write + cw
            details = getattr(usage, "output_tokens_details", None) or getattr(usage, "completion_tokens_details", None)
            if details is not None:
                rt = details.get("reasoning_tokens") if isinstance(details, dict) else getattr(details, "reasoning_tokens", None)
                if isinstance(rt, int) and not isinstance(rt, bool):
                    reasoning = rt if reasoning is None else reasoning + rt
    if not has_usage:
        return None
    return prompt, completion, cached, cache_write, reasoning


def _openai_usage(raw_responses) -> Optional[dict]:
    """Cross-turn sums, chat.completions dialect."""
    sums = _usage_sums(raw_responses)
    if sums is None:
        return None
    prompt, completion, cached, cache_write, reasoning = sums
    prompt_details = None
    if cached is not None or cache_write is not None:
        prompt_details = {}
        if cached is not None:
            prompt_details["cached_tokens"] = cached
        if cache_write is not None:
            prompt_details["cache_write_tokens"] = cache_write
    completion_details = (
        {"reasoning_tokens": reasoning} if reasoning is not None else None
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": completion_details,
    }


def _responses_usage(raw_responses) -> Optional[dict]:
    """Cross-turn sums, Responses dialect."""
    sums = _usage_sums(raw_responses)
    if sums is None:
        return None
    prompt, completion, cached, cache_write, reasoning = sums
    input_details = None
    if cached is not None or cache_write is not None:
        input_details = {}
        if cached is not None:
            input_details["cached_tokens"] = cached
        if cache_write is not None:
            input_details["cache_write_tokens"] = cache_write
    output_details = (
        {"reasoning_tokens": reasoning} if reasoning is not None else None
    )
    return {
        "input_tokens": prompt,
        "input_tokens_details": input_details,
        "output_tokens": completion,
        "output_tokens_details": output_details,
        "total_tokens": prompt + completion,
    }



def _chat_agent(client, messages, doc_id, model, temperature=None,
                top_p=None, reasoning_effort=None, extra_body=None,
                max_tokens=None, backend=None, extra_headers=None,
                folder_id=None,
                ) -> "tuple[Any, list, str]":
    """The chat lane's shared prologue: validated history, doc targeting,
    and the configured agent. Returns (agent, input items, model name)."""
    system_texts, history = _split_chat_messages(messages)
    scope = client._local_doc_scope(doc_id)
    block = targeting_block(client, doc_id, folder_id)
    items = ([{"role": "user", "content": block}] if block else []) + history
    model_name = model or client.chat_model
    managed = _managed_instructions(client, system_texts)
    agent = _openai_agent(client, "chat", model_name, managed,
                          temperature, top_p, doc_ids=scope,
                          cache_key=_conversation_cache_key(
                              model_name, managed, doc_id, history,
                              folder_id),
                          reasoning_effort=reasoning_effort,
                          extra_body=extra_body, max_tokens=max_tokens,
                          backend=_merged_backend(client, backend),
                          extra_headers=extra_headers)
    return agent, items, model_name


def _output_text(output) -> str:
    """A tool result for the display: a string as it is, the framework's
    structured items by their text, anything else as JSON with a data
    URL's base64 payload elided."""
    if isinstance(output, str):
        return output
    lines = []
    for item in output if isinstance(output, list) else [output]:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            lines.append(item["text"])
        elif kind == "image" and str(item.get("image_url", "")).startswith("data:"):
            head, _, _ = item["image_url"].partition(",")
            lines.append(json.dumps({**item, "image_url": head + ",..."},
                                    ensure_ascii=False))
        else:
            lines.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(lines)


def _clip(text, cap: int = 200) -> str:
    """One display line: whitespace flattened, capped for the terminal."""
    flat = " ".join(str(text).split())
    if len(flat) <= cap:
        return flat
    return f"{flat[:cap]}... (+{len(flat) - cap} chars)"


_PROCESS_DEFAULTS = {"thinking": True, "tool_call": True,
                     "tool_result": True, "max_chars": 200}


def _process_options(show_process) -> dict:
    """chat(show_process=...) normalized: True (or {}) is all defaults, a
    mapping overrides per key, anything else chokes loudly."""
    if show_process is True:
        return dict(_PROCESS_DEFAULTS)
    if not isinstance(show_process, Mapping):
        raise PageIndexAPIError(
            "show_process must be True, False, or a dict with the keys "
            "thinking / tool_call / tool_result (bools) and max_chars "
            "(int).")
    unknown = set(show_process) - set(_PROCESS_DEFAULTS)
    if unknown:
        raise PageIndexAPIError(
            "Unknown show_process keys: "
            f"{', '.join(sorted(map(repr, unknown)))} "
            "— valid keys: thinking, tool_call, tool_result, max_chars.")
    options = {**_PROCESS_DEFAULTS, **show_process}
    for key in ("thinking", "tool_call", "tool_result"):
        if not isinstance(options[key], bool):
            raise PageIndexAPIError(f"show_process[{key!r}] must be a bool.")
    cap = options["max_chars"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise PageIndexAPIError(
            "show_process['max_chars'] must be a positive int.")
    return options


async def _chat_events_agen(client, agent, items, run_kwargs,
                            on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
                            seen: Optional[set] = None,
                            reported_model: Optional[str] = None,
                            collected_measurements: Optional[list[dict[str, Any]]] = None):
    """The chat stream's primitive: the run as typed event dicts —
    thinking/answer deltas, each tool call (arguments parsed when JSON)
    and its full result. Both ChatStream views are built on it."""
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    from openai.types.responses import (
        ResponseReasoningSummaryTextDeltaEvent,
        ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent)
    streamed = Runner.run_streamed(agent, input=items, **run_kwargs)
    completed = False
    names = {}  # call_id -> tool name, to label results
    try:
        async for event in streamed.stream_events():
            if event.type == "raw_response_event":
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    if data.delta:
                        yield {"type": "answer", "delta": data.delta}
                elif isinstance(data, (
                        ResponseReasoningTextDeltaEvent,
                        ResponseReasoningSummaryTextDeltaEvent)):
                    if data.delta:
                        yield {"type": "thinking", "delta": data.delta}
            elif event.type == "run_item_stream_event":
                raw = getattr(event.item, "raw_item", None)
                if event.name == "tool_called":
                    name = getattr(raw, "name", None) or "tool"
                    call_id = getattr(raw, "call_id", None)
                    if call_id:
                        names[call_id] = name
                    arguments = getattr(raw, "arguments", "") or ""
                    try:
                        arguments = json.loads(arguments)
                    except (ValueError, TypeError):
                        pass
                    yield {"type": "tool_call", "call_id": call_id,
                           "name": name, "arguments": arguments}
                elif event.name == "tool_output":
                    call_id = (raw.get("call_id") if isinstance(raw, dict)
                               else getattr(raw, "call_id", None))
                    yield {"type": "tool_result", "call_id": call_id,
                           "name": names.get(call_id, "tool"),
                           "output": event.item.output}
        completed = True
    except (MaxTurnsExceeded, AgentsException, openai.OpenAIError) as exc:
        raise _translate_run_error(exc, run_kwargs.get("max_turns"),
                                   "chat", client,
                                   measurements=collected_measurements) from exc
    finally:
        if not completed and hasattr(streamed, "cancel"):
            streamed.cancel()  # abandoned/failed: stop the agent task
        if completed and hasattr(streamed, "raw_responses"):
            for idx, r in enumerate(streamed.raw_responses):
                _emit_measurement(
                    on_measurement, seen if seen is not None else set(),
                    call_id=_extract_call_id(r),
                    turn_index=idx,
                    model=getattr(r, "model", None) or reported_model,
                    backend="chat_completions",
                    usage=getattr(r, "usage", None),
                    collected_measurements=collected_measurements,
                )
        await _aclose_backend(agent)


def _weave(events, options) -> Iterator[str]:
    """Render the typed event stream as display text: a "[thinking] "
    section per thinking burst, a "[tool_call] name args" line per call
    with its "[tool_result]" line, the answer unlabeled — labels are the
    event type names. options=None is the plain answer-only view;
    closing this generator closes the source."""
    try:
        if options is None:
            for ev in events:
                if ev["type"] == "answer":
                    yield ev["delta"]
            return
        section = None   # the open flowing section: thinking/answer/tool
        opened = False   # anything yielded yet (first section takes no gap)
        cap = options["max_chars"]
        last_call = None  # the [tool_call] line still open for nesting
        call_args = {}    # call_id -> clipped arguments, to label orphans

        def enter(kind, label: str = "") -> str:
            nonlocal section, opened
            gap = "\n\n" if opened else ""
            opened = True
            section = kind
            return gap + label

        for ev in events:
            kind = ev["type"]
            if kind == "answer":
                head = enter("answer") if section != "answer" else ""
                yield head + ev["delta"]
            elif kind == "thinking":
                if not options["thinking"]:
                    continue
                head = (enter("thinking", "[thinking] ")
                        if section != "thinking" else "")
                yield head + ev["delta"]
            elif kind == "tool_call":
                arguments = ev["arguments"]
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                clipped = _clip(arguments, cap)
                call_args[ev["call_id"]] = clipped  # even with call lines hidden
                if not options["tool_call"]:
                    continue
                last_call = ev["call_id"]
                line = f"[tool_call] {ev['name']} {clipped}"
                yield enter("tool") + line.rstrip()
            elif kind == "tool_result":
                if not options["tool_result"]:
                    continue
                out = _clip(_output_text(ev["output"]), cap)
                if section == "tool" and ev["call_id"] == last_call:
                    # directly under its own call line
                    yield f"\n[tool_result] {ev['name']}: {out}"
                else:
                    # parallel calls, or call lines hidden: standalone,
                    # arguments echoed to say whose result this is
                    args = call_args.get(ev["call_id"], "")
                    head = f"[tool_result] {ev['name']} {args}".rstrip()
                    gap = ("\n" if section == "tool" and last_call is None
                           else enter("tool"))
                    yield f"{gap}{head}: {out}"
                last_call = None
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()  # cancel the underlying run on abandonment


def _cloud_chunk_events(chunks) -> Iterator[dict]:
    """Typed events from the managed endpoint's chunk stream: answer
    deltas, and each tool call (name + accumulated arguments) from the
    block_metadata tags — the endpoint interleaves tool-argument JSON
    into delta.content, distinguished only by those tags. It streams no
    thinking and no tool results. Outside a tool block, chunks without
    block_metadata (an older server) are answer text."""
    tool = None  # [name, argument pieces] while inside a tool_use block
    try:
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            meta = chunk.get("block_metadata") or {}
            kind = meta.get("type")
            if kind == "mcp_tool_use_start":
                tool = [meta.get("tool_name") or "tool", []]
                continue
            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            content = delta.get("content")
            if kind == "tool_use_stop":
                if tool is not None:
                    name, pieces = tool
                    arguments = "".join(map(str, pieces))
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        pass
                    yield {"type": "tool_call", "call_id": None,
                           "name": name, "arguments": arguments}
                    tool = None
                continue
            if kind == "tool_use" or tool is not None:
                # inside an open block nothing is answer text: argument
                # chunks accumulate under any tag rather than leaking
                if tool is not None and content:
                    tool[1].append(content)
                continue
            if content:
                yield {"type": "answer", "delta": content}
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()


def run_cloud_chat_stream(chunks,
                          show_process: Union[bool, Mapping[str, Any]] = True,
                          ) -> ChatStream:
    """chat(stream=True) on a managed client: the text view weaves what
    the endpoint serves — tool-call lines from its block_metadata tags
    (that wire carries no thinking and no tool results); .events needs
    the in-process agent."""
    options = (None if show_process is False
               else _process_options(show_process))
    return ChatStream(
        text=lambda: StreamIterator(_weave(_cloud_chunk_events(chunks), options)),
        events=("chat events are produced by the in-process agent, "
                "which the managed chat endpoint does not serve — "
                "construct the client with chat_model=... (or a chat= "
                "model) to run the agent in your process."))


def run_chat_stream(client, messages, doc_id=None, model=None,
                    reasoning_effort=None,
                    show_process: Union[bool, Mapping[str, Any]] = False,
                    max_turns=None, backend=None, extra_headers=None,
                    extra_body=None, folder_id=None,
                    on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
                    ) -> ChatStream:
    """chat(stream=True): validation and the agent build run here, eagerly;
    the run itself starts when the returned stream's chosen view is first
    consumed."""
    options = (None if show_process is False or show_process is None
               else _process_options(show_process))
    _require_openai_agents("chat")
    _validate_max_turns(max_turns)
    if isinstance(messages, str):
        if not messages.strip():
            raise PageIndexAPIError(
                "messages must be a non-empty string or a list of "
                "message dicts.")
        messages = [{"role": "user", "content": messages}]
    agent, items, _ = _chat_agent(client, messages, doc_id, model,
                                  reasoning_effort=reasoning_effort,
                                  extra_body=extra_body, backend=backend,
                                  extra_headers=extra_headers,
                                  folder_id=folder_id)
    run_kwargs = _run_kwargs(max_turns)
    reported_model = _reported_model(model or client.chat_model)
    recorded: dict = {}
    seen = set()
    turn_counter = [0]
    collected_measurements: list[dict[str, Any]] = []
    _record_chat_finish(agent, recorded, on_measurement=on_measurement,
                        seen=seen, turn_counter=turn_counter,
                        reported_model=reported_model,
                        backend="chat_completions",
                        collected_measurements=collected_measurements)

    def events():
        raw = _stream_sync(
            lambda: _chat_events_agen(client, agent, items, run_kwargs,
                                      on_measurement=on_measurement,
                                      seen=seen,
                                      reported_model=reported_model,
                                      collected_measurements=collected_measurements))
        return StreamIterator(raw,
                              measurements=collected_measurements,
                              get_status=lambda: _aggregate_measurement_status(collected_measurements),
                              get_usage=lambda: _openai_usage(collected_measurements))

    def text():
        ev = events()
        w = _weave(ev, options)
        return StreamIterator(w,
                              on_close=ev.close,
                              measurements=collected_measurements,
                              get_status=lambda: _aggregate_measurement_status(collected_measurements),
                              get_usage=lambda: _openai_usage(collected_measurements))

    return ChatStream(text=text, events=events)


def run_chat_completions(client, messages, stream: bool = False,
                         doc_id=None, temperature: Optional[float] = None,
                         stream_metadata: bool = False,
                         enable_citations: bool = False,
                         model: Optional[str] = None,
                         max_turns: Optional[int] = None,
                         top_p: Optional[float] = None,
                         max_tokens: Optional[int] = None,
                         reasoning_effort: Optional[str] = None,
                         extra_body: Optional[dict] = None,
                         extra_headers: Optional[dict] = None,
                         backend: Optional[dict] = None,
                         folder_id: Optional[str] = None,
                         on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
                         ) -> Union[dict, Iterator[str], Iterator[dict]]:
    """Execute chat_completions across agent turns with per-turn measurement tracking.

    Returns the completed response dictionary (with usage, measurement_status, and
    execution_outcome="completed") or a StreamIterator yielding text or chunk events.
    """
    if enable_citations:
        raise PageIndexAPIError(
            "enable_citations needs the managed chat endpoint — "
            + ("drop the chat model configuration to use it, or "
               if getattr(client, "api_key", None) else "")
            + "cite with your own model via chat(citations=True).")
    _require_openai_agents("chat")
    _validate_max_turns(max_turns)
    agent, items, model_name = _chat_agent(
        client, messages, doc_id, model, temperature=temperature,
        top_p=top_p, reasoning_effort=reasoning_effort,
        extra_body=extra_body, max_tokens=max_tokens, backend=backend,
        extra_headers=extra_headers, folder_id=folder_id)
    reported_model = _reported_model(model_name)
    recorded: dict = {}
    seen = set()
    turn_counter = [0]
    collected_measurements: list[dict[str, Any]] = []
    _record_chat_finish(agent, recorded, on_measurement=on_measurement,
                        seen=seen, turn_counter=turn_counter,
                        reported_model=reported_model,
                        backend="chat_completions",
                        collected_measurements=collected_measurements)
    run_kwargs = _run_kwargs(max_turns)
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    if not stream:
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=items, **run_kwargs)))
        except (MaxTurnsExceeded, AgentsException,
                openai.OpenAIError) as exc:
            raise _translate_run_error(exc, max_turns, "chat", client,
                                       measurements=collected_measurements) from exc
        finally:
            if "result" in locals() and hasattr(result, "raw_responses"):
                for idx, r in enumerate(result.raw_responses):
                    _emit_measurement(
                        on_measurement, seen,
                        call_id=_extract_call_id(r),
                        turn_index=idx,
                        model=getattr(r, "model", None) or reported_model,
                        backend="chat_completions",
                        usage=getattr(r, "usage", None),
                        collected_measurements=collected_measurements,
                    )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": reported_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": result.final_output or ""},
                "finish_reason": recorded.get("finish_reason") or "stop",
            }],
            "usage": _openai_usage(result.raw_responses),
            "measurement_status": _aggregate_measurement_status(result.raw_responses),
            "execution_outcome": "completed",
        }

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish=None) -> dict:
        return {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": reported_model,
            "choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish}],
        }

    async def agen():
        from openai.types.responses import ResponseTextDeltaEvent
        streamed = Runner.run_streamed(agent, input=items, **run_kwargs)
        completed = False
        # First yield inside the try: a consumer that stops on the opening
        # chunk must still tear the run down via the finally below.
        try:
            yield chunk({"role": "assistant", "content": ""})
            async for event in streamed.stream_events():
                if (event.type == "raw_response_event"
                        and isinstance(event.data, ResponseTextDeltaEvent)):
                    yield chunk({"content": event.data.delta})
            completed = True
        except (MaxTurnsExceeded, AgentsException,
                openai.OpenAIError) as exc:
            raise _translate_run_error(exc, max_turns, "chat", client,
                                       measurements=collected_measurements) from exc
        finally:
            if not completed and hasattr(streamed, "cancel"):
                streamed.cancel()  # abandoned/failed: stop the agent task
            if completed and hasattr(streamed, "raw_responses"):
                for idx, r in enumerate(streamed.raw_responses):
                    _emit_measurement(
                        on_measurement, seen,
                        call_id=_extract_call_id(r),
                        turn_index=idx,
                        model=getattr(r, "model", None) or reported_model,
                        backend="chat_completions",
                        usage=getattr(r, "usage", None),
                        collected_measurements=collected_measurements,
                    )
            await _aclose_backend(agent)
        yield chunk({}, finish=recorded.get("finish_reason") or "stop")
        yield {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": reported_model, "choices": [],
            "usage": _openai_usage(streamed.raw_responses),
            "measurement_status": _aggregate_measurement_status(streamed.raw_responses),
            "execution_outcome": "completed",
        }

    raw_stream = _stream_sync(agen)
    def _status():
        return _aggregate_measurement_status(collected_measurements)
    def _usage():
        return _openai_usage(collected_measurements)

    if stream_metadata:
        return StreamIterator(raw_stream,
                              measurements=collected_measurements,
                              get_status=_status,
                              get_usage=_usage)
    text_stream = (piece["choices"][0]["delta"]["content"]
                   for piece in raw_stream
                   if piece.get("choices")
                   and "content" in piece["choices"][0]["delta"]
                   and piece["choices"][0]["delta"]["content"])
    return StreamIterator(text_stream,
                          on_close=raw_stream.close if hasattr(raw_stream, "close") else None,
                          measurements=collected_measurements,
                          get_status=_status,
                          get_usage=_usage)


def run_responses(client, input, model: Optional[str] = None,
                  stream: bool = False, doc_id=None,
                  instructions: Optional[str] = None,
                  temperature: Optional[float] = None,
                  top_p: Optional[float] = None,
                  max_turns: Optional[int] = None,
                  max_output_tokens: Optional[int] = None,
                  reasoning: Optional[dict] = None,
                  extra_body: Optional[dict] = None,
                  extra_headers: Optional[dict] = None,
                  backend: Optional[dict] = None,
                  folder_id: Optional[str] = None,
                  on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
                  ) -> Union[dict, Iterator[dict]]:
    """Execute responses API run with per-turn measurement tracking and transcript items.

    Returns the completed response dictionary (with items, usage, measurement_status, and
    execution_outcome="completed") or a StreamIterator yielding stream event dicts.
    """
    _require_openai_agents("chat(protocol='responses')")
    _validate_max_turns(max_turns)
    if isinstance(input, str) and input.strip():
        items = [{"role": "user", "content": input}]
    elif (isinstance(input, list) and input
            and all(isinstance(item, dict) for item in input)):
        items = list(input)
    else:
        raise PageIndexAPIError("messages must be a non-empty string or list "
                               "of item dicts.")
    scope = client._local_doc_scope(doc_id)
    block = targeting_block(client, doc_id, folder_id)
    conversation = items
    if block:
        items = [{"role": "user", "content": block}] + items
    extra = [instructions] if instructions else []
    model_name = model or client.chat_model
    managed = _managed_instructions(client, extra)
    agent = _openai_agent(client, "responses", model_name, managed,
                          temperature, top_p, doc_ids=scope,
                          cache_key=_conversation_cache_key(
                              model_name, managed, doc_id, conversation,
                              folder_id),
                          reasoning=reasoning, extra_body=extra_body,
                          max_tokens=max_output_tokens,
                          backend=_merged_backend(client, backend),
                          extra_headers=extra_headers)
    run_kwargs = _run_kwargs(max_turns)
    reported_model = _reported_model(model_name)
    recorded: dict = {}
    seen = set()
    turn_counter = [0]
    collected_measurements: list[dict[str, Any]] = []
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    given = extra_body or {}

    def envelope(transcript: list, raw_responses) -> dict:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": reported_model,
            "status": recorded.get("status") or "completed",
            "output": [item for item in transcript
                       if item.get("type") != "function_call_output"],
            "items": transcript,
            "usage": _responses_usage(raw_responses),
            "measurement_status": _aggregate_measurement_status(raw_responses),
            "execution_outcome": "completed",
            "instructions": managed,
            "tools": [{"type": "function", "name": tool.name,
                       "description": tool.description,
                       "parameters": tool.params_json_schema,
                       "strict": getattr(tool, "strict_json_schema", True)}
                      for tool in agent.tools],
            # Backend echo when captured; the request sends neither param.
            "tool_choice": recorded.get("tool_choice", "auto"),
            "parallel_tool_calls": recorded.get("parallel_tool_calls", True),
            "temperature": given.get("temperature", temperature),
            "top_p": given.get("top_p", top_p),
            "reasoning": given.get("reasoning", reasoning),
            "max_output_tokens": given.get("max_output_tokens",
                                           max_output_tokens),
            "error": recorded.get("error"),
            "incomplete_details": recorded.get("incomplete_details"),
            "metadata": given.get("metadata"),
        }

    if not stream:
        _record_response_status(agent, recorded, on_measurement=on_measurement,
                                seen=seen, turn_counter=turn_counter,
                                reported_model=reported_model,
                                backend="responses",
                                collected_measurements=collected_measurements)
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=[dict(item) for item in items],
                           **run_kwargs)))
        except (MaxTurnsExceeded, AgentsException,
                openai.OpenAIError) as exc:
            raise _translate_run_error(exc, max_turns,
                                       "responses", client,
                                       measurements=collected_measurements) from exc
        finally:
            if "result" in locals() and hasattr(result, "raw_responses"):
                for idx, r in enumerate(result.raw_responses):
                    _emit_measurement(
                        on_measurement, seen,
                        call_id=_extract_call_id(r),
                        turn_index=idx,
                        model=getattr(r, "model", None) or reported_model,
                        backend="responses",
                        usage=getattr(r, "usage", None),
                        collected_measurements=collected_measurements,
                    )
        transcript = result.to_input_list()[len(items):]
        return envelope(transcript, result.raw_responses)

    lifecycle = {"response.created", "response.in_progress",
                 "response.completed", "response.failed",
                 "response.incomplete", "response.queued"}

    async def agen():
        streamed = Runner.run_streamed(agent,
                                       input=[dict(item) for item in items],
                                       **run_kwargs)
        sequence = 0
        # output_index addresses an item's position in the logical
        # response.output (the final envelope's list). Backend events
        # carry per-turn indexes that restart at 0 each turn, so they are
        # re-based by the count of items already committed by prior turns.
        output_offset = 0
        completed = False
        opened = False
        try:
            async for event in streamed.stream_events():
                if event.type == "raw_response_event":
                    data = event.data.model_dump(exclude_unset=True)
                    if data.get("type") in lifecycle:
                        if data["type"] == "response.created" and not opened:
                            # N per-turn openings collapse to one, carrying
                            # the id and created_at the terminal event will
                            # report.
                            opened = True
                            if data.get("response"):
                                data["response"]["id"] = response_id
                                data["response"]["created_at"] = created_at
                            sequence += 1
                            data["sequence_number"] = sequence
                            yield data
                            continue
                        if data["type"] in ("response.completed",
                                            "response.incomplete",
                                            "response.failed"):
                            # Per-turn terminal state; the last turn's wins
                            # and feeds the final envelope below.
                            state = data.get("response") or {}
                            for field in ("status", "incomplete_details",
                                          "error"):
                                recorded[field] = state.get(field)
                            for field in ("tool_choice",
                                          "parallel_tool_calls"):
                                if state.get(field) is not None:
                                    recorded[field] = state[field]
                            output_offset += len(state.get("output") or [])
                            turn_idx = turn_counter[0]
                            turn_counter[0] += 1
                            _emit_measurement(
                                on_measurement, seen,
                                call_id=_extract_call_id(state),
                                turn_index=turn_idx,
                                model=state.get("model") or reported_model,
                                backend="responses",
                                usage=state.get("usage"),
                                collected_measurements=collected_measurements,
                            )
                        continue
                    if isinstance(data.get("output_index"), int):
                        data["output_index"] += output_offset
                    sequence += 1
                    data["sequence_number"] = sequence
                    yield data
            completed = True
        except AgentsException as exc:
            # a run the envelope already reports failed/incomplete is done —
            # except for max_turns, which always gets its guidance
            if (isinstance(exc, MaxTurnsExceeded)
                    or recorded.get("status") not in ("failed", "incomplete")):
                raise _translate_run_error(exc, max_turns,
                                           "responses", client,
                                           measurements=collected_measurements) from exc
            completed = True
        except openai.OpenAIError as exc:
            raise _translate_run_error(exc, max_turns,
                                       "responses", client,
                                       measurements=collected_measurements) from exc
        finally:
            if not completed and hasattr(streamed, "cancel"):
                streamed.cancel()  # abandoned/failed: stop the agent task
            if completed and hasattr(streamed, "raw_responses"):
                for idx, r in enumerate(streamed.raw_responses):
                    _emit_measurement(
                        on_measurement, seen,
                        call_id=_extract_call_id(r),
                        turn_index=idx,
                        model=getattr(r, "model", None) or reported_model,
                        backend="responses",
                        usage=getattr(r, "usage", None),
                        collected_measurements=collected_measurements,
                    )
            await _aclose_backend(agent)
        transcript = streamed.to_input_list()[len(items):]
        sequence += 1
        status = recorded.get("status") or "completed"
        terminal = {"incomplete": "response.incomplete",
                    "failed": "response.failed"}.get(status,
                                                     "response.completed")
        yield {"type": terminal, "sequence_number": sequence,
               "response": envelope(transcript, streamed.raw_responses)}

    raw_stream = _stream_sync(agen)
    def _status():
        return _aggregate_measurement_status(collected_measurements)
    def _usage():
        return _responses_usage(collected_measurements)

    return StreamIterator(raw_stream,
                          measurements=collected_measurements,
                          get_status=_status,
                          get_usage=_usage)


# ── Anthropic engine (messages) ──

def _require_anthropic() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise PageIndexAPIError(
            "chat(protocol='messages') drives your own chat model and "
            "requires the Anthropic SDK — "
            "pip install anthropic (or pip install 'pageindex[anthropic]')."
        ) from exc
    try:
        from anthropic import beta_tool  # noqa: F401
        from anthropic.lib.tools import ToolError  # noqa: F401
    except ImportError as exc:
        raise PageIndexAPIError(
            "chat(protocol='messages') requires the anthropic SDK tool "
            "runner (with ToolError) — pip install -U anthropic."
        ) from exc


_ANTHROPIC_CLIENTS: dict = {}  # (route, backend) key -> client, kept open

# The transport class per routing prefix — the anthropic SDK ships one
# client per channel, so a row here is what makes a route reachable.
_ROUTE_CLIENTS = {"anthropic": "Anthropic", "bedrock": "AnthropicBedrock",
                  "vertex_ai": "AnthropicVertex",
                  "azure_ai": "AnthropicFoundry"}


def _anthropic_client(backend, route):
    """The backend client — the seam tests replace with a fake transport.
    ``route`` (declared by the model's prefix) picks the SDK client
    class. One client per (route, backend): each construction pays
    ~45 ms of SSL-context build and a cold connection pool. A backend
    whose values defeat hashing constructs per call, as before."""
    import anthropic
    kwargs = _sdk_backend(backend)
    try:
        key = (route, tuple(sorted(
            (k, tuple(sorted(v.items())) if isinstance(v, dict) else v)
            for k, v in kwargs.items())))
        hash(key)
    except TypeError:
        key = None
    if key in _ANTHROPIC_CLIENTS:
        return _ANTHROPIC_CLIENTS[key]
    cls = getattr(anthropic, _ROUTE_CLIENTS[route], None)
    if cls is None:
        # A build predating this route's client class: same contract as
        # the tool-runner probe, one step earlier.
        raise PageIndexAPIError(
            f"messages on this route needs the anthropic SDK's "
            f"{_ROUTE_CLIENTS[route]} client, which this anthropic build "
            "lacks — pip install -U anthropic.")
    try:
        client = cls(**kwargs)
    except (anthropic.AnthropicError, ValueError, TypeError) as exc:
        # Vertex/Foundry refuse a missing region or credential right at
        # construction, each with its own type; same contract for all.
        raise PageIndexAPIError(
            f"The Anthropic backend is not configured: {exc}") from exc
    if key is not None and len(_ANTHROPIC_CLIENTS) < 8:
        # ponytail: cache capped at 8 backends; the tail constructs per call.
        # setdefault: never evict a client another thread may already hold.
        client = _ANTHROPIC_CLIENTS.setdefault(key, client)
    return client


def _anthropic_system(client, extra_system) -> list[dict]:
    """System blocks: cache_control marks the stable managed prefix only
    (the API allows 4 breakpoints total — caller blocks must not consume
    the budget); caller system content follows as its own blocks."""
    blocks = [{"type": "text",
               "text": CHAT_HEADER + "\n\n" + _base_instructions(client),
               "cache_control": {"type": "ephemeral"}}]
    if extra_system is None:
        return blocks
    if isinstance(extra_system, str):
        if extra_system.strip():
            blocks.append({"type": "text", "text": extra_system})
        return blocks
    if isinstance(extra_system, list):
        return blocks + list(extra_system)
    raise PageIndexAPIError("system must be a string or a list of blocks.")


def _cache_marks(system_blocks, messages) -> int:
    """Breakpoints already on the request. The API allows 4 total; the
    top-level moving breakpoint is only added when it fits."""
    blocks = list(system_blocks)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            blocks += [b for b in content if isinstance(b, dict)]
    return sum(1 for b in blocks
               if isinstance(b, dict) and b.get("cache_control"))


def _dump_block(block) -> Any:
    """A content block as a plain JSON dict, minus SDK-internal fields the
    API rejects (ParsedBetaTextBlock.__api_exclude__, e.g. parsed_output)
    and unset response-only defaults (exclude_unset, like the SDK's own
    request serializer — an explicit null fails the request schema)."""
    if hasattr(block, "model_dump"):
        exclude = getattr(type(block), "__api_exclude__", None)
        return block.model_dump(mode="json", exclude_unset=True,
                                exclude=set(exclude) if exclude else None)
    return block


def _dump_message(message) -> dict:
    message = dict(message)
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = [_dump_block(item) for item in content]
    return message


def _anthropic_usage(turns, final_usage: dict) -> dict:
    """The final turn's native usage dict with the token counters replaced
    by cross-turn sums (None-safe); all other native fields survive."""
    totals = dict(final_usage)
    for field in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
        values = []
        for turn in turns:
            usage = getattr(turn, "usage", None) if not isinstance(turn, dict) else turn.get("usage")
            if usage is not None:
                val = usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)
                if isinstance(val, int) and not isinstance(val, bool):
                    values.append(val)
        if values:
            totals[field] = sum(values)

    # Multi-turn cache_creation breakdown (ephemeral_5m, ephemeral_1h)
    e5m_vals = []
    e1h_vals = []
    for turn in turns:
        usage = getattr(turn, "usage", None) if not isinstance(turn, dict) else turn.get("usage")
        if usage is not None:
            cc = usage.get("cache_creation") if isinstance(usage, dict) else getattr(usage, "cache_creation", None)
            if cc is not None:
                v5m = cc.get("ephemeral_5m_input_tokens") if isinstance(cc, dict) else getattr(cc, "ephemeral_5m_input_tokens", None)
                if isinstance(v5m, int) and not isinstance(v5m, bool):
                    e5m_vals.append(v5m)
                v1h = cc.get("ephemeral_1h_input_tokens") if isinstance(cc, dict) else getattr(cc, "ephemeral_1h_input_tokens", None)
                if isinstance(v1h, int) and not isinstance(v1h, bool):
                    e1h_vals.append(v1h)
    if e5m_vals or e1h_vals:
        cc_base = totals.get("cache_creation")
        cc_dict = dict(cc_base) if isinstance(cc_base, dict) else {}
        cc_dict["ephemeral_5m_input_tokens"] = sum(e5m_vals)
        cc_dict["ephemeral_1h_input_tokens"] = sum(e1h_vals)
        totals["cache_creation"] = cc_dict

    # Multi-turn output_tokens_details.thinking_tokens
    thinking_vals = []
    for turn in turns:
        usage = getattr(turn, "usage", None) if not isinstance(turn, dict) else turn.get("usage")
        if usage is not None:
            otd = usage.get("output_tokens_details") if isinstance(usage, dict) else getattr(usage, "output_tokens_details", None)
            if otd is not None:
                tt = otd.get("thinking_tokens") if isinstance(otd, dict) else getattr(otd, "thinking_tokens", None)
                if isinstance(tt, int) and not isinstance(tt, bool):
                    thinking_vals.append(tt)
    if thinking_vals:
        otd_base = totals.get("output_tokens_details")
        otd_dict = dict(otd_base) if isinstance(otd_base, dict) else {}
        otd_dict["thinking_tokens"] = sum(thinking_vals)
        totals["output_tokens_details"] = otd_dict

    return totals


def _default_max_tokens(thinking=None) -> int:
    """The wire-required per-turn budget when the caller sets none: 8192.
    The wire also requires max_tokens > thinking.budget_tokens, so an
    enabled budget lifts the default above itself. Pure arithmetic on the
    caller's own inputs — whether the sum fits the model's output ceiling
    is the API's own ruling (its 400 names both numbers), never a lookup
    here."""
    budget = (thinking.get("budget_tokens")
              if isinstance(thinking, dict) else None)
    if isinstance(budget, int) and not isinstance(budget, bool):
        return budget + 8192
    return 8192


def run_messages(client, messages, model: str, route: str,
                 max_tokens: Optional[int] = None,
                 stream: bool = False, doc_id=None, system=None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 stop_sequences: Optional[list[str]] = None,
                 max_turns: Optional[int] = None,
                 thinking: Optional[dict] = None,
                 extra_body: Optional[dict] = None,
                 extra_headers: Optional[dict] = None,
                 backend: Optional[dict] = None,
                 folder_id: Optional[str] = None,
                 on_measurement: Optional[Callable[[dict[str, Any]], None]] = None,
                 ) -> Union[dict, Iterator[Any]]:
    """Execute Messages API multi-turn agent with per-turn measurement tracking.

    Returns the completed response envelope (with messages, aggregated usage,
    measurement_status, and execution_outcome="completed") or a StreamIterator
    yielding Anthropic event objects.
    """
    from .integrations.anthropic_sdk import build_anthropic_tools

    _require_anthropic()
    import anthropic
    _validate_max_turns(max_turns)
    _refuse_skeleton(extra_body)
    seen = set()
    collected_measurements: list[dict[str, Any]] = []
    if isinstance(messages, str) and messages.strip():
        messages = [{"role": "user", "content": messages}]
    if (not isinstance(messages, list) or not messages
            or not all(isinstance(message, dict) for message in messages)):
        raise PageIndexAPIError("messages must be a non-empty string or a "
                                "list of message dicts.")
    scope = client._local_doc_scope(doc_id)
    block = targeting_block(client, doc_id, folder_id)
    prepared = [dict(message) for message in messages]
    if block:
        prepared = [{"role": "user", "content": block}] + prepared
    passthrough = {key: value for key, value in {
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "stop_sequences": stop_sequences, "thinking": thinking,
        "extra_body": extra_body, "extra_headers": extra_headers,
    }.items() if value is not None}
    system_blocks = _anthropic_system(client, system)
    # Top-level cache_control: the server re-marks the newest block each
    # turn, so the loop re-reads the growing conversation from cache.
    # Counts toward the 4-breakpoint limit (live-verified 400 past it).
    # Bedrock's InvokeModel integration rejects the field for Opus 4.6 and
    # earlier: there each turn's tool results carry the breakpoint instead.
    marks_fit = _cache_marks(system_blocks, prepared) < 4
    cached: dict[str, Any] = (
        {"cache_control": {"type": "ephemeral"}}
        if marks_fit and route != "bedrock" else {})
    # Tools before the transport: on a bridge client building them is
    # network I/O, and a failure there must not strand the client below.
    failures: list = []
    tools = build_anthropic_tools(client, doc_ids=scope, failures=failures)
    merged = _merged_backend(client, backend)
    backend_client = _anthropic_client(merged, route)
    # Close only a per-call construction: cached clients stay open for
    # reuse; a caller-owned http_client survives regardless.
    # list(): an atomic snapshot — a bare .values() scan breaks under a
    # concurrent setdefault.
    owns_transport = ("http_client" not in (merged or {})
                      and backend_client
                      not in list(_ANTHROPIC_CLIENTS.values()))
    try:
        if not hasattr(backend_client.beta.messages, "tool_runner"):
            # An anthropic build predating this route's tool runner passes
            # _require_anthropic (its probes are older): name the gap here.
            raise PageIndexAPIError(
                "messages on this route needs the anthropic SDK's tool "
                "runner, which this anthropic build lacks — "
                "pip install -U anthropic.")
        if max_tokens is None:
            max_tokens = _default_max_tokens(
                (extra_body or {}).get("thinking", thinking))
        runner = backend_client.beta.messages.tool_runner(
            max_tokens=max_tokens,
            messages=prepared,
            model=model,
            tools=tools,
            system=system_blocks,
            stream=stream,
            # Bounded like the OpenAI surfaces (their framework default is 10).
            max_iterations=max_turns if max_turns is not None else 10,
            **passthrough,
            **cached,
        )
    except BaseException:
        # A failure before the runner handoff must not strand the transport
        # the branches below would have closed.
        if owns_transport:
            backend_client.close()
        raise
    # Older Anthropic versions also execute tools on max_tokens turns, newer
    # ones skip them: check right after the runner's own tool step.
    generate_tool_response = runner.generate_tool_call_response
    moved: list = []  # the block holding the bedrock breakpoint

    def checked_tool_response():
        response = generate_tool_response()
        if failures:
            raise failures[0]
        if response and marks_fit and route == "bedrock":
            for block in moved:
                block.pop("cache_control", None)
            moved[:] = [response["content"][-1]]
            moved[0]["cache_control"] = {"type": "ephemeral"}
        return response

    runner.generate_tool_call_response = checked_tool_response

    if stream:
        def events() -> Iterator[Any]:
            try:
                for turn_index, turn_stream in enumerate(runner):
                    for event in turn_stream:
                        yield event
                    final_msg = None
                    try:
                        final_msg = turn_stream.get_final_message()
                    except Exception:
                        final_msg = None
                    _emit_measurement(
                        on_measurement, seen,
                        call_id=_extract_call_id(final_msg) if final_msg is not None else None,
                        turn_index=turn_index,
                        model=(getattr(final_msg, "model", None) if final_msg is not None else None) or model,
                        backend="messages",
                        usage=getattr(final_msg, "usage", None) if final_msg is not None else None,
                        collected_measurements=collected_measurements,
                    )
            except anthropic.AnthropicError as exc:
                raise _model_backend_error(exc, "messages", client,
                                           measurements=collected_measurements) from exc
            finally:
                # runs on exhaustion and abandonment (GeneratorExit) alike
                if owns_transport:
                    backend_client.close()

        ev = events()
        def _status():
            return _aggregate_measurement_status(collected_measurements)
        def _usage():
            last_usage = (
                collected_measurements[-1].get("usage")
                if (collected_measurements and isinstance(collected_measurements[-1], dict))
                else {}
            )
            return (
                _anthropic_usage(collected_measurements, last_usage or {})
                if collected_measurements
                else None
            )

        return StreamIterator(ev,
                              measurements=collected_measurements,
                              get_status=_status,
                              get_usage=_usage)

    turns = []
    try:
        for turn_index, turn in enumerate(runner):
            turns.append(turn)
            _emit_measurement(
                on_measurement, seen,
                call_id=_extract_call_id(turn),
                turn_index=turn_index,
                model=getattr(turn, "model", None) or model,
                backend="messages",
                usage=getattr(turn, "usage", None),
                collected_measurements=collected_measurements,
            )
    except anthropic.AnthropicError as exc:
        raise _model_backend_error(exc, "messages", client,
                                   measurements=collected_measurements) from exc
    finally:
        # safe here: the params read-back below does no HTTP
        if owns_transport:
            backend_client.close()
    if not turns:
        raise PageIndexAPIError("The model returned no response.")
    for block in moved:
        block.pop("cache_control", None)  # a request's breakpoint, not history
    captured: dict = {}

    def capture(params):
        captured.update(params)
        return params

    runner.set_messages_params(capture)
    if not captured.get("messages"):
        # The conversation is read back through a mutator; if a vendor
        # change stops it delivering params, the envelope would silently
        # lose the tool turns — fail loudly instead.
        raise PageIndexAPIError(
            "Could not read the conversation back from the anthropic tool "
            "runner — the installed anthropic version is incompatible with "
            "this pageindex release."
        )
    conversation = list(captured["messages"])
    final = turns[-1]
    envelope = final.model_dump(mode="json")
    envelope["content"] = [_dump_block(item) for item in final.content]
    envelope["usage"] = _anthropic_usage(turns, envelope.get("usage") or {})
    envelope["measurement_status"] = _aggregate_measurement_status(turns)
    envelope["execution_outcome"] = "completed"
    # The full turn sequence (assistant tool_use + user tool_result + final),
    # valid for verbatim append to the caller's history. The runner appends
    # a turn to its params only when it executed tools from it, and which
    # turns qualify is vendor policy that has changed across anthropic
    # releases, so stop_reason alone cannot tell. Whether final's tool_use
    # ids already sit in the history is the ground truth for "already
    # appended".
    new_messages = [_dump_message(message)
                    for message in conversation[len(prepared):]]
    final_blocks = [_dump_block(item) for item in final.content]
    final_ids = {block.get("id") for block in final_blocks
                 if block.get("type") == "tool_use"}
    history_ids = {block.get("id")
                   for message in new_messages
                   if (message.get("role") == "assistant"
                       and isinstance(message.get("content"), list))
                   for block in message["content"]
                   if (isinstance(block, dict)
                       and block.get("type") == "tool_use")}
    if not final_ids or not final_ids <= history_ids:
        # Unexecuted tool_use blocks (refusal turns) have no tool_result,
        # so they cannot enter an appendable history — strip them, as the
        # SDK itself does when it rebuilds params around such a turn.
        appendable = [block for block in final_blocks
                      if block.get("type") != "tool_use"]
        if appendable:
            new_messages = new_messages + [
                {"role": "assistant", "content": appendable}]
    envelope["messages"] = new_messages
    return envelope
