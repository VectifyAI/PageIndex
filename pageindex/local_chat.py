"""Managed local chat: document-QA agents over the local tools."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Iterator, Optional, Union

from .agent_tools import AGENT_INSTRUCTIONS, doc_targeting_block
from .errors import PageIndexAPIError

CHAT_HEADER = (
    "You are PageIndex by Vectify AI, a document-focused assistant. "
    "Be concise, never use emojis, and do not expose tool names."
)


# ── shared: prompt, doc targeting, validation, sync bridges ──

def _managed_instructions(extra_system: list[str]) -> str:
    return "\n\n".join([CHAT_HEADER, AGENT_INSTRUCTIONS, *extra_system])


def _doc_block(client, doc_id) -> Optional[str]:
    if doc_id is None:
        return None
    if not isinstance(doc_id, (str, list)):
        raise PageIndexAPIError("doc_id must be a string or a list of "
                                "strings.")
    # scoped: the chat surfaces also pass doc_id into the tool layer, so
    # name resolution happens inside the allowlist — only a duplicate name
    # within the targeted set shadows.
    return doc_targeting_block(client, doc_id, scoped=True)


def _system_text(content: Any) -> str:
    """Text of a system/developer message: a string, or text parts joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [part.get("text") for part in content
                 if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if texts:
            return "\n".join(texts)
    raise PageIndexAPIError(
        "system message content must be a string or a list of text parts."
    )


def _split_chat_messages(messages) -> "tuple[list[str], list[dict]]":
    """Validate the chat_completions surface's messages: system/developer
    content joins the managed instructions; user/assistant history passes
    through. Tool-history round-trips belong to responses()/messages()."""
    if not isinstance(messages, list) or not messages:
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
                    "chat_completions content must be a string; for "
                    "structured items use responses() or messages()."
                )
            history.append({"role": role, "content": content})
        else:
            raise PageIndexAPIError(
                f"Unsupported role for chat_completions: {role!r}. Tool "
                "history round-trips belong to responses() or messages()."
            )
    if not history:
        raise PageIndexAPIError("messages must contain a user or assistant "
                                "message.")
    return system_texts, history


def _run_sync(coro):
    from .utils import run_off_loop
    return run_off_loop(asyncio.run, coro)


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
            f"{method} in local mode requires the OpenAI Agents SDK — "
            "pip install openai-agents. "
            "messages() runs on the anthropic extra instead."
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
        if "/" in model_name and not model_name.startswith("openai/"):
            raise PageIndexAPIError(
                f"responses() cannot drive '{model_name}': provider-prefixed "
                "models route through LiteLLM, which speaks chat.completions, "
                "not the Responses API. Use chat_completions() (or messages() "
                "for Anthropic models), or point OPENAI_BASE_URL at a "
                "Responses-capable backend and use a bare or "
                "'openai/'-prefixed model name."
            )
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
            "installed. Run:  pip install 'litellm>=1.97'"
        )
    from .utils import _litellm_model, _repair_litellm_types
    _repair_litellm_types()
    try:
        wire = _litellm_model(model_name, backend)
    except litellm.AuthenticationError as exc:
        raise PageIndexAPIError(
            "The OpenAI backend is not configured: set the "
            "OPENAI_API_KEY environment variable, pass an api_key "
            "in chat_backend / backend (any value works for keyless "
            "OPENAI_BASE_URL servers), or point chat_model at "
            "another provider (e.g. 'anthropic/...')."
        ) from exc
    except litellm.NotFoundError as exc:
        raise PageIndexAPIError(str(exc)) from exc
    return LitellmModel(wire, api_key=(backend or {}).get("api_key"),
                        base_url=(backend or {}).get("base_url"))


def _reported_model(model_name: str) -> str:
    """The name the provider actually serves — routing prefixes stripped."""
    return model_name.removeprefix("litellm/").removeprefix("openai/")


def _cache_extra_args(model_name: str) -> Optional[dict]:
    """Claude's prompt caching is opt-in per request: on Claude models
    routed through LiteLLM (Anthropic direct, Bedrock, Vertex — each
    channel live-verified), mark the managed system prefix and the newest
    message via LiteLLM's injection param so the loop's later turns and a
    conversation's next calls read them instead of repaying full price.
    Provider resolution is LiteLLM's own, so this predicate can never
    disagree with where the request actually routes."""
    if "/" not in model_name or model_name.startswith("openai/"):
        return None
    try:
        from litellm import get_llm_provider
        model, provider, _, _ = get_llm_provider(
            model=model_name.removeprefix("litellm/"))
    except Exception:
        return None
    if provider == "anthropic" or (provider in ("bedrock", "vertex_ai")
                                   and "claude" in model.lower()):
        # The pair LiteLLM itself seeds for Anthropic and Bedrock: the
        # stable prefix plus the newest message, so each turn re-reads
        # the turns before it. Passing it explicitly extends it to Vertex.
        return {"cache_control_injection_points": [
            {"location": "message", "role": "system"},
            {"location": "message", "index": -1}]}
    return None


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


def _openai_agent(client, protocol: str, model_name: str, instructions: str,
                  temperature, top_p, doc_ids=None, cache_key=None,
                  reasoning=None, reasoning_effort=None, extra_body=None,
                  max_tokens=None, backend=None, extra_headers=None):
    from agents import Agent, ModelSettings
    from .integrations.openai_agents import build_openai_tools
    # ModelSettings.extra_body is the one channel all three engines put on
    # the wire: LiteLLM drops the bare prompt_cache_key kwarg (wire-verified),
    # and both OpenAI model classes pass extra_body through verbatim. OpenAI
    # destinations only — LiteLLM plants extra_body as a literal field in
    # other providers' request bodies, which Anthropic rejects as unknown.
    openai_backend = _openai_protocol(model_name)
    # Chat-lane effort rides extra_args: LiteLLM takes it as its own
    # top-level kwarg on every supported openai-agents version, and the
    # channel admits values outside the OpenAI enum ("none").
    extra_args = _cache_extra_args(model_name)
    if reasoning_effort is not None:
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
    if extra_body:
        if openai_backend:
            body = {**(body or {}), **extra_body}
        else:
            extra_args = {**(extra_args or {}), **extra_body}
    return Agent(
        name="PageIndex",
        instructions=instructions,
        tools=build_openai_tools(client, doc_ids=doc_ids),
        model=_openai_model(protocol, model_name, conn or None),
        model_settings=ModelSettings(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            reasoning=reasoning,
            # Streamed runs otherwise carry no usage at all (agents forwards
            # this as stream_options only on streaming calls).
            include_usage=True,
            extra_body=body,
            extra_headers=extra_headers,
            extra_args=extra_args),
    )


def _validate_max_turns(max_turns) -> None:
    if max_turns is not None and (not isinstance(max_turns, int)
                                  or max_turns < 1):
        raise PageIndexAPIError("max_turns must be a positive integer.")


def _conversation_cache_key(model_name: str, instructions: str, doc_id,
                            items) -> str:
    """Stable per-conversation cache-routing key, sent as the OpenAI
    ``prompt_cache_key`` through ModelSettings.extra_body (openai-agents
    0.20 no longer derives it from RunConfig.group_id — verified against a
    captured wire). Keyed on the prefix identity — model, instructions,
    doc targeting, first conversation item — so a conversation's
    continuations share one route without pooling unrelated conversations.
    Callers pass the conversation's own items, never the SDK-prepended
    doc-targeting block: that block is byte-identical for every
    conversation about a document and would pool them all under one key.
    doc_id carries the targeting identity instead — the same opening
    question against different documents is different conversations."""
    scope = [doc_id] if isinstance(doc_id, str) else doc_id
    seed = json.dumps([model_name, instructions, scope,
                       items[0] if items else None],
                      sort_keys=True, default=str)
    return "pageindex-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _model_backend_error(exc) -> PageIndexAPIError:
    """Wrap a provider failure; the sol-class refusal (chatcmpl rejects
    function tools while reasoning is on) gets its two documented exits
    appended, since the fix is a different lane, not a retry."""
    message = f"The model backend failed: {exc}"
    if "Function tools with reasoning_effort" in str(exc):
        message += (
            " — this model runs tools on the Responses lane: upgrade "
            "litellm (newer releases route it there automatically), pass "
            "reasoning_effort (older litellm routes explicit efforts), or "
            "call responses() instead."
        )
    return PageIndexAPIError(message)


def _run_kwargs(max_turns) -> dict:
    # No traces — the caller opted into QA, not telemetry.
    from agents import RunConfig
    kwargs: dict = {"run_config": RunConfig(tracing_disabled=True)}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    return kwargs


def _record_response_status(agent, recorded: dict) -> None:
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

    async def recording_create(*args, **kwargs):
        response = await create(*args, **kwargs)
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
        return response

    responses.create = recording_create


def _record_chat_finish(agent, recorded: dict) -> None:
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

    def note(item) -> None:
        choices = getattr(item, "choices", None)
        finish = getattr(choices[0], "finish_reason", None) if choices else None
        if finish:
            recorded["finish_reason"] = finish

    class _Tee:
        """Iteration passthrough that notes each chunk; everything else
        (aclose/close/...) delegates to the provider stream itself."""

        def __init__(self, inner):
            self._inner = inner

        def __aiter__(self):
            return self

        async def __anext__(self):
            chunk = await self._inner.__anext__()
            note(chunk)
            return chunk

        def __getattr__(self, name):
            return getattr(self._inner, name)

    async def recording_fetch(*args, **kwargs):
        result = await fetch(*args, **kwargs)
        if isinstance(result, tuple):
            response, stream = result
            return response, _Tee(stream)
        note(result)
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


def _usage_sums(raw_responses) -> "tuple[int, int, int, int, int]":
    prompt = completion = cached = cache_write = reasoning = 0
    for r in raw_responses:
        if r.usage is None:
            continue
        prompt += r.usage.input_tokens or 0
        completion += r.usage.output_tokens or 0
        details = getattr(r.usage, "input_tokens_details", None)
        cached += getattr(details, "cached_tokens", 0) or 0
        cache_write += getattr(details, "cache_write_tokens", 0) or 0
        details = getattr(r.usage, "output_tokens_details", None)
        reasoning += getattr(details, "reasoning_tokens", 0) or 0
    return prompt, completion, cached, cache_write, reasoning


def _openai_usage(raw_responses) -> dict:
    """Cross-turn sums, chat.completions dialect."""
    prompt, completion, cached, _, reasoning = _usage_sums(raw_responses)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning}}


def _responses_usage(raw_responses) -> dict:
    """Cross-turn sums, Responses dialect."""
    prompt, completion, cached, cache_write, reasoning = (
        _usage_sums(raw_responses))
    return {"input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": cached,
                                     "cache_write_tokens": cache_write},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": prompt + completion}


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
                         ) -> Union[dict, Iterator[str], Iterator[dict]]:
    if enable_citations:
        raise PageIndexAPIError(
            "enable_citations is cloud-only — citations need block-level OCR "
            "data that local mode does not store."
        )
    _require_openai_agents("chat_completions")
    _validate_max_turns(max_turns)
    system_texts, history = _split_chat_messages(messages)
    block = _doc_block(client, doc_id)
    items = ([{"role": "user", "content": block}] if block else []) + history
    model_name = model or client.chat_model
    reported_model = _reported_model(model_name)
    managed = _managed_instructions(system_texts)
    agent = _openai_agent(client, "chat", model_name, managed,
                          temperature, top_p, doc_ids=doc_id,
                          cache_key=_conversation_cache_key(
                              model_name, managed, doc_id, history),
                          reasoning_effort=reasoning_effort,
                          extra_body=extra_body, max_tokens=max_tokens,
                          backend=_merged_backend(client, backend),
                          extra_headers=extra_headers)
    recorded: dict = {}
    _record_chat_finish(agent, recorded)
    run_kwargs = _run_kwargs(max_turns)
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    if not stream:
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=items, **run_kwargs)))
        except MaxTurnsExceeded as exc:
            raise _wrap_max_turns(max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        except openai.OpenAIError as exc:
            raise _model_backend_error(exc) from exc
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
        except MaxTurnsExceeded as exc:
            raise _wrap_max_turns(max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        except openai.OpenAIError as exc:
            raise _model_backend_error(exc) from exc
        finally:
            if not completed and hasattr(streamed, "cancel"):
                streamed.cancel()  # abandoned/failed: stop the agent task
            await _aclose_backend(agent)
        yield chunk({}, finish=recorded.get("finish_reason") or "stop")
        yield {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": reported_model, "choices": [],
            "usage": _openai_usage(streamed.raw_responses),
        }

    if stream_metadata:
        return _stream_sync(agen)
    return (piece["choices"][0]["delta"]["content"]
            for piece in _stream_sync(agen)
            if piece.get("choices")
            and "content" in piece["choices"][0]["delta"]
            and piece["choices"][0]["delta"]["content"])


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
                  ) -> Union[dict, Iterator[dict]]:
    _require_openai_agents("responses")
    _validate_max_turns(max_turns)
    if isinstance(input, str) and input.strip():
        items = [{"role": "user", "content": input}]
    elif (isinstance(input, list) and input
            and all(isinstance(item, dict) for item in input)):
        items = list(input)
    else:
        raise PageIndexAPIError("input must be a non-empty string or list "
                                "of item dicts.")
    block = _doc_block(client, doc_id)
    conversation = items
    if block:
        items = [{"role": "user", "content": block}] + items
    extra = [instructions] if instructions else []
    model_name = model or client.chat_model
    managed = _managed_instructions(extra)
    agent = _openai_agent(client, "responses", model_name, managed,
                          temperature, top_p, doc_ids=doc_id,
                          cache_key=_conversation_cache_key(
                              model_name, managed, doc_id, conversation),
                          reasoning=reasoning, extra_body=extra_body,
                          max_tokens=max_output_tokens,
                          backend=_merged_backend(client, backend),
                          extra_headers=extra_headers)
    run_kwargs = _run_kwargs(max_turns)
    recorded: dict = {}
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    def envelope(transcript: list, raw_responses) -> dict:
        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": _reported_model(model_name),
            "status": recorded.get("status") or "completed",
            "output": [item for item in transcript
                       if item.get("type") != "function_call_output"],
            "items": transcript,
            "usage": _responses_usage(raw_responses),
            "instructions": managed,
            "tools": [{"type": "function", "name": tool.name,
                       "description": tool.description,
                       "parameters": tool.params_json_schema,
                       "strict": getattr(tool, "strict_json_schema", True)}
                      for tool in agent.tools],
            # Backend echo when captured; the request sends neither param.
            "tool_choice": recorded.get("tool_choice", "auto"),
            "parallel_tool_calls": recorded.get("parallel_tool_calls", True),
            "temperature": temperature,
            "top_p": top_p,
            "reasoning": reasoning,
            "max_output_tokens": max_output_tokens,
            "error": recorded.get("error"),
            "incomplete_details": recorded.get("incomplete_details"),
            "metadata": None,
        }

    if not stream:
        _record_response_status(agent, recorded)
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=[dict(item) for item in items],
                           **run_kwargs)))
        except MaxTurnsExceeded as exc:
            raise _wrap_max_turns(max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        except openai.OpenAIError as exc:
            raise PageIndexAPIError(
                f"The model backend failed: {exc}") from exc
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
                        continue
                    if isinstance(data.get("output_index"), int):
                        data["output_index"] += output_offset
                    sequence += 1
                    data["sequence_number"] = sequence
                    yield data
            completed = True
        except MaxTurnsExceeded as exc:
            raise _wrap_max_turns(max_turns) from exc
        except AgentsException as exc:
            if recorded.get("status") not in ("failed", "incomplete"):
                raise PageIndexAPIError(
                    f"The agent backend failed: {exc}") from exc
            completed = True
        except openai.OpenAIError as exc:
            raise PageIndexAPIError(
                f"The model backend failed: {exc}") from exc
        finally:
            if not completed and hasattr(streamed, "cancel"):
                streamed.cancel()  # abandoned/failed: stop the agent task
            await _aclose_backend(agent)
        transcript = streamed.to_input_list()[len(items):]
        sequence += 1
        status = recorded.get("status") or "completed"
        terminal = {"incomplete": "response.incomplete",
                    "failed": "response.failed"}.get(status,
                                                     "response.completed")
        yield {"type": terminal, "sequence_number": sequence,
               "response": envelope(transcript, streamed.raw_responses)}

    return _stream_sync(agen)


# ── Anthropic engine (messages) ──

def _require_anthropic() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise PageIndexAPIError(
            "messages in local mode requires the Anthropic SDK — "
            "pip install anthropic (or pip install 'pageindex[anthropic]')."
        ) from exc
    try:
        from anthropic import beta_tool  # noqa: F401
        from anthropic.lib.tools import ToolError  # noqa: F401
    except ImportError as exc:
        raise PageIndexAPIError(
            "messages in local mode requires anthropic >= 0.108.0 (the tool "
            "runner with ToolError) — pip install -U anthropic."
        ) from exc


def _anthropic_client(backend=None):
    """The backend client — the seam tests replace with a fake transport."""
    import anthropic
    try:
        return anthropic.Anthropic(**_sdk_backend(backend))
    except TypeError as exc:
        raise PageIndexAPIError(
            f"The Anthropic backend is not configured: {exc}") from exc


def _anthropic_system(extra_system, block: Optional[str]) -> list[dict]:
    """System blocks: cache_control marks the stable managed prefix only
    (the API allows 4 breakpoints total — the varying doc block and caller
    blocks must not consume the budget); the doc block and caller system
    content follow as their own blocks."""
    blocks = [{"type": "text",
               "text": CHAT_HEADER + "\n\n" + AGENT_INSTRUCTIONS,
               "cache_control": {"type": "ephemeral"}}]
    if block:
        blocks.append({"type": "text", "text": block})
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
    API rejects (ParsedBetaTextBlock.__api_exclude__, e.g. parsed_output)."""
    if hasattr(block, "model_dump"):
        exclude = getattr(type(block), "__api_exclude__", None)
        return block.model_dump(mode="json",
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
        values = [getattr(turn.usage, field, None) for turn in turns]
        counted = [value for value in values if isinstance(value, int)]
        if counted:
            totals[field] = sum(counted)
    return totals


_CLAUDE_4096_MODELS = ("claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
                       "claude-3-5-sonnet-20240620")


def _default_max_tokens(model: str, thinking=None) -> int:
    """The wire-required per-turn budget when the caller sets none: 8192,
    except the claude-3 generation whose output ceiling is 4096. The wire
    also requires max_tokens > thinking.budget_tokens, so an enabled
    budget lifts the default above itself."""
    budget = (thinking.get("budget_tokens")
              if isinstance(thinking, dict) else None)
    if isinstance(budget, int):
        return budget + 8192
    return 4096 if model.startswith(_CLAUDE_4096_MODELS) else 8192


def run_messages(client, messages, model: str,
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
                 ) -> Union[dict, Iterator[Any]]:
    from .integrations.anthropic_sdk import build_anthropic_tools

    _require_anthropic()
    import anthropic
    _validate_max_turns(max_turns)
    if isinstance(messages, str) and messages.strip():
        messages = [{"role": "user", "content": messages}]
    if (not isinstance(messages, list) or not messages
            or not all(isinstance(message, dict) for message in messages)):
        raise PageIndexAPIError("messages must be a non-empty string or a "
                                "list of message dicts.")
    block = _doc_block(client, doc_id)
    prepared = [dict(message) for message in messages]
    passthrough = {key: value for key, value in {
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "stop_sequences": stop_sequences, "thinking": thinking,
        "extra_body": extra_body, "extra_headers": extra_headers,
    }.items() if value is not None}
    system_blocks = _anthropic_system(system, block)
    # Top-level cache_control: the server re-marks the newest block each
    # turn, so the loop re-reads the growing conversation from cache.
    # Counts toward the 4-breakpoint limit (live-verified 400 past it).
    cached: dict[str, Any] = (
        {"cache_control": {"type": "ephemeral"}}
        if _cache_marks(system_blocks, prepared) < 4 else {})
    merged = _merged_backend(client, backend)
    # The SDK defers credential resolution to request time and raises a
    # bare TypeError there — pre-check for the contract's PageIndexAPIError.
    if not merged and not (os.environ.get("ANTHROPIC_API_KEY")
                           or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise PageIndexAPIError(
            "The Anthropic backend is not configured: set the "
            "ANTHROPIC_API_KEY environment variable, or pass an api_key "
            "in chat_backend / backend.")
    backend_client = _anthropic_client(merged)
    # A caller-owned http_client must survive the per-call closes below.
    owns_transport = "http_client" not in (merged or {})
    if max_tokens is None:
        max_tokens = _default_max_tokens(model, thinking)
    runner = backend_client.beta.messages.tool_runner(
        max_tokens=max_tokens,
        messages=prepared,
        model=model,
        tools=build_anthropic_tools(client, doc_ids=doc_id),
        system=system_blocks,
        stream=stream,
        # Bounded like the OpenAI surfaces (their framework default is 10).
        max_iterations=max_turns if max_turns is not None else 10,
        **passthrough,
        **cached,
    )

    if stream:
        def events() -> Iterator[Any]:
            try:
                for turn_stream in runner:
                    for event in turn_stream:
                        yield event
            except anthropic.AnthropicError as exc:
                raise PageIndexAPIError(
                    f"The model backend failed: {exc}") from exc
            finally:
                # runs on exhaustion and abandonment (GeneratorExit) alike
                if owns_transport:
                    backend_client.close()
        return events()

    try:
        turns = [turn for turn in runner]
    except anthropic.AnthropicError as exc:
        raise PageIndexAPIError(
            f"The model backend failed: {exc}") from exc
    finally:
        # safe here: the params read-back below does no HTTP
        if owns_transport:
            backend_client.close()
    if not turns:
        raise PageIndexAPIError("The model returned no response.")
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
    # The full turn sequence (assistant tool_use + user tool_result + final),
    # valid for verbatim append to the caller's history. The runner appends
    # a turn to its params only when it executed tools from it — content
    # carried tool_use blocks and the turn was not a refusal. stop_reason
    # alone cannot tell: a max_tokens turn with complete tool_use blocks
    # still executes. Whether final's tool_use ids already sit in the
    # history is the ground truth for "already appended".
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
