"""Managed local chat: document-QA agents over the local tools.

Three methods, three backend protocols, routed 1:1: ``chat_completions``
drives the backend's /chat/completions (any OpenAI-compatible backend,
final answer only), ``responses`` drives /responses (process items are
standard output; round-trip them for provider prompt-cache continuation and
agent memory), ``messages`` drives Anthropic's /v1/messages via the SDK's
own tool runner (tool_use/tool_result round-trip is the format's native
behavior).

Content passes through untouched — the caller's messages, the model's
answers, tool outputs. Native stop reasons pass through on ``messages``;
the OpenAI engine's abstraction does not surface per-turn finish reasons,
so ``chat_completions`` reports loop completion as ``"stop"``, while
``responses`` reports the backend's terminal ``status`` where the wire
surfaces one (recorded at the transport layer — the framework discards
it). The SDK owns gatekeeping (structural validation), table-setting
(managed instructions, tools, doc targeting), tool execution, and billing
(usage aggregation, envelope ids).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
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
    doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
    missing = []
    for one_id in doc_ids:
        try:
            client.get_document(one_id)
        except PageIndexAPIError:
            missing.append(str(one_id))
    if missing:
        raise PageIndexAPIError(
            "Documents not found or access denied: " + ", ".join(missing)
        )
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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


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
            "pip install openai-agents (or pip install 'pageindex[openai]')."
        ) from exc


def _openai_model(protocol: str, model_name: str):
    """The backend protocol driver — the seam tests replace with a fake.

    ``litellm/<provider>/<model>`` (the client's normalized retrieve_model
    form) and bare ``<provider>/<model>`` paths drive the provider through
    LiteLLM; an ``openai/`` prefix strips to the OpenAI SDK; bare names go
    to the OpenAI SDK as-is."""
    if "/" in model_name and not model_name.startswith("openai/"):
        from agents.extensions.models.litellm_model import LitellmModel
        return LitellmModel(model_name.removeprefix("litellm/"))
    from openai import AsyncOpenAI
    model_name = model_name.removeprefix("openai/")
    if protocol == "chat":
        from agents.models.openai_chatcompletions import (
            OpenAIChatCompletionsModel)
        return OpenAIChatCompletionsModel(model_name, AsyncOpenAI())
    from agents.models.openai_responses import OpenAIResponsesModel
    return OpenAIResponsesModel(model_name, openai_client=AsyncOpenAI())


def _openai_agent(client, protocol: str, model_name: str, instructions: str,
                  temperature, top_p, doc_ids=None):
    from agents import Agent, ModelSettings
    from .integrations.openai_agents import build_openai_tools
    return Agent(
        name="PageIndex",
        instructions=instructions,
        tools=build_openai_tools(client, doc_ids=doc_ids),
        model=_openai_model(protocol, model_name),
        model_settings=ModelSettings(temperature=temperature, top_p=top_p),
    )


def _validate_max_turns(max_turns) -> None:
    if max_turns is not None and (not isinstance(max_turns, int)
                                  or max_turns < 1):
        raise PageIndexAPIError("max_turns must be a positive integer.")


def _conversation_group_id(model_name: str, instructions: str, items) -> str:
    """Stable per-conversation cache-routing key: openai-agents hashes
    RunConfig.group_id into the OpenAI prompt_cache_key, and without one it
    stamps every run with a fresh key, tagging a round-tripped prefix as a
    different cache group. Keyed on the prefix identity — model,
    instructions, first conversation item — so a conversation's
    continuations share one route without pooling unrelated conversations.
    Callers pass the conversation's own items, never the SDK-prepended
    doc-targeting block: that block is byte-identical for every
    conversation about a document and would pool them all under one key."""
    seed = json.dumps([model_name, instructions,
                       items[0] if items else None],
                      sort_keys=True, default=str)
    return "pageindex-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _run_kwargs(max_turns, group_id: str) -> dict:
    # Managed runs never export traces — the caller opted into document QA,
    # not telemetry.
    from agents import RunConfig
    kwargs: dict = {"run_config": RunConfig(tracing_disabled=True,
                                            group_id=group_id)}
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
        return response

    responses.create = recording_create


async def _aclose_backend(agent) -> None:
    """Close the per-call AsyncOpenAI client before its event loop ends —
    otherwise httpx tears down pooled connections on a closed loop and
    emits 'Task exception was never retrieved' noise."""
    backend = getattr(getattr(agent, "model", None), "_client", None)
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


def _wrap_max_turns(exc, max_turns) -> PageIndexAPIError:
    limit = max_turns if max_turns is not None else "the default limit"
    return PageIndexAPIError(
        f"The agent did not finish within max_turns ({limit}). Raise "
        "max_turns, or narrow the question."
    )


def _openai_usage(raw_responses) -> dict:
    prompt = sum(r.usage.input_tokens for r in raw_responses)
    completion = sum(r.usage.output_tokens for r in raw_responses)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def run_chat_completions(client, messages, stream: bool = False,
                         doc_id=None, temperature: Optional[float] = None,
                         stream_metadata: bool = False,
                         enable_citations: bool = False,
                         model: Optional[str] = None,
                         max_turns: Optional[int] = None,
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
    model_name = model or client.retrieve_model
    managed = _managed_instructions(system_texts)
    agent = _openai_agent(client, "chat", model_name, managed,
                          temperature, None, doc_ids=doc_id)
    run_kwargs = _run_kwargs(max_turns,
                             _conversation_group_id(model_name, managed,
                                                    history))
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    if not stream:
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=items, **run_kwargs)))
        except MaxTurnsExceeded as exc:
            raise _wrap_max_turns(exc, max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": result.final_output or ""},
                "finish_reason": "stop",
            }],
            "usage": _openai_usage(result.raw_responses),
        }

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish=None) -> dict:
        return {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": model_name,
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
            raise _wrap_max_turns(exc, max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        finally:
            if not completed and hasattr(streamed, "cancel"):
                streamed.cancel()  # abandoned/failed: stop the agent task
            await _aclose_backend(agent)
        yield chunk({}, finish="stop")
        yield {
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": model_name, "choices": [],
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
    model_name = model or client.retrieve_model
    managed = _managed_instructions(extra)
    agent = _openai_agent(client, "responses", model_name, managed,
                          temperature, top_p, doc_ids=doc_id)
    run_kwargs = _run_kwargs(max_turns,
                             _conversation_group_id(model_name, managed,
                                                    conversation))
    recorded: dict = {}
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded

    def envelope(output: list, raw_responses) -> dict:
        usage = _openai_usage(raw_responses)
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "model": model_name,
            "status": recorded.get("status") or "completed",
            "output": output,
            "usage": {"input_tokens": usage["prompt_tokens"],
                      "output_tokens": usage["completion_tokens"],
                      "total_tokens": usage["total_tokens"]},
            "instructions": managed,
            "tools": [{"type": "function", "name": tool.name,
                       "description": tool.description,
                       "parameters": tool.params_json_schema,
                       "strict": getattr(tool, "strict_json_schema", True)}
                      for tool in agent.tools],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": None,
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
            raise _wrap_max_turns(exc, max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        output = result.to_input_list()[len(items):]
        return envelope(output, result.raw_responses)

    # One logical response per call: per-turn backend lifecycle events
    # (created/completed/...) are collapsed — forwarding them verbatim would
    # end a canonical consumer at the first turn — and sequence numbers are
    # reassigned monotonically across the whole run.
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
        # re-based by the count of items already committed by prior turns
        # — and the tool outputs the SDK injects between turns take the
        # next slot on that same axis.
        output_offset = 0
        completed = False
        try:
            async for event in streamed.stream_events():
                if event.type == "raw_response_event":
                    data = event.data.model_dump(exclude_unset=True)
                    if data.get("type") in lifecycle:
                        if data["type"] in ("response.completed",
                                            "response.incomplete",
                                            "response.failed"):
                            # Per-turn terminal state; the last turn's wins
                            # and feeds the final envelope below.
                            state = data.get("response") or {}
                            for field in ("status", "incomplete_details",
                                          "error"):
                                recorded[field] = state.get(field)
                            output_offset += len(state.get("output") or [])
                        continue
                    if isinstance(data.get("output_index"), int):
                        data["output_index"] += output_offset
                    sequence += 1
                    data["sequence_number"] = sequence
                    yield data
                elif (event.type == "run_item_stream_event"
                        and event.item.type == "tool_call_output_item"):
                    # We are the tool executor, so we emit the output item
                    # the way the platform streams its own server-side tools.
                    sequence += 1
                    yield {"type": "response.output_item.done",
                           "output_index": output_offset,
                           "sequence_number": sequence,
                           "item": dict(event.item.to_input_item())}
                    output_offset += 1
            completed = True
        except MaxTurnsExceeded as exc:
            raise _wrap_max_turns(exc, max_turns) from exc
        except AgentsException as exc:
            raise PageIndexAPIError(
                f"The agent backend failed: {exc}") from exc
        finally:
            if not completed and hasattr(streamed, "cancel"):
                streamed.cancel()  # abandoned/failed: stop the agent task
            await _aclose_backend(agent)
        output = streamed.to_input_list()[len(items):]
        sequence += 1
        status = recorded.get("status") or "completed"
        terminal = {"incomplete": "response.incomplete",
                    "failed": "response.failed"}.get(status,
                                                     "response.completed")
        yield {"type": terminal, "sequence_number": sequence,
               "response": envelope(output, streamed.raw_responses)}

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
            "messages in local mode requires anthropic >= 0.84.0 (the tool "
            "runner with ToolError) — pip install -U anthropic."
        ) from exc


def _anthropic_client():
    """The backend client — the seam tests replace with a fake transport."""
    import anthropic
    return anthropic.Anthropic()


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


def _default_max_tokens(model: str) -> int:
    """The wire-required per-turn budget when the caller sets none: 8192,
    except the claude-3 generation whose output ceiling is 4096."""
    return 4096 if model.startswith(_CLAUDE_4096_MODELS) else 8192


def run_messages(client, messages, model: str,
                 max_tokens: Optional[int] = None,
                 stream: bool = False, doc_id=None, system=None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 stop_sequences: Optional[list[str]] = None,
                 max_turns: Optional[int] = None,
                 ) -> Union[dict, Iterator[Any]]:
    from .integrations.anthropic_sdk import build_anthropic_tools

    _require_anthropic()
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
        "stop_sequences": stop_sequences,
    }.items() if value is not None}
    runner = _anthropic_client().beta.messages.tool_runner(
        max_tokens=(max_tokens if max_tokens is not None
                    else _default_max_tokens(model)),
        messages=prepared,
        model=model,
        tools=build_anthropic_tools(client, doc_ids=doc_id),
        system=_anthropic_system(system, block),
        stream=stream,
        # Bounded like the OpenAI surfaces (their framework default is 10).
        max_iterations=max_turns if max_turns is not None else 10,
        **passthrough,
    )

    if stream:
        def events() -> Iterator[Any]:
            for turn_stream in runner:
                for event in turn_stream:
                    yield event
        return events()

    turns = [turn for turn in runner]
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
    final_ids = {block["id"] for block in final_blocks
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
