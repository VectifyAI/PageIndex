"""Managed local chat: document-QA agents over the local tools.

Three methods, three wire protocols, 1:1 with the backend and no translation
layer: ``chat_completions`` drives the backend's /chat/completions (any
OpenAI-compatible backend, final answer only), ``responses`` drives
/responses (process items are standard output; round-trip them for provider
prompt-cache continuation and agent memory), ``messages`` drives Anthropic's
/v1/messages via the SDK's own tool runner (tool_use/tool_result round-trip
is the format's native behavior).

Content passes through untouched — the caller's messages, the model's
answers, tool outputs, finish/stop reasons. The SDK owns only gatekeeping
(structural validation), table-setting (managed instructions, tools, doc
targeting), tool execution, and billing (usage aggregation, envelope ids).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
import uuid
from typing import Any, Iterator, Optional, Union

from .agent_tools import (AGENT_INSTRUCTIONS, _local_description,
                          _local_schema, call_tool, doc_targeting_block,
                          tool_names)
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
    return doc_targeting_block(client, doc_id)


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
    """Drive an async generator from a background thread; yield synchronously."""
    items: "queue.Queue[Any]" = queue.Queue()

    def pump():
        async def consume():
            async for item in agen_factory():
                items.put(item)

        try:
            asyncio.run(consume())
        except BaseException as exc:  # re-raised on the consumer thread
            items.put(exc)
            return
        items.put(_SENTINEL)

    threading.Thread(target=pump, daemon=True).start()
    while True:
        item = items.get()
        if item is _SENTINEL:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


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
    """The backend protocol driver — the seam tests replace with a fake."""
    from openai import AsyncOpenAI
    if protocol == "chat":
        from agents.models.openai_chatcompletions import (
            OpenAIChatCompletionsModel)
        return OpenAIChatCompletionsModel(model_name, AsyncOpenAI())
    from agents.models.openai_responses import OpenAIResponsesModel
    return OpenAIResponsesModel(model_name, openai_client=AsyncOpenAI())


def _openai_agent(client, protocol: str, model_name: str, instructions: str,
                  temperature, top_p):
    from agents import Agent, ModelSettings
    from .integrations.openai_agents import build_openai_tools
    return Agent(
        name="PageIndex",
        instructions=instructions,
        tools=build_openai_tools(client),
        model=_openai_model(protocol, model_name),
        model_settings=ModelSettings(temperature=temperature, top_p=top_p),
    )


def _run_kwargs(max_turns) -> dict:
    # Managed runs never export traces — the caller opted into document QA,
    # not telemetry.
    from agents import RunConfig
    kwargs: dict = {"run_config": RunConfig(tracing_disabled=True)}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    return kwargs


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
    _require_openai_agents("chat_completions")
    if enable_citations:
        raise PageIndexAPIError(
            "enable_citations is cloud-only — citations need block-level OCR "
            "data that local mode does not store."
        )
    system_texts, history = _split_chat_messages(messages)
    block = _doc_block(client, doc_id)
    items = ([{"role": "user", "content": block}] if block else []) + history
    model_name = model or client.retrieve_model
    agent = _openai_agent(client, "chat", model_name,
                          _managed_instructions(system_texts),
                          temperature, None)
    from agents import Runner
    if not stream:
        result = _run_sync(
            Runner.run(agent, input=items, **_run_kwargs(max_turns)))
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
        streamed = Runner.run_streamed(agent, input=items,
                                       **_run_kwargs(max_turns))
        first = True
        async for event in streamed.stream_events():
            if (event.type == "raw_response_event"
                    and isinstance(event.data, ResponseTextDeltaEvent)):
                if first:
                    yield chunk({"role": "assistant", "content": ""})
                    first = False
                yield chunk({"content": event.data.delta})
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
    if isinstance(input, str):
        items = [{"role": "user", "content": input}]
    elif (isinstance(input, list) and input
            and all(isinstance(item, dict) for item in input)):
        items = list(input)
    else:
        raise PageIndexAPIError("input must be a non-empty string or list "
                                "of item dicts.")
    block = _doc_block(client, doc_id)
    if block:
        items = [{"role": "user", "content": block}] + items
    extra = [instructions] if instructions else []
    model_name = model or client.retrieve_model
    agent = _openai_agent(client, "responses", model_name,
                          _managed_instructions(extra), temperature, top_p)
    from agents import Runner

    def envelope(output: list, raw_responses) -> dict:
        usage = _openai_usage(raw_responses)
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "model": model_name,
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": usage["prompt_tokens"],
                      "output_tokens": usage["completion_tokens"],
                      "total_tokens": usage["total_tokens"]},
        }

    if not stream:
        result = _run_sync(
            Runner.run(agent, input=[dict(item) for item in items],
                       **_run_kwargs(max_turns)))
        output = result.to_input_list()[len(items):]
        return envelope(output, result.raw_responses)

    async def agen():
        streamed = Runner.run_streamed(agent,
                                       input=[dict(item) for item in items],
                                       **_run_kwargs(max_turns))
        async for event in streamed.stream_events():
            if event.type == "raw_response_event":
                yield event.data.model_dump(exclude_unset=True)
            elif (event.type == "run_item_stream_event"
                    and event.item.type == "tool_call_output_item"):
                # We are the tool executor, so we emit the output item the
                # way the platform streams its own server-side tools.
                yield {"type": "response.output_item.done",
                       "item": dict(event.item.to_input_item())}
        output = streamed.to_input_list()[len(items):]
        yield {"type": "response.completed",
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


def _anthropic_client():
    """The backend client — the seam tests replace with a fake transport."""
    import anthropic
    return anthropic.Anthropic()


def _runnable_tools(client) -> list:
    from anthropic import beta_tool

    def make(name: str):
        def _fn(**kwargs: Any) -> str:
            return call_tool(client, name, kwargs)[0]

        _fn.__name__ = name
        return beta_tool(_fn, name=name, description=_local_description(name),
                         input_schema=_local_schema(name))

    return [make(name) for name in tool_names()]


def _anthropic_system(extra_system, block: Optional[str]) -> list[dict]:
    """System blocks with cache_control on the stable managed prefix; the
    doc block and caller system content follow as their own blocks."""
    blocks = [{"type": "text",
               "text": CHAT_HEADER + "\n\n" + AGENT_INSTRUCTIONS,
               "cache_control": {"type": "ephemeral"}}]
    if block:
        blocks.append({"type": "text", "text": block,
                       "cache_control": {"type": "ephemeral"}})
    if extra_system is None:
        return blocks
    if isinstance(extra_system, str):
        return blocks + [{"type": "text", "text": extra_system}]
    if isinstance(extra_system, list):
        return blocks + list(extra_system)
    raise PageIndexAPIError("system must be a string or a list of blocks.")


def _anthropic_usage(turns) -> dict:
    fields = ("input_tokens", "output_tokens",
              "cache_creation_input_tokens", "cache_read_input_tokens")
    totals = {field: 0 for field in fields}
    for turn in turns:
        for field in fields:
            value = getattr(turn.usage, field, None)
            if isinstance(value, int):
                totals[field] += value
    return totals


def run_messages(client, messages, model: str, max_tokens: int,
                 stream: bool = False, doc_id=None, system=None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 stop_sequences: Optional[list[str]] = None,
                 max_turns: Optional[int] = None,
                 ) -> Union[dict, Iterator[Any]]:
    _require_anthropic()
    if not isinstance(messages, list) or not messages:
        raise PageIndexAPIError("messages must be a non-empty list.")
    block = _doc_block(client, doc_id)
    prepared = [dict(message) for message in messages]
    passthrough = {key: value for key, value in {
        "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "stop_sequences": stop_sequences,
    }.items() if value is not None}
    runner = _anthropic_client().beta.messages.tool_runner(
        max_tokens=max_tokens,
        messages=prepared,
        model=model,
        tools=_runnable_tools(client),
        system=_anthropic_system(system, block),
        stream=stream,
        **({"max_iterations": max_turns} if max_turns is not None else {}),
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
    conversation = list(captured.get("messages") or [])
    final = turns[-1]
    envelope = final.model_dump(mode="json")
    envelope["usage"] = _anthropic_usage(turns)
    # The full turn sequence (assistant tool_use + user tool_result + final),
    # valid for verbatim append to the caller's history. The runner appends
    # intermediate turns to its params but not the final assistant message.
    new_messages = conversation[len(prepared):]
    if not new_messages or new_messages[-1].get("role") != "assistant":
        new_messages = new_messages + [{
            "role": "assistant",
            "content": [block.model_dump(mode="json")
                        for block in final.content],
        }]
    envelope["messages"] = new_messages
    return envelope
