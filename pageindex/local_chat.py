"""Own-model chat: document-QA agents over the local or cloud agent
tools, and the runs behind both ChatStream views, which also weave the
managed endpoint's chunk stream."""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import queue
import re
import threading
import time
import uuid
import warnings
from typing import Any, Generator, Iterator, Mapping, Optional, Union
from urllib.parse import urlparse

from .agent_tools import _base_instructions, targeting_block
from .chat_stream import (ChatStream, MessagesStream, ChatObject,
                          _yaml_as_dict)
from .errors import PageIndexAPIError, _pageindex_cause

_yaml_as_dict()  # every ChatObject subclass lives in this module

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


def _text_reply(message: dict) -> bool:
    """An assistant turn of text alone: no tool calls, no refusal."""
    return (isinstance(message.get("content"), str)
            and not message.get("tool_calls") and not message.get("refusal"))


def _assistant_items(message) -> list:
    """An assistant turn as the Agents SDK's items, by the SDK's own
    converter for a model reply: the items its run held, from the model it
    names."""
    from openai.types.chat import ChatCompletionMessage
    from pydantic import ValidationError
    from agents.models.chatcmpl_converter import Converter
    try:
        parsed = ChatCompletionMessage.model_validate(message)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise PageIndexAPIError(
            f"Invalid assistant message: {error['msg']} at "
            f"{'.'.join(map(str, error['loc']))}.") from exc
    writer = message.get(_WRITER)
    items = [item.model_dump(exclude_unset=True)
             for item in Converter.message_to_output_items(
                 parsed, provider_data={"model": writer}
                 if isinstance(writer, str) else None)]
    if any(item.get("type") != "reasoning" for item in items):
        return items
    if isinstance(message.get("content"), str):
        return [{"role": "assistant", "content": message["content"]}]
    raise PageIndexAPIError(
        "An assistant message needs content, a refusal or tool_calls.")


def _split_chat_messages(messages, text_only: Optional[str] = None,
                         ) -> "tuple[list[str], list[dict]]":
    """Validate the chat_completions surface's messages: system/developer
    content joins the managed instructions; user/assistant text passes
    through, a text reply's other fields coming back through the fill (as
    a reply item, the SDK would fold the next tool call into it); assistant
    turns with tool calls or a refusal, and tool results, become the Agents
    SDK's items, so a returned ``messages`` transcript appends back
    verbatim. ``text_only`` names a model that takes text alone
    (claude-code/): its errors say so and point to what takes more."""
    lanes = 'chat(protocol="responses") or chat(protocol="messages")'
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
        elif (role == "assistant" and not text_only
              and not _text_reply(message)):
            history.extend(_assistant_items(message))
        elif role == "tool" and not text_only:
            call_id = message.get("tool_call_id")
            output = message.get("content")
            if (not isinstance(call_id, str)
                    or not isinstance(output, (str, list))):
                raise PageIndexAPIError(
                    "A tool message needs a string tool_call_id and string "
                    "or list content.")
            # the SDK reads Chat Completions content parts as they are
            history.append({"type": "function_call_output",
                            "call_id": call_id, "output": output})
        elif role in ("user", "assistant"):
            content = message.get("content")
            if not isinstance(content, str):
                raise PageIndexAPIError(
                    f"{text_only} takes text messages only: content must be "
                    "a string. For images or other structured items, use an "
                    f"API model with {lanes}; for tool history, an API chat "
                    "model." if text_only else
                    "content must be a string on this lane; for structured "
                    f"items use {lanes}."
                )
            history.append({"role": role, "content": content})
        else:
            raise PageIndexAPIError(
                f"{text_only} takes text messages only: no {role!r} "
                "messages. For tool history, use an API chat model."
                if text_only else f"Unsupported role {role!r} on this lane."
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

class _ChatCompletion(ChatObject):
    @staticmethod
    def _parse(data):
        from openai._models import construct_type
        from openai.types.chat import ChatCompletion
        return construct_type(type_=ChatCompletion, value=data)


class _ChatCompletionChunk(ChatObject):
    @staticmethod
    def _parse(data):
        from openai._models import construct_type
        from openai.types.chat import ChatCompletionChunk
        return construct_type(type_=ChatCompletionChunk, value=data)


def _sdk_backend(backend) -> dict:
    """chat_backend for an SDK constructor: LiteLLM takes either endpoint
    spelling, the openai and anthropic SDKs only ``base_url``."""
    return {("base_url" if key == "api_base" else key): value
            for key, value in (backend or {}).items()}


# LitellmModel rebuilds each reply into the Agents SDK's items and back,
# dropping what it doesn't convert (signed or encrypted reasoning, thought
# signatures, ...): every request of a chat run gets that back, unchanged,
# for the model that wrote it. The hook fills only the requests listed
# here, by their LiteLLM call id. A test fails once the Agents SDK keeps
# those fields itself; the fill can go then.
_REQUESTS: dict = {}
_FILL_LOCK = threading.Lock()
_fill_hook = None
# Returned assistant messages name the model that wrote them.
_WRITER = "model"
# What the Agents SDK writes itself, with the writer's name, a tool
# call's identity, and the stream delta fields LiteLLM's own assembler
# keeps.
_REBUILT = frozenset({"role", "content", "tool_calls", _WRITER})
_CALL_IDENTITY = frozenset({"id", "type", "function", "index"})
_ASSEMBLED = frozenset({"content", "role", "tool_calls", "function_call",
                        "refusal", "audio", "reasoning_content",
                        "thinking_blocks", "provider_specific_fields",
                        "annotations", "images"})


def _reply_fields(message: dict, nested: bool = False) -> dict:
    """A reply's fields as LiteLLM parsed them, empty ones too; a field it
    lacks is None. LiteLLM's provider_specific_fields stay nested where its
    own adapter reads them, and go on top for an OpenAI-protocol host."""
    fields = {key: value for key, value in message.items()
              if key != "provider_specific_fields" and value is not None}
    provider = message.get("provider_specific_fields") or {}
    if nested and provider:
        fields["provider_specific_fields"] = provider
    elif not nested:
        for key, value in provider.items():
            if value is not None:
                fields.setdefault(key, value)
    return fields


def _nth(records: dict, key, seen: dict):
    """The record for this request's next message under key: records keep
    conversation order, so a repeated answer or reused call id still
    finds its own reply."""
    n = seen[key] = seen.get(key, -1) + 1
    found = records.get(key) or ()
    return found[n] if n < len(found) else None


class _Replies:
    """A chat run's replies as their model returned them, and its latest
    request as sent."""

    def __init__(self, model: str, nested: bool = False):
        self.model = model
        self.nested = nested  # LiteLLM's adapter reads provider fields itself
        self.messages: dict = {}  # ("call", first call id) | ("text", text)
        self.calls: dict = {}     # call id -> the call's own fields
        self.streams: list = []   # (chunks, raw delta parts) to assemble
        self.first: Optional[int] = None  # messages in the first request
        self.last: list = []
        self.request: Optional[str] = None  # the latest filled call id
        self.missed = False  # a request went out without the fill
        self.sent: set = set()  # latest request's fields a 400 may refuse
        self.refused: set = set()  # fields a 400 refused in this call

    def _call_id(self, call: dict) -> str:
        # LiteLLM folds a Gemini signature into the id; the SDK strips it
        from agents.models.chatcmpl_helpers import ChatCmplHelpers
        return ChatCmplHelpers.clean_gemini_tool_call_id(
            call.get("id") or "", self.model)

    def record(self, message: dict, own: bool = True) -> None:
        """A reply's fields under its first call id or its text; another
        model's message, or one this SDK didn't return, holds its place
        with nothing to put back."""
        fields = _reply_fields(message, self.nested)
        calls = [call for call in fields.get("tool_calls") or ()
                 if isinstance(call, dict)]
        for call in calls:
            self.calls.setdefault(self._call_id(call), []).append(
                {key: value for key, value in call.items()
                 if own and key not in _CALL_IDENTITY and value is not None})
        key = (("call", self._call_id(calls[0])) if calls
               else ("text", fields["content"])
               if isinstance(fields.get("content"), str) else None)
        if key is not None:
            self.messages.setdefault(key, []).append(
                {k: v for k, v in fields.items()
                 if own and k not in _REBUILT})

    def _assemble(self) -> None:
        """Streamed replies: LiteLLM's assembler over its parsed chunks,
        plus the raw delta fields its parser drops."""
        import litellm
        while self.streams:
            chunks, raw = self.streams.pop(0)
            try:
                built = (litellm.stream_chunk_builder(chunks)
                         if chunks else None)
            except Exception:  # LiteLLM's APIError: nothing to put back
                built = None
            choices = getattr(built, "choices", None)
            reply = getattr(choices[0], "message", None) if choices else None
            if reply is None:
                continue
            message = reply.model_dump()
            nested = message.get("provider_specific_fields") or {}
            for key, parts in raw.items():
                if message.get(key) is None and not nested.get(key):
                    message[key] = ("".join(parts) if isinstance(parts[0], str)
                                    else [x for part in parts for x in part])
            self.record(message)

    def _put(self, target: dict, fields, own: frozenset) -> dict:
        """Put back what the reply had, as it had it, then drop what a
        provider refused; what's left beside ``own`` counts as sent."""
        target.update(fields or {})
        for key in [key for key in target if key not in own]:
            if key in self.refused:
                del target[key]
            else:
                self.sent.add(key)
        return target

    def fill(self, messages: list) -> list:
        """Each assistant message with its own model's reply fields as the
        reply had them, over what the Agents SDK wrote in their place (its
        Gemini signature placeholder): a reply it split per call fills its
        first message, each call by id. Every assistant message has
        content, as the SDK writes it for history, so a split reply's later
        messages read the same on the next turn."""
        self._assemble()
        self.sent = set()
        seen: dict = {}
        out = []
        for message in messages:
            if not (isinstance(message, dict)
                    and message.get("role") == "assistant"):
                out.append(message)
                continue
            message = {"role": "assistant", "content": None, **message}
            calls = message.get("tool_calls") or []
            ids = [self._call_id(call) for call in calls]
            content = message.get("content")
            key = (("call", ids[0]) if calls
                   else ("text", content) if isinstance(content, str)
                   else None)
            self._put(message, _nth(self.messages, key, seen), _REBUILT)
            if calls:
                message["tool_calls"] = [
                    self._put(dict(call), _nth(self.calls, call_id, seen),
                              _CALL_IDENTITY)
                    for call, call_id in zip(calls, ids)]
            out.append(message)
        return out


def _sending(data) -> bool:
    """Encrypted reasoning that a run's request carries right now, as the
    fill put it back."""
    return bool(data) and any(
        isinstance(block, dict) and block.get("data") == data
        for replies in tuple(_REQUESTS.values())
        for message in replies.last if isinstance(message, dict)
        for block in message.get("thinking_blocks") or ())


def _keep_redacted(translate):
    """LiteLLM's Converse thinking converter, except that a run's encrypted
    reasoning, which it turns into a blank text its assembly then drops,
    goes back as Bedrock's redactedContent, as Bedrock returned it."""
    @functools.wraps(translate)
    def converting(thinking_blocks, *args, **kwargs):
        out = []
        for block in thinking_blocks:
            converted = translate([block], *args, **kwargs)
            blank = converted == [
                {"reasoningContent": {"reasoningText": {"text": ""}}}]
            if (blank and block.get("type") == "redacted_thinking"
                    and _sending(block.get("data"))):
                converted = [{"reasoningContent": {
                    "redactedContent": block["data"]}}]
            out += converted
        return out
    return converting


def _register_fill() -> None:
    """The LiteLLM pre-request hook that fills a run's requests, in
    litellm.callbacks: added before each request, again if a caller
    replaced the list. The first call also wraps LiteLLM's Converse
    thinking converter (_keep_redacted)."""
    global _fill_hook
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    with _FILL_LOCK:
        if _fill_hook is None:
            name = "translate_thinking_blocks_to_reasoning_content_blocks"
            try:
                from litellm.litellm_core_utils.prompt_templates.factory \
                    import BedrockConverseMessagesProcessor as converse
                translate = getattr(converse, name)
            except (ImportError, AttributeError):
                pass  # LiteLLM moved it: Converse drops it, as before
            else:
                setattr(converse, name,
                        staticmethod(_keep_redacted(translate)))
            class _Fill(CustomLogger):
                async def async_pre_call_deployment_hook(self, kwargs,
                                                         call_type):
                    replies = _REQUESTS.get(kwargs.get("litellm_call_id"))
                    if (replies is None
                            or not isinstance(kwargs.get("messages"), list)):
                        return None
                    messages = replies.fill(kwargs["messages"])
                    if (kwargs.get("cache_control_injection_points")
                            and _gateway(str(kwargs.get("model") or ""),
                                         kwargs.get("base_url"))):
                        if "cache_control" in replies.refused:
                            kwargs = {key: value for key, value
                                      in kwargs.items() if key
                                      != "cache_control_injection_points"}
                        else:
                            # A gateway keeps a mark only inside a content
                            # block and merges adjacent text messages, so
                            # user text goes as a block on every request.
                            messages = [
                                {**m, "content": [{"type": "text",
                                                   "text": m["content"]}]}
                                if m.get("role") == "user"
                                and isinstance(m.get("content"), str)
                                and m["content"] else m
                                for m in messages]
                            replies.sent.add("cache_control")
                    if replies.first is None:
                        replies.first = len(messages)
                    replies.last = messages
                    replies.request = kwargs["litellm_call_id"]
                    return {**kwargs, "messages": messages}
            _fill_hook = _Fill()
        if _fill_hook not in litellm.callbacks:
            litellm.callbacks.append(_fill_hook)


def _recording(create, replies: _Replies):
    """LiteLLM's per-chunk parser, observed: what it returns assembles the
    reply, and the raw delta fields beside it keep what its filter drops.
    A thinking block's closing repeat of its text is blanked here, for the
    Agents SDK and the record alike."""
    chunks: list = []
    raw: dict = {}  # key -> its fragments, all str or all list
    thinking: list = []
    replies.streams.append((chunks, raw))

    def creating(chunk):
        choices = getattr(chunk, "choices", None) or ()
        delta = getattr(choices[0], "delta", None) if choices else None
        dump = getattr(delta, "model_dump", None)
        fields = delta if isinstance(delta, dict) else dump() if dump else {}
        for key, value in fields.items():
            if key in _ASSEMBLED or not isinstance(value, (str, list)):
                continue
            parts = raw.setdefault(key, [])
            if not parts or type(parts[0]) is type(value):
                parts.append(value)
        processed = create(chunk=chunk)
        if processed is not None:
            _drop_repeated_thinking(processed, thinking)
            chunks.append(processed)
        return processed
    return creating


async def _again_with_published_models():
    from agents import RetryDecision
    from .utils import learn_published_models
    if await learn_published_models():
        return RetryDecision(retry=True, delay=0)
    return False


def _chat_retry(replies: _Replies):
    """Once again after LiteLLM refused the request itself, with the
    entries its packaged model map lacks taken from the published one (a
    model newer than the package). Again at once without the fields a 400
    names, of those the request had that may be refused, for the rest of
    the call: each retry refuses what its 400 named. Anything else by the
    SDK's one retry rule, within retry.CHAT_RETRIES."""
    from agents import ModelRetrySettings, RetryDecision
    from litellm.exceptions import UnsupportedParamsError
    from .retry import CHAT_RETRIES, agent_policy
    learned = False

    def policy(context):
        nonlocal learned
        if context.normalized.status_code in (400, 422):
            if (isinstance(context.error, UnsupportedParamsError)
                    and not learned):
                learned = True
                return _again_with_published_models()
            text = str(context.error)
            named = {key for key in replies.sent
                     if re.search(rf"\b{re.escape(key)}\b", text)}
            if named:
                replies.refused.update(named)
                return RetryDecision(retry=True, delay=0)
        return agent_policy(context)
    return ModelRetrySettings(max_retries=CHAT_RETRIES, policy=policy)


def _keep_reply_fields(agent, messages) -> Optional[_Replies]:
    """Records the caller's history and each reply around LitellmModel's
    _fetch_response; the hook fills each request it sends, known by the
    call id it passes LiteLLM."""
    from agents.extensions.models.litellm_model import LitellmModel
    model = agent.model
    if not isinstance(model, LitellmModel):
        return None
    fetch = model._fetch_response
    replies = _Replies(model.model, nested=not _openai_protocol(model.model))
    for message in messages:
        if not (isinstance(message, dict)
                and message.get("role") == "assistant"):
            continue
        if isinstance(message.get(_WRITER), str) or not _text_reply(message):
            replies.record(message, own=message.get(_WRITER) == model.model)
        else:  # it names no model: what the Agents SDK itself replays
            replies.record(_as_sent(agent, _assistant_items(message))[-1])

    async def fetching(system_instructions, input, model_settings, *args,
                       **kwargs):
        call_id = str(uuid.uuid4())
        model_settings = dataclasses.replace(model_settings, extra_args={
            **(model_settings.extra_args or {}), "litellm_call_id": call_id})
        _register_fill()
        _REQUESTS[call_id] = replies
        try:
            result = await fetch(system_instructions, input, model_settings,
                                 *args, **kwargs)
        finally:
            del _REQUESTS[call_id]
        if replies.request != call_id:
            replies.missed = True
        if not isinstance(result, tuple):
            if result.choices:
                replies.record(result.choices[0].message.model_dump())
            return result
        create = getattr(result[1], "chunk_creator", None)
        if create is not None:  # LiteLLM's CustomStreamWrapper
            setattr(result[1], "chunk_creator", _recording(create, replies))
        return result

    model._fetch_response = fetching
    agent.model_settings = dataclasses.replace(
        agent.model_settings, retry=_chat_retry(replies))
    return replies


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
    except ImportError as exc:
        raise PageIndexAPIError(
            f"'{model_name}' routes through LiteLLM, which failed to "
            f"import: {exc}") from exc
    from .utils import (_litellm_model, _mute_litellm_bridge_usage_warning,
                        _quiet_litellm)
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
        return _injected_marks()
    return None


def _injected_marks() -> dict:
    # The stable prefix plus the newest message, so each turn re-reads
    # the turns before it. LiteLLM seeds nothing unprompted, so this
    # pair is the marks' sole source.
    return {"cache_control_injection_points": [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1}]}


def _gateway(wire: str, base_url=None) -> bool:
    """A relay serving other vendors' models: the OpenAI protocol to a
    host that isn't OpenAI's own (with no base URL passed, the one LiteLLM
    reads from OPENAI_BASE_URL or OPENAI_API_BASE), or LiteLLM's
    OpenRouter adapter."""
    if "/" not in wire or wire.startswith("openai/"):
        base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("OPENAI_API_BASE"))
        host = urlparse(base_url).hostname if base_url else None
        return bool(host) and not host.endswith(".openai.com")
    return wire.startswith("openrouter/")


def _cache_extra_args(model_name: str,
                      base_url: Optional[str] = None) -> Optional[dict]:
    """The chat lane's marks: normalized exactly as _litellm_model
    normalizes the wire (bare names get openai/), so this predicate
    cannot disagree with where chat_completions actually routes. Claude
    caches only on marks, so it gets them on every route, a gateway too;
    every other model caches by itself, and on OpenAI's own API marks
    turn on explicit caching, which loses its automatic hits."""
    wire = model_name.removeprefix("litellm/")
    if _gateway(wire, base_url):
        return _injected_marks() if "claude" in wire.lower() else None
    if "/" not in wire or wire.startswith("openai/"):
        return None
    return _litellm_claude_marks(wire)


def _openai_protocol(model_name: str) -> bool:
    """Destinations that speak the OpenAI protocol on the wire, where
    prompt_cache_key means something and extra_body lands in the request
    body. Resolution is LiteLLM's own (same as _cache_extra_args), so the
    answer follows actual routing; azure and openrouter ride the OpenAI
    protocol without appearing in openai_compatible_providers, and
    azure_ai's Claude rides Anthropic's although azure_ai appears."""
    wire = model_name.removeprefix("litellm/")
    if "/" not in wire or wire.startswith("openai/"):
        return True
    try:
        import litellm
        from litellm.constants import openai_compatible_providers
        from .utils import _quiet_litellm
        _quiet_litellm()
        model, provider, _, _ = litellm.get_llm_provider(model=wire)
    except Exception:
        return False
    if provider == "azure_ai" and "claude" in model.lower():
        return False
    return (provider in ("openai", "azure", "openrouter")
            or provider in openai_compatible_providers)


def _merged_backend(client, backend):
    """This call's connection overrides: the client's ``chat_backend``
    under the per-call dict, per-call keys winning."""
    merged = {**(getattr(client, "chat_backend", None) or {}),
              **(backend or {})}
    return merged or None


_SKELETON_KEYS = frozenset({"system", "instructions", "input", "messages",
                            "tools"})
_ARGUMENT_KEYS = {"stream": "stream=", "doc_id": "doc_id=",
                  "folder_id": "folder_id=", "model": "model=",
                  "reasoning_effort":
                      "reasoning_effort= (own chat models only)",
                  **dict.fromkeys(("api_key", "base_url", "api_base"),
                                  "backend= (API chat models only)")}
# The ModelSettings fields openai-agents' LitellmModel reads.
_LITELLM_MODEL_SETTINGS = frozenset({
    "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "max_tokens", "tool_choice", "parallel_tool_calls", "reasoning",
    "top_logprobs", "include_usage", "extra_headers", "extra_query",
    "metadata"})


def _refuse_skeleton(extra_body) -> None:
    """The managed prompt, conversation and tools are the SDK's on every
    lane; extra_body merges last, so a caller's copy would replace them.
    Fields with their own argument are refused too, naming it."""
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
    hit = sorted(set(_ARGUMENT_KEYS).intersection(extra_body))
    if hit:
        use = dict.fromkeys(_ARGUMENT_KEYS[key] for key in hit)
        raise PageIndexAPIError(
            f"extra_body cannot carry {', '.join(hit)}: use "
            f"{' / '.join(use)} instead.")


def _openai_agent(client, protocol: str, model_name: str, instructions: str,
                  temperature, top_p, doc_ids=None, cache_key=None,
                  reasoning=None, reasoning_effort=None, extra_body=None,
                  max_tokens=None, backend=None, extra_headers=None):
    _refuse_skeleton(extra_body)
    from agents import Agent, ModelSettings
    from openai.types.shared import Reasoning
    from pydantic import ValidationError
    from .integrations.openai_agents import build_openai_tools
    # ModelSettings.extra_body is the one channel all three engines put on
    # the wire verbatim: LiteLLM and both OpenAI model classes. OpenAI
    # destinations only — LiteLLM plants extra_body as a literal field in
    # other providers' request bodies, which Anthropic rejects as unknown.
    # The responses lane is the OpenAI SDK whatever the name says.
    openai_backend = protocol == "responses" or _openai_protocol(model_name)
    conn = _sdk_backend(backend)
    # LiteLLM's own retries off: the runner's (_chat_retry) are the chat lane's one layer.
    extra_args = ({"max_retries": 0,
                   **(_cache_extra_args(model_name, conn.get("base_url")) or {})}
                  if protocol == "chat" else None)
    if reasoning_effort:
        try:
            reasoning = Reasoning(effort=reasoning_effort)
        except ValidationError:
            extra_args = {**(extra_args or {}),
                          "reasoning_effort": reasoning_effort}
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
            routed = {k: v for k, v in extra_body.items()
                      if k in _LITELLM_MODEL_SETTINGS}
            given = routed.get("reasoning")
            if isinstance(given, Reasoning):
                given = given.model_dump()
            if reasoning is not None and "reasoning" in routed and (
                    given is None or isinstance(given, Mapping)):
                # reasoning_effort's effort, unless extra_body names one
                routed["reasoning"] = {"effort": reasoning.effort, **{
                    k: v for k, v in (given or {}).items() if v is not None}}
            rest = {k: v for k, v in extra_body.items()
                    if k not in _LITELLM_MODEL_SETTINGS}
            if rest:
                extra_args = {**(extra_args or {}), **rest}
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
    """prompt_cache_key per conversation; items exclude the targeting block."""
    scope = [doc_id] if isinstance(doc_id, str) else doc_id
    seed = json.dumps([model_name, instructions, scope,
                       *([folder_id] if folder_id else []),
                       items[0] if items else None],
                      sort_keys=True, default=str)
    return "pageindex-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def _model_backend_error(exc, lane: str, client=None) -> PageIndexAPIError:
    """Wrap a provider failure, appending the hint that fixes it, if any."""
    message = f"The model backend failed: {exc}"
    if lane == "chat" and "Function tools with reasoning_effort" in str(exc):
        message += (
            " — this model runs tools on the Responses lane, where LiteLLM "
            "sends it from a custom base URL only with reasoning_effort "
            "set: pass reasoning_effort, or use chat(protocol='responses') "
            "instead.")
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
    return PageIndexAPIError(message,
                             status_code=getattr(exc, "status_code", None))


def _translate_run_error(exc, max_turns, lane, client=None) -> PageIndexAPIError:
    """The uncaught-run ladder every agent door shares."""
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    if isinstance(exc, MaxTurnsExceeded):
        return _wrap_max_turns(max_turns)
    if isinstance(exc, AgentsException):
        cause = _pageindex_cause(exc)
        if cause is not None:
            # a tool failure the invoker re-raised, wrapped on its way out
            return PageIndexAPIError(str(cause), status_code=cause.status_code)
        return PageIndexAPIError(f"The agent backend failed: {exc}")
    return _model_backend_error(exc, lane, client)


def _run_kwargs(max_turns) -> dict:
    # No traces — the caller opted into QA, not telemetry.
    from agents import RunConfig
    kwargs: dict = {"run_config": RunConfig(tracing_disabled=True)}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    return kwargs


def _record_response_status(agent, recorded: dict) -> None:
    """Capture each turn's terminal Response and its status at the transport
    client: openai-agents' non-streaming path discards Response.status, so a
    final turn truncated at the output cap would otherwise report as a clean
    completion. A response cut short ends the run there, as from 0.22 and on
    the stream: before 0.22 it ran the cut turn's half-written tool calls.
    From 0.24 openai-agents raises on such a response itself
    (response_terminal_failure_error): this raise goes once the floor gets
    there. No-op for backends without an OpenAI responses resource
    (the streaming path records from lifecycle events instead)."""
    responses = getattr(getattr(getattr(agent, "model", None), "_client", None),
                        "responses", None)
    create = getattr(responses, "create", None)
    if create is None:
        return

    async def recording_create(*args, **kwargs):
        response = await create(*args, **kwargs)
        recorded["response"] = response
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
        if getattr(response, "status", None) in ("failed", "incomplete"):
            from agents.exceptions import ModelBehaviorError
            raise ModelBehaviorError(f"The response ended {response.status}.")
        return response

    responses.create = recording_create


def _cut_text(output) -> list:
    """A cut-short response's messages, for the history: its partial tool
    calls have no outputs to pair with. Each message keeps the reasoning
    item before it, as the API refuses a message without it; a dropped
    call's reasoning goes with the call."""
    output = output or []
    return [item for item, after in zip(output, output[1:] + [{}])
            if item.get("type") == "message"
            or (item.get("type") == "reasoning"
                and after.get("type") == "message")]


def _drop_repeated_thinking(chunk, streamed: list) -> None:
    """Blank a closing thinking block's text that repeats what streamed:
    LiteLLM's Anthropic stream sends the whole text again beside the
    signature. A test fails once it stops; this goes then."""
    choices = getattr(chunk, "choices", None)
    delta = getattr(choices[0], "delta", None) if choices else None
    for block in getattr(delta, "thinking_blocks", None) or ():
        if not isinstance(block, dict) or block.get("type") != "thinking":
            continue
        text = block.get("thinking") or ""
        if block.get("signature"):
            if text and text == "".join(streamed):
                block["thinking"] = ""
            streamed.clear()
        elif text:
            streamed.append(text)


def _record_chat_finish(agent, recorded: dict) -> None:
    """Record each turn's finish_reason, which openai-agents' ModelResponse
    drops (0.24 too), so a truncated or filtered final turn would report
    "stop". A test fails once ModelResponse carries it; this goes then."""
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


def _run_usage(shape, raw_responses) -> dict:
    """Cross-turn sums in one dialect, plus each model call's own usage in
    request_usage_entries (openai-agents' name for that list)."""
    counts = []  # (input, output, cache read, cache write, reasoning)
    for r in raw_responses:
        if r.usage is None:
            continue
        details_in = getattr(r.usage, "input_tokens_details", None)
        details_out = getattr(r.usage, "output_tokens_details", None)
        counts.append((r.usage.input_tokens or 0, r.usage.output_tokens or 0,
                       getattr(details_in, "cached_tokens", 0) or 0,
                       getattr(details_in, "cache_write_tokens", 0) or 0,
                       getattr(details_out, "reasoning_tokens", 0) or 0))
    usage = shape(*([sum(column) for column in zip(*counts)] or [0] * 5))
    usage["request_usage_entries"] = [shape(*call) for call in counts]
    return usage


def _chat_shape(prompt, completion, cached, cache_write, reasoning) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached,
                                      "cache_write_tokens": cache_write},
            "completion_tokens_details": {"reasoning_tokens": reasoning}}


def _responses_shape(prompt, completion, cached, cache_write,
                     reasoning) -> dict:
    return {"input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": cached,
                                     "cache_write_tokens": cache_write},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": prompt + completion}


def _openai_usage(raw_responses) -> dict:
    """chat.completions dialect."""
    return _run_usage(_chat_shape, raw_responses)


def _responses_usage(raw_responses) -> dict:
    """Responses dialect."""
    return _run_usage(_responses_shape, raw_responses)


def _usage_recorder():
    """Run hooks keeping every model response: a non-streamed run that
    raises has no result to read raw_responses from, and openai-agents
    attaches its run data only to its own exceptions, not a provider's;
    a streamed run adds a turn to raw_responses only after its tools ran."""
    from agents import RunHooks

    class Recorder(RunHooks):
        def __init__(self):
            self.responses = []

        async def on_llm_end(self, context, agent, response):
            self.responses.append(response)

    return Recorder()


def _with_usage(error: PageIndexAPIError, usage: dict) -> PageIndexAPIError:
    error.usage = usage
    return error


# the finish_reasons of an answer the model stopped short of finishing;
# the managed endpoint reports Anthropic's names for them
_CUT_SHORT = ("length", "content_filter", "max_tokens", "refusal",
              "model_context_window_exceeded")


def _cut_short_error(finish_reason: str) -> PageIndexAPIError:
    """chat()'s answer is a bare string with nowhere to mark it unfinished,
    so a cut-short one raises rather than pass as complete."""
    return PageIndexAPIError(
        f"The model stopped before finishing its answer (finish_reason "
        f"{finish_reason!r}). protocol=\"chat_completions\" returns such "
        "an answer marked by its finish_reason.")


def _text_only(chunks) -> Iterator[str]:
    """chat_completions' text stream: the answer deltas, then a cut answer
    raises, as chat(stream=True) does."""
    finish, usage = None, {}
    for piece in chunks:
        usage = piece.get("usage") or usage
        if piece.get("choices"):
            choice = piece["choices"][0]
            finish = choice.get("finish_reason") or finish
            content = (choice.get("delta") or {}).get("content")
            if content:
                yield content
    if finish in _CUT_SHORT:
        raise _with_usage(_cut_short_error(finish), usage)


def _chat_agent(client, messages, doc_id, model, temperature=None,
                top_p=None, reasoning_effort=None, extra_body=None,
                max_tokens=None, backend=None, extra_headers=None,
                folder_id=None,
                ) -> "tuple[Any, list, str, Optional[_Replies]]":
    """The chat lane's shared prologue: validated history, doc targeting,
    and the configured agent. Returns (agent, input items, model name,
    the run's replies)."""
    messages = list(messages)
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
    return agent, items, model_name, _keep_reply_fields(agent, messages)


def _as_sent(agent, items) -> list:
    """Items as Chat Completions messages, converted as LitellmModel
    converts each request."""
    from agents.models.chatcmpl_converter import Converter
    model, reasoning = agent.model, agent.model_settings.reasoning
    return Converter.items_to_messages(
        items, model=getattr(model, "model", None),
        base_url=getattr(model, "base_url", None),
        preserve_thinking_blocks=(reasoning is not None
                                  and reasoning.effort is not None),
        preserve_tool_output_all_content=True,
        should_replay_reasoning_content=getattr(
            model, "should_replay_reasoning_content", None))


def _chat_transcript(agent, new_items, replies=None) -> list:
    """The run's new items as Chat Completions messages as the model saw
    them, so a caller who appends them resends that: its last request past
    the first one's length, then the final reply, if it has a message,
    filled the same way, each assistant message naming its model. Calls
    the SDK folded into the history's last message (a refusal) come first,
    as sent, without that message's own refusal and text. With a request
    the hook didn't see (a test double, or litellm.callbacks replaced
    mid-call), converted as LitellmModel converts each request."""
    converted = _as_sent(agent, new_items)
    if replies is None or replies.first is None or replies.missed:
        return converted
    final = [m for m in converted[-1:] if m.get("role") == "assistant"]
    host = replies.last[replies.first - 1]
    folded = []
    if host.get("tool_calls"):
        folded = [{**{k: v for k, v in host.items() if k != "refusal"},
                   "content": None}]
    messages = (folded + replies.last[replies.first:] + replies.fill(
        replies.last + final)[len(replies.last):])
    return [{**m, _WRITER: replies.model} if m.get("role") == "assistant"
            else m for m in messages]


def _output_text(output) -> str:
    """A tool result for the display: a string as it is, the framework's
    structured items by their text, anything else as JSON with an image's
    base64 payload elided (a data URL, or a Claude block's source data)."""
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
        elif (kind == "image" and isinstance(item.get("source"), dict)
              and "data" in item["source"]):
            lines.append(json.dumps(
                {**item, "source": {**item["source"], "data": "..."}},
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


async def _chat_events_agen(client, agent, items, run_kwargs):
    """The chat stream's primitive: the run as typed event dicts —
    thinking/answer deltas, each tool call (arguments parsed when JSON)
    and its full result. Both ChatStream views are built on it."""
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    from openai.types.responses import (
        ResponseReasoningSummaryTextDeltaEvent,
        ResponseReasoningTextDeltaEvent, ResponseTextDeltaEvent)
    recorder = _usage_recorder()
    recorded: dict = {}
    _record_chat_finish(agent, recorded)
    streamed = Runner.run_streamed(agent, input=items, hooks=recorder,
                                   **run_kwargs)
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
        raise _with_usage(
            _translate_run_error(exc, run_kwargs.get("max_turns"), "chat",
                                 client),
            _openai_usage(recorder.responses)) from exc
    finally:
        if not completed:
            streamed.cancel()  # abandoned/failed: stop the agent task
        await _aclose_backend(agent)
    if recorded.get("finish_reason") in _CUT_SHORT:
        raise _with_usage(_cut_short_error(recorded["finish_reason"]),
                          _openai_usage(recorder.responses))


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
    finish = None
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
            if choices and choices[0].get("finish_reason"):
                finish = choices[0]["finish_reason"]
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
        if finish in _CUT_SHORT:
            raise _cut_short_error(finish)
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
        text=lambda: _weave(_cloud_chunk_events(chunks), options),
        events=("chat events are produced by the in-process agent, "
                "which the managed chat endpoint does not serve — "
                "construct the client with chat_model=... (or a chat= "
                "model) to run the agent in your process."))


def run_chat_stream(client, messages, doc_id=None, model=None,
                    reasoning_effort=None,
                    show_process: Union[bool, Mapping[str, Any]] = False,
                    max_turns=None, backend=None, extra_headers=None,
                    extra_body=None, folder_id=None,
                    ) -> ChatStream:
    """chat(stream=True): validation and the agent build run here, eagerly;
    the run itself starts when the returned stream's chosen view is first
    consumed."""
    options = (None if show_process is False or show_process is None
               else _process_options(show_process))
    _validate_max_turns(max_turns)
    if isinstance(messages, str):
        if not messages.strip():
            raise PageIndexAPIError(
                "messages must be a non-empty string or a list of "
                "message dicts.")
        messages = [{"role": "user", "content": messages}]
    from .integrations.claude_agent_sdk import is_claude_code
    if is_claude_code(model or client.chat_model):
        run = _claude_code_run(client, messages, model or client.chat_model,
                               doc_id=doc_id, folder_id=folder_id,
                               reasoning_effort=reasoning_effort,
                               max_turns=max_turns, partial=True,
                               extra_body=extra_body,
                               extra_headers=extra_headers,
                               # a configured chat_backend is for API
                               # models; only one passed here is refused
                               backend=backend)

        def claude_events():
            return _stream_sync(lambda: run(False))

        return ChatStream(text=lambda: _weave(claude_events(), options),
                          events=claude_events)
    agent, items, _, _ = _chat_agent(client, messages, doc_id, model,
                                     reasoning_effort=reasoning_effort,
                                     extra_body=extra_body, backend=backend,
                                     extra_headers=extra_headers,
                                     folder_id=folder_id)
    run_kwargs = _run_kwargs(max_turns)

    def events():
        return _stream_sync(
            lambda: _chat_events_agen(client, agent, items, run_kwargs))

    return ChatStream(text=lambda: _weave(events(), options), events=events)


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
                         ) -> Union[ChatObject, Iterator[str],
                                    Iterator[ChatObject]]:
    if enable_citations:
        raise PageIndexAPIError(
            "enable_citations needs the managed chat endpoint — "
            + ("drop the chat model configuration to use it, or "
               if getattr(client, "api_key", None) else "")
            + "cite with your own model via chat(citations=True).")
    _validate_max_turns(max_turns)
    from .integrations.claude_agent_sdk import is_claude_code
    if is_claude_code(model or client.chat_model):
        run = _claude_code_run(client, messages, model or client.chat_model,
                               doc_id=doc_id, folder_id=folder_id,
                               reasoning_effort=reasoning_effort,
                               max_turns=max_turns, partial=stream,
                               temperature=temperature, top_p=top_p,
                               max_tokens=max_tokens, extra_body=extra_body,
                               extra_headers=extra_headers,
                               # a configured chat_backend is for API
                               # models; only one passed here is refused
                               backend=backend)
        return _claude_code_completions(run, model or client.chat_model,
                                        stream, stream_metadata)
    agent, items, model_name, replies = _chat_agent(
        client, messages, doc_id, model, temperature=temperature,
        top_p=top_p, reasoning_effort=reasoning_effort,
        extra_body=extra_body, max_tokens=max_tokens, backend=backend,
        extra_headers=extra_headers, folder_id=folder_id)
    reported_model = _reported_model(model_name)
    recorded: dict = {}
    _record_chat_finish(agent, recorded)
    run_kwargs = _run_kwargs(max_turns)
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded
    if not stream:
        recorder = _usage_recorder()
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=items, hooks=recorder,
                           **run_kwargs)))
        except (MaxTurnsExceeded, AgentsException,
                openai.OpenAIError) as exc:
            raise _with_usage(
                _translate_run_error(exc, max_turns, "chat", client),
                _openai_usage(recorder.responses)) from exc
        return _ChatCompletion({
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
            "messages": _chat_transcript(
                agent, result.to_input_list()[len(items):], replies),
        })

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish=None) -> dict:
        return _ChatCompletionChunk({
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": reported_model,
            "choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish}],
        })

    async def agen():
        from openai.types.responses import ResponseTextDeltaEvent
        recorder = _usage_recorder()
        streamed = Runner.run_streamed(agent, input=items, hooks=recorder,
                                       **run_kwargs)
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
            raise _with_usage(
                _translate_run_error(exc, max_turns, "chat", client),
                _openai_usage(recorder.responses)) from exc
        finally:
            if not completed:
                streamed.cancel()  # abandoned/failed: stop the agent task
            await _aclose_backend(agent)
        yield chunk({}, finish=recorded.get("finish_reason") or "stop")
        yield _ChatCompletionChunk({
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": reported_model, "choices": [],
            "usage": _openai_usage(streamed.raw_responses),
            "messages": _chat_transcript(
                agent, streamed.to_input_list()[len(items):], replies),
        })

    if stream_metadata:
        return _stream_sync(agen)
    return _text_only(_stream_sync(agen))


class _ResponsesEnvelope(ChatObject):
    """The Responses envelope. Its transcript was ``items`` before it became
    ``messages``; the old key still reads it, with a DeprecationWarning, and
    is never stored."""

    @staticmethod
    def _parse(data):
        from openai._models import construct_type
        from openai.types.responses import Response
        return construct_type(type_=Response, value=data)

    def _key(self, key):
        if key == "items":
            warnings.warn(
                'The Responses envelope\'s "items" is now "messages"; '
                '"items" will be removed in a future release.',
                DeprecationWarning, stacklevel=3)
            return "messages"
        return key

    def __getitem__(self, key):
        return dict.__getitem__(self, self._key(key))

    def get(self, key, default=None):
        return dict.get(self, self._key(key), default)

    def __contains__(self, key):
        return dict.__contains__(self, self._key(key))


class _ResponsesEvent(ChatObject):
    @staticmethod
    def _parse(data):
        from openai._models import construct_type
        from openai.types.responses import ResponseStreamEvent
        return construct_type(type_=ResponseStreamEvent, value=data)


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
                  ) -> Union[ChatObject, Iterator[ChatObject]]:
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
    recorded: dict = {}
    import openai
    from agents import Runner
    from agents.exceptions import AgentsException, MaxTurnsExceeded

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    given = extra_body or {}

    def envelope(transcript: list, raw_responses) -> _ResponsesEnvelope:
        return _ResponsesEnvelope({
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": _reported_model(model_name),
            "status": recorded.get("status") or "completed",
            "output": [item for item in transcript
                       if item.get("type") != "function_call_output"],
            "messages": transcript,
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
            "temperature": given.get("temperature", temperature),
            "top_p": given.get("top_p", top_p),
            "reasoning": given.get("reasoning", reasoning),
            "max_output_tokens": given.get("max_output_tokens",
                                           max_output_tokens),
            "error": recorded.get("error"),
            "incomplete_details": recorded.get("incomplete_details"),
            "metadata": given.get("metadata"),
        })

    if not stream:
        _record_response_status(agent, recorded)
        recorder = _usage_recorder()
        try:
            result = _run_sync(_run_closing(agent,
                Runner.run(agent, input=[dict(item) for item in items],
                           hooks=recorder, **run_kwargs)))
        except (MaxTurnsExceeded, AgentsException,
                openai.OpenAIError) as exc:
            run_data = getattr(exc, "run_data", None)
            if (run_data is not None
                    and recorded.get("status") in ("failed", "incomplete")):
                # the run ended on a response cut short, before openai-agents'
                # hooks or items saw it; this goes once they see it
                cut = recorded["response"]
                transcript = [item.to_input_item()
                              for item in run_data.new_items]
                transcript += _cut_text(
                    cut.model_dump(exclude_unset=True)["output"])
                return envelope(transcript, recorder.responses + [cut])
            raise _with_usage(
                _translate_run_error(exc, max_turns, "responses", client),
                _responses_usage(recorder.responses)) from exc
        transcript = result.to_input_list()[len(items):]
        return envelope(transcript, result.raw_responses)

    lifecycle = {"response.created", "response.in_progress",
                 "response.completed", "response.failed",
                 "response.incomplete", "response.queued"}

    async def agen():
        from types import SimpleNamespace
        recorder = _usage_recorder()
        streamed = Runner.run_streamed(agent,
                                       input=[dict(item) for item in items],
                                       hooks=recorder, **run_kwargs)
        sequence = 0
        # output_index addresses an item's position in the logical
        # response.output (the final envelope's list). Backend events
        # carry per-turn indexes that restart at 0 each turn, so they are
        # re-based by the count of items already committed by prior turns.
        output_offset = 0
        cut_text: list = []
        dropped: list = []
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
                            # as the official stream: in_progress follows,
                            # with its own copy taken before created is
                            # handed out, so an edit to one stays in it
                            progress = {**copy.deepcopy(data),
                                        "sequence_number": sequence + 1,
                                        "type": "response.in_progress"}
                            yield _ResponsesEvent(data)
                            sequence += 1
                            yield _ResponsesEvent(progress)
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
                            if data["type"] != "response.completed":
                                cut_text = _cut_text(state.get("output"))
                            cut = getattr(getattr(event.data, "response",
                                                  None), "usage", None)
                            if data["type"] != "response.completed" and cut:
                                # openai-agents raises on a call cut short
                                # before its hooks see it; it was billed.
                                # This goes once its hooks see it.
                                recorder.responses.append(
                                    SimpleNamespace(usage=cut))
                        continue
                    if isinstance(data.get("output_index"), int):
                        data["output_index"] += output_offset
                    sequence += 1
                    data["sequence_number"] = sequence
                    yield _ResponsesEvent(data)
            completed = True
        except AgentsException as exc:
            # a run the envelope already reports failed/incomplete is done —
            # except for max_turns, which always gets its guidance
            if (isinstance(exc, MaxTurnsExceeded)
                    or recorded.get("status") not in ("failed", "incomplete")):
                raise _with_usage(
                    _translate_run_error(exc, max_turns, "responses", client),
                    _responses_usage(recorder.responses)) from exc
            # it raised before adding the cut turn to its items
            dropped = cut_text
            completed = True
        except openai.OpenAIError as exc:
            raise _with_usage(
                _translate_run_error(exc, max_turns, "responses", client),
                _responses_usage(recorder.responses)) from exc
        finally:
            if not completed:
                streamed.cancel()  # abandoned/failed: stop the agent task
            await _aclose_backend(agent)
        transcript = streamed.to_input_list()[len(items):] + dropped
        sequence += 1
        status = recorded.get("status") or "completed"
        terminal = {"incomplete": "response.incomplete",
                    "failed": "response.failed"}.get(status,
                                                     "response.completed")
        yield _ResponsesEvent({"type": terminal, "sequence_number": sequence,
                               "response": envelope(transcript,
                                                    recorder.responses)})

    return _stream_sync(agen)


# ── Anthropic engine (messages) ──

class _Message(ChatObject):
    @staticmethod
    def _parse(data):
        from anthropic._models import construct_type
        from anthropic.types.beta import BetaMessage
        return construct_type(type_=BetaMessage, value=data)


class _MessagesEvent(ChatObject):
    @staticmethod
    def _parse(data):
        from anthropic._models import construct_type
        from anthropic.lib.streaming import ParsedBetaMessageStreamEvent
        return construct_type(type_=ParsedBetaMessageStreamEvent, value=data)


def _require_anthropic() -> None:
    try:
        import anthropic
    except ImportError as exc:
        raise PageIndexAPIError(
            "chat(protocol='messages') drives your own chat model and "
            "requires the Anthropic SDK — "
            "pip install anthropic (or pip install 'pageindex[anthropic]')."
        ) from exc
    if tuple(int(n) for n in anthropic.__version__.split(".")[:2]) < (0, 122):
        raise PageIndexAPIError(
            "chat(protocol='messages') requires anthropic 0.122.0 or newer, "
            f"found {anthropic.__version__} — pip install -U anthropic.")


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
    try:
        client = getattr(anthropic, _ROUTE_CLIENTS[route])(**kwargs)
    except (anthropic.AnthropicError, ValueError, TypeError) as exc:
        # Vertex/Foundry refuse a missing region or credential right at
        # construction, each with its own type; same contract for all.
        raise PageIndexAPIError(
            f"The Anthropic backend is not configured: {exc}") from exc
    if key is not None and len(_ANTHROPIC_CLIENTS) < 8:
        # cache capped at 8 backends; the tail constructs per call.
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


def _anthropic_usage(turns) -> dict:
    """The turns' native usage dicts, given as (usage, model) pairs, merged
    field by field (None-safe): counts add up, nested count objects
    (cache_creation, server_tool_use, output_tokens_details) key by key,
    labels (service_tier, fallback_credit) keep the last value a turn set,
    and iterations — the API's per-sampling-iteration breakdown —
    concatenate, a turn that reports none counting as one iteration."""
    dumps = []
    for usage, model in turns:
        if not usage.get("iterations"):
            usage = {**usage, "iterations": [{
                "type": "message", "model": model,
                "input_tokens": usage.get("input_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or 0,
                "cache_creation_input_tokens":
                    usage.get("cache_creation_input_tokens") or 0,
                "cache_read_input_tokens":
                    usage.get("cache_read_input_tokens") or 0,
                "cache_creation": usage.get("cache_creation")}]}
        dumps.append(usage)
    if not dumps:
        return {"input_tokens": 0, "output_tokens": 0, "iterations": []}
    keys = dict.fromkeys(key for dump in dumps for key in dump)
    return {key: _merge_usage([dump.get(key) for dump in dumps])
            for key in keys}


def _is_count(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count_object(value) -> bool:
    return isinstance(value, dict) and all(
        item is None or _is_count(item) or _is_count_object(item)
        for item in value.values())


def _merge_usage(values):
    present = [value for value in values if value is not None]
    if all(map(_is_count, present)):
        return sum(present) if present else None
    if all(isinstance(value, list) for value in present):
        return [item for value in present for item in value]
    if all(map(_is_count_object, present)):
        keys = dict.fromkeys(key for value in present for key in value)
        return {key: _merge_usage([value.get(key) for value in present])
                for key in keys}
    return present[-1]


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
                 ) -> Union[ChatObject, MessagesStream]:
    from .integrations.anthropic_sdk import build_anthropic_tools

    _require_anthropic()
    import anthropic
    _validate_max_turns(max_turns)
    _refuse_skeleton(extra_body)
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
    # Bedrock's InvokeModel rejects the field for the ARN-versioned models
    # (Anthropic's legacy Bedrock page: Opus 4.6, Sonnet 4.6, Haiku 4.5 and
    # earlier), so every Bedrock request, whatever the model, puts the
    # breakpoint on each turn's tool results instead. The field can cover
    # Bedrock once those models are gone.
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
        if max_tokens is None:
            max_tokens = _default_max_tokens(
                (extra_body or {}).get("thinking", thinking))
        tool_runner = backend_client.beta.messages.tool_runner
        # anthropic 2.2+ threads the loop server-side, which proxies reject
        threadless = ({"thread": False} if "thread" in
                      inspect.signature(tool_runner).parameters else {})
        runner = tool_runner(
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
            **threadless,
        )
    except BaseException:
        # A failure before the runner handoff must not strand the transport
        # the branches below would have closed.
        if owns_transport:
            backend_client.close()
        raise
    # anthropic before 1.1 also executes tools on max_tokens turns, 1.1 and
    # later skip them: check right after the runner's own tool step.
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

    def run_usage(turns) -> dict:
        return _anthropic_usage([(turn.usage.model_dump(mode="json"),
                                  turn.model) for turn in turns
                                 if turn.usage is not None])

    def transcript(turns) -> list:
        """The full turn sequence (assistant tool_use + user tool_result +
        final), valid for verbatim append to the caller's history. Reads
        the runner's params back, which does no HTTP."""
        final = turns[-1]
        for block in moved:
            # a request's breakpoint, not history
            block.pop("cache_control", None)
        captured: dict = {}

        def capture(params):
            captured.update(params)
            return params

        runner.set_messages_params(capture)
        if not captured.get("messages"):
            # The conversation is read back through a mutator; if a vendor
            # change stops it delivering params, the envelope would silently
            # lose the tool turns — fail loudly instead.
            raise _with_usage(PageIndexAPIError(
                "Could not read the conversation back from the anthropic "
                "tool runner — the installed anthropic version is "
                "incompatible with this pageindex release."
            ), run_usage(turns))
        conversation = list(captured["messages"])
        # The runner appends a turn to its params only when it executed
        # tools from it, and which turns qualify is vendor policy that has
        # changed across anthropic releases, so stop_reason alone cannot
        # tell. Whether final's tool_use ids already sit in the history is
        # the ground truth for "already appended".
        new_messages = [_dump_message(message)
                        for message in conversation[len(prepared):]]
        final_blocks = [_dump_block(item) for item in final.content]
        # anthropic 2.2+ appends final too; it is re-added below
        if new_messages and new_messages[-1].get("content") == final_blocks:
            new_messages.pop()
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
        return new_messages

    def envelope(turns):
        final = turns[-1]
        message = _Message(final.model_dump(mode="json"))
        message["content"] = [_dump_block(item) for item in final.content]
        message["usage"] = run_usage(turns)
        message["messages"] = transcript(turns)
        return message

    if stream:
        def events() -> Generator[Any, None, Any]:
            turns: list = []  # each finished turn's message snapshot
            try:
                for turn_stream in runner:
                    for event in turn_stream:
                        yield _MessagesEvent(event.model_dump(mode="json"))
                    message = turn_stream.get_final_message()
                    if message.stop_reason is None:
                        raise PageIndexAPIError(
                            "The model backend failed: the response stream "
                            "ended before the message was complete.")
                    turns.append(message)
            except anthropic.AnthropicError as exc:
                raise _with_usage(
                    _model_backend_error(exc, "messages", client),
                    run_usage(turns)) from exc
            except PageIndexAPIError as exc:  # a tool failure or a cut stream
                raise _with_usage(exc, run_usage(turns))
            finally:
                # runs on exhaustion and abandonment (GeneratorExit) alike
                if owns_transport:
                    backend_client.close()
            return envelope(turns)
        return MessagesStream(events())

    turns: list = []
    try:
        for turn in runner:
            turns.append(turn)
    except anthropic.AnthropicError as exc:
        raise _with_usage(_model_backend_error(exc, "messages", client),
                          run_usage(turns)) from exc
    except PageIndexAPIError as exc:  # a tool failure, re-raised
        raise _with_usage(exc, run_usage(turns))
    finally:
        # safe here: the params read-back below does no HTTP
        if owns_transport:
            backend_client.close()
    if not turns:
        raise PageIndexAPIError("The model returned no response.")
    return envelope(turns)


# ── Claude Agent SDK engine (claude-code/: the user's Claude subscription) ──

_TURNS_NOTE = ("The conversation so far arrives as <turn> blocks in the user "
               "message, oldest first. Answer the last user turn.")

# Claude's stop reasons for an unfinished answer, as OpenAI finish_reasons
_CLAUDE_CUT: dict[Optional[str], str] = {"max_tokens": "length",
                                          "model_context_window_exceeded": "length",
                                          "refusal": "content_filter"}


def _claude_code_run(client, messages, model_name: str, doc_id=None,
                     folder_id=None, reasoning_effort=None, max_turns=None,
                     partial: bool = False, **unsupported):
    """The claude-code/ engine: one Claude Code CLI run per call. The
    conversation goes in as ONE user message, a text block per turn: the
    CLI caches the request up to its own trailing marker, so each call
    reads the previous call's prefix (one growing block would not cache).
    Validates eagerly; returns run(done) -> the run's typed events, ending
    with {"type": "done", ...} (answer, usage, finish_reason) when done."""
    given = sorted(name for name, value in unsupported.items()
                   if value is not None)
    if given:
        raise PageIndexAPIError(
            f"{model_name} runs the Claude Code CLI, which takes no "
            f"{', '.join(given)} — drop them.")
    if reasoning_effort == "none":
        raise PageIndexAPIError(
            f"{model_name} can't turn thinking off, so it can't honor "
            "reasoning_effort='none'; pass 'low' or leave it unset.")
    from contextlib import aclosing
    from .integrations.claude_agent_sdk import (
        claude_code_messages, claude_code_options, tool_server)
    system_texts, history = _split_chat_messages(messages, text_only=model_name)
    scope = client._local_doc_scope(doc_id)
    block = targeting_block(client, doc_id, folder_id)
    blocks = [{"type": "text", "text": block}] if block else []
    blocks += [{"type": "text", "text": f'<turn role="{m["role"]}">\n'
                f'{m["content"].replace("</turn>", "</ turn>")}\n</turn>'}
               for m in history]
    # cloud tool failures (auth, limits): the run fails on them, as the
    # bridge lanes do, instead of the model answering around them
    failures: list = []
    options = claude_code_options(
        model_name, reasoning_effort,
        system_prompt=_managed_instructions(client, [_TURNS_NOTE, *system_texts]),
        mcp_servers={"pageindex": tool_server(client, doc_ids=scope,
                                              failures=failures)},
        allowed_tools=["mcp__pageindex"],
        max_turns=max_turns if max_turns is not None else 10,
        include_partial_messages=partial)

    async def prompt():
        yield {"type": "user", "message": {"role": "user", "content": blocks}}

    async def run(done: bool):
        from claude_agent_sdk import (AssistantMessage, ResultMessage,
                                      StreamEvent, TextBlock, ToolResultBlock,
                                      ToolUseBlock, UserMessage)
        names: dict = {}  # tool_use id -> tool name, to label results
        answer = ""  # the text since the last tool call
        sent = ""  # answer text out since the last text block
        streamed = ""  # this model call's text deltas
        cut = None  # (finish_reason, usage) of a streamed run cut short
        failures.clear()
        try:
            async with aclosing(claude_code_messages(prompt(), options)) as sdk:
                async for message in sdk:
                    if failures:  # stop at the failed tool, before any answer around it
                        break
                    if isinstance(message, StreamEvent):
                        if message.event.get("type") == "message_start":
                            streamed = ""  # a retried call streams afresh
                        delta = message.event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            streamed += delta["text"]
                            if not sent.startswith(streamed):  # else a retry's replay
                                rest = (streamed[len(sent):]
                                        if streamed.startswith(sent)
                                        else "\n\n" + streamed)
                                sent = streamed
                                yield {"type": "answer", "delta": rest}
                        elif (delta.get("type") == "thinking_delta"
                              and delta.get("thinking")):
                            yield {"type": "thinking", "delta": delta["thinking"]}
                    elif isinstance(message, AssistantMessage):
                        text = "".join(item.text for item in message.content
                                       if isinstance(item, TextBlock))
                        if text:
                            answer += text
                            # the CLI's non-streaming fallback sends the text
                            # whole, after whatever the failed stream got out
                            rest = (text[len(sent):] if text.startswith(sent)
                                    else "\n\n" + text)  # regenerated differently
                            if rest:
                                yield {"type": "answer", "delta": rest}
                            sent = streamed = ""
                        for item in message.content:
                            if isinstance(item, ToolUseBlock):
                                answer = ""
                                name = item.name.removeprefix("mcp__pageindex__")
                                names[item.id] = name
                                yield {"type": "tool_call", "call_id": item.id,
                                       "name": name, "arguments": item.input}
                    elif (isinstance(message, UserMessage)
                          and isinstance(message.content, list)):
                        for item in message.content:
                            if isinstance(item, ToolResultBlock):
                                yield {"type": "tool_result",
                                       "call_id": item.tool_use_id,
                                       "name": names.get(item.tool_use_id, "tool"),
                                       "output": item.content}
                    elif isinstance(message, ResultMessage) and done and not failures:
                        yield {"type": "done", "answer": answer,
                               "usage": _claude_code_usage(message.usage),
                               "finish_reason": _CLAUDE_CUT.get(
                                   message.stop_reason, "stop")}
                    elif (isinstance(message, ResultMessage)
                          and message.stop_reason in _CLAUDE_CUT):
                        cut = (_CLAUDE_CUT[message.stop_reason],
                               _claude_code_usage(message.usage))
        except PageIndexAPIError:
            if failures:  # a tool failure (402, 429) is the cause to report
                raise failures[0] from None
            raise
        if failures:  # raised after the run, once the CLI is closed
            raise failures[0]
        if cut is not None:
            raise _with_usage(_cut_short_error(cut[0]), cut[1])
    return run


def _claude_code_usage(usage) -> dict:
    """The run's usage, chat.completions dialect: prompt tokens include the
    cache reads and writes, as LiteLLM counts Claude's."""
    usage = usage or {}
    cached = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    prompt = (usage.get("input_tokens") or 0) + cached + cache_write
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    return _chat_shape(prompt, usage.get("output_tokens") or 0, cached,
                       cache_write, thinking)


def _claude_code_completions(run, model_name: str, stream: bool,
                             stream_metadata: bool):
    """run_chat_completions' envelope and chunks over a claude-code/ run."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    reported = model_name.removeprefix("claude-code/")

    def chunk(delta: dict, finish=None) -> dict:
        return _ChatCompletionChunk({
            "id": chat_id, "object": "chat.completion.chunk",
            "created": created, "model": reported,
            "choices": [{"index": 0, "delta": delta,
                         "finish_reason": finish}]})

    if not stream:
        async def last():
            done = None
            async for event in run(True):
                done = event
            return done
        done = _run_sync(last())
        if not done:
            raise PageIndexAPIError("The model returned no response.")
        return _ChatCompletion({
            "id": chat_id, "object": "chat.completion",
            "created": created, "model": reported,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": done["answer"]},
                         "finish_reason": done["finish_reason"]}],
            "usage": done["usage"]})

    async def agen():
        from contextlib import aclosing
        yield chunk({"role": "assistant", "content": ""})
        async with aclosing(run(True)) as events:  # an abandoned stream closes the run
            async for event in events:
                if event["type"] == "answer":
                    yield chunk({"content": event["delta"]})
                elif event["type"] == "done":
                    yield chunk({}, finish=event["finish_reason"])
                    yield _ChatCompletionChunk({
                        "id": chat_id, "object": "chat.completion.chunk",
                        "created": created, "model": reported, "choices": [],
                        "usage": event["usage"]})

    if stream_metadata:
        return _stream_sync(agen)
    return _text_only(_stream_sync(agen))
