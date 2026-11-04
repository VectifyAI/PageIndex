"""Explicit, process-local conversation state for the own-model chat lane."""
from __future__ import annotations

import asyncio
import copy
import os

from ..client import PageIndexClient, PageIndexLocalClient
from ..errors import PageIndexAPIError


def _backend_copy(backend):
    # A caller-owned transport is a resource, not serializable settings.
    return {key: value if key == "http_client" else copy.deepcopy(value)
            for key, value in (backend or {}).items()}


async def _settle(task):
    """Finish cancellation cleanup even if the caller cancels again."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class MemoryChatSession:
    """A serial, in-memory Chat Completions conversation over local documents.

    ``await session.chat(question)`` returns the usual protocol envelope;
    ``session.messages`` returns a detached copy of the committed transcript,
    including tool calls/results. Only complete turns are committed. Failed,
    canceled and truncated turns leave the previous history intact.

    The document allowlist, model, instructions and protocol are fixed at
    construction. Changing the source client's chat/storage configuration
    refuses further turns; create a new session for a new configuration.
    ``await clear()`` removes history/context, not the configuration binding.
    Each session belongs to one event loop. No disk/cloud persistence,
    streaming, managed cloud, or claude-code session is created.
    """

    def __init__(self, client: PageIndexClient, *, doc_id: str | list[str],
                 model: str, instructions: str | None = None,
                 protocol: str = "chat_completions", max_turns: int | None = None):
        from ..agent_tools import _require_doc_selection
        from ..integrations.claude_agent_sdk import is_claude_code
        from ..local_api import LocalAPI
        from ..local_chat import _validate_max_turns
        if (not isinstance(client, PageIndexClient) or getattr(client, "api_key", None)
                or not isinstance(client._api, LocalAPI)):
            raise PageIndexAPIError("MemoryChatSession initially supports local documents only")
        if protocol != "chat_completions":
            raise PageIndexAPIError("MemoryChatSession supports only protocol='chat_completions'")
        if not isinstance(model, str) or not model.strip() or is_claude_code(model):
            raise PageIndexAPIError("Choose a non-empty own-model Chat Completions model, not claude-code/")
        if doc_id is None:
            raise PageIndexAPIError("MemoryChatSession requires an explicit document allowlist")
        _require_doc_selection(doc_id)
        if (not isinstance(doc_id, (str, list)) or
                any(not isinstance(value, str) or not value.strip()
                    for value in ([doc_id] if isinstance(doc_id, str) else doc_id))):
            raise PageIndexAPIError("doc_id must be a non-empty string or list of non-empty strings")
        if instructions is not None and not isinstance(instructions, str):
            raise PageIndexAPIError("instructions must be a string")
        _validate_max_turns(max_turns)
        self._source = client
        self._binding = self._source_binding()
        self._doc_ids = tuple([doc_id] if isinstance(doc_id, str) else doc_id)
        self._model, self._instructions = model, instructions
        self._max_turns = max_turns
        # The executing client is private: changing the source during a turn
        # cannot redirect that turn's tools to another store or model backend.
        self._client = PageIndexLocalClient(
            storage_path=self._binding["storage_path"], chat_model=model,
            chat_backend=_backend_copy(client.chat_backend) or None,
            instructions=client.instructions)
        self._messages: list[dict] = []
        self._context = None
        self._lock = asyncio.Lock()
        self._loop = None

    def _source_binding(self):
        return {"api": self._source._api, "api_key": getattr(self._source, "api_key", None),
                "storage_path": os.path.abspath(self._source.storage_path),
                "model": self._source.chat_model,
                "instructions": self._source.instructions,
                "backend": _backend_copy(self._source.chat_backend)}

    def _check_binding(self):
        try:
            unchanged = self._source_binding() == self._binding
        except (AttributeError, TypeError, ValueError):
            unchanged = False
        if not unchanged:
            raise PageIndexAPIError(
                "The session client's configuration changed; create a new MemoryChatSession")

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("MemoryChatSession must be used on one event loop")

    @property
    def model(self) -> str:
        return self._model

    @property
    def protocol(self) -> str:
        return "chat_completions"

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    @property
    def messages(self) -> list[dict]:
        """Committed user/assistant/tool history, detached from session state."""
        return copy.deepcopy(self._messages)

    @property
    def context(self) -> dict | None:
        """The committed targeting block and system instructions, if started."""
        if self._context is None:
            return None
        return {"document_context": self._context[0], "instructions": self._context[1]}

    def _prepare(self, history):
        from ..local_chat import _chat_agent, _managed_instructions, targeting_block
        context = self._context
        if context is None:
            context = (targeting_block(self._client, list(self._doc_ids)),
                       _managed_instructions(self._client, [self._instructions]
                                             if self._instructions else []))
        return (_chat_agent(self._client, history, list(self._doc_ids), self.model,
                            _context=context), context)

    async def chat(self, question: str):
        """Run one question through the existing Runner and commit its full turn."""
        if not isinstance(question, str) or not question.strip():
            raise PageIndexAPIError("question must be a non-empty string")
        self._check_loop()
        async with self._lock:
            from ..local_chat import (_aclose_backend, _cut_short_error,
                                      _run_chat_completion, _with_usage)
            self._check_binding()
            history = self.messages + [{"role": "user", "content": question}]
            # Tool/agent preparation can import frameworks and read document
            # metadata. Settle it on cancellation so its backend can be closed.
            preparing = asyncio.create_task(asyncio.to_thread(self._prepare, history))
            try:
                prepared, context = await asyncio.shield(preparing)
            except asyncio.CancelledError:
                try:
                    prepared, _ = await _settle(preparing)
                except Exception:
                    pass
                else:
                    await _settle(asyncio.create_task(_aclose_backend(prepared[0])))
                raise
            agent, items, model_name, replies = prepared
            try:
                self._check_binding()
            except Exception:
                await _aclose_backend(agent)
                raise
            result = await _run_chat_completion(
                self._client, agent, items, model_name, replies, self._max_turns)
            self._check_binding()
            finish = result["choices"][0]["finish_reason"]
            if finish != "stop":
                error = _with_usage(_cut_short_error(finish), result.get("usage"))
                error.partial_answer = result["choices"][0]["message"].get("content")
                raise error
            turn = result["messages"]
            if not turn or turn[-1].get("role") != "assistant" or turn[-1].get("tool_calls"):
                raise PageIndexAPIError("The run did not return a complete assistant transcript")
            # No await between the final checks and the atomic commit.
            self._messages = copy.deepcopy(history + turn)
            self._context = context
            return result

    async def clear(self):
        """Serialize with active turns, then forget the history and prompt context."""
        self._check_loop()
        async with self._lock:
            self._messages = []
            self._context = None
