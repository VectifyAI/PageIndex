"""PageIndex SDK client: the 0.2.x cloud surface, now with a local mode."""
from __future__ import annotations

import os
import threading
import time
import warnings
from typing import Any, Callable, Iterator, Optional, Union, cast

from .errors import PageIndexAPIError


def _preload_litellm() -> None:
    try:
        import litellm  # noqa: F401
    except Exception:
        pass


def _parse_pages(pages: str) -> list[int]:
    result: set[int] = set()
    too_many = (f"Page specification '{pages}' spans more than "
                "10000 pages; request a narrower range")
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
        else:
            start = end = int(part)
        # Bound each part arithmetically before materializing it — a spec
        # like "1-999999999" would otherwise expand to a billion integers.
        # The cap is on distinct pages, so overlapping parts (a parent
        # section plus its children) don't double-count.
        if end - start + 1 > 10_000:
            raise ValueError(too_many)
        result.update(range(start, end + 1))
        if len(result) > 10_000:
            raise ValueError(too_many)
    return sorted(result)


def _agents_sdk_model_name(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    Python SDK client for PageIndex.

    Cloud mode (an ``api_key`` is given) talks to the PageIndex API at
    api.pageindex.ai, exactly like the 0.2.x SDK. Local mode (no ``api_key``)
    runs the same operations on your machine: documents are indexed with the
    open-source PageIndex pipeline using your own LLM provider key (e.g.
    ``OPENAI_API_KEY`` in the environment) and stored under ``storage_path``.

    Args:
        api_key (str, optional): PageIndex cloud API key
            (https://dash.pageindex.ai/api-keys). Omit for local mode.
        index_model (str, optional): Local mode only — LLM used to index
            documents (structure and summaries). Defaults to the SDK
            default (fast and cheap).
        chat_model (str, optional): Local mode only — the model the chat
            surfaces (``chat``, ``chat_completions``, ``responses``)
            default to, exposed as ``client.chat_model``. Chat names
            route through LiteLLM and mean what LiteLLM says they mean;
            bare names are OpenAI-compatible shorthand, and
            ``openai/Qwen/...`` is the form for an OpenAI-compatible
            server that itself serves slashed model ids (vLLM, TGI).
            Defaults to the SDK default (strong).
        model (str, optional): Local mode only — one model for both roles:
            sets the default for ``index_model`` and ``chat_model`` at
            once. The role-specific arguments win over it. (Also the
            0.2.8-era name for the indexing model — old configs keep
            working unchanged.)
        summary_model (str, optional): Local mode only — legacy: overrides
            the model used for node summaries and document descriptions;
            ``index_model`` covers this.
        retrieve_model (str, optional): Local mode only — legacy name for
            ``chat_model``.
        storage_path (str, optional): Local mode only — directory where
            indexed documents are stored. Defaults to ``./.pageindex``.

    Usage:
        client = PageIndexClient(api_key="...")   # cloud
        client = PageIndexClient()                # local

    PageIndexCloudClient / PageIndexLocalClient pin the mode at construction
    instead of inferring it from api_key.

    Local mode differences (all documented per method): indexing is
    synchronous, only PDFs are supported, and folders / ``beta_headers`` /
    the deprecated retrieval API (``submit_query``, ``get_retrieval``) are
    cloud-only.
    """

    BASE_URL = "https://api.pageindex.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        index_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[str] = None,
        index_backend: Optional[dict[str, Any]] = None,
        chat_backend: Optional[dict[str, Any]] = None,
    ):
        if api_key == "":
            raise PageIndexAPIError(
                "api_key is an empty string. Pass a real PageIndex API key for "
                "cloud mode, or omit api_key entirely for local mode."
            )
        model_args = {"index_model": index_model, "chat_model": chat_model,
                      "model": model, "summary_model": summary_model,
                      "retrieve_model": retrieve_model}
        if api_key is not None:
            local_only = dict(model_args, storage_path=storage_path,
                              index_backend=index_backend,
                              chat_backend=chat_backend)
            passed = [name for name, value in local_only.items() if value is not None]
            if passed:
                raise PageIndexAPIError(
                    f"Local-mode arguments ({', '.join(passed)}) cannot be "
                    "combined with api_key — remove them, or omit api_key to "
                    "run locally."
                )
            self.api_key = api_key
            from .cloud_api import CloudAPI
            self._api = CloudAPI(self)
        else:
            from .utils import ConfigLoader
            overrides = {key: value for key, value in model_args.items()
                         if value}
            opt = ConfigLoader().load(overrides or None)
            self.model = opt.model
            self.index_model = opt.index_model
            self.summary_model = opt.summary_model
            self.chat_model = _agents_sdk_model_name(opt.chat_model)
            self.chat_backend = chat_backend
            self.storage_path = storage_path or ".pageindex"
            from .local_api import LocalAPI
            self._api = LocalAPI(
                storage_path=self.storage_path,
                model=self.model,
                summary_model=self.summary_model,
                retrieve_model=self.chat_model,
                index_backend=index_backend,
            )
            # LiteLLM's multi-second import would otherwise land on the
            # first chat call; failures resurface there with real context.
            threading.Thread(target=_preload_litellm, daemon=True).start()

    @property
    def retrieve_model(self):
        """Legacy name for ``chat_model``."""
        return self.chat_model

    # ---------- DOCUMENT SUBMISSION ----------

    def submit_document(
        self,
        file_path: str,
        mode: Optional[str] = None,
        beta_headers: Optional[list[str]] = None,
        folder_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """
        Submit a PDF document for processing. Returns {'doc_id': ..., 'name': ...}.

        Cloud: uploads the file; processing is asynchronous. Pass
        ``wait=True`` to block until the document is ready, or poll
        ``get_document(doc_id)['status']`` yourself.

        Local: indexes the document in this call and stores it under
        ``storage_path``. Defaults to Flash indexing: layout-based extraction,
        refined for retrieval (a deterministic merge, then an LLM expansion
        pass); node summaries, the expansion pass, and the document
        description use ``summary_model``. Pass ``mode="standard"`` for a
        full LLM-built tree (slower). ``beta_headers`` and ``folder_id`` are
        cloud-only.

        Args:
            file_path (str): Path to the PDF file.
            mode (str, optional): Processing mode. Local defaults to "flash";
                pass "standard" for a full LLM-built tree. Cloud modes are
                passed through (e.g. "mcp").
            beta_headers (list[str], optional): Cloud-only beta feature headers.
            folder_id (str, optional): Cloud-only folder (workspace) ID.
            metadata (dict, optional): Your own JSON-serializable tags for the
                document; returned in get_tree/get_ocr responses and
                list_documents entries (both modes).
            wait (bool): Return only once the document is ready for use.
                Cloud: polls status until "completed" (raises on "failed" or
                after 30 minutes). Local: indexing is synchronous already, so
                this changes nothing. Leave False to submit many documents
                concurrently and poll afterwards.

        Returns:
            dict: {'doc_id': ..., 'name': ...}. 'name' is the stored document
                name: a taken name gains a numeric suffix (name_1..name_99)
                and a UserWarning is emitted. Older cloud servers omit 'name'.
        """
        result = self._api.submit_document(
            file_path=file_path, mode=mode,
            beta_headers=beta_headers, folder_id=folder_id, metadata=metadata,
        )
        stored = result.get("name")
        if stored and stored != os.path.basename(file_path):
            warnings.warn(
                f'Document "{os.path.basename(file_path)}" was stored as '
                f'"{stored}".',
                stacklevel=2,
            )
        if wait:
            self._wait_until_ready(result["doc_id"])
        return result

    def _wait_until_ready(self, doc_id: str, timeout: float = 1800.0) -> None:
        import requests
        interval = 2.0
        deadline = time.monotonic() + timeout
        poll_failures = 0
        while True:
            try:
                status = self.get_document(doc_id).get("status")
                poll_failures = 0
            except (PageIndexAPIError, requests.RequestException) as exc:
                # Tolerate transient poll failures; a 30-minute wait should
                # not die on one 502 or dropped connection.
                poll_failures += 1
                if poll_failures >= 3:
                    raise PageIndexAPIError(
                        f"Could not poll document status (doc_id: {doc_id}): "
                        f"{exc}. Processing continues in the cloud — poll "
                        "get_document(doc_id) for status."
                    ) from exc
                status = None
            if status == "completed":
                return
            if status == "failed":
                raise PageIndexAPIError(
                    f"Document processing failed (doc_id: {doc_id})."
                )
            if time.monotonic() >= deadline:
                raise PageIndexAPIError(
                    f"Timed out after {int(timeout)}s waiting for document "
                    f"processing (doc_id: {doc_id}, last status: {status}). "
                    "Processing continues in the cloud — poll "
                    "get_document(doc_id) for status."
                )
            time.sleep(interval)
            interval = min(interval * 1.5, 15.0)

    # ---------- OCR FUNCTIONALITY ----------

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        """
        Get OCR status and results.

        Args:
            doc_id (str): Document ID.
            format (str): 'page' for page-based results, 'node' for node-based
                results, or 'raw' for concatenated markdown.

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...}.
            With 'page', result entries are {'page_index', 'markdown', ...}.

        Local: the "OCR" result is the text extracted from the PDF while
        indexing (no OCR model runs locally, so scanned/image-only PDFs have
        no local text).
        """
        return self._api.get_ocr(doc_id=doc_id, format=format)

    def get_page_content(self, doc_id: str, pages: str) -> list[dict[str, Any]]:
        """
        Get text content of specific pages.

        Args:
            doc_id (str): Document ID.
            pages (str): Page specifier — '5-7', '3,8', or '12'.

        Returns:
            list: Matching entries from get_ocr (format='page').
        """
        wanted = set(_parse_pages(pages))
        result = self.get_ocr(doc_id, format="page")
        all_pages = result["result"]
        if all_pages is None:
            raise PageIndexAPIError(
                f"Document '{doc_id}' is not ready "
                f"(status: {result.get('status', 'unknown')})"
            )
        return [p for p in all_pages if p["page_index"] in wanted]

    # ---------- TREE GENERATION ----------

    def get_tree(self, doc_id: str, node_summary: bool = False,
                 include_text: bool = True) -> dict[str, Any]:
        """
        Get tree generation status and results.

        Args:
            doc_id (str): Document ID.
            node_summary (bool): Include node summaries in the tree.
            include_text (bool): Include node text (default True).
                False is useful for structure-only views (saves tokens).

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...} where
            result nodes are {'title', 'node_id', 'page_index', ('summary' /
            'prefix_summary',) ('text',) 'nodes'}.
        """
        tree = self._api.get_tree(doc_id=doc_id, node_summary=node_summary,
                                  include_text=include_text)
        if not include_text and tree.get("result"):
            from .utils import remove_fields
            tree["result"] = remove_fields(tree["result"], fields=["text"])
        return tree

    def get_document_structure(self, doc_id: str) -> list[dict[str, Any]]:
        """
        Get the document's tree structure without text — summaries included.

        Returns:
            list: Tree nodes with titles, page ranges, and summaries.
        """
        return self.get_tree(doc_id, node_summary=True, include_text=False)["result"]

    def is_retrieval_ready(self, doc_id: str) -> bool:
        """
        Check if a document is ready for retrieval. API errors (including a
        missing document) are reported as False; transport errors (connection
        failures, timeouts) propagate.
        """
        try:
            result = self.get_tree(doc_id)
            return result.get("retrieval_ready", False)
        except PageIndexAPIError:
            return False

    # ---------- RETRIEVAL (cloud-only, deprecated) ----------

    def submit_query(self, doc_id: str, query: str, thinking: bool = False) -> dict[str, Any]:
        """
        Submit a retrieval query for a document. Returns {'retrieval_id': ...}.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat_completions`` instead.
        """
        return self._require_cloud(
            "submit_query is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat_completions instead."
        ).submit_query(doc_id=doc_id, query=query, thinking=thinking)

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        """
        Get retrieval status and results for a submitted query.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat_completions`` instead.
        """
        return self._require_cloud(
            "get_retrieval is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat_completions instead."
        ).get_retrieval(retrieval_id=retrieval_id)

    # ---------- CHAT ----------

    def chat(
        self,
        messages: Union[str, list[dict[str, str]]],
        doc_id: Optional[Union[str, list[str]]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Ask a question about your documents, get the answer.

        Thin sugar over ``chat_completions()`` in both modes — same
        engine, same wire, minus the envelope. Multi-turn: keep your own
        role/content list of the visible conversation (append each answer
        as an assistant message) and pass it back. For usage accounting,
        streaming metadata, or the tool-use process, use the protocol
        surfaces: ``chat_completions()``, ``responses()``, ``messages()``.

        Args:
            messages: A question string, or role/content conversation
                history.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls.
            stream: Yield the answer as text chunks as it is produced.
            model: Local only — backend model name (defaults to
                ``chat_model``).
            reasoning_effort: Local only — how hard the model thinks
                (``"low"`` / ``"medium"`` / ``"high"``; what a backend
                accepts is its own). Unset sends nothing — the model's
                default behavior applies.

        Returns:
            - stream=False: the answer string
            - stream=True: iterator of text chunks
        """
        result = self.chat_completions(messages, stream=stream,
                                       doc_id=doc_id, model=model,
                                       reasoning_effort=reasoning_effort)
        if stream:
            return cast(Iterator[str], result)
        envelope = cast(dict[str, Any], result)
        return envelope["choices"][0]["message"]["content"] or ""

    def chat_completions(
        self,
        messages: Union[str, list[dict[str, str]]],
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        temperature: Optional[float] = None,
        stream_metadata: bool = False,
        enable_citations: bool = False,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        extra_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        backend: Optional[dict[str, Any]] = None,
    ) -> Union[dict[str, Any], Iterator[str], Iterator[dict[str, Any]]]:
        """
        PageIndex Chat Completions: document QA in one call.

        Cloud: the hosted chat endpoint. Local: a managed document-QA agent
        run over the local tools against your own LLM backend, routed
        through LiteLLM — model names mean what LiteLLM says they mean.
        Bare names are OpenAI-compatible shorthand (the OpenAI SDK's usual
        env config — OPENAI_API_KEY, OPENAI_BASE_URL — selects the
        backend, so any OpenAI-compatible server works; write
        ``openai/Qwen/...`` when the server itself serves slashed ids),
        provider-prefixed names — ``anthropic/…``, ``bedrock/…`` — reach
        that provider, and LiteLLM-routed Claude models get the managed
        prompt prefix cache-marked automatically. The non-stream
        response carries the final answer only; streaming yields the
        agent's visible text as it is produced, including narration before
        tool calls. ``finish_reason`` reports loop completion ("stop") —
        the engine does not surface per-turn backend finish reasons. For
        the tool-use process and prompt-cache round-trip use
        ``responses()`` or ``messages()``.

        Args:
            messages: Conversation messages with 'role' and 'content' keys,
                or a bare query string (it becomes a single user message).
                Local also accepts system/developer messages — their content
                is appended to the managed system prompt. Local takes text
                history only: tool-role turns are rejected (the cloud
                endpoint forwards them verbatim), and message fields beyond
                role/content are dropped.
            stream: Enable streaming responses.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call and is part
                of the cached prompt prefix.
            temperature: Sampling temperature, passed through to the model.
            stream_metadata: With stream=True, yield chunk dicts instead of
                text pieces.
            enable_citations: Cloud-only — local mode raises (citations need
                block-level OCR data local mode does not store).
            model: Local only — backend model name (defaults to
                ``chat_model``). The cloud endpoint selects its own.
            max_turns: Local only — cap on agent turns per call.
            top_p: Local only — nucleus sampling, passed through to the
                model.
            max_tokens: Local only — per-call output cap, passed through;
                it bounds each backend call in the agent loop (the way
                max_turns bounds the loop), not the whole run.
            reasoning_effort: Local only — passed through verbatim as
                LiteLLM's ``reasoning_effort``; each provider maps it to
                its own thinking control, and the values mean what the
                backend says they mean. Unset sends nothing (the
                backend's default applies).
            extra_body: Local only — extra request fields beyond this
                method's parameters, merged last so they win.
                OpenAI-compatible backends take them verbatim in the
                request body; LiteLLM-routed providers take them as
                LiteLLM's own params (mapped or refused per provider).
                Credentials belong in ``backend``, never here.
            extra_headers: Local only — extra HTTP headers merged into
                each backend request; caller headers win. One exception:
                LiteLLM's anthropic adapter owns the ``anthropic-beta``
                header (your value is dropped there) — use ``messages()``
                for Anthropic beta flags.
            backend: Local only — connection overrides for this call's
                backend, merged over the client's ``chat_backend``
                (per-call keys win). Keys are LiteLLM's own connection
                params — ``api_key``, ``base_url``, ``api_version``,
                ``aws_*``, … — passed through verbatim.

        Returns:
            - stream=False: complete response dict ({'id', 'object', 'created',
              'choices', 'usage'})
            - stream=True, stream_metadata=False: iterator of text chunks
            - stream=True, stream_metadata=True: iterator of chunk dicts
        """
        if isinstance(messages, str):
            if not messages.strip():
                raise PageIndexAPIError(
                    "messages must be a non-empty string or a list of "
                    "message dicts.")
            messages = [{"role": "user", "content": messages}]
        from .cloud_api import CloudAPI
        if not isinstance(self._api, CloudAPI):
            from .local_chat import run_chat_completions
            return run_chat_completions(
                self, messages, stream=stream, doc_id=doc_id,
                temperature=temperature, stream_metadata=stream_metadata,
                enable_citations=enable_citations, model=model,
                max_turns=max_turns, top_p=top_p, max_tokens=max_tokens,
                reasoning_effort=reasoning_effort, extra_body=extra_body,
                extra_headers=extra_headers, backend=backend,
            )
        if (model is not None or max_turns is not None or top_p is not None
                or max_tokens is not None or reasoning_effort is not None
                or extra_body is not None or extra_headers is not None
                or backend is not None):
            raise PageIndexAPIError(
                "model, max_turns, top_p, max_tokens, reasoning_effort, "
                "extra_body, extra_headers and backend are local-mode "
                "parameters — the cloud chat endpoint selects its own model."
            )
        return self._api.chat_completions(
            messages=messages, stream=stream, doc_id=doc_id,
            temperature=temperature, stream_metadata=stream_metadata,
            enable_citations=enable_citations,
        )

    def responses(
        self,
        input: Union[str, list[dict[str, Any]]],
        model: Optional[str] = None,
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_turns: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        reasoning: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        backend: Optional[dict[str, Any]] = None,
    ) -> Union[dict[str, Any], Iterator[dict[str, Any]]]:
        """
        Document QA over the OpenAI Responses protocol — the agentic surface.

        Local only for now. Drives your backend's /responses end to end (no
        translation layer). The envelope is official Responses shape —
        ``output`` carries the model-produced items and parses with the
        openai SDK types — and the whole process transcript (including the
        tool outputs the SDK executed) rides in the extra ``items`` field.
        Append the returned ``items`` to your next call's ``input`` verbatim
        to keep provider prompt-cache prefix continuity and the agent's
        memory of what it already read.

        Requires a backend that supports the
        Responses API; backends that only speak chat.completions should use
        ``chat_completions()``. Provider-prefixed models (``anthropic/…``)
        route through LiteLLM's chat.completions adapter and are therefore
        refused here — use ``chat_completions()`` or ``messages()`` for
        those.

        Args:
            input: A user message string, or a list of Responses input items
                (round-trip prior ``items`` here).
            model: Backend model name (defaults to ``chat_model``).
            stream: Yield Responses stream events as dicts — one logical
                response per call: per-turn backend lifecycle events are
                collapsed, sequence numbers are reassigned monotonically,
                and ``output_index`` is re-based onto the single logical
                ``output``. The single final event is the terminal
                ``response.*`` for the run's status; its ``response``
                carries the tool outputs in ``items``.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call and is part
                of the cached prompt prefix.
            instructions: Appended to the managed system prompt.
            temperature / top_p: Passed through to the model.
            max_turns: Cap on agent turns per call.
            max_output_tokens: Per-call output cap, passed through; it
                bounds each backend call in the agent loop (the way
                max_turns bounds the loop), not the whole run. Echoed in
                the envelope.
            reasoning: Responses reasoning options, forwarded verbatim
                (e.g. ``{"effort": "low", "summary": "auto"}``) — the
                values mean what the backend says they mean. Unset sends
                nothing (the backend's default applies).
            extra_body: Extra request fields beyond this method's
                parameters, merged verbatim into the request body (last,
                so they win). Credentials belong in ``backend``, never
                here.
            extra_headers: Extra HTTP headers merged into each request;
                caller headers win over defaults.
            backend: Connection overrides for this call's backend client,
                merged over the client's ``chat_backend`` (per-call keys
                win). Keys are the openai SDK's client params —
                ``api_key``, ``base_url``, ``organization``, … — passed
                verbatim; unknown keys raise.
        """
        from .cloud_api import CloudAPI
        if isinstance(self._api, CloudAPI):
            raise PageIndexAPIError(
                "responses is not available on PageIndex cloud yet — it is "
                "a local-mode surface for now."
            )
        from .local_chat import run_responses
        return run_responses(
            self, input, model=model, stream=stream, doc_id=doc_id,
            instructions=instructions, temperature=temperature, top_p=top_p,
            max_turns=max_turns, max_output_tokens=max_output_tokens,
            reasoning=reasoning, extra_body=extra_body,
            extra_headers=extra_headers, backend=backend,
        )

    def messages(
        self,
        messages: Union[str, list[dict[str, Any]]],
        model: str,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        system: Optional[Union[str, list[dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        max_turns: Optional[int] = None,
        thinking: Optional[dict[str, Any]] = None,
        extra_body: Optional[dict[str, Any]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        backend: Optional[dict[str, Any]] = None,
    ) -> Union[dict[str, Any], Iterator[Any]]:
        """
        Document QA over the Anthropic Messages protocol — Claude-native.

        Local only for now. Drives Anthropic's /v1/messages via the
        Anthropic SDK's own tool runner (requires ``pageindex[anthropic]``;
        ANTHROPIC_API_KEY selects the backend). ``tool_use``/``tool_result``
        round-trip is the format's native behavior: the response is the
        final message envelope with cross-turn aggregated ``usage`` plus a
        ``messages`` field — the full new turn sequence, valid for verbatim
        append to your history. The managed system prompt carries a
        ``cache_control`` breakpoint.

        Args:
            messages: Native Messages-format history (including prior
                tool_use/tool_result blocks on round-trip), or a bare query
                string (it becomes a single user message).
            model: Required — there is no cross-vendor default to guess.
            max_tokens: Per-turn output budget the Messages API requires on
                the wire; the default is resolved per model (8192, or 4096
                for the claude-3 generation whose ceiling is lower) so the
                simple call needs only a question. Passed through.
            stream: Yield the Anthropic SDK's event stream across turns
                (its native event objects, including SDK-synthesized
                convenience events), one message sequence per turn.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call.
            system: Appended after the managed system blocks.
            temperature / top_p / top_k / stop_sequences: Passed through.
            max_turns: Cap on agent turns per call (default 10, like the
                OpenAI surfaces). A truncated run reports
                ``stop_reason: "tool_use"`` and its ``messages`` remain
                valid for continuation.
            thinking: Anthropic thinking configuration, forwarded verbatim
                (e.g. ``{"type": "adaptive"}``) — the values and their
                constraints are the backend's. Unset sends nothing.
            extra_body: Extra request fields beyond this method's
                parameters, merged verbatim into each request body.
                Credentials belong in ``backend``, never here.
            extra_headers: Extra HTTP headers merged into each request
                (e.g. ``anthropic-beta`` feature flags); caller headers
                win over defaults.
            backend: Connection overrides for this call's backend client,
                merged over the client's ``chat_backend`` (per-call keys
                win). Keys are the anthropic SDK's client params —
                ``api_key``, ``base_url``, ``auth_token``, … — passed
                verbatim; unknown keys raise.
        """
        from .cloud_api import CloudAPI
        if isinstance(self._api, CloudAPI):
            raise PageIndexAPIError(
                "messages is not available on PageIndex cloud yet — it is "
                "a local-mode surface for now."
            )
        from .local_chat import run_messages
        return run_messages(
            self, messages, model=model, max_tokens=max_tokens,
            stream=stream, doc_id=doc_id, system=system,
            temperature=temperature, top_p=top_p, top_k=top_k,
            stop_sequences=stop_sequences, max_turns=max_turns,
            thinking=thinking, extra_body=extra_body,
            extra_headers=extra_headers, backend=backend,
        )

    # ---------- DOCUMENT MANAGEMENT ----------

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """
        Get document metadata: {'id', 'name', 'description', 'status',
        'createdAt', 'pageNum', 'folderId'}. Status is one of "queued",
        "processing", "completed", "failed" (local documents are
        always "completed"; local 'folderId' is always None).

        'createdAt' is UTC with no timezone marker, in both modes. To show
        it in the user's timezone::

            from datetime import datetime, timezone
            datetime.fromisoformat(doc["createdAt"]).replace(
                tzinfo=timezone.utc).astimezone()
        """
        return self._api.get_document(doc_id=doc_id)

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        """
        Delete a PageIndex document and all its associated data.

        Returns:
            dict: {'message': 'Document deleted successfully.'}, or an empty
            dict if the cloud API responds with no body.
        """
        return self._api.delete_document(doc_id=doc_id)

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        List documents with pagination, newest first.

        Args:
            limit (int): Maximum documents to return (1-100).
            offset (int): Number of documents to skip.
            folder_id (str, optional): Cloud-only folder filter.

        Returns:
            dict: {'documents': [...], 'total', 'limit', 'offset'}.
        """
        return self._api.list_documents(limit=limit, offset=offset, folder_id=folder_id)

    # ---------- AGENT INTEGRATION ----------

    def agent_tools(self, include_management: bool = False) -> list[Callable[..., str]]:
        """
        Plain functions for any agent framework (LangChain, PydanticAI, ...).
        For the OpenAI / Claude Agent SDKs, prefer ``as_openai_tools()`` /
        ``as_claude_mcp()``.

        Cloud: the full cloud tool set, discovered live from the PageIndex
        MCP server when this method is called — one function per tool,
        signature and docstring synthesized from the server's schemas, calls
        executed from your process over MCP. Raises PageIndexAPIError if the
        server cannot be reached. Local: the built-in tools over the local
        store (``browse_documents``, ``get_document``,
        ``get_document_structure``, ``get_page_content``).

        Each function takes JSON-serializable arguments, returns a JSON
        string, and reports failures inside that JSON instead of raising.

        Args:
            include_management (bool): Also expose tools that modify the
                library. Local: adds ``remove_document``. Cloud: by default
                only tools the server marks read-only are exposed; True
                exposes the server's complete list (upload, delete, ...).
        """
        from .agent_tools import build_agent_tools
        return build_agent_tools(self, include_management)

    def as_openai_tools(self, include_management: bool = False,
                        hosted: bool = False,
                        doc_id: Optional[Union[str, list[str]]] = None) -> list:
        """
        Tools for the OpenAI Agents SDK — pass to ``Agent(tools=...)``
        (or ``openai_agent_config()`` for all the Agent slots in one
        call).

        Cloud (default): the full live read tool set (search, folders,
        images — as enabled for your key) as plain function tools,
        discovered from the PageIndex MCP server and executed from your
        process — works with any model backend. Binary tool results
        (e.g. ``get_document_image``) arrive as text placeholder stubs
        on this in-process path. Pass ``hosted=True`` to
        hand the connection to OpenAI instead: one hosted MCP tool, tool
        calls executed server-side (lowest latency; requires an
        OpenAI-hosted model on the Responses API). The framework's own
        ``MCPServerStreamableHttp`` — ``params={"url":
        f"{BASE_URL}/mcp?tools=read", "headers": {"Authorization":
        "Bearer <your PageIndex API key>"}}`` (drop ``?tools=read`` for
        the full tool set) — is the async-native alternative for its
        ``mcp_servers=`` slot.

        Local: the in-process tools, any model backend; ``hosted`` does
        not apply.

        ``openai-agents`` is imported only when this method is called.

        Args:
            include_management (bool): Also expose tools that modify the
                library (delete, upload). Default off: the in-process
                cloud default serves only server-annotated read-only
                tools, and ``hosted=True`` connects OpenAI to the
                read-only endpoint (``/mcp?tools=read``) instead.
            hosted (bool): Cloud only — hand the MCP connection to OpenAI
                for server-side tool execution (OpenAI models only).
            doc_id: Local only — restrict the tools to this document ID
                (or list of IDs), enforced at the tool layer: out-of-scope
                lookups return NOT_FOUND. Raises on cloud, where scoping
                is server-side.
        """
        from .integrations.openai_agents import build_openai_tools
        return build_openai_tools(self, include_management, hosted,
                                  doc_ids=doc_id)

    def _local_doc_scope(self, doc_id):
        """doc_id for the tool layer: passed through locally (structural
        allowlist), dropped on cloud where scoping is server-side and the
        config helpers keep prompt-level targeting."""
        if not getattr(self, "api_key", None):
            return doc_id
        if doc_id is not None and not doc_id:
            # Cloud has no tool-layer allowlist to make an empty scope mean
            # "nothing"; dropping it would silently mean "everything".
            raise PageIndexAPIError(
                "doc_id is empty. Pass one or more document IDs, or omit "
                "doc_id to give the agent the whole library.")
        return None

    def openai_agent_config(
        self,
        doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Document QA ``Agent`` kwargs for the OpenAI Agents SDK in one
        call::

            agent = Agent(**client.openai_agent_config())

        Sugar over the explicit form — ``agent_instructions`` (with
        ``doc_id`` targeting) as the instructions and
        ``as_openai_tools`` as the tools; local clients also carry their
        configured ``chat_model`` (cloud omits ``model`` so the
        framework default applies). To customize further, switch to
        those methods directly. You run this config in your own
        environment, so its model auth comes from there —
        ``chat_backend`` does not travel with it.

        Args:
            doc_id: Document ID or list of IDs to target, as in
                ``agent_instructions``. Local: also enforced at the tool
                layer, not just prompted. Cloud: prompt-level targeting
                (tool scoping is server-side).
            include_management (bool): Also expose tools that modify the
                library.
            model: Backend model name; overrides the local default.
        """
        from .agent_tools import build_agent_instructions
        scope = self._local_doc_scope(doc_id)
        config: dict[str, Any] = {
            "name": "PageIndex",
            "instructions": build_agent_instructions(self, doc_id,
                                                     scoped=scope is not None),
            "tools": self.as_openai_tools(include_management, doc_id=scope),
        }
        model = model or getattr(self, "chat_model", None)
        if model:
            config["model"] = model
        return config

    def as_anthropic_tools(self, include_management: bool = False,
                           asynchronous: bool = False,
                           doc_id: Optional[Union[str, list[str]]] = None,
                           ) -> list:
        """
        Runnable tools for the Anthropic SDK's tool runner — pass to
        ``client.beta.messages.tool_runner(tools=...)`` (or
        ``anthropic_runner_config()`` for the whole setup in one call).
        The default flavor is for the sync ``Anthropic`` client; pass
        ``asynchronous=True`` for ``AsyncAnthropic``. For a manual
        ``messages.create`` loop, serialize with
        ``[tool.to_dict() for tool in ...]``.

        Cloud: the full live read tool set (search, folders, images — as
        enabled for your key), discovered from the PageIndex MCP server
        and executed from your process; the server's input schemas pass
        through verbatim (MCP and the Messages API share the schema
        shape), and binary tool results (e.g. ``get_document_image``)
        arrive as text placeholder stubs on this in-process path. The
        server-side alternative is the Messages API's beta
        MCP connector — ``mcp_servers=[{"type": "url", "name":
        "pageindex", "url": f"{BASE_URL}/mcp?tools=read",
        "authorization_token": <your PageIndex API key>}]`` (drop
        ``?tools=read`` for the full tool set) — with no client-side
        tools involved. Local: the in-process tools — the same set
        ``messages()`` runs internally.

        Requires ``anthropic>=0.108.0``
        (``pip install 'pageindex[anthropic]'``), imported only when this
        method is called.

        Args:
            include_management (bool): Also expose tools that modify the
                library. Local: adds ``remove_document``. Cloud: by default
                only tools the server marks read-only are exposed; True
                exposes the server's complete list (upload, delete, ...).
            asynchronous (bool): Build ``beta_async_tool`` runnables for
                ``AsyncAnthropic`` (each tool call runs in a worker
                thread, keeping blocking I/O off your event loop). The
                sync and async runners each accept only their own flavor.
            doc_id: Local only — restrict the tools to this document ID
                (or list of IDs), enforced at the tool layer: out-of-scope
                lookups return NOT_FOUND. Raises on cloud, where scoping
                is server-side.
        """
        from .integrations.anthropic_sdk import build_anthropic_tools
        return build_anthropic_tools(self, include_management, asynchronous,
                                     doc_ids=doc_id)

    def anthropic_runner_config(
        self,
        model: str,
        doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
        asynchronous: bool = False,
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Document QA ``tool_runner`` kwargs for the Anthropic SDK in one
        call — only your ``messages`` remain::

            runner = anthropic_client.beta.messages.tool_runner(
                **client.anthropic_runner_config(model="claude-sonnet-4-5"),
                messages=[{"role": "user", "content": "..."}],
            )

        Sugar over the explicit form — ``agent_instructions`` (with
        ``doc_id`` targeting) as the system prompt and
        ``as_anthropic_tools`` as the tools — plus the same defaults
        ``messages()`` applies: a per-model ``max_tokens`` and a
        ``max_iterations`` bound of 10. To customize further, switch to
        those methods directly.

        Args:
            model: Backend model name (also resolves the ``max_tokens``
                default).
            doc_id: Document ID or list of IDs to target, as in
                ``agent_instructions``. Local: also enforced at the tool
                layer, not just prompted. Cloud: prompt-level targeting
                (tool scoping is server-side).
            include_management (bool): Also expose tools that modify the
                library.
            asynchronous (bool): Build async runnables for
                ``AsyncAnthropic``.
            max_tokens: Per-turn output budget; default resolved per
                model.
            max_turns: Agent-loop bound; default 10.
        """
        from .agent_tools import build_agent_instructions
        from .local_chat import _default_max_tokens
        scope = self._local_doc_scope(doc_id)
        return {
            "model": model,
            "max_tokens": (max_tokens if max_tokens is not None
                           else _default_max_tokens(model)),
            "system": build_agent_instructions(self, doc_id,
                                               scoped=scope is not None),
            "tools": self.as_anthropic_tools(include_management, asynchronous,
                                             doc_id=scope),
            "max_iterations": max_turns if max_turns is not None else 10,
        }

    def as_claude_mcp(self, include_management: bool = False,
                      doc_id: Optional[Union[str, list[str]]] = None):
        """
        ``mcp_servers`` entry for the Claude Agent SDK.

        Cloud: returns the remote PageIndex MCP config.
        ``include_management`` picks the endpoint, so the URL itself is
        the gate — the default connects to the read-only endpoint
        (``/mcp?tools=read``: the server registers only read-only tools),
        ``True`` connects to the full tool set. Local: returns an
        in-process SDK MCP server exposing the agent tools, gated the
        same way at registration (requires ``claude-agent-sdk``;
        ``pip install 'pageindex[claude]'``). ``doc_id`` (local only)
        restricts those tools to that document ID (or list), enforced at
        the tool layer; it raises on cloud, where scoping is server-side.

        Cloud hosts that surface MCP server instructions receive the same
        guidance ``agent_instructions()`` returns natively — passing both
        duplicates the text (harmless). ``system_prompt`` stays the
        recommended channel: it is guaranteed delivery, carries ``doc_id``
        targeting, and is the only channel local mode has.

        Usage (or ``claude_agent_config()`` for all three slots in one
        call)::

            options = ClaudeAgentOptions(
                system_prompt=client.agent_instructions(),
                mcp_servers={"pageindex": client.as_claude_mcp()},
                # Pre-approval only — the server itself is already gated.
                allowed_tools=["mcp__pageindex"],
            )
        """
        from .integrations.claude_agent_sdk import build_claude_mcp
        return build_claude_mcp(self, include_management, doc_ids=doc_id)

    def claude_agent_config(
        self,
        doc_id: Optional[Union[str, list[str]]] = None,
        include_management: bool = False,
        server_name: str = "pageindex",
    ) -> dict[str, Any]:
        """
        Document QA ``ClaudeAgentOptions`` kwargs in one call::

            options = ClaudeAgentOptions(**client.claude_agent_config())

        Sugar over the explicit form — the managed system prompt
        (``agent_instructions``) and the server entry (``as_claude_mcp``,
        itself the tool gate) with its ``allowed_tools`` pre-approval,
        one ``include_management`` and ``server_name`` applied
        everywhere. To customize (your own system prompt, extra
        servers), switch to those methods directly.

        Args:
            doc_id: Document ID or list of IDs to target, as in
                ``agent_instructions``. Local: also enforced at the tool
                layer, not just prompted. Cloud: prompt-level targeting
                (tool scoping is server-side).
            include_management (bool): Also allow tools that modify the
                library.
            server_name (str): Key the server is registered under.
        """
        from .agent_tools import build_agent_instructions
        scope = self._local_doc_scope(doc_id)
        return {
            "system_prompt": build_agent_instructions(self, doc_id,
                                                      scoped=scope is not None),
            "mcp_servers": {server_name: self.as_claude_mcp(
                include_management, doc_id=scope)},
            # Pre-approval only — the server itself is already gated (the
            # read-only endpoint on cloud, the registered set locally).
            "allowed_tools": [f"mcp__{server_name}"],
        }

    def agent_instructions(self, doc_id: Optional[Union[str, list[str]]] = None) -> str:
        """
        Orchestration guidance for document QA agents — pass as the agent's
        system prompt (or append to your own).

        Cloud: the live instructions the PageIndex MCP server serves for
        your key's tool set, fetched over the same session as
        ``agent_tools()`` — server-side guidance updates arrive without an
        SDK release. Raises PageIndexAPIError if the server cannot be
        reached. Local: the built-in guidance for the in-process tools.

        With ``doc_id`` (str or list, same shape as ``chat_completions``),
        appends the target documents' names and metadata and directs the
        agent to work within them. Raises PageIndexAPIError if a doc_id does
        not exist, or if its name is shadowed by a newer same-name document
        (the name-addressed tools could not reach it).
        """
        from .agent_tools import build_agent_instructions
        return build_agent_instructions(self, doc_id)

    # ---------- FOLDER MANAGEMENT ----------

    def create_folder(
        self,
        name: str,
        description: Optional[str] = None,
        parent_folder_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a folder (workspace). Cloud-only: local mode raises
        PageIndexAPIError.
        """
        return self._require_cloud(
            "create_folder is cloud-only — folders are not supported in local "
            "mode. Create the client with an api_key to use folders."
        ).create_folder(
            name=name, description=description, parent_folder_id=parent_folder_id,
        )

    def list_folders(self, parent_folder_id: Optional[str] = None) -> dict[str, Any]:
        """
        List folders. Cloud-only: local mode raises PageIndexAPIError.
        """
        return self._require_cloud(
            "list_folders is cloud-only — folders are not supported in local "
            "mode. Create the client with an api_key to use folders."
        ).list_folders(
            parent_folder_id=parent_folder_id,
        )

    def _require_cloud(self, message: str):
        from .cloud_api import CloudAPI
        if not isinstance(self._api, CloudAPI):
            raise PageIndexAPIError(message)
        return self._api


class PageIndexCloudClient(PageIndexClient):
    """Cloud mode — requires a real API key at construction."""

    def __init__(self, api_key: str):
        if not api_key:
            raise PageIndexAPIError(
                "PageIndexCloudClient requires a PageIndex API key — get one "
                "at https://dash.pageindex.ai/api-keys."
            )
        super().__init__(api_key)


class PageIndexLocalClient(PageIndexClient):
    """Local mode — no api_key parameter, no cloud access."""

    def __init__(
        self,
        *,
        index_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[str] = None,
        index_backend: Optional[dict[str, Any]] = None,
        chat_backend: Optional[dict[str, Any]] = None,
    ):
        super().__init__(None, index_model=index_model, chat_model=chat_model,
                         model=model, summary_model=summary_model,
                         retrieve_model=retrieve_model, storage_path=storage_path,
                         index_backend=index_backend, chat_backend=chat_backend)
