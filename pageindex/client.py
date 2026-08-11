"""PageIndex SDK client: the 0.2.x cloud surface, now with a local mode."""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable, Iterator, Optional, Union

from .errors import PageIndexAPIError


def _parse_pages(pages: str) -> list[int]:
    result = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _normalize_retrieve_model(model: str) -> str:
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
        model (str, optional): Local mode only — LLM used to build document
            trees. Defaults to the packaged config (see pageindex/config.yaml).
        summary_model (str, optional): Local mode only — LLM used for node
            summaries and document descriptions.
        retrieve_model (str, optional): Local mode only — exposed as
            ``client.retrieve_model`` (the agent demo reads it); the SDK
            itself consumes it once agent-based local chat lands in a
            later release.
        storage_path (str, optional): Local mode only — directory where
            indexed documents are stored. Defaults to ``./.pageindex``.

    Usage:
        client = PageIndexClient(api_key="...")   # cloud
        client = PageIndexClient()                # local

    PageIndexCloudClient / PageIndexLocalClient pin the mode at construction
    instead of inferring it from api_key.

    Local mode differences (all documented per method): indexing is
    synchronous, only PDFs are supported, and ``chat_completions`` (until
    agent-based local chat lands in a later release) / folders /
    ``beta_headers`` / the deprecated retrieval API (``submit_query``,
    ``get_retrieval``) are cloud-only.
    """

    BASE_URL = "https://api.pageindex.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        if api_key == "":
            raise PageIndexAPIError(
                "api_key is an empty string. Pass a real PageIndex API key for "
                "cloud mode, or omit api_key entirely for local mode."
            )
        if api_key is not None:
            local_only = {"model": model, "summary_model": summary_model,
                          "retrieve_model": retrieve_model, "storage_path": storage_path}
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
            overrides = {key: value for key, value in
                         {"model": model, "summary_model": summary_model,
                          "retrieve_model": retrieve_model}.items()
                         if value}
            opt = ConfigLoader().load(overrides or None)
            self.model = opt.model
            self.summary_model = getattr(opt, "summary_model", None) or opt.model
            self.retrieve_model = _normalize_retrieve_model(
                getattr(opt, "retrieve_model", None) or opt.model)
            self.storage_path = storage_path or ".pageindex"
            from .local_api import LocalAPI
            self._api = LocalAPI(
                storage_path=self.storage_path,
                model=self.model,
                summary_model=self.summary_model,
                retrieve_model=self.retrieve_model,
            )

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

        Local: indexes the document in this call (it blocks while your LLM
        builds the tree — minutes for a standard index of a long document),
        then stores it under ``storage_path``. Pass ``mode="flash"`` to build
        the tree with PageIndex Flash (layout-based extraction, no LLM calls
        for the structure; node summaries and the document description still
        use ``summary_model``). ``beta_headers`` and ``folder_id`` are
        cloud-only.

        Args:
            file_path (str): Path to the PDF file.
            mode (str, optional): Processing mode. Local mode supports
                "standard" and "flash"; omit it for standard indexing. Cloud
                modes are passed through (e.g. "mcp").
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
                    if isinstance(exc, PageIndexAPIError):
                        raise
                    raise PageIndexAPIError(
                        f"Could not poll document status: {exc}"
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
        PageIndexAPIError. Use ``chat_completions`` (cloud) instead.
        """
        return self._require_cloud(
            "submit_query is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat_completions in cloud mode."
        ).submit_query(doc_id=doc_id, query=query, thinking=thinking)

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        """
        Get retrieval status and results for a submitted query.

        Cloud-only: the cloud API marks this endpoint deprecated in favor of
        chat completions, so local mode does not implement it — raises
        PageIndexAPIError. Use ``chat_completions`` (cloud) instead.
        """
        return self._require_cloud(
            "get_retrieval is cloud-only — the retrieval API is deprecated in "
            "favor of chat completions; use chat_completions in cloud mode."
        ).get_retrieval(retrieval_id=retrieval_id)

    # ---------- CHAT COMPLETIONS ----------

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
    ) -> Union[dict[str, Any], Iterator[str], Iterator[dict[str, Any]]]:
        """
        PageIndex Chat Completions: document QA in one call.

        Cloud: the hosted chat endpoint. Local: a managed document-QA agent
        run over the local tools against your own LLM backend's
        /chat/completions (requires ``pageindex[openai]``; the OpenAI SDK's
        usual env config — OPENAI_API_KEY, OPENAI_BASE_URL — selects the
        backend, so any OpenAI-compatible server works). The non-stream
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
                is appended to the managed system prompt.
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
                ``retrieve_model``). The cloud endpoint selects its own.
            max_turns: Local only — cap on agent turns per call.

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
                max_turns=max_turns,
            )
        if model is not None or max_turns is not None:
            raise PageIndexAPIError(
                "model and max_turns are local-mode parameters — the cloud "
                "chat endpoint selects its own model."
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
    ) -> Union[dict[str, Any], Iterator[dict[str, Any]]]:
        """
        Document QA over the OpenAI Responses protocol — the agentic surface.

        Local only for now. Drives your backend's /responses end to end (no
        translation layer), so the ``output`` carries the whole process as
        standard items — messages, function calls, and function outputs
        (the SDK executes the tools). Append the returned ``output`` to your
        next call's ``input`` verbatim to keep provider prompt-cache prefix
        continuity and the agent's memory of what it already read.

        Requires ``pageindex[openai]`` and a backend that supports the
        Responses API; backends that only speak chat.completions should use
        ``chat_completions()``.

        Args:
            input: A user message string, or a list of Responses input items
                (round-trip prior ``output`` items here).
            model: Backend model name (defaults to ``retrieve_model``).
            stream: Yield Responses stream events as dicts — one logical
                response per call: per-turn backend lifecycle events are
                collapsed and sequence numbers reassigned monotonically;
                tool outputs are emitted as ``response.output_item.done``
                events and the single final event is ``response.completed``.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call and is part
                of the cached prompt prefix.
            instructions: Appended to the managed system prompt.
            temperature / top_p: Passed through to the model.
            max_turns: Cap on agent turns per call.
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
            max_turns=max_turns,
        )

    def messages(
        self,
        messages: Union[str, list[dict[str, Any]]],
        model: str,
        max_tokens: int,
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        system: Optional[Union[str, list[dict[str, Any]]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        max_turns: Optional[int] = None,
    ) -> Union[dict[str, Any], Iterator[Any]]:
        """
        Document QA over the Anthropic Messages protocol — Claude-native.

        Local only for now. Drives Anthropic's /v1/messages via the
        Anthropic SDK's own tool runner (requires ``pageindex[anthropic]``;
        ANTHROPIC_API_KEY selects the backend). ``tool_use``/``tool_result``
        round-trip is the format's native behavior: the response is the
        final message envelope with cross-turn aggregated ``usage`` plus a
        ``messages`` field — the full new turn sequence, valid for verbatim
        append to your history. The managed system prompt and the doc
        targeting block carry ``cache_control`` breakpoints.

        Args:
            messages: Native Messages-format history (including prior
                tool_use/tool_result blocks on round-trip), or a bare query
                string (it becomes a single user message).
            model / max_tokens: Required by the Messages API; passed through.
            stream: Yield the Anthropic SDK's event stream across turns
                (its native event objects, including SDK-synthesized
                convenience events), one message sequence per turn.
            doc_id: Document ID or list of IDs to scope the conversation.
                Keep it identical across a conversation's calls — the
                targeting block it adds is re-set each call and is part
                of the cached prompt prefix.
            system: Appended after the managed system blocks.
            temperature / top_p / top_k / stop_sequences: Passed through.
            max_turns: Cap on agent turns per call (default 10, like the
                OpenAI surfaces). A truncated run reports
                ``stop_reason: "tool_use"`` and its ``messages`` remain
                valid for continuation.
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
                        hosted: bool = False) -> list:
        """
        Tools for the OpenAI Agents SDK — pass to ``Agent(tools=...)``.

        Cloud (default): the full live read tool set (search, folders,
        images — as enabled for your key) as plain function tools,
        discovered from the PageIndex MCP server and executed from your
        process — works with any model backend. Pass ``hosted=True`` to
        hand the connection to OpenAI instead: one hosted MCP tool, tool
        calls executed server-side (lowest latency; requires an
        OpenAI-hosted model on the Responses API).

        Local: the in-process tools, any model backend; ``hosted`` does
        not apply. (The framework's own ``MCPServerStreamableHttp``
        against ``{BASE_URL}/mcp`` is the async-native alternative for
        its ``mcp_servers=`` slot.)

        Requires ``openai-agents`` (``pip install 'pageindex[openai]'``),
        imported only when this method is called.

        Args:
            include_management (bool): Also expose tools that modify the
                library (delete, upload). Default off: the cloud default
                serves only server-annotated read-only tools, and
                ``hosted=True`` routes non-read-only tools through the
                Responses API approval flow instead.
            hosted (bool): Cloud only — hand the MCP connection to OpenAI
                for server-side tool execution (OpenAI models only).
        """
        from .integrations.openai_agents import build_openai_tools
        return build_openai_tools(self, include_management, hosted)

    def as_anthropic_tools(self, include_management: bool = False,
                           asynchronous: bool = False) -> list:
        """
        Runnable tools for the Anthropic SDK's tool runner — pass to
        ``client.beta.messages.tool_runner(tools=...)``. The default
        flavor is for the sync ``Anthropic`` client; pass
        ``asynchronous=True`` for ``AsyncAnthropic``. For a manual
        ``messages.create`` loop, serialize with
        ``[tool.to_dict() for tool in ...]``.

        Cloud: the full live read tool set (search, folders, images — as
        enabled for your key), discovered from the PageIndex MCP server
        and executed from your process; the server's input schemas pass
        through verbatim (MCP and the Messages API share the schema
        shape). The server-side alternative is the Messages API's beta
        MCP connector — ``mcp_servers=[{"type": "url", "name":
        "pageindex", "url": f"{BASE_URL}/mcp", "authorization_token":
        <your PageIndex API key>}]`` — with no client-side tools
        involved. Local: the in-process tools — the same set
        ``messages()`` runs internally.

        Requires ``anthropic>=0.68.0``
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
        """
        from .integrations.anthropic_sdk import build_anthropic_tools
        return build_anthropic_tools(self, include_management, asynchronous)

    def as_claude_mcp(self, include_management: bool = False):
        """
        ``mcp_servers`` entry for the Claude Agent SDK.

        Cloud: returns the remote PageIndex MCP config — the framework
        connects to api.pageindex.ai/mcp directly and discovers the full
        cloud tool set. ``include_management`` has no effect there; gate
        destructive tools with the framework's permission layer (e.g. list
        read tools in ``allowed_tools`` instead of the ``*`` wildcard, or
        add ``disallowed_tools=["mcp__pageindex__remove_document"]``).
        Local: returns an in-process SDK MCP server exposing the agent
        tools (requires ``claude-agent-sdk``;
        ``pip install 'pageindex[claude]'``).

        Cloud hosts that surface MCP server instructions receive the same
        guidance ``agent_instructions()`` returns natively — passing both
        duplicates the text (harmless). ``system_prompt`` stays the
        recommended channel: it is guaranteed delivery, carries ``doc_id``
        targeting, and is the only channel local mode has.

        Usage::

            options = ClaudeAgentOptions(
                mcp_servers={"pageindex": client.as_claude_mcp()},
                allowed_tools=["mcp__pageindex__*"],
            )
        """
        from .integrations.claude_agent_sdk import build_claude_mcp
        return build_claude_mcp(self, include_management)

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
        model: Optional[str] = None,
        summary_model: Optional[str] = None,
        retrieve_model: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        super().__init__(None, model=model, summary_model=summary_model,
                         retrieve_model=retrieve_model, storage_path=storage_path)
