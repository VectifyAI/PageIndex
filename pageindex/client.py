"""PageIndex SDK client: the 0.2.x cloud surface, now with a local mode."""
from __future__ import annotations

from typing import Any, Iterator, Optional, Union

from .errors import PageIndexAPIError


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
        retrieve_model (str, optional): Local mode only — LLM used for
            retrieval tree search and chat completions.
        storage_path (str, optional): Local mode only — directory where
            indexed documents are stored. Defaults to ``./.pageindex``.

    Usage:
        client = PageIndexClient(api_key="...")   # cloud
        client = PageIndexClient()                # local

    Local mode differences (all documented per method): indexing is
    synchronous, only PDFs are supported, and folders / ``beta_headers`` /
    ``enable_citations`` / the deprecated retrieval API (``submit_query``,
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
                raise ValueError(
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
                          "retrieve_model": retrieve_model}.items() if value}
            opt = ConfigLoader().load(overrides or None)
            self.model = opt.model
            self.summary_model = getattr(opt, "summary_model", None) or opt.model
            self.retrieve_model = getattr(opt, "retrieve_model", None) or opt.model
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
    ) -> dict[str, Any]:
        """
        Submit a PDF document for processing. Returns {'doc_id': ...}.

        Cloud: uploads the file; processing is asynchronous — poll
        ``is_retrieval_ready(doc_id)`` before retrieving.

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
                "flash"; cloud modes are passed through (e.g. "mcp").
            beta_headers (list[str], optional): Cloud-only beta feature headers.
            folder_id (str, optional): Cloud-only folder (workspace) ID.
            metadata (dict, optional): Your own JSON-serializable tags for the
                document; returned in get_tree/get_ocr responses and
                list_documents entries (both modes).

        Returns:
            dict: {'doc_id': ...}
        """
        return self._api.submit_document(
            file_path=file_path, mode=mode,
            beta_headers=beta_headers, folder_id=folder_id, metadata=metadata,
        )

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

    # ---------- TREE GENERATION ----------

    def get_tree(self, doc_id: str, node_summary: bool = False) -> dict[str, Any]:
        """
        Get tree generation status and results.

        Args:
            doc_id (str): Document ID.
            node_summary (bool): Include node summaries in the tree.

        Returns:
            dict: {'doc_id', 'status', 'retrieval_ready', 'result', ...} where
            result nodes are {'title', 'node_id', 'page_index', ('summary' /
            'prefix_summary',) ('text',) 'nodes'}.

        Local: status is always "completed" (indexing happened during
        ``submit_document``) and nodes always carry 'text'.
        """
        return self._api.get_tree(doc_id=doc_id, node_summary=node_summary)

    def is_retrieval_ready(self, doc_id: str) -> bool:
        """
        Check if a document is ready for retrieval. Errors (including a
        missing document) are reported as False, matching the 0.2.x SDK.
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
            "favor of chat completions; use chat_completions in local mode."
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
            "favor of chat completions; use chat_completions in local mode."
        ).get_retrieval(retrieval_id=retrieval_id)

    # ---------- CHAT COMPLETIONS ----------

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        doc_id: Optional[Union[str, list[str]]] = None,
        temperature: Optional[float] = None,
        stream_metadata: bool = False,
        enable_citations: bool = False,
    ) -> Union[dict[str, Any], Iterator[str], Iterator[dict[str, Any]]]:
        """
        PageIndex Chat Completions, scoped to specific PageIndex documents.

        Args:
            messages: Conversation messages with 'role' and 'content' keys.
            stream: Enable streaming responses.
            doc_id: Document ID or list of IDs to scope the conversation.
                Required in local mode.
            temperature: Sampling temperature (0.0-1.0).
            stream_metadata: With stream=True, yield chunk dicts instead of
                text pieces.
            enable_citations: Cloud-only citation support; raises in local mode.

        Returns:
            - stream=False: complete response dict ({'id', 'object', 'created',
              'choices', 'usage'})
            - stream=True, stream_metadata=False: iterator of text chunks
            - stream=True, stream_metadata=True: iterator of chunk dicts

        Local: the question (last user message) is answered by tree-search
        retrieval over the given documents plus a ``retrieve_model`` chat call.
        """
        return self._api.chat_completions(
            messages=messages, stream=stream, doc_id=doc_id,
            temperature=temperature, stream_metadata=stream_metadata,
            enable_citations=enable_citations,
        )

    # ---------- DOCUMENT MANAGEMENT ----------

    def get_document(self, doc_id: str) -> dict[str, Any]:
        """
        Get document metadata: {'id', 'name', 'description', 'status',
        'createdAt', 'pageNum', 'folderId'}. Status is one of "pending",
        "queued", "processing", "completed", "failed" (local documents are
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
            dict: {'message': 'Document deleted successfully.'}
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
