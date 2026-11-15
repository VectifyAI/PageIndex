"""Native async cloud document operations; local indexing and chat stay separate."""
from __future__ import annotations

import asyncio
import json
import math
import os
import warnings
from functools import partial
from pathlib import Path

import anyio
import httpx

from .client import PageIndexClient, _parse_pages
from .cloud_api import _enc
from .errors import PageIndexAPIError
from .naming import sanitize_filename
from .retry import RETRY_STATUS, SDK_MAX_DELAY, SDK_RETRIES, delay


class _AsyncUpload(httpx.AsyncByteStream):
    """Keep HTTPX's multipart encoding, but read file chunks off the loop."""

    def __init__(self, stream):
        self._chunks = iter(stream)

    async def __aiter__(self):
        sentinel = object()
        while True:
            chunk = await anyio.to_thread.run_sync(next, self._chunks, sentinel)
            if chunk is sentinel:
                return
            yield chunk


class AsyncPageIndexCloudClient:
    """Cloud document API with an explicit async connection lifetime.

    Use ``async with`` or ``await aclose()`` on the same event loop as the
    requests. A supplied ``http_client`` is borrowed and is never closed.
    Timeouts bound each connect/read/write/pool wait, not the whole job;
    callers can use ``asyncio.wait_for`` for an overall deadline. Processing
    after upload is separate: poll ``get_document(doc_id)['status']``.

    This is a limited companion to PageIndexClient, not an async local
    indexer or a chat client. No model credentials are used by this class.
    """

    def __init__(self, api_key: str | None = None, *,
                 base_url: str = PageIndexClient.BASE_URL,
                 timeout: float | httpx.Timeout = 30.0,
                 max_retries: int = SDK_RETRIES,
                 http_client: httpx.AsyncClient | None = None):
        key = os.environ.get("PAGEINDEX_API_KEY") if api_key is None else api_key
        if not isinstance(key, str) or not key.strip():
            raise PageIndexAPIError(
                "AsyncPageIndexCloudClient requires a PageIndex API key: "
                "pass api_key=... or set PAGEINDEX_API_KEY.")
        self.api_key = key
        self.BASE_URL = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout)
        if any(value is None or not math.isfinite(value) or value <= 0
               for value in self.timeout.as_dict().values()):
            raise ValueError("Every connect/read/write/pool timeout must be finite and positive")
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be an integer from 0 to 10")
        self.max_retries = max_retries
        self._owns_http = http_client is None
        self._http = http_client if http_client is not None else httpx.AsyncClient()
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed or self._http.is_closed

    def _check_open(self):
        if self.is_closed:
            raise RuntimeError("AsyncPageIndexCloudClient is closed")

    async def __aenter__(self):
        self._check_open()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        if not self._closed:
            if self._owns_http:
                await self._http.aclose()
            self._closed = True

    async def _request(self, method, path, *, operation, params=None,
                       file_path=None, data=None):
        self._check_open()
        for attempt in range(self.max_retries + 1):
            headers = None
            try:
                kwargs = dict(headers={"api_key": self.api_key},
                              params=params, timeout=self.timeout)
                url = f"{self.BASE_URL}{path}"
                if file_path is None:
                    response = await self._http.request(method, url, follow_redirects=False, **kwargs)
                else:
                    # Each permitted retry opens a fresh file. Opening,
                    # seeking and reading it are the only threaded work;
                    # HTTP, backoff and cancellation remain native async.
                    async with await anyio.open_file(file_path, "rb") as file:
                        request = await anyio.to_thread.run_sync(partial(
                            self._http.build_request, method, url, **kwargs,
                            files={"file": (sanitize_filename(Path(file_path).name),
                                            file.wrapped)}, data=data))
                        request.stream = _AsyncUpload(request.stream)
                        response = await self._http.send(request, follow_redirects=False)
                try:
                    if response.status_code == 200:
                        return await anyio.to_thread.run_sync(response.json)
                    retryable = (response.status_code == 429 if method == "POST"
                                 else response.status_code in RETRY_STATUS
                                 or response.status_code >= 500)
                    if not retryable or attempt == self.max_retries:
                        raise PageIndexAPIError(
                            f"Failed to {operation}: {response.text}",
                            status_code=response.status_code)
                    headers = response.headers
                finally:
                    await response.aclose()
            except httpx.TransportError as exc:
                # A POST may have reached the server on a read/write failure.
                # Without an idempotency key, never resend that upload.
                retryable = (isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
                             if method == "POST" else not isinstance(
                                 exc, (httpx.LocalProtocolError, httpx.UnsupportedProtocol)))
                if not retryable or attempt == self.max_retries:
                    raise
            await asyncio.sleep(delay(attempt, headers, SDK_MAX_DELAY))

    async def submit_document(self, file_path: str, mode: str | None = None,
                              beta_headers: list[str] | None = None,
                              folder_id: str | None = None,
                              metadata: dict | None = None) -> dict:
        """Upload once and return the server's dict; no processing wait.

        Only connection failures before sending and HTTP 429 permit a retry.
        A read/write timeout is raised unchanged, with upload outcome unknown.
        """
        data = {"if_retrieval": True}
        if mode is not None:
            data["mode"] = mode
        if beta_headers is not None:
            data["beta_headers"] = json.dumps(beta_headers)
        if folder_id is not None:
            data["folder_id"] = folder_id
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        result = await self._request("POST", "/doc/", operation="submit document",
                                     file_path=file_path, data=data)
        stored = result.get("name")
        if stored and stored != os.path.basename(file_path):
            warnings.warn(f'Document "{os.path.basename(file_path)}" was stored as '
                          f'"{stored}".', stacklevel=2)
        return result

    async def get_document(self, doc_id: str) -> dict:
        """Return metadata, including the processing ``status`` field."""
        return await self._request("GET", f"/doc/{_enc(doc_id)}/metadata/",
                                   operation="get document metadata")

    async def get_tree(self, doc_id: str, node_summary: bool = False,
                       include_text: bool = True) -> dict:
        return await self._request("GET", f"/doc/{_enc(doc_id)}/",
                                   operation="get tree result",
                                   params={"type": "tree", "summary": str(node_summary),
                                           "include_text": str(include_text).lower()})

    async def get_ocr(self, doc_id: str, format: str = "page") -> dict:
        if format not in ("page", "node", "raw"):
            raise ValueError("Format parameter must be 'page', 'node', or 'raw'")
        return await self._request("GET", f"/doc/{_enc(doc_id)}/",
                                   operation="get OCR result",
                                   params={"type": "ocr", "format": format})

    async def get_page_content(self, doc_id: str, pages: str) -> list[dict]:
        wanted = set(_parse_pages(pages))
        result = await self.get_ocr(doc_id, format="page")
        if result["result"] is None:
            raise PageIndexAPIError(
                f"Document '{doc_id}' is not ready (status: {result.get('status', 'unknown')})")
        return [page for page in result["result"] if page["page_index"] in wanted]

    async def list_documents(self, limit: int = 50, offset: int = 0,
                             folder_id: str | None = None, name: str | None = None,
                             recursive: bool = False) -> dict:
        if limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        params = {"limit": limit, "offset": offset}
        if folder_id is not None:
            params["folder_id"] = folder_id
        if name is not None:
            params["name"] = name
        if recursive:
            params["recursive"] = str(recursive)
        return await self._request("GET", "/docs/", operation="list documents", params=params)
