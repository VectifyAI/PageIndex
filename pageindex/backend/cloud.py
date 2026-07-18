# pageindex/backend/cloud.py
"""CloudBackend — connects to PageIndex cloud service (api.pageindex.ai).

API reference: https://github.com/VectifyAI/pageindex_sdk
"""
from __future__ import annotations
import json
import logging
import os
import re
import time
import urllib.parse
import requests
from typing import AsyncIterator

from ..cloud_api import API_BASE  # single source of truth for the cloud base URL
from ..errors import (AUTH_HINT, CloudAPIError, CollectionNotFoundError,
                      DocumentNotFoundError, PageIndexError)
from ..events import QueryEvent

logger = logging.getLogger(__name__)

_INTERNAL_TOOLS = frozenset({"ToolSearch", "Read", "Grep", "Glob", "Bash", "Edit", "Write"})


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CloudBackend:
    def __init__(self, api_key: str, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url or API_BASE
        self._headers = {"api_key": api_key}
        self._folder_id_cache: dict[str, str | None] = {}
        self._folder_warning_shown = False

    # ── HTTP helpers ──────────────────────────────────────────────────────

    # Folder API statuses meaning "folders are not available on this account"
    # (403: requires Max plan; 404: endpoint not exposed). Anything else is a
    # real error and must propagate rather than silently degrade.
    _FOLDER_UNAVAILABLE = (403, 404)

    def _warn_folder_upgrade(self) -> None:
        if not self._folder_warning_shown:
            import warnings
            warnings.warn(
                "Folders (collections) are not available on this plan. "
                "All documents are stored in a single global space — collection names are ignored. "
                "Upgrade at https://dash.pageindex.ai/subscription",
                UserWarning,
                stacklevel=4,
            )
            self._folder_warning_shown = True

    def _request(self, method: str, path: str, retries: int = 3, **kwargs) -> dict:
        """HTTP helper. ``retries`` caps total attempts — pass 1 for
        non-idempotent, expensive calls (e.g. chat completions) where a
        retry would redo the full server-side work."""
        url = f"{self._base_url}{path}"
        kwargs.setdefault("timeout", 30)
        last_status: int | None = None
        for attempt in range(retries):
            if attempt and "files" in kwargs:
                # Rewind file objects before a retry — the previous attempt
                # consumed them, and re-sending without seek(0) would upload
                # an empty multipart body.
                for value in kwargs["files"].values():
                    fobj = value[1] if isinstance(value, tuple) else value
                    if hasattr(fobj, "seek"):
                        fobj.seek(0)
            try:
                resp = requests.request(method, url, headers=self._headers, **kwargs)
                if resp.status_code in (429, 500, 502, 503):
                    last_status = resp.status_code
                    if attempt == retries - 1:
                        break
                    logger.warning("Cloud API %s %s returned %d, retrying...", method, path, resp.status_code)
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    body = resp.text[:500] if resp.text else ""
                    msg = f"Cloud API error {resp.status_code}: {body}"
                    if resp.status_code == 401:
                        msg += f" — {AUTH_HINT}"
                    raise CloudAPIError(msg, status_code=resp.status_code)
                return resp.json() if resp.content else {}
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise CloudAPIError(f"Cloud API request failed: {e}") from e
                time.sleep(2 ** attempt)
        raise CloudAPIError(f"Cloud API {method} {path} failed after retries"
                            + (f" (last status {last_status})" if last_status else ""),
                            status_code=last_status)

    @staticmethod
    def _validate_collection_name(name: str) -> None:
        # .fullmatch() (not .match()): a $-anchored .match() would accept a
        # trailing newline ("papers\n") because $ matches just before a final \n.
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', name):
            raise PageIndexError(
                f"Invalid collection name: {name!r}. "
                "Must be 1-128 chars of [a-zA-Z0-9_-]."
            )

    @staticmethod
    def _enc(value: str) -> str:
        return urllib.parse.quote(value, safe="")

    # ── Collection management (mapped to folders) ─────────────────────────

    def _create_folder(self, name: str) -> str:
        """POST /folder/ and return the new folder id, never a falsy value."""
        resp = self._request("POST", "/folder/", json={"name": name})
        folder_id = resp.get("folder", {}).get("id")
        if not folder_id:
            raise PageIndexError(
                f"Cloud API returned no folder id when creating {name!r} "
                f"(response keys: {list(resp)})"
            )
        return folder_id

    def create_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        try:
            self._folder_id_cache[name] = self._create_folder(name)
        except CloudAPIError as e:
            if e.status_code in self._FOLDER_UNAVAILABLE:
                self._warn_folder_upgrade()
                self._folder_id_cache[name] = None
            else:
                raise

    def get_or_create_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        try:
            data = self._request("GET", "/folders/")
            for folder in data.get("folders", []) or []:
                if folder.get("name") == name:
                    self._folder_id_cache[name] = folder["id"]
                    return
            self._folder_id_cache[name] = self._create_folder(name)
        except CloudAPIError as e:
            if e.status_code in self._FOLDER_UNAVAILABLE:
                self._warn_folder_upgrade()
                self._folder_id_cache[name] = None
            else:
                raise

    def _get_folder_id(self, name: str) -> str | None:
        """Resolve collection name to folder ID. Returns None if folders are
        not available on this plan; raises CollectionNotFoundError if folders
        are available but no folder has this name.

        Only "folders unavailable on this plan" (403/404) is cached as None —
        transient errors (network, 5xx) propagate so a blip can't silently
        drop documents into the global space forever.
        """
        if name in self._folder_id_cache:
            return self._folder_id_cache.get(name)
        try:
            data = self._request("GET", "/folders/")
        except CloudAPIError as e:
            if e.status_code in self._FOLDER_UNAVAILABLE:
                self._warn_folder_upgrade()
                self._folder_id_cache[name] = None
                return None
            raise
        for folder in data.get("folders", []) or []:
            if folder.get("name") == name:
                self._folder_id_cache[name] = folder["id"]
                return folder["id"]
        raise CollectionNotFoundError(
            f"Collection '{name}' does not exist; "
            f"create it first (e.g. client.collection('{name}'))."
        )

    def list_collections(self) -> list[str]:
        try:
            data = self._request("GET", "/folders/")
        except CloudAPIError as e:
            if e.status_code in self._FOLDER_UNAVAILABLE:
                self._warn_folder_upgrade()
                return []
            raise
        return [f["name"] for f in data.get("folders", []) or []]

    def delete_collection(self, name: str) -> None:
        try:
            folder_id = self._get_folder_id(name)
        except CollectionNotFoundError:
            return  # already gone — delete is idempotent
        if folder_id:
            self._request("DELETE", f"/folder/{self._enc(folder_id)}/")
            # Drop the cached id so a later same-name op re-resolves instead of
            # reusing the now-deleted folder_id. Only when it was a REAL id —
            # if folder_id was falsy, the cache holds the "folders unavailable
            # on this plan" None sentinel, which must survive so we don't
            # re-issue a doomed GET /folders/ on the next call.
            self._folder_id_cache.pop(name, None)

    # ── Document management ───────────────────────────────────────────────

    def add_document(self, collection: str, file_path: str) -> str:
        folder_id = self._get_folder_id(collection)
        data = {"if_retrieval": "true"}
        if folder_id:
            data["folder_id"] = folder_id

        with open(file_path, "rb") as f:
            resp = self._request("POST", "/doc/", files={"file": f}, data=data)

        doc_id = resp.get("doc_id")
        if not doc_id:
            raise CloudAPIError("Cloud API upload response missing 'doc_id'")

        # Poll until indexing completes. The cloud API signals readiness via
        # status == "completed"; retrieval_ready is not a reliable indicator.
        for _ in range(120):  # 10 min max
            tree_resp = self._request("GET", f"/doc/{self._enc(doc_id)}/", params={"type": "tree"})
            status = tree_resp.get("status", "")
            if status == "completed":
                return doc_id
            if status == "failed":
                raise CloudAPIError(f"Document {doc_id} indexing failed")
            time.sleep(5)

        raise CloudAPIError(f"Document {doc_id} indexing timed out")

    def _doc_request(self, doc_id: str, method: str, path: str, **kwargs) -> dict:
        """Doc-scoped request: maps HTTP 404 to DocumentNotFoundError for
        parity with the local backend's error taxonomy."""
        try:
            return self._request(method, path, **kwargs)
        except CloudAPIError as e:
            if e.status_code == 404:
                raise DocumentNotFoundError(f"Document {doc_id} not found") from e
            raise

    def _get_metadata(self, doc_id: str) -> dict:
        return self._doc_request(doc_id, "GET", f"/doc/{self._enc(doc_id)}/metadata/")

    def _require_document(self, collection: str, doc_id: str) -> dict | None:
        """Membership guard (the server's doc endpoints are user-scoped, not
        folder-scoped, so this is checked client-side). Returns the doc's
        metadata, or None when folders are unavailable on this plan."""
        folder_id = self._get_folder_id(collection)
        if folder_id is None:
            return None
        meta = self._get_metadata(doc_id)
        if meta.get("folderId") != folder_id:
            raise DocumentNotFoundError(
                f"Document {doc_id} not found in collection '{collection}'"
            )
        return meta

    def _require_documents(self, collection: str, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            self._require_document(collection, doc_id)

    def get_document(self, collection: str, doc_id: str, include_text: bool = False) -> dict:
        if include_text:
            import warnings
            warnings.warn(
                "include_text is not supported by the cloud backend; "
                "returning the structure without node text. "
                "Use get_page_content(doc_id, pages) to fetch content.",
                UserWarning,
                stacklevel=3,
            )
        resp = self._require_document(collection, doc_id)
        if resp is None:
            resp = self._get_metadata(doc_id)
        # Fetch structure in the same call via tree endpoint
        tree_resp = self._doc_request(doc_id, "GET", f"/doc/{self._enc(doc_id)}/",
                                      params={"type": "tree", "summary": "true"})
        raw_tree = tree_resp.get("tree", tree_resp.get("structure", tree_resp.get("result", [])))
        return {
            "doc_id": resp.get("id", doc_id),
            "doc_name": resp.get("name", ""),
            "doc_description": resp.get("description", ""),
            "doc_type": "pdf",
            "status": resp.get("status", ""),
            "structure": self._normalize_tree(raw_tree),
        }

    def get_document_structure(self, collection: str, doc_id: str) -> list:
        self._require_document(collection, doc_id)
        resp = self._doc_request(doc_id, "GET", f"/doc/{self._enc(doc_id)}/",
                                 params={"type": "tree", "summary": "true"})
        raw_tree = resp.get("tree", resp.get("structure", resp.get("result", [])))
        return self._normalize_tree(raw_tree)

    def get_page_content(self, collection: str, doc_id: str, pages: str) -> list:
        self._require_document(collection, doc_id)
        resp = self._doc_request(doc_id, "GET", f"/doc/{self._enc(doc_id)}/",
                                 params={"type": "ocr", "format": "page"})
        # Filter to requested pages
        from ..index.utils import parse_pages
        page_nums = set(parse_pages(pages))
        all_pages = resp.get("pages", resp.get("ocr", resp.get("result", [])))
        if not isinstance(all_pages, list):
            return []
        result = []
        for p in all_pages:
            page = _as_int(p.get("page", p.get("page_index")))
            if page not in page_nums:
                continue
            entry = {"page": page,
                     "content": p.get("content", p.get("markdown", ""))}
            # Cloud OCR pages carry an `images` list (empty on text-only
            # pages). Preserve it — omitting when empty, mirroring the local
            # backend — so cloud callers get the same PageContent shape and
            # the SDK-prompted UI can render figures.
            if p.get("images"):
                entry["images"] = p["images"]
            result.append(entry)
        return result

    @staticmethod
    def _normalize_tree(nodes: list | None) -> list:
        """Normalize cloud tree nodes to match local schema."""
        if not nodes:
            return []
        result = []
        for node in nodes:
            normalized = {
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "summary": node.get("summary", node.get("prefix_summary", "")),
                "start_index": node.get("start_index", node.get("page_index")),
                "end_index": node.get("end_index", node.get("page_index")),
            }
            if "text" in node:
                normalized["text"] = node["text"]
            children = node.get("nodes", [])
            if children:
                normalized["nodes"] = CloudBackend._normalize_tree(children)
            result.append(normalized)
        return result

    def list_documents(self, collection: str) -> list[dict]:
        folder_id = self._get_folder_id(collection)
        # The API caps `limit` at 100; paginate with `offset` until a short
        # page comes back so collections with >100 docs aren't silently
        # truncated (queries over the whole collection rely on this list).
        page_size = 100
        offset = 0
        docs: list[dict] = []
        while True:
            params = {"limit": page_size, "offset": offset}
            if folder_id:
                params["folder_id"] = folder_id
            data = self._request("GET", "/docs/", params=params)
            batch = data.get("documents", []) or []
            docs.extend(
                {
                    "doc_id": d.get("id", ""),
                    "doc_name": d.get("name", ""),
                    "doc_description": d.get("description", ""),
                    "doc_type": "pdf",
                }
                for d in batch
            )
            if len(batch) < page_size:
                return docs
            offset += page_size

    def delete_document(self, collection: str, doc_id: str) -> None:
        self._require_document(collection, doc_id)
        self._doc_request(doc_id, "DELETE", f"/doc/{self._enc(doc_id)}/")

    # ── Query (uses cloud chat/completions, no LLM key needed) ────────────

    def query(self, collection: str, question: str,
              doc_ids: str | list[str] | None = None) -> str:
        """Non-streaming query via cloud chat/completions."""
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        elif doc_ids == []:
            raise ValueError(
                "doc_ids cannot be empty; pass None to query the whole collection"
            )
        if doc_ids:
            self._require_documents(collection, doc_ids)
        doc_id = doc_ids if doc_ids else self._get_all_doc_ids(collection)
        if not doc_id:
            raise ValueError("collection has no documents to query")
        # A non-streaming completion returns nothing until generation
        # finishes, so it needs far more than the default 30s. retries=1:
        # retrying this non-idempotent call would redo the full server-side
        # retrieval + generation (and bill it) on every attempt.
        resp = self._request("POST", "/chat/completions/", retries=1, timeout=300, json={
            "messages": [{"role": "user", "content": question}],
            "doc_id": doc_id,
            "stream": False,
        })
        # Extract answer from response
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return resp.get("content", resp.get("answer", ""))

    async def query_stream(self, collection: str, question: str,
                           doc_ids: str | list[str] | None = None) -> AsyncIterator[QueryEvent]:
        """Streaming query via cloud chat/completions SSE.

        Events are yielded in real-time as they arrive from the server.
        A background thread handles the blocking HTTP stream and pushes
        events through an asyncio.Queue for true async streaming.
        """
        import asyncio
        import threading

        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        elif doc_ids == []:
            raise ValueError(
                "doc_ids cannot be empty; pass None to query the whole collection"
            )
        if doc_ids:
            await asyncio.to_thread(self._require_documents, collection, doc_ids)
        doc_id = doc_ids if doc_ids else await asyncio.to_thread(
            self._get_all_doc_ids, collection
        )
        if not doc_id:
            raise ValueError("collection has no documents to query")
        headers = self._headers
        base_url = self._base_url
        # Queue carries QueryEvent, an Exception to re-raise, or None (end).
        queue: asyncio.Queue[QueryEvent | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Set when the consumer stops early (break / GeneratorExit) so the
        # background thread stops draining the SSE stream instead of pulling
        # it to completion in the background.
        stop = threading.Event()
        resp_holder: dict[str, requests.Response] = {}

        def _put(item: QueryEvent | Exception | None) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass  # event loop already closed; consumer is gone

        def _stream():
            """Background thread: read SSE and push events to queue.

            Everything — including the initial connect — runs inside try so a
            failure can never die silently and leave the consumer awaiting a
            sentinel that never arrives. Errors are forwarded as exceptions
            (raised in the consumer), never disguised as answer events.
            """
            resp = None
            answer_parts: list[str] = []
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions/",
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": question}],
                        "doc_id": doc_id,
                        "stream": True,
                        "stream_metadata": True,
                    },
                    stream=True,
                    timeout=120,
                )
                resp_holder["resp"] = resp
                # The consumer may have abandoned the stream while we were still
                # blocked in requests.post() (its connect phase, before resp
                # existed to close). Now that resp exists, bail immediately
                # rather than reading/draining a stream nobody is listening to;
                # the finally block closes resp and pushes the sentinel.
                if stop.is_set():
                    return
                if resp.status_code != 200:
                    body = resp.text[:500] if resp.text else ""
                    msg = f"Cloud streaming error {resp.status_code}: {body}"
                    if resp.status_code == 401:
                        msg += f" — {AUTH_HINT}"
                    raise CloudAPIError(msg, status_code=resp.status_code)

                current_tool_name = None
                current_tool_args: list[str] = []

                for line in resp.iter_lines(decode_unicode=True):
                    if stop.is_set():
                        return  # consumer abandoned the stream
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    meta = chunk.get("block_metadata", {})
                    block_type = meta.get("type", "")
                    choices = chunk.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}
                    content = delta.get("content", "")

                    if block_type == "mcp_tool_use_start":
                        current_tool_name = meta.get("tool_name", "")
                        current_tool_args = []

                    elif block_type == "tool_use":
                        if content:
                            current_tool_args.append(content)

                    elif block_type == "tool_use_stop":
                        if current_tool_name and current_tool_name not in _INTERNAL_TOOLS:
                            args_str = "".join(current_tool_args)
                            _put(QueryEvent(type="tool_call", data={
                                "name": current_tool_name,
                                "args": args_str,
                            }))
                        current_tool_name = None
                        current_tool_args = []

                    elif block_type == "text" and content:
                        answer_parts.append(content)
                        _put(QueryEvent(type="text_delta", data=content))

                # The whole cloud answer is one text message, so its text_done
                # carries the full answer text.
                _put(QueryEvent(type="text_done", data="".join(answer_parts)))

            except requests.RequestException as e:
                _put(CloudAPIError(f"Cloud streaming request failed: {e}"))
            except Exception as e:
                _put(e)
            finally:
                if resp is not None:
                    resp.close()
                _put(None)  # sentinel

        thread = threading.Thread(target=_stream, daemon=True)
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # On early break / GeneratorExit / raised error: tell the thread to
            # stop and force-close the response so a read blocked mid-stream
            # unblocks instead of draining the rest in the background.
            stop.set()
            resp = resp_holder.get("resp")
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    # Best-effort: the response may already be closed/invalid
                    # during teardown; closing is just to unblock the thread.
                    logger.debug("Ignoring error closing streaming response during cleanup",
                                 exc_info=True)
            try:
                await asyncio.to_thread(thread.join, 5)
            except RuntimeError:
                # event loop already shutting down — fall back to a bounded sync join
                thread.join(timeout=5)

    def _get_all_doc_ids(self, collection: str) -> list[str]:
        """Get all document IDs in a collection."""
        docs = self.list_documents(collection)
        return [d["doc_id"] for d in docs]
