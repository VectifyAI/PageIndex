import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx


DocId = Union[str, List[str]]
Message = Dict[str, Any]


class PageIndexClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PAGEINDEX_API_KEY")
        self.base_url = (base_url or os.getenv("PAGEINDEX_BASE_URL") or "https://api.pageindex.ai").rstrip("/")

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["api_key"] = self.api_key

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PageIndexClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = self._client.request(method, url, **kwargs)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    def submit_document(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as file:
            files = {"file": (os.path.basename(file_path), file, "application/pdf")}
            return self._request_json("POST", "/doc/", files=files)

    def submit_markdown(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as file:
            files = {"file": (os.path.basename(file_path), file, "text/markdown")}
            return self._request_json("POST", "/markdown/", files=files)

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/doc/{doc_id}/")

    def get_document_status(self, doc_id: str) -> Optional[str]:
        return self.get_document(doc_id).get("status")

    def get_tree(self, doc_id: str, summary: bool = False) -> Dict[str, Any]:
        params = {"type": "tree", "summary": summary}
        return self._request_json("GET", f"/doc/{doc_id}/", params=params)

    def get_ocr(self, doc_id: str, format: str = "page") -> Dict[str, Any]:
        params = {"type": "ocr", "format": format}
        return self._request_json("GET", f"/doc/{doc_id}/", params=params)

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        response = self._client.request("DELETE", f"/doc/{doc_id}/")
        response.raise_for_status()
        if not response.content:
            return {"status": "deleted"}
        return response.json()

    def is_retrieval_ready(self, doc_id: str) -> bool:
        try:
            result = self.get_tree(doc_id)
        except httpx.HTTPError:
            return False
        return bool(result.get("retrieval_ready"))

    def submit_retrieval_query(self, doc_id: str, query: str, thinking: bool = False) -> Dict[str, Any]:
        payload = {"doc_id": doc_id, "query": query, "thinking": thinking}
        return self._request_json("POST", "/retrieval/", json=payload)

    def get_retrieval_result(self, retrieval_id: str) -> Dict[str, Any]:
        return self._request_json("GET", f"/retrieval/{retrieval_id}/")

    def chat_completions(self, messages: List[Message], doc_id: Optional[DocId] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages, "stream": False}
        if doc_id is not None:
            payload["doc_id"] = doc_id
        return self._request_json("POST", "/chat/completions", json=payload)

    def chat_completions_stream(
        self, messages: List[Message], doc_id: Optional[DocId] = None
    ) -> Iterator[Dict[str, Any]]:
        payload: Dict[str, Any] = {"messages": messages, "stream": True}
        if doc_id is not None:
            payload["doc_id"] = doc_id

        with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield json_loads_safe(data)


class AsyncPageIndexClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("PAGEINDEX_API_KEY")
        self.base_url = (base_url or os.getenv("PAGEINDEX_BASE_URL") or "https://api.pageindex.ai").rstrip("/")

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["api_key"] = self.api_key

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncPageIndexClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        response = await self._client.request(method, url, **kwargs)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    async def submit_document(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as file:
            files = {"file": (os.path.basename(file_path), file.read(), "application/pdf")}
        return await self._request_json("POST", "/doc/", files=files)

    async def submit_markdown(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as file:
            files = {"file": (os.path.basename(file_path), file.read(), "text/markdown")}
        return await self._request_json("POST", "/markdown/", files=files)

    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        return await self._request_json("GET", f"/doc/{doc_id}/")

    async def get_document_status(self, doc_id: str) -> Optional[str]:
        return (await self.get_document(doc_id)).get("status")

    async def get_tree(self, doc_id: str, summary: bool = False) -> Dict[str, Any]:
        params = {"type": "tree", "summary": summary}
        return await self._request_json("GET", f"/doc/{doc_id}/", params=params)

    async def get_ocr(self, doc_id: str, format: str = "page") -> Dict[str, Any]:
        params = {"type": "ocr", "format": format}
        return await self._request_json("GET", f"/doc/{doc_id}/", params=params)

    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        response = await self._client.request("DELETE", f"/doc/{doc_id}/")
        response.raise_for_status()
        if not response.content:
            return {"status": "deleted"}
        return response.json()

    async def is_retrieval_ready(self, doc_id: str) -> bool:
        try:
            result = await self.get_tree(doc_id)
        except httpx.HTTPError:
            return False
        return bool(result.get("retrieval_ready"))

    async def submit_retrieval_query(self, doc_id: str, query: str, thinking: bool = False) -> Dict[str, Any]:
        payload = {"doc_id": doc_id, "query": query, "thinking": thinking}
        return await self._request_json("POST", "/retrieval/", json=payload)

    async def get_retrieval_result(self, retrieval_id: str) -> Dict[str, Any]:
        return await self._request_json("GET", f"/retrieval/{retrieval_id}/")

    async def chat_completions(self, messages: List[Message], doc_id: Optional[DocId] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages, "stream": False}
        if doc_id is not None:
            payload["doc_id"] = doc_id
        return await self._request_json("POST", "/chat/completions", json=payload)

    async def chat_completions_stream(
        self, messages: List[Message], doc_id: Optional[DocId] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        payload: Dict[str, Any] = {"messages": messages, "stream": True}
        if doc_id is not None:
            payload["doc_id"] = doc_id

        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield json_loads_safe(data)


def json_loads_safe(data: str) -> Dict[str, Any]:
    try:
        import json

        value = json.loads(data)
    except Exception:
        return {"raw": data}
    if isinstance(value, dict):
        return value
    return {"value": value}
