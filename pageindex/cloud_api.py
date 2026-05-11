from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from .errors import PageIndexAPIError


class LegacyCloudAPI:
    """Compatibility layer for the pageindex 0.2.x cloud SDK API."""

    BASE_URL = "https://api.pageindex.ai"

    def __init__(self, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL

    def _headers(self) -> dict[str, str]:
        return {"api_key": self.api_key}

    def _request(self, method: str, path: str, error_prefix: str, **kwargs) -> requests.Response:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                headers=self._headers(),
                **kwargs,
            )
        except requests.RequestException as e:
            raise PageIndexAPIError(f"{error_prefix}: {e}") from e

        if response.status_code != 200:
            raise PageIndexAPIError(f"{error_prefix}: {response.text}")
        return response

    def submit_document(
        self,
        file_path: str,
        mode: str | None = None,
        beta_headers: list[str] | None = None,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"if_retrieval": True}
        if mode is not None:
            data["mode"] = mode
        if beta_headers is not None:
            data["beta_headers"] = json.dumps(beta_headers)
        if folder_id is not None:
            data["folder_id"] = folder_id

        with open(file_path, "rb") as f:
            response = self._request(
                "POST",
                "/doc/",
                "Failed to submit document",
                files={"file": f},
                data=data,
            )

        return response.json()

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        if format not in ["page", "node", "raw"]:
            raise ValueError("Format parameter must be 'page', 'node', or 'raw'")

        response = self._request(
            "GET",
            f"/doc/{doc_id}/?type=ocr&format={format}",
            "Failed to get OCR result",
        )
        return response.json()

    def get_tree(self, doc_id: str, node_summary: bool = False) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/doc/{doc_id}/?type=tree&summary={node_summary}",
            "Failed to get tree result",
        )
        return response.json()

    def is_retrieval_ready(self, doc_id: str) -> bool:
        try:
            result = self.get_tree(doc_id)
            return result.get("retrieval_ready", False)
        except PageIndexAPIError:
            return False

    def submit_query(self, doc_id: str, query: str, thinking: bool = False) -> dict[str, Any]:
        payload = {
            "doc_id": doc_id,
            "query": query,
            "thinking": thinking,
        }
        response = self._request(
            "POST",
            "/retrieval/",
            "Failed to submit retrieval",
            json=payload,
        )
        return response.json()

    def get_retrieval(self, retrieval_id: str) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/retrieval/{retrieval_id}/",
            "Failed to get retrieval result",
        )
        return response.json()

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        doc_id: str | list[str] | None = None,
        temperature: float | None = None,
        stream_metadata: bool = False,
        enable_citations: bool = False,
    ) -> dict[str, Any] | Iterator[str] | Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
        }

        if doc_id is not None:
            payload["doc_id"] = doc_id
        if temperature is not None:
            payload["temperature"] = temperature
        if enable_citations:
            payload["enable_citations"] = enable_citations

        response = self._request(
            "POST",
            "/chat/completions/",
            "Failed to get chat completion",
            json=payload,
            stream=stream,
        )

        if stream:
            if stream_metadata:
                return self._stream_chat_response_raw(response)
            return self._stream_chat_response(response)
        return response.json()

    def _stream_chat_response(self, response: requests.Response) -> Iterator[str]:
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    yield content
        except requests.RequestException as e:
            raise PageIndexAPIError(f"Failed to stream chat completion: {e}") from e
        finally:
            response.close()

    def _stream_chat_response_raw(self, response: requests.Response) -> Iterator[dict[str, Any]]:
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
        except requests.RequestException as e:
            raise PageIndexAPIError(f"Failed to stream chat completion: {e}") from e
        finally:
            response.close()

    def get_document(self, doc_id: str) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/doc/{doc_id}/metadata/",
            "Failed to get document metadata",
        )
        return response.json()

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        response = self._request(
            "DELETE",
            f"/doc/{doc_id}/",
            "Failed to delete document",
        )
        return response.json()

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if folder_id is not None:
            params["folder_id"] = folder_id

        response = self._request(
            "GET",
            "/docs/",
            "Failed to list documents",
            params=params,
        )
        return response.json()

    def create_folder(
        self,
        name: str,
        description: str | None = None,
        parent_folder_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if parent_folder_id is not None:
            payload["parent_folder_id"] = parent_folder_id

        response = self._request(
            "POST",
            "/folder/",
            "Failed to create folder",
            json=payload,
        )
        return response.json()

    def list_folders(self, parent_folder_id: str | None = None) -> dict[str, Any]:
        params = {}
        if parent_folder_id is not None:
            params["parent_folder_id"] = parent_folder_id

        response = self._request(
            "GET",
            "/folders/",
            "Failed to list folders",
            params=params,
        )
        return response.json()
