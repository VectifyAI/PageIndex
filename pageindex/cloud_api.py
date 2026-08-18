"""Cloud mode of the PageIndex SDK, based on the 0.2.8 client."""
import requests
from typing import Optional, Dict, Any, List, Union, Iterator
import json
import urllib.parse

from .errors import PageIndexAPIError


def _enc(value: str) -> str:
    """URL-encode a path segment (ids may contain / ? # or spaces)."""
    return urllib.parse.quote(str(value), safe="")


class CloudAPI:
    """
    Python SDK client for the PageIndex API.
    """

    def __init__(self, client):
        self._client = client

    @property
    def BASE_URL(self) -> str:
        return self._client.BASE_URL

    @property
    def api_key(self) -> str:
        return self._client.api_key

    def _headers(self) -> Dict[str, str]:
        return {"api_key": self.api_key}

    # ---------- DOCUMENT SUBMISSION ----------

    def submit_document(
        self,
        file_path: str,
        mode: Optional[str] = None,
        beta_headers: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Upload a PDF document for processing. The system will automatically process both tree generation and OCR.
        Immediately returns a document identifier (`doc_id`) for subsequent operations.

        Args:
            file_path (str): Path to the PDF file.
            mode (str, optional): Processing mode (e.g., "mcp"). Defaults to None.
            beta_headers (List[str], optional): Beta feature headers (e.g., ["block_reference"]
                to enable block-level content with bounding boxes). Defaults to None.
            folder_id (str, optional): Folder (workspace) ID to assign the document to. Defaults to None.
            metadata (dict, optional): Your own JSON-serializable tags for the document;
                returned in get_tree/get_ocr responses and list_documents entries. Defaults to None.

        Returns:
            dict: {'doc_id': ...} — plus 'name', the stored document name
                (a taken name gains a numeric suffix), when the server
                returns it.
        """
        data: Dict[str, Any] = {'if_retrieval': True}
        if mode is not None:
            data['mode'] = mode
        if beta_headers is not None:
            data['beta_headers'] = json.dumps(beta_headers)
        if folder_id is not None:
            data['folder_id'] = folder_id
        if metadata is not None:
            data['metadata'] = json.dumps(metadata)

        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.BASE_URL}/doc/",
                headers=self._headers(),
                files={'file': f},
                data=data
            )

        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to submit document: {response.text}")
        return response.json()

    # ---------- OCR FUNCTIONALITY ----------

    def get_ocr(self, doc_id: str, format: str = "page") -> Dict[str, Any]:
        """
        Get OCR processing status and results.

        Args:
            doc_id (str): Document ID.
            format (str): Result format. Use 'page' for page-based results, 'node' for node-based results, or 'raw' for concatenated markdown. Defaults to 'page'.

        Returns:
            dict: API response with status and, if ready, OCR results.
        """
        if format not in ["page", "node", "raw"]:
            raise ValueError("Format parameter must be 'page', 'node', or 'raw'")

        response = requests.get(
            f"{self.BASE_URL}/doc/{_enc(doc_id)}/?type=ocr&format={format}",
            headers=self._headers(),
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to get OCR result: {response.text}")
        return response.json()

    # ---------- TREE GENERATION ----------

    def get_tree(self, doc_id: str, node_summary: bool = False,
                 include_text: bool = True) -> Dict[str, Any]:
        """
        Get tree generation status and results.

        Args:
            doc_id (str): Document ID.
            node_summary (bool): Include node summaries (default False).

        Returns:
            dict: API response with status and, if ready, tree structure.
        """
        response = requests.get(
            f"{self.BASE_URL}/doc/{_enc(doc_id)}/?type=tree&summary={node_summary}"
            f"&include_text={str(include_text).lower()}",
            headers=self._headers(),
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to get tree result: {response.text}")
        return response.json()

    # ---------- RETRIEVAL ----------

    def submit_query(self, doc_id: str, query: str, thinking: bool = False) -> Dict[str, Any]:
        """
        Submit a retrieval query for a specific PageIndex document.

        Args:
            doc_id (str): Document ID.
            query (str): User question or information need.
            thinking (bool, optional): If true, enables deeper retrieval. Default is False.

        Returns:
            dict: {'retrieval_id': ...}
        """
        payload = {
            "doc_id": doc_id,
            "query": query,
            "thinking": thinking
        }
        response = requests.post(
            f"{self.BASE_URL}/retrieval/",
            headers=self._headers(),
            json=payload,
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to submit retrieval: {response.text}")
        return response.json()

    def get_retrieval(self, retrieval_id: str) -> Dict[str, Any]:
        """
        Get retrieval status and results.

        Args:
            retrieval_id (str): Retrieval ID.

        Returns:
            dict: Retrieval status and results.
        """
        response = requests.get(
            f"{self.BASE_URL}/retrieval/{_enc(retrieval_id)}/",
            headers=self._headers(),
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to get retrieval result: {response.text}")
        return response.json()

    # ---------- CHAT COMPLETIONS ----------

    def chat_completions(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        doc_id: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        stream_metadata: bool = False,
        enable_citations: bool = False
    ) -> Union[Dict[str, Any], Iterator[str], Iterator[Dict[str, Any]]]:
        """
        PageIndex Chat Completions. Optionally scoped to specific PageIndex documents.

        Args:
            messages (List[Dict[str, str]]): Conversation messages with 'role' and 'content' keys.
            stream (bool, optional): Enable streaming responses. Default is False.
            doc_id (Optional[Union[str, List[str]]], optional): Document ID(s) to scope the conversation. Can be a single ID or a list of IDs.
            temperature (Optional[float], optional): Sampling temperature. Default is None (uses API default).
            stream_metadata (bool, optional): If True and stream=True, return raw chunks with metadata instead of just text. Default is False.
            enable_citations (bool, optional): Enable citation instructions in responses. Default is False.

        Returns:
            Union[Dict[str, Any], Iterator[str], Iterator[Dict[str, Any]]]:
                - If stream=False: Complete response dictionary
                - If stream=True and stream_metadata=False: Iterator of text content chunks
                - If stream=True and stream_metadata=True: Iterator of raw response chunks with metadata
        """
        payload = {
            "messages": messages,
            "stream": stream
        }

        if doc_id is not None:
            payload["doc_id"] = doc_id

        if temperature is not None:
            payload["temperature"] = temperature

        if enable_citations:
            payload["enable_citations"] = enable_citations

        response = requests.post(
            f"{self.BASE_URL}/chat/completions/",
            headers=self._headers(),
            json=payload,
            stream=stream,
            timeout=120 if stream else 600
        )

        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to get chat completion: {response.text}")

        if stream:
            if stream_metadata:
                return self._stream_chat_response_raw(response)
            else:
                return self._stream_chat_response(response)
        else:
            return response.json()

    def _stream_chat_response(self, response: requests.Response) -> Iterator[str]:
        """
        Parse streaming chat completion response.

        Args:
            response: Streaming HTTP response

        Yields:
            str: Content chunks from the streaming response
        """
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices") or [{}]
                            content = choices[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        finally:
            response.close()

    def _stream_chat_response_raw(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """Streaming chat completion with full metadata, including citation events."""
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        finally:
            response.close()

    # ---------- DOCUMENT MANAGEMENT ----------

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document metadata including id, name, description, status, createdAt, and pageNum.

        Args:
            doc_id (str): Document ID.

        Returns:
            dict: Document metadata containing:
                - id (str): Document ID
                - name (str): Document name
                - description (str): Document description
                - status (str): Processing status (e.g., "queued", "processing", "completed", "failed")
                - createdAt (str): Creation timestamp in ISO format
                - pageNum (int): Number of pages in the document
        """
        response = requests.get(
            f"{self.BASE_URL}/doc/{_enc(doc_id)}/metadata/",
            headers=self._headers(),
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to get document metadata: {response.text}")
        return response.json()

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a PageIndex document and all its associated data.

        Args:
            doc_id (str): Document ID.

        Returns:
            dict: API response.
        """
        response = requests.delete(
            f"{self.BASE_URL}/doc/{_enc(doc_id)}/",
            headers=self._headers(),
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to delete document: {response.text}")
        return response.json() if response.content else {}

    def list_documents(self, limit: int = 50, offset: int = 0, folder_id: Optional[str] = None) -> Dict[str, Any]:
        """
        List all documents for the authenticated user with pagination.

        Args:
            limit (int, optional): Maximum number of documents to return (1-100). Defaults to 50.
            offset (int, optional): Number of documents to skip. Defaults to 0.
            folder_id (str, optional): Filter by folder (workspace) ID. If provided, only documents
                in the specified folder are returned. Defaults to None (all documents).

        Returns:
            dict: API response containing:
                - documents (List[Dict]): List of document metadata objects (each includes folderId)
                - total (int): Total number of documents
                - limit (int): Applied limit
                - offset (int): Applied offset
        """
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if folder_id is not None:
            params["folder_id"] = folder_id

        response = requests.get(
            f"{self.BASE_URL}/docs/",
            headers=self._headers(),
            params=params,
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to list documents: {response.text}")
        return response.json()

    # ---------- FOLDER MANAGEMENT ----------

    def create_folder(
        self,
        name: str,
        description: Optional[str] = None,
        parent_folder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new folder (workspace).

        Args:
            name (str): Folder name.
            description (str, optional): Folder description. Defaults to None.
            parent_folder_id (str, optional): Parent folder ID for nesting. Defaults to None (root level).

        Returns:
            dict: Created folder metadata containing:
                - folder (dict): Folder info with id, name, description, parent_folder_id,
                    created_at, file_count, children_count
        """
        payload = {"name": name}
        if description is not None:
            payload["description"] = description
        if parent_folder_id is not None:
            payload["parent_folder_id"] = parent_folder_id

        response = requests.post(
            f"{self.BASE_URL}/folder/",
            headers=self._headers(),
            json=payload,
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to create folder: {response.text}")
        return response.json()

    def list_folders(self, parent_folder_id: Optional[str] = None) -> Dict[str, Any]:
        """
        List folders.

        Args:
            parent_folder_id (str, optional): Use "root" for root-level folders only,
                a folder ID for subfolders, or omit for all folders.

        Returns:
            dict: API response containing:
                - folders (List[Dict]): List of folder metadata objects
                - total (int): Total number of folders
        """
        params = {}
        if parent_folder_id is not None:
            params["parent_folder_id"] = parent_folder_id

        response = requests.get(
            f"{self.BASE_URL}/folders/",
            headers=self._headers(),
            params=params,
            timeout=30
        )
        if response.status_code != 200:
            raise PageIndexAPIError(f"Failed to list folders: {response.text}")
        return response.json()
