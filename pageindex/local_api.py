"""Local implementation of the PageIndex SDK surface.

Mirrors the response shapes of api.pageindex.ai so code written against the
0.2.x cloud SDK works unchanged, with these documented differences:

- Indexing is synchronous: ``submit_document`` blocks until the tree is built
  (your machine runs the LLM calls), so ``status`` is always ``"completed"``
  and polling loops succeed on the first try.
- Only PDF files are supported.
- Folders, ``beta_headers``, ``enable_citations``, and the deprecated
  retrieval API (``submit_query``/``get_retrieval``) are cloud-only and
  raise; use ``chat_completions`` for retrieval-backed answers.
- Cloud responses carry OCR ``images`` and a ``block_metadata`` key on
  streaming chunks; local responses omit them.
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

from .errors import PageIndexAPIError
from .local_store import DocStore

TREE_SEARCH_PROMPT = """
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{tree}

Please reply in the following JSON format:
{{
    "thinking": "<Your thinking process on which nodes are relevant to the question>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}
Directly return the final JSON structure. Do not output anything else.
"""

CHAT_SYSTEM_PROMPT = """You are PageIndex Chat, a document question-answering assistant.
Answer the user's question based on the retrieved document context below.
If the context does not contain the information needed, say so instead of guessing.

<retrieved_context>
{context}
</retrieved_context>"""

# Retrieved context handed to the chat model is capped to leave room for the
# conversation and the answer.
CHAT_CONTEXT_TOKEN_LIMIT = 100_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalAPI:
    """Backs PageIndexClient's local mode. One instance per client."""

    def __init__(self, storage_path: str, model: str, summary_model: str,
                 retrieve_model: str):
        self._store = DocStore(storage_path)
        self._model = model
        self._summary_model = summary_model
        self._retrieve_model = retrieve_model

    # ── indexing ──

    def submit_document(
        self,
        file_path: str,
        mode: str | None = None,
        beta_headers: list[str] | None = None,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        if beta_headers is not None:
            raise PageIndexAPIError(
                "Failed to submit document: beta_headers is not supported in local mode."
            )
        if folder_id is not None:
            raise PageIndexAPIError(
                "Failed to submit document: folders are not supported in local mode."
            )
        if mode not in (None, "flash"):
            raise PageIndexAPIError(
                f"Failed to submit document: unknown local processing mode {mode!r}. "
                "Supported: None (standard) or 'flash'."
            )
        if not os.path.isfile(file_path):
            # The 0.2.8 client fails on open(); keep the same exception type.
            raise FileNotFoundError(f"No such file: {file_path}")
        if not file_path.lower().endswith(".pdf"):
            raise PageIndexAPIError(
                "Failed to submit document: only PDF files are supported in local mode."
            )

        page_texts = self._extract_page_texts(file_path)
        if not any(text.strip() for text in page_texts):
            raise PageIndexAPIError(
                "Failed to submit document: PDF has no content. All pages are blank."
            )

        try:
            if mode == "flash":
                structure, description = self._index_flash(file_path, page_texts)
            else:
                structure, description = self._index_standard(file_path)
        except PageIndexAPIError:
            raise
        except Exception as e:
            raise PageIndexAPIError(f"Failed to submit document: {e}") from e

        doc_id = str(uuid.uuid4())
        meta = {
            "id": doc_id,
            "name": os.path.basename(file_path),
            "description": description,
            "status": "completed",
            "createdAt": _now_iso(),
            "pageNum": len(page_texts),
            "folderId": None,
            "mode": mode or "standard",
        }
        pages = [{"page_index": i + 1, "markdown": text}
                 for i, text in enumerate(page_texts)]
        self._store.save_document(doc_id, meta, structure, pages)
        return {"doc_id": doc_id}

    @staticmethod
    def _extract_page_texts(file_path: str) -> list[str]:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return [page.extract_text() or "" for page in reader.pages]

    def _index_standard(self, file_path: str) -> tuple[list, str | None]:
        from .page_index import page_index_main
        from .utils import ConfigLoader
        opt = ConfigLoader().load({
            "model": self._model,
            "summary_model": self._summary_model,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_node_text": "yes",
            "if_add_doc_description": "yes",
        })
        result = page_index_main(file_path, opt, logger=_SilentLogger())
        return result["structure"], result.get("doc_description")

    def _index_flash(self, file_path: str, page_texts: list[str]) -> tuple[list, str | None]:
        from .flash import page_index_flash
        from .utils import (add_node_text, create_clean_structure_for_description,
                            generate_doc_description, write_node_id)
        result = page_index_flash(file_path, summary=True,
                                  summary_model=self._summary_model)
        structure = result.get("structure", [])
        if not structure:
            raise PageIndexAPIError(
                "Failed to submit document: PageIndex Flash could not extract "
                "a structure from this PDF."
            )
        write_node_id(structure)
        add_node_text(structure, [(text, 0) for text in page_texts])
        description = generate_doc_description(
            create_clean_structure_for_description(structure),
            model=self._summary_model,
        )
        return structure, description

    # ── tree / ocr ──

    def get_tree(self, doc_id: str, node_summary: bool = False) -> dict[str, Any]:
        self._require_doc(doc_id, "Failed to get tree result")
        structure = copy.deepcopy(self._store.get_tree(doc_id)) or []
        result = [_format_tree_node(node, node_summary) for node in structure]
        return self._completed_envelope(doc_id, result)

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        if format not in ["page", "node", "raw"]:
            raise ValueError("Format parameter must be 'page', 'node', or 'raw'")
        self._require_doc(doc_id, "Failed to get OCR result")
        pages = self._store.get_pages(doc_id) or []
        if format == "page":
            result: Any = pages
        elif format == "raw":
            result = "\n\n".join(p.get("markdown", "") for p in pages)
        else:  # node
            result = []
            def _walk(nodes, level):
                for node in nodes:
                    result.append({
                        "title": node.get("title", ""),
                        "level": level,
                        "page_index": node.get("start_index"),
                        "text": node.get("text", ""),
                    })
                    _walk(node.get("nodes") or [], level + 1)
            _walk(self._store.get_tree(doc_id) or [], 1)
        return self._completed_envelope(doc_id, result)

    def _completed_envelope(self, doc_id: str, result) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "status": "completed",
            "retrieval_ready": True,
            "result": result,
            "metadata": None,
            "features": {},
        }

    # ── document management ──

    def _require_doc(self, doc_id: str, error_prefix: str) -> dict:
        meta = self._store.get_meta(doc_id)
        if meta is None:
            raise PageIndexAPIError(f"{error_prefix}: Document not found.")
        return meta

    def get_document(self, doc_id: str) -> dict[str, Any]:
        meta = self._store.get_meta(doc_id)
        if meta is None:
            # The metadata endpoint's cloud message has no trailing period.
            raise PageIndexAPIError("Failed to get document metadata: Document not found")
        return {key: meta.get(key) for key in
                ("id", "name", "description", "status", "createdAt", "pageNum", "folderId")}

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        if not self._store.delete_document(doc_id):
            raise PageIndexAPIError("Failed to delete document: Document not found.")
        return {"message": "Document deleted successfully."}

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
        if folder_id is not None:
            raise PageIndexAPIError(
                "Failed to list documents: folders are not supported in local mode."
            )
        metas = sorted(self._store.list_metas(), key=lambda m: m.get("id") or "")
        metas.sort(key=lambda m: m.get("createdAt") or "", reverse=True)
        documents = [{
            "id": m.get("id"),
            "name": m.get("name"),
            "description": m.get("description"),
            "status": m.get("status"),
            "createdAt": m.get("createdAt"),
            "pageNum": m.get("pageNum", 0),
            "folderId": None,
            "metadata": None,
            "features": {},
        } for m in metas[offset:offset + limit]]
        return {
            "documents": documents,
            "total": len(metas),
            "limit": limit,
            "offset": offset,
        }

    # ── retrieval (internal: backs chat_completions) ──

    def _require_llm_key(self) -> None:
        """Fail fast with a clear message instead of ten silent retries."""
        from .utils import _is_openai_model
        if _is_openai_model(self._retrieve_model) and not os.getenv("OPENAI_API_KEY"):
            raise PageIndexAPIError(
                f"OPENAI_API_KEY is not set (retrieve model: {self._retrieve_model}). "
                "Local mode uses your own LLM provider key."
            )

    def _tree_search(self, doc_id: str, query: str) -> list[str]:
        """Ask the retrieve model which tree nodes answer the query."""
        from .utils import llm_completion, remove_fields
        self._require_llm_key()
        structure = self._store.get_tree(doc_id) or []
        tree_without_text = remove_fields(copy.deepcopy(structure), fields=["text"])
        prompt = TREE_SEARCH_PROMPT.format(
            query=query, tree=json.dumps(tree_without_text, indent=2, ensure_ascii=False)
        )
        reply = llm_completion(self._retrieve_model, prompt)
        if not reply:
            raise RuntimeError("tree search model returned no output")
        parsed = self._parse_json_reply(reply)
        node_list = parsed.get("node_list") if isinstance(parsed, dict) else None
        if not isinstance(node_list, list):
            raise RuntimeError(f"tree search reply had no node_list: {reply[:200]}")
        known = self._node_map(structure)
        seen = set()
        node_ids = []
        for node_id in node_list:
            node_id = str(node_id)
            if node_id in known and node_id not in seen:
                seen.add(node_id)
                node_ids.append(node_id)
        return node_ids

    @staticmethod
    def _parse_json_reply(reply: str) -> dict:
        from .utils import extract_json
        try:
            return extract_json(reply)
        except Exception as e:
            raise RuntimeError(f"tree search reply was not valid JSON: {reply[:200]}") from e

    @staticmethod
    def _node_map(structure: list) -> dict[str, dict]:
        mapping = {}
        def _walk(nodes):
            for node in nodes:
                if node.get("node_id") is not None:
                    mapping[str(node["node_id"])] = node
                _walk(node.get("nodes") or [])
        _walk(structure)
        return mapping

    # ── chat completions ──

    def chat_completions(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        doc_id: str | list[str] | None = None,
        temperature: float | None = None,
        stream_metadata: bool = False,
        enable_citations: bool = False,
    ) -> dict[str, Any] | Iterator[str] | Iterator[dict[str, Any]]:
        prefix = "Failed to get chat completion"
        if enable_citations:
            raise PageIndexAPIError(f"{prefix}: enable_citations is not supported in local mode.")
        if doc_id is None:
            raise PageIndexAPIError(
                f"{prefix}: doc_id is required in local mode — pass a doc_id or a list of them."
            )
        self._validate_chat_messages(messages, prefix)
        if temperature is not None and not 0.0 <= temperature <= 1.0:
            raise PageIndexAPIError(f"{prefix}: temperature must be between 0.0 and 1.0.")

        doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
        if not doc_ids:
            raise PageIndexAPIError(f"{prefix}: doc_id list cannot be empty.")
        for one_id in doc_ids:
            if self._store.get_meta(one_id) is None:
                raise PageIndexAPIError(f"{prefix}: Document not found or access denied: {one_id}")

        query = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        try:
            context = self._build_chat_context(doc_ids, query)
            llm_messages = (
                [{"role": "system", "content": CHAT_SYSTEM_PROMPT.format(context=context)}]
                + [{"role": m["role"], "content": m["content"]} for m in messages]
            )
            response = self._chat_llm(llm_messages, temperature=temperature, stream=stream)
        except PageIndexAPIError:
            raise
        except Exception as e:
            raise PageIndexAPIError(f"{prefix}: {e}") from e

        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if stream:
            chunks = self._stream_chunks(response, chat_id, created)
            if stream_metadata:
                return chunks
            return (chunk["choices"][0]["delta"].get("content", "")
                    for chunk in chunks
                    if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"))

        choice = response.choices[0]
        result = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": choice.message.content or ""},
                "finish_reason": choice.finish_reason,
            }],
        }
        usage = getattr(response, "usage", None)
        if usage is not None:
            result["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        return result

    @staticmethod
    def _validate_chat_messages(messages: list[dict[str, str]], prefix: str) -> None:
        if not messages:
            raise PageIndexAPIError(
                f"{prefix}: Messages array cannot be empty. Please provide at least one message."
            )
        if messages[0].get("role") != "user":
            raise PageIndexAPIError(f"{prefix}: First message must be from 'user' role.")
        for i, message in enumerate(messages):
            role = message.get("role")
            if role == "system":
                raise PageIndexAPIError(
                    f"{prefix}: System messages are not allowed. "
                    "The system prompt is managed internally."
                )
            if role not in ("user", "assistant"):
                raise PageIndexAPIError(
                    f"{prefix}: Message at index {i} has invalid role {role!r}. "
                    "Valid roles are: user, assistant."
                )
            if not message.get("content"):
                raise PageIndexAPIError(
                    f"{prefix}: Message at index {i} has empty or missing 'content' field."
                )

    def _build_chat_context(self, doc_ids: list[str], query: str) -> str:
        from .utils import count_tokens
        sections = []
        used_tokens = 0
        truncated = False
        for one_id in doc_ids:
            meta = self._store.get_meta(one_id)
            structure = self._store.get_tree(one_id) or []
            node_map = self._node_map(structure)
            for node_id in self._tree_search(one_id, query):
                node = node_map[node_id]
                text = node.get("text", "")
                block = (f"[{meta.get('name')} — {node.get('title', '')} "
                         f"(pages {node.get('start_index')}-{node.get('end_index')})]\n{text}")
                block_tokens = count_tokens(block, model=self._retrieve_model)
                if used_tokens + block_tokens > CHAT_CONTEXT_TOKEN_LIMIT:
                    truncated = True
                    break
                used_tokens += block_tokens
                sections.append(block)
        if truncated:
            print("PageIndex: retrieved context exceeded the token limit; "
                  "some retrieved sections were dropped.")
        if not sections:
            return "(No relevant sections were retrieved for this question.)"
        return "\n\n".join(sections)

    def _chat_llm(self, messages: list[dict], temperature: float | None, stream: bool):
        """One chat call against the retrieve model, OpenAI SDK or LiteLLM."""
        from .utils import _is_openai_model
        self._require_llm_key()
        model = self._retrieve_model
        kwargs: dict[str, Any] = {"messages": messages, "stream": stream}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if _is_openai_model(model):
            import openai
            model = model.removeprefix("openai/")
            if stream:
                kwargs["stream_options"] = {"include_usage": True}
            return openai.OpenAI().chat.completions.create(model=model, **kwargs)
        import litellm
        model = model.removeprefix("litellm/")
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return litellm.completion(model=model, drop_params=True, **kwargs)

    @staticmethod
    def _stream_chunks(response, chat_id: str, created: int) -> Iterator[dict[str, Any]]:
        """Adapt a provider stream into cloud-shaped chat.completion.chunk dicts."""
        def _generate():
            yield {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": ""},
                             "finish_reason": None}],
            }
            finish_reason = None
            usage = None
            for piece in response:
                if getattr(piece, "usage", None) is not None:
                    usage = piece.usage
                if not piece.choices:
                    continue
                choice = piece.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
                content = getattr(choice.delta, "content", None)
                if content:
                    yield {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "choices": [{"index": 0,
                                     "delta": {"content": content},
                                     "finish_reason": None}],
                    }
            final = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": finish_reason or "stop"}],
            }
            if usage is not None:
                final["usage"] = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            yield final
        return _generate()


class _SilentLogger:
    """Drop-in for JsonLogger that writes nothing (the SDK must not create
    a ./logs directory in the caller's working directory)."""

    def log(self, *args, **kwargs):
        pass

    info = error = debug = exception = log


def _format_tree_node(node: dict, node_summary: bool) -> dict:
    """Reshape one stored (local-format) tree node into the cloud tree format.

    Mirrors the server: ``start_index`` is renamed ``page_index``,
    ``end_index`` is dropped, a non-leaf node's ``summary`` is renamed
    ``prefix_summary``, and keys come back in a fixed order.
    """
    children = node.get("nodes") or []
    out = {
        "title": node.get("title", ""),
        "node_id": node.get("node_id"),
        "page_index": node.get("start_index"),
    }
    if node_summary:
        summary = node.get("summary")
        if summary is not None:
            if children:
                out["prefix_summary"] = summary
            else:
                out["summary"] = summary
    if "text" in node:
        out["text"] = node["text"]
    if children:
        out["nodes"] = [_format_tree_node(child, node_summary) for child in children]
    return out
