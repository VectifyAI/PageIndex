"""Agent tools: the cloud MCP tool contract, executed against a PageIndexClient.

Tool names and the surviving input-schema structure match the PageIndex
cloud MCP server — the local surface hides the documented cloud-only
parameters — so agent prompts port across the cloud MCP connection and
this in-process layer. Only the tools that exist in every mode are
registered (no folders, search_documents, or get_document_image), and the
guidance strings (tool descriptions) adapt to the local surface the same
way the agent instructions do — they never teach capabilities that only
exist on the cloud.

Tools never raise for any invocation their signatures accept: every
outcome, including errors, is returned as the same JSON envelope the cloud
emits ({"success": true, ...} / {"error": ...}). Arguments outside a pruned
local signature fail at the Python call boundary; the call_tool path
answers them with the guided error envelope instead.
"""
from __future__ import annotations

import copy
import difflib
import json
import re
import threading
import time
import weakref
from typing import Any, Callable, Optional

from .errors import PageIndexAPIError

TOOL_RESPONSE_CHAR_LIMIT = 100_000
STRUCTURE_FIRST_PAGE_THRESHOLD = 20

_CHAR_BUDGET = int(TOOL_RESPONSE_CHAR_LIMIT * 0.95)
_PAGES_SPEC_RE = re.compile(r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$")
_MAX_REQUESTED_PAGES = 10_000
_SIMILAR_NAMES_LIMIT = 3
_TOOL_WAIT_TIMEOUT = 180.0  # "up to 3 minutes", per the wait_for_completion schema
_TOOL_WAIT_INTERVAL = 5.0

_DOC_NAME_DESCRIPTION = (
    'Copy the `name` field verbatim from a browse_documents() or '
    'search_documents() response (case-sensitive, include extension). '
    'Example: "Q3 Report.pdf". If the response shows two documents with the '
    'same name, pass `folder_id` alongside to disambiguate.'
)
_FOLDER_ID_DISAMBIGUATOR_DESCRIPTION = (
    'Disambiguator for same-name documents. Copy the `folder_id` from the '
    'intended browse/search result; use "root" for root-level documents, or '
    '"shared-with-me"/"following" for the read-only folders at the library '
    'root; omit if `doc_name` is unique. Copy any folder_id verbatim from a '
    'browse_documents()/get_folder_structure() response, never construct one.'
)
_WAIT_FOR_COMPLETION_DESCRIPTION = (
    "If true and document is processing, automatically wait up to 3 minutes "
    "until completed. Reduces repeated tool calls."
)

#: Tool names, descriptions, and parameter schemas, identical to the cloud
#: MCP server's tools/list.
TOOL_CONTRACT: dict[str, dict[str, Any]] = {
    "browse_documents": {
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
        "description": (
            "Primary document retrieval tool. After orienting with "
            "get_folder_structure() (when available), use this for all "
            "document-related questions. The bare call returns root-level "
            "sub-folders and documents; pass folder_id to drill into a "
            'sub-folder level by level. Use sort="relevance" + query for '
            "semantic ranking. Do NOT jump to search_documents() first — it "
            "is an escalation path, only after "
            'browse_documents(sort="relevance") has failed.'
        ),
        "schema": {
            "type": "object",
            "properties": {
                "folder_id": {
                    "type": "string",
                    "default": "root",
                    "description": (
                        'Folder scope (default "root"). Pass a specific folder '
                        'ID to scope into that folder, or "root" to reference '
                        "the library root. The read-only \"shared-with-me\" and "
                        '"following" folders live at the library root — pass '
                        "one of those ids to browse them. Copy any folder_id "
                        "verbatim from a browse/tree response, never construct "
                        "one. Combine with `recursive` to control breadth."
                    ),
                },
                "recursive": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Whether to include documents from descendant folders. "
                        "When false (default), returns the direct contents of "
                        "folder_id along with its sub-folders — prefer this for "
                        "level-by-level exploration so you retain folder "
                        "hierarchy context. When true, flattens all descendant "
                        "documents into one list and omits sub-folders — use "
                        "only when a non-recursive browse of the target folder "
                        "returned no relevant results and you need to widen the "
                        "scope, or the user explicitly requests a flat listing."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["time", "relevance"],
                    "default": "time",
                    "description": (
                        'Sort order. "time" (default) sorts by upload date '
                        '(newest first); "relevance" orders documents by '
                        "semantic relevance to `query`. Relevance also works "
                        "inside the read-only shared folders — pass their "
                        "folder_id — but at the library root it ranks only "
                        "your own documents."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Search query for relevance ranking. Required when "
                        'sort="relevance"; must be omitted when sort="time".'
                    ),
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9007199254740991,
                    "default": 0,
                    "description": (
                        "Zero-based pagination offset. Pass the value of "
                        "`next_offset` from the previous response to fetch the "
                        "next page."
                    ),
                },
                "limit": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": (
                        "Number of documents to return per page (1-50, "
                        "default 10)"
                    ),
                },
            },
            "required": [],
        },
    },
    "get_document": {
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
        "description": (
            "Check a document's processing status and metadata. `status` is "
            'one of "pending", "queued", "processing", "completed", or '
            '"failed" — call this before `get_document_structure()` or '
            "`get_page_content()` to confirm the document is ready."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": _DOC_NAME_DESCRIPTION,
                },
                "folder_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": _FOLDER_ID_DISAMBIGUATOR_DESCRIPTION,
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "default": False,
                    "description": _WAIT_FOR_COMPLETION_DESCRIPTION,
                },
            },
            "required": ["doc_name"],
        },
    },
    "get_document_structure": {
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
        "description": (
            "Extract a document's hierarchical outline (headers, sections, "
            f"page references). REQUIRED for documents over "
            f"{STRUCTURE_FIRST_PAGE_THRESHOLD} pages — call this first to "
            "locate relevant sections, then pass their page numbers to "
            "`get_page_content()`. Use the `part` parameter to iterate large "
            "outlines until `pagination.has_more` is false."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": _DOC_NAME_DESCRIPTION,
                },
                "folder_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": _FOLDER_ID_DISAMBIGUATOR_DESCRIPTION,
                },
                "part": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9007199254740991,
                    "default": 1,
                    "description": (
                        "Part number for pagination (1-based, default 1). For "
                        "large outlines, increment until the response's "
                        "`pagination.has_more` becomes false."
                    ),
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "default": False,
                    "description": _WAIT_FOR_COMPLETION_DESCRIPTION,
                },
            },
            "required": ["doc_name"],
        },
    },
    "get_page_content": {
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
        "description": (
            "Extract page content from a processed document. Use tight, "
            "targeted page ranges — never the whole document at once. For "
            f"documents over {STRUCTURE_FIRST_PAGE_THRESHOLD} pages, call "
            "`get_document_structure()` first to pick relevant sections. "
            "Embedded image paths in the response feed into "
            "`get_document_image()`."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "doc_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": _DOC_NAME_DESCRIPTION,
                },
                "folder_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": _FOLDER_ID_DISAMBIGUATOR_DESCRIPTION,
                },
                "pages": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": r"^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$",
                    "description": (
                        'Page specification: "5", "3,7,10", "5-10", or '
                        '"1-3,7,9-12"'
                    ),
                },
                "wait_for_completion": {
                    "type": "boolean",
                    "default": False,
                    "description": _WAIT_FOR_COMPLETION_DESCRIPTION,
                },
            },
            "required": ["doc_name", "pages"],
        },
    },
    "remove_document": {
        "annotations": {"readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": True, "openWorldHint": False},
        "description": (
            "Permanently delete documents and all associated data. Only invoke "
            "when the user explicitly names the documents AND confirms "
            "deletion. Returns `results` — one entry per requested document: "
            '`{ doc_name, status: "deleted" | "not_found" | "failed", '
            "error? }`. Inspect each entry for per-document failures. This "
            "action is irreversible."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "doc_names": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "maxItems": 10,
                    "description": (
                        "Array of document names to delete. Each name must be "
                        "copied verbatim from the `name` field of a "
                        "browse_documents() or search_documents() response "
                        "(case-sensitive, include extension). Example: "
                        '["Q3 Report.pdf", "draft.pdf"]. Max 10 per call.'
                    ),
                },
                "folder_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": _FOLDER_ID_DISAMBIGUATOR_DESCRIPTION,
                },
            },
            "required": ["doc_names"],
        },
    },
}

_READ_TOOLS = ("browse_documents", "get_document", "get_document_structure",
               "get_page_content")
_MANAGEMENT_TOOLS = ("remove_document",)


# ── response envelopes ──

_ToolResult = tuple[dict, bool]


def _success(data: dict[str, Any], next_steps: dict[str, Any]) -> tuple[dict, bool]:
    return {"success": True, **data, "next_steps": next_steps}, False


def _failure(error: str, details: Optional[dict[str, Any]],
             next_steps: dict[str, Any], error_code: Optional[str] = None,
             ) -> tuple[dict, bool]:
    payload: dict[str, Any] = {"error": error}
    if error_code:
        payload["errorCode"] = error_code
    if details:
        payload.update(details)
    payload["next_steps"] = next_steps
    return payload, True


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ── document listing / name resolution ──

def _all_documents(client) -> list[dict[str, Any]]:
    """Every document the client can list, newest first (both modes list
    newest-first; paging preserves that order)."""
    documents: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = client.list_documents(limit=100, offset=offset)
        batch = page.get("documents") or []
        documents.extend(batch)
        offset += 100
        if not batch or offset >= page.get("total", 0):
            return documents


def _normalize_created_at(value: Any) -> str:
    """Emit the cloud tool format (ISO-8601 UTC with 'Z', millisecond
    precision) from either mode's createdAt string."""
    if not isinstance(value, str) or not value:
        return ""
    try:
        from datetime import datetime, timezone
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except ValueError:
        return value


def _flat_metadata(value: Any) -> Optional[dict[str, Any]]:
    """User-facing string|number|boolean metadata fields only, or None."""
    if not isinstance(value, dict):
        return None
    flat = {key: val for key, val in value.items()
            if isinstance(val, (str, int, float, bool))}
    return flat or None


def _resolve_document(
    client, doc_name: str,
    documents: Optional[list[dict[str, Any]]] = None,
) -> "tuple[Optional[dict[str, Any]], Optional[_ToolResult]]":
    """Resolve doc_name to a list entry. Same-name duplicates resolve to the
    newest match. Returns (entry, None) or (None, error_payload_pair)."""
    if documents is None:
        documents = _all_documents(client)
    matches = [doc for doc in documents if doc.get("name") == doc_name]
    if matches:
        return max(matches, key=lambda d: d.get("createdAt") or ""), None
    names = [str(doc.get("name")) for doc in documents if doc.get("name")]
    similar = difflib.get_close_matches(doc_name, names, n=_SIMILAR_NAMES_LIMIT,
                                        cutoff=0.5)
    message = (
        "Document not found. Did you mean: "
        + ", ".join(f'"{name}"' for name in similar) + "?"
        if similar else "Document not found or you do not have access to it"
    )
    return None, _failure(
        message,
        {"doc_name": doc_name, "similar_files": similar},
        {
            "summary": "The requested document does not exist or is not accessible",
            "options": [
                "Verify the document name is correct",
                "Use browse_documents() to see your recent documents",
                "Check if the document was deleted",
            ],
        },
        "NOT_FOUND",
    )


def _refetch_entry(client, doc_id: str) -> Optional[dict[str, Any]]:
    try:
        return client.get_document(doc_id)
    except PageIndexAPIError:
        return None


def _await_completion(client, entry: dict[str, Any], wait: bool) -> dict[str, Any]:
    """Re-poll a processing document for up to 3 minutes when wait is set."""
    doc_id = entry.get("id")
    if not wait or not doc_id or entry.get("status") in ("completed", "failed"):
        return entry
    deadline = time.monotonic() + _TOOL_WAIT_TIMEOUT
    current = entry
    while time.monotonic() < deadline:
        time.sleep(_TOOL_WAIT_INTERVAL)
        refreshed = _refetch_entry(client, doc_id)
        if refreshed is None:
            return current
        if refreshed.get("metadata") is None:
            # Status refetches omit (or null out) custom metadata; keep the
            # listing's copy.
            refreshed["metadata"] = current.get("metadata")
        current = {**current, **refreshed}
        if current.get("status") in ("completed", "failed"):
            return current
    return current


def _not_ready_error(doc_name: str, status: Any, operation: str,
                     timed_out: bool) -> tuple[dict, bool]:
    if status == "failed":
        return _failure(
            f"Document processing failed. Current status: {status}",
            {"doc_name": doc_name},
            {
                "summary": "Document processing has failed",
                "options": [
                    "Index the document again with submit_document()",
                    "Use browse_documents() to work with other documents",
                ],
            },
            "INVALID_INPUT",
        )
    if timed_out:
        return _failure(
            f"Document is still processing. Current status: {status}",
            {"doc_name": doc_name},
            {
                "summary": "Document processing timeout",
                "options": [
                    "Try again later when processing is complete",
                    "Check status with get_document()",
                ],
            },
            "INVALID_INPUT",
        )
    return _failure(
        f"Document is not ready for {operation}. Current status: {status}",
        {"doc_name": doc_name},
        {
            "summary": "Document is still processing",
            "options": [
                "Wait for document processing to complete",
                "Check status with browse_documents() or get_document()",
            ],
        },
        "INVALID_INPUT",
    )


def _folder_unsupported(param: str) -> tuple[dict, bool]:
    return _failure(
        f"Folders are not supported in local mode yet — omit {param}.",
        None,
        {
            "summary": "This local library does not have folders yet",
            "options": ["Retry the call without a folder_id",
                        "Use browse_documents() to list the library root",
                        "Folders are available on PageIndex cloud "
                        "(PageIndexCloudClient with an API key)"],
        },
        "INVALID_INPUT",
    )


# ── page spec handling ──

def _parse_page_spec(
    pages: str, doc_name: str,
) -> "tuple[Optional[list[int]], Optional[_ToolResult]]":
    """Expand '1-3,7' into a sorted, deduplicated page list, or an error."""
    invalid = _failure(
        "Invalid page specification format",
        {"doc_name": doc_name},
        {
            "summary": "Failed to parse the pages parameter",
            "options": [
                'Use valid formats: "5", "3,7,10", "5-10", or "1-3,7,9-12"',
                "Ensure page numbers are positive integers",
            ],
        },
        "INVALID_INPUT",
    )
    if not isinstance(pages, str) or not _PAGES_SPEC_RE.match(pages.strip()):
        return None, invalid
    expanded: set[int] = set()
    requested_total = 0
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if start > end:
                return None, invalid
        else:
            start = end = int(part)
        # Bound the span arithmetically before materializing it: a spec like
        # "1-1000000000" would otherwise expand to billions of integers
        # inside the caller's process.
        requested_total += end - start + 1
        if requested_total > _MAX_REQUESTED_PAGES:
            return None, _failure(
                f"Too many pages requested (over {_MAX_REQUESTED_PAGES})",
                {"doc_name": doc_name},
                {
                    "summary": "The page specification spans too many pages",
                    "options": [
                        "Request a narrower page range",
                        "The response holds only a few pages per call - page "
                        "through with several smaller requests",
                    ],
                },
                "INVALID_INPUT",
            )
        expanded.update(range(start, end + 1))
    if any(page < 1 for page in expanded):
        return None, _failure(
            "Invalid page numbers. Page numbers must be positive integers",
            {"doc_name": doc_name},
            {
                "summary": "Invalid page numbers provided",
                "options": [
                    "Page numbers must be positive integers (>= 1)",
                    "Check the page specification format",
                ],
            },
            "INVALID_INPUT",
        )
    return sorted(expanded), None


def _format_page_spec(pages: list[int]) -> str:
    """Compress [1,2,3,5] into '1-3,5'."""
    if not pages:
        return ""
    ordered = sorted(set(pages))
    ranges = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


# ── structure formatting / splitting ──

_STRUCTURE_KEY_ORDER = ("title", "node_id", "start_index", "end_index",
                        "page_index", "prefix_summary", "summary", "nodes")


def _format_structure(node: Any) -> Any:
    """Drop node text and normalize key order, recursively."""
    if isinstance(node, list):
        return [_format_structure(item) for item in node]
    if isinstance(node, dict):
        stripped = {key: value for key, value in node.items() if key != "text"}
        if "nodes" in stripped:
            stripped["nodes"] = _format_structure(stripped["nodes"])
        ordered = {key: stripped[key] for key in _STRUCTURE_KEY_ORDER
                   if key in stripped}
        ordered.update({key: value for key, value in stripped.items()
                        if key not in ordered})
        return ordered
    return node


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def _split_structure(structure: Any, budget: int) -> list[Any]:
    """Split a formatted structure into chunks of at most ~budget serialized
    chars. The paginated response shape matches the cloud tool; chunk
    boundaries are implementation-defined."""
    if _serialized_size(structure) <= budget:
        return [structure]
    nodes = structure if isinstance(structure, list) else [structure]
    chunks: list[Any] = []
    group: list[Any] = []
    group_size = 0
    for node in nodes:
        size = _serialized_size(node)
        if size > budget:
            if group:
                chunks.append(group if len(group) > 1 else group[0])
                group, group_size = [], 0
            chunks.extend(_split_oversized_node(node, budget))
            continue
        if group and group_size + size > budget:
            chunks.append(group if len(group) > 1 else group[0])
            group, group_size = [], 0
        group.append(node)
        group_size += size
    if group:
        chunks.append(group if len(group) > 1 else group[0])
    return chunks or [structure]


def _split_oversized_node(node: Any, budget: int) -> list[Any]:
    children = node.get("nodes") if isinstance(node, dict) else None
    if not children:
        return [node]
    shell = {key: value for key, value in node.items() if key != "nodes"}
    shell_size = _serialized_size(shell)
    child_budget = max(budget - shell_size, budget // 2)
    parts = []
    for chunk in _split_structure(children, child_budget):
        parts.append({**shell,
                      "nodes": chunk if isinstance(chunk, list) else [chunk]})
    return parts


# ── tool implementations (client-backed; mode-blind) ──

def _browse_documents(client, folder_id: str = "root", recursive: bool = False,
                      sort: str = "time", query: Optional[str] = None,
                      offset: int = 0, limit: int = 10) -> tuple[dict, bool]:
    if folder_id != "root":
        return _folder_unsupported("folder_id")
    if sort not in ("time", "relevance"):
        return _failure(
            'Invalid sort mode — only the default "time" sort is available '
            "in local mode.", None,
            {"summary": "Invalid sort mode",
             "options": ['Use sort="time" (newest first) or omit sort',
                         "Semantic ranking is available on PageIndex cloud "
                         "(PageIndexCloudClient with an API key)"]},
            "INVALID_INPUT",
        )
    if sort == "relevance" or query:
        # Semantic ranking is a cloud capability; like folders, it is not
        # imitated here.
        return _failure(
            "Relevance ranking is not supported in local mode yet — use "
            "the default time sort.", None,
            {"summary": "This local library does not have semantic ranking yet",
             "options": ["Retry without sort/query and match the returned "
                         "names and descriptions against the intent yourself",
                         "Page through the full library with "
                         "`offset: next_offset`",
                         "Semantic ranking is available on PageIndex cloud "
                         "(PageIndexCloudClient with an API key)"]},
            "INVALID_INPUT",
        )
    try:
        offset = max(int(offset), 0)
        limit = min(max(int(limit), 1), 50)
    except (TypeError, ValueError):
        return _failure("offset and limit must be numbers", None,
                        {"summary": "Invalid pagination parameters",
                         "options": ["Pass integer offset and limit values"]},
                        "INVALID_INPUT")

    listing = client.list_documents(limit=limit, offset=offset)
    window = listing.get("documents") or []
    has_more = offset + limit < listing.get("total", 0)
    next_offset = offset + limit if has_more else None

    page_has_processing = False
    page_has_failed = False
    items = []
    for doc in window:
        status = doc.get("status") or "unknown"
        if status == "failed":
            page_has_failed = True
        elif status != "completed":
            page_has_processing = True
        item = {
            "name": doc.get("name") or "Unknown Document",
            "description": doc.get("description") or "No description provided",
            "status": status,
            "created_at": _normalize_created_at(doc.get("createdAt")),
        }
        metadata = _flat_metadata(doc.get("metadata"))
        if metadata is not None:
            item["metadata"] = metadata
        items.append(item)

    data: dict[str, Any] = {
        "documents": items,
        "sort": sort,
        "next_offset": next_offset,
        "has_more": has_more,
    }
    if not recursive:
        data["folders"] = []

    if not items and offset == 0:
        next_steps = {
            "summary": "Nothing to show",
            "options": ["Nothing here. Index documents with "
                        "PageIndexClient.submit_document() to get started."],
            "auto_retry": "Index a document with submit_document() to get started",
        }
        return _success(data, next_steps)

    options = []
    if items:
        options.append("Use get_document() with a document name to view details")
        options.append(
            "Results returned ≠ correct results. Verify these documents match "
            "the user's actual intent (topic, time period, document type) "
            "before proceeding."
            + (" If they do not match, page through the rest of the library."
               if has_more else "")
            + " Do NOT use general knowledge as a substitute."
        )
    if page_has_processing:
        options.append("Some documents on this page are still processing. "
                       "Use get_document() to check individual status.")
    if page_has_failed:
        options.append("Some documents on this page failed processing. "
                       "Use get_document() to see error details.")
    if has_more:
        options.append("Use browse_documents() with `offset: next_offset` to "
                       "load more documents")
    summary = (f"Showing {len(items)} document(s)"
               + (" (more available)" if has_more else "")
               if items else "Nothing to show")
    return _success(data, {"summary": summary, "options": options})


def _get_document(client, doc_name: str, folder_id: Optional[str] = None,
                  wait_for_completion: bool = False) -> tuple[dict, bool]:
    if folder_id not in (None, "root"):
        return _folder_unsupported("folder_id")
    entry, error = _resolve_document(client, doc_name)
    if error is not None:
        return error
    assert entry is not None
    entry = _await_completion(client, entry, wait_for_completion)

    status = entry.get("status") or "unknown"
    is_processing = status not in ("completed", "failed")
    is_ready = status == "completed"
    page_num = entry.get("pageNum") or 0
    name = entry.get("name") or "Unknown Document"

    suggestions: list[str] = []
    if is_processing:
        suggestions.append("Document is still processing. Processing status "
                           "can be checked later.")
    elif is_ready:
        suggestions.append("Document is ready for analysis.")
        if page_num > 0:
            if page_num <= 5:
                suggestions.extend([
                    f"This is a short document with {page_num} pages.",
                    f'First explore structure: get_document_structure(doc_name: "{name}")',
                    f'Then extract all content: get_page_content(doc_name: "{name}", pages: "1-{page_num}")',
                ])
            elif page_num <= STRUCTURE_FIRST_PAGE_THRESHOLD:
                suggestions.extend([
                    f"This document has {page_num} pages.",
                    f'First explore structure: get_document_structure(doc_name: "{name}")',
                    f'Then extract key pages: get_page_content(doc_name: "{name}", pages: "1,5,10")',
                ])
            else:
                suggestions.extend([
                    f"This is a large document with {page_num} pages.",
                    f'First explore structure: get_document_structure(doc_name: "{name}")',
                    f'Then target specific sections: get_page_content(doc_name: "{name}", pages: "1-3")',
                ])
    else:
        suggestions.append("Document processing failed. Index the document "
                           "again with submit_document().")

    data: dict[str, Any] = {
        "name": name,
        "description": entry.get("description") or "No description provided",
        "status": status,
        "created_at": _normalize_created_at(entry.get("createdAt")),
        "page_count": page_num or None,
        "folder_id": entry.get("folderId"),
    }
    metadata = _flat_metadata(entry.get("metadata"))
    if metadata is not None:
        data["metadata"] = metadata

    return _success(data, {
        "summary": ("Document is ready for analysis and querying." if is_ready
                    else "Document is still being processed." if is_processing
                    else "Document processing has failed."),
        "options": suggestions,
        **({"auto_retry": "Document processing status can be monitored periodically"}
           if is_processing else {}),
    })


def _get_document_structure(client, doc_name: str,
                            folder_id: Optional[str] = None, part: int = 1,
                            wait_for_completion: bool = False) -> tuple[dict, bool]:
    if folder_id not in (None, "root"):
        return _folder_unsupported("folder_id")
    entry, error = _resolve_document(client, doc_name)
    if error is not None:
        return error
    assert entry is not None
    waited = wait_for_completion and entry.get("status") not in ("completed", "failed")
    entry = _await_completion(client, entry, wait_for_completion)
    if entry.get("status") != "completed":
        return _not_ready_error(doc_name, entry.get("status"),
                                "structure retrieval",
                                waited and entry.get("status") != "failed")

    try:
        # Prefer the raw stored tree: its nodes carry start_index/end_index
        # like the cloud structure tool, where client.get_tree() drops
        # end_index and renames fields.
        raw_tree = getattr(getattr(client, "_api", None), "raw_tree", None)
        tree = raw_tree(entry["id"]) if raw_tree is not None else None
        if tree is None:
            tree = client.get_tree(entry["id"], node_summary=True).get("result")
    except PageIndexAPIError as exc:
        return _failure(
            f"Failed to retrieve document structure: {exc}",
            {"doc_name": doc_name},
            {
                "summary": "Failed to retrieve document structure due to an error",
                "options": [
                    "The document may not exist or is not accessible",
                    "Check if the document name is correct",
                    "Try again in a few moments",
                ],
            },
            "INTERNAL_ERROR",
        )
    if tree is None:
        return _failure(
            "Structure not available for this document",
            {"doc_name": doc_name},
            {
                "summary": "Structure not available for this document",
                "options": [
                    "The document may not have been processed correctly or "
                    "structure extraction may have failed",
                    "Try processing the document again if possible",
                ],
            },
            "INTERNAL_ERROR",
        )

    formatted = _format_structure(tree)
    chunks = _split_structure(formatted, _CHAR_BUDGET)
    total_parts = max(1, len(chunks))
    try:
        requested_part = int(part)
    except (TypeError, ValueError):
        requested_part = 1
    current = min(max(requested_part, 1), total_parts)

    if total_parts == 1:
        return _success(
            {"doc_name": doc_name, "structure": chunks[0]},
            {
                "summary": "Document structure retrieved successfully.",
                "options": [
                    "Use get_page_content() to extract specific content from pages",
                ],
            },
        )

    next_steps = (
        {
            "summary": f"Showing part {current} of {total_parts}.",
            "options": [
                f"Request next part with part: {current + 1}",
                f"Jump to last part with part: {total_parts}",
                "Proceed to get_page_content() for specific sections",
            ],
        }
        if current < total_parts else
        {
            "summary": "All parts retrieved for current pagination.",
            "options": [
                "Use get_page_content() to extract specific content from pages",
            ],
        }
    )
    return _success(
        {
            "doc_name": doc_name,
            "total_parts": total_parts,
            "structure": chunks[current - 1],
            "pagination": {
                "part": current,
                "total_parts": total_parts,
                "has_more": current < total_parts,
            },
        },
        next_steps,
    )


def _get_page_content(client, doc_name: str, pages: str,
                      folder_id: Optional[str] = None,
                      wait_for_completion: bool = False) -> tuple[dict, bool]:
    if folder_id not in (None, "root"):
        return _folder_unsupported("folder_id")
    entry, error = _resolve_document(client, doc_name)
    if error is not None:
        return error
    assert entry is not None
    waited = wait_for_completion and entry.get("status") not in ("completed", "failed")
    entry = _await_completion(client, entry, wait_for_completion)
    if entry.get("status") != "completed":
        return _not_ready_error(doc_name, entry.get("status"),
                                "page content retrieval",
                                waited and entry.get("status") != "failed")

    requested, error = _parse_page_spec(pages, doc_name)
    if error is not None:
        return error
    assert requested is not None

    try:
        page_data = client.get_ocr(entry["id"], format="page").get("result") or []
    except PageIndexAPIError as exc:
        return _failure(
            f"Failed to retrieve page content: {exc}",
            {"doc_name": doc_name},
            {
                "summary": "Unable to retrieve page content due to a service issue.",
                "options": [
                    "Verify the document name is correct using browse_documents()",
                    "Check if the document processing is complete with get_document()",
                    "Ensure the requested page numbers are valid",
                ],
                "auto_retry": "This may be a temporary issue - you can try "
                              "the request again",
            },
            "INTERNAL_ERROR",
        )

    by_index = {item["page_index"]: item for item in page_data
                if isinstance(item, dict)
                and isinstance(item.get("page_index"), int)}
    max_page = max(by_index, default=0)

    out_of_range = [page for page in requested if page > max_page]
    valid_pages = [page for page in requested if page <= max_page]
    if out_of_range and not valid_pages:
        return _failure(
            f"All requested pages are out of range. Document has {max_page} "
            f"pages, but you requested pages: {', '.join(map(str, out_of_range))}",
            {
                "doc_name": doc_name,
                "max_pages": max_page,
                "requested_pages": _format_page_spec(out_of_range),
            },
            {
                "summary": "All requested pages are out of range for this document",
                "options": [
                    f"Request pages between 1 and {max_page}",
                    "Use get_document() to check document page count",
                ],
            },
            "INVALID_INPUT",
        )

    content = []
    included: list[int] = []
    remaining: list[int] = []
    budget = _CHAR_BUDGET
    for page in valid_pages:
        item = by_index.get(page)
        markdown = item.get("markdown") if item else None
        text = (markdown if isinstance(markdown, str)
                else f"Page {page} content not available")
        if not included or budget - len(text) >= 0:
            content.append({"page": page, "text": text})
            included.append(page)
            budget -= len(text)
        else:
            remaining.append(page)

    options = [
        "Use get_document_structure() to understand document organization",
        "Request additional pages as needed",
    ]
    if remaining:
        options.insert(0, f"For remaining pages, request: {_format_page_spec(remaining)}")
    if out_of_range:
        options.insert(0, f"Document has {max_page} pages total - request "
                          f"pages 1-{max_page}")
    summary = (
        f"Retrieved {len(included)} pages. Pages "
        f"{', '.join(map(str, out_of_range))} were out of range."
        if out_of_range
        else f"Returned {len(included)} of {len(requested)} requested pages "
             "due to response size limits."
        if remaining
        else f"Successfully retrieved content for {len(content)} "
             f"page{'' if len(content) == 1 else 's'}."
    )
    return _success(
        {
            "doc_name": doc_name,
            "total_pages": max_page,
            "requested_pages": _format_page_spec(requested),
            "returned_pages": _format_page_spec(included),
            "content": content,
        },
        {"summary": summary, "options": options},
    )


def _remove_document(client, doc_names: list[str],
                     folder_id: Optional[str] = None) -> tuple[dict, bool]:
    if folder_id not in (None, "root"):
        return _folder_unsupported("folder_id")
    if not isinstance(doc_names, list) or not doc_names:
        return _failure("At least one document name is required", None,
                        {"summary": "No document names provided",
                         "options": ["Pass doc_names as a non-empty array"]},
                        "INVALID_INPUT")
    if len(doc_names) > 10:
        return _failure("Maximum 10 documents can be deleted at once", None,
                        {"summary": "Too many documents in one call",
                         "options": ["Delete at most 10 documents per call"]},
                        "INVALID_INPUT")
    documents = _all_documents(client)
    results = []
    for doc_name in doc_names:
        entry, error = _resolve_document(client, doc_name, documents=documents)
        if error is not None or entry is None:
            results.append({"doc_name": doc_name, "status": "not_found"})
            continue
        try:
            client.delete_document(entry["id"])
            results.append({"doc_name": doc_name, "status": "deleted"})
        except PageIndexAPIError as exc:
            results.append({"doc_name": doc_name, "status": "failed",
                            "error": str(exc)})
    deleted = sum(1 for item in results if item["status"] == "deleted")
    return _success(
        {"results": results},
        {
            "summary": f"Deleted {deleted} of {len(doc_names)} document(s).",
            "options": ["Use browse_documents() to review the remaining library"],
        },
    )


_IMPLEMENTATIONS: dict[str, Callable[..., tuple[dict, bool]]] = {
    "browse_documents": _browse_documents,
    "get_document": _get_document,
    "get_document_structure": _get_document_structure,
    "get_page_content": _get_page_content,
    "remove_document": _remove_document,
}


def tool_names(include_management: bool = False) -> tuple[str, ...]:
    return _READ_TOOLS + (_MANAGEMENT_TOOLS if include_management else ())


def call_tool(client, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Run one contract tool; returns (envelope_json, is_error). Never raises
    for tool-level failures — unexpected exceptions become error envelopes."""
    implementation = _IMPLEMENTATIONS.get(name)
    if implementation is None:
        payload, _ = _failure(
            f"Unknown tool: {name}",
            {"tool_name": name, "available_tools": list(_IMPLEMENTATIONS)},
            {"summary": "Tool not found",
             "options": [f"Available tools: {', '.join(_IMPLEMENTATIONS)}"]},
            "INVALID_INPUT",
        )
        return json.dumps(payload), True
    try:
        payload, is_error = implementation(client, **arguments)
    except TypeError as exc:
        payload, is_error = _failure(
            f"Invalid arguments for {name}: {exc}", None,
            {"summary": "Invalid tool arguments",
             "options": [f"Check the {name}() parameter names and types"]},
            "INVALID_INPUT",
        )
    except Exception as exc:  # tool calls must never raise into the agent loop
        payload, is_error = _failure(
            f"{name} failed: {exc}", None,
            {"summary": "Unexpected error while running the tool",
             "options": ["Try the request again"],
             "auto_retry": "This is likely a temporary issue - you can try "
                           "the request again"},
            "INTERNAL_ERROR",
        )
    return _dumps(payload), is_error


# ── plain-function materialization (the `client.agent_tools()` surface) ──

def _tool_docstring(description: str, properties: dict[str, Any]) -> str:
    lines = [description, "", "Args:"]
    for param, spec in properties.items():
        lines.append(f"    {param}: {spec.get('description', '')}")
    return "\n".join(lines)


# Local guidance layer: schema STRUCTURE stays byte-identical to the cloud
# contract minus the hidden cloud-only parameters, and description strings
# adapt to the local surface the same way AGENT_INSTRUCTIONS does — guidance
# must not teach capabilities (folders, semantic ranking) or tools
# (search_documents, get_document_image) that do not exist here. Guard
# tests pin structure (contract-minus-hidden equality), tool references
# (the dead-reference test), and capability phrases (the per-docstring
# phrase test) — a contract refresh that reintroduces a cloud-only
# reference fails loudly.

#: Cloud-only parameters hidden from the local surface — strict-schema
#: frameworks make the dead-end calls inexpressible, and lenient framework
#: argument models drop them before the call (degrading to the bare call).
#: The call_tool path still answers folder_id/sort/query with the guided
#: error envelope; recursive is simply accepted (flattening a folderless
#: library is the identity). Plain functions reject unknown parameters at
#: the Python call boundary.
_LOCAL_HIDDEN_PARAMS: dict[str, tuple[str, ...]] = {
    "browse_documents": ("folder_id", "recursive", "sort", "query"),
    "get_document": ("folder_id",),
    "get_document_structure": ("folder_id",),
    "get_page_content": ("folder_id",),
    "remove_document": ("folder_id",),
}

_LOCAL_DOC_NAME_DESCRIPTION = (
    'Copy the `name` field verbatim from a browse_documents() response '
    '(case-sensitive, include extension). Example: "Q3 Report.pdf". '
    "Document names are unique in a local library."
)

_LOCAL_DESCRIPTIONS: dict[str, str] = {
    "browse_documents": (
        "Primary document retrieval tool — first choice for any "
        "document-related question. Lists your documents newest first with "
        "names and descriptions; match them against the user's intent and "
        "page through with `offset: next_offset` (limit up to 50) while "
        "`has_more` is true. "
        'Folder browsing and semantic ranking (sort="relevance") are not '
        "supported in local mode yet — they work on PageIndex cloud."
    ),
    # The image sentence points at a tool that is not registered locally.
    "get_page_content": TOOL_CONTRACT["get_page_content"]["description"]
    .replace(" Embedded image paths in the response feed into "
             "`get_document_image()`.", ""),
}

_LOCAL_PARAM_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("get_document", "doc_name"): _LOCAL_DOC_NAME_DESCRIPTION,
    ("get_document_structure", "doc_name"): _LOCAL_DOC_NAME_DESCRIPTION,
    ("get_page_content", "doc_name"): _LOCAL_DOC_NAME_DESCRIPTION,
    ("remove_document", "doc_names"): (
        "Array of document names to delete. Each name must be copied "
        "verbatim from the `name` field of a browse_documents() response "
        '(case-sensitive, include extension). Example: ["Q3 Report.pdf", '
        '"draft.pdf"]. Max 10 per call.'
    ),
}


def _local_description(name: str) -> str:
    return _LOCAL_DESCRIPTIONS.get(name) or TOOL_CONTRACT[name]["description"]


def _local_schema(name: str) -> dict[str, Any]:
    schema = copy.deepcopy(TOOL_CONTRACT[name]["schema"])
    for param in _LOCAL_HIDDEN_PARAMS.get(name, ()):
        schema["properties"].pop(param, None)
    for (tool_name, param), text in _LOCAL_PARAM_DESCRIPTIONS.items():
        if tool_name == name and param in schema["properties"]:
            schema["properties"][param]["description"] = text
    return schema


def _docstring(name: str) -> str:
    return _tool_docstring(_local_description(name),
                           _local_schema(name)["properties"])


_SCHEMA_TYPE_MAP = {"string": str, "integer": int, "number": float,
                    "boolean": bool, "array": list, "object": dict}


def _annotation_for(spec: dict) -> Any:
    schema_type = spec.get("type")
    if schema_type is None and isinstance(spec.get("anyOf"), list):
        # Nullable unions arrive as anyOf: [{type: string}, {type: null}].
        schema_type = [option.get("type") for option in spec["anyOf"]
                       if isinstance(option, dict) and option.get("type")]
    if isinstance(schema_type, list):
        bases = [t for t in schema_type if t != "null"]
        base = _SCHEMA_TYPE_MAP.get(bases[0], Any) if bases else Any
        return Optional[base] if "null" in schema_type else base
    return _SCHEMA_TYPE_MAP.get(schema_type, Any)


def _make_bridge_function(bridge, meta: dict) -> Callable[..., str]:
    """One plain function for a cloud tool: real signature and docstring from
    the server's schema, invocation proxied over MCP, errors contained."""
    import keyword

    name = str(meta.get("name") or "")
    schema = meta.get("inputSchema") or {}
    properties: dict[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    def _invoke(arguments: dict[str, Any]) -> str:
        # None ≡ omitted, matching the contract's "omit if ..." semantics.
        arguments = {key: value for key, value in arguments.items()
                     if value is not None}
        try:
            return bridge.call_tool(name, arguments)
        except Exception as exc:
            payload, _ = _failure(
                f"{name} failed: {exc}", None,
                {"summary": "Unexpected error while running the tool",
                 "options": ["Try the request again"],
                 "auto_retry": "This is likely a temporary issue - you can "
                               "try the request again"},
                "INTERNAL_ERROR",
            )
            return _dumps(payload)

    params_usable = all(param.isidentifier() and not keyword.iskeyword(param)
                        and param != "_invoke"
                        for param in properties)
    if not params_usable:
        def proxy(**kwargs: Any) -> str:
            return _invoke(kwargs)
    else:
        ordered = ([p for p in properties if p in required]
                   + [p for p in properties if p not in required])
        rendered = ", ".join(
            p if p in required else f"{p}={properties[p].get('default')!r}"
            for p in ordered
        )
        args_literal = "{" + ", ".join(f"'{p}': {p}" for p in ordered) + "}"
        namespace: dict[str, Any] = {"_invoke": _invoke}
        exec(f"def _synthesized({rendered}):\n"
             f"    return _invoke({args_literal})", namespace)
        proxy = namespace["_synthesized"]
        annotations: dict[str, Any] = {}
        for p in ordered:
            annotation = _annotation_for(properties[p])
            if p not in required and "default" not in properties[p]:
                # Absent-but-non-nullable params must admit None, or strict
                # schemas force the model to always send a value.
                annotation = Optional[annotation]
            annotations[p] = annotation
        annotations["return"] = str
        proxy.__annotations__ = annotations
    proxy.__name__ = proxy.__qualname__ = name or "tool"
    proxy.__doc__ = _tool_docstring(meta.get("description", ""), properties)
    return proxy


_BRIDGES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_BRIDGES_LOCK = threading.Lock()


def _cloud_bridge(client):
    """One bridge per client: tool discovery and instructions share a single
    MCP session. Weak-keyed off the instance so clients stay picklable; the
    lock closes the check-then-set race under concurrent first calls."""
    with _BRIDGES_LOCK:
        bridge = _BRIDGES.get(client)
        if bridge is None:
            from .mcp_bridge import McpBridge
            bridge = McpBridge(
                f"{client.BASE_URL}/mcp",
                {"Authorization": f"Bearer {client.api_key}"},
            )
            _BRIDGES[client] = bridge
        return bridge


def _build_cloud_agent_tools(client, include_management: bool) -> list[Callable[..., str]]:
    bridge = _cloud_bridge(client)
    tools_meta = bridge.list_tools()
    if not include_management:
        # Plain functions have no framework permission layer, so the
        # management gate lives here: only tools the server marks read-only.
        filtered = [meta for meta in tools_meta
                    if (meta.get("annotations") or {}).get("readOnlyHint") is True]
        if tools_meta and not filtered:
            raise PageIndexAPIError(
                "The MCP server returned tools but none are annotated "
                "read-only — a server annotation regression would otherwise "
                "silently disable every tool. Pass include_management=True "
                "to expose the unfiltered list."
            )
        tools_meta = filtered
    return [_make_bridge_function(bridge, meta) for meta in tools_meta]


def build_agent_tools(client, include_management: bool = False) -> list[Callable[..., str]]:
    """Plain synchronous functions bound to `client`.

    Cloud: one function per tool of the live cloud MCP tool set, signatures
    synthesized from the server's schemas, calls proxied over MCP. Local:
    the built-in contract tools over the local store. Every function returns
    the JSON envelope as a string and never raises for arguments its
    signature accepts (cloud-only parameters are absent from the local
    signatures; the call_tool path answers them with the guided envelope).
    """
    if getattr(client, "api_key", None):
        return _build_cloud_agent_tools(client, include_management)

    def browse_documents(offset: int = 0, limit: int = 10) -> str:
        return call_tool(client, "browse_documents", {
            "offset": offset, "limit": limit,
        })[0]

    def get_document(doc_name: str, wait_for_completion: bool = False) -> str:
        return call_tool(client, "get_document", {
            "doc_name": doc_name,
            "wait_for_completion": wait_for_completion,
        })[0]

    def get_document_structure(doc_name: str, part: int = 1,
                               wait_for_completion: bool = False) -> str:
        return call_tool(client, "get_document_structure", {
            "doc_name": doc_name, "part": part,
            "wait_for_completion": wait_for_completion,
        })[0]

    def get_page_content(doc_name: str, pages: str,
                         wait_for_completion: bool = False) -> str:
        return call_tool(client, "get_page_content", {
            "doc_name": doc_name, "pages": pages,
            "wait_for_completion": wait_for_completion,
        })[0]

    def remove_document(doc_names: list[str]) -> str:
        return call_tool(client, "remove_document", {
            "doc_names": doc_names,
        })[0]

    functions = {
        "browse_documents": browse_documents,
        "get_document": get_document,
        "get_document_structure": get_document_structure,
        "get_page_content": get_page_content,
        "remove_document": remove_document,
    }
    tools = []
    for name in tool_names(include_management):
        function = functions[name]
        function.__doc__ = _docstring(name)
        tools.append(function)
    return tools


# ── agent instructions ──
# Local subset of the cloud MCP server's initialize instructions (its
# no-folders variant), trimmed to what exists here: the search_documents
# escalation steps, get_document_image, and the shared read-only-folders
# block are removed, and the sort="relevance" guidance is replaced with
# name/description matching (semantic ranking is cloud-side). Cloud
# clients receive the server's live instructions instead — see
# _base_instructions().

_INSTRUCTIONS_HEADER = (
    "PageIndex by Vectify AI is a document platform for uploading and "
    "managing long PDFs (research papers, financial reports, legal docs, "
    "textbooks, etc.)."
)

_READING_WORKFLOW = f"""\
READING WORKFLOW:
- For documents over {STRUCTURE_FIRST_PAGE_THRESHOLD} pages: call get_document_structure() first to locate relevant sections, then get_page_content() with targeted page ranges.
- For small documents ({STRUCTURE_FIRST_PAGE_THRESHOLD} pages or fewer): call get_page_content() directly."""

_TOOL_USAGE_RULES = """\
TOOL USAGE RULES:
- Invoke a tool only when all required parameters are present or clearly inferable. Never invent placeholder values.
- If a tool returns an error, present the provided next_steps/options to the user instead of retrying blindly."""

_DISCOVERY = """\
DOCUMENT DISCOVERY:
- browse_documents() — DEFAULT discovery tool, first choice for any document-related question. It lists your documents newest first with names and descriptions; match them against the user's intent, and page through with `offset: next_offset` while has_more is true."""

_DECISION = """\
DECISION:
- "What do I have / list / recent" → browse_documents()
- ANY question that needs a document to answer (including "find THE paper about Y") → browse_documents(), then pick the documents whose name/description matches the question"""

_AFTER_DISCOVERY = """\
- Skip discovery ONLY for questions with NO possible document connection (e.g., "capital of France").
- After discovery: 1 match or 1 clearly best match → proceed to read and answer without asking. Multiple equally relevant → ask user to pick.
- Results returned ≠ correct results. If the returned documents do not clearly match the user's intent (e.g., wrong topic, wrong time period, wrong document type), treat it the same as "not found" and continue the PERSISTENCE protocol below."""

_PERSISTENCE = """\
PERSISTENCE (before concluding the target document is not in the library):
This protocol applies both when results are empty AND when results are returned but none match the user's intent. Do NOT give up after a single discovery attempt. Follow these steps in order:
1. browse_documents() and compare every returned name/description against the user's intent
2. Page through the ENTIRE library with `limit: 50` and `offset: next_offset` until has_more is false — MANDATORY, must be completed before concluding "not found"
3. Re-scan for loose matches: synonyms, abbreviations, and partial titles in names/descriptions can identify the target
Only after ALL three steps have been tried may you conclude the document is not in the library. Do NOT fall back to general knowledge — if the user's question references their own documents, exhaust every discovery path first."""

AGENT_INSTRUCTIONS = "\n\n".join([
    _INSTRUCTIONS_HEADER,
    _READING_WORKFLOW,
    _TOOL_USAGE_RULES,
    _DISCOVERY,
    _DECISION,
    _AFTER_DISCOVERY,
    _PERSISTENCE,
])


def _base_instructions(client) -> str:
    """Cloud: the live instructions the MCP server serves for this key's
    tool set. Local: the built-in subset instructions."""
    if not getattr(client, "api_key", None):
        return AGENT_INSTRUCTIONS
    instructions = _cloud_bridge(client).instructions()
    if not isinstance(instructions, str) or not instructions.strip():
        raise PageIndexAPIError(
            "The MCP server returned no agent instructions — refusing to "
            "substitute the SDK's local-subset guidance, which does not "
            "cover the cloud tool set."
        )
    return instructions


def doc_targeting_block(client, doc_id) -> Optional[str]:
    """The doc_id targeting text: names, metadata, and the directive to work
    within those documents. Shared by agent_instructions and the local chat
    surfaces (which place it as a leading conversation item). Raises when a
    doc_id's name is shadowed by a newer same-name document — the
    name-addressed tools could not reach it."""
    if doc_id is None:
        return None
    doc_ids = [doc_id] if isinstance(doc_id, str) else list(doc_id)
    if not doc_ids:
        return None
    details = [client.get_document(one_id) for one_id in doc_ids]
    documents = _all_documents(client)
    for one_id, detail in zip(doc_ids, details):
        entry, _ = _resolve_document(client, str(detail.get("name")),
                                     documents=documents)
        if entry is not None and entry.get("id") != one_id:
            raise PageIndexAPIError(
                f'Document "{detail.get("name")}" (doc_id: {one_id}) is '
                "shadowed by a newer document with the same name (doc_id: "
                f'{entry.get("id")}). The tools address documents by name '
                "and would read the newer one. Rename or remove the "
                "duplicate, or pass the newer doc_id."
            )
    context = json.dumps(details, ensure_ascii=False)
    if len(details) == 1:
        return (
            f"The user has specified document: {details[0].get('name')}\n"
            f"Document metadata: {context}\n"
            "Use this document's name to retrieve its content with "
            "get_document_structure() and get_page_content()."
        )
    names = ", ".join(str(item.get("name")) for item in details)
    return (
        f"The user has specified documents: {names}\n"
        f"Documents metadata: {context}\n"
        "Use these documents' names to retrieve their content with "
        "get_document_structure() and get_page_content()."
    )


def build_agent_instructions(client, doc_id=None) -> str:
    """Orchestration guidance for document QA agents; with doc_id, appends
    the target documents and directs the agent to work within them."""
    base = _base_instructions(client)
    block = doc_targeting_block(client, doc_id)
    return base if block is None else base + "\n\n" + block
