import json
import PyPDF2

try:
    from .utils import count_tokens, get_number_of_pages, remove_fields
except ImportError:
    from utils import count_tokens, get_number_of_pages, remove_fields


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            start, end = int(part.split("-", 1)[0].strip()), int(
                part.split("-", 1)[1].strip()
            )
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get("page_count"):
        return doc_info["page_count"]
    if doc_info.get("pages"):
        return len(doc_info["pages"])
    return get_number_of_pages(doc_info["path"])


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get("pages")
    if cached_pages:
        page_map = {p["page"]: p["content"] for p in cached_pages}
        return [{"page": p, "content": page_map[p]} for p in page_nums if p in page_map]
    path = doc_info["path"]
    with open(path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {"page": p, "content": pdf_reader.pages[p - 1].extract_text() or ""}
            for p in valid_pages
        ]


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get("line_num")
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({"page": ln, "content": node.get("text", "")})
            if node.get("nodes"):
                _traverse(node["nodes"])

    _traverse(doc_info.get("structure", []))
    results.sort(key=lambda x: x["page"])
    return results


def _estimate_tokens(data: object, token_model: str | None = None) -> int:
    """Estimate token usage for serialized JSON data in a deterministic way."""
    payload = json.dumps(data, ensure_ascii=False)
    try:
        return count_tokens(payload, model=token_model)
    except Exception:
        # Fallback keeps behavior deterministic even if token counting fails.
        return max(1, len(payload) // 4)


def _apply_depth(
    nodes: list[dict], depth: int | None, current_depth: int = 1
) -> list[dict]:
    """Return a copy of nodes truncated to the requested depth."""
    if depth is None:
        return [_add_has_children(dict(node)) for node in nodes]
    if depth < 1:
        return []

    out = []
    for node in nodes:
        node_copy = {k: v for k, v in node.items() if k != "nodes"}
        children = node.get("nodes") or []
        node_copy["has_children"] = bool(children)
        if current_depth < depth and children:
            node_copy["nodes"] = _apply_depth(children, depth, current_depth + 1)
        out.append(node_copy)
    return out


def _add_has_children(node: dict) -> dict:
    """Annotate a node with a boolean children marker and recurse."""
    children = node.get("nodes") or []
    node["has_children"] = bool(children)
    if children:
        node["nodes"] = [_add_has_children(dict(child)) for child in children]
    return node


def _paginate_top_level_nodes(
    top_level_nodes: list[dict], token_limit: int, token_model: str | None = None
) -> list[list[dict]]:
    """Split structure deterministically by top-level nodes under a token budget."""
    if token_limit <= 0:
        raise ValueError("token_limit must be > 0")

    pages: list[list[dict]] = []
    current_page: list[dict] = []

    for node in top_level_nodes:
        candidate = current_page + [node]
        if current_page and _estimate_tokens(candidate, token_model) > token_limit:
            pages.append(current_page)
            current_page = [node]
            continue

        # Always make progress even if a single top-level node exceeds the limit.
        if not current_page and _estimate_tokens([node], token_model) > token_limit:
            pages.append([node])
            continue

        current_page = candidate

    if current_page:
        pages.append(current_page)

    return pages


def _find_node_by_id(nodes: list[dict], node_id: str) -> dict | None:
    for node in nodes:
        if str(node.get("node_id")) == str(node_id):
            return node
        children = node.get("nodes") or []
        found = _find_node_by_id(children, node_id)
        if found:
            return found
    return None


def _find_node_by_path(nodes: list[dict], node_path: str) -> dict | None:
    """Find a node by 1-based index path like '2.3.1' (top-level.child.grandchild)."""
    try:
        parts = [int(p) for p in node_path.split(".") if p.strip()]
    except ValueError:
        return None
    if not parts:
        return None

    current_list = nodes
    current_node = None
    for index in parts:
        if index < 1 or index > len(current_list):
            return None
        current_node = current_list[index - 1]
        current_list = current_node.get("nodes") or []
    return current_node


# ── Tool functions ────────────────────────────────────────────────────────────


def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    result = {
        "doc_id": doc_id,
        "doc_name": doc_info.get("doc_name", ""),
        "doc_description": doc_info.get("doc_description", ""),
        "type": doc_info.get("type", ""),
        "status": "completed",
    }
    if doc_info.get("type") == "pdf":
        result["page_count"] = _count_pages(doc_info)
    else:
        result["line_count"] = doc_info.get("line_count", 0)
    return json.dumps(result)


def get_document_structure(
    documents: dict,
    doc_id: str,
    parts: int = 1,
    depth: int | None = None,
    token_limit: int | None = None,
    token_model: str | None = None,
) -> str:
    """
    Return document structure JSON with text fields removed.

    Backward-compatible mode:
    - If token_limit is not provided and depth is None, returns the full structure list.

    Extended mode:
    - depth: limit nested levels (depth=1 => only top-level nodes).
    - token_limit + parts: deterministic pagination by complete top-level nodes.
      The returned page is selected via 1-based `parts`.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})

    structure = doc_info.get("structure", [])
    structure_no_text = [
        _add_has_children(node) for node in remove_fields(structure, fields=["text"])
    ]

    # Preserve original response shape for existing callers.
    if token_limit is None and depth is None:
        return json.dumps(structure_no_text, ensure_ascii=False)

    if depth is not None:
        structure_no_text = _apply_depth(structure_no_text, depth)

    if token_limit is None:
        return json.dumps(structure_no_text, ensure_ascii=False)

    try:
        requested_part = int(parts)
    except Exception:
        requested_part = 1
    requested_part = max(1, requested_part)

    page_list = _paginate_top_level_nodes(
        structure_no_text, token_limit=token_limit, token_model=token_model
    )
    total_parts = max(1, len(page_list))
    selected_part = min(requested_part, total_parts)
    selected_nodes = page_list[selected_part - 1] if page_list else []

    # Build next_steps guidance for the agent.
    options = []
    if selected_part < total_parts:
        next_part = selected_part + 1
        options.append(f"Request next part with part: {next_part}")
    if selected_part < total_parts and selected_part != total_parts:
        options.append(f"Jump to last part with part: {total_parts}")
    options.append("Proceed to get_page_content() for specific sections")
    if not options:
        options.append(
            "All parts reviewed; use get_page_content() for content retrieval"
        )

    pagination_summary = f"Showing part {selected_part} of {total_parts}."
    if selected_part == total_parts:
        pagination_summary += " (Last part)"

    next_steps = {
        "options": options,
        "summary": pagination_summary,
    }

    result = {
        "structure": selected_nodes,
        "pagination": {
            "part": selected_part,
            "has_more": selected_part < total_parts,
            "total_parts": total_parts,
        },
        "next_steps": next_steps,
        "total_parts": total_parts,
    }
    return json.dumps(result, ensure_ascii=False)


def get_children(
    documents: dict,
    doc_id: str,
    node_id: str = "",
    node_path: str = "",
    depth: int = 1,
) -> str:
    """
    Return children of a specific node.

    Node can be addressed by:
    - node_id: stable id from indexing output (preferred when available)
    - node_path: 1-based index path such as "2.3" (top-level 2 -> child 3)

    depth controls recursive expansion of descendants from the child level.
    depth=1 returns immediate children only.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})

    structure = remove_fields(doc_info.get("structure", []), fields=["text"])
    target_node = None

    if node_id:
        target_node = _find_node_by_id(structure, node_id)
    elif node_path:
        target_node = _find_node_by_path(structure, node_path)
    else:
        return json.dumps({"error": "Provide node_id or node_path"})

    if not target_node:
        return json.dumps(
            {
                "error": "Target node not found",
                "node_id": node_id,
                "node_path": node_path,
            }
        )

    children = target_node.get("nodes") or []
    clipped_depth = max(1, int(depth)) if isinstance(depth, int) else 1
    expanded_children = _apply_depth(children, clipped_depth)

    # Build next_steps guidance for the agent.
    options = []
    node_start = target_node.get("start_index")
    node_end = target_node.get("end_index")

    if not children:
        options.append(
            f'Use get_page_content(pages="{node_start}-{node_end}") to fetch this section\'s content'
        )
        next_steps_summary = "This is a leaf node with no children."
    else:
        # Find children that have further descendants to suggest expansion.
        expandable = [child for child in expanded_children if child.get("has_children")]
        if expandable:
            child_examples = [
                f"Expand child '{child.get('title', 'Untitled')}' (ID: {child.get('node_id')})"
                for child in expandable[:2]
            ]
            options.extend(child_examples)

        # Always suggest fetching page content for the first few children.
        first_child_range = None
        if expanded_children:
            start = expanded_children[0].get("start_index")
            # Find the end of the last child at this level
            end = expanded_children[-1].get("end_index")
            if start and end:
                first_child_range = f"{start}-{end}"
                options.append(
                    f'Use get_page_content(pages="{first_child_range}") to preview all children content'
                )

        next_steps_summary = f"Showing {len(expanded_children)} children. "
        if expandable:
            next_steps_summary += (
                f"{len(expandable)} have further structure to explore."
            )
        else:
            next_steps_summary += "All are leaf nodes."

    next_steps = {
        "options": options,
        "summary": next_steps_summary,
    }

    result = {
        "node": {
            "node_id": target_node.get("node_id"),
            "title": target_node.get("title", ""),
            "start_index": target_node.get("start_index"),
            "end_index": target_node.get("end_index"),
            "has_children": bool(children),
        },
        "children": expanded_children,
        "child_count": len(children),
        "depth": clipped_depth,
        "next_steps": next_steps,
    }
    return json.dumps(result, ensure_ascii=False)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    For PDF: pages are physical page numbers (1-indexed).
    For Markdown: pages are line numbers corresponding to node headers.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps(
            {
                "error": f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'
            }
        )

    try:
        if doc_info.get("type") == "pdf":
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({"error": f"Failed to read page content: {e}"})

    return json.dumps(content, ensure_ascii=False)
