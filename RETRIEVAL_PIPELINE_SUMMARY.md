# PageIndex Retrieval Pipeline Summary

This document explains how retrieval works after indexing has produced a tree structure. The core idea is that retrieval is tree-first, not vector-first: the agent gets document metadata, the tree structure, and targeted page content, then reasons over those tools to navigate the document.

## Main files involved

- `pageindex/retrieve.py` defines the retrieval tools.
- `pageindex/client.py` exposes those tools through `PageIndexClient`.
- `examples/agentic_vectorless_rag_demo.py` shows how an agent uses the tools.
- `pageindex/__init__.py` re-exports the retrieval helpers for convenience.

## The retrieval surface

There are four retrieval tools:

- `get_document(documents, doc_id)` returns metadata about the indexed document.
- `get_document_structure(documents, doc_id, parts, depth, token_limit)` returns structure with optional depth and deterministic top-level pagination.
- `get_children(documents, doc_id, node_id or node_path, depth)` expands one subtree recursively.
- `get_page_content(documents, doc_id, pages)` returns the actual text for selected pages or line numbers.

The client version in `pageindex/client.py` simply routes those calls to the in-memory document map and, when needed, back to the original PDF or Markdown structure.

## How each retrieval function works

### `get_document`

This is the smallest and cheapest call. It returns JSON with:

- `doc_id`
- `doc_name`
- `doc_description`
- `type`
- `status`
- `page_count` for PDFs or `line_count` for Markdown

The agent uses this first to confirm that the document exists and to learn whether it is dealing with a PDF or Markdown file.

### `get_document_structure`

This returns the hierarchical tree that was built during indexing. The function removes `text` fields before serialization so the agent sees a compact outline rather than a full text dump.

New traversal-oriented arguments:

- `depth`: caps nested levels (for example, `depth=1` returns only top-level nodes).
- `token_limit`: enables deterministic pagination.
- `parts`: selects the 1-based page of top-level nodes under the token budget.
- `has_children`: marks whether a node has nested children available for expansion.

Pagination is deterministic and only splits on top-level node boundaries.

Backward compatibility: if you call it without `depth` and `token_limit`, it still returns the full structure list as before.

### `get_children`

This returns children of a selected node and supports recursive expansion with a `depth` argument.

- Address by `node_id` (preferred) or `node_path` (1-based path such as `2.3`).
- `depth=1` returns immediate children; larger values include deeper descendants.

This is the core building block for subtree-by-subtree traversal in large documents.

For PDFs, this structure is the thing the agent reasons over when deciding which section to open next. For Markdown, the tree includes `line_num` fields and may be traversed recursively by the line-based retrieval logic.

### `get_page_content`

This is the content fetch tool. It accepts page ranges like `5-7`, comma-separated selections like `3,8`, or a single page like `12`.

For PDFs:

- The function first tries the cached page text stored during indexing.
- If cached text is not available, it falls back to reopening the PDF and extracting text directly with PyPDF2.
- Returned content is a JSON array of page/content pairs.

For Markdown:

- The code treats the requested numbers as line numbers.
- It walks the tree recursively and returns nodes whose `line_num` falls inside the requested range.
- This makes Markdown retrieval work like a structural lookup rather than a page lookup.

## How the demo agent uses retrieval

In `examples/agentic_vectorless_rag_demo.py`, the agent follows a narrow retrieval loop:

1. Call `get_document()` to confirm the document is available and learn its size.
2. Use the demo-level `STRUCTURE_MODE` constant to choose one traversal style.
3. In top-level mode, call `get_document_structure(parts=...)` and use `has_children` to decide whether to open a section later.
4. In recursive mode, call `get_document_structure()` once and use `get_children(...)` only for relevant branches.
5. Call `get_page_content()` with a tight page range instead of fetching the whole document.
6. Repeat only as needed.

The prompt tells the agent not to fetch the whole document, which is important because PageIndex is designed around selective, reasoned retrieval rather than dumping the entire structure into context.

## What the agent is really doing

The agent is not using embedding search. It is doing a controlled tree walk:

- It inspects the outline.
- It reasons about which branch is likely relevant.
- It opens a small page window.
- It uses the page text to refine the answer or choose the next branch.

That is why the tool design is so minimal: the retrieval loop depends more on the structure of the document than on a broad search index.

## Why retrieval is cheaper than indexing

Retrieval is cheaper because it is selective.

- It does not need to analyze every page.
- It does not need to regenerate the tree.
- It only fetches content for the pages or line ranges the agent asks for.

Indexing is expensive because it must build the whole tree first. Retrieval assumes that tree already exists.

## Practical limitations

The current retrieval surface returns the whole structure when `get_document_structure()` is called. For very large documents, that can still be too much context for the agent.

The structure is compact compared with raw text, but the full tree can still be large enough to slow reasoning or crowd the context window.

## Suggestions for very large documents

For very large trees, the best improvement is to stop sending the whole structure at once.

- Return only top-level nodes first.
- Add a tool like `get_children(node_id)` so the agent can expand one branch at a time.
- Add a tool like `get_node(node_id)` to fetch a node and its immediate metadata on demand.
- Add a `search_subtree(node_id, query)` tool so the agent can search within a branch instead of the whole document.
- Make the agent recurse: top-level choice first, then child selection, then page fetch.
- Cache expanded branches to avoid repeated tool calls on the same subtree.

That would keep context smaller, reduce tool output, and make the retrieval loop scale better for long reports, legal documents, and technical manuals.
