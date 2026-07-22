# pageindex/index/pipeline.py
from __future__ import annotations
from ..parser.protocol import ContentNode, ParsedDocument


def detect_strategy(nodes: list[ContentNode]) -> str:
    """Determine which indexing strategy to use based on node data."""
    if not nodes:
        # No content at all (e.g. an empty/whitespace-only source file) ->
        # level_based's build_tree_from_levels([]) returns an empty structure
        # immediately with zero LLM calls. content_based's TOC-detection
        # pipeline needs real page content; on an empty page_list it wastes an
        # LLM call and then still raises, for no benefit.
        return "level_based"
    if any(n.level is not None for n in nodes):
        return "level_based"
    return "content_based"


def build_tree_from_levels(nodes: list[ContentNode]) -> list[dict]:
    """Strategy 0: Build tree from explicit level information.
    Adapted from pageindex/page_index_md.py:build_tree_from_nodes."""
    stack = []
    root_nodes = []

    for node in nodes:
        tree_node = {
            "title": node.title or "",
            "text": node.content,
            "line_num": node.index,
            "nodes": [],
        }
        current_level = 1 if node.level is None else node.level

        while stack and stack[-1][1] >= current_level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, _ = stack[-1]
            parent_node["nodes"].append(tree_node)

        stack.append((tree_node, current_level))

    return root_nodes


def _run_async(coro):
    """Run an async coroutine, handling the case where an event loop is already running."""
    import asyncio
    import concurrent.futures
    import contextvars
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # In a running loop: run in a worker thread, with the current context copied
    # so ContextVar-based settings propagate.
    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


def build_index(parsed: ParsedDocument, model: str = None, opt=None) -> dict:
    """Main entry point: ParsedDocument -> tree structure dict.
    Routes to the appropriate strategy and runs enhancement."""
    from .utils import (write_node_id, add_node_text, remove_structure_text,
                        generate_summaries_for_structure, generate_doc_description,
                        create_clean_structure_for_description, format_structure)
    from ..config import IndexConfig, max_concurrency_scope, llm_params_scope

    if opt is None:
        opt = IndexConfig(model=model) if model else IndexConfig()

    # Scope the per-index concurrency cap AND llm kwargs to THIS call only (per
    # thread/async context), so concurrent indexing of other documents isn't
    # affected and a one-off value never sticks as the process default.
    with max_concurrency_scope(getattr(opt, "max_concurrency", None)), \
         llm_params_scope(getattr(opt, "llm_params", None)):
        nodes = parsed.nodes
        strategy = detect_strategy(nodes)

        if strategy == "level_based":
            structure = build_tree_from_levels(nodes)
            # For level-based, text is already in the tree nodes
        else:
            # Strategies 1-3: convert ContentNode list to page_list format for existing pipeline
            page_list = [(n.content, n.tokens) for n in nodes]
            structure = _run_async(_content_based_pipeline(page_list, opt))

        # Unified enhancement
        if opt.if_add_node_id:
            write_node_id(structure)

        if strategy != "level_based":
            if opt.if_add_node_text or opt.if_add_node_summary:
                add_node_text(structure, page_list)

        if opt.if_add_node_summary:
            if strategy == "level_based":
                # Markdown: legacy summarizer — nodes under 200 tokens reuse their text.
                from .page_index_md import generate_summaries_for_structure_md
                _run_async(generate_summaries_for_structure_md(
                    structure, summary_token_threshold=200, model=opt.model))
            else:
                _run_async(generate_summaries_for_structure(structure, model=opt.model))

        result = {
            "doc_name": parsed.doc_name,
            "structure": structure,
        }

        # Description requires summaries (legacy gating): the clean structure
        # fed to the LLM carries only titles/summaries, so without summaries
        # the description would be generated from bare titles.
        if opt.if_add_node_summary and opt.if_add_doc_description:
            clean_structure = create_clean_structure_for_description(structure)
            result["doc_description"] = generate_doc_description(
                clean_structure, model=opt.model
            )

        # Strip 'text' last unless explicitly requested; skip when it was never added.
        text_present = strategy == "level_based" or opt.if_add_node_text or opt.if_add_node_summary
        if text_present and not opt.if_add_node_text:
            remove_structure_text(structure)

        if strategy == "level_based":
            order = ['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes']
        else:
            order = ['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes']
        result["structure"] = format_structure(structure, order=order)

        return result


class _NullLogger:
    """Minimal logger that satisfies the tree_parser interface without writing files."""
    def info(self, message, **kwargs): pass
    def error(self, message, **kwargs): pass
    def debug(self, message, **kwargs): pass


async def _content_based_pipeline(page_list, opt):
    """Strategies 1-3: delegates to the existing PDF pipeline from pageindex/page_index.py.

    The page_list is already in the format expected by tree_parser:
    [(page_text, token_count), ...]
    """
    from .page_index import tree_parser

    logger = _NullLogger()
    structure = await tree_parser(page_list, opt, doc=None, logger=logger)
    return structure
