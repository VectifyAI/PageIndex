import asyncio
import json
import logging
import os
import re

_PAGE_TAG_RE = re.compile(r"(<page_number=\d+>.*?</page_number=\d+>)", re.DOTALL)

from pageindex.page_index import (
    add_preface_if_needed,
    check_title_appearance_in_start_concurrent,
    meta_processor,
    process_large_node_recursively,
)
from pageindex.utils import (
    ConfigLoader,
    add_node_text,
    agenerate_doc_description,
    count_tokens,
    create_clean_structure_for_description,
    format_structure,
    generate_summaries_for_structure,
    post_processing,
    remove_structure_text,
    write_node_id,
)

logger = logging.getLogger(__name__)
# ── S3 helpers ────────────────────────────────────────────────────────────────

async def read_markdown_from_s3(key: str, s3_session, bucket: str) -> str:
    async with s3_session.client("s3") as s3:
        resp = await s3.get_object(Bucket=bucket, Key=key)
        return (await resp["Body"].read()).decode("utf-8")


async def upload_tree_to_s3(key: str, data: dict, s3_session, bucket: str) -> None:
    async with s3_session.client("s3") as s3:
        await s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )


# ── Markdown → page_list ──────────────────────────────────────────────────────

def markdown_to_page_list(content: str, tokens_per_page: int, model: str) -> list[tuple[str, int]]:
    """Split markdown text into virtual pages for the pipeline.

    When the content contains <page_number=N>...</page_number=N> tags (produced
    by the OCR-to-markdown converter), each tag block becomes exactly one virtual
    page. This keeps physical_index values accurate to real document page numbers.

    Falls back to token-budget chunking for plain markdown without page tags.
    tokens_per_page is unused in the tagged path.
    """
    tag_matches = _PAGE_TAG_RE.findall(content)
    if tag_matches:
        return [(page, count_tokens(page, model)) for page in tag_matches if page.strip()]

    lines = content.split("\n")
    pages: list[tuple[str, int]] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line, model) if line.strip() else 1
        if current_tokens + line_tokens > tokens_per_page and current_lines:
            chunk = "\n".join(current_lines)
            pages.append((chunk, count_tokens(chunk, model)))
            current_lines = []
            current_tokens = 0
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunk = "\n".join(current_lines)
        pages.append((chunk, count_tokens(chunk, model)))

    return pages


async def _build_tree(page_list: list[tuple[str, int]], opt) -> tuple[list, str]:
    """Run the full PDF pipeline on virtual markdown pages.

    Calls meta_processor(mode='process_no_toc') directly — skipping check_toc()
    — then follows the same post-processing steps as page_index_main():
    preface, boundary detection, tree build, large-node recursion, optional
    summaries/description. Wrapped in asyncio.run() so it can be called from
    asyncio.to_thread() without touching the server's event loop.
    """
    logger.info({"total_page_number": len(page_list)})
    logger.info({"total_token": sum(t for _, t in page_list)})

    # Directly enter process_no_toc — no TOC scanning for markdown
    flat_toc = await meta_processor(
        page_list, mode="process_no_toc", start_index=1, opt=opt, logger=logger
    )

    flat_toc = add_preface_if_needed(flat_toc)
    flat_toc = await check_title_appearance_in_start_concurrent(
        flat_toc, page_list, model=opt.model, logger=logger
    )
    valid_items = [item for item in flat_toc if item.get("physical_index") is not None]

    tree = post_processing(valid_items, len(page_list))
    await asyncio.gather(
        *[process_large_node_recursively(node, page_list, opt, logger=logger) for node in tree]
    )

    if opt.if_add_node_id == "yes":
        write_node_id(tree)
    if opt.if_add_node_text == "yes" or opt.if_add_node_summary == "yes":
        add_node_text(tree, page_list)
    description = ""
    if opt.if_add_node_summary == "yes":
        await generate_summaries_for_structure(tree, model=opt.model)
        if opt.if_add_node_text == "no":
            remove_structure_text(tree)
        # Matches original page_index_builder(): description is only generated
        # when summaries are also enabled (nested inside the summary block).
        if opt.if_add_doc_description == "yes":
            clean = create_clean_structure_for_description(tree)
            description = await agenerate_doc_description(clean, model=opt.model)

    tree = format_structure(
        tree,
        order=["title", "node_id", "start_index", "end_index", "summary", "text", "nodes"],
    )
    return tree, description


# ── Public orchestrator ───────────────────────────────────────────────────────

def _build_config_overrides(payload) -> dict:
    """Build config overrides with three-tier priority.

    Priority (highest → lowest):
      1. Request fields — explicit per-call values
      2. PAGEINDEX_* env vars — deploy-time defaults
      3. pageindex/config.yaml — library defaults (handled by ConfigLoader)
    """
    _ENV_MAP = {
        "model": "PAGEINDEX_MODEL",
        "if_add_node_id": "PAGEINDEX_IF_ADD_NODE_ID",
        "if_add_node_summary": "PAGEINDEX_IF_ADD_NODE_SUMMARY",
        "if_add_node_text": "PAGEINDEX_IF_ADD_NODE_TEXT",
        "if_add_doc_description": "PAGEINDEX_IF_ADD_DOC_DESCRIPTION",
    }
    # Start with env vars as base layer
    overrides = {key: os.environ[env] for key, env in _ENV_MAP.items() if env in os.environ}

    # Overlay request fields (non-None values take precedence over env vars)
    request_fields = {
        "model": payload.model,
        "if_add_node_id": payload.if_add_node_id,
        "if_add_node_summary": payload.if_add_node_summary,
        "if_add_node_text": payload.if_add_node_text,
        "if_add_doc_description": payload.if_add_doc_description,
        **(payload.extra_config or {}),
    }
    overrides.update({k: v for k, v in request_fields.items() if v is not None})

    return overrides


async def process_markdown(payload, s3_session, bucket: str) -> dict:
    content = await read_markdown_from_s3(payload.input_s3_key, s3_session, bucket)

    opt = ConfigLoader().load(_build_config_overrides(payload))
    page_list = markdown_to_page_list(content, payload.tokens_per_page, opt.model)

    tree, description = await _build_tree(page_list, opt)

    output = {
        "doc_description": description,
        "structure": tree,
    }
    await upload_tree_to_s3(payload.output_s3_key, output, s3_session, bucket)

    return {**output, "output_s3_key": payload.output_s3_key}
