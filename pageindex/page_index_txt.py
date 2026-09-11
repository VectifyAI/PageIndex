"""Plain text (.txt) document indexing for PageIndex."""
import asyncio
import json
import os
import re

try:
    from .utils import (
        count_tokens,
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        structure_to_list,
        write_node_id,
    )
    from .page_index_md import (
        build_tree_from_nodes,
        generate_summaries_for_structure_md,
        tree_thinning_for_index,
        update_node_list_with_text_token_count,
    )
except ImportError:
    from utils import (
        count_tokens,
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        structure_to_list,
        write_node_id,
    )
    from page_index_md import (
        build_tree_from_nodes,
        generate_summaries_for_structure_md,
        tree_thinning_for_index,
        update_node_list_with_text_token_count,
    )


# Heading detection regex patterns for plain text documents
_CHAPTER_SECTION_RE = re.compile(
    r"^(?:chapter|section|part|appendix)\s+([0-9a-zA-ZIVXLCDM]+)(?:\s*[:.-]\s*(.*))?$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z0-9].*)$")
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s,':;.-]{2,60}$")


def extract_nodes_from_txt(txt_content: str):
    """Extract structural nodes from plain text content.

    Detects:
    1. Markdown-style headers (# Header) if present
    2. Chapter/Section/Part/Appendix labels (e.g. 'Chapter 1: Intro')
    3. Numbered section headers (e.g. '1. Introduction', '1.1 Overview')
    4. Underlined headings ('Title' followed by '---' or '===')
    5. ALL-CAPS standalone headings

    If no structured headings are detected, falls back to paragraph-level nodes.
    """
    lines = txt_content.split("\n")
    node_list = []
    total_lines = len(lines)

    i = 0
    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        line_num = i + 1

        # Check Markdown-style headers
        md_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if md_match:
            level = len(md_match.group(1))
            title = md_match.group(2).strip()
            node_list.append({"node_title": title, "line_num": line_num, "level": level})
            i += 1
            continue

        # Check underlined headings (line followed by ===== or -----)
        if i + 1 < total_lines:
            next_stripped = lines[i + 1].strip()
            if next_stripped and len(next_stripped) >= 3:
                if set(next_stripped) == {"="}:
                    node_list.append({"node_title": stripped, "line_num": line_num, "level": 1})
                    i += 2
                    continue
                elif set(next_stripped) == {"-"}:
                    node_list.append({"node_title": stripped, "line_num": line_num, "level": 2})
                    i += 2
                    continue

        # Check Chapter / Section / Part headings
        cs_match = _CHAPTER_SECTION_RE.match(stripped)
        if cs_match:
            title = stripped
            node_list.append({"node_title": title, "line_num": line_num, "level": 1})
            i += 1
            continue

        # Check Numbered headings (e.g. 1.2 Title)
        num_match = _NUMBERED_HEADING_RE.match(stripped)
        if num_match:
            num_prefix = num_match.group(1)
            title = stripped
            level = num_prefix.count(".") + 1
            node_list.append({"node_title": title, "line_num": line_num, "level": min(level, 6)})
            i += 1
            continue

        # Check standalone ALL-CAPS heading (preceded and followed by blank lines or start/end)
        prev_blank = (i == 0) or (not lines[i - 1].strip())
        next_blank = (i + 1 >= total_lines) or (not lines[i + 1].strip())
        if prev_blank and next_blank and _ALL_CAPS_HEADING_RE.match(stripped) and len(stripped.split()) <= 8:
            node_list.append({"node_title": stripped, "line_num": line_num, "level": 1})
            i += 1
            continue

        i += 1

    # Fallback: if no headings detected, split by paragraph blocks
    if not node_list:
        p_index = 1
        for line_idx, line in enumerate(lines):
            stripped = line.strip()
            prev_empty = (line_idx == 0) or (not lines[line_idx - 1].strip())
            if stripped and prev_empty:
                # Use first sentence or up to 50 chars as title
                first_sentence = stripped.split(".")[0].strip()
                if len(first_sentence) > 50:
                    first_sentence = first_sentence[:47] + "..."
                title = f"Section {p_index}: {first_sentence}" if first_sentence else f"Section {p_index}"
                node_list.append({
                    "node_title": title,
                    "line_num": line_idx + 1,
                    "level": 1,
                })
                p_index += 1

    return node_list, lines


def extract_node_text_content(node_list, txt_lines):
    all_nodes = []
    for node in node_list:
        processed_node = {
            "title": node["node_title"],
            "line_num": node["line_num"],
            "level": node["level"],
        }
        all_nodes.append(processed_node)

    for i, node in enumerate(all_nodes):
        start_line = node["line_num"] - 1
        if i + 1 < len(all_nodes):
            end_line = all_nodes[i + 1]["line_num"] - 1
        else:
            end_line = len(txt_lines)

        node["text"] = "\n".join(txt_lines[start_line:end_line]).strip()
    return all_nodes


async def txt_to_tree(
    txt_path,
    if_thinning=False,
    min_token_threshold=None,
    if_add_node_summary="no",
    summary_token_threshold=None,
    model=None,
    if_add_doc_description="no",
    if_add_node_text="no",
    if_add_node_id="yes",
    summary_model=None,
):
    """Process a plain text document into a PageIndex tree structure."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="latin-1") as f:
            txt_content = f.read()

    line_count = txt_content.count("\n") + 1

    print("Extracting nodes from text document...")
    node_list, txt_lines = extract_nodes_from_txt(txt_content)

    print("Extracting text content from nodes...")
    nodes_with_content = extract_node_text_content(node_list, txt_lines)

    if if_thinning:
        nodes_with_content = update_node_list_with_text_token_count(
            nodes_with_content, model=model
        )
        print("Thinning nodes...")
        nodes_with_content = tree_thinning_for_index(
            nodes_with_content, min_token_threshold, model=model
        )

    print("Building tree from nodes...")
    tree_structure = build_tree_from_nodes(nodes_with_content)

    if if_add_node_id == "yes":
        write_node_id(tree_structure)

    print("Formatting tree structure...")

    if if_add_node_summary == "yes":
        summary_model = summary_model or model
        tree_structure = format_structure(
            tree_structure,
            order=[
                "title",
                "node_id",
                "line_num",
                "summary",
                "prefix_summary",
                "text",
                "nodes",
            ],
        )

        print("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_structure_md(
            tree_structure,
            summary_token_threshold=summary_token_threshold,
            model=summary_model,
        )

        if if_add_node_text == "no":
            tree_structure = format_structure(
                tree_structure,
                order=[
                    "title",
                    "node_id",
                    "line_num",
                    "summary",
                    "prefix_summary",
                    "nodes",
                ],
            )

        if if_add_doc_description == "yes":
            print("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=summary_model)
            return {
                "doc_name": os.path.splitext(os.path.basename(txt_path))[0],
                "doc_description": doc_description,
                "line_count": line_count,
                "structure": tree_structure,
            }
    else:
        if if_add_node_text == "yes":
            tree_structure = format_structure(
                tree_structure,
                order=[
                    "title",
                    "node_id",
                    "line_num",
                    "summary",
                    "prefix_summary",
                    "text",
                    "nodes",
                ],
            )
        else:
            tree_structure = format_structure(
                tree_structure,
                order=[
                    "title",
                    "node_id",
                    "line_num",
                    "summary",
                    "prefix_summary",
                    "nodes",
                ],
            )

    return {
        "doc_name": os.path.splitext(os.path.basename(txt_path))[0],
        "line_count": line_count,
        "structure": tree_structure,
    }
