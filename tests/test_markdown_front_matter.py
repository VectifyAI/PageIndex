"""Cross-feature contract: YAML metadata must not change body parsing.

The preface/Setext/fence implementations belong to #610/#611/#555. These
tests deliberately compare body behavior before/after adding metadata,
so they also apply when those changes land, without duplicating them.
"""
import pytest

from pageindex.page_index_md import (extract_node_text_content,
                                     extract_nodes_from_markdown)


BODIES = [
    "plain text with no heading\nsecond line\n",
    "intro fact\n\n# ATX title\nbody fact\n\n## Child\nchild fact\n",
    "Setext title\n============\nsetext body\n\n## ATX child\nchild body\n",
    "intro fact\n\nSetext title\n------------\nbody\n\n# Tail\ntail body\n",
    "# Before\n```md\n# example\nExample\n=======\n```\n## After\nend\n",
    "~~~md\n# example\nExample\n=======\n~~~\n# After\nend\n",
    "# Before\n````md\n```\n# example\n```\n````\n# After\nend\n",
    "# Before\n```\n# unclosed example\nno heading after this\n",
    "paragraph\n\n---\n\n# 中文标题\n正文事实\n\n**Tail**\nend\n",
    "\n\n# Title\nbody with  two spaces\n",
]


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("end", ["---", "...  "])
def test_metadata_is_transparent_to_body_outline_and_source_slices(body, end):
    # A valid YAML literal block can contain Markdown syntax and indented
    # document markers. None may open/close the body's fence or headings.
    prefix = "\n".join([
        "---", "# metadata comment", "example: |", "  **Metadata heading**",
        "  ---", "  ...", "  ```markdown", "  # metadata example", end, ""])
    offset = prefix.count("\n")
    plain, plain_lines = extract_nodes_from_markdown(body)
    wrapped, wrapped_lines = extract_nodes_from_markdown(prefix + body)
    assert wrapped_lines == prefix.split("\n")[:-1] + plain_lines
    assert wrapped == [{**node, "line_num": node["line_num"] + offset} for node in plain]
    plain_text = extract_node_text_content(plain, plain_lines)
    wrapped_text = extract_node_text_content(wrapped, wrapped_lines)
    assert wrapped_text == [{**node, "line_num": node["line_num"] + offset} for node in plain_text]


@pytest.mark.parametrize("metadata", ["# Comment\n", "sample: |\n  **Example**\n", "sample: |\n  ```\n"])
def test_metadata_only_does_not_create_a_heading_or_empty_preface(metadata):
    content = f"---\n{metadata}---\n\n"
    nodes, lines = extract_nodes_from_markdown(content)
    assert nodes == []
    assert extract_node_text_content(nodes, lines) == []


def test_unterminated_front_matter_does_not_hide_the_document():
    content = "---\n# Actual heading\nbody\n"
    nodes, lines = extract_nodes_from_markdown(content)
    assert nodes == [{"node_title": "Actual heading", "line_num": 2, "level": 1}]
    assert lines == content.split("\n")


def test_body_separator_is_not_a_second_front_matter_block():
    content = "# First\nbody\n\n---\n# Second\nmore\n"
    nodes, lines = extract_nodes_from_markdown(content)
    assert [node["node_title"] for node in nodes] == ["First", "Second"]
    texts = extract_node_text_content(nodes, lines)
    assert [node["text"] for node in texts] == ["# First\nbody\n\n---", "# Second\nmore"]
