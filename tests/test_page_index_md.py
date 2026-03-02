"""Unit tests for pageindex.page_index_md pure functions (no API/IO)."""
import pytest
from pageindex.page_index_md import (
    extract_nodes_from_markdown,
    extract_node_text_content,
    build_tree_from_nodes,
    clean_tree_for_output,
)


class TestExtractNodesFromMarkdown:
    def test_single_header(self):
        md = "# Title"
        node_list, lines = extract_nodes_from_markdown(md)
        assert len(node_list) == 1
        assert node_list[0]["node_title"] == "Title"
        assert node_list[0]["line_num"] == 1

    def test_multiple_levels(self):
        md = """# One
## Two
### Three
"""
        node_list, lines = extract_nodes_from_markdown(md)
        assert len(node_list) == 3
        assert node_list[0]["node_title"] == "One"
        assert node_list[1]["node_title"] == "Two"
        assert node_list[2]["node_title"] == "Three"

    def test_skips_headers_in_code_block(self):
        md = """# Real Header
```
# Not a header
```
## Another
"""
        node_list, lines = extract_nodes_from_markdown(md)
        titles = [n["node_title"] for n in node_list]
        assert "Real Header" in titles
        assert "Another" in titles
        assert "Not a header" not in titles


class TestExtractNodeTextContent:
    def test_assigns_level_and_text(self):
        md = """# A
content for A
## B
content for B
"""
        node_list, lines = extract_nodes_from_markdown(md)
        nodes = extract_node_text_content(node_list, lines)
        assert len(nodes) == 2
        assert nodes[0]["level"] == 1
        assert nodes[0]["title"] == "A"
        assert "content for A" in nodes[0]["text"]
        assert nodes[1]["level"] == 2
        assert "content for B" in nodes[1]["text"]


class TestBuildTreeFromNodes:
    def test_single_node(self):
        nodes = [
            {"title": "Root", "line_num": 1, "level": 1, "text": "x"}
        ]
        tree = build_tree_from_nodes(nodes)
        assert len(tree) == 1
        assert tree[0]["title"] == "Root"
        assert tree[0]["node_id"] == "0001"
        assert tree[0]["nodes"] == []

    def test_parent_child(self):
        nodes = [
            {"title": "Parent", "line_num": 1, "level": 1, "text": "p"},
            {"title": "Child", "line_num": 2, "level": 2, "text": "c"},
        ]
        tree = build_tree_from_nodes(nodes)
        assert len(tree) == 1
        assert len(tree[0]["nodes"]) == 1
        assert tree[0]["nodes"][0]["title"] == "Child"

    def test_empty_list_returns_empty(self):
        assert build_tree_from_nodes([]) == []


class TestCleanTreeForOutput:
    def test_keeps_title_node_id_text_line_num(self):
        node = {
            "title": "T",
            "node_id": "0001",
            "text": "content",
            "line_num": 1,
            "nodes": [],
            "extra": "ignored",
        }
        tree = [node]
        cleaned = clean_tree_for_output(tree)
        assert cleaned[0]["title"] == "T"
        assert cleaned[0]["node_id"] == "0001"
        assert cleaned[0]["text"] == "content"
        assert cleaned[0]["line_num"] == 1
        assert "extra" not in cleaned[0]

    def test_recurses_children(self):
        tree = [
            {
                "title": "A",
                "node_id": "0001",
                "text": "a",
                "line_num": 1,
                "nodes": [
                    {"title": "B", "node_id": "0002", "text": "b", "line_num": 2, "nodes": []}
                ],
            }
        ]
        cleaned = clean_tree_for_output(tree)
        assert len(cleaned[0]["nodes"]) == 1
        assert cleaned[0]["nodes"][0]["title"] == "B"
