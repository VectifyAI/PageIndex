import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

page_index_module = importlib.import_module("pageindex.page_index")
from pageindex.page_index import (
    add_page_offset_to_toc_json,
    calculate_page_offset,
    process_none_page_numbers,
)
from pageindex.utils import get_leaf_nodes, list_to_tree


class TestIssue330GetLeafNodes:
    def test_leaf_node_without_nodes_key(self):
        leaf = {"title": "Section", "start_index": 1, "end_index": 1}
        result = get_leaf_nodes(leaf)
        assert len(result) == 1
        assert result[0]["title"] == "Section"
        assert "nodes" not in result[0]

    def test_tree_from_list_to_tree(self):
        flat = [
            {"structure": "1", "title": "Root", "start_index": 1, "end_index": 2},
            {"structure": "1.1", "title": "Child", "start_index": 2, "end_index": 2},
        ]
        tree = list_to_tree(flat)
        leaves = get_leaf_nodes(tree)
        titles = {node["title"] for node in leaves}
        assert titles == {"Child"}


class TestIssue153PageOffset:
    def test_calculate_page_offset_returns_none_for_empty_pairs(self):
        assert calculate_page_offset([]) is None

    def test_add_page_offset_with_none_offset_defaults_to_zero(self):
        data = [{"title": "Intro", "page": 3}]
        offset = calculate_page_offset([]) or 0
        result = add_page_offset_to_toc_json(data, offset)
        assert result[0]["physical_index"] == 3
        assert "page" not in result[0]


class TestIssue97ProcessNonePageNumbers:
    @pytest.mark.parametrize(
        "item",
        [
            {"title": "Missing page key"},
            {"title": "Explicit None page", "page": None},
        ],
    )
    def test_missing_page_key_does_not_raise(self, item, monkeypatch):
        def fake_add_page_number_to_toc(page_contents, item_copy, model=None):
            return [{"physical_index": "<physical_index_2>"}]

        monkeypatch.setattr(
            page_index_module,
            "add_page_number_to_toc",
            fake_add_page_number_to_toc,
        )

        toc_items = [item]
        page_list = [("page one",), ("page two",)]

        result = process_none_page_numbers(toc_items, page_list, start_index=1, model="test")

        assert result[0]["physical_index"] == 2
        assert "page" not in result[0]
