import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
