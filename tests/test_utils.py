import pytest
import json
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import (
    extract_json,
    write_node_id,
    get_nodes,
    get_leaf_nodes,
    is_leaf_node,
    structure_to_list,
    list_to_tree,
    sanitize_filename,
    convert_physical_index_to_int,
    format_structure,
    reorder_dict,
    remove_structure_text,
    add_preface_if_needed,
    print_toc,
    clean_structure_post,
    remove_fields,
    get_last_node,
    post_processing,
)


SAMPLE_TREE = {
    "title": "Root",
    "node_id": "0000",
    "start_index": 1,
    "end_index": 10,
    "nodes": [
        {
            "title": "Chapter 1",
            "node_id": "0001",
            "start_index": 1,
            "end_index": 5,
            "nodes": [
                {
                    "title": "Section 1.1",
                    "node_id": "0002",
                    "start_index": 1,
                    "end_index": 3,
                    "nodes": [],
                },
            ],
        },
        {
            "title": "Chapter 2",
            "node_id": "0003",
            "start_index": 6,
            "end_index": 10,
            "nodes": [],
        },
    ],
}


class TestExtractJson:
    def test_plain_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n```json\n{"answer": "yes"}\n```\nDone.'
        result = extract_json(text)
        assert result == {"answer": "yes"}

    def test_trailing_comma_cleanup(self):
        text = '{"items": [1, 2, 3,]}'
        result = extract_json(text)
        assert result == {"items": [1, 2, 3]}

    def test_python_none_converted_to_null(self):
        text = '{"value": None}'
        result = extract_json(text)
        assert result == {"value": None}

    def test_invalid_json_returns_empty_dict(self):
        result = extract_json("not json at all {{{}}")
        assert result == {}

    def test_nested_json(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        result = extract_json(json.dumps(data))
        assert result == data

    def test_json_array(self):
        text = '[{"a": 1}, {"b": 2}]'
        result = extract_json(text)
        assert result == [{"a": 1}, {"b": 2}]


class TestWriteNodeId:
    def test_single_dict(self):
        data = {"title": "Test", "nodes": []}
        write_node_id(data, 0)
        assert data["node_id"] == "0000"

    def test_nested_structure(self):
        data = {
            "title": "Root",
            "nodes": [
                {"title": "Child 1", "nodes": []},
                {"title": "Child 2", "nodes": []},
            ],
        }
        write_node_id(data, 0)
        assert data["node_id"] == "0000"
        assert data["nodes"][0]["node_id"] == "0001"
        assert data["nodes"][1]["node_id"] == "0002"

    def test_return_value_is_next_id(self):
        data = {"title": "Root", "nodes": [{"title": "Child", "nodes": []}]}
        next_id = write_node_id(data, 0)
        assert next_id == 2

    def test_list_input(self):
        data = [
            {"title": "A", "nodes": []},
            {"title": "B", "nodes": []},
        ]
        next_id = write_node_id(data, 5)
        assert data[0]["node_id"] == "0005"
        assert data[1]["node_id"] == "0006"
        assert next_id == 7


class TestGetNodes:
    def test_flat_extraction(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        nodes = get_nodes(tree)
        titles = [n["title"] for n in nodes]
        assert "Root" in titles
        assert "Chapter 1" in titles
        assert "Section 1.1" in titles
        assert "Chapter 2" in titles
        assert len(nodes) == 4

    def test_nodes_have_no_children_key(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        nodes = get_nodes(tree)
        for node in nodes:
            assert "nodes" not in node

    def test_list_input(self):
        items = [
            {"title": "A", "nodes": [{"title": "B", "nodes": []}]},
            {"title": "C", "nodes": []},
        ]
        nodes = get_nodes(items)
        assert len(nodes) == 3


class TestGetLeafNodes:
    def test_leaf_nodes(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        leaves = get_leaf_nodes(tree)
        titles = [n["title"] for n in leaves]
        assert "Section 1.1" in titles
        assert "Chapter 2" in titles
        assert len(leaves) == 2

    def test_single_leaf(self):
        tree = {"title": "Alone", "nodes": []}
        leaves = get_leaf_nodes(tree)
        assert len(leaves) == 1
        assert leaves[0]["title"] == "Alone"


class TestIsLeafNode:
    def test_leaf_node(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        assert is_leaf_node(tree, "0002") is True
        assert is_leaf_node(tree, "0003") is True

    def test_non_leaf_node(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        assert is_leaf_node(tree, "0001") is False
        assert is_leaf_node(tree, "0000") is False

    def test_nonexistent_node(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        assert is_leaf_node(tree, "9999") is False


class TestStructureToList:
    def test_basic(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        flat = structure_to_list(tree)
        assert len(flat) == 4

    def test_preserves_all_fields(self):
        tree = copy.deepcopy(SAMPLE_TREE)
        flat = structure_to_list(tree)
        assert flat[0]["title"] == "Root"


class TestListToTree:
    def test_simple_tree(self):
        flat = [
            {"structure": "1", "title": "Intro", "start_index": 1, "end_index": 5},
            {"structure": "1.1", "title": "Background", "start_index": 1, "end_index": 3},
            {"structure": "2", "title": "Methods", "start_index": 6, "end_index": 10},
        ]
        tree = list_to_tree(flat)
        assert len(tree) == 2
        assert tree[0]["title"] == "Intro"
        assert len(tree[0]["nodes"]) == 1
        assert tree[0]["nodes"][0]["title"] == "Background"
        assert "nodes" not in tree[1]

    def test_orphan_becomes_root(self):
        flat = [
            {"structure": "2.1", "title": "Orphan", "start_index": 1, "end_index": 2},
        ]
        tree = list_to_tree(flat)
        assert len(tree) == 1
        assert tree[0]["title"] == "Orphan"


class TestSanitizeFilename:
    def test_replaces_slash(self):
        assert sanitize_filename("a/b/c") == "a-b-c"

    def test_custom_replacement(self):
        assert sanitize_filename("a/b", replacement="_") == "a_b"

    def test_no_slash(self):
        assert sanitize_filename("normal.pdf") == "normal.pdf"


class TestConvertPhysicalIndexToInt:
    def test_tag_format(self):
        result = convert_physical_index_to_int("<physical_index_42>")
        assert result == 42

    def test_plain_format(self):
        result = convert_physical_index_to_int("physical_index_7")
        assert result == 7

    def test_list_format(self):
        data = [
            {"physical_index": "<physical_index_3>"},
            {"physical_index": "physical_index_5"},
        ]
        result = convert_physical_index_to_int(data)
        assert result[0]["physical_index"] == 3
        assert result[1]["physical_index"] == 5

    def test_non_matching_string(self):
        result = convert_physical_index_to_int("random_string")
        assert result is None


class TestFormatStructure:
    def test_reorder_keys(self):
        data = {"b": 2, "a": 1, "c": 3}
        result = format_structure(data, order=["a", "b", "c"])
        assert list(result.keys()) == ["a", "b", "c"]

    def test_no_order_returns_same(self):
        data = {"x": 1}
        result = format_structure(data, order=None)
        assert result == {"x": 1}

    def test_nested_reorder(self):
        data = {
            "b": 2,
            "a": 1,
            "nodes": [{"b": 20, "a": 10}],
        }
        result = format_structure(data, order=["a", "b", "nodes"])
        assert list(result.keys()) == ["a", "b", "nodes"]

    def test_empty_nodes_removed(self):
        data = {"title": "X", "nodes": []}
        result = format_structure(data, order=["title", "nodes"])
        assert "nodes" not in result


class TestReorderDict:
    def test_basic(self):
        data = {"c": 3, "a": 1, "b": 2}
        result = reorder_dict(data, ["a", "b", "c"])
        assert list(result.keys()) == ["a", "b", "c"]

    def test_missing_keys_skipped(self):
        data = {"a": 1}
        result = reorder_dict(data, ["a", "b"])
        assert result == {"a": 1}

    def test_empty_order(self):
        data = {"a": 1}
        result = reorder_dict(data, [])
        assert result == {"a": 1}


class TestRemoveStructureText:
    def test_removes_text_from_dict(self):
        data = {"title": "X", "text": "content", "nodes": []}
        result = remove_structure_text(data)
        assert "text" not in result
        assert result["title"] == "X"

    def test_removes_text_recursively(self):
        data = {"title": "X", "text": "a", "nodes": [{"title": "Y", "text": "b"}]}
        result = remove_structure_text(data)
        assert "text" not in result
        assert "text" not in result["nodes"][0]


class TestRemoveFields:
    def test_removes_text_by_default(self):
        data = {"title": "X", "text": "content"}
        result = remove_fields(data)
        assert "text" not in result
        assert result["title"] == "X"

    def test_custom_fields(self):
        data = {"a": 1, "b": 2, "c": 3}
        result = remove_fields(data, fields=["a", "c"])
        assert result == {"b": 2}


class TestAddPrefaceIfNeeded:
    def test_adds_preface_when_first_page_gt_1(self):
        data = [{"physical_index": 5, "structure": "1", "title": "Intro"}]
        result = add_preface_if_needed(data)
        assert len(result) == 2
        assert result[0]["title"] == "Preface"
        assert result[0]["physical_index"] == 1

    def test_no_preface_when_first_page_is_1(self):
        data = [{"physical_index": 1, "structure": "1", "title": "Intro"}]
        result = add_preface_if_needed(data)
        assert len(result) == 1

    def test_no_preface_when_none(self):
        data = [{"physical_index": None, "structure": "1", "title": "Intro"}]
        result = add_preface_if_needed(data)
        assert len(result) == 1

    def test_empty_list(self):
        assert add_preface_if_needed([]) == []


class TestCleanStructurePost:
    def test_removes_page_fields(self):
        data = {"page_number": 1, "start_index": 2, "end_index": 3, "title": "X"}
        result = clean_structure_post(data)
        assert "page_number" not in result
        assert "start_index" not in result
        assert "end_index" not in result
        assert result["title"] == "X"


class TestGetLastNode:
    def test_returns_last(self):
        data = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        assert get_last_node(data)["title"] == "C"
