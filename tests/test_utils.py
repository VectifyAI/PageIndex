"""Unit tests for pageindex.utils pure functions (no API/IO)."""
import json
import pytest
from pageindex.utils import (
    count_tokens,
    get_json_content,
    extract_json,
    write_node_id,
    get_nodes,
    structure_to_list,
    get_leaf_nodes,
    get_first_start_page_from_text,
    get_last_start_page_from_text,
    sanitize_filename,
    list_to_tree,
    add_preface_if_needed,
    remove_fields,
    reorder_dict,
    convert_physical_index_to_int,
    convert_page_to_int,
)


class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_non_empty_string_returns_positive(self):
        n = count_tokens("hello world", model="gpt-4o")
        assert n > 0

    def test_model_none_uses_default(self):
        """count_tokens with model=None should not raise (uses default gpt-4o)."""
        n = count_tokens("test", model=None)
        assert n >= 0


class TestGetJsonContent:
    def test_extracts_content_between_delimiters(self):
        response = "prefix\n```json\n{\"a\": 1}\n```\nsuffix"
        assert get_json_content(response) == "{\"a\": 1}"

    def test_no_delimiters_returns_stripped(self):
        response = "  {\"x\": 2}  "
        assert get_json_content(response) == "{\"x\": 2}"


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_json_in_code_block(self):
        content = "```json\n{\"b\": 2}\n```"
        assert extract_json(content) == {"b": 2}

    def test_none_word_boundary_not_corrupted(self):
        """'None of the Above' should not become 'null of the Above'."""
        content = '{"title": "None of the Above", "value": null}'
        result = extract_json(content)
        assert result.get("title") == "None of the Above"
        assert result.get("value") is None

    def test_python_none_replaced(self):
        content = '{"key": None}'
        result = extract_json(content)
        assert result == {"key": None}


class TestGetLeafNodes:
    def test_leaf_node_without_nodes_key(self):
        """Node with no 'nodes' key should be treated as leaf (no KeyError)."""
        structure = {"title": "A", "node_id": "0001"}
        leaves = get_leaf_nodes(structure)
        assert len(leaves) == 1
        assert leaves[0]["title"] == "A"

    def test_leaf_node_with_empty_nodes(self):
        structure = {"title": "A", "nodes": []}
        leaves = get_leaf_nodes(structure)
        assert len(leaves) == 1

    def test_nested_structure(self):
        structure = {
            "title": "Root",
            "nodes": [
                {"title": "Child", "nodes": []}
            ],
        }
        leaves = get_leaf_nodes(structure)
        assert len(leaves) == 1
        assert leaves[0]["title"] == "Child"


class TestPageIndexFromText:
    def test_get_first_start_page(self):
        text = "x <start_index_5> y <end_index_5>"
        assert get_first_start_page_from_text(text) == 5

    def test_get_first_start_page_no_match(self):
        assert get_first_start_page_from_text("no tags") == -1

    def test_get_last_start_page(self):
        text = "<start_index_1> a <start_index_2> b <start_index_10> c"
        assert get_last_start_page_from_text(text) == 10

    def test_get_last_start_page_no_match(self):
        assert get_last_start_page_from_text("no tags") == -1


class TestSanitizeFilename:
    def test_replaces_slash(self):
        assert sanitize_filename("a/b/c") == "a-b-c"

    def test_custom_replacement(self):
        assert sanitize_filename("a/b", replacement="_") == "a_b"


class TestListToTree:
    def test_flat_list(self):
        data = [
            {"structure": "1", "title": "A", "start_index": 1, "end_index": 2},
            {"structure": "2", "title": "B", "start_index": 3, "end_index": 4},
        ]
        tree = list_to_tree(data)
        assert len(tree) == 2
        assert tree[0]["title"] == "A" and tree[1]["title"] == "B"

    def test_hierarchy(self):
        data = [
            {"structure": "1", "title": "A", "start_index": 1, "end_index": 5},
            {"structure": "1.1", "title": "A1", "start_index": 2, "end_index": 3},
        ]
        tree = list_to_tree(data)
        assert len(tree) == 1
        assert tree[0]["title"] == "A"
        assert len(tree[0]["nodes"]) == 1
        assert tree[0]["nodes"][0]["title"] == "A1"


class TestAddPrefaceIfNeeded:
    def test_empty_list_unchanged(self):
        assert add_preface_if_needed([]) == []
        assert add_preface_if_needed(None) is None

    def test_first_physical_index_one_no_preface(self):
        data = [{"physical_index": 1, "title": "Intro"}]
        assert add_preface_if_needed(data) == data
        assert len(data) == 1

    def test_first_physical_index_gt_one_adds_preface(self):
        data = [{"physical_index": 3, "title": "Chapter 1"}]
        result = add_preface_if_needed(data)
        assert len(result) == 2
        assert result[0]["title"] == "Preface"
        assert result[0]["physical_index"] == 1


class TestRemoveFields:
    def test_removes_specified_field(self):
        data = {"a": 1, "text": "hello", "nodes": []}
        out = remove_fields(data, fields=["text"])
        assert "text" not in out
        assert out["a"] == 1

    def test_nested_removal(self):
        data = {"text": "x", "nodes": [{"text": "y"}]}
        out = remove_fields(data, fields=["text"])
        assert "text" not in out
        assert "text" not in out["nodes"][0]


class TestReorderDict:
    def test_reorders_by_key_order(self):
        data = {"b": 2, "a": 1, "c": 3}
        out = reorder_dict(data, ["a", "b", "c"])
        assert list(out.keys()) == ["a", "b", "c"]

    def test_skips_missing_keys(self):
        data = {"a": 1}
        out = reorder_dict(data, ["a", "b", "c"])
        assert out == {"a": 1}

    def test_empty_key_order_returns_unchanged(self):
        data = {"a": 1}
        assert reorder_dict(data, None) == data


class TestConvertPhysicalIndexToInt:
    def test_list_of_dicts_with_tag_format(self):
        data = [{"physical_index": "<physical_index_5>"}]
        convert_physical_index_to_int(data)
        assert data[0]["physical_index"] == 5

    def test_string_tag_returns_int(self):
        assert convert_physical_index_to_int("<physical_index_10>") == 10


class TestConvertPageToInt:
    def test_converts_string_page_to_int(self):
        data = [{"page": "3", "title": "A"}]
        convert_page_to_int(data)
        assert data[0]["page"] == 3

    def test_invalid_string_left_unchanged(self):
        data = [{"page": "nope", "title": "A"}]
        convert_page_to_int(data)
        assert data[0]["page"] == "nope"


class TestWriteNodeId:
    def test_assigns_zero_padded_ids(self):
        structure = [{"title": "A", "nodes": []}]
        write_node_id(structure)
        assert structure[0]["node_id"] == "0000"

    def test_nested_ids(self):
        structure = [{"title": "A", "nodes": [{"title": "B"}]}]
        write_node_id(structure)
        assert structure[0]["node_id"] == "0000"
        assert structure[0]["nodes"][0]["node_id"] == "0001"


class TestGetNodesAndStructureToList:
    def test_structure_to_list_flattens(self):
        tree = [{"title": "A", "nodes": [{"title": "B"}]}]
        flat = structure_to_list(tree)
        assert len(flat) == 2
        assert flat[0]["title"] == "A" and flat[1]["title"] == "B"

    def test_get_nodes_excludes_children_from_node_dict(self):
        tree = [{"title": "A", "nodes": [{"title": "B"}]}]
        nodes = get_nodes(tree)
        assert len(nodes) == 2
        assert "nodes" not in nodes[0] or nodes[0].get("nodes") is None
