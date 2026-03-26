import pytest
from pageindex.core.tree import list_to_tree, structure_to_list, get_nodes, write_node_id

@pytest.fixture
def sample_structure():
    return [
        {"structure": "1", "title": "Chapter 1", "start_index": 1, "end_index": 5},
        {"structure": "1.1", "title": "Section 1.1", "start_index": 1, "end_index": 3},
        {"structure": "1.2", "title": "Section 1.2", "start_index": 4, "end_index": 5},
        {"structure": "2", "title": "Chapter 2", "start_index": 6, "end_index": 10}
    ]

def test_list_to_tree(sample_structure):
    tree = list_to_tree(sample_structure)
    assert len(tree) == 2
    assert tree[0]["title"] == "Chapter 1"
    assert len(tree[0]["nodes"]) == 2
    assert tree[0]["nodes"][0]["title"] == "Section 1.1"
    assert tree[1]["title"] == "Chapter 2"
    assert "nodes" not in tree[1] or len(tree[1]["nodes"]) == 0

def test_structure_to_list(sample_structure):
    tree = list_to_tree(sample_structure)
    flat_list = structure_to_list(tree)
    # Note: structure_to_list might not preserve original order exactly or might include container nodes
    # But for our simple case it should be close.
    assert len(flat_list) == 4
    titles = [item["title"] for item in flat_list]
    assert "Chapter 1" in titles
    assert "Section 1.1" in titles

def test_write_node_id(sample_structure):
    tree = list_to_tree(sample_structure)
    write_node_id(tree)
    assert tree[0]["node_id"] == "0000"
    assert tree[0]["nodes"][0]["node_id"] == "0001"
