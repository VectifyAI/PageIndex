from pageindex.utils import get_leaf_nodes


def test_get_leaf_nodes_handles_missing_nodes_key():
    """Tree cleanup (see `clean_node` in utils.py) deletes the `nodes` key
    entirely from childless nodes rather than setting it to `[]`, so
    `get_leaf_nodes` must not assume the key is always present (#330)."""
    tree = {
        "title": "Root",
        "nodes": [
            {"title": "Leaf A"},  # no 'nodes' key at all
            {
                "title": "Branch",
                "nodes": [
                    {"title": "Leaf B"},
                ],
            },
        ],
    }

    leaves = get_leaf_nodes(tree)

    assert {leaf["title"] for leaf in leaves} == {"Leaf A", "Leaf B"}
    assert all("nodes" not in leaf for leaf in leaves)


def test_get_leaf_nodes_still_handles_empty_nodes_list():
    tree = {
        "title": "Root",
        "nodes": [
            {"title": "Leaf A", "nodes": []},
        ],
    }

    leaves = get_leaf_nodes(tree)

    assert [leaf["title"] for leaf in leaves] == ["Leaf A"]


def test_get_leaf_nodes_handles_list_of_trees():
    trees = [
        {"title": "Leaf A"},
        {"title": "Root", "nodes": [{"title": "Leaf B"}]},
    ]

    leaves = get_leaf_nodes(trees)

    assert {leaf["title"] for leaf in leaves} == {"Leaf A", "Leaf B"}
