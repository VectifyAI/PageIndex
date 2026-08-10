"""Tests for incremental MD update (section-hash diff).

Covers the deterministic layer only — section identity, hashing, diff
classification and ancestor expansion — so it runs without an API key.

Run: python tests/test_incremental_update.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageindex.page_index_md import (
    extract_nodes_from_markdown,
    extract_node_text_content,
    build_tree_from_nodes,
    split_summary_fields,
)
from pageindex.utils import compute_section_hashes, find_ancestors


def _hashes(md):
    node_list, lines = extract_nodes_from_markdown(md)
    nodes = extract_node_text_content(node_list, lines)
    return compute_section_hashes(nodes)


def _diff(old, new):
    old_k, new_k = set(old), set(new)
    added = new_k - old_k
    deleted = old_k - new_k
    changed = {p for p in old_k & new_k if old[p] != new[p]}
    return added, deleted, changed


def test_title_path_is_hierarchical():
    md = "# Root\nintro\n## A\nalpha\n### A1\nsub\n## B\nbeta\n"
    node_list, lines = extract_nodes_from_markdown(md)
    nodes = extract_node_text_content(node_list, lines)
    paths = [n["title_path"] for n in nodes]
    assert paths == ["Root", "Root > A", "Root > A > A1", "Root > B"], paths


def test_unchanged_doc_has_identical_hashes():
    md = "# Root\nintro\n## A\nalpha\n## B\nbeta\n"
    assert _hashes(md) == _hashes(md)


def test_changed_added_deleted_classification():
    v1 = "# Root\nintro\n## A\nalpha\n## B\nbeta\n"
    v2 = "# Root\nintro\n## A\nalpha CHANGED\n## C\ngamma\n"
    added, deleted, changed = _diff(_hashes(v1), _hashes(v2))
    assert changed == {"Root > A"}, changed
    assert added == {"Root > C"}, added
    assert deleted == {"Root > B"}, deleted


def test_ancestors_expand_to_root():
    assert find_ancestors("Root > A > A1") == ["Root", "Root > A"]
    assert find_ancestors("Root") == []


def test_dirty_set_includes_ancestors():
    v1 = "# Root\nintro\n## A\nalpha\n### A1\nsub\n"
    v2 = "# Root\nintro\n## A\nalpha\n### A1\nsub CHANGED\n"
    _, _, changed = _diff(_hashes(v1), _hashes(v2))
    to_summarize = set(changed)
    for p in changed:
        to_summarize.update(find_ancestors(p))
    assert to_summarize == {"Root", "Root > A", "Root > A > A1"}, to_summarize


def test_build_tree_preserves_summaries():
    """update() attaches summaries before building the tree; they must survive."""
    md = "# Root\nintro\n## A\nalpha\n## B\nbeta\n"
    node_list, lines = extract_nodes_from_markdown(md)
    nodes = extract_node_text_content(node_list, lines)
    for n in nodes:
        n["summary"] = f"S:{n['title_path']}"

    tree = split_summary_fields(build_tree_from_nodes(nodes))
    root = tree[0]
    # Parents carry prefix_summary, leaves carry summary — as index() produces.
    assert root["prefix_summary"] == "S:Root", root
    assert "summary" not in root, root
    assert [c["summary"] for c in root["nodes"]] == ["S:Root > A", "S:Root > B"]


def test_tree_walk_paths_match_flat_title_paths():
    """The cached-summary lookup keys the old tree by walking it; those paths
    must equal the title_paths the new flat node list is keyed by."""
    md = "# Root\nintro\n## A\nalpha\n### A1\nsub\n## B\nbeta\n"
    node_list, lines = extract_nodes_from_markdown(md)
    nodes = extract_node_text_content(node_list, lines)
    tree = build_tree_from_nodes(nodes)

    walked = []

    def collect(ns, prefix=""):
        for n in ns:
            path = f"{prefix} > {n['title']}" if prefix else n["title"]
            walked.append(path)
            collect(n.get("nodes", []), path)

    collect(tree)
    assert walked == [n["title_path"] for n in nodes], walked


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
