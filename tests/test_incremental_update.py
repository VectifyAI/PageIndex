"""Tests for incremental MD update (section-hash diff).

Covers the deterministic layer only — section identity, hashing, diff
classification and ancestor expansion — so it runs without an API key.

Run: python tests/test_incremental_update.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageindex.page_index_md import extract_nodes_from_markdown, extract_node_text_content
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
