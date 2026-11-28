"""Standard mode shares Flash's deterministic same-range consolidation."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from pageindex import page_index_classic as classic, tree_optimize, utils


def leaf(title, start, end):
    return {"title": title, "start_index": start, "end_index": end}


@pytest.mark.parametrize("titles", [("Install", "Usage"), ("安装", "使用")])
def test_root_siblings_preserve_all_original_titles_and_text(titles):
    tree = [leaf(title, 1, 1) for title in titles]
    body = "\n".join(titles) + "\nAll source facts remain."
    tree_optimize.merge_tree(tree)
    utils.add_node_text(tree, [(body, 0)])
    assert len(tree) == 1
    assert tree[0]["key_items"] == list(titles)
    assert tree[0]["text"] == body
    before = copy.deepcopy(tree)
    assert tree_optimize.merge_tree(tree) == before


def test_nested_siblings_merge_without_collapsing_distinct_long_sections():
    tree = [dict(leaf("Root", 1, 20), nodes=[
        dict(leaf("Chapter", 1, 10), nodes=[leaf("A", 1, 1), leaf("B", 1, 1),
                                          leaf("C", 2, 5), leaf("D", 6, 10)]),
        leaf("Next", 11, 20)])]
    tree_optimize.merge_tree(tree)
    children = tree[0]["nodes"][0]["nodes"]
    assert [n["title"] for n in children] == ["A; B", "C", "D"]
    assert [(n["start_index"], n["end_index"]) for n in children] == [(1, 1), (2, 5), (6, 10)]
    assert tree[0]["nodes"][1] == leaf("Next", 11, 20)


def test_matching_start_with_different_end_is_not_merged():
    tree = [leaf("Short", 1, 1), leaf("Long", 1, 5), leaf("Later", 6, 8)]
    expected = copy.deepcopy(tree)
    assert tree_optimize.merge_tree(tree) == expected


def test_merge_seam_consolidates_a_newly_collapsed_subtree():
    tree = [dict(leaf("Parent", 1, 1), nodes=[leaf("A", 1, 1), leaf("B", 1, 1)]),
            leaf("Sibling", 1, 1)]
    tree_optimize.merge_tree(tree)
    assert len(tree) == 1
    assert all(title in tree[0]["key_items"] for title in ("Parent", "A", "B", "Sibling"))
    assert not tree[0].get("nodes")


def test_identical_cross_page_ranges_keep_every_page():
    tree = [leaf("A", 1, 3), leaf("B", 1, 3)]
    tree_optimize.merge_tree(tree)
    utils.add_node_text(tree, [("first fact", 0), ("middle fact", 0), ("last fact", 0)])
    assert len(tree) == 1
    assert tree[0]["start_index"] == 1 and tree[0]["end_index"] == 3
    assert all(fact in tree[0]["text"] for fact in ("first fact", "middle fact", "last fact"))


def test_standard_pipeline_consolidates_before_numbering_and_summaries(monkeypatch, sample_pdf):
    async def parse(*args, **kwargs):
        return [leaf("安装", 1, 1), leaf("使用", 1, 1), leaf("跨页章节", 2, 3)]
    observed = []
    async def summarize(structure, pages, **kwargs):
        observed.extend(copy.deepcopy(structure))
        for node in structure:
            node["summary"] = node["text"]
    def no_llm(*args, **kwargs):
        pytest.fail("deterministic processing made a model call")
    monkeypatch.setattr(classic, "tree_parser", parse)
    monkeypatch.setattr(classic, "summarize_tree", summarize)
    monkeypatch.setattr(utils, "llm_completion", no_llm)
    monkeypatch.setattr(utils, "llm_acompletion", no_llm)
    pages = [("安装\n安装正文\n使用\n使用正文", 10), ("跨页第一部分", 10), ("跨页最后部分", 10)]
    opt = SimpleNamespace(model="fake", if_add_node_id="yes", if_add_node_text="yes",
                          if_add_node_summary="yes", if_add_doc_description="no")
    result = classic.page_index_main(sample_pdf, opt, page_list=pages)
    assert len(observed) == 2
    assert [n["node_id"] for n in observed] == ["0000", "0001"]
    assert observed[0]["key_items"] == ["安装", "使用"]
    assert observed[0]["text"] == pages[0][0]
    assert observed[1]["text"] == pages[1][0] + pages[2][0]
    assert result["structure"][0]["text"] == pages[0][0]


@pytest.mark.parametrize("small_node_tokens,expected_calls", [(0, 1), (100, 0)])
def test_real_summary_scheduler_has_no_extra_title_call(monkeypatch, small_node_tokens, expected_calls):
    tree = [leaf("安装", 1, 1), leaf("使用", 1, 1)]
    prompts = []
    async def model(model, prompt, **kwargs):
        prompts.append(prompt)
        return '{"summary":"all facts", "title":"Combined"}'
    monkeypatch.setattr(utils, "count_tokens", lambda text, **kwargs: len(text))
    monkeypatch.setattr(utils, "llm_acompletion", model)
    tree_optimize.merge_tree(tree)
    asyncio.run(utils.summarize_tree(tree, [("安装正文和使用正文", 0)], model="fake",
                                    small_node_tokens=small_node_tokens))
    assert len(prompts) == expected_calls
    assert tree[0]["key_items"] == ["安装", "使用"]
    if prompts:
        assert "安装正文和使用正文" in prompts[0]
