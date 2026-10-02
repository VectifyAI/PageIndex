"""summary_max_input_tokens: prompts that grow with the document stay within the model's context."""
import logging

import pytest

import pageindex.utils as utils
from pageindex import PageIndexLocalClient
from pageindex.client import PageIndexAPIError


def _tree():
    long = "word " * 400
    return [{"title": "A", "node_id": "0000", "summary": long, "nodes": [
                {"title": "A1", "node_id": "0001", "summary": long, "nodes": [
                    {"title": "A1a", "node_id": "0002", "summary": long}]}]},
            {"title": "B", "node_id": "0003", "summary": long}]


def _titles(nodes):
    return [n["title"] for n in nodes] + [t for n in nodes for t in _titles(n.get("nodes", []))]


def test_input_budget():
    assert utils.input_budget(None) is None
    assert utils.input_budget(8192) == 6763
    assert utils.input_budget(10) == 1


def test_fit_structure_returns_a_structure_that_fits_unchanged():
    tree = _tree()
    assert utils.fit_structure(tree, 10 ** 6) is tree


def test_fit_structure_cuts_from_the_deepest_level():
    tree = _tree()
    full = utils.count_tokens(str(tree))
    two_levels = utils.count_tokens(str(utils._prune_depth(tree, 1)))
    one_level = utils.count_tokens(str(utils._prune_depth(tree, 0)))
    assert one_level < two_levels < full
    assert _titles(utils.fit_structure(tree, full - 1)) == ["A", "B", "A1"]
    assert _titles(utils.fit_structure(tree, two_levels - 1)) == ["A", "B"]
    assert tree == _tree()


def test_fit_structure_warns_when_the_top_level_alone_is_too_big(caplog):
    with caplog.at_level(logging.WARNING):
        out = utils.fit_structure(_tree(), 10)
    assert _titles(out) == ["A", "B"]
    assert "exceeds the 10 token input budget" in caplog.text


def _description_prompt(monkeypatch, **kwargs):
    prompts = []
    monkeypatch.setattr(utils, "llm_completion", lambda model, prompt: prompts.append(prompt) or "d")
    utils.generate_doc_description(_tree(), **kwargs)
    return prompts[0]


def test_description_prompt_is_unchanged_without_a_budget(monkeypatch):
    assert f"Document Structure: {_tree()}" in _description_prompt(monkeypatch)


def test_description_prompt_is_cut_to_the_budget(monkeypatch):
    prompt = _description_prompt(monkeypatch, max_input_tokens=1000)
    assert "'title': 'A'" in prompt and "'title': 'A1a'" not in prompt


def test_client_takes_the_budget_flat_or_in_the_index_slot():
    assert PageIndexLocalClient(summary_max_input_tokens=8192)._api._summary_max_input_tokens == 8192
    assert PageIndexLocalClient(index={"summary_max_input_tokens": 8192})._api._summary_max_input_tokens == 8192
    assert PageIndexLocalClient()._api._summary_max_input_tokens is None
    with pytest.raises(PageIndexAPIError, match="summary_max_input_tokens must be a positive int"):
        PageIndexLocalClient(summary_max_input_tokens=-1)
    with pytest.raises(PageIndexAPIError, match="summary_max_input_tokens must be a"):
        PageIndexLocalClient(summary_max_input_tokens="8192")


def test_the_budget_reaches_the_description(tmp_path, sample_pdf, monkeypatch):
    import pageindex.flash
    seen = {}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", lambda p, **kw: {
        "structure": [{"title": "T", "start_index": 1, "end_index": 1, "summary": "s"}]})
    monkeypatch.setattr(utils, "generate_doc_description",
                        lambda structure, model=None, max_input_tokens=None:
                        seen.update(max_input_tokens=max_input_tokens) or "d")
    monkeypatch.chdir(tmp_path)
    PageIndexLocalClient(summary_max_input_tokens=8192).submit_document(sample_pdf)
    assert seen == {"max_input_tokens": 8192}


def _leaf(monkeypatch, lines, max_input_tokens, reply=lambda prompt: '{"summary": "ok"}', **node):
    import asyncio
    prompts = []

    async def fake(model, prompt):
        prompts.append(prompt)
        return reply(prompt)
    monkeypatch.setattr(utils, "llm_acompletion", fake)
    structure = [{"title": "T", "start_index": 1, "end_index": 1, **node}]
    asyncio.run(utils.summarize_tree(structure, [("\n".join(lines), 0)], small_node_tokens=0,
                                     max_input_tokens=max_input_tokens))
    return prompts, structure[0]["summary"]


def _chunk(prompt):
    return prompt.split("Given Text: ")[1].split("\n\n    Reply strictly")[0].split("\n")


def test_leaf_within_the_budget_is_one_call(monkeypatch):
    lines = [f"line {i} about apples" for i in range(20)]
    assert len(_leaf(monkeypatch, lines, None)[0]) == 1
    assert len(_leaf(monkeypatch, lines, 8192)[0]) == 1


def test_long_leaf_is_split_by_lines_and_combined(monkeypatch):
    lines = [f"line {i} " + "word " * 10 for i in range(400)]
    budget = utils.input_budget(1200)
    prompts, summary = _leaf(monkeypatch, lines, 1200,
                             reply=lambda p: '{"summary": "combined"}' if "Part Summaries" in p else '{"summary": "part"}')
    parts, combine = prompts[:-1], prompts[-1]
    assert len(parts) > 2 and "Part Summaries" in combine and summary == "combined"
    assert [line for p in parts for line in _chunk(p)] == lines
    assert all(utils.count_tokens("\n".join(_chunk(p))) <= budget for p in parts)
    assert combine.count("Part ") == len(parts) + 1  # one per part, plus the instruction


def test_many_parts_are_combined_in_levels(monkeypatch):
    lines = ["word " * 40 for _ in range(300)]
    prompts, summary = _leaf(monkeypatch, lines, 1000,
                             reply=lambda p: '{"summary": "' + "long " * 120 + '"}')
    assert sum("Part Summaries" in p for p in prompts) > 1
    assert summary.startswith("long")


def test_a_line_over_the_budget_is_cut_at_spaces(monkeypatch):
    prompts, _ = _leaf(monkeypatch, ["start", "word " * 3000, "end"], 1000)
    budget = utils.input_budget(1000)
    parts = [p for p in prompts if "Part Summaries" not in p]
    assert len(parts) > 3
    assert all(utils.count_tokens("\n".join(_chunk(p))) <= budget for p in parts)
    assert _chunk(parts[0])[0] == "start" and _chunk(parts[-1])[-1].endswith("end")


def test_a_node_merged_from_same_page_siblings_is_not_split(monkeypatch):
    lines = [f"line {i} " + "word " * 10 for i in range(400)]
    prompts, _ = _leaf(monkeypatch, lines, 1200, _same_page=True, key_items=["A", "B"])
    assert len(prompts) == 1


def _expand(monkeypatch, args, pages):
    import asyncio
    from types import SimpleNamespace
    import pageindex.tree_optimize as tree_optimize
    asked = []

    async def fake_ask(model, prompt):
        asked.append(prompt)
        return {"subsections": []}
    monkeypatch.setattr(tree_optimize, "ask_model", fake_ask)
    node = {"title": "References", "start_index": 1, "end_index": 3, "node_id": "n1"}
    out = asyncio.run(tree_optimize.propose_children(node, pages, SimpleNamespace(model="m", **args)))
    return out, asked


def test_expand_is_skipped_when_the_pages_overrun_the_budget(monkeypatch, caplog):
    pages = ["word " * 400] * 3
    with caplog.at_level(logging.WARNING):
        out, asked = _expand(monkeypatch, {"input_budget": 500}, pages)
    assert out == [] and asked == []
    assert "expand skipped for 'References': pages 1-3 exceed the 500 token input budget" in caplog.text


def test_expand_asks_the_model_within_the_budget_or_without_one(monkeypatch):
    pages = ["word " * 400] * 3
    assert len(_expand(monkeypatch, {"input_budget": 10 ** 6}, pages)[1]) == 1
    assert len(_expand(monkeypatch, {"input_budget": None}, pages)[1]) == 1
    assert len(_expand(monkeypatch, {}, pages)[1]) == 1


def test_optimize_leaves_the_node_collapsed_when_the_budget_is_too_small(monkeypatch):
    import asyncio
    import pageindex.tree_optimize as tree_optimize
    from test_client import _expand_fixture
    tree, pages, lines = _expand_fixture()
    asked = []

    async def fake_ask(model, prompt):
        asked.append(prompt)
        return {"subsections": [{"title": "Sub One", "page": 4}]}
    monkeypatch.setattr(tree_optimize, "ask_model", fake_ask)
    asyncio.run(tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True,
                                       max_input_tokens=300))
    assert asked == []


def test_budget_tokens_count_the_digits_tiktoken_packs_together():
    text = "kernel, 8, 10, 317, 3482, 2024"
    assert utils.budget_tokens(text) == utils.count_tokens(text) + 0 + 1 + 2 + 2 + 2  # 8, 10, 317, 3482, 2024
    assert utils.budget_tokens("no digits here") == utils.count_tokens("no digits here")


def test_an_index_full_of_page_numbers_is_split_by_its_digit_aware_size(monkeypatch):
    lines = [f"term {i}, " + ", ".join(str(100 + 7 * k) for k in range(30)) for i in range(120)]
    budget = utils.input_budget(1200)
    prompts, _ = _leaf(monkeypatch, lines, 1200)
    parts = [p for p in prompts if "Part Summaries" not in p]
    assert all(utils.budget_tokens("\n".join(_chunk(p))) <= budget for p in parts)
    assert any(utils.count_tokens("\n".join(_chunk(p))) < budget * 0.8 for p in parts)
