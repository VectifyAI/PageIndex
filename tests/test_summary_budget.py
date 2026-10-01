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
