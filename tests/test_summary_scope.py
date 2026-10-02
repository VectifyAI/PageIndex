"""summary_scope="section": a leaf is summarized from the blocks between its heading and the next one."""
import asyncio
from pathlib import Path

import pytest

import pageindex.utils as utils
from pageindex import PageIndexLocalClient
from pageindex.client import PageIndexAPIError
from pageindex.flash import page_index_flash
from pageindex.flash.main import extract_toc

PDF = str(Path(__file__).parent / "data" / "flash" / "ja_report.pdf")


def _flat(nodes):
    for n in nodes:
        yield n
        yield from _flat(n.get("nodes", []))


def test_extract_toc_reports_the_heading_blocks_on_request():
    result = extract_toc(PDF, use_embedded_toc=False, with_blocks=True)
    nodes = list(_flat(result["structure"]))
    positions = [n["_pos"] for n in nodes]
    assert nodes and all(isinstance(p, int) for p in positions) and positions == sorted(positions)
    assert all(n["title"] in result["block_texts"][n["_pos"]] for n in nodes)


def test_extract_toc_is_unchanged_without_blocks():
    result = extract_toc(PDF, use_embedded_toc=False)
    assert "block_texts" not in result
    assert not any("_pos" in n for n in _flat(result["structure"]))


def test_the_internal_positions_do_not_reach_the_result():
    result = page_index_flash(PDF, summary=False, optimize=False, use_embedded_toc=False,
                              summary_scope="section")
    assert "block_texts" not in result
    assert not any("_pos" in n for n in _flat(result["structure"]))


def _prompts(structure, blocks, pages=(("page one\npage two", 0),)):
    prompts = []

    async def fake(model, prompt):
        prompts.append(prompt.split("Given Text: ")[1].split("\n\n    Reply strictly")[0])
        return '{"summary": "ok"}'
    utils.llm_acompletion, saved = fake, utils.llm_acompletion
    try:
        asyncio.run(utils.summarize_tree(structure, list(pages), small_node_tokens=0, blocks=blocks))
    finally:
        utils.llm_acompletion = saved
    return prompts


def _tree(*positions):
    return [{"title": f"S{i}", "start_index": 1, "end_index": 1, **({} if p is None else {"_pos": p})}
            for i, p in enumerate(positions)]


def test_a_leaf_is_summarized_from_its_own_blocks():
    prompts = _prompts(_tree(0, 3), ["a0", "a1", "a2", "b0", "b1"])
    assert sorted(prompts) == ["a0\na1\na2", "b0\nb1"]


def test_a_section_ends_at_the_next_heading_in_the_document_not_in_the_tree():
    prompts = _prompts(_tree(3, 0), ["a0", "a1", "a2", "b0", "b1"])
    assert sorted(prompts) == ["a0\na1\na2", "b0\nb1"]
    assert _prompts(_tree(0), ["a0", "a1"]) == ["a0\na1"]


def test_a_leaf_falls_back_to_its_pages_without_its_own_position_or_blocks():
    blocks = ["a0", "a1", "b0"]
    assert _prompts(_tree(None, 2), blocks)[0] == "page one\npage two"
    assert _prompts(_tree(0, 2), None) == ["page one\npage two"] * 2


def test_a_leaf_falls_back_to_its_pages_when_a_section_without_a_position_starts_inside_it():
    pages = (("page one", 0), ("page two", 0), ("page three", 0))
    blocks = ["a0", "a1", "b0"]

    def tree(unlocated_page):
        return [{"title": "S0", "start_index": 1, "end_index": 2, "_pos": 0},
                {"title": "S1", "start_index": unlocated_page, "end_index": unlocated_page},
                {"title": "S2", "start_index": 3, "end_index": 3, "_pos": 2}]
    assert set(_prompts(tree(2), blocks, pages)) == {"page onepage two", "page two", "b0"}
    assert "a0\na1" in set(_prompts(tree(3), blocks, pages))


def test_the_internal_position_is_dropped_from_the_summarized_tree():
    structure = _tree(0, 2)
    _prompts(structure, ["a0", "a1", "b0"])
    assert not any("_pos" in n for n in structure)


def test_client_takes_the_scope_flat_or_in_the_index_slot():
    assert PageIndexLocalClient(summary_scope="section")._api._summary_scope == "section"
    assert PageIndexLocalClient(index={"summary_scope": "section"})._api._summary_scope == "section"
    assert PageIndexLocalClient()._api._summary_scope == "pages"
    with pytest.raises(PageIndexAPIError, match='summary_scope must be "pages" or "section"'):
        PageIndexLocalClient(summary_scope="paragraph")


def test_flash_refuses_an_unknown_scope():
    with pytest.raises(ValueError, match='summary_scope must be "pages" or "section"'):
        page_index_flash(PDF, summary_scope="paragraph")


def test_the_scope_reaches_the_flash_indexer(tmp_path, sample_pdf, monkeypatch):
    import pageindex.flash
    seen = {}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", lambda p, **kw: seen.update(kw) or {
        "structure": [{"title": "T", "start_index": 1, "end_index": 1, "summary": "s"}]})
    monkeypatch.setattr(utils, "llm_completion", lambda model, prompt, **kw: "d")
    monkeypatch.chdir(tmp_path)
    PageIndexLocalClient(summary_scope="section").submit_document(sample_pdf)
    assert seen["summary_scope"] == "section"


def test_a_title_is_located_by_its_heading_block_then_by_a_block_that_is_mostly_the_title():
    from pageindex.flash.main import _find_heading
    blocks = [(0, "mgsm is evaluated across seven languages and the results are listed below", 0),
              (1, "5 2 4 multilingual benchmarks", 7),
              (2, "mgsm", 0),
              (3, "safety pretraining", 7)]
    assert _find_heading("Multilingual Benchmarks", blocks) == 1      # numbering prefix
    assert _find_heading("MGSM", blocks) == 2                         # not the paragraph that starts with it
    assert _find_heading("Safety Pre-training", blocks) == 3          # close enough
    assert _find_heading("Something else", blocks) is None


def test_the_closest_match_on_a_page_wins_over_the_first_one():
    from pageindex.flash.main import _find_heading
    blocks = [(0, "contributors and acknowledgements", 7), (1, "6 1 pre training evaluations", 7),
              (2, "contributors", 7), (3, "6 2 post training evaluations", 7)]
    assert _find_heading("Contributors", blocks) == 2
    assert _find_heading("6.2. Post-training Evaluations", blocks) == 3


def test_a_bookmark_one_page_off_finds_the_heading_block_on_the_next_or_previous_page():
    from pageindex.flash.main import locate_headings

    class B:
        def __init__(self, text, kind):
            self.text, self.type = text, kind
    import pageindex.flash.main as fm
    body = [B("previous paragraph mentions reliability and operational challenges", 0),
            B("3.3.4 Reliability and Operational Challenges", 8), B("body", 0), B("4 Annex", 7)]
    saved, fm.block_text = fm.block_text, lambda b: b.text
    try:
        tree = [{"title": "Reliability and Operational Challenges", "start_index": 12},
                {"title": "Annex", "start_index": 14}, {"title": "Missing", "start_index": 12}]
        locate_headings(tree, body, [12, 13, 13, 13])
    finally:
        fm.block_text = saved
    assert tree[0]["_pos"] == 1 and tree[1]["_pos"] == 3 and "_pos" not in tree[2]
