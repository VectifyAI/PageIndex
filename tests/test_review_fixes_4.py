"""Regression tests for the PR #272 max-review findings #2-#4 (indexing crashes)."""
import logging

from pageindex.index import page_index as pi


# ── #2: unresolvable page offset must fall back, not TypeError ───────────────

def test_toc_with_page_numbers_falls_back_when_offset_unresolvable(monkeypatch):
    toc = [{"structure": "1", "title": "Intro", "page": 5}]
    monkeypatch.setattr(pi, "toc_transformer",
                        lambda content, model=None: [dict(item) for item in toc])
    # no title matches between transformed TOC and physical-index extraction
    monkeypatch.setattr(pi, "toc_index_extractor", lambda t, c, model=None: [])

    def _fail(*a, **k):
        raise AssertionError("no per-item LLM lookups when the offset is unresolvable")
    monkeypatch.setattr(pi, "add_page_number_to_toc", _fail)

    result = pi.process_toc_with_page_numbers(
        "toc text", [0], [("page one", 5), ("page two", 5)],
        toc_check_page_num=1, model=None, logger=logging.getLogger("test"),
    )
    # items come back without physical_index so meta_processor cascades to the
    # no-page-number mode
    assert all(item.get("physical_index") is None for item in result)


# ── #3: empty/unparseable LLM result must not KeyError ───────────────────────

import pytest


@pytest.mark.parametrize("llm_result", [
    {},                                        # unparseable → extract_json {}
    ["garbage string"],                        # list of non-dicts
    [{"physical_index": "<physical_index_abc>"}],  # tag with a non-numeric index
    [{"title": "B"}],                          # dict missing physical_index
])
def test_process_none_page_numbers_tolerates_malformed_llm_result(monkeypatch, llm_result):
    monkeypatch.setattr(pi, "add_page_number_to_toc", lambda *a, **k: llm_result)
    items = [
        {"title": "A", "physical_index": 1},
        {"title": "B", "page": 2},
        {"title": "C", "physical_index": 3},
    ]
    out = pi.process_none_page_numbers(items, [("p1", 5), ("p2", 5), ("p3", 5)])
    assert out[1].get("physical_index") is None


# ── #4: non-int physical_index must be invalidated, not TypeError ────────────

def test_validate_and_truncate_tolerates_non_int_physical_index():
    items = [
        {"title": "A", "physical_index": "5"},
        {"title": "B", "physical_index": 3},
        {"title": "C", "physical_index": 99},
    ]
    out = pi.validate_and_truncate_physical_indices(items, 20)
    assert out[0]["physical_index"] is None
    assert out[1]["physical_index"] == 3
    assert out[2]["physical_index"] is None
