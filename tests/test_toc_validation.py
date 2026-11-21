"""Physical-page invariants and repair lifecycle; no model/network calls."""
import asyncio
import copy
import logging
import re
from types import SimpleNamespace

import pytest

from pageindex import page_index_classic as classic
from pageindex.utils import LLMRetriesExhausted

LOG = logging.getLogger(__name__)


def toc_at(*indices):
    return [{"title": chr(65 + i), "physical_index": value}
            for i, value in enumerate(indices)]


def bad(toc, *indices):
    return [dict(toc[i], list_index=i) for i in indices]


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected model call")
    monkeypatch.setattr(classic, "llm_completion", forbidden)
    monkeypatch.setattr(classic, "llm_acompletion", forbidden)


@pytest.mark.parametrize("indices", [(2, 2, 8), (2, 8, 20), (1, 1, 1)])
def test_same_page_and_large_gaps_are_valid(indices):
    toc = toc_at(*indices)
    assert classic.validate_and_truncate_physical_indices(toc, 20) == toc_at(*indices)


@pytest.mark.parametrize("value", [0, -1, 21, True, False, 2.5, "2", "bad", None])
def test_invalid_physical_indices_are_not_accepted(value):
    toc = toc_at(value)
    classic.validate_and_truncate_physical_indices(toc, 20)
    assert toc[0]["physical_index"] is None


def test_decrease_is_detected_without_reordering_or_incrementing():
    toc = toc_at(2, 8, 4, 10)
    classic.validate_and_truncate_physical_indices(toc, 10)
    assert [item["title"] for item in toc] == ["A", "B", "C", "D"]
    assert [item["physical_index"] for item in toc] == [2, None, None, 10]


@pytest.mark.parametrize("value", [True, False, 2.0, 2.8, "2.5"])
def test_model_index_coercion_does_not_invent_integers(value):
    toc = toc_at(value)
    classic._validate_physical_indices(toc, 10)
    assert toc[0]["physical_index"] is None


def test_model_markers_and_integer_strings_still_work():
    toc = toc_at("<physical_index_2>", "8")
    assert classic._validate_physical_indices(toc, 10) == toc_at(2, 8)


def test_transient_failure_stays_unresolved_through_retry_limit(monkeypatch):
    calls = []
    async def fail(*args):
        calls.append(args)
        raise TimeoutError("synthetic timeout")
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", fail)
    toc = toc_at(1, 2)
    result, unresolved = asyncio.run(classic.fix_incorrect_toc_with_retries(
        toc, [("page", 1)] * 2, bad(toc, 1), max_attempts=3, logger=LOG))
    assert len(calls) == 3
    assert result == toc_at(1, 2)
    assert unresolved[0]["list_index"] == 1
    assert unresolved[0]["physical_index"] == 2
    assert "TimeoutError" in unresolved[0]["reason"]
    assert unresolved[0]["attempts"] == 3


def test_mixed_batch_retries_only_remaining_item(monkeypatch):
    calls = []
    async def locate(title, *args):
        calls.append(title)
        if title == "B" and calls.count("B") == 1:
            raise TimeoutError("once")
        return {"B": 2, "C": 3}[title]
    async def yes(*args):
        return {"answer": "yes"}
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    monkeypatch.setattr(classic, "check_title_appearance", yes)
    toc = toc_at(1, 2, 3)
    _, unresolved = asyncio.run(classic.fix_incorrect_toc_with_retries(
        toc, [("page", 1)] * 3, bad(toc, 1, 2), logger=LOG))
    assert calls == ["B", "C", "B"]
    assert unresolved == []


@pytest.mark.parametrize("error", [LLMRetriesExhausted("fatal", status_code=429), asyncio.CancelledError()])
def test_fatal_errors_and_cancellation_propagate(monkeypatch, error):
    async def fail(*args):
        raise error
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", fail)
    toc = toc_at(1, 2)
    with pytest.raises(type(error)):
        asyncio.run(classic.fix_incorrect_toc_with_retries(
            toc, [("page", 1)] * 2, bad(toc, 1), logger=LOG))


@pytest.mark.parametrize("candidate", [None, 0, -1, True, 2.5, "6", 3, 8])
def test_repair_must_stay_within_confirmed_neighbors(monkeypatch, candidate):
    async def locate(*args):
        return candidate
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    toc = toc_at(4, 9, 7)
    _, unresolved = asyncio.run(classic.fix_incorrect_toc(
        toc, [("page", 1)] * 6, bad(toc, 1), start_index=4, logger=LOG))
    assert unresolved[0]["list_index"] == 1
    assert unresolved[0]["reason"]
    assert toc[1]["physical_index"] == 9


def test_repair_search_uses_inclusive_neighbor_bounds_with_offset(monkeypatch):
    seen = []
    async def locate(title, content, model):
        seen.extend(map(int, re.findall(r"<physical_index_(\d+)>", content)))
        return 4  # same page as previous confirmed entry is legal
    async def yes(*args):
        return {"answer": "yes"}
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    monkeypatch.setattr(classic, "check_title_appearance", yes)
    toc = toc_at(4, 9, 7)
    _, unresolved = asyncio.run(classic.fix_incorrect_toc(
        toc, [("page", 1)] * 6, bad(toc, 1), start_index=4, logger=LOG))
    assert set(seen) == {4, 5, 6, 7}
    assert toc[1]["physical_index"] == 4
    assert unresolved == []


def test_parallel_candidates_are_checked_against_each_other(monkeypatch):
    async def locate(title, *args):
        return {"B": 8, "C": 4}[title]
    async def yes(*args):
        return {"answer": "yes"}
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    monkeypatch.setattr(classic, "check_title_appearance", yes)
    toc = toc_at(2, None, None, 10)
    _, unresolved = asyncio.run(classic.fix_incorrect_toc(
        toc, [("page", 1)] * 10, bad(toc, 1, 2), logger=LOG))
    assert [item["list_index"] for item in unresolved] == [1, 2]
    assert all("order" in item["reason"] for item in unresolved)
    assert [item["physical_index"] for item in toc] == [2, None, None, 10]


def test_verification_does_not_drop_invalid_indices_or_failed_checks(monkeypatch):
    async def check(item, *args):
        if item["title"] == "B":
            raise TimeoutError("check failed")
        return dict(item, answer="yes")
    monkeypatch.setattr(classic, "check_title_appearance", check)
    accuracy, unresolved = asyncio.run(classic.verify_toc(
        [("page", 1)] * 10, toc_at(None, 3, 10)))
    assert accuracy == pytest.approx(1 / 3)
    assert [item["list_index"] for item in unresolved] == [0, 1]
    assert all(item["reason"] for item in unresolved)


@pytest.mark.parametrize("repair_fails", [False, True])
def test_meta_processor_routes_order_conflicts_to_repair(monkeypatch, repair_fails):
    initial = toc_at(2, 8, 4, 10)
    monkeypatch.setattr(classic, "process_no_toc", lambda *a, **kw: copy.deepcopy(initial))
    checked, repaired = [], []
    async def check(item, *args):
        checked.append(item["title"])
        return dict(item, answer="yes")
    async def locate(title, content, model):
        repaired.append(title)
        assert set(map(int, re.findall(r"<physical_index_(\d+)>", content))) == set(range(2, 11))
        if repair_fails:
            raise TimeoutError("not located")
        return {"B": 4, "C": 8}[title]
    monkeypatch.setattr(classic, "check_title_appearance", check)
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    call = classic.meta_processor([("page", 1)] * 10, mode="process_no_toc",
                                  opt=SimpleNamespace(model="fake"), logger=LOG)
    if repair_fails:
        with pytest.raises(classic.TOCRepairError) as error:
            asyncio.run(call)
        assert [item["list_index"] for item in error.value.unresolved] == [1, 2]
        assert all(item["attempts"] == 3 for item in error.value.unresolved)
        assert error.value.toc == initial
        assert repaired == ["B", "C"] * 3
    else:
        assert asyncio.run(call) == toc_at(2, 4, 8, 10)
        assert repaired == ["B", "C"]
    assert checked[:2] == ["A", "D"]


def test_printed_page_resets_do_not_reject_valid_physical_order(monkeypatch):
    toc = toc_at(2, 2, 8)
    for item, page in zip(toc, ["iv", 1, 2]):
        item["page"] = page
    async def check(item, *args):
        return dict(item, answer="yes")
    monkeypatch.setattr(classic, "check_title_appearance", check)
    accuracy, unresolved = asyncio.run(classic.verify_toc([("page", 1)] * 20, toc))
    assert accuracy == 1
    assert unresolved == []


def test_failed_original_mapping_does_not_block_a_successful_neighbor(monkeypatch):
    calls = []
    async def locate(title, *args):
        calls.append(title)
        if title == "C" and calls.count("C") == 1:
            raise TimeoutError("once")
        return {"B": 6, "C": 8}[title]
    async def yes(*args):
        return {"answer": "yes"}
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    monkeypatch.setattr(classic, "check_title_appearance", yes)
    toc = toc_at(2, 9, 4, 10)
    result, unresolved = asyncio.run(classic.fix_incorrect_toc_with_retries(
        toc, [("page", 1)] * 10, bad(toc, 1, 2), logger=LOG))
    assert result == toc_at(2, 6, 8, 10)
    assert calls == ["B", "C", "C"]
    assert unresolved == []


def test_unrecoverable_failure_cancels_other_repairs(monkeypatch):
    cancelled = []
    async def locate(title, *args):
        if title == "B":
            await asyncio.sleep(0)
            raise LLMRetriesExhausted("fatal", status_code=401)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(title)
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", locate)
    toc = toc_at(1, 2, 3)
    async def run():
        with pytest.raises(LLMRetriesExhausted):
            await asyncio.wait_for(classic.fix_incorrect_toc(
                toc, [("page", 1)] * 3, bad(toc, 1, 2)), timeout=1)
    asyncio.run(run())
    assert cancelled == ["C"]
    assert toc == toc_at(1, 2, 3)


def test_missing_model_index_retains_original_evidence(monkeypatch):
    toc = classic._validate_physical_indices(toc_at(900), 10)
    async def fail(*args):
        raise TimeoutError("not found")
    monkeypatch.setattr(classic, "single_toc_item_index_fixer", fail)
    _, incorrect = asyncio.run(classic.verify_toc([("page", 1)] * 10, toc))
    _, unresolved = asyncio.run(classic.fix_incorrect_toc_with_retries(
        toc, [("page", 1)] * 10, incorrect, max_attempts=1))
    assert unresolved[0]["physical_index"] == 900
    assert "TimeoutError" in unresolved[0]["reason"]


@pytest.mark.parametrize("attempts", [0, -1, True, 1.5])
def test_retry_budget_must_be_a_positive_integer(attempts):
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(classic.fix_incorrect_toc_with_retries([], [], [], max_attempts=attempts))
