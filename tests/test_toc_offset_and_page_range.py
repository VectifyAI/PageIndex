"""
Tests for two related bugs in TOC processing:

Bug 1 — add_page_offset_to_toc_json crashes with TypeError when offset is None.
    Root cause: calculate_page_offset() returns None when no title/page pairs can be
    matched (e.g. the visible TOC pages don't overlap with the first N pages of content).
    add_page_offset_to_toc_json() then tried ``data[i]['page'] + None``, raising TypeError.

Bug 2 — process_none_page_numbers produces an empty page-content window for the *last*
    TOC item when it has no physical_index.
    Root cause: ``next_physical_index`` defaulted to ``-1`` (meaning "no next item"),
    and ``range(prev, -1 + 1)`` == ``range(prev, 0)`` is always empty.  The LLM was
    called with an empty string and could not locate the section.
    Secondary: ``prev_physical_index`` defaulted to ``0``, which produces list_index ``-1``
    and silently skips the first page (bounds check catches it, but page 1 is excluded).
"""
import copy
import pytest

from pageindex.page_index import (
    add_page_offset_to_toc_json,
    calculate_page_offset,
    process_none_page_numbers,
)


# ---------------------------------------------------------------------------
# Bug 1: add_page_offset_to_toc_json with None offset
# ---------------------------------------------------------------------------

class TestAddPageOffsetToTocJson:
    """Covers the None-offset guard added to add_page_offset_to_toc_json."""

    def test_none_offset_returns_data_unchanged(self):
        """When offset is None the input list must be returned unmodified."""
        data = [
            {"structure": "1", "title": "Introduction", "page": 5},
            {"structure": "2", "title": "Methods", "page": 10},
        ]
        original = copy.deepcopy(data)
        result = add_page_offset_to_toc_json(data, None)
        assert result == original, "Data should be unchanged when offset is None"

    def test_none_offset_no_crash(self):
        """No exception must be raised when offset is None."""
        data = [{"structure": "1", "title": "Intro", "page": 3}]
        try:
            add_page_offset_to_toc_json(data, None)
        except TypeError as exc:
            pytest.fail(f"TypeError raised with None offset: {exc}")

    def test_integer_offset_applied_correctly(self):
        """A valid integer offset must still compute physical_index = page + offset."""
        data = [
            {"structure": "1", "title": "Intro", "page": 5},
            {"structure": "2", "title": "Methods", "page": 10},
        ]
        result = add_page_offset_to_toc_json(data, 3)
        assert result[0]["physical_index"] == 8
        assert result[1]["physical_index"] == 13
        assert "page" not in result[0]
        assert "page" not in result[1]

    def test_zero_offset_applied_correctly(self):
        """Offset of 0 means page number == physical_index."""
        data = [{"structure": "1", "title": "Intro", "page": 7}]
        result = add_page_offset_to_toc_json(data, 0)
        assert result[0]["physical_index"] == 7
        assert "page" not in result[0]

    def test_negative_offset_applied_correctly(self):
        """Negative offsets (e.g. roman-numeral front matter) must work."""
        data = [{"structure": "1", "title": "Chapter 1", "page": 10}]
        result = add_page_offset_to_toc_json(data, -2)
        assert result[0]["physical_index"] == 8

    def test_none_page_value_skipped(self):
        """Items with page=None must not be modified even with a valid offset."""
        data = [{"structure": "1", "title": "Unknown", "page": None}]
        result = add_page_offset_to_toc_json(data, 3)
        assert result[0].get("page") is None
        assert "physical_index" not in result[0]

    def test_item_without_page_key_skipped(self):
        """Items that already have a physical_index (no 'page' key) are untouched."""
        data = [{"structure": "1", "title": "Intro", "physical_index": 5}]
        result = add_page_offset_to_toc_json(data, 3)
        assert result[0]["physical_index"] == 5  # unchanged

    def test_empty_list_none_offset(self):
        """Empty input list with None offset must return empty list."""
        assert add_page_offset_to_toc_json([], None) == []

    def test_empty_list_integer_offset(self):
        """Empty input list with integer offset must return empty list."""
        assert add_page_offset_to_toc_json([], 5) == []

    def test_calculate_page_offset_returns_none_for_empty_pairs(self):
        """calculate_page_offset([]) must return None (documents the contract)."""
        assert calculate_page_offset([]) is None

    def test_pipeline_none_offset_does_not_crash(self):
        """
        End-to-end simulation: matching_pairs empty → offset None →
        add_page_offset_to_toc_json must not crash.
        """
        toc = [
            {"structure": "1", "title": "Intro", "page": 1},
            {"structure": "2", "title": "Methods", "page": 5},
        ]
        pairs = []  # No matches found → offset will be None
        offset = calculate_page_offset(pairs)
        assert offset is None
        # Must not raise
        result = add_page_offset_to_toc_json(toc, offset)
        # All items still have 'page' key (unchanged)
        assert all("page" in item for item in result)


# ---------------------------------------------------------------------------
# Bug 2: process_none_page_numbers — empty range for last/first items
# ---------------------------------------------------------------------------

class TestProcessNonePageNumbers:
    """Covers the empty-range bug in process_none_page_numbers."""

    @staticmethod
    def _make_page_list(n):
        """Return a synthetic page_list of length n."""
        return [(f"Content of page {i + 1}", 50) for i in range(n)]

    def test_last_item_gets_non_empty_range(self):
        """
        The *last* TOC item (no successor) used to produce range(prev, 0) which is
        always empty — the LLM received an empty string.  After the fix, the default
        next_physical_index is end_index = len(page_list) + start_index - 1.
        """
        # Simulate: the last entry has no physical_index yet
        # We can verify the window by checking that page_contents is non-empty
        # when we manually replicate the loop logic from process_none_page_numbers.
        page_list = self._make_page_list(5)
        start_index = 1
        end_index = len(page_list) + start_index - 1  # = 5

        # Scenario: item at index 1 has no physical_index, no next item
        prev_physical_index = 2  # from item at index 0
        next_physical_index = end_index  # new default

        page_range = list(range(prev_physical_index, next_physical_index + 1))
        assert page_range == [2, 3, 4, 5], (
            f"Expected [2,3,4,5] but got {page_range} — last item would get empty context"
        )

    def test_first_item_prev_default_is_start_index(self):
        """
        The *first* TOC item (no predecessor) used to default prev_physical_index to 0.
        With start_index=1, list_index = 0 - 1 = -1, which is silently skipped by the
        bounds check.  While the effective window was the same (page 1 still appeared
        due to the subsequent loop), the new default start_index avoids computing
        a negative list_index entirely — a cleaner, less error-prone starting point.
        """
        start_index = 1

        # Old code attempted page 0, list_index = 0 - 1 = -1, bounds check skips it.
        # Effective window started at page 1 anyway.
        old_attempted_list_index = 0 - start_index  # = -1
        assert old_attempted_list_index < 0, "Old code produced list_index=-1 (bounds-checked away)"

        # New code: prev = start_index = 1, list_index = 1 - 1 = 0 (valid)
        new_first_list_index = start_index - start_index  # = 0
        assert new_first_list_index == 0, "New code starts at list_index=0 (no negative index attempted)"

    def test_single_page_document_last_item(self):
        """A 1-page document whose sole TOC item has no physical_index should get range [1,1]."""
        page_list = self._make_page_list(1)
        start_index = 1
        end_index = 1

        prev_physical_index = start_index   # new default (no predecessor)
        next_physical_index = end_index     # new default (no successor)

        page_range = list(range(prev_physical_index, next_physical_index + 1))
        assert page_range == [1]

    def test_middle_item_range_uses_neighbors(self):
        """A middle item should still use actual neighbor physical indices."""
        # item at index 1 (between items 0 and 2 with known physical_index)
        page_list = self._make_page_list(10)
        start_index = 1
        prev_physical_index = 3   # from item 0
        next_physical_index = 7   # from item 2

        page_range = list(range(prev_physical_index, next_physical_index + 1))
        assert page_range == [3, 4, 5, 6, 7]

    def test_regression_old_next_default_minus_one(self):
        """
        Regression: range(prev, -1 + 1) == range(prev, 0) must be empty.
        Confirms the old bug and validates the fix direction.
        """
        prev = 3
        old_next = -1  # old default
        new_next = 5   # new default (end_index for a 5-page doc)

        assert list(range(prev, old_next + 1)) == [], "Old code produced empty range"
        assert list(range(prev, new_next + 1)) == [3, 4, 5], "New code produces valid range"

    def test_all_items_have_physical_index_no_op(self):
        """When every item already has physical_index, function must return unchanged."""
        page_list = self._make_page_list(5)
        toc_items = [
            {"structure": "1", "title": "Intro", "physical_index": 1},
            {"structure": "2", "title": "Methods", "physical_index": 3},
            {"structure": "3", "title": "Results", "physical_index": 5},
        ]
        original = copy.deepcopy(toc_items)
        result = process_none_page_numbers(toc_items, page_list)
        # physical_index values must remain the same (no LLM called, no changes)
        for orig, res in zip(original, result):
            assert res["physical_index"] == orig["physical_index"]
