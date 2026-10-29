import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.page_index_classic import (
    add_page_offset_to_toc_json,
    calculate_page_offset,
)


class TestIssue153NoneOffset:
    """Regression tests for issue #153.

    calculate_page_offset() returns None when no reliable page->physical
    offset can be computed (e.g. no title matches). add_page_offset_to_toc_json
    must not crash with `TypeError: unsupported operand type(s) for +:
    'int' and 'NoneType'` in that case.
    """

    def test_calculate_page_offset_empty_pairs_returns_none(self):
        assert calculate_page_offset([]) is None

    def test_calculate_page_offset_all_invalid_pairs_returns_none(self):
        # every pair raises inside the try -> differences stays empty
        pairs = [{"physical_index": "not-a-number", "page": "also-not-a-number"}]
        assert calculate_page_offset(pairs) is None

    def test_add_offset_none_leaves_items_untouched(self):
        data = [
            {"title": "Introduction", "page": 5},
            {"title": "Methods", "page": 12},
        ]
        result = add_page_offset_to_toc_json(data, None)
        assert result == [
            {"title": "Introduction", "page": 5},
            {"title": "Methods", "page": 12},
        ]

    def test_add_offset_none_does_not_raise(self):
        # exact repro from the issue: int page + None offset
        data = [{"title": "Introduction", "page": 5}]
        add_page_offset_to_toc_json(data, calculate_page_offset([]))

    def test_add_offset_still_applies_when_present(self):
        data = [
            {"title": "Introduction", "page": 5},
            {"title": "NoPageNumber"},
        ]
        result = add_page_offset_to_toc_json(data, 3)
        assert result[0] == {"title": "Introduction", "physical_index": 8}
        assert "page" not in result[0]
        assert result[1] == {"title": "NoPageNumber"}
