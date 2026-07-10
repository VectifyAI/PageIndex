"""Markdown page-content selection must return exactly the requested lines,
mirroring the PDF path — not the whole [min, max] range (PR #272 review / #280)."""


def _md_structure():
    # line_num 40 sits *between* 5 and 100 but is NOT requested below.
    return [
        {"line_num": 5, "text": "line five", "nodes": [
            {"line_num": 40, "text": "line forty (should be excluded)", "nodes": []},
        ]},
        {"line_num": 100, "text": "line hundred", "nodes": []},
        {"line_num": 101, "text": "line 101", "nodes": []},
    ]


def test_get_md_page_content_returns_only_requested_lines():
    from pageindex.index.utils import get_md_page_content

    out = get_md_page_content(_md_structure(), [5, 100])
    # exactly the two requested lines — not 5, 40, 100 (the old range behavior)
    assert [r["page"] for r in out] == [5, 100]
    assert all("forty" not in r["content"] for r in out)


def test_get_md_page_content_empty_spec():
    from pageindex.index.utils import get_md_page_content

    assert get_md_page_content(_md_structure(), []) == []


def test_retrieve_md_page_content_returns_only_requested_lines():
    # The legacy retrieve path has its own copy of the same logic.
    from pageindex.retrieve import _get_md_page_content

    out = _get_md_page_content({"structure": _md_structure()}, [5, 100])
    assert [r["page"] for r in out] == [5, 100]


def test_retrieve_parse_pages_delegates_to_canonical_and_enforces_dos_cap():
    """retrieve._parse_pages used to be an independent copy that lacked the
    canonical parse_pages' p>=1 filter and 1000-page cap — a caller of the
    legacy pageindex.get_page_content could bypass the DoS guard the SDK path
    enforces. Now it's a one-line delegate, so they can't drift again."""
    from pageindex.retrieve import _parse_pages
    from pageindex.index.utils import parse_pages
    import pytest

    assert _parse_pages("5-7") == parse_pages("5-7") == [5, 6, 7]
    with pytest.raises(ValueError, match="too large"):
        _parse_pages("1-99999999")


def test_parse_pages_caps_a_huge_range_without_materializing_it():
    """Regression: the 1000-page cap was checked only AFTER
    `result.extend(range(start, end + 1))`, so a single huge span like
    '1-2000000000' allocated billions of ints and OOM'd before the check ran.
    The span must be rejected up front, quickly, without building the list."""
    import time
    import pytest
    from pageindex.index.utils import parse_pages

    start = time.monotonic()
    with pytest.raises(ValueError, match="too large"):
        parse_pages("1-2000000000")
    # Must be near-instant (no billion-element allocation). Generous bound to
    # avoid flakiness while still failing loudly on a re-materializing regression.
    assert time.monotonic() - start < 1.0

    # Boundary: exactly 1000 pages is allowed; 1001 is rejected.
    assert parse_pages("1-1000") == list(range(1, 1001))
    with pytest.raises(ValueError, match="too large"):
        parse_pages("1-1001")
    # A range that fits but whose accumulation across parts crosses the cap.
    with pytest.raises(ValueError, match="too large"):
        parse_pages("1-600,700-1400")


def test_retrieve_get_pdf_page_content_falls_back_to_canonical(tmp_path, monkeypatch):
    """When no cached 'pages' are present, the file-read fallback must
    delegate to the canonical get_pdf_page_content instead of re-implementing
    PDF text extraction inline (a second, independently-maintained copy)."""
    from pageindex.retrieve import _get_pdf_page_content
    import pageindex.retrieve as retrieve_mod

    calls = []
    monkeypatch.setattr(
        retrieve_mod, "get_pdf_page_content",
        lambda path, page_nums: calls.append((path, page_nums)) or [{"page": 1, "content": "x"}],
    )
    result = _get_pdf_page_content({"path": "/fake/doc.pdf"}, [1])
    assert calls == [("/fake/doc.pdf", [1])]
    assert result == [{"page": 1, "content": "x"}]


def test_retrieve_get_pdf_page_content_prefers_cache_over_file():
    from pageindex.retrieve import _get_pdf_page_content

    doc_info = {"path": "/should/not/be/opened.pdf",
                "pages": [{"page": 1, "content": "cached one"}, {"page": 2, "content": "cached two"}]}
    result = _get_pdf_page_content(doc_info, [2])
    assert result == [{"page": 2, "content": "cached two"}]
