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
