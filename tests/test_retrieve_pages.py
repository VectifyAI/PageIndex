"""Regression tests for pageindex.retrieve page selection."""
import json

from pageindex.retrieve import get_page_content


def _docs():
    """Markdown and PDF documents with headings/pages at the same positions."""
    return {
        'D_MD': {
            'type': 'md',
            'structure': [
                {'line_num': 5, 'text': 'L5', 'nodes': []},
                {'line_num': 10, 'text': 'L10', 'nodes': []},
                {'line_num': 50, 'text': 'L50', 'nodes': []},
                {'line_num': 100, 'text': 'L100', 'nodes': []},
            ],
        },
        'D_PDF': {
            'type': 'pdf',
            'pages': [
                {'page': 5, 'content': 'P5'},
                {'page': 10, 'content': 'P10'},
                {'page': 50, 'content': 'P50'},
                {'page': 100, 'content': 'P100'},
            ],
        },
    }


def _pages(result_json):
    return sorted(r['page'] for r in json.loads(result_json))


def test_md_comma_returns_only_requested_pages():
    docs = _docs()
    assert _pages(get_page_content(docs, 'D_MD', '5,100')) == [5, 100]


def test_md_and_pdf_agree_on_comma_separated_pages():
    docs = _docs()
    md = _pages(get_page_content(docs, 'D_MD', '5,100'))
    pdf = _pages(get_page_content(docs, 'D_PDF', '5,100'))
    assert md == pdf == [5, 100]


def test_md_range_still_returns_nodes_inside_range():
    docs = _docs()
    assert _pages(get_page_content(docs, 'D_MD', '5-50')) == [5, 10, 50]


def test_md_single_page_returns_just_that_node():
    docs = _docs()
    assert _pages(get_page_content(docs, 'D_MD', '50')) == [50]
