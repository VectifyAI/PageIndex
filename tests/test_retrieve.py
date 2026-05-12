import json

import pytest

from pageindex import retrieve


def test_parse_pages_sorts_deduplicates_and_rejects_bad_range():
    assert retrieve._parse_pages("3, 1-2,2") == [1, 2, 3]

    with pytest.raises(ValueError, match="start must be <= end"):
        retrieve._parse_pages("5-3")


def test_get_document_metadata_for_pdf_md_and_missing(monkeypatch):
    docs = {
        "pdf": {"type": "pdf", "doc_name": "p", "doc_description": "d", "page_count": 4},
        "md": {"type": "md", "doc_name": "m", "line_count": 7},
        "fallback": {"type": "pdf", "path": "doc.pdf"},
    }
    monkeypatch.setattr(retrieve, "get_number_of_pages", lambda path: 9)

    assert json.loads(retrieve.get_document(docs, "missing")) == {"error": "Document missing not found"}
    assert json.loads(retrieve.get_document(docs, "pdf"))["page_count"] == 4
    assert json.loads(retrieve.get_document(docs, "md"))["line_count"] == 7
    assert json.loads(retrieve.get_document(docs, "fallback"))["page_count"] == 9
    assert retrieve._count_pages({"pages": [{"page": 1}, {"page": 2}]}) == 2


def test_get_document_structure_removes_nested_text():
    docs = {"d": {"structure": [{"title": "A", "text": "hide", "nodes": [{"text": "hide"}]}]}}

    assert json.loads(retrieve.get_document_structure(docs, "missing")) == {
        "error": "Document missing not found"
    }
    assert json.loads(retrieve.get_document_structure(docs, "d")) == [
        {"title": "A", "nodes": [{}]}
    ]


def test_get_page_content_uses_cached_pdf_pages_and_handles_errors():
    docs = {
        "pdf": {
            "type": "pdf",
            "pages": [{"page": 2, "content": "two"}, {"page": 3, "content": "three"}],
        }
    }

    assert json.loads(retrieve.get_page_content(docs, "missing", "1")) == {
        "error": "Document missing not found"
    }
    assert "Invalid pages format" in json.loads(retrieve.get_page_content(docs, "pdf", "x"))["error"]
    assert json.loads(retrieve.get_page_content(docs, "pdf", "1-3")) == [
        {"page": 2, "content": "two"},
        {"page": 3, "content": "three"},
    ]


def test_get_page_content_for_markdown_structure():
    docs = {
        "md": {
            "type": "md",
            "structure": [
                {
                    "line_num": 2,
                    "text": "intro",
                    "nodes": [{"line_num": 4, "text": "child"}],
                },
                {"line_num": 8, "text": "later"},
            ],
        }
    }

    assert json.loads(retrieve.get_page_content(docs, "md", "1-5")) == [
        {"page": 2, "content": "intro"},
        {"page": 4, "content": "child"},
    ]


def test_get_pdf_page_content_reads_file_when_no_cache(monkeypatch, tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"pdf")

    class Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class Reader:
        def __init__(self, _file):
            self.pages = [Page("one"), Page(None)]

    monkeypatch.setattr(retrieve.PyPDF2, "PdfReader", Reader)

    assert retrieve._get_pdf_page_content({"path": str(path)}, [1, 2, 9]) == [
        {"page": 1, "content": "one"},
        {"page": 2, "content": ""},
    ]


def test_get_page_content_reports_reader_failure(monkeypatch):
    monkeypatch.setattr(retrieve, "_get_pdf_page_content", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    error = json.loads(retrieve.get_page_content({"d": {"type": "pdf"}}, "d", "1"))["error"]
    assert "Failed to read page content: boom" == error
