import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.retrieve import (
    _parse_pages,
    get_document,
    get_document_structure,
    get_page_content,
)


SAMPLE_DOCS = {
    "doc-001": {
        "doc_name": "Annual Report",
        "doc_description": "2024 Annual Report for ACME Corp",
        "type": "pdf",
        "path": os.path.join(os.path.dirname(__file__), "pdfs", "earthmover.pdf"),
        "structure": [
            {
                "title": "Introduction",
                "node_id": "0001",
                "start_index": 1,
                "end_index": 3,
                "text": "Introduction content",
                "nodes": [],
            },
            {
                "title": "Financials",
                "node_id": "0002",
                "start_index": 4,
                "end_index": 8,
                "text": "Financials content",
                "nodes": [
                    {
                        "title": "Revenue",
                        "node_id": "0003",
                        "start_index": 4,
                        "end_index": 6,
                        "text": "Revenue details",
                        "nodes": [],
                    }
                ],
            },
        ],
    },
    "doc-md": {
        "doc_name": "readme",
        "doc_description": "Project readme",
        "type": "md",
        "structure": [
            {
                "title": "Overview",
                "node_id": "0001",
                "line_num": 1,
                "text": "Overview text",
                "nodes": [],
            },
            {
                "title": "Usage",
                "node_id": "0002",
                "line_num": 25,
                "text": "Usage text",
                "nodes": [],
            },
        ],
    },
}


class TestParsePages:
    def test_single_page(self):
        assert _parse_pages("5") == [5]

    def test_range(self):
        assert _parse_pages("3-6") == [3, 4, 5, 6]

    def test_comma_separated(self):
        assert _parse_pages("1,3,5") == [1, 3, 5]

    def test_mixed_format(self):
        assert _parse_pages("1-3,7,10-12") == [1, 2, 3, 7, 10, 11, 12]

    def test_deduplication(self):
        assert _parse_pages("1,1,2,2") == [1, 2]

    def test_sorted_output(self):
        assert _parse_pages("9,3,1") == [1, 3, 9]

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            _parse_pages("5-3")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _parse_pages("abc")

    def test_whitespace_handling(self):
        assert _parse_pages(" 1 , 3 - 5 ") == [1, 3, 4, 5]


class TestGetDocument:
    def test_existing_pdf_document(self):
        result = json.loads(get_document(SAMPLE_DOCS, "doc-001"))
        assert result["doc_id"] == "doc-001"
        assert result["doc_name"] == "Annual Report"
        assert result["status"] == "completed"
        assert "page_count" in result

    def test_existing_md_document(self):
        result = json.loads(get_document(SAMPLE_DOCS, "doc-md"))
        assert result["doc_id"] == "doc-md"
        assert result["type"] == "md"
        assert "line_count" in result

    def test_nonexistent_document_returns_error(self):
        result = json.loads(get_document(SAMPLE_DOCS, "nonexistent"))
        assert "error" in result


class TestGetDocumentStructure:
    def test_returns_structure_without_text(self):
        result = json.loads(get_document_structure(SAMPLE_DOCS, "doc-001"))
        assert isinstance(result, list)
        assert len(result) == 2

        def check_no_text(nodes):
            for node in nodes:
                assert "text" not in node
                if node.get("nodes"):
                    check_no_text(node["nodes"])

        check_no_text(result)

    def test_nonexistent_document(self):
        result = json.loads(get_document_structure(SAMPLE_DOCS, "nonexistent"))
        assert "error" in result


class TestGetPageContent:
    def test_invalid_pages_format(self):
        result = json.loads(get_page_content(SAMPLE_DOCS, "doc-001", "abc"))
        assert "error" in result

    def test_nonexistent_document(self):
        result = json.loads(get_page_content(SAMPLE_DOCS, "nonexistent", "1-3"))
        assert "error" in result
