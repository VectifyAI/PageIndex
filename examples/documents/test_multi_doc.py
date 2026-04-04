import json
import pytest
from unittest.mock import patch, MagicMock
from pageindex.retrieve import get_document, get_document_structure, get_page_content

@pytest.fixture
def mock_documents():
    return {
        "doc1": {
            "id": "doc1",
            "path": "test1.pdf",
            "type": "pdf",
            "doc_name": "Document 1",
            "doc_description": "First test doc",
            "structure": [{"title": "Section 1", "page": 1, "text": "Content 1"}]
        },
        "doc2": {
            "id": "doc2",
            "path": "test2.md",
            "type": "md",
            "doc_name": "Document 2",
            "doc_description": "Second test doc",
            "structure": [{"title": "Header 2", "line_num": 1, "text": "Content 2"}]
        }
    }

def test_get_document_multi(mock_documents):
    with patch("pageindex.retrieve._count_pages", return_value=5):
        result_json = get_document(mock_documents, ["doc1", "doc2"])
        result = json.loads(result_json)
        
        assert "doc1" in result
        assert "doc2" in result
        assert result["doc1"]["doc_name"] == "Document 1"
        assert result["doc1"]["page_count"] == 5
        assert result["doc2"]["line_count"] == 5

def test_get_document_structure_multi(mock_documents):
    result_json = get_document_structure(mock_documents, ["doc1", "doc2"])
    result = json.loads(result_json)
    
    assert "doc1" in result
    assert "doc2" in result
    # Verify text field is removed
    assert "text" not in result["doc1"][0]
    assert result["doc1"][0]["title"] == "Section 1"

def test_get_page_content_multi(mock_documents):
    with patch("pageindex.retrieve._get_pdf_page_content", return_value=[{"page": 1, "content": "PDF Content"}]), \
         patch("pageindex.retrieve._get_md_page_content", return_value=[{"page": 1, "content": "MD Content"}]):
        
        result_json = get_page_content(mock_documents, ["doc1", "doc2"], "1")
        result = json.loads(result_json)
        
        assert "doc1" in result
        assert "doc2" in result
        assert result["doc1"][0]["content"] == "PDF Content"
        assert result["doc2"][0]["content"] == "MD Content"

def test_get_document_multi_with_invalid_id(mock_documents):
    with patch("pageindex.retrieve._count_pages", return_value=5):
        result_json = get_document(mock_documents, ["doc1", "invalid-id"])
        result = json.loads(result_json)
        
        assert "doc1" in result
        assert "invalid-id" in result
        assert "error" in result["invalid-id"]
        assert "not found" in result["invalid-id"]["error"]

def test_backward_compatibility(mock_documents):
    # Single doc_id as string should return flat result, not nested
    with patch("pageindex.retrieve._count_pages", return_value=5):
        result_json = get_document(mock_documents, "doc1")
        result = json.loads(result_json)
        
        assert "doc_id" in result
        assert result["doc_id"] == "doc1"
        assert "doc1" not in result # Should not be nested
