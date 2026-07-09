from pageindex.types import DocumentDetail, DocumentInfo, PageContent


def test_document_detail_structure_field_is_required():
    """structure is always populated by both LocalBackend.get_document and
    CloudBackend.get_document — must be a required key, not optional, or
    type checkers/tooling built on this TypedDict wrongly treat a
    DocumentDetail missing 'structure' as valid."""
    assert "structure" in DocumentDetail.__required_keys__
    assert "structure" not in DocumentDetail.__optional_keys__


def test_document_detail_backend_specific_fields_stay_optional():
    assert "file_path" in DocumentDetail.__optional_keys__
    assert "status" in DocumentDetail.__optional_keys__


def test_document_detail_inherits_document_info_as_required():
    for key in ("doc_id", "doc_name", "doc_description", "doc_type"):
        assert key in DocumentDetail.__required_keys__
