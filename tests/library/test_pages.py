import pytest

from pageindex.library.pages import leaf_spans, page_text, parse_page_spec


def test_page_text_is_one_based_inclusive(sample_pages):
    assert page_text(sample_pages, 2, 3) == "Page 2 text word2.\n\nPage 3 text word3."
    assert page_text(sample_pages, 6, 6) == "Page 6 text word6."


def test_parse_page_spec_forms():
    assert parse_page_spec("5", 10) == [5]
    assert parse_page_spec("3,7,10", 10) == [3, 7, 10]
    assert parse_page_spec("5-7", 10) == [5, 6, 7]
    assert parse_page_spec(" 2 , 4-5 ", 10) == [2, 4, 5]


def test_parse_page_spec_rejects_out_of_range_and_garbage():
    with pytest.raises(ValueError):
        parse_page_spec("0", 10)
    with pytest.raises(ValueError):
        parse_page_spec("11", 10)
    with pytest.raises(ValueError):
        parse_page_spec("a-b", 10)


def test_leaf_spans(sample_tree):
    assert leaf_spans(sample_tree) == [("Section A", 2, 3), ("Section B", 4, 4),
                                       ("Chapter Two", 5, 6)]
