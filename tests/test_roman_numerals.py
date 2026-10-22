import pytest
from pageindex.utils import roman_to_int, convert_page_to_int


@pytest.mark.parametrize(
    "roman_str,expected",
    [
        ("i", 1),
        ("ii", 2),
        ("iii", 3),
        ("iv", 4),
        ("v", 5),
        ("vi", 6),
        ("vii", 7),
        ("viii", 8),
        ("ix", 9),
        ("x", 10),
        ("xi", 11),
        ("xii", 12),
        ("xiv", 14),
        ("xv", 15),
        ("xix", 19),
        ("xx", 20),
        ("xl", 40),
        ("l", 50),
        ("xc", 90),
        ("c", 100),
        ("cd", 400),
        ("d", 500),
        ("cm", 900),
        ("m", 1000),
        ("MCMLIV", 1954),
        ("mmxxvi", 2026),
        # Mixed casing and whitespace
        ("  IV  ", 4),
        ("Xii", 12),
        ("vIi", 7),
    ],
)
def test_roman_to_int_valid(roman_str, expected):
    assert roman_to_int(roman_str) == expected


@pytest.mark.parametrize(
    "invalid_str",
    [
        "",
        "   ",
        "abc",
        "123",
        "iv2",
        "iiii",  # invalid roman syntax
        "vx",    # invalid subtraction
        "ll",    # invalid repeated 50
        None,
        123,
        [],
    ],
)
def test_roman_to_int_invalid(invalid_str):
    assert roman_to_int(invalid_str) is None


def test_convert_page_to_int_handles_roman_and_arabic_pages():
    toc = [
        {"title": "Title Page", "page": "i"},
        {"title": "Dedication", "page": "ii"},
        {"title": "Table of Contents", "page": " iv "},
        {"title": "Preface", "page": "VII"},
        {"title": "Introduction", "page": "1"},
        {"title": "Chapter 1", "page": 5},
        {"title": "Chapter 2", "page": " 12 "},
        {"title": "Appendix", "page": None},
        {"title": "Unnumbered Note", "page": "not_a_page"},
    ]

    converted = convert_page_to_int(toc)

    assert converted[0]["page"] == 1
    assert converted[1]["page"] == 2
    assert converted[2]["page"] == 4
    assert converted[3]["page"] == 7
    assert converted[4]["page"] == 1
    assert converted[5]["page"] == 5
    assert converted[6]["page"] == 12
    assert converted[7]["page"] is None
    assert converted[8]["page"] == "not_a_page"
