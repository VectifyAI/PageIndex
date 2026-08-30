from pageindex.library.splitters import (apply_diary_profile, build_diary_structure,
                                         find_dated_entries)


def diary_pages():
    """Page 1 foreword, pages 2-6 diary, page 7 glossary."""
    return [
        "Foreword\nSome words.",
        "1 January 1981 – Thursday\nWent to Madras.\n3 January 1981 – Saturday\nRain.",
        "More about the rain.\n12 March 1981 – Thursday\nSulochana is unwell.",
        "2 April 1981 – Thursday\nTravel.\n5 January 1982 – Tuesday\nNew year.",
        "Continued entry.",
        "20 February 1982 – Saturday\nSatsangh.",
        "Glossary\nabhyasi: practitioner",
    ]


def test_find_dated_entries_with_offset():
    entries = find_dated_entries(diary_pages()[1:3], first_page=2)
    assert [e["title"] for e in entries] == ["1 January 1981", "3 January 1981", "12 March 1981"]
    assert [e["page"] for e in entries] == [2, 2, 3]
    assert entries[2]["year"] == 1981 and entries[2]["month"] == 3


def test_find_dated_entries_ignores_inline_dates():
    assert find_dated_entries(["On 3 January 1981 – Saturday we met."]) == []


def test_build_diary_structure_years_and_months():
    entries = find_dated_entries(diary_pages())
    tree = build_diary_structure(entries, end_page=6)
    assert [n["title"] for n in tree] == ["1981", "1982"]
    y81 = tree[0]
    assert (y81["start_index"], y81["end_index"]) == (2, 4)
    assert [m["title"] for m in y81["nodes"]] == ["January 1981", "March 1981", "April 1981"]
    jan = y81["nodes"][0]
    assert (jan["start_index"], jan["end_index"]) == (2, 3)
    assert jan["key_items"] == ["1 January 1981", "3 January 1981"]
    assert jan["nodes"] == []
    y82 = tree[1]
    assert (y82["start_index"], y82["end_index"]) == (4, 6)
    assert [m["title"] for m in y82["nodes"]] == ["January 1982", "February 1982"]
    assert (y82["nodes"][1]["start_index"], y82["nodes"][1]["end_index"]) == (6, 6)


def test_apply_diary_profile_keeps_front_and_back_matter():
    flash_tree = [
        {"title": "Foreword", "start_index": 1, "end_index": 6,
         "nodes": [{"title": "8.15 a.m. junk", "start_index": 3, "end_index": 6, "nodes": []}]},
        {"title": "Glossary", "start_index": 7, "end_index": 7, "nodes": []},
    ]
    out = apply_diary_profile(flash_tree, diary_pages())
    assert [n["title"] for n in out] == ["Foreword", "1981", "1982", "Glossary"]
    assert (out[0]["start_index"], out[0]["end_index"]) == (1, 1)
    assert out[0]["nodes"] == []
    assert (out[3]["start_index"], out[3]["end_index"]) == (7, 7)


def test_apply_diary_profile_without_entries_returns_input_unchanged():
    tree = [{"title": "A", "start_index": 1, "end_index": 2, "nodes": []}]
    assert apply_diary_profile(tree, ["no dates", "here"]) == tree
