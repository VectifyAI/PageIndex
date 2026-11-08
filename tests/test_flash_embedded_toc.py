from io import BytesIO

import pytest

from pageindex.flash.embedded_toc import (
    _heading_line_reader, _readings, merge_bookmark_tree, read_bookmarks, read_printed_toc)


def _node(title, start, children=(), style=None, y=None):
    node = {"title": title, "node_id": "", "start_index": start,
            "end_index": start, "nodes": [dict(c) for c in children]}
    if style is not None:
        node["_style"] = style
    if y is not None:
        node["_y"] = float(y)
    return node


def _titles(nodes, out=None):
    if out is None:
        out = []
    for n in nodes:
        out.append(n["title"])
        _titles(n.get("nodes", []), out)
    return out


def _children(nodes, title):
    for n in nodes:
        if n["title"] == title:
            return [c["title"] for c in n.get("nodes", [])]
        found = _children(n.get("nodes", []), title)
        if found is not None:
            return found
    return None


def test_numbering_prefixed_duplicates_dissolve_into_frame():
    entries = [
        {"title": "Experiments", "level": 1, "page": 8},
        {"title": "Scaling Laws", "level": 2, "page": 8},
        {"title": "Main Results", "level": 2, "page": 10},
    ]
    detected = [_node("5 Experiments", 8,
                      [_node("5.1 Scaling Laws", 8)])]
    merged = merge_bookmark_tree(detected, entries, n_pages=12)
    assert _titles(merged) == ["Experiments", "Scaling Laws", "Main Results"]


def test_wordy_overlap_of_distinct_section_survives():
    entries = [{"title": "Federal Reserve Banks and Branches", "level": 1, "page": 3}]
    detected = [_node("Federal Reserve Banks and Branches", 3,
                      [_node("Reserve Bank and Branch Directors", 3,
                             [_node("District 1", 4)])])]
    merged = merge_bookmark_tree(detected, entries, n_pages=6)
    assert _titles(merged) == ["Federal Reserve Banks and Branches",
                               "Reserve Bank and Branch Directors",
                               "District 1"]
    chapter = merged[0]
    assert _titles(chapter["nodes"]) == ["Reserve Bank and Branch Directors",
                                         "District 1"]


def test_printed_heading_with_extra_text_dissolves_once():
    entries = [
        {"title": "Consolidated balance sheets", "level": 1, "page": 3},
        {"title": "Revenue", "level": 1, "page": 4},
        {"title": "Note 18 - Leases", "level": 1, "page": 5},
        {"title": "Fair value", "level": 1, "page": 6},
        {"title": "Results", "level": 1, "page": 7},
        {"title": "Derivatives netting and collateral", "level": 1, "page": 8},
        {"title": "Note 26 - Restrictions on cash", "level": 1, "page": 9},
        {"title": "2. Detailed Terms", "level": 1, "page": 10},
        {"title": "Item 4. Mine Safety Disclosures.", "level": 1, "page": 11},
        {"title": "3、项目进度", "level": 1, "page": 12},
        {"title": "Note 6 - Leases", "level": 1, "page": 13},
        {"title": "4.1 DOT", "level": 1, "page": 14},
        {"title": "第十二条 不可抗力 Article 12 Force Majeure", "level": 1, "page": 15},
        {"title": "2.1 Short title", "level": 1, "page": 16},
        {"title": "2. Do the Right Thing for consumers: The law", "level": 1,
         "page": 17},
        {"title": "Line of business equity", "level": 1, "page": 18},
        {"title": "Setting a PIN", "level": 1, "page": 19},
        {"title": "Appendix Proofs", "level": 1, "page": 20},
        {"title": "二、邻里公约 2. Neighborhood Convention", "level": 1, "page": 21},
        {"title": "End", "level": 1, "page": 22},
    ]
    detected = [_node("JPMorgan Chase & Co. Consolidated balance sheets "
                      "(unaudited)", 3),
                _node("Revenue (dollars in millions)", 4),
                _node("Note 18 - Leases Firm as lessee", 5),
                _node("Fair value", 6),
                _node("Assets measured at fair value", 6),
                _node("5.2 Ablation Results", 7),
                _node("Derivatives netting", 8),
                _node("Note 26 – Restricted cash and other assets", 9),
                _node("2. Defined Terms", 10),
                _node("Item 4. Mine Safety Disclosures. Not applicable.", 11),
                _node("3 金迈斯", 12),
                _node("Note 6 - Leases 6.1 Rent.", 13),
                _node("4 System Overview 4.1 DOT", 14),
                _node("第十二条 不可抗力 第 十 二 条 不 可 抗 力 A A rticle 12 "
                      "Force Majeure", 15),
                _node("2.1 Reasonably long section title", 16),
                _node("DO THE RIGHT THING FOR CONSUMERS", 17),
                _node("Line of business and Corporate equity", 18),
                _node("Setting a PIN 1 Set a PIN.", 19),
                _node("A. Proofs", 20),
                _node("二、 邻里公 约 二 、 邻 里 公 约 2. Neighborhood Convention "
                      "2 . N e i g h b o r h o o d C o n v e n t i o n", 21)]
    merged = merge_bookmark_tree(detected, entries, n_pages=23)
    assert _titles(merged) == [
        "Consolidated balance sheets", "Revenue", "Note 18 - Leases",
        "Note 18 - Leases Firm as lessee", "Fair value",
        "Assets measured at fair value", "Results", "5.2 Ablation Results",
        "Derivatives netting and collateral", "Note 26 - Restrictions on cash",
        "2. Detailed Terms", "Item 4. Mine Safety Disclosures.", "3、项目进度",
        "3 金迈斯", "Note 6 - Leases", "Note 6 - Leases 6.1 Rent.", "4.1 DOT",
        "第十二条 不可抗力 Article 12 Force Majeure", "2.1 Short title",
        "2. Do the Right Thing for consumers: The law",
        "Line of business equity", "Setting a PIN", "Setting a PIN 1 Set a PIN.",
        "Appendix Proofs", "二、邻里公约 2. Neighborhood Convention", "End"]


def test_printed_reference_or_continuation_dissolves():
    contents_line = ("7.4 How we assess financeability 136 "
                     "7.5 Setting a fairer price control 140")
    entries = [
        {"title": "Missing Proofs in Section Enumeration", "level": 1, "page": 2},
        {"title": "WordNet: tab:wordnet-results Visualizations", "level": 1,
         "page": 3},
        {"title": "Financial Performance Summary", "level": 1, "page": 4},
        {"title": "1. General Information (3.2.S.1)", "level": 1, "page": 5},
        {"title": "How we'll finance the plan", "level": 1, "page": 6},
        {"title": "End", "level": 1, "page": 7},
    ]
    detected = [_node("D Missing Proofs in section 4", 2),
                _node("E. WordNet: Table 1 Visualizations", 3),
                _node("Financial Performance Summary for Fiscal 2015", 4),
                _node("1. General Information (3.2.S.1) a. Nomenclature "
                      "(3.2.S.1.1)", 5),
                _node(contents_line, 6)]
    merged = merge_bookmark_tree(detected, entries, n_pages=8)
    assert _titles(merged) == [
        "Missing Proofs in Section Enumeration",
        "WordNet: tab:wordnet-results Visualizations",
        "Financial Performance Summary", "1. General Information (3.2.S.1)",
        "1. General Information (3.2.S.1) a. Nomenclature (3.2.S.1.1)",
        "How we'll finance the plan", contents_line, "End"]


def test_letter_spaced_duplicate_dissolves():
    entries = [
        {"title": "3 Methodology", "level": 1, "page": 11},
        {"title": "Hard Heads and Soft Hearts", "level": 1, "page": 12},
        {"title": "Part I", "level": 1, "page": 13},
        {"title": "Conclusion", "level": 1, "page": 14},
        {"title": "References", "level": 1, "page": 15},
    ]
    detected = [_node("M e tho do logy", 11),
                _node("Har d hea d an d soft heart s", 12),
                _node("Part II", 13), _node("R E F E R E N C E S", 15)]
    merged = merge_bookmark_tree(detected, entries, n_pages=16)
    assert _titles(merged) == ["3 Methodology", "Hard Heads and Soft Hearts",
                               "Part I", "Part II", "Conclusion", "References"]


def test_similar_title_under_another_number_survives():
    entries = [
        {"title": "13.6 Completion and Exit", "level": 1, "page": 114},
        {"title": "14 Settlement", "level": 1, "page": 116},
    ]
    detected = [_node("13.6.1 Completion and Exit", 114)]
    merged = merge_bookmark_tree(detected, entries, n_pages=118)
    assert _titles(merged[0]["nodes"]) == ["13.6.1 Completion and Exit"]


def test_garbled_duplicate_dissolves():
    entries = [
        {"title": "Applications of PCA", "level": 1, "page": 2},
        {"title": "Kernel PCA", "level": 1, "page": 4},
    ]
    detected = [_node("Applications of peA", 2)]
    merged = merge_bookmark_tree(detected, entries, n_pages=6)
    assert _titles(merged) == ["Applications of PCA", "Kernel PCA"]


def test_same_style_tail_section_climbs():
    entries = [
        {"title": "Related Work", "level": 1, "page": 16},
        {"title": "Contributions", "level": 1, "page": 20},
    ]
    detected = [_node("7 Related Work", 16, style="F1", y=0.08,
                      children=[_node("Conclusion", 16, style="F1", y=0.55),
                                _node("References", 17, style="F1", y=0.10)])]
    merged = merge_bookmark_tree(detected, entries, n_pages=22)
    assert _titles(merged) == ["Related Work", "Conclusion", "References",
                               "Contributions"]
    assert all(not n.get("nodes") for n in merged)


def test_empty_section_claim_blocks_climb():
    entries = [
        {"title": "Bayesian probabilities", "level": 1, "page": 41},
        {"title": "The Gaussian distribution", "level": 1, "page": 44},
    ]
    detected = [_node("Bayesian probabilities", 41, style="F1", y=0.10,
                      children=[_node("Thomas Bayes", 41, style="F1",
                                      y=0.14)])]
    merged = merge_bookmark_tree(detected, entries, n_pages=50)
    assert _titles(merged) == ["Bayesian probabilities", "Thomas Bayes",
                               "The Gaussian distribution"]
    assert _titles(merged[0]["nodes"]) == ["Thomas Bayes"]


def test_different_style_stays_child():
    entries = [
        {"title": "Introduction", "level": 1, "page": 2},
        {"title": "Motivation", "level": 1, "page": 3},
    ]
    detected = [_node("1 Introduction", 2, style="F1", y=0.05,
                      children=[_node("Contributions", 2, style="F2",
                                      y=0.60)])]
    merged = merge_bookmark_tree(detected, entries, n_pages=6)
    assert _titles(merged) == ["Introduction", "Contributions", "Motivation"]
    assert _titles(merged[0]["nodes"]) == ["Contributions"]


def test_numbering_outside_candidate_climbs():
    entries = [
        {"title": "2. Background", "level": 1, "page": 3},
        {"title": "3. Methods", "level": 1, "page": 5},
    ]
    detected = [_node("2.1 Setup", 3), _node("4 Results", 5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=8)
    assert _titles(merged) == ["2. Background", "2.1 Setup", "3. Methods",
                               "4 Results"]
    assert _titles(merged[0]["nodes"]) == ["2.1 Setup"]
    assert not merged[1].get("nodes")


def test_climb_never_cuts_a_later_bookmark_section_short():
    entries = [
        {"title": "A", "level": 1, "page": 10},
        {"title": "A1", "level": 2, "page": 11},
        {"title": "A2", "level": 2, "page": 15},
        {"title": "B", "level": 1, "page": 20},
    ]
    detected = [_node("A", 10, style="F1", y=0.1),
                _node("A1", 11, style="F1", y=0.1),
                _node("Tail", 12, style="F1", y=0.5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=25)
    assert _titles(merged) == ["A", "A1", "Tail", "A2", "B"]
    assert _children(merged, "A") == ["A1", "Tail", "A2"]
    a2 = merged[0]["nodes"][-1]
    assert (a2["title"], a2["start_index"], a2["end_index"]) == ("A2", 15, 20)


def test_climb_the_next_bookmark_stops_keeps_its_height():
    entries = [
        {"title": "Part I", "level": 1, "page": 1},
        {"title": "First-Order Query Evaluation", "level": 2, "page": 2},
        {"title": "Homomorphisms", "level": 2, "page": 8},
        {"title": "Part II", "level": 1, "page": 12},
    ]
    detected = [_node("Part I", 1, style="S", y=0.1),
                _node("First-Order Query Evaluation", 2, style="S", y=0.1),
                _node("Static Analysis", 5, style="S", y=0.1),
                _node("Homomorphisms", 8, style="S", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=15)
    assert _children(merged, "Part I") == ["First-Order Query Evaluation", "Static Analysis", "Homomorphisms"]


def test_climb_the_next_bookmark_stops_stays_when_numbered_unlike_it():
    entries = [
        {"title": "2.1 Protocol Overview", "level": 1, "page": 10},
        {"title": "2.1.1 Pool Models", "level": 2, "page": 11},
        {"title": "1 SLP Model", "level": 3, "page": 11},
        {"title": "1.1.2 Security Layers", "level": 2, "page": 13},
        {"title": "Tokenomics", "level": 1, "page": 14},
    ]
    detected = [_node("3. Seize Every Airdrop Opportunity", 12, style="S", y=0.5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=15)
    assert _children(merged, "1 SLP Model") == ["3. Seize Every Airdrop Opportunity"]


def _chapters(level_three_printed, exchanges="Feedback Exchanges", printed_page=11, content_follows=None):
    entries = [{"title": "Chapter 1", "level": 1, "page": 1},
               {"title": "Implicit Theory", "level": 2, "page": 2},
               {"title": "Chapter 2", "level": 1, "page": 10},
               {"title": exchanges, "level": 2, "page": 11},
               {"title": "Chapter 3", "level": 1, "page": 20}]
    detected = [_node("Chapter 1", 1, style="C", y=0.1), _node("Implicit Theory", 2, style="S", y=0.1),
                _node("Chapter 2", 10, style="C", y=0.1), _node(exchanges, printed_page, style="S", y=0.1),
                _node("Overall Summary", printed_page, style="S", y=0.5), _node("Chapter 3", 20, style="C", y=0.1)]
    if content_follows is not None:
        detected[3]["_content_follows"] = content_follows
    if level_three_printed:
        entries.insert(2, {"title": "Dyadic Considerations", "level": 3, "page": 3})
        detected.insert(2, _node("Dyadic Considerations", 3, style="I", y=0.1))
    return entries, detected


@pytest.mark.parametrize("level_three_printed, exchanges, parent, children", [
    (True, "Feedback Exchanges", "Chapter 2", ["Feedback Exchanges", "Overall Summary"]),
    (True, "2.3 Feedback Exchanges", "2.3 Feedback Exchanges", ["Overall Summary"]),
])
def test_climb_stopped_halfway_keeps_its_height_only_against_how_the_bookmarks_print_the_level_below(
        level_three_printed, exchanges, parent, children):
    entries, detected = _chapters(level_three_printed, exchanges)
    # the bookmarks leave most of its print out, so only the halfway rule moves it
    detected[3:3] = [_node("Aside", 4, style="S", y=0.5), _node("Remark", 5, style="S", y=0.5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=25)
    assert _children(merged, parent) == children


@pytest.mark.parametrize("summary, extra, chapter_style, parent", [
    ("Overall Summary", [], "C", "Chapter 2"),
    ("Overall Summary:", [], "C", "Feedback Exchanges"),  # a run-in lead
    ("• Overall Summary", [], "C", "Feedback Exchanges"),  # a list item
    ("1. Overall Summary", [], "C", "Feedback Exchanges"),  # numbered beside unnumbered bookmarks
    ("Overall Summary", [_node("Overall Summary", 5, style="X", y=0.5)], "C", "Feedback Exchanges"),  # recurs
    ("Overall Summary", [_node("Aside", 3, style="S", y=0.5), _node("Remark", 4, style="S", y=0.5)],
     "C", "Feedback Exchanges"),  # the bookmarks leave most of this print out
    ("Overall Summary", [], "S", "Chapter 2"),  # the other level printed alike is numbered unlike it
])
def test_heading_stopped_halfway_is_a_peer_the_bookmarks_skipped_only_as_their_prints_vouch(
        summary, extra, chapter_style, parent):
    entries, detected = _chapters(False)
    detected[4]["title"] = summary
    detected[5]["_style"] = chapter_style
    detected[2:2] = extra
    merged = merge_bookmark_tree(detected, entries, n_pages=25)
    assert summary in _children(merged, parent)


def test_bookmark_printed_a_page_after_its_target_holds_nothing_were_the_heading_after_it_its_peer():
    entries, detected = _chapters(True, printed_page=12, content_follows=False)
    merged = merge_bookmark_tree(detected, entries, n_pages=25)
    assert _children(merged, "Feedback Exchanges") == ["Overall Summary"]


@pytest.mark.parametrize("chapters, closed, parent", [
    (3, (1, 2, 3), "{} Topic"),
    (7, (1, 4, 7), "{}.1 Part"),  # most chapters go without it
])
def test_title_closing_most_sections_of_a_level_alike_is_a_part_of_each(chapters, closed, parent):
    entries, detected = [], []
    for n in range(1, chapters + 1):
        entries += [{"title": f"{n} Topic", "level": 1, "page": 10 * n},
                    {"title": f"{n}.1 Part", "level": 2, "page": 10 * n + 1}]
        detected += [_node(f"{n} Topic", 10 * n, style="C", y=0.1), _node(f"{n}.1 Part", 10 * n + 1, style="S", y=0.1)]
        if n in closed:
            detected.append(_node("Exercises", 10 * n + 4, style="S", y=0.5))
    merged = merge_bookmark_tree(detected, entries, n_pages=10 * chapters + 10)
    assert all("Exercises" in _children(merged, parent.format(n)) for n in closed)


def test_title_sitting_unalike_in_its_sections_stays_where_it_reads():
    entries = [{"title": "Credit", "level": 1, "page": 1}, {"title": "Market", "level": 1, "page": 10},
               {"title": "Reputation", "level": 2, "page": 11}, {"title": "Country", "level": 1, "page": 20},
               {"title": "Climate", "level": 1, "page": 30}]
    detected = [_node("Credit", 1, style="H", y=0.1), _node("Organization", 1, style="O", y=0.5),
                _node("Market", 10, style="H", y=0.1), _node("Reputation", 11, style="O", y=0.1),
                _node("Organization", 11, style="O", y=0.5), _node("Country", 20, style="H", y=0.1),
                _node("Organization", 20, style="O", y=0.5), _node("Climate", 30, style="H", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=32)
    assert [_children(merged, title) for title in ("Credit", "Market", "Reputation", "Country")] == [
        ["Organization"], ["Reputation"], ["Organization"], ["Organization"]]


def test_detected_nesting_the_numbers_set_apart_is_dropped():
    entries = [{"title": "Item 8. Financial Statements", "level": 1, "page": 50},
               {"title": "Item 9. Changes in Accountants", "level": 1, "page": 90}]
    detected = [_node("Item 8. Financial Statements", 50, style="I", y=0.1, children=[
                    _node("NOTE 18. SEGMENTS", 80, style="N", y=0.1, children=[
                        _node("NOTE 19. QUARTERLY DATA", 82, style="N", y=0.1)])]),
                _node("Item 9. Changes in Accountants", 90, style="I", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=95)
    assert _children(merged, "Item 8. Financial Statements") == ["NOTE 18. SEGMENTS", "NOTE 19. QUARTERLY DATA"]


def test_climb_without_evidence_to_the_top_stays_at_the_anchor():
    entries = [
        {"title": "Part I", "level": 1, "page": 1},
        {"title": "Item 1. Business", "level": 2, "page": 3},
        {"title": "Item 1A. Risk Factors", "level": 2, "page": 18},
        {"title": "Part II", "level": 1, "page": 40},
    ]
    detected = [_node("Part I", 1, style="P", y=0.05),
                _node("Item 1A. Risk Factors", 18, style="S", y=0.1),
                _node("We Face Intense Competition", 18, style="S", y=0.5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=60)
    risk = merged[0]["nodes"][1]
    assert (risk["title"], risk["end_index"]) == ("Item 1A. Risk Factors", 40)
    assert _titles(risk["nodes"]) == ["We Face Intense Competition"]


@pytest.mark.parametrize("first, second", [("1. Directors", "2. Controller"),
                                           ("一、 董事声明", "二、 实际控制人声明")])
def test_peer_of_a_same_style_detected_child_stays(first, second):
    entries = [
        {"title": "Part 11", "level": 1, "page": 541},
        {"title": "Part 12 Declarations", "level": 1, "page": 548},
        {"title": "Part 13 Annexes", "level": 1, "page": 559},
    ]
    detected = [_node("Part 12 Declarations", 548, style="S", y=0.1),
                _node(first, 548, style="S", y=0.15),
                _node(second, 549, style="S", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=560)
    declarations = merged[1]
    assert declarations["end_index"] == 559
    assert _titles(declarations["nodes"]) == [first, second]


def test_tail_section_leaves_a_conclusion_holding_another_tail():
    entries = [
        {"title": "Discussion", "level": 1, "page": 15},
        {"title": "Conclusion", "level": 1, "page": 16},
    ]
    detected = [_node("Conclusion", 16, style="S", y=0.1),
                _node("Acknowledgements", 16, style="S", y=0.12),
                _node("References", 17, style="S", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=26)
    assert _titles(merged) == ["Discussion", "Conclusion", "Acknowledgements",
                               "References"]
    assert (merged[1]["end_index"], merged[2]["start_index"]) == (17, 17)


def test_section_a_heading_climbed_out_of_takes_no_later_heading():
    entries = [
        {"title": "Introduction", "level": 1, "page": 1},
        {"title": "Conclusion", "level": 1, "page": 9},
    ]
    detected = [_node("Conclusion", 9, style="S", y=0.1),
                _node("References", 9, style="S", y=0.5),
                _node("Prompt for Planning", 11, style="P", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=17)
    assert _titles(merged) == ["Introduction", "Conclusion", "References",
                               "Prompt for Planning"]
    assert merged[1]["end_index"] == 9


def test_tail_section_climbs_to_the_children_of_a_lone_root():
    entries = [
        {"title": "Agentic AI: a survey", "level": 1, "page": 1},
        {"title": "Introduction", "level": 2, "page": 1},
        {"title": "References", "level": 2, "page": 5},
    ]
    detected = [_node("References", 5, style="S", y=0.1),
                _node("Authors and Affiliations", 7, style="S", y=0.2)]
    merged = merge_bookmark_tree(detected, entries, n_pages=8)
    assert _titles(merged[0]["nodes"]) == ["Introduction", "References",
                                           "Authors and Affiliations"]
    assert merged[0]["nodes"][1]["end_index"] == 7


def test_heading_read_before_a_bookmarks_own_heading_goes_before_it():
    entries = [
        {"title": "Introduction", "level": 1, "page": 1},
        {"title": "Conclusion", "level": 1, "page": 5},
        {"title": "References", "level": 1, "page": 5},
    ]
    detected = [_node("Abstract", 1, style="S", y=0.2),
                _node("1 Introduction", 1, style="S", y=0.6),
                _node("6 Conclusion", 5, style="S", y=0.1),
                _node("Acknowledgments", 5, style="S", y=0.5),
                _node("References", 5, style="S", y=0.7)]
    merged = merge_bookmark_tree(detected, entries, n_pages=8)
    assert [n["title"] for n in merged] == [
        "Abstract", "Introduction", "Conclusion", "Acknowledgments",
        "References"]


def test_heading_read_before_a_bookmark_skips_no_section_between():
    entries = [
        {"title": "Conclusion", "level": 1, "page": 10},
        {"title": "Additional Discussion", "level": 1, "page": 16},
    ]
    detected = [_node("Conclusion", 10, style="S", y=0.1),
                _node("References", 12, style="S", y=0.1),
                _node("Appendix", 16, style="A", y=0.1),
                _node("Additional Discussion", 16, style="S", y=0.3)]
    merged = merge_bookmark_tree(detected, entries, n_pages=20)
    assert [n["title"] for n in merged] == ["Conclusion", "References",
                                            "Appendix", "Additional Discussion"]
    assert merged[0]["end_index"] == 12


def test_recurring_section_in_a_heading_style_is_kept():
    entries = [{"title": "Basics", "level": 1, "page": 1},
               {"title": "Methods", "level": 1, "page": 10},
               {"title": "Results", "level": 1, "page": 20}]
    detected = [
        _node("1 Basics", 1, style="H", y=0.1, children=[
            _node("ACME Corp", 2, style="R"), _node("ACME Corp", 3, style="R"),
            _node("Key Terms", 4, style="H"), _node("Key Terms", 5, style="H"),
            _node("Key Terms", 6, style="H"), _node("Note:", 7, style="N"),
            _node("Notes", 8, style="H")]),
        _node("2 Methods", 10, style="H", y=0.1, children=[
            _node("ACME Corp", 11, style="R"), _node("Note:", 15, style="N"),
            _node("Notes", 18, style="H")]),
        _node("3 Results", 20, style="H", y=0.1, children=[
            _node("Note:", 25, style="N"), _node("Notes", 28, style="H")])]
    merged = merge_bookmark_tree(detected, entries, n_pages=30)
    assert _titles(merged) == ["Basics", "Notes", "Methods", "Notes",
                               "Results", "Notes"]


def test_numbered_children_stay_with_a_spanning_detected_parent():
    entries = [
        {"title": "9.3 Engagement methods", "level": 1, "page": 170},
        {"title": "9.4 Results", "level": 1, "page": 178},
    ]
    detected = [_node("9.3.2 Qualitative engagement", 174,
                      [_node("9.3.2.1 Panel", 174),
                       _node("9.3.2.2 Online community", 176),
                       _node("9.4 Results", 178)])]
    merged = merge_bookmark_tree(detected, entries, n_pages=180)
    section = merged[0]["nodes"][0]
    assert section["title"] == "9.3.2 Qualitative engagement"
    assert _titles(section["nodes"]) == ["9.3.2.1 Panel",
                                         "9.3.2.2 Online community"]
    assert _titles(merged[1:]) == ["9.4 Results"]


def test_subsection_on_the_next_bookmarks_page_joins_its_section():
    entries = [
        {"title": "3 Computing", "level": 1, "page": 4},
        {"title": "3.2 CPU", "level": 2, "page": 4},
        {"title": "3.3 GPU", "level": 2, "page": 5},
        {"title": "4 Storage", "level": 1, "page": 7},
    ]
    detected = [_node("3.2.2 SIMD", 5), _node("3.3 GPU Kernels", 5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=9)
    cpu, gpu = merged[0]["nodes"]
    assert _titles(cpu["nodes"]) == ["3.2.2 SIMD"]
    assert (gpu["end_index"], _titles(gpu["nodes"])) == (7, ["3.3 GPU Kernels"])


def test_word_led_by_digits_is_no_section_number():
    entries = [
        {"title": "3 Methods", "level": 1, "page": 4},
        {"title": "3.1 Data", "level": 2, "page": 5},
        {"title": "3.2 Model", "level": 2, "page": 8},
        {"title": "4 Results", "level": 1, "page": 12},
    ]
    for title in ("3D Point Clouds", "2.5D Imaging", "2019 Survey",
                  "10-K Summary"):
        merged = merge_bookmark_tree([_node(title, 6)], entries, n_pages=14)
        data = merged[0]["nodes"][0]
        assert _titles(data["nodes"]) == [title]
        assert merged[0]["nodes"][1]["end_index"] == 12


def test_ending_in_a_frame_title_is_no_duplicate():
    entries = [
        {"title": "Intro", "level": 1, "page": 1},
        {"title": "Results", "level": 1, "page": 7},
        {"title": "Main", "level": 2, "page": 8},
        {"title": "End", "level": 1, "page": 10},
    ]
    detected = [_node("Item 5. Results", 7,
                      [_node("5.2 Ablation Results", 7),
                       _node("Executive Results", 7)])]
    merged = merge_bookmark_tree(detected, entries, n_pages=12)
    assert _titles(merged[1]["nodes"]) == ["5.2 Ablation Results",
                                           "Executive Results", "Main"]


def test_new_sections_still_graft():
    entries = [
        {"title": "Introduction", "level": 1, "page": 2},
        {"title": "Methods", "level": 1, "page": 5},
    ]
    detected = [_node("Abstract", 1),
                _node("Data Collection", 6)]
    merged = merge_bookmark_tree(detected, entries, n_pages=9)
    assert _titles(merged) == ["Abstract", "Introduction", "Methods",
                               "Data Collection"]
    assert _titles(merged[2]["nodes"]) == ["Data Collection"]


@pytest.mark.parametrize("text, readings", [
    ("4.3 Results", [((False, "digit", ""), (4, 3))]),
    ("1. Introduction", [((False, "digit", "."), (1,))]),
    ("IV. Results", [((False, "ROMAN", "."), (4,))]),
    ("I. Overview", [((False, "ROMAN", "."), (1,)), ((False, "LETTER", "."), (9,))]),
    ("A.2 Proofs", [((False, "LETTER", ""), (1, 2))]),
    ("Chapter 8 Methods", [(("chapter",), (8,))]),
    ("Appendix A Proofs", [(("label", "appendix"), (1,))]),
    ("第三章 总则", [(("chapter",), (3,))]),
    ("三、 总体要求", [((False, "cjk", "、"), (3,))]),
    ("3D Vision", []), ("10-K Filing", []), ("2019 Survey", []),
    ("xiv Introduction", []), ("Table 3: Results", []), ("4.3", []),
])
def test_section_number_reads_as_an_address(text, readings):
    assert _readings(text) == readings


def test_section_number_forks_off_an_unnumbered_bookmark():
    entries = [{"title": "Introduction", "level": 1, "page": 1},
               {"title": "Method", "level": 1, "page": 3},
               {"title": "Data", "level": 2, "page": 3},
               {"title": "Results", "level": 1, "page": 8}]
    detected = [_node("1 Introduction", 1), _node("2 Method", 3, y=0.1),
                _node("2.1 Data", 3, y=0.4), _node("2.2 Model", 5),
                _node("3 Results", 8)]
    merged = merge_bookmark_tree(detected, entries, n_pages=10)
    assert _children(merged, "Method") == ["Data", "2.2 Model"]
    assert _children(merged, "Data") == []


def test_section_number_read_off_the_page_at_a_bookmark():
    entries = [{"title": "Introduction", "level": 1, "page": 1},
               {"title": "Method", "level": 1, "page": 3},
               {"title": "Data", "level": 2, "page": 3},
               {"title": "Results", "level": 1, "page": 8}]
    texts = ["1 Introduction\nWe study", "", "2 Method\nText\n2.1 Data\nMore",
             "", "2.2 Model\nText", "", "", "3 Results", "", ""]
    merged = merge_bookmark_tree([_node("2.2 Model", 5)], entries, n_pages=10,
                                 line_of=_heading_line_reader(texts))
    assert _children(merged, "Method") == ["Data", "2.2 Model"]


def test_younger_sibling_bookmark_names_the_parent():
    entries = [{"title": "Part I", "level": 1, "page": 1},
               {"title": "Overview", "level": 2, "page": 1},
               {"title": "Background", "level": 2, "page": 3},
               {"title": "Part II", "level": 1, "page": 8}]
    detected = [_node("Part I", 1, y=0.1), _node("Overview", 1, y=0.3),
                _node("1 Introduction", 2), _node("2 Background", 3)]
    merged = merge_bookmark_tree(detected, entries, n_pages=10)
    assert _children(merged, "Part I") == ["Overview", "1 Introduction", "Background"]


def test_numbered_parent_bookmark_beats_a_detected_box_title():
    entries = [{"title": "5 Retrieval", "level": 1, "page": 1},
               {"title": "5.3 Memory", "level": 2, "page": 2},
               {"title": "6 End", "level": 1, "page": 6}]
    detected = [_node("5.3 Memory", 2, y=0.1, children=[
        _node("Four Steps", 3, children=[_node("5.3.1 Encoding", 3)])])]
    merged = merge_bookmark_tree(detected, entries, n_pages=8)
    assert _children(merged, "5.3 Memory") == ["Four Steps", "5.3.1 Encoding"]


_TOC_ROWS = [[("Introduction ........ 1", 0.1, 0.9, 0.10, "S"),
              ("Background ........ 2", 0.15, 0.9, 0.15, "S"),
              ("Methods ........ 3", 0.1, 0.9, 0.20, "S"),
              ("Setup ........ 4", 0.15, 0.9, 0.25, "S"),
              ("Tuning ........ 5", 0.15, 0.9, 0.30, "S"),
              ("Results ........ 6", 0.1, 0.9, 0.35, "S")]]
_TOC_BOOKMARKS = [{"title": "Introduction", "level": 1, "page": 3},
                  {"title": "Methods", "level": 1, "page": 5},
                  {"title": "Setup", "level": 2, "page": 6},
                  {"title": "Results", "level": 1, "page": 8}]


def test_printed_toc_levels_parents_and_page_offset():
    toc = read_printed_toc(_TOC_ROWS, _TOC_BOOKMARKS)
    assert [(e["title"], e["parent"], e["page"]) for e in toc] == [
        ("Introduction", None, 3), ("Background", 0, 4), ("Methods", None, 5),
        ("Setup", 2, 6), ("Tuning", 2, 7), ("Results", None, 8)]


def test_printed_toc_that_nests_the_bookmarks_alike_places_a_heading():
    toc = read_printed_toc(_TOC_ROWS, _TOC_BOOKMARKS)
    merged = merge_bookmark_tree([_node("Background", 4), _node("Tuning", 7)],
                                 _TOC_BOOKMARKS, n_pages=10, toc=toc)
    assert _children(merged, "Introduction") == ["Background"]
    assert _children(merged, "Methods") == ["Setup", "Tuning"]


def test_detected_references_before_back_matter_go_to_the_top():
    entries = [{"title": "Introduction", "level": 1, "page": 1},
               {"title": "Conclusion", "level": 1, "page": 8},
               {"title": "Appendix A Proofs", "level": 1, "page": 12}]
    references = _node("References", 10)
    references["_type"] = 7
    merged = merge_bookmark_tree([references], entries, n_pages=14)
    assert [n["title"] for n in merged] == [
        "Introduction", "Conclusion", "References", "Appendix A Proofs"]


def test_doubled_dot_in_a_section_number_reads_as_no_number():
    entries = [{"title": "A Appendix", "level": 1, "page": 1},
               {"title": "A..1 Notation", "level": 2, "page": 2},
               {"title": "B Data", "level": 1, "page": 5}]
    merged = merge_bookmark_tree([_node("A.2 Proofs", 2, y=0.1)], entries, n_pages=6)
    assert "A.2 Proofs" in _titles(merged)


def test_bookmark_target_splits_its_page():
    entries = [{"title": "Method", "level": 1, "page": 3, "x": 0.1, "y": 0.1},
               {"title": "Results", "level": 1, "page": 4, "x": 0.1, "y": 0.5},
               {"title": "End", "level": 1, "page": 6}]
    merged = merge_bookmark_tree([_node("Setup", 4, y=0.2), _node("Accuracy", 4, y=0.7)],
                                 entries, n_pages=8)
    assert _children(merged, "Method") == ["Setup"]
    assert _children(merged, "Results") == ["Accuracy"]


def test_bookmark_target_in_the_right_column_follows_the_left_column():
    entries = [{"title": "Method", "level": 1, "page": 3, "x": 0.1, "y": 0.1},
               {"title": "Results", "level": 1, "page": 4, "x": 0.55, "y": 0.25},
               {"title": "End", "level": 1, "page": 6}]
    detected = [_node("Setup", 4, y=0.2), _node("Training", 4, y=0.8),
                _node("Accuracy", 4, y=0.3)]
    for node, col in zip(detected, (0, 0, 1)):
        node["_col"] = col
    # flash's column regions on page 4, in its reading order: left column, then right
    columns = [[]] * 3 + [[(0.1, 0.45, 0.1, 0.9), (0.55, 0.9, 0.1, 0.9)]] + [[]] * 4
    merged = merge_bookmark_tree(detected, entries, n_pages=8, columns=columns)
    assert _children(merged, "Method") == ["Setup", "Training"]
    assert _children(merged, "Results") == ["Accuracy"]


def test_printed_toc_joins_label_cells_and_wrapped_titles():
    rows = [[("1.", 0.1, 0.13, 0.10, "S"), ("Introduction ........ 1", 0.15, 0.9, 0.10, "S"),
             ("2.", 0.1, 0.13, 0.15, "S"), ("Related Work on Long", 0.15, 0.6, 0.15, "S"),
             ("Documents ........ 2", 0.15, 0.9, 0.17, "S"),
             ("3. Methods ........ 3", 0.1, 0.9, 0.22, "S"),
             ("4. Results ........ 4", 0.1, 0.9, 0.27, "S"),
             ("5. Conclusion ........ 5", 0.1, 0.9, 0.32, "S")]]
    entries = [{"title": t, "level": 1, "page": p} for t, p in [
        ("Introduction", 3), ("Related Work on Long Documents", 4), ("Methods", 5),
        ("Results", 6), ("Conclusion", 7)]]
    assert [(e["title"], e["page"]) for e in read_printed_toc(rows, entries)] == [
        ("1. Introduction", 3), ("2. Related Work on Long Documents", 4), ("3. Methods", 5),
        ("4. Results", 6), ("5. Conclusion", 7)]


def test_printed_toc_indents_take_the_levels_of_the_bookmarks_listed_at_them():
    rows = [[("1 Introduction ........ 1", 0.10, 0.9, 0.10, "S"),
             ("1.1 Background ........ 1", 0.15, 0.9, 0.15, "S"),
             ("1.2 Scope ........ 2", 0.18, 0.9, 0.20, "S"),
             ("1.2.1 Details ........ 2", 0.22, 0.9, 0.25, "S"),
             ("2 Methods ........ 3", 0.10, 0.9, 0.30, "S")]]
    entries = [{"title": "Introduction", "level": 1, "page": 3}, {"title": "Background", "level": 2, "page": 3},
               {"title": "Scope", "level": 2, "page": 4}, {"title": "Methods", "level": 1, "page": 5}]
    assert [e["parent"] for e in read_printed_toc(rows, entries)] == [None, 0, 0, 2, None]


def test_heading_printed_larger_in_the_same_face_climbs_out_of_a_bookmark():
    def merged(small):
        entries = [{"title": "Terms", "level": 1, "page": 1, "x": 0.1, "y": 0.1},
                   {"title": "Miscellaneous", "level": 2, "page": 2, "x": 0.1, "y": 0.1},
                   {"title": "Pricing", "level": 1, "page": 4, "x": 0.1, "y": 0.1}]
        detected = [_node("Terms", 1, style="Arial-Bold B 14.0 false", y=0.1),
                    _node("Miscellaneous", 2, style=small, y=0.1),
                    _node("Exhibit B", 3, style="Arial-Bold B 14.0 false", y=0.1),
                    _node("Pricing", 4, style="Arial-Bold B 14.0 false", y=0.1)]
        return [n["title"] for n in merge_bookmark_tree(detected, entries, n_pages=5)]
    assert merged("Arial-Bold B 11.0 false") == ["Terms", "Exhibit B", "Pricing"]
    assert merged("Arial-Bold B 11.0 true") == ["Terms", "Pricing"]  # small caps are set small by design


def test_page_number_above_a_heading_is_no_section_number():
    assert _heading_line_reader(["17\nBackground\nText"])(1, "Background") is None


def test_heading_numbered_in_another_list_style_is_no_peer():
    entries = [{"title": "三、人工智能应用", "level": 1, "page": 1},
               {"title": "（二）工程化", "level": 2, "page": 2},
               {"title": "（三）体系化推动落地", "level": 2, "page": 3},
               {"title": "四、产业", "level": 1, "page": 6}]
    detected = [_node(t, p, style="h") for t, p in (
        ("三、人工智能应用", 1), ("（二）工程化", 2), ("（三）体系化推动落地", 3),
        ("2. 明确选型方案", 4), ("3. 建设能力平台", 5), ("四、产业", 6))]
    merged = merge_bookmark_tree(detected, entries, n_pages=7)
    assert _children(merged, "（三）体系化推动落地") == ["2. 明确选型方案", "3. 建设能力平台"]


def test_numbered_prints_of_one_title_take_their_bookmarks_in_order():
    entries = [{"title": "Retrieval", "level": 1, "page": 1},
               {"title": "Future Perspective", "level": 2, "page": 2},
               {"title": "Memory Management", "level": 1, "page": 2},
               {"title": "Future Perspective", "level": 2, "page": 2},
               {"title": "End", "level": 1, "page": 5}]
    detected = [_node("1 Retrieval", 1), _node("1.2 Future Perspective", 2, y=0.2),
                _node("2 Memory Management", 2, y=0.5), _node("Background", 2, y=0.6),
                _node("2.2 Future Perspective", 2, y=0.8), _node("End", 5)]
    merged = merge_bookmark_tree(detected, entries, n_pages=6)
    assert _children(merged, "Memory Management") == ["Background", "Future Perspective"]


def test_print_at_a_bookmark_target_keeps_the_heading_glued_after_its_title():
    title = "Academic Advising and Satisfactory Progress Policies"
    entries = [{"title": title, "level": 1, "page": 2, "x": 0.1, "y": 0.28},
               {"title": "Tuition", "level": 1, "page": 4, "x": 0.1, "y": 0.1}]
    detected = [_node(title + " Academic Advising", 2, y=0.285), _node("Tuition", 4, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=5)
    assert _children(merged, title) == ["Academic Advising"]


def test_children_printed_like_the_heading_glued_after_a_bookmark_title_are_its_peers():
    entries = [{"title": "Bottom Duct Application", "level": 1, "page": 2, "x": 0.1, "y": 0.1},
               {"title": "Side Duct Application", "level": 1, "page": 5, "x": 0.1, "y": 0.1}]
    glued = _node("Bottom Duct Application J15ZF (15 Ton)", 2, style="b10", y=0.1,
                  children=[_node("Sizing table", 2, style="r9"), _node("J18ZF (17.5 Ton)", 3, style="b9")])
    glued["_lines"] = [["Bottom Duct Application", "b10"], ["J15ZF (15 Ton)", "b9"]]
    merged = merge_bookmark_tree([glued, _node("Side Duct Application", 5, y=0.1)], entries, n_pages=6)
    assert _children(merged, "Bottom Duct Application") == ["J15ZF (15 Ton)", "J18ZF (17.5 Ton)"]
    assert _children(merged, "J15ZF (15 Ton)") == ["Sizing table"]


def test_remark_after_a_bookmark_title_at_its_target_is_no_heading():
    title = "Item 4. Mine Safety Disclosures."
    entries = [{"title": title, "level": 1, "page": 2, "x": 0.1, "y": 0.1},
               {"title": "Item 5. Market", "level": 1, "page": 3, "x": 0.1, "y": 0.1}]
    detected = [_node(title + " Not applicable.", 2, y=0.1), _node("Item 5. Market", 3, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=4)
    assert _children(merged, title) == []


def test_heading_below_a_bookmark_target_keeps_the_bookmark_title_it_starts_with():
    entries = [{"title": "Evaluation", "level": 1, "page": 2, "x": 0.1, "y": 0.1},
               {"title": "Discussion", "level": 1, "page": 4, "x": 0.1, "y": 0.1}]
    detected = [_node("Evaluation", 2, y=0.1), _node("Evaluation Metrics", 2, y=0.4),
                _node("Discussion", 4, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=5)
    assert _children(merged, "Evaluation") == ["Evaluation Metrics"]


def test_print_adding_words_to_a_bookmark_title_away_from_its_target_is_another_heading():
    entries = [{"title": "Earth Sciences", "level": 1, "page": 2, "x": 0.1, "y": 0.09},
               {"title": "Earth Sciences, B.A.", "level": 2, "page": 2, "x": 0.1, "y": 0.66},
               {"title": "Geology", "level": 1, "page": 4, "x": 0.1, "y": 0.1}]
    detected = [_node("Earth Sciences", 2, y=0.09), _node("General Information", 2, y=0.27),
                _node("Earth Sciences, B.A.", 2, y=0.66), _node("Degree Requirements", 2, y=0.8),
                _node("Geology", 4, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=5)
    assert _children(merged, "Earth Sciences") == ["General Information", "Earth Sciences, B.A."]
    assert _children(merged, "Earth Sciences, B.A.") == ["Degree Requirements"]


def test_initials_after_a_bookmark_title_are_no_remark():
    entries = [{"title": "Chemistry", "level": 1, "page": 1, "x": 0.1, "y": 0.1},
               {"title": "Mathematics", "level": 1, "page": 2, "x": 0.1, "y": 0.09},
               {"title": "Physics", "level": 1, "page": 4, "x": 0.1, "y": 0.1}]
    detected = [_node("Chemistry", 1, y=0.1), _node("Double Major Options", 2, y=0.3),
                _node("Mathematics, B.A.", 2, y=0.6), _node("Physics", 4, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=5)
    assert _children(merged, "Mathematics") == ["Double Major Options", "Mathematics, B.A."]


def test_heading_beside_a_bookmark_print_is_its_peer_only_with_content_between():
    # "Keywords" opens the next column level with "Abstract": only flash's reading order tells them apart
    entries = [{"title": "Abstract", "level": 1, "page": 1, "x": 0.1, "y": 0.2},
               {"title": "1 Introduction", "level": 1, "page": 2, "x": 0.1, "y": 0.1}]
    for content, children in ((True, []), (False, ["Keywords"])):
        abstract = _node("Abstract", 1, style="B", y=0.2)
        abstract["_content_follows"] = content
        detected = [abstract, _node("Keywords", 1, style="B", y=0.2), _node("1 Introduction", 2, style="B", y=0.1)]
        merged = merge_bookmark_tree(detected, entries, n_pages=3)
        assert _children(merged, "Abstract") == children
    # another heading between them is content of "Abstract" too
    abstract = _node("Abstract", 1, style="B", y=0.2)
    abstract["_content_follows"] = False
    detected = [abstract, _node("Notes", 1, style="C", y=0.3), _node("Keywords", 1, style="B", y=0.5),
                _node("1 Introduction", 2, style="B", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=3)
    assert _children(merged, "Abstract") == ["Notes"]


def test_label_printed_above_a_bookmark_heading_holding_content_stays_inside_it():
    entries = [{"title": "Training", "level": 1, "page": 1, "x": 0.1, "y": 0.1},
               {"title": "Log Analysis", "level": 1, "page": 2, "x": 0.1, "y": 0.09}]
    label, heading = _node("EIGHTEEN", 2, style="B", y=0.14), _node("LOG ANALYSIS", 2, style="B", y=0.21)
    label["_content_follows"], heading["_content_follows"] = False, True
    detected = [_node("Training", 1, style="B", y=0.1), label, heading]
    merged = merge_bookmark_tree(detected, entries, n_pages=3)
    assert [node["title"] for node in merged] == ["Training", "Log Analysis"]


def test_heading_numbered_unlike_the_level_of_an_unnumbered_bookmark_is_no_peer():
    entries = [{"title": "I. Introduction", "level": 1, "page": 1, "x": 0.1, "y": 0.1},
               {"title": "II. References", "level": 1, "page": 2, "x": 0.1, "y": 0.1},
               {"title": "APPENDIX", "level": 1, "page": 3, "x": 0.1, "y": 0.1}]
    appendix = _node("APPENDIX", 3, style="B", y=0.1)
    appendix["_content_follows"] = True
    detected = [_node("I. Introduction", 1, style="B", y=0.1), _node("II. References", 2, style="B", y=0.1),
                appendix, _node("3.2. Drug Substance", 3, style="B", y=0.3)]
    merged = merge_bookmark_tree(detected, entries, n_pages=4)
    assert _children(merged, "APPENDIX") == ["3.2. Drug Substance"]


def test_bookmark_reading_unlike_its_siblings_is_dropped():
    # "V." reads as Roman 5 and as the letter V; beside "IV." and "VI." it is Roman, so "C." may follow "B." under it
    titles = [("IV. Summary", 1, 1), ("V. Manufacturing", 1, 2), ("A. Drug Substance", 2, 2),
              ("B. Drug Product", 2, 3), ("1. Description", 3, 3), ("2. Development", 3, 4), ("VI. References", 1, 6)]
    entries = [{"title": t, "level": level, "page": page} for t, level, page in titles]
    detected = [_node(t, page, y=0.1) for t, _, page in titles[:6]]
    detected += [_node("C. Appendices", 5, y=0.1), _node("VI. References", 6, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=7)
    assert _children(merged, "V. Manufacturing") == ["A. Drug Substance", "B. Drug Product", "C. Appendices"]


def test_label_words_read_in_any_language_flash_knows():
    assert _readings("Kapitel 3 Methoden") == _readings("Chapter 3 Methods") == [(("chapter",), (3,))]


def test_number_whose_depth_the_bookmarks_disagree_on_falls_back_to_its_younger_sibling():
    # "1.1" and "1、" both sit at level 2, so digit numbers say nothing about depth here
    titles = [("一、概述", 1, 1), ("1.1 背景", 2, 1), ("1.2 目标", 2, 2), ("1.3 范围", 2, 2),
              ("七、产业全景图", 1, 3), ("1、AIGC：生成内容", 2, 3), ("4、行业进程", 2, 5)]
    entries = [{"title": t, "level": level, "page": page} for t, level, page in titles]
    detected = [_node(t, page, y=0.1) for t, _, page in titles[:6]]
    detected += [_node("2、当下为何关注", 4, y=0.1), _node("4、行业进程", 5, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=6)
    assert _children(merged, "七、产业全景图") == ["1、AIGC：生成内容", "2、当下为何关注", "4、行业进程"]


def test_one_reading_of_a_leading_number_serves_every_rule():
    digit = (False, "digit", "")
    assert _readings("1.3.4商业模式") == [(digit, (1, 3, 4))]  # glued to the title it numbers
    assert _readings("5.1、AI超算中心") == [(digit, (5, 1))]  # after "、" the title may start anyhow
    assert _readings("3D Printing") == _readings("2019 Survey") == []
    assert _readings("② 报告期内") == [((False, "enclosed", ""), (2,))]  # its own list style, not "2"
    assert _readings("２．方法") == [((False, "digit", "."), (2,))]  # a fullwidth digit is a digit


def test_heading_ending_in_initials_glued_after_a_bookmark_title_is_no_remark():
    entries = [{"title": "Interdisciplinary Programs", "level": 1, "page": 2, "x": 0.1, "y": 0.09},
               {"title": "Economics", "level": 1, "page": 4, "x": 0.1, "y": 0.1}]
    detected = [_node("Interdisciplinary Programs Biochemistry, B.S.", 2, y=0.09),
                _node("Economics", 4, y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=5)
    assert _children(merged, "Interdisciplinary Programs") == ["Biochemistry, B.S."]


def test_print_takes_the_bookmark_it_prints_exactly_over_a_similar_one():
    entries = [{"title": "Sociology, B.A.", "level": 1, "page": 2, "x": 0.1, "y": 0.17},
               {"title": "Sociology, B.S.", "level": 1, "page": 2, "x": 0.1, "y": 0.65}]
    detected = [_node("Sociology, B.A.", 2, y=0.17), _node("Core Courses", 2, y=0.39),
                _node("Sociology, B.S.", 2, y=0.65), _node("Electives", 2, y=0.8)]
    merged = merge_bookmark_tree(detected, entries, n_pages=3)
    assert _children(merged, "Sociology, B.A.") == ["Core Courses"]
    assert _children(merged, "Sociology, B.S.") == ["Electives"]


def test_bookmark_target_on_a_page_whose_box_is_inherited():
    # A4 MediaBox on /Pages only: pdfium's get_cropbox would fall back to US Letter
    pdf = (b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R /Outlines 4 0 R >> endobj\n"
           b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 595 842] >> endobj\n"
           b"3 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
           b"4 0 obj << /Type /Outlines /First 5 0 R /Last 5 0 R /Count 1 >> endobj\n"
           b"5 0 obj << /Title (Intro) /Parent 4 0 R /Dest [3 0 R /XYZ 119 421 0] >> endobj\n"
           b"trailer << /Root 1 0 R >>\n%%EOF\n")
    want = [{"title": "Intro", "level": 1, "page": 1, "x": pytest.approx(0.2), "y": pytest.approx(0.5)}]
    assert read_bookmarks(BytesIO(pdf)) == want
    assert read_bookmarks(BytesIO(pdf), [((0, 0, 595, 842), 0)]) == want


def test_bookmark_target_on_a_rotated_page_reads_as_flash_places_headings():
    pdf = (b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R /Outlines 4 0 R >> endobj\n"
           b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 595 842] >> endobj\n"
           b"3 0 obj << /Type /Page /Parent 2 0 R /Rotate 90 >> endobj\n"
           b"4 0 obj << /Type /Outlines /First 5 0 R /Last 5 0 R /Count 1 >> endobj\n"
           b"5 0 obj << /Title (Intro) /Parent 4 0 R /Dest [3 0 R /XYZ 119 421 0] >> endobj\n"
           b"trailer << /Root 1 0 R >>\n%%EOF\n")
    # turned a quarter, the page reads down its user-space x: 119 of 595
    want = [{"title": "Intro", "level": 1, "page": 1, "x": None, "y": pytest.approx(0.2)}]
    assert read_bookmarks(BytesIO(pdf)) == want
    assert read_bookmarks(BytesIO(pdf), [((0, 0, 595, 842), 90)]) == want
