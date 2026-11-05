import pytest

from pageindex.flash.embedded_toc import merge_bookmark_tree


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
        {"title": "End", "level": 1, "page": 21},
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
                _node("A. Proofs", 20)]
    merged = merge_bookmark_tree(detected, entries, n_pages=22)
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
        "Appendix Proofs", "End"]


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
    assert _titles(merged[0]["nodes"][0]["nodes"]) == ["Tail"]
    a2 = merged[0]["nodes"][-1]
    assert (a2["title"], a2["start_index"], a2["end_index"]) == ("A2", 15, 20)


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
    assert _titles(merged) == ["Conclusion", "References",
                               "Additional Discussion", "Appendix"]
    assert merged[0]["end_index"] == 12


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
