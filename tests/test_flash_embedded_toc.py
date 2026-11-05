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
        {"title": "End", "level": 1, "page": 10},
    ]
    detected = [_node("JPMorgan Chase & Co. Consolidated balance sheets "
                      "(unaudited)", 3),
                _node("Revenue (dollars in millions)", 4),
                _node("Note 18 - Leases Firm as lessee", 5),
                _node("Fair value", 6),
                _node("Assets measured at fair value", 6),
                _node("5.2 Ablation Results", 7),
                _node("Derivatives netting", 8),
                _node("Note 26 – Restricted cash and other assets", 9)]
    merged = merge_bookmark_tree(detected, entries, n_pages=11)
    assert _titles(merged) == [
        "Consolidated balance sheets", "Revenue", "Note 18 - Leases",
        "Note 18 - Leases Firm as lessee", "Fair value",
        "Assets measured at fair value", "Results", "5.2 Ablation Results",
        "Derivatives netting and collateral", "Note 26 - Restrictions on cash",
        "End"]


def test_letter_spaced_duplicate_dissolves():
    entries = [
        {"title": "Conclusion", "level": 1, "page": 12},
        {"title": "References", "level": 1, "page": 13},
    ]
    detected = [_node("R E F E R E N C E S", 13)]
    merged = merge_bookmark_tree(detected, entries, n_pages=14)
    assert _titles(merged) == ["Conclusion", "References"]


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


def test_peer_of_a_same_style_detected_child_stays():
    entries = [
        {"title": "Part 11", "level": 1, "page": 541},
        {"title": "Part 12 Declarations", "level": 1, "page": 548},
        {"title": "Part 13 Annexes", "level": 1, "page": 559},
    ]
    detected = [_node("Part 12 Declarations", 548, style="S", y=0.1),
                _node("1. Directors", 548, style="S", y=0.15),
                _node("2. Controller", 549, style="S", y=0.1)]
    merged = merge_bookmark_tree(detected, entries, n_pages=560)
    declarations = merged[1]
    assert declarations["end_index"] == 559
    assert _titles(declarations["nodes"]) == ["1. Directors", "2. Controller"]


def test_references_leave_a_conclusion_holding_its_acknowledgements():
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
