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
