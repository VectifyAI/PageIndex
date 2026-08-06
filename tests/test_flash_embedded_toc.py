from pageindex.flash.embedded_toc import merge_bookmark_tree


def _node(title, start, children=()):
    return {"title": title, "node_id": "", "start_index": start,
            "end_index": start, "nodes": [dict(c) for c in children]}


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
