"""The tree every local index writes: intro nodes, covering ranges, leaf-only
splitting, and summaries every node gets, a parent's built from its children's."""

import asyncio
import copy
import importlib
from types import SimpleNamespace

import pytest

import pageindex.flash
import pageindex.tree_optimize as tree_optimize
import pageindex.utils as utils
from conftest import build_pdf
from pageindex import PageIndexClient

classic = importlib.import_module("pageindex.page_index_classic")


def shape(nodes):
    return [(n["title"], n["start_index"], n["end_index"]) for n in utils._subtree(nodes)]


def test_intro_node_holds_the_pages_a_parent_opens_with():
    lines = [["body"] for _ in range(40)]
    lines[11] = ["3.1 Scope", "body"]      # 3.1 opens its page, 4.1 starts mid-page
    tree = [{"title": "Ch 3", "start_index": 10, "end_index": 30, "nodes": [
                {"title": "3.1 Scope", "start_index": 12, "end_index": 20},
                {"title": "3.2", "start_index": 21, "end_index": 30, "nodes": [
                    {"title": "3.2.1", "start_index": 21, "end_index": 30}]}]},
            {"title": "", "start_index": 31, "end_index": 40, "nodes": [
                {"title": "4.1", "start_index": 33, "end_index": 40}]}]

    tree_optimize.add_intro_nodes(tree, lines)

    assert shape(tree) == [
        ("Ch 3", 10, 30), ("Ch 3 (intro)", 10, 11), ("3.1 Scope", 12, 20),
        ("3.2", 21, 30), ("3.2.1", 21, 30),     # a first child on its parent's page: no intro
        ("", 31, 40), ("Intro", 31, 33), ("4.1", 33, 40)]
    assert shape(tree_optimize.add_intro_nodes(copy.deepcopy(tree), lines)) == shape(tree)
    # a standard parent ends where its first child starts; a split intro keeps its title
    split = [{"title": "Ch 3 (intro)", "start_index": 10, "end_index": 11, "nodes": [
        {"title": "Background", "start_index": 12, "end_index": 14}]}]
    assert shape(tree_optimize.add_intro_nodes(split))[1] == ("Ch 3 (intro)", 10, 11)
    # a heading with nothing to match cannot be placed on its page, so the page is shared
    assert not tree_optimize.heading_at_page_start([["第一章 总则"]], 1, "第一章 总则")
    # nor can digits alone, which may be a page number
    assert not tree_optimize.heading_at_page_start([["2", "Body text"]], 1, "2")


def test_expand_gives_a_split_node_its_intro(monkeypatch):
    body = "body " * 250
    pages = [body] * 12
    pages[5] = "Sub One\n" + body
    pages[8] = "Sub Two\n" + body
    lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 12, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "X", "start_index": 4, "end_index": 12, "node_id": "0002"}]}]

    async def propose(model, prompt):
        return {"subsections": [{"title": "Sub One", "page": 6}, {"title": "Sub Two", "page": 9}]}

    async def summarize(model, prompt):
        return '{"summary": "ok"}'
    monkeypatch.setattr(tree_optimize, "ask_model", propose)
    monkeypatch.setattr(utils, "llm_acompletion", summarize)

    async def run():
        scheduler = utils.SummaryScheduler(tree, [(page, 0) for page in pages], model="m")
        await tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True,
                                     on_final=scheduler.mark_final)
        await scheduler.finish()
    asyncio.run(run())

    x = tree[0]["nodes"][1]
    assert [(n["title"], n["start_index"], n["end_index"], n["node_id"]) for n in x["nodes"]] == [
        ("X (intro)", 4, 5, "0003"), ("Sub One", 6, 8, "0004"), ("Sub Two", 9, 12, "0005")]
    assert all(n["summary"] == "ok" for n in utils._subtree(tree))


def test_a_running_header_or_a_word_prefix_does_not_open_the_page():
    page = ["4.1. Discriminant Functions 181", "opening words", "4.1. Discriminant Functions"]
    assert not tree_optimize.heading_at_page_start([page], 1, "4.1. Discriminant Functions")
    assert tree_optimize.heading_at_page_start([page[2:]], 1, "4.1. Discriminant Functions")
    assert not tree_optimize.heading_at_page_start([["Filed 03/04/24", "I. Background"]], 1, "I.")


def test_expand_prices_a_level_with_the_intro_it_gets(monkeypatch):
    body = "body " * 250
    pages = [body] * 12
    pages[10] = body + "\nSub Late\n" + body     # mid-page: the intro shares page 11
    lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 12, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "X", "start_index": 4, "end_index": 12, "node_id": "0002"}]}]

    async def propose(model, prompt):
        return {"subsections": [{"title": "Sub Late", "page": 11}]}

    async def summarize(model, prompt):
        return '{"summary": "ok"}'
    monkeypatch.setattr(tree_optimize, "ask_model", propose)
    monkeypatch.setattr(utils, "llm_acompletion", summarize)

    async def run():
        scheduler = utils.SummaryScheduler(tree, [(page, 0) for page in pages], model="m")
        await tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True,
                                     on_final=scheduler.mark_final)
        await scheduler.finish()
    asyncio.run(run())

    # intro 4-11 + Sub Late 11-12 costs 1 + 8 = 9 pages, no better than reading X's 9
    assert "nodes" not in tree[0]["nodes"][1]


def test_expand_skips_the_heading_of_a_neighbor_sharing_the_last_page(monkeypatch):
    body = "body " * 250
    pages = [body] * 20
    pages[5] = "Setup\n" + body
    # mid-page, so Methods runs onto page 12; run in, so only the tree knows the headings below it
    pages[11] = body + "\n3 Results. We report three findings.\n3.1 Data\n" + body
    lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 20, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "Methods", "start_index": 4, "end_index": 12, "node_id": "0002"},
        {"title": "3 Results", "start_index": 12, "end_index": 20, "node_id": "0003", "nodes": [
            {"title": "3.1 Data", "start_index": 12, "end_index": 15, "node_id": "0004"},
            {"title": "3.2 More", "start_index": 16, "end_index": 20, "node_id": "0005"}]}]}]

    replies = [[{"title": "Results", "page": 12}, {"title": "3.1 Data", "page": 12}],
               [{"title": "Setup", "page": 6}, {"title": "Results", "page": 12},
                {"title": "3.1 Data", "page": 12}]]

    async def propose(model, prompt):
        if "Section title: Methods\n" in prompt:   # a reply the filter empties is asked again
            return {"subsections": replies.pop(0)}
        return {"subsections": []}
    monkeypatch.setattr(tree_optimize, "ask_model", propose)

    asyncio.run(tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True))

    assert not replies
    assert shape(tree[0]["nodes"][1]["nodes"]) == [("Methods (intro)", 4, 5), ("Setup", 6, 12)]


def test_expand_filters_cached_headings_like_proposed_ones(monkeypatch):
    body = "body " * 250
    pages = [body] * 20
    pages[5] = "Setup\n" + body
    pages[11] = body + "\n3 Results\n" + body
    lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 20, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "Methods", "start_index": 4, "end_index": 12, "node_id": "0002"},
        {"title": "3 Results", "start_index": 12, "end_index": 20, "node_id": "0003"}]}]
    cache = {6: [{"title": "Setup", "kind": "section"}],
             12: [{"title": "3 Results", "kind": "section"}]}

    async def propose(model, prompt):
        return {"subsections": []}
    monkeypatch.setattr(tree_optimize, "ask_model", propose)

    asyncio.run(tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True, cache=cache))

    assert shape(tree[0]["nodes"][1]["nodes"]) == [("Methods (intro)", 4, 5), ("Setup", 6, 12)]


def test_expand_hands_each_new_child_its_ancestors_and_next_node(monkeypatch):
    body = "body " * 250
    pages = [body] * 30
    pages[3] = "P\nP.a Early\n" + body
    pages[6] = "P.b Later\n" + body
    pages[9] = body + "\nQ\nQ.0 Overview\n" + body
    pages[12] = "Q.i Intro part\n" + body
    pages[16] = "Q.1 First\n" + body
    pages[19] = "Q.2 Second\n" + body
    pages[24] = body + "\nS\nS.1 Part\n" + body
    lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 30, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "P", "start_index": 4, "end_index": 25, "node_id": "0002"},
        {"title": "S", "start_index": 25, "end_index": 30, "node_id": "0003"}]}]
    replies = {
        "P": [("Q", 10)],
        # Q, made in this pass, is P's intro's next node: what follows its heading is not the intro's
        "P (intro)": [("P.a Early", 4), ("P.b Later", 7), ("Q.0 Overview", 10)],
        # Q's next node is P's: S
        "Q": [("Q.1 First", 17), ("Q.2 Second", 20), ("S.1 Part", 25)],
        # Q's intro sits under Q, an ancestor made in this pass
        "Q (intro)": [("Q", 10), ("Q.0 Overview", 10), ("Q.i Intro part", 13)]}

    async def propose(model, prompt):
        title = prompt.split("Section title: ")[1].split("\n")[0]
        return {"subsections": [{"title": t, "page": p} for t, p in replies.get(title, [])]}
    monkeypatch.setattr(tree_optimize, "ask_model", propose)

    asyncio.run(tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True, do_relabel=False))

    assert shape(tree) == [
        ("R", 1, 30), ("A", 1, 3), ("P", 4, 25),
        ("P (intro)", 4, 10), ("P.a Early", 4, 6), ("P.b Later", 7, 10),
        ("Q", 10, 25), ("Q (intro)", 10, 16), ("Q.0 Overview", 10, 12), ("Q.i Intro part", 13, 16),
        ("Q.1 First", 17, 19), ("Q.2 Second", 20, 25), ("S", 25, 30)]


def test_a_proposed_child_must_be_printed_inside_its_nodes_own_text():
    lines = [["body"] for _ in range(20)]
    lines[11] = ["2.3 Limits", "3 Results", "Scope", "3.1 Data", "3.2 Results"]
    methods = {"title": "2 Methods", "start_index": 4, "end_index": 12}
    results = {"title": "3 Results", "start_index": 12, "end_index": 20}
    root = {"title": "R", "start_index": 1, "end_index": 20, "nodes": [methods, results]}
    known = {}
    for node, _ in tree_optimize.flatten([root]):
        known.setdefault(node["start_index"], []).append(node["title"])

    def kept(node, ancestors, nxt, *titles, known=known):
        children = [{"title": title, "start_index": 12} for title in titles]
        return [c["title"] for c in
                tree_optimize.own_children(node, children, lines, known, ancestors, nxt)]

    # on a shared last page: the next node's heading, and what is printed below it
    assert kept(methods, [root], results, "2.3 Limits", "Results", "3.1 Data") == ["2.3 Limits"]
    # a heading that already is a node, wherever it sits on the page and in the tree
    assert kept(methods, [root], None, "2.3 Limits", "Results") == ["2.3 Limits"]
    assert kept(methods, [root], None, "2.3 Limits", "3.1 Data", known={12: ["3.1 Data"]}) == [
        "2.3 Limits"]
    # on a shared first page: what is printed above the node's heading; a number tells headings apart
    assert kept(results, [root], None, "2.3 Limits", "Results", "3.1 Data", "3.2 Results") == [
        "3.1 Data", "3.2 Results"]
    # an intro sits under its parent's heading and above the siblings expand made with it
    data = {"title": "3.1 Data", "start_index": 12, "end_index": 20}
    intro = {"title": "3 Results (intro)", "start_index": 12, "end_index": 12}
    results["nodes"] = [intro, data]
    assert kept(intro, [results, root], data, "2.3 Limits", "Results", "Scope", "3.1 Data") == ["Scope"]
    # ... and knows them through its lineage when this pass made them
    assert kept(intro, [results, root], None, "Results", "Scope", "3.1 Data", known={}) == ["Scope"]
    # a next heading not found on the page decides nothing
    assert kept(methods, [root], dict(results, title="Findings"), "2.3 Limits", "3.2 Results") == [
        "2.3 Limits", "3.2 Results"]


def test_own_children_errs_toward_keeping_where_a_heading_repeats():
    methods = {"title": "2 Methods", "start_index": 4, "end_index": 12}
    results = {"title": "3 Results", "start_index": 12, "end_index": 20}

    def kept(node, nxt, page, *titles):
        lines = [["body"] for _ in range(20)]
        lines[11] = page
        children = [{"title": title, "start_index": 12} for title in titles]
        return [c["title"] for c in tree_optimize.own_children(node, children, lines, {}, [], nxt)]

    # a running header repeats the next heading: the heading is its last line
    assert kept(methods, results, ["3 Results", "2.4 Tail", "3 Results", "3.1 Data"],
                "2.4 Tail", "3.1 Data") == ["2.4 Tail"]
    # a child also mentioned above the node's heading: the child is its last line
    assert kept(results, None, ["see 3.1 Data", "3 Results", "3.1 Data"], "3.1 Data") == ["3.1 Data"]
    # the next heading spelled otherwise, on the very line the proposal is printed on
    assert kept(methods, dict(results, title="IV. Results"), ["Tail", "IV. Results"],
                "Tail", "Results") == ["Tail"]
    # a same-page fusion is found by its headings, not the title the summary pass rewrites
    fused = dict(results, title="Rewritten", _same_page=True, key_items=["3 Results"])
    assert kept(methods, fused, ["2.4 Tail", "3 Results", "3.1 Data"], "2.4 Tail", "3.1 Data") == [
        "2.4 Tail"]


def test_a_number_alone_does_not_name_a_non_latin_heading():
    # normalize keeps no CJK: "1. 概要" reads as "1", "1.1 背景" as "1 1"
    lines = [["body"] for _ in range(20)]
    lines[1] = ["1. 概要", "body", "1.1 背景"]
    lines[11] = ["1.2 目的", "2. 方法", "2.1 データ", "2.2 手順"]
    overview = {"title": "1. 概要", "start_index": 2, "end_index": 12}
    method = {"title": "2. 方法", "start_index": 12, "end_index": 20}
    known = {2: ["1. 概要"], 12: ["2. 方法"]}
    children = [{"title": title, "start_index": page}
                for title, page in [("1.1 背景", 2), ("1.2 目的", 12), ("2.1 データ", 12)]]
    kept = tree_optimize.own_children(overview, children, lines, known, [], method)
    # 1.1 is not "1.", 1.2 is not "2.", and "2.2" is not the line "2." is printed on
    assert [c["title"] for c in kept] == ["1.1 背景", "1.2 目的"]


def test_a_node_sharing_its_first_page_with_its_parent_owns_only_what_follows_its_heading():
    data = {"title": "3.1 Data", "start_index": 4, "end_index": 12}
    analysis = {"title": "3.2 Analysis", "start_index": 12, "end_index": 20}
    results = {"title": "3 Results", "start_index": 4, "end_index": 20, "nodes": [data, analysis]}

    def kept(node, nxt, page, *children):
        lines = [["body"] for _ in range(20)]
        lines[3] = page
        return [c["title"] for c in tree_optimize.own_children(
            node, [{"title": t, "start_index": p} for t, p in children], lines, {}, [results], nxt)]

    # the parent's heading is printed above the node's: what sits between is the parent's
    assert kept(data, analysis, ["3 Results", "Overview", "3.1 Data"],
                ("Overview", 4), ("Data sources", 7)) == ["Data sources"]
    # ... and a sibling's subsection printed above the node's heading is the sibling's
    data["end_index"] = analysis["start_index"] = 4
    assert kept(analysis, None, ["3 Results", "3.1 Data", "3.1.1 Sources", "3.2 Analysis"],
                ("3.1.1 Sources", 4), ("Method", 9)) == ["Method"]


@pytest.mark.parametrize("methods_delay", [0, 0.05])
def test_expand_gives_one_tree_whichever_reply_lands_first(monkeypatch, methods_delay):
    body = "body " * 250
    pages = [body] * 20
    pages[5] = "Setup\n" + body
    pages[11] = body + "\n3 Results\nlead\n3.1 Data\n" + body
    pages[14] = "3.2 Analysis\n" + body
    lines = [[line for line in page.splitlines() if line.strip()] for page in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 20, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "Methods", "start_index": 4, "end_index": 12, "node_id": "0002"},
        {"title": "3 Results", "start_index": 12, "end_index": 20, "node_id": "0003"}]}]

    async def propose(model, prompt):
        if "Section title: Methods\n" in prompt:
            await asyncio.sleep(methods_delay)
            return {"subsections": [{"title": "Setup", "page": 6}, {"title": "3.1 Data", "page": 12}]}
        if "Section title: 3 Results\n" in prompt:
            await asyncio.sleep(0.05 - methods_delay)
            return {"subsections": [{"title": "3.1 Data", "page": 12}, {"title": "3.2 Analysis", "page": 15}]}
        return {"subsections": []}
    monkeypatch.setattr(tree_optimize, "ask_model", propose)

    asyncio.run(tree_optimize.optimize(tree, pages, lines, model="m", do_expand=True))

    assert shape(tree[0]["nodes"][1:]) == [
        ("Methods", 4, 12), ("Methods (intro)", 4, 5), ("Setup", 6, 12),
        ("3 Results", 12, 20), ("3.1 Data", 12, 14), ("3.2 Analysis", 15, 20)]


def test_merge_folds_an_intro_without_listing_its_title():
    tree = [{"title": "P", "start_index": 1, "end_index": 4, "nodes": [
        {"title": "P (intro)", "start_index": 1, "end_index": 1},
        {"title": "C", "start_index": 2, "end_index": 4}]}]
    tree_optimize.merge_tree(tree)
    assert "nodes" not in tree[0] and tree[0]["key_items"] == ["C"]


def test_a_toc_item_whose_page_the_next_opens_still_ends_on_its_own_page():
    items = [{"structure": "1", "title": "Overview", "physical_index": 5},
             {"structure": "2", "title": "Scope", "physical_index": 5, "appear_start": "yes"},
             {"structure": "3", "title": "Terms", "physical_index": 9},
             {"structure": "4", "title": "Annex", "physical_index": 8}]    # listed out of order
    assert shape(utils.post_processing(items, 12)) == [
        ("Overview", 5, 5), ("Scope", 5, 9), ("Terms", 9, 9), ("Annex", 8, 12)]


LARGE = SimpleNamespace(max_page_num_each_node=10, max_token_num_each_node=20000, model="m")


def test_large_node_split_keeps_existing_children(monkeypatch):
    async def meta_processor(*args, **kwargs):
        raise AssertionError("a parent's subsections must not be replaced")
    monkeypatch.setattr(classic, "meta_processor", meta_processor)
    node = {"title": "Ch", "start_index": 1, "end_index": 20,
            "nodes": [{"title": "S", "start_index": 20, "end_index": 25}]}

    asyncio.run(classic.process_large_node_recursively(node, [("page", 5000)] * 25, LARGE))

    assert [child["title"] for child in node["nodes"]] == ["S"]


def test_split_intro_skips_its_parents_heading(monkeypatch):
    async def meta_processor(*args, **kwargs):
        return [{"title": "Ch", "physical_index": 1},
                {"title": "Background", "physical_index": 5},
                {"title": "Scope", "physical_index": 9}]

    async def appear(items, *args, **kwargs):
        for item in items:
            item["appear_start"] = "yes"
        return items
    monkeypatch.setattr(classic, "meta_processor", meta_processor)
    monkeypatch.setattr(classic, "check_title_appearance_in_start_concurrent", appear)
    intro = {"title": "Ch (intro)", "start_index": 1, "end_index": 14}

    asyncio.run(classic.process_large_node_recursively(intro, [("page", 5000)] * 14, LARGE))

    assert [child["title"] for child in intro["nodes"]] == ["Background", "Scope"]


def test_standard_index_stores_intros_covering_ranges_and_section_summaries(tmp_path, monkeypatch):
    texts = ["Opening words"] + [f"page {n}" for n in range(2, 41)]
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(texts))

    async def tree_parser(page_list, opt, doc=None, logger=None):
        return [{"title": "P", "start_index": 1, "end_index": 2, "nodes": [
            {"title": "P (intro)", "start_index": 1, "end_index": 2},
            {"title": "C1", "start_index": 3, "end_index": 20},
            {"title": "C2", "start_index": 21, "end_index": 21, "nodes": [
                {"title": "C2.1", "start_index": 21, "end_index": 30},
                {"title": "C2.2", "start_index": 31, "end_index": 40}]}]}]
    prompts = []

    async def reply(model, prompt):
        prompts.append(prompt)
        return "whole " + prompt.split("Section Title: ")[1].split("\n")[0]
    monkeypatch.setattr(classic, "tree_parser", tree_parser)
    monkeypatch.setattr(utils, "llm_acompletion", reply)
    monkeypatch.setattr(utils, "llm_completion", lambda model, prompt, **kw: "d")
    opt = utils.ConfigLoader().load({"model": "m", "if_add_node_id": "yes",
                                     "if_add_node_summary": "yes", "if_add_node_text": "yes",
                                     "if_add_doc_description": "yes"})

    result = classic.page_index_main(str(pdf), opt, logger=SimpleNamespace(info=print),
                                     page_list=[(text, 1) for text in texts])

    p = result["structure"][0]
    c2 = p["nodes"][2]
    assert shape([p]) == [("P", 1, 40), ("P (intro)", 1, 2), ("C1", 3, 20),
                          ("C2", 21, 40), ("C2.1", 21, 30), ("C2.2", 31, 40)]
    assert (p["text"], p["summary"]) == ("", "whole P")      # its intro holds pages 1-2
    assert "Opening words" in p["nodes"][0]["text"]
    # C2's first child starts on C2's page; C2 keeps that page as its opening
    assert (c2["text"], c2["summary"]) == ("page 21", "whole C2")
    # short leaves keep their own text; only the parents are asked, deepest first
    assert [n["summary"] for n in p["nodes"][:2]] == [
        "Opening wordspage 2", "".join(f"page {n}" for n in range(3, 21))]
    assert [q.split("Section Title: ")[1].split("\n")[0] for q in prompts] == ["C2", "P"]
    assert '"summary": "whole C2"' in prompts[-1]


def test_standard_index_gives_toc_and_split_parents_intros(tmp_path, monkeypatch):
    texts = [f"page {n}" for n in range(1, 41)]
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(texts))

    splits = {1: [{"structure": "1", "title": "P", "physical_index": 1},       # P's long opening
                  {"structure": "2", "title": "Background", "physical_index": 8}],
              20: [{"structure": "1", "title": "C2", "physical_index": 20},    # the large leaf C2
                   {"structure": "2", "title": "C2.a", "physical_index": 25},
                   {"structure": "3", "title": "C2.b", "physical_index": 32}]}

    async def meta_processor(page_list, mode=None, start_index=1, **kwargs):
        if len(page_list) == len(texts):
            return [{"structure": "1", "title": "P", "physical_index": 1},
                    {"structure": "1.1", "title": "C1", "physical_index": 15},
                    {"structure": "1.2", "title": "C2", "physical_index": 20}]
        return splits[start_index]

    async def appear(items, *args, **kwargs):
        return items
    monkeypatch.setattr(classic, "check_toc", lambda page_list, opt: {"toc_content": None})
    monkeypatch.setattr(classic, "meta_processor", meta_processor)
    monkeypatch.setattr(classic, "check_title_appearance_in_start_concurrent", appear)
    opt = utils.ConfigLoader().load({"model": "m", "if_add_node_summary": "no"})

    result = classic.page_index_main(str(pdf), opt, logger=SimpleNamespace(info=print),
                                     page_list=[(text, 3000) for text in texts])

    assert shape(result["structure"]) == [
        ("P", 1, 40), ("P (intro)", 1, 15), ("P (intro)", 1, 8), ("Background", 8, 15),
        ("C1", 15, 20),
        ("C2", 20, 40), ("C2 (intro)", 20, 25), ("C2.a", 25, 32), ("C2.b", 32, 40)]


def test_flash_gives_parents_their_intro_nodes(tmp_path, monkeypatch):
    import pageindex.flash.api as flash_api
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["x"]))
    monkeypatch.setattr(flash_api, "extract_toc", lambda pdf, **kw: {
        "structure": [{"title": "R", "node_id": "0000", "start_index": 1, "end_index": 6, "nodes": [
            {"title": "A", "node_id": "0001", "start_index": 3, "end_index": 4},
            {"title": "X", "node_id": "0002", "start_index": 5, "end_index": 6}]}],
        "page_texts": ["Cover", "Foreword", "A\nalpha", "alpha", "X\nbody", "body"]})

    result = flash_api.page_index_flash(str(pdf), summary=False, optimize=False)

    assert [(n["title"], n["node_id"], n["start_index"], n["end_index"])
            for n in utils._subtree(result["structure"])] == [
        ("R", "0000", 1, 6), ("R (intro)", "0001", 1, 2), ("A", "0002", 3, 4),
        ("X", "0003", 5, 6)]


def test_flash_preface_runs_onto_a_page_its_first_heading_does_not_open(tmp_path, monkeypatch):
    import pageindex.flash.api as flash_api
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(build_pdf(["x"]))
    ends = []
    for third in ["A\nalpha", "Contents, continued\nA\nalpha"]:
        monkeypatch.setattr(flash_api, "extract_toc", lambda pdf, third=third, **kw: {
            "structure": [{"title": "A", "node_id": "0000", "start_index": 3, "end_index": 4}],
            "page_texts": ["Cover", "Contents", third, "alpha"]})
        preface = flash_api.page_index_flash(str(pdf), summary=False, optimize=False)["structure"][0]
        ends.append((preface["title"], preface["node_id"], preface["end_index"]))
    assert ends == [("Preface", "0000", 2), ("Preface", "0000", 3)]


def test_summarize_tree_parent_falls_back_to_its_subsection_titles(monkeypatch):
    async def reply(model, prompt):
        return "" if "Section Title" in prompt else '{"summary": "ok"}'
    monkeypatch.setattr(utils, "llm_acompletion", reply)
    structure = [{"title": "R", "start_index": 1, "end_index": 2, "nodes": [
        {"title": "A", "start_index": 1, "end_index": 1},
        {"title": "B", "start_index": 2, "end_index": 2}]}]

    out = asyncio.run(utils.summarize_tree(structure, [("alpha " * 300, 0), ("beta " * 300, 0)]))

    assert [n["summary"] for n in utils._subtree(out)] == ["A; B", "ok", "ok"]


def test_get_tree_parent_text_shares_its_first_childs_page_unless_an_intro_holds_it(
        tmp_path, monkeypatch):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(build_pdf(["Opening words", "Alpha body", "Beta body"]))
    structure = [{"title": "Report", "node_id": "0000", "start_index": 1, "end_index": 3,
                  "summary": "report", "nodes": [
                      {"title": "Report (intro)", "node_id": "0001", "start_index": 1,
                       "end_index": 1, "summary": "opening"},
                      {"title": "Alpha", "node_id": "0002", "start_index": 2,
                       "end_index": 3, "summary": "alpha", "nodes": [
                           {"title": "Alpha one", "node_id": "0003", "start_index": 2,
                            "end_index": 2, "summary": "one"},
                           {"title": "Alpha two", "node_id": "0004", "start_index": 3,
                            "end_index": 3, "summary": "two"}]}]}]
    monkeypatch.setattr(pageindex.flash, "page_index_flash", lambda pdf, **kw: {
        "doc_name": "report.pdf", "structure": structure})
    monkeypatch.setattr(utils, "llm_completion", lambda model, prompt, **kw: "d")
    client = PageIndexClient(storage_path=str(tmp_path / "store"))
    doc_id = client.submit_document(str(pdf), mode="flash")["doc_id"]

    root = client.get_tree(doc_id)["result"][0]

    assert [n["text"] for n in utils._subtree([root])] == [
        "", "Opening words", "Alpha body", "Alpha body", "Beta body"]
