"""The tree every local index writes: intro nodes, covering ranges, leaf-only
splitting, and summaries every node gets, a parent's built from its children's."""

import asyncio
import importlib
from types import SimpleNamespace

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
    assert tree_optimize.add_intro_nodes(tree, lines) == tree
    # a standard parent ends where its first child starts; a split intro keeps its title
    split = [{"title": "Ch 3 (intro)", "start_index": 10, "end_index": 11, "nodes": [
        {"title": "Background", "start_index": 12, "end_index": 14}]}]
    assert shape(tree_optimize.add_intro_nodes(split))[1] == ("Ch 3 (intro)", 10, 11)
    # a heading with nothing to match cannot be placed on its page, so the page is shared
    assert not tree_optimize.heading_at_page_start([["第一章 总则"]], 1, "第一章 总则")


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


def test_merge_folds_an_intro_without_listing_its_title():
    tree = [{"title": "P", "start_index": 1, "end_index": 4, "nodes": [
        {"title": "P (intro)", "start_index": 1, "end_index": 1},
        {"title": "C", "start_index": 2, "end_index": 4}]}]
    tree_optimize.merge_tree(tree)
    assert "nodes" not in tree[0] and tree[0]["key_items"] == ["C"]


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
        if "Section Title: " in prompt:
            return "whole " + prompt.split("Section Title: ")[1].split("\n")[0]
        return "leaf"
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
    assert any("Section Title: C2" in q and "Opening Text: page 21" in q for q in prompts)
    assert [n["summary"] for n in p["nodes"]] == ["leaf", "leaf", "whole C2"]
    assert '"summary": "whole C2"' in prompts[-1]    # deepest first: P is asked last


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
