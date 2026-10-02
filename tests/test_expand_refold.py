"""An expanded node must survive the next round's merge: its children were already marked final."""
import asyncio

import pageindex.tree_optimize as tree_optimize


def _run(subsections, mid_page=()):
    body = "body " * 250
    pages = [body] * 12
    for title, page in subsections:
        pages[page - 1] = (f"{body}\n{title}\n" if title in mid_page else f"{title}\n") + body
    lines = [[l for l in p.splitlines() if l.strip()] for p in pages]
    tree = [{"title": "R", "start_index": 1, "end_index": 12, "node_id": "0000", "nodes": [
        {"title": "A", "start_index": 1, "end_index": 3, "node_id": "0001"},
        {"title": "X", "start_index": 4, "end_index": 9, "node_id": "0002"},
        {"title": "B", "start_index": 10, "end_index": 12, "node_id": "0003"}]}]

    async def ask(model, prompt):
        return {"subsections": [{"title": t, "page": p} for t, p in subsections]}
    finals = {}

    def on_final(nodes):
        for node in nodes:
            finals.setdefault(id(node), (node, tuple(id(c) for c in node.get("nodes") or [])))
    saved, tree_optimize.ask_model = tree_optimize.ask_model, ask
    try:
        asyncio.run(tree_optimize.optimize(tree, pages, lines, model="m", on_final=on_final,
                                           do_relabel=False))
    finally:
        tree_optimize.ask_model = saved
    live = {id(n) for n, _ in tree_optimize.flatten(tree)}
    broken = [n["title"] for n, kids in finals.values()
              if id(n) not in live or tuple(id(c) for c in n.get("nodes") or []) != kids]
    return broken, tree[0]["nodes"][1]


def test_a_node_whose_intro_runs_onto_the_first_child_page_stays_collapsed():
    # X spans 6 pages; the first heading sits mid-page 8, so X's intro covers pages 4-8
    # and the expanded X is no cheaper than a scan: the next merge would fold it back
    broken, x = _run([("Deep Recursion", 8), ("Final Step", 9)], mid_page={"Deep Recursion"})
    assert broken == [] and "nodes" not in x


def test_a_node_that_pays_off_is_still_expanded():
    broken, x = _run([("Cost", 4), ("Depth", 6), ("Error", 8)])
    assert broken == [] and [c["title"] for c in x["nodes"]] == ["Cost", "Depth", "Error"]
