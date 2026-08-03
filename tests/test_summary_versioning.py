"""Tests for node-level text/summary versioning and deferred regeneration.

The summarizer is stubbed, so these run without an API key and without
spending anything. Covers: version stamping, staleness detection, batching
across repeated updates, monotonicity across re-index, and the invariant
that a reconciled read leaves nothing stale.

Run: python tests/test_summary_versioning.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pageindex.client as client_mod
from pageindex import PageIndexClient
from pageindex.utils import walk_with_paths

V1 = "# Root\nroot intro\n## A\nalpha text\n## B\nbeta text\n"

CALLS = []


async def _fake_summary(node, summary_token_threshold=200, model=None):
    """Deterministic stand-in for get_node_summary — records what it is asked to do."""
    CALLS.append(node.get("title"))
    return f"SUMMARY({node.get('title')})@{node.get('text_version')}"


class _Harness:
    """Client with summarization stubbed out at every call site."""

    def __enter__(self):
        self.dir = Path(tempfile.mkdtemp())
        self._orig_client = client_mod.get_node_summary
        client_mod.get_node_summary = _fake_summary
        # index() summarizes via md_to_tree -> generate_summaries_for_structure_md,
        # and also generates a doc description. Both would hit the network.
        import pageindex.page_index_md as md_mod
        self._orig_md = md_mod.get_node_summary
        self._orig_desc = md_mod.generate_doc_description
        md_mod.get_node_summary = _fake_summary
        md_mod.generate_doc_description = lambda structure, model=None: "DESC"
        self._md_mod = md_mod
        CALLS.clear()
        return self

    def __exit__(self, *a):
        client_mod.get_node_summary = self._orig_client
        self._md_mod.get_node_summary = self._orig_md
        self._md_mod.generate_doc_description = self._orig_desc
        shutil.rmtree(self.dir, ignore_errors=True)

    def client(self):
        return PageIndexClient(workspace=str(self.dir))

    def write(self, text):
        p = self.dir / "d.md"
        p.write_text(text, encoding="utf-8")
        return str(p)


def _versions(c, doc_id):
    c._ensure_doc_loaded(doc_id)
    return {
        p: (n.get("text_version"), n.get("summary_version"))
        for p, n in walk_with_paths(c.documents[doc_id]["structure"])
    }


def _stale(c, doc_id):
    return {p for p, (tv, sv) in _versions(c, doc_id).items() if tv != sv}


def test_index_stamps_versions_and_nothing_is_stale():
    with _Harness() as h:
        c = h.client()
        doc_id = c.index(h.write(V1))
        vs = _versions(c, doc_id)
        assert vs, "structure should not be empty"
        assert all(v == (1, 1) for v in vs.values()), vs
        assert _stale(c, doc_id) == set()


def test_update_bumps_text_version_only_for_changed_nodes():
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)
        Path(path).write_text(V1.replace("alpha text", "alpha CHANGED"), encoding="utf-8")

        CALLS.clear()
        c.update(doc_id)

        assert CALLS == [], f"update() must not summarize, but called {CALLS}"
        vs = _versions(c, doc_id)
        assert vs["Root > A"] == (2, 1), vs        # changed -> stale
        assert vs["Root > B"] == (1, 1), vs        # untouched
        assert vs["Root"] == (1, 1), vs            # no propagation to parent
        assert _stale(c, doc_id) == {"Root > A"}


def test_repeated_updates_batch_into_one_regeneration():
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)

        for i in range(3):
            Path(path).write_text(V1.replace("alpha text", f"alpha v{i}"), encoding="utf-8")
            c.update(doc_id)

        vs = _versions(c, doc_id)
        assert vs["Root > A"] == (4, 1), vs        # 3 edits -> 3 bumps, summary still at 1

        CALLS.clear()
        n = c._reconcile_summaries(doc_id)
        assert n == 1, n
        assert CALLS == ["A"], CALLS               # one call, not three
        assert _stale(c, doc_id) == set()


def test_reconcile_is_idempotent():
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)
        Path(path).write_text(V1.replace("beta text", "beta CHANGED"), encoding="utf-8")
        c.update(doc_id)

        assert c._reconcile_summaries(doc_id) == 1
        CALLS.clear()
        assert c._reconcile_summaries(doc_id) == 0, "second read must regenerate nothing"
        assert CALLS == []


def test_added_node_starts_stale_and_gets_a_summary():
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)
        Path(path).write_text(V1 + "## C\ngamma text\n", encoding="utf-8")
        c.update(doc_id)

        vs = _versions(c, doc_id)
        assert vs["Root > C"] == (1, 0), vs         # no summary yet
        assert "Root > C" in _stale(c, doc_id)

        c._reconcile_summaries(doc_id)
        c._ensure_doc_loaded(doc_id)
        node = dict(walk_with_paths(c.documents[doc_id]["structure"]))["Root > C"]
        assert node["summary"] == "SUMMARY(C)@1", node
        assert _stale(c, doc_id) == set()


def test_structure_is_not_empty_after_reconciling_read():
    """_save_doc evicts structure for lazy reload; a reconciling read must not
    hand the agent an empty tree."""
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)
        Path(path).write_text(V1.replace("alpha text", "alpha CHANGED"), encoding="utf-8")
        c.update(doc_id)
        assert _stale(c, doc_id), "precondition: something must be stale"

        payload = json.loads(c.get_document_structure(doc_id))
        assert payload, f"reconciling read returned empty structure: {payload!r}"
        titles = [t for t, _ in walk_with_paths(payload)]
        assert "Root > A" in titles, titles


def test_reindex_does_not_regress_versions():
    """Re-indexing reuses the doc_id; a reader at version N must not see it drop."""
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)
        Path(path).write_text(V1.replace("alpha text", "alpha CHANGED"), encoding="utf-8")
        c.update(doc_id)
        before = _versions(c, doc_id)["Root > A"][0]
        assert before == 2

        # Full re-index of the same path.
        same_id = c.index(path)
        assert same_id == doc_id
        after = _versions(c, doc_id)["Root > A"]
        assert after[0] >= before, f"version regressed: {before} -> {after[0]}"
        assert after[0] == after[1], "re-index regenerates everything, so nothing is stale"


def test_reindex_after_edit_bumps_changed_node():
    with _Harness() as h:
        c = h.client()
        path = h.write(V1)
        doc_id = c.index(path)
        Path(path).write_text(V1.replace("alpha text", "alpha CHANGED"), encoding="utf-8")
        c.index(path)

        vs = _versions(c, doc_id)
        assert vs["Root > A"] == (2, 2), vs         # changed -> bumped, summary fresh
        assert vs["Root > B"] == (1, 1), vs         # untouched -> held


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
