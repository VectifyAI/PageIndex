import asyncio
import json

import pytest

from pageindex.library import summaries


def leaf(tree, node_id):
    from pageindex.utils import structure_to_list
    return next(n for n in structure_to_list(tree) if n["node_id"] == node_id)


def test_summary_tier_bottom_up_and_fields(sample_tree, sample_pages, fake_llm):
    result = asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="Book", small_node_tokens=0))
    assert result["generated"] == 4 and result["failed"] == 0
    for node_id in ("0000", "0001", "0002", "0003"):
        node = leaf(sample_tree, node_id)
        assert node["summary"].startswith("S") and node["summary_model"] == "m"
    # parent prompt carried its children's summaries as JSON
    parent_prompt = next(p for m, p in fake_llm if "Subsections (JSON)" in p)
    assert leaf(sample_tree, "0001")["summary"] in parent_prompt
    # leaf prompt carried exactly its pages
    leaf_prompt = next(p for m, p in fake_llm if "word2" in p)
    assert "word3" in leaf_prompt and "word4" not in leaf_prompt


def test_small_leaves_use_raw_text_without_a_call(sample_tree, sample_pages, fake_llm):
    asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="B", small_node_tokens=10_000))
    assert leaf(sample_tree, "0002")["summary"] == "Page 4 text word4."
    assert leaf(sample_tree, "0002")["summary_model"] == "raw"


def test_small_leaves_rule_counts_calls_precisely(sample_tree, sample_pages, fake_llm):
    # Chapter Two is a leaf (2 pages) -> raw; Section A, B raw; Chapter One parent -> 1 call
    asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="B", small_node_tokens=10_000))
    assert len(fake_llm) == 1


def test_digest_tier_is_plain_markdown_and_never_raw(sample_tree, sample_pages, fake_llm):
    asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="digest", profile="nonfiction",
        model="m", book="B", small_node_tokens=10_000))
    assert len(fake_llm) == 4
    assert leaf(sample_tree, "0002")["digest"].startswith('{"points"')  # fake reply verbatim
    assert leaf(sample_tree, "0002")["digest_model"] == "m"
    assert "summary" not in leaf(sample_tree, "0002")


def test_resume_skips_nodes_that_already_have_the_field(sample_tree, sample_pages, fake_llm):
    leaf(sample_tree, "0001")["summary"] = "kept"
    result = asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="B", small_node_tokens=0))
    assert leaf(sample_tree, "0001")["summary"] == "kept"
    assert result["skipped"] == 1 and result["generated"] == 3


def test_force_regenerates_and_node_ids_restrict(sample_tree, sample_pages, fake_llm):
    leaf(sample_tree, "0001")["summary"] = "old"
    result = asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="B", small_node_tokens=0, force=True, node_ids={"0001"}))
    assert leaf(sample_tree, "0001")["summary"] != "old"
    assert result["generated"] == 1 and "summary" not in leaf(sample_tree, "0003")


def test_on_node_called_after_each_write(sample_tree, sample_pages, fake_llm):
    seen = []
    asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="B", small_node_tokens=0, on_node=lambda n: seen.append(n["node_id"])))
    assert sorted(seen) == ["0000", "0001", "0002", "0003"]
    assert seen.index("0000") > seen.index("0001")   # parent after its children


def test_failed_node_is_left_absent_and_reported(sample_tree, sample_pages, monkeypatch):
    async def flaky(model, prompt):
        if "word5" in prompt:
            raise RuntimeError("boom")
        return '{"summary": "ok"}'
    monkeypatch.setattr("pageindex.utils.llm_acompletion", flaky)
    result = asyncio.run(summaries.summarize_tier(
        sample_tree, sample_pages, tier="summary", profile="nonfiction",
        model="m", book="B", small_node_tokens=0))
    assert result["failed"] == 1 and "boom" in result["errors"][0]
    assert "summary" not in leaf(sample_tree, "0003")
    assert leaf(sample_tree, "0000")["summary"] == "ok"


def test_not_logged_in_aborts_the_run(sample_tree, sample_pages, monkeypatch):
    from pageindex.backends.base import CliBackendError

    async def dead(model, prompt):
        raise CliBackendError("Not logged in", status_code=401, retryable=False)
    monkeypatch.setattr("pageindex.utils.llm_acompletion", dead)
    with pytest.raises(CliBackendError):
        asyncio.run(summaries.summarize_tier(
            sample_tree, sample_pages, tier="summary", profile="nonfiction",
            model="m", book="B", small_node_tokens=0))


def test_unrecoverable_error_aborts_immediately_not_deferred(monkeypatch):
    """Verify that an unrecoverable error (401) causes early termination,
    not full tree traversal. The fix raises immediately instead of deferring
    via a fatal list, reducing unnecessary LLM calls after login failure."""
    from pageindex.backends.base import CliBackendError

    # Build a tree with clear node count: 2 chapters × 8 leaves each = 16 leaves + 2 parents = 18 total
    # If old code ran (deferred exception), would make ~18 calls before raising
    # If fixed code runs (early abort), should make much fewer calls
    structure = [
        {
            "node_id": "ch1",
            "title": "Chapter 1",
            "start_index": 0,
            "end_index": 8,
            "nodes": [
                {
                    "node_id": f"s1_{i}",
                    "title": f"Section 1.{i}",
                    "start_index": i,
                    "end_index": i,
                }
                for i in range(8)
            ],
        },
        {
            "node_id": "ch2",
            "title": "Chapter 2",
            "start_index": 8,
            "end_index": 16,
            "nodes": [
                {
                    "node_id": f"s2_{i}",
                    "title": f"Section 2.{i}",
                    "start_index": 8 + i,
                    "end_index": 8 + i,
                }
                for i in range(8)
            ],
        },
    ]
    total_nodes = 2 + 16  # 2 chapter parents + 16 sections
    page_texts = [f"Page {i} text" for i in range(16)]

    # Use a list for call tracking (list.append is atomic under GIL, thread-safe)
    calls = []

    async def flaky(model, prompt):
        calls.append(1)
        # Yield control to let event loop interleave concurrent tasks, like real I/O would.
        # Without this, tasks run synchronously in sequence, masking real concurrency behavior.
        await asyncio.sleep(0)
        # Fail on Section 1.2, early in the tree (before most leaves)
        # This ensures clear separation: buggy code ~18 calls, fixed code <10
        if "Section 1.2" in prompt:
            raise CliBackendError("Not logged in", status_code=401, retryable=False)
        return '{"summary": "ok"}'

    monkeypatch.setattr("pageindex.utils.llm_acompletion", flaky)

    # With the fix, unrecoverable error is raised immediately and stops processing.
    # With the old buggy code (deferred via fatal list), it would process the
    # entire tree (all 18 nodes) then raise.
    with pytest.raises(CliBackendError):
        asyncio.run(summaries.summarize_tier(
            structure, page_texts, tier="summary", profile="nonfiction",
            model="m", book="B", small_node_tokens=0))

    # The key assertion: we made far fewer calls than total nodes.
    # With yield point, concurrent interleaving is real and measurable:
    # - Fixed code (with immediate raise): ~9 calls (stops after Section 1.2 error)
    # - Buggy code (with fatal list): ~16 calls (most of tree walked before raise)
    # Setting threshold at 15 to clearly distinguish both behaviors.
    assert len(calls) < 15, \
        f"Expected early abort (<15 calls) but made {len(calls)} calls"


def test_summarize_book_checkpoints_and_marks_meta(store, sample_tree, sample_pages, fake_llm):
    meta = {"id": "pi-x", "name": "b.pdf", "description": None, "status": "indexed",
            "createdAt": "2026-08-30T00:00:00.000000", "pageNum": 6, "folderId": None,
            "metadata": {"title": "B", "profile": "nonfiction"}, "mode": "flash"}
    store.save_document("pi-x", meta, sample_tree,
                        [{"page_index": i + 1, "markdown": t} for i, t in enumerate(sample_pages)])
    result = summaries.summarize_book(store, "pi-x", tier="summary", model="m",
                                      small_node_tokens=0)
    assert result["generated"] == 4
    tree = store.get_tree("pi-x")
    assert all("summary" in n for n in tree)
    assert store.get_meta("pi-x")["metadata"]["summary_tier_done"] is True


def test_describe_book_uses_description_template(store, sample_tree, sample_pages, fake_llm):
    meta = {"id": "pi-y", "name": "b.pdf", "description": None, "status": "indexed",
            "createdAt": "2026-08-30T00:00:00.000000", "pageNum": 6, "folderId": None,
            "metadata": {"title": "The Book", "profile": "nonfiction"}, "mode": "flash"}
    store.save_document("pi-y", meta, sample_tree, [])
    text = summaries.describe_book(store, "pi-y", model="m")
    assert text.startswith("D1 via m")
    assert store.get_meta("pi-y")["description"] == text
    assert "The Book" in fake_llm[0][1] and "Chapter One" in fake_llm[0][1]
