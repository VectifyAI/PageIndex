# tests/sdk/test_pipeline.py
import asyncio
from unittest.mock import patch, AsyncMock

from pageindex.parser.protocol import ContentNode, ParsedDocument
from pageindex.index.pipeline import (
    detect_strategy, build_tree_from_levels, build_index,
    _content_based_pipeline, _NullLogger,
)


def test_detect_strategy_with_level():
    nodes = [
        ContentNode(content="# Intro", tokens=10, title="Intro", index=1, level=1),
        ContentNode(content="## Details", tokens=10, title="Details", index=5, level=2),
    ]
    assert detect_strategy(nodes) == "level_based"


def test_detect_strategy_without_level():
    nodes = [
        ContentNode(content="Page 1 text", tokens=100, index=1),
        ContentNode(content="Page 2 text", tokens=100, index=2),
    ]
    assert detect_strategy(nodes) == "content_based"


def test_detect_strategy_empty_nodes_is_level_based():
    """An empty node list (e.g. an empty/whitespace-only source file) must
    route to level_based, whose build_tree_from_levels([]) returns an empty
    structure with zero LLM calls — not content_based, whose TOC-detection
    pipeline needs real page content and wastes an LLM call before failing."""
    assert detect_strategy([]) == "level_based"


def test_build_index_on_empty_document_makes_no_llm_calls():
    from pageindex.config import IndexConfig
    parsed = ParsedDocument(doc_name="empty", nodes=[])
    opt = IndexConfig(if_add_node_summary=False, if_add_doc_description=False)
    result = build_index(parsed, opt=opt)
    assert result == {"doc_name": "empty", "structure": []}


def test_build_tree_from_levels():
    nodes = [
        ContentNode(content="ch1 text", tokens=10, title="Chapter 1", index=1, level=1),
        ContentNode(content="s1.1 text", tokens=10, title="Section 1.1", index=5, level=2),
        ContentNode(content="s1.2 text", tokens=10, title="Section 1.2", index=10, level=2),
        ContentNode(content="ch2 text", tokens=10, title="Chapter 2", index=20, level=1),
    ]
    tree = build_tree_from_levels(nodes)
    assert len(tree) == 2  # 2 root nodes (chapters)
    assert tree[0]["title"] == "Chapter 1"
    assert len(tree[0]["nodes"]) == 2  # 2 sections under chapter 1
    assert tree[0]["nodes"][0]["title"] == "Section 1.1"
    assert tree[0]["nodes"][1]["title"] == "Section 1.2"
    assert tree[1]["title"] == "Chapter 2"
    assert len(tree[1]["nodes"]) == 0


def test_build_tree_from_levels_single_level():
    nodes = [
        ContentNode(content="a", tokens=5, title="A", index=1, level=1),
        ContentNode(content="b", tokens=5, title="B", index=2, level=1),
    ]
    tree = build_tree_from_levels(nodes)
    assert len(tree) == 2
    assert tree[0]["title"] == "A"
    assert tree[1]["title"] == "B"


def test_build_tree_from_levels_zero_based_levels():
    nodes = [
        ContentNode(content="c", tokens=5, title="Chapter", index=1, level=0),
        ContentNode(content="s", tokens=5, title="Section", index=2, level=1),
    ]
    tree = build_tree_from_levels(nodes)
    assert len(tree) == 1
    assert tree[0]["title"] == "Chapter"
    assert [n["title"] for n in tree[0]["nodes"]] == ["Section"]


def test_build_tree_from_levels_deep_nesting():
    nodes = [
        ContentNode(content="h1", tokens=5, title="H1", index=1, level=1),
        ContentNode(content="h2", tokens=5, title="H2", index=2, level=2),
        ContentNode(content="h3", tokens=5, title="H3", index=3, level=3),
    ]
    tree = build_tree_from_levels(nodes)
    assert len(tree) == 1
    assert tree[0]["title"] == "H1"
    assert len(tree[0]["nodes"]) == 1
    assert tree[0]["nodes"][0]["title"] == "H2"
    assert len(tree[0]["nodes"][0]["nodes"]) == 1
    assert tree[0]["nodes"][0]["nodes"][0]["title"] == "H3"


def test_content_based_pipeline_does_not_raise():
    """_content_based_pipeline should delegate to tree_parser, not raise NotImplementedError."""
    fake_tree = [{"title": "Intro", "start_index": 1, "end_index": 2, "nodes": []}]

    async def fake_tree_parser(page_list, opt, doc=None, logger=None):
        return fake_tree

    page_list = [("Page 1 text", 50), ("Page 2 text", 60)]

    from types import SimpleNamespace
    opt = SimpleNamespace(model="test-model")

    with patch("pageindex.index.page_index.tree_parser", new=fake_tree_parser):
        result = asyncio.run(_content_based_pipeline(page_list, opt))

    assert result == fake_tree


def test_null_logger_methods():
    """NullLogger should have info/error/debug and not raise."""
    logger = _NullLogger()
    logger.info("test message")
    logger.error("test error")
    logger.debug("test debug")
    logger.info({"key": "value"})


def _structure_has_text(nodes) -> bool:
    for n in nodes:
        if "text" in n:
            return True
        if n.get("nodes") and _structure_has_text(n["nodes"]):
            return True
    return False


def test_level_based_strips_text_by_default():
    """Markdown (level_based) must honor if_add_node_text=False — build_tree_from_
    levels seeds 'text', and it used to leak into the output/storage."""
    from pageindex.config import IndexConfig
    nodes = [
        ContentNode(content="# Intro\nbody one", tokens=5, title="Intro", index=1, level=1),
        ContentNode(content="## Sub\nbody two", tokens=5, title="Sub", index=2, level=2),
    ]
    parsed = ParsedDocument(doc_name="d", nodes=nodes)
    # No summary/description -> no LLM calls.
    opt = IndexConfig(if_add_node_summary=False, if_add_doc_description=False,
                      if_add_node_text=False)
    result = build_index(parsed, opt=opt)
    assert not _structure_has_text(result["structure"])


def test_level_based_keeps_text_when_requested():
    from pageindex.config import IndexConfig
    nodes = [ContentNode(content="# Intro\nbody", tokens=5, title="Intro", index=1, level=1)]
    parsed = ParsedDocument(doc_name="d", nodes=nodes)
    opt = IndexConfig(if_add_node_summary=False, if_add_doc_description=False,
                      if_add_node_text=True)
    result = build_index(parsed, opt=opt)
    assert _structure_has_text(result["structure"])


def test_build_index_scopes_llm_params_to_the_call(monkeypatch):
    """IndexConfig(llm_params=...) must reach get_llm_params() for the duration
    of this build_index() call only, and not leak into the process default."""
    from pageindex.config import IndexConfig, get_llm_params, set_llm_params

    set_llm_params(temperature=0)
    seen = {}

    async def fake_generate_summaries(structure, model=None):
        seen["llm_params"] = get_llm_params()

    monkeypatch.setattr(
        "pageindex.index.utils.generate_summaries_for_structure",
        fake_generate_summaries,
    )

    # level_based (Markdown) strategy avoids the content_based path's own real
    # LLM-driven TOC detection, so this stays a fast, network-free unit test.
    nodes = [ContentNode(content="# Intro\nbody", tokens=5, title="Intro", index=1, level=1)]
    parsed = ParsedDocument(doc_name="d", nodes=nodes)
    opt = IndexConfig(if_add_node_summary=True, if_add_doc_description=False,
                      llm_params={"temperature": 1})
    build_index(parsed, opt=opt)

    assert seen["llm_params"]["temperature"] == 1   # scoped override was in effect
    assert get_llm_params()["temperature"] == 0     # process default untouched afterward


def test_check_title_appearance_tolerates_out_of_range_physical_index():
    """An LLM-emitted physical_index outside page_list must be marked 'no', not
    raise IndexError (which happens during task construction, outside the
    gather's return_exceptions protection, and would abort the whole build)."""
    from pageindex.index.page_index import check_title_appearance_in_start_concurrent

    page_list = [("only page text", 3)]  # length 1
    structure = [
        {"title": "A", "physical_index": 5},     # out of range -> would IndexError
        {"title": "B", "physical_index": 0},     # 0 -> would wrap to page_list[-1]
        {"title": "C", "physical_index": None},  # missing
        {"title": "D"},                          # no physical_index key at all
    ]
    result = asyncio.run(
        check_title_appearance_in_start_concurrent(structure, page_list)
    )
    assert all(item["appear_start"] == "no" for item in result)
