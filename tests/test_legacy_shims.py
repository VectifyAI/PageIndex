"""The top-level pageindex.page_index / .page_index_md / .utils modules are
now deprecation shims over the canonical pageindex.index.* modules. These
tests pin the compatibility contract."""
import asyncio
import importlib
import warnings

import pytest


def test_plain_import_pageindex_does_not_warn():
    # `import pageindex` must not route through the deprecation shims.
    with warnings.catch_warnings():
        warnings.simplefilter("error", PendingDeprecationWarning)
        importlib.import_module("pageindex")


@pytest.mark.parametrize("mod", [
    "pageindex.utils",
    "pageindex.page_index",
    "pageindex.page_index_md",
])
def test_legacy_submodule_import_warns(mod):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(importlib.import_module(mod))
    assert any(issubclass(w.category, PendingDeprecationWarning) for w in caught)


def test_legacy_symbols_resolve_through_shims():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pageindex.utils import (  # noqa: F401
            get_page_tokens, ConfigLoader, convert_page_to_int,
            get_leaf_nodes, remove_fields,
        )
        from pageindex.page_index import page_index, page_index_main  # noqa: F401
        from pageindex.page_index_md import md_to_tree  # noqa: F401


def test_canonical_and_shim_share_one_implementation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pageindex.utils as shim
    import pageindex.index.utils as canonical
    # Same function object -> a single source of truth (no divergence possible).
    assert shim.get_leaf_nodes is canonical.get_leaf_nodes
    assert shim.get_page_tokens is canonical.get_page_tokens


def test_get_leaf_nodes_has_331_fix():
    """Canonical get_leaf_nodes must use .get('nodes'); clean_node deletes the
    key on leaf nodes so [...]['nodes'] would KeyError (issue #330)."""
    from pageindex.index.utils import get_leaf_nodes
    # A leaf node with the 'nodes' key deleted (as clean_node leaves it).
    leaves = get_leaf_nodes({"title": "Leaf", "start_index": 1, "end_index": 2})
    assert leaves == [{"title": "Leaf", "start_index": 1, "end_index": 2}]


def test_configloader_no_longer_needs_config_yaml():
    """config.yaml was removed; ConfigLoader must build defaults from IndexConfig."""
    from pageindex.index.utils import ConfigLoader
    cfg = ConfigLoader().load({"model": "gpt-5.4"})
    assert cfg.model == "gpt-5.4"
    assert cfg.if_add_node_summary is True  # IndexConfig default
    with pytest.raises(ValueError, match="Unknown config keys"):
        ConfigLoader().load({"nope": 1})


def test_md_to_tree_shim_is_the_canonical_function():
    """The shim no longer wraps md_to_tree with its own coercion — the
    canonical implementation coerces internally, so the shim is a pure
    re-export (single source of truth, can't diverge from the canonical
    behavior)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pageindex.page_index_md as shim
    import pageindex.index.page_index_md as canonical
    assert shim.md_to_tree is canonical.md_to_tree


def test_md_to_tree_coerces_legacy_yes_no_strings(tmp_path):
    """A bare 'no' must not read as truthy True — exercised end-to-end (no
    LLM calls needed with summary/description disabled)."""
    from pageindex.index.page_index_md import md_to_tree

    md_path = tmp_path / "doc.md"
    md_path.write_text("# Title\nbody\n\n## Sub\nmore body\n")

    result = asyncio.run(md_to_tree(
        md_path=str(md_path),
        if_add_node_summary="no",
        if_add_node_id="yes",
        if_add_doc_description="no",
    ))
    assert "doc_description" not in result

    def _has_summary(nodes):
        return any("summary" in n or (n.get("nodes") and _has_summary(n["nodes"]))
                   for n in nodes)

    assert not _has_summary(result["structure"])
    assert all("node_id" in n for n in result["structure"])
