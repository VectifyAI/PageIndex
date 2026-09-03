"""What `pip install pageindex` exposes: 0.2.8 helper compat and import cost."""
import os
import subprocess
import sys

from pageindex.utils import create_node_mapping, print_tree, remove_fields

TREE = [
    {"title": "Root", "node_id": "0000", "page_index": 1,
     "text": "root text",
     "nodes": [
         {"title": "Child", "node_id": "0001", "page_index": 3,
          "text": "child text"},
     ]},
    {"title": "Tail", "node_id": "0002", "page_index": 5, "text": "tail text"},
]


# ── the published 0.2.8 pageindex.utils surface, as the cookbooks call it ──

def test_remove_fields_max_len():
    out = remove_fields({"keep": "x" * 50, "text": "gone"}, max_len=10)
    assert out == {"keep": "x" * 10 + "..."}
    assert remove_fields({"keep": "short"}, max_len=10) == {"keep": "short"}


def test_create_node_mapping_flat():
    mapping = create_node_mapping(TREE)
    assert set(mapping) == {"0000", "0001", "0002"}
    assert mapping["0001"]["title"] == "Child"


def test_create_node_mapping_page_ranges():
    mapping = create_node_mapping(TREE, include_page_ranges=True, max_page=9)
    assert mapping["0000"] == {"node": TREE[0], "start_index": 1, "end_index": 3}
    assert mapping["0001"]["start_index"] == 3
    assert mapping["0001"]["end_index"] == 5
    assert mapping["0002"] == {"node": TREE[1], "start_index": 5, "end_index": 9}


def test_print_tree_exclude_fields(capsys):
    print_tree(TREE, exclude_fields=["text"])
    out = capsys.readouterr().out
    assert "Root" in out and "'text'" not in out

    print_tree(TREE)
    assert "[0000] Root" in capsys.readouterr().out


# ── import cost: the SDK must not pay for the indexing stack ──

def test_import_pageindex_is_lazy():
    probe = (
        "import sys; import pageindex; "
        "heavy = [m for m in ('pageindex.page_index_classic', 'pageindex.flash', "
        "'pageindex.utils', 'pageindex.tree_optimize', 'numpy', 'PyPDF2') "
        "if m in sys.modules]; "
        "print(','.join(heavy) or 'clean'); "
        "print(type(pageindex.page_index_main).__name__)"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["clean", "function"]


def test_sdk_submodules_reachable_and_dunder_probes_stay_lazy():
    """The 0.2.10 modules resolve as attributes, and underscore probes (the
    frequent unknown names: copy/pickle/inspect dunders) raise without
    dragging in the indexing stack. A non-underscore unknown name still
    raises AttributeError — after the compat fallthrough's one classic
    import, which is the pre-0.2.10 behavior."""
    probe = (
        "import sys, pageindex\n"
        "pageindex.agent_tools; pageindex.local_chat\n"
        "pageindex.mcp_bridge; pageindex.integrations\n"
        "assert not hasattr(pageindex, '__wrapped__')\n"
        "heavy = [m for m in ('pageindex.page_index_classic', "
        "'pageindex.flash', 'pageindex.utils') if m in sys.modules]\n"
        "print(','.join(heavy) or 'clean')\n"
        "try:\n"
        "    pageindex.definitely_missing\n"
        "    raise SystemExit('no AttributeError')\n"
        "except AttributeError:\n"
        "    pass\n"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "clean"


def test_classic_compat_surface_still_reachable():
    """The pre-0.2.10 catch-all made every classic/utils public name a
    package attribute; dropping it broke `from pageindex import
    ConfigLoader` on upgrade with no deprecation path."""
    probe = (
        "import pageindex\n"
        "assert callable(pageindex.count_tokens)\n"
        "assert isinstance(pageindex.ConfigLoader, type)\n"
        "from pageindex import check_toc  # noqa: F401\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_import_leaves_litellm_env_untouched(tmp_path):
    """Importing the package must not configure litellm for the host
    process; constructing a local client (which will use litellm) does."""
    env = {k: v for k, v in os.environ.items()
           if k != "LITELLM_LOCAL_MODEL_COST_MAP"}
    probe = (
        "import os, pageindex\n"
        "assert 'LITELLM_LOCAL_MODEL_COST_MAP' not in os.environ, "
        "'stamped at import'\n"
        f"pageindex.PageIndexLocalClient(storage_path={str(tmp_path / 's')!r})\n"
        "assert os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] == 'True'\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_utils_import_keeps_litellm_off_the_network():
    """utils' import sets litellm's no-fetch default; an explicit choice wins."""
    probe = ("import os, pageindex.utils; "
             "print(os.environ['LITELLM_LOCAL_MODEL_COST_MAP'])")
    env = {k: v for k, v in os.environ.items()
           if k != "LITELLM_LOCAL_MODEL_COST_MAP"}
    fresh = subprocess.run([sys.executable, "-c", probe], env=env,
                           capture_output=True, text=True, check=True)
    assert fresh.stdout.strip() == "True"
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "False"
    chosen = subprocess.run([sys.executable, "-c", probe], env=env,
                            capture_output=True, text=True, check=True)
    assert chosen.stdout.strip() == "False"
