"""What `pip install pageindex` exposes: 0.2.8 helper compat and import cost."""
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
        "heavy = [m for m in ('pageindex.page_index', 'pageindex.flash', "
        "'pageindex.utils', 'pageindex.tree_optimize', 'numpy', 'PyPDF2') "
        "if m in sys.modules]; "
        "print(','.join(heavy) or 'clean'); "
        "print(type(pageindex.page_index_main).__name__)"
    )
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["clean", "function"]
