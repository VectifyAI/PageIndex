import pytest

from pageindex.library import digest
from pageindex.library.config import LibraryConfig


META = {"id": "pi-1", "name": "b.pdf", "description": "About things.", "pageNum": 6,
        "metadata": {"title": "The Book", "profile": "nonfiction"}}


def tree_with_text(sample_tree):
    sample_tree[0]["summary"] = "Chapter one routing"
    sample_tree[0]["nodes"][0]["digest"] = "**Thesis** — A.\n- point a"
    sample_tree[0]["nodes"][0]["summary"] = "Section A routing"
    sample_tree[0]["nodes"][1]["key_items"] = ["merged x", "merged y"]
    sample_tree[1]["summary"] = "Chapter two routing"
    return sample_tree


def test_render_book_digest_shape(sample_tree):
    md = digest.render_book_digest(META, tree_with_text(sample_tree))
    assert md.startswith("# The Book\n")
    assert "About things." in md
    assert "## Chapter One (pp. 1–4)" in md
    assert "### Section A (pp. 2–3)" in md
    assert "**Thesis** — A." in md                      # digest used when present
    assert "_Chapter one routing_" in md                # summary fallback, marked as such
    assert "Includes: merged x; merged y" in md         # key_items
    assert "## Chapter Two (pp. 5–6)" in md


def test_render_node_digest(sample_tree):
    md = digest.render_node_digest(META, tree_with_text(sample_tree), "0001")
    assert md.startswith("# The Book — Section A (pp. 2–3)\n")
    assert "point a" in md
    with pytest.raises(KeyError):
        digest.render_node_digest(META, sample_tree, "9999")


def test_slugify():
    assert digest.slugify("Complete Works of Ram Chandra Volume I - Ram Chandra.pdf") \
        == "complete-works-of-ram-chandra-volume-i-ram-chandra-pdf"
    assert digest.slugify("  Ünïcode: yes!  ") == "unicode-yes"


def test_write_digest_paths(home, store, sample_tree):
    meta = {**META, "status": "completed", "createdAt": "2026-08-30T00:00:00.000000",
            "folderId": None, "mode": "flash"}
    store.save_document("pi-1", meta, tree_with_text(sample_tree), [])
    cfg = LibraryConfig.load()
    book_path = digest.write_digest(cfg, store, "pi-1")
    node_path = digest.write_digest(cfg, store, "pi-1", node_id="0001")
    assert book_path == home / "digests" / "the-book" / "book.md"
    assert node_path == home / "digests" / "the-book" / "0001-section-a.md"
    assert "# The Book" in book_path.read_text()
