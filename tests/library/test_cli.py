import json

import pytest

from pageindex.library import cli, ingest
from pageindex.library.config import LibraryConfig
from pageindex.local_store import DocStore


@pytest.fixture
def indexed(home, sample_tree, sample_pages, fake_llm, monkeypatch, tmp_path):
    import copy
    pdf = tmp_path / "Book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(ingest, "extract_toc", lambda p, use_embedded_toc=True: {
        "structure": copy.deepcopy(sample_tree), "page_texts": list(sample_pages),
        "doc_title": "Book Title"})
    monkeypatch.setattr(ingest, "optimize_structure", lambda *a: {})
    return pdf


def test_add_and_list(indexed, capsys):
    assert cli.main(["add", str(indexed)]) == 0
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Book.pdf" in out and "completed" in out and "4 nodes" in out


def test_add_no_summaries_and_profile(indexed, capsys, fake_llm):
    assert cli.main(["add", str(indexed), "--no-summaries", "--profile", "diary"]) == 0
    assert fake_llm == []
    capsys.readouterr()
    cli.main(["list", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["status"] == "indexed" and rows[0]["profile"] == "diary"


def test_show_prints_tree_with_summaries(indexed, capsys):
    cli.main(["add", str(indexed)])
    capsys.readouterr()
    assert cli.main(["show", "Book", "--depth", "1"]) == 0
    out = capsys.readouterr().out
    assert "[0000] Chapter One" in out and "p1-4" in out
    assert "Section A" not in out
    cli.main(["show", "Book"])
    assert "Section A" in capsys.readouterr().out


def test_summarize_digest_tier(indexed, capsys, fake_llm):
    cli.main(["add", str(indexed), "--no-summaries"])
    assert cli.main(["summarize", "Book", "--tier", "digest", "--node", "0003"]) == 0
    tree = DocStore(str(LibraryConfig.load().storage_path)).get_tree(
        ingest.file_doc_id(str(indexed)))
    assert "digest" in tree[1] and "digest" not in tree[0]
    assert "1 generated" in capsys.readouterr().out


def test_unknown_book_is_an_error(indexed, capsys):
    assert cli.main(["show", "nothing"]) == 1
    assert "No book matches" in capsys.readouterr().err


def test_home_flag_overrides_env(indexed, tmp_path, capsys):
    other = tmp_path / "elsewhere"
    assert cli.main(["--home", str(other), "add", str(indexed), "--no-summaries"]) == 0
    assert (other / ".pageindex" / "manifest.json").exists()
