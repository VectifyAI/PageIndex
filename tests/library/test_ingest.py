import pytest

from pageindex.library import ingest
from pageindex.library.config import LibraryConfig
from pageindex.local_store import DocStore


@pytest.fixture
def pdf(tmp_path):
    p = tmp_path / "My Book.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.fixture
def fake_flash(monkeypatch, sample_tree, sample_pages):
    import copy

    def extract_toc(path, use_embedded_toc=True):
        return {"structure": copy.deepcopy(sample_tree), "page_texts": list(sample_pages),
                "doc_title": "My Book Title"}

    def optimize(structure, page_texts, do_expand, model):
        assert do_expand is False and model is None
        return {"merges": 0}
    monkeypatch.setattr(ingest, "extract_toc", extract_toc)
    monkeypatch.setattr(ingest, "optimize_structure", optimize)


def test_add_book_indexes_then_summarizes(home, pdf, fake_flash, fake_llm):
    cfg = LibraryConfig.load()
    out = ingest.add_book(str(pdf), cfg, log=lambda *_: None)
    store = DocStore(str(cfg.storage_path))
    meta = store.get_meta(out["doc_id"])
    assert out["doc_id"] == ingest.file_doc_id(str(pdf))
    assert meta["status"] == "completed"
    assert meta["name"] == "My Book.pdf"
    assert meta["metadata"]["title"] == "My Book Title"
    assert meta["metadata"]["profile"] == "nonfiction"
    assert meta["metadata"]["sha256"] and meta["metadata"]["source"] == str(pdf)
    assert meta["description"].startswith("D")
    tree = store.get_tree(out["doc_id"])
    assert [n["node_id"] for n in tree] == ["0000", "0003"]
    assert all("summary" in n for n in tree)
    assert store.get_pages(out["doc_id"])[1]["markdown"] == "Page 2 text word2."
    assert out["nodes"] == 4 and out["pages"] == 6


def test_add_book_without_summaries_leaves_status_indexed(home, pdf, fake_flash, fake_llm):
    out = ingest.add_book(str(pdf), LibraryConfig.load(), summaries=False, log=lambda *_: None)
    assert out["status"] == "indexed" and out["summary"] is None
    assert fake_llm == []


def test_add_book_is_idempotent_unless_forced(home, pdf, fake_flash, fake_llm):
    cfg = LibraryConfig.load()
    first = ingest.add_book(str(pdf), cfg, log=lambda *_: None)
    calls = len(fake_llm)
    again = ingest.add_book(str(pdf), cfg, log=lambda *_: None)
    assert again["doc_id"] == first["doc_id"] and len(fake_llm) == calls
    forced = ingest.add_book(str(pdf), cfg, force=True, log=lambda *_: None)
    assert forced["doc_id"] == first["doc_id"] and len(fake_llm) > calls


def test_add_book_force_warns_about_discarded_summaries(home, pdf, fake_flash, fake_llm):
    cfg = LibraryConfig.load()
    logs = []
    ingest.add_book(str(pdf), cfg, log=logs.append)  # first index: has summaries
    logs.clear()
    ingest.add_book(str(pdf), cfg, force=True, log=logs.append)
    warnings = [m for m in logs if "--force discards" in m]
    assert len(warnings) == 1
    assert "summaries" in warnings[0] and "digests" in warnings[0]
    assert "0 summaries" not in warnings[0]


def test_add_book_force_no_warning_on_first_time_index(home, pdf, fake_flash, fake_llm):
    logs = []
    # force=True but nothing was previously indexed: no prior data to lose.
    ingest.add_book(str(pdf), LibraryConfig.load(), force=True, log=logs.append)
    assert not any("--force discards" in m for m in logs)


def test_add_book_diary_profile_uses_splitter(home, pdf, fake_flash, fake_llm, monkeypatch):
    called = {}

    def fake_apply(structure, page_texts):
        called["n"] = len(page_texts)
        return [{"title": "1981", "start_index": 1, "end_index": 6, "nodes": []}]
    monkeypatch.setattr(ingest, "apply_diary_profile", fake_apply)
    out = ingest.add_book(str(pdf), LibraryConfig.load(), profile="diary",
                          summaries=False, log=lambda *_: None)
    assert called["n"] == 6
    store = DocStore(str(LibraryConfig.load().storage_path))
    assert [n["title"] for n in store.get_tree(out["doc_id"])] == ["1981"]
    assert store.get_meta(out["doc_id"])["metadata"]["profile"] == "diary"


def test_add_book_reports_long_leaves(home, pdf, fake_flash, fake_llm):
    cfg = LibraryConfig(home=home, max_leaf_pages=1)
    out = ingest.add_book(str(pdf), cfg, summaries=False, log=lambda *_: None)
    assert ("Section A", 2, 3) in out["long_leaves"]
    assert ("Section B", 4, 4) not in out["long_leaves"]


def test_add_book_empty_structure_raises(home, pdf, monkeypatch):
    monkeypatch.setattr(ingest, "extract_toc",
                        lambda path, use_embedded_toc=True: {"structure": [], "page_texts": ["x"]})
    with pytest.raises(ingest.IngestError, match="standard"):
        ingest.add_book(str(pdf), LibraryConfig.load(), log=lambda *_: None)


def test_find_book(home, pdf, fake_flash, fake_llm):
    cfg = LibraryConfig.load()
    out = ingest.add_book(str(pdf), cfg, summaries=False, log=lambda *_: None)
    store = DocStore(str(cfg.storage_path))
    assert ingest.find_book(store, out["doc_id"])["id"] == out["doc_id"]
    assert ingest.find_book(store, "My Book.pdf")["id"] == out["doc_id"]
    assert ingest.find_book(store, "book title")["id"] == out["doc_id"]
    with pytest.raises(LookupError, match="No book"):
        ingest.find_book(store, "zzz")
