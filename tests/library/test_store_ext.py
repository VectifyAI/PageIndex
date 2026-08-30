def seed(store, doc_id="pi-1"):
    meta = {"id": doc_id, "name": "b.pdf", "description": None, "status": "indexed",
            "createdAt": "2026-08-30T00:00:00.000000", "pageNum": 1, "folderId": None,
            "metadata": {"title": "B"}, "mode": "flash"}
    store.save_document(doc_id, meta, [{"title": "T", "node_id": "0000",
                                       "start_index": 1, "end_index": 1, "nodes": []}],
                        [{"page_index": 1, "markdown": "x"}])
    return doc_id


def test_save_tree_replaces_only_the_tree(store):
    doc_id = seed(store)
    store.save_tree(doc_id, [{"title": "New", "node_id": "0000", "start_index": 1,
                              "end_index": 1, "summary": "s", "nodes": []}])
    assert store.get_tree(doc_id)[0]["summary"] == "s"
    assert store.get_pages(doc_id)[0]["markdown"] == "x"
    assert store.get_meta(doc_id)["status"] == "indexed"


def test_update_meta_merges_and_refreshes_manifest(store):
    doc_id = seed(store)
    meta = store.update_meta(doc_id, status="completed", description="desc")
    assert meta["status"] == "completed"
    assert store.get_meta(doc_id)["description"] == "desc"
    assert store.get_meta(doc_id)["metadata"] == {"title": "B"}
    assert [m["status"] for m in store.list_metas()] == ["completed"]


def test_update_meta_unknown_doc_raises(store):
    import pytest
    with pytest.raises(KeyError):
        store.update_meta("pi-missing", status="x")
