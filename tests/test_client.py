# tests/sdk/test_client.py
import pytest
from pageindex.client import PageIndexClient, LocalClient, CloudClient


def test_local_client_is_pageindex_client(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    assert isinstance(client, PageIndexClient)


def test_cloud_client_is_pageindex_client():
    client = CloudClient(api_key="pi-test")
    assert isinstance(client, PageIndexClient)


def test_empty_api_key_legacy_method_error_is_specific(tmp_path, caplog):
    """Empty api_key falls back to local mode; legacy methods raise a clear error."""
    import warnings
    from pageindex.errors import PageIndexAPIError

    client = PageIndexClient(api_key="", storage_path=str(tmp_path / "pi"))
    # Empty api_key → local mode; legacy methods should explain why
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        with pytest.raises(PageIndexAPIError, match="empty string"):
            client.submit_document("some.pdf")


def test_none_api_key_legacy_method_error_is_generic(tmp_path):
    """api_key=None → local mode; legacy methods raise generic error (not 'empty')."""
    import warnings
    from pageindex.errors import PageIndexAPIError

    client = PageIndexClient(api_key=None, model="gpt-4o", storage_path=str(tmp_path / "pi"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        with pytest.raises(PageIndexAPIError) as exc_info:
            client.submit_document("some.pdf")
    assert "empty" not in str(exc_info.value)


def test_collection_default_name(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    col = client.collection()
    assert col.name == "default"


def test_collection_custom_name(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    col = client.collection("papers")
    assert col.name == "papers"


def test_list_collections_empty(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    assert client.list_collections() == []


def test_list_collections_after_create(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    client.collection("papers")
    assert "papers" in client.list_collections()


def test_delete_collection(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    client.collection("papers")
    client.delete_collection("papers")
    assert "papers" not in client.list_collections()


def test_retrieve_model_defaults_to_strong_reasoner(tmp_path):
    """Retrieval must not silently follow the (cheaper) indexing model."""
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    assert client._backend.get_retrieve_model() == "gpt-5.4"


def test_explicit_retrieve_model_wins(tmp_path):
    client = LocalClient(model="gpt-4o", retrieve_model="ollama/llama3",
                         storage_path=str(tmp_path / "pi"))
    assert client._backend.get_retrieve_model() == "litellm/ollama/llama3"


def test_retrieve_model_none_follows_model(tmp_path):
    from pageindex.config import IndexConfig
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"),
                         index_config=IndexConfig(retrieve_model=None))
    assert client._backend.get_retrieve_model() == "gpt-4o"


def test_register_parser(tmp_path):
    client = LocalClient(model="gpt-4o", storage_path=str(tmp_path / "pi"))
    class FakeParser:
        def supported_extensions(self): return [".txt"]
        def parse(self, file_path, **kwargs): pass
    client.register_parser(FakeParser())
