import pytest
import requests

from pageindex.client import PageIndexAPIError as ClientPageIndexAPIError
from pageindex import PageIndexAPIError, PageIndexClient
from pageindex.client import CloudClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="ok", lines=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self._lines = lines or []
        self.closed = False

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


class StreamingErrorResponse(FakeResponse):
    def iter_lines(self):
        raise requests.ReadTimeout("stream stalled")


def test_legacy_imports_and_initializers():
    positional = PageIndexClient("pi-test")
    keyword = PageIndexClient(api_key="pi-test")
    cloud = CloudClient(api_key="pi-test")

    assert positional._legacy_cloud_api.api_key == "pi-test"
    assert keyword._legacy_cloud_api.api_key == "pi-test"
    assert cloud._legacy_cloud_api.api_key == "pi-test"
    assert issubclass(PageIndexAPIError, Exception)
    assert ClientPageIndexAPIError is PageIndexAPIError


def test_legacy_methods_exist():
    client = PageIndexClient("pi-test")
    for method_name in [
        "submit_document",
        "get_ocr",
        "get_tree",
        "is_retrieval_ready",
        "submit_query",
        "get_retrieval",
        "chat_completions",
        "get_document",
        "delete_document",
        "list_documents",
        "create_folder",
        "list_folders",
    ]:
        assert callable(getattr(client, method_name))


def test_legacy_base_url_can_be_overridden_from_client(monkeypatch):
    calls = []

    def fake_request(method, url, headers=None, **kwargs):
        calls.append({"method": method, "url": url, "headers": headers})
        return FakeResponse(payload={"id": "doc-1"})

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)
    monkeypatch.setattr(PageIndexClient, "BASE_URL", "https://staging.pageindex.test")

    result = PageIndexClient("pi-test").get_document("doc-1")

    assert result == {"id": "doc-1"}
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://staging.pageindex.test/doc/doc-1/metadata/"
    assert calls[0]["headers"] == {"api_key": "pi-test"}


def test_submit_document_uses_legacy_endpoint(monkeypatch, tmp_path):
    calls = []

    def fake_request(method, url, headers=None, files=None, data=None, **kwargs):
        calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "data": data,
            "files": files,
            "kwargs": kwargs,
        })
        return FakeResponse(payload={"doc_id": "doc-1"})

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    result = PageIndexClient("pi-test").submit_document(
        str(pdf),
        mode="mcp",
        beta_headers=["block_reference"],
        folder_id="folder-1",
    )

    assert result == {"doc_id": "doc-1"}
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "https://api.pageindex.ai/doc/"
    assert calls[0]["headers"] == {"api_key": "pi-test"}
    assert "timeout" not in calls[0]["kwargs"]
    assert calls[0]["data"]["if_retrieval"] is True
    assert calls[0]["data"]["mode"] == "mcp"
    assert calls[0]["data"]["beta_headers"] == '["block_reference"]'
    assert calls[0]["data"]["folder_id"] == "folder-1"


def test_get_ocr_and_tree_use_legacy_urls(monkeypatch):
    get_calls = []

    def fake_request(method, url, headers=None, **kwargs):
        get_calls.append({"method": method, "url": url, "headers": headers})
        return FakeResponse(payload={"status": "completed", "retrieval_ready": True})

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)
    client = PageIndexClient("pi-test")

    assert client.get_ocr("doc-1", format="page")["status"] == "completed"
    assert client.get_tree("doc-1", node_summary=True)["retrieval_ready"] is True

    assert get_calls[0]["method"] == "GET"
    assert get_calls[0]["url"] == "https://api.pageindex.ai/doc/doc-1/?type=ocr&format=page"
    assert get_calls[1]["url"] == "https://api.pageindex.ai/doc/doc-1/?type=tree&summary=True"


def test_get_ocr_rejects_invalid_format():
    with pytest.raises(ValueError, match="Format parameter must be"):
        PageIndexClient("pi-test").get_ocr("doc-1", format="bad")


def test_submit_query_uses_legacy_payload(monkeypatch):
    calls = []

    def fake_request(method, url, headers=None, json=None, **kwargs):
        calls.append({"method": method, "url": url, "headers": headers, "json": json})
        return FakeResponse(payload={"retrieval_id": "ret-1"})

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    result = PageIndexClient("pi-test").submit_query("doc-1", "What changed?", thinking=True)

    assert result == {"retrieval_id": "ret-1"}
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "https://api.pageindex.ai/retrieval/"
    assert calls[0]["json"] == {
        "doc_id": "doc-1",
        "query": "What changed?",
        "thinking": True,
    }


def test_chat_completions_non_stream_returns_json(monkeypatch):
    calls = []
    payload = {"choices": [{"message": {"content": "answer"}}]}

    def fake_request(method, url, headers=None, json=None, stream=False, **kwargs):
        calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
            "stream": stream,
        })
        return FakeResponse(payload=payload)

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    result = PageIndexClient("pi-test").chat_completions(
        [{"role": "user", "content": "hi"}],
        doc_id=["doc-1"],
        temperature=0.1,
        enable_citations=True,
    )

    assert result == payload
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "https://api.pageindex.ai/chat/completions/"
    assert calls[0]["stream"] is False
    assert calls[0]["json"] == {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "doc_id": ["doc-1"],
        "temperature": 0.1,
        "enable_citations": True,
    }


def test_chat_completions_stream_parses_text_chunks(monkeypatch):
    calls = []
    lines = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}',
        b"data: [DONE]",
    ]

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, "kwargs": kwargs})
        return FakeResponse(lines=lines)

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    chunks = list(PageIndexClient("pi-test").chat_completions(
        [{"role": "user", "content": "hi"}],
        stream=True,
    ))

    assert chunks == ["hel", "lo"]
    assert "timeout" not in calls[0]["kwargs"]


def test_chat_completions_stream_metadata_returns_raw_chunks(monkeypatch):
    calls = []
    lines = [
        b'data: {"object":"chat.completion.chunk"}',
        b"data: [DONE]",
    ]

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, "json": kwargs.get("json")})
        return FakeResponse(lines=lines)

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    chunks = list(PageIndexClient("pi-test").chat_completions(
        [{"role": "user", "content": "hi"}],
        stream=True,
        stream_metadata=True,
    ))

    assert chunks == [{"object": "chat.completion.chunk"}]
    assert "stream_metadata" not in calls[0]["json"]


def test_chat_completions_stream_errors_are_pageindex_api_error(monkeypatch):
    def fake_request(*args, **kwargs):
        return StreamingErrorResponse()

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    stream = PageIndexClient("pi-test").chat_completions(
        [{"role": "user", "content": "hi"}],
        stream=True,
    )

    with pytest.raises(PageIndexAPIError, match="Failed to stream chat completion: stream stalled"):
        list(stream)


def test_api_errors_are_pageindex_api_error(monkeypatch):
    def fake_request(*args, **kwargs):
        return FakeResponse(status_code=500, text="server error")

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    with pytest.raises(PageIndexAPIError, match="Failed to get document metadata"):
        PageIndexClient("pi-test").get_document("doc-1")


def test_network_errors_are_wrapped_as_pageindex_api_error(monkeypatch):
    def fake_request(*args, **kwargs):
        raise requests.Timeout("slow network")

    monkeypatch.setattr("pageindex.cloud_api.requests.request", fake_request)

    with pytest.raises(PageIndexAPIError, match="Failed to get document metadata: slow network"):
        PageIndexClient("pi-test").get_document("doc-1")


def test_list_documents_validates_legacy_pagination():
    client = PageIndexClient("pi-test")

    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        client.list_documents(limit=0)
    with pytest.raises(ValueError, match="offset must be non-negative"):
        client.list_documents(offset=-1)


def test_chat_completions_stream_closes_response_after_done(monkeypatch):
    fake = FakeResponse(lines=[
        b'data: {"choices":[{"delta":{"content":"hi"}}]}',
        b"data: [DONE]",
    ])
    monkeypatch.setattr("pageindex.cloud_api.requests.request",
                       lambda *a, **kw: fake)

    list(PageIndexClient("pi-test").chat_completions(
        [{"role": "user", "content": "x"}], stream=True,
    ))
    assert fake.closed is True


def test_chat_completions_stream_closes_response_on_early_abandon(monkeypatch):
    fake = FakeResponse(lines=[
        b'data: {"choices":[{"delta":{"content":"a"}}]}',
        b'data: {"choices":[{"delta":{"content":"b"}}]}',
        b"data: [DONE]",
    ])
    monkeypatch.setattr("pageindex.cloud_api.requests.request",
                       lambda *a, **kw: fake)

    gen = PageIndexClient("pi-test").chat_completions(
        [{"role": "user", "content": "x"}], stream=True,
    )
    next(gen)
    gen.close()
    assert fake.closed is True


def test_empty_api_key_warns_and_falls_back_to_local(caplog, tmp_path, monkeypatch):
    import logging
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with caplog.at_level(logging.WARNING, logger="pageindex.client"):
        client = PageIndexClient(api_key="", storage_path=str(tmp_path))

    assert any("empty api_key" in r.message for r in caplog.records)
    assert client._legacy_cloud_api is None
