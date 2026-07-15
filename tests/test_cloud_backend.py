import asyncio
import io
import json
import time

import pytest

import pageindex.backend.cloud as cloud_mod
from pageindex.backend.cloud import CloudBackend, API_BASE
from pageindex.errors import CloudAPIError, CollectionNotFoundError, DocumentNotFoundError

# Real sleep captured at import, before the _no_sleep autouse fixture patches
# time.sleep on the shared module object. Tests that need genuine pacing (e.g.
# to let a consumer break before a background thread drains a stream) must use
# this, since _no_sleep would otherwise no-op an `import time; time.sleep(...)`.
_REAL_SLEEP = time.sleep


def test_cloud_backend_init():
    backend = CloudBackend(api_key="pi-test")
    assert backend._api_key == "pi-test"
    assert backend._headers["api_key"] == "pi-test"


def test_api_base_url():
    assert "pageindex.ai" in API_BASE


def test_query_rejects_empty_doc_ids():
    backend = CloudBackend(api_key="pi-test")
    with pytest.raises(ValueError, match="cannot be empty"):
        backend.query("col", "q", doc_ids=[])


# ── helpers ──────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", lines=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.content = json.dumps(self._json).encode() if json_data is not None else b""
        self._lines = lines or []

    def json(self):
        return self._json

    def iter_lines(self, decode_unicode=True):
        yield from self._lines

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(cloud_mod.time, "sleep", lambda *_: None)


# ── _request: retry must rewind file objects (empty-upload regression) ──────

def test_request_rewinds_file_on_retry(monkeypatch):
    backend = CloudBackend(api_key="pi-test")
    payload = b"%PDF-1.4 fake body"
    fobj = io.BytesIO(payload)
    uploads = []

    def fake_request(method, url, headers=None, timeout=None, **kwargs):
        # Simulate requests consuming the file body on every attempt.
        uploads.append(kwargs["files"]["file"].read())
        if len(uploads) == 1:
            return FakeResponse(status_code=500)
        return FakeResponse(status_code=200, json_data={"doc_id": "d1"})

    monkeypatch.setattr(cloud_mod.requests, "request", fake_request)
    resp = backend._request("POST", "/doc/", files={"file": fobj}, data={})
    assert resp == {"doc_id": "d1"}
    assert uploads[0] == payload
    # Without seek(0) the retry would upload an empty body.
    assert uploads[1] == payload


# ── list_documents: pagination beyond the API's 100-doc page cap ─────────────

def test_list_documents_paginates(monkeypatch):
    backend = CloudBackend(api_key="pi-test")
    backend._folder_id_cache["col"] = None  # skip folder lookup
    calls = []

    def fake_request(method, path, **kwargs):
        offset = kwargs["params"]["offset"]
        calls.append(offset)
        n = 100 if offset == 0 else 30
        return {"documents": [{"id": f"d{offset + i}", "name": ""} for i in range(n)]}

    monkeypatch.setattr(backend, "_request", fake_request)
    docs = backend.list_documents("col")
    assert len(docs) == 130
    assert calls == [0, 100]
    assert docs[-1]["doc_id"] == "d129"


# ── base_url override must reach every request path ──────────────────────────

def test_base_url_override_routes_requests(monkeypatch):
    backend = CloudBackend(api_key="pi-test", base_url="https://staging.example.com")
    urls = []

    def fake_request(method, url, headers=None, **kwargs):
        urls.append(url)
        return FakeResponse(status_code=200, json_data={"folders": []})

    monkeypatch.setattr(cloud_mod.requests, "request", fake_request)
    backend.list_collections()
    assert urls == ["https://staging.example.com/folders/"]


def test_base_url_override_routes_streaming(monkeypatch):
    backend = CloudBackend(api_key="pi-test", base_url="https://staging.example.com")
    urls = []

    def fake_post(url, **kwargs):
        urls.append(url)
        return FakeResponse(status_code=200, lines=["data: [DONE]"])

    monkeypatch.setattr(cloud_mod.requests, "post", fake_post)
    _collect_events(backend)
    assert urls == ["https://staging.example.com/chat/completions/"]


def test_client_base_url_override_reaches_both_backends():
    # PageIndexClient.BASE_URL is the documented override point; it must apply
    # to the Collection API (CloudBackend), not just the legacy SDK methods.
    from pageindex.client import PageIndexClient

    class StagingClient(PageIndexClient):
        BASE_URL = "https://staging.example.com"

    client = StagingClient(api_key="pi-test")
    assert client._backend._base_url == "https://staging.example.com"
    assert client._legacy_cloud_api.base_url == "https://staging.example.com"


# ── folder resolution: plan-limit vs transient errors ────────────────────────

def test_folder_unavailable_warns_and_caches(monkeypatch):
    backend = CloudBackend(api_key="pi-test")

    def fake_request(method, path, **kwargs):
        raise CloudAPIError("Cloud API error 403: upgrade", status_code=403)

    monkeypatch.setattr(backend, "_request", fake_request)
    with pytest.warns(UserWarning, match="not available on this plan"):
        assert backend._get_folder_id("col") is None
    assert backend._folder_id_cache["col"] is None


def test_folder_transient_error_propagates_and_is_not_cached(monkeypatch):
    backend = CloudBackend(api_key="pi-test")

    def fake_request(method, path, **kwargs):
        raise CloudAPIError("Cloud API request failed: connection reset")

    monkeypatch.setattr(backend, "_request", fake_request)
    with pytest.raises(CloudAPIError):
        backend._get_folder_id("col")
    # A blip must not permanently route documents to the global space.
    assert "col" not in backend._folder_id_cache


def test_missing_folder_raises_collection_not_found(monkeypatch):
    # Folders ARE available but the name has no match (e.g. deleted): must
    # raise, never fall back to the account-wide global space.
    backend = CloudBackend(api_key="pi-test")
    monkeypatch.setattr(
        backend, "_request",
        lambda *a, **k: {"folders": [{"name": "other", "id": "f1"}]},
    )
    with pytest.raises(CollectionNotFoundError):
        backend._get_folder_id("gone")
    assert "gone" not in backend._folder_id_cache


def test_list_documents_raises_for_missing_collection(monkeypatch):
    # Guards the query path: a stale collection must error out, not silently
    # list (and query over) every document in the account.
    backend = CloudBackend(api_key="pi-test")
    monkeypatch.setattr(backend, "_request", lambda *a, **k: {"folders": []})
    with pytest.raises(CollectionNotFoundError):
        backend.list_documents("gone")


def test_delete_collection_missing_is_noop(monkeypatch):
    backend = CloudBackend(api_key="pi-test")
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"folders": []}

    monkeypatch.setattr(backend, "_request", fake_request)
    backend.delete_collection("gone")  # idempotent, matching the local backend
    assert ("GET", "/folders/") in calls
    assert not any(method == "DELETE" for method, _ in calls)


def test_list_collections_degrades_when_folders_unavailable(monkeypatch):
    backend = CloudBackend(api_key="pi-test")

    def fake_request(method, path, **kwargs):
        raise CloudAPIError("Cloud API error 403: upgrade", status_code=403)

    monkeypatch.setattr(backend, "_request", fake_request)
    with pytest.warns(UserWarning, match="not available on this plan"):
        assert backend.list_collections() == []


def test_list_collections_propagates_transient_error(monkeypatch):
    backend = CloudBackend(api_key="pi-test")

    def fake_request(method, path, **kwargs):
        raise CloudAPIError("Cloud API error 503: unavailable", status_code=503)

    monkeypatch.setattr(backend, "_request", fake_request)
    with pytest.raises(CloudAPIError):
        backend.list_collections()


# ── doc endpoints: 404 maps to DocumentNotFoundError (local parity) ──────────

def test_doc_404_maps_to_document_not_found(monkeypatch):
    backend = CloudBackend(api_key="pi-test")

    def fake_request(method, path, **kwargs):
        raise CloudAPIError("Cloud API error 404: not found", status_code=404)

    monkeypatch.setattr(backend, "_request", fake_request)
    with pytest.raises(DocumentNotFoundError):
        backend.get_document_structure("col", "missing")
    with pytest.raises(DocumentNotFoundError):
        backend.delete_document("col", "missing")


def test_get_document_include_text_warns(monkeypatch):
    backend = CloudBackend(api_key="pi-test")
    monkeypatch.setattr(backend, "_doc_request", lambda *a, **k: {"tree": []})
    with pytest.warns(UserWarning, match="include_text is not supported"):
        backend.get_document("col", "d1", include_text=True)


def test_get_page_content_preserves_images(monkeypatch):
    # Cloud OCR pages carry an `images` list; get_page_content must pass it
    # through (parity with the local backend and the documented PageContent
    # shape) so the SDK-prompted UI can render figures — omitting it only when
    # empty. Real API uses page_index/markdown/images keys.
    backend = CloudBackend(api_key="pi-test")
    imgs = [{"path": "fig1.png", "width": 640, "height": 480}]
    monkeypatch.setattr(backend, "_doc_request", lambda *a, **k: {"result": [
        {"page_index": 1, "markdown": "page one", "images": imgs},
        {"page_index": 2, "markdown": "page two", "images": []},  # empty -> omitted
    ]})
    out = backend.get_page_content("col", "d1", "1,2")
    assert out == [
        {"page": 1, "content": "page one", "images": imgs},
        {"page": 2, "content": "page two"},
    ]


# ── query_stream: terminal contract and error propagation ───────────────────

def _collect_events(backend, **kwargs):
    async def _run():
        events = []
        async for ev in backend.query_stream("col", "q", doc_ids=["d1"], **kwargs):
            events.append(ev)
        return events
    return asyncio.run(_run())


def _sse(block_type, content):
    return "data: " + json.dumps({
        "block_metadata": {"type": block_type},
        "choices": [{"delta": {"content": content}}],
    })


def test_query_stream_emits_answer_done(monkeypatch):
    backend = CloudBackend(api_key="pi-test")
    lines = [_sse("text", "Hello "), _sse("text", "world"), "data: [DONE]"]
    monkeypatch.setattr(
        cloud_mod.requests, "post",
        lambda *a, **k: FakeResponse(status_code=200, lines=lines),
    )
    events = _collect_events(backend)
    assert [e.type for e in events] == ["answer_delta", "answer_delta", "answer_done"]
    # Same contract as the local backend: answer_done carries the full text.
    assert events[-1].data == "Hello world"


def test_query_stream_http_error_raises(monkeypatch):
    backend = CloudBackend(api_key="pi-test")
    monkeypatch.setattr(
        cloud_mod.requests, "post",
        lambda *a, **k: FakeResponse(status_code=401, text="unauthorized"),
    )

    async def _run():
        async for _ in backend.query_stream("col", "q", doc_ids=["d1"]):
            pass

    with pytest.raises(CloudAPIError, match="401"):
        asyncio.run(_run())


def test_query_stream_connect_failure_raises_instead_of_hanging(monkeypatch):
    backend = CloudBackend(api_key="pi-test")

    def fake_post(*a, **k):
        raise cloud_mod.requests.ConnectionError("dns failure")

    monkeypatch.setattr(cloud_mod.requests, "post", fake_post)

    async def _run():
        async for _ in backend.query_stream("col", "q", doc_ids=["d1"]):
            pass

    with pytest.raises(CloudAPIError, match="request failed"):
        asyncio.run(_run())


def test_query_uses_long_timeout_and_single_attempt(monkeypatch):
    """Non-streaming chat completion is non-idempotent and slow: it must get
    a long timeout and must NOT be retried (each retry re-bills the query)."""
    backend = CloudBackend(api_key="pi-test")
    calls = []

    def fake_request(method, url, headers=None, **kwargs):
        calls.append(kwargs)
        raise cloud_mod.requests.ConnectionError("boom")

    monkeypatch.setattr(cloud_mod.requests, "request", fake_request)
    with pytest.raises(CloudAPIError):
        backend.query("col", "q", doc_ids=["d1"])
    assert len(calls) == 1
    assert calls[0]["timeout"] == 300


def test_query_stream_early_break_stops_background_thread(monkeypatch):
    """Consumer breaking early must signal the SSE thread to stop, not let it
    drain the whole stream in the background."""
    import threading
    backend = CloudBackend(api_key="pi-test")
    drained_all = threading.Event()

    class SlowResponse:
        status_code = 200
        text = ""
        def iter_lines(self, decode_unicode=True):
            for i in range(1000):
                yield _sse("text", f"chunk{i} ")
                # _REAL_SLEEP, not time.sleep: the _no_sleep autouse fixture
                # patches the shared time module, so time.sleep here would be a
                # no-op and the thread would race to drain all 1000 chunks
                # before the consumer's early break propagates -> flaky.
                _REAL_SLEEP(0.002)  # pace so the consumer reliably breaks first
            drained_all.set()  # only reached if the thread was NOT stopped
        def close(self):
            pass

    monkeypatch.setattr(cloud_mod.requests, "post", lambda *a, **k: SlowResponse())

    async def _run():
        async for _ in backend.query_stream("col", "q", doc_ids=["d1"]):
            break  # consume one event, then abandon

    asyncio.run(_run())
    assert not drained_all.is_set()
