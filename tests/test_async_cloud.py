"""Offline tests: a real loopback HTTP service plus injected pre-send faults."""
import asyncio
from contextlib import asynccontextmanager
import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from pageindex import AsyncPageIndexCloudClient, PageIndexAPIError


@asynccontextmanager
async def service(reply):
    requests, tasks, writers = [], set(), set()

    async def handle(reader, writer):
        tasks.add(asyncio.current_task())
        writers.add(writer)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            first, *lines = head.decode().split("\r\n")
            method, target, _ = first.split(" ")
            headers = dict(line.lower().split(": ", 1) for line in lines if line)
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            request = dict(method=method, target=target, headers=headers, body=body)
            requests.append(request)
            status, payload = await reply(request, len(requests))
            encoded = json.dumps(payload).encode()
            writer.write(f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(encoded)}\r\nConnection: close\r\n\r\n".encode()
                         + encoded)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            writers.discard(writer)
            tasks.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", requests
    finally:
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_cloud_document_contract_and_multipart(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"contents" * 20000)

    async def run():
        async def reply(req, count):
            if req["method"] == "POST":
                return 200, {"doc_id": "a/b", "name": "report.pdf"}
            if "metadata" in req["target"]:
                return 200, {"id": "a/b", "status": "completed", "metadata": {"x": 1}}
            if "type=ocr" in req["target"]:
                return 200, {"status": "completed", "result": [
                    {"page_index": 3, "text": "third"}, {"page_index": 1, "text": "first"}]}
            if req["target"].startswith("/docs/"):
                return 200, {"documents": [{"id": "a/b"}], "total": 1, "limit": 2, "offset": 1}
            return 200, {"status": "completed", "result": [{"title": "Tree", "nodes": []}]}

        async with service(reply) as (url, requests):
            async with AsyncPageIndexCloudClient("test-pageindex", base_url=url) as client:
                assert await client.submit_document(str(pdf), mode="mcp", beta_headers=["block_reference"],
                                                    folder_id="f/1", metadata={"x": "中文"}) == {
                                                        "doc_id": "a/b", "name": "report.pdf"}
                assert (await client.get_document("a/b"))["status"] == "completed"
                assert (await client.get_tree("a/b", node_summary=True, include_text=False))["result"][0]["title"] == "Tree"
                assert await client.get_page_content("a/b", " 1 , 3 ") == [
                    {"page_index": 3, "text": "third"}, {"page_index": 1, "text": "first"}]
                assert (await client.list_documents(2, 1, "f/1", name="r & q", recursive=True))["total"] == 1
            assert client.is_closed
            assert client._http.is_closed
            assert all(req["headers"]["api_key"] == "test-pageindex" for req in requests)
            assert len(requests[0]["body"]) == int(requests[0]["headers"]["content-length"])
            body = requests[0]["body"]
            for value in (pdf.read_bytes(), b'filename="report.pdf"', b'name="if_retrieval"',
                          b"true", b"mcp", b"block_reference", b"f/1", json.dumps({"x": "中文"}).encode()):
                assert value in body
            assert requests[1]["target"] == "/doc/a%2Fb/metadata/"
            assert parse_qs(urlsplit(requests[2]["target"]).query) == {
                "type": ["tree"], "summary": ["True"], "include_text": ["false"]}
            assert parse_qs(urlsplit(requests[4]["target"]).query) == {
                "limit": ["2"], "offset": ["1"], "folder_id": ["f/1"], "name": ["r & q"], "recursive": ["True"]}
    asyncio.run(run())


def test_concurrent_requests_and_cancellation_leave_client_usable():
    async def run():
        both_arrived, release = asyncio.Event(), asyncio.Event()
        async def reply(req, count):
            if count == 2:
                both_arrived.set()
            if count <= 2:
                await release.wait()
            return 200, {"id": req["target"], "status": "completed"}

        async with service(reply) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url) as client:
                first = asyncio.create_task(client.get_document("first"))
                second = asyncio.create_task(client.get_document("second"))
                # The service cannot receive both if HTTP blocks the loop.
                await asyncio.wait_for(both_arrived.wait(), 2)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                release.set()
                assert (await second)["status"] == "completed"
                assert (await client.get_document("third"))["status"] == "completed"
            assert len(requests) == 3
    asyncio.run(run())


@pytest.mark.parametrize("status,expected", [(503, 3), (429, 3), (401, 1), (404, 1)])
def test_get_retries_are_bounded(status, expected, monkeypatch):
    monkeypatch.setattr("pageindex.async_cloud.delay", lambda *args: 0)
    async def run():
        async def reply(req, count):
            return status, {"error": "fixture"}
        async with service(reply) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url, max_retries=2) as client:
                with pytest.raises(PageIndexAPIError) as caught:
                    await client.get_document("doc")
                assert caught.value.status_code == status
            assert len(requests) == expected
    asyncio.run(run())


@pytest.mark.parametrize("status,expected", [(429, 3), (500, 1), (408, 1), (409, 1)])
def test_upload_retries_only_explicit_refusal(status, expected, tmp_path, monkeypatch):
    monkeypatch.setattr("pageindex.async_cloud.delay", lambda *args: 0)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7 fixture")
    async def run():
        async def reply(req, count):
            return status, {"error": "fixture"}
        async with service(reply) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url) as client:
                with pytest.raises(PageIndexAPIError):
                    await client.submit_document(str(pdf))
            assert len(requests) == expected
            assert all(pdf.read_bytes() in req["body"] for req in requests)
    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
def test_upload_read_timeout_or_cancel_never_resends(cancel, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7 fixture")
    async def run():
        arrived, release = asyncio.Event(), asyncio.Event()
        async def reply(req, count):
            arrived.set()
            await release.wait()
            return 200, {"doc_id": "d"}
        async with service(reply) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url, timeout=0.15) as client:
                task = asyncio.create_task(client.submit_document(str(pdf)))
                await asyncio.wait_for(arrived.wait(), 2)
                if cancel:
                    task.cancel()
                with pytest.raises(asyncio.CancelledError if cancel else httpx.ReadTimeout):
                    await task
                assert len(requests) == 1
    asyncio.run(run())


def test_pre_send_faults_retry_and_borrowed_client_lifetime(tmp_path, monkeypatch):
    monkeypatch.setattr("pageindex.async_cloud.delay", lambda *args: 0)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF fixture")
    async def run():
        calls = []
        async def handle(request):
            calls.append(request)
            assert request.extensions["timeout"] == dict(connect=1, read=2, write=3, pool=4)
            if len(calls) < 3:
                raise httpx.ConnectTimeout("before sending", request=request)
            return httpx.Response(200, json={"doc_id": "d"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as borrowed:
            async with AsyncPageIndexCloudClient("test", http_client=borrowed,
                    timeout=httpx.Timeout(connect=1, read=2, write=3, pool=4)) as client:
                assert await client.submit_document(str(pdf)) == {"doc_id": "d"}
            assert not borrowed.is_closed
            await client.aclose()
            with pytest.raises(RuntimeError, match="closed"):
                await client.get_document("d")
        assert len(calls) == 3
    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "timeout", "cancel"])
def test_upload_file_operations_do_not_block_loop(outcome, tmp_path, monkeypatch):
    import builtins
    import threading
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF fixture")
    real_open, loop_thread, handles = builtins.open, threading.get_ident(), []
    class CheckedFile:
        def __init__(self, file):
            self.file = file
            handles.append(file)
        def __getattr__(self, name):
            attr = getattr(self.file, name)
            if name in ("read", "seek", "tell", "close"):
                def check(*args, **kwargs):
                    assert threading.get_ident() != loop_thread
                    return attr(*args, **kwargs)
                return check
            return attr
    def checked_open(path, *args, **kwargs):
        file = real_open(path, *args, **kwargs)
        if str(path) == str(pdf):
            assert threading.get_ident() != loop_thread
            return CheckedFile(file)
        return file
    monkeypatch.setattr(builtins, "open", checked_open)
    async def run():
        arrived = asyncio.Event()
        async def reply(req, count):
            arrived.set()
            if outcome != "success":
                await asyncio.Event().wait()
            return 200, {"doc_id": "d"}
        async with service(reply) as (url, _):
            async with AsyncPageIndexCloudClient("test", base_url=url, timeout=0.15) as client:
                task = asyncio.create_task(client.submit_document(str(pdf)))
                await asyncio.wait_for(arrived.wait(), 2)
                if outcome == "cancel":
                    task.cancel()
                if outcome == "success":
                    assert await task == {"doc_id": "d"}
                else:
                    with pytest.raises(asyncio.CancelledError if outcome == "cancel" else httpx.ReadTimeout):
                        await task
    asyncio.run(run())
    assert handles and all(file.closed for file in handles)


def test_get_read_timeouts_retry_but_cancellation_during_backoff_does_not(monkeypatch):
    monkeypatch.setattr("pageindex.async_cloud.delay", lambda *args: 0)
    async def run():
        async def stall(req, count):
            await asyncio.Event().wait()
        async with service(stall) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url, timeout=0.05) as client:
                with pytest.raises(httpx.ReadTimeout):
                    await client.get_document("d")
                assert len(requests) == 3
        backing_off = asyncio.Event()
        def slow_backoff(*args):
            backing_off.set()
            return 60
        monkeypatch.setattr("pageindex.async_cloud.delay", slow_backoff)
        async def unavailable(req, count):
            return 503, {"error": "unavailable"}
        async with service(unavailable) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url) as client:
                task = asyncio.create_task(client.get_document("d"))
                await asyncio.wait_for(backing_off.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert len(requests) == 1
    asyncio.run(run())


def test_cloud_key_and_retry_configuration(monkeypatch):
    monkeypatch.delenv("PAGEINDEX_API_KEY", raising=False)
    for key in (None, "", "   "):
        with pytest.raises(PageIndexAPIError, match="PageIndex API key"):
            AsyncPageIndexCloudClient(key)
    for retries in (-1, 11, True, 1.5):
        with pytest.raises(ValueError, match="max_retries"):
            AsyncPageIndexCloudClient("test", max_retries=retries)


@pytest.mark.parametrize("timeout", [None, 0, -1, float("inf"), float("nan"), httpx.Timeout(1, read=None)])
def test_timeout_cannot_be_disabled(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        AsyncPageIndexCloudClient("test", timeout=timeout)


def test_validation_and_not_ready():
    async def run():
        async def reply(req, count):
            return 200, {"result": None, "status": "processing"}
        async with service(reply) as (url, requests):
            async with AsyncPageIndexCloudClient("test", base_url=url) as client:
                with pytest.raises(ValueError):
                    await client.get_ocr("doc", "invalid")
                with pytest.raises(ValueError):
                    await client.list_documents(limit=0)
                with pytest.raises(ValueError):
                    await client.list_documents(offset=-1)
                with pytest.raises(PageIndexAPIError):
                    await client.get_page_content("doc", "0")
                assert requests == []
                with pytest.raises(PageIndexAPIError, match="not ready.*processing"):
                    await client.get_page_content("doc", "1")
    asyncio.run(run())
