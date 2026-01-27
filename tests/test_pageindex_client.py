import json
import unittest

import httpx

from pageindex.client import AsyncPageIndexClient, PageIndexClient


class TestPageIndexClient(unittest.TestCase):
    def test_chat_completions_builds_expected_request(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["json"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        transport = httpx.MockTransport(handler)

        with PageIndexClient(api_key="test-key", base_url="https://api.pageindex.ai/", transport=transport) as client:
            response = client.chat_completions(
                messages=[{"role": "user", "content": "hi"}],
                doc_id="pi-abc123",
            )

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["url"], "https://api.pageindex.ai/chat/completions")
        self.assertEqual(captured["headers"].get("api_key"), "test-key")
        self.assertEqual(captured["json"]["doc_id"], "pi-abc123")
        self.assertFalse(captured["json"]["stream"])
        self.assertIn("choices", response)

    def test_get_tree_sets_query_params(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            return httpx.Response(200, json={"status": "completed", "result": [], "retrieval_ready": True})

        transport = httpx.MockTransport(handler)

        with PageIndexClient(api_key="test-key", base_url="https://api.pageindex.ai", transport=transport) as client:
            response = client.get_tree("pi-abc123", summary=True)

        self.assertEqual(str(captured["url"]).split("?")[0], "https://api.pageindex.ai/doc/pi-abc123/")
        self.assertEqual(captured["url"].params.get("type"), "tree")
        self.assertEqual(captured["url"].params.get("summary"), "true")
        self.assertTrue(response.get("retrieval_ready"))


class TestAsyncPageIndexClient(unittest.IsolatedAsyncioTestCase):
    async def test_async_get_document(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json={"doc_id": "pi-abc123", "status": "processing"})

        transport = httpx.MockTransport(handler)

        async with AsyncPageIndexClient(
            api_key="test-key",
            base_url="https://api.pageindex.ai",
            transport=transport,
        ) as client:
            result = await client.get_document("pi-abc123")

        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["url"], "https://api.pageindex.ai/doc/pi-abc123/")
        self.assertEqual(captured["headers"].get("api_key"), "test-key")
        self.assertEqual(result.get("status"), "processing")
