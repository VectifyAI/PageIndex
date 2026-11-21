import json

import pytest

from pageindex.client import PageIndexClient
from pageindex.utils import default_summary_reasoning_effort, llm_completion


def test_flash_uses_server_default_thinking(monkeypatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {})
    assert default_summary_reasoning_effort("minimax/MiniMax-M3.1-Flash") is None
    assert default_summary_reasoning_effort("litellm/minimax/MiniMax-M3.1-Flash") is None


def test_m3_omits_ignored_effort_field(monkeypatch):
    monkeypatch.setattr("litellm.get_model_info", lambda model: {})
    assert default_summary_reasoning_effort("minimax/MiniMax-M3") is None


def test_openai_compatible_wire_request(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return type("Response", (), {
            "choices": [type("Choice", (), {
                "message": type("Message", (), {"content": "ok"})(),
                "finish_reason": "stop",
            })()],
        })()

    monkeypatch.setattr("litellm.completion", fake_completion)
    token = __import__("pageindex.utils", fromlist=["_llm_backend"])._llm_backend.set({
        "api_key": "test-key",
        "api_base": "https://api.minimax.io/v1",
    })
    try:
        assert llm_completion("minimax/MiniMax-M3.1-Flash", "hello") == "ok"
    finally:
        __import__("pageindex.utils", fromlist=["_llm_backend"])._llm_backend.reset(token)
    assert captured["model"] == "minimax/MiniMax-M3.1-Flash"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["api_base"] == "https://api.minimax.io/v1"
    assert "reasoning_effort" not in captured


def test_client_preserves_model_and_backend_configuration(tmp_path):
    client = PageIndexClient(
        mode="local",
        index={
            "model": "minimax/MiniMax-M3.1-Flash",
            "summary_model": "minimax/MiniMax-M3.1-Flash",
            "backend": {
                "api_key": "test-key",
                "api_base": "https://api.minimax.io/v1",
            },
            "storage_path": str(tmp_path),
        },
        chat={
            "model": "minimax/MiniMax-M3",
            "backend": {
                "api_key": "test-key",
                "api_base": "https://api.minimax.io/v1",
            },
        },
    )
    assert client.model == "minimax/MiniMax-M3.1-Flash"
    assert client.summary_model == "minimax/MiniMax-M3.1-Flash"
    assert client.chat_model == "minimax/MiniMax-M3"
    assert client._api._index_backend["api_base"] == "https://api.minimax.io/v1"


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("minimax/MiniMax-M3.1-Flash", "https://api.minimax.io/v1"),
        ("minimax/MiniMax-M3", "https://api.minimaxi.com/v1"),
    ],
)
def test_litellm_endpoint_urls_offline(monkeypatch, model, base_url):
    litellm = pytest.importorskip("litellm")
    httpx = pytest.importorskip("httpx")
    captured = []

    def fake_send(_client, request, *args, **kwargs):
        captured.append({"url": str(request.url), "body": json.loads(request.content)})
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": model.rsplit("/", 1)[-1],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    response = litellm.completion(
        model=model,
        api_base=base_url,
        api_key="test-key",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert response.choices[0].message.content == "ok"
    assert captured[0]["url"] == base_url + "/chat/completions"
    assert "reasoning_effort" not in captured[0]["body"]
