import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import (
    ATLASCLOUD_API_BASE,
    _llm_backend,
    llm_acompletion,
    llm_completion,
    prepare_litellm_call,
)


def completion_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_prepare_litellm_call_keeps_regular_models():
    model, kwargs = prepare_litellm_call("gpt-4o")
    assert model == "gpt-4o"
    assert kwargs == {}


def test_prepare_litellm_call_strips_litellm_prefix():
    model, kwargs = prepare_litellm_call("litellm/anthropic/claude-sonnet-4")
    assert model == "anthropic/claude-sonnet-4"
    assert kwargs == {}


def test_prepare_litellm_call_maps_atlascloud_models(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    model, kwargs = prepare_litellm_call("atlascloud/qwen/qwen3.5-flash")
    assert model == "openai/qwen/qwen3.5-flash"
    assert kwargs == {
        "api_base": ATLASCLOUD_API_BASE,
        "api_key": "test-key",
    }


def test_prepare_litellm_call_respects_custom_atlascloud_base(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    monkeypatch.setenv("ATLASCLOUD_API_BASE", "https://atlas.example/v1")
    model, kwargs = prepare_litellm_call("litellm/atlascloud/deepseek-ai/deepseek-v4-pro")
    assert model == "openai/deepseek-ai/deepseek-v4-pro"
    assert kwargs["api_base"] == "https://atlas.example/v1"
    assert kwargs["api_key"] == "test-key"


def test_prepare_litellm_call_requires_atlascloud_api_key(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        prepare_litellm_call("atlascloud/qwen/qwen3.5-flash")


def test_llm_completion_routes_atlascloud_through_litellm(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return completion_response("sync response")

    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))

    result = llm_completion("atlascloud/qwen/qwen3.5-flash", "hello")

    assert result == "sync response"
    assert calls == [{
        "api_base": ATLASCLOUD_API_BASE,
        "api_key": "test-key",
        "drop_params": True,
        "messages": [{"role": "user", "content": "hello"}],
        "model": "openai/qwen/qwen3.5-flash",
        "temperature": 0,
    }]


def test_index_backend_overrides_atlascloud_defaults(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return completion_response("sync response")

    monkeypatch.setenv("ATLASCLOUD_API_KEY", "environment-key")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    token = _llm_backend.set({
        "api_base": "https://backend.example/v1",
        "api_key": "backend-key",
        "timeout": 30,
    })
    try:
        result = llm_completion("atlascloud/qwen/qwen3.5-flash", "hello")
    finally:
        _llm_backend.reset(token)

    assert result == "sync response"
    assert calls[0]["api_base"] == "https://backend.example/v1"
    assert calls[0]["api_key"] == "backend-key"
    assert calls[0]["timeout"] == 30


def test_llm_acompletion_routes_atlascloud_through_litellm(monkeypatch):
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return completion_response("async response")

    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=acompletion))

    result = asyncio.run(
        llm_acompletion("atlascloud/qwen/qwen3.5-flash", "hello")
    )

    assert result == "async response"
    assert calls == [{
        "api_base": ATLASCLOUD_API_BASE,
        "api_key": "test-key",
        "drop_params": True,
        "messages": [{"role": "user", "content": "hello"}],
        "model": "openai/qwen/qwen3.5-flash",
        "temperature": 0,
    }]
