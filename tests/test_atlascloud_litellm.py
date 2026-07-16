import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import ATLASCLOUD_API_BASE, prepare_litellm_call


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
