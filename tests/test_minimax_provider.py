import os
from unittest.mock import patch

import httpx
import litellm
import pytest

from pageindex.client import PageIndexClient


@pytest.mark.parametrize(
    ("model", "api_base", "api_key_env", "api_base_env"),
    [
        (
            "minimax/MiniMax-M3",
            "https://api.minimax.io/v1",
            "MINIMAX_API_KEY",
            "MINIMAX_API_BASE",
        ),
        (
            "minimax/MiniMax-M2.7",
            "https://api.minimaxi.com/v1",
            "MINIMAX_API_KEY",
            "MINIMAX_API_BASE",
        ),
        (
            "anthropic/MiniMax-M3",
            "https://api.minimax.io/anthropic",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_BASE",
        ),
        (
            "anthropic/MiniMax-M2.7",
            "https://api.minimaxi.com/anthropic",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_API_BASE",
        ),
    ],
)
def test_client_configures_provider_environment(model, api_base, api_key_env, api_base_env):
    with patch.dict(os.environ, {}, clear=True):
        PageIndexClient(model=model, api_key="test-key", api_base=api_base)

        assert os.environ[api_key_env] == "test-key"
        assert os.environ[api_base_env] == api_base


def test_litellm_routes_compatible_endpoints():
    captured_urls = []

    def fake_send(_client, request, *args, **kwargs):
        captured_urls.append(str(request.url))
        if "/anthropic/" in str(request.url):
            payload = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "MiniMax-M3",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            payload = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "MiniMax-M3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        return httpx.Response(200, json=payload, request=request)

    cases = [
        ("minimax/MiniMax-M3", "https://api.minimax.io/v1"),
        ("minimax/MiniMax-M2.7", "https://api.minimaxi.com/v1"),
        ("anthropic/MiniMax-M3", "https://api.minimax.io/anthropic"),
        ("anthropic/MiniMax-M2.7", "https://api.minimaxi.com/anthropic"),
    ]

    with patch.object(httpx.Client, "send", fake_send):
        for model, api_base in cases:
            response = litellm.completion(
                model=model,
                api_base=api_base,
                api_key="test-key",
                messages=[{"role": "user", "content": "hello"}],
            )
            assert response.choices[0].message.content == "ok"

    assert captured_urls == [
        "https://api.minimax.io/v1/chat/completions",
        "https://api.minimaxi.com/v1/chat/completions",
        "https://api.minimax.io/anthropic/v1/messages",
        "https://api.minimaxi.com/anthropic/v1/messages",
    ]
