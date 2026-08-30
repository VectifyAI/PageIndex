"""Subscription-backed CLI model backends.

Model strings ``claude-cli/<model>`` and ``codex-cli/<model>`` route to a
local CLI subprocess instead of LiteLLM (see utils.llm_completion)."""
from __future__ import annotations

import os

from .base import CliBackend, CliBackendError, CliResult
from .claude_cli import ClaudeCliBackend
from .codex_cli import CodexCliBackend

CLI_PREFIXES: dict[str, type[CliBackend]] = {
    "claude-cli": ClaudeCliBackend,
    "codex-cli": CodexCliBackend,
}
_registry: dict[str, CliBackend] = {}


def is_cli_model(model) -> bool:
    return (isinstance(model, str) and "/" in model
            and model.split("/", 1)[0] in CLI_PREFIXES) or model in _registry


def register(model: str, backend: CliBackend) -> None:
    """Bind an explicit backend instance to a model string (tests, custom CLIs)."""
    _registry[model] = backend


def resolve(model) -> CliBackend | None:
    if model in _registry:
        return _registry[model]
    if not is_cli_model(model):
        return None
    prefix, _, name = model.partition("/")
    concurrency = int(os.environ.get("PAGEINDEX_CLI_CONCURRENCY", "3"))
    backend = CLI_PREFIXES[prefix](name, concurrency=concurrency)
    _registry[model] = backend
    return backend


__all__ = ["CliBackend", "CliBackendError", "CliResult", "ClaudeCliBackend",
           "CodexCliBackend", "CLI_PREFIXES", "is_cli_model", "register", "resolve"]
