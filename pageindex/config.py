# pageindex/config.py
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel


class IndexConfig(BaseModel):
    """Configuration for the PageIndex indexing pipeline.

    All fields have sensible defaults. Advanced users can override
    via LocalClient(index_config=IndexConfig(...)) or a dict.
    """
    model_config = {"extra": "forbid"}

    model: str = "gpt-4o-2024-11-20"
    retrieve_model: str | None = None
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 10
    max_token_num_each_node: int = 20000
    if_add_node_id: bool = True
    if_add_node_summary: bool = True
    if_add_doc_description: bool = True
    if_add_node_text: bool = False
    # Max concurrent in-flight LLM calls during indexing. None = use the global
    # default (get_max_concurrency(), overridable via PAGEINDEX_MAX_CONCURRENCY).
    # An explicit value here wins for this client.
    max_concurrency: int | None = None


def _env_drop_params_default() -> bool:
    return os.getenv("PAGEINDEX_DROP_PARAMS", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Per-call kwargs PageIndex passes to every litellm completion. These are
# PageIndex-OWNED and applied PER CALL — never written to litellm's shared module
# globals, so they don't leak into other libraries sharing the litellm module.
# Defaults preserve historical behavior: temperature=0 keeps structure
# extraction deterministic; drop_params=True lets a provider that rejects a param
# (e.g. temperature on some local / reasoning models) succeed by dropping it.
# Override/extend via set_llm_params(); the common drop_params case also has the
# PAGEINDEX_DROP_PARAMS env shortcut.
_LLM_PARAMS: dict = {"temperature": 0, "drop_params": _env_drop_params_default()}

# Structural kwargs PageIndex always supplies itself — not overridable here.
_RESERVED_LLM_PARAMS = ("model", "messages")


# Built-in fallback cap on concurrent in-flight LLM calls during indexing, used
# when PAGEINDEX_MAX_CONCURRENCY is unset or invalid. Kept conservative so a
# default run won't trip provider rate limits or the process fd ceiling; raise
# it via the env var / set_max_concurrency() / IndexConfig(max_concurrency=…).
_DEFAULT_MAX_CONCURRENCY = 5


def _env_max_concurrency_default() -> int:
    """Default max in-flight LLM calls, from PAGEINDEX_MAX_CONCURRENCY.

    A missing, non-integer, or non-positive value falls back to
    ``_DEFAULT_MAX_CONCURRENCY``. Read once at import; change it at runtime via
    set_max_concurrency() (a later env change doesn't apply). Bounding
    concurrency keeps a many-node document from opening one socket per node all
    at once and exhausting the process file-descriptor limit (Errno 24).
    """
    raw = os.getenv("PAGEINDEX_MAX_CONCURRENCY", str(_DEFAULT_MAX_CONCURRENCY)).strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CONCURRENCY
    return value if value > 0 else _DEFAULT_MAX_CONCURRENCY


# Process-wide default for concurrent in-flight LLM completions during indexing.
# Overridable process-wide via set_max_concurrency() / the env var above, or
# per-index via max_concurrency_scope() (used by build_index for
# IndexConfig(max_concurrency=…)). Read through get_max_concurrency().
_MAX_CONCURRENCY: int = _env_max_concurrency_default()

# Per-index override, isolated per thread / async context so concurrent indexing
# of different documents never leaks one document's limit into another (and a
# one-off override never "sticks" as the new process default). None = no
# override -> fall back to the process-wide _MAX_CONCURRENCY.
_MAX_CONCURRENCY_OVERRIDE: ContextVar[int | None] = ContextVar(
    "pageindex_max_concurrency_override", default=None
)


def get_max_concurrency() -> int:
    """Return the effective cap on concurrent in-flight LLM calls during indexing.

    A per-index override (max_concurrency_scope) wins for the current context;
    otherwise the process-wide default applies.
    """
    override = _MAX_CONCURRENCY_OVERRIDE.get()
    return override if override is not None else _MAX_CONCURRENCY


def set_max_concurrency(value: int) -> None:
    """Set the process-wide default cap on concurrent in-flight LLM calls."""
    global _MAX_CONCURRENCY
    if not isinstance(value, int) or value <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    _MAX_CONCURRENCY = value


@contextmanager
def max_concurrency_scope(value: int | None):
    """Scope a per-index max-concurrency override to the current context.

    ``value=None`` means "no override" (fall back to the process default).
    Isolated per thread / async context and reset on exit, so concurrent
    indexing doesn't leak across documents and a one-off value never becomes
    the sticky new default.
    """
    if value is not None and (not isinstance(value, int) or value <= 0):
        raise ValueError("max_concurrency must be a positive integer")
    token = _MAX_CONCURRENCY_OVERRIDE.set(value)
    try:
        yield
    finally:
        _MAX_CONCURRENCY_OVERRIDE.reset(token)


def get_llm_params() -> dict:
    """Return a copy of the per-call kwargs PageIndex passes to litellm."""
    return dict(_LLM_PARAMS)


def set_llm_params(**kwargs) -> None:
    """Override or extend the litellm completion kwargs PageIndex sends per call.

    e.g. ``set_llm_params(drop_params=False, temperature=1, num_retries=5)``.
    Applied per call; never writes litellm's global state, so it can't leak into
    other litellm users in the same process. ``model`` / ``messages`` are
    reserved (PageIndex supplies them) and rejected.
    """
    reserved = [k for k in kwargs if k in _RESERVED_LLM_PARAMS]
    if reserved:
        raise ValueError(f"cannot override reserved litellm kwargs: {reserved}")
    _LLM_PARAMS.update(kwargs)
