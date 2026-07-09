# pageindex/config.py
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, field_validator


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
    # Per-call litellm completion kwargs for this client's indexing calls only
    # (e.g. {"temperature": 1}). None = use the process-wide defaults
    # (get_llm_params(), overridable via set_llm_params()). Scoped via
    # llm_params_scope so it doesn't leak into other concurrent indexing calls.
    llm_params: dict | None = None

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def _validate_max_concurrency_field(cls, v):
        # Reject bool before pydantic coerces True->1 / False->0, and reject
        # non-positive ints, so a bad value fails loudly instead of silently
        # serializing (Semaphore(1)) or crashing (Semaphore(0)).
        if v is not None:
            _validate_max_concurrency(v)
        return v


def _env_drop_params_default() -> bool:
    return os.getenv("PAGEINDEX_DROP_PARAMS", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Built-in per-request network timeout (seconds) for every litellm completion.
# Bounds a single in-flight call so a stalled / half-open connection (e.g. a
# flaky proxy that keeps a socket ESTABLISHED but never sends data) fails fast
# with a litellm Timeout — caught by the retry loops in index/utils.py — instead
# of hanging indefinitely. Generous enough for legitimate slow responses on
# large prompts; tune via PAGEINDEX_LLM_TIMEOUT / set_llm_params(timeout=…).
_DEFAULT_LLM_TIMEOUT = 120


def _env_llm_timeout_default():
    """Default per-request litellm timeout in seconds, from PAGEINDEX_LLM_TIMEOUT.

    A missing or non-numeric value falls back to ``_DEFAULT_LLM_TIMEOUT``. A
    value <= 0 means "no timeout" (returns None -> litellm's own default), so a
    caller can explicitly opt out. Read once at import.
    """
    raw = os.getenv("PAGEINDEX_LLM_TIMEOUT", str(_DEFAULT_LLM_TIMEOUT)).strip()
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_LLM_TIMEOUT
    return value if value > 0 else None


# Per-call kwargs PageIndex passes to every litellm completion. These are
# PageIndex-OWNED and applied PER CALL — never written to litellm's shared module
# globals, so they don't leak into other libraries sharing the litellm module.
# Defaults preserve historical behavior: temperature=0 keeps structure
# extraction deterministic; drop_params=True lets a provider that rejects a param
# (e.g. temperature on some local / reasoning models) succeed by dropping it;
# timeout bounds a single hung request (see _env_llm_timeout_default). Override/
# extend via set_llm_params(); the common drop_params / timeout cases also have
# the PAGEINDEX_DROP_PARAMS / PAGEINDEX_LLM_TIMEOUT env shortcuts.
_LLM_PARAMS: dict = {
    "temperature": 0,
    "drop_params": _env_drop_params_default(),
    "timeout": _env_llm_timeout_default(),
}

# Per-call override, isolated per thread / async context — mirrors
# _MAX_CONCURRENCY_OVERRIDE below. Without this, set_llm_params() is the only
# way to change llm params and it mutates the process-wide dict directly, so
# concurrently indexing two documents with different llm_params_scope() would
# otherwise leak one caller's settings (e.g. temperature) into the other's
# in-flight calls. None = no override -> fall back to the process-wide _LLM_PARAMS.
_LLM_PARAMS_OVERRIDE: ContextVar[dict | None] = ContextVar(
    "pageindex_llm_params_override", default=None
)

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
_MAX_CONCURRENCY_SCOPE_SEMAPHORE: ContextVar[threading.Semaphore | None] = ContextVar(
    "pageindex_max_concurrency_scope_semaphore", default=None
)


def _validate_max_concurrency(value) -> None:
    """Raise ValueError unless ``value`` is a positive int.

    ``bool`` is an ``int`` subclass, so it's rejected explicitly — otherwise
    ``set_max_concurrency(True)`` would pass and become ``Semaphore(1)``,
    silently serializing all indexing instead of failing loudly.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_concurrency must be a positive integer")


def get_max_concurrency() -> int:
    """Return the effective cap on concurrent in-flight LLM calls during indexing.

    A per-index override (max_concurrency_scope) wins for the current context;
    otherwise the process-wide default applies.
    """
    override = _MAX_CONCURRENCY_OVERRIDE.get()
    return override if override is not None else _MAX_CONCURRENCY


def _process_wide_max_concurrency() -> int:
    """The process-wide default cap, ignoring any active max_concurrency_scope
    override. This is the TRUE ceiling shared across every thread/event loop in
    the process (see index/utils.py's _llm_semaphore) — a per-call override may
    only narrow the effective cap within that ceiling, never widen it, so the
    ceiling itself must not vary with a context-local override.
    """
    return _MAX_CONCURRENCY


def _max_concurrency_scope_semaphore() -> threading.Semaphore | None:
    """Return the semaphore backing the active scoped override, if any.

    Internal helper used by the LLM call sites so sync and async completions
    share the same scoped cap when a context is copied across threads.
    """
    return _MAX_CONCURRENCY_SCOPE_SEMAPHORE.get()


def set_max_concurrency(value: int) -> None:
    """Set the process-wide default cap on concurrent in-flight LLM calls."""
    global _MAX_CONCURRENCY
    _validate_max_concurrency(value)
    _MAX_CONCURRENCY = value


@contextmanager
def max_concurrency_scope(value: int | None):
    """Scope a per-index max-concurrency override to the current context.

    ``value=None`` means "no override" (fall back to the process default).
    Isolated per thread / async context and reset on exit, so concurrent
    indexing doesn't leak across documents and a one-off value never becomes
    the sticky new default.
    """
    if value is not None:
        _validate_max_concurrency(value)
    scoped_sem = threading.Semaphore(value) if value is not None else None
    token = _MAX_CONCURRENCY_OVERRIDE.set(value)
    sem_token = _MAX_CONCURRENCY_SCOPE_SEMAPHORE.set(scoped_sem)
    try:
        yield
    finally:
        _MAX_CONCURRENCY_SCOPE_SEMAPHORE.reset(sem_token)
        _MAX_CONCURRENCY_OVERRIDE.reset(token)


def get_llm_params() -> dict:
    """Return a copy of the effective per-call kwargs PageIndex passes to litellm.

    A per-index override (llm_params_scope) is merged over the process-wide
    defaults for the current context; otherwise just the process-wide defaults
    apply.
    """
    params = dict(_LLM_PARAMS)
    override = _LLM_PARAMS_OVERRIDE.get()
    if override:
        params.update(override)
    return params


def set_llm_params(**kwargs) -> None:
    """Override or extend the process-wide default litellm completion kwargs.

    e.g. ``set_llm_params(drop_params=False, temperature=1, num_retries=5)``.
    Never writes litellm's global state, so it can't leak into other litellm
    users in the same process — but it DOES mutate PageIndex's own process-wide
    default, so it affects every concurrent caller in this process. For a
    one-off override scoped to a single indexing call, use ``llm_params_scope``
    instead. ``model`` / ``messages`` are reserved (PageIndex supplies them) and
    rejected.
    """
    reserved = [k for k in kwargs if k in _RESERVED_LLM_PARAMS]
    if reserved:
        raise ValueError(f"cannot override reserved litellm kwargs: {reserved}")
    _LLM_PARAMS.update(kwargs)


@contextmanager
def llm_params_scope(overrides: dict | None):
    """Scope a per-index override of the litellm completion kwargs to the
    current context.

    ``overrides=None`` (or ``{}``) means "no override" (fall back to the
    process-wide defaults). Isolated per thread / async context and reset on
    exit, so concurrent indexing doesn't leak one call's kwargs into another's
    and a one-off override never becomes the sticky new process default —
    mirrors ``max_concurrency_scope``.
    """
    if overrides:
        reserved = [k for k in overrides if k in _RESERVED_LLM_PARAMS]
        if reserved:
            raise ValueError(f"cannot override reserved litellm kwargs: {reserved}")
    token = _LLM_PARAMS_OVERRIDE.set(overrides or None)
    try:
        yield
    finally:
        _LLM_PARAMS_OVERRIDE.reset(token)
