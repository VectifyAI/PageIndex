import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pydantic
import pytest

from pageindex.config import (
    IndexConfig,
    _env_llm_timeout_default,
    _env_max_concurrency_default,
    _max_concurrency_scope_semaphore,
    get_llm_params,
    get_max_concurrency,
    llm_params_scope,
    max_concurrency_scope,
    set_llm_params,
    set_max_concurrency,
)
from pageindex.index.utils import (
    _llm_semaphore,
    _process_ceiling_semaphore,
    llm_acompletion,
    llm_completion,
)


@pytest.fixture(autouse=True)
def _restore_max_concurrency():
    """Keep tests isolated — the concurrency setting is a module global."""
    prev = get_max_concurrency()
    yield
    set_max_concurrency(prev)


@pytest.fixture(autouse=True)
def _restore_llm_params():
    """Keep tests isolated — llm params are a module global too."""
    prev = get_llm_params()
    yield
    set_llm_params(**prev)


async def _nested_llm_load(state, *, branches=5, leaves=5):
    """Drive branches*leaves leaf calls, nested two levels deep, each holding
    the shared per-loop LLM semaphore — the exact shape of the indexing pipeline
    (tree_parser gather -> per-node gather -> leaf LLM call)."""

    async def leaf():
        async with _llm_semaphore():
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1

    async def branch():
        await asyncio.gather(*(leaf() for _ in range(leaves)))

    await asyncio.gather(*(branch() for _ in range(branches)))


def test_llm_semaphore_bounds_concurrency_even_when_nested():
    # The core fix: the cap is a TRUE global bound even when acquired from
    # deeply nested gathers. 25 leaf calls nested two levels, cap 3 -> peak 3.
    # A per-gather-call semaphore (the previous design) would let this reach
    # branches*leaves and blow past the cap.
    set_max_concurrency(3)
    state = {"in_flight": 0, "peak": 0}
    asyncio.run(_nested_llm_load(state, branches=5, leaves=5))
    assert state["peak"] == 3


def test_llm_semaphore_is_a_true_process_wide_ceiling_across_threads():
    # The bug this fixes: each asyncio.run() (its own event loop) used to get
    # an independent full-size semaphore, so N concurrently-indexing threads
    # multiplied the effective cap by N. 2 threads, cap=3 -> combined peak
    # must stay at 3, not 6.
    set_max_concurrency(3)
    state = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    async def leaf():
        async with _llm_semaphore():
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            await asyncio.sleep(0.05)
            with lock:
                state["in_flight"] -= 1

    async def load():
        await asyncio.gather(*(leaf() for _ in range(5)))

    threads = [threading.Thread(target=lambda: asyncio.run(load())) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert state["peak"] == 3


def test_llm_semaphore_cancellation_while_waiting_does_not_leak_a_permit():
    # A blocking ceiling_sem.acquire() run via asyncio.to_thread() would leak a
    # permit under cancellation: the worker thread can't be interrupted, so if
    # the awaiting coroutine is cancelled while the thread is still parked
    # inside acquire(), the thread can go on to actually acquire the permit
    # *after* the coroutine already unwound, and the matching finally:
    # release() never runs for that attempt. Cancel a task waiting on an
    # already-exhausted ceiling and confirm the permit count fully recovers.
    set_max_concurrency(1)

    async def run():
        async def hold():
            async with _llm_semaphore():
                await asyncio.sleep(10)

        holder = asyncio.create_task(hold())
        await asyncio.sleep(0.1)  # let it acquire the single permit

        waiter = asyncio.create_task(_llm_semaphore().__aenter__())
        await asyncio.sleep(0.1)  # let it start waiting for the permit
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        holder.cancel()
        with pytest.raises(asyncio.CancelledError):
            await holder

        await asyncio.sleep(0.2)  # give any orphaned acquire a chance to land
        assert _process_ceiling_semaphore()._value == 1

    asyncio.run(run())


def test_scoped_semaphore_cancellation_while_waiting_does_not_over_release():
    # Mirror of the ceiling test for the scoped override: a coroutine cancelled
    # while polling for a scoped permit must NOT let the finally release a permit
    # it never acquired, which would inflate the scoped cap for later calls.
    set_max_concurrency(5)  # high ceiling so the scope is the narrower cap

    async def run():
        with max_concurrency_scope(1):
            sem = _max_concurrency_scope_semaphore()
            assert sem._value == 1

            async def hold():
                async with _llm_semaphore():
                    await asyncio.sleep(10)

            holder = asyncio.create_task(hold())
            await asyncio.sleep(0.1)  # holder takes the single scoped permit

            waiter = asyncio.create_task(_llm_semaphore().__aenter__())
            await asyncio.sleep(0.1)  # waiter is now polling for the scoped permit
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

            holder.cancel()
            with pytest.raises(asyncio.CancelledError):
                await holder

            await asyncio.sleep(0.2)
            assert sem._value == 1  # fully recovered, not inflated to 2

    asyncio.run(run())


def test_llm_semaphore_uses_scoped_override():
    # A per-index max_concurrency_scope active when the loop's semaphore is first
    # created must set its size, and must not mutate the process default.
    set_max_concurrency(10)
    state = {"in_flight": 0, "peak": 0}

    async def run():
        with max_concurrency_scope(2):
            await _nested_llm_load(state, branches=4, leaves=4)

    asyncio.run(run())
    assert state["peak"] == 2
    assert get_max_concurrency() == 10


def test_llm_acompletion_holds_the_shared_semaphore(monkeypatch):
    # Prove llm_acompletion actually acquires the shared cap around the async
    # network call.
    set_max_concurrency(3)
    state = {"in_flight": 0, "peak": 0}

    async def fake_acompletion(**kwargs):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)

    async def run():
        await asyncio.gather(*(llm_acompletion("gpt-x", f"p{i}") for i in range(20)))

    asyncio.run(run())
    assert state["peak"] == 3


def test_llm_acompletion_passes_a_timeout_to_litellm(monkeypatch):
    # A per-request timeout must reach litellm so a hung / half-open connection
    # fails fast instead of stalling indexing forever. It rides get_llm_params(),
    # so both the default and an override flow through automatically.
    seen = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    asyncio.run(llm_acompletion("gpt-x", "hi"))
    assert "timeout" in seen and seen["timeout"] == get_llm_params()["timeout"]


def test_llm_completion_degrades_to_empty_on_exhaustion(monkeypatch, caplog):
    # A persistently-failing LLM call must NOT abort the whole index: it returns
    # an empty result (logged at WARNING, not silent) so callers degrade
    # (extract_json('') -> {} -> .get(default)) and the rest still indexes.
    import logging

    def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr("litellm.completion", boom)
    monkeypatch.setattr("pageindex.index.utils.time.sleep", lambda *a: None)  # skip retry backoff

    with caplog.at_level(logging.WARNING, logger="pageindex.index.utils"):
        assert llm_completion("gpt-x", "hi") == ""
        assert llm_completion("gpt-x", "hi", return_finish_reason=True) == ("", "error")
    assert any("failed after" in r.message and "degrading" in r.message for r in caplog.records)


def test_llm_acompletion_degrades_to_empty_on_exhaustion(monkeypatch):
    # Async counterpart: exhausted retries return "" instead of raising, so the
    # return_exceptions gathers see a plain empty result and callers degrade.
    async def boom(**kwargs):
        raise RuntimeError("provider down")

    async def _instant_sleep(*a):
        return None

    monkeypatch.setattr("litellm.acompletion", boom)
    monkeypatch.setattr("pageindex.index.utils.asyncio.sleep", _instant_sleep)  # skip retry backoff

    assert asyncio.run(llm_acompletion("gpt-x", "hi")) == ""


def test_llm_completion_holds_the_shared_semaphore(monkeypatch):
    # Sync litellm.completion calls must share the same process-wide cap as the
    # async path; otherwise concurrent indexing threads can exceed
    # set_max_concurrency().
    set_max_concurrency(1)
    state = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    def fake_completion(**kwargs):
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr("litellm.completion", fake_completion)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: llm_completion("gpt-x", f"p{i}"), range(6)))

    assert results == ["ok"] * 6
    assert state["peak"] == 1


def test_run_async_propagates_scope_into_worker_thread():
    # When build_index runs inside an already-running loop, _run_async hops to a
    # worker thread. The max_concurrency_scope override must ride along (copied
    # context) and still bound the (nested) LLM load in that worker loop.
    from pageindex.index.pipeline import _run_async

    set_max_concurrency(10)
    state = {"in_flight": 0, "peak": 0}

    async def outer():
        # We're inside a running loop -> _run_async uses the worker thread.
        with max_concurrency_scope(3):
            _run_async(_nested_llm_load(state, branches=4, leaves=4))

    asyncio.run(outer())
    assert state["peak"] == 3


def test_set_get_max_concurrency_round_trip():
    set_max_concurrency(3)
    assert get_max_concurrency() == 3


def test_set_max_concurrency_rejects_invalid():
    # bool is an int subclass -> must be rejected, not silently -> Semaphore(1).
    for bad in (0, -1, True, False, 2.5, "3", None):
        with pytest.raises(ValueError):
            set_max_concurrency(bad)


def test_env_default_parsing(monkeypatch):
    monkeypatch.delenv("PAGEINDEX_MAX_CONCURRENCY", raising=False)
    assert _env_max_concurrency_default() == 5
    monkeypatch.setenv("PAGEINDEX_MAX_CONCURRENCY", "20")
    assert _env_max_concurrency_default() == 20
    monkeypatch.setenv("PAGEINDEX_MAX_CONCURRENCY", "garbage")
    assert _env_max_concurrency_default() == 5
    monkeypatch.setenv("PAGEINDEX_MAX_CONCURRENCY", "0")
    assert _env_max_concurrency_default() == 5


def test_env_llm_timeout_parsing(monkeypatch):
    monkeypatch.delenv("PAGEINDEX_LLM_TIMEOUT", raising=False)
    assert _env_llm_timeout_default() == 120
    monkeypatch.setenv("PAGEINDEX_LLM_TIMEOUT", "45")
    assert _env_llm_timeout_default() == 45
    monkeypatch.setenv("PAGEINDEX_LLM_TIMEOUT", "garbage")
    assert _env_llm_timeout_default() == 120
    monkeypatch.setenv("PAGEINDEX_LLM_TIMEOUT", "0")  # <=0 opts out of the timeout
    assert _env_llm_timeout_default() is None


def test_index_config_max_concurrency_field():
    # Default is None → "use the global/env default"; explicit value overrides.
    assert IndexConfig().max_concurrency is None
    assert IndexConfig(max_concurrency=7).max_concurrency == 7


def test_index_config_rejects_bool_and_non_positive_max_concurrency():
    # bool would otherwise be coerced by pydantic to 1/0; both must be rejected.
    for bad in (True, False, 0, -1):
        with pytest.raises(pydantic.ValidationError):
            IndexConfig(max_concurrency=bad)


def test_max_concurrency_scope_overrides_then_restores():
    # A per-index override applies inside the scope and, crucially, does NOT
    # stick as the new process default afterwards (no stickiness).
    set_max_concurrency(10)
    with max_concurrency_scope(3):
        assert get_max_concurrency() == 3
    assert get_max_concurrency() == 10


def test_max_concurrency_scope_none_is_a_no_op():
    set_max_concurrency(8)
    with max_concurrency_scope(None):
        assert get_max_concurrency() == 8
    assert get_max_concurrency() == 8


def test_max_concurrency_scope_rejects_invalid():
    for bad in (0, -1, True, False):
        with pytest.raises(ValueError):
            with max_concurrency_scope(bad):
                pass


def test_max_concurrency_scope_is_isolated_across_threads():
    # A per-index override in one indexing thread must not leak into another
    # thread indexing a different document concurrently. The override is a
    # ContextVar, so it's invisible outside its own context.
    set_max_concurrency(10)
    seen = {}
    barrier = threading.Barrier(2)

    def worker():
        with max_concurrency_scope(2):
            barrier.wait()            # let main read while we're inside the scope
            seen["worker"] = get_max_concurrency()
            barrier.wait()

    t = threading.Thread(target=worker)
    t.start()
    barrier.wait()
    seen["main"] = get_max_concurrency()
    barrier.wait()
    t.join()

    assert seen["worker"] == 2   # worker sees its own scoped override
    assert seen["main"] == 10    # main is unaffected by the worker's scope


def test_llm_params_scope_overrides_then_restores():
    set_llm_params(temperature=0)
    with llm_params_scope({"temperature": 1}):
        assert get_llm_params()["temperature"] == 1
    assert get_llm_params()["temperature"] == 0


def test_llm_params_scope_none_is_a_no_op():
    set_llm_params(temperature=0)
    with llm_params_scope(None):
        assert get_llm_params()["temperature"] == 0
    assert get_llm_params()["temperature"] == 0


def test_llm_params_scope_rejects_reserved_keys():
    with pytest.raises(ValueError):
        with llm_params_scope({"model": "x"}):
            pass


def test_llm_params_scope_is_isolated_across_threads():
    set_llm_params(temperature=0)
    seen = {}
    barrier = threading.Barrier(2)

    def worker():
        with llm_params_scope({"temperature": 1}):
            barrier.wait()
            seen["worker"] = get_llm_params()["temperature"]
            barrier.wait()

    t = threading.Thread(target=worker)
    t.start()
    barrier.wait()
    seen["main"] = get_llm_params()["temperature"]
    barrier.wait()
    t.join()

    assert seen["worker"] == 1
    assert seen["main"] == 0


def test_llm_params_scope_does_not_leak_across_concurrent_indexing():
    # The bug this fixes: set_llm_params() mutates a bare process-wide dict, so
    # two documents indexed concurrently with different llm_params_scope()
    # overrides must not see each other's temperature.
    set_llm_params(temperature=0)
    seen = {"a": None, "b": None}

    async def job(name, temperature, delay_before, delay_after):
        with llm_params_scope({"temperature": temperature}):
            await asyncio.sleep(delay_before)
            seen[name] = get_llm_params()["temperature"]
            await asyncio.sleep(delay_after)

    async def run():
        await asyncio.gather(
            job("a", 1, 0.0, 0.05),
            job("b", 2, 0.02, 0.0),
        )

    asyncio.run(run())
    assert seen["a"] == 1
    assert seen["b"] == 2


def test_utils_star_import_does_not_leak_config_name():
    # `from .utils import *` (used by the page_index modules) must not export a
    # name `config` that would shadow the real pageindex.config submodule for
    # those modules. The SimpleNamespace alias is now `_config`.
    ns = {}
    exec("from pageindex.index.utils import *", ns)
    assert "config" not in ns
