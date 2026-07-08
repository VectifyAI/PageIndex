import asyncio
import threading

import pytest

from pageindex.config import (
    IndexConfig,
    _env_max_concurrency_default,
    get_max_concurrency,
    max_concurrency_scope,
    set_max_concurrency,
)
from pageindex.index.utils import bounded_gather


@pytest.fixture(autouse=True)
def _restore_max_concurrency():
    """Keep tests isolated — the concurrency setting is a module global."""
    prev = get_max_concurrency()
    yield
    set_max_concurrency(prev)


def test_bounded_gather_never_exceeds_the_cap():
    set_max_concurrency(5)
    state = {"in_flight": 0, "peak": 0}

    async def worker(i):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return i

    async def run():
        return await bounded_gather(worker(i) for i in range(30))

    results = asyncio.run(run())

    # Order is preserved (gather semantics) and the cap is respected: with 30
    # tasks and 5 slots, exactly 5 run at once — never the unbounded 30 that
    # exhausted file descriptors.
    assert results == list(range(30))
    assert state["peak"] == 5


def test_bounded_gather_propagates_return_exceptions():
    async def ok():
        return "ok"

    async def boom():
        raise ValueError("boom")

    async def run():
        return await bounded_gather([ok(), boom()], return_exceptions=True)

    results = asyncio.run(run())
    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)


def test_set_get_max_concurrency_round_trip():
    set_max_concurrency(3)
    assert get_max_concurrency() == 3


def test_set_max_concurrency_rejects_non_positive():
    with pytest.raises(ValueError):
        set_max_concurrency(0)
    with pytest.raises(ValueError):
        set_max_concurrency(-1)


def test_env_default_parsing(monkeypatch):
    monkeypatch.delenv("PAGEINDEX_MAX_CONCURRENCY", raising=False)
    assert _env_max_concurrency_default() == 5
    monkeypatch.setenv("PAGEINDEX_MAX_CONCURRENCY", "20")
    assert _env_max_concurrency_default() == 20
    monkeypatch.setenv("PAGEINDEX_MAX_CONCURRENCY", "garbage")
    assert _env_max_concurrency_default() == 5
    monkeypatch.setenv("PAGEINDEX_MAX_CONCURRENCY", "0")
    assert _env_max_concurrency_default() == 5


def test_index_config_max_concurrency_field():
    # Default is None → "use the global/env default"; explicit value overrides.
    assert IndexConfig().max_concurrency is None
    assert IndexConfig(max_concurrency=7).max_concurrency == 7


def test_max_concurrency_scope_overrides_then_restores():
    # A per-index override applies inside the scope and, crucially, does NOT
    # stick as the new process default afterwards (Finding A: no stickiness).
    set_max_concurrency(10)
    with max_concurrency_scope(3):
        assert get_max_concurrency() == 3
    assert get_max_concurrency() == 10


def test_max_concurrency_scope_none_is_a_no_op():
    set_max_concurrency(8)
    with max_concurrency_scope(None):
        assert get_max_concurrency() == 8
    assert get_max_concurrency() == 8


def test_max_concurrency_scope_rejects_non_positive():
    with pytest.raises(ValueError):
        with max_concurrency_scope(0):
            pass
    with pytest.raises(ValueError):
        with max_concurrency_scope(-1):
            pass


def test_max_concurrency_scope_is_isolated_across_threads():
    # A per-index override in one indexing thread must not leak into another
    # thread indexing a different document concurrently (Finding B). The
    # override is a ContextVar, so it's invisible outside its own context.
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


def test_bounded_gather_respects_scoped_override():
    # bounded_gather reads the cap at semaphore-creation time; a surrounding
    # max_concurrency_scope must win and must not mutate the process default.
    set_max_concurrency(10)
    state = {"in_flight": 0, "peak": 0}

    async def worker(i):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return i

    async def run():
        with max_concurrency_scope(4):
            return await bounded_gather(worker(i) for i in range(20))

    asyncio.run(run())
    assert state["peak"] == 4
    assert get_max_concurrency() == 10


def test_run_async_propagates_scope_into_worker_thread():
    # When build_index runs inside an already-running loop, _run_async hops to a
    # worker thread. The max_concurrency_scope override must ride along (copied
    # context) instead of silently falling back to the process default.
    from pageindex.index.pipeline import _run_async

    set_max_concurrency(10)
    state = {"in_flight": 0, "peak": 0}

    async def worker(i):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        await asyncio.sleep(0.01)
        state["in_flight"] -= 1
        return i

    async def inner():
        return await bounded_gather(worker(i) for i in range(20))

    async def outer():
        # We're inside a running loop -> _run_async uses the worker thread.
        with max_concurrency_scope(3):
            _run_async(inner())

    asyncio.run(outer())
    assert state["peak"] == 3


def test_utils_star_import_does_not_leak_config_name():
    # `from .utils import *` (used by the page_index modules) must not export a
    # name `config` that would shadow the real pageindex.config submodule for
    # those modules (Finding D). The SimpleNamespace alias is now `_config`.
    ns = {}
    exec("from pageindex.index.utils import *", ns)
    assert "config" not in ns
