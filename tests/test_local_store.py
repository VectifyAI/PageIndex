"""Regression tests for DocStore.lock() on Windows (issue #561).

Windows has no fcntl, so lock() used to silently yield without acquiring any
lock: concurrent submit_document() calls could both observe the same filename
as available and store documents under duplicate names. These tests simulate
the Windows branch on any platform by hiding fcntl and providing a fake
msvcrt whose locking() truly serializes.
"""

import sys
import threading
import time

import pytest

from pageindex import PageIndexClient
from pageindex.local_api import LocalAPI
from pageindex.local_store import DocStore


@pytest.fixture
def windows_lock_env(monkeypatch):
    """Simulate Windows: fcntl unavailable; msvcrt.locking backed by a real
    lock, recording every acquire/release so tests can prove the OS-level
    lock was actually taken."""
    monkeypatch.setitem(sys.modules, "fcntl", None)
    calls = {"lock": 0, "unlock": 0}
    guard = threading.Lock()

    class FakeMsvcrt:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, mode, nbytes):
            assert nbytes >= 1
            if mode == FakeMsvcrt.LK_LOCK:
                guard.acquire()
                calls["lock"] += 1
            elif mode == FakeMsvcrt.LK_UNLCK:
                calls["unlock"] += 1
                guard.release()
            else:
                raise AssertionError(f"unexpected msvcrt locking mode: {mode}")

    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)
    return calls


def test_lock_windows_branch_serializes_threads(tmp_path, windows_lock_env):
    """On the Windows branch, concurrent critical sections must not overlap —
    across threads and across separate DocStore instances on one directory."""
    root = tmp_path / "store"
    stores = [DocStore(str(root)), DocStore(str(root))]
    workers, iterations = 8, 25
    barrier = threading.Barrier(workers)
    state = {"n": 0}

    def worker(i):
        store = stores[i % 2]
        barrier.wait()
        for _ in range(iterations):
            with store.lock():
                seen = state["n"]
                # Widen the race window: without a lock, increments are lost.
                time.sleep(0.001)
                state["n"] = seen + 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["n"] == workers * iterations
    # The OS-level lock was actually taken, not just the in-process one.
    assert windows_lock_env["lock"] == windows_lock_env["unlock"] == workers * iterations


def test_concurrent_same_name_submits_store_unique_names(
    tmp_path, sample_pdf, windows_lock_env, monkeypatch
):
    """Issue #561 repro: concurrent same-name submits on the Windows branch
    must receive distinct stored names."""
    monkeypatch.setattr(
        LocalAPI, "_index_flash", lambda self, file_path: ([], "stub description")
    )
    client = PageIndexClient(storage_path=str(tmp_path / "store"))
    workers = 8
    barrier = threading.Barrier(workers)
    names = [None] * workers
    errors = []

    def worker(i):
        barrier.wait()
        try:
            names[i] = client.submit_document(str(sample_pdf))["name"]
        except Exception as exc:  # noqa: BLE001 - surfaced via the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    expected = ["sample.pdf"] + [f"sample_{i}.pdf" for i in range(1, workers)]
    assert sorted(names) == expected
