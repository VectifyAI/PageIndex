import threading

from pageindex.local_store import DocStore


def test_lock_serializes_concurrent_critical_sections(tmp_path):
    """DocStore.lock() must be a real mutex on every platform, including
    Windows, where fcntl is unavailable and the lock previously no-op'd
    (see #concurrent_same_name_submits_store_unique_names).

    Both threads contend for the lock at the same instant (a Barrier,
    not a sleep-based ordering guess), and the assertion is a direct
    "was anyone else in here at the same time" check taken from inside
    the critical section itself — deterministic either way, unlike an
    after-the-fact ordering check that could pass by luck on a broken
    lock if the OS happens to schedule the threads sequentially."""
    store = DocStore(str(tmp_path / "store"))
    state_guard = threading.Lock()
    occupied = False
    violations = []
    start = threading.Barrier(2)

    def worker():
        nonlocal occupied
        start.wait()
        with store.lock():
            with state_guard:
                if occupied:
                    violations.append("overlap")
                occupied = True
            # Widen the window a broken (no-op) lock would be caught in.
            threading.Event().wait(0.05)
            with state_guard:
                occupied = False

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert violations == []


def test_lock_is_reentrant_safe_across_repeated_calls(tmp_path):
    """Sequential lock() calls on the same store must not deadlock or
    error, including the msvcrt path's one-time lock-file initialization."""
    store = DocStore(str(tmp_path / "store"))
    for _ in range(5):
        with store.lock():
            pass
