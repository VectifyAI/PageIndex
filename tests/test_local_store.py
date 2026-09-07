import threading
import time

from pageindex.local_store import DocStore


def test_lock_serializes_concurrent_critical_sections(tmp_path):
    """DocStore.lock() must be a real mutex on every platform, including
    Windows, where fcntl is unavailable and the lock previously no-op'd
    (see #concurrent_same_name_submits_store_unique_names)."""
    store = DocStore(str(tmp_path / "store"))
    order = []
    lock_obj = threading.Lock()

    def worker(name):
        with store.lock():
            # If DocStore.lock() is not a real mutex, both workers can be
            # inside this block at once and their appends interleave.
            with lock_obj:
                order.append(f"{name}-enter")
            time.sleep(0.2)
            with lock_obj:
                order.append(f"{name}-exit")

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Whichever thread goes first, its enter/exit pair must be contiguous —
    # a real mutex never lets the other thread's enter land in between.
    assert order[0][-5:] == "enter"
    assert order[1][-4:] == "exit"
    assert order[0][0] == order[1][0]


def test_lock_is_reentrant_safe_across_repeated_calls(tmp_path):
    """Sequential lock() calls on the same store must not deadlock or
    error, including the msvcrt path's one-time lock-file initialization."""
    store = DocStore(str(tmp_path / "store"))
    for _ in range(5):
        with store.lock():
            pass
