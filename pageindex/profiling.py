from __future__ import annotations

import contextlib
import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ProfileReport:
    elapsed_seconds: float
    peak_memory_mb: float
    rss_mb: float | None


@contextlib.contextmanager
def profile_run(enabled: bool = False):
    if not enabled:
        yield None
        return

    tracemalloc.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_mb = None
        try:
            import resource

            # Linux returns KB, macOS returns bytes.
            ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = ru_maxrss / 1024.0
            if rss_mb > 10_000:  # likely macOS bytes -> convert to MB
                rss_mb = ru_maxrss / (1024.0 * 1024.0)
        except Exception:
            rss_mb = None

        yield_data = ProfileReport(
            elapsed_seconds=round(elapsed, 3),
            peak_memory_mb=round(peak / (1024.0 * 1024.0), 3),
            rss_mb=round(rss_mb, 3) if rss_mb is not None else None,
        )

        # stash report on context manager instance for caller retrieval
        profile_run.last_report = yield_data


def write_profile_report(output_path: str | os.PathLike[str], report: ProfileReport) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


profile_run.last_report = None
