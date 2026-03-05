import json
from pathlib import Path

from pageindex.profiling import profile_run, write_profile_report


def test_profile_run_collects_report():
    profile_run.last_report = None
    with profile_run(True):
        _ = sum(i for i in range(10_000))

    report = profile_run.last_report
    assert report is not None
    assert report.elapsed_seconds >= 0
    assert report.peak_memory_mb >= 0


def test_write_profile_report(tmp_path: Path):
    profile_run.last_report = None
    with profile_run(True):
        _ = [str(i) for i in range(1000)]

    out = tmp_path / "profile.json"
    write_profile_report(out, profile_run.last_report)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "elapsed_seconds" in payload
    assert "peak_memory_mb" in payload
    assert "rss_mb" in payload
