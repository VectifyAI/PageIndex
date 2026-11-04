import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from pageindex.indexing_metrics import (IndexMeasurement, aggregate_reports,
    measured_phase, model_request, phase, retry_scope, _distribution, _union_seconds)


def fake_response():
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop",
        message=SimpleNamespace(content='{"summary":"fixture summary", "subsections":[]}'))])


@pytest.fixture
def fake_dispatch(monkeypatch):
    import litellm
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        return fake_response()
    async def acomplete(**kwargs):
        return complete(**kwargs)
    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(litellm, "acompletion", acomplete)
    return calls


def test_real_layout_tree_is_identical_with_measurement(sample_pdf):
    from pageindex.flash import page_index_flash
    plain = page_index_flash(sample_pdf, summary=False, optimize="merge")
    with IndexMeasurement() as run:
        observed = page_index_flash(sample_pdf, summary=False, optimize="merge")
    assert plain == observed
    report = run.report()
    assert report["status"] == "succeeded" and report["complete"]
    assert report["phases"]["parse_layout"]["spans"] == 1
    assert report["phases"]["tree_optimization"]["spans"] == 1
    assert report["phases"]["storage"]["spans"] == 0
    assert report["model_requests"]["attempts"] == 0
    assert report["parameters"]["summary"] is False
    assert report["parameters"]["parser_workers"] == 1
    assert report["parameters"]["page_count"] == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("in_loop", [False, True])
def test_local_submit_covers_all_phases_without_changing_tree(
        sample_pdf, tmp_path, fake_dispatch, in_loop):
    from pageindex import PageIndexLocalClient
    def client(directory):
        return PageIndexLocalClient(storage_path=str(directory), summary_model="gpt-test",
                                    optimize="full", summary_max_words=1)
    plain, measured = client(tmp_path / "plain"), client(tmp_path / "measured")
    before = plain.submit_document(sample_pdf)
    def submit():
        with IndexMeasurement(parameters={"fixture": True}) as run:
            result = measured.submit_document(sample_pdf)
        return result, run.report()
    async def inside():
        return submit()  # exercises the SDK's existing run_off_loop bridge
    result, report = asyncio.run(inside()) if in_loop else submit()
    assert set(result) == set(before) == {"doc_id", "name", "usage"}
    assert plain.get_tree(before["doc_id"])["result"] == measured.get_tree(result["doc_id"])["result"]
    assert all(report["phases"][name]["spans"] >= 1 for name in (
        "parse_pages", "parse_layout", "tree_optimization", "summaries", "document_description", "storage"))
    assert report["model_requests"]["attempts"] == len(fake_dispatch) // 2
    assert report["model_requests"]["retry_attempts"] == 0
    assert report["parameters"]["summary_concurrency"] == 128
    assert report["parameters"]["expand_concurrency"] == 32
    assert report["process_peak_rss_bytes"] is None or report["process_peak_rss_bytes"] > 0


def test_overlapping_async_phases_have_independent_total():
    async def run():
        ready = [asyncio.Event(), asyncio.Event()]
        @measured_phase("tree_optimization")
        async def optimize():
            ready[0].set()
            await ready[1].wait()
            await asyncio.sleep(0.02)
        @measured_phase("summaries")
        async def summaries():
            ready[1].set()
            await ready[0].wait()
            await asyncio.sleep(0.02)
        with IndexMeasurement() as measured:
            await asyncio.gather(optimize(), summaries())
        return measured.report()
    report = asyncio.run(run())
    spans = [s for s in report["spans"] if s["kind"] == "phase"]
    assert min(s["end_seconds"] for s in spans) > max(s["start_seconds"] for s in spans)
    assert sum(s["elapsed_seconds"] for s in spans) > report["elapsed_seconds"]
    assert _union_seconds(spans) <= report["elapsed_seconds"]


def test_transport_and_semantic_retries_are_counted(monkeypatch):
    import litellm
    from pageindex import utils
    monkeypatch.setattr("pageindex.retry.delay", lambda *a: 0)
    calls = []
    def fail_once(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise httpx.ConnectTimeout("fixture")
        return fake_response()
    monkeypatch.setattr(litellm, "completion", fail_once)
    with IndexMeasurement() as run:
        with phase("document_description"):
            utils.llm_completion("gpt-test", "never record this private prompt")
        with retry_scope("expand_repeat"):
            utils.llm_completion("gpt-test", "another private prompt")
    report = run.report()
    assert report["model_requests"]["attempts"] == 3
    assert report["model_requests"]["retry_attempts"] == 2
    assert report["model_requests"]["failed_attempts"] == 1
    assert report["model_requests"]["retries_by_reason"] == {"transport": 1, "expand_repeat": 1}
    assert "private prompt" not in json.dumps(report)


def test_async_retry_exhaustion_records_failure(monkeypatch):
    import litellm
    from pageindex import utils
    monkeypatch.setattr("pageindex.retry.delay", lambda *a: 0)
    monkeypatch.setattr("pageindex.retry.INDEX_RETRIES", 1)
    async def failed(**kwargs):
        raise httpx.ReadTimeout("fixture")
    monkeypatch.setattr(litellm, "acompletion", failed)
    measured = IndexMeasurement()
    with pytest.raises(utils.LLMRetriesExhausted):
        with measured:
            asyncio.run(utils.llm_acompletion("gpt-test", "prompt"))
    report = measured.report()
    assert report["status"] == "failed"
    assert report["error_type"] == "LLMRetriesExhausted"
    assert report["model_requests"]["attempts"] == 2
    assert report["model_requests"]["retry_attempts"] == 1


def test_summary_effort_fallback_is_counted(monkeypatch):
    import litellm
    from pageindex.utils import SummaryScheduler
    calls = []
    class Refused(Exception):
        status_code = 400
    async def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Refused("fixture")
        return fake_response()
    monkeypatch.setattr(litellm, "acompletion", complete)
    async def run():
        scheduler = SummaryScheduler([], [], model="gpt-test", reasoning_effort="none", drop_refused_effort=True)
        with IndexMeasurement() as measured:
            await scheduler._complete("prompt")
        return measured.report()
    report = asyncio.run(run())
    assert report["model_requests"]["attempts"] == 2
    assert report["model_requests"]["retries_by_reason"] == {"summary_effort_fallback": 1}


def test_opaque_dispatch_and_backend_retries_are_explicit(fake_dispatch, monkeypatch):
    from pageindex import utils
    async def cli(*a, **k):
        return "fixture"
    monkeypatch.setattr("pageindex.integrations.claude_agent_sdk.claude_code_complete", cli)
    with IndexMeasurement() as measured:
        utils.llm_completion("claude-code/fixture", "prompt")
        token = utils._llm_backend.set({"max_retries": 2})
        try:
            utils.llm_completion("gpt-test", "prompt")
        finally:
            utils._llm_backend.reset(token)
    report = measured.report()["model_requests"]
    assert report["attempts"] == 2 and report["opaque_dispatches"] == 1
    assert report["dispatches_with_backend_retries"] == 1


def test_independent_concurrent_measurements_and_aggregation():
    def work(n):
        with IndexMeasurement(parameters={"run": n}) as measured:
            with phase("storage"), model_request(f"fixture-{n}"):
                pass
        return measured.report()
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(work, range(2)))
    assert [r["parameters"]["run"] for r in reports] == [0, 1]
    assert all(r["model_requests"]["attempts"] == 1 for r in reports)
    summary = aggregate_reports(reports, elapsed_seconds=123)
    assert summary["elapsed_seconds"] == 123 and summary["succeeded"] == 2
    assert summary["model_request_attempts"] == 2 and summary["dispatch_counts_complete"]
    assert _distribution([1, 2, 3, 4])["p50"] == 2
    assert _distribution([])["p95"] is None


def test_failed_parse_and_storage_are_not_reported_as_success(sample_pdf, tmp_path, fake_dispatch, monkeypatch):
    from pageindex import PageIndexLocalClient
    client = PageIndexLocalClient(storage_path=str(tmp_path / "store"), summary_model="gpt-test", optimize="merge")
    def fail(*a, **k):
        raise OSError("fixture storage fault")
    monkeypatch.setattr(client._api._store, "save_document", fail)
    with pytest.raises(OSError):
        with IndexMeasurement() as measured:
            client.submit_document(sample_pdf)
    report = measured.report()
    assert report["status"] == "failed" and report["phases"]["storage"]["failed_spans"] == 1
    broken = tmp_path / "broken.pdf"
    broken.write_text("not a pdf")
    with pytest.raises(Exception):
        with IndexMeasurement() as measured:
            client.submit_document(str(broken))
    report = measured.report()
    assert report["phases"]["parse_pages"]["failed_spans"] == 1
    assert report["phases"]["storage"]["spans"] == 0


def test_cancelled_operation_and_unfinished_tasks_are_visible():
    async def run():
        with pytest.raises(asyncio.CancelledError):
            with IndexMeasurement() as measured:
                with phase("summaries"):
                    raise asyncio.CancelledError()
        assert measured.report()["status"] == "cancelled"
        entered, release = asyncio.Event(), asyncio.Event()
        async def background():
            with phase("summaries"):
                entered.set()
                await release.wait()
        with IndexMeasurement() as unfinished:
            task = asyncio.create_task(background())
            await entered.wait()
        assert unfinished.report()["complete"] is False
        assert unfinished.report()["unfinished_spans"] == 1
        release.set()
        await task
    asyncio.run(run())


def test_real_scheduler_overlap_is_recorded(monkeypatch):
    import litellm
    from pageindex.flash.api import _optimize_and_summarize
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def completion(**kwargs):
            started.set()
            await release.wait()
            return fake_response()
        async def optimize(structure, pages, lines, *, on_final, **kwargs):
            on_final(structure)
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.sleep(0.01)
            release.set()
            return {name: 0 for name in ("merges", "expands", "same_page_merges",
                    "same_page_dropped", "kept_collapsed", "before", "after")}
        monkeypatch.setattr(litellm, "acompletion", completion)
        monkeypatch.setattr("pageindex.tree_optimize.optimize", optimize)
        tree = [{"title": "A", "node_id": "0000", "start_index": 1, "end_index": 1}]
        with IndexMeasurement() as measured:
            await _optimize_and_summarize(tree, ["body " * 100], "gpt-test", "gpt-test", 1, max_words=1)
        return tree, measured.report()
    tree, report = asyncio.run(run())
    assert tree[0]["summary"] == "fixture summary"
    phases = report["phases"]
    assert phases["tree_optimization"]["elapsed_seconds"] + phases["summaries"]["elapsed_seconds"] > report["elapsed_seconds"]
    assert report["model_requests"]["attempts"] == 1
    assert next(s for s in report["spans"] if s["kind"] == "model_request")["phase"] == "summaries"


def test_expand_repeat_is_observed_at_actual_dispatch(fake_dispatch):
    from pageindex.tree_optimize import optimize
    pages = ["body " * 250] * 20
    tree = [{"title": "R", "node_id": "0000", "start_index": 1, "end_index": 20}]
    with IndexMeasurement() as measured:
        asyncio.run(optimize(tree, pages, [[p] for p in pages], model="gpt-test", empty_retries=1))
    requests = measured.report()["model_requests"]
    assert len(fake_dispatch) == requests["attempts"] == 2
    assert requests["retries_by_reason"] == {"expand_repeat": 1}


@pytest.mark.parametrize("system,expected", [("Darwin", 1024), ("Linux", 1024 * 1024), ("Other", None)])
def test_rss_units_use_platform_not_value_heuristics(system, expected, monkeypatch):
    resource = pytest.importorskip("resource")
    from pageindex.indexing_metrics import _peak_rss_bytes
    monkeypatch.setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=1024))
    monkeypatch.setattr("pageindex.indexing_metrics.platform.system", lambda: system)
    assert _peak_rss_bytes() == expected


def test_benchmark_real_pdf_and_failure_counts(sample_pdf, tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "benchmark_flash.py"
    output = tmp_path / "report.json"
    proc = subprocess.run([sys.executable, str(script), sample_pdf, str(tmp_path / "missing.pdf"),
        "--repeats", "1", "--workers", "2", "--output", str(output)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1, proc.stderr
    report = json.loads(output.read_text())
    assert report["summary"]["succeeded"] == report["summary"]["failed"] == 1
    assert report["model_mode"] == "layout"
    assert report["summary"]["model_request_attempts"] == 0
    assert report["reports"][0]["tree_sha256"]
    assert report["reports"][0]["parameters"]["sha256"]
