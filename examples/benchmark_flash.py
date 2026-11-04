"""Reproducible Flash measurements in fresh processes, one per document/run.

Default: layout/merge only, no model. --mode fixture exercises local storage
and the model dispatch hooks with synthetic responses; it is not a model
performance benchmark. --mode configured uses the caller's existing model
configuration and must be explicitly selected.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pageindex.indexing_metrics import IndexMeasurement, aggregate_reports, phase


def fixture_model():
    """Replace dispatch endpoints, not the pipeline/retry/measurement code."""
    import litellm
    import pageindex.utils as utils
    from types import SimpleNamespace
    utils._MAP_SET_BY_USER = True
    def response(**kwargs):
        content = json.dumps({"summary": "Deterministic fixture summary.", "subsections": []})
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=content))])
    async def aresponse(**kwargs):
        return response(**kwargs)
    litellm.completion, litellm.acompletion = response, aresponse


def environment():
    packages = {}
    for name in ("pageindex", "pypdfium2", "PyPDF2", "litellm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, text=True)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_tree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def worker(args):
    if args.mode != "configured":
        # No dotenv or model-map fetch, and no real keys in offline workers.
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        for key in tuple(os.environ):
            if key.endswith("_API_KEY"):
                os.environ.pop(key, None)
    path = Path(args.worker)
    digest = None
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    except OSError:
        pass  # the measured pipeline records the actual file failure
    measurement = IndexMeasurement(parameters={
        "document": path.name, "sha256": digest, "model_mode": args.mode,
        "environment": environment(), "optimize_requested": args.optimize,
        "summary_concurrency_requested": args.summary_concurrency})
    structure = None
    with tempfile.TemporaryDirectory(prefix="pageindex-measure-") as storage:
        try:
            with redirect_stdout(sys.stderr), measurement:
                if args.mode == "layout":
                    from pageindex.flash import page_index_flash
                    result = page_index_flash(path, summary=False,
                        optimize=False if args.optimize == "off" else "merge")
                    from pageindex.flash.api import flash_rejection_reason
                    reason = flash_rejection_reason(result)
                    if reason:
                        raise ValueError(reason)
                    structure = result["structure"]
                else:
                    if args.mode == "fixture":
                        fixture_model()
                    from pageindex import PageIndexLocalClient
                    client = PageIndexLocalClient(
                        storage_path=storage,
                        summary_model="openai/fixture-model" if args.mode == "fixture" else args.model,
                        optimize=args.optimize, summary_concurrency=args.summary_concurrency,
                        summary_max_words=args.summary_max_words)
                    result = client.submit_document(str(path))
                    with phase("result_read"):
                        structure = client._api._store.get_tree(result["doc_id"])
        except Exception:
            if measurement._finished is None:
                raise  # setup failed before the measured operation could start
    report = measurement.report()
    report["tree_sha256"] = (hashlib.sha256(json.dumps(structure, sort_keys=True,
                              ensure_ascii=False).encode()).hexdigest() if structure is not None else None)
    return report


def run_worker(args, path, repeat):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(path),
               "--mode", args.mode, "--optimize", args.optimize,
               "--summary-concurrency", str(args.summary_concurrency),
               "--summary-max-words", str(args.summary_max_words)]
    if args.model:
        command += ["--model", args.model]
    start = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        if result.returncode:
            raise RuntimeError(f"worker exit {result.returncode}")
        report = json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        # A killed/crashed worker has no complete phase/request/memory report.
        report = {"status": "failed", "error_type": type(exc).__name__,
                  "elapsed_seconds": time.perf_counter() - start, "complete": False,
                  "parameters": {"document": path.name, "model_mode": args.mode},
                  "phases": {}, "model_requests": {"attempts": 0, "retry_attempts": 0},
                  "process_peak_rss_bytes": None}
    report["repeat"] = repeat
    report["worker_wall_seconds"] = time.perf_counter() - start
    return report


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", type=Path)
    parser.add_argument("--mode", choices=("layout", "fixture", "configured"), default="layout")
    parser.add_argument("--model", help="existing own-model configuration; required with --mode configured")
    parser.add_argument("--optimize", choices=("off", "merge", "full"), default="merge")
    parser.add_argument("--summary-concurrency", type=positive, default=128)
    parser.add_argument("--summary-max-words", type=positive, default=150)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--workers", type=positive, default=1)
    parser.add_argument("--timeout", type=positive, default=1800, help="whole worker deadline in seconds")
    parser.add_argument("--output", type=Path, default=Path("flash_measurements.json"))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "configured" and not args.model:
        parser.error("--mode configured requires --model; no credentials are provisioned")
    if args.mode == "layout" and args.optimize == "full":
        parser.error("layout mode supports --optimize off or merge, without model expansion")
    if args.worker:
        print(json.dumps(worker(args)))
        return 0
    if not args.pdfs:
        parser.error("provide at least one PDF")
    jobs = [(path.resolve(), repeat) for path in args.pdfs for repeat in range(1, args.repeats + 1)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(lambda job: run_worker(args, *job), jobs))
    report = {"schema_version": 1, "model_mode": args.mode,
              "workers": args.workers, "repeats": args.repeats,
              "summary": aggregate_reports(reports, time.perf_counter() - start), "reports": reports,
              "notes": ["Batch wall time includes worker startup and teardown; document timers do not sum to it.",
                        "Phase intervals overlap; elapsed_seconds is independently measured.",
                        "RSS is each worker's own process-lifetime peak, excluding child parser processes; not total batch peak memory.",
                        "Fixture mode uses synthetic responses and cannot establish real model performance or speedups.",
                        "Incomplete worker reports make request counts lower bounds."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return int(bool(report["summary"]["failed"] or report["summary"]["cancelled"]))


if __name__ == "__main__":
    raise SystemExit(main())
