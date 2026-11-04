"""Opt-in indexing measurements, separate from the returned/stored tree.

This reports SDK dispatches, not provider-internal HTTP requests. Phase
intervals can overlap: their sums are never the end-to-end elapsed time.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import contextvars
import copy
from functools import wraps
import inspect
import json
import math
import platform
import threading
import time

_active = contextvars.ContextVar("pageindex_measurement", default=None)
_phase = contextvars.ContextVar("pageindex_measurement_phase", default=None)
_retry_reason = contextvars.ContextVar("pageindex_measurement_retry", default=None)

PHASES = ("parse_pages", "parse_layout", "tree_optimization", "summaries",
          "document_description", "storage")


def _distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None,
                "p99": None, "max": None, "mean": None}
    # Nearest-rank quantiles; a sample of one has that value at every rank.
    return {"count": len(values), "min": values[0], "max": values[-1],
            "mean": sum(values) / len(values),
            **{f"p{q}": values[max(0, math.ceil(len(values) * q / 100) - 1)]
               for q in (50, 95, 99)}}


def _union_seconds(spans):
    total, end = 0.0, 0.0
    for span in sorted(spans, key=lambda item: item["start_seconds"]):
        start, stop = span["start_seconds"], span["end_seconds"]
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _peak_rss_bytes():
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return int(peak)
        if platform.system() == "Linux":
            return int(peak * 1024)
    except (ImportError, OSError, ValueError):
        pass
    return None


def _outcome(exc):
    return "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed" if exc else "succeeded"


class IndexMeasurement:
    """Measure one indexing operation, including its failures.

    ``with IndexMeasurement(parameters={...}) as run: ...`` then
    ``run.report()``. A fresh instance is required for each operation.
    Parameters are JSON data supplied by the caller plus selected pipeline
    settings; prompts, responses and credentials are never captured by hooks.
    RSS is the *process lifetime* high-water mark, including native memory;
    use isolated worker processes for comparable per-document measurements.
    """

    def __init__(self, parameters=None):
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError("parameters must be a JSON object")
        self._parameters = json.loads(json.dumps(parameters or {}, allow_nan=False))
        self._lock = threading.Lock()
        self._spans = []
        self._in_flight = 0
        self._started = None
        self._finished = None

    def __enter__(self):
        if self._started is not None or _active.get() is not None:
            raise RuntimeError("Use a fresh, non-nested IndexMeasurement for each operation")
        self._started = time.perf_counter()
        self._token = _active.set(self)
        return self

    def __exit__(self, kind, exc, traceback):
        with self._lock:
            self._finished = time.perf_counter()
            self._unfinished = self._in_flight
            self._status = _outcome(exc)
            self._error_type = type(exc).__name__ if exc else None
            self._peak_rss = _peak_rss_bytes()
        _active.reset(self._token)

    @contextmanager
    def _span(self, kind, name, **details):
        start = time.perf_counter()
        with self._lock:
            self._in_flight += 1
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            end = time.perf_counter()
            with self._lock:
                self._in_flight -= 1
                if self._finished is None:
                    self._spans.append({"kind": kind, "name": name,
                        "start_seconds": start - self._started,
                        "end_seconds": end - self._started,
                        "elapsed_seconds": end - start, "status": _outcome(error),
                        "error_type": type(error).__name__ if error else None, **details})

    def report(self):
        if self._finished is None:
            raise RuntimeError("Read the report after the measurement context exits")
        with self._lock:
            spans, parameters = copy.deepcopy(self._spans), copy.deepcopy(self._parameters)
        phases = {}
        for name in dict.fromkeys([*PHASES, *[s["name"] for s in spans if s["kind"] == "phase"]]):
            group = [s for s in spans if s["kind"] == "phase" and s["name"] == name]
            phases[name] = {"spans": len(group), "elapsed_seconds": _union_seconds(group),
                "work_seconds": sum(s["elapsed_seconds"] for s in group),
                "failed_spans": sum(s["status"] != "succeeded" for s in group)}
        requests = [s for s in spans if s["kind"] == "model_request"]
        reasons = [s["retry_reason"] for s in requests if s["retry_reason"]]
        return {"schema_version": 1, "status": self._status, "error_type": self._error_type,
            "elapsed_seconds": self._finished - self._started,
            "parameters": parameters, "phases": phases,
            "model_requests": {
                "attempts": len(requests), "retry_attempts": len(reasons),
                "retries_by_reason": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
                "failed_attempts": sum(s["status"] != "succeeded" for s in requests),
                "opaque_dispatches": sum(s["opaque"] for s in requests),
                "dispatches_with_backend_retries": sum(s["backend_retries"] for s in requests),
                "latency_seconds": _distribution([s["elapsed_seconds"] for s in requests]),
                "scope": "PageIndex dispatches; provider/transport internal retries are not observed"},
            "process_peak_rss_bytes": self._peak_rss,
            "memory_scope": "current process lifetime high-water RSS; child parser processes excluded; null when unavailable",
            "unfinished_spans": self._unfinished, "complete": self._unfinished == 0,
            "spans": spans}


@contextmanager
def phase(name):
    run = _active.get()
    if run is None:
        yield
        return
    token = _phase.set(name)
    try:
        with run._span("phase", name):
            yield
    finally:
        _phase.reset(token)


def measured_phase(name):
    """Instrument sync/async functions without changing their result or errors."""
    def decorate(fn):
        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_call(*args, **kwargs):
                with phase(name):
                    return await fn(*args, **kwargs)
            return async_call
        @wraps(fn)
        def call(*args, **kwargs):
            with phase(name):
                return fn(*args, **kwargs)
        return call
    return decorate


def record_parameters(**parameters):
    run = _active.get()
    if run is not None:
        with run._lock:
            if run._finished is None:
                run._parameters.update(parameters)


@contextmanager
def retry_scope(reason):
    token = _retry_reason.set(reason or _retry_reason.get())
    try:
        yield
    finally:
        _retry_reason.reset(token)


@contextmanager
def model_request(model, retry=False, opaque=False, backend_retries=False):
    run = _active.get()
    if run is None:
        yield
        return
    with run._span("model_request", str(model), phase=_phase.get(),
                   retry_reason="transport" if retry else _retry_reason.get(),
                   opaque=bool(opaque), backend_retries=bool(backend_retries)):
        yield


def aggregate_reports(reports, elapsed_seconds):
    """Batch elapsed is measured by the caller, never summed from stages/jobs."""
    reports = list(reports)
    return {"runs": len(reports),
            "complete_runs": sum(r["complete"] for r in reports),
            "dispatch_counts_complete": all(r["complete"] for r in reports),
            **{status: sum(r["status"] == status for r in reports)
               for status in ("succeeded", "failed", "cancelled")},
            "elapsed_seconds": elapsed_seconds,
            "latency_seconds": _distribution([r["elapsed_seconds"] for r in reports]),
            "successful_latency_seconds": _distribution([r["elapsed_seconds"] for r in reports
                                                          if r["status"] == "succeeded"]),
            "model_request_attempts": sum(r["model_requests"]["attempts"] for r in reports),
            "model_retry_attempts": sum(r["model_requests"]["retry_attempts"] for r in reports),
            "opaque_dispatches": sum(r["model_requests"].get("opaque_dispatches", 0) for r in reports),
            "dispatches_with_backend_retries": sum(r["model_requests"].get("dispatches_with_backend_retries", 0) for r in reports),
            "max_worker_process_peak_rss_bytes": max(
                (r["process_peak_rss_bytes"] for r in reports if r["process_peak_rss_bytes"] is not None),
                default=None)}
