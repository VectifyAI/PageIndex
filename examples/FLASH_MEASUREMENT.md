# Measuring Flash indexing

These hooks produce measurements, not a performance optimization. They help
separate the costs raised in #576. The general CLI profiling proposal in
#149 remains separate: this adds phase/dispatch events and a batch report,
without competing `--enable-profile` flags or copying its global profiler.
A future CLI integration can consume `IndexMeasurement.report()`.

## One operation

```python
from pageindex.indexing_metrics import IndexMeasurement

with IndexMeasurement(parameters={"experiment": "baseline"}) as run:
    result = client.submit_document("document.pdf")
report = run.report()
```

Use a fresh context per operation. On failure, keep the exception and read
`run.report()` after exiting the context; it records the outcome and error
class without exception text, prompts, responses or credentials. The result
and stored tree do not gain measurement fields. Without a context the hooks
are inactive. Independent operations may have independent contexts.

Phases are `parse_pages` (local PyPDF2 page cache), `parse_layout` (PDFium and
Flash extraction), `tree_optimization`, `summaries`, `document_description`,
and `storage` (including its lock wait). A direct `page_index_flash` call
does not store anything, so its storage span count is zero. Summary spans
cover each leaf/parent summary computation and its model gate wait; they
exclude waiting for children to become ready. A zero span count means the
phase was not observed, not that it ran instantly.

Each phase reports the union of its active intervals as `elapsed_seconds`
and the sum of its task durations as `work_seconds`. Both different phases
and same-phase tasks may overlap. **Do not add phase or document durations
to obtain total wall time.** The operation and batch each use an independent
monotonic timer. Raw offsets permit inspection of overlap.

`model_requests` counts PageIndex dispatch attempts and their latency
distribution (nearest-rank p50/p95/p99). Retries are labeled `transport`,
`summary_effort_fallback`, or `expand_repeat`. Provider-internal/transport
retries are not visible. `claude-code/` dispatches are marked opaque, and
overridden backend retry settings are flagged; do not interpret these
counts as exact provider HTTP traffic or billing. Request and phase
failures are counted even if an outer pipeline fallback recovers.

## Repeatable batch

From the repository root, with dependencies already installed:

```sh
# Default: real layout/merge work, no model calls or local SDK storage.
PYTHONPATH=. python examples/benchmark_flash.py tests/data/flash/zh_body_en_headings.pdf \
  --mode layout --repeats 3 --workers 1 --output layout-report.json

# Real parse/optimization/summary/storage code, synthetic model responses.
PYTHONPATH=. python examples/benchmark_flash.py tests/data/flash/zh_body_en_headings.pdf \
  --mode fixture --optimize full --summary-max-words 1 --repeats 3 --workers 1 \
  --output fixture-report.json
```

Every document/repeat gets a fresh process and temporary store. Reports
include input SHA-256, commit, Python/platform/dependency versions, effective
pipeline knobs, successes/failures/cancellations, document latency
distribution, requests/retries, and peak RSS. A separate worker wall time
includes process startup/teardown. The document timer includes pipeline
setup/imports and, for local runs, a separately labeled `result_read` phase.
Tree hashes allow a repeated fixture run to be checked for determinism.
Failures still produce reports and a nonzero batch exit status. A crashed
or timed-out worker is marked incomplete; aggregate dispatch counts then
become lower bounds. `--timeout` bounds the whole worker, not each request.

RSS is the current process's lifetime high-water mark including its native
PDFium memory, not a per-phase allocation count. Child parser processes are
excluded: this report does not measure their simultaneous total memory.
The effective parser worker count/mode is recorded to make that limit visible.
Fresh benchmark workers make the reported process scope repeatable.
The largest worker peak is **not** the batch's simultaneous peak memory.
RSS is null on platforms without the supported resource measurement.

Fixture mode strips keys from its worker environment, disables dotenv/model
map fetching, and replaces only the completion endpoints with fixed
responses. Its timings validate the measurement machinery and local work;
they **cannot establish real model latency, throughput, cache benefits or
speedups**. This work does not resolve all performance problems in #576.

`--mode configured --model ...` is an explicit opt-in for an already
configured real backend. It does not provision keys. No real-model run is
required by the tests, and no such run was used to validate this contribution.
Choose later optimization work only after collecting representative,
appropriately labeled measurements for the intended workload.
