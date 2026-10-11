"""Own-model chat usage: one record per backend model call, and how
complete the run's measurement is. Facts only — pricing, plans and
charging policy belong to the application.

Token inclusion, the same on every protocol: ``input_tokens`` counts the
whole prompt, cache reads and cache writes included; ``output_tokens``
counts the whole completion, reasoning included. The OpenAI lanes report
the backend's OpenAI-protocol counters, which already follow this
convention; the Messages lane's native counters (whose ``input_tokens``
excludes the cache) are folded into it, the native payload kept in
``provider_usage``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import threading
from typing import Any, Callable, Literal, Mapping, Optional

from .errors import PageIndexAPIError

logger = logging.getLogger(__name__)

Measurement = Literal["complete", "partial", "missing"]
Outcome = Literal["running", "completed", "failed", "cancelled"]
Completeness = Literal["complete", "partial", "unknown"]

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens",
                 "cache_write_tokens", "reasoning_tokens")


@dataclasses.dataclass(frozen=True)
class ModelCallUsage:
    """One backend model call of a chat run.

    ``measurement``: ``"complete"`` — the provider delivered this call's
    usage; ``"partial"`` — it delivered some, but the call was cut off
    before its final counts (a stream closed mid-turn), so the counters
    are lower bounds; ``"missing"`` — the call was made and no usage
    arrived (the provider omitted it, or the call failed or was
    cancelled in flight). A missing call's token fields are all None,
    never 0.

    Detail fields (cache, reasoning) are None when the provider did not
    report them — 0 means it reported zero. Where the provider payload
    is unavailable (``provider_usage`` None on the OpenAI lanes), the
    Agents SDK's normalized counters stand in, and those read 0 for an
    omitted detail. ``cache_write_tokens_by_ttl`` splits cache writes by
    cache lifetime (``{"5m": ..., "1h": ...}``) where the provider
    reports it — today Anthropic models.
    """

    index: int
    model: str
    backend: str
    measurement: Measurement
    response_id: Optional[str] = None
    request_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cache_write_tokens_by_ttl: Optional[Mapping[str, int]] = None
    reasoning_tokens: Optional[int] = None
    provider_usage: Optional[dict[str, Any]] = None


class ChatUsage:
    """The usage measurement of one ``chat()`` call, handed to its
    ``on_usage`` callback once the run has ended, and attached as
    ``.usage`` to the ``PageIndexAPIError`` a failed run raises.

    ``outcome``: ``"completed"`` (the call returned its result — whose
    own status may still say incomplete), ``"failed"``, or
    ``"cancelled"`` (a stream closed early, or the call interrupted);
    ``"running"`` only while the run is live. ``error`` is the
    exception a failed run raised.

    ``completeness``: ``"complete"`` — every model call the run made
    delivered its usage; ``"partial"`` — some usage is known, but at
    least one call's is missing or cut short, or the run ended in a way
    that may have left a call unobserved; ``"unknown"`` — no usage is
    known at all. Usage the provider never delivered is not recovered:
    a cancelled in-flight request may still be billed by the provider.
    """

    def __init__(self, protocol: str, model: str, backend: str, *,
                 observes_starts: bool,
                 on_finish: Optional[Callable[["ChatUsage"], Any]] = None):
        self.protocol = protocol
        self.outcome: Outcome = "running"
        self.error: Optional[BaseException] = None
        self._model = model
        self._backend = backend
        # True where the engine reports each call's start, so a call cut
        # off in flight is recorded as missing rather than unseen.
        self._observes_starts = observes_starts
        self._on_finish = on_finish
        self._calls: list[ModelCallUsage] = []
        self._open = 0  # calls started and not yet ended
        self._lock = threading.Lock()

    @property
    def calls(self) -> tuple[ModelCallUsage, ...]:
        """Every model call of the run, in order."""
        with self._lock:
            return tuple(self._calls)

    @property
    def completeness(self) -> Completeness:
        calls = self.calls
        ended = (self.outcome == "completed"
                 or (self._observes_starts and self.outcome != "running"))
        if calls and ended and all(call.measurement == "complete"
                                   for call in calls):
            return "complete"
        if any(call.measurement != "missing" for call in calls):
            return "partial"
        return "unknown"

    def totals(self) -> dict[str, Optional[int]]:
        """Each token field summed over the calls with known usage. A
        field is None when no call reported it or any of them left it
        unreported — never a silent undercount. Whether the sums cover
        the whole run is ``completeness``."""
        counted = [call for call in self.calls
                   if call.measurement != "missing"]
        totals: dict[str, Optional[int]] = {}
        for field in _TOKEN_FIELDS:
            values = [getattr(call, field) for call in counted]
            totals[field] = (sum(values) if values and None not in values
                             else None)
        return totals

    def __repr__(self) -> str:
        return (f"ChatUsage(protocol={self.protocol!r}, "
                f"outcome={self.outcome!r}, "
                f"completeness={self.completeness!r}, "
                f"calls={len(self.calls)})")

    # ── engine side ──

    def _start(self) -> None:
        with self._lock:
            if self.outcome == "running":
                self._open += 1

    def _record(self, build: Callable[[int], ModelCallUsage]) -> None:
        """Record an ended call; build(index) makes its record. A call
        that ends after the run was finalized is not recorded — the
        report is frozen, and that call already counts as missing."""
        with self._lock:
            if self.outcome != "running":
                return
            self._open = max(self._open - 1, 0)
            self._calls.append(build(len(self._calls)))

    def _missing(self, index: int) -> ModelCallUsage:
        return ModelCallUsage(index=index, model=self._model,
                              backend=self._backend, measurement="missing")

    def _attach(self, exc: BaseException) -> BaseException:
        if isinstance(exc, PageIndexAPIError):
            exc.usage = self
        return exc

    def _finish(self, outcome: Outcome,
                error: Optional[BaseException] = None) -> None:
        """Freeze the report and hand it to the callback, once. A
        callback's exception propagates from a completed run; from a
        failed or cancelled one it is logged, never masking the run's
        own exception."""
        with self._lock:
            if self.outcome != "running":
                return
            for _ in range(self._open):
                self._calls.append(self._missing(len(self._calls)))
            self._open = 0
            self.outcome = outcome
            self.error = error
        if self._on_finish is None:
            return
        if outcome == "completed":
            self._on_finish(self)
            return
        try:
            self._on_finish(self)
        except Exception:
            logger.exception("on_usage callback raised")


def _ending(report: ChatUsage, exc: BaseException
            ) -> "tuple[Outcome, Optional[BaseException]]":
    if isinstance(exc, (GeneratorExit, KeyboardInterrupt,
                        asyncio.CancelledError)):
        return "cancelled", None
    return "failed", report._attach(exc)


def _metered_call(report: ChatUsage, run: Callable[[], Any]) -> Any:
    """A non-streaming run under its report: finalized on every exit."""
    outcome: Outcome = "failed"
    error: Optional[BaseException] = None
    try:
        result = run()
        outcome = "completed"
        return result
    except BaseException as exc:
        outcome, error = _ending(report, exc)
        raise
    finally:
        report._finish(outcome, error)


async def _metered_agen(report: ChatUsage, agen):
    """An async stream under its report: finalized after the stream's
    own teardown, on exhaustion, failure and close alike."""
    outcome: Outcome = "failed"
    error: Optional[BaseException] = None
    try:
        async for item in agen:
            yield item
        outcome = "completed"
    except BaseException as exc:
        outcome, error = _ending(report, exc)
        raise
    finally:
        try:
            await agen.aclose()
        finally:
            report._finish(outcome, error)


def _metered_iter(report: ChatUsage, gen):
    """_metered_agen for a synchronous stream."""
    outcome: Outcome = "failed"
    error: Optional[BaseException] = None
    try:
        for item in gen:
            yield item
        outcome = "completed"
    except BaseException as exc:
        outcome, error = _ending(report, exc)
        raise
    finally:
        try:
            gen.close()
        finally:
            report._finish(outcome, error)


# ── provider payloads → records ──

# Where each detail counter sits in a provider payload: Responses,
# Chat Completions, and Anthropic-native spellings.
_RAW_PATHS = {
    "cache_read_tokens": (("input_tokens_details", "cached_tokens"),
                          ("prompt_tokens_details", "cached_tokens"),
                          ("cache_read_input_tokens",)),
    "cache_write_tokens": (("input_tokens_details", "cache_write_tokens"),
                           ("prompt_tokens_details", "cache_write_tokens"),
                           ("prompt_tokens_details", "cache_creation_tokens"),
                           ("cache_creation_input_tokens",)),
    "reasoning_tokens": (("output_tokens_details", "reasoning_tokens"),
                         ("completion_tokens_details", "reasoning_tokens")),
}

_TTL_KEY = re.compile(r"ephemeral_(\w+?)_input_tokens")


def _dig(payload: Any, path: tuple) -> Any:
    for key in path:
        if not isinstance(payload, Mapping):
            return None
        payload = payload.get(key)
    return payload


def _ttl_buckets(cache_creation: Any) -> Optional[dict[str, int]]:
    """{"5m": n, "1h": n} from an Anthropic ``cache_creation`` breakdown
    (a model or a dict); None when there is none."""
    if hasattr(cache_creation, "model_dump"):
        cache_creation = cache_creation.model_dump()
    if not isinstance(cache_creation, Mapping):
        return None
    buckets = {}
    for key, value in cache_creation.items():
        match = _TTL_KEY.fullmatch(str(key))
        if match and isinstance(value, int):
            buckets[match.group(1)] = value
    return buckets or None


def _agents_call(response: Any, index: int, model: str,
                 backend: str) -> ModelCallUsage:
    """An Agents SDK ModelResponse as a record. The SDK fills a zero
    Usage() when the provider sends none, so all-zero counters without a
    provider payload read as missing — a completed model call always
    consumes input tokens. With the payload (preserve_raw_usage), its
    presence is the evidence, and a reported zero stays a zero."""
    ids = {"response_id": getattr(response, "response_id", None),
           "request_id": getattr(response, "request_id", None)}
    usage = getattr(response, "usage", None)
    raw = getattr(response, "raw_usage", None)
    raw = raw if isinstance(raw, dict) else None
    if usage is None or (raw is None and not (
            usage.input_tokens or usage.output_tokens
            or usage.total_tokens)):
        return ModelCallUsage(index=index, model=model, backend=backend,
                              measurement="missing", **ids)
    inputs = getattr(usage, "input_tokens_details", None)
    outputs = getattr(usage, "output_tokens_details", None)
    details = {
        "cache_read_tokens": getattr(inputs, "cached_tokens", None),
        "cache_write_tokens": getattr(inputs, "cache_write_tokens", None),
        "reasoning_tokens": getattr(outputs, "reasoning_tokens", None),
    }
    if raw is not None:
        for field, paths in _RAW_PATHS.items():
            if not details[field] and all(_dig(raw, path) is None
                                          for path in paths):
                details[field] = None  # the provider did not report it
    ttl = None
    if raw is not None:
        for path in (("cache_creation",),
                     ("input_tokens_details", "cache_creation"),
                     ("prompt_tokens_details", "cache_creation")):
            ttl = _ttl_buckets(_dig(raw, path))
            if ttl:
                break
    return ModelCallUsage(
        index=index, model=model, backend=backend, measurement="complete",
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        cache_write_tokens_by_ttl=ttl, provider_usage=raw, **details, **ids)


def _anthropic_call(message: Any, index: int, model: str, backend: str,
                    finished: bool) -> ModelCallUsage:
    """An Anthropic message (or a stream's snapshot of one) as a record;
    ``finished``: its final counts arrived (a stream reached
    message_stop). Native input_tokens excludes the cache, so the
    normalized one adds cache reads and writes back."""
    usage = getattr(message, "usage", None)
    ids = {"response_id": getattr(message, "id", None),
           "request_id": getattr(message, "_request_id", None)}
    model = getattr(message, "model", None) or model
    if usage is None or getattr(usage, "input_tokens", None) is None:
        return ModelCallUsage(index=index, model=model, backend=backend,
                              measurement="missing", **ids)
    read = getattr(usage, "cache_read_input_tokens", None)
    write = getattr(usage, "cache_creation_input_tokens", None)
    return ModelCallUsage(
        index=index, model=model, backend=backend,
        measurement="complete" if finished else "partial",
        input_tokens=usage.input_tokens + (read or 0) + (write or 0),
        output_tokens=getattr(usage, "output_tokens", None),
        cache_read_tokens=read, cache_write_tokens=write,
        cache_write_tokens_by_ttl=_ttl_buckets(
            getattr(usage, "cache_creation", None)),
        # thinking is billed inside output_tokens; Messages has no split
        reasoning_tokens=None,
        provider_usage=usage.model_dump(mode="json", exclude_none=True),
        **ids)


def _agents_hooks(report: ChatUsage, model: str, backend: str):
    """RunHooks feeding the report: each call's start, and its
    ModelResponse once it returns (streamed and non-streamed alike)."""
    from agents import RunHooks

    class _UsageHooks(RunHooks):
        async def on_llm_start(self, context, agent, system_prompt,
                               input_items) -> None:
            report._start()

        async def on_llm_end(self, context, agent, response) -> None:
            report._record(lambda index: _agents_call(
                response, index, model, backend))

    return _UsageHooks()
