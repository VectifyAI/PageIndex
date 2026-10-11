from __future__ import annotations

from typing import Any


def _derive_measurement_status(measurements: list[Any]) -> str:
    if not measurements:
        return "unknown"
    statuses = []
    for m in measurements:
        if isinstance(m, dict) and m.get("status") in ("complete", "partial", "unknown"):
            statuses.append(m["status"])
            continue
        u = getattr(m, "usage", None) if not isinstance(m, dict) else m.get("usage")
        in_val = u.get("input_tokens") if isinstance(u, dict) else getattr(u, "input_tokens", None)
        if in_val is None:
            in_val = u.get("prompt_tokens") if isinstance(u, dict) else getattr(u, "prompt_tokens", None)
        out_val = u.get("output_tokens") if isinstance(u, dict) else getattr(u, "output_tokens", None)
        if out_val is None:
            out_val = u.get("completion_tokens") if isinstance(u, dict) else getattr(u, "completion_tokens", None)
        has_in = isinstance(in_val, int) and not isinstance(in_val, bool) and in_val >= 0
        has_out = isinstance(out_val, int) and not isinstance(out_val, bool) and out_val >= 0
        if has_in and has_out:
            statuses.append("complete")
        elif has_in or has_out:
            statuses.append("partial")
        else:
            statuses.append("unknown")
    if not statuses or all(s == "unknown" for s in statuses):
        return "unknown"
    if all(s == "complete" for s in statuses):
        return "complete"
    return "partial"


class PageIndexAPIError(Exception):
    """Raised on PageIndex API or backend provider failures.

    Attributes:
        status_code: HTTP status code when reported by the backend or endpoint;
            None does not imply a local or client-side failure.
        measurements: Defensive copies of individual model-call measurement dicts
            collected prior to failure, each retaining its call_id, turn_index,
            model, backend, usage, and per-call completeness status. Empty list
            when no turns completed before failure.
        measurement_status: Aggregate measurement completeness across completed
            turns (``"complete"``, ``"partial"``, or ``"unknown"``). Defaults to
            ``"unknown"`` when no measurements exist; otherwise derived from
            contributing calls. Execution outcome (``"failed"``) is orthogonal to
            measurement completeness: completed turns retain their accurate status.
        usage: Aggregate token usage dictionary across completed contributing turns,
            or None when no turns reported usage.
        execution_outcome: Always ``"failed"`` for exception instances.
    """

    def __init__(
        self,
        *args: object,
        status_code: int | None = None,
        measurements: list[dict[str, Any]] | None = None,
        measurement_status: str | None = None,
        usage: dict[str, Any] | None = None,
        execution_outcome: str = "failed",
    ) -> None:
        super().__init__(*args)
        self.status_code = status_code
        if measurements is not None:
            copied = []
            for m in measurements:
                if isinstance(m, dict):
                    item = dict(m)
                    if isinstance(item.get("usage"), dict):
                        u_dict = dict(item["usage"])
                        for k, v in list(u_dict.items()):
                            if isinstance(v, dict):
                                u_dict[k] = dict(v)
                        item["usage"] = u_dict
                    copied.append(item)
                else:
                    copied.append(m)
            self.measurements = copied
        else:
            self.measurements = []
        if measurement_status is not None:
            self.measurement_status = measurement_status
        elif not self.measurements:
            self.measurement_status = "unknown"
        else:
            self.measurement_status = _derive_measurement_status(self.measurements)
        if isinstance(usage, dict):
            u_dict = dict(usage)
            for k, v in list(u_dict.items()):
                if isinstance(v, dict):
                    u_dict[k] = dict(v)
            self.usage = u_dict
        else:
            self.usage = usage
        self.execution_outcome = execution_outcome


def _pageindex_cause(exc: BaseException | None) -> PageIndexAPIError | None:
    """The PageIndexAPIError behind a framework's wrapper exception, if any."""
    while exc is not None:
        if isinstance(exc, PageIndexAPIError):
            return exc
        exc = exc.__cause__
    return None
