"""chat(stream=True)'s return type: one run, one view — text or events."""
from __future__ import annotations

from typing import Any, Iterator, Optional

from .errors import PageIndexAPIError


class ChatStream:
    """chat(stream=True)'s stream: iterate it for the answer text pieces
    (with show_process, the woven display); read ``.events`` instead for
    the typed process event dicts. One underlying run — consume exactly
    one view; call chat() again for the other.

    Streaming Lifecycle and Teardown:
        - ``execution_outcome``: ``"in_progress"`` while streaming or suspended;
          ``"completed"`` on normal exhaustion; ``"failed"`` on provider or client
          error; ``"cancelled"`` when explicitly stopped via ``close()``.
        - Calling ``close()`` on an already completed or failed stream preserves
          its terminal outcome without overwriting it.
        - Supported local cleanup is executed on ``close()``; in-flight remote
          provider requests cannot be aborted mid-turn. Remote execution may
          continue until the current turn finishes, but subsequent turns and tools
          will not run. Usage never delivered by the provider cannot be recovered
          or fabricated.
        - Collected metadata (``measurements``, ``measurement_status``, and
          ``usage``) reflects the latest state and remains accessible after close.
    """

    def __init__(self, text, events):
        self._text = text      # () -> Iterator[str]
        self._events = events  # () -> Iterator[dict], or the refusal text
        self._view: Optional[str] = None
        self._it: Any = None
        self._closed = False

    def _claim(self, view: str) -> None:
        if self._view is not None and self._view != view:
            raise PageIndexAPIError(
                f"This chat stream is being consumed as {self._view}; one "
                "run serves one view — call chat() again for the other.")
        self._view = view

    def __iter__(self) -> "ChatStream":
        return self

    def __next__(self) -> str:
        if self._closed:
            raise StopIteration
        self._claim("text")
        if self._it is None:
            self._it = self._text()
        return next(self._it)

    @property
    def events(self) -> Iterator[dict]:
        """The run as typed event dicts: {"type": "thinking"|"answer",
        "delta": ...}, {"type": "tool_call", "call_id", "name",
        "arguments"}, {"type": "tool_result", "call_id", "name",
        "output"} — the result as the framework recorded it (a string,
        or its structured text/image items), never clipped. Consuming —
        not merely reading the attribute — claims the view, so debugger
        panes and getattr probing stay side-effect free."""
        def consume():
            if self._closed:
                return
            if isinstance(self._events, str):
                raise PageIndexAPIError(self._events)
            self._claim("events")
            if self._it is None:
                self._it = self._events()
            # no `yield from`: a dropped handle must not close the run
            for ev in self._it:
                yield ev
        return consume()

    def close(self) -> None:
        """Stop the run: closes the open view, performs local cleanup, and
        transitions in-progress streams to 'cancelled'. An already completed
        or failed stream never overwrites its terminal outcome. Safe against
        exceptions raised by underlying view close."""
        self._closed = True
        close = getattr(self._it, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    @property
    def execution_outcome(self) -> str:
        """Stream execution outcome: 'in_progress', 'completed', 'failed', or 'cancelled'."""
        if self._it is not None and hasattr(self._it, "execution_outcome"):
            outcome = self._it.execution_outcome
            if outcome in ("completed", "failed"):
                return outcome
        if self._closed:
            return "cancelled"
        if self._it is not None and hasattr(self._it, "execution_outcome"):
            return self._it.execution_outcome
        return "in_progress"

    @property
    def measurements(self) -> list[dict[str, Any]]:
        """List of individual model-call measurement dicts collected during the run."""
        if self._it is not None and hasattr(self._it, "measurements"):
            return self._it.measurements
        return []

    @property
    def measurement_status(self) -> str:
        """Aggregate measurement completeness ('complete', 'partial', or 'unknown')."""
        if self._it is not None and hasattr(self._it, "measurement_status"):
            return self._it.measurement_status
        return "unknown"

    @property
    def usage(self) -> Optional[dict[str, Any]]:
        """Aggregated token usage dictionary across collected model calls, or None."""
        if self._it is not None and hasattr(self._it, "usage"):
            return self._it.usage
        return None
