"""chat(timeout=...): one deadline for the whole run — setup, document
access, model requests, tool calls and retries.

The active deadline rides a context variable, so the work a run starts
sees it wherever it executes: the agent loop's tasks inherit it, and
``asyncio.to_thread`` copies it into tool threads. Each blocking layer
asks it for the budget left (``bound``) and refuses to start work once
it is spent (``check``).
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import math
import time
from typing import Any, AsyncIterator, Iterator, Optional

from .errors import PageIndexAPIError, PageIndexTimeoutError

_ACTIVE: contextvars.ContextVar[Optional["Deadline"]] = contextvars.ContextVar(
    "pageindex_deadline", default=None)


class Deadline:
    """A wall-clock budget, counted from its creation (monotonic)."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._expires = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._expires - time.monotonic())

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self._expires

    def error(self) -> PageIndexTimeoutError:
        return PageIndexTimeoutError(
            f"The chat run exceeded its timeout of {self.seconds:g}s and was "
            "stopped; no further model or tool calls were started.",
            timeout=self.seconds)

    def check(self) -> None:
        if self.expired:
            raise self.error()


def validate_timeout(timeout: Any) -> None:
    if timeout is None:
        return
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise PageIndexAPIError(
            "timeout must be a positive number of seconds — the budget for "
            f"the whole chat run — got {timeout!r}.")


def current() -> Optional[Deadline]:
    return _ACTIVE.get()


def for_run(timeout: Optional[float]) -> Optional[Deadline]:
    """The run's deadline: the one already active (chat() starts it before
    its own setup), else a new one from ``timeout``, else none."""
    active = _ACTIVE.get()
    if active is not None:
        return active
    validate_timeout(timeout)
    return Deadline(timeout) if timeout is not None else None


@contextlib.contextmanager
def active(deadline: Optional[Deadline]) -> Iterator[None]:
    """Make ``deadline`` the one the work in this block sees."""
    if deadline is None:
        yield
        return
    token = _ACTIVE.set(deadline)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def check() -> None:
    """Refuse to start work once the active deadline is spent."""
    deadline = _ACTIVE.get()
    if deadline is not None:
        deadline.check()


def bound(seconds: float) -> float:
    """``seconds`` capped at the active deadline's remaining budget."""
    deadline = _ACTIVE.get()
    return seconds if deadline is None else min(seconds, deadline.remaining())


def raise_if_expired(cause: Optional[BaseException] = None) -> None:
    """A failure that coincides with the deadline is the deadline's."""
    deadline = _ACTIVE.get()
    if deadline is not None and deadline.expired:
        raise deadline.error() from cause


async def run_within(deadline: Optional[Deadline], coro):
    """Await ``coro`` under ``deadline``: at expiry the run is cancelled
    (in-flight requests and retries included) and the deadline's error
    raised once that cancellation has unwound."""
    if deadline is None:
        return await coro
    with active(deadline):
        try:
            return await asyncio.wait_for(coro, deadline.remaining())
        except asyncio.TimeoutError:
            if deadline.expired:
                raise deadline.error() from None
            raise


async def iterate_within(deadline: Optional[Deadline],
                         source) -> AsyncIterator[Any]:
    """Each item of an async stream within ``deadline``: a stalled stream
    is cancelled at expiry instead of waiting out its own timeouts."""
    iterator = source.__aiter__()
    while True:
        try:
            if deadline is None:
                item = await iterator.__anext__()
            else:
                item = await asyncio.wait_for(iterator.__anext__(),
                                              deadline.remaining())
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            if deadline is not None and deadline.expired:
                raise deadline.error() from None
            raise
        yield item
