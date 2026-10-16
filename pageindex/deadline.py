"""One chat budget across setup, model turns, tools and retries."""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from .errors import PageIndexAPIError, PageIndexTimeoutError

_CURRENT: ContextVar[Optional["Deadline"]] = ContextVar(
    "pageindex_deadline", default=None)


class Deadline:
    def __init__(self, seconds: float):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise PageIndexAPIError(
                "timeout must be a positive finite number.")
        self.seconds = seconds
        self.expires = time.monotonic() + seconds

    def error(self) -> PageIndexTimeoutError:
        return PageIndexTimeoutError(
            f"The chat run exceeded its timeout of {self.seconds:g}s.",
            timeout=self.seconds)

    def remaining(self) -> float:
        remaining = self.expires - time.monotonic()
        if remaining <= 0:
            raise self.error()
        return remaining

    async def run(self, coro):
        import asyncio

        with active(self):
            try:
                result = await asyncio.wait_for(
                    coro, max(0.0, self.expires - time.monotonic()))
                self.remaining()
                return result
            except asyncio.TimeoutError:
                if time.monotonic() >= self.expires:
                    raise self.error() from None
                raise


def current() -> Optional[Deadline]:
    return _CURRENT.get()


@contextmanager
def active(deadline: Optional[Deadline]):
    token = _CURRENT.set(deadline)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def check() -> None:
    deadline = current()
    if deadline is not None:
        deadline.remaining()


def request(method: str, url: str, *, timeout, **kwargs):
    """A cancellable cloud HTTP request inside the active chat budget."""
    import asyncio
    import httpx
    import requests
    from .utils import run_off_loop

    deadline = current()
    remaining = deadline.remaining()
    connect, read = (timeout if isinstance(timeout, tuple)
                     else (timeout, timeout))
    limits = httpx.Timeout(min(read, remaining),
                           connect=min(connect, remaining))

    async def send():
        async with httpx.AsyncClient(timeout=limits,
                                      follow_redirects=True) as client:
            return await client.request(method, url, **kwargs)

    try:
        return run_off_loop(asyncio.run, deadline.run(send()))
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        deadline.remaining()
        raise requests.ConnectionError(str(exc)) from exc
    except httpx.HTTPError as exc:
        deadline.remaining()
        raise requests.RequestException(str(exc)) from exc


def get(url: str, **kwargs):
    return request("GET", url, **kwargs)
